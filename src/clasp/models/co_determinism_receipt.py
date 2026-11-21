"""Create value-free, per-rank strict-execution evidence after successful training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from clasp.baselines.co_deterministic_dev import (
    PROFILE_ID,
    validate_environment,
    validate_strict_determinism,
)

PROTOCOL = "co-strict-determinism-postrun-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(path):
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or not path.is_file()):
        raise ValueError("Determinism evidence input must be an unlinked absolute regular file")
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _postrun_record(args, runtime, rank):
    flags = validate_strict_determinism()
    contract_path = Path(os.environ["CLASP_TRAINING_RUN_CONTRACT_PATH"])
    original_contract = _identity(contract_path)
    contract = json.loads(contract_path.read_text())
    if (contract_path.stat().st_mode & 0o222 or Path(args.output_dir) != contract_path.parent
            or contract.get("method") != runtime.method):
        raise ValueError("Postrun evidence must retain the original method and read-only run contract")
    source = contract["source"]
    manifest = _identity(Path(source["root"]) / "SOURCE_MANIFEST.sha256")
    if (manifest["sha256"] != source["manifest_sha256"]
            or runtime.base.source_binding["manifest"] != manifest):
        raise ValueError("Postrun source identity differs from the verified initialization binding")
    step = int(os.environ["GRPO_STOP_AT_GLOBAL_STEP"])
    boundaries = (5, 10) if contract["diagnostic"] else (100, 200, 300, 400)
    if step not in boundaries or contract["fixed_horizon"] != {"max_steps": 400, "world_size": 4}:
        raise ValueError("Postrun step or original fixed horizon differs")
    if os.environ.get("RANK") != str(rank) or os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Postrun rank environment differs from the actual process group")
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdigit():
        raise ValueError("Postrun evidence requires the actual scheduler job ID")
    initialization = dict(runtime.initialization_receipt)
    if (initialization.get("profile_id") != PROFILE_ID
            or initialization.get("applied_before_cuda_initialization") is not True
            or initialization.get("torch_version") != str(torch.__version__)
            or initialization.get("environment") != validate_environment()):
        raise ValueError("Postrun execution no longer matches the pre-CUDA initialization proof")
    if _identity(contract_path) != original_contract:
        raise ValueError("Original contract changed during postrun admission")
    record = {
        "protocol": PROTOCOL, "status": "passed_postrun_strict_flags", "profile_id": PROFILE_ID,
        "scope": "endpoint_execution_flags_not_per_kernel_or_continuation_proof",
        "method": runtime.method, "step": step, "rank": rank, "world_size": 4, "job_id": job,
        "source": {"root": source["root"], "manifest": manifest}, "original_contract": original_contract,
        "initialization": initialization, "training_flags": flags, "environment": validate_environment(),
        "torch_version": str(torch.__version__),
        "requires_all_rank_receipts_and_successful_job": True,
    }
    record["receipt_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return contract_path.parent / f"postrun_determinism_step{step}_rank{rank}.json", record


def write_postrun_receipt(args, runtime):
    """Publish all four local proofs or fail every rank; never overwrite a proof."""
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise RuntimeError("Postrun deterministic evidence requires four distributed ranks")
    rank = dist.get_rank()
    dist.barrier()
    cause = None
    record = None
    try:
        path, record = _postrun_record(args, runtime, rank)
        with path.open("xb") as stream:
            stream.write(_canonical(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o440)
        local = {"rank": rank, "record": record, "file": _identity(path), "error": None}
    except Exception as exc:
        cause = exc
        local = {"rank": rank, "error": type(exc).__name__}
    gathered = [None] * 4
    dist.all_gather_object(gathered, local)
    outcome = [None]
    if rank == 0:
        try:
            if any(item.get("error") is not None for item in gathered):
                raise ValueError("One or more ranks could not publish their postrun receipt")
            comparable = [{k: v for k, v in item["record"].items() if k not in ("rank", "receipt_sha256")}
                          for item in gathered]
            if (any(value != comparable[0] for value in comparable[1:])
                    or [item["rank"] for item in gathered] != list(range(4))):
                raise ValueError("Ranks disagree on their postrun execution identity or flags")
            outcome[0] = {"status": "all_postrun_receipts_published"}
        except Exception as exc:
            cause = exc
            outcome[0] = {"status": "failed_postrun_receipt_publication", "exception_class": type(exc).__name__}
    dist.broadcast_object_list(outcome, src=0)
    dist.barrier()
    if outcome[0].get("status") != "all_postrun_receipts_published":
        raise RuntimeError("Distributed postrun deterministic evidence failed: "
                           + str(outcome[0].get("exception_class"))) from cause
    return record
