"""Source-bound, create-only endpoint flag receipts and distributed failures."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from clasp.baselines.co_deterministic_dev import PROFILE_ID
from clasp.models import co_determinism_receipt as evidence


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    output, source = tmp_path / "run", tmp_path / "source"
    output.mkdir(); source.mkdir()
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("synthetic source manifest\n")
    identity = evidence._identity(manifest)
    contract = {"method": "classic_grpo", "source": {"root": str(source), "manifest_sha256": identity["sha256"]},
                "diagnostic": True, "fixed_horizon": {"max_steps": 400, "world_size": 4}}
    path = output / "local_development_run_contract.json"
    path.write_text(json.dumps(contract))
    path.chmod(0o440)
    environment = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "43", "CUDA_LAUNCH_BLOCKING": None}
    initialization = {"profile_id": PROFILE_ID, "applied_before_cuda_initialization": True,
                      "torch_version": str(torch.__version__), "environment": environment}
    runtime = SimpleNamespace(method="classic_grpo", initialization_receipt=initialization,
                              base=SimpleNamespace(source_binding={"manifest": identity}))
    args = SimpleNamespace(output_dir=str(output))
    monkeypatch.setenv("CLASP_TRAINING_RUN_CONTRACT_PATH", str(path))
    monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", "10")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setattr(evidence, "validate_strict_determinism", lambda: {"deterministic_algorithms": True})
    monkeypatch.setattr(evidence, "validate_environment", lambda: environment)
    return args, runtime, path, identity


def test_record_binds_pre_cuda_actual_version_source_contract_method_step_and_rank(bundle):
    args, runtime, contract, identity = bundle
    path, record = evidence._postrun_record(args, runtime, 0)
    assert path == contract.parent / "postrun_determinism_step10_rank0.json"
    assert record["source"]["manifest"] == identity
    assert record["original_contract"] == evidence._identity(contract)
    assert record["initialization"] == runtime.initialization_receipt
    assert record["torch_version"] == str(torch.__version__)
    assert record["scope"] == "endpoint_execution_flags_not_per_kernel_or_continuation_proof"
    assert record["requires_all_rank_receipts_and_successful_job"] is True
    assert record["rank"] == 0 and record["world_size"] == 4 and record["step"] == 10
    unsigned = {k: v for k, v in record.items() if k != "receipt_sha256"}
    assert record["receipt_sha256"] == hashlib.sha256(evidence._canonical(unsigned)).hexdigest()


@pytest.mark.parametrize("fault", ["source", "method", "output", "writable_contract", "rank", "world", "step",
                                  "job", "pre_cuda", "initial_version", "initial_environment"])
def test_changed_provenance_never_produces_a_postrun_record(bundle, monkeypatch, fault):
    args, runtime, contract, identity = bundle
    if fault == "source":
        Path(identity["path"]).write_text("changed source manifest\n")
    elif fault == "method":
        runtime.method = "foarl_teacher"
    elif fault == "output":
        args.output_dir += "/other"
    elif fault == "writable_contract":
        contract.chmod(0o640)
    elif fault == "rank":
        monkeypatch.setenv("RANK", "3")
    elif fault == "world":
        monkeypatch.setenv("WORLD_SIZE", "2")
    elif fault == "step":
        monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", "9")
    elif fault == "job":
        monkeypatch.setenv("SLURM_JOB_ID", "unknown")
    elif fault == "pre_cuda":
        runtime.initialization_receipt["applied_before_cuda_initialization"] = False
    elif fault == "initial_version":
        runtime.initialization_receipt["torch_version"] = "unknown"
    else:
        runtime.initialization_receipt["environment"] = {}
    with pytest.raises(ValueError):
        evidence._postrun_record(args, runtime, 0)
    assert list(contract.parent.glob("postrun*.json")) == []


def _worker(rank, rendezvous, output, mode):
    distributed.init_process_group("gloo", rank=rank, world_size=4,
                                   init_method="file://" + rendezvous, timeout=dt.timedelta(seconds=30))
    try:
        root = Path(output)
        manifest = evidence._identity(root / "source/SOURCE_MANIFEST.sha256")
        os.environ.update(CLASP_TRAINING_RUN_CONTRACT_PATH=str(root / "run/contract.json"),
                          GRPO_STOP_AT_GLOBAL_STEP="10", WORLD_SIZE="4", RANK=str(rank), SLURM_JOB_ID="123456")
        environment = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "43", "CUDA_LAUNCH_BLOCKING": None}
        initialization = {"profile_id": PROFILE_ID, "applied_before_cuda_initialization": True,
                          "torch_version": str(torch.__version__), "environment": environment}
        runtime = SimpleNamespace(method="classic_grpo", initialization_receipt=initialization,
                                  base=SimpleNamespace(source_binding={"manifest": manifest}))
        evidence.validate_strict_determinism = lambda: {"deterministic_algorithms": True}
        evidence.validate_environment = lambda: environment
        if mode == "rank2_admission" and rank == 2:
            os.environ["RANK"] = "0"
        if mode == "rank2_identity" and rank == 2:
            os.environ["SLURM_JOB_ID"] = "999999"
        before = torch.random.get_rng_state().clone()
        try:
            evidence.write_postrun_receipt(SimpleNamespace(output_dir=str(root / "run")), runtime)
            status = "success"
        except RuntimeError:
            status = "all_ranks_failed"
        report = {"status": status, "rng_unchanged": torch.equal(before, torch.random.get_rng_state())}
        (root / f"rank{rank}.json").write_text(json.dumps(report))
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["success", "rank2_admission", "rank2_identity", "existing_rank2_receipt"])
def test_real_four_rank_receipt_publication_is_create_only_and_collectively_fails(tmp_path, mode):
    root = tmp_path / mode
    (root / "source").mkdir(parents=True); (root / "run").mkdir()
    manifest = root / "source/SOURCE_MANIFEST.sha256"
    manifest.write_text("synthetic manifest\n")
    contract = {"method": "classic_grpo", "source": {"root": str(manifest.parent),
                "manifest_sha256": evidence._identity(manifest)["sha256"]}, "diagnostic": True,
                "fixed_horizon": {"max_steps": 400, "world_size": 4}}
    (root / "run/contract.json").write_text(json.dumps(contract))
    (root / "run/contract.json").chmod(0o440)
    old = root / "run/postrun_determinism_step10_rank2.json"
    if mode == "existing_rank2_receipt":
        old.write_text("retained prior evidence\n")
    multiprocessing.start_processes(_worker, args=(str(tmp_path / (mode + ".rendezvous")), str(root), mode),
                                    nprocs=4, join=True, start_method="fork")
    expected = {"status": "success" if mode == "success" else "all_ranks_failed", "rng_unchanged": True}
    assert [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(4)] == [expected] * 4
    if mode == "existing_rank2_receipt":
        assert old.read_text() == "retained prior evidence\n"
    if mode == "success":
        for rank in range(4):
            path = root / f"run/postrun_determinism_step10_rank{rank}.json"
            record = json.loads(path.read_text())
            assert record["rank"] == rank
            assert path.stat().st_mode & 0o222 == 0
            assert record["status"] == "passed_postrun_strict_flags"
