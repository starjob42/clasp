#!/usr/bin/env python3
"""Create-once certification for one-H100 Jiang-f638 Trainer checkpoints.

Every continuation allocation reopens this receipt before loading optimizer
state.  It binds the exact checkpoint tree (including adapter, Trainer,
optimizer, scheduler, and the single-process RNG), the frozen campaign, its
authoritative run contract, and the sealed Python/environment identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from freeze_base_sft_campaign import (  # type: ignore[import-not-found]
    PROTOCOL as CAMPAIGN_PROTOCOL,
)
from freeze_base_sft_campaign import (
    expected_run_contract,
    load_campaign,
    sha256_json,
)
from freeze_remote_environment import environment_identity  # type: ignore[import-not-found]

from clasp.entrypoints.train_sft_jiang_f638 import (
    ALLOWED_CHUNK_STOP_STEPS,
    REQUIRED_CHUNK_CHECKPOINT_FILES,
    TARGET_STEP,
)
from clasp.models.publication_fidelity import publication_fidelity_identity

SCHEMA_VERSION = 1
PROTOCOL = "clasp-publication-base-sft-single-h100-resume-state-v1"
STATUS = "sealed_exact_single_h100_trainer_resume_state"
RECEIPT_FILENAME = "base_sft_single_h100_resume_fidelity.json"
RUN_CONTRACT_FILENAME = "base_sft_run_contract.json"
CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)")
MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")
RESUME_STATE_FILES = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
FILE_MODE = 0o440
DIRECTORY_MODE = 0o550


def _canonical(path: Path, *, label: str, directory: bool = False) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise ValueError(f"{label} must be canonical: supplied={raw}, resolved={resolved}")
    if directory and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _canonical_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, raw


def _stable_file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint artifact is not a regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"checkpoint artifact must be single-link: {path.name}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise ValueError(f"checkpoint artifact changed while read: {path.name}")
        return {
            "path": path.name,
            "size_bytes": after.st_size,
            "sha256": digest.hexdigest(),
            "mode": f"{stat.S_IMODE(after.st_mode):04o}",
            "device": after.st_dev,
            "inode": after.st_ino,
            "nlink": after.st_nlink,
            "mtime_ns": after.st_mtime_ns,
        }
    finally:
        os.close(descriptor)


def _expected_artifact_names(checkpoint: Path) -> set[str]:
    present_models = {
        name
        for name in MODEL_FILES
        if (checkpoint / name).is_file() and not (checkpoint / name).is_symlink()
    }
    if len(present_models) != 1:
        raise ValueError("resume checkpoint must contain exactly one adapter model artifact")
    return set(REQUIRED_CHUNK_CHECKPOINT_FILES) | present_models


def _checkpoint_step(checkpoint: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"invalid resume checkpoint name: {checkpoint.name}")
    step = int(match.group(1))
    if step not in ALLOWED_CHUNK_STOP_STEPS:
        raise ValueError(f"checkpoint step {step} is not an approved chunk boundary")
    return step


def _campaign_and_run_identity(
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    *,
    require_accelerator: bool,
    verify_live_environment: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign_file = _canonical(campaign_path, label="base-SFT campaign")
    campaign, campaign_file_sha256 = load_campaign(campaign_file)
    if campaign.get("protocol") != CAMPAIGN_PROTOCOL:
        raise ValueError("single-H100 resume state requires the v9 campaign protocol")
    run_contract_file = _canonical(run_contract_path, label="base-SFT run contract")
    run_contract, run_contract_bytes = _canonical_json_file(
        run_contract_file, label="base-SFT run contract"
    )
    task = run_contract.get("task")
    if not isinstance(task, str):
        raise ValueError("base-SFT run contract has no task")
    expected = expected_run_contract(
        campaign=campaign,
        campaign_file_sha256=campaign_file_sha256,
        task=task,
    )
    if run_contract != expected:
        raise ValueError("base-SFT run contract differs from sealed campaign authority")
    authority = _canonical(
        Path(expected["authoritative_path"]), label="authoritative base-SFT run contract"
    )
    if authority.read_bytes() != run_contract_bytes or os.path.samefile(authority, run_contract_file):
        raise ValueError("run-contract copy differs from authority or aliases its inode")
    if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent != Path(expected["run_dir"]):
        raise ValueError("resume checkpoint is outside its frozen run cell")
    environment = campaign["remote_environment"]
    if verify_live_environment:
        observed = environment_identity(
            Path(environment["canonical_path"]),
            require_accelerator=require_accelerator,
        )
        if observed != environment:
            raise ValueError("live sealed environment differs from the campaign")
    campaign_identity = {
        "canonical_path": str(campaign_file),
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_file_sha256": campaign_file_sha256,
    }
    run_identity = {
        "canonical_path": str(run_contract_file),
        "run_contract_sha256": expected["run_contract_sha256"],
        "run_contract_file_sha256": hashlib.sha256(run_contract_bytes).hexdigest(),
        "authority_path": str(authority),
        "authority_file_sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
        "task": task,
        "run_dir": expected["run_dir"],
    }
    return campaign_identity, run_identity, environment


def _artifact_inventory(checkpoint: Path) -> tuple[list[dict[str, Any]], str]:
    expected = _expected_artifact_names(checkpoint)
    observed = {entry.name for entry in checkpoint.iterdir() if entry.name != RECEIPT_FILENAME}
    missing = sorted(expected - observed)
    extras = sorted(observed - expected)
    if missing or extras:
        raise ValueError(
            f"resume checkpoint inventory is not exact: missing={missing}, extras={extras}"
        )
    records = [_stable_file_record(checkpoint / name) for name in sorted(expected)]
    if any(record["mode"] != f"{FILE_MODE:04o}" for record in records):
        raise ValueError("sealed resume checkpoint artifacts must have exact mode 0440")
    return records, sha256_json(records)


def _core_identity(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    verify_live_environment: bool,
) -> dict[str, Any]:
    checkpoint_path = _canonical(checkpoint, label="single-H100 resume checkpoint", directory=True)
    step = _checkpoint_step(checkpoint_path)
    campaign, run_contract, environment = _campaign_and_run_identity(
        campaign_path,
        run_contract_path,
        checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )
    state, _ = _canonical_json_file(
        checkpoint_path / "trainer_state.json", label="Trainer state"
    )
    if state.get("global_step") != step or state.get("max_steps") != TARGET_STEP:
        raise ValueError("Trainer state disagrees with checkpoint boundary/fixed horizon")
    root_stat = checkpoint_path.stat(follow_symlinks=False)
    return {
        "campaign": campaign,
        "run_contract": run_contract,
        "environment": environment,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_device": root_stat.st_dev,
        "checkpoint_inode": root_stat.st_ino,
        "global_step": step,
        "max_steps": TARGET_STEP,
        "world_size": 1,
        "adapter_fidelity": publication_fidelity_identity(checkpoint_path),
    }


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite single-H100 resume receipt: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, FILE_MODE)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite single-H100 resume receipt: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def seal_resume_state(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    checkpoint_path = _canonical(checkpoint, label="single-H100 resume checkpoint", directory=True)
    receipt_path = checkpoint_path / RECEIPT_FILENAME
    if receipt_path.exists() or receipt_path.is_symlink():
        return resume_state_identity(
            campaign_path=campaign_path,
            run_contract_path=run_contract_path,
            checkpoint=checkpoint_path,
            require_accelerator=require_accelerator,
            verify_live_environment=verify_live_environment,
        )
    expected = _expected_artifact_names(checkpoint_path)
    observed = {entry.name for entry in checkpoint_path.iterdir()}
    if observed != expected:
        raise ValueError(
            "resume checkpoint cannot be sealed with missing/extra artifacts: "
            f"missing={sorted(expected - observed)}, extras={sorted(observed - expected)}"
        )
    for name in sorted(expected):
        path = checkpoint_path / name
        if path.is_symlink() or not path.is_file() or path.stat(follow_symlinks=False).st_nlink != 1:
            raise ValueError(f"unsafe checkpoint artifact: {name}")
        path.chmod(FILE_MODE)
    records, records_sha256 = _artifact_inventory(checkpoint_path)
    core = _core_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )
    by_name = {record["path"]: record for record in records}
    model_name = next(name for name in MODEL_FILES if name in by_name)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **core,
        "artifact_inventory": records,
        "artifact_inventory_sha256": records_sha256,
        "resume_state": {
            name: by_name[name] for name in (*RESUME_STATE_FILES, model_name)
        },
    }
    payload["receipt_sha256"] = sha256_json(payload)
    _write_create_once(receipt_path, payload)
    checkpoint_path.chmod(DIRECTORY_MODE)
    return resume_state_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )


def resume_state_identity(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    checkpoint_path = _canonical(checkpoint, label="single-H100 resume checkpoint", directory=True)
    if stat.S_IMODE(checkpoint_path.stat(follow_symlinks=False).st_mode) != DIRECTORY_MODE:
        raise ValueError("single-H100 resume checkpoint must have exact mode 0550")
    receipt_path = checkpoint_path / RECEIPT_FILENAME
    receipt, raw = _canonical_json_file(receipt_path, label="single-H100 resume receipt")
    if raw != json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n":
        raise ValueError("single-H100 resume receipt is not canonically serialized")
    receipt_stat = receipt_path.stat(follow_symlinks=False)
    if stat.S_IMODE(receipt_stat.st_mode) != FILE_MODE or receipt_stat.st_nlink != 1:
        raise ValueError("single-H100 resume receipt must be single-link mode 0440")
    required_keys = {
        "schema_version", "protocol", "status", "created_at_utc", "campaign",
        "run_contract", "environment", "checkpoint_path", "checkpoint_device",
        "checkpoint_inode", "global_step", "max_steps", "world_size",
        "adapter_fidelity", "artifact_inventory", "artifact_inventory_sha256",
        "resume_state", "receipt_sha256",
    }
    if set(receipt) != required_keys:
        raise ValueError("single-H100 resume receipt has an invalid schema")
    unsigned = dict(receipt)
    stored_hash = unsigned.pop("receipt_sha256", None)
    if (
        stored_hash != sha256_json(unsigned)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != STATUS
        or receipt.get("world_size") != 1
        or receipt.get("max_steps") != TARGET_STEP
        or receipt.get("global_step") not in ALLOWED_CHUNK_STOP_STEPS
    ):
        raise ValueError("single-H100 resume receipt disagrees with the v9 contract")
    core = _core_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )
    for name, value in core.items():
        if receipt.get(name) != value:
            raise ValueError(f"live resume {name} differs from its receipt")
    records, records_sha256 = _artifact_inventory(checkpoint_path)
    if (
        receipt.get("artifact_inventory") != records
        or receipt.get("artifact_inventory_sha256") != records_sha256
    ):
        raise ValueError("live checkpoint artifacts differ from the resume receipt")
    by_name = {record["path"]: record for record in records}
    model_name = next(name for name in MODEL_FILES if name in by_name)
    expected_state = {name: by_name[name] for name in (*RESUME_STATE_FILES, model_name)}
    if receipt.get("resume_state") != expected_state:
        raise ValueError("resume-state projection does not bind all required state")
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "canonical_path": str(receipt_path),
        "checkpoint_path": str(checkpoint_path),
        "global_step": receipt["global_step"],
        "world_size": 1,
        "receipt_sha256": stored_hash,
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_inventory_sha256": records_sha256,
        "campaign_sha256": receipt["campaign"]["campaign_sha256"],
        "run_contract_sha256": receipt["run_contract"]["run_contract_sha256"],
        "environment_identity_sha256": receipt["environment"]["identity_sha256"],
        "adapter_fidelity_identity_sha256": receipt["adapter_fidelity"]["identity_sha256"],
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--require-live-accelerator", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = seal_resume_state if args.seal else resume_state_identity
    try:
        identity = operation(
            campaign_path=args.campaign_contract,
            run_contract_path=args.run_contract,
            checkpoint=args.checkpoint,
            require_accelerator=args.require_live_accelerator,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[BaseSFTSingleH100ResumeError] {exc}", file=os.sys.stderr)
        return 2
    print(
        f"VALID:{identity['global_step']}:{identity['checkpoint_path']}:"
        f"{identity['identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
