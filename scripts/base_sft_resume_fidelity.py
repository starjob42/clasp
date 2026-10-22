#!/usr/bin/env python3
"""Seal and live-verify an exact base-SFT Trainer resume checkpoint.

The receipt is create-once and lives inside the checkpoint it authorizes.  It
binds the complete allowlisted checkpoint tree (excluding itself), the exact
Trainer/optimizer/scheduler/rank-RNG bytes, publication adapter fidelity, and
the frozen campaign/run/environment identities.  Publication workers verify it
again inside the same Slurm step immediately before ``torch.distributed.run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from freeze_base_sft_campaign import load_campaign, sha256_json  # type: ignore[import-not-found]
from freeze_remote_environment import environment_identity  # type: ignore[import-not-found]
from transformers import TrainerCallback  # type: ignore[import-not-found]

from clasp.models.checkpoint_fidelity import audit_trainable_state_closure
from clasp.models.publication_fidelity import (
    IDENTITY_PROTOCOL,
    IDENTITY_SCHEMA_VERSION,
    IDENTITY_STATUS,
    publication_fidelity_identity,
    validate_publication_fidelity_identity,
)

SCHEMA_VERSION = 1
PROTOCOL = "clasp-publication-base-sft-resume-state-v1"
STATUS = "sealed_exact_trainer_resume_state"
CLAIM_SCOPE = "publication_base_specialist_sft"
RECEIPT_FILENAME = "base_sft_resume_fidelity.json"
RUN_CONTRACT_FILENAME = "base_sft_run_contract.json"
CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
MODEL_FILENAME = "adapter_model.safetensors"
REQUIRED_CHECKPOINT_FILES = {
    "README.md",
    "adapter_config.json",
    MODEL_FILENAME,
    "added_tokens.json",
    "checkpoint_fidelity.json",
    "merges.txt",
    "optimizer.pt",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "scheduler.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "vocab.json",
}
ENABLE_ENV = "CLASP_BASE_SFT_RESUME_FIDELITY"
CAMPAIGN_ENV = "CLASP_BASE_SFT_CAMPAIGN_CONTRACT"
RUN_CONTRACT_ENV = "CLASP_BASE_SFT_RUN_CONTRACT"
EXPECTED_MAX_STEPS_ENV = "CLASP_SFT_EXPECTED_MAX_STEPS"
EXPECTED_OPTIMIZER_CLASS = "bitsandbytes.optim.adamw.AdamW"
EXPECTED_SCHEDULER_CLASS = "torch.optim.lr_scheduler.LambdaLR"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {raw}")
    resolved = raw.resolve(strict=True)
    if raw != resolved:
        raise ValueError(f"{label} must be canonical: supplied={raw}, resolved={resolved}")
    return resolved


def _sha256_file_fd(parent_fd: int, name: str, *, label: str) -> tuple[str, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name,
        os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label} changed while it was hashed")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _file_record(parent_fd: int, name: str) -> dict[str, Any]:
    digest, value = _sha256_file_fd(parent_fd, name, label=f"resume file {name}")
    mode = stat.S_IMODE(value.st_mode)
    if mode != 0o440:
        raise ValueError(f"resume file must have exact mode 0440: {name} has {mode:04o}")
    return {
        "path": name,
        "size_bytes": value.st_size,
        "sha256": digest,
        "mode": f"{mode:04o}",
        "device": value.st_dev,
        "inode": value.st_ino,
        "nlink": value.st_nlink,
        "mtime_ns": value.st_mtime_ns,
    }


def _json_object_fd(parent_fd: int, name: str, *, label: str) -> dict[str, Any]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name,
        os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label} changed while it was read")
        try:
            value = json.loads(b"".join(blocks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse {label}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} is not a JSON object")
        return value
    finally:
        os.close(descriptor)


def _regular_file_bytes_fd(
    parent_fd: int,
    name: str,
    *,
    label: str,
    required_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one stable, single-link regular file below a retained directory fd."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, os.O_RDONLY | nofollow | nonblock, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        mode = stat.S_IMODE(before.st_mode)
        if required_mode is not None and mode != required_mode:
            raise ValueError(f"{label} must have exact mode {required_mode:04o}, got {mode:04o}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(blocks), after
    finally:
        os.close(descriptor)


def _decode_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _bound_adapter_fidelity_identity(
    checkpoint_fd: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the sealed adapter identity without reopening a pathname."""

    raw, _ = _regular_file_bytes_fd(
        checkpoint_fd,
        "checkpoint_fidelity.json",
        label="checkpoint-fidelity receipt",
        required_mode=0o440,
    )
    receipt = _decode_json_bytes(raw, label="checkpoint-fidelity receipt")
    canonical = (
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    )
    if raw != canonical:
        raise ValueError("checkpoint-fidelity receipt is not canonically serialized")
    unsigned = dict(receipt)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if not isinstance(receipt_sha256, str) or receipt_sha256 != sha256_json(unsigned):
        raise ValueError("checkpoint-fidelity receipt self-hash mismatch")
    artifacts = receipt.get("artifacts")
    closure = receipt.get("trainable_closure")
    publication = receipt.get("publication_invariant")
    adapter_config = artifacts.get("adapter_config") if isinstance(artifacts, dict) else None
    adapter_weights = artifacts.get("adapter_weights") if isinstance(artifacts, dict) else None
    if (
        not isinstance(adapter_config, dict)
        or not isinstance(adapter_weights, dict)
        or not isinstance(closure, dict)
        or not isinstance(publication, dict)
    ):
        raise ValueError("checkpoint-fidelity receipt lacks exact adapter evidence")
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "protocol": IDENTITY_PROTOCOL,
        "status": IDENTITY_STATUS,
        "receipt_sha256": receipt_sha256,
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "adapter_config_sha256": adapter_config.get("sha256"),
        "adapter_weights_sha256": adapter_weights.get("sha256"),
        "trainable_tensor_inventory_sha256": closure.get("tensor_inventory_sha256"),
        "publication_invariant_sha256": sha256_json(publication),
    }
    identity["identity_sha256"] = sha256_json(identity)
    validated = validate_publication_fidelity_identity(identity)
    return validated, receipt


def _torch_load_fd(parent_fd: int, name: str, *, weights_only: bool) -> Any:
    import torch

    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            return torch.load(handle, map_location="cpu", weights_only=weights_only)
    finally:
        os.close(descriptor)


def _checkpoint_inventory(checkpoint: Path | int) -> dict[str, Any]:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    root_fd = (
        os.dup(checkpoint)
        if isinstance(checkpoint, int)
        else os.open(checkpoint, os.O_RDONLY | directory_flag | nofollow)
    )
    try:
        root_stat = os.fstat(root_fd)
        root_mode = stat.S_IMODE(root_stat.st_mode)
        if root_mode != 0o550:
            raise ValueError(
                f"resume checkpoint root must have exact mode 0550, got {root_mode:04o}"
            )
        names = sorted(os.listdir(root_fd))
        if RECEIPT_FILENAME not in names:
            raise ValueError(f"resume checkpoint lacks {RECEIPT_FILENAME}")
        artifact_names = [name for name in names if name != RECEIPT_FILENAME]
        extras = sorted(set(artifact_names) - REQUIRED_CHECKPOINT_FILES)
        missing = sorted(REQUIRED_CHECKPOINT_FILES - set(artifact_names))
        if extras or missing:
            raise ValueError(
                f"resume checkpoint inventory is not exact: missing={missing}, extras={extras}"
            )
        records: list[dict[str, Any]] = []
        for name in artifact_names:
            entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError(f"resume checkpoint contains a non-regular entry: {name}")
            records.append(_file_record(root_fd, name))
        inventory: dict[str, Any] = {
            "root": {
                "mode": f"{root_mode:04o}",
                "device": root_stat.st_dev,
                "inode": root_stat.st_ino,
                "uid": root_stat.st_uid,
                "gid": root_stat.st_gid,
                "nlink": root_stat.st_nlink,
            },
            "file_count": len(records),
            "size_bytes": sum(record["size_bytes"] for record in records),
            "files": records,
        }
        inventory["tree_sha256"] = sha256_json(inventory)
        return inventory
    finally:
        os.close(root_fd)


def _validate_checkpoint_inventory(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "root",
        "file_count",
        "size_bytes",
        "files",
        "tree_sha256",
    }:
        raise ValueError("resume receipt has an invalid checkpoint-tree schema")
    unsigned = dict(value)
    stored_hash = unsigned.pop("tree_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("resume checkpoint-tree self-hash mismatch")
    root = value.get("root")
    root_keys = {"mode", "device", "inode", "uid", "gid", "nlink"}
    if (
        not isinstance(root, dict)
        or set(root) != root_keys
        or re.fullmatch(r"[0-7]{4}", str(root.get("mode", ""))) is None
        or root.get("mode") != "0550"
        or any(
            type(root.get(field)) is not int or root[field] < 0 for field in root_keys - {"mode"}
        )
    ):
        raise ValueError("resume receipt has invalid checkpoint-root evidence")
    files = value.get("files")
    file_keys = {
        "path",
        "size_bytes",
        "sha256",
        "mode",
        "device",
        "inode",
        "nlink",
        "mtime_ns",
    }
    if (
        type(value.get("file_count")) is not int
        or type(value.get("size_bytes")) is not int
        or not isinstance(files, list)
        or value["file_count"] != len(files)
        or value["size_bytes"]
        != sum(record.get("size_bytes", -1) if isinstance(record, dict) else -1 for record in files)
    ):
        raise ValueError("resume receipt has invalid checkpoint-tree counts")
    paths: list[str] = []
    for record in files:
        if (
            not isinstance(record, dict)
            or set(record) != file_keys
            or not isinstance(record.get("path"), str)
            or "/" in record["path"]
            or record["path"] in {"", ".", "..", RECEIPT_FILENAME}
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
            or re.fullmatch(r"[0-7]{4}", str(record.get("mode", ""))) is None
            or record.get("mode") != "0440"
            or any(
                type(record.get(field)) is not int or record[field] < 0
                for field in ("device", "inode", "nlink", "mtime_ns")
            )
            or record["nlink"] != 1
        ):
            raise ValueError("resume receipt has an invalid checkpoint file record")
        paths.append(record["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("resume receipt checkpoint file paths are not unique and sorted")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_contract_identity(
    path: Path, campaign: Mapping[str, Any], campaign_file_sha: str
) -> dict[str, Any]:
    contract = _canonical(path, label="base-SFT run contract")
    if contract.name != RUN_CONTRACT_FILENAME:
        raise ValueError("base-SFT run contract must be the read-only canonical contract")
    contract_parent_fd = os.open(
        contract.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        contract_bytes, contract_stat = _regular_file_bytes_fd(
            contract_parent_fd,
            contract.name,
            label="base-SFT run contract",
            required_mode=0o440,
        )
    finally:
        os.close(contract_parent_fd)
    payload = _decode_json_bytes(contract_bytes, label="base-SFT run contract")
    stored_hash = payload.get("run_contract_sha256")
    unsigned = dict(payload)
    unsigned.pop("run_contract_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("base-SFT run-contract self-hash mismatch")
    if (
        payload.get("campaign_id") != campaign["campaign_id"]
        or payload.get("campaign_sha256") != campaign["campaign_sha256"]
        or payload.get("campaign_file_sha256") != campaign_file_sha
        or payload.get("remote_environment_identity_sha256")
        != campaign["remote_environment"]["identity_sha256"]
    ):
        raise ValueError("base-SFT run contract differs from the campaign/environment")
    authority = _canonical(
        Path(str(payload.get("authoritative_path", ""))),
        label="authoritative base-SFT run contract",
    )
    expected_authority_name = campaign["run_contract_authority"]["contracts"].get(
        payload.get("task")
    )
    authority_parent_fd = os.open(
        authority.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        authority_bytes, authority_stat = _regular_file_bytes_fd(
            authority_parent_fd,
            authority.name,
            label="authoritative base-SFT run contract",
            required_mode=0o440,
        )
        authority_parent_stat = os.fstat(authority_parent_fd)
    finally:
        os.close(authority_parent_fd)
    if (
        authority.parent != Path(campaign["run_contract_authority"]["canonical_path"])
        or authority.name != expected_authority_name
        or stat.S_IMODE(authority_parent_stat.st_mode) != 0o550
        or (authority_stat.st_dev, authority_stat.st_ino)
        == (contract_stat.st_dev, contract_stat.st_ino)
        or authority_bytes != contract_bytes
    ):
        raise ValueError("mutable base-SFT run contract differs from sealed authority")
    return {
        "canonical_path": str(contract),
        "run_contract_sha256": stored_hash,
        "run_contract_file_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "run_contract_device": contract_stat.st_dev,
        "run_contract_inode": contract_stat.st_ino,
        "authority_file_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "authority_device": authority_stat.st_dev,
        "authority_inode": authority_stat.st_ino,
        "task": payload.get("task"),
        "run_dir": payload.get("run_dir"),
    }


def _core_identity(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    verify_live_environment: bool = True,
    checkpoint_fd: int | None = None,
) -> dict[str, Any]:
    campaign, campaign_file_sha = load_campaign(campaign_path)
    environment = campaign["remote_environment"]
    if verify_live_environment:
        observed_environment = environment_identity(
            Path(environment["canonical_path"]),
            require_accelerator=require_accelerator,
        )
        if observed_environment != environment:
            raise ValueError("live remote environment differs from the base-SFT campaign")
    run_contract = _run_contract_identity(run_contract_path, campaign, campaign_file_sha)
    checkpoint_path = (
        _canonical(checkpoint, label="base-SFT resume checkpoint")
        if checkpoint_fd is None
        else checkpoint
    )
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError("base-SFT resume checkpoint name is invalid")
    step = int(match.group(1))
    if (
        checkpoint_path.parent.name != "checkpoints"
        or str(checkpoint_path.parent.parent) != run_contract["run_dir"]
    ):
        raise ValueError("base-SFT resume checkpoint is outside its run-contract cell")
    owned_fd = checkpoint_fd is None
    active_fd = (
        os.open(
            checkpoint_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        if checkpoint_fd is None
        else os.dup(checkpoint_fd)
    )
    try:
        trainer_state = _json_object_fd(
            active_fd,
            "trainer_state.json",
            label="trainer_state.json",
        )
    finally:
        os.close(active_fd)
    if (
        type(trainer_state.get("global_step")) is not int
        or trainer_state["global_step"] != step
        or type(trainer_state.get("max_steps")) is not int
        or trainer_state["max_steps"] != 31_250
    ):
        raise ValueError("base-SFT trainer state disagrees with checkpoint step/horizon")
    if owned_fd:
        adapter = publication_fidelity_identity(checkpoint_path)
    else:
        adapter, _ = _bound_adapter_fidelity_identity(checkpoint_fd)
    return {
        "campaign": {
            "canonical_path": str(_canonical(campaign_path, label="base-SFT campaign")),
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "campaign_file_sha256": campaign_file_sha,
        },
        "run_contract": run_contract,
        "environment": environment,
        "checkpoint_path": str(checkpoint_path),
        "global_step": step,
        "max_steps": 31_250,
        "world_size": 4,
        "adapter_fidelity": adapter,
    }


def _validate_checkpoint_tree_entries(checkpoint: Path) -> None:
    for candidate in sorted(checkpoint.iterdir(), key=lambda item: item.name):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"resume checkpoint contains a non-regular entry: {candidate.name}")


def _prepare_checkpoint_for_seal(checkpoint: Path) -> None:
    _validate_checkpoint_tree_entries(checkpoint)
    checkpoint.chmod(0o750)
    for candidate in sorted(checkpoint.iterdir(), key=lambda item: item.name):
        candidate.chmod(0o440)


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite base-SFT resume receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite base-SFT resume receipt: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            if path.exists():
                receipt_fd = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    os.fsync(receipt_fd)
                finally:
                    os.close(receipt_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def seal_resume_state(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    distributed: Mapping[str, Any] | None = None,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    checkpoint_path = _canonical(checkpoint, label="base-SFT resume checkpoint")
    receipt_path = checkpoint_path / RECEIPT_FILENAME
    if receipt_path.exists() or receipt_path.is_symlink():
        return resume_state_identity(
            campaign_path=campaign_path,
            run_contract_path=run_contract_path,
            checkpoint=checkpoint_path,
            require_accelerator=require_accelerator,
            verify_live_environment=verify_live_environment,
        )
    if distributed is None:
        raise ValueError("base-SFT resume receipt requires live all-rank consensus")
    core = _core_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )
    names = {candidate.name for candidate in checkpoint_path.iterdir()}
    extras = sorted(names - REQUIRED_CHECKPOINT_FILES)
    missing = sorted(REQUIRED_CHECKPOINT_FILES - names)
    if extras or missing:
        raise ValueError(f"resume checkpoint cannot be sealed: missing={missing}, extras={extras}")
    _validate_checkpoint_tree_entries(checkpoint_path)
    root_stat = checkpoint_path.stat(follow_symlinks=False)
    projected_root = {
        "mode": "0550",
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
        "uid": root_stat.st_uid,
        "gid": root_stat.st_gid,
        "nlink": root_stat.st_nlink,
    }
    directory_fd = os.open(checkpoint_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        records = [_file_record(directory_fd, name) for name in sorted(names)]
    finally:
        os.close(directory_fd)
    tree: dict[str, Any] = {
        "root": projected_root,
        "file_count": len(records),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    tree["tree_sha256"] = sha256_json(tree)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "claim_scope": CLAIM_SCOPE,
        "publication_eligible": True,
        "sota_eligible": True,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **core,
        "checkpoint_tree": tree,
        "resume_state": {
            name: next(record for record in records if record["path"] == name)
            for name in (*REQUIRED_STATE_FILES, *(f"rng_state_{rank}.pth" for rank in range(4)))
        },
        "distributed": dict(distributed),
    }
    payload["receipt_sha256"] = sha256_json(payload)
    _write_create_once(receipt_path, payload)
    checkpoint_path.chmod(0o550)
    directory_fd = os.open(
        checkpoint_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    parent_fd = os.open(
        checkpoint_path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return resume_state_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
    )


def _validate_receipt(payload: object) -> dict[str, Any]:
    keys = {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "publication_eligible",
        "sota_eligible",
        "created_at_utc",
        "campaign",
        "run_contract",
        "environment",
        "checkpoint_path",
        "global_step",
        "max_steps",
        "world_size",
        "adapter_fidelity",
        "checkpoint_tree",
        "resume_state",
        "distributed",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("base-SFT resume receipt has an invalid top-level schema")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("receipt_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("base-SFT resume receipt self-hash mismatch")
    created_at = payload.get("created_at_utc")
    try:
        valid_created_at = (
            isinstance(created_at, str)
            and datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")
            == created_at
        )
    except ValueError:
        valid_created_at = False
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("status") != STATUS
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("publication_eligible") is not True
        or payload.get("sota_eligible") is not True
        or not valid_created_at
        or type(payload.get("global_step")) is not int
        or type(payload.get("max_steps")) is not int
        or payload["max_steps"] != 31_250
        or type(payload.get("world_size")) is not int
        or payload["world_size"] != 4
    ):
        raise ValueError("base-SFT resume receipt disagrees with the publication contract")
    tree = _validate_checkpoint_inventory(payload.get("checkpoint_tree"))
    distributed = payload.get("distributed")
    if (
        not isinstance(distributed, dict)
        or set(distributed) != {"phase", "world_size", "ordered_rank_records", "consensus_sha256"}
        or distributed.get("phase") != "post_checkpoint_durable_state"
        or distributed.get("world_size") != 4
        or not isinstance(distributed.get("ordered_rank_records"), list)
        or len(distributed["ordered_rank_records"]) != 4
        or distributed.get("consensus_sha256")
        != sha256_json(
            {
                "phase": distributed["phase"],
                "world_size": distributed["world_size"],
                "ordered_rank_records": distributed["ordered_rank_records"],
            }
        )
    ):
        raise ValueError("base-SFT resume receipt has invalid all-rank consensus")
    ranks = [
        record.get("rank")
        for record in distributed["ordered_rank_records"]
        if isinstance(record, dict)
    ]
    if ranks != list(range(4)):
        raise ValueError("base-SFT resume receipt has invalid ordered ranks")
    rank_keys = {
        "rank",
        "world_size",
        "global_step",
        "checkpoint_device",
        "checkpoint_inode",
        "shared_state",
        "rank_rng",
        "record_sha256",
    }
    state_stat_keys = {"path", "size_bytes", "mode", "device", "inode", "mtime_ns"}
    rng_keys = state_stat_keys | {"sha256"}
    for expected_rank, record in enumerate(distributed["ordered_rank_records"]):
        if not isinstance(record, dict) or set(record) != rank_keys:
            raise ValueError("base-SFT resume receipt has an invalid rank-record schema")
        unsigned_record = dict(record)
        record_hash = unsigned_record.pop("record_sha256", None)
        shared = record.get("shared_state")
        rank_rng = record.get("rank_rng")
        if (
            record_hash != sha256_json(unsigned_record)
            or record.get("rank") != expected_rank
            or record.get("world_size") != 4
            or record.get("global_step") != payload["global_step"]
            or any(
                type(record.get(field)) is not int or record[field] < 0
                for field in ("checkpoint_device", "checkpoint_inode")
            )
            or not isinstance(shared, list)
            or len(shared) != len(REQUIRED_STATE_FILES)
            or not isinstance(rank_rng, dict)
            or set(rank_rng) != rng_keys
            or rank_rng.get("path") != f"rng_state_{expected_rank}.pth"
            or not isinstance(rank_rng.get("sha256"), str)
            or SHA256_RE.fullmatch(rank_rng["sha256"]) is None
        ):
            raise ValueError("base-SFT resume receipt has invalid rank consensus content")
        for index, state_record in enumerate(shared):
            if (
                not isinstance(state_record, dict)
                or set(state_record) != state_stat_keys
                or state_record.get("path") != REQUIRED_STATE_FILES[index]
            ):
                raise ValueError("base-SFT resume receipt has invalid shared-state evidence")
        for state_record in [*shared, rank_rng]:
            if (
                type(state_record.get("size_bytes")) is not int
                or state_record["size_bytes"] < 0
                or re.fullmatch(r"[0-7]{4}", str(state_record.get("mode", ""))) is None
                or any(
                    type(state_record.get(field)) is not int or state_record[field] < 0
                    for field in ("device", "inode", "mtime_ns")
                )
            ):
                raise ValueError("base-SFT resume receipt has malformed rank file evidence")
    state = payload.get("resume_state")
    expected_state_keys = {
        *REQUIRED_STATE_FILES,
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    }
    records = {record["path"]: record for record in tree["files"]}
    for rank_record in distributed["ordered_rank_records"]:
        for observed in [*rank_record["shared_state"], rank_record["rank_rng"]]:
            sealed = records.get(observed["path"])
            if sealed is None or any(
                observed[field] != sealed[field]
                for field in ("path", "size_bytes", "mode", "device", "inode", "mtime_ns")
            ):
                raise ValueError("rank consensus does not match the sealed checkpoint tree")
            if "sha256" in observed and observed["sha256"] != sealed["sha256"]:
                raise ValueError("rank RNG digest does not match the sealed checkpoint tree")
    if (
        not isinstance(state, dict)
        or set(state) != expected_state_keys
        or any(state[name] != records.get(name) for name in expected_state_keys)
    ):
        raise ValueError("base-SFT resume receipt has an invalid resume-state projection")
    return payload


def resume_state_identity(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint: Path,
    require_accelerator: bool,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    checkpoint_path = _canonical(checkpoint, label="base-SFT resume checkpoint")
    descriptor = os.open(
        checkpoint_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return _resume_state_identity_fd(
            campaign_path=campaign_path,
            run_contract_path=run_contract_path,
            checkpoint_path=checkpoint_path,
            checkpoint_fd=descriptor,
            require_accelerator=require_accelerator,
            verify_live_environment=verify_live_environment,
        )
    finally:
        os.close(descriptor)


def _resume_state_identity_fd(
    *,
    campaign_path: Path,
    run_contract_path: Path,
    checkpoint_path: Path,
    checkpoint_fd: int,
    require_accelerator: bool,
    verify_live_environment: bool,
) -> dict[str, Any]:
    receipt_bytes, receipt_stat = _regular_file_bytes_fd(
        checkpoint_fd,
        RECEIPT_FILENAME,
        label="base-SFT resume receipt",
        required_mode=0o440,
    )
    payload = _validate_receipt(_decode_json_bytes(receipt_bytes, label="base-SFT resume receipt"))
    expected_serialization = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if receipt_bytes != expected_serialization:
        raise ValueError("base-SFT resume receipt is not canonically serialized")
    observed_root = os.fstat(checkpoint_fd)
    expected_root = payload["checkpoint_tree"]["root"]
    if (
        not stat.S_ISDIR(observed_root.st_mode)
        or observed_root.st_dev != expected_root["device"]
        or observed_root.st_ino != expected_root["inode"]
        or f"{stat.S_IMODE(observed_root.st_mode):04o}" != expected_root["mode"]
    ):
        raise ValueError("live resume checkpoint directory differs from the sealed inode")
    core = _core_identity(
        campaign_path=campaign_path,
        run_contract_path=run_contract_path,
        checkpoint=checkpoint_path,
        require_accelerator=require_accelerator,
        verify_live_environment=verify_live_environment,
        checkpoint_fd=checkpoint_fd,
    )
    for key, expected in core.items():
        if payload.get(key) != expected:
            raise ValueError(f"live base-SFT resume {key} differs from the receipt")
    observed_tree = _checkpoint_inventory(checkpoint_fd)
    if observed_tree != payload["checkpoint_tree"]:
        raise ValueError("live base-SFT resume checkpoint tree differs from the receipt")
    adapter, checkpoint_receipt = _bound_adapter_fidelity_identity(checkpoint_fd)
    if adapter != payload["adapter_fidelity"]:
        raise ValueError("live checkpoint-fidelity identity differs from resume receipt")
    records = {item["path"]: item for item in observed_tree["files"]}
    if (
        records["checkpoint_fidelity.json"]["sha256"] != adapter["receipt_file_sha256"]
        or records["adapter_config.json"]["sha256"] != adapter["adapter_config_sha256"]
        or records[MODEL_FILENAME]["sha256"] != adapter["adapter_weights_sha256"]
        or checkpoint_receipt.get("receipt_sha256") != adapter["receipt_sha256"]
    ):
        raise ValueError("resume tree is not cross-bound to exact adapter fidelity")
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "canonical_path": str(checkpoint_path / RECEIPT_FILENAME),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_device": observed_root.st_dev,
        "checkpoint_inode": observed_root.st_ino,
        "global_step": payload["global_step"],
        "world_size": payload["world_size"],
        "receipt_sha256": payload["receipt_sha256"],
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_device": receipt_stat.st_dev,
        "receipt_inode": receipt_stat.st_ino,
        "checkpoint_tree_sha256": observed_tree["tree_sha256"],
        "adapter_fidelity_identity_sha256": adapter["identity_sha256"],
        "campaign_sha256": payload["campaign"]["campaign_sha256"],
        "run_contract_sha256": payload["run_contract"]["run_contract_sha256"],
        "environment_identity_sha256": payload["environment"]["identity_sha256"],
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _state_stat(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resume consensus file is missing or unsafe: {path}")
    value = path.stat(follow_symlinks=False)
    return {
        "path": path.name,
        "size_bytes": value.st_size,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
    }


def _state_stat_fd(parent_fd: int, name: str) -> dict[str, Any]:
    digest, value = _sha256_file_fd(parent_fd, name, label=f"resume consensus file {name}")
    return {
        "path": name,
        "size_bytes": value.st_size,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
        "sha256": digest,
    }


def distributed_checkpoint_consensus(
    checkpoint: Path | int,
    *,
    global_step: int,
) -> dict[str, Any]:
    """Collect exact four-rank post-save evidence from the active process group."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("base-SFT resume fidelity requires initialized torch.distributed")
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    if world_size != 4 or rank not in range(4):
        raise ValueError(f"base-SFT resume fidelity requires ranks 0..3, got {rank}/{world_size}")
    dist.barrier()
    try:
        owned_fd = not isinstance(checkpoint, int)
        if owned_fd:
            checkpoint_path = _canonical(checkpoint, label="base-SFT resume checkpoint")
            match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
            if match is None or int(match.group(1)) != global_step:
                raise ValueError("resume checkpoint name disagrees with callback global step")
            root_fd = os.open(
                checkpoint_path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        else:
            root_fd = os.dup(checkpoint)
        root_stat = os.fstat(root_fd)
        rng_record = _state_stat_fd(root_fd, f"rng_state_{rank}.pth")
        record: dict[str, Any] = {
            "rank": rank,
            "world_size": world_size,
            "global_step": global_step,
            "checkpoint_device": root_stat.st_dev,
            "checkpoint_inode": root_stat.st_ino,
            "shared_state": [
                {
                    key: value
                    for key, value in _state_stat_fd(root_fd, name).items()
                    if key != "sha256"
                }
                for name in REQUIRED_STATE_FILES
            ],
            "rank_rng": rng_record,
        }
        record["record_sha256"] = sha256_json(record)
        local_result: dict[str, Any] = {"rank": rank, "ok": True, "record": record}
    except Exception as exc:
        local_result = {
            "rank": rank,
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        if "root_fd" in locals():
            os.close(root_fd)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    envelopes = sorted(gathered, key=lambda value: int(value["rank"]))  # type: ignore[index]
    if [value.get("rank") for value in envelopes if isinstance(value, dict)] != list(range(4)):
        raise ValueError("base-SFT resume fidelity rank consensus is incomplete")
    failures = [
        f"rank{value['rank']}={value.get('error', 'unknown')}"
        for value in envelopes
        if isinstance(value, dict) and value.get("ok") is not True
    ]
    if failures:
        raise ValueError("base-SFT resume rank-local verification failed: " + "; ".join(failures))
    ordered = [value["record"] for value in envelopes if isinstance(value, dict)]
    shared_projections = {
        sha256_json(
            {
                "checkpoint_device": value["checkpoint_device"],
                "checkpoint_inode": value["checkpoint_inode"],
                "shared_state": value["shared_state"],
            }
        )
        for value in ordered
        if isinstance(value, dict)
    }
    if len(shared_projections) != 1:
        raise ValueError("base-SFT ranks disagree on shared checkpoint state")
    evidence: dict[str, Any] = {
        "phase": "post_checkpoint_durable_state",
        "world_size": world_size,
        "ordered_rank_records": ordered,
    }
    evidence["consensus_sha256"] = sha256_json(evidence)
    return evidence


def _fsync_checkpoint(checkpoint: Path) -> None:
    directory_fd = os.open(
        checkpoint,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for name in sorted(REQUIRED_CHECKPOINT_FILES):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _enabled() -> bool:
    return str(os.environ.get(ENABLE_ENV, "0")).strip().lower() in {"1", "true", "yes", "on"}


def _bound_paths() -> tuple[Path, Path]:
    campaign = str(os.environ.get(CAMPAIGN_ENV, "")).strip()
    run_contract = str(os.environ.get(RUN_CONTRACT_ENV, "")).strip()
    if not campaign or not run_contract:
        raise ValueError(f"{ENABLE_ENV}=1 requires {CAMPAIGN_ENV} and {RUN_CONTRACT_ENV}")
    return Path(campaign), Path(run_contract)


def _semantic_value(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        elements_per_chunk = 8 * 1024 * 1024
        for offset in range(0, raw.numel(), elements_per_chunk):
            digest.update(raw[offset : offset + elements_per_chunk].numpy().tobytes())
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": digest.hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        items = [(_semantic_value(key), _semantic_value(item)) for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_semantic_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_semantic_value(item) for item in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    if is_dataclass(value):
        return _semantic_value(asdict(value))
    raise ValueError(f"unsupported resume-state semantic value: {type(value).__name__}")


def _semantic_sha256(value: Any) -> str:
    return sha256_json(_semantic_value(value))


def _checkpoint_payload_fd(checkpoint_fd: int) -> dict[str, Any]:
    raw, _ = _regular_file_bytes_fd(
        checkpoint_fd,
        RECEIPT_FILENAME,
        label="base-SFT resume receipt",
        required_mode=0o440,
    )
    return _validate_receipt(_decode_json_bytes(raw, label="base-SFT resume receipt"))


def _audit_loaded_rng_state(checkpoint_fd: int, *, rank: int) -> dict[str, Any]:
    import numpy as np
    import torch

    expected = _torch_load_fd(checkpoint_fd, f"rng_state_{rank}.pth", weights_only=False)
    if not isinstance(expected, dict):
        raise ValueError("sealed rank RNG state is not a mapping")
    actual: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        actual["cuda"] = torch.cuda.random.get_rng_state_all()
    if set(actual) != set(expected) or _semantic_sha256(actual) != _semantic_sha256(expected):
        raise ValueError(f"rank {rank} live RNG state differs immediately after Trainer restore")
    return {
        "rank": rank,
        "rng_state_sha256": _semantic_sha256(actual),
        "cuda_device_count": len(actual.get("cuda", [])),
    }


def _rank_synchronized_operation(
    label: str,
    operation: Any,
    *,
    require_equal_evidence: bool,
) -> list[dict[str, Any]]:
    """Run rank-local verification and make every rank fail together."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError(f"{label} requires initialized torch.distributed")
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    if world_size != 4 or rank not in range(4):
        raise ValueError(f"{label} requires ranks 0..3, got {rank}/{world_size}")
    try:
        evidence = operation()
        envelope: dict[str, Any] = {"rank": rank, "ok": True, "evidence": evidence}
    except Exception as exc:
        envelope = {
            "rank": rank,
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, envelope)
    if any(not isinstance(item, dict) for item in gathered):
        raise ValueError(f"{label} returned a malformed rank envelope")
    ordered = sorted(gathered, key=lambda item: int(item["rank"]))  # type: ignore[index]
    if [item.get("rank") for item in ordered] != list(range(4)):
        raise ValueError(f"{label} rank envelopes are incomplete or duplicated")
    failures = [
        f"rank{item['rank']}={item.get('error', 'unknown')}"
        for item in ordered
        if item.get("ok") is not True
    ]
    if failures:
        raise ValueError(f"{label} failed: " + "; ".join(failures))
    evidence = [item["evidence"] for item in ordered]
    if require_equal_evidence and len({sha256_json(item) for item in evidence}) != 1:
        raise ValueError(f"{label} evidence differs across ranks")
    return evidence


def _class_identity(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _optimizer_state_identity(optimizer: Any) -> dict[str, Any]:
    wrappers: list[str] = []
    current = optimizer
    seen: set[int] = set()
    while hasattr(current, "optimizer"):
        if id(current) in seen:
            raise ValueError("optimizer wrapper chain contains a cycle")
        seen.add(id(current))
        wrappers.append(_class_identity(current))
        nested = current.optimizer
        if nested is current:
            raise ValueError("optimizer wrapper points to itself")
        current = nested
    wrapper_state = optimizer.state_dict()
    state = current.state_dict()
    if _semantic_sha256(wrapper_state) != _semantic_sha256(state):
        raise ValueError("optimizer wrapper state differs from its unwrapped optimizer")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("live optimizer state_dict has an unexpected schema")
    groups = state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("live optimizer has no parameter groups")
    layout = [
        {
            "keys": sorted(group),
            "parameter_ids": list(group.get("params", [])),
        }
        for group in groups
        if isinstance(group, dict)
    ]
    if len(layout) != len(groups):
        raise ValueError("live optimizer contains a malformed parameter group")
    return {
        "wrapper_classes": wrappers,
        "unwrapped_class": _class_identity(current),
        "wrapper_state_exact": True,
        "state_dict_sha256": _semantic_sha256(state),
        "parameter_group_layout_sha256": sha256_json(layout),
        "parameter_group_count": len(groups),
    }


def _scheduler_state_identity(scheduler: Any) -> dict[str, Any]:
    wrappers: list[str] = []
    current = scheduler
    seen: set[int] = set()
    while hasattr(current, "scheduler"):
        if id(current) in seen:
            raise ValueError("scheduler wrapper chain contains a cycle")
        seen.add(id(current))
        wrappers.append(_class_identity(current))
        nested = current.scheduler
        if nested is current:
            raise ValueError("scheduler wrapper points to itself")
        current = nested
    wrapper_state = scheduler.state_dict()
    state = current.state_dict()
    if not isinstance(state, dict):
        raise ValueError("live scheduler state_dict has an unexpected schema")
    if _semantic_sha256(wrapper_state) != _semantic_sha256(state):
        raise ValueError("scheduler wrapper state differs from its unwrapped scheduler")
    return {
        "wrapper_classes": wrappers,
        "unwrapped_class": _class_identity(current),
        "wrapper_state_exact": True,
        "state_dict_sha256": _semantic_sha256(state),
        "state_keys": sorted(state),
    }


def _trajectory_state(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = asdict(value) if is_dataclass(value) else dict(vars(value))
    for process_local in ("is_local_process_zero", "is_world_process_zero"):
        raw.pop(process_local, None)
    return raw


def pin_verified_resume_for_trainer(trainer: Any, checkpoint: Path) -> str:
    """Pin a verified checkpoint inode and install post-load state/RNG audits."""

    if not _enabled():
        return str(checkpoint)
    holder: dict[str, Any] = {}

    def open_and_check_local() -> dict[str, Any]:
        canonical = _canonical(checkpoint, label="base-SFT resume checkpoint")
        descriptor = os.open(
            canonical,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        holder["descriptor"] = descriptor
        holder["canonical"] = canonical
        payload = _checkpoint_payload_fd(descriptor)
        holder["payload"] = payload
        observed = os.fstat(descriptor)
        expected_root = payload["checkpoint_tree"]["root"]
        if (
            observed.st_dev != expected_root["device"]
            or observed.st_ino != expected_root["inode"]
            or f"{stat.S_IMODE(observed.st_mode):04o}" != expected_root["mode"]
            or _checkpoint_inventory(descriptor) != payload["checkpoint_tree"]
        ):
            raise ValueError("opened resume checkpoint differs from its sealed receipt")
        return {
            "canonical_path": str(canonical),
            "checkpoint_device": observed.st_dev,
            "checkpoint_inode": observed.st_ino,
            "receipt_sha256": payload["receipt_sha256"],
            "checkpoint_tree_sha256": payload["checkpoint_tree"]["tree_sha256"],
            "global_step": payload["global_step"],
        }

    try:
        opened = _rank_synchronized_operation(
            "base-SFT resume checkpoint open",
            open_and_check_local,
            require_equal_evidence=True,
        )[0]
        descriptor = int(holder["descriptor"])
        canonical = Path(holder["canonical"])
        proc_path = Path(f"/proc/self/fd/{descriptor}")
        payload = holder["payload"]
        identity = verify_resume_before_trainer_load(
            canonical,
            checkpoint_fd=descriptor,
        )

        def final_pin_readiness() -> dict[str, Any]:
            if identity is None or any(
                opened[field] != identity[field]
                for field in (
                    "checkpoint_device",
                    "checkpoint_inode",
                    "receipt_sha256",
                    "checkpoint_tree_sha256",
                    "global_step",
                )
            ):
                raise ValueError("pinned resume evidence differs from final all-rank identity")
            if not callable(getattr(trainer, "_load_rng_state", None)):
                raise ValueError("Trainer lacks the required RNG-state loader")
            if _checkpoint_inventory(descriptor) != payload["checkpoint_tree"]:
                raise ValueError("pinned resume tree changed during final readiness check")
            return {
                "identity_sha256": identity["identity_sha256"],
                "checkpoint_tree_sha256": payload["checkpoint_tree"]["tree_sha256"],
            }

        _rank_synchronized_operation(
            "base-SFT final resume-pin readiness",
            final_pin_readiness,
            require_equal_evidence=True,
        )
        original_rng_loader = trainer._load_rng_state
        trainer._clasp_base_sft_rng_restore_calls = 0

        def audited_rng_loader(path: str | os.PathLike[str] | None) -> None:
            import torch.distributed as dist

            rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
            trainer._clasp_base_sft_rng_restore_calls += 1

            def restore_and_audit() -> dict[str, Any]:
                if trainer._clasp_base_sft_rng_restore_calls != 1:
                    raise ValueError("Trainer must restore sealed RNG exactly once")
                if path is None:
                    raise ValueError("Trainer attempted resume RNG restore without a checkpoint")
                supplied = os.stat(path, follow_symlinks=True)
                pinned = os.fstat(descriptor)
                if supplied.st_dev != pinned.st_dev or supplied.st_ino != pinned.st_ino:
                    raise ValueError("Trainer RNG loader received a different checkpoint inode")
                original_rng_loader(str(proc_path))
                evidence = _audit_loaded_rng_state(descriptor, rank=rank)
                if _checkpoint_inventory(descriptor) != payload["checkpoint_tree"]:
                    raise ValueError("pinned resume checkpoint changed during RNG restore")
                return evidence

            _rank_synchronized_operation(
                "base-SFT immediate RNG restore",
                restore_and_audit,
                require_equal_evidence=False,
            )
            trainer._clasp_base_sft_rng_restored = True

        trainer._load_rng_state = audited_rng_loader
        trainer._clasp_base_sft_pinned_resume = {
            "descriptor": descriptor,
            "canonical_path": str(canonical),
            "proc_path": str(proc_path),
            "receipt": payload,
            "initial_identity": identity,
        }
        trainer._clasp_base_sft_rng_restored = False
        trainer._clasp_base_sft_loaded_state_verified = False
        trainer._clasp_base_sft_first_step_gated = False
        return str(proc_path)
    except Exception:
        if "descriptor" in holder:
            os.close(int(holder["descriptor"]))
        raise


def finalize_pinned_resume_for_trainer(trainer: Any) -> None:
    pinned = getattr(trainer, "_clasp_base_sft_pinned_resume", None)
    if pinned is None:
        return
    try:

        def final_local_check() -> dict[str, Any]:
            if getattr(trainer, "_clasp_base_sft_loaded_state_verified", False) is not True:
                raise ValueError(
                    "Trainer completed resume without audited model/optimizer/scheduler load"
                )
            if getattr(trainer, "_clasp_base_sft_rng_restored", False) is not True:
                raise ValueError("Trainer completed resume without audited rank RNG restoration")
            if getattr(trainer, "_clasp_base_sft_rng_restore_calls", 0) != 1:
                raise ValueError("Trainer did not restore sealed RNG exactly once")
            tree = _checkpoint_inventory(int(pinned["descriptor"]))
            if tree != pinned["receipt"]["checkpoint_tree"]:
                raise ValueError("pinned resume checkpoint changed during Trainer load/use")
            return {
                "checkpoint_tree_sha256": tree["tree_sha256"],
                "receipt_sha256": pinned["receipt"]["receipt_sha256"],
            }

        _rank_synchronized_operation(
            "base-SFT final pinned-resume audit",
            final_local_check,
            require_equal_evidence=True,
        )
    finally:
        os.close(int(pinned["descriptor"]))
        trainer._clasp_base_sft_pinned_resume = None


def release_pinned_resume_for_trainer(trainer: Any) -> None:
    """Close a pinned descriptor after a failed Trainer call without masking it."""

    pinned = getattr(trainer, "_clasp_base_sft_pinned_resume", None)
    if pinned is not None:
        os.close(int(pinned["descriptor"]))
        trainer._clasp_base_sft_pinned_resume = None


def verify_resume_before_trainer_load(
    checkpoint: Path,
    *,
    checkpoint_fd: int | None = None,
) -> dict[str, Any] | None:
    """Recompute all-rank evidence and verify exact bytes immediately pre-load."""

    if not _enabled():
        return None
    import torch.distributed as dist

    bound_paths: dict[str, Path] = {}

    def resolve_bound_paths() -> dict[str, str]:
        campaign_path, run_contract_path = _bound_paths()
        bound_paths["campaign"] = campaign_path
        bound_paths["run_contract"] = run_contract_path
        return {
            "campaign": str(campaign_path.expanduser().absolute()),
            "run_contract": str(run_contract_path.expanduser().absolute()),
        }

    _rank_synchronized_operation(
        "base-SFT resume contract paths",
        resolve_bound_paths,
        require_equal_evidence=True,
    )
    campaign_path = bound_paths["campaign"]
    run_contract_path = bound_paths["run_contract"]
    resolved_checkpoint: dict[str, Path] = {}

    def resolve_checkpoint() -> dict[str, str]:
        checkpoint_path = (
            _canonical(checkpoint, label="base-SFT resume checkpoint")
            if checkpoint_fd is None
            else checkpoint
        )
        resolved_checkpoint["path"] = checkpoint_path
        return {"checkpoint_path": str(checkpoint_path)}

    _rank_synchronized_operation(
        "base-SFT resume checkpoint provenance",
        resolve_checkpoint,
        require_equal_evidence=True,
    )
    checkpoint_path = resolved_checkpoint["path"]
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError("base-SFT resume checkpoint name is invalid")
    evidence = distributed_checkpoint_consensus(
        checkpoint_path if checkpoint_fd is None else checkpoint_fd,
        global_step=int(match.group(1)),
    )
    rank = int(dist.get_rank())
    result: list[object] = [None]
    if rank == 0:
        try:
            if checkpoint_fd is None:
                identity = resume_state_identity(
                    campaign_path=campaign_path,
                    run_contract_path=run_contract_path,
                    checkpoint=checkpoint_path,
                    require_accelerator=False,
                    verify_live_environment=False,
                )
                payload = _validate_receipt(
                    _load_json_object(
                        checkpoint_path / RECEIPT_FILENAME,
                        label="base-SFT resume receipt",
                    )
                )
            else:
                identity = _resume_state_identity_fd(
                    campaign_path=campaign_path,
                    run_contract_path=run_contract_path,
                    checkpoint_path=checkpoint_path,
                    checkpoint_fd=checkpoint_fd,
                    require_accelerator=False,
                    verify_live_environment=True,
                )
                payload = _checkpoint_payload_fd(checkpoint_fd)
            if evidence["ordered_rank_records"] != payload["distributed"]["ordered_rank_records"]:
                raise ValueError("live all-rank resume evidence differs from the sealed receipt")
            result[0] = {"ok": True, "identity": identity}
        except Exception as exc:  # synchronized failure is re-raised on every rank
            result[0] = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    response = result[0]
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error") if isinstance(response, dict) else response
        raise ValueError(f"base-SFT in-process resume verification failed: {detail}")
    dist.barrier()
    return response["identity"]


class BaseSFTResumeFidelityCallback(TrainerCallback):
    """Publish the resume receipt after the existing fidelity callback."""

    def __init__(self, trainer: Any):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        expected_horizon = int(os.environ[EXPECTED_MAX_STEPS_ENV])

        def verify_horizon() -> dict[str, Any]:
            actual_horizon = int(getattr(state, "max_steps", -1))
            if actual_horizon != expected_horizon:
                raise ValueError(
                    "base-SFT Trainer horizon differs from the sealed retention contract: "
                    f"actual={actual_horizon}, expected={expected_horizon}"
                )
            return {"max_steps": actual_horizon}

        _rank_synchronized_operation(
            "base-SFT sealed horizon",
            verify_horizon,
            require_equal_evidence=True,
        )
        pinned = getattr(self._trainer, "_clasp_base_sft_pinned_resume", None)
        if pinned is not None:
            payload = pinned["receipt"]
            descriptor = int(pinned["descriptor"])

            def audit_loaded_state() -> dict[str, Any]:
                if int(getattr(state, "global_step", -1)) != payload["global_step"]:
                    raise ValueError("Trainer restored the wrong global step")
                checkpoint_receipt = _bound_adapter_fidelity_identity(descriptor)[1]
                expected_closure = checkpoint_receipt.get("trainable_closure")
                closure = audit_trainable_state_closure(self._trainer.model)
                if closure != expected_closure:
                    raise ValueError(
                        "live trainable policy differs from the sealed adapter inventory"
                    )
                if (
                    closure.get("tensor_inventory_sha256")
                    != payload["adapter_fidelity"]["trainable_tensor_inventory_sha256"]
                ):
                    raise ValueError("live trainable policy identity cross-binding failed")
                expected_optimizer = _torch_load_fd(
                    descriptor,
                    "optimizer.pt",
                    weights_only=True,
                )
                expected_scheduler = _torch_load_fd(
                    descriptor,
                    "scheduler.pt",
                    weights_only=True,
                )
                if self._trainer.optimizer is None:
                    raise ValueError("Trainer did not construct an optimizer before resume audit")
                optimizer = _optimizer_state_identity(self._trainer.optimizer)
                if optimizer["unwrapped_class"] != EXPECTED_OPTIMIZER_CLASS or optimizer[
                    "state_dict_sha256"
                ] != _semantic_sha256(expected_optimizer):
                    raise ValueError("live optimizer state/class differs after Trainer load")
                if self._trainer.lr_scheduler is None:
                    raise ValueError("Trainer did not construct a scheduler before resume audit")
                scheduler = _scheduler_state_identity(self._trainer.lr_scheduler)
                if scheduler["unwrapped_class"] != EXPECTED_SCHEDULER_CLASS or scheduler[
                    "state_dict_sha256"
                ] != _semantic_sha256(expected_scheduler):
                    raise ValueError("live scheduler state/class differs after Trainer load")
                expected_trainer_state = _json_object_fd(
                    descriptor,
                    "trainer_state.json",
                    label="sealed trainer state",
                )
                if _trajectory_state(state) != _trajectory_state(expected_trainer_state):
                    raise ValueError("live Trainer trajectory state differs after resume load")
                tree = _checkpoint_inventory(descriptor)
                if tree != payload["checkpoint_tree"]:
                    raise ValueError("pinned resume checkpoint changed during Trainer state load")
                return {
                    "trainable_closure_sha256": closure["tensor_inventory_sha256"],
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "trainer_state_sha256": _semantic_sha256(_trajectory_state(state)),
                    "checkpoint_tree_sha256": tree["tree_sha256"],
                }

            _rank_synchronized_operation(
                "base-SFT loaded resume state",
                audit_loaded_state,
                require_equal_evidence=True,
            )
            self._trainer._clasp_base_sft_loaded_state_verified = True
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        pinned = getattr(self._trainer, "_clasp_base_sft_pinned_resume", None)
        if pinned is not None and not getattr(
            self._trainer,
            "_clasp_base_sft_first_step_gated",
            False,
        ):

            def rng_gate() -> dict[str, Any]:
                restored = getattr(
                    self._trainer,
                    "_clasp_base_sft_rng_restored",
                    False,
                )
                calls = getattr(self._trainer, "_clasp_base_sft_rng_restore_calls", 0)
                if restored is not True or calls != 1:
                    raise ValueError("sealed RNG was not restored exactly once before resumed step")
                return {"rng_restored": restored, "restore_calls": calls}

            _rank_synchronized_operation(
                "base-SFT pre-first-step RNG gate",
                rng_gate,
                require_equal_evidence=True,
            )
            self._trainer._clasp_base_sft_first_step_gated = True
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if not _enabled():
            return control
        import torch.distributed as dist

        global_step = int(getattr(state, "global_step", -1))
        checkpoint = Path(str(args.output_dir)) / f"checkpoint-{global_step}"
        rank = int(dist.get_rank())
        # HF writes per-rank RNG files without a post-save barrier.  No rank may
        # inspect or seal the shared checkpoint until every writer has arrived.
        dist.barrier()
        prepare_status: list[object] = [None]
        if rank == 0:
            try:
                _prepare_checkpoint_for_seal(checkpoint)
                _fsync_checkpoint(checkpoint)
                prepare_status[0] = {"ok": True}
            except Exception as exc:
                prepare_status[0] = {
                    "ok": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        dist.broadcast_object_list(prepare_status, src=0)
        prepared = prepare_status[0]
        if not isinstance(prepared, dict) or prepared.get("ok") is not True:
            detail = prepared.get("error") if isinstance(prepared, dict) else prepared
            raise ValueError(f"base-SFT checkpoint durable-save preparation failed: {detail}")
        evidence = distributed_checkpoint_consensus(checkpoint, global_step=global_step)
        seal_status: list[object] = [None]
        if rank == 0:
            try:
                campaign_path, run_contract_path = _bound_paths()
                seal_resume_state(
                    campaign_path=campaign_path,
                    run_contract_path=run_contract_path,
                    checkpoint=checkpoint,
                    require_accelerator=False,
                    distributed=evidence,
                    verify_live_environment=False,
                )
                seal_status[0] = {"ok": True}
            except Exception as exc:
                seal_status[0] = {
                    "ok": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        dist.broadcast_object_list(seal_status, src=0)
        sealed = seal_status[0]
        if not isinstance(sealed, dict) or sealed.get("ok") is not True:
            detail = sealed.get("error") if isinstance(sealed, dict) else sealed
            raise ValueError(f"base-SFT resume receipt publication failed: {detail}")
        verify_resume_before_trainer_load(checkpoint)
        return control


class SynchronizedBaseSFTPublicationCallback(BaseSFTResumeFidelityCallback):
    """Serialize publication-receipt and resume-receipt callback failures."""

    def __init__(self, trainer: Any):
        super().__init__(trainer)
        from clasp.models.training_fidelity import PublicationFidelityCheckpointCallback

        self._publication = PublicationFidelityCheckpointCallback(trainer)

    def on_train_begin(self, args, state, control, **kwargs):
        publication_control: list[Any] = [control]

        def publication_begin() -> dict[str, Any]:
            publication_control[0] = self._publication.on_train_begin(
                args,
                state,
                control,
                **kwargs,
            )
            return {"publication_preflight": True}

        _rank_synchronized_operation(
            "base-SFT checkpoint-fidelity train-begin",
            publication_begin,
            require_equal_evidence=True,
        )
        return super().on_train_begin(
            args,
            state,
            publication_control[0],
            **kwargs,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(self._trainer, "_clasp_exact_peft_kl_reference", None) is not None:
            raise ValueError("base-SFT must not install an exact-PEFT KL reference")
        return control

    def on_save(self, args, state, control, **kwargs):
        publication_control: list[Any] = [control]

        def publication_save() -> dict[str, Any]:
            publication_control[0] = self._publication.on_save(
                args,
                state,
                control,
                **kwargs,
            )
            return {
                "publication_checkpoint_fidelity": True,
                "global_step": int(getattr(state, "global_step", -1)),
            }

        _rank_synchronized_operation(
            "base-SFT checkpoint-fidelity save",
            publication_save,
            require_equal_evidence=True,
        )
        return super().on_save(
            args,
            state,
            publication_control[0],
            **kwargs,
        )


def install_base_sft_resume_fidelity_callback(trainer: Any) -> bool:
    if not _enabled():
        return False

    def preflight() -> dict[str, Any]:
        raw_horizon = str(os.environ.get(EXPECTED_MAX_STEPS_ENV, "")).strip()
        try:
            horizon = int(raw_horizon)
        except ValueError as exc:
            raise ValueError(f"{EXPECTED_MAX_STEPS_ENV} must be an integer") from exc
        trainer_args = trainer.args
        save_strategy = getattr(
            getattr(trainer_args, "save_strategy", None),
            "value",
            None,
        )
        if save_strategy is None:
            save_strategy = str(getattr(trainer_args, "save_strategy", "")).lower()
        save_steps = getattr(trainer_args, "save_steps", None)
        save_limit = getattr(trainer_args, "save_total_limit", None)
        configured_horizon = int(getattr(trainer_args, "max_steps", -1))
        if (
            horizon != 31_250
            or type(save_steps) is not int
            or save_steps != 500
            or type(save_limit) is not int
            or save_limit != 64
            or save_strategy != "steps"
            or bool(getattr(trainer_args, "save_only_model", False))
            or not bool(getattr(trainer_args, "save_safetensors", False))
            or bool(getattr(trainer_args, "ignore_data_skip", False))
            or configured_horizon not in {-1, horizon}
            or (horizon + save_steps - 1) // save_steps > save_limit
        ):
            raise ValueError(
                "base-SFT resume fidelity requires horizon=31250, save_steps=500, "
                "save_total_limit=64, full optimizer/RNG saves, safetensors, and "
                "ignore_data_skip=False"
            )
        if getattr(trainer, "_clasp_exact_peft_kl_reference", None) is not None:
            raise ValueError("base-SFT resume fidelity forbids exact-PEFT KL state")
        return {
            "horizon": horizon,
            "save_steps": save_steps,
            "save_total_limit": save_limit,
            "ignore_data_skip": False,
        }

    _rank_synchronized_operation(
        "base-SFT resume callback installation",
        preflight,
        require_equal_evidence=True,
    )
    trainer.add_callback(SynchronizedBaseSFTPublicationCallback(trainer))
    return True


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
    try:
        identity = (
            seal_resume_state(
                campaign_path=args.campaign_contract,
                run_contract_path=args.run_contract,
                checkpoint=args.checkpoint,
                require_accelerator=args.require_live_accelerator,
            )
            if args.seal
            else resume_state_identity(
                campaign_path=args.campaign_contract,
                run_contract_path=args.run_contract,
                checkpoint=args.checkpoint,
                require_accelerator=args.require_live_accelerator,
            )
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"[BaseSFTResumeFidelityError] {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
