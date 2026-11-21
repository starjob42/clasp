#!/usr/bin/env python3
"""Seal one certified Jiang-f638 specialist run for immutable RL handoff.

The trainer must finish all terminal and periodic artifacts before this script
is called.  The script rejects incomplete or unsafe trees before changing any
mode, then makes every regular file read-only (0440) and every directory
read/search-only (0550).  Re-running it on an already sealed tree is an
idempotent verification; it never repairs drift after the tree is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from audit_lora_persistence import validate_audit_receipt  # type: ignore[import-not-found]
from audit_sft_reload_equivalence import (  # type: ignore[import-not-found]
    RECEIPT_FILENAME as RELOAD_EQUIVALENCE_FILENAME,
)
from audit_sft_reload_equivalence import validate_reload_receipt

from clasp.models.publication_fidelity import publication_fidelity_identity

TARGET_STEP = 31_250
SAVE_STEPS = 5_000
PERIODIC_CHECKPOINT_STEPS = tuple(range(SAVE_STEPS, TARGET_STEP, SAVE_STEPS))
EXPECTED_TRAINER_CHECKPOINT_STEPS = (
    *PERIODIC_CHECKPOINT_STEPS,
    TARGET_STEP,
)
EXPECTED_CHECKPOINT_NAMES = tuple(
    f"checkpoint-{step}" for step in EXPECTED_TRAINER_CHECKPOINT_STEPS
)
FILE_MODE = 0o440
DIRECTORY_MODE = 0o550
SEAL_RECEIPT_FILENAME = "completed_sft_run_seal.json"
SEAL_PROTOCOL = "clasp-jiang-f638-completed-sft-run-seal-v1"
MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_ROOT_FILES = (
    "README.md",
    "added_tokens.json",
    "adapter_config.json",
    "adapter_persistence_receipt.json",
    "base_sft_run_contract.json",
    "checkpoint_fidelity.json",
    "hyperparams.json",
    "merges.txt",
    RELOAD_EQUIVALENCE_FILENAME,
    "sft_log_history.json",
    "sft_trainer_state.json",
    "special_tokens_map.json",
    "stage_status.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "vocab.json",
)
REQUIRED_CHECKPOINT_FILES = (
    "README.md",
    "added_tokens.json",
    "adapter_config.json",
    "checkpoint_fidelity.json",
    "merges.txt",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "vocab.json",
)


class CompletedSFTRunSealError(ValueError):
    """Raised when a completed run cannot be safely sealed or verified."""


def _canonical_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise CompletedSFTRunSealError(f"{label} is not a regular directory: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise CompletedSFTRunSealError(
            f"{label} must be supplied canonically: supplied={raw}, resolved={resolved}"
        )
    return resolved


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompletedSFTRunSealError(f"required artifact is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletedSFTRunSealError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CompletedSFTRunSealError(f"expected a JSON object in {path}")
    return payload


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_model_file(root: Path, *, label: str) -> Path:
    candidates = [root / name for name in MODEL_FILES]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise CompletedSFTRunSealError(
            f"{label} must contain exactly one adapter weight artifact, found={present}"
        )
    return present[0]


def _require_base_binding(adapter_root: Path, base_model_path: Path) -> None:
    config = _load_object(adapter_root / "adapter_config.json")
    configured = Path(str(config.get("base_model_name_or_path", ""))).expanduser().resolve()
    if configured != base_model_path:
        raise CompletedSFTRunSealError(
            f"adapter at {adapter_root} targets another base: {configured} != {base_model_path}"
        )


def _tree_inventory(root: Path) -> tuple[list[Path], list[Path]]:
    """Return a stable regular-file/directory inventory without following links."""

    root = _canonical_directory(root, label="completed SFT run")
    files: list[Path] = []
    directories: list[Path] = [root]
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise CompletedSFTRunSealError(f"cannot enumerate completed run: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CompletedSFTRunSealError(
                    f"cannot inspect completed-run entry: {path}"
                ) from exc
            if stat.S_ISLNK(observed.st_mode):
                raise CompletedSFTRunSealError(f"completed run contains a symbolic link: {path}")
            if stat.S_ISDIR(observed.st_mode):
                directories.append(path)
                pending.append(path)
            elif stat.S_ISREG(observed.st_mode):
                files.append(path)
            else:
                raise CompletedSFTRunSealError(f"completed run contains a special entry: {path}")
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    directories.sort(key=lambda path: path.relative_to(root).as_posix())
    return files, directories


def _stable_file_sha256(path: Path) -> tuple[int, str]:
    """Hash one regular file and reject replacement or mutation during the read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        path_before = path.stat(follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompletedSFTRunSealError(f"cannot safely open completed-run file: {path}") from exc
    try:
        descriptor_before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CompletedSFTRunSealError(f"cannot hash stable completed-run file: {path}") from exc
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    identity = tuple(getattr(descriptor_before, field) for field in identity_fields)
    if (
        not stat.S_ISREG(descriptor_before.st_mode)
        or identity != tuple(getattr(descriptor_after, field) for field in identity_fields)
        or identity != tuple(getattr(path_before, field) for field in identity_fields)
        or identity != tuple(getattr(path_after, field) for field in identity_fields)
    ):
        raise CompletedSFTRunSealError(f"completed-run file changed while hashing: {path}")
    return int(descriptor_after.st_size), digest.hexdigest()


def _portable_inventory(run: Path) -> list[dict[str, Any]]:
    files, directories = _tree_inventory(run)
    records: list[dict[str, Any]] = []
    for path in directories:
        records.append(
            {
                "entry_type": "directory",
                "relative_path": "." if path == run else path.relative_to(run).as_posix(),
                "size_bytes": 0,
            }
        )
    for path in files:
        if path == run / SEAL_RECEIPT_FILENAME:
            continue
        size_bytes, sha256 = _stable_file_sha256(path)
        records.append(
            {
                "entry_type": "file",
                "relative_path": path.relative_to(run).as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    records.sort(key=lambda record: (str(record["relative_path"]), str(record["entry_type"])))
    return records


def _seal_receipt_payload(run: Path, base: Path) -> dict[str, Any]:
    inventory = _portable_inventory(run)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SEAL_PROTOCOL,
        "status": "completed_run_seal_committed_before_mode_transition",
        "run_dir": str(run),
        "base_model_path": str(base),
        "terminal_step": TARGET_STEP,
        "periodic_checkpoint_steps": list(PERIODIC_CHECKPOINT_STEPS),
        "forced_terminal_checkpoint_step": TARGET_STEP,
        "trainer_checkpoint_steps": list(EXPECTED_TRAINER_CHECKPOINT_STEPS),
        "file_mode": f"{FILE_MODE:04o}",
        "directory_mode": f"{DIRECTORY_MODE:04o}",
        "preseal_inventory": inventory,
        "preseal_inventory_sha256": _sha256_json(inventory),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def _write_seal_receipt_create_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CompletedSFTRunSealError(f"refusing to replace completed-run seal receipt: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CompletedSFTRunSealError(
                f"refusing to replace completed-run seal receipt: {path}"
            ) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_seal_receipt(run_dir: Path, base_model_path: Path) -> dict[str, Any]:
    run = _canonical_directory(run_dir, label="completed SFT run")
    base = _canonical_directory(base_model_path, label="base model")
    receipt_path = run / SEAL_RECEIPT_FILENAME
    receipt = _load_object(receipt_path)
    claimed_hash = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    inventory = receipt.get("preseal_inventory")
    if (
        set(receipt)
        != {
            "schema_version",
            "protocol",
            "status",
            "run_dir",
            "base_model_path",
            "terminal_step",
            "periodic_checkpoint_steps",
            "forced_terminal_checkpoint_step",
            "trainer_checkpoint_steps",
            "file_mode",
            "directory_mode",
            "preseal_inventory",
            "preseal_inventory_sha256",
            "receipt_sha256",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("protocol") != SEAL_PROTOCOL
        or receipt.get("status") != "completed_run_seal_committed_before_mode_transition"
        or receipt.get("run_dir") != str(run)
        or receipt.get("base_model_path") != str(base)
        or receipt.get("terminal_step") != TARGET_STEP
        or receipt.get("periodic_checkpoint_steps") != list(PERIODIC_CHECKPOINT_STEPS)
        or receipt.get("forced_terminal_checkpoint_step") != TARGET_STEP
        or receipt.get("trainer_checkpoint_steps")
        != list(EXPECTED_TRAINER_CHECKPOINT_STEPS)
        or receipt.get("file_mode") != f"{FILE_MODE:04o}"
        or receipt.get("directory_mode") != f"{DIRECTORY_MODE:04o}"
        or not isinstance(inventory, list)
        or receipt.get("preseal_inventory_sha256") != _sha256_json(inventory)
        or claimed_hash != _sha256_json(unsigned)
    ):
        raise CompletedSFTRunSealError("completed-run seal receipt is invalid")
    if inventory != _portable_inventory(run):
        raise CompletedSFTRunSealError("completed-run inventory drifted after sealing")
    return receipt


def validate_completed_run_artifacts(run_dir: Path, base_model_path: Path) -> dict[str, Any]:
    """Require the terminal adapter and all seven Trainer checkpoints."""

    run = _canonical_directory(run_dir, label="completed SFT run")
    base = _canonical_directory(base_model_path, label="base model")
    _tree_inventory(run)

    missing_root = [name for name in REQUIRED_ROOT_FILES if not (run / name).is_file()]
    if missing_root:
        raise CompletedSFTRunSealError(
            "completed SFT run is missing terminal artifacts: " + ", ".join(missing_root)
        )
    terminal_adapter = _require_model_file(run, label="terminal SFT run")
    _require_base_binding(run, base)

    hyperparams = _load_object(run / "hyperparams.json")
    dataset = hyperparams.get("dataset")
    distributed = hyperparams.get("distributed")
    if (
        hyperparams.get("protocol") != "clasp-jiang-llmcosolver-f638-sft-corrected-v2"
        or hyperparams.get("stage") != "sft"
        or not isinstance(dataset, dict)
        or dataset.get("train_size") != 500_000
        or dataset.get("eval_size") != 0
        or not isinstance(distributed, dict)
        or distributed.get("world_size") != 1
    ):
        raise CompletedSFTRunSealError("terminal hyperparameters are not the Jiang-f638 SFT run")

    status = _load_object(run / "stage_status.json")
    latest = run / "checkpoints" / EXPECTED_CHECKPOINT_NAMES[-1]
    if (
        status.get("stage") != "sft"
        or status.get("completed") is not True
        or status.get("has_saved_model") is not True
        or status.get("global_step") != TARGET_STEP
        or status.get("max_steps") != TARGET_STEP
        or Path(str(status.get("latest_checkpoint_dir", ""))).resolve() != latest
    ):
        raise CompletedSFTRunSealError(
            "terminal stage status does not bind step 31250 to checkpoint-31250"
        )

    checkpoint_root = run / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise CompletedSFTRunSealError(
            f"Trainer checkpoint root is unavailable: {checkpoint_root}"
        )
    observed_checkpoint_entries = sorted(entry.name for entry in checkpoint_root.iterdir())
    if observed_checkpoint_entries != sorted(EXPECTED_CHECKPOINT_NAMES):
        raise CompletedSFTRunSealError(
            "Trainer checkpoint inventory differs from the exact seven-save schedule: "
            f"observed={observed_checkpoint_entries}"
        )

    forced_terminal_adapter: Path | None = None
    for step, name in zip(
        EXPECTED_TRAINER_CHECKPOINT_STEPS,
        EXPECTED_CHECKPOINT_NAMES,
        strict=True,
    ):
        checkpoint = checkpoint_root / name
        checkpoint_label = (
            "forced terminal checkpoint" if step == TARGET_STEP else "periodic checkpoint"
        )
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise CompletedSFTRunSealError(f"{checkpoint_label} is unsafe: {checkpoint}")
        missing = [item for item in REQUIRED_CHECKPOINT_FILES if not (checkpoint / item).is_file()]
        if missing:
            raise CompletedSFTRunSealError(
                f"{checkpoint_label} {name} is missing: {', '.join(missing)}"
            )
        checkpoint_adapter = _require_model_file(
            checkpoint, label=f"{checkpoint_label} {name}"
        )
        if step == TARGET_STEP:
            forced_terminal_adapter = checkpoint_adapter
        _require_base_binding(checkpoint, base)
        trainer_state = _load_object(checkpoint / "trainer_state.json")
        if (
            trainer_state.get("global_step") != step
            or trainer_state.get("max_steps") != TARGET_STEP
        ):
            raise CompletedSFTRunSealError(
                f"{checkpoint_label} {name} has inconsistent trainer state"
            )
        try:
            publication_fidelity_identity(checkpoint)
        except CompletedSFTRunSealError:
            raise
        except Exception as exc:
            raise CompletedSFTRunSealError(
                f"{checkpoint_label} {name} has invalid publication-fidelity evidence: {exc}"
            ) from exc

    if forced_terminal_adapter is None:  # pragma: no cover - inventory proves this above
        raise CompletedSFTRunSealError("forced terminal adapter is unavailable")
    if terminal_adapter.name != forced_terminal_adapter.name:
        raise CompletedSFTRunSealError(
            "root and checkpoint-31250 use different adapter serialization formats"
        )
    _, terminal_adapter_sha256 = _stable_file_sha256(terminal_adapter)
    _, forced_terminal_adapter_sha256 = _stable_file_sha256(forced_terminal_adapter)
    if terminal_adapter_sha256 != forced_terminal_adapter_sha256:
        raise CompletedSFTRunSealError(
            "root adapter is not byte-identical to checkpoint-31250 adapter"
        )

    try:
        publication_fidelity_identity(run)
        validate_audit_receipt(
            run / "adapter_persistence_receipt.json",
            base_model_path=base,
            adapter_path=run,
        )
        validate_reload_receipt(
            run / RELOAD_EQUIVALENCE_FILENAME,
            base_model_path=base,
            adapter_path=run,
        )
    except CompletedSFTRunSealError:
        raise
    except Exception as exc:
        raise CompletedSFTRunSealError(
            f"terminal completed-run certification evidence is invalid: {exc}"
        ) from exc
    return {
        "status": "complete_before_seal",
        "run_dir": str(run),
        "base_model_path": str(base),
        "terminal_step": TARGET_STEP,
        "periodic_checkpoint_steps": list(PERIODIC_CHECKPOINT_STEPS),
        "forced_terminal_checkpoint_step": TARGET_STEP,
        "root_terminal_adapter_sha256": terminal_adapter_sha256,
        "trainer_checkpoint_steps": list(EXPECTED_TRAINER_CHECKPOINT_STEPS),
    }


def require_exact_sealed_modes(run_dir: Path) -> dict[str, int]:
    """Reject any entry whose completed-run mode drifted from 0440/0550."""

    run = _canonical_directory(run_dir, label="completed SFT run")
    files, directories = _tree_inventory(run)
    for path in files:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode != FILE_MODE:
            raise CompletedSFTRunSealError(
                f"completed-run file mode drift: {path} has {mode:04o}, expected {FILE_MODE:04o}"
            )
    for path in directories:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode != DIRECTORY_MODE:
            raise CompletedSFTRunSealError(
                f"completed-run directory mode drift: {path} has {mode:04o}, "
                f"expected {DIRECTORY_MODE:04o}"
            )
    return {"file_count": len(files), "directory_count": len(directories)}


def _fchmod_entry(path: Path, *, directory: bool, mode: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompletedSFTRunSealError(f"cannot safely open completed-run entry: {path}") from exc
    try:
        observed = os.fstat(descriptor)
        if directory and not stat.S_ISDIR(observed.st_mode):
            raise CompletedSFTRunSealError(f"completed-run directory changed during seal: {path}")
        if not directory and not stat.S_ISREG(observed.st_mode):
            raise CompletedSFTRunSealError(f"completed-run file changed during seal: {path}")
        os.fchmod(descriptor, mode)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != mode:
            raise CompletedSFTRunSealError(f"cannot set exact completed-run mode on {path}")
    finally:
        os.close(descriptor)


def seal_completed_run(
    run_dir: Path,
    base_model_path: Path,
    *,
    artifact_validator: Callable[[Path, Path], dict[str, Any]] = validate_completed_run_artifacts,
) -> dict[str, Any]:
    """Validate first, then seal once; an already sealed tree is verify-only."""

    run = _canonical_directory(run_dir, label="completed SFT run")
    base = _canonical_directory(base_model_path, label="base model")
    artifact_evidence = artifact_validator(run, base)
    receipt_path = run / SEAL_RECEIPT_FILENAME
    if receipt_path.exists() or receipt_path.is_symlink():
        validate_seal_receipt(run, base)
        counts = require_exact_sealed_modes(run)
        return {**artifact_evidence, **counts, "status": "already_sealed_and_verified"}

    receipt = _seal_receipt_payload(run, base)
    _write_seal_receipt_create_once(receipt_path, receipt)
    # Reopen and rehash the entire tree before the first chmod. This catches a
    # mutation between inventory construction and create-once receipt commit.
    validate_seal_receipt(run, base)
    files, directories = _tree_inventory(run)

    for path in files:
        _fchmod_entry(path, directory=False, mode=FILE_MODE)
    for path in sorted(
        (path for path in directories if path != run),
        key=lambda path: len(path.relative_to(run).parts),
        reverse=True,
    ):
        _fchmod_entry(path, directory=True, mode=DIRECTORY_MODE)
    _fchmod_entry(run, directory=True, mode=DIRECTORY_MODE)

    counts = require_exact_sealed_modes(run)
    validate_seal_receipt(run, base)
    artifact_validator(run, base)
    return {**artifact_evidence, **counts, "status": "sealed_and_verified"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = seal_completed_run(args.run_dir, args.base_model_path)
    except (CompletedSFTRunSealError, OSError, ValueError) as exc:
        print(f"[CompletedSFTRunSealError] {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
