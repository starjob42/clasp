#!/usr/bin/env python3
"""Atomically publish a read-only, hash-manifested CLASP source snapshot.

The caller first transfers a curated source tree into a unique staging sibling.
This tool rejects caches, artifacts, links, special files, pre-existing outputs,
and nonempty logs; hashes a stable file set; makes it read-only; then renames it
to a never-before-used final path.  Historical snapshots are never modified.

The checked-in ``experiments/`` tree is handled as executable publication
configuration, not as a general artifact directory.  Only the exact static
contracts and templates declared below may enter a source snapshot.  Runtime
evidence, local-development models, checkpoints, logs, raw generations, and
repair outputs belong outside the immutable source tree.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import secrets
import socket
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "SOURCE_MANIFEST.sha256"
SCHEMA_VERSION = 1
FORBIDDEN_TOP_LEVEL = {
    ".agents",
    ".codex",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv_iid",
    "__pycache__",
    "checkpoints",
    "data",
    "models",
    "outputs",
    "pytest-of-habgaryan",
    "results",
    "torchinductor_habgaryan",
}
FORBIDDEN_DIRECTORY_COMPONENTS = {
    ".agents",
    ".cache",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "adapter",
    "adapters",
    "artifacts",
    "cache",
    "checkpoint",
    "checkpoints",
    "local_development",
    "log",
    "logs",
    "model",
    "output",
    "outputs",
    "optimizer",
    "raw",
    "repair",
    "repairs",
    "result",
    "results",
    "run",
    "runs",
    "runtime",
    "runtime_outputs",
    "scheduler",
    "tensorboard",
    "trainer",
    "wandb",
    "weight",
    "weights",
}
FORBIDDEN_DIRECTORY_PREFIXES = (
    ".venv",
    "checkpoint-",
    "checkpoint_",
    "raw_",
    "repair_",
    "repaired_",
    "runtime_",
    "wandb-",
)
FORBIDDEN_DIRECTORY_SUFFIXES = (
    "_adapters",
    "_artifacts",
    "_checkpoints",
    "_logs",
    "_outputs",
    "_raw",
    "_repair",
    "_repairs",
    "_results",
    "_runs",
    "_runtime",
    "_weights",
)
FORBIDDEN_FILE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".err",
    ".gguf",
    ".h5",
    ".hdf5",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".out",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
}
FORBIDDEN_RUNTIME_FILE_NAMES = {
    "adapter_config.json",
    "adapter_persistence_receipt.json",
    "checkpoint_fidelity.json",
    "completed_sft_run_seal.json",
    "exact_peft_kl_post_step.json",
    "exact_peft_kl_reference.json",
    "optimizer.pt",
    "scheduler.pt",
    "sft_cleanliness_audit.json",
    "sft_reload_equivalence.json",
    "sft_reserved_source_registry.json",
    "sft_token_length_audit.json",
    "trainer_state.json",
    "training_args.bin",
}

# Source snapshots carry only static, reviewable publication configuration from
# experiments/.  Additions are deliberate code changes, not implicit inclusion
# of whatever happens to be present in a developer's experiment workspace.
STATIC_EXPERIMENT_FILES = frozenset(
    {
        "experiments/sota_20260814/PROTOCOL.md",
        "experiments/sota_20260814/campaign_manifest.template.json",
        "experiments/sota_20260814/checkpoint_selection_policy.template.json",
        "experiments/sota_20260814/forbidden_source_registry.template.json",
        "experiments/sota_20260814/iid_generator_contract.json",
        "experiments/sota_20260814/lane_contracts.json",
        "experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json",
        "experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json",
        "experiments/sota_20260814/llmcosolver_f638_semantic_contract.json",
        "experiments/sota_20260814/llmcosolver_f638_semantic_contract_v2.json",
        "experiments/sota_20260814/llmcosolver_f638_stack_probe.json",
        "experiments/sota_20260814/reference_solver_contract.template.json",
        "experiments/sota_20260814/targets.json",
        "experiments/sota_20260814/tsp_fcdf_beta_pilot.template.json",
    }
)
REQUIRED_FILES = {
    "pyproject.toml",
    "requirements.jiang-f638-runtime-closure.txt",
    "requirements.jiang-f638-sft.txt",
    "scripts/audit_lora_persistence.py",
    "scripts/audit_sft_reload_equivalence.py",
    "scripts/audit_sft_token_lengths.py",
    "scripts/base_sft_resume_fidelity.py",
    "scripts/base_sft_single_h100_resume_fidelity.py",
    "scripts/build_clean_sft500k_corpus.py",
    "scripts/freeze_base_sft_campaign.py",
    "scripts/freeze_benchmark.py",
    "scripts/freeze_forbidden_source_registry.py",
    "scripts/freeze_remote_environment.py",
    "scripts/freeze_sft_reserved_source_registry.py",
    "scripts/freeze_source_snapshot.py",
    "scripts/generate_clean_sft_replacements.py",
    "scripts/generate_clean_sft_replacements_a100.slurm",
    "scripts/generate_iid_benchmark.py",
    "scripts/prepare_sota_run.py",
    "scripts/run_reference_evidence.py",
    "scripts/run_jiang_f638_sft_in_sealed_environment.py",
    "scripts/seal_completed_sft_run.py",
    "scripts/smoke_jiang_f638_sft_environment_h100.slurm",
    "scripts/submit_sft500k_base_specialists_h100.sh",
    "scripts/train_clasp_sota_a100.slurm",
    "scripts/train_sft500k_base_specialist_h100.slurm",
    "scripts/validate_base_sft500k_specialist.py",
    "src/clasp/__init__.py",
    "src/clasp/data/benchmark.py",
    "src/clasp/data/loaders.py",
    "src/clasp/data/prompts.py",
    "src/clasp/entrypoints/evaluate.py",
    "src/clasp/entrypoints/train_sft_jiang_f638.py",
    "src/clasp/models/checkpoint_fidelity.py",
    "src/clasp/models/exact_peft_kl_reference.py",
    "src/clasp/models/jiang_f638_base.py",
    "src/clasp/models/loading.py",
    "src/clasp/models/publication_fidelity.py",
    "src/clasp/rewards/co_tasks.py",
    *STATIC_EXPERIMENT_FILES,
}


@dataclass(frozen=True)
class _ExpectedFile:
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    mode: int = 0o440


@dataclass(frozen=True)
class _ExpectedDirectory:
    device: int
    inode: int
    mode: int


def _stable_file(path: Path) -> _ExpectedFile:
    digest = hashlib.sha256()
    try:
        path_before = path.stat(follow_symlinks=False)
        with path.open("rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            descriptor_after = os.fstat(handle.fileno())
        path_after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot hash stable file snapshot {path}: {exc}") from exc
    identities = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    before = tuple(getattr(descriptor_before, name) for name in identities)
    if (
        before != tuple(getattr(descriptor_after, name) for name in identities)
        or before != tuple(getattr(path_before, name) for name in identities)
        or before != tuple(getattr(path_after, name) for name in identities)
    ):
        raise ValueError(f"file changed while snapshot was hashed: {path}")
    if not stat.S_ISREG(descriptor_after.st_mode):
        raise ValueError(f"snapshot entry is not a regular file: {path}")
    return _ExpectedFile(
        sha256=digest.hexdigest(),
        size_bytes=int(descriptor_after.st_size),
        device=int(descriptor_after.st_dev),
        inode=int(descriptor_after.st_ino),
        mtime_ns=int(descriptor_after.st_mtime_ns),
    )


def _stable_sha256(path: Path) -> tuple[str, int]:
    observed = _stable_file(path)
    return observed.sha256, observed.size_bytes


def _expected_directory(path: Path, *, mode: int) -> _ExpectedDirectory:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot stat staged directory {path}: {exc}") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"snapshot entry is not a directory: {path}")
    return _ExpectedDirectory(
        device=int(observed.st_dev),
        inode=int(observed.st_ino),
        mode=mode,
    )


def _forbidden_directory_component(name: str) -> bool:
    normalized = name.casefold()
    return (
        normalized in FORBIDDEN_DIRECTORY_COMPONENTS
        or normalized.startswith(FORBIDDEN_DIRECTORY_PREFIXES)
        or normalized.endswith(FORBIDDEN_DIRECTORY_SUFFIXES)
    )


def _validate_relative_entry(relative_path: Path, *, is_directory: bool) -> None:
    relative = relative_path.as_posix()
    if relative_path.parts[0].casefold() in FORBIDDEN_TOP_LEVEL:
        raise ValueError(f"source snapshot contains a forbidden artifact: {relative}")

    forbidden_components = [
        component
        for index, component in enumerate(relative_path.parts)
        if _forbidden_directory_component(component)
        and not (
            component.casefold() == "models"
            and tuple(part.casefold() for part in relative_path.parts[: index + 1])
            == ("src", "clasp", "models")
        )
    ]
    if is_directory and forbidden_components:
        raise ValueError(
            f"source snapshot contains a forbidden artifact directory component: {relative}"
        )

    if relative_path.parts[0].casefold() == "experiments":
        if is_directory:
            prefix = f"{relative}/"
            permitted = any(path.startswith(prefix) for path in STATIC_EXPERIMENT_FILES)
        else:
            permitted = relative in STATIC_EXPERIMENT_FILES
        if not permitted:
            raise ValueError(
                f"source snapshot experiments/ entry is not in the static allowlist: {relative}"
            )

    if is_directory:
        return
    normalized_name = relative_path.name.casefold()
    if (
        relative_path.suffix.casefold() in FORBIDDEN_FILE_SUFFIXES
        or normalized_name in FORBIDDEN_RUNTIME_FILE_NAMES
        or normalized_name.startswith(("raw_", "repaired_", "runtime_"))
    ):
        raise ValueError(f"source snapshot contains a forbidden artifact: {relative}")


def _enumerate(root: Path) -> tuple[list[tuple[str, Path]], list[Path]]:
    files: list[tuple[str, Path]] = []
    directories: list[Path] = []

    def visit(directory: Path, relative_parent: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"cannot enumerate staged directory {directory}: {exc}") from exc
        for entry in entries:
            candidate = directory / entry.name
            relative_path = relative_parent / entry.name
            relative = relative_path.as_posix()
            if entry.is_symlink():
                raise ValueError(f"source snapshot contains a symbolic link: {relative}")
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ValueError(f"cannot stat staged entry {candidate}: {exc}") from exc
            if stat.S_ISDIR(mode):
                if relative_path == Path("logs"):
                    try:
                        with os.scandir(candidate) as log_entries:
                            if next(log_entries, None) is not None:
                                raise ValueError("staging logs/ must be a regular empty directory")
                    except OSError as exc:
                        raise ValueError(
                            f"cannot enumerate staging logs directory {candidate}: {exc}"
                        ) from exc
                    continue
                _validate_relative_entry(relative_path, is_directory=True)
                directories.append(candidate)
                visit(candidate, relative_path)
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"source snapshot contains a special entry: {relative}")
            if relative == MANIFEST_NAME:
                raise ValueError(f"staging tree already contains {MANIFEST_NAME}")
            _validate_relative_entry(relative_path, is_directory=False)
            files.append((relative, candidate))

    visit(root, Path())
    files.sort(key=lambda item: item[0])
    directories.sort(key=lambda item: item.relative_to(root).as_posix())
    return files, directories


def _stat_identity(observed: os.stat_result) -> tuple[int, int]:
    return int(observed.st_dev), int(observed.st_ino)


def _stable_descriptor_stat(observed: os.stat_result) -> tuple[int, ...]:
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_nlink),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise ValueError(
            "platform cannot perform no-follow directory verification; missing: "
            + ", ".join(missing)
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform cannot perform no-follow file verification")
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


def _open_absolute_directory(path: Path) -> int:
    """Open an absolute directory one no-follow component at a time."""

    if not path.is_absolute():
        raise ValueError(f"directory path must be absolute: {path}")
    flags = _directory_open_flags()
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_canonical_directory_identity(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    try:
        descriptor = _open_absolute_directory(path)
    except OSError as exc:
        raise ValueError(f"cannot reopen canonical {label}: {exc}") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != expected_identity:
            raise ValueError(f"canonical {label} no longer names the pinned directory")
    finally:
        os.close(descriptor)


def _expected_children(
    *,
    files: dict[str, _ExpectedFile],
    directories: dict[str, _ExpectedDirectory],
) -> dict[str, dict[str, tuple[str, str]]]:
    if "" not in directories:
        raise ValueError("sealed-tree expectation has no root directory")
    children: dict[str, dict[str, tuple[str, str]]] = {relative: {} for relative in directories}
    for entry_type, relative_paths in (
        ("directory", (path for path in directories if path)),
        ("file", iter(files)),
    ):
        for relative in relative_paths:
            relative_path = Path(relative)
            parent = relative_path.parent.as_posix()
            if parent == ".":
                parent = ""
            if parent not in children:
                raise ValueError(f"sealed-tree expectation has no parent directory for {relative}")
            name = relative_path.name
            if name in children[parent]:
                raise ValueError(f"duplicate sealed-tree expectation: {relative}")
            children[parent][name] = (entry_type, relative)
    return children


def _assert_directory(
    observed: os.stat_result,
    expected: _ExpectedDirectory,
    *,
    relative: str,
) -> None:
    label = relative or "."
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"sealed snapshot entry changed type: {label}")
    if _stat_identity(observed) != (expected.device, expected.inode):
        raise ValueError(f"sealed snapshot directory was replaced: {label}")
    observed_mode = stat.S_IMODE(observed.st_mode)
    if observed_mode != expected.mode:
        raise ValueError(
            f"sealed snapshot directory mode changed: {label} "
            f"(expected {expected.mode:04o}, observed {observed_mode:04o})"
        )


def _assert_file_metadata(
    observed: os.stat_result,
    expected: _ExpectedFile,
    *,
    relative: str,
) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"sealed snapshot entry changed type: {relative}")
    if _stat_identity(observed) != (expected.device, expected.inode):
        raise ValueError(f"sealed snapshot file was replaced: {relative}")
    observed_mode = stat.S_IMODE(observed.st_mode)
    if observed_mode != expected.mode:
        raise ValueError(
            f"sealed snapshot file mode changed: {relative} "
            f"(expected {expected.mode:04o}, observed {observed_mode:04o})"
        )
    if observed.st_nlink != 1:
        raise ValueError(
            f"sealed snapshot file has forbidden hard-link count: {relative} "
            f"(observed {observed.st_nlink})"
        )
    if int(observed.st_mtime_ns) != expected.mtime_ns:
        raise ValueError(f"sealed snapshot file timestamp changed: {relative}")


def _hash_open_file(
    descriptor: int,
    *,
    before: os.stat_result,
    expected: _ExpectedFile,
    relative: str,
) -> None:
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"cannot rehash sealed snapshot file {relative}: {exc}") from exc
    if _stable_descriptor_stat(before) != _stable_descriptor_stat(after):
        raise ValueError(f"sealed snapshot file changed while rehashed: {relative}")
    _assert_file_metadata(after, expected, relative=relative)
    if int(after.st_size) != expected.size_bytes or digest.hexdigest() != expected.sha256:
        raise ValueError(f"sealed snapshot file content changed: {relative}")


def _sealed_tree_sha256(
    *,
    files: dict[str, _ExpectedFile],
    directories: dict[str, _ExpectedDirectory],
) -> str:
    entries: list[dict[str, object]] = []
    for relative, expected in sorted(directories.items()):
        entries.append(
            {
                "mode": f"{expected.mode:04o}",
                "path": relative or ".",
                "type": "directory",
            }
        )
    for relative, expected in sorted(files.items()):
        entries.append(
            {
                "mode": f"{expected.mode:04o}",
                "path": relative,
                "sha256": expected.sha256,
                "size_bytes": expected.size_bytes,
                "type": "regular_file",
            }
        )
    return hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()


def _verify_sealed_tree(
    *,
    root_descriptor: int,
    files: dict[str, _ExpectedFile],
    directories: dict[str, _ExpectedDirectory],
) -> str:
    """Verify one exact sealed tree using only pinned, no-follow descriptors."""

    children = _expected_children(files=files, directories=directories)
    directory_flags = _directory_open_flags()
    file_flags = _file_open_flags()

    def visit(directory_descriptor: int, relative: str) -> None:
        expected_directory = directories[relative]
        try:
            directory_before = os.fstat(directory_descriptor)
        except OSError as exc:
            raise ValueError(
                f"cannot stat sealed snapshot directory {relative or '.'}: {exc}"
            ) from exc
        _assert_directory(directory_before, expected_directory, relative=relative)
        try:
            with os.scandir(directory_descriptor) as iterator:
                observed_names = sorted(entry.name for entry in iterator)
        except OSError as exc:
            raise ValueError(
                f"cannot enumerate sealed snapshot directory {relative or '.'}: {exc}"
            ) from exc
        expected_names = sorted(children[relative])
        if observed_names != expected_names:
            missing = sorted(set(expected_names) - set(observed_names))
            extra = sorted(set(observed_names) - set(expected_names))
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise ValueError(
                f"sealed snapshot directory entry set changed: {relative or '.'} "
                f"({' '.join(details)})"
            )

        for name in expected_names:
            entry_type, child_relative = children[relative][name]
            try:
                path_stat = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    f"cannot no-follow stat sealed snapshot entry {child_relative}: {exc}"
                ) from exc
            if entry_type == "directory":
                expected_child = directories[child_relative]
                _assert_directory(path_stat, expected_child, relative=child_relative)
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    raise ValueError(
                        f"cannot no-follow open sealed snapshot directory {child_relative}: {exc}"
                    ) from exc
                try:
                    descriptor_stat = os.fstat(child_descriptor)
                    if _stable_descriptor_stat(path_stat) != _stable_descriptor_stat(
                        descriptor_stat
                    ):
                        raise ValueError(
                            f"sealed snapshot directory changed while opened: {child_relative}"
                        )
                    visit(child_descriptor, child_relative)
                finally:
                    os.close(child_descriptor)
                continue

            expected_file = files[child_relative]
            _assert_file_metadata(path_stat, expected_file, relative=child_relative)
            try:
                file_descriptor = os.open(
                    name,
                    file_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ValueError(
                    f"cannot no-follow open sealed snapshot file {child_relative}: {exc}"
                ) from exc
            try:
                descriptor_stat = os.fstat(file_descriptor)
                if _stable_descriptor_stat(path_stat) != _stable_descriptor_stat(descriptor_stat):
                    raise ValueError(f"sealed snapshot file changed while opened: {child_relative}")
                _assert_file_metadata(
                    descriptor_stat,
                    expected_file,
                    relative=child_relative,
                )
                _hash_open_file(
                    file_descriptor,
                    before=descriptor_stat,
                    expected=expected_file,
                    relative=child_relative,
                )
            finally:
                os.close(file_descriptor)

        try:
            directory_after = os.fstat(directory_descriptor)
        except OSError as exc:
            raise ValueError(
                f"cannot restat sealed snapshot directory {relative or '.'}: {exc}"
            ) from exc
        if _stable_descriptor_stat(directory_before) != _stable_descriptor_stat(directory_after):
            raise ValueError(f"sealed snapshot directory changed while verified: {relative or '.'}")
        _assert_directory(directory_after, expected_directory, relative=relative)

    visit(root_descriptor, "")
    return _sealed_tree_sha256(files=files, directories=directories)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _remove_created_artifact_at(
    *,
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only the link created by this transaction, never a replacement."""

    try:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect failed transaction artifact {name}: {exc}") from exc
    if _stat_identity(observed) != expected_identity:
        raise ValueError(f"refusing to remove replaced transaction artifact: {name}")
    try:
        os.unlink(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise ValueError(f"cannot remove failed transaction artifact {name}: {exc}") from exc


def _write_atomic_new_at(
    *,
    parent_descriptor: int,
    name: str,
    content: bytes,
    mode: int,
) -> tuple[int, int]:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"artifact name must be one safe path component: {name!r}")
    open_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary_name = ""
    descriptor = -1
    for _ in range(128):
        candidate = f".{name}.{secrets.token_hex(12)}"
        try:
            descriptor = os.open(
                candidate,
                open_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if descriptor < 0:
        raise ValueError(f"cannot allocate unique temporary artifact for {name}")

    created_identity: tuple[int, int] | None = None
    linked = False
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:  # pragma: no cover - regular-file write contract
                raise OSError("zero-byte write while publishing artifact")
            remaining = remaining[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        created_identity = _stat_identity(os.fstat(descriptor))
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError(f"refusing to replace existing artifact: {name}") from exc
        linked = True
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = ""
        os.fsync(parent_descriptor)
        return created_identity
    except Exception:
        if linked and created_identity is not None:
            _remove_created_artifact_at(
                parent_descriptor=parent_descriptor,
                name=name,
                expected_identity=created_identity,
            )
        raise
    finally:
        os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def _write_atomic_new(path: Path, content: bytes, *, mode: int) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_descriptor = _open_absolute_directory(path.parent)
    except OSError as exc:
        raise ValueError(f"cannot pin artifact parent directory {path.parent}: {exc}") from exc
    try:
        return _write_atomic_new_at(
            parent_descriptor=parent_descriptor,
            name=path.name,
            content=content,
            mode=mode,
        )
    finally:
        os.close(parent_descriptor)


def _remove_created_artifact(path: Path, expected_identity: tuple[int, int]) -> None:
    """Path wrapper for cleanup outside an already pinned transaction."""

    try:
        parent_descriptor = _open_absolute_directory(path.parent)
    except OSError as exc:
        raise ValueError(f"cannot pin failed artifact parent {path.parent}: {exc}") from exc
    try:
        _remove_created_artifact_at(
            parent_descriptor=parent_descriptor,
            name=path.name,
            expected_identity=expected_identity,
        )
    finally:
        os.close(parent_descriptor)


def _renameat2_noreplace_errno(
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> int | None:
    """Return ``None`` on renameat2 success, otherwise its positive errno."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    return None if result == 0 else ctypes.get_errno()


def _remove_reservation_at(
    *,
    parent_descriptor: int,
    destination_name: str,
    reservation_identity: tuple[int, int],
) -> None:
    """Remove only the still-empty directory inode created by this transaction."""

    try:
        observed = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(observed.st_mode) or _stat_identity(observed) != reservation_identity:
        raise ValueError(
            "refusing to clean a replaced atomic-publication reservation: "
            f"{destination_name}"
        )
    try:
        os.rmdir(destination_name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError(
            f"cannot remove failed atomic-publication reservation {destination_name}: {exc}"
        ) from exc
    os.fsync(parent_descriptor)


def _rename_with_empty_directory_reservation_at(
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Portable directory publication after an exclusive empty-dir reservation.

    ``mkdirat`` atomically fails when the destination already exists.  The
    reservation inode is pinned, checked again immediately before ``rename``,
    and removed on failure only while the destination still names that inode.
    Both current callers publish sealed directories below an already pinned,
    owner-controlled parent.
    """

    source_descriptor = os.open(
        source_name,
        _directory_open_flags(),
        dir_fd=parent_descriptor,
    )
    reservation_descriptor: int | None = None
    reservation_identity: tuple[int, int] | None = None
    published = False
    try:
        source_identity = _stat_identity(os.fstat(source_descriptor))
        try:
            os.mkdir(destination_name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise ValueError(
                f"cannot atomically publish snapshot {source_name} -> {destination_name}: "
                "destination already exists"
            ) from exc
        reservation_stat = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(reservation_stat.st_mode):  # pragma: no cover - mkdir contract
            raise ValueError("atomic-publication reservation is not a directory")
        reservation_identity = _stat_identity(reservation_stat)
        reservation_descriptor = os.open(
            destination_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        if _stat_identity(os.fstat(reservation_descriptor)) != reservation_identity:
            raise ValueError("atomic-publication reservation changed while it was pinned")
        with os.scandir(reservation_descriptor) as entries:
            if next(entries, None) is not None:  # pragma: no cover - fresh mkdir contract
                raise ValueError("atomic-publication reservation is not empty")
        os.fsync(parent_descriptor)

        immediately_before = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(immediately_before.st_mode)
            or _stat_identity(immediately_before) != reservation_identity
        ):
            raise ValueError("atomic-publication reservation changed before rename")
        try:
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot atomically publish snapshot {source_name} -> "
                f"{destination_name}: {exc}"
            ) from exc
        published = True
        os.fsync(parent_descriptor)
        published_stat = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(published_stat.st_mode) or _stat_identity(
            published_stat
        ) != source_identity:
            raise ValueError("portable atomic publication did not install the pinned source")
    finally:
        if not published and reservation_identity is not None:
            _remove_reservation_at(
                parent_descriptor=parent_descriptor,
                destination_name=destination_name,
                reservation_identity=reservation_identity,
            )
        if reservation_descriptor is not None:
            os.close(reservation_descriptor)
        os.close(source_descriptor)


def _rename_noreplace_at(
    *,
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Publish sibling directories without replacing a pre-existing name."""

    error_number = _renameat2_noreplace_errno(
        parent_descriptor=parent_descriptor,
        source_name=source_name,
        destination_name=destination_name,
    )
    if error_number is None:
        return
    unsupported = {
        errno.ENOSYS,
        errno.EINVAL,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
    if error_number not in unsupported:
        raise ValueError(
            f"cannot atomically publish snapshot {source_name} -> {destination_name}: "
            f"{os.strerror(error_number)}"
        )
    _rename_with_empty_directory_reservation_at(
        parent_descriptor=parent_descriptor,
        source_name=source_name,
        destination_name=destination_name,
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Compatibility wrapper for sibling-path callers."""

    if source.parent != destination.parent:
        raise ValueError("atomic source and destination must be siblings")
    parent_descriptor = os.open(source.parent, _directory_open_flags())
    try:
        _rename_noreplace_at(
            parent_descriptor=parent_descriptor,
            source_name=source.name,
            destination_name=destination.name,
        )
    finally:
        os.close(parent_descriptor)


def _verify_canonical_published_tree(
    *,
    final: Path,
    parent_identity: tuple[int, int],
    root_identity: tuple[int, int],
    files: dict[str, _ExpectedFile],
    directories: dict[str, _ExpectedDirectory],
) -> str:
    """Reopen the canonical parent and final name before trusting publication."""

    try:
        parent_descriptor = _open_absolute_directory(final.parent)
    except OSError as exc:
        raise ValueError(f"cannot reopen canonical snapshot parent: {exc}") from exc
    try:
        parent_stat = os.fstat(parent_descriptor)
        if _stat_identity(parent_stat) != parent_identity:
            raise ValueError("canonical snapshot parent no longer names the pinned directory")
        try:
            final_descriptor = os.open(
                final.name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ValueError(f"cannot reopen canonical published snapshot: {exc}") from exc
        try:
            final_stat = os.fstat(final_descriptor)
            if _stat_identity(final_stat) != root_identity:
                raise ValueError("canonical published snapshot no longer names the verified root")
            tree_sha256 = _verify_sealed_tree(
                root_descriptor=final_descriptor,
                files=files,
                directories=directories,
            )
        finally:
            os.close(final_descriptor)
    finally:
        os.close(parent_descriptor)
    try:
        parent_descriptor = _open_absolute_directory(final.parent)
    except OSError as exc:
        raise ValueError(f"cannot restabilize canonical snapshot parent: {exc}") from exc
    try:
        if _stat_identity(os.fstat(parent_descriptor)) != parent_identity:
            raise ValueError("canonical snapshot parent changed during tree verification")
        try:
            final_descriptor = os.open(
                final.name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ValueError(f"cannot restabilize canonical published snapshot: {exc}") from exc
        try:
            if _stat_identity(os.fstat(final_descriptor)) != root_identity:
                raise ValueError("canonical published snapshot changed during tree verification")
        finally:
            os.close(final_descriptor)
    finally:
        os.close(parent_descriptor)
    return tree_sha256


def freeze(
    *,
    staging_root: Path,
    final_root: Path,
    receipt_path: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    raw_staging = staging_root.expanduser().absolute()
    staging = raw_staging.resolve()
    final = final_root.expanduser().absolute()
    receipt = receipt_path.expanduser().absolute()
    if raw_staging.is_symlink() or not staging.is_dir():
        raise ValueError(f"staging root is not a regular directory: {raw_staging}")
    if final.exists() or final.is_symlink():
        raise ValueError(f"refusing to replace existing final snapshot: {final}")
    if receipt.exists() or receipt.is_symlink():
        raise ValueError(f"refusing to replace existing receipt: {receipt}")
    if staging == final or staging.parent != final.parent:
        raise ValueError("staging and final snapshot must be distinct siblings")
    if not snapshot_id or any(character.isspace() for character in snapshot_id):
        raise ValueError("snapshot_id must be a nonempty token without whitespace")
    try:
        receipt.relative_to(staging)
    except ValueError:
        pass
    else:
        raise ValueError("receipt must be outside the staging/final snapshot")
    try:
        receipt.relative_to(final)
    except ValueError:
        pass
    else:
        raise ValueError("receipt must be outside the staging/final snapshot")

    logs = staging / "logs"
    logs.mkdir(mode=0o770, exist_ok=True)
    if logs.is_symlink() or not logs.is_dir() or any(logs.iterdir()):
        raise ValueError("staging logs/ must be a regular empty directory")

    files, directories = _enumerate(staging)
    relative_files = [relative for relative, _ in files]
    missing_required = sorted(REQUIRED_FILES - set(relative_files))
    if missing_required:
        raise ValueError("staging source is incomplete; missing: " + ", ".join(missing_required))

    required_source_directories: set[str] = set()
    for relative in relative_files:
        parent = Path(relative).parent
        while parent != Path("."):
            required_source_directories.add(parent.as_posix())
            parent = parent.parent
    observed_source_directories = {
        directory.relative_to(staging).as_posix() for directory in directories
    }
    if observed_source_directories != required_source_directories:
        missing_directories = sorted(required_source_directories - observed_source_directories)
        extra_directories = sorted(observed_source_directories - required_source_directories)
        details = []
        if missing_directories:
            details.append("missing=" + ",".join(missing_directories))
        if extra_directories:
            details.append("extra=" + ",".join(extra_directories))
        raise ValueError(
            "staging source contains an unmanifested directory entry set: " + " ".join(details)
        )

    expected_directories: dict[str, _ExpectedDirectory] = {
        "": _expected_directory(staging, mode=0o2550),
        "logs": _expected_directory(logs, mode=0o770),
    }
    for directory in directories:
        relative = directory.relative_to(staging).as_posix()
        expected_directories[relative] = _expected_directory(directory, mode=0o550)

    receipts: list[dict[str, Any]] = []
    expected_files: dict[str, _ExpectedFile] = {}
    manifest_lines: list[str] = []
    aggregate = hashlib.sha256()
    total_size = 0
    for relative, path in files:
        expected = _stable_file(path)
        expected_files[relative] = expected
        manifest_lines.append(f"{expected.sha256}  {relative}\n")
        receipts.append(
            {
                "path": relative,
                "size_bytes": expected.size_bytes,
                "sha256": expected.sha256,
            }
        )
        total_size += expected.size_bytes
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(expected.size_bytes).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(expected.sha256.encode("ascii"))
        aggregate.update(b"\n")
    observed_files, observed_directories = _enumerate(staging)
    if observed_files != files or observed_directories != directories:
        raise ValueError("staging entry set changed while snapshot was hashed")

    manifest = staging / MANIFEST_NAME
    manifest_content = "".join(manifest_lines).encode("utf-8")
    _write_atomic_new(manifest, manifest_content, mode=0o440)
    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    expected_manifest = _stable_file(manifest)
    if expected_manifest.sha256 != manifest_sha256 or expected_manifest.size_bytes != len(
        manifest_content
    ):
        raise ValueError("source manifest changed immediately after creation")
    expected_files[MANIFEST_NAME] = expected_manifest

    try:
        for _, path in files:
            os.chmod(path, 0o440)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            os.chmod(directory, 0o550)
        os.chmod(logs, 0o770)
        os.chmod(staging, 0o2550)
    except OSError as exc:
        raise ValueError(f"source changed while sealed permissions were applied: {exc}") from exc

    directory_flags = _directory_open_flags()
    try:
        parent_descriptor = _open_absolute_directory(staging.parent)
    except OSError as exc:
        raise ValueError(f"cannot pin snapshot parent directory: {exc}") from exc
    try:
        parent_identity = _stat_identity(os.fstat(parent_descriptor))
        _require_canonical_directory_identity(
            staging.parent,
            parent_identity,
            label="snapshot parent",
        )
        try:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt_parent_descriptor = _open_absolute_directory(receipt.parent)
        except OSError as exc:
            raise ValueError(f"cannot pin receipt parent directory: {exc}") from exc
        try:
            try:
                staging_descriptor = os.open(
                    staging.name,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise ValueError(f"cannot pin sealed staging root: {exc}") from exc
            try:
                root_identity = _stat_identity(os.fstat(staging_descriptor))
                sealed_tree_sha256 = _verify_sealed_tree(
                    root_descriptor=staging_descriptor,
                    files=expected_files,
                    directories=expected_directories,
                )

                created_at = (
                    datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                payload: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "immutable_source_snapshot",
                    "snapshot_id": snapshot_id,
                    "created_at_utc": created_at,
                    "cluster": socket.getfqdn(),
                    "path": str(final),
                    "source_parent_identity": {
                        "path": str(staging.parent),
                        "device": parent_identity[0],
                        "inode": parent_identity[1],
                    },
                    "source_root_identity": {
                        "device": root_identity[0],
                        "inode": root_identity[1],
                    },
                    "source_manifest_sha256": manifest_sha256,
                    "source_tree_sha256": aggregate.hexdigest(),
                    "sealed_tree_protocol": "clasp-source-sealed-tree-v1",
                    "sealed_tree_sha256": sealed_tree_sha256,
                    "file_count_excluding_manifest": len(receipts),
                    "file_count_including_manifest": len(receipts) + 1,
                    "size_bytes_excluding_manifest": total_size,
                    "permissions": {
                        "snapshot_root": "2550",
                        "source_directories": "0550",
                        "source_files": "0440",
                        "logs_directory": "0770",
                    },
                }
                payload["receipt_sha256"] = hashlib.sha256(
                    _canonical_json_bytes(payload)
                ).hexdigest()
                receipt_content = (
                    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
                ).encode("utf-8")

                immediately_before_rename = _verify_sealed_tree(
                    root_descriptor=staging_descriptor,
                    files=expected_files,
                    directories=expected_directories,
                )
                if immediately_before_rename != sealed_tree_sha256:
                    raise ValueError("sealed source tree binding changed before publication")
                _require_canonical_directory_identity(
                    staging.parent,
                    parent_identity,
                    label="snapshot parent before rename",
                )
                _rename_noreplace_at(
                    parent_descriptor=parent_descriptor,
                    source_name=staging.name,
                    destination_name=final.name,
                )

                try:
                    final_descriptor = os.open(
                        final.name,
                        directory_flags,
                        dir_fd=parent_descriptor,
                    )
                except OSError as exc:
                    raise ValueError(
                        f"snapshot was published, but final root cannot be pinned: {exc}"
                    ) from exc
                try:
                    if _stat_identity(os.fstat(final_descriptor)) != root_identity:
                        raise ValueError(
                            "published final snapshot is not the verified staging root"
                        )
                    published_tree_sha256 = _verify_sealed_tree(
                        root_descriptor=final_descriptor,
                        files=expected_files,
                        directories=expected_directories,
                    )
                    if published_tree_sha256 != sealed_tree_sha256:
                        raise ValueError("published source tree binding changed after rename")
                    canonical_tree_sha256 = _verify_canonical_published_tree(
                        final=final,
                        parent_identity=parent_identity,
                        root_identity=root_identity,
                        files=expected_files,
                        directories=expected_directories,
                    )
                    if canonical_tree_sha256 != sealed_tree_sha256:
                        raise ValueError("canonical source tree binding changed before receipt")

                    try:
                        receipt_identity = _write_atomic_new_at(
                            parent_descriptor=receipt_parent_descriptor,
                            name=receipt.name,
                            content=receipt_content,
                            mode=0o440,
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"snapshot was published at {final}, but its receipt could not be "
                            f"created: {exc}"
                        ) from exc
                    try:
                        _require_canonical_directory_identity(
                            receipt.parent,
                            _stat_identity(os.fstat(receipt_parent_descriptor)),
                            label="receipt parent after publication",
                        )
                        after_receipt_sha256 = _verify_canonical_published_tree(
                            final=final,
                            parent_identity=parent_identity,
                            root_identity=root_identity,
                            files=expected_files,
                            directories=expected_directories,
                        )
                        if after_receipt_sha256 != sealed_tree_sha256:
                            raise ValueError(
                                "canonical source tree binding changed during receipt publication"
                            )
                    except Exception as verification_exc:
                        try:
                            _remove_created_artifact_at(
                                parent_descriptor=receipt_parent_descriptor,
                                name=receipt.name,
                                expected_identity=receipt_identity,
                            )
                        except Exception as cleanup_exc:
                            raise ValueError(
                                "publication verification failed and the transaction receipt "
                                f"could not be removed: {cleanup_exc}"
                            ) from verification_exc
                        raise ValueError(
                            "publication changed while its receipt was created; receipt removed"
                        ) from verification_exc
                finally:
                    os.close(final_descriptor)
            finally:
                os.close(staging_descriptor)
        finally:
            os.close(receipt_parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = freeze(
            staging_root=args.staging_root,
            final_root=args.final_root,
            receipt_path=args.receipt,
            snapshot_id=args.snapshot_id,
        )
    except ValueError as exc:
        print(f"[SourceSnapshotError] {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
