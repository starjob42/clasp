#!/usr/bin/env python3
"""Freeze and live-verify the Jiang-f638 specialist-SFT environment.

The receipt is produced from the single H100 allocation used by the worker. It
binds the byte-identical dependency lock released at Jiang et al. commit f638,
the complete non-writable Python prefix, every visible installed distribution,
the Python/Torch runtime, and the live H100 identity. Every worker allocation
revalidates those identities immediately before training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4
PROTOCOL = "clasp-jiang-f638-sealed-sft-environment-v4"
STATUS = "frozen_before_publication_training"
CLAIM_SCOPE = "publication_base_specialist_sft"
PRODUCER_PROTOCOL = "clasp-jiang-f638-environment-producer-v4"
EXPECTED_GPU_COUNT = 1
EXPECTED_GPU_FAMILY = "H100"
EXPECTED_COMPUTE_CAPABILITY = (9, 0)
JIANG_F638_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
REQUIREMENTS_LOCK_RELATIVE_PATH = "requirements.jiang-f638-sft.txt"
REQUIREMENTS_LOCK_PATH = Path(__file__).resolve().parents[1] / REQUIREMENTS_LOCK_RELATIVE_PATH
RUNTIME_CLOSURE_LOCK_RELATIVE_PATH = "requirements.jiang-f638-runtime-closure.txt"
RUNTIME_CLOSURE_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / RUNTIME_CLOSURE_LOCK_RELATIVE_PATH
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
PACKAGE_INVENTORY_SCOPE = (
    "exact-released-lock-plus-receipt-proven-runtime-closure-and-pinned-pip-"
    "under-sys-prefix-v3"
)
# Jiang's released ``pip freeze`` is not an executable closure by itself.  The
# historical cluster runtime receipt at
# ``jiang_reproduction/envs/llmcosolver_f6388991_py311_plus_transitives_v1/``
# proves these eight installed dependencies; notably, datasets cannot import
# its native ``_cffi_backend`` without cffi/pycparser.  They are declared
# separately instead of being misrepresented as released lock rows.
# ``pip freeze`` also omits pip itself.  It remains the sole construction
# bootstrap exception.  Every other extra visible distribution is fatal.
BOOTSTRAP_DISTRIBUTION_VERSIONS = {"pip": "25.1.1"}
# The sealed conda-forge pip build has a deterministic set of 412 stale RECORD
# rows: one conda-rewritten console entry point and 411 CPython-3.12 bytecode
# cache paths.  The individual path shape is only an initial filter; count,
# class counts, and the canonical sorted-path digest bind the exact complete
# set.  No other distribution, version, or missing RECORD path is tolerated.
PIP_STALE_RECORD_VERSION = "25.1.1"
PIP_STALE_CONSOLE_RECORD_PATH = "../../../bin/pip3.12"
PIP_STALE_MISSING_RECORD_COUNT = 412
PIP_STALE_MISSING_PYC_RECORD_COUNT = 411
PIP_STALE_MISSING_PATHS_SHA256 = (
    "bb087d4ccd6bcafbcf2930356ef97aa0df66039675beb73ae0da8cf52cf0c8f3"
)
PIP_STALE_PYC_RECORD_RE = re.compile(
    r"^pip/(?:[^/]+/)*__pycache__/[^/]+\.cpython-312\.pyc$"
)


def _read_released_requirements_lock() -> tuple[dict[str, str], str]:
    try:
        raw = REQUIREMENTS_LOCK_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"missing Jiang-f638 dependency lock: {REQUIREMENTS_LOCK_PATH}"
        ) from exc
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(
                f"non-exact requirement at {REQUIREMENTS_LOCK_PATH}:{line_number}: {line!r}"
            )
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if not normalized or not version or normalized in expected:
            raise RuntimeError(
                f"invalid or duplicate requirement at line {line_number}: {line!r}"
            )
        expected[normalized] = version
    if not expected:
        raise RuntimeError("Jiang-f638 dependency lock is empty")
    return dict(sorted(expected.items())), hashlib.sha256(raw).hexdigest()


EXPECTED_DISTRIBUTION_VERSIONS, REQUIREMENTS_LOCK_SHA256 = (
    _read_released_requirements_lock()
)


def _read_runtime_closure_lock() -> tuple[dict[str, str], str]:
    try:
        raw = RUNTIME_CLOSURE_LOCK_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"missing Jiang-f638 runtime-closure lock: {RUNTIME_CLOSURE_LOCK_PATH}"
        ) from exc
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(
                f"non-exact runtime requirement at {RUNTIME_CLOSURE_LOCK_PATH}:"
                f"{line_number}: {line!r}"
            )
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if not normalized or not version or normalized in expected:
            raise RuntimeError(f"invalid runtime-closure requirement: {line!r}")
        expected[normalized] = version
    if len(expected) != 8:
        raise RuntimeError("Jiang-f638 runtime-closure lock must contain exactly 8 pins")
    return dict(sorted(expected.items())), hashlib.sha256(raw).hexdigest()


RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS, RUNTIME_CLOSURE_LOCK_SHA256 = (
    _read_runtime_closure_lock()
)
if set(EXPECTED_DISTRIBUTION_VERSIONS) & set(RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS):
    raise RuntimeError("runtime-closure distributions overlap the released Jiang-f638 lock")
if (
    set(EXPECTED_DISTRIBUTION_VERSIONS) | set(RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS)
) & set(BOOTSTRAP_DISTRIBUTION_VERSIONS):
    raise RuntimeError("bootstrap distributions overlap the Jiang-f638 executable closure")
EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS = dict(
    sorted(
        {
            **EXPECTED_DISTRIBUTION_VERSIONS,
            **RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS,
            **BOOTSTRAP_DISTRIBUTION_VERSIONS,
        }.items()
    )
)
REQUIRED_DISTRIBUTIONS = tuple(EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_plain_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_optional_string(value: object) -> bool:
    return value is None or isinstance(value, str)


def _is_absolute_normalized_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path.is_absolute() and ".." not in path.parts


def _validate_timestamp(value: object) -> bool:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        return False
    try:
        return (
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M:%SZ") == value
        )
    except ValueError:
        return False


def _require_not_writable(path: Path, *, label: str) -> None:
    try:
        writable = os.access(path, os.W_OK, effective_ids=True)
    except TypeError:  # pragma: no cover - effective_ids is available on Linux
        writable = os.access(path, os.W_OK)
    if writable:
        raise ValueError(f"{label} is writable by the publication-training identity: {path}")


_STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, field)) for field in _STAT_FIELDS)


def _stable_file_identity(path: Path) -> tuple[str, int, int, tuple[int, ...]]:
    digest = hashlib.sha256()
    try:
        before_path = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode):
            raise ValueError(f"environment entry is not a regular file: {path}")
        with path.open("rb") as handle:
            before_fd = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after_fd = os.fstat(handle.fileno())
        after_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot hash stable environment file {path}: {exc}") from exc
    identities = [
        _stat_token(item)
        for item in (
            before_path,
            before_fd,
            after_fd,
            after_path,
        )
    ]
    if len(set(identities)) != 1:
        raise ValueError(f"environment file changed while it was hashed: {path}")
    return (
        digest.hexdigest(),
        int(after_path.st_size),
        stat.S_IMODE(after_path.st_mode),
        identities[-1],
    )


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _is_pip_stale_pyc_record_path(metadata_path: str) -> bool:
    parts = metadata_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return PIP_STALE_PYC_RECORD_RE.fullmatch(metadata_path) is not None


def _is_allowlisted_missing_distribution_row(
    *,
    normalized_name: str,
    version: object,
    metadata_path: str,
) -> bool:
    if normalized_name != "pip" or type(version) is not str:
        return False
    if version != PIP_STALE_RECORD_VERSION:
        return False
    return (
        metadata_path == PIP_STALE_CONSOLE_RECORD_PATH
        or _is_pip_stale_pyc_record_path(metadata_path)
    )


def _missing_metadata_paths_sha256(metadata_paths: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(metadata_paths),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_pip_stale_missing_metadata_paths(metadata_paths: Sequence[str]) -> None:
    paths = list(metadata_paths)
    console_count = paths.count(PIP_STALE_CONSOLE_RECORD_PATH)
    pyc_count = sum(_is_pip_stale_pyc_record_path(path) for path in paths)
    other = sorted(
        path
        for path in paths
        if path != PIP_STALE_CONSOLE_RECORD_PATH
        and not _is_pip_stale_pyc_record_path(path)
    )
    observed_digest = _missing_metadata_paths_sha256(paths)
    if (
        len(paths) != PIP_STALE_MISSING_RECORD_COUNT
        or console_count != 1
        or pyc_count != PIP_STALE_MISSING_PYC_RECORD_COUNT
        or other
        or observed_digest != PIP_STALE_MISSING_PATHS_SHA256
    ):
        raise ValueError(
            "pip 25.1.1 stale missing RECORD invariant failed: "
            + json.dumps(
                {
                    "expected_count": PIP_STALE_MISSING_RECORD_COUNT,
                    "observed_count": len(paths),
                    "expected_console_count": 1,
                    "observed_console_count": console_count,
                    "expected_pyc_count": PIP_STALE_MISSING_PYC_RECORD_COUNT,
                    "observed_pyc_count": pyc_count,
                    "other": other,
                    "expected_paths_sha256": PIP_STALE_MISSING_PATHS_SHA256,
                    "observed_paths_sha256": observed_digest,
                },
                sort_keys=True,
            )
        )


def _sealed_prefix_tree(prefix: Path) -> dict[str, Any]:
    prefix = prefix.resolve(strict=True)
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError(f"active environment prefix is not a regular directory: {prefix}")
    _require_not_writable(prefix.parent, label="active environment parent")
    entries = [prefix, *sorted(prefix.rglob("*"), key=lambda item: item.as_posix())]
    files: list[tuple[str, Path]] = []
    directory_records: list[dict[str, object]] = []
    directory_tokens: dict[str, tuple[int, ...]] = {}
    symlink_records: list[dict[str, str]] = []
    symlink_tokens: dict[str, tuple[int, ...]] = {}
    for candidate in entries:
        try:
            observed = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"cannot inspect active environment entry: {candidate}") from exc
        relative = "." if candidate == prefix else candidate.relative_to(prefix).as_posix()
        if stat.S_ISLNK(observed.st_mode):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(prefix)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"active environment symlink escapes the sealed prefix: {candidate}"
                ) from exc
            # Conda environments legitimately contain internal directory
            # aliases (for example ``lib/python3.1 -> python3.10``).  The link
            # itself is lstat-bound below and the resolved target tree is
            # independently hashed at its canonical in-prefix location, so an
            # internal regular-directory target is as auditable as a regular
            # file target.  Broken links, escapes, and special targets remain
            # fail-closed.
            if not (resolved.is_file() or resolved.is_dir()):
                raise ValueError(
                    f"active environment contains a special symlink target: {candidate}"
                )
            symlink_records.append({"path": relative, "target": os.readlink(candidate)})
            symlink_tokens[relative] = _stat_token(observed)
        elif stat.S_ISDIR(observed.st_mode):
            _require_not_writable(candidate, label="active environment directory")
            directory_records.append({"path": relative, "mode": stat.S_IMODE(observed.st_mode)})
            directory_tokens[relative] = _stat_token(observed)
        elif stat.S_ISREG(observed.st_mode):
            _require_not_writable(candidate, label="active environment file")
            files.append((relative, candidate))
        else:
            raise ValueError(f"active environment contains a special entry: {candidate}")
    if not files:
        raise ValueError("active environment prefix contains no files")

    aggregate = hashlib.sha256()
    total_size = 0
    final_tokens: dict[str, tuple[int, ...]] = {}
    for relative, candidate in files:
        digest, size, mode, token = _stable_file_identity(candidate)
        total_size += size
        final_tokens[relative] = token
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(mode).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    for relative, candidate in files:
        try:
            token = _stat_token(candidate.stat(follow_symlinks=False))
        except OSError as exc:
            raise ValueError(
                f"active environment file disappeared during audit: {candidate}"
            ) from exc
        if token != final_tokens[relative]:
            raise ValueError(f"active environment file changed during audit: {candidate}")
    for relative, expected_token in directory_tokens.items():
        candidate = prefix if relative == "." else prefix / relative
        try:
            token = _stat_token(candidate.stat(follow_symlinks=False))
        except OSError as exc:
            raise ValueError(
                f"active environment directory disappeared during audit: {candidate}"
            ) from exc
        if token != expected_token:
            raise ValueError(f"active environment directory changed during audit: {candidate}")
    for relative, expected_token in symlink_tokens.items():
        candidate = prefix / relative
        try:
            token = _stat_token(candidate.stat(follow_symlinks=False))
        except OSError as exc:
            raise ValueError(
                f"active environment symlink disappeared during audit: {candidate}"
            ) from exc
        if token != expected_token:
            raise ValueError(f"active environment symlink changed during audit: {candidate}")
    return {
        "canonical_path": str(prefix),
        "file_count": len(files),
        "size_bytes": total_size,
        "tree_sha256": aggregate.hexdigest(),
        "directories_sha256": sha256_json(directory_records),
        "symlink_count": len(symlink_records),
        "symlinks_sha256": sha256_json(symlink_records),
    }


def _distribution_inventory(
    name: str,
    *,
    prefix: Path,
    distribution: importlib.metadata.Distribution | None = None,
) -> dict[str, Any]:
    if distribution is None:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required training distribution is missing: {name}") from exc
    metadata_name = str(distribution.metadata.get("Name", ""))
    distribution_version = distribution.version
    if _normalized_distribution_name(metadata_name) != _normalized_distribution_name(name):
        raise ValueError(
            f"distribution metadata name mismatch: requested={name}, observed={metadata_name}"
        )
    package_files = distribution.files
    if package_files is None or not package_files:
        raise ValueError(f"required distribution has no installed-file inventory: {name}")

    prefix = prefix.resolve(strict=True)
    candidates_by_relative: dict[str, Path] = {}
    metadata_rows: list[tuple[str, str, str]] = []
    metadata_resolutions: dict[str, tuple[str, str]] = {}

    def resolve_allowlisted_missing_target(candidate: Path) -> str:
        normalized = Path(os.path.abspath(os.fspath(candidate)))
        try:
            relative_path = normalized.relative_to(prefix)
        except ValueError as exc:
            raise ValueError(
                f"allowlisted missing distribution {name} file escapes the active "
                f"environment after lexical normalization: {normalized}"
            ) from exc
        if not relative_path.parts:
            raise ValueError(
                f"allowlisted missing distribution {name} file resolves to the "
                "active environment root"
            )

        current = prefix
        missing_ancestor = False
        for component in relative_path.parts[:-1]:
            current /= component
            try:
                observed = current.stat(follow_symlinks=False)
            except FileNotFoundError:
                missing_ancestor = True
                break
            except OSError as exc:
                raise ValueError(
                    f"allowlisted missing distribution {name} file has an "
                    f"uninspectable parent: {current}"
                ) from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(
                    f"allowlisted missing distribution {name} file traverses a "
                    f"symbolic link: {current}"
                )
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(
                    f"allowlisted missing distribution {name} file has a "
                    f"non-directory parent: {current}"
                )

        target = prefix / relative_path
        if not missing_ancestor:
            try:
                observed = target.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ValueError(
                    f"cannot prove allowlisted distribution {name} file is absent: {target}"
                ) from exc
            else:
                entry_type = "symbolic link" if stat.S_ISLNK(observed.st_mode) else "entry"
                raise ValueError(
                    f"allowlisted missing distribution {name} file is a live {entry_type}: "
                    f"{target}"
                )
        return relative_path.as_posix()

    def resolve_metadata_path(
        package_path: object,
    ) -> tuple[str, str, str, Path | None]:
        metadata_path = Path(package_path).as_posix()
        candidate = Path(distribution.locate_file(package_path)).absolute()
        try:
            lexical_relative = candidate.relative_to(prefix)
            current = prefix
            for component in lexical_relative.parts:
                current /= component
                if current.is_symlink():
                    raise ValueError(f"distribution {name} contains a symbolic link: {current}")
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(prefix).as_posix()
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and "contains a symbolic link" in str(exc):
                raise
            if _is_allowlisted_missing_distribution_row(
                normalized_name=_normalized_distribution_name(metadata_name),
                version=distribution_version,
                metadata_path=metadata_path,
            ):
                relative = resolve_allowlisted_missing_target(candidate)
                return metadata_path, "missing", relative, None
            raise ValueError(
                f"distribution {name} file escapes the active environment: {candidate}"
            ) from exc
        return metadata_path, "present", relative, resolved

    for package_path in package_files:
        metadata_path, status, relative, resolved = resolve_metadata_path(package_path)
        resolution = (status, relative)
        prior_resolution = metadata_resolutions.setdefault(metadata_path, resolution)
        if prior_resolution != resolution:
            raise ValueError(
                f"distribution {name} repeats metadata path with a different "
                f"installed-file resolution: "
                f"{metadata_path}"
            )
        if resolved is not None:
            prior_candidate = candidates_by_relative.setdefault(relative, resolved)
            if prior_candidate != resolved:
                raise ValueError(
                    f"distribution {name} repeats installed file with a different "
                    f"path: {relative}"
                )
        metadata_rows.append((metadata_path, status, relative))
    candidates = sorted(candidates_by_relative.items())
    initial_metadata_paths = sorted(metadata_path for metadata_path, _, _ in metadata_rows)
    initial_metadata_rows = sorted(metadata_rows)
    initial_missing_metadata_paths = [
        metadata_path
        for metadata_path, status, _ in metadata_rows
        if status == "missing"
    ]
    if (
        _normalized_distribution_name(metadata_name) == "pip"
        and type(distribution_version) is str
        and distribution_version == PIP_STALE_RECORD_VERSION
    ):
        _validate_pip_stale_missing_metadata_paths(initial_missing_metadata_paths)
    elif initial_missing_metadata_paths:
        raise ValueError(f"distribution {name} contains missing installed-file metadata")

    aggregate = hashlib.sha256()
    total_size = 0
    final_tokens: dict[str, tuple[int, ...]] = {}
    for relative, path in candidates:
        digest, size, mode, token = _stable_file_identity(path)
        total_size += size
        final_tokens[relative] = token
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(mode).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")

    repeated = distribution.files
    if repeated is None:
        raise ValueError(f"distribution inventory disappeared during audit: {name}")
    repeated_rows = [resolve_metadata_path(item) for item in repeated]
    repeated_metadata_paths = sorted(item[0] for item in repeated_rows)
    if repeated_metadata_paths != initial_metadata_paths:
        raise ValueError(f"distribution metadata path multiset changed during audit: {name}")
    repeated_metadata_rows = sorted((item[0], item[1], item[2]) for item in repeated_rows)
    if repeated_metadata_rows != initial_metadata_rows:
        raise ValueError(f"distribution file resolution changed during audit: {name}")
    repeated_missing_metadata_paths = [
        metadata_path
        for metadata_path, status, _, _ in repeated_rows
        if status == "missing"
    ]
    if (
        _normalized_distribution_name(metadata_name) == "pip"
        and type(distribution_version) is str
        and distribution_version == PIP_STALE_RECORD_VERSION
    ):
        _validate_pip_stale_missing_metadata_paths(repeated_missing_metadata_paths)
    elif repeated_missing_metadata_paths:
        raise ValueError(f"distribution {name} contains missing installed-file metadata")
    for relative, path in candidates:
        try:
            current_token = _stat_token(path.stat(follow_symlinks=False))
        except OSError as exc:
            raise ValueError(f"distribution file disappeared during audit: {path}") from exc
        if current_token != final_tokens[relative]:
            raise ValueError(f"distribution file changed during audit: {path}")
    return {
        "name": _normalized_distribution_name(metadata_name),
        "version": distribution_version,
        "file_count": len(candidates),
        "size_bytes": total_size,
        "tree_sha256": aggregate.hexdigest(),
    }


def _package_identity(*, prefix: Path) -> dict[str, Any]:
    """Hash every visible installed distribution, not only the headline stack.

    Training behavior can depend on transitive packages imported by Transformers,
    TRL, Unsloth, datasets, or WandB.  A version-only direct-dependency lock would
    therefore be weaker than the byte identity required by this receipt.
    """

    prefix_tree = _sealed_prefix_tree(prefix)
    by_name: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        metadata_name = str(distribution.metadata.get("Name", ""))
        if not metadata_name:
            raise ValueError("visible distribution lacks a metadata Name")
        normalized = _normalized_distribution_name(metadata_name)
        if normalized in by_name:
            raise ValueError(f"duplicate visible distribution name: {normalized}")
        by_name[normalized] = distribution
    missing = [
        name
        for name in REQUIRED_DISTRIBUTIONS
        if _normalized_distribution_name(name) not in by_name
    ]
    if missing:
        raise ValueError("required training distributions are missing: " + ", ".join(missing))
    unexpected = sorted(set(by_name) - set(REQUIRED_DISTRIBUTIONS))
    if unexpected:
        raise ValueError(
            "sealed Jiang-f638 environment contains undeclared distributions: "
            + ", ".join(unexpected)
        )
    inventories = [
        _distribution_inventory(name, prefix=prefix, distribution=by_name[name])
        for name in sorted(by_name)
    ]
    observed_versions = {
        str(record["name"]): str(record["version"]) for record in inventories
    }
    version_drift = {
        name: {
            "expected": version,
            "observed": observed_versions.get(name),
        }
        for name, version in EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS.items()
        if observed_versions.get(name) != version
    }
    if version_drift:
        raise ValueError(
            "installed environment differs from the released Jiang-f638 lock: "
            + json.dumps(version_drift, sort_keys=True)
        )
    return {
        "inventory_scope": PACKAGE_INVENTORY_SCOPE,
        "prefix_tree": prefix_tree,
        "requirements_lock": {
            "relative_path": REQUIREMENTS_LOCK_RELATIVE_PATH,
            "sha256": REQUIREMENTS_LOCK_SHA256,
            "jiang_source_commit": JIANG_F638_COMMIT,
            "distribution_count": len(EXPECTED_DISTRIBUTION_VERSIONS),
            "versions_sha256": sha256_json(EXPECTED_DISTRIBUTION_VERSIONS),
            "runtime_closure_distributions": dict(
                RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS
            ),
            "runtime_closure_lock_relative_path": RUNTIME_CLOSURE_LOCK_RELATIVE_PATH,
            "runtime_closure_lock_sha256": RUNTIME_CLOSURE_LOCK_SHA256,
            "runtime_closure_versions_sha256": sha256_json(
                RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS
            ),
            "bootstrap_distributions": dict(BOOTSTRAP_DISTRIBUTION_VERSIONS),
            "bootstrap_versions_sha256": sha256_json(BOOTSTRAP_DISTRIBUTION_VERSIONS),
            "visible_distribution_count": len(EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS),
            "visible_versions_sha256": sha256_json(EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS),
        },
        "required_distributions": list(REQUIRED_DISTRIBUTIONS),
        "distributions": inventories,
        "inventory_sha256": sha256_json(inventories),
    }


def _python_identity() -> dict[str, Any]:
    executable = Path(sys.executable).absolute()
    resolved = executable.resolve(strict=True)
    digest, size, mode, _token = _stable_file_identity(resolved)
    prefix = Path(sys.prefix).resolve(strict=True)
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    return {
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": sys.version,
        "version_info": list(sys.version_info),
        "abiflags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "executable_path": str(executable),
        "resolved_executable_path": str(resolved),
        "executable_sha256": digest,
        "executable_size_bytes": size,
        "executable_mode": mode,
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
    }


def _runtime_identity() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        raise ValueError(f"cannot import torch from the sealed environment: {exc}") from exc
    loaded_modules = sorted(item for item in os.environ.get("LOADEDMODULES", "").split(":") if item)
    source_root = Path(__file__).resolve(strict=True).parents[1]
    expected_python_path = os.pathsep.join((str(source_root / "src"), str(source_root / "scripts")))
    if os.environ.get("PYTHONPATH") != expected_python_path:
        raise ValueError(
            "PYTHONPATH must equal the immutable source src/scripts roots exactly: "
            f"expected={expected_python_path!r}, observed={os.environ.get('PYTHONPATH')!r}"
        )
    allowed_roots = {
        source_root,
        Path(sys.prefix).resolve(strict=True),
        Path(sys.base_prefix).resolve(strict=True),
    }
    resolved_sys_path: list[str] = []
    for raw_entry in sys.path:
        candidate = Path(raw_entry or os.getcwd()).expanduser().resolve(strict=False)
        if not any(candidate == root or candidate.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"sys.path escapes the sealed source/environment roots: {candidate}")
        resolved_sys_path.append(str(candidate))
    libc_name, libc_version = platform.libc_ver()
    return {
        "platform": {
            "os_name": os.name,
            "sys_platform": sys.platform,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_build": list(platform.python_build()),
            "python_compiler": platform.python_compiler(),
            "libc": [libc_name, libc_version],
        },
        "activation": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "loaded_modules": loaded_modules,
            "python_path": expected_python_path,
            "sys_path": resolved_sys_path,
            "path": os.environ.get("PATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "ld_preload": os.environ.get("LD_PRELOAD"),
        },
        "torch_build": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "git_version": torch.version.git_version,
            "debug": bool(torch.version.debug),
            "cudnn": torch.backends.cudnn.version(),
            "config": torch.__config__.show(),
        },
    }


def _accelerator_identity(*, expected_gpu_count: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        raise ValueError(f"cannot import torch for accelerator audit: {exc}") from exc
    if expected_gpu_count <= 0:
        raise ValueError("expected_gpu_count must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu_count:
        raise ValueError(
            "sealed environment producer requires the exact accelerator allocation: "
            f"expected={expected_gpu_count}, observed={torch.cuda.device_count()}"
        )
    torch_devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        torch_devices.append(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )
    torch_devices.sort(key=lambda item: _canonical_json_bytes(item))
    if (
        any(EXPECTED_GPU_FAMILY not in item["name"] for item in torch_devices)
        or any(
            tuple(item["compute_capability"]) != EXPECTED_COMPUTE_CAPABILITY
            for item in torch_devices
        )
        or len({_canonical_json_bytes(item) for item in torch_devices}) != 1
    ):
        raise ValueError(
            "Jiang-f638 specialist SFT requires one NVIDIA H100 GPU "
            "with compute capability 9.0"
        )

    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ValueError("nvidia-smi is unavailable in the accelerator allocation")
    executable_path = Path(executable).absolute()
    resolved = executable_path.resolve(strict=True)
    digest, size, mode, _token = _stable_file_identity(resolved)
    try:
        completed = subprocess.run(
            [
                str(executable_path),
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot query the live accelerator runtime: {exc}") from exc
    records = []
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1] or not fields[2].isdigit():
            raise ValueError(f"unexpected nvidia-smi identity row: {raw_line!r}")
        records.append(
            {
                "driver_version": fields[0],
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
            }
        )
    records.sort(key=lambda item: _canonical_json_bytes(item))
    if len(records) != expected_gpu_count:
        raise ValueError(
            "nvidia-smi visible-device count differs from the sealed allocation: "
            f"expected={expected_gpu_count}, observed={len(records)}"
        )
    if (
        any(EXPECTED_GPU_FAMILY not in item["name"] for item in records)
        or len({_canonical_json_bytes(item) for item in records}) != 1
        or records[0]["name"] != torch_devices[0]["name"]
        or abs(
            records[0]["memory_total_mib"] - torch_devices[0]["total_memory_bytes"] / (1024 * 1024)
        )
        > 2048
    ):
        raise ValueError("torch and nvidia-smi do not report the same H100 device")
    return {
        "expected_gpu_count": expected_gpu_count,
        "torch_devices": torch_devices,
        "nvidia_smi": {
            "executable_path": str(executable_path),
            "resolved_executable_path": str(resolved),
            "executable_sha256": digest,
            "executable_size_bytes": size,
            "executable_mode": mode,
            "devices": records,
        },
    }


def _producer_identity() -> dict[str, Any]:
    implementation = Path(__file__).resolve(strict=True)
    digest, size, _mode, _token = _stable_file_identity(implementation)
    return {
        "protocol": PRODUCER_PROTOCOL,
        "implementation": implementation.name,
        "implementation_sha256": digest,
        "implementation_size_bytes": size,
    }


def _live_environment_core(*, include_accelerator: bool) -> dict[str, Any]:
    python = _python_identity()
    prefix = Path(str(python["prefix"]))
    core: dict[str, Any] = {
        "producer": _producer_identity(),
        "python": python,
        "packages": _package_identity(prefix=prefix),
        "runtime": _runtime_identity(),
    }
    if include_accelerator:
        core["accelerator"] = _accelerator_identity(expected_gpu_count=EXPECTED_GPU_COUNT)
    return core


def build_receipt(*, created_at_utc: str | None = None) -> dict[str, Any]:
    timestamp = created_at_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not _validate_timestamp(timestamp):
        raise ValueError("created_at_utc must use canonical YYYY-MM-DDTHH:MM:SSZ")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "claim_scope": CLAIM_SCOPE,
        "created_at_utc": timestamp,
        **_live_environment_core(include_accelerator=True),
    }
    payload["receipt_sha256"] = sha256_json(payload)
    validate_receipt(payload)
    return payload


def _require_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} has an invalid exact schema")
    return value


def _validate_executable_identity(value: Mapping[str, Any], *, label: str) -> None:
    if (
        not _is_absolute_normalized_path(value.get("executable_path"))
        or not _is_absolute_normalized_path(value.get("resolved_executable_path"))
        or not _is_sha256(value.get("executable_sha256"))
        or not _is_plain_int(value.get("executable_size_bytes"), minimum=1)
        or not _is_plain_int(value.get("executable_mode"))
        or int(value["executable_mode"]) > 0o777
        or not int(value["executable_mode"]) & 0o111
    ):
        raise ValueError(f"sealed {label} executable identity is invalid")


def _validate_python_identity(value: object) -> Mapping[str, Any]:
    python = _require_keys(
        value,
        {
            "implementation",
            "cache_tag",
            "version",
            "version_info",
            "abiflags",
            "byteorder",
            "executable_path",
            "resolved_executable_path",
            "executable_sha256",
            "executable_size_bytes",
            "executable_mode",
            "prefix",
            "base_prefix",
        },
        label="environment Python identity",
    )
    version_info = python.get("version_info")
    if (
        python.get("implementation") != "cpython"
        or not _is_nonempty_string(python.get("cache_tag"))
        or not _is_nonempty_string(python.get("version"))
        or not isinstance(python.get("abiflags"), str)
        or python.get("byteorder") not in {"little", "big"}
        or not isinstance(version_info, list)
        or len(version_info) != 5
        or any(not _is_plain_int(item) for item in version_info[:3])
        or not _is_nonempty_string(version_info[3])
        or not _is_plain_int(version_info[4])
        or not _is_absolute_normalized_path(python.get("prefix"))
        or not _is_absolute_normalized_path(python.get("base_prefix"))
    ):
        raise ValueError("sealed environment Python identity is invalid")
    _validate_executable_identity(python, label="Python")
    return python


def _validate_package_identity(value: object) -> Mapping[str, Any]:
    packages = _require_keys(
        value,
        {
            "inventory_scope",
            "prefix_tree",
            "requirements_lock",
            "required_distributions",
            "distributions",
            "inventory_sha256",
        },
        label="environment packages",
    )
    distributions = packages.get("distributions")
    requirements_lock = _require_keys(
        packages.get("requirements_lock"),
        {
            "relative_path",
            "sha256",
            "jiang_source_commit",
            "distribution_count",
            "versions_sha256",
            "runtime_closure_distributions",
            "runtime_closure_lock_relative_path",
            "runtime_closure_lock_sha256",
            "runtime_closure_versions_sha256",
            "bootstrap_distributions",
            "bootstrap_versions_sha256",
            "visible_distribution_count",
            "visible_versions_sha256",
        },
        label="Jiang-f638 requirements lock",
    )
    prefix_tree = _require_keys(
        packages.get("prefix_tree"),
        {
            "canonical_path",
            "file_count",
            "size_bytes",
            "tree_sha256",
            "directories_sha256",
            "symlink_count",
            "symlinks_sha256",
        },
        label="sealed environment prefix tree",
    )
    if (
        packages.get("inventory_scope") != PACKAGE_INVENTORY_SCOPE
        or requirements_lock.get("relative_path") != REQUIREMENTS_LOCK_RELATIVE_PATH
        or requirements_lock.get("sha256") != REQUIREMENTS_LOCK_SHA256
        or requirements_lock.get("jiang_source_commit") != JIANG_F638_COMMIT
        or requirements_lock.get("distribution_count")
        != len(EXPECTED_DISTRIBUTION_VERSIONS)
        or requirements_lock.get("versions_sha256")
        != sha256_json(EXPECTED_DISTRIBUTION_VERSIONS)
        or requirements_lock.get("runtime_closure_distributions")
        != RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS
        or requirements_lock.get("runtime_closure_lock_relative_path")
        != RUNTIME_CLOSURE_LOCK_RELATIVE_PATH
        or requirements_lock.get("runtime_closure_lock_sha256")
        != RUNTIME_CLOSURE_LOCK_SHA256
        or requirements_lock.get("runtime_closure_versions_sha256")
        != sha256_json(RUNTIME_CLOSURE_DISTRIBUTION_VERSIONS)
        or requirements_lock.get("bootstrap_distributions")
        != BOOTSTRAP_DISTRIBUTION_VERSIONS
        or requirements_lock.get("bootstrap_versions_sha256")
        != sha256_json(BOOTSTRAP_DISTRIBUTION_VERSIONS)
        or requirements_lock.get("visible_distribution_count")
        != len(EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS)
        or requirements_lock.get("visible_versions_sha256")
        != sha256_json(EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS)
        or not _is_absolute_normalized_path(prefix_tree.get("canonical_path"))
        or not _is_plain_int(prefix_tree.get("file_count"), minimum=1)
        or not _is_plain_int(prefix_tree.get("size_bytes"), minimum=1)
        or not _is_sha256(prefix_tree.get("tree_sha256"))
        or not _is_sha256(prefix_tree.get("directories_sha256"))
        or not _is_plain_int(prefix_tree.get("symlink_count"))
        or not _is_sha256(prefix_tree.get("symlinks_sha256"))
        or packages.get("required_distributions") != list(REQUIRED_DISTRIBUTIONS)
        or not isinstance(distributions, list)
        or len(distributions) != len(REQUIRED_DISTRIBUTIONS)
        or packages.get("inventory_sha256") != sha256_json(distributions)
    ):
        raise ValueError("sealed remote-environment package inventory is invalid")
    observed_names: list[str] = []
    for index, distribution in enumerate(distributions):
        record = _require_keys(
            distribution,
            {"name", "version", "file_count", "size_bytes", "tree_sha256"},
            label=f"environment distribution {index}",
        )
        name = record.get("name")
        if (
            not _is_nonempty_string(name)
            or name != _normalized_distribution_name(str(name))
            or not _is_nonempty_string(record.get("version"))
            or not _is_plain_int(record.get("file_count"), minimum=1)
            or not _is_plain_int(record.get("size_bytes"), minimum=1)
            or not _is_sha256(record.get("tree_sha256"))
        ):
            raise ValueError(f"environment distribution identity is invalid: {name}")
        observed_names.append(str(name))
    if observed_names != sorted(set(observed_names)):
        raise ValueError("environment distribution inventory is not sorted and unique")
    if observed_names != list(REQUIRED_DISTRIBUTIONS):
        raise ValueError(
            "sealed remote environment distribution names differ from the exact released-lock "
            "plus receipt-proven runtime closure and pinned-pip inventory"
        )
    observed_versions = {
        str(record["name"]): str(record["version"])
        for record in distributions
        if isinstance(record, Mapping)
    }
    drift = {
        name: (version, observed_versions.get(name))
        for name, version in EXPECTED_VISIBLE_DISTRIBUTION_VERSIONS.items()
        if observed_versions.get(name) != version
    }
    if drift:
        raise ValueError(
            "sealed remote environment differs from the Jiang-f638 executable closure: "
            + json.dumps(drift, sort_keys=True)
        )
    return packages


def _validate_runtime_identity(value: object) -> Mapping[str, Any]:
    runtime = _require_keys(
        value,
        {"platform", "activation", "torch_build"},
        label="environment runtime identity",
    )
    platform_identity = _require_keys(
        runtime.get("platform"),
        {
            "os_name",
            "sys_platform",
            "system",
            "release",
            "machine",
            "python_build",
            "python_compiler",
            "libc",
        },
        label="environment platform identity",
    )
    python_build = platform_identity.get("python_build")
    libc = platform_identity.get("libc")
    if (
        any(
            not _is_nonempty_string(platform_identity.get(field))
            for field in (
                "os_name",
                "sys_platform",
                "system",
                "release",
                "machine",
                "python_compiler",
            )
        )
        or not isinstance(python_build, list)
        or len(python_build) != 2
        or any(not isinstance(item, str) for item in python_build)
        or not isinstance(libc, list)
        or len(libc) != 2
        or any(not isinstance(item, str) for item in libc)
    ):
        raise ValueError("sealed environment platform identity is invalid")
    activation = _require_keys(
        runtime.get("activation"),
        {
            "conda_default_env",
            "conda_prefix",
            "virtual_env",
            "loaded_modules",
            "python_path",
            "sys_path",
            "path",
            "ld_library_path",
            "ld_preload",
        },
        label="environment activation identity",
    )
    loaded_modules = activation.get("loaded_modules")
    sys_path = activation.get("sys_path")
    if (
        not _is_optional_string(activation.get("conda_default_env"))
        or not _is_optional_string(activation.get("conda_prefix"))
        or not _is_optional_string(activation.get("virtual_env"))
        or any(
            path is not None and not _is_absolute_normalized_path(path)
            for path in (activation.get("conda_prefix"), activation.get("virtual_env"))
        )
        or not isinstance(loaded_modules, list)
        or any(not _is_nonempty_string(item) for item in loaded_modules)
        or loaded_modules != sorted(set(loaded_modules))
        or not _is_nonempty_string(activation.get("python_path"))
        or not isinstance(sys_path, list)
        or not sys_path
        or any(not _is_absolute_normalized_path(item) for item in sys_path)
        or not _is_nonempty_string(activation.get("path"))
        or not _is_optional_string(activation.get("ld_library_path"))
        or not _is_optional_string(activation.get("ld_preload"))
    ):
        raise ValueError("sealed environment activation identity is invalid")
    torch_build = _require_keys(
        runtime.get("torch_build"),
        {"version", "cuda", "hip", "git_version", "debug", "cudnn", "config"},
        label="environment Torch-build identity",
    )
    if (
        not _is_nonempty_string(torch_build.get("version"))
        or not _is_optional_string(torch_build.get("cuda"))
        or not _is_optional_string(torch_build.get("hip"))
        or not _is_optional_string(torch_build.get("git_version"))
        or type(torch_build.get("debug")) is not bool
        or not (
            torch_build.get("cudnn") is None or _is_plain_int(torch_build.get("cudnn"), minimum=1)
        )
        or not _is_nonempty_string(torch_build.get("config"))
    ):
        raise ValueError("sealed environment Torch-build identity is invalid")
    return runtime


def _validate_accelerator_identity(value: object) -> Mapping[str, Any]:
    accelerator = _require_keys(
        value,
        {"expected_gpu_count", "torch_devices", "nvidia_smi"},
        label="environment accelerator",
    )
    torch_devices = accelerator.get("torch_devices")
    if (
        accelerator.get("expected_gpu_count") != EXPECTED_GPU_COUNT
        or not isinstance(torch_devices, list)
        or len(torch_devices) != EXPECTED_GPU_COUNT
    ):
        raise ValueError("sealed remote-environment accelerator count is invalid")
    for index, device in enumerate(torch_devices):
        record = _require_keys(
            device,
            {
                "name",
                "total_memory_bytes",
                "compute_capability",
                "multi_processor_count",
            },
            label=f"Torch accelerator {index}",
        )
        capability = record.get("compute_capability")
        if (
            not _is_nonempty_string(record.get("name"))
            or not _is_plain_int(record.get("total_memory_bytes"), minimum=1)
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(not _is_plain_int(item) for item in capability)
            or not _is_plain_int(record.get("multi_processor_count"), minimum=1)
            or EXPECTED_GPU_FAMILY not in str(record.get("name"))
            or tuple(capability) != EXPECTED_COMPUTE_CAPABILITY
        ):
            raise ValueError(f"sealed Torch accelerator identity is invalid at index {index}")
    if torch_devices != sorted(torch_devices, key=_canonical_json_bytes):
        raise ValueError("sealed Torch accelerator identity is not canonical")
    if len({_canonical_json_bytes(item) for item in torch_devices}) != 1:
        raise ValueError("sealed Torch accelerator inventory is not homogeneous")
    nvidia = _require_keys(
        accelerator.get("nvidia_smi"),
        {
            "executable_path",
            "resolved_executable_path",
            "executable_sha256",
            "executable_size_bytes",
            "executable_mode",
            "devices",
        },
        label="nvidia-smi identity",
    )
    _validate_executable_identity(nvidia, label="nvidia-smi")
    devices = nvidia.get("devices")
    if not isinstance(devices, list) or len(devices) != EXPECTED_GPU_COUNT:
        raise ValueError("sealed nvidia-smi device count is invalid")
    for index, device in enumerate(devices):
        record = _require_keys(
            device,
            {"driver_version", "name", "memory_total_mib"},
            label=f"nvidia-smi device {index}",
        )
        if (
            not _is_nonempty_string(record.get("driver_version"))
            or not _is_nonempty_string(record.get("name"))
            or not _is_plain_int(record.get("memory_total_mib"), minimum=1)
            or EXPECTED_GPU_FAMILY not in str(record.get("name"))
        ):
            raise ValueError(f"sealed nvidia-smi device identity is invalid at index {index}")
    if devices != sorted(devices, key=_canonical_json_bytes):
        raise ValueError("sealed nvidia-smi device identity is not canonical")
    if (
        len({_canonical_json_bytes(item) for item in devices}) != 1
        or devices[0]["name"] != torch_devices[0]["name"]
        or abs(
            devices[0]["memory_total_mib"] - torch_devices[0]["total_memory_bytes"] / (1024 * 1024)
        )
        > 2048
    ):
        raise ValueError("sealed accelerator inventory is not the required H100 device")
    return accelerator


def validate_receipt(payload: object) -> dict[str, Any]:
    receipt = _require_keys(
        payload,
        {
            "schema_version",
            "protocol",
            "status",
            "claim_scope",
            "created_at_utc",
            "producer",
            "python",
            "packages",
            "runtime",
            "accelerator",
            "receipt_sha256",
        },
        label="sealed remote-environment receipt",
    )
    unsigned = dict(receipt)
    stored = unsigned.pop("receipt_sha256", None)
    if not _is_sha256(stored) or stored != sha256_json(unsigned):
        raise ValueError("sealed remote-environment receipt self-hash mismatch")
    if (
        not _is_plain_int(receipt.get("schema_version"), minimum=1)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != STATUS
        or receipt.get("claim_scope") != CLAIM_SCOPE
        or not _validate_timestamp(receipt.get("created_at_utc"))
    ):
        raise ValueError("sealed remote-environment receipt protocol/status mismatch")
    producer = _require_keys(
        receipt.get("producer"),
        {
            "protocol",
            "implementation",
            "implementation_sha256",
            "implementation_size_bytes",
        },
        label="environment producer",
    )
    if (
        producer.get("protocol") != PRODUCER_PROTOCOL
        or producer.get("implementation") != Path(__file__).name
        or not _is_sha256(producer.get("implementation_sha256"))
        or not _is_plain_int(producer.get("implementation_size_bytes"), minimum=1)
    ):
        raise ValueError("sealed remote-environment producer identity is invalid")
    python = _validate_python_identity(receipt.get("python"))
    packages = _validate_package_identity(receipt.get("packages"))
    runtime = _validate_runtime_identity(receipt.get("runtime"))
    _validate_accelerator_identity(receipt.get("accelerator"))
    prefix = str(python["prefix"])
    resolved_executable = Path(str(python["resolved_executable_path"]))
    if (
        python.get("base_prefix") != prefix
        or not resolved_executable.is_relative_to(Path(prefix))
        or packages["prefix_tree"]["canonical_path"] != prefix
        or runtime["activation"].get("conda_prefix") != prefix
        or not _is_nonempty_string(runtime["activation"].get("conda_default_env"))
        or runtime["activation"].get("virtual_env") is not None
        or runtime["torch_build"].get("cuda") is None
        or runtime["torch_build"].get("hip") is not None
        or runtime["torch_build"].get("cudnn") is None
    ):
        raise ValueError("sealed remote environment has invalid cross-field bindings")
    return dict(receipt)


def _load_read_only_receipt(path: Path) -> tuple[dict[str, Any], str, Path]:
    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"environment receipt is not a regular non-symlink file: {raw}")
    resolved = raw.resolve(strict=True)
    if raw != resolved:
        raise ValueError(f"environment receipt path must be canonical: {raw} != {resolved}")
    if resolved.parent.is_symlink() or not resolved.parent.is_dir():
        raise ValueError(
            f"environment receipt parent is not a regular directory: {resolved.parent}"
        )
    _require_not_writable(resolved.parent, label="environment receipt parent")
    try:
        before_path = resolved.stat(follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode):
            raise ValueError(f"environment receipt is not a regular file: {resolved}")
        if before_path.st_nlink != 1:
            raise ValueError(f"environment receipt must have exactly one hard link: {resolved}")
        if stat.S_IMODE(before_path.st_mode) & 0o222:
            raise ValueError(f"environment receipt must be read-only: {resolved}")
        with resolved.open("rb") as handle:
            before_fd = os.fstat(handle.fileno())
            receipt_bytes = handle.read()
            after_fd = os.fstat(handle.fileno())
        after_path = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot read sealed environment receipt: {exc}") from exc
    identities = [_stat_token(item) for item in (before_path, before_fd, after_fd, after_path)]
    if len(set(identities)) != 1:
        raise ValueError(f"sealed environment receipt changed while read: {resolved}")
    try:
        payload = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sealed environment receipt: {exc}") from exc
    receipt = validate_receipt(payload)
    file_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    return receipt, file_sha256, resolved


def environment_identity(path: Path, *, require_accelerator: bool) -> dict[str, Any]:
    receipt, file_sha256, resolved = _load_read_only_receipt(path)
    live = _live_environment_core(include_accelerator=require_accelerator)
    for field in ("producer", "python", "packages", "runtime"):
        if live[field] != receipt[field]:
            raise ValueError(f"live environment {field} differs from sealed receipt")
    if require_accelerator and live.get("accelerator") != receipt["accelerator"]:
        raise ValueError("live accelerator runtime differs from sealed receipt")
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "claim_scope": CLAIM_SCOPE,
        "canonical_path": str(resolved),
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": file_sha256,
        "producer_implementation_sha256": receipt["producer"]["implementation_sha256"],
        "python_sha256": sha256_json(receipt["python"]),
        "packages_sha256": sha256_json(receipt["packages"]),
        "runtime_sha256": sha256_json(receipt["runtime"]),
        "accelerator_sha256": sha256_json(receipt["accelerator"]),
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite environment receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite environment receipt: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_new_sealed_receipt_directory(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    parent = destination.parent
    if parent.exists() or parent.is_symlink():
        raise FileExistsError(
            f"environment receipt must be published in a new create-once directory: {parent}"
        )
    if not parent.parent.is_dir() or parent.parent.is_symlink():
        raise ValueError(
            f"environment receipt grandparent is not a regular directory: {parent.parent}"
        )
    parent.mkdir(mode=0o700)
    _write_create_once(destination, payload)
    os.chmod(parent, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    _load_read_only_receipt(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at-utc")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-receipt", type=Path)
    parser.add_argument("--require-accelerator", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_receipt is not None:
        if args.created_at_utc is not None:
            raise ValueError("--created-at-utc is valid only while producing a receipt")
        identity = environment_identity(
            args.verify_receipt,
            require_accelerator=args.require_accelerator,
        )
        print(f"verified remote environment: {args.verify_receipt.expanduser().absolute()}")
        print(f"identity_sha256: {identity['identity_sha256']}")
        return 0
    if args.require_accelerator:
        raise ValueError("receipt production always requires the exact accelerator allocation")
    if args.output is None:  # pragma: no cover - argparse enforces the exclusive group
        raise ValueError("--output is required while producing a receipt")
    payload = build_receipt(created_at_utc=args.created_at_utc)
    _write_new_sealed_receipt_directory(args.output, payload)
    print(f"frozen remote environment: {args.output.expanduser().absolute()}")
    print(f"receipt_sha256: {payload['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
