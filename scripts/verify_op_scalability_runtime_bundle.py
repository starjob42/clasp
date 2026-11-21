#!/usr/bin/env python3
"""Fail-closed verifier for the staged OP scalability Python bundle.

This verifier intentionally uses only the Python standard library.  The Slurm
launcher authenticates this file and the manifest before invoking it.  The
verifier then checks the manifest self-hash, exact bundle inventory, every
file's content/size/mode/link count, and the complete local import closure
before any bundled CLASP module is imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL = "clasp-op-scalability-runtime-python-bundle-v1"
EXPECTED_FILE_MODE = "0440"
EXPECTED_FILE_NLINK = 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _strict_regular_file(
    path: Path,
    *,
    label: str,
    expected_mode: str,
    expected_nlink: int,
) -> os.stat_result:
    absolute = path.absolute()
    if path.is_symlink() or path.resolve() != absolute:
        raise ValueError(f"{label} must have a canonical nonsymlink path: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    mode = f"{stat.S_IMODE(metadata.st_mode):04o}"
    if mode != expected_mode or metadata.st_nlink != expected_nlink:
        raise ValueError(
            f"{label} mode/link mismatch: observed={mode}/{metadata.st_nlink}, "
            f"expected={expected_mode}/{expected_nlink}"
        )
    return metadata


def _load_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, object]:
    if not _is_sha256(expected_file_sha256):
        raise ValueError("expected manifest file SHA-256 is invalid")
    _strict_regular_file(
        path,
        label="runtime bundle manifest",
        expected_mode=EXPECTED_FILE_MODE,
        expected_nlink=EXPECTED_FILE_NLINK,
    )
    observed_file_sha256 = _sha256_file(path)
    if observed_file_sha256 != expected_file_sha256:
        raise ValueError(
            "runtime bundle manifest file SHA-256 mismatch: "
            f"observed={observed_file_sha256}, expected={expected_file_sha256}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime bundle manifest is not valid UTF-8 JSON") from exc
    manifest = _require_exact_keys(
        value,
        {
            "schema_version",
            "protocol",
            "status",
            "purpose",
            "confirmation_data_used",
            "pythonpath_order",
            "entrypoint_modules",
            "required_local_modules",
            "entries",
            "manifest_sha256",
        },
        label="runtime bundle manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("runtime bundle manifest schema version differs")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("runtime bundle manifest protocol differs")
    if manifest["status"] != "frozen_read_only_development_runtime_bundle":
        raise ValueError("runtime bundle manifest status is not frozen")
    if manifest["purpose"] != "development_only_op_scalability_sweep":
        raise ValueError("runtime bundle manifest purpose differs")
    if manifest["confirmation_data_used"] is not False:
        raise ValueError("runtime bundle must forbid confirmation data")
    stored = manifest["manifest_sha256"]
    if not _is_sha256(stored):
        raise ValueError("runtime bundle manifest self-hash is invalid")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if observed != stored:
        raise ValueError(
            f"runtime bundle manifest self-hash mismatch: {observed} != {stored}"
        )
    return manifest


def _validated_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} is not a normalized safe relative path: {value!r}")
    if not (
        value in {
            "scripts/run_op_scalability_sweep.py",
            "scripts/run_reference_evidence.py",
        }
        or value.startswith("src/clasp/")
    ):
        raise ValueError(f"{label} is outside the closed runtime source roots: {value}")
    if not value.endswith(".py"):
        raise ValueError(f"{label} is not a Python source file: {value}")
    return value


def _module_for_relative_path(relative_path: str) -> tuple[str, bool]:
    pure = PurePosixPath(relative_path)
    if pure.parts[0] == "scripts":
        return pure.stem, False
    module_parts = list(pure.with_suffix("").parts[1:])
    is_package = module_parts[-1] == "__init__"
    if is_package:
        module_parts.pop()
    return ".".join(module_parts), is_package


def _local_import_targets(
    tree: ast.AST,
    *,
    module_name: str,
    is_package: bool,
) -> set[str]:
    targets: set[str] = set()
    package = module_name if is_package else module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                retained = len(package_parts) - (node.level - 1)
                if retained < 0:
                    raise ValueError(
                        f"relative import escapes package in module {module_name}"
                    )
                base = ".".join(package_parts[:retained])
                target = ".".join(part for part in (base, node.module or "") if part)
                if target:
                    candidates.append(target)
                if node.module is None:
                    candidates.extend(
                        ".".join(part for part in (base, alias.name) if part)
                        for alias in node.names
                    )
            elif node.module:
                candidates.append(node.module)
        for candidate in candidates:
            if candidate == "scripts.run_reference_evidence":
                candidate = "run_reference_evidence"
            if candidate == "run_reference_evidence" or candidate == "clasp" or candidate.startswith("clasp."):
                targets.add(candidate)
    return targets


def _verify_import_closure(
    *,
    manifest: Mapping[str, object],
    files_by_relative_path: Mapping[str, Path],
) -> list[str]:
    module_files: dict[str, tuple[Path, bool]] = {}
    for relative_path, path in files_by_relative_path.items():
        module, is_package = _module_for_relative_path(relative_path)
        if module in module_files:
            raise ValueError(f"duplicate runtime module {module}")
        module_files[module] = (path, is_package)

    raw_required = manifest["required_local_modules"]
    raw_entrypoints = manifest["entrypoint_modules"]
    if (
        not isinstance(raw_required, list)
        or not raw_required
        or any(not isinstance(value, str) or not value for value in raw_required)
    ):
        raise ValueError("required_local_modules must be a nonempty string list")
    if (
        not isinstance(raw_entrypoints, list)
        or raw_entrypoints != ["run_op_scalability_sweep", "run_reference_evidence"]
    ):
        raise ValueError("entrypoint_modules differ from the exact OP runtime entrypoints")
    if raw_required != sorted(set(raw_required)):
        raise ValueError("required_local_modules must be sorted and unique")
    required = set(raw_required)
    if required != set(module_files):
        raise ValueError(
            "manifest module inventory differs from file inventory; "
            f"missing={sorted(required - set(module_files))}, "
            f"extra={sorted(set(module_files) - required)}"
        )

    discovered: set[str] = set()
    pending = list(raw_entrypoints)
    while pending:
        module = pending.pop()
        if module in discovered:
            continue
        if module not in module_files:
            raise ValueError(f"local import closure is missing module {module}")
        discovered.add(module)
        path, is_package = module_files[module]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ValueError(f"cannot parse bundled source {path}") from exc
        for target in _local_import_targets(
            tree,
            module_name=module,
            is_package=is_package,
        ):
            parts = target.split(".")
            for index in range(1, len(parts)):
                package = ".".join(parts[:index])
                if package in module_files:
                    pending.append(package)
            if target in module_files:
                pending.append(target)
            elif target == "clasp" or target.startswith("clasp.") or target == "run_reference_evidence":
                raise ValueError(f"local import closure is missing module {target}")
    if discovered != required:
        raise ValueError(
            "manifest contains unreachable local modules; "
            f"unreachable={sorted(required - discovered)}"
        )
    return sorted(discovered)


def verify_bundle(
    *,
    manifest_path: str | Path,
    bundle_root: str | Path,
    expected_manifest_file_sha256: str,
) -> dict[str, object]:
    manifest_file = Path(manifest_path).absolute()
    root = Path(bundle_root).absolute()
    manifest = _load_manifest(
        manifest_file,
        expected_file_sha256=expected_manifest_file_sha256,
    )
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError("runtime bundle root must be a canonical nonsymlink directory")
    if manifest["pythonpath_order"] != ["src", "scripts"]:
        raise ValueError("runtime bundle Python path order differs")

    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("runtime bundle entries must be a nonempty list")
    expected_paths: set[str] = set()
    files_by_relative_path: dict[str, Path] = {}
    previous = ""
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_exact_keys(
            raw_entry,
            {"relative_path", "sha256", "size_bytes", "mode", "nlink", "role"},
            label=f"runtime bundle entry {index}",
        )
        relative_path = _validated_relative_path(
            entry["relative_path"], label=f"runtime bundle entry {index} path"
        )
        if relative_path <= previous:
            raise ValueError("runtime bundle entries must be strictly path-sorted")
        previous = relative_path
        if relative_path in expected_paths:
            raise ValueError(f"duplicate runtime bundle path {relative_path}")
        expected_paths.add(relative_path)
        if not _is_sha256(entry["sha256"]):
            raise ValueError(f"runtime bundle entry {relative_path} has invalid SHA-256")
        if (
            isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] <= 0
        ):
            raise ValueError(f"runtime bundle entry {relative_path} has invalid size")
        if entry["mode"] != EXPECTED_FILE_MODE or entry["nlink"] != EXPECTED_FILE_NLINK:
            raise ValueError(f"runtime bundle entry {relative_path} weakens file identity")
        module, is_package = _module_for_relative_path(relative_path)
        expected_role = (
            "entrypoint"
            if module in {"run_op_scalability_sweep", "run_reference_evidence"}
            else "package_marker"
            if is_package
            else "local_dependency"
        )
        if entry["role"] != expected_role:
            raise ValueError(f"runtime bundle entry {relative_path} has wrong role")
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        metadata = _strict_regular_file(
            path,
            label=f"runtime bundle entry {relative_path}",
            expected_mode=EXPECTED_FILE_MODE,
            expected_nlink=EXPECTED_FILE_NLINK,
        )
        if metadata.st_size != entry["size_bytes"]:
            raise ValueError(f"runtime bundle entry {relative_path} size mismatch")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != entry["sha256"]:
            raise ValueError(
                f"runtime bundle entry {relative_path} SHA-256 mismatch: "
                f"{observed_sha256} != {entry['sha256']}"
            )
        files_by_relative_path[relative_path] = path

    observed_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or child.is_symlink():
                raise ValueError(f"runtime bundle contains an unsafe directory entry: {child}")
        for name in file_names:
            child = directory_path / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or child.is_symlink():
                raise ValueError(f"runtime bundle contains an unsafe file entry: {child}")
            observed_paths.add(child.relative_to(root).as_posix())
    if observed_paths != expected_paths:
        raise ValueError(
            "runtime bundle exact inventory mismatch; "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"extra={sorted(observed_paths - expected_paths)}"
        )

    closure = _verify_import_closure(
        manifest=manifest,
        files_by_relative_path=files_by_relative_path,
    )
    return {
        "status": "PASS_EXACT_RUNTIME_BUNDLE",
        "confirmation_data_used": False,
        "manifest_file_sha256": expected_manifest_file_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": len(expected_paths),
        "local_module_count": len(closure),
        "local_import_closure": closure,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_bundle(
            manifest_path=args.manifest,
            bundle_root=args.bundle_root,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[OPRuntimeBundleError] {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
