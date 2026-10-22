#!/usr/bin/env python3
"""Freeze the exact leakage-reservation boundary for specialist SFT.

The clean-corpus audit proves that it compared training rows against four
reserved classes.  This independent registry makes those class labels
authoritative: exactly one immutable JSON source is registered for every
task/class pair, all paths live in the prescribed sealed trust-root layout,
and an optional IID generation receipt is content-bound to the five IID files.
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
from datetime import datetime
from pathlib import Path
from typing import Any

from build_clean_sft500k_corpus import iter_json_array

SCHEMA_VERSION = 2
PROTOCOL = "clasp-sft-reserved-source-registry-v2"
STATUS = "frozen_before_specialist_sft_campaign"
TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
SOURCE_CLASSES = ("sft_eval", "rl_eval", "rl_train", "iid_confirmation")
REGISTRY_FILENAME = "sft_reserved_source_registry.json"
RESERVED_ROOT_NAME = "reserved_sources_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KEYS = {
    "task",
    "source_class",
    "canonical_path",
    "size_bytes",
    "sha256",
    "rows",
}
LEGACY_IID_PROVENANCE_CLAIM = "legacy_iid_generated_without_forbidden_registry_binding"
BOUND_IID_PROVENANCE_CLAIM = "iid_generated_with_pre_iid_forbidden_registry_binding"
LEGACY_IID_PROVENANCE_REASON = "schema1_receipt_predates_forbidden_registry_binding"
BOUND_IID_PROVENANCE_REASON = "schema2_receipt_embeds_pre_iid_forbidden_registry_identity"
_LEGACY_GENERATION_KEYS = {
    "contains_references",
    "created_at_utc",
    "generator_code_sha256",
    "generator_contract_sha256",
    "publishable",
    "receipt_sha256",
    "runtime_audit",
    "schema_version",
    "status",
    "tasks",
}
_LEGACY_TASK_KEYS = {
    "attempts",
    "distribution_counts",
    "file_sha256",
    "ordered_instance_hashes_sha256",
    "ordered_instance_sha256",
    "relative_path",
    "rows",
    "seed",
}


class SFTReservedSourceRegistryError(ValueError):
    """Raised when the specialist-SFT reservation boundary is incomplete."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _canonical_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise SFTReservedSourceRegistryError(f"{label} is not a regular directory: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise SFTReservedSourceRegistryError(
            f"{label} must be canonical: supplied={raw}, resolved={resolved}"
        )
    return resolved


def _canonical_file(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_file():
        raise SFTReservedSourceRegistryError(f"{label} is not a regular file: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise SFTReservedSourceRegistryError(
            f"{label} must be canonical: supplied={raw}, resolved={resolved}"
        )
    return resolved


def _require_read_only(path: Path, *, label: str) -> None:
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222:
        raise SFTReservedSourceRegistryError(f"{label} must be read-only: {path}")


def _stable_file_identity(path: Path, *, rows: bool) -> dict[str, Any]:
    source = _canonical_file(path, label="reserved source")
    before = source.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    row_count = sum(1 for _index, _row in iter_json_array(source)) if rows else None
    after = source.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise SFTReservedSourceRegistryError(f"reserved source changed while hashing: {source}")
    record: dict[str, Any] = {
        "canonical_path": str(source),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }
    if rows:
        record["rows"] = row_count
    return record


def _validate_source_record(
    value: object,
    *,
    task: str,
    source_class: str,
    trust_root: Path,
    reopen: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_KEYS:
        raise SFTReservedSourceRegistryError("reserved-source record has an invalid schema")
    path_text = value.get("canonical_path")
    if (
        not isinstance(path_text, str)
        or not path_text.startswith("/")
        or os.path.normpath(path_text) != path_text
    ):
        raise SFTReservedSourceRegistryError("reserved-source path is not canonical absolute")
    expected_parent = trust_root / RESERVED_ROOT_NAME / source_class / task
    source_path = Path(path_text)
    try:
        source_path.relative_to(expected_parent)
    except ValueError as exc:
        raise SFTReservedSourceRegistryError(
            f"{task}/{source_class} source is outside {expected_parent}"
        ) from exc
    if source_path == expected_parent:
        raise SFTReservedSourceRegistryError(f"{task}/{source_class} must name a JSON file")
    if (
        value.get("task") != task
        or value.get("source_class") != source_class
        or source_path.suffix.lower() != ".json"
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
        or not isinstance(value.get("sha256"), str)
        or _SHA256_RE.fullmatch(value["sha256"]) is None
        or type(value.get("rows")) is not int
        or value["rows"] <= 0
    ):
        raise SFTReservedSourceRegistryError(
            f"{task}/{source_class} reserved-source record is invalid"
        )
    if reopen:
        observed = _stable_file_identity(source_path, rows=True)
        _require_read_only(source_path, label=f"{task}/{source_class} reserved source")
        for field in ("canonical_path", "size_bytes", "sha256", "rows"):
            if value[field] != observed[field]:
                raise SFTReservedSourceRegistryError(
                    f"{task}/{source_class} live source differs for {field}"
                )
    return dict(value)


def _is_canonical_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ) == value
    except ValueError:
        return False


def _validate_legacy_runtime_audit(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "source_audit",
        "version_mismatches",
        "versions",
    }:
        raise SFTReservedSourceRegistryError("legacy IID runtime audit is noncanonical")
    source = value.get("source_audit")
    if (
        not isinstance(source, dict)
        or set(source) != {"root", "verified_locally", "observed_files_sha256"}
        or source.get("verified_locally") is not True
        or not isinstance(source.get("root"), str)
        or not Path(source["root"]).is_absolute()
        or not isinstance(source.get("observed_files_sha256"), dict)
        or not source["observed_files_sha256"]
        or any(
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for relative, digest in source["observed_files_sha256"].items()
        )
        or value.get("version_mismatches") != {}
    ):
        raise SFTReservedSourceRegistryError("legacy IID source/runtime audit is invalid")
    versions = value.get("versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"python", "numpy", "networkx"}
        or any(not isinstance(version, str) or not version for version in versions.values())
    ):
        raise SFTReservedSourceRegistryError("legacy IID runtime versions are invalid")


def _legacy_task_binding(task: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LEGACY_TASK_KEYS:
        raise SFTReservedSourceRegistryError(f"legacy IID {task} task record is noncanonical")
    rows = value.get("rows")
    attempts = value.get("attempts")
    hashes = value.get("ordered_instance_sha256")
    distributions = value.get("distribution_counts")
    relative = value.get("relative_path")
    if (
        type(rows) is not int
        or rows <= 0
        or type(attempts) is not int
        or attempts < rows
        or type(value.get("seed")) is not int
        or not isinstance(relative, str)
        or relative != f"{task}/eval/test.json"
        or not isinstance(value.get("file_sha256"), str)
        or _SHA256_RE.fullmatch(value["file_sha256"]) is None
        or not isinstance(hashes, list)
        or len(hashes) != rows
        or len(set(hashes)) != rows
        or any(not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None for digest in hashes)
        or value.get("ordered_instance_hashes_sha256") != _sha256_json(hashes)
        or not isinstance(distributions, dict)
        or not distributions
        or any(
            not isinstance(name, str)
            or not name
            or type(count) is not int
            or count <= 0
            for name, count in distributions.items()
        )
        or sum(distributions.values()) != rows
    ):
        raise SFTReservedSourceRegistryError(f"legacy IID {task} task record is invalid")
    return {
        "file_sha256": value["file_sha256"],
        "rows": rows,
        "relative_path": relative,
        "ordered_instance_hashes_sha256": value["ordered_instance_hashes_sha256"],
    }


def _modern_task_binding(task: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SFTReservedSourceRegistryError(f"IID generation receipt has invalid {task} record")
    relative = value.get("relative_path")
    if (
        not isinstance(value.get("file_sha256"), str)
        or _SHA256_RE.fullmatch(value["file_sha256"]) is None
        or type(value.get("rows")) is not int
        or value["rows"] <= 0
        or not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(value.get("ordered_instance_hashes_sha256"), str)
        or _SHA256_RE.fullmatch(value["ordered_instance_hashes_sha256"]) is None
    ):
        raise SFTReservedSourceRegistryError(
            f"IID generation receipt has invalid {task} file binding"
        )
    return {
        "file_sha256": value["file_sha256"],
        "rows": value["rows"],
        "relative_path": relative,
        "ordered_instance_hashes_sha256": value["ordered_instance_hashes_sha256"],
    }


def _load_generation_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = _canonical_file(path, label="IID generation receipt")
    _require_read_only(receipt_path, label="IID generation receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTReservedSourceRegistryError("IID generation receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise SFTReservedSourceRegistryError("IID generation receipt must be an object")
    unsigned = dict(receipt)
    stored = unsigned.pop("receipt_sha256", None)
    tasks = receipt.get("tasks")
    if (
        not isinstance(stored, str)
        or _SHA256_RE.fullmatch(stored) is None
        or stored != _sha256_json(unsigned)
        or receipt.get("status") != "instances_generated"
        or receipt.get("contains_references") is not False
        or receipt.get("publishable") is not False
        or not isinstance(tasks, dict)
        or tuple(tasks) != TASKS
    ):
        raise SFTReservedSourceRegistryError(
            "IID generation receipt is not a self-hashed unlabeled production receipt"
        )

    schema_version = receipt.get("schema_version")
    pre_iid_identity: dict[str, Any] | None
    if schema_version == 1:
        if (
            set(receipt) != _LEGACY_GENERATION_KEYS
            or not _is_canonical_timestamp(receipt.get("created_at_utc"))
            or any(
                not isinstance(receipt.get(field), str)
                or _SHA256_RE.fullmatch(receipt[field]) is None
                for field in ("generator_code_sha256", "generator_contract_sha256")
            )
        ):
            raise SFTReservedSourceRegistryError("legacy IID generation receipt is noncanonical")
        _validate_legacy_runtime_audit(receipt.get("runtime_audit"))
        task_bindings = {
            task: _legacy_task_binding(task, tasks[task]) for task in TASKS
        }
        provenance_claim = LEGACY_IID_PROVENANCE_CLAIM
        provenance_reason = LEGACY_IID_PROVENANCE_REASON
        pre_iid_identity = None
    elif schema_version == 2:
        task_bindings = {
            task: _modern_task_binding(task, tasks[task]) for task in TASKS
        }
        raw_pre_iid = receipt.get("forbidden_source_registry_identity")
        if not isinstance(raw_pre_iid, dict):
            raise SFTReservedSourceRegistryError(
                "schema-v2 IID receipt lacks pre-IID forbidden-registry provenance"
            )
        pre_iid_identity = dict(raw_pre_iid)
        provenance_claim = BOUND_IID_PROVENANCE_CLAIM
        provenance_reason = BOUND_IID_PROVENANCE_REASON
    else:
        raise SFTReservedSourceRegistryError("unsupported IID generation receipt schema")

    file_record = _stable_file_identity(receipt_path, rows=False)
    return receipt, {
        **file_record,
        "receipt_sha256": stored,
        "receipt_schema_version": schema_version,
        "receipt_status": receipt["status"],
        "provenance_claim": provenance_claim,
        "provenance_reason": provenance_reason,
        "pre_iid_forbidden_registry_identity": pre_iid_identity,
        "task_bindings": task_bindings,
    }


def _validate_generation_identity(value: object, *, reopen: bool) -> dict[str, Any] | None:
    if value is None:
        return None
    expected = {
        "canonical_path",
        "size_bytes",
        "sha256",
        "receipt_sha256",
        "receipt_schema_version",
        "receipt_status",
        "provenance_claim",
        "provenance_reason",
        "pre_iid_forbidden_registry_identity",
        "task_bindings",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise SFTReservedSourceRegistryError("IID generation identity has an invalid schema")
    path_text = value.get("canonical_path")
    if (
        not isinstance(path_text, str)
        or not path_text.startswith("/")
        or os.path.normpath(path_text) != path_text
    ):
        raise SFTReservedSourceRegistryError("IID generation identity path is not canonical")
    if any(
        not isinstance(value.get(field), str)
        or _SHA256_RE.fullmatch(value[field]) is None
        for field in ("sha256", "receipt_sha256")
    ):
        raise SFTReservedSourceRegistryError("IID generation identity has invalid hashes")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0:
        raise SFTReservedSourceRegistryError("IID generation identity has invalid size")
    schema_version = value.get("receipt_schema_version")
    pre_iid_identity = value.get("pre_iid_forbidden_registry_identity")
    if (
        schema_version not in {1, 2}
        or value.get("receipt_status") != "instances_generated"
        or (
            schema_version == 1
            and (
                value.get("provenance_claim") != LEGACY_IID_PROVENANCE_CLAIM
                or value.get("provenance_reason") != LEGACY_IID_PROVENANCE_REASON
                or pre_iid_identity is not None
            )
        )
        or (
            schema_version == 2
            and (
                value.get("provenance_claim") != BOUND_IID_PROVENANCE_CLAIM
                or value.get("provenance_reason") != BOUND_IID_PROVENANCE_REASON
                or not isinstance(pre_iid_identity, dict)
            )
        )
    ):
        raise SFTReservedSourceRegistryError("IID generation provenance claim is invalid")
    bindings = value.get("task_bindings")
    if not isinstance(bindings, dict) or tuple(bindings) != TASKS:
        raise SFTReservedSourceRegistryError("IID generation identity has invalid task bindings")
    for task in TASKS:
        binding = bindings[task]
        if (
            not isinstance(binding, dict)
            or set(binding)
            != {
                "file_sha256",
                "rows",
                "relative_path",
                "ordered_instance_hashes_sha256",
            }
            or not isinstance(binding.get("file_sha256"), str)
            or _SHA256_RE.fullmatch(binding["file_sha256"]) is None
            or not isinstance(binding.get("ordered_instance_hashes_sha256"), str)
            or _SHA256_RE.fullmatch(binding["ordered_instance_hashes_sha256"]) is None
            or type(binding.get("rows")) is not int
            or binding["rows"] <= 0
            or not isinstance(binding.get("relative_path"), str)
            or not binding["relative_path"]
            or Path(binding["relative_path"]).is_absolute()
            or ".." in Path(binding["relative_path"]).parts
        ):
            raise SFTReservedSourceRegistryError(
                f"IID generation identity has invalid {task} binding"
            )
    if reopen:
        _receipt, observed = _load_generation_receipt(Path(str(value["canonical_path"])))
        if observed != value:
            raise SFTReservedSourceRegistryError("IID generation receipt changed after freeze")
    return dict(value)


def iid_generation_receipt_identity(path: Path) -> dict[str, Any]:
    """Return the validated compact identity for a retained IID receipt."""

    _receipt, identity = _load_generation_receipt(path)
    return identity


def validate_iid_generation_receipt_identity(
    value: object,
    *,
    reopen: bool = False,
) -> dict[str, Any] | None:
    """Validate a portable IID receipt identity, optionally reopening its file."""

    return _validate_generation_identity(value, reopen=reopen)


def build_registry(
    *,
    trust_root: Path,
    sources: Sequence[tuple[str, str, Path]],
    iid_generation_receipt: Path | None,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    root = _canonical_directory(trust_root, label="immutable input trust root")
    expected = {(task, source_class) for task in TASKS for source_class in SOURCE_CLASSES}
    by_key: dict[tuple[str, str], Path] = {}
    for task, source_class, path in sources:
        key = (task, source_class)
        if key not in expected or key in by_key:
            raise SFTReservedSourceRegistryError(f"duplicate or invalid reserved source: {key}")
        by_key[key] = path
    if set(by_key) != expected:
        raise SFTReservedSourceRegistryError(
            f"reserved registry requires exactly 20 task/class sources; missing={sorted(expected-set(by_key))}"
        )
    records: list[dict[str, Any]] = []
    for task in TASKS:
        for source_class in SOURCE_CLASSES:
            record = {
                "task": task,
                "source_class": source_class,
                **_stable_file_identity(by_key[(task, source_class)], rows=True),
            }
            _require_read_only(
                Path(record["canonical_path"]),
                label=f"{task}/{source_class} reserved source",
            )
            records.append(
                _validate_source_record(
                    record,
                    task=task,
                    source_class=source_class,
                    trust_root=root,
                    reopen=False,
                )
            )
    generation_identity = None
    if iid_generation_receipt is not None:
        receipt_path = _canonical_file(iid_generation_receipt, label="IID generation receipt")
        expected_receipt_root = root / RESERVED_ROOT_NAME / "iid_confirmation"
        try:
            receipt_path.relative_to(expected_receipt_root)
        except ValueError as exc:
            raise SFTReservedSourceRegistryError(
                f"IID generation receipt is outside {expected_receipt_root}"
            ) from exc
        _receipt, generation_identity = _load_generation_receipt(receipt_path)
        by_task = {(record["task"], record["source_class"]): record for record in records}
        for task in TASKS:
            source = by_task[(task, "iid_confirmation")]
            binding = generation_identity["task_bindings"][task]
            if source["sha256"] != binding["file_sha256"] or source["rows"] != binding["rows"]:
                raise SFTReservedSourceRegistryError(
                    f"{task} IID reserved source differs from its generation receipt"
                )
    timestamp = frozen_at_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        if datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ) != timestamp:
            raise ValueError
    except ValueError as exc:
        raise SFTReservedSourceRegistryError("frozen_at_utc is not canonical UTC") from exc
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "frozen_at_utc": timestamp,
        "immutable_input_trust_root": str(root),
        "tasks": list(TASKS),
        "source_classes": list(SOURCE_CLASSES),
        "sources": records,
        "source_manifest_sha256": _sha256_json(records),
        "iid_generation_receipt": generation_identity,
    }
    payload["registry_sha256"] = _sha256_json(payload)
    return payload


def validate_registry_identity(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "canonical_path",
        "registry_sha256",
        "registry_file_sha256",
        "immutable_input_trust_root",
        "source_manifest_sha256",
        "sources",
        "iid_generation_receipt",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SFTReservedSourceRegistryError("reserved-source registry identity schema is invalid")
    unsigned = dict(value)
    stored = unsigned.pop("identity_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL
        or value.get("status") != STATUS
        or any(
            not isinstance(value.get(field), str)
            or _SHA256_RE.fullmatch(value[field]) is None
            for field in ("registry_sha256", "registry_file_sha256", "source_manifest_sha256")
        )
        or stored != _sha256_json(unsigned)
    ):
        raise SFTReservedSourceRegistryError("reserved-source registry identity is invalid")
    root_text = value.get("immutable_input_trust_root")
    path_text = value.get("canonical_path")
    if (
        not isinstance(root_text, str)
        or not root_text.startswith("/")
        or os.path.normpath(root_text) != root_text
        or not isinstance(path_text, str)
        or not path_text.startswith("/")
        or os.path.normpath(path_text) != path_text
    ):
        raise SFTReservedSourceRegistryError("reserved-source identity paths are invalid")
    root = Path(root_text)
    records = value.get("sources")
    if not isinstance(records, list) or len(records) != len(TASKS) * len(SOURCE_CLASSES):
        raise SFTReservedSourceRegistryError("reserved-source identity does not contain 20 sources")
    checked: list[dict[str, Any]] = []
    for record, (task, source_class) in zip(
        records,
        ((task, source_class) for task in TASKS for source_class in SOURCE_CLASSES),
        strict=True,
    ):
        checked.append(
            _validate_source_record(
                record,
                task=task,
                source_class=source_class,
                trust_root=root,
                reopen=False,
            )
        )
    if value["source_manifest_sha256"] != _sha256_json(checked):
        raise SFTReservedSourceRegistryError("reserved-source identity manifest hash mismatch")
    generation = _validate_generation_identity(
        value.get("iid_generation_receipt"), reopen=False
    )
    if generation is not None:
        generation_path = Path(generation["canonical_path"])
        expected_generation_parent = root / RESERVED_ROOT_NAME / "iid_confirmation"
        try:
            generation_path.relative_to(expected_generation_parent)
        except ValueError as exc:
            raise SFTReservedSourceRegistryError(
                "IID generation receipt is outside the reserved IID authority"
            ) from exc
        by_key = {(record["task"], record["source_class"]): record for record in checked}
        for task in TASKS:
            source = by_key[(task, "iid_confirmation")]
            binding = generation["task_bindings"][task]
            if source["sha256"] != binding["file_sha256"] or source["rows"] != binding["rows"]:
                raise SFTReservedSourceRegistryError(
                    f"{task} IID source differs from embedded generation provenance"
                )
    return dict(value)


def validate_registry(
    path: Path,
    *,
    expected_trust_root: Path,
) -> dict[str, Any]:
    registry_path = _canonical_file(path, label="SFT reserved-source registry")
    _require_read_only(registry_path, label="SFT reserved-source registry")
    root = _canonical_directory(expected_trust_root, label="immutable input trust root")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTReservedSourceRegistryError("cannot read SFT reserved-source registry") from exc
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "frozen_at_utc",
        "immutable_input_trust_root",
        "tasks",
        "source_classes",
        "sources",
        "source_manifest_sha256",
        "iid_generation_receipt",
        "registry_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise SFTReservedSourceRegistryError("SFT reserved-source registry schema is invalid")
    if registry_path.read_bytes() != _canonical_file_bytes(payload):
        raise SFTReservedSourceRegistryError("SFT reserved-source registry is not canonical JSON")
    unsigned = dict(payload)
    stored = unsigned.pop("registry_sha256", None)
    if (
        stored != _sha256_json(unsigned)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("status") != STATUS
        or payload.get("immutable_input_trust_root") != str(root)
        or payload.get("tasks") != list(TASKS)
        or payload.get("source_classes") != list(SOURCE_CLASSES)
    ):
        raise SFTReservedSourceRegistryError("SFT reserved-source registry contract is invalid")
    records = payload.get("sources")
    if not isinstance(records, list) or len(records) != len(TASKS) * len(SOURCE_CLASSES):
        raise SFTReservedSourceRegistryError("SFT reserved-source registry lacks 20 sources")
    checked: list[dict[str, Any]] = []
    for record, (task, source_class) in zip(
        records,
        ((task, source_class) for task in TASKS for source_class in SOURCE_CLASSES),
        strict=True,
    ):
        checked.append(
            _validate_source_record(
                record,
                task=task,
                source_class=source_class,
                trust_root=root,
                reopen=True,
            )
        )
    if payload.get("source_manifest_sha256") != _sha256_json(checked):
        raise SFTReservedSourceRegistryError("SFT reserved-source registry manifest hash mismatch")
    generation = _validate_generation_identity(
        payload.get("iid_generation_receipt"), reopen=True
    )
    if generation is not None:
        by_key = {(record["task"], record["source_class"]): record for record in checked}
        for task in TASKS:
            source = by_key[(task, "iid_confirmation")]
            binding = generation["task_bindings"][task]
            if source["sha256"] != binding["file_sha256"] or source["rows"] != binding["rows"]:
                raise SFTReservedSourceRegistryError(
                    f"{task} IID source differs from the live generation receipt"
                )
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "canonical_path": str(registry_path),
        "registry_sha256": payload["registry_sha256"],
        "registry_file_sha256": _stable_file_identity(registry_path, rows=False)["sha256"],
        "immutable_input_trust_root": str(root),
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "sources": checked,
        "iid_generation_receipt": generation,
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return validate_registry_identity(identity)


def sources_by_task_class(identity: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    checked = validate_registry_identity(dict(identity))
    result = {task: {} for task in TASKS}
    for record in checked["sources"]:
        result[record["task"]][record["source_class"]] = dict(record)
    return result


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace reserved-source registry: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_file_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        os.link(temporary, destination)
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


def _parse_source(value: str) -> tuple[str, str, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3 or parts[0] not in TASKS or parts[1] not in SOURCE_CLASSES or not parts[2]:
        raise argparse.ArgumentTypeError("source must be TASK,CLASS,PATH")
    return parts[0], parts[1], Path(parts[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--immutable-input-trust-root", type=Path, required=True)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--iid-generation-receipt", type=Path)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_registry(
            trust_root=args.immutable_input_trust_root,
            sources=args.source,
            iid_generation_receipt=args.iid_generation_receipt,
            frozen_at_utc=args.frozen_at_utc,
        )
        _write_create_once(args.output, payload)
        identity = validate_registry(
            args.output,
            expected_trust_root=args.immutable_input_trust_root,
        )
        print(json.dumps(identity, sort_keys=True))
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[SFTReservedSourceRegistryError] {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
