#!/usr/bin/env python3
"""Freeze complete leakage-source registries at an explicit temporal boundary.

This receipt closes a subtle leakage-audit loophole: naming one non-empty file
for each required label does not prove that the complete SFT, RL, and historical
evaluation trees were audited. The registry enumerates and hashes every JSON
file in every required task/label tree, self-hashes the resulting artifact, and
is later cross-bound to the publication benchmark manifest.  The legacy mode
retains the pre-IID registry contract.  The base-SFT mode has a distinct
post-IID/pre-training protocol and explicitly records that the retained
schema-v1 IID set had no pre-IID forbidden-registry binding.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from freeze_benchmark import (  # type: ignore[import-not-found]
    DatasetSpec,
    _forbidden_hashes,
    _parse_dataset_spec,
    _write_json_atomic,
)

from clasp.data.benchmark import (
    FORBIDDEN_SOURCE_REGISTRY_PROTOCOL,
    FORBIDDEN_SOURCE_REGISTRY_SCHEMA_VERSION,
    FORBIDDEN_SOURCE_REGISTRY_STATUS,
    REQUIRED_FORBIDDEN_SOURCE_LABELS,
    SUPPORTED_BENCHMARK_TASKS,
    forbidden_source_manifest_identity,
    forbidden_source_registry_identity,
    sha256_json,
    validate_forbidden_source_registry,
)

from freeze_sft_reserved_source_registry import (  # type: ignore[import-not-found]
    LEGACY_IID_PROVENANCE_CLAIM,
    iid_generation_receipt_identity,
    validate_iid_generation_receipt_identity,
)

BASE_SFT_SCHEMA_VERSION = 1
BASE_SFT_PROTOCOL = "clasp-base-sft-post-iid-source-registry-v1"
BASE_SFT_STATUS = "frozen_after_iid_generation_before_specialist_sft_training"
BASE_SFT_TEMPORAL_CLAIM = "post_iid_registry_for_specialist_sft_only"
BASE_SFT_PRE_IID_REGISTRY_IDENTITY = None
BASE_SFT_PRE_IID_PROVENANCE_REASON = (
    "retained_schema1_iid_receipt_has_no_forbidden_registry_binding"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _audit_source_records(
    specs: Sequence[DatasetSpec],
    *,
    expected_tasks: Sequence[str],
) -> tuple[list[str], list[dict[str, object]]]:
    tasks = sorted(expected_tasks)
    if (
        not tasks
        or any(task not in SUPPORTED_BENCHMARK_TASKS for task in tasks)
        or len(tasks) != len(set(tasks))
    ):
        raise ValueError("expected tasks must be a unique non-empty supported set")
    expected_keys = {
        (task, label)
        for task in tasks
        for label in REQUIRED_FORBIDDEN_SOURCE_LABELS
    }
    observed_keys = [(spec.task, spec.label) for spec in specs]
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError("duplicate forbidden-source registry specification")
    if set(observed_keys) != expected_keys:
        missing = sorted(expected_keys - set(observed_keys))
        extra = sorted(set(observed_keys) - expected_keys)
        raise ValueError(
            "registry specifications must exactly cover every task/label pair: "
            f"missing={missing}, extra={extra}"
        )

    sources: list[dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: (item.task, item.label)):
        hashes, audited = _forbidden_hashes(spec)
        if not hashes or int(audited.get("rows", 0)) <= 0:
            raise ValueError(
                f"forbidden-source registry tree {spec.task}/{spec.label} is empty"
            )
        source: dict[str, object] = {
            "task": audited["task"],
            "label": audited["label"],
            "registered_source_locator": os.path.realpath(os.path.abspath(spec.path)),
            "files": audited["files"],
            "file_manifest_sha256": audited["file_manifest_sha256"],
            "rows": audited["rows"],
            "unique_instance_hashes": audited["unique_instance_hashes"],
        }
        source["source_identity_sha256"] = forbidden_source_manifest_identity(source)[
            "source_identity_sha256"
        ]
        sources.append(source)
    return tasks, sources


def build_registry(
    specs: Sequence[DatasetSpec],
    *,
    expected_tasks: Sequence[str],
) -> dict[str, object]:
    """Audit and freeze the exact required task/label source-tree matrix."""

    tasks, sources = _audit_source_records(specs, expected_tasks=expected_tasks)

    core: dict[str, object] = {
        "schema_version": FORBIDDEN_SOURCE_REGISTRY_SCHEMA_VERSION,
        "protocol": FORBIDDEN_SOURCE_REGISTRY_PROTOCOL,
        "status": FORBIDDEN_SOURCE_REGISTRY_STATUS,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "required_labels": sorted(REQUIRED_FORBIDDEN_SOURCE_LABELS),
        "sources": sources,
        "source_tree_manifest_sha256": sha256_json(
            [forbidden_source_manifest_identity(source) for source in sources]
        ),
    }
    core["registry_sha256"] = sha256_json(core)
    validate_forbidden_source_registry(core, expected_tasks=tasks)
    return core


def _legacy_source_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    """Project the shared source manifest through the established strict validator."""

    projection: dict[str, object] = {
        "schema_version": FORBIDDEN_SOURCE_REGISTRY_SCHEMA_VERSION,
        "protocol": FORBIDDEN_SOURCE_REGISTRY_PROTOCOL,
        "status": FORBIDDEN_SOURCE_REGISTRY_STATUS,
        "created_at_utc": payload.get("created_at_utc"),
        "tasks": payload.get("tasks"),
        "required_labels": payload.get("required_labels"),
        "sources": payload.get("sources"),
        "source_tree_manifest_sha256": payload.get("source_tree_manifest_sha256"),
    }
    projection["registry_sha256"] = sha256_json(projection)
    return projection


def base_sft_source_registry_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = forbidden_source_registry_identity(dict(payload))
    identity: dict[str, Any] = {
        **core,
        "temporal_claim": payload.get("temporal_claim"),
        "iid_generation_receipt": payload.get("iid_generation_receipt"),
        "pre_iid_forbidden_registry_identity": payload.get(
            "pre_iid_forbidden_registry_identity"
        ),
        "pre_iid_provenance_reason": payload.get("pre_iid_provenance_reason"),
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def validate_base_sft_source_registry_identity(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "registry_sha256",
        "source_tree_manifest_sha256",
        "tasks",
        "required_labels",
        "source_count",
        "sources_by_task_label",
        "temporal_claim",
        "iid_generation_receipt",
        "pre_iid_forbidden_registry_identity",
        "pre_iid_provenance_reason",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("base-SFT source-registry identity schema is invalid")
    unsigned = dict(value)
    stored = unsigned.pop("identity_sha256", None)
    if (
        value.get("schema_version") != BASE_SFT_SCHEMA_VERSION
        or value.get("protocol") != BASE_SFT_PROTOCOL
        or value.get("status") != BASE_SFT_STATUS
        or value.get("temporal_claim") != BASE_SFT_TEMPORAL_CLAIM
        or value.get("pre_iid_forbidden_registry_identity")
        is not BASE_SFT_PRE_IID_REGISTRY_IDENTITY
        or value.get("pre_iid_provenance_reason") != BASE_SFT_PRE_IID_PROVENANCE_REASON
        or value.get("tasks") != sorted(SUPPORTED_BENCHMARK_TASKS)
        or value.get("required_labels") != sorted(REQUIRED_FORBIDDEN_SOURCE_LABELS)
        or value.get("source_count")
        != len(SUPPORTED_BENCHMARK_TASKS) * len(REQUIRED_FORBIDDEN_SOURCE_LABELS)
        or any(
            not isinstance(value.get(field), str)
            or _SHA256_RE.fullmatch(value[field]) is None
            for field in ("registry_sha256", "source_tree_manifest_sha256")
        )
        or stored != sha256_json(unsigned)
    ):
        raise ValueError("base-SFT source-registry identity is invalid")
    generation = validate_iid_generation_receipt_identity(
        value.get("iid_generation_receipt"), reopen=False
    )
    if (
        generation is None
        or generation.get("receipt_schema_version") != 1
        or generation.get("provenance_claim") != LEGACY_IID_PROVENANCE_CLAIM
        or generation.get("pre_iid_forbidden_registry_identity") is not None
    ):
        raise ValueError("base-SFT source registry lacks honest legacy IID provenance")
    sources = value.get("sources_by_task_label")
    if not isinstance(sources, dict) or tuple(sources) != tuple(
        sorted(SUPPORTED_BENCHMARK_TASKS)
    ):
        raise ValueError("base-SFT source-registry identity lacks five task sources")
    for task in sorted(SUPPORTED_BENCHMARK_TASKS):
        task_sources = sources.get(task)
        if not isinstance(task_sources, dict) or set(task_sources) != set(
            REQUIRED_FORBIDDEN_SOURCE_LABELS
        ):
            raise ValueError(f"base-SFT source-registry identity lacks {task} labels")
    return dict(value)


def build_base_sft_registry(
    specs: Sequence[DatasetSpec],
    *,
    expected_tasks: Sequence[str],
    iid_generation_receipt: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the honest post-IID registry used only by specialist SFT."""

    tasks, sources = _audit_source_records(specs, expected_tasks=expected_tasks)
    generation = iid_generation_receipt_identity(iid_generation_receipt)
    if (
        generation.get("receipt_schema_version") != 1
        or generation.get("provenance_claim") != LEGACY_IID_PROVENANCE_CLAIM
        or generation.get("pre_iid_forbidden_registry_identity") is not None
    ):
        raise ValueError(
            "base-SFT post-IID registry requires the retained legacy schema-v1 IID receipt"
        )
    core: dict[str, Any] = {
        "schema_version": BASE_SFT_SCHEMA_VERSION,
        "protocol": BASE_SFT_PROTOCOL,
        "status": BASE_SFT_STATUS,
        "created_at_utc": created_at_utc
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": tasks,
        "required_labels": sorted(REQUIRED_FORBIDDEN_SOURCE_LABELS),
        "sources": sources,
        "source_tree_manifest_sha256": sha256_json(
            [forbidden_source_manifest_identity(source) for source in sources]
        ),
        "temporal_claim": BASE_SFT_TEMPORAL_CLAIM,
        "iid_generation_receipt": generation,
        "pre_iid_forbidden_registry_identity": BASE_SFT_PRE_IID_REGISTRY_IDENTITY,
        "pre_iid_provenance_reason": BASE_SFT_PRE_IID_PROVENANCE_REASON,
    }
    core["registry_sha256"] = sha256_json(core)
    validate_base_sft_source_registry(core, expected_tasks=tasks)
    return core


def _write_create_once_read_only(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a canonical receipt without an overwrite race."""

    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite frozen forbidden-source registry: {destination}"
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


def validate_base_sft_source_registry(
    payload: object,
    *,
    expected_tasks: Sequence[str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "created_at_utc",
        "tasks",
        "required_labels",
        "sources",
        "source_tree_manifest_sha256",
        "temporal_claim",
        "iid_generation_receipt",
        "pre_iid_forbidden_registry_identity",
        "pre_iid_provenance_reason",
        "registry_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("base-SFT post-IID source registry schema is invalid")
    unsigned = dict(payload)
    stored = unsigned.pop("registry_sha256", None)
    if (
        stored != sha256_json(unsigned)
        or payload.get("schema_version") != BASE_SFT_SCHEMA_VERSION
        or payload.get("protocol") != BASE_SFT_PROTOCOL
        or payload.get("status") != BASE_SFT_STATUS
        or payload.get("temporal_claim") != BASE_SFT_TEMPORAL_CLAIM
        or payload.get("pre_iid_forbidden_registry_identity")
        is not BASE_SFT_PRE_IID_REGISTRY_IDENTITY
        or payload.get("pre_iid_provenance_reason") != BASE_SFT_PRE_IID_PROVENANCE_REASON
    ):
        raise ValueError("base-SFT post-IID source registry contract is invalid")
    legacy = _legacy_source_projection(payload)
    validate_forbidden_source_registry(legacy, expected_tasks=expected_tasks)
    generation = validate_iid_generation_receipt_identity(
        payload.get("iid_generation_receipt"), reopen=True
    )
    if (
        generation is None
        or generation.get("receipt_schema_version") != 1
        or generation.get("provenance_claim") != LEGACY_IID_PROVENANCE_CLAIM
        or generation.get("pre_iid_forbidden_registry_identity") is not None
    ):
        raise ValueError("base-SFT registry IID provenance is not the retained legacy receipt")
    return validate_base_sft_source_registry_identity(
        base_sft_source_registry_identity(payload)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_dataset_spec,
        help="Required once per TASK,LABEL,PATH data tree.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=SUPPORTED_BENCHMARK_TASKS,
        help="Expected task; defaults to the complete five-task publication suite.",
    )
    parser.add_argument(
        "--base-sft-post-iid-generation-receipt",
        type=Path,
        help=(
            "Emit the distinct post-IID/pre-specialist-SFT registry and bind the "
            "retained IID generation receipt."
        ),
    )
    parser.add_argument("--created-at-utc")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        raise FileExistsError(
            "refusing to overwrite frozen forbidden-source registry: "
            f"{os.path.abspath(args.output)}"
        )
    expected_tasks = args.task or SUPPORTED_BENCHMARK_TASKS
    if args.base_sft_post_iid_generation_receipt is None:
        if args.created_at_utc is not None:
            parser.error("--created-at-utc is available only for the base-SFT registry mode")
        registry = build_registry(args.source, expected_tasks=expected_tasks)
    else:
        registry = build_base_sft_registry(
            args.source,
            expected_tasks=expected_tasks,
            iid_generation_receipt=args.base_sft_post_iid_generation_receipt,
            created_at_utc=args.created_at_utc,
        )
    if args.base_sft_post_iid_generation_receipt is not None:
        _write_create_once_read_only(Path(args.output), registry)
        validate_base_sft_source_registry(registry, expected_tasks=expected_tasks)
    else:
        _write_json_atomic(args.output, registry)
    print(f"frozen forbidden-source registry: {os.path.abspath(args.output)}")
    print(f"registry_sha256: {registry['registry_sha256']}")
    print(
        "source_tree_manifest_sha256: "
        f"{registry['source_tree_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
