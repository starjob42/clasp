#!/usr/bin/env python3
"""Fail-closed paired comparison of two CLASP development artifacts.

This utility is intentionally narrower than :mod:`scripts.aggregate_sota`.  It compares
exactly two completed current-schema evaluator artifacts, binds the exact input bytes,
and emits descriptive *development-only* evidence.  It cannot create a confirmation or
SOTA receipt.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clasp.data.benchmark import (
    benchmark_row_sha256,
    reference_output_sha256,
    semantic_instance_sha256,
)

COMPARISON_SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BASE_SEED = 20_260_814
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PAIR_KEY_FIELDS = ("task", "semantic_instance_sha256", "per_example_seed")
SUPPORTED_TASKS = frozenset({"tsp", "cvrp", "op", "mis", "jssp"})
STRICT_VERIFIER_NAME = "clasp_strict_shared_verifier"
_HEX_DIGITS = frozenset("0123456789abcdef")
_COMPARATOR_FILE = os.path.abspath(__file__)

PairKey = tuple[str, str, int]


class DevelopmentComparisonError(ValueError):
    """Raised when inputs cannot support an exact development-only comparison."""


@dataclass(frozen=True)
class BinarySnapshot:
    path: str
    realpath: str
    payload: bytes
    sha256: str
    size_bytes: int

    def identity(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "realpath": self.realpath,
            "basename": os.path.basename(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    key: PairKey
    task: str
    split: str
    example_index: int
    source_row_locator: dict[str, Any]
    instance_sha256: str
    semantic_instance_sha256: str
    benchmark_row_sha256: str
    reference_output_sha256: str
    transformed_example_sha256: str
    sampling_seed: int
    protocol_fingerprint: str
    strict_verifier: dict[str, Any]
    candidate_budget: int
    pass_at_k: bool
    candidate_feasibility: float
    best_feasible_gap: float | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class LoadedArtifact:
    artifact_type: str
    fingerprint_field: str
    fingerprint: str
    record_self_hash_field: str
    protocol_fingerprint: str
    strict_verifier: dict[str, Any]
    candidate_budget: int
    records: dict[PairKey, ArtifactRecord]
    details_snapshot: BinarySnapshot
    companion_snapshot: BinarySnapshot
    companion_self_hash_field: str


def _canonical_json_bytes(payload: Any) -> bytes:
    """Match the canonical representation used by current evaluator self-hashes."""

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise DevelopmentComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _read_binary_snapshot(path: str, *, label: str) -> BinarySnapshot:
    absolute = os.path.abspath(path)
    try:
        path_before = os.stat(absolute)
    except OSError as exc:
        raise DevelopmentComparisonError(f"cannot stat {label}: {absolute}: {exc}") from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise DevelopmentComparisonError(f"{label} is not a regular file: {absolute}")
    try:
        with open(absolute, "rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            payload = handle.read()
            descriptor_after = os.fstat(handle.fileno())
        path_after = os.stat(absolute)
    except OSError as exc:
        raise DevelopmentComparisonError(f"cannot snapshot {label}: {absolute}: {exc}") from exc
    signatures = {
        _stat_signature(path_before),
        _stat_signature(descriptor_before),
        _stat_signature(descriptor_after),
        _stat_signature(path_after),
    }
    if len(signatures) != 1 or len(payload) != descriptor_after.st_size:
        raise DevelopmentComparisonError(f"{label} changed while being snapshotted: {absolute}")
    return BinarySnapshot(
        path=absolute,
        realpath=os.path.realpath(absolute),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _reject_constant(value: str) -> None:
    raise DevelopmentComparisonError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentComparisonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DevelopmentComparisonError(f"{label} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise DevelopmentComparisonError(f"{label} is malformed JSON: {exc}") from exc


def _decode_jsonl(snapshot: BinarySnapshot, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(snapshot.payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = _decode_json(raw_line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise DevelopmentComparisonError(f"{label}:{line_number} must contain one JSON object")
        records.append(value)
    if not records:
        raise DevelopmentComparisonError(f"{label} contains no detail records")
    return records


def _decode_json_object(snapshot: BinarySnapshot, *, label: str) -> dict[str, Any]:
    value = _decode_json(snapshot.payload, label=label)
    if not isinstance(value, dict):
        raise DevelopmentComparisonError(f"{label} must contain one JSON object")
    return value


def _validate_self_hash(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    stored = _require_sha256(payload.get(field), label=f"{label}.{field}")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    observed = _sha256_json(unsigned)
    if stored != observed:
        raise DevelopmentComparisonError(
            f"{label}.{field} mismatch: stored={stored}, observed={observed}"
        )
    return stored


def _protocol_fingerprint_material(material: Mapping[str, Any]) -> dict[str, Any]:
    protocol = {
        key: value
        for key, value in material.items()
        if key not in {"model", "training_run_contract", "training_run_contract_sha256"}
    }
    model = material.get("model")
    if isinstance(model, dict):
        protocol["tokenizer"] = {
            "artifact": model.get("tokenizer_artifact"),
            "loaded": model.get("loaded_tokenizer"),
        }
    selection = protocol.get("selection")
    if isinstance(selection, dict):
        shared_selection = dict(selection)
        shared_selection.pop("checkpoint_selection_receipt", None)
        shared_selection.pop("checkpoint_selection_binding", None)
        shared_selection.pop("campaign_run_binding", None)
        protocol["selection"] = shared_selection
    return protocol


def _example_sampling_seed(global_seed: int, record: ArtifactRecord) -> int:
    material = {
        "schema_version": 1,
        "global_seed": int(global_seed),
        "task": record.task,
        "split": record.split,
        "example_index": record.example_index,
    }
    raw_value = int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(material)).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return int(raw_value % (2**63 - 1))


def _artifact_type(record: Mapping[str, Any], *, label: str) -> tuple[str, str, str, str]:
    specs = {
        "native_raw": (
            "experiment_fingerprint",
            "record_sha256",
            "protocol_fingerprint",
        ),
        "strict_rescore_raw": (
            "rescore_fingerprint",
            "rescore_record_sha256",
            "source_protocol_fingerprint",
        ),
        "repair": (
            "repair_fingerprint",
            "repair_record_sha256",
            "protocol_fingerprint",
        ),
    }
    present = [name for name, spec in specs.items() if record.get(spec[0])]
    if len(present) != 1:
        raise DevelopmentComparisonError(
            f"{label} must identify exactly one native-raw, strict-rescore-raw, or repair "
            f"schema; observed={present}"
        )
    name = present[0]
    fingerprint_field, self_hash_field, protocol_field = specs[name]
    return name, fingerprint_field, self_hash_field, protocol_field


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise DevelopmentComparisonError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DevelopmentComparisonError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise DevelopmentComparisonError(f"{label} must be a finite number")
    return number


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DevelopmentComparisonError(f"{label} must be a nonnegative integer")
    return int(value)


def _numbers_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _repair_metric_view(record: dict[str, Any], *, label: str) -> dict[str, Any]:
    repaired_view = record.get("repaired_strict_view")
    raw_record = record.get("raw_record")
    if not isinstance(repaired_view, dict) or not isinstance(raw_record, dict):
        raise DevelopmentComparisonError(f"{label} repair record has incomplete strict views")
    _validate_self_hash(raw_record, "record_sha256", label=f"{label}.raw_record")
    expected_raw_hash = _sha256_json(raw_record)
    if record.get("raw_record_sha256") != expected_raw_hash:
        raise DevelopmentComparisonError(f"{label}.raw_record_sha256 mismatch")
    wrapped_candidates = record.get("candidates")
    view_candidates = repaired_view.get("candidates")
    if not isinstance(wrapped_candidates, list) or not isinstance(view_candidates, list):
        raise DevelopmentComparisonError(f"{label} repair candidates are incomplete")
    if len(wrapped_candidates) != len(view_candidates):
        raise DevelopmentComparisonError(f"{label} repair candidate views differ in length")
    for index, (wrapped, view_candidate) in enumerate(
        zip(wrapped_candidates, view_candidates, strict=True)
    ):
        if not isinstance(wrapped, dict) or not isinstance(view_candidate, dict):
            raise DevelopmentComparisonError(f"{label}.candidates[{index}] is invalid")
        if wrapped.get("repaired_metrics") != view_candidate:
            raise DevelopmentComparisonError(
                f"{label}.candidates[{index}].repaired_metrics disagrees with repaired view"
            )
        for field, expected in view_candidate.items():
            if wrapped.get(field) != expected:
                raise DevelopmentComparisonError(
                    f"{label}.candidates[{index}].{field} disagrees with repaired view"
                )
    for field, expected in repaired_view.items():
        if field != "candidates" and record.get(field) != expected:
            raise DevelopmentComparisonError(f"{label}.{field} disagrees with repaired_strict_view")
    return repaired_view


def _validate_headline_view(
    view: Mapping[str, Any],
    *,
    expected_k: int | None,
    label: str,
) -> tuple[int, bool, float, float | None]:
    candidates = view.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DevelopmentComparisonError(f"{label} has no candidate array")
    k = _nonnegative_int(view.get("num_return_sequences"), label=f"{label}.num_return_sequences")
    if k == 0 or k != len(candidates):
        raise DevelopmentComparisonError(f"{label} candidate budget disagrees with candidates")
    if expected_k is not None and k != expected_k:
        raise DevelopmentComparisonError(
            f"{label} candidate budget changed within artifact: expected {expected_k}, got {k}"
        )

    feasible_gaps: list[tuple[int, float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise DevelopmentComparisonError(f"{label}.candidates[{index}] is not an object")
        if candidate.get("candidate_index") != index:
            raise DevelopmentComparisonError(
                f"{label}.candidates[{index}] has a mismatched candidate_index"
            )
        feasible = candidate.get("is_feasible")
        if not isinstance(feasible, bool):
            raise DevelopmentComparisonError(
                f"{label}.candidates[{index}].is_feasible is not boolean"
            )
        gap = candidate.get("gap_to_ground_truth")
        if feasible:
            parsed_gap = _finite_float(gap, label=f"{label}.candidates[{index}].gap")
            if parsed_gap < 0.0:
                raise DevelopmentComparisonError(
                    f"{label}.candidates[{index}] has a negative conditional gap"
                )
            feasible_gaps.append((index, parsed_gap))
        elif gap is not None:
            raise DevelopmentComparisonError(
                f"{label}.candidates[{index}] is infeasible but carries a gap"
            )

    feasible_count = len(feasible_gaps)
    expected_pass = feasible_count > 0
    expected_rate = feasible_count / k
    expected_gap = min((value for _index, value in feasible_gaps), default=None)
    if view.get("feasible_candidates") != feasible_count:
        raise DevelopmentComparisonError(f"{label}.feasible_candidates is inconsistent")
    if not _numbers_equal(view.get("candidate_feasibility_rate"), expected_rate):
        raise DevelopmentComparisonError(f"{label}.candidate_feasibility_rate is inconsistent")
    if view.get("best_of_n_feasible") is not expected_pass:
        raise DevelopmentComparisonError(f"{label}.best_of_n_feasible is inconsistent")
    if not _numbers_equal(view.get("best_feasible_gap"), expected_gap):
        raise DevelopmentComparisonError(f"{label}.best_feasible_gap is inconsistent")
    best_index = view.get("best_candidate_index")
    if not expected_pass:
        if best_index is not None:
            raise DevelopmentComparisonError(f"{label}.best_candidate_index is inconsistent")
    else:
        if isinstance(best_index, bool) or not isinstance(best_index, int):
            raise DevelopmentComparisonError(f"{label}.best_candidate_index is invalid")
        if not any(
            index == best_index and _numbers_equal(gap, expected_gap)
            for index, gap in feasible_gaps
        ):
            raise DevelopmentComparisonError(
                f"{label}.best_candidate_index does not select a minimum-gap candidate"
            )
    return k, expected_pass, float(expected_rate), expected_gap


def _validate_record(
    record: dict[str, Any],
    *,
    record_index: int,
    expected_type: str | None,
    expected_fingerprint: str | None,
    expected_protocol: str | None,
    expected_verifier: dict[str, Any] | None,
    expected_k: int | None,
) -> tuple[ArtifactRecord, str, str, str]:
    label = f"detail record {record_index}"
    artifact_type, fingerprint_field, self_hash_field, protocol_field = _artifact_type(
        record, label=label
    )
    if expected_type is not None and artifact_type != expected_type:
        raise DevelopmentComparisonError(
            f"{label} mixes artifact types: expected {expected_type}, got {artifact_type}"
        )
    fingerprint = _require_sha256(
        record.get(fingerprint_field), label=f"{label}.{fingerprint_field}"
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise DevelopmentComparisonError(f"{label} fingerprint changed within artifact")
    _validate_self_hash(record, self_hash_field, label=label)

    protocol = _require_sha256(record.get(protocol_field), label=f"{label}.{protocol_field}")
    if expected_protocol is not None and protocol != expected_protocol:
        raise DevelopmentComparisonError(f"{label} protocol changed within artifact")
    verifier = record.get("strict_verifier")
    if not isinstance(verifier, dict) or verifier.get("name") != STRICT_VERIFIER_NAME:
        raise DevelopmentComparisonError(f"{label} does not identify the strict shared verifier")
    _require_sha256(
        verifier.get("implementation_manifest_sha256"),
        label=f"{label}.strict_verifier.implementation_manifest_sha256",
    )
    if expected_verifier is not None and verifier != expected_verifier:
        raise DevelopmentComparisonError(f"{label} strict verifier changed within artifact")

    task = str(record.get("task", "")).lower()
    split = str(record.get("split", ""))
    if task not in SUPPORTED_TASKS or not split:
        raise DevelopmentComparisonError(f"{label} has an invalid task or split")
    example_index = _nonnegative_int(record.get("example_index"), label=f"{label}.example_index")
    sampling_seed = _nonnegative_int(record.get("sampling_seed"), label=f"{label}.sampling_seed")
    semantic_hash = _require_sha256(
        record.get("semantic_instance_sha256"), label=f"{label}.semantic_instance_sha256"
    )
    hashes = {
        field: _require_sha256(record.get(field), label=f"{label}.{field}")
        for field in (
            "instance_sha256",
            "benchmark_row_sha256",
            "reference_output_sha256",
            "transformed_example_sha256",
        )
    }
    if record.get("instance_json") is None:
        raise DevelopmentComparisonError(f"{label} is missing instance_json")
    if _sha256_json(record["instance_json"]) != hashes["instance_sha256"]:
        raise DevelopmentComparisonError(f"{label}.instance_sha256 is inconsistent")
    row_projection = {
        "instruction": str(record.get("instruction", "")),
        "input": str(record.get("input_problem", "")),
        "output": str(record.get("ground_truth_output", "")),
        "instance_json": record["instance_json"],
    }
    if not all(row_projection[field] for field in ("instruction", "input", "output")):
        raise DevelopmentComparisonError(f"{label} has incomplete transformed row text")
    if benchmark_row_sha256(row_projection) != hashes["benchmark_row_sha256"]:
        raise DevelopmentComparisonError(f"{label}.benchmark_row_sha256 is inconsistent")
    if _sha256_json(row_projection) != hashes["transformed_example_sha256"]:
        raise DevelopmentComparisonError(f"{label}.transformed_example_sha256 is inconsistent")
    if reference_output_sha256(row_projection["output"]) != hashes["reference_output_sha256"]:
        raise DevelopmentComparisonError(f"{label}.reference_output_sha256 is inconsistent")
    if semantic_instance_sha256(task, row_projection, require_structured=True) != semantic_hash:
        raise DevelopmentComparisonError(f"{label}.semantic_instance_sha256 is inconsistent")

    locator = record.get("source_row_locator")
    if not isinstance(locator, dict):
        raise DevelopmentComparisonError(f"{label} is missing source_row_locator")
    expected_locator_projection = {
        "task": task,
        "split": split,
        "merged_dataset_index": example_index,
    }
    observed_locator_projection = {
        "task": locator.get("task"),
        "split": locator.get("split"),
        "merged_dataset_index": locator.get("merged_dataset_index"),
    }
    if observed_locator_projection != expected_locator_projection:
        raise DevelopmentComparisonError(f"{label}.source_row_locator is inconsistent")
    _require_sha256(
        locator.get("data_manifest_sha256"),
        label=f"{label}.source_row_locator.data_manifest_sha256",
    )

    metric_view = _repair_metric_view(record, label=label) if artifact_type == "repair" else record
    k, pass_at_k, candidate_feasibility, best_gap = _validate_headline_view(
        metric_view,
        expected_k=expected_k,
        label=f"{label}.metric_view",
    )
    key = (task, semantic_hash, sampling_seed)
    normalized = ArtifactRecord(
        key=key,
        task=task,
        split=split,
        example_index=example_index,
        source_row_locator=dict(locator),
        instance_sha256=hashes["instance_sha256"],
        semantic_instance_sha256=semantic_hash,
        benchmark_row_sha256=hashes["benchmark_row_sha256"],
        reference_output_sha256=hashes["reference_output_sha256"],
        transformed_example_sha256=hashes["transformed_example_sha256"],
        sampling_seed=sampling_seed,
        protocol_fingerprint=protocol,
        strict_verifier=dict(verifier),
        candidate_budget=k,
        pass_at_k=pass_at_k,
        candidate_feasibility=candidate_feasibility,
        best_feasible_gap=best_gap,
        raw=record,
    )
    return normalized, artifact_type, fingerprint_field, self_hash_field


def _infer_companion_path(details_path: str, artifact_type: str) -> str:
    path = Path(details_path)
    if artifact_type == "native_raw":
        prefix = "run_details__"
        if not path.name.startswith(prefix):
            raise DevelopmentComparisonError(
                "native raw companion could not be inferred; pass --left-companion/"
                "--right-companion explicitly"
            )
        return str(path.with_name("run_config__" + path.name[len(prefix) :]).with_suffix(".json"))
    return str(path.with_name(path.stem + "__summary.json"))


def _validate_details_identity(
    companion: Mapping[str, Any],
    details: BinarySnapshot,
    *,
    record_count: int,
    label: str,
) -> None:
    identity = companion.get("details_artifact")
    expected = {
        "basename": os.path.basename(details.path),
        "sha256": details.sha256,
        "size_bytes": details.size_bytes,
        "record_count": record_count,
    }
    if identity != expected:
        raise DevelopmentComparisonError(
            f"{label}.details_artifact does not bind the exact supplied JSONL bytes"
        )


def _validate_development_provenance(
    provenance: Any,
    *,
    expected_protocol: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise DevelopmentComparisonError(f"{label} is missing experiment provenance")
    observed_experiment = _sha256_json(provenance)
    observed_protocol = _sha256_json(_protocol_fingerprint_material(provenance))
    if observed_protocol != expected_protocol:
        raise DevelopmentComparisonError(f"{label} protocol fingerprint is inconsistent")
    selection = provenance.get("selection")
    if not isinstance(selection, dict) or selection.get("evaluation_purpose") != "development":
        raise DevelopmentComparisonError(
            f"{label} is not explicitly scoped to evaluation_purpose=development"
        )
    if (
        selection.get("benchmark_manifest") is not None
        or selection.get("campaign_manifest") is not None
    ):
        raise DevelopmentComparisonError(f"{label} improperly binds confirmation assets")
    return {"experiment_fingerprint": observed_experiment, "selection": selection}


def _validate_companion(
    companion: dict[str, Any],
    *,
    artifact_type: str,
    fingerprint: str,
    protocol: str,
    verifier: dict[str, Any],
    k: int,
    records: Sequence[ArtifactRecord],
    details: BinarySnapshot,
    label: str,
) -> str:
    if artifact_type == "native_raw":
        self_hash_field = "run_config_sha256"
        _validate_self_hash(companion, self_hash_field, label=label)
        if companion.get("run_status") != "complete":
            raise DevelopmentComparisonError(f"{label} does not describe a completed run")
        if companion.get("evaluation_purpose") != "development":
            raise DevelopmentComparisonError(f"{label} is not a development run")
        provenance_info = _validate_development_provenance(
            companion.get("experiment_provenance"),
            expected_protocol=protocol,
            label=label,
        )
        if companion.get("experiment_fingerprint") != fingerprint:
            raise DevelopmentComparisonError(f"{label} experiment fingerprint mismatch")
        if provenance_info["experiment_fingerprint"] != fingerprint:
            raise DevelopmentComparisonError(f"{label} experiment provenance mismatch")
        if companion.get("protocol_fingerprint") != protocol:
            raise DevelopmentComparisonError(f"{label} protocol fingerprint mismatch")
        if companion.get("verifier_identity") != verifier:
            raise DevelopmentComparisonError(f"{label} verifier identity mismatch")
        generation = companion.get("generation")
    else:
        self_hash_field = "summary_sha256"
        _validate_self_hash(companion, self_hash_field, label=label)
        lane = companion.get("system_lane")
        expected_lane = {
            "repair": "deterministic_repair_and_improve_hybrid",
            "strict_rescore_raw": "strict_rescore_raw_generation",
        }[artifact_type]
        if lane != expected_lane:
            raise DevelopmentComparisonError(f"{label} has the wrong derived system lane")
        fingerprint_key = {
            "repair": "repair_fingerprint",
            "strict_rescore_raw": "rescore_fingerprint",
        }[artifact_type]
        if companion.get(fingerprint_key) != fingerprint:
            raise DevelopmentComparisonError(f"{label} derived fingerprint mismatch")
        if companion.get("source_protocol_fingerprint") != protocol:
            raise DevelopmentComparisonError(f"{label} source protocol mismatch")
        provenance_info = _validate_development_provenance(
            companion.get("source_experiment_provenance"),
            expected_protocol=protocol,
            label=label,
        )
        if any(
            companion.get(field) is not None
            for field in (
                "source_campaign_manifest",
                "source_campaign_run_binding",
                "source_confirmation_attempt_claim",
            )
        ):
            raise DevelopmentComparisonError(f"{label} derives from confirmation evidence")
        if companion.get("record_count") != len(records):
            raise DevelopmentComparisonError(f"{label}.record_count disagrees with details")
        if companion.get("candidate_count") not in (None, len(records) * k):
            raise DevelopmentComparisonError(f"{label}.candidate_count disagrees with details")
        generation = companion.get("source_experiment_provenance", {}).get("generation")

    _validate_details_identity(
        companion,
        details,
        record_count=len(records),
        label=label,
    )
    if not isinstance(generation, dict) or generation.get("seed_policy") != "per_example_sha256_v1":
        raise DevelopmentComparisonError(f"{label} does not use per-example sampling seeds")
    if generation.get("num_return_sequences") != k:
        raise DevelopmentComparisonError(f"{label} candidate budget disagrees with details")
    global_seed = _nonnegative_int(generation.get("seed"), label=f"{label}.generation.seed")
    provenance = (
        companion.get("experiment_provenance")
        if artifact_type == "native_raw"
        else companion.get("source_experiment_provenance")
    )
    assert isinstance(provenance, dict)
    data = provenance.get("data")
    if not isinstance(data, dict):
        raise DevelopmentComparisonError(f"{label} has no data provenance")
    data_manifest = _require_sha256(
        data.get("manifest_sha256"), label=f"{label}.data.manifest_sha256"
    )
    data_files = data.get("files")
    if not isinstance(data_files, list) or _sha256_json(data_files) != data_manifest:
        raise DevelopmentComparisonError(f"{label} data manifest is internally inconsistent")
    for record in records:
        if record.source_row_locator.get("data_manifest_sha256") != data_manifest:
            raise DevelopmentComparisonError(
                f"{label} data manifest disagrees with a source_row_locator"
            )
        if record.sampling_seed != _example_sampling_seed(global_seed, record):
            raise DevelopmentComparisonError(
                f"{label} contains a sampling_seed inconsistent with the declared seed policy"
            )
    tasks = sorted({record.task for record in records})
    splits = sorted({record.split for record in records})
    selection = provenance_info["selection"]
    if sorted(selection.get("tasks", [])) != tasks or splits != [str(selection.get("split", ""))]:
        raise DevelopmentComparisonError(f"{label} selection disagrees with detail rows")
    return self_hash_field


def _load_artifact(
    details_path: str,
    companion_path: str | None,
    *,
    label: str,
) -> LoadedArtifact:
    details = _read_binary_snapshot(details_path, label=f"{label} details")
    raw_records = _decode_jsonl(details, label=f"{label} details")
    records: dict[PairKey, ArtifactRecord] = {}
    artifact_type = fingerprint_field = self_hash_field = ""
    fingerprint = protocol = ""
    verifier: dict[str, Any] | None = None
    k: int | None = None
    for index, raw_record in enumerate(raw_records):
        normalized, observed_type, observed_fingerprint_field, observed_self_hash = (
            _validate_record(
                raw_record,
                record_index=index,
                expected_type=artifact_type or None,
                expected_fingerprint=fingerprint or None,
                expected_protocol=protocol or None,
                expected_verifier=verifier,
                expected_k=k,
            )
        )
        observed_fingerprint = str(raw_record[observed_fingerprint_field])
        if not artifact_type:
            artifact_type = observed_type
            fingerprint_field = observed_fingerprint_field
            self_hash_field = observed_self_hash
            fingerprint = observed_fingerprint
            protocol = normalized.protocol_fingerprint
            verifier = normalized.strict_verifier
            k = normalized.candidate_budget
        if normalized.key in records:
            raise DevelopmentComparisonError(
                f"{label} contains duplicate pairing key {normalized.key}"
            )
        records[normalized.key] = normalized
    assert verifier is not None and k is not None

    resolved_companion = companion_path or _infer_companion_path(details.path, artifact_type)
    companion_snapshot = _read_binary_snapshot(resolved_companion, label=f"{label} companion")
    companion = _decode_json_object(companion_snapshot, label=f"{label} companion")
    companion_self_hash = _validate_companion(
        companion,
        artifact_type=artifact_type,
        fingerprint=fingerprint,
        protocol=protocol,
        verifier=verifier,
        k=k,
        records=list(records.values()),
        details=details,
        label=f"{label} companion",
    )
    return LoadedArtifact(
        artifact_type=artifact_type,
        fingerprint_field=fingerprint_field,
        fingerprint=fingerprint,
        record_self_hash_field=self_hash_field,
        protocol_fingerprint=protocol,
        strict_verifier=verifier,
        candidate_budget=k,
        records=records,
        details_snapshot=details,
        companion_snapshot=companion_snapshot,
        companion_self_hash_field=companion_self_hash,
    )


def _key_for_json(key: PairKey) -> dict[str, Any]:
    return {
        "task": key[0],
        "semantic_instance_sha256": key[1],
        "per_example_seed": key[2],
    }


def _verify_native_repair_linkage(
    left: LoadedArtifact,
    right: LoadedArtifact,
) -> str:
    checks: list[tuple[LoadedArtifact, LoadedArtifact]] = []
    if left.artifact_type == "native_raw" and right.artifact_type == "repair":
        checks.append((left, right))
    if right.artifact_type == "native_raw" and left.artifact_type == "repair":
        checks.append((right, left))
    if not checks:
        return "not_applicable"
    statuses: list[str] = []
    for native, repair in checks:
        repair_sources = {
            str(record.raw.get("source_experiment_fingerprint", ""))
            for record in repair.records.values()
        }
        if repair_sources != {native.fingerprint}:
            statuses.append("not_applicable_different_source_experiments")
            continue
        for key in sorted(native.records):
            embedded = repair.records[key].raw.get("raw_record")
            if embedded != native.records[key].raw:
                raise DevelopmentComparisonError(
                    "native/repair artifacts claim the same source experiment but the repair "
                    f"does not embed the exact native record at {_key_for_json(key)}"
                )
        statuses.append("verified_exact_embedded_native_records")
    return statuses[0] if len(set(statuses)) == 1 else "+".join(statuses)


def _validate_pairing(left: LoadedArtifact, right: LoadedArtifact) -> dict[str, Any]:
    if left.candidate_budget != right.candidate_budget:
        raise DevelopmentComparisonError("paired artifacts have different candidate budgets")
    if left.protocol_fingerprint != right.protocol_fingerprint:
        raise DevelopmentComparisonError("paired artifacts have different protocol fingerprints")
    if left.strict_verifier != right.strict_verifier:
        raise DevelopmentComparisonError("paired artifacts use different strict verifiers")
    left_keys = set(left.records)
    right_keys = set(right.records)
    if left_keys != right_keys:
        left_only = [_key_for_json(key) for key in sorted(left_keys - right_keys)[:8]]
        right_only = [_key_for_json(key) for key in sorted(right_keys - left_keys)[:8]]
        raise DevelopmentComparisonError(
            f"paired key sets differ: left_only={left_only}, right_only={right_only}"
        )
    identity_fields = (
        "split",
        "example_index",
        "source_row_locator",
        "instance_sha256",
        "benchmark_row_sha256",
        "reference_output_sha256",
        "transformed_example_sha256",
    )
    for key in sorted(left_keys):
        left_record = left.records[key]
        right_record = right.records[key]
        for field in identity_fields:
            if getattr(left_record, field) != getattr(right_record, field):
                raise DevelopmentComparisonError(f"paired {field} mismatch at {_key_for_json(key)}")
    linkage = _verify_native_repair_linkage(left, right)
    return {
        "status": "exact",
        "pair_count": len(left_keys),
        "key_fields": list(PAIR_KEY_FIELDS),
        "per_example_seed_source_field": "sampling_seed",
        "protocol_fingerprint": left.protocol_fingerprint,
        "candidate_budget": left.candidate_budget,
        "strict_verifier": left.strict_verifier,
        "source_row_and_content_identity": "exact",
        "native_repair_verbatim_linkage": linkage,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(math.fsum(values) / len(values)) if values else None


def _difference(right: float, left: float) -> float:
    delta = float(right) - float(left)
    return 0.0 if math.isclose(delta, 0.0, rel_tol=1e-12, abs_tol=1e-12) else delta


def _derived_seed(scope: str, metric: str) -> int:
    material = f"{BOOTSTRAP_BASE_SEED}:{scope}:{metric}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _percentile(sorted_values: np.ndarray, probability: float) -> float:
    if sorted_values.size == 1:
        return float(sorted_values[0])
    position = (sorted_values.size - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    scope: str,
    metric: str,
    resampling_unit: str,
) -> dict[str, Any] | None:
    if not deltas:
        return None
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise DevelopmentComparisonError(f"non-finite paired bootstrap input for {scope}/{metric}")
    seed = _derived_seed(scope, metric)
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    max_index_entries = 1_000_000
    batch_size = max(1, min(512, max_index_entries // len(values)))
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(BOOTSTRAP_REPLICATES, start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    estimates.sort()
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    return {
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BOOTSTRAP_BASE_SEED,
        "derived_seed": seed,
        "rng": "numpy.random.PCG64",
        "interval": "paired_percentile",
        "resampling_unit": resampling_unit,
    }


def _exact_two_sided_binomial(first: int, second: int) -> float:
    n = first + second
    if n == 0:
        return 1.0
    observed = min(first, second)
    numerator = 2 * sum(math.comb(n, value) for value in range(observed + 1))
    return float(min(Fraction(1, 1), Fraction(numerator, 1 << n)))


def _sign_test(deltas: Sequence[float], *, lower_is_better: bool) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in deltas)
    negative = sum(value < 0.0 for value in deltas)
    ties = len(deltas) - positive - negative
    return {
        "right_improvements": negative if lower_is_better else positive,
        "right_regressions": positive if lower_is_better else negative,
        "ties": ties,
        "non_tied_pairs": positive + negative,
        "exact_two_sided_p": _exact_two_sided_binomial(positive, negative),
        "null": "equal probability of a positive or negative non-tied paired difference",
    }


def _summarize_pairs(
    pairs: Sequence[tuple[ArtifactRecord, ArtifactRecord]],
    *,
    scope: str,
) -> dict[str, Any]:
    if not pairs:
        raise DevelopmentComparisonError(f"no paired records for {scope}")
    left_pass = [float(left.pass_at_k) for left, _right in pairs]
    right_pass = [float(right.pass_at_k) for _left, right in pairs]
    pass_deltas = [
        _difference(right, left) for left, right in zip(left_pass, right_pass, strict=True)
    ]
    left_only = sum(left.pass_at_k and not right.pass_at_k for left, right in pairs)
    right_only = sum(right.pass_at_k and not left.pass_at_k for left, right in pairs)

    left_candidate = [left.candidate_feasibility for left, _right in pairs]
    right_candidate = [right.candidate_feasibility for _left, right in pairs]
    candidate_deltas = [
        _difference(right, left)
        for left, right in zip(left_candidate, right_candidate, strict=True)
    ]

    left_feasible_gaps = [
        float(record.best_feasible_gap)
        for record, _other in pairs
        if record.best_feasible_gap is not None
    ]
    right_feasible_gaps = [
        float(record.best_feasible_gap)
        for _other, record in pairs
        if record.best_feasible_gap is not None
    ]
    joint_gap_pairs = [
        (float(left.best_feasible_gap), float(right.best_feasible_gap))
        for left, right in pairs
        if left.best_feasible_gap is not None and right.best_feasible_gap is not None
    ]
    gap_deltas = [_difference(right, left) for left, right in joint_gap_pairs]

    return {
        "paired_instances": len(pairs),
        "pass_at_k": {
            "left": _mean(left_pass),
            "right": _mean(right_pass),
            "right_minus_left": _mean(pass_deltas),
            "paired_bootstrap_95pct_ci": _paired_bootstrap_ci(
                pass_deltas,
                scope=scope,
                metric="pass_at_k",
                resampling_unit="paired_instance",
            ),
            "exact_mcnemar": {
                "left_only_success": left_only,
                "right_only_success": right_only,
                "discordant_pairs": left_only + right_only,
                "exact_two_sided_p": _exact_two_sided_binomial(left_only, right_only),
            },
        },
        "candidate_feasibility": {
            "left": _mean(left_candidate),
            "right": _mean(right_candidate),
            "right_minus_left": _mean(candidate_deltas),
            "paired_bootstrap_95pct_ci": _paired_bootstrap_ci(
                candidate_deltas,
                scope=scope,
                metric="candidate_feasibility",
                resampling_unit="paired_instance_candidate_rate",
            ),
            "instance_level_exact_sign_test": _sign_test(candidate_deltas, lower_is_better=False),
            "note": "Inference is paired at the instance level; candidates within an instance "
            "are not treated as independent replicates.",
        },
        "conditional_best_feasible_gap": {
            "left_feasible_instances": len(left_feasible_gaps),
            "right_feasible_instances": len(right_feasible_gaps),
            "left_mean_on_left_feasible_instances": _mean(left_feasible_gaps),
            "right_mean_on_right_feasible_instances": _mean(right_feasible_gaps),
            "jointly_feasible_instances": len(joint_gap_pairs),
            "left_mean_on_jointly_feasible_instances": _mean(
                [left for left, _right in joint_gap_pairs]
            ),
            "right_mean_on_jointly_feasible_instances": _mean(
                [right for _left, right in joint_gap_pairs]
            ),
            "right_minus_left_on_jointly_feasible_instances": _mean(gap_deltas),
            "paired_bootstrap_95pct_ci": _paired_bootstrap_ci(
                gap_deltas,
                scope=scope,
                metric="conditional_best_feasible_gap",
                resampling_unit="jointly_feasible_paired_instance",
            ),
            "jointly_feasible_exact_sign_test": (
                _sign_test(gap_deltas, lower_is_better=True) if gap_deltas else None
            ),
            "note": "The paired gap delta is conditioned on both systems having at least one "
            "strictly feasible candidate; negative right-minus-left is better.",
        },
    }


def _artifact_receipt(artifact: LoadedArtifact) -> dict[str, Any]:
    return {
        "artifact_type": artifact.artifact_type,
        "fingerprint_field": artifact.fingerprint_field,
        "fingerprint": artifact.fingerprint,
        "record_self_hash_field": artifact.record_self_hash_field,
        "protocol_fingerprint": artifact.protocol_fingerprint,
        "candidate_budget": artifact.candidate_budget,
        "record_count": len(artifact.records),
        "tasks": sorted({record.task for record in artifact.records.values()}),
        "details_snapshot": artifact.details_snapshot.identity(),
        "companion_snapshot": {
            **artifact.companion_snapshot.identity(),
            "self_hash_field": artifact.companion_self_hash_field,
        },
    }


def build_receipt(
    left: LoadedArtifact,
    right: LoadedArtifact,
    *,
    left_label: str,
    right_label: str,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    pairing = _validate_pairing(left, right)
    all_pairs = [(left.records[key], right.records[key]) for key in sorted(left.records)]
    tasks = sorted({left_record.task for left_record, _right_record in all_pairs})
    implementation = _read_binary_snapshot(
        _COMPARATOR_FILE, label="comparison implementation"
    ).identity()
    receipt = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_scope": "development_only",
        "claim_eligibility": "not_confirmation_and_not_sota_evidence",
        "comparison_direction": "right_minus_left",
        "labels": {"left": left_label, "right": right_label},
        "artifacts": {
            "left": _artifact_receipt(left),
            "right": _artifact_receipt(right),
        },
        "pairing": pairing,
        "statistics_contract": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
            "bootstrap_rng": "numpy.random.PCG64",
            "bootstrap_interval": "paired_percentile_95pct",
            "candidate_feasibility_inference_unit": "paired_instance_candidate_rate",
            "gap_conditioning": "jointly_best_of_k_feasible_instances",
            "multiple_comparison_adjustment": None,
        },
        "per_task": {
            task: _summarize_pairs(
                [pair for pair in all_pairs if pair[0].task == task],
                scope=f"task:{task}",
            )
            for task in tasks
        },
        "pooled_descriptive_only": _summarize_pairs(all_pairs, scope="pooled"),
        "implementation_snapshot": implementation,
        "limitations": [
            "This receipt is development-only and cannot support a confirmation or SOTA claim.",
            "Per-task analyses are primary; pooled instances are descriptive because tasks may "
            "differ in size and difficulty.",
            "Instance bootstrap intervals quantify paired development-set sampling variation, "
            "not training-seed uncertainty.",
            "No multiplicity correction is applied; exact tests are descriptive diagnostics.",
            "Conditional gap comparisons use only jointly feasible paired instances and must be "
            "read together with pass@K and candidate feasibility.",
        ],
    }
    sealed = dict(receipt)
    sealed["receipt_sha256"] = _sha256_json(receipt)
    return sealed


def _write_create_once(path: str, payload: Mapping[str, Any]) -> None:
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute) or os.curdir
    os.makedirs(parent, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".development_compare.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, absolute)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise DevelopmentComparisonError(
                    f"refusing to overwrite existing comparison receipt: {absolute}"
                ) from exc
            raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_comparison(args: argparse.Namespace) -> str:
    left_label = str(args.left_label).strip()
    right_label = str(args.right_label).strip()
    if not left_label or not right_label or left_label == right_label:
        raise DevelopmentComparisonError("left/right labels must be non-empty and distinct")
    output = os.path.abspath(args.output)
    protected = {
        os.path.realpath(str(path))
        for path in (
            args.left,
            args.right,
            getattr(args, "left_companion", None),
            getattr(args, "right_companion", None),
        )
        if path
    }
    if os.path.realpath(output) in protected:
        raise DevelopmentComparisonError("comparison receipt must not overwrite an input artifact")

    left = _load_artifact(
        str(args.left),
        getattr(args, "left_companion", None),
        label="left",
    )
    right = _load_artifact(
        str(args.right),
        getattr(args, "right_companion", None),
        label="right",
    )
    receipt = build_receipt(
        left,
        right,
        left_label=left_label,
        right_label=right_label,
    )
    _write_create_once(output, receipt)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a self-hashed DEVELOPMENT-ONLY paired comparison receipt for two "
            "current CLASP native/rescored-raw or deterministic-repair detail artifacts."
        )
    )
    parser.add_argument("--left", required=True, help="Left details JSONL.")
    parser.add_argument("--right", required=True, help="Right details JSONL.")
    parser.add_argument(
        "--left-companion",
        default=None,
        help="Left native run_config or repair/rescore summary; inferred when omitted.",
    )
    parser.add_argument(
        "--right-companion",
        default=None,
        help="Right native run_config or repair/rescore summary; inferred when omitted.",
    )
    parser.add_argument("--left-label", required=True, help="Stable display label for left.")
    parser.add_argument("--right-label", required=True, help="Stable display label for right.")
    parser.add_argument("--output", required=True, help="Create-once JSON receipt path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_comparison(args)
    print(f"Development-only paired comparison: {output}")


if __name__ == "__main__":
    main()
