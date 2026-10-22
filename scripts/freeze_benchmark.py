#!/usr/bin/env python3
"""Freeze and verify an independent, leakage-audited CLASP benchmark manifest.

The manifest is deliberately created before inference.  It records exact dataset,
generator, reference-solver, and strict-verifier identities; rejects duplicate or
previously seen instances; and verifies every stored reference solution with the
same strict task evaluator used for model outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from clasp.data.benchmark import (
    NONPUBLICATION_BENCHMARK_MODE,
    PUBLICATION_INSTANCES_PER_TASK,
    PUBLICATION_BENCHMARK_MODE,
    REQUIRED_FORBIDDEN_SOURCE_LABELS,
    SUPPORTED_BENCHMARK_TASKS,
    benchmark_row_sha256,
    forbidden_source_manifest_identity,
    forbidden_source_registry_identity,
    instance_sha256,
    reference_output_sha256,
    validate_forbidden_source_registry,
    validate_forbidden_sources_against_registry,
    validate_frozen_benchmark_manifest,
    validate_reference_evidence_approval_receipt,
    validate_reference_solver_contract_policy,
)

SUPPORTED_TASKS = SUPPORTED_BENCHMARK_TASKS


@dataclass(frozen=True)
class DatasetSpec:
    task: str
    label: str
    path: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_files(path: str) -> tuple[str, list[str]]:
    resolved = os.path.abspath(path)
    if os.path.islink(resolved):
        raise ValueError(f"dataset source root must not be a symbolic link: {path}")
    if os.path.isfile(resolved):
        if not resolved.endswith(".json"):
            raise ValueError(f"dataset file must end in .json: {path}")
        return os.path.dirname(resolved), [resolved]
    if not os.path.isdir(resolved):
        raise FileNotFoundError(path)
    for candidate in Path(resolved).rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"dataset source tree contains a symbolic link: {candidate}")
    files = sorted(
        str(candidate) for candidate in Path(resolved).rglob("*.json") if candidate.is_file()
    )
    if not files:
        raise ValueError(f"dataset path contains no JSON files: {path}")
    return resolved, files


def _iter_json_array(path: str, *, chunk_size: int = 1024 * 1024):
    """Stream a top-level JSON array without loading a 500k-row SFT file."""

    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def compact_and_read() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        compact_and_read()
        while not eof and not buffer.strip():
            compact_and_read()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"dataset must be a top-level JSON list: {path}")
        position += 1
        row_index = 0
        need_comma = False

        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"unterminated top-level JSON list: {path}")
                compact_and_read()

            if need_comma:
                if buffer[position] == "]":
                    position += 1
                    break
                if buffer[position] != ",":
                    raise ValueError(
                        f"malformed dataset JSON {path}: expected comma before row {row_index}"
                    )
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        break
                    if eof:
                        raise ValueError(f"unterminated top-level JSON list: {path}")
                    compact_and_read()
                if buffer[position] == "]":
                    raise ValueError(f"malformed dataset JSON {path}: trailing comma")
            elif buffer[position] == "]":
                position += 1
                break

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"malformed dataset JSON {path}: {exc}") from exc
                    compact_and_read()
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{row_index}")
            yield row_index, value
            row_index += 1
            need_comma = True
            if position > chunk_size:
                compact_and_read()

        trailing = buffer[position:] + handle.read()
        if trailing.strip():
            raise ValueError(f"malformed dataset JSON {path}: data after top-level list")


def _load_contract(path: str, *, label: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} contract must be a non-empty JSON object: {path}")
    return value


def _verify_generation_receipt(receipt: dict[str, object], *, allow_smoke: bool) -> None:
    stored = str(receipt.get("receipt_sha256", ""))
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    observed = _sha256_json(unsigned)
    if not stored or stored != observed:
        raise ValueError(
            f"generation receipt hash mismatch: stored={stored or '<missing>'}, observed={observed}"
        )
    status = receipt.get("status")
    allowed = {"references_attached"}
    if allow_smoke:
        allowed.add("smoke_references_attached")
    if status not in allowed:
        raise ValueError(f"generation receipt status {status!r} is not freeze-eligible")
    if receipt.get("schema_version") != 2:
        raise ValueError("generation receipt must use identity-binding schema version 2")
    if receipt.get("contains_references") is not True:
        raise ValueError("generation receipt does not contain solver references")
    if not allow_smoke and receipt.get("publishable") is not True:
        raise ValueError("generation receipt is not marked publishable")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _objective_matches(task: str, left: object, right: object) -> bool:
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(first) and math.isfinite(second)):
        return False
    if task in {"op", "mis", "jssp"}:
        return (
            abs(first - round(first)) <= 1e-9
            and abs(second - round(second)) <= 1e-9
            and round(first) == round(second)
        )
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9)


def _validate_frozen_reference_evidence(
    *,
    task: str,
    provenance: dict[str, object],
    solver_contract_task: dict[str, object],
    verifier_identity: dict[str, object],
    strict_metrics: dict[str, object],
    runner_code_sha256: str,
    evidence_root: Path,
    approval_receipt_sha256: str,
) -> tuple[str, str, str]:
    """Reject assembled production rows that dropped or altered native evidence."""

    export_record_sha = provenance.get("reference_export_record_sha256")
    evidence = provenance.get("reference_evidence")
    if not _is_sha256(export_record_sha) or not isinstance(evidence, dict):
        raise ValueError(f"missing bound reference evidence for {task}")
    if evidence.get("schema_version") != 1:
        raise ValueError(f"unsupported reference evidence schema for {task}")
    for field in (
        "reference_evidence_record_sha256",
        "reference_evidence_summary_sha256",
        "runner_code_sha256",
        "solver_parameters_sha256",
        "strict_verifier_identity_sha256",
        "strict_verifier_result_sha256",
    ):
        if not _is_sha256(evidence.get(field)):
            raise ValueError(f"invalid reference evidence hash {field} for {task}")
    if evidence.get("runner_protocol") != "clasp-iid-reference-evidence-v1":
        raise ValueError(f"unsupported reference evidence runner for {task}")
    if evidence.get("runner_code_sha256") != runner_code_sha256:
        raise ValueError(f"reference evidence runner source mismatch for {task}")
    if provenance.get(
        "reference_evidence_approval_receipt_sha256"
    ) != approval_receipt_sha256:
        raise ValueError(f"reference evidence approval receipt mismatch for {task}")
    parameters = solver_contract_task.get("parameters")
    if evidence.get("solver_parameters") != parameters:
        raise ValueError(f"reference evidence parameters mismatch for {task}")
    if evidence.get("solver_parameters_sha256") != _sha256_json(parameters):
        raise ValueError(f"reference evidence parameter hash mismatch for {task}")
    runner_contract = solver_contract_task.get("evidence_runner")
    if not isinstance(runner_contract, dict):
        raise ValueError(f"solver contract has no evidence runner for {task}")
    seed_parameter = runner_contract.get("seed_parameter")
    observed_seed = evidence.get("solver_seed")
    if seed_parameter is None:
        if task not in {"tsp", "cvrp"} or observed_seed is not None:
            raise ValueError(f"invalid deterministic no-seed evidence for {task}")
    else:
        if not isinstance(seed_parameter, str) or not isinstance(parameters, dict):
            raise ValueError(f"solver contract has no evidence seed for {task}")
        try:
            expected_seed = parameters[seed_parameter]
        except KeyError as exc:
            raise ValueError(f"invalid reference evidence seed for {task}") from exc
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or isinstance(observed_seed, bool)
            or not isinstance(observed_seed, int)
            or observed_seed != expected_seed
        ):
            raise ValueError(f"reference evidence seed mismatch for {task}")
    expected_semantics = {
        "tsp": "clasp_raw_euclidean_float64",
        "cvrp": "clasp_raw_euclidean_float64",
        "op": "clasp_raw_euclidean_float64_integer_prizes",
        "mis": "clasp_integer_mis_cardinality",
        "jssp": "clasp_integer_jssp_makespan",
    }[task]
    expected_quality = (
        "best_known_heuristic" if task in {"tsp", "cvrp"} else "certified_optimum"
    )
    expected_native = "HEURISTIC_COMPLETE" if task in {"tsp", "cvrp"} else "OPTIMAL"
    expected_public = (
        "BEST_KNOWN_HEURISTIC_VERIFIED" if task in {"tsp", "cvrp"} else "OPTIMAL_CERTIFIED"
    )
    if (
        runner_contract.get("objective_semantics") != expected_semantics
        or evidence.get("objective_semantics") != expected_semantics
    ):
        raise ValueError(f"reference evidence objective semantics mismatch for {task}")
    if (
        runner_contract.get("reference_quality") != expected_quality
        or evidence.get("reference_quality") != expected_quality
    ):
        raise ValueError(f"reference evidence quality classification mismatch for {task}")
    if provenance.get("solver_status") != expected_public:
        raise ValueError(f"invalid public reference status for {task}")
    if evidence.get("native_status") != expected_native:
        raise ValueError(f"invalid native reference status for {task}")
    identity = evidence.get("solver_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"missing solver identity evidence for {task}")
    if not _is_sha256(identity.get("artifact_sha256")):
        raise ValueError(f"invalid solver artifact hash evidence for {task}")
    for field in ("backend_name", "native_version", "artifact_kind"):
        if not str(identity.get(field, "")).strip():
            raise ValueError(f"missing solver identity field {field} for {task}")
    try:
        artifact_size = int(identity.get("size_bytes", 0))
    except (TypeError, ValueError):
        artifact_size = 0
    if artifact_size <= 0:
        raise ValueError(f"empty solver artifact evidence for {task}")
    contract_identity = runner_contract.get("solver_identity")
    if not isinstance(contract_identity, dict):
        raise ValueError(f"missing solver contract identity for {task}")
    for field, expected in contract_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"solver identity field {field} mismatch for {task}")
    raw_log = evidence.get("raw_native_log")
    if not isinstance(raw_log, dict) or not _is_sha256(raw_log.get("sha256")):
        raise ValueError(f"invalid native-log evidence for {task}")
    try:
        log_size = int(raw_log.get("size_bytes", 0))
        runtime = float(evidence.get("wall_time_seconds", -1))
    except (TypeError, ValueError):
        log_size, runtime = 0, -1.0
    if log_size <= 0 or not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"invalid native-log size or wall time evidence for {task}")
    relative_log = raw_log.get("relative_path")
    if not isinstance(relative_log, str) or not relative_log:
        raise ValueError(f"missing native-log path for {task}")
    log_path = (evidence_root / relative_log).resolve()
    try:
        log_path.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise ValueError(f"native-log path escapes the benchmark bundle for {task}") from exc
    live_sha, live_size = _sha256_file(str(log_path))
    if live_sha != raw_log.get("sha256") or live_size != raw_log.get("size_bytes"):
        raise ValueError(f"native-log live bytes mismatch for {task}")
    if not _objective_matches(
        task, evidence.get("native_incumbent"), strict_metrics.get("objective_value")
    ):
        raise ValueError(f"native incumbent does not match frozen objective for {task}")
    if task in {"tsp", "cvrp"}:
        if evidence.get("optimality_certificate") is not False:
            raise ValueError(f"heuristic reference falsely claims optimality for {task}")
        if evidence.get("native_best_bound") is not None or evidence.get("native_gap") is not None:
            raise ValueError(f"heuristic reference carries a false bound for {task}")
    else:
        expected_code = 2 if task in {"op", "mis"} else 4
        if evidence.get("optimality_certificate") is not True:
            raise ValueError(f"exact reference lacks certificate for {task}")
        if evidence.get("native_status_code") != expected_code:
            raise ValueError(f"exact reference native status code is invalid for {task}")
        if not _objective_matches(
            task, evidence.get("native_incumbent"), evidence.get("native_best_bound")
        ):
            raise ValueError(f"exact reference incumbent/bound mismatch for {task}")
        try:
            gap = float(evidence.get("native_gap"))
        except (TypeError, ValueError):
            gap = math.inf
        if gap != 0.0:
            raise ValueError(f"exact reference native gap is nonzero for {task}")
        if task in {"op", "mis"}:
            metadata = evidence.get("native_metadata")
            if not isinstance(metadata, dict) or int(metadata.get("solution_count", 0)) <= 0:
                raise ValueError(f"{task.upper()} exact reference lacks SolCount > 0")
        else:
            metadata = evidence.get("native_metadata")
            if not isinstance(metadata, dict) or metadata.get(
                "response_stats_sha256"
            ) != live_sha:
                raise ValueError("JSSP response statistics do not match the retained log")
            parsed_stats: dict[str, str] = {}
            with open(log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    key, separator, value = line.partition(":")
                    if separator and key.strip() and key.strip() not in parsed_stats:
                        parsed_stats[key.strip()] = value.strip()
            if parsed_stats.get("status", "").upper() != "OPTIMAL":
                raise ValueError("JSSP retained response statistics are not OPTIMAL")
            if not _objective_matches(
                task, parsed_stats.get("objective"), evidence.get("native_incumbent")
            ) or not _objective_matches(
                task, parsed_stats.get("best_bound"), evidence.get("native_best_bound")
            ):
                raise ValueError("JSSP response statistics objective/bound mismatch")
    if evidence.get("strict_verifier_identity") != verifier_identity:
        raise ValueError(f"reference evidence verifier identity mismatch for {task}")
    if evidence.get("strict_verifier_identity_sha256") != _sha256_json(verifier_identity):
        raise ValueError(f"reference evidence verifier hash mismatch for {task}")
    if evidence.get("strict_verifier_result") != strict_metrics:
        raise ValueError(f"stored strict-verifier evidence is stale for {task}")
    if evidence.get("strict_verifier_result_sha256") != _sha256_json(strict_metrics):
        raise ValueError(f"stored strict-verifier evidence hash is invalid for {task}")
    if not _objective_matches(
        task,
        evidence.get("strict_verified_objective"),
        strict_metrics.get("objective_value"),
    ):
        raise ValueError(f"stored strict objective is stale for {task}")
    return (
        str(evidence["reference_evidence_record_sha256"]),
        str(export_record_sha),
        str(evidence["reference_evidence_summary_sha256"]),
    )


def _default_verifier_bundle() -> tuple[dict[str, Callable[..., dict]], dict[str, object]]:
    from clasp.entrypoints.evaluate import TASK_EVALUATORS, _strict_verifier_identity

    return dict(TASK_EVALUATORS), dict(_strict_verifier_identity())


def _audit_test_task(
    spec: DatasetSpec,
    *,
    minimum_instances: int,
    evaluator: Callable[..., dict],
    solver_contract_task: dict[str, object] | None = None,
    verifier_identity: dict[str, object] | None = None,
    require_reference_evidence: bool = False,
    runner_code_sha256: str | None = None,
    evidence_root: Path | None = None,
    approval_receipt_sha256: str | None = None,
) -> tuple[dict[str, object], set[str]]:
    root, paths = _json_files(spec.path)
    files: list[dict[str, object]] = []
    instance_hashes: list[str] = []
    row_hashes: list[str] = []
    reference_hashes: list[str] = []
    reference_objectives: list[float] = []
    evidence_record_hashes: list[str] = []
    export_record_hashes: list[str] = []
    evidence_summary_hashes: set[str] = set()
    seen: dict[str, tuple[str, int]] = {}
    merged_index = 0
    for path in paths:
        relative_path = os.path.relpath(path, root)
        sha256, size = _sha256_file(path)
        file_rows = 0
        for file_row_index, clean in _iter_json_array(path):
            for field in ("instruction", "input", "output"):
                if not str(clean.get(field, "")).strip():
                    raise ValueError(f"empty {field} in {spec.task} at merged row {merged_index}")
            try:
                instance_hash = instance_sha256(spec.task, clean, require_structured=True)
            except ValueError as exc:
                raise ValueError(
                    f"invalid independent instance for {spec.task} at "
                    f"{relative_path}:{file_row_index}: {exc}"
                ) from exc
            if instance_hash in seen:
                first_path, first_index = seen[instance_hash]
                raise ValueError(
                    f"duplicate independent instance for {spec.task}: "
                    f"{first_path}:{first_index} and {relative_path}:{file_row_index}"
                )
            seen[instance_hash] = (relative_path, file_row_index)
            for identity_field in ("instance_sha256", "semantic_instance_sha256"):
                if identity_field in clean and clean.get(identity_field) != instance_hash:
                    raise ValueError(
                        f"stored {identity_field} disagrees with the semantic instance "
                        f"for {spec.task} at merged row {merged_index}"
                    )
            if solver_contract_task is not None:
                provenance = clean.get("reference_provenance")
                if not isinstance(provenance, dict):
                    raise ValueError(
                        f"missing reference provenance for {spec.task} at merged row {merged_index}"
                    )
                solver_task_sha = _sha256_json(solver_contract_task)
                if provenance.get("solver_contract_task_sha256") != solver_task_sha:
                    raise ValueError(
                        f"reference solver task contract mismatch for {spec.task} "
                        f"at merged row {merged_index}"
                    )
                accepted_statuses = solver_contract_task.get("accepted_statuses")
                if (
                    not isinstance(accepted_statuses, list)
                    or provenance.get("solver_status") not in accepted_statuses
                ):
                    raise ValueError(
                        f"reference solver status is not accepted for {spec.task} "
                        f"at merged row {merged_index}"
                    )
                if provenance.get("quality_semantics") != solver_contract_task.get(
                    "quality_semantics"
                ):
                    raise ValueError(
                        f"reference quality semantics mismatch for {spec.task} "
                        f"at merged row {merged_index}"
                    )
                try:
                    runtime = float(provenance.get("solver_runtime_seconds", -1))
                except (TypeError, ValueError):
                    runtime = -1.0
                if not math.isfinite(runtime) or runtime < 0:
                    raise ValueError(
                        f"invalid reference solver runtime for {spec.task} "
                        f"at merged row {merged_index}"
                    )
            instance = clean.get("instance_json", clean.get("instance"))
            if isinstance(instance, str):
                instance = json.loads(instance)
            metrics = evaluator(
                str(clean["instruction"]),
                str(clean["input"]),
                str(clean["output"]),
                instance,
                str(clean["output"]),
            )
            if not isinstance(metrics, dict) or not bool(metrics.get("is_feasible")):
                raise ValueError(
                    f"strict verifier rejects reference for {spec.task} "
                    f"at merged row {merged_index}"
                )
            objective = metrics.get("ground_truth_objective", metrics.get("objective_value"))
            try:
                objective_float = float(objective)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reference objective is missing for {spec.task} at merged row {merged_index}"
                ) from exc
            if not math.isfinite(objective_float):
                raise ValueError(
                    f"reference objective is non-finite for {spec.task} "
                    f"at merged row {merged_index}"
                )
            gap = metrics.get("gap_to_ground_truth")
            if gap is not None and (not math.isfinite(float(gap)) or abs(float(gap)) > 1e-9):
                raise ValueError(
                    f"reference does not reproduce zero verified gap for {spec.task} "
                    f"at merged row {merged_index}: {gap!r}"
                )
            if require_reference_evidence:
                if (
                    solver_contract_task is None
                    or verifier_identity is None
                    or runner_code_sha256 is None
                    or evidence_root is None
                    or approval_receipt_sha256 is None
                ):
                    raise ValueError("production reference evidence audit is not configured")
                provenance = clean.get("reference_provenance")
                if not isinstance(provenance, dict):
                    raise ValueError(
                        f"missing reference provenance for {spec.task} at merged row {merged_index}"
                    )
                evidence_sha, export_sha, summary_sha = _validate_frozen_reference_evidence(
                    task=spec.task,
                    provenance=provenance,
                    solver_contract_task=solver_contract_task,
                    verifier_identity=verifier_identity,
                    strict_metrics=metrics,
                    runner_code_sha256=runner_code_sha256,
                    evidence_root=evidence_root,
                    approval_receipt_sha256=approval_receipt_sha256,
                )
                evidence_record_hashes.append(evidence_sha)
                export_record_hashes.append(export_sha)
                evidence_summary_hashes.add(summary_sha)
            instance_hashes.append(instance_hash)
            row_hashes.append(benchmark_row_sha256(clean))
            reference_hashes.append(reference_output_sha256(clean["output"]))
            reference_objectives.append(objective_float)
            merged_index += 1
            file_rows += 1
        files.append(
            {
                "relative_path": relative_path,
                "sha256": sha256,
                "size_bytes": size,
                "rows": file_rows,
            }
        )
    if merged_index < minimum_instances:
        raise ValueError(
            f"independent task {spec.task} has {merged_index} rows; "
            f"at least {minimum_instances} are required"
        )

    task_manifest = {
        "label": spec.label,
        "source_locator": spec.path,
        "files": files,
        "file_manifest_sha256": _sha256_json(files),
        "rows": merged_index,
        "instance_hashes_sha256": _sha256_json(instance_hashes),
        "row_hashes_sha256": _sha256_json(row_hashes),
        "reference_hashes_sha256": _sha256_json(reference_hashes),
        "ordered_instance_sha256": instance_hashes,
        "ordered_semantic_instance_sha256": instance_hashes,
        "ordered_benchmark_row_sha256": row_hashes,
        "ordered_reference_output_sha256": reference_hashes,
        "reference_audit": {
            "strictly_feasible": merged_index,
            "zero_gap": merged_index,
            "objective_min": min(reference_objectives),
            "objective_max": max(reference_objectives),
        },
    }
    if require_reference_evidence:
        if len(evidence_summary_hashes) != 1:
            raise ValueError(f"{spec.task} rows do not share one reference evidence summary")
        task_manifest.update(
            {
                "reference_evidence_schema_version": 1,
                "reference_evidence_summary_sha256": next(iter(evidence_summary_hashes)),
                "ordered_reference_evidence_record_sha256": evidence_record_hashes,
                "reference_evidence_records_sha256": _sha256_json(evidence_record_hashes),
                "ordered_reference_export_record_sha256": export_record_hashes,
                "reference_export_records_sha256": _sha256_json(export_record_hashes),
            }
        )
    return task_manifest, set(instance_hashes)


def _forbidden_hashes(spec: DatasetSpec) -> tuple[set[str], dict[str, object]]:
    root, paths = _json_files(spec.path)
    files: list[dict[str, object]] = []
    hashes: set[str] = set()
    rows = 0
    for path in paths:
        relative_path = os.path.relpath(path, root)
        before = os.stat(path, follow_symlinks=True)
        sha256, size = _sha256_file(path)
        file_rows = 0
        for file_row_index, row in _iter_json_array(path):
            try:
                hashes.add(instance_sha256(spec.task, row))
            except ValueError as exc:
                raise ValueError(
                    f"cannot complete leakage audit for forbidden source "
                    f"{spec.label} at {relative_path}:{file_row_index}: {exc}"
                ) from exc
            rows += 1
            file_rows += 1
        observed_sha256, observed_size = _sha256_file(path)
        after = os.stat(path, follow_symlinks=True)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or sha256 != observed_sha256 or size != observed_size:
            raise RuntimeError(
                f"forbidden-source file changed during registry/leakage audit: {path}"
            )
        files.append(
            {
                "relative_path": relative_path,
                "sha256": sha256,
                "size_bytes": size,
                "rows": file_rows,
            }
        )
    observed_root, observed_paths = _json_files(spec.path)
    if observed_root != root or observed_paths != paths:
        raise RuntimeError(
            f"forbidden-source data tree changed during registry/leakage audit: {spec.path}"
        )
    source_manifest: dict[str, object] = {
        "task": spec.task,
        "label": spec.label,
        "source_locator": spec.path,
        "files": files,
        "file_manifest_sha256": _sha256_json(files),
        "rows": rows,
        "unique_instance_hashes": len(hashes),
    }
    source_manifest["source_identity_sha256"] = forbidden_source_manifest_identity(source_manifest)[
        "source_identity_sha256"
    ]
    return hashes, source_manifest


def build_manifest(
    *,
    test_specs: Sequence[DatasetSpec],
    forbidden_specs: Sequence[DatasetSpec],
    forbidden_source_registry: dict[str, object] | None,
    publication_mode: bool,
    generator_contract: dict[str, object],
    solver_contract: dict[str, object],
    minimum_instances: int,
    manifest_directory: str | None = None,
    evaluators: dict[str, Callable[..., dict]] | None = None,
    verifier_identity: dict[str, object] | None = None,
    generation_receipt: dict[str, object] | None = None,
    allow_smoke_receipt: bool = False,
) -> dict[str, object]:
    if not isinstance(publication_mode, bool):
        raise TypeError("publication_mode must be an explicit boolean")
    if minimum_instances <= 0:
        raise ValueError("minimum_instances must be positive")
    by_task: dict[str, DatasetSpec] = {}
    for spec in test_specs:
        if spec.task not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported task: {spec.task}")
        if spec.task in by_task:
            raise ValueError(f"duplicate independent test specification for {spec.task}")
        by_task[spec.task] = spec
    if not by_task:
        raise ValueError("at least one independent test dataset is required")
    if publication_mode:
        if set(by_task) != set(SUPPORTED_TASKS):
            raise ValueError("publication freeze requires exactly all five benchmark tasks")
        if minimum_instances != PUBLICATION_INSTANCES_PER_TASK:
            raise ValueError("publication freeze requires exactly 1,000 instances per task")
    forbidden_keys: set[tuple[str, str]] = set()
    for spec in forbidden_specs:
        key = (spec.task, spec.label)
        if key in forbidden_keys:
            raise ValueError(f"duplicate forbidden source specification: {key}")
        forbidden_keys.add(key)
    production_receipt = bool(
        generation_receipt is not None and generation_receipt.get("status") == "references_attached"
    )
    if publication_mode and not production_receipt:
        raise ValueError("publication freeze requires a production references_attached receipt")
    registry_identity: dict[str, object] | None = None
    if publication_mode:
        if forbidden_source_registry is None:
            raise ValueError("publication freeze requires --forbidden-source-registry")
        registry_identity = validate_forbidden_source_registry(
            forbidden_source_registry,
            expected_tasks=sorted(by_task),
        )
    elif forbidden_source_registry is not None:
        registry_identity = validate_forbidden_source_registry(
            forbidden_source_registry,
            expected_tasks=sorted(by_task),
        )
    if publication_mode:
        expected_keys = {
            (task, label) for task in by_task for label in REQUIRED_FORBIDDEN_SOURCE_LABELS
        }
        if forbidden_keys != expected_keys:
            missing = sorted(expected_keys - forbidden_keys)
            extra = sorted(forbidden_keys - expected_keys)
            raise ValueError(
                "publication forbidden specifications must exactly cover the frozen "
                f"task/label registry: missing={missing}, extra={extra}"
            )
    if evaluators is None or verifier_identity is None:
        default_evaluators, default_identity = _default_verifier_bundle()
        evaluators = evaluators or default_evaluators
        verifier_identity = verifier_identity or default_identity

    runner_code_sha256: str | None = None
    approval_receipt_sha256: str | None = None
    evidence_root: Path | None = None
    if publication_mode:
        assert generation_receipt is not None
        validate_reference_solver_contract_policy(
            solver_contract,
            verifier_identity_sha256=_sha256_json(verifier_identity),
            expected_tasks=SUPPORTED_TASKS,
            expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
            require_production_identity=True,
        )
        runner_protocol = solver_contract["reference_evidence_runner"]
        runner_code_sha256 = str(runner_protocol["runner_code_sha256"])
        approval = generation_receipt.get("reference_evidence_approval")
        receipt_tasks = generation_receipt.get("tasks")
        if not isinstance(receipt_tasks, dict):
            raise ValueError("production receipt has no reference task summaries")
        validate_reference_evidence_approval_receipt(
            approval,
            solver_contract=solver_contract,
            source_generation_receipt_sha256=str(
                generation_receipt.get("source_generation_receipt_sha256", "")
            ),
            verifier_identity_sha256=_sha256_json(verifier_identity),
            task_evidence_summary_sha256={
                task: str(receipt_tasks[task]["reference_evidence_summary_sha256"])
                for task in SUPPORTED_TASKS
            },
            expected_tasks=SUPPORTED_TASKS,
            expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
        )
        assert isinstance(approval, dict)
        approval_receipt_sha256 = str(
            approval["reference_evidence_approval_receipt_sha256"]
        )
        if generation_receipt.get(
            "reference_evidence_approval_receipt_sha256"
        ) != approval_receipt_sha256:
            raise ValueError("generation receipt approval identity is stale")
        candidate_roots: set[Path] = set()
        for task, spec in by_task.items():
            _dataset_root, dataset_paths = _json_files(spec.path)
            if len(dataset_paths) != 1:
                raise ValueError(f"publication task {task} must be one assembled JSON file")
            relative = Path(str(receipt_tasks[task].get("relative_path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"generation receipt path is unsafe for {task}")
            candidate = Path(dataset_paths[0]).resolve()
            root_candidate = candidate
            for _part in relative.parts:
                root_candidate = root_candidate.parent
            if (root_candidate / relative).resolve() != candidate:
                raise ValueError(f"generation receipt path does not locate {task} data")
            candidate_roots.add(root_candidate)
        if len(candidate_roots) != 1:
            raise ValueError("publication tasks do not share one portable benchmark bundle")
        evidence_root = next(iter(candidate_roots))

    solver_tasks: dict[str, object] | None = None
    require_reference_evidence = bool(
        generation_receipt is not None and generation_receipt.get("status") == "references_attached"
    )
    if generation_receipt is not None:
        candidate_solver_tasks = solver_contract.get("tasks")
        if not isinstance(candidate_solver_tasks, dict) or not all(
            isinstance(candidate_solver_tasks.get(task), dict) for task in by_task
        ):
            raise ValueError("reference solver contract must define every frozen benchmark task")
        solver_tasks = candidate_solver_tasks

    test_sets: dict[str, object] = {}
    test_hashes: dict[str, set[str]] = {}
    for task, spec in sorted(by_task.items()):
        if task not in evaluators:
            raise ValueError(f"strict evaluator is missing task {task}")
        task_manifest, hashes = _audit_test_task(
            spec,
            minimum_instances=minimum_instances,
            evaluator=evaluators[task],
            solver_contract_task=(solver_tasks[task] if solver_tasks is not None else None),
            verifier_identity=verifier_identity,
            require_reference_evidence=require_reference_evidence,
            runner_code_sha256=runner_code_sha256,
            evidence_root=evidence_root,
            approval_receipt_sha256=approval_receipt_sha256,
        )
        if publication_mode:
            if manifest_directory is None:
                raise ValueError(
                    "publication freeze requires the final manifest directory for "
                    "portable source locators"
                )
            manifest_root = os.path.realpath(os.path.abspath(manifest_directory))
            source_path = os.path.realpath(os.path.abspath(spec.path))
            try:
                shared_root = os.path.commonpath([manifest_root, source_path])
            except ValueError as exc:
                raise ValueError(
                    f"publication source for {task} is outside the manifest filesystem"
                ) from exc
            if shared_root != manifest_root:
                raise ValueError(
                    f"publication source for {task} must be inside the relocatable "
                    "benchmark bundle rooted at the manifest directory"
                )
            relative_locator = os.path.relpath(source_path, manifest_root).replace(os.sep, "/")
            if relative_locator in {"", "."} or relative_locator.startswith("../"):
                raise ValueError(f"publication source locator is unsafe for {task}")
            task_manifest["source_locator"] = relative_locator
        test_sets[task] = task_manifest
        test_hashes[task] = hashes

    generation_identity = None
    if generation_receipt is not None:
        _verify_generation_receipt(generation_receipt, allow_smoke=allow_smoke_receipt)
        if (
            publication_mode
            and generation_receipt.get("forbidden_source_registry_identity") != registry_identity
        ):
            raise ValueError(
                "generation receipt was not created from the frozen forbidden-source registry"
            )
        expected_contract_sha = _sha256_json(generator_contract)
        if generation_receipt.get("generator_contract_sha256") != expected_contract_sha:
            raise ValueError("generation receipt was produced from a different generator contract")
        expected_solver_sha = _sha256_json(solver_contract)
        if generation_receipt.get("reference_solver_contract_sha256") != expected_solver_sha:
            raise ValueError(
                "generation receipt was produced from a different reference solver contract"
            )
        if generation_receipt.get("strict_verifier") != verifier_identity:
            raise ValueError("generation receipt was verified by a different strict verifier")
        receipt_tasks = generation_receipt.get("tasks")
        if not isinstance(receipt_tasks, dict) or set(receipt_tasks) != set(by_task):
            raise ValueError("generation receipt task set does not match independent test task set")
        for task in sorted(by_task):
            receipt_task = receipt_tasks.get(task)
            if not isinstance(receipt_task, dict):
                raise ValueError(f"generation receipt has invalid task record for {task}")
            expected_order = test_sets[task]["ordered_instance_sha256"]
            observed_order = receipt_task.get("ordered_instance_sha256")
            if observed_order != expected_order:
                raise ValueError(
                    f"generation receipt ordered instance hashes do not match {task} rows"
                )
            if receipt_task.get("ordered_instance_hashes_sha256") != _sha256_json(expected_order):
                raise ValueError(
                    f"generation receipt aggregate instance hash is invalid for {task}"
                )
            if receipt_task.get("ordered_semantic_instance_sha256") != expected_order:
                raise ValueError(
                    f"generation receipt semantic instance hashes do not match {task} rows"
                )
            if receipt_task.get("semantic_instance_hashes_sha256") != _sha256_json(expected_order):
                raise ValueError(
                    f"generation receipt aggregate semantic hash is invalid for {task}"
                )
            expected_rows = test_sets[task]["ordered_benchmark_row_sha256"]
            if receipt_task.get("ordered_benchmark_row_sha256") != expected_rows:
                raise ValueError(
                    f"generation receipt ordered benchmark row hashes do not match {task} rows"
                )
            if receipt_task.get("benchmark_row_hashes_sha256") != _sha256_json(expected_rows):
                raise ValueError(
                    f"generation receipt aggregate benchmark row hash is invalid for {task}"
                )
            expected_references = test_sets[task]["ordered_reference_output_sha256"]
            if receipt_task.get("ordered_reference_output_sha256") != expected_references:
                raise ValueError(
                    f"generation receipt ordered reference hashes do not match {task} rows"
                )
            if receipt_task.get("reference_output_hashes_sha256") != _sha256_json(
                expected_references
            ):
                raise ValueError(
                    f"generation receipt aggregate reference hash is invalid for {task}"
                )
            if int(receipt_task.get("rows", -1)) != int(test_sets[task]["rows"]):
                raise ValueError(f"generation receipt row count does not match {task} rows")
            if require_reference_evidence:
                if generation_receipt.get("reference_evidence_schema_version") != 1:
                    raise ValueError(
                        "production generation receipt lacks reference evidence schema"
                    )
                if not _is_sha256(generation_receipt.get("reference_export_summary_sha256")):
                    raise ValueError(
                        "production generation receipt has no reference export summary hash"
                    )
                for field in (
                    "reference_evidence_summary_sha256",
                    "ordered_reference_evidence_record_sha256",
                    "reference_evidence_records_sha256",
                    "ordered_reference_export_record_sha256",
                    "reference_export_records_sha256",
                ):
                    if receipt_task.get(field) != test_sets[task].get(field):
                        raise ValueError(
                            f"generation receipt reference evidence {field} "
                            f"does not match {task} rows"
                        )
        # Embed the full self-hashed receipt, not a lossy identity projection.
        # Evaluator and aggregator validation can therefore independently
        # recompute its self-hash and close every ordered task sequence.
        generation_identity = copy.deepcopy(generation_receipt)

    forbidden_manifests: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    for spec in sorted(forbidden_specs, key=lambda item: (item.task, item.label)):
        if spec.task not in by_task:
            continue
        hashes, source_manifest = _forbidden_hashes(spec)
        if publication_mode:
            assert manifest_directory is not None
            manifest_root = os.path.realpath(os.path.abspath(manifest_directory))
            source_path = os.path.realpath(os.path.abspath(spec.path))
            if os.path.commonpath([manifest_root, source_path]) != manifest_root:
                raise ValueError(
                    f"publication forbidden source {spec.task}/{spec.label} must be "
                    "inside the relocatable benchmark bundle"
                )
            relative_locator = os.path.relpath(source_path, manifest_root).replace(os.sep, "/")
            if relative_locator in {"", "."} or relative_locator.startswith("../"):
                raise ValueError(
                    f"publication forbidden source locator is unsafe for {spec.task}/{spec.label}"
                )
            source_manifest["source_locator"] = relative_locator
        if publication_mode and (int(source_manifest.get("rows", 0)) <= 0 or not hashes):
            raise ValueError(f"production forbidden source {spec.task}/{spec.label} is empty")
        forbidden_manifests.append(source_manifest)
        collision_hashes = sorted(test_hashes[spec.task] & hashes)
        if collision_hashes:
            overlaps.append(
                {
                    "task": spec.task,
                    "forbidden_label": spec.label,
                    "count": len(collision_hashes),
                    "example_instance_sha256": collision_hashes[:5],
                }
            )
    if overlaps:
        raise ValueError(f"independent-test leakage detected: {overlaps}")
    if forbidden_source_registry is not None:
        observed_registry_identity = validate_forbidden_sources_against_registry(
            forbidden_manifests,
            forbidden_source_registry,
            expected_tasks=sorted(by_task),
        )
        if observed_registry_identity != registry_identity:
            raise ValueError("forbidden-source registry identity changed during freeze")

    core: dict[str, object] = {
        "schema_version": 3,
        "status": "frozen_before_inference",
        "benchmark_mode": (
            PUBLICATION_BENCHMARK_MODE if publication_mode else NONPUBLICATION_BENCHMARK_MODE
        ),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "minimum_instances_per_task": minimum_instances,
        "tasks": sorted(by_task),
        "contracts": {
            "generator": generator_contract,
            "generator_contract_sha256": _sha256_json(generator_contract),
            "reference_solver": solver_contract,
            "reference_solver_contract_sha256": _sha256_json(solver_contract),
        },
        "strict_verifier": verifier_identity,
        "generation_receipt": generation_identity,
        "test_sets": test_sets,
        "forbidden_sources": forbidden_manifests,
        "forbidden_source_registry": forbidden_source_registry,
        "forbidden_source_registry_identity": (
            forbidden_source_registry_identity(forbidden_source_registry)
            if forbidden_source_registry is not None
            else None
        ),
        "overlap_audit": {
            "passed": True,
            "tested_forbidden_sources": len(forbidden_manifests),
            "collisions": 0,
        },
    }
    core["manifest_sha256"] = _sha256_json(core)
    return core


def verify_manifest_integrity(manifest: dict[str, object]) -> None:
    mode = manifest.get("benchmark_mode")
    if mode not in {PUBLICATION_BENCHMARK_MODE, NONPUBLICATION_BENCHMARK_MODE}:
        raise ValueError("benchmark manifest has no explicit publication mode")
    production = mode == PUBLICATION_BENCHMARK_MODE
    validate_frozen_benchmark_manifest(
        manifest,
        require_generation_receipt=production,
        require_reference_evidence=production,
    )


def _write_json_atomic(path: str, value: object) -> None:
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".benchmark-manifest-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split(",", 2)
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("dataset spec must be TASK,LABEL,PATH")
    task, label, path = (part.strip() for part in parts)
    if task not in SUPPORTED_TASKS:
        raise argparse.ArgumentTypeError(f"unsupported task {task!r}")
    return DatasetSpec(task=task, label=label, path=path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append", required=True, type=_parse_dataset_spec)
    parser.add_argument("--forbidden", action="append", default=[], type=_parse_dataset_spec)
    parser.add_argument(
        "--forbidden-source-registry",
        help=(
            "Self-hashed registry frozen before confirmation generation. Required "
            "for publication mode."
        ),
    )
    parser.add_argument("--generator-contract", required=True)
    parser.add_argument("--solver-contract", required=True)
    parser.add_argument("--generation-receipt")
    parser.add_argument(
        "--allow-missing-generation-receipt",
        action="store_true",
        help="Legacy-only escape hatch; confirmation manifests must not use it.",
    )
    parser.add_argument(
        "--allow-smoke-receipt",
        action="store_true",
        help="Allow a non-publishable smoke receipt for pipeline testing.",
    )
    parser.add_argument(
        "--nonpublication-mode",
        action="store_true",
        help=(
            "Explicitly produce a development/smoke-only manifest. Such a manifest "
            "is rejected by confirmation evaluation and publication aggregation."
        ),
    )
    parser.add_argument("--minimum-instances-per-task", type=int, default=1_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        raise FileExistsError(
            f"refusing to overwrite frozen benchmark manifest: {os.path.abspath(args.output)}"
        )
    if (
        args.allow_missing_generation_receipt or args.allow_smoke_receipt
    ) and not args.nonpublication_mode:
        raise ValueError("smoke/legacy receipt escape hatches require --nonpublication-mode")
    if not args.generation_receipt and not args.nonpublication_mode:
        raise ValueError(
            "--generation-receipt is required for a freeze-eligible independent benchmark"
        )
    if not args.nonpublication_mode and not args.forbidden_source_registry:
        raise ValueError("--forbidden-source-registry is required in publication mode")
    generation_receipt = None
    if args.generation_receipt:
        generation_receipt = _load_contract(args.generation_receipt, label="generation receipt")
    forbidden_source_registry = None
    if args.forbidden_source_registry:
        forbidden_source_registry = _load_contract(
            args.forbidden_source_registry,
            label="forbidden-source registry",
        )
    manifest = build_manifest(
        test_specs=args.test,
        forbidden_specs=args.forbidden,
        forbidden_source_registry=forbidden_source_registry,
        publication_mode=not args.nonpublication_mode,
        generator_contract=_load_contract(args.generator_contract, label="generator"),
        solver_contract=_load_contract(args.solver_contract, label="reference solver"),
        minimum_instances=args.minimum_instances_per_task,
        manifest_directory=os.path.dirname(os.path.abspath(args.output)),
        generation_receipt=generation_receipt,
        allow_smoke_receipt=args.allow_smoke_receipt,
    )
    verify_manifest_integrity(manifest)
    _write_json_atomic(args.output, manifest)
    print(f"frozen benchmark manifest: {os.path.abspath(args.output)}")
    print(f"manifest_sha256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
