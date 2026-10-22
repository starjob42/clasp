#!/usr/bin/env python3
"""Run and export hash-bound reference-solver evidence for the IID benchmark.

This tool deliberately separates solving from reference export:

``solve-task``
    Verifies the unlabeled generation receipt and an approved, self-hashed
    solver contract *before* importing or invoking a backend.  It then writes
    one evidence record and one native log per benchmark row into a new atomic
    directory.

``export-references``
    Reopens the frozen draft and contract, independently verifies every
    evidence record, native-log identity, task-specific certificate rule, and
    strict CLASP metric, then emits the compact JSON files consumed by
    ``generate_iid_benchmark.py assemble``.

The script never changes the draft, solver contract, or assembled benchmark.
In particular, running ``solve-task`` is not approval to freeze labels or run
model inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from clasp.data.benchmark import (
    AUDITED_GUROBIPY_WHEEL_SHA256,
    AUDITED_GUROBI_VERSION,
    AUDITED_ORTOOLS_VERSION,
    AUDITED_ORTOOLS_WHEEL_SHA256,
    PUBLICATION_INSTANCES_PER_TASK,
    REFERENCE_ADAPTER,
    REFERENCE_EVIDENCE_APPROVAL_SCHEMA_VERSION,
    REFERENCE_EVIDENCE_APPROVAL_STATUS,
    REFERENCE_EVIDENCE_CONTRACT_STATUS,
    REFERENCE_EVIDENCE_RUNNER_PROTOCOL,
    REFERENCE_EVIDENCE_SCHEMA_VERSION,
    REFERENCE_NATIVE_STATUS,
    REFERENCE_OBJECTIVE_SEMANTICS,
    REFERENCE_PUBLIC_STATUS,
    REFERENCE_QUALITY,
    SUPPORTED_BENCHMARK_TASKS,
    benchmark_row_sha256,
    is_sha256,
    reference_output_sha256,
    semantic_instance_sha256,
    sha256_json,
    validate_reference_evidence_approval_receipt,
    validate_reference_solver_contract_policy,
    validate_reference_solver_task_policy,
)

TASKS = SUPPORTED_BENCHMARK_TASKS
EVIDENCE_SCHEMA_VERSION = REFERENCE_EVIDENCE_SCHEMA_VERSION
RUNNER_PROTOCOL = REFERENCE_EVIDENCE_RUNNER_PROTOCOL
OBJECTIVE_SEMANTICS = REFERENCE_OBJECTIVE_SEMANTICS
PUBLIC_STATUS = REFERENCE_PUBLIC_STATUS
REQUIRED_NATIVE_STATUS = REFERENCE_NATIVE_STATUS
HEURISTIC_REFERENCE_TASKS = frozenset({"tsp", "cvrp"})
GUROBI_EXACT_TASKS = frozenset({"op", "mis"})

# The adapter consumes no implicit Gurobi defaults for settings that can alter
# the search/certificate.  Resource limits remain contract-selected and
# hash-bound; hitting any one of them is a hard failure, never a best-known
# fallback.  The exact adapter name prevents a historical ``gurobi_op_v1``
# heuristic contract from being silently reinterpreted under this rule.
OP_EXACT_ADAPTER = REFERENCE_ADAPTER["op"]
OP_AUDITED_GUROBI_VERSION = AUDITED_GUROBI_VERSION
OP_AUDITED_GUROBIPY_WHEEL_SHA256 = AUDITED_GUROBIPY_WHEEL_SHA256
OP_EXACT_REQUIRED_PARAMETERS: dict[str, int | float] = {
    "threads": 1,
    "seed": 0,
    "method": 1,
    "mip_gap": 0.0,
    "mip_gap_abs": 0.0,
    "feasibility_tolerance": 1e-9,
    "integrality_tolerance": 1e-9,
    "numeric_focus": 3,
    "output_flag": 1,
    "log_to_console": 0,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _self_hash(value: dict[str, object], field: str) -> dict[str, object]:
    sealed = dict(value)
    sealed.pop(field, None)
    sealed[field] = hashlib.sha256(_canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _verify_self_hash(value: dict[str, object], field: str, *, label: str) -> None:
    stored = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored != observed:
        raise ValueError(
            f"{label} self-hash mismatch: stored={stored or '<missing>'}, observed={observed}"
        )


def _file_identity(path: str | Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _installed_distribution_identity(
    distribution_name: str, module: object
) -> dict[str, str]:
    """Hash the live installed distribution and bind the imported module origin.

    An offline wheel hash alone does not prove which code Python imported.  The
    contract therefore freezes both the wheel and this manifest of the files
    visible through the installed distribution's RECORD metadata.
    """

    distribution = importlib.metadata.distribution(distribution_name)
    files = distribution.files
    if not files:
        raise RuntimeError(f"installed distribution {distribution_name} has no RECORD files")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"imported module {distribution_name} has no filesystem origin")
    module_origin = str(Path(module_file).resolve())
    manifest: list[dict[str, object]] = []
    for item in sorted(files, key=lambda value: str(value)):
        relative = str(item).replace(os.sep, "/")
        if relative.endswith((".pyc", ".pyo")) or "/__pycache__/" in f"/{relative}":
            continue
        path = Path(distribution.locate_file(item)).resolve()
        if not path.is_file():
            raise RuntimeError(
                f"installed distribution {distribution_name} RECORD file is missing: {relative}"
            )
        manifest.append({"relative_path": relative, **_file_identity(path)})
    if not manifest:
        raise RuntimeError(f"installed distribution {distribution_name} has no hashable files")
    return {
        "distribution_name": distribution_name,
        "module_origin": module_origin,
        "installed_distribution_manifest_sha256": sha256_json(manifest),
    }


def _load_object(path: str | Path, *, label: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object: {path}")
    return value


def _load_rows(path: str | Path, *, label: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a JSON list of objects: {path}")
    return value


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_directory(destination: str | Path, writer: Callable[[Path], None]) -> None:
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        writer(temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _safe_relative_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} has no relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its artifact directory: {relative!r}") from exc
    return candidate


def _finite_number(value: object, *, label: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _exact_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be integral, got {value!r}")
    number = _finite_number(value, label=label)
    assert number is not None
    rounded = round(number)
    if abs(number - rounded) > 1e-9:
        raise ValueError(f"{label} must be integral, got {value!r}")
    return int(rounded)


def _default_verifier_bundle() -> tuple[dict[str, Callable[..., dict]], dict[str, object]]:
    from clasp.entrypoints.evaluate import TASK_EVALUATORS, _strict_verifier_identity

    return dict(TASK_EVALUATORS), dict(_strict_verifier_identity())


@dataclass(frozen=True)
class NativeSolveResult:
    """Backend result before task-specific acceptance and strict verification."""

    output: str
    native_status: str
    native_status_code: int | str | None
    incumbent: int | float | None
    best_bound: int | float | None
    gap: float | None
    raw_log: bytes
    objective_semantics: str
    native_wall_time_seconds: float | None = None
    native_metadata: dict[str, object] | None = None


class EvidenceBackend(Protocol):
    def probe_identity(self) -> dict[str, str]: ...

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult: ...


@dataclass(frozen=True)
class PreparedRun:
    task: str
    draft_root: Path
    rows: list[dict[str, object]]
    generation_receipt: dict[str, object]
    solver_contract: dict[str, object]
    task_contract: dict[str, object]
    runner_contract: dict[str, object]
    solver_artifact: dict[str, object]
    parameters: dict[str, object]
    solver_seed: int | None
    verifier_identity: dict[str, object]


def _verify_contract_self_hash(contract: dict[str, object]) -> None:
    if contract.get("schema_version") != 2:
        raise ValueError("reference solver contract must use schema_version 2")
    _verify_self_hash(contract, "contract_sha256", label="reference solver contract")


def _validate_task_contract(
    contract: dict[str, object], task: str, verifier_identity: dict[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], int | None]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    _verify_contract_self_hash(contract)
    if contract.get("status") != REFERENCE_EVIDENCE_CONTRACT_STATUS:
        raise ValueError("reference solver contract is not approved for evidence execution")
    protocol = contract.get("reference_evidence_runner")
    if not isinstance(protocol, dict) or protocol.get("protocol") != RUNNER_PROTOCOL:
        raise ValueError(f"solver contract does not select {RUNNER_PROTOCOL}")
    if protocol.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("solver contract has an unsupported evidence schema")
    expected_runner_code_sha = protocol.get("runner_code_sha256")
    observed_runner_code_sha = str(_file_identity(__file__)["sha256"])
    if expected_runner_code_sha != observed_runner_code_sha:
        raise ValueError(
            "solver contract evidence-runner source hash does not match the active runner"
        )
    expected_verifier_sha = protocol.get("strict_verifier_identity_sha256")
    observed_verifier_sha = sha256_json(verifier_identity)
    if expected_verifier_sha != observed_verifier_sha:
        raise ValueError(
            "solver contract strict-verifier identity does not match the active verifier"
        )

    tasks = contract.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(tasks.get(task), dict):
        raise ValueError(f"solver contract does not define task {task}")
    task_contract = copy.deepcopy(tasks[task])
    runner = task_contract.get("evidence_runner")
    seed = validate_reference_solver_task_policy(
        task,
        task_contract,
        runner_code_sha256=observed_runner_code_sha,
        require_production_identity=False,
    )
    assert isinstance(runner, dict)
    identity = runner.get("solver_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{task} solver identity is missing")
    for field in (
        "backend_name",
        "native_version",
        "artifact_kind",
        "artifact_path",
        "artifact_sha256",
    ):
        value = identity.get(field)
        if value in (None, "") or "UNRESOLVED" in str(value).upper():
            raise ValueError(f"{task} solver identity field {field} is unresolved")
    artifact_path = Path(str(identity["artifact_path"])).resolve()
    if not artifact_path.is_file():
        raise ValueError(f"{task} solver identity artifact is not a file: {artifact_path}")
    artifact = _file_identity(artifact_path)
    if artifact["sha256"] != identity["artifact_sha256"]:
        raise ValueError(f"{task} solver artifact hash mismatch")
    if int(artifact["size_bytes"]) <= 0:
        raise ValueError(f"{task} solver identity artifact is empty")
    artifact.update(
        {
            "artifact_sha256": artifact["sha256"],
            "artifact_kind": str(identity["artifact_kind"]),
            "artifact_path": str(artifact_path),
            "backend_name": str(identity["backend_name"]),
            "native_version": str(identity["native_version"]),
        }
    )
    parameters = task_contract.get("parameters")
    assert isinstance(parameters, dict)
    return task_contract, copy.deepcopy(runner), artifact, seed


def _load_and_validate_draft(
    draft_root: str | Path,
    task: str,
    *,
    expected_instances: int,
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    root = Path(draft_root).resolve()
    receipt = _load_object(root / "generation_receipt.json", label="generation receipt")
    _verify_self_hash(receipt, "receipt_sha256", label="generation receipt")
    if receipt.get("schema_version") != 2 or receipt.get("status") != "instances_generated":
        raise ValueError("reference evidence requires a schema-v2 unlabeled production draft")
    if receipt.get("contains_references") is not False or receipt.get("publishable") is not False:
        raise ValueError("reference evidence input must remain unlabeled and non-publishable")
    tasks = receipt.get("tasks")
    task_receipt = tasks.get(task) if isinstance(tasks, dict) else None
    if not isinstance(task_receipt, dict):
        raise ValueError(f"generation receipt does not define task {task}")
    try:
        row_count = int(task_receipt.get("rows", -1))
    except (TypeError, ValueError):
        row_count = -1
    if row_count != expected_instances:
        raise ValueError(
            f"{task} draft has {row_count} rows but the contract requires {expected_instances}"
        )
    dataset_path = _safe_relative_path(
        root, task_receipt.get("relative_path"), label=f"{task} draft"
    )
    file_identity = _file_identity(dataset_path)
    if file_identity["sha256"] != task_receipt.get("file_sha256"):
        raise ValueError(f"{task} draft file hash disagrees with its generation receipt")
    rows = _load_rows(dataset_path, label=f"{task} unlabeled draft")
    if len(rows) != row_count:
        raise ValueError(f"{task} draft row count disagrees with its generation receipt")
    semantic_hashes: list[str] = []
    row_hashes: list[str] = []
    for index, row in enumerate(rows):
        row_id = f"{task}-{index:06d}"
        if row.get("benchmark_row_id") != row_id:
            raise ValueError(f"{task} draft has a noncanonical row id at index {index}")
        if str(row.get("output", "")).strip():
            raise ValueError(f"{task} draft row {index} already contains a reference output")
        if row.get("reference_status") != "pending":
            raise ValueError(f"{task} draft row {index} is not pending reference solving")
        semantic_hash = semantic_instance_sha256(task, row, require_structured=True)
        if row.get("instance_sha256") != semantic_hash:
            raise ValueError(f"{task} draft row {index} has a stale instance_sha256")
        if row.get("semantic_instance_sha256") != semantic_hash:
            raise ValueError(
                f"{task} draft row {index} has a stale semantic_instance_sha256"
            )
        semantic_hashes.append(semantic_hash)
        row_hashes.append(benchmark_row_sha256(row))
    if task_receipt.get("ordered_instance_sha256") != semantic_hashes:
        raise ValueError(f"{task} ordered legacy instance hashes are invalid")
    if task_receipt.get("ordered_semantic_instance_sha256") != semantic_hashes:
        raise ValueError(f"{task} ordered semantic instance hashes are invalid")
    if task_receipt.get("semantic_instance_hashes_sha256") != sha256_json(
        semantic_hashes
    ):
        raise ValueError(f"{task} aggregate semantic instance hash is invalid")
    if task_receipt.get("ordered_benchmark_row_sha256") != row_hashes:
        raise ValueError(f"{task} ordered benchmark-row hashes are invalid")
    if task_receipt.get("benchmark_row_hashes_sha256") != sha256_json(row_hashes):
        raise ValueError(f"{task} aggregate benchmark-row hash is invalid")
    return root, rows, receipt


def prepare_run(
    *,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    task: str,
    verifier_identity: dict[str, object],
) -> PreparedRun:
    contract = _load_object(solver_contract_path, label="reference solver contract")
    task_contract, runner, artifact, seed = _validate_task_contract(
        contract, task, verifier_identity
    )
    protocol = contract["reference_evidence_runner"]
    expected_instances_raw = protocol.get("expected_instances_per_task")
    if (
        isinstance(expected_instances_raw, bool)
        or not isinstance(expected_instances_raw, int)
        or expected_instances_raw <= 0
    ):
        raise ValueError("solver contract expected_instances_per_task must be a positive integer")
    expected_instances = expected_instances_raw
    root, rows, receipt = _load_and_validate_draft(
        draft_root, task, expected_instances=expected_instances
    )
    return PreparedRun(
        task=task,
        draft_root=root,
        rows=rows,
        generation_receipt=receipt,
        solver_contract=contract,
        task_contract=task_contract,
        runner_contract=runner,
        solver_artifact=artifact,
        parameters=copy.deepcopy(task_contract["parameters"]),
        solver_seed=seed,
        verifier_identity=copy.deepcopy(verifier_identity),
    )


def _verify_backend_identity(prepared: PreparedRun, backend: EvidenceBackend) -> dict[str, str]:
    observed = backend.probe_identity()
    if not isinstance(observed, dict):
        raise ValueError("backend identity probe did not return an object")
    expected = prepared.solver_artifact
    required_fields = ["backend_name", "native_version"]
    if expected.get("artifact_kind") == "python_wheel":
        required_fields.extend(
            (
                "distribution_name",
                "module_origin",
                "installed_distribution_manifest_sha256",
            )
        )
    for field in required_fields:
        if observed.get(field) != expected[field]:
            raise ValueError(
                f"{prepared.task} backend {field} mismatch: "
                f"expected {expected[field]!r}, observed {observed.get(field)!r}"
            )
    return {field: observed[field] for field in required_fields}


def _objective_matches(task: str, left: object, right: object, *, label: str) -> None:
    if task in {"op", "mis", "jssp"}:
        if _exact_integer(left, label=f"{label} left") != _exact_integer(
            right, label=f"{label} right"
        ):
            raise ValueError(f"{label} disagrees under integer benchmark semantics")
        return
    first = _finite_number(left, label=f"{label} left")
    second = _finite_number(right, label=f"{label} right")
    assert first is not None and second is not None
    if not math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{label} disagrees under raw Euclidean benchmark semantics")


def _parse_cp_sat_response_stats(raw_log: bytes) -> dict[str, str]:
    try:
        text = raw_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("JSSP retained response statistics are not UTF-8") from exc
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and key.strip() not in parsed:
            parsed[key.strip()] = value.strip()
    return parsed


def _validate_native_acceptance(
    task: str,
    result: NativeSolveResult | Mapping[str, object],
    *,
    verified_objective: object,
) -> tuple[str, str, bool]:
    get = result.get if isinstance(result, Mapping) else lambda key: getattr(result, key)
    status = str(get("native_status"))
    if status != REQUIRED_NATIVE_STATUS[task]:
        raise ValueError(
            f"{task} native status must be {REQUIRED_NATIVE_STATUS[task]!r}, got {status!r}"
        )
    if get("objective_semantics") != OBJECTIVE_SEMANTICS[task]:
        raise ValueError(f"{task} backend used objective semantics different from CLASP")
    incumbent = get("incumbent")
    _objective_matches(task, incumbent, verified_objective, label="native incumbent")

    if task in HEURISTIC_REFERENCE_TASKS:
        if get("best_bound") is not None or get("gap") is not None:
            raise ValueError(
                f"{task} heuristic evidence must not manufacture a bound or optimality gap"
            )
        return PUBLIC_STATUS[task], REFERENCE_QUALITY[task], False

    incumbent_int = _exact_integer(incumbent, label=f"{task} native incumbent")
    bound_int = _exact_integer(get("best_bound"), label=f"{task} native best bound")
    if incumbent_int != bound_int:
        raise ValueError(f"{task} OPTIMAL incumbent does not equal the native best bound")
    gap = _finite_number(get("gap"), label=f"{task} native gap")
    if gap != 0.0:
        raise ValueError(f"{task} certified optimum requires an exactly zero native gap")
    metadata = get("native_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{task} native certificate metadata is missing")
    if task in GUROBI_EXACT_TASKS:
        if get("native_status_code") != 2:
            raise ValueError(
                f"{task.upper()} requires Gurobi GRB.OPTIMAL numeric status 2"
            )
        if int(metadata.get("solution_count", 0)) <= 0:
            raise ValueError(f"{task.upper()} requires Gurobi SolCount > 0")
        if task == "op":
            if metadata.get("gurobi_status") != "OPTIMAL":
                raise ValueError("OP metadata must retain Gurobi status OPTIMAL")
            if metadata.get("gurobi_status_code") != 2:
                raise ValueError("OP metadata must retain Gurobi status code 2")
            _objective_matches(
                task,
                metadata.get("gurobi_best_bound"),
                bound_int,
                label="OP retained Gurobi bound",
            )
            retained_gap = _finite_number(
                metadata.get("gurobi_gap"), label="OP retained Gurobi gap"
            )
            if retained_gap != 0.0:
                raise ValueError("OP retained Gurobi gap must be exactly zero")
    elif get("native_status_code") != 4:
        raise ValueError("JSSP requires OR-Tools CP-SAT OPTIMAL numeric status 4")
    else:
        raw_log = get("raw_log")
        if not isinstance(raw_log, bytes) or not raw_log:
            raise ValueError("JSSP requires retained CpSolverResponseStats bytes")
        response_stats_sha = hashlib.sha256(raw_log).hexdigest()
        if metadata.get("response_stats_sha256") != response_stats_sha:
            raise ValueError("JSSP response-stats hash does not match the retained native log")
        response_stats = _parse_cp_sat_response_stats(raw_log)
        if response_stats.get("status", "").upper() != "OPTIMAL":
            raise ValueError("JSSP response statistics do not retain OPTIMAL status")
        _objective_matches(
            task,
            response_stats.get("objective"),
            incumbent_int,
            label="JSSP response-stats objective",
        )
        _objective_matches(
            task,
            response_stats.get("best_bound"),
            bound_int,
            label="JSSP response-stats best bound",
        )
    return PUBLIC_STATUS[task], REFERENCE_QUALITY[task], True


def _strict_verify(
    task: str,
    row: dict[str, object],
    output: str,
    evaluator: Callable[..., dict],
) -> tuple[dict[str, object], float]:
    metrics = evaluator(
        str(row.get("instruction", "")),
        str(row.get("input", "")),
        output,
        row.get("instance_json", row.get("instance")),
        output,
    )
    if not isinstance(metrics, dict):
        raise ValueError(f"{task} strict verifier returned a non-object result")
    if metrics.get("is_feasible") is not True:
        raise ValueError(f"{task} strict verifier rejected the backend output")
    objective = metrics.get("objective_value")
    objective_float = _finite_number(objective, label=f"{task} verified objective")
    assert objective_float is not None
    ground_truth = metrics.get("ground_truth_objective")
    _objective_matches(task, ground_truth, objective_float, label="strict self-reference")
    declared = metrics.get("declared_objective")
    _objective_matches(task, declared, objective_float, label="declared objective")
    gap = metrics.get("gap_to_ground_truth")
    if gap is not None and _finite_number(gap, label="strict self-gap") != 0.0:
        raise ValueError(f"{task} strict verifier did not reproduce a zero self-gap")
    # Round-trip through canonical JSON to reject NaN and non-serializable backend data.
    clean_metrics = json.loads(_canonical_json_bytes(metrics).decode("utf-8"))
    return clean_metrics, objective_float


def _record_from_result(
    *,
    prepared: PreparedRun,
    row: dict[str, object],
    row_index: int,
    result: NativeSolveResult,
    measured_wall_time: float,
    log_relative_path: str,
    log_identity: dict[str, object],
    evaluator: Callable[..., dict],
    backend_identity: dict[str, str],
    runner_code_sha256: str,
) -> dict[str, object]:
    if not isinstance(result, NativeSolveResult):
        raise TypeError("backend solve must return NativeSolveResult")
    if not result.output.strip():
        raise ValueError(f"{prepared.task} backend returned an empty solution")
    if not isinstance(result.raw_log, bytes) or not result.raw_log:
        raise ValueError(f"{prepared.task} backend returned no raw native log")
    strict_metrics, verified_objective = _strict_verify(
        prepared.task, row, result.output, evaluator
    )
    solver_status, quality, optimality_certificate = _validate_native_acceptance(
        prepared.task, result, verified_objective=verified_objective
    )
    native_wall = _finite_number(
        result.native_wall_time_seconds,
        label="native wall time",
        allow_none=True,
    )
    if native_wall is not None and native_wall < 0:
        raise ValueError("native wall time cannot be negative")
    metadata = result.native_metadata or {}
    if not isinstance(metadata, dict):
        raise ValueError("native metadata must be an object")
    clean_metadata = json.loads(_canonical_json_bytes(metadata).decode("utf-8"))
    task_contract_sha = sha256_json(prepared.task_contract)
    semantic_hash = semantic_instance_sha256(
        prepared.task, row, require_structured=True
    )
    record = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "task": prepared.task,
        "row_index": row_index,
        "benchmark_row_id": row["benchmark_row_id"],
        "instance_sha256": semantic_hash,
        "semantic_instance_sha256": semantic_hash,
        "benchmark_row_sha256": benchmark_row_sha256(row),
        "output": result.output,
        "reference_output_sha256": reference_output_sha256(result.output),
        "solver_status": solver_status,
        "reference_quality": quality,
        "optimality_certificate": optimality_certificate,
        "solver_contract_sha256": prepared.solver_contract["contract_sha256"],
        "solver_contract_task_sha256": task_contract_sha,
        "runner_protocol": RUNNER_PROTOCOL,
        "runner_code_sha256": runner_code_sha256,
        "solver_identity": {
            **prepared.solver_artifact,
            **backend_identity,
        },
        "solver_parameters": prepared.parameters,
        "solver_parameters_sha256": sha256_json(prepared.parameters),
        "solver_seed": prepared.solver_seed,
        "objective_semantics": result.objective_semantics,
        "native_status": result.native_status,
        "native_status_code": result.native_status_code,
        "native_incumbent": result.incumbent,
        "native_best_bound": result.best_bound,
        "native_gap": result.gap,
        "wall_time_seconds": measured_wall_time,
        "native_wall_time_seconds": native_wall,
        "raw_native_log": {
            "relative_path": log_relative_path,
            **log_identity,
        },
        "native_metadata": clean_metadata,
        "strict_verifier_identity": prepared.verifier_identity,
        "strict_verifier_identity_sha256": sha256_json(prepared.verifier_identity),
        "strict_verifier_result": strict_metrics,
        "strict_verifier_result_sha256": sha256_json(strict_metrics),
        "strict_verified_objective": verified_objective,
    }
    return _self_hash(record, "reference_evidence_record_sha256")


class RawTSPHeuristicBackend:
    """Deterministic all-start nearest-neighbor plus raw-distance 2-opt."""

    def probe_identity(self) -> dict[str, str]:
        return {
            "backend_name": "clasp_raw_tsp_heuristic",
            "native_version": "1",
        }

    @staticmethod
    def _distance(coords: list[list[float]], left: int, right: int) -> float:
        return math.hypot(
            float(coords[left][0]) - float(coords[right][0]),
            float(coords[left][1]) - float(coords[right][1]),
        )

    def _tour_distance(self, coords: list[list[float]], route: list[int]) -> float:
        return sum(self._distance(coords, left, right) for left, right in zip(route, route[1:]))

    def _two_opt(
        self, coords: list[list[float]], route: list[int], max_passes: int
    ) -> tuple[list[int], int]:
        current = list(route)
        passes = 0
        for _ in range(max_passes):
            best_gain = 1e-12
            best_move: tuple[int, int] | None = None
            for left in range(1, len(current) - 2):
                a, b = current[left - 1], current[left]
                for right in range(left + 1, len(current) - 1):
                    c, d = current[right], current[right + 1]
                    gain = (
                        self._distance(coords, a, b)
                        + self._distance(coords, c, d)
                        - self._distance(coords, a, c)
                        - self._distance(coords, b, d)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (left, right)
            if best_move is None:
                break
            left, right = best_move
            current[left : right + 1] = reversed(current[left : right + 1])
            passes += 1
        return current, passes

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        started = time.monotonic()
        coords = row.get("instance_json")
        if not isinstance(coords, list) or len(coords) < 2:
            raise ValueError("TSP backend received an invalid structured instance")
        if parameters.get("starts") != "all":
            raise ValueError("raw TSP heuristic requires starts='all'")
        max_passes = int(parameters.get("max_2opt_passes", -1))
        if max_passes <= 0:
            raise ValueError("raw TSP heuristic max_2opt_passes must be positive")
        candidates: list[tuple[float, list[int], int, int]] = []
        for start in range(len(coords)):
            unvisited = set(range(len(coords)))
            unvisited.remove(start)
            route = [start]
            while unvisited:
                current = route[-1]
                chosen = min(
                    unvisited,
                    key=lambda node: (self._distance(coords, current, node), node),
                )
                route.append(chosen)
                unvisited.remove(chosen)
            route.append(start)
            route, passes = self._two_opt(coords, route, max_passes)
            candidates.append((self._tour_distance(coords, route), route, start, passes))
        objective, route, start, passes = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        cycle = route[:-1]
        depot_index = cycle.index(0)
        route = cycle[depot_index:] + cycle[:depot_index] + [0]
        objective = self._tour_distance(coords, route)
        log = {
            "backend": "clasp_raw_tsp_heuristic",
            "native_status": "HEURISTIC_COMPLETE",
            "candidate_starts": len(candidates),
            "selected_start": start,
            "selected_2opt_passes": passes,
            "raw_euclidean_objective": objective,
        }
        return NativeSolveResult(
            output=f"Route: {route}, Objective: {objective:.17g}",
            native_status="HEURISTIC_COMPLETE",
            native_status_code="HEURISTIC_COMPLETE",
            incumbent=objective,
            best_bound=None,
            gap=None,
            raw_log=_canonical_json_bytes(log) + b"\n",
            objective_semantics=OBJECTIVE_SEMANTICS["tsp"],
            native_wall_time_seconds=time.monotonic() - started,
            native_metadata={
                "candidate_starts": len(candidates),
                "selected_start": start,
                "selected_2opt_passes": passes,
            },
        )


class RawCVRPHeuristicBackend:
    """Deterministic Clarke-Wright heuristic scored with raw Euclidean distance."""

    @staticmethod
    def _distance(coords: list[list[float]], left: int, right: int) -> float:
        return math.hypot(
            float(coords[left][0]) - float(coords[right][0]),
            float(coords[left][1]) - float(coords[right][1]),
        )

    def probe_identity(self) -> dict[str, str]:
        return {
            "backend_name": "clasp_raw_cvrp_heuristic",
            "native_version": "1",
        }

    def _route_distance(self, coords: list[list[float]], route: list[int]) -> float:
        return sum(self._distance(coords, left, right) for left, right in zip(route, route[1:]))

    def _two_opt(
        self, coords: list[list[float]], route: list[int], max_passes: int
    ) -> tuple[list[int], int]:
        current = list(route)
        passes = 0
        for _ in range(max_passes):
            best_gain = 1e-12
            best_move: tuple[int, int] | None = None
            for left in range(1, len(current) - 2):
                a, b = current[left - 1], current[left]
                for right in range(left + 1, len(current) - 1):
                    c, d = current[right], current[right + 1]
                    gain = (
                        self._distance(coords, a, b)
                        + self._distance(coords, c, d)
                        - self._distance(coords, a, c)
                        - self._distance(coords, b, d)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (left, right)
            if best_move is None:
                break
            left, right = best_move
            current[left : right + 1] = reversed(current[left : right + 1])
            passes += 1
        return current, passes

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        started = time.monotonic()
        instance = row.get("instance_json")
        if not isinstance(instance, list) or len(instance) < 3:
            raise ValueError("CVRP backend received an invalid structured instance")
        coords, raw_demands, raw_capacity = instance[:3]
        if not isinstance(coords, list) or not isinstance(raw_demands, list):
            raise ValueError("CVRP backend received malformed coordinates or demands")
        demands = [float(value) for value in raw_demands]
        capacity = float(raw_capacity)
        if len(coords) != len(demands) or len(coords) < 2 or capacity <= 0:
            raise ValueError("CVRP backend received inconsistent instance dimensions")
        if any(value < 0 for value in demands) or any(
            value > capacity + 1e-9 for value in demands[1:]
        ):
            raise ValueError("CVRP instance has a customer demand outside capacity")
        max_passes = int(parameters.get("max_2opt_passes", -1))
        if max_passes <= 0:
            raise ValueError("raw CVRP heuristic max_2opt_passes must be positive")
        routes = [[customer] for customer in range(1, len(coords))]
        savings = sorted(
            (
                (
                    self._distance(coords, 0, left)
                    + self._distance(coords, 0, right)
                    - self._distance(coords, left, right),
                    left,
                    right,
                )
                for left in range(1, len(coords))
                for right in range(left + 1, len(coords))
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        merges = 0
        for _saving, left, right in savings:
            left_idx = next((idx for idx, route in enumerate(routes) if left in route), None)
            right_idx = next((idx for idx, route in enumerate(routes) if right in route), None)
            if left_idx is None or right_idx is None or left_idx == right_idx:
                continue
            left_route = routes[left_idx]
            right_route = routes[right_idx]
            if left not in {left_route[0], left_route[-1]}:
                continue
            if right not in {right_route[0], right_route[-1]}:
                continue
            if sum(demands[node] for node in left_route + right_route) > capacity + 1e-9:
                continue
            if left_route[-1] != left:
                left_route = list(reversed(left_route))
            if right_route[0] != right:
                right_route = list(reversed(right_route))
            merged = left_route + right_route
            for idx in sorted((left_idx, right_idx), reverse=True):
                routes.pop(idx)
            routes.append(merged)
            merges += 1
        closed_routes: list[list[int]] = []
        two_opt_passes = 0
        for route in routes:
            improved, passes = self._two_opt(coords, [0, *route, 0], max_passes)
            closed_routes.append(improved)
            two_opt_passes += passes
        closed_routes.sort(key=lambda route: (route[1] if len(route) > 2 else -1, route))
        objective = sum(self._route_distance(coords, route) for route in closed_routes)
        log = {
            "backend": "clasp_raw_cvrp_heuristic",
            "native_status": "HEURISTIC_COMPLETE",
            "clarke_wright_merges": merges,
            "two_opt_passes": two_opt_passes,
            "routes": len(closed_routes),
            "raw_euclidean_objective": objective,
        }
        return NativeSolveResult(
            output=f"Routes: {closed_routes}, Objective: {objective:.17g}",
            native_status="HEURISTIC_COMPLETE",
            native_status_code="HEURISTIC_COMPLETE",
            incumbent=objective,
            best_bound=None,
            gap=None,
            raw_log=_canonical_json_bytes(log) + b"\n",
            objective_semantics=OBJECTIVE_SEMANTICS["cvrp"],
            native_wall_time_seconds=time.monotonic() - started,
            native_metadata={
                "clarke_wright_merges": merges,
                "two_opt_passes": two_opt_passes,
                "routes": len(closed_routes),
            },
        )


@dataclass(frozen=True)
class OPInstance:
    """Validated benchmark semantics consumed by the exact OP adapter."""

    coordinates: tuple[tuple[float, float], ...]
    prizes: tuple[int, ...]
    route_limit: float


def _validated_op_instance(row: Mapping[str, object]) -> OPInstance:
    instance = row.get("instance_json")
    if not isinstance(instance, list) or len(instance) < 3:
        raise ValueError("OP backend received an invalid structured instance")
    raw_coords, raw_prizes, raw_route_limit = instance[:3]
    if not isinstance(raw_coords, list) or not isinstance(raw_prizes, list):
        raise ValueError("OP backend received malformed coordinates or prizes")
    if len(raw_coords) != len(raw_prizes) or not raw_coords:
        raise ValueError("OP backend received inconsistent instance dimensions")
    coordinates: list[tuple[float, float]] = []
    for coordinate in raw_coords:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            raise ValueError("OP backend received a malformed coordinate")
        point = (float(coordinate[0]), float(coordinate[1]))
        if any(not math.isfinite(value) for value in point):
            raise ValueError("OP backend received a non-finite coordinate")
        coordinates.append(point)
    prizes = tuple(
        _exact_integer(value, label=f"OP prize[{index}]")
        for index, value in enumerate(raw_prizes)
    )
    if any(prize < 0 for prize in prizes):
        raise ValueError("OP backend requires nonnegative integer prizes")
    route_limit = _finite_number(raw_route_limit, label="OP route limit")
    assert route_limit is not None
    if route_limit < 0:
        raise ValueError("OP route limit must be nonnegative")
    return OPInstance(tuple(coordinates), prizes, route_limit)


def _op_pair_distance(instance: OPInstance, left: int, right: int) -> float:
    left_point = instance.coordinates[left]
    right_point = instance.coordinates[right]
    return math.hypot(
        left_point[0] - right_point[0], left_point[1] - right_point[1]
    )


def _op_route_distance(instance: OPInstance, route: Sequence[int]) -> float:
    return sum(
        _op_pair_distance(instance, left, right)
        for left, right in zip(route, route[1:])
    )


def _deterministic_op_start(instance: OPInstance) -> list[int]:
    """Build a deterministic feasible open path used only as a MIP start."""

    route = [0]
    remaining = {
        node for node in range(1, len(instance.coordinates)) if instance.prizes[node] > 0
    }
    length = 0.0
    while remaining:
        current = route[-1]
        feasible: list[tuple[float, int, float, int]] = []
        for node in remaining:
            increment = _op_pair_distance(instance, current, node)
            if length + increment <= instance.route_limit + 1e-12:
                ratio = math.inf if increment == 0.0 else instance.prizes[node] / increment
                feasible.append((ratio, instance.prizes[node], -increment, -node))
        if not feasible:
            break
        chosen = -max(feasible)[3]
        length += _op_pair_distance(instance, current, chosen)
        route.append(chosen)
        remaining.remove(chosen)
    return route


def _op_budget_incompatible_pairs(instance: OPInstance) -> list[tuple[int, int]]:
    """Return customer pairs no budget-feasible depot-rooted path can contain."""

    customers = list(range(1, len(instance.coordinates)))
    incompatible: list[tuple[int, int]] = []
    for left_position, left in enumerate(customers):
        for right in customers[left_position + 1 :]:
            pair_lower_bound = _op_pair_distance(instance, left, right) + min(
                _op_pair_distance(instance, 0, left),
                _op_pair_distance(instance, 0, right),
            )
            if pair_lower_bound > instance.route_limit + 1e-12:
                incompatible.append((left, right))
    return incompatible


class GurobiOPBackend:
    """Exact raw-Euclidean open-orienteering MIP with a dummy sink."""

    def __init__(self) -> None:
        try:
            import gurobipy as gp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("gurobipy is not installed in this runtime") from exc
        self.gp = gp

    def probe_identity(self) -> dict[str, str]:
        version = ".".join(str(part) for part in self.gp.gurobi.version())
        return {
            "backend_name": "gurobi",
            "native_version": version,
            **_installed_distribution_identity("gurobipy", self.gp),
        }

    def solve_diagnostic(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        """Run the exact OP model while retaining every native termination state.

        This method exists only for preregistered development scalability
        audits.  It never promotes an incumbent to an exact reference.  The
        publication evidence path continues to call :meth:`solve`, which
        accepts only a complete native OPTIMAL certificate.
        """

        gp = self.gp
        instance = _validated_op_instance(row)
        node_count = len(instance.coordinates)
        depot = 0
        sink = node_count
        customers = list(range(1, node_count))
        arcs = [(depot, sink)]
        arcs.extend((depot, node) for node in customers)
        arcs.extend((node, sink) for node in customers)
        arcs.extend(
            (left, right)
            for left in customers
            for right in customers
            if left != right
        )
        with tempfile.TemporaryDirectory(prefix="clasp-gurobi-op-") as temp:
            log_path = Path(temp) / "gurobi.log"
            model = gp.Model("clasp_iid_open_orienteering")
            model.Params.TimeLimit = float(parameters["time_limit_seconds"])
            model.Params.Threads = int(parameters["threads"])
            model.Params.Seed = int(parameters["seed"])
            model.Params.Method = int(parameters["method"])
            model.Params.MIPGap = float(parameters["mip_gap"])
            model.Params.MIPGapAbs = float(parameters["mip_gap_abs"])
            if "work_limit" in parameters:
                model.Params.WorkLimit = float(parameters["work_limit"])
            if "node_limit" in parameters:
                model.Params.NodeLimit = float(parameters["node_limit"])
            model.Params.FeasibilityTol = float(parameters.get("feasibility_tolerance", 1e-9))
            model.Params.IntFeasTol = float(parameters.get("integrality_tolerance", 1e-9))
            model.Params.NumericFocus = int(parameters.get("numeric_focus", 3))
            model.Params.LogToConsole = int(parameters.get("log_to_console", 0))
            model.Params.OutputFlag = int(parameters.get("output_flag", 1))
            model.Params.LogFile = str(log_path)

            used = model.addVars(range(node_count), vtype=gp.GRB.BINARY, name="used")
            edge = model.addVars(arcs, vtype=gp.GRB.BINARY, name="edge")
            model.addConstr(used[depot] == 1, name="depot_used")
            model.addConstr(
                gp.quicksum(edge[depot, target] for target in [*customers, sink]) == 1,
                name="leave_depot_once",
            )
            model.addConstr(
                gp.quicksum(edge[source, sink] for source in [depot, *customers]) == 1,
                name="enter_sink_once",
            )
            for node in customers:
                incoming = gp.quicksum(
                    edge[source, node]
                    for source in [depot, *customers]
                    if source != node
                )
                outgoing = gp.quicksum(
                    edge[node, target]
                    for target in [*customers, sink]
                    if target != node
                )
                model.addConstr(incoming == used[node], name=f"incoming_{node}")
                model.addConstr(outgoing == used[node], name=f"outgoing_{node}")

            arrival = None
            if customers:
                # A single commodity leaves the depot and one unit is consumed
                # at every selected customer.  Together with the degree equations
                # this excludes disconnected cycles and is substantially tighter
                # than an MTZ ordering relaxation on the 100-node publication set.
                flow_arcs = [(left, right) for left, right in arcs if right != sink]
                flow = model.addVars(
                    flow_arcs,
                    lb=0.0,
                    ub=float(len(customers)),
                    vtype=gp.GRB.CONTINUOUS,
                    name="connectivity_flow",
                )
                for left, right in flow_arcs:
                    model.addConstr(
                        flow[left, right] <= len(customers) * edge[left, right],
                        name=f"flow_capacity_{left}_{right}",
                    )
                model.addConstr(
                    gp.quicksum(flow[depot, node] for node in customers)
                    == gp.quicksum(used[node] for node in customers),
                    name="depot_flow_supply",
                )
                for node in customers:
                    incoming_flow = gp.quicksum(
                        flow[source, node]
                        for source in [depot, *customers]
                        if source != node
                    )
                    outgoing_flow = gp.quicksum(
                        flow[node, target]
                        for target in customers
                        if target != node
                    )
                    model.addConstr(
                        incoming_flow - outgoing_flow == used[node],
                        name=f"flow_consumption_{node}",
                    )

                # Redundant resource-ordering variables expose cumulative raw
                # distance along the selected path.  The aggregate budget
                # below remains the defining constraint; these indicators
                # preserve exactly the same integer routes while giving the
                # MIP solver substantially more local propagation.
                arrival = model.addVars(
                    customers,
                    lb=0.0,
                    ub=instance.route_limit,
                    vtype=gp.GRB.CONTINUOUS,
                    name="arrival_distance",
                )
                for node in customers:
                    model.addConstr(
                        arrival[node] <= instance.route_limit * used[node],
                        name=f"arrival_selected_{node}",
                    )
                    model.addConstr(
                        arrival[node]
                        >= _op_pair_distance(instance, depot, node) * used[node],
                        name=f"arrival_depot_lower_bound_{node}",
                    )
                    model.addGenConstrIndicator(
                        edge[depot, node],
                        True,
                        arrival[node]
                        >= _op_pair_distance(instance, depot, node),
                        name=f"arrival_from_depot_{node}",
                    )
                for left in customers:
                    for right in customers:
                        if left == right:
                            continue
                        model.addGenConstrIndicator(
                            edge[left, right],
                            True,
                            arrival[right]
                            >= arrival[left]
                            + _op_pair_distance(instance, left, right),
                            name=f"arrival_arc_{left}_{right}",
                        )

            # If even the shortest possible depot-to-{i,j} open path exceeds
            # the budget, the pair cannot be selected.  These O(n^2) valid
            # inequalities materially strengthen the relaxation while using
            # only the raw metric already consumed by the strict verifier.
            incompatible_pairs = _op_budget_incompatible_pairs(instance)
            for left, right in incompatible_pairs:
                model.addConstr(
                    used[left] + used[right] <= 1,
                    name=f"budget_incompatible_{left}_{right}",
                )

            model.addConstr(
                gp.quicksum(
                    _op_pair_distance(instance, left, right) * edge[left, right]
                    for left, right in arcs
                    if right != sink
                )
                <= instance.route_limit,
                name="raw_euclidean_budget",
            )
            model.setObjective(
                gp.quicksum(
                    instance.prizes[node] * used[node] for node in range(node_count)
                ),
                gp.GRB.MAXIMIZE,
            )

            start_route = _deterministic_op_start(instance)
            start_edges = set(zip(start_route, [*start_route[1:], sink]))
            start_nodes = set(start_route)
            for node in range(node_count):
                used[node].Start = float(node in start_nodes)
            for arc in arcs:
                edge[arc].Start = float(arc in start_edges)
            if customers:
                selected_customers = len(start_route) - 1
                for arc in flow_arcs:
                    flow[arc].Start = 0.0
                for position, arc in enumerate(zip(start_route, start_route[1:])):
                    flow[arc].Start = float(selected_customers - position)
                assert arrival is not None
                for node in customers:
                    arrival[node].Start = 0.0
                cumulative_distance = 0.0
                for left, right in zip(start_route, start_route[1:]):
                    cumulative_distance += _op_pair_distance(instance, left, right)
                    arrival[right].Start = cumulative_distance
            model.optimize()

            status_code = int(model.Status)
            names = {
                int(gp.GRB.OPTIMAL): "OPTIMAL",
                int(gp.GRB.TIME_LIMIT): "TIME_LIMIT",
                int(gp.GRB.WORK_LIMIT): "WORK_LIMIT",
                int(gp.GRB.NODE_LIMIT): "NODE_LIMIT",
                int(gp.GRB.INFEASIBLE): "INFEASIBLE",
                int(gp.GRB.INTERRUPTED): "INTERRUPTED",
            }
            solver_status = names.get(status_code, f"STATUS_{status_code}")
            solution_count = int(model.SolCount)
            route: list[int] = []
            objective: float | None = None
            model_objective: float | None = None
            output = ""
            if solution_count > 0:
                successor = {
                    left: right
                    for left, right in arcs
                    if float(edge[left, right].X) > 0.5
                }
                route = [depot]
                current = depot
                while True:
                    if current not in successor:
                        raise RuntimeError("OP solver solution has no path successor")
                    current = successor[current]
                    if current == sink:
                        break
                    if current in route or len(route) > node_count:
                        raise RuntimeError("OP solver solution contains a subtour")
                    route.append(current)
                selected_nodes = {
                    node for node in range(node_count) if float(used[node].X) > 0.5
                }
                if selected_nodes != set(route):
                    raise RuntimeError("OP path extraction disagrees with selected nodes")
                objective = float(sum(instance.prizes[node] for node in route))
                model_objective = float(model.ObjVal)
                output = f"Route: {route}, Objective: {objective:.17g}"
            bound_value = float(model.ObjBound)
            bound = bound_value if math.isfinite(bound_value) else None
            raw_gap = float(model.MIPGap) if solution_count > 0 else math.inf
            gap = raw_gap if math.isfinite(raw_gap) else None
            native_runtime = float(model.Runtime)
            deterministic_work = float(model.Work)
            search_nodes = float(model.NodeCount)
            model_variables = int(model.NumVars)
            model_binary_variables = int(model.NumBinVars)
            model_linear_constraints = int(model.NumConstrs)
            model_general_constraints = int(model.NumGenConstrs)
            model_nonzeros = int(model.NumNZs)
            model.dispose()
            raw_log = log_path.read_bytes() if log_path.exists() else b""
        route_length: float | None = None
        if solution_count > 0:
            assert objective is not None and model_objective is not None
            _objective_matches("op", model_objective, objective, label="OP model objective")
            route_length = _op_route_distance(instance, route)
            if route_length > instance.route_limit + 1e-9:
                raise RuntimeError("OP exact MIP route violates the raw Euclidean budget")
        if not raw_log:
            raise RuntimeError("OP exact MIP produced no native Gurobi log")
        return NativeSolveResult(
            output=output,
            native_status=solver_status,
            native_status_code=status_code,
            incumbent=objective,
            best_bound=bound,
            gap=gap,
            raw_log=raw_log,
            objective_semantics=OBJECTIVE_SEMANTICS["op"],
            native_wall_time_seconds=native_runtime,
            native_metadata={
                "solution_count": solution_count,
                "node_count": search_nodes,
                "deterministic_work": deterministic_work,
                "model_variables": model_variables,
                "model_binary_variables": model_binary_variables,
                "model_linear_constraints": model_linear_constraints,
                "model_general_constraints": model_general_constraints,
                "model_nonzeros": model_nonzeros,
                "gurobi_status": solver_status,
                "gurobi_status_code": status_code,
                "gurobi_best_bound": bound,
                "gurobi_gap": gap,
                "route_nodes": len(route),
                "route_length": route_length,
                "route_limit": instance.route_limit,
                "mip_start_route": start_route,
                "mip_start_objective": sum(
                    instance.prizes[node] for node in start_route
                ),
                "budget_incompatible_pair_constraints": len(incompatible_pairs),
                "model_sense": "maximize",
                "path_semantics": "start_at_depot_open_end",
                "distance_semantics": "raw_euclidean_float64",
            },
        )

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        """Return only a solver-certified exact OP reference.

        Resource-limit and all other non-OPTIMAL states remain hard failures
        for the reference-evidence runner.  Diagnostic callers must use the
        explicitly non-publishable :meth:`solve_diagnostic` surface.
        """

        result = self.solve_diagnostic(row, parameters=parameters)
        metadata = result.native_metadata
        solution_count = (
            metadata.get("solution_count") if isinstance(metadata, dict) else None
        )
        if not isinstance(solution_count, int) or solution_count <= 0:
            raise RuntimeError("OP exact MIP produced no feasible incumbent")
        if result.native_status != "OPTIMAL":
            raise RuntimeError(
                "OP exact MIP did not certify optimality; refusing a best-known "
                "fallback "
                f"(status={result.native_status}, bound={result.best_bound}, "
                f"gap={result.gap})"
            )
        if result.native_status_code != 2:
            raise RuntimeError("OP exact MIP returned OPTIMAL without Gurobi status code 2")
        if result.incumbent is None or result.best_bound is None:
            raise RuntimeError("OP exact MIP returned OPTIMAL without incumbent and bound")
        _objective_matches(
            "op",
            result.best_bound,
            result.incumbent,
            label="OP native bound",
        )
        if result.gap != 0.0:
            raise RuntimeError("OP exact MIP returned OPTIMAL with a nonzero native gap")
        return result


class GurobiMISBackend:
    def __init__(self) -> None:
        try:
            import gurobipy as gp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("gurobipy is not installed in this runtime") from exc
        self.gp = gp

    def probe_identity(self) -> dict[str, str]:
        version = ".".join(str(part) for part in self.gp.gurobi.version())
        return {
            "backend_name": "gurobi",
            "native_version": version,
            **_installed_distribution_identity("gurobipy", self.gp),
        }

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        gp = self.gp
        instance = row.get("instance_json")
        if not isinstance(instance, dict):
            raise ValueError("MIS backend received an invalid structured instance")
        num_nodes = int(instance.get("num_nodes", -1))
        edges = instance.get("edges")
        if num_nodes <= 0 or not isinstance(edges, list):
            raise ValueError("MIS backend received an incomplete graph")
        with tempfile.TemporaryDirectory(prefix="clasp-gurobi-mis-") as temp:
            log_path = Path(temp) / "gurobi.log"
            model = gp.Model("clasp_iid_mis")
            model.Params.TimeLimit = float(parameters["time_limit_seconds"])
            model.Params.Threads = int(parameters["threads"])
            model.Params.Seed = int(parameters["seed"])
            model.Params.Method = int(parameters["method"])
            model.Params.MIPGap = float(parameters["mip_gap"])
            model.Params.MIPGapAbs = float(parameters["mip_gap_abs"])
            model.Params.FeasibilityTol = float(parameters["feasibility_tolerance"])
            model.Params.IntFeasTol = float(parameters["integrality_tolerance"])
            model.Params.NumericFocus = int(parameters["numeric_focus"])
            model.Params.LogToConsole = int(parameters.get("log_to_console", 0))
            model.Params.OutputFlag = int(parameters.get("output_flag", 1))
            model.Params.LogFile = str(log_path)
            variables = model.addVars(num_nodes, vtype=gp.GRB.BINARY, name="selected")
            for edge in edges:
                if not isinstance(edge, list) or len(edge) != 2:
                    raise ValueError("MIS backend received a malformed edge")
                left, right = int(edge[0]), int(edge[1])
                model.addConstr(variables[left] + variables[right] <= 1)
            model.setObjective(variables.sum(), gp.GRB.MAXIMIZE)
            model.optimize()
            status_code = int(model.Status)
            names = {
                int(gp.GRB.OPTIMAL): "OPTIMAL",
                int(gp.GRB.TIME_LIMIT): "TIME_LIMIT",
                int(gp.GRB.INFEASIBLE): "INFEASIBLE",
                int(gp.GRB.INTERRUPTED): "INTERRUPTED",
            }
            status = names.get(status_code, f"STATUS_{status_code}")
            solution_count = int(model.SolCount)
            if solution_count > 0:
                selected = [node for node in range(num_nodes) if variables[node].X > 0.5]
                incumbent: float | None = float(model.ObjVal)
                output = f"Set: {selected}, Objective: {len(selected)}"
            else:
                incumbent = None
                output = ""
            bound = float(model.ObjBound) if math.isfinite(float(model.ObjBound)) else None
            gap = float(model.MIPGap) if solution_count > 0 else None
            native_runtime = float(model.Runtime)
            node_count = float(model.NodeCount)
            model.dispose()
            raw_log = log_path.read_bytes() if log_path.exists() else b""
        return NativeSolveResult(
            output=output,
            native_status=status,
            native_status_code=status_code,
            incumbent=incumbent,
            best_bound=bound,
            gap=gap,
            raw_log=raw_log,
            objective_semantics=OBJECTIVE_SEMANTICS["mis"],
            native_wall_time_seconds=native_runtime,
            native_metadata={
                "solution_count": solution_count,
                "node_count": node_count,
                "model_sense": "maximize",
            },
        )


class ORToolsJSSPBackend:
    def __init__(self) -> None:
        try:
            import ortools  # type: ignore[import-not-found]
            from ortools.sat.python import cp_model  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OR-Tools is not installed in this runtime") from exc
        self.ortools = ortools
        self.cp_model = cp_model

    def probe_identity(self) -> dict[str, str]:
        return {
            "backend_name": "ortools_cp_sat",
            "native_version": str(self.ortools.__version__),
            **_installed_distribution_identity("ortools", self.ortools),
        }

    def solve(
        self, row: dict[str, object], *, parameters: dict[str, object]
    ) -> NativeSolveResult:
        cp_model = self.cp_model
        instance = row.get("instance_json")
        if not isinstance(instance, list) or not instance:
            raise ValueError("JSSP backend received an invalid structured instance")
        jobs: list[list[tuple[int, int]]] = []
        for raw_job in instance:
            if not isinstance(raw_job, list) or len(raw_job) % 2:
                raise ValueError("JSSP backend received a malformed job")
            job = [
                (int(raw_job[index]), int(raw_job[index + 1]))
                for index in range(0, len(raw_job), 2)
            ]
            jobs.append(job)
        machine_count = len(jobs[0])
        if machine_count <= 0 or any(len(job) != machine_count for job in jobs):
            raise ValueError("JSSP backend received inconsistent machine counts")
        horizon = sum(duration for job in jobs for _, duration in job)
        model = cp_model.CpModel()
        starts: dict[tuple[int, int], Any] = {}
        ends: dict[tuple[int, int], Any] = {}
        machine_intervals: dict[int, list[Any]] = {machine: [] for machine in range(machine_count)}
        machine_ops: dict[int, list[tuple[int, int]]] = {
            machine: [] for machine in range(machine_count)
        }
        for job_index, job in enumerate(jobs):
            previous_end = None
            for operation_index, (machine, duration) in enumerate(job):
                key = (job_index, operation_index)
                start = model.NewIntVar(0, horizon, f"start_{job_index}_{operation_index}")
                end = model.NewIntVar(0, horizon, f"end_{job_index}_{operation_index}")
                interval = model.NewIntervalVar(
                    start, duration, end, f"interval_{job_index}_{operation_index}"
                )
                starts[key], ends[key] = start, end
                machine_intervals[machine].append(interval)
                machine_ops[machine].append(key)
                if previous_end is not None:
                    model.Add(start >= previous_end)
                previous_end = end
        for machine in range(machine_count):
            model.AddNoOverlap(machine_intervals[machine])
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(
            makespan, [ends[(job_index, len(job) - 1)] for job_index, job in enumerate(jobs)]
        )
        model.Minimize(makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(parameters["max_time_seconds"])
        solver.parameters.num_search_workers = int(parameters["num_search_workers"])
        solver.parameters.random_seed = int(parameters["random_seed"])
        solver.parameters.absolute_gap_limit = float(parameters["absolute_gap_limit"])
        solver.parameters.relative_gap_limit = float(parameters["relative_gap_limit"])
        solver.parameters.log_search_progress = parameters["log_search_progress"]
        started = time.monotonic()
        status_code = int(solver.Solve(model))
        measured = time.monotonic() - started
        status = str(solver.StatusName(status_code)).upper()
        response_stats = str(solver.ResponseStats())
        incumbent: float | None = None
        bound: float | None = None
        output = ""
        if status in {"OPTIMAL", "FEASIBLE"}:
            incumbent = float(solver.ObjectiveValue())
            bound = float(solver.BestObjectiveBound())
            schedule: list[list[int]] = []
            for machine in range(machine_count):
                ordered = sorted(
                    machine_ops[machine],
                    key=lambda key: (int(solver.Value(starts[key])), key[0]),
                )
                schedule.append([job for job, _operation in ordered])
            output = f"Schedule: {schedule}, Makespan: {int(round(incumbent))}"
        gap = None
        if incumbent is not None and bound is not None:
            gap = abs(incumbent - bound) / max(1.0, abs(incumbent))
        return NativeSolveResult(
            output=output,
            native_status=status,
            native_status_code=status_code,
            incumbent=incumbent,
            best_bound=bound,
            gap=gap,
            raw_log=response_stats.encode("utf-8"),
            objective_semantics=OBJECTIVE_SEMANTICS["jssp"],
            native_wall_time_seconds=float(solver.WallTime()),
            native_metadata={
                "response_stats_sha256": hashlib.sha256(
                    response_stats.encode("utf-8")
                ).hexdigest(),
                "measured_solve_seconds": measured,
            },
        )


def _make_backend(task: str, adapter: object) -> EvidenceBackend:
    expected = {
        "tsp": "raw_euclidean_tsp_heuristic_v1",
        "cvrp": "raw_euclidean_cvrp_heuristic_v1",
        "op": OP_EXACT_ADAPTER,
        "mis": "gurobi_mis_v1",
        "jssp": "ortools_jssp_v1",
    }[task]
    if adapter != expected:
        raise ValueError(
            f"{task} evidence adapter must be {expected!r}; got {adapter!r}"
        )
    return {
        "tsp": RawTSPHeuristicBackend,
        "cvrp": RawCVRPHeuristicBackend,
        "op": GurobiOPBackend,
        "mis": GurobiMISBackend,
        "jssp": ORToolsJSSPBackend,
    }[task]()


def solve_task(
    *,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    task: str,
    output_dir: str | Path,
    backend: EvidenceBackend | None = None,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Solve one complete task into a new, atomically committed evidence directory."""

    evaluators, verifier_identity = verifier_bundle or _default_verifier_bundle()
    if task not in evaluators:
        raise ValueError(f"strict verifier bundle does not define task {task}")
    # All contract, draft, artifact, status, and semantics gates run before a
    # backend is constructed or an injected backend method is called.
    prepared = prepare_run(
        draft_root=draft_root,
        solver_contract_path=solver_contract_path,
        task=task,
        verifier_identity=verifier_identity,
    )
    selected_backend = backend or _make_backend(task, prepared.runner_contract.get("adapter"))
    backend_identity = _verify_backend_identity(prepared, selected_backend)
    runner_code_sha256 = str(_file_identity(__file__)["sha256"])
    completed_summary: dict[str, object] = {}

    def write(destination: Path) -> None:
        nonlocal completed_summary
        _write_json_new(destination / "solver_contract.json", prepared.solver_contract)
        _write_json_new(
            destination / "source_generation_receipt.json", prepared.generation_receipt
        )
        records: list[dict[str, object]] = []
        log_manifest: list[dict[str, object]] = []
        for row_index, row in enumerate(prepared.rows):
            started = time.monotonic()
            result = selected_backend.solve(row, parameters=copy.deepcopy(prepared.parameters))
            measured_wall = time.monotonic() - started
            if not math.isfinite(measured_wall) or measured_wall < 0:
                raise ValueError("measured solver wall time is invalid")
            if not isinstance(result, NativeSolveResult):
                raise TypeError("backend solve must return NativeSolveResult")
            row_id = str(row["benchmark_row_id"])
            log_relative = f"native_logs/{row_id}.log"
            log_path = destination / log_relative
            _write_bytes_new(log_path, result.raw_log)
            log_identity = _file_identity(log_path)
            record = _record_from_result(
                prepared=prepared,
                row=row,
                row_index=row_index,
                result=result,
                measured_wall_time=measured_wall,
                log_relative_path=log_relative,
                log_identity=log_identity,
                evaluator=evaluators[task],
                backend_identity=backend_identity,
                runner_code_sha256=runner_code_sha256,
            )
            records.append(record)
            log_manifest.append(
                {
                    "benchmark_row_id": row_id,
                    "relative_path": log_relative,
                    **log_identity,
                }
            )
        _write_json_new(destination / "details.json", records)
        details_identity = {
            "relative_path": "details.json",
            **_file_identity(destination / "details.json"),
            "record_count": len(records),
        }
        solver_snapshot_identity = {
            "relative_path": "solver_contract.json",
            **_file_identity(destination / "solver_contract.json"),
        }
        receipt_snapshot_identity = {
            "relative_path": "source_generation_receipt.json",
            **_file_identity(destination / "source_generation_receipt.json"),
        }
        summary = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "complete_reference_evidence",
            "publishable": False,
            "contains_model_inference": False,
            "task": task,
            "rows": len(records),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner_protocol": RUNNER_PROTOCOL,
            "runner_code_sha256": runner_code_sha256,
            "source_generation_receipt_sha256": prepared.generation_receipt[
                "receipt_sha256"
            ],
            "solver_contract_sha256": prepared.solver_contract["contract_sha256"],
            "solver_contract_task_sha256": sha256_json(prepared.task_contract),
            "solver_contract_snapshot": solver_snapshot_identity,
            "generation_receipt_snapshot": receipt_snapshot_identity,
            "solver_identity": {**prepared.solver_artifact, **backend_identity},
            "solver_parameters": prepared.parameters,
            "solver_parameters_sha256": sha256_json(prepared.parameters),
            "solver_seed": prepared.solver_seed,
            "objective_semantics": OBJECTIVE_SEMANTICS[task],
            "reference_quality": REFERENCE_QUALITY[task],
            "strict_verifier_identity": prepared.verifier_identity,
            "strict_verifier_identity_sha256": sha256_json(prepared.verifier_identity),
            "native_status_counts": dict(
                sorted(Counter(str(record["native_status"]) for record in records).items())
            ),
            "details_artifact": details_identity,
            "raw_native_logs": {
                "count": len(log_manifest),
                "manifest_sha256": sha256_json(log_manifest),
                "files": log_manifest,
            },
            "ordered_reference_evidence_record_sha256": [
                record["reference_evidence_record_sha256"] for record in records
            ],
        }
        summary["reference_evidence_records_sha256"] = sha256_json(
            summary["ordered_reference_evidence_record_sha256"]
        )
        completed_summary = _self_hash(summary, "reference_evidence_summary_sha256")
        _write_json_new(destination / "summary.json", completed_summary)
        validate_evidence_directory(
            evidence_dir=destination,
            draft_root=prepared.draft_root,
            solver_contract_path=destination / "solver_contract.json",
            verifier_bundle=(evaluators, verifier_identity),
        )

    _atomic_directory(output_dir, write)
    return completed_summary


def _verify_identity_snapshot(
    root: Path, descriptor: object, *, expected: dict[str, object], label: str
) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} snapshot descriptor is missing")
    path = _safe_relative_path(root, descriptor.get("relative_path"), label=label)
    observed = _file_identity(path)
    if descriptor.get("sha256") != observed["sha256"] or descriptor.get(
        "size_bytes"
    ) != observed["size_bytes"]:
        raise ValueError(f"{label} snapshot artifact identity mismatch")
    loaded = _load_object(path, label=label)
    if loaded != expected:
        raise ValueError(f"{label} snapshot content mismatch")


def _record_mapping_for_acceptance(
    record: dict[str, object], *, raw_log: bytes
) -> dict[str, object]:
    return {
        "native_status": record.get("native_status"),
        "native_status_code": record.get("native_status_code"),
        "incumbent": record.get("native_incumbent"),
        "best_bound": record.get("native_best_bound"),
        "gap": record.get("native_gap"),
        "objective_semantics": record.get("objective_semantics"),
        "native_metadata": record.get("native_metadata"),
        "raw_log": raw_log,
    }


def validate_evidence_directory(
    *,
    evidence_dir: str | Path,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fail closed on any stale, altered, missing, or semantically invalid evidence."""

    root = Path(evidence_dir).resolve()
    summary = _load_object(root / "summary.json", label="reference evidence summary")
    _verify_self_hash(
        summary, "reference_evidence_summary_sha256", label="reference evidence summary"
    )
    if summary.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("reference evidence has an unsupported schema version")
    if summary.get("status") != "complete_reference_evidence":
        raise ValueError("reference evidence summary is not complete")
    task = str(summary.get("task", ""))
    evaluators, verifier_identity = verifier_bundle or _default_verifier_bundle()
    if task not in evaluators:
        raise ValueError(f"strict verifier bundle does not define evidence task {task}")
    prepared = prepare_run(
        draft_root=draft_root,
        solver_contract_path=solver_contract_path,
        task=task,
        verifier_identity=verifier_identity,
    )
    expected_summary_fields = {
        "runner_protocol": RUNNER_PROTOCOL,
        "runner_code_sha256": str(_file_identity(__file__)["sha256"]),
        "source_generation_receipt_sha256": prepared.generation_receipt["receipt_sha256"],
        "solver_contract_sha256": prepared.solver_contract["contract_sha256"],
        "solver_contract_task_sha256": sha256_json(prepared.task_contract),
        "solver_parameters_sha256": sha256_json(prepared.parameters),
        "solver_seed": prepared.solver_seed,
        "objective_semantics": OBJECTIVE_SEMANTICS[task],
        "reference_quality": REFERENCE_QUALITY[task],
        "strict_verifier_identity_sha256": sha256_json(verifier_identity),
        "rows": len(prepared.rows),
    }
    for field, expected in expected_summary_fields.items():
        if summary.get(field) != expected:
            raise ValueError(f"reference evidence summary field {field} is stale or invalid")
    if summary.get("solver_parameters") != prepared.parameters:
        raise ValueError("reference evidence summary solver parameters are stale")
    expected_solver_identity = prepared.solver_artifact
    actual_solver_identity = summary.get("solver_identity")
    if not isinstance(actual_solver_identity, dict):
        raise ValueError("reference evidence summary solver identity is missing")
    for field, expected in expected_solver_identity.items():
        if actual_solver_identity.get(field) != expected:
            raise ValueError(f"reference evidence solver identity field {field} is stale")
    _verify_identity_snapshot(
        root,
        summary.get("solver_contract_snapshot"),
        expected=prepared.solver_contract,
        label="solver contract",
    )
    _verify_identity_snapshot(
        root,
        summary.get("generation_receipt_snapshot"),
        expected=prepared.generation_receipt,
        label="generation receipt",
    )
    details = summary.get("details_artifact")
    if not isinstance(details, dict):
        raise ValueError("reference evidence summary has no details artifact")
    details_path = _safe_relative_path(root, details.get("relative_path"), label="details")
    details_identity = _file_identity(details_path)
    records = _load_rows(details_path, label="reference evidence details")
    if (
        details.get("sha256") != details_identity["sha256"]
        or details.get("size_bytes") != details_identity["size_bytes"]
        or details.get("record_count") != len(records)
        or len(records) != len(prepared.rows)
    ):
        raise ValueError("reference evidence details artifact identity mismatch")
    log_bundle = summary.get("raw_native_logs")
    if not isinstance(log_bundle, dict) or not isinstance(log_bundle.get("files"), list):
        raise ValueError("reference evidence raw-log manifest is missing")
    log_files = log_bundle["files"]
    if log_bundle.get("count") != len(log_files) or len(log_files) != len(records):
        raise ValueError("reference evidence raw-log count mismatch")
    if log_bundle.get("manifest_sha256") != sha256_json(log_files):
        raise ValueError("reference evidence raw-log manifest hash mismatch")
    ordered_record_hashes: list[str] = []
    observed_log_manifest: list[dict[str, object]] = []
    for index, (record, row) in enumerate(zip(records, prepared.rows)):
        _verify_self_hash(
            record,
            "reference_evidence_record_sha256",
            label=f"reference evidence record {index}",
        )
        semantic_hash = semantic_instance_sha256(task, row, require_structured=True)
        expected_fields = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "task": task,
            "row_index": index,
            "benchmark_row_id": row["benchmark_row_id"],
            "instance_sha256": semantic_hash,
            "semantic_instance_sha256": semantic_hash,
            "benchmark_row_sha256": benchmark_row_sha256(row),
            "reference_output_sha256": reference_output_sha256(record.get("output")),
            "solver_status": PUBLIC_STATUS[task],
            "reference_quality": REFERENCE_QUALITY[task],
            "solver_contract_sha256": prepared.solver_contract["contract_sha256"],
            "solver_contract_task_sha256": sha256_json(prepared.task_contract),
            "runner_protocol": RUNNER_PROTOCOL,
            "runner_code_sha256": str(_file_identity(__file__)["sha256"]),
            "solver_parameters_sha256": sha256_json(prepared.parameters),
            "solver_seed": prepared.solver_seed,
            "objective_semantics": OBJECTIVE_SEMANTICS[task],
            "strict_verifier_identity_sha256": sha256_json(verifier_identity),
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise ValueError(f"reference evidence row {index} field {field} is invalid")
        if record.get("solver_parameters") != prepared.parameters:
            raise ValueError(f"reference evidence row {index} solver parameters are stale")
        record_identity = record.get("solver_identity")
        if not isinstance(record_identity, dict):
            raise ValueError(f"reference evidence row {index} solver identity is missing")
        for field, expected in expected_solver_identity.items():
            if record_identity.get(field) != expected:
                raise ValueError(
                    f"reference evidence row {index} solver identity field {field} is stale"
                )
        strict_metrics, verified_objective = _strict_verify(
            task, row, str(record.get("output", "")), evaluators[task]
        )
        if record.get("strict_verifier_identity") != verifier_identity:
            raise ValueError(f"reference evidence row {index} verifier identity is stale")
        if record.get("strict_verifier_result") != strict_metrics:
            raise ValueError(f"reference evidence row {index} strict result is stale")
        if record.get("strict_verifier_result_sha256") != sha256_json(strict_metrics):
            raise ValueError(f"reference evidence row {index} strict result hash is invalid")
        _objective_matches(
            task,
            record.get("strict_verified_objective"),
            verified_objective,
            label="stored strict objective",
        )
        raw_log = record.get("raw_native_log")
        if not isinstance(raw_log, dict):
            raise ValueError(f"reference evidence row {index} raw log identity is missing")
        log_path = _safe_relative_path(root, raw_log.get("relative_path"), label="native log")
        log_identity = _file_identity(log_path)
        if log_identity["size_bytes"] <= 0:
            raise ValueError(f"reference evidence row {index} native log is empty")
        if (
            raw_log.get("sha256") != log_identity["sha256"]
            or raw_log.get("size_bytes") != log_identity["size_bytes"]
        ):
            raise ValueError(f"reference evidence row {index} native log identity mismatch")
        raw_log_bytes = log_path.read_bytes()
        solver_status, quality, certificate = _validate_native_acceptance(
            task,
            _record_mapping_for_acceptance(record, raw_log=raw_log_bytes),
            verified_objective=verified_objective,
        )
        if (
            record.get("solver_status") != solver_status
            or record.get("reference_quality") != quality
            or record.get("optimality_certificate") is not certificate
        ):
            raise ValueError(f"reference evidence row {index} quality classification is invalid")
        wall = _finite_number(record.get("wall_time_seconds"), label="measured wall time")
        if wall is None or wall < 0:
            raise ValueError(f"reference evidence row {index} wall time is invalid")
        observed_log_manifest.append(
            {
                "benchmark_row_id": row["benchmark_row_id"],
                "relative_path": raw_log["relative_path"],
                **log_identity,
            }
        )
        ordered_record_hashes.append(str(record["reference_evidence_record_sha256"]))
    if log_files != observed_log_manifest:
        raise ValueError("reference evidence raw-log manifest does not match row records")
    if summary.get("ordered_reference_evidence_record_sha256") != ordered_record_hashes:
        raise ValueError("reference evidence ordered record hashes are invalid")
    if summary.get("reference_evidence_records_sha256") != sha256_json(
        ordered_record_hashes
    ):
        raise ValueError("reference evidence aggregate record hash is invalid")
    expected_counts = dict(
        sorted(Counter(str(record["native_status"]) for record in records).items())
    )
    if summary.get("native_status_counts") != expected_counts:
        raise ValueError("reference evidence native status counts are invalid")
    return summary, records


def _evidence_tree_identity(evidence_dir: str | Path) -> dict[str, object]:
    root = Path(evidence_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"reference evidence directory is missing: {root}")
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"reference evidence tree contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"reference evidence tree contains a special file: {path}")
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                **_file_identity(path),
            }
        )
    if not files:
        raise ValueError(f"reference evidence tree is empty: {root}")
    return {
        "file_count": len(files),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "manifest_sha256": sha256_json(files),
        "files": files,
    }


def _validated_all_task_evidence(
    *,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    evidence_dirs: Mapping[str, str | Path],
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, Callable[..., dict]],
    dict[str, object],
    dict[str, tuple[dict[str, object], list[dict[str, object]], dict[str, object]]],
]:
    if set(evidence_dirs) != set(TASKS):
        raise ValueError("reference approval requires exactly all five task evidence directories")
    evaluators, verifier_identity = verifier_bundle or _default_verifier_bundle()
    contract = _load_object(solver_contract_path, label="reference solver contract")
    validate_reference_solver_contract_policy(
        contract,
        verifier_identity_sha256=sha256_json(verifier_identity),
        expected_tasks=TASKS,
        expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
        require_production_identity=True,
    )
    draft_receipt = _load_object(
        Path(draft_root) / "generation_receipt.json", label="generation receipt"
    )
    _verify_self_hash(draft_receipt, "receipt_sha256", label="generation receipt")
    validated: dict[
        str, tuple[dict[str, object], list[dict[str, object]], dict[str, object]]
    ] = {}
    for task in TASKS:
        summary, records = validate_evidence_directory(
            evidence_dir=evidence_dirs[task],
            draft_root=draft_root,
            solver_contract_path=solver_contract_path,
            verifier_bundle=(evaluators, verifier_identity),
        )
        tree = _evidence_tree_identity(evidence_dirs[task])
        validated[task] = (summary, records, tree)
    return contract, draft_receipt, evaluators, verifier_identity, validated


def approve_reference_evidence(
    *,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    evidence_dirs: Mapping[str, str | Path],
    output_path: str | Path,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Create the one immutable receipt that closes the all-row evidence gate."""

    contract, draft_receipt, _evaluators, verifier_identity, validated = (
        _validated_all_task_evidence(
            draft_root=draft_root,
            solver_contract_path=solver_contract_path,
            evidence_dirs=evidence_dirs,
            verifier_bundle=verifier_bundle,
        )
    )
    tasks: dict[str, object] = {}
    for task in sorted(TASKS):
        summary, records, tree = validated[task]
        tasks[task] = {
            "rows": len(records),
            "solver_contract_task_sha256": summary["solver_contract_task_sha256"],
            "reference_evidence_summary_sha256": summary[
                "reference_evidence_summary_sha256"
            ],
            "reference_quality": REFERENCE_QUALITY[task],
            "required_public_status": PUBLIC_STATUS[task],
            "required_native_status": REQUIRED_NATIVE_STATUS[task],
            "native_status_counts": summary["native_status_counts"],
            "evidence_tree": tree,
        }
    payload = {
        "schema_version": REFERENCE_EVIDENCE_APPROVAL_SCHEMA_VERSION,
        "status": REFERENCE_EVIDENCE_APPROVAL_STATUS,
        "approval_scope": "exactly_five_tasks_x_1000_rows",
        "publishable": False,
        "contains_model_inference": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_protocol": RUNNER_PROTOCOL,
        "runner_code_sha256": _file_identity(__file__)["sha256"],
        "source_generation_receipt_sha256": draft_receipt["receipt_sha256"],
        "solver_contract_sha256": contract["contract_sha256"],
        "solver_contract_status": REFERENCE_EVIDENCE_CONTRACT_STATUS,
        "strict_verifier_identity_sha256": sha256_json(verifier_identity),
        "tasks": tasks,
    }
    receipt = _self_hash(payload, "reference_evidence_approval_receipt_sha256")
    validate_reference_evidence_approval_receipt(
        receipt,
        solver_contract=contract,
        source_generation_receipt_sha256=str(draft_receipt["receipt_sha256"]),
        verifier_identity_sha256=sha256_json(verifier_identity),
        task_evidence_summary_sha256={
            task: str(validated[task][0]["reference_evidence_summary_sha256"])
            for task in TASKS
        },
        expected_tasks=TASKS,
        expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
    )
    _write_json_new(Path(output_path), receipt)
    return receipt


def validate_reference_evidence_approval(
    *,
    approval_receipt_path: str | Path,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    evidence_dirs: Mapping[str, str | Path],
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, tuple[dict[str, object], list[dict[str, object]], dict[str, object]]],
]:
    contract, draft_receipt, _evaluators, verifier_identity, validated = (
        _validated_all_task_evidence(
            draft_root=draft_root,
            solver_contract_path=solver_contract_path,
            evidence_dirs=evidence_dirs,
            verifier_bundle=verifier_bundle,
        )
    )
    receipt = _load_object(
        approval_receipt_path, label="reference evidence approval receipt"
    )
    validate_reference_evidence_approval_receipt(
        receipt,
        solver_contract=contract,
        source_generation_receipt_sha256=str(draft_receipt["receipt_sha256"]),
        verifier_identity_sha256=sha256_json(verifier_identity),
        task_evidence_summary_sha256={
            task: str(validated[task][0]["reference_evidence_summary_sha256"])
            for task in TASKS
        },
        expected_tasks=TASKS,
        expected_instances_per_task=PUBLICATION_INSTANCES_PER_TASK,
    )
    for task in TASKS:
        task_receipt = receipt["tasks"][task]
        if task_receipt.get("evidence_tree") != validated[task][2]:
            raise ValueError(f"reference approval evidence tree changed for {task}")
    return receipt, validated


def _export_record(
    record: dict[str, object], *, task: str, evidence_summary_sha256: str
) -> dict[str, object]:
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reference_evidence_record_sha256": record[
            "reference_evidence_record_sha256"
        ],
        "reference_evidence_summary_sha256": evidence_summary_sha256,
        "runner_protocol": record["runner_protocol"],
        "runner_code_sha256": record["runner_code_sha256"],
        "solver_identity": record["solver_identity"],
        "solver_parameters": record["solver_parameters"],
        "solver_parameters_sha256": record["solver_parameters_sha256"],
        "solver_seed": record["solver_seed"],
        "objective_semantics": record["objective_semantics"],
        "reference_quality": record["reference_quality"],
        "optimality_certificate": record["optimality_certificate"],
        "native_status": record["native_status"],
        "native_status_code": record["native_status_code"],
        "native_incumbent": record["native_incumbent"],
        "native_best_bound": record["native_best_bound"],
        "native_gap": record["native_gap"],
        "wall_time_seconds": record["wall_time_seconds"],
        "native_wall_time_seconds": record["native_wall_time_seconds"],
        "raw_native_log": record["raw_native_log"],
        "native_metadata": record["native_metadata"],
        "strict_verifier_identity": record["strict_verifier_identity"],
        "strict_verifier_identity_sha256": record["strict_verifier_identity_sha256"],
        "strict_verifier_result": record["strict_verifier_result"],
        "strict_verifier_result_sha256": record["strict_verifier_result_sha256"],
        "strict_verified_objective": record["strict_verified_objective"],
    }
    raw_log = copy.deepcopy(record["raw_native_log"])
    raw_log["relative_path"] = f"evidence/{task}/{raw_log['relative_path']}"
    evidence["raw_native_log"] = raw_log
    exported = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "benchmark_row_id": record["benchmark_row_id"],
        "instance_sha256": record["semantic_instance_sha256"],
        "semantic_instance_sha256": record["semantic_instance_sha256"],
        "benchmark_row_sha256": record["benchmark_row_sha256"],
        "output": record["output"],
        "reference_output_sha256": record["reference_output_sha256"],
        "solver_status": record["solver_status"],
        "solver_runtime_seconds": record["wall_time_seconds"],
        "solver_contract_task_sha256": record["solver_contract_task_sha256"],
        "reference_evidence": evidence,
    }
    return _self_hash(exported, "reference_export_record_sha256")


def export_references(
    *,
    draft_root: str | Path,
    solver_contract_path: str | Path,
    evidence_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    approval_receipt_path: str | Path | None = None,
    verifier_bundle: tuple[dict[str, Callable[..., dict]], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Validate completed task evidence and atomically export assembly inputs."""

    if not evidence_dirs:
        raise ValueError("at least one task evidence directory is required")
    unknown = set(evidence_dirs) - set(TASKS)
    if unknown:
        raise ValueError(f"unsupported evidence tasks: {sorted(unknown)}")
    bundle = verifier_bundle or _default_verifier_bundle()
    approval_receipt: dict[str, object] | None = None
    validated: dict[
        str, tuple[dict[str, object], list[dict[str, object]], dict[str, object]]
    ] = {}
    if approval_receipt_path is not None:
        approval_receipt, validated = validate_reference_evidence_approval(
            approval_receipt_path=approval_receipt_path,
            draft_root=draft_root,
            solver_contract_path=solver_contract_path,
            evidence_dirs=evidence_dirs,
            verifier_bundle=bundle,
        )
    else:
        if set(evidence_dirs) == set(TASKS):
            raise ValueError(
                "all-five reference export requires a create-once approval receipt"
            )
        for task, directory in sorted(evidence_dirs.items()):
            summary, records = validate_evidence_directory(
                evidence_dir=directory,
                draft_root=draft_root,
                solver_contract_path=solver_contract_path,
                verifier_bundle=bundle,
            )
            if summary.get("task") != task:
                raise ValueError(
                    f"evidence mapping labels {task} but summary contains {summary.get('task')}"
                )
            validated[task] = (summary, records, _evidence_tree_identity(directory))
    contract = _load_object(solver_contract_path, label="reference solver contract")
    draft_receipt = _load_object(
        Path(draft_root) / "generation_receipt.json", label="generation receipt"
    )
    exported_summary: dict[str, object] = {}

    def write(destination: Path) -> None:
        nonlocal exported_summary
        task_summaries: dict[str, object] = {}
        for task, (summary, records, expected_tree) in sorted(validated.items()):
            summary_sha = str(summary["reference_evidence_summary_sha256"])
            exported = [
                _export_record(
                    record, task=task, evidence_summary_sha256=summary_sha
                )
                for record in records
            ]
            source_evidence_root = Path(evidence_dirs[task]).resolve()
            copied_evidence_root = destination / "evidence" / task
            for entry in expected_tree["files"]:
                assert isinstance(entry, dict)
                source_path = _safe_relative_path(
                    source_evidence_root,
                    entry["relative_path"],
                    label=f"{task} evidence export source",
                )
                target_path = copied_evidence_root / str(entry["relative_path"])
                _write_bytes_new(target_path, source_path.read_bytes())
            copied_tree = _evidence_tree_identity(copied_evidence_root)
            if copied_tree != expected_tree:
                raise ValueError(f"copied reference evidence tree changed for {task}")
            task_path = destination / f"{task}.json"
            _write_json_new(task_path, exported)
            task_summaries[task] = {
                "rows": len(exported),
                "relative_path": f"{task}.json",
                **_file_identity(task_path),
                "reference_evidence_summary_sha256": summary_sha,
                "evidence_relative_root": f"evidence/{task}",
                "evidence_tree": copied_tree,
                "source_details_artifact": {
                    **summary["details_artifact"],
                    "relative_path": f"evidence/{task}/details.json",
                },
                "ordered_reference_evidence_record_sha256": [
                    record["reference_evidence_record_sha256"] for record in records
                ],
                "ordered_reference_export_record_sha256": [
                    record["reference_export_record_sha256"] for record in exported
                ],
            }
            task_summaries[task]["reference_export_records_sha256"] = sha256_json(
                task_summaries[task]["ordered_reference_export_record_sha256"]
            )
        summary_payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": (
                "reference_evidence_export_complete"
                if approval_receipt is not None
                else "partial_reference_evidence_export_nonpublication"
            ),
            "publishable": False,
            "contains_model_inference": False,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner_protocol": RUNNER_PROTOCOL,
            "runner_code_sha256": str(_file_identity(__file__)["sha256"]),
            "source_generation_receipt_sha256": draft_receipt["receipt_sha256"],
            "solver_contract_sha256": contract["contract_sha256"],
            "tasks": task_summaries,
        }
        if approval_receipt is not None:
            approval_path = destination / "reference_evidence_approval.json"
            _write_json_new(approval_path, approval_receipt)
            summary_payload.update(
                {
                    "reference_evidence_approval_receipt_sha256": approval_receipt[
                        "reference_evidence_approval_receipt_sha256"
                    ],
                    "reference_evidence_approval_snapshot": {
                        "relative_path": "reference_evidence_approval.json",
                        **_file_identity(approval_path),
                    },
                }
            )
        exported_summary = _self_hash(
            summary_payload, "reference_export_summary_sha256"
        )
        _write_json_new(destination / "reference_export_summary.json", exported_summary)

    _atomic_directory(output_dir, write)
    return exported_summary


def _parse_evidence_spec(value: str) -> tuple[str, str]:
    task, separator, path = value.partition(",")
    if not separator or task not in TASKS or not path.strip():
        raise argparse.ArgumentTypeError("evidence spec must be TASK,PATH")
    return task, path.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve-task", help="run one approved task backend")
    solve.add_argument("--draft-root", required=True)
    solve.add_argument("--solver-contract", required=True)
    solve.add_argument("--task", choices=TASKS, required=True)
    solve.add_argument("--output-dir", required=True)
    approve = subparsers.add_parser(
        "approve-evidence",
        help="validate exactly five complete task trees and create the assembly approval",
    )
    approve.add_argument("--draft-root", required=True)
    approve.add_argument("--solver-contract", required=True)
    approve.add_argument("--evidence", action="append", type=_parse_evidence_spec, required=True)
    approve.add_argument("--output", required=True)
    export = subparsers.add_parser(
        "export-references", help="verify evidence and export assembly reference JSON"
    )
    export.add_argument("--draft-root", required=True)
    export.add_argument("--solver-contract", required=True)
    export.add_argument("--evidence", action="append", type=_parse_evidence_spec, required=True)
    export.add_argument("--approval-receipt")
    export.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "solve-task":
        summary = solve_task(
            draft_root=args.draft_root,
            solver_contract_path=args.solver_contract,
            task=args.task,
            output_dir=args.output_dir,
        )
        print(f"reference evidence: {Path(args.output_dir).resolve()}")
        print(f"summary_sha256: {summary['reference_evidence_summary_sha256']}")
        return 0
    evidence_dirs: dict[str, str] = {}
    for task, path in args.evidence:
        if task in evidence_dirs:
            raise ValueError(f"duplicate evidence directory for task {task}")
        evidence_dirs[task] = path
    if args.command == "approve-evidence":
        receipt = approve_reference_evidence(
            draft_root=args.draft_root,
            solver_contract_path=args.solver_contract,
            evidence_dirs=evidence_dirs,
            output_path=args.output,
        )
        print(f"reference approval: {Path(args.output).resolve()}")
        print(
            "approval_sha256: "
            f"{receipt['reference_evidence_approval_receipt_sha256']}"
        )
        return 0
    summary = export_references(
        draft_root=args.draft_root,
        solver_contract_path=args.solver_contract,
        evidence_dirs=evidence_dirs,
        output_dir=args.output_dir,
        approval_receipt_path=args.approval_receipt,
    )
    print(f"reference export: {Path(args.output_dir).resolve()}")
    print(f"summary_sha256: {summary['reference_export_summary_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
