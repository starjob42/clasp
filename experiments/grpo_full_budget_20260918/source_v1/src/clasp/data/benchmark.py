"""Canonical identities for CLASP combinatorial-optimization instances.

Benchmark leakage checks must compare the optimization instance, not the row's
prompt wording or reference label.  Released LLMCoSolver SFT files do not
consistently retain ``instance`` while their prompts do retain all numerical
data.  This module deliberately uses the same parsers as the strict evaluator
so structured and prompt-only copies receive the same identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from clasp.rewards.co_tasks import (
    _cvrp_meta,
    _jssp_jobs,
    _mis_meta,
    _op_meta,
    _tsp_coords_from_instance,
)

SUPPORTED_BENCHMARK_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
PUBLICATION_INSTANCES_PER_TASK = 1_000
REFERENCE_EVIDENCE_SCHEMA_VERSION = 1
REFERENCE_EVIDENCE_RUNNER_PROTOCOL = "clasp-iid-reference-evidence-v1"
REFERENCE_EVIDENCE_CONTRACT_STATUS = "approved_for_reference_evidence"
REFERENCE_EVIDENCE_APPROVAL_SCHEMA_VERSION = 1
REFERENCE_EVIDENCE_APPROVAL_STATUS = "approved_for_confirmation"
REFERENCE_OBJECTIVE_SEMANTICS = {
    "tsp": "clasp_raw_euclidean_float64",
    "cvrp": "clasp_raw_euclidean_float64",
    "op": "clasp_raw_euclidean_float64_integer_prizes",
    "mis": "clasp_integer_mis_cardinality",
    "jssp": "clasp_integer_jssp_makespan",
}
REFERENCE_QUALITY = {
    "tsp": "best_known_heuristic",
    "cvrp": "best_known_heuristic",
    "op": "certified_optimum",
    "mis": "certified_optimum",
    "jssp": "certified_optimum",
}
REFERENCE_PUBLIC_STATUS = {
    "tsp": "BEST_KNOWN_HEURISTIC_VERIFIED",
    "cvrp": "BEST_KNOWN_HEURISTIC_VERIFIED",
    "op": "OPTIMAL_CERTIFIED",
    "mis": "OPTIMAL_CERTIFIED",
    "jssp": "OPTIMAL_CERTIFIED",
}
REFERENCE_NATIVE_STATUS = {
    "tsp": "HEURISTIC_COMPLETE",
    "cvrp": "HEURISTIC_COMPLETE",
    "op": "OPTIMAL",
    "mis": "OPTIMAL",
    "jssp": "OPTIMAL",
}
REFERENCE_ADAPTER = {
    "tsp": "raw_euclidean_tsp_heuristic_v1",
    "cvrp": "raw_euclidean_cvrp_heuristic_v1",
    "op": "gurobi_op_exact_v2",
    "mis": "gurobi_mis_v1",
    "jssp": "ortools_jssp_v1",
}
AUDITED_GUROBI_VERSION = "13.0.1"
AUDITED_GUROBIPY_WHEEL_SHA256 = (
    "7fc1b5ff46530aee391ab0f616620a1399abea0778b25e374b37f7ed222d5dfb"
)
AUDITED_ORTOOLS_VERSION = "9.15.6755"
AUDITED_ORTOOLS_WHEEL_SHA256 = (
    "03424136aa48555e7f4d1bc73edeb99f80ec35a2e5f17700e2072640344980fc"
)
REQUIRED_FORBIDDEN_SOURCE_LABELS = frozenset({"sft-train", "rl-train", "historical-eval"})
FORBIDDEN_SOURCE_REGISTRY_SCHEMA_VERSION = 1
FORBIDDEN_SOURCE_REGISTRY_PROTOCOL = "clasp-forbidden-source-registry-v1"
FORBIDDEN_SOURCE_REGISTRY_STATUS = "frozen_before_confirmation_generation"
PUBLICATION_BENCHMARK_MODE = "publication"
NONPUBLICATION_BENCHMARK_MODE = "nonpublication"


def canonical_json_bytes(value: object) -> bytes:
    """Return the single JSON serialization used by benchmark fingerprints."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether *value* is a lowercase or uppercase hexadecimal SHA-256."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_contract_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _contract_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer without coercion")
    return value


def _require_exact_keys(
    value: dict[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}"
        )


def validate_reference_solver_parameters(
    task: str,
    parameters: object,
    *,
    seed_parameter: object,
) -> int | None:
    """Validate the exact typed parameter surface consumed by each adapter.

    No value is silently coerced and no contract field may be ignored by the
    implementation.  The deterministic TSP/CVRP adapters intentionally expose
    no synthetic seed because they use no random-number generator.
    """

    if task not in SUPPORTED_BENCHMARK_TASKS or not isinstance(parameters, dict):
        raise ValueError(f"{task} solver parameters must be an object")
    if task == "tsp":
        _require_exact_keys(
            parameters,
            {"starts", "max_2opt_passes"},
            label="tsp solver parameters",
        )
        if parameters["starts"] != "all":
            raise ValueError("tsp starts must be exactly 'all'")
        if _contract_int(
            parameters["max_2opt_passes"], label="tsp max_2opt_passes"
        ) <= 0:
            raise ValueError("tsp max_2opt_passes must be positive")
        if seed_parameter is not None:
            raise ValueError("tsp deterministic adapter must not declare a seed parameter")
        return None
    if task == "cvrp":
        _require_exact_keys(
            parameters,
            {"max_2opt_passes"},
            label="cvrp solver parameters",
        )
        if _contract_int(
            parameters["max_2opt_passes"], label="cvrp max_2opt_passes"
        ) <= 0:
            raise ValueError("cvrp max_2opt_passes must be positive")
        if seed_parameter is not None:
            raise ValueError("cvrp deterministic adapter must not declare a seed parameter")
        return None

    if task in {"op", "mis"}:
        fixed: dict[str, int | float] = {
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
        allowed = {"time_limit_seconds", *fixed}
        if task == "op":
            allowed.update({"work_limit", "node_limit"})
        _require_exact_keys(parameters, allowed, label=f"{task} solver parameters")
        integer_fields = {
            "threads",
            "seed",
            "method",
            "numeric_focus",
            "output_flag",
            "log_to_console",
        }
        for field, expected in fixed.items():
            observed = parameters[field]
            if field in integer_fields:
                observed = _contract_int(observed, label=f"{task} {field}")
            else:
                observed = _finite_contract_number(observed, label=f"{task} {field}")
            if observed != expected:
                raise ValueError(f"{task} {field} must be fixed to {expected!r}")
        if _finite_contract_number(
            parameters["time_limit_seconds"], label=f"{task} time_limit_seconds"
        ) <= 0:
            raise ValueError(f"{task} time_limit_seconds must be positive")
        if task == "op":
            if _finite_contract_number(
                parameters["work_limit"], label="op work_limit"
            ) <= 0:
                raise ValueError("op work_limit must be positive")
            if _contract_int(parameters["node_limit"], label="op node_limit") <= 0:
                raise ValueError("op node_limit must be positive")
        if seed_parameter != "seed":
            raise ValueError(f"{task} seed_parameter must be exactly 'seed'")
        return int(parameters["seed"])

    _require_exact_keys(
        parameters,
        {
            "max_time_seconds",
            "num_search_workers",
            "random_seed",
            "absolute_gap_limit",
            "relative_gap_limit",
            "log_search_progress",
        },
        label="jssp solver parameters",
    )
    if _finite_contract_number(
        parameters["max_time_seconds"], label="jssp max_time_seconds"
    ) <= 0:
        raise ValueError("jssp max_time_seconds must be positive")
    if _contract_int(
        parameters["num_search_workers"], label="jssp num_search_workers"
    ) != 1:
        raise ValueError("jssp num_search_workers must be fixed to 1")
    seed = _contract_int(parameters["random_seed"], label="jssp random_seed")
    for field in ("absolute_gap_limit", "relative_gap_limit"):
        if _finite_contract_number(parameters[field], label=f"jssp {field}") != 0.0:
            raise ValueError(f"jssp {field} must be fixed to 0.0")
    if not isinstance(parameters["log_search_progress"], bool):
        raise ValueError("jssp log_search_progress must be bool without coercion")
    if seed_parameter != "random_seed":
        raise ValueError("jssp seed_parameter must be exactly 'random_seed'")
    return seed


def validate_reference_solver_task_policy(
    task: str,
    task_contract: object,
    *,
    runner_code_sha256: str,
    require_production_identity: bool,
) -> int | None:
    """Validate one immutable task policy independently of solver execution."""

    if task not in SUPPORTED_BENCHMARK_TASKS or not isinstance(task_contract, dict):
        raise ValueError(f"reference solver contract has no valid {task} task policy")
    if task_contract.get("accepted_statuses") != [REFERENCE_PUBLIC_STATUS[task]]:
        raise ValueError(f"reference solver contract has invalid accepted status for {task}")
    if not str(task_contract.get("backend", "")).strip():
        raise ValueError(f"reference solver contract has no backend for {task}")
    if not str(task_contract.get("version", "")).strip():
        raise ValueError(f"reference solver contract has no version for {task}")
    if not str(task_contract.get("license", "")).strip():
        raise ValueError(f"reference solver contract has no license classification for {task}")
    if not str(task_contract.get("quality_semantics", "")).strip():
        raise ValueError(f"reference solver contract has no quality semantics for {task}")
    runner = task_contract.get("evidence_runner")
    if not isinstance(runner, dict):
        raise ValueError(f"reference solver contract has no evidence runner for {task}")
    if (
        runner.get("approval") != "approved"
        or runner.get("adapter") != REFERENCE_ADAPTER[task]
        or runner.get("accepted_native_statuses") != [REFERENCE_NATIVE_STATUS[task]]
        or runner.get("objective_semantics") != REFERENCE_OBJECTIVE_SEMANTICS[task]
        or runner.get("reference_quality") != REFERENCE_QUALITY[task]
    ):
        raise ValueError(f"reference solver contract machine policy is invalid for {task}")
    seed = validate_reference_solver_parameters(
        task,
        task_contract.get("parameters"),
        seed_parameter=runner.get("seed_parameter"),
    )
    identity = runner.get("solver_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"reference solver contract has no solver identity for {task}")
    for field in (
        "backend_name",
        "native_version",
        "artifact_kind",
        "artifact_path",
        "artifact_sha256",
    ):
        value = identity.get(field)
        if value in (None, "") or "UNRESOLVED" in str(value).upper():
            raise ValueError(f"reference solver identity {task}.{field} is unresolved")
    if not is_sha256(identity.get("artifact_sha256")):
        raise ValueError(f"reference solver identity artifact hash is invalid for {task}")
    if identity.get("artifact_kind") == "python_wheel":
        for field in ("distribution_name", "module_origin"):
            value = identity.get(field)
            if value in (None, "") or "UNRESOLVED" in str(value).upper():
                raise ValueError(f"reference solver identity {task}.{field} is unresolved")
        if not is_sha256(identity.get("installed_distribution_manifest_sha256")):
            raise ValueError(
                f"reference solver identity {task} lacks an installed-distribution manifest"
            )
    if not require_production_identity:
        return seed

    if task in {"tsp", "cvrp"}:
        expected = {
            "backend_name": f"clasp_raw_{task}_heuristic",
            "native_version": "1",
            "artifact_kind": "python_source",
            "artifact_sha256": runner_code_sha256,
        }
    elif task in {"op", "mis"}:
        expected = {
            "backend_name": "gurobi",
            "native_version": AUDITED_GUROBI_VERSION,
            "artifact_kind": "python_wheel",
            "artifact_sha256": AUDITED_GUROBIPY_WHEEL_SHA256,
            "distribution_name": "gurobipy",
        }
    else:
        expected = {
            "backend_name": "ortools_cp_sat",
            "native_version": AUDITED_ORTOOLS_VERSION,
            "artifact_kind": "python_wheel",
            "artifact_sha256": AUDITED_ORTOOLS_WHEEL_SHA256,
            "distribution_name": "ortools",
        }
    for field, expected_value in expected.items():
        if identity.get(field) != expected_value:
            raise ValueError(
                f"reference solver identity {task}.{field} must equal {expected_value!r}"
            )
    if task in {"op", "mis", "jssp"}:
        module_origin = identity.get("module_origin")
        if (
            not isinstance(module_origin, str)
            or not module_origin.startswith("/")
            or "UNRESOLVED" in module_origin.upper()
        ):
            raise ValueError(
                f"reference solver identity {task}.module_origin must be resolved and absolute"
            )
        if not is_sha256(identity.get("installed_distribution_manifest_sha256")):
            raise ValueError(
                f"reference solver identity {task} lacks an installed-distribution manifest"
            )
    return seed


def validate_reference_solver_contract_policy(
    contract: object,
    *,
    verifier_identity_sha256: str,
    expected_tasks: Sequence[str] = SUPPORTED_BENCHMARK_TASKS,
    expected_instances_per_task: int = PUBLICATION_INSTANCES_PER_TASK,
    require_production_identity: bool = True,
) -> dict[str, int | None]:
    """Validate a self-hashed immutable contract approved only to create evidence."""

    if not isinstance(contract, dict) or contract.get("schema_version") != 2:
        raise ValueError("reference solver contract must use schema_version 2")
    stored_hash = contract.get("contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if not is_sha256(stored_hash) or stored_hash != sha256_json(unsigned):
        raise ValueError("reference solver contract self-hash is invalid")
    if contract.get("status") != REFERENCE_EVIDENCE_CONTRACT_STATUS:
        raise ValueError(
            "reference solver contract must remain immutable with status "
            f"{REFERENCE_EVIDENCE_CONTRACT_STATUS!r}"
        )
    protocol = contract.get("reference_evidence_runner")
    if not isinstance(protocol, dict):
        raise ValueError("reference solver contract has no runner policy")
    if (
        protocol.get("schema_version") != REFERENCE_EVIDENCE_SCHEMA_VERSION
        or protocol.get("protocol") != REFERENCE_EVIDENCE_RUNNER_PROTOCOL
        or protocol.get("strict_verifier_identity_sha256")
        != verifier_identity_sha256
        or protocol.get("expected_instances_per_task") != expected_instances_per_task
    ):
        raise ValueError("reference solver contract runner policy is invalid")
    runner_code_sha = protocol.get("runner_code_sha256")
    if not is_sha256(runner_code_sha):
        raise ValueError("reference solver contract does not pin the evidence runner source")
    tasks = contract.get("tasks")
    required_tasks = set(expected_tasks)
    if not isinstance(tasks, dict) or set(tasks) != required_tasks:
        raise ValueError(
            "reference solver contract must define exactly the complete publication task set"
        )
    seeds: dict[str, int | None] = {}
    for task in expected_tasks:
        seeds[task] = validate_reference_solver_task_policy(
            task,
            tasks[task],
            runner_code_sha256=str(runner_code_sha),
            require_production_identity=require_production_identity,
        )
    return seeds


def _validate_evidence_tree_manifest(
    tree: object,
    *,
    task: str,
    rows: int,
) -> None:
    if not isinstance(tree, dict) or not isinstance(tree.get("files"), list):
        raise ValueError(f"reference approval has no evidence tree for {task}")
    files = tree["files"]
    expected_paths = {
        "details.json",
        "solver_contract.json",
        "source_generation_receipt.json",
        "summary.json",
        *(f"native_logs/{task}-{index:06d}.log" for index in range(rows)),
    }
    observed_paths: list[str] = []
    total_size = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"reference approval evidence tree entry is invalid for {task}")
        relative = entry.get("relative_path")
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or posixpath.normpath(relative) != relative
            or relative.startswith("../")
        ):
            raise ValueError(f"reference approval evidence path is unsafe for {task}")
        if not is_sha256(entry.get("sha256")):
            raise ValueError(f"reference approval evidence hash is invalid for {task}")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"reference approval evidence file is empty for {task}")
        observed_paths.append(relative)
        total_size += size
    if observed_paths != sorted(observed_paths) or set(observed_paths) != expected_paths:
        raise ValueError(f"reference approval evidence tree is incomplete for {task}")
    if (
        tree.get("file_count") != len(files)
        or tree.get("total_size_bytes") != total_size
        or tree.get("manifest_sha256") != sha256_json(files)
    ):
        raise ValueError(f"reference approval evidence tree identity is invalid for {task}")


def validate_reference_evidence_approval_receipt(
    receipt: object,
    *,
    solver_contract: dict[str, object],
    source_generation_receipt_sha256: str,
    verifier_identity_sha256: str,
    task_evidence_summary_sha256: dict[str, str],
    expected_tasks: Sequence[str] = SUPPORTED_BENCHMARK_TASKS,
    expected_instances_per_task: int = PUBLICATION_INSTANCES_PER_TASK,
) -> dict[str, object]:
    """Validate the create-once all-row receipt that authorizes assembly."""

    if not isinstance(receipt, dict):
        raise ValueError("reference evidence approval receipt must be an object")
    stored_hash = receipt.get("reference_evidence_approval_receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("reference_evidence_approval_receipt_sha256", None)
    if not is_sha256(stored_hash) or stored_hash != sha256_json(unsigned):
        raise ValueError("reference evidence approval receipt self-hash is invalid")
    if (
        receipt.get("schema_version") != REFERENCE_EVIDENCE_APPROVAL_SCHEMA_VERSION
        or receipt.get("status") != REFERENCE_EVIDENCE_APPROVAL_STATUS
        or receipt.get("runner_protocol") != REFERENCE_EVIDENCE_RUNNER_PROTOCOL
        or receipt.get("publishable") is not False
        or receipt.get("contains_model_inference") is not False
    ):
        raise ValueError("reference evidence approval receipt has invalid lifecycle metadata")
    contract_sha = solver_contract.get("contract_sha256")
    protocol = solver_contract.get("reference_evidence_runner")
    runner_code_sha = protocol.get("runner_code_sha256") if isinstance(protocol, dict) else None
    if (
        receipt.get("solver_contract_sha256") != contract_sha
        or receipt.get("solver_contract_status") != REFERENCE_EVIDENCE_CONTRACT_STATUS
        or receipt.get("source_generation_receipt_sha256")
        != source_generation_receipt_sha256
        or receipt.get("strict_verifier_identity_sha256")
        != verifier_identity_sha256
        or receipt.get("runner_code_sha256") != runner_code_sha
    ):
        raise ValueError("reference evidence approval receipt targets stale inputs")
    tasks = receipt.get("tasks")
    required_tasks = set(expected_tasks)
    if (
        not isinstance(tasks, dict)
        or set(tasks) != required_tasks
        or set(task_evidence_summary_sha256) != required_tasks
    ):
        raise ValueError("reference evidence approval must cover exactly all five tasks")
    contract_tasks = solver_contract.get("tasks")
    if not isinstance(contract_tasks, dict):
        raise ValueError("reference evidence approval has no contract task policies")
    for task in expected_tasks:
        task_receipt = tasks.get(task)
        if not isinstance(task_receipt, dict):
            raise ValueError(f"reference evidence approval lacks task {task}")
        expected = {
            "rows": expected_instances_per_task,
            "solver_contract_task_sha256": sha256_json(contract_tasks[task]),
            "reference_evidence_summary_sha256": task_evidence_summary_sha256[task],
            "reference_quality": REFERENCE_QUALITY[task],
            "required_public_status": REFERENCE_PUBLIC_STATUS[task],
            "required_native_status": REFERENCE_NATIVE_STATUS[task],
            "native_status_counts": {
                REFERENCE_NATIVE_STATUS[task]: expected_instances_per_task
            },
        }
        for field, expected_value in expected.items():
            if task_receipt.get(field) != expected_value:
                raise ValueError(
                    f"reference evidence approval field {task}.{field} is invalid"
                )
        _validate_evidence_tree_manifest(
            task_receipt.get("evidence_tree"),
            task=task,
            rows=expected_instances_per_task,
        )
    return dict(receipt)


def benchmark_row_sha256(row: dict[str, object]) -> str:
    """Hash the source fields that survive CLASP's dataset normalization.

    Benchmark JSON rows can carry generator and solver metadata which the
    evaluation loader intentionally drops.  Hashing this normalized projection
    lets the freezer and evaluator independently derive the same exact row
    identity while still binding the prompt, reference answer, and raw
    structured-instance representation.
    """

    return sha256_json(
        {
            "instruction": str(row.get("instruction", "")),
            "input": str(row.get("input", "")),
            "output": str(row.get("output", "")),
            "instance_json": structured_instance(row),
        }
    )


def reference_output_sha256(output: object) -> str:
    """Return the exact, canonical identity of a benchmark reference answer."""

    return sha256_json(str(output))


def parse_instance_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def structured_instance(row: dict[str, object]) -> Any:
    for key in ("instance_json", "instance"):
        if key in row:
            value = parse_instance_value(row.get(key))
            if value is not None:
                return value
    return None


def _number(value: float) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"instance contains non-finite number: {value!r}")
    rounded = round(number)
    if abs(number - rounded) <= 1e-12:
        return int(rounded)
    return number


def _coordinates(values: list[list[float]]) -> list[list[int | float]]:
    return [[_number(row[0]), _number(row[1])] for row in values]


def canonical_instance(
    task: str,
    row: dict[str, object],
    *,
    require_structured: bool = False,
) -> dict[str, object]:
    """Extract the task semantics used for duplicate/leakage detection.

    ``require_structured`` is used for a newly generated confirmation set.  It
    remains false for historical forbidden sources because several released
    SFT files stripped their raw instance after rendering the prompt.
    """

    if task not in SUPPORTED_BENCHMARK_TASKS:
        raise ValueError(f"unsupported benchmark task: {task}")
    instance = structured_instance(row)
    if require_structured and instance is None:
        raise ValueError("row has no non-null structured instance_json/instance")
    if require_structured:
        shape_ok = {
            "tsp": isinstance(instance, list),
            "cvrp": isinstance(instance, list) and len(instance) >= 3,
            "op": isinstance(instance, list) and len(instance) >= 3,
            "mis": isinstance(instance, dict) and "edges" in instance,
            "jssp": isinstance(instance, list),
        }[task]
        if not shape_ok:
            raise ValueError(f"structured {task} instance has the wrong shape")

    instruction = str(row.get("instruction", ""))
    input_text = str(row.get("input", ""))
    context = f"{instruction}\n{input_text}"

    if task == "tsp":
        coords = _tsp_coords_from_instance(instance, input_text)
        if not coords:
            raise ValueError("could not parse TSP coordinates")
        return {"task": task, "coordinates": _coordinates(coords)}

    if task == "cvrp":
        coords, demands, capacity = _cvrp_meta(instance, context)
        if not coords or not demands or capacity is None or len(coords) != len(demands):
            raise ValueError("could not parse complete CVRP instance")
        return {
            "task": task,
            "coordinates": _coordinates(coords),
            "demands": [_number(value) for value in demands],
            "vehicle_capacity": _number(capacity),
        }

    if task == "op":
        coords, prizes, route_limit = _op_meta(instance, context)
        if not coords or not prizes or route_limit is None or len(coords) != len(prizes):
            raise ValueError("could not parse complete OP instance")
        # The released prompt exposes T to one decimal.  Hash that observable
        # precision so a prompt-only historical copy collides with its retained
        # structured instance instead of evading the leakage audit.
        return {
            "task": task,
            "coordinates": _coordinates(coords),
            "prizes": [_number(value) for value in prizes],
            "max_route_length_prompt_precision": round(float(route_limit), 1),
        }

    if task == "mis":
        num_nodes, edges = _mis_meta(instance, context)
        if num_nodes is None or num_nodes <= 0:
            raise ValueError("could not parse complete MIS instance")
        return {
            "task": task,
            "num_nodes": int(num_nodes),
            "edges": [[int(a), int(b)] for a, b in sorted(edges)],
        }

    jobs = _jssp_jobs(instance, context)
    if not jobs:
        raise ValueError("could not parse complete JSSP instance")
    machine_count = len(jobs[0])
    if machine_count <= 0 or any(len(job) != machine_count for job in jobs):
        raise ValueError("JSSP jobs do not have a common positive machine count")
    for job in jobs:
        machines = [machine for machine, _ in job]
        if sorted(machines) != list(range(machine_count)):
            raise ValueError("each JSSP job must use every machine exactly once")
        if any(duration <= 0 for _, duration in job):
            raise ValueError("JSSP processing times must be positive")
    return {
        "task": task,
        "jobs": [[[int(machine), int(duration)] for machine, duration in job] for job in jobs],
    }


def semantic_instance_sha256(
    task: str,
    row: dict[str, object],
    *,
    require_structured: bool = False,
) -> str:
    return sha256_json(canonical_instance(task, row, require_structured=require_structured))


def instance_sha256(
    task: str,
    row: dict[str, object],
    *,
    require_structured: bool = False,
) -> str:
    """Backward-compatible alias for :func:`semantic_instance_sha256`."""

    return semantic_instance_sha256(task, row, require_structured=require_structured)


def is_sha256(value: object) -> bool:
    """Return whether *value* is a lowercase/uppercase 64-digit SHA-256 hex string."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"frozen benchmark has invalid {label}")
    return str(value)


def _normalized_relative_path(value: object, *, label: str) -> str:
    """Validate the portable path stored in a frozen data-tree manifest."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} has no relative path")
    path = value.replace("\\", "/")
    normalized = posixpath.normpath(path)
    if (
        path != normalized
        or normalized in {"", ".", ".."}
        or normalized.startswith("../")
        or normalized.startswith("/")
    ):
        raise ValueError(f"{label} has a non-canonical relative path: {value!r}")
    return normalized


def _forbidden_source_projection(source: dict[str, object]) -> dict[str, object]:
    """Return the path-independent content identity of one forbidden data tree."""

    base: dict[str, object] = {
        "task": source.get("task"),
        "label": source.get("label"),
        "files": source.get("files"),
        "file_manifest_sha256": source.get("file_manifest_sha256"),
        "rows": source.get("rows"),
        "unique_instance_hashes": source.get("unique_instance_hashes"),
    }
    return {**base, "source_identity_sha256": sha256_json(base)}


def forbidden_source_manifest_identity(
    source: dict[str, object],
) -> dict[str, object]:
    """Return a portable identity for one audited forbidden-source tree."""

    return _forbidden_source_projection(source)


def _validate_forbidden_source(
    source: object,
    *,
    label: str,
    require_stored_identity: bool,
) -> tuple[tuple[str, str], dict[str, object]]:
    if not isinstance(source, dict):
        raise ValueError(f"{label} is not an object")
    task = source.get("task")
    source_label = source.get("label")
    if task not in SUPPORTED_BENCHMARK_TASKS:
        raise ValueError(f"{label} has invalid task")
    if not isinstance(source_label, str) or not source_label.strip():
        raise ValueError(f"{label} has an empty label")
    if source_label != source_label.strip():
        raise ValueError(f"{label} has a non-canonical label")

    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} has no files")
    relative_paths: list[str] = []
    source_rows = 0
    for file_index, file_record in enumerate(files):
        file_label = f"{label}.files[{file_index}]"
        if not isinstance(file_record, dict):
            raise ValueError(f"{file_label} is invalid")
        relative_path = _normalized_relative_path(
            file_record.get("relative_path"), label=file_label
        )
        relative_paths.append(relative_path)
        _require_sha256(file_record.get("sha256"), label=f"{file_label}.sha256")
        size_value = file_record.get("size_bytes")
        count_value = file_record.get("rows")
        size = (
            size_value if isinstance(size_value, int) and not isinstance(size_value, bool) else -1
        )
        count = (
            count_value
            if isinstance(count_value, int) and not isinstance(count_value, bool)
            else -1
        )
        if size <= 0 or count <= 0:
            raise ValueError(f"{file_label} has invalid size/rows")
        source_rows += count
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"{label} file paths must be sorted and unique")

    declared_value = source.get("rows")
    unique_value = source.get("unique_instance_hashes")
    declared_rows = (
        declared_value
        if isinstance(declared_value, int) and not isinstance(declared_value, bool)
        else -1
    )
    unique_rows = (
        unique_value if isinstance(unique_value, int) and not isinstance(unique_value, bool) else -1
    )
    if source_rows != declared_rows or declared_rows <= 0 or not 0 < unique_rows <= declared_rows:
        raise ValueError(f"{label} row counts are invalid")
    if source.get("file_manifest_sha256") != sha256_json(files):
        raise ValueError(f"{label} file-manifest hash is invalid")

    projection = _forbidden_source_projection(source)
    if (
        require_stored_identity
        and source.get("source_identity_sha256") != projection["source_identity_sha256"]
    ):
        raise ValueError(f"{label} source-identity hash is invalid")
    return (str(task), source_label), projection


def forbidden_source_registry_identity(
    registry: dict[str, object],
) -> dict[str, object]:
    """Return the compact identity embedded in evaluator provenance."""

    sources_by_task_label: dict[str, dict[str, dict[str, object]]] = {}
    sources = registry.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            task = source.get("task")
            label = source.get("label")
            if not isinstance(task, str) or not isinstance(label, str):
                continue
            sources_by_task_label.setdefault(task, {})[label] = {
                "source_identity_sha256": source.get("source_identity_sha256"),
                "file_manifest_sha256": source.get("file_manifest_sha256"),
                "rows": source.get("rows"),
                "unique_instance_hashes": source.get("unique_instance_hashes"),
                "files": source.get("files"),
            }
    return {
        "schema_version": registry.get("schema_version"),
        "protocol": registry.get("protocol"),
        "status": registry.get("status"),
        "registry_sha256": registry.get("registry_sha256"),
        "source_tree_manifest_sha256": registry.get("source_tree_manifest_sha256"),
        "tasks": registry.get("tasks"),
        "required_labels": registry.get("required_labels"),
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "sources_by_task_label": sources_by_task_label,
    }


def validate_forbidden_source_registry(
    registry: dict[str, object],
    *,
    expected_tasks: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate a frozen, exact registry of every forbidden source data tree.

    The registry is intended to be created and its hash published before IID
    confirmation generation. It enumerates the complete binary file manifest
    for every required task/label pair; a later benchmark freeze must match the
    registry byte-for-byte at the file-manifest level.
    """

    if not isinstance(registry, dict):
        raise ValueError("forbidden-source registry must be a JSON object")
    stored_hash = _require_sha256(
        registry.get("registry_sha256"), label="forbidden-source registry self-hash"
    )
    unsigned = dict(registry)
    unsigned.pop("registry_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("forbidden-source registry self-hash mismatch")
    if registry.get("schema_version") != FORBIDDEN_SOURCE_REGISTRY_SCHEMA_VERSION:
        raise ValueError("forbidden-source registry must use schema_version exactly 1")
    if registry.get("protocol") != FORBIDDEN_SOURCE_REGISTRY_PROTOCOL:
        raise ValueError("forbidden-source registry protocol is invalid")
    if registry.get("status") != FORBIDDEN_SOURCE_REGISTRY_STATUS:
        raise ValueError("forbidden-source registry status is not pre-generation frozen")
    created_at = registry.get("created_at_utc")
    if not isinstance(created_at, str):
        raise ValueError("forbidden-source registry has no creation timestamp")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(
            "forbidden-source registry creation timestamp is not canonical UTC"
        ) from exc

    tasks = registry.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(task not in SUPPORTED_BENCHMARK_TASKS for task in tasks)
        or tasks != sorted(tasks)
        or len(tasks) != len(set(tasks))
    ):
        raise ValueError("forbidden-source registry tasks must be a sorted unique supported list")
    if expected_tasks is not None and list(sorted(expected_tasks)) != tasks:
        raise ValueError(
            "forbidden-source registry task set mismatch: "
            f"expected={sorted(expected_tasks)}, observed={tasks}"
        )
    required_labels = registry.get("required_labels")
    expected_labels = sorted(REQUIRED_FORBIDDEN_SOURCE_LABELS)
    if required_labels != expected_labels:
        raise ValueError("forbidden-source registry required labels must equal the publication set")

    sources = registry.get("sources")
    expected_keys = {(str(task), label) for task in tasks for label in expected_labels}
    if not isinstance(sources, list):
        raise ValueError("forbidden-source registry sources must be a list")
    observed_keys: list[tuple[str, str]] = []
    projections: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        key, projection = _validate_forbidden_source(
            source,
            label=f"forbidden-source registry source {index}",
            require_stored_identity=True,
        )
        observed_keys.append(key)
        projections.append(projection)
    if observed_keys != sorted(observed_keys) or len(observed_keys) != len(set(observed_keys)):
        raise ValueError("forbidden-source registry source entries must be sorted and unique")
    if set(observed_keys) != expected_keys:
        missing = sorted(expected_keys - set(observed_keys))
        extra = sorted(set(observed_keys) - expected_keys)
        raise ValueError(
            "forbidden-source registry does not exactly cover every task/label pair: "
            f"missing={missing}, extra={extra}"
        )
    tree_hash = sha256_json(projections)
    if registry.get("source_tree_manifest_sha256") != tree_hash:
        raise ValueError("forbidden-source registry data-tree hash is invalid")
    return forbidden_source_registry_identity(registry)


def validate_forbidden_sources_against_registry(
    sources: object,
    registry: dict[str, object],
    *,
    expected_tasks: Sequence[str],
) -> dict[str, object]:
    """Require actual leakage inputs to equal the pre-frozen registry exactly."""

    identity = validate_forbidden_source_registry(registry, expected_tasks=expected_tasks)
    if not isinstance(sources, list):
        raise ValueError("frozen benchmark forbidden_sources must be a list")
    keys: list[tuple[str, str]] = []
    projections: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        key, projection = _validate_forbidden_source(
            source,
            label=f"frozen benchmark forbidden source {index}",
            require_stored_identity=True,
        )
        keys.append(key)
        projections.append(projection)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("frozen benchmark forbidden sources must be sorted and unique")
    registered = registry.get("sources")
    assert isinstance(registered, list)  # established by registry validation
    registered_projections = [
        _forbidden_source_projection(source) for source in registered if isinstance(source, dict)
    ]
    if projections != registered_projections:
        raise ValueError("production forbidden sources do not exactly match the frozen registry")
    return identity


def _validate_generation_receipt_closure(
    receipt: object,
    *,
    tasks: Sequence[str],
    test_sets: dict[str, object],
    generator_sha: str,
    solver_sha: str,
    verifier: dict[str, object],
    registry_identity: dict[str, object] | None,
    require_reference_evidence: bool,
) -> None:
    """Validate the complete assembled receipt embedded in a frozen manifest."""

    if not isinstance(receipt, dict):
        raise ValueError("frozen benchmark is not bound to a generation receipt")
    stored_hash = _require_sha256(
        receipt.get("receipt_sha256"), label="generation receipt self-hash"
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if stored_hash != sha256_json(unsigned):
        raise ValueError("frozen benchmark generation receipt self-hash is invalid")
    if receipt.get("schema_version") != 2:
        raise ValueError("frozen benchmark generation receipt must use schema_version 2")
    if (
        receipt.get("status") != "references_attached"
        or receipt.get("contains_references") is not True
        or receipt.get("publishable") is not True
    ):
        raise ValueError("frozen benchmark generation receipt is not publishable")
    created_at = receipt.get("created_at_utc")
    if not isinstance(created_at, str):
        raise ValueError("frozen benchmark generation receipt has no timestamp")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(
            "frozen benchmark generation receipt timestamp is not canonical UTC"
        ) from exc
    if receipt.get("generator_contract_sha256") != generator_sha:
        raise ValueError(
            "frozen benchmark generation receipt is bound to another generator contract"
        )
    _require_sha256(receipt.get("generator_code_sha256"), label="generation code hash")
    _require_sha256(
        receipt.get("source_generation_receipt_sha256"),
        label="source generation receipt hash",
    )
    if receipt.get("reference_solver_contract_sha256") != solver_sha:
        raise ValueError("frozen benchmark receipt is bound to another solver contract")
    if receipt.get("strict_verifier") != verifier:
        raise ValueError("frozen benchmark receipt is bound to another strict verifier")
    if receipt.get("forbidden_source_registry_identity") != registry_identity:
        raise ValueError(
            "frozen benchmark generation receipt is bound to another forbidden-source registry"
        )

    receipt_tasks = receipt.get("tasks")
    if not isinstance(receipt_tasks, dict) or set(receipt_tasks) != set(tasks):
        raise ValueError("frozen benchmark generation receipt task set does not match test sets")
    sequence_fields = (
        (
            "ordered_instance_sha256",
            "ordered_semantic_instance_sha256",
            "ordered_instance_hashes_sha256",
        ),
        (
            "ordered_semantic_instance_sha256",
            "ordered_semantic_instance_sha256",
            "semantic_instance_hashes_sha256",
        ),
        (
            "ordered_benchmark_row_sha256",
            "ordered_benchmark_row_sha256",
            "benchmark_row_hashes_sha256",
        ),
        (
            "ordered_reference_output_sha256",
            "ordered_reference_output_sha256",
            "reference_output_hashes_sha256",
        ),
    )
    for task in tasks:
        receipt_task = receipt_tasks.get(task)
        test_task = test_sets.get(task)
        if not isinstance(receipt_task, dict) or not isinstance(test_task, dict):
            raise ValueError(f"generation receipt has invalid task record for {task}")
        rows = test_task.get("rows")
        if receipt_task.get("rows") != rows:
            raise ValueError(f"generation receipt row count does not match {task}")
        files = test_task.get("files")
        if (
            not isinstance(files, list)
            or len(files) != 1
            or not isinstance(files[0], dict)
            or receipt_task.get("file_sha256") != files[0].get("sha256")
        ):
            raise ValueError(f"generation receipt file identity does not match frozen {task} data")
        if not str(receipt_task.get("relative_path", "")).strip():
            raise ValueError(f"generation receipt has no relative path for {task}")
        for receipt_field, manifest_field, aggregate_field in sequence_fields:
            expected = test_task.get(manifest_field)
            if receipt_task.get(receipt_field) != expected:
                raise ValueError(f"generation receipt {receipt_field} does not match {task} rows")
            if receipt_task.get(aggregate_field) != sha256_json(expected):
                raise ValueError(f"generation receipt {aggregate_field} is invalid for {task}")
        audit = test_task.get("reference_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"frozen benchmark task {task} has no reference audit")
        for receipt_field, audit_field in (
            ("strict_reference_objective_min", "objective_min"),
            ("strict_reference_objective_max", "objective_max"),
        ):
            try:
                observed = float(receipt_task.get(receipt_field))
                expected_objective = float(audit.get(audit_field))
            except (TypeError, ValueError):
                observed, expected_objective = math.nan, math.nan
            if not (
                math.isfinite(observed)
                and math.isfinite(expected_objective)
                and math.isclose(
                    observed,
                    expected_objective,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"generation receipt objective range does not match {task}")
        if require_reference_evidence:
            for field in (
                "reference_evidence_summary_sha256",
                "ordered_reference_evidence_record_sha256",
                "reference_evidence_records_sha256",
                "ordered_reference_export_record_sha256",
                "reference_export_records_sha256",
            ):
                if receipt_task.get(field) != test_task.get(field):
                    raise ValueError(
                        f"generation receipt reference evidence {field} does not match {task} rows"
                    )
    if require_reference_evidence:
        if receipt.get("reference_evidence_schema_version") != 1:
            raise ValueError("frozen benchmark receipt lacks reference-evidence schema")
        _require_sha256(
            receipt.get("reference_export_summary_sha256"),
            label="reference export summary hash",
        )


def _validate_ordered_hashes(
    task_manifest: dict[str, object],
    *,
    task: str,
    rows: int,
    ordered_field: str,
    aggregate_fields: Sequence[str],
) -> list[str]:
    values = task_manifest.get(ordered_field)
    if not isinstance(values, list) or len(values) != rows:
        raise ValueError(f"frozen benchmark task {task} has invalid {ordered_field} length")
    for index, value in enumerate(values):
        _require_sha256(value, label=f"{task}.{ordered_field}[{index}]")
    observed = sha256_json(values)
    for field in aggregate_fields:
        if task_manifest.get(field) != observed:
            raise ValueError(f"frozen benchmark task {task} has invalid {field}")
    return [str(value) for value in values]


def validate_frozen_benchmark_manifest(
    manifest: dict[str, object],
    *,
    expected_tasks: Sequence[str] | None = None,
    require_generation_receipt: bool = True,
    require_reference_evidence: bool = True,
) -> dict[str, object]:
    """Validate the complete schema-v3 frozen confirmation manifest.

    The freezer, evaluator, and publication aggregator call this same function
    so they cannot silently disagree about semantic-instance, normalized-row,
    reference-output, solver-contract, verifier, or native-evidence identities.

    ``require_generation_receipt=False`` exists only for legacy/unit-level
    leakage-manifest checks. Confirmation and publication aggregation must use
    the defaults. ``require_reference_evidence=False`` is likewise restricted
    to explicitly non-publishable smoke/legacy validation.

    The returned object is a compact, derived identity suitable for embedding
    in evaluation provenance; it is not a replacement for the manifest.
    """

    if not isinstance(manifest, dict):
        raise ValueError("frozen benchmark manifest must be a JSON object")
    stored_manifest_sha = _require_sha256(manifest.get("manifest_sha256"), label="manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    observed_manifest_sha = sha256_json(unsigned)
    if stored_manifest_sha != observed_manifest_sha:
        raise ValueError(
            "frozen benchmark manifest hash mismatch; self-hash mismatch: "
            f"stored={stored_manifest_sha}, observed={observed_manifest_sha}"
        )
    if manifest.get("schema_version") != 3:
        raise ValueError("frozen benchmark manifest must use schema_version exactly 3")
    if manifest.get("status") != "frozen_before_inference":
        raise ValueError("frozen benchmark manifest status is not frozen_before_inference")
    benchmark_mode = manifest.get("benchmark_mode")
    if benchmark_mode not in {
        PUBLICATION_BENCHMARK_MODE,
        NONPUBLICATION_BENCHMARK_MODE,
    }:
        raise ValueError(
            "frozen benchmark must declare benchmark_mode as publication or nonpublication"
        )
    publication_validation = bool(require_generation_receipt or require_reference_evidence)
    if publication_validation and benchmark_mode != PUBLICATION_BENCHMARK_MODE:
        raise ValueError("nonpublication benchmark cannot be used for confirmation/publication")
    if benchmark_mode == PUBLICATION_BENCHMARK_MODE and not (
        require_generation_receipt and require_reference_evidence
    ):
        raise ValueError(
            "publication benchmark validation cannot disable receipt or reference evidence"
        )
    try:
        minimum_instances = int(manifest.get("minimum_instances_per_task", -1))
    except (TypeError, ValueError):
        minimum_instances = -1
    if minimum_instances <= 0:
        raise ValueError("frozen benchmark minimum_instances_per_task must be positive")

    tasks = manifest.get("tasks")
    test_sets = manifest.get("test_sets")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(task not in SUPPORTED_BENCHMARK_TASKS for task in tasks)
        or len(tasks) != len(set(tasks))
        or tasks != sorted(tasks)
    ):
        raise ValueError("frozen benchmark tasks must be a sorted unique supported list")
    if not isinstance(test_sets, dict) or set(test_sets) != set(tasks):
        raise ValueError("frozen benchmark task list and test_sets disagree")
    if expected_tasks is not None and sorted(expected_tasks) != sorted(tasks):
        raise ValueError(
            "frozen benchmark task set mismatch: "
            f"expected={sorted(expected_tasks)}, observed={sorted(tasks)}"
        )
    verifier = manifest.get("strict_verifier")
    if not isinstance(verifier, dict) or not verifier:
        raise ValueError("frozen benchmark has no strict-verifier identity")
    verifier_sha = sha256_json(verifier)

    contracts = manifest.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("frozen benchmark has no contracts object")
    generator_contract = contracts.get("generator")
    solver_contract = contracts.get("reference_solver")
    if not isinstance(generator_contract, dict) or not generator_contract:
        raise ValueError("frozen benchmark has no generator contract")
    if not isinstance(solver_contract, dict) or not solver_contract:
        raise ValueError("frozen benchmark has no reference solver contract")
    generator_sha = sha256_json(generator_contract)
    solver_sha = sha256_json(solver_contract)
    if contracts.get("generator_contract_sha256") != generator_sha:
        raise ValueError("frozen benchmark generator contract hash is invalid")
    if contracts.get("reference_solver_contract_sha256") != solver_sha:
        raise ValueError("frozen benchmark reference solver contract hash is invalid")
    solver_tasks = solver_contract.get("tasks")
    if (require_generation_receipt or require_reference_evidence) and (
        not isinstance(solver_tasks, dict) or not set(tasks).issubset(solver_tasks)
    ):
        raise ValueError("frozen benchmark solver contract does not define every task")
    if require_reference_evidence:
        if solver_contract.get("schema_version") != 2:
            raise ValueError("reference-evidence benchmark requires solver contract schema 2")
        stored_contract_self_hash = _require_sha256(
            solver_contract.get("contract_sha256"),
            label="reference solver contract self-hash",
        )
        unsigned_solver = dict(solver_contract)
        unsigned_solver.pop("contract_sha256", None)
        if stored_contract_self_hash != sha256_json(unsigned_solver):
            raise ValueError("frozen benchmark reference solver contract self-hash is invalid")
        runner_contract = solver_contract.get("reference_evidence_runner")
        if (
            not isinstance(runner_contract, dict)
            or runner_contract.get("schema_version") != 1
            or runner_contract.get("protocol") != "clasp-iid-reference-evidence-v1"
            or runner_contract.get("strict_verifier_identity_sha256") != verifier_sha
        ):
            raise ValueError(
                "frozen benchmark solver contract has invalid reference-evidence runner binding"
            )
        expected_instances = runner_contract.get("expected_instances_per_task")
        if (
            isinstance(expected_instances, bool)
            or not isinstance(expected_instances, int)
            or expected_instances <= 0
        ):
            raise ValueError("frozen benchmark solver contract has invalid expected instance count")

    rows_by_task: dict[str, int] = {}
    evidence_summaries: dict[str, str] = {}
    for task in tasks:
        task_manifest = test_sets[task]
        if not isinstance(task_manifest, dict):
            raise ValueError(f"frozen benchmark task {task} manifest is not an object")
        source_locator = task_manifest.get("source_locator")
        if benchmark_mode == PUBLICATION_BENCHMARK_MODE:
            _normalized_relative_path(
                source_locator,
                label=f"frozen benchmark task {task}.source_locator",
            )
        elif not isinstance(source_locator, str) or not source_locator.strip():
            raise ValueError(f"frozen benchmark task {task} has no source locator")
        try:
            row_count = int(task_manifest.get("rows", -1))
        except (TypeError, ValueError):
            row_count = -1
        if row_count < minimum_instances:
            raise ValueError(
                f"frozen benchmark task {task} has {row_count} rows, below "
                f"minimum {minimum_instances}"
            )
        rows_by_task[task] = row_count
        if require_reference_evidence:
            if row_count != expected_instances:
                raise ValueError(
                    f"frozen benchmark task {task} row count differs from the solver contract"
                )
            task_solver = solver_tasks.get(task) if isinstance(solver_tasks, dict) else None
            if not isinstance(task_solver, dict):
                raise ValueError(f"frozen benchmark solver contract lacks task {task}")
            task_runner = task_solver.get("evidence_runner")
            # TSP and CVRP deliberately retain heuristic references.  OP now
            # uses the exact Gurobi evidence lane, whose exported rows are
            # accepted only after native OPTIMAL incumbent/bound equality and
            # strict route replay.  Keep this classification aligned with
            # scripts/run_reference_evidence.py and the assembly/freezer
            # certificate checks so a heuristic OP contract cannot become a
            # publication benchmark merely by resealing this outer manifest.
            heuristic_reference = task in {"tsp", "cvrp"}
            expected_public = (
                "BEST_KNOWN_HEURISTIC_VERIFIED"
                if heuristic_reference
                else "OPTIMAL_CERTIFIED"
            )
            expected_native = "HEURISTIC_COMPLETE" if heuristic_reference else "OPTIMAL"
            expected_semantics = {
                "tsp": "clasp_raw_euclidean_float64",
                "cvrp": "clasp_raw_euclidean_float64",
                "op": "clasp_raw_euclidean_float64_integer_prizes",
                "mis": "clasp_integer_mis_cardinality",
                "jssp": "clasp_integer_jssp_makespan",
            }[task]
            expected_quality = (
                "best_known_heuristic" if heuristic_reference else "certified_optimum"
            )
            parameters = task_solver.get("parameters")
            identity = task_runner.get("solver_identity") if isinstance(task_runner, dict) else None
            if task_solver.get("accepted_statuses") != [expected_public]:
                raise ValueError(
                    f"frozen benchmark solver contract has invalid accepted status for {task}"
                )
            if (
                not isinstance(task_runner, dict)
                or task_runner.get("approval") != "approved"
                or task_runner.get("accepted_native_statuses") != [expected_native]
                or task_runner.get("objective_semantics") != expected_semantics
                or task_runner.get("reference_quality") != expected_quality
            ):
                raise ValueError(
                    f"frozen benchmark solver contract evidence rule is invalid for {task}"
                )
            if not isinstance(parameters, dict) or not parameters:
                raise ValueError(
                    f"frozen benchmark solver contract parameters are invalid for {task}"
                )
            seed_parameter = task_runner.get("seed_parameter")
            if (
                not isinstance(seed_parameter, str)
                or seed_parameter not in parameters
                or isinstance(parameters[seed_parameter], bool)
                or not isinstance(parameters[seed_parameter], int)
            ):
                raise ValueError(f"frozen benchmark solver contract seed is invalid for {task}")
            if not str(task_runner.get("adapter", "")).strip() or not isinstance(identity, dict):
                raise ValueError(f"frozen benchmark solver adapter/identity is invalid for {task}")
            for field in (
                "backend_name",
                "native_version",
                "artifact_kind",
                "artifact_path",
            ):
                if not str(identity.get(field, "")).strip():
                    raise ValueError(
                        f"frozen benchmark solver identity {field} is invalid for {task}"
                    )
            _require_sha256(
                identity.get("artifact_sha256"),
                label=f"reference_solver.tasks.{task}.solver_identity.artifact_sha256",
            )
        semantic = _validate_ordered_hashes(
            task_manifest,
            task=task,
            rows=row_count,
            ordered_field="ordered_semantic_instance_sha256",
            aggregate_fields=("instance_hashes_sha256",),
        )
        if task_manifest.get("ordered_instance_sha256") != semantic:
            raise ValueError(
                f"frozen benchmark task {task} legacy/semantic instance hashes disagree"
            )
        _validate_ordered_hashes(
            task_manifest,
            task=task,
            rows=row_count,
            ordered_field="ordered_benchmark_row_sha256",
            aggregate_fields=("row_hashes_sha256",),
        )
        _validate_ordered_hashes(
            task_manifest,
            task=task,
            rows=row_count,
            ordered_field="ordered_reference_output_sha256",
            aggregate_fields=("reference_hashes_sha256",),
        )
        files = task_manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"frozen benchmark task {task} has no file manifest")
        file_rows = 0
        for index, file_record in enumerate(files):
            if not isinstance(file_record, dict):
                raise ValueError(f"frozen benchmark task {task} file {index} is invalid")
            _normalized_relative_path(
                file_record.get("relative_path"),
                label=f"frozen benchmark task {task} file {index}",
            )
            _require_sha256(file_record.get("sha256"), label=f"{task}.files[{index}].sha256")
            try:
                size = int(file_record.get("size_bytes", -1))
                count = int(file_record.get("rows", -1))
            except (TypeError, ValueError):
                size, count = -1, -1
            if size <= 0 or count <= 0:
                raise ValueError(f"frozen benchmark task {task} file {index} has invalid size/rows")
            file_rows += count
        if file_rows != row_count:
            raise ValueError(f"frozen benchmark task {task} file row counts disagree")
        if task_manifest.get("file_manifest_sha256") != sha256_json(files):
            raise ValueError(f"frozen benchmark task {task} file manifest hash is invalid")
        audit = task_manifest.get("reference_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"frozen benchmark task {task} has no reference audit")
        if audit.get("strictly_feasible") != row_count or audit.get("zero_gap") != row_count:
            raise ValueError(f"frozen benchmark task {task} reference audit is incomplete")
        for field in ("objective_min", "objective_max"):
            try:
                objective = float(audit.get(field))
            except (TypeError, ValueError):
                objective = math.nan
            if not math.isfinite(objective):
                raise ValueError(
                    f"frozen benchmark task {task} reference audit has invalid {field}"
                )
        if require_reference_evidence:
            if task_manifest.get("reference_evidence_schema_version") != 1:
                raise ValueError(f"frozen benchmark task {task} lacks reference-evidence schema")
            summary_sha = _require_sha256(
                task_manifest.get("reference_evidence_summary_sha256"),
                label=f"{task}.reference_evidence_summary_sha256",
            )
            evidence_summaries[task] = summary_sha
            _validate_ordered_hashes(
                task_manifest,
                task=task,
                rows=row_count,
                ordered_field="ordered_reference_evidence_record_sha256",
                aggregate_fields=("reference_evidence_records_sha256",),
            )
            _validate_ordered_hashes(
                task_manifest,
                task=task,
                rows=row_count,
                ordered_field="ordered_reference_export_record_sha256",
                aggregate_fields=("reference_export_records_sha256",),
            )

    overlap = manifest.get("overlap_audit")
    if (
        not isinstance(overlap, dict)
        or overlap.get("passed") is not True
        or overlap.get("collisions") != 0
    ):
        raise ValueError("frozen benchmark has no zero-collision overlap audit")
    forbidden = manifest.get("forbidden_sources")
    if not isinstance(forbidden, list):
        raise ValueError("frozen benchmark forbidden_sources must be a list")
    seen_forbidden: set[tuple[str, str]] = set()
    labels_by_task: dict[str, set[str]] = {task: set() for task in tasks}
    for index, source in enumerate(forbidden):
        if benchmark_mode == PUBLICATION_BENCHMARK_MODE and isinstance(source, dict):
            _normalized_relative_path(
                source.get("source_locator"),
                label=f"frozen benchmark forbidden source {index}.source_locator",
            )
        key, _projection = _validate_forbidden_source(
            source,
            label=f"frozen benchmark forbidden source {index}",
            require_stored_identity=benchmark_mode == PUBLICATION_BENCHMARK_MODE,
        )
        source_task, source_label = key
        if key in seen_forbidden:
            raise ValueError(f"frozen benchmark has duplicate forbidden source {key}")
        seen_forbidden.add(key)
        if source_task in labels_by_task:
            labels_by_task[source_task].add(source_label)
    registry_identity: dict[str, object] | None = None
    registry = manifest.get("forbidden_source_registry")
    embedded_registry_identity = manifest.get("forbidden_source_registry_identity")
    if benchmark_mode == PUBLICATION_BENCHMARK_MODE:
        if not isinstance(registry, dict):
            raise ValueError("publication benchmark has no frozen forbidden-source registry")
        registry_identity = validate_forbidden_sources_against_registry(
            forbidden,
            registry,
            expected_tasks=tasks,
        )
        if embedded_registry_identity != registry_identity:
            raise ValueError("frozen benchmark forbidden-source registry identity is invalid")
    elif registry is not None or embedded_registry_identity is not None:
        if not isinstance(registry, dict):
            raise ValueError("nonpublication benchmark registry snapshot must be an object")
        registry_identity = validate_forbidden_sources_against_registry(
            forbidden,
            registry,
            expected_tasks=tasks,
        )
        if embedded_registry_identity != registry_identity:
            raise ValueError("frozen benchmark forbidden-source registry identity is invalid")
    if require_generation_receipt or require_reference_evidence:
        for task, labels in labels_by_task.items():
            missing = sorted(REQUIRED_FORBIDDEN_SOURCE_LABELS - labels)
            if missing:
                raise ValueError(
                    f"frozen benchmark task {task} is missing required forbidden sources: {missing}"
                )
    if overlap.get("tested_forbidden_sources") != len(forbidden):
        raise ValueError(
            "frozen benchmark overlap-audit source count does not match forbidden sources"
        )

    receipt = manifest.get("generation_receipt")
    if require_generation_receipt:
        _validate_generation_receipt_closure(
            receipt,
            tasks=tasks,
            test_sets=test_sets,
            generator_sha=generator_sha,
            solver_sha=solver_sha,
            verifier=verifier,
            registry_identity=registry_identity,
            require_reference_evidence=require_reference_evidence,
        )
    elif receipt is not None and not isinstance(receipt, dict):
        raise ValueError("frozen benchmark generation_receipt must be an object or null")

    return {
        "manifest_sha256": stored_manifest_sha,
        "schema_version": 3,
        "status": "frozen_before_inference",
        "benchmark_mode": benchmark_mode,
        "tasks": list(tasks),
        "rows_by_task": rows_by_task,
        "generator_contract_sha256": generator_sha,
        "reference_solver_contract_sha256": solver_sha,
        "strict_verifier_identity_sha256": verifier_sha,
        "forbidden_source_registry_sha256": (
            registry_identity["registry_sha256"] if registry_identity is not None else None
        ),
        "forbidden_source_tree_manifest_sha256": (
            registry_identity["source_tree_manifest_sha256"]
            if registry_identity is not None
            else None
        ),
        "forbidden_source_registry_identity": registry_identity,
        "reference_evidence_schema_version": 1 if require_reference_evidence else None,
        "reference_evidence_summary_sha256_by_task": evidence_summaries,
    }
