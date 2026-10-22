"""Deterministic repair-and-improve evaluation lane for CLASP.

This entry point is deliberately separate from :mod:`clasp.entrypoints.evaluate`.
Raw generation remains the primary, directly comparable system.  This module consumes
completed, fingerprinted ``clasp-eval`` artifacts and evaluates a hybrid system made of
the frozen generator plus an explicitly versioned deterministic post-processor.

The repair functions only receive the instance, prompt context, and raw completion.
Ground-truth solutions are passed exclusively to the shared verifier, before and after
repair, and therefore cannot influence the constructed solution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from typing import Any

from clasp.entrypoints.evaluate import (
    EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
    TASK_DIRECTIONS,
    TASK_EVALUATORS,
    _aggregate_records,
    _build_instance_record,
    _canonical_json_bytes,
    _constraint_diagnostics,
    _details_artifact_identity,
    _example_sampling_seed,
    _experiment_fingerprint,
    _load_json_snapshot,
    _parse_instance_value,
    _protocol_fingerprint_material,
    _read_file_snapshot,
    _self_hash_payload,
    _sha256_file,
    _strict_verifier_identity,
    _validate_self_hash,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from clasp.models.publication_fidelity import (
    PublicationFidelityError,
    build_derived_source_config_snapshot,
)
from clasp.rewards.co_tasks import (
    _check_cvrp_constraints,
    _check_jssp_constraints,
    _check_mis_constraints,
    _check_op_constraints,
    _check_tsp_constraints,
    _cvrp_local_search_improved_routes,
    _cvrp_meta,
    _cvrp_total_distance,
    _distance,
    _jssp_jobs,
    _jssp_order_descent_improved_schedule,
    _mis_maximalized_set,
    _mis_meta,
    _op_meta,
    _op_prize,
    _parse_cvrp_solution,
    _parse_jssp_solution,
    _parse_mis_solution,
    _parse_op_solution,
    _parse_tsp_solution,
    _schedule_makespan,
    _tsp_coords_from_instance,
    _tsp_two_opt_improved_route,
)

REPAIR_SCHEMA_VERSION = 1
REPAIR_ALGORITHM_VERSION = "clasp_deterministic_repair_v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_REPAIR_FILE = os.path.abspath(__file__)
_CLASP_ROOT = os.path.dirname(os.path.dirname(_REPAIR_FILE))


class RepairContractError(ValueError):
    """Raised when an input artifact cannot be repaired without ambiguity."""


def build_repair_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLASP's versioned deterministic repair-and-improve hybrid lane on a "
            "completed, fingerprinted clasp-eval artifact."
        )
    )
    parser.add_argument("--details", nargs="+", required=True, help="clasp-eval JSONL file(s).")
    parser.add_argument(
        "--source_config",
        required=True,
        help="Completed clasp-eval run_config JSON establishing the source fingerprint.",
    )
    parser.add_argument("--output", required=True, help="Destination repaired-details JSONL.")
    parser.add_argument("--system_name", default="clasp_repair_v1")
    parser.add_argument("--tsp_max_passes", type=int, default=12)
    parser.add_argument("--cvrp_max_passes", type=int, default=8)
    parser.add_argument("--op_max_passes", type=int, default=8)
    parser.add_argument("--jssp_max_passes", type=int, default=2)
    return parser


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in _HEX_DIGITS for char in text)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not _is_sha256(text):
        raise RepairContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _load_json_object(path: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"{label} does not exist: {absolute}")
    try:
        payload, identity = _load_json_snapshot(absolute, label=label)
    except (RuntimeError, ValueError) as exc:
        raise RepairContractError(str(exc)) from exc
    return payload, {
        "basename": os.path.basename(absolute),
        "size": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def _load_jsonl_records(paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        absolute = os.path.abspath(raw_path)
        if not os.path.isfile(absolute):
            raise FileNotFoundError(f"Details file does not exist: {absolute}")
        try:
            raw_payload, identity = _read_file_snapshot(absolute)
        except RuntimeError as exc:
            raise RepairContractError(str(exc)) from exc
        for line_number, raw_line in enumerate(raw_payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RepairContractError(f"Invalid UTF-8 at {absolute}:{line_number}") from exc
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RepairContractError(f"Malformed JSON at {absolute}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise RepairContractError(f"Expected a JSON object at {absolute}:{line_number}")
            records.append(payload)
        sources.append(
            {
                "basename": os.path.basename(absolute),
                "size": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    if not records:
        raise RepairContractError("No non-empty detail records were supplied")
    return records, sources


def _validate_source_config(config: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if config.get("run_status") != "complete":
        raise RepairContractError("source_config must describe a completed clasp-eval run")
    try:
        _validate_self_hash(config, "run_config_sha256", label="source_config")
    except RuntimeError as exc:
        raise RepairContractError(str(exc)) from exc
    if config.get("experiment_fingerprint_schema_version") != EXPERIMENT_FINGERPRINT_SCHEMA_VERSION:
        raise RepairContractError("source_config uses an unsupported experiment schema")
    provenance = config.get("experiment_provenance")
    if not isinstance(provenance, dict):
        raise RepairContractError("source_config is missing experiment_provenance")
    if provenance.get("schema_version") != EXPERIMENT_FINGERPRINT_SCHEMA_VERSION:
        raise RepairContractError("experiment_provenance uses an unsupported schema")

    data = provenance.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise RepairContractError("source_config data identity is incomplete")
    expected_data_manifest = hashlib.sha256(_canonical_json_bytes(data["files"])).hexdigest()
    if data.get("manifest_sha256") != expected_data_manifest:
        raise RepairContractError("source_config data manifest is internally inconsistent")

    code = provenance.get("code")
    if (
        not isinstance(code, dict)
        or not isinstance(code.get("files"), list)
        or config.get("code_identity") != code
    ):
        raise RepairContractError("source_config code identity is missing or inconsistent")
    expected_code_manifest = hashlib.sha256(_canonical_json_bytes(code["files"])).hexdigest()
    if code.get("manifest_sha256") != expected_code_manifest:
        raise RepairContractError("source_config code manifest is internally inconsistent")

    expected_experiment = _experiment_fingerprint(provenance)
    recorded_experiment = _require_sha256(
        config.get("experiment_fingerprint"), "source_config experiment_fingerprint"
    )
    if recorded_experiment != expected_experiment:
        raise RepairContractError(
            "source_config experiment_fingerprint does not match experiment_provenance"
        )

    protocol_material = _protocol_fingerprint_material(provenance)
    expected_protocol = _experiment_fingerprint(protocol_material)
    recorded_protocol = _require_sha256(
        config.get("protocol_fingerprint"), "source_config protocol_fingerprint"
    )
    if recorded_protocol != expected_protocol:
        raise RepairContractError(
            "source_config protocol_fingerprint does not match experiment_provenance"
        )

    verifier = provenance.get("verifier")
    if not isinstance(verifier, dict) or config.get("verifier_identity") != verifier:
        raise RepairContractError("source_config verifier identity is missing or inconsistent")
    _require_sha256(
        verifier.get("implementation_manifest_sha256"),
        "source_config verifier implementation fingerprint",
    )
    if verifier.get("name") != "clasp_strict_shared_verifier":
        raise RepairContractError("source_config does not identify the strict shared verifier")
    if sorted(verifier.get("tasks", [])) != sorted(TASK_EVALUATORS):
        raise RepairContractError("source_config verifier task registry is inconsistent")
    implementations = verifier.get("implementations")
    if not isinstance(implementations, list):
        raise RepairContractError("source_config verifier implementation list is missing")
    expected_verifier_manifest = hashlib.sha256(_canonical_json_bytes(implementations)).hexdigest()
    if verifier.get("implementation_manifest_sha256") != expected_verifier_manifest:
        raise RepairContractError("source_config verifier manifest is internally inconsistent")

    selection = provenance.get("selection")
    if not isinstance(selection, dict):
        raise RepairContractError("source_config selection contract is missing")
    if list(config.get("tasks", [])) != list(selection.get("tasks", [])):
        raise RepairContractError("source_config tasks disagree with its selection contract")
    if str(config.get("split", "")) != str(selection.get("split", "")):
        raise RepairContractError("source_config split disagrees with its selection contract")
    campaign_binding = selection.get("campaign_run_binding")
    if config.get("evaluation_purpose") == "confirmation":
        claim = config.get("confirmation_attempt_claim")
        if (
            not isinstance(campaign_binding, dict)
            or config.get("campaign_manifest") != selection.get("campaign_manifest")
            or config.get("campaign_run_binding") != campaign_binding
            or not isinstance(claim, dict)
            or claim.get("confirmation_attempt_id")
            != campaign_binding.get("confirmation_attempt_id")
        ):
            raise RepairContractError("source_config confirmation campaign binding is incomplete")
    generation = provenance.get("generation")
    config_generation = config.get("generation")
    if not isinstance(generation, dict) or not isinstance(config_generation, dict):
        raise RepairContractError("source_config generation contract is missing")
    if any(config_generation.get(key) != value for key, value in generation.items()):
        raise RepairContractError("source_config generation contract is internally inconsistent")
    if generation.get("seed_policy") != "per_example_sha256_v1":
        raise RepairContractError("source_config uses an unsupported sampling-seed policy")
    try:
        int(generation["seed"])
        int(generation["num_return_sequences"])
    except (KeyError, TypeError, ValueError):
        raise RepairContractError(
            "source_config generation seed or candidate count is invalid"
        ) from None
    return recorded_experiment, recorded_protocol, provenance


def _validate_instance_for_task(task: str, instance: Any) -> None:
    if task == "tsp":
        coords = _tsp_coords_from_instance(instance, "")
        if len(coords) < 2 or any(not _finite(value) for row in coords for value in row[:2]):
            raise RepairContractError(
                "TSP instance does not contain at least two finite coordinates"
            )
        return
    if task == "cvrp":
        coords, demands, capacity = _cvrp_meta(instance, "")
        if (
            len(coords) < 2
            or len(coords) != len(demands)
            or not _finite(capacity)
            or float(capacity) <= 0.0
            or any(not _finite(value) for row in coords for value in row[:2])
            or any(not _finite(value) or float(value) < 0.0 for value in demands)
            or any(float(value) > float(capacity) + 1e-9 for value in demands[1:])
        ):
            raise RepairContractError("CVRP instance is incomplete or has an impossible demand")
        return
    if task == "op":
        coords, prizes, max_len = _op_meta(instance, "")
        if (
            len(coords) < 2
            or len(coords) != len(prizes)
            or not _finite(max_len)
            or float(max_len) < 0.0
            or any(not _finite(value) for row in coords for value in row[:2])
            or any(not _finite(value) for value in prizes)
        ):
            raise RepairContractError("OP instance is incomplete or non-finite")
        pair_distances = [_distance(coords, [0, node]) for node in range(1, len(coords))]
        if not any(
            value is not None and float(value) <= float(max_len) + 1e-9 for value in pair_distances
        ):
            raise RepairContractError("OP instance has no strictly representable feasible route")
        return
    if task == "mis":
        num_nodes, edges = _mis_meta(instance, "")
        if num_nodes is None or num_nodes <= 0:
            raise RepairContractError("MIS instance is missing a positive num_nodes")
        if any(a < 0 or b < 0 or a >= num_nodes or b >= num_nodes for a, b in edges):
            raise RepairContractError("MIS instance contains an out-of-range edge")
        return
    if task == "jssp":
        jobs = _jssp_jobs(instance, "")
        if not jobs or not jobs[0] or any(len(job) != len(jobs[0]) for job in jobs):
            raise RepairContractError("JSSP instance is empty or non-rectangular")
        machines = set(range(len(jobs[0])))
        for job in jobs:
            if {machine for machine, _duration in job} != machines:
                raise RepairContractError("Each JSSP job must visit every machine exactly once")
            if any(duration <= 0 for _machine, duration in job):
                raise RepairContractError("JSSP durations must be positive")
        return
    raise RepairContractError(f"Unsupported task: {task!r}")


def _candidate_context(
    record: dict[str, Any], candidate: dict[str, Any], candidate_index: int
) -> tuple[str, str, str, Any]:
    task = str(record["task"])
    base = (
        str(record["instruction"]),
        str(record["input_problem"]),
        str(record["ground_truth_output"]),
        _parse_instance_value({"instance_json": record["instance_json"]}),
    )
    has_context = "verification_context" in candidate
    has_hash = "verification_context_sha256" in candidate
    if has_context != has_hash:
        raise RepairContractError(
            f"candidate {candidate_index} has an incomplete verification_context contract"
        )
    if not has_context:
        if candidate.get("tsp_label_tta_variant") not in (None, "", "identity"):
            raise RepairContractError(
                f"candidate {candidate_index} declares TSP label TTA without its context"
            )
        return base
    if task != "tsp":
        raise RepairContractError(
            "candidate-specific verification contexts are only supported for TSP"
        )
    context = candidate.get("verification_context")
    required = {"instruction", "input_problem", "ground_truth_output", "instance_json"}
    if not isinstance(context, dict) or set(context) != required:
        raise RepairContractError(
            f"candidate {candidate_index} has an unsupported verification_context schema"
        )
    expected = hashlib.sha256(_canonical_json_bytes(context)).hexdigest()
    if (
        _require_sha256(
            candidate.get("verification_context_sha256"),
            f"candidate {candidate_index} verification_context_sha256",
        )
        != expected
    ):
        raise RepairContractError(f"candidate {candidate_index} verification_context is corrupted")
    instruction = str(context.get("instruction", ""))
    input_text = str(context.get("input_problem", ""))
    answer = str(context.get("ground_truth_output", ""))
    if not instruction or not input_text or not answer or context.get("instance_json") is None:
        raise RepairContractError(f"candidate {candidate_index} has an incomplete context")
    instance = _parse_instance_value({"instance_json": context["instance_json"]})
    _validate_instance_for_task(task, instance)
    return instruction, input_text, answer, instance


def _validate_source_record(
    record: dict[str, Any],
    record_index: int,
    *,
    experiment_fingerprint: str,
    protocol_fingerprint: str,
    provenance: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[str, str, int]:
    task = str(record.get("task", "")).lower()
    split = str(record.get("split", ""))
    try:
        example_index = int(record.get("example_index"))
    except (TypeError, ValueError):
        raise RepairContractError(f"record {record_index} has an invalid example_index") from None
    if task not in TASK_EVALUATORS or not split or example_index < 0:
        raise RepairContractError(f"record {record_index} has an invalid task/split/index")
    if task not in list(source_config.get("tasks", [])) or split != str(
        source_config.get("split", "")
    ):
        raise RepairContractError(f"record {record_index} lies outside source_config selection")
    if record.get("experiment_fingerprint_schema_version") != EXPERIMENT_FINGERPRINT_SCHEMA_VERSION:
        raise RepairContractError(f"record {record_index} uses an unsupported experiment schema")
    if record.get("experiment_fingerprint") != experiment_fingerprint:
        raise RepairContractError(f"record {record_index} has a mismatched experiment fingerprint")
    if record.get("protocol_fingerprint") != protocol_fingerprint:
        raise RepairContractError(f"record {record_index} has a mismatched protocol fingerprint")
    if record.get("strict_verifier") != provenance.get("verifier"):
        raise RepairContractError(f"record {record_index} has a mismatched verifier identity")
    campaign_binding = provenance.get("selection", {}).get("campaign_run_binding")
    if campaign_binding is not None and (
        not isinstance(campaign_binding, dict)
        or record.get("campaign_manifest_sha256")
        != campaign_binding.get("campaign_manifest_sha256")
        or record.get("confirmation_attempt_id") != campaign_binding.get("confirmation_attempt_id")
    ):
        raise RepairContractError(f"record {record_index} has a mismatched confirmation campaign")
    try:
        _validate_self_hash(
            record,
            "record_sha256",
            label=f"source detail record {record_index}",
        )
    except RuntimeError as exc:
        raise RepairContractError(str(exc)) from exc

    instruction = record.get("instruction")
    input_text = record.get("input_problem")
    answer = record.get("ground_truth_output")
    if not all(isinstance(value, str) and value for value in (instruction, input_text, answer)):
        raise RepairContractError(f"record {record_index} has incomplete transformed text")
    if record.get("instance_json") is None:
        raise RepairContractError(f"record {record_index} is missing embedded instance_json")
    direct_instance_hash = hashlib.sha256(
        _canonical_json_bytes(record["instance_json"])
    ).hexdigest()
    if record.get("instance_sha256") != direct_instance_hash:
        raise RepairContractError(f"record {record_index} has a corrupted embedded instance")
    for identity_field in (
        "semantic_instance_sha256",
        "benchmark_row_sha256",
        "reference_output_sha256",
    ):
        if not _is_sha256(record.get(identity_field)):
            raise RepairContractError(f"record {record_index} is missing a valid {identity_field}")
    transformed_hash = hashlib.sha256(
        _canonical_json_bytes(
            {
                "instruction": instruction,
                "input": input_text,
                "output": answer,
                "instance_json": record["instance_json"],
            }
        )
    ).hexdigest()
    if record.get("transformed_example_sha256") != transformed_hash:
        raise RepairContractError(f"record {record_index} has corrupted transformed row content")

    data = provenance.get("data")
    data_manifest = data.get("manifest_sha256") if isinstance(data, dict) else None
    locator = record.get("source_row_locator")
    expected_locator = {
        "data_manifest_sha256": data_manifest,
        "task": task,
        "split": split,
        "merged_dataset_index": example_index,
    }
    if not _is_sha256(data_manifest) or locator != expected_locator:
        raise RepairContractError(f"record {record_index} has a corrupt source_row_locator")
    try:
        sampling_seed = int(record.get("sampling_seed"))
    except (TypeError, ValueError):
        raise RepairContractError(f"record {record_index} is missing its sampling_seed") from None
    expected_sampling_seed = _example_sampling_seed(
        int(provenance["generation"]["seed"]), task, split, example_index
    )
    if sampling_seed != expected_sampling_seed:
        raise RepairContractError(f"record {record_index} has a corrupt sampling_seed")

    candidates = record.get("candidates")
    try:
        expected_candidates = int(record.get("num_return_sequences"))
    except (TypeError, ValueError):
        raise RepairContractError(f"record {record_index} has an invalid candidate count") from None
    if not isinstance(candidates, list) or not candidates or len(candidates) != expected_candidates:
        raise RepairContractError(f"record {record_index} has an incomplete candidate list")
    if expected_candidates != int(provenance["generation"]["num_return_sequences"]):
        raise RepairContractError(f"record {record_index} has a mismatched candidate budget")
    context_flags: list[bool] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise RepairContractError(
                f"record {record_index}, candidate {candidate_index} is not an object"
            )
        if candidate.get("candidate_index") != candidate_index:
            raise RepairContractError(
                f"record {record_index}, candidate {candidate_index} has a corrupt index"
            )
        if not isinstance(candidate.get("completion"), str):
            raise RepairContractError(
                f"record {record_index}, candidate {candidate_index} is missing raw completion"
            )
        completion_tokens = candidate.get("completion_tokens")
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 0
        ):
            raise RepairContractError(
                f"record {record_index}, candidate {candidate_index} has no token denominator"
            )
        _candidate_context(record, candidate, candidate_index)
        context_flags.append("verification_context" in candidate)
    if any(context_flags) and not all(context_flags):
        raise RepairContractError(
            f"record {record_index} mixes candidate-context and row-context schemas"
        )
    if int(record.get("generated_tokens", -1)) != sum(
        int(candidate["completion_tokens"]) for candidate in candidates
    ):
        raise RepairContractError(f"record {record_index} has an inconsistent token total")

    instance = _parse_instance_value({"instance_json": record["instance_json"]})
    _validate_instance_for_task(task, instance)
    return task, split, example_index


def _repair_tsp(
    instance: Any, context: str, completion: str, max_passes: int
) -> tuple[str, list[str]]:
    coords = _tsp_coords_from_instance(instance, context)
    n = len(coords)
    parsed = _parse_tsp_solution(completion)["route"]
    _, _, raw_feasible = _check_tsp_constraints(parsed, n)
    actions: list[str] = []
    if raw_feasible:
        route = list(parsed)
    else:
        actions.append("sanitize_and_complete_route")
        kept: list[int] = []
        seen: set[int] = set()
        for node in parsed:
            if 0 < node < n and node not in seen:
                kept.append(node)
                seen.add(node)
        route = [0, *kept, 0]
        missing = set(range(1, n)) - seen
        while missing:
            best: tuple[float, int, int] | None = None
            for node in sorted(missing):
                for pos in range(len(route) - 1):
                    a, b = route[pos], route[pos + 1]
                    delta = (
                        _distance(coords, [a, node])
                        + _distance(coords, [node, b])
                        - _distance(coords, [a, b])
                    )
                    key = (float(delta), node, pos)
                    if best is None or key < best:
                        best = key
            assert best is not None
            _delta, node, pos = best
            route.insert(pos + 1, node)
            missing.remove(node)
    improved = _tsp_two_opt_improved_route(coords, route, max_passes=max_passes)
    if improved is not None:
        route = improved[0]
        actions.append(f"two_opt:{improved[4]}")
    objective = _distance(coords, route)
    if objective is None:
        raise RepairContractError("TSP repair failed to construct a measurable route")
    return f"Route: {route}\nObjective: {objective:.9f}", actions or ["canonicalize_format"]


def _route_load(route: list[int], demands: list[float]) -> float:
    return float(sum(float(demands[node]) for node in route if node != 0))


def _repair_cvrp(
    instance: Any, context: str, completion: str, max_passes: int
) -> tuple[str, list[str]]:
    coords, demands, capacity_raw = _cvrp_meta(instance, context)
    assert capacity_raw is not None
    capacity = float(capacity_raw)
    parsed = _parse_cvrp_solution(completion)["routes"]
    _, _, raw_feasible = _check_cvrp_constraints(parsed, coords, demands, capacity)
    actions: list[str] = []
    if raw_feasible:
        routes = [list(route) for route in parsed]
    else:
        actions.append("sanitize_split_and_complete_routes")
        n = len(coords)
        seen: set[int] = set()
        routes = []
        for raw_route in parsed:
            current: list[int] = []
            current_load = 0.0
            for node in raw_route:
                if node <= 0 or node >= n or node in seen:
                    continue
                demand = float(demands[node])
                if current and current_load + demand > capacity + 1e-9:
                    routes.append([0, *current, 0])
                    current = []
                    current_load = 0.0
                current.append(node)
                current_load += demand
                seen.add(node)
            if current:
                routes.append([0, *current, 0])

        missing = set(range(1, n)) - seen
        while missing:
            best: tuple[float, int, int, int] | None = None
            for node in sorted(missing):
                demand = float(demands[node])
                for route_index, route in enumerate(routes):
                    if _route_load(route, demands) + demand > capacity + 1e-9:
                        continue
                    for pos in range(len(route) - 1):
                        a, b = route[pos], route[pos + 1]
                        delta = (
                            _distance(coords, [a, node])
                            + _distance(coords, [node, b])
                            - _distance(coords, [a, b])
                        )
                        key = (float(delta), node, route_index, pos)
                        if best is None or key < best:
                            best = key
                new_route_cost = float(2.0 * (_distance(coords, [0, node]) or 0.0))
                new_key = (new_route_cost, node, len(routes), 0)
                if best is None or new_key < best:
                    best = new_key
            assert best is not None
            _delta, node, route_index, pos = best
            if route_index == len(routes):
                routes.append([0, node, 0])
            else:
                routes[route_index].insert(pos + 1, node)
            missing.remove(node)
    improved = _cvrp_local_search_improved_routes(
        coords, demands, capacity, routes, max_passes=max_passes
    )
    if improved is not None:
        routes = improved[0]
        actions.append(f"capacity_preserving_local_search:{improved[4]}")
    objective = _cvrp_total_distance(coords, routes)
    if objective is None:
        raise RepairContractError("CVRP repair failed to construct measurable routes")
    return f"Routes: {routes}\nObjective: {objective:.9f}", actions or ["canonicalize_format"]


def _path_length(coords: list[list[float]], route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    value = _distance(coords, route)
    return float(value) if value is not None else float("inf")


def _op_two_opt(
    coords: list[list[float]], route: list[int], max_passes: int
) -> tuple[list[int], int]:
    current = list(route)
    passes = 0
    for _ in range(max_passes):
        current_length = _path_length(coords, current)
        best_key: tuple[float, int, int] | None = None
        for i in range(1, len(current) - 1):
            for end in range(i + 2, len(current) + 1):
                candidate = current[:i] + list(reversed(current[i:end])) + current[end:]
                gain = current_length - _path_length(coords, candidate)
                key = (-gain, i, end)
                if gain > 1e-12 and (best_key is None or key < best_key):
                    best_key = key
        if best_key is None:
            break
        _negative_gain, i, end = best_key
        current[i:end] = reversed(current[i:end])
        passes += 1
    return current, passes


def _best_op_insertion(
    coords: list[list[float]], prizes: list[float], max_len: float, route: list[int]
) -> tuple[int, int] | None:
    missing = set(range(1, len(coords))) - set(route)
    current_length = _path_length(coords, route)
    best: tuple[float, float, float, int, int] | None = None
    for node in sorted(missing):
        prize = float(prizes[node])
        if prize <= 0.0:
            continue
        for pos in range(1, len(route) + 1):
            candidate = route[:pos] + [node] + route[pos:]
            new_length = _path_length(coords, candidate)
            if new_length > max_len + 1e-9:
                continue
            delta = max(0.0, new_length - current_length)
            ratio = prize / max(delta, 1e-12)
            key = (-ratio, -prize, delta, node, pos)
            if best is None or key < best:
                best = key
    return (best[3], best[4]) if best is not None else None


def _repair_op(
    instance: Any, context: str, completion: str, max_passes: int
) -> tuple[str, list[str]]:
    coords, prizes, max_len_raw = _op_meta(instance, context)
    assert max_len_raw is not None
    max_len = float(max_len_raw)
    parsed = _parse_op_solution(completion)["route"]
    _, _, raw_feasible = _check_op_constraints(parsed, coords, max_len)
    actions: list[str] = []
    if raw_feasible:
        route = list(parsed)
    else:
        actions.append("sanitize_and_budget_route")
        route = [0]
        seen = {0}
        for node in parsed:
            if 0 < node < len(coords) and node not in seen:
                route.append(node)
                seen.add(node)
        while len(route) > 1 and _path_length(coords, route) > max_len + 1e-9:
            current_length = _path_length(coords, route)
            choices: list[tuple[float, float, int, int]] = []
            for pos in range(1, len(route)):
                node = route[pos]
                shorter = route[:pos] + route[pos + 1 :]
                saved = current_length - _path_length(coords, shorter)
                ratio = float(prizes[node]) / max(saved, 1e-12) if saved > 0.0 else float("inf")
                choices.append((ratio, float(prizes[node]), node, pos))
            _ratio, _prize, _node, remove_pos = min(choices)
            route.pop(remove_pos)
    route, passes = _op_two_opt(coords, route, max_passes=max_passes)
    if passes:
        actions.append(f"open_path_two_opt:{passes}")
    inserted = 0
    while True:
        insertion = _best_op_insertion(coords, prizes, max_len, route)
        if insertion is None:
            break
        node, pos = insertion
        route.insert(pos, node)
        inserted += 1
        route, extra_passes = _op_two_opt(coords, route, max_passes=max_passes)
        passes += extra_passes
    if inserted:
        actions.append(f"prize_insertion:{inserted}")
    if len(route) < 2:
        raise RepairContractError("OP repair failed to construct a strictly verifiable route")
    objective = _op_prize(route, prizes)
    if objective is None:
        raise RepairContractError("OP repair failed to compute a prize")
    return f"Route: {route}\nObjective: {objective:.9f}", actions or ["canonicalize_format"]


def _repair_mis(instance: Any, context: str, completion: str) -> tuple[str, list[str]]:
    num_nodes, edges = _mis_meta(instance, context)
    assert num_nodes is not None
    parsed = _parse_mis_solution(completion)["node_set"]
    _, _, raw_feasible = _check_mis_constraints(parsed, num_nodes, edges)
    actions: list[str] = []
    if raw_feasible:
        selected = list(parsed)
    else:
        actions.append("sanitize_conflicts")
        neighbors = {node: set() for node in range(num_nodes)}
        for a, b in edges:
            neighbors[a].add(b)
            neighbors[b].add(a)
        raw_rank: dict[int, int] = {}
        for rank, node in enumerate(parsed):
            if 0 <= node < num_nodes and node not in raw_rank:
                raw_rank[node] = rank
        preferred = sorted(raw_rank, key=lambda node: (len(neighbors[node]), raw_rank[node], node))
        selected_set: set[int] = set()
        for node in preferred:
            if not (neighbors[node] & selected_set):
                selected_set.add(node)
        if not selected_set:
            first = min(range(num_nodes), key=lambda node: (len(neighbors[node]), node))
            selected_set.add(first)
        selected = sorted(selected_set)
    maximal = _mis_maximalized_set(selected, num_nodes, edges)
    if maximal is not None:
        selected = maximal[0]
        actions.append(f"maximalize:{maximal[4]}")
    selected = sorted(selected)
    return f"Set: {selected}\nObjective: {len(selected)}", actions or ["canonicalize_format"]


def _serial_jssp_schedule(
    jobs: list[list[tuple[int, int]]], raw_schedule: list[list[int]]
) -> list[list[int]]:
    num_jobs = len(jobs)
    num_machines = len(jobs[0])
    raw_rank: list[dict[int, int]] = []
    for machine in range(num_machines):
        order = raw_schedule[machine] if machine < len(raw_schedule) else []
        ranks: dict[int, int] = {}
        for rank, job in enumerate(order):
            if 0 <= job < num_jobs and job not in ranks:
                ranks[job] = rank
        raw_rank.append(ranks)

    next_operation = [0] * num_jobs
    job_ready = [0.0] * num_jobs
    machine_ready = [0.0] * num_machines
    schedule = [[] for _ in range(num_machines)]
    for _ in range(num_jobs * num_machines):
        choices: list[tuple[float, int, int, int, int]] = []
        for job in range(num_jobs):
            operation = next_operation[job]
            if operation >= num_machines:
                continue
            machine, duration = jobs[job][operation]
            start = max(job_ready[job], machine_ready[machine])
            rank = raw_rank[machine].get(job, num_jobs + job)
            choices.append((start, rank, duration, job, machine))
        if not choices:
            raise RepairContractError("JSSP serial construction terminated early")
        start, _rank, duration, job, machine = min(choices)
        schedule[machine].append(job)
        finish = float(start + duration)
        job_ready[job] = finish
        machine_ready[machine] = finish
        next_operation[job] += 1
    return schedule


def _repair_jssp(
    instance: Any, context: str, completion: str, max_passes: int
) -> tuple[str, list[str]]:
    jobs = _jssp_jobs(instance, context)
    parsed = _parse_jssp_solution(completion)["schedule"]
    _, _, raw_feasible = _check_jssp_constraints(jobs, parsed)
    actions: list[str] = []
    if raw_feasible:
        schedule = [list(order) for order in parsed]
    else:
        schedule = _serial_jssp_schedule(jobs, parsed)
        actions.append("serial_schedule_generation")
    improved = _jssp_order_descent_improved_schedule(jobs, schedule, max_passes=max_passes)
    if improved is not None:
        schedule = improved[0]
        actions.append(f"adjacent_swap_descent:{improved[4]}")
    makespan = _schedule_makespan(jobs, schedule)
    if makespan is None:
        raise RepairContractError("JSSP repair failed to construct an acyclic schedule")
    return f"Schedule: {schedule}\nMakespan: {makespan:.9f}", actions or ["canonicalize_format"]


def repair_completion(
    task: str,
    *,
    instance: Any,
    completion: str,
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    """Repair one candidate without accepting or consulting a ground-truth solution."""

    # Fingerprinted clasp-eval rows are required to embed the structured instance.
    # Intentionally disable the reward module's prompt-text fallbacks here so the
    # algorithm depends on exactly two semantic inputs: that instance and the candidate.
    context = ""
    if task == "tsp":
        return _repair_tsp(instance, context, completion, int(config["tsp_max_passes"]))
    if task == "cvrp":
        return _repair_cvrp(instance, context, completion, int(config["cvrp_max_passes"]))
    if task == "op":
        return _repair_op(instance, context, completion, int(config["op_max_passes"]))
    if task == "mis":
        return _repair_mis(instance, context, completion)
    if task == "jssp":
        return _repair_jssp(instance, context, completion, int(config["jssp_max_passes"]))
    raise RepairContractError(f"Unsupported task: {task!r}")


def _objective_improved_or_equal(task: str, before: Any, after: Any) -> bool:
    if not (_finite(before) and _finite(after)):
        return False
    tolerance = 1e-9 * max(1.0, abs(float(before)), abs(float(after)))
    if TASK_DIRECTIONS[task] == "min":
        return float(after) <= float(before) + tolerance
    return float(after) + tolerance >= float(before)


def _strict_view(record: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_instance_record(
        task=str(record["task"]),
        split=str(record["split"]),
        example_index=int(record["example_index"]),
        source_json=record.get("source_json"),
        source_json_files=list(record.get("source_json_files", []) or []),
        instruction=str(record["instruction"]),
        input_text=str(record["input_problem"]),
        answer=str(record["ground_truth_output"]),
        instance_size=record.get("instance_size"),
        prompt_tokens=int(record.get("prompt_tokens", 0) or 0),
        prompt_was_truncated=bool(record.get("prompt_was_truncated", False)),
        generation_time_sec=float(record.get("generation_time_sec", 0.0) or 0.0),
        num_return_sequences=len(candidates),
        candidates=candidates,
        peak_gpu_memory_allocated_bytes=record.get("peak_gpu_memory_allocated_bytes"),
        peak_gpu_memory_reserved_bytes=record.get("peak_gpu_memory_reserved_bytes"),
    )


def _repair_record(
    record: dict[str, Any],
    *,
    repair_fingerprint: str,
    verifier: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = str(record["task"])
    evaluator = TASK_EVALUATORS[task]
    raw_metrics_list: list[dict[str, Any]] = []
    repaired_metrics_list: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(record["candidates"]):
        instruction, input_text, answer, instance = _candidate_context(
            record, candidate, candidate_index
        )
        completion = str(candidate["completion"])
        raw_metrics = evaluator(instruction, input_text, answer, instance, completion)
        raw_metrics.update(
            _constraint_diagnostics(
                task,
                prompt=instruction,
                input_text=input_text,
                answer=answer,
                instance=instance,
                completion=completion,
            )
        )
        start_time = time.perf_counter()
        repaired_completion, actions = repair_completion(
            task,
            instance=instance,
            completion=completion,
            config=config,
        )
        repair_time = float(time.perf_counter() - start_time)
        repaired_metrics = evaluator(instruction, input_text, answer, instance, repaired_completion)
        repaired_metrics.update(
            _constraint_diagnostics(
                task,
                prompt=instruction,
                input_text=input_text,
                answer=answer,
                instance=instance,
                completion=repaired_completion,
            )
        )
        if not repaired_metrics.get("is_feasible"):
            raise RepairContractError(
                f"repair failed strict verification for {task} candidate {candidate_index}"
            )
        if raw_metrics.get("is_feasible") and not _objective_improved_or_equal(
            task, raw_metrics.get("objective_value"), repaired_metrics.get("objective_value")
        ):
            raise RepairContractError(
                f"repair degraded a strictly feasible {task} candidate {candidate_index}"
            )
        completion_tokens = int(candidate["completion_tokens"])
        raw_metrics = {
            **raw_metrics,
            "candidate_index": candidate_index,
            "completion": completion,
            "completion_tokens": completion_tokens,
        }
        repaired_metrics = {
            **repaired_metrics,
            "candidate_index": candidate_index,
            "completion": repaired_completion,
            "completion_tokens": completion_tokens,
        }
        raw_metrics_list.append(raw_metrics)
        repaired_metrics_list.append(repaired_metrics)
        if not raw_metrics["is_feasible"]:
            status = "repaired"
        elif raw_metrics["parsed_solution"] != repaired_metrics["parsed_solution"]:
            status = "improved"
        else:
            status = "validated_or_reformatted"
        candidate_results.append(
            {
                **repaired_metrics,
                "candidate_index": candidate_index,
                "repair_status": status,
                "repair_actions": actions,
                "repair_time_sec": repair_time,
                "source_completion_tokens": completion_tokens,
                "raw_completion": completion,
                "raw_metrics": raw_metrics,
                "repaired_completion": repaired_completion,
                "repaired_metrics": repaired_metrics,
                "verification_context_sha256": candidate.get("verification_context_sha256"),
            }
        )

    raw_view = _strict_view(record, raw_metrics_list)
    repaired_view = _strict_view(record, repaired_metrics_list)
    raw_record = copy.deepcopy(record)
    raw_record_sha256 = hashlib.sha256(_canonical_json_bytes(raw_record)).hexdigest()
    row_identity = {
        "source_experiment_fingerprint": record["experiment_fingerprint"],
        "training_run_contract_sha256": record.get("training_run_contract_sha256"),
        "source_protocol_fingerprint": record["protocol_fingerprint"],
        "source_row_locator": copy.deepcopy(record["source_row_locator"]),
        "instance_sha256": record["instance_sha256"],
        "semantic_instance_sha256": record["semantic_instance_sha256"],
        "benchmark_row_sha256": record["benchmark_row_sha256"],
        "reference_output_sha256": record["reference_output_sha256"],
        "transformed_example_sha256": record["transformed_example_sha256"],
        "sampling_seed": int(record["sampling_seed"]),
    }
    output = {
        **copy.deepcopy(repaired_view),
        "repair_schema_version": REPAIR_SCHEMA_VERSION,
        "repair_algorithm_version": REPAIR_ALGORITHM_VERSION,
        "repair_fingerprint": repair_fingerprint,
        "protocol_fingerprint": record["protocol_fingerprint"],
        "strict_verifier": verifier,
        "repair_verifier": verifier,
        "row_identity": row_identity,
        "source_row_locator": copy.deepcopy(record["source_row_locator"]),
        "instance_json": copy.deepcopy(record["instance_json"]),
        "instance_sha256": record["instance_sha256"],
        "semantic_instance_sha256": record["semantic_instance_sha256"],
        "benchmark_row_sha256": record["benchmark_row_sha256"],
        "reference_output_sha256": record["reference_output_sha256"],
        "transformed_example_sha256": record["transformed_example_sha256"],
        "sampling_seed": int(record["sampling_seed"]),
        "source_experiment_fingerprint": record["experiment_fingerprint"],
        "raw_record_sha256": raw_record_sha256,
        "raw_record": raw_record,
        "candidates": candidate_results,
        "raw_strict_view": raw_view,
        "repaired_strict_view": repaired_view,
        "total_repair_time_sec": float(
            sum(candidate["repair_time_sec"] for candidate in candidate_results)
        ),
    }
    for field in ("campaign_manifest_sha256", "confirmation_attempt_id"):
        if record.get(field) is not None:
            row_identity[field] = record[field]
            output[field] = record[field]
    output["repair_record_sha256"] = hashlib.sha256(_canonical_json_bytes(output)).hexdigest()
    return output, raw_view, repaired_view


def _implementation_identity() -> dict[str, Any]:
    paths = {
        "entrypoints/repair.py": _REPAIR_FILE,
        "entrypoints/evaluate.py": os.path.join(_CLASP_ROOT, "entrypoints", "evaluate.py"),
        "rewards/co_tasks.py": os.path.join(_CLASP_ROOT, "rewards", "co_tasks.py"),
    }
    files = []
    for relative_path, path in sorted(paths.items()):
        sha256, size = _sha256_file(path)
        files.append({"path": relative_path, "size": size, "sha256": sha256})
    return {
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
        "files": files,
    }


def _repair_config(args: Any) -> dict[str, Any]:
    config = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "tsp_max_passes": int(args.tsp_max_passes),
        "cvrp_max_passes": int(args.cvrp_max_passes),
        "op_max_passes": int(args.op_max_passes),
        "jssp_max_passes": int(args.jssp_max_passes),
        "selection_policy": "strict_feasibility_then_task_objective",
        "ground_truth_access_policy": "verifier_only",
    }
    for name in ("tsp_max_passes", "cvrp_max_passes", "op_max_passes", "jssp_max_passes"):
        if config[name] < 0 or config[name] > 1000:
            raise RepairContractError(f"{name} must be in [0, 1000]")
    return config


def run_repair(args: Any) -> tuple[str, str]:
    output_path = os.path.abspath(args.output)
    summary_path = f"{os.path.splitext(output_path)[0]}__summary.json"
    protected_inputs = {
        os.path.realpath(str(args.source_config)),
        *(os.path.realpath(str(path)) for path in list(args.details)),
    }
    if (
        os.path.realpath(output_path) in protected_inputs
        or os.path.realpath(summary_path) in protected_inputs
    ):
        raise RepairContractError("repair output paths must not overwrite source artifacts")
    if not str(args.system_name).strip():
        raise RepairContractError("system_name must be non-empty")
    source_config, source_config_identity = _load_json_object(args.source_config, "source_config")
    experiment_fingerprint, protocol_fingerprint, provenance = _validate_source_config(
        source_config
    )
    try:
        source_config_snapshot = build_derived_source_config_snapshot(
            source_config,
            source_config_identity,
        )
    except PublicationFidelityError as exc:
        raise RepairContractError(str(exc)) from exc
    records, detail_sources = _load_jsonl_records(list(args.details))
    details_artifact = source_config.get("details_artifact")
    if len(detail_sources) != 1 or not isinstance(details_artifact, dict):
        raise RepairContractError("source_config must bind exactly one completed details artifact")
    detail_source = detail_sources[0]
    if (
        details_artifact.get("sha256") != detail_source["sha256"]
        or details_artifact.get("size_bytes") != detail_source["size"]
        or details_artifact.get("record_count") != len(records)
    ):
        raise RepairContractError(
            "source_config details_artifact does not match the supplied details file"
        )
    seen: set[tuple[str, str, int]] = set()
    for record_index, record in enumerate(records):
        key = _validate_source_record(
            record,
            record_index,
            experiment_fingerprint=experiment_fingerprint,
            protocol_fingerprint=protocol_fingerprint,
            provenance=provenance,
            source_config=source_config,
        )
        if key in seen:
            raise RepairContractError(f"duplicate source row identity: {key}")
        seen.add(key)

    verifier = _strict_verifier_identity()
    implementation = _implementation_identity()
    config = _repair_config(args)
    fingerprint_material = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "system_name": str(args.system_name),
        "source_experiment_fingerprint": experiment_fingerprint,
        "source_protocol_fingerprint": protocol_fingerprint,
        "source_experiment_provenance": provenance,
        "training_run_contract": source_config.get("training_run_contract"),
        "training_run_contract_sha256": source_config.get("training_run_contract_sha256"),
        "source_campaign_manifest": source_config.get("campaign_manifest"),
        "source_campaign_run_binding": source_config.get("campaign_run_binding"),
        "source_confirmation_attempt_claim": source_config.get("confirmation_attempt_claim"),
        "source_config": source_config_identity,
        "source_details": detail_sources,
        "repair_config": config,
        "repair_implementation": implementation,
        "repair_verifier": verifier,
    }
    repair_fingerprint = _experiment_fingerprint(fingerprint_material)

    output_records: list[dict[str, Any]] = []
    raw_views: list[dict[str, Any]] = []
    repaired_views: list[dict[str, Any]] = []
    for record in records:
        output, raw_view, repaired_view = _repair_record(
            record,
            repair_fingerprint=repair_fingerprint,
            verifier=verifier,
            config=config,
        )
        output_records.append(output)
        raw_views.append(raw_view)
        repaired_views.append(repaired_view)

    tasks = sorted({record["task"] for record in records})
    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_lane": "deterministic_repair_and_improve_hybrid",
        "system_name": str(args.system_name),
        "repair_schema_version": REPAIR_SCHEMA_VERSION,
        "repair_algorithm_version": REPAIR_ALGORITHM_VERSION,
        "repair_fingerprint": repair_fingerprint,
        "fingerprint_material": fingerprint_material,
        "source_experiment_fingerprint": experiment_fingerprint,
        "source_protocol_fingerprint": protocol_fingerprint,
        "source_experiment_provenance": provenance,
        "training_run_contract": source_config.get("training_run_contract"),
        "training_run_contract_sha256": source_config.get("training_run_contract_sha256"),
        "source_campaign_manifest": source_config.get("campaign_manifest"),
        "source_campaign_run_binding": source_config.get("campaign_run_binding"),
        "source_confirmation_attempt_claim": source_config.get("confirmation_attempt_claim"),
        "source_config_snapshot": source_config_snapshot,
        "record_count": len(output_records),
        "candidate_count": sum(len(record["candidates"]) for record in output_records),
        "raw_strict": _aggregate_records(raw_views),
        "repaired_strict": _aggregate_records(repaired_views),
        "per_task": {
            task: {
                "raw_strict": _aggregate_records(
                    [view for view in raw_views if view["task"] == task]
                ),
                "repaired_strict": _aggregate_records(
                    [view for view in repaired_views if view["task"] == task]
                ),
            }
            for task in tasks
        },
        "total_repair_time_sec": float(
            sum(record["total_repair_time_sec"] for record in output_records)
        ),
        "files": {"details": output_path, "summary": summary_path},
        "limitations": [
            "This is a hybrid repair-assisted system and is not a raw-generation result.",
            "The deterministic heuristics are not equivalent to FALCON or any other repair system.",
            "Feasibility construction is task-specific and does not guarantee a globally optimal objective.",
            "Repair runtime is Python wall-clock time and must be reported with generation runtime.",
        ],
    }
    _write_jsonl_atomic(output_path, output_records)
    summary["details_artifact"] = _details_artifact_identity(
        output_path,
        record_count=len(output_records),
    )
    summary = _self_hash_payload(summary, "summary_sha256")
    _write_json_atomic(summary_path, summary)
    return output_path, summary_path


def main() -> None:
    args = build_repair_parser().parse_args()
    output_path, summary_path = run_repair(args)
    print(f"Repaired hybrid details: {output_path}")
    print(f"Repaired hybrid summary: {summary_path}")


if __name__ == "__main__":
    main()
