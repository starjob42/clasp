"""Prospective SFT-anchored RLOO pilot contracts and independent raw evaluation.

This module has no model, dataset or reward-stack dependency. Production fixes
100 updates, four groups of nine responses and a 3,000-token completion cap.
Explicit ``system_smoke`` contracts exercise integration only. Reference values
are recomputed from typed feasible witnesses; printed objectives are ignored.
All percentages and paired differences below are in percentage points.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
import re
from collections import Counter
from typing import Any, Mapping, Sequence

PROTOCOL = "sft-anchor-rloo-op-cvrp-20261013-v1"
TASKS = ("op", "cvrp")
SCOPES = ("production", "system_smoke")
ARMS = {
    "rloo_scalar_beta0": {"method": "rloo", "quality_credit_lambda": 0.0, "beta": 0.0},
    "rloo_scalar_beta005": {"method": "rloo", "quality_credit_lambda": 0.0, "beta": 0.05},
    "rloo_cardinal_beta0": {"method": "rloo", "quality_credit_lambda": 0.25, "beta": 0.0},
    "rloo_cardinal_beta005": {"method": "rloo", "quality_credit_lambda": 0.25, "beta": 0.05},
    "foarl": {"method": "foarl", "quality_credit_lambda": 0.0, "beta": 0.05},
    "sft": {"method": "sft", "quality_credit_lambda": 0.0, "beta": 0.0},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def numeric_instance(task: str, instance: Any) -> list:
    """Normalize numeric representation, preserving every node and constraint."""
    require(task in TASKS, "Only OP and CVRP are admitted")
    if isinstance(instance, Mapping):
        instance = instance.get("instance_json", instance.get("instance"))
    if type(instance) is str:
        try:
            instance = json.loads(instance)
        except (ValueError, TypeError) as exc:
            raise ValueError("Structured numeric instance required") from exc
    require(type(instance) in (list, tuple) and len(instance) == 3,
            "Instance must be [coordinates, prizes/demands, budget/capacity]")
    coords, values, limit = instance
    require(type(coords) is list and len(coords) >= 2 and all(
        type(point) is list and len(point) == 2 and all(_finite(v) for v in point)
        for point in coords), "Finite two-dimensional coordinates required")
    require(type(values) is list and len(values) == len(coords) and
            all(_finite(v) and v >= 0 for v in values) and values[0] == 0,
            "Aligned nonnegative prizes/demands and zero depot value required")
    require(_finite(limit) and limit > 0, "Positive finite budget/capacity required")
    if task == "cvrp":
        require(all(0 < v <= limit for v in values[1:]),
                "Every CVRP customer must have positive demand within capacity")
    # Normalize integer/float spellings and signed zero for overlap detection.
    def number(value):
        # Keep exact integers: float conversion must not merge distinct large IDs.
        return int(value) if value == int(value) else float(value)
    return [[[number(v) for v in point] for point in coords],
            [number(v) for v in values], number(limit)]


def semantic_id(task: str, instance: Any) -> str:
    """Ordered mathematical-instance identity; independent of text and IDs."""
    return digest({"task": task, "numeric_instance": numeric_instance(task, instance)})


def graph_semantic_id(task: str, instance: Any) -> str:
    """Conservative overlap identity ignoring customer order and scalar limits.

    Reusing the same depot and customer coordinates/values with another budget,
    capacity or numbering is still a reused graph for data admission.
    """
    coords, values, _ = numeric_instance(task, instance)
    return digest({"task": task, "depot": coords[0],
                   "customer_vertices": sorted([point + [value]
                                                for point, value in zip(coords[1:], values[1:])])})


def parse_solution(task: str, text: str) -> list | None:
    """First balanced labelled list, with exact integer leaves and no repairs."""
    require(task in TASKS, "Unknown task")
    if type(text) is not str:
        return None
    label = "Routes" if task == "cvrp" else "Route"
    match = re.search(r"\b" + label + r"\s*:\s*", text, re.IGNORECASE)
    start = text.find("[", match.end() if match else 0)
    if start < 0:
        return None
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == "[") - (text[end] == "]")
        if depth == 0:
            try:
                value = ast.literal_eval(text[start:end + 1])
            except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
                return None
            if type(value) is not list:
                return None
            if task == "op":
                return value if all(type(node) is int for node in value) else None
            return value if value and all(
                type(route) is list and route and all(type(node) is int for node in route)
                for route in value) else None
    return None


def _raw_score(task: str, text: str, instance: Any) -> dict:
    coords, values, limit = numeric_instance(task, instance)
    solution = parse_solution(task, text)
    failures = []
    if solution is None or not solution:
        failures.append("format_or_integer_type")
    elif task == "op":
        if len(solution) < 2 or solution[0] != 0:
            failures.append("open_path_depot_or_minimum_visits")
        if len(solution) != len(set(solution)):
            failures.append("duplicate_visit")
        if any(node < 0 or node >= len(coords) for node in solution):
            failures.append("node_range")
    else:
        customers = [node for route in solution for node in route if node != 0]
        if any(node < 0 or node >= len(coords) for route in solution for node in route):
            failures.append("node_range")
        if any(len(route) < 2 or route[0] != 0 or route[-1] != 0 for route in solution):
            failures.append("route_boundary")
        if any(0 in route[1:-1] for route in solution):
            failures.append("interior_depot")
        if any(count != 1 for count in Counter(customers).values()):
            failures.append("duplicate_customer")
        if set(customers) != set(range(1, len(coords))):
            failures.append("customer_coverage")
        if any(math.fsum(values[node] for node in route if 0 < node < len(coords))
               > limit + 1e-9 for route in solution):
            failures.append("capacity_exceeded")
    distance = None
    if not failures:
        routes = [solution] if task == "op" else solution
        distance = math.fsum(math.hypot(coords[a][0] - coords[b][0],
                                       coords[a][1] - coords[b][1])
                             for route in routes for a, b in zip(route, route[1:]))
        require(math.isfinite(distance), "Recomputed route distance must be finite")
        if task == "op" and distance > limit + 1e-9:
            failures.append("raw_budget_exceeded")
    objective = None
    if not failures:
        objective = (math.fsum(values[node] for node in solution)
                     if task == "op" else distance)
        require(_finite(objective) and objective >= 0, "Recomputed objective must be finite")
    return {"parsed": solution is not None, "feasible": not failures,
            "objective": objective, "distance": distance, "solution": solution,
            "violations": failures, "constraint_view": "strict_typed_no_repair"}


def reference_value(task: str, instance: Any, reference_answer: str) -> float:
    score = _raw_score(task, reference_answer, instance)
    require(score["feasible"] and score["objective"] > 0,
            "Reference witness must be strictly feasible with positive recomputed objective")
    return score["objective"]


def score_completion(task: str, text: str, instance: Any, reference_answer: str) -> dict:
    reference = reference_value(task, instance, reference_answer)
    score = _raw_score(task, text, instance)
    gap = None
    if score["feasible"]:
        gap = ((score["objective"] - reference) / reference if task == "cvrp"
               else (reference - score["objective"]) / reference)
        require(math.isfinite(gap), "Reference gap must be finite")
    return {**score, "reference_objective": reference, "signed_gap": gap,
            "clipped_gap": max(0.0, gap) if gap is not None else None,
            "u10": max(0.0, 1.0 - max(0.0, gap) / 0.10) if gap is not None else 0.0}


def _normalize_row(task: str, row: Mapping[str, Any]) -> dict:
    require(isinstance(row, Mapping), "Dataset rows must be mappings")
    require(row.get("task_type", task) == task, "Row task differs from contract task")
    instance = numeric_instance(task, row)
    sem = semantic_id(task, instance)
    identifier = row.get("instance_id", row.get("id", sem))
    require(type(identifier) is str and identifier, "Instance ID must be nonempty text")
    answer = row.get("reference_answer", row.get("answer", row.get("output")))
    require(type(answer) is str, "A raw reference answer is required")
    ref_value = reference_value(task, instance, answer)
    instruction, input_text = row.get("instruction", ""), row.get("input", "")
    require(type(instruction) is str and type(input_text) is str,
            "Instruction and input must be text")
    prompt = row.get("prompt", instruction + "\n" + input_text)
    require(type(prompt) is str and bool(prompt.strip()), "Nonempty prompt text required")
    return {"instance_id": identifier, "semantic_id": sem,
            "graph_semantic_id": graph_semantic_id(task, instance), "task_type": task,
            "instance_json": instance, "instruction": instruction, "input": input_text,
            "prompt": prompt, "prompt_sha256": digest(prompt), "output": answer,
            "reference_answer": answer, "reference_objective": ref_value,
            "reference_id": digest({"semantic_id": sem, "answer": answer,
                                    "recomputed_objective": ref_value})}


def validate_cohorts(train_rows: Sequence[Mapping[str, Any]],
                     validation_rows: Sequence[Mapping[str, Any]], *, task: str,
                     forbidden_rows: Sequence[Mapping[str, Any]] = ()) -> dict:
    """Reject duplicate mathematics and leakage, regardless of row IDs/spelling."""
    require(isinstance(train_rows, (list, tuple)) and train_rows and
            isinstance(validation_rows, (list, tuple)) and validation_rows,
            "Nonempty training and validation cohorts required")
    normalized = {}
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        values = [_normalize_row(task, row) for row in rows]
        require(len({row["instance_id"] for row in values}) == len(values),
                name + " contains duplicate instance IDs")
        require(len({row["semantic_id"] for row in values}) == len(values),
                name + " contains duplicate mathematical instances")
        require(len({row["graph_semantic_id"] for row in values}) == len(values),
                name + " contains repeated coordinate/value graphs")
        normalized[name] = values
    train_ids = {row["semantic_id"] for row in normalized["train"]}
    validation_ids = {row["semantic_id"] for row in normalized["validation"]}
    require(not train_ids & validation_ids, "Training/validation semantic overlap")
    train_graphs = {row["graph_semantic_id"] for row in normalized["train"]}
    validation_graphs = {row["graph_semantic_id"] for row in normalized["validation"]}
    require(not train_graphs & validation_graphs, "Training/validation graph semantic overlap")
    require(not {row["prompt_sha256"] for row in normalized["train"]} &
            {row["prompt_sha256"] for row in normalized["validation"]},
            "Training/validation prompt overlap")
    forbidden_ids = {semantic_id(task, row) for row in forbidden_rows}
    forbidden_graphs = {graph_semantic_id(task, row) for row in forbidden_rows}
    require(not validation_ids & forbidden_ids, "Validation overlaps a previously used cohort")
    require(not validation_graphs & forbidden_graphs, "Validation graph overlaps a previously used cohort")
    return {"task": task, "training_rows": normalized["train"],
            "validation_rows": normalized["validation"],
            "train_semantic_sha256": digest(sorted(train_ids)),
            "validation_semantic_sha256": digest(sorted(validation_ids)),
            "train_graph_semantic_sha256": digest(sorted(train_graphs)),
            "validation_graph_semantic_sha256": digest(sorted(validation_graphs)),
            "validation_cohort_sha256": digest(normalized["validation"]),
            "forbidden_semantic_sha256": digest(sorted(forbidden_ids)),
            "forbidden_graph_semantic_sha256": digest(sorted(forbidden_graphs)),
            "forbidden_instance_count": len(forbidden_ids),
            "semantic_overlap_count": 0,
            "freshness_claim": "disjoint_from_supplied_training_and_forbidden_instances_only"}


def make_contract(task: str, initializer: Mapping[str, Any], train_rows: Sequence,
                  validation_rows: Sequence, *, train_seed: int = 43,
                  eval_seed: int = 20261013, horizon: int = 100,
                  scope: str = "production", max_completion_tokens: int = 3000,
                  forbidden_rows: Sequence = (), cohort_provenance: Mapping | None = None) -> dict:
    require(scope in SCOPES and type(horizon) is int and 0 < horizon <= 100,
            "Unknown scope or invalid horizon")
    require(type(max_completion_tokens) is int and 0 < max_completion_tokens <= 3000,
            "Invalid completion token cap")
    if scope == "production":
        require(horizon == 100 and max_completion_tokens == 3000,
                "Production fixes horizon100 and completion cap3000")
    require(type(train_seed) is int and type(eval_seed) is int,
            "Training and evaluation seeds must be integers")
    require(isinstance(initializer, Mapping) and initializer and
            (initializer.get("id") or initializer.get("sha256")),
            "Initializer requires a nonempty JSON identity with id or sha256")
    cohorts = validate_cohorts(train_rows, validation_rows, task=task,
                               forbidden_rows=forbidden_rows)
    ids = [row["instance_id"] for row in cohorts["training_rows"]]
    rng = random.Random(train_seed)
    draws = [[rng.choice(ids) for _ in range(4)] for _ in range(horizon)]
    contract = {"schema_version": 1, "protocol": PROTOCOL, "task": task,
                "scope": scope, "initializer": copy.deepcopy(dict(initializer)),
                "initializer_sha256": digest(initializer), "train_seed": train_seed,
                "eval_seed": eval_seed, "arms": copy.deepcopy(ARMS),
                **cohorts, "cohort_provenance": copy.deepcopy(dict(cohort_provenance or {})),
                "training": {"horizon": horizon, "prompt_groups_per_step": 4,
                             "num_generations": 9, "responses_per_step": 36,
                             "max_completion_tokens": max_completion_tokens,
                             "fixed_loss_denominator": max_completion_tokens,
                             "prompt_draws": draws, "prompt_draw_sha256": digest(draws),
                             "learning_rate": 1e-5,
                             "temperature": 0.7, "top_p": 0.9, "top_k": 50},
                "evaluation": {"decoders": ["greedy", "k8"], "raw_candidates": 8,
                               "max_completion_tokens": max_completion_tokens,
                               "temperature": 0.7, "top_p": 0.9, "top_k": 50,
                               "repair": False, "seed": eval_seed,
                               "instance_seeds": {row["instance_id"]: eval_seed + i
                                                  for i, row in enumerate(cohorts["validation_rows"])}},
                "compute_claim": "matched_updates_prompts_responses_and_caps;actual_tokens_and_walltime_reported"}
    contract["contract_sha256"] = digest(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> dict:
    require(isinstance(contract, Mapping) and contract.get("protocol") == PROTOCOL and
            contract.get("schema_version") == 1, "Unknown pilot contract")
    supplied = copy.deepcopy(dict(contract))
    require(supplied.get("contract_sha256") == digest(
        {key: value for key, value in supplied.items() if key != "contract_sha256"}),
        "Pilot contract digest differs")
    expected = make_contract(supplied["task"], supplied["initializer"], supplied["training_rows"],
                             supplied["validation_rows"], train_seed=supplied["train_seed"],
                             eval_seed=supplied["eval_seed"], horizon=supplied["training"]["horizon"],
                             scope=supplied["scope"],
                             max_completion_tokens=supplied["training"]["max_completion_tokens"],
                             cohort_provenance=supplied["cohort_provenance"])
    # Forbidden rows are not reread; their admitted digest remains part of the receipt.
    for key in ("forbidden_semantic_sha256", "forbidden_graph_semantic_sha256", "forbidden_instance_count"):
        expected[key] = supplied[key]
    expected["contract_sha256"] = digest({key: value for key, value in expected.items()
                                          if key != "contract_sha256"})
    require(expected == supplied, "Pilot numerical recipe/cohort differs")
    return supplied


def _summary(scores: Sequence[Mapping[str, Any]]) -> dict:
    feasible = [score for score in scores if score["feasible"]]
    n, total = len(feasible), len(scores)
    return {"N": total, "n": n, "feasibility_percent": 100.0 * n / total,
            "signed_gap_percent": 100.0 * math.fsum(s["signed_gap"] for s in feasible) / n if n else None,
            "clipped_gap_percent": 100.0 * math.fsum(s["clipped_gap"] for s in feasible) / n if n else None,
            "u10_percent": 100.0 * math.fsum(s["u10"] for s in scores) / total,
            "gap_denominator": "method_own_strictly_feasible_support",
            "feasibility_utility_denominator": "all_instances"}


def evaluate_arm(contract: Mapping[str, Any], arm: str, completions: Sequence[Mapping],
                 checkpoint: Mapping[str, Any], resources: Mapping[str, Any]) -> dict:
    """Score exactly one greedy and eight raw draws per admitted validation ID."""
    contract = validate_contract(contract)
    require(arm in ARMS, "Unknown pilot arm")
    require(isinstance(checkpoint, Mapping) and checkpoint, "Checkpoint identity required")
    require(checkpoint.get("initializer_sha256") == contract["initializer_sha256"],
            "Checkpoint initializer differs from common initialization")
    require(checkpoint.get("train_seed") == contract["train_seed"], "Checkpoint training seed differs")
    expected_steps = 0 if arm == "sft" else contract["training"]["horizon"]
    require(checkpoint.get("step") == expected_steps, "Checkpoint is not the fixed terminal checkpoint")
    required_resources = {"training_optimizer_steps", "training_responses", "training_generated_tokens",
                          "training_wall_seconds", "evaluation_wall_seconds", "prompt_draw_sha256"}
    require(isinstance(resources, Mapping) and required_resources <= resources.keys(),
            "Actual response, token, walltime and sampler receipts required")
    require(type(resources["training_optimizer_steps"]) is int and
            resources["training_optimizer_steps"] == expected_steps and
            type(resources["training_responses"]) is int and
            resources["training_responses"] == 36 * expected_steps,
            "Actual training updates/responses differ from declaration")
    require(type(resources["training_generated_tokens"]) is int and
            resources["training_generated_tokens"] >= 0 and
            all(_finite(resources[key]) and resources[key] >= 0
                for key in ("training_wall_seconds", "evaluation_wall_seconds")),
            "Invalid actual token/walltime receipt")
    require(resources["prompt_draw_sha256"] == contract["training"]["prompt_draw_sha256"],
            "Actual prompt draws differ from common schedule")
    if ARMS[arm]["beta"] > 0:
        require(checkpoint.get("reference_initializer_sha256") == contract["initializer_sha256"],
                "Positive KL requires the common frozen SFT reference identity")
    cohort = {row["instance_id"]: row for row in contract["validation_rows"]}
    slots, completion_ids, raw = {}, set(), []
    require(isinstance(completions, (list, tuple)), "Completions must be a raw row list")
    for item in completions:
        require(isinstance(item, Mapping), "Raw completion row required")
        identifier, mode, index = item.get("instance_id"), item.get("decode"), item.get("sample_index")
        require(identifier in cohort and mode in ("greedy", "k8") and type(index) is int and
                (index == 0 if mode == "greedy" else 0 <= index < 8),
                "Completion cohort/decoder/sample index differs")
        completion_id = item.get("completion_id")
        require(type(completion_id) is str and completion_id and completion_id not in completion_ids,
                "Completion IDs must be nonempty and unique")
        slot = (identifier, mode, index)
        require(slot not in slots, "Duplicate completion slot")
        require(type(item.get("text")) is str, "Raw completion text required")
        tokens = item.get("generated_tokens")
        require(type(tokens) is int and 0 <= tokens <= contract["evaluation"]["max_completion_tokens"],
                "Actual generated-token count required and must obey common cap")
        row = cohort[identifier]
        scored = score_completion(contract["task"], item["text"], row["instance_json"], row["reference_answer"])
        # Truncation/error is a raw runtime failure even if its text happens to parse.
        runtime_complete = item.get("runtime_complete", True)
        require(type(runtime_complete) is bool, "runtime_complete must be explicit boolean when present")
        if not runtime_complete:
            scored.update(feasible=False, objective=None, signed_gap=None, clipped_gap=None, u10=0.0)
            scored["violations"].append("runtime_incomplete")
        record = {**copy.deepcopy(dict(item)), "runtime_complete": runtime_complete,
                  "semantic_id": row["semantic_id"], "reference_id": row["reference_id"],
                  "reference_answer": row["reference_answer"], "score": scored}
        slots[slot] = record
        completion_ids.add(completion_id)
        raw.append(record)
    expected_slots = {(identifier, mode, index) for identifier in cohort
                      for mode, count in (("greedy", 1), ("k8", 8)) for index in range(count)}
    require(set(slots) == expected_slots, "Incomplete evaluation: exact greedy plus raw K8 required")
    selected = {"greedy": [], "k8": []}
    for identifier in cohort:
        for mode, count in (("greedy", 1), ("k8", 8)):
            candidates = [slots[identifier, mode, i] for i in range(count)]
            feasible = [record for record in candidates if record["score"]["feasible"]]
            if feasible:
                chosen = min(feasible, key=lambda record: (
                    record["score"]["objective"] * (-1 if contract["task"] == "op" else 1),
                    record["sample_index"]))
                score, chosen_id = chosen["score"], chosen["completion_id"]
            else:
                score = {"feasible": False, "objective": None, "signed_gap": None,
                         "clipped_gap": None, "u10": 0.0}
                chosen_id = None
            selected[mode].append({"instance_id": identifier, "semantic_id": cohort[identifier]["semantic_id"],
                                   "reference_id": cohort[identifier]["reference_id"],
                                   "selected_completion_id": chosen_id, **score})
    actual_resources = {**copy.deepcopy(dict(resources)), "evaluation_responses": len(raw),
                        "evaluation_generated_tokens": sum(row["generated_tokens"] for row in raw)}
    result = {"schema_version": 1, "protocol": PROTOCOL, "status": "complete_raw_evaluation",
              "scope": contract["scope"], "task": contract["task"], "arm": arm,
              "hyperparameters": copy.deepcopy(ARMS[arm]), "train_seed": contract["train_seed"],
              "eval_seed": contract["eval_seed"], "contract_sha256": contract["contract_sha256"],
              "initializer_sha256": contract["initializer_sha256"],
              "validation_cohort_sha256": contract["validation_cohort_sha256"],
              "evaluation_settings": contract["evaluation"], "training_settings": contract["training"],
              "checkpoint": copy.deepcopy(dict(checkpoint)), "resources": actual_resources,
              "raw_completions": raw, "selected": selected,
              "metrics": {mode: _summary(rows) for mode, rows in selected.items()},
              "reference_optimality_claimed": False, "superiority_established": False,
              "control_semantics": "local_matched_foarl_recipe;not_official_benchmark_reproduction",
              "compute_claim": contract["compute_claim"]}
    result["result_sha256"] = digest(result)
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(left: Sequence[Mapping], right: Sequence[Mapping], *,
                     seed: int = 20261013, replicates: int = 50000) -> dict:
    """Paired instance bootstrap; gap support is fixed to observed joint feasibility."""
    require(type(seed) is int and type(replicates) is int and replicates >= 100,
            "Bootstrap requires an integer seed and at least100 replicates")
    require(left and len(left) == len(right) and
            [row["instance_id"] for row in left] == [row["instance_id"] for row in right] and
            [row["reference_id"] for row in left] == [row["reference_id"] for row in right],
            "Paired instance/reference identities differ")
    all_vectors = [[100.0 * (float(a["feasible"]) - float(b["feasible"])),
                    100.0 * (a["u10"] - b["u10"])] for a, b in zip(left, right)]
    joint_vectors = [[100.0 * (a["signed_gap"] - b["signed_gap"]),
                      100.0 * (a["clipped_gap"] - b["clipped_gap"])]
                     for a, b in zip(left, right) if a["feasible"] and b["feasible"]]
    rng = random.Random(seed)
    output = {"N": len(left), "n_joint": len(joint_vectors), "bootstrap_seed": seed,
              "bootstrap_replicates": replicates, "interval": "pointwise95_percentile",
              "uncertainty_scope": "fixed_policies_saved_pools_and_cohort;not_training_seed_or_familywise"}
    for names, vectors in ((("feasibility_delta_pp", "u10_delta_pp"), all_vectors),
                           (("joint_signed_gap_delta_pp", "joint_clipped_gap_delta_pp"), joint_vectors)):
        if not vectors:
            output.update({name: {"estimate": None, "ci95": None} for name in names})
            continue
        n = len(vectors)
        samples = [[], []]
        for _ in range(replicates):
            totals = [0.0, 0.0]
            for _ in range(n):
                row = vectors[rng.randrange(n)]
                totals[0] += row[0]
                totals[1] += row[1]
            samples[0].append(totals[0] / n)
            samples[1].append(totals[1] / n)
        for index, name in enumerate(names):
            output[name] = {"estimate": math.fsum(row[index] for row in vectors) / n,
                            "ci95": [_percentile(samples[index], .025),
                                     _percentile(samples[index], .975)]}
    return output


def compare_results(results: Sequence[Mapping], *, bootstrap_seed: int = 20261013,
                    bootstrap_reps: int = 50000) -> dict:
    """Report matched development evidence, never manufacture a benchmark win."""
    require(type(bootstrap_seed) is int and type(bootstrap_reps) is int and bootstrap_reps >= 100,
            "Invalid bootstrap settings")
    groups = {}
    for value in results:
        require(isinstance(value, Mapping) and value.get("protocol") == PROTOCOL and
                value.get("status") == "complete_raw_evaluation", "Incomplete or unknown result")
        require(value.get("result_sha256") == digest(
            {key: item for key, item in value.items() if key != "result_sha256"}),
            "Result digest differs")
        require(value.get("arm") in ARMS and value.get("hyperparameters") == ARMS[value["arm"]],
                "Result arm/hyperparameters differ")
        key = (value["task"], value["train_seed"])
        require(value["arm"] not in groups.setdefault(key, {}), "Duplicate task/seed/arm result")
        groups[key][value["arm"]] = value
    output = {"protocol": PROTOCOL, "superiority_established": False,
              "benchmark_superiority_claim": "not_established", "groups": [], "reasons": [],
              "bootstrap_seed": bootstrap_seed, "bootstrap_replicates": bootstrap_reps,
              "multiple_comparisons": "pointwise_intervals;no_familywise_or_selection_adjustment"}
    if not groups:
        output["reasons"].append("No completed matched arms")
    for (task, train_seed), arms in sorted(groups.items()):
        missing = sorted(set(ARMS) - set(arms))
        group = {"task": task, "train_seed": train_seed, "completed_arms": sorted(arms),
                 "missing_arms": missing, "comparisons": [], "matched": True}
        output["groups"].append(group)
        if missing:
            output["reasons"].append(f"{task}/seed{train_seed}: incomplete arms: {', '.join(missing)}")
        values = list(arms.values())
        match_keys = ("scope", "contract_sha256", "initializer_sha256", "validation_cohort_sha256",
                      "eval_seed", "evaluation_settings", "training_settings")
        if any(any(value[key] != values[0][key] for key in match_keys) for value in values[1:]):
            group["matched"] = False
            output["reasons"].append(f"{task}/seed{train_seed}: unmatched protocol/cohort/initializer")
            continue
        if values[0]["scope"] == "system_smoke":
            output["reasons"].append(f"{task}/seed{train_seed}: system smoke is integration evidence only")
        for candidate in ARMS:
            if not candidate.startswith("rloo_") or candidate not in arms:
                continue
            for control in ("foarl", "sft"):
                if control not in arms:
                    continue
                for mode in ("greedy", "k8"):
                    effect = paired_bootstrap(arms[candidate]["selected"][mode],
                                              arms[control]["selected"][mode], seed=bootstrap_seed,
                                              replicates=bootstrap_reps)
                    group["comparisons"].append({"left": candidate, "right": control,
                                                  "decoder": mode, **effect})
    seeds = {task: sorted({seed for candidate_task, seed in groups if candidate_task == task})
             for task in TASKS}
    output["training_seeds_by_task"] = seeds
    for task, task_seeds in seeds.items():
        if len(task_seeds) < 5:
            output["reasons"].append(f"{task}: fewer than five paired training seeds; no seed-robust claim")
    output["reasons"].append("Development validation pilot requires a separately frozen held-out confirmation and preselected recipe")
    return output
