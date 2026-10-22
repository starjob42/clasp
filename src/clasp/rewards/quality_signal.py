"""Versioned reward ablations; historical rewards and evaluation stay unchanged.

``legacy`` delegates the exact existing profile. ``graded_v2`` retains its
feasible quality and fixes OP/JSSP penalties whose diagnostics disappeared when
an invalid ID or duplicate was added. ``reference_sigmoid_v2`` also calibrates
feasible quality to a canonically verified, positive training-reference value.
All modes use the existing canonical parsers and feasibility predicates.

OP measures budget excess on the in-range subsequence; invalid nodes cannot
erase distance already implied by valid nodes. JSSP diagnoses cycles on the
valid-job projection only when every job has an unambiguous machine order.
Missing/extra machine rows, missing jobs or repeated valid jobs make topology
unknown and receive the conservative maximum cycle fraction, 1.
These projections only measure violations: they never make an output feasible.
"""
from __future__ import annotations

import json
import math

from clasp.rewards import co_tasks as co
from clasp.rewards.external_baseline_tasks import validate_graded_instance

REWARD_MODES = ("legacy", "graded_v2", "reference_sigmoid_v2")
QUALITY_SLOPE = 8.0
_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
_PARSERS = {
    "tsp": (co._parse_tsp_solution, "route"),
    "cvrp": (co._parse_cvrp_solution, "routes"),
    "op": (co._parse_op_solution, "route"),
    "mis": (co._parse_mis_solution, "node_set"),
    "jssp": (co._parse_jssp_solution, "schedule"),
}


def _context(task, instance, input_text=""):
    if task not in _TASKS:
        raise ValueError(f"Unsupported quality-signal task: {task}")
    if isinstance(instance, str):
        try:
            instance = json.loads(instance)
        except (ValueError, TypeError) as exc:
            raise ValueError("Quality signal requires structured instance JSON") from exc
    if task in ("tsp", "op", "jssp"):
        return validate_graded_instance(task, instance)
    if task == "cvrp":
        if not isinstance(instance, (list, tuple)) or len(instance) < 3:
            raise ValueError("Quality signal requires CVRP coordinates, demands and capacity")
        context = tuple(instance[:3])
        co._validate_grpo_cvrp_metadata(*context)
        return context
    if not isinstance(instance, dict) or type(instance.get("num_nodes")) is not int:
        raise ValueError("Quality signal requires MIS integer num_nodes metadata")
    n, raw = instance["num_nodes"], instance.get("edges")
    if n <= 0 or not isinstance(raw, list) or any(
        not isinstance(edge, (list, tuple)) or len(edge) != 2
        or any(type(v) is not int or not 0 <= v < n for v in edge)
        or edge[0] == edge[1] for edge in raw
    ):
        raise ValueError("Quality signal requires a positive MIS size and valid simple-graph edges")
    return n, {tuple(sorted(edge)) for edge in raw}


def _parse(task, text):
    parser, key = _PARSERS[task]
    try:
        return parser(co._as_text(text))[key]
    except (ValueError, TypeError, OverflowError):
        # Generated numeric overflow is a malformed action, not bad training data.
        return []


def _objective(task, solution, context):
    if task == "tsp":
        feasible = co._check_tsp_constraints(solution, len(context))[2]
        value = co._distance(context, solution) if feasible else None
    elif task == "cvrp":
        feasible = co._check_cvrp_constraints(solution, *context)[2]
        value = co._cvrp_total_distance(context[0], solution) if feasible else None
    elif task == "op":
        coords, prizes, budget = context
        feasible = co._check_op_constraints(solution, coords, budget)[2]
        value = co._op_prize(solution, prizes) if feasible else None
    elif task == "mis":
        feasible = co._check_mis_constraints(solution, *context)[2]
        value = len(solution) if feasible else None
    else:
        feasible = co._check_jssp_constraints(context, solution)[2]
        value = co._schedule_makespan(context, solution) if feasible else None
    if not feasible:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Canonical feasible objective must be finite and nonnegative")
    return float(value)


def validate_training_reference(task, answer, instance, input_text=""):
    """Recompute a feasible, positive reference; ignore all declared objectives."""
    context = _context(task, instance, input_text)
    value = _objective(task, _parse(task, answer), context)
    if value is None or value <= 0:
        raise ValueError(f"{task.upper()} training reference must be feasible with a positive finite objective")
    return value


def _op_invalid(route, context):
    coords, _, budget = context
    n = len(coords)
    valid = [v for v in route if 0 <= v < n]
    # With no valid segment there is no geometric diagnostic to trust.
    if len(valid) < 2:
        return -1.0
    distance = co._distance(coords, valid)
    if distance is None or not math.isfinite(distance):
        raise ValueError("OP violation distance must be finite")
    severity = (len(route) - len(set(route)) + sum(not 0 <= v < n for v in route)
                + int(route[0] != 0) + int(len(route) < 2)) / n
    if distance > budget + 1e-9:
        severity += (distance - budget) / budget
    return co._grpo_invalid_reward(severity)


def _jssp_invalid(schedule, jobs):
    if not schedule:
        return -1.0
    n, m = len(jobs), len(jobs[0])
    defects = 0
    projected = []
    ambiguous = len(schedule) != m
    for row in schedule:
        valid = [v for v in row if 0 <= v < n]
        ambiguous = ambiguous or len(valid) != len(set(valid))
        projected.append(list(dict.fromkeys(valid)))
        defects += n - len(set(valid)) + len(row) - len(set(row))
        defects += sum(not 0 <= v < n for v in row)
    if not any(projected):
        return -1.0
    # Choosing the first occurrence of a duplicate or dropping a machine row
    # could remove an existing cycle. Such ambiguous orders stay unknown.
    diagnostic = co._jssp_schedule_diagnostics(jobs, projected) if not ambiguous else {}
    cycle_fraction = (float(diagnostic["cycle_node_fraction"])
                      if diagnostic.get("topology_applicable") else 1.0)
    severity = abs(len(schedule) - m) / m + defects / (n * m) + cycle_fraction
    return co._grpo_invalid_reward(severity)


def _sigmoid_quality(task, value, reference):
    # Division may overflow to infinity for extreme finite costs. The stable
    # sigmoid has a well-defined limiting reward in that case.
    ratio = value / reference
    gap = ratio - 1.0 if task in ("tsp", "cvrp", "jssp") else 1.0 - ratio
    z = -QUALITY_SLOPE * gap
    if z >= 0:
        return 1.0 + 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return 1.0 + exp_z / (1.0 + exp_z)


def reward_batch(task, prompts, completions, *, mode, **kwargs):
    """Score aligned training rows under one explicit, prospective reward mode."""
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown quality-signal reward mode: {mode}")
    # Imported lazily so successor profiles can import this module safely.
    from clasp.baselines.external_baseline_dev import _graded_reward

    if mode == "legacy":
        return _graded_reward(task, prompts, completions, kwargs)
    if task not in _TASKS:
        raise ValueError(f"Unsupported quality-signal task: {task}")
    if not isinstance(completions, (list, tuple)):
        raise ValueError("Quality signal requires aligned completion lists")
    count = len(completions)
    for key in ("task_type", "answer", "instance_json", "input"):
        if not isinstance(kwargs.get(key), (list, tuple)) or len(kwargs[key]) != count:
            raise ValueError(f"Quality signal requires one {key} per completion")
    contexts = [_context(task, instance, text)
                for instance, text in zip(kwargs["instance_json"], kwargs["input"])]
    references = None
    if mode == "reference_sigmoid_v2":
        # Validate every reference before looking at sampled feasibility.
        references = [validate_training_reference(task, answer, instance, text)
                      for answer, instance, text in zip(kwargs["answer"], kwargs["instance_json"], kwargs["input"])]
    solutions = [_parse(task, completion) for completion in completions]
    # Preserve legacy scoring for all ordinary parses; malformed numeric actions
    # are replaced only in the scoring call and never in stored model outputs.
    safe_completions = [completion if solution else "" for completion, solution in zip(completions, solutions)]
    rewards = _graded_reward(task, prompts, safe_completions, kwargs)
    for i, (solution, context) in enumerate(zip(solutions, contexts)):
        value = _objective(task, solution, context)
        if value is not None and references is not None:
            rewards[i] = _sigmoid_quality(task, value, references[i])
        elif value is None and task == "op":
            rewards[i] = _op_invalid(solution, context)
        elif value is None and task == "jssp":
            rewards[i] = _jssp_invalid(solution, context)
    if any(not math.isfinite(r) or not -1.0 <= r <= 2.0 for r in rewards):
        raise ValueError("Quality-signal reward must be finite and bounded")
    return rewards
