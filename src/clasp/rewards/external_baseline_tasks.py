"""Opt-in graded rewards for the remaining common-contract CO tasks.

These use the canonical task parsers and strict feasibility predicates. They
do not change the existing TSP/OP/JSSP objective heads, or the historical
CVRP/MIS graded heads. Empty/unparseable outputs score -1; other invalid
outputs score -V/(1+V), with instance-fixed denominators. Feasible outputs
score 1+quality in [1,2]. Declared objectives never affect these rewards.

TSP: V=(missing cities + duplicates + invalid IDs + invalid boundary)/n.
OP: V=(duplicates + invalid IDs + invalid start + too-short route)/n plus
distance-budget excess/budget; feasible quality=prize/sum(instance prizes).
JSSP: V=missing/extra machine rows/m + missing/duplicate/invalid job IDs/(nm)
plus the canonical fraction of operations in a precedence cycle. Machine
orders induce nonoverlapping schedules through the same topological decoder
used by the evaluator. There is no separate hard makespan bound in this task.
TSP/JSSP feasible quality=reference_cost/(reference_cost+recomputed_cost).
"""
from __future__ import annotations

import json
import math
from numbers import Real

from clasp.rewards import co_tasks as co


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _coordinates(coords):
    if not isinstance(coords, list) or len(coords) < 2 or any(
        not isinstance(point, list) or len(point) != 2 or not all(_finite(v) for v in point)
        for point in coords
    ):
        raise ValueError("Graded routing reward requires finite 2-D coordinates for at least two nodes")
    return coords


def validate_graded_instance(task, value):
    """Validate the new tasks' structured metadata before scoring any action."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("Graded reward requires valid structured instance JSON") from exc
    if task == "tsp":
        return _coordinates(value)
    if task == "op":
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("Graded OP requires [coordinates, prizes, distance_budget]")
        coords, prizes, budget = value
        _coordinates(coords)
        if (not isinstance(prizes, list) or len(prizes) != len(coords)
                or any(not _finite(v) or v < 0 for v in prizes)
                or not _finite(sum(prizes))):
            raise ValueError("Graded OP requires one finite nonnegative prize per node")
        if not _finite(budget) or budget <= 0:
            raise ValueError("Graded OP requires a finite positive distance budget")
        return coords, prizes, budget
    if task == "jssp":
        if not isinstance(value, list) or not value or not isinstance(value[0], list):
            raise ValueError("Graded JSSP requires a nonempty operation matrix")
        width = len(value[0])
        if width == 0 or width % 2:
            raise ValueError("Graded JSSP rows require alternating machine,duration pairs")
        machines = set(range(width // 2))
        for job in value:
            if (not isinstance(job, list) or len(job) != width
                    or any(not _finite(v) or int(v) != v or v < 0 for v in job)
                    or set(job[::2]) != machines):
                raise ValueError("Graded JSSP requires rectangular integral jobs visiting each machine once")
        if sum(sum(job[1::2]) for job in value) <= 0:
            raise ValueError("Graded JSSP requires positive total processing time")
        return co._jssp_jobs(value, "")
    raise ValueError(f"Unsupported new graded reward task: {task}")


def _cost_quality(reference, candidate):
    if (not _finite(reference) or reference <= 0
            or not _finite(candidate) or candidate < 0):
        raise ValueError("Feasible graded reward requires a positive reference and finite recomputed cost")
    return 1.0 + 1.0 / (1.0 + candidate / reference)


def graded_tsp_reward(route, coords, reference_route):
    if not route:
        return -1.0
    n = len(coords)
    if co._check_tsp_constraints(route, n)[2]:
        if not co._check_tsp_constraints(reference_route, n)[2]:
            raise ValueError("Feasible graded TSP reward requires a feasible reference tour")
        return _cost_quality(co._distance(coords, reference_route), co._distance(coords, route))
    middle = route[:-1] if len(route) >= 2 and route[-1] == route[0] else route
    present = set(v for v in middle if 0 <= v < n)
    missing = n - len(present)
    duplicates = len(middle) - len(set(middle))
    invalid = sum(not 0 <= v < n for v in route)
    boundary = int(len(route) < 2 or route[0] != 0 or route[-1] != 0)
    return co._grpo_invalid_reward((missing + duplicates + invalid + boundary) / n)


def graded_op_reward(route, coords, prizes, budget):
    if not route:
        return -1.0
    if co._check_op_constraints(route, coords, budget)[2]:
        collected = co._op_prize(route, prizes)
        total = sum(prizes)
        return 1.0 + (collected / total if total > 0 else 0.0)
    n = len(coords)
    severity = (len(route) - len(set(route)) + sum(not 0 <= v < n for v in route)
                + int(route[0] != 0) + int(len(route) < 2)) / n
    distance = co._distance(coords, route)
    if distance is not None:
        if not math.isfinite(distance):
            raise ValueError("Graded OP recomputed distance must be finite")
        # Match the canonical strict check's absolute 1e-9 feasibility tolerance.
        if distance > budget + 1e-9:
            severity += (distance - budget) / budget
    return co._grpo_invalid_reward(severity)


def graded_jssp_reward(schedule, jobs, reference_schedule):
    if not schedule:
        return -1.0
    diagnostics = co._jssp_schedule_diagnostics(jobs, schedule)
    if co._check_jssp_constraints(jobs, schedule)[2]:
        if not co._check_jssp_constraints(jobs, reference_schedule)[2]:
            raise ValueError("Feasible graded JSSP reward requires a feasible reference schedule")
        return _cost_quality(co._schedule_makespan(jobs, reference_schedule), diagnostics["makespan"])
    n, m = len(jobs), len(jobs[0])
    defects = 0
    for order in schedule:
        present = set(v for v in order if 0 <= v < n)
        defects += n - len(present) + len(order) - len(set(order))
        defects += sum(not 0 <= v < n for v in order)
    severity = abs(len(schedule) - m) / m + defects / (n * m)
    severity += float(diagnostics["cycle_node_fraction"])
    return co._grpo_invalid_reward(severity)


def additional_task_graded_reward(task, prompts, completions, **kwargs):
    """Score aligned validated rows; the profile enforces specialist/env gates."""
    rewards = []
    for completion, answer, instance in zip(completions, kwargs["answer"], kwargs["instance_json"]):
        metadata = validate_graded_instance(task, instance)
        if task == "tsp":
            parse = co._parse_tsp_solution
            value = graded_tsp_reward(parse(co._as_text(completion))["route"], metadata,
                                      parse(co._as_text(answer))["route"])
        elif task == "op":
            value = graded_op_reward(co._parse_op_solution(co._as_text(completion))["route"], *metadata)
        elif task == "jssp":
            parse = co._parse_jssp_solution
            value = graded_jssp_reward(parse(co._as_text(completion))["schedule"], metadata,
                                       parse(co._as_text(answer))["schedule"])
        else:
            raise ValueError(f"Unsupported new graded reward task: {task}")
        rewards.append(value)
    return rewards
