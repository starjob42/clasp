"""Canonical teachers from the current training row for TSP, OP and JSSP.

No dataset access or search occurs here. The caller supplies the aligned
training answer; its declared objective is discarded and feasibility/objective
are recomputed with the same parser and strict predicates as raw evaluation.
"""
from __future__ import annotations

import math

from clasp.rewards import co_tasks as co
from clasp.rewards.external_baseline_tasks import additional_task_graded_reward, validate_graded_instance

TASKS = ("tsp", "op", "jssp")
MINIMIZE = {"tsp", "cvrp", "jssp"}


def canonical_verified_solution(task, text, instance, input_text=""):
    metadata = validate_graded_instance(task, instance)
    if task == "tsp":
        route = co._parse_tsp_solution(co._as_text(text))["route"]
        if not co._check_tsp_constraints(route, len(metadata))[2]:
            return None
        objective = co._distance(metadata, route)
        target = f"Route: {route}, Objective: {objective:.15g}"
    elif task == "op":
        coords, prizes, budget = metadata
        route = co._parse_op_solution(co._as_text(text))["route"]
        if not co._check_op_constraints(route, coords, budget)[2]:
            return None
        objective = co._op_prize(route, prizes)
        target = f"Route: {route}, Objective: {objective:.15g}"
    elif task == "jssp":
        schedule = co._parse_jssp_solution(co._as_text(text))["schedule"]
        if not co._check_jssp_constraints(metadata, schedule)[2]:
            return None
        objective = co._schedule_makespan(metadata, schedule)
        target = co._format_jssp_distillation_target(schedule, objective)
    else:
        raise ValueError(f"Unsupported training-reference teacher task: {task}")
    if objective is None or not math.isfinite(objective) or objective < 0:
        raise ValueError("Verified solution has an invalid recomputed objective")
    return target, float(objective)


def training_reference_target(task, answer, instance, input_text=""):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Training-reference teacher requires an aligned nonempty answer")
    result = canonical_verified_solution(task, answer, instance, input_text)
    if result is None:
        raise ValueError("Training-reference teacher received an infeasible training answer")
    # A reference has no claimed local-search gain/passes or model repair.
    return result[0], 0.0, 0, False


def training_reference_graded_reward(task, prompts, completions, **kwargs):
    count = len(completions)
    if len(prompts) != count or any(not isinstance(kwargs.get(key), (list, tuple))
        or len(kwargs[key]) != count for key in ("answer", "instance_json", "input", "task_type")):
        raise ValueError("Reference-teacher reward requires aligned training metadata")
    if any(value != task for value in kwargs["task_type"]):
        raise ValueError("Reference-teacher reward received another task")
    return additional_task_graded_reward(task, prompts, completions, **kwargs)
