"""Select one verified teacher per prompt without reference-solution access.

The caller must independently verify source feasibility and every target, then
gather *all* rows belonging to each prompt before calling this pure helper.
Rows with no teacher still contribute their raw feasible incumbent. This module
does not verify text, repair solutions, change on-policy data, or compute loss.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

_SUPPORTED_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
_MAXIMIZE_TASKS = ("op", "mis")


def _objective(value: Any, *, task: str, label: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric objective")
    if task == "mis" and isinstance(value, Integral):
        if value < 0:
            raise ValueError(f"{label} must be a finite nonnegative objective")
        return int(value)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite numeric objective") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative objective")
    if task == "mis":
        if not result.is_integer():
            raise ValueError(f"{label} must be an integer MIS cardinality")
        return int(result)
    return result


def _quality_relation(task: str, candidate: float, incumbent: float, tolerance: float) -> str:
    if task == "mis":
        return "better" if candidate > incumbent else "dominated" if candidate < incumbent else "tied"
    threshold = tolerance * max(abs(candidate), abs(incumbent))
    if task == "op":
        if candidate > incumbent + threshold:
            return "better"
        if candidate < incumbent - threshold:
            return "dominated"
        return "tied"
    if candidate < incumbent - threshold:
        return "better"
    if candidate > incumbent + threshold:
        return "dominated"
    return "tied"


def select_verified_teachers(
    rows: Sequence[Mapping[str, Any]], *, cvrp_relative_tolerance: float = 1e-9,
    selection_mode: str = "strict_improvement",
) -> dict[str, Any]:
    """Return selected row IDs, per-prompt decisions, and explicit-rate diagnostics.

    Required row fields are ``task``, ``prompt_key``, ``row_id``,
    ``source_feasible``, ``source_objective``, ``target_text``, and
    ``target_objective``. Row IDs must be unique nonnegative integers across the
    gathered batch. Prompt keys must be nonempty strings; groups use both task
    and prompt key. An absent teacher uses ``None`` for both target fields.
    Infeasible source objectives may be ``None`` or a finite number, but never
    contribute to the incumbent. MIS objectives must be integer cardinalities.
    TSP/CVRP distance and JSSP makespan are minimized; OP prize and MIS
    cardinality are maximized. All objectives must be finite and nonnegative.

    The default ``strict_improvement`` mode requires a strictly better target
    when there is a feasible raw incumbent. ``feasibility_or_improvement`` also
    admits targets at least as good as that incumbent when any raw source in
    the prompt group is infeasible. This additional admission uses an exact
    directional comparison, so a target slightly worse than the incumbent is
    rejected even if their objective difference is within the tie tolerance.
    In both modes, all-infeasible groups select the best available repair.
    The caller can include canonical raw feasible solutions as verified targets;
    this helper neither constructs targets nor infers them from source metadata.
    TSP/CVRP/OP/JSSP compare objectives using ``rtol * max(abs(a), abs(b))``;
    the existing ``cvrp_relative_tolerance`` keyword supplies that tolerance for
    all four tasks. MIS comparisons remain exact. JSSP allows real-valued
    makespans because a verified schedule can have noninteger start times.
    Among targets tied with the best admissible
    objective, ``target_text`` and then ``row_id`` provide deterministic ties.
    Selected row IDs are returned in ascending order, independent of input
    order. Caller-owned rows are never modified.
    """
    if not isinstance(selection_mode, str) or selection_mode not in (
        "strict_improvement", "feasibility_or_improvement",
    ):
        raise ValueError("selection_mode must be strict_improvement or feasibility_or_improvement")
    if (isinstance(cvrp_relative_tolerance, bool)
            or not isinstance(cvrp_relative_tolerance, Real)
            or not math.isfinite(float(cvrp_relative_tolerance))
            or not 0 <= cvrp_relative_tolerance < 1):
        raise ValueError("cvrp_relative_tolerance must be finite and in [0, 1)")
    tolerance = float(cvrp_relative_tolerance)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a sequence of row mappings")
    required = {"task", "prompt_key", "row_id", "source_feasible", "source_objective",
                "target_text", "target_objective"}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    observed_row_ids = set()
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or not required <= raw.keys():
            raise ValueError(f"row {position} lacks required teacher-selection fields")
        task, prompt_key, row_id = raw["task"], raw["prompt_key"], raw["row_id"]
        if task not in _SUPPORTED_TASKS:
            raise ValueError(f"row {position} has unsupported task")
        if not isinstance(prompt_key, str) or not prompt_key.strip():
            raise ValueError(f"row {position} requires a nonempty string prompt_key")
        if isinstance(row_id, bool) or not isinstance(row_id, Integral) or row_id < 0:
            raise ValueError(f"row {position} requires a nonnegative integer row_id")
        row_id = int(row_id)
        if row_id in observed_row_ids:
            raise ValueError(f"duplicate row_id: {row_id}")
        observed_row_ids.add(row_id)
        feasible = raw["source_feasible"]
        if not isinstance(feasible, bool):
            raise ValueError(f"row {row_id} source_feasible must be bool")
        source = raw["source_objective"]
        if feasible or source is not None:
            source = _objective(source, task=task, label=f"row {row_id} source_objective")
        text, target = raw["target_text"], raw["target_objective"]
        if text is None:
            if target is not None:
                raise ValueError(f"row {row_id} absent target requires both target fields None")
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row {row_id} target_text must be nonempty or None")
            target = _objective(target, task=task, label=f"row {row_id} target_objective")
        grouped.setdefault((task, prompt_key), []).append({
            "row_id": row_id, "source_feasible": feasible, "source_objective": source,
            "target_text": text, "target_objective": target,
        })

    selected, groups = [], []
    counts = {
        "input_rows": len(rows), "prompt_groups": len(grouped),
        "groups_with_feasible_incumbent": 0, "groups_with_available_teacher": 0,
        "groups_all_sources_infeasible": 0, "selected_improvement_groups": 0,
        "selected_repair_groups": 0, "eligible_teachers": 0,
        "groups_mixed_feasibility": 0, "selected_mixed_feasibility_groups": 0,
        "comparable_teachers": 0, "better_teachers": 0, "tied_teachers": 0,
        "dominated_teachers": 0, "all_invalid_group_teachers": 0,
    }
    for (task, prompt_key), group in sorted(grouped.items()):
        objective_best = max if task in _MAXIMIZE_TASKS else min
        feasible_sources = [row["source_objective"] for row in group if row["source_feasible"]]
        mixed_feasibility = 0 < len(feasible_sources) < len(group)
        incumbent = objective_best(feasible_sources) if feasible_sources else None
        teachers = [row for row in group if row["target_text"] is not None]
        best_target = objective_best(row["target_objective"] for row in teachers) if teachers else None
        counts["groups_with_feasible_incumbent"] += bool(feasible_sources)
        counts["groups_all_sources_infeasible"] += not bool(feasible_sources)
        counts["groups_mixed_feasibility"] += mixed_feasibility
        counts["groups_with_available_teacher"] += bool(teachers)
        counts["eligible_teachers"] += len(teachers)
        relations = {"better": 0, "tied": 0, "dominated": 0}
        admissible = []
        target_values: dict[str, tuple[float, float]] = {}
        for teacher in teachers:
            text, value = teacher["target_text"], teacher["target_objective"]
            low, high = target_values.get(text, (value, value))
            low, high = min(low, value), max(high, value)
            if _quality_relation(task, low, high, tolerance) != "tied":
                raise ValueError(f"prompt {prompt_key!r} has inconsistent objectives for identical target_text")
            target_values[text] = (low, high)
            if incumbent is None:
                admissible.append(teacher)
                counts["all_invalid_group_teachers"] += 1
            else:
                relation = _quality_relation(task, value, incumbent, tolerance)
                relations[relation] += 1
                counts[f"{relation}_teachers"] += 1
                counts["comparable_teachers"] += 1
                preserves_incumbent = value >= incumbent if task in _MAXIMIZE_TASKS else value <= incumbent
                feasibility_admission = (
                    selection_mode == "feasibility_or_improvement"
                    and mixed_feasibility and preserves_incumbent
                )
                if relation == "better" or feasibility_admission:
                    admissible.append(teacher)
        selected_row_id = None
        if admissible:
            best_admissible = objective_best(row["target_objective"] for row in admissible)
            tied_best = [row for row in admissible
                         if _quality_relation(task, row["target_objective"], best_admissible, tolerance) == "tied"]
            winner = min(tied_best, key=lambda row: (row["target_text"], row["row_id"]))
            selected_row_id = winner["row_id"]
            selected.append(selected_row_id)
            if incumbent is None:
                reason = "all_sources_infeasible_repair"
                counts["selected_repair_groups"] += 1
            elif _quality_relation(task, winner["target_objective"], incumbent, tolerance) == "better":
                reason = "strict_improvement"
                counts["selected_improvement_groups"] += 1
            else:
                reason = "mixed_feasibility"
                counts["selected_mixed_feasibility_groups"] += 1
        else:
            reason = "no_teacher" if not teachers else "no_strict_improvement"
        groups.append({
            "task": task, "prompt_key": prompt_key, "input_rows": len(group),
            "raw_feasible_sources": len(feasible_sources), "eligible_teachers": len(teachers),
            "raw_incumbent_objective": incumbent, "best_available_teacher_objective": best_target,
            "better_teachers": relations["better"], "tied_teachers": relations["tied"],
            "dominated_teachers": relations["dominated"],
            "selected_row_id": selected_row_id, "reason": reason,
        })
    counts["selected_teachers"] = len(selected)
    counts["selected_fraction_of_eligible_teachers"] = (
        len(selected) / counts["eligible_teachers"] if counts["eligible_teachers"] else None
    )
    for relation in ("better", "tied", "dominated"):
        counts[f"{relation}_rate_among_comparable_teachers"] = (
            counts[f"{relation}_teachers"] / counts["comparable_teachers"]
            if counts["comparable_teachers"] else None
        )
    return {
        "selected_row_ids": sorted(selected), "diagnostics": counts, "groups": groups,
        "cvrp_relative_tolerance": tolerance,
        "selection_mode": selection_mode,
    }
