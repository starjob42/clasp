"""Per-task solution parsing and scoring -- the single source of truth for rewards.

Each of the five supported tasks (``tsp``, ``cvrp``, ``op``, ``mis``, ``jssp``) exposes a
*reward triplet* of three reward heads indexed 0/1/2, published as
``TASK_REWARD_TRIPLETS``, plus its constraint cost components, published as
``TASK_COST_COMPONENTS``. Both are re-exported by :mod:`clasp.rewards.registry`.

Parsing is deliberately strict: a solution that cannot be parsed counts as a
``parse_failure`` constraint violation rather than silently scoring zero, so the
feasibility signal stays aligned with the evaluator.
"""
import ast
import json
import math
import os
import re
from typing import Any


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return ""


def _combined_context(prompt: Any, input_text: str) -> str:
    prompt_text = _as_text(prompt)
    if not input_text:
        return prompt_text
    if input_text in prompt_text:
        return prompt_text
    return f"{prompt_text}\n{input_text}"


def _extract_number(text: str, label: str) -> float | None:
    pattern = rf"{label}\s*:\s*(-?\d+(?:\.\d+)?)"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_bracket_block(text: str, label: str) -> str | None:
    m = re.search(rf"{label}\s*:\s*", text, flags=re.IGNORECASE)
    search_start = m.end() if m else 0
    start = text.find("[", search_start)
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _safe_literal_eval_list(block: str | None) -> list:
    if not block:
        return []
    try:
        value = ast.literal_eval(block)
    except Exception:
        return []
    if isinstance(value, list):
        return value
    return []


def _to_int_list(values: list) -> list[int]:
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return []
    return out


def _to_nested_int_lists(values: list) -> list[list[int]]:
    out: list[list[int]] = []
    for row in values:
        if not isinstance(row, list):
            return []
        cast_row = _to_int_list(row)
        if not cast_row:
            return []
        out.append(cast_row)
    return out


def _distance(coords: list[list[float]], route: list[int]) -> float | None:
    if len(route) < 2 or not coords:
        return None

    n = len(coords)
    total = 0.0
    for a, b in zip(route, route[1:]):
        if a < 0 or b < 0 or a >= n or b >= n:
            return None
        ax, ay = coords[a]
        bx, by = coords[b]
        total += math.hypot(float(ax) - float(bx), float(ay) - float(by))
    return float(total)


def _pair_distance(coords: list[list[float]], a: int, b: int) -> float:
    ax, ay = coords[a]
    bx, by = coords[b]
    return float(math.hypot(float(ax) - float(bx), float(ay) - float(by)))


def _tsp_best_two_opt_improvement_ratio(
    coords: list[list[float]],
    route: list[int],
    route_total: float,
) -> float:
    if route_total <= 0.0 or len(route) < 5 or route[0] != route[-1]:
        return 0.0
    n = len(coords)
    if any(node < 0 or node >= n for node in route):
        return 0.0

    best_gain = 0.0
    # Keep the depot fixed and test all segment reversals between non-depot nodes.
    for i in range(1, len(route) - 2):
        a = route[i - 1]
        b = route[i]
        for k in range(i + 1, len(route) - 1):
            c = route[k]
            d = route[k + 1]
            before = _pair_distance(coords, a, b) + _pair_distance(coords, c, d)
            after = _pair_distance(coords, a, c) + _pair_distance(coords, b, d)
            gain = before - after
            if gain > best_gain:
                best_gain = gain
    if best_gain <= 0.0:
        return 0.0
    return float(best_gain / route_total)


def _tsp_two_opt_improved_route(
    coords: list[list[float]],
    route: list[int],
    max_passes: int = 1,
) -> tuple[list[int], float, float, float, int] | None:
    """Return a bounded two-opt improvement for a feasible closed TSP route."""
    max_passes = max(0, int(max_passes))
    if max_passes <= 0 or len(route) < 5 or route[0] != route[-1]:
        return None

    n = len(coords)
    if any(node < 0 or node >= n for node in route):
        return None

    original_total = _distance(coords, route)
    if original_total is None or original_total <= 0.0:
        return None

    current_route = list(route)
    passes = 0
    for _ in range(max_passes):
        best_gain = 0.0
        best_i = -1
        best_k = -1
        for i in range(1, len(current_route) - 2):
            a = current_route[i - 1]
            b = current_route[i]
            for k in range(i + 1, len(current_route) - 1):
                c = current_route[k]
                d = current_route[k + 1]
                before = _pair_distance(coords, a, b) + _pair_distance(coords, c, d)
                after = _pair_distance(coords, a, c) + _pair_distance(coords, b, d)
                gain = before - after
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_k = k

        if best_gain <= 1e-12 or best_i < 0 or best_k < 0:
            break

        current_route[best_i : best_k + 1] = reversed(current_route[best_i : best_k + 1])
        passes += 1

    if passes <= 0:
        return None

    improved_total = _distance(coords, current_route)
    if improved_total is None or improved_total >= original_total:
        return None

    improvement_ratio = float((original_total - improved_total) / original_total)
    return current_route, float(original_total), float(improved_total), improvement_ratio, passes


def _extract_coords_from_text(input_text: str) -> list[list[float]]:
    coords: dict[int, list[float]] = {}
    for m in re.finditer(
        r"Node\s*(\d+)\s*,\s*coordinates\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        input_text,
        flags=re.IGNORECASE,
    ):
        node = int(m.group(1))
        x = float(m.group(2))
        y = float(m.group(3))
        coords[node] = [x, y]

    if not coords:
        return []

    max_node = max(coords)
    out: list[list[float]] = []
    for i in range(max_node + 1):
        if i not in coords:
            return []
        out.append(coords[i])
    return out


def _get_instance(kwargs: dict, idx: int):
    instances = kwargs.get("instance")
    if not isinstance(instances, list):
        instances = kwargs.get("instance_json")
    if isinstance(instances, list) and idx < len(instances):
        value = instances[idx]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    return None


def _parse_tsp_solution(text: str) -> dict:
    block = _extract_bracket_block(text, "Route")
    route = _to_int_list(_safe_literal_eval_list(block))
    obj = _extract_number(text, "Objective")
    return {"route": route, "declared_objective": obj}


def _tsp_coords_from_instance(instance, input_text: str) -> list[list[float]]:
    if isinstance(instance, list) and instance and isinstance(instance[0], list):
        coords = []
        ok = True
        for row in instance:
            if not (isinstance(row, list) and len(row) >= 2):
                ok = False
                break
            coords.append([float(row[0]), float(row[1])])
        if ok:
            return coords
    return _extract_coords_from_text(input_text)


def _check_tsp_constraints(route: list[int], num_nodes: int) -> tuple[int, int, bool]:
    total = 5
    passed = 0

    if route:
        passed += 1

    if len(route) >= 2 and route[0] == 0 and route[-1] == 0:
        passed += 1

    if route and all(0 <= node < num_nodes for node in route):
        passed += 1

    middle = route[:-1] if len(route) >= 2 and route[-1] == route[0] else route
    if middle and len(middle) == len(set(middle)):
        passed += 1

    if middle and set(middle) == set(range(num_nodes)) and len(route) == num_nodes + 1:
        passed += 1

    return passed, total, passed == total


def tsp_structure_reward(prompts, completions, **kwargs):
    rewards = []
    route_only = _tsp_route_only_output_enabled()
    for completion in completions:
        text = _as_text(completion)
        parsed = _parse_tsp_solution(text)
        has_route_block = _extract_bracket_block(text, "Route") is not None
        has_route = len(parsed["route"]) > 0
        if route_only:
            rewards.append(float((int(has_route_block) + int(has_route)) / 2.0))
            continue
        has_obj = parsed["declared_objective"] is not None
        rewards.append(float((int(has_route_block) + int(has_route) + int(has_obj)) / 3.0))
    return rewards


def tsp_constraints_reward(prompts, completions, **kwargs):
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_tsp_solution(_as_text(completion))
        route = parsed["route"]
        instance = _get_instance(kwargs, i)
        coords = _tsp_coords_from_instance(instance, _as_text(kwargs.get("input", [""])[i] if isinstance(kwargs.get("input"), list) else _as_text(prompt)))
        if not coords:
            rewards.append(0.0)
            continue
        passed, total, _ = _check_tsp_constraints(route, len(coords))
        rewards.append(float(passed / total))
    return rewards


def _objective_gap_slope() -> float:
    raw_value = os.environ.get("CLASP_OBJECTIVE_GAP_SLOPE", "")
    if raw_value:
        try:
            value = float(raw_value)
        except ValueError:
            value = 3.0
        if math.isfinite(value) and value > 0.0:
            return value
    return 3.0


def _objective_quality_mode() -> str:
    raw_value = str(os.environ.get("CLASP_OBJECTIVE_QUALITY_MODE", "") or "").strip().lower()
    if raw_value in {"", "exp", "exponential", "default"}:
        return "exp"
    if raw_value in {"foarl", "foarl_inverse", "inverse", "jiang"}:
        return "foarl_inverse"
    return "exp"


def _declared_objective_match_weight() -> float:
    """Weight assigned to matching a self-declared objective value.

    The declaration is a formatting/consistency signal, not solution quality.
    It is disabled by default so it cannot distort solution quality. Legacy
    ablations can explicitly restore the old 0.1 mixture through the environment.
    """

    raw_value = os.environ.get("CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT", "0.0")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    return float(min(1.0, max(0.0, value)))


def _tsp_route_only_output_enabled() -> bool:
    raw_value = str(os.environ.get("CLASP_TSP_ROUTE_ONLY_OUTPUT", "") or "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _tsp_two_opt_penalty_weight() -> float:
    raw_value = os.environ.get("CLASP_TSP_TWO_OPT_PENALTY", "")
    if raw_value:
        try:
            value = float(raw_value)
        except ValueError:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    return 0.0


def _tsp_two_opt_min_gain_ratio() -> float:
    raw_value = os.environ.get("CLASP_TSP_TWO_OPT_MIN_GAIN", "")
    if raw_value:
        try:
            value = float(raw_value)
        except ValueError:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    return 0.0


def _tsp_edge_overlap_mix() -> float:
    raw_value = os.environ.get("CLASP_TSP_EDGE_OVERLAP_MIX", "")
    if raw_value:
        try:
            value = float(raw_value)
        except ValueError:
            value = 0.0
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    return 0.0


def _tsp_target_gap_bonus_params() -> tuple[float, float]:
    raw_gap = os.environ.get("CLASP_TSP_TARGET_GAP", "")
    raw_bonus = os.environ.get("CLASP_TSP_TARGET_GAP_BONUS", "")
    if not raw_gap or not raw_bonus:
        return 0.0, 0.0
    try:
        target_gap = float(raw_gap)
        bonus = float(raw_bonus)
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(bonus)):
        return 0.0, 0.0
    if target_gap <= 0.0 or bonus <= 0.0:
        return 0.0, 0.0
    return target_gap, bonus


def _tsp_target_gap_excess_penalty_params() -> tuple[float, float]:
    raw_gap = os.environ.get("CLASP_TSP_TARGET_GAP", "")
    raw_penalty = os.environ.get("CLASP_TSP_TARGET_GAP_EXCESS_PENALTY", "")
    if not raw_gap or not raw_penalty:
        return 0.0, 0.0
    try:
        target_gap = float(raw_gap)
        penalty = float(raw_penalty)
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(penalty)):
        return 0.0, 0.0
    if target_gap <= 0.0 or penalty <= 0.0:
        return 0.0, 0.0
    return target_gap, penalty


def _tsp_target_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_TSP_TARGET_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = os.environ.get("CLASP_TSP_TARGET_GAP_CONSTRAINT_GAP", "") or os.environ.get(
        "CLASP_TSP_TARGET_GAP",
        "",
    )
    raw_scale = os.environ.get("CLASP_TSP_TARGET_GAP_CONSTRAINT_SCALE", "")
    try:
        target_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if target_gap <= 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return target_gap, scale


def _tsp_target_tail_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_TSP_TAIL_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = os.environ.get("CLASP_TSP_TAIL_GAP_CONSTRAINT_GAP", "") or os.environ.get(
        "CLASP_TSP_TAIL_GAP",
        "",
    )
    raw_scale = os.environ.get("CLASP_TSP_TAIL_GAP_CONSTRAINT_SCALE", "")
    try:
        tail_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(tail_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if tail_gap <= 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return tail_gap, scale


def _tsp_objective_gap_cost_params() -> tuple[bool, float, float]:
    raw_enable = str(os.environ.get("CLASP_TSP_OBJECTIVE_GAP_COST", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return False, 0.05, 1.0

    raw_cap = os.environ.get("CLASP_TSP_OBJECTIVE_GAP_COST_CAP", "")
    raw_scale = os.environ.get("CLASP_TSP_OBJECTIVE_GAP_COST_SCALE", "")
    try:
        cap = float(raw_cap) if raw_cap else 0.05
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return False, 0.05, 1.0
    if not (math.isfinite(cap) and math.isfinite(scale)):
        return False, 0.05, 1.0
    if cap <= 0.0 or scale <= 0.0:
        return False, 0.05, 1.0
    return True, cap, scale


def _cvrp_target_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_CVRP_TARGET_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = os.environ.get("CLASP_CVRP_TARGET_GAP_CONSTRAINT_GAP", "") or os.environ.get(
        "CLASP_CVRP_TARGET_GAP",
        "",
    )
    raw_scale = os.environ.get("CLASP_CVRP_TARGET_GAP_CONSTRAINT_SCALE", "")
    try:
        target_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if target_gap <= 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return target_gap, scale


def _cvrp_objective_gap_cost_params() -> tuple[bool, float, float]:
    raw_enable = str(os.environ.get("CLASP_CVRP_OBJECTIVE_GAP_COST", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return False, 0.20, 1.0

    raw_cap = os.environ.get("CLASP_CVRP_OBJECTIVE_GAP_COST_CAP", "")
    raw_scale = os.environ.get("CLASP_CVRP_OBJECTIVE_GAP_COST_SCALE", "")
    try:
        cap = float(raw_cap) if raw_cap else 0.20
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return False, 0.20, 1.0
    if not (math.isfinite(cap) and math.isfinite(scale)):
        return False, 0.20, 1.0
    if cap <= 0.0 or scale <= 0.0:
        return False, 0.20, 1.0
    return True, cap, scale


def _op_length_margin_ratio() -> float:
    raw_value = os.environ.get("CLASP_OP_LENGTH_MARGIN_RATIO", "")
    if raw_value:
        try:
            value = float(raw_value)
        except ValueError:
            value = 0.0
        if math.isfinite(value) and 0.0 < value < 1.0:
            return value
    return 0.0


def _op_prize_surplus_constraint_enabled() -> bool:
    raw_value = str(os.environ.get("CLASP_OP_PRIZE_SURPLUS_CONSTRAINT", "") or "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _op_prize_cap_constraint_enabled() -> bool:
    raw_value = str(os.environ.get("CLASP_OP_PRIZE_CAP_CONSTRAINT", "") or "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _mis_conflict_adjusted_partial_params() -> tuple[bool, float]:
    raw_enable = str(os.environ.get("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return False, 2.0

    raw_penalty = os.environ.get("CLASP_MIS_CONFLICT_NODE_PENALTY", "")
    try:
        penalty = float(raw_penalty) if raw_penalty else 2.0
    except ValueError:
        penalty = 2.0
    if not math.isfinite(penalty) or penalty <= 0.0:
        penalty = 2.0
    return True, penalty


def _mis_conflict_cover_partial_enabled() -> bool:
    raw_enable = str(os.environ.get("CLASP_MIS_CONFLICT_COVER_PARTIAL", "") or "").strip().lower()
    return raw_enable in {"1", "true", "yes", "on"}


def _mis_greedy_conflict_cover_size(selected: list[int], edges: set[tuple[int, int]]) -> int:
    selected_set = set(selected)
    remaining_edges = {
        edge
        for edge in edges
        if edge[0] in selected_set and edge[1] in selected_set
    }
    cover_size = 0

    while remaining_edges:
        incident_counts: dict[int, int] = {}
        for a, b in remaining_edges:
            incident_counts[a] = incident_counts.get(a, 0) + 1
            incident_counts[b] = incident_counts.get(b, 0) + 1
        chosen = max(incident_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        cover_size += 1
        remaining_edges = {
            edge
            for edge in remaining_edges
            if chosen not in edge
        }

    return cover_size


def _mis_addable_vertex_count(selected: list[int], num_nodes: int, edges: set[tuple[int, int]]) -> int:
    selected_set = set(selected)
    if not selected_set:
        return 0

    blocked = set(selected_set)
    for a, b in edges:
        if a in selected_set:
            blocked.add(b)
        if b in selected_set:
            blocked.add(a)
    return sum(1 for node in range(num_nodes) if node not in blocked)


def _mis_target_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_MIS_TARGET_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = os.environ.get("CLASP_MIS_TARGET_GAP_CONSTRAINT_GAP", "") or os.environ.get(
        "CLASP_MIS_TARGET_GAP",
        "",
    )
    raw_scale = os.environ.get("CLASP_MIS_TARGET_GAP_CONSTRAINT_SCALE", "")
    try:
        target_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if target_gap < 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return target_gap, scale


def _mis_repair_target_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = (
        os.environ.get("CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT_GAP", "")
        or os.environ.get("CLASP_MIS_REPAIR_TARGET_GAP", "")
        or os.environ.get("CLASP_MIS_TARGET_GAP", "")
    )
    raw_scale = os.environ.get("CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT_SCALE", "")
    try:
        target_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if target_gap < 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return target_gap, scale


def _jssp_target_gap_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_JSSP_TARGET_GAP_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_gap = os.environ.get("CLASP_JSSP_TARGET_GAP_CONSTRAINT_GAP", "") or os.environ.get(
        "CLASP_JSSP_TARGET_GAP",
        "",
    )
    raw_scale = os.environ.get("CLASP_JSSP_TARGET_GAP_CONSTRAINT_SCALE", "")
    try:
        target_gap = float(raw_gap)
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(target_gap) and math.isfinite(scale)):
        return 0.0, 0.0
    if target_gap < 0.0 or scale <= 0.0:
        return 0.0, 0.0
    return target_gap, scale


def _jssp_cycle_fraction_constraint_scale() -> float:
    raw_enable = str(os.environ.get("CLASP_JSSP_CYCLE_FRACTION_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0

    raw_scale = os.environ.get("CLASP_JSSP_CYCLE_FRACTION_SCALE", "")
    try:
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return scale


def _jssp_processed_deficit_constraint_scale() -> float:
    raw_enable = str(os.environ.get("CLASP_JSSP_PROCESSED_DEFICIT_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0

    raw_scale = os.environ.get("CLASP_JSSP_PROCESSED_DEFICIT_SCALE", "")
    try:
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return scale


def _jssp_parse_bad_token_constraint_scale() -> float:
    raw_enable = str(os.environ.get("CLASP_JSSP_PARSE_BAD_TOKEN_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0

    raw_scale = os.environ.get("CLASP_JSSP_PARSE_BAD_TOKEN_SCALE", "")
    try:
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return scale


def _mis_size_excess_constraint_params() -> tuple[float, float]:
    raw_enable = str(os.environ.get("CLASP_MIS_SIZE_EXCESS_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, 0.0

    raw_scale = os.environ.get("CLASP_MIS_SIZE_EXCESS_CONSTRAINT_SCALE", "")
    raw_max_ratio = os.environ.get("CLASP_MIS_SIZE_EXCESS_MAX_REF_RATIO", "")
    try:
        scale = float(raw_scale) if raw_scale else 1.0
        max_ref_ratio = float(raw_max_ratio) if raw_max_ratio else 1.0
    except ValueError:
        return 0.0, 0.0
    if not (math.isfinite(scale) and math.isfinite(max_ref_ratio)):
        return 0.0, 0.0
    if scale <= 0.0 or max_ref_ratio <= 0.0:
        return 0.0, 0.0
    return scale, max_ref_ratio


def _mis_conflict_node_constraint_scale() -> float:
    raw_enable = str(os.environ.get("CLASP_MIS_CONFLICT_NODE_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0

    raw_scale = os.environ.get("CLASP_MIS_CONFLICT_NODE_CONSTRAINT_SCALE", "")
    try:
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return scale


def _mis_maximality_constraint_params() -> tuple[float, bool]:
    raw_enable = str(os.environ.get("CLASP_MIS_MAXIMALITY_CONSTRAINT", "") or "").strip().lower()
    if raw_enable not in {"1", "true", "yes", "on"}:
        return 0.0, True

    raw_scale = os.environ.get("CLASP_MIS_MAXIMALITY_CONSTRAINT_SCALE", "")
    raw_independent_only = str(
        os.environ.get("CLASP_MIS_MAXIMALITY_INDEPENDENT_ONLY", "1") or "1"
    ).strip().lower()
    try:
        scale = float(raw_scale) if raw_scale else 1.0
    except ValueError:
        return 0.0, True
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0, True
    independent_only = raw_independent_only not in {"0", "false", "no", "off"}
    return scale, independent_only


def _tsp_undirected_edges(route: list[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in zip(route, route[1:]):
        edge = (a, b) if a <= b else (b, a)
        edges.add(edge)
    return edges


def _tsp_edge_overlap_fraction(pred_route: list[int], ref_route: list[int]) -> float:
    ref_edges = _tsp_undirected_edges(ref_route)
    if not ref_edges:
        return 0.0
    pred_edges = _tsp_undirected_edges(pred_route)
    return float(len(pred_edges & ref_edges) / len(ref_edges))


def _objective_quality_from_signed_gap(signed_gap: float) -> float:
    if not math.isfinite(signed_gap):
        return 0.0
    if _objective_quality_mode() == "foarl_inverse":
        if signed_gap >= 0.0:
            return 1.0 / (1.0 + signed_gap)
        improvement = -signed_gap
        return 1.0 + (0.5 * math.tanh(2.0 * improvement))
    if signed_gap >= 0.0:
        return math.exp(-_objective_gap_slope() * signed_gap)
    improvement = -signed_gap
    return 1.0 + (0.5 * math.tanh(2.0 * improvement))


def _objective_reward_from_signed_gap(
    signed_gap: float,
    declared_match: float,
    *,
    apply_constraint_gate: bool,
    constraints_fraction: float,
) -> float:
    declared_weight = _declared_objective_match_weight()
    reward = ((1.0 - declared_weight) * _objective_quality_from_signed_gap(signed_gap)) + (
        declared_weight * float(declared_match)
    )
    if apply_constraint_gate:
        reward *= constraints_fraction
    return float(2.0 * reward)


def _objective_reward_if_feasible(
    *,
    feasible: bool,
    signed_gap: float,
    declared_match: float,
    apply_constraint_gate: bool,
    constraints_fraction: float,
    allow_infeasible_partial_credit: bool = False,
) -> float:
    """Objective quality is only meaningful on task-valid candidates.

    We deliberately return zero on infeasible completions so the primal term
    cannot over-reward shortcuts such as missing customers, oversized sets,
    or underspecified routes. Feasibility pressure is then handled explicitly
    by the dense constrained objective.
    """

    if not feasible:
        if not allow_infeasible_partial_credit:
            return 0.0
        # Dense partial credit helps the policy climb toward feasibility instead
        # of receiving only zeros when it is close but not fully valid yet.
        return _objective_reward_from_signed_gap(
            signed_gap,
            declared_match,
            apply_constraint_gate=True,
            constraints_fraction=constraints_fraction,
        )
    return _objective_reward_from_signed_gap(
        signed_gap,
        declared_match,
        apply_constraint_gate=apply_constraint_gate,
        constraints_fraction=constraints_fraction,
    )


def _grpo_graded_reward_enabled() -> bool:
    """Opt in independently of CLASP; an absent flag preserves legacy rewards."""
    value = os.environ.get("GRPO_GRADED_REWARD", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("GRPO_GRADED_REWARD must be 0 or 1")
    return value == "1"


def _grpo_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _grpo_invalid_reward(severity: float) -> float:
    if not math.isfinite(severity) or severity <= 0.0:
        raise ValueError("An invalid parsed solution must have finite positive violation severity")
    return -severity / (1.0 + severity)


def _validate_grpo_cvrp_metadata(
    coords: list[list[float]],
    demands: list[float],
    capacity: float,
) -> None:
    if not isinstance(coords, (list, tuple)) or len(coords) < 2 or any(
        not isinstance(point, (list, tuple)) or len(point) != 2
        or not all(_grpo_finite_number(v) for v in point) for point in coords
    ):
        raise ValueError("Graded CVRP reward requires finite 2-D coordinates and at least one customer")
    if not isinstance(demands, (list, tuple)) or len(demands) != len(coords) or any(
        not _grpo_finite_number(v) or v < 0 for v in demands
    ):
        raise ValueError("Graded CVRP reward requires one finite nonnegative demand per node")
    if not _grpo_finite_number(capacity) or capacity <= 0:
        raise ValueError("Graded CVRP reward requires finite positive capacity")


def grpo_graded_cvrp_reward(
    routes: list[list[int]],
    coords: list[list[float]],
    demands: list[float],
    capacity: float,
    reference_cost: float | None,
) -> float:
    """Reward strict feasibility and distance, grading only defects when invalid.

    Feasible reward is ``1 + reference_cost / (reference_cost + distance)``.
    Invalid reward is ``-V/(1+V)``; an empty/unparseable solution receives -1.
    V is (missing + duplicate + invalid nodes + bad route boundaries)/(n-1),
    plus total capacity overflow/max(total customer demand, capacity).
    Denominators depend only on the instance, preventing output-length dilution.
    The same hard predicates as the evaluator decide feasibility. Disjoint
    candidate rewards do not imply population-level lexicographic optimization.
    """
    _validate_grpo_cvrp_metadata(coords, demands, capacity)
    if not routes:
        return -1.0
    if any(not isinstance(route, (list, tuple)) or not route
           or any(type(v) is not int for v in route) for route in routes):
        return -1.0

    _, _, feasible = _check_cvrp_constraints(routes, coords, demands, capacity)
    if feasible:
        if not _grpo_finite_number(reference_cost) or reference_cost <= 0:
            raise ValueError("A feasible graded CVRP reward requires a finite positive reference distance")
        distance = _cvrp_total_distance(coords, routes)
        if distance is None or not math.isfinite(distance) or distance < 0:
            raise ValueError("A feasible graded CVRP route has an invalid recomputed distance")
        # Divide first to avoid overflow in reference_cost + distance.
        quality = 1.0 / (1.0 + distance / reference_cost)
        return 1.0 + quality

    n = len(coords)
    customers = [v for route in routes for v in route if 0 < v < n]
    missing = n - 1 - len(set(customers))
    duplicate = len(customers) - len(set(customers))
    invalid = sum(not 0 <= v < n for route in routes for v in route)
    boundaries = sum(len(route) < 2 or route[0] != 0 or route[-1] != 0 for route in routes)
    overflow = sum(max(0.0, sum(demands[v] for v in route if 0 < v < n) - capacity)
                   for route in routes)
    severity = (missing + duplicate + invalid + boundaries) / (n - 1)
    severity += overflow / max(sum(demands[1:]), capacity)
    return _grpo_invalid_reward(float(severity))


def grpo_graded_mis_reward(
    node_set: list[int],
    num_nodes: int,
    edges: set[tuple[int, int]],
) -> float:
    """Return 1+|S|/n when independent; otherwise grade only graph defects.

    V=(invalid IDs + duplicate IDs + selected conflicting edges)/n uses a
    fixed instance denominator. Adding conflict-free vertices to an invalid
    set cannot increase its reward by diluting an existing conflict. Equal
    feasible cardinalities remain tied. No reference optimum is required.
    """
    if type(num_nodes) is not int or num_nodes <= 0:
        raise ValueError("Graded MIS reward requires a positive integer node count")
    if not isinstance(edges, (set, frozenset)) or any(
        not isinstance(edge, tuple) or len(edge) != 2
        or any(type(v) is not int or not 0 <= v < num_nodes for v in edge)
        or edge[0] >= edge[1] for edge in edges
    ):
        raise ValueError("Graded MIS reward requires canonical in-bounds edges of a simple graph")
    if not node_set or any(type(v) is not int for v in node_set):
        return -1.0
    _, _, feasible = _check_mis_constraints(node_set, num_nodes, edges)
    if feasible:
        return 1.0 + len(node_set) / num_nodes
    selected = set(node_set)
    invalid = sum(not 0 <= v < num_nodes for v in node_set)
    duplicate = len(node_set) - len(selected)
    conflicts = sum(u in selected and v in selected for u, v in edges)
    return _grpo_invalid_reward((invalid + duplicate + conflicts) / num_nodes)


def tsp_objective_reward(prompts, completions, **kwargs):
    rewards = []
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    apply_constraint_gate = bool(kwargs.get("apply_constraint_gate", True))
    allow_infeasible_partial_credit = bool(kwargs.get("allow_infeasible_partial_credit", False))
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion, gt) in enumerate(zip(prompts, completions, gt_outputs)):
        if gt is None:
            rewards.append(0.0)
            continue

        pred = _parse_tsp_solution(_as_text(completion))
        ref = _parse_tsp_solution(_as_text(gt))

        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords = _tsp_coords_from_instance(_get_instance(kwargs, i), input_text)
        if not coords:
            rewards.append(0.0)
            continue

        pred_total = _distance(coords, pred["route"])
        ref_total = _distance(coords, ref["route"])

        if ref_total is None or pred_total is None or ref_total <= 0.0 or pred_total <= 0.0:
            rewards.append(0.0)
            continue

        passed, total, feasible = _check_tsp_constraints(pred["route"], len(coords))
        constraints_fraction = passed / total

        signed_gap = (pred_total - ref_total) / ref_total

        declared = pred["declared_objective"]
        declared_match = 0.0
        if declared is not None:
            tol = max(1.0, 0.01 * pred_total)
            declared_match = 1.0 if abs(declared - pred_total) <= tol else 0.0

        reward = _objective_reward_if_feasible(
            feasible=feasible,
            signed_gap=signed_gap,
            declared_match=declared_match,
            apply_constraint_gate=apply_constraint_gate,
            constraints_fraction=constraints_fraction,
            allow_infeasible_partial_credit=allow_infeasible_partial_credit,
        )
        edge_overlap_mix = _tsp_edge_overlap_mix()
        if feasible and edge_overlap_mix > 0.0:
            edge_overlap = _tsp_edge_overlap_fraction(pred["route"], ref["route"])
            edge_overlap_reward = 2.0 * edge_overlap
            reward = ((1.0 - edge_overlap_mix) * reward) + (edge_overlap_mix * edge_overlap_reward)

        target_gap, target_bonus = _tsp_target_gap_bonus_params()
        if feasible and target_gap > 0.0 and target_bonus > 0.0:
            nonnegative_gap = max(0.0, signed_gap)
            if nonnegative_gap <= target_gap:
                closeness = (target_gap - nonnegative_gap) / target_gap
                reward += target_bonus * (1.0 + closeness)

        excess_target_gap, excess_penalty = _tsp_target_gap_excess_penalty_params()
        if feasible and excess_target_gap > 0.0 and excess_penalty > 0.0:
            nonnegative_gap = max(0.0, signed_gap)
            excess_ratio = (nonnegative_gap - excess_target_gap) / excess_target_gap
            if excess_ratio > 0.0:
                reward = max(0.0, reward - (excess_penalty * excess_ratio))

        two_opt_penalty = _tsp_two_opt_penalty_weight()
        if feasible and two_opt_penalty > 0.0:
            local_improvement = _tsp_best_two_opt_improvement_ratio(
                coords,
                pred["route"],
                pred_total,
            )
            excess_improvement = local_improvement - _tsp_two_opt_min_gain_ratio()
            if excess_improvement > 0.0:
                reward = max(0.0, reward - (two_opt_penalty * excess_improvement))

        rewards.append(reward)

    return rewards


def _parse_cvrp_solution(text: str) -> dict:
    block = _extract_bracket_block(text, "Routes")
    routes = _to_nested_int_lists(_safe_literal_eval_list(block))
    obj = _extract_number(text, "Objective")
    return {"routes": routes, "declared_objective": obj}


def _cvrp_meta(instance, input_text: str) -> tuple[list[list[float]], list[float], float | None]:
    if (
        isinstance(instance, (list, tuple))
        and len(instance) >= 3
        and isinstance(instance[0], (list, tuple))
        and isinstance(instance[1], (list, tuple))
    ):
        coords = []
        for row in instance[0]:
            if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                coords = []
                break
            coords.append([float(row[0]), float(row[1])])

        demands = []
        for d in instance[1]:
            try:
                demands.append(float(d))
            except (TypeError, ValueError):
                demands = []
                break

        capacity = None
        try:
            capacity = float(instance[2])
        except (TypeError, ValueError):
            capacity = None

        if coords and demands and len(coords) == len(demands):
            return coords, demands, capacity

    coords = _extract_coords_from_text(input_text)
    demand_map = {}
    for m in re.finditer(r"Node\s*(\d+).*?demand\s*:\s*(-?\d+(?:\.\d+)?)", input_text, flags=re.IGNORECASE):
        demand_map[int(m.group(1))] = float(m.group(2))

    if coords and demand_map and len(demand_map) == len(coords):
        demands = [demand_map.get(i, 0.0) for i in range(len(coords))]
    else:
        demands = []

    cap = None
    m = re.search(r"capacity\s*(?:of|=)?\s*(-?\d+(?:\.\d+)?)", input_text, flags=re.IGNORECASE)
    if m:
        cap = float(m.group(1))
    return coords, demands, cap


def _check_cvrp_constraints(routes: list[list[int]], coords: list[list[float]], demands: list[float], capacity: float | None):
    total = 6
    passed = 0

    if routes:
        passed += 1

    if routes and all(len(r) >= 2 and r[0] == 0 and r[-1] == 0 for r in routes):
        passed += 1

    n = len(coords)
    flat = [node for r in routes for node in r]
    if routes and all(0 <= node < n for node in flat):
        passed += 1

    customers = [node for r in routes for node in r if node != 0]
    if customers and len(customers) == len(set(customers)):
        passed += 1

    expected = set(range(1, n))
    if customers and set(customers) == expected:
        passed += 1

    if capacity is not None and demands and routes:
        cap_ok = True
        for r in routes:
            route_demand = 0.0
            for node in r:
                if node == 0:
                    continue
                if node < 0 or node >= len(demands):
                    cap_ok = False
                    break
                route_demand += demands[node]
            if not cap_ok:
                break
            if route_demand > capacity + 1e-9:
                cap_ok = False
                break
        if cap_ok:
            passed += 1

    return passed, total, passed == total


def _cvrp_total_distance(coords: list[list[float]], routes: list[list[int]]) -> float | None:
    if not routes:
        return None
    total = 0.0
    for route in routes:
        d = _distance(coords, route)
        if d is None:
            return None
        total += d
    return total


def _cvrp_local_search_improved_routes(
    coords: list[list[float]],
    demands: list[float],
    capacity: float | None,
    routes: list[list[int]],
    max_passes: int = 2,
) -> tuple[list[list[int]], float, float, float, int] | None:
    """Bounded capacity-preserving local search for feasible CVRP routes.

    Applies the best-improvement move per pass among intra-route 2-opt,
    inter-route customer relocation, and inter-route customer exchange.
    Emptied routes are dropped. Returns
    (routes, original_total, improved_total, improvement_ratio, passes)
    or None when no strict improvement exists.
    """
    max_passes = max(0, int(max_passes))
    if max_passes <= 0 or not routes:
        return None
    n = len(coords)
    if len(demands) != n:
        return None
    current: list[list[int]] = []
    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return None
        if any(node < 0 or node >= n for node in route):
            return None
        if any(node == 0 for node in route[1:-1]):
            return None
        current.append(list(route))

    original_total = _cvrp_total_distance(coords, current)
    if original_total is None or original_total <= 0.0:
        return None

    loads = [sum(float(demands[node]) for node in route[1:-1]) for route in current]
    if capacity is not None and any(load > float(capacity) + 1e-9 for load in loads):
        return None

    passes = 0
    for _ in range(max_passes):
        best_gain = 1e-9
        best_move: tuple | None = None

        # Intra-route 2-opt with the depot fixed at both ends.
        for r_idx, route in enumerate(current):
            if len(route) < 5:
                continue
            for i in range(1, len(route) - 2):
                a = route[i - 1]
                b = route[i]
                for k in range(i + 1, len(route) - 1):
                    c = route[k]
                    d = route[k + 1]
                    gain = (
                        _pair_distance(coords, a, b)
                        + _pair_distance(coords, c, d)
                        - _pair_distance(coords, a, c)
                        - _pair_distance(coords, b, d)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ("two_opt", r_idx, i, k)

        # Inter-route relocation of a single customer.
        for ra, route_a in enumerate(current):
            for i in range(1, len(route_a) - 1):
                cust = route_a[i]
                removal_gain = (
                    _pair_distance(coords, route_a[i - 1], cust)
                    + _pair_distance(coords, cust, route_a[i + 1])
                    - _pair_distance(coords, route_a[i - 1], route_a[i + 1])
                )
                for rb, route_b in enumerate(current):
                    if rb == ra:
                        continue
                    if (
                        capacity is not None
                        and loads[rb] + float(demands[cust]) > float(capacity) + 1e-9
                    ):
                        continue
                    for j in range(len(route_b) - 1):
                        insertion_cost = (
                            _pair_distance(coords, route_b[j], cust)
                            + _pair_distance(coords, cust, route_b[j + 1])
                            - _pair_distance(coords, route_b[j], route_b[j + 1])
                        )
                        gain = removal_gain - insertion_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ("relocate", ra, i, rb, j)

        # Inter-route exchange of two customers.
        for ra in range(len(current)):
            route_a = current[ra]
            for rb in range(ra + 1, len(current)):
                route_b = current[rb]
                for i in range(1, len(route_a) - 1):
                    c1 = route_a[i]
                    for j in range(1, len(route_b) - 1):
                        c2 = route_b[j]
                        if capacity is not None:
                            if (
                                loads[ra] - float(demands[c1]) + float(demands[c2])
                                > float(capacity) + 1e-9
                            ):
                                continue
                            if (
                                loads[rb] - float(demands[c2]) + float(demands[c1])
                                > float(capacity) + 1e-9
                            ):
                                continue
                        gain = (
                            _pair_distance(coords, route_a[i - 1], c1)
                            + _pair_distance(coords, c1, route_a[i + 1])
                            + _pair_distance(coords, route_b[j - 1], c2)
                            + _pair_distance(coords, c2, route_b[j + 1])
                            - _pair_distance(coords, route_a[i - 1], c2)
                            - _pair_distance(coords, c2, route_a[i + 1])
                            - _pair_distance(coords, route_b[j - 1], c1)
                            - _pair_distance(coords, c1, route_b[j + 1])
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ("swap", ra, i, rb, j)

        if best_move is None:
            break
        if best_move[0] == "two_opt":
            _, r_idx, i, k = best_move
            route = current[r_idx]
            route[i : k + 1] = reversed(route[i : k + 1])
        elif best_move[0] == "relocate":
            _, ra, i, rb, j = best_move
            cust = current[ra].pop(i)
            current[rb].insert(j + 1, cust)
            loads[ra] -= float(demands[cust])
            loads[rb] += float(demands[cust])
            if len(current[ra]) == 2:
                current.pop(ra)
                loads.pop(ra)
        else:
            _, ra, i, rb, j = best_move
            c1 = current[ra][i]
            c2 = current[rb][j]
            current[ra][i] = c2
            current[rb][j] = c1
            loads[ra] += float(demands[c2]) - float(demands[c1])
            loads[rb] += float(demands[c1]) - float(demands[c2])
        passes += 1

    if passes <= 0:
        return None
    improved_total = _cvrp_total_distance(coords, current)
    if improved_total is None or improved_total >= original_total:
        return None
    improvement_ratio = float((original_total - improved_total) / original_total)
    return current, float(original_total), float(improved_total), improvement_ratio, passes


def _format_cvrp_distillation_target(routes: list[list[int]], objective: float) -> str:
    routes_text = ", ".join(
        "[" + ", ".join(str(int(node)) for node in route) + "]" for route in routes
    )
    return f"Routes: [{routes_text}], Objective: {objective:.6f}"


def cvrp_structure_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _as_text(completion)
        parsed = _parse_cvrp_solution(text)
        routes = parsed["routes"]
        has_routes_block = _extract_bracket_block(text, "Routes") is not None
        has_routes = len(routes) > 0
        has_obj = parsed["declared_objective"] is not None
        rewards.append(float((int(has_routes_block) + int(has_routes) + int(has_obj)) / 3.0))
    return rewards


def cvrp_constraints_reward(prompts, completions, **kwargs):
    rewards = []
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_cvrp_solution(_as_text(completion))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, demands, capacity = _cvrp_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))
        if not coords or not demands:
            rewards.append(0.0)
            continue
        passed, total, _ = _check_cvrp_constraints(parsed["routes"], coords, demands, capacity)
        rewards.append(float(passed / total))
    return rewards


def cvrp_objective_reward(prompts, completions, **kwargs):
    if _grpo_graded_reward_enabled():
        return _grpo_graded_cvrp_objective_reward(prompts, completions, **kwargs)
    rewards = []
    gt_outputs = kwargs.get("answer", [None] * len(completions))
    apply_constraint_gate = bool(kwargs.get("apply_constraint_gate", True))
    allow_infeasible_partial_credit = bool(kwargs.get("allow_infeasible_partial_credit", False))
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion, gt) in enumerate(zip(prompts, completions, gt_outputs)):
        if gt is None:
            rewards.append(0.0)
            continue

        pred = _parse_cvrp_solution(_as_text(completion))
        ref = _parse_cvrp_solution(_as_text(gt))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, demands, capacity = _cvrp_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))

        if not coords or not demands or capacity is None:
            rewards.append(0.0)
            continue

        pred_total = _cvrp_total_distance(coords, pred["routes"])
        ref_total = _cvrp_total_distance(coords, ref["routes"])

        if ref_total is None or pred_total is None or ref_total <= 0.0 or pred_total <= 0.0:
            rewards.append(0.0)
            continue

        passed, total, feasible = _check_cvrp_constraints(pred["routes"], coords, demands, capacity)
        constraints_fraction = passed / total

        signed_gap = (pred_total - ref_total) / ref_total

        declared = pred["declared_objective"]
        declared_match = 0.0
        if declared is not None:
            tol = max(1.0, 0.01 * pred_total)
            declared_match = 1.0 if abs(declared - pred_total) <= tol else 0.0

        rewards.append(
            _objective_reward_if_feasible(
                feasible=feasible,
                signed_gap=signed_gap,
                declared_match=declared_match,
                apply_constraint_gate=apply_constraint_gate,
                constraints_fraction=constraints_fraction,
                allow_infeasible_partial_credit=allow_infeasible_partial_credit,
            )
        )

    return rewards


def _grpo_graded_cvrp_objective_reward(prompts, completions, **kwargs):
    if len(prompts) != len(completions):
        raise ValueError("Graded CVRP prompt/completion counts differ")
    answers = kwargs.get("answer", [None] * len(completions))
    if not isinstance(answers, (list, tuple)) or len(answers) != len(completions):
        raise ValueError("Graded CVRP requires one reference entry per completion")
    inputs = kwargs.get("input")
    if inputs is not None and (not isinstance(inputs, list) or len(inputs) != len(completions)):
        raise ValueError("Graded CVRP input/completion counts differ")
    rewards = []
    for i, (prompt, completion, answer) in enumerate(zip(prompts, completions, answers)):
        instance = _get_instance(kwargs, i)
        context = _combined_context(prompt, _as_text(inputs[i]) if inputs is not None else "")
        if instance is None:
            coords, demands, capacity = _cvrp_meta(None, context)
        else:
            if not isinstance(instance, (list, tuple)) or len(instance) < 3:
                raise ValueError("Graded CVRP requires [coordinates, demands, capacity] metadata")
            coords, demands, capacity = instance[:3]
        _validate_grpo_cvrp_metadata(coords, demands, capacity)
        routes = _parse_cvrp_solution(_as_text(completion))["routes"]
        reference_cost = None
        if _check_cvrp_constraints(routes, coords, demands, capacity)[2]:
            reference_routes = _parse_cvrp_solution(_as_text(answer))["routes"]
            if not _check_cvrp_constraints(reference_routes, coords, demands, capacity)[2]:
                raise ValueError("A feasible graded CVRP reward requires a feasible reference route set")
            reference_cost = _cvrp_total_distance(coords, reference_routes)
        rewards.append(grpo_graded_cvrp_reward(routes, coords, demands, capacity, reference_cost))
    return rewards


def _parse_op_solution(text: str) -> dict:
    block = _extract_bracket_block(text, "Route")
    route = _to_int_list(_safe_literal_eval_list(block))
    obj = _extract_number(text, "Objective")
    return {"route": route, "declared_objective": obj}


def _op_meta(instance, input_text: str) -> tuple[list[list[float]], list[float], float | None]:
    if (
        isinstance(instance, list)
        and len(instance) >= 3
        and isinstance(instance[0], list)
        and isinstance(instance[1], list)
    ):
        coords = []
        for row in instance[0]:
            if not (isinstance(row, list) and len(row) >= 2):
                coords = []
                break
            coords.append([float(row[0]), float(row[1])])

        prizes = []
        for p in instance[1]:
            try:
                prizes.append(float(p))
            except (TypeError, ValueError):
                prizes = []
                break

        max_len = None
        try:
            max_len = float(instance[2])
        except (TypeError, ValueError):
            max_len = None

        if coords and prizes and len(coords) == len(prizes):
            return coords, prizes, max_len

    coords = _extract_coords_from_text(input_text)
    prize_map = {}
    for m in re.finditer(r"Node\s*(\d+).*?prize\s*:\s*(-?\d+(?:\.\d+)?)", input_text, flags=re.IGNORECASE):
        prize_map[int(m.group(1))] = float(m.group(2))

    if coords and prize_map and len(prize_map) == len(coords):
        prizes = [prize_map.get(i, 0.0) for i in range(len(coords))]
    else:
        prizes = []

    max_len = None
    m = re.search(r"T\s*=\s*(-?\d+(?:\.\d+)?)", input_text)
    if m:
        max_len = float(m.group(1))
    return coords, prizes, max_len


def _check_op_constraints(route: list[int], coords: list[list[float]], max_len: float | None):
    total = 5
    passed = 0

    if route:
        passed += 1

    if route and route[0] == 0:
        passed += 1

    n = len(coords)
    if route and all(0 <= node < n for node in route):
        passed += 1

    if route and len(route) == len(set(route)):
        passed += 1

    route_len = _distance(coords, route)
    if route_len is not None and max_len is not None and route_len <= max_len + 1e-9:
        passed += 1

    return passed, total, passed == total


def _op_prize(route: list[int], prizes: list[float]) -> float | None:
    if not route or not prizes:
        return None
    n = len(prizes)
    if any(node < 0 or node >= n for node in route):
        return None
    unique_nodes = list(dict.fromkeys(route))
    total = 0.0
    for node in unique_nodes:
        total += float(prizes[node])
    return total


def op_structure_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _as_text(completion)
        parsed = _parse_op_solution(text)
        route = parsed["route"]
        has_route_block = _extract_bracket_block(text, "Route") is not None
        has_route = len(route) > 0
        has_obj = parsed["declared_objective"] is not None
        rewards.append(float((int(has_route_block) + int(has_route) + int(has_obj)) / 3.0))
    return rewards


def op_constraints_reward(prompts, completions, **kwargs):
    rewards = []
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_op_solution(_as_text(completion))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, _, max_len = _op_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))
        if not coords or max_len is None:
            rewards.append(0.0)
            continue
        passed, total, _ = _check_op_constraints(parsed["route"], coords, max_len)
        rewards.append(float(passed / total))
    return rewards


def op_objective_reward(prompts, completions, **kwargs):
    rewards = []
    gt_outputs = kwargs.get("answer", [None] * len(completions))
    apply_constraint_gate = bool(kwargs.get("apply_constraint_gate", True))
    allow_infeasible_partial_credit = bool(kwargs.get("allow_infeasible_partial_credit", False))
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion, gt) in enumerate(zip(prompts, completions, gt_outputs)):
        if gt is None:
            rewards.append(0.0)
            continue

        pred = _parse_op_solution(_as_text(completion))
        ref = _parse_op_solution(_as_text(gt))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, prizes, max_len = _op_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))

        if not coords or not prizes or max_len is None:
            rewards.append(0.0)
            continue

        pred_prize = _op_prize(pred["route"], prizes)
        ref_prize = _op_prize(ref["route"], prizes)

        if ref_prize is None or pred_prize is None or ref_prize <= 0.0:
            rewards.append(0.0)
            continue

        passed, total, feasible = _check_op_constraints(pred["route"], coords, max_len)
        constraints_fraction = passed / total

        signed_gap = (ref_prize - pred_prize) / ref_prize

        declared = pred["declared_objective"]
        declared_match = 0.0
        if declared is not None:
            tol = max(1.0, 0.01 * max(pred_prize, 1.0))
            declared_match = 1.0 if abs(declared - pred_prize) <= tol else 0.0

        rewards.append(
            _objective_reward_if_feasible(
                feasible=feasible,
                signed_gap=signed_gap,
                declared_match=declared_match,
                apply_constraint_gate=apply_constraint_gate,
                constraints_fraction=constraints_fraction,
                allow_infeasible_partial_credit=allow_infeasible_partial_credit,
            )
        )

    return rewards


def _parse_mis_solution(text: str) -> dict:
    block = _extract_bracket_block(text, "Set")
    node_set = _to_int_list(_safe_literal_eval_list(block))
    obj = _extract_number(text, "Objective")
    return {"node_set": node_set, "declared_objective": obj}


def _mis_meta(instance, input_text: str) -> tuple[int | None, set[tuple[int, int]]]:
    if isinstance(instance, dict) and "edges" in instance:
        num_nodes = None
        try:
            num_nodes = int(instance.get("num_nodes"))
        except (TypeError, ValueError):
            num_nodes = None

        edges = set()
        raw_edges = instance.get("edges", [])
        if isinstance(raw_edges, list):
            for e in raw_edges:
                if isinstance(e, list) and len(e) >= 2:
                    a = int(e[0])
                    b = int(e[1])
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return num_nodes, edges

    m_nodes = re.search(r"graph with\s+(\d+)\s+nodes", input_text, flags=re.IGNORECASE)
    num_nodes = int(m_nodes.group(1)) if m_nodes else None

    edges = set()
    for m in re.finditer(r"\((\d+)\s*,\s*(\d+)\)", input_text):
        a = int(m.group(1))
        b = int(m.group(2))
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return num_nodes, edges


def _check_mis_constraints(node_set: list[int], num_nodes: int | None, edges: set[tuple[int, int]]):
    total = 4
    passed = 0

    if node_set:
        passed += 1

    if node_set and num_nodes is not None and all(0 <= n < num_nodes for n in node_set):
        passed += 1

    if node_set and len(node_set) == len(set(node_set)):
        passed += 1

    selected = set(node_set)
    if not selected:
        independent = False
    else:
        independent = True
        for a, b in edges:
            if a in selected and b in selected:
                independent = False
                break

    if independent:
        passed += 1

    return passed, total, passed == total


def _mis_maximalized_set(
    node_set: list[int],
    num_nodes: int | None,
    edges: set[tuple[int, int]],
) -> tuple[list[int], float, float, float, int] | None:
    """Greedily extend a feasible independent set to maximality.

    Addable vertices (no neighbor in the set) are inserted in ascending
    (degree, id) order; every addition keeps the set independent, so the
    result is a strictly larger feasible solution. Returns
    (sorted_set, original_size, improved_size, added_fraction, added_count)
    or None when the input set is invalid or already maximal.
    """
    if num_nodes is None or num_nodes <= 0 or not node_set:
        return None
    selected = {int(v) for v in node_set}
    if len(selected) != len(node_set):
        return None
    if any(v < 0 or v >= num_nodes for v in selected):
        return None

    neighbors: dict[int, set[int]] = {}
    for a, b in edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    for v in selected:
        if neighbors.get(v, set()) & selected:
            return None

    blocked = set(selected)
    for v in selected:
        blocked |= neighbors.get(v, set())
    candidates = sorted(
        (v for v in range(num_nodes) if v not in blocked),
        key=lambda v: (len(neighbors.get(v, set())), v),
    )

    added = 0
    while candidates:
        v = candidates.pop(0)
        selected.add(v)
        added += 1
        conflict = neighbors.get(v, set())
        candidates = [u for u in candidates if u not in conflict]

    if added <= 0:
        return None
    original_size = len(node_set)
    improved_size = len(selected)
    added_fraction = float((improved_size - original_size) / improved_size)
    return sorted(selected), float(original_size), float(improved_size), added_fraction, added


def _format_mis_distillation_target(node_set: list[int]) -> str:
    nodes_text = ", ".join(str(int(v)) for v in sorted(node_set))
    return f"Set: [{nodes_text}], Objective: {len(node_set)}"


def mis_structure_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _as_text(completion)
        parsed = _parse_mis_solution(text)
        has_set_block = _extract_bracket_block(text, "Set") is not None
        has_set = len(parsed["node_set"]) > 0
        has_obj = parsed["declared_objective"] is not None
        rewards.append(float((int(has_set_block) + int(has_set) + int(has_obj)) / 3.0))
    return rewards


def mis_constraints_reward(prompts, completions, **kwargs):
    rewards = []
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_mis_solution(_as_text(completion))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        num_nodes, edges = _mis_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))
        passed, total, _ = _check_mis_constraints(parsed["node_set"], num_nodes, edges)
        rewards.append(float(passed / total))
    return rewards


def mis_objective_reward(prompts, completions, **kwargs):
    if _grpo_graded_reward_enabled():
        return _grpo_graded_mis_objective_reward(prompts, completions, **kwargs)
    rewards = []
    gt_outputs = kwargs.get("answer", [None] * len(completions))
    apply_constraint_gate = bool(kwargs.get("apply_constraint_gate", True))
    allow_infeasible_partial_credit = bool(kwargs.get("allow_infeasible_partial_credit", False))
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion, gt) in enumerate(zip(prompts, completions, gt_outputs)):
        if gt is None:
            rewards.append(0.0)
            continue

        pred = _parse_mis_solution(_as_text(completion))
        ref = _parse_mis_solution(_as_text(gt))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        num_nodes, edges = _mis_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))

        if num_nodes is None:
            rewards.append(0.0)
            continue

        node_set = pred["node_set"]
        pred_size = float(len(node_set))
        if not ref["node_set"]:
            rewards.append(0.0)
            continue
        ref_size = float(len(ref["node_set"]))
        if ref_size <= 0.0:
            rewards.append(0.0)
            continue

        passed, total, feasible = _check_mis_constraints(node_set, num_nodes, edges)
        constraints_fraction = passed / total

        signed_gap = (ref_size - pred_size) / ref_size

        declared = pred["declared_objective"]
        declared_match = 1.0 if declared is not None and abs(declared - pred_size) < 1e-6 else 0.0

        use_conflict_cover_partial = _mis_conflict_cover_partial_enabled()
        adjust_partial, conflict_penalty = _mis_conflict_adjusted_partial_params()
        if allow_infeasible_partial_credit and (use_conflict_cover_partial or adjust_partial) and not feasible:
            duplicate_nodes = len(node_set) - len(set(node_set))
            if num_nodes is not None:
                invalid_nodes = sum(1 for node in node_set if node < 0 or node >= num_nodes)
                selected = list(dict.fromkeys(node for node in node_set if 0 <= node < num_nodes))
            else:
                invalid_nodes = 0
                selected = list(dict.fromkeys(node_set))
            if use_conflict_cover_partial:
                conflict_penalty_term = float(_mis_greedy_conflict_cover_size(selected, edges))
            else:
                selected_set = set(selected)
                conflicting_edges = sum(1 for edge in edges if edge[0] in selected_set and edge[1] in selected_set)
                conflict_penalty_term = conflict_penalty * float(conflicting_edges)
            effective_size = max(
                0.0,
                float(len(selected))
                - float(invalid_nodes)
                - float(duplicate_nodes)
                - conflict_penalty_term,
            )
            signed_gap = (ref_size - effective_size) / ref_size
            declared_match = 0.0

        rewards.append(
            _objective_reward_if_feasible(
                feasible=feasible,
                signed_gap=signed_gap,
                declared_match=declared_match,
                apply_constraint_gate=apply_constraint_gate,
                constraints_fraction=constraints_fraction,
                allow_infeasible_partial_credit=allow_infeasible_partial_credit,
            )
        )

    return rewards


def _grpo_graded_mis_objective_reward(prompts, completions, **kwargs):
    if len(prompts) != len(completions):
        raise ValueError("Graded MIS prompt/completion counts differ")
    inputs = kwargs.get("input")
    if inputs is not None and (not isinstance(inputs, list) or len(inputs) != len(completions)):
        raise ValueError("Graded MIS input/completion counts differ")
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        instance = _get_instance(kwargs, i)
        context = _combined_context(prompt, _as_text(inputs[i]) if inputs is not None else "")
        if instance is None:
            num_nodes, edges = _mis_meta(None, context)
        else:
            if not isinstance(instance, dict) or type(instance.get("num_nodes")) is not int:
                raise ValueError("Graded MIS requires explicit integer num_nodes metadata")
            num_nodes, raw_edges = instance["num_nodes"], instance.get("edges")
            if not isinstance(raw_edges, list) or any(
                not isinstance(edge, (list, tuple)) or len(edge) != 2
                or any(type(v) is not int for v in edge) or edge[0] == edge[1] for edge in raw_edges
            ):
                raise ValueError("Graded MIS requires explicit simple-graph edge metadata")
            edges = {tuple(sorted(edge)) for edge in raw_edges}
        nodes = _parse_mis_solution(_as_text(completion))["node_set"]
        rewards.append(grpo_graded_mis_reward(nodes, num_nodes, edges))
    return rewards


def _parse_jssp_solution(text: str) -> dict:
    block = _extract_bracket_block(text, "Schedule")
    schedule = _to_nested_int_lists(_safe_literal_eval_list(block))
    makespan = _extract_number(text, "Makespan")
    return {"schedule": schedule, "declared_makespan": makespan}


def _jssp_jobs(instance, input_text: str) -> list[list[tuple[int, int]]]:
    jobs: list[list[tuple[int, int]]] = []
    if isinstance(instance, list):
        parse_ok = True
        for row in instance:
            if not isinstance(row, list) or len(row) % 2 != 0:
                parse_ok = False
                break
            ops: list[tuple[int, int]] = []
            for i in range(0, len(row), 2):
                try:
                    machine = int(row[i])
                    duration = int(row[i + 1])
                except (TypeError, ValueError):
                    parse_ok = False
                    break
                ops.append((machine, duration))
            if not parse_ok:
                break
            jobs.append(ops)
        if parse_ok and jobs:
            return jobs

    # Fallback parser from input text.
    for m in re.finditer(r"Job\s*(\d+).*?\[(.*?)\]", input_text, flags=re.IGNORECASE):
        raw_block = m.group(2)
        pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", raw_block)
        if not pairs:
            continue
        ops = [(int(machine), int(duration)) for machine, duration in pairs]
        jobs.append(ops)
    return jobs


def _jssp_schedule_diagnostics(
    jobs: list[list[tuple[int, int]]],
    schedule: list[list[int]],
) -> dict[str, float | bool | None]:
    empty = {
        "makespan": None,
        "topology_applicable": False,
        "processed_fraction": 0.0,
        "cycle_node_fraction": 0.0,
    }
    if not jobs or not schedule:
        return empty

    n_jobs = len(jobs)
    m_machines = len(jobs[0])
    if any(len(ops) != m_machines for ops in jobs):
        return empty
    if len(schedule) != m_machines:
        return empty

    # Build map: for each job, where each machine appears in its operation sequence.
    op_index_by_machine: list[dict[int, int]] = []
    for ops in jobs:
        mp: dict[int, int] = {}
        for idx, (machine, _) in enumerate(ops):
            mp[machine] = idx
        op_index_by_machine.append(mp)

    all_jobs = set(range(n_jobs))
    for machine_id, order in enumerate(schedule):
        if len(order) != n_jobs:
            return empty
        if set(order) != all_jobs:
            return empty
        if len(order) != len(set(order)):
            return empty
        for job in order:
            if machine_id not in op_index_by_machine[job]:
                return empty

    nodes = []
    duration = {}
    indegree = {}
    adj = {}
    earliest_start = {}

    for j, ops in enumerate(jobs):
        for k, (_, d) in enumerate(ops):
            node = (j, k)
            nodes.append(node)
            duration[node] = float(d)
            indegree[node] = 0
            adj[node] = []
            earliest_start[node] = 0.0

    def add_edge(u, v):
        adj[u].append(v)
        indegree[v] += 1

    # Job precedence edges.
    for j, ops in enumerate(jobs):
        for k in range(1, len(ops)):
            add_edge((j, k - 1), (j, k))

    # Machine precedence edges from schedule order.
    for machine_id, order in enumerate(schedule):
        for idx in range(1, len(order)):
            prev_job = order[idx - 1]
            next_job = order[idx]
            prev_node = (prev_job, op_index_by_machine[prev_job][machine_id])
            next_node = (next_job, op_index_by_machine[next_job][machine_id])
            add_edge(prev_node, next_node)

    queue = [node for node in nodes if indegree[node] == 0]
    processed = 0
    processed_nodes = set()

    while queue:
        node = queue.pop()
        processed += 1
        processed_nodes.add(node)
        finish = earliest_start[node] + duration[node]
        for nxt in adj[node]:
            if finish > earliest_start[nxt]:
                earliest_start[nxt] = finish
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if processed != len(nodes):
        remaining_nodes = set(nodes) - processed_nodes

        # Count nodes that are actually inside a directed cycle. If something
        # unexpected prevents SCC detection, fall back to all unprocessed nodes.
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        cyclic_nodes = set()

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for nxt in adj[node]:
                if nxt not in remaining_nodes:
                    continue
                if nxt not in indices:
                    strongconnect(nxt)
                    lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
                elif nxt in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[nxt])

            if lowlinks[node] == indices[node]:
                component = []
                while True:
                    cur = stack.pop()
                    on_stack.remove(cur)
                    component.append(cur)
                    if cur == node:
                        break
                if len(component) > 1:
                    cyclic_nodes.update(component)
                elif component and component[0] in adj[component[0]]:
                    cyclic_nodes.add(component[0])

        for node in list(remaining_nodes):
            if node not in indices:
                strongconnect(node)

        cycle_count = len(cyclic_nodes) if cyclic_nodes else len(remaining_nodes)
        total_nodes = max(len(nodes), 1)
        return {
            "makespan": None,
            "topology_applicable": True,
            "processed_fraction": _safe_fraction(float(processed), float(total_nodes)),
            "cycle_node_fraction": _safe_fraction(float(cycle_count), float(total_nodes)),
        }

    makespan = 0.0
    for node in nodes:
        makespan = max(makespan, earliest_start[node] + duration[node])
    return {
        "makespan": float(makespan),
        "topology_applicable": True,
        "processed_fraction": 1.0,
        "cycle_node_fraction": 0.0,
    }


def _schedule_makespan(jobs: list[list[tuple[int, int]]], schedule: list[list[int]]) -> float | None:
    diagnostics = _jssp_schedule_diagnostics(jobs, schedule)
    makespan = diagnostics.get("makespan")
    return float(makespan) if makespan is not None else None


def _check_jssp_constraints(jobs: list[list[tuple[int, int]]], schedule: list[list[int]]) -> tuple[int, int, bool]:
    total = 3
    passed = 0

    if schedule:
        passed += 1

    n_jobs = len(jobs)
    m_machines = len(jobs[0]) if jobs else 0
    all_jobs = set(range(n_jobs))

    if schedule and len(schedule) == m_machines and all(len(row) == n_jobs and set(row) == all_jobs for row in schedule):
        passed += 1

    makespan = _schedule_makespan(jobs, schedule)
    if makespan is not None:
        passed += 1

    return passed, total, passed == total


def _jssp_order_descent_improved_schedule(
    jobs: list[list[tuple[int, int]]],
    schedule: list[list[int]],
    max_passes: int = 1,
) -> tuple[list[list[int]], float, float, float, int] | None:
    """Bounded adjacent-swap descent on the per-machine job orders.

    Every candidate order is re-simulated with the same disjunctive-graph
    earliest-start evaluation the reward uses (`_schedule_makespan`); swapped
    orders that deadlock evaluate to None and are skipped, so each accepted
    step stays feasible by construction. Returns
    (schedule, original_makespan, improved_makespan, improvement_ratio, passes)
    or None when no strict improvement exists.
    """
    max_passes = max(0, int(max_passes))
    if max_passes <= 0:
        return None
    original = _schedule_makespan(jobs, schedule)
    if original is None or original <= 0.0:
        return None

    current = [list(order) for order in schedule]
    best_makespan = float(original)
    passes = 0
    for _ in range(max_passes):
        best_candidate = None
        best_value = best_makespan - 1e-9
        for machine_id, order in enumerate(current):
            for pos in range(len(order) - 1):
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
                value = _schedule_makespan(jobs, current)
                order[pos], order[pos + 1] = order[pos + 1], order[pos]
                if value is not None and value < best_value:
                    best_value = float(value)
                    best_candidate = (machine_id, pos)
        if best_candidate is None:
            break
        machine_id, pos = best_candidate
        current[machine_id][pos], current[machine_id][pos + 1] = (
            current[machine_id][pos + 1],
            current[machine_id][pos],
        )
        best_makespan = best_value
        passes += 1

    if passes <= 0 or best_makespan >= float(original):
        return None
    improvement_ratio = float((float(original) - best_makespan) / float(original))
    return current, float(original), float(best_makespan), improvement_ratio, passes


def _format_jssp_distillation_target(schedule: list[list[int]], makespan: float) -> str:
    rows = ", ".join(
        "[" + ", ".join(str(int(job)) for job in order) + "]" for order in schedule
    )
    return f"Schedule: [{rows}], Makespan: {int(round(float(makespan)))}"


def jssp_structure_reward_hybrid(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _as_text(completion)
        parsed = _parse_jssp_solution(text)
        schedule = parsed["schedule"]
        has_schedule_block = _extract_bracket_block(text, "Schedule") is not None
        has_schedule = len(schedule) > 0
        has_makespan = parsed["declared_makespan"] is not None
        rewards.append(float((int(has_schedule_block) + int(has_schedule) + int(has_makespan)) / 3.0))
    return rewards


def jssp_constraints_reward_hybrid(prompts, completions, **kwargs):
    rewards = []
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_jssp_solution(_as_text(completion))
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        jobs = _jssp_jobs(_get_instance(kwargs, i), input_text)
        if not jobs:
            rewards.append(0.0)
            continue
        passed, total, _ = _check_jssp_constraints(jobs, parsed["schedule"])
        rewards.append(float(passed / total))
    return rewards


def jssp_makespan_reward_hybrid(prompts, completions, **kwargs):
    rewards = []
    gt_outputs = kwargs.get("answer", [None] * len(completions))
    apply_constraint_gate = bool(kwargs.get("apply_constraint_gate", True))
    allow_infeasible_partial_credit = bool(kwargs.get("allow_infeasible_partial_credit", False))
    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None

    for i, (prompt, completion, gt) in enumerate(zip(prompts, completions, gt_outputs)):
        if gt is None:
            rewards.append(0.0)
            continue

        pred = _parse_jssp_solution(_as_text(completion))
        ref = _parse_jssp_solution(_as_text(gt))

        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        jobs = _jssp_jobs(_get_instance(kwargs, i), input_text)
        if not jobs:
            rewards.append(0.0)
            continue

        pred_makespan = _schedule_makespan(jobs, pred["schedule"])
        ref_makespan = _schedule_makespan(jobs, ref["schedule"])

        if pred_makespan is None or ref_makespan is None or ref_makespan <= 0.0:
            rewards.append(0.0)
            continue

        passed, total, feasible = _check_jssp_constraints(jobs, pred["schedule"])
        constraints_fraction = passed / total

        signed_gap = (pred_makespan - ref_makespan) / ref_makespan

        declared = pred["declared_makespan"]
        declared_match = 0.0
        if declared is not None:
            tol = max(1.0, 0.01 * pred_makespan)
            declared_match = 1.0 if abs(declared - pred_makespan) <= tol else 0.0

        rewards.append(
            _objective_reward_if_feasible(
                feasible=feasible,
                signed_gap=signed_gap,
                declared_match=declared_match,
                apply_constraint_gate=apply_constraint_gate,
                constraints_fraction=constraints_fraction,
                allow_infeasible_partial_credit=allow_infeasible_partial_credit,
            )
        )

    return rewards


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return _clip01(numerator / denominator)


def _masked_constraint_score(row: list[float], mask: list[float]) -> float:
    active_values = [value for value, active in zip(row, mask) if active > 0.0]
    if not active_values:
        return 1.0
    return float(1.0 - (sum(active_values) / len(active_values)))


def _finalize_cost_row(row: list[float], mask: list[float]) -> tuple[list[float], list[float], float]:
    clipped_row = [_clip01(value) for value in row]
    clipped_mask = [_clip01(value) for value in mask]
    return clipped_row, clipped_mask, _masked_constraint_score(clipped_row, clipped_mask)


def tsp_cost_components(prompts, completions, **kwargs):
    target_gap, target_gap_scale = _tsp_target_gap_constraint_params()
    use_target_gap_constraint = target_gap > 0.0 and target_gap_scale > 0.0
    tail_gap, tail_gap_scale = _tsp_target_tail_gap_constraint_params()
    use_tail_gap_constraint = tail_gap > 0.0 and tail_gap_scale > 0.0
    use_objective_gap_cost, objective_gap_cap, objective_gap_scale = _tsp_objective_gap_cost_params()
    cost_names = [
        "parse_failure",
        "depot_boundary",
        "route_length_mismatch",
        "invalid_node",
        "duplicate_customer",
        "missing_customer",
        "interior_depot",
    ]
    if use_target_gap_constraint:
        cost_names.append("target_gap_excess")
    if use_tail_gap_constraint:
        cost_names.append("target_tail_gap_excess")
    if use_objective_gap_cost:
        cost_names.append("tsp_objective_gap")
    cost_vectors = []
    cost_masks = []
    constraint_scores = []

    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_tsp_solution(_as_text(completion))
        route = parsed["route"]
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords = _tsp_coords_from_instance(_get_instance(kwargs, i), input_text)

        if not route:
            row = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            mask = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            interior_depots = sum(1 for node in route[1:-1] if node == 0) if len(route) > 2 else 0
            row = [
                0.0,
                0.0 if len(route) >= 2 and route[0] == 0 and route[-1] == 0 else 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                _safe_fraction(interior_depots, max(len(route) - 2, 1)),
            ]
            mask = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            if coords:
                num_nodes = len(coords)
                num_customers = max(num_nodes - 1, 1)
                invalid_nodes = sum(1 for node in route if node < 0 or node >= num_nodes)
                non_depot_valid = [node for node in route if 0 < node < num_nodes]
                duplicate_customers = len(non_depot_valid) - len(set(non_depot_valid))
                missing_customers = len(set(range(1, num_nodes)) - set(non_depot_valid))
                row[2] = _safe_fraction(abs(len(route) - (num_nodes + 1)), max(num_nodes + 1, 1))
                row[3] = _safe_fraction(invalid_nodes, max(len(route), 1))
                row[4] = _safe_fraction(duplicate_customers, num_customers)
                row[5] = _safe_fraction(missing_customers, num_customers)
                mask[2] = 1.0
                mask[3] = 1.0
                mask[4] = 1.0
                mask[5] = 1.0

        if use_target_gap_constraint or use_tail_gap_constraint or use_objective_gap_cost:
            target_violation = 0.0
            target_mask = 0.0
            tail_violation = 0.0
            tail_mask = 0.0
            objective_gap_cost = 0.0
            objective_gap_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if coords and gt is not None:
                ref = _parse_tsp_solution(_as_text(gt))
                pred_total = _distance(coords, route)
                ref_total = _distance(coords, ref["route"])
                if ref_total is not None and pred_total is not None and ref_total > 0.0 and pred_total > 0.0:
                    _, _, feasible = _check_tsp_constraints(route, len(coords))
                    if feasible:
                        signed_gap = (pred_total - ref_total) / ref_total
                        nonnegative_gap = max(0.0, signed_gap)
                        if use_target_gap_constraint:
                            excess = max(0.0, nonnegative_gap - target_gap)
                            target_violation = _clip01(target_gap_scale * _safe_fraction(excess, target_gap))
                            target_mask = 1.0
                        if use_tail_gap_constraint:
                            tail_excess = max(0.0, nonnegative_gap - tail_gap)
                            tail_violation = _clip01(tail_gap_scale * _safe_fraction(tail_excess, tail_gap))
                            tail_mask = 1.0
                        if use_objective_gap_cost:
                            objective_gap_cost = _clip01(
                                objective_gap_scale
                                * _safe_fraction(nonnegative_gap, objective_gap_cap)
                            )
                            objective_gap_mask = 1.0
            if use_target_gap_constraint:
                row.append(target_violation)
                mask.append(target_mask)
            if use_tail_gap_constraint:
                row.append(tail_violation)
                mask.append(tail_mask)
            if use_objective_gap_cost:
                row.append(objective_gap_cost)
                mask.append(objective_gap_mask)

        row, mask, constraint_score = _finalize_cost_row(row, mask)
        cost_vectors.append(row)
        cost_masks.append(mask)
        constraint_scores.append(constraint_score)

    return {
        "cost_names": cost_names,
        "cost_vectors": cost_vectors,
        "cost_masks": cost_masks,
        "constraint_scores": constraint_scores,
    }


def cvrp_cost_components(prompts, completions, **kwargs):
    target_gap, target_gap_scale = _cvrp_target_gap_constraint_params()
    use_target_gap_constraint = target_gap > 0.0 and target_gap_scale > 0.0
    use_objective_gap_cost, objective_gap_cap, objective_gap_scale = _cvrp_objective_gap_cost_params()
    base_cost_names = [
        "parse_failure",
        "route_count_mismatch",
        "start_at_depot",
        "end_at_depot",
        "depot_boundary",
        "route_shape_invalid",
        "invalid_node",
        "duplicate_customer",
        "missing_customer",
        "customer_count_mismatch",
        "interior_depot",
        "capacity_overflow",
        "over_capacity_route_fraction",
        "max_route_capacity_overflow",
    ]
    cost_names = list(base_cost_names)
    if use_target_gap_constraint:
        cost_names.append("cvrp_target_gap_excess")
    if use_objective_gap_cost:
        cost_names.append("cvrp_objective_gap")
    cost_vectors = []
    cost_masks = []
    constraint_scores = []

    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_cvrp_solution(_as_text(completion))
        routes = parsed["routes"]
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, demands, capacity = _cvrp_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))

        if not routes:
            row = [1.0] + [0.0] * (len(base_cost_names) - 1)
            mask = [1.0] + [0.0] * (len(base_cost_names) - 1)
        else:
            expected_vehicles = kwargs.get("num_vehicles", None)
            if expected_vehicles is None:
                prompt_vehicle_match = re.search(r"(\d+)\s*vehicle", _combined_context(prompt, input_text), flags=re.IGNORECASE)
                expected_vehicles = int(prompt_vehicle_match.group(1)) if prompt_vehicle_match else None

            all_nodes = [node for route in routes for node in route]
            bad_start_routes = sum(
                1 for route in routes if len(route) < 1 or route[0] != 0
            )
            bad_end_routes = sum(
                1 for route in routes if len(route) < 1 or route[-1] != 0
            )
            bad_shape_routes = sum(1 for route in routes if len(route) < 2)
            bad_boundary_routes = sum(
                1 for route in routes if len(route) < 2 or route[0] != 0 or route[-1] != 0
            )
            interior_depots = sum(
                1 for route in routes for node in route[1:-1] if node == 0
            )
            row = [
                0.0,
                0.0 if expected_vehicles is None else _safe_fraction(abs(len(routes) - expected_vehicles), max(expected_vehicles, 1)),
                _safe_fraction(bad_start_routes, max(len(routes), 1)),
                _safe_fraction(bad_end_routes, max(len(routes), 1)),
                _safe_fraction(bad_boundary_routes, max(len(routes), 1)),
                _safe_fraction(bad_shape_routes, max(len(routes), 1)),
                0.0,
                0.0,
                0.0,
                0.0,
                _safe_fraction(interior_depots, max(sum(max(len(route) - 2, 0) for route in routes), 1)),
                0.0,
                0.0,
                0.0,
            ]
            mask = [
                1.0,
                0.0 if expected_vehicles is None else 1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]

            if coords:
                valid_nodes = set(range(len(coords)))
                expected_customers = set(range(1, len(coords)))
                non_depot = [node for node in all_nodes if node != 0]
                valid_customers = [node for node in non_depot if node in expected_customers]
                invalid_nodes = sum(1 for node in all_nodes if node not in valid_nodes)
                duplicate_customers = len(valid_customers) - len(set(valid_customers))
                missing_customers = len(expected_customers - set(valid_customers))
                row[6] = _safe_fraction(invalid_nodes, max(len(all_nodes), 1))
                row[7] = _safe_fraction(duplicate_customers, max(len(expected_customers), 1))
                row[8] = _safe_fraction(missing_customers, max(len(expected_customers), 1))
                row[9] = _safe_fraction(
                    abs(len(valid_customers) - len(expected_customers)),
                    max(len(expected_customers), 1),
                )
                mask[6] = 1.0
                mask[7] = 1.0
                mask[8] = 1.0
                mask[9] = 1.0

            if capacity is not None and demands:
                overflow = 0.0
                max_overflow = 0.0
                over_capacity_routes = 0
                for route in routes:
                    route_demand = sum(demands[node] for node in route if 0 <= node < len(demands) and node != 0)
                    route_overflow = max(0.0, route_demand - capacity)
                    overflow += route_overflow
                    max_overflow = max(max_overflow, route_overflow)
                    if route_overflow > 1e-9:
                        over_capacity_routes += 1
                row[11] = _safe_fraction(overflow, max(float(capacity) * max(len(routes), 1), 1.0))
                row[12] = _safe_fraction(over_capacity_routes, max(len(routes), 1))
                row[13] = _safe_fraction(max_overflow, max(float(capacity), 1.0))
                mask[11] = 1.0
                mask[12] = 1.0
                mask[13] = 1.0

        if use_target_gap_constraint or use_objective_gap_cost:
            target_violation = 0.0
            target_mask = 0.0
            objective_gap_cost = 0.0
            objective_gap_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if coords and demands and capacity is not None and gt is not None:
                ref = _parse_cvrp_solution(_as_text(gt))
                pred_total = _cvrp_total_distance(coords, routes)
                ref_total = _cvrp_total_distance(coords, ref["routes"])
                if (
                    pred_total is not None
                    and ref_total is not None
                    and ref_total > 0.0
                    and pred_total > 0.0
                ):
                    _, _, feasible = _check_cvrp_constraints(routes, coords, demands, capacity)
                    if feasible:
                        signed_gap = (pred_total - ref_total) / ref_total
                        nonnegative_gap = max(0.0, signed_gap)
                        if use_target_gap_constraint:
                            excess = max(0.0, nonnegative_gap - target_gap)
                            denom = max(target_gap, 1.0 / float(ref_total))
                            target_violation = _clip01(
                                target_gap_scale * _safe_fraction(excess, denom)
                            )
                            target_mask = 1.0
                        if use_objective_gap_cost:
                            objective_gap_cost = _clip01(
                                objective_gap_scale
                                * _safe_fraction(nonnegative_gap, objective_gap_cap)
                            )
                            objective_gap_mask = 1.0
            if use_target_gap_constraint:
                row.append(target_violation)
                mask.append(target_mask)
            if use_objective_gap_cost:
                row.append(objective_gap_cost)
                mask.append(objective_gap_mask)

        row, mask, constraint_score = _finalize_cost_row(row, mask)
        cost_vectors.append(row)
        cost_masks.append(mask)
        constraint_scores.append(constraint_score)

    return {
        "cost_names": cost_names,
        "cost_vectors": cost_vectors,
        "cost_masks": cost_masks,
        "constraint_scores": constraint_scores,
    }


def op_cost_components(prompts, completions, **kwargs):
    length_margin_ratio = _op_length_margin_ratio()
    use_length_margin = length_margin_ratio > 0.0
    use_prize_surplus = _op_prize_surplus_constraint_enabled()
    use_prize_cap = _op_prize_cap_constraint_enabled()
    cost_names = [
        "parse_failure",
        "start_at_depot",
        "invalid_node",
        "duplicate_visit",
        "length_overflow",
    ]
    if use_length_margin:
        cost_names.append("length_margin_overflow")
    if use_prize_surplus:
        cost_names.append("prize_surplus_overbudget")
    if use_prize_cap:
        cost_names.append("prize_cap_over_ref")
    prize_surplus_idx = cost_names.index("prize_surplus_overbudget") if use_prize_surplus else None
    prize_cap_idx = cost_names.index("prize_cap_over_ref") if use_prize_cap else None
    cost_vectors = []
    cost_masks = []
    constraint_scores = []

    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_op_solution(_as_text(completion))
        route = parsed["route"]
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        coords, prizes, max_len = _op_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))

        if not route:
            row = [1.0, 0.0, 0.0, 0.0, 0.0]
            mask = [1.0, 0.0, 0.0, 0.0, 0.0]
            if use_length_margin:
                row.append(0.0)
                mask.append(0.0)
            if use_prize_surplus:
                row.append(0.0)
                mask.append(0.0)
            if use_prize_cap:
                row.append(0.0)
                mask.append(0.0)
        else:
            duplicate_visits = len(route) - len(set(route))
            row = [
                0.0,
                0.0 if route and route[0] == 0 else 1.0,
                0.0,
                _safe_fraction(duplicate_visits, max(len(route), 1)),
                0.0,
            ]
            mask = [1.0, 1.0, 0.0, 1.0, 0.0]
            if use_length_margin:
                row.append(0.0)
                mask.append(0.0)
            if use_prize_surplus:
                row.append(0.0)
                mask.append(0.0)
            if use_prize_cap:
                row.append(0.0)
                mask.append(0.0)
            if coords:
                invalid_nodes = sum(1 for node in route if node < 0 or node >= len(coords))
                row[2] = _safe_fraction(invalid_nodes, max(len(route), 1))
                mask[2] = 1.0
                route_len = _distance(coords, route)
                if max_len is not None:
                    row[4] = (
                        1.0
                        if route_len is None
                        else _safe_fraction(max(0.0, route_len - max_len), max(max_len, 1.0))
                    )
                    mask[4] = 1.0
                    if use_length_margin and route_len is not None:
                        margin_len = length_margin_ratio * float(max_len)
                        row[5] = _safe_fraction(max(0.0, route_len - margin_len), max(max_len, 1.0))
                        mask[5] = 1.0
                if use_prize_surplus or use_prize_cap:
                    gt = gt_outputs[i] if i < len(gt_outputs) else None
                    ref = _parse_op_solution(_as_text(gt)) if gt is not None else {"route": []}
                    pred_prize = _op_prize(route, prizes)
                    ref_prize = _op_prize(ref["route"], prizes)
                    if pred_prize is not None and ref_prize is not None and ref_prize > 0.0:
                        surplus = max(0.0, float(pred_prize) - float(ref_prize))
                        if use_prize_cap and prize_cap_idx is not None:
                            row[prize_cap_idx] = _safe_fraction(surplus, float(ref_prize))
                            mask[prize_cap_idx] = 1.0
                        if (
                            use_prize_surplus
                            and prize_surplus_idx is not None
                            and max_len is not None
                            and route_len is not None
                        ):
                            over_budget = route_len > float(max_len) + 1e-9
                            row[prize_surplus_idx] = (
                                _safe_fraction(surplus, float(ref_prize)) if over_budget else 0.0
                            )
                            mask[prize_surplus_idx] = 1.0

        row, mask, constraint_score = _finalize_cost_row(row, mask)
        cost_vectors.append(row)
        cost_masks.append(mask)
        constraint_scores.append(constraint_score)

    return {
        "cost_names": cost_names,
        "cost_vectors": cost_vectors,
        "cost_masks": cost_masks,
        "constraint_scores": constraint_scores,
    }


def mis_cost_components(prompts, completions, **kwargs):
    target_gap, target_gap_scale = _mis_target_gap_constraint_params()
    use_target_gap_constraint = target_gap_scale > 0.0
    repair_target_gap, repair_target_gap_scale = _mis_repair_target_gap_constraint_params()
    use_repair_target_gap_constraint = repair_target_gap_scale > 0.0
    size_excess_scale, size_excess_max_ref_ratio = _mis_size_excess_constraint_params()
    use_size_excess_constraint = size_excess_scale > 0.0
    conflict_node_scale = _mis_conflict_node_constraint_scale()
    use_conflict_node_constraint = conflict_node_scale > 0.0
    maximality_scale, maximality_independent_only = _mis_maximality_constraint_params()
    use_maximality_constraint = maximality_scale > 0.0
    cost_names = [
        "parse_failure",
        "invalid_node",
        "duplicate_node",
        "edge_conflict",
    ]
    if use_conflict_node_constraint:
        cost_names.append("conflict_node_fraction")
    if use_target_gap_constraint:
        cost_names.append("size_deficit_over_ref")
    if use_size_excess_constraint:
        cost_names.append("size_excess_over_ref")
    if use_repair_target_gap_constraint:
        cost_names.append("repair_size_deficit_over_ref")
    if use_maximality_constraint:
        cost_names.append("addable_vertex_fraction")
    cost_vectors = []
    cost_masks = []
    constraint_scores = []

    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        parsed = _parse_mis_solution(_as_text(completion))
        node_set = parsed["node_set"]
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        num_nodes, edges = _mis_meta(_get_instance(kwargs, i), _combined_context(prompt, input_text))
        has_graph_meta = num_nodes is not None or bool(edges)
        selected = []
        conflict_nodes: set[int] = set()

        if not node_set:
            row = [1.0, 0.0, 0.0, 0.0]
            mask = [1.0, 0.0, 0.0, 0.0]
        else:
            duplicate_nodes = len(node_set) - len(set(node_set))
            row = [
                0.0,
                0.0,
                _safe_fraction(duplicate_nodes, max(len(node_set), 1)),
                0.0,
            ]
            mask = [1.0, 0.0, 1.0, 1.0 if has_graph_meta else 0.0]
            if num_nodes is not None:
                invalid_nodes = sum(1 for node in node_set if node < 0 or node >= num_nodes)
                selected = list(dict.fromkeys(node for node in node_set if 0 <= node < num_nodes))
                row[1] = _safe_fraction(invalid_nodes, max(len(node_set), 1))
                mask[1] = 1.0
            else:
                selected = list(dict.fromkeys(node_set))
            if has_graph_meta:
                selected_set = set(selected)
                conflicting_edges = 0
                for edge in edges:
                    if edge[0] in selected_set and edge[1] in selected_set:
                        conflicting_edges += 1
                        conflict_nodes.add(edge[0])
                        conflict_nodes.add(edge[1])
                row[3] = _safe_fraction(conflicting_edges, max(len(selected), 1))

        if use_conflict_node_constraint:
            node_violation = 0.0
            node_mask = 0.0
            if node_set and has_graph_meta:
                node_violation = _clip01(
                    conflict_node_scale * _safe_fraction(len(conflict_nodes), max(len(selected), 1))
                )
                node_mask = 1.0
            row.append(node_violation)
            mask.append(node_mask)

        if use_target_gap_constraint:
            deficit_violation = 0.0
            deficit_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if gt is not None:
                ref = _parse_mis_solution(_as_text(gt))
                ref_size = len(ref["node_set"])
                if ref_size > 0:
                    _, _, feasible = _check_mis_constraints(node_set, num_nodes, edges)
                    if feasible:
                        signed_gap = (float(ref_size) - float(len(node_set))) / float(ref_size)
                        excess = max(0.0, signed_gap - target_gap)
                        # MIS objectives are integer set sizes. A one-node deficit
                        # is often already above Jiang's 1.04% target, so normalize
                        # by the larger of the target and one missing node.
                        denom = max(target_gap, 1.0 / float(ref_size))
                        deficit_violation = _clip01(target_gap_scale * _safe_fraction(excess, denom))
                        deficit_mask = 1.0
            row.append(deficit_violation)
            mask.append(deficit_mask)

        if use_size_excess_constraint:
            excess_violation = 0.0
            excess_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if gt is not None:
                ref = _parse_mis_solution(_as_text(gt))
                ref_size = len(ref["node_set"])
                ref_ratio = None
                if ref_size > 0:
                    if num_nodes is not None and num_nodes > 0:
                        ref_ratio = float(ref_size) / float(num_nodes)
                    if ref_ratio is None or ref_ratio <= size_excess_max_ref_ratio + 1e-12:
                        excess = max(0.0, float(len(node_set) - ref_size))
                        denom = max(1.0 / float(ref_size), 1e-12)
                        signed_gap = excess / float(ref_size)
                        excess_violation = _clip01(
                            size_excess_scale * _safe_fraction(signed_gap, denom)
                        )
                        excess_mask = 1.0
            row.append(excess_violation)
            mask.append(excess_mask)

        if use_repair_target_gap_constraint:
            repair_deficit_violation = 0.0
            repair_deficit_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if gt is not None and has_graph_meta:
                ref = _parse_mis_solution(_as_text(gt))
                ref_size = len(ref["node_set"])
                if ref_size > 0:
                    if num_nodes is not None:
                        valid_selected = list(dict.fromkeys(node for node in node_set if 0 <= node < num_nodes))
                    else:
                        valid_selected = list(dict.fromkeys(node_set))
                    cover_size = (
                        _mis_greedy_conflict_cover_size(valid_selected, edges)
                        if valid_selected
                        else 0
                    )
                    repaired_size = max(0.0, float(len(valid_selected) - cover_size))
                    signed_gap = (float(ref_size) - repaired_size) / float(ref_size)
                    excess = max(0.0, signed_gap - repair_target_gap)
                    # MIS gaps are discrete; one missed vertex is the practical
                    # unit even when the target percentage is much smaller.
                    denom = max(repair_target_gap, 1.0 / float(ref_size))
                    repair_deficit_violation = _clip01(
                        repair_target_gap_scale * _safe_fraction(excess, denom)
                    )
                    repair_deficit_mask = 1.0
            row.append(repair_deficit_violation)
            mask.append(repair_deficit_mask)

        if use_maximality_constraint:
            addable_violation = 0.0
            addable_mask = 0.0
            if num_nodes is not None and num_nodes > 0 and selected:
                _, _, feasible = _check_mis_constraints(node_set, num_nodes, edges)
                if feasible or not maximality_independent_only:
                    addable_count = _mis_addable_vertex_count(selected, num_nodes, edges)
                    denom = float(num_nodes)
                    gt = gt_outputs[i] if i < len(gt_outputs) else None
                    if gt is not None:
                        ref = _parse_mis_solution(_as_text(gt))
                        ref_size = len(ref["node_set"])
                        if ref_size > 0:
                            denom = float(ref_size)
                    addable_violation = _clip01(
                        maximality_scale * _safe_fraction(float(addable_count), max(denom, 1.0))
                    )
                    addable_mask = 1.0
            row.append(addable_violation)
            mask.append(addable_mask)

        row, mask, constraint_score = _finalize_cost_row(row, mask)
        cost_vectors.append(row)
        cost_masks.append(mask)
        constraint_scores.append(constraint_score)

    return {
        "cost_names": cost_names,
        "cost_vectors": cost_vectors,
        "cost_masks": cost_masks,
        "constraint_scores": constraint_scores,
    }


def jssp_cost_components_hybrid(prompts, completions, **kwargs):
    target_gap, target_gap_scale = _jssp_target_gap_constraint_params()
    use_target_gap_constraint = target_gap_scale > 0.0
    cycle_fraction_scale = _jssp_cycle_fraction_constraint_scale()
    use_cycle_fraction_constraint = cycle_fraction_scale > 0.0
    processed_deficit_scale = _jssp_processed_deficit_constraint_scale()
    use_processed_deficit_constraint = processed_deficit_scale > 0.0
    parse_bad_token_scale = _jssp_parse_bad_token_constraint_scale()
    use_parse_bad_token_constraint = parse_bad_token_scale > 0.0
    cost_names = [
        "parse_failure",
        "machine_row_mismatch",
        "row_length_mismatch",
        "invalid_job",
        "duplicate_job",
        "missing_job",
        "row_set_mismatch",
        "schedule_invalid",
    ]
    if use_cycle_fraction_constraint:
        cost_names.append("cycle_node_fraction")
    if use_processed_deficit_constraint:
        cost_names.append("processed_node_deficit")
    if use_parse_bad_token_constraint:
        cost_names.append("parse_bad_token")
    if use_target_gap_constraint:
        cost_names.append("makespan_gap_over_ref")
    cost_vectors = []
    cost_masks = []
    constraint_scores = []

    input_list = kwargs.get("input") if isinstance(kwargs.get("input"), list) else None
    raw_gt_outputs = kwargs.get("answer")
    gt_outputs = raw_gt_outputs if isinstance(raw_gt_outputs, list) else [None] * len(completions)
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        completion_text = _as_text(completion)
        parsed = _parse_jssp_solution(completion_text)
        schedule = parsed["schedule"]
        input_text = _as_text(input_list[i]) if input_list and i < len(input_list) else _as_text(prompt)
        jobs = _jssp_jobs(_get_instance(kwargs, i), input_text)
        topology_diagnostics = _jssp_schedule_diagnostics(jobs, schedule)

        if not schedule:
            row = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            mask = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif not jobs:
            row = [0.0] * 8
            mask = [1.0] + [0.0] * 7
        else:
            n_jobs = len(jobs)
            m_machines = len(jobs[0]) if jobs else 0
            machine_row_mismatch = _safe_fraction(abs(len(schedule) - m_machines), max(m_machines, 1))

            row_length_mismatch = 0.0
            invalid_job = 0.0
            duplicate_job = 0.0
            missing_job = 0.0
            row_set_mismatch = 0.0
            expected_jobs = set(range(n_jobs))
            denom_rows = max(len(schedule), 1)
            for row_schedule in schedule:
                row_length_mismatch += _safe_fraction(abs(len(row_schedule) - n_jobs), max(n_jobs, 1))
                valid_jobs = [job for job in row_schedule if 0 <= job < n_jobs]
                invalid_job += _safe_fraction(len(row_schedule) - len(valid_jobs), max(n_jobs, 1))
                duplicate_job += _safe_fraction(len(valid_jobs) - len(set(valid_jobs)), max(n_jobs, 1))
                missing_job += _safe_fraction(len(expected_jobs - set(valid_jobs)), max(n_jobs, 1))
                if len(row_schedule) != n_jobs or set(row_schedule) != expected_jobs:
                    row_set_mismatch += 1.0
            row_length_mismatch /= denom_rows
            invalid_job /= denom_rows
            duplicate_job /= denom_rows
            missing_job /= denom_rows
            row_set_mismatch /= denom_rows
            schedule_invalid = 0.0 if topology_diagnostics["makespan"] is not None else 1.0

            row = [
                0.0,
                machine_row_mismatch,
                row_length_mismatch,
                invalid_job,
                duplicate_job,
                missing_job,
                row_set_mismatch,
                schedule_invalid,
            ]
            mask = [1.0] * 8

        if use_cycle_fraction_constraint:
            cycle_violation = 0.0
            cycle_mask = 0.0
            if topology_diagnostics["topology_applicable"]:
                cycle_violation = cycle_fraction_scale * float(
                    topology_diagnostics["cycle_node_fraction"] or 0.0
                )
                cycle_mask = 1.0
            row.append(cycle_violation)
            mask.append(cycle_mask)

        if use_processed_deficit_constraint:
            processed_violation = 0.0
            processed_mask = 0.0
            if topology_diagnostics["topology_applicable"]:
                processed_violation = processed_deficit_scale * (
                    1.0 - float(topology_diagnostics["processed_fraction"] or 0.0)
                )
                processed_mask = 1.0
            row.append(processed_violation)
            mask.append(processed_mask)

        if use_parse_bad_token_constraint:
            block = _extract_bracket_block(completion_text, "Schedule") or ""
            bad_token = any(token in block for token in ("|", "...", "…"))
            row.append(parse_bad_token_scale if bad_token else 0.0)
            mask.append(1.0)

        if use_target_gap_constraint:
            target_violation = 0.0
            target_mask = 0.0
            gt = gt_outputs[i] if i < len(gt_outputs) else None
            if jobs and schedule and gt is not None:
                ref = _parse_jssp_solution(_as_text(gt))
                pred_makespan = _schedule_makespan(jobs, schedule)
                ref_makespan = _schedule_makespan(jobs, ref["schedule"])
                if (
                    pred_makespan is not None
                    and ref_makespan is not None
                    and ref_makespan > 0.0
                    and pred_makespan > 0.0
                ):
                    signed_gap = (pred_makespan - ref_makespan) / ref_makespan
                    excess = max(0.0, signed_gap - target_gap)
                    denom = max(target_gap, 1.0 / float(ref_makespan))
                    target_violation = _clip01(target_gap_scale * _safe_fraction(excess, denom))
                    target_mask = 1.0
            row.append(target_violation)
            mask.append(target_mask)

        row, mask, constraint_score = _finalize_cost_row(row, mask)
        cost_vectors.append(row)
        cost_masks.append(mask)
        constraint_scores.append(constraint_score)

    return {
        "cost_names": cost_names,
        "cost_vectors": cost_vectors,
        "cost_masks": cost_masks,
        "constraint_scores": constraint_scores,
    }


TASK_REWARD_TRIPLETS = {
    "tsp": (tsp_structure_reward, tsp_constraints_reward, tsp_objective_reward),
    "cvrp": (cvrp_structure_reward, cvrp_constraints_reward, cvrp_objective_reward),
    "op": (op_structure_reward, op_constraints_reward, op_objective_reward),
    "mis": (mis_structure_reward, mis_constraints_reward, mis_objective_reward),
    "jssp": (jssp_structure_reward_hybrid, jssp_constraints_reward_hybrid, jssp_makespan_reward_hybrid),
}


TASK_COST_COMPONENTS = {
    "tsp": tsp_cost_components,
    "cvrp": cvrp_cost_components,
    "op": op_cost_components,
    "mis": mis_cost_components,
    "jssp": jssp_cost_components_hybrid,
}
