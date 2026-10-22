"""Bounded training-only CVRP repair and improvement; no solver labels required."""
from __future__ import annotations

import math
from dataclasses import dataclass

from clasp.rewards.co_tasks import (
    _check_cvrp_constraints,
    _cvrp_local_search_improved_routes,
    _cvrp_total_distance,
)


@dataclass(frozen=True)
class CVRPImprovement:
    routes: list[list[int]]
    objective: float
    gain: float
    passes: int
    repaired: bool


def improve_cvrp_routes(
    routes, coords, demands, capacity, *, max_passes=2, max_nodes=256
) -> CVRPImprovement | None:
    """Keep the sampled ordering, repair coverage/capacity, then improve cost.

    A feasible source requires a strict cost improvement. An infeasible source
    requires a verified feasible result and gets no cost-based gain claim.
    This assumes the repository's unlimited-fleet CVRP contract.
    """
    if not routes or not 2 <= len(coords) <= max_nodes or len(demands) != len(coords):
        return None
    if capacity is None or not math.isfinite(capacity) or capacity <= 0:
        return None
    if any(not math.isfinite(d) or d < 0 or d > capacity for d in demands):
        return None
    if any(len(p) != 2 or any(not math.isfinite(x) for x in p) for p in coords):
        return None
    if sum(map(len, routes)) > 8 * max_nodes or max_passes < 0:
        return None
    if any(type(node) is not int for route in routes for node in route):
        return None
    _, _, source_feasible = _check_cvrp_constraints(routes, coords, demands, capacity)
    source_cost = _cvrp_total_distance(coords, routes) if source_feasible else None
    n = len(coords)
    current, seen = [], set()
    for route in routes:
        part, load = [0], 0.0
        for node in route:
            if not 0 < node < n or node in seen:
                continue
            if load + demands[node] > capacity + 1e-9:
                current.append(part + [0])
                part, load = [0], 0.0
            part.append(node)
            load += demands[node]
            seen.add(node)
        if len(part) > 1:
            current.append(part + [0])

    def distance(a, b):
        return math.dist(coords[a], coords[b])

    loads = [sum(demands[i] for i in route[1:-1]) for route in current]
    for node in sorted(set(range(1, n)) - seen):
        # A new route is always available for a valid unlimited-fleet instance.
        best = (2 * distance(0, node), len(current), 1)
        for idx, route in enumerate(current):
            if loads[idx] + demands[node] > capacity + 1e-9:
                continue
            for pos in range(1, len(route)):
                a, b = route[pos - 1], route[pos]
                candidate = (distance(a, node) + distance(node, b) - distance(a, b), idx, pos)
                best = min(best, candidate)
        _, idx, pos = best
        if idx == len(current):
            current.append([0, node, 0])
            loads.append(demands[node])
        else:
            current[idx].insert(pos, node)
            loads[idx] += demands[node]
    if not _check_cvrp_constraints(current, coords, demands, capacity)[2]:
        return None
    improved = _cvrp_local_search_improved_routes(
        coords, demands, capacity, current, max_passes=max_passes
    )
    passes = 0
    if improved is not None:
        current, _, _, _, passes = improved
    if not _check_cvrp_constraints(current, coords, demands, capacity)[2]:
        return None
    cost = _cvrp_total_distance(coords, current)
    if cost is None or not math.isfinite(cost):
        return None
    if source_feasible and (source_cost is None or cost >= source_cost - 1e-9):
        return None
    gain = (source_cost - cost) / source_cost if source_feasible and source_cost > 0 else 0.0
    return CVRPImprovement(current, cost, gain, passes, not source_feasible)
