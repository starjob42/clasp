"""Bounded, deterministic MIS teachers checked by the shared task verifier.

This module only constructs training targets. It does not change the reward,
the candidate being scored, or evaluation-time decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .co_tasks import _check_mis_constraints, _mis_maximalized_set


@dataclass(frozen=True)
class MISImprovement:
    """A verified target; ``gain`` is the signed raw cardinality difference.

    A repaired infeasible source can have nonpositive gain: becoming feasible
    is the improvement in that case. ``passes`` counts accepted one-for-two
    exchanges, not greedy insertions or conflict-removal operations.
    """

    node_set: list[int]
    original_size: int
    improved_size: int
    gain: int
    passes: int
    repaired: bool
    source_feasible: bool
    pair_checks: int
    termination: str


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def improve_mis_set(
    node_set: list[int],
    num_nodes: int | None,
    edges: set[tuple[int, int]],
    *,
    allow_repair: bool = True,
    max_exchanges: int = 2,
    max_pair_checks: int = 4096,
    max_nodes: int = 512,
    max_edges: int = 65536,
) -> MISImprovement | None:
    """Repair/fill a parsed set, then try bounded remove-one/add-two moves.

    Feasible inputs must strictly increase cardinality to produce a target.
    With repair enabled, infeasible inputs may instead produce any verified
    feasible target. An empty *parsed* set is supported; callers must reject
    parser failures separately if those should not receive a target.

    Work is bounded by graph/input size, accepted exchanges and pair checks.
    Limits count operations rather than elapsed time so results do not depend
    on machine speed. Exhausting a budget can return an improvement already
    found, but never an unverified intermediate set. Malformed/oversized graph
    metadata is rejected; it is never silently repaired or truncated.
    """
    for name, value, minimum in (
        ("max_exchanges", max_exchanges, 0),
        ("max_pair_checks", max_pair_checks, 0),
        ("max_nodes", max_nodes, 1),
        ("max_edges", max_edges, 0),
    ):
        if not _integer(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(allow_repair, bool):
        raise ValueError("allow_repair must be a bool")
    if not _integer(num_nodes) or not 0 < num_nodes <= max_nodes:
        return None
    if not isinstance(node_set, (list, tuple)) or len(node_set) > max_nodes:
        return None
    if not isinstance(edges, (set, frozenset, list, tuple)) or len(edges) > max_edges:
        return None

    n = int(num_nodes)
    neighbors = [set() for _ in range(n)]
    graph: set[tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return None
        a, b = edge
        if not _integer(a) or not _integer(b) or not (0 <= a < n and 0 <= b < n) or a == b:
            return None
        a, b = int(a), int(b)
        graph.add((min(a, b), max(a, b)))
        neighbors[a].add(b)
        neighbors[b].add(a)

    original_size = len(node_set)
    valid_nodes = [int(v) for v in node_set if _integer(v) and 0 <= v < n]
    source_feasible = len(valid_nodes) == original_size and _check_mis_constraints(valid_nodes, n, graph)[2]
    if not source_feasible and not allow_repair:
        return None
    selected = set(valid_nodes)

    # Remove the most conflicted selected vertex. Degree and id break ties,
    # making repair independent of input/edge iteration order.
    if not source_feasible:
        while selected:
            conflicts = {v: len(neighbors[v] & selected) for v in selected}
            victim = max(selected, key=lambda v: (conflicts[v], len(neighbors[v]), v))
            if conflicts[victim] == 0:
                break
            selected.remove(victim)

    def fill(current: set[int]) -> set[int]:
        # The existing maximalizer requires a nonempty feasible source.
        if not current:
            current = {min(range(n), key=lambda v: (len(neighbors[v]), v))}
        maximal = _mis_maximalized_set(sorted(current), n, graph)
        return set(maximal[0]) if maximal is not None else set(current)

    selected = fill(selected)
    passes = pair_checks = 0
    termination = "exchange_limit"
    while passes < max_exchanges:
        if pair_checks >= max_pair_checks:
            termination = "pair_check_limit"
            break
        # In a maximal set, an outside vertex becomes addable after removing
        # v exactly when its selected-neighbor set is {v}. A nonadjacent pair
        # in that bucket gives a feasible, strictly larger replacement.
        buckets: dict[int, list[int]] = {}
        for v in range(n):
            if v not in selected:
                blockers = neighbors[v] & selected
                if len(blockers) == 1:
                    buckets.setdefault(next(iter(blockers)), []).append(v)
        replacement = None
        exhausted = False
        for removed in sorted(buckets, key=lambda v: (-len(neighbors[v]), v)):
            candidates = sorted(buckets[removed], key=lambda v: (len(neighbors[v]), v))
            for i, a in enumerate(candidates):
                for b in candidates[i + 1 :]:
                    if pair_checks >= max_pair_checks:
                        exhausted = True
                        break
                    pair_checks += 1
                    if b not in neighbors[a]:
                        replacement = (removed, a, b)
                        break
                if replacement is not None or exhausted:
                    break
            if replacement is not None or exhausted:
                break
        if replacement is None:
            termination = "pair_check_limit" if exhausted else "local_optimum"
            break
        removed, a, b = replacement
        selected.remove(removed)
        selected.update((a, b))
        selected = fill(selected)
        passes += 1

    target = sorted(selected)
    if not _check_mis_constraints(target, n, graph)[2]:
        return None
    if source_feasible and len(target) <= original_size:
        return None
    return MISImprovement(
        node_set=target,
        original_size=original_size,
        improved_size=len(target),
        gain=len(target) - original_size,
        passes=passes,
        repaired=not source_feasible,
        source_feasible=source_feasible,
        pair_checks=pair_checks,
        termination=termination,
    )
