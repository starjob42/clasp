"""Safety, strict-gain and bounded-search tests for MIS training teachers."""
import random
from itertools import combinations

import pytest

from clasp.rewards.co_tasks import _check_mis_constraints, _mis_maximalized_set
from clasp.rewards.verified_improvement import improve_mis_set


def independent(nodes, n, edges):
    """Small separate oracle, without calling the production verifier."""
    return (
        bool(nodes)
        and len(nodes) == len(set(nodes))
        and all(isinstance(v, int) and 0 <= v < n for v in nodes)
        and all(a not in nodes or b not in nodes for a, b in edges)
    )


def test_nonmaximal_feasible_set_reuses_existing_greedy_extension():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
    expected = _mis_maximalized_set([0], 5, edges)
    result = improve_mis_set([0], 5, edges, max_exchanges=0)
    assert result.node_set == expected[0] == [0, 2, 4]
    assert (result.original_size, result.improved_size, result.gain) == (1, 3, 2)
    assert result.source_feasible and not result.repaired
    assert result.passes == result.pair_checks == 0


def test_maximal_but_improvable_set_escapes_with_one_for_two():
    edges = {(0, 1), (0, 2)}
    assert _mis_maximalized_set([0], 3, edges) is None
    result = improve_mis_set([0], 3, edges)
    assert result.node_set == [1, 2]
    assert (result.passes, result.gain, result.pair_checks) == (1, 1, 1)
    assert result.termination == "local_optimum"
    assert _check_mis_constraints(result.node_set, 3, edges)[2]


def test_greedy_fill_runs_after_an_exchange():
    edges = {(0, leaf) for leaf in range(1, 6)}
    result = improve_mis_set([0], 6, edges, max_exchanges=1)
    assert result.node_set == [1, 2, 3, 4, 5]
    assert result.gain == 4 and result.passes == 1


def test_exchange_budget_stops_between_independent_improvements():
    edges = {(0, 1), (0, 2), (3, 4), (3, 5)}
    first = improve_mis_set([0, 3], 6, edges, max_exchanges=1)
    second = improve_mis_set([0, 3], 6, edges, max_exchanges=2)
    assert first.improved_size == 3 and first.passes == 1
    assert second.node_set == [1, 2, 4, 5] and second.passes == 2
    assert first.termination == second.termination == "exchange_limit"
    assert independent(first.node_set, 6, edges)


def test_optimal_feasible_set_emits_no_teacher():
    assert improve_mis_set([1, 2], 3, {(0, 1), (0, 2)}) is None
    assert improve_mis_set([0], 1, set()) is None


def test_one_for_two_local_optimum_is_not_claimed_globally_optimal():
    edges = {(a, b) for a in (0, 1) for b in (2, 3, 4)}
    assert independent([2, 3, 4], 5, edges)
    assert improve_mis_set([0, 1], 5, edges) is None


def test_repair_conflicts_invalid_ids_duplicates_and_noninteger_ids():
    edges = {(0, 1), (1, 2), (2, 3)}
    source = [0, 1, 1, -1, 10, "2", True]
    result = improve_mis_set(source, 4, edges)
    assert result.repaired and not result.source_feasible
    assert result.original_size == len(source)
    assert result.gain == len(result.node_set) - len(source) < 0
    assert independent(result.node_set, 4, edges)
    assert _check_mis_constraints(result.node_set, 4, edges)[2]
    assert source == [0, 1, 1, -1, 10, "2", True]


@pytest.mark.parametrize("source", [[], [0, 0], [-1], [0, 1]])
def test_repair_can_be_disabled(source):
    assert improve_mis_set(source, 2, {(0, 1)}, allow_repair=False) is None


def test_empty_parsed_set_can_be_repaired_to_nonempty_feasible_set():
    result = improve_mis_set([], 3, {(0, 1), (0, 2)})
    assert result.node_set == [1, 2]
    assert result.repaired and not result.source_feasible
    assert (result.original_size, result.gain) == (0, 2)


def test_pair_budget_is_global_and_can_return_an_existing_valid_repair():
    edges = set(combinations(range(8), 2))
    result = improve_mis_set([0, 0], 8, edges, max_pair_checks=3)
    assert result.node_set == [0]
    assert result.pair_checks == 3 and result.passes == 0
    assert result.termination == "pair_check_limit"
    assert result.repaired and result.gain == -1


def test_pair_budget_can_return_prior_feasible_improvement():
    edges = {(0, 1), (0, 2), (3, 4), (3, 5)}
    result = improve_mis_set([0, 3], 6, edges, max_pair_checks=1)
    assert result.gain == 1 and result.passes == 1 and result.pair_checks == 1
    assert result.termination == "pair_check_limit"
    assert independent(result.node_set, 6, edges)
    assert improve_mis_set([0, 3], 6, edges, max_pair_checks=0) is None


def test_deterministic_under_node_and_edge_order_and_no_input_mutation():
    edges = [(0, 1), (0, 2), (3, 4), (3, 5)]
    before = list(edges)
    left = improve_mis_set([3, 0], 6, edges)
    right = improve_mis_set([0, 3], 6, [(b, a) for a, b in reversed(edges)])
    assert left == right
    assert edges == before


@pytest.mark.parametrize("n,edges", [
    (None, set()), (0, set()), (-1, set()), (True, set()),
    (3, {(0, 3)}), (3, {(0, -1)}), (3, {(0, 0)}),
    (3, {(0, "1")}), (3, {(0, 1, 2)}),
])
def test_invalid_graph_metadata_is_rejected(n, edges):
    assert improve_mis_set([0], n, edges) is None


def test_graph_and_raw_input_size_limits_fail_closed():
    assert improve_mis_set([0], 4, set(), max_nodes=3) is None
    assert improve_mis_set([0], 3, {(0, 1), (0, 2)}, max_edges=1) is None
    assert improve_mis_set([0] * 5, 3, set(), max_nodes=4) is None


@pytest.mark.parametrize("kwargs", [
    {"max_exchanges": -1}, {"max_exchanges": 1.5},
    {"max_pair_checks": -1}, {"max_pair_checks": True},
    {"max_nodes": 0}, {"max_edges": -1}, {"allow_repair": "true"},
])
def test_bad_limits_raise_before_search(kwargs):
    with pytest.raises(ValueError):
        improve_mis_set([0], 2, set(), **kwargs)


def test_final_shared_verifier_is_required(monkeypatch):
    import clasp.rewards.verified_improvement as helper

    monkeypatch.setattr(helper, "_check_mis_constraints", lambda *_: (0, 4, False))
    assert helper.improve_mis_set([], 3, set()) is None


def test_small_graphs_against_exhaustive_feasibility_and_optimum_oracle():
    rng = random.Random(6102026)
    for _ in range(24):
        n = 7
        edges = {edge for edge in combinations(range(n), 2) if rng.random() < 0.35}
        feasible = [list(nodes) for size in range(1, n + 1)
                    for nodes in combinations(range(n), size) if independent(nodes, n, edges)]
        optimum = max(feasible, key=len)
        assert improve_mis_set(optimum, n, edges) is None
        for source in rng.sample(feasible, min(6, len(feasible))) + [[0, 0, -1]]:
            result = improve_mis_set(source, n, edges)
            if result is None:
                continue
            assert independent(result.node_set, n, edges)
            assert result.improved_size <= len(optimum)
            assert result.passes <= 2 and result.pair_checks <= 4096
            assert result.repaired or len(result.node_set) > len(source)
