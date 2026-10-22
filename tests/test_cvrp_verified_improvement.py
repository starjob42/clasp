import copy

from clasp.rewards.co_tasks import _check_cvrp_constraints, _cvrp_total_distance
from clasp.rewards.cvrp_verified_improvement import improve_cvrp_routes

COORDS = [[0, 0], [1, 0], [0, 1], [1, 1]]
DEMANDS = [0, 1, 1, 1]


def test_repairs_duplicates_missing_capacity_and_invalid_nodes_without_mutation():
    routes = [[9, 1, 1, 3, 2]]
    original = copy.deepcopy(routes)
    result = improve_cvrp_routes(routes, COORDS, DEMANDS, 2)
    assert result is not None and result.repaired
    assert result.gain == 0  # Never compare an invalid objective with a feasible one.
    assert _check_cvrp_constraints(result.routes, COORDS, DEMANDS, 2)[2]
    assert routes == original


def test_missing_customer_cheapest_insertion():
    result = improve_cvrp_routes([[0, 1, 0]], COORDS, DEMANDS, 3)
    assert result is not None and result.repaired
    assert sorted(n for r in result.routes for n in r if n) == [1, 2, 3]


def test_feasible_crossing_route_strictly_improves():
    routes = [[0, 1, 2, 3, 0]]
    result = improve_cvrp_routes(routes, COORDS, DEMANDS, 3)
    assert result is not None and not result.repaired
    assert result.objective < _cvrp_total_distance(COORDS, routes)
    assert result.gain > 0
    assert 0 < result.passes <= 2


def test_optimal_source_is_not_repeated_as_teacher():
    assert improve_cvrp_routes([[0, 1, 3, 2, 0]], COORDS, DEMANDS, 3) is None


def test_invalid_instance_and_resource_limits_fail_closed():
    for demands, capacity in (([0, 4, 1, 1], 3), (DEMANDS, None), (DEMANDS, float("nan"))):
        assert improve_cvrp_routes([[0, 1, 0]], COORDS, demands, capacity) is None
    assert improve_cvrp_routes([], COORDS, DEMANDS, 3) is None
    assert improve_cvrp_routes([[0, 1, 0]], COORDS, DEMANDS, 3, max_nodes=3) is None


def test_zero_local_search_budget_still_repairs():
    result = improve_cvrp_routes([[0, 1, 0]], COORDS, DEMANDS, 2, max_passes=0)
    assert result is not None and result.repaired and result.passes == 0
