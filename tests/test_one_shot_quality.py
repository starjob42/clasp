"""Strict physical one-shot reward properties; no models or policy outcomes."""
import copy
import itertools
import math
import random

import pytest

from clasp.rewards import one_shot_quality as q

OP = [[[0, 0], [1, 0], [0, 1]], [0, 7, 8], 9]
CVRP = [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 2]


def args(task, replies, reference=None, instance=None):
    reference = reference or ("Route: [0, 1, 2]" if task == "op" else "Routes: [[0, 1, 2, 0]]")
    instance = copy.deepcopy(instance or (OP if task == "op" else CVRP))
    return ["solve"] * len(replies), replies, [{"task_type": task, "prompt": "solve", "input": "",
        "answer": reference, "instance_json": instance} for _ in replies]


def test_valid_op_linear_prize_and_cvrp_linear_cost_ignore_claimed_objectives():
    op = q.reward_batch("op", *args("op", ["Route: [0, 1], Objective: 999", "Route: [0,1,2], Objective: -999"]))
    assert op["rewards"] == [1 + 7/15, 2]
    assert [r["objective"] for r in op["metadata"]] == [7, 15]
    cvrp = q.reward_batch("cvrp", *args("cvrp", ["Routes: [[0,1,2,0]]", "Routes: [[0,1,0],[0,2,0]]"]))
    assert cvrp["rewards"] == pytest.approx([2-(2+math.sqrt(2))/4, 1])
    assert cvrp["rewards"][0] > cvrp["rewards"][1]
    assert cvrp["metadata"][0]["physical_scale"]["value"] == 4


@pytest.mark.parametrize("task,text", [("op", "Route: [0,1.1]"), ("op", "Route: [0,True]"),
    ("op", 'Route: [0,"1"]'), ("op", "Route: [0,1,1]"), ("op", "invalid"),
    ("cvrp", "Routes: [[0,1,0,2,0]]"), ("cvrp", "Routes: [[0,1.9,2,0]]"),
    ("cvrp", "Routes: [[0,True,2,0]]"), ("cvrp", "Routes: [[0,1,0]]")])
def test_invalid_raw_outputs_never_receive_objective_or_quality(task, text):
    result = q.reward_batch(task, *args(task, [text])); row = result["metadata"][0]
    assert not row["feasible"] and row["objective"] is None and row["utility"] == row["quality_head"] == 0
    assert -1 <= result["rewards"][0] < 0 and result["quality_head"] == [0]


def test_same_reward_one_and_two_head_forms_are_explicit_not_released_foarl():
    replies = ["Route: [0,1]", "Route: [0,1,2]", "invalid", "Route: [0,1,1]"]
    result = q.reward_batch("op", *args("op", replies))
    assert result["rewards"] == [a+b for a,b in zip(result["quality_head"], result["validity_head"])]
    assert result["metadata"][2]["validity_head"] == -1


def test_physical_scale_is_reference_independent_and_op_depot_has_zero_prize():
    first = q.reward_batch("op", *args("op", ["Route: [0,1]"]))
    second = q.reward_batch("op", *args("op", ["Route: [0,1]"], reference="Route: [0,1]"))
    assert first["rewards"] == second["rewards"] and first["metadata"][0]["reference"] != second["metadata"][0]["reference"]
    bad = copy.deepcopy(OP); bad[1][0] = 9
    with pytest.raises(ValueError): q.physical_scale("op", bad)


@pytest.mark.parametrize("task,instance", [("op", [[[0,0],[1,1]],[0,0],2]),
    ("cvrp", [[[0,0],[0,0]],[0,1],2])])
def test_degenerate_zero_bounds_fail_before_training(task, instance):
    with pytest.raises(ValueError, match="positive finite"): q.physical_scale(task, instance)


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), -1, True, 4.01])
def test_invalid_or_materially_exceeded_bounds_are_rejected_not_hidden_by_clipping(objective):
    with pytest.raises(ValueError): q.physical_utility("cvrp", objective, q.physical_scale("cvrp", CVRP))


def test_only_declared_roundoff_overshoot_is_projected():
    scale = q.physical_scale("cvrp", CVRP)
    assert q.physical_utility("cvrp", 4 * (1+5e-13), scale) == 0
    with pytest.raises(ValueError): q.physical_utility("cvrp", 4 * (1+2e-12), scale)


def test_triangle_bound_and_monotonic_utility_randomized_complete_route_permutations():
    rng = random.Random(63016)
    for _ in range(15):
        coords = [[0,0]] + [[rng.randint(-10,10),rng.randint(-10,10)] for _ in range(5)]
        instance = [coords, [0,1,1,1,1,1], 5]
        scale = q.physical_scale("cvrp", instance)
        measures = []
        for permutation in itertools.islice(itertools.permutations(range(1,6)), 30):
            route = [0,*permutation,0]
            cost = math.fsum(math.dist(coords[a],coords[b]) for a,b in zip(route,route[1:]))
            assert cost <= scale["value"] * (1+q.ROUND_OFF_TOLERANCE)
            measures.append((cost,q.physical_utility("cvrp",cost,scale)))
        ordered = sorted(measures)
        assert all(a[1] >= b[1]-1e-15 for a,b in zip(ordered,ordered[1:]))


def test_current_scope_is_explicitly_op_cvrp_and_greedy_gain_unproved():
    spec = q.contract()
    assert spec["tasks"] == ["op","cvrp"] and not spec["greedy_improvement_guaranteed"]
    assert spec["best8_auxiliary"] == spec["cardinal_auxiliary"] == 0
    assert not spec["repair"] and not spec["search"]
    with pytest.raises(ValueError): q.physical_scale("tsp", CVRP)
