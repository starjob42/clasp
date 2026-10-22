"""Canonical quality direction and invalid-diagnostic masking counterexamples."""
from __future__ import annotations

import json
import math

import pytest

from clasp.baselines.external_baseline_dev import REWARD_ENVIRONMENT, _graded_reward
from clasp.rewards.quality_signal import (
    QUALITY_SLOPE,
    REWARD_MODES,
    _sigmoid_quality,
    reward_batch,
    validate_training_reference,
)

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
INSTANCES = {
    "tsp": SQUARE,
    "cvrp": [SQUARE, [0, 1, 1, 1], 3],
    "op": [SQUARE, [0, 1, 2, 3], 20],
    "mis": {"num_nodes": 4, "edges": [[0, 1]]},
    "jssp": [[0, 1, 1, 1], [1, 1, 0, 1]],
}
REFERENCE = {
    "tsp": "Route: [0, 2, 1, 3, 0], Objective: 999",
    "cvrp": "Routes: [[0, 1, 0], [0, 2, 0], [0, 3, 0]], Objective: 999",
    "op": "Route: [0, 2], Objective: 999",
    "mis": "Set: [0, 2], Objective: 999",
    "jssp": "Schedule: [[0, 1], [0, 1]], Makespan: 999",
}
BETTER = {
    "tsp": "Route: [0, 1, 2, 3, 0]",
    "cvrp": "Routes: [[0, 1, 2, 3, 0]]",
    "op": "Route: [0, 1, 2]",
    "mis": "Set: [0, 2, 3]",
    "jssp": "Schedule: [[0, 1], [1, 0]]",
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name, value in REWARD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def batch(task, completions, *, mode="reference_sigmoid_v2", instance=None, answer=None):
    count = len(completions)
    kwargs = dict(task_type=[task] * count, input=["instance"] * count,
                  instance_json=[json.dumps(INSTANCES[task] if instance is None else instance)] * count,
                  answer=[REFERENCE[task] if answer is None else answer] * count)
    return reward_batch(task, ["solve"] * count, completions, mode=mode, **kwargs)


@pytest.mark.parametrize("task", INSTANCES)
def test_all_tasks_recompute_reference_and_preserve_better_than_reference_sign(task):
    reference = validate_training_reference(task, REFERENCE[task], INSTANCES[task])
    better = validate_training_reference(task, BETTER[task], INSTANCES[task])
    assert 0 < reference < 999 and 0 < better < 999
    direction = 1 if task in ("tsp", "cvrp", "jssp") else -1
    gap = direction * (better - reference) / reference
    assert gap < 0
    values = batch(task, ["invalid", REFERENCE[task], BETTER[task]])
    assert values[0] == -1
    assert values[1] == 1.5
    assert values[2] == pytest.approx(1 + 1 / (1 + math.exp(QUALITY_SLOPE * gap)))
    assert 1.5 < values[2] <= 2


@pytest.mark.parametrize("task", INSTANCES)
@pytest.mark.parametrize("mode", REWARD_MODES)
def test_claimed_objectives_do_not_change_reward_or_reference(task, mode):
    claimed = REFERENCE[task]
    plain = claimed.split(", Objective:")[0].split(", Makespan:")[0]
    assert batch(task, [claimed], mode=mode) == batch(task, [plain], mode=mode, answer=plain)
    assert validate_training_reference(task, claimed, INSTANCES[task]) == validate_training_reference(
        task, plain, INSTANCES[task])


@pytest.mark.parametrize("task", INSTANCES)
def test_legacy_is_exact_parent_reward_and_corrected_mode_preserves_feasible_values(task):
    completions = ["invalid", REFERENCE[task], BETTER[task]]
    kwargs = dict(task_type=[task] * 3, input=["instance"] * 3,
                  instance_json=[json.dumps(INSTANCES[task])] * 3, answer=[REFERENCE[task]] * 3)
    parent = _graded_reward(task, ["solve"] * 3, completions, kwargs)
    assert reward_batch(task, ["solve"] * 3, completions, mode="legacy", **kwargs) == parent
    assert reward_batch(task, ["solve"] * 3, completions, mode="graded_v2", **kwargs) == parent


@pytest.mark.parametrize("task", INSTANCES)
def test_bad_training_reference_rejected_even_when_every_sample_is_invalid(task):
    with pytest.raises(ValueError, match="training reference"):
        batch(task, ["invalid", "invalid"], answer="invalid")
    # Historical behavior stays deliberately distinct from reference calibration.
    assert batch(task, ["invalid"], mode="legacy", answer="invalid") == [-1]


@pytest.mark.parametrize("task,instance,answer", [
    ("tsp", [[0, 0], [0, 0]], "Route: [0, 1, 0]"),
    ("cvrp", [[[0, 0], [0, 0]], [0, 1], 1], "Routes: [[0, 1, 0]]"),
    ("op", [[[0, 0], [1, 0]], [0, 0], 2], "Route: [0, 1]"),
    ("mis", {"num_nodes": 2, "edges": []}, "Set: []"),
    ("jssp", [[0, 0]], "Schedule: [[0]]"),
])
def test_zero_reference_objective_is_rejected(task, instance, answer):
    with pytest.raises(ValueError):
        validate_training_reference(task, answer, instance)
    with pytest.raises(ValueError):
        batch(task, ["invalid"], instance=instance, answer=answer)


@pytest.mark.parametrize("task,instance", [
    ("tsp", [[0, 0], [math.inf, 0]]),
    ("cvrp", [SQUARE, [0, 1, 1, 1], math.inf]),
    ("op", [SQUARE, [0, 1, math.nan, 1], 20]),
    ("mis", {"num_nodes": math.inf, "edges": []}),
    ("jssp", [[0, math.inf]]),
])
def test_nonfinite_instance_cannot_supply_a_reference(task, instance):
    with pytest.raises(ValueError):
        validate_training_reference(task, REFERENCE[task], instance)


@pytest.mark.parametrize("mode", ("graded_v2", "reference_sigmoid_v2"))
def test_op_invalid_id_cannot_hide_distance_budget_violation(mode):
    instance = [[[0, 0], [10, 0], [20, 0]], [0, 1, 1], 11]
    answer = "Route: [0, 1]"
    original = [0, 1, 2]
    corruptions = [original[:i] + [bad] + original[i:]
                   for i in range(4) for bad in (-1, 3)]
    texts = [f"Route: {route}" for route in [original, *corruptions]]
    values = batch("op", texts, mode=mode, instance=instance, answer=answer)
    legacy = batch("op", texts, mode="legacy", instance=instance, answer=answer)
    assert legacy[0] < legacy[-1]  # the original independent counterexample
    assert all(-1 <= value < 0 and value <= values[0] for value in values[1:])
    assert batch("op", ["Route: [0]", "Route: [0, 99]"], mode=mode,
                 instance=instance, answer=answer) == [-1, -1]


@pytest.mark.parametrize("mode", ("graded_v2", "reference_sigmoid_v2"))
def test_op_duplicate_insertion_does_not_improve_an_overbudget_route(mode):
    instance = [[[0, 0], [10, 0], [20, 0]], [0, 1, 1], 11]
    routes = [[0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 1]]
    rewards = batch("op", [f"Route: {r}" for r in routes], mode=mode,
                    instance=instance, answer="Route: [0, 1]")
    assert all(value < rewards[0] for value in rewards[1:])


@pytest.mark.parametrize("mode", ("graded_v2", "reference_sigmoid_v2"))
def test_jssp_duplicate_invalid_id_extra_row_cannot_hide_known_cycle(mode):
    schedules = [[[1, 0], [0, 1]], [[1, 0, 0], [0, 1]],
                 [[1, -1, 0], [0, 1]], [[1, 0], [0, 1, 2]],
                 [[1, 0], [0, 1], [0, 1]], [[0, 1, 0], [0, 1]],
                 [[0, 1], [1, 0], [0, 1]]]
    texts = [f"Schedule: {s}" for s in schedules]
    old = batch("jssp", texts, mode="legacy")
    assert old[0] == -.5 and old[1] == -.2
    new = batch("jssp", texts, mode=mode)
    assert new[0] == -.5
    assert all(-1 <= r < new[0] for r in new[1:])


def test_jssp_unknown_topology_uses_conservative_penalty_not_false_zero_cycle():
    rewards = batch("jssp", ["Schedule: [[1, 0], [0, 1]]", "Schedule: [[1], [0, 1]]"],
                    mode="graded_v2")
    assert rewards[1] < rewards[0]


@pytest.mark.parametrize("task,completions", [
    ("tsp", ["Route: [0, 1, 0]", "Route: [0, 1, 2, 0]"]),
    ("cvrp", ["Routes: [[0, 1, 0]]", "Routes: [[0, 1, 2, 0]]"]),
    ("op", ["Route: [0, 99, 1]", "Route: [0, 99, 98, 1]"]),
    ("mis", ["Set: [0, 1]", "Set: [0, 1, 1]"]),
    ("jssp", ["Schedule: [[1, 0], [0, 1]]", "Schedule: [[1, 0, 0], [0, 1]]"]),
])
def test_all_invalid_groups_retain_real_verifier_differences(task, completions):
    values = batch(task, completions)
    assert all(-1 < v < 0 for v in values)
    assert values[0] != values[1]


@pytest.mark.parametrize("task", INSTANCES)
def test_quality_direction_and_local_slope_are_explicit_and_finite(task):
    values = [_sigmoid_quality(task, value, 100) for value in (0, 95, 100, 105, 1e308)]
    assert all(math.isfinite(v) and 1 <= v <= 2 for v in values)
    expected = sorted(values, reverse=task in ("tsp", "cvrp", "jssp"))
    assert values == expected
    assert values[2] == 1.5
    epsilon = 1e-6
    direction = -1 if task in ("tsp", "cvrp", "jssp") else 1
    derivative = (_sigmoid_quality(task, 100 * (1 + epsilon), 100)
                  - _sigmoid_quality(task, 100 * (1 - epsilon), 100)) / (2 * epsilon)
    assert derivative == pytest.approx(direction * QUALITY_SLOPE / 4, rel=1e-7)
    assert math.isfinite(_sigmoid_quality(task, 1e308, 1e-308))


def test_new_mode_converts_generated_numeric_overflow_to_invalid_not_exception():
    assert batch("tsp", ["Route: [0, 1e309, 2, 3, 0]"]) == [-1]
    with pytest.raises(OverflowError):
        batch("tsp", ["Route: [0, 1e309, 2, 3, 0]"], mode="legacy")


def test_zero_candidate_prize_is_feasible_but_zero_reference_is_not():
    instance = [[[0, 0], [1, 0], [2, 0]], [0, 0, 1], 4]
    value = batch("op", ["Route: [0, 1]"], instance=instance, answer="Route: [0, 2]")[0]
    assert value == pytest.approx(1 + 1 / (1 + math.exp(8)))
    assert 1 < value < 1.5


def test_reference_never_broadcast_from_another_training_row():
    with pytest.raises(ValueError, match="training reference"):
        reward_batch("mis", ["p", "p"], ["invalid", "invalid"], mode="reference_sigmoid_v2",
                     task_type=["mis"] * 2, answer=[REFERENCE["mis"], None],
                     instance_json=[INSTANCES["mis"]] * 2, input=["instance"] * 2)


def test_mode_alignment_task_and_environment_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        batch("tsp", ["invalid"], mode="adaptive")
    with pytest.raises(ValueError, match="one answer"):
        reward_batch("tsp", ["p"], ["invalid"], mode="graded_v2", task_type=["tsp"])
    with pytest.raises(ValueError, match="counts differ"):
        reward_batch("tsp", [], ["invalid"], mode="graded_v2", task_type=["tsp"],
                     answer=[REFERENCE["tsp"]], instance_json=[SQUARE], input=["instance"])
    monkeypatch.setenv("GRPO_GRADED_REWARD", "0")
    with pytest.raises(ValueError, match="environment changed"):
        batch("mis", ["invalid"])
