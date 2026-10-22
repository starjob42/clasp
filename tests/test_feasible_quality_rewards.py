"""Verified training provenance and production quality coefficient identities."""
from __future__ import annotations

import json
import math

import pytest

from clasp.baselines.external_baseline_dev import REWARD_ENVIRONMENT
from clasp.rewards.feasible_quality import (
    METHODS, decompose_advantages, make_advantage_contract,
    validate_advantage_contract, verified_reward_metadata,
)

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
INSTANCES = {
    "tsp": SQUARE, "cvrp": [SQUARE, [0, 1, 1, 1], 3],
    "op": [SQUARE, [0, 1, 2, 3], 20],
    "mis": {"num_nodes": 4, "edges": [[0, 1]]},
    "jssp": [[0, 1, 1, 1], [1, 1, 0, 1]],
}
REFERENCE = {
    "tsp": "Route: [0, 2, 1, 3, 0], Objective: 999",
    "cvrp": "Routes: [[0, 1, 0], [0, 2, 0], [0, 3, 0]], Objective: 999",
    "op": "Route: [0, 2], Objective: 999", "mis": "Set: [0, 2], Objective: 999",
    "jssp": "Schedule: [[0, 1], [0, 1]], Makespan: 999",
}
BETTER = {
    "tsp": "Route: [0, 1, 2, 3, 0]", "cvrp": "Routes: [[0, 1, 2, 3, 0]]",
    "op": "Route: [0, 1, 2]", "mis": "Set: [0, 2, 3]",
    "jssp": "Schedule: [[0, 1], [1, 0]]",
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name, value in REWARD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def records(flags, gaps):
    return [{"feasible": flag, "signed_gap": gap} for flag, gap in zip(flags, gaps)]


def calculate(method, flags, gaps, base=None):
    return decompose_advantages([0.0] * len(flags) if base is None else base,
                                records(flags, gaps), make_advantage_contract(method))


@pytest.mark.parametrize("task", INSTANCES)
def test_metadata_recomputes_feasibility_objective_reference_and_signed_gap(task):
    row = {"task_type": task, "instance_json": json.dumps(INSTANCES[task]),
           "input": "instance", "answer": REFERENCE[task], "prompt": "solve"}
    completions = ["invalid", REFERENCE[task], BETTER[task]]
    result = verified_reward_metadata(task, ["solve"] * 3, completions, [row] * 3)
    assert result[0]["feasible"] is False
    assert result[0]["objective"] is result[0]["signed_gap"] is None
    assert result[0]["anchor_reward"] == -1
    assert result[1]["feasible"] is result[2]["feasible"] is True
    assert result[1]["signed_gap"] == 0 and result[2]["signed_gap"] < 0
    assert result[1]["anchor_reward"] == 1.5 < result[2]["anchor_reward"]
    assert all(0 < item["reference"] < 999 for item in result)
    assert len({item["completion_sha256"] for item in result}) == 3
    assert len({item["prompt_key"] for item in result}) == 1


@pytest.mark.parametrize("task", INSTANCES)
def test_invalid_training_reference_rejected_even_for_invalid_output(task):
    row = {"task_type": task, "instance_json": INSTANCES[task], "input": "x",
           "answer": "invalid", "prompt": "solve"}
    with pytest.raises(ValueError, match="training reference"):
        verified_reward_metadata(task, ["solve"], ["invalid"], [row])


def test_metadata_refuses_misaligned_and_wrong_task_rows():
    row = {"task_type": "mis", "instance_json": INSTANCES["mis"], "input": "x",
           "answer": REFERENCE["mis"], "prompt": "solve"}
    with pytest.raises(ValueError, match="aligned"):
        verified_reward_metadata("mis", ["solve"], [], [row])
    with pytest.raises(ValueError, match="mismatched"):
        verified_reward_metadata("op", ["solve"], ["invalid"], [row])


def test_cardinal_mixed_group_has_no_invalid_quality_and_negative_poor_feasible_credit():
    flags, gaps = [True, True] + [False] * 7, [-0.05, 0.05] + [None] * 7
    result = calculate("cardinal", flags, gaps, [1.0] * 9)
    assert result["quality"] == [0.25, -0.25] + [0.0] * 7
    assert result["combined"] == [1.0625, 0.9375] + [1.0] * 7
    assert sum(result["quality"]) == 0
    assert result["diagnostics"]["quality/invalid_nonzero_auxiliary_fraction"] == 0


@pytest.mark.parametrize("method", ("cardinal", "midrank"))
@pytest.mark.parametrize("count", (0, 1, 9))
def test_all_invalid_one_feasible_and_quality_ties_have_zero_added_credit(method, count):
    flags = [True] * count + [False] * (9 - count)
    gaps = [0.03] * count + [None] * (9 - count)
    assert calculate(method, flags, gaps)["quality"] == [0.0] * 9


def test_scalar_reweight_includes_ordinary_cross_feasibility_credit():
    flags, gaps = [True] + [False] * 8, [-0.05] + [None] * 8
    assert calculate("scalar_reweight", flags, gaps)["quality"] == [1.0] + [-0.125] * 8
    assert calculate("cardinal", flags, gaps)["quality"] == [0.0] * 9


def test_midrank_uses_signs_and_fixed_denominator_while_cardinal_preserves_distances():
    flags, gaps = [True] * 3 + [False] * 6, [-0.05, 0.0, 0.15] + [None] * 6
    assert calculate("midrank", flags, gaps)["quality"] == [0.25, 0.0, -0.25] + [0.0] * 6
    assert calculate("cardinal", flags, gaps)["quality"] == pytest.approx([0.625, 0.25, -0.875] + [0.0] * 6)


def test_all_feasible_cardinal_equals_scalar_quality_rloo():
    gaps = [i / 100 for i in range(9)]
    a = calculate("cardinal", [True] * 9, gaps)["quality"]
    b = calculate("scalar_reweight", [True] * 9, gaps)["quality"]
    assert a == pytest.approx(b, abs=1e-15)


def test_control_is_exact_base_including_signed_zero():
    base = [-0.0, 1e200, -1e200, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    result = calculate("control", [True] * 9, [i / 10 for i in range(9)], base)
    assert result["combined"] == base
    assert math.copysign(1.0, result["combined"][0]) == -1
    assert result["quality"] == [0.0] * 9


@pytest.mark.parametrize("changes", [
    {"method": "unknown"}, {"weight": 0.0}, {"gap_scale": 0.1},
    {"denominator": "feasible_siblings"}, {"num_generations": 8},
    {"scalar_invalid_quality": False}, {"extra": 1},
])
def test_immutable_advantage_contract_rejects_changed_recipe(changes):
    value = make_advantage_contract("cardinal")
    value.update(changes)
    with pytest.raises(ValueError):
        validate_advantage_contract(value)


@pytest.mark.parametrize("bad", [
    {"feasible": 1, "signed_gap": 0.0},
    {"feasible": False, "signed_gap": 0.0},
    {"feasible": True, "signed_gap": None},
    {"feasible": True, "signed_gap": math.nan},
    {"feasible": True, "signed_gap": math.inf},
    {"feasible": True, "signed_gap": True},
])
def test_noncanonical_or_nonfinite_metadata_is_rejected(bad):
    rows = records([False] * 9, [None] * 9)
    rows[0] = bad
    with pytest.raises(ValueError):
        decompose_advantages([0.0] * 9, rows, make_advantage_contract("cardinal"))


@pytest.mark.parametrize("method", METHODS)
def test_multiple_groups_do_not_share_quality_credit(method):
    flags = [True] * 9 + [False] * 9
    gaps = [i / 100 for i in range(9)] + [None] * 9
    result = calculate(method, flags, gaps)
    assert result["quality"][9:] == [0.0] * 9
    assert result["quality"][:9] == calculate(method, flags[:9], gaps[:9])["quality"]


def test_exact_categorical_score_expectation_matches_p_squared_conditional_quality_gradient():
    probabilities = [0.4, 0.35, 0.25]
    feasible = [False, True, True]
    gaps = [None, 0.05, -0.025]
    q = [0.0, -1.0, 0.5]
    p = sum(probabilities[1:])
    mu = sum(probabilities[i] * q[i] for i in range(3))
    expected = [p * probabilities[j] * (q[j] - mu)
                - mu * probabilities[j] * (int(feasible[j]) - p) for j in range(3)]
    # The first derivative term is grad E[fq]; q_invalid=0 already includes f.
    measured = [0.0] * 3
    for n0 in range(10):
        for n1 in range(10 - n0):
            counts = [n0, n1, 9 - n0 - n1]
            ids = [i for i, count in enumerate(counts) for _ in range(count)]
            probability = math.factorial(9)
            for count, prob in zip(counts, probabilities):
                probability *= prob ** count / math.factorial(count)
            coeff = calculate("cardinal", [feasible[i] for i in ids], [gaps[i] for i in ids])["quality"]
            for j in range(3):
                measured[j] += probability * sum(a * (int(i == j) - probabilities[j])
                                                 for a, i in zip(coeff, ids)) / 9
    assert measured == pytest.approx(expected, abs=1e-13)
    assert measured[0] == pytest.approx(0.0, abs=1e-14)
