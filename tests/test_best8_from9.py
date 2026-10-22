"""Exact finite-action expectation checks; no policy output or jobs are read."""
import itertools
import math
import copy

import pytest

from clasp.rewards.best8_from9 import best8_from9_coefficients, strict_best8_from9_coefficients
from clasp.rewards.feasible_quality import decompose_advantages, make_advantage_contract
from clasp.rewards.strict_raw_training import strict_reward_metadata


def test_unique_best_centered_native_mean_loss_coefficients_and_h9():
    utility = [.8, .5, .2, 0, 0, 0, 0, 0, 0]
    result = best8_from9_coefficients(utility)
    assert result["coefficients"] == pytest.approx([2.4] + [-.3] * 8)
    assert math.fsum(result["coefficients"]) == 0
    brute_h9 = math.fsum(max(utility[:j] + utility[j + 1:]) for j in range(9)) / 9
    assert result["groups"][0]["h9"] == pytest.approx(brute_h9)
    assert result["coefficients"] == pytest.approx([
        9 * (brute_h9 - max(utility[:i] + utility[i + 1:])) for i in range(9)])
    assert result["sampling_score_identity_applies_to_current_pilot"] is False


@pytest.mark.parametrize("utility", [[0] * 9, [.8, .8] + [0] * 7, [.4] * 9])
def test_top_ties_and_all_invalid_have_no_arbitrary_winner(utility):
    result = best8_from9_coefficients(utility)
    assert result["coefficients"] == [0] * 9
    assert not result["groups"][0]["active"]


def test_single_feasible_gets_signal_and_invalid_responses_receive_only_negative_baseline():
    result = best8_from9_coefficients([.5] + [0] * 8)
    assert result["coefficients"] == [4] + [-.5] * 8
    assert result["groups"][0]["h9"] == 4 / 9


def test_focal_baseline_does_not_depend_on_the_focal_response():
    row = [.8, .5, .2, 0, 0, 0, 0, 0, 0]
    original = best8_from9_coefficients(row)
    for focal in range(9):
        for changed in (0, .1, .7, 1):
            modified = list(row)
            modified[focal] = changed
            assert best8_from9_coefficients(modified)["focal_independent_baselines"][focal] == original["focal_independent_baselines"][focal]


def test_permutation_equivariance_and_groups_do_not_cross_prompt_boundaries():
    row = [.8, .5, .2, 0, 0, 0, 0, 0, 0]
    permutation = [3, 0, 8, 2, 6, 1, 7, 4, 5]
    expected = best8_from9_coefficients(row)["coefficients"]
    result = best8_from9_coefficients([row[i] for i in permutation] + [0] * 9)
    assert result["coefficients"][:9] == [expected[i] for i in permutation]
    assert result["coefficients"][9:] == [0] * 9


@pytest.mark.parametrize("values", [[], [0] * 8, [0] * 10, [True] * 9,
                                   [math.nan] * 9, [math.inf] * 9,
                                   [-.1] * 9, [1.1] * 9, [".5"] * 9])
def test_rejects_unbounded_or_partial_groups(values):
    with pytest.raises(ValueError):
        best8_from9_coefficients(values)


def test_exact_iid_finite_action_score_expectation_targets_best8_and_cardinal_conditional_quality():
    """Enumerate all3**9 groups and compare against analytic softmax derivatives."""
    probabilities = [.2, .3, .5]
    utility_table = [0, .4, .8]
    flags, q_table = [False, True, True], [0, -2, 1]
    best8_gradient = [0.0] * 3
    cardinal_gradient = [0.0] * 3
    expected_h9 = 0.0
    contract = make_advantage_contract("cardinal")
    for actions in itertools.product(range(3), repeat=9):
        probability = math.prod(probabilities[action] for action in actions)
        utility = [utility_table[action] for action in actions]
        result = best8_from9_coefficients(utility)
        expected_h9 += probability * result["groups"][0]["h9"]
        metadata = [{"feasible": flags[action], "signed_gap": -.05 * q_table[action]
                     if flags[action] else None} for action in actions]
        cardinal = decompose_advantages([0] * 9, metadata, contract)["quality"]
        for i, action in enumerate(actions):
            for axis, p in enumerate(probabilities):
                score = float(action == axis) - p
                best8_gradient[axis] += probability * result["coefficients"][i] * score / 9
                cardinal_gradient[axis] += probability * cardinal[i] * score / 9
    p0, p1, _ = probabilities
    expected_j8 = .8 - .4 * (p0 + p1) ** 8 - .4 * p0 ** 8
    raw_partial = [-3.2 * ((p0 + p1) ** 7 + p0 ** 7),
                   -3.2 * (p0 + p1) ** 7, 0]
    weighted_partial = math.fsum(p * derivative for p, derivative in zip(probabilities, raw_partial))
    analytic_best8 = [p * (derivative - weighted_partial)
                      for p, derivative in zip(probabilities, raw_partial)]
    assert expected_h9 == pytest.approx(expected_j8, abs=1e-13)
    assert best8_gradient == pytest.approx(analytic_best8, abs=1e-13)
    f = math.fsum(p * flag for p, flag in zip(probabilities, flags))
    m = math.fsum(p * q for p, q in zip(probabilities, q_table))
    grad_f = [p * (float(flag) - f) for p, flag in zip(probabilities, flags)]
    grad_m = [p * (q - m) for p, q in zip(probabilities, q_table)]
    # F*dM-M*dF=F**2*dE[q|feasible]; this is not a direct feasibility objective.
    analytic_cardinal = [f * dm - m * df for dm, df in zip(grad_m, grad_f)]
    assert cardinal_gradient == pytest.approx(analytic_cardinal, abs=1e-12)


def test_strict_metadata_wrapper_does_not_treat_coerced_id_as_feasible_utility():
    instance = [[[0, 0], [1, 0], [0, 1]], [0, 7, 8], 9]
    completions = ["Route: [0, 1, 2]"] + ['Route: [0, "1"]'] * 8
    rows = [{"task_type": "op", "instance_json": instance, "input": "",
             "prompt": "solve", "answer": "Route: [0, 1, 2]"} for _ in range(9)]
    metadata = strict_reward_metadata("op", ["solve"] * 9, completions, rows)
    assert metadata[0]["quality_utility"] == .5
    assert all(not row["feasible"] and row["quality_utility"] == 0 for row in metadata[1:])
    result = strict_best8_from9_coefficients(metadata)
    assert result["coefficients"] == [4] + [-.5] * 8


def strict_group(task="op", completions=None, prompt="solve"):
    instance = [[[0, 0], [1, 0], [0, 1]], [0, 7, 8] if task == "op" else [0, 1, 1], 9 if task == "op" else 2]
    answer = "Route: [0, 1, 2]" if task == "op" else "Routes: [[0, 1, 2, 0]]"
    completions = completions or [answer] * 9
    rows = [{"task_type": task, "instance_json": instance, "input": "", "prompt": prompt,
             "answer": answer} for _ in completions]
    return strict_reward_metadata(task, [prompt] * len(rows), completions, rows)


def test_strict_wrapper_rejects_cross_prompt_group_but_allows_separate_complete_groups():
    first, second = strict_group(prompt="first"), strict_group(prompt="second")
    assert first[0]["prompt_key"] != second[0]["prompt_key"]
    with pytest.raises(ValueError, match="one prompt"):
        strict_best8_from9_coefficients(first[:8] + second[:1])
    combined = strict_best8_from9_coefficients(first + second)
    assert combined["coefficients"] == [0] * 18


@pytest.mark.parametrize("key", [None, "", "x", "0" * 63, "g" * 64, "A" * 64, 123])
def test_strict_wrapper_rejects_missing_or_malformed_prompt_keys(key):
    metadata = strict_group()
    metadata[4]["prompt_key"] = key
    with pytest.raises(ValueError, match="64-hex prompt keys"):
        strict_best8_from9_coefficients(metadata)


def test_strict_wrapper_rejects_absent_prompt_field():
    metadata = strict_group()
    metadata[4].pop("prompt_key")
    with pytest.raises(ValueError, match="64-hex prompt keys"):
        strict_best8_from9_coefficients(metadata)


def test_strict_wrapper_rejects_cross_task_group_even_if_prompt_key_is_forged_equal():
    op, cvrp = strict_group(), strict_group("cvrp")
    cvrp[0]["prompt_key"] = op[0]["prompt_key"]
    with pytest.raises(ValueError, match="one task"):
        strict_best8_from9_coefficients(op[:8] + cvrp[:1])
    # Separate complete task groups remain legal; grouping is prompt-local.
    assert len(strict_best8_from9_coefficients(op + strict_group("cvrp"))["groups"]) == 2


@pytest.mark.parametrize("task", ["tsp", None, ""])
def test_strict_wrapper_rejects_invalid_task_metadata(task):
    metadata = strict_group()
    metadata[4]["task"] = task
    with pytest.raises(ValueError, match="metadata contract"):
        strict_best8_from9_coefficients(metadata)


@pytest.mark.parametrize("utility", [.7, .5 + 2e-12, math.nan, math.inf, True])
def test_strict_wrapper_rejects_forged_or_nonfinite_utility(utility):
    metadata = strict_group()
    metadata[0]["quality_utility"] = utility
    with pytest.raises(ValueError):
        strict_best8_from9_coefficients(metadata)


def test_strict_wrapper_uses_recomputed_utility_with_explicit_absolute_tolerance():
    metadata = strict_group()
    metadata[0]["quality_utility"] += 1e-13
    metadata[1]["quality_utility"] -= 1e-13
    before = copy.deepcopy(metadata)
    # Stored rounding must not create a false winner or a microscopic margin.
    result = strict_best8_from9_coefficients(metadata)
    assert result["coefficients"] == [0] * 9
    assert result["groups"][0]["top_utility"] == .5
    assert metadata == before


@pytest.mark.parametrize("completions", [["invalid"] * 9,
    ["Route: [0, 1, 2]"] * 9,
    ["Route: [0, 1, 2]"] * 2 + ["invalid"] * 7])
def test_strict_wrapper_all_invalid_ties_and_duplicate_valid_actions_remain_legal(completions):
    metadata = strict_group(completions=completions)
    result = strict_best8_from9_coefficients(metadata)
    assert result["coefficients"] == [0] * 9
    assert not result["groups"][0]["active"]


@pytest.mark.parametrize("count", [0, 8, 10])
def test_strict_wrapper_rejects_incomplete_groups_before_metadata_admission(count):
    metadata = strict_group()
    values = metadata[:count] if count <= 9 else metadata + metadata[:1]
    with pytest.raises(ValueError, match="complete nonempty groups of nine"):
        strict_best8_from9_coefficients(values)
