"""Prospective strict reward-gate regressions; no trained outcomes are read."""
import copy
import math
import random

import pytest

from clasp.baselines.external_baseline_dev import seal_reward_environment
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.baselines.sft_anchor_pilot import score_completion
from clasp.rewards.feasible_quality import verified_reward_metadata
from clasp.rewards.strict_raw_training import (
    gate_foarl_reward_heads, gate_rloo_rewards, strict_aligned_foarl_reward_batch,
    strict_reward_metadata, strict_rloo_reward_batch,
)


OP = [[[0, 0], [1, 0], [0, 1]], [0, 7, 8], 9]
CVRP = [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 2]
REFERENCES = {"op": "Route: [0, 1, 2], Objective: 15",
              "cvrp": "Routes: [[0, 1, 2, 0]], Objective: 3.41421356237"}


def aligned(task, completions, *, instance=None, reference=None):
    instance = copy.deepcopy(instance or (OP if task == "op" else CVRP))
    rows = [{"task_type": task, "answer": reference or REFERENCES[task],
             "instance_json": instance, "input": "", "prompt": "solve"}
            for _ in completions]
    return ["solve"] * len(rows), completions, rows


@pytest.fixture(autouse=True)
def sealed_environment(monkeypatch):
    from clasp.baselines.external_baseline_dev import REWARD_ENVIRONMENT
    for key in REWARD_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("CLASP_KL_REFERENCE_MODEL_PATH", raising=False)
    monkeypatch.delenv("CLASP_KL_REFERENCE_RECEIPT", raising=False)
    seal_reward_environment()


@pytest.mark.parametrize("task,completion,violation", [
    ("op", "Route: [0, 1.9]", "format_or_integer_type"),
    ("op", 'Route: [0, "1"]', "format_or_integer_type"),
    ("op", "Route: [0, True]", "format_or_integer_type"),
    ("cvrp", "Routes: [[0, 1.9, 2, 0]]", "format_or_integer_type"),
    ("cvrp", "Routes: [[0, True, 2, 0]]", "format_or_integer_type"),
    ("cvrp", "Routes: [[0, 1, 0, 2, 0]]", "interior_depot"),
])
def test_raw_invalid_canonical_false_positives_get_no_quality(task, completion, violation):
    prompts, completions, rows = aligned(task, [completion])
    old = verified_reward_metadata(task, prompts, completions, rows)[0]
    assert old["feasible"] and old["anchor_reward"] > 1
    new = strict_rloo_reward_batch(task, prompts, completions, rows)
    metadata = new["metadata"][0]
    strict = score_completion(task, completion, rows[0]["instance_json"], rows[0]["answer"])
    assert strict["feasible"] is metadata["feasible"] is False
    assert violation in metadata["violations"]
    assert metadata["objective"] is metadata["signed_gap"] is None
    assert metadata["quality_utility"] == 0
    assert -1 <= new["rewards"][0] < 0
    assert 0 < metadata["strict_constraint_fraction"] < 1
    foarl = strict_aligned_foarl_reward_batch(task, prompts, completions, rows)
    assert foarl["strict_feasible"] == [False]
    assert foarl["optimality"] == [0]
    assert 0 <= foarl["feasibility_fraction"][0] < 1


@pytest.mark.parametrize("task,completion,reference", [
    ("op", "Route: [0, 1]", "Route: [0, 1], Objective: 1e90"),
    ("op", "Route: [0, 1, 2], Objective: 0", "Route: [0, 1], Objective: 0"),
    ("cvrp", "Routes: [[0, 1, 2, 0]], Objective: 1e90",
     "Routes: [[0, 1, 0], [0, 2, 0]], Objective: 0"),
])
def test_valid_objectives_and_reference_ignore_printed_numbers(task, completion, reference):
    prompts, completions, rows = aligned(task, [completion], reference=reference)
    new = strict_rloo_reward_batch(task, prompts, completions, rows)
    score = score_completion(task, completion, rows[0]["instance_json"], reference)
    metadata = new["metadata"][0]
    assert metadata["feasible"]
    assert metadata["objective"] == score["objective"]
    assert metadata["reference"] == score["reference_objective"]
    assert metadata["signed_gap"] == score["signed_gap"]
    assert new["rewards"][0] == pytest.approx(1 + 1 / (1 + math.exp(8 * score["signed_gap"])))
    foarl = strict_aligned_foarl_reward_batch(task, prompts, completions, rows)
    expected = (2 * score["objective"] / max(.1, score["reference_objective"])
                if task == "op" else score["reference_objective"] / score["objective"])
    assert foarl["optimality"] == pytest.approx([expected])
    assert foarl["strict_feasible"] == [True]


def test_original_foarl_control_unchanged_but_successor_masks_interior_depot():
    prompts, completions, rows = aligned("cvrp", ["Routes: [[0, 1, 0, 2, 0]]"])
    kwargs = {key: [rows[0][key]] for key in ("task_type", "answer", "instance_json")}
    original = [head(prompts, completions, **kwargs)[0] for head in FOARL_F638_REWARD_FUNCS]
    assert original[0] > 0 and original[1] == pytest.approx(1)
    new = strict_aligned_foarl_reward_batch("cvrp", prompts, completions, rows)
    assert new["optimality"] == [0]
    assert new["feasibility_fraction"] == [7 / 8]
    assert "successor" in new["method_label"]
    assert [head(prompts, completions, **kwargs)[0] for head in FOARL_F638_REWARD_FUNCS] == original


def test_boolean_cvrp_quality_is_never_sent_to_released_numpy_indexer():
    prompts, completions, rows = aligned("cvrp", ["Routes: [[0, True, 2, 0]]"])
    kwargs = {key: [rows[0][key]] for key in ("task_type", "answer", "instance_json")}
    with pytest.raises(RuntimeError, match="optimality reward failed"):
        FOARL_F638_REWARD_FUNCS[0](prompts, completions, **kwargs)
    new = strict_aligned_foarl_reward_batch("cvrp", prompts, completions, rows)
    assert new["optimality"] == [0]
    assert new["feasibility_fraction"] == [1 / 8]


@pytest.mark.parametrize("completion", ["Route: [0,1]", "Route: [0, 1]"])
def test_valid_compact_and_spaced_op_grammar_share_strict_quality_and_full_feasibility(completion):
    args = aligned("op", [completion])
    new = strict_aligned_foarl_reward_batch("op", *args)
    assert new["strict_feasible"] == [True]
    assert new["optimality"] == pytest.approx([2 * 7 / 15])
    assert new["feasibility_fraction"] == [1]
    kwargs = {key: [args[2][0][key]] for key in ("task_type", "answer", "instance_json")}
    original = [head(args[0], args[1], **kwargs)[0] for head in FOARL_F638_REWARD_FUNCS]
    if completion == "Route: [0,1]":
        assert original == [0, 0]
    else:
        assert original == pytest.approx([2 * 7 / 15, 1])


def test_negative_invalid_rloo_grading_preserved_and_false_positive_graded():
    args = aligned("op", ["invalid", "Route: [0, 1, 1]", 'Route: [0, "1"]'])
    metadata = strict_reward_metadata("op", *args)
    assert gate_rloo_rewards(metadata, [-1, -.2, 1.8]) == [-1, -.2, -5 / 6]
    heads = gate_foarl_reward_heads(metadata, [10, 4, 8], [.05, .4, 1])
    assert heads["optimality"] == [0, 0, 0]
    assert heads["feasibility_fraction"] == [0, .4, 1 / 6]


def test_raw_physical_failures_share_gate_across_methods_and_keep_partial_credit():
    instance = [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 1]
    completions = ["invalid", "Routes: [[0, 1, 0]]", "Routes: [[0, 1, 2, 0]]",
                   "Routes: [[0, 1, 0], [0, 2, 0]]"]
    prompts, completions, rows = aligned("cvrp", completions, instance=instance,
        reference="Routes: [[0, 1, 0], [0, 2, 0]], Objective: 4")
    rloo = strict_rloo_reward_batch("cvrp", prompts, completions, rows)
    foarl = strict_aligned_foarl_reward_batch("cvrp", prompts, completions, rows)
    assert [row["feasible"] for row in rloo["metadata"]] == foarl["strict_feasible"] == [False, False, False, True]
    assert foarl["feasibility_fraction"][0] < foarl["feasibility_fraction"][2] < 1
    assert foarl["optimality"][:3] == [0, 0, 0]
    assert all(row["objective"] is row["signed_gap"] is None for row in rloo["metadata"][:3])
    assert "customer_coverage" in rloo["metadata"][1]["violations"]
    assert "capacity_exceeded" in rloo["metadata"][2]["violations"]


def test_raw_budget_excess_has_no_quality_even_if_printed_objective_is_good():
    instance = copy.deepcopy(OP)
    instance[2] = 1.1
    prompts, completions, rows = aligned("op", ["Route: [0, 1, 2], Objective: 100000"],
        instance=instance, reference="Route: [0, 1], Objective: 7")
    rloo = strict_rloo_reward_batch("op", prompts, completions, rows)
    assert "raw_budget_exceeded" in rloo["metadata"][0]["violations"]
    assert rloo["rewards"][0] < 0
    assert strict_aligned_foarl_reward_batch("op", prompts, completions, rows)["optimality"] == [0]


def test_tiny_positive_cvrp_reference_is_not_rounded_to_zero_in_upstream_adapter():
    instance = [[[0, 0], [1e-20, 0]], [0, 1], 1]
    args = aligned("cvrp", ["Routes: [[0, 1, 0]]"], instance=instance,
                   reference="Routes: [[0, 1, 0]], Objective: 0")
    new = strict_aligned_foarl_reward_batch("cvrp", *args)
    assert new["metadata"][0]["reference"] == pytest.approx(2e-20, abs=0)
    assert new["optimality"] == pytest.approx([1])


@pytest.mark.parametrize("task,reference", [
    ("op", 'Route: [0, "1"]'),
    ("op", "Route: [0, True]"),
    ("op", "Route: [0]"),
    ("cvrp", "Routes: [[0, 1, 0, 2, 0]]"),
    ("cvrp", "Routes: [[0, 1.9, 2, 0]]"),
])
def test_invalid_references_fail_before_any_sampled_quality(task, reference):
    args = aligned(task, ["invalid"], reference=reference)
    with pytest.raises(ValueError, match="Reference witness"):
        strict_rloo_reward_batch(task, *args)
    with pytest.raises(ValueError, match="Reference witness"):
        strict_aligned_foarl_reward_batch(task, *args)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "1"])
def test_nonfinite_or_nonnumeric_supplied_rewards_are_rejected(bad):
    metadata = strict_reward_metadata("op", *aligned("op", ["invalid"]))
    with pytest.raises(ValueError):
        gate_rloo_rewards(metadata, [bad])
    with pytest.raises(ValueError):
        gate_foarl_reward_heads(metadata, [bad], [.5])
    with pytest.raises(ValueError):
        gate_foarl_reward_heads(metadata, [0], [bad])


def test_alignment_task_and_invalid_quality_metadata_are_fail_closed():
    args = aligned("op", ["invalid"])
    with pytest.raises(ValueError, match="aligned"):
        strict_reward_metadata("op", [], args[1], args[2])
    with pytest.raises(ValueError, match="only OP"):
        strict_reward_metadata("tsp", *args)
    metadata = strict_reward_metadata("op", *args)
    metadata[0]["quality_utility"] = .9
    with pytest.raises(ValueError, match="Invalid outputs"):
        gate_rloo_rewards(metadata, [-1])


def test_returned_metadata_is_detached_from_rows_and_preserves_raw_text():
    args = aligned("op", ['Route: [0, "1"]'])
    original = copy.deepcopy(args)
    result = strict_rloo_reward_batch("op", *args)
    assert args == original
    assert result["metadata"][0]["strict_constraint_checks"]["integer_ids"] is False
    assert result["metadata"][0]["anchor_reward"] < 0


@pytest.mark.parametrize("task", ["op", "cvrp"])
def test_seeded_route_families_match_independent_physical_gate(task):
    """Exercise diagnostic admission over 1,500 routes per task, not policy draws."""
    rng = random.Random(20261014 if task == "op" else 20261015)
    leaves = [-1, 0, 1, 2, 3, True, False, 1.2, "1", None]
    malformed = ["", "[", "[[", "Route: [0,", "Routes: [[0, 1, 0], []]",
                 "Routes: [0, 1, 0]", "Routes: [[0, 1, 0]] extra [",
                 "Route: [0, nan]", "Route: [0, 1e999]", "Route: [0, (1,)]",
                 "Route: [0, {'x': 1}]", "Routes: [[0, {'x': 1}, 0]]",
                 "Route: [0, '\\x00']", "Routes: [[0, '1', 0]]",
                 "Route: []", "Routes: [[]]", "Routes: []"]
    generated = []
    for _ in range(1500):
        if task == "op":
            route = [rng.choice(leaves) for _ in range(rng.randrange(7))]
            generated.append("Route: " + repr(route))
        else:
            routes = [[rng.choice(leaves) for _ in range(rng.randrange(7))]
                      for _ in range(rng.randrange(5))]
            generated.append("Routes: " + repr(routes))
    # Guaranteed valid and near-boundary witnesses supplement mostly invalid
    # combinatorial draws, including redundant depots and exact duplicates.
    generated += (["Route: [0, 1]", "Route: [0, 1, 2]", "Route: [0, 2, 1]",
                   "Route: [0, 1, 1]", "Route: [0]", "Route: [1, 0]"] if task == "op" else
                  ["Routes: [[0, 1, 2, 0]]", "Routes: [[0, 1, 0], [0, 2, 0]]",
                   "Routes: [[0, 1, 0, 2, 0]]", "Routes: [[0, 1, 1, 2, 0]]",
                   "Routes: [[0, 0], [0, 1, 2, 0]]", "Routes: [[0, 1, 0]]"])
    tested_valid = tested_invalid = 0
    for completion in malformed + generated:
        if task == "op":
            instance = copy.deepcopy(OP)
            instance[2] = rng.choice([1 - 1e-9, 1, 1 + 1e-9,
                                     1 + math.sqrt(2) - 1e-9,
                                     1 + math.sqrt(2), 9])
            reference = "Route: [0, 1]"
        else:
            instance = copy.deepcopy(CVRP)
            instance[1] = [0, .6, .7]
            instance[2] = rng.choice([.7, .7 + 1e-9, 1.3 - 1e-9, 1.3, 2])
            reference = "Routes: [[0, 1, 0], [0, 2, 0]]"
        args = aligned(task, [completion], instance=instance, reference=reference)
        score = score_completion(task, completion, instance, reference)
        metadata = strict_reward_metadata(task, *args)[0]
        assert metadata["feasible"] is score["feasible"]
        assert metadata["objective"] == score["objective"]
        assert metadata["signed_gap"] == score["signed_gap"]
        if score["feasible"]:
            tested_valid += 1
            assert metadata["strict_constraint_fraction"] == 1
        else:
            tested_invalid += 1
            assert metadata["strict_constraint_fraction"] < 1
            assert metadata["quality_utility"] == 0
            assert metadata["objective"] is metadata["signed_gap"] is None
    assert tested_valid >= 2 and tested_invalid >= 1500
