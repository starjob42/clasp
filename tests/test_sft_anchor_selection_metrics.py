"""Pre-outcome controls for reference-saturation-resistant selection quantities."""
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/rloo_sft_anchor_20261013/selection_metrics.py"
SPEC = importlib.util.spec_from_file_location("anchor_signed_selection", SOURCE)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def score(gap, feasible=True, identifier="a"):
    return {"instance_id": identifier, "reference_id": identifier + "-ref", "feasible": feasible,
            "signed_gap": gap if feasible else None}


def test_negative_gap_gain_remains_visible_when_u10_ties_at_one():
    baseline, improved = score(-.10), score(-.20)
    u10 = lambda row: max(0, 1 - max(0, row["signed_gap"]) / .1)
    assert u10(baseline) == u10(improved) == 1
    assert metrics.sigmoid_quality(improved) > metrics.sigmoid_quality(baseline) > .5
    assert metrics.sigmoid_quality(score(0)) == .5
    assert metrics.sigmoid_quality(score(.10)) == pytest.approx(1 / (1 + math.exp(.8)))


def test_failures_contribute_zero_and_all_instances_keep_equal_weight():
    rows = [score(0), score(0, False, "b")]
    summary = metrics.summarize_selected(rows)
    assert summary["N"] == 2 and summary["n"] == 1
    assert summary["feasibility_percent"] == 50 and summary["q_sigmoid_percent"] == 25


def test_macro_weights_tasks_equally_despite_different_row_counts():
    result = metrics.macro_quality({"op": [score(0)], "cvrp": [score(0, False)] * 3})
    assert result["macro_q_sigmoid_percent"] == 25


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_feasible_gap_requires_finite_numeric_fraction(value):
    with pytest.raises(ValueError):
        metrics.sigmoid_quality(score(value))


def test_stable_extreme_arithmetic_is_explicitly_bounded():
    assert metrics.sigmoid_quality(score(1e308)) == 0
    assert metrics.sigmoid_quality(score(-1e308)) == 1


def test_physical_gate_prevents_quality_score_from_hiding_feasibility_loss():
    left = [score(-.5), score(0, False, "b")]
    right = [score(0), score(0, True, "b")]
    assert metrics.summarize_selected(left)["q_sigmoid_percent"] < metrics.summarize_selected(right)["q_sigmoid_percent"]
    gate = metrics.physical_pair_gate(left, right)
    assert gate["n_joint"] == 1 and gate["joint_signed_gap_delta_fraction"] == -.5
    assert not gate["passed"] and not gate["population_noninferiority_established"]


def test_empty_joint_support_or_worse_physical_gap_fails_consistency():
    assert not metrics.physical_pair_gate([score(0)], [score(0, False)])["passed"]
    assert not metrics.physical_pair_gate([score(.1)], [score(0)])["passed"]
    assert metrics.physical_pair_gate([score(-.1)], [score(0)])["passed"]
