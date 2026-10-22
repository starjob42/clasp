"""Development selection guards; synthetic vectors are not model evidence."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sft_anchor_candidate_selection", ROOT / "experiments/rloo_sft_anchor_20261013/select_candidate.py")
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


def complete():
    return {(task, arm): {"selected": {"k8": [
        {"instance_id": f"{task}-{i}", "reference_id": f"{task}-reference-{i}",
         "feasible": True, "signed_gap": .1} for i in range(96)]}}
        for task in SELECT.TASKS for arm in SELECT.ARMS}


def gaps(results, arm, gap, task=None):
    for name in (SELECT.TASKS if task is None else (task,)):
        for row in results[name, arm]["selected"]["k8"]:
            if row["feasible"]:
                row["signed_gap"] = gap


def invalid(results, arm, indices, task=None):
    for name in (SELECT.TASKS if task is None else (task,)):
        for i in indices:
            results[name, arm]["selected"]["k8"][i].update(feasible=False, signed_gap=None)


def assessment(result, arm):
    return next(item for item in result["assessments"] if item["arm"] == arm)


def test_equal_policies_have_no_eligible_recipe_and_no_win():
    result = SELECT.assess_results(complete())
    assert result["selected_arm"] is None
    assert result["superiority_established"] is False
    assert result["population_feasibility_noninferiority_established"] is False
    assert result["assessment_requires_separate_production_admission"] is True


def test_shared_recipe_with_consistent_physical_gain_is_eligible():
    data = complete()
    gaps(data, "rloo_cardinal_beta005", .08)
    result = SELECT.assess_results(data)
    assert result["selected_arm"] == "rloo_cardinal_beta005"
    chosen = assessment(result, result["selected_arm"])
    assert chosen["eligible"] and all(chosen["conditions"].values())
    assert result["superiority_established"] is False


def test_fixed_tie_order_does_not_choose_per_task_recipes():
    data = complete()
    for arm in SELECT.CANDIDATES:
        gaps(data, arm, .08)
    result = SELECT.assess_results(data)
    assert result["eligible_arm_ranking"] == list(SELECT.CANDIDATES)


def test_larger_macro_gain_ranks_first_before_tie_order():
    data = complete()
    gaps(data, "rloo_scalar_beta005", .09)
    gaps(data, "rloo_cardinal_beta0", .07)
    result = SELECT.assess_results(data)
    assert result["selected_arm"] == "rloo_cardinal_beta0"


def test_better_macro_score_with_adverse_task_gap_is_ineligible():
    data = complete()
    arm = "rloo_cardinal_beta005"
    gaps(data, arm, -.1, "op")
    gaps(data, arm, .11, "cvrp")
    result = SELECT.assess_results(data)
    item = assessment(result, arm)
    assert item["macro_q_sigmoid_delta_foarl_pp"] > 0
    assert not item["conditions"]["each_task_joint_signed_gap_defined_and_nonworse"]
    assert result["selected_arm"] is None


def test_quality_gain_cannot_hide_one_lower_feasible_count():
    data = complete()
    arm = "rloo_scalar_beta005"
    gaps(data, arm, -.1)
    invalid(data, arm, [0], "op")
    result = SELECT.assess_results(data)
    item = assessment(result, arm)
    assert item["macro_q_sigmoid_delta_foarl_pp"] > 0
    assert not item["conditions"]["each_task_feasibility_not_lower_than_foarl"]
    assert result["selected_arm"] is None


def test_candidate_must_also_preserve_sft_feasibility_and_macro_quality():
    data = complete()
    arm = "rloo_scalar_beta005"
    gaps(data, arm, .08)
    gaps(data, "sft", .01)
    result = SELECT.assess_results(data)
    assert not assessment(result, arm)["conditions"]["macro_quality_not_lower_than_sft"]
    assert result["selected_arm"] is None
    invalid(data, "foarl", [0])
    invalid(data, arm, [0])
    gaps(data, "sft", .1)
    result = SELECT.assess_results(data)
    assert not assessment(result, arm)["conditions"]["each_task_feasibility_not_lower_than_sft"]


def test_better_own_support_mean_cannot_hide_worse_joint_gap():
    data = complete()
    arm = "rloo_cardinal_beta005"
    for task in SELECT.TASKS:
        invalid(data, "foarl", range(90, 96), task)
        invalid(data, "sft", range(90, 96), task)
        invalid(data, arm, range(84, 90), task)
        gaps(data, arm, .11, task)
        for row in data[task, arm]["selected"]["k8"][90:]:
            row["signed_gap"] = -.8
        own = [row["signed_gap"] for row in data[task, arm]["selected"]["k8"] if row["feasible"]]
        assert sum(own) / len(own) < .1
    result = SELECT.assess_results(data)
    item = assessment(result, arm)
    assert item["macro_q_sigmoid_delta_foarl_pp"] > 0
    assert all(v["left_feasible_n"] == v["right_feasible_n"] == 90 for v in item["per_task"].values())
    assert not item["conditions"]["each_task_joint_signed_gap_defined_and_nonworse"]
    assert result["selected_arm"] is None


def test_empty_joint_support_cannot_establish_physical_gain():
    data = complete()
    for arm in SELECT.ARMS:
        invalid(data, arm, range(96))
    result = SELECT.assess_results(data)
    assert result["selected_arm"] is None
    assert all(item["macro_joint_signed_gap_delta_fraction"] is None for item in result["assessments"])


@pytest.mark.parametrize("change", ["missing_arm", "short_rows", "duplicate", "reference", "reorder", "empty_id"])
def test_incomplete_or_unpaired_inputs_are_rejected(change):
    data = complete()
    key = ("op", "rloo_scalar_beta005")
    rows = data[key]["selected"]["k8"]
    if change == "missing_arm":
        del data[key]
    elif change == "short_rows":
        rows.pop()
    elif change == "duplicate":
        rows[1]["instance_id"] = rows[0]["instance_id"]
    elif change == "reference":
        rows[0]["reference_id"] = "different"
    elif change == "reorder":
        rows.reverse()
    else:
        rows[0]["instance_id"] = ""
    with pytest.raises(ValueError):
        SELECT.assess_results(data)


@pytest.mark.parametrize("gap", [True, float("nan"), float("inf"), "0.1"])
def test_invalid_physical_numbers_fail_closed(gap):
    data = complete()
    data["op", "rloo_scalar_beta005"]["selected"]["k8"][0]["signed_gap"] = gap
    with pytest.raises(ValueError):
        SELECT.assess_results(data)


def test_infeasible_rows_cannot_keep_gap():
    data = complete()
    data["op", "rloo_scalar_beta005"]["selected"]["k8"][0]["feasible"] = False
    with pytest.raises(ValueError):
        SELECT.assess_results(data)


def test_real_queued_snapshot_cannot_select_recipe():
    collection = ROOT / "experiments/rloo_sft_anchor_20261013/collected_status_v2_20261014T0911"
    with pytest.raises(ValueError, match="Queued, failed or partial"):
        SELECT.select_collection(collection)
