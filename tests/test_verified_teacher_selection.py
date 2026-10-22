"""Selection must preserve raw incumbents and give each prompt at most one target."""
import copy
import random

import pytest

from clasp.rewards.verified_teacher_selection import select_verified_teachers


def row(row_id, *, task="cvrp", prompt="p", feasible=False, source=None, target=None, text=None):
    return {"row_id": row_id, "task": task, "prompt_key": prompt,
            "source_feasible": feasible, "source_objective": source,
            "target_objective": target, "target_text": text}


def test_best_raw_incumbent_filters_dominated_repairs_and_tied_targets():
    rows = [row(0, feasible=True, source=100, target=98, text="improved own source"),
            row(1, target=110, text="expensive repair"),
            row(2, feasible=True, source=95),
            row(3, target=95, text="tied repair"),
            row(4, target=94, text="best repair")]
    result = select_verified_teachers(rows)
    assert result["selected_row_ids"] == [4]
    assert result["groups"][0]["raw_incumbent_objective"] == 95
    assert result["groups"][0]["reason"] == "strict_improvement"
    diagnostics = result["diagnostics"]
    assert diagnostics["eligible_teachers"] == 4
    assert diagnostics["dominated_teachers"] == 2
    assert diagnostics["better_rate_among_comparable_teachers"] == 0.25
    assert diagnostics["tied_rate_among_comparable_teachers"] == 0.25
    assert diagnostics["dominated_rate_among_comparable_teachers"] == 0.5


def test_no_teacher_row_still_supplies_best_raw_incumbent():
    result = select_verified_teachers([
        row(0, feasible=True, source=50),
        row(8, feasible=True, source=100, target=90, text="worse than existing raw best"),
    ])
    assert result["selected_row_ids"] == []
    assert result["groups"][0]["reason"] == "no_strict_improvement"


def test_all_invalid_mis_group_selects_best_repair_and_only_one_duplicate():
    result = select_verified_teachers([
        row(10, task="mis", target=3, text="small"),
        row(9, task="mis", target=5, text="z"),
        row(8, task="mis", target=5, text="a"),
        row(6, task="mis", target=5, text="a"),
        row(2, task="mis"),
    ])
    assert result["selected_row_ids"] == [6]
    assert result["groups"][0]["reason"] == "all_sources_infeasible_repair"
    diagnostics = result["diagnostics"]
    assert diagnostics["all_invalid_group_teachers"] == 4
    assert diagnostics["selected_repair_groups"] == 1
    assert diagnostics["comparable_teachers"] == 0
    assert diagnostics["dominated_rate_among_comparable_teachers"] is None


def test_gathered_partial_rank_groups_use_incumbent_from_every_rank():
    rank0 = [row(0, feasible=True, source=100, target=95, text="rank0 improvement")]
    rank1 = [row(4, feasible=True, source=90)]
    rank2 = [row(8, target=89, text="rank2 repair")]
    result = select_verified_teachers(rank0 + rank1 + rank2)
    assert result["selected_row_ids"] == [8]
    assert result["diagnostics"]["prompt_groups"] == 1
    assert result["diagnostics"]["dominated_teachers"] == 1


def test_task_and_prompt_separate_groups_with_sorted_global_row_ids():
    result = select_verified_teachers([
        row(11, task="mis", prompt="same", feasible=True, source=5, target=6, text="large"),
        row(2, task="cvrp", prompt="same", feasible=True, source=5, target=4, text="small"),
        row(20, task="mis", prompt="other"),
    ])
    assert result["selected_row_ids"] == [2, 11]
    assert result["diagnostics"]["prompt_groups"] == 3
    assert [g["reason"] for g in result["groups"]] == ["strict_improvement", "no_teacher", "strict_improvement"]


def test_cvrp_relative_tolerance_does_not_call_roundoff_an_improvement():
    rows = [row(0, feasible=True, source=100),
            row(1, target=100 - 5e-8, text="roundoff"),
            row(2, target=100 - 2e-7, text="real improvement")]
    result = select_verified_teachers(rows)
    assert result["selected_row_ids"] == [2]
    assert result["diagnostics"]["better_teachers"] == 1
    assert result["diagnostics"]["tied_teachers"] == 1
    scaled = copy.deepcopy(rows)
    for item in scaled:
        for key in ("source_objective", "target_objective"):
            if item[key] is not None:
                item[key] *= 1000
    assert select_verified_teachers(scaled)["selected_row_ids"] == [2]


def test_near_best_target_ties_break_by_text_then_row_id():
    result = select_verified_teachers([
        row(0, feasible=True, source=110),
        row(5, target=100, text="z"),
        row(4, target=100 + 1e-8, text="a"),
        row(3, target=100 + 1e-8, text="a"),
    ])
    assert result["selected_row_ids"] == [3]


def test_input_order_does_not_change_selection_or_diagnostics_and_rows_are_unchanged():
    rows = [row(8, feasible=True, source=100, target=90, text="z"),
            row(5, target=90, text="a"), row(2, target=90, text="a"),
            row(10, task="mis", prompt="q", target=5, text="x")]
    original = copy.deepcopy(rows)
    expected = select_verified_teachers(rows)
    assert rows == original
    rng = random.Random(20260906)
    for _ in range(20):
        rng.shuffle(rows)
        assert select_verified_teachers(rows) == expected


@pytest.mark.parametrize("field,value", [
    ("source_objective", float("nan")), ("source_objective", float("inf")),
    ("source_objective", -1), ("source_objective", True),
    ("target_objective", float("nan")), ("target_objective", float("-inf")),
    ("target_objective", -1), ("target_objective", True), ("target_objective", "5"),
    ("target_text", ""), ("target_text", "   "), ("target_text", None),
    ("prompt_key", ""), ("prompt_key", None), ("row_id", True),
    ("row_id", -1), ("row_id", 1.5), ("source_feasible", 1), ("task", "unknown_co_task"),
])
def test_malformed_or_nonfinite_metadata_fails(field, value):
    item = row(0, target=10, text="target")
    item[field] = value
    with pytest.raises(ValueError):
        select_verified_teachers([item])


@pytest.mark.parametrize("source,target", [(None, 5), (5.5, 6), (5, 6.5)])
def test_feasible_mis_source_and_targets_require_integer_cardinalities(source, target):
    with pytest.raises(ValueError):
        select_verified_teachers([row(0, task="mis", feasible=True, source=source, target=target, text="x")])


def test_mis_large_integer_cardinality_is_not_rounded_by_float_conversion():
    size = 2**53
    result = select_verified_teachers([row(0, task="mis", feasible=True, source=size,
                                          target=size + 1, text="one larger")])
    assert result["selected_row_ids"] == [0]


@pytest.mark.parametrize("tolerance", [-1, 1, float("nan"), float("inf"), True, "1e-9"])
def test_invalid_tolerance_fails(tolerance):
    with pytest.raises(ValueError):
        select_verified_teachers([], cvrp_relative_tolerance=tolerance)


def test_duplicate_row_ids_missing_fields_and_inconsistent_target_values_fail():
    with pytest.raises(ValueError, match="duplicate row_id"):
        select_verified_teachers([row(0), row(0)])
    with pytest.raises(ValueError, match="required"):
        select_verified_teachers([{"row_id": 0}])
    with pytest.raises(ValueError, match="inconsistent objectives"):
        select_verified_teachers([row(0, target=90, text="same"), row(1, target=80, text="same")])


def test_empty_input_has_no_selected_teacher_or_manufactured_rates():
    result = select_verified_teachers([])
    assert result["selected_row_ids"] == []
    assert result["groups"] == []
    assert result["diagnostics"]["eligible_teachers"] == 0
    assert result["diagnostics"]["selected_fraction_of_eligible_teachers"] is None


@pytest.mark.parametrize("task,incumbent", [("cvrp", 100), ("mis", 5)])
def test_mixed_feasibility_can_teach_canonical_incumbent_without_objective_gain(task, incumbent):
    rows = [row(0, task=task, feasible=True, source=incumbent,
                target=incumbent, text="canonical raw incumbent"),
            row(1, task=task)]
    strict = select_verified_teachers(rows)
    assert strict == select_verified_teachers(rows, selection_mode="strict_improvement")
    assert strict["selected_row_ids"] == []

    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [0]
    assert result["selection_mode"] == "feasibility_or_improvement"
    assert result["groups"][0]["reason"] == "mixed_feasibility"
    assert result["diagnostics"]["groups_mixed_feasibility"] == 1
    assert result["diagnostics"]["selected_mixed_feasibility_groups"] == 1
    assert result["diagnostics"]["selected_improvement_groups"] == 0
    assert result["diagnostics"]["selected_repair_groups"] == 0


@pytest.mark.parametrize("task,incumbent,improved,worse", [("cvrp", 100, 90, 110), ("mis", 5, 6, 4)])
def test_all_feasible_group_still_requires_strict_improvement(task, incumbent, improved, worse):
    rows = [row(0, task=task, feasible=True, source=incumbent,
                target=incumbent, text="best raw"),
            row(1, task=task, feasible=True, source=worse, target=worse, text="other raw")]
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == []
    assert result["diagnostics"]["groups_mixed_feasibility"] == 0
    rows[1].update(target_objective=improved, target_text="strictly improved")
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [1]
    assert result["groups"][0]["reason"] == "strict_improvement"
    assert result["diagnostics"]["selected_improvement_groups"] == 1


@pytest.mark.parametrize("task,incumbent,improved", [("cvrp", 100, 90), ("mis", 5, 6)])
def test_strict_improvement_is_preferred_and_classified_before_mixed_feasibility(task, incumbent, improved):
    result = select_verified_teachers([
        row(0, task=task, feasible=True, source=incumbent, target=incumbent, text="a incumbent"),
        row(1, task=task, target=improved, text="z improvement"),
    ], selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [1]
    assert result["groups"][0]["reason"] == "strict_improvement"
    assert result["diagnostics"]["groups_mixed_feasibility"] == 1
    assert result["diagnostics"]["selected_mixed_feasibility_groups"] == 0


@pytest.mark.parametrize("task,incumbent,worse", [("cvrp", 100, 110), ("mis", 5, 4)])
def test_mixed_feasibility_rejects_inferior_teacher_using_raw_row_without_target(task, incumbent, worse):
    # The strongest raw source is on another rank and has no eligible target.
    rows = [row(0, task=task, target=worse, text="inferior repair"),
            row(3 << 32, task=task, feasible=True, source=incumbent)]
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == []
    assert result["groups"][0]["raw_incumbent_objective"] == incumbent
    assert result["diagnostics"]["dominated_teachers"] == 1


def test_mixed_feasibility_exact_incumbent_floor_rejects_tolerance_small_cvrp_regression():
    rows = [row(0, feasible=True, source=100, target=100, text="z raw incumbent"),
            row(1, target=100 + 5e-8, text="a slightly worse repair")]
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [0]
    # Relation telemetry retains the declared numerical tolerance; admission has
    # a separate exact floor and does not turn a near tie into an improvement.
    assert result["diagnostics"]["tied_teachers"] == 2
    rows[0].update(target_text=None, target_objective=None)
    assert select_verified_teachers(rows, selection_mode="feasibility_or_improvement")["selected_row_ids"] == []


def test_mixed_feasibility_admits_small_cvrp_gain_without_claiming_strict_improvement():
    result = select_verified_teachers([
        row(0, feasible=True, source=100),
        row(1, target=100 - 5e-8, text="near tie below incumbent"),
    ], selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [1]
    assert result["groups"][0]["reason"] == "mixed_feasibility"
    assert result["diagnostics"]["better_teachers"] == 0


@pytest.mark.parametrize("task,targets,expected", [("cvrp", [110, 100], 1), ("mis", [5, 4], 0)])
def test_all_invalid_best_repair_behavior_is_unchanged_in_feasibility_mode(task, targets, expected):
    rows = [row(i, task=task, target=value, text=f"repair {i}") for i, value in enumerate(targets)]
    strict = select_verified_teachers(rows)
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [expected]
    assert result["groups"] == strict["groups"]
    assert result["diagnostics"] == strict["diagnostics"]
    assert result["diagnostics"]["selected_repair_groups"] == 1


def test_mixed_feasibility_cross_rank_duplicate_selection_is_unique_and_permutation_invariant():
    rows = [row(0, feasible=True, source=100, target=100, text="z incumbent"),
            row(1 << 32, target=100, text="a canonical incumbent"),
            row(2 << 32, target=100, text="a canonical incumbent"),
            row(3 << 32, feasible=True, source=110)]
    original = copy.deepcopy(rows)
    expected = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert expected["selected_row_ids"] == [1 << 32]
    assert rows == original
    rng = random.Random(20260907)
    for _ in range(20):
        rng.shuffle(rows)
        assert select_verified_teachers(rows, selection_mode="feasibility_or_improvement") == expected


def test_feasibility_mode_diagnostics_partition_selected_reasons():
    result = select_verified_teachers([
        row(0, prompt="mixed", feasible=True, source=100, target=100, text="incumbent"),
        row(1, prompt="mixed"),
        row(2, prompt="strict", feasible=True, source=100, target=90, text="improvement"),
        row(3, prompt="all_invalid", target=110, text="repair"),
        row(4, prompt="no_target"),
    ], selection_mode="feasibility_or_improvement")
    counts = result["diagnostics"]
    assert counts["selected_teachers"] == 3
    assert counts["selected_mixed_feasibility_groups"] == 1
    assert counts["selected_improvement_groups"] == 1
    assert counts["selected_repair_groups"] == 1
    assert counts["selected_teachers"] == sum(counts[key] for key in (
        "selected_mixed_feasibility_groups", "selected_improvement_groups", "selected_repair_groups",
    ))
    assert next(group for group in result["groups"] if group["prompt_key"] == "no_target")["reason"] == "no_teacher"


@pytest.mark.parametrize("mode", [None, False, 1, [], "", "best_per_prompt", "per_completion"])
def test_invalid_selection_mode_fails(mode):
    with pytest.raises(ValueError, match="selection_mode"):
        select_verified_teachers([], selection_mode=mode)


@pytest.mark.parametrize("task,incumbent,improved,inferior", [
    ("tsp", 12.5, 11.75, 14.0),
    ("op", 12.5, 14.0, 11.75),
    ("jssp", 12.5, 11.75, 14.0),
])
def test_added_task_uses_best_raw_incumbent_across_ranks(task, incumbent, improved, inferior):
    # A target that improves one rollout still loses to a stronger sibling.
    rows = [row(0, task=task, target=inferior, text="inferior local target"),
            row(1 << 32, task=task, feasible=True, source=incumbent),
            row(2 << 32, task=task, target=improved, text="global improvement")]
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [2 << 32]
    assert result["groups"][0]["raw_incumbent_objective"] == incumbent
    assert result["groups"][0]["reason"] == "strict_improvement"
    assert result["diagnostics"]["dominated_teachers"] == 1


@pytest.mark.parametrize("task,incumbent", [("tsp", 12.5), ("op", 0.0), ("jssp", 12.5)])
def test_added_task_equal_incumbent_teaches_only_mixed_feasibility(task, incumbent):
    rows = [row(0, task=task, feasible=True, source=incumbent,
                target=incumbent, text="canonical incumbent"), row(1, task=task)]
    assert select_verified_teachers(rows)["selected_row_ids"] == []
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [0]
    assert result["groups"][0]["reason"] == "mixed_feasibility"
    rows[1].update(source_feasible=True, source_objective=incumbent)
    assert select_verified_teachers(rows, selection_mode="feasibility_or_improvement")["selected_row_ids"] == []


@pytest.mark.parametrize("task,regression", [("tsp", 5e-8), ("op", -5e-8), ("jssp", 5e-8)])
def test_added_task_exact_incumbent_floor_rejects_tolerance_small_regression(task, regression):
    rows = [row(0, task=task, feasible=True, source=100,
                target=100, text="z incumbent"),
            row(1, task=task, target=100 + regression, text="a near-tied inferior target")]
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["diagnostics"]["tied_teachers"] == 2
    assert result["selected_row_ids"] == [0]
    rows[0].update(target_text=None, target_objective=None)
    assert select_verified_teachers(rows, selection_mode="feasibility_or_improvement")["selected_row_ids"] == []


@pytest.mark.parametrize("task,sign", [("tsp", -1), ("op", 1), ("jssp", -1)])
def test_added_task_roundoff_and_real_improvement_have_distinct_admission(task, sign):
    rows = [row(0, task=task, feasible=True, source=100),
            row(1, task=task, target=100 + sign * 5e-8, text="near tie"),
            row(2, task=task, target=100 + sign * 2e-7, text="real improvement")]
    result = select_verified_teachers(rows)
    assert result["selected_row_ids"] == [2]
    assert result["diagnostics"]["tied_teachers"] == 1
    assert result["diagnostics"]["better_teachers"] == 1


@pytest.mark.parametrize("task,targets,selected", [
    ("tsp", [14.5, 12.0, 12.0], 1),
    ("op", [12.0, 14.5, 14.5], 1),
    ("jssp", [14.5, 12.0, 12.0], 1),
])
def test_added_task_all_invalid_selects_one_best_supplied_teacher(task, targets, selected):
    rows = [row(i, task=task, target=value, text="a best" if i else "z inferior")
            for i, value in enumerate(targets)]
    original = copy.deepcopy(rows)
    result = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert result["selected_row_ids"] == [selected]
    assert result["groups"][0]["reason"] == "all_sources_infeasible_repair"
    assert result["diagnostics"]["selected_repair_groups"] == 1
    assert rows == original
    assert select_verified_teachers([row(0, task=task)])["selected_row_ids"] == []


@pytest.mark.parametrize("task", ["tsp", "op", "jssp"])
@pytest.mark.parametrize("objective", [float("nan"), float("inf"), -1, True, "12"])
def test_added_task_rejects_invalid_source_and_teacher_objectives(task, objective):
    with pytest.raises(ValueError):
        select_verified_teachers([row(0, task=task, feasible=True, source=objective)])
    with pytest.raises(ValueError):
        select_verified_teachers([row(0, task=task, target=objective, text="target")])


@pytest.mark.parametrize("task", ["tsp", "op", "jssp"])
def test_added_task_identical_teacher_text_must_have_consistent_objective(task):
    with pytest.raises(ValueError, match="inconsistent objectives"):
        select_verified_teachers([row(0, task=task, target=12, text="same"),
                                  row(1, task=task, target=14, text="same")])


def test_all_five_tasks_sharing_prompt_text_remain_separate_and_order_invariant():
    rows = []
    for index, task in enumerate(("tsp", "cvrp", "op", "mis", "jssp")):
        best = 11 if task in ("mis", "op") else 9
        rows.extend([row(index * 2, task=task, feasible=True, source=10),
                     row(index * 2 + 1, task=task, target=best, text="teacher")])
    expected = select_verified_teachers(rows, selection_mode="feasibility_or_improvement")
    assert expected["selected_row_ids"] == [1, 3, 5, 7, 9]
    assert expected["diagnostics"]["prompt_groups"] == 5
    random.Random(20260913).shuffle(rows)
    assert select_verified_teachers(rows, selection_mode="feasibility_or_improvement") == expected
