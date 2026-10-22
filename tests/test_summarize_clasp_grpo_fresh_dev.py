import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fresh_summary", ROOT / "scripts/summarize_clasp_grpo_fresh_dev.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def record(feasible_count, gap, *, objective=90.0, reference=100.0):
    return SimpleNamespace(
        pass_at_k=feasible_count > 0, best_feasible_gap=gap,
        candidate_feasibility=feasible_count / 8,
        raw={"prompt_was_truncated": False, "best_objective_value": objective,
             "ground_truth_objective": reference,
             "candidates": [{"completion_tokens": 10, "parse_failure": False,
                             "completion": str(index)} for index in range(8)]})


def test_failed_instances_lower_feasibility_and_quality_without_zero_gap_imputation():
    result = summary.summarize([record(4, 0.1), record(0, None)], "mis", "v4")
    assert result["candidate_feasibility_percent"] == 25
    assert result["feasible_at_8_count"] == 1
    assert result["feasible_at_8_percent"] == 50
    assert result["mean_feasible_gap_percent"] == 10
    assert result["unconditional_solved_quality"] == pytest.approx(summary.math.exp(-0.3) / 2)


def test_all_fail_gap_is_missing_and_negative_reference_gap_quality_is_bounded():
    result = summary.summarize([record(0, None)], "cvrp", "v3")
    assert result["mean_feasible_gap_percent"] is None
    assert result["unconditional_solved_quality"] == 0
    assert summary.quality(record(8, -0.1)) == 1


def test_signed_reference_improvement_is_retained_without_changing_primary_clipped_gap():
    result = summary.summarize([record(8, 0, objective=90)], "cvrp", "v4")
    assert result["mean_feasible_gap_percent"] == 0
    assert result["mean_signed_reference_gap_percent"] == -10
    assert result["better_than_reference_instances"] == 1
    assert result["unconditional_solved_quality"] == 1
    with pytest.raises(ValueError, match="disagrees"):
        summary.signed_gap(record(8, 0.1, objective=90), "cvrp")


def test_summary_refuses_missing_predeclared_arms():
    with pytest.raises(ValueError, match="missing predeclared arms"):
        summary.collect([])


def test_existing_raw_evidence_reproduces_independently_audited_historical_values():
    result_root = ROOT / "analysis_outputs/clasp_grpo_fair_dev_20260905/historical/v3_triplet/evaluation/results"
    paths = list(result_root.glob("eval/grpo_z_step200/**/run_details__*.jsonl"))
    if not paths:
        pytest.skip("retained historical development evidence is not available")
    artifact = summary.strict._load_artifact(str(paths[0]), None, label="historical GRPO")
    result = summary.summarize(list(artifact.records.values()), "cvrp", "grpo_z")
    assert result["feasible_at_8_count"] == 97
    assert result["candidate_feasibility_percent"] == pytest.approx(63.875)
    assert result["mean_feasible_gap_percent"] == pytest.approx(4.350623323, abs=1e-7)
    assert result["unconditional_solved_quality"] == pytest.approx(0.855234, abs=5e-7)
