"""Protect decoder selection against leakage, missing results and success-only bias."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "decoder_search_analysis", ROOT / "experiments/rloo_decoder_search_20261014/analyze.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def candidate(method="rloo", checkpoint="ckpt-a", config="baseline", **overrides):
    return {"method": method, "checkpoint_id": checkpoint, "config_id": config,
            "configuration": {**analysis.BASELINE, **overrides}}


def spec(candidates, phase="development", count=4):
    prefix = "dev" if phase == "development" else "fresh"
    return {"schema_version": 1, "task": "cvrp",
            "dataset_sha256": ("a" if phase == "development" else "b") * 64,
            "sampling_seeds": [101, 102, 103],
            "instances": [{"instance_id": f"{prefix}-{i}", "semantic_id": f"{prefix}-semantic-{i}"}
                          for i in range(count)],
            "candidates": candidates, "bootstrap_replicates": 1000, "bootstrap_seed": 19}


def record(candidate_, spec_, gaps, phase="development"):
    rows = []
    for seed in spec_["sampling_seeds"]:
        for instance, gap in zip(spec_["instances"], gaps):
            rows.append({**instance, "replicate_seed": seed, "feasible": gap is not None,
                         "gap_signed_percent": gap,
                         "gap_clipped_percent": max(0, gap) if gap is not None else None,
                         "utility": analysis.quality(gap, gap is not None),
                         "candidate_count": 8, "candidates": [{} for _ in range(8)]})
    return {**candidate_, "schema_version": 1, "phase": phase, "task": spec_["task"],
            "dataset_sha256": spec_["dataset_sha256"],
            "sampling_seeds": spec_["sampling_seeds"], "complete": True, "rows": rows}


def frozen_pair():
    candidates = [candidate(), candidate("foarl")]
    development_spec = spec(candidates)
    frozen = analysis.select_development(
        [record(row, development_spec, [3, 4, 5, 6]) for row in candidates], development_spec)
    return frozen, spec(candidates, "test")


def test_empirical_coverage_gate_prevents_selecting_easy_survivors():
    baseline = candidate()
    tempting = candidate(config="lower-gap-fewer-successes", top_p=0.7)
    eligible = candidate(config="improved-full-coverage", top_k=0)
    declared = spec([baseline, tempting, eligible])
    # The tempting config improves conditional gap but loses a hard instance.
    records = [record(baseline, declared, [30, 30, 30, 30]),
               record(tempting, declared, [-50, -50, -50, None]),
               record(eligible, declared, [20, 20, 20, 20])]
    selected = analysis.select_development(records, declared)
    assert selected["selected"]["rloo"]["config_id"] == eligible["config_id"]
    assert selected["baseline_feasibility_floors"]["rloo"]["feasible_count"] == 12
    assert {row["config_id"] for row in selected["pareto_frontier"]["rloo"]} == {
        eligible["config_id"], tempting["config_id"]}
    assert not selected["population_feasibility_noninferiority_established"]


def test_all_checkpoints_share_the_best_baseline_floor():
    candidates = [candidate(checkpoint="weak"), candidate(checkpoint="strong"),
                  candidate(checkpoint="weak", config="lucky", top_k=0)]
    declared = spec(candidates)
    records = [record(candidates[0], declared, [10, 10, None, None]),
               record(candidates[1], declared, [5, 5, 5, 5]),
               record(candidates[2], declared, [-50, -50, -50, None])]
    selected = analysis.select_development(records, declared)
    assert selected["selected"]["rloo"]["checkpoint_id"] == "strong"


def test_signed_improvements_beyond_reference_can_win_when_u10_saturates():
    candidates = [candidate(), candidate(config="better-reference", top_k=0)]
    declared = spec(candidates)
    result = analysis.select_development(
        [record(candidates[0], declared, [-1] * 4),
         record(candidates[1], declared, [-10] * 4)], declared)
    assert result["selected"]["rloo"]["config_id"] == "better-reference"
    assert all(row["u10_percent"] == 100 for row in result["development_summaries"])


@pytest.mark.parametrize("mutation", ["missing_record", "duplicate_record", "missing_row", "duplicate_row",
                                      "partial", "wrong_dataset", "wrong_seed", "wrong_config",
                                      "wrong_semantic", "wrong_count", "tampered_utility", "nan_gap"])
def test_incomplete_or_incompatible_outcomes_cannot_enter_selection(mutation):
    candidates = [candidate(), candidate(config="alternate", top_k=0)]
    declared = spec(candidates)
    records = [record(row, declared, [2] * 4) for row in candidates]
    if mutation == "missing_record":
        records.pop()
    elif mutation == "duplicate_record":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "missing_row":
        records[0]["rows"].pop()
    elif mutation == "duplicate_row":
        records[0]["rows"][-1] = copy.deepcopy(records[0]["rows"][0])
    elif mutation == "partial":
        records[0]["complete"] = False
    elif mutation == "wrong_dataset":
        records[0]["dataset_sha256"] = "f" * 64
    elif mutation == "wrong_seed":
        records[0]["sampling_seeds"] = [7]
    elif mutation == "wrong_config":
        records[0]["configuration"] = {**analysis.BASELINE, "top_p": 0.5}
    elif mutation == "wrong_semantic":
        records[0]["rows"][0]["semantic_id"] = "different"
    elif mutation == "wrong_count":
        records[0]["rows"][0]["candidate_count"] = 9
    elif mutation == "tampered_utility":
        records[0]["rows"][0]["utility"] = 1
    elif mutation == "nan_gap":
        records[0]["rows"][0]["gap_signed_percent"] = float("nan")
    with pytest.raises(ValueError):
        analysis.select_development(records, declared)


def test_each_checkpoint_requires_baseline_even_if_other_configuration_exists():
    declared = spec([candidate(config="different", top_k=0)])
    with pytest.raises(ValueError, match="Every checkpoint"):
        analysis.select_development([record(declared["candidates"][0], declared, [1] * 4)], declared)


def test_ties_are_deterministic_and_records_do_not_select_test_winner():
    candidates = [candidate(config="z"), candidate(config="a", top_k=0)]
    declared = spec(candidates)
    records = [record(row, declared, [1] * 4) for row in candidates]
    first = analysis.select_development(records, declared)
    second = analysis.select_development(records[::-1], declared)
    assert first == second
    assert first["selected"]["rloo"]["config_id"] == "a"
    for row in records:
        row["phase"] = "test"
    with pytest.raises(ValueError, match="phase"):
        analysis.select_development(records, declared)


def test_test_rejects_both_identical_dataset_and_semantic_leakage():
    frozen, declared = frozen_pair()
    records = [record(row, declared, [2] * 4, "test") for row in declared["candidates"]]
    declared["dataset_sha256"] = frozen["development_dataset_sha256"]
    with pytest.raises(ValueError, match="dataset must differ"):
        analysis.analyze_test(records, frozen, declared)
    declared["dataset_sha256"] = "b" * 64
    declared["instances"][0]["semantic_id"] = frozen["development_semantic_ids"][0]
    with pytest.raises(ValueError, match="semantic overlap"):
        analysis.analyze_test(records, frozen, declared)


def test_test_rejects_config_reselection_and_tampered_frozen_selection():
    frozen, declared = frozen_pair()
    records = [record(row, declared, [2] * 4, "test") for row in declared["candidates"]]
    declared["candidates"][0]["configuration"] = {**analysis.BASELINE, "top_k": 0}
    with pytest.raises(ValueError, match="frozen winners"):
        analysis.analyze_test(records, frozen, declared)
    frozen["selected"]["rloo"]["config_id"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        analysis.analyze_test(records, frozen, declared)


def test_paired_gap_uses_only_joint_feasible_outcomes_and_bootstrap_is_reproducible():
    frozen, declared = frozen_pair()
    records = [record(declared["candidates"][0], declared, [1, 50, None, 3], "test"),
               record(declared["candidates"][1], declared, [2, None, 90, 5], "test")]
    result = analysis.analyze_test(records, frozen, declared)
    contrast = result["paired_contrast"]
    assert contrast["joint_feasible_outcome_count"] == 6
    assert contrast["paired_signed_gap_difference_percent"] == -1.5
    assert contrast["feasibility_difference_pp"] == 0
    assert contrast["paired_signed_gap_ci95_percent"][1] < 0
    assert "no training-seed variation" in contrast["uncertainty_scope"]
    assert not contrast["supports_training_algorithm_superiority"]
    assert result == analysis.analyze_test(records, frozen, declared)


def test_missing_foarl_abstains_instead_of_inventing_a_comparison():
    candidates = [candidate()]
    development_spec = spec(candidates)
    frozen = analysis.select_development([record(candidates[0], development_spec, [1] * 4)],
                                         development_spec)
    declared = spec(candidates, "test")
    result = analysis.analyze_test([record(candidates[0], declared, [1] * 4, "test")], frozen, declared)
    assert result["comparison_status"] == "abstain_missing_rloo_or_foarl"
    assert result["paired_contrast"] is None


def test_decoder_block_variation_contributes_to_uncertainty():
    frozen, declared = frozen_pair()
    records = [record(row, declared, [0] * 4, "test") for row in declared["candidates"]]
    for row in records[0]["rows"]:
        gap = -20 if row["replicate_seed"] == 101 else 10
        row.update(gap_signed_percent=gap, gap_clipped_percent=max(0, gap),
                   utility=analysis.quality(gap, True))
    contrast = analysis.analyze_test(records, frozen, declared)["paired_contrast"]
    assert contrast["paired_signed_gap_difference_percent"] == 0
    assert contrast["paired_signed_gap_ci95_percent"][0] == -20
    assert contrast["paired_signed_gap_ci95_percent"][1] == 10
    assert not contrast["supports_fixed_checkpoint_quality_improvement"]


def test_shared_decoder_diagnostics_cannot_replace_frozen_winners():
    candidates = [candidate(method=method) for method in ("rloo", "foarl")]
    declared = spec(candidates)
    declared["test_diagnostic_candidates"] = [candidate(method=method, config="shared", top_p=0.7, top_k=0)
                                               for method in ("rloo", "foarl")]
    frozen = analysis.select_development([record(row, declared, [1] * 4) for row in candidates], declared)
    test_spec = spec([*candidates, *frozen["diagnostic_candidates"]], "test")
    records = [record(row, test_spec, [4 if row["method"] == "rloo" else 2] * 4, "test")
               for row in candidates]
    records += [record(row, test_spec, [-10 if row["method"] == "rloo" else 10] * 4, "test")
                for row in frozen["diagnostic_candidates"]]
    result = analysis.analyze_test(records, frozen, test_spec)
    assert result["paired_contrast"]["paired_signed_gap_difference_percent"] == 2
    assert result["per_method"]["rloo"]["config_id"] == "baseline"
    assert len(result["diagnostic_summaries"]) == 2
    assert result["diagnostic_comparisons"][0]["contrast"]["paired_signed_gap_difference_percent"] == -20


def test_winner_identical_baseline_diagnostic_is_reused_and_snapshot_is_detached():
    candidates = [candidate()]
    declared = spec(candidates)
    declared["test_diagnostic_candidates"] = [candidate(config="same-config-other-name")]
    frozen = analysis.select_development([record(candidates[0], declared, [1] * 4)], declared)
    assert frozen["diagnostic_candidates"] == []
    declared["candidates"][0]["configuration"]["top_p"] = 0.5
    assert frozen["selected"]["rloo"]["configuration"]["top_p"] == 0.9
    snapshot = dict(frozen)
    digest = snapshot.pop("frozen_selection_sha256")
    assert digest == analysis.digest(snapshot)


def test_no_joint_feasible_outcomes_produces_no_conditional_gap_claim():
    frozen, declared = frozen_pair()
    records = [record(declared["candidates"][0], declared, [1, 1, None, None], "test"),
               record(declared["candidates"][1], declared, [None, None, 1, 1], "test")]
    contrast = analysis.analyze_test(records, frozen, declared)["paired_contrast"]
    assert contrast["paired_signed_gap_difference_percent"] is None
    assert contrast["paired_signed_gap_ci95_percent"] is None
    assert contrast["bootstrap_replicates_without_joint_feasible_outcomes"] == 1000
    assert not contrast["supports_fixed_checkpoint_quality_improvement"]
