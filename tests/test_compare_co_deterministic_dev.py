"""Three-method conditional cohorts, unchanged statistics and strict admission."""
from __future__ import annotations

import ast
import copy
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_compare_foarl_teacher_dev import (
    EVAL,
    STRICT,
    allocation,
    artifact,
    contrasting_pair,
    record,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("co_deterministic_report_test", ROOT / "scripts/compare_co_deterministic_dev.py")
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def three_artifacts():
    control, teacher = contrasting_pair()
    classic = artifact([record(i, 8, 105.) if 50 <= i < 75 else record(i, 2, 95.) if i >= 75 else record(i)
                        for i in range(100)])
    return {"foarl_control": control, "foarl_teacher": teacher, "classic_grpo": classic}


def metrics(values, task="cvrp", step=400):
    return {baseline: REPORT.summarize_metrics(STRICT, values[baseline], values["foarl_teacher"], task, step, baseline)
            for baseline in REPORT.BASELINES}


def test_signed_and_cost_function_bodies_preserve_the_reviewed_old_implementation():
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
    old = functions(ROOT / "scripts/compare_foarl_teacher_dev.py")
    current = functions(ROOT / "scripts/compare_co_deterministic_dev.py")
    for name in ("signed_record", "teacher_diagnostics", "compute_accounting", "finite", "percent", "ci_text"):
        assert current[name] == old[name]


def test_two_pairwise_cohorts_have_independent_effects_and_one_candidate_own_mean():
    values = three_artifacts()
    before = copy.deepcopy(values)
    comparisons = metrics(values)
    a = comparisons["foarl_control"]["signed"]["conditional_best_feasible_gap"]
    b = comparisons["classic_grpo"]["signed"]["conditional_best_feasible_gap"]
    assert a["jointly_feasible_instances"] == b["jointly_feasible_instances"] == 25
    assert a["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(.01)
    assert b["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(-.05)
    assert a["unpaired_right_minus_left_own_conditional_means"] == pytest.approx(-.095)
    assert b["unpaired_right_minus_left_own_conditional_means"] == pytest.approx(0)
    assert a["paired_bootstrap_95pct_ci"]["lower"] == pytest.approx(.01)
    assert b["paired_bootstrap_95pct_ci"]["upper"] == pytest.approx(-.05)
    methods = REPORT.method_metrics(comparisons)
    assert list(methods) == list(REPORT.METHODS)
    assert all(methods[m]["feasible_instances"] == 50 for m in REPORT.METHODS)
    assert methods["foarl_teacher"]["own_signed_gap"] == 0
    assert methods["foarl_teacher"]["feasible_candidates"] == 125
    assert methods["classic_grpo"]["feasible_candidates"] == 250
    assert values == before


def test_empty_pair_keeps_full_feasibility_denominators_and_undefined_gap():
    values = three_artifacts()
    values["classic_grpo"] = artifact([record(i) for i in range(100)])
    comparison = metrics(values)["classic_grpo"]
    gap = comparison["signed"]["conditional_best_feasible_gap"]
    assert gap["jointly_feasible_instances"] == 0
    assert gap["paired_bootstrap_95pct_ci"] is None
    assert gap["left_mean_on_left_feasible_instances"] is None
    assert comparison["signed"]["pass_at_k"]["right_minus_left"] == .5
    assert comparison["counts"]["classic_grpo"]["instances"] == 100
    assert comparison["counts"]["classic_grpo"]["candidates"] == 800


def test_mis_both_comparators_use_exact_cardinality_direction():
    values = {method: artifact([record(i, 8, score, "mis") for i in range(100)])
              for method, score in (("classic_grpo", 95.), ("foarl_control", 96.), ("foarl_teacher", 97.))}
    comparisons = metrics(values, "mis", 200)
    assert comparisons["classic_grpo"]["signed"]["conditional_best_feasible_gap"]["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(-.02)
    assert comparisons["foarl_control"]["signed"]["conditional_best_feasible_gap"]["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(-.01)
    for baseline in REPORT.BASELINES:
        assert comparisons[baseline]["signed"]["conditional_best_feasible_gap"]["paired_bootstrap_95pct_ci"] == comparisons[baseline]["clipped_gap_sensitivity"]["paired_bootstrap_95pct_ci"]
    with pytest.raises(ValueError, match="exceeds the certified optimum"):
        REPORT.signed_record(record(0, 8, 101., "mis"), "mis")


@pytest.mark.parametrize("baseline", REPORT.BASELINES)
@pytest.mark.parametrize("mutation", ["protocol", "reference", "budget", "missing", "not_raw"])
def test_original_strict_pairing_is_retained_for_each_baseline(baseline, mutation):
    values = three_artifacts()
    left, right = values[baseline], values["foarl_teacher"]
    if mutation == "protocol": left.protocol_fingerprint = "c" * 64
    elif mutation == "reference":
        key = next(iter(left.records))
        left.records[key] = dataclasses.replace(left.records[key], reference_output_sha256="d" * 64)
    elif mutation == "budget": left.candidate_budget = right.candidate_budget = 9
    elif mutation == "missing": left.records.pop(next(iter(left.records)))
    else: left.artifact_type = "assisted"
    with pytest.raises(ValueError):
        REPORT.summarize_metrics(STRICT, left, right, "cvrp", 400, baseline)


def test_both_prespecified_comparisons_and_identical_candidate_metrics_are_required():
    comparisons = metrics(three_artifacts())
    partial = {"foarl_control": comparisons["foarl_control"]}
    with pytest.raises(ValueError, match="both predeclared"):
        REPORT.method_metrics(partial)
    comparisons["classic_grpo"]["counts"]["foarl_teacher"]["feasible_candidates"] += 1
    with pytest.raises(ValueError, match="candidate own-success"):
        REPORT.method_metrics(comparisons)


def admitted(tmp_path, step=400, complete=True):
    arms = {}
    for mi, method in enumerate(REPORT.METHODS):
        chunks = []
        for stop in range(100, step + 1, 100):
            job = str(10000 + mi * 1000 + stop)
            path = tmp_path / method / f"checkpoint-{stop}" / "trainer_state.json"
            path.parent.mkdir(parents=True)
            history = [{"step": i, "loss": .1,
                        "grpo/grpo_verified_cvrp_seconds_per_completion": .001,
                        "grpo/grpo_verified_cvrp_selection_selected_teachers": 2,
                        "grpo/grpo_verified_cvrp_selection_prompt_groups": 4} for i in range(1, stop + 1)]
            if not complete: history.pop()
            path.write_text(json.dumps({"global_step": stop, "max_steps": 400, "log_history": history}))
            chunks.append({"step": stop, "job_id": job, "allocation": allocation(job, 4, 900),
                           "complete_checkpoint": {"files": {"trainer_state.json": EVAL.identity(path)}}})
        job = str(20000 + mi)
        evaluation = {"step": step, "job_id": job, "allocation": allocation(job, 1, 300),
                      "details": {"path": method}, "run_config": {"path": method + "_config"}}
        arms[method] = {"training": chunks[-1], "training_chunks": chunks, "evaluation": evaluation}
    return {"task": "cvrp", "step": step, "arms": arms}


def test_each_method_compute_is_counted_once_and_teacher_cost_limits_remain_explicit(tmp_path):
    admission = admitted(tmp_path)
    result = REPORT.compute_accounting(EVAL, admission)
    assert set(result["arms"]) == set(REPORT.METHODS)
    assert sum(v["cumulative_training_allocated_gpu_hours"] for v in result["arms"].values()) == 12
    for method in REPORT.METHODS:
        arm = result["arms"][method]
        assert [c["step"] for c in arm["training_chunks"]] == [100, 200, 300, 400]
        assert arm["endpoint_evaluation"]["allocated_gpu_hours"] == pytest.approx(1 / 12)
    teacher = result["arms"]["foarl_teacher"]["teacher"]
    assert teacher["rank0_preparation_row_wall_seconds_approx"] == pytest.approx(3.6)
    assert teacher["global_selection_counts_when_complete"]["selected_teachers"] == 800
    assert teacher["global_teacher_cpu_seconds"] is None and teacher["teacher_nll_gpu_seconds"] is None
    assert result["arms"]["classic_grpo"]["teacher"]["rank0_preparation_row_wall_seconds_approx"] is None
    assert result["matched_compute_claim"] is False


@pytest.mark.parametrize("fault", ["failed", "duplicate", "missing", "cost"])
def test_cross_method_cost_reuse_or_invalid_jobs_are_rejected(tmp_path, fault):
    admission = admitted(tmp_path)
    classic, teacher = admission["arms"]["classic_grpo"], admission["arms"]["foarl_teacher"]
    if fault == "failed": classic["evaluation"]["allocation"]["slurm"]["State"] = "FAILED"
    elif fault == "duplicate": teacher["evaluation"]["job_id"] = classic["evaluation"]["job_id"]
    elif fault == "missing": classic["training_chunks"].pop(1)
    else: classic["training_chunks"][0]["allocation"]["allocated_gpu_hours"] += 1
    with pytest.raises(ValueError): REPORT.compute_accounting(EVAL, admission)


def test_partial_teacher_telemetry_stays_unavailable(tmp_path):
    result = REPORT.compute_accounting(EVAL, admitted(tmp_path, 100, complete=False))
    teacher = result["arms"]["foarl_teacher"]["teacher"]
    assert teacher["logged_optimizer_steps"] == 99 and not teacher["complete_step_coverage"]
    assert teacher["rank0_preparation_row_wall_seconds_approx"] is None
    assert teacher["global_selection_counts_when_complete"] == {}


@pytest.fixture
def report_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in (REPORT.SELF, REPORT.STATISTICS):
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    values = three_artifacts()
    admitted_all = admitted(tmp_path, 100)
    calls, loads = [], []
    def read_only_pair(campaign, protocol_sha, task, step, *, baseline_method):
        calls.append(baseline_method)
        return {"task": task, "step": step, "arms": copy.deepcopy({m: admitted_all["arms"][m]
                                            for m in (baseline_method, REPORT.CANDIDATE)})}
    controller = SimpleNamespace(read_only_pair=read_only_pair)
    monkeypatch.setattr(REPORT, "bind_runtime", lambda *args: (EVAL, controller, STRICT, source))
    def load(details, config, label):
        loads.append(label)
        return values[label]
    monkeypatch.setattr(STRICT, "_load_artifact", load)
    args = SimpleNamespace(campaign=tmp_path / "campaign", protocol_sha="a" * 64,
                           task="cvrp", step=100, output=tmp_path / "comparison")
    return SimpleNamespace(args=args, controller=controller, data=admitted_all, calls=calls, loads=loads)


def test_report_admits_both_pairs_twice_and_loads_teacher_only_once(report_run):
    result = REPORT.run(report_run.args)
    assert report_run.calls == [*REPORT.BASELINES, *REPORT.BASELINES]
    assert report_run.loads == list(REPORT.METHODS)
    output = report_run.args.output
    assert {p.name for p in output.iterdir()} == {"comparison.json", "table.csv", "paired_effects.csv", "report.md"}
    assert result["receipt_sha256"] == REPORT.sha(REPORT.canonical({k: v for k, v in result.items() if k != "receipt_sha256"}))
    rows = list(csv.DictReader(io.StringIO((output / "table.csv").read_text())))
    assert [row["method"] for row in rows] == list(REPORT.METHODS)
    assert len(rows) == 3 and all(row["instances"] == "100" and row["candidates"] == "800" for row in rows)
    effects = list(csv.DictReader(io.StringIO((output / "paired_effects.csv").read_text())))
    assert len(effects) == 6
    gaps = [row for row in effects if row["metric"] == "shared_success_signed_gap"]
    assert [float(row["candidate_minus_baseline_pp"]) for row in gaps] == pytest.approx([1., -5.])
    text = (output / "report.md").read_text()
    assert "Gap delta on 25 shared-success instances: 1.000000 pp" in text
    assert "Gap delta on 25 shared-success instances: -5.000000 pp" in text
    assert "unadjusted" in text and "not simultaneous confidence" in text
    assert result["statistics"]["multiple_comparison_adjustment"] is None
    assert result["statistics"]["primary400_nominal_intervals_across_tasks_baselines_metrics"] == 12
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="already exists"):
        REPORT.run(report_run.args)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert len(report_run.calls) == 4


@pytest.mark.parametrize("fault", ["first_pair", "second_pair", "candidate_identity", "second_pass"])
def test_failed_or_changed_admission_produces_no_output(report_run, fault):
    original = report_run.controller.read_only_pair
    count = 0
    def changed(*args, **kwargs):
        nonlocal count
        count += 1
        if fault == "first_pair" and count == 1 or fault == "second_pair" and count == 2:
            raise ValueError("actual job not completed")
        value = original(*args, **kwargs)
        if fault == "candidate_identity" and count == 2:
            value["arms"][REPORT.CANDIDATE]["changed"] = True
        if fault == "second_pass" and count == 4:
            value["changed"] = True
        return value
    report_run.controller.read_only_pair = changed
    with pytest.raises(ValueError): REPORT.run(report_run.args)
    assert not report_run.args.output.exists()


@pytest.mark.parametrize("changed", [REPORT.SELF, REPORT.CONTROL, REPORT.EVALUATOR, REPORT.STATISTICS])
def test_source_pin_precedes_any_dynamic_import(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    names = [REPORT.SELF, REPORT.CONTROL, REPORT.EVALUATOR, REPORT.STATISTICS]
    for name in names:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("unexecuted fixture")
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(hashlib.sha256((source / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in names))
    campaign = tmp_path / "campaign"; campaign.mkdir()
    protocol = {"management_source_manifest": EVAL.identity(manifest)}
    protocol_path = campaign / "protocol.json"; protocol_path.write_text(json.dumps(protocol))
    pin = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    monkeypatch.setattr(REPORT, "__file__", str(source / REPORT.SELF))
    (source / changed).write_text("changed producer")
    monkeypatch.setattr(REPORT, "load_module", lambda *args: pytest.fail("unverified code imported"))
    with pytest.raises(ValueError, match="unbound"):
        REPORT.bind_runtime(campaign, pin)
