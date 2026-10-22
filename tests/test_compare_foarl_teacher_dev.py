"""Independent metric arithmetic and fail-closed report admission tests."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


REPORT = load("foarl_teacher_report_test", "compare_foarl_teacher_dev.py")
STRICT = load("foarl_teacher_report_original_statistics", "compare_development.py")
EVAL = load("foarl_teacher_report_test_eval", "evaluate_foarl_teacher_dev.py")


def record(i, count=0, objective=None, task="cvrp"):
    reference = 100.
    signed = ((objective - reference) if task == "cvrp" else (reference - objective)) / reference if count else None
    gap = max(0., signed) if count else None
    candidates = [{"candidate_index": j, "is_feasible": j < count,
                   "objective_value": objective if j < count else None,
                   "ground_truth_objective": reference, "gap_to_ground_truth": gap if j < count else None}
                  for j in range(8)]
    raw = {"candidates": candidates, "ground_truth_objective": reference,
           "best_candidate_index": 0 if count else None, "best_objective_value": objective,
           "best_feasible_gap": gap}
    return STRICT.ArtifactRecord(key=(task, f"{i:064x}", i), task=task, split="eval", example_index=i,
        source_row_locator={"index": i}, instance_sha256=f"{i:064x}", semantic_instance_sha256=f"{i:064x}",
        benchmark_row_sha256=f"{i:064x}", reference_output_sha256="a" * 64,
        transformed_example_sha256=f"{i:064x}", sampling_seed=i, protocol_fingerprint="b" * 64,
        strict_verifier={"name": "shared"}, candidate_budget=8, pass_at_k=bool(count),
        candidate_feasibility=count / 8, best_feasible_gap=gap, raw=raw)


def artifact(records):
    return SimpleNamespace(records={r.key: r for r in records}, artifact_type="native_raw",
                           candidate_budget=8, protocol_fingerprint="b" * 64, strict_verifier={"name": "shared"})


def contrasting_pair():
    left, right = [], []
    for i in range(100):
        if i < 25:  # Both succeed, teacher is worse despite the lower own mean.
            left.append(record(i, 4, 99.)); right.append(record(i, 3, 100.))
        elif i < 50:
            left.append(record(i, 1, 120.)); right.append(record(i))
        elif i < 75:
            left.append(record(i)); right.append(record(i, 2, 100.))
        else:
            left.append(record(i)); right.append(record(i))
    return artifact(left), artifact(right)


def test_own_gap_can_favor_teacher_while_paired_gap_favors_control_and_inputs_unchanged():
    left, right = contrasting_pair()
    before = copy.deepcopy((left, right))
    metrics = REPORT.summarize_metrics(STRICT, left, right, "cvrp", 400)
    gap = metrics["signed"]["conditional_best_feasible_gap"]
    assert gap["left_mean_on_left_feasible_instances"] == pytest.approx(.095)
    assert gap["right_mean_on_right_feasible_instances"] == 0
    assert gap["unpaired_right_minus_left_own_conditional_means"] == pytest.approx(-.095)
    assert gap["jointly_feasible_instances"] == 25
    assert gap["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(.01)
    assert gap["paired_bootstrap_95pct_ci"]["lower"] == pytest.approx(.01)
    assert gap["paired_bootstrap_95pct_ci"]["upper"] == pytest.approx(.01)
    assert gap["paired_bootstrap_95pct_ci"]["replicates"] == 20_000
    assert metrics["clipped_gap_sensitivity"]["right_minus_left_on_jointly_feasible_instances"] == 0
    assert metrics["signed"]["pass_at_k"]["exact_mcnemar"] == {
        "left_only_success": 25, "right_only_success": 25, "discordant_pairs": 50, "exact_two_sided_p": 1.}
    assert all(metrics["counts"][m]["feasible_instances"] == 50 for m in REPORT.METHODS)
    assert all(metrics["counts"][m]["feasible_candidates"] == 125 for m in REPORT.METHODS)
    assert metrics["counts"]["foarl_control"]["better_than_reference_instances"] == 25
    assert (left, right) == before


def test_empty_shared_cohort_stays_undefined_without_dropping_feasibility_failures():
    left = artifact([record(i, 8, 110.) if i == 0 else record(i) for i in range(100)])
    right = artifact([record(i) for i in range(100)])
    metrics = REPORT.summarize_metrics(STRICT, left, right, "cvrp", 100)
    gap = metrics["signed"]["conditional_best_feasible_gap"]
    assert gap["jointly_feasible_instances"] == 0 and gap["paired_bootstrap_95pct_ci"] is None
    assert gap["right_mean_on_right_feasible_instances"] is None
    assert gap["unpaired_right_minus_left_own_conditional_means"] is None
    assert metrics["signed"]["paired_instances"] == 100
    assert metrics["signed"]["pass_at_k"]["right_minus_left"] == pytest.approx(-.01)


def test_mis_direction_and_pair_reversal_are_consistent():
    left = artifact([record(i, 8, 95., "mis") for i in range(100)])
    right = artifact([record(i, 8, 96., "mis") for i in range(100)])
    forward = REPORT.summarize_metrics(STRICT, left, right, "mis", 200)
    backward = REPORT.summarize_metrics(STRICT, right, left, "mis", 200)
    a, b = [x["signed"]["conditional_best_feasible_gap"] for x in (forward, backward)]
    assert a["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(-.01)
    assert b["right_minus_left_on_jointly_feasible_instances"] == pytest.approx(.01)
    assert a["paired_bootstrap_95pct_ci"]["lower"] == pytest.approx(-b["paired_bootstrap_95pct_ci"]["upper"])
    assert a["paired_bootstrap_95pct_ci"] == forward["clipped_gap_sensitivity"]["paired_bootstrap_95pct_ci"]


def test_mis_cannot_report_a_feasible_solution_above_the_certified_optimum():
    with pytest.raises(ValueError, match="exceeds the certified optimum"):
        REPORT.signed_record(record(0, 8, 101., "mis"), "mis")


@pytest.mark.parametrize("mutation", ["best_objective", "best_index", "reference", "candidate_gap", "nan", "cardinality"])
def test_signed_gap_rejects_inconsistent_objectives(mutation):
    row = record(0, 8, 95., "mis")
    if mutation == "best_objective": row.raw["best_objective_value"] = 96.
    elif mutation == "best_index": row.raw["best_candidate_index"] = 3
    elif mutation == "reference": row.raw["ground_truth_objective"] = 99.
    elif mutation == "candidate_gap": row.raw["candidates"][0]["gap_to_ground_truth"] = .1
    elif mutation == "nan": row.raw["candidates"][0]["objective_value"] = float("nan")
    else: row.raw["candidates"][0]["objective_value"] = 95.5
    with pytest.raises(ValueError): REPORT.signed_record(row, "mis")


@pytest.mark.parametrize("mutation", ["protocol", "key", "reference", "budget", "count"])
def test_original_pair_guards_remain_active(mutation):
    left, right = contrasting_pair()
    if mutation == "protocol": right.protocol_fingerprint = "c" * 64
    elif mutation == "key": right.records.pop(next(iter(right.records)))
    elif mutation == "reference":
        key = next(iter(right.records)); right.records[key] = dataclasses.replace(right.records[key], reference_output_sha256="d" * 64)
    elif mutation == "budget": left.candidate_budget = right.candidate_budget = 9
    else:
        key = next(iter(right.records)); left.records.pop(key); right.records.pop(key)
    with pytest.raises(ValueError): REPORT.summarize_metrics(STRICT, left, right, "cvrp", 100)


def allocation(job, gpus, elapsed):
    return {"slurm": {"JobIDRaw": job, "State": "COMPLETED", "ExitCode": "0:0", "ElapsedRaw": str(elapsed)},
            "elapsed_seconds": elapsed, "allocated_gpu_hours": gpus * elapsed / 3600,
            "allocated_cpu_hours": 8 * elapsed / 3600, "billing_unit_hours": elapsed / 3600}


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


def test_costs_include_every_chunk_and_rank0_timing_is_not_scaled_to_four_ranks(tmp_path):
    receipt = admitted(tmp_path)
    result = REPORT.compute_accounting(EVAL, receipt)
    for method in REPORT.METHODS:
        arm = result["arms"][method]
        assert arm["cumulative_training_allocated_gpu_hours"] == 4.
        assert arm["endpoint_evaluation"]["allocated_gpu_hours"] == pytest.approx(1 / 12)
        assert [c["step"] for c in arm["training_chunks"]] == [100, 200, 300, 400]
    teacher = result["arms"]["foarl_teacher"]["teacher"]
    assert teacher["rank0_preparation_row_wall_seconds_approx"] == pytest.approx(3.6)
    assert teacher["global_selection_counts_when_complete"]["selected_teachers"] == 800
    assert teacher["global_teacher_cpu_seconds"] is teacher["teacher_nll_gpu_seconds"] is None
    assert result["arms"]["foarl_control"]["teacher"]["rank0_preparation_row_wall_seconds_approx"] is None


def test_partial_telemetry_is_unavailable_not_zero_or_extrapolated(tmp_path):
    result = REPORT.compute_accounting(EVAL, admitted(tmp_path, step=100, complete=False))
    teacher = result["arms"]["foarl_teacher"]["teacher"]
    assert teacher["logged_optimizer_steps"] == 99 and not teacher["complete_step_coverage"]
    assert teacher["rank0_preparation_row_wall_seconds_approx"] is None
    assert teacher["global_selection_counts_when_complete"] == {}


@pytest.mark.parametrize("mutation", ["failed", "gpu_arithmetic", "missing_chunk", "duplicate_job", "telemetry_change"])
def test_invalid_cost_or_telemetry_evidence_fails(tmp_path, mutation):
    value = admitted(tmp_path)
    arm = value["arms"]["foarl_teacher"]
    if mutation == "failed": arm["evaluation"]["allocation"]["slurm"]["State"] = "FAILED"
    elif mutation == "gpu_arithmetic": arm["training_chunks"][0]["allocation"]["allocated_gpu_hours"] = 0.
    elif mutation == "missing_chunk": arm["training_chunks"].pop(1)
    elif mutation == "duplicate_job": arm["evaluation"]["job_id"] = value["arms"]["foarl_control"]["evaluation"]["job_id"]
    else: Path(arm["training"]["complete_checkpoint"]["files"]["trainer_state.json"]["path"]).write_text("changed")
    with pytest.raises(ValueError): REPORT.compute_accounting(EVAL, value)


@pytest.fixture
def report_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in (REPORT.SELF, REPORT.STATISTICS):
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / name).read_bytes())
    pair = contrasting_pair()
    data = admitted(tmp_path, step=100)
    calls = []
    def read_only_pair(*args):
        calls.append(args)
        return copy.deepcopy(data)
    controller = SimpleNamespace(read_only_pair=read_only_pair)
    monkeypatch.setattr(REPORT, "bind_runtime", lambda *args: (EVAL, controller, STRICT, source))
    monkeypatch.setattr(STRICT, "_load_artifact", lambda details, config, label: pair[REPORT.METHODS.index(label)])
    args = SimpleNamespace(campaign=tmp_path / "campaign", protocol_sha="a" * 64,
                           task="cvrp", step=100, output=tmp_path / "comparison")
    return SimpleNamespace(args=args, calls=calls, controller=controller, data=data)


def test_report_requires_two_unchanged_admissions_then_creates_aggregate_files_once(report_run):
    result = REPORT.run(report_run.args)
    assert len(report_run.calls) == 2
    output = report_run.args.output
    assert {p.name for p in output.iterdir()} == {"comparison.json", "table.csv", "report.md"}
    assert result["receipt_sha256"] == REPORT.sha(REPORT.canonical({k: v for k, v in result.items() if k != "receipt_sha256"}))
    raw = (output / "table.csv").read_bytes()
    assert b"\r\n" in raw
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert float(rows[0]["own_signed_gap_percent"]) == pytest.approx(9.5)
    assert float(rows[1]["own_signed_gap_percent"]) == 0
    text = (output / "report.md").read_text()
    assert "Gap delta on 25 shared-success instances: 1.000000 pp" in text
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="already exists"): REPORT.run(report_run.args)
    assert len(report_run.calls) == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("phase", ["first", "second"])
def test_failed_or_changed_admission_creates_no_report(report_run, phase):
    count = 0
    def reject(*args):
        nonlocal count
        count += 1
        if phase == "first": raise ValueError("actual Slurm not COMPLETED0:0")
        result = copy.deepcopy(report_run.data)
        if count == 2: result["changed"] = True
        return result
    report_run.controller.read_only_pair = reject
    with pytest.raises(ValueError): REPORT.run(report_run.args)
    assert not report_run.args.output.exists()


@pytest.mark.parametrize("changed", [REPORT.SELF, REPORT.CONTROL, REPORT.EVALUATOR, REPORT.STATISTICS])
def test_source_pin_rejects_changed_code_before_any_dynamic_import(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    paths = [REPORT.SELF, REPORT.CONTROL, REPORT.EVALUATOR, REPORT.STATISTICS]
    for name in paths:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("unexecuted fixture")
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(hashlib.sha256((source / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in paths))
    campaign = tmp_path / "campaign"; campaign.mkdir()
    protocol = {"management_source_manifest": EVAL.identity(manifest)}
    protocol_path = campaign / "protocol.json"; protocol_path.write_text(json.dumps(protocol))
    pin = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    monkeypatch.setattr(REPORT, "__file__", str(source / REPORT.SELF))
    (source / changed).write_text("changed producer")
    def forbidden(*args): raise AssertionError("unverified code was imported")
    monkeypatch.setattr(REPORT, "load_module", forbidden)
    with pytest.raises(ValueError, match="unbound"): REPORT.bind_runtime(campaign, pin)
