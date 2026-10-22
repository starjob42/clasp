"""Admitted, paired development comparison of FOARL control and teacher.

No scheduler submission, evaluation, repair, checkpoint selection or mutation
of scientific artifacts. Statistics reuse the frozen development comparator.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path

SELF = "scripts/compare_foarl_teacher_dev.py"
CONTROL = "scripts/foarl_teacher_development_campaign.py"
EVALUATOR = "scripts/evaluate_foarl_teacher_dev.py"
STATISTICS = "scripts/compare_development.py"
PROTOCOL = "foarl-teacher-paired-development-report-v1"
METHODS = ("foarl_control", "foarl_teacher")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bind_runtime(root, protocol_sha):
    """Check the externally pinned declaration and helper bytes before import."""
    root = Path(root).absolute()
    protocol_path = root / "protocol.json"
    require(protocol_path.is_file() and not protocol_path.is_symlink(), "missing regular campaign declaration")
    require(sha(protocol_path.read_bytes()) == protocol_sha, "campaign declaration file hash mismatch")
    protocol = read_json(protocol_path)
    source = Path(protocol["management_source_manifest"]["path"]).parent
    require(source == Path(__file__).resolve().parents[1], "report must execute from the declared management source")
    manifest = source / "SOURCE_MANIFEST.sha256"
    require(manifest.is_file() and not manifest.is_symlink()
            and sha(manifest.read_bytes()) == protocol["management_source_manifest"]["sha256"], "management manifest changed")
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        require(separator and name not in entries and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "invalid source manifest line")
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe source name")
        entries[name] = digest
    for name in (SELF, CONTROL, EVALUATOR, STATISTICS):
        path = source / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path
                and sha(path.read_bytes()) == entries.get(name), "unbound report/admission implementation")
    helper = load_module(source / EVALUATOR, "_foarl_teacher_report_eval")
    helper.validate_source(source, protocol["management_source_manifest"]["sha256"], management=True)
    sys.path.insert(0, str(source / "src"))
    controller = load_module(source / CONTROL, "_foarl_teacher_report_control")
    strict = load_module(source / STATISTICS, "_foarl_teacher_report_statistics")
    return helper, controller, strict, source


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "invalid numeric " + label)
    return float(value)


def signed_record(record, task):
    """Verify the emitted best-candidate objectives; retain raw records intact."""
    result = copy.copy(record)
    candidates = record.raw["candidates"]
    require(len(candidates) == record.candidate_budget == 8, "signed view requires raw K8")
    feasible = [(i, c) for i, c in enumerate(candidates) if c["is_feasible"]]
    require(bool(feasible) == record.pass_at_k, "inconsistent feasibility in signed view")
    require(math.isclose(record.candidate_feasibility, len(feasible) / 8, abs_tol=1e-12), "invalid K8 feasibility fraction")
    if not feasible:
        require(record.best_feasible_gap is None and record.raw["best_objective_value"] is None
                and record.raw["best_candidate_index"] is None, "infeasible instance carries a best result")
        return result
    reference = finite(record.raw["ground_truth_objective"], "reference")
    require(reference > 0, "signed relative gap needs a positive reference")
    objectives = []
    for index, candidate in feasible:
        objective = finite(candidate["objective_value"], "candidate objective")
        require(objective >= 0 and finite(candidate["ground_truth_objective"], "candidate reference") == reference,
                "candidate objective/reference mismatch")
        if task == "mis":
            require(objective.is_integer() and reference.is_integer(), "MIS cardinalities must be integral")
            require(objective <= reference, "MIS feasible objective exceeds the certified optimum")
        signed = (objective - reference) / reference if task == "cvrp" else (reference - objective) / reference
        require(math.isclose(finite(candidate["gap_to_ground_truth"], "candidate gap"), max(0., signed),
                             rel_tol=1e-10, abs_tol=1e-12), "candidate gap disagrees with its objective")
        objectives.append((max(0., signed), objective if task == "cvrp" else -objective, index, signed))
    clipped, directional, best, signed = min(objectives)
    objective = directional if task == "cvrp" else -directional
    require(record.raw["best_candidate_index"] == best
            and finite(record.raw["best_objective_value"], "best objective") == objective
            and math.isclose(float(record.best_feasible_gap), clipped, rel_tol=1e-10, abs_tol=1e-12),
            "raw best objective/index is inconsistent with evaluator selection")
    object.__setattr__(result, "best_feasible_gap", signed)
    return result


def summarize_metrics(strict, left, right, task, step):
    require(task in ("cvrp", "mis") and type(step) is int and step in (100, 200, 400), "unsupported comparison")
    pairing = strict._validate_pairing(left, right)
    require(left.artifact_type == right.artifact_type == "native_raw" and left.candidate_budget == right.candidate_budget == 8,
            "comparison requires original raw K8 artifacts")
    require(len(left.records) == len(right.records) == 100
            and {r.task for a in (left, right) for r in a.records.values()} == {task}
            and all(r.split == "eval" for a in (left, right) for r in a.records.values()), "comparison requires same-task development100")
    pairs = [(left.records[k], right.records[k]) for k in sorted(left.records)]
    signed_pairs = [(signed_record(a, task), signed_record(b, task)) for a, b in pairs]
    scope = f"foarl_teacher:{task}:step{step}"
    signed = strict._summarize_pairs(signed_pairs, scope=scope)
    clipped = strict._summarize_pairs(pairs, scope=scope)["conditional_best_feasible_gap"]
    gap = signed["conditional_best_feasible_gap"]
    own_left, own_right = gap["left_mean_on_left_feasible_instances"], gap["right_mean_on_right_feasible_instances"]
    gap["unpaired_right_minus_left_own_conditional_means"] = (
        own_right - own_left if own_left is not None and own_right is not None else None)
    gap["own_mean_difference_is_not_the_paired_effect"] = True
    counts = {}
    for method, index in zip(METHODS, (0, 1), strict=True):
        records = [p[index] for p in signed_pairs]
        counts[method] = {"instances": 100, "candidates": 800,
                          "feasible_instances": sum(r.pass_at_k for r in records),
                          "feasible_candidates": sum(sum(c["is_feasible"] for c in r.raw["candidates"]) for r in records),
                          "better_than_reference_instances": sum(r.best_feasible_gap is not None and r.best_feasible_gap < 0 for r in records)}
    return {"pairing": pairing, "counts": counts, "signed": signed, "clipped_gap_sensitivity": clipped,
            "gap_semantics": "signed_relative_gap_to_verified_heuristic_reference" if task == "cvrp" else "signed_relative_gap_to_certified_optimum",
            "units": "JSON rates/gaps are fractions; displayed gaps/deltas are percent/percentage points"}


def teacher_diagnostics(helper, training, task, step, method):
    identity = training["complete_checkpoint"]["files"]["trainer_state.json"]
    require(helper.identity(identity["path"]) == identity, "trainer telemetry file changed")
    payload = read_json(identity["path"])
    require(payload["global_step"] == step and payload["max_steps"] == 400, "telemetry checkpoint horizon mismatch")
    rows = [r for r in payload.get("log_history", []) if "loss" in r]
    require(all(type(r.get("step")) is int and 1 <= r["step"] <= step for r in rows)
            and len({r["step"] for r in rows}) == len(rows), "invalid/duplicate optimizer-step telemetry")
    complete = {r["step"] for r in rows} == set(range(1, step + 1))
    prefix = f"grpo/grpo_verified_{task}"
    totals = {}
    for suffix in ("selected_teachers", "prompt_groups", "selected_improvement_groups", "selected_repair_groups"):
        name = prefix + "_selection_" + suffix
        if complete and all(name in r for r in rows):
            values = [finite(r[name], name) for r in rows]
            require(all(v >= 0 and v.is_integer() for v in values), "invalid global teacher selection counter")
            totals[suffix] = int(sum(values))
    timing = prefix + "_seconds_per_completion"
    preparation = None
    if method == "foarl_teacher" and complete and all(timing in r for r in rows):
        values = [finite(r[timing], timing) for r in rows]
        require(all(v >= 0 for v in values), "negative teacher preparation duration")
        preparation = 9 * sum(values)
    require(helper.identity(identity["path"]) == identity, "trainer telemetry changed while reading")
    return {"trainer_state": identity, "logged_optimizer_steps": len(rows), "complete_step_coverage": complete,
            "global_selection_counts_when_complete": totals,
            "rank0_preparation_row_wall_seconds_approx": preparation,
            "global_teacher_cpu_seconds": None, "teacher_nll_gpu_seconds": None,
            "definition": "Stock TRL logs rank 0 local microbatch means; 9 local rows/update times their sum approximates rank 0 preparation wall work only. Global selection counters are gathered before logging.",
            "limits": "No rank extrapolation or inference of teacher NLL runtime. NLL and length means include zero/dummy microbatches and are not reported as whole-teacher means. Absent/incomplete telemetry remains unavailable; allocation differences do not isolate teacher cost."}


def compute_accounting(helper, admission):
    step, task = admission["step"], admission["task"]
    result = {}
    all_jobs = set()
    for method in METHODS:
        arm = admission["arms"][method]
        chunks = arm["training_chunks"]
        require([x["step"] for x in chunks] == list(range(100, step + 1, 100)), "incomplete cumulative training chunks")
        require(chunks[-1] == arm["training"], "current training endpoint differs from cumulative chain")
        for item, gpus in [(x, 4) for x in chunks] + [(arm["evaluation"], 1)]:
            allocation = item["allocation"]
            slurm = allocation["slurm"]
            require(item["job_id"] not in all_jobs, "one job counted in multiple roles")
            all_jobs.add(item["job_id"])
            require(slurm["JobIDRaw"] == item["job_id"] and slurm["State"] == "COMPLETED" and slurm["ExitCode"] == "0:0",
                    "costs require completed actual jobs")
            elapsed = int(slurm["ElapsedRaw"])
            require(elapsed == allocation["elapsed_seconds"] and elapsed > 0
                    and math.isclose(allocation["allocated_gpu_hours"], gpus * elapsed / 3600, rel_tol=1e-12), "GPU allocation arithmetic mismatch")
        result[method] = {
            "training_chunks": [{"step": c["step"], "job_id": c["job_id"], **c["allocation"]} for c in chunks],
            "cumulative_training_allocated_gpu_hours": sum(c["allocation"]["allocated_gpu_hours"] for c in chunks),
            "cumulative_training_elapsed_seconds": sum(c["allocation"]["elapsed_seconds"] for c in chunks),
            "endpoint_evaluation": {"job_id": arm["evaluation"]["job_id"], **arm["evaluation"]["allocation"]},
            "teacher": teacher_diagnostics(helper, arm["training"], task, step, method)}
    return {"arms": result,
            "scope": "Training sums all 100-step allocations through this endpoint; inference includes only this endpoint's K8 evaluation. Earlier evaluations, SFT, diagnostics and other tasks are excluded, so this is not total project compute.",
            "matched_compute_claim": False,
            "limits": "Allocated GPU hours include loading, auditing, CPU work and idle time; they are not GPU-active kernel hours. Equal updates/samples do not imply equal compute."}


def percent(value):
    return "undefined" if value is None else f"{100 * value:.6f}"


def ci_text(ci):
    return "undefined" if ci is None else f"[{percent(ci['lower'])}, {percent(ci['upper'])}]"


def render(receipt):
    metrics, costs = receipt["metrics"], receipt["compute"]["arms"]
    gap = metrics["signed"]["conditional_best_feasible_gap"]
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(["task", "step", "method", "feasible_at8_count", "instances", "candidate_feasible_count", "candidates",
                     "feasible_at8_percent", "candidate_feasibility_percent", "own_signed_gap_percent",
                     "own_gap_denominator", "training_allocated_gpu_hours", "endpoint_eval_allocated_gpu_hours"])
    lines = [f"# FOARL teacher development comparison: {receipt['task']}, step {receipt['step']}", "",
             "Development only, one training seed. Step 400 is the primary endpoint; 100/200 are declared intermediate checks. No confirmation, SOTA or training-seed superiority claim.", "",
             "| Method | Feasible@8 | Candidate feasibility | Signed gap on own successes | Cumulative training GPUh | This evaluation GPUh |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, side in zip(METHODS, ("left", "right"), strict=True):
        c = metrics["counts"][method]
        own = gap[f"{side}_mean_on_{side}_feasible_instances"]
        train = costs[method]["cumulative_training_allocated_gpu_hours"]
        evaluation = costs[method]["endpoint_evaluation"]["allocated_gpu_hours"]
        writer.writerow([receipt["task"], receipt["step"], method, c["feasible_instances"], 100,
                         c["feasible_candidates"], 800, percent(c["feasible_instances"] / 100),
                         percent(c["feasible_candidates"] / 800), percent(own), c["feasible_instances"], train, evaluation])
        lines.append(f"| {method} | {c['feasible_instances']}/100 | {c['feasible_candidates']}/800 ({percent(c['feasible_candidates']/800)}%) | {percent(own)}% (n={c['feasible_instances']}) | {train:.6f} | {evaluation:.6f} |")
    signed = metrics["signed"]
    discordant = signed["pass_at_k"]["exact_mcnemar"]
    lines += ["", "All deltas below are teacher minus control; rates and gaps are displayed in percentage points.", "",
              f"- Feasible@8 delta: {percent(signed['pass_at_k']['right_minus_left'])} pp; paired 95% CI {ci_text(signed['pass_at_k']['paired_bootstrap_95pct_ci'])}. Gained {discordant['right_only_success']}, lost {discordant['left_only_success']} instances.",
              f"- Candidate-feasibility delta: {percent(signed['candidate_feasibility']['right_minus_left'])} pp; paired-instance 95% CI {ci_text(signed['candidate_feasibility']['paired_bootstrap_95pct_ci'])}.",
              f"- Gap delta on {gap['jointly_feasible_instances']} shared-success instances: {percent(gap['right_minus_left_on_jointly_feasible_instances'])} pp; paired 95% CI {ci_text(gap['paired_bootstrap_95pct_ci'])}. Negative favors teacher.",
              f"- Difference of own-success gap means: {percent(gap['unpaired_right_minus_left_own_conditional_means'])} pp. This uses different conditional cohorts and is not the paired effect.", "",
              "The signed CVRP gap preserves better-than-heuristic-reference solutions. Failed instances remain in feasibility denominators; undefined conditional gaps stay undefined. JSON also retains clipped-gap sensitivity.", "",
              "CIs use the frozen bootstrap implementation, 20,000 paired percentile draws, its fixed base seed, and task/milestone-derived seeds. Candidate resampling is by instance; gap resampling is within the fixed shared-success cohort. Signed/clipped sensitivity uses identical resampling draws. These intervals describe development-instance variation, not training-seed uncertainty; no multiplicity correction or significance claim is made.", "",
              "Costs above count 4-GPU training allocations through the milestone and this 1-GPU evaluation. Earlier evaluations, SFT and diagnostics are excluded. Available rank 0 teacher-preparation timing is retained in JSON as a local wall-time approximation; global CPU and teacher NLL GPU runtime are unmeasured."]
    return table.getvalue().encode(), ("\n".join(lines) + "\n").encode()


def run(args):
    require(args.task in ("cvrp", "mis") and args.step in (100, 200, 400), "unsupported endpoint")
    output = Path(args.output).absolute()
    require(not output.exists() and not output.is_symlink(), "comparison output already exists; no overwrite")
    helper, controller, strict, source = bind_runtime(args.campaign, args.protocol_sha)
    admission = controller.read_only_pair(args.campaign, args.protocol_sha, args.task, args.step)
    require(admission["task"] == args.task and admission["step"] == args.step, "admitted pair differs from requested endpoint")
    artifacts = []
    for method in METHODS:
        evidence = admission["arms"][method]["evaluation"]
        artifacts.append(strict._load_artifact(evidence["details"]["path"], evidence["run_config"]["path"], label=method))
    metrics = summarize_metrics(strict, *artifacts, args.task, args.step)
    costs = compute_accounting(helper, admission)
    require(controller.read_only_pair(args.campaign, args.protocol_sha, args.task, args.step) == admission,
            "pair admission changed during comparison")
    receipt = {"protocol": PROTOCOL, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "task": args.task, "step": args.step, "primary_endpoint": args.step == 400,
               "scope": "reused_development100_single_training_seed_no_confirmation_or_sota",
               "direction": "foarl_teacher_minus_foarl_control", "training_seed": 43, "generation_seed": 271828,
               "admission": admission, "metrics": metrics, "compute": costs,
               "statistics": {"bootstrap_replicates": strict.BOOTSTRAP_REPLICATES,
                              "bootstrap_base_seed": strict.BOOTSTRAP_BASE_SEED, "confidence_level": .95,
                              "implementation": helper.identity(source / STATISTICS), "report_implementation": helper.identity(source / SELF),
                              "multiple_comparison_adjustment": None, "single_seed_superiority_claim": False}}
    receipt["receipt_sha256"] = sha(canonical(receipt))
    csv_bytes, markdown = render(receipt)
    output.mkdir(parents=False, exist_ok=False)
    for name, payload in (("comparison.json", json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"),
                          ("table.csv", csv_bytes), ("report.md", markdown)):
        with (output / name).open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        (output / name).chmod(0o440)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True, help="Expected whole-file SHA256 of campaign protocol.json")
    parser.add_argument("--task", choices=("cvrp", "mis"), required=True)
    parser.add_argument("--step", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory with an existing parent")
    receipt = run(parser.parse_args())
    print(json.dumps({"status": "created_admitted_development_comparison", "task": receipt["task"],
                      "step": receipt["step"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
