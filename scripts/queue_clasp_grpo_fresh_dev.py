#!/usr/bin/env python3
"""Freeze the fresh-evaluation addendum, then fill available Slurm slots once.

The training protocol is preserved. This manager schedules only the eight fresh
SFT/GRPO/v3/v4 evaluations, never the superseded public-slice evaluations.
"""
import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clasp_grpo_fresh_dev_artifacts as fresh
import resume_clasp_grpo_fair_dev as scheduler

TASKS = ("cvrp", "mis")
METHODS = ("sft", "grpo_z", "v3", "v4")
MANAGEMENT_FILES = (
    "clasp_grpo_fresh_dev_artifacts.py", "evaluate_clasp_grpo_fresh_dev_a100.slurm",
    "queue_clasp_grpo_fresh_dev.py", "resume_clasp_grpo_fair_dev.py", "summarize_clasp_grpo_fresh_dev.py",
    "compare_development.py",
)


def freeze(args):
    output, training = args.output.resolve(), args.training_root.resolve(strict=True)
    fresh.require(not output.exists(), "fresh evaluation root already exists; inspect its addendum instead")
    protocol_path = training / "ledgers/protocol.json"
    protocol = fresh.checked_json(protocol_path, "receipt_sha256")
    fresh.require(protocol["training"]["seed"] == 43 and protocol["training"]["steps"] == 100,
                  "fresh evaluation requires the original fixed seed-43 training protocol")
    source = Path(protocol["source_manifest"]["path"]).parent
    fresh.require(fresh.identity(source / "SOURCE_MANIFEST.sha256")["sha256"] == fresh.SOURCE_MANIFEST_SHA256,
                  "unexpected frozen training/evaluation source")
    benchmark = fresh.validate_benchmark(args.data)
    management = args.management.resolve(strict=True)
    code = {name: fresh.identity(management / "scripts" / name) for name in MANAGEMENT_FILES}
    fresh.require(code["compare_development.py"]["sha256"] == fresh.identity(source / "scripts/compare_development.py")["sha256"],
                  "management comparator differs from the frozen source comparator")
    evaluations = []
    for task in TASKS:
        sft = Path(protocol["initializations"][task]["adapter_model.safetensors"]["path"]).parent
        fresh.validate_checkpoint(source, sft, None, task, "sft")
        for method in METHODS:
            label = f"{task}_{method}_fresh_k8_seed271828"
            train = None if method == "sft" else training / "train" / f"{task}_{fresh.METHODS[method]}_seed43_step100"
            checkpoint = sft if train is None else train / "checkpoints/checkpoint-100"
            evaluations.append({"task": task, "method": method, "checkpoint": str(checkpoint),
                                "training_root": str(train) if train else None,
                                "output": str(output / "evaluation" / label),
                                "stdout": str(output / "logs" / f"{label}_%j.out")})
    payload = {
        "schema_version": 1, "status": "frozen_fresh_evaluation_addendum_before_policy_outputs",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accounting_start_date": protocol.get("created_at_utc", dt.datetime.now(dt.timezone.utc).isoformat()).split("T")[0],
        "purpose": "fresh_development_only", "training_protocol": fresh.identity(protocol_path),
        "training_campaign_root": str(training),
        "supersedes": "public eval[100:200] evaluation clauses only; training remains unchanged",
        "source_manifest": protocol["source_manifest"], "management_root": str(management), "management_files": code,
        "runtime": protocol["runtime"], "data_root": str(args.data.resolve(strict=True)), "benchmark": benchmark,
        "generation": fresh.GENERATION, "evaluation_slice": {"split": "eval", "start": 0, "count": 100},
        "representation_profiles": {"sft": "jiang_f638_sft", "grpo": "jiang_f638_clasp"},
        "effective_task_prompt": "identical Jiang-f638 no-reason CVRP/MIS prompts; stage profiles preserve certified unused TSP metadata",
        "repair": False, "evaluations": evaluations,
        "queue": {"qos": scheduler.QOS, "max_submit_jobs_per_user": 10, "gpus_per_job": 1, "walltime": "02:00:00"},
        "analysis": {
            "required_completed_indices": 8, "training_seed": 43, "confirmation_claim": False,
            "primary_comparisons": ["grpo_z_vs_v3", "grpo_z_vs_v4", "v3_vs_v4"],
            "sft_comparisons": "paired only if strict protocol fingerprints match; otherwise descriptive with reason",
            "metrics": ["candidate_feasibility", "pass_at_8", "own_feasible_gap", "jointly_feasible_gap",
                        "unconditional_quality_exp_minus_3_max_gap_0_zero_if_infeasible"],
            "primary_gap": "max(0, directional_relative_gap), as emitted by the frozen evaluator",
            "supplementary_metrics": ["mean_signed_reference_gap_percent", "better_than_reference_instances"],
            "supplementary_inference": "descriptive only; no extra confidence intervals",
            "units": "table rates/gaps are percentages; strict comparison JSON rates/gaps are fractions",
            "quality_formula": "mean(exp(-3 * max(best_feasible_gap, 0))) with zero for no feasible candidate",
            "gap_reference": {"cvrp": "verified_heuristic_reference", "mis": "certified_optimum"},
            "size_bands": ["<=25", "26-50", "51-75", ">75"],
            "size_semantics": "evaluator instance_size; CVRP includes the depot",
            "quality_bootstrap": {"replicates": 20000, "seed": 20260905, "resampling_unit": "paired_instance",
                                  "interval": "percentile_95pct"},
            "other_paired_inference": "frozen strict compare_development.py definitions and its existing fixed seed",
        },
    }
    output.mkdir()
    for name in ("ledgers", "logs", "evaluation"):
        (output / name).mkdir()
    result = fresh.write_new(output / "protocol_addendum.json", payload, "addendum_sha256")
    return {"status": result["status"], "addendum_sha256": result["addendum_sha256"], "evaluations": len(evaluations),
            "jobs_submitted": 0}


def validate_addendum(output):
    addendum = fresh.checked_json(output / "protocol_addendum.json", "addendum_sha256")
    files = [addendum["training_protocol"], addendum["source_manifest"], *addendum["management_files"].values(),
             addendum["benchmark"]["manifest"], addendum["benchmark"]["overlap_audit"]]
    for group in addendum["benchmark"]["files"].values():
        files.extend(group.values())
    for file in files:
        fresh.require(fresh.identity(file["path"])["sha256"] == file["sha256"], "addendum-bound artifact changed")
    fresh.require(len(addendum["evaluations"]) == 8 and
                  {(item["task"], item["method"]) for item in addendum["evaluations"]} ==
                  {(task, method) for task in TASKS for method in METHODS}, "addendum does not define exactly eight evaluations")
    return addendum


def discover(accounting, plan, trusted):
    jobs = {}
    for fields in csv.reader(accounting.splitlines(), delimiter="|"):
        if len(fields) < 4:
            continue
        job_id, state, _, stdout = (value.strip() for value in fields[:4])
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            continue
        for item in plan:
            key = (item["task"], item["method"])
            exact = stdout in (item["stdout"], item["stdout"].replace("%j", job_id))
            trusted_missing = stdout.lower() in ("", "unknown", "none", "(null)") and trusted.get(job_id) == key
            if exact or trusted_missing:
                jobs.setdefault(key, []).append({"job_id": job_id, "state": state.split()[0].rstrip("+")})
    return jobs


def command_for(addendum, item):
    values = {
        "FRESH_SOURCE": str(Path(addendum["source_manifest"]["path"]).parent),
        "FRESH_MANAGEMENT_ROOT": addendum["management_root"], "FRESH_RUNTIME": addendum["runtime"],
        "FRESH_DATA_ROOT": addendum["data_root"], "FRESH_CHECKPOINT": item["checkpoint"],
        "FRESH_OUTPUT": item["output"], "FRESH_TASK": item["task"], "FRESH_METHOD": item["method"],
        "FRESH_TRAIN": item["training_root"] or "",
    }
    fresh.require(all("," not in str(value) and "\n" not in str(value) for value in values.values()), "unsupported Slurm export path")
    return ["sbatch", "--parsable", f"--job-name=fresh-{item['task']}-{item['method']}-k8",
            f"--output={item['stdout']}", f"--error={item['stdout'].removesuffix('.out')}.err",
            "--export=ALL," + ",".join(f"{key}={value}" for key, value in values.items()),
            str(Path(addendum["management_root"]) / "scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm")]


def manage(output, user, *, submit=False, max_new=2, runner=scheduler.run):
    output = Path(output).resolve(strict=True)
    addendum = validate_addendum(output)
    plan = addendum["evaluations"]
    directory = output / "ledgers/fresh_evaluation_events"
    events = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    valid_keys = {(item["task"], item["method"]) for item in plan}
    trusted, recorded = {}, {}
    for entry in events:
        if entry["kind"] == "submitted":
            key = tuple(entry["key"])
            fresh.require(key in valid_keys and re.fullmatch(r"[1-9][0-9]*", entry["job_id"]), "invalid submission event binding")
            fresh.require(entry["job_id"] not in trusted or trusted[entry["job_id"]] == key, "conflicting submitted job binding")
            trusted[entry["job_id"]] = key
            recorded.setdefault(key, set()).add(entry["job_id"])
    accounting = runner(["sacct", "-X", "-n", "-u", user, "-S",
                         addendum.get("accounting_start_date", addendum["created_at_utc"].split("T")[0]),
                         "--format=JobIDRaw,State,JobName%100,StdOut%500", "-P"])
    jobs = discover(accounting, plan, trusted)
    training_jobs = scheduler.discover_jobs(Path(addendum["training_campaign_root"]), accounting)
    missing = set(trusted) - {item["job_id"] for group in jobs.values() for item in group}
    if missing:
        extra = runner(["sacct", "-X", "-n", "-j", ",".join(sorted(missing, key=int)),
                        "--format=JobIDRaw,State,JobName%100,StdOut%500", "-P"])
        for key, group in discover(extra, plan, trusted).items():
            seen = {item["job_id"] for item in jobs.get(key, [])}
            jobs.setdefault(key, []).extend(item for item in group if item["job_id"] not in seen)
    queue = runner(["squeue", "-r", "-h", "-u", user, "-o", "%i|%T|%q"])
    occupied = sum(1 for row in csv.reader(queue.splitlines(), delimiter="|") if len(row) == 3 and row[2] == scheduler.QOS)
    available = max(0, 10 - occupied)
    ended = {item["transaction"] for item in events if item["kind"] in ("submitted", "rejected")}
    unresolved = {tuple(item["key"]): item for item in events if item["kind"] == "intent" and item["transaction"] not in ended}
    observations, submitted = [], []
    for item in plan:
        key = (item["task"], item["method"])
        known = jobs.get(key, [])
        seen = {job["job_id"] for job in known}
        destination = Path(item["output"])
        reason = None
        if (destination / "result_index.json").exists():
            result = fresh.checked_json(destination / "result_index.json", "index_sha256")
            fresh.require((result["task"], result["method"]) == key and result["status"] == "complete_fresh_development_only",
                          "completed result index does not match the planned evaluation")
            reason = "complete"
        elif any(job["state"] in scheduler.ACTIVE for job in known):
            reason = "active:" + ",".join(job["job_id"] for job in known if job["state"] in scheduler.ACTIVE)
        elif recorded.get(key, set()) - seen or (key in unresolved and not seen.difference(unresolved[key].get("known_job_ids", seen))):
            reason = "awaiting_accounting_reconciliation"
        elif any(job["state"] == "COMPLETED" for job in known):
            reason = "completed_job_missing_result_index"
        elif any(job["state"] not in scheduler.RETRYABLE for job in known):
            reason = "unrecognized_scheduler_state"
        elif destination.exists():
            reason = "existing_output_requires_review"
        elif item["method"] != "sft" and not any(
            job["state"] == "COMPLETED"
            for job in training_jobs.get(("train", item["task"], fresh.METHODS[item["method"]]), [])
        ):
            reason = "waiting_for_training_completion"
        elif not (Path(item["checkpoint"]) / "checkpoint_fidelity.json").is_file():
            reason = "waiting_for_certified_terminal_checkpoint"
        if reason:
            observations.append({"key": key, "status": reason})
            continue
        if submit and (available == 0 or len(submitted) >= max_new):
            observations.append({"key": key, "status": "ready_for_slot"})
            continue
        # The frozen training validator verifies step 100, seed 43, same SFT,
        # source and coefficient choices; SFT checks pin the original adapter.
        fresh.validate_checkpoint(Path(addendum["source_manifest"]["path"]).parent, item["checkpoint"],
                                  item["training_root"], item["task"], item["method"])
        command = command_for(addendum, item)
        if not submit:
            observations.append({"key": key, "status": "ready", "command": command})
            continue
        directory.mkdir(exist_ok=True)
        transaction = uuid.uuid4().hex
        scheduler.event(directory, {"kind": "intent", "transaction": transaction, "key": key,
                                    "known_job_ids": sorted(seen), "command": command,
                                    "addendum_sha256": addendum["addendum_sha256"]})
        try:
            job_id = runner(command).strip().split(";", 1)[0]
        except subprocess.CalledProcessError as error:
            scheduler.event(directory, {"kind": "rejected", "transaction": transaction, "key": key,
                                        "returncode": error.returncode, "stderr": error.stderr or ""})
            observations.append({"key": key, "status": "submission_rejected", "stderr": error.stderr or ""})
            break
        fresh.require(re.fullmatch(r"[1-9][0-9]*", job_id), "ambiguous sbatch reply; reconcile the pending intent before retrying")
        receipt = scheduler.event(directory, {"kind": "submitted", "transaction": transaction, "key": key,
                                              "job_id": job_id, "addendum_sha256": addendum["addendum_sha256"]})
        submitted.append(receipt)
        available -= 1
    return {"occupied_slots": occupied, "available_slots": available, "submitted": submitted, "observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("output", "training-root", "data", "management"):
        prepare.add_argument("--" + name, required=True, type=Path)
    fill = commands.add_parser("run")
    fill.add_argument("--output", type=Path, required=True)
    fill.add_argument("--user", default=os.environ.get("USER", "uag72kz"))
    fill.add_argument("--submit", action="store_true")
    fill.add_argument("--max-new-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args)
    else:
        if args.max_new_jobs <= 0:
            parser.error("--max-new-jobs must be positive")
        with (args.output / "ledgers/fresh_queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = manage(args.output, args.user, submit=args.submit, max_new=args.max_new_jobs)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
