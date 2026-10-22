#!/usr/bin/env python3
"""Declare, run and compare four raw K8 evaluations after verified DDP training.

The source, training declaration, controller and benchmark are pinned at init.
An arm is never retried. Failed training is never evaluated. The management
wrapper copies the existing fresh evaluator with only the new method/helper
accepted; generation, representation and verification remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
TASKS = ("cvrp", "mis")
METHODS = ("grpo_z", "grpo_verified")
BENCHMARK_SHA = "8d2565e29e60a1f5e12fe7fa3635bd2739cc108e7eb2415f112a8941d6ab4752"
OVERLAP_SHA = "fa10f2a70456223258d83e5d758acdfc552ac5a03681436a044f9969a072a7b7"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
WRAPPER = "scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm"
SELF = "scripts/evaluate_grpo_verified_ddpfix.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


def signed(path, field="receipt_sha256"):
    value = json.loads(Path(path).read_text())
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), "invalid receipt: " + str(path))
    return value


def write_new(path, value, field="receipt_sha256"):
    value = {**value, field: digest(value)}
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    Path(path).chmod(0o440)
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run_command(command):
    return subprocess.check_output(command, text=True)


def environment(campaign):
    campaign = Path(campaign).resolve(strict=True)
    training = signed(campaign / "ledgers/protocol.json")
    require(identity(training["controller"]["path"]) == training["controller"], "training controller changed")
    control = module("ddpfix_training_control", training["controller"]["path"])
    verified, source = control.protocol_for(campaign)
    require(verified == training, "training protocol changed")
    require(set(training["tasks"]) == set(TASKS) and set(training["methods"]) == set(METHODS), "wrong training arms")
    require(training["training"]["steps"] == 100 and training["training"]["seed"] == 43
            and training["training"]["rollout_batch_b"] == 9, "wrong training budget")
    fresh = module("ddpfix_fresh_artifacts", source / "scripts/clasp_grpo_fresh_dev_artifacts.py")
    benchmark = fresh.validate_benchmark(Path(training["evaluation"]["data_root"]))
    require(benchmark["manifest_sha256"] == BENCHMARK_SHA and benchmark["audit_sha256"] == OVERLAP_SHA,
            "benchmark does not match declared data_v3")
    return training, source, control, fresh, benchmark


def patched_wrapper(text):
    old = 'grpo_z|v3|v4)'
    helper = 'HELPER="${MANAGEMENT}/scripts/clasp_grpo_fresh_dev_artifacts.py"'
    require(text.count(old) == text.count(helper) == 1, "fresh wrapper interface changed")
    return text.replace(old, 'grpo_z|grpo_verified|v3|v4)').replace(
        helper, 'HELPER="${MANAGEMENT}/scripts/evaluate_grpo_verified_ddpfix.py"')


def command_for(item, training, source, management):
    values = {"FRESH_SOURCE": source, "FRESH_MANAGEMENT_ROOT": management,
              "FRESH_RUNTIME": training["runtime"], "FRESH_DATA_ROOT": training["evaluation"]["data_root"],
              "FRESH_CHECKPOINT": item["checkpoint"], "FRESH_OUTPUT": item["output"],
              "FRESH_TASK": item["task"], "FRESH_METHOD": item["method"], "FRESH_TRAIN": item["training_root"]}
    require(all("," not in str(v) and not any(c.isspace() for c in str(v)) for v in values.values()), "unsupported export path")
    return ["sbatch", "--parsable", "--job-name=" + item["job_name"], "--output=" + item["stdout"],
            "--error=" + item["stdout"].removesuffix(".out") + ".err",
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()), str(management / WRAPPER)]


def initialize(campaign):
    campaign = Path(campaign).resolve(strict=True)
    training, source, _, fresh, benchmark = environment(campaign)
    require(identity(source / SELF)["sha256"] == identity(__file__)["sha256"], "evaluator not included unchanged in frozen source")
    management = campaign / "evaluation_management"
    require(not management.exists() and not (campaign / "evaluation").exists(), "evaluation declaration must precede outputs")
    (management / "scripts").mkdir(parents=True)
    (management / WRAPPER).write_text(patched_wrapper((source / WRAPPER).read_text()))
    (management / SELF).write_bytes(Path(__file__).read_bytes())
    for path in (management / "scripts").iterdir():
        path.chmod(0o440)
    plans = []
    for train in training["planned_train_commands"]:
        label = train["key"].replace("/", "_") + "_ddpfix_k8_seed271828"
        item = {"key": train["key"], "task": train["task"], "method": train["method"],
                "training_root": train["training_root"], "checkpoint": str(Path(train["training_root"]) / "checkpoints/checkpoint-100"),
                "output": str(campaign / "evaluation" / label), "stdout": str(campaign / "logs" / (label + "_%j.out")),
                "job_name": "ddpfix-eval-" + train["key"].replace("/", "-")}
        item["command"] = command_for(item, training, source, management)
        plans.append(item)
    require({item["key"] for item in plans} == {t + "/" + m for t in TASKS for m in METHODS} and len(plans) == 4, "expected four unique arms")
    payload = {"status": "frozen_four_arm_raw_evaluation", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "campaign": str(campaign), "controller_sha256": identity(__file__)["sha256"],
               "training_protocol": identity(campaign / "ledgers/protocol.json"), "training_protocol_sha256": training["receipt_sha256"],
               "source_manifest": training["source_manifest"], "management": {"wrapper": identity(management / WRAPPER), "helper": identity(management / SELF)},
               "benchmark_manifest_sha256": benchmark["manifest_sha256"], "overlap_audit_sha256": benchmark["audit_sha256"],
               "generation": fresh.GENERATION, "repair": False, "plans": plans,
               "scope": "Inspected development data, one seed, 100 updates; no untouched confirmation or SOTA claim.",
               "cost": "Equal on-policy samples; verified teachers add training computation. Report allocated GPU-hours separately."}
    (campaign / "evaluation").mkdir()
    (campaign / "ledgers/evaluation_events").mkdir()
    return write_new(campaign / "ledgers/evaluation_protocol.json", payload)


def protocol_for(campaign):
    campaign = Path(campaign).resolve(strict=True)
    p = signed(campaign / "ledgers/evaluation_protocol.json")
    require(p["campaign"] == str(campaign) and p["controller_sha256"] == identity(__file__)["sha256"], "evaluation controller binding changed")
    require(identity(p["training_protocol"]["path"]) == p["training_protocol"], "training declaration changed")
    for item in p["management"].values():
        require(identity(item["path"]) == item, "evaluation management changed")
    training, source, control, fresh, benchmark = environment(campaign)
    require(training["receipt_sha256"] == p["training_protocol_sha256"] and training["source_manifest"] == p["source_manifest"], "source or protocol mismatch")
    require(p["generation"] == fresh.GENERATION and not p["repair"], "generation changed")
    expected = {Path(item["output"]) for item in p["plans"]}
    require(set((campaign / "evaluation").iterdir()) <= expected, "undeclared evaluation output")
    require(not any(path.is_symlink() for path in expected), "evaluation output symlink")
    return p, training, source, control, fresh


def training_ready(campaign, training, source, control, runner=run_command, *, revalidate=False):
    events = control.read_events(campaign, training)
    bindings = {key: [e["job_id"] for e in group if e["kind"] == "submitted"] for key, group in events.items()}
    require(all(len(ids) <= 1 for ids in bindings.values()), "duplicate training submissions")
    jobs = [ids[0] for ids in bindings.values() if ids]
    if not jobs:
        return False
    rows = control.scheduler_rows(runner(["sacct", "-X", "-P", "-n", "-j", ",".join(jobs),
                                          "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw"]))
    ready = len(jobs) == 4
    for item in training["planned_train_commands"]:
        ids = bindings[item["key"]]
        if not ids or ids[0] not in rows:
            ready = False; continue
        row = rows[ids[0]]
        if row["state"] in ACTIVE:
            ready = False; continue
        require(row["state"] == "COMPLETED" and row["exit_code"] == "0:0", "failed training is comparison-ineligible: " + item["key"])
        receipt_path = campaign / "ledgers/terminal" / (item["key"].replace("/", "_") + ".json")
        if not receipt_path.exists():
            ready = False; continue
        receipt = signed(receipt_path)
        require(receipt["status"] == "terminal_training_verified" and receipt["global_step"] == 100
                and receipt["job_id"] == ids[0] and receipt["key"] == item["key"]
                and receipt["protocol_sha256"] == training["receipt_sha256"], "terminal receipt binding mismatch")
        if revalidate:
            actual = control.terminal_validation(item, source, training)
            require(all(receipt[k] == v for k, v in actual.items()), "terminal checkpoint changed after validation")
    return ready


def read_events(campaign, p):
    result = {item["key"]: [] for item in p["plans"]}
    commands = {item["key"]: item["command"] for item in p["plans"]}
    jobs = set()
    for path in sorted((campaign / "ledgers/evaluation_events").glob("*.json")):
        e = signed(path, "event_sha256")
        require(e["key"] in result and e["protocol_sha256"] == p["receipt_sha256"] and e["command"] == commands[e["key"]], "evaluation event binding mismatch")
        require(e["kind"] in {"intent", "submitted", "rejected"}, "unknown evaluation event")
        result[e["key"]].append(e)
    for group in result.values():
        intents = [e for e in group if e["kind"] == "intent"]
        ends = [e for e in group if e["kind"] != "intent"]
        require(len(intents) <= 1 and len(ends) <= 1, "duplicate evaluation attempt")
        require(not ends or (intents and ends[0]["transaction"] == intents[0]["transaction"]), "orphan evaluation event")
        for e in ends:
            if e["kind"] == "submitted":
                require(re.fullmatch(r"[1-9][0-9]*", e["job_id"]) and e["job_id"] not in jobs, "duplicate/invalid evaluation job")
                jobs.add(e["job_id"])
    return result


def reconcile(campaign, *, submit=False, runner=run_command):
    campaign = Path(campaign).resolve(strict=True)
    p, training, source, control, fresh = protocol_for(campaign)
    if not training_ready(campaign, training, source, control, runner):
        return {"status": "waiting_for_all_four_validated_training_arms", "complete": False}
    events = read_events(campaign, p)
    jobs = [e["job_id"] for group in events.values() for e in group if e["kind"] == "submitted"]
    rows = control.scheduler_rows(runner(["sacct", "-X", "-P", "-n", "-j", ",".join(jobs),
                                          "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw"])) if jobs else {}
    observations, accepted = [], set()
    for item in p["plans"]:
        group = events[item["key"]]
        submitted = [e for e in group if e["kind"] == "submitted"]
        if submitted:
            job = submitted[0]["job_id"]; row = rows.get(job)
            state = row["state"] if row else "awaiting_accounting"
            if row and state not in ACTIVE:
                require(state == "COMPLETED" and row["exit_code"] == "0:0", "failed evaluation requires review: " + item["key"])
                index = signed(Path(item["output"]) / "result_index.json", "index_sha256")
                require(index["task"] == item["task"] and index["method"] == item["method"], "result identity mismatch")
            observations.append({"key": item["key"], "state": state, "job_id": job}); continue
        require(not group, "unresolved or rejected attempt requires review; no automatic retry")
        require(not Path(item["output"]).exists(), "unbound evaluation output")
        if not submit:
            observations.append({"key": item["key"], "state": "ready"}); continue
        queued = control.expanded_queue(runner(["squeue", "-r", "-h", "-u", "uag72kz", "-q", "qos_gpu_a100-dev", "-o", "%i|%T"]))
        lagging = {job for job in jobs if job not in rows or rows[job]["state"] in ACTIVE}
        if len(queued | lagging | accepted) >= 10:
            observations.append({"key": item["key"], "state": "waiting_for_queue_slot"}); continue
        require(training_ready(campaign, training, source, control, runner, revalidate=True), "training changed before submission")
        e = {"key": item["key"], "protocol_sha256": p["receipt_sha256"], "command": item["command"],
             "transaction": uuid.uuid4().hex, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        root = campaign / "ledgers/evaluation_events"
        write_new(root / (uuid.uuid4().hex + ".json"), {**e, "kind": "intent"}, "event_sha256")
        try:
            answer = runner(item["command"]).strip()
            require(re.fullmatch(r"[1-9][0-9]*(;[^\s;]+)?", answer), "sbatch returned invalid job ID")
            job = answer.split(";", 1)[0]
            require(job not in set(jobs) | accepted, "sbatch returned duplicate job")
        except Exception as exc:
            write_new(root / (uuid.uuid4().hex + ".json"), {**e, "kind": "rejected", "error": str(exc)}, "event_sha256")
            raise
        write_new(root / (uuid.uuid4().hex + ".json"), {**e, "kind": "submitted", "job_id": job}, "event_sha256")
        accepted.add(job)
        observations.append({"key": item["key"], "state": "submitted", "job_id": job})
    return {"status": "evaluation_reconciled", "observations": observations,
            "complete": len(observations) == 4 and all(r["state"] == "COMPLETED" for r in observations)}


def launch(args):
    campaign = args.train.resolve(strict=True).parents[1]
    p, training, source, control, fresh = protocol_for(campaign)
    require(training_ready(campaign, training, source, control, revalidate=True), "all training must complete and validate before evaluation")
    items = [i for i in p["plans"] if (i["task"], i["method"]) == (args.task, args.method)]
    require(len(items) == 1, "undeclared evaluation arm")
    item = items[0]
    for field in ("output", "checkpoint"):
        require(getattr(args, field).resolve() == Path(item[field]).resolve(), "evaluation " + field + " changed")
    require(args.train.resolve() == Path(item["training_root"]).resolve(), "training root changed")
    require(args.management.resolve() == Path(p["management"]["wrapper"]["path"]).parents[1]
            and identity(args.running_wrapper) == p["management"]["wrapper"], "running management wrapper changed")
    require(args.source.resolve() == source and args.runtime.resolve() == Path(training["runtime"]).resolve()
            and args.data.resolve() == Path(training["evaluation"]["data_root"]).resolve(), "source/runtime/data changed")
    submitted = [e for e in read_events(campaign, p)[item["key"]] if e["kind"] == "submitted"]
    require(len(submitted) == 1 and submitted[0]["job_id"] == os.environ.get("SLURM_JOB_ID"), "running job is not bound to evaluation declaration")
    terminal = signed(campaign / "ledgers/terminal" / (item["key"].replace("/", "_") + ".json"))
    # Reuse the exact launch/index verifier while replacing only its historical
    # v30 checkpoint gate with this campaign's stronger terminal/source checks.
    def validate_checkpoint(*unused):
        return {"source_manifest": training["source_manifest"], "training_contract": terminal["contract"],
                "checkpoint": {name: terminal["checkpoint"][name] for name in
                               ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}}
    fresh.validate_checkpoint = validate_checkpoint
    fresh.__file__ = p["management"]["helper"]["path"]
    return fresh.launch(args)


def index(output):
    receipt = signed(Path(output) / "launch_receipt.json", "launch_sha256")
    source = Path(receipt["source_manifest"]["path"]).parent
    campaign = Path(output).resolve().parents[1]
    p, training, checked_source, control, fresh = protocol_for(campaign)
    require(source == checked_source and training_ready(campaign, training, source, control, revalidate=True), "training/source changed during evaluation")
    return fresh.index(output)


def summarize(campaign):
    campaign = Path(campaign).resolve(strict=True)
    require(reconcile(campaign)["complete"], "all evaluations must succeed before comparison")
    p, training, source, control, _ = protocol_for(campaign)
    require(training_ready(campaign, training, source, control, revalidate=True), "terminal policies changed")
    report_dir = campaign / "comparison_ddpfix_v1"
    require(not report_dir.exists(), "comparison output already exists")
    summary = module("ddpfix_summary", source / "scripts/summarize_clasp_grpo_fresh_dev.py")
    summary.METHODS = METHODS
    indices = [Path(i["output"]) / "result_index.json" for i in p["plans"]]
    loaded, sources, launches = summary.collect(indices)
    require({r["source_manifest"]["sha256"] for r in launches.values()} == {training["source_manifest"]["sha256"]}, "evaluation source differs from training")
    rows, comparisons = [], {}
    for task in TASKS:
        for method in METHODS:
            rows.append(summary.summarize(list(loaded[task, method].records.values()), task, method))
        a, b = [loaded[task, method] for method in METHODS]
        comparisons[task] = {"strict_comparison": summary.strict.build_receipt(a, b, left_label=METHODS[0], right_label=METHODS[1]),
                             "unconditional_solved_quality": summary.paired_quality(a, b)}
    accounting = signed(campaign / "ledgers/training_accounting.json")
    require(accounting["status"] == "all_four_training_jobs_verified" and accounting["protocol_sha256"] == training["receipt_sha256"], "training accounting mismatch")
    events = read_events(campaign, p)
    jobs = {e["job_id"]: key for key, group in events.items() for e in group if e["kind"] == "submitted"}
    fields = ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "TimelimitRaw", "AllocTRES", "QOS", "Account")
    raw = run_command(["sacct", "-X", "-P", "-n", "-j", ",".join(jobs), "--format=" + ",".join(fields)])
    costs = []
    for parts in csv.reader(raw.splitlines(), delimiter="|"):
        require(len(parts) == len(fields), "malformed evaluation accounting")
        r = dict(zip(fields, parts))
        require(r["JobIDRaw"] in jobs and r["State"] == "COMPLETED" and r["ExitCode"] == "0:0"
                and r["TimelimitRaw"] == "120" and r["QOS"] == "qos_gpu_a100-dev" and r["Account"] == "vas@a100"
                and re.search(r"(?:^|,)gres/gpu=1(?:,|$)", r["AllocTRES"]), "invalid evaluation allocation")
        require(0 < int(r["ElapsedRaw"]) <= 7200, "evaluation exceeded development time limit")
        costs.append({**r, "key": jobs[r["JobIDRaw"]], "allocated_gpu_hours": int(r["ElapsedRaw"]) / 3600})
    require(len(costs) == 4 and {r["JobIDRaw"] for r in costs} == set(jobs), "missing/duplicate evaluation accounting")
    report = {"status": "complete_raw_development_comparison", "evaluation_protocol_sha256": p["receipt_sha256"],
              "benchmark_manifest_sha256": BENCHMARK_SHA, "sources": sources, "summary": rows, "comparisons": comparisons,
              "training_accounting": accounting, "evaluation_accounting": costs,
              "definitions": {"headline_gap": "Own-success nonnegative directional relative reference excess, in percent.",
                              "paired_gap": "Modified minus classic, jointly feasible instances; JSON fractions, multiply by100 for percentage points.",
                              "solved_quality": "All-instance mean(feasible*exp(-3*max(gap,0))); failures score0."},
              "limitations": ["One training seed and already inspected100-instance/task development benchmark.",
                              "Paired instance uncertainty does not measure training-seed uncertainty; no multiplicity correction.",
                              "CVRP verified heuristic references; MIS certified optima.",
                              "3600 matched on-policy completions; teacher supervision adds work, so compute is reported separately.",
                              "Earlier failed development allocations remain separate historical costs; this report does not hide or overwrite them."]}
    report_dir.mkdir()
    write_new(report_dir / "comparison.json", report)
    with (report_dir / "table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["| Task | Method | Feasible@8 | Candidate feasibility | Own-feasible gap |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        gap = "—" if row["mean_feasible_gap_percent"] is None else f'{row["mean_feasible_gap_percent"]:.4f}%'
        label = "Classic GRPO" if row["method"] == "grpo_z" else "GRPO + graded reward + verified teacher"
        lines.append(f'| {row["task"].upper()} | {label} | {row["feasible_at_8_count"]}/100 | {row["candidate_feasibility_percent"]:.3f}% | {gap} |')
    lines += ["", "Raw K8; no repair. Lower gap is better; failures have no conditional gap.",
              "Joint-success gap, feasibility uncertainty, unconditional solved quality and allocation costs are in comparison.json."]
    (report_dir / "table.md").write_text("\n".join(lines) + "\n")
    return {"status": report["status"], "output": str(report_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "watch", "summary", "launch", "index"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--hours", type=float, default=6)
    for name in ("source", "management", "runtime", "data", "checkpoint", "output", "running-wrapper", "train"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    require(not args.submit or args.action in {"run", "watch"}, "only run/watch may submit")
    if args.action == "launch":
        result = launch(args)
    elif args.action == "index":
        result = index(args.output)
    elif args.action == "init":
        result = initialize(args.campaign)
    elif args.action == "summary":
        result = summarize(args.campaign)
    else:
        require(args.campaign is not None, "campaign is required")
        require(0 < args.hours <= 12, "watch must be bounded to at most12 hours")
        deadline = time.monotonic() + args.hours * 3600
        while True:
            with (args.campaign / "ledgers/evaluation.lock").open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                result = reconcile(args.campaign, submit=args.submit)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.action == "run" or result["complete"] or time.monotonic() >= deadline:
                return
            time.sleep(60)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
