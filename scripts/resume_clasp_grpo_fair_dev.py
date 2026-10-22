#!/usr/bin/env python3
"""Run once to fill available A100 development slots for the fixed fair pilot.

Default is inspection only. Pass --submit to submit at most --max-new-jobs (2).
Never cancels jobs, changes frozen source, or deletes/reuses partial run outputs.
Scheduler accounting identifies accepted submissions by their exact output path,
including jobs accepted just before a prior submitter stopped writing its ledger.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

QOS = "qos_gpu_a100-dev"
TASKS = ("cvrp", "mis")
METHODS = ("grpo_z", "clasp_grpo_anchor_v3", "clasp_grpo_anchor_v4")
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
RETRYABLE = {"CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def event(directory, payload):
    payload = dict(payload, created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["event_sha256"] = digest
    path = directory / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + digest[:12] + ".json")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def label(task, method):
    return f"{task}_{method}_seed43_step100"


def paths(output, role, task, method):
    train = output / "train" / label(task, method)
    destination = train if role == "train" else output / "evaluation" / label(task, method)
    stdout = output / "logs" / f"{role}_{label(task, method)}_%j.out"
    return train, destination, stdout


def discover_jobs(output, accounting, trusted_ids=None):
    trusted_ids = trusted_ids or {}
    jobs = {}
    for fields in csv.reader(accounting.splitlines(), delimiter="|"):
        if len(fields) < 4:
            continue
        job_id, state, _, stdout = (value.strip() for value in fields[:4])
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            continue
        for role in ("train", "eval"):
            for task in TASKS:
                for method in METHODS:
                    expected = str(paths(output, role, task, method)[2])
                    exact_path = stdout in (expected, expected.replace("%j", job_id))
                    # Before-start cancellations may lose output metadata in
                    # accounting. Only an immutable receipt can bind such an ID.
                    trusted_without_path = (stdout.lower() in ("", "unknown", "(null)", "none")
                                            and trusted_ids.get(job_id) == (role, task, method))
                    if exact_path or trusted_without_path:
                        jobs.setdefault((role, task, method), []).append({
                            "job_id": job_id, "state": state.split()[0].rstrip("+"), "stdout": stdout,
                        })
    return jobs


def command_for(output, protocol, role, task, method, training_job_id=None):
    source = Path(protocol["source_manifest"]["path"]).parent
    runtime, data = protocol["runtime"], protocol["data_root"]
    train, destination, stdout = paths(output, role, task, method)
    command = ["sbatch", "--parsable", f"--job-name={'eval-' if role == 'eval' else ''}{task}-{method}-s43",
               f"--output={stdout}", f"--error={str(stdout).removesuffix('.out')}.err"]
    if role == "train":
        checkpoint = Path(protocol["initializations"][task]["adapter_model.safetensors"]["path"]).parent
        values = {
            "DEV_TASK": task, "DEV_METHOD": method, "DEV_SFT_CHECKPOINT": checkpoint,
            "DEV_OUTPUT_DIR": destination, "DEV_PROJECT_ROOT": source, "DEV_SEALED_ENV": runtime,
            "DEV_DATA_ROOT": data, "DEV_SEED": 43, "DEV_MAX_STEPS": 100,
            "DEV_SAVE_STEPS": 100, "DEV_SAVE_TOTAL_LIMIT": 1, "DEV_RESUME_CHECKPOINT": "",
            "DEV_ESTIMATOR_ARM": "token_a0_control", "DEV_GROUP_STD_CORRECTION": 1,
            "DEV_ROLLOUT_BATCH_SIZE": "",
        }
        script = "train_clasp_canonical_checkpoint_dev_a100.slurm"
    else:
        command.append(f"--dependency=afterok:{training_job_id}")
        values = {"FAIR_EVAL_SOURCE": source, "FAIR_EVAL_RUNTIME": runtime, "FAIR_EVAL_DATA": data,
                  "FAIR_EVAL_OUTPUT": destination, "FAIR_EVAL_TASK": task, "FAIR_EVAL_METHOD": method,
                  "FAIR_EVAL_TRAIN": train}
        script = "evaluate_clasp_grpo_fair_dev_a100.slurm"
    command.extend(["--export=ALL," + ",".join(f"{key}={value}" for key, value in values.items()), str(source / "scripts" / script)])
    return command


def reconcile(output, user, *, submit=False, limit=10, max_new=2, training_only=False, runner=run):
    output = Path(output).resolve(strict=True)
    protocol = json.loads((output / "ledgers/protocol.json").read_text())
    source_manifest = protocol["source_manifest"]
    observed = hashlib.sha256(Path(source_manifest["path"]).read_bytes()).hexdigest()
    if observed != source_manifest["sha256"]:
        raise ValueError("frozen source manifest hash changed")
    if protocol["training"]["seed"] != 43 or protocol["training"]["steps"] != 100:
        raise ValueError("unexpected pilot protocol")
    directory = output / "ledgers/orchestration_events"
    events = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    recorded, trusted_ids = {}, {}
    bindings = [(tuple(item["key"]), item["job_id"]) for item in events if item["kind"] == "submitted"]
    with (output / "ledgers/submission.tsv").open() as handle:
        bindings.extend(((row["role"], row["task"], row["method"]), row["job_id"])
                        for row in csv.DictReader(handle, delimiter="\t"))
    for key, job_id in bindings:
        if key[0] not in ("train", "eval") or key[1] not in TASKS or key[2] not in METHODS:
            raise ValueError("invalid job binding in submission receipt")
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise ValueError("invalid job ID in submission receipt")
        if job_id in trusted_ids and trusted_ids[job_id] != key:
            raise ValueError("conflicting job bindings in submission receipts")
        recorded.setdefault(key, set()).add(job_id)
        trusted_ids[job_id] = key
    date = protocol["created_at_utc"].split("T")[0]
    accounting = runner(["sacct", "-X", "-n", "-u", user, "-S", date,
                         "--format=JobIDRaw,State,JobName%100,StdOut%500", "-P"])
    jobs = discover_jobs(output, accounting, trusted_ids)
    observed_ids = {item["job_id"] for values in jobs.values() for item in values}
    missing_ids = sorted(set(trusted_ids) - observed_ids, key=int)
    if missing_ids:
        # Slurm's date-window query can omit jobs canceled before they start;
        # an explicit ID query still returns their terminal state and metadata.
        extra = runner(["sacct", "-X", "-n", "-j", ",".join(missing_ids),
                        "--format=JobIDRaw,State,JobName%100,StdOut%500", "-P"])
        for key, values in discover_jobs(output, extra, trusted_ids).items():
            known = {item["job_id"] for item in jobs.get(key, [])}
            jobs.setdefault(key, []).extend(item for item in values if item["job_id"] not in known)
    queue = runner(["squeue", "-r", "-h", "-u", user, "-o", "%i|%T|%q"])
    occupied = sum(1 for row in csv.reader(queue.splitlines(), delimiter="|") if len(row) == 3 and row[2] == QOS)
    available = max(0, limit - occupied)
    observations, submitted = [], []
    # An intent without a terminal submission event is reconciled through sacct;
    # until accounting sees it, fail closed rather than risking duplicate jobs.
    ended = {item["transaction"] for item in events if item["kind"] in ("submitted", "rejected")}
    unresolved = {tuple(item["key"]): item for item in events
                  if item["kind"] == "intent" and item["transaction"] not in ended}
    for role in (("train",) if training_only else ("train", "eval")):
        for task in TASKS:
            for method in METHODS:
                key = (role, task, method)
                known = sorted(jobs.get(key, []), key=lambda item: int(item["job_id"]))
                seen_ids = {item["job_id"] for item in known}
                active = [item for item in known if item["state"] in ACTIVE]
                completed = [item for item in known if item["state"] == "COMPLETED"]
                reason = None
                if active:
                    reason = "active:" + ",".join(item["job_id"] for item in active)
                elif completed:
                    reason = "completed:" + completed[-1]["job_id"]
                elif recorded.get(key, set()) - seen_ids or (
                    key in unresolved and not seen_ids.difference(unresolved[key].get("known_job_ids", seen_ids))
                ):
                    reason = "awaiting_accounting_reconciliation"
                elif any(item["state"] not in RETRYABLE for item in known):
                    reason = "unrecognized_scheduler_state"
                train, destination, _ = paths(output, role, task, method)
                training_id = None
                if reason is None and role == "eval":
                    trained = [item for item in jobs.get(("train", task, method), []) if item["state"] == "COMPLETED"]
                    if not trained:
                        reason = "waiting_for_training_completion"
                    elif not (train / "checkpoints/checkpoint-100/checkpoint_fidelity.json").is_file():
                        reason = "completed_training_missing_terminal_checkpoint"
                    else:
                        training_id = max(trained, key=lambda item: int(item["job_id"]))["job_id"]
                if reason is None and destination.exists():
                    reason = "existing_output_requires_review"
                if reason is not None:
                    observations.append({"key": key, "status": reason})
                    continue
                command = command_for(output, protocol, role, task, method, training_id)
                if not submit or available == 0 or len(submitted) >= max_new:
                    observations.append({"key": key, "status": "ready", "command": command})
                    continue
                directory.mkdir(exist_ok=True)
                transaction = uuid.uuid4().hex
                event(directory, {"kind": "intent", "transaction": transaction, "key": key,
                                  "known_job_ids": sorted(seen_ids), "command": command})
                try:
                    answer = runner(command).strip().split(";", 1)[0]
                except subprocess.CalledProcessError as error:
                    event(directory, {"kind": "rejected", "transaction": transaction, "key": key,
                                      "returncode": error.returncode, "stderr": error.stderr or ""})
                    observations.append({"key": key, "status": "submission_rejected", "stderr": error.stderr or ""})
                    return {"occupied_slots": occupied, "available_slots": available, "submitted": submitted, "observations": observations}
                if not re.fullmatch(r"[1-9][0-9]*", answer):
                    # Preserve unresolved intent; acceptance cannot safely be inferred.
                    raise ValueError("ambiguous sbatch response; reconcile intent before retrying: " + answer)
                receipt = event(directory, {"kind": "submitted", "transaction": transaction, "key": key, "job_id": answer})
                submitted.append(receipt)
                available -= 1
    return {"occupied_slots": occupied, "available_slots": available, "submitted": submitted, "observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user", default=os.environ.get("USER", "uag72kz"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--max-submit-jobs", type=int, default=10)
    parser.add_argument("--max-new-jobs", type=int, default=2)
    parser.add_argument("--training-only", action="store_true", help="never schedule the superseded public-slice evaluation jobs")
    args = parser.parse_args()
    if args.max_submit_jobs < 1 or args.max_new_jobs < 1:
        parser.error("job limits must be positive")
    # Serialize independent invocations; the immutable per-action events remain
    # the provenance record, while this empty file serves only as an OS lock.
    with (args.output / "ledgers/orchestration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = reconcile(args.output, args.user, submit=args.submit, limit=args.max_submit_jobs,
                           max_new=args.max_new_jobs, training_only=args.training_only)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
