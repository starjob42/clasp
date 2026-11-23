"""Bounded full-budget campaign: immutable attempts, no retries or cancellation.

The controller starts fresh full-horizon runs only after all four methods pass
their 5→10 resume diagnostics on that task. It leaves existing campaigns alone.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from common import (HORIZONS, PROTOCOL, atomic, identity, load_declaration,
                    output_dir, read, require, signed, verify, write_once)

TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                     "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED"}


def run(command, *, timeout=90, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout, **kwargs).stdout


def comment(binding, name):
    return "fb-" + binding["sha256"][:16] + "-" + name


def verify_source(d):
    from clasp.baselines.external_baseline_dev import verify_source_manifest
    return verify_source_manifest(Path(d["source"]["root"]) / "SOURCE_MANIFEST.sha256",
                                  d["source"]["manifest_sha256"])


def seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    h, m, s = map(int, clock.split(":"))
    return int(days) * 86400 + h * 3600 + m * 60 + s


def parse_allocation(raw):
    return dict(re.findall(r"(?:^|\s)(\w+)=(\S*)", raw))


def allocation_checks(fields, binding, name, job):
    expected = {"Account": "vas@a100", "Partition": "gpu_p5", "NumNodes": "1",
                "Requeue": "0", "Comment": comment(binding, name), "JobName": "fb-" + name}
    for field, value in expected.items():
        require(fields.get(field) == value, "Allocation differs: " + field)
    require(seconds(fields["TimeLimit"]) == job["walltime_seconds"], "Walltime differs")
    tres = dict(part.split("=", 1) for part in fields.get("ReqTRES", "").split(",") if "=" in part)
    require(int(tres.get("gres/gpu", "0")) == job["gpus"], "Requested GPU count differs")
    require(int(tres.get("cpu", "0")) == (8 if job["kind"] == "eval" else 40), "Requested CPU count differs")
    require(fields.get("QOS") == ("qos_gpu_a100-dev" if job["walltime_seconds"] <= 7200 else "qos_gpu_a100-t3"), "QOS differs")


def verify_allocation(d, name, job, job_id):
    binding = identity(os.environ["FB_DECLARATION"])
    require(binding["sha256"] == os.environ["FB_DECLARATION_SHA"], "Allocation declaration differs")
    intent = read(Path(d["root"]) / "intents" / (name + ".json"), sealed=True)
    require(intent["declaration"] == binding and intent["job"] == job and intent["name"] == name,
            "Allocation intent differs")
    allocation_checks(parse_allocation(run(["scontrol", "show", "job", "-o", str(job_id)])), binding, name, job)


def query_jobs():
    raw = run(["squeue", "--array", "-h", "-u", "uag72kz", "-o", "%i|%T|%j|%k|%q"])
    rows = {}
    for line in raw.splitlines():
        cells = line.split("|")
        require(len(cells) == 5 and re.fullmatch(r"\d+(?:_\d+)?", cells[0]), "Unrecognized queue record")
        rows[cells[0]] = dict(zip(("id", "state", "name", "comment", "qos"), cells))
    return rows


def accounting(job_id):
    raw = run(["sacct", "-n", "-X", "-P", "-S", "2026-09-18", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"])
    values = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == str(job_id)]
    if not values:
        return None
    require(len(values) == 1, "Ambiguous accounting record")
    row = values[0]
    return {"job_id": row[0], "state": row[1].split()[0], "exit_code": row[2],
            "elapsed_seconds": int(row[3]), "allocated_tres": row[4]}


def evaluation_module(d):
    path = Path(d["source"]["root"]) / "experiments/grpo_full_budget_20260918/budget_evaluation.py"
    spec = importlib.util.spec_from_file_location("full_budget_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluation_root(d, job):
    return Path(d["root"]) / "evaluations" / job["task"] / job["method"] / f"step{job['step']}"


def evaluation_binding(d, binding, job, name):
    path = Path(d["root"]) / "evaluation_contracts" / (name + ".json")
    require(not path.exists(), "Evaluation contract already prepared; no repeated submission")
    audit = output_dir(d, job) / f"chunk-{job['step']}-complete.json"
    return evaluation_module(d).prepare_contract(
        source=d["source"], campaign_declaration=binding, cohort_declaration=d["cohort_declaration"],
        runtime_python=d["runtime_python"], task=job["task"], method=job["method"], step=job["step"],
        horizon=HORIZONS[job["task"]], terminal_audit=identity(audit),
        output_root=str(evaluation_root(d, job)), output_path=str(path))


def submission_command(d, binding, name, job, eval_binding=None):
    source = Path(d["source"]["root"])
    env = {"FB_DECLARATION": binding["path"], "FB_DECLARATION_SHA": binding["sha256"],
           "FB_JOB": name, "FB_SOURCE": str(source)}
    if eval_binding:
        env.update(BUDGET_EVAL_SOURCE=str(source), BUDGET_EVAL_CONTRACT=eval_binding["path"],
                   BUDGET_EVAL_CONTRACT_SHA=eval_binding["sha256"])
    require(all(not any(c in str(v) for c in ",\n\r") for v in env.values()), "Unsafe Slurm export")
    hours = job["walltime_seconds"] // 3600
    launcher = "evaluate_a100.slurm" if job["kind"] == "eval" else "train_a100.slurm"
    return ["sbatch", "--parsable", "--account=vas@a100", "--partition=gpu_p5", "--constraint=a100",
            "--nodes=1", f"--gres=gpu:{job['gpus']}", "--ntasks-per-node=1",
            "--cpus-per-task=" + ("8" if job["kind"] == "eval" else "40"), "--hint=nomultithread",
            f"--time={hours:02d}:00:00", "--qos=" + ("qos_gpu_a100-dev" if hours <= 2 else "qos_gpu_a100-t3"),
            "--no-requeue", "--job-name=fb-" + name, "--comment=" + comment(binding, name),
            "--output=" + str(Path(d["root"]) / "logs" / (name + "-%j.out")),
            "--error=" + str(Path(d["root"]) / "logs" / (name + "-%j.err")),
            "--export=ALL," + ",".join(k + "=" + v for k, v in env.items()),
            str(source / "experiments/grpo_full_budget_20260918" / launcher)]


def priority(item):
    name, job = item
    # Predeclared ordering; never chooses a method based on observed performance.
    stages = {"eval": 0, "diagnostic": 1, "train": 2}
    from common import METHODS, TASKS
    return stages[job["kind"]], job["step"], TASKS.index(job["task"]), METHODS.index(job["method"])


class Campaign:
    def __init__(self, d, binding):
        self.d, self.binding = d, binding
        self.root = Path(d["root"])
        self.path = self.root / "campaign_status.json"
        if self.path.exists():
            self.state = read(self.path, sealed=True)
            require(self.state["declaration"] == binding and set(self.state["jobs"]) == set(d["jobs"]), "Restart scope differs")
        else:
            self.state = {"protocol": self.d.get("protocol", PROTOCOL), "declaration": binding, "started_at_unix": time.time(),
                          "deadline_unix": time.time() + 60 * 86400, "status": "monitoring",
                          "jobs": {name: {"status": "not_submitted"} for name in d["jobs"]}}
            self.save()

    def save(self):
        self.state.pop("receipt_sha256", None)
        self.state["updated_at_unix"] = time.time()
        atomic(self.path, signed(self.state))

    def initialization(self, task):
        path = self.root / "initializations" / (task + ".json")
        if path.exists():
            receipt = read(path, sealed=True)
            require(receipt["declaration"] == self.binding and receipt["task"] == task,
                    "Initializer declaration/task differs")
            if "initialization" in self.d["tasks"][task]:
                require(receipt["initialization"] == self.d["tasks"][task]["initialization"], "Fixed initializer changed")
            return True
        entry = self.d["tasks"][task]
        if "initialization" in entry:
            init = entry["initialization"]
        else:
            from tsp_initialization import gate
            init = gate(self.d)
            if init is None:
                return False
        for record in init["files"].values():
            verify(record)
        from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
        verify_checkpoint_fidelity(init["checkpoint"])
        state_path = Path(init["checkpoint"]) / "trainer_state.json"
        attestation = self.d.get("initialization_attestation")
        if state_path.is_file():
            state = read(state_path)
            require(state.get("global_step") == 31250 and state.get("max_steps") == 31250,
                    "Incomplete SFT initialization")
            sft_state = identity(state_path)
            recorded = None
        else:
            # The pruned 2026-11-13/14 cleanup removed one initializer's trainer
            # state. The trainer never consumes it: the starting policy is the
            # adapter itself, attested by checkpoint_fidelity.json. Record the
            # declared attestation instead of pretending the state was seen.
            require(attestation == "checkpoint_fidelity_adapter_bytes",
                    "Missing SFT trainer state without a declared attestation")
            sft_state = None
            recorded = attestation
        receipt = {"protocol": self.d.get("protocol", PROTOCOL), "task": task, "declaration": self.binding,
                   "initialization": init, "sft_trainer_state": sft_state}
        if recorded:
            receipt["initialization_attestation"] = recorded
        write_once(path, signed(receipt))
        return True

    def audit_completion(self, name, record):
        job = self.d["jobs"][name]
        if job["kind"] == "eval":
            bound = record["evaluation_contract"]
            verify(bound)
            script = Path(self.d["source"]["root"]) / "experiments/grpo_full_budget_20260918/budget_evaluation.py"
            run([self.d["runtime_python"], "-B", str(script), "audit", "--contract", bound["path"], "--contract-sha", bound["sha256"]], timeout=1800)
            artifact = evaluation_root(self.d, job) / "result_index.json"
            receipt = read(artifact, sealed=True)
            require(receipt["status"] == "complete_raw_evaluation_canonical_audit", "Evaluation audit incomplete")
        else:
            artifact = output_dir(self.d, job) / f"chunk-{job['step']}-complete.json"
            receipt = read(artifact, sealed=True)
            require(receipt["job_id"] == record["job_id"] and receipt["step"] == job["step"]
                    and receipt["task"] == job["task"] and receipt["method"] == job["method"], "Wrong training completion")
            verify(receipt["contract"])
            require(read(receipt["contract"]["path"])["campaign_declaration"] == self.binding, "Wrong completion declaration")
        record["completion"] = identity(artifact)

    def reports(self):
        from common import METHODS, TASKS, job_key
        for task in TASKS:
            for step in (100, 400, HORIZONS[task]):
                names = [job_key(task, method, "eval", step) for method in METHODS]
                if not all(name in self.state["jobs"] and self.state["jobs"][name]["status"] == "completed" for name in names):
                    continue
                key = f"{task}_step{step}"
                if key in self.state.get("reports", {}):
                    verify(self.state["reports"][key])
                    continue
                paths = [self.state["jobs"][name]["evaluation_contract"]["path"] for name in names]
                report = self.root / "reports" / (key + ".json")
                script = Path(self.d["source"]["root"]) / "experiments/grpo_full_budget_20260918/budget_evaluation.py"
                run([self.d["runtime_python"], "-B", str(script), "report-task", "--contracts", *paths,
                     "--output", str(report)], timeout=3600)
                self.state.setdefault("reports", {})[key] = identity(report)
                self.save()

    def tick(self, execute):
        require(time.time() < self.state["deadline_unix"], "Controller duration expired; existing jobs remain untouched")
        queue = query_jobs()
        outstanding = 0
        for name, record in self.state["jobs"].items():
            if record["status"] == "submitting":
                raise RuntimeError("Uncertain submission; inspect immutable intent before any manual recovery: " + name)
            if record["status"] != "submitted":
                continue
            if record["job_id"] in queue:
                observed = queue[record["job_id"]]
                require(observed["comment"] == comment(self.binding, name), "Queued job identity changed")
                record["observed_state"] = observed["state"]
                outstanding += 1
                continue
            result = accounting(record["job_id"])
            if result is None or result["state"] not in TERMINAL_FAILURES | {"COMPLETED"}:
                outstanding += 1
                continue
            record["accounting"] = result
            if result["state"] != "COMPLETED" or result["exit_code"] != "0:0":
                record["status"] = "failed"
            else:
                try:
                    self.audit_completion(name, record)
                    record["status"] = "completed"
                except Exception as exc:
                    record.update(status="failed_audit", error=f"{type(exc).__name__}: {exc}")
            self.save()
        failed = [name for name, r in self.state["jobs"].items() if r["status"].startswith("failed")]
        if failed:
            self.state.update(status="stopped_for_diagnosis", failed_jobs=failed)
            self.save()
            return False
        self.reports()
        if all(r["status"] == "completed" for r in self.state["jobs"].values()):
            self.state["status"] = "complete"
            self.save()
            return False
        eligible = []
        ready_initializers = {}
        for name, job in sorted(self.d["jobs"].items(), key=priority):
            if self.state["jobs"][name]["status"] != "not_submitted":
                continue
            if not all(self.state["jobs"][dep]["status"] == "completed" for dep in job["depends"]):
                continue
            if job["task"] not in ready_initializers:
                ready_initializers[job["task"]] = self.initialization(job["task"])
            if not ready_initializers[job["task"]]:
                continue
            eligible.append(name)
        self.state.update(status="monitoring", eligible=eligible, outstanding=outstanding,
                          queue_size=len(queue), observed_at_unix=time.time())
        if not execute:
            self.save()
            return True
        submitted_now = 0
        related = sum(r["comment"].startswith(("at1:", "prev1:", "qs-", "fb-")) for r in queue.values())
        for name in eligible:
            if outstanding >= 2 or related + submitted_now >= 8:
                break
            job = self.d["jobs"][name]
            if job["walltime_seconds"] <= 7200 and sum(r["qos"] == "qos_gpu_a100-dev" for r in queue.values()) + submitted_now >= 10:
                continue
            verify_source(self.d)
            for entry in self.d["tasks"][job["task"]]["training_data_files"].values():
                verify(entry)
            attempts = [self.d["jobs"][key] for key, rec in self.state["jobs"].items() if rec["status"] != "not_submitted"]
            reserved = sum(j["gpus"] * j["walltime_seconds"] / 3600 for j in attempts + [job])
            require(reserved <= self.d["budget"]["maximum_a100_reservation_hours"], "Reservation bound exceeded")
            eval_binding = evaluation_binding(self.d, self.binding, job, name) if job["kind"] == "eval" else None
            cmd = submission_command(self.d, self.binding, name, job, eval_binding)
            intent = {"protocol": self.d.get("protocol", PROTOCOL), "name": name, "job": job, "declaration": self.binding,
                      "command": cmd, "created_at_unix": time.time(), "evaluation_contract": eval_binding}
            write_once(self.root / "intents" / (name + ".json"), signed(intent))
            record = self.state["jobs"][name]
            record.update(status="submitting", intent=identity(self.root / "intents" / (name + ".json")),
                          evaluation_contract=eval_binding)
            self.save()
            raw = run(cmd).strip()
            require(re.fullmatch(r"\d+(?:;[^\s;]+)?", raw), "Ambiguous sbatch result; no retry")
            record.update(status="submitted", job_id=raw.split(";", 1)[0])
            self.save()
            outstanding += 1
            submitted_now += 1
            print(json.dumps({"submitted": name, "job_id": record["job_id"], "reserved_a100_hours": reserved}), flush=True)
        self.save()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    d, binding = load_declaration(args.declaration, args.sha)
    sys.path.insert(0, str(Path(d["source"]["root"])))
    verify_source(d)
    verify(d["cohort_declaration"])
    verify(d["cpu_preflight"])
    preflight = read(d["cpu_preflight"]["path"])
    require(preflight.get("status") == "passed", "CPU preflight failed")
    import hashlib
    from common import canonical
    require(preflight["numerical_declaration_sha256"] == hashlib.sha256(canonical(
        {k: v for k, v in d.items() if k != "cpu_preflight"})).hexdigest(), "CPU preflight covered a different experiment")
    root = Path(d["root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(d, binding)
        while True:
            try:
                proceed = campaign.tick(args.execute)
            except (subprocess.SubprocessError, OSError) as exc:
                # Read failures never authorize a new job. Uncertain submission
                # leaves the persisted 'submitting' state and fails closed next tick.
                campaign.state["observation_error"] = f"{type(exc).__name__}: {exc}"
                campaign.save()
                proceed = True
            if args.once or not proceed:
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
