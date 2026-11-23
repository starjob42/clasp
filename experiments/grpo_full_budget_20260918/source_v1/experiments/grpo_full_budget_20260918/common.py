"""Immutable inputs and deterministic job graph for the full-budget study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "budget-matched-grpo-development-20260918-v1"
METHODS = ("classic_grpo", "foarl", "rloo_legacy", "rloo_quality")
TASKS = ("op", "jssp", "cvrp", "mis", "tsp")
HORIZONS = {"cvrp": 500, "mis": 800, "jssp": 800, "tsp": 800, "op": 1250}
COUNTS = {"cvrp": 2000, "mis": 2900, "jssp": 3200, "tsp": 3200, "op": 5000}
CHUNK_HOURS = {"cvrp": 3, "mis": 2, "jssp": 3, "tsp": 6, "op": 2}
CONTRACT_NAME = "local_development_run_contract.json"
SEAL_NAME = "budget_checkpoint_inventory.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def signed(value):
    return {**value, "receipt_sha256": hashlib.sha256(canonical(value)).hexdigest()}


def read(path, *, sealed=False):
    path = Path(path)
    require(path.is_file() and path.resolve() == path.absolute(), f"Missing/linked JSON: {path}")
    value = json.loads(path.read_text())
    if sealed:
        require(value == signed({k: v for k, v in value.items() if k != "receipt_sha256"}),
                f"Receipt hash differs: {path}")
    return value


def identity(path):
    path = Path(path).absolute()
    require(path.is_file() and path.resolve() == path, f"Missing/linked input: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"Input changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def verify(binding):
    require(identity(binding["path"]) == binding, f"Input binding changed: {binding['path']}")
    return binding


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o440)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def boundaries(task):
    return [v for v in range(50, HORIZONS[task] + 1, 50) if task != "mis" or v != 750]


def job_key(task, method, kind, step):
    return f"{task}_{method}_{kind}_{step}"


def graph():
    jobs = {}
    for task in TASKS:
        diagnostic_gates = [job_key(task, m, "diagnostic", 10) for m in METHODS]
        for method in METHODS:
            for kind, stops in (("diagnostic", [5, 10]), ("train", boundaries(task))):
                previous = 0
                for stop in stops:
                    dependencies = ([job_key(task, method, kind, previous)] if previous else
                                    diagnostic_gates if kind == "train" else [])
                    # Finish a declared evaluation before checkpoints can rotate.
                    if kind == "train" and previous in (100, 400):
                        dependencies.append(job_key(task, method, "eval", previous))
                    hours = 1 if kind == "diagnostic" else CHUNK_HOURS[task] * ((stop - previous) // 50)
                    name = job_key(task, method, kind, stop)
                    jobs[name] = {"task": task, "method": method, "kind": kind, "step": stop,
                                  "prior_step": previous, "depends": dependencies, "gpus": 4,
                                  "walltime_seconds": hours * 3600}
                    if kind == "train" and stop in (100, 400, HORIZONS[task]):
                        jobs[job_key(task, method, "eval", stop)] = {
                            "task": task, "method": method, "kind": "eval", "step": stop,
                            "depends": [name], "gpus": 1, "walltime_seconds": 7200}
                    previous = stop
    return jobs


def output_dir(declaration, job):
    return Path(declaration["root"]) / ("diagnostics" if job["kind"] == "diagnostic" else "train") / job["task"] / job["method"]


def validate_declaration(d):
    require(d["protocol"] == PROTOCOL, "Wrong full-budget protocol")
    require(d["methods"] == list(METHODS) and d["horizons"] == HORIZONS, "Changed numerical graph")
    require(set(d["tasks"]) == set(TASKS) and d["seed"] == 43, "Changed task/seed scope")
    jobs = graph()
    require(d["jobs"] == jobs, "Changed declared job graph")
    require(d["budget"] == {"maximum_jobs": len(jobs), "maximum_outstanding": 2,
                           "maximum_a100_reservation_hours": sum(j["gpus"] * j["walltime_seconds"] / 3600 for j in jobs.values()),
                           "controller_duration_days": 60, "automatic_retry": False}, "Changed budget")
    root = Path(d["root"])
    require(root.is_absolute() and root.resolve() == root and ".." not in root.parts
            and str(root).startswith("/lustre/fsn1/projects/rech/vas/uag72kz/grpo_full_budget_20260918/"),
            "Unexpected campaign output root")
    return d


def load_declaration(path, sha):
    binding = identity(path)
    require(binding["sha256"] == sha, "Declaration byte hash differs")
    require(not Path(path).stat().st_mode & 0o222, "Declaration must be immutable")
    return validate_declaration(read(path)), binding


def run_contract(d, binding, job, initialization):
    task, method = job["task"], job["method"]
    diagnostic = job["kind"] == "diagnostic"
    return {"protocol": PROTOCOL, "schema_version": 1, "stage": "grpo", "task": task,
            "method": method, "status": "development_only_not_publication_eligible",
            "horizon": HORIZONS[task], "fixed_horizon": {"max_steps": HORIZONS[task], "world_size": 4},
            "chunk_boundaries": [5, 10] if diagnostic else boundaries(task), "expected_train_count": COUNTS[task],
            "source": d["source"], "campaign_declaration": binding, "seed": 43,
            "initialization": initialization, "data_root": d["tasks"][task]["training_data_root"],
            "data_files": d["tasks"][task]["training_data_files"], "diagnostic": diagnostic,
            "output_dir": str(output_dir(d, job)), "teacher_weight": 0.0,
            "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                         "steps_per_generation": 3, "save_steps": 5 if diagnostic else 50,
                         "save_total_limit": 3},
            "sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 50,
                         "repetition_penalty": 1.0, "max_completion_length": 3000},
            "equal_tokens_or_compute_claimed": False}
