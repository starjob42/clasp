#!/usr/bin/env python3
"""Declare and monitor a fresh, consensus-checked six-arm incumbent/FOARL development pilot.

init never submits; status is read-only; run/watch require --submit to submit.
Each arm gets at most one attempt. Evaluation plans are declared before training; each role/arm gets one attempt.
No path invokes scancel. Native five-update diagnostics are separate campaigns.
"""
import argparse
import ast
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
TASKS = ("cvrp", "mis")
METHODS = ("grpo_z", "grpo_verified", "foarl_f638_dev")
QOS = "qos_gpu_a100-dev"
USER = "uag72kz"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
RUNTIME = Path("/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env")
DATA = Path("/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl")
SFT = Path("/lustre/fsn1/projects/rech/vas/uag72kz/clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14")
BENCHMARK = Path("/lustre/fsn1/projects/rech/vas/uag72kz/clasp_fair_development_benchmark_20260905_v1/data_v3")
BENCHMARK_SHA = "8d2565e29e60a1f5e12fe7fa3635bd2739cc108e7eb2415f112a8941d6ab4752"
OVERLAP_SHA = "fa10f2a70456223258d83e5d758acdfc552ac5a03681436a044f9969a072a7b7"
LAUNCHER = "scripts/train_grpo_incumbent_dev_a100.slurm"
EVAL_WRAPPER = "scripts/evaluate_grpo_incumbent_dev_a100.slurm"
SELF = "scripts/grpo_incumbent_dev_campaign.py"
REQUIRED_SOURCE_FILES = {
    LAUNCHER, EVAL_WRAPPER, SELF, "src/clasp/baselines/foarl_f638_dev_20260907.py",
    "src/clasp/entrypoints/train_foarl_f638_dev_20260907.py",
    "tests/test_grpo_incumbent_dev_launcher.py", "tests/test_grpo_incumbent_dev_campaign.py",
    "docs/grpo_incumbent_foarl_development_20260907.md",
    "src/clasp/rewards/cvrp_verified_improvement.py",
    "src/clasp/rewards/verified_improvement.py", "src/clasp/rewards/co_tasks.py",
    "src/clasp/trainers/primal_dual_grpo.py", "src/clasp/models/checkpoint_fidelity.py",
    "scripts/clasp_grpo_fair_dev_artifacts.py", "tests/test_grpo_graded_reward.py",
    "tests/test_verified_mis_improvement.py", "tests/test_grpo_verified_trainer.py",
    "tests/test_cvrp_verified_improvement.py", "tests/test_grpo_verified_dev_launcher.py",
    "tests/test_verified_teacher_integration.py", "docs/grpo_verified_improvement_development_20260906.md",
    "src/clasp/rewards/heads.py", "tests/test_grpo_graded_ground_truth.py",
    "scripts/probe_verified_grpo_distributed.py", "tests/test_grpo_verified_distributed.py",
    "src/clasp/rewards/verified_teacher_selection.py", "tests/test_verified_teacher_selection.py",
    "src/clasp/models/training_fidelity.py",
    "scripts/evaluate_grpo_verified_ddpfix.py",
}


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
    require(value.get(field) == digest({k: v for k, v in value.items() if k != field}), "invalid self hash: " + str(path))
    return value


def write_new(path, value, field="receipt_sha256"):
    value = {**value, field: digest(value)}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_files(source, expected_sha):
    source = Path(source).resolve(strict=True)
    manifest = identity(source / "SOURCE_MANIFEST.sha256")
    require(manifest["sha256"] == expected_sha, "source manifest differs from the approved pin")
    seen = set()
    for line in Path(manifest["path"]).read_text().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        require(match is not None, "malformed source manifest")
        sha, relative = match.groups()
        require(relative not in seen, "duplicate source manifest path")
        seen.add(relative)
        path = source / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source)
                and path.resolve().relative_to(source).as_posix() == relative
                and identity(path)["sha256"] == sha, "source file changed: " + relative)
    require(REQUIRED_SOURCE_FILES <= seen, "new launcher/helper/test missing from frozen source: " + str(sorted(REQUIRED_SOURCE_FILES - seen)))
    observed = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    require(observed == seen | {"SOURCE_MANIFEST.sha256"}, "source inventory has unmanifested files")
    require(identity(__file__)["sha256"] == identity(source / SELF)["sha256"], "executing controller differs from frozen source")
    return manifest


def variant(method):
    require(method in METHODS, "unknown method")
    enabled = method == "grpo_verified"
    return {"graded_reward": enabled, "verified_improvement": enabled,
            "improvement_weight": 0.05 if enabled else 0.0,
            "improvement_max_tokens": 3000, "improvement_max_passes": 2,
            "teacher_selection": "feasibility_incumbent" if enabled else "per_completion",
            "teacher_normalization": "optimizer_window" if enabled else "microbatch"}


def plans(output, source, initializations):
    plans = []
    for task in TASKS:
        for method in METHODS:
            label = f"{task}_{method}_seed43_step100"
            train = output / "train" / label
            stdout = output / "logs" / (label + "_%j.out")
            checkpoint = Path(initializations[task]["adapter_model.safetensors"]["path"]).parent
            settings = variant(method)
            values = {"DEV_TASK": task, "DEV_METHOD": method, "DEV_SFT_CHECKPOINT": checkpoint,
                      "DEV_OUTPUT_DIR": train, "DEV_PROJECT_ROOT": source, "DEV_SEALED_ENV": RUNTIME,
                      "DEV_DATA_ROOT": DATA, "DEV_SOURCE_MANIFEST_SHA256": identity(source / "SOURCE_MANIFEST.sha256")["sha256"],
                      "DEV_SEED": 43, "DEV_MAX_STEPS": 100,
                      "DEV_SAVE_STEPS": 100, "DEV_SAVE_TOTAL_LIMIT": 1, "DEV_RESUME_CHECKPOINT": "",
                      "DEV_ROLLOUT_BATCH_SIZE": "", "DEV_GROUP_STD_CORRECTION": 1,
                      "DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES": 0,
                      "DEV_TEACHER_SELECTION": settings["teacher_selection"],
                      "DEV_TEACHER_NORMALIZATION": settings["teacher_normalization"],
                      "GRPO_GRADED_REWARD": int(settings["graded_reward"]),
                      "GRPO_VERIFIED_IMPROVEMENT": int(settings["verified_improvement"]),
                      "GRPO_VERIFIED_IMPROVEMENT_WEIGHT": settings["improvement_weight"],
                      "GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS": 3000,
                      "GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES": 2}
            require(all("," not in str(v) and not any(c.isspace() for c in str(v)) for v in values.values()), "unsupported environment export path")
            argv = ["sbatch", "--parsable", "--job-name=" + f"incumbent-{task}-{method}-s43",
                    "--output=" + str(stdout), "--error=" + str(stdout).removesuffix(".out") + ".err",
                    "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()), str(source / LAUNCHER)]
            plans.append({"key": task + "/" + method, "role": "train", "task": task, "method": method,
                          "training_root": str(train), "stdout": str(stdout), "variant": settings, "command": argv})
    return plans


def initialize(output, source, source_sha, diagnostic_campaign):
    output, source = Path(output).resolve(), Path(source).resolve(strict=True)
    require(not output.exists() or set(output.iterdir()) == {Path(__file__).resolve()}, "initialization requires an unused output root or only this controller")
    manifest = source_files(source, source_sha)
    artifacts = module("verified_input_artifacts", source / "scripts/clasp_grpo_fair_dev_artifacts.py")
    parent = artifacts.preflight(source, RUNTIME, DATA, SFT)
    require(parent["source_manifest"] == manifest, "preflight source mismatch")
    benchmark = signed(BENCHMARK / "development_manifest.json", "manifest_sha256")
    overlap = signed(BENCHMARK / "training_overlap_audit.json", "audit_sha256")
    require(benchmark["manifest_sha256"] == BENCHMARK_SHA and overlap["audit_sha256"] == OVERLAP_SHA, "unexpected development benchmark")
    payload = {k: copy.deepcopy(v) for k, v in parent.items() if k not in ("receipt_sha256", "status")}
    payload.update(status="frozen_incumbent_foarl_six_arm_development_protocol_v1", controller=identity(__file__),
                   output_root=str(output), methods=list(METHODS), tasks=list(TASKS),
                   input_preflight_sha256=parent["receipt_sha256"],
                   method_settings={method: variant(method) for method in METHODS})
    payload["training"].update(prompt_groups_per_update=4, total_onpolicy_completions=3600,
                               supervised_overhead_is_additional=True, automatic_retries=False)
    payload["evaluation"].update(start=0, data_root=str(BENCHMARK), repair=False,
                                 inspected_development_data=True, untouched_confirmation=False,
                                 benchmark_manifest_sha256=BENCHMARK_SHA, overlap_audit_sha256=OVERLAP_SHA,
                                 submission_scope="Six raw-K8 evaluations are predeclared and only submitted after their verified training endpoint.")
    payload["fairness"] = {"matched": "Same source/runtime/SFT/data/seed/100 updates/B9/global36/3600 on-policy completions and optimizer/decoder settings.",
                           "additional_cost": "Verified repair/local search and supervised forward/backward work are measured separately; this is not a compute-matched superiority claim.",
                           "report": "Report all six train arms, terminal fidelity, auxiliary target rate/time/NLL, and actual allocated GPU-hours.",
                           "confirmation": "A later independently declared seed and untouched benchmark are required for confirmation; the existing data_v3 pilot is already inspected."}
    payload["planned_train_commands"] = plans(output, source, payload["initializations"])
    fresh = module("incumbent_fresh_benchmark", source / "scripts/clasp_grpo_fresh_dev_artifacts.py")
    payload["benchmark"] = fresh.validate_benchmark(BENCHMARK)
    payload["generation"] = dict(fresh.GENERATION)
    payload["training_data_files"] = training_data_files(DATA)
    payload["planned_eval_commands"] = evaluation_plans(output, source, payload["planned_train_commands"])
    payload["diagnostic_prerequisite"] = declare_diagnostics(diagnostic_campaign, source)
    payload["algorithm_differences"] = {
        "grpo_z": "strict feasible-only quality, group sample std, beta0",
        "grpo_verified": "graded violations plus feasibility_incumbent teacher NLL.05, optimizer_window",
        "foarl_f638_dev": "released two reward heads, sample std+1e-4, fixed initial SFT KL beta.05; common no-reason prompt and budget",
    }
    payload["evaluation_management"] = {"wrapper": identity(source / EVAL_WRAPPER), "helper": identity(__file__)}
    payload["queue"] = {"user": USER, "qos": QOS, "max_expanded_jobs": 10, "walltime_minutes": 120,
                        "gpus_per_job": 4, "preexisting_jobs_must_be_untouched": True}
    output.mkdir(exist_ok=True)
    for name in ("train", "evaluation", "logs", "ledgers", "ledgers/events", "ledgers/terminal", "ledgers/evaluation_events", "ledgers/evaluation_terminal"):
        (output / name).mkdir()
    # Retain the original sealed preflight without changing its fields.
    with (output / "ledgers/input_preflight.json").open("x") as handle:
        json.dump(parent, handle, indent=2, sort_keys=True); handle.write("\n")
    (output / "ledgers/input_preflight.json").chmod(0o440)
    return write_new(output / "ledgers/protocol.json", payload)


def protocol_for(output):
    protocol = signed(output / "ledgers/protocol.json")
    require(protocol["controller"] == identity(__file__), "controller changed after declaration")
    require(protocol["output_root"] == str(output), "campaign output binding mismatch")
    source = Path(protocol["source_manifest"]["path"]).parent
    source_files(source, protocol["source_manifest"]["sha256"])
    require(protocol["planned_train_commands"] == plans(output, source, protocol["initializations"]), "frozen command or recipe changed")
    require(protocol["planned_eval_commands"] == evaluation_plans(output, source, protocol["planned_train_commands"]), "evaluation command drift")
    require(protocol["training_data_files"] == training_data_files(Path(protocol["data_root"])), "training dataset files changed")
    for binding in protocol["evaluation_management"].values():
        require(identity(binding["path"]) == binding, "evaluation management changed")
    return protocol, source


def run_command(command):
    return subprocess.check_output(command, text=True)


def read_events(output, protocol, *, role="train"):
    plans_key = "planned_train_commands" if role == "train" else "planned_eval_commands"
    directory = "events" if role == "train" else "evaluation_events"
    result = {item["key"]: [] for item in protocol[plans_key]}
    expected = {item["key"]: item["command"] for item in protocol[plans_key]}
    for path in sorted((output / "ledgers" / directory).glob("*.json")):
        event = signed(path, "event_sha256")
        require(event["key"] in result and event["kind"] in ("intent", "submitted", "rejected"), "unknown submission event")
        require(event["protocol_sha256"] == protocol["receipt_sha256"] and event["command"] == expected[event["key"]], "event declaration mismatch")
        result[event["key"]].append(event)
    ids = set()
    for key, events in result.items():
        intents = [e for e in events if e["kind"] == "intent"]
        endings = [e for e in events if e["kind"] != "intent"]
        require(len(intents) <= 1 and len(endings) <= 1, "duplicate submission attempt")
        require(not endings or (intents and endings[0]["transaction"] == intents[0]["transaction"]), "orphan submission event")
        if endings and endings[0]["kind"] == "submitted":
            job = endings[0]["job_id"]
            require(re.fullmatch(r"[1-9][0-9]*", job) and job not in ids, "invalid or duplicate job ID")
            ids.add(job)
    return result


def scheduler_rows(raw):
    result = {}
    for fields in csv.reader(raw.splitlines(), delimiter="|"):
        if not fields or not any(fields):
            continue
        require(len(fields) == 5, "malformed accounting rows")
        job, state, code, stdout, elapsed = (x.strip() for x in fields)
        # Expanded unrelated arrays may appear in broad user accounting. Their
        # IDs are retained for inspection but cannot become our numeric jobs.
        require(re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", job), "unexpected scheduler step or job ID")
        row = {"job_id": job, "state": state.split()[0].rstrip("+"), "exit_code": code,
               "stdout": stdout, "elapsed_seconds": int(elapsed)}
        require(job not in result or result[job] == row, "conflicting accounting rows")
        result[job] = row
    return result


def expanded_queue(raw):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    require(all(len(r) == 2 and all(v.strip() for v in r) for r in rows), "malformed expanded queue")
    ids = [r[0].strip() for r in rows]
    require(len(ids) == len(set(ids)), "duplicate expanded queue IDs")
    return set(ids)


def validate_policy_consistency(consensus, fidelity, *, max_steps=100, gradients=False):
    """Require complete four-rank evidence, bound to the saved terminal policy."""
    require(max_steps in (5, 100) and (not gradients or max_steps == 5), "unsupported consistency profile")
    expected_steps = list(range(1, 6)) if max_steps == 5 else [1, 2, 5, *range(10, 101, 10)]
    expected_protocol = "clasp-pre-optimizer-exact-gradient-consistency-v1" if gradients else "clasp-optimizer-step-exact-policy-consistency-v1"
    steps_key = "verified_gradient_optimizer_steps" if gradients else "verified_optimizer_steps"
    require(isinstance(consensus, dict), "policy consistency receipt is not an object")
    require(consensus.get("receipt_sha256") == digest({k: v for k, v in consensus.items() if k != "receipt_sha256"}),
            "policy consistency self hash mismatch")
    require(type(consensus.get("schema_version")) is int and consensus["schema_version"] == 1
            and consensus.get("protocol") == expected_protocol
            and consensus.get("claim_scope") == "development"
            and consensus.get("status") == "passed", "policy consistency protocol/status/scope mismatch")
    require(not any(k in consensus for k in ("failed_optimizer_step", "error", "policy_mismatch_details")),
            "passed policy receipt contains failure evidence")
    expected_settings = {"enabled": True, "gradient_enabled": max_steps == 5, "initial_optimizer_steps": [1, 2, 5], "interval": 1 if max_steps == 5 else 10}
    settings = consensus.get("settings")
    require(isinstance(settings, dict) and settings == expected_settings
            and settings["enabled"] is True and settings["gradient_enabled"] is (max_steps == 5)
            and type(settings["interval"]) is int
            and all(type(step) is int for step in settings["initial_optimizer_steps"]),
            "policy consistency settings differ from the fixed production guard")
    steps = consensus.get(steps_key)
    require(isinstance(steps, list) and all(type(step) is int for step in steps) and steps == expected_steps,
            "missing exact distributed policy checks at declared optimizer steps")
    checks = consensus.get("checks")
    require(isinstance(checks, list) and len(checks) == len(expected_steps), "policy consistency checks are incomplete")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    for step, check in zip(expected_steps, checks):
        require(isinstance(check, dict) and type(check.get("optimizer_step")) is int
                and check["optimizer_step"] == step, "policy consistency check order/step mismatch")
        inventory_sha = check.get("tensor_inventory_sha256")
        require(isinstance(inventory_sha, str) and sha_pattern.fullmatch(inventory_sha), "invalid policy tensor inventory hash")
        distributed = check.get("distributed")
        require(isinstance(distributed, dict)
                and distributed.get("phase") == (f"before_optimizer_step_{step}_gradient_consistency" if gradients else f"optimizer_step_{step}_policy_consistency")
                and type(distributed.get("world_size")) is int and distributed["world_size"] == 4,
                "policy consistency phase or four-rank world size mismatch")
        consensus_sha = distributed.get("consensus_digest")
        require(isinstance(consensus_sha, str) and sha_pattern.fullmatch(consensus_sha), "invalid distributed policy digest")
        ranks = distributed.get("rank_digests")
        require(isinstance(ranks, list) and len(ranks) == 4
                and all(isinstance(row, dict) and type(row.get("rank")) is int for row in ranks),
                "missing distributed rank digests")
        require(sorted(row["rank"] for row in ranks) == [0, 1, 2, 3]
                and all(row.get("digest") == consensus_sha for row in ranks),
                "distributed rank policy digests do not agree")
    if gradients:
        require(all(check.get("gradient_tensor_count") == len(fidelity["trainable_closure"]["tensors"]) for check in checks), "missing trainable gradients")
        return
    expected_inventory = fidelity.get("trainable_closure", {}).get("tensor_inventory_sha256")
    require(isinstance(expected_inventory, str) and sha_pattern.fullmatch(expected_inventory)
            and checks[-1]["tensor_inventory_sha256"] == expected_inventory,
            "step-100 policy inventory differs from the certified checkpoint")


def terminal_validation(item, source, protocol, job_id=None):
    train = Path(item["training_root"])
    is_foarl = item["method"] == "foarl_f638_dev"
    contract = json.loads((train / "local_development_run_contract.json").read_text())
    require(contract.get("protocol") == "grpo-incumbent-foarl-matched-development-20260907-v1"
            and contract.get("schema_version") == 2 and contract.get("distributed_correctness_diagnostic") is False,
            "wrong launch protocol or diagnostic endpoint")
    require(job_id is not None and contract.get("execution", {}).get("slurm_job_id") == str(job_id),
            "actual training job differs from immutable submission")
    require(contract["execution"]["source_manifest_expected_sha256"] == protocol["source_manifest"]["sha256"], "launch source expectation mismatch")
    require(contract["launcher"]["path"] == str(source / LAUNCHER)
            and contract["launcher"]["sha256"] == identity(source / LAUNCHER)["sha256"], "actual training launcher bytes differ")
    require(contract["task"] == item["task"] and contract["method"] == item["method"] and contract["seed"] == 43, "terminal arm identity mismatch")
    require(contract["fixed_horizon"] == {"max_steps": 100, "world_size": 4}, "terminal budget mismatch")
    require(contract["continuation"] == {"resume_checkpoint": None, "resume_global_step": 0, "strict_optimizer_resume": False}, "unexpected resumed training")
    require(contract["plain_grpo_variant"] == item["variant"], "variant coefficients differ from declaration")
    initial = protocol["initializations"][item["task"]]["adapter_model.safetensors"]
    require(contract["initialization"]["adapter_model_sha256"] == initial["sha256"] and Path(contract["initialization"]["checkpoint"]) == Path(initial["path"]).parent, "initial SFT mismatch")
    require(contract["source"] == {"root": str(source), "manifest_sha256": protocol["source_manifest"]["sha256"]}, "terminal source mismatch")
    shared = {"rollout_batch_b": 9, "global_generation_batch_size": 36, "learning_rate": 1e-6,
              "max_completion_length": 3000, "max_prompt_length": 29768, "per_device_train_batch_size": 3,
              "gradient_accumulation_steps": 3, "beta": 0.05 if is_foarl else 0.0, "deployment_set_k": 8, "dual_learning_rate": 0.0,
              "recovery_scale": 0.0, "prompt_style": "jiang_f638", "no_reason": True}
    require(all(contract["production_preserved"].get(k) == v for k, v in shared.items()), "shared training settings changed")
    policy = {"group_std_correction": 1, "importance_sampling_level": "token", "policy_length_alpha": 0.0,
              "canonical_set": False, "use_clasp": False, "canonical_grpo_anchor_scale": 0.0,
              "canonical_quality_anchor_ratio": 0.0, "canonical_dense_quality_scale": 0.0,
              "canonical_dense_feasibility_scale": 0.0, "canonical_dense_violation_scale": 0.0,
              "canonical_dense_recovery_scale": 0.0, "component_dual_recovery_scale": 0.0}
    require(all(contract["policy_estimator"].get(k) == v for k, v in policy.items()), "plain-GRPO or CLASP-disabled contract mismatch")
    hp = json.loads((train / "hyperparams.json").read_text())
    require(hp["dataset"]["train_size"] == (2000 if item["task"] == "cvrp" else 2900), "effective dataset size changed")
    cfg = hp["trainer_config"]
    required = {"max_steps": 100, "num_generations": 9, "generation_batch_size": 36, "seed": 43,
                "data_seed": 43, "learning_rate": 1e-6, "optim": "adamw_8bit", "weight_decay": 0.01,
                "warmup_ratio": 0.03, "lr_scheduler_type": "linear", "gradient_accumulation_steps": 3,
                "per_device_train_batch_size": 3, "max_prompt_length": 29768, "max_completion_length": 3000,
                "num_iterations": 1, "loss_type": "grpo", "beta": 0.05 if is_foarl else 0.0, "scale_rewards": "group" if is_foarl else "none",
                "epsilon": 0.2, "epsilon_high": 0.2, "importance_sampling_level": "token"}
    require(all(cfg.get(k) == v for k, v in required.items()), "actual trainer settings changed")
    validate_actual_arm(item, hp, contract, source, protocol)
    cp = train / "checkpoints/checkpoint-100"
    state = json.loads((cp / "trainer_state.json").read_text())
    rows = [row for row in state["log_history"] if "loss" in row]
    require(state["global_step"] == state["max_steps"] == 100 and [row["step"] for row in rows] == list(range(1, 101)), "terminal optimizer history is not exactly 100 steps")
    require(all(math.isfinite(float(row["loss"])) and math.isfinite(float(row.get("grad_norm", 0))) for row in rows), "nonfinite terminal training history")
    fidelity = module("verified_terminal_fidelity", source / "src/clasp/models/checkpoint_fidelity.py").verify_checkpoint_fidelity(cp)
    require(fidelity["status"] == "exact_trainable_policy_persisted", "checkpoint did not persist exact policy")
    consensus = signed(train / "checkpoints/policy_consistency_checks.json")
    validate_policy_consistency(consensus, fidelity)
    names = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json", "trainer_state.json",
             "added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")
    telemetry = {key: {"count": len(values), "mean": sum(values) / len(values), "last": values[-1]}
                 for key in sorted({key for row in rows for key in row if "grpo_verified_" in key})
                 if (values := [row[key] for row in rows if key in row])}
    reference = exact_reference_evidence(train, hp, contract, source) if is_foarl else None
    return {"status": "terminal_training_verified", "key": item["key"], "global_step": 100,
            "exact_reference_evidence": reference,
            "job_id": str(job_id), "policy_consistency": identity(train / "checkpoints/policy_consistency_checks.json"),
            "contract": identity(train / "local_development_run_contract.json"), "hyperparams": identity(train / "hyperparams.json"),
            "checkpoint": {name: identity(cp / name) for name in names}, "native_fidelity_sha256": fidelity["receipt_sha256"],
            "persisted_trainable_tensors": len(fidelity["trainable_closure"]["tensors"]), "auxiliary_training_diagnostics": telemetry}


def reconcile(output, *, submit=False, record=False, runner=run_command, validate=None, max_new_jobs=5, role="train"):
    require(role in ("train", "eval"), "unsupported job role")
    validate = validate or (terminal_validation if role == "train" else evaluation_validation)
    plans_key = "planned_train_commands" if role == "train" else "planned_eval_commands"
    event_directory = "events" if role == "train" else "evaluation_events"
    terminal_directory = "terminal" if role == "train" else "evaluation_terminal"
    require(type(max_new_jobs) is int and 0 <= max_new_jobs <= 5, "each tick admits at most five new jobs")
    output = Path(output).resolve(strict=True)
    protocol, source = protocol_for(output)
    prerequisite = check_diagnostics(protocol, source, runner=runner)
    if submit and not prerequisite["passed"]:
        diagnostics = prerequisite.get("observations", [])
        require(len(diagnostics) == len(DIAGNOSTIC_JOBS) and all(row.get("passed") is True or row.get("state") in ACTIVE | {"accounting_unavailable"}
                                             for row in diagnostics), "native correctness prerequisites failed or are malformed")
        return {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "diagnostic_prerequisite": prerequisite,
                "observations": [{"key": item["key"], "state": "waiting_for_native_correctness_prerequisites"}
                                 for item in protocol[plans_key]],
                ("all_training_verified" if role == "train" else "all_evaluation_verified"): False}
    events = read_events(output, protocol, role=role)
    all_events = [read_events(output, protocol, role=kind) for kind in ("train", "eval")]
    trusted = {event["job_id"]: key for key, group in events.items() for event in group if event["kind"] == "submitted"}
    submitted_ids = [event["job_id"] for collection in all_events for group in collection.values() for event in group if event["kind"] == "submitted"]
    all_trusted = set(submitted_ids)
    require(len(submitted_ids) == len(all_trusted), "one submitted job ID is bound to multiple campaign roles or arms")
    date = protocol["created_at_utc"].split("T")[0]
    query = ["sacct", "-X", "-P", "-n", "-u", USER, "-S", date, "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw"]
    rows = scheduler_rows(runner(query))
    missing = all_trusted - set(rows)
    if missing:
        rows.update(scheduler_rows(runner(["sacct", "-X", "-P", "-n", "-j", ",".join(sorted(missing, key=int)), "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw"])))
    observations, accepted = [], set()
    for item in protocol[plans_key]:
        key, expected = item["key"], item["stdout"]
        if role == "eval" and not (output / "ledgers/terminal" / (key.replace("/", "_") + ".json")).is_file():
            observations.append({"key": key, "state": "waiting_for_verified_training"})
            continue
        matching = [row for job, row in rows.items() if row["stdout"] in (expected, expected.replace("%j", job))
                    or (row["stdout"].lower() in ("", "none", "unknown", "(null)") and trusted.get(job) == key)]
        require(len(matching) <= 1, "multiple jobs bind the same arm")
        if matching:
            row = matching[0]
            require(trusted.get(row["job_id"]) == key and re.fullmatch(r"[1-9][0-9]*", row["job_id"]), "scheduler job does not match the submitted-event job ID")
            require(not any(job != row["job_id"] and bound == key for job, bound in trusted.items()), "job conflicts with immutable ledger")
            status = {"key": key, **row}
            if row["state"] == "COMPLETED" and row["exit_code"] == "0:0":
                receipt_path = output / "ledgers" / terminal_directory / (key.replace("/", "_") + ".json")
                if receipt_path.exists():
                    receipt = signed(receipt_path)
                    require(receipt["job_id"] == row["job_id"] and receipt["protocol_sha256"] == protocol["receipt_sha256"], "cached terminal audit binding mismatch")
                else:
                    receipt = {**validate(item, source, protocol, job_id=row["job_id"]), "job_id": row["job_id"], "protocol_sha256": protocol["receipt_sha256"]}
                    if record:
                        receipt = write_new(receipt_path, receipt)
                status["terminal_verified"] = True
            observations.append(status)
            continue
        if events[key]:
            observations.append({"key": key, "state": "prior_attempt_awaiting_accounting_no_retry"})
            continue
        require(not Path(item["training_root"] if role == "train" else item["output"]).exists(), "unbound or partial output exists")
        if role == "eval":
            training_endpoint(output, item["key"], protocol, source, runner=runner)
        queued = expanded_queue(runner(["squeue", "-r", "-h", "-u", USER, "-q", QOS, "-o", "%i|%T"]))
        lagging = {job for job in all_trusted if job not in rows or rows[job]["state"] in ACTIVE}
        unresolved_intents = sum(bool(group) and not any(e["kind"] == "submitted" for e in group)
                                 for collection in all_events for group in collection.values())
        occupied = len(queued) + len((lagging | accepted) - queued) + unresolved_intents
        if not submit or occupied >= 10 or len(accepted) >= max_new_jobs:
            observations.append({"key": key, "state": "ready", "occupied_slots": occupied, "command": item["command"]})
            continue
        payload = {"key": key, "protocol_sha256": protocol["receipt_sha256"], "command": item["command"], "transaction": uuid.uuid4().hex,
                   "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_new(output / "ledgers" / event_directory / (uuid.uuid4().hex + ".json"), {**payload, "kind": "intent"}, "event_sha256")
        try:
            answer = runner(item["command"]).strip()
            require(re.fullmatch(r"[1-9][0-9]*(;[^\s;]+)?", answer), "sbatch did not return a unique job ID")
            job = answer.split(";", 1)[0]
            require(job not in all_trusted and job not in accepted, "sbatch returned an already bound job ID")
        except Exception as error:
            write_new(output / "ledgers" / event_directory / (uuid.uuid4().hex + ".json"), {**payload, "kind": "rejected", "error": str(error)}, "event_sha256")
            raise
        write_new(output / "ledgers" / event_directory / (uuid.uuid4().hex + ".json"), {**payload, "kind": "submitted", "job_id": job}, "event_sha256")
        accepted.add(job)
        observations.append({"key": key, "state": "submitted", "job_id": job})
    return {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "diagnostic_prerequisite": prerequisite, "observations": observations,
            ("all_training_verified" if role == "train" else "all_evaluation_verified"): len(observations) == len(TASKS) * len(METHODS) and all(item.get("terminal_verified") for item in observations)}


def locked_run(output, submit, record, max_new_jobs=5, role="train"):
    with (output / "ledgers/controller.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result = {}
        remaining = max_new_jobs
        for kind in (("train", "eval") if role == "both" else (role,)):
            result[kind] = reconcile(output, submit=submit, record=record, max_new_jobs=remaining, role=kind)
            remaining -= sum(row["state"] == "submitted" for row in result[kind]["observations"])
        if record and result.get("train", {}).get("all_training_verified"):
            result["accounting"] = finalize_training(output)
        return result


def training_data_files(root):
    files = {}
    for task in TASKS:
        directory = Path(root) / task / "train"
        task_files = sorted(path for path in directory.rglob("*") if path.is_file())
        require(task_files and all(not path.is_symlink() for path in task_files), "missing or linked training data")
        files[task] = {path.relative_to(directory).as_posix(): identity(path) for path in task_files}
    return files


def evaluation_plans(output, source, train_plans):
    result = []
    for train in train_plans:
        label = train["key"].replace("/", "_") + "_raw_k8"
        destination = output / "evaluation" / label
        stdout = output / "logs" / (label + "_%j.out")
        values = {"INCUMBENT_CAMPAIGN": output, "INCUMBENT_SOURCE": source,
                  "INCUMBENT_SOURCE_SHA256": identity(source / "SOURCE_MANIFEST.sha256")["sha256"],
                  "INCUMBENT_TASK": train["task"], "INCUMBENT_METHOD": train["method"],
                  "INCUMBENT_RUNTIME": RUNTIME, "INCUMBENT_OUTPUT": destination}
        require(all("," not in str(v) and not any(c.isspace() for c in str(v)) for v in values.values()), "unsupported evaluation export path")
        command = ["sbatch", "--parsable", "--job-name=incumbent-eval-" + train["key"].replace("/", "-"),
                   "--output=" + str(stdout), "--error=" + str(stdout).removesuffix(".out") + ".err",
                   "--export=ALL," + ",".join(f"{k}={v}" for k, v in values.items()), str(source / EVAL_WRAPPER)]
        result.append({"key": train["key"], "role": "eval", "task": train["task"], "method": train["method"],
                       "training_root": train["training_root"],
                       "checkpoint": str(Path(train["training_root"]) / "checkpoints/checkpoint-100"),
                       "output": str(destination), "stdout": str(stdout), "command": command})
    return result


# The first diagnostic source exposed a dtype-name mismatch in the KL auditor.
# Four fresh native jobs on the corrected source must be pinned before init.
DIAGNOSTIC_SOURCE_SHA256 = "623630f4b17b586d5ec317b7e4dbcad65c307ec3a516fcd05ad15972fac3abe5"
DIAGNOSTIC_JOBS = {"cvrp_grpo_verified": "1839879", "mis_grpo_verified": "1839880",
                   "cvrp_foarl_f638_dev": "1839884", "mis_foarl_f638_dev": "1839885"}


def manifest_entries(source):
    result = {}
    for line in (Path(source) / "SOURCE_MANIFEST.sha256").read_text().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        require(match is not None and match[2] not in result, "malformed or duplicate manifest entry")
        result[match[2]] = match[1]
    return result


def declare_diagnostics(campaign, source):
    require(re.fullmatch(r"[a-f0-9]{64}", DIAGNOSTIC_SOURCE_SHA256)
            and all(isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job) for job in DIAGNOSTIC_JOBS.values())
            and len(set(DIAGNOSTIC_JOBS.values())) == 4, "four corrected native diagnostic source/job pins are not yet declared")
    campaign = Path(campaign).resolve(strict=True)
    declaration = json.loads((campaign / "protocol.json").read_text())
    require(declaration["status"] == "five_update_correctness_only_excluded_from_performance"
            and declaration["source_manifest_sha256"] == DIAGNOSTIC_SOURCE_SHA256
            and declaration["automatic_retry"] is False, "unexpected native diagnostic declaration")
    require(declaration["gpus_per_job"] == 4 and declaration["walltime"] == "00:25:00",
            "corrected native diagnostics require four A100s with a declared 25-minute limit")
    diagnostic_source = Path(declaration["source"]).resolve(strict=True)
    require(identity(diagnostic_source / "SOURCE_MANIFEST.sha256")["sha256"] == DIAGNOSTIC_SOURCE_SHA256,
            "diagnostic source manifest changed")
    observed, new = manifest_entries(diagnostic_source), manifest_entries(source)
    scientific = {name: sha for name, sha in observed.items() if name.startswith("src/clasp/") and name.endswith(".py") or name == LAUNCHER}
    current = {name: sha for name, sha in new.items() if name.startswith("src/clasp/") and name.endswith(".py") or name == LAUNCHER}
    require(scientific == current, "100-update source changes diagnosed training/algorithm/launcher bytes")
    require(len(declaration["plans"]) == 4 and {plan["key"] for plan in declaration["plans"]} == set(DIAGNOSTIC_JOBS), "wrong diagnostic arm set")
    jobs = []
    for plan in declaration["plans"]:
        stdout = str(campaign / "logs" / (plan["key"] + "_%j.out"))
        require(plan["output"] == str(campaign / plan["key"])
                and plan["command"][:2] == ["sbatch", "--parsable"]
                and plan["command"][-1] == str(diagnostic_source / LAUNCHER)
                and [arg for arg in plan["command"] if arg.startswith("--time=")] == ["--time=00:25:00"]
                and [arg for arg in plan["command"] if arg.startswith("--output=")] == ["--output=" + stdout],
                "diagnostic launcher, output or walltime command differs")
        path = campaign / (plan["key"] + "_submission.json")
        event = json.loads(path.read_text())
        require(event["key"] == plan["key"] and event["returncode"] == 0
                and event["job_id"] == DIAGNOSTIC_JOBS[plan["key"]]
                and event["stdout"].strip().split(";")[0] == event["job_id"], "native diagnostic submission changed")
        jobs.append({"key": plan["key"], "job_id": event["job_id"], "output": plan["output"],
                     "stdout": stdout, "submission": identity(path)})
    amendments = {name: identity(campaign / name) for name in (
        "pending_diagnostic_walltime_amendment_v1.json", "pending_diagnostic_walltime_amendment_result_v1.json")
        if (campaign / name).is_file()}
    return {"protocol": identity(campaign / "protocol.json"), "source": str(diagnostic_source),
            "source_manifest": identity(diagnostic_source / "SOURCE_MANIFEST.sha256"),
            "scientific_files": scientific, "jobs": jobs, "operational_amendments": amendments}


def check_diagnostics(protocol, source, *, runner=run_command):
    expected = protocol["diagnostic_prerequisite"]
    current = declare_diagnostics(Path(expected["protocol"]["path"]).parent, source)
    require(current == expected, "diagnostic prerequisite identities changed")
    ids = [job["job_id"] for job in current["jobs"]]
    raw = runner(["sacct", "-X", "-P", "-n", "-j", ",".join(ids),
                  "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw,TimelimitRaw,AllocTRES,QOS,Account,Partition"])
    rows, allocations = {}, {}
    for fields in csv.reader(raw.splitlines(), delimiter="|"):
        if not fields or not any(fields):
            continue
        require(len(fields) == 10 and fields[0] in ids and fields[0] not in rows, "invalid native diagnostic accounting row")
        rows.update(scheduler_rows("|".join(fields[:5])))
        allocations[fields[0]] = dict(zip(("time_limit_minutes", "allocated_tres", "qos", "account", "partition"), fields[5:]))
    observations = []
    fidelity_module = None
    for job in current["jobs"]:
        row = rows.get(job["job_id"])
        observation = {"key": job["key"], "job_id": job["job_id"], "passed": False,
                       "state": row["state"] if row else "accounting_unavailable"}
        if row and row["state"] == "COMPLETED" and row["exit_code"] == "0:0":
            allocation = allocations[job["job_id"]]
            require(row["stdout"] in (job["stdout"], job["stdout"].replace("%j", job["job_id"]))
                    and allocation["time_limit_minutes"] == "25"
                    and allocation["qos"] == QOS and allocation["account"] == "vas@a100"
                    and allocation["partition"] == "gpu_p5"
                    and re.search(r"(?:^|,)gres/gpu=4(?:,|$)", allocation["allocated_tres"])
                    and 0 < row["elapsed_seconds"] <= 1500, "actual native diagnostic allocation or stdout differs")
            train = Path(job["output"])
            contract = json.loads((train / "local_development_run_contract.json").read_text())
            task, method = job["key"].split("_", 1)
            require(contract["execution"]["slurm_job_id"] == job["job_id"]
                    and contract["task"] == task and contract["method"] == method
                    and contract["fixed_horizon"] == {"max_steps": 5, "world_size": 4}
                    and contract["distributed_correctness_diagnostic"] is True
                    and contract["source"] == {"root": current["source"], "manifest_sha256": DIAGNOSTIC_SOURCE_SHA256},
                    "native diagnostic launch/source/horizon mismatch")
            require(contract["plain_grpo_variant"] == variant(method)
                    and contract["initialization"]["adapter_model_sha256"] == protocol["initializations"][task]["adapter_model.safetensors"]["sha256"],
                    "diagnostic recipe or common SFT initialization mismatch")
            cp = train / "checkpoints/checkpoint-5"
            state = json.loads((cp / "trainer_state.json").read_text())
            require(state["global_step"] == state["max_steps"] == 5, "native diagnostic did not end at five updates")
            if fidelity_module is None:
                fidelity_module = module("incumbent_diag_fidelity", source / "src/clasp/models/checkpoint_fidelity.py")
            fidelity = fidelity_module.verify_checkpoint_fidelity(cp)
            require(fidelity["status"] == "exact_trainable_policy_persisted", "native diagnostic checkpoint fidelity failed")
            for gradients, name in ((False, "policy_consistency_checks.json"), (True, "gradient_consistency_checks.json")):
                validate_policy_consistency(signed(train / "checkpoints" / name), fidelity, max_steps=5, gradients=gradients)
            observation.update(passed=True, fidelity=identity(cp / "checkpoint_fidelity.json"),
                               contract=identity(train / "local_development_run_contract.json"), allocation=allocation,
                               elapsed_seconds=row["elapsed_seconds"])
            if method == "foarl_f638_dev":
                hp = json.loads((train / "hyperparams.json").read_text())
                observation["reference_evidence"] = exact_reference_evidence(train, hp, contract, source, step=5)
        observations.append(observation)
    return {"passed": len(observations) == 4 and all(row["passed"] for row in observations), "observations": observations}


def validate_actual_arm(item, hp, contract, source, protocol):
    task, method = item["task"], item["method"]
    train = Path(item["training_root"])
    initial = protocol["initializations"][task]
    checkpoint = str(Path(initial["adapter_model.safetensors"]["path"]).parent)
    for name, field in (("adapter_model.safetensors", "adapter_model_sha256"),
                        ("adapter_config.json", "adapter_config_sha256"), ("checkpoint_fidelity.json", "checkpoint_fidelity_sha256")):
        require(contract["initialization"][field] == initial[name]["sha256"], "actual SFT identity mismatch: " + name)
    require(hp["stage"] == "grpo" and hp["tasks"] == [task] and hp["distributed"] == {"world_size": 4, "rank": 0}, "actual stage/task/device mismatch")
    require(hp["dataset"] == {"train_size": 2000 if task == "cvrp" else 2900, "eval_size": 0,
                              "train_task_counts": {task: 2000 if task == "cvrp" else 2900}, "eval_task_counts": {}}, "actual dataset composition mismatch")
    args, stage = hp["args"], hp["stage_hyperparameters"]
    require(hp["model_path"] == args["model_path"] == args["tokenizer_path"] == stage["tokenizer_path"] == checkpoint, "actual model/tokenizer path mismatch")
    require(hp["output_dir"] == args["output_dir"] == str(train)
            and args["output_root"] == str(train.parent) and args["data_root"] == protocol["data_root"], "actual output/data mismatch")
    require(hp["artifacts"] == {"model_dir": str(train), "checkpoint_dir": str(train / "checkpoints")}, "actual artifact layout mismatch")
    require(hp["trainer_config"]["output_dir"] == str(train / "checkpoints")
            and Path(hp["trainer_config"]["logging_dir"]).is_relative_to(train / "checkpoints")
            and hp["trainer_config"]["run_name"] == hp["run_name"], "actual trainer output/run mismatch")
    require(stage["training_run_contract_path"] == str(train / "local_development_run_contract.json"), "stage contract path mismatch")
    contract_file = identity(stage["training_run_contract_path"])
    require(stage["training_run_contract_identity"] == {
        "protocol": "clasp-training-run-contract-file-identity-v1", "status": "immutable_verified",
        "file_sha256": contract_file["sha256"], "payload_sha256": digest(contract), "size_bytes": contract_file["size_bytes"],
        "payload_schema_version": 2, "payload_stage": "grpo", "fixed_horizon_max_steps": 100,
        "fixed_horizon_world_size": 4}, "stage does not bind the actual launch-contract bytes")
    require(contract["runtime"]["environment"] == protocol["runtime"]
            and contract["runtime"]["environment_is_publication_sealed"] is False
            and all(isinstance(contract["runtime"].get(key), str) and contract["runtime"][key]
                    for key in ("python", "torch", "transformers", "trl", "peft", "bitsandbytes")), "actual runtime identity incomplete")
    require(contract["compute"] == {"accelerator": "A100", "gpu_count": 4, "qos": QOS, "walltime": "02:00:00"}, "actual hardware declaration mismatch")
    require(args["no_reason"] is True and stage["no_reason"] is True
            and args["prompt_style"] == stage["prompt_style"] == "jiang_f638", "actual common prompt mismatch")
    is_foarl = method == "foarl_f638_dev"
    require(args["use_primal_dual"] is (not is_foarl) and stage["use_primal_dual"] is (not is_foarl)
            and args["use_clasp"] is False and stage["use_clasp"] is False
            and args["verify_gt_rewards"] is (not is_foarl), "actual algorithm dispatch mismatch")
    require(contract["budget_accounting"]["onpolicy_completions"] == 3600
            and contract["budget_accounting"]["prompt_groups"] == 400, "actual declared sample count mismatch")
    if is_foarl:
        profile = stage["runtime_profile"]
        require(profile["method_id"] == "foarl_f638_dev" and profile["profile_id"] == "foarl-f638-matched-b9-dev-20260907-v1"
                and profile["source_binding"]["manifest"] == protocol["source_manifest"], "FOARL runtime/source profile mismatch")
        require(hp["trainer_config"]["reward_weights"] == [1.0, 1.0]
                and hp["trainer_config"]["sync_ref_model"] is False, "FOARL reward or reference semantics mismatch")
        require(stage["kl_reference_mode"] == "explicit_frozen_exact_peft_step_zero"
                and stage["kl_reference_model_path"] == checkpoint
                and stage["kl_reference_receipt_path"] == str(Path(checkpoint) / "checkpoint_fidelity.json"), "FOARL reference is not the frozen initial SFT")
        reference_receipt = Path(stage["kl_reference_runtime_receipt_path"])
        require(reference_receipt.is_relative_to(train) and reference_receipt.is_file(), "missing actual frozen-reference runtime proof")
        reference, _ = exact_reference_modules(source)
        reference.inspect_runtime_receipt(reference_receipt, claim_scope="development", rehash_sources=False)
    else:
        require(stage["runtime_profile"] is None and stage["kl_reference_mode"] == "none_beta_zero", "unexpected baseline reference/runtime")


def exact_reference_modules(source):
    source = Path(source).resolve(strict=True)
    sys.path.insert(0, str(source / "src"))
    import importlib
    reference = importlib.import_module("clasp.models.exact_peft_kl_reference")
    post = importlib.import_module("clasp.models.exact_peft_kl_post_step")
    for name, loaded in (("exact_peft_kl_reference", reference), ("exact_peft_kl_post_step", post)):
        require(Path(loaded.__file__).resolve() == source / "src/clasp/models" / (name + ".py"), "exact-reference verifier imported from another source")
    return reference, post


def exact_reference_evidence(train, hp, contract, source, *, step=100):
    reference, post = exact_reference_modules(source)
    checkpoint = train / f"checkpoints/checkpoint-{step}"
    pre_path = train / reference.RECEIPT_FILENAME
    pre = reference.inspect_runtime_receipt(pre_path, claim_scope="development", rehash_sources=True)
    binding = pre["source_binding"]
    require(binding["adapter_path"] == binding["tokenizer_path"] == contract["initialization"]["checkpoint"]
            and binding["base_model_path"] == contract["initialization"]["base_model"]
            and binding["adapter_config_sha256"] == contract["initialization"]["adapter_config_sha256"]
            and binding["authoritative_receipt_file_sha256"] == contract["initialization"]["checkpoint_fidelity_sha256"],
            "exact-reference proof names different initial model/tokenizer/base/authority")
    require(pre["training_contract"]["beta"] == 0.05 and pre["training_contract"]["dtype"] == "bfloat16"
            and pre["distributed"]["world_size"] == 4, "exact-reference runtime beta/dtype/world mismatch")
    cp_pre = reference.inspect_runtime_receipt(checkpoint / reference.RECEIPT_FILENAME, claim_scope="development", rehash_sources=False)
    require(pre == cp_pre, "root/checkpoint step-zero reference proofs differ")
    root_step1 = post.inspect_post_step_receipt(train / post.STEP1_RECEIPT_FILENAME, claim_scope="development",
                                               expected_global_step=1, expected_artifact_role="step1", rehash_sources=False, rehash_policy_artifact=False)
    cp_step1 = post.inspect_post_step_receipt(checkpoint / post.STEP1_RECEIPT_FILENAME, claim_scope="development",
                                             expected_global_step=1, expected_artifact_role="step1", rehash_sources=False, rehash_policy_artifact=False)
    require(root_step1 == cp_step1, "root/checkpoint step-one reference proofs differ")
    terminal = post.inspect_post_step_receipt(checkpoint / post.POST_STEP_RECEIPT_FILENAME, claim_scope="development",
                                              expected_global_step=step, expected_artifact_role="checkpoint", rehash_sources=False, rehash_policy_artifact=True)
    run_identity = hp["stage_hyperparameters"]["training_run_contract_identity"]
    pre_identity = identity(pre_path)
    runtime = pre["runtime_pair"]
    prepared = runtime["prepared_runtime"]
    expected_pre = {"protocol": pre["protocol"], "status": pre["status"],
                    "receipt_sha256": pre["receipt_sha256"], "receipt_file_sha256": pre_identity["sha256"],
                    "reference_tensor_count": runtime["tensor_count"],
                    "reference_parameter_count": runtime["parameter_count"],
                    "reference_tensor_inventory_sha256": runtime["tensor_inventory_sha256"],
                    "model_class": runtime["model_class"], "prepared_class": prepared["reference_prepared_class"],
                    "unwrapped_class": prepared["reference_unwrapped_class"], "active_adapter": runtime["active_adapter"],
                    "model_config_sha256": runtime["model_config_sha256"], "device_types": list(prepared["device_types"])}
    for value in (root_step1, terminal):
        require(value["training_run_contract"] == run_identity
                and value["pre_step_reference"] == expected_pre, "reference post-step source/run lineage mismatch")
    expected_step1 = post.step1_reference_binding(train / post.STEP1_RECEIPT_FILENAME,
                                                claim_scope="development", rehash_sources=False)
    require(terminal["step1_reference"] == expected_step1, "terminal reference proof belongs to another step-one policy")
    return {"status": "exact_frozen_initial_sft_reference_lineage_verified",
            "files": {str(path.relative_to(train)): identity(path) for path in (
                pre_path, checkpoint / reference.RECEIPT_FILENAME, train / post.STEP1_RECEIPT_FILENAME,
                checkpoint / post.STEP1_RECEIPT_FILENAME, checkpoint / post.POST_STEP_RECEIPT_FILENAME)},
            "pre_step_receipt_sha256": pre["receipt_sha256"], "terminal_reference_receipt_sha256": terminal["receipt_sha256"]}


def compare_actual_configuration(task, baseline, modified, protocol, source):
    methods = (baseline["key"].split("/")[1], modified["key"].split("/")[1])
    require(methods[0] == "grpo_z" and methods[1] in METHODS[1:], "unsupported comparison pair")
    hp, contracts = [], []
    for method, audit in zip(methods, (baseline, modified)):
        require(audit["protocol_sha256"] == protocol["receipt_sha256"] and audit["key"] == task + "/" + method, "comparison audit identity mismatch")
        for field in ("hyperparams", "contract"):
            require(identity(audit[field]["path"]) == audit[field], "audited metadata changed")
        h = json.loads(Path(audit["hyperparams"]["path"]).read_text())
        c = json.loads(Path(audit["contract"]["path"]).read_text())
        item = next(item for item in protocol["planned_train_commands"] if item["key"] == audit["key"])
        validate_actual_arm(item, h, c, source, protocol)
        hp.append(h); contracts.append(c)
    exclusions = {"trainer_config": {"output_dir", "logging_dir", "run_name"},
                  "stage_hyperparameters": {"training_run_contract_path", "training_run_contract_identity"},
                  "args": {"output_dir"}}
    if methods[1] == "foarl_f638_dev":
        exclusions["trainer_config"] |= {"beta", "scale_rewards", "reward_weights"}
        exclusions["stage_hyperparameters"] |= {"runtime_profile", "grpo_beta", "resolved_trust_region_kwargs", "use_primal_dual",
                                               "kl_reference_mode", "kl_reference_model_path", "kl_reference_receipt_path", "kl_reference_runtime_receipt_path"}
        exclusions["args"] |= {"use_primal_dual", "grpo_beta", "verify_gt_rewards", "wandb_project",
                               "sft_checkpoint_path", "foarl_advantage_epsilon", "foarl_scale_rewards", "foarl_num_iterations",
                               "foarl_token_aggregation", "foarl_sync_ref_model", "foarl_dev_source_manifest",
                               "foarl_dev_source_manifest_sha256", "foarl_task_identity_binding", "foarl_loader_environment",
                               "foarl_prompt_template_validation", "foarl_dev_runtime_receipt"}
    def compare(a, b, label, excluded=()):
        diff = sorted(key for key in set(a) | set(b) if key not in excluded and
                      (key not in a or key not in b or digest(a[key]) != digest(b[key])))
        require(not diff, "paired actual " + label + " mismatch: " + ",".join(diff))
    for section, excluded in exclusions.items():
        compare(hp[0][section], hp[1][section], section, excluded)
    for section in ("dataset", "context", "distributed"):
        require(hp[0][section] == hp[1][section], "paired " + section + " mismatch")
    for section in ("runtime", "compute", "initialization", "source"):
        require(contracts[0][section] == contracts[1][section], "paired " + section + " mismatch")
    return {"status": "passed_common_actual_configuration_pairing", "task": task, "methods": list(methods),
            "compared_all_trainer_fields": True, "excluded_fields": {key: sorted(value) for key, value in exclusions.items()},
            "sources": {method: {key: audit[key] for key in ("contract", "hyperparams")}
                        for method, audit in zip(methods, (baseline, modified))}}


def allocation_records(job_to_key, *, gpus, runner=run_command):
    fields = ("JobIDRaw", "State", "ExitCode", "Start", "End", "ElapsedRaw", "TimelimitRaw", "AllocTRES", "QOS", "Account", "Partition")
    command = ["env", "TZ=UTC", "SLURM_TIME_FORMAT=standard", "sacct", "-X", "-P", "-n", "-j", ",".join(sorted(job_to_key, key=int)), "--format=" + ",".join(fields)]
    raw = runner(command)
    records, seen = [], set()
    for parts in csv.reader(raw.splitlines(), delimiter="|"):
        if not parts:
            continue
        require(len(parts) == len(fields), "malformed terminal allocation")
        row = dict(zip(fields, parts)); job = row["JobIDRaw"]
        require(job in job_to_key and job not in seen, "unknown/duplicate allocated job")
        seen.add(job)
        require(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0" and row["TimelimitRaw"] == "120", "wrong final state or dev limit")
        require(row["QOS"] == QOS and row["Account"] == "vas@a100" and row["Partition"] == "gpu_p5", "wrong actual accelerator queue")
        require(re.search(r"(?:^|,)gres/gpu=" + str(gpus) + r"(?:,|$)", row["AllocTRES"]), "wrong actual GPU allocation")
        seconds = int(row["ElapsedRaw"])
        require(0 < seconds <= 7200 and int((dt.datetime.fromisoformat(row["End"]) - dt.datetime.fromisoformat(row["Start"])).total_seconds()) == seconds, "invalid allocation duration")
        records.append({**row, "key": job_to_key[job], "allocated_gpus": gpus,
                        "allocated_gpu_seconds": gpus * seconds, "allocated_gpu_hours": gpus * seconds / 3600})
    require(seen == set(job_to_key), "missing final allocation row")
    return {"jobs": records, "sacct_command": command, "sacct_utc": raw,
            "allocated_gpu_hours": sum(row["allocated_gpu_seconds"] for row in records) / 3600}


def finalize_training(output, runner=run_command):
    protocol, source = protocol_for(output)
    audits = []
    for item in protocol["planned_train_commands"]:
        path = output / "ledgers/terminal" / (item["key"].replace("/", "_") + ".json")
        saved = signed(path)
        actual = terminal_validation(item, source, protocol, job_id=saved["job_id"])
        require(all(saved.get(key) == value for key, value in actual.items()), "saved terminal receipt no longer matches actual training")
        audits.append(saved)
    require(len({audit["job_id"] for audit in audits}) == 6, "six distinct training jobs required")
    checks = []
    for task in TASKS:
        baseline = next(a for a in audits if a["key"] == task + "/grpo_z")
        for method in METHODS[1:]:
            candidate = next(a for a in audits if a["key"] == task + "/" + method)
            checks.append(compare_actual_configuration(task, baseline, candidate, protocol, source))
            for name in ("added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
                require(baseline["checkpoint"][name]["sha256"] == candidate["checkpoint"][name]["sha256"], "paired tokenizer bytes differ")
    payload = {"status": "all_six_training_jobs_verified", "protocol_sha256": protocol["receipt_sha256"],
               "source_manifest": protocol["source_manifest"], "actual_configuration_comparisons": checks,
               "terminal_audits": [identity(output / "ledgers/terminal" / (a["key"].replace("/", "_") + ".json")) for a in audits],
               **allocation_records({a["job_id"]: a["key"] for a in audits}, gpus=4, runner=runner),
               "cost_interpretation": "Same raw rollout counts and token caps; teacher/search/reference computation is additional."}
    path = output / "ledgers/training_accounting.json"
    if path.exists():
        saved = signed(path)
        require({k: v for k, v in saved.items() if k != "receipt_sha256"} == payload, "training accounting changed")
        return saved
    return write_new(path, payload)


def submitted_job(output, protocol, key, *, role):
    events = read_events(output, protocol, role=role)[key]
    submissions = [event for event in events if event["kind"] == "submitted"]
    require(len(submissions) == 1, "exactly one accepted submission required: " + role + "/" + key)
    return submissions[0]["job_id"]


def training_endpoint(output, key, protocol, source, *, runner=run_command):
    job = submitted_job(output, protocol, key, role="train")
    rows = scheduler_rows(runner(["sacct", "-X", "-P", "-n", "-j", job,
                                  "--format=JobIDRaw,State,ExitCode,StdOut%600,ElapsedRaw"]))
    require(job in rows and rows[job]["state"] == "COMPLETED" and rows[job]["exit_code"] == "0:0", "training did not complete successfully")
    item = next(item for item in protocol["planned_train_commands"] if item["key"] == key)
    require(rows[job]["stdout"] in (item["stdout"], item["stdout"].replace("%j", job)), "training scheduler output mismatch")
    terminal = signed(output / "ledgers/terminal" / (key.replace("/", "_") + ".json"))
    require(terminal["job_id"] == job and terminal["protocol_sha256"] == protocol["receipt_sha256"], "training terminal/submit binding mismatch")
    actual = terminal_validation(item, source, protocol, job_id=job)
    require(all(terminal.get(field) == value for field, value in actual.items()), "training changed since its terminal audit")
    return terminal


def evaluation_launch(output, key, running_wrapper, *, runner=run_command):
    output = Path(output).resolve(strict=True)
    protocol, source = protocol_for(output)
    plan = next(item for item in protocol["planned_eval_commands"] if item["key"] == key)
    job = submitted_job(output, protocol, key, role="eval")
    require(os.environ.get("SLURM_JOB_ID") == job, "actual evaluation job is not the declared submission")
    for name, value in (("INCUMBENT_CAMPAIGN", output), ("INCUMBENT_SOURCE", source),
                        ("INCUMBENT_RUNTIME", protocol["runtime"]), ("INCUMBENT_OUTPUT", plan["output"])):
        require(Path(os.environ[name]).resolve() == Path(value).resolve(), "evaluation worker path drift: " + name)
    require(os.environ["INCUMBENT_SOURCE_SHA256"] == protocol["source_manifest"]["sha256"]
            and os.environ["INCUMBENT_TASK"] == plan["task"] and os.environ["INCUMBENT_METHOD"] == plan["method"], "evaluation worker source/task/method drift")
    actual_wrapper = identity(running_wrapper)
    expected_wrapper = protocol["evaluation_management"]["wrapper"]
    require(all(actual_wrapper[field] == expected_wrapper[field] for field in ("sha256", "size_bytes")), "executed evaluation spool bytes differ")
    terminal = training_endpoint(output, key, protocol, source, runner=runner)
    fresh = module("incumbent_launch_benchmark", source / "scripts/clasp_grpo_fresh_dev_artifacts.py")
    benchmark = fresh.validate_benchmark(Path(protocol["evaluation"]["data_root"]))
    require(benchmark == protocol["benchmark"], "evaluation benchmark changed")
    destination = Path(plan["output"])
    require(not destination.exists(), "create-once evaluation output exists")
    payload = {"schema_version": 1, "status": "validated_fresh_development_launch", "task": plan["task"],
               "method": plan["method"], "stage": "grpo", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "generation": protocol["generation"], "evaluation_slice": {"split": "eval", "start": 0, "count": 100},
               "repair": False, "representation": "jiang_f638_clasp", "benchmark": benchmark,
               "source_manifest": protocol["source_manifest"],
               "checkpoint": {name: terminal["checkpoint"][name] for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")},
               "training_contract": terminal["contract"], "management": protocol["evaluation_management"],
               "runtime_python": identity(Path(protocol["runtime"]) / "bin/python"),
               "reference_semantics": {"cvrp": "gap_to_verified_heuristic_reference", "mis": "gap_to_certified_optimum"},
               "slurm_job_id": job, "training_job_id": terminal["job_id"],
               "campaign_protocol_sha256": protocol["receipt_sha256"], "spool_execution_path": str(Path(running_wrapper).resolve())}
    destination.mkdir()
    return write_new(destination / "launch_receipt.json", payload, "launch_sha256")


def evaluation_validation(item, source, protocol, job_id=None):
    """Recompute raw-file/header/binding audits; never trust only a cached index."""
    campaign = Path(protocol["output_root"])
    output = Path(item["output"])
    expected_job = submitted_job(campaign, protocol, item["key"], role="eval")
    require(str(job_id) == expected_job, "evaluation scheduler and submitted job differ")
    launch = signed(output / "launch_receipt.json", "launch_sha256")
    require(launch["campaign_protocol_sha256"] == protocol["receipt_sha256"]
            and launch["slurm_job_id"] == expected_job, "evaluation launch declaration/job mismatch")
    terminal = training_endpoint(campaign, item["key"], protocol, source)
    require(launch["training_job_id"] == terminal["job_id"], "evaluation training-job linkage mismatch")
    fresh = module("incumbent_index_readonly", source / "scripts/clasp_grpo_fresh_dev_artifacts.py")
    require(fresh.GENERATION == protocol["generation"], "frozen generation helper differs from declaration")
    # The original helper's sole mutation is its final index write. Capture its
    # validated payload in memory, finish cross-bindings, and only then publish.
    fresh.write_new = lambda path, value, field: {**value, field: digest(value)}
    index = fresh.index(output)
    benchmark = fresh.validate_benchmark(Path(protocol["evaluation"]["data_root"]))
    require(benchmark == protocol["benchmark"], "evaluation benchmark identities changed")
    strict = module("incumbent_raw_strict", source / "scripts/compare_development.py")
    artifact = strict._load_artifact(index["details"]["path"], index["run_config"]["path"], label=item["key"])
    config = signed(index["run_config"]["path"], "run_config_sha256")
    binding = audit_evaluation_arm(item, launch, config, index, terminal, job_id=expected_job,
                                   training=protocol, evaluation={"generation": protocol["generation"], "management": protocol["evaluation_management"]},
                                   benchmark=benchmark, source=source, records=artifact.records)
    path = output / "result_index.json"
    if path.exists():
        require(signed(path, "index_sha256") == index, "existing result index differs from revalidated raw bytes")
    return {"status": "terminal_evaluation_verified", "key": item["key"], "job_id": expected_job,
            "index_payload": index, "cross_bindings": binding,
            "raw_details": index["details"], "run_config": index["run_config"]}


def evaluation_index(output, key):
    output = Path(output).resolve(strict=True)
    protocol, source = protocol_for(output)
    item = next(item for item in protocol["planned_eval_commands"] if item["key"] == key)
    job = submitted_job(output, protocol, key, role="eval")
    require(os.environ.get("SLURM_JOB_ID") == job, "index worker job differs from launch")
    verified = evaluation_validation(item, source, protocol, job_id=job)
    index = verified["index_payload"]
    return write_new(Path(item["output"]) / "result_index.json", {k: v for k, v in index.items() if k != "index_sha256"}, "index_sha256")


def summarize_campaign(output):
    output = Path(output).resolve(strict=True)
    protocol, source = protocol_for(output)
    # Mandatory definitive audit occurs before the comparison directory exists.
    accounting = finalize_training(output)
    state = reconcile(output, role="eval")
    require(state["all_evaluation_verified"], "all six evaluations must have completed successfully")
    indices, audits, eval_jobs = [], [], {}
    for item in protocol["planned_eval_commands"]:
        job = submitted_job(output, protocol, item["key"], role="eval")
        audits.append(evaluation_validation(item, source, protocol, job_id=job))
        indices.append(Path(item["output"]) / "result_index.json")
        eval_jobs[job] = item["key"]
    costs = allocation_records(eval_jobs, gpus=1)
    summary = module("incumbent_metric_summary", source / "scripts/summarize_clasp_grpo_fresh_dev.py")
    summary.METHODS = METHODS
    loaded, identities, _ = summary.collect(indices)
    rows, comparisons = [], {}
    for task in TASKS:
        for method in METHODS:
            rows.append(summary.summarize(list(loaded[task, method].records.values()), task, method))
        for left, right in (("grpo_z", "grpo_verified"), ("grpo_z", "foarl_f638_dev"), ("foarl_f638_dev", "grpo_verified")):
            a, b = loaded[task, left], loaded[task, right]
            comparisons[task + "/" + left + "_vs_" + right] = {
                "strict_comparison": summary.strict.build_receipt(a, b, left_label=left, right_label=right),
                "unconditional_solved_quality": summary.paired_quality(a, b)}
    payload = {"status": "all_six_arms_raw_evaluation_and_provenance_verified", "source_manifest": protocol["source_manifest"],
               "protocol_sha256": protocol["receipt_sha256"], "summary": rows, "comparisons": comparisons,
               "result_indices": identities, "cross_bindings": [audit["cross_bindings"] for audit in audits],
               "training_accounting": accounting, "evaluation_accounting": costs,
               "definitions": {"headline_gap": "Each arm's own feasible instances; percentage points in table.",
                               "paired_gap": "Right minus left on jointly feasible instances; fractions in strict JSON.",
                               "candidate_denominator": "800 raw candidates per arm", "feasible_at_8_denominator": "100 instances per arm",
                               "unconditional_quality": "Mean(feasible * exp(-3*max(0,gap))); failures contribute zero.",
                               "signed_gap": "Directional unclipped reference gap; preserve negative better-than-reference values.",
                               "uncertainty": "Paired instance bootstrap; no training-seed replication or multiplicity adjustment."},
               "limitations": ["One training seed and reused development100/task; not confirmation or SOTA evidence.",
                               "FOARL is a corrected common-budget algorithm port, not an official Jiang reproduction.",
                               "CVRP references are verified heuristics; MIS references are certified optima.",
                               "Equal rollout counts and token caps do not equate training compute."]}
    directory = output / "comparison"
    require(not directory.exists(), "comparison output already exists")
    directory.mkdir()
    result = write_new(directory / "comparison.json", payload)
    columns = ("task", "method", "feasible_at_8_percent", "candidate_feasibility_percent", "mean_feasible_gap_percent", "mean_signed_reference_gap_percent", "unconditional_solved_quality")
    with (directory / "table.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    lines = ["| Task | Method | Feasible@8 | Candidate feasibility | Gap on own successes |", "|---|---|---:|---:|---:|"]
    for row in rows:
        gap = "undefined" if row["mean_feasible_gap_percent"] is None else f'{row["mean_feasible_gap_percent"]:.4f}%'
        lines.append(f'| {row["task"]} | {row["method"]} | {row["feasible_at_8_count"]}/100 | {row["candidate_feasibility_percent"]:.3f}% | {gap} |')
    (directory / "table.md").write_text("\n".join(lines) + "\n\nOwn-success gaps have different denominators when success sets differ; use paired results alongside feasibility. CVRP gaps use heuristic references.\n")
    return result


# Cross-binding logic retained from the independently reviewed September 6 auditor.
def audit_evaluation_arm(plan, launch, config, index, terminal, *, job_id, training, evaluation,
              benchmark, source, records):
    """Cross-bind already byte-validated receipts; return a compact arm receipt."""
    key, task, method = plan["key"], plan["task"], plan["method"]
    prefix = key + ": "
    checkpoint = Path(plan["checkpoint"]).resolve(strict=True)
    train = Path(plan["training_root"]).resolve(strict=True)
    output = Path(plan["output"]).resolve(strict=True)
    require(checkpoint == train / "checkpoints/checkpoint-100", prefix + "planned checkpoint is not step 100")
    require(launch["slurm_job_id"] == job_id, prefix + "launch job differs from submission ledger")
    require((launch["task"], launch["method"], launch["stage"]) == (task, method, "grpo")
            and (index["task"], index["method"]) == (task, method), prefix + "method/task mismatch")
    require(launch["source_manifest"] == training["source_manifest"], prefix + "launch source mismatch")
    require(launch["training_contract"] == terminal["contract"], prefix + "launch training contract mismatch")
    require(launch["benchmark"] == benchmark, prefix + "launch benchmark mismatch")
    require(launch["management"] == evaluation["management"], prefix + "launch management mismatch")
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        require(launch["checkpoint"][name] == terminal["checkpoint"][name]
                and Path(terminal["checkpoint"][name]["path"]).resolve() == checkpoint / name,
                prefix + "launch checkpoint hash/path mismatch: " + name)
    expected_slice = {"split": "eval", "start": 0, "count": 100}
    require(launch["generation"] == evaluation["generation"] and launch["repair"] is False
            and index["repair"] is False and launch["evaluation_slice"] == index["evaluation_slice"] == expected_slice,
            prefix + "launch generation/repair/slice mismatch")
    reference = "gap_to_verified_heuristic_reference" if task == "cvrp" else "gap_to_certified_optimum"
    require(index["reference_semantics"] == launch["reference_semantics"][task] == reference,
            prefix + "reference semantics mismatch")
    require(config["stage"] == "grpo" and config["tasks"] == [task] and config["split"] == "eval"
            and config["run_status"] == "complete" and config["evaluation_purpose"] == "development",
            prefix + "run task/stage/status mismatch")
    require(Path(config["resolved_model_path"]).resolve(strict=True) == checkpoint,
            prefix + "raw run model differs from planned checkpoint")
    require(Path(config["resolved_run_dir"]).resolve(strict=True) == train,
            prefix + "raw run training directory mismatch")
    require(Path(config["data_root"]).resolve(strict=True) == Path(training["evaluation"]["data_root"]).resolve(strict=True),
            prefix + "raw run data root mismatch")
    require(Path(config["results_dir"]).resolve(strict=True).is_relative_to(output),
            prefix + "raw run output belongs to a different evaluation")
    for name, field in (("run_config", "run_config"), ("details", "details")):
        require(Path(config["files"][name]).resolve(strict=True) == Path(index[field]["path"]).resolve(strict=True)
                and Path(index[field]["path"]).resolve(strict=True).is_relative_to(output),
                prefix + "raw artifact path linkage mismatch: " + name)
    provenance = config["experiment_provenance"]
    generation = config["generation"]
    require(all(generation.get(k) == v for k, v in evaluation["generation"].items())
            and generation["seed_policy"] == "per_example_sha256_v1" and generation["best_of_n"] == 8,
            prefix + "raw generation differs from declaration")
    require(generation == {**provenance["generation"], "best_of_n": 8}, prefix + "generation/provenance mismatch")
    require(all(generation.get(k) == [] for k in ("temperature_schedule", "top_p_schedule", "op_size_band_top_p_schedule"))
            and generation.get("op_size_band_top_p_min_size") == generation.get("op_size_band_top_p_max_size") == 0,
            prefix + "undeclared sampling schedule")
    prompting, limits = config["prompting"], config["limits"]
    require(prompting["prompt_style_resolved"] == "jiang_f638" and prompting["no_reason"] is True
            and prompting["clasp_mis_jiang_response_output"] == "0" and prompting["clasp_mis_relabel_order"] == ""
            and Path(prompting["tokenizer_path"]).resolve(strict=True) == checkpoint,
            prefix + "prompt/tokenizer configuration mismatch")
    require(limits["eval_start_index"] == 0 and limits["max_eval_samples_per_task"] == 100
            and limits["prefer_subset_files"] is False, prefix + "raw evaluation slice mismatch")
    artifact = provenance["model"]["checkpoint_artifact"]
    require(artifact["kind"] == "local_directory" and artifact["basename"] == "checkpoint-100"
            and artifact["manifest_sha256"] == digest(artifact["files"]), prefix + "model artifact manifest mismatch")
    seen = set()
    for file in artifact["files"]:
        name = file["path"]
        require(name not in seen and name in terminal["checkpoint"], prefix + "unknown/duplicate checkpoint artifact")
        seen.add(name)
        expected = terminal["checkpoint"][name]
        require(file["sha256"] == expected["sha256"] and file["size"] == expected["size_bytes"],
                prefix + "raw run checkpoint bytes differ from training: " + name)
    require({"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"} <= seen,
            prefix + "raw model provenance lacks checkpoint identity")
    data_files = provenance["data"]["files"]
    dataset = benchmark["files"][task]["dataset"]
    require(data_files == [{"path": task + "/eval/test.json", "sha256": dataset["sha256"],
                            "size": dataset["size_bytes"], "split": "eval", "task": task}],
            prefix + "raw run dataset hash differs from benchmark")
    code = config["code_identity"]
    require(code == provenance["code"] and code["manifest_sha256"] == digest(code["files"]),
            prefix + "raw code provenance mismatch")
    seen_code = set()
    for file in code["files"]:
        path = (source / "src/clasp" / file["path"]).resolve(strict=True)
        require(path.is_relative_to(source / "src/clasp") and path not in seen_code, prefix + "code path escaped/duplicated")
        seen_code.add(path)
        actual = identity(path)
        require(file["sha256"] == actual["sha256"] and file["size"] == actual["size_bytes"],
                prefix + "raw evaluator code differs from frozen source")
    require(source / "src/clasp/entrypoints/evaluate.py" in seen_code, prefix + "missing evaluator code identity")
    ordered = sorted(records.values(), key=lambda row: row.example_index)
    require(len(ordered) == 100, prefix + "expected 100 raw records")
    for row, expected in zip(ordered, benchmark["ordered_instances"][task], strict=True):
        require(all(row.raw.get(k) == v for k, v in expected.items())
                and row.raw["task"] == task and row.raw["split"] == "eval"
                and row.raw["num_return_sequences"] == len(row.raw["candidates"]) == 8,
                prefix + "raw record differs from benchmark/K8")
    return {"key": key, "job_id": job_id, "checkpoint": str(checkpoint), "instances": 100, "candidates": 800,
            "launch_sha256": launch["launch_sha256"], "index_sha256": index["index_sha256"],
            "run_config_sha256": config["run_config_sha256"], "details": index["details"],
            "model_artifact_manifest_sha256": artifact["manifest_sha256"], "status": "all_cross_bindings_verified"}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "status", "run", "watch", "eval-launch", "eval-index", "summarize"))
    parser.add_argument("--output", required=True, type=Path, help="Create-once campaign root")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--diagnostic-campaign", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--role", choices=("train", "eval", "both"), default="train")
    parser.add_argument("--max-new-jobs", type=int, default=5)
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--key")
    parser.add_argument("--running-wrapper", type=Path)
    args = parser.parse_args()
    require(0 <= args.max_new_jobs <= 5, "at most five new jobs per tick")
    if args.action == "init":
        require(args.source and args.source_sha and args.diagnostic_campaign and not args.submit,
                "init requires source/hash/diagnostic campaign and never submits")
        result = initialize(args.output, args.source, args.source_sha, args.diagnostic_campaign)
    else:
        output = args.output.resolve(strict=True)
        if args.action == "status":
            require(not args.submit, "status is read-only")
            result = {kind: reconcile(output, role=kind) for kind in (("train", "eval") if args.role == "both" else (args.role,))}
        elif args.action == "run":
            result = locked_run(output, args.submit, True, args.max_new_jobs, args.role)
        elif args.action == "eval-launch":
            require(not args.submit and args.key and args.running_wrapper, "launch binds an already submitted evaluation")
            result = evaluation_launch(output, args.key, args.running_wrapper)
        elif args.action == "eval-index":
            require(not args.submit and args.key, "index requires one completed raw arm")
            result = evaluation_index(output, args.key)
        elif args.action == "summarize":
            require(not args.submit, "summarize never submits")
            result = summarize_campaign(output)
        else:
            require(0 < args.hours <= 12 and args.poll_seconds == 60, "watch is bounded to12h with60s polls")
            deadline = time.monotonic() + args.hours * 3600
            previous = None
            while time.monotonic() < deadline:
                result = locked_run(output, args.submit, True, args.max_new_jobs, args.role)
                comparable = digest(result)
                if comparable != previous:
                    print(json.dumps(result, sort_keys=True), flush=True); previous = comparable
                complete = ((args.role == "train" and result["train"]["all_training_verified"])
                            or (args.role == "eval" and result["eval"]["all_evaluation_verified"])
                            or (args.role == "both" and result["train"]["all_training_verified"] and result["eval"]["all_evaluation_verified"]))
                if complete:
                    return
                failure_states = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED", "BOOT_FAIL"}
                failures = [row for kind in ("train", "eval") for row in result.get(kind, {}).get("observations", []) if row["state"] in failure_states]
                require(not failures, "failed arm requires review; no automatic retry: " + json.dumps(failures))
                time.sleep(args.poll_seconds)
            result = {"status": "bounded_watch_deadline_reached", "campaign": str(output)}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
