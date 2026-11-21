#!/usr/bin/env python3
"""Standard-library provenance checks for the fixed seed-43 fair development run."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys


SFT_HASHES = {
    "cvrp": "f8f634c3a59f53bf411bb0dfc85af98dbc16d936a53263cdfcd6e74ce88b25b8",
    "mis": "cb5e979325ba7691f75998326ce79e47038d6462596dbc815f2b54d2b550da15",
}
METHODS = ("grpo_z", "clasp_grpo_anchor_v3", "clasp_grpo_anchor_v4")


def require(condition, message):
    if not condition:
        raise SystemExit("[FairDevError] " + message)


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def seal(payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return dict(payload, receipt_sha256=hashlib.sha256(raw).hexdigest())


def validate_source(source):
    source = Path(source).resolve(strict=True)
    manifest = source / "SOURCE_MANIFEST.sha256"
    seen = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = (source / relative).resolve(strict=True)
        require(path.is_relative_to(source), "manifest path escapes source snapshot")
        require(identity(path)["sha256"] == digest, "source hash mismatch: " + relative)
        seen.add(relative)
    required = {
        "scripts/train_clasp_canonical_checkpoint_dev_a100.slurm",
        "scripts/submit_clasp_grpo_fair_dev_a100.sh",
        "scripts/evaluate_clasp_grpo_fair_dev_a100.slurm",
        "scripts/clasp_grpo_fair_dev_artifacts.py",
        "scripts/resume_clasp_grpo_fair_dev.py",
        "scripts/evaluate_a100.slurm",
        "scripts/compare_development.py",
    }
    require(required <= seen, "required scripts are absent from source manifest")
    launcher = (source / "scripts/train_clasp_canonical_checkpoint_dev_a100.slurm").read_text()
    require("clasp_grpo_anchor_v4)" in launcher, "source launcher lacks the v4 arm")
    return identity(manifest)


def preflight(source, runtime, data, sft):
    manifest = validate_source(source)
    initializations = {}
    for task, expected in SFT_HASHES.items():
        checkpoint = Path(sft) / "runs" / task / "seed42/checkpoints/checkpoint-31250"
        files = {name: identity(checkpoint / name) for name in (
            "adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}
        require(files["adapter_model.safetensors"]["sha256"] == expected, "unexpected SFT adapter for " + task)
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        require(Path(config["base_model_name_or_path"]).is_dir(), "SFT base model is missing")
        for split in ("train", "eval"):
            require((Path(data) / task / split).is_dir(), "missing task data: " + task + "/" + split)
        initializations[task] = files
    return seal({
        "schema_version": 1, "status": "frozen_development_protocol",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_manifest": manifest, "runtime": str(Path(runtime).resolve()),
        "data_root": str(Path(data).resolve()), "initializations": initializations,
        "methods": list(METHODS), "tasks": list(SFT_HASHES),
        "training": {"seed": 43, "steps": 100, "resume": False, "group_std_correction": 1, "rollout_batch_b": 9,
                     "global_generation_batch": 36, "learning_rate": 1e-6,
                     "gpus_per_job": 4, "qos": "qos_gpu_a100-dev", "walltime": "02:00:00"},
        "evaluation": {"split": "eval", "start": 100, "count": 100, "seed": 271828,
                       "candidate_budget": 8, "dtype": "float16", "temperature": 0.7,
                       "top_p": 0.9, "top_k": 50, "max_new_tokens": 3000,
                       "max_prompt_length": 29768, "gpus_per_job": 1,
                       "qos": "qos_gpu_a100-dev", "walltime": "02:00:00"},
    })


def validate_training(train, task, method, source, *, rollout_batch_b=9):
    require(isinstance(rollout_batch_b, int) and not isinstance(rollout_batch_b, bool)
            and rollout_batch_b in (9, 18), "unsupported training rollout budget")
    require(rollout_batch_b == 9 or method in ("grpo_z", "clasp_grpo_anchor_v3"),
            "B=18 follow-up requires its matched GRPO or v3 arm")
    manifest = validate_source(source)
    train = Path(train)
    contract = json.loads((train / "local_development_run_contract.json").read_text())
    expected = {"method": method, "task": task, "seed": 43}
    require(all(contract.get(key) == value for key, value in expected.items()), "training identity mismatch")
    require(contract["fixed_horizon"] == {"max_steps": 100, "world_size": 4}, "training budget mismatch")
    continuation = contract.get("continuation") or {}
    require(not continuation.get("resume_checkpoint") and continuation.get("resume_global_step", 0) == 0
            and not continuation.get("strict_optimizer_resume"),
            "expected fresh SFT training, found continuation")
    require(contract["initialization"]["adapter_model_sha256"] == SFT_HASHES[task], "SFT initialization mismatch")
    require(Path(contract["source"]["root"]).resolve() == Path(source).resolve(), "training/evaluation source mismatch")
    require(contract["source"]["manifest_sha256"] == manifest["sha256"], "training source manifest hash mismatch")
    production = contract["production_preserved"]
    require(all(production.get(key) == value for key, value in {
        "rollout_batch_b": rollout_batch_b, "global_generation_batch_size": 36, "learning_rate": 1e-6,
        "max_completion_length": 3000, "max_prompt_length": 29768,
        "per_device_train_batch_size": 3, "gradient_accumulation_steps": 3,
        "beta": 0.0, "deployment_set_k": 8, "prompt_style": "jiang_f638", "no_reason": True,
    }.items()), "shared training settings mismatch")
    policy = contract["policy_estimator"]
    if rollout_batch_b == 18:
        require(policy["canonical_max_subsets"] == 43758 and policy["canonical_exact_scale"] == 1.0,
                "B=18 requires the complete 18-choose-8 subset system")
    require(policy["group_std_correction"] == 1, "expected sample-standardized GRPO credit")
    require(policy["importance_sampling_level"] == "token" and policy["policy_length_alpha"] == 0.0,
            "policy importance sampling or length normalization mismatch")
    require(policy["canonical_set"] == (method != "grpo_z") and policy["use_clasp"] == (method != "grpo_z"),
            "method selection mismatch")
    require(policy["canonical_grpo_anchor_scale"] == (0.0 if method == "grpo_z" else 1.0),
            "GRPO anchor coefficient mismatch")
    require(policy.get("canonical_quality_anchor_ratio", 0.0) == (0.0 if method == "grpo_z" else 0.5),
            "tail coefficient mismatch")
    require(policy["canonical_dense_feasibility_scale"] == (0.5 if method.endswith("v4") else 0.0),
            "dense feasibility coefficient mismatch")
    checkpoint = train / "checkpoints/checkpoint-100"
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        require((checkpoint / name).is_file(), "incomplete terminal checkpoint: " + name)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    require(state["global_step"] == 100, "terminal checkpoint step mismatch")
    return {"status": "training_validated", "task": task, "method": method}


def result_index(output, task, method, train):
    output = Path(output)
    details = list(output.glob("eval/**/run_details__*.jsonl"))
    configs = list(output.glob("eval/**/run_config__*.json"))
    require(len(details) == len(configs) == 1, "expected exactly one evaluation result")
    records = [json.loads(line) for line in details[0].read_text().splitlines() if line.strip()]
    require(len(records) == 100, "evaluation did not complete 100 instances")
    require(all(record["task"] == task and record["split"] == "eval"
                and record["num_return_sequences"] == len(record["candidates"]) == 8
                for record in records), "evaluation task, split, or candidate budget mismatch")
    require({record["example_index"] for record in records} == set(range(100, 200)),
            "evaluation slice mismatch")
    config = json.loads(configs[0].read_text())
    require(all(config["generation"].get(key) == value for key, value in {
        "seed": 271828, "dtype": "float16", "num_return_sequences": 8, "generation_batch_size": 8,
        "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0,
        "max_new_tokens": 3000, "max_prompt_length": 29768,
    }.items()), "evaluation decoding mismatch")
    payload = seal({
        "schema_version": 1, "status": "complete_development_only", "task": task, "method": method,
        "candidate_budget": 8, "generation_seed": 271828,
        "evaluation_slice": {"split": "eval", "start": 100, "count": 100},
        "details": identity(details[0]), "run_config": identity(configs[0]),
        "training_contract": identity(Path(train) / "local_development_run_contract.json"),
    })
    descriptor = os.open(output / "result_index.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    action = {"preflight": preflight, "validate-training": validate_training, "index": result_index}[command]
    print(json.dumps(action(*args), indent=2, sort_keys=True))
