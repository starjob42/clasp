"""Read-only admission of complete, same-contract FOARL continuation inputs.

This pre-torchrun check closes Trainer's silent missing-optimizer/RNG fallback.
It does not deserialize pickle files or claim that moments have been loaded:
the native resume diagnostic and Trainer's live checks still have that role.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from .checkpoint_fidelity import verify_checkpoint_fidelity
from .exact_peft_kl_post_step import (
    POST_STEP_RECEIPT_FILENAME,
    STEP1_RECEIPT_FILENAME,
    inspect_post_step_receipt,
    training_run_contract_identity,
)

PROTOCOL = "foarl-teacher-complete-resume-inputs-v1"
_SHA = re.compile(r"[0-9a-f]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    def nonfinite(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    value = json.loads(path.read_text(), object_pairs_hook=_object_pairs,
                       parse_constant=nonfinite)
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _absolute(value, label):
    _require(isinstance(value, (str, os.PathLike)), f"{label} must be a path")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    return path


def _identity(path):
    path = _absolute(path, "Artifact")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_size > 0,
             f"Missing/nonregular/empty resume input: {path}")
    digest = hashlib.sha256()
    # O_NOFOLLOW also closes a symlink replacement between lstat and open.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        _require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
                 f"Resume input changed while opening: {path}")
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    _require((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             f"Resume input changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _match_identity(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256", "size_bytes"},
             f"{label} identity schema mismatch")
    _require(isinstance(record["sha256"], str) and _SHA.fullmatch(record["sha256"]),
             f"{label} has invalid SHA256")
    _require(type(record["size_bytes"]) is int and record["size_bytes"] > 0,
             f"{label} has invalid size")
    observed = _identity(record["path"])
    _require(observed == record, f"{label} bytes differ from original contract")
    return observed


def validate_foarl_teacher_resume(
    checkpoint_dir, *, expected_step, expected_contract, run_contract_path,
    world_size=4, final_horizon=400,
):
    """Require a complete checkpoint and the exact original immutable contract.

    ``run_contract_path`` lives directly in the output run directory; its child
    ``checkpoints/checkpoint-N`` must be the requested checkpoint. All returned identities
    describe live bytes. No inputs are modified and no pickle is executed.
    A successful result is input admission, not a claim of exact resumed RNG or
    optimizer behavior. Those must be checked after actual Trainer restoration.
    """
    _require(type(world_size) is int and world_size == 4, "Exactly four ranks required")
    _require(type(final_horizon) is int and final_horizon == 400, "Final horizon must remain 400")
    _require(type(expected_step) is int and 0 < expected_step <= final_horizon,
             "Checkpoint step must not exceed the fixed final horizon")
    _require(isinstance(expected_contract, Mapping), "Expected contract must be an object")
    contract_path = _absolute(run_contract_path, "Run contract")
    contract_identity = _identity(contract_path)
    _require(not contract_path.lstat().st_mode & 0o222, "Original run contract must be read-only")
    contract = _load_json(contract_path)
    _require(_canonical(contract) == _canonical(dict(expected_contract)),
             "Original run contract differs from expected contract")
    required = {"schema_version", "stage", "status", "method", "source", "initialization",
                "fixed_horizon", "teacher_weight", "seed", "data_root", "schedule", "diagnostic"}
    _require(required <= contract.keys() and not (contract.keys() - required - {"data_files"}),
             "Run contract fields differ from the declared schema")
    _require(type(contract["schema_version"]) is int and contract["schema_version"] == 1,
             "Unsupported run contract schema")
    _require(contract["stage"] == "grpo"
             and contract["status"] == "development_only_not_publication_eligible",
             "Resume contract must retain development-only scope")
    _require(contract["method"] in {"foarl_control", "foarl_teacher"}, "Unknown FOARL method")
    weight = contract["teacher_weight"]
    _require(type(weight) in (int, float) and math.isfinite(weight)
             and weight == (0.05 if contract["method"] == "foarl_teacher" else 0.0),
             "Teacher weight does not match the original method")
    _require(type(contract["diagnostic"]) is bool, "Diagnostic flag must be boolean")
    _require(type(contract["seed"]) is int and contract["seed"] == 43, "Policy/data seed must remain 43")
    _require(_canonical(contract["fixed_horizon"])
             == _canonical({"max_steps": final_horizon, "world_size": world_size}),
             "Fixed scheduler horizon/world size changed")
    save_steps = 5 if contract["diagnostic"] else 100
    schedule = {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                "steps_per_generation": 3, "save_steps": save_steps}
    _require(_canonical(contract["schedule"]) == _canonical(schedule), "Rollout/save schedule changed")
    _require(expected_step % save_steps == 0, "Resume step is not a declared checkpoint boundary")
    _absolute(contract["data_root"], "Dataset root")

    root = _absolute(checkpoint_dir, "Checkpoint")
    _require(stat.S_ISDIR(root.lstat().st_mode), "Checkpoint must be a non-symlink directory")
    _require(root.name == f"checkpoint-{expected_step}"
             and root.parent == contract_path.parent / "checkpoints",
             "Checkpoint is not the expected child of the original run directory")
    names = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "checkpoint_fidelity.json",
             "adapter_config.json", "adapter_model.safetensors", "exact_peft_kl_reference.json",
             STEP1_RECEIPT_FILENAME, POST_STEP_RECEIPT_FILENAME]
    names += [f"rng_state_{rank}.pth" for rank in range(world_size)]
    files = {name: _identity(root / name) for name in names}
    state = _load_json(root / "trainer_state.json")
    _require(type(state.get("global_step")) is int and state["global_step"] == expected_step,
             "Trainer global_step differs from the requested checkpoint")
    _require(type(state.get("max_steps")) is int and state["max_steps"] == final_horizon,
             "Trainer scheduler horizon differs from the original 400-step run")
    fidelity = verify_checkpoint_fidelity(root)
    post = inspect_post_step_receipt(
        root / POST_STEP_RECEIPT_FILENAME, claim_scope="development",
        expected_global_step=expected_step, expected_artifact_role="checkpoint",
        rehash_sources=False, rehash_policy_artifact=False,
        expected_policy_trainable_tensor_inventory_sha256=(
            fidelity["trainable_closure"]["tensor_inventory_sha256"]),
    )
    # Checkpoint policy evidence itself must name this original contract. Parent
    # directory placement alone cannot detect a copied checkpoint from another arm.
    _require(post["training_run_contract"]
             == training_run_contract_identity(contract_path, claim_scope="development"),
             "Checkpoint evidence belongs to a different original run contract")
    _require(post["pre_step_reference"]["receipt_file_sha256"]
             == files["exact_peft_kl_reference.json"]["sha256"],
             "Checkpoint pre-step KL lineage differs from its sibling receipt")
    _require(post["step1_reference"]["receipt_file_sha256"]
             == files[STEP1_RECEIPT_FILENAME]["sha256"],
             "Checkpoint step-one KL lineage differs from its sibling receipt")

    source = contract["source"]
    _require(isinstance(source, dict) and set(source) == {"root", "manifest_sha256"},
             "Source binding schema mismatch")
    source_root = _absolute(source["root"], "Source root")
    source_identity = _identity(source_root / "SOURCE_MANIFEST.sha256")
    _require(source_identity["sha256"] == source["manifest_sha256"], "Original source manifest changed")
    initialization = contract["initialization"]
    _require(isinstance(initialization, dict) and set(initialization) == {"checkpoint", "files"},
             "SFT initialization schema mismatch")
    sft_root = _absolute(initialization["checkpoint"], "Original SFT checkpoint")
    _require(sft_root != root, "The KL reference must remain original SFT, not the resumed policy")
    initial_files = initialization["files"]
    _require(isinstance(initial_files, dict) and set(initial_files)
             == {"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"},
             "Original SFT initialization must bind weights, config and fidelity")
    observed_initialization = {}
    for name, record in initial_files.items():
        _require(isinstance(record, dict) and Path(record.get("path", "")) == sft_root / name,
                 "SFT initialization file path differs from its declared checkpoint")
        observed_initialization[name] = _match_identity(record, f"Original SFT {name}")
    verify_checkpoint_fidelity(sft_root)
    data_files = contract.get("data_files", {})
    _require(isinstance(data_files, dict), "Dataset file identities must be an object")
    observed_data = {name: _match_identity(record, f"Dataset {name}")
                     for name, record in data_files.items()}
    # A long filesystem audit must not admit changing checkpoint/contract bytes.
    _require(_identity(contract_path) == contract_identity, "Run contract changed during admission")
    _require({name: _identity(root / name) for name in names} == files,
             "Checkpoint changed during resume admission")
    result = {
        "protocol": PROTOCOL, "status": "complete_same_contract_resume_inputs_verified",
        "checkpoint": str(root), "expected_step": expected_step, "max_steps": final_horizon,
        "world_size": world_size, "method": contract["method"], "run_contract": contract_identity,
        "source_manifest": source_identity, "initialization_files": observed_initialization,
        "data_files": observed_data, "files": files,
        "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
        "post_step_receipt_sha256": post["receipt_sha256"],
        "further_continuation_possible": expected_step < final_horizon,
        "all_rank_rng_files_present": True,
        "live_optimizer_scheduler_rng_restoration_verified": False,
        "limitation": "Input bytes/completeness are verified; native loaded-state parity remains a runtime check.",
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result
