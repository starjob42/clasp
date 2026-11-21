"""Complete beta-zero classic-GRPO checkpoints and immutable chunk lineage.

The existing beta-positive FOARL validator is deliberately unchanged. This
module admits optimizer/RNG input bytes; only an actual native continuation
comparison can establish that Trainer restores them correctly.
"""
from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from .checkpoint_fidelity import _qwen_publication_invariant, verify_checkpoint_fidelity
from .foarl_teacher_resume import (
    _absolute,
    _canonical,
    _identity,
    _load_json,
    _match_identity,
    _require,
)

PROTOCOL = "classic-grpo-beta-zero-complete-resume-inputs-v1"
LINEAGE_PROTOCOL = "classic-grpo-beta-zero-checkpoint-lineage-v1"
LINEAGE_NAME = "classic_grpo_checkpoint_lineage.json"
POLICY_NAME = "classic_grpo_policy_consistency.json"
KL_NAMES = ("exact_peft_kl_reference.json", "exact_peft_kl_reference_step1.json",
            "exact_peft_kl_reference_post_step.json")
STATE_NAMES = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "checkpoint_fidelity.json",
               "adapter_config.json", "adapter_model.safetensors",
               *(f"rng_state_{rank}.pth" for rank in range(4)))


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _signed_json(path):
    value = _load_json(path)
    _require(value.get("receipt_sha256") == _digest({k: v for k, v in value.items() if k != "receipt_sha256"}),
             "Invalid checkpoint lineage or consensus self hash")
    return value


def _write_once(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o440)


def _reject_kl(checkpoint, contract_path):
    for root in (checkpoint, checkpoint.parent, contract_path.parent):
        _require(all(not (root / name).exists() and not (root / name).is_symlink() for name in KL_NAMES),
                 "Beta-zero classic GRPO cannot contain KL reference sidecars")


def _base(checkpoint_dir, expected_step, expected_contract, run_contract_path):
    _require(type(expected_step) is int and 0 < expected_step <= 400, "Invalid checkpoint step")
    _require(isinstance(expected_contract, Mapping), "Expected original contract must be an object")
    contract_path = _absolute(run_contract_path, "Run contract")
    contract_file = _identity(contract_path)
    _require(not contract_path.lstat().st_mode & 0o222, "Original contract must be read-only")
    contract = _load_json(contract_path)
    _require(_canonical(contract) == _canonical(dict(expected_contract)), "Original contract differs")
    fields = {"schema_version", "stage", "status", "method", "source", "initialization", "fixed_horizon",
              "teacher_weight", "beta", "seed", "data_root", "data_files", "schedule", "diagnostic"}
    _require(set(contract) == fields and type(contract["schema_version"]) is int
             and contract["schema_version"] == 1, "Unsupported classic contract schema")
    _require(contract["stage"] == "grpo" and contract["method"] == "classic_grpo"
             and contract["status"] == "development_only_not_publication_eligible", "Wrong method or scope")
    for field in ("teacher_weight", "beta"):
        _require(type(contract[field]) in (int, float) and math.isfinite(contract[field]) and contract[field] == 0,
                 "Classic GRPO requires beta and teacher weight zero")
    _require(type(contract["seed"]) is int and contract["seed"] == 43, "Seed changed")
    _require(type(contract["diagnostic"]) is bool, "Diagnostic flag must be boolean")
    _require(_canonical(contract["fixed_horizon"]) == _canonical({"max_steps": 400, "world_size": 4}),
             "Fixed training horizon/world size changed")
    save_steps = 5 if contract["diagnostic"] else 100
    _require(expected_step % save_steps == 0, "Undeclared checkpoint boundary")
    _require(_canonical(contract["schedule"]) == _canonical({"generation_batch_size": 36,
             "gradient_accumulation_steps": 3, "steps_per_generation": 3, "save_steps": save_steps}),
             "Rollout/save schedule changed")
    root = _absolute(checkpoint_dir, "Checkpoint")
    _require(stat.S_ISDIR(root.lstat().st_mode) and root.name == f"checkpoint-{expected_step}"
             and root.parent == contract_path.parent / "checkpoints", "Checkpoint belongs to another run/boundary")
    _reject_kl(root, contract_path)
    files = {name: _identity(root / name) for name in STATE_NAMES}
    state = _load_json(root / "trainer_state.json")
    _require(type(state.get("global_step")) is int and state["global_step"] == expected_step
             and type(state.get("max_steps")) is int and state["max_steps"] == 400, "Trainer step/horizon differs")
    fidelity = verify_checkpoint_fidelity(root)
    closure = fidelity["trainable_closure"]
    # The callback's distributed consensus covers the complete preflight,
    # whereas its adjacent tensor_inventory_sha256 covers only tensor records.
    # Reconstruct the former from verified saved bytes using the same structural
    # inspector as the live preflight; no model or torch import is needed.
    qwen = _qwen_publication_invariant(_load_json(root / "adapter_config.json"), closure["tensors"])
    consensus_digest = _digest({"closure": closure, "qwen_structural_invariant": qwen,
                                "claim_scope": "development_only"})
    source = contract["source"]
    _require(isinstance(source, dict) and set(source) == {"root", "manifest_sha256"}, "Source schema differs")
    manifest = _identity(_absolute(source["root"], "Source") / "SOURCE_MANIFEST.sha256")
    _require(manifest["sha256"] == source["manifest_sha256"], "Original source manifest changed")
    initial = contract["initialization"]
    _require(isinstance(initial, dict) and set(initial) == {"checkpoint", "files"}, "SFT schema differs")
    sft = _absolute(initial["checkpoint"], "SFT")
    _require(sft != root and isinstance(initial["files"], dict)
             and set(initial["files"]) == {"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"},
             "Original SFT identity is incomplete")
    initialization = {}
    for name, record in initial["files"].items():
        _require(isinstance(record, dict) and Path(record.get("path", "")) == sft / name, "Wrong SFT artifact path")
        initialization[name] = _match_identity(record, "Original SFT")
    verify_checkpoint_fidelity(sft)
    data_root = _absolute(contract["data_root"], "Data root")
    _require(isinstance(contract["data_files"], dict) and contract["data_files"], "Training data binding is required")
    data = {}
    for name, record in contract["data_files"].items():
        _require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                 and isinstance(record, dict) and Path(record.get("path", "")) == data_root / name, "Dataset path binding differs")
        data[name] = _match_identity(record, "Training data")
    return root, contract_path, contract, {
        "checkpoint": str(root), "expected_step": expected_step, "max_steps": 400, "world_size": 4,
        "method": "classic_grpo", "run_contract": contract_file, "source_manifest": manifest,
        "initialization_files": initialization, "data_files": data, "files": files,
        "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
        "policy_tensor_inventory_sha256": closure["tensor_inventory_sha256"],
        "policy_preflight_consensus_sha256": consensus_digest,
    }


def _policy_record(path, step, closure_digest, consensus_digest):
    receipt = _signed_json(path)
    _require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
             and receipt.get("protocol") == "clasp-optimizer-step-exact-policy-consistency-v1"
             and receipt.get("status") == "passed" and receipt.get("claim_scope") == "development", "Invalid policy consensus")
    checks = receipt.get("checks")
    _require(isinstance(checks, list) and checks and all(isinstance(c, dict) for c in checks), "Missing consensus checks")
    steps = [c.get("optimizer_step") for c in checks]
    _require(all(type(s) is int and 0 < s <= step for s in steps) and len(set(steps)) == len(steps)
             and receipt.get("verified_optimizer_steps") == steps and max(steps) == step, "Consensus step mismatch")
    record = next(c for c in checks if c["optimizer_step"] == step)
    distributed = record.get("distributed")
    expected = {"phase": f"optimizer_step_{step}_policy_consistency", "world_size": 4,
                "rank_digests": [{"rank": rank, "digest": consensus_digest} for rank in range(4)],
                "consensus_digest": consensus_digest}
    _require(record.get("tensor_inventory_sha256") == closure_digest and _canonical(distributed) == _canonical(expected),
             "Four-rank policy consensus does not match the saved trainable policy")
    return record


def seal_classic_grpo_checkpoint(checkpoint_dir, *, expected_step, expected_contract, run_contract_path):
    """After successful own training, preserve its beta-zero policy/contract proof.

    The launcher must first verify actual subprocess success and the declared
    source, command and GPU allocation. This function submits no job and never
    overwrites existing lineage. The copy survives later root telemetry writes.
    """
    root, contract_path, _, base = _base(checkpoint_dir, expected_step, expected_contract, run_contract_path)
    _require(not (root / LINEAGE_NAME).exists() and not (root / LINEAGE_NAME).is_symlink()
             and not (root / POLICY_NAME).exists() and not (root / POLICY_NAME).is_symlink(), "Checkpoint already sealed")
    original = root.parent / "policy_consistency_checks.json"
    before = _identity(original)
    record = _policy_record(original, expected_step, base["policy_tensor_inventory_sha256"],
                            base["policy_preflight_consensus_sha256"])
    raw = original.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == before["sha256"] and _identity(original) == before,
             "Policy consensus changed during capture")
    _write_once(root / POLICY_NAME, raw)
    policy = _identity(root / POLICY_NAME)
    _, _, _, after = _base(checkpoint_dir, expected_step, expected_contract, run_contract_path)
    _require(after == base, "Checkpoint/contract changed while sealing")
    lineage = {"protocol": LINEAGE_PROTOCOL, "status": "sealed_own_beta_zero_checkpoint",
               "checkpoint_inputs": base, "policy_consistency": policy, "endpoint_policy_consensus": record}
    lineage["receipt_sha256"] = _digest(lineage)
    _write_once(root / LINEAGE_NAME, _canonical(lineage) + b"\n")
    return validate_classic_grpo_resume(root, expected_step=expected_step, expected_contract=expected_contract,
                                        run_contract_path=contract_path)


def validate_classic_grpo_resume(checkpoint_dir, *, expected_step, expected_contract, run_contract_path):
    root, contract_path, _, base = _base(checkpoint_dir, expected_step, expected_contract, run_contract_path)
    lineage_file = _identity(root / LINEAGE_NAME)
    policy_file = _identity(root / POLICY_NAME)
    _require(not (root / LINEAGE_NAME).lstat().st_mode & 0o222
             and not (root / POLICY_NAME).lstat().st_mode & 0o222, "Checkpoint lineage must be read-only")
    lineage = _signed_json(root / LINEAGE_NAME)
    record = _policy_record(root / POLICY_NAME, expected_step, base["policy_tensor_inventory_sha256"],
                            base["policy_preflight_consensus_sha256"])
    _require(lineage.get("protocol") == LINEAGE_PROTOCOL and lineage.get("status") == "sealed_own_beta_zero_checkpoint"
             and lineage.get("checkpoint_inputs") == base and lineage.get("policy_consistency") == policy_file
             and lineage.get("endpoint_policy_consensus") == record, "Checkpoint lineage belongs to different live inputs")
    files = {**base["files"], LINEAGE_NAME: lineage_file, POLICY_NAME: policy_file}
    _require({name: _identity(root / name) for name in files} == files
             and _identity(contract_path) == base["run_contract"], "Checkpoint changed during admission")
    _reject_kl(root, contract_path)
    result = {**base, "protocol": PROTOCOL, "status": "complete_same_contract_beta_zero_resume_inputs_verified",
              "files": files, "further_continuation_possible": expected_step < 400,
              "all_rank_rng_files_present": True, "no_kl_sidecars": True,
              "live_optimizer_scheduler_rng_restoration_verified": False,
              "limitation": "Input completeness and saved policy consensus only; native loaded-state parity remains required."}
    result["receipt_sha256"] = _digest(result)
    return result
