"""A real tiny fidelity/post-step bundle exercises strict continuation admission."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import struct
from pathlib import Path

import pytest
import torch

from clasp.models import checkpoint_fidelity as fidelity
from clasp.models import exact_peft_kl_post_step as post
from clasp.models import foarl_teacher_resume as resume
from test_checkpoint_fidelity import _closure, _record, _write_adapter
from test_exact_peft_kl_post_step import _receipt_inputs


def write_json(path, value, readonly=False):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if readonly:
        path.chmod(0o440)


def tiny_adapter(path, value):
    key = "base_model.model.layer.lora_A.weight"
    encoded = struct.pack("<f", value)
    _write_adapter(path, {key: ("F32", [1], encoded)})
    closure = _closure([_record(key, "F32", [1], encoded)])
    fidelity.write_checkpoint_fidelity_receipt(path, closure)
    return closure


def reseal_post(values):
    fields = {key: value for key, value in values.items() if key != "distributed"}
    digest = post._sha256_json(fields)
    old_memory = values["distributed"]["rank_cuda_memory"][0]
    memory = [{**old_memory, "rank": rank} for rank in range(4)]
    ranks = [{"rank": rank, "digest": digest} for rank in range(4)]
    values["distributed"] = {
        "phase": f"exact_peft_kl_{values['artifact_role']}_{values['global_step']}",
        "world_size": 4, "rank_digests": ranks, "consensus_digest": digest,
        "rank_cuda_memory": memory,
        "distributed_evidence_sha256": post._sha256_json({"rank_digests": ranks,
                                                         "rank_cuda_memory": memory}),
    }
    return post.build_post_step_receipt(
        artifact_role=values["artifact_role"], global_step=values["global_step"],
        pre_step_reference=values["pre_step_reference"], step1_reference=values["step1_reference"],
        training_contract=values["training_contract"], training_run_contract=values["training_run_contract"],
        reference_audit=values["reference_audit"], policy_context=values["policy_context"],
        distributed=values["distributed"], claim_scope="development",
    )


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    checkpoint = output / "checkpoints/checkpoint-100"
    closure = tiny_adapter(checkpoint, 2)
    sft = tmp_path / "sft"
    tiny_adapter(sft, 1)
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("original pinned source manifest\n")
    data = tmp_path / "data"
    data.mkdir()
    train = data / "train.json"
    train.write_text('[{"synthetic":true}]\n')
    contract = {
        "schema_version": 1, "stage": "grpo", "status": "development_only_not_publication_eligible",
        "method": "foarl_teacher", "source": {"root": str(source),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "initialization": {"checkpoint": str(sft), "files": {
            name: resume._identity(sft / name) for name in
            ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}},
        "fixed_horizon": {"max_steps": 400, "world_size": 4}, "teacher_weight": .05,
        "seed": 43, "data_root": str(data), "data_files": {"train.json": resume._identity(train)},
        "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                     "steps_per_generation": 3, "save_steps": 100}, "diagnostic": False,
    }
    contract_path = output / "local_development_run_contract.json"
    write_json(contract_path, contract, readonly=True)
    write_json(checkpoint / "trainer_state.json", {"global_step": 100, "max_steps": 400, "epoch": .2})
    torch.save({"state": {0: {"step": torch.tensor(100), "exp_avg": torch.ones(1)}},
                "param_groups": [{"params": [0], "lr": 1e-6}]}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 100, "_step_count": 101}, checkpoint / "scheduler.pt")
    for rank in range(4):
        torch.save({"python": random.Random(rank).getstate(), "cpu": torch.get_rng_state()},
                   checkpoint / f"rng_state_{rank}.pth")
    # The loader/full exact-KL verifier independently validates these opaque
    # siblings; this gate binds their exact bytes to the valid post-step proof.
    for name in ("exact_peft_kl_reference.json", post.STEP1_RECEIPT_FILENAME):
        write_json(checkpoint / name, {"synthetic_sibling": name}, readonly=True)
    values = _receipt_inputs(role="checkpoint", changed=True, global_step=100)
    values["training_run_contract"] = post.training_run_contract_identity(contract_path, claim_scope="development")
    values["pre_step_reference"]["receipt_file_sha256"] = resume._identity(checkpoint / "exact_peft_kl_reference.json")["sha256"]
    values["step1_reference"]["receipt_file_sha256"] = resume._identity(checkpoint / post.STEP1_RECEIPT_FILENAME)["sha256"]
    policy = values["policy_context"]
    trainable = {key: closure[key] for key in ("tensor_count", "parameter_count", "tensor_inventory_sha256", "tensors")}
    policy["trainable_inventory"] = trainable
    policy["policy_trainable_tensor_inventory_sha256"] = trainable["tensor_inventory_sha256"]
    full = copy.deepcopy(trainable)
    full["tensors"][0]["key"] = "base_model.model.layer.lora_A.default.weight"
    full["tensors"][0]["dtype"] = "torch.float32"
    full["tensor_inventory_sha256"] = post._sha256_json(full["tensors"])
    policy["full_inventory"] = full
    proof = reseal_post(values)
    write_json(checkpoint / post.POST_STEP_RECEIPT_FILENAME, proof, readonly=True)
    return checkpoint, contract_path, contract, values


def validate(bundle, **kwargs):
    checkpoint, contract_path, contract, _ = bundle
    return resume.validate_foarl_teacher_resume(
        checkpoint, expected_step=100, expected_contract=contract, run_contract_path=contract_path, **kwargs)


def test_complete_bundle_binds_policy_contract_all_optimizer_rng_files_without_mutation(bundle):
    checkpoint = bundle[0]
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    result = validate(bundle)
    assert result["status"] == "complete_same_contract_resume_inputs_verified"
    assert len(result["files"]) == 13
    assert all(f"rng_state_{rank}.pth" in result["files"] for rank in range(4))
    assert result["all_rank_rng_files_present"]
    assert result["live_optimizer_scheduler_rng_restoration_verified"] is False
    assert {p.name: p.read_bytes() for p in checkpoint.iterdir()} == before


@pytest.mark.parametrize("name", ["optimizer.pt", "scheduler.pt", "trainer_state.json", "checkpoint_fidelity.json",
    "adapter_config.json", "adapter_model.safetensors", "exact_peft_kl_reference.json",
    post.STEP1_RECEIPT_FILENAME, post.POST_STEP_RECEIPT_FILENAME,
    *[f"rng_state_{rank}.pth" for rank in range(4)]])
def test_every_required_file_is_mandatory(bundle, name):
    (bundle[0] / name).unlink()
    with pytest.raises((OSError, ValueError)):
        validate(bundle)


@pytest.mark.parametrize("kind", ["empty", "symlink", "dangling"])
def test_opaque_resume_files_must_be_nonempty_regular_files(bundle, kind):
    path = bundle[0] / "optimizer.pt"
    path.unlink()
    if kind == "empty":
        path.touch()
    else:
        path.symlink_to(bundle[0] / ("scheduler.pt" if kind == "symlink" else "absent"))
    with pytest.raises(ValueError):
        validate(bundle)


@pytest.mark.parametrize("field,value", [("global_step", 99), ("global_step", True), ("max_steps", 100)])
def test_trainer_step_and_original_horizon_must_match(bundle, field, value):
    state = {"global_step": 100, "max_steps": 400, field: value}
    write_json(bundle[0] / "trainer_state.json", state)
    with pytest.raises(ValueError, match="Trainer"):
        validate(bundle)


def test_changed_expected_method_cannot_reuse_original_contract(bundle):
    bundle[2]["method"] = "foarl_control"
    with pytest.raises(ValueError, match="Original run contract differs"):
        validate(bundle)


def test_resealed_other_arm_post_step_proof_still_rejected(bundle):
    values = bundle[3]
    values["training_run_contract"]["file_sha256"] = "f" * 64
    proof = reseal_post(values)
    path = bundle[0] / post.POST_STEP_RECEIPT_FILENAME
    path.unlink()
    write_json(path, proof, readonly=True)
    with pytest.raises(ValueError, match="different original run contract"):
        validate(bundle)


@pytest.mark.parametrize("target", ["source", "data", "sft", "policy", "pre_kl", "step1"])
def test_live_bytes_cannot_drift_from_their_bound_evidence(bundle, target):
    checkpoint, _, contract, _ = bundle
    path = {
        "source": Path(contract["source"]["root"]) / "SOURCE_MANIFEST.sha256",
        "data": Path(contract["data_files"]["train.json"]["path"]),
        "sft": Path(contract["initialization"]["checkpoint"]) / "adapter_model.safetensors",
        "policy": checkpoint / "adapter_model.safetensors",
        "pre_kl": checkpoint / "exact_peft_kl_reference.json",
        "step1": checkpoint / post.STEP1_RECEIPT_FILENAME,
    }[target]
    path.chmod(0o640)
    with path.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ValueError):
        validate(bundle)


def test_expected_checkpoint_must_belong_to_run_directory(bundle, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    copied_contract = other / "local_development_run_contract.json"
    copied_contract.write_bytes(bundle[1].read_bytes())
    copied_contract.chmod(0o440)
    with pytest.raises(ValueError, match="original run directory"):
        resume.validate_foarl_teacher_resume(bundle[0], expected_step=100,
            expected_contract=bundle[2], run_contract_path=copied_contract)


def test_checkpoint_step400_is_completion_admission_not_another_resumable_chunk(bundle):
    old, contract_path, contract, values = bundle
    checkpoint = old.with_name("checkpoint-400")
    old.rename(checkpoint)
    write_json(checkpoint / "trainer_state.json", {"global_step": 400, "max_steps": 400})
    values["global_step"] = 400
    path = checkpoint / post.POST_STEP_RECEIPT_FILENAME
    path.unlink()
    write_json(path, reseal_post(values), readonly=True)
    result = resume.validate_foarl_teacher_resume(checkpoint, expected_step=400,
        expected_contract=contract, run_contract_path=contract_path)
    assert result["further_continuation_possible"] is False


@pytest.mark.parametrize("kwargs", [{"world_size": 2}, {"world_size": True}, {"final_horizon": 100}])
def test_no_world_or_horizon_change(bundle, kwargs):
    with pytest.raises(ValueError):
        validate(bundle, **kwargs)
