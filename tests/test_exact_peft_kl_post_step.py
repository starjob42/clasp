from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from clasp.models import exact_peft_kl_post_step as post

ROOT = Path(__file__).resolve().parents[1]


def _record(key: str, value: str, *, numel: int = 1) -> dict[str, object]:
    return {
        "key": key,
        "shape": [numel],
        "dtype": "torch.float32",
        "numel": numel,
        "value_sha256": value * 64,
    }


def _inventory(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "tensor_count": len(records),
        "parameter_count": sum(int(record["numel"]) for record in records),
        "tensor_inventory_sha256": post._sha256_json(records),
        "tensors": records,
    }


def _receipt_inputs(
    *,
    role: str,
    changed: bool,
    global_step: int | None = None,
) -> dict[str, object]:
    raw_key = "base_model.model.layer.lora_A.default.weight"
    canonical_key = "base_model.model.layer.lora_A.weight"
    baseline_record = _record(raw_key, "a")
    policy_record = _record(raw_key, "b" if changed else "a")
    trainable_record = _record(canonical_key, "b" if changed else "a")
    baseline = _inventory([baseline_record])
    live = copy.deepcopy(baseline)
    full = _inventory([policy_record])
    trainable = _inventory([trainable_record])
    pre = {
        "protocol": post.PRE_DEVELOPMENT_PROTOCOL,
        "status": post.PRE_DEVELOPMENT_STATUS,
        "receipt_sha256": "1" * 64,
        "receipt_file_sha256": "2" * 64,
        "reference_tensor_count": 1,
        "reference_parameter_count": 1,
        "reference_tensor_inventory_sha256": baseline["tensor_inventory_sha256"],
        "model_class": "tests.Tiny",
        "prepared_class": "tests.Tiny",
        "unwrapped_class": "tests.Tiny",
        "active_adapter": "default",
        "model_config_sha256": "3" * 64,
        "device_types": ["cpu"],
    }
    step1 = None
    if role != "step1":
        step1 = {
            "protocol": post.DEVELOPMENT_PROTOCOL,
            "status": post.DEVELOPMENT_STATUS,
            "receipt_sha256": "4" * 64,
            "receipt_file_sha256": "5" * 64,
            "global_step": 1,
            "reference_tensor_inventory_sha256": baseline["tensor_inventory_sha256"],
            "policy_trainable_tensor_inventory_sha256": trainable["tensor_inventory_sha256"],
        }
    training = {
        "dtype": "bfloat16",
        "beta": 0.05,
        "sync_ref_model": False,
        "use_vllm": False,
        "parameterization": "continue_same_bias_all_sft_adapter",
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
    }
    run = {
        "protocol": post.TRAINING_RUN_CONTRACT_IDENTITY_PROTOCOL,
        "status": post.TRAINING_RUN_CONTRACT_IDENTITY_STATUS,
        "file_sha256": "6" * 64,
        "payload_sha256": "7" * 64,
        "size_bytes": 100,
        "payload_schema_version": 1,
        "payload_stage": "grpo",
        "fixed_horizon_max_steps": 400,
        "fixed_horizon_world_size": 1,
    }
    reference = {
        "method": "live_prepared_reference_immutability_v1",
        "trainer_ref_model_identity": True,
        "reference_prepared_object_identity": True,
        "reference_unwrapped_identity": True,
        "prepared_class": "tests.Tiny",
        "unwrapped_class": "tests.Tiny",
        "active_adapter": "default",
        "model_config_sha256": "3" * 64,
        "reference_trainable_tensor_count": 0,
        "reference_eval_mode": True,
        "reference_input_grad_hook_absent": True,
        "baseline_inventory": baseline,
        "live_inventory": live,
        "live_equals_pre_step_bitwise": True,
        "policy_reference_storage_independent": True,
        "optimizer_excludes_reference_storage": True,
        "optimizer_matches_policy_trainables": True,
        "optimizer_parameter_count": 1,
        "device_types": ["cpu"],
    }
    changed_keys = [canonical_key] if changed else []
    policy = {
        "method": "full_live_policy_state_and_trainable_closure_v1",
        "adapter_name": "default",
        "full_inventory": full,
        "pre_step_tensor_inventory_sha256": baseline["tensor_inventory_sha256"],
        "changed_from_pre_step": changed,
        "changed_tensor_count": len(changed_keys),
        "changed_tensor_keys": changed_keys,
        "changed_tensor_keys_sha256": post._sha256_json(changed_keys),
        "changed_trainable_tensor_count": len(changed_keys),
        "changed_trainable_tensor_keys": changed_keys,
        "changed_trainable_tensor_keys_sha256": post._sha256_json(changed_keys),
        "policy_trainable_tensor_count": 1,
        "policy_trainable_parameter_count": 1,
        "policy_trainable_tensor_inventory_sha256": trainable["tensor_inventory_sha256"],
        "trainable_inventory": trainable,
    }
    fields = {
        "claim_scope": "development_only",
        "artifact_role": role,
        "global_step": global_step if global_step is not None else (1 if role == "step1" else 7),
        "pre_step_reference": pre,
        "step1_reference": step1,
        "training_contract": training,
        "training_run_contract": run,
        "reference_audit": reference,
        "policy_context": policy,
        "source_binding_unchanged": True,
    }
    digest = post._sha256_json(fields)
    memory = {
        "rank": 0,
        "cuda_available": False,
        "device_index": None,
        "physical_device_index": None,
        "visible_device_token": None,
        "device_uuid": None,
        "device_name": None,
        "driver_version": None,
        "cuda_runtime_version": None,
        "total_memory_bytes": None,
        "nvidia_smi_total_memory_mib": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }
    rank_digests = [{"rank": 0, "digest": digest}]
    distributed = {
        "phase": f"exact_peft_kl_{role}_{fields['global_step']}",
        "world_size": 1,
        "rank_digests": rank_digests,
        "consensus_digest": digest,
        "rank_cuda_memory": [memory],
        "distributed_evidence_sha256": post._sha256_json(
            {"rank_digests": rank_digests, "rank_cuda_memory": [memory]}
        ),
    }
    return {**fields, "distributed": distributed}


def _build(
    *,
    role: str,
    changed: bool,
    global_step: int | None = None,
) -> dict[str, object]:
    values = _receipt_inputs(role=role, changed=changed, global_step=global_step)
    return post.build_post_step_receipt(
        artifact_role=role,
        global_step=values["global_step"],
        pre_step_reference=values["pre_step_reference"],
        step1_reference=values["step1_reference"],
        training_contract=values["training_contract"],
        training_run_contract=values["training_run_contract"],
        reference_audit=values["reference_audit"],
        policy_context=values["policy_context"],
        distributed=values["distributed"],
        claim_scope="development",
    )


def test_step1_allows_zero_lr_unchanged_policy_but_checkpoint_does_not() -> None:
    assert _build(role="step1", changed=False)["global_step"] == 1
    assert _build(role="checkpoint", global_step=1, changed=False)["global_step"] == 1
    with pytest.raises(post.ExactPeftKLPostStepError, match="update/closure"):
        _build(role="checkpoint", changed=False)
    with pytest.raises(post.ExactPeftKLPostStepError, match="update/closure"):
        _build(role="final", global_step=400, changed=False)
    assert _build(role="checkpoint", changed=True)["global_step"] == 7


def test_role_chronology_is_bounded_by_the_fixed_horizon() -> None:
    assert _build(role="final", global_step=400, changed=True)["global_step"] == 400
    with pytest.raises(post.ExactPeftKLPostStepError, match="fixed horizon"):
        _build(role="checkpoint", global_step=401, changed=True)
    with pytest.raises(post.ExactPeftKLPostStepError, match="fixed horizon"):
        _build(role="final", global_step=399, changed=True)


def test_resealed_nontrainable_policy_change_is_rejected() -> None:
    values = _receipt_inputs(role="checkpoint", changed=True)
    values["policy_context"]["changed_trainable_tensor_keys"] = []
    values["policy_context"]["changed_trainable_tensor_count"] = 0
    values["policy_context"]["changed_trainable_tensor_keys_sha256"] = post._sha256_json([])
    fields = {key: value for key, value in values.items() if key != "distributed"}
    digest = post._sha256_json(fields)
    values["distributed"]["rank_digests"][0]["digest"] = digest
    values["distributed"]["consensus_digest"] = digest
    values["distributed"]["distributed_evidence_sha256"] = post._sha256_json(
        {
            "rank_digests": values["distributed"]["rank_digests"],
            "rank_cuda_memory": values["distributed"]["rank_cuda_memory"],
        }
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="update/closure"):
        post.build_post_step_receipt(
            artifact_role=values["artifact_role"],
            global_step=values["global_step"],
            pre_step_reference=values["pre_step_reference"],
            step1_reference=values["step1_reference"],
            training_contract=values["training_contract"],
            training_run_contract=values["training_run_contract"],
            reference_audit=values["reference_audit"],
            policy_context=values["policy_context"],
            distributed=values["distributed"],
            claim_scope="development",
        )


def test_integrity_only_identity_is_self_hashed_and_tamper_strict() -> None:
    receipt = _build(role="checkpoint", changed=True)
    raw = post._receipt_bytes(receipt)
    identity = post.post_step_reference_identity_from_receipt_payload(
        receipt,
        receipt_file_sha256=post._sha256_bytes(raw),
        claim_scope="development",
    )
    assert identity["identity_sha256"] == post._sha256_json(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    tampered = dict(identity)
    tampered["world_size"] = 2
    with pytest.raises(post.ExactPeftKLPostStepError, match="self-hash"):
        post.validate_post_step_reference_identity(tampered)

    resealed = copy.deepcopy(identity)
    resealed["invariant"]["extra"] = True
    resealed["invariant_sha256"] = post._sha256_json(resealed["invariant"])
    resealed["identity_sha256"] = post._sha256_json(
        {key: value for key, value in resealed.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="invariant"):
        post.validate_post_step_reference_identity(resealed)

    resealed = copy.deepcopy(identity)
    resealed["distributed_evidence_sha256"] = "f" * 64
    resealed["identity_sha256"] = post._sha256_json(
        {key: value for key, value in resealed.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="distributed evidence"):
        post.validate_post_step_reference_identity(resealed)

    resealed = copy.deepcopy(identity)
    resealed["global_step"] = 401
    resealed["identity_sha256"] = post._sha256_json(
        {key: value for key, value in resealed.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="fixed horizon"):
        post.validate_post_step_reference_identity(resealed)

    resealed = copy.deepcopy(identity)
    resealed["invariant"]["reference_method"] = "hand_authored"
    resealed["invariant_sha256"] = post._sha256_json(resealed["invariant"])
    resealed["identity_sha256"] = post._sha256_json(
        {key: value for key, value in resealed.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="nested invariant"):
        post.validate_post_step_reference_identity(resealed)

    resealed = copy.deepcopy(identity)
    resealed["invariant"]["pre_step_reference"]["model_class"] = "tests.Forged"
    resealed["invariant_sha256"] = post._sha256_json(resealed["invariant"])
    resealed["identity_sha256"] = post._sha256_json(
        {key: value for key, value in resealed.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="projection"):
        post.validate_post_step_reference_identity(resealed)


def test_portable_final_identity_requires_the_contracted_terminal_step() -> None:
    receipt = _build(role="final", global_step=400, changed=True)
    identity = post.post_step_reference_identity_from_receipt_payload(
        receipt,
        receipt_file_sha256=post._sha256_bytes(post._receipt_bytes(receipt)),
        claim_scope="development",
    )
    tampered = copy.deepcopy(identity)
    tampered["global_step"] = 399
    tampered["identity_sha256"] = post._sha256_json(
        {key: value for key, value in tampered.items() if key != "identity_sha256"}
    )
    with pytest.raises(post.ExactPeftKLPostStepError, match="fixed horizon"):
        post.validate_post_step_reference_identity(tampered)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Module()
        self.layer.lora_A = nn.Module()
        self.layer.lora_A.default = nn.Linear(1, 1, bias=False)
        self.active_adapter = "default"
        self.config = SimpleNamespace(to_dict=lambda: {"model_type": "tiny"})


def test_live_audit_requires_exact_optimizer_and_reference_eval() -> None:
    policy = _Tiny()
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    baseline_records = [
        post._tensor_record(key, tensor) for key, tensor in sorted(reference.state_dict().items())
    ]
    pre = {
        "runtime_pair": {
            **_inventory(baseline_records),
            "tensors": baseline_records,
        }
    }
    with torch.no_grad():
        next(policy.parameters()).add_(1.0)
    canonical = "layer.lora_A.weight"
    tensor = next(policy.parameters())
    record = post._tensor_record(canonical, tensor)
    closure = {
        "adapter_name": "default",
        "tensor_count": 1,
        "parameter_count": int(tensor.numel()),
        "tensor_inventory_sha256": post._sha256_json([record]),
        "tensors": [record],
    }
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    runtime = {
        "trainer_ref_model_identity": True,
        "reference_prepared_object_identity": True,
        "reference_unwrapped_identity": True,
        "prepared_class": f"{reference.__class__.__module__}.{reference.__class__.__qualname__}",
        "unwrapped_class": f"{reference.__class__.__module__}.{reference.__class__.__qualname__}",
    }
    reference_audit, policy_context = post.audit_live_post_step_reference(
        policy,
        reference,
        pre_step_receipt=pre,
        trainable_closure=closure,
        optimizer=optimizer,
        runtime_identity=runtime,
        require_policy_update=True,
    )
    assert reference_audit["optimizer_matches_policy_trainables"] is True
    assert policy_context["changed_tensor_keys"] == [canonical]

    reference.train()
    with pytest.raises(post.ExactPeftKLPostStepError, match="eval mode"):
        post.audit_live_post_step_reference(
            policy,
            reference,
            pre_step_receipt=pre,
            trainable_closure=closure,
            optimizer=optimizer,
            runtime_identity=runtime,
            require_policy_update=True,
        )
    reference.eval()
    reference.layer.train()
    with pytest.raises(post.ExactPeftKLPostStepError, match="eval mode"):
        post.audit_live_post_step_reference(
            policy,
            reference,
            pre_step_receipt=pre,
            trainable_closure=closure,
            optimizer=optimizer,
            runtime_identity=runtime,
            require_policy_update=True,
        )
    reference.eval()
    reference._require_grads_hook = object()
    with pytest.raises(post.ExactPeftKLPostStepError, match="input-requires-grad hook"):
        post.audit_live_post_step_reference(
            policy,
            reference,
            pre_step_receipt=pre,
            trainable_closure=closure,
            optimizer=optimizer,
            runtime_identity=runtime,
            require_policy_update=True,
        )


def test_strict_parser_and_tensor_shape_validation(tmp_path: Path) -> None:
    duplicate = tmp_path / post.STEP1_RECEIPT_FILENAME
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    duplicate.chmod(0o440)
    with pytest.raises(post.ExactPeftKLPostStepError, match="duplicate JSON key"):
        post.inspect_post_step_receipt(
            duplicate,
            claim_scope="development",
            rehash_sources=False,
            rehash_policy_artifact=False,
        )
    malformed = _record("tensor", "a", numel=2)
    malformed["shape"] = [3]
    with pytest.raises(post.ExactPeftKLPostStepError, match="numel/shape"):
        post._validate_tensor(malformed, label="malformed")


def test_writer_rejects_filename_role_mismatch(tmp_path: Path) -> None:
    receipt = _build(role="checkpoint", changed=True)
    with pytest.raises(post.ExactPeftKLPostStepError, match="filename/artifact-role"):
        post.write_post_step_receipt(
            tmp_path,
            receipt,
            claim_scope="development",
            filename=post.STEP1_RECEIPT_FILENAME,
            rehash_sources=False,
        )


def test_run_contract_identity_binds_fixed_horizon(tmp_path: Path) -> None:
    path = tmp_path / "run_contract.json"
    payload = {
        "schema_version": 2,
        "stage": "grpo",
        "fixed_horizon": {"max_steps": 3_200, "world_size": 4},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o440)
    identity = post.training_run_contract_identity(path, claim_scope="publication")
    assert identity["fixed_horizon_max_steps"] == 3_200
    assert identity["fixed_horizon_world_size"] == 4

    path.chmod(0o640)
    payload["fixed_horizon"]["max_steps"] = 400
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o440)
    changed = post.training_run_contract_identity(path, claim_scope="publication")
    assert changed["file_sha256"] != identity["file_sha256"]
    assert changed["fixed_horizon_max_steps"] == 400


def test_train_grpo_nonterminal_path_skips_root_finalization() -> None:
    text = (ROOT / "src/clasp/entrypoints/train_grpo.py").read_text(encoding="utf-8")
    branch = text.index("if not terminal_horizon:")
    verify = text.index("verify_publication_resume_checkpoint(", branch)
    early_return = text.index("return output_dir", verify)
    final_post = text.index("final_post_step = run_exact_peft_kl_post_step_audit(", early_return)
    save = text.index("trainer.save_model(output_dir)", final_post)
    assert branch < verify < early_return < final_post < save


def test_two_nonterminal_chunks_do_not_claim_terminal_horizon() -> None:
    from clasp.entrypoints.train_grpo import (
        fixed_horizon_exit_is_terminal,
        validate_fixed_horizon_preflight,
    )

    assert fixed_horizon_exit_is_terminal(400, 3_200) is False
    assert fixed_horizon_exit_is_terminal(800, 3_200) is False
    assert fixed_horizon_exit_is_terminal(3_200, 3_200) is True
    with pytest.raises(RuntimeError, match="beyond"):
        fixed_horizon_exit_is_terminal(3_201, 3_200)

    contract = {
        "fixed_horizon_max_steps": 3_200,
        "fixed_horizon_world_size": 4,
    }
    validate_fixed_horizon_preflight(
        max_steps=3_200,
        runtime_world_size=4,
        contract_identity=contract,
    )
    with pytest.raises(RuntimeError, match="immutable run contract"):
        validate_fixed_horizon_preflight(
            max_steps=400,
            runtime_world_size=4,
            contract_identity=contract,
        )
    with pytest.raises(RuntimeError, match="immutable run contract"):
        validate_fixed_horizon_preflight(
            max_steps=3_200,
            runtime_world_size=1,
            contract_identity=contract,
        )
