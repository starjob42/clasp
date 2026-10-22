"""Exercise the actual PEFT closure/full-policy audit dtype boundary on CPU."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from clasp.models import exact_peft_kl_post_step as post
from clasp.models.checkpoint_fidelity import audit_trainable_state_closure


def _live_case(dtype):
    torch.manual_seed(43)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )).to(dtype=dtype)
    policy = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=2, target_modules=["q_proj"],
        bias="all", task_type="CAUSAL_LM",
    ))
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    pre = {"runtime_pair": post._state_inventory(reference)}
    optimizer = torch.optim.SGD(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.add_(0.125)
    closure = audit_trainable_state_closure(policy)
    model_class = f"{type(reference).__module__}.{type(reference).__qualname__}"
    return {
        "policy": policy, "reference": reference,
        "pre_step_receipt": pre, "trainable_closure": closure,
        "optimizer": optimizer,
        "runtime_identity": {
            "trainer_ref_model_identity": True,
            "reference_prepared_object_identity": True,
            "reference_unwrapped_identity": True,
            "prepared_class": model_class, "unwrapped_class": model_class,
        }, "require_policy_update": True,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_actual_peft_trainable_closure_matches_exact_live_policy(dtype):
    case = _live_case(dtype)
    closure = case["trainable_closure"]
    audit, context = post.audit_live_post_step_reference(**case)
    assert audit["live_equals_pre_step_bitwise"] is True
    assert context["changed_trainable_tensor_count"] == closure["tensor_count"]
    assert context["trainable_inventory"]["tensors"] == closure["tensors"]
    assert context["policy_trainable_tensor_inventory_sha256"] == closure["tensor_inventory_sha256"]


def _mutate_closure(closure, mutation):
    closure = copy.deepcopy(closure)
    record = next(record for record in closure["tensors"] if record["dtype"] == "BF16")
    if mutation == "same_width_wrong_dtype":
        record["dtype"] = "F16"
    elif mutation == "unknown_dtype":
        record["dtype"] = "unknown_dtype"
    else:
        record["value_sha256"] = "0" * 64
    closure["tensor_inventory_sha256"] = post._sha256_json(closure["tensors"])
    return closure


@pytest.mark.parametrize("mutation", ["same_width_wrong_dtype", "unknown_dtype", "wrong_value_hash"])
def test_live_audit_rejects_real_dtype_or_value_mismatch(mutation):
    case = _live_case(torch.bfloat16)
    case["trainable_closure"] = _mutate_closure(case["trainable_closure"], mutation)
    with pytest.raises(post.ExactPeftKLPostStepError):
        post.audit_live_post_step_reference(**case)


def _reseal_distributed(receipt):
    distributed = receipt["distributed"]
    local = post._sha256_json(post._local_digest_fields(receipt))
    distributed["consensus_digest"] = local
    for row in distributed["rank_digests"]:
        row["digest"] = local
    distributed["distributed_evidence_sha256"] = post._sha256_json({
        "rank_digests": distributed["rank_digests"],
        "rank_cuda_memory": distributed["rank_cuda_memory"],
    })


def _real_step1_receipt(dtype):
    # Reuse only the receipt's schema scaffolding; all model, tensor, and
    # trainable-closure evidence below comes from the actual PEFT model.
    path = Path(__file__).with_name("test_exact_peft_kl_post_step.py")
    spec = importlib.util.spec_from_file_location("post_step_schema_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    values = fixture._receipt_inputs(role="step1", changed=True)
    case = _live_case(dtype)
    audit, context = post.audit_live_post_step_reference(**case)
    values["reference_audit"] = audit
    values["policy_context"] = context
    pre = values["pre_step_reference"]
    pre.update(
        reference_tensor_count=audit["baseline_inventory"]["tensor_count"],
        reference_parameter_count=audit["baseline_inventory"]["parameter_count"],
        reference_tensor_inventory_sha256=audit["baseline_inventory"]["tensor_inventory_sha256"],
        model_class=audit["unwrapped_class"], prepared_class=audit["prepared_class"],
        unwrapped_class=audit["unwrapped_class"], active_adapter=audit["active_adapter"],
        model_config_sha256=audit["model_config_sha256"], device_types=audit["device_types"],
    )
    _reseal_distributed(values)
    arguments = {key: values[key] for key in (
        "artifact_role", "global_step", "pre_step_reference", "step1_reference",
        "training_contract", "training_run_contract", "reference_audit",
        "policy_context", "distributed",
    )}
    return post.build_post_step_receipt(**arguments, claim_scope="development")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_serialized_receipt_round_trip_preserves_real_closure_encoding(tmp_path, dtype):
    receipt = _real_step1_receipt(dtype)
    post.write_post_step_receipt(tmp_path, receipt, claim_scope="development",
                                 filename=post.STEP1_RECEIPT_FILENAME, rehash_sources=False)
    actual = post.inspect_post_step_receipt(
        tmp_path / post.STEP1_RECEIPT_FILENAME, claim_scope="development",
        expected_global_step=1, expected_artifact_role="step1",
        rehash_sources=False, rehash_policy_artifact=False,
    )
    assert actual == receipt
    assert all(record["dtype"] in {"F32", "F16", "BF16"}
               for record in actual["policy_context"]["trainable_inventory"]["tensors"])


@pytest.mark.parametrize("mutation", ["same_width_wrong_dtype", "unknown_dtype", "wrong_value_hash"])
def test_resealed_serialized_receipt_rejects_real_dtype_or_value_mismatch(tmp_path, mutation):
    receipt = _real_step1_receipt(torch.bfloat16)
    context = receipt["policy_context"]
    closure = _mutate_closure(context["trainable_inventory"], mutation)
    context["trainable_inventory"] = closure
    context["policy_trainable_tensor_inventory_sha256"] = closure["tensor_inventory_sha256"]
    _reseal_distributed(receipt)
    receipt["receipt_sha256"] = post._sha256_json({key: value for key, value in receipt.items()
                                                   if key != "receipt_sha256"})
    path = tmp_path / post.STEP1_RECEIPT_FILENAME
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True) + "\n")
    path.chmod(0o440)
    with pytest.raises(post.ExactPeftKLPostStepError):
        post.inspect_post_step_receipt(path, claim_scope="development",
                                      rehash_sources=False, rehash_policy_artifact=False)
