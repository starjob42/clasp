from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from clasp.models import publication_fidelity as fidelity


def _verified_receipt() -> dict:
    return {
        "schema_version": fidelity.CHECKPOINT_FIDELITY_SCHEMA_VERSION,
        "protocol": fidelity.CHECKPOINT_FIDELITY_PROTOCOL,
        "status": "exact_trainable_policy_persisted",
        "receipt_sha256": "1" * 64,
        "artifacts": {
            "adapter_config": {"sha256": "2" * 64},
            "adapter_weights": {"sha256": "3" * 64},
        },
        "trainable_closure": {"tensor_inventory_sha256": "4" * 64},
        "publication_invariant": {
            "protocol": fidelity.QWEN_PUBLICATION_PROTOCOL,
            "status": "exact",
            "trainable_tensor_count": 476,
        },
    }


def _tree(label: str) -> dict:
    files = [{"path": label, "size_bytes": 1, "sha256": hashlib.sha256(label.encode()).hexdigest()}]
    return {
        "file_count": 1,
        "size_bytes": 1,
        "files": files,
        "tree_sha256": fidelity._sha256_json(files),
    }


def _source_binding(source: Path) -> dict:
    source_identity = fidelity._identity_from_verified_receipt(
        _verified_receipt(), receipt_file_sha256="5" * 64
    )
    return {
        "method": "immutable_local_base_adapter_tokenizer_trees_v2",
        "claim_scope": "publication",
        "adapter_path": str(source),
        "adapter_tree": _tree("adapter"),
        "base_model_path": str(source / "base"),
        "base_model_tree": _tree("base"),
        "tokenizer_path": str(source),
        "tokenizer_tree": _tree("tokenizer"),
        "authoritative_receipt_path": str(source / fidelity.RECEIPT_FILENAME),
        "authoritative_receipt_protocol": fidelity.CHECKPOINT_FIDELITY_PROTOCOL,
        "authoritative_receipt_sha256": source_identity["receipt_sha256"],
        "authoritative_receipt_file_sha256": source_identity["receipt_file_sha256"],
        "adapter_config_sha256": source_identity["adapter_config_sha256"],
        "saved_adapter_tensor_inventory_sha256": "e" * 64,
        "saved_adapter_dtypes": ["F32"],
        "saved_lora_dtype": "F32",
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
    }


def _fixed_identity(source: Path) -> dict:
    source_identity = fidelity._identity_from_verified_receipt(
        _verified_receipt(), receipt_file_sha256="5" * 64
    )
    return fidelity._fixed_kl_identity_from_source_binding(
        _source_binding(source),
        source_identity=source_identity,
    )


def _runtime_identity(source: Path) -> dict:
    fixed = _fixed_identity(source)
    receipt = _runtime_receipt(source)
    training = receipt["training_contract"]
    runtime_pair = receipt["runtime_pair"]
    functional = runtime_pair["functional_equivalence"]
    distributed = receipt["distributed"]
    receipt_bytes = (
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    )
    identity = {
        "schema_version": fidelity.EXACT_PEFT_KL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "protocol": fidelity.EXACT_PEFT_KL_RUNTIME_IDENTITY_PROTOCOL,
        "status": fidelity.EXACT_PEFT_KL_RUNTIME_IDENTITY_STATUS,
        "runtime_schema_version": fidelity.exact_peft_kl.SCHEMA_VERSION,
        "runtime_protocol": fidelity.exact_peft_kl.PUBLICATION_PROTOCOL,
        "runtime_status": fidelity.exact_peft_kl.PUBLICATION_STATUS,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "verifier_protocol": fidelity.exact_peft_kl.VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": fixed["verifier_implementation_sha256"],
        "fixed_kl_reference": fixed,
        "training_dtype": training["dtype"],
        "training_beta": training["beta"],
        "training_sync_ref_model": training["sync_ref_model"],
        "training_use_vllm": training["use_vllm"],
        "training_parameterization": training["parameterization"],
        "training_policy_adapter_autocast_dtype": training["policy_adapter_autocast_dtype"],
        "training_reference_adapter_autocast_dtype": training["reference_adapter_autocast_dtype"],
        "training_contract_sha256": fidelity._sha256_json(training),
        "runtime_pair_tensor_inventory_sha256": runtime_pair["tensor_inventory_sha256"],
        "runtime_pair_sha256": fidelity._sha256_json(runtime_pair),
        "functional_equivalence_sha256": fidelity._sha256_json(functional),
        "distributed_world_size": distributed["world_size"],
        "distributed_consensus_digest": distributed["consensus_digest"],
        "distributed_sha256": fidelity._sha256_json(distributed),
    }
    identity["identity_sha256"] = fidelity._sha256_json(identity)
    return identity


def _tensor_record(key: str) -> dict:
    return {
        "key": key,
        "shape": [1],
        "dtype": "torch.float32",
        "numel": 1,
        "value_sha256": hashlib.sha256(key.encode()).hexdigest(),
    }


def _runtime_receipt(source: Path) -> dict:
    binding = _source_binding(source)
    tensors = [_tensor_record("policy.weight")]
    model_class = "tests.FrozenPair"
    functional = {
        "method": "three_fixed_probes_bitwise_runtime_dtype_full_distribution_v2",
        "runtime_dtype": "bfloat16",
        "autocast_enabled": True,
        "autocast_device_type": "cpu",
        "input_ids": [list(row) for row in fidelity.exact_peft_kl.FIXED_PROBES],
        "input_ids_sha256": fidelity._sha256_json(fidelity.exact_peft_kl.FIXED_PROBES),
        "logits": _tensor_record("logits"),
        "log_probabilities": _tensor_record("log_probabilities"),
        "argmax": _tensor_record("argmax"),
        "bitwise_equal": True,
        "max_absolute_error": 0.0,
    }
    runtime_pair = {
        "method": "independent_same_base_same_adapter_exact_runtime_pair_v2",
        "model_class": model_class,
        "active_adapter": "default",
        "model_config_sha256": "1" * 64,
        "model_objects_distinct": True,
        "independent_storage": True,
        "reference_trainable_tensor_count": 0,
        "policy_trainable_tensor_count": 1,
        "tensor_count": 1,
        "parameter_count": 1,
        "tensor_inventory_sha256": fidelity._sha256_json(tensors),
        "tensors": tensors,
        "state_bitwise_equal": True,
        "prepared_runtime": {
            "trainer_ref_model_identity": True,
            "reference_unwrapped_identity": True,
            "policy_unwrapped_class": model_class,
            "reference_prepared_class": model_class,
            "reference_unwrapped_class": model_class,
            "device_types": ["cpu"],
        },
        "functional_equivalence": functional,
    }
    consensus = fidelity._sha256_json(
        {
            "source_binding": binding,
            "runtime_pair": runtime_pair,
            "claim_scope": "publication",
        }
    )
    distributed = {
        "phase": "exact_peft_kl_step_zero",
        "world_size": 1,
        "rank_digests": [{"rank": 0, "digest": consensus}],
        "consensus_digest": consensus,
    }
    return fidelity.exact_peft_kl.build_runtime_receipt(
        source_binding=binding,
        runtime_pair=runtime_pair,
        distributed=distributed,
        training_contract={
            "dtype": "bfloat16",
            "beta": 0.01,
            "sync_ref_model": False,
            "use_vllm": False,
            "parameterization": "continue_same_bias_all_sft_adapter",
            "policy_adapter_autocast_dtype": True,
            "reference_adapter_autocast_dtype": True,
        },
        claim_scope="publication",
    )


def _post_step_lineage(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal_step: int = 400,
    selected_step: int = 400,
) -> tuple[dict, dict, dict]:
    fixed = _fixed_identity(source)
    runtime = _runtime_identity(source)
    checkpoint_fidelity = fidelity._identity_from_verified_receipt(
        _verified_receipt(), receipt_file_sha256="5" * 64
    )
    source_reference = fidelity.exact_peft_kl_source_reference(fixed)
    invariant = {
        "training_run_contract": {
            "fixed_horizon_max_steps": terminal_step,
            "fixed_horizon_world_size": 1,
        }
    }
    invariant_sha256 = fidelity._sha256_json(invariant)
    contract_sha256 = "6" * 64

    def identity(role: str, step: int, *, changed: int) -> dict:
        return {
            "artifact_role": role,
            "global_step": step,
            "invariant": copy.deepcopy(invariant),
            "invariant_sha256": invariant_sha256,
            "verifier_protocol": source_reference["post_step_verifier_protocol"],
            "verifier_implementation_sha256": source_reference[
                "post_step_verifier_implementation_sha256"
            ],
            "training_run_contract_file_sha256": contract_sha256,
            "training_run_contract_max_steps": terminal_step,
            "pre_step_receipt_sha256": runtime["receipt_sha256"],
            "pre_step_receipt_file_sha256": runtime["receipt_file_sha256"],
            "beta": runtime["training_beta"],
            "training_dtype": runtime["training_dtype"],
            "world_size": runtime["distributed_world_size"],
            "policy_trainable_tensor_inventory_sha256": checkpoint_fidelity[
                "trainable_tensor_inventory_sha256"
            ],
            "changed_trainable_tensor_count": changed,
        }

    root_step1 = identity("step1", 1, changed=0)
    root_final = identity("final", terminal_step, changed=1)
    selected_post = identity("checkpoint", selected_step, changed=1)
    lineage = {
        "schema_version": fidelity.EXACT_PEFT_KL_POST_STEP_LINEAGE_SCHEMA_VERSION,
        "protocol": fidelity.EXACT_PEFT_KL_POST_STEP_LINEAGE_PROTOCOL,
        "status": fidelity.EXACT_PEFT_KL_POST_STEP_LINEAGE_STATUS,
        "terminal_global_step": terminal_step,
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_is_terminal": selected_step == terminal_step,
        "training_run_contract_file_sha256": contract_sha256,
        "exact_peft_kl_runtime_identity_sha256": runtime["identity_sha256"],
        "source_reference": source_reference,
        "source_reference_sha256": source_reference["projection_sha256"],
        "run_invariant": invariant,
        "run_invariant_sha256": invariant_sha256,
        "root_checkpoint_fidelity": checkpoint_fidelity,
        "selected_checkpoint_fidelity": checkpoint_fidelity,
        "root_step1": root_step1,
        "selected_checkpoint_step1": copy.deepcopy(root_step1),
        "root_final": root_final,
        "selected_checkpoint_post_step": selected_post,
    }
    lineage["lineage_sha256"] = fidelity._sha256_json(lineage)
    monkeypatch.setattr(
        fidelity.exact_peft_kl_post_step,
        "validate_post_step_reference_identity",
        lambda value, **_kwargs: copy.deepcopy(value),
    )
    return lineage, fixed, runtime


def test_live_identity_is_path_free_and_relocatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _verified_receipt()
    receipt_bytes = b'{"frozen":"receipt"}\n'
    roots = [tmp_path / "first" / "checkpoint-7", tmp_path / "moved" / "checkpoint-7"]
    for root in roots:
        root.mkdir(parents=True)
        path = root / fidelity.RECEIPT_FILENAME
        path.write_bytes(receipt_bytes)
        path.chmod(0o440)
    monkeypatch.setattr(fidelity, "verify_checkpoint_fidelity", lambda _root: receipt)

    first = fidelity.publication_fidelity_identity(roots[0])
    second = fidelity.publication_fidelity_identity(roots[1])

    assert first == second == fidelity.validate_publication_fidelity_identity(first)
    assert first["receipt_file_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    encoded = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert set(first) == fidelity.IDENTITY_KEYS


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered", "boolean_schema"])
def test_embedded_identity_fails_closed_on_schema_or_self_hash_drift(
    mutation: str,
) -> None:
    identity = fidelity._identity_from_verified_receipt(
        _verified_receipt(),
        receipt_file_sha256="5" * 64,
    )
    if mutation == "missing":
        identity.pop("adapter_weights_sha256")
    elif mutation == "extra":
        identity["checkpoint_path"] = "/must/not/be/embedded"
    elif mutation == "boolean_schema":
        identity["schema_version"] = True
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    else:
        identity["receipt_sha256"] = "9" * 64
    with pytest.raises(fidelity.PublicationFidelityError):
        fidelity.validate_publication_fidelity_identity(identity)


def test_missing_publication_invariant_is_never_publication_eligible() -> None:
    receipt = _verified_receipt()
    receipt["publication_invariant"] = None
    with pytest.raises(
        fidelity.PublicationFidelityError,
        match="lacks the exact Qwen2.5-7B publication invariant",
    ):
        fidelity._identity_from_verified_receipt(
            receipt,
            receipt_file_sha256="5" * 64,
        )


def test_receipt_path_must_be_a_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-9"
    (checkpoint / fidelity.RECEIPT_FILENAME).mkdir(parents=True)
    monkeypatch.setattr(
        fidelity,
        "verify_checkpoint_fidelity",
        lambda _root: _verified_receipt(),
    )
    with pytest.raises(fidelity.PublicationFidelityError, match="regular file"):
        fidelity.publication_fidelity_identity(checkpoint)


def test_fixed_kl_identity_is_path_free_and_closes_source_fidelity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source_identity = fidelity._identity_from_verified_receipt(
        _verified_receipt(), receipt_file_sha256="5" * 64
    )
    identity = _fixed_identity(source)

    assert identity == fidelity.validate_fixed_kl_reference_identity(identity)
    assert identity["source_checkpoint_fidelity"] == source_identity
    assert identity["runtime_protocol"] == fidelity.exact_peft_kl.PUBLICATION_PROTOCOL
    assert identity["parameterization"] == "continue_same_bias_all_sft_adapter"
    assert str(tmp_path) not in json.dumps(identity, sort_keys=True)


@pytest.mark.parametrize("mutation", ["missing", "development", "tampered", "source", "dtype"])
def test_fixed_kl_embedded_identity_fails_closed(
    mutation: str,
    tmp_path: Path,
) -> None:
    identity = _fixed_identity(tmp_path)
    if mutation == "missing":
        identity.pop("adapter_tree_sha256")
    elif mutation == "development":
        identity["runtime_protocol"] = fidelity.exact_peft_kl.DEVELOPMENT_PROTOCOL
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    elif mutation == "source":
        identity["source_checkpoint_fidelity"]["receipt_sha256"] = "0" * 64
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    elif mutation == "dtype":
        identity["saved_lora_dtype"] = "BF16"
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    else:
        identity["verifier_implementation_sha256"] = "0" * 64
    with pytest.raises(fidelity.PublicationFidelityError):
        fidelity.validate_fixed_kl_reference_identity(identity)


@pytest.mark.parametrize("mutation", ["missing", "development", "tampered", "source", "settings"])
def test_exact_peft_runtime_embedded_identity_fails_closed(
    mutation: str,
    tmp_path: Path,
) -> None:
    identity = _runtime_identity(tmp_path)
    if mutation == "missing":
        identity.pop("distributed_sha256")
    elif mutation == "development":
        identity["runtime_protocol"] = fidelity.exact_peft_kl.DEVELOPMENT_PROTOCOL
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    elif mutation == "source":
        identity["fixed_kl_reference"]["receipt_sha256"] = "0" * 64
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    elif mutation == "settings":
        identity["training_use_vllm"] = True
        identity["identity_sha256"] = fidelity._sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
    else:
        identity["runtime_pair_sha256"] = "0" * 64
    with pytest.raises(fidelity.PublicationFidelityError):
        fidelity.validate_exact_peft_kl_runtime_identity(identity)


def test_exact_peft_runtime_identity_is_path_free_and_binds_source(
    tmp_path: Path,
) -> None:
    identity = _runtime_identity(tmp_path)
    assert identity == fidelity.validate_exact_peft_kl_runtime_identity(identity)
    assert identity["fixed_kl_reference"] == _fixed_identity(tmp_path)
    assert str(tmp_path) not in json.dumps(identity, sort_keys=True)


def test_embedded_runtime_receipt_snapshot_is_exact_and_relocatable(
    tmp_path: Path,
) -> None:
    receipt = _runtime_receipt(tmp_path)
    runtime = _runtime_identity(tmp_path)
    encoded = (
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    )
    assert fidelity.exact_peft_kl_runtime_identity_from_receipt_payload(
        receipt,
        receipt_file_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_runtime_identity=runtime,
    ) == fidelity.validate_exact_peft_kl_runtime_identity(runtime)

    snapshot = {
        "schema_version": fidelity.EXACT_PEFT_KL_RECEIPT_SNAPSHOT_SCHEMA_VERSION,
        "protocol": fidelity.EXACT_PEFT_KL_RECEIPT_SNAPSHOT_PROTOCOL,
        "status": "publication_root_and_checkpoint_verified",
        "receipt_filename": fidelity.exact_peft_kl.RECEIPT_FILENAME,
        "receipt_size_bytes": len(encoded),
        "receipt_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "receipt_payload": receipt,
        "receipt_payload_sha256": fidelity._sha256_json(receipt),
        "runtime_identity": runtime,
        "verified_source_roles": ["run_root", "selected_checkpoint"],
    }
    snapshot["snapshot_sha256"] = fidelity._sha256_json(snapshot)
    assert fidelity.validate_exact_peft_kl_receipt_snapshot(
        snapshot,
        expected_runtime_identity=runtime,
    ) == json.loads(json.dumps(snapshot, sort_keys=True))

    tampered = json.loads(json.dumps(snapshot))
    tampered["receipt_payload"]["runtime_pair"]["independent_storage"] = False
    unsigned_receipt = dict(tampered["receipt_payload"])
    unsigned_receipt.pop("receipt_sha256")
    tampered["receipt_payload"]["receipt_sha256"] = fidelity._sha256_json(unsigned_receipt)
    tampered_encoded = (
        json.dumps(
            tampered["receipt_payload"],
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    tampered["receipt_size_bytes"] = len(tampered_encoded)
    tampered["receipt_file_sha256"] = hashlib.sha256(tampered_encoded).hexdigest()
    tampered["receipt_payload_sha256"] = fidelity._sha256_json(tampered["receipt_payload"])
    tampered.pop("snapshot_sha256")
    tampered["snapshot_sha256"] = fidelity._sha256_json(tampered)
    with pytest.raises(fidelity.PublicationFidelityError, match="exactness invariant"):
        fidelity.validate_exact_peft_kl_receipt_snapshot(
            tampered,
            expected_runtime_identity=runtime,
        )


def test_embedded_runtime_receipt_rejects_training_probe_dtype_drift(
    tmp_path: Path,
) -> None:
    receipt = _runtime_receipt(tmp_path)
    receipt["training_contract"]["dtype"] = "float16"
    receipt["receipt_sha256"] = fidelity._sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    encoded = (
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    )

    with pytest.raises(fidelity.PublicationFidelityError, match="runtime probe dtype"):
        fidelity.exact_peft_kl_runtime_identity_from_receipt_payload(
            receipt,
            receipt_file_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_runtime_identity=_runtime_identity(tmp_path),
        )


def test_post_step_lineage_binds_runtime_contract_and_selected_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage, fixed, runtime = _post_step_lineage(tmp_path, monkeypatch)

    assert fidelity.validate_exact_peft_kl_post_step_lineage(
        lineage,
        expected_fixed_kl_reference=fixed,
        expected_runtime_identity=runtime,
        expected_run_contract_file_sha256="6" * 64,
        expected_terminal_global_step=400,
        expected_selected_checkpoint_step=400,
    ) == json.loads(json.dumps(lineage, sort_keys=True))


@pytest.mark.parametrize("mutation", ["missing", "tampered", "wrong_step"])
def test_post_step_lineage_fails_closed_on_missing_tampered_or_wrong_step(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage, fixed, runtime = _post_step_lineage(tmp_path, monkeypatch)
    if mutation == "missing":
        lineage.pop("root_final")
    elif mutation == "tampered":
        lineage["training_run_contract_file_sha256"] = "0" * 64
    else:
        lineage["selected_checkpoint_step"] = 399
        lineage["selected_checkpoint_is_terminal"] = False
        lineage["lineage_sha256"] = fidelity._sha256_json(
            {key: value for key, value in lineage.items() if key != "lineage_sha256"}
        )
    with pytest.raises(fidelity.PublicationFidelityError):
        fidelity.validate_exact_peft_kl_post_step_lineage(
            lineage,
            expected_fixed_kl_reference=fixed,
            expected_runtime_identity=runtime,
            expected_run_contract_file_sha256="6" * 64,
            expected_terminal_global_step=400,
            expected_selected_checkpoint_step=400,
        )
