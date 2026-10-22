from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.models import checkpoint_fidelity as fidelity


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _closure(
    records: list[dict[str, object]], *, adapter_name: str = "default"
) -> dict[str, object]:
    records = sorted(records, key=lambda record: str(record["key"]))
    return {
        "schema_version": 1,
        "protocol": fidelity.CLOSURE_PROTOCOL,
        "status": "exact",
        "adapter_name": adapter_name,
        "tensor_count": len(records),
        "parameter_count": sum(int(record["numel"]) for record in records),
        "tensor_inventory_sha256": _sha256_json(records),
        "tensors": records,
    }


def _write_adapter(
    root: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    *,
    config: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    adapter_config = config or {
        "peft_type": "LORA",
        "bias": "none",
        "r": 2,
        "target_modules": ["q_proj"],
    }
    (root / fidelity.ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(adapter_config, sort_keys=True),
        encoding="utf-8",
    )
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for key, (dtype, shape, value) in sorted(tensors.items()):
        start = len(payload)
        payload.extend(value)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (root / fidelity.ADAPTER_WEIGHTS_FILENAME).write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    )
    return root


def _record(key: str, dtype: str, shape: list[int], value: bytes) -> dict[str, object]:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return {
        "key": key,
        "shape": shape,
        "dtype": dtype,
        "numel": numel,
        "value_sha256": hashlib.sha256(value).hexdigest(),
    }


def _small_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tensors = {
        "base_model.model.layer.lora_A.weight": (
            "F32",
            [1, 2],
            struct.pack("<ff", 1.25, -2.5),
        ),
        "base_model.model.layer.base_layer.bias": (
            "F32",
            [1],
            struct.pack("<f", 0.75),
        ),
    }
    checkpoint = _write_adapter(tmp_path / "checkpoint", tensors)
    records = [_record(key, *spec) for key, spec in tensors.items()]
    return checkpoint, _closure(records)


@pytest.mark.parametrize(
    ("key", "adapter_name", "expected"),
    [
        (
            "base_model.model.q_proj.lora_A.default.weight",
            "default",
            "base_model.model.q_proj.lora_A.weight",
        ),
        (
            "base_model.model.embed.lora_embedding_A.experiment",
            "experiment",
            "base_model.model.embed.lora_embedding_A",
        ),
        (
            "base_model.model.q_proj.base_layer.bias",
            "default",
            "base_model.model.q_proj.base_layer.bias",
        ),
        (
            "base_model.default_module.q_proj.weight",
            "default",
            "base_model.default_module.q_proj.weight",
        ),
    ],
)
def test_canonicalize_peft_adapter_name_syntax(
    key: str,
    adapter_name: str,
    expected: str,
) -> None:
    assert fidelity.canonicalize_peft_adapter_key(key, adapter_name) == expected


def test_dependency_light_receipt_round_trip_is_exact_and_read_only(tmp_path: Path) -> None:
    checkpoint, closure = _small_checkpoint(tmp_path)

    receipt = fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)
    verified = fidelity.verify_checkpoint_fidelity(checkpoint)
    reopened = fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)

    assert receipt == verified == reopened
    assert receipt["status"] == "exact_trainable_policy_persisted"
    assert receipt["trainable_closure"]["tensor_count"] == 2
    assert receipt["artifacts"]["adapter_weights"]["tensor_count"] == 2
    assert stat.S_IMODE((checkpoint / fidelity.RECEIPT_FILENAME).stat().st_mode) == 0o440


def test_build_rejects_missing_serialized_trainable_tensor(tmp_path: Path) -> None:
    checkpoint, closure = _small_checkpoint(tmp_path)
    closure["tensors"] = closure["tensors"][:-1]
    closure["tensor_count"] = 1
    closure["parameter_count"] = closure["tensors"][0]["numel"]
    closure["tensor_inventory_sha256"] = _sha256_json(closure["tensors"])

    with pytest.raises(fidelity.CheckpointFidelityError, match="fidelity mismatch"):
        fidelity.build_checkpoint_fidelity_receipt(checkpoint, closure)


def test_verifier_rejects_safetensors_value_tamper(tmp_path: Path) -> None:
    checkpoint, closure = _small_checkpoint(tmp_path)
    fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)
    weights = checkpoint / fidelity.ADAPTER_WEIGHTS_FILENAME
    with weights.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 1]))

    with pytest.raises(fidelity.CheckpointFidelityError, match="fidelity mismatch"):
        fidelity.verify_checkpoint_fidelity(checkpoint)


def test_verifier_rejects_adapter_config_byte_tamper(tmp_path: Path) -> None:
    checkpoint, closure = _small_checkpoint(tmp_path)
    fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)
    config = checkpoint / fidelity.ADAPTER_CONFIG_FILENAME
    parsed = json.loads(config.read_text(encoding="utf-8"))
    parsed["new_field"] = "tampered"
    config.write_text(json.dumps(parsed, sort_keys=True), encoding="utf-8")

    with pytest.raises(fidelity.CheckpointFidelityError, match="live checkpoint evidence"):
        fidelity.verify_checkpoint_fidelity(checkpoint)


def test_verifier_rejects_receipt_self_hash_tamper(tmp_path: Path) -> None:
    checkpoint, closure = _small_checkpoint(tmp_path)
    fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)
    receipt_path = checkpoint / fidelity.RECEIPT_FILENAME
    receipt_path.chmod(0o640)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "tampered"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o440)

    with pytest.raises(fidelity.CheckpointFidelityError, match="self-hash mismatch"):
        fidelity.verify_checkpoint_fidelity(checkpoint)


def test_verifier_fails_on_missing_receipt_or_symlink(tmp_path: Path) -> None:
    checkpoint, _ = _small_checkpoint(tmp_path)
    with pytest.raises(fidelity.CheckpointFidelityError, match="required file"):
        fidelity.verify_checkpoint_fidelity(checkpoint)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (checkpoint / fidelity.RECEIPT_FILENAME).symlink_to(outside)
    with pytest.raises(fidelity.CheckpointFidelityError, match="regular file"):
        fidelity.verify_checkpoint_fidelity(checkpoint)


def _qwen_inventory() -> list[dict[str, object]]:
    records = []
    for key, shape_tuple in fidelity.expected_qwen25_7b_publication_shapes().items():
        shape = list(shape_tuple)
        numel = 1
        for dimension in shape:
            numel *= dimension
        records.append(
            {
                "key": key,
                "shape": shape,
                "dtype": "BF16",
                "numel": numel,
                "value_sha256": "0" * 64,
            }
        )
    return sorted(records, key=lambda record: str(record["key"]))


def test_qwen_publication_invariant_has_exact_scope_and_parameter_count() -> None:
    closure = _closure(_qwen_inventory())
    adapter_config = SimpleNamespace(
        peft_type="LORA",
        bias="all",
        r=64,
        target_modules=set(fidelity.QWEN_LORA_TARGETS),
    )
    model_config = SimpleNamespace(
        model_type="qwen2",
        hidden_size=3_584,
        intermediate_size=18_944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
    )
    model = SimpleNamespace(peft_config={"default": adapter_config}, config=model_config)

    report = fidelity.assert_qwen25_7b_publication_invariant(model, closure)

    assert report["bias_policy"] == "all"
    assert report["target_modules"] == sorted(fidelity.QWEN_LORA_TARGETS)
    assert report["lora_tensor_count"] == 392
    assert report["bias_tensor_count"] == 84
    assert report["trainable_tensor_count"] == 476
    assert report["trainable_parameter_count"] == 161_609_728


def test_qwen_publication_invariant_rejects_lora_only() -> None:
    closure = _closure(_qwen_inventory())
    adapter_config = SimpleNamespace(
        peft_type="LORA",
        bias="lora_only",
        r=64,
        target_modules=set(fidelity.QWEN_LORA_TARGETS),
    )
    model_config = SimpleNamespace(
        model_type="qwen2",
        hidden_size=3_584,
        intermediate_size=18_944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
    )
    model = SimpleNamespace(peft_config={"default": adapter_config}, config=model_config)

    with pytest.raises(fidelity.CheckpointFidelityError, match="bias='all'"):
        fidelity.assert_qwen25_7b_publication_invariant(model, closure)


def test_qwen_bias_trainability_is_explicitly_restored_after_unsloth() -> None:
    class Parameter:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.requires_grad = False

        def requires_grad_(self, enabled: bool):
            self.requires_grad = enabled
            return self

    expected = {
        key: shape
        for key, shape in fidelity.expected_qwen25_7b_publication_shapes().items()
        if key.endswith(".base_layer.bias")
    }
    parameters = {key: Parameter(shape) for key, shape in expected.items()}
    model = SimpleNamespace(named_parameters=lambda: parameters.items())

    evidence = fidelity.enable_qwen25_targeted_bias_training(model)

    assert evidence["status"] == "exact"
    assert evidence["bias_tensor_count"] == 84
    assert evidence["bias_parameter_count"] == 129_024
    assert all(parameter.requires_grad for parameter in parameters.values())


def test_qwen_bias_trainability_restore_fails_on_incomplete_scope() -> None:
    class Parameter:
        shape = (3_584,)
        requires_grad = False

        def requires_grad_(self, enabled: bool):
            self.requires_grad = enabled
            return self

    model = SimpleNamespace(
        named_parameters=lambda: [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.base_layer.bias",
                Parameter(),
            )
        ]
    )
    with pytest.raises(fidelity.CheckpointFidelityError, match="cannot restore exact"):
        fidelity.enable_qwen25_targeted_bias_training(model)


_HAS_FULL_ML_STACK = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "peft", "safetensors")
)


@pytest.mark.skipif(not _HAS_FULL_ML_STACK, reason="tiny Qwen fidelity test needs the ML stack")
def test_tiny_qwen_bias_all_round_trips_and_lora_only_fails(tmp_path: Path) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    def tiny_qwen(bias: str):
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        base = Qwen2ForCausalLM(config)
        return get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=4,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias=bias,
                task_type="CAUSAL_LM",
            ),
        )

    safe_model = tiny_qwen("all")
    closure = fidelity.audit_trainable_state_closure(safe_model)
    assert closure["tensor_count"] == 9
    checkpoint = tmp_path / "tiny-safe"
    safe_model.save_pretrained(checkpoint, safe_serialization=True)
    receipt = fidelity.write_checkpoint_fidelity_receipt(checkpoint, closure)
    assert fidelity.verify_checkpoint_fidelity(checkpoint) == receipt

    unsafe_model = tiny_qwen("lora_only")
    with pytest.raises(
        fidelity.CheckpointFidelityError,
        match="missing_from_peft",
    ):
        fidelity.audit_trainable_state_closure(unsafe_model)
