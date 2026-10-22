from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_sft_reload_equivalence.py"
SPEC = importlib.util.spec_from_file_location("audit_sft_reload_equivalence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reload_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reload_audit
SPEC.loader.exec_module(reload_audit)


def _tensor(
    key: str,
    value: str = "1",
    *,
    shape: list[int] | None = None,
    dtype: str = "torch.bfloat16",
) -> dict[str, object]:
    dimensions = shape or [1]
    numel = 1
    for dimension in dimensions:
        numel *= dimension
    return {
        "key": key,
        "shape": dimensions,
        "dtype": dtype,
        "numel": numel,
        "value_sha256": value * 64,
    }


def _runtime_pair() -> dict[str, object]:
    tensors = [_tensor("weight")]
    functional = {
        "method": "three_fixed_probes_bitwise_runtime_dtype_full_distribution_v2",
        "runtime_dtype": "bfloat16",
        "autocast_enabled": True,
        "autocast_device_type": "cuda",
        "input_ids": [list(row) for row in reload_audit.FIXED_PROBES],
        "input_ids_sha256": reload_audit._sha256_json(reload_audit.FIXED_PROBES),
        "logits": _tensor("logits", "2", shape=[3, 6, 32]),
        "log_probabilities": _tensor(
            "log_probabilities",
            "3",
            shape=[3, 5, 32],
            dtype="torch.float32",
        ),
        "argmax": _tensor("argmax", "4", shape=[3, 6], dtype="torch.int64"),
        "bitwise_equal": True,
        "max_absolute_error": 0.0,
    }
    return {
        "method": "independent_same_base_same_adapter_exact_runtime_pair_v2",
        "model_class": "test.FakePeftModel",
        "active_adapter": "default",
        "model_config_sha256": "5" * 64,
        "model_objects_distinct": True,
        "independent_storage": True,
        "reference_trainable_tensor_count": 0,
        "policy_trainable_tensor_count": 1,
        "tensor_count": 1,
        "parameter_count": 1,
        "tensor_inventory_sha256": reload_audit._sha256_json(tensors),
        "tensors": tensors,
        "state_bitwise_equal": True,
        "prepared_runtime": {
            "trainer_ref_model_identity": True,
            "reference_unwrapped_identity": True,
            "policy_unwrapped_class": "test.FakePeftModel",
            "reference_prepared_class": "test.FakePeftModel",
            "reference_unwrapped_class": "test.FakePeftModel",
            "device_types": ["cuda"],
        },
        "functional_equivalence": functional,
    }


def test_runtime_pair_requires_bitwise_state_and_fixed_probe_distributions() -> None:
    pair = _runtime_pair()
    assert reload_audit._validate_runtime_pair(pair) == pair

    broken = _runtime_pair()
    broken["functional_equivalence"]["bitwise_equal"] = False
    with pytest.raises(ValueError, match="functional probe"):
        reload_audit._validate_runtime_pair(broken)


def test_build_receipt_rejects_reloaded_bias_or_lora_value_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    serialized = {
        "adapter_name": "default",
        "tensor_count": 1,
        "parameter_count": 1,
        "tensor_inventory_sha256": "6" * 64,
        "tensors": [_tensor("base_model.model.q_proj.base_layer.bias", "6")],
    }
    monkeypatch.setattr(
        reload_audit,
        "verify_checkpoint_fidelity",
        lambda _path: {"trainable_closure": serialized},
    )
    monkeypatch.setattr(
        reload_audit,
        "publication_fidelity_identity",
        lambda _path: {"identity_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        reload_audit,
        "inspect_checkpoint_adapter",
        lambda _path: {
            "adapter_config": {"base_model_name_or_path": str(base)},
            "artifacts": {
                "adapter_config": {"sha256": "8" * 64},
                "adapter_weights": {"sha256": "9" * 64},
            },
            "tensors": serialized["tensors"],
        },
    )
    reloaded = dict(serialized)
    reloaded["tensors"] = [_tensor("base_model.model.q_proj.base_layer.bias", "a")]
    with pytest.raises(ValueError, match="freshly reloaded trainable tensors differ"):
        reload_audit.build_receipt(
            base_model_path=base,
            adapter_path=adapter,
            reloaded_trainable_closure=reloaded,
            runtime_pair=_runtime_pair(),
        )


def test_compact_reload_identity_is_self_hashed() -> None:
    identity = {
        "schema_version": reload_audit.SCHEMA_VERSION,
        "protocol": reload_audit.PROTOCOL,
        "status": reload_audit.STATUS,
        "receipt_sha256": "1" * 64,
        "receipt_file_sha256": "2" * 64,
        "checkpoint_fidelity_identity_sha256": "3" * 64,
        "serialized_trainable_closure_sha256": "4" * 64,
        "runtime_pair_sha256": "5" * 64,
        "functional_equivalence_sha256": "6" * 64,
        "producer_sha256": "7" * 64,
    }
    identity["identity_sha256"] = reload_audit._sha256_json(identity)
    assert reload_audit.validate_reload_equivalence_identity(identity) == identity
    identity["runtime_pair_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="identity is invalid"):
        reload_audit.validate_reload_equivalence_identity(identity)


def test_reference_reload_is_explicitly_frozen_after_peft_bias_all_load() -> None:
    class FakeParameter:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

        def requires_grad_(self, value: bool) -> "FakeParameter":
            self.requires_grad = value
            return self

    class FakeReference:
        def __init__(self) -> None:
            self.parameters = {
                "base_model.model.q_proj.base_layer.bias": FakeParameter(True),
                "base_model.model.q_proj.lora_A.default.weight": FakeParameter(False),
            }

        def named_parameters(self):
            return self.parameters.items()

    reference = FakeReference()
    reload_audit._freeze_reference_parameters(reference)
    assert all(not parameter.requires_grad for parameter in reference.parameters.values())


def test_reload_receipt_pins_precision_preserving_adapter_load_path() -> None:
    source = (ROOT / "src" / "clasp" / "models" / "loading.py").read_text(
        encoding="utf-8"
    )
    adapter_load = source[source.index("return PeftModel.from_pretrained(") :]
    assert "autocast_adapter_dtype=True" in adapter_load
    assert "low_cpu_mem_usage=False" in adapter_load
    assert '"low_cpu_mem_usage": True' in source[: source.index("return PeftModel.from_pretrained(")]

    audit_source = SCRIPT.read_text(encoding="utf-8")
    assert audit_source.count('"adapter_low_cpu_mem_usage": False') == 2
