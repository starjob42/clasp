from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_lora_persistence",
    ROOT / "scripts" / "audit_lora_persistence.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _base_model(path: Path, *, extra_bias: bool = False) -> Path:
    path.mkdir()
    config = {
        "model_type": "qwen2",
        "hidden_size": 3_584,
        "intermediate_size": 18_944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shard = path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"frozen-test-shard")
    names = set(audit.expected_base_bias_shapes()) | {"model.embed_tokens.weight"}
    if extra_bias:
        names.add("model.norm.bias")
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {name: shard.name for name in sorted(names)},
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    return path


def _adapter(path: Path, *, bias_policy: str = "all", omit_bias: bool = False) -> Path:
    path.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": bias_policy,
        "target_modules": sorted(audit.JIANG_LORA_MODULES),
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        **audit.expected_adapter_bias_shapes(),
        **audit.expected_adapter_lora_shapes(),
    }
    if omit_bias:
        tensors.pop(sorted(audit.expected_adapter_bias_shapes())[0])
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for key, shape in sorted(tensors.items()):
        tensor_bytes = 2
        for dimension in shape:
            tensor_bytes *= dimension
        header[key] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + tensor_bytes],
        }
        offset += tensor_bytes
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    artifact = path / "adapter_model.safetensors"
    with artifact.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded_header)))
        handle.write(encoded_header)
        handle.seek(8 + len(encoded_header) + offset - 1)
        handle.write(b"\0")
    return path


def test_exact_qwen_bias_scope_and_adapter_inventory(tmp_path: Path) -> None:
    base = _base_model(tmp_path / "base")
    adapter = _adapter(tmp_path / "adapter")

    base_receipt = audit.audit_qwen25_base_bias_scope(base)
    adapter_receipt = audit.audit_adapter_persistence(adapter)

    assert base_receipt["policy"] == "all"
    assert base_receipt["bias_tensor_count"] == 84
    assert base_receipt["bias_parameter_count"] == 129_024
    assert adapter_receipt["bias_tensor_count"] == 84
    assert adapter_receipt["bias_parameter_count"] == 129_024
    assert adapter_receipt["lora_tensor_count"] == 392
    assert adapter_receipt["lora_parameter_count"] == 161_480_704
    assert adapter_receipt["trainable_tensor_count"] == 476
    assert adapter_receipt["trainable_parameter_count"] == 161_609_728
    assert len(adapter_receipt["bias_value_receipt"]) == 84


def test_bias_all_is_rejected_when_base_has_any_untargeted_bias(tmp_path: Path) -> None:
    base = _base_model(tmp_path / "base", extra_bias=True)

    with pytest.raises(ValueError, match="not scope-equivalent"):
        audit.audit_qwen25_base_bias_scope(base)


@pytest.mark.parametrize(
    ("bias_policy", "omit_bias", "message"),
    [
        ("lora_only", False, "adapter_config.bias"),
        ("all", True, "tensor inventory mismatch"),
    ],
)
def test_publication_adapter_rejects_legacy_policy_or_missing_saved_bias(
    tmp_path: Path,
    bias_policy: str,
    omit_bias: bool,
    message: str,
) -> None:
    adapter = _adapter(
        tmp_path / "adapter",
        bias_policy=bias_policy,
        omit_bias=omit_bias,
    )

    with pytest.raises(ValueError, match=message):
        audit.audit_adapter_persistence(adapter)


def test_truncated_adapter_payload_fails_closed(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "adapter")
    artifact = adapter / "adapter_model.safetensors"
    with artifact.open("r+b") as handle:
        handle.truncate(artifact.stat().st_size - 1)

    with pytest.raises(ValueError, match="payload extent|trailing bytes"):
        audit.audit_adapter_persistence(adapter)


def test_create_once_receipt_reopens_identically_and_detects_drift(tmp_path: Path) -> None:
    base = _base_model(tmp_path / "base")
    adapter = _adapter(tmp_path / "adapter")
    receipt = tmp_path / "adapter_persistence_receipt.json"
    payload = audit.build_audit(base, adapter)

    audit._write_or_verify_receipt(receipt, payload)
    audit._write_or_verify_receipt(receipt, audit.build_audit(base, adapter))
    assert receipt.stat().st_mode & 0o222 == 0

    receipt.chmod(0o640)
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    stored["status"] = "tampered"
    receipt.write_text(json.dumps(stored), encoding="utf-8")
    receipt.chmod(0o440)
    with pytest.raises(ValueError, match="differs from live artifact"):
        audit._write_or_verify_receipt(receipt, audit.build_audit(base, adapter))


def test_cli_output_is_idempotent_for_complete_worker_no_op(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _base_model(tmp_path / "base")
    adapter = _adapter(tmp_path / "adapter")
    receipt = tmp_path / "adapter_persistence_receipt.json"
    argv = [
        "--base-model-path",
        str(base),
        "--adapter-path",
        str(adapter),
        "--output",
        str(receipt),
    ]

    assert audit.main(argv) == 0
    first_bytes = receipt.read_bytes()
    assert audit.main(argv) == 0
    capsys.readouterr()

    assert receipt.read_bytes() == first_bytes
    assert receipt.stat().st_mode & 0o222 == 0
    validated = audit.validate_audit_receipt(
        receipt,
        base_model_path=base,
        adapter_path=adapter,
    )
    assert validated["audit_sha256"] == audit.build_audit(base, adapter)["audit_sha256"]


def test_future_publication_launchers_use_persistence_safe_policy() -> None:
    base_worker = (ROOT / "scripts" / "train_sft500k_base_specialist_h100.slurm").read_text(
        encoding="utf-8"
    )
    rl_worker = (ROOT / "scripts" / "train_clasp_sota_a100.slurm").read_text(
        encoding="utf-8"
    )
    assert "export CLASP_LORA_BIAS=all" in base_worker
    assert "adapter_persistence_receipt.json" in base_worker
    assert "export CLASP_LORA_BIAS=all" in rl_worker
    assert 'lora_bias=all' in rl_worker
    assert "export CLASP_LORA_BIAS=lora_only" not in base_worker
    assert "export CLASP_LORA_BIAS=lora_only" not in rl_worker


def test_sparse_fixture_does_not_consume_adapter_logical_size(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "adapter")
    artifact = adapter / "adapter_model.safetensors"
    blocks = artifact.stat().st_blocks * 512
    assert artifact.stat().st_size > 300_000_000
    if os.name == "posix":
        assert blocks < artifact.stat().st_size // 10
