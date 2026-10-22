from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "experiments/sota_20260814/llmcosolver_f638_stack_probe.json"


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def test_pinned_stack_probe_receipt_is_self_hashed_and_exact_versioned():
    receipt = json.loads(RECEIPT.read_text())
    expected = receipt["receipt_sha256"]
    receipt["receipt_sha256"] = None
    assert hashlib.sha256(_canonical(receipt)).hexdigest() == expected
    assert receipt["runtime"]["packages"] == {
        "peft": "0.17.0",
        "torch": "2.7.1",
        "transformers": "4.50.3",
        "trl": "0.15.2",
    }


def test_probe_proves_disabled_adapter_reference_evolves_and_bias_is_not_saved():
    receipt = json.loads(RECEIPT.read_text())
    observations = receipt["observations"]
    assert observations["trainable_bias_count"] == 3
    assert all(
        component in " ".join(observations["trainable_bias_names"])
        for component in ("q_proj", "k_proj", "v_proj")
    )
    assert observations["disabled_logits_changed_after_bias_mutation"] is True
    assert observations["disabled_logits_max_abs_change"] > 0
    assert observations["saved_bias_tensor_count"] == 0
    assert receipt["conclusion"] == {
        "disable_adapter_reference_is_fixed": False,
        "disable_adapter_reference_is_foundation_base": False,
        "lora_only_checkpoint_persists_live_bias": False,
        "reference_includes_live_trainable_qkv_bias": True,
    }


def test_probe_script_refuses_to_mislabel_a_different_stack():
    source = (ROOT / "scripts/probe_llmcosolver_f638_stack.py").read_text()
    for package, version in {
        "peft": "0.17.0",
        "torch": "2.7.1",
        "transformers": "4.50.3",
        "trl": "0.15.2",
    }.items():
        assert f'"{package}": "{version}"' in source
    assert "observed_versions != PINNED_VERSIONS" in source
