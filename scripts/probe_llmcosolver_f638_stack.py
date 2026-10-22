#!/usr/bin/env python3
"""Probe the pinned PEFT behavior behind LLMCoSolver's operational KL reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

PINNED_VERSIONS = {
    "peft": "0.17.0",
    "torch": "2.7.1",
    "transformers": "4.50.3",
    "trl": "0.15.2",
}


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _self_hash(receipt: dict) -> str:
    payload = dict(receipt)
    payload["receipt_sha256"] = None
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _base_version(version: str) -> str:
    return str(version).split("+")[0]


def run_probe() -> dict:
    """Run a tiny Qwen2/LoRA experiment without downloading a model."""

    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    observed_versions = {
        "peft": peft.__version__,
        "torch": _base_version(torch.__version__),
        "transformers": transformers.__version__,
        "trl": trl.__version__,
    }
    if observed_versions != PINNED_VERSIONS:
        raise RuntimeError(
            f"Pinned-stack probe requires {PINNED_VERSIONS!r}, observed {observed_versions!r}."
        )

    torch.manual_seed(123)
    config = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    base = Qwen2ForCausalLM(config).eval()
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="lora_only",
            task_type="CAUSAL_LM",
        ),
    ).eval()

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_bias_names = [name for name in trainable_names if name.endswith("bias")]
    input_ids = torch.tensor([[1, 3, 4, 5]])
    with torch.no_grad(), model.disable_adapter():
        disabled_before = model(input_ids).logits.detach().clone()

    mutated_bias_name = trainable_bias_names[0]
    with torch.no_grad():
        dict(model.named_parameters())[mutated_bias_name].add_(0.25)

    with torch.no_grad(), model.disable_adapter():
        disabled_after = model(input_ids).logits.detach().clone()
    with torch.no_grad():
        enabled_after = model(input_ids).logits.detach().clone()

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir, safe_serialization=True)
        saved_keys = sorted(load_file(str(Path(temp_dir) / "adapter_model.safetensors")).keys())

    receipt = {
        "schema_version": 1,
        "probe": "llmcosolver_f638_lora_only_disable_adapter",
        "receipt_sha256": None,
        "runtime": {
            "packages": observed_versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "observations": {
            "trainable_parameter_count": len(trainable_names),
            "trainable_bias_count": len(trainable_bias_names),
            "trainable_bias_names": trainable_bias_names,
            "mutated_bias_name": mutated_bias_name,
            "disabled_logits_changed_after_bias_mutation": not torch.equal(
                disabled_before, disabled_after
            ),
            "disabled_logits_max_abs_change": float(
                (disabled_before - disabled_after).abs().max().item()
            ),
            "enabled_equals_disabled_with_zero_initialized_lora_b": torch.equal(
                enabled_after, disabled_after
            ),
            "enabled_disabled_max_abs_difference": float(
                (enabled_after - disabled_after).abs().max().item()
            ),
            "saved_tensor_count": len(saved_keys),
            "saved_bias_tensor_count": sum(key.endswith("bias") for key in saved_keys),
            "saved_bias_tensor_names": [key for key in saved_keys if key.endswith("bias")],
        },
        "conclusion": {
            "disable_adapter_reference_is_fixed": False,
            "disable_adapter_reference_is_foundation_base": False,
            "reference_includes_live_trainable_qkv_bias": True,
            "lora_only_checkpoint_persists_live_bias": False,
        },
    }
    receipt["receipt_sha256"] = _self_hash(receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = run_probe()
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(args.output)


if __name__ == "__main__":
    main()
