from __future__ import annotations

import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import LoraConfig, get_peft_model  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_legacy_saved_policy",
    ROOT / "scripts" / "convert_legacy_saved_policy.py",
)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)

RELOCATION_SPEC = importlib.util.spec_from_file_location(
    "relocate_development_adapter_base",
    ROOT / "scripts" / "relocate_development_adapter_base.py",
)
assert RELOCATION_SPEC is not None and RELOCATION_SPEC.loader is not None
relocator = importlib.util.module_from_spec(RELOCATION_SPEC)
sys.modules[RELOCATION_SPEC.name] = relocator
RELOCATION_SPEC.loader.exec_module(relocator)


def _tiny_qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _create_tiny_legacy(root: Path) -> tuple[Path, Path]:
    base = root / "base"
    legacy = root / "legacy"
    torch.manual_seed(20260814)
    Qwen2ForCausalLM(_tiny_qwen_config()).save_pretrained(base, safe_serialization=True)
    model = Qwen2ForCausalLM.from_pretrained(base, local_files_only=True)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="lora_only",
        ),
    )
    with torch.no_grad():
        for index, (name, parameter) in enumerate(peft_model.named_parameters()):
            if ".lora_" in name:
                values = torch.arange(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape(parameter.shape)
                parameter.copy_((values + index + 1) / (parameter.numel() + index + 3))
            elif name.endswith(".base_layer.bias"):
                # These values represent the historical terminal policy and are
                # intentionally dropped by the affected PEFT serializer.
                parameter.add_(0.25 + index / 1000)
    peft_model.save_pretrained(legacy, safe_serialization=True)
    with safe_open(legacy / "adapter_model.safetensors", framework="pt") as handle:
        keys = list(handle.keys())
    assert keys
    assert all(not key.endswith("bias") for key in keys)
    return base, legacy


@pytest.fixture()
def tiny_legacy(tmp_path: Path) -> tuple[Path, Path]:
    return _create_tiny_legacy(tmp_path)


def _thaw_conversion(path: Path) -> None:
    path.chmod(0o700)
    for child in path.rglob("*"):
        if child.is_dir():
            child.chmod(0o700)
        else:
            child.chmod(0o600)


def _freeze_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o550 if child.is_dir() else 0o440)
    path.chmod(0o550)


def test_tiny_qwen_conversion_proves_saved_policy_exactly(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    output = tmp_path / "converted"
    legacy_before = converter.build_tree_manifest(legacy)

    receipt = converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=output,
        device="cpu",
        dtype="float32",
    )

    assert receipt["status"] == "converted_saved_policy_verified"
    assert receipt["development_only"] is True
    assert receipt["recovers_historical_terminal_biases"] is False
    assert receipt["converted_saved_policy_fidelity"] is True
    assert receipt["terminal_sft_policy_fidelity"] is False
    assert receipt["publication_eligible"] is False
    assert receipt["sota_eligible"] is False
    assert receipt["publication_sota_eligible"] is False
    assert receipt["verification"]["canonical_adapter_state_bitwise_equal"] is True
    assert receipt["verification"]["deterministic_logits_bitwise_equal"] is True
    assert receipt["verification"]["deterministic_log_probabilities_bitwise_equal"] is True
    assert len(receipt["converted_output"]["serialized_bias_keys"]) == 6
    assert json.loads((output / "adapter_config.json").read_text())["bias"] == "all"
    assert converter.build_tree_manifest(legacy) == legacy_before
    assert stat.S_IMODE(output.stat().st_mode) & 0o222 == 0
    assert all(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0 for path in output.rglob("*"))

    reopened = converter.verify_conversion_receipt(
        output,
        base_model_path=base,
        legacy_adapter_path=legacy,
    )
    assert reopened == receipt


def test_conversion_rejects_existing_output_before_model_load(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    output = tmp_path / "converted"
    output.mkdir()
    sentinel = output / "owned_by_user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        converter.convert_legacy_saved_policy(
            base_model_path=base,
            legacy_adapter_path=legacy,
            output_dir=output,
            dtype="float32",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_conversion_rejects_base_path_mismatch(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    _base, legacy = tiny_legacy
    wrong_base = tmp_path / "wrong_base"
    torch.manual_seed(99)
    Qwen2ForCausalLM(_tiny_qwen_config()).save_pretrained(
        wrong_base,
        safe_serialization=True,
    )
    output = tmp_path / "converted"

    with pytest.raises(converter.ConversionError, match="base mismatch"):
        converter.convert_legacy_saved_policy(
            base_model_path=wrong_base,
            legacy_adapter_path=legacy,
            output_dir=output,
            dtype="float32",
        )

    assert not output.exists()


def test_verifier_rejects_missing_converted_bias(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    output = tmp_path / "converted"
    receipt = converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=output,
        dtype="float32",
    )
    _thaw_conversion(output)
    artifact = output / "adapter_model.safetensors"
    tensors = load_file(artifact)
    missing = receipt["converted_output"]["serialized_bias_keys"][0]
    tensors.pop(missing)
    save_file(tensors, artifact, metadata={"format": "pt"})
    for path in output.rglob("*"):
        path.chmod(0o550 if path.is_dir() else 0o440)
    output.chmod(0o550)

    with pytest.raises(converter.ConversionError, match="missing converted biases"):
        converter.verify_conversion_receipt(output)


def test_verifier_rejects_receipt_tamper(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    output = tmp_path / "converted"
    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=output,
        dtype="float32",
    )
    _thaw_conversion(output)
    receipt_path = output / converter.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["publication_eligible"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.chmod(0o440)
    output.chmod(0o550)

    with pytest.raises(converter.ConversionError, match="self-hash mismatch"):
        converter.verify_conversion_receipt(output)


def test_relocation_proves_config_only_move_to_immutable_equivalent_base(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    converted = tmp_path / "converted"
    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=converted,
        dtype="float32",
    )
    frozen_base = tmp_path / "frozen_base"
    shutil.copytree(base, frozen_base)
    _freeze_tree(frozen_base)
    relocated = tmp_path / "relocated"

    receipt = relocator.relocate_development_adapter_base(
        source_adapter_path=converted,
        frozen_base_model_path=frozen_base,
        output_dir=relocated,
        device="cpu",
        dtype="float32",
    )

    assert receipt["protocol"] == relocator.PROTOCOL
    assert receipt["development_only"] is True
    assert receipt["recovers_historical_terminal_biases"] is False
    assert receipt["publication_eligible"] is False
    assert receipt["sota_eligible"] is False
    assert receipt["verification"]["full_model_state_exact"] is True
    assert receipt["verification"]["functional_equivalence_exact"] is True
    source_config = json.loads((converted / "adapter_config.json").read_text())
    relocated_config = json.loads((relocated / "adapter_config.json").read_text())
    assert {
        key: value for key, value in relocated_config.items() if key != "base_model_name_or_path"
    } == {key: value for key, value in source_config.items() if key != "base_model_name_or_path"}
    assert relocated_config["base_model_name_or_path"] == str(frozen_base.resolve())
    assert (converted / "adapter_model.safetensors").read_bytes() == (
        relocated / "adapter_model.safetensors"
    ).read_bytes()
    assert stat.S_IMODE(relocated.stat().st_mode) & 0o222 == 0
    assert all(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0 for path in relocated.rglob("*"))
    assert relocator.verify_relocation_receipt(relocated) == receipt


def test_relocation_rejects_nonidentical_frozen_base_before_model_load(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    converted = tmp_path / "converted"
    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=converted,
        dtype="float32",
    )
    frozen_base = tmp_path / "wrong_frozen_base"
    shutil.copytree(base, frozen_base)
    config_path = frozen_base / "config.json"
    config = json.loads(config_path.read_text())
    config["vocab_size"] += 1
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    _freeze_tree(frozen_base)
    relocated = tmp_path / "relocated"

    with pytest.raises(relocator.RelocationError, match="not byte-identical"):
        relocator.relocate_development_adapter_base(
            source_adapter_path=converted,
            frozen_base_model_path=frozen_base,
            output_dir=relocated,
            device="cpu",
            dtype="float32",
        )

    assert not relocated.exists()


def test_relocation_is_create_once(
    tiny_legacy: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    base, legacy = tiny_legacy
    converted = tmp_path / "converted"
    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=converted,
        dtype="float32",
    )
    frozen_base = tmp_path / "frozen_base"
    shutil.copytree(base, frozen_base)
    _freeze_tree(frozen_base)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    sentinel = relocated / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        relocator.relocate_development_adapter_base(
            source_adapter_path=converted,
            frozen_base_model_path=frozen_base,
            output_dir=relocated,
            device="cpu",
            dtype="float32",
        )

    assert sentinel.read_text() == "keep"
