from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from clasp.models.exact_peft_kl_reference import (  # noqa: E402
    ExactPeftKLReferenceError,
    verify_development_relocation_authority,
)
from clasp.models.training_fidelity import (  # noqa: E402
    PublicationFidelityError,
    verify_development_relocation_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_script("test_legacy_policy_converter", "convert_legacy_saved_policy.py")
relocator = _load_script(
    "test_development_adapter_relocator",
    "relocate_development_adapter_base.py",
)


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


def _freeze_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o550 if child.is_dir() else 0o440)
    path.chmod(0o550)


def _thaw_tree(path: Path) -> None:
    path.chmod(0o700)
    for child in path.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)


def _create_relocation(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    base = tmp_path / "base"
    legacy = tmp_path / "legacy"
    converted = tmp_path / "converted"
    frozen = tmp_path / "frozen-base"
    relocated = tmp_path / "relocated"

    torch.manual_seed(20260814)
    Qwen2ForCausalLM(_tiny_qwen_config()).save_pretrained(base, safe_serialization=True)
    policy = get_peft_model(
        Qwen2ForCausalLM.from_pretrained(base, local_files_only=True),
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
        for index, (name, parameter) in enumerate(policy.named_parameters()):
            if ".lora_" in name:
                values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(
                    parameter.shape
                )
                parameter.copy_((values + index + 1) / (parameter.numel() + index + 3))
    policy.save_pretrained(legacy, safe_serialization=True)
    del policy

    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=converted,
        device="cpu",
        dtype="float32",
    )
    shutil.copytree(base, frozen, copy_function=shutil.copyfile)
    _freeze_tree(frozen)
    receipt = relocator.relocate_development_adapter_base(
        source_adapter_path=converted,
        frozen_base_model_path=frozen,
        output_dir=relocated,
        device="cpu",
        dtype="float32",
    )
    return converted, frozen, relocated, receipt


def test_tiny_qwen_relocation_is_exact_and_development_only(tmp_path: Path) -> None:
    source, frozen, relocated, receipt = _create_relocation(tmp_path)

    assert receipt["protocol"] == relocator.PROTOCOL
    assert receipt["status"] == relocator.STATUS
    assert receipt["claim_scope"] == "development_only"
    assert receipt["development_only"] is True
    assert receipt["publication_eligible"] is False
    assert receipt["sota_eligible"] is False
    assert receipt["verification"]["full_model_state_exact"] is True
    assert receipt["verification"]["functional_equivalence_exact"] is True
    assert receipt["transformation"]["changed_config_fields"] == ["base_model_name_or_path"]
    assert (
        receipt["upstream_conversion"]["recorded_base_manifest"]
        == receipt["frozen_base"]["manifest"]
    )
    assert json.loads((relocated / "adapter_config.json").read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ] == str(frozen.resolve())
    assert all(path.stat().st_mode & 0o222 == 0 for path in [relocated, *relocated.rglob("*")])
    assert (
        relocator.verify_relocation_receipt(
            relocated,
            source_adapter_path=source,
            frozen_base_model_path=frozen,
        )
        == receipt
    )
    assert verify_development_relocation_checkpoint(relocated, purpose="tiny relocation") == receipt


def test_verifier_rejects_resealed_forged_upstream_lineage(tmp_path: Path) -> None:
    _source, _frozen, relocated, _receipt = _create_relocation(tmp_path)
    _thaw_tree(relocated)
    receipt_path = relocated / relocator.RECEIPT_FILENAME
    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["upstream_conversion"]["authoritative_receipt_sha256"] = "0" * 64
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = relocator._sha256_json(forged)
    receipt_path.write_bytes(relocator._receipt_bytes(forged))
    _freeze_tree(relocated)

    with pytest.raises(relocator.RelocationError, match="upstream receipt identity mismatch"):
        relocator.verify_relocation_receipt(relocated)
    with pytest.raises(PublicationFidelityError, match="upstream bytes mismatch"):
        verify_development_relocation_checkpoint(relocated, purpose="forged relocation")


def test_verifier_rejects_changed_source_adapter(tmp_path: Path) -> None:
    source, _frozen, relocated, _receipt = _create_relocation(tmp_path)
    _thaw_tree(source)
    readme = source / "README.md"
    readme.write_bytes(readme.read_bytes() + b"\nchanged-after-relocation\n")
    _freeze_tree(source)

    with pytest.raises(relocator.RelocationError, match="source adapter bytes differ"):
        relocator.verify_relocation_receipt(relocated)
    with pytest.raises(PublicationFidelityError, match="upstream conversion bytes differ"):
        verify_development_relocation_checkpoint(relocated, purpose="changed source")


def test_verifier_rejects_resealed_transformation_and_inventory_forgery(
    tmp_path: Path,
) -> None:
    _source, _frozen, relocated, original = _create_relocation(tmp_path)
    receipt_path = relocated / relocator.RECEIPT_FILENAME
    cases = (
        (
            "transformation method",
            "transformation invariant",
            lambda value: value["transformation"].__setitem__("method", "forged"),
        ),
        (
            "changed fields",
            "transformation invariant",
            lambda value: value["transformation"].__setitem__("changed_config_fields", []),
        ),
        (
            "config-only flag",
            "transformation invariant",
            lambda value: value["transformation"].__setitem__("config_only_delta", False),
        ),
        (
            "non-config equality flag",
            "transformation invariant",
            lambda value: value["transformation"].__setitem__(
                "non_config_files_bitwise_identical", False
            ),
        ),
        (
            "verification method",
            "verification method mismatch",
            lambda value: value["verification"].__setitem__("method", "forged"),
        ),
        (
            "truncated unchanged payload",
            "unchanged payload inventory is incomplete",
            lambda value: value["transformation"].__setitem__(
                "unchanged_file_paths",
                value["transformation"]["unchanged_file_paths"][:-1],
            ),
        ),
        (
            "extra unrecognized evidence",
            "top-level schema mismatch",
            lambda value: value.__setitem__("unrecognized_evidence", True),
        ),
    )
    for _label, message, mutate in cases:
        forged = copy.deepcopy(original)
        mutate(forged)
        forged.pop("receipt_sha256")
        forged["receipt_sha256"] = relocator._sha256_json(forged)
        _thaw_tree(relocated)
        receipt_path.write_bytes(relocator._receipt_bytes(forged))
        _freeze_tree(relocated)
        with pytest.raises(relocator.RelocationError, match=message):
            relocator.verify_relocation_receipt(relocated)
        with pytest.raises(ExactPeftKLReferenceError, match="trusted development relocation"):
            verify_development_relocation_authority(
                relocated,
                authoritative_receipt=forged,
            )
        with pytest.raises(PublicationFidelityError, match="strict relocation authority"):
            verify_development_relocation_checkpoint(relocated, purpose="forged relocation")
