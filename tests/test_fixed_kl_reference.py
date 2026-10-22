from __future__ import annotations

import errno
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
pytest.importorskip("tokenizers")

from peft import LoraConfig, get_peft_model  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import (  # noqa: E402
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from clasp.models import fixed_kl_reference as fixed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fixed_reference_test_converter",
    ROOT / "scripts" / "convert_legacy_saved_policy.py",
)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _tiny_qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _save_tiny_tokenizer(path: Path) -> None:
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        **{f"token-{index}": index for index in range(4, 64)},
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def _create_converted_saved_policy(root: Path) -> tuple[Path, Path]:
    base = root / "base"
    legacy = root / "legacy"
    converted = root / "converted"
    torch.manual_seed(20260814)
    Qwen2ForCausalLM(_tiny_qwen_config()).save_pretrained(
        base,
        safe_serialization=True,
    )
    _save_tiny_tokenizer(base)
    model = Qwen2ForCausalLM.from_pretrained(base, local_files_only=True)
    policy = get_peft_model(
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
        for index, (name, parameter) in enumerate(policy.named_parameters()):
            if ".lora_" in name:
                values = torch.arange(parameter.numel(), dtype=parameter.dtype)
                parameter.copy_(
                    (values.reshape(parameter.shape) + index + 1) / (parameter.numel() + index + 3)
                )
            elif name.endswith(".base_layer.bias"):
                parameter.add_(0.125 + index / 1000)
    policy.save_pretrained(legacy, safe_serialization=True)
    converter.convert_legacy_saved_policy(
        base_model_path=base,
        legacy_adapter_path=legacy,
        output_dir=converted,
        device="cpu",
        dtype="float32",
    )
    return base, converted


def _thaw(path: Path) -> None:
    if path.is_dir():
        path.chmod(0o700)
        for child in path.rglob("*"):
            child.chmod(0o700 if child.is_dir() else 0o600)
    elif path.exists():
        path.chmod(0o600)


@pytest.fixture(scope="module")
def pristine_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("fixed-kl-reference")
    base, converted = _create_converted_saved_policy(root)
    model = root / "merged-full-policy"
    receipt_path = root / "fixed_kl_reference.json"
    receipt = fixed.build_fixed_kl_reference(
        base_model_path=base,
        source_adapter_path=converted,
        tokenizer_path=base,
        output_model_path=model,
        receipt_path=receipt_path,
        claim_scope="development",
        dtype="float32",
    )
    yield model, receipt_path, receipt
    _thaw(model)
    _thaw(converted)
    _thaw(receipt_path)


@pytest.fixture()
def built_reference(
    pristine_reference: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    pristine_model, _, pristine_receipt = pristine_reference
    model = tmp_path / "merged-full-policy"
    shutil.copytree(pristine_model, model, copy_function=shutil.copy2)
    receipt = json.loads(json.dumps(pristine_receipt))
    receipt["model_path"] = str(model)
    receipt["receipt_sha256"] = fixed._sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path = tmp_path / "fixed_kl_reference.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o440)
    yield model, receipt_path, receipt
    _thaw(model)
    _thaw(receipt_path)


def test_real_tiny_peft_merge_fresh_reload_and_tokenizer_are_proven(
    built_reference: tuple[Path, Path, dict[str, object]],
) -> None:
    model, receipt_path, receipt = built_reference

    assert receipt["schema_version"] == 2
    assert receipt["protocol"] == fixed.DEVELOPMENT_PROTOCOL
    assert receipt["status"] == fixed.DEVELOPMENT_STATUS
    assert receipt["development_only"] is True
    assert receipt["publication_eligible"] is False
    assert receipt["source_policy"]["evidence_kind"] == (
        "development_legacy_saved_policy_conversion"
    )
    assert receipt["state_equivalence"]["exact"] is True
    assert receipt["tokenizer_equivalence"]["exact"] is True
    assert all(
        comparison["merged_vs_fresh_reference_bitwise_equal"] is True
        for comparison in receipt["functional_equivalence"]["comparisons"].values()
    )
    assert (
        fixed.inspect_fixed_kl_reference_artifact(
            model,
            receipt_path,
            claim_scope="development",
        )
        == receipt
    )
    assert (
        fixed.verify_fixed_kl_reference(
            model,
            receipt_path,
            claim_scope="development",
        )
        == receipt
    )

    with pytest.raises(fixed.FixedKLReferenceError, match="claim boundary"):
        fixed.inspect_fixed_kl_reference_artifact(
            model,
            receipt_path,
            claim_scope="publication",
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fixed.build_fixed_kl_reference(
            base_model_path=receipt["base_model"]["path"],
            source_adapter_path=receipt["source_policy"]["checkpoint_path"],
            tokenizer_path=receipt["tokenizer_source"]["path"],
            output_model_path=model,
            receipt_path=receipt_path,
            claim_scope="development",
            dtype="float32",
        )


def test_equal_hand_authored_random_digests_never_authorize_beta(
    built_reference: tuple[Path, Path, dict[str, object]],
) -> None:
    model, receipt_path, receipt = built_reference
    forged = json.loads(json.dumps(receipt))
    random_digest = "d" * 64
    for group in ("source", "merged", "fresh_reference"):
        for descriptor in forged["functional_equivalence"][group].values():
            descriptor["value_sha256"] = random_digest
    forged["receipt_sha256"] = fixed._sha256_json(
        {key: value for key, value in forged.items() if key != "receipt_sha256"}
    )
    _thaw(receipt_path)
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")
    receipt_path.chmod(0o440)

    # Integrity-only inspection cannot authenticate who computed a digest.  It
    # is intentionally never used to authorize training.
    assert (
        fixed.inspect_fixed_kl_reference_artifact(
            model,
            receipt_path,
            claim_scope="development",
        )
        == forged
    )
    with pytest.raises(
        fixed.FixedKLReferenceError,
        match="freshly reconstructed deterministic functional evidence",
    ):
        fixed.verify_fixed_kl_reference(
            model,
            receipt_path,
            claim_scope="development",
        )


def test_missing_or_tampered_artifacts_fail_closed(
    built_reference: tuple[Path, Path, dict[str, object]],
) -> None:
    model, receipt_path, _ = built_reference
    config_path = model / "config.json"
    _thaw(model)
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    fixed._make_read_only(model)
    with pytest.raises(fixed.FixedKLReferenceError, match="model bytes changed"):
        fixed.inspect_fixed_kl_reference_artifact(
            model,
            receipt_path,
            claim_scope="development",
        )

    receipt_path.chmod(0o600)
    receipt_path.unlink()
    with pytest.raises(fixed.FixedKLReferenceError, match="regular JSON file"):
        fixed.inspect_fixed_kl_reference_artifact(
            model,
            receipt_path,
            claim_scope="development",
        )


def test_equivalence_failure_reports_elementwise_error_budget() -> None:
    source = {
        "logits": torch.tensor([[[1.0]]]),
        "continuation_log_probabilities": torch.tensor([[1.0]]),
    }
    merged = {
        "logits": torch.tensor([[[0.0]]]),
        "continuation_log_probabilities": torch.tensor([[0.0]]),
    }
    with pytest.raises(fixed.FixedKLReferenceError) as captured:
        fixed._functional_evidence(
            source,
            merged,
            merged,
            dtype="float32",
        )
    message = str(captured.value)
    assert "probe logits" in message
    assert "max_absolute_error=1.0" in message
    assert "max_normalized_error=" in message
    assert "failing_elements=1/1" in message
    assert "atol=1e-05" in message
    assert "rtol=1e-05" in message


def test_nfs_cleanup_failure_never_masks_primary_equivalence_error(
    pristine_reference: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, receipt = pristine_reference
    output = tmp_path / "failed-output"
    output_receipt = tmp_path / "failed-receipt.json"
    primary_message = "primary equivalence failure remains authoritative"
    cleanup_warnings: list[str] = []

    def fail_equivalence(*args, **kwargs):
        raise fixed.FixedKLReferenceError(primary_message)

    def fail_nfs_cleanup(path):
        raise OSError(errno.ENOTEMPTY, "simulated NFS .nfs race", path)

    def warning_promoted_to_error(message, *args, **kwargs):
        cleanup_warnings.append(str(message))
        raise RuntimeWarning(message)

    with monkeypatch.context() as patch:
        patch.setattr(fixed, "_functional_evidence", fail_equivalence)
        patch.setattr(fixed.shutil, "rmtree", fail_nfs_cleanup)
        patch.setattr(fixed.warnings, "warn", warning_promoted_to_error)
        with pytest.raises(fixed.FixedKLReferenceError, match=primary_message):
            fixed.build_fixed_kl_reference(
                base_model_path=receipt["base_model"]["path"],
                source_adapter_path=receipt["source_policy"]["checkpoint_path"],
                tokenizer_path=receipt["tokenizer_source"]["path"],
                output_model_path=output,
                receipt_path=output_receipt,
                claim_scope="development",
                dtype="float32",
            )

    assert not output.exists()
    assert not output_receipt.exists()
    assert cleanup_warnings
    assert "primary build failure was preserved" in cleanup_warnings[0]
    staging = list(tmp_path.glob(".failed-output.staging.*"))
    assert len(staging) == 1
    fixed._thaw_owned_tree(staging[0])
    shutil.rmtree(staging[0])


def test_builder_clears_caller_model_references_before_staging_cleanup() -> None:
    source = (ROOT / "src/clasp/models/fixed_kl_reference.py").read_text(encoding="utf-8")
    finalizer = source.index("primary_error = sys.exc_info()[1]")
    clear_source = source.index("source_model = None", finalizer)
    clear_merged = source.index("merged_model = None", clear_source)
    clear_fresh = source.index("fresh_model = None", clear_merged)
    collect = source.index("gc.collect()", clear_fresh)
    remove = source.index("if staging.exists():", collect)
    assert finalizer < clear_source < clear_merged < clear_fresh < collect < remove
    assert "_release_model(" not in source
