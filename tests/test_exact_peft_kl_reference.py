from __future__ import annotations

import hashlib
import json
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from clasp.models import exact_peft_kl_reference as exact  # noqa: E402
from clasp.models import loading as model_loading  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _Config:
    vocab_size = 32
    use_cache = False
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "use_cache": self.use_cache,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }


class _TinyPeftPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.projection = torch.nn.Linear(4, 32, bias=True)
        self.register_buffer("scale", torch.tensor(1.0))
        self.config = _Config()
        self.active_adapters = ["default"]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)) * self.scale)


def _paired_models():
    torch.manual_seed(17)
    policy = _TinyPeftPair()
    reference = _TinyPeftPair()
    reference.load_state_dict(policy.state_dict())
    reference.requires_grad_(False)
    reference.eval()
    return policy, reference


def _make_read_only_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o550 if child.is_dir() else 0o440)
    path.chmod(0o550)


def _write_one_tensor_safetensors(path: Path, *, dtype: str, raw: bytes) -> None:
    header = {
        "base_model.model.q_proj.lora_A.weight": {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, len(raw)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)


def _source_binding(tmp_path: Path):
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    adapter = tmp_path / "adapter"
    for path in (base, tokenizer, adapter):
        path.mkdir()
    (base / "model.safetensors").write_bytes(b"base-state")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {
        "bias": "all",
        "base_model_name_or_path": str(base.absolute()),
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_one_tensor_safetensors(
        adapter / "adapter_model.safetensors",
        dtype="F32",
        raw=struct.pack("<f", 1.5),
    )
    source_receipt = {
        "schema_version": 1,
        "protocol": exact._LEGACY_CONVERSION_PROTOCOL,
        "status": "converted_saved_policy_verified",
    }
    source_receipt["receipt_sha256"] = _sha256_json(source_receipt)
    source_receipt_path = adapter / exact._LEGACY_RECEIPT
    source_receipt_path.write_text(json.dumps(source_receipt), encoding="utf-8")
    for path in (base, tokenizer, adapter):
        _make_read_only_tree(path)
    binding = exact.build_source_binding(
        policy_adapter_path=adapter,
        reference_adapter_path=adapter,
        authoritative_receipt_path=source_receipt_path,
        authoritative_receipt=source_receipt,
        tokenizer_path=tokenizer,
        claim_scope="development",
    )
    return binding


def _receipt(tmp_path: Path):
    binding = _source_binding(tmp_path)
    policy, reference = _paired_models()
    runtime_pair = exact.audit_exact_runtime_pair(
        policy,
        reference,
        runtime_dtype="bfloat16",
    )
    local_digest = _sha256_json(
        {
            "source_binding": binding,
            "runtime_pair": runtime_pair,
            "claim_scope": "development",
        }
    )
    distributed = {
        "phase": "exact_peft_kl_step_zero",
        "world_size": 1,
        "rank_digests": [{"rank": 0, "digest": local_digest}],
        "consensus_digest": local_digest,
    }
    receipt = exact.build_runtime_receipt(
        source_binding=binding,
        runtime_pair=runtime_pair,
        distributed=distributed,
        training_contract={
            "dtype": "bfloat16",
            "beta": 0.05,
            "sync_ref_model": False,
            "use_vllm": False,
            "parameterization": "continue_same_bias_all_sft_adapter",
            "policy_adapter_autocast_dtype": True,
            "reference_adapter_autocast_dtype": True,
        },
        claim_scope="development",
    )
    return receipt


def test_exact_runtime_pair_is_bitwise_and_storage_independent() -> None:
    policy, reference = _paired_models()

    evidence = exact.audit_exact_runtime_pair(policy, reference)

    assert evidence["state_bitwise_equal"] is True
    assert evidence["independent_storage"] is True
    assert evidence["reference_trainable_tensor_count"] == 0
    assert evidence["functional_equivalence"]["bitwise_equal"] is True
    assert evidence["functional_equivalence"]["max_absolute_error"] == 0.0
    assert len(evidence["functional_equivalence"]["input_ids"]) == 3


def test_exact_runtime_pair_rejects_trainable_or_changed_reference() -> None:
    policy, reference = _paired_models()
    reference.projection.bias.requires_grad_(True)
    with pytest.raises(exact.ExactPeftKLReferenceError, match="trainable tensors"):
        exact.audit_exact_runtime_pair(policy, reference)

    reference.requires_grad_(False)
    with torch.no_grad():
        reference.projection.bias.add_(1.0)
    with pytest.raises(exact.ExactPeftKLReferenceError, match="state tensor differs"):
        exact.audit_exact_runtime_pair(policy, reference)


def test_runtime_receipt_is_create_once_self_hashed_and_reopenable(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    output = tmp_path / "run"

    written = exact.write_runtime_receipt(output, receipt, claim_scope="development")

    path = output / exact.RECEIPT_FILENAME
    assert written == receipt
    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    assert exact.inspect_runtime_receipt(path, claim_scope="development") == receipt
    assert exact.write_runtime_receipt(output, receipt, claim_scope="development") == receipt


def test_recomputed_self_hash_cannot_hide_inconsistent_rank_evidence(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    output = tmp_path / "run"
    exact.write_runtime_receipt(output, receipt, claim_scope="development")
    path = output / exact.RECEIPT_FILENAME
    path.chmod(0o640)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["distributed"]["consensus_digest"] = "0" * 64
    tampered["receipt_sha256"] = _sha256_json(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    path.write_text(
        json.dumps(tampered, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o440)

    with pytest.raises(exact.ExactPeftKLReferenceError, match="digest consensus"):
        exact.inspect_runtime_receipt(path, claim_scope="development")


def test_old_v1_runtime_protocol_cannot_validate_as_symmetric_v2(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    receipt["protocol"] = "clasp-development-exact-peft-kl-reference-v1"
    receipt["receipt_sha256"] = _sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path = tmp_path / exact.RECEIPT_FILENAME
    path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o440)

    with pytest.raises(exact.ExactPeftKLReferenceError, match="claim boundary"):
        exact.inspect_runtime_receipt(path, claim_scope="development")


def test_source_binding_requires_immutable_trees(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "weights").write_bytes(b"base")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"bias": "all", "base_model_name_or_path": str(base.absolute())}),
        encoding="utf-8",
    )
    source = {"protocol": exact._LEGACY_CONVERSION_PROTOCOL}
    source["receipt_sha256"] = _sha256_json(source)
    receipt = adapter / exact._LEGACY_RECEIPT
    receipt.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(exact.ExactPeftKLReferenceError, match="read-only|must be immutable"):
        exact.build_source_binding(
            policy_adapter_path=adapter,
            reference_adapter_path=adapter,
            authoritative_receipt_path=receipt,
            authoritative_receipt=source,
            tokenizer_path=base,
            claim_scope="development",
        )


def test_saved_bfloat16_lora_is_rejected_before_source_precision_is_reduced(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "bf16-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"bias":"all"}', encoding="utf-8")
    _write_one_tensor_safetensors(
        adapter / "adapter_model.safetensors",
        dtype="BF16",
        raw=struct.pack("<H", 0x3FC0),
    )

    with pytest.raises(
        exact.ExactPeftKLReferenceError,
        match=r"observed_lora_dtypes=\['BF16'\]",
    ):
        exact.saved_adapter_dtype_evidence(adapter)


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


def test_tiny_qwen_bfloat16_exact_pair_requires_symmetric_adapter_autocast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real policy/reference loaders that gate beta-positive GRPO."""

    from clasp.entrypoints.train_grpo import _load_frozen_exact_peft_reference
    from clasp.models.exact_peft_kl_post_step import reference_has_input_grad_hook

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    _save_tiny_tokenizer(base)
    config = Qwen2Config(
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
    torch.manual_seed(20260815)
    Qwen2ForCausalLM(config).float().save_pretrained(base, safe_serialization=True)
    source = get_peft_model(
        Qwen2ForCausalLM.from_pretrained(
            base,
            dtype=torch.float32,
            local_files_only=True,
        ),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="all",
        ),
    )
    with torch.no_grad():
        for index, (name, parameter) in enumerate(source.named_parameters()):
            if ".lora_" not in name:
                continue
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_((values + index + 0.123) / (parameter.numel() + index + 7))
    source_lora = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
        if ".lora_" in name
    }
    source.save_pretrained(adapter, safe_serialization=True)

    policy, tokenizer, is_peft = model_loading.load_model_and_tokenizer(
        str(adapter),
        "bfloat16",
        padding_side="left",
        tokenizer_path=str(base),
    )
    assert is_peft is True
    policy_parameters = dict(policy.named_parameters())
    assert source_lora
    assert all(
        torch.equal(value, policy_parameters[name].detach().cpu())
        for name, value in source_lora.items()
    )

    bad_base = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    asymmetric_reference = PeftModel.from_pretrained(
        bad_base,
        adapter,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_loading._align_special_token_ids_with_tokenizer(asymmetric_reference, tokenizer)
    asymmetric_reference.requires_grad_(False)
    asymmetric_reference.eval()
    with pytest.raises(exact.ExactPeftKLReferenceError, match="paired state dtype mismatch"):
        exact.audit_exact_runtime_pair(
            policy,
            asymmetric_reference,
            runtime_dtype="bfloat16",
        )

    reference = _load_frozen_exact_peft_reference(
        {
            "base_model_path": str(base),
            "adapter_path": str(adapter),
        },
        "bfloat16",
        tokenizer=tokenizer,
    )
    evidence = exact.audit_exact_runtime_pair(
        policy,
        reference,
        runtime_dtype="bfloat16",
    )
    lora_key = next(key for key in policy.state_dict() if ".lora_A." in key)
    assert policy.state_dict()[lora_key].dtype == torch.float32
    assert reference.state_dict()[lora_key].dtype == torch.float32
    assert evidence["state_bitwise_equal"] is True
    assert evidence["functional_equivalence"]["runtime_dtype"] == "bfloat16"
    assert evidence["functional_equivalence"]["autocast_enabled"] is True
    assert evidence["functional_equivalence"]["logits"]["dtype"] == "torch.bfloat16"
    assert evidence["functional_equivalence"]["bitwise_equal"] is True
    assert not any(parameter.requires_grad for parameter in reference.parameters())
    assert reference_has_input_grad_hook(reference) is False


def test_resealed_development_relocation_cannot_disable_live_proof() -> None:
    verification = {
        "method": "independent_source_and_relocated_policy_reload_v1",
        "upstream_receipt_verified": True,
        "path_independent_base_manifests_exact": True,
        "source_and_output_stable": True,
        "independent_model_loads": True,
        "full_model_state_exact": True,
        "full_model_state": {},
        "functional_equivalence_exact": True,
        "functional_equivalence": {},
        "device": "cpu",
        "requested_dtype": "float32",
    }
    receipt = {
        "schema_version": 1,
        "protocol": exact._DEVELOPMENT_RELOCATION_PROTOCOL,
        "status": "development_saved_policy_base_relocation_verified",
        "claim_scope": "development_only",
        "development_only": True,
        "relocation_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
        "claim_boundary": exact._DEVELOPMENT_RELOCATION_CLAIM_BOUNDARY,
        "verifier": {},
        "upstream_conversion": {},
        "frozen_base": {},
        "relocated_output": {},
        "transformation": {},
        "verification": verification,
        "software": {
            "python": "test",
            "torch": "test",
            "transformers": "test",
            "peft": "test",
            "safetensors": "test",
        },
        "receipt_sha256": "0" * 64,
    }
    assert exact._validate_development_relocation_claim(receipt) is verification
    receipt["verification"] = {**verification, "independent_model_loads": False}

    with pytest.raises(exact.ExactPeftKLReferenceError, match="verification is incomplete"):
        exact._validate_development_relocation_claim(receipt)


def test_train_grpo_exact_pair_ordering_and_fail_closed_gates() -> None:
    text = (ROOT / "src/clasp/entrypoints/train_grpo.py").read_text(encoding="utf-8")
    load = text.index("explicit_kl_reference_model = _load_frozen_exact_peft_reference(")
    loader = text.index("def _load_frozen_exact_peft_reference(")
    peft_load = text.index(
        "reference = load_peft_causal_lm_with_explicit_adapter_autocast(",
        loader,
    )
    align = text.index("_align_special_token_ids_with_tokenizer(reference, tokenizer)", peft_load)
    freeze = text.index("reference.requires_grad_(False)", align)
    trainer = text.index("trainer = trainer_cls(")
    prepare = text.index("prepared_kl_reference = trainer.accelerator.prepare_model(", trainer)
    inject = text.index("trainer.ref_model = prepared_kl_reference", prepare)
    token_check = text.index(
        "assert_exact_paired_special_token_ids(",
        inject,
    )
    audit = text.index("exact_kl_runtime_receipt = run_exact_peft_kl_runtime_preflight(", inject)
    train = text.index("trainer.train(", inject)
    assert (
        peft_load < align < freeze < load < trainer < prepare < inject < token_check < audit < train
    )
    assert "explicit_kl_reference_model = None" not in text[load:trainer]
    assert "tokenizer=tokenizer" in text[load:trainer]
    assert 'optional_grpo_kwargs["sync_ref_model"] = False' in text
    assert "exact PEFT KL mode forbids vLLM" in text
    assert "local_files_only=True" in text
    assert "trust_remote_code=False" in text
    loading_text = (ROOT / "src/clasp/models/loading.py").read_text(encoding="utf-8")
    assert "autocast_adapter_dtype=True" in loading_text
    assert "low_cpu_mem_usage=False" in loading_text
    assert '"low_cpu_mem_usage": True' in loading_text
    assert "autocast_adapter_dtype=False" not in text
    assert "random.setstate(python_rng)" in text
    assert "torch.cuda.set_rng_state_all(cuda_rng)" in text


def _qwen_token_configuration_pair():
    class _Tokenizer(SimpleNamespace):
        def __len__(self):
            return 151_665

    tokenizer = _Tokenizer(bos_token_id=None, eos_token_id=151_645, pad_token_id=151_643)

    def raw_base_model():
        return SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=152_064,
                bos_token_id=151_643,
                eos_token_id=151_645,
                pad_token_id=None,
            ),
            generation_config=SimpleNamespace(
                bos_token_id=151_643,
                eos_token_id=[151_645, 151_643],
                pad_token_id=151_643,
            ),
        )

    policy = raw_base_model()
    reference = raw_base_model()
    model_loading._align_special_token_ids_with_tokenizer(policy, tokenizer)
    return tokenizer, policy, reference


def _special_token_projection(model, attribute: str) -> tuple[object, object, object]:
    config = getattr(model, attribute)
    return tuple(
        getattr(config, field) for field in ("bos_token_id", "eos_token_id", "pad_token_id")
    )


def test_exact_reference_replays_policy_tokenizer_alignment_for_model_and_generation() -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()

    assert _special_token_projection(policy, "config") == (None, 151_645, 151_643)
    assert _special_token_projection(reference, "config") == (151_643, 151_645, None)
    assert _special_token_projection(policy, "generation_config") == (
        None,
        151_645,
        151_643,
    )
    assert _special_token_projection(reference, "generation_config") == (
        151_643,
        [151_645, 151_643],
        151_643,
    )
    with pytest.raises(RuntimeError, match="model config mismatch for bos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)

    reference.config.bos_token_id = None
    with pytest.raises(RuntimeError, match="model config mismatch for pad_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)

    _, _, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)
    for attribute in ("config", "generation_config"):
        assert _special_token_projection(policy, attribute) == (None, 151_645, 151_643)
        assert _special_token_projection(reference, attribute) == (None, 151_645, 151_643)


def test_qwen_tokenizer_null_bos_remains_the_canonical_policy_value() -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()
    assert tokenizer.bos_token_id is None
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    for model in (policy, reference):
        assert model.config.bos_token_id is None
        assert model.generation_config.bos_token_id is None
    model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)


def test_exact_pair_rejects_equal_but_tokenizer_wrong_special_token_ids() -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    policy.config.pad_token_id = 151_644
    reference.config.pad_token_id = 151_644
    with pytest.raises(RuntimeError, match="model config mismatch for pad_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)


def test_exact_pair_requires_explicit_special_token_fields_on_every_surface() -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    del reference.generation_config.bos_token_id
    with pytest.raises(RuntimeError, match="generation config lacks bos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)

    tokenizer, policy, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    del tokenizer.bos_token_id
    with pytest.raises(RuntimeError, match="tokenizer lacks bos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)


def test_exact_pair_rejects_generation_token_drift_without_sequence_normalization() -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    reference.generation_config.eos_token_id = [151_645]
    with pytest.raises(RuntimeError, match="generation config mismatch for eos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)

    reference.generation_config.eos_token_id = (151_645,)
    with pytest.raises(RuntimeError, match="generation config mismatch for eos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)


@pytest.mark.parametrize("bad_value", [True, -1, 152_064, [], "151645", 151645.0])
def test_exact_pair_rejects_equal_but_invalid_special_token_ids(bad_value: object) -> None:
    tokenizer, policy, reference = _qwen_token_configuration_pair()
    model_loading._align_special_token_ids_with_tokenizer(reference, tokenizer)
    policy.generation_config.eos_token_id = bad_value
    reference.generation_config.eos_token_id = bad_value
    with pytest.raises(RuntimeError, match="invalid eos_token_id"):
        model_loading.assert_exact_paired_special_token_ids(policy, reference, tokenizer)
