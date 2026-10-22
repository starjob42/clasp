from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.models import training_fidelity as training

ROOT = Path(__file__).resolve().parents[1]


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clear_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(training.PUBLICATION_FIDELITY_ENV, raising=False)
    monkeypatch.delenv(training.DEVELOPMENT_FIDELITY_ENV, raising=False)


def test_fidelity_modes_are_strict_and_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    assert training.checkpoint_fidelity_mode() == "off"

    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    assert training.checkpoint_fidelity_mode() == "development"
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "true")
    with pytest.raises(training.PublicationFidelityError, match="mutually exclusive"):
        training.checkpoint_fidelity_enabled()

    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "0")
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "maybe")
    with pytest.raises(training.PublicationFidelityError, match="explicit boolean"):
        training.publication_fidelity_enabled()


def test_safe_qwen_lora_configuration_is_required_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "1")
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    training.require_publication_lora_configuration(
        full_finetune=False,
        lora_r=64,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=targets,
        effective_bias="all",
    )
    with pytest.raises(training.PublicationFidelityError, match="expected 'all'"):
        training.require_publication_lora_configuration(
            full_finetune=False,
            lora_r=64,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules=targets,
            effective_bias="lora_only",
        )


def test_development_preflight_gathers_exact_closure_without_publication_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    closure = {"tensor_inventory_sha256": "a" * 64, "tensors": []}
    invariant = {"status": "exact", "trainable_tensor_count": 476}
    monkeypatch.setattr(training, "audit_trainable_state_closure", lambda model, name: closure)
    monkeypatch.setattr(
        training,
        "assert_qwen25_7b_publication_invariant",
        lambda model, observed, name: invariant,
    )
    trainer = SimpleNamespace(
        model=object(),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )

    report = training.run_publication_fidelity_preflight(trainer, phase="pre_train")

    assert report is not None
    assert report["closure"] is closure
    assert report["qwen_structural_invariant"] is invariant
    assert report["publication_invariant"] is None
    assert report["development_only"] is True
    assert report["distributed"]["world_size"] == 1


def test_distributed_preflight_rejects_rank_digest_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def disagree(values, local):
        values[:] = [local, {"rank": 1, "digest": "f" * 64}]

    monkeypatch.setattr(dist, "all_gather_object", disagree)
    with pytest.raises(training.PublicationFidelityError, match="differs across ranks"):
        training._distributed_digest_agreement("a" * 64, phase="test")


def test_cuda_peak_evidence_binds_torch_peaks_and_physical_nvidia_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="NVIDIA RTX A6000", total_memory=51_041_271_808),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 10_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda index: 12_000)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    monkeypatch.setattr(
        training.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "3, GPU-025e10e6-263f-d814-6dd5-added86fc8af, NVIDIA RTX A6000, 49140, 535.261.03\n"
            )
        ),
    )

    evidence = training._local_cuda_memory_evidence(0)

    assert evidence == {
        "rank": 0,
        "cuda_available": True,
        "device_index": 0,
        "physical_device_index": 3,
        "visible_device_token": "3",
        "device_uuid": "GPU-025e10e6-263f-d814-6dd5-added86fc8af",
        "device_name": "NVIDIA RTX A6000",
        "driver_version": "535.261.03",
        "cuda_runtime_version": "12.1",
        "total_memory_bytes": 51_041_271_808,
        "nvidia_smi_total_memory_mib": 49_140,
        "peak_allocated_bytes": 10_000,
        "peak_reserved_bytes": 12_000,
    }


def test_checkpoint_callback_writes_only_on_world_zero_after_directory_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    calls = []
    monkeypatch.setattr(
        training,
        "_write_live_receipt",
        lambda trainer, path, **kwargs: calls.append(Path(path)),
    )
    trainer = SimpleNamespace()
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    control = object()

    callback.on_save(
        args,
        SimpleNamespace(global_step=7, is_world_process_zero=False),
        control,
    )
    assert calls == []
    assert (
        callback.on_save(
            args,
            SimpleNamespace(global_step=7, is_world_process_zero=True),
            control,
        )
        is control
    )
    assert calls == [checkpoint]


def test_primary_final_writer_consumes_consensus_without_collective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    closure = {"id": "terminal"}
    invariant = {"status": "exact"}
    trainer = SimpleNamespace(
        model=object(),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )
    monkeypatch.setattr(training, "audit_trainable_state_closure", lambda model: closure)
    monkeypatch.setattr(
        training,
        "assert_qwen25_7b_publication_invariant",
        lambda model, observed: invariant,
    )
    monkeypatch.setattr(
        training,
        "_distributed_digest_agreement",
        lambda *args, **kwargs: pytest.fail("rank-zero writer must not run a collective"),
    )
    written = {}

    def write(path, observed_closure, publication):
        written.update(path=Path(path), closure=observed_closure, publication=publication)
        return {"status": "ok"}

    monkeypatch.setattr(training, "write_checkpoint_fidelity_receipt", write)
    expected = {
        "closure": closure,
        "qwen_structural_invariant": invariant,
    }

    result = training.write_final_publication_fidelity(
        trainer,
        tmp_path,
        expected_preflight=expected,
    )

    assert result == {"status": "ok"}
    assert written == {"path": tmp_path, "closure": closure, "publication": None}


def test_loaded_resume_is_reaudited_before_first_new_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=7),
        _clasp_pending_exact_kl_resume_checkpoint=str(checkpoint),
        _clasp_pending_exact_kl_resume_post_receipt={"receipt_sha256": "a" * 64},
        _clasp_pending_exact_kl_resume_closure_digest="b" * 64,
        _clasp_exact_peft_kl_reference={"receipt_sha256": "c" * 64},
    )
    calls: list[dict[str, object]] = []

    def verify(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        return {"status": "exact"}

    monkeypatch.setattr(training, "verify_publication_resume_checkpoint", verify)
    assert training.verify_loaded_exact_kl_resume_state(trainer) == {"status": "exact"}
    assert calls == [
        {
            "path": checkpoint,
            "expected_kl_reference": {"receipt_sha256": "c" * 64},
            "trainer": trainer,
            "expect_live_policy_matches_checkpoint": True,
        }
    ]
    assert not hasattr(trainer, "_clasp_pending_exact_kl_resume_checkpoint")


def test_callback_reaudits_loaded_resume_on_train_begin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    trainer = SimpleNamespace()
    calls: list[object] = []
    monkeypatch.setattr(
        training,
        "verify_loaded_exact_kl_resume_state",
        lambda observed: calls.append(observed),
    )
    control = object()
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    assert callback.on_train_begin(None, None, control) is control
    assert calls == [trainer]


def test_development_conversion_input_is_accepted_only_in_development_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "converted"
    checkpoint.mkdir()
    for filename in (
        training.ADAPTER_CONFIG_FILENAME,
        "adapter_model.safetensors",
        training.LEGACY_CONVERSION_RECEIPT_FILENAME,
    ):
        (checkpoint / filename).write_bytes(filename.encode("utf-8"))
    expected = {"development_only": True}
    monkeypatch.setattr(
        training,
        "verify_development_conversion_checkpoint",
        lambda path, purpose: expected,
    )

    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    assert (
        training.verify_publication_input_adapter(checkpoint, purpose="test conversion") is expected
    )

    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "0")
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "1")
    with pytest.raises(training.PublicationFidelityError, match="cannot stat publication"):
        training.verify_publication_input_adapter(checkpoint, purpose="publication")


def test_development_conversion_receipt_is_live_reverified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "converted-live"
    checkpoint.mkdir()
    config = {
        "peft_type": "LORA",
        "bias": "all",
        "r": 1,
        "target_modules": ["q_proj"],
    }
    (checkpoint / training.ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    tensor_value = struct.pack("<f", 1.5)
    header = {
        "base_model.model.q_proj.lora_A.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, len(tensor_value)],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + tensor_value
    )
    for path in checkpoint.iterdir():
        path.chmod(0o440)
    checkpoint.chmod(0o550)
    output_tree = training._tree_manifest(checkpoint)
    inspection = training.inspect_checkpoint_adapter(checkpoint)

    checkpoint.chmod(0o750)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "protocol": training.LEGACY_CONVERSION_PROTOCOL,
        "status": "converted_saved_policy_verified",
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
        "converted_output": {
            "path": str(checkpoint),
            "serialization_bias_policy": "all",
            "manifest_excluding_receipt": {
                "file_count": output_tree["file_count"],
                "files": output_tree["files"],
                "tree_sha256": output_tree["tree_sha256"],
            },
            "checkpoint_fidelity_adapter_inspection": inspection,
        },
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    receipt_path = checkpoint / training.LEGACY_CONVERSION_RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o440)
    checkpoint.chmod(0o550)

    assert (
        training.verify_development_conversion_checkpoint(
            checkpoint,
            purpose="test",
        )
        == receipt
    )

    receipt_path.chmod(0o640)
    receipt_path.write_text("{}", encoding="utf-8")
    receipt_path.chmod(0o440)
    with pytest.raises(training.PublicationFidelityError, match="self-hash mismatch"):
        training.verify_development_conversion_checkpoint(checkpoint, purpose="tampered")


def test_fixed_kl_live_verifier_receives_an_unambiguous_claim_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, str]] = []

    def verify(model, receipt, *, claim_scope):
        calls.append((model, receipt, claim_scope))
        return {"claim_scope": claim_scope}

    monkeypatch.setattr(training, "_verify_fixed_kl_reference_live", verify)

    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    assert training.verify_fixed_kl_reference("model", "receipt") == {"claim_scope": "development"}

    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "0")
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "1")
    assert training.verify_fixed_kl_reference("model", "receipt") == {"claim_scope": "publication"}
    assert calls == [
        ("model", "receipt", "development"),
        ("model", "receipt", "publication"),
    ]


def test_training_entrypoints_gather_before_primary_only_final_write() -> None:
    for relative in (
        "src/clasp/entrypoints/train_sft.py",
        "src/clasp/entrypoints/train_grpo.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        gather = text.index('phase="pre_final_save"')
        barrier = text.index("maybe_torch_distributed_barrier()", gather)
        primary = text.index("if is_primary_process():", barrier)
        save = text.index("trainer.save_model(output_dir)", primary)
        receipt = text.index("write_final_publication_fidelity(", save)
        assert gather < barrier < primary < save < receipt
        assert "expected_preflight=final_fidelity_preflight" in text[receipt : receipt + 300]


def test_biased_lora_never_infers_foundation_kl_reference() -> None:
    text = (ROOT / "src/clasp/entrypoints/train_grpo.py").read_text(encoding="utf-8")
    assert "infer_base_model_path_from_peft_checkpoint" not in text
    assert "resolve_explicit_fixed_kl_reference" in text
    assert "verify_fixed_kl_reference_distributed" in text
    assert "Checkpoint-fidelity mode forbids fallback" in text
    assert "found a newer checkpoint that is not the" in text
    live_verify = text.index("verified_kl_reference = verify_fixed_kl_reference_distributed(")
    trainer_construction = text.index("trainer = trainer_cls(")
    assert live_verify < trainer_construction
    assert "policy_realpath != reference_realpath" in text
    assert "tokenizer_realpath != reference_realpath" in text

    sft_text = (ROOT / "src/clasp/entrypoints/train_sft.py").read_text(encoding="utf-8")
    assert "Checkpoint-fidelity mode forbids fallback" in sft_text
    assert "found a newer checkpoint that is not the" in sft_text

    standard_text = (ROOT / "src/clasp/entrypoints/train_grpo_standard.py").read_text(
        encoding="utf-8"
    )
    assert "from clasp.entrypoints.train_grpo import run_grpo" in standard_text
    assert "run_grpo(args)" in standard_text


def test_publication_input_check_accepts_full_merged_model_for_shared_grpo_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_model = tmp_path / "merged-full-model"
    full_model.mkdir()
    (full_model / "config.json").write_text("{}", encoding="utf-8")
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "1")

    assert (
        training.verify_publication_input_adapter(
            full_model,
            purpose="standard GRPO merged initialization",
        )
        is None
    )


def test_exact_kl_preflight_rejects_a_prepared_reference_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_modes(monkeypatch)
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    policy = object()
    reference = object()
    wrapped_reference = object()

    class Accelerator:
        @staticmethod
        def unwrap_model(model):
            if model is wrapped_reference:
                return reference
            return model

    trainer = SimpleNamespace(
        model=policy,
        ref_model=wrapped_reference,
        accelerator=Accelerator(),
    )
    with pytest.raises(training.PublicationFidelityError, match="unsupported distributed wrapper"):
        training.run_exact_peft_kl_runtime_preflight(
            trainer,
            wrapped_reference,
            {},
            tmp_path,
            dtype="bfloat16",
            beta=0.05,
            sync_ref_model=False,
            use_vllm=False,
        )
