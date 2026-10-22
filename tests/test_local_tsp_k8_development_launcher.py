from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_local_tsp_k8_development",
    ROOT / "scripts" / "run_local_tsp_k8_development.py",
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _option_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _completed(stdout: str, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def test_modes_preserve_k8_effective_batch_and_actual_train_derived_eval_sizes() -> None:
    gate = launcher.MODES["memory_gate"]
    gate_args = launcher.build_train_args(gate, launcher.OUTPUT_ROOT / gate.output_name)
    assert _option_value(gate_args, "--max_steps") == "1"
    assert _option_value(gate_args, "--max_samples_per_task") == "160"
    assert _option_value(gate_args, "--max_eval_samples_per_task") == "8"
    assert _option_value(gate_args, "--min_eval_samples_per_task") == "8"
    assert (gate.expected_train_rows, gate.expected_eval_rows) == (152, 8)

    dense = launcher.MODES["dense40"]
    dense_args = launcher.build_train_args(dense, launcher.OUTPUT_ROOT / dense.output_name)
    assert _option_value(dense_args, "--max_steps") == "40"
    assert _option_value(dense_args, "--max_samples_per_task") == "0"
    assert _option_value(dense_args, "--max_eval_samples_per_task") == "100"
    assert (dense.expected_train_rows, dense.expected_eval_rows) == (3040, 100)

    for arguments in (gate_args, dense_args):
        assert _option_value(arguments, "--num_generations") == "8"
        assert _option_value(arguments, "--generation_batch_size") == "8"
        assert _option_value(arguments, "--per_device_train_batch_size") == "2"
        assert _option_value(arguments, "--gradient_accumulation_steps") == "16"
        assert _option_value(arguments, "--max_prompt_length") == "29768"
        assert _option_value(arguments, "--max_completion_length") == "3000"
        assert "--eval-from-train-split" in arguments
        assert "--no-prefer_subset_files" in arguments
        assert "--clasp_strict_defaults" in arguments
        assert _option_value(arguments, "--clasp_bon_success_credit") == "dense_rank"
        assert _option_value(arguments, "--clasp_proper_dual_gradient_mode") == "signed"
        assert "--pd_group_scalarization" in arguments
        assert "--no-clasp_self_replay" in arguments
        assert "--no-clasp_adaptive_length" in arguments


def test_environment_is_fixed_and_scrubs_resume_research_and_distributed_state() -> None:
    inherited = {
        "PATH": "/bin",
        "CLASP_BON_ALIGNED": "stale",
        "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
        "GRPO_DATA_SEED": "999",
        "WANDB_PROJECT": "stale",
        "RESUME_FROM_CHECKPOINT": "/unsafe",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "2",
    }
    result = launcher.build_environment(inherited)
    assert result["PATH"] == "/bin"
    assert result["CUDA_VISIBLE_DEVICES"] == "3"
    assert result["GRPO_DATA_SEED"] == "42"
    assert result["PD_IMPORTANCE_SAMPLING_LEVEL"] == "sequence"
    assert result["CLASP_BON_ALIGNED"] == "1"
    assert result["WANDB_MODE"] == "disabled"
    for name in (
        "WANDB_PROJECT",
        "RESUME_FROM_CHECKPOINT",
        "WORLD_SIZE",
        "LOCAL_RANK",
    ):
        assert name not in result


def test_gpu_gate_requires_idle_physical_gpu3_a6000() -> None:
    def good(*args, **kwargs):
        return _completed(
            "0, NVIDIA GeForce RTX 2080 Ti, 11264, 0\n3, NVIDIA RTX A6000, 49140, 23\n"
        )

    assert launcher.validate_gpu3(good) == {
        "physical_index": 3,
        "name": "NVIDIA RTX A6000",
        "memory_total_mib": 49140,
        "memory_used_mib_at_launch": 23,
        "cuda_visible_devices": "3",
        "world_size": 1,
    }

    def wrong_gpu(*args, **kwargs):
        return _completed("3, NVIDIA GeForce RTX 2080 Ti, 11264, 0\n")

    with pytest.raises(RuntimeError, match="must be the 48 GB RTX A6000"):
        launcher.validate_gpu3(wrong_gpu)

    def busy(*args, **kwargs):
        return _completed("3, NVIDIA RTX A6000, 49140, 2048\n")

    with pytest.raises(RuntimeError, match="not idle enough"):
        launcher.validate_gpu3(busy)


def test_runtime_probe_is_exact_and_rejects_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.PYTHON.__class__, "is_file", lambda self: True)
    monkeypatch.setattr(os, "access", lambda path, mode: True)

    def good(*args, **kwargs):
        return _completed(json.dumps(launcher.EXPECTED_RUNTIME) + "\n")

    assert launcher.probe_runtime(good) == launcher.EXPECTED_RUNTIME
    drifted = dict(launcher.EXPECTED_RUNTIME)
    drifted["peft"] = "0.18.1"

    def bad(*args, **kwargs):
        return _completed(json.dumps(drifted) + "\n")

    with pytest.raises(RuntimeError, match="runtime version drift"):
        launcher.probe_runtime(bad)


def test_create_once_contract_is_immutable_self_hashed_and_never_reused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    payload = {"schema_version": 1, "protocol": launcher.PROTOCOL, "mode": "dense40"}
    path = launcher.create_contract_once(run_dir, payload)
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(launcher.canonical_json_bytes(payload)).hexdigest()
    assert stored["contract_sha256"] == expected_hash
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    with pytest.raises(RuntimeError, match="already exists"):
        launcher.require_fresh_run_dir(run_dir)
    with pytest.raises(RuntimeError, match="appeared during launch"):
        launcher.create_contract_once(run_dir, payload)


def _write_manifest_tree(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    root.mkdir(parents=True)
    manifest = {}
    for name, contents in files.items():
        path = root / name
        path.write_bytes(contents)
        manifest[name] = hashlib.sha256(contents).hexdigest()
    return manifest


def test_artifact_verifier_requires_exact_copies_and_excludes_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    base_manifest = _write_manifest_tree(base, {"config.json": b"base"})
    sft = tmp_path / "sft"
    adapter = {
        "base_model_name_or_path": str(base.resolve()),
        "r": 64,
        "lora_alpha": 64,
        "bias": "lora_only",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    status = {
        "completed": True,
        "global_step": 31_250,
        "max_steps": 31_250,
        "has_saved_model": True,
    }
    sft_manifest = _write_manifest_tree(
        sft,
        {
            "adapter_config.json": json.dumps(adapter).encode(),
            "stage_status.json": json.dumps(status).encode(),
            "adapter_model.safetensors": b"weights",
        },
    )
    data_root = tmp_path / "data_rl"
    train = data_root / "tsp" / "train" / "train_rl.json"
    train.parent.mkdir(parents=True)
    train.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")

    monkeypatch.setattr(launcher, "BASE_MODEL", base.resolve())
    monkeypatch.setattr(launcher, "BASE_MANIFEST", base_manifest)
    monkeypatch.setattr(launcher, "SFT_MODEL", sft)
    monkeypatch.setattr(launcher, "SFT_MANIFEST", sft_manifest)
    monkeypatch.setattr(launcher, "DATA_ROOT", data_root)
    monkeypatch.setattr(launcher, "TRAIN_DATA", train)
    monkeypatch.setattr(
        launcher, "TRAIN_DATA_SHA256", hashlib.sha256(train.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(launcher, "EXPECTED_TRAIN_SOURCE_ROWS", 2)

    receipt = launcher.verify_artifacts()
    assert receipt["rl_train"]["rows"] == 2
    train.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RL train SHA-256 mismatch"):
        launcher.verify_artifacts()
    train.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")
    eval_dir = data_root / "tsp" / "eval"
    eval_dir.mkdir()
    with pytest.raises(RuntimeError, match="must not contain tsp/eval"):
        launcher.verify_artifacts()


def test_launcher_is_explicitly_nonpublication_and_gate_path_is_not_adopted() -> None:
    assert launcher.LIMITATIONS["development_only"] is True
    assert launcher.LIMITATIONS["publication_eligible"] is False
    assert launcher.LIMITATIONS["sota_claim_eligible"] is False
    assert "Instruct" in launcher.LIMITATIONS["instruction_tuned_initialization"]
    assert "PEFT 0.19.1" in launcher.LIMITATIONS["runtime_contract_mismatch"]
    assert "one RTX A6000" in launcher.LIMITATIONS["single_gpu_mismatch"]
    assert launcher.MODES["memory_gate"].output_name != "k8_memory_gate"
