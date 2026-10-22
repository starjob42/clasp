"""Explicit B=18 recipe; the original nine-rollout pilot stays unchanged."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("b18_artifacts", ROOT / "scripts/clasp_grpo_fair_dev_artifacts.py")
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def recipe(method, override):
    source = (ROOT / "scripts/train_clasp_canonical_checkpoint_dev_a100.slurm").read_text()
    body = 'case "${METHOD}" in' + source.split('case "${METHOD}" in', 1)[1].split('case "${TASK}" in', 1)[0]
    command = 'set -eu\nCANONICAL_QUALITY_ANCHOR_RATIO=0.0\nESTIMATOR_ARM=token_a0_control\n' + body
    command += 'printf "%s\\n" "$ROLLOUT_BATCH_SIZE" "$CANONICAL_MAX_SUBSETS" "$USE_CLASP" "$CANONICAL_QUALITY_ANCHOR_RATIO" "$DENSE_FEASIBILITY_SCALE" "$ESTIMATOR_ARM"\n'
    return subprocess.run(["bash", "-c", command], env=dict(os.environ, METHOD=method, DEV_ROLLOUT_BATCH_SIZE=override),
                          capture_output=True, text=True)


@pytest.mark.parametrize("method,clasp,ratio", [("grpo_z", "0", "0.0"), ("clasp_grpo_anchor_v3", "1", "0.5")])
def test_matched_b18_changes_subset_budget_without_adding_v4_bonus(method, clasp, ratio):
    result = recipe(method, "18")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:5] == ["18", "43758", clasp, ratio, "0.0"]
    assert "_b9_" not in result.stdout
    standard = recipe(method, "")
    assert standard.returncode == 0, standard.stderr
    assert standard.stdout.splitlines()[:2] == ["9", "9"]


@pytest.mark.parametrize("method,override", [("grpo_z", "9"), ("grpo_z", "12"), ("grpo_z", "18.0"),
                                             ("clasp_grpo_anchor_v4", "18"), ("clasp_strong_v1", "18")])
def test_unsupported_override_fails_before_model_loading(method, override):
    result = recipe(method, override)
    assert result.returncode == 2
    assert "rollout override requires" in result.stderr


def training_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    contract = {
        "method": "clasp_grpo_anchor_v3", "task": "cvrp", "seed": 43,
        "fixed_horizon": {"max_steps": 100, "world_size": 4}, "continuation": {},
        "initialization": {"adapter_model_sha256": artifacts.SFT_HASHES["cvrp"]},
        "source": {"root": str(source), "manifest_sha256": "a" * 64},
        "production_preserved": {
            "rollout_batch_b": 18, "global_generation_batch_size": 36, "learning_rate": 1e-6,
            "max_completion_length": 3000, "max_prompt_length": 29768,
            "per_device_train_batch_size": 3, "gradient_accumulation_steps": 3,
            "beta": 0.0, "deployment_set_k": 8, "prompt_style": "jiang_f638", "no_reason": True,
        },
        "policy_estimator": {
            "canonical_max_subsets": 43758, "canonical_exact_scale": 1.0,
            "group_std_correction": 1, "importance_sampling_level": "token", "policy_length_alpha": 0.0,
            "canonical_set": True, "use_clasp": True, "canonical_grpo_anchor_scale": 1.0,
            "canonical_quality_anchor_ratio": 0.5, "canonical_dense_feasibility_scale": 0.0,
        },
    }
    run = tmp_path / "train"
    checkpoint = run / "checkpoints/checkpoint-100"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        (checkpoint / name).write_text("fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 100}')
    (run / "local_development_run_contract.json").write_text(json.dumps(contract))
    monkeypatch.setattr(artifacts, "validate_source", lambda path: {"sha256": "a" * 64})
    return run, source, contract


def test_b18_validation_requires_explicit_expected_budget(tmp_path, monkeypatch):
    run, source, contract = training_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="shared training settings mismatch"):
        artifacts.validate_training(run, "cvrp", "clasp_grpo_anchor_v3", source)
    assert artifacts.validate_training(run, "cvrp", "clasp_grpo_anchor_v3", source, rollout_batch_b=18)["status"] == "training_validated"
    contract["policy_estimator"]["canonical_max_subsets"] = 1024
    (run / "local_development_run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(SystemExit, match="complete 18-choose-8"):
        artifacts.validate_training(run, "cvrp", "clasp_grpo_anchor_v3", source, rollout_batch_b=18)


@pytest.mark.parametrize("value", [True, 18.0, 12, "18"])
def test_invalid_expected_budget_is_rejected(value):
    with pytest.raises(SystemExit, match="unsupported training rollout budget"):
        artifacts.validate_training(None, "cvrp", "grpo_z", None, rollout_batch_b=value)
