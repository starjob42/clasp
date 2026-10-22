"""Recipe isolation, train-only reference admission and resume identity checks."""
from __future__ import annotations

import copy
import os
from dataclasses import replace

import pytest

from clasp.baselines.external_baseline_dev import ExternalBaselineRuntime, seal_reward_environment
from clasp.baselines.quality_signal_dev import (
    REWARD_HEADS,
    REWARD_MODES,
    build_runtime,
    check_resume_reward_contract,
    validate_reward_contract,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_quality_signal_grpo import build_parser


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items()
                                       if not k.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_"))})
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("GRPO_DATA_SEED", "43")
    monkeypatch.setenv("GRPO_STRICT_OPTIMIZER_RESUME", "1")
    monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", "50")
    seal_reward_environment()


def profile(task="mis", mode="reference_sigmoid_v2", estimator="external", extra=()):
    args = finalize_grpo_args(build_parser().parse_args([
        "--tasks", task, "--sft_checkpoint_path", f"/synthetic/{task}",
        "--external_dev_source_manifest", "/synthetic/SOURCE_MANIFEST.sha256",
        "--external_dev_source_manifest_sha256", "a" * 64,
        "--advantage_estimator", estimator, "--quality_reward", mode, *extra,
    ]))
    runtime = build_runtime({"status": "passed_frozen_manifest_successor_source",
                             "manifest": {"sha256": "a" * 64}}, args)
    return args, runtime


@pytest.mark.parametrize("task", ["tsp", "cvrp", "op", "mis", "jssp"])
@pytest.mark.parametrize("mode", REWARD_MODES)
@pytest.mark.parametrize("estimator", ["rloo", "external"])
def test_reward_ablation_preserves_training_budget_and_sampler(task, mode, estimator):
    args, runtime = profile(task, mode, estimator)
    runtime.validate_args(args)
    assert (args.max_steps, args.resolved_generation_batch_size, args.num_generations) == (100, 36, 9)
    assert (args.generation_temperature, args.generation_top_p, args.generation_top_k) == (1, 1, 0)
    assert (args.grpo_loss_type, args.grpo_beta, args.learning_rate) == ("dr_grpo", 0, 1e-6)
    info = runtime.metadata()
    assert info["method_id"] == f"{estimator}_b9_{mode}"
    assert runtime.reward_functions() == (REWARD_HEADS[task, mode],)
    assert info["reward_contract"]["source_manifest_sha256"] == "a" * 64
    assert not info["quality_calibration"]["method_superiority_established"]
    assert info["quality_calibration"]["reference_dependency_added_for_task"] == (
        mode == "reference_sigmoid_v2" and task in ("op", "mis"))


@pytest.mark.parametrize("extra", [
    ["--num_generations", "1"], ["--generation_temperature", "0.7"],
    ["--grpo_beta", "0.05"], ["--learning_rate", "0.00001"],
    ["--max_steps", "200"], ["--external_baseline_scale", "2"],
])
def test_quality_experiment_cannot_silently_change_other_factors(extra):
    args, runtime = profile(extra=extra)
    with pytest.raises(ValueError):
        runtime.validate_args(args)


def test_checkpoint_identity_is_copied_and_cross_mode_continuation_is_rejected():
    _, runtime = profile()
    expected = runtime.reward_contract()
    saved = {"reward_contract": copy.deepcopy(expected)}
    check_resume_reward_contract(expected, saved)
    # Legacy payloads remain admissible only for a legacy trainer with no contract.
    check_resume_reward_contract(None, {})
    for left, right in [(None, saved), (expected, {}), (expected, {"reward_contract": None})]:
        with pytest.raises(ValueError, match="Resume reward contract differs"):
            check_resume_reward_contract(left, right)
    for task, mode in [("op", "reference_sigmoid_v2"), ("mis", "legacy"), ("mis", "graded_v2")]:
        other = replace(runtime, task=task, quality_reward=mode).reward_contract()
        with pytest.raises(ValueError, match="Resume reward contract differs"):
            check_resume_reward_contract(expected, {"reward_contract": other})
    changed_source = {**expected, "source_manifest_sha256": "b" * 64}
    with pytest.raises(ValueError, match="Resume reward contract differs"):
        check_resume_reward_contract(expected, {"reward_contract": changed_source})
    independent = validate_reward_contract(expected)
    independent["mode"] = "legacy"
    assert runtime.reward_contract() == expected


@pytest.mark.parametrize("field,value", [
    ("quality_slope", 10), ("quality_slope", True), ("quality_slope", float("nan")),
    ("reward_head", "external_mis_graded_reward"), ("source_manifest_sha256", None),
    ("profile_id", "external-baseline-graded-raw-development-20260910-v1"),
])
def test_contract_rejects_parameter_or_implementation_drift(field, value):
    contract = profile()[1].reward_contract()
    with pytest.raises(ValueError):
        validate_reward_contract({**contract, field: value})


def test_all_training_references_checked_but_eval_labels_not_used(monkeypatch):
    # Isolate the new reference check from the separately tested dataset loader.
    monkeypatch.setattr(ExternalBaselineRuntime, "validate_dataset",
                        lambda self, dataset, label: {"task_counts": {self.task: len(dataset)}})
    runtime = profile()[1]
    row = {"answer": "Set: [0, 2], Objective: 999", "input": "",
           "instance_json": {"num_nodes": 3, "edges": [[0, 1]]}}
    accepted = runtime.validate_dataset([row], label="train")
    assert accepted["quality_signal_training_references"]["count"] == 1
    bad = {**row, "answer": "Set: [0, 1], Objective: 999"}
    with pytest.raises(ValueError, match="row 1"):
        runtime.validate_dataset([row, bad], label="train")
    assert "quality_signal_training_references" not in runtime.validate_dataset([bad], label="eval")
    assert "quality_signal_training_references" not in replace(runtime, quality_reward="graded_v2").validate_dataset([bad], label="train")
