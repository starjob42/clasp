"""All-task successor gates without changing the historical FOARL numerics."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os

import pytest

from clasp.baselines.foarl_f638_all_tasks_dev import (
    PROFILE_ID, SOURCE_REQUIRED_FILES, SOURCE_VERSION,
    build_all_tasks_foarl_runtime, verify_all_tasks_source_manifest,
)
from clasp.baselines.foarl_f638_dev_20260907 import (
    SOURCE_REQUIRED_FILES as OLD_REQUIRED, build_foarl_f638_dev_runtime,
)
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_TASKS
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_all_tasks_dev import build_parser


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items()
                                       if not k.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_", "DEV_"))
                                       and k != "RESUME_FROM_CHECKPOINT"})
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("GRPO_DATA_SEED", "43")


def parsed(task="tsp", extra=()):
    return finalize_grpo_args(build_parser().parse_args([
        "--tasks", task, "--sft_checkpoint_path", f"/synthetic/{task}",
        "--foarl_dev_source_manifest", "/synthetic/SOURCE_MANIFEST.sha256",
        "--foarl_dev_source_manifest_sha256", "a" * 64, *extra,
    ]))


def binding():
    return {"status": "passed_frozen_manifest_successor_source",
            "version": SOURCE_VERSION,
            "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "b" * 64}}}


@pytest.mark.parametrize("task", FOARL_F638_TASKS)
def test_every_task_admitted_with_preserved_foarl_numerical_recipe(task):
    args = parsed(task)
    runtime = build_all_tasks_foarl_runtime(binding())
    runtime.validate_args(args)
    old = build_foarl_f638_dev_runtime(binding())
    before, after = asdict(old.semantics), asdict(runtime.semantics)
    allowed = {"profile_id", "publication_label", "supported_tasks", "trainer_source_version"}
    assert {k: v for k, v in before.items() if k not in allowed} == {
        k: v for k, v in after.items() if k not in allowed}
    assert runtime.reward_functions() == old.reward_functions()
    assert (runtime.semantics.beta, runtime.semantics.scale_rewards,
            runtime.semantics.loss_type, runtime.semantics.std_estimator) == (
        0.05, "group", "grpo", "sample_unbiased")
    assert (args.generation_temperature, args.generation_top_p, args.generation_top_k) == (0.7, 0.9, 50)
    receipt = runtime.metadata()
    assert receipt["profile_id"] == PROFILE_ID
    assert receipt["all_tasks_extension"]["fresh_uninterrupted_updates"] == 100
    assert receipt["all_tasks_extension"]["resume_allowed"] is False
    assert receipt["reference"] == "explicit_frozen_corrected_sft_step_zero"
    assert old.semantics.supported_tasks == ("cvrp", "mis")


@pytest.mark.parametrize("task", ["tsp", "op", "jssp"])
def test_historical_two_task_gate_is_still_closed(task):
    with pytest.raises(ValueError):
        build_foarl_f638_dev_runtime(binding()).validate_args(parsed(task))


@pytest.mark.parametrize("task", FOARL_F638_TASKS)
def test_five_step_diagnostic_is_explicit_and_excluded_from_comparison(task, monkeypatch):
    monkeypatch.setenv("DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES", "5")
    args = parsed(task)
    runtime = build_all_tasks_foarl_runtime(binding())
    runtime.validate_args(args)
    assert (args.max_steps, args.save_steps) == (5, 5)
    assert runtime.metadata()["all_tasks_extension"]["fresh_uninterrupted_updates"] == 5
    assert "excluded_from_comparison" in runtime.metadata()["study_scope"]


@pytest.mark.parametrize("key,value", [("RESUME_FROM_CHECKPOINT", "/cp"),
                                       ("DEV_RESUME_CHECKPOINT", "/cp"),
                                       ("GRPO_STOP_AT_GLOBAL_STEP", "50"),
                                       ("GRPO_REBASE_LOADED_SCHEDULER_AT_STEP", "50"),
                                       ("GRPO_EXTENSION_MAX_STEPS", "200")])
def test_resume_and_partial_stop_cannot_silently_change_controls(key, value, monkeypatch):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="fresh uninterrupted"):
        build_all_tasks_foarl_runtime(binding()).validate_args(parsed())


@pytest.mark.parametrize("extra", [("--max_steps", "50"), ("--save_steps", "50"),
                                  ("--save_total_limit", "2"), ("--grpo_beta", "0"),
                                  ("--generation_temperature", "1"),
                                  ("--num_generations", "1")])
def test_recipe_drift_is_rejected(extra):
    with pytest.raises(ValueError):
        build_all_tasks_foarl_runtime(binding()).validate_args(parsed(extra=extra))


def test_source_verification_binds_new_entrypoint_profile_and_launcher(tmp_path):
    names = set(OLD_REQUIRED) | set(SOURCE_REQUIRED_FILES)
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned fixture\n")
    lines = [hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() + "  " + name
             for name in sorted(names)]
    manifest = tmp_path / "SOURCE_MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    verified = verify_all_tasks_source_manifest(manifest, sha, source_root=tmp_path)
    assert verified["version"] == SOURCE_VERSION
    assert set(SOURCE_REQUIRED_FILES) <= verified["required_files"].keys()
    (tmp_path / SOURCE_REQUIRED_FILES[0]).write_text("# tampered\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_all_tasks_source_manifest(manifest, sha, source_root=tmp_path)


def test_omitting_the_executed_successor_launcher_is_rejected(tmp_path):
    names = (set(OLD_REQUIRED) | set(SOURCE_REQUIRED_FILES)) - {SOURCE_REQUIRED_FILES[-1]}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    manifest = tmp_path / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                                + "  " + name + "\n" for name in sorted(names)))
    with pytest.raises(ValueError, match="omits required successor"):
        verify_all_tasks_source_manifest(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                         source_root=tmp_path)
