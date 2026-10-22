from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines.foarl_f638_dev_20260907 import (
    COMMON_PROMPT_SHA256,
    DEV_DEFAULTS,
    SOURCE_REQUIRED_FILES,
    build_foarl_f638_dev_runtime,
    diagnostic_updates,
    verify_dev_source_manifest,
)
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_REWARD_FUNCS
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_f638_dev_20260907 import (
    build_parser,
    seal_dev_loader_environment,
)


@pytest.fixture(autouse=True)
def isolate_shared_parser_environment(monkeypatch):
    # finalize_grpo_args exports resolved settings; do not let one parser
    # invocation turn an implicit default into another test's explicit input.
    monkeypatch.setattr(os, "environ", dict(os.environ))


def _source_tree(tmp_path):
    for relative in SOURCE_REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture source {relative}\n")
    manifest = tmp_path / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(SOURCE_REQUIRED_FILES)
    ))
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _runtime(tmp_path):
    manifest, digest = _source_tree(tmp_path)
    return build_foarl_f638_dev_runtime(
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)
    )


def _args(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES", raising=False)
    monkeypatch.delenv("GRPO_DATA_SEED", raising=False)
    return finalize_grpo_args(build_parser().parse_args([
        "--sft_checkpoint_path", "/frozen/cvrp_sft", "--tasks", "cvrp",
        "--foarl_dev_source_manifest", "/frozen/SOURCE_MANIFEST.sha256",
        "--foarl_dev_source_manifest_sha256", "a" * 64,
    ]))


def test_new_parser_and_runtime_use_common_budget_prompt_and_f638_loss(tmp_path, monkeypatch):
    args = _args(monkeypatch)
    runtime = _runtime(tmp_path)
    runtime.validate_args(args)
    assert args.no_reason is True  # validation never rewrites the actual prompt choice
    assert runtime.requires_explicit_fixed_reference
    assert runtime.reward_functions() == FOARL_F638_REWARD_FUNCS
    metadata = runtime.metadata()
    assert metadata["method_id"] == "foarl_f638_dev"
    assert metadata["prompt_sha256"] == COMMON_PROMPT_SHA256
    assert metadata["no_reason"] is True
    assert metadata["reproduction_claim"] == "not_an_official_reproduction"
    assert metadata["reward_heads"] == ["optimality", "feasibility"]
    assert metadata["num_generations"] == 9
    assert metadata["beta"] == 0.05
    assert metadata["advantage_normalization"]["std_estimator"] == "sample_unbiased"
    assert metadata["shared_trainer_source"]["sha256"] == (
        metadata["source_binding"]["required_files"]["src/clasp/entrypoints/train_grpo.py"]["sha256"]
    )
    assert args.resolved_generation_batch_size == 36
    assert args.resolved_steps_per_generation is None  # GRPOConfig derives 3
    metadata["source_binding"]["manifest"]["sha256"] = "changed"
    assert runtime.metadata()["source_binding"]["manifest"]["sha256"] != "changed"


@pytest.mark.parametrize("field", sorted(DEV_DEFAULTS))
def test_changed_budget_or_prompt_rejected(field, tmp_path, monkeypatch):
    args = _args(monkeypatch)
    runtime = _runtime(tmp_path)
    value = getattr(args, field)
    setattr(args, field, not value if isinstance(value, bool) else (
        value + 1 if isinstance(value, (int, float)) else "changed"
    ))
    with pytest.raises(ValueError):
        runtime.validate_args(args)


@pytest.mark.parametrize("field,value", [
    ("grpo_beta", 0.0), ("grpo_loss_type", "dr_grpo"), ("foarl_scale_rewards", "none"),
    ("foarl_advantage_epsilon", 0.001), ("foarl_num_iterations", 2),
    ("foarl_sync_ref_model", True), ("verify_gt_rewards", True), ("use_clasp", True),
    ("tasks", ["tsp"]), ("tasks", ["cvrp", "mis"]),
])
def test_released_loss_and_specialist_gates_retained(field, value, tmp_path, monkeypatch):
    args = _args(monkeypatch)
    runtime = _runtime(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        runtime.validate_args(args)


def test_five_step_diagnostic_is_explicit_and_excluded(tmp_path, monkeypatch):
    args = _args(monkeypatch)
    runtime = _runtime(tmp_path)
    args.max_steps = args.save_steps = 5
    with pytest.raises(ValueError, match="recipe mismatch"):
        runtime.validate_args(args)
    monkeypatch.setenv("DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES", "5")
    runtime.validate_args(args)
    assert runtime.metadata()["study_scope"] == "distributed_correctness_diagnostic_excluded_from_comparison"
    assert runtime.metadata()["common_budget"]["max_steps"] == 5
    args.max_steps = args.save_steps = 100
    with pytest.raises(ValueError, match="recipe mismatch"):
        runtime.validate_args(args)
    for value in ("1", "05", "100"):
        with pytest.raises(ValueError):
            diagnostic_updates({"DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES": value})


def test_world_size_and_data_seed_are_not_silent_overrides(tmp_path, monkeypatch):
    args = _args(monkeypatch)
    runtime = _runtime(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        runtime.validate_args(args)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("GRPO_DATA_SEED", "42")
    with pytest.raises(ValueError, match="GRPO_DATA_SEED"):
        runtime.validate_args(args)


def test_source_gate_rejects_drift_missing_coverage_and_digest(tmp_path):
    manifest, digest = _source_tree(tmp_path)
    verify_dev_source_manifest(manifest, digest, source_root=tmp_path)
    with pytest.raises(ValueError, match="manifest digest"):
        verify_dev_source_manifest(manifest, "0" * 64, source_root=tmp_path)
    extra = tmp_path / "src/clasp/unbound.py"
    extra.write_text("# unmanifested source\n")
    with pytest.raises(ValueError, match="omits Python"):
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)
    extra.unlink()
    (tmp_path / SOURCE_REQUIRED_FILES[0]).write_text("# changed\n")
    with pytest.raises(ValueError, match="file digest mismatch"):
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)


@pytest.mark.parametrize("bad_line", [
    "0" * 64 + "  ../outside.py\n", "0" * 64 + "  /absolute.py\n",
    "0" * 64 + "  ./src/extra.py\n", "not a sha256sum line\n",
])
def test_source_manifest_rejects_unsafe_paths_and_malformed_entries(tmp_path, bad_line):
    manifest, _ = _source_tree(tmp_path)
    manifest.write_text(bad_line + manifest.read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)


def test_source_manifest_rejects_duplicate_and_symlink(tmp_path):
    manifest, _ = _source_tree(tmp_path)
    first = manifest.read_text().splitlines()[0] + "\n"
    manifest.write_text(manifest.read_text() + first)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)
    manifest, digest = _source_tree(tmp_path)
    path = tmp_path / SOURCE_REQUIRED_FILES[0]
    target = tmp_path / "target"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="unlinked regular"):
        verify_dev_source_manifest(manifest, digest, source_root=tmp_path)


def test_common_mis_environment_and_resolved_grpo_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASP_MIS_JIANG_RESPONSE_OUTPUT", "0")
    assert seal_dev_loader_environment()["CLASP_MIS_JIANG_RESPONSE_OUTPUT"] == "0"
    monkeypatch.setenv("CLASP_MIS_JIANG_RESPONSE_OUTPUT", "1")
    with pytest.raises(ValueError, match="representation environment"):
        seal_dev_loader_environment()
    runtime = _runtime(tmp_path)
    fields = ("beta epsilon epsilon_high loss_type scale_rewards num_iterations sync_ref_model "
              "reward_weights importance_sampling_level mask_truncated_completions use_liger_loss").split()
    overrides = runtime.grpo_config_overrides(dict.fromkeys(fields))
    config = SimpleNamespace(num_generations=9, **overrides)
    runtime.validate_grpo_config(config)
    bad = copy.copy(config)
    bad.reward_weights = [0.5, 1.0]
    with pytest.raises(RuntimeError, match="reward_weights"):
        runtime.validate_grpo_config(bad)


@pytest.mark.parametrize("task", ["cvrp", "mis"])
@pytest.mark.parametrize("steps", [5, 100])
def test_actual_launcher_argv_finalizes_and_passes_runtime(task, steps, tmp_path, monkeypatch):
    # Exercise the literal launcher arrays after its own environment scrub,
    # without filesystem gates, model loading, or any scheduler operation.
    fixture_path = Path(__file__).with_name("test_grpo_incumbent_dev_launcher.py")
    spec = importlib.util.spec_from_file_location("incumbent_launcher_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    diagnostic = "5" if steps == 5 else "0"
    run = fixture.IncumbentLauncherTests().result(
        "foarl_f638_dev", DEV_TASK=task, DEV_MAX_STEPS=str(steps),
        DEV_SAVE_STEPS=str(steps), DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES=diagnostic,
    )
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_"))
    }
    environment.update(run["env"])
    environment.update(WORLD_SIZE="4", DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES=diagnostic)
    monkeypatch.setattr(os, "environ", environment)
    argv = run["argv"]
    marker = "clasp.entrypoints.train_foarl_f638_dev_20260907"
    args = finalize_grpo_args(build_parser().parse_args(argv[argv.index(marker) + 1:]))
    runtime = _runtime(tmp_path)
    runtime.validate_args(args)
    assert args.tasks == [task]
    assert args.per_device_eval_batch_size == 9
    assert args.max_steps == args.save_steps == steps
    assert args.no_reason is True
    assert seal_dev_loader_environment()["CLASP_MIS_JIANG_RESPONSE_OUTPUT"] == "0"
