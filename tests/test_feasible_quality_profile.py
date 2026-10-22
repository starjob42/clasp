"""Admission checks specific to the opt-in quality-credit experiment."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from clasp.baselines import feasible_quality_dev as q
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_feasible_quality_grpo import build_parser, resolve_declared_defaults
from clasp.rewards.feasible_quality import make_advantage_contract
from test_budget_matched_dev import env_for, make_runtime as baseline_runtime


def runtime(task="op", method="rloo_quality_cardinal", *, diagnostic=False):
    original = baseline_runtime(task, "rloo_quality", diagnostic)
    contract = copy.deepcopy(original.contract)
    contract.update(protocol=q.PROTOCOL, method=method,
                    advantage_contract=make_advantage_contract(q.ADVANTAGE_METHODS[method]))
    return q.build_runtime(original.source_binding, contract, original.contract_binding)


@pytest.mark.parametrize("task", q.HORIZONS)
@pytest.mark.parametrize("method", q.METHODS)
def test_only_auxiliary_identity_changes_from_calibrated_budget_anchor(task, method):
    current, anchor = runtime(task, method), baseline_runtime(task, "rloo_quality")
    assert current.algorithm_config() == anchor.algorithm_config()
    assert current.reward_contract() == anchor.reward_contract()
    assert current.reward_functions() == anchor.reward_functions()
    assert q.reward_environment(method) == anchor.metadata()["reward_environment"]
    meta = current.metadata()
    assert meta["common_budget"] == anchor.metadata()["common_budget"]
    assert meta["continuation"] == anchor.metadata()["continuation"]
    assert meta["common_budget"]["responses"] == 36 * q.HORIZONS[task]
    assert meta["teacher_weight"] == 0 and meta["policy_improvement_established"] is False
    assert current.feasible_quality_extension and current.budget_matched_extension
    assert current.external_baseline_extension and not current.requires_explicit_fixed_reference
    expected = make_advantage_contract(q.ADVANTAGE_METHODS[method])
    assert meta["advantage_contract"] == current.resume_contract()["advantage_contract"] == expected
    assert expected["weight"] == (0.0 if method.endswith("control") else .25)


@pytest.mark.parametrize("method", q.METHODS)
def test_finalized_cli_admits_exact_shared_recipe(method, monkeypatch):
    monkeypatch.setattr(q.os, "environ", dict(q.os.environ))
    current = runtime(method=method)
    env_for(monkeypatch, baseline_runtime())
    sealed_environment = dict(q.os.environ)
    argv = ["--tasks", "op", "--sft_checkpoint_path", "/tmp/sft",
            "--budget_contract", "/tmp/contract", "--budget_contract_sha256", "a" * 64,
            "--budget_method", method, "--output_dir", "/tmp/output"]
    args = finalize_grpo_args(resolve_declared_defaults(build_parser().parse_args(argv), current.contract))
    current.validate_args(args)
    assert args.max_steps == 1250 and args.effective_train_batch_size == 36
    # Explicitly asking for a short schedule must fail; a 100-update observation
    # is a checkpoint on the full horizon, not a newly decayed 100-step run.
    monkeypatch.setattr(q.os, "environ", sealed_environment)
    short = finalize_grpo_args(resolve_declared_defaults(
        build_parser().parse_args(argv + ["--max_steps", "100"]), current.contract))
    with pytest.raises(ValueError, match="numerical recipe"):
        current.validate_args(short)


@pytest.mark.parametrize("field,value", [
    ("method", "midrank"), ("weight", .5), ("gap_scale", .1),
    ("denominator", "feasible_siblings"), ("num_generations", 8),
    ("anchor", "graded_v2_rloo"), ("weight", True),
])
def test_cross_method_or_hyperparameter_changes_are_not_admitted(field, value):
    changed = runtime().contract
    changed["advantage_contract"][field] = value
    with pytest.raises(ValueError, match="advantage contract|declared method"):
        q.validate_contract(changed)


def test_matching_valid_other_method_cannot_be_smuggled_into_named_arm():
    changed = runtime().contract
    changed["advantage_contract"] = make_advantage_contract("midrank")
    with pytest.raises(ValueError, match="declared method"):
        q.validate_contract(changed)


def test_resume_contract_is_detached_and_binds_auxiliary_source_horizon_and_reward():
    current = runtime()
    saved = current.resume_contract()
    assert saved["source"] == current.contract["source"]
    assert saved["horizon"] == current.contract["horizon"] == 1250
    assert saved["reward_contract"]["mode"] == "reference_sigmoid_v2"
    saved["advantage_contract"]["weight"] = .75
    saved["declaration"]["advantage_contract"]["gap_scale"] = .75
    assert current.resume_contract()["advantage_contract"]["weight"] == .25
    current.contract["advantage_contract"] = make_advantage_contract("midrank")
    with pytest.raises(ValueError, match="mutated after validation"):
        current.resume_contract()


def test_new_recipe_has_no_legacy_historical_method_admission():
    changed = runtime().contract
    changed.update(method="rloo_quality")
    with pytest.raises(ValueError, match="Unknown budget"):
        q.validate_contract(changed)
    changed = runtime().contract
    changed["protocol"] = baseline_runtime().contract["protocol"]
    with pytest.raises(ValueError, match="Unknown budget"):
        q.validate_contract(changed)


def test_diagnostic_and_main_contracts_cannot_be_interchanged():
    diagnostic, main = runtime(diagnostic=True), runtime()
    assert diagnostic.contract["horizon"] == main.contract["horizon"] == 1250
    assert diagnostic.contract["chunk_boundaries"] == [5, 10]
    assert diagnostic.resume_contract() != main.resume_contract()
    assert diagnostic.metadata()["continuation"]["diagnostic_policy_must_not_be_promoted"]


def test_contract_byte_binding_rejects_even_semantically_equal_rewrite(tmp_path):
    contract = runtime().contract
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    bound, _ = q.file_identity(path)
    assert q.read_contract(path, bound["sha256"])[0] == contract
    path.write_text(json.dumps(contract, indent=2))
    with pytest.raises(ValueError, match="byte hash"):
        q.read_contract(path, bound["sha256"])


def test_resolved_config_forbids_silent_group_standardization():
    current = runtime()
    config = SimpleNamespace(**current.algorithm_config(), num_generations=9,
                             max_steps=1250, generation_batch_size=36,
                             gradient_accumulation_steps=3, steps_per_generation=3,
                             ignore_data_skip=False, eval_strategy="no")
    current.validate_grpo_config(config)
    config.scale_rewards = "group"
    with pytest.raises(ValueError, match="scale_rewards"):
        current.validate_grpo_config(config)
