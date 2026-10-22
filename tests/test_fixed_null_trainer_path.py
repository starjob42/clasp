from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from clasp.config import parse_grpo_args
from clasp.set_grpo import (
    compute_fixed_null_clasp_set_advantages,
    fixed_null_finite_batch_advantages,
    load_fixed_null_score_receipt,
    prompt_sha256,
)
from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_receipt(path: Path, *, prompt: object = "prompt", task: str = "tsp") -> Path:
    null_candidate = {"completion": ""}
    payload = {
        "schema_version": 1,
        "protocol": "clasp-fixed-null-construction-v1",
        "status": "frozen_before_training",
        "null_candidate": null_candidate,
        "null_candidate_sha256": hashlib.sha256(
            _canonical_bytes(null_candidate)
        ).hexdigest(),
        "evaluator_identity_sha256": "e" * 64,
        "records": [
            {
                "task": task,
                "prompt_sha256": prompt_sha256(prompt),
                "objective_score": 0.0,
                "max_constraint_violation_score": 0.75,
                "feasibility_flag": 0.0,
            }
        ],
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o440)
    return path


class _SetControllerState:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def get_clasp_bon_lambdas(self) -> dict[str, float]:
        return dict(self.values)

    def set_clasp_bon_lambdas(self, values: dict[str, float]) -> None:
        self.values = dict(values)


def _bare_fixed_null_trainer() -> PrimalDualGRPOTrainer:
    trainer = object.__new__(PrimalDualGRPOTrainer)
    trainer.num_generations = 4
    trainer.clasp_deployment_set_size = 2
    trainer.clasp_canonical_recovery_scale = 0.7
    trainer.clasp_canonical_recovery_temperature = 0.2
    trainer.clasp_bon_lambda_init = 1.0
    trainer.clasp_bon_target_failure_rate = 0.25
    trainer.clasp_bon_target_failure_rates = {}
    trainer.clasp_bon_dual_lr = 0.5
    trainer.clasp_bon_dual_ema_beta = 0.0
    trainer.clasp_bon_lambda_max = 10.0
    trainer.importance_sampling_level = "sequence"
    trainer.model = SimpleNamespace(training=True)
    trainer.primal_dual_reward_manager = _SetControllerState()
    trainer._clasp_bon_lambdas = {}
    trainer._clasp_bon_failure_ema = {}
    trainer._clasp_bon_observed_groups = {}
    trainer._clasp_bon_pending_failure_sums = {}
    trainer._clasp_bon_pending_group_counts = {}
    return trainer


def test_actual_trainer_fixed_null_path_matches_helper_and_uses_k_times_a() -> None:
    trainer = _bare_fixed_null_trainer()
    objective = torch.tensor([0.0, 0.4, 0.0, 0.9], dtype=torch.float64)
    violation = torch.tensor([0.8, 0.0, 0.3, 0.0], dtype=torch.float64)
    feasibility = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    null_objective = torch.zeros(4, dtype=torch.float64)
    null_violation = torch.full((4,), 0.75, dtype=torch.float64)
    null_feasibility = torch.zeros(4, dtype=torch.float64)

    coefficients, stats = trainer._compute_fixed_null_set_advantages(
        objective,
        violation,
        feasibility,
        null_objective,
        null_violation,
        null_feasibility,
        ["tsp"] * 4,
    )
    adapter = compute_fixed_null_clasp_set_advantages(
        objective,
        violation,
        feasibility,
        null_objective,
        null_violation,
        null_feasibility,
        rollout_batch_size=4,
        deployment_set_size=2,
        group_lambdas=torch.tensor([1.0], dtype=torch.float64),
        recovery_scale=0.7,
        recovery_temperature=0.2,
        target_failure_rates=torch.tensor([0.25], dtype=torch.float64),
    )
    helper = fixed_null_finite_batch_advantages(
        adapter.subset_utilities,
        adapter.subset_indices,
        adapter.replacement_utilities,
        batch_size=4,
    )
    torch.testing.assert_close(
        coefficients,
        helper.mean_reduced_policy_coefficients.reshape(-1),
    )
    torch.testing.assert_close(coefficients, 2.0 * helper.advantages.reshape(-1))
    assert abs(float(helper.realized_advantage_sums.item())) > 1e-8
    assert stats["clasp/fixed_null_set"] == 1.0
    assert stats["clasp/canonical_set"] == 0.0
    assert stats["clasp/fixed_null_realized_advantage_sum_abs_max"] > 0.0
    assert trainer._clasp_bon_pending_group_counts == {"tsp": 1}


def test_fixed_null_adapter_rejects_candidate_dependent_null_scores() -> None:
    actual = torch.zeros(4)
    with pytest.raises(ValueError, match="identical across every rollout row"):
        compute_fixed_null_clasp_set_advantages(
            actual,
            torch.ones(4),
            actual,
            actual,
            torch.tensor([0.5, 0.5, 0.6, 0.5]),
            actual,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=1.0,
        )


def test_fixed_null_cli_requires_and_identities_a_valid_receipt(tmp_path: Path) -> None:
    receipt_path = _write_receipt(tmp_path / "fixed-null.json")
    argv = [
        "clasp-grpo",
        "--model_path",
        "/tmp/model",
        "--tasks",
        "tsp",
        "--clasp_fixed_null_set",
        "--clasp_fixed_null_receipt_path",
        str(receipt_path),
        "--num_generations",
        "4",
        "--clasp_deployment_set_size",
        "2",
        "--per_device_train_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "1",
    ]
    with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
        "sys.argv", argv
    ):
        args = parse_grpo_args()
        resolved_env = dict(os.environ)
    receipt = load_fixed_null_score_receipt(receipt_path)
    assert args.use_clasp is True
    assert args.clasp_fixed_null_set is True
    assert args.clasp_canonical_set is False
    assert args.clasp_fixed_null_receipt_identity == receipt.identity
    assert resolved_env["CLASP_FIXED_NULL_SET"] == "1"
    assert resolved_env["CLASP_CANONICAL_SET"] == "0"
    assert resolved_env["CLASP_FIXED_NULL_RECEIPT_PATH"] == str(receipt_path.resolve())


def test_fixed_null_cli_fails_closed_without_construction_receipt() -> None:
    argv = [
        "clasp-grpo",
        "--model_path",
        "/tmp/model",
        "--tasks",
        "tsp",
        "--clasp_fixed_null_set",
        "--num_generations",
        "4",
        "--clasp_deployment_set_size",
        "2",
    ]
    with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
        "sys.argv", argv
    ), pytest.raises(ValueError, match="requires an explicit.*receipt"):
        parse_grpo_args()


def test_receipt_lookup_fails_closed_on_uncovered_prompt(tmp_path: Path) -> None:
    receipt = load_fixed_null_score_receipt(
        _write_receipt(tmp_path / "fixed-null.json", prompt="covered")
    )
    assert receipt.scores_for("tsp", "covered").max_constraint_violation_score == 0.75
    with pytest.raises(RuntimeError, match="no evaluator scores"):
        receipt.scores_for("tsp", "uncovered")
