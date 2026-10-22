from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest
from collections import OrderedDict, defaultdict, deque
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset

from clasp.config import parse_grpo_args
from clasp.data.loaders import (
    ensure_no_instance_overlap,
    load_mixed_grpo_dataset,
    load_mixed_sft_dataset,
    split_dataset,
    stable_instance_signature,
)
from clasp.entrypoints.train_grpo import ClaspPassAtKValidationGateCheckpointCallback
from clasp.rewards.clasp_math import compute_bon_aligned_advantages
from clasp.rewards.heads import MultiTaskPrimalDualManager
from clasp.rewards.primal_dual import PrimalDualConstrainedReward
from clasp.set_grpo import (
    compute_canonical_clasp_set_advantages,
    compute_dense_clasp_auxiliary_advantages,
)
from clasp.trainers.primal_dual_grpo import (
    DUAL_SEPARATE_Z_FORMULA_ID,
    PrimalDualGRPOTrainer,
    _canonical_task_order,
    _complete_group_permutation,
    _compose_clasp_recovery_gaps,
    _strict_env_flag,
    _strict_env_task_thresholds,
    _strict_nonnegative_float_env,
    compute_scalarized_group_advantages,
    compute_separate_z_group_advantages,
    reduce_clasp_policy_loss,
)


class _BonStateManager:
    def __init__(self):
        self.values: dict[str, float] = {}

    def get_clasp_bon_lambdas(self) -> dict[str, float]:
        return dict(self.values)

    def set_clasp_bon_lambdas(self, values: dict[str, float]) -> None:
        self.values = dict(values)


def _bare_bon_trainer() -> PrimalDualGRPOTrainer:
    trainer = object.__new__(PrimalDualGRPOTrainer)
    trainer.num_generations = 2
    trainer.model = SimpleNamespace(training=True)
    trainer.primal_dual_reward_manager = _BonStateManager()
    trainer.clasp_bon_lambda_init = 1.0
    trainer.clasp_bon_lambda_max = 10.0
    trainer.clasp_bon_target_failure_rate = 0.25
    trainer.clasp_bon_target_failure_rates = {}
    trainer.clasp_bon_dual_lr = 0.5
    trainer.clasp_bon_dual_ema_beta = 0.0
    trainer.clasp_bon_quality_temperature = 0.0
    trainer.clasp_bon_success_credit = "dense_rank"
    trainer.clasp_bon_dense_quality_scale = 1.0
    trainer.clasp_bon_dense_feasibility_scale = 1.0
    trainer.clasp_bon_dense_violation_mix = 0.5
    trainer.clasp_bon_recovery_scale = 1.0
    trainer._clasp_bon_lambdas = {}
    trainer._clasp_bon_failure_ema = {}
    trainer._clasp_bon_observed_groups = {}
    trainer._clasp_bon_pending_failure_sums = {}
    trainer._clasp_bon_pending_group_counts = {}
    return trainer


def _bare_frontier_cdf_trainer() -> PrimalDualGRPOTrainer:
    trainer = _bare_bon_trainer()
    trainer.clasp_frontier_cdf = True
    trainer.clasp_frontier_cdf_gap_targets = {
        "tsp": 0.0107,
        "cvrp": 0.0453,
        "op": 0.0185,
        "mis": 0.0104,
        "jssp": 0.0820,
    }
    trainer.clasp_frontier_cdf_quality_temperature_initial = 1.0
    trainer.clasp_frontier_cdf_quality_temperature_final = 0.1
    trainer.clasp_frontier_cdf_quality_anneal_fraction = 0.5
    trainer.clasp_frontier_cdf_violation_temperature = 0.15
    trainer.clasp_frontier_cdf_recovery_scale = 1.0
    trainer.clasp_frontier_cdf_relative_failed_recovery = False
    trainer.clasp_frontier_cdf_relative_recovery_scale = 1.0
    trainer.clasp_frontier_cdf_objective_score_scale = 2.0
    trainer.clasp_frontier_cdf_gap_slope = 3.0
    trainer.clasp_frontier_cdf_feasibility_violation_tolerance = 1e-8
    trainer.state = SimpleNamespace(global_step=0, max_steps=100)
    trainer.args = SimpleNamespace(max_steps=100)
    return trainer


def _bare_canonical_set_trainer() -> PrimalDualGRPOTrainer:
    trainer = _bare_bon_trainer()
    trainer.num_generations = 4
    trainer.clasp_canonical_set = True
    trainer.clasp_deployment_set_size = 2
    trainer.clasp_canonical_recovery_scale = 0.7
    trainer.clasp_canonical_dual_recovery_coupling = 0.0
    trainer.clasp_canonical_recovery_temperature = 0.2
    trainer.clasp_canonical_exact_scale = 1.0
    trainer.clasp_canonical_grpo_anchor_scale = 0.0
    trainer.clasp_canonical_quality_anchor_ratio = 0.0
    trainer.importance_sampling_level = "sequence"
    return trainer


class ClaspTrainerCorrectnessTests(unittest.TestCase):
    def test_canonical_cli_resolves_b9_k8_clean_core(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_canonical_set",
            "--num_generations",
            "9",
            "--clasp_deployment_set_size",
            "8",
            "--per_device_train_batch_size",
            "9",
            "--gradient_accumulation_steps",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
            resolved_env = dict(os.environ)
        self.assertTrue(args.use_clasp)
        self.assertTrue(args.use_primal_dual)
        self.assertTrue(args.clasp_canonical_set)
        self.assertEqual(args.num_generations, 9)
        self.assertEqual(args.clasp_deployment_set_size, 8)
        self.assertEqual(args.grpo_loss_type, "grpo")
        self.assertFalse(args.clasp_adaptive_length)
        self.assertFalse(args.clasp_proper_dual)
        self.assertEqual(resolved_env["CLASP_CANONICAL_SET"], "1")
        self.assertEqual(resolved_env["CLASP_DEPLOYMENT_SET_SIZE"], "8")

    def test_strong_canonical_cli_resolves_dense_b12_k8_credit(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "cvrp",
            "--clasp_canonical_set",
            "--num_generations",
            "12",
            "--generation_batch_size",
            "36",
            "--clasp_deployment_set_size",
            "8",
            "--clasp_canonical_max_subsets",
            "495",
            "--clasp_canonical_exact_scale",
            "0.25",
            "--clasp_canonical_dense_quality_scale",
            "0.5",
            "--clasp_canonical_dense_feasibility_scale",
            "1.5",
            "--clasp_canonical_dense_violation_scale",
            "0.5",
            "--clasp_canonical_dense_recovery_scale",
            "1.5",
            "--clasp_canonical_dual_recovery_coupling",
            "1.0",
            "--clasp_policy_length_alpha",
            "0.0",
            "--per_device_train_batch_size",
            "3",
            "--gradient_accumulation_steps",
            "3",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
        self.assertEqual(args.num_generations, 12)
        self.assertEqual(args.clasp_deployment_set_size, 8)
        self.assertEqual(args.clasp_canonical_max_subsets, 495)
        self.assertEqual(args.clasp_canonical_exact_scale, 0.25)
        self.assertEqual(args.clasp_canonical_dense_feasibility_scale, 1.5)
        self.assertEqual(args.clasp_canonical_dense_recovery_scale, 1.5)
        self.assertEqual(args.clasp_canonical_dual_recovery_coupling, 1.0)

    def test_canonical_cli_resolves_grpo_anchor_residual_credit(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "cvrp",
            "--clasp_canonical_set",
            "--num_generations",
            "9",
            "--clasp_deployment_set_size",
            "8",
            "--clasp_canonical_exact_scale",
            "0.25",
            "--clasp_canonical_grpo_anchor_scale",
            "1.0",
            "--per_device_train_batch_size",
            "9",
            "--gradient_accumulation_steps",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
        self.assertEqual(args.clasp_canonical_exact_scale, 0.25)
        self.assertEqual(args.clasp_canonical_grpo_anchor_scale, 1.0)

    def test_canonical_cli_resolves_norm_matched_quality_residual(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "cvrp",
            "--clasp_canonical_set",
            "--num_generations",
            "9",
            "--clasp_deployment_set_size",
            "8",
            "--clasp_canonical_grpo_anchor_scale",
            "1.0",
            "--clasp_canonical_quality_anchor_ratio",
            "0.5",
            "--per_device_train_batch_size",
            "9",
            "--gradient_accumulation_steps",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
        self.assertEqual(args.clasp_canonical_grpo_anchor_scale, 1.0)
        self.assertEqual(args.clasp_canonical_quality_anchor_ratio, 0.5)

    def test_canonical_cli_rejects_deployment_k_equal_to_rollout_b(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_canonical_set",
            "--num_generations",
            "4",
            "--clasp_deployment_set_size",
            "4",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ), self.assertRaisesRegex(ValueError, "deployment K < rollout B"):
            parse_grpo_args()

    def test_canonical_cli_resolves_explicit_policy_length_alpha(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_canonical_set",
            "--num_generations",
            "9",
            "--clasp_deployment_set_size",
            "8",
            "--clasp_policy_length_alpha",
            "0.5",
            "--per_device_train_batch_size",
            "9",
            "--gradient_accumulation_steps",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
            resolved_env = dict(os.environ)
        self.assertEqual(args.clasp_policy_length_alpha, 0.5)
        self.assertEqual(resolved_env["CLASP_POLICY_LENGTH_ALPHA"], "0.5")

    def test_policy_length_alpha_rejected_without_exact_set_credit(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_policy_length_alpha",
            "0.5",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ), self.assertRaisesRegex(ValueError, "only for exact canonical"):
            parse_grpo_args()

    def test_canonical_cli_resolves_dual_recovery_coupling(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_canonical_set",
            "--num_generations",
            "9",
            "--clasp_deployment_set_size",
            "8",
            "--clasp_canonical_dual_recovery_coupling",
            "1.5",
            "--per_device_train_batch_size",
            "9",
            "--gradient_accumulation_steps",
            "1",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ):
            args = parse_grpo_args()
            resolved_env = dict(os.environ)
        self.assertEqual(args.clasp_canonical_dual_recovery_coupling, 1.5)
        self.assertEqual(
            resolved_env["CLASP_CANONICAL_DUAL_RECOVERY_COUPLING"],
            "1.5",
        )

    def test_dual_recovery_coupling_rejected_without_exact_set_credit(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--tasks",
            "tsp",
            "--clasp_canonical_dual_recovery_coupling",
            "1.0",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", argv
        ), self.assertRaisesRegex(ValueError, "only for exact canonical"):
            parse_grpo_args()

    def test_policy_length_alpha_reduction_has_auditable_endpoints(self):
        per_token_loss = torch.tensor(
            [[2.0, 2.0, 99.0, 99.0], [4.0, 4.0, 4.0, 4.0]]
        )
        completion_mask = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )

        grpo_loss, grpo_weights, lengths = reduce_clasp_policy_loss(
            per_token_loss,
            completion_mask,
            length_alpha=0.0,
            max_completion_length=4,
        )
        dr_loss, dr_weights, _ = reduce_clasp_policy_loss(
            per_token_loss,
            completion_mask,
            length_alpha=1.0,
            max_completion_length=4,
        )
        middle_loss, middle_weights, _ = reduce_clasp_policy_loss(
            per_token_loss,
            completion_mask,
            length_alpha=0.5,
            max_completion_length=4,
        )

        torch.testing.assert_close(lengths, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(grpo_weights, torch.ones(2))
        torch.testing.assert_close(grpo_loss, torch.tensor(3.0))
        torch.testing.assert_close(dr_weights, torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(dr_loss, torch.tensor(2.5))
        torch.testing.assert_close(
            middle_weights,
            torch.tensor([math.sqrt(0.5), 1.0]),
        )
        torch.testing.assert_close(
            middle_loss,
            torch.tensor((2.0 * math.sqrt(0.5) + 4.0) / 2.0),
        )

        # Sequence-level TRL ratios intentionally arrive as [batch, 1] and
        # broadcast over the token mask. This must be numerically identical to
        # an explicitly expanded constant per-token loss.
        singleton_loss = torch.tensor([[2.0], [4.0]])
        singleton_grpo, _, _ = reduce_clasp_policy_loss(
            singleton_loss,
            completion_mask,
            length_alpha=0.0,
            max_completion_length=4,
        )
        singleton_sequence, _, _ = reduce_clasp_policy_loss(
            singleton_loss,
            completion_mask,
            length_alpha=1.0,
            max_completion_length=4,
        )
        torch.testing.assert_close(singleton_grpo, grpo_loss)
        torch.testing.assert_close(singleton_sequence, dr_loss)

        integer_mask_loss, integer_mask_weights, integer_mask_lengths = (
            reduce_clasp_policy_loss(
                singleton_loss,
                completion_mask.to(dtype=torch.long),
                length_alpha=0.5,
                max_completion_length=4,
            )
        )
        torch.testing.assert_close(integer_mask_loss, middle_loss)
        torch.testing.assert_close(integer_mask_weights, middle_weights)
        torch.testing.assert_close(integer_mask_lengths, lengths)
        self.assertTrue(integer_mask_lengths.is_floating_point())

    def test_policy_length_alpha_reduction_fails_closed(self):
        loss = torch.ones((1, 3))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            reduce_clasp_policy_loss(
                loss,
                torch.zeros_like(loss),
                length_alpha=0.5,
                max_completion_length=3,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            reduce_clasp_policy_loss(
                loss,
                torch.ones_like(loss),
                length_alpha=0.5,
                max_completion_length=2,
            )
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            reduce_clasp_policy_loss(
                loss,
                torch.ones_like(loss),
                length_alpha=1.1,
                max_completion_length=3,
            )

    def test_policy_length_alpha_endpoint_gradients_match_definitions(self):
        mask = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )
        grpo_tokens = torch.ones((2, 4), requires_grad=True)
        grpo_loss, _, _ = reduce_clasp_policy_loss(
            grpo_tokens,
            mask,
            length_alpha=0.0,
            max_completion_length=4,
        )
        grpo_loss.backward()
        torch.testing.assert_close(
            grpo_tokens.grad,
            torch.tensor(
                [[0.25, 0.25, 0.0, 0.0], [0.125, 0.125, 0.125, 0.125]]
            ),
        )

        sequence_tokens = torch.ones((2, 4), requires_grad=True)
        sequence_loss, _, _ = reduce_clasp_policy_loss(
            sequence_tokens,
            mask,
            length_alpha=1.0,
            max_completion_length=4,
        )
        sequence_loss.backward()
        torch.testing.assert_close(
            sequence_tokens.grad,
            torch.tensor(
                [[0.125, 0.125, 0.0, 0.0], [0.125, 0.125, 0.125, 0.125]]
            ),
        )

    def test_canonical_controller_env_parsers_fail_closed(self):
        with patch.dict(
            os.environ,
            {
                "CLASP_BON_TARGET_FAILURE_RATES": "tsp:not-a-number",
                "CLASP_BON_DUAL_LR": "-0.5",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "not numeric"):
                _strict_env_task_thresholds("CLASP_BON_TARGET_FAILURE_RATES")
            with self.assertRaisesRegex(ValueError, "non-negative"):
                _strict_nonnegative_float_env("CLASP_BON_DUAL_LR", 0.05)

    def test_single_task_clasp_filters_implicit_multitask_warmup_defaults(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--use_clasp",
            "--tasks",
            "tsp",
        ]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch("sys.argv", argv):
            args = parse_grpo_args()
        self.assertEqual(args.clasp_feasibility_warmup_target_map, {"tsp": 0.995})
        self.assertEqual(args.clasp_proper_dual_gradient_mode, "signed")
        self.assertEqual(args.clasp_mixed_adaptive_length_mode, "error")

    def test_single_task_clasp_rejects_explicit_inactive_warmup_target(self):
        argv = [
            "clasp-grpo",
            "--model_path",
            "/tmp/model",
            "--use_clasp",
            "--tasks",
            "tsp",
            "--clasp_feasibility_warmup_targets",
            "cvrp:0.9",
        ]
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True),
            patch("sys.argv", argv),
            self.assertRaisesRegex(ValueError, "active tasks"),
        ):
            parse_grpo_args()

    def test_dense_bon_credit_is_signed_dense_and_tie_safe_in_mixed_group(self):
        advantages, stats = compute_bon_aligned_advantages(
            torch.zeros(4),
            torch.zeros(4),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            num_generations=4,
            group_lambdas=2.0,
            quality_scores=torch.tensor([0.7, 0.7, -100.0, 100.0]),
            success_credit_mode="dense_rank",
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.5, 1.5, -1.5, -1.5])))
        self.assertAlmostEqual(float(advantages.mean().item()), 0.0, places=7)
        self.assertEqual(stats["clasp/bon_success_credit_dense_rank"], 1.0)
        self.assertEqual(stats["clasp/bon_success_credit_exact_loo"], 0.0)
        self.assertEqual(stats["clasp/bon_zero_quality_signal_success_group_rate"], 0.0)
        self.assertEqual(stats["clasp/bon_best_quality_tie_rate"], 1.0)

    def test_dense_bon_credit_uses_tie_aware_quality_midranks(self):
        advantages, stats = compute_bon_aligned_advantages(
            torch.zeros(4),
            torch.zeros(4),
            torch.ones(4),
            num_generations=4,
            quality_scores=torch.tensor([3.0, 2.0, 1.0, 0.0]),
            success_credit_mode="dense_rank",
        )
        self.assertTrue(
            torch.allclose(
                advantages,
                torch.tensor([1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0]),
                atol=1e-6,
            )
        )
        self.assertEqual(stats["clasp/bon_dense_quality_nonzero_rate"], 1.0)

    def test_exact_loo_remains_explicit_sparse_ablation(self):
        advantages, stats = compute_bon_aligned_advantages(
            torch.zeros(4),
            torch.zeros(4),
            torch.ones(4),
            num_generations=4,
            quality_scores=torch.ones(4),
            success_credit_mode="exact_loo",
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))
        self.assertEqual(stats["clasp/bon_success_credit_exact_loo"], 1.0)
        self.assertEqual(stats["clasp/bon_zero_quality_signal_success_group_rate"], 1.0)

    def test_component_duals_use_bounded_convex_recovery_geometry(self):
        raw = torch.tensor([0.2, 2.2])
        component_dual = torch.tensor([0.8, 2.8])
        active = _compose_clasp_recovery_gaps(
            raw,
            component_dual,
            component_dual_scale=2.0,
        )
        self.assertTrue(torch.allclose(active, torch.tensor([0.6, 2.6])))
        self.assertLess(float(active[0]), 1.0)
        self.assertGreater(float(active[1]), 2.0)
        self.assertTrue(
            torch.equal(
                _compose_clasp_recovery_gaps(raw, component_dual, component_dual_scale=0.0),
                raw,
            )
        )

    def test_component_dual_recovery_is_save_resume_symmetric(self):
        def build_manager() -> MultiTaskPrimalDualManager:
            reward = PrimalDualConstrainedReward(
                structure_fn=lambda prompts, completions, **kwargs: [0.0] * len(completions),
                constraint_fn=None,
                objective_fn=lambda prompts, completions, **kwargs: [0.0] * len(completions),
                cost_components_fn=lambda prompts, completions, **kwargs: {
                    "cost_names": ["validity"],
                    "cost_vectors": [[1.0], [0.5]],
                    "cost_masks": [[1.0], [1.0]],
                    "constraint_scores": [0.0, 0.5],
                },
                cost_limit=0.0,
                dual_lr=0.0,
                log_every=0,
            )
            return MultiTaskPrimalDualManager({"tsp": reward})

        original = build_manager()
        original(["p", "p"], ["a", "b"], task_type=["tsp", "tsp"])
        original.wrappers["tsp"].dual_lambdas = [2.0]
        state = original.state_dict()

        restored = build_manager()
        restored.load_state_dict(state)
        restored(["p", "p"], ["a", "b"], task_type=["tsp", "tsp"])
        components = restored.get_last_components()
        active = _compose_clasp_recovery_gaps(
            torch.tensor(components["raw_positive_gap_scores"]),
            torch.tensor(components["dual_aware_recovery_gap_scores"]),
            component_dual_scale=1.0,
        )
        self.assertEqual(restored.wrappers["tsp"].dual_lambdas, [2.0])
        expected = (
            torch.tensor(components["raw_positive_gap_scores"])
            + torch.tensor(components["dual_aware_recovery_gap_scores"])
        ) / 2.0
        self.assertTrue(torch.allclose(active, expected))
        self.assertTrue((active > torch.tensor(components["raw_positive_gap_scores"])).all())

    def test_replace_logged_advantages_supports_trl_deque(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer._logs = {"advantages": deque([10.0, 20.0, 30.0, 40.0])}

        trainer._replace_logged_advantages(torch.tensor([-1.0, 1.0]))

        self.assertEqual(
            list(trainer._logs["advantages"]),
            [10.0, 20.0, -1.0, 1.0],
        )

    def test_strong_dual_primal_standardizes_after_scalarization(self):
        primal = torch.tensor([1.0, 2.0, 10.0, 14.0])
        penalty = torch.tensor([0.0, 0.5, 2.0, 0.0])
        actual = compute_scalarized_group_advantages(
            primal,
            penalty,
            num_generations=2,
        )
        scalarized = (primal - penalty).view(2, 2)
        expected = (
            (scalarized - scalarized.mean(dim=1, keepdim=True))
            / (scalarized.std(dim=1, keepdim=True, unbiased=False) + 1e-4)
        ).reshape(-1)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual.view(2, 2).mean(dim=1), torch.zeros(2), atol=1e-6))

    def test_dual_separate_z_matches_the_exact_formula(self):
        rewards = torch.tensor([1.0, 3.0, 8.0, 11.0], dtype=torch.float32)
        positive_penalties = torch.tensor([0.0, 2.0, 4.0, 5.0], dtype=torch.float32)

        actual = compute_separate_z_group_advantages(
            rewards,
            positive_penalties,
            num_generations=2,
        )

        grouped_rewards = rewards.view(2, 2)
        grouped_penalties = positive_penalties.view(2, 2)
        reward_z = (grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)) / (
            grouped_rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-4
        )
        penalty_z = (
            grouped_penalties - grouped_penalties.mean(dim=1, keepdim=True)
        ) / (grouped_penalties.std(dim=1, keepdim=True, unbiased=False) + 1e-4)
        expected = (reward_z - penalty_z).reshape(-1)

        self.assertEqual(
            DUAL_SEPARATE_Z_FORMULA_ID,
            "group_z_reward_minus_group_z_positive_component_penalty_v1",
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_dual_separate_z_is_exactly_group_centered(self):
        rewards = torch.tensor([0.0, 1.0, 4.0, 3.0, 9.0, 12.0])
        positive_penalties = torch.tensor([0.0, 2.0, 1.0, 7.0, 7.0, 8.0])

        actual = compute_separate_z_group_advantages(
            rewards,
            positive_penalties,
            num_generations=3,
        ).view(2, 3)

        self.assertTrue(torch.allclose(actual.mean(dim=1), torch.zeros(2), atol=1e-6))

    def test_dual_separate_z_zero_spread_terms_contribute_zero(self):
        rewards = torch.tensor([2.0, 2.0, 1.0, 3.0])
        positive_penalties = torch.tensor([4.0, 4.0, 7.0, 7.0])

        actual = compute_separate_z_group_advantages(
            rewards,
            positive_penalties,
            num_generations=2,
        )
        expected_reward_only = torch.tensor([0.0, 0.0, -0.9999, 0.9999])

        self.assertTrue(torch.allclose(actual, expected_reward_only, atol=2e-4))
        both_constant = compute_separate_z_group_advantages(
            torch.ones(4),
            torch.zeros(4),
            num_generations=2,
        )
        self.assertTrue(torch.equal(both_constant, torch.zeros(4)))

    def test_dual_separate_z_is_not_scalarize_then_standardize(self):
        rewards = torch.tensor([0.0, 1.0, 3.0, 9.0])
        positive_penalties = torch.tensor([0.0, 0.2, 1.0, 8.0])

        separate = compute_separate_z_group_advantages(
            rewards,
            positive_penalties,
            num_generations=4,
        )
        scalarized = compute_scalarized_group_advantages(
            rewards,
            positive_penalties,
            num_generations=4,
        )

        self.assertFalse(torch.allclose(separate, scalarized))

    def test_dual_separate_z_cli_is_fail_closed_against_other_credit_modes(self):
        base = ["clasp-grpo", "--model_path", "/tmp/model", "--tasks", "tsp"]
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", base + ["--pd_separate_z"]
        ):
            args = parse_grpo_args()
        self.assertTrue(args.pd_separate_z)
        self.assertFalse(args.pd_group_scalarization)
        self.assertFalse(args.use_clasp)

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", base + ["--pd_separate_z", "--pd_group_scalarization"]
        ), self.assertRaisesRegex(ValueError, "mutually exclusive"):
            parse_grpo_args()

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True), patch(
            "sys.argv", base + ["--pd_separate_z", "--use_clasp"]
        ), self.assertRaisesRegex(ValueError, "mutually exclusive"):
            parse_grpo_args()

    def test_publication_grpo_z_is_exact_group_z_when_all_multipliers_are_zero(self):
        verified_quality = torch.tensor(
            [0.0, 0.1, 0.2, 0.4, 0.8, 0.3, 0.6, 0.9, 0.5],
            dtype=torch.float32,
        )
        zero_lagrangian_penalty = torch.zeros_like(verified_quality)
        actual = compute_scalarized_group_advantages(
            verified_quality,
            zero_lagrangian_penalty,
            num_generations=9,
        )
        expected = (
            verified_quality - verified_quality.mean()
        ) / (verified_quality.std(unbiased=False) + 1e-4)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertAlmostEqual(float(actual.mean().item()), 0.0, places=6)

    def test_strong_dual_primal_is_invariant_to_group_constants(self):
        primal = torch.tensor([1.0, 2.0, 4.0, 8.0])
        penalty = torch.tensor([0.2, 0.1, 1.0, 0.5])
        shifted = primal + torch.tensor([100.0, 100.0, -30.0, -30.0])
        baseline = compute_scalarized_group_advantages(
            primal,
            penalty,
            num_generations=2,
        )
        actual = compute_scalarized_group_advantages(
            shifted,
            penalty,
            num_generations=2,
        )
        self.assertTrue(torch.allclose(actual, baseline, atol=1e-6))

    def test_relative_failed_recovery_switch_rejects_typos(self):
        with patch.dict(
            os.environ,
            {"CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY": "tru"},
        ):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                _strict_env_flag("CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY")

    def test_frontier_cdf_temperature_anneals_geometrically(self):
        trainer = _bare_frontier_cdf_trainer()
        self.assertAlmostEqual(trainer._frontier_cdf_quality_temperature(), 1.0)
        trainer.state.global_step = 25
        self.assertAlmostEqual(trainer._frontier_cdf_quality_temperature(), math.sqrt(0.1))
        trainer.state.global_step = 50
        self.assertAlmostEqual(trainer._frontier_cdf_quality_temperature(), 0.1)
        trainer.state.global_step = 100
        self.assertAlmostEqual(trainer._frontier_cdf_quality_temperature(), 0.1)

    def test_frontier_cdf_trainer_uses_objective_quality_and_fixed_set_credit(self):
        trainer = _bare_frontier_cdf_trainer()
        trainer.model.training = False
        target = trainer.clasp_frontier_cdf_gap_targets["tsp"]
        qualities = torch.tensor([2.0 * math.exp(-3.0 * target), 2.0 * math.exp(-6.0 * target)])
        advantages, stats = trainer._compute_bon_advantages(
            torch.tensor([100.0, -100.0]),
            torch.zeros(2),
            torch.ones(2),
            ["tsp", "tsp"],
            quality_scores=qualities,
        )
        self.assertTrue(torch.isfinite(advantages).all().item())
        self.assertTrue((advantages > 0.0).all().item())
        self.assertGreater(float(advantages[0].item()), float(advantages[1].item()))
        self.assertEqual(stats["clasp/bon_frontier_cdf"], 1.0)
        self.assertIn("frontier_cdf/quality_utility_mean", stats)

    def test_frontier_cdf_failed_group_recovery_ignores_objective_shortcuts(self):
        trainer = _bare_frontier_cdf_trainer()
        trainer.model.training = False
        advantages, stats = trainer._compute_bon_advantages(
            torch.tensor([-100.0, 100.0]),
            torch.tensor([0.9, 0.1]),
            torch.zeros(2),
            ["cvrp", "cvrp"],
            quality_scores=torch.tensor([2.0, 0.0]),
        )
        self.assertGreater(float(advantages[1].item()), float(advantages[0].item()))
        self.assertEqual(stats["frontier_cdf/group_failure_rate"], 1.0)

    def test_frontier_cdf_relative_failed_recovery_is_signed_centered_and_lambda_sensitive(self):
        low_pressure = _bare_frontier_cdf_trainer()
        high_pressure = _bare_frontier_cdf_trainer()
        for trainer in (low_pressure, high_pressure):
            trainer.num_generations = 4
            trainer.model.training = False
            trainer.clasp_frontier_cdf_relative_failed_recovery = True
            trainer.clasp_frontier_cdf_relative_recovery_scale = 0.25
        low_pressure.primal_dual_reward_manager.values = {"cvrp": 0.0}
        high_pressure.primal_dual_reward_manager.values = {"cvrp": 5.0}

        inputs = (
            torch.zeros(4),
            torch.tensor([0.1, 0.3, 1.0, 2.0]),
            torch.zeros(4),
            ["cvrp"] * 4,
        )
        low_advantages, _ = low_pressure._compute_bon_advantages(
            *inputs,
            quality_scores=torch.zeros(4),
        )
        high_advantages, stats = high_pressure._compute_bon_advantages(
            *inputs,
            quality_scores=torch.zeros(4),
        )

        self.assertGreater(float(high_advantages[0].item()), 0.0)
        self.assertLess(float(high_advantages[-1].item()), 0.0)
        self.assertAlmostEqual(float(high_advantages.mean().item()), 0.0, places=6)
        self.assertTrue((high_advantages.abs() > low_advantages.abs()).all().item())
        self.assertEqual(stats["frontier_cdf/relative_failed_recovery"], 1.0)
        self.assertEqual(
            stats["frontier_cdf/relative_recovery_signed_group_rate_failed"],
            1.0,
        )
        self.assertGreater(stats["clasp/bon_controller_advantage_delta"], 0.0)
        self.assertEqual(stats["clasp/bon_controller_pressure_observed"], 1.0)

    def test_frontier_cdf_relative_failed_recovery_does_not_reinforce_ties(self):
        trainer = _bare_frontier_cdf_trainer()
        trainer.num_generations = 4
        trainer.model.training = False
        trainer.clasp_frontier_cdf_relative_failed_recovery = True
        advantages, stats = trainer._compute_bon_advantages(
            torch.zeros(4),
            torch.ones(4),
            torch.zeros(4),
            ["mis"] * 4,
            quality_scores=torch.zeros(4),
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))
        self.assertEqual(stats["frontier_cdf/relative_recovery_zero_rate_failed"], 1.0)
        self.assertEqual(
            stats["frontier_cdf/relative_recovery_signed_group_rate_failed"],
            0.0,
        )

    def test_frontier_cdf_relative_failed_recovery_preserves_successful_group_credit(self):
        baseline = _bare_frontier_cdf_trainer()
        hybrid = _bare_frontier_cdf_trainer()
        baseline.model.training = False
        hybrid.model.training = False
        hybrid.clasp_frontier_cdf_relative_failed_recovery = True
        hybrid.clasp_frontier_cdf_relative_recovery_scale = 0.25
        qualities = torch.tensor([1.9, 0.0])
        common = (
            qualities,
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            ["tsp", "tsp"],
        )
        baseline_advantages, _ = baseline._compute_bon_advantages(
            *common,
            quality_scores=qualities,
        )
        hybrid_advantages, stats = hybrid._compute_bon_advantages(
            *common,
            quality_scores=qualities,
        )
        self.assertTrue(torch.equal(hybrid_advantages, baseline_advantages))
        self.assertEqual(stats["frontier_cdf/relative_failed_recovery"], 1.0)
        self.assertEqual(stats["frontier_cdf/relative_recovery_abs_mean_failed"], 0.0)

    def test_biased_lora_with_beta_requires_explicit_frozen_reference(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.beta = 0.05
        trainer.ref_model = None
        trainer.model = SimpleNamespace(peft_config={"default": SimpleNamespace(bias="lora_only")})
        with self.assertRaisesRegex(RuntimeError, "explicit frozen full-model"):
            trainer._validate_fixed_kl_reference()

        trainer.ref_model = object()
        trainer._validate_fixed_kl_reference()

    def test_unbiased_lora_can_use_disable_adapter_reference(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.beta = 0.05
        trainer.ref_model = None
        trainer.model = SimpleNamespace(peft_config={"default": SimpleNamespace(bias="none")})
        trainer._validate_fixed_kl_reference()

    def test_task_order_uses_global_union_not_rank_local_axis(self):
        rank_zero = _canonical_task_order(
            ["tsp", "op"],
            gathered_task_lists=[["tsp", "op"], ["cvrp"]],
        )
        rank_one = _canonical_task_order(
            ["cvrp"],
            gathered_task_lists=[["tsp", "op"], ["cvrp"]],
        )
        self.assertEqual(rank_zero, ("cvrp", "op", "tsp"))
        self.assertEqual(rank_zero, rank_one)

    def test_complete_group_permutation_recovers_shuffled_groups(self):
        group_ids = torch.tensor([11, 10, 11, 10])
        positions = torch.tensor([1, 0, 0, 1])
        permutation, reason = _complete_group_permutation(
            group_ids,
            positions,
            batch_size=4,
            num_generations=2,
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(permutation.tolist(), [1, 3, 2, 0])

    def test_incomplete_shuffled_group_fails_closed(self):
        permutation, reason = _complete_group_permutation(
            torch.tensor([10, 11]),
            torch.tensor([0, 1]),
            batch_size=2,
            num_generations=2,
        )
        self.assertIsNone(permutation)
        self.assertEqual(reason, "incomplete_prompt_group")

    def test_auxiliary_schedule_validation_only_raises_when_aux_is_enabled(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.num_generations = 8
        trainer.clasp_pairwise_loss_weight = 0.0
        trainer.clasp_elite_imitation_loss_weight = 0.0
        trainer.clasp_multi_elite_imitation_loss_weight = 0.0
        trainer.clasp_aux_grouping_strict = True
        trainer.args = SimpleNamespace(
            per_device_train_batch_size=2,
            steps_per_generation=4,
        )
        # The shipped default has every auxiliary disabled and must remain valid.
        trainer._validate_group_auxiliary_schedule()

        trainer.clasp_pairwise_loss_weight = 0.1
        with self.assertRaisesRegex(ValueError, "--steps_per_generation 1"):
            trainer._validate_group_auxiliary_schedule()

        trainer.args = SimpleNamespace(
            per_device_train_batch_size=8,
            steps_per_generation=1,
        )
        trainer._validate_group_auxiliary_schedule()

    def test_mixed_task_adaptive_length_fails_closed_or_logs_global_max_ablation(self):
        class FakeEstimator:
            @staticmethod
            def budget(task):
                return {"tsp": 128, "cvrp": 256}[task]

        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.clasp_enabled = True
        trainer.clasp_adaptive_length = FakeEstimator()
        trainer.clasp_mixed_adaptive_length_mode = "error"
        trainer.max_completion_length = 512
        trainer.args = SimpleNamespace(max_completion_length=512)
        trainer.generation_config = SimpleNamespace(max_new_tokens=512)

        with self.assertRaisesRegex(RuntimeError, "unequal per-row limits"):
            trainer._set_clasp_generation_budget(["tsp", "cvrp"])

        trainer.clasp_mixed_adaptive_length_mode = "global_max"
        restorers = trainer._set_clasp_generation_budget(["tsp", "cvrp"])
        self.assertEqual(trainer.max_completion_length, 256)
        self.assertEqual(trainer.args.max_completion_length, 256)
        self.assertEqual(trainer.generation_config.max_new_tokens, 256)
        self.assertEqual(
            trainer._clasp_last_generation_budget_stats,
            {
                "clasp/adaptive_budget_enabled": 1.0,
                "clasp/adaptive_budget_requested_min": 128.0,
                "clasp/adaptive_budget_requested_max": 256.0,
                "clasp/adaptive_budget_effective": 256.0,
                "clasp/adaptive_budget_mixed_unequal": 1.0,
                "clasp/adaptive_budget_global_max_ablation": 1.0,
            },
        )
        trainer._restore_generation_budget(restorers)
        self.assertEqual(trainer.max_completion_length, 512)
        self.assertEqual(trainer.args.max_completion_length, 512)
        self.assertEqual(trainer.generation_config.max_new_tokens, 512)

    @staticmethod
    def _bare_self_replay_trainer(
        manager: MultiTaskPrimalDualManager,
    ) -> PrimalDualGRPOTrainer:
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.clasp_enabled = True
        trainer.clasp_self_replay_enabled = True
        trainer.clasp_self_replay_max_tokens = 64
        trainer.clasp_self_replay_max_keys = 4
        trainer.clasp_self_replay_max_per_key = 2
        trainer.primal_dual_reward_manager = manager
        trainer._clasp_self_replay_buffer = OrderedDict()
        trainer._clasp_self_replay_rng = random.Random(17)
        trainer._clasp_self_replay_state_loaded = False
        trainer._clasp_self_replay_resume_load_serial = -1
        return trainer

    def test_self_replay_buffer_and_rng_round_trip_through_manager_state(self):
        original_manager = MultiTaskPrimalDualManager({})
        original = self._bare_self_replay_trainer(original_manager)
        original._clasp_self_replay_buffer["cvrp:abc"] = [
            {"text": "Routes: [[0, 1, 0]]", "score": 2.5}
        ]
        original._clasp_self_replay_rng.random()
        original._sync_self_replay_state_to_manager()
        serialized = json.loads(json.dumps(original_manager.state_dict()))
        expected_next_draw = original._clasp_self_replay_rng.random()

        restored_manager = MultiTaskPrimalDualManager({})
        restored_manager.load_state_dict(serialized, from_checkpoint=True)
        restored = self._bare_self_replay_trainer(restored_manager)
        restored._ensure_self_replay_state_loaded()

        self.assertEqual(restored._clasp_self_replay_buffer, original._clasp_self_replay_buffer)
        self.assertAlmostEqual(restored._clasp_self_replay_rng.random(), expected_next_draw)
        self.assertIn("__clasp_trainer__", restored_manager.state_dict())

    def test_self_replay_resume_fails_closed_without_compatible_runtime_state(self):
        missing_manager = MultiTaskPrimalDualManager({})
        missing_manager.mark_resume_checkpoint_requested(has_trainer_state=False)
        missing = self._bare_self_replay_trainer(missing_manager)
        with self.assertRaisesRegex(RuntimeError, "silently reset"):
            missing._ensure_self_replay_state_loaded()

        source_manager = MultiTaskPrimalDualManager({})
        source = self._bare_self_replay_trainer(source_manager)
        source._sync_self_replay_state_to_manager()
        incompatible = source_manager.state_dict()
        incompatible["__clasp_trainer__"]["schema_version"] = 999
        restored_manager = MultiTaskPrimalDualManager({})
        restored_manager.load_state_dict(incompatible, from_checkpoint=True)
        restored = self._bare_self_replay_trainer(restored_manager)
        with self.assertRaisesRegex(RuntimeError, "schema_version=1"):
            restored._ensure_self_replay_state_loaded()

    def test_validation_warmup_state_uses_pass_at_k_not_candidate_rate(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.clasp_enabled = True
        trainer.num_generations = 4
        trainer.primal_dual_reward_manager = SimpleNamespace(wrappers={})
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            gather=lambda value: value,
        )
        trainer._metrics = {
            "eval": defaultdict(list),
            "train": defaultdict(list),
        }
        trainer._clasp_validation_task_feasibility = {}
        trainer._clasp_validation_task_candidate_feasibility = {}
        trainer._clasp_validation_group_success_sums = {}
        trainer._clasp_validation_group_counts = {}
        trainer._clasp_validation_candidate_success_sums = {}
        trainer._clasp_validation_candidate_counts = {}
        trainer._clasp_last_generation_mode = "train"

        trainer._update_clasp_validation_feasibility(
            "eval",
            {
                "task_types": ["cvrp"] * 4 + ["mis"] * 4,
                "feasibility_flags": [0.0, 0.0, 0.0, 1.0] + [0.0] * 4,
            },
        )
        self.assertEqual(trainer._clasp_validation_task_feasibility["cvrp"], 1.0)
        self.assertEqual(
            trainer._clasp_validation_task_candidate_feasibility["cvrp"],
            0.25,
        )
        self.assertEqual(trainer._clasp_validation_task_feasibility["mis"], 0.0)
        self.assertEqual(
            trainer._metrics["eval"]["constrained_grpo/task/cvrp/pass_at_k_feasibility_rate"],
            [1.0],
        )

    def test_canonical_validation_uses_pass_at_deployment_k_u_statistic(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.clasp_enabled = True
        trainer.clasp_canonical_set = True
        trainer.clasp_deployment_set_size = 2
        trainer.num_generations = 4
        trainer.primal_dual_reward_manager = SimpleNamespace(wrappers={})
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            gather=lambda value: value,
        )
        trainer._metrics = {"eval": defaultdict(list), "train": defaultdict(list)}
        trainer._clasp_validation_task_feasibility = {}
        trainer._clasp_validation_task_candidate_feasibility = {}
        trainer._clasp_validation_group_success_sums = {}
        trainer._clasp_validation_group_counts = {}
        trainer._clasp_validation_candidate_success_sums = {}
        trainer._clasp_validation_candidate_counts = {}
        trainer._clasp_last_generation_mode = "train"

        trainer._update_clasp_validation_feasibility(
            "eval",
            {
                "task_types": ["cvrp"] * 4,
                "feasibility_flags": [0.0, 0.0, 0.0, 1.0],
            },
        )
        # Three of six size-2 subsets fail, so pass@2 is 1 - 3/6 = 0.5.
        self.assertEqual(trainer._clasp_validation_task_feasibility["cvrp"], 0.5)
        self.assertEqual(
            trainer._metrics["eval"][
                "constrained_grpo/task/cvrp/pass_at_2_feasibility_rate"
            ],
            [0.5],
        )

    def test_checkpoint_gate_reads_pass_at_k_and_ignores_candidate_feasibility(self):
        callback = ClaspPassAtKValidationGateCheckpointCallback({"cvrp": 0.9})
        state = SimpleNamespace(
            log_history=[
                {
                    "eval_constrained_grpo/task/cvrp/candidate_feasibility_rate": 1.0,
                    "eval_constrained_grpo/task/cvrp/pass_at_k_feasibility_rate": 0.8,
                }
            ]
        )
        control = SimpleNamespace(should_save=True)
        callback._gate_save(state, control)
        self.assertFalse(control.should_save)

    def test_generation_metadata_survives_arbitrary_row_permutation(self):
        trainer = object.__new__(PrimalDualGRPOTrainer)
        trainer.clasp_enabled = True
        trainer.num_generations = 2
        trainer._clasp_generation_serial = 0
        trainer.accelerator = SimpleNamespace(process_index=0)
        output = {
            "prompt_ids": torch.tensor([[1], [1], [2], [2]]),
            "completion_ids": torch.tensor([[10], [11], [20], [21]]),
        }
        trainer._attach_clasp_group_metadata(
            output,
            {"task_types": ["tsp", "tsp", "cvrp", "cvrp"]},
            4,
        )
        shuffle = torch.tensor([2, 0, 3, 1])
        permutation, reason = _complete_group_permutation(
            output["clasp_group_ids"][shuffle],
            output["clasp_group_positions"][shuffle],
            batch_size=4,
            num_generations=2,
        )
        self.assertEqual(reason, "ok")
        # Applying the recovered order to the shuffled completion rows restores
        # the two original prompt groups in position order.
        restored = output["completion_ids"][shuffle][permutation].reshape(-1).tolist()
        self.assertEqual(restored, [10, 11, 20, 21])

    def test_generation_metadata_supports_one_k_group_spanning_four_ranks(self):
        rank_group_ids = []
        rank_positions = []
        for rank in range(4):
            trainer = object.__new__(PrimalDualGRPOTrainer)
            trainer.clasp_enabled = True
            trainer.num_generations = 8
            trainer._clasp_generation_serial = 0
            trainer.accelerator = SimpleNamespace(process_index=rank, num_processes=4)
            output = {
                "prompt_ids": torch.tensor([[7], [7]]),
                "completion_ids": torch.tensor([[rank * 2], [rank * 2 + 1]]),
            }
            trainer._attach_clasp_group_metadata(
                output,
                {"task_types": ["cvrp", "cvrp"]},
                2,
            )
            rank_group_ids.append(output["clasp_group_ids"])
            rank_positions.append(output["clasp_group_positions"])

        self.assertEqual(torch.cat(rank_group_ids).tolist(), [0] * 8)
        self.assertEqual(torch.cat(rank_positions).tolist(), list(range(8)))

    def test_bon_controller_is_frozen_until_gradient_sync_commit(self):
        trainer = _bare_bon_trainer()
        advantages, first_stats = trainer._compute_bon_advantages(
            torch.tensor([0.1, 0.2]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 0.0]),
            ["cvrp", "cvrp"],
        )
        self.assertTrue(torch.isfinite(advantages).all().item())
        self.assertGreater(first_stats["clasp/bon_controller_advantage_delta"], 0.0)
        self.assertEqual(first_stats["clasp/bon_lambda_used/cvrp"], 1.0)
        self.assertEqual(first_stats["clasp/bon_lambda_next/cvrp"], 1.0)
        self.assertEqual(trainer.primal_dual_reward_manager.values, {})

        _, second_stats = trainer._compute_bon_advantages(
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            ["cvrp", "cvrp"],
        )
        self.assertEqual(second_stats["clasp/bon_lambda_used/cvrp"], 1.0)
        self.assertEqual(second_stats["clasp/bon_lambda_next/cvrp"], 1.0)
        self.assertEqual(trainer.primal_dual_reward_manager.values, {})
        self.assertEqual(trainer._clasp_bon_pending_group_counts, {"cvrp": 2})
        self.assertEqual(trainer._clasp_bon_pending_failure_sums, {"cvrp": 1.0})

        commit_stats = trainer._commit_pending_bon_controller_update()
        # Combined failure rate is 1/2. One signed step is therefore
        # 1.0 + 0.5 * (0.5 - 0.25) = 1.125. Updating once per generation
        # call would yield a different result and this regression would fail.
        self.assertAlmostEqual(
            trainer.primal_dual_reward_manager.values["cvrp"],
            1.125,
            places=7,
        )
        self.assertEqual(commit_stats["clasp/bon_committed_group_count"], 2.0)
        self.assertEqual(commit_stats["clasp/bon_committed_failure_rate/cvrp"], 0.5)
        self.assertEqual(trainer._clasp_bon_pending_group_counts, {})
        self.assertEqual(trainer._clasp_bon_pending_failure_sums, {})

    def test_canonical_trainer_queues_failure_at_k_and_emits_k_scaled_credit(self):
        trainer = _bare_canonical_set_trainer()
        objective = torch.tensor(
            [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
        )
        violation = torch.tensor([0.5, 0.4, 0.3, 0.0, 0.8, 0.7, 0.6, 0.5])
        feasibility = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        tasks = ["cvrp"] * 8

        coefficients, stats = trainer._compute_canonical_set_advantages(
            objective,
            violation,
            feasibility,
            tasks,
        )
        expected = compute_canonical_clasp_set_advantages(
            objective,
            violation,
            feasibility,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=torch.tensor([1.0, 1.0]),
            recovery_scale=0.7,
            recovery_temperature=0.2,
            target_failure_rates=torch.tensor([0.25, 0.25]),
        )
        torch.testing.assert_close(
            coefficients,
            expected.mean_reduced_policy_coefficients,
        )
        self.assertEqual(trainer._clasp_bon_pending_group_counts, {"cvrp": 2})
        # First group has failure@2=3/6; the all-infeasible second group has 1.
        self.assertAlmostEqual(
            trainer._clasp_bon_pending_failure_sums["cvrp"],
            1.5,
            places=7,
        )
        self.assertAlmostEqual(stats["clasp/canonical_failure_at_k"], 0.75)
        self.assertAlmostEqual(stats["clasp/canonical_full_rollout_failure"], 0.5)
        self.assertEqual(stats["clasp/canonical_subset_count"], 6.0)
        self.assertGreater(stats["clasp/canonical_informative_group_fraction"], 0.0)
        self.assertIn("clasp/canonical_zero_signal_group_fraction", stats)
        self.assertEqual(stats["clasp/canonical_feasible_candidate_fraction"], 0.125)
        self.assertLess(
            stats["clasp/canonical_component_decomposition_residual_abs_max"],
            1e-5,
        )
        self.assertLess(
            stats["clasp/canonical_coefficient_group_mean_abs_max"],
            1e-5,
        )

        commit = trainer._commit_pending_bon_controller_update()
        # lambda <- 1 + 0.5 * (0.75 - 0.25) = 1.25.
        self.assertAlmostEqual(
            trainer.primal_dual_reward_manager.values["cvrp"],
            1.25,
            places=7,
        )
        self.assertAlmostEqual(
            commit["clasp/bon_committed_failure_rate/cvrp"],
            0.75,
        )
        self.assertAlmostEqual(
            commit["clasp/bon_committed_signed_residual/cvrp"],
            0.5,
        )
        self.assertAlmostEqual(
            commit["clasp/bon_committed_requested_delta/cvrp"],
            0.25,
        )
        self.assertAlmostEqual(
            commit["clasp/bon_committed_actual_delta/cvrp"],
            0.25,
        )

    def test_canonical_dual_recovery_arm_scales_only_failed_set_recovery(self):
        trainer = _bare_canonical_set_trainer()
        trainer.model.training = False
        trainer.clasp_canonical_dual_recovery_coupling = 1.5
        objective = torch.zeros(4)
        violation = torch.tensor([0.1, 0.3, 0.6, 0.9])
        feasibility = torch.zeros(4)

        coefficients, stats = trainer._compute_canonical_set_advantages(
            objective,
            violation,
            feasibility,
            ["cvrp"] * 4,
        )
        expected_scale = 0.7 + 1.5 * 1.0
        expected = compute_canonical_clasp_set_advantages(
            objective,
            violation,
            feasibility,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=torch.tensor([1.0]),
            recovery_scale=expected_scale,
            recovery_temperature=0.2,
            target_failure_rates=torch.tensor([0.25]),
        )
        torch.testing.assert_close(
            coefficients,
            expected.mean_reduced_policy_coefficients,
        )
        self.assertAlmostEqual(
            stats["clasp/canonical_effective_recovery_scale_mean"],
            expected_scale,
        )
        self.assertGreater(
            stats["clasp/canonical_dual_recovery_coefficient_abs_mean"],
            0.0,
        )
        # Binary failure@K remains the controller observation; the new arm is
        # intentionally identified as auxiliary recovery pressure.
        self.assertEqual(stats["clasp/canonical_failure_at_k"], 1.0)

    def test_strong_canonical_combines_scaled_exact_and_dense_credit(self):
        trainer = _bare_canonical_set_trainer()
        trainer.model.training = False
        trainer.clasp_canonical_exact_scale = 0.25
        trainer.clasp_canonical_dense_quality_scale = 0.5
        trainer.clasp_canonical_dense_feasibility_scale = 1.5
        trainer.clasp_canonical_dense_violation_scale = 0.5
        trainer.clasp_canonical_dense_recovery_scale = 1.5
        objective = torch.tensor([0.9, 0.7, 0.0, 0.0])
        exact_violation = torch.tensor([0.0, 0.0, 0.8, 0.4])
        dense_violation = torch.tensor([0.0, 0.0, 4.0, 1.0])
        feasibility = torch.tensor([1.0, 1.0, 0.0, 0.0])

        coefficients, stats = trainer._compute_canonical_set_advantages(
            objective,
            exact_violation,
            feasibility,
            ["cvrp"] * 4,
            dense_violation,
        )
        exact = compute_canonical_clasp_set_advantages(
            objective,
            exact_violation,
            feasibility,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=1.0,
            recovery_scale=0.7,
            recovery_temperature=0.2,
            target_failure_rates=0.25,
        )
        dense = compute_dense_clasp_auxiliary_advantages(
            objective,
            dense_violation,
            feasibility,
            rollout_batch_size=4,
            group_lambdas=1.0,
            quality_scale=0.5,
            feasibility_scale=1.5,
            violation_scale=0.5,
            recovery_scale=1.5,
        )
        torch.testing.assert_close(
            coefficients,
            0.25 * exact.mean_reduced_policy_coefficients + dense.advantages,
        )
        self.assertEqual(stats["clasp/canonical_dense_auxiliary"], 1.0)
        self.assertGreater(stats["clasp/canonical_dense_feasibility_abs_mean"], 0.0)
        self.assertGreater(stats["clasp/canonical_dense_violation_abs_mean"], 0.0)
        self.assertLess(
            stats["clasp/canonical_coefficient_group_mean_abs_max"],
            1e-5,
        )

    def test_canonical_grpo_anchor_preserves_classic_credit_and_adds_exact_residual(self):
        trainer = _bare_canonical_set_trainer()
        trainer.model.training = False
        trainer.clasp_canonical_exact_scale = 0.25
        trainer.clasp_canonical_grpo_anchor_scale = 1.0
        objective = torch.tensor([0.9, 0.7, 0.0, 0.0])
        violation = torch.tensor([0.0, 0.0, 0.8, 0.4])
        feasibility = torch.tensor([1.0, 1.0, 0.0, 0.0])

        coefficients, stats = trainer._compute_canonical_set_advantages(
            objective,
            violation,
            feasibility,
            ["cvrp"] * 4,
        )
        exact = compute_canonical_clasp_set_advantages(
            objective,
            violation,
            feasibility,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=1.0,
            recovery_scale=0.7,
            recovery_temperature=0.2,
            target_failure_rates=0.25,
        )
        classic_anchor = compute_scalarized_group_advantages(
            objective,
            torch.zeros_like(objective),
            4,
        )
        torch.testing.assert_close(
            coefficients,
            classic_anchor + 0.25 * exact.mean_reduced_policy_coefficients,
        )
        self.assertEqual(stats["clasp/canonical_grpo_anchor_scale"], 1.0)
        self.assertGreater(stats["clasp/canonical_grpo_anchor_abs_mean"], 0.0)
        self.assertEqual(
            stats["clasp/canonical_grpo_anchor_signal_group_fraction"],
            1.0,
        )
        self.assertLess(
            stats["clasp/canonical_component_decomposition_residual_abs_max"],
            1e-5,
        )

    def test_norm_matched_quality_residual_targets_half_anchor_rms(self):
        trainer = _bare_canonical_set_trainer()
        trainer.model.training = False
        trainer.clasp_canonical_exact_scale = 1.0
        trainer.clasp_canonical_grpo_anchor_scale = 1.0
        trainer.clasp_canonical_quality_anchor_ratio = 0.5
        objective = torch.tensor([0.9, 0.7, 0.0, 0.0])
        violation = torch.tensor([0.0, 0.0, 0.8, 0.4])
        feasibility = torch.tensor([1.0, 1.0, 0.0, 0.0])

        coefficients, stats = trainer._compute_canonical_set_advantages(
            objective,
            violation,
            feasibility,
            ["cvrp"] * 4,
        )
        exact = compute_canonical_clasp_set_advantages(
            objective,
            violation,
            feasibility,
            rollout_batch_size=4,
            deployment_set_size=2,
            group_lambdas=1.0,
            recovery_scale=0.7,
            recovery_temperature=0.2,
            target_failure_rates=0.25,
        )
        anchor = compute_scalarized_group_advantages(
            objective,
            torch.zeros_like(objective),
            4,
        )
        raw_quality = 2.0 * exact.quality_advantages
        raw_constraint = exact.mean_reduced_policy_coefficients - raw_quality
        rescale = min(
            4.0,
            float(torch.sqrt(anchor.square().mean()) / torch.sqrt(raw_quality.square().mean())),
        )
        expected = anchor + 0.5 * rescale * raw_quality + raw_constraint
        torch.testing.assert_close(coefficients, expected)
        self.assertEqual(stats["clasp/canonical_quality_anchor_ratio"], 0.5)
        self.assertEqual(stats["clasp/canonical_quality_norm_active"], 1.0)
        self.assertAlmostEqual(stats["clasp/canonical_quality_norm_rescale"], rescale)
        self.assertAlmostEqual(
            float(torch.sqrt((0.5 * rescale * raw_quality).square().mean())),
            0.5 * float(torch.sqrt(anchor.square().mean())),
            places=6,
        )
        self.assertAlmostEqual(
            stats["clasp/canonical_quality_to_anchor_rms_ratio"],
            0.5,
            places=6,
        )
        self.assertEqual(
            stats["clasp/canonical_quality_coefficient_is_exact_raw"],
            0.0,
        )
        self.assertLess(
            stats["clasp/canonical_component_decomposition_residual_abs_max"],
            1e-5,
        )

    def test_frontier_cdf_commit_uses_the_signed_bucket_controller(self):
        trainer = _bare_frontier_cdf_trainer()
        trainer._compute_bon_advantages(
            torch.zeros(2),
            torch.tensor([1.0, 0.5]),
            torch.zeros(2),
            ["cvrp", "cvrp"],
            quality_scores=torch.zeros(2),
        )
        trainer._compute_bon_advantages(
            torch.ones(2),
            torch.zeros(2),
            torch.ones(2),
            ["cvrp", "cvrp"],
            quality_scores=torch.ones(2),
        )
        stats = trainer._commit_pending_bon_controller_update()
        self.assertAlmostEqual(
            trainer.primal_dual_reward_manager.values["cvrp"],
            1.125,
            places=7,
        )
        self.assertEqual(stats["frontier_cdf/dual_active_bucket_count"], 1.0)
        self.assertEqual(stats["clasp/bon_committed_group_count"], 2.0)

    def test_bon_rejects_mixed_task_prompt_group(self):
        trainer = _bare_bon_trainer()
        with self.assertRaisesRegex(RuntimeError, "one task label"):
            trainer._compute_bon_advantages(
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([1.0, 1.0]),
                ["tsp", "cvrp"],
            )

    def test_missing_gathered_tasks_infers_only_single_configured_task(self):
        trainer = _bare_bon_trainer()
        trainer.primal_dual_reward_manager.wrappers = {"TSP": object()}
        self.assertEqual(
            trainer._bon_group_task_names(None, 4),
            ["tsp", "tsp"],
        )
        trainer.primal_dual_reward_manager.wrappers = {
            "tsp": object(),
            "cvrp": object(),
        }
        with self.assertRaisesRegex(RuntimeError, "task labels are missing"):
            trainer._bon_group_task_names(None, 4)

    def test_tiny_valid_controller_pressure_never_causes_runtime_abort(self):
        trainer = _bare_bon_trainer()
        trainer.model.training = False
        trainer.primal_dual_reward_manager.values = {"tsp": 5e-8}
        advantages, stats = trainer._compute_bon_advantages(
            torch.tensor([2.0, 1.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            ["tsp", "tsp"],
            quality_scores=torch.tensor([2.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(advantages).all().item())
        self.assertIn("clasp/bon_controller_advantage_delta", stats)

    def test_successful_group_quality_uses_objective_not_structure_mixed_primal(self):
        first = _bare_bon_trainer()
        second = _bare_bon_trainer()
        first.model.training = False
        second.model.training = False
        objective_quality = torch.tensor([0.1, 0.9])
        common = (
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            ["tsp", "tsp"],
        )
        advantages_a, _ = first._compute_bon_advantages(
            torch.tensor([100.0, -100.0]),
            common[0],
            common[1],
            common[2],
            quality_scores=objective_quality,
        )
        advantages_b, _ = second._compute_bon_advantages(
            torch.tensor([-7.0, 12.0]),
            common[0],
            common[1],
            common[2],
            quality_scores=objective_quality,
        )
        self.assertTrue(torch.allclose(advantages_a, advantages_b, atol=1e-7))
        self.assertEqual(int(advantages_a.argmax().item()), 1)
        self.assertLess(float(advantages_a[0].item()), 0.0)
        self.assertGreater(float(advantages_a[1].item()), 0.0)
        self.assertAlmostEqual(float(advantages_a.mean().item()), 0.0, places=7)

    def test_objective_quality_is_exposed_independently_of_structure(self):
        reward = PrimalDualConstrainedReward(
            structure_fn=lambda prompts, completions, **kwargs: [10.0, -10.0],
            constraint_fn=None,
            objective_fn=lambda prompts, completions, **kwargs: [0.2, 0.8],
            cost_components_fn=lambda prompts, completions, **kwargs: {
                "cost_names": ["validity"],
                "cost_vectors": [[0.0], [0.0]],
                "cost_masks": [[1.0], [1.0]],
                "constraint_scores": [1.0, 1.0],
            },
            cost_limit=0.0,
            objective_weight=1.0,
            structure_weight=1.0,
            dual_lr=0.0,
            log_every=0,
        )
        manager = MultiTaskPrimalDualManager({"tsp": reward})
        manager(
            ["p", "p"],
            ["a", "b"],
            task_type=["tsp", "tsp"],
        )
        components = manager.get_last_components()
        self.assertEqual(components["objective_scores"], [0.2, 0.8])
        self.assertEqual(components["structure_scores"], [10.0, -10.0])
        self.assertEqual(components["max_constraint_violation_scores"], [0.0, 0.0])
        self.assertNotEqual(components["primal_scores"], components["objective_scores"])

    def test_bon_lambdas_round_trip_through_manager_checkpoint_state(self):
        manager = MultiTaskPrimalDualManager({})
        manager.set_clasp_bon_controller_state(
            {
                "lambdas": {"CVRP": 1.25},
                "failure_ema": {"CVRP": 0.125},
                "observed_groups": {"CVRP": 64},
            }
        )
        state = manager.state_dict()
        restored = MultiTaskPrimalDualManager({})
        restored.load_state_dict(state)
        self.assertEqual(restored.get_clasp_bon_lambdas(), {"cvrp": 1.25})
        self.assertEqual(
            restored.get_clasp_bon_controller_state(),
            {
                "lambdas": {"cvrp": 1.25},
                "failure_ema": {"cvrp": 0.125},
                "observed_groups": {"cvrp": 64},
            },
        )

    def test_instance_overlap_ignores_output_and_json_formatting(self):
        train = Dataset.from_list(
            [
                {
                    "instance_json": '{"nodes":[1,2], "budget":3}',
                    "input": "train rendering",
                    "output": "train solver label",
                },
                {
                    "instance_json": '{"nodes":[3,4],"budget":5}',
                    "input": "clean rendering",
                    "output": "another label",
                },
            ]
        )
        external_eval = Dataset.from_list(
            [
                {
                    "instance_json": '{ "budget" : 3, "nodes" : [1, 2] }',
                    "input": "different prompt rendering",
                    "output": "different solver label",
                }
            ]
        )
        cleaned = ensure_no_instance_overlap(
            task="op",
            train_dataset=train,
            eval_dataset=external_eval,
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(
            stable_instance_signature(dict(cleaned[0])),
            stable_instance_signature(dict(train[1])),
        )

    def test_eval_from_train_decontaminates_before_cap_and_can_disable_internal_eval(self):
        with tempfile.TemporaryDirectory() as data_root:
            train_dir = os.path.join(data_root, "op", "train")
            eval_dir = os.path.join(data_root, "op", "eval")
            os.makedirs(train_dir)
            os.makedirs(eval_dir)
            train_rows = [
                {
                    "instruction": "solve op",
                    "input": f"training instance {idx}",
                    "output": f"solution {idx}",
                    "instance_json": {"instance_id": idx},
                }
                for idx in range(3)
            ]
            eval_rows = [
                {
                    "instruction": "evaluate op",
                    "input": "external rendering",
                    "output": "external solution",
                    "instance_json": '{ "instance_id" : 0 }',
                }
            ]
            with open(os.path.join(train_dir, "train.json"), "w", encoding="utf-8") as handle:
                json.dump(train_rows, handle)
            with open(os.path.join(eval_dir, "test.json"), "w", encoding="utf-8") as handle:
                json.dump(eval_rows, handle)

            train_dataset, internal_eval = load_mixed_grpo_dataset(
                tasks=["op"],
                data_root=data_root,
                eval_ratio=0.0,
                eval_from_train_split=True,
                seed=42,
                max_samples_per_task=2,
                max_eval_samples_per_task=0,
                min_train_samples_per_task=2,
                min_eval_samples_per_task=0,
                prefer_subset_files=False,
            )
            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(internal_eval), 0)
            external_signature = stable_instance_signature(eval_rows[0])
            self.assertNotIn(
                external_signature,
                {stable_instance_signature(dict(row)) for row in train_dataset},
            )

    def test_train_derived_eval_split_honors_explicit_eval_cap(self):
        with tempfile.TemporaryDirectory() as data_root:
            train_dir = os.path.join(data_root, "op", "train")
            os.makedirs(train_dir)
            train_rows = [
                {
                    "instruction": "solve op",
                    "input": f"training instance {idx}",
                    "output": f"solution {idx}",
                    "instance_json": {"instance_id": idx},
                }
                for idx in range(20)
            ]
            with open(os.path.join(train_dir, "train.json"), "w", encoding="utf-8") as handle:
                json.dump(train_rows, handle)

            grpo_train, grpo_eval = load_mixed_grpo_dataset(
                tasks=["op"],
                data_root=data_root,
                eval_ratio=0.5,
                eval_from_train_split=True,
                seed=42,
                max_samples_per_task=20,
                max_eval_samples_per_task=3,
                min_train_samples_per_task=10,
                min_eval_samples_per_task=3,
                prefer_subset_files=False,
            )
            self.assertEqual(len(grpo_train), 10)
            self.assertEqual(len(grpo_eval), 3)

            sft_train, sft_eval = load_mixed_sft_dataset(
                tasks=["op"],
                data_root=data_root,
                eval_ratio=0.5,
                eval_from_train_split=True,
                seed=42,
                eos_token="<eos>",
                max_samples_per_task=20,
                max_eval_samples_per_task=3,
                min_train_samples_per_task=10,
                min_eval_samples_per_task=3,
                prefer_subset_files=False,
            )
            self.assertEqual(len(sft_train), 10)
            self.assertEqual(len(sft_eval), 3)

    def test_split_dataset_zero_ratio_keeps_every_row_for_training(self):
        dataset = Dataset.from_dict({"row_id": [0, 1, 2]})
        split = split_dataset(dataset, eval_ratio=0.0, seed=42)
        self.assertEqual(split["train"]["row_id"], [0, 1, 2])
        self.assertEqual(len(split["test"]), 0)


def test_canonical_anchor_and_scalar_baseline_share_opt_in_trl_std():
    objective = torch.tensor([0.0, 0.5, 0.0, 0.9])
    violation = torch.tensor([0.8, 0.0, 0.3, 0.0])
    feasibility = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pure = _bare_canonical_set_trainer()
    anchored = _bare_canonical_set_trainer()
    anchored.group_std_correction = 1
    anchored.clasp_canonical_grpo_anchor_scale = 1.0
    pure_credit, _ = pure._compute_canonical_set_advantages(
        objective, violation, feasibility, ["tsp"] * 4
    )
    actual, stats = anchored._compute_canonical_set_advantages(
        objective, violation, feasibility, ["tsp"] * 4
    )
    stock_trl_credit = (objective - objective.mean()) / (objective.std() + 1e-4)
    torch.testing.assert_close(actual - pure_credit, stock_trl_credit)
    assert stats["clasp/canonical_grpo_anchor_std_correction"] == 1.0
    torch.testing.assert_close(
        actual - pure_credit,
        compute_scalarized_group_advantages(
            objective, torch.zeros_like(objective), 4, std_correction=1
        ),
    )


if __name__ == "__main__":
    unittest.main()
