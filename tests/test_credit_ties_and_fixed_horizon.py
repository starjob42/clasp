"""Permutation and gradient regressions for legacy credit/loss alternatives."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from clasp.rewards.clasp_math import (
    _centered_descending_rank_scores,
    _centered_elite_scores,
    _masked_descending_rank_scores,
    compute_pareto_anchored_advantages,
)
from clasp.trainers.primal_dual_grpo import (
    PrimalDualGRPOTrainer,
    compute_scalarized_group_advantages,
)


def test_legacy_rank_and_elite_do_not_reward_identical_candidates():
    values = torch.ones((2, 9))
    assert torch.equal(_centered_descending_rank_scores(values, 1e-4), torch.zeros_like(values))
    elite, count = _centered_elite_scores(values, 0.125, 1e-4)
    assert count == 2
    assert torch.equal(elite, torch.zeros_like(values))


@pytest.mark.parametrize("kind", ["rank", "elite", "masked"])
def test_partial_ties_have_equal_credit_and_follow_candidate_permutations(kind):
    values = torch.tensor([[4.0, 3.0, 3.0, 1.0]])
    mask = torch.tensor([[False, True, True, True]])
    permutation = torch.tensor([2, 3, 0, 1])

    def credit(v, m):
        if kind == "rank":
            return _centered_descending_rank_scores(v, 1e-4)
        if kind == "elite":
            return _centered_elite_scores(v, 0.5, 1e-4)[0]
        return _masked_descending_rank_scores(v, m, 1e-4)

    original = credit(values, mask)
    permuted = credit(values[:, permutation], mask[:, permutation])
    assert original[0, 1] == original[0, 2]
    assert original[0, 1] > original[0, 3]
    torch.testing.assert_close(permuted, original[:, permutation])
    torch.testing.assert_close(original.sum(1), torch.zeros(1), atol=1e-6, rtol=0)
    if kind == "masked":
        assert original[0, 0] == 0


def test_legacy_ranking_preserves_untied_rank_and_elite_scales():
    values = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    old_rank = torch.tensor([[-1.0, 1.0, -1.0 / 3.0, 1.0 / 3.0]])
    old_rank /= old_rank.std(1, keepdim=True, unbiased=False) + 1e-4
    old_elite = torch.tensor([[-0.5, 0.5, -0.5, 0.5]])
    old_elite /= old_elite.std(1, keepdim=True, unbiased=False) + 1e-4
    torch.testing.assert_close(_centered_descending_rank_scores(values, 1e-4), old_rank)
    torch.testing.assert_close(_centered_elite_scores(values, 0.5, 1e-4)[0], old_elite)


@pytest.mark.parametrize("mix", ["CLASP_PRIMAL_RANK_MIX", "CLASP_PRIMAL_ELITE_MIX"])
def test_actual_legacy_advantages_preserve_feasible_quality_ties(monkeypatch, mix):
    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "0")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_MIX", "0")
    monkeypatch.setenv(mix, "1")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_FRACTION", "0.5")
    monkeypatch.setenv("CLASP_PRIMAL_SCORE_GAP_PENALTY", "0")
    rewards = torch.tensor([2.0, 2.0, 1.0, 1.0])
    advantages, _ = compute_pareto_anchored_advantages(
        rewards, torch.zeros(4), torch.ones(4), 4
    )
    assert advantages[0] == advantages[1]
    assert advantages[2] == advantages[3]
    assert advantages[0] > advantages[2]


def _actual_loss_with_padding(*, width, loss_type, horizon=10, args_fallback=False):
    # Exercise the complete compute_loss method with a differentiable model
    # score and inactive auxiliary hooks; no pretrained model is necessary.
    score = torch.tensor(0.0, requires_grad=True)
    model = SimpleNamespace(training=False)
    trainer = SimpleNamespace(
        model=model,
        top_entropy_quantile=1.0,
        beta=0.0,
        importance_sampling_level="token",
        epsilon_low=0.2,
        epsilon_high=0.2,
        args=SimpleNamespace(delta=None, max_completion_length=horizon),
        use_vllm=False,
        loss_type=loss_type,
        current_gradient_accumulation_steps=2,
        clasp_enabled=False,
        accelerator=SimpleNamespace(gather=lambda x: x),
        _metrics={"eval": defaultdict(list)},
    )
    if not args_fallback:
        trainer.max_completion_length = horizon
    trainer._get_per_token_logps_and_entropies = lambda *a, **k: (
        score.expand(2, width), torch.zeros((2, width))
    )
    for hook in (
        "_compute_reference_imitation_loss",
        "_compute_reference_contrastive_loss",
        "_compute_self_replay_imitation_loss",
        "_compute_self_replay_contrastive_loss",
        "_compute_tsp_two_opt_distillation_loss",
        "_compute_improvement_distillation_losses",
    ):
        setattr(trainer, hook, lambda *a, **k: (None, {}))
    mask = torch.zeros((2, width))
    mask[0, :2] = 1
    mask[1, :3] = 1
    inputs = {
        "prompt_ids": torch.ones((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1)),
        "completion_ids": torch.ones((2, width), dtype=torch.long),
        "completion_mask": mask,
        "weighted_gap_scores": torch.zeros(2),
        "advantages": torch.tensor([1.0, -1.0]),
    }
    loss = PrimalDualGRPOTrainer.compute_loss(trainer, model, inputs)
    loss.backward()
    return loss.detach(), score.grad


@pytest.mark.parametrize("args_fallback", [False, True])
def test_dr_grpo_gradient_uses_fixed_horizon_and_is_padding_invariant(args_fallback):
    short_loss, short_grad = _actual_loss_with_padding(
        width=3, loss_type="dr_grpo", args_fallback=args_fallback
    )
    padded_loss, padded_grad = _actual_loss_with_padding(
        width=7, loss_type="dr_grpo", args_fallback=args_fallback
    )
    # -(2*1 + 3*(-1)) / (batch=2 * fixed_horizon=10 * accumulation=2).
    torch.testing.assert_close(short_loss, torch.tensor(0.025))
    torch.testing.assert_close(short_grad, torch.tensor(0.025))
    torch.testing.assert_close(padded_loss, short_loss)
    torch.testing.assert_close(padded_grad, short_grad)


def test_dr_grpo_rejects_missing_fixed_horizon():
    with pytest.raises(ValueError, match="positive fixed max_completion_length"):
        _actual_loss_with_padding(width=3, loss_type="dr_grpo", horizon=0)


def test_classic_grpo_token_mean_stays_independent_of_configured_horizon():
    loss_a, grad_a = _actual_loss_with_padding(width=3, loss_type="grpo", horizon=10)
    loss_b, grad_b = _actual_loss_with_padding(width=7, loss_type="grpo", horizon=20)
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(grad_a, grad_b)


@pytest.mark.parametrize("correction", [0, 1])
def test_scalarized_credit_explicitly_matches_population_or_trl_sample_std(correction):
    rewards = torch.tensor([0.0, 0.0, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0])
    penalties = torch.zeros_like(rewards)
    actual = compute_scalarized_group_advantages(
        rewards, penalties, 4, std_correction=correction
    )
    grouped = rewards.view(2, 4)
    expected = (grouped - grouped.mean(1, keepdim=True)) / (
        grouped.std(1, keepdim=True, correction=correction) + 1e-4
    )
    torch.testing.assert_close(actual, expected.reshape(-1))
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[4:], torch.zeros(4))
    if correction == 0:
        torch.testing.assert_close(
            actual, compute_scalarized_group_advantages(rewards, penalties, 4)
        )


@pytest.mark.parametrize("invalid", [-1, 2, 0.5, True, "1"])
def test_scalarized_credit_rejects_invalid_std_corrections(invalid):
    with pytest.raises(ValueError, match="std_correction"):
        compute_scalarized_group_advantages(
            torch.arange(4.0), torch.zeros(4), 4, std_correction=invalid
        )


@pytest.mark.parametrize("invalid", ["-1", "2", "0.5", "true"])
def test_trainer_fails_closed_on_invalid_std_correction_env(monkeypatch, invalid):
    monkeypatch.setenv("PD_GROUP_STD_CORRECTION", invalid)
    with pytest.raises(ValueError, match="PD_GROUP_STD_CORRECTION"):
        PrimalDualGRPOTrainer(primal_dual_reward_manager=SimpleNamespace())


def test_generation_logging_forwards_actual_std_correction_stat(monkeypatch):
    trainer = object.__new__(PrimalDualGRPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.accelerator = SimpleNamespace(process_index=0, num_processes=1)
    trainer.clasp_enabled = False
    trainer._clasp_last_generation_budget_stats = {}
    trainer._metrics = {"train": defaultdict(list)}
    trainer.primal_dual_reward_manager = SimpleNamespace()
    for hook in (
        "_validate_fixed_kl_reference", "_extract_task_types", "_set_clasp_generation_budget",
        "_restore_generation_budget", "_attach_reference_imitation_tensors",
        "_attach_reference_contrastive_tensors", "_attach_tsp_two_opt_distillation_tensors",
        "_attach_improvement_distillation_tensors", "_attach_self_replay_tensors",
        "_attach_clasp_group_metadata", "_replace_logged_advantages",
        "_record_clasp_dual_feasibility", "_update_clasp_validation_feasibility",
        "_append_local_per_task_metrics",
    ):
        setattr(trainer, hook, lambda *a, **k: None)
    trainer._get_local_primal_dual_components = lambda: {}
    trainer._observe_self_replay = lambda *a: {}
    stats = defaultdict(float, {"dual_primal/group_scalarization": 1.0,
                                "dual_primal/group_std_correction": 1.0})
    advantages = torch.tensor([-1.0, 1.0])
    trainer._recompute_constrained_terms = lambda _: (
        advantages, advantages, torch.zeros(2), torch.zeros(2), torch.ones(2), 2, stats
    )
    monkeypatch.setattr(
        PrimalDualGRPOTrainer.__bases__[0], "_generate_and_score_completions",
        lambda *a: {"completion_ids": torch.ones((2, 1), dtype=torch.long)},
    )
    output = trainer._generate_and_score_completions([])
    assert trainer._metrics["train"]["constrained_grpo/dual_primal/group_std_correction"] == [1.0]
    torch.testing.assert_close(output["advantages"], advantages)
