from __future__ import annotations

import torch

from clasp.baselines.llmcosolver_f638_objective import (
    combine_two_reward_heads,
    normalize_group_advantages,
    per_token_reverse_kl,
    sequence_mean_grpo_loss,
)


def test_two_official_heads_are_summed_with_unit_weights():
    optimality = torch.tensor([0.5, 2.0])
    feasibility = torch.tensor([1.0, 0.2])
    torch.testing.assert_close(
        combine_two_reward_heads(optimality, feasibility), torch.tensor([1.5, 2.2])
    )


def test_advantages_match_trl_015_unbiased_group_std():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0])
    actual = normalize_group_advantages(rewards, num_generations=4)
    grouped = rewards.reshape(-1, 4)
    expected = (grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, keepdim=True) + 1e-4
    )
    torch.testing.assert_close(actual, expected.flatten())
    assert torch.equal(actual[4:], torch.zeros(4))


def test_sequence_mean_loss_matches_pinned_trl_unsloth_formula_and_not_global_token_mean():
    policy = torch.tensor([[-1.0, -2.0, -3.0], [-0.4, -0.8, -1.2]], requires_grad=True)
    reference = torch.tensor([[-1.2, -2.1, -2.5], [-0.7, -0.9, -1.1]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    advantages = torch.tensor([1.5, -0.5])
    beta = 0.05

    loss, completion_length, mean_kl = sequence_mean_grpo_loss(
        policy, reference, mask, advantages, beta=beta
    )
    log_ratio = reference - policy
    kl = torch.exp(log_ratio) - log_ratio - 1
    per_token = -(torch.exp(policy - policy.detach()) * advantages.unsqueeze(1) - beta * kl)
    lengths = mask.sum(dim=1)
    expected_loss = ((per_token * mask).sum(dim=1) / lengths).mean()
    expected_kl = ((kl * mask).sum(dim=1) / lengths).mean()
    global_token_mean = (per_token * mask).sum() / mask.sum()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(mean_kl, expected_kl)
    torch.testing.assert_close(completion_length, torch.tensor(2.0))
    assert not torch.isclose(loss, global_token_mean)
    loss.backward()
    assert policy.grad is not None


def test_reverse_kl_estimator_is_zero_at_identical_policy_and_reference():
    logps = torch.tensor([[-1.0, -2.0]])
    assert torch.equal(per_token_reverse_kl(logps, logps), torch.zeros_like(logps))
