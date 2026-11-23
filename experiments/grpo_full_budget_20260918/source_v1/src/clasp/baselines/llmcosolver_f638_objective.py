"""Pinned TRL 0.15.2 / Unsloth 2025.3 GRPO objective used by LLMCoSolver."""

from __future__ import annotations

import torch


def combine_two_reward_heads(
    optimality: torch.Tensor,
    feasibility: torch.Tensor,
) -> torch.Tensor:
    """Combine the two official heads with the implicit unit weights."""

    return optimality + feasibility


def normalize_group_advantages(
    rewards: torch.Tensor,
    *,
    num_generations: int = 8,
) -> torch.Tensor:
    """Apply TRL 0.15.2's prompt-group sample-standard-deviation normalization."""

    grouped = rewards.reshape(-1, int(num_generations))
    means = grouped.mean(dim=1).repeat_interleave(int(num_generations), dim=0)
    stds = grouped.std(dim=1).repeat_interleave(int(num_generations), dim=0)
    return (rewards - means) / (stds + 1e-4)


def per_token_reverse_kl(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
) -> torch.Tensor:
    """Return the Schulman nonnegative reverse-KL estimator used upstream."""

    log_ratio = reference_logps - policy_logps
    return torch.exp(log_ratio) - log_ratio - 1.0


def sequence_mean_grpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    beta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the pinned one-iteration, per-sequence-token-mean objective.

    The return values are ``(loss, mean_completion_length, mean_kl)``.  The
    sequence mean followed by batch mean is materially different from a global
    token mean when completion lengths vary.
    """

    kl = per_token_reverse_kl(policy_logps, reference_logps)
    surrogate = torch.exp(policy_logps - policy_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(surrogate - float(beta) * kl)
    mask = completion_mask.to(torch.float32)
    lengths = mask.sum(dim=1)
    per_sequence_loss = (per_token_loss * mask).sum(dim=1) / lengths
    mean_kl = ((kl * mask).sum(dim=1) / lengths).mean()
    return per_sequence_loss.mean(), lengths.mean(), mean_kl


__all__ = [
    "combine_two_reward_heads",
    "normalize_group_advantages",
    "per_token_reverse_kl",
    "sequence_mean_grpo_loss",
]
