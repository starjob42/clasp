"""CLASP-GRPO helpers: centered primal-gap coefficients, feasibility-conditional
multiplier steps, and task-adaptive completion-length estimation.

This module is the single home for CLASP-specific math. The advantage
computation is implemented as a pure function over tensors, so it can
be unit-tested without spinning up the trainer.

All entry points are only invoked when the caller has already verified
that ``use_clasp`` is True. When CLASP is disabled the module is not
imported from any hot path (the trainer's existing combiner is unchanged).
"""
from __future__ import annotations

import collections
import math
import os
import threading
from typing import Deque, Dict, Iterable

import numpy as np
import torch  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Contribution A - Centered within-group primal-gap coefficient.
# ---------------------------------------------------------------------------

def _float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return float(default)
    try:
        value = float(raw_value)
    except ValueError:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _clamped_env(name: str, default: float, *, lower: float, upper: float) -> float:
    value = _float_env(name, default)
    return float(max(lower, min(upper, value)))


def _centered_descending_rank_scores(values: torch.Tensor, eps_std: float) -> torch.Tensor:
    """Return standardized midranks, assigning equal credit to equal values."""
    if values.dim() != 2:
        raise ValueError("_centered_descending_rank_scores expects a rank-2 tensor.")
    if values.size(1) <= 1:
        return torch.zeros_like(values)

    ranked = _tie_aware_descending_rank_scores(
        values, torch.ones_like(values, dtype=torch.bool), 0.0
    )
    return ranked / (ranked.std(dim=1, keepdim=True, unbiased=False) + eps_std)


def _centered_elite_scores(
    values: torch.Tensor,
    elite_fraction: float,
    eps_std: float,
) -> "tuple[torch.Tensor, int]":
    """Return standardized top-k credit, sharing a tied cutoff fractionally."""
    if values.dim() != 2:
        raise ValueError("_centered_elite_scores expects a rank-2 tensor.")
    if values.size(1) <= 1:
        return torch.zeros_like(values), 0

    K = int(values.size(1))
    top_count = max(1, min(K, int(math.ceil(float(elite_fraction) * K))))
    cutoff = torch.topk(values, k=top_count, dim=1).values[:, -1:]
    above_cutoff = values > cutoff
    at_cutoff = values == cutoff
    # Splitting remaining elite mass equally across the boundary tie keeps
    # the requested top-k mass without rewarding arbitrary sample positions.
    remaining = top_count - above_cutoff.sum(dim=1, keepdim=True)
    tie_share = remaining.to(values.dtype) / at_cutoff.sum(dim=1, keepdim=True)
    elite = above_cutoff.to(values.dtype) + at_cutoff.to(values.dtype) * tie_share
    elite = torch.where(
        at_cutoff.all(dim=1, keepdim=True),
        torch.zeros_like(elite),
        elite - elite.mean(dim=1, keepdim=True),
    )
    return elite / (elite.std(dim=1, keepdim=True, unbiased=False) + eps_std), top_count


def _masked_descending_rank_scores(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps_tie: float,
) -> torch.Tensor:
    """Return bounded row-local ranks only over selected entries.

    Scores are in [-1, 1] for rows with at least two distinct selected values
    and zero otherwise. Unlike the standardized rank helper above, this avoids
    injecting arbitrary preferences when all selected values are tied.
    """
    if values.dim() != 2 or mask.dim() != 2:
        raise ValueError("_masked_descending_rank_scores expects rank-2 tensors.")
    if values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch in _masked_descending_rank_scores.")

    return _tie_aware_descending_rank_scores(values, mask, eps_tie)


def compute_feasibility_gated_advantages(
    primal_scores: torch.Tensor,
    violation_scores: torch.Tensor,
    feasibility_flags: torch.Tensor,
    num_generations: int,
    *,
    rank_violation_scores: torch.Tensor | None = None,
    eps_std: float = 1e-4,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return strict feasibility-gated CLASP advantages.

    This mode separates the learning problem into three regimes:

    * all infeasible: ignore objective, rank only by violation reduction;
    * mixed: every feasible completion dominates every infeasible completion,
      then feasible samples are ranked by objective and infeasible samples by
      violation reduction;
    * all feasible: optimize objective/gap aggressively inside the feasible
      region.

    Inference remains raw model decoding. No completion is repaired or
    rewritten by this helper.
    """
    if primal_scores.dim() != 1:
        raise ValueError("compute_feasibility_gated_advantages expects 1-D tensors.")
    if primal_scores.numel() != violation_scores.numel():
        raise ValueError("primal/violation tensor size mismatch.")
    if primal_scores.numel() != feasibility_flags.numel():
        raise ValueError("primal/feasibility tensor size mismatch.")
    if rank_violation_scores is not None and primal_scores.numel() != rank_violation_scores.numel():
        raise ValueError("primal/rank-violation tensor size mismatch.")
    if num_generations <= 0 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {primal_scores.numel()} not divisible by "
            f"num_generations={num_generations}."
        )

    K = int(num_generations)
    r = primal_scores.view(-1, K).to(torch.float32)
    rank_source = violation_scores if rank_violation_scores is None else rank_violation_scores
    g = rank_source.view(-1, K).to(torch.float32).clamp(min=0.0)
    f = feasibility_flags.view(-1, K) > 0.5
    not_f = ~f

    margin = _float_env("CLASP_FEASIBILITY_GATE_MARGIN", 2.0)
    margin = max(0.0, margin)
    feasible_rank_scale = max(0.0, _float_env("CLASP_FEASIBILITY_GATE_FEASIBLE_RANK_SCALE", 0.5))
    infeasible_rank_scale = max(0.0, _float_env("CLASP_FEASIBILITY_GATE_INFEASIBLE_RANK_SCALE", 0.5))
    all_infeasible_weight = max(
        0.0,
        _float_env("CLASP_FEASIBILITY_GATE_ALL_INFEASIBLE_WEIGHT", 1.5),
    )
    mixed_weight = max(0.0, _float_env("CLASP_FEASIBILITY_GATE_MIXED_WEIGHT", 1.25))
    all_feasible_weight = max(
        0.0,
        _float_env("CLASP_FEASIBILITY_GATE_ALL_FEASIBLE_WEIGHT", 1.0),
    )
    feasible_z_mix = _clamped_env(
        "CLASP_FEASIBILITY_GATE_FEASIBLE_Z_MIX",
        0.5,
        lower=0.0,
        upper=1.0,
    )

    feasible_count = f.float().sum(dim=1, keepdim=True)
    all_infeasible = feasible_count == 0.0
    all_feasible = feasible_count == float(K)
    mixed = ~(all_infeasible | all_feasible)

    quality_gap_penalty = max(0.0, _float_env("CLASP_PRIMAL_SCORE_GAP_PENALTY", 0.0))
    quality_values = r - (quality_gap_penalty * g) if quality_gap_penalty > 0.0 else r
    feasible_rank = _masked_descending_rank_scores(quality_values, f, eps_std)
    infeasible_rank = _masked_descending_rank_scores(-g, not_f, eps_std)

    quality_mean = quality_values.mean(dim=1, keepdim=True)
    quality_std = quality_values.std(dim=1, keepdim=True, unbiased=False)
    quality_z = torch.where(
        quality_std > eps_std,
        (quality_values - quality_mean) / (quality_std + eps_std),
        torch.zeros_like(quality_values),
    )
    full_objective_rank = _masked_descending_rank_scores(
        quality_values,
        torch.ones_like(f, dtype=torch.bool),
        eps_std,
    )
    feasible_objective = (feasible_z_mix * quality_z) + ((1.0 - feasible_z_mix) * full_objective_rank)

    mixed_scores = torch.where(
        f,
        margin + (feasible_rank_scale * feasible_rank),
        -margin + (infeasible_rank_scale * infeasible_rank),
    )
    all_infeasible_scores = infeasible_rank_scale * infeasible_rank
    all_feasible_scores = feasible_objective

    scores = torch.where(
        all_infeasible,
        all_infeasible_scores,
        torch.where(all_feasible, all_feasible_scores, mixed_scores),
    )
    scores = scores - scores.mean(dim=1, keepdim=True)

    group_weights = torch.where(
        all_infeasible,
        torch.full_like(feasible_count, all_infeasible_weight),
        torch.where(
            all_feasible,
            torch.full_like(feasible_count, all_feasible_weight),
            torch.full_like(feasible_count, mixed_weight),
        ),
    )
    advantages = scores * group_weights

    dominance_margins: list[float] = []
    if bool(mixed.any().item()):
        mixed_scores_detached = mixed_scores.detach()
        for row_idx in torch.nonzero(mixed.squeeze(1), as_tuple=False).squeeze(1).tolist():
            row_f = f[row_idx]
            if bool(row_f.any().item()) and bool((~row_f).any().item()):
                min_feasible = mixed_scores_detached[row_idx][row_f].min()
                max_infeasible = mixed_scores_detached[row_idx][~row_f].max()
                dominance_margins.append(float((min_feasible - max_infeasible).item()))

    feas_rate = feasible_count / float(K)
    stats = {
        "clasp/feasibility_gated": 1.0,
        "clasp/all_infeasible_group_rate": float(all_infeasible.float().mean().item()),
        "clasp/mixed_group_rate": float(mixed.float().mean().item()),
        "clasp/fully_feasible_group_rate": float(all_feasible.float().mean().item()),
        "clasp/mean_group_feasibility_rate": float(feas_rate.mean().item()),
        "clasp/feasibility_gate_margin": float(margin),
        "clasp/feasibility_gate_feasible_rank_scale": float(feasible_rank_scale),
        "clasp/feasibility_gate_infeasible_rank_scale": float(infeasible_rank_scale),
        "clasp/feasibility_gate_all_infeasible_weight": float(all_infeasible_weight),
        "clasp/feasibility_gate_mixed_weight": float(mixed_weight),
        "clasp/feasibility_gate_all_feasible_weight": float(all_feasible_weight),
        "clasp/feasibility_gate_feasible_z_mix": float(feasible_z_mix),
        "clasp/feasibility_gate_quality_gap_penalty": float(quality_gap_penalty),
        "clasp/feasibility_gate_group_weight_mean": float(group_weights.mean().item()),
        "clasp/feasibility_gate_mixed_dominance_margin": (
            float(sum(dominance_margins) / len(dominance_margins))
            if dominance_margins
            else 0.0
        ),
        "clasp/advantage_l2": float(advantages.pow(2).mean().sqrt().item()),
        "clasp/advantage_abs_mean": float(advantages.abs().mean().item()),
    }
    return advantages.reshape(-1), stats


def compute_lexicographic_clasp_advantages(
    primal_scores: torch.Tensor,
    violation_scores: torch.Tensor,
    feasibility_flags: torch.Tensor,
    num_generations: int,
    *,
    raw_violation_scores: torch.Tensor,
    eps: float = 1e-4,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return parameter-free feasibility-first grouped advantages.

    This revision preserves the learned controller whenever its weighted gap
    distinguishes candidates.  At cold start (or whenever those gaps tie), it
    falls back to the raw normalized positive violation so ``lambda=0`` cannot
    erase all feasibility ordering.

    Groups are handled lexicographically:

    * all infeasible: lower effective violation wins and objective is ignored;
    * mixed: every feasible completion beats every infeasible completion,
      feasible completions compare by objective, and infeasible completions
      compare by lower effective violation;
    * all feasible: standard centered objective optimization.

    Pairwise win/loss scores provide the mixed/all-infeasible ordering without
    a margin, scale, or task-specific tuning constant.  Ties remain ties.
    """

    tensors = {
        "primal_scores": primal_scores,
        "violation_scores": violation_scores,
        "feasibility_flags": feasibility_flags,
        "raw_violation_scores": raw_violation_scores,
    }
    for name, tensor in tensors.items():
        if tensor.dim() != 1:
            raise ValueError(
                f"compute_lexicographic_clasp_advantages expects 1-D {name}."
            )
        if tensor.numel() != primal_scores.numel():
            raise ValueError(
                "primal/violation/feasibility/raw-violation size mismatch."
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains non-finite values.")
    if num_generations <= 1 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {primal_scores.numel()} must be divisible by "
            f"num_generations={num_generations}>1."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if bool((violation_scores < 0.0).any().item()) or bool(
        (raw_violation_scores < 0.0).any().item()
    ):
        raise ValueError("violation scores must be non-negative.")

    K = int(num_generations)
    r = primal_scores.reshape(-1, K).to(torch.float32)
    learned_g = violation_scores.reshape(-1, K).to(torch.float32)
    raw_g = raw_violation_scores.reshape(-1, K).to(torch.float32)
    f = feasibility_flags.reshape(-1, K) > 0.5

    # Pairwise preference i > j. Cross-feasibility comparisons are exact;
    # among infeasible samples, the raw positive gap breaks only pairwise
    # ties in the learned controller gap.  This pairwise fallback matters
    # when one part of a group is distinguished by the controller while a
    # tied subset still contains useful raw violation ordering.
    feasible_i = f.unsqueeze(2)
    feasible_j = f.unsqueeze(1)
    objective_i = r.unsqueeze(2)
    objective_j = r.unsqueeze(1)
    learned_i = learned_g.unsqueeze(2)
    learned_j = learned_g.unsqueeze(1)
    raw_i = raw_g.unsqueeze(2)
    raw_j = raw_g.unsqueeze(1)
    learned_i_better = learned_i + float(eps) < learned_j
    learned_tie = (learned_i - learned_j).abs() <= float(eps)
    raw_i_better = raw_i + float(eps) < raw_j
    raw_fallback_pair = (
        ~feasible_i
        & ~feasible_j
        & learned_tie
        & raw_i_better
    )
    prefer_i = (
        (feasible_i & ~feasible_j)
        | (
            feasible_i
            & feasible_j
            & (objective_i > objective_j + float(eps))
        )
        | (
            ~feasible_i
            & ~feasible_j
            & (learned_i_better | (learned_tie & raw_i_better))
        )
    )
    prefer_j = prefer_i.transpose(1, 2)
    ordinal = (
        prefer_i.to(torch.float32).sum(dim=2)
        - prefer_j.to(torch.float32).sum(dim=2)
    ) / float(K - 1)

    feasible_count = f.to(torch.float32).sum(dim=1, keepdim=True)
    all_infeasible = feasible_count == 0.0
    all_feasible = feasible_count == float(K)
    mixed = ~(all_infeasible | all_feasible)

    objective_mean = r.mean(dim=1, keepdim=True)
    objective_std = r.std(dim=1, keepdim=True, unbiased=False)
    objective_z = torch.where(
        objective_std > float(eps),
        (r - objective_mean) / (objective_std + float(eps)),
        torch.zeros_like(r),
    )
    ordinal_std = ordinal.std(dim=1, keepdim=True, unbiased=False)
    ordinal_z = torch.where(
        ordinal_std > float(eps),
        ordinal / (ordinal_std + float(eps)),
        torch.zeros_like(ordinal),
    )
    # Preserve standard objective optimization once every completion is
    # feasible; the pairwise rule is needed only while feasibility remains a
    # distinguishing constraint.
    advantages = torch.where(all_feasible, objective_z, ordinal_z)
    advantages = advantages - advantages.mean(dim=1, keepdim=True)

    dominance_margins: list[float] = []
    for row_idx in torch.nonzero(
        mixed.squeeze(1), as_tuple=False
    ).squeeze(1).tolist():
        row_feasible = f[row_idx]
        min_feasible = advantages[row_idx][row_feasible].min()
        max_infeasible = advantages[row_idx][~row_feasible].max()
        dominance_margins.append(
            float((min_feasible - max_infeasible).item())
        )

    stats = {
        "clasp/lexicographic_gate": 1.0,
        "clasp/lexicographic_raw_fallback_group_rate": float(
            raw_fallback_pair.any(dim=2).any(dim=1).to(torch.float32).mean().item()
        ),
        "clasp/lexicographic_all_infeasible_group_rate": float(
            all_infeasible.to(torch.float32).mean().item()
        ),
        "clasp/lexicographic_mixed_group_rate": float(
            mixed.to(torch.float32).mean().item()
        ),
        "clasp/lexicographic_all_feasible_group_rate": float(
            all_feasible.to(torch.float32).mean().item()
        ),
        "clasp/lexicographic_mixed_dominance_margin": (
            min(dominance_margins) if dominance_margins else 0.0
        ),
        "clasp/advantage_l2": float(
            advantages.pow(2).mean().sqrt().item()
        ),
        "clasp/advantage_abs_mean": float(advantages.abs().mean().item()),
    }
    return advantages.reshape(-1), stats


def _expand_group_lambdas(
    group_lambdas: torch.Tensor | float | None,
    *,
    num_groups: int,
    num_generations: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    """Normalize scalar/group/sample multiplier inputs to ``[num_groups, 1]``.

    Sample-shaped inputs are accepted for trainer convenience, but the
    multiplier must be constant inside a prompt group.  Silently averaging
    inconsistent sample multipliers would corrupt the set-level constraint.
    """
    if group_lambdas is None:
        return torch.zeros((num_groups, 1), dtype=torch.float32, device=device)

    lambdas = torch.as_tensor(group_lambdas, dtype=torch.float32, device=device)
    if lambdas.numel() == 1:
        grouped = lambdas.reshape(1, 1).expand(num_groups, 1)
    elif lambdas.shape in {(num_groups,), (num_groups, 1)}:
        grouped = lambdas.reshape(num_groups, 1)
    elif lambdas.numel() == num_groups * num_generations:
        per_sample = lambdas.reshape(num_groups, num_generations)
        grouped = per_sample[:, :1]
        if not bool(torch.all((per_sample - grouped).abs() <= float(eps)).item()):
            raise ValueError("group_lambdas must be constant inside each prompt group.")
    else:
        raise ValueError(
            "group_lambdas must be scalar, [num_groups], [num_groups, 1], "
            "or sample-shaped [num_groups * num_generations]."
        )

    if not bool(torch.isfinite(grouped).all().item()):
        raise ValueError("group_lambdas contains non-finite values.")
    if bool((grouped < 0.0).any().item()):
        raise ValueError("group_lambdas must be non-negative.")
    return grouped


def _tie_aware_descending_rank_scores(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps_tie: float,
) -> torch.Tensor:
    """Return bounded high-is-good ranks without arbitrary ordering of ties."""
    if values.dim() != 2 or values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch in tie-aware rank helper.")

    scores = torch.zeros_like(values)
    for row_idx in range(values.size(0)):
        selected_idx = torch.nonzero(mask[row_idx], as_tuple=False).squeeze(1)
        count = int(selected_idx.numel())
        if count <= 1:
            continue
        selected = values[row_idx].index_select(0, selected_idx)
        differences = selected.unsqueeze(1) - selected.unsqueeze(0)
        wins = (differences > float(eps_tie)).float().sum(dim=1)
        losses = (differences < -float(eps_tie)).float().sum(dim=1)
        row_scores = (wins - losses).to(values.dtype) / float(count - 1)
        row_scores = row_scores - row_scores.mean()
        scores[row_idx].scatter_(0, selected_idx, row_scores)
    return scores


def _robust_row_scores(values: torch.Tensor, eps: float, clip: float) -> torch.Tensor:
    """Robustly center/scale rows, falling back to RMS when MAD is zero."""
    if values.dim() != 2:
        raise ValueError("_robust_row_scores expects a rank-2 tensor.")

    output = torch.zeros_like(values)
    for row_idx in range(values.size(0)):
        row = values[row_idx]
        center = row.median()
        centered = row - center
        scale = centered.abs().median()
        if float(scale.item()) <= float(eps):
            scale = centered.pow(2).mean().sqrt()
        if float(scale.item()) <= float(eps):
            continue
        output[row_idx] = (centered / scale).clamp(min=-float(clip), max=float(clip))
        output[row_idx] = output[row_idx] - output[row_idx].mean()
    return output


def _smooth_max(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Fixed-cardinality smooth max; exact set-monotone max at temperature zero.

    Subtracting ``tau * log(n)`` removes raw log-sum-exp's duplicate-candidate
    bonus. For positive temperature every coordinate is monotone while the
    cardinality is fixed, but leave-one-out changes cardinality and can give a
    low-quality feasible candidate negative marginal credit. Corrected CLASP
    therefore defaults to temperature zero; positive values are an explicit
    smoothing ablation only.
    """
    if values.numel() == 0:
        raise ValueError("_smooth_max requires at least one value.")
    if temperature == 0.0:
        return values.max()
    return float(temperature) * (
        torch.logsumexp(values / float(temperature), dim=0)
        - math.log(float(values.numel()))
    )


def compute_bon_aligned_advantages(
    primal_scores: torch.Tensor,
    violation_scores: torch.Tensor,
    feasibility_flags: torch.Tensor,
    num_generations: int,
    *,
    group_lambdas: torch.Tensor | float | None = None,
    quality_scores: torch.Tensor | None = None,
    quality_temperature: float = 0.0,
    success_credit_mode: str = "dense_rank",
    dense_quality_scale: float = 1.0,
    dense_feasibility_scale: float = 1.0,
    dense_violation_mix: float = 0.5,
    recovery_scale: float = 1.0,
    robust_clip: float = 5.0,
    eps: float = 1e-6,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return BoN-aligned CLASP advantages for raw K-sample prompt groups.

    CLASP evaluation succeeds when *any* raw completion in a K-sample group is
    feasible and judges the best feasible solution.  This helper therefore
    uses the set Lagrangian

    ``smooth_best(feasible_quality) - lambda * I(no feasible completion)``

    The default ``dense_rank`` estimator is a variance-reduced, lexicographic
    surrogate for that set objective.  In every successful mixed group it
    contrasts feasible against infeasible completions, with pressure scaled by
    the learned group multiplier.  Among feasible completions it adds bounded
    pairwise midrank quality credit.  Equal-quality candidates receive exactly
    equal credit, so discrete objective ties never create an arbitrary winner.
    The surrogate is centered within each group and supplies dense signed
    credit whenever feasibility or feasible quality differs.

    ``success_credit_mode="exact_loo"`` preserves the exact leave-one-out
    set-Lagrangian estimator as a publication ablation.  It assigns completion
    ``i`` the marginal contribution ``L(G) - L(G \\ {i})``.  This estimator is
    faithful to exact best-of-K utility, but is intentionally not the default:
    it has zero quality signal when the best feasible quality is tied and gives
    no credit to non-frontier feasible candidates.

    Pure leave-one-out credit is flat when a group is entirely infeasible.  In
    that regime only, the helper uses a robust dense lower-violation surrogate
    plus tie-aware ranks.  Its scale grows with the group multiplier, so the
    signed controller affects the actual policy advantages.

    ``quality_scores`` are high-is-good and default to ``primal_scores``.
    ``violation_scores`` are non-negative and lower-is-better.  This function
    changes training credit only; it performs no inference repair, search, or
    candidate rewriting. ``quality_temperature=0`` is the exact BoN objective;
    positive temperatures are a deliberately non-exact smoothing ablation for
    ``exact_loo``.  They do not affect the rank-based dense estimator.
    """
    if primal_scores.dim() != 1:
        raise ValueError("compute_bon_aligned_advantages expects 1-D tensors.")
    if violation_scores.dim() != 1 or feasibility_flags.dim() != 1:
        raise ValueError("violation_scores and feasibility_flags must be 1-D tensors.")
    if primal_scores.numel() != violation_scores.numel():
        raise ValueError("primal/violation tensor size mismatch.")
    if primal_scores.numel() != feasibility_flags.numel():
        raise ValueError("primal/feasibility tensor size mismatch.")
    if quality_scores is not None:
        if quality_scores.dim() != 1:
            raise ValueError("quality_scores must be a 1-D tensor.")
        if quality_scores.numel() != primal_scores.numel():
            raise ValueError("quality/primal tensor size mismatch.")
    if num_generations <= 1 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {primal_scores.numel()} must be divisible by "
            f"num_generations={num_generations}, with num_generations > 1."
        )
    if not math.isfinite(quality_temperature) or quality_temperature < 0.0:
        raise ValueError("quality_temperature must be finite and >= 0.")
    normalized_success_credit_mode = str(success_credit_mode).strip().lower()
    if normalized_success_credit_mode not in {"dense_rank", "exact_loo"}:
        raise ValueError(
            "success_credit_mode must be 'dense_rank' or 'exact_loo', got "
            f"{success_credit_mode!r}."
        )
    if not math.isfinite(dense_quality_scale) or dense_quality_scale < 0.0:
        raise ValueError("dense_quality_scale must be finite and >= 0.")
    if not math.isfinite(dense_feasibility_scale) or dense_feasibility_scale < 0.0:
        raise ValueError("dense_feasibility_scale must be finite and >= 0.")
    if not 0.0 <= dense_violation_mix <= 1.0:
        raise ValueError("dense_violation_mix must be in [0, 1].")
    if not math.isfinite(recovery_scale) or recovery_scale < 0.0:
        raise ValueError("recovery_scale must be finite and >= 0.")
    if not math.isfinite(robust_clip) or robust_clip <= 0.0:
        raise ValueError("robust_clip must be finite and > 0.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0.")

    K = int(num_generations)
    num_groups = int(primal_scores.numel() // K)
    device = primal_scores.device
    primal = primal_scores.reshape(num_groups, K).to(dtype=torch.float32)
    quality_source = primal_scores if quality_scores is None else quality_scores
    quality = quality_source.reshape(num_groups, K).to(device=device, dtype=torch.float32)
    violations = violation_scores.reshape(num_groups, K).to(device=device, dtype=torch.float32)
    raw_feasibility = feasibility_flags.reshape(num_groups, K).to(
        device=device,
        dtype=torch.float32,
    )
    feasible = raw_feasibility > 0.5
    lambdas = _expand_group_lambdas(
        group_lambdas,
        num_groups=num_groups,
        num_generations=K,
        device=device,
        eps=eps,
    )

    for name, values in (
        ("primal_scores", primal),
        ("quality_scores", quality),
        ("violation_scores", violations),
        ("feasibility_flags", raw_feasibility),
    ):
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{name} contains non-finite values.")
    if bool((violations < 0.0).any().item()):
        raise ValueError("violation_scores must be non-negative.")
    if bool(((raw_feasibility < 0.0) | (raw_feasibility > 1.0)).any().item()):
        raise ValueError("feasibility_flags must lie in [0, 1].")

    feasible_count = feasible.float().sum(dim=1)
    all_infeasible = feasible_count == 0.0
    singleton_feasible = feasible_count == 1.0
    multiple_feasible = feasible_count >= 2.0
    all_feasible = feasible_count == float(K)

    marginal_credit = torch.zeros_like(quality)
    full_utilities = torch.zeros(num_groups, dtype=torch.float32, device=device)
    critical_marginals: list[torch.Tensor] = []

    for group_idx in range(num_groups):
        row_feasible = feasible[group_idx]
        if not bool(row_feasible.any().item()):
            continue

        row_quality = quality[group_idx]
        feasible_quality = row_quality[row_feasible]
        full_utility = _smooth_max(feasible_quality, float(quality_temperature))
        full_utilities[group_idx] = full_utility

        feasible_indices = torch.nonzero(row_feasible, as_tuple=False).squeeze(1)
        for sample_idx_tensor in feasible_indices:
            sample_idx = int(sample_idx_tensor.item())
            without = row_feasible.clone()
            without[sample_idx] = False
            if bool(without.any().item()):
                without_utility = _smooth_max(
                    row_quality[without],
                    float(quality_temperature),
                )
                without_lagrangian = without_utility
            else:
                # Removing the only feasible candidate causes a BoN failure.
                # Empty-set quality is zero, hence L(empty) = -lambda.
                without_lagrangian = -lambdas[group_idx, 0]
            marginal_credit[group_idx, sample_idx] = full_utility - without_lagrangian
            if bool(singleton_feasible[group_idx].item()):
                critical_marginals.append(marginal_credit[group_idx, sample_idx])

    all_entries = torch.ones_like(feasible, dtype=torch.bool)
    dense_recovery = _robust_row_scores(-violations, float(eps), float(robust_clip))
    rank_recovery = _tie_aware_descending_rank_scores(-violations, all_entries, float(eps))
    recovery = (
        float(dense_violation_mix) * dense_recovery
        + (1.0 - float(dense_violation_mix)) * rank_recovery
    )
    recovery = recovery - recovery.mean(dim=1, keepdim=True)
    recovery_pressure = float(recovery_scale) * (1.0 + lambdas)
    recovery_advantages = recovery * recovery_pressure

    # Dense successful-group credit is a deliberately biased, lower-variance
    # lexicographic surrogate for the exact set utility. Feasibility credit is
    # zero when all K candidates are feasible, and tie-aware quality midranks
    # are centered over feasible candidates only. Thus an all-feasible quality
    # tie correctly remains a zero-information group, while a mixed group with
    # tied feasible candidates still reinforces every feasible candidate
    # equally against the infeasible candidates.
    dense_quality_credit = _tie_aware_descending_rank_scores(
        quality,
        feasible,
        float(eps),
    )
    dense_feasibility_credit = feasible.to(dtype=torch.float32)
    dense_feasibility_credit = (
        dense_feasibility_credit
        - dense_feasibility_credit.mean(dim=1, keepdim=True)
    )
    dense_success_credit = (
        float(dense_quality_scale) * dense_quality_credit
        + float(dense_feasibility_scale)
        * (1.0 + lambdas)
        * dense_feasibility_credit
    )
    dense_success_credit = torch.where(
        all_infeasible.unsqueeze(1),
        torch.zeros_like(dense_success_credit),
        dense_success_credit,
    )

    if normalized_success_credit_mode == "exact_loo":
        # Do not group-center exact successful-group marginals. Centering would
        # turn zero-credit, non-critical infeasible candidates negative and
        # change the exact leave-one-out estimator.
        successful_credit = marginal_credit
    else:
        successful_credit = dense_success_credit

    # The all-infeasible recovery surrogate is centered independently above.
    advantages = torch.where(
        all_infeasible.unsqueeze(1),
        recovery_advantages,
        successful_credit,
    )

    critical_mean = (
        float(torch.stack(critical_marginals).mean().item())
        if critical_marginals
        else 0.0
    )
    nonempty_utilities = full_utilities[~all_infeasible]
    successful_groups = ~all_infeasible
    successful_count = int(successful_groups.sum().item())
    zero_quality_signal_rate = 0.0
    if successful_count:
        per_group_success_mass = successful_credit.abs().sum(dim=1)
        zero_quality_signal_rate = float(
            ((per_group_success_mass <= float(eps)) & successful_groups).float().sum().item()
            / successful_count
        )
    best_runner_margin_mean = 0.0
    best_quality_tie_rate = 0.0
    multiple_count = int(multiple_feasible.sum().item())
    if multiple_count:
        masked_quality = quality.masked_fill(~feasible, float("-inf"))
        top_two = torch.topk(masked_quality, k=2, dim=1).values
        best_runner_margins = (top_two[:, 0] - top_two[:, 1])[multiple_feasible]
        best_runner_margin_mean = float(best_runner_margins.mean().item())
        best_quality_tie_rate = float(
            (best_runner_margins <= float(eps)).float().mean().item()
        )
    stats = {
        "clasp/bon_aligned": 1.0,
        "clasp/bon_group_failure_rate": float(all_infeasible.float().mean().item()),
        "clasp/bon_group_success_rate": float((~all_infeasible).float().mean().item()),
        "clasp/bon_singleton_feasible_group_rate": float(singleton_feasible.float().mean().item()),
        "clasp/bon_multiple_feasible_group_rate": float(multiple_feasible.float().mean().item()),
        "clasp/bon_fully_feasible_group_rate": float(all_feasible.float().mean().item()),
        "clasp/bon_mean_candidate_feasibility": float((feasible_count / float(K)).mean().item()),
        "clasp/bon_mean_group_lambda": float(lambdas.mean().item()),
        "clasp/bon_max_group_lambda": float(lambdas.max().item()),
        "clasp/bon_lambda_sensitive_group_rate": float(
            (
                all_infeasible
                | (
                    singleton_feasible
                    if normalized_success_credit_mode == "exact_loo"
                    else (successful_groups & ~all_feasible)
                )
            ).float().mean().item()
        ),
        "clasp/bon_success_credit_dense_rank": float(
            normalized_success_credit_mode == "dense_rank"
        ),
        "clasp/bon_success_credit_exact_loo": float(
            normalized_success_credit_mode == "exact_loo"
        ),
        "clasp/bon_dense_quality_scale": float(dense_quality_scale),
        "clasp/bon_dense_feasibility_scale": float(dense_feasibility_scale),
        "clasp/bon_dense_quality_nonzero_rate": float(
            (dense_quality_credit.abs() > float(eps)).float().mean().item()
        ),
        "clasp/bon_dense_feasibility_nonzero_rate": float(
            (dense_feasibility_credit.abs() > float(eps)).float().mean().item()
        ),
        "clasp/bon_success_credit_nonzero_rate": float(
            (
                (successful_credit.abs() > float(eps))
                & successful_groups.unsqueeze(1)
            ).float().sum().item()
            / max(successful_count * K, 1)
        ),
        "clasp/bon_quality_temperature": float(quality_temperature),
        "clasp/bon_dense_violation_mix": float(dense_violation_mix),
        "clasp/bon_recovery_scale": float(recovery_scale),
        "clasp/bon_robust_clip": float(robust_clip),
        "clasp/bon_successful_group_utility_mean": (
            float(nonempty_utilities.mean().item()) if nonempty_utilities.numel() else 0.0
        ),
        "clasp/bon_critical_marginal_mean": critical_mean,
        "clasp/bon_best_runner_margin_mean": best_runner_margin_mean,
        "clasp/bon_best_quality_tie_rate": best_quality_tie_rate,
        "clasp/bon_zero_success_credit_group_rate": zero_quality_signal_rate,
        # Backward-compatible telemetry name retained for old dashboards. In
        # dense_rank mode this is zero *total successful-group credit*, which
        # includes feasibility as well as quality credit.
        "clasp/bon_zero_quality_signal_success_group_rate": zero_quality_signal_rate,
        "clasp/bon_marginal_abs_mean": float(marginal_credit.abs().mean().item()),
        "clasp/bon_marginal_negative_rate": float((marginal_credit < 0.0).float().mean().item()),
        "clasp/bon_recovery_abs_mean": float(
            recovery_advantages[all_infeasible].abs().mean().item()
        ) if bool(all_infeasible.any().item()) else 0.0,
        "clasp/advantage_l2": float(advantages.pow(2).mean().sqrt().item()),
        "clasp/advantage_abs_mean": float(advantages.abs().mean().item()),
        "clasp/advantage_group_mean_abs_max": float(
            advantages.mean(dim=1).abs().max().item()
        ),
    }
    return advantages.reshape(-1), stats


def _expand_task_parameter(
    name: str,
    value: torch.Tensor | float,
    *,
    num_tasks: int,
    device: torch.device,
) -> torch.Tensor:
    parameter = torch.as_tensor(value, dtype=torch.float32, device=device)
    if parameter.numel() == 1:
        return parameter.reshape(1).expand(num_tasks)
    if parameter.dim() == 1 and parameter.numel() == num_tasks:
        return parameter
    raise ValueError(f"{name} must be scalar or a length-{num_tasks} tensor.")


def update_signed_bon_lambdas(
    current_lambdas: torch.Tensor,
    group_failure_flags: torch.Tensor,
    *,
    group_task_indices: torch.Tensor | None = None,
    target_failure_rates: torch.Tensor | float = 0.01,
    learning_rates: torch.Tensor | float = 0.05,
    lambda_max: float = 100.0,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Apply a signed projected dual step for per-task BoN failure rates.

    ``lambda[t] <- clip(lambda[t] + eta[t] * (failure[t] - target[t]))``

    Unlike a positive-violation-only controller, this update reduces pressure
    when the observed raw BoN failure rate is below target.  In distributed
    training the caller must pass globally gathered group flags (or otherwise
    globally reduced sufficient statistics) so every rank takes the same
    step.  Tasks absent from the supplied groups are left unchanged.
    """
    if current_lambdas.dim() != 1 or current_lambdas.numel() == 0:
        raise ValueError("current_lambdas must be a non-empty 1-D tensor.")
    if not current_lambdas.is_floating_point():
        raise ValueError("current_lambdas must use a floating-point dtype.")
    if group_failure_flags.dim() != 1:
        raise ValueError("group_failure_flags must be a 1-D tensor.")
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise ValueError("lambda_max must be finite and > 0.")

    device = current_lambdas.device
    original_dtype = current_lambdas.dtype
    lambdas = current_lambdas.to(dtype=torch.float32)
    failures = group_failure_flags.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(lambdas).all().item()) or bool((lambdas < 0.0).any().item()):
        raise ValueError("current_lambdas must be finite and non-negative.")
    if not bool(torch.isfinite(failures).all().item()):
        raise ValueError("group_failure_flags contains non-finite values.")
    if bool(((failures < 0.0) | (failures > 1.0)).any().item()):
        raise ValueError("group_failure_flags must lie in [0, 1].")

    num_tasks = int(lambdas.numel())
    targets = _expand_task_parameter(
        "target_failure_rates",
        target_failure_rates,
        num_tasks=num_tasks,
        device=device,
    )
    rates = _expand_task_parameter(
        "learning_rates",
        learning_rates,
        num_tasks=num_tasks,
        device=device,
    )
    if not bool(torch.isfinite(targets).all().item()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any().item()
    ):
        raise ValueError("target_failure_rates must be finite and lie in [0, 1].")
    if not bool(torch.isfinite(rates).all().item()) or bool((rates < 0.0).any().item()):
        raise ValueError("learning_rates must be finite and non-negative.")

    if group_task_indices is None:
        if num_tasks != 1:
            raise ValueError("group_task_indices is required when more than one task lambda is used.")
        task_indices = torch.zeros(failures.numel(), dtype=torch.long, device=device)
    else:
        if group_task_indices.dim() != 1 or group_task_indices.numel() != failures.numel():
            raise ValueError("group_task_indices must be 1-D and match group_failure_flags.")
        raw_indices = group_task_indices.to(device=device)
        task_indices = raw_indices.to(dtype=torch.long)
        if not bool((raw_indices == task_indices.to(dtype=raw_indices.dtype)).all().item()):
            raise ValueError("group_task_indices must contain integer task indices.")
        if task_indices.numel() and bool(
            ((task_indices < 0) | (task_indices >= num_tasks)).any().item()
        ):
            raise ValueError("group_task_indices contains an out-of-range task index.")

    counts = torch.zeros(num_tasks, dtype=torch.float32, device=device)
    failure_sums = torch.zeros(num_tasks, dtype=torch.float32, device=device)
    if task_indices.numel():
        counts.scatter_add_(0, task_indices, torch.ones_like(failures))
        failure_sums.scatter_add_(0, task_indices, failures)
    active = counts > 0.0
    observed = torch.zeros_like(lambdas)
    observed[active] = failure_sums[active] / counts[active]
    signed_steps = rates * (observed - targets)
    signed_steps = torch.where(active, signed_steps, torch.zeros_like(signed_steps))
    updated = (lambdas + signed_steps).clamp(min=0.0, max=float(lambda_max))
    actual_delta = updated - lambdas

    active_count = int(active.sum().item())
    stats = {
        "clasp/bon_dual_active_task_count": float(active_count),
        "clasp/bon_dual_group_failure_rate": (
            float(failures.mean().item()) if failures.numel() else 0.0
        ),
        "clasp/bon_dual_observed_failure_rate_mean": (
            float(observed[active].mean().item()) if active_count else 0.0
        ),
        "clasp/bon_dual_target_failure_rate_mean": (
            float(targets[active].mean().item()) if active_count else 0.0
        ),
        "clasp/bon_dual_lambda_mean": float(updated.mean().item()),
        "clasp/bon_dual_lambda_max_observed": float(updated.max().item()),
        "clasp/bon_dual_delta_mean": (
            float(actual_delta[active].mean().item()) if active_count else 0.0
        ),
        "clasp/bon_dual_increase_task_rate": (
            float((actual_delta[active] > 0.0).float().mean().item()) if active_count else 0.0
        ),
        "clasp/bon_dual_decrease_task_rate": (
            float((actual_delta[active] < 0.0).float().mean().item()) if active_count else 0.0
        ),
        "clasp/bon_dual_at_zero_rate": float((updated == 0.0).float().mean().item()),
        "clasp/bon_dual_at_max_rate": float((updated == float(lambda_max)).float().mean().item()),
    }
    return updated.to(dtype=original_dtype), stats


def compute_feasibility_warmup_advantages(
    violation_scores: torch.Tensor,
    feasibility_flags: torch.Tensor,
    num_generations: int,
    *,
    sample_warmup_flags: torch.Tensor | None = None,
    feasible_margin: float = 3.0,
    violation_scale: float = 2.0,
    rank_scale: float = 1.0,
    eps_std: float = 1e-4,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return feasibility-only CLASP warmup advantages.

    Objective values are intentionally absent from this helper. Feasible
    completions get a fixed positive margin, infeasible completions get dense
    negative pressure from exact evaluator violations, and all-infeasible
    groups rank by violation reduction.
    """
    if violation_scores.dim() != 1:
        raise ValueError("compute_feasibility_warmup_advantages expects 1-D tensors.")
    if violation_scores.numel() != feasibility_flags.numel():
        raise ValueError("violation/feasibility tensor size mismatch.")
    if sample_warmup_flags is not None and violation_scores.numel() != sample_warmup_flags.numel():
        raise ValueError("violation/warmup tensor size mismatch.")
    if num_generations <= 0 or violation_scores.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {violation_scores.numel()} not divisible by "
            f"num_generations={num_generations}."
        )

    K = int(num_generations)
    g = violation_scores.view(-1, K).to(torch.float32).clamp(min=0.0)
    f = feasibility_flags.view(-1, K) > 0.5
    margin = max(0.0, float(feasible_margin))
    violation_scale = max(0.0, float(violation_scale))
    rank_scale = max(0.0, float(rank_scale))

    feasible_count = f.float().sum(dim=1, keepdim=True)
    all_infeasible = feasible_count == 0.0
    all_feasible = feasible_count == float(K)
    mixed = ~(all_infeasible | all_feasible)

    dense_infeasible = -violation_scale * g
    ranked_infeasible = rank_scale * _masked_descending_rank_scores(-g, torch.ones_like(f, dtype=torch.bool), eps_std)
    infeasible_scores = dense_infeasible + ranked_infeasible
    mixed_scores = torch.where(f, torch.full_like(g, margin), -margin + infeasible_scores)
    all_infeasible_scores = infeasible_scores
    all_feasible_scores = torch.zeros_like(g)

    scores = torch.where(
        all_infeasible,
        all_infeasible_scores,
        torch.where(all_feasible, all_feasible_scores, mixed_scores),
    )
    scores = scores - scores.mean(dim=1, keepdim=True)

    if sample_warmup_flags is not None:
        warmup = sample_warmup_flags.view(-1, K).to(dtype=torch.bool, device=scores.device)
        scores = torch.where(warmup, scores, torch.zeros_like(scores))

    active = (
        torch.ones_like(scores, dtype=torch.bool)
        if sample_warmup_flags is None
        else sample_warmup_flags.view(-1, K).to(dtype=torch.bool, device=scores.device)
    )
    stats = {
        "clasp/feasibility_warmup": 1.0,
        "clasp/feasibility_warmup_active_rate": float(active.float().mean().item()),
        "clasp/feasibility_warmup_group_active_rate": float(active.any(dim=1).float().mean().item()),
        "clasp/feasibility_warmup_all_infeasible_group_rate": float(all_infeasible.float().mean().item()),
        "clasp/feasibility_warmup_mixed_group_rate": float(mixed.float().mean().item()),
        "clasp/feasibility_warmup_all_feasible_group_rate": float(all_feasible.float().mean().item()),
        "clasp/feasibility_warmup_margin": float(margin),
        "clasp/feasibility_warmup_violation_scale": float(violation_scale),
        "clasp/feasibility_warmup_rank_scale": float(rank_scale),
        "clasp/feasibility_warmup_advantage_abs_mean": float(scores.abs().mean().item()),
    }
    return scores.reshape(-1), stats


def compute_pareto_anchored_advantages(
    primal_scores: torch.Tensor,       # [B*K]
    violation_scores: torch.Tensor,    # [B*K], non-negative (positive_weighted_gap preferred)
    feasibility_flags: torch.Tensor,   # [B*K], 1.0 feasible / 0.0 otherwise
    num_generations: int,
    *,
    rank_violation_scores: torch.Tensor | None = None,
    tau: float = 4.0,
    feasibility_center: float = 0.5,
    eps_std: float = 1e-4,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return (coefficients [B*K], stats).

    The public name is retained for checkpoint/configuration compatibility.
    The effective update is the centered CLASP normal form: an implementation
    may compute an anchor first, but group centering cancels all additive
    anchor terms.

    Algorithm
    ---------
    1. Reshape to [B, K].
    2. alpha_G = sigmoid(tau * (feas_rate - feasibility_center)) in [0, 1].
    3. Pareto frontier in (r, g) space is computed only for diagnostics.
    4. coefficient_i = alpha * (r_i - mean(r)) / sigma_r
       + (1 - alpha) * (mean(g) - g_i) / sigma_g.

    The coefficient is well-defined even on fully-infeasible groups, and is
    nonconstant whenever the positive weighted gaps vary within the group.
    """
    if primal_scores.dim() != 1:
        raise ValueError("compute_pareto_anchored_advantages expects 1-D tensors.")
    if primal_scores.numel() != violation_scores.numel():
        raise ValueError("primal/violation tensor size mismatch.")
    if primal_scores.numel() != feasibility_flags.numel():
        raise ValueError("primal/feasibility tensor size mismatch.")
    if rank_violation_scores is not None and primal_scores.numel() != rank_violation_scores.numel():
        raise ValueError("primal/rank-violation tensor size mismatch.")
    if num_generations <= 0 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {primal_scores.numel()} not divisible by "
            f"num_generations={num_generations}."
        )
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not 0.0 <= feasibility_center <= 1.0:
        raise ValueError(
            "feasibility_center must lie in [0, 1], "
            f"got {feasibility_center}"
        )

    K = int(num_generations)
    r = primal_scores.view(-1, K).to(torch.float32)
    g = violation_scores.view(-1, K).to(torch.float32).clamp(min=0.0)
    f = (feasibility_flags.view(-1, K) > 0.5).float()
    rank_gap_penalty = max(0.0, _float_env("CLASP_PRIMAL_SCORE_GAP_PENALTY", 0.0))
    if rank_gap_penalty > 0.0:
        rank_source = violation_scores if rank_violation_scores is None else rank_violation_scores
        rank_g = rank_source.view(-1, K).to(torch.float32).clamp(min=0.0)
        rank_values = r - (float(rank_gap_penalty) * rank_g)
        rank_gap_mean = float(rank_g.mean().item())
    else:
        rank_g = torch.zeros_like(g)
        rank_values = r
        rank_gap_mean = 0.0

    feas_rate = f.mean(dim=1, keepdim=True)                          # [B, 1]
    alpha = torch.sigmoid(
        tau * (feas_rate - float(feasibility_center))
    )                                                                # [B, 1]

    sigma_r = r.std(dim=1, keepdim=True, unbiased=False) + eps_std
    sigma_g = g.std(dim=1, keepdim=True, unbiased=False) + eps_std

    # Pareto dominance: j strictly dominates i iff r_j > r_i AND g_j < g_i.
    r_i = r.unsqueeze(2)        # [B, K, 1]
    r_j = r.unsqueeze(1)        # [B, 1, K]
    g_i = g.unsqueeze(2)
    g_j = g.unsqueeze(1)
    dominated_by = (r_j > r_i) & (g_j < g_i)
    on_frontier = ~dominated_by.any(dim=2)                           # [B, K]

    # Anchor-equivalent implementation: centering cancels the selected anchor.
    score = g - alpha * r
    score = torch.where(on_frontier, score, torch.full_like(score, float("inf")))
    anchor_idx = score.argmin(dim=1, keepdim=True)                   # [B, 1]
    r_star = r.gather(1, anchor_idx)                                 # [B, 1]
    g_star = g.gather(1, anchor_idx)                                 # [B, 1]

    standard_primal_component = (r - r_star) / sigma_r
    rank_mix_raw = _clamped_env("CLASP_PRIMAL_RANK_MIX", 0.0, lower=0.0, upper=1.0)
    elite_mix_raw = _clamped_env("CLASP_PRIMAL_ELITE_MIX", 0.0, lower=0.0, upper=1.0)
    elite_fraction = _clamped_env(
        "CLASP_PRIMAL_ELITE_FRACTION",
        0.125,
        lower=1e-6,
        upper=1.0,
    )
    if rank_mix_raw + elite_mix_raw > 1.0:
        nonstandard_total = rank_mix_raw + elite_mix_raw
        rank_mix = rank_mix_raw / nonstandard_total
        elite_mix = elite_mix_raw / nonstandard_total
        standard_mix = 0.0
    else:
        rank_mix = rank_mix_raw
        elite_mix = elite_mix_raw
        standard_mix = 1.0 - rank_mix - elite_mix

    elite_count = 0
    if rank_mix > 0.0 or elite_mix > 0.0:
        rank_component = _centered_descending_rank_scores(rank_values, eps_std)
        primal_component = (standard_mix * standard_primal_component) + (rank_mix * rank_component)
        if elite_mix > 0.0:
            elite_component, elite_count = _centered_elite_scores(rank_values, elite_fraction, eps_std)
            primal_component = primal_component + (elite_mix * elite_component)
    else:
        primal_component = standard_primal_component

    primal_term = alpha * primal_component
    dual_term = (1.0 - alpha) * (g_star - g) / sigma_g
    advantages = primal_term + dual_term

    # Group-centre (preserves GRPO's invariance to additive group constants).
    advantages = advantages - advantages.mean(dim=1, keepdim=True)

    stats = {
        "clasp/mean_alpha": float(alpha.mean().item()),
        "clasp/std_alpha": float(alpha.std(unbiased=False).item()) if alpha.numel() > 1 else 0.0,
        "clasp/feasibility_center": float(feasibility_center),
        "clasp/mean_frontier_size": float(on_frontier.float().sum(dim=1).mean().item()),
        "clasp/all_infeasible_group_rate": float((feas_rate.squeeze(-1) == 0).float().mean().item()),
        "clasp/fully_feasible_group_rate": float((feas_rate.squeeze(-1) == 1).float().mean().item()),
        "clasp/advantage_l2": float(advantages.pow(2).mean().sqrt().item()),
        "clasp/advantage_abs_mean": float(advantages.abs().mean().item()),
        "clasp/primal_rank_mix": float(rank_mix),
        "clasp/primal_elite_mix": float(elite_mix),
        "clasp/primal_standard_mix": float(standard_mix),
        "clasp/primal_elite_fraction": float(elite_fraction),
        "clasp/primal_elite_count": float(elite_count),
        "clasp/primal_score_gap_penalty": float(rank_gap_penalty),
        "clasp/primal_rank_gap_mean": float(rank_gap_mean),
    }
    return advantages.reshape(-1), stats


def compute_hard_group_weighted_advantages(
    advantages: torch.Tensor,
    raw_positive_gap_scores: torch.Tensor,
    num_generations: int,
    *,
    scale: float,
    threshold: float = 0.0,
    max_weight: float = 0.0,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Boost updates for prompt groups whose best candidate is still over target.

    The weighting is group-constant, so it preserves the within-group CLASP
    ordering and centered shape. It only changes how much a hard prompt group
    contributes to the policy update.
    """
    if advantages.dim() != 1:
        raise ValueError("compute_hard_group_weighted_advantages expects 1-D advantages.")
    if raw_positive_gap_scores.dim() != 1:
        raise ValueError("compute_hard_group_weighted_advantages expects 1-D raw gap scores.")
    if advantages.numel() != raw_positive_gap_scores.numel():
        raise ValueError("advantage/raw-gap tensor size mismatch.")
    if num_generations <= 0 or advantages.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {advantages.numel()} not divisible by "
            f"num_generations={num_generations}."
        )
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if max_weight < 0.0:
        raise ValueError(f"max_weight must be >= 0, got {max_weight}")

    K = int(num_generations)
    grouped_advantages = advantages.view(-1, K)
    grouped_raw_gap = raw_positive_gap_scores.view(-1, K).to(torch.float32).clamp(min=0.0)
    group_min_gap = grouped_raw_gap.min(dim=1, keepdim=True).values
    weights = 1.0 + (float(scale) * (group_min_gap - float(threshold)).clamp(min=0.0))
    if max_weight > 0.0:
        weights = weights.clamp(max=float(max_weight))
    weighted_advantages = grouped_advantages * weights.to(grouped_advantages.dtype)

    active_groups = weights > 1.0 + 1e-8
    stats = {
        "clasp/hard_group_weight_scale": float(scale),
        "clasp/hard_group_weight_threshold": float(threshold),
        "clasp/hard_group_weight_max": float(max_weight),
        "clasp/hard_group_weight_mean": float(weights.mean().item()),
        "clasp/hard_group_weight_max_observed": float(weights.max().item()),
        "clasp/hard_group_active_rate": float(active_groups.float().mean().item()),
        "clasp/hard_group_min_raw_gap_mean": float(group_min_gap.mean().item()),
    }
    return weighted_advantages.reshape(-1), stats


def compute_pairwise_preference_loss(
    sequence_log_ratios: torch.Tensor,
    advantages: torch.Tensor,
    num_generations: int,
    *,
    margin: float = 0.0,
    advantage_delta: float = 0.0,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return a group-local best-vs-worst logistic margin loss.

    The helper is intentionally small and default-off at the trainer level. It
    uses the already-computed CLASP advantage only to select a preferred and
    dispreferred completion inside each prompt group, then optimizes the
    sequence log-ratio gap between them. This keeps the final inference budget
    unchanged while sharpening credit assignment during training.
    """
    if sequence_log_ratios.dim() != 1:
        raise ValueError("compute_pairwise_preference_loss expects 1-D log ratios.")
    if advantages.dim() != 1:
        raise ValueError("compute_pairwise_preference_loss expects 1-D advantages.")
    if sequence_log_ratios.numel() != advantages.numel():
        raise ValueError("log-ratio/advantage tensor size mismatch.")
    if num_generations <= 1 or sequence_log_ratios.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {sequence_log_ratios.numel()} not divisible by "
            f"num_generations={num_generations}."
        )
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if advantage_delta < 0.0:
        raise ValueError(f"advantage_delta must be >= 0, got {advantage_delta}")

    K = int(num_generations)
    grouped_ratios = sequence_log_ratios.view(-1, K)
    grouped_advantages = advantages.detach().to(torch.float32).view(-1, K)

    best_idx = grouped_advantages.argmax(dim=1, keepdim=True)
    worst_idx = grouped_advantages.argmin(dim=1, keepdim=True)
    best_adv = grouped_advantages.gather(1, best_idx).squeeze(1)
    worst_adv = grouped_advantages.gather(1, worst_idx).squeeze(1)
    advantage_spread = best_adv - worst_adv
    valid = advantage_spread >= float(advantage_delta)

    zero = sequence_log_ratios.sum() * 0.0
    if not bool(valid.any().item()):
        stats = {
            "clasp/pairwise_valid_group_rate": 0.0,
            "clasp/pairwise_advantage_spread": 0.0,
            "clasp/pairwise_log_ratio_gap": 0.0,
        }
        return zero, stats

    best_ratio = grouped_ratios.gather(1, best_idx).squeeze(1)
    worst_ratio = grouped_ratios.gather(1, worst_idx).squeeze(1)
    ratio_gap = best_ratio - worst_ratio
    selected_ratio_gap = ratio_gap[valid]
    loss = torch.nn.functional.softplus(float(margin) - selected_ratio_gap).mean()
    stats = {
        "clasp/pairwise_valid_group_rate": float(valid.float().mean().item()),
        "clasp/pairwise_advantage_spread": float(advantage_spread[valid].mean().item()),
        "clasp/pairwise_log_ratio_gap": float(selected_ratio_gap.detach().mean().item()),
    }
    return loss, stats


def compute_elite_imitation_loss(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    num_generations: int,
    *,
    advantage_delta: float = 0.0,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Return a group-local imitation loss for the highest-advantage completion.

    The helper is default-off at the trainer level. It uses the CLASP
    advantage only to select one elite completion per prompt group, then
    applies a standard token-level NLL to that generated completion. This gives
    a direct probability-concentration signal while keeping inference as raw
    BoN sampling from a single checkpoint.
    """
    if per_token_logps.dim() != 2:
        raise ValueError("compute_elite_imitation_loss expects 2-D per-token log-probs.")
    if completion_mask.shape != per_token_logps.shape:
        raise ValueError("completion_mask/per_token_logps shape mismatch.")
    if advantages.dim() != 1:
        raise ValueError("compute_elite_imitation_loss expects 1-D advantages.")
    if advantages.numel() != per_token_logps.size(0):
        raise ValueError("advantage batch size must match per-token log-prob batch size.")
    if num_generations <= 1 or advantages.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {advantages.numel()} not divisible by "
            f"num_generations={num_generations}."
        )
    if advantage_delta < 0.0:
        raise ValueError(f"advantage_delta must be >= 0, got {advantage_delta}")

    K = int(num_generations)
    grouped_advantages = advantages.detach().to(torch.float32).view(-1, K)
    best_idx = grouped_advantages.argmax(dim=1, keepdim=True)
    best_adv = grouped_advantages.gather(1, best_idx).squeeze(1)
    worst_adv = grouped_advantages.min(dim=1).values
    advantage_spread = best_adv - worst_adv
    valid = advantage_spread >= float(advantage_delta)

    zero = per_token_logps.sum() * 0.0
    if not bool(valid.any().item()):
        stats = {
            "clasp/elite_imitation_valid_group_rate": 0.0,
            "clasp/elite_imitation_advantage_spread": 0.0,
            "clasp/elite_imitation_selected_nll": 0.0,
            "clasp/elite_imitation_selected_length": 0.0,
        }
        return zero, stats

    token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
    sequence_nll = -((per_token_logps * completion_mask).sum(dim=1) / token_counts)
    grouped_nll = sequence_nll.view(-1, K)
    grouped_lengths = token_counts.view(-1, K)

    selected_nll = grouped_nll.gather(1, best_idx).squeeze(1)
    selected_lengths = grouped_lengths.gather(1, best_idx).squeeze(1)
    loss = selected_nll[valid].mean()
    stats = {
        "clasp/elite_imitation_valid_group_rate": float(valid.float().mean().item()),
        "clasp/elite_imitation_advantage_spread": float(advantage_spread[valid].mean().item()),
        "clasp/elite_imitation_selected_nll": float(selected_nll[valid].detach().mean().item()),
        "clasp/elite_imitation_selected_length": float(selected_lengths[valid].detach().mean().item()),
    }
    return loss, stats


def compute_multi_elite_imitation_loss(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    raw_positive_gap_scores: torch.Tensor,
    feasibility_flags: torch.Tensor,
    num_generations: int,
    *,
    elite_fraction: float = 0.25,
    max_raw_gap: float = 0.0,
    advantage_delta: float = 0.0,
    temperature: float = 1.0,
    require_feasible: bool = True,
    fallback_to_best_feasible: bool = True,
) -> "tuple[torch.Tensor, dict[str, float]]":
    """Imitate several high-advantage completions per group.

    This default-off helper is a softer alternative to top-1 elite imitation.
    It selects a small group-local elite set using the CLASP advantage, with an
    optional feasible-and-low-gap filter, then applies a weighted NLL to the
    selected generated completions. The helper is train-time only; inference
    remains raw BoN sampling from one checkpoint.
    """
    if per_token_logps.dim() != 2:
        raise ValueError("compute_multi_elite_imitation_loss expects 2-D per-token log-probs.")
    if completion_mask.shape != per_token_logps.shape:
        raise ValueError("completion_mask/per_token_logps shape mismatch.")
    if advantages.dim() != 1:
        raise ValueError("compute_multi_elite_imitation_loss expects 1-D advantages.")
    if raw_positive_gap_scores.dim() != 1:
        raise ValueError("compute_multi_elite_imitation_loss expects 1-D raw gap scores.")
    if feasibility_flags.dim() != 1:
        raise ValueError("compute_multi_elite_imitation_loss expects 1-D feasibility flags.")
    if advantages.numel() != per_token_logps.size(0):
        raise ValueError("advantage batch size must match per-token log-prob batch size.")
    if raw_positive_gap_scores.numel() != advantages.numel():
        raise ValueError("raw-gap/advantage tensor size mismatch.")
    if feasibility_flags.numel() != advantages.numel():
        raise ValueError("feasibility/advantage tensor size mismatch.")
    if num_generations <= 1 or advantages.numel() % num_generations != 0:
        raise ValueError(
            f"Total sample count {advantages.numel()} not divisible by "
            f"num_generations={num_generations}."
        )
    if elite_fraction <= 0.0:
        raise ValueError(f"elite_fraction must be > 0, got {elite_fraction}")
    if max_raw_gap < 0.0:
        raise ValueError(f"max_raw_gap must be >= 0, got {max_raw_gap}")
    if advantage_delta < 0.0:
        raise ValueError(f"advantage_delta must be >= 0, got {advantage_delta}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    K = int(num_generations)
    grouped_advantages = advantages.detach().to(torch.float32).view(-1, K)
    grouped_raw_gap = raw_positive_gap_scores.detach().to(torch.float32).clamp(min=0.0).view(-1, K)
    grouped_feasible = (feasibility_flags.detach().view(-1, K) > 0.5)
    token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
    sequence_nll = -((per_token_logps * completion_mask).sum(dim=1) / token_counts)
    grouped_nll = sequence_nll.view(-1, K)
    grouped_lengths = token_counts.view(-1, K)

    zero = per_token_logps.sum() * 0.0
    top_count = max(1, min(K, int(math.ceil(float(elite_fraction) * K))))
    group_losses: list[torch.Tensor] = []
    selected_counts: list[float] = []
    selected_nlls: list[torch.Tensor] = []
    selected_lengths: list[torch.Tensor] = []
    selected_gaps: list[torch.Tensor] = []
    selected_feasible_rates: list[torch.Tensor] = []
    target_groups = 0
    fallback_groups = 0

    for group_idx in range(grouped_advantages.size(0)):
        adv = grouped_advantages[group_idx]
        raw_gap = grouped_raw_gap[group_idx]
        feasible = grouped_feasible[group_idx]
        advantage_spread = adv.max() - adv.min()
        if float(advantage_spread.item()) < float(advantage_delta):
            continue

        eligible = feasible if require_feasible else torch.ones_like(feasible, dtype=torch.bool)
        target_eligible = eligible
        if max_raw_gap > 0.0:
            target_eligible = eligible & (raw_gap <= float(max_raw_gap))

        use_fallback = False
        if bool(target_eligible.any().item()):
            chosen_pool = target_eligible
            target_groups += 1
        elif fallback_to_best_feasible and bool(eligible.any().item()):
            chosen_pool = eligible
            use_fallback = True
            fallback_groups += 1
        else:
            continue

        candidate_idx = torch.nonzero(chosen_pool, as_tuple=False).squeeze(1)
        candidate_adv = adv.index_select(0, candidate_idx)
        select_count = min(top_count, int(candidate_idx.numel()))
        order = torch.argsort(candidate_adv, descending=True)[:select_count]
        selected_idx = candidate_idx.index_select(0, order)
        selected_adv = grouped_advantages[group_idx].index_select(0, selected_idx)
        selected_nll = grouped_nll[group_idx].index_select(0, selected_idx)

        if temperature > 0.0 and selected_adv.numel() > 1:
            weights = torch.softmax(selected_adv / float(temperature), dim=0).to(selected_nll.dtype)
        else:
            weights = torch.full_like(selected_nll, 1.0 / float(max(int(selected_nll.numel()), 1)))

        group_losses.append((weights * selected_nll).sum())
        selected_counts.append(float(selected_idx.numel()))
        selected_nlls.append(selected_nll.detach().mean())
        selected_lengths.append(grouped_lengths[group_idx].index_select(0, selected_idx).detach().mean())
        selected_gaps.append(raw_gap.index_select(0, selected_idx).detach().mean())
        selected_feasible_rates.append(feasible.float().index_select(0, selected_idx).detach().mean())
        if use_fallback:
            # Kept only to make the branch obvious for future readers; the
            # counter was already incremented above.
            pass

    if not group_losses:
        stats = {
            "clasp/multi_elite_imitation_valid_group_rate": 0.0,
            "clasp/multi_elite_imitation_target_group_rate": 0.0,
            "clasp/multi_elite_imitation_fallback_group_rate": 0.0,
            "clasp/multi_elite_imitation_selected_count": 0.0,
            "clasp/multi_elite_imitation_selected_nll": 0.0,
            "clasp/multi_elite_imitation_selected_length": 0.0,
            "clasp/multi_elite_imitation_selected_raw_gap": 0.0,
            "clasp/multi_elite_imitation_selected_feasible_rate": 0.0,
        }
        return zero, stats

    num_groups = max(int(grouped_advantages.size(0)), 1)
    loss = torch.stack(group_losses).mean()
    stats = {
        "clasp/multi_elite_imitation_valid_group_rate": float(len(group_losses) / num_groups),
        "clasp/multi_elite_imitation_target_group_rate": float(target_groups / num_groups),
        "clasp/multi_elite_imitation_fallback_group_rate": float(fallback_groups / num_groups),
        "clasp/multi_elite_imitation_selected_count": float(sum(selected_counts) / len(selected_counts)),
        "clasp/multi_elite_imitation_selected_nll": float(torch.stack(selected_nlls).mean().item()),
        "clasp/multi_elite_imitation_selected_length": float(torch.stack(selected_lengths).mean().item()),
        "clasp/multi_elite_imitation_selected_raw_gap": float(torch.stack(selected_gaps).mean().item()),
        "clasp/multi_elite_imitation_selected_feasible_rate": float(
            torch.stack(selected_feasible_rates).mean().item()
        ),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Contribution B - Feasibility-conditional dual learning rate.
# ---------------------------------------------------------------------------

def feasibility_conditional_dual_lr(
    base_lr: float,
    feasibility_rate: float,
    *,
    eps: float = 1e-3,
) -> float:
    """eta(feas) = base_lr * sqrt((1 - feas + eps) / (feas + eps)).

    Low feas -> large dual step; high feas -> small dual step. Monotone
    decreasing in feasibility_rate over (0, 1].
    """
    if base_lr < 0.0:
        raise ValueError(f"base_lr must be >= 0, got {base_lr}")
    feas = float(max(eps, min(1.0 - eps, feasibility_rate)))
    return float(base_lr) * math.sqrt((1.0 - feas + eps) / (feas + eps))


# ---------------------------------------------------------------------------
# Contribution C - Task-adaptive completion length.
# ---------------------------------------------------------------------------

class AdaptiveLengthEstimator:
    """Rolling per-task quantile of observed completion lengths.

    The trainer calls ``observe(task, length)`` for every generated completion
    and ``budget(task)`` before sampling. ``maybe_refresh(step)`` is invoked
    at each logging boundary and recomputes budgets from the buffer every
    ``refresh_steps`` optimizer steps.

    The estimator is thread-safe (updates guarded by a lock), so it can be
    safely called from DataLoader workers and the main training loop.
    """

    def __init__(
        self,
        tasks: Iterable[str],
        *,
        global_cap: int,
        floor: int = 128,
        window: int = 1000,
        quantile: float = 0.99,
        refresh_steps: int = 50,
        margin: int = 32,
    ) -> None:
        if global_cap <= 0:
            raise ValueError(f"global_cap must be positive, got {global_cap}")
        if floor <= 0 or floor > global_cap:
            raise ValueError(f"floor must be in (0, global_cap], got floor={floor} cap={global_cap}")
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")

        self._cap = int(global_cap)
        self._floor = int(floor)
        self._window = int(window)
        self._quantile = float(quantile)
        self._refresh_steps = int(refresh_steps)
        self._margin = int(margin)
        self._lock = threading.Lock()
        self._buf: Dict[str, Deque[int]] = {
            str(t): collections.deque(maxlen=self._window) for t in tasks
        }
        self._budget: Dict[str, int] = {t: self._cap for t in self._buf}
        self._last_refresh_step = -1

    def observe(self, task: str, length: int) -> None:
        if length <= 0:
            return
        with self._lock:
            if task not in self._buf:
                self._buf[task] = collections.deque(maxlen=self._window)
                self._budget[task] = self._cap
            self._buf[task].append(int(length))

    def maybe_refresh(self, global_step: int) -> bool:
        with self._lock:
            if self._refresh_steps <= 0:
                return False
            if global_step - self._last_refresh_step < self._refresh_steps:
                return False
            for task, buf in self._buf.items():
                if len(buf) >= 32:
                    q = int(np.quantile(np.asarray(buf, dtype=np.int64), self._quantile))
                    self._budget[task] = int(
                        max(self._floor, min(self._cap, q + self._margin))
                    )
            self._last_refresh_step = int(global_step)
        return True

    def budget(self, task: str) -> int:
        with self._lock:
            return int(self._budget.get(task, self._cap))

    def snapshot(self) -> "dict[str, int]":
        with self._lock:
            return dict(self._budget)


__all__ = [
    "compute_bon_aligned_advantages",
    "compute_lexicographic_clasp_advantages",
    "compute_pareto_anchored_advantages",
    "compute_pairwise_preference_loss",
    "compute_elite_imitation_loss",
    "compute_multi_elite_imitation_loss",
    "update_signed_bon_lambdas",
    "feasibility_conditional_dual_lr",
    "AdaptiveLengthEstimator",
]
