"""Frontier-CDF CLASP v1 set utility and null-difference rewards.

This module deliberately has no dependency on TRL, Accelerate, a trainer, or
the task parsers.  It consumes the already-scored tensors produced by a strict
task evaluator and returns one advantage per sampled completion.

For each fixed-K prompt group, feasible objective scores are converted back to
clipped relative gaps using the current CLASP reward convention

    objective_score = 2 * exp(-3 * clipped_gap).

The quality utility is a quadrature approximation to the CDF of the best
feasible normalized gap.  Feasibility is enforced with a per-group Lagrange
multiplier.  If a group is entirely infeasible, a smooth minimum of absolute
normalized violation supplies dense recovery credit.  Per-candidate credit is
the full set utility minus the utility after replacing that candidate with a
fixed null candidate.  The replacement keeps K constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_THRESHOLD_MULTIPLIERS: tuple[float, ...] = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
)

# Right-Riemann quadrature widths on the default multiplier grid. The values
# are normalized inside the public function (their raw sum is three), so the
# quality utility lies in [0, 1].
DEFAULT_QUADRATURE_WEIGHTS: tuple[float, ...] = (
    0.5,
    0.25,
    0.25,
    0.25,
    0.25,
    0.5,
    1.0,
)


@dataclass(frozen=True)
class FrontierCDFResult:
    """Outputs of :func:`compute_frontier_cdf_advantages`.

    Candidate-shaped fields match the shape of the input scores (flat or
    ``[num_groups, K]``).  Group fields have shape ``[num_groups]`` and null
    utilities have shape ``[num_groups, K]``.
    """

    advantages: Tensor
    quality_credits: Tensor
    failure_credits: Tensor
    recovery_credits: Tensor
    normalized_gaps: Tensor
    group_utilities: Tensor
    null_group_utilities: Tensor
    quality_utilities: Tensor
    recovery_utilities: Tensor
    group_failures: Tensor
    stats: dict[str, float]


def _working_dtype(*values: Tensor) -> torch.dtype:
    return (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def _logit_bound(dtype: torch.dtype) -> float:
    # exp(+-80) already saturates float32 probabilities to machine precision;
    # float64 has enough exponent range to retain a wider useful interval.
    return 700.0 if dtype == torch.float64 else 80.0


def _candidate_matrix(value: Tensor, num_generations: int, name: str) -> tuple[Tensor, bool]:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dim() == 1:
        if value.numel() == 0 or value.numel() % num_generations != 0:
            raise ValueError(
                f"{name} has {value.numel()} entries, which is not a positive multiple "
                f"of num_generations={num_generations}."
            )
        return value.reshape(-1, num_generations), True
    if value.dim() == 2:
        if value.size(0) == 0 or value.size(1) != num_generations:
            raise ValueError(
                f"{name} must have shape [num_groups, {num_generations}], got "
                f"{tuple(value.shape)}."
            )
        return value, False
    raise ValueError(f"{name} must be one- or two-dimensional, got {value.dim()} dimensions.")


def _group_parameter(
    value: Tensor | float,
    *,
    num_groups: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    parameter = torch.as_tensor(value, device=device, dtype=dtype)
    if parameter.numel() == 1:
        return parameter.reshape(1).expand(num_groups)
    if parameter.dim() == 1 and parameter.numel() == num_groups:
        return parameter
    raise ValueError(f"{name} must be scalar or a length-{num_groups} tensor.")


def _quadrature_parameter(
    value: Tensor | Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    parameter = torch.as_tensor(value, device=device, dtype=dtype)
    if parameter.dim() != 1 or parameter.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    return parameter


def _restore_candidate_shape(value: Tensor, was_flat: bool) -> Tensor:
    return value.reshape(-1) if was_flat else value


def compute_frontier_cdf_advantages(
    objective_scores: Tensor,
    violation_scores: Tensor,
    feasibility_flags: Tensor,
    num_generations: int,
    *,
    group_lambdas: Tensor | float,
    group_target_gaps: Tensor | float,
    threshold_multipliers: Tensor | Sequence[float] = DEFAULT_THRESHOLD_MULTIPLIERS,
    quadrature_weights: Tensor | Sequence[float] = DEFAULT_QUADRATURE_WEIGHTS,
    quality_temperature: float = 0.15,
    violation_temperature: float = 0.15,
    recovery_scale: float = 1.0,
    objective_score_scale: float = 2.0,
    gap_slope: float = 3.0,
    objective_epsilon: float = 1e-12,
    feasibility_violation_tolerance: float | None = None,
) -> FrontierCDFResult:
    """Compute Frontier-CDF CLASP v1 null-difference advantages.

    Args:
        objective_scores: High-is-good scores following
            ``2 * exp(-3 * clipped_gap)`` for feasible candidates. Infeasible
            values are ignored and conventionally zero.
        violation_scores: Non-negative absolute violation scores. They are
            mapped to ``d = v / (1 + v)`` before recovery credit is computed.
        feasibility_flags: Boolean or exact 0/1 feasibility indicators.
        num_generations: Fixed number K of raw completions per prompt.
        group_lambdas: Scalar or one non-negative multiplier per prompt group.
        group_target_gaps: Scalar or one positive target gap per prompt group,
            in the same units as the gap encoded by ``objective_scores``.
        objective_score_scale: Leading scale in the declared objective mapping
            ``score = scale * exp(-slope * clipped_gap)``.
        gap_slope: Positive slope in the declared objective mapping.
        feasibility_violation_tolerance: If provided, fail closed unless every
            feasibility flag agrees with ``violation <= tolerance``. Leave
            ``None`` only when feasibility and the aggregate violation follow
            deliberately different evaluator contracts.

    The per-threshold candidate coverage is

    ``p_im = feasible_i * sigmoid((threshold_m - normalized_gap_i) / tau_q)``.

    Group quality is a normalized weighted noisy-OR over candidates.  The set
    utility is

    ``U = Q - lambda * I(group_failed) + recovery_scale * I(group_failed) * R``

    with ``R = -softmin(d)``.  Candidate ``i`` receives ``U(G) - U(G_i<-null)``
    where the null is infeasible, has objective score zero, and normalized
    violation one. No near-frontier or per-candidate feasibility auxiliary is
    included.
    """
    if not isinstance(num_generations, int) or isinstance(num_generations, bool):
        raise TypeError("num_generations must be an integer.")
    if num_generations <= 1:
        raise ValueError("num_generations must be greater than one.")
    for name, value in (
        ("quality_temperature", quality_temperature),
        ("violation_temperature", violation_temperature),
        ("objective_score_scale", objective_score_scale),
        ("gap_slope", gap_slope),
        ("objective_epsilon", objective_epsilon),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero.")
    if float(objective_epsilon) > 1.0:
        raise ValueError("objective_epsilon must be less than or equal to one.")
    if not math.isfinite(float(recovery_scale)) or float(recovery_scale) < 0.0:
        raise ValueError("recovery_scale must be finite and non-negative.")
    if feasibility_violation_tolerance is not None and (
        not math.isfinite(float(feasibility_violation_tolerance))
        or float(feasibility_violation_tolerance) < 0.0
    ):
        raise ValueError(
            "feasibility_violation_tolerance must be None or finite and non-negative."
        )

    objective, was_flat = _candidate_matrix(
        objective_scores, num_generations, "objective_scores"
    )
    violations, violations_were_flat = _candidate_matrix(
        violation_scores, num_generations, "violation_scores"
    )
    feasibility, feasibility_was_flat = _candidate_matrix(
        feasibility_flags, num_generations, "feasibility_flags"
    )
    if violations_were_flat != was_flat or feasibility_was_flat != was_flat:
        raise ValueError("All candidate inputs must use the same flat or matrix shape convention.")
    if violations.shape != objective.shape or feasibility.shape != objective.shape:
        raise ValueError(
            "objective, violation, and feasibility tensors must have identical shapes."
        )
    if violations.device != objective.device or feasibility.device != objective.device:
        raise ValueError("All candidate inputs must be on the same device.")

    dtype = _working_dtype(objective, violations, feasibility)
    device = objective.device
    objective = objective.to(dtype=dtype)
    violations = violations.to(dtype=dtype)
    raw_feasibility = feasibility.to(dtype=dtype)
    for name, value in (
        ("objective_scores", objective),
        ("violation_scores", violations),
        ("feasibility_flags", raw_feasibility),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains non-finite values.")
    if bool((objective < 0.0).any().item()):
        raise ValueError("objective_scores must be non-negative.")
    if bool((violations < 0.0).any().item()):
        raise ValueError("violation_scores must be non-negative.")
    if bool(((raw_feasibility != 0.0) & (raw_feasibility != 1.0)).any().item()):
        raise ValueError("feasibility_flags must contain exact boolean or 0/1 values.")
    feasible = raw_feasibility.bool()
    if feasibility_violation_tolerance is not None:
        violation_feasible = violations <= float(feasibility_violation_tolerance)
        inconsistent = feasible != violation_feasible
        if bool(inconsistent.any().item()):
            mismatch_count = int(inconsistent.sum().item())
            raise ValueError(
                "feasibility_flags disagree with violation_scores at "
                f"{mismatch_count} candidate(s) under "
                "feasibility_violation_tolerance="
                f"{float(feasibility_violation_tolerance):g}."
            )

    num_groups, K = objective.shape
    lambdas = _group_parameter(
        group_lambdas,
        num_groups=num_groups,
        device=device,
        dtype=dtype,
        name="group_lambdas",
    )
    target_gaps = _group_parameter(
        group_target_gaps,
        num_groups=num_groups,
        device=device,
        dtype=dtype,
        name="group_target_gaps",
    )
    if not bool(torch.isfinite(lambdas).all().item()) or bool((lambdas < 0.0).any().item()):
        raise ValueError("group_lambdas must be finite and non-negative.")
    if not bool(torch.isfinite(target_gaps).all().item()) or bool(
        (target_gaps <= 0.0).any().item()
    ):
        raise ValueError("group_target_gaps must be finite and greater than zero.")

    thresholds = _quadrature_parameter(
        threshold_multipliers,
        device=device,
        dtype=dtype,
        name="threshold_multipliers",
    )
    weights = _quadrature_parameter(
        quadrature_weights,
        device=device,
        dtype=dtype,
        name="quadrature_weights",
    )
    if thresholds.numel() != weights.numel():
        raise ValueError("threshold_multipliers and quadrature_weights must have equal length.")
    if not bool(torch.isfinite(thresholds).all().item()) or bool(
        (thresholds <= 0.0).any().item()
    ):
        raise ValueError("threshold_multipliers must be finite and greater than zero.")
    if thresholds.numel() > 1 and bool((thresholds[1:] <= thresholds[:-1]).any().item()):
        raise ValueError("threshold_multipliers must be strictly increasing.")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0.0).any().item()):
        raise ValueError("quadrature_weights must be finite and non-negative.")
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 0.0:
        raise ValueError("quadrature_weights must have a positive sum.")
    weights = weights / weight_sum

    # Invert q = scale*exp(-slope*g). Feasible scores above the declared scale
    # represent a better-than-clipped-zero gap and therefore remain at zero. A
    # positive lower clamp keeps underflowed finite scores well-defined.
    score_ratio = (objective / float(objective_score_scale)).clamp(
        min=float(objective_epsilon), max=1.0
    )
    clipped_gaps = (-torch.log(score_ratio) / float(gap_slope)).clamp_min(0.0)
    normalized_gaps = clipped_gaps / target_gaps.unsqueeze(1)
    normalized_gaps = torch.nan_to_num(
        normalized_gaps,
        nan=0.0,
        posinf=torch.finfo(dtype).max,
        neginf=0.0,
    )

    # log(1 - sigmoid((t-z)/tau)) = logsigmoid((z-t)/tau). Working in log
    # non-coverage space avoids unstable products when K or logits are large.
    quality_logits = (
        (normalized_gaps.unsqueeze(-1) - thresholds.reshape(1, 1, -1))
        / float(quality_temperature)
    ).clamp(min=-_logit_bound(dtype), max=_logit_bound(dtype))
    log_noncoverage_terms = F.logsigmoid(quality_logits)
    log_noncoverage_terms = torch.where(
        feasible.unsqueeze(-1),
        log_noncoverage_terms,
        torch.zeros_like(log_noncoverage_terms),
    )
    log_noncoverage = log_noncoverage_terms.sum(dim=1)
    threshold_coverage = -torch.expm1(log_noncoverage)
    quality_utilities = (threshold_coverage * weights).sum(dim=-1)

    # Replacing candidate i by the null sets all of its p_im values to zero,
    # hence its log non-coverage contribution becomes exactly zero.
    null_log_noncoverage = (
        log_noncoverage.unsqueeze(1) - log_noncoverage_terms
    )
    null_threshold_coverage = -torch.expm1(null_log_noncoverage)
    null_quality_utilities = (null_threshold_coverage * weights).sum(dim=-1)

    normalized_violations = violations / (1.0 + violations)
    violation_logits = (
        -normalized_violations / float(violation_temperature)
    ).clamp(min=-_logit_bound(dtype), max=0.0)
    recovery_utilities = float(violation_temperature) * (
        torch.logsumexp(
            violation_logits, dim=1
        )
        - math.log(float(K))
    )

    # [B, replacement_i, original_j]. Each row is the original fixed-K set
    # with one normalized violation replaced by the null value one.
    diagonal = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
    null_violation_sets = torch.where(
        diagonal,
        torch.ones((1, K, K), dtype=dtype, device=device),
        normalized_violations.unsqueeze(1),
    )
    null_violation_logits = (
        -null_violation_sets / float(violation_temperature)
    ).clamp(min=-_logit_bound(dtype), max=0.0)
    null_recovery_utilities = float(violation_temperature) * (
        torch.logsumexp(
            null_violation_logits, dim=-1
        )
        - math.log(float(K))
    )

    feasible_counts = feasible.sum(dim=1)
    group_failures = feasible_counts == 0
    null_group_failures = (feasible_counts.unsqueeze(1) - feasible.to(torch.long)) == 0
    failure_values = group_failures.to(dtype=dtype)
    null_failure_values = null_group_failures.to(dtype=dtype)

    group_utilities = (
        quality_utilities
        - lambdas * failure_values
        + float(recovery_scale) * failure_values * recovery_utilities
    )
    null_group_utilities = (
        null_quality_utilities
        - lambdas.unsqueeze(1) * null_failure_values
        + float(recovery_scale) * null_failure_values * null_recovery_utilities
    )

    quality_credits = quality_utilities.unsqueeze(1) - null_quality_utilities
    failure_credits = lambdas.unsqueeze(1) * (
        null_failure_values - failure_values.unsqueeze(1)
    )
    recovery_credits = float(recovery_scale) * (
        failure_values.unsqueeze(1) * recovery_utilities.unsqueeze(1)
        - null_failure_values * null_recovery_utilities
    )
    direct_advantages = group_utilities.unsqueeze(1) - null_group_utilities

    # This is both a semantic invariant and an exactness guarantee required by
    # the set objective: once any other completion succeeds, changing an
    # infeasible candidate to the null cannot affect utility.
    noncritical_invalid = (~feasible) & (~group_failures.unsqueeze(1))
    zeros = torch.zeros((), dtype=dtype, device=device)
    quality_credits = torch.where(noncritical_invalid, zeros, quality_credits)
    failure_credits = torch.where(noncritical_invalid, zeros, failure_credits)
    recovery_credits = torch.where(noncritical_invalid, zeros, recovery_credits)
    direct_advantages = torch.where(noncritical_invalid, zeros, direct_advantages)

    # The null is weakly worse than any normalized input violation and has no
    # quality or feasibility. Exact arithmetic therefore gives non-negative
    # marginal credit. Clamp only roundoff-scale negative residues.
    quality_credits = quality_credits.clamp_min(0.0)
    failure_credits = failure_credits.clamp_min(0.0)
    recovery_credits = recovery_credits.clamp_min(0.0)
    # Construct the optimized result from its reported components so the
    # decomposition is exact by construction. The direct utility subtraction
    # is retained above as an independently useful numerical diagnostic.
    advantages = quality_credits + failure_credits + recovery_credits
    decomposition_error = (advantages - direct_advantages.clamp_min(0.0)).abs().max()

    successful = ~group_failures
    failed_count = int(group_failures.sum().item())
    singleton = feasible_counts == 1
    feasible_normalized_gaps = normalized_gaps[feasible]
    masked_normalized_gaps = normalized_gaps.masked_fill(~feasible, float("inf"))
    best_feasible_normalized_gaps = masked_normalized_gaps.min(dim=1).values[
        successful
    ]
    group_quality_credit_mass = quality_credits.sum(dim=1)

    def quantile_or_zero(values: Tensor, quantile: float) -> float:
        if values.numel() == 0:
            return 0.0
        return float(torch.quantile(values, quantile).item())

    stats = {
        "frontier_cdf/group_failure_rate": float(group_failures.to(dtype).mean().item()),
        "frontier_cdf/singleton_feasible_group_rate": float(singleton.to(dtype).mean().item()),
        "frontier_cdf/multiple_feasible_group_rate": float(
            (feasible_counts >= 2).to(dtype).mean().item()
        ),
        "frontier_cdf/quality_utility_mean": float(quality_utilities.mean().item()),
        "frontier_cdf/quality_utility_success_mean": float(
            quality_utilities[successful].mean().item() if bool(successful.any().item()) else 0.0
        ),
        "frontier_cdf/recovery_utility_failed_mean": float(
            recovery_utilities[group_failures].mean().item() if failed_count else 0.0
        ),
        "frontier_cdf/advantage_mean": float(advantages.mean().item()),
        "frontier_cdf/advantage_abs_max": float(advantages.abs().max().item()),
        "frontier_cdf/zero_advantage_rate": float((advantages == 0.0).to(dtype).mean().item()),
        "frontier_cdf/quality_credit_mean": float(quality_credits.mean().item()),
        "frontier_cdf/quality_credit_group_mass_mean": float(
            group_quality_credit_mass.mean().item()
        ),
        "frontier_cdf/quality_credit_group_nonzero_rate": float(
            (group_quality_credit_mass > 1e-8).to(dtype).mean().item()
        ),
        "frontier_cdf/failure_credit_mean": float(failure_credits.mean().item()),
        "frontier_cdf/recovery_credit_mean": float(recovery_credits.mean().item()),
        "frontier_cdf/feasible_normalized_gap_p50": quantile_or_zero(
            feasible_normalized_gaps, 0.50
        ),
        "frontier_cdf/feasible_normalized_gap_p90": quantile_or_zero(
            feasible_normalized_gaps, 0.90
        ),
        "frontier_cdf/best_feasible_normalized_gap_p50": quantile_or_zero(
            best_feasible_normalized_gaps, 0.50
        ),
        "frontier_cdf/best_feasible_normalized_gap_p90": quantile_or_zero(
            best_feasible_normalized_gaps, 0.90
        ),
        "frontier_cdf/null_decomposition_error_max": float(decomposition_error.item()),
        "frontier_cdf/mean_lambda": float(lambdas.mean().item()),
        "frontier_cdf/mean_target_gap": float(target_gaps.mean().item()),
        "frontier_cdf/quality_temperature": float(quality_temperature),
        "frontier_cdf/violation_temperature": float(violation_temperature),
        "frontier_cdf/recovery_scale": float(recovery_scale),
        "frontier_cdf/objective_score_scale": float(objective_score_scale),
        "frontier_cdf/gap_slope": float(gap_slope),
    }

    return FrontierCDFResult(
        advantages=_restore_candidate_shape(advantages, was_flat),
        quality_credits=_restore_candidate_shape(quality_credits, was_flat),
        failure_credits=_restore_candidate_shape(failure_credits, was_flat),
        recovery_credits=_restore_candidate_shape(recovery_credits, was_flat),
        normalized_gaps=_restore_candidate_shape(normalized_gaps, was_flat),
        group_utilities=group_utilities,
        null_group_utilities=null_group_utilities,
        quality_utilities=quality_utilities,
        recovery_utilities=recovery_utilities,
        group_failures=group_failures,
        stats=stats,
    )
