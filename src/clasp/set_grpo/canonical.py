"""Canonical finite-batch advantages for symmetric fixed-cardinality utilities.

For one prompt, draw a rollout batch of ``B`` conditionally i.i.d. responses
from the current policy, while the deployed set utility consumes ``K < B``
responses.  Given the utility of every size-``K`` subset, this module computes

``u_i``
    the average utility of subsets that contain response ``i``;
``v_i``
    the average utility of subsets that exclude response ``i``; and
``A_i = u_i - v_i``
    the canonical finite-batch leave-out advantage.

For every realized batch, exact finite-subset counting gives
``sum_i A_i = 0`` in real arithmetic.  If the utility is integrable,
permutation invariant, has no direct policy-parameter dependence, and its
configuration is fixed independently of the current responses, then

``(K / B) * sum_i A_i * grad(log pi(response_i))``

is an unbiased on-policy score-function estimator of the expected deployed
size-``K`` utility.  Consequently, a loss that *averages* over the ``B``
responses must use ``K * A_i`` as its per-response coefficient.  PPO clipping,
off-policy reuse, action-dependent normalization, correlated rollouts, and
token-length normalization are outside this guarantee.

The functions here contain no trainer, evaluator, or controller state.  They
operate only on already evaluated subset utilities and feasibility flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import torch
from torch import Tensor


@dataclass(frozen=True)
class CanonicalSetAdvantageResult:
    """Outputs of :func:`canonical_finite_batch_advantages`.

    All candidate-shaped fields have the leading shape of ``subset_utilities``
    followed by ``batch_size``. ``subset_mean_utilities`` and
    ``centering_residuals`` have only the leading shape.

    ``mean_reduced_policy_coefficients`` is exactly ``K * advantages``.  It is
    the correctly scaled coefficient for a sequence-level policy-gradient loss
    whose outer reduction is a mean over the ``B`` responses.  It does not make
    a clipped, reused, or token-length-normalized surrogate unbiased.

    ``centering_residuals`` records floating-point error in an identity that is
    exact in real arithmetic.  The implementation deliberately does not apply
    an additional sample-dependent numerical recentering step.
    """

    conditional_utilities: Tensor
    leave_out_utilities: Tensor
    advantages: Tensor
    mean_reduced_policy_coefficients: Tensor
    subset_mean_utilities: Tensor
    centering_residuals: Tensor


@dataclass(frozen=True)
class CanonicalClaspSetResult:
    """Canonical CLASP credit for a batch of scored rollout groups.

    Candidate-shaped tensors follow the input convention: flat inputs return
    flat tensors, while ``[num_groups, B]`` inputs return matrices.  Subset
    tensors always have shape ``[num_groups, comb(B, K)]`` because controller
    observations and component accounting are group-level quantities.

    ``advantages`` contains the canonical leave-out advantages ``A``.
    ``mean_reduced_policy_coefficients`` contains ``K * A`` and is the value a
    loss whose outer reduction is a mean over the ``B`` sequences must consume.
    The component advantages are unscaled, so the exact decomposition is

    ``A = quality_A + rho * recovery_A - lambda * failure_A``.
    """

    advantages: Tensor
    mean_reduced_policy_coefficients: Tensor
    quality_advantages: Tensor
    recovery_advantages: Tensor
    failure_advantages: Tensor
    subset_indices: Tensor
    subset_utilities: Tensor
    subset_quality_utilities: Tensor
    subset_recovery_utilities: Tensor
    subset_failures: Tensor
    deployment_failure_estimates: Tensor
    centering_residuals: Tensor


@dataclass(frozen=True)
class DenseClaspAuxiliaryResult:
    """Dense, centered auxiliary credit for canonical CLASP.

    This is deliberately reported separately from the exact finite-set
    estimator.  It supplies lower-variance lexicographic credit to candidates
    that the hard max / binary-failure set utility cannot distinguish.
    """

    advantages: Tensor
    quality_advantages: Tensor
    feasibility_advantages: Tensor
    violation_advantages: Tensor
    recovery_advantages: Tensor
    all_infeasible_groups: Tensor
    mixed_groups: Tensor
    all_feasible_groups: Tensor
    centering_residuals: Tensor


def _strict_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return int(value)


@lru_cache(maxsize=16)
def _immutable_k_subset_indices(batch_size: int, set_size: int) -> tuple[tuple[int, ...], ...]:
    """Cache only immutable indices; returned tensors must never alias a cache."""

    return tuple(combinations(range(batch_size), set_size))


def enumerate_k_subsets(
    batch_size: int,
    set_size: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return every increasing size-``set_size`` subset of ``range(batch_size)``.

    The output has shape ``[comb(batch_size, set_size), set_size]``, dtype
    ``torch.long``, and resides on ``device``.  The strict ``set_size <
    batch_size`` requirement is the leave-out condition: without at least one
    extra rollout, ``v_i`` is undefined.
    """

    B = _strict_positive_int(batch_size, "batch_size")
    K = _strict_positive_int(set_size, "set_size")
    if K >= B:
        raise ValueError("set_size must be strictly less than batch_size.")

    # torch.combinations constructs a B**K Cartesian-product mask in current
    # PyTorch implementations, even when comb(B, K) is small.  Itertools emits
    # only the required lexicographically ordered rows.  Copy the immutable
    # cached tuples into a fresh tensor so caller mutation cannot corrupt later
    # enumerations or couple devices.
    return torch.tensor(
        _immutable_k_subset_indices(B, K),
        dtype=torch.long,
        device=device,
    )


def _validate_complete_subset_system(
    subset_indices: Tensor,
    *,
    batch_size: int,
    expected_device: torch.device,
) -> tuple[int, int]:
    if not isinstance(subset_indices, Tensor):
        raise TypeError("subset_indices must be a torch.Tensor.")
    if subset_indices.dim() != 2:
        raise ValueError("subset_indices must have shape [num_subsets, set_size].")
    if subset_indices.dtype != torch.long:
        raise TypeError("subset_indices must have dtype torch.long.")
    if subset_indices.device != expected_device:
        raise ValueError("subset_indices and subset_utilities must be on the same device.")

    num_subsets, K = subset_indices.shape
    if K <= 0:
        raise ValueError("subset_indices must contain at least one index per subset.")
    if K >= batch_size:
        raise ValueError("set_size must be strictly less than batch_size.")
    expected_subsets = math.comb(batch_size, K)
    if num_subsets != expected_subsets:
        raise ValueError(
            "subset_indices must contain every size-"
            f"{K} subset exactly once: expected {expected_subsets} rows, "
            f"got {num_subsets}."
        )
    if bool(((subset_indices < 0) | (subset_indices >= batch_size)).any().item()):
        raise ValueError("subset_indices contains an index outside [0, batch_size).")
    if K > 1 and bool((subset_indices[:, 1:] <= subset_indices[:, :-1]).any().item()):
        raise ValueError("Every row of subset_indices must be strictly increasing.")

    # With valid increasing rows and exactly comb(B, K) rows, uniqueness is
    # equivalent to completeness. Accept any row order so utilities can be
    # streamed or permuted together with their index rows.
    unique_rows = torch.unique(subset_indices, dim=0)
    if int(unique_rows.size(0)) != expected_subsets:
        raise ValueError("subset_indices contains duplicate rows or omits a required subset.")
    return int(num_subsets), int(K)


def canonical_finite_batch_advantages(
    subset_utilities: Tensor,
    subset_indices: Tensor,
    *,
    batch_size: int,
) -> CanonicalSetAdvantageResult:
    """Compute canonical leave-out advantages from an exact subset utility table.

    Args:
        subset_utilities: Floating-point tensor with shape
            ``[..., comb(batch_size, K)]``. The final axis supplies one utility
            for each row of ``subset_indices``; arbitrary leading axes are
            preserved and may represent prompt groups or utility components.
        subset_indices: Complete, duplicate-free collection of increasing
            size-``K`` index rows. Row order is arbitrary but must match the
            final axis of ``subset_utilities``.
        batch_size: Rollout count ``B``. The inferred subset size must satisfy
            ``1 <= K < B``.

    Returns:
        Conditional utilities ``u_i``, leave-out utilities ``v_i``, centered
        canonical advantages ``A_i = u_i - v_i``, and ``K * A_i`` coefficients
        for an outer mean-reduced sequence-level score-function loss.

    The policy-gradient theorem additionally assumes conditionally i.i.d.
    on-policy rollouts, a symmetric integrable size-``K`` utility, no direct
    utility dependence on policy parameters, and utility/controller
    configuration fixed without inspecting the current rollout responses.
    This pure algebra function cannot enforce those statistical assumptions.
    """

    B = _strict_positive_int(batch_size, "batch_size")
    if not isinstance(subset_utilities, Tensor):
        raise TypeError("subset_utilities must be a torch.Tensor.")
    if subset_utilities.dim() < 1:
        raise ValueError("subset_utilities must have at least one dimension.")
    if not subset_utilities.is_floating_point():
        raise TypeError("subset_utilities must have a floating-point dtype.")
    if not bool(torch.isfinite(subset_utilities).all().item()):
        raise ValueError("subset_utilities contains non-finite values.")

    num_subsets, K = _validate_complete_subset_system(
        subset_indices,
        batch_size=B,
        expected_device=subset_utilities.device,
    )
    if int(subset_utilities.size(-1)) != num_subsets:
        raise ValueError(
            "The final subset_utilities dimension must equal the number of "
            f"subset_indices rows ({num_subsets})."
        )

    membership = torch.zeros(
        (num_subsets, B),
        dtype=subset_utilities.dtype,
        device=subset_utilities.device,
    )
    membership.scatter_(1, subset_indices, 1.0)
    nonmembership = 1.0 - membership

    leading_shape = subset_utilities.shape[:-1]
    flat_utilities = subset_utilities.reshape(-1, num_subsets)
    contain_count = math.comb(B - 1, K - 1)
    exclude_count = math.comb(B - 1, K)
    conditional = (flat_utilities @ membership) / contain_count
    leave_out = (flat_utilities @ nonmembership) / exclude_count
    conditional = conditional.reshape(*leading_shape, B)
    leave_out = leave_out.reshape(*leading_shape, B)

    advantages = conditional - leave_out
    mean_coefficients = float(K) * advantages
    subset_means = subset_utilities.mean(dim=-1)
    centering_residuals = advantages.sum(dim=-1)

    return CanonicalSetAdvantageResult(
        conditional_utilities=conditional,
        leave_out_utilities=leave_out,
        advantages=advantages,
        mean_reduced_policy_coefficients=mean_coefficients,
        subset_mean_utilities=subset_means,
        centering_residuals=centering_residuals,
    )


def deployment_failure_u_statistic(
    feasibility_flags: Tensor,
    *,
    set_size: int,
) -> Tensor:
    """Estimate deployment failure@``K`` from a larger rollout batch.

    ``feasibility_flags`` has shape ``[..., B]`` and contains exact boolean or
    0/1 values, where one means feasible. If ``n_bad`` candidates are
    infeasible, the returned statistic is

    ``comb(n_bad, K) / comb(B, K)``.

    Under conditionally i.i.d. rollouts this is an unbiased U-statistic for the
    probability that all ``K`` deployed responses are infeasible. It is not the
    full-rollout failure indicator unless ``K == B``; this function requires
    the theorem-backed leave-out regime ``K < B``.

    Floating inputs retain their dtype and device. Boolean or integer flags
    return ``torch.float32`` on the input device because fractional failure
    probabilities cannot be represented in their source dtype.
    """

    K = _strict_positive_int(set_size, "set_size")
    if not isinstance(feasibility_flags, Tensor):
        raise TypeError("feasibility_flags must be a torch.Tensor.")
    if feasibility_flags.dim() < 1 or feasibility_flags.size(-1) <= 0:
        raise ValueError("feasibility_flags must have a non-empty final batch dimension.")
    B = int(feasibility_flags.size(-1))
    if K >= B:
        raise ValueError("set_size must be strictly less than the rollout batch size.")

    if feasibility_flags.is_complex():
        raise TypeError("feasibility_flags must be boolean, integer, or floating point.")
    if feasibility_flags.is_floating_point() and not bool(
        torch.isfinite(feasibility_flags).all().item()
    ):
        raise ValueError("feasibility_flags contains non-finite values.")
    if bool(((feasibility_flags != 0) & (feasibility_flags != 1)).any().item()):
        raise ValueError("feasibility_flags must contain exact boolean or 0/1 values.")

    output_dtype = (
        feasibility_flags.dtype
        if feasibility_flags.is_floating_point()
        else torch.float32
    )
    flags = feasibility_flags.to(dtype=output_dtype)
    infeasible_counts = float(B) - flags.sum(dim=-1)

    probability = torch.ones_like(infeasible_counts)
    for offset in range(K):
        numerator = (infeasible_counts - float(offset)).clamp_min(0.0)
        probability = probability * (numerator / float(B - offset))
    return torch.where(
        infeasible_counts >= float(K),
        probability,
        torch.zeros_like(probability),
    )


def _canonical_candidate_matrix(
    value: Tensor,
    *,
    rollout_batch_size: int,
    name: str,
) -> tuple[Tensor, bool]:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dim() == 1:
        if value.numel() == 0 or value.numel() % rollout_batch_size != 0:
            raise ValueError(
                f"{name} must contain a positive multiple of rollout_batch_size="
                f"{rollout_batch_size} values."
            )
        return value.reshape(-1, rollout_batch_size), True
    if value.dim() == 2:
        if value.size(0) == 0 or value.size(1) != rollout_batch_size:
            raise ValueError(
                f"{name} must have shape [num_groups, {rollout_batch_size}], "
                f"got {tuple(value.shape)}."
            )
        return value, False
    raise ValueError(f"{name} must be one- or two-dimensional.")


def _canonical_group_parameter(
    value: Tensor | float,
    *,
    num_groups: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    nonnegative: bool,
) -> Tensor:
    parameter = torch.as_tensor(value, device=device, dtype=dtype)
    if parameter.numel() == 1:
        parameter = parameter.reshape(1).expand(num_groups)
    elif parameter.dim() != 1 or parameter.numel() != num_groups:
        raise ValueError(f"{name} must be scalar or a length-{num_groups} tensor.")
    if not bool(torch.isfinite(parameter).all().item()):
        raise ValueError(f"{name} contains non-finite values.")
    if nonnegative and bool((parameter < 0.0).any().item()):
        raise ValueError(f"{name} must be non-negative.")
    return parameter


def _masked_pairwise_midrank(
    values: Tensor,
    mask: Tensor,
    *,
    tie_tolerance: float,
) -> Tensor:
    """Bounded high-is-good midranks, centered over each selected row."""

    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    differences = values.unsqueeze(2) - values.unsqueeze(1)
    wins = (differences > float(tie_tolerance)) & pair_mask
    losses = (differences < -float(tie_tolerance)) & pair_mask
    counts = mask.sum(dim=1, keepdim=True)
    denominator = (counts - 1).clamp_min(1).to(dtype=values.dtype)
    scores = (
        wins.to(dtype=values.dtype).sum(dim=2)
        - losses.to(dtype=values.dtype).sum(dim=2)
    ) / denominator
    return torch.where(mask & (counts > 1), scores, torch.zeros_like(scores))


def compute_dense_clasp_auxiliary_advantages(
    objective_scores: Tensor,
    violation_scores: Tensor,
    feasibility_flags: Tensor,
    *,
    rollout_batch_size: int,
    group_lambdas: Tensor | float,
    quality_scale: float = 0.0,
    feasibility_scale: float = 0.0,
    violation_scale: float = 0.0,
    recovery_scale: float = 0.0,
    tie_tolerance: float = 1e-6,
) -> DenseClaspAuxiliaryResult:
    """Compute dense lexicographic credit that complements exact set credit.

    Exact best-of-K credit is intentionally sparse: only frontier-quality
    candidates matter, while binary failure credit is flat in many realized
    rollout batches.  This auxiliary keeps the exact estimator intact and adds
    a centered, bounded control variate-like training signal:

    * mixed batches reinforce every feasible candidate against every
      infeasible candidate, rank feasible candidates by quality, and rank
      infeasible candidates by lower evaluator violation;
    * all-infeasible batches rank every candidate by lower violation;
    * all-feasible batches rank every candidate by quality.

    Constraint terms are multiplied by ``1 + lambda`` using a multiplier that
    is frozen for the complete prompt group.  Every component is a pairwise
    midrank or centered indicator, so task-specific violation magnitudes cannot
    dominate the policy update.  The result is centered exactly up to floating
    point error but is a deliberately biased dense surrogate, not part of the
    unbiased canonical finite-set theorem.
    """

    B = _strict_positive_int(rollout_batch_size, "rollout_batch_size")
    if B <= 1:
        raise ValueError("rollout_batch_size must be greater than one.")
    for name, value in (
        ("quality_scale", quality_scale),
        ("feasibility_scale", feasibility_scale),
        ("violation_scale", violation_scale),
        ("recovery_scale", recovery_scale),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) <= 0.0:
        raise ValueError("tie_tolerance must be finite and greater than zero.")

    objective, objective_was_flat = _canonical_candidate_matrix(
        objective_scores,
        rollout_batch_size=B,
        name="objective_scores",
    )
    violations, violations_were_flat = _canonical_candidate_matrix(
        violation_scores,
        rollout_batch_size=B,
        name="violation_scores",
    )
    feasibility, feasibility_was_flat = _canonical_candidate_matrix(
        feasibility_flags,
        rollout_batch_size=B,
        name="feasibility_flags",
    )
    if violations_were_flat != objective_was_flat or feasibility_was_flat != objective_was_flat:
        raise ValueError("All candidate inputs must use the same flat or matrix convention.")
    if objective.shape != violations.shape or objective.shape != feasibility.shape:
        raise ValueError("objective, violation, and feasibility tensors must have equal shapes.")
    if objective.device != violations.device or objective.device != feasibility.device:
        raise ValueError("All candidate inputs must be on the same device.")

    dtype = (
        torch.float64
        if any(value.dtype == torch.float64 for value in (objective, violations, feasibility))
        else torch.float32
    )
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
        raise ValueError("objective_scores must be non-negative high-is-good values.")
    if bool((violations < 0.0).any().item()):
        raise ValueError("violation_scores must be non-negative.")
    if bool(((raw_feasibility != 0.0) & (raw_feasibility != 1.0)).any().item()):
        raise ValueError("feasibility_flags must contain exact boolean or 0/1 values.")

    num_groups = int(objective.size(0))
    lambdas = _canonical_group_parameter(
        group_lambdas,
        num_groups=num_groups,
        device=objective.device,
        dtype=dtype,
        name="group_lambdas",
        nonnegative=True,
    ).unsqueeze(1)
    feasible = raw_feasibility.bool()
    infeasible = ~feasible
    feasible_count = feasible.sum(dim=1, keepdim=True)
    all_infeasible = feasible_count == 0
    all_feasible = feasible_count == B
    mixed = ~(all_infeasible | all_feasible)

    quality_credit = _masked_pairwise_midrank(
        objective,
        feasible,
        tie_tolerance=float(tie_tolerance),
    )
    violation_credit = _masked_pairwise_midrank(
        -violations,
        infeasible,
        tie_tolerance=float(tie_tolerance),
    )
    centered_feasibility = (
        raw_feasibility - raw_feasibility.mean(dim=1, keepdim=True)
    )
    constraint_pressure = 1.0 + lambdas

    quality_advantages = float(quality_scale) * quality_credit
    feasibility_advantages = torch.where(
        mixed,
        float(feasibility_scale) * constraint_pressure * centered_feasibility,
        torch.zeros_like(centered_feasibility),
    )
    violation_advantages = torch.where(
        mixed,
        float(violation_scale) * constraint_pressure * violation_credit,
        torch.zeros_like(violation_credit),
    )
    recovery_advantages = torch.where(
        all_infeasible,
        float(recovery_scale) * constraint_pressure * violation_credit,
        torch.zeros_like(violation_credit),
    )
    advantages = (
        quality_advantages
        + feasibility_advantages
        + violation_advantages
        + recovery_advantages
    )
    centering_residuals = advantages.sum(dim=1)

    def restore(value: Tensor) -> Tensor:
        return value.reshape(-1) if objective_was_flat and value.dim() == 2 else value

    return DenseClaspAuxiliaryResult(
        advantages=restore(advantages),
        quality_advantages=restore(quality_advantages),
        feasibility_advantages=restore(feasibility_advantages),
        violation_advantages=restore(violation_advantages),
        recovery_advantages=restore(recovery_advantages),
        all_infeasible_groups=all_infeasible.squeeze(1),
        mixed_groups=mixed.squeeze(1),
        all_feasible_groups=all_feasible.squeeze(1),
        centering_residuals=centering_residuals,
    )


def compute_canonical_clasp_set_advantages(
    objective_scores: Tensor,
    violation_scores: Tensor,
    feasibility_flags: Tensor,
    *,
    rollout_batch_size: int,
    deployment_set_size: int,
    group_lambdas: Tensor | float,
    recovery_scale: Tensor | float = 1.0,
    recovery_temperature: float = 0.15,
    target_failure_rates: Tensor | float = 0.0,
) -> CanonicalClaspSetResult:
    """Evaluate the symmetric CLASP utility on every size-``K`` subset.

    The rollout batch contains ``B=rollout_batch_size`` conditionally iid
    candidates and the deployment selector consumes
    ``K=deployment_set_size < B`` candidates.  For each size-``K`` subset the
    strict utility is

    ``max feasible objective + rho * failure * recovery - lambda * failure``.

    The declared Lagrangian also contains ``+lambda * target``. It is constant
    across every subset in a prompt group, so it is validated here but omitted
    before floating-point credit computation; the canonical difference cancels
    it exactly in real arithmetic.

    Infeasible candidates never receive objective-quality credit.  When the
    subset is entirely infeasible, violations are mapped to
    ``d = violation / (1 + violation)`` and the bounded recovery is

    ``temperature * (logsumexp(-d / temperature) - log(K))``.

    This recovery lies in ``[-1, 0]`` up to floating-point error and is zeroed
    for successful subsets.  The function then applies the exact canonical
    leave-out transform and returns the correctly scaled ``K * A`` policy
    coefficient for an outer sequence-mean loss.
    """

    B = _strict_positive_int(rollout_batch_size, "rollout_batch_size")
    K = _strict_positive_int(deployment_set_size, "deployment_set_size")
    if K >= B:
        raise ValueError(
            "deployment_set_size must be strictly less than rollout_batch_size."
        )
    if not math.isfinite(float(recovery_temperature)) or float(recovery_temperature) <= 0.0:
        raise ValueError("recovery_temperature must be finite and greater than zero.")

    objective, objective_was_flat = _canonical_candidate_matrix(
        objective_scores,
        rollout_batch_size=B,
        name="objective_scores",
    )
    violations, violations_were_flat = _canonical_candidate_matrix(
        violation_scores,
        rollout_batch_size=B,
        name="violation_scores",
    )
    feasibility, feasibility_was_flat = _canonical_candidate_matrix(
        feasibility_flags,
        rollout_batch_size=B,
        name="feasibility_flags",
    )
    if violations_were_flat != objective_was_flat or feasibility_was_flat != objective_was_flat:
        raise ValueError("All candidate inputs must use the same flat or matrix convention.")
    if objective.shape != violations.shape or objective.shape != feasibility.shape:
        raise ValueError("objective, violation, and feasibility tensors must have equal shapes.")
    if objective.device != violations.device or objective.device != feasibility.device:
        raise ValueError("All candidate inputs must be on the same device.")

    dtype = (
        torch.float64
        if any(value.dtype == torch.float64 for value in (objective, violations, feasibility))
        else torch.float32
    )
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
        raise ValueError("objective_scores must be non-negative high-is-good values.")
    if bool((violations < 0.0).any().item()):
        raise ValueError("violation_scores must be non-negative.")
    if bool(((raw_feasibility != 0.0) & (raw_feasibility != 1.0)).any().item()):
        raise ValueError("feasibility_flags must contain exact boolean or 0/1 values.")

    num_groups = int(objective.size(0))
    lambdas = _canonical_group_parameter(
        group_lambdas,
        num_groups=num_groups,
        device=device,
        dtype=dtype,
        name="group_lambdas",
        nonnegative=True,
    )
    recovery_scales = _canonical_group_parameter(
        recovery_scale,
        num_groups=num_groups,
        device=device,
        dtype=dtype,
        name="recovery_scale",
        nonnegative=True,
    )
    targets = _canonical_group_parameter(
        target_failure_rates,
        num_groups=num_groups,
        device=device,
        dtype=dtype,
        name="target_failure_rates",
        nonnegative=True,
    )
    if bool((targets > 1.0).any().item()):
        raise ValueError("target_failure_rates must lie in [0, 1].")

    subsets = enumerate_k_subsets(B, K, device=device)
    selected_feasible = raw_feasibility.bool()[:, subsets]
    selected_objective = objective[:, subsets]
    subset_success = selected_feasible.any(dim=-1)
    negative_infinity = torch.full_like(selected_objective, -torch.inf)
    feasible_objective = torch.where(
        selected_feasible,
        selected_objective,
        negative_infinity,
    )
    subset_quality = torch.where(
        subset_success,
        feasible_objective.max(dim=-1).values,
        torch.zeros_like(subset_success, dtype=dtype),
    )

    normalized_violations = violations / (1.0 + violations)
    selected_violations = normalized_violations[:, subsets]
    temperature = float(recovery_temperature)
    subset_recovery = temperature * (
        torch.logsumexp(-selected_violations / temperature, dim=-1)
        - math.log(float(K))
    )
    subset_failures = (~subset_success).to(dtype=dtype)
    failed_recovery = subset_failures * subset_recovery
    subset_utilities = (
        subset_quality
        + recovery_scales.unsqueeze(1) * failed_recovery
        - lambdas.unsqueeze(1) * subset_failures
    )
    if not bool(torch.isfinite(subset_utilities).all().item()):
        raise ValueError(
            "Canonical CLASP subset utility became non-finite; check recovery_temperature."
        )

    total = canonical_finite_batch_advantages(
        subset_utilities,
        subsets,
        batch_size=B,
    )
    quality = canonical_finite_batch_advantages(
        subset_quality,
        subsets,
        batch_size=B,
    )
    recovery = canonical_finite_batch_advantages(
        failed_recovery,
        subsets,
        batch_size=B,
    )
    failure = canonical_finite_batch_advantages(
        subset_failures,
        subsets,
        batch_size=B,
    )

    expected_advantages = (
        quality.advantages
        + recovery_scales.unsqueeze(1) * recovery.advantages
        - lambdas.unsqueeze(1) * failure.advantages
    )
    tolerance = 1e-12 if dtype == torch.float64 else 2e-5
    if not torch.allclose(total.advantages, expected_advantages, rtol=tolerance, atol=tolerance):
        raise RuntimeError("Canonical CLASP component decomposition failed.")

    failure_estimates = deployment_failure_u_statistic(
        raw_feasibility,
        set_size=K,
    )
    subset_failure_means = subset_failures.mean(dim=-1)
    if not torch.allclose(
        failure_estimates,
        subset_failure_means,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise RuntimeError("Canonical CLASP failure U-statistic accounting failed.")

    def restore(value: Tensor) -> Tensor:
        return value.reshape(-1) if objective_was_flat else value

    return CanonicalClaspSetResult(
        advantages=restore(total.advantages),
        mean_reduced_policy_coefficients=restore(
            total.mean_reduced_policy_coefficients
        ),
        quality_advantages=restore(quality.advantages),
        recovery_advantages=restore(recovery.advantages),
        failure_advantages=restore(failure.advantages),
        subset_indices=subsets,
        subset_utilities=subset_utilities,
        subset_quality_utilities=subset_quality,
        subset_recovery_utilities=failed_recovery,
        subset_failures=subset_failures,
        deployment_failure_estimates=failure_estimates,
        centering_residuals=total.centering_residuals,
    )
