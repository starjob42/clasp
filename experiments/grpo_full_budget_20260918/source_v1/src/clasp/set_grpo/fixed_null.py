"""Fixed-null finite-batch advantages for symmetric set utilities.

For one prompt, let ``B > K`` conditionally i.i.d. on-policy responses be
available and let ``S`` range over every size-``K`` subset of those responses.
For every member ``i`` of ``S``, the caller supplies both the deployed utility
``U(S)`` and the utility ``U(S_{i -> null})`` after replacing that member by
one fixed null candidate.  This module computes

``A_i = average_{S: i in S} [U(S) - U(S_{i -> null})]``.

If the null candidate and evaluator configuration are fixed independently of
the current responses, the responses are conditionally i.i.d. and on-policy,
and ``U`` is a symmetric integrable size-``K`` utility with no direct policy
parameter dependence, then

``(K / B) * sum_i A_i * grad(log pi(response_i))``

is an unbiased score-function estimator of the gradient of expected deployed
utility.  Therefore a sequence-level loss whose outer reduction is a mean
over ``B`` responses must use ``K * A_i`` as its per-response coefficient.

Unlike a leave-out contrast, fixed-null advantages do not generally sum to
zero on a realized batch.  This implementation intentionally performs no
sample-dependent centering: such centering would define a different estimator.
There is also no universal variance ordering between this estimator and a
centered leave-out estimator (or other policy-gradient baselines).

The functions here contain no trainer or evaluator state.  In particular,
they cannot verify that all replacement utilities really use the same fixed
null semantics; that is part of the caller's statistical contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FixedNullAdvantageResult:
    """Outputs of :func:`fixed_null_finite_batch_advantages`.

    ``advantages`` and ``mean_reduced_policy_coefficients`` have the leading
    shape of ``subset_utilities`` followed by ``batch_size``.
    ``subset_member_differences`` has shape ``[..., comb(B, K), K]`` and stores
    the formula's individual ``U(S) - U(S_{i -> null})`` terms.

    ``realized_advantage_sums`` and ``realized_advantage_means`` have only the
    leading shape.  They expose the estimator's realized common mode and are
    not residuals expected to vanish.  No centering is applied.
    """

    advantages: Tensor
    mean_reduced_policy_coefficients: Tensor
    subset_member_differences: Tensor
    realized_advantage_sums: Tensor
    realized_advantage_means: Tensor


def _strict_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return int(value)


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
        raise ValueError(
            "subset_indices, subset_utilities, and replacement_utilities "
            "must be on the same device."
        )

    num_subsets, set_size = subset_indices.shape
    if set_size <= 0:
        raise ValueError("subset_indices must contain at least one index per subset.")
    if set_size >= batch_size:
        raise ValueError("set_size must be strictly less than batch_size.")

    expected_subsets = math.comb(batch_size, set_size)
    if num_subsets != expected_subsets:
        raise ValueError(
            "subset_indices must contain every size-"
            f"{set_size} subset exactly once: expected {expected_subsets} rows, "
            f"got {num_subsets}."
        )
    if bool(((subset_indices < 0) | (subset_indices >= batch_size)).any().item()):
        raise ValueError("subset_indices contains an index outside [0, batch_size).")
    if set_size > 1 and bool(
        (subset_indices[:, 1:] <= subset_indices[:, :-1]).any().item()
    ):
        raise ValueError("Every row of subset_indices must be strictly increasing.")
    if int(torch.unique(subset_indices, dim=0).size(0)) != expected_subsets:
        raise ValueError("subset_indices contains duplicate rows or omits a required subset.")

    return int(num_subsets), int(set_size)


def fixed_null_finite_batch_advantages(
    subset_utilities: Tensor,
    subset_indices: Tensor,
    replacement_utilities: Tensor,
    *,
    batch_size: int,
) -> FixedNullAdvantageResult:
    """Compute formula-matched fixed-null advantages from complete tables.

    Args:
        subset_utilities: Floating-point tensor with shape
            ``[..., comb(batch_size, K)]`` containing ``U(S)`` for each row of
            ``subset_indices``.
        subset_indices: Complete, duplicate-free collection of increasing
            size-``K`` index rows. Row order is arbitrary and defines the
            subset axis in both utility inputs.
        replacement_utilities: Floating-point tensor with shape
            ``[..., comb(batch_size, K), K]``. Entry ``[..., m, j]`` is
            ``U(S_{i -> null})`` for subset row ``m`` and candidate
            ``i = subset_indices[m, j]``. Every entry must use the same fixed
            null candidate and evaluator semantics for the theorem to apply.
        batch_size: Rollout count ``B``. The inferred subset size must satisfy
            ``1 <= K < B``.

    Returns:
        The uncentered fixed-null advantages, ``K * A`` coefficients for an
        outer mean-reduced sequence-level loss, the individual subset-member
        differences, and realized sum/mean diagnostics.

    This function verifies the finite-table algebra (shape, completeness,
    dtype, device, and finiteness), but it cannot enforce the theorem's
    on-policy independence, symmetry, or fixed-null assumptions.
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

    if not isinstance(replacement_utilities, Tensor):
        raise TypeError("replacement_utilities must be a torch.Tensor.")
    if replacement_utilities.dim() < 2:
        raise ValueError(
            "replacement_utilities must have shape [..., num_subsets, set_size]."
        )
    if not replacement_utilities.is_floating_point():
        raise TypeError("replacement_utilities must have a floating-point dtype.")
    if replacement_utilities.dtype != subset_utilities.dtype:
        raise TypeError(
            "replacement_utilities and subset_utilities must have the same dtype."
        )
    if replacement_utilities.device != subset_utilities.device:
        raise ValueError(
            "replacement_utilities and subset_utilities must be on the same device."
        )
    if not bool(torch.isfinite(replacement_utilities).all().item()):
        raise ValueError("replacement_utilities contains non-finite values.")

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
    expected_replacement_shape = (*subset_utilities.shape, K)
    if tuple(replacement_utilities.shape) != expected_replacement_shape:
        raise ValueError(
            "replacement_utilities must have shape "
            f"{expected_replacement_shape}, got {tuple(replacement_utilities.shape)}."
        )

    differences = subset_utilities.unsqueeze(-1) - replacement_utilities
    leading_shape = subset_utilities.shape[:-1]
    flat_differences = differences.reshape(-1, num_subsets * K)
    candidate_indices = subset_indices.reshape(1, num_subsets * K).expand(
        flat_differences.size(0), -1
    )
    summed_differences = torch.zeros(
        (flat_differences.size(0), B),
        dtype=subset_utilities.dtype,
        device=subset_utilities.device,
    ).scatter_add(1, candidate_indices, flat_differences)

    subsets_per_candidate = math.comb(B - 1, K - 1)
    advantages = (summed_differences / float(subsets_per_candidate)).reshape(
        *leading_shape, B
    )
    mean_coefficients = float(K) * advantages
    realized_sums = advantages.sum(dim=-1)
    realized_means = advantages.mean(dim=-1)

    return FixedNullAdvantageResult(
        advantages=advantages,
        mean_reduced_policy_coefficients=mean_coefficients,
        subset_member_differences=differences,
        realized_advantage_sums=realized_sums,
        realized_advantage_means=realized_means,
    )


__all__ = [
    "FixedNullAdvantageResult",
    "fixed_null_finite_batch_advantages",
]
