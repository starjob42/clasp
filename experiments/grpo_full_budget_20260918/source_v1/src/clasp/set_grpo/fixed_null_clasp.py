"""Fixed-null credit for the exact canonical CLASP set utility.

This adapter deliberately delegates every utility evaluation to
``compute_canonical_clasp_set_advantages`` and delegates the finite-batch
transform to :func:`fixed_null_finite_batch_advantages`.  It therefore chooses
no null semantics and contains no alternate approximation of the set utility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .canonical import compute_canonical_clasp_set_advantages
from .fixed_null import fixed_null_finite_batch_advantages


@dataclass(frozen=True)
class FixedNullClaspSetResult:
    advantages: Tensor
    mean_reduced_policy_coefficients: Tensor
    quality_advantages: Tensor
    recovery_advantages: Tensor
    failure_advantages: Tensor
    subset_indices: Tensor
    subset_utilities: Tensor
    replacement_utilities: Tensor
    subset_quality_utilities: Tensor
    subset_recovery_utilities: Tensor
    subset_failures: Tensor
    deployment_failure_estimates: Tensor
    realized_advantage_sums: Tensor
    realized_advantage_means: Tensor


def _candidate_matrix(value: Tensor, *, batch_size: int, name: str) -> tuple[Tensor, bool]:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dim() == 1:
        if value.numel() == 0 or value.numel() % batch_size != 0:
            raise ValueError(f"{name} must contain a positive multiple of B={batch_size} values.")
        return value.reshape(-1, batch_size), True
    if value.dim() == 2 and value.size(0) > 0 and value.size(1) == batch_size:
        return value, False
    raise ValueError(f"{name} must be flat or have shape [num_groups, {batch_size}].")


def compute_fixed_null_clasp_set_advantages(
    objective_scores: Tensor,
    violation_scores: Tensor,
    feasibility_flags: Tensor,
    null_objective_scores: Tensor,
    null_violation_scores: Tensor,
    null_feasibility_flags: Tensor,
    *,
    rollout_batch_size: int,
    deployment_set_size: int,
    group_lambdas: Tensor | float,
    recovery_scale: Tensor | float = 1.0,
    recovery_temperature: float = 0.15,
    target_failure_rates: Tensor | float = 0.0,
) -> FixedNullClaspSetResult:
    """Evaluate ``U(S)-U(S_{i->null})`` using one receipted null per prompt.

    The three null-score tensors follow the candidate input shape only so the
    distributed trainer can gather them without a side channel.  Within each
    rollout group all ``B`` rows must be bitwise identical; otherwise the input
    does not describe one fixed null candidate for that prompt.
    """

    B = int(rollout_batch_size)
    K = int(deployment_set_size)
    actual_values = (objective_scores, violation_scores, feasibility_flags)
    null_values = (
        null_objective_scores,
        null_violation_scores,
        null_feasibility_flags,
    )
    actual_matrices: list[Tensor] = []
    null_matrices: list[Tensor] = []
    was_flat: bool | None = None
    for name, actual, null in zip(
        ("objective_scores", "violation_scores", "feasibility_flags"),
        actual_values,
        null_values,
    ):
        actual_matrix, actual_flat = _candidate_matrix(actual, batch_size=B, name=name)
        null_matrix, null_flat = _candidate_matrix(
            null,
            batch_size=B,
            name=f"null_{name}",
        )
        if actual_flat != null_flat or actual_matrix.shape != null_matrix.shape:
            raise ValueError(f"{name} and null_{name} must use the same shape convention.")
        if actual_matrix.device != null_matrix.device:
            raise ValueError(f"{name} and null_{name} must be on the same device.")
        if was_flat is not None and actual_flat != was_flat:
            raise ValueError("All fixed-null candidate inputs must use one shape convention.")
        was_flat = actual_flat
        actual_matrices.append(actual_matrix)
        null_matrices.append(null_matrix)

    for name, matrix in zip(
        ("null_objective_scores", "null_violation_scores", "null_feasibility_flags"),
        null_matrices,
    ):
        if not bool((matrix == matrix[:, :1]).all().item()):
            raise ValueError(
                f"{name} must be identical across every rollout row in a prompt group."
            )

    base = compute_canonical_clasp_set_advantages(
        *actual_matrices,
        rollout_batch_size=B,
        deployment_set_size=K,
        group_lambdas=group_lambdas,
        recovery_scale=recovery_scale,
        recovery_temperature=recovery_temperature,
        target_failure_rates=target_failure_rates,
    )
    subsets = base.subset_indices
    num_groups = int(actual_matrices[0].size(0))
    num_subsets = int(subsets.size(0))
    counterfactual_count = num_groups * num_subsets * K

    counterfactual_inputs = [
        matrix[:, None, None, :]
        .expand(num_groups, num_subsets, K, B)
        .clone()
        .reshape(counterfactual_count, B)
        for matrix in actual_matrices
    ]
    replaced_indices = (
        subsets.reshape(1, num_subsets, K)
        .expand(num_groups, num_subsets, K)
        .reshape(-1)
    )
    row_indices = torch.arange(counterfactual_count, device=subsets.device)
    for counterfactual, null_matrix in zip(counterfactual_inputs, null_matrices):
        null_by_group = (
            null_matrix[:, 0]
            .reshape(num_groups, 1, 1)
            .expand(num_groups, num_subsets, K)
            .reshape(-1)
            .to(dtype=counterfactual.dtype)
        )
        counterfactual[row_indices, replaced_indices] = null_by_group

    def repeat_group_parameter(value: Tensor | float) -> Tensor | float:
        tensor = torch.as_tensor(
            value,
            device=actual_matrices[0].device,
            dtype=base.subset_utilities.dtype,
        )
        if tensor.numel() == 1:
            return tensor
        if tensor.dim() != 1 or tensor.numel() != num_groups:
            raise ValueError(
                f"fixed-null group parameter must be scalar or length {num_groups}."
            )
        return (
            tensor.reshape(num_groups, 1, 1)
            .expand(num_groups, num_subsets, K)
            .reshape(-1)
        )

    counterfactual = compute_canonical_clasp_set_advantages(
        *counterfactual_inputs,
        rollout_batch_size=B,
        deployment_set_size=K,
        group_lambdas=repeat_group_parameter(group_lambdas),
        recovery_scale=repeat_group_parameter(recovery_scale),
        recovery_temperature=recovery_temperature,
        target_failure_rates=repeat_group_parameter(target_failure_rates),
    )
    selected_subset_rows = (
        torch.arange(num_subsets, device=subsets.device)
        .reshape(1, num_subsets, 1)
        .expand(num_groups, num_subsets, K)
        .reshape(-1)
    )

    def replacement_table(table: Tensor) -> Tensor:
        selected = table[row_indices, selected_subset_rows]
        return selected.reshape(num_groups, num_subsets, K)

    replacement_total = replacement_table(counterfactual.subset_utilities)
    replacement_quality = replacement_table(counterfactual.subset_quality_utilities)
    replacement_recovery = replacement_table(counterfactual.subset_recovery_utilities)
    replacement_failure = replacement_table(counterfactual.subset_failures)

    total = fixed_null_finite_batch_advantages(
        base.subset_utilities,
        subsets,
        replacement_total,
        batch_size=B,
    )
    quality = fixed_null_finite_batch_advantages(
        base.subset_quality_utilities,
        subsets,
        replacement_quality,
        batch_size=B,
    )
    recovery = fixed_null_finite_batch_advantages(
        base.subset_recovery_utilities,
        subsets,
        replacement_recovery,
        batch_size=B,
    )
    failure = fixed_null_finite_batch_advantages(
        base.subset_failures,
        subsets,
        replacement_failure,
        batch_size=B,
    )

    lambda_values = torch.as_tensor(
        group_lambdas,
        device=base.subset_utilities.device,
        dtype=base.subset_utilities.dtype,
    )
    if lambda_values.numel() == 1:
        lambda_values = lambda_values.reshape(1).expand(num_groups)
    recovery_values = torch.as_tensor(
        recovery_scale,
        device=base.subset_utilities.device,
        dtype=base.subset_utilities.dtype,
    )
    if recovery_values.numel() == 1:
        recovery_values = recovery_values.reshape(1).expand(num_groups)
    expected = (
        quality.advantages
        + recovery_values.reshape(-1, 1) * recovery.advantages
        - lambda_values.reshape(-1, 1) * failure.advantages
    )
    tolerance = 1e-12 if base.subset_utilities.dtype == torch.float64 else 2e-5
    if not torch.allclose(total.advantages, expected, rtol=tolerance, atol=tolerance):
        raise RuntimeError("Fixed-null CLASP component decomposition failed.")

    def restore(value: Tensor) -> Tensor:
        return value.reshape(-1) if bool(was_flat) else value

    return FixedNullClaspSetResult(
        advantages=restore(total.advantages),
        mean_reduced_policy_coefficients=restore(
            total.mean_reduced_policy_coefficients
        ),
        quality_advantages=restore(quality.advantages),
        recovery_advantages=restore(recovery.advantages),
        failure_advantages=restore(failure.advantages),
        subset_indices=subsets,
        subset_utilities=base.subset_utilities,
        replacement_utilities=replacement_total,
        subset_quality_utilities=base.subset_quality_utilities,
        subset_recovery_utilities=base.subset_recovery_utilities,
        subset_failures=base.subset_failures,
        deployment_failure_estimates=base.deployment_failure_estimates,
        realized_advantage_sums=total.realized_advantage_sums,
        realized_advantage_means=total.realized_advantage_means,
    )


__all__ = [
    "FixedNullClaspSetResult",
    "compute_fixed_null_clasp_set_advantages",
]
