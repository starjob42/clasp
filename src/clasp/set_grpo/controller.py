"""Projected signed per-bucket dual updates for set-level failure constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SignedBucketDualUpdate:
    """Result of one globally aggregated projected signed dual step."""

    lambdas: Tensor
    observed_failure_rates: Tensor
    group_counts: Tensor
    deltas: Tensor
    active_buckets: Tensor
    stats: dict[str, float]


def _bucket_parameter(
    value: Tensor | float,
    *,
    num_buckets: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    parameter = torch.as_tensor(value, dtype=dtype, device=device)
    if parameter.numel() == 1:
        return parameter.reshape(1).expand(num_buckets)
    if parameter.dim() == 1 and parameter.numel() == num_buckets:
        return parameter
    raise ValueError(f"{name} must be scalar or a length-{num_buckets} tensor.")


def update_signed_bucket_lambdas_from_counts(
    current_lambdas: Tensor,
    failure_sums: Tensor,
    group_counts: Tensor,
    *,
    target_failure_rates: Tensor | float,
    learning_rates: Tensor | float,
    lambda_max: float,
) -> SignedBucketDualUpdate:
    """Update bucket multipliers from globally reduced sufficient statistics.

    The update for every bucket with at least one observation is

    ``lambda <- clip(lambda + eta * (failure_rate - target), 0, lambda_max)``.

    Buckets absent from the aggregated batch are left unchanged. The caller is
    responsible for reducing ``failure_sums`` and ``group_counts`` across
    ranks and for calling this function exactly once per optimizer step.
    """
    if not isinstance(current_lambdas, Tensor) or current_lambdas.dim() != 1:
        raise ValueError("current_lambdas must be a one-dimensional tensor.")
    if current_lambdas.numel() == 0:
        raise ValueError("current_lambdas must not be empty.")
    if not current_lambdas.is_floating_point():
        raise ValueError("current_lambdas must have a floating-point dtype.")
    if not isinstance(failure_sums, Tensor) or not isinstance(group_counts, Tensor):
        raise TypeError("failure_sums and group_counts must be torch tensors.")
    if failure_sums.shape != current_lambdas.shape or group_counts.shape != current_lambdas.shape:
        raise ValueError("failure_sums and group_counts must match current_lambdas shape.")
    if (
        failure_sums.device != current_lambdas.device
        or group_counts.device != current_lambdas.device
    ):
        raise ValueError("All controller tensors must be on the same device.")
    if not math.isfinite(float(lambda_max)) or float(lambda_max) <= 0.0:
        raise ValueError("lambda_max must be finite and greater than zero.")

    dtype = torch.float64 if current_lambdas.dtype == torch.float64 else torch.float32
    device = current_lambdas.device
    lambdas = current_lambdas.to(dtype=dtype)
    failures = failure_sums.to(dtype=dtype)
    counts = group_counts.to(dtype=dtype)
    for name, value in (
        ("current_lambdas", lambdas),
        ("failure_sums", failures),
        ("group_counts", counts),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains non-finite values.")
    if bool((lambdas < 0.0).any().item()):
        raise ValueError("current_lambdas must be non-negative.")
    if bool((counts < 0.0).any().item()):
        raise ValueError("group_counts must be non-negative.")
    if bool((failures < 0.0).any().item()) or bool((failures > counts).any().item()):
        raise ValueError("failure_sums must lie between zero and group_counts.")

    num_buckets = int(lambdas.numel())
    targets = _bucket_parameter(
        target_failure_rates,
        num_buckets=num_buckets,
        device=device,
        dtype=dtype,
        name="target_failure_rates",
    )
    rates = _bucket_parameter(
        learning_rates,
        num_buckets=num_buckets,
        device=device,
        dtype=dtype,
        name="learning_rates",
    )
    if not bool(torch.isfinite(targets).all().item()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any().item()
    ):
        raise ValueError("target_failure_rates must be finite and lie in [0, 1].")
    if not bool(torch.isfinite(rates).all().item()) or bool((rates < 0.0).any().item()):
        raise ValueError("learning_rates must be finite and non-negative.")

    active = counts > 0.0
    observed = torch.zeros_like(lambdas)
    observed[active] = failures[active] / counts[active]
    requested_delta = rates * (observed - targets)
    requested_delta = torch.where(active, requested_delta, torch.zeros_like(requested_delta))
    updated = (lambdas + requested_delta).clamp(min=0.0, max=float(lambda_max))
    updated = torch.where(active, updated, lambdas)
    deltas = updated - lambdas

    active_count = int(active.sum().item())
    total_count = float(counts.sum().item())
    total_failures = float(failures.sum().item())
    stats = {
        "frontier_cdf/dual_active_bucket_count": float(active_count),
        "frontier_cdf/dual_observed_failure_rate": (
            total_failures / total_count if total_count > 0.0 else 0.0
        ),
        "frontier_cdf/dual_observed_active_mean": float(
            observed[active].mean().item() if active_count else 0.0
        ),
        "frontier_cdf/dual_target_active_mean": float(
            targets[active].mean().item() if active_count else 0.0
        ),
        "frontier_cdf/dual_lambda_mean": float(updated.mean().item()),
        "frontier_cdf/dual_lambda_max": float(updated.max().item()),
        "frontier_cdf/dual_delta_active_mean": float(
            deltas[active].mean().item() if active_count else 0.0
        ),
        "frontier_cdf/dual_increase_bucket_rate": float(
            (deltas[active] > 0.0).to(dtype).mean().item() if active_count else 0.0
        ),
        "frontier_cdf/dual_decrease_bucket_rate": float(
            (deltas[active] < 0.0).to(dtype).mean().item() if active_count else 0.0
        ),
        "frontier_cdf/dual_at_zero_rate": float((updated == 0.0).to(dtype).mean().item()),
        "frontier_cdf/dual_at_max_rate": float(
            (updated == float(lambda_max)).to(dtype).mean().item()
        ),
    }
    return SignedBucketDualUpdate(
        lambdas=updated.to(dtype=current_lambdas.dtype),
        observed_failure_rates=observed.to(dtype=current_lambdas.dtype),
        group_counts=counts.to(dtype=current_lambdas.dtype),
        deltas=deltas.to(dtype=current_lambdas.dtype),
        active_buckets=active,
        stats=stats,
    )


def update_signed_bucket_lambdas(
    current_lambdas: Tensor,
    group_failure_flags: Tensor,
    group_bucket_indices: Tensor,
    *,
    target_failure_rates: Tensor | float,
    learning_rates: Tensor | float,
    lambda_max: float,
) -> SignedBucketDualUpdate:
    """Aggregate per-group failures and apply one projected signed dual step."""
    if not isinstance(group_failure_flags, Tensor) or group_failure_flags.dim() != 1:
        raise ValueError("group_failure_flags must be a one-dimensional tensor.")
    if not isinstance(group_bucket_indices, Tensor) or group_bucket_indices.dim() != 1:
        raise ValueError("group_bucket_indices must be a one-dimensional tensor.")
    if group_failure_flags.numel() != group_bucket_indices.numel():
        raise ValueError("group_failure_flags and group_bucket_indices must have equal length.")
    if group_failure_flags.device != current_lambdas.device:
        raise ValueError("group_failure_flags must be on the same device as current_lambdas.")
    if group_bucket_indices.device != current_lambdas.device:
        raise ValueError("group_bucket_indices must be on the same device as current_lambdas.")

    dtype = torch.float64 if current_lambdas.dtype == torch.float64 else torch.float32
    failures = group_failure_flags.to(dtype=dtype)
    if not bool(torch.isfinite(failures).all().item()) or bool(
        ((failures < 0.0) | (failures > 1.0)).any().item()
    ):
        raise ValueError("group_failure_flags must be finite and lie in [0, 1].")

    raw_indices = group_bucket_indices
    indices = raw_indices.to(dtype=torch.long)
    if not bool((raw_indices == indices.to(dtype=raw_indices.dtype)).all().item()):
        raise ValueError("group_bucket_indices must contain integer values.")
    num_buckets = int(current_lambdas.numel())
    if indices.numel() and bool(((indices < 0) | (indices >= num_buckets)).any().item()):
        raise ValueError("group_bucket_indices contains an out-of-range bucket.")

    failure_sums = torch.zeros(num_buckets, dtype=dtype, device=current_lambdas.device)
    counts = torch.zeros_like(failure_sums)
    if indices.numel():
        failure_sums.scatter_add_(0, indices, failures)
        counts.scatter_add_(0, indices, torch.ones_like(failures))
    return update_signed_bucket_lambdas_from_counts(
        current_lambdas,
        failure_sums,
        counts,
        target_failure_rates=target_failure_rates,
        learning_rates=learning_rates,
        lambda_max=lambda_max,
    )
