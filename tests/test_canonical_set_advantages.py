from __future__ import annotations

import itertools
import math

import pytest
import torch

from clasp.set_grpo import (
    CanonicalClaspSetResult,
    CanonicalSetAdvantageResult,
    canonical_finite_batch_advantages,
    compute_canonical_clasp_set_advantages,
    compute_dense_clasp_auxiliary_advantages,
    deployment_failure_u_statistic,
    enumerate_k_subsets,
)


def test_dense_auxiliary_is_centered_dense_and_lexicographic() -> None:
    objective = torch.tensor(
        [
            [0.9, 0.6, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.4, 0.8, 0.5, 0.7, 0.1],
        ],
        dtype=torch.float64,
    )
    violations = torch.tensor(
        [
            [0.0, 0.0, 5.0, 2.0, 3.0, 4.0],
            [6.0, 1.0, 3.0, 5.0, 2.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    feasibility = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float64,
    )

    result = compute_dense_clasp_auxiliary_advantages(
        objective,
        violations,
        feasibility,
        rollout_batch_size=6,
        group_lambdas=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64),
        quality_scale=0.5,
        feasibility_scale=1.5,
        violation_scale=0.5,
        recovery_scale=1.5,
    )

    torch.testing.assert_close(
        result.centering_residuals,
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert bool(result.mixed_groups[0])
    assert result.advantages[0, :2].min() > result.advantages[0, 2:].max()
    assert result.advantages[0, 0] > result.advantages[0, 1]
    assert result.advantages[0, 3] > result.advantages[0, 4]
    assert bool(result.all_infeasible_groups[1])
    assert result.advantages[1, 1] == result.advantages[1].max()
    assert result.advantages[1, 0] == result.advantages[1].min()
    assert bool(result.all_feasible_groups[2])
    assert result.advantages[2, 2] == result.advantages[2].max()
    assert result.advantages[2, 5] == result.advantages[2].min()


def test_dense_auxiliary_preserves_exact_ties_and_flat_shape() -> None:
    result = compute_dense_clasp_auxiliary_advantages(
        torch.ones(12),
        torch.zeros(12),
        torch.ones(12),
        rollout_batch_size=12,
        group_lambdas=3.0,
        quality_scale=0.5,
        feasibility_scale=1.5,
        violation_scale=0.5,
        recovery_scale=1.5,
    )
    assert result.advantages.shape == (12,)
    torch.testing.assert_close(result.advantages, torch.zeros(12))


def _brute_leave_out(
    subset_utilities: torch.Tensor,
    subset_indices: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    conditional = []
    leave_out = []
    for candidate_idx in range(batch_size):
        contains = (subset_indices == candidate_idx).any(dim=1)
        conditional.append(subset_utilities[..., contains].mean(dim=-1))
        leave_out.append(subset_utilities[..., ~contains].mean(dim=-1))
    return torch.stack(conditional, dim=-1), torch.stack(leave_out, dim=-1)


def _symmetric_subset_utility(
    candidate_values: torch.Tensor,
    subset_indices: torch.Tensor,
) -> torch.Tensor:
    selected = candidate_values[subset_indices]
    return (
        selected.square().sum(dim=-1)
        + 0.3 * selected.prod(dim=-1)
        + 0.2 * selected.max(dim=-1).values
    )


@pytest.mark.parametrize("batch_size,set_size", [(2, 1), (4, 2), (5, 3), (6, 5)])
def test_enumerate_k_subsets_is_complete_and_lexicographic(
    batch_size: int,
    set_size: int,
) -> None:
    actual = enumerate_k_subsets(batch_size, set_size)
    expected = torch.tensor(
        list(itertools.combinations(range(batch_size), set_size)),
        dtype=torch.long,
    )
    assert actual.dtype == torch.long
    assert actual.device.type == "cpu"
    assert actual.shape == (math.comb(batch_size, set_size), set_size)
    assert torch.equal(actual, expected)


def test_generic_transform_matches_brute_force_with_arbitrary_leading_axes() -> None:
    batch_size = 6
    set_size = 3
    subsets = enumerate_k_subsets(batch_size, set_size)
    generator = torch.Generator().manual_seed(7391)
    utilities = torch.randn(
        (2, 4, subsets.size(0)),
        generator=generator,
        dtype=torch.float64,
    )

    result = canonical_finite_batch_advantages(
        utilities,
        subsets,
        batch_size=batch_size,
    )
    expected_u, expected_v = _brute_leave_out(utilities, subsets, batch_size)

    assert isinstance(result, CanonicalSetAdvantageResult)
    assert result.conditional_utilities.shape == (2, 4, batch_size)
    torch.testing.assert_close(result.conditional_utilities, expected_u)
    torch.testing.assert_close(result.leave_out_utilities, expected_v)
    torch.testing.assert_close(result.advantages, expected_u - expected_v)
    torch.testing.assert_close(
        result.mean_reduced_policy_coefficients,
        float(set_size) * result.advantages,
    )
    torch.testing.assert_close(result.subset_mean_utilities, utilities.mean(dim=-1))
    torch.testing.assert_close(
        result.centering_residuals,
        torch.zeros_like(result.centering_residuals),
        rtol=0.0,
        atol=2e-15,
    )

    subset_mean_from_i = (
        (float(set_size) / float(batch_size)) * result.conditional_utilities
        + (float(batch_size - set_size) / float(batch_size))
        * result.leave_out_utilities
    )
    torch.testing.assert_close(
        subset_mean_from_i,
        result.subset_mean_utilities.unsqueeze(-1).expand_as(subset_mean_from_i),
    )


def test_subset_row_order_is_arbitrary_when_utilities_are_permuted_with_it() -> None:
    batch_size = 5
    set_size = 2
    subsets = enumerate_k_subsets(batch_size, set_size)
    utilities = torch.linspace(-1.5, 2.0, subsets.size(0), dtype=torch.float64)
    permutation = torch.tensor([7, 1, 9, 0, 4, 6, 2, 8, 3, 5])

    original = canonical_finite_batch_advantages(
        utilities,
        subsets,
        batch_size=batch_size,
    )
    reordered = canonical_finite_batch_advantages(
        utilities[permutation],
        subsets[permutation],
        batch_size=batch_size,
    )
    torch.testing.assert_close(reordered.conditional_utilities, original.conditional_utilities)
    torch.testing.assert_close(reordered.leave_out_utilities, original.leave_out_utilities)
    torch.testing.assert_close(reordered.advantages, original.advantages)


def test_shift_invariance_linearity_and_dtype_device_preservation() -> None:
    batch_size = 5
    set_size = 3
    subsets = enumerate_k_subsets(batch_size, set_size)
    first = torch.linspace(-0.8, 1.3, subsets.size(0), dtype=torch.float64)
    second = torch.cos(torch.arange(subsets.size(0), dtype=torch.float64))

    first_result = canonical_finite_batch_advantages(
        first,
        subsets,
        batch_size=batch_size,
    )
    second_result = canonical_finite_batch_advantages(
        second,
        subsets,
        batch_size=batch_size,
    )
    sum_result = canonical_finite_batch_advantages(
        first + second,
        subsets,
        batch_size=batch_size,
    )
    shifted = canonical_finite_batch_advantages(
        first + 17.25,
        subsets,
        batch_size=batch_size,
    )

    torch.testing.assert_close(
        sum_result.advantages,
        first_result.advantages + second_result.advantages,
    )
    torch.testing.assert_close(shifted.advantages, first_result.advantages)
    torch.testing.assert_close(
        shifted.conditional_utilities,
        first_result.conditional_utilities + 17.25,
    )
    torch.testing.assert_close(
        shifted.leave_out_utilities,
        first_result.leave_out_utilities + 17.25,
    )
    for value in (
        first_result.conditional_utilities,
        first_result.leave_out_utilities,
        first_result.advantages,
        first_result.mean_reduced_policy_coefficients,
        first_result.subset_mean_utilities,
        first_result.centering_residuals,
    ):
        assert value.dtype == first.dtype
        assert value.device == first.device


def test_candidate_permutation_equivariance_for_symmetric_utility() -> None:
    batch_size = 5
    set_size = 3
    subsets = enumerate_k_subsets(batch_size, set_size)
    values = torch.tensor([0.2, 1.3, -0.7, 2.1, 0.8], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    original = canonical_finite_batch_advantages(
        _symmetric_subset_utility(values, subsets),
        subsets,
        batch_size=batch_size,
    )
    permuted = canonical_finite_batch_advantages(
        _symmetric_subset_utility(values[permutation], subsets),
        subsets,
        batch_size=batch_size,
    )

    torch.testing.assert_close(permuted.advantages, original.advantages[permutation])
    torch.testing.assert_close(
        permuted.mean_reduced_policy_coefficients,
        original.mean_reduced_policy_coefficients[permutation],
    )


def test_leave_out_utility_excludes_the_candidate_exactly() -> None:
    batch_size = 5
    set_size = 3
    candidate_idx = 2
    subsets = enumerate_k_subsets(batch_size, set_size)
    values = torch.tensor([0.1, -0.4, 0.7, 1.2, 2.0], dtype=torch.float64)
    changed = values.clone()
    changed[candidate_idx] = 20.0

    original = canonical_finite_batch_advantages(
        _symmetric_subset_utility(values, subsets),
        subsets,
        batch_size=batch_size,
    )
    modified = canonical_finite_batch_advantages(
        _symmetric_subset_utility(changed, subsets),
        subsets,
        batch_size=batch_size,
    )

    torch.testing.assert_close(
        modified.leave_out_utilities[candidate_idx],
        original.leave_out_utilities[candidate_idx],
        rtol=0.0,
        atol=0.0,
    )
    assert modified.conditional_utilities[candidate_idx] != original.conditional_utilities[
        candidate_idx
    ]


def test_constant_subset_utility_has_exactly_zero_advantage() -> None:
    batch_size = 6
    set_size = 4
    subsets = enumerate_k_subsets(batch_size, set_size)
    utilities = torch.full((3, subsets.size(0)), 2.5, dtype=torch.float64)
    result = canonical_finite_batch_advantages(
        utilities,
        subsets,
        batch_size=batch_size,
    )
    torch.testing.assert_close(
        result.advantages,
        torch.zeros_like(result.advantages),
        rtol=0.0,
        atol=0.0,
    )


def test_exhaustive_policy_gradient_unbiasedness_and_mean_loss_scaling() -> None:
    """The canonical estimator matches a nonlinear symmetric binary set utility."""

    batch_size = 3
    set_size = 2
    subsets = enumerate_k_subsets(batch_size, set_size)
    utility_00 = torch.tensor(-0.7, dtype=torch.float64)
    utility_01 = torch.tensor(1.2, dtype=torch.float64)
    utility_11 = torch.tensor(0.3, dtype=torch.float64)

    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    probability = torch.sigmoid(theta)
    expected_utility = (
        (1.0 - probability).square() * utility_00
        + 2.0 * probability * (1.0 - probability) * utility_01
        + probability.square() * utility_11
    )
    analytic_gradient = torch.autograd.grad(expected_utility, theta)[0]

    detached_probability = probability.detach()
    expected_canonical_gradient = torch.zeros((), dtype=torch.float64)
    expected_mean_coefficient_gradient = torch.zeros((), dtype=torch.float64)
    expected_unscaled_mean_gradient = torch.zeros((), dtype=torch.float64)
    for action_tuple in itertools.product((0, 1), repeat=batch_size):
        actions = torch.tensor(action_tuple, dtype=torch.float64)
        selected_counts = actions[subsets].sum(dim=-1)
        subset_utilities = torch.where(
            selected_counts == 0.0,
            utility_00,
            torch.where(selected_counts == 1.0, utility_01, utility_11),
        )
        result = canonical_finite_batch_advantages(
            subset_utilities,
            subsets,
            batch_size=batch_size,
        )
        success_count = int(sum(action_tuple))
        batch_probability = (
            detached_probability**success_count
            * (1.0 - detached_probability) ** (batch_size - success_count)
        )
        scores = actions - detached_probability
        canonical_gradient = (
            float(set_size) / float(batch_size)
        ) * (result.advantages * scores).sum()
        mean_coefficient_gradient = (
            result.mean_reduced_policy_coefficients * scores
        ).mean()
        unscaled_mean_gradient = (result.advantages * scores).mean()
        torch.testing.assert_close(canonical_gradient, mean_coefficient_gradient)

        expected_canonical_gradient += batch_probability * canonical_gradient
        expected_mean_coefficient_gradient += batch_probability * mean_coefficient_gradient
        expected_unscaled_mean_gradient += batch_probability * unscaled_mean_gradient

    torch.testing.assert_close(
        expected_canonical_gradient,
        analytic_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        expected_mean_coefficient_gradient,
        analytic_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        expected_unscaled_mean_gradient,
        analytic_gradient / float(set_size),
        rtol=1e-12,
        atol=1e-12,
    )


def test_all_infeasible_softmin_recovery_is_centered_signed_and_monotone() -> None:
    batch_size = 5
    set_size = 3
    temperature = 0.2
    subsets = enumerate_k_subsets(batch_size, set_size)
    violations = torch.tensor([0.01, 0.1, 0.3, 0.7, 0.95], dtype=torch.float64)
    selected = violations[subsets]
    recovery_utilities = temperature * (
        torch.logsumexp(-selected / temperature, dim=-1) - math.log(float(set_size))
    )
    result = canonical_finite_batch_advantages(
        recovery_utilities,
        subsets,
        batch_size=batch_size,
    )

    assert bool((result.advantages[:-1] > result.advantages[1:]).all().item())
    assert result.advantages[0] > 0.0
    assert result.advantages[-1] < 0.0
    torch.testing.assert_close(
        result.advantages.sum(),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )

    tied_recovery = torch.zeros(subsets.size(0), dtype=torch.float64)
    tied = canonical_finite_batch_advantages(
        tied_recovery,
        subsets,
        batch_size=batch_size,
    )
    torch.testing.assert_close(tied.advantages, torch.zeros_like(tied.advantages))


def test_binary_failure_component_is_signed_in_mixed_batch_and_zero_when_all_fail() -> None:
    batch_size = 5
    set_size = 2
    multiplier = 1.7
    subsets = enumerate_k_subsets(batch_size, set_size)
    feasible = torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool)
    subset_failed = ~feasible[subsets].any(dim=-1)
    failure_utilities = -multiplier * subset_failed.to(torch.float64)
    mixed = canonical_finite_batch_advantages(
        failure_utilities,
        subsets,
        batch_size=batch_size,
    )

    assert bool((mixed.advantages[feasible] > 0.0).all().item())
    assert bool((mixed.advantages[~feasible] < 0.0).all().item())
    torch.testing.assert_close(
        mixed.advantages.sum(),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )

    all_failed_utilities = torch.full_like(failure_utilities, -multiplier)
    all_failed = canonical_finite_batch_advantages(
        all_failed_utilities,
        subsets,
        batch_size=batch_size,
    )
    torch.testing.assert_close(
        all_failed.advantages,
        torch.zeros_like(all_failed.advantages),
        rtol=0.0,
        atol=0.0,
    )


def test_deployment_failure_u_statistic_matches_exact_combinatorics() -> None:
    flags = torch.tensor(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.float64,
    )
    actual = deployment_failure_u_statistic(flags, set_size=2)
    expected = torch.tensor([0.5, 1.0, 1.0 / 6.0, 0.0], dtype=torch.float64)
    assert actual.dtype == flags.dtype
    assert actual.device == flags.device
    torch.testing.assert_close(actual, expected, rtol=1e-15, atol=1e-15)

    boolean = deployment_failure_u_statistic(flags.bool(), set_size=2)
    assert boolean.dtype == torch.float32
    torch.testing.assert_close(boolean, expected.to(torch.float32))


def test_deployment_failure_u_statistic_is_unbiased_for_failure_at_k() -> None:
    batch_size = 4
    set_size = 2
    failure_probability = 0.37
    expected_statistic = 0.0
    for invalid_tuple in itertools.product((0, 1), repeat=batch_size):
        invalid = torch.tensor(invalid_tuple, dtype=torch.bool)
        invalid_count = int(sum(invalid_tuple))
        batch_probability = (
            failure_probability**invalid_count
            * (1.0 - failure_probability) ** (batch_size - invalid_count)
        )
        feasible = ~invalid
        statistic = deployment_failure_u_statistic(feasible, set_size=set_size)
        expected_statistic += batch_probability * float(statistic.item())
    assert expected_statistic == pytest.approx(failure_probability**set_size, abs=1e-8)


def test_subset_enumeration_rejects_invalid_cardinalities() -> None:
    with pytest.raises(TypeError, match="batch_size"):
        enumerate_k_subsets(True, 1)
    with pytest.raises(ValueError, match="batch_size"):
        enumerate_k_subsets(0, 1)
    with pytest.raises(ValueError, match="set_size"):
        enumerate_k_subsets(4, 0)
    with pytest.raises(ValueError, match="strictly less"):
        enumerate_k_subsets(4, 4)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("incomplete", "every size"),
        ("duplicate", "duplicate"),
        ("unsorted", "strictly increasing"),
        ("out_of_range", "outside"),
        ("wrong_index_dtype", "torch.long"),
        ("wrong_utility_length", "final subset_utilities"),
        ("integer_utilities", "floating-point"),
        ("nonfinite_utilities", "non-finite"),
    ],
)
def test_canonical_transform_rejects_malformed_inputs(mutation: str, match: str) -> None:
    batch_size = 4
    subsets = enumerate_k_subsets(batch_size, 2)
    utilities = torch.arange(subsets.size(0), dtype=torch.float64)

    if mutation == "incomplete":
        subsets = subsets[:-1]
        utilities = utilities[:-1]
    elif mutation == "duplicate":
        subsets = subsets.clone()
        subsets[-1] = subsets[0]
    elif mutation == "unsorted":
        subsets = subsets.clone()
        subsets[0] = subsets[0].flip(0)
    elif mutation == "out_of_range":
        subsets = subsets.clone()
        subsets[-1, -1] = batch_size
    elif mutation == "wrong_index_dtype":
        subsets = subsets.to(torch.int32)
    elif mutation == "wrong_utility_length":
        utilities = utilities[:-1]
    elif mutation == "integer_utilities":
        utilities = utilities.to(torch.long)
    elif mutation == "nonfinite_utilities":
        utilities = utilities.clone()
        utilities[0] = float("nan")
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(f"Unknown mutation {mutation}")

    with pytest.raises((TypeError, ValueError), match=match):
        canonical_finite_batch_advantages(
            utilities,
            subsets,
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    "flags,set_size,match",
    [
        (torch.tensor([0.0, 0.5, 1.0]), 2, "exact boolean"),
        (torch.tensor([0.0, float("nan"), 1.0]), 2, "non-finite"),
        (torch.tensor([], dtype=torch.bool), 1, "non-empty"),
        (torch.tensor([0, 1, 1], dtype=torch.bool), 3, "strictly less"),
    ],
)
def test_deployment_failure_u_statistic_rejects_invalid_inputs(
    flags: torch.Tensor,
    set_size: int,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        deployment_failure_u_statistic(flags, set_size=set_size)


def test_canonical_clasp_b9_k8_uses_nine_subsets_and_exact_mean_loss_scale() -> None:
    objective = torch.tensor(
        [[0.9, 0.7, 0.0, 0.4, 0.0, 0.8, 0.2, 0.0, 0.6]],
        dtype=torch.float64,
    )
    feasibility = torch.tensor(
        [[1, 1, 0, 1, 0, 1, 1, 0, 1]],
        dtype=torch.bool,
    )
    violations = (~feasibility).to(torch.float64)
    result = compute_canonical_clasp_set_advantages(
        objective,
        violations,
        feasibility,
        rollout_batch_size=9,
        deployment_set_size=8,
        group_lambdas=2.0,
        recovery_scale=0.5,
        target_failure_rates=0.01,
    )

    assert isinstance(result, CanonicalClaspSetResult)
    assert result.subset_indices.shape == (9, 8)
    assert result.advantages.shape == (1, 9)
    torch.testing.assert_close(
        result.mean_reduced_policy_coefficients,
        8.0 * result.advantages,
    )
    torch.testing.assert_close(
        result.advantages.sum(dim=1),
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=1e-14,
    )


def test_canonical_clasp_uses_best_feasible_quality_and_ignores_infeasible_quality() -> None:
    feasibility = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    violations = (~feasibility).to(torch.float64)
    base_objective = torch.tensor([[0.4, 0.0, 0.9, 0.0]], dtype=torch.float64)
    poisoned_objective = base_objective.clone()
    poisoned_objective[0, 1] = 1_000_000.0
    poisoned_objective[0, 3] = 2_000_000.0

    base = compute_canonical_clasp_set_advantages(
        base_objective,
        violations,
        feasibility,
        rollout_batch_size=4,
        deployment_set_size=2,
        group_lambdas=0.0,
        recovery_scale=0.0,
    )
    poisoned = compute_canonical_clasp_set_advantages(
        poisoned_objective,
        violations,
        feasibility,
        rollout_batch_size=4,
        deployment_set_size=2,
        group_lambdas=0.0,
        recovery_scale=0.0,
    )

    torch.testing.assert_close(poisoned.subset_quality_utilities, base.subset_quality_utilities)
    torch.testing.assert_close(poisoned.advantages, base.advantages)
    selected = base_objective[:, base.subset_indices]
    selected_feasible = feasibility[:, base.subset_indices]
    expected_quality = torch.where(
        selected_feasible,
        selected,
        torch.full_like(selected, -torch.inf),
    ).max(dim=-1).values
    expected_quality = torch.where(
        selected_feasible.any(dim=-1),
        expected_quality,
        torch.zeros_like(expected_quality),
    )
    torch.testing.assert_close(base.subset_quality_utilities, expected_quality)


def test_canonical_clasp_all_infeasible_recovery_is_bounded_signed_and_ordered() -> None:
    violations = torch.tensor([[0.01, 0.1, 0.4, 1.0, 4.0]], dtype=torch.float64)
    result = compute_canonical_clasp_set_advantages(
        torch.zeros_like(violations),
        violations,
        torch.zeros_like(violations, dtype=torch.bool),
        rollout_batch_size=5,
        deployment_set_size=4,
        group_lambdas=7.0,
        recovery_scale=1.0,
        recovery_temperature=0.2,
    )

    assert bool((result.subset_recovery_utilities >= -1.0).all().item())
    assert bool((result.subset_recovery_utilities <= 0.0).all().item())
    assert bool((result.advantages[0, :-1] > result.advantages[0, 1:]).all().item())
    assert result.advantages[0, 0] > 0.0
    assert result.advantages[0, -1] < 0.0
    # Every size-K subset fails, so the signed lambda term is a constant and
    # supplies no fabricated relative credit.
    without_lambda = compute_canonical_clasp_set_advantages(
        torch.zeros_like(violations),
        violations,
        torch.zeros_like(violations, dtype=torch.bool),
        rollout_batch_size=5,
        deployment_set_size=4,
        group_lambdas=0.0,
        recovery_scale=1.0,
        recovery_temperature=0.2,
    )
    torch.testing.assert_close(result.advantages, without_lambda.advantages)


def test_canonical_clasp_component_decomposition_and_target_shift_invariance() -> None:
    objective = torch.tensor([[0.8, 0.0, 0.5, 0.0]], dtype=torch.float64)
    violation = torch.tensor([[0.0, 0.2, 0.0, 1.5]], dtype=torch.float64)
    feasibility = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    rho = 0.7
    multiplier = 1.3
    zero_target = compute_canonical_clasp_set_advantages(
        objective,
        violation,
        feasibility,
        rollout_batch_size=4,
        deployment_set_size=2,
        group_lambdas=multiplier,
        recovery_scale=rho,
        target_failure_rates=0.0,
    )
    shifted_target = compute_canonical_clasp_set_advantages(
        objective,
        violation,
        feasibility,
        rollout_batch_size=4,
        deployment_set_size=2,
        group_lambdas=multiplier,
        recovery_scale=rho,
        target_failure_rates=0.6,
    )
    expected = (
        zero_target.quality_advantages
        + rho * zero_target.recovery_advantages
        - multiplier * zero_target.failure_advantages
    )
    torch.testing.assert_close(zero_target.advantages, expected)
    torch.testing.assert_close(shifted_target.advantages, zero_target.advantages)
    torch.testing.assert_close(
        zero_target.deployment_failure_estimates,
        zero_target.subset_failures.mean(dim=-1),
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"deployment_set_size": 4}, "strictly less"),
        ({"recovery_temperature": 0.0}, "temperature"),
        ({"recovery_scale": -1.0}, "recovery_scale"),
        ({"target_failure_rates": 1.1}, "target_failure_rates"),
    ],
)
def test_canonical_clasp_rejects_invalid_experiment_parameters(
    kwargs: dict[str, float | int],
    match: str,
) -> None:
    values = {
        "rollout_batch_size": 4,
        "deployment_set_size": 3,
        "group_lambdas": 0.0,
        **kwargs,
    }
    with pytest.raises((TypeError, ValueError), match=match):
        compute_canonical_clasp_set_advantages(
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(4),
            **values,
        )
