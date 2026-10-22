from __future__ import annotations

import itertools

import pytest
import torch

from clasp.set_grpo.fixed_null import (
    FixedNullAdvantageResult,
    fixed_null_finite_batch_advantages,
)


def _subsets(batch_size: int, set_size: int) -> torch.Tensor:
    return torch.combinations(torch.arange(batch_size), r=set_size)


def _brute_fixed_null_advantages(
    subset_utilities: torch.Tensor,
    subset_indices: torch.Tensor,
    replacement_utilities: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    candidate_advantages = []
    for candidate_idx in range(batch_size):
        terms = []
        for subset_idx, subset in enumerate(subset_indices.tolist()):
            for member_idx, member in enumerate(subset):
                if member == candidate_idx:
                    terms.append(
                        subset_utilities[..., subset_idx]
                        - replacement_utilities[..., subset_idx, member_idx]
                    )
        candidate_advantages.append(torch.stack(terms, dim=-1).mean(dim=-1))
    return torch.stack(candidate_advantages, dim=-1)


def _symmetric_utility(selected: torch.Tensor) -> torch.Tensor:
    return (
        selected.square().sum(dim=-1)
        + 0.3 * selected.prod(dim=-1)
        + 0.2 * selected.max(dim=-1).values
    )


def _fixed_null_tables(
    candidate_values: torch.Tensor,
    subset_indices: torch.Tensor,
    null_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = candidate_values[subset_indices]
    set_size = int(subset_indices.size(1))
    replaced = selected.unsqueeze(-2).expand(-1, set_size, -1).clone()
    positions = torch.arange(set_size)
    replaced[:, positions, positions] = null_value
    return _symmetric_utility(selected), _symmetric_utility(replaced)


def test_fixed_null_transform_matches_brute_force_with_leading_axes() -> None:
    batch_size = 6
    set_size = 3
    subsets = _subsets(batch_size, set_size)
    generator = torch.Generator().manual_seed(81021)
    utilities = torch.randn(
        (2, 4, subsets.size(0)),
        generator=generator,
        dtype=torch.float64,
    )
    replacements = torch.randn(
        (2, 4, subsets.size(0), set_size),
        generator=generator,
        dtype=torch.float64,
    )

    result = fixed_null_finite_batch_advantages(
        utilities,
        subsets,
        replacements,
        batch_size=batch_size,
    )
    expected = _brute_fixed_null_advantages(
        utilities,
        subsets,
        replacements,
        batch_size,
    )

    assert isinstance(result, FixedNullAdvantageResult)
    assert result.advantages.shape == (2, 4, batch_size)
    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(
        result.subset_member_differences,
        utilities.unsqueeze(-1) - replacements,
    )
    torch.testing.assert_close(
        result.realized_advantage_sums,
        expected.sum(dim=-1),
    )
    torch.testing.assert_close(
        result.realized_advantage_means,
        expected.mean(dim=-1),
    )
    for value in (
        result.advantages,
        result.mean_reduced_policy_coefficients,
        result.subset_member_differences,
        result.realized_advantage_sums,
        result.realized_advantage_means,
    ):
        assert value.dtype == utilities.dtype
        assert value.device == utilities.device


def test_candidate_permutation_equivariance_for_symmetric_fixed_null_utility() -> None:
    batch_size = 5
    set_size = 3
    subsets = _subsets(batch_size, set_size)
    values = torch.tensor([0.2, 1.3, -0.7, 2.1, 0.8], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    utilities, replacements = _fixed_null_tables(values, subsets, null_value=-0.4)
    original = fixed_null_finite_batch_advantages(
        utilities,
        subsets,
        replacements,
        batch_size=batch_size,
    )
    permuted_utilities, permuted_replacements = _fixed_null_tables(
        values[permutation],
        subsets,
        null_value=-0.4,
    )
    permuted = fixed_null_finite_batch_advantages(
        permuted_utilities,
        subsets,
        permuted_replacements,
        batch_size=batch_size,
    )

    torch.testing.assert_close(permuted.advantages, original.advantages[permutation])
    torch.testing.assert_close(
        permuted.mean_reduced_policy_coefficients,
        original.mean_reduced_policy_coefficients[permutation],
    )


def test_fixed_null_advantages_preserve_realized_common_mode_without_centering() -> None:
    batch_size = 4
    set_size = 2
    subsets = _subsets(batch_size, set_size)
    values = torch.tensor([-0.5, 0.25, 1.25, 2.0], dtype=torch.float64)

    def additive_tables(null_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        selected = values[subsets]
        utilities = selected.sum(dim=-1)
        replacements = utilities.unsqueeze(-1) - selected + null_value
        return utilities, replacements

    zero_null_utilities, zero_null_replacements = additive_tables(0.0)
    zero_null = fixed_null_finite_batch_advantages(
        zero_null_utilities,
        subsets,
        zero_null_replacements,
        batch_size=batch_size,
    )
    fixed_null_value = -1.5
    utilities, replacements = additive_tables(fixed_null_value)
    result = fixed_null_finite_batch_advantages(
        utilities,
        subsets,
        replacements,
        batch_size=batch_size,
    )

    expected = values - fixed_null_value
    torch.testing.assert_close(result.advantages, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        result.advantages,
        zero_null.advantages - fixed_null_value,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.realized_advantage_sums,
        expected.sum(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.realized_advantage_means,
        expected.mean(),
        rtol=0.0,
        atol=0.0,
    )
    assert result.realized_advantage_sums != 0.0


def test_equal_original_and_replacement_utilities_give_exact_zero() -> None:
    batch_size = 5
    set_size = 3
    subsets = _subsets(batch_size, set_size)
    utilities = torch.linspace(-1.0, 2.0, subsets.size(0), dtype=torch.float64).repeat(
        2, 1
    )
    replacements = utilities.unsqueeze(-1).expand(-1, -1, set_size).clone()

    result = fixed_null_finite_batch_advantages(
        utilities,
        subsets,
        replacements,
        batch_size=batch_size,
    )

    torch.testing.assert_close(
        result.advantages,
        torch.zeros_like(result.advantages),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.subset_member_differences,
        torch.zeros_like(result.subset_member_differences),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.realized_advantage_sums,
        torch.zeros_like(result.realized_advantage_sums),
        rtol=0.0,
        atol=0.0,
    )


def test_mean_reduced_policy_coefficient_is_exactly_k_times_advantage() -> None:
    batch_size = 5
    set_size = 3
    subsets = _subsets(batch_size, set_size)
    utilities = torch.arange(subsets.size(0), dtype=torch.float64).reshape(1, -1)
    replacements = torch.cos(
        torch.arange(subsets.numel(), dtype=torch.float64)
    ).reshape(1, subsets.size(0), set_size)
    result = fixed_null_finite_batch_advantages(
        utilities,
        subsets,
        replacements,
        batch_size=batch_size,
    )
    scores = torch.tensor([0.2, -0.4, 0.7, 0.1, -0.3], dtype=torch.float64)

    torch.testing.assert_close(
        result.mean_reduced_policy_coefficients,
        float(set_size) * result.advantages,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        (result.mean_reduced_policy_coefficients * scores).mean(dim=-1),
        (float(set_size) / float(batch_size))
        * (result.advantages * scores).sum(dim=-1),
    )


def test_exhaustive_binary_policy_gradient_unbiasedness() -> None:
    """The fixed-null estimator matches a nonlinear symmetric binary utility."""

    batch_size = 3
    set_size = 2
    subsets = _subsets(batch_size, set_size)
    utility_by_count = torch.tensor([-0.7, 1.2, 0.3], dtype=torch.float64)

    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    probability = torch.sigmoid(theta)
    expected_utility = (
        (1.0 - probability).square() * utility_by_count[0]
        + 2.0 * probability * (1.0 - probability) * utility_by_count[1]
        + probability.square() * utility_by_count[2]
    )
    analytic_gradient = torch.autograd.grad(expected_utility, theta)[0]

    detached_probability = probability.detach()
    expected_estimator = torch.zeros((), dtype=torch.float64)
    expected_mean_coefficient_estimator = torch.zeros((), dtype=torch.float64)
    for action_tuple in itertools.product((0, 1), repeat=batch_size):
        actions = torch.tensor(action_tuple, dtype=torch.long)
        selected = actions[subsets]
        selected_counts = selected.sum(dim=-1)
        subset_utilities = utility_by_count[selected_counts]
        # The fixed null is action zero, so replacing member j subtracts that
        # member's binary action from the selected success count.
        replacement_counts = selected_counts.unsqueeze(-1) - selected
        replacement_utilities = utility_by_count[replacement_counts]
        result = fixed_null_finite_batch_advantages(
            subset_utilities,
            subsets,
            replacement_utilities,
            batch_size=batch_size,
        )

        success_count = int(sum(action_tuple))
        batch_probability = (
            detached_probability**success_count
            * (1.0 - detached_probability) ** (batch_size - success_count)
        )
        scores = actions.to(torch.float64) - detached_probability
        estimator = (float(set_size) / float(batch_size)) * (
            result.advantages * scores
        ).sum()
        mean_coefficient_estimator = (
            result.mean_reduced_policy_coefficients * scores
        ).mean()
        torch.testing.assert_close(estimator, mean_coefficient_estimator)
        expected_estimator += batch_probability * estimator
        expected_mean_coefficient_estimator += (
            batch_probability * mean_coefficient_estimator
        )

    torch.testing.assert_close(
        expected_estimator,
        analytic_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        expected_mean_coefficient_estimator,
        analytic_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("incomplete", "every size"),
        ("duplicate", "duplicate"),
        ("unsorted", "strictly increasing"),
        ("out_of_range", "outside"),
        ("wrong_index_dtype", "torch.long"),
        ("wrong_utility_length", "final subset_utilities"),
        ("wrong_replacement_subset_axis", "replacement_utilities must have shape"),
        ("wrong_replacement_member_axis", "replacement_utilities must have shape"),
        ("wrong_replacement_leading_shape", "replacement_utilities must have shape"),
        ("integer_utilities", "subset_utilities must have a floating-point"),
        ("integer_replacements", "replacement_utilities must have a floating-point"),
        ("different_dtypes", "same dtype"),
        ("nonfinite_utilities", "subset_utilities contains non-finite"),
        ("nonfinite_replacements", "replacement_utilities contains non-finite"),
        ("set_size_not_less_than_batch", "strictly less"),
    ],
)
def test_fixed_null_transform_rejects_malformed_tables(
    mutation: str,
    match: str,
) -> None:
    batch_size = 4
    subsets = _subsets(batch_size, 2)
    utilities = torch.arange(subsets.size(0), dtype=torch.float64)
    replacements = utilities.unsqueeze(-1).repeat(1, subsets.size(1))

    if mutation == "incomplete":
        subsets = subsets[:-1]
        utilities = utilities[:-1]
        replacements = replacements[:-1]
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
    elif mutation == "wrong_replacement_subset_axis":
        replacements = replacements[:-1]
    elif mutation == "wrong_replacement_member_axis":
        replacements = replacements[:, :-1]
    elif mutation == "wrong_replacement_leading_shape":
        utilities = utilities.repeat(2, 1)
        replacements = replacements.unsqueeze(0)
    elif mutation == "integer_utilities":
        utilities = utilities.to(torch.long)
        replacements = replacements.to(torch.long)
    elif mutation == "integer_replacements":
        replacements = replacements.to(torch.long)
    elif mutation == "different_dtypes":
        replacements = replacements.to(torch.float32)
    elif mutation == "nonfinite_utilities":
        utilities = utilities.clone()
        utilities[0] = float("nan")
    elif mutation == "nonfinite_replacements":
        replacements = replacements.clone()
        replacements[0, 0] = float("inf")
    elif mutation == "set_size_not_less_than_batch":
        batch_size = 2
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(f"Unknown mutation {mutation}")

    with pytest.raises((TypeError, ValueError), match=match):
        fixed_null_finite_batch_advantages(
            utilities,
            subsets,
            replacements,
            batch_size=batch_size,
        )


def test_fixed_null_transform_rejects_non_tensor_and_invalid_rank_inputs() -> None:
    subsets = _subsets(3, 2)
    utilities = torch.ones(subsets.size(0), dtype=torch.float64)
    replacements = torch.ones(subsets.size(0), 2, dtype=torch.float64)

    with pytest.raises(TypeError, match="subset_utilities"):
        fixed_null_finite_batch_advantages(
            [1.0, 2.0, 3.0],  # type: ignore[arg-type]
            subsets,
            replacements,
            batch_size=3,
        )
    with pytest.raises(TypeError, match="replacement_utilities"):
        fixed_null_finite_batch_advantages(
            utilities,
            subsets,
            [[1.0, 1.0]],  # type: ignore[arg-type]
            batch_size=3,
        )
    with pytest.raises(ValueError, match="replacement_utilities must have shape"):
        fixed_null_finite_batch_advantages(
            utilities,
            subsets,
            torch.ones(3, dtype=torch.float64),
            batch_size=3,
        )
    with pytest.raises(TypeError, match="batch_size"):
        fixed_null_finite_batch_advantages(
            utilities,
            subsets,
            replacements,
            batch_size=True,  # type: ignore[arg-type]
        )
