from __future__ import annotations

import numpy as np
import torch

from clasp.set_grpo import canonical_finite_batch_advantages, enumerate_k_subsets
from experiments.canonical_bandit.run_experiment import (
    UtilityConfig,
    enumerate_action_batches,
    estimate_batch,
    exact_target,
    make_contexts,
    policy_for_seed,
    tuple_probabilities,
    utility_from_features,
)


def test_synthetic_set_utility_is_permutation_invariant_and_strictly_gated() -> None:
    config = UtilityConfig()
    feasibility = np.array([[False, True, False], [False, False, False]])
    quality = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0]])
    violations = np.array([[0.4, 0.0, 0.2], [0.1, 0.4, 0.8]])

    original = utility_from_features(feasibility, quality, violations, config)
    permutation = np.array([2, 0, 1])
    permuted = utility_from_features(
        feasibility[:, permutation],
        quality[:, permutation],
        violations[:, permutation],
        config,
    )

    np.testing.assert_allclose(permuted, original, rtol=0.0, atol=0.0)
    assert original[0] == 0.7
    assert original[1] < -config.failure_penalty


def test_exact_estimator_means_and_naive_finite_population_bias() -> None:
    context = make_contexts()["mixed_quality"]
    config = UtilityConfig()
    probabilities = policy_for_seed(context, 4)
    set_size, batch_size = 2, 4
    expected_utility, true_gradient = exact_target(
        probabilities,
        context,
        set_size,
        config,
    )
    actions = enumerate_action_batches(context.action_count, batch_size)
    probability_mass = tuple_probabilities(actions, probabilities)
    batch = estimate_batch(
        actions,
        probabilities,
        context,
        set_size,
        batch_size,
        config,
        expected_utility,
    )

    for method in ("canonical", "fixed_null", "raw", "oracle_constant"):
        exact_mean = np.einsum(
            "n,na->a",
            probability_mass,
            batch.gradients[method],
        )
        np.testing.assert_allclose(exact_mean, true_gradient, rtol=1e-11, atol=1e-12)

    naive_mean = np.einsum(
        "n,na->a",
        probability_mass,
        batch.gradients["naive_centered"],
    )
    expected_factor = (batch_size - set_size) / batch_size
    np.testing.assert_allclose(
        naive_mean,
        expected_factor * true_gradient,
        rtol=1e-11,
        atol=1e-12,
    )


def test_tied_all_infeasible_context_rejects_spurious_canonical_signal() -> None:
    context = make_contexts()["all_infeasible_tied"]
    config = UtilityConfig()
    probabilities = policy_for_seed(context, 7)
    set_size, batch_size = 3, 4
    expected_utility, true_gradient = exact_target(
        probabilities,
        context,
        set_size,
        config,
    )
    actions = enumerate_action_batches(context.action_count, batch_size)
    batch = estimate_batch(
        actions,
        probabilities,
        context,
        set_size,
        batch_size,
        config,
        expected_utility,
    )

    np.testing.assert_allclose(true_gradient, 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        batch.credits["canonical"],
        0.0,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        batch.gradients["canonical"],
        0.0,
        rtol=0.0,
        atol=1e-13,
    )
    assert float(np.var(batch.gradients["fixed_null"], axis=0).sum()) > 1e-8


def test_canonical_candidate_credits_are_exactly_centered() -> None:
    context = make_contexts()["rare_feasible"]
    config = UtilityConfig()
    probabilities = policy_for_seed(context, 2)
    expected_utility, _ = exact_target(probabilities, context, 3, config)
    actions = enumerate_action_batches(context.action_count, 5)
    batch = estimate_batch(
        actions,
        probabilities,
        context,
        3,
        5,
        config,
        expected_utility,
    )

    np.testing.assert_allclose(
        batch.credits["canonical"].sum(axis=-1),
        0.0,
        rtol=0.0,
        atol=2e-15,
    )


def test_numpy_experiment_credit_matches_library_estimator() -> None:
    context = make_contexts()["rare_feasible"]
    config = UtilityConfig()
    probabilities = policy_for_seed(context, 5)
    expected_utility, _ = exact_target(probabilities, context, 3, config)
    actions = enumerate_action_batches(context.action_count, 5)[::137]
    batch = estimate_batch(
        actions,
        probabilities,
        context,
        3,
        5,
        config,
        expected_utility,
    )
    subsets = enumerate_k_subsets(5, 3)
    library = canonical_finite_batch_advantages(
        torch.as_tensor(batch.subset_utilities, dtype=torch.float64),
        subsets,
        batch_size=5,
    )

    np.testing.assert_allclose(
        batch.credits["canonical"],
        library.advantages.numpy(),
        rtol=1e-13,
        atol=1e-14,
    )
