from __future__ import annotations

import math

import pytest
import torch

from clasp.set_grpo import (
    DEFAULT_QUADRATURE_WEIGHTS,
    DEFAULT_THRESHOLD_MULTIPLIERS,
    compute_frontier_cdf_advantages,
    update_signed_bucket_lambdas,
    update_signed_bucket_lambdas_from_counts,
)


def _objective_from_gap(gap: float) -> float:
    return 2.0 * math.exp(-3.0 * max(0.0, gap))


def _default_call(
    objective: torch.Tensor,
    violations: torch.Tensor,
    feasible: torch.Tensor,
    *,
    lambdas: torch.Tensor | float = 1.0,
    target_gaps: torch.Tensor | float = 0.1,
    K: int | None = None,
    **kwargs,
):
    return compute_frontier_cdf_advantages(
        objective,
        violations,
        feasible,
        int(K if K is not None else objective.shape[-1]),
        group_lambdas=lambdas,
        group_target_gaps=target_gaps,
        **kwargs,
    )


def _brute_utility(
    objective: torch.Tensor,
    violations: torch.Tensor,
    feasible: torch.Tensor,
    *,
    group_lambda: float,
    target_gap: float,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_MULTIPLIERS,
    weights: tuple[float, ...] = DEFAULT_QUADRATURE_WEIGHTS,
    tau_q: float = 0.15,
    tau_v: float = 0.15,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    objective = objective.to(torch.float64)
    violations = violations.to(torch.float64)
    feasible = feasible.bool()
    gap = -torch.log((objective / 2.0).clamp(min=1e-12, max=1.0)) / 3.0
    z = gap / target_gap
    thresholds_t = torch.tensor(thresholds, dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)
    weights_t = weights_t / weights_t.sum()
    probabilities = torch.sigmoid((thresholds_t.unsqueeze(0) - z.unsqueeze(1)) / tau_q)
    probabilities = probabilities * feasible.unsqueeze(1)
    quality = (weights_t * (1.0 - torch.prod(1.0 - probabilities, dim=0))).sum()
    failed = not bool(feasible.any().item())
    d = violations / (1.0 + violations)
    recovery = tau_v * (torch.logsumexp(-d / tau_v, dim=0) - math.log(d.numel()))
    return quality - group_lambda * float(failed) + recovery_scale * float(failed) * recovery


def test_default_weights_are_the_declared_normalized_right_riemann_widths() -> None:
    assert sum(DEFAULT_QUADRATURE_WEIGHTS) == pytest.approx(3.0)
    assert DEFAULT_THRESHOLD_MULTIPLIERS == (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def test_quality_utility_matches_direct_frontier_cdf_formula() -> None:
    objective = torch.tensor(
        [[_objective_from_gap(0.05), _objective_from_gap(0.20), 0.0]],
        dtype=torch.float64,
    )
    violations = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    feasible = torch.tensor([[1, 1, 0]])
    result = _default_call(objective, violations, feasible, lambdas=0.7, target_gaps=0.1)

    expected = _brute_utility(
        objective[0],
        violations[0],
        feasible[0],
        group_lambda=0.7,
        target_gap=0.1,
    )
    torch.testing.assert_close(result.quality_utilities[0], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.group_utilities[0], expected, rtol=1e-12, atol=1e-12)


def test_null_difference_rewards_match_brute_force_fixed_k_replacement() -> None:
    objective = torch.tensor(
        [
            [_objective_from_gap(0.04), 0.0, _objective_from_gap(0.12), 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    violations = torch.tensor(
        [[0.0, 0.3, 0.0, 4.0], [0.05, 0.2, 1.0, 20.0]], dtype=torch.float64
    )
    feasible = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]])
    lambdas = torch.tensor([0.8, 2.5], dtype=torch.float64)
    targets = torch.tensor([0.1, 0.25], dtype=torch.float64)
    result = _default_call(
        objective,
        violations,
        feasible,
        lambdas=lambdas,
        target_gaps=targets,
    )

    for group_idx in range(2):
        full = _brute_utility(
            objective[group_idx],
            violations[group_idx],
            feasible[group_idx],
            group_lambda=float(lambdas[group_idx]),
            target_gap=float(targets[group_idx]),
        )
        torch.testing.assert_close(result.group_utilities[group_idx], full)
        for candidate_idx in range(4):
            null_objective = objective[group_idx].clone()
            null_violation_d = violations[group_idx] / (1.0 + violations[group_idx])
            null_feasible = feasible[group_idx].clone()
            null_objective[candidate_idx] = 0.0
            null_feasible[candidate_idx] = 0

            # The utility helper consumes raw violations, while the specified
            # null has normalized d exactly one. Compute this one term directly
            # rather than approximating it with a huge raw violation.
            gap = -torch.log((null_objective / 2.0).clamp(min=1e-12, max=1.0)) / 3.0
            z = gap / targets[group_idx]
            threshold_t = torch.tensor(DEFAULT_THRESHOLD_MULTIPLIERS, dtype=torch.float64)
            weight_t = torch.tensor(DEFAULT_QUADRATURE_WEIGHTS, dtype=torch.float64)
            weight_t /= weight_t.sum()
            p = torch.sigmoid((threshold_t.unsqueeze(0) - z.unsqueeze(1)) / 0.15)
            p *= null_feasible.bool().unsqueeze(1)
            q_null = (weight_t * (1.0 - torch.prod(1.0 - p, dim=0))).sum()
            null_violation_d[candidate_idx] = 1.0
            r_null = 0.15 * (
                torch.logsumexp(-null_violation_d / 0.15, dim=0) - math.log(4.0)
            )
            failed_null = not bool(null_feasible.any().item())
            null_u = (
                q_null
                - lambdas[group_idx] * float(failed_null)
                + float(failed_null) * r_null
            )
            torch.testing.assert_close(
                result.null_group_utilities[group_idx, candidate_idx], null_u
            )
            torch.testing.assert_close(
                result.advantages[group_idx, candidate_idx], full - null_u, atol=1e-12, rtol=1e-12
            )


def test_infeasible_candidate_in_successful_group_gets_exactly_zero() -> None:
    objective = torch.tensor([[_objective_from_gap(0.08), 0.0, 0.0]])
    violations = torch.tensor([[0.0, 0.001, 1000.0]])
    feasible = torch.tensor([[1, 0, 0]])
    result = _default_call(objective, violations, feasible, lambdas=3.0)

    assert result.advantages[0, 1].item() == 0.0
    assert result.advantages[0, 2].item() == 0.0
    assert result.quality_credits[0, 1].item() == 0.0
    assert result.failure_credits[0, 1].item() == 0.0
    assert result.recovery_credits[0, 1].item() == 0.0


def test_soft_cdf_gives_dense_quality_credit_to_ties_and_prefers_better_gap() -> None:
    tied_objective = torch.full((1, 4), _objective_from_gap(0.1), dtype=torch.float64)
    all_feasible = torch.ones((1, 4), dtype=torch.bool)
    tied = _default_call(tied_objective, torch.zeros_like(tied_objective), all_feasible)
    assert bool((tied.quality_credits > 0.0).all().item())
    torch.testing.assert_close(
        tied.quality_credits,
        tied.quality_credits[:, :1].expand_as(tied.quality_credits),
    )

    objective = torch.tensor(
        [[_objective_from_gap(0.04), _objective_from_gap(0.18)]], dtype=torch.float64
    )
    result = _default_call(objective, torch.zeros_like(objective), torch.ones_like(objective))
    assert result.quality_credits[0, 0] > result.quality_credits[0, 1]


def test_lambda_affects_only_singleton_group_success_credit() -> None:
    objective = torch.tensor(
        [
            [_objective_from_gap(0.08), 0.0, 0.0],
            [_objective_from_gap(0.08), _objective_from_gap(0.12), 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    feasible = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    violations = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 1.0], [0.1, 0.3, 2.0]])
    low = _default_call(objective, violations, feasible, lambdas=0.0)
    high = _default_call(objective, violations, feasible, lambdas=2.0)

    assert high.advantages[0, 0] - low.advantages[0, 0] == pytest.approx(2.0)
    torch.testing.assert_close(high.advantages[1], low.advantages[1])
    torch.testing.assert_close(high.advantages[2], low.advantages[2])
    assert high.failure_credits[0, 0].item() == pytest.approx(2.0)
    assert int(torch.count_nonzero(high.failure_credits).item()) == 1


def test_all_infeasible_recovery_is_dense_monotone_and_lambda_independent() -> None:
    objective = torch.zeros((1, 4), dtype=torch.float64)
    violations = torch.tensor([[0.01, 0.2, 2.0, 200.0]], dtype=torch.float64)
    feasible = torch.zeros((1, 4), dtype=torch.bool)
    low_lambda = _default_call(objective, violations, feasible, lambdas=0.0)
    high_lambda = _default_call(objective, violations, feasible, lambdas=4.0)

    assert bool((low_lambda.advantages > 0.0).all().item())
    assert bool(
        (low_lambda.advantages[0, :-1] > low_lambda.advantages[0, 1:]).all().item()
    )
    torch.testing.assert_close(low_lambda.advantages, high_lambda.advantages)
    torch.testing.assert_close(low_lambda.advantages, low_lambda.recovery_credits)


def test_equal_all_infeasible_candidates_receive_equal_positive_recovery() -> None:
    result = _default_call(
        torch.zeros((1, 8), dtype=torch.float64),
        torch.full((1, 8), 0.25, dtype=torch.float64),
        torch.zeros((1, 8), dtype=torch.bool),
    )
    assert result.advantages[0, 0].item() > 0.0
    torch.testing.assert_close(
        result.advantages,
        result.advantages[:, :1].expand_as(result.advantages),
    )


def test_permutation_equivariance() -> None:
    objective = torch.tensor(
        [[_objective_from_gap(0.03), 0.0, _objective_from_gap(0.15), 0.0]],
        dtype=torch.float64,
    )
    violations = torch.tensor([[0.0, 0.2, 0.0, 3.0]], dtype=torch.float64)
    feasible = torch.tensor([[1, 0, 1, 0]])
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    original = _default_call(objective, violations, feasible)
    permuted = _default_call(
        objective[:, permutation], violations[:, permutation], feasible[:, permutation]
    )
    torch.testing.assert_close(original.advantages, permuted.advantages[:, inverse])
    torch.testing.assert_close(original.group_utilities, permuted.group_utilities)


def test_flat_and_matrix_inputs_agree_and_preserve_candidate_shape() -> None:
    objective = torch.tensor(
        [
            [_objective_from_gap(0.05), 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    violations = torch.tensor([[0.0, 1.0, 3.0], [0.1, 0.2, 0.3]])
    feasible = torch.tensor([[1, 0, 0], [0, 0, 0]])
    matrix = _default_call(objective, violations, feasible, K=3)
    flat = _default_call(
        objective.flatten(), violations.flatten(), feasible.flatten(), K=3
    )
    assert matrix.advantages.shape == (2, 3)
    assert flat.advantages.shape == (6,)
    torch.testing.assert_close(matrix.advantages.flatten(), flat.advantages)
    torch.testing.assert_close(matrix.normalized_gaps.flatten(), flat.normalized_gaps)


def test_per_group_targets_are_applied_independently() -> None:
    objective = torch.full((2, 2), _objective_from_gap(0.1), dtype=torch.float64)
    feasible = torch.ones((2, 2), dtype=torch.bool)
    result = _default_call(
        objective,
        torch.zeros_like(objective),
        feasible,
        target_gaps=torch.tensor([0.05, 0.2], dtype=torch.float64),
        lambdas=torch.tensor([0.0, 4.0], dtype=torch.float64),
    )
    assert result.quality_utilities[1] > result.quality_utilities[0]
    # Both groups have multiple feasible candidates, so different lambdas do
    # not contaminate quality credit.
    assert int(torch.count_nonzero(result.failure_credits).item()) == 0


def test_configurable_objective_mapping_is_inverted_exactly() -> None:
    scale = 5.0
    slope = 7.0
    gaps = torch.tensor([[0.0, 0.03, 0.2]], dtype=torch.float64)
    objective = scale * torch.exp(-slope * gaps)
    result = _default_call(
        objective,
        torch.zeros_like(objective),
        torch.ones_like(objective),
        target_gaps=0.1,
        objective_score_scale=scale,
        gap_slope=slope,
    )
    torch.testing.assert_close(result.normalized_gaps, gaps / 0.1, rtol=1e-12, atol=1e-12)
    assert result.stats["frontier_cdf/objective_score_scale"] == pytest.approx(scale)
    assert result.stats["frontier_cdf/gap_slope"] == pytest.approx(slope)


def test_optional_feasibility_violation_contract_fails_closed() -> None:
    objective = torch.tensor([[_objective_from_gap(0.1), 0.0]])
    # The first candidate is feasible exactly at the tolerance; the second is
    # infeasible just above it.
    violations = torch.tensor([[1e-6, 1.1e-6]])
    feasible = torch.tensor([[1, 0]])
    _default_call(
        objective,
        violations,
        feasible,
        feasibility_violation_tolerance=1e-6,
    )

    with pytest.raises(ValueError, match="disagree"):
        _default_call(
            objective,
            violations,
            torch.tensor([[0, 0]]),
            feasibility_violation_tolerance=1e-6,
        )

    # The contract is deliberately optional for evaluators whose aggregate
    # diagnostic violation is not the exact feasibility predicate.
    _default_call(objective, violations, torch.tensor([[0, 0]]))


def test_advantage_decomposition_is_exact_by_construction() -> None:
    result = _default_call(
        torch.tensor(
            [
                [_objective_from_gap(0.08), 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [_objective_from_gap(0.04), _objective_from_gap(0.14), 0.0],
            ],
            dtype=torch.float64,
        ),
        torch.tensor([[0.0, 0.2, 2.0], [0.03, 0.4, 3.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 0]]),
        lambdas=torch.tensor([0.7, 1.4, 2.1]),
    )
    decomposition = result.quality_credits + result.failure_credits + result.recovery_credits
    assert torch.equal(result.advantages, decomposition)
    torch.testing.assert_close(
        result.advantages,
        result.group_utilities.unsqueeze(1) - result.null_group_utilities,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.stats["frontier_cdf/null_decomposition_error_max"] < 1e-12


def test_null_baseline_preserves_the_exact_enumerated_policy_gradient() -> None:
    """Difference rewards have the same expected score gradient as set utility."""
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    probability = torch.sigmoid(theta)
    expected_utility = torch.zeros((), dtype=torch.float64)
    expected_difference_gradient = torch.zeros((), dtype=torch.float64)

    for first in (0, 1):
        for second in (0, 1):
            actions = torch.tensor([[first, second]])
            feasible = actions.bool()
            objective = feasible.to(torch.float64) * _objective_from_gap(0.1)
            violations = (~feasible).to(torch.float64)
            result = _default_call(
                objective,
                violations,
                feasible,
                lambdas=0.8,
                target_gaps=0.1,
            )
            count_successes = first + second
            sequence_probability = (
                probability**count_successes
                * (1.0 - probability) ** (2 - count_successes)
            )
            expected_utility = expected_utility + sequence_probability * result.group_utilities[0]
            score_derivative = actions.to(torch.float64) - probability.detach()
            expected_difference_gradient = expected_difference_gradient + (
                sequence_probability.detach()
                * (result.advantages.detach() * score_derivative).sum()
            )

    analytic_gradient = torch.autograd.grad(expected_utility, theta)[0]
    torch.testing.assert_close(
        expected_difference_gradient,
        analytic_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_extreme_finite_inputs_remain_finite() -> None:
    objective = torch.tensor(
        [[0.0, 1e-30, 2.0, 1e30], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    violations = torch.tensor(
        [[0.0, 1e-30, 1e10, 1e30], [0.0, 1e-20, 1e20, 1e30]], dtype=torch.float32
    )
    feasible = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    result = _default_call(
        objective,
        violations,
        feasible,
        target_gaps=torch.tensor([1e-30, 1e30]),
        quality_temperature=1e-30,
        violation_temperature=1e-30,
    )
    for tensor in (
        result.advantages,
        result.quality_credits,
        result.recovery_credits,
        result.normalized_gaps,
        result.group_utilities,
        result.null_group_utilities,
    ):
        assert bool(torch.isfinite(tensor).all().item())
    assert bool((result.advantages >= 0.0).all().item())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("objective", [[-0.1, 0.0]], "objective_scores"),
        ("violations", [[-0.1, 0.0]], "violation_scores"),
        ("feasible", [[0.5, 0.0]], "feasibility_flags"),
        ("lambdas", -0.1, "group_lambdas"),
        ("target_gaps", 0.0, "group_target_gaps"),
        ("quality_temperature", 0.0, "quality_temperature"),
        ("violation_temperature", -1.0, "violation_temperature"),
        ("recovery_scale", -1.0, "recovery_scale"),
        ("objective_score_scale", 0.0, "objective_score_scale"),
        ("gap_slope", 0.0, "gap_slope"),
        ("objective_epsilon", 2.0, "objective_epsilon"),
        (
            "feasibility_violation_tolerance",
            -1.0,
            "feasibility_violation_tolerance",
        ),
    ],
)
def test_invalid_scalar_or_candidate_config_is_rejected(field, value, message) -> None:
    args = {
        "objective": torch.tensor([[1.0, 0.0]]),
        "violations": torch.tensor([[0.0, 1.0]]),
        "feasible": torch.tensor([[1.0, 0.0]]),
        "lambdas": 1.0,
        "target_gaps": 0.1,
        "quality_temperature": 0.15,
        "violation_temperature": 0.15,
        "recovery_scale": 1.0,
        "objective_score_scale": 2.0,
        "gap_slope": 3.0,
        "objective_epsilon": 1e-12,
        "feasibility_violation_tolerance": None,
    }
    if field in {"objective", "violations", "feasible"}:
        args[field] = torch.tensor(value)
    else:
        args[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        _default_call(
            args["objective"],
            args["violations"],
            args["feasible"],
            lambdas=args["lambdas"],
            target_gaps=args["target_gaps"],
            quality_temperature=args["quality_temperature"],
            violation_temperature=args["violation_temperature"],
            recovery_scale=args["recovery_scale"],
            objective_score_scale=args["objective_score_scale"],
            gap_slope=args["gap_slope"],
            objective_epsilon=args["objective_epsilon"],
            feasibility_violation_tolerance=args["feasibility_violation_tolerance"],
        )


def test_invalid_quadrature_and_shapes_are_rejected() -> None:
    objective = torch.ones((1, 2))
    violations = torch.zeros((1, 2))
    feasible = torch.ones((1, 2))
    with pytest.raises(ValueError, match="equal length"):
        _default_call(
            objective,
            violations,
            feasible,
            threshold_multipliers=[0.5, 1.0],
            quadrature_weights=[1.0],
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        _default_call(
            objective,
            violations,
            feasible,
            threshold_multipliers=[1.0, 0.5],
            quadrature_weights=[1.0, 1.0],
        )
    with pytest.raises(ValueError, match="positive sum"):
        _default_call(
            objective,
            violations,
            feasible,
            threshold_multipliers=[0.5, 1.0],
            quadrature_weights=[0.0, 0.0],
        )
    with pytest.raises(ValueError, match="same flat or matrix"):
        _default_call(objective.flatten(), violations, feasible.flatten(), K=2)
    with pytest.raises(ValueError, match="greater than one"):
        _default_call(objective[:, :1], violations[:, :1], feasible[:, :1], K=1)


def test_stats_report_core_set_diagnostics() -> None:
    result = _default_call(
        torch.tensor([[_objective_from_gap(0.1), 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.1, 1.0]]),
        torch.tensor([[1, 0], [0, 0]]),
    )
    assert result.stats["frontier_cdf/group_failure_rate"] == pytest.approx(0.5)
    assert result.stats["frontier_cdf/singleton_feasible_group_rate"] == pytest.approx(0.5)
    assert result.stats["frontier_cdf/mean_lambda"] == pytest.approx(1.0)
    assert result.stats["frontier_cdf/mean_target_gap"] == pytest.approx(0.1)
    assert result.stats["frontier_cdf/quality_credit_group_nonzero_rate"] == pytest.approx(0.5)
    assert result.stats["frontier_cdf/feasible_normalized_gap_p50"] == pytest.approx(1.0)
    assert result.stats["frontier_cdf/best_feasible_normalized_gap_p90"] == pytest.approx(1.0)


def test_normalized_gap_diagnostics_are_zero_when_no_feasible_candidate_exists() -> None:
    result = _default_call(
        torch.zeros((2, 3)),
        torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.4, 0.8]]),
        torch.zeros((2, 3), dtype=torch.bool),
        K=3,
    )
    assert result.stats["frontier_cdf/feasible_normalized_gap_p50"] == 0.0
    assert result.stats["frontier_cdf/feasible_normalized_gap_p90"] == 0.0
    assert result.stats["frontier_cdf/best_feasible_normalized_gap_p50"] == 0.0
    assert result.stats["frontier_cdf/best_feasible_normalized_gap_p90"] == 0.0


def test_signed_bucket_controller_updates_in_both_directions_and_skips_absent_bucket() -> None:
    update = update_signed_bucket_lambdas(
        torch.tensor([0.2, 0.5, 0.7], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0, 0, 1, 1]),
        target_failure_rates=torch.tensor([0.25, 0.5, 0.1], dtype=torch.float64),
        learning_rates=torch.tensor([0.4, 0.2, 1.0], dtype=torch.float64),
        lambda_max=2.0,
    )
    # Bucket 0 observes 100% failures: 0.2 + .4*(1-.25) = .5.
    # Bucket 1 observes 0% failures:   0.5 + .2*(0-.5)  = .4.
    torch.testing.assert_close(update.lambdas, torch.tensor([0.5, 0.4, 0.7], dtype=torch.float64))
    torch.testing.assert_close(
        update.observed_failure_rates, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        update.group_counts, torch.tensor([2.0, 2.0, 0.0], dtype=torch.float64)
    )
    assert update.active_buckets.tolist() == [True, True, False]


def test_signed_bucket_controller_projects_at_zero_and_maximum() -> None:
    update = update_signed_bucket_lambdas_from_counts(
        torch.tensor([0.05, 0.95]),
        torch.tensor([0.0, 10.0]),
        torch.tensor([10.0, 10.0]),
        target_failure_rates=torch.tensor([1.0, 0.0]),
        learning_rates=2.0,
        lambda_max=1.0,
    )
    torch.testing.assert_close(update.lambdas, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(update.deltas, torch.tensor([-0.05, 0.05]))


def test_controller_flag_and_preaggregated_interfaces_agree() -> None:
    current = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    flags = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    buckets = torch.tensor([0, 0, 1, 1, 1])
    direct = update_signed_bucket_lambdas(
        current,
        flags,
        buckets,
        target_failure_rates=0.2,
        learning_rates=0.1,
        lambda_max=3.0,
    )
    aggregated = update_signed_bucket_lambdas_from_counts(
        current,
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 3.0, 0.0], dtype=torch.float64),
        target_failure_rates=0.2,
        learning_rates=0.1,
        lambda_max=3.0,
    )
    torch.testing.assert_close(direct.lambdas, aggregated.lambdas)
    torch.testing.assert_close(direct.observed_failure_rates, aggregated.observed_failure_rates)
    torch.testing.assert_close(direct.deltas, aggregated.deltas)


def test_empty_controller_observation_is_a_noop() -> None:
    current = torch.tensor([0.2, 0.4])
    update = update_signed_bucket_lambdas(
        current,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        target_failure_rates=0.1,
        learning_rates=0.5,
        lambda_max=2.0,
    )
    torch.testing.assert_close(update.lambdas, current)
    assert update.active_buckets.tolist() == [False, False]
    assert update.stats["frontier_cdf/dual_active_bucket_count"] == 0.0


@pytest.mark.parametrize(
    "failure_sums,counts,match",
    [
        ([-1.0, 0.0], [1.0, 1.0], "failure_sums"),
        ([2.0, 0.0], [1.0, 1.0], "failure_sums"),
        ([0.0, 0.0], [-1.0, 1.0], "group_counts"),
    ],
)
def test_controller_rejects_invalid_preaggregated_counts(failure_sums, counts, match) -> None:
    with pytest.raises(ValueError, match=match):
        update_signed_bucket_lambdas_from_counts(
            torch.tensor([0.2, 0.3]),
            torch.tensor(failure_sums),
            torch.tensor(counts),
            target_failure_rates=0.1,
            learning_rates=0.1,
            lambda_max=1.0,
        )


def test_controller_rejects_invalid_indices_flags_and_parameters() -> None:
    current = torch.tensor([0.2, 0.3])
    with pytest.raises(ValueError, match="out-of-range"):
        update_signed_bucket_lambdas(
            current,
            torch.tensor([1.0]),
            torch.tensor([2]),
            target_failure_rates=0.1,
            learning_rates=0.1,
            lambda_max=1.0,
        )
    with pytest.raises(ValueError, match="integer"):
        update_signed_bucket_lambdas(
            current,
            torch.tensor([1.0]),
            torch.tensor([0.5]),
            target_failure_rates=0.1,
            learning_rates=0.1,
            lambda_max=1.0,
        )
    with pytest.raises(ValueError, match="group_failure_flags"):
        update_signed_bucket_lambdas(
            current,
            torch.tensor([1.5]),
            torch.tensor([0]),
            target_failure_rates=0.1,
            learning_rates=0.1,
            lambda_max=1.0,
        )
    with pytest.raises(ValueError, match="target_failure_rates"):
        update_signed_bucket_lambdas(
            current,
            torch.tensor([1.0]),
            torch.tensor([0]),
            target_failure_rates=1.1,
            learning_rates=0.1,
            lambda_max=1.0,
        )
    with pytest.raises(ValueError, match="learning_rates"):
        update_signed_bucket_lambdas(
            current,
            torch.tensor([1.0]),
            torch.tensor([0]),
            target_failure_rates=0.1,
            learning_rates=-0.1,
            lambda_max=1.0,
        )


@pytest.mark.parametrize(
    ("task", "sft_z90", "initial_temperature"),
    [
        ("tsp", 8.609, 1.40),
        ("cvrp", 3.176, 0.25),
        ("op", 4.633, 0.45),
        ("mis", 8.741, 1.45),
        ("jssp", 4.245, 0.35),
    ],
)
def test_preregistered_initial_temperature_covers_frozen_sft_frontier(
    task: str,
    sft_z90: float,
    initial_temperature: float,
) -> None:
    del task
    # The loosest threshold carries weight 1/3. Even in the conservative case
    # where all eight candidates sit at z90, its analytic CDF marginal must be
    # materially nonzero; otherwise the run would start with a dead quality head.
    p = torch.sigmoid(torch.tensor((3.0 - sft_z90) / initial_temperature)).item()
    equal_group_marginal = (1.0 / 3.0) * p * ((1.0 - p) ** 7)
    assert equal_group_marginal >= 5e-3
