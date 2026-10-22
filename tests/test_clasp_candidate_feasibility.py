"""The v4 auxiliary closes a specific hole in B=9/K=8 feasibility credit."""

from pathlib import Path

import pytest
import torch

from clasp.set_grpo import (
    compute_canonical_clasp_set_advantages,
    compute_dense_clasp_auxiliary_advantages,
)


@pytest.mark.parametrize("feasible_count", [2, 4, 8])
def test_candidate_credit_survives_saturated_set_success(feasible_count):
    # Every eight-subset succeeds, so even a large hard-failure multiplier
    # cannot distinguish feasible from infeasible candidates in these groups.
    flags = (torch.arange(9) < feasible_count).to(torch.float64)
    objective = torch.linspace(1.0, 0.1, 9, dtype=torch.float64) * flags
    violation = 1.0 - flags
    canonical = compute_canonical_clasp_set_advantages(
        objective, violation, flags,
        rollout_batch_size=9, deployment_set_size=8,
        group_lambdas=3.0, recovery_scale=1.0, recovery_temperature=0.15,
    )
    torch.testing.assert_close(canonical.failure_advantages, torch.zeros(9, dtype=torch.float64))

    auxiliary = compute_dense_clasp_auxiliary_advantages(
        objective, violation, flags,
        rollout_batch_size=9, group_lambdas=3.0, feasibility_scale=0.5,
    ).advantages
    assert auxiliary[:feasible_count].min() > 0
    assert auxiliary[feasible_count:].max() < 0
    assert auxiliary.sum().abs() < 1e-12
    # The additional feasible-vs-infeasible margin is exactly .5*(1+lambda).
    assert (auxiliary[0] - auxiliary[-1]).item() == pytest.approx(2.0)
    # No new preference among feasible solutions, regardless of their quality.
    torch.testing.assert_close(
        auxiliary[:feasible_count], auxiliary[0].expand(feasible_count)
    )
    original = canonical.mean_reduced_policy_coefficients
    amended = original + auxiliary
    torch.testing.assert_close(amended[0] - amended[1], original[0] - original[1])


@pytest.mark.parametrize("feasible", [False, True])
def test_candidate_term_leaves_uniform_feasibility_groups_unchanged(feasible):
    flags = torch.full((9,), float(feasible), dtype=torch.float64)
    result = compute_dense_clasp_auxiliary_advantages(
        torch.linspace(0.1, 1.0, 9, dtype=torch.float64) * flags,
        torch.arange(9, dtype=torch.float64) * (1 - flags), flags,
        rollout_batch_size=9, group_lambdas=5.0, feasibility_scale=0.5,
    )
    torch.testing.assert_close(result.advantages, torch.zeros_like(flags))


def test_v4_changes_only_candidate_feasibility_from_v3():
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "train_clasp_canonical_checkpoint_dev_a100.slurm").read_text()

    def settings(method):
        block = source.split(method + ")\n", 1)[1].split("    ;;", 1)[0]
        return dict(line.strip().split("=", 1) for line in block.splitlines() if "=" in line)

    old, new = settings("clasp_grpo_anchor_v3"), settings("clasp_grpo_anchor_v4")
    differences = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    assert differences == {"ESTIMATOR_ARM", "POLICY_CREDIT_FORMULA", "DENSE_FEASIBILITY_SCALE"}
    assert new["DENSE_FEASIBILITY_SCALE"] == '"0.5"'
