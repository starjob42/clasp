"""Regression tests for the CLASP helpers.

Run with:
    pytest tests/test_clasp.py -q

These tests only exercise ``clasp.rewards.clasp_math`` and require
``torch`` + ``numpy``. They do not spin up the GRPO trainer.
"""
from __future__ import annotations

import math

import pytest
import torch

from clasp.data.loaders import augment_input_for_task, normalize_rows, transform_example_for_task
from clasp.rewards.clasp_math import (
    AdaptiveLengthEstimator,
    compute_bon_aligned_advantages,
    compute_elite_imitation_loss,
    compute_feasibility_gated_advantages,
    compute_feasibility_warmup_advantages,
    compute_hard_group_weighted_advantages,
    compute_lexicographic_clasp_advantages,
    compute_multi_elite_imitation_loss,
    compute_pairwise_preference_loss,
    compute_pareto_anchored_advantages,
    feasibility_conditional_dual_lr,
)
from clasp.rewards.co_tasks import (
    _check_cvrp_constraints,
    _check_jssp_constraints,
    _check_mis_constraints,
    _cvrp_local_search_improved_routes,
    _cvrp_total_distance,
    _format_cvrp_distillation_target,
    _format_jssp_distillation_target,
    _format_mis_distillation_target,
    _jssp_order_descent_improved_schedule,
    _mis_maximalized_set,
    _parse_cvrp_solution,
    _parse_jssp_solution,
    _parse_mis_solution,
    _schedule_makespan,
    _tsp_two_opt_improved_route,
    cvrp_cost_components,
    jssp_cost_components_hybrid,
    mis_cost_components,
    mis_objective_reward,
    op_cost_components,
    tsp_cost_components,
    tsp_structure_reward,
)
from clasp.rewards.heads import MultiTaskPrimalDualManager, build_component_profile
from clasp.rewards.primal_dual import (
    PrimalDualConstrainedReward,
    _lexicographic_recovery_violation_score,
)
from clasp.trainers.primal_dual_grpo import (
    PrimalDualGRPOTrainer,
    _compose_clasp_recovery_gaps,
)


def _build_group(primal, violation, feasible):
    r = torch.tensor(primal, dtype=torch.float32)
    g = torch.tensor(violation, dtype=torch.float32)
    f = torch.tensor(feasible, dtype=torch.float32)
    return r, g, f


def test_self_replay_group_rows_expand_before_trl_shuffle():
    group_tensor = torch.arange(6, dtype=torch.long).reshape(3, 2)

    expanded = PrimalDualGRPOTrainer._repeat_group_tensor_for_generation_batch(
        group_tensor,
        num_generations=4,
        batch_size=12,
    )

    assert expanded.shape == (12, 2)
    assert torch.equal(expanded[0:4], group_tensor[0].repeat(4, 1))
    assert torch.equal(expanded[4:8], group_tensor[1].repeat(4, 1))
    assert torch.equal(expanded[8:12], group_tensor[2].repeat(4, 1))

    output = {
        "completion_ids": torch.zeros((12, 5), dtype=torch.long),
        "self_replay_completion_ids": expanded,
        "advantages": torch.zeros(12),
    }
    PrimalDualGRPOTrainer._validate_trl_generation_batch_rows(output)


def test_generation_batch_row_invariant_catches_group_level_tensors():
    output = {
        "completion_ids": torch.zeros((12, 5), dtype=torch.long),
        "self_replay_completion_ids": torch.zeros((3, 5), dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="self_replay_completion_ids"):
        PrimalDualGRPOTrainer._validate_trl_generation_batch_rows(output)


def test_normalize_rows_preserves_optional_imitation_outputs():
    rows = [
        {
            "instruction": "Solve this TSP.",
            "input": "coords",
            "output": "Route: [0, 1]\nObjective: 1.0",
            "instance": {"coords": [[0.0, 0.0], [1.0, 0.0]]},
            "reference_imitation_output": "Route: [0, 1]",
            "imitation_output": "Route: [0, 1]",
            "negative_imitation_output": "Route: [1, 0]",
        }
    ]

    normalized = normalize_rows(rows, "memory.json")

    assert normalized[0]["reference_imitation_output"] == "Route: [0, 1]"
    assert normalized[0]["imitation_output"] == "Route: [0, 1]"
    assert normalized[0]["negative_imitation_output"] == "Route: [1, 0]"


def test_op_strict_length_profile_matches_eval_feasibility():
    default_limits, default_weights = build_component_profile("op", "co_default")
    strict_limits, strict_weights = build_component_profile("op", "co_op_strict_length")
    cap_limits, cap_weights = build_component_profile("op", "co_op_strict_length_prizecap")
    margin_limits, margin_weights = build_component_profile("op", "co_op_strict_length_margin")
    mis_limits, mis_weights = build_component_profile("mis", "co_op_strict_length")
    cap_mis_limits, cap_mis_weights = build_component_profile("mis", "co_op_strict_length_prizecap")
    margin_mis_limits, margin_mis_weights = build_component_profile("mis", "co_op_strict_length_margin")
    default_mis_limits, default_mis_weights = build_component_profile("mis", "co_default")

    assert default_limits["length_overflow"] > 0.0
    assert strict_limits["length_overflow"] == 0.0
    assert strict_weights["length_overflow"] > default_weights["length_overflow"]
    assert cap_limits["length_overflow"] == 0.0
    assert cap_limits["prize_surplus_overbudget"] == 0.0
    assert cap_limits["prize_cap_over_ref"] == 0.0
    assert "length_margin_overflow" not in cap_limits
    assert cap_weights["length_overflow"] > default_weights["length_overflow"]
    assert cap_weights["prize_surplus_overbudget"] > default_weights["prize_surplus_overbudget"]
    assert cap_weights["prize_cap_over_ref"] > default_weights["prize_cap_over_ref"]
    assert margin_limits["length_overflow"] == 0.0
    assert margin_limits["length_margin_overflow"] == 0.0
    assert margin_limits["prize_surplus_overbudget"] == 0.0
    assert margin_limits["prize_cap_over_ref"] == 0.0
    assert margin_weights["length_overflow"] > margin_weights["length_margin_overflow"]
    assert margin_weights["length_margin_overflow"] > default_weights["length_margin_overflow"]
    assert margin_weights["prize_surplus_overbudget"] > default_weights["prize_surplus_overbudget"]
    assert margin_weights["prize_cap_over_ref"] > default_weights["prize_cap_over_ref"]
    assert mis_limits == default_mis_limits
    assert mis_weights == default_mis_weights
    assert cap_mis_limits == default_mis_limits
    assert cap_mis_weights == default_mis_weights
    assert margin_mis_limits == default_mis_limits
    assert margin_mis_weights == default_mis_weights


def test_feasibility_lock_all_profile_tightens_all_task_validity_heads():
    expected = {
        "tsp": {
            "parse_failure": 10.0,
            "depot_boundary": 10.0,
            "route_length_mismatch": 10.0,
            "invalid_node": 10.0,
            "duplicate_customer": 10.0,
            "missing_customer": 10.0,
            "interior_depot": 8.0,
        },
        "op": {
            "parse_failure": 10.0,
            "start_at_depot": 10.0,
            "invalid_node": 10.0,
            "duplicate_visit": 8.0,
            "length_overflow": 12.0,
        },
        "cvrp": {
            "parse_failure": 10.0,
            "start_at_depot": 10.0,
            "end_at_depot": 10.0,
            "depot_boundary": 10.0,
            "route_shape_invalid": 10.0,
            "invalid_node": 10.0,
            "duplicate_customer": 10.0,
            "missing_customer": 12.0,
            "customer_count_mismatch": 10.0,
            "capacity_overflow": 12.0,
            "over_capacity_route_fraction": 14.0,
            "max_route_capacity_overflow": 14.0,
        },
        "mis": {
            "parse_failure": 10.0,
            "invalid_node": 10.0,
            "duplicate_node": 8.0,
            "edge_conflict": 24.0,
            "conflict_node_fraction": 24.0,
        },
        "jssp": {
            "parse_failure": 10.0,
            "machine_row_mismatch": 8.0,
            "row_length_mismatch": 8.0,
            "invalid_job": 10.0,
            "duplicate_job": 8.0,
            "missing_job": 10.0,
            "row_set_mismatch": 10.0,
            "schedule_invalid": 12.0,
        },
    }

    for task, locked_heads in expected.items():
        default_limits, default_weights = build_component_profile(task, "co_default")
        limits, weights = build_component_profile(task, "co_feasibility_lock_all")
        for name, min_weight in locked_heads.items():
            assert limits[name] == 0.0
            assert weights[name] >= min_weight
            assert weights[name] >= default_weights.get(name, 0.0)

        if task == "cvrp":
            assert default_limits["capacity_overflow"] > 0.0
            assert limits["capacity_overflow"] == 0.0
            assert limits["route_count_mismatch"] == 1.0
            assert limits["interior_depot"] == 1.0
            assert weights["route_count_mismatch"] >= default_weights["route_count_mismatch"]
            assert weights["interior_depot"] >= default_weights["interior_depot"]
        if task == "op":
            assert default_limits["length_overflow"] > 0.0
            assert limits["length_overflow"] == 0.0
            assert limits["length_margin_overflow"] == 1.0
            assert limits["prize_surplus_overbudget"] == 1.0
            assert limits["prize_cap_over_ref"] == 1.0
        if task == "mis":
            assert limits["size_deficit_over_ref"] == 1.0
            assert limits["size_excess_over_ref"] == 1.0
            assert limits["repair_size_deficit_over_ref"] == 1.0
            assert limits["addable_vertex_fraction"] == 1.0
        if task == "jssp":
            assert limits["makespan_gap_over_ref"] == 1.0


def test_jssp_tail_feas_lock_profile_adds_makespan_pressure():
    for task in ("tsp", "op", "cvrp", "mis"):
        lock_limits, lock_weights = build_component_profile(task, "co_feasibility_lock_all")
        rescue_limits, rescue_weights = build_component_profile(task, "co_jssp_tail_feas_lock_all")
        assert rescue_limits == lock_limits
        assert rescue_weights == lock_weights

    default_limits, default_weights = build_component_profile("jssp", "co_default")
    lock_limits, lock_weights = build_component_profile("jssp", "co_feasibility_lock_all")
    rescue_limits, rescue_weights = build_component_profile("jssp", "co_jssp_tail_feas_lock_all")

    for name in (
        "parse_failure",
        "machine_row_mismatch",
        "row_length_mismatch",
        "invalid_job",
        "duplicate_job",
        "missing_job",
        "row_set_mismatch",
        "schedule_invalid",
    ):
        assert rescue_limits[name] == lock_limits[name] == 0.0
        assert rescue_weights[name] >= lock_weights[name]

    assert lock_limits["makespan_gap_over_ref"] == 1.0
    assert rescue_limits["makespan_gap_over_ref"] == 0.0
    assert rescue_weights["makespan_gap_over_ref"] >= 24.0


def test_cvrp_cost_components_expose_last_mile_feasibility_heads():
    coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    demands = [0.0, 2.0, 2.0, 1.0]
    kwargs = {
        "instance": [[coords, demands, 3.0]],
        "input": ["capacity = 3 with 2 vehicles"],
        "num_vehicles": 2,
    }
    completion = "Routes: [[1, 2, 0], [0, 2, 2, 0]]\nObjective: 10"

    output = cvrp_cost_components(["prompt"], [completion], **kwargs)
    names = output["cost_names"]
    row = output["cost_vectors"][0]
    by_name = dict(zip(names, row))

    for name in (
        "start_at_depot",
        "end_at_depot",
        "route_shape_invalid",
        "duplicate_customer",
        "missing_customer",
        "customer_count_mismatch",
        "capacity_overflow",
        "over_capacity_route_fraction",
        "max_route_capacity_overflow",
    ):
        assert name in names

    assert by_name["start_at_depot"] > 0.0
    assert by_name["end_at_depot"] == 0.0
    assert by_name["duplicate_customer"] > 0.0
    assert by_name["missing_customer"] > 0.0
    assert by_name["customer_count_mismatch"] > 0.0
    assert by_name["capacity_overflow"] > 0.0
    assert by_name["over_capacity_route_fraction"] > 0.0
    assert by_name["max_route_capacity_overflow"] > 0.0


def test_cvrp_recovery_never_prefers_blank_over_capacity_only_route():
    """Regression for the observed 10.0-vs-31.33 recovery inversion.

    The second completion is structurally valid and covers its only customer,
    but demand 5 exceeds capacity 3.  The legacy additive scalar triple-counted
    that one capacity failure and ranked the blank completion as easier to
    recover.  The lexicographic scalar must produce the opposite ordering and
    therefore positive all-infeasible BoN credit for the parsed route.
    """
    coords = [[0.0, 0.0], [1.0, 0.0]]
    demands = [0.0, 5.0]
    instances = [[coords, demands, 3.0], [coords, demands, 3.0]]
    prompts = ["Solve CVRP with capacity = 3 and 1 vehicle"] * 2
    completions = ["", "Routes: [[0, 1, 0]]\nObjective: 2"]

    limits, weights = build_component_profile("cvrp", "co_feasibility_lock_all")

    def zero_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    reward = PrimalDualConstrainedReward(
        structure_fn=zero_reward,
        constraint_fn=None,
        objective_fn=zero_reward,
        cost_components_fn=cvrp_cost_components,
        cost_limit=0.0,
        lambda_init=0.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
        component_cost_limit_map=limits,
        component_weight_map=weights,
    )
    reward(
        prompts,
        completions,
        instance=instances,
        input=["capacity = 3 with 1 vehicle"] * 2,
    )
    components = reward.get_last_components()

    raw_costs = cvrp_cost_components(
        prompts,
        completions,
        instance=instances,
        input=["capacity = 3 with 1 vehicle"] * 2,
    )
    additive_scores = [
        sum(
            weights[name] * mask * max(0.0, cost - limits[name])
            for name, cost, mask in zip(
                raw_costs["cost_names"],
                row,
                mask_row,
            )
        )
        for row, mask_row in zip(
            raw_costs["cost_vectors"],
            raw_costs["cost_masks"],
        )
    ]
    assert additive_scores[0] == pytest.approx(10.0)
    assert additive_scores[1] == pytest.approx(31.333333333333332)

    recovery_scores = components["raw_positive_gap_scores"]
    assert recovery_scores[0] > 1.0
    assert 0.0 < recovery_scores[1] < 1.0
    assert recovery_scores[1] < recovery_scores[0]
    assert components["feasibility_flags"] == [0.0, 0.0]

    advantages, _ = compute_bon_aligned_advantages(
        primal_scores=torch.zeros(2),
        violation_scores=torch.tensor(recovery_scores),
        feasibility_flags=torch.zeros(2),
        num_generations=2,
    )
    assert advantages[1] > advantages[0]


def test_large_component_duals_preserve_cvrp_parseability_after_save_resume():
    coords = [[0.0, 0.0], [1.0, 0.0]]
    demands = [0.0, 5.0]
    instances = [[coords, demands, 3.0], [coords, demands, 3.0]]
    prompts = ["Solve CVRP with capacity = 3 and 1 vehicle"] * 2
    completions = ["", "Routes: [[0, 1, 0]]\nObjective: 2"]
    inputs = ["capacity = 3 with 1 vehicle"] * 2
    limits, weights = build_component_profile("cvrp", "co_feasibility_lock_all")

    def build_manager():
        def zero_reward(prompts, completions, **kwargs):
            return [0.0] * len(completions)

        reward = PrimalDualConstrainedReward(
            structure_fn=zero_reward,
            constraint_fn=None,
            objective_fn=zero_reward,
            cost_components_fn=cvrp_cost_components,
            cost_limit=0.0,
            lambda_init=0.0,
            dual_lr=0.0,
            lambda_max=1e9,
            log_every=0,
            component_cost_limit_map=limits,
            component_weight_map=weights,
        )
        return MultiTaskPrimalDualManager({"cvrp": reward})

    original = build_manager()
    original(
        prompts,
        completions,
        task_type=["cvrp", "cvrp"],
        instance=instances,
        input=inputs,
    )
    wrapper = original.wrappers["cvrp"]
    wrapper.dual_lambdas = [
        1.0 if name == "parse_failure" else 1e8
        for name in wrapper.cost_names
    ]
    checkpoint_state = original.state_dict()

    restored = build_manager()
    restored.load_state_dict(checkpoint_state)
    restored(
        prompts,
        completions,
        task_type=["cvrp", "cvrp"],
        instance=instances,
        input=inputs,
    )
    components = restored.get_last_components()
    raw = torch.tensor(components["raw_positive_gap_scores"])
    dual_aware = torch.tensor(components["dual_aware_recovery_gap_scores"])
    active = _compose_clasp_recovery_gaps(
        raw,
        dual_aware,
        component_dual_scale=1e6,
    )

    assert restored.wrappers["cvrp"].dual_lambdas == wrapper.dual_lambdas
    assert raw[0] > 2.0 and raw[1] < 1.0
    assert components["dual_aware_recovery_gap_scores"][0] > 2.0
    assert components["dual_aware_recovery_gap_scores"][1] < 1.0
    assert dual_aware[0] > 2.0 and dual_aware[1] <= 1.0
    assert active[0] > 2.0 and active[1] <= 1.0
    assert active[0] > active[1]


@pytest.mark.parametrize(
    ("task", "nonparse_names"),
    [
        (
            "tsp",
            [
                "depot_boundary",
                "route_length_mismatch",
                "invalid_node",
                "duplicate_customer",
                "missing_customer",
            ],
        ),
        (
            "op",
            ["start_at_depot", "invalid_node", "duplicate_visit", "length_overflow"],
        ),
        (
            "mis",
            ["invalid_node", "duplicate_node", "edge_conflict", "conflict_node_fraction"],
        ),
        (
            "jssp",
            [
                "machine_row_mismatch",
                "row_length_mismatch",
                "invalid_job",
                "missing_job",
                "schedule_invalid",
            ],
        ),
    ],
)
def test_recovery_parseability_tier_dominates_many_active_task_heads(
    task,
    nonparse_names,
):
    """Parse masking cannot make malformed output look closer to feasible."""
    limits, weights = build_component_profile(task, "co_feasibility_lock_all")
    cost_names = ["parse_failure", *nonparse_names]

    blank_score = _lexicographic_recovery_violation_score(
        cost_names,
        [1.0] + ([0.0] * len(nonparse_names)),
        [1.0] + ([0.0] * len(nonparse_names)),
        [weights[name] for name in cost_names],
        [limits[name] for name in cost_names],
    )
    parsed_many_violations_score = _lexicographic_recovery_violation_score(
        cost_names,
        [0.0] + ([1.0] * len(nonparse_names)),
        [1.0] * len(cost_names),
        [weights[name] for name in cost_names],
        [limits[name] for name in cost_names],
    )

    assert blank_score > 1.0
    assert 0.0 < parsed_many_violations_score < 1.0
    assert parsed_many_violations_score < blank_score


def test_recovery_collapses_correlated_capacity_heads_by_maximum():
    names = [
        "capacity_overflow",
        "over_capacity_route_fraction",
        "max_route_capacity_overflow",
    ]
    row = [0.2, 0.5, 0.4]
    weights = [12.0, 14.0, 14.0]

    combined = _lexicographic_recovery_violation_score(
        names,
        row,
        [1.0, 1.0, 1.0],
        weights,
        [0.0, 0.0, 0.0],
    )
    strongest_only = _lexicographic_recovery_violation_score(
        ["over_capacity_route_fraction"],
        [0.5],
        [1.0],
        [14.0],
        [0.0],
    )

    assert combined == pytest.approx(strongest_only)


def test_jssp_cost_components_expose_invalid_job_and_row_set_mismatch():
    jobs = [[0, 1, 1, 1], [1, 1, 0, 1]]
    kwargs = {
        "instance": [jobs],
        "input": [""],
    }
    completion = "Schedule: [[0, 2], [1, 1]]\nMakespan: 5"

    output = jssp_cost_components_hybrid(["prompt"], [completion], **kwargs)
    names = output["cost_names"]
    row = output["cost_vectors"][0]
    by_name = dict(zip(names, row))

    assert "invalid_job" in names
    assert "row_set_mismatch" in names
    assert by_name["invalid_job"] > 0.0
    assert by_name["duplicate_job"] > 0.0
    assert by_name["missing_job"] > 0.0
    assert by_name["row_set_mismatch"] > 0.0


def test_op_length_margin_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    completions = ["Route: [0, 1]\nObjective: 10"]
    kwargs = {
        "input": [""],
        "instance": [[[[0.0, 0.0], [10.0, 0.0]], [0.0, 10.0], 10.0]],
    }

    monkeypatch.delenv("CLASP_OP_LENGTH_MARGIN_RATIO", raising=False)
    default_output = op_cost_components(prompts, completions, **kwargs)
    assert "length_margin_overflow" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_OP_LENGTH_MARGIN_RATIO", "0.985")
    margin_output = op_cost_components(prompts, completions, **kwargs)
    names = margin_output["cost_names"]
    assert "length_margin_overflow" in names
    assert margin_output["cost_vectors"][0][names.index("length_overflow")] == 0.0
    assert math.isclose(
        margin_output["cost_vectors"][0][names.index("length_margin_overflow")],
        0.015,
        rel_tol=1e-6,
    )


def test_op_singleton_depot_route_has_hard_length_violation():
    kwargs = {
        "input": [""],
        "instance": [[[[0.0, 0.0], [5.0, 0.0]], [0.0, 10.0], 10.0]],
    }

    output = op_cost_components([""], ["Route: [0]\nObjective: 0"], **kwargs)
    names = output["cost_names"]
    row = output["cost_vectors"][0]
    mask = output["cost_masks"][0]

    assert row[names.index("length_overflow")] == 1.0
    assert mask[names.index("length_overflow")] == 1.0


def test_op_prize_surplus_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Route: [0, 1]\nObjective: 10"],
        "instance": [[[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], [0.0, 10.0, 10.0], 9.0]],
    }

    monkeypatch.delenv("CLASP_OP_PRIZE_SURPLUS_CONSTRAINT", raising=False)
    default_output = op_cost_components(prompts, ["Route: [0, 1, 2]\nObjective: 20"], **kwargs)
    assert "prize_surplus_overbudget" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_OP_PRIZE_SURPLUS_CONSTRAINT", "1")
    surplus_output = op_cost_components(prompts, ["Route: [0, 1, 2]\nObjective: 20"], **kwargs)
    names = surplus_output["cost_names"]
    assert "prize_surplus_overbudget" in names
    assert surplus_output["cost_vectors"][0][names.index("length_overflow")] > 0.0
    assert math.isclose(
        surplus_output["cost_vectors"][0][names.index("prize_surplus_overbudget")],
        1.0,
        rel_tol=1e-6,
    )

    feasible_output = op_cost_components(prompts, ["Route: [0, 1]\nObjective: 10"], **kwargs)
    assert feasible_output["cost_vectors"][0][names.index("length_overflow")] == 0.0
    assert feasible_output["cost_vectors"][0][names.index("prize_surplus_overbudget")] == 0.0


def test_op_prize_cap_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Route: [0, 1]\nObjective: 10"],
        "instance": [[[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], [0.0, 10.0, 10.0], 100.0]],
    }

    monkeypatch.delenv("CLASP_OP_PRIZE_CAP_CONSTRAINT", raising=False)
    default_output = op_cost_components(prompts, ["Route: [0, 1, 2]\nObjective: 20"], **kwargs)
    assert "prize_cap_over_ref" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_OP_PRIZE_CAP_CONSTRAINT", "1")
    cap_output = op_cost_components(prompts, ["Route: [0, 1, 2]\nObjective: 20"], **kwargs)
    names = cap_output["cost_names"]
    assert "prize_cap_over_ref" in names
    assert cap_output["cost_vectors"][0][names.index("length_overflow")] == 0.0
    assert math.isclose(
        cap_output["cost_vectors"][0][names.index("prize_cap_over_ref")],
        1.0,
        rel_tol=1e-6,
    )

    on_ref_output = op_cost_components(prompts, ["Route: [0, 1]\nObjective: 10"], **kwargs)
    assert on_ref_output["cost_vectors"][0][names.index("prize_cap_over_ref")] == 0.0


def test_mis_conflict_adjusted_partial_credit_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [2, 3]\nObjective: 2"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1]]}],
        "apply_constraint_gate": True,
        "allow_infeasible_partial_credit": True,
    }
    completion = "Set: [0, 1, 2]\nObjective: 3"

    monkeypatch.delenv("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", raising=False)
    default_reward = mis_objective_reward(prompts, [completion], **kwargs)[0]

    monkeypatch.setenv("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", "1")
    monkeypatch.setenv("CLASP_MIS_CONFLICT_NODE_PENALTY", "2.0")
    adjusted_reward = mis_objective_reward(prompts, [completion], **kwargs)[0]

    assert adjusted_reward < default_reward


def test_mis_conflict_cover_partial_credit_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2, 3]\nObjective: 3"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1], [1, 2]]}],
        "apply_constraint_gate": True,
        "allow_infeasible_partial_credit": True,
    }
    completion = "Set: [0, 1, 2, 3]\nObjective: 4"

    monkeypatch.delenv("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", raising=False)
    monkeypatch.delenv("CLASP_MIS_CONFLICT_COVER_PARTIAL", raising=False)
    default_reward = mis_objective_reward(prompts, [completion], **kwargs)[0]

    monkeypatch.setenv("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", "1")
    monkeypatch.setenv("CLASP_MIS_CONFLICT_NODE_PENALTY", "1.0")
    edge_count_reward = mis_objective_reward(prompts, [completion], **kwargs)[0]

    monkeypatch.delenv("CLASP_MIS_CONFLICT_ADJUSTED_PARTIAL", raising=False)
    monkeypatch.setenv("CLASP_MIS_CONFLICT_COVER_PARTIAL", "1")
    cover_reward = mis_objective_reward(prompts, [completion], **kwargs)[0]

    assert edge_count_reward < cover_reward < default_reward


def test_mis_target_gap_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2, 3]\nObjective: 3"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1]]}],
    }

    monkeypatch.delenv("CLASP_MIS_TARGET_GAP_CONSTRAINT", raising=False)
    default_output = mis_cost_components(prompts, ["Set: [0, 2]\nObjective: 2"], **kwargs)
    assert "size_deficit_over_ref" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_MIS_TARGET_GAP_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_MIS_TARGET_GAP", "0.0104")
    constrained_output = mis_cost_components(prompts, ["Set: [0, 2]\nObjective: 2"], **kwargs)
    names = constrained_output["cost_names"]
    assert "size_deficit_over_ref" in names
    deficit_idx = names.index("size_deficit_over_ref")
    assert constrained_output["cost_masks"][0][deficit_idx] == 1.0
    assert constrained_output["cost_vectors"][0][deficit_idx] > 0.0

    optimal_output = mis_cost_components(prompts, ["Set: [0, 2, 3]\nObjective: 3"], **kwargs)
    assert optimal_output["cost_vectors"][0][deficit_idx] == 0.0

    infeasible_output = mis_cost_components(prompts, ["Set: [0, 1, 2]\nObjective: 3"], **kwargs)
    assert infeasible_output["cost_masks"][0][deficit_idx] == 0.0


def test_mis_size_excess_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2]\nObjective: 2"],
        "instance": [{"num_nodes": 10, "edges": []}],
    }

    monkeypatch.delenv("CLASP_MIS_SIZE_EXCESS_CONSTRAINT", raising=False)
    default_output = mis_cost_components(prompts, ["Set: [0, 2, 3]\nObjective: 3"], **kwargs)
    assert "size_excess_over_ref" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_MIS_SIZE_EXCESS_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_MIS_SIZE_EXCESS_CONSTRAINT_SCALE", "0.5")
    monkeypatch.setenv("CLASP_MIS_SIZE_EXCESS_MAX_REF_RATIO", "0.25")
    constrained_output = mis_cost_components(prompts, ["Set: [0, 2, 3]\nObjective: 3"], **kwargs)
    names = constrained_output["cost_names"]
    assert "size_excess_over_ref" in names
    excess_idx = names.index("size_excess_over_ref")
    assert constrained_output["cost_masks"][0][excess_idx] == 1.0
    assert math.isclose(constrained_output["cost_vectors"][0][excess_idx], 0.5, rel_tol=1e-6)

    on_ref_output = mis_cost_components(prompts, ["Set: [0, 2]\nObjective: 2"], **kwargs)
    assert on_ref_output["cost_vectors"][0][excess_idx] == 0.0

    high_ratio_kwargs = dict(kwargs)
    high_ratio_kwargs["instance"] = [{"num_nodes": 4, "edges": []}]
    gated_output = mis_cost_components(
        prompts,
        ["Set: [0, 2, 3]\nObjective: 3"],
        **high_ratio_kwargs,
    )
    assert gated_output["cost_masks"][0][excess_idx] == 0.0
    assert gated_output["cost_vectors"][0][excess_idx] == 0.0


def test_mis_repair_target_gap_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2, 3]\nObjective: 3"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1], [1, 2]]}],
    }

    monkeypatch.delenv("CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT", raising=False)
    default_output = mis_cost_components(prompts, ["Set: [0, 1, 2]\nObjective: 3"], **kwargs)
    assert "repair_size_deficit_over_ref" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_MIS_REPAIR_TARGET_GAP", "0.0104")
    constrained_output = mis_cost_components(prompts, ["Set: [0, 1, 2]\nObjective: 3"], **kwargs)
    names = constrained_output["cost_names"]
    assert "repair_size_deficit_over_ref" in names
    repair_idx = names.index("repair_size_deficit_over_ref")
    assert constrained_output["cost_masks"][0][repair_idx] == 1.0
    assert constrained_output["cost_vectors"][0][repair_idx] > 0.0

    repairable_output = mis_cost_components(prompts, ["Set: [0, 1, 2, 3]\nObjective: 4"], **kwargs)
    assert repairable_output["cost_masks"][0][repair_idx] == 1.0
    assert repairable_output["cost_vectors"][0][repair_idx] == 0.0

    optimal_output = mis_cost_components(prompts, ["Set: [0, 2, 3]\nObjective: 3"], **kwargs)
    assert optimal_output["cost_vectors"][0][repair_idx] == 0.0


def test_mis_conflict_node_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2, 3]\nObjective: 3"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1], [1, 2]]}],
    }

    monkeypatch.delenv("CLASP_MIS_CONFLICT_NODE_CONSTRAINT", raising=False)
    default_output = mis_cost_components(prompts, ["Set: [0, 1, 2]\nObjective: 3"], **kwargs)
    assert "conflict_node_fraction" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_MIS_CONFLICT_NODE_CONSTRAINT", "1")
    constrained_output = mis_cost_components(prompts, ["Set: [0, 1, 2]\nObjective: 3"], **kwargs)
    names = constrained_output["cost_names"]
    assert "conflict_node_fraction" in names
    node_idx = names.index("conflict_node_fraction")
    assert constrained_output["cost_masks"][0][node_idx] == 1.0
    assert math.isclose(constrained_output["cost_vectors"][0][node_idx], 1.0, rel_tol=1e-6)

    feasible_output = mis_cost_components(prompts, ["Set: [0, 2]\nObjective: 2"], **kwargs)
    assert feasible_output["cost_vectors"][0][node_idx] == 0.0


def test_mis_maximality_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "answer": ["Set: [0, 2, 3]\nObjective: 3"],
        "instance": [{"num_nodes": 4, "edges": [[0, 1], [1, 2]]}],
    }

    monkeypatch.delenv("CLASP_MIS_MAXIMALITY_CONSTRAINT", raising=False)
    default_output = mis_cost_components(prompts, ["Set: [0]\nObjective: 1"], **kwargs)
    assert "addable_vertex_fraction" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_MIS_MAXIMALITY_CONSTRAINT", "1")
    constrained_output = mis_cost_components(prompts, ["Set: [0]\nObjective: 1"], **kwargs)
    names = constrained_output["cost_names"]
    assert "addable_vertex_fraction" in names
    addable_idx = names.index("addable_vertex_fraction")
    assert constrained_output["cost_masks"][0][addable_idx] == 1.0
    assert math.isclose(constrained_output["cost_vectors"][0][addable_idx], 2.0 / 3.0, rel_tol=1e-6)

    maximal_output = mis_cost_components(prompts, ["Set: [0, 2, 3]\nObjective: 3"], **kwargs)
    assert maximal_output["cost_vectors"][0][addable_idx] == 0.0

    infeasible_output = mis_cost_components(prompts, ["Set: [0, 1]\nObjective: 2"], **kwargs)
    assert infeasible_output["cost_masks"][0][addable_idx] == 0.0

    monkeypatch.setenv("CLASP_MIS_MAXIMALITY_INDEPENDENT_ONLY", "0")
    near_independent_output = mis_cost_components(prompts, ["Set: [0, 1]\nObjective: 2"], **kwargs)
    assert near_independent_output["cost_masks"][0][addable_idx] == 1.0


def test_mis_target_gap_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    target_limits, target_weights = build_component_profile("mis", "co_mis_target_gap")
    op_limits, op_weights = build_component_profile("op", "co_mis_target_gap")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert target_limits["edge_conflict"] == 0.0
    assert target_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert target_limits["size_deficit_over_ref"] == 0.0
    assert target_weights["size_deficit_over_ref"] > default_weights["size_deficit_over_ref"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_conflict_node_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    node_limits, node_weights = build_component_profile("mis", "co_mis_conflict_nodes")
    op_limits, op_weights = build_component_profile("op", "co_mis_conflict_nodes")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert node_limits["edge_conflict"] == 0.0
    assert node_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert node_limits["conflict_node_fraction"] == 0.0
    assert node_weights["conflict_node_fraction"] > default_weights["conflict_node_fraction"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_conflict_target_gap_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    combined_limits, combined_weights = build_component_profile("mis", "co_mis_conflict_target_gap")
    op_limits, op_weights = build_component_profile("op", "co_mis_conflict_target_gap")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert combined_limits["edge_conflict"] == 0.0
    assert combined_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert combined_limits["conflict_node_fraction"] == 0.0
    assert combined_weights["conflict_node_fraction"] > default_weights["conflict_node_fraction"]
    assert combined_limits["size_deficit_over_ref"] == 0.0
    assert combined_weights["size_deficit_over_ref"] > default_weights["size_deficit_over_ref"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_conflict_target_gap_sizecap_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    cap_limits, cap_weights = build_component_profile("mis", "co_mis_conflict_target_gap_sizecap")
    op_limits, op_weights = build_component_profile("op", "co_mis_conflict_target_gap_sizecap")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert cap_limits["edge_conflict"] == 0.0
    assert cap_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert cap_limits["conflict_node_fraction"] == 0.0
    assert cap_weights["conflict_node_fraction"] > default_weights["conflict_node_fraction"]
    assert cap_limits["size_deficit_over_ref"] == 0.0
    assert cap_weights["size_deficit_over_ref"] > default_weights["size_deficit_over_ref"]
    assert cap_limits["size_excess_over_ref"] == 0.0
    assert cap_weights["size_excess_over_ref"] > default_weights["size_excess_over_ref"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_conflict_target_gap_maximality_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    max_limits, max_weights = build_component_profile("mis", "co_mis_conflict_target_gap_maximality")
    op_limits, op_weights = build_component_profile("op", "co_mis_conflict_target_gap_maximality")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert max_limits["edge_conflict"] == 0.0
    assert max_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert max_limits["conflict_node_fraction"] == 0.0
    assert max_weights["conflict_node_fraction"] > default_weights["conflict_node_fraction"]
    assert max_limits["size_deficit_over_ref"] == 0.0
    assert max_weights["size_deficit_over_ref"] > default_weights["size_deficit_over_ref"]
    assert max_limits["addable_vertex_fraction"] == 0.0
    assert max_weights["addable_vertex_fraction"] > default_weights["addable_vertex_fraction"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_repair_target_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    repair_limits, repair_weights = build_component_profile("mis", "co_mis_repair_target")
    op_limits, op_weights = build_component_profile("op", "co_mis_repair_target")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert repair_limits["edge_conflict"] == 0.0
    assert repair_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert repair_limits["repair_size_deficit_over_ref"] == 0.0
    assert repair_weights["repair_size_deficit_over_ref"] > default_weights["repair_size_deficit_over_ref"]
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_mis_conflict_repair_target_gap_profile_is_mis_only():
    default_limits, default_weights = build_component_profile("mis", "co_default")
    profile_limits, profile_weights = build_component_profile(
        "mis",
        "co_mis_conflict_repair_target_gap",
    )
    op_limits, op_weights = build_component_profile("op", "co_mis_conflict_repair_target_gap")
    default_op_limits, default_op_weights = build_component_profile("op", "co_default")

    assert profile_limits["edge_conflict"] == 0.0
    assert profile_weights["edge_conflict"] > default_weights["edge_conflict"]
    assert profile_limits["conflict_node_fraction"] == 0.0
    assert profile_weights["conflict_node_fraction"] > default_weights["conflict_node_fraction"]
    assert profile_limits["size_deficit_over_ref"] == 0.0
    assert profile_weights["size_deficit_over_ref"] > default_weights["size_deficit_over_ref"]
    assert profile_limits["repair_size_deficit_over_ref"] == 0.0
    assert (
        profile_weights["repair_size_deficit_over_ref"]
        > default_weights["repair_size_deficit_over_ref"]
    )
    assert op_limits == default_op_limits
    assert op_weights == default_op_weights


def test_jssp_target_gap_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    instance = [[[0, 3, 1, 2], [1, 2, 0, 1]]]
    kwargs = {
        "input": [""],
        "answer": ["Schedule: [[0, 1], [1, 0]]\nMakespan: 5"],
        "instance": instance,
    }

    slow_completion = "Schedule: [[0, 1], [0, 1]]\nMakespan: 8"
    monkeypatch.delenv("CLASP_JSSP_TARGET_GAP_CONSTRAINT", raising=False)
    default_output = jssp_cost_components_hybrid(prompts, [slow_completion], **kwargs)
    assert "makespan_gap_over_ref" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_JSSP_TARGET_GAP_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_JSSP_TARGET_GAP", "0.082")
    constrained_output = jssp_cost_components_hybrid(prompts, [slow_completion], **kwargs)
    names = constrained_output["cost_names"]
    assert "makespan_gap_over_ref" in names
    gap_idx = names.index("makespan_gap_over_ref")
    assert constrained_output["cost_masks"][0][gap_idx] == 1.0
    assert constrained_output["cost_vectors"][0][gap_idx] > 0.0

    optimal_output = jssp_cost_components_hybrid(
        prompts,
        ["Schedule: [[0, 1], [1, 0]]\nMakespan: 5"],
        **kwargs,
    )
    assert optimal_output["cost_vectors"][0][gap_idx] == 0.0

    invalid_output = jssp_cost_components_hybrid(
        prompts,
        ["Schedule: [[0, 0], [1, 0]]\nMakespan: 5"],
        **kwargs,
    )
    assert invalid_output["cost_masks"][0][gap_idx] == 0.0


def test_jssp_cycle_fraction_cost_components_are_opt_in(monkeypatch):
    prompts = [""]
    instance = [[[0, 1, 1, 1], [1, 1, 0, 1]]]
    kwargs = {
        "input": [""],
        "answer": ["Schedule: [[0, 1], [0, 1]]\nMakespan: 4"],
        "instance": instance,
    }
    cyclic_completion = "Schedule: [[1, 0], [0, 1]]\nMakespan: 4"

    monkeypatch.delenv("CLASP_JSSP_CYCLE_FRACTION_CONSTRAINT", raising=False)
    monkeypatch.delenv("CLASP_JSSP_PROCESSED_DEFICIT_CONSTRAINT", raising=False)
    default_output = jssp_cost_components_hybrid(prompts, [cyclic_completion], **kwargs)
    assert "cycle_node_fraction" not in default_output["cost_names"]
    assert "processed_node_deficit" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_JSSP_CYCLE_FRACTION_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_JSSP_PROCESSED_DEFICIT_CONSTRAINT", "1")
    constrained_output = jssp_cost_components_hybrid(prompts, [cyclic_completion], **kwargs)
    names = constrained_output["cost_names"]
    cycle_idx = names.index("cycle_node_fraction")
    processed_idx = names.index("processed_node_deficit")
    invalid_idx = names.index("schedule_invalid")
    assert constrained_output["cost_vectors"][0][invalid_idx] == 1.0
    assert constrained_output["cost_vectors"][0][cycle_idx] > 0.0
    assert constrained_output["cost_vectors"][0][processed_idx] > 0.0
    assert constrained_output["cost_masks"][0][cycle_idx] == 1.0
    assert constrained_output["cost_masks"][0][processed_idx] == 1.0

    valid_output = jssp_cost_components_hybrid(
        prompts,
        ["Schedule: [[0, 1], [0, 1]]\nMakespan: 4"],
        **kwargs,
    )
    assert valid_output["cost_vectors"][0][cycle_idx] == 0.0
    assert valid_output["cost_vectors"][0][processed_idx] == 0.0


def test_jssp_parse_bad_token_cost_component_is_opt_in(monkeypatch):
    prompts = [""]
    kwargs = {
        "input": [""],
        "instance": [[[0, 1, 1, 1], [1, 1, 0, 1]]],
    }
    malformed_completion = "Schedule: [[0, 1, |], [1, 0, |]]\nMakespan: 4"

    monkeypatch.delenv("CLASP_JSSP_PARSE_BAD_TOKEN_CONSTRAINT", raising=False)
    default_output = jssp_cost_components_hybrid(prompts, [malformed_completion], **kwargs)
    assert "parse_bad_token" not in default_output["cost_names"]

    monkeypatch.setenv("CLASP_JSSP_PARSE_BAD_TOKEN_CONSTRAINT", "1")
    constrained_output = jssp_cost_components_hybrid(prompts, [malformed_completion], **kwargs)
    names = constrained_output["cost_names"]
    bad_token_idx = names.index("parse_bad_token")
    parse_idx = names.index("parse_failure")
    assert constrained_output["cost_vectors"][0][parse_idx] == 1.0
    assert constrained_output["cost_vectors"][0][bad_token_idx] == 1.0
    assert constrained_output["cost_masks"][0][bad_token_idx] == 1.0


def test_jssp_target_gap_profile_is_jssp_only():
    default_limits, default_weights = build_component_profile("jssp", "co_default")
    profile_limits, profile_weights = build_component_profile("jssp", "co_jssp_valid_target_gap")
    mis_limits, mis_weights = build_component_profile("mis", "co_jssp_valid_target_gap")
    default_mis_limits, default_mis_weights = build_component_profile("mis", "co_default")

    assert profile_limits["schedule_invalid"] == 0.0
    assert profile_weights["schedule_invalid"] > default_weights["schedule_invalid"]
    assert profile_limits["cycle_node_fraction"] == 0.0
    assert profile_weights["cycle_node_fraction"] > default_weights["cycle_node_fraction"]
    assert profile_limits["processed_node_deficit"] == 0.0
    assert profile_weights["processed_node_deficit"] > default_weights["processed_node_deficit"]
    assert profile_limits["missing_job"] == 0.0
    assert profile_weights["missing_job"] > default_weights["missing_job"]
    assert profile_limits["makespan_gap_over_ref"] == 0.0
    assert profile_weights["makespan_gap_over_ref"] > default_weights["makespan_gap_over_ref"]
    assert mis_limits == default_mis_limits
    assert mis_weights == default_mis_weights


def test_mis_prompt_node_degree_hints_are_opt_in(monkeypatch):
    input_text = "Edges: [(0,1),(1,2)]\nN0:[1,#2];N1:[0,#1];N2:[1,#2]"
    instance = {"num_nodes": 3, "edges": [[0, 1], [1, 2]]}

    monkeypatch.delenv("CLASP_MIS_PROMPT_NODE_DEGREES", raising=False)
    assert augment_input_for_task("mis", input_text, instance) == input_text

    monkeypatch.setenv("CLASP_MIS_PROMPT_NODE_DEGREES", "1")
    monkeypatch.setenv("CLASP_MIS_PROMPT_LOW_DEGREE_K", "2")
    augmented = augment_input_for_task("mis", input_text, instance)

    assert "MIS node-degree hints:" in augmented
    assert "Degrees (node:degree): 0:1, 1:2, 2:1" in augmented
    assert "Lowest-degree nodes to consider first: [0(deg=1), 2(deg=1)]" in augmented
    assert augment_input_for_task("mis", augmented, instance) == augmented


def test_mis_degree_relabel_transform_is_opt_in(monkeypatch):
    instruction = "Find a maximum independent set."
    input_text = "Edges: [(0,1),(0,2),(1,2)]"
    output_text = "Set: [0, 3], Objective: 2"
    instance = {"num_nodes": 4, "edges": [[0, 1], [0, 2], [1, 2]], "distribution": "toy"}

    monkeypatch.delenv("CLASP_MIS_RELABEL_ORDER", raising=False)
    default = transform_example_for_task("mis", instruction, input_text, output_text, instance)
    assert default["input"] == input_text
    assert default["output"] == output_text
    assert default["instance_json"] == instance

    monkeypatch.setenv("CLASP_MIS_RELABEL_ORDER", "degree_asc")
    transformed = transform_example_for_task("mis", instruction, input_text, output_text, instance)

    assert transformed["input"] == "Edges: [(1,2),(1,3),(2,3)]"
    assert transformed["output"] == "Set: [0, 1], Objective: 2"
    assert transformed["instance_json"]["num_nodes"] == 4
    assert transformed["instance_json"]["edges"] == [[1, 2], [1, 3], [2, 3]]
    assert transformed["instance_json"]["distribution"] == "toy"


def test_tsp_route_only_transform_is_opt_in(monkeypatch):
    instruction = (
        "Solve the Traveling Salesman Problem. Provide the solution in the following format:\n"
        "1. Route: List the nodes.\n2. Objective: The objective value."
    )
    input_text = "Node 0, coordinates: [0, 0];Node 1, coordinates: [1, 0]."
    output_text = "Route: [0, 1, 0], Objective: 2.000"
    instance = [[0, 0], [1, 0]]

    monkeypatch.delenv("CLASP_TSP_ROUTE_ONLY_OUTPUT", raising=False)
    default = transform_example_for_task("tsp", instruction, input_text, output_text, instance)
    assert default["output"] == output_text
    assert "Route-only output requirement" not in default["instruction"]

    monkeypatch.setenv("CLASP_TSP_ROUTE_ONLY_OUTPUT", "1")
    transformed = transform_example_for_task("tsp", instruction, input_text, output_text, instance)

    assert transformed["output"] == "Route: [0, 1, 0]"
    assert "Route-only output requirement" in transformed["instruction"]
    assert "Do not include an Objective field" in transformed["instruction"]


def test_tsp_route_only_structure_reward_does_not_require_objective(monkeypatch):
    completion = "Route: [0, 1, 0]"

    monkeypatch.delenv("CLASP_TSP_ROUTE_ONLY_OUTPUT", raising=False)
    assert tsp_structure_reward([], [completion]) == [pytest.approx(2.0 / 3.0)]

    monkeypatch.setenv("CLASP_TSP_ROUTE_ONLY_OUTPUT", "1")
    assert tsp_structure_reward([], [completion]) == [1.0]


# ---------------------------------------------------------------------------
# Centered primal-gap coefficient
# ---------------------------------------------------------------------------


def test_pareto_advantage_nonzero_on_all_infeasible_group():
    # Single group of K=4 infeasible samples with varying violations.
    r, g, f = _build_group(
        primal=[1.0, 1.1, 1.2, 0.9],
        violation=[0.8, 0.5, 1.2, 0.2],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    adv, stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    # Advantage must NOT collapse to constant (the exact pathology CLASP fixes).
    assert adv.std(unbiased=False).item() > 0.1
    assert stats["clasp/all_infeasible_group_rate"] == 1.0
    # The lowest-violation sample is preferred. This ordering comes from the
    # centered violation component, not from the diagnostic anchor: the
    # group-constant anchor cancels exactly below.
    assert int(adv.argmax().item()) == 3


def test_legacy_pareto_anchor_cancels_to_anchor_free_normal_form(monkeypatch):
    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "0")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_MIX", "0")
    monkeypatch.setenv("CLASP_PRIMAL_SCORE_GAP_PENALTY", "0")
    r, g, f = _build_group(
        primal=[0.4, 1.7, -0.2, 0.9],
        violation=[1.1, 0.3, 0.8, 0.05],
        feasible=[0.0, 1.0, 0.0, 1.0],
    )
    tau = 3.25
    center = 0.4
    eps = 1e-4

    actual, _ = compute_pareto_anchored_advantages(
        r,
        g,
        f,
        num_generations=4,
        tau=tau,
        feasibility_center=center,
        eps_std=eps,
    )

    r_group = r.reshape(1, 4).to(torch.float32)
    g_group = g.reshape(1, 4).to(torch.float32)
    f_group = f.reshape(1, 4).to(torch.float32)
    alpha = torch.sigmoid(tau * (f_group.mean(dim=1, keepdim=True) - center))
    expected = alpha * (
        (r_group - r_group.mean(dim=1, keepdim=True))
        / (r_group.std(dim=1, keepdim=True, unbiased=False) + eps)
    ) + (1.0 - alpha) * (
        (g_group.mean(dim=1, keepdim=True) - g_group)
        / (g_group.std(dim=1, keepdim=True, unbiased=False) + eps)
    )

    torch.testing.assert_close(actual, expected.reshape(-1), rtol=1e-6, atol=1e-6)


def test_alpha_saturates_on_fully_feasible_group():
    r, g, f = _build_group(
        primal=[1.0, 1.2, 1.1, 0.9],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )
    _, stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    # feas_rate = 1 -> alpha = sigmoid(2.0) ~= 0.881
    assert 0.85 < stats["clasp/mean_alpha"] < 0.92


def test_alpha_collapses_on_fully_infeasible_group():
    r, g, f = _build_group(
        primal=[1.0, 1.2, 1.1, 0.9],
        violation=[0.8, 0.5, 1.2, 0.2],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    _, stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    # feas_rate = 0 -> alpha = sigmoid(-2.0) ~= 0.119
    assert 0.08 < stats["clasp/mean_alpha"] < 0.15


def test_pareto_advantage_is_group_centred():
    r, g, f = _build_group(
        primal=[1.0, 1.3, 0.9, 1.1, 0.7, 1.2],
        violation=[0.1, 0.0, 0.3, 0.0, 0.5, 0.0],
        feasible=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    adv, _ = compute_pareto_anchored_advantages(
        r, g, f, num_generations=6, tau=4.0
    )
    for group in adv.view(-1, 6):
        assert abs(group.mean().item()) < 1e-5


def test_pareto_advantage_rank_mix_is_opt_in(monkeypatch):
    r, g, f = _build_group(
        primal=[1.0, 1.3, 1.1, 0.9],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )

    monkeypatch.delenv("CLASP_PRIMAL_RANK_MIX", raising=False)
    default_adv, default_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    assert default_stats["clasp/primal_rank_mix"] == 0.0

    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "1.0")
    rank_adv, rank_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    assert rank_stats["clasp/primal_rank_mix"] == 1.0
    assert int(rank_adv.argmax().item()) == 1
    assert not torch.allclose(default_adv, rank_adv)
    assert abs(rank_adv.view(1, 4).mean().item()) < 1e-5


def test_pareto_advantage_elite_mix_is_opt_in(monkeypatch):
    r, g, f = _build_group(
        primal=[1.0, 1.3, 1.1, 0.9],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )

    monkeypatch.delenv("CLASP_PRIMAL_RANK_MIX", raising=False)
    monkeypatch.delenv("CLASP_PRIMAL_ELITE_MIX", raising=False)
    default_adv, default_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    assert default_stats["clasp/primal_elite_mix"] == 0.0

    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "0.5")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_MIX", "0.25")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_FRACTION", "0.25")
    elite_adv, elite_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    assert elite_stats["clasp/primal_rank_mix"] == 0.5
    assert elite_stats["clasp/primal_elite_mix"] == 0.25
    assert elite_stats["clasp/primal_standard_mix"] == 0.25
    assert elite_stats["clasp/primal_elite_count"] == 1.0
    assert int(elite_adv.argmax().item()) == 1
    assert not torch.allclose(default_adv, elite_adv)
    assert abs(elite_adv.view(1, 4).mean().item()) < 1e-5


def test_pareto_advantage_rank_elite_can_use_raw_gap_penalty(monkeypatch):
    r, g, f = _build_group(
        primal=[5.0, 4.8, 4.7, 4.6],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )
    raw_gap = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "1.0")
    monkeypatch.delenv("CLASP_PRIMAL_ELITE_MIX", raising=False)
    monkeypatch.delenv("CLASP_PRIMAL_SCORE_GAP_PENALTY", raising=False)
    default_adv, default_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0, rank_violation_scores=raw_gap
    )
    assert int(default_adv.argmax().item()) == 0
    assert default_stats["clasp/primal_score_gap_penalty"] == 0.0

    monkeypatch.setenv("CLASP_PRIMAL_SCORE_GAP_PENALTY", "1.0")
    penalized_adv, penalized_stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0, rank_violation_scores=raw_gap
    )
    assert int(penalized_adv.argmax().item()) == 1
    assert penalized_stats["clasp/primal_score_gap_penalty"] == 1.0
    assert math.isclose(penalized_stats["clasp/primal_rank_gap_mean"], 0.25, rel_tol=1e-6)
    assert abs(penalized_adv.view(1, 4).mean().item()) < 1e-5


def test_hard_group_weighted_advantages_are_default_off_by_zero_scale():
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    raw_gap = torch.tensor([0.4, 0.2, 0.0, 0.6], dtype=torch.float32)

    weighted, stats = compute_hard_group_weighted_advantages(
        advantages,
        raw_gap,
        num_generations=2,
        scale=0.0,
        threshold=0.1,
        max_weight=2.0,
    )

    assert torch.allclose(weighted, advantages)
    assert stats["clasp/hard_group_weight_mean"] == 1.0
    assert stats["clasp/hard_group_active_rate"] == 0.0


def test_hard_group_weighted_advantages_boost_only_groups_with_positive_best_gap():
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    raw_gap = torch.tensor([0.4, 0.2, 0.0, 0.6], dtype=torch.float32)

    weighted, stats = compute_hard_group_weighted_advantages(
        advantages,
        raw_gap,
        num_generations=2,
        scale=1.0,
        threshold=0.1,
        max_weight=2.0,
    )

    expected = torch.tensor([1.1, -1.1, 0.5, -0.5], dtype=torch.float32)
    assert torch.allclose(weighted, expected)
    assert math.isclose(stats["clasp/hard_group_weight_mean"], 1.05, rel_tol=1e-6)
    assert math.isclose(stats["clasp/hard_group_active_rate"], 0.5, rel_tol=1e-6)
    assert math.isclose(stats["clasp/hard_group_min_raw_gap_mean"], 0.1, rel_tol=1e-6)


def test_multitask_manager_propagates_raw_positive_gap_scores():
    class DummyWrapper:
        def __call__(self, prompts, completions, **kwargs):
            assert len(prompts) == 2
            return [0.1, 0.2]

        def get_last_components(self):
            return {
                "primal_scores": [1.0, 2.0],
                "weighted_gap_scores": [0.0, 0.0],
                "positive_weighted_gap_scores": [0.0, 0.0],
                "raw_positive_gap_scores": [0.25, 0.0],
                "constraint_violation_scores": [0.0, 0.0],
                "feasibility_flags": [1.0, 1.0],
                "constraint_scores": [1.0, 1.0],
                "combined_scores": [1.0, 2.0],
            }

    manager = MultiTaskPrimalDualManager({"tsp": DummyWrapper()})
    rewards = manager(
        ["p0", "p1"],
        ["c0", "c1"],
        task_type=["tsp", "tsp"],
    )
    assert rewards == [0.1, 0.2]
    assert manager.get_last_components()["raw_positive_gap_scores"] == [0.25, 0.0]
    assert manager.get_last_components()["dual_aware_recovery_gap_scores"] == [0.25, 0.0]


def test_pareto_advantage_rank_elite_weights_renormalize(monkeypatch):
    r, g, f = _build_group(
        primal=[1.0, 1.3, 1.1, 0.9],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )

    monkeypatch.setenv("CLASP_PRIMAL_RANK_MIX", "0.8")
    monkeypatch.setenv("CLASP_PRIMAL_ELITE_MIX", "0.4")
    _, stats = compute_pareto_anchored_advantages(
        r, g, f, num_generations=4, tau=4.0
    )
    assert math.isclose(stats["clasp/primal_rank_mix"], 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(stats["clasp/primal_elite_mix"], 1.0 / 3.0, rel_tol=1e-6)
    assert stats["clasp/primal_standard_mix"] == 0.0


def test_pareto_advantage_rejects_mismatched_sizes():
    r = torch.zeros(4)
    g = torch.zeros(5)
    f = torch.zeros(4)
    with pytest.raises(ValueError):
        compute_pareto_anchored_advantages(r, g, f, num_generations=4)


def test_pareto_advantage_rejects_mismatched_rank_violation_sizes():
    r = torch.zeros(4)
    g = torch.zeros(4)
    f = torch.zeros(4)
    raw_gap = torch.zeros(5)
    with pytest.raises(ValueError):
        compute_pareto_anchored_advantages(
            r,
            g,
            f,
            num_generations=4,
            rank_violation_scores=raw_gap,
        )


def test_pareto_advantage_rejects_bad_group_size():
    r = torch.zeros(5)
    g = torch.zeros(5)
    f = torch.zeros(5)
    with pytest.raises(ValueError):
        compute_pareto_anchored_advantages(r, g, f, num_generations=4)


def test_feasibility_gated_mixed_group_feasible_dominates_infeasible(monkeypatch):
    # The best-looking objective is infeasible, but all feasible completions
    # must still outrank all infeasible completions.
    r, g, f = _build_group(
        primal=[10.0, 1.0, 2.0, 9.0],
        violation=[0.8, 0.0, 0.3, 0.0],
        feasible=[0.0, 1.0, 0.0, 1.0],
    )
    monkeypatch.setenv("CLASP_FEASIBILITY_GATE_MARGIN", "2.0")
    adv, stats = compute_feasibility_gated_advantages(r, g, f, num_generations=4)

    feasible_adv = adv[f > 0.5]
    infeasible_adv = adv[f <= 0.5]
    assert feasible_adv.min().item() > infeasible_adv.max().item()
    assert stats["clasp/feasibility_gated"] == 1.0
    assert stats["clasp/mixed_group_rate"] == 1.0
    assert stats["clasp/feasibility_gate_mixed_dominance_margin"] > 0.0


@pytest.mark.parametrize("num_generations", (2, 3, 8))
def test_feasibility_gated_exhaustively_dominates_for_every_mixed_mask(
    monkeypatch,
    num_generations: int,
):
    frozen_environment = {
        "CLASP_FEASIBILITY_GATE_MARGIN": "2.0",
        "CLASP_FEASIBILITY_GATE_FEASIBLE_RANK_SCALE": "0.5",
        "CLASP_FEASIBILITY_GATE_INFEASIBLE_RANK_SCALE": "0.5",
        "CLASP_FEASIBILITY_GATE_ALL_INFEASIBLE_WEIGHT": "1.5",
        "CLASP_FEASIBILITY_GATE_MIXED_WEIGHT": "1.25",
        "CLASP_FEASIBILITY_GATE_ALL_FEASIBLE_WEIGHT": "1.0",
        "CLASP_FEASIBILITY_GATE_FEASIBLE_Z_MIX": "0.5",
        "CLASP_PRIMAL_SCORE_GAP_PENALTY": "0.0",
    }
    for key, value in frozen_environment.items():
        monkeypatch.setenv(key, value)

    generator = torch.Generator().manual_seed(20_000 + num_generations)
    for bit_mask in range(1, (1 << num_generations) - 1):
        feasible = torch.tensor(
            [
                float(bool(bit_mask & (1 << index)))
                for index in range(num_generations)
            ],
            dtype=torch.float32,
        )
        primal = torch.randn(num_generations, generator=generator)
        violation = torch.rand(num_generations, generator=generator)

        advantages, stats = compute_feasibility_gated_advantages(
            primal,
            violation,
            feasible,
            num_generations=num_generations,
        )

        observed_margin = (
            advantages[feasible > 0.5].min()
            - advantages[feasible <= 0.5].max()
        )
        assert observed_margin.item() >= 3.75 - 1e-5
        assert stats["clasp/feasibility_gate_mixed_dominance_margin"] > 0.0

        changed_primal = primal.clone()
        changed_primal[feasible <= 0.5] = torch.randn(
            int((feasible <= 0.5).sum().item()),
            generator=generator,
        ) * 1_000.0
        changed_advantages, _ = compute_feasibility_gated_advantages(
            changed_primal,
            violation,
            feasible,
            num_generations=num_generations,
        )
        assert torch.allclose(advantages, changed_advantages)


def test_feasibility_gated_all_infeasible_ignores_objective():
    r, g, f = _build_group(
        primal=[100.0, 1.0, 2.0, 3.0],
        violation=[0.9, 0.2, 0.4, 0.6],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    adv, stats = compute_feasibility_gated_advantages(r, g, f, num_generations=4)

    assert int(adv.argmax().item()) == 1
    assert stats["clasp/all_infeasible_group_rate"] == 1.0
    assert stats["clasp/fully_feasible_group_rate"] == 0.0
    assert abs(adv.view(1, 4).mean().item()) < 1e-5


def test_feasibility_gated_all_feasible_optimizes_objective():
    r, g, f = _build_group(
        primal=[1.0, 4.0, 2.0, 3.0],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )
    adv, stats = compute_feasibility_gated_advantages(r, g, f, num_generations=4)

    assert int(adv.argmax().item()) == 1
    assert stats["clasp/fully_feasible_group_rate"] == 1.0
    assert stats["clasp/all_infeasible_group_rate"] == 0.0
    assert abs(adv.view(1, 4).mean().item()) < 1e-5


def test_feasibility_gated_rejects_mismatched_sizes():
    r = torch.zeros(4)
    g = torch.zeros(5)
    f = torch.zeros(4)
    with pytest.raises(ValueError):
        compute_feasibility_gated_advantages(r, g, f, num_generations=4)


def test_lexicographic_clasp_mixed_group_guarantees_feasible_dominance():
    r, learned_g, f = _build_group(
        primal=[100.0, 1.0, 2.0, 90.0],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[0.0, 1.0, 0.0, 1.0],
    )
    raw_g = torch.tensor([0.1, 0.0, 0.9, 0.0], dtype=torch.float32)

    adv, stats = compute_lexicographic_clasp_advantages(
        r,
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=raw_g,
    )

    assert adv[f > 0.5].min() > adv[f <= 0.5].max()
    assert adv[3] > adv[1]
    assert adv[0] > adv[2]
    assert stats["clasp/lexicographic_gate"] == 1.0
    assert stats["clasp/lexicographic_raw_fallback_group_rate"] == 1.0
    assert stats["clasp/lexicographic_mixed_dominance_margin"] > 0.0
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)


@pytest.mark.parametrize("num_generations", (2, 3, 8))
def test_lexicographic_clasp_exhaustively_dominates_for_every_mixed_mask(
    num_generations: int,
):
    generator = torch.Generator().manual_seed(10_000 + num_generations)
    for bit_mask in range(1, (1 << num_generations) - 1):
        feasible = torch.tensor(
            [
                float(bool(bit_mask & (1 << index)))
                for index in range(num_generations)
            ],
            dtype=torch.float32,
        )
        primal = torch.randn(num_generations, generator=generator)
        # Quantization deliberately creates both full and partial learned-gap
        # ties, exercising the raw-violation fallback as well as the learned
        # controller ordering.
        learned_gap = (
            torch.randint(
                0,
                4,
                (num_generations,),
                generator=generator,
            ).to(torch.float32)
            / 10.0
        )
        raw_gap = (
            torch.randint(
                0,
                8,
                (num_generations,),
                generator=generator,
            ).to(torch.float32)
            / 10.0
        )

        advantages, _ = compute_lexicographic_clasp_advantages(
            primal,
            learned_gap,
            feasible,
            num_generations=num_generations,
            raw_violation_scores=raw_gap,
        )

        assert (
            advantages[feasible > 0.5].min()
            > advantages[feasible <= 0.5].max()
        )

        changed_primal = primal.clone()
        changed_primal[feasible <= 0.5] = torch.randn(
            int((feasible <= 0.5).sum().item()),
            generator=generator,
        ) * 1_000.0
        changed_advantages, _ = compute_lexicographic_clasp_advantages(
            changed_primal,
            learned_gap,
            feasible,
            num_generations=num_generations,
            raw_violation_scores=raw_gap,
        )
        assert torch.allclose(advantages, changed_advantages)


def test_lexicographic_clasp_all_infeasible_uses_raw_cold_start_fallback():
    r, learned_g, f = _build_group(
        primal=[100.0, 1.0, 2.0, 3.0],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    raw_g = torch.tensor([0.9, 0.2, 0.4, 0.6], dtype=torch.float32)

    adv, stats = compute_lexicographic_clasp_advantages(
        r,
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=raw_g,
    )

    assert int(adv.argmax().item()) == 1
    assert int(adv.argmin().item()) == 0
    assert stats["clasp/lexicographic_all_infeasible_group_rate"] == 1.0
    assert stats["clasp/lexicographic_raw_fallback_group_rate"] == 1.0

    changed_objective, _ = compute_lexicographic_clasp_advantages(
        torch.tensor([-999.0, 500.0, 400.0, 300.0]),
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=raw_g,
    )
    assert torch.allclose(adv, changed_objective)


def test_lexicographic_clasp_prefers_learned_controller_gap_when_informative():
    r, learned_g, f = _build_group(
        primal=[0.0, 0.0, 0.0, 0.0],
        violation=[0.4, 0.1, 0.3, 0.2],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    # Raw gaps deliberately prefer the opposite ordering.
    raw_g = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.float32)

    adv, stats = compute_lexicographic_clasp_advantages(
        r,
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=raw_g,
    )

    assert int(adv.argmax().item()) == 1
    assert int(adv.argmin().item()) == 0
    assert stats["clasp/lexicographic_raw_fallback_group_rate"] == 0.0


def test_lexicographic_clasp_uses_raw_gap_for_partial_learned_ties():
    r, learned_g, f = _build_group(
        primal=[0.0, 0.0, 0.0, 0.0],
        violation=[0.4, 0.1, 0.1, 0.2],
        feasible=[0.0, 0.0, 0.0, 0.0],
    )
    # Samples 1 and 2 tie under the learned controller, but raw violation
    # strictly prefers sample 2. Other learned scores remain informative.
    raw_g = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.float32)

    adv, stats = compute_lexicographic_clasp_advantages(
        r,
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=raw_g,
    )

    assert adv[2] > adv[1]
    assert adv[1] > adv[3]
    assert adv[3] > adv[0]
    assert stats["clasp/lexicographic_raw_fallback_group_rate"] == 1.0


def test_lexicographic_clasp_all_feasible_optimizes_objective():
    r, learned_g, f = _build_group(
        primal=[1.0, 4.0, 2.0, 3.0],
        violation=[0.0, 0.0, 0.0, 0.0],
        feasible=[1.0, 1.0, 1.0, 1.0],
    )

    adv, stats = compute_lexicographic_clasp_advantages(
        r,
        learned_g,
        f,
        num_generations=4,
        raw_violation_scores=learned_g,
    )

    assert int(adv.argmax().item()) == 1
    assert stats["clasp/lexicographic_all_feasible_group_rate"] == 1.0
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    expected = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    assert torch.allclose(adv, expected, atol=1e-6)


def test_lexicographic_clasp_preserves_true_ties_and_rejects_bad_inputs():
    zeros = torch.zeros(4)
    adv, _ = compute_lexicographic_clasp_advantages(
        zeros,
        zeros,
        zeros,
        num_generations=4,
        raw_violation_scores=zeros,
    )
    assert torch.equal(adv, zeros)

    with pytest.raises(ValueError, match="size mismatch"):
        compute_lexicographic_clasp_advantages(
            zeros,
            zeros,
            zeros,
            num_generations=4,
            raw_violation_scores=torch.zeros(5),
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_lexicographic_clasp_advantages(
            zeros,
            torch.tensor([0.0, -0.1, 0.0, 0.0]),
            zeros,
            num_generations=4,
            raw_violation_scores=zeros,
        )


def test_feasibility_warmup_prefers_feasible_then_lower_violation():
    g = torch.tensor([0.0, 0.8, 0.1, 0.4], dtype=torch.float32)
    f = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    adv, stats = compute_feasibility_warmup_advantages(
        g,
        f,
        num_generations=4,
        feasible_margin=3.0,
        violation_scale=2.0,
    )

    assert stats["clasp/feasibility_warmup"] == 1.0
    assert adv[0] > adv[1:].max()
    assert adv[2] > adv[1]
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)


def test_feasibility_warmup_can_mask_released_samples():
    g = torch.tensor([0.0, 0.2, 0.1, 0.4], dtype=torch.float32)
    f = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    active = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    adv, stats = compute_feasibility_warmup_advantages(
        g,
        f,
        num_generations=2,
        sample_warmup_flags=active,
    )

    assert stats["clasp/feasibility_warmup_group_active_rate"] == 0.5
    assert adv[0] > adv[1]
    assert adv[2].item() == 0.0
    assert adv[3].item() == 0.0


def test_pairwise_preference_loss_prefers_best_advantage_completion():
    log_ratios = torch.tensor([0.0, 0.2, -0.1, 0.0], dtype=torch.float32)
    advantages = torch.tensor([0.1, 1.0, -0.5, 0.0], dtype=torch.float32)

    loss, stats = compute_pairwise_preference_loss(
        log_ratios,
        advantages,
        num_generations=4,
        margin=0.1,
        advantage_delta=0.25,
    )

    expected = torch.nn.functional.softplus(torch.tensor(0.1 - (0.2 - (-0.1))))
    assert torch.allclose(loss, expected)
    assert stats["clasp/pairwise_valid_group_rate"] == 1.0
    assert math.isclose(stats["clasp/pairwise_advantage_spread"], 1.5, rel_tol=1e-6)


def test_pairwise_preference_loss_honors_advantage_delta():
    log_ratios = torch.tensor([0.0, 0.2, -0.1, 0.0], dtype=torch.float32)
    advantages = torch.tensor([0.1, 0.2, 0.0, 0.1], dtype=torch.float32)

    loss, stats = compute_pairwise_preference_loss(
        log_ratios,
        advantages,
        num_generations=4,
        margin=0.1,
        advantage_delta=0.5,
    )

    assert loss.item() == 0.0
    assert stats["clasp/pairwise_valid_group_rate"] == 0.0


def test_elite_imitation_loss_selects_highest_advantage_completion():
    per_token_logps = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-0.2, -0.4, -0.6],
            [-2.0, -2.0, -2.0],
            [-0.5, -0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    completion_mask = torch.ones_like(per_token_logps)
    advantages = torch.tensor([0.1, 1.0, -0.5, 0.0], dtype=torch.float32)

    loss, stats = compute_elite_imitation_loss(
        per_token_logps,
        completion_mask,
        advantages,
        num_generations=4,
        advantage_delta=0.25,
    )

    expected = torch.tensor(0.4)
    assert torch.allclose(loss, expected)
    assert stats["clasp/elite_imitation_valid_group_rate"] == 1.0
    assert math.isclose(stats["clasp/elite_imitation_advantage_spread"], 1.5, rel_tol=1e-6)
    assert math.isclose(stats["clasp/elite_imitation_selected_length"], 3.0, rel_tol=1e-6)


def test_elite_imitation_loss_honors_advantage_delta():
    per_token_logps = torch.full((4, 2), -0.5, dtype=torch.float32)
    completion_mask = torch.ones_like(per_token_logps)
    advantages = torch.tensor([0.1, 0.2, 0.0, 0.1], dtype=torch.float32)

    loss, stats = compute_elite_imitation_loss(
        per_token_logps,
        completion_mask,
        advantages,
        num_generations=4,
        advantage_delta=0.5,
    )

    assert loss.item() == 0.0
    assert stats["clasp/elite_imitation_valid_group_rate"] == 0.0


def test_multi_elite_imitation_loss_selects_low_gap_feasible_elites():
    per_token_logps = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-0.2, -0.4, -0.6],
            [-2.0, -2.0, -2.0],
            [-0.5, -0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    completion_mask = torch.ones_like(per_token_logps)
    advantages = torch.tensor([0.1, 1.0, -0.5, 0.8], dtype=torch.float32)
    raw_gaps = torch.tensor([0.03, 0.01, 0.0, 0.015], dtype=torch.float32)
    feasible = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    loss, stats = compute_multi_elite_imitation_loss(
        per_token_logps,
        completion_mask,
        advantages,
        raw_gaps,
        feasible,
        num_generations=4,
        elite_fraction=0.5,
        max_raw_gap=0.02,
        temperature=0.0,
    )

    assert torch.allclose(loss, torch.tensor(0.45))
    assert stats["clasp/multi_elite_imitation_valid_group_rate"] == 1.0
    assert stats["clasp/multi_elite_imitation_target_group_rate"] == 1.0
    assert stats["clasp/multi_elite_imitation_fallback_group_rate"] == 0.0
    assert math.isclose(stats["clasp/multi_elite_imitation_selected_count"], 2.0, rel_tol=1e-6)
    assert math.isclose(stats["clasp/multi_elite_imitation_selected_raw_gap"], 0.0125, rel_tol=1e-6)


def test_multi_elite_imitation_loss_falls_back_to_best_feasible():
    per_token_logps = torch.tensor(
        [
            [-1.0, -1.0],
            [-0.25, -0.75],
            [-2.0, -2.0],
            [-0.75, -0.75],
        ],
        dtype=torch.float32,
    )
    completion_mask = torch.ones_like(per_token_logps)
    advantages = torch.tensor([0.1, 1.0, -0.5, 0.8], dtype=torch.float32)
    raw_gaps = torch.tensor([0.03, 0.04, 0.00, 0.05], dtype=torch.float32)
    feasible = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    loss, stats = compute_multi_elite_imitation_loss(
        per_token_logps,
        completion_mask,
        advantages,
        raw_gaps,
        feasible,
        num_generations=4,
        elite_fraction=0.25,
        max_raw_gap=0.02,
        temperature=0.0,
    )

    assert torch.allclose(loss, torch.tensor(0.5))
    assert stats["clasp/multi_elite_imitation_valid_group_rate"] == 1.0
    assert stats["clasp/multi_elite_imitation_target_group_rate"] == 0.0
    assert stats["clasp/multi_elite_imitation_fallback_group_rate"] == 1.0
    assert math.isclose(stats["clasp/multi_elite_imitation_selected_raw_gap"], 0.04, rel_tol=1e-6)


def test_multi_elite_imitation_loss_honors_advantage_delta():
    per_token_logps = torch.full((4, 2), -0.5, dtype=torch.float32)
    completion_mask = torch.ones_like(per_token_logps)
    advantages = torch.tensor([0.1, 0.2, 0.0, 0.1], dtype=torch.float32)
    raw_gaps = torch.zeros(4, dtype=torch.float32)
    feasible = torch.ones(4, dtype=torch.float32)

    loss, stats = compute_multi_elite_imitation_loss(
        per_token_logps,
        completion_mask,
        advantages,
        raw_gaps,
        feasible,
        num_generations=4,
        advantage_delta=0.5,
    )

    assert loss.item() == 0.0
    assert stats["clasp/multi_elite_imitation_valid_group_rate"] == 0.0


# ---------------------------------------------------------------------------
# Feasibility-conditional dual learning rate
# ---------------------------------------------------------------------------


def test_feasibility_conditional_dual_lr_monotone():
    eta_low = feasibility_conditional_dual_lr(base_lr=0.05, feasibility_rate=0.1)
    eta_mid = feasibility_conditional_dual_lr(base_lr=0.05, feasibility_rate=0.5)
    eta_high = feasibility_conditional_dual_lr(base_lr=0.05, feasibility_rate=0.9)
    assert eta_low > eta_mid > eta_high
    assert eta_low / eta_high > 2.5


def test_feasibility_conditional_dual_lr_clamps_extremes():
    eta0 = feasibility_conditional_dual_lr(base_lr=0.05, feasibility_rate=0.0)
    eta1 = feasibility_conditional_dual_lr(base_lr=0.05, feasibility_rate=1.0)
    assert math.isfinite(eta0) and math.isfinite(eta1)
    assert eta1 < eta0


def test_feasibility_conditional_dual_lr_matches_paper_formula():
    base_lr = 0.05
    feas = 0.25
    eps = 1e-3
    expected = base_lr * math.sqrt((1.0 - feas + eps) / (feas + eps))
    actual = feasibility_conditional_dual_lr(
        base_lr=base_lr,
        feasibility_rate=feas,
        eps=eps,
    )
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


def test_clasp_controller_signed_slack_can_cancel_balanced_batch():
    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["gap"],
            "cost_vectors": [[0.0], [0.2]],
            "cost_masks": [[1.0], [1.0]],
            "constraint_scores": [1.0, 0.8],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.1,
        lambda_init=0.0,
        dual_lr=1.0,
        lambda_max=10.0,
        log_every=0,
    )
    reward.enable_clasp_proper_dual(base_dual_lr=1.0, dual_update_interval=1)
    reward.record_batch_feasibility_rate(0.5)

    reward(["p", "p"], ["under", "over"])
    reward.update_lambda()

    # The proper masked signed mean is ((0 - .1) + (.2 - .1)) / 2 = 0.
    assert math.isclose(reward.dual_lambdas[0], 0.0, rel_tol=0.0, abs_tol=1e-8)


def test_clasp_positive_excess_is_explicit_one_sided_ablation():
    reward = PrimalDualConstrainedReward(
        structure_fn=lambda prompts, completions, **kwargs: [0.0] * len(completions),
        constraint_fn=None,
        objective_fn=lambda prompts, completions, **kwargs: [0.0] * len(completions),
        cost_components_fn=lambda prompts, completions, **kwargs: {
            "cost_names": ["gap"],
            "cost_vectors": [[0.0], [0.2]],
            "cost_masks": [[1.0], [1.0]],
            "constraint_scores": [1.0, 0.8],
        },
        cost_limit=0.1,
        lambda_init=0.0,
        dual_lr=1.0,
        lambda_max=10.0,
        log_every=0,
    )
    reward.enable_clasp_proper_dual(
        base_dual_lr=1.0,
        dual_update_interval=1,
        gradient_mode="positive_excess",
    )
    reward.record_batch_feasibility_rate(0.5)
    reward(["p", "p"], ["under", "over"])
    reward.update_lambda()

    assert math.isclose(reward.dual_lambdas[0], 0.05, rel_tol=1e-6, abs_tol=1e-8)


@pytest.mark.parametrize(
    ("lambda_init", "observed_cost", "direction"),
    [
        (0.2, 0.8, "increase"),
        (0.5, 0.4, "decrease"),
        (0.01, 0.0, "floor"),
    ],
)
def test_clasp_signed_projected_dual_increase_decrease_and_floor(
    lambda_init,
    observed_cost,
    direction,
):
    reward = PrimalDualConstrainedReward(
        structure_fn=lambda prompts, completions, **kwargs: [0.0],
        constraint_fn=None,
        objective_fn=lambda prompts, completions, **kwargs: [0.0],
        cost_components_fn=lambda prompts, completions, **kwargs: {
            "cost_names": ["gap"],
            "cost_vectors": [[observed_cost]],
            "cost_masks": [[1.0]],
            "constraint_scores": [1.0 - observed_cost],
        },
        cost_limit=0.5,
        lambda_init=lambda_init,
        dual_lr=1.0,
        lambda_max=10.0,
        violation_tolerance=0.0,
        log_every=0,
    )
    reward.enable_clasp_proper_dual(base_dual_lr=1.0, dual_update_interval=1)
    reward.record_batch_feasibility_rate(0.5)
    reward(["p"], ["c"])
    reward.update_lambda()

    if direction == "increase":
        assert reward.dual_lambdas[0] > lambda_init
    elif direction == "decrease":
        assert 0.0 < reward.dual_lambdas[0] < lambda_init
    else:
        assert reward.dual_lambdas[0] == 0.0


def test_clasp_signed_dual_uses_masked_count_not_total_batch_count():
    reward = PrimalDualConstrainedReward(
        structure_fn=lambda prompts, completions, **kwargs: [0.0, 0.0],
        constraint_fn=None,
        objective_fn=lambda prompts, completions, **kwargs: [0.0, 0.0],
        cost_components_fn=lambda prompts, completions, **kwargs: {
            "cost_names": ["gap", "guard"],
            "cost_vectors": [[0.8, 0.0], [0.0, 0.0]],
            "cost_masks": [[1.0, 0.0], [0.0, 1.0]],
            "constraint_scores": [0.2, 1.0],
        },
        cost_limit=[0.5, 0.0],
        lambda_init=[0.0, 0.0],
        dual_lr=1.0,
        lambda_max=10.0,
        violation_tolerance=0.0,
        log_every=0,
    )
    reward.enable_clasp_proper_dual(base_dual_lr=1.0, dual_update_interval=1)
    reward.record_batch_feasibility_rate(0.5)
    reward(["p", "p"], ["observed", "masked"])
    reward.update_lambda()

    assert math.isclose(reward.dual_lambdas[0], 0.3, rel_tol=1e-6, abs_tol=1e-8)


def test_clasp_signed_dual_pending_state_resumes_exactly():
    def build_reward():
        reward = PrimalDualConstrainedReward(
            structure_fn=lambda prompts, completions, **kwargs: [0.0],
            constraint_fn=None,
            objective_fn=lambda prompts, completions, **kwargs: [0.0],
            cost_components_fn=lambda prompts, completions, **kwargs: {
                "cost_names": ["gap"],
                "cost_vectors": [[0.8]],
                "cost_masks": [[1.0]],
                "constraint_scores": [0.2],
            },
            cost_limit=0.5,
            lambda_init=0.2,
            dual_lr=1.0,
            lambda_max=10.0,
            violation_tolerance=0.0,
            log_every=0,
        )
        reward.enable_clasp_proper_dual(base_dual_lr=1.0, dual_update_interval=2)
        reward.record_batch_feasibility_rate(0.5)
        return reward

    uninterrupted = build_reward()
    uninterrupted(["p"], ["first"])
    uninterrupted.update_lambda()
    checkpoint = uninterrupted.state_dict()

    restored = build_reward()
    restored.load_state_dict(checkpoint)
    uninterrupted(["p"], ["second"])
    restored(["p"], ["second"])
    uninterrupted.update_lambda()
    restored.update_lambda()

    assert restored.dual_lambdas == pytest.approx(uninterrupted.dual_lambdas)
    assert restored.update_count == uninterrupted.update_count
    assert restored.state_dict()["controller_state"] == uninterrupted.state_dict()[
        "controller_state"
    ]


def test_resume_state_tightens_limits_and_weights_to_current_profile():
    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["capacity_overflow", "over_capacity_route_fraction"],
            "cost_vectors": [[0.01, 0.5]],
            "cost_masks": [[1.0, 1.0]],
            "constraint_scores": [0.5],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.0,
        lambda_init=0.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
        component_cost_limit_map={
            "capacity_overflow": 0.0,
            "over_capacity_route_fraction": 0.0,
        },
        component_weight_map={
            "capacity_overflow": 12.0,
            "over_capacity_route_fraction": 14.0,
        },
    )
    reward.load_state_dict(
        {
            "cost_names": ["capacity_overflow"],
            "cost_limits": [0.02],
            "cost_weights": [3.0],
            "dual_lambdas": [0.7],
        }
    )

    reward(["p"], ["candidate"])

    assert reward.cost_names == ["capacity_overflow", "over_capacity_route_fraction"]
    assert reward.cost_limits == [0.0, 0.0]
    assert reward.cost_weights == [12.0, 14.0]
    assert reward.dual_lambdas[0] == 0.7
    assert reward.get_last_components()["feasibility_flags"] == [0.0]


def test_resume_state_relaxes_limits_when_current_profile_relaxes_head():
    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["interior_depot"],
            "cost_vectors": [[0.02]],
            "cost_masks": [[1.0]],
            "constraint_scores": [0.98],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.0,
        lambda_init=0.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
        component_cost_limit_map={"interior_depot": 1.0},
        component_weight_map={"interior_depot": 2.5},
    )
    reward.load_state_dict(
        {
            "cost_names": ["interior_depot"],
            "cost_limits": [0.0],
            "cost_weights": [8.0],
            "dual_lambdas": [0.5],
        }
    )

    reward(["p"], ["candidate"])

    assert reward.cost_limits == [1.0]
    assert reward.cost_weights == [2.5]
    assert reward.dual_lambdas == [0.5]
    assert reward.get_last_components()["feasibility_flags"] == [1.0]


def test_feasibility_ignore_cost_names_preserves_positive_gap(monkeypatch):
    monkeypatch.setenv(
        "CLASP_FEASIBILITY_IGNORE_COST_NAMES",
        "target_gap_excess+target_tail_gap_excess",
    )

    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["route_error", "target_gap_excess", "target_tail_gap_excess"],
            "cost_vectors": [[0.0, 0.5, 0.2]],
            "cost_masks": [[1.0, 1.0, 1.0]],
            "constraint_scores": [0.75],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.0,
        lambda_init=1.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
    )

    reward(["p"], ["route-valid-but-above-target"])
    components = reward.get_last_components()

    assert components["feasibility_flags"] == [1.0]
    assert components["constraint_violation_scores"] == [0.0]
    assert components["positive_weighted_gap_scores"] == [0.7]
    assert components["weighted_gap_scores"] == [0.7]


def test_target_gap_cost_affects_feasibility_by_default(monkeypatch):
    monkeypatch.delenv("CLASP_FEASIBILITY_IGNORE_COST_NAMES", raising=False)

    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["route_error", "target_gap_excess"],
            "cost_vectors": [[0.0, 0.5]],
            "cost_masks": [[1.0, 1.0]],
            "constraint_scores": [0.75],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.0,
        lambda_init=1.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
    )

    reward(["p"], ["route-valid-but-above-target"])
    components = reward.get_last_components()

    assert components["feasibility_flags"] == [0.0]
    assert components["constraint_violation_scores"] == [0.25]
    assert components["positive_weighted_gap_scores"] == [0.5]


def test_tsp_target_tail_gap_cost_is_default_off_and_thresholded(monkeypatch):
    for name in [
        "CLASP_TSP_TARGET_GAP",
        "CLASP_TSP_TARGET_GAP_CONSTRAINT",
        "CLASP_TSP_TARGET_GAP_CONSTRAINT_GAP",
        "CLASP_TSP_TARGET_GAP_CONSTRAINT_SCALE",
        "CLASP_TSP_TAIL_GAP",
        "CLASP_TSP_TAIL_GAP_CONSTRAINT",
        "CLASP_TSP_TAIL_GAP_CONSTRAINT_GAP",
        "CLASP_TSP_TAIL_GAP_CONSTRAINT_SCALE",
    ]:
        monkeypatch.delenv(name, raising=False)

    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
    prompts = ["tsp"]
    completions = ["Route: [0, 1, 2, 3, 0]\nObjective: 30.0"]
    answers = ["Route: [0, 1, 3, 2, 0]\nObjective: 28.9"]

    base = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    assert "target_tail_gap_excess" not in base["cost_names"]

    monkeypatch.setenv("CLASP_TSP_TARGET_GAP", "0.0107")
    monkeypatch.setenv("CLASP_TSP_TARGET_GAP_CONSTRAINT", "1")
    monkeypatch.setenv("CLASP_TSP_TAIL_GAP", "0.50")
    monkeypatch.setenv("CLASP_TSP_TAIL_GAP_CONSTRAINT", "1")

    high_tail = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    target_idx = high_tail["cost_names"].index("target_gap_excess")
    tail_idx = high_tail["cost_names"].index("target_tail_gap_excess")
    assert high_tail["cost_vectors"][0][target_idx] > 0.0
    assert high_tail["cost_masks"][0][target_idx] == 1.0
    assert high_tail["cost_vectors"][0][tail_idx] == 0.0
    assert high_tail["cost_masks"][0][tail_idx] == 1.0

    monkeypatch.setenv("CLASP_TSP_TAIL_GAP", "0.025")
    low_tail = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    tail_idx = low_tail["cost_names"].index("target_tail_gap_excess")
    assert low_tail["cost_vectors"][0][tail_idx] > 0.0
    assert low_tail["cost_masks"][0][tail_idx] == 1.0


def test_tsp_objective_gap_cost_is_default_off_and_cap_normalized(monkeypatch):
    for name in [
        "CLASP_TSP_OBJECTIVE_GAP_COST",
        "CLASP_TSP_OBJECTIVE_GAP_COST_CAP",
        "CLASP_TSP_OBJECTIVE_GAP_COST_SCALE",
    ]:
        monkeypatch.delenv(name, raising=False)

    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
    prompts = ["tsp"]
    completions = ["Route: [0, 1, 2, 3, 0]\nObjective: 30.0"]
    answers = ["Route: [0, 1, 3, 2, 0]\nObjective: 28.9"]

    base = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    assert "tsp_objective_gap" not in base["cost_names"]

    monkeypatch.setenv("CLASP_TSP_OBJECTIVE_GAP_COST", "1")
    monkeypatch.setenv("CLASP_TSP_OBJECTIVE_GAP_COST_CAP", "0.05")
    enabled = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    gap_idx = enabled["cost_names"].index("tsp_objective_gap")
    assert enabled["cost_masks"][0][gap_idx] == 1.0
    assert 0.0 < enabled["cost_vectors"][0][gap_idx] < 1.0

    monkeypatch.setenv("CLASP_TSP_OBJECTIVE_GAP_COST_SCALE", "2.0")
    saturated = tsp_cost_components(prompts, completions, instance=[coords], answer=answers)
    assert saturated["cost_vectors"][0][gap_idx] == 1.0


def test_tsp_objective_gap_profile_and_feasibility_ignore(monkeypatch):
    limits, weights = build_component_profile("tsp", "co_default")
    assert limits["tsp_objective_gap"] == 0.0
    assert weights["tsp_objective_gap"] == 1.0

    monkeypatch.setenv("CLASP_FEASIBILITY_IGNORE_COST_NAMES", "tsp_objective_gap")

    def structure_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def objective_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def cost_components_fn(prompts, completions, **kwargs):
        return {
            "cost_names": ["route_error", "tsp_objective_gap"],
            "cost_vectors": [[0.0, 0.5]],
            "cost_masks": [[1.0, 1.0]],
            "constraint_scores": [0.75],
        }

    reward = PrimalDualConstrainedReward(
        structure_fn=structure_fn,
        constraint_fn=None,
        objective_fn=objective_fn,
        cost_components_fn=cost_components_fn,
        cost_limit=0.0,
        lambda_init=1.0,
        dual_lr=0.0,
        lambda_max=10.0,
        log_every=0,
    )

    reward(["p"], ["route-valid-with-positive-gap"])
    components = reward.get_last_components()

    assert components["feasibility_flags"] == [1.0]
    assert components["constraint_violation_scores"] == [0.0]
    assert components["positive_weighted_gap_scores"] == [0.5]
    assert components["raw_positive_gap_scores"] == pytest.approx([1.0 / 3.0])


def test_tsp_two_opt_improved_route_returns_better_closed_tour():
    coords = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    crossing_route = [0, 1, 2, 3, 0]

    improved = _tsp_two_opt_improved_route(coords, crossing_route, max_passes=2)

    assert improved is not None
    improved_route, original_total, improved_total, gain_ratio, passes = improved
    assert improved_route[0] == 0
    assert improved_route[-1] == 0
    assert sorted(improved_route[:-1]) == [0, 1, 2, 3]
    assert improved_total < original_total
    assert gain_ratio > 0.0
    assert passes >= 1


# ---------------------------------------------------------------------------
# Adaptive length estimator
# ---------------------------------------------------------------------------


def test_adaptive_length_estimator_respects_floor_cap():
    est = AdaptiveLengthEstimator(
        tasks=["tsp", "cvrp"],
        global_cap=1024,
        floor=128,
        window=256,
        quantile=0.99,
        refresh_steps=1,
    )
    for _ in range(100):
        est.observe("tsp", 40)
        est.observe("cvrp", 900)
    est.maybe_refresh(global_step=1)
    # tsp: 99-th percentile is 40 -> clamped to floor 128
    assert est.budget("tsp") == 128
    # cvrp: 99-th percentile is ~900, +32 headroom, capped at 1024
    assert 900 <= est.budget("cvrp") <= 1024


def test_adaptive_length_estimator_cold_budget_is_cap():
    est = AdaptiveLengthEstimator(
        tasks=["tsp"],
        global_cap=1024,
        floor=128,
        refresh_steps=10,
    )
    # Never refreshed; budget stays at cap.
    assert est.budget("tsp") == 1024
    # Unknown task defaults to cap.
    assert est.budget("mystery") == 1024


def test_adaptive_length_estimator_margin_is_configurable():
    no_margin = AdaptiveLengthEstimator(
        tasks=["tsp"],
        global_cap=1024,
        floor=128,
        quantile=0.5,
        refresh_steps=1,
        margin=0,
    )
    default_margin = AdaptiveLengthEstimator(
        tasks=["tsp"],
        global_cap=1024,
        floor=128,
        quantile=0.5,
        refresh_steps=1,
        margin=32,
    )
    for _ in range(40):
        no_margin.observe("tsp", 200)
        default_margin.observe("tsp", 200)
    no_margin.maybe_refresh(global_step=1)
    default_margin.maybe_refresh(global_step=1)
    assert no_margin.budget("tsp") == 200
    assert default_margin.budget("tsp") == 232


def test_adaptive_length_estimator_rejects_negative_margin():
    with pytest.raises(ValueError, match="margin"):
        AdaptiveLengthEstimator(
            tasks=["tsp"],
            global_cap=1024,
            margin=-1,
        )


def test_pareto_center_changes_alpha_and_is_reported():
    primal = torch.tensor([0.0, 1.0, 2.0, 3.0])
    gaps = torch.tensor([1.0, 0.0, 1.0, 0.0])
    feasible = torch.tensor([0.0, 1.0, 0.0, 1.0])

    _, low_stats = compute_pareto_anchored_advantages(
        primal,
        gaps,
        feasible,
        num_generations=4,
        tau=4.0,
        feasibility_center=0.25,
    )
    _, high_stats = compute_pareto_anchored_advantages(
        primal,
        gaps,
        feasible,
        num_generations=4,
        tau=4.0,
        feasibility_center=0.75,
    )

    assert low_stats["clasp/feasibility_center"] == pytest.approx(0.25)
    assert high_stats["clasp/feasibility_center"] == pytest.approx(0.75)
    assert low_stats["clasp/mean_alpha"] > high_stats["clasp/mean_alpha"]


def test_pareto_center_rejects_out_of_range_values():
    primal = torch.tensor([0.0, 1.0])
    gaps = torch.tensor([1.0, 0.0])
    feasible = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="feasibility_center"):
        compute_pareto_anchored_advantages(
            primal,
            gaps,
            feasible,
            num_generations=2,
            feasibility_center=1.01,
        )


def test_public_surface():
    # run_grpo.py imports these names directly; make sure they stay public.
    import clasp.rewards.clasp_math as m
    assert hasattr(m, "compute_pareto_anchored_advantages")
    assert hasattr(m, "feasibility_conditional_dual_lr")
    assert hasattr(m, "AdaptiveLengthEstimator")


# ---------------------------------------------------------------------------
# Improvement-operator distillation teachers (CVRP / JSSP / MIS)
# ---------------------------------------------------------------------------


def test_cvrp_local_search_relocates_and_drops_empty_route():
    # Customers 1 and 2 sit on the same ray; serving them in one route is
    # strictly shorter than two out-and-back routes.
    coords = [[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    demands = [0.0, 1.0, 1.0]
    capacity = 2.0
    routes = [[0, 1, 0], [0, 2, 0]]

    improved = _cvrp_local_search_improved_routes(coords, demands, capacity, routes, max_passes=2)

    assert improved is not None
    improved_routes, original_total, improved_total, gain_ratio, passes = improved
    assert improved_total < original_total
    assert gain_ratio > 0.0
    assert passes >= 1
    # The emptied route is dropped and the result stays fully feasible.
    assert len(improved_routes) == 1
    _, _, feasible = _check_cvrp_constraints(improved_routes, coords, demands, capacity)
    assert feasible
    assert math.isclose(_cvrp_total_distance(coords, improved_routes), improved_total)


def test_cvrp_local_search_respects_capacity():
    # Same geometry, but each vehicle can carry only one customer: the
    # profitable relocation is capacity-infeasible and nothing else improves.
    coords = [[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    demands = [0.0, 1.0, 1.0]
    capacity = 1.0
    routes = [[0, 1, 0], [0, 2, 0]]

    assert _cvrp_local_search_improved_routes(coords, demands, capacity, routes, max_passes=3) is None


def test_cvrp_local_search_uncrosses_route_with_two_opt():
    coords = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    demands = [0.0, 1.0, 1.0, 1.0]
    capacity = 10.0
    crossing = [[0, 1, 2, 3, 0]]

    improved = _cvrp_local_search_improved_routes(coords, demands, capacity, crossing, max_passes=2)

    assert improved is not None
    improved_routes, original_total, improved_total, _, _ = improved
    assert improved_total < original_total
    assert sorted(node for r in improved_routes for node in r if node != 0) == [1, 2, 3]
    _, _, feasible = _check_cvrp_constraints(improved_routes, coords, demands, capacity)
    assert feasible


def test_cvrp_local_search_rejects_interior_depot():
    coords = [[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    demands = [0.0, 1.0, 1.0]
    assert (
        _cvrp_local_search_improved_routes(coords, demands, 2.0, [[0, 1, 0, 2, 0]], max_passes=1)
        is None
    )


def test_cvrp_distillation_target_parses_back():
    routes = [[0, 1, 2, 0], [0, 3, 0]]
    text = _format_cvrp_distillation_target(routes, 12.5)
    parsed = _parse_cvrp_solution(text)
    assert parsed["routes"] == routes
    assert parsed["declared_objective"] == pytest.approx(12.5)


def test_jssp_order_descent_finds_better_machine_order():
    # J0 = M0(1) then M1(5); J1 = M1(1) then M0(5).
    # Serving J0 before J1 on both machines forces makespan 12; swapping the
    # M1 order to [1, 0] interleaves the jobs for makespan 6.
    jobs = [[(0, 1), (1, 5)], [(1, 1), (0, 5)]]
    schedule = [[0, 1], [0, 1]]
    assert _schedule_makespan(jobs, schedule) == pytest.approx(12.0)

    improved = _jssp_order_descent_improved_schedule(jobs, schedule, max_passes=2)

    assert improved is not None
    improved_schedule, original_makespan, improved_makespan, gain_ratio, passes = improved
    assert original_makespan == pytest.approx(12.0)
    assert improved_makespan == pytest.approx(6.0)
    assert gain_ratio == pytest.approx(0.5)
    assert passes >= 1
    _, _, feasible = _check_jssp_constraints(jobs, improved_schedule)
    assert feasible


def test_jssp_order_descent_noop_at_local_optimum():
    jobs = [[(0, 1), (1, 5)], [(1, 1), (0, 5)]]
    optimal_schedule = [[0, 1], [1, 0]]
    assert _schedule_makespan(jobs, optimal_schedule) == pytest.approx(6.0)
    assert _jssp_order_descent_improved_schedule(jobs, optimal_schedule, max_passes=2) is None


def test_jssp_distillation_target_parses_back():
    schedule = [[0, 1], [1, 0]]
    text = _format_jssp_distillation_target(schedule, 6.0)
    parsed = _parse_jssp_solution(text)
    assert parsed["schedule"] == schedule
    assert parsed["declared_makespan"] == pytest.approx(6.0)


def test_mis_maximalized_set_extends_non_maximal_set():
    # Path graph 0-1-2-3-4; {0} extends to the maximum independent set {0, 2, 4}.
    edges = {(0, 1), (1, 2), (2, 3), (3, 4)}

    improved = _mis_maximalized_set([0], 5, edges)

    assert improved is not None
    improved_set, original_size, improved_size, added_fraction, added = improved
    assert improved_set == [0, 2, 4]
    assert original_size == pytest.approx(1.0)
    assert improved_size == pytest.approx(3.0)
    assert added == 2
    assert added_fraction == pytest.approx(2.0 / 3.0)
    _, _, feasible = _check_mis_constraints(improved_set, 5, edges)
    assert feasible


def test_mis_maximalized_set_noop_when_already_maximal():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert _mis_maximalized_set([0, 2, 4], 5, edges) is None


def test_mis_maximalized_set_rejects_infeasible_input():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert _mis_maximalized_set([0, 1], 5, edges) is None


def test_mis_distillation_target_parses_back():
    text = _format_mis_distillation_target([4, 0, 2])
    parsed = _parse_mis_solution(text)
    assert parsed["node_set"] == [0, 2, 4]
    assert parsed["declared_objective"] == pytest.approx(3.0)
