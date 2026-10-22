"""Strict reward ordering and unchanged defaults for plain GRPO's opt-in head."""
import math
import os

import pytest

from clasp.rewards import co_tasks as rewards

COORDS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
DEMANDS = [0.0, 1.0, 1.0]
ROUTE = [[0, 1, 2, 0]]
REFERENCE = "Routes: [[0, 1, 2, 0]], Objective: 999999"
CVRP_INSTANCE = [COORDS, DEMANDS, 2.0]
MIS_INSTANCE = {"num_nodes": 4, "edges": [[0, 1], [1, 2]]}


@pytest.fixture(autouse=True)
def clean_reward_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("CLASP_", "PD_")) or name == "GRPO_GRADED_REWARD":
            monkeypatch.delenv(name)


def cvrp_head(text, *, instance=CVRP_INSTANCE, answer=REFERENCE, **kwargs):
    return rewards.cvrp_objective_reward([""], [text], instance=[instance], answer=[answer], **kwargs)[0]


def mis_head(text, *, instance=MIS_INSTANCE, answer="Set: [0, 2], Objective: 2", **kwargs):
    return rewards.mis_objective_reward([""], [text], instance=[instance], answer=[answer], **kwargs)[0]


@pytest.mark.parametrize("mode", [None, "0"])
def test_default_and_explicit_off_preserve_original_rewards(monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("GRPO_GRADED_REWARD", mode)
    assert cvrp_head(REFERENCE) == 2.0
    assert cvrp_head("Routes: [[0, 1, 0]], Objective: 2") == 0.0
    assert mis_head("Set: [0, 2], Objective: 2") == 2.0
    assert mis_head("Set: [0, 1, 2], Objective: 3") == 0.0
    assert mis_head("Set: [0], Objective: 1") == pytest.approx(2 * math.exp(-1.5))


@pytest.mark.parametrize("value", ["", "true", "false", "2", "-1", "0.5"])
def test_malformed_opt_in_flag_fails_closed(monkeypatch, value):
    monkeypatch.setenv("GRPO_GRADED_REWARD", value)
    with pytest.raises(ValueError, match="GRPO_GRADED_REWARD"):
        mis_head("Set: [0, 2]")


def test_opt_in_has_no_clasp_dependency_and_ignores_legacy_soft_gating(monkeypatch):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    assert cvrp_head(REFERENCE) == 1.5
    assert mis_head("Set: [0, 2], Objective: 2") == 1.5
    assert mis_head("Set: [0, 1, 2], Objective: 999", allow_infeasible_partial_credit=True,
                    apply_constraint_gate=False) < 0
    assert cvrp_head("Routes: [[0, 1, 0]], Objective: 0", allow_infeasible_partial_credit=True,
                     apply_constraint_gate=False) < 0


@pytest.mark.parametrize("head,text", [(cvrp_head, "garbled route"), (mis_head, "garbled set"),
                                       (cvrp_head, "Routes: []"), (mis_head, "Set: []")])
def test_empty_or_unparseable_solutions_receive_minus_one(monkeypatch, head, text):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    assert head(text) == -1.0


def test_invalid_cvrp_never_evaluates_distance_or_declared_objective(monkeypatch):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    monkeypatch.setattr(rewards, "_cvrp_total_distance", lambda *args: pytest.fail("invalid distance read"))
    values = [cvrp_head("Routes: [[0, 1, 0]], Objective: " + declaration, answer=answer)
              for declaration in ("0", "-1000", "999999999")
              for answer in (REFERENCE, None, "not a reference")]
    assert values == [pytest.approx(-1 / 3)] * len(values)


def test_invalid_mis_never_receives_size_or_declared_objective_credit(monkeypatch):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    # Vertex 3 has no edges: adding it leaves the one existing conflict intact.
    short = mis_head("Set: [0, 1], Objective: -1", answer=None)
    padded = mis_head("Set: [0, 1, 3], Objective: 999999", answer="wrong reference")
    assert short == padded == pytest.approx(-0.25 / 1.25)
    assert mis_head("Set: [0, 1, 2, 3], Objective: 999999") < short < 0


def test_cvrp_quality_is_scale_invariant_and_improves_beyond_heuristic_reference():
    distance = 2 + math.sqrt(2)
    baseline = rewards.grpo_graded_cvrp_reward(ROUTE, COORDS, DEMANDS, 2.0, distance)
    improved = rewards.grpo_graded_cvrp_reward(ROUTE, COORDS, DEMANDS, 2.0, 4.0)
    worse = rewards.grpo_graded_cvrp_reward([[0, 1, 0], [0, 2, 0]], COORDS, DEMANDS, 2.0, distance)
    assert 1 < worse < baseline == 1.5 < improved < 2
    assert rewards.grpo_graded_cvrp_reward(ROUTE, [[10 * x, 10 * y] for x, y in COORDS],
                                           DEMANDS, 2.0, 40.0) == pytest.approx(improved)


def test_mis_feasible_cardinality_is_normalized_and_ties_stay_tied():
    edges = {(0, 1), (1, 2)}
    assert rewards.grpo_graded_mis_reward([0], 4, edges) == 1.25
    assert rewards.grpo_graded_mis_reward([0, 2], 4, edges) == rewards.grpo_graded_mis_reward([1, 3], 4, edges)
    assert rewards.grpo_graded_mis_reward([0, 2, 3], 4, edges) == 1.75
    assert rewards.grpo_graded_mis_reward([0, 1, 2, 3], 4, set()) == 2.0


def test_invalid_cvrp_components_have_severity_and_no_output_length_dilution():
    def reward(routes, capacity=2.0):
        return rewards.grpo_graded_cvrp_reward(routes, COORDS, DEMANDS, capacity, None)
    near = reward([[0, 1, 0]])
    assert near == reward([[0, 1, 0]] + [[0, 0]] * 25)
    assert reward([[0, 0]]) < near < 0  # Two missing customers versus one.
    assert reward([[0, 1, 1, 0]]) < near  # Added duplication and capacity overflow.
    assert reward([[0, 1, 8, 0]]) < near  # Added invalid ID; never indexed into demands.
    assert reward([[1, 2]]) < 0  # Coverage alone does not excuse depot boundaries.
    assert reward(ROUTE, 0.5) < reward(ROUTE, 1.0) < 0


def test_all_invalid_groups_can_have_nonzero_credit_without_arbitrary_tie_breaks():
    edges = {(0, 1), (1, 2)}
    values = [rewards.grpo_graded_mis_reward(s, 4, edges)
              for s in ([0, 1], [0, 1, 2], [0, 1, 3], [])]
    assert values[0] == values[2]
    assert -1 == values[3] < values[1] < values[0] < 0


@pytest.mark.parametrize("coords,demands,capacity", [
    ([], [], 1), (COORDS, [0, 1], 2), (COORDS, [0, -1, 1], 2),
    (COORDS, DEMANDS, 0), (COORDS, DEMANDS, float("nan")),
    (COORDS, DEMANDS, True), ([[0, 0], [float("inf"), 1]], [0, 1], 2),
    ([[0, 0], [1]], [0, 1], 2), ([[0, 0], [1, "bad"]], [0, 1], 2),
])
def test_malformed_cvrp_metadata_fails_even_for_parse_failures(coords, demands, capacity):
    with pytest.raises(ValueError):
        rewards.grpo_graded_cvrp_reward([], coords, demands, capacity, 1)


@pytest.mark.parametrize("n,edges", [(0, set()), (True, set()), (3.5, set()),
                                     (3, {(0, 3)}), (3, {(1, 1)}), (3, {(2, 1)}),
                                     (3, {(0, "bad")})])
def test_malformed_mis_metadata_fails_even_for_parse_failures(n, edges):
    with pytest.raises(ValueError):
        rewards.grpo_graded_mis_reward([], n, edges)


@pytest.mark.parametrize("instance", [{"num_nodes": 4}, {"num_nodes": 4, "edges": "lost"},
                                      {"num_nodes": 4.5, "edges": []},
                                      {"num_nodes": 4, "edges": [[0, 0]]},
                                      {"num_nodes": 4, "edges": [[0, 4]]}])
def test_mis_head_does_not_silently_drop_malformed_graph_metadata(monkeypatch, instance):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    with pytest.raises(ValueError):
        mis_head("Set: [0, 1]", instance=instance)


@pytest.mark.parametrize("answer", [None, "Routes: [[0, 1, 0]]", "Routes: [[0, 7, 0]]"])
def test_feasible_cvrp_requires_valid_reference_routes(monkeypatch, answer):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    with pytest.raises(ValueError, match="reference"):
        cvrp_head(REFERENCE, answer=answer)


@pytest.mark.parametrize("reference", [None, 0, -1, float("nan"), float("inf")])
def test_feasible_cvrp_requires_finite_positive_reference_scale(reference):
    with pytest.raises(ValueError, match="reference"):
        rewards.grpo_graded_cvrp_reward(ROUTE, COORDS, DEMANDS, 2.0, reference)


@pytest.mark.parametrize("task", ["cvrp", "mis"])
@pytest.mark.parametrize("enabled", [False, True])
def test_plain_grpo_manager_retains_scalar_reward_and_zero_penalties(monkeypatch, task, enabled):
    import torch

    from clasp.rewards.primal_dual import PrimalDualConstrainedReward
    from clasp.trainers.primal_dual_grpo import compute_scalarized_group_advantages

    monkeypatch.setenv("GRPO_GRADED_REWARD", str(int(enabled)))
    if task == "cvrp":
        texts = [REFERENCE, "Routes: [[0, 1, 0]], Objective: 0", "broken"] * 3
        instance, answer = CVRP_INSTANCE, REFERENCE
    else:
        texts = ["Set: [0, 2]", "Set: [0, 1]", "broken"] * 3
        instance, answer = MIS_INSTANCE, "Set: [0, 2]"
    objective = getattr(rewards, task + "_objective_reward")
    kwargs = {"instance": [instance] * 9, "answer": [answer] * 9,
              "allow_infeasible_partial_credit": False}
    expected = objective([""] * 9, texts, **kwargs)
    manager = PrimalDualConstrainedReward(
        structure_fn=getattr(rewards, task + "_structure_reward"),
        constraint_fn=getattr(rewards, task + "_constraints_reward"), objective_fn=objective,
        cost_components_fn=getattr(rewards, task + "_cost_components"),
        objective_weight=1.0, structure_weight=0.0, cost_limit=0.0,
        lambda_init=0.0, dual_lr=0.0, log_every=0,
    )
    assert manager([""] * 9, texts, **kwargs) == expected
    components = manager.get_last_components()
    assert components["primal_scores"] == expected
    assert components["weighted_gap_scores"] == [0.0] * 9
    values = torch.tensor(expected)
    actual = compute_scalarized_group_advantages(values, torch.tensor(components["weighted_gap_scores"]),
                                                  9, std_correction=1)
    torch.testing.assert_close(actual, (values - values.mean()) / (values.std(correction=1) + 1e-4))
    if enabled:
        assert components["primal_scores"][2] == -1.0
        assert -1 < components["primal_scores"][1] < 0
