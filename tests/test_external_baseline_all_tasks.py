"""Task integration, native feasibility, and bounded graded-signal checks."""
from __future__ import annotations

import itertools
import json
import os

import pytest

from clasp.baselines.external_baseline_dev import (
    ExternalBaselineRuntime, TASK_GRADED_REWARDS, build_runtime, seal_reward_environment,
)
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_external_baseline_grpo import build_parser
from clasp.rewards import co_tasks as co
from clasp.rewards.external_baseline import (
    ExternalBaselineTracker, SUPPORTED_CO_TASKS, baseline_instance_size,
)
from clasp.rewards.external_baseline_tasks import validate_graded_instance


COORDS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
INSTANCES = {
    "tsp": COORDS,
    "cvrp": [COORDS, [0, 1, 1], 2],
    "op": [COORDS, [0, 2, 1], 4],
    "mis": {"num_nodes": 3, "edges": [[0, 1]]},
    "jssp": [[0, 1, 1, 1], [1, 1, 0, 1]],
}
SOLUTIONS = {
    "tsp": "Route: [0, 1, 2, 0], Objective: 3.414",
    "cvrp": "Routes: [[0, 1, 2, 0]], Objective: 3.414",
    "op": "Route: [0, 1, 2], Objective: 3",
    "mis": "Set: [0, 2], Objective: 2",
    "jssp": "Schedule: [[0, 1], [0, 1]], Makespan: 4",
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items()
                                       if not k.startswith(("CLASP_", "GRPO_", "PD_", "FOARL_"))})
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("GRPO_DATA_SEED", "43")
    monkeypatch.setenv("GRPO_STRICT_OPTIMIZER_RESUME", "1")
    monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", "50")
    seal_reward_environment()


def score(task, completions, *, instance=None, answer=None):
    n = len(completions)
    return TASK_GRADED_REWARDS[task](
        ["solve"] * n, completions, task_type=[task] * n,
        instance_json=[json.dumps(INSTANCES[task] if instance is None else instance)] * n,
        input=["instance"] * n, answer=[SOLUTIONS[task] if answer is None else answer] * n,
    )


@pytest.mark.parametrize("task", SUPPORTED_CO_TASKS)
@pytest.mark.parametrize("estimator,g", [("external", 1), ("external", 9), ("rloo", 9)])
def test_all_fifteen_profile_arms_share_recipe_and_route_to_correct_head(task, estimator, g):
    args = finalize_grpo_args(build_parser().parse_args([
        "--tasks", task, "--sft_checkpoint_path", f"/synthetic/{task}",
        "--external_dev_source_manifest", "/synthetic/SOURCE_MANIFEST.sha256",
        "--external_dev_source_manifest_sha256", "a" * 64,
        "--advantage_estimator", estimator, "--num_generations", str(g),
    ]))
    runtime = build_runtime({"status": "passed_frozen_manifest_successor_source"}, args)
    runtime.validate_args(args)
    assert runtime.reward_functions() == (TASK_GRADED_REWARDS[task],)
    metadata = runtime.metadata()
    assert metadata["supported_tasks"] == list(SUPPORTED_CO_TASKS)
    assert metadata["generation_batch_size"] == 36
    assert metadata["baseline"]["current_batch_centering"] is False
    assert (args.max_steps, args.grpo_loss_type, args.grpo_beta) == (100, "dr_grpo", 0)


@pytest.mark.parametrize("task,expected", [("tsp", 3), ("cvrp", 2), ("op", 3), ("mis", 3), ("jssp", 4)])
def test_size_is_input_only_and_uses_correct_units(task, expected):
    row = {"instance_json": json.dumps(INSTANCES[task]), "answer": "ignored", "input": "ignored"}
    assert baseline_instance_size(task, row) == expected
    row.update(answer="an answer describing 999 nodes", expected_reward=999, completion="wrong")
    assert baseline_instance_size(task, row) == expected
    tracker = ExternalBaselineTracker()
    snapshot = tracker.snapshot([(task, baseline_instance_size(task, row))])
    assert snapshot.values == (0.0,)


@pytest.mark.parametrize("task,value", [("tsp", None), ("cvrp", [[], [], 1]),
                                        ("op", [[], [], 1]), ("mis", {"num_nodes": True}),
                                        ("jssp", [[0, 1, 2]])])
def test_missing_or_bad_sizes_fail_closed(task, value):
    with pytest.raises(ValueError):
        baseline_instance_size(task, {"instance_json": value})


@pytest.mark.parametrize("task,expected", [("tsp", 1.5), ("cvrp", 1.5), ("op", 2),
                                          ("mis", 1 + 2 / 3), ("jssp", 1.5)])
def test_feasible_reward_and_parse_failure_bounds(task, expected):
    assert score(task, [SOLUTIONS[task], "cannot solve"]) == pytest.approx([expected, -1])


def test_tsp_grades_missing_duplicate_out_of_range_and_boundary_defects():
    values = score("tsp", ["Route: [0, 1, 0]", "Route: [0, 1, 1, 2, 0]",
                           "Route: [0, 1, 2, 9, 0]", "Route: [1, 2, 0, 1]"])
    assert values == pytest.approx([-0.25] * 4)
    # A better valid tour gets more reward, using recomputed geometry only.
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    reference = "Route: [0, 1, 3, 2, 0], Objective: 1"
    assert score("tsp", ["Route: [0, 1, 2, 3, 0], Objective: 999"],
                 instance=square, answer=reference)[0] > 1.5


def test_op_budget_and_structure_are_gated_and_not_replaced_by_prize():
    instance = [COORDS, [0, 2, 1], 1.5]
    values = score("op", ["Route: [0, 1]", "Route: [0, 1, 2]", "Route: [0, 1, 1]",
                          "Route: [1, 2]", "Route: [0]", "Route: [0, 9]"], instance=instance)
    assert values[0] == pytest.approx(1 + 2 / 3)
    assert all(-1 < value < 0 for value in values[1:])
    distance = co._distance(COORDS, [0, 1, 2])
    severity = (distance - 1.5) / 1.5
    assert values[1] == pytest.approx(-severity / (1 + severity))
    bigger_prize = [COORDS, [0, 200, 100], 1.5]
    assert score("op", ["Route: [0, 1, 2]"], instance=bigger_prize)[0] == values[1]
    assert score("op", ["Route: [0, 1]"], instance=[COORDS, [0, 0, 0], 4]) == [1.0]


def test_jssp_detects_precedence_cycle_and_recomputes_improved_makespan():
    values = score("jssp", ["Schedule: [[1, 0], [0, 1]]", "Schedule: [[0, 1]]",
                            "Schedule: [[0, 0], [0, 1]]", "Schedule: [[0, 2], [0, 1]]",
                            "Schedule: [[0, 1], [1, 0]], Makespan: 999"])
    assert values[0] == pytest.approx(-0.5)  # all four operations belong to a cycle
    assert values[1] == pytest.approx(-1 / 3)  # one of two machine rows missing
    assert all(-1 < value < 0 for value in values[2:4])
    assert values[4] == pytest.approx(1 + 4 / 6)


@pytest.mark.parametrize("task", ["tsp", "op"])
def test_exhaustive_short_routes_agree_with_native_strict_feasibility(task):
    routes = [list(route) for length in range(6)
              for route in itertools.product((-1, 0, 1, 2, 3), repeat=length)]
    values = score(task, [f"Route: {route}" for route in routes])
    for route, reward in zip(routes, values):
        feasible = (co._check_tsp_constraints(route, 3)[2] if task == "tsp"
                    else co._check_op_constraints(route, COORDS, 4)[2])
        assert -1 <= reward <= 2
        assert (reward >= 1) == feasible, (task, route, reward)


def test_exhaustive_small_machine_orders_agree_with_native_jssp_feasibility():
    orders = [list(order) for length in range(1, 4)
              for order in itertools.product((-1, 0, 1, 2), repeat=length)]
    schedules = [[a, b] for a in orders for b in orders]
    jobs = validate_graded_instance("jssp", INSTANCES["jssp"])
    values = score("jssp", [f"Schedule: {schedule}" for schedule in schedules])
    for schedule, reward in zip(schedules, values):
        assert -1 <= reward <= 2
        assert (reward >= 1) == co._check_jssp_constraints(jobs, schedule)[2]


@pytest.mark.parametrize("task,instance", [
    ("tsp", [[0, 0], [float("nan"), 1]]),
    ("op", [COORDS, [0, -1, 1], 4]), ("op", [COORDS, [0, 1, 1], 0]),
    ("jssp", [[0, 1, 0, 1], [1, 1, 0, 1]]),
    ("jssp", [[0, 1.5, 1, 1], [1, 1, 0, 1]]),
])
def test_bad_instance_metadata_rejected_even_for_invalid_output(task, instance):
    with pytest.raises(ValueError):
        score(task, ["invalid"], instance=instance)


@pytest.mark.parametrize("task", ["tsp", "jssp"])
def test_bad_reference_rejected_only_when_needed_for_feasible_quality(task):
    assert score(task, ["invalid"], answer="invalid") == [-1.0]
    with pytest.raises(ValueError, match="feasible reference"):
        score(task, [SOLUTIONS[task]], answer="invalid")


class Rows(list):
    @property
    def column_names(self):
        return list(self[0])


@pytest.mark.parametrize("task", SUPPORTED_CO_TASKS)
def test_all_task_preflight_validates_input_size_and_metadata(task):
    row = {"prompt": "solve", "answer": SOLUTIONS[task], "input": "instance",
           "task_type": task, "instance_json": json.dumps(INSTANCES[task])}
    runtime = ExternalBaselineRuntime(task, "external", 1, diagnostic=True)
    assert runtime.validate_dataset(Rows([row] * 36), label="train")["status"] == "PASS"
    with pytest.raises(ValueError, match="Too few retained"):
        runtime.validate_dataset(Rows([row] * 35), label="train")


def test_task_mismatch_and_missing_aligned_metadata_fail_closed():
    with pytest.raises(ValueError, match="one instance_json"):
        TASK_GRADED_REWARDS["tsp"](["p"], ["invalid"], task_type=["tsp"], answer=["answer"])
    with pytest.raises(ValueError, match="different task"):
        TASK_GRADED_REWARDS["tsp"](["p"], ["invalid"], task_type=["op"], answer=["answer"],
                                   instance_json=[json.dumps(COORDS)], input=["input"])
