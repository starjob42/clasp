"""Exercise the actual launch-time reference precheck with both reward modes."""
import json
import os

import pytest

from clasp.rewards import heads


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("CLASP_", "PD_")) or name == "GRPO_GRADED_REWARD":
            monkeypatch.delenv(name)


def reference_row(task):
    if task == "cvrp":
        instance = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.0, 1.0, 1.0], 2.0]
        output = "Routes: [[0, 1, 2, 0]], Objective: 999999"
    else:
        instance = {"num_nodes": 4, "edges": [[0, 1], [1, 2]]}
        output = "Set: [0, 2], Objective: 999999"
    return {"instruction": "Solve the instance.", "input": "Use the attached instance.", "output": output,
            "instance_json": json.dumps(instance)}


def write_rows(root, task, rows):
    path = root / task / "train" / "train_rl.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def verify(root, tasks):
    heads.verify_ground_truth_rewards(tasks=tasks, data_root=str(root), max_samples_per_task=5,
                                      prefer_subset_files=False, no_reason=True)


@pytest.mark.parametrize("mode", [None, "0", "1"])
def test_actual_reference_precheck_passes_both_tasks(tmp_path, monkeypatch, capsys, mode):
    if mode is not None:
        monkeypatch.setenv("GRPO_GRADED_REWARD", mode)
    for task in ("cvrp", "mis"):
        write_rows(tmp_path, task, [reference_row(task)])
    verify(tmp_path, ["cvrp", "mis"])
    message = capsys.readouterr().out
    assert "Checked=2" in message
    assert f"GRPO_GRADED_REWARD={int(mode == '1')}" in message


@pytest.mark.parametrize("task", ["cvrp", "mis"])
@pytest.mark.parametrize("mode,wrong_score", [("0", 1.5), ("1", 2.0)])
def test_expected_value_is_independent_of_objective_head(tmp_path, monkeypatch, task, mode, wrong_score):
    monkeypatch.setenv("GRPO_GRADED_REWARD", mode)
    write_rows(tmp_path, task, [reference_row(task)])
    triplet = heads.TASK_REWARD_TRIPLETS[task]
    monkeypatch.setitem(heads.TASK_REWARD_TRIPLETS, task,
                        (triplet[0], triplet[1], lambda *args, **kwargs: [wrong_score]))
    with pytest.raises(RuntimeError, match="Ground-truth reward verification failed"):
        verify(tmp_path, [task])


def test_graded_mis_uses_transformed_verified_cardinality(tmp_path, monkeypatch):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    write_rows(tmp_path, "mis", [reference_row("mis")])
    transform = heads.transform_example_for_task

    def transformed(*args):
        row = transform(*args)
        row["instance_json"] = {"num_nodes": 8, "edges": [[0, 1], [1, 2]]}
        row["output"] = "Set: [0, 2, 3], Objective: 999999"
        return row

    monkeypatch.setattr(heads, "transform_example_for_task", transformed)
    # The new graph/output gives 1 + 3/8; raw input would give 1 + 2/4.
    verify(tmp_path, ["mis"])


@pytest.mark.parametrize("task", ["cvrp", "mis"])
@pytest.mark.parametrize("head_index", [0, 1])
def test_graded_mode_preserves_structure_and_constraint_checks(tmp_path, monkeypatch, task, head_index):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    write_rows(tmp_path, task, [reference_row(task)])
    triplet = list(heads.TASK_REWARD_TRIPLETS[task])
    triplet[head_index] = lambda *args, **kwargs: [0.0]
    monkeypatch.setitem(heads.TASK_REWARD_TRIPLETS, task, tuple(triplet))
    with pytest.raises(RuntimeError, match="Ground-truth reward verification failed"):
        verify(tmp_path, [task])


@pytest.mark.parametrize("output", ["Set: [0, 1]", "Set: []", "Set: [0, 0]", "Set: [8]"])
def test_graded_mis_rejects_invalid_ground_truth(tmp_path, monkeypatch, output):
    monkeypatch.setenv("GRPO_GRADED_REWARD", "1")
    row = reference_row("mis")
    row["output"] = output
    write_rows(tmp_path, "mis", [row])
    with pytest.raises(RuntimeError, match="invalid MIS reference"):
        verify(tmp_path, ["mis"])


@pytest.mark.parametrize("mode", ["", "true", "2"])
def test_precheck_validates_strict_mode_even_without_rows(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GRPO_GRADED_REWARD", mode)
    with pytest.raises(ValueError, match="GRPO_GRADED_REWARD"):
        verify(tmp_path, [])


@pytest.mark.parametrize("mode", ["0", "1"])
def test_tsp_target_bonus_expectation_is_unchanged(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GRPO_GRADED_REWARD", mode)
    monkeypatch.setenv("CLASP_TSP_TARGET_GAP", "0.1")
    monkeypatch.setenv("CLASP_TSP_TARGET_GAP_BONUS", "0.25")
    row = {"instruction": "Solve.", "input": "Two cities.", "output": "Tour: [0, 1, 0]"}
    write_rows(tmp_path, "tsp", [row])
    monkeypatch.setitem(heads.TASK_REWARD_TRIPLETS, "tsp", (
        lambda *args, **kwargs: [1.0], lambda *args, **kwargs: [1.0],
        lambda *args, **kwargs: [2.5],
    ))
    verify(tmp_path, ["tsp"])
