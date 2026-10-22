from __future__ import annotations

import json

import numpy as np
import pytest

from clasp.baselines.llmcosolver_f638_rewards import OFFICIAL_REWARD_HEADS
from clasp.baselines.llmcosolver_f638_training import (
    FOARL_F638_REWARD_FUNCS,
    FOARL_F638_REWARD_HEAD_ORDER,
    adapt_foarl_f638_row,
    foarl_f638_feasibility_reward,
    foarl_f638_optimality_reward,
    validate_foarl_f638_dataset,
)


ROWS = {
    "tsp": {
        "task_type": "tsp",
        "answer": "Route: [0, 1, 2, 3, 0], Objective: 4.000",
        "instance_json": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "completion": "Route: [0, 1, 2, 3, 0], Objective: 4.0",
    },
    "cvrp": {
        "task_type": "cvrp",
        "answer": "Routes: [[0, 1, 2, 0]], Objective: 3.414",
        "instance_json": [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [0, 1, 1],
            10,
        ],
        "completion": "Routes: [[0, 1, 2, 0]], Objective: 3.4",
    },
    "op": {
        "task_type": "op",
        "answer": "Route: [0, 1, 2], Objective: 10.0",
        "instance_json": [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [0.0, 5.0, 5.0],
            4.0,
        ],
        "completion": "Route: [0, 1, 2], Objective: 10",
    },
    "mis": {
        "task_type": "mis",
        "answer": "Response: [0, 2], Objective: 2",
        "instance_json": {"num_nodes": 3, "edges": [[0, 1], [1, 2]]},
        "completion": "Response: [0, 2]",
    },
    "jssp": {
        "task_type": "jssp",
        "answer": "Schedule: [[0, 1], [1, 0]], Makespan: 2",
        "instance_json": [[0, 1, 1, 1], [1, 1, 0, 1]],
        "completion": "Schedule: [[0, 1], [1, 0]], Makespan: 2",
    },
}


def _reward_kwargs(task: str, row: dict) -> dict:
    adapted = adapt_foarl_f638_row(row)
    return {
        "completions": [row["completion"]],
        "ground_truth": [adapted.ground_truth],
        **{key: [value] for key, value in adapted.reward_kwargs.items()},
    }


def test_adapter_maps_all_five_publication_tasks_to_exact_upstream_keyword_schema():
    assert set(ROWS) == {"tsp", "cvrp", "op", "mis", "jssp"}
    expected_keys = {
        "tsp": {"instance"},
        "cvrp": {"instance_coords", "instance_demands", "instance_capacity"},
        "op": {"instance_coords", "instance_prizes", "instance_max_dist"},
        "mis": {"instance"},
        "jssp": {"instance"},
    }
    for task, raw_row in ROWS.items():
        row = dict(raw_row)
        row["instance_json"] = json.dumps(row["instance_json"])
        adapted = adapt_foarl_f638_row(row)
        assert adapted.task_type == task
        assert adapted.ground_truth == row["answer"]
        assert set(adapted.reward_kwargs) == expected_keys[task]


def test_ordered_mixed_wrappers_are_exactly_differential_to_each_official_task_pair():
    task_order = ["mis", "tsp", "op", "cvrp", "jssp", "tsp"]
    rows = [ROWS[task] for task in task_order]
    kwargs = {
        "task_type": [row["task_type"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "instance_json": [json.dumps(row["instance_json"]) for row in rows],
    }
    completions = [row["completion"] for row in rows]
    prompts = [f"prompt-{idx}" for idx in range(len(rows))]

    actual_optimality = foarl_f638_optimality_reward(prompts, completions, **kwargs)
    actual_feasibility = foarl_f638_feasibility_reward(prompts, completions, **kwargs)
    expected_optimality = []
    expected_feasibility = []
    for task, row in zip(task_order, rows):
        optimality, feasibility = OFFICIAL_REWARD_HEADS[task]
        expected_optimality.extend(optimality(**_reward_kwargs(task, row)))
        expected_feasibility.extend(feasibility(**_reward_kwargs(task, row)))

    np.testing.assert_allclose(actual_optimality, expected_optimality, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_feasibility, expected_feasibility, rtol=0.0, atol=0.0)
    assert FOARL_F638_REWARD_HEAD_ORDER == ("optimality", "feasibility")
    assert FOARL_F638_REWARD_FUNCS == (
        foarl_f638_optimality_reward,
        foarl_f638_feasibility_reward,
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"task_type": "mvc", "answer": "x", "instance_json": {}}, "task_type"),
        ({"task_type": "tsp", "answer": "", "instance_json": [[0, 0], [1, 1]]}, "answer"),
        ({"task_type": "tsp", "answer": "x"}, "instance_json"),
        (
            {"task_type": "cvrp", "answer": "x", "instance_json": [[[0, 0], [1, 1]], [0], 2]},
            "demands",
        ),
        (
            {"task_type": "op", "answer": "x", "instance_json": [[[0, 0], [1, 1]], [0, 1], 0]},
            "max_distance",
        ),
        ({"task_type": "mis", "answer": "x", "instance_json": {"nodes": 2}}, "edges"),
        ({"task_type": "jssp", "answer": "x", "instance_json": [[0, 1], [1, 2, 0, 2]]}, "rectangular"),
        ({"task_type": "tsp", "answer": "x", "instance_json": "{bad"}, "valid JSON"),
    ],
)
def test_adapter_fails_closed_on_unsupported_or_malformed_rows(row, message):
    with pytest.raises(ValueError, match=message):
        adapt_foarl_f638_row(row)


def test_reward_wrapper_fails_closed_on_column_cardinality_and_non_text_completion():
    row = ROWS["tsp"]
    kwargs = {
        "task_type": ["tsp"],
        "answer": [row["answer"]],
        "instance_json": [json.dumps(row["instance_json"])],
    }
    with pytest.raises(ValueError, match="one 'answer'"):
        foarl_f638_optimality_reward(["p"], [row["completion"]], **{**kwargs, "answer": []})
    with pytest.raises(ValueError, match=r"completion\[0\] must be plain text"):
        foarl_f638_optimality_reward(["p"], [{"content": row["completion"]}], **kwargs)


def test_reward_wrapper_rejects_nonfinite_or_wrong_cardinality_upstream_output(monkeypatch):
    row = ROWS["tsp"]
    kwargs = {
        "task_type": ["tsp"],
        "answer": [row["answer"]],
        "instance_json": [json.dumps(row["instance_json"])],
    }

    def nonfinite(**_kwargs):
        return [float("nan")]

    def wrong_length(**_kwargs):
        return []

    _original_optimality, original_feasibility = OFFICIAL_REWARD_HEADS["tsp"]
    monkeypatch.setitem(OFFICIAL_REWARD_HEADS, "tsp", (nonfinite, original_feasibility))
    with pytest.raises(RuntimeError, match="not finite"):
        foarl_f638_optimality_reward(["p"], [row["completion"]], **kwargs)
    monkeypatch.setitem(OFFICIAL_REWARD_HEADS, "tsp", (wrong_length, original_feasibility))
    with pytest.raises(RuntimeError, match="returned 0 values"):
        foarl_f638_optimality_reward(["p"], [row["completion"]], **kwargs)


class _TinyDataset:
    column_names = ["prompt", "answer", "instance_json", "task_type"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_preflight_validates_every_prepared_row_and_receipts_task_counts():
    rows = [
        {
            "prompt": f"prompt-{task}",
            "answer": row["answer"],
            "instance_json": json.dumps(row["instance_json"]),
            "task_type": task,
        }
        for task, row in ROWS.items()
    ]
    receipt = validate_foarl_f638_dataset(_TinyDataset(rows), label="train")
    assert receipt == {
        "label": "train",
        "rows": 5,
        "task_counts": {"tsp": 1, "cvrp": 1, "op": 1, "mis": 1, "jssp": 1},
        "status": "PASS",
    }

