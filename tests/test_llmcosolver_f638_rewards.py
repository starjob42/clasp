from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from clasp.baselines import llmcosolver_f638_rewards as port

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/llmcosolver_f6388991"


def _load_vendored_oracle():
    previous_utils = sys.modules.get("utils")
    try:
        utils_spec = importlib.util.spec_from_file_location("utils", VENDOR / "utils.py")
        assert utils_spec is not None and utils_spec.loader is not None
        utils_module = importlib.util.module_from_spec(utils_spec)
        sys.modules["utils"] = utils_module
        utils_spec.loader.exec_module(utils_module)

        rewards_spec = importlib.util.spec_from_file_location(
            "_llmcosolver_f638_official_rewards", VENDOR / "rewards.py"
        )
        assert rewards_spec is not None and rewards_spec.loader is not None
        rewards_module = importlib.util.module_from_spec(rewards_spec)
        rewards_spec.loader.exec_module(rewards_module)
        return rewards_module
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils


ORACLE = _load_vendored_oracle()


CASES = {
    "tsp": {
        "completions": [
            "Route: [0, 1, 2, 3, 0], Objective: 4.0",
            "Route: [0, 1, 2, 3]",
            "Route: [9, 10, 11, 12, 9]",
            "not a route",
        ],
        "ground_truth": ["Route: [0, 1, 2, 3, 0], Objective: 4.000"] * 4,
        "instance": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
        * 4,
    },
    "cvrp": {
        "completions": [
            "Routes: [[0, 1, 2, 0]], Objective: 3.4",
            "Routes: [[0, 1, 1, 2, 0]]",
            "Routes: [[0, 1, 0]]",
            "unparseable",
        ],
        "ground_truth": ["Routes: [[0, 1, 2, 0]], Objective: 3.414"] * 4,
        "instance_coords": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]] * 4,
        "instance_demands": [[0, 1, 1]] * 4,
        "instance_capacity": [10] * 4,
    },
    "op": {
        "completions": [
            "Route: [0, 1, 2], Objective: 10",
            "Route: [0, 1, 0]",
            "Route: [1, 2]",
            "bad",
        ],
        "ground_truth": ["Route: [0, 1, 2], Objective: 10.0"] * 4,
        "instance_coords": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]] * 4,
        "instance_max_dist": [4.0] * 4,
        "instance_prizes": [[0.0, 5.0, 5.0]] * 4,
    },
    "mvc": {
        "completions": ["Response: [1]", "Response: [99]", "Response: [0, 2]", "bad"],
        "ground_truth": ["Response: [1], Objective: 1"] * 4,
        "instance": [{"edges": [(0, 1), (1, 2)]}] * 4,
    },
    "mis": {
        "completions": [
            "Response: [0, 2]",
            "Response: [0, 1]",
            "Response: [99, 100]",
            "bad",
        ],
        "ground_truth": ["Response: [0, 2], Objective: 2"] * 4,
        "instance": [{"edges": [(0, 1), (1, 2)]}] * 4,
    },
    "jssp": {
        "completions": [
            "Schedule: [[0, 1], [1, 0]], Makespan: 2",
            "Schedule: [[1, 0], [0, 1]], Makespan: 999",
            "Schedule: [[0, 1]], Makespan: 2",
            "bad",
        ],
        "ground_truth": ["Schedule: [[0, 1], [1, 0]], Makespan: 2"] * 4,
        "instance": [
            [[0, 1, 1, 1], [1, 1, 0, 1]],
        ]
        * 4,
    },
}


def test_all_six_reward_pairs_are_exactly_differential_to_vendored_official_source():
    assert list(port.OFFICIAL_REWARD_HEADS) == ["cvrp", "op", "tsp", "mvc", "mis", "jssp"]
    for task, kwargs in CASES.items():
        port_optimality, port_feasibility = port.OFFICIAL_REWARD_HEADS[task]
        oracle_optimality = getattr(ORACLE, f"optimality_reward_func_{task}")
        oracle_feasibility = getattr(ORACLE, f"feasibility_reward_func_{task}")
        np.testing.assert_allclose(
            port_feasibility(**kwargs),
            oracle_feasibility(**kwargs),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            port_optimality(**kwargs),
            oracle_optimality(**kwargs),
            rtol=0.0,
            atol=0.0,
        )


def test_representative_upstream_quirks_are_locked_not_silently_repaired():
    tsp_opt, tsp_feas = port.OFFICIAL_REWARD_HEADS["tsp"]
    assert tsp_feas(**CASES["tsp"]) == [1.0, 0.7, 1.0, 0.0]
    assert tsp_opt(**CASES["tsp"])[1:] == [0.06999999999999999, 0.0, 0.0]

    cvrp_opt, cvrp_feas = port.OFFICIAL_REWARD_HEADS["cvrp"]
    assert cvrp_feas(**CASES["cvrp"]) == [1.0, 1.0, 0.9500000000000001, 0.0]
    assert cvrp_opt(**CASES["cvrp"])[2:] == [0.0, 0.0]

    op_opt, op_feas = port.OFFICIAL_REWARD_HEADS["op"]
    assert op_feas(**CASES["op"]) == [1.0, 0.8, 0.9, 0.0]
    assert op_opt(**CASES["op"])[0] == 2.0

    mis_opt, mis_feas = port.OFFICIAL_REWARD_HEADS["mis"]
    assert mis_feas(**CASES["mis"]) == [1.0, 0.2, 1.0, 0.0]
    assert mis_opt(**CASES["mis"])[2] == 1.0

    jssp_opt, jssp_feas = port.OFFICIAL_REWARD_HEADS["jssp"]
    assert jssp_feas(**CASES["jssp"]) == [1.0, 0.6000000000000001, 0.2, 0.0]
    assert jssp_opt(**CASES["jssp"])[0] == 3.0


def test_port_is_explicitly_pinned_and_never_used_as_publication_evaluator():
    assert port.UPSTREAM_COMMIT == "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
    assert port.UPSTREAM_REWARDS_SHA256 == (
        "879f73a0b4f6ae5d4227a8d4ce8dbf1aa5cc966b526c6d76bba5a2c29a2380c5"
    )
    assert "strict evaluator" in (port.__doc__ or "")
