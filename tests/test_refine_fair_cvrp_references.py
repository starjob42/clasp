"""CVRP LKH serialization, strict parsing, failure retention, and exact sanity."""

import importlib.util
import itertools
import math
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/refine_fair_cvrp_references.py"
SPEC = importlib.util.spec_from_file_location("refine_fair_cvrp_references", SCRIPT)
refine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refine)
INSTANCE = [[[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 1, 1], 2]


def test_serialization_preserves_scaled_raw_distances_capacity_and_vehicle_budget():
    text = refine.serialize_cvrp(INSTANCE, name="tiny")
    assert "DIMENSION : 4\nCAPACITY : 2\nVEHICLES : 3" in text
    assert "EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX" in text
    matrix_text = text.split("EDGE_WEIGHT_SECTION\n")[1].split("DEMAND_SECTION")[0]
    matrix = [[int(v) for v in line.split()] for line in matrix_text.splitlines()]
    assert matrix == [[0, 1000, 1000, 1414], [1000, 0, 1414, 1000],
                      [1000, 1414, 0, 1000], [1414, 1000, 1000, 0]]
    assert "1 0\n2 1\n3 1\n4 1\nDEPOT_SECTION\n1\n-1\nEOF" in text


@pytest.mark.parametrize("instance", [
    [[[0, 0], [10**9, 0], [0, 1]], [0, 1, 1], 2],
    [[[0, 0], [1, 0], [0, 1]], [0, 0.5, 1], 2],
    [[[0, 0], [1, 0], [0, 1]], [0, 3, 1], 2],
])
def test_serialization_rejects_overflow_and_incompatible_demands(instance):
    with pytest.raises(ValueError):
        refine.serialize_cvrp(instance, name="bad")


SOLUTION = """tiny, Cost: 0_5414
The tours traveled by the 3 salesmen are:
1 2 4 1 (#2)  Cost: 3414
1 3 1 (#1)  Cost: 2000
1 1 (#0)  Cost: 0
"""


def test_solution_parser_handles_one_based_ids_and_empty_vehicles():
    assert refine.parse_lkh_routes(SOLUTION, node_count=4) == [[0, 1, 3, 0], [0, 2, 0]]


@pytest.mark.parametrize("bad", [
    SOLUTION.replace("0_5414", "1_5414"),
    SOLUTION.replace("3 salesmen", "2 salesmen"),
    SOLUTION.replace("1 2 4 1", "1 2 5 1"),
    SOLUTION.replace("(#2)", "(#3)"),
    SOLUTION.replace("1 1 (#0)  Cost: 0\n", ""),
])
def test_solution_parser_rejects_invalid_or_incomplete_tours(bad):
    with pytest.raises(ValueError):
        refine.parse_lkh_routes(bad, node_count=4)


def parent_reference():
    routes = [[0, 1, 0], [0, 2, 0], [0, 3, 0]]
    objective = 4 + 2 * math.sqrt(2)
    row = {"instance_json": INSTANCE, "instruction": "fixed", "input": "fixed",
           "output": f"Routes: {routes}, Objective: {objective}"}
    return row, {"objective": objective, "reference_routes": routes}


def test_failed_solver_retains_exact_parent_output_and_all_fields(tmp_path):
    binary = tmp_path / "failing_solver"
    binary.write_text("#!/bin/sh\nexit 3\n")
    binary.chmod(0o700)
    params = tmp_path / "run.par"
    params.write_text("unused")
    candidate = refine.solve_candidate(binary, params, tmp_path / "missing.sol",
                                       tmp_path / "failed.log", INSTANCE, seconds=0.1)
    candidate["seed"] = 101
    assert candidate["returncode"] == 3
    assert candidate["strict_verifier_feasible"] is False
    row, parent = parent_reference()
    actual, evidence = refine.choose_reference(row, parent, [candidate])
    assert actual == row
    assert evidence["reference_selected_from"] == "parent"
    assert evidence["refinement_improvement"] == 0


def test_reference_selection_uses_raw_euclidean_cost_and_preserves_prompt_fields():
    routes = [[0, 1, 3, 0], [0, 2, 0]]
    raw_cost = 4 + math.sqrt(2)
    candidate = {"seed": 101, "strict_verifier_feasible": True,
                 "raw_euclidean_objective": raw_cost, "reference_routes": routes}
    row, parent = parent_reference()
    actual, evidence = refine.choose_reference(row, parent, [candidate])
    assert evidence["objective"] == raw_cost
    assert {k: v for k, v in actual.items() if k != "output"} == {
        k: v for k, v in row.items() if k != "output"}
    assert refine._parse_cvrp_solution(actual["output"])["routes"] == routes


@pytest.mark.skipif(not os.environ.get("CLASP_TEST_LKH_BINARY"), reason="explicit real LKH binary required")
def test_real_lkh_tiny_cvrp_matches_exhaustively_enumerated_exact_cost(tmp_path):
    # Enumerate every customer permutation and route break pattern for this
    # three-customer problem; do not use LKH's rounded objective as ground truth.
    exact = math.inf
    for order in itertools.permutations(range(1, 4)):
        for cuts in itertools.product((False, True), repeat=2):
            routes, route = [], [0, order[0]]
            for node, cut in zip(order[1:], cuts):
                if cut:
                    routes.append(route + [0])
                    route = [0]
                route.append(node)
            routes.append(route + [0])
            if refine._check_cvrp_constraints(routes, *INSTANCE)[2]:
                exact = min(exact, refine._cvrp_total_distance(INSTANCE[0], routes))
    problem, parameter = tmp_path / "tiny.vrp", tmp_path / "tiny.par"
    solution, log = tmp_path / "tiny.sol", tmp_path / "tiny.log"
    problem.write_text(refine.serialize_cvrp(INSTANCE, name="tiny"))
    parameter.write_text(refine.parameter_text(problem, solution, seed=101, seconds=0.2))
    result = refine.solve_candidate(Path(os.environ["CLASP_TEST_LKH_BINARY"]), parameter,
                                    solution, log, INSTANCE, seconds=0.2)
    assert result["strict_verifier_feasible"], log.read_text()
    assert result["raw_euclidean_objective"] == pytest.approx(exact, abs=1e-12)
    assert exact == pytest.approx(4 + math.sqrt(2))
