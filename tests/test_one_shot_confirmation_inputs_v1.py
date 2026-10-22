"""CPU-only freshness and raw physical-witness guards; no policy imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_one_shot_20261016/prepare_confirmation_inputs_v1.py"
SPEC = importlib.util.spec_from_file_location("prospective_op_input_preparer", PATH)
prepare = importlib.util.module_from_spec(SPEC)
prepare.__executed_source_sha256__ = hashlib.sha256(PATH.read_bytes()).hexdigest()
SPEC.loader.exec_module(prepare)


@pytest.fixture(scope="module")
def actual_inputs():
    generator, sources = prepare.closed_engine()
    excluded, exclusions, original = prepare.load_exclusions(generator)
    generated = generator.build_task_cohorts("op", seed=prepare.SEED, train_count=3,
                                              validation_count=256, excluded_rows=excluded)
    return generator, sources, excluded, exclusions, original, generated


def test_all256_physical_witnesses_and_original_builder_replay(actual_inputs):
    generator, _, excluded, _, _, value = actual_inputs
    admission, proofs = prepare.verify_inputs(value["training_rows"], value["validation_rows"], excluded, generator)
    assert len(proofs) == 256 and all(row["strictly_feasible"] for row in proofs)
    assert all(row["route_distance"] <= row["route_budget"] + 1e-9 for row in proofs)
    assert all(row["recomputed_prize"] > 0 and row["optimality_certified"] is False for row in proofs)
    assert admission["semantic_overlap_count"] == 0 and admission["forbidden_instance_count"] == 5550
    assert value == generator.build_task_cohorts("op", seed=2026101603, train_count=3,
                                                validation_count=256, excluded_rows=excluded)
    assert value["generation"]["draws"] == 259
    assert value["generation"]["rejected_reused_graphs"] == 0
    assert [sum(row["development_band"] == list(band) for row in value["validation_rows"])
            for band in generator.BANDS] == [86, 85, 85]


def test_bound_all_original_and_current_exclusion_inputs_and_source_buffers(actual_inputs):
    generator, sources, excluded, exclusions, original, value = actual_inputs
    assert original["sha256"] == prepare.ADMISSION_SHA
    assert len(exclusions) == 8 and sum(row["input_rows"] for row in exclusions) == 5550
    assert [row["input_rows"] for row in exclusions] == [5000, 96, 128, 96, 3, 3, 128, 96]
    assert sources["closed_project_python_files"] == 107
    assert sources["all_project_imports_execute_verified_raw_buffers"] is True
    assert sources["original_generator"]["sha256"] == prepare.GENERATOR_SHA
    assert sources["original_one_shot_cohort_preparer"]["sha256"] == prepare.PREDECESSOR_SHA
    assert not any(name in sys.modules for name in prepare.MODEL_MODULES)
    forbidden = {generator.graph_semantic_id("op", row) for row in excluded}
    assert len(forbidden) == 5550
    assert not forbidden & {row["graph_semantic_id"] for row in value["validation_rows"]}
    assert {row["generation_purpose"] for row in value["validation_rows"]} == {"fresh_development_pilot_20261013"}


def test_reject_reused_graph_with_changed_budget_and_label(actual_inputs):
    generator, _, excluded, _, _, value = actual_inputs
    row = copy.deepcopy(value["validation_rows"][0])
    row["instance_json"][2] *= 2
    row["instance_id"] = "different_label_and_budget"
    with pytest.raises(ValueError, match="graph overlaps"):
        generator.validate_cohorts(value["training_rows"], [row], task="op",
                                   forbidden_rows=[value["validation_rows"][0]])


def test_reject_reused_graph_with_permuted_customer_numbers(actual_inputs):
    generator, _, _, _, _, value = actual_inputs
    from clasp.baselines.sft_anchor_pilot import parse_solution
    old = value["validation_rows"][0]
    row = copy.deepcopy(old)
    row["instance_json"][0][1:] = reversed(row["instance_json"][0][1:])
    row["instance_json"][1][1:] = reversed(row["instance_json"][1][1:])
    size = len(row["instance_json"][0])
    route = [0 if node == 0 else size - node for node in parse_solution("op", old["output"])]
    row["output"] = "Route: " + repr(route)
    row["instance_id"] = "new_numbering"
    row["instruction"], row["input"] = generator.render_instance("op", row["instance_json"])
    row["prompt"] = generator.build_jiang_f638_inference_prompt(row["instruction"], row["input"])
    assert generator.graph_semantic_id("op", row) == generator.graph_semantic_id("op", old)
    assert generator.score_completion("op", row["output"], row["instance_json"], row["output"])["feasible"]
    with pytest.raises(ValueError, match="graph overlaps"):
        generator.validate_cohorts(value["training_rows"], [row], task="op", forbidden_rows=[old])


def test_duplicate_semantic_graph_is_not_saved_as_fresh(actual_inputs):
    generator, _, excluded, _, _, value = actual_inputs
    rows = copy.deepcopy(value["validation_rows"])
    rows[1] = copy.deepcopy(rows[0]); rows[1]["instance_id"] = "new_label"
    with pytest.raises(ValueError, match="duplicate mathematical"):
        prepare.verify_inputs(value["training_rows"], rows, excluded, generator)


@pytest.mark.parametrize("field,value", [
    ("feasibility_independently_verified", False),
    ("optimality_proven", True),
    ("declared_objective_used", True),
    ("negative_policy_gaps_allowed", False),
    ("objective", -100),
])
def test_scope_and_optimality_tampering_rejected(actual_inputs, field, value):
    generator, _, _, _, _, cohort = actual_inputs
    row = copy.deepcopy(cohort["validation_rows"][0])
    row["reference_evidence"][field] = value
    with pytest.raises(ValueError, match="heuristic scope"):
        prepare.physical_proof(row, generator)


@pytest.mark.parametrize("mutation", ["prompt", "instruction", "input", "semantic_id", "graph_semantic_id", "reference_kind"])
def test_prompt_and_physical_identity_tampering_rejected(actual_inputs, mutation):
    generator, _, _, _, _, value = actual_inputs
    row = copy.deepcopy(value["validation_rows"][0])
    row[mutation] += " tamper"
    with pytest.raises(ValueError):
        prepare.physical_proof(row, generator)


@pytest.mark.parametrize("route", ["Route: [0, 0, 1]", "Route: [1, 0]", "Route: [0, 1.0]", "Route: [0, -1]", "Route: [0, 9999]"])
def test_raw_route_violations_fail_without_repair(actual_inputs, route):
    generator, _, _, _, _, value = actual_inputs
    row = copy.deepcopy(value["validation_rows"][0]); row["output"] = route
    with pytest.raises(ValueError, match="physical constraints"):
        prepare.physical_proof(row, generator)


def test_printed_objective_does_not_replace_physical_recomputation(actual_inputs):
    generator, _, _, _, _, value = actual_inputs
    row = copy.deepcopy(value["validation_rows"][0])
    original = prepare.physical_proof(row, generator)
    row["output"] = row["output"].split("Objective:")[0] + "Objective: -999999"
    assert prepare.physical_proof(row, generator) == original


def test_over_budget_open_path_and_duplicate_points_rejected(actual_inputs):
    generator, _, _, _, _, value = actual_inputs
    row = copy.deepcopy(value["validation_rows"][0]); row["instance_json"][2] = 0.1
    with pytest.raises(ValueError, match="exceeds budget"):
        prepare.physical_proof(row, generator)
    row = copy.deepcopy(value["validation_rows"][0]); row["instance_json"][0][1] = row["instance_json"][0][0]
    with pytest.raises(ValueError, match="Distinct bounded"):
        prepare.physical_proof(row, generator)


def test_unbalanced_stratum_or_wrong_count_rejected(actual_inputs):
    generator, _, excluded, _, _, value = actual_inputs
    rows = copy.deepcopy(value["validation_rows"])
    rows[0]["development_band"] = list(generator.BANDS[1])
    with pytest.raises(ValueError, match="strata"):
        prepare.verify_inputs(value["training_rows"], rows, excluded, generator)
    with pytest.raises(ValueError, match="Exactly three"):
        prepare.verify_inputs(value["training_rows"], rows[:-1], excluded, generator)


def test_preexisting_destination_rejected_without_import_or_overwrite(tmp_path):
    marker = tmp_path / "preserved.txt"; marker.write_text("retained")
    with pytest.raises(ValueError, match="Preserve existing"):
        prepare.prepare(tmp_path)
    assert marker.read_text() == "retained"


def test_byte_binding_tamper_and_link_rejected(tmp_path):
    path = tmp_path / "input.json"; path.write_text("[]")
    _, bound = prepare.binding(path)
    path.write_text("[1]")
    with pytest.raises(ValueError, match="Pinned input"):
        prepare.binding(path, bound["sha256"])
    link = tmp_path / "link.json"; link.symlink_to(path)
    with pytest.raises(ValueError, match="linked"):
        prepare.binding(link)
