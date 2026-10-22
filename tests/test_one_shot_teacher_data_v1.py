"""Metadata/strict-target replay only; no policy results, model or search runs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/"experiments/rloo_one_shot_20261016/prepare_teacher_data_v1.py"
spec = importlib.util.spec_from_file_location("prepare_one_shot_shared_teacher_data_v1",PATH)
p = importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@pytest.mark.parametrize("task",["op","cvrp"])
def test_actual_admitted_inputs_are_exact_training_validation_engineering_metadata_without_models(task,monkeypatch):
    monkeypatch.setattr(p.core.v2,"loaded_model",lambda *a,**kw:pytest.fail("No model may load in teacher data preparation"))
    rows,ids,initializer,provenance = p.admitted_inputs(task)
    assert {key:len(value) for key,value in rows.items()} == {"train":128,"validation":96,"engineering_train":3,"engineering_validation":3}
    assert len(set().union(*(set(value["semantic_ids"]) for value in ids.values()))) == 230
    assert len(set().union(*(set(value["graph_ids"]) for value in ids.values()))) == 230
    assert initializer["id"] == "original_task_specific_"+task+"_SFT"
    assert not provenance["prior_excluded_data_re_read"] and not provenance["SFT500K_full_exclusion_certified"]
    first = p.canonical_contract(task,initializer,rows["train"],rows["validation"])
    second = p.canonical_contract(task,initializer,rows["train"],rows["validation"])
    assert first == second and first["scope"] == "prospective_teacher_development"
    assert "arm" not in first and "local_execution_contract_sha256" not in first
    assert first["training"]["horizon"] == 100 and first["training"]["groups_per_update"] == 1
    assert first["training"]["responses_per_step"] == 9 and first["train_seed"] == 73


@pytest.mark.parametrize("task",["op","cvrp"])
def test_all_saved_full_training_targets_replay_strictly_and_bundle_is_shared_arm_agnostic(task):
    folder = PATH.parent/"teacher_data_v1"/task
    contract = json.loads((folder/"logical_contract.json").read_bytes())
    bundle = json.loads((folder/"teacher_bundle.json").read_bytes())
    summary = json.loads((folder/"summary.json").read_bytes())
    adapter = p.adapter_module()
    assert adapter.validate_teacher_bundle(contract,bundle,bundle["bundle_sha256"]) == bundle
    assert bundle["shared_identically_across_recipes"] == ["rloo_cost","foarl_released"]
    assert len(bundle["targets"]) == bundle["search_calls"] == 128
    assert summary["strict_feasible_before"] == summary["strict_feasible_after"] == 128
    assert summary["actual_move_evaluations"] <= 128*2048
    assert bundle["admission"]["limits"] == adapter.engine_module().LIMITS
    assert summary["one_shared_bundle_sha256"] == bundle["bundle_sha256"]
    assert not summary["production_registration_ready"] and not summary["production_launch_enabled"]
    assert not summary["learned_gain_established"] and summary["policy_evaluation_calls"] == 0
    assert not summary["objective_printing_prompt_compliance_qualified"]
    assert len(bundle["admission"]["excluded_semantic_ids_by_split"]["validation"]) == 96
    assert len(bundle["admission"]["excluded_semantic_ids_by_split"]["engineering_eval"]) == 6
    assert bundle["admission"]["excluded_semantic_ids_by_split"]["test"] == []


@pytest.mark.parametrize("task",["op","cvrp"])
def test_validation_and_both_engineering_splits_cannot_supply_a_teacher_target(task):
    folder = PATH.parent/"teacher_data_v1"/task
    bundle = json.loads((folder/"teacher_bundle.json").read_bytes())
    engine = p.adapter_module().engine_module();rows,_,_,_ = p.admitted_inputs(task)
    for split in ("validation","engineering_train","engineering_validation"):
        candidate = copy.deepcopy(rows[split][0]);candidate["split"] = "training"
        with pytest.raises(ValueError,match="outside the exact admitted training rows"):
            engine._admit_row(candidate,bundle["admission"],bundle["admission_sha256"],bundle["request_context"])


@pytest.mark.parametrize("fault",["data_receipt","initializer_receipt"])
def test_original_input_custody_drift_is_denied_before_teacher_generation(monkeypatch,fault):
    monkeypatch.setattr(p,"DATA_ADMISSION_SHA256" if fault == "data_receipt" else "INITIALIZER_AUDIT_SHA256","0"*64)
    with pytest.raises(ValueError,match="receipt changed|metadata differs"):
        p.admitted_inputs("op")


def test_unsupported_task_and_missing_input_binding_fail_closed(tmp_path):
    with pytest.raises(ValueError):p.admitted_inputs("tsp")
    source = tmp_path/"dataset.json";source.write_text("[]")
    with pytest.raises(ValueError,match="identity changed"):
        p.read_bound({"path":str(source),"sha256":"0"*64,"size_bytes":2})
