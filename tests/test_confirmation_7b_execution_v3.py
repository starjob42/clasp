"""Execution/admission guards only; no policy, model, GPU or quality calls."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "experiments/rloo_one_shot_20261016/run_confirmation_7b_v3.py"
spec = importlib.util.spec_from_file_location("confirmation_execution_guards", SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def store(path, value):
    path.write_bytes(M.encoded(value) + b"\n")
    return M.binding(path)


def test_default_is_inert_even_with_missing_contract(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--contract", str(tmp_path / "absent")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"status": "prepared_only", "model_calls": 0, "GPU_calls": 0, "fresh_policy_calls": 0}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("branch,count,hours", [("pure", 28, 102.5), ("teacher", 48, 182.5)])
def test_full_fixed_experiment_and_engineering_first(tmp_path, branch, count, hours):
    slots = M.fixed_slots(branch, tmp_path)
    assert len(slots) == len({s["slot"] for s in slots}) == count
    assert sum(s["wall_seconds"] for s in slots) == hours*3600
    assert [s["mode"] for s in slots[:3]] == ["qualification", "qualification", "baseline"]
    assert slots[2]["arm"] == "sft" and slots[2]["evaluation_rows"] == 256
    assert all(s["evaluation_rows"] == 2 for s in slots[:2])
    assert {s["seed"] for s in slots[3:]} == set(M.SEEDS)
    assert all(s["maximum_attempts"] == 1 for s in slots)
    assert sum(s["evaluation_rows"] for s in slots if s["mode"] != "qualification") == (26 if branch == "pure" else 46)*256


@pytest.mark.parametrize("branch", ["optional_teacher_grpo", "selected_after_peeking", "", None])
def test_unregistered_branch_denied(tmp_path, branch):
    with pytest.raises(ValueError): M.fixed_slots(branch, tmp_path)


def test_training_binding_projection_retains_outer_row_admission(tmp_path):
    row = store(tmp_path / "input.json", {"fixture": "input bytes only"})
    original = {"training": {**row, "rows": 128}, "evaluation": row, "admission": row}
    projected = M.factory_inputs(original)
    assert original["training"]["rows"] == 128
    assert all(set(r) == {"path", "sha256", "size_bytes"} for r in projected.values())
    assert M.read_bound(projected["training"]) == M.read_bound(original["training"])


@pytest.mark.parametrize("change", ["symlink", "bytes", "boolean_size", "malformed_sha"])
def test_changed_or_malformed_bound_evidence_denied(tmp_path, change):
    p = tmp_path / "evidence"; p.write_text("fixture")
    row = M.binding(p)
    if change == "symlink":
        q = tmp_path / "alias"; q.symlink_to(p); row["path"] = str(q)
    elif change == "bytes": p.write_text("drift")
    elif change == "boolean_size": row["size_bytes"] = True
    else: row["sha256"] = "x"*64
    with pytest.raises(ValueError): M.read_bound(row)


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), 3601])
def test_deadline_requires_whole_finite_slot(seconds):
    a = {"execution_deadline_utc": "2026-10-30T12:00:00+00:00"}
    with pytest.raises(ValueError):
        M.require_time_fit(a, seconds, now=datetime(2026, 10, 30, 11, tzinfo=timezone.utc))


def test_exact_deadline_fit_is_allowed():
    M.require_time_fit({"execution_deadline_utc": "2026-10-30T12:00:00+00:00"}, 3600,
                       now=datetime(2026, 10, 30, 11, tzinfo=timezone.utc))


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    """Invented metadata fixture, never an actual model admission."""
    slot = next(s for s in M.fixed_slots("pure", tmp_path) if s["slot"] == "qualification.op.rloo_cost.seed101")
    out = Path(slot["output"]); out.mkdir(parents=True)
    input_binding = store(tmp_path / "input.json", {"synthetic_fixture": True})
    inputs = {"training": {**input_binding, "rows": 128}, "evaluation": input_binding, "admission": input_binding}
    a = {"sources": {}, "master_manifest": input_binding, "inputs": inputs, "selection_receipt": input_binding}
    student = {"contract_sha256": "d"*64, "training": {"prompt_draws": [["fixture"]]*100}}
    contract = {"authority": input_binding}
    cb = store(tmp_path / "contract.json", contract)
    old_metadata = M.binding(ROOT / "experiments/rloo_one_shot_20261016/pipeline_v1/analysis_admission_v1.py")
    context = {"slot": slot, "authority": a, "roster": {"source_manifest": input_binding}, "student": student,
               "base": types.SimpleNamespace(GPU_UUID="fixture_A6000"), "packet": {"old_metadata": old_metadata}}
    monkeypatch.setattr(M, "validate_contract", lambda c: context)
    monkeypatch.setattr(M, "evaluation_rows", lambda c: ([{}, {}], input_binding))
    dtype = {"parameter_counts_by_dtype": {"torch.bfloat16": 7_615_616_512, "torch.float32": 161_480_704},
             "trainable_parameters": 161_609_728, "devices": ["cuda:0"], "actual_BF16_base_weights_verified": True,
             "FP32_adapter_parameter_names": ["fixture_lora_A"]}
    numerical = {"input_bindings": M.factory_inputs(inputs), "student_contract_sha256": student["contract_sha256"],
                 "confirmation_policy_authority_granted": False}
    terminal = out / "terminal"; terminal.mkdir()
    files = []
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json"):
        p = terminal / name; p.write_text("synthetic fixture; no weights")
        b = M.binding(p); files.append({**b, "path": name})
    files.sort(key=lambda r: r["path"])
    tree = {"files": files, "tree_sha256": M.sha(M.encoded(files))}
    training = {"trained_updates": 1, "RL_responses": 9, "teacher_forwards": 0, "actor_changed": True,
                "initial_whole_actor_sha256": M.INITIAL_ACTOR_SHA, "whole_actor_sha256": "b"*64, "terminal_tree": tree,
                "numerical_projection": numerical, "checkpoint_policy": "terminal_only_no_optimizer_resume",
                "superiority_established": False, "actual_windows": [["fixture"]*9], "reference_sha256": M.INITIAL_ACTOR_SHA,
                "initial_reference_audit": {"bitwise_equal": True}, "prepared_reference_audit": {"bitwise_equal": True},
                "effective_checkpointing": {"effective_use_reentrant": False}, "generated_tokens": 9}
    tb = store(out / "training_complete.json", training)
    reload = {"bitwise_state_equal": True, "whole_actor_sha256": "b"*64, "training_complete_binding": tb, "terminal_tree": tree}
    raw = out / "raw_completions.jsonl"; raw.write_text("Deliberately invalid JSON; quality must stay closed\n")
    proof = {"protocol": M.PROTOCOL, "status": "complete_confirmation_qualification", "contract": contract,
             "contract_file_binding": cb, "authority_binding": input_binding, "slot": slot, "sources": {},
             "master_manifest": input_binding, "effective_source_manifest_sha256": M.effective_manifest(a, context["roster"]),
             "initial_whole_actor_sha256": M.INITIAL_ACTOR_SHA, "whole_actor_sha256": "b"*64,
             "checkpoint_reload_exact": True, "superiority_established": False, "publication_eligible": False,
             "trained_updates": 1, "RL_responses": 9, "teacher_forwards": 0, "evaluation_responses": 2,
             "reference_integrity": True, "training_complete_binding": tb,
             "checkpoint_reload_binding": store(out / "checkpoint_reload_exact.json", reload),
             "input_admission_binding": store(out / "input_admission.json", {"contract_file_binding": cb,
                 "authority_binding": input_binding, "inputs": inputs, "selection_receipt": input_binding, "policy_calls_before_admission": 0}),
             "model_admission_binding": store(out / "model_admission.json", {**dtype, "runtime": M.PACKAGES,
                 "initial_whole_actor_sha256": M.INITIAL_ACTOR_SHA, "numerical_projection": numerical, "outer_authority_grants_policy_role": True}),
             "reference_admission_binding": store(out / "reference_admission.json", {**dtype, "trainable_parameters": 0,
                 "contract_file_binding": cb, "initial_whole_actor_sha256": M.INITIAL_ACTOR_SHA, "independent_original_SFT_load": True}),
             "gpu_owner_binding": store(out / "gpu_owner.json", {"protocol": "one-shot-exclusive-gpu-owner-v1",
                 "pid": 123, "uid": M.os.getuid(), "gpu_uuid": "fixture_A6000", "all_locks_held_until_GPU_lifetime_closed": True}),
             "raw_completions_binding": M.binding(raw), "evaluated_input_binding": input_binding, "fresh_confirmation_policy_calls": 0,
             "resources": {"physical_gpus": 1, "GPU_uuid": "fixture_A6000", "worker_wall_seconds": 9.,
                 "training_wall_seconds": 4., "evaluation_wall_seconds": 3., "peak_allocated_bytes": 47_000_000_000,
                 "peak_reserved_bytes": 48_000_000_000,
                 "runtime": M.PACKAGES, "world_size": 1, "native_bfloat16": True, "compute_capability": [8, 6]}}
    store(out / "result.json", proof)
    parent = {"status": "completed", "error": None, "contract_file_sha256": cb["sha256"],
              "parent_elapsed_wall_seconds": 10., "actual_GPU_hour_upper_bound": 10/3600,
              "requested_gpu_hours": .5, "automatic_retry": False}
    store(out / "completion.json", parent)
    store(out / "attempt.json", {"contract_file_binding": cb, "authority_binding": input_binding, "slot": slot,
        "maximum_wall_seconds": 1800, "requested_gpu_hours": .5,
        "recorded_utc": "2026-10-16T11:00:00+00:00", "automatic_retry": False})
    return cb, out, proof, training, parent


def test_actual_closed_metadata_helper_and_parent_interface(endpoint):
    cb, out, _, _, _ = endpoint
    result = M.verify_completed(cb, out)
    assert result["status"] == "complete_confirmation_endpoint_metadata"
    assert result["contract_binding"] == cb
    assert result["slot"] == "qualification.op.rloo_cost.seed101"
    assert result["quality_analyzed"] is False


def test_metadata_never_opens_raw_bytes(endpoint, monkeypatch):
    cb, out, proof, _, _ = endpoint
    original = M.read_bound
    def closed(row):
        assert row["path"] != proof["raw_completions_binding"]["path"], "Raw opened before all endpoints"
        return original(row)
    monkeypatch.setattr(M, "read_bound", closed)
    M.verify_completed(cb, out)


@pytest.mark.parametrize("field,value", [("trained_updates", True), ("RL_responses", 8), ("evaluation_responses", 1),
    ("fresh_confirmation_policy_calls", 256), ("initial_whole_actor_sha256", "a"*64),
    ("reference_integrity", False), ("checkpoint_reload_exact", False), ("superiority_established", True)])
def test_invalid_endpoint_never_admitted(endpoint, field, value):
    cb, out, proof, _, _ = endpoint; proof[field] = value; store(out / "result.json", proof)
    with pytest.raises(ValueError): M.verify_completed(cb, out)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("error", "timeout"), ("parent_elapsed_wall_seconds", True),
    ("parent_elapsed_wall_seconds", 1801), ("actual_GPU_hour_upper_bound", 1.), ("requested_gpu_hours", 4.), ("automatic_retry", True)])
def test_failed_or_over_budget_parent_never_admitted(endpoint, field, value):
    cb, out, _, _, parent = endpoint; parent[field] = value; store(out / "completion.json", parent)
    with pytest.raises(ValueError): M.verify_completed(cb, out)


def test_terminal_weight_drift_denied(endpoint):
    cb, out, _, _, _ = endpoint
    (out / "terminal/adapter_model.safetensors").write_text("drift")
    with pytest.raises(ValueError, match="terminal"): M.verify_completed(cb, out)


def test_reference_drift_denied(endpoint):
    cb, out, proof, _, _ = endpoint
    p = out / "reference_admission.json"; value = json.loads(p.read_text()); value["initial_whole_actor_sha256"] = "a"*64
    proof["reference_admission_binding"] = store(p, value); store(out / "result.json", proof)
    with pytest.raises(ValueError, match="reference"): M.verify_completed(cb, out)


def test_false_raw_length_cannot_become_policy_admission(endpoint):
    # Admission deliberately uses only stat; the all-slot analyzer is separately
    # responsible for exact256 token/physics replay. It must stay labeled closed.
    cb, out, _, _, _ = endpoint
    assert M.verify_completed(cb, out)["quality_analyzed"] is False


@pytest.mark.parametrize("change", ["missing", "contract", "retry", "cap", "future", "naive"])
def test_first_attempt_custody_is_required(endpoint, change):
    cb, out, _, _, _ = endpoint; p = out / "attempt.json"
    if change == "missing":
        p.unlink()
    else:
        value = json.loads(p.read_text())
        if change == "contract": value["contract_file_binding"] = {"sha256": "a"*64}
        elif change == "retry": value["automatic_retry"] = True
        elif change == "cap": value["maximum_wall_seconds"] = 1801
        elif change == "future": value["recorded_utc"] = "2027-01-01T00:00:00+00:00"
        else: value["recorded_utc"] = "2026-10-16T11:00:00"
        store(p, value)
    with pytest.raises((ValueError, FileNotFoundError)): M.verify_completed(cb, out)


@pytest.mark.parametrize("field,value", [("worker_wall_seconds", 11.), ("training_wall_seconds", float("inf")),
    ("evaluation_wall_seconds", True), ("peak_reserved_bytes", 60_000_000_000), ("peak_allocated_bytes", True),
    ("world_size", True)])
def test_physical_resource_proof_is_required(endpoint, field, value):
    cb, out, proof, _, _ = endpoint; proof["resources"][field] = value
    # Write deliberately nonfinite fixture via json; production encoded rejects it.
    (out / "result.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError): M.verify_completed(cb, out)
