"""CPU integration: real checkpoint sealing must satisfy the canonical raw auditor."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from clasp.baselines.budget_matched_dev import build_runtime
from clasp.models import checkpoint_fidelity as fidelity
from clasp.rewards.external_baseline import ExternalBaselineTracker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "merges.txt", "vocab.json")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def canonical_identity(monkeypatch):
    """Execute the canonical fingerprint code without importing model/dataset stacks.

    These are the actual evaluator definitions, not a second classifier or a
    mocked manifest. The full canonical audit_arm is imported normally below.
    """
    path = ROOT / "src/clasp/entrypoints/evaluate.py"
    names = {"_INFERENCE_ARTIFACT_EXACT_NAMES", "_INFERENCE_WEIGHT_PREFIXES",
             "_INFERENCE_ARTIFACT_SUFFIXES", "_is_inference_artifact",
             "_directory_content_identity", "_sha256_file", "_canonical_json_bytes"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in names
                         for target in node.targets))]
    assert len(nodes) == len(names)
    core = ModuleType("clasp.entrypoints.evaluate")
    core.__dict__.update(os=os, json=json, hashlib=hashlib, Any=Any,
        CHECKPOINT_FIDELITY_FILENAME=fidelity.RECEIPT_FILENAME,
        EXACT_PEFT_KL_REFERENCE_FILENAME="exact_peft_kl_reference.json",
        EXACT_PEFT_KL_STEP1_FILENAME="exact_peft_kl_reference_step1.json",
        EXACT_PEFT_KL_POST_STEP_FILENAME="exact_peft_kl_reference_post_step.json")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), core.__dict__)
    monkeypatch.setitem(sys.modules, core.__name__, core)
    return core._directory_content_identity


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def small_adapter(checkpoint):
    """A real safetensors file and verified trainable-tensor fidelity receipt."""
    checkpoint.mkdir(parents=True)
    key, payload = "base_model.model.layer.lora_A.weight", struct.pack("<ff", 1.25, -2.5)
    header = json.dumps({"__metadata__": {"format": "pt"}, key:
                        {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, len(payload)]}}).encode()
    (checkpoint / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    put(checkpoint / "adapter_config.json", {"peft_type": "LORA", "bias": "none",
                                           "r": 2, "target_modules": ["q_proj"]})
    tensors = [{"key": key, "dtype": "F32", "shape": [1, 2], "numel": 2,
                "value_sha256": hashlib.sha256(payload).hexdigest()}]
    inventory = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    fidelity.write_checkpoint_fidelity_receipt(checkpoint, {
        "schema_version": 1, "protocol": fidelity.CLOSURE_PROTOCOL, "status": "exact",
        "adapter_name": "default", "tensor_count": 1, "parameter_count": 2,
        "tensor_inventory_sha256": inventory, "tensors": tensors})


@pytest.fixture(params=["classic_grpo", "rloo_legacy", "rloo_quality"])
def endpoint(tmp_path, monkeypatch, canonical_identity, request):
    common = load(HERE / "common.py", "chunk_test_common")
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.syspath_prepend(str(ROOT))
    chunk = load(HERE / "run_chunk.py", "chunk_test_runner")
    method, task, step = request.param, "op", 100
    source = {"root": str(tmp_path / "source"), "manifest_sha256": "1" * 64}
    source_binding = {"status": "passed_frozen_manifest_successor_source",
                      "source_root": source["root"], "manifest": {"sha256": source["manifest_sha256"]}}
    declaration = {"root": str(tmp_path / "campaign"), "source": source,
                   "tasks": {task: {"training_data_root": str(tmp_path / "data"), "training_data_files": {}}}}
    contract = common.run_contract(declaration, {},
        {"task": task, "method": method, "kind": "train"}, {"checkpoint": "/initial", "files": {}})
    contract_path = Path(contract["output_dir"]) / common.CONTRACT_NAME
    put(contract_path, contract)
    checkpoint = contract_path.parent / "checkpoints/checkpoint-100"
    small_adapter(checkpoint)
    for name in (*TOKENIZER_FILES, "config.json", "generation_config.json", "nested/chat_template.jinja",
                 "nested/modeling_custom.py", "optimizer.pt", "scheduler.pt", "README.md",
                 "runs/tokenizer.json", ".cache/tokenizer.json"):
        put(checkpoint / name, {"fixture": name})
    put(checkpoint / "trainer_state.json", {"global_step": step, "max_steps": contract["horizon"]})
    expected = build_runtime(source_binding, contract, common.identity(contract_path)).resume_contract()
    put(checkpoint / "budget_resume_contract.json", {
        "schema_version": 1, "status": "passed", "global_step": step, "world_size": 4,
        "contract": expected, "contract_sha256": hashlib.sha256(common.canonical(expected)).hexdigest()})
    if method.startswith("rloo_"):
        tracker = ExternalBaselineTracker()
        for _ in range(step):
            tracker.update(tracker.snapshot([(task, 50)] * 36), [1.0] * 36)
        put(checkpoint / "external_baseline_state.json", {"global_step": step,
            "advantage_estimator": "rloo", "num_generations": 9, "max_completion_length": 3000,
            "tracker": tracker.state_dict(), "reward_contract": expected["reward_contract"]})
    for rank in range(4):
        put(checkpoint / f"rng_state_{rank}.pth", {"rank": rank})
        states = {}
        for owner in ("optimizer", "scheduler"):
            name = f"{owner}_rank{rank}.pt"
            put(checkpoint / name, {"rank": rank, "owner": owner})
            states[owner] = {"file": name, "file_sha256": common.identity(checkpoint / name)["sha256"],
                             "semantic_sha256": "2" * 64}
        put(checkpoint / f"external_rank_state_{rank}.json", {
            "schema_version": 1, "status": "passed", "rank": rank, "world_size": 4,
            "global_step": step, "states": states})
    material = chunk.checkpoint_material(checkpoint, contract, contract_path, step, source_binding)
    return SimpleNamespace(common=common, checkpoint=checkpoint, material=material, method=method,
                           identity=canonical_identity, tmp=tmp_path)


def audit_endpoint(endpoint, monkeypatch, *, omit=()):
    """Feed sealed training files and raw fingerprint through the entire auditor."""
    audit = load(ROOT / "experiments/grpo_external_baseline_all_tasks_20260910/all_tasks_evaluation.py",
                 "chunk_test_canonical_auditor")
    ep = endpoint
    files = {name: value for name, value in ep.material["files"].items() if name not in omit}
    lineage_path = ep.tmp / "lineage.json"
    audit.write_signed(lineage_path, {"task": "op", "method": ep.method, "step": 100,
        "seed": 43, "train_completions": 3600, "checkpoint_files": files,
        "status": "verified_terminal_training"})
    raw_config = {"run_status": "complete", "evaluation_purpose": "development", "tasks": ["op"],
        "stage": "grpo", "resolved_model_path": str(ep.checkpoint), "data_root": str(ep.tmp / "data"),
        "generation": {**audit.GENERATION, "seed_policy": "per_example_sha256_v1",
            "temperature_schedule": [], "top_p_schedule": [], "op_size_band_top_p_schedule": []},
        "best_run_update_requested": False, "benchmark_manifest": None, "campaign_manifest": None,
        "prompting": {"no_reason": True, "prompt_style_resolved": "jiang_f638"},
        "limits": {"eval_start_index": 0, "max_eval_samples_per_task": 1, "prefer_subset_files": False},
        "experiment_provenance": {"model": {"checkpoint_artifact":
            ep.identity(str(ep.checkpoint), inference_artifacts_only=True)}}}
    config_path, details = ep.tmp / "raw_config.json", ep.tmp / "raw_details.jsonl"
    audit.write_signed(config_path, raw_config, "run_config_sha256")
    metrics = {"is_feasible": False, "objective_value": None, "ground_truth_objective": 10.0}
    raw = {"example_index": 0, "prompt_was_truncated": False, "instance_json": {},
           "instruction": "fixture", "input_problem": "fixture", "ground_truth_output": "fixture",
           "candidates": [{"completion": "fixture", **metrics} for _ in range(8)],
           "ground_truth_objective": 10.0, "best_candidate_index": None, "best_objective_value": None}
    put(details, raw)
    record = SimpleNamespace(example_index=0, raw=raw, pass_at_k=False)
    artifact = SimpleNamespace(artifact_type="native_raw", strict_verifier={"fixture": "verifier"},
                               records={0: record}, candidate_budget=8, protocol_fingerprint="fixture")
    monkeypatch.setattr(audit, "task_tools", lambda spec: (None, SimpleNamespace(_load_artifact=lambda *a, **k: artifact)))
    core = SimpleNamespace(_strict_verifier_identity=lambda: artifact.strict_verifier,
        build_formatted_prompt=lambda *a, **k: "fixture", TASK_EVALUATORS={"op": lambda *a: metrics},
        _select_best_candidate_index=lambda *a: None)
    arm = {"checkpoint": str(ep.checkpoint), "checkpoint_files": files,
           "lineage_receipt": audit.identity(lineage_path), "training_recipe": {"method": ep.method},
           "existing": {"details": audit.identity(details), "run_config": audit.identity(config_path)}}
    spec = {"data_root": str(ep.tmp / "data"), "start": 0, "count": 1,
            "reference": {"kind": "verified_reference"}}
    return audit.audit_arm("op", ep.method, spec, arm, core,
                           {"ordered_rows": [{"example_index": 0}], "reference_objectives": [10.0]}, {})


def test_training_inventory_passes_canonical_raw_audit(endpoint, monkeypatch):
    files = endpoint.material["files"]
    assert set(TOKENIZER_FILES) <= files.keys()
    assert {"nested/chat_template.jinja", "nested/modeling_custom.py", "generation_config.json"} <= files.keys()
    assert not {"README.md", "runs/tokenizer.json", ".cache/tokenizer.json"} & files.keys()
    _, evidence = audit_endpoint(endpoint, monkeypatch)
    assert evidence["canonical_candidates_rechecked"] == 8


def test_original_missing_tokenizer_inventory_fails_canonical_audit(endpoint, monkeypatch):
    with pytest.raises(ValueError, match="Raw policy differs from trained endpoint"):
        audit_endpoint(endpoint, monkeypatch, omit=TOKENIZER_FILES)


@pytest.mark.parametrize("name", ["tokenizer.json", "adapter_model.safetensors", "nested/chat_template.jinja"])
def test_sealed_inference_corruption_fails_canonical_audit(endpoint, monkeypatch, name):
    with (endpoint.checkpoint / name).open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="Bound file changed"):
        audit_endpoint(endpoint, monkeypatch)
