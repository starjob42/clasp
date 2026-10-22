import importlib.util
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys
import copy
import ast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fresh_artifacts", ROOT / "scripts/clasp_grpo_fresh_dev_artifacts.py")
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)


def save_signed(path, value, field):
    value = {key: item for key, item in value.items() if key != field}
    value[field] = fresh.digest(value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "benchmark"
    source = ROOT / "data/development_fair_cvrp_mis_20260905_v2"
    root.mkdir()
    shutil.copy2(source / "development_manifest.json", root / "development_manifest.json")
    for task in ("cvrp", "mis"):
        (root / task / "eval").mkdir(parents=True)
        shutil.copy2(source / task / "eval/test.json", root / task / "eval/test.json")
        shutil.copy2(source / task / "reference_evidence.json", root / task / "reference_evidence.json")
    manifest = json.loads((root / "development_manifest.json").read_text())
    audit = {"status": "passed_zero_semantic_overlap", "total_overlaps": 0,
             "development_manifest_sha256": manifest["manifest_sha256"],
             "sources": [{"task": task, "label": label, "rows": 100, "overlaps": [], "source": {"sha256": "a" * 64}}
                         for task in ("cvrp", "mis") for label in ("sft_train", "rl_train", "public_eval")]}
    save_signed(root / "training_overlap_audit.json", audit, "audit_sha256")
    return root


def test_validates_actual_fresh_instances_and_all_native_mis_proofs(benchmark):
    result = fresh.validate_benchmark(benchmark)
    assert len(result["ordered_instances"]["mis"]) == len(result["ordered_instances"]["cvrp"]) == 100


def test_missing_or_stale_overlap_audit_rejected(benchmark):
    path = benchmark / "training_overlap_audit.json"
    audit = json.loads(path.read_text())
    audit["development_manifest_sha256"] = "b" * 64
    save_signed(path, audit, "audit_sha256")
    with pytest.raises(ValueError, match="different dataset manifest"):
        fresh.validate_benchmark(benchmark)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        fresh.validate_benchmark(benchmark)


def test_reference_file_mutation_rejected_by_manifest(benchmark):
    path = benchmark / "mis/reference_evidence.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="file hash mismatch"):
        fresh.validate_benchmark(benchmark)


def test_feasible_but_uncertified_mis_cannot_claim_optimality(benchmark):
    row = json.loads((benchmark / "mis/eval/test.json").read_text())[84]
    proof = json.loads((benchmark / "mis/reference_evidence.json").read_text())[84]
    proof.update(native_status="FEASIBLE", native_status_code=2, optimality_certified=False, best_bound=76.0)
    proof.pop("evidence_sha256")
    proof["evidence_sha256"] = fresh.digest(proof)
    with pytest.raises(ValueError, match="native optimality certification"):
        fresh.validate_reference("mis", row, proof)


def test_native_proof_bound_and_incumbent_must_match(benchmark):
    row = json.loads((benchmark / "mis/eval/test.json").read_text())[0]
    proof = json.loads((benchmark / "mis/reference_evidence.json").read_text())[0]
    proof["native_response_stats"] = proof["native_response_stats"].replace("best_bound: 23", "best_bound: 24")
    proof.pop("evidence_sha256")
    proof["evidence_sha256"] = fresh.digest(proof)
    with pytest.raises(ValueError, match="native bound/objective mismatch"):
        fresh.validate_reference("mis", row, proof)


def test_rendered_cvrp_reference_cannot_be_weaker_than_its_proof(benchmark):
    row = json.loads((benchmark / "cvrp/eval/test.json").read_text())[0]
    proof = json.loads((benchmark / "cvrp/reference_evidence.json").read_text())[0]
    routes = copy.deepcopy(proof["reference_routes"])
    routes[0][1], routes[0][2] = routes[0][2], routes[0][1]
    coords = row["instance_json"][0]
    objective = sum(sum(fresh.math.dist(coords[a], coords[b]) for a, b in zip(route, route[1:])) for route in routes)
    assert objective > proof["objective"]
    row["output"] = f"Routes: {routes}, Objective: {objective:.12f}"
    with pytest.raises(ValueError, match="rendered CVRP routes differ"):
        fresh.validate_reference("cvrp", row, proof)


def test_rendered_mis_wrong_set_size_rejected(benchmark):
    row = json.loads((benchmark / "mis/eval/test.json").read_text())[0]
    proof = json.loads((benchmark / "mis/reference_evidence.json").read_text())[0]
    smaller = proof["reference_set"][:-1]
    row["output"] = f"Set: {smaller}, Objective: {len(smaller)}"
    with pytest.raises(ValueError, match="rendered MIS set differs"):
        fresh.validate_reference("mis", row, proof)


@pytest.mark.parametrize("task", ["cvrp", "mis"])
def test_rendered_objective_and_malformed_output_rejected(task, benchmark):
    row = json.loads((benchmark / task / "eval/test.json").read_text())[0]
    proof = json.loads((benchmark / task / "reference_evidence.json").read_text())[0]
    row["output"] = row["output"].split("Objective:")[0] + f"Objective: {proof['objective'] + 1}"
    with pytest.raises(ValueError, match="rendered .* objective differs"):
        fresh.validate_reference(task, row, proof)
    row["output"] = "unparseable reference"
    with pytest.raises(ValueError, match="malformed rendered reference"):
        fresh.validate_reference(task, row, proof)


@pytest.mark.parametrize("task", ["cvrp", "mis"])
def test_refreshed_dataset_manifest_and_audit_hashes_do_not_hide_stale_label(task, benchmark):
    dataset = benchmark / task / "eval/test.json"
    rows = json.loads(dataset.read_text())
    proof = json.loads((benchmark / task / "reference_evidence.json").read_text())[0]
    if task == "mis":
        solution = proof["reference_set"][:-1]
        rows[0]["output"] = f"Set: {solution}, Objective: {len(solution)}"
    else:
        routes = copy.deepcopy(proof["reference_routes"])
        routes[0][1], routes[0][2] = routes[0][2], routes[0][1]
        coords = rows[0]["instance_json"][0]
        objective = sum(sum(fresh.math.dist(coords[a], coords[b]) for a, b in zip(route, route[1:])) for route in routes)
        rows[0]["output"] = f"Routes: {routes}, Objective: {objective:.12f}"
    dataset.write_text(json.dumps(rows))
    manifest_path = benchmark / "development_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][task]["dataset"] = fresh.identity(dataset)
    manifest = save_signed(manifest_path, manifest, "manifest_sha256")
    audit_path = benchmark / "training_overlap_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["development_manifest_sha256"] = manifest["manifest_sha256"]
    save_signed(audit_path, audit, "audit_sha256")
    with pytest.raises(ValueError, match="rendered .* differ"):
        fresh.validate_benchmark(benchmark)


def make_evaluation(tmp_path, benchmark):
    output = tmp_path / "evaluation"
    (output / "eval").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_text("test-weights")
    immutable = tmp_path / "code_manifest"
    immutable.write_text("test-code")
    receipt = {"status": "validated_fresh_development_launch", "task": "mis", "method": "sft", "stage": "sft",
               "benchmark": fresh.validate_benchmark(benchmark), "generation": fresh.GENERATION,
               "checkpoint": {"adapter_model.safetensors": fresh.identity(adapter)},
               "source_manifest": fresh.identity(immutable), "management": {"helper": fresh.identity(immutable)},
               "evaluation_slice": {"split": "eval", "start": 0, "count": 100},
               "reference_semantics": {"mis": "gap_to_certified_optimum"}}
    save_signed(output / "launch_receipt.json", receipt, "launch_sha256")
    records = []
    for target in receipt["benchmark"]["ordered_instances"]["mis"]:
        record = dict(target, task="mis", split="eval", num_return_sequences=8, candidates=[{} for _ in range(8)],
                      protocol_fingerprint="protocol")
        record["record_sha256"] = fresh.digest(record)
        records.append(record)
    detail = output / "eval/run_details__mock.jsonl"
    detail.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    config = {"run_status": "complete", "stage": "sft", "tasks": ["mis"], "generation": fresh.GENERATION,
              "resolved_model_path": str(checkpoint), "protocol_fingerprint": "protocol",
              "prompting": {"prompt_style_resolved": "jiang_f638", "no_reason": True,
                            "clasp_mis_jiang_response_output": "0", "clasp_mis_relabel_order": ""},
              "limits": {"eval_start_index": 0, "max_eval_samples_per_task": 100, "prefer_subset_files": False},
              "details_artifact": fresh.identity(detail)}
    config_path = output / "eval/run_config__mock.json"
    save_signed(config_path, config, "run_config_sha256")
    return output, detail, config_path


def test_completed_index_binds_100_instance_references_and_is_immutable(tmp_path, benchmark):
    output, _, _ = make_evaluation(tmp_path, benchmark)
    result = fresh.index(output)
    assert result["status"] == "complete_fresh_development_only"
    assert result["reference_semantics"] == "gap_to_certified_optimum"
    assert (output / "result_index.json").stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        fresh.index(output)


def test_wrong_instance_rejected_even_with_valid_artifact_hashes(tmp_path, benchmark):
    output, detail, config_path = make_evaluation(tmp_path, benchmark)
    records = [json.loads(line) for line in detail.read_text().splitlines()]
    records[0]["semantic_instance_sha256"] = "f" * 64
    records[0].pop("record_sha256")
    records[0]["record_sha256"] = fresh.digest(records[0])
    detail.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    config = json.loads(config_path.read_text())
    config["details_artifact"] = fresh.identity(detail)
    save_signed(config_path, config, "run_config_sha256")
    with pytest.raises(ValueError, match="instance/reference mismatch"):
        fresh.index(output)


def test_evaluation_budget_and_checkpoint_are_verified(tmp_path, benchmark):
    output, _, config_path = make_evaluation(tmp_path, benchmark)
    config = json.loads(config_path.read_text())
    config["generation"]["num_return_sequences"] = 16
    save_signed(config_path, config, "run_config_sha256")
    with pytest.raises(ValueError, match="decoding mismatch"):
        fresh.index(output)
    config["generation"]["num_return_sequences"] = 8
    config["resolved_model_path"] = "/different-checkpoint"
    save_signed(config_path, config, "run_config_sha256")
    with pytest.raises(ValueError, match="checkpoint differs"):
        fresh.index(output)


@pytest.mark.parametrize("method,stage", [("sft", "sft"), ("grpo_z", "grpo"), ("v3", "grpo"), ("v4", "grpo")])
def test_wrapper_pins_identical_decoding_and_clears_inherited_research_knobs(tmp_path, method, stage):
    source, management, runtime = tmp_path / "source", tmp_path / "management", tmp_path / "runtime"
    (source / "scripts").mkdir(parents=True)
    (management / "scripts").mkdir(parents=True)
    (runtime / "bin").mkdir(parents=True)
    for name in ("data", "checkpoint", "train"):
        (tmp_path / name).mkdir()
    mock_python = runtime / "bin/python"
    mock_python.write_text("#!/bin/bash\nprintf '{}\\n'\n")
    mock_python.chmod(0o755)
    # Execute the real shell wrapper with a stand-in evaluator. The separate
    # validator tests above exercise the actual receipt/proof checks.
    evaluator = source / "scripts/evaluate_a100.slurm"
    evaluator.write_text("#!/bin/bash\n" + sys.executable + " - <<'PY'\n"
                         "import json,os,pathlib\n"
                         "pathlib.Path(os.environ['MOCK_ENV_PATH']).write_text(json.dumps(dict(os.environ)))\nPY\n")
    captured = tmp_path / "captured.json"
    env = dict(os.environ, FRESH_SOURCE=str(source), FRESH_MANAGEMENT_ROOT=str(management), FRESH_RUNTIME=str(runtime),
               FRESH_DATA_ROOT=str(tmp_path / "data"), FRESH_CHECKPOINT=str(tmp_path / "checkpoint"),
               FRESH_OUTPUT=str(tmp_path / "output"), FRESH_TASK="mis", FRESH_METHOD=method,
               MOCK_ENV_PATH=str(captured), CLASP_MIS_RELABEL_ORDER="inherited", MAX_PROMPT_LENGTH="0",
               TEMPERATURE="1.5", NUM_RETURN_SEQUENCES="16", EVAL_START_INDEX="200", GRPO_BAD_KNOB="1")
    if method != "sft":
        env["FRESH_TRAIN"] = str(tmp_path / "train")
    else:
        env.pop("FRESH_TRAIN", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    observed = json.loads(captured.read_text())
    assert observed["STAGE"] == stage
    assert observed["MAX_PROMPT_LENGTH"] == "29768"
    assert observed["NUM_RETURN_SEQUENCES"] == observed["GENERATION_BATCH_SIZE"] == "8"
    assert observed["TEMPERATURE"] == "0.7" and observed["TOP_P"] == "0.9"
    assert observed["EVAL_START_INDEX"] == "0" and observed["MAX_EVAL_SAMPLES_PER_TASK"] == "100"
    assert observed["EVALUATION_REPRESENTATION_PROFILE"] == ("jiang_f638_sft" if stage == "sft" else "jiang_f638_clasp")
    assert "CLASP_MIS_RELABEL_ORDER" not in observed and "GRPO_BAD_KNOB" not in observed


@pytest.mark.parametrize("recorded_key", ["sft_cvrp", "sft_mis", "grpo_cvrp", "v3_cvrp"])
def test_stage_profiles_satisfy_actual_certified_hyperparameters(recorded_key, monkeypatch):
    fixture = ROOT / "analysis_outputs/clasp_grpo_fair_dev_20260905/fresh_stage_representation_metadata.json"
    metadata = json.loads(fixture.read_text())[recorded_key]
    tree = ast.parse((ROOT / "src/clasp/entrypoints/evaluate.py").read_text())
    selected = [node for node in tree.body if
                (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == "_REPRESENTATION_HPARAM_TO_ENV" for target in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "_restore_stage_representation_env")]
    namespace = {"os": os}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "frozen-representation-restorer", "exec"), namespace)
    sft = recorded_key.startswith("sft_")
    settings = {"CLASP_TSP_PROMPT_KNN": "0", "CLASP_TSP_PROMPT_KNN_DISTANCES": "" if sft else "0",
                "CLASP_TSP_INPUT_ORDER": "", "CLASP_TSP_RELABEL_ORDER": "", "CLASP_TSP_ROUTE_CANONICALIZE": "",
                "CLASP_TSP_ROUTE_ONLY_OUTPUT": "" if sft else "0", "CLASP_TSP_LABEL_TTA_SCHEDULE": "",
                "CLASP_MIS_RELABEL_ORDER": "", "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0",
                "CLASP_MIS_PROMPT_NODE_DEGREES": "0", "CLASP_MIS_PROMPT_LOW_DEGREE_K": "32"}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    namespace["_restore_stage_representation_env"](metadata["stage_hyperparameters"])
    if sft:
        monkeypatch.setenv("CLASP_TSP_PROMPT_KNN_DISTANCES", "0")
        with pytest.raises(RuntimeError, match="conflicts with the recorded representation contract"):
            namespace["_restore_stage_representation_env"](metadata["stage_hyperparameters"])
