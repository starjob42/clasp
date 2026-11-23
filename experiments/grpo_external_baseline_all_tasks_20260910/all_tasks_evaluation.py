"""Raw five-method evaluation and reports across all five CO tasks.

Each task keeps its own immutable source/data/reference provenance. Separate CPU
workers prevent imports from one historical evaluator source leaking into another.
There are no scheduler submissions and no incomplete final comparison tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

PROTOCOL = "external-baseline-all-five-tasks-raw8-20260910-v1"
DIRECTIONS = {"tsp": "min", "cvrp": "min", "op": "max", "mis": "max", "jssp": "min"}
METHODS = ("grpo_z", "foarl_f638_dev", "rloo_b9", "external_b9", "external_b1")
GENERATION = {"seed": 271828, "dtype": "float16", "num_return_sequences": 8, "generation_batch_size": 8,
              "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0,
              "max_new_tokens": 3000, "max_prompt_length": 29768}
GAP_THRESHOLDS = {"0pct": 0.0, "1pct": 0.01, "3pct": 0.03, "5pct": 0.05, "10pct": 0.1}
REPRESENTATION = {"CLASP_TSP_PROMPT_KNN": "0", "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_INPUT_ORDER": "", "CLASP_TSP_RELABEL_ORDER": "", "CLASP_TSP_ROUTE_CANONICALIZE": "",
    "CLASP_TSP_LABEL_TTA_SCHEDULE": "", "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0", "CLASP_MIS_RELABEL_ORDER": "",
    "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0", "CLASP_MIS_PROMPT_NODE_DEGREES": "0", "CLASP_MIS_PROMPT_LOW_DEGREE_K": "32",
    "CLASP_OP_STRICT_BUDGET_HINT": "0", "CLASP_OP_PROMPT_BUDGET_SCALE": "1.0",
    "CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE": "0", "CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE": "0",
    "CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES": "", "CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE": "",
    "CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE": "0", "CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE": "0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and path.resolve(strict=True) == path, f"Missing/linked input: {path}")
    before = path.stat(); sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "File changed while hashing")
    return {"path": str(path), "sha256": sha.hexdigest(), "size_bytes": after.st_size}


def checked_file(expected):
    current = identity(expected["path"])
    require(current["sha256"] == expected["sha256"] and current["size_bytes"] == expected.get("size_bytes", current["size_bytes"]), "Bound file changed: " + expected["path"])
    return current


def checked_tree(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            checked_file(value)
        for item in value.values():
            checked_tree(item)
    elif isinstance(value, list):
        for item in value:
            checked_tree(item)


def read_signed(path, field):
    identity(path); value = json.loads(Path(path).read_text())
    require(value.get(field) == digest({key: item for key, item in value.items() if key != field}), "Invalid self-hash: " + str(path))
    return value


def write_signed(path, value, field="receipt_sha256"):
    result = {**value, field: digest(value)}
    with Path(path).open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    return result


def verify_manifest(root, sha, filename):
    root = Path(root); manifest = checked_file({"path": str(root / filename), "sha256": sha})
    names = set()
    for line in Path(manifest["path"]).read_text().splitlines():
        parts = line.split("  "); require(len(parts) == 2, "Expected sha256sum manifest")
        expected, name = parts
        require(name not in names and not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid manifest path")
        checked_file({"path": str(root / name), "sha256": expected}); names.add(name)
    require(names, "Empty source manifest")
    return names


def read_registry(path, sha):
    bound = checked_file({"path": str(path), "sha256": sha})
    registry = json.loads(Path(path).read_text())
    require(registry["protocol"] == PROTOCOL and registry["tasks"] and set(registry["tasks"]) <= set(DIRECTIONS), "Registry must declare supported tasks")
    bundle = registry["bundle"]
    require(Path(bundle["root"]) == Path(__file__).absolute().parent, "Wrong executing report bundle")
    require(Path(__file__).name in verify_manifest(bundle["root"], bundle["manifest_sha256"], "MANAGEMENT_MANIFEST.sha256"), "Unbound executing helper")
    for task, spec in registry["tasks"].items():
        require(set(spec["methods"]) == set(METHODS), "Each task needs all five actual methods")
        require(type(spec["count"]) is int and spec["count"] > 0 and type(spec["start"]) is int and spec["start"] >= 0, "Invalid frozen evaluation slice")
        require(spec["reference"]["kind"] in ("verified_reference", "verified_heuristic", "certified_optimum"), "Unrecognized reference claim")
        require(spec.get("split", "eval") == "eval", "Only the declared development eval split is supported")
    return registry, bound


def load_module(path, prefix):
    name = prefix + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded; spec.loader.exec_module(loaded)
    return sys.modules[name]


def environment(spec, output):
    prefixes = ("CLASP_", "GRPO_", "PD_", "FOARL_", "GENERATION_", "HF_", "WANDB_", "TRANSFORMERS_", "TOKENIZERS_", "UNSLOTH_", "ACCELERATE_", "NCCL_")
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes) and key not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "PYTHONHOME", "PYTHONPATH")}
    env.update(REPRESENTATION)
    env.update({"PYTHONPATH": str(Path(spec["source"]["root"]) / "src"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "271828", "HF_HOME": str(Path(output) / "cache"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false"})
    return env


def task_tools(spec):
    source = Path(spec["source"]["root"])
    names = verify_manifest(source, spec["source"]["manifest_sha256"], "SOURCE_MANIFEST.sha256")
    require({"src/clasp/entrypoints/evaluate.py", "scripts/compare_development.py"} <= names, "Source lacks canonical evaluator/comparator")
    sys.path.insert(0, str(source / "src"))
    from clasp.entrypoints import evaluate as core
    require(Path(core.__file__).resolve() == source / "src/clasp/entrypoints/evaluate.py", "Use separate processes for different evaluation sources")
    return core, load_module(source / "scripts/compare_development.py", "all_tasks_strict_")


def dataset_contract(task, spec, core):
    files = [checked_file(item) for item in spec["data_files"]]
    actual = core.list_split_json_files(spec["data_root"], task, "eval", prefer_subset_files=False, required=True)
    require(actual == [item["path"] for item in files], "Evaluation file inventory/order differs")
    rows = core.load_and_merge_task_datasets(actual)
    selected = list(range(spec["start"], spec["start"] + spec["count"]))
    require(selected[-1] < len(rows), "Declared slice is larger than the normalized dataset")
    expected, references = [], []
    for index in selected:
        row = rows[index]
        expected.append({"example_index": index, "semantic_instance_sha256": core.semantic_instance_sha256(task, row),
                         "benchmark_row_sha256": core.benchmark_row_sha256(row), "reference_output_sha256": core.reference_output_sha256(row["output"])})
        transformed = core.transform_example_for_task(task, row["instruction"], row["input"], row["output"], row["instance_json"])
        instance = transformed["instance_json"]
        if isinstance(instance, str):
            instance = json.loads(instance)
        prompt = core.build_formatted_prompt(transformed["instruction"], transformed["input"], prompt_style="jiang_f638", no_reason=True)
        result = core.TASK_EVALUATORS[task](prompt, transformed["input"], transformed["output"], instance, transformed["output"])
        reference = result["ground_truth_objective"]
        require(result["is_feasible"] and type(reference) in (float, int) and math.isfinite(reference) and reference > 0,
                f"Invalid/nonpositive reference for {task} index{index}; gap semantics need a separate decision")
        references.append(float(reference))
    checked_tree(spec["reference"].get("evidence", []))
    if spec["reference"]["kind"] == "certified_optimum":
        certification = spec["reference"].get("certification_audit")
        require(certification, "Optimality claim needs a bound native certification audit")
        checked_file(certification); proof = read_signed(certification["path"], "receipt_sha256")
        checked_tree(proof)
        require(proof.get("task") == task and proof.get("status") == "all_selected_references_certified_optimal"
                and proof.get("data_files") == files and proof.get("start") == spec["start"] and proof.get("count") == spec["count"], "Certification does not cover this exact slice")
    return {"files": files, "ordered_rows": expected, "reference_objectives": references,
            "reference_kind": spec["reference"]["kind"], "reference_evidence": spec["reference"].get("evidence", [])}


def signed_gap(record, reference_kind):
    if not record.pass_at_k:
        return None
    objective, reference = (float(record.raw[name]) for name in ("best_objective_value", "ground_truth_objective"))
    require(math.isfinite(objective) and math.isfinite(reference) and reference > 0, "Invalid feasible objective/reference")
    sign = 1 if DIRECTIONS[record.task] == "min" else -1
    value = sign * (objective - reference) / reference
    require(math.isclose(max(0.0, value), record.best_feasible_gap, rel_tol=1e-10, abs_tol=1e-12), "Signed/clipped gaps disagree")
    require(reference_kind != "certified_optimum" or value >= -1e-12, "Policy contradicts certified optimality")
    return value


def validate_lineage(task, method, arm):
    checked_file(arm["lineage_receipt"])
    lineage = read_signed(arm["lineage_receipt"]["path"], "receipt_sha256")
    require(lineage.get("task") == task and lineage.get("method") == method and lineage.get("step") == 100
            and lineage.get("seed") == 43 and lineage.get("train_completions") == 3600,
            "Prepared terminal lineage differs from task/method/100step/seed43/3600response declaration")
    checked_tree(lineage); checked_tree(arm["checkpoint_files"])
    require(lineage.get("checkpoint_files") == arm["checkpoint_files"], "Lineage/checkpoint inventory differs")
    require(Path(arm["checkpoint"]).name == "checkpoint-100" and all(Path(item["path"]) == Path(arm["checkpoint"]) / name
            for name, item in arm["checkpoint_files"].items()), "Files do not belong to the fixed terminal100 checkpoint")
    require({"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"} <= set(arm["checkpoint_files"]), "Policy inventory incomplete")
    require(lineage.get("status") in ("verified_original_terminal_lineage", "verified_terminal_training"), "A successful original training audit is required")
    return lineage


def command(task, method, spec, arm):
    argv = [spec["runtime_python"], "-B", "-m", "clasp.entrypoints.evaluate", "--evaluation_purpose", "development",
        "--checkpoint_path", arm["checkpoint"], "--stage", "grpo", "--tasks", task, "--data_root", spec["data_root"],
        "--split", "eval", "--eval_start_index", str(spec["start"]), "--max_eval_samples_per_task", str(spec["count"]),
        "--no-prefer_subset_files", "--prompt_style", "jiang_f638", "--force_no_reason_prompt", "--save_results", "--save_details",
        "--no-update_best_run", "--resume", "--inference_root", str(Path(arm["output"]) / "eval"), "--eval_name", f"{task}_{method}_terminal100_raw8"]
    for name, value in GENERATION.items():
        argv += ["--" + name] if value is True else ["--" + name, str(value)]
    return argv


def wrapper_execution(executed_slurm, bundle, source):
    require(executed_slurm is not None, "Run requires the actual executed Slurm wrapper")
    filename = "evaluate_all_tasks_a100.slurm"
    require(filename in verify_manifest(bundle["root"], bundle["manifest_sha256"], "MANAGEMENT_MANIFEST.sha256"), "Evaluation wrapper is not manifest-bound")
    actual = identity(executed_slurm); frozen = identity(Path(bundle["root"]) / filename)
    require((actual["sha256"], actual["size_bytes"]) == (frozen["sha256"], frozen["size_bytes"]), "Executed evaluation wrapper differs from frozen bytes")
    return {"executed_slurm": actual, "frozen_slurm": frozen,
            "bundle_manifest": identity(Path(bundle["root"]) / "MANAGEMENT_MANIFEST.sha256"), "source": source}


def validate_wrapper_execution(execution, bundle, source):
    frozen = checked_file(execution["frozen_slurm"])
    require(Path(frozen["path"]) == Path(bundle["root"]) / "evaluate_all_tasks_a100.slurm"
            and execution["source"] == source, "Evaluation wrapper source binding differs")
    manifest = checked_file(execution["bundle_manifest"])
    require(manifest["path"] == str(Path(bundle["root"]) / "MANAGEMENT_MANIFEST.sha256")
            and manifest["sha256"] == bundle["manifest_sha256"], "Evaluation launch bundle changed")
    require("evaluate_all_tasks_a100.slurm" in verify_manifest(bundle["root"], bundle["manifest_sha256"], "MANAGEMENT_MANIFEST.sha256"), "Unbound evaluation wrapper")
    actual = execution["executed_slurm"]
    # Slurm deletes the spool script when the job ends; the admission receipt
    # preserves its observed identity and equality with the immutable wrapper.
    require((actual["sha256"], actual["size_bytes"]) == (frozen["sha256"], frozen["size_bytes"]), "Executed evaluation wrapper did not match frozen bytes")


def run_evaluation(task, method, spec, arm, registry_identity, *, executed_slurm, bundle):
    require("existing" not in arm, "This arm declares historical raw reuse; generation is unnecessary")
    execution = wrapper_execution(executed_slurm, bundle, spec["source"])
    lineage = validate_lineage(task, method, arm)
    core, _ = task_tools(spec); benchmark = dataset_contract(task, spec, core)
    require(os.environ.get("SLURM_JOB_ID") and Path(sys.executable).resolve() == Path(spec["runtime_python"]).resolve(), "Use the declared native runtime in an allocated job")
    import torch
    require(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0), "Exactly one A100 required")
    output = Path(arm["output"])
    require(output.is_absolute() and output.resolve() == output and not output.exists(), "Evaluation output must be new and unlinked")
    output.mkdir(parents=True)
    write_signed(output / "launch.json", {"protocol": PROTOCOL, "task": task, "method": method, "registry": registry_identity,
        "lineage": arm["lineage_receipt"], "dataset": benchmark, "command": command(task, method, spec, arm),
        "execution": execution, "job_id": os.environ["SLURM_JOB_ID"]})
    result = subprocess.run(command(task, method, spec, arm), cwd=spec["source"]["root"], env=environment(spec, output))
    write_signed(output / "exit.json", {"returncode": result.returncode, "job_id": os.environ["SLURM_JOB_ID"]})
    require(result.returncode == 0, "Evaluation failed; incomplete raw output retained")
    _, evidence = audit_arm(task, method, spec, arm, core, benchmark, registry_identity)
    return save_arm_index(task, method, spec, arm, benchmark, evidence, registry_identity)


def audit_arm(task, method, spec, arm, core, benchmark, registry_identity):
    lineage = validate_lineage(task, method, arm)
    _, strict = task_tools(spec)
    if "existing" in arm:
        checked_tree(arm["existing"])
        details, config_path = (arm["existing"][key]["path"] for key in ("details", "run_config"))
        provenance = {"mode": "historical_raw_reuse", "original": arm["existing"]}
    else:
        output = Path(arm["output"])
        launch = read_signed(output / "launch.json", "receipt_sha256"); exit_receipt = read_signed(output / "exit.json", "receipt_sha256")
        original_registry = spec.get("evaluation_registry", registry_identity)
        checked_file(original_registry)
        original_bundle = json.loads(Path(original_registry["path"]).read_text())["bundle"]
        validate_wrapper_execution(launch["execution"], original_bundle, spec["source"])
        require(launch["registry"] == original_registry and launch["task"] == task and launch["method"] == method
                and launch["lineage"] == arm["lineage_receipt"] and launch["dataset"] == benchmark
                and exit_receipt["returncode"] == 0 and exit_receipt["job_id"] == launch["job_id"], "Fresh evaluation launch/exit binding differs")
        detail_paths = list(output.glob("eval/**/run_details__*.jsonl")); config_paths = list(output.glob("eval/**/run_config__*.json"))
        require(len(detail_paths) == len(config_paths) == 1, "Need exactly one complete raw evaluation")
        details, config_path = str(detail_paths[0]), str(config_paths[0]); provenance = {"mode": "fresh", "launch": identity(output / "launch.json"), "exit": identity(output / "exit.json")}
    artifact = strict._load_artifact(details, config_path, label=task + "/" + method)
    config = read_signed(config_path, "run_config_sha256")
    require(artifact.artifact_type == "native_raw" and config["run_status"] == "complete" and config["evaluation_purpose"] == "development"
        and config["tasks"] == [task] and config["stage"] == "grpo" and Path(config["resolved_model_path"]) == Path(arm["checkpoint"]), "Raw task/method/model/scope mismatch")
    require(all(config["generation"].get(key) == value for key, value in GENERATION.items()) and config["best_run_update_requested"] is False
        and config["benchmark_manifest"] is None and config["campaign_manifest"] is None, "Raw generation or selection mismatch")
    require(config["generation"].get("seed_policy") == "per_example_sha256_v1"
        and all(config["generation"].get(key) == [] for key in ("temperature_schedule", "top_p_schedule", "op_size_band_top_p_schedule")),
        "Per-candidate generation schedules differ from the fixed raw recipe")
    require(config["prompting"]["no_reason"] is True and config["prompting"]["prompt_style_resolved"] == "jiang_f638", "Prompt recipe mismatch")
    require(Path(config["data_root"]) == Path(spec["data_root"]) and config["limits"]["eval_start_index"] == spec["start"]
        and config["limits"]["max_eval_samples_per_task"] == spec["count"] and config["limits"]["prefer_subset_files"] is False, "Dataset selection mismatch")
    model = config["experiment_provenance"]["model"]["checkpoint_artifact"]
    require(model["manifest_sha256"] == digest(model["files"]), "Corrupted model manifest")
    seen = set()
    for item in model["files"]:
        name = item["path"]; bound = arm["checkpoint_files"].get(name)
        require(name not in seen and bound and bound["sha256"] == item["sha256"] and bound["size_bytes"] == item["size"], "Raw policy differs from trained endpoint")
        seen.add(name)
    require({"adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"} <= seen, "Raw policy manifest incomplete")
    require(artifact.strict_verifier == core._strict_verifier_identity(), "Canonical verifier source differs")
    records = sorted(artifact.records.values(), key=lambda row: row.example_index)
    require(len(records) == spec["count"] and artifact.candidate_budget == 8, "Incomplete denominator")
    for row, expected, reference in zip(records, benchmark["ordered_rows"], benchmark["reference_objectives"], strict=True):
        require(all(row.raw.get(key) == value for key, value in expected.items()) and not row.raw["prompt_was_truncated"], "Row identity/truncation mismatch")
        raw = row.raw; instance = raw["instance_json"]
        if isinstance(instance, str):
            instance = json.loads(instance)
        prompt = core.build_formatted_prompt(raw["instruction"], raw["input_problem"], prompt_style="jiang_f638", no_reason=True)
        for candidate in raw["candidates"]:
            metrics = core.TASK_EVALUATORS[task](prompt, raw["input_problem"], raw["ground_truth_output"], instance, candidate["completion"])
            require(all(same_value(candidate.get(key), value) for key, value in metrics.items()), "Canonical raw re-scoring mismatch")
            require(same_value(raw["ground_truth_objective"], metrics["ground_truth_objective"]) and same_value(reference, metrics["ground_truth_objective"]), "Reference objective mismatch")
        selected = core._select_best_candidate_index(task, raw["candidates"])
        require(selected == raw["best_candidate_index"] and same_value(raw["best_objective_value"], raw["candidates"][selected]["objective_value"] if selected is not None else None), "Raw best-candidate selection mismatch")
        signed_gap(row, spec["reference"]["kind"])
    evidence = {"task": task, "method": method, "details": identity(details), "run_config": identity(config_path), "lineage": arm["lineage_receipt"],
                "training_recipe": arm["training_recipe"], "original_execution": provenance, "strict_verifier": artifact.strict_verifier,
                "protocol_fingerprint": artifact.protocol_fingerprint, "canonical_candidates_rechecked": len(records) * 8}
    return artifact, evidence


def same_value(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) if type(a) in (int, float) and type(b) in (int, float) else a == b


def save_arm_index(task, method, spec, arm, benchmark, evidence, registry_identity):
    result = {"protocol": PROTOCOL, "status": "complete_raw_evaluation_canonical_audit", "task": task, "method": method,
              "registry": registry_identity, "evaluation_source": spec["source"], "dataset": benchmark, "evidence": evidence}
    output = Path(arm["output"])
    require(output.is_absolute() and output.resolve() == output, "Index output must be absolute and unlinked")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "result_index.json"
    if path.exists():
        existing = read_signed(path, "receipt_sha256")
        require(existing == {**result, "receipt_sha256": digest(result)}, "Existing index differs from live evidence")
        return existing
    return write_signed(path, result)


def index_arm(task, method, spec, arm, registry_identity):
    os.environ.update(REPRESENTATION)
    core, _ = task_tools(spec); benchmark = dataset_contract(task, spec, core)
    _, evidence = audit_arm(task, method, spec, arm, core, benchmark, registry_identity)
    return save_arm_index(task, method, spec, arm, benchmark, evidence, registry_identity)


def summarize(records, task, method, reference_kind):
    n = len(records); require(n > 0, "Empty report denominator")
    successful = [row for row in records if row.pass_at_k]
    candidates = [candidate for row in records for candidate in row.raw["candidates"]]
    require(len(candidates) == n * 8, "Every instance must retain eight raw candidates")
    first = sum(row.raw["candidates"][0]["is_feasible"] for row in records)
    count = sum(candidate["is_feasible"] for candidate in candidates)
    result = {"task": task, "method": method, "instances": n, "candidate_count": n * 8,
        "feasible_at_1_count": first, "feasible_at_1_percent": 100 * first / n,
        "feasible_at_8_count": len(successful), "feasible_at_8_percent": 100 * len(successful) / n,
        "feasible_candidate_count": count, "candidate_feasibility_percent": 100 * count / (n * 8),
        "own_success_gap_denominator": len(successful), "reference_kind": reference_kind,
        "mean_signed_reference_gap_percent": 100 * statistics.fmean(signed_gap(row, reference_kind) for row in successful) if successful else None}
    for label, threshold in GAP_THRESHOLDS.items():
        value = sum(signed_gap(row, reference_kind) <= threshold + 1e-12 for row in successful)
        result["success_within_" + label + "_percent"] = 100 * value / n
    return result


def paired_quality(left, right, kind, strict):
    strict._validate_pairing(left, right)
    pairs = [(left.records[key], right.records[key]) for key in sorted(left.records)]
    joint = [(a, b) for a, b in pairs if a.pass_at_k and b.pass_at_k]
    first = [int(b.raw["candidates"][0]["is_feasible"]) - int(a.raw["candidates"][0]["is_feasible"]) for a, b in pairs]
    gaps = [signed_gap(b, kind) - signed_gap(a, kind) for a, b in joint]
    def contrast(values, metric, unit):
        return {"denominator": len(values), "right_minus_left": statistics.fmean(values) if values else None,
                "paired_bootstrap": strict._paired_bootstrap_ci(values, scope="all_tasks", metric=metric, resampling_unit=unit)}
    return {"first_sample_feasibility": contrast(first, "feasible_at_1", "paired_instance"),
        "joint_signed_gap": contrast(gaps, "signed_gap", "jointly_feasible_paired_instance"), "joint_success_count": len(joint),
        "joint_left_signed_gap": statistics.fmean(signed_gap(a, kind) for a, b in joint) if joint else None,
        "joint_right_signed_gap": statistics.fmean(signed_gap(b, kind) for a, b in joint) if joint else None}


def task_report(task, spec, registry_identity, output):
    os.environ.update(REPRESENTATION)
    core, strict = task_tools(spec); dataset = dataset_contract(task, spec, core)
    arms, evidence = {}, {}
    for method in METHODS:
        arms[method], evidence[method] = audit_arm(task, method, spec, spec["methods"][method], core, dataset, registry_identity)
        strict._validate_pairing(arms[METHODS[0]], arms[method])
    kind = spec["reference"]["kind"]; rows = [summarize(list(arms[method].records.values()), task, method, kind) for method in METHODS]
    comparisons = {left + "_vs_" + right: {"strict_comparison": strict.build_receipt(arms[left], arms[right], left_label=left, right_label=right),
        **paired_quality(arms[left], arms[right], kind, strict)} for left, right in itertools.combinations(METHODS, 2)}
    common = [key for key in arms[METHODS[0]].records if all(arms[method].records[key].pass_at_k for method in METHODS)]
    thresholds = set(GAP_THRESHOLDS.values()) | {signed_gap(row, kind) for arm in arms.values() for row in arm.records.values() if row.pass_at_k}
    cdf = [{"task": task, "method": method, "signed_gap_threshold": threshold, "instances": spec["count"],
            "success_fraction": sum(row.pass_at_k and signed_gap(row, kind) <= threshold + 1e-12 for row in arms[method].records.values()) / spec["count"]}
           for threshold in sorted(thresholds) for method in METHODS]
    payload = {"protocol": PROTOCOL, "status": "complete_five_methods_within_task", "task": task, "registry": registry_identity,
        "evaluation_source": spec["source"], "dataset": dataset, "reference": spec["reference"], "evidence": evidence,
        "summary": rows, "paired_comparisons": comparisons, "quality_cdf": cdf,
        "all_five_joint_success": {"count": len(common), "mean_signed_gap": {method: statistics.fmean(signed_gap(arms[method].records[key], kind) for key in common) if common else None for method in METHODS}}}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    return write_signed(output / "task_report.json", payload)


def availability(registry):
    missing = []
    for task in DIRECTIONS:
        if task not in registry["tasks"]:
            missing.extend(task + "/" + method for method in METHODS)
            continue
        spec = registry["tasks"][task]
        for method, arm in spec["methods"].items():
            paths = [arm.get("lineage_receipt", {}).get("path"), arm.get("checkpoint")]
            if "existing" in arm:
                paths += [arm["existing"].get(key, {}).get("path") for key in ("details", "run_config")]
            else:
                paths += [str(Path(arm.get("output", "/missing")) / "exit.json")]
            if any(not path or not Path(path).exists() for path in paths):
                missing.append(task + "/" + method)
    return {"status": "all_artifacts_present_pending_validation" if not missing else "incomplete_no_final_comparison",
            "missing_task_methods": missing, "expected_rows": 25}


def compare_all(registry, registry_identity, registry_path, sha, output):
    status = availability(registry)
    if status["missing_task_methods"]:
        return status
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for task, spec in registry["tasks"].items():
        argv = [spec["runtime_python"], "-B", str(Path(__file__).absolute()), "audit-task", "--registry", str(registry_path), "--registry-sha256", sha, "--task", task, "--output", str(output / task)]
        with (output / (task + "_audit.log")).open("x") as log:
            result = subprocess.run(argv, cwd=spec["source"]["root"], env=environment(spec, output / task), stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, "Task audit failed: " + task + "; no final comparison table published")
        reports[task] = read_signed(output / task / "task_report.json", "receipt_sha256")
    rows = [row for task in DIRECTIONS for row in reports[task]["summary"]]
    require(len(rows) == 25 and {(row["task"], row["method"]) for row in rows} == set(itertools.product(DIRECTIONS, METHODS)), "Final report is incomplete")
    result = write_signed(output / "comparison.json", {"protocol": PROTOCOL, "status": "complete_all_five_tasks_all_five_methods",
        "registry": registry_identity, "task_reports": {task: identity(output / task / "task_report.json") for task in DIRECTIONS}, "summary": rows,
        "limitations": ["Task-specific sources, datasets, references and training lineages remain distinct; exact pairing is within each task.",
            "Reused development data and one training seed; not confirmation, guaranteed superiority or official Jiang reproduction.",
            "Gap is conditional on success; interpret alongside feasibility, paired shared-success gaps and the all-instance CDF.",
            "Reference optimality is claimed only where a matching certification audit exists. Negative better-than-reference gaps remain signed.",
            "Training recipes and compute may differ; all original lineage, decoder, reward, KL and continuation metadata are retained.",
            "Instance bootstrap intervals do not quantify training-seed variation; comparisons are descriptive and unadjusted."]})
    for filename, values in (("table.csv", rows), ("quality_cdf.csv", [item for report in reports.values() for item in report["quality_cdf"]])):
        with (output / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["| Task | Method | Feasible@1 | Feasible@8 | Feasible candidates | Signed gap on own successes | Reference |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        gap = "undefined" if row["mean_signed_reference_gap_percent"] is None else f"{row['mean_signed_reference_gap_percent']:.3f}% (n={row['own_success_gap_denominator']})"
        lines.append(f"| {row['task']} | {row['method']} | {row['feasible_at_1_count']}/{row['instances']} | {row['feasible_at_8_count']}/{row['instances']} | {row['feasible_candidate_count']}/{row['candidate_count']} | {gap} | {row['reference_kind']} |")
    (output / "table.md").write_text("\n".join(lines) + "\n\nAll25 rows validated. Feasible@1 is the first of eight sampled outputs. Task reports retain paired and shared-success gaps; quality_cdf.csv retains failed-instance denominators. Training and reference provenance differ where declared.\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "plan", "run", "index", "audit-task", "compare-all"))
    parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--task", choices=DIRECTIONS); parser.add_argument("--method", choices=METHODS); parser.add_argument("--output", type=Path)
    parser.add_argument("--executed-slurm", type=Path)
    args = parser.parse_args(); registry, bound = read_registry(args.registry, args.registry_sha256)
    if args.action == "status":
        result = availability(registry)
    elif args.action == "compare-all":
        require(args.output, "New final report directory required")
        result = compare_all(registry, bound, args.registry, args.registry_sha256, args.output)
    else:
        require(args.task, "Task required"); spec = registry["tasks"][args.task]
        if args.action == "audit-task":
            require(args.output, "New task report directory required")
            result = task_report(args.task, spec, bound, args.output)
        else:
            require(args.method, "Actual method required"); arm = spec["methods"][args.method]
            if args.action == "plan":
                result = {"status": "plan_no_generation_or_submission", "command": command(args.task, args.method, spec, arm)}
            elif args.action == "index":
                result = index_arm(args.task, args.method, spec, arm, bound)
            else:
                os.environ.update(REPRESENTATION)
                run_evaluation(args.task, args.method, spec, arm, bound, executed_slurm=args.executed_slurm, bundle=registry["bundle"])
                result = {"status": "completed_raw_evaluation_and_canonical_audit", "task": args.task, "method": args.method}
    print(json.dumps({key: value for key, value in result.items() if key in ("status", "task", "method", "command", "missing_task_methods", "expected_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
