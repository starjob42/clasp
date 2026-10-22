"""Bounded raw DEV100 evaluation for the three external-baseline pilot arms.

Wraps the existing evaluator, benchmark validator and strict paired comparator.
There is no scheduler submission, repair, checkpoint selection or confirmation
data access. `run` needs an allocated A100; `plan` only prints the command.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
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

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/lustre/fsn1/projects/rech/vas/uag72kz")
RUNTIME = Path("/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env")
DEV = SCRATCH / "clasp_fair_development_benchmark_20260905_v1/data_v3"
SFT = SCRATCH / "clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14"
TASKS = ("cvrp", "mis")
ARMS = ("rloo_b9", "external_b9", "external_b1")
METHODS = (*ARMS, "sft")
GAP_THRESHOLDS = {"0pct": 0.0, "1pct": 0.01, "3pct": 0.03, "5pct": 0.05, "10pct": 0.1}
PROTOCOL = "external-baseline-inspected-dev100-raw8-20260910-v1"
GENERATION = {"seed": 271828, "dtype": "float16", "num_return_sequences": 8,
              "generation_batch_size": 8, "do_sample": True, "temperature": 0.7,
              "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0,
              "max_new_tokens": 3000, "max_prompt_length": 29768}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, relative):
    name = "external_baseline_eval_" + name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def helpers():
    sys.path.insert(0, str(ROOT / "src"))
    return (module("benchmark", "scripts/clasp_grpo_fresh_dev_artifacts.py"),
            module("strict", "scripts/compare_development.py"),
            module("training", "scripts/run_external_baseline_dev_chunk.py"))


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stage(method):
    return "sft" if method == "sft" else "grpo"


def evaluation_command(task, method, checkpoint, output):
    require(task in TASKS and method in METHODS, "Unsupported development task/arm")
    require(Path(checkpoint).name == ("checkpoint-31250" if method == "sft" else "checkpoint-100"),
            "Evaluate only the initial SFT or predeclared terminal step100")
    argv = [str(RUNTIME / "bin/python"), "-B", "-m", "clasp.entrypoints.evaluate",
            "--evaluation_purpose", "development", "--checkpoint_path", str(checkpoint),
            "--stage", stage(method), "--tasks", task, "--data_root", str(DEV), "--split", "eval",
            "--eval_start_index", "0", "--max_eval_samples_per_task", "100", "--no-prefer_subset_files",
            "--prompt_style", "jiang_f638", "--force_no_reason_prompt", "--size_bucket", "all",
            "--min_instance_size", "0", "--max_instance_size", "0", "--save_results", "--save_details",
            "--no-update_best_run", "--resume", "--inference_root", str(Path(output) / "eval"),
            "--eval_name", f"{method}_{task}_terminal_dev100_raw8_seed271828"]
    for name, value in GENERATION.items():
        argv += ["--" + name] if value is True else ["--" + name, str(value)]
    return argv


def evaluation_environment(source, output, method, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    prefixes = ("CLASP_", "PD_", "GRPO_", "FOARL_", "GENERATION_", "WANDB_", "HF_",
                "TRANSFORMERS_", "TOKENIZERS_", "BITSANDBYTES_", "UNSLOTH_", "ACCELERATE_", "NCCL_")
    names = {"PYTHONPATH", "PYTHONHOME", "RESUME_FROM_CHECKPOINT", "WORLD_SIZE", "RANK", "LOCAL_RANK"}
    env = {key: value for key, value in env.items() if key not in names and not key.startswith(prefixes)}
    env.update({"PYTHONPATH": str(Path(source) / "src"), "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "271828",
                "HF_HOME": str(Path(output) / "cache"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false",
                "CLASP_PROJECT_ROOT": str(source), "CLASP_TSP_PROMPT_KNN": "0",
                "CLASP_TSP_INPUT_ORDER": "", "CLASP_TSP_RELABEL_ORDER": "",
                "CLASP_TSP_ROUTE_CANONICALIZE": "", "CLASP_TSP_LABEL_TTA_SCHEDULE": "",
                "CLASP_TSP_PROMPT_KNN_DISTANCES": "" if method == "sft" else "0",
                "CLASP_TSP_ROUTE_ONLY_OUTPUT": "" if method == "sft" else "0",
                "CLASP_MIS_RELABEL_ORDER": "", "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0",
                "CLASP_MIS_PROMPT_NODE_DEGREES": "0", "CLASP_MIS_PROMPT_LOW_DEGREE_K": "32",
                "CLASP_OP_STRICT_BUDGET_HINT": "0", "CLASP_OP_PROMPT_BUDGET_SCALE": "1.0",
                "CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE": "0", "CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE": "0",
                "CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES": "", "CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE": "",
                "CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE": "0", "CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE": "0"})
    return env


def verify_source(source_sha):
    from clasp.baselines.external_baseline_dev import verify_source_manifest
    binding = verify_source_manifest(ROOT / "SOURCE_MANIFEST.sha256", source_sha, source_root=ROOT)
    listed = {line.split(maxsplit=1)[1].lstrip("*")
              for line in (ROOT / "SOURCE_MANIFEST.sha256").read_text().splitlines() if line.strip()}
    required = {"scripts/evaluate_external_baseline_dev.py", "scripts/evaluate_external_baseline_dev_a100.slurm",
                "scripts/compare_development.py", "scripts/clasp_grpo_fresh_dev_artifacts.py",
                "scripts/run_external_baseline_dev_chunk.py"}
    require(required <= listed, "Source manifest omits evaluation or admission implementations")
    return binding


def validate_checkpoint(checkpoint, task, method, source_sha, training):
    checkpoint = Path(checkpoint).absolute()
    require(checkpoint == checkpoint.resolve(strict=True), "Checkpoint path may not traverse symlinks")
    sft = SFT / f"runs/{task}/seed42/checkpoints/checkpoint-31250"
    if method == "sft":
        require(checkpoint == sft, "SFT evaluation must use this task's exact common initialization")
        from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
        verify_checkpoint_fidelity(checkpoint)
        material = None
    else:
        require(checkpoint.name == "checkpoint-100", "Only terminal100 may enter the pilot comparison")
        contract_path = checkpoint.parent.parent / training.CONTRACT_NAME
        contract = training.load_json(contract_path)
        require(contract.get("task") == task and contract.get("method") == method
                and contract.get("diagnostic") is False and contract.get("seed") == 43,
                "Wrong task/arm/seed or engineering checkpoint")
        require(contract.get("source") == {"root": str(ROOT), "manifest_sha256": source_sha},
                "Training source differs from frozen evaluator source")
        require(contract.get("initialization", {}).get("checkpoint") == str(sft)
                and contract.get("beta") == 0.0 and contract.get("teacher_weight") == 0.0,
                "Training initialization, KL or teacher contract differs")
        material = training.validate_resume(checkpoint, contract, contract_path, 100)
    files = {name: training.identity(checkpoint / name) for name in
             ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}
    init_hash = training.identity(sft / "adapter_model.safetensors")["sha256"]
    require(init_hash == training.SFT_HASHES[task], "Initial SFT adapter hash changed")
    return files, material


def launch(args):
    fresh, _, training = helpers()
    source_binding = verify_source(args.source_manifest_sha256)
    output = args.output.absolute()
    require(ROOT.is_relative_to(SCRATCH) and output.is_relative_to(SCRATCH)
            and output == output.resolve(strict=False), "Use unlinked SCRATCH source and output paths")
    require(not output.exists(), "Fresh evaluation output already exists; use index for completed artifacts")
    require(not any(path.is_relative_to(output) for path in (ROOT, DEV, args.checkpoint.resolve())),
            "Evaluation output cannot contain source, data or checkpoint inputs")
    require(os.environ.get("SLURM_JOB_ID") and args.executed_slurm, "Run inside the declared evaluation allocation")
    wrapper = ROOT / "scripts/evaluate_external_baseline_dev_a100.slurm"
    require(fresh.identity(args.executed_slurm)["sha256"] == fresh.identity(wrapper)["sha256"],
            "Executed Slurm wrapper differs from frozen source")
    require(Path(sys.executable).resolve() == (RUNTIME / "bin/python").resolve(), "Wrong native Python runtime")
    import torch
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    require(len(gpu_names) == 1 and "A100" in gpu_names[0], "Exactly one visible A100 is required")
    benchmark = fresh.validate_benchmark(DEV)
    require(list((DEV / args.task / "eval").glob("*.json")) == [DEV / args.task / "eval/test.json"],
            "Extra eval JSON files would change deterministic input ordering")
    checkpoint, terminal = validate_checkpoint(args.checkpoint, args.task, args.arm, args.source_manifest_sha256, training)
    command = evaluation_command(args.task, args.arm, args.checkpoint, output)
    env = evaluation_environment(ROOT, output, args.arm)
    payload = {"schema_version": 1, "protocol": PROTOCOL, "status": "validated_inspected_development_launch",
               "task": args.task, "method": args.arm, "stage": stage(args.arm), "created_at_utc": timestamp(),
               "generation": GENERATION, "evaluation_slice": {"split": "eval", "start": 0, "count": 100},
               "repair": False, "training_seed": 42 if args.arm == "sft" else 43,
               "representation": "jiang_f638_sft" if args.arm == "sft" else "jiang_f638_clasp",
               "benchmark": benchmark, "checkpoint": checkpoint, "terminal_training": terminal,
               "source_manifest": fresh.identity(ROOT / "SOURCE_MANIFEST.sha256"),
               "source_verified_file_count": source_binding["verified_file_count"],
               "management": {"wrapper": fresh.identity(wrapper), "helper": fresh.identity(__file__)},
               "runtime_python": fresh.identity(RUNTIME / "bin/python"), "command": command,
               "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu_names": gpu_names,
               "reference_semantics": {"cvrp": "gap_to_verified_heuristic_reference", "mis": "gap_to_certified_optimum"}}
    output.mkdir(parents=True, exist_ok=False)
    fresh.write_new(output / "launch_receipt.json", payload, "launch_sha256")
    result = subprocess.run(command, env=env, cwd=ROOT)
    fresh.write_new(output / "evaluation_exit.json", {"returncode": result.returncode,
                    "finished_at_utc": timestamp(), "slurm_job_id": os.environ["SLURM_JOB_ID"]}, "exit_sha256")
    require(result.returncode == 0, f"Evaluator failed with code {result.returncode}; raw partial artifacts retained")
    return index(output)


def same_value(left, right):
    if type(left) in (float, int) and type(right) in (float, int):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def canonical_rescore(artifact):
    """Read generated text verbatim; call the same canonical verifier again."""
    from clasp.entrypoints import evaluate as evaluator
    from clasp.data.prompts import build_formatted_prompt
    require(artifact.strict_verifier == evaluator._strict_verifier_identity(), "Canonical verifier source changed")
    count = 0
    for record in artifact.records.values():
        raw = record.raw
        prompt = build_formatted_prompt(raw["instruction"], raw["input_problem"],
                                        prompt_style="jiang_f638", no_reason=True)
        instance = raw["instance_json"]
        if isinstance(instance, str):
            instance = json.loads(instance)
        for candidate in raw["candidates"]:
            require(isinstance(candidate.get("completion"), str), "Missing verbatim raw completion")
            recomputed = evaluator.TASK_EVALUATORS[record.task](prompt, raw["input_problem"],
                            raw["ground_truth_output"], instance, candidate["completion"])
            require(all(same_value(candidate.get(key), value) for key, value in recomputed.items()),
                    f"Canonical candidate re-score mismatch: {record.task}/{record.example_index}/{candidate['candidate_index']}")
            require(same_value(raw.get("ground_truth_objective"), recomputed.get("ground_truth_objective")),
                    "Top-level reference objective differs from canonical reference")
            count += 1
        best = evaluator._select_best_candidate_index(record.task, raw["candidates"])
        require(best == raw["best_candidate_index"], "Best-of-eight candidate selection differs")
        expected_objective = raw["candidates"][best]["objective_value"] if best is not None else None
        require(same_value(raw["best_objective_value"], expected_objective), "Selected objective differs")
    return {"status": "all_verbatim_completions_reverified", "candidate_count": count,
            "strict_verifier": artifact.strict_verifier}


def validate_evaluation(output):
    fresh, strict, training = helpers()
    output = Path(output).resolve(strict=True)
    receipt = fresh.checked_json(output / "launch_receipt.json", "launch_sha256")
    require(receipt.get("protocol") == PROTOCOL and receipt["method"] in METHODS and receipt["task"] in TASKS,
            "Not a declared external-baseline development evaluation")
    verify_source(receipt["source_manifest"]["sha256"])
    require(fresh.validate_benchmark(DEV) == receipt["benchmark"], "Inspected DEV100 benchmark changed")
    checkpoint = Path(receipt["checkpoint"]["adapter_model.safetensors"]["path"]).parent
    files, terminal = validate_checkpoint(checkpoint, receipt["task"], receipt["method"],
                                          receipt["source_manifest"]["sha256"], training)
    require(files == receipt["checkpoint"] and terminal == receipt["terminal_training"], "Training/checkpoint inputs changed")
    exit_receipt = fresh.checked_json(output / "evaluation_exit.json", "exit_sha256")
    require(exit_receipt["returncode"] == 0 and exit_receipt["slurm_job_id"] == receipt["slurm_job_id"],
            "Evaluation process did not exit successfully in the launch allocation")
    # Existing helper only writes its final index. Capture it while reusing every
    # raw/header/source/benchmark check, and publish after canonical re-scoring.
    original_write = fresh.write_new
    fresh.write_new = lambda path, value, field: {**value, field: fresh.digest(value)}
    try:
        result = fresh.index(output)
    finally:
        fresh.write_new = original_write
    artifact = strict._load_artifact(result["details"]["path"], result["run_config"]["path"], label=receipt["method"])
    require(artifact.artifact_type == "native_raw", "Only raw generation artifacts are eligible")
    config = fresh.checked_json(result["run_config"]["path"], "run_config_sha256")
    require(config["evaluation_purpose"] == "development" and config["best_run_update_requested"] is False
            and config["benchmark_manifest"] is None and config["campaign_manifest"] is None,
            "Evaluation purpose or checkpoint-selection policy differs")
    require(Path(config["data_root"]).resolve() == DEV, "Evaluation used a different data root")
    require(config["generation"]["seed_policy"] == "per_example_sha256_v1"
            and config["generation"]["temperature_schedule"] == [] and config["generation"]["top_p_schedule"] == [],
            "Seed policy or per-candidate sampling differs")
    require(not any(row.raw["prompt_was_truncated"] for row in artifact.records.values()), "A DEV100 prompt was truncated")
    result.update(protocol=PROTOCOL, training_seed=receipt["training_seed"],
                  status="complete_inspected_external_baseline_development_only",
                  canonical_rescore=canonical_rescore(artifact), terminal_training=terminal)
    result.pop("index_sha256", None)
    return result, artifact


def index(output):
    fresh, _, _ = helpers()
    result, _ = validate_evaluation(output)
    path = Path(output) / "result_index.json"
    if path.exists():
        existing = fresh.checked_json(path, "index_sha256")
        require(existing == {**result, "index_sha256": fresh.digest(result)}, "Existing index differs from current raw evidence")
        return existing
    return fresh.write_new(path, result, "index_sha256")


def signed_gap(record):
    if not record.pass_at_k:
        return None
    objective, reference = (float(record.raw[key]) for key in ("best_objective_value", "ground_truth_objective"))
    require(math.isfinite(objective) and math.isfinite(reference) and reference > 0, "Invalid feasible objective/reference")
    value = (objective - reference) / reference if record.task == "cvrp" else (reference - objective) / reference
    require(math.isclose(max(0.0, value), record.best_feasible_gap, rel_tol=1e-10, abs_tol=1e-12), "Signed/clipped gaps disagree")
    require(record.task != "mis" or value >= -1e-12, "MIS result contradicts the certified optimum")
    return value


def summarize(records, task, method):
    require(len(records) == 100 and all(row.task == task and len(row.raw["candidates"]) == 8 for row in records),
            "Exactly100 instances and800 raw candidates are required")
    successes = [row for row in records if row.pass_at_k]
    candidates = [candidate for row in records for candidate in row.raw["candidates"]]
    first = sum(row.raw["candidates"][0]["is_feasible"] for row in records)
    result = {"task": task, "method": method, "instances": 100, "candidates": 800,
            "feasible_at_1_count": first, "feasible_at_1_percent": float(first),
            "feasible_at_8_count": len(successes), "feasible_at_8_percent": float(len(successes)),
            "feasible_candidate_count": sum(candidate["is_feasible"] for candidate in candidates),
            "candidate_feasibility_percent": 100 * sum(candidate["is_feasible"] for candidate in candidates) / 800,
            "own_success_gap_denominator": len(successes),
            "mean_signed_reference_gap_percent": 100 * statistics.fmean(signed_gap(row) for row in successes) if successes else None,
            "mean_clipped_reference_gap_percent": 100 * statistics.fmean(row.best_feasible_gap for row in successes) if successes else None,
            "better_than_reference_instances": sum(signed_gap(row) < -1e-12 for row in successes),
            "completion_tokens": sum(candidate["completion_tokens"] for candidate in candidates),
            "generation_time_sec": sum(row.raw["generation_time_sec"] for row in records),
            "reference": "verified_heuristic" if task == "cvrp" else "certified_optimum"}
    for label, threshold in GAP_THRESHOLDS.items():
        count = sum(signed_gap(row) <= threshold + 1e-12 for row in successes)
        result[f"success_within_gap_{label}_count"] = count
        result[f"success_within_gap_{label}_percent"] = float(count)
    return result


def paired_extra(left, right, strict):
    strict._validate_pairing(left, right)
    pairs = [(left.records[key], right.records[key]) for key in sorted(left.records)]
    joint = [(a, b) for a, b in pairs if a.pass_at_k and b.pass_at_k]
    first = [int(b.raw["candidates"][0]["is_feasible"]) - int(a.raw["candidates"][0]["is_feasible"]) for a, b in pairs]
    gap = [signed_gap(b) - signed_gap(a) for a, b in joint]
    def contrast(values, name, unit):
        return {"denominator": len(values), "right_minus_left": statistics.fmean(values) if values else None,
                "paired_bootstrap": strict._paired_bootstrap_ci(values, scope="external_dev100", metric=name, resampling_unit=unit)}
    result = {"feasible_at_1": contrast(first, "first_candidate_feasibility", "paired_instance"),
            "jointly_feasible_signed_gap": contrast(gap, "joint_signed_reference_gap", "jointly_feasible_paired_instance"),
            "joint_success_count": len(joint),
            "joint_left_signed_gap": statistics.fmean(signed_gap(a) for a, b in joint) if joint else None,
            "joint_right_signed_gap": statistics.fmean(signed_gap(b) for a, b in joint) if joint else None,
            "joint_quality_right_wins": sum(value < -1e-12 for value in gap),
            "joint_quality_right_losses": sum(value > 1e-12 for value in gap),
            "joint_quality_ties": sum(abs(value) <= 1e-12 for value in gap)}
    result["success_within_gap"] = {}
    for label, threshold in GAP_THRESHOLDS.items():
        values = [int(b.pass_at_k and signed_gap(b) <= threshold + 1e-12)
                  - int(a.pass_at_k and signed_gap(a) <= threshold + 1e-12) for a, b in pairs]
        result["success_within_gap"][label] = contrast(values, f"success_within_gap_{label}", "paired_instance")
    return result


def compare(indices, output):
    fresh, strict, _ = helpers()
    loaded, provenance, sources = {}, {}, set()
    for path in indices:
        declared = fresh.checked_json(path, "index_sha256")
        verified, artifact = validate_evaluation(Path(path).parent)
        require(declared == {**verified, "index_sha256": fresh.digest(verified)}, "Result index differs from raw revalidation")
        key = (declared["task"], declared["method"])
        require(key not in loaded, "Duplicate task/method index")
        loaded[key] = artifact
        launch_receipt = fresh.checked_json(Path(path).parent / "launch_receipt.json", "launch_sha256")
        sources.add(launch_receipt["source_manifest"]["sha256"])
        provenance["/".join(key)] = fresh.identity(path)
    present_tasks = {key[0] for key in loaded}
    require(present_tasks and present_tasks <= set(TASKS), "No supported task indices supplied")
    require(len(sources) == 1, "Comparison source snapshots differ")
    for task in present_tasks:
        require(all((task, arm) in loaded for arm in ARMS), "Supply all three predeclared arms for each included task")
    rows, contrasts, sft_status = [], {}, {}
    for task in sorted(present_tasks):
        methods = [method for method in METHODS if (task, method) in loaded]
        for method in methods:
            rows.append(summarize(list(loaded[task, method].records.values()), task, method))
        for left, right in itertools.combinations(ARMS, 2):
            a, b = loaded[task, left], loaded[task, right]
            contrasts[f"{task}/{left}_vs_{right}"] = {
                "strict_comparison": strict.build_receipt(a, b, left_label=left, right_label=right),
                **paired_extra(a, b, strict)}
        if (task, "sft") in loaded:
            for arm in ARMS:
                a, b = loaded[task, "sft"], loaded[task, arm]
                try:
                    strict._validate_pairing(a, b)
                except strict.DevelopmentComparisonError as error:
                    sft_status[f"{task}/{arm}"] = {"status": "descriptive_only", "reason": str(error)}
                else:
                    sft_status[f"{task}/{arm}"] = {"status": "exact_paired"}
                    contrasts[f"{task}/sft_vs_{arm}"] = {"strict_comparison": strict.build_receipt(a, b, left_label="sft", right_label=arm),
                                                        **paired_extra(a, b, strict)}
    report = {"protocol": PROTOCOL, "scope": "inspected_development_only_single_training_seed43",
              "source_manifest_sha256": next(iter(sources)), "indices": provenance,
              "summary": rows, "paired_comparisons": contrasts, "sft_pairing": sft_status,
              "definitions": {"feasible_at_1": "Feasibility of candidate zero from the sampled raw8 call; not greedy decoding.",
                              "candidate_feasibility": "All800 candidates per task/arm; failures stay in the denominator.",
                              "signed_gap": "(pred-ref)/ref for CVRP; (ref-pred)/ref for MIS; negative CVRP gaps preserved.",
                              "joint_quality": "Signed reference-gap difference, right minus left, on jointly feasible instances; negative favors right.",
                              "units": "Table rates and gaps are percent; contrasts and confidence intervals use fractions."},
              "limitations": ["Inspected DEV100; one training seed43; no confirmation, SOTA or universal-superiority claim.",
                              "Feasible@1 uses the first sampled candidate, not an independent generation or greedy pass.",
                              "Own-success gap denominators differ when feasibility changes; report joint-success counts alongside them.",
                              "CVRP references are verified heuristics; MIS references are certified optima.",
                              "Paired bootstrap intervals describe instance variation, not training-seed uncertainty; no multiplicity adjustment.",
                              "Terminal100 was predeclared. Equal steps and rollout caps do not establish equal measured GPU compute."]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = fresh.write_new(output / "comparison.json", report, "comparison_sha256")
    with (output / "table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["| Task | Arm | Feasible@1 | Feasible@8 | Feasible candidates | Signed gap on own successes |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        gap = "undefined" if row["mean_signed_reference_gap_percent"] is None else f"{row['mean_signed_reference_gap_percent']:.3f}% (n={row['own_success_gap_denominator']})"
        lines.append(f"| {row['task']} | {row['method']} | {row['feasible_at_1_count']}/100 | {row['feasible_at_8_count']}/100 | {row['feasible_candidate_count']}/800 | {gap} |")
    lines += ["", "Inspected DEV100, seed43, terminal100 only. Feasible@1 means the first of eight sampled candidates.",
              "CVRP gaps use heuristic references. Paired feasibility and joint-success signed quality are in comparison.json.", ""]
    (output / "table.md").write_text("\n".join(lines))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        sub = commands.add_parser(action)
        sub.add_argument("--task", choices=TASKS, required=True)
        sub.add_argument("--arm", choices=METHODS, required=True)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if action == "run":
            sub.add_argument("--source-manifest-sha256", required=True)
            sub.add_argument("--executed-slurm", type=Path, required=True)
    sub = commands.add_parser("index")
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("compare")
    sub.add_argument("--index", type=Path, action="append", required=True)
    sub.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.action == "plan":
        result = {"protocol": PROTOCOL, "submits_jobs": False, "command": evaluation_command(args.task, args.arm, args.checkpoint, args.output),
                  "expected_data_root": str(DEV), "expected_generation": GENERATION,
                  "allocation": {"gpus": 1, "gpu": "A100", "max_time": "02:00:00"}}
    elif args.action == "run":
        result = launch(args)
    elif args.action == "index":
        result = index(args.output)
    else:
        result = compare(args.index, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
