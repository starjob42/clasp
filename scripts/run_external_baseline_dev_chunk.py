"""Execute one immutable, beta-zero external-baseline development chunk.

No Slurm submission, cancellation, automatic retry, historical run mutation or
deletion occurs here. Native training runs only after source, initialization,
data and allocation admission. A resumed chunk requires its own successful
previous execution and sealed checkpoint inventory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUNTIME = Path("/lustre/fswork/projects/rech/vas/uag72kz/.conda/envs/new_unsloth_env")
DATA = Path("/lustre/fswork/projects/rech/vas/uag72kz/primal_dual/data/data_rl")
SCRATCH = Path("/lustre/fsn1/projects/rech/vas/uag72kz")
SFT = SCRATCH / "clasp_jiang_f638_sft500k_sourcev17_production_output_iclr_20260828_v14"
SFT_HASHES = {
    "cvrp": "f8f634c3a59f53bf411bb0dfc85af98dbc16d936a53263cdfcd6e74ce88b25b8",
    "mis": "cb5e979325ba7691f75998326ce79e47038d6462596dbc815f2b54d2b550da15",
}
ARMS = {"rloo_b9": ("rloo", 9), "external_b9": ("external", 9), "external_b1": ("external", 1)}
CONTRACT_NAME = "local_development_run_contract.json"
BASELINE_NAME = "external_baseline_state.json"
CHECKPOINT_SEAL = "external_baseline_checkpoint_inventory.json"
PROTOCOL = "external-baseline-graded-development-chunk-20260910-v1"
WORLD_SIZE = 4
HORIZON = 100


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def signed(value):
    return {**value, "receipt_sha256": hashlib.sha256(canonical(value)).hexdigest()}


def load_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked JSON input: {path}")
    return json.loads(path.read_text())


def load_signed(path):
    value = load_json(path)
    unsigned = {k: v for k, v in value.items() if k != "receipt_sha256"}
    require(value.get("receipt_sha256") == hashlib.sha256(canonical(unsigned)).hexdigest(),
            f"Invalid receipt self hash: {path}")
    return value


def identity(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), f"Missing or linked input: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"Input changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def write_once(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o440)


def validate_selection(task, arm, stop, diagnostic, resume):
    require(task in SFT_HASHES and arm in ARMS, "Unsupported task or external-baseline arm")
    require(type(diagnostic) is bool, "Diagnostic mode must be boolean")
    allowed = (5, 10) if diagnostic else (50, 100)
    require(type(stop) is int and stop in allowed, "Undeclared external-baseline stop")
    if resume is not None:
        require(stop == allowed[1], "Only the second declared boundary can resume")
    return allowed[0] if resume is not None else 0


def rank_checkpoint_material(checkpoint, step):
    """Check hashes against the trainer's exact saved-to-live rank receipts."""
    checkpoint = Path(checkpoint)
    files, receipts = {}, {}
    for rank in range(WORLD_SIZE):
        name = f"external_rank_state_{rank}.json"
        receipt = load_json(checkpoint / name)
        for field, expected in (("schema_version", 1), ("status", "passed"),
                                ("rank", rank), ("world_size", WORLD_SIZE),
                                ("global_step", step)):
            require(type(receipt.get(field)) is type(expected) and receipt[field] == expected,
                    f"Rank {rank} checkpoint receipt differs: {field}")
        require(set(receipt.get("states", {})) == {"optimizer", "scheduler"},
                f"Rank {rank} checkpoint state owners differ")
        files[name] = identity(checkpoint / name)
        for owner in ("optimizer", "scheduler"):
            binding = receipt["states"][owner]
            filename = f"{owner}_rank{rank}.pt"
            require(set(binding) == {"file", "semantic_sha256", "file_sha256"}
                    and binding["file"] == filename,
                    f"Rank {rank} {owner} binding differs")
            for digest in (binding["semantic_sha256"], binding["file_sha256"]):
                require(isinstance(digest, str) and len(digest) == 64
                        and all(c in "0123456789abcdef" for c in digest),
                        f"Rank {rank} {owner} digest malformed")
            files[filename] = identity(checkpoint / filename)
            require(files[filename]["sha256"] == binding["file_sha256"],
                    f"Rank {rank} {owner} file changed after live-state verification")
        receipts[str(rank)] = receipt
    return {"receipts": receipts, "files": files}


def verify_live_resume_receipts(output, checkpoint, prior_step, sealed):
    """Require every rank's exact restored-value check after a resumed run."""
    records = {}
    for rank in range(WORLD_SIZE):
        path = Path(output) / "checkpoints" / f"external_resume_{prior_step}_rank{rank}.json"
        receipt = load_json(path)
        original = sealed["rank_state"]["receipts"][str(rank)]
        require(receipt == {**original, "checkpoint": str(checkpoint)},
                f"Rank {rank} live restoration differs from sealed predecessor")
        records[str(rank)] = {"receipt": receipt, "file": identity(path)}
    return records


def checkpoint_material(checkpoint, contract, contract_path, step):
    """Validate independently saved inputs; live restoration is trainer-owned."""
    checkpoint = Path(checkpoint).absolute()
    require(checkpoint == Path(contract_path).parent / "checkpoints" / f"checkpoint-{step}",
            "Checkpoint does not belong to this arm and boundary")
    require(load_json(contract_path) == contract, "Original run contract changed")
    state = load_json(checkpoint / "trainer_state.json")
    require(type(state.get("global_step")) is int and state["global_step"] == step
            and state.get("max_steps") == HORIZON, "Trainer step or fixed horizon changed")
    baseline = load_json(checkpoint / BASELINE_NAME)
    estimator, generations = ARMS[contract["method"]]
    require(baseline.get("schema_version") == 1 and baseline.get("global_step") == step
            and baseline.get("advantage_estimator") == estimator
            and baseline.get("num_generations") == generations
            and baseline.get("max_completion_length") == 3000,
            "Checkpoint baseline recipe or step differs")
    from clasp.rewards.external_baseline import ExternalBaselineTracker, DEFAULT_SIZE_BOUNDARIES
    tracker = ExternalBaselineTracker.from_state_dict(baseline["tracker"])
    restored = tracker.state_dict()
    require(restored["config"] == {"initial_value": 0.0, "scale": 1.0,
                                  "size_boundaries": list(DEFAULT_SIZE_BOUNDARIES)},
            "Checkpoint baseline initialization or frozen scale changed")
    require(tracker.revision == step and tracker.seen_samples == step * 36
            and {cell["task"] for cell in restored["cells"]} == {contract["task"]},
            "Checkpoint baseline revision, observation count or task differs")
    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
    fidelity = verify_checkpoint_fidelity(checkpoint)
    names = ["adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json",
             "trainer_state.json", "optimizer.pt", "scheduler.pt", BASELINE_NAME]
    names.extend(f"rng_state_{rank}.pth" for rank in range(WORLD_SIZE))
    rank_receipt = rank_checkpoint_material(checkpoint, step)
    files = {name: identity(checkpoint / name) for name in sorted(set(names))}
    files.update(rank_receipt["files"])
    require(all(not (checkpoint / name).exists() for name in
                ("exact_peft_kl_reference.json", "exact_peft_kl_reference_step1.json",
                 "exact_peft_kl_reference_post_step.json")), "Beta-zero arm contains KL sidecars")
    return {"protocol": PROTOCOL, "status": "complete_external_baseline_checkpoint_inputs",
            "method": contract["method"], "step": step, "max_steps": HORIZON,
            "world_size": WORLD_SIZE, "contract": identity(contract_path), "files": files,
            "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
            "rank_state": rank_receipt,
            "sampler_future_trajectory_equality_claimed": False}


def validate_resume(checkpoint, contract, contract_path, step):
    output = Path(contract_path).parent
    exit_receipt = load_json(output / f"chunk-{step}-exit.json")
    require(exit_receipt.get("returncode") == 0, "Cannot resume a failed predecessor")
    completion = load_signed(output / f"chunk-{step}-complete.json")
    sealed = load_signed(Path(checkpoint) / CHECKPOINT_SEAL)
    current = checkpoint_material(checkpoint, contract, contract_path, step)
    require(sealed == signed(current) and completion.get("checkpoint") == sealed,
            "Checkpoint files changed or predecessor has no matching completion")
    return sealed


def configure_environment(source, output, contract_path, stop, diagnostic, resume):
    prefixes = ("CLASP_", "PD_", "GENERATION_", "GRPO_", "FOARL_", "WANDB_", "NCCL_",
                "PYTORCH_", "HF_", "TRANSFORMERS_", "TOKENIZERS_", "BITSANDBYTES_",
                "UNSLOTH_", "ACCELERATE_")
    for name in list(os.environ):
        if name.startswith(prefixes) or name in ("RESUME_FROM_CHECKPOINT", "CUDA_LAUNCH_BLOCKING"):
            del os.environ[name]
    cache = output.with_name(output.name + ".cache")
    for name in ("huggingface/datasets", "tmp", "torch_extensions", "wandb"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CLASP_LORA_TARGET_MODULES": "jiang", "CLASP_LORA_BIAS": "all",
        "CLASP_DEVELOPMENT_FIDELITY": "1", "CLASP_TRAINING_RUN_CONTRACT_PATH": str(contract_path),
        "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0", "CLASP_MIS_RELABEL_ORDER": "original",
        "CLASP_POLICY_CONSISTENCY_CHECKS": "1", "CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL": "1",
        "CLASP_GRADIENT_CONSISTENCY_CHECKS": "1",
        "GRPO_DATA_SEED": "43", "GRPO_STRICT_OPTIMIZER_RESUME": "1",
        "GRPO_STOP_AT_GLOBAL_STEP": str(stop), "GRPO_VERIFIED_IMPROVEMENT": "0",
        "GRPO_VERIFIED_IMPROVEMENT_WEIGHT": "0.0", "GRPO_GRADED_REWARD": "1",
        "CLASP_OBJECTIVE_QUALITY_MODE": "exp", "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
        "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8",
        "NCCL_DEBUG": "WARN", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONHASHSEED": "43", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": str(source / "src"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(cache / "huggingface"), "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "TMPDIR": str(cache / "tmp"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "WANDB_MODE": "offline", "WANDB_DIR": str(cache / "wandb"),
    })
    if resume is not None:
        os.environ["RESUME_FROM_CHECKPOINT"] = str(resume)


def build_command(source, source_sha, task, arm, sft, output, diagnostic):
    estimator, generations = ARMS[arm]
    command = ["srun", "--ntasks=1", "--ntasks-per-node=1", str(RUNTIME / "bin/python"),
               "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "-m", "clasp.entrypoints.train_external_baseline_grpo", "--tasks", task,
               "--sft_checkpoint_path", str(sft), "--tokenizer_path", str(sft),
               "--data_root", str(DATA), "--output_dir", str(output),
               "--external_dev_source_manifest", str(source / "SOURCE_MANIFEST.sha256"),
               "--external_dev_source_manifest_sha256", source_sha,
               "--advantage_estimator", estimator, "--num_generations", str(generations),
               "--external_baseline_initial", "0.0", "--external_baseline_scale", "1.0",
               "--max_steps", str(HORIZON), "--save_steps", "5" if diagnostic else "50",
               "--save_total_limit", "2", "--generation_temperature", "1.0",
               "--generation_top_p", "1.0", "--generation_top_k", "0",
               "--eval_ratio", "0", "--eval-from-train-split", "--max_samples_per_task", "0",
               "--max_eval_samples_per_task", "0", "--min_train_samples_per_task", "1900",
               "--min_eval_samples_per_task", "0", "--no-prefer_subset_files", "--report_to", "none"]
    if diagnostic:
        command.append("--external_baseline_diagnostic")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed-slurm", required=True)
    args = parser.parse_args()
    task, arm = os.environ["EB_TASK"], os.environ["EB_ARM"]
    diagnostic_raw = os.environ.get("EB_DIAGNOSTIC", "0")
    require(diagnostic_raw in ("0", "1"), "EB_DIAGNOSTIC must be 0 or 1")
    diagnostic, stop = diagnostic_raw == "1", int(os.environ["EB_STOP"])
    resume = Path(os.environ["EB_RESUME"]).resolve(strict=True) if os.environ.get("EB_RESUME") else None
    prior_step = validate_selection(task, arm, stop, diagnostic, resume)
    source = Path(os.environ["EB_SOURCE"]).resolve(strict=True)
    source_sha = os.environ["EB_SOURCE_SHA"]
    output = Path(os.environ["EB_OUTPUT"]).absolute()
    require(output.resolve() == output and output.is_relative_to(SCRATCH),
            "External-baseline output must be an unlinked SCRATCH path")
    require(source.is_relative_to(SCRATCH), "Frozen native source must be staged under SCRATCH")
    require(Path(sys.executable).resolve() == (RUNTIME / "bin/python").resolve(), "Wrong native interpreter")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "Requires an existing Slurm allocation")
    from clasp.baselines.external_baseline_dev import verify_source_manifest
    binding = verify_source_manifest(source / "SOURCE_MANIFEST.sha256", source_sha, source_root=source)
    launcher = source / "scripts/train_external_baseline_dev_a100.slurm"
    require(Path(args.executed_slurm).read_bytes() == launcher.read_bytes(),
            "Executed Slurm file differs from frozen source")
    manifest_names = {line.split("  ", 1)[1] for line in (source / "SOURCE_MANIFEST.sha256").read_text().splitlines()}
    require({"scripts/train_external_baseline_dev_a100.slurm", "scripts/run_external_baseline_dev_chunk.py"}
            <= manifest_names, "Source manifest omits the executed launcher")
    import torch
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    require(len(gpu_names) == WORLD_SIZE and all("A100" in name for name in gpu_names),
            "Requires exactly four visible A100 GPUs")
    sft = SFT / "runs" / task / "seed42/checkpoints/checkpoint-31250"
    sft_files = {name: identity(sft / name) for name in
                 ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}
    require(sft_files["adapter_model.safetensors"]["sha256"] == SFT_HASHES[task], "SFT adapter differs")
    data_files = {str(path.relative_to(DATA)): identity(path) for path in sorted((DATA / task / "train").glob("*.json"))}
    require(bool(data_files), "No declared training files")
    estimator, generations = ARMS[arm]
    contract = {"protocol": PROTOCOL, "schema_version": 1, "stage": "grpo",
                "status": "development_only_not_publication_eligible", "method": arm, "task": task,
                "source": {"root": str(source), "manifest_sha256": source_sha},
                "initialization": {"checkpoint": str(sft), "files": sft_files},
                "fixed_horizon": {"max_steps": HORIZON, "world_size": WORLD_SIZE},
                "beta": 0.0, "teacher_weight": 0.0, "seed": 43,
                "data_root": str(DATA), "data_files": data_files, "diagnostic": diagnostic,
                "advantage_estimator": estimator, "num_generations": generations,
                "baseline": {"initial_value": 0.0, "scale": 1.0, "update": "past_batches_only"},
                "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                             "steps_per_generation": 3, "save_steps": 5 if diagnostic else 50},
                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                             "repetition_penalty": 1.0, "max_completion_length": 3000},
                "loss_type": "dr_grpo", "reward": "existing_graded_verifier_scalar_v1"}
    contract_path = output / CONTRACT_NAME
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        write_once(contract_path, contract)
    else:
        require(output.is_dir() and load_json(contract_path) == contract, "Original run contract differs")
    with (output / ".chunk.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resume_receipt = validate_resume(resume, contract, contract_path, prior_step) if resume else None
        require(not (output / f"chunk-{stop}-admission.json").exists(), "This chunk has already been attempted")
        configure_environment(source, output, contract_path, stop, diagnostic, resume)
        command = build_command(source, source_sha, task, arm, sft, output, diagnostic)
        admission = signed({"protocol": PROTOCOL, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "job_id": os.environ["SLURM_JOB_ID"], "stop": stop, "command": command,
                            "contract": identity(contract_path), "resume": resume_receipt,
                            "source_verified_file_count": binding["verified_file_count"],
                            "gpu_names": gpu_names, "performance_claim": False})
        write_once(output / f"chunk-{stop}-admission.json", admission)
        result = subprocess.run(command)
        write_once(output / f"chunk-{stop}-exit.json", {"returncode": result.returncode,
                   "job_id": os.environ["SLURM_JOB_ID"], "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        if result.returncode:
            raise SystemExit(result.returncode)
        live_resume = (verify_live_resume_receipts(output, resume, prior_step, resume_receipt)
                       if resume else None)
        checkpoint = output / "checkpoints" / f"checkpoint-{stop}"
        material = signed(checkpoint_material(checkpoint, contract, contract_path, stop))
        write_once(checkpoint / CHECKPOINT_SEAL, material)
        write_once(output / f"chunk-{stop}-complete.json", signed({"protocol": PROTOCOL,
                   "status": "completed_declared_external_baseline_chunk", "checkpoint": material,
                   "live_rank_restoration": live_resume,
                   "admission": identity(output / f"chunk-{stop}-admission.json"),
                   "job_id": os.environ["SLURM_JOB_ID"], "stop": stop}))
        print(f"EXTERNAL_BASELINE_CHUNK_COMPLETE task={task} arm={arm} step={stop}", flush=True)


if __name__ == "__main__":
    main()
