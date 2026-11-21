"""Execute one beta-zero classic-GRPO chunk; never submit or retry a job."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from run_foarl_teacher_dev_chunk import DATA, RUNTIME, SFT, SFT_HASHES, identity, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed-slurm", required=True)
    args = parser.parse_args()
    source = Path(os.environ["CG_SOURCE"]).resolve(strict=True)
    expected_sha = os.environ["CG_SOURCE_SHA"]
    task, method = os.environ["CG_TASK"], "classic_grpo"
    if task not in SFT_HASHES:
        raise ValueError("Unsupported specialist task")
    weight = 0.
    diagnostic_raw = os.environ.get("CG_DIAGNOSTIC", "0")
    if diagnostic_raw not in ("0", "1"):
        raise ValueError("CG_DIAGNOSTIC must be 0 or 1")
    diagnostic = diagnostic_raw == "1"
    stop = int(os.environ["CG_STOP"])
    if stop not in ((5, 10) if diagnostic else (100, 200, 300, 400)):
        raise ValueError("Undeclared stop boundary")
    output = Path(os.environ["CG_OUTPUT"]).absolute()
    resume = Path(os.environ["CG_RESUME"]).resolve(strict=True) if os.environ.get("CG_RESUME") else None
    compare_with = Path(os.environ["CG_COMPARE_WITH"]).resolve(strict=True) if os.environ.get("CG_COMPARE_WITH") else None
    if compare_with is not None and not (diagnostic and resume is not None and stop == 10):
        raise ValueError("Continuation comparison is restricted to the resumed diagnostic")
    sft = SFT / "runs" / task / "seed42/checkpoints/checkpoint-31250"
    if Path(sys.executable).resolve() != (RUNTIME / "bin/python").resolve():
        raise ValueError("Wrong native interpreter")
    from clasp.baselines.foarl_f638_dev_20260907 import verify_dev_source_manifest
    binding = verify_dev_source_manifest(source / "SOURCE_MANIFEST.sha256", expected_sha)
    if Path(args.executed_slurm).read_bytes() != (source / "scripts/train_classic_grpo_400_dev_a100.slurm").read_bytes():
        raise ValueError("Executed Slurm file differs from frozen source")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(gpu_names) != 4 or not all("A100" in name for name in gpu_names):
        raise ValueError("Requires exactly four A100 GPUs")
    files = {name: identity(sft / name) for name in
             ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}
    if files["adapter_model.safetensors"]["sha256"] != SFT_HASHES[task]:
        raise ValueError("SFT initialization differs from matched baseline")
    data_files = {str(path.relative_to(DATA)): identity(path)
                  for path in sorted((DATA / task / "train").glob("*.json"))}
    if not data_files:
        raise ValueError("No training data")
    contract = {
        "schema_version": 1, "stage": "grpo", "status": "development_only_not_publication_eligible",
        "method": method, "source": {"root": str(source), "manifest_sha256": expected_sha},
        "initialization": {"checkpoint": str(sft), "files": files},
        "fixed_horizon": {"max_steps": 400, "world_size": 4},
        "teacher_weight": weight, "beta": 0.0, "seed": 43, "data_root": str(DATA), "data_files": data_files,
        "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                     "steps_per_generation": 3, "save_steps": 5 if diagnostic else 100},
        "diagnostic": diagnostic,
    }
    contract_path = output / "local_development_run_contract.json"
    resume_receipt = None
    if resume is None:
        if stop not in ((5, 10) if diagnostic else (100,)):
            raise ValueError("Fresh production arm must start at step100")
        output.mkdir(parents=True, exist_ok=False)
        write_once(contract_path, contract)
    elif not output.is_dir():
        raise ValueError("Resume requires the existing training output")
    # Hold across validation, training, and terminal receipt. Parallel chunks
    # cannot mutate the same optimizer/checkpoint directory.
    with (output / ".chunk.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume is not None:
            expected_step = stop - (5 if diagnostic else 100)
            if resume != output / "checkpoints" / f"checkpoint-{expected_step}":
                raise ValueError("Resume must use the immediately preceding chunk in this arm")
            from clasp.models.classic_grpo_resume import validate_classic_grpo_resume
            resume_receipt = validate_classic_grpo_resume(
                resume, expected_step=expected_step, expected_contract=contract,
                run_contract_path=contract_path)
        # Clear inherited research, caching and resume controls before restating
        # this study's complete environment. CG_* only selects the above contract.
        prefixes = ("CLASP_", "PD_", "GENERATION_", "GRPO_", "FOARL_", "WANDB_", "NCCL_",
                    "PYTORCH_", "HF_", "TRANSFORMERS_", "TOKENIZERS_", "BITSANDBYTES_",
                    "UNSLOTH_", "ACCELERATE_")
        for name in list(os.environ):
            if name.startswith(prefixes) or name == "RESUME_FROM_CHECKPOINT":
                del os.environ[name]
        cache = output.with_name(output.name + ".cache")
        for name in ("huggingface/datasets", "tmp", "torch_extensions", "wandb"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "CLASP_LORA_TARGET_MODULES": "jiang", "CLASP_LORA_BIAS": "all",
            "CLASP_DEVELOPMENT_FIDELITY": "1", "CLASP_TRAINING_RUN_CONTRACT_PATH": str(contract_path),
            "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0", "CLASP_MIS_RELABEL_ORDER": "original",
            "CLASP_POLICY_CONSISTENCY_CHECKS": "1",
            "CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL": "1" if diagnostic else "10",
            "CLASP_GRADIENT_CONSISTENCY_CHECKS": "1" if diagnostic else "0",
            "GRPO_DATA_SEED": "43", "GRPO_STRICT_OPTIMIZER_RESUME": "1",
            "GRPO_STOP_AT_GLOBAL_STEP": str(stop), "GRPO_VERIFIED_IMPROVEMENT": "0",
            "GRPO_GRADED_REWARD": "0", "CLASP_OBJECTIVE_QUALITY_MODE": "exp",
            "CLASP_OBJECTIVE_GAP_SLOPE": "3.0", "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8", "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONHASHSEED": "43",
            "HF_HOME": str(cache / "huggingface"), "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
            "TMPDIR": str(cache / "tmp"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "WANDB_MODE": "offline", "WANDB_DIR": str(cache / "wandb"),
        })
        if resume:
            os.environ["RESUME_FROM_CHECKPOINT"] = str(resume)
        command = ["srun", "--ntasks=1", "--ntasks-per-node=1", str(RUNTIME / "bin/python"),
                   "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   "-m", "clasp.entrypoints.train_classic_grpo_400_dev",
                   "--tasks", task, "--sft_checkpoint_path", str(sft), "--tokenizer_path", str(sft),
                   "--data_root", str(DATA), "--output_dir", str(output),
                   "--classic_dev_source_manifest", str(source / "SOURCE_MANIFEST.sha256"),
                   "--classic_dev_source_manifest_sha256", expected_sha,
                   "--eval_ratio", "0", "--eval-from-train-split",
                   "--max_samples_per_task", "0", "--max_eval_samples_per_task", "0",
                   "--min_train_samples_per_task", "1900", "--min_eval_samples_per_task", "0",
                   "--no-prefer_subset_files", "--report_to", "none"]
        if diagnostic:
            command += ["--classic_grpo_diagnostic", "--save_steps", "5"]
        receipt = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "job_id": os.environ["SLURM_JOB_ID"], "stop": stop, "command": command,
                   "contract": identity(contract_path), "resume": resume_receipt,
                   "source_verified_file_count": binding["verified_file_count"],
                   "gpu_names": gpu_names, "performance_claim": False}
        write_once(output / f"chunk-{stop}-admission.json", receipt)
        if diagnostic and resume is None:
            subprocess.run([sys.executable, "-B", str(source / "scripts/probe_foarl_teacher_tokenizer.py"),
                            "--tokenizer", str(sft), "--task", task,
                            "--output", str(output / "tokenizer_alignment.json")], check=True)
        result = subprocess.run(command)
        write_once(output / f"chunk-{stop}-exit.json", {"returncode": result.returncode,
                   "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        if result.returncode:
            raise SystemExit(result.returncode)
        checkpoint = output / "checkpoints" / f"checkpoint-{stop}"
        from clasp.models.classic_grpo_resume import (
            seal_classic_grpo_checkpoint,
            validate_classic_grpo_resume,
        )
        sealed = seal_classic_grpo_checkpoint(
            checkpoint, expected_step=stop, expected_contract=contract, run_contract_path=contract_path)
        completed = validate_classic_grpo_resume(
            checkpoint, expected_step=stop, expected_contract=contract, run_contract_path=contract_path)
        if completed != sealed:
            raise ValueError("Classic checkpoint changed after sealing")
        write_once(output / f"chunk-{stop}-complete.json", completed)
        if compare_with is not None:
            subprocess.run([sys.executable, "-B", str(source / "scripts/compare_classic_grpo_resume.py"),
                            "--uninterrupted", str(compare_with), "--resumed", str(checkpoint),
                            "--output", str(output / "resume_equivalence.json")], check=True)
        print(f"CLASSIC_GRPO_CHUNK_COMPLETE task={task} method={method} step={stop}", flush=True)


if __name__ == "__main__":
    main()
