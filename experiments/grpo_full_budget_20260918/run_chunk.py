"""Execute one admitted full-budget chunk. Never submits, retries or cancels."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from common import (CONTRACT_NAME, HORIZONS, PROTOCOL, SEAL_NAME, identity,
                    load_declaration, output_dir, read, require, run_contract,
                    signed, verify, write_once)


def checkpoint_material(checkpoint, contract, contract_path, step, source_binding):
    from scripts.run_external_baseline_dev_chunk import rank_checkpoint_material
    from clasp.entrypoints.evaluate import _directory_content_identity
    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
    from clasp.baselines.budget_matched_dev import build_runtime
    checkpoint = Path(checkpoint)
    require(checkpoint == contract_path.parent / "checkpoints" / f"checkpoint-{step}", "Wrong checkpoint parent")
    state = read(checkpoint / "trainer_state.json")
    require(state["global_step"] == step and state["max_steps"] == contract["horizon"], "Wrong checkpoint step/horizon")
    fidelity = verify_checkpoint_fidelity(checkpoint)
    resume = read(checkpoint / "budget_resume_contract.json")
    expected = build_runtime(source_binding, contract, identity(contract_path)).resume_contract()
    import hashlib
    from common import canonical
    require(resume == {"schema_version": 1, "status": "passed", "global_step": step,
                       "world_size": 4, "contract": expected,
                       "contract_sha256": hashlib.sha256(canonical(expected)).hexdigest()}, "Wrong checkpoint run binding")
    names = ["adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json",
             "trainer_state.json", "optimizer.pt", "scheduler.pt", "budget_resume_contract.json"]
    names += [f"rng_state_{rank}.pth" for rank in range(4)]
    kl_names = ["exact_peft_kl_reference.json", "exact_peft_kl_reference_step1.json", "exact_peft_kl_reference_post_step.json"]
    if contract["method"] == "foarl":
        from clasp.models.exact_peft_kl_post_step import inspect_post_step_receipt, training_run_contract_identity
        post = inspect_post_step_receipt(checkpoint / kl_names[-1], claim_scope="development",
                    expected_global_step=step, expected_artifact_role="checkpoint",
                    rehash_sources=False, rehash_policy_artifact=False,
                    expected_policy_trainable_tensor_inventory_sha256=fidelity["trainable_closure"]["tensor_inventory_sha256"])
        require(post["training_run_contract"] == training_run_contract_identity(contract_path, claim_scope="development"), "FOARL KL contract differs")
        require(post["pre_step_reference"]["receipt_file_sha256"] == identity(checkpoint / kl_names[0])["sha256"]
                and post["step1_reference"]["receipt_file_sha256"] == identity(checkpoint / kl_names[1])["sha256"], "FOARL KL lineage differs")
        names += kl_names
    else:
        require(not any((checkpoint / name).exists() for name in kl_names), "Beta-zero checkpoint contains KL reference")
    if contract["method"].startswith("rloo_"):
        from clasp.rewards.external_baseline import ExternalBaselineTracker
        baseline = read(checkpoint / "external_baseline_state.json")
        require(baseline["global_step"] == step and baseline["advantage_estimator"] == "rloo"
                and baseline["num_generations"] == 9 and baseline["max_completion_length"] == 3000,
                "RLOO saved recipe changed")
        tracker = ExternalBaselineTracker.from_state_dict(baseline["tracker"])
        require(tracker.revision == step and tracker.seen_samples == step * 36, "RLOO tracker exposure differs")
        require(baseline.get("reward_contract") == expected["reward_contract"], "RLOO reward contract differs")
        names.append("external_baseline_state.json")
    rank_state = rank_checkpoint_material(checkpoint, step)
    # Seal the same recursive inference inventory that the canonical evaluator
    # records, including tokenizer/configuration files and custom model code.
    # The explicit names above remain required training/resume inputs.
    inference = _directory_content_identity(str(checkpoint), inference_artifacts_only=True)
    names += [item["path"] for item in inference["files"]]
    files = {name: identity(checkpoint / name) for name in sorted(set(names))}
    for item in inference["files"]:
        bound = files[item["path"]]
        require(bound["sha256"] == item["sha256"] and bound["size_bytes"] == item["size"],
                "Inference artifact changed while sealing: " + item["path"])
    files.update(rank_state["files"])
    return {"protocol": PROTOCOL, "status": "verified_budget_checkpoint", "task": contract["task"],
            "method": contract["method"], "step": step, "max_steps": contract["horizon"], "world_size": 4,
            "contract": identity(contract_path), "files": files, "rank_state": rank_state,
            "initialization": contract["initialization"], "source": contract["source"],
            "checkpoint_fidelity_receipt_sha256": fidelity["receipt_sha256"],
            "sampler_future_trajectory_equality_claimed": False}


def environment(d, contract, contract_path, job):
    from scripts.run_external_baseline_dev_chunk import configure_environment
    from clasp.baselines.budget_matched_dev import reward_environment
    source, output = Path(d["source"]["root"]), Path(contract["output_dir"])
    prior = job["prior_step"]
    resume = output / "checkpoints" / f"checkpoint-{prior}" if prior else None
    configure_environment(source, output, contract_path, job["step"], contract["diagnostic"], resume)
    os.environ.update(reward_environment(job["method"]))
    # Equal diagnostic checks for every method; preserve them throughout training.
    if job["method"] == "foarl":
        sft = Path(contract["initialization"]["checkpoint"])
        os.environ.update(CLASP_KL_REFERENCE_MODEL_PATH=str(sft),
                          CLASP_KL_REFERENCE_RECEIPT=str(sft / "checkpoint_fidelity.json"))
    return resume


def command(d, contract, contract_path):
    sft = contract["initialization"]["checkpoint"]
    values = ["srun", "--ntasks=1", "--ntasks-per-node=1", d["runtime_python"], "-B", "-m",
              "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m",
              "clasp.entrypoints.train_budget_matched_grpo", "--tasks", contract["task"],
              "--sft_checkpoint_path", sft, "--tokenizer_path", sft,
              "--data_root", contract["data_root"], "--output_dir", contract["output_dir"],
              "--budget_method", contract["method"], "--budget_contract", str(contract_path),
              "--budget_contract_sha256", identity(contract_path)["sha256"],
              "--eval_ratio", "0", "--eval-from-train-split", "--max_samples_per_task", "0",
              "--max_eval_samples_per_task", "0", "--min_train_samples_per_task", "1900",
              "--min_eval_samples_per_task", "0", "--no-prefer_subset_files", "--report_to", "none"]
    if contract["diagnostic"]:
        values.append("--budget_diagnostic")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed-slurm", required=True)
    args = parser.parse_args()
    d, binding = load_declaration(os.environ["FB_DECLARATION"], os.environ["FB_DECLARATION_SHA"])
    name = os.environ["FB_JOB"]
    job = d["jobs"][name]
    require(job["kind"] in ("train", "diagnostic"), "Wrong job kind")
    source = Path(d["source"]["root"])
    sys.path.insert(0, str(source))
    from clasp.baselines.external_baseline_dev import verify_source_manifest
    verified = verify_source_manifest(source / "SOURCE_MANIFEST.sha256", d["source"]["manifest_sha256"])
    require(Path(args.executed_slurm).read_bytes() == (source / "experiments/grpo_full_budget_20260918/train_a100.slurm").read_bytes(), "Executed launcher changed")
    require(Path(sys.executable).resolve() == Path(d["runtime_python"]).resolve(), "Wrong interpreter")
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "Training requires Slurm")
    from campaign import verify_allocation
    verify_allocation(d, name, job, os.environ["SLURM_JOB_ID"])
    import torch
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    require(len(gpu_names) == 4 and all("A100" in name for name in gpu_names), "Requires four A100 GPUs")
    initialization_receipt = Path(d["root"]) / "initializations" / (job["task"] + ".json")
    initial_receipt = read(initialization_receipt, sealed=True)
    require(initial_receipt["declaration"] == binding and initial_receipt["task"] == job["task"], "Initializer binding differs")
    initialization = initial_receipt["initialization"]
    if "initialization" in d["tasks"][job["task"]]:
        require(initialization == d["tasks"][job["task"]]["initialization"], "Declared initializer changed")
    else:
        require(initialization["checkpoint"] == d["tasks"][job["task"]]["initialization_pending"]["checkpoint"], "Future initializer path changed")
    for entry in initialization["files"].values():
        verify(entry)
    for entry in d["tasks"][job["task"]]["training_data_files"].values():
        verify(entry)
    contract = run_contract(d, binding, job, initialization)
    output = output_dir(d, job)
    contract_path = output / CONTRACT_NAME
    if job["prior_step"] == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_once(contract_path, contract)
    else:
        require(read(contract_path) == contract, "Original contract changed")
    with (output / ".chunk.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (output / f"chunk-{job['step']}-admission.json").exists(), "No automatic retry")
        predecessor = None
        if job["prior_step"]:
            prior = job["prior_step"]
            checkpoint = output / "checkpoints" / f"checkpoint-{prior}"
            require(read(output / f"chunk-{prior}-exit.json")["returncode"] == 0, "Predecessor did not succeed")
            completed = read(output / f"chunk-{prior}-complete.json", sealed=True)
            predecessor = read(checkpoint / SEAL_NAME, sealed=True)
            require(completed["checkpoint"] == predecessor == signed(checkpoint_material(checkpoint, contract, contract_path, prior, verified)), "Predecessor seal differs")
        resume = environment(d, contract, contract_path, job)
        cmd = command(d, contract, contract_path)
        admission = {"protocol": PROTOCOL, "job": name, "job_id": os.environ["SLURM_JOB_ID"],
                     "contract": identity(contract_path), "command": cmd, "gpu_names": gpu_names,
                     "started_at_unix": time.time(), "source_verified_file_count": verified["verified_file_count"]}
        write_once(output / f"chunk-{job['step']}-admission.json", signed(admission))
        result = subprocess.run(cmd)
        write_once(output / f"chunk-{job['step']}-exit.json", {"returncode": result.returncode, "finished_at_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]})
        require(result.returncode == 0, f"Training failed with exit {result.returncode}")
        live = None
        if resume:
            from scripts.run_external_baseline_dev_chunk import verify_live_resume_receipts
            live = verify_live_resume_receipts(output, resume, job["prior_step"], predecessor)
            for rank in range(4):
                value = read(output / "checkpoints" / f"budget_resume_{job['prior_step']}_rank{rank}.json")
                expected = read(resume / "budget_resume_contract.json")
                require(value == {**expected, "rank": rank, "checkpoint": str(resume)}, "Live budget restoration differs")
        checkpoint = output / "checkpoints" / f"checkpoint-{job['step']}"
        material = signed(checkpoint_material(checkpoint, contract, contract_path, job["step"], verified))
        write_once(checkpoint / SEAL_NAME, material)
        write_once(output / f"chunk-{job['step']}-complete.json", signed({
            "protocol": PROTOCOL, "status": "verified_budget_training", "task": job["task"],
            "method": job["method"], "step": job["step"], "max_steps": HORIZONS[job["task"]],
            "world_size": 4, "contract": identity(contract_path), "checkpoint": material,
            "live_rank_restoration": live, "admission": identity(output / f"chunk-{job['step']}-admission.json"),
            "job_id": os.environ["SLURM_JOB_ID"]}))
        print(f"FULL_BUDGET_CHUNK_COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
