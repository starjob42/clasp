"""Execute one strictly deterministic CO development chunk; never submit or retry a job."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from run_foarl_teacher_dev_chunk import DATA, RUNTIME, SFT, SFT_HASHES, identity, write_once

METHODS = ("foarl_control", "foarl_teacher", "classic_grpo")
PROFILE_ID = "co-strict-deterministic-development-20260909-v1"
DECLARATION_NAME = "deterministic_runtime_declaration.json"
EVIDENCE_NAME = "co_deterministic_runtime_evidence.json"
SNAPSHOT_NAME = "co_deterministic_hyperparams.json"
DECLARATION_PROTOCOL = "co-deterministic-runtime-declaration-v1"
EVIDENCE_PROTOCOL = "co-applied-deterministic-runtime-evidence-v1"
RUNTIME_ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "43", "CUDA_LAUNCH_BLOCKING": None}
STRICT_FLAGS = {"deterministic_algorithms": True, "deterministic_warn_only": False,
                "cudnn_deterministic": True, "cudnn_benchmark": False,
                "flash_sdp_enabled": True, "efficient_sdp_enabled": True, "math_sdp_enabled": True}
REQUIREMENTS = {"torch_version": "2.9.1", "applied_before_cuda_initialization": True,
                "manual_strict_algorithms": True, "hf_full_determinism": False,
                "seed_or_rng_state_reset_by_this_profile": False, "exact_continuation_proved": False}
IMPLEMENTATIONS = ("scripts/run_co_deterministic_dev_chunk.py", "scripts/train_co_deterministic_dev_a100.slurm",
                   "src/clasp/baselines/co_deterministic_dev.py", "src/clasp/entrypoints/train_co_deterministic_dev.py",
                   "src/clasp/models/classic_grpo_save_lineage.py", "src/clasp/models/co_determinism_receipt.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular_identity(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, "Runtime proof must be a regular unlinked file")
    return identity(path)


def read_json(path):
    regular_identity(path)
    def unique(pairs):
        value = {}
        for name, item in pairs:
            require(name not in value, "Duplicate runtime JSON key")
            value[name] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite runtime JSON")))


def signed(value):
    return {**value, "receipt_sha256": digest(value)}


def read_signed(path):
    value = read_json(path)
    require(value.get("receipt_sha256") == digest({k: v for k, v in value.items() if k != "receipt_sha256"}),
            "Runtime proof self hash differs")
    require(not Path(path).stat().st_mode & 0o222, "Runtime proof must be read-only")
    return value


def validate_launcher_environment():
    require({k: os.environ.get(k) for k in RUNTIME_ENVIRONMENT} == RUNTIME_ENVIRONMENT,
            "Deterministic environment must be fixed before launcher Python startup")
    version = importlib.metadata.version("torch")
    require(version.split("+")[0] == "2.9.1", "Native deterministic profile requires torch 2.9.1")
    return version


def _declaration_material(output, contract):
    output = Path(output).absolute()
    source = Path(contract["source"]["root"])
    require(source == Path(__file__).resolve().parents[1], "Runtime helper must belong to the declared source")
    require(contract["method"] in METHODS, "Unsupported deterministic method")
    require(regular_identity(source / "SOURCE_MANIFEST.sha256")["sha256"] == contract["source"]["manifest_sha256"],
            "Runtime source manifest changed")
    path = output / "local_development_run_contract.json"
    require(canonical(read_json(path)) == canonical(contract) and not path.stat().st_mode & 0o222,
            "Runtime original contract differs or is writable")
    return {"protocol": DECLARATION_PROTOCOL, "profile_id": PROFILE_ID, "method": contract["method"],
            "source": contract["source"], "run_contract": regular_identity(path),
            "environment": dict(RUNTIME_ENVIRONMENT), "requirements": {**REQUIREMENTS, "training_flags": dict(STRICT_FLAGS)},
            "implementation_files": {name: regular_identity(source / name) for name in IMPLEMENTATIONS},
            "scope": "prospective_runtime_requirement_not_applied_state_or_performance_evidence"}


def write_runtime_declaration(output, contract):
    write_once(Path(output) / DECLARATION_NAME, signed(_declaration_material(output, contract)))


def read_only_runtime_declaration(output, contract):
    value = read_signed(Path(output) / DECLARATION_NAME)
    require(value == signed(_declaration_material(output, contract)), "Deterministic runtime declaration changed")
    return value


def validate_checkpoint(method, checkpoint, step, contract, contract_path):
    if method == "classic_grpo":
        from clasp.models.classic_grpo_resume import validate_classic_grpo_resume as validate
    elif method in ("foarl_control", "foarl_teacher"):
        from clasp.models.foarl_teacher_resume import validate_foarl_teacher_resume as validate
    else:
        raise ValueError("Unsupported checkpoint method")
    return validate(checkpoint, expected_step=step, expected_contract=contract, run_contract_path=contract_path)


def _task(contract):
    initial = Path(contract["initialization"]["checkpoint"])
    task = initial.parents[2].name
    require(task in SFT_HASHES and initial == SFT / "runs" / task / "seed42/checkpoints/checkpoint-31250",
            "Runtime evidence has another original specialist SFT")
    return task


def _validate_applied_state(state):
    require(state["profile_id"] == PROFILE_ID and state["torch_version"].split("+")[0] == "2.9.1"
            and state["environment"] == RUNTIME_ENVIRONMENT, "Actual native runtime version/environment differs")
    for name, expected in REQUIREMENTS.items():
        if name != "torch_version":
            require(state.get(name) is expected, "Actual strict runtime initialization differs")
    require(isinstance(state["initial_flags"], dict)
            and set(state["initial_flags"]) == set(state["training_flags"])
            and all(type(v) is bool for v in state["initial_flags"].values())
            and state["cudnn_sdp_flag_unchanged"] is True,
            "Actual initial backend flags are missing or changed unexpectedly")
    require(all(state["training_flags"].get(k) is v for k, v in STRICT_FLAGS.items()),
            "Actual strict training flags differ")


def _validate_hyperparams(payload, contract, output):
    task, method = _task(contract), contract["method"]
    args, stage, config = payload["args"], payload["stage_hyperparameters"], payload["trainer_config"]
    profile = args["co_deterministic_runtime_receipt"]
    actual = stage["runtime_profile"]
    require(payload["stage"] == "grpo" and payload["tasks"] == [task]
            and payload["output_dir"] == str(output)
            and canonical(payload["distributed"]) == canonical({"world_size": 4, "rank": 0}),
            "Actual runtime metadata belongs to another task/output/rank")
    require(args["co_method"] == method and args["tasks"] == [task]
            and args["model_path"] == contract["initialization"]["checkpoint"]
            and args["data_root"] == contract["data_root"] and args["output_dir"] == str(output),
            "Actual deterministic arguments differ from the original method/data/SFT")
    for metadata in (profile, actual):
        require(metadata["profile_id"] == PROFILE_ID and metadata["method_id"] == method
                and metadata["base_profile_id"] != PROFILE_ID
                and metadata["source_binding"]["manifest"]["sha256"] == contract["source"]["manifest_sha256"],
                "Actual runtime profile/source is not the declared successor")
        _validate_applied_state(metadata["determinism"])
    require(profile["determinism"] == actual["determinism"], "Argument/profile applied runtime receipts disagree")
    require(config["max_steps"] == 400 and type(config["beta"]) in (int, float)
            and config["beta"] == (0. if method == "classic_grpo" else .05)
            and config["full_determinism"] is False and config["num_generations"] == 9
            and config["generation_batch_size"] == 36 and config["gradient_accumulation_steps"] == 3,
            "Actual GRPO horizon/algorithm differs from deterministic common study")
    return profile


def _postrun_bindings(output, stop, contract, job_id, profile):
    result = []
    source = Path(contract["source"]["root"])
    for rank in range(4):
        path = output / f"postrun_determinism_step{stop}_rank{rank}.json"
        value = read_signed(path)
        expected = {"protocol": "co-strict-determinism-postrun-v1", "status": "passed_postrun_strict_flags",
                    "profile_id": PROFILE_ID, "method": contract["method"], "step": stop, "rank": rank,
                    "world_size": 4, "job_id": job_id,
                    "source": {"root": str(source), "manifest": regular_identity(source / "SOURCE_MANIFEST.sha256")},
                    "original_contract": regular_identity(output / "local_development_run_contract.json"),
                    "environment": RUNTIME_ENVIRONMENT}
        require(canonical({k: value.get(k) for k in expected}) == canonical(expected),
                "Per-rank postrun determinism belongs to another run/source/job/step")
        require(value["torch_version"].split("+")[0] == "2.9.1"
                and all(value["training_flags"].get(k) is v for k, v in STRICT_FLAGS.items()),
                "Per-rank postrun deterministic flags/version differ")
        _validate_applied_state(value["initialization"])
        if rank == 0:
            require(value["initialization"] == profile["determinism"],
                    "Rank-zero pre-CUDA setup and actual hyperparameters disagree")
        result.append({"rank": rank, "receipt": regular_identity(path)})
    return result


def _evidence_material(checkpoint, producer_stop, contract, original_hyperparams):
    checkpoint = Path(checkpoint).absolute()
    output = checkpoint.parent.parent
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    require(checkpoint.parent == output / "checkpoints" and checkpoint.name == f"checkpoint-{step}"
            and type(producer_stop) is int and producer_stop >= step
            and producer_stop in ((5, 10) if contract["diagnostic"] else (100, 200, 300, 400)),
            "Runtime evidence checkpoint/producer boundary differs")
    declaration = read_only_runtime_declaration(output, contract)
    admission_path = output / f"chunk-{producer_stop}-admission.json"
    admission = read_json(admission_path)
    require(admission["stop"] == producer_stop and admission["contract"] == declaration["run_contract"]
            and admission["deterministic_runtime"] == regular_identity(output / DECLARATION_NAME)
            and admission["torch_distribution_version"].split("+")[0] == "2.9.1",
            "Applied runtime does not bind its original chunk admission")
    command = admission["command"]
    entry = "clasp.entrypoints.train_co_deterministic_dev"
    require(command.count(entry) == 1 and command[command.index(entry) + 1:command.index(entry) + 3]
            == ["--co_method", contract["method"]], "Applied runtime admission did not execute the successor entry")
    exit_path = output / f"chunk-{producer_stop}-exit.json"
    returncode = read_json(exit_path)["returncode"]
    require(type(returncode) is int and returncode == 0, "Applied runtime producer did not complete training")
    snapshot = checkpoint / SNAPSHOT_NAME
    snapshot_identity = regular_identity(snapshot)
    require(original_hyperparams["path"] == str(output / "hyperparams.json")
            and {k: snapshot_identity[k] for k in ("sha256", "size_bytes")}
            == {k: original_hyperparams[k] for k in ("sha256", "size_bytes")},
            "Runtime hyperparameter snapshot differs from the retained original identity")
    profile = _validate_hyperparams(read_json(snapshot), contract, output)
    postrun = _postrun_bindings(output, producer_stop, contract, admission["job_id"], profile)
    return {"protocol": EVIDENCE_PROTOCOL, "status": "verified_applied_deterministic_runtime",
            "checkpoint": str(checkpoint), "checkpoint_step": step,
            "producer_stop": producer_stop, "producer_job_id": admission["job_id"],
            "admission": regular_identity(admission_path), "successful_subprocess_exit": regular_identity(exit_path),
            "declaration": regular_identity(output / DECLARATION_NAME),
            "hyperparams": snapshot_identity, "original_hyperparams_historical_identity": original_hyperparams,
            "runtime_profile": profile, "postrun_per_rank_checks": postrun,
            "scope": "applied_runtime_readiness_not_proof_of_exact_numerical_reproducibility"}


def save_runtime_evidence(output, checkpoint, producer_stop, contract):
    source = Path(output) / "hyperparams.json"
    before = regular_identity(source)
    raw = source.read_bytes()
    _validate_hyperparams(read_json(source), contract, Path(output))
    require(regular_identity(source) == before and hashlib.sha256(raw).hexdigest() == before["sha256"],
            "Actual runtime metadata changed during snapshot")
    snapshot = Path(checkpoint) / SNAPSHOT_NAME
    with snapshot.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    snapshot.chmod(0o440)
    material = _evidence_material(checkpoint, producer_stop, contract, before)
    write_once(Path(checkpoint) / EVIDENCE_NAME, signed(material))
    return read_only_runtime_evidence(checkpoint, contract)


def read_only_runtime_evidence(checkpoint, contract):
    path = Path(checkpoint) / EVIDENCE_NAME
    value = read_signed(path)
    expected = _evidence_material(checkpoint, value["producer_stop"], contract,
                                 value["original_hyperparams_historical_identity"])
    require(value == signed(expected), "Applied runtime evidence changed")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed-slurm", required=True)
    args = parser.parse_args()
    torch_version = validate_launcher_environment()
    source = Path(os.environ["CO_SOURCE"]).resolve(strict=True)
    expected_sha = os.environ["CO_SOURCE_SHA"]
    task, method = os.environ["CO_TASK"], os.environ["CO_METHOD"]
    if task not in SFT_HASHES or method not in METHODS:
        raise ValueError("Unsupported specialist task")
    weight = .05 if method == "foarl_teacher" else 0.
    diagnostic_raw = os.environ.get("CO_DIAGNOSTIC", "0")
    if diagnostic_raw not in ("0", "1"):
        raise ValueError("CO_DIAGNOSTIC must be 0 or 1")
    diagnostic = diagnostic_raw == "1"
    stop = int(os.environ["CO_STOP"])
    if stop not in ((5, 10) if diagnostic else (100, 200, 300, 400)):
        raise ValueError("Undeclared stop boundary")
    output = Path(os.environ["CO_OUTPUT"]).absolute()
    resume = Path(os.environ["CO_RESUME"]).resolve(strict=True) if os.environ.get("CO_RESUME") else None
    compare_with = Path(os.environ["CO_COMPARE_WITH"]).resolve(strict=True) if os.environ.get("CO_COMPARE_WITH") else None
    if compare_with is not None and not (diagnostic and resume is not None and stop == 10):
        raise ValueError("Continuation comparison is restricted to the resumed diagnostic")
    sft = SFT / "runs" / task / "seed42/checkpoints/checkpoint-31250"
    if Path(sys.executable).resolve() != (RUNTIME / "bin/python").resolve():
        raise ValueError("Wrong native interpreter")
    from clasp.baselines.foarl_f638_dev_20260907 import verify_dev_source_manifest
    binding = verify_dev_source_manifest(source / "SOURCE_MANIFEST.sha256", expected_sha)
    if Path(args.executed_slurm).read_bytes() != (source / "scripts/train_co_deterministic_dev_a100.slurm").read_bytes():
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
        "teacher_weight": weight, "seed": 43, "data_root": str(DATA), "data_files": data_files,
        "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3,
                     "steps_per_generation": 3, "save_steps": 5 if diagnostic else 100},
        "diagnostic": diagnostic,
    }
    if method == "classic_grpo":
        contract["beta"] = 0.0
    contract_path = output / "local_development_run_contract.json"
    resume_receipt = None
    if resume is None:
        if stop not in ((5, 10) if diagnostic else (100,)):
            raise ValueError("Fresh production arm must start at step100")
        output.mkdir(parents=True, exist_ok=False)
        write_once(contract_path, contract)
        write_runtime_declaration(output, contract)
    elif not output.is_dir():
        raise ValueError("Resume requires the existing training output")
    # Hold across validation, training, and terminal receipt. Parallel chunks
    # cannot mutate the same optimizer/checkpoint directory.
    with (output / ".chunk.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_only_runtime_declaration(output, contract)
        if resume is not None:
            expected_step = stop - (5 if diagnostic else 100)
            if resume != output / "checkpoints" / f"checkpoint-{expected_step}":
                raise ValueError("Resume must use the immediately preceding chunk in this arm")
            resume_receipt = validate_checkpoint(method, resume, expected_step, contract, contract_path)
            read_only_runtime_evidence(resume, contract)
        # Clear inherited research, caching and resume controls before restating
        # this study's complete environment. CO_* only selects the above contract.
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
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8", "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONHASHSEED": "43",
            "HF_HOME": str(cache / "huggingface"), "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
            "TMPDIR": str(cache / "tmp"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "WANDB_MODE": "offline", "WANDB_DIR": str(cache / "wandb"),
        })
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="43")
        if method == "classic_grpo":
            os.environ.update(GRPO_GRADED_REWARD="0", CLASP_OBJECTIVE_QUALITY_MODE="exp",
                              CLASP_OBJECTIVE_GAP_SLOPE="3.0", CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT="0.0")
        else:
            os.environ.update(CLASP_KL_REFERENCE_MODEL_PATH=str(sft),
                              CLASP_KL_REFERENCE_RECEIPT=str(sft / "checkpoint_fidelity.json"))
        if resume:
            os.environ["RESUME_FROM_CHECKPOINT"] = str(resume)
        flag_prefix = "classic" if method == "classic_grpo" else "foarl"
        command = ["srun", "--ntasks=1", "--ntasks-per-node=1", str(RUNTIME / "bin/python"),
                   "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   "-m", "clasp.entrypoints.train_co_deterministic_dev", "--co_method", method,
                   "--tasks", task, "--sft_checkpoint_path", str(sft), "--tokenizer_path", str(sft),
                   "--data_root", str(DATA), "--output_dir", str(output),
                   "--" + flag_prefix + "_dev_source_manifest", str(source / "SOURCE_MANIFEST.sha256"),
                   "--" + flag_prefix + "_dev_source_manifest_sha256", expected_sha,
                   "--eval_ratio", "0", "--eval-from-train-split",
                   "--max_samples_per_task", "0", "--max_eval_samples_per_task", "0",
                   "--min_train_samples_per_task", "1900", "--min_eval_samples_per_task", "0",
                   "--no-prefer_subset_files", "--report_to", "none"]
        if method != "classic_grpo":
            command += ["--foarl_teacher_weight", str(weight)]
        if diagnostic:
            command += ["--classic_grpo_diagnostic" if method == "classic_grpo" else "--foarl_teacher_diagnostic",
                        "--save_steps", "5"]
        receipt = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "job_id": os.environ["SLURM_JOB_ID"], "stop": stop, "command": command,
                   "contract": identity(contract_path), "resume": resume_receipt,
                   "deterministic_runtime": identity(output / DECLARATION_NAME),
                   "torch_distribution_version": torch_version,
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
        # Classic checkpoints (including full-run intermediate5) are sealed by
        # the new entry's on-save callback while their live consensus is current.
        completed = validate_checkpoint(method, checkpoint, stop, contract, contract_path)
        save_runtime_evidence(output, checkpoint, stop, contract)
        if diagnostic and stop == 10 and resume is None:
            intermediate = output / "checkpoints/checkpoint-5"
            validate_checkpoint(method, intermediate, 5, contract, contract_path)
            save_runtime_evidence(output, intermediate, stop, contract)
        write_once(output / f"chunk-{stop}-complete.json", completed)
        if compare_with is not None:
            subprocess.run([sys.executable, "-B", str(source / "scripts/compare_co_deterministic_state.py"),
                            "--uninterrupted", str(compare_with), "--resumed", str(checkpoint), "--step", "10",
                            "--output", str(output / "resume_equivalence.json")], check=True)
        print(f"CO_DETERMINISTIC_CHUNK_COMPLETE task={task} method={method} step={stop}", flush=True)


if __name__ == "__main__":
    main()
