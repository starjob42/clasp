"""Actual common argv, provenance snapshots and fail-closed continuation gates.

Only machine interfaces and already-tested method checkpoint inspectors are
stubbed. The runtime declaration/evidence readers run against real files.
"""
from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines import foarl_f638_dev_20260907 as source_guard
from clasp.baselines.classic_grpo_400_dev import build_runtime as classic_runtime
from clasp.baselines.foarl_teacher_dev import build_runtime as foarl_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_classic_grpo_400_dev import build_parser as classic_parser
from clasp.entrypoints.train_foarl_teacher_dev import build_parser as foarl_parser
from clasp.models import classic_grpo_resume, foarl_teacher_resume

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_json(path, value):
    path.chmod(0o640)
    path.write_text(json.dumps(value, sort_keys=True))
    path.chmod(0o440)


@pytest.fixture
def chunk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "run_foarl_teacher_dev_chunk", load_script("run_foarl_teacher_dev_chunk"))
    m = load_script("run_co_deterministic_dev_chunk")
    source = tmp_path / "source"
    for name in m.IMPLEMENTATIONS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("synthetic pinned source\n")
    manifest_sha = m.identity(manifest)["sha256"]
    monkeypatch.setattr(m, "__file__", str(source / m.IMPLEMENTATIONS[0]))
    spool = tmp_path / "spool.slurm"
    spool.write_bytes((source / m.IMPLEMENTATIONS[1]).read_bytes())
    sft_root, data = tmp_path / "sft", tmp_path / "data"
    sft = sft_root / "runs/mis/seed42/checkpoints/checkpoint-31250"
    sft.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        (sft / name).write_text(name)
    (data / "mis/train").mkdir(parents=True)
    (data / "mis/train/train.json").write_text('[{"synthetic":true}]\n')
    output = tmp_path / "arm"
    binding = {"status": "passed_frozen_manifest_successor_source", "verified_file_count": 19,
               "manifest": m.identity(manifest),
               "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "a" * 64}}}
    monkeypatch.setattr(m, "RUNTIME", Path(sys.executable).parent.parent)
    monkeypatch.setattr(m, "DATA", data)
    monkeypatch.setattr(m, "SFT", sft_root)
    monkeypatch.setattr(m, "SFT_HASHES", {"mis": m.identity(sft / "adapter_model.safetensors")["sha256"]})
    monkeypatch.setattr(m.importlib.metadata, "version", lambda _: "2.9.1+cu128")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    os.environ.update(CO_SOURCE=str(source), CO_SOURCE_SHA=manifest_sha, CO_TASK="mis",
                      CO_METHOD="foarl_teacher", CO_OUTPUT=str(output), CO_STOP="100", CO_DIAGNOSTIC="0",
                      CO_RESUME="", CO_COMPARE_WITH="", SLURM_JOB_ID="12345", WORLD_SIZE="4",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="43",
                      RESUME_FROM_CHECKPOINT="/poisoned/resume", CLASP_KL_REFERENCE_MODEL_PATH="/poisoned/reference",
                      CLASP_KL_REFERENCE_RECEIPT="/poisoned/receipt", GRPO_GRADED_REWARD="1",
                      GRPO_VERIFIED_IMPROVEMENT="1", FOARL_UNDECLARED_CONTROL="poison")
    monkeypatch.setattr(sys, "argv", ["chunk", "--executed-slurm", str(spool)])
    source_checks, validations, commands = [], [], []
    state = SimpleNamespace(module=m, source=source, spool=spool, sft=sft, data=data, output=output,
                            binding=binding, source_checks=source_checks, validations=validations,
                            commands=commands, mutate_payload=None, mutate_postrun=None)

    def verify(path, digest):
        source_checks.append((Path(path), digest))
        assert m.identity(path)["sha256"] == digest
        return binding

    def inspect(kind, checkpoint, **kwargs):
        validations.append((kind, Path(checkpoint), kwargs["expected_step"]))
        assert Path(checkpoint).is_dir(), "Training must save before inspection"
        assert m.read_json(kwargs["run_contract_path"]) == kwargs["expected_contract"]
        return {"status": "synthetic_checkpoint_inspected", "step": kwargs["expected_step"], "kind": kind}

    def payload_for(contract, stop):
        method = contract["method"]
        flags = {**m.STRICT_FLAGS, "cudnn_sdp_enabled": True}
        initialization = {**m.REQUIREMENTS, "torch_version": "2.9.1+cu128", "profile_id": m.PROFILE_ID,
                          "environment": dict(m.RUNTIME_ENVIRONMENT), "training_flags": flags,
                          "initial_flags": {**flags, "deterministic_algorithms": False},
                          "cudnn_sdp_flag_unchanged": True}
        base = (classic_runtime(binding, "mis", contract["diagnostic"]) if method == "classic_grpo"
                else foarl_runtime(binding, contract["teacher_weight"], contract["diagnostic"]))
        profile = base.metadata()
        profile.update(base_profile_id=profile["profile_id"], profile_id=m.PROFILE_ID,
                       determinism=initialization)
        payload = {"stage": "grpo", "tasks": ["mis"], "output_dir": str(output),
                   "distributed": {"world_size": 4, "rank": 0},
                   "args": {"co_method": method, "tasks": ["mis"], "model_path": str(sft),
                            "data_root": str(data), "output_dir": str(output),
                            "co_deterministic_runtime_receipt": copy.deepcopy(profile)},
                   "stage_hyperparameters": {"runtime_profile": copy.deepcopy(profile)},
                   "trainer_config": {"max_steps": 400, "beta": 0. if method == "classic_grpo" else .05,
                                      "full_determinism": False, "num_generations": 9,
                                      "generation_batch_size": 36, "gradient_accumulation_steps": 3}}
        return payload, initialization

    def execute(argv, **kwargs):
        with (output / ".chunk.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands.append({"argv": argv, "environment": dict(os.environ), "kwargs": kwargs})
        if argv[0] != "srun":
            return SimpleNamespace(returncode=0)
        stop = int(os.environ["CO_STOP"])
        contract = m.read_json(output / "local_development_run_contract.json")
        for saved in ([5, 10] if contract["diagnostic"] and stop == 10 and not os.environ.get("CO_RESUME") else [stop]):
            (output / "checkpoints" / f"checkpoint-{saved}").mkdir(parents=True, exist_ok=False)
        payload, initialization = payload_for(contract, stop)
        if state.mutate_payload:
            state.mutate_payload(payload)
        (output / "hyperparams.json").write_text(json.dumps(payload, sort_keys=True))
        for rank in range(4):
            proof = {"protocol": "co-strict-determinism-postrun-v1", "status": "passed_postrun_strict_flags",
                     "profile_id": m.PROFILE_ID, "method": contract["method"], "step": stop, "rank": rank,
                     "world_size": 4, "job_id": os.environ["SLURM_JOB_ID"],
                     "source": {"root": str(source), "manifest": m.identity(manifest)},
                     "original_contract": m.identity(output / "local_development_run_contract.json"),
                     "environment": dict(m.RUNTIME_ENVIRONMENT), "torch_version": "2.9.1+cu128",
                     "initialization": copy.deepcopy(initialization), "training_flags": copy.deepcopy(initialization["training_flags"]),
                     "scope": "endpoint_execution_flags_not_per_kernel_or_continuation_proof",
                     "requires_all_rank_receipts_and_successful_job": True}
            if state.mutate_postrun:
                state.mutate_postrun(proof)
            m.write_once(output / f"postrun_determinism_step{stop}_rank{rank}.json", m.signed(proof))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(source_guard, "verify_dev_source_manifest", verify)
    monkeypatch.setattr(classic_grpo_resume, "validate_classic_grpo_resume", lambda p, **kw: inspect("classic", p, **kw))
    monkeypatch.setattr(foarl_teacher_resume, "validate_foarl_teacher_resume", lambda p, **kw: inspect("foarl", p, **kw))
    # An endpoint must already have been sealed by the new on-save callback.
    monkeypatch.setattr(classic_grpo_resume, "seal_classic_grpo_checkpoint",
                        lambda *a, **kw: pytest.fail("Runner must never seal again"))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 4)
    monkeypatch.setattr(m.subprocess, "run", execute)
    state.execute = execute
    return state


@pytest.mark.parametrize("method", ["foarl_control", "foarl_teacher", "classic_grpo"])
@pytest.mark.parametrize("diagnostic,stop", [(False, 100), (True, 5), (True, 10)])
def test_actual_three_method_argv_and_applied_evidence(chunk, method, diagnostic, stop):
    os.environ.update(CO_METHOD=method, CO_DIAGNOSTIC=str(int(diagnostic)), CO_STOP=str(stop))
    chunk.module.main()
    train = next(c for c in chunk.commands if c["argv"][0] == "srun")
    argv = train["argv"]
    entry = argv.index("clasp.entrypoints.train_co_deterministic_dev")
    assert argv[entry + 1:entry + 3] == ["--co_method", method]
    parser = classic_parser() if method == "classic_grpo" else foarl_parser()
    args = finalize_grpo_args(parser.parse_args(argv[entry + 3:]))
    runtime = (classic_runtime(chunk.binding, "mis", diagnostic) if method == "classic_grpo"
               else foarl_runtime(chunk.binding, .05 if method == "foarl_teacher" else 0., diagnostic))
    runtime.validate_args(args)
    assert (args.max_steps, args.generation_batch_size, args.gradient_accumulation_steps) == (400, 36, 3)
    assert args.grpo_beta == (0. if method == "classic_grpo" else .05)
    assert args.save_steps == (5 if diagnostic else 100)
    env = train["environment"]
    assert {k: env.get(k) for k in chunk.module.RUNTIME_ENVIRONMENT} == chunk.module.RUNTIME_ENVIRONMENT
    assert env["GRPO_STOP_AT_GLOBAL_STEP"] == str(stop) and env["GRPO_STRICT_OPTIMIZER_RESUME"] == "1"
    assert "RESUME_FROM_CHECKPOINT" not in env and "FOARL_UNDECLARED_CONTROL" not in env
    if method == "classic_grpo":
        assert not any(k.startswith("CLASP_KL_REFERENCE_") for k in env)
        assert env["GRPO_GRADED_REWARD"] == "0"
    else:
        assert env["CLASP_KL_REFERENCE_MODEL_PATH"] == str(chunk.sft)
    contract = chunk.module.read_json(chunk.output / "local_development_run_contract.json")
    assert ("beta" in contract) == (method == "classic_grpo")  # Retain old validators' exact schema.
    proof = chunk.module.read_only_runtime_evidence(chunk.output / "checkpoints" / f"checkpoint-{stop}", contract)
    assert len(proof["postrun_per_rank_checks"]) == 4 and proof["producer_stop"] == stop
    assert {v[0] for v in chunk.validations} == {"classic" if method == "classic_grpo" else "foarl"}
    if diagnostic and stop == 10:
        intermediate = chunk.module.read_only_runtime_evidence(chunk.output / "checkpoints/checkpoint-5", contract)
        assert intermediate["checkpoint_step"] == 5 and intermediate["producer_stop"] == 10
    assert (chunk.output / f"chunk-{stop}-complete.json").is_file()


@pytest.mark.parametrize("method", ["foarl_control", "foarl_teacher", "classic_grpo"])
def test_same_arm_resume_retains_original_contract_and_old_checkpoint_evidence(chunk, method):
    os.environ["CO_METHOD"] = method
    chunk.module.main()
    contract = chunk.module.read_json(chunk.output / "local_development_run_contract.json")
    prior = chunk.output / "checkpoints/checkpoint-100"
    proof = chunk.module.read_only_runtime_evidence(prior, contract)
    declaration = (chunk.output / chunk.module.DECLARATION_NAME).read_bytes()
    os.environ.update(CO_STOP="200", CO_RESUME=str(prior), SLURM_JOB_ID="12346")
    chunk.commands.clear()
    chunk.module.main()
    assert chunk.commands[0]["environment"]["RESUME_FROM_CHECKPOINT"] == str(prior)
    assert (chunk.output / chunk.module.DECLARATION_NAME).read_bytes() == declaration
    assert chunk.module.read_only_runtime_evidence(prior, contract) == proof
    before = len(chunk.commands)
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(chunk.commands) == before


@pytest.mark.parametrize("change", ["other_arm", "wrong_step", "method", "data", "missing_evidence"])
def test_resume_rejects_changed_or_unproved_arm_before_child(chunk, change):
    chunk.module.main()
    prior = chunk.output / "checkpoints/checkpoint-100"
    if change in ("other_arm", "wrong_step"):
        prior = chunk.output / ("other/checkpoints/checkpoint-100" if change == "other_arm" else "checkpoints/checkpoint-5")
        prior.mkdir(parents=True)
    elif change == "method":
        os.environ["CO_METHOD"] = "foarl_control"
    elif change == "data":
        (chunk.data / "mis/train/train.json").write_text("changed")
    else:
        (prior / chunk.module.EVIDENCE_NAME).unlink()
    os.environ.update(CO_STOP="200", CO_RESUME=str(prior))
    chunk.commands.clear()
    with pytest.raises((ValueError, FileNotFoundError)):
        chunk.module.main()
    assert not chunk.commands and not (chunk.output / "chunk-200-admission.json").exists()


@pytest.mark.parametrize("failure", ["manifest", "slurm", "gpu", "sft", "cublas", "hashseed", "blocking", "torch"])
def test_bad_environment_source_or_hardware_precedes_output(chunk, monkeypatch, failure):
    if failure == "manifest":
        monkeypatch.setattr(source_guard, "verify_dev_source_manifest", lambda *a: (_ for _ in ()).throw(ValueError("source")))
    elif failure == "slurm":
        chunk.spool.write_text("different")
    elif failure == "gpu":
        monkeypatch.setattr(chunk.module.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 3)
    elif failure == "sft":
        (chunk.sft / "adapter_model.safetensors").write_text("different")
    elif failure == "torch":
        monkeypatch.setattr(chunk.module.importlib.metadata, "version", lambda _: "2.5.1")
    else:
        os.environ[{"cublas": "CUBLAS_WORKSPACE_CONFIG", "hashseed": "PYTHONHASHSEED", "blocking": "CUDA_LAUNCH_BLOCKING"}[failure]] = "1"
    with pytest.raises(ValueError):
        chunk.module.main()
    assert not chunk.output.exists() and not chunk.commands


@pytest.mark.parametrize("failure", ["torch", "strict", "late_cuda", "hf_debug", "method", "source", "sft", "beta"])
def test_actual_applied_profile_drift_prevents_completion(chunk, failure):
    def mutate(payload):
        if failure in ("torch", "strict", "late_cuda", "hf_debug"):
            for profile in (payload["args"]["co_deterministic_runtime_receipt"], payload["stage_hyperparameters"]["runtime_profile"]):
                state = profile["determinism"]
                if failure == "strict":
                    state["training_flags"]["deterministic_algorithms"] = False
                else:
                    state[{"torch": "torch_version", "late_cuda": "applied_before_cuda_initialization", "hf_debug": "hf_full_determinism"}[failure]] = {"torch": "2.5.1", "late_cuda": False, "hf_debug": True}[failure]
        elif failure == "method":
            payload["args"]["co_method"] = "foarl_control"
        elif failure == "source":
            payload["args"]["co_deterministic_runtime_receipt"]["source_binding"]["manifest"]["sha256"] = "f" * 64
        elif failure == "sft":
            payload["args"]["model_path"] = "wrong"
        else:
            payload["trainer_config"]["beta"] = 0.
    chunk.mutate_payload = mutate
    with pytest.raises(ValueError):
        chunk.module.main()
    assert not (chunk.output / "chunk-100-complete.json").exists()


@pytest.mark.parametrize("failure", ["rank", "job", "source", "torch", "strict", "initialization"])
def test_each_worker_postrun_proof_is_required(chunk, failure):
    def mutate(proof):
        if proof["rank"] != 2:
            return
        if failure in ("rank", "job", "torch"):
            proof[{"rank": "rank", "job": "job_id", "torch": "torch_version"}[failure]] = {"rank": 0, "job": "999", "torch": "2.5.1"}[failure]
        elif failure == "source":
            proof["source"]["manifest"]["sha256"] = "f" * 64
        elif failure == "strict":
            proof["training_flags"]["deterministic_warn_only"] = True
        else:
            proof["initialization"]["applied_before_cuda_initialization"] = False
    chunk.mutate_postrun = mutate
    with pytest.raises(ValueError):
        chunk.module.main()
    assert not (chunk.output / "chunk-100-complete.json").exists()


@pytest.mark.parametrize("artifact", ["declaration", "snapshot", "postrun", "admission", "source", "missing_rank"])
def test_readmission_rehashes_current_retained_artifacts(chunk, artifact):
    chunk.module.main()
    m = chunk.module
    contract = m.read_json(chunk.output / "local_development_run_contract.json")
    checkpoint = chunk.output / "checkpoints/checkpoint-100"
    if artifact == "source":
        (chunk.source / m.IMPLEMENTATIONS[2]).write_text("changed source")
    elif artifact == "missing_rank":
        (chunk.output / "postrun_determinism_step100_rank3.json").unlink()
    else:
        path = {"declaration": chunk.output / m.DECLARATION_NAME, "snapshot": checkpoint / m.SNAPSHOT_NAME,
                "postrun": chunk.output / "postrun_determinism_step100_rank1.json",
                "admission": chunk.output / "chunk-100-admission.json"}[artifact]
        value = m.read_json(path)
        value["tampered"] = True
        if "receipt_sha256" in value:
            value = m.signed({k: v for k, v in value.items() if k != "receipt_sha256"})
        replace_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.read_only_runtime_evidence(checkpoint, contract)


def test_failure_retains_attempt_without_completion_or_retry(chunk, monkeypatch):
    monkeypatch.setattr(chunk.module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit) as failed:
        chunk.module.main()
    assert failed.value.code == 7 and not chunk.validations
    assert chunk.module.read_json(chunk.output / "chunk-100-exit.json")["returncode"] == 7
    assert not (chunk.output / "chunk-100-complete.json").exists()
    with pytest.raises(FileExistsError):
        chunk.module.main()


def test_diagnostic_resume_routes_generic_step10_comparator(chunk, tmp_path):
    os.environ.update(CO_DIAGNOSTIC="1", CO_STOP="5")
    chunk.module.main()
    full = tmp_path / "full/checkpoints/checkpoint-10"
    full.mkdir(parents=True)
    os.environ.update(CO_STOP="10", CO_RESUME=str(chunk.output / "checkpoints/checkpoint-5"), CO_COMPARE_WITH=str(full), SLURM_JOB_ID="12346")
    chunk.commands.clear()
    chunk.module.main()
    command = chunk.commands[-1]
    assert command["kwargs"] == {"check": True}
    assert command["argv"] == [sys.executable, "-B", str(chunk.source / "scripts/compare_co_deterministic_state.py"),
                               "--uninterrupted", str(full), "--resumed", str(chunk.output / "checkpoints/checkpoint-10"),
                               "--step", "10", "--output", str(chunk.output / "resume_equivalence.json")]


def test_scientific_inputs_keep_non_symlink_guard_and_json_types(chunk, tmp_path):
    target = tmp_path / "real.json"
    target.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        chunk.module.read_json(target)
    target.write_text('{"a":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        chunk.module.read_json(target)
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        chunk.module.read_json(alias)


def test_slurm_sets_native_determinism_before_python_and_retains_resources():
    text = (ROOT / "scripts/train_co_deterministic_dev_a100.slurm").read_text()
    for token in ("#SBATCH --constraint=a100", "#SBATCH --gres=gpu:4", "#SBATCH --time=02:00:00",
                  "#SBATCH --qos=qos_gpu_a100-dev", "#SBATCH --no-requeue", "#SBATCH --cpus-per-task=40"):
        assert token in text
    startup = text.index('exec "${RUNTIME}/bin/python" -B')
    assert text.index('export CUBLAS_WORKSPACE_CONFIG=:4096:8') < startup
    assert text.index('PYTHONHASHSEED=43') < startup
    assert text.index('unset CUDA_LAUNCH_BLOCKING') < startup
