"""Exercise actual chunk argv/environment and fail-closed launch boundaries.

Only machine interfaces and the separately tested checkpoint inspector are
stubbed. These tests neither load a model nor invoke Slurm or GPUs.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines import foarl_f638_dev_20260907 as source_guard
from clasp.baselines.classic_grpo_400_dev import REWARD_ENVIRONMENT, build_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_classic_grpo_400_dev import build_parser
from clasp.models import classic_grpo_resume

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chunk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "run_foarl_teacher_dev_chunk", load_script("run_foarl_teacher_dev_chunk"))
    module = load_script("run_classic_grpo_400_dev_chunk")
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    slurm = source / "scripts/train_classic_grpo_400_dev_a100.slurm"
    slurm.write_bytes((ROOT / "scripts/train_classic_grpo_400_dev_a100.slurm").read_bytes())
    spool = tmp_path / "spool.slurm"
    spool.write_bytes(slurm.read_bytes())
    (source / "SOURCE_MANIFEST.sha256").write_text("synthetic pinned manifest\n")
    sft_root, data = tmp_path / "sft", tmp_path / "data"
    sft = sft_root / "runs/mis/seed42/checkpoints/checkpoint-31250"
    sft.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        (sft / name).write_text(name)
    (data / "mis/train").mkdir(parents=True)
    (data / "mis/train/train.json").write_text('[{"synthetic":true}]\n')
    output = tmp_path / "teacher"
    binding = {"status": "passed_frozen_manifest_successor_source", "verified_file_count": 17,
               "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "a" * 64}}}
    monkeypatch.setattr(module, "RUNTIME", Path(sys.executable).parent.parent)
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "SFT", sft_root)
    monkeypatch.setattr(module, "SFT_HASHES", {"mis": module.identity(sft / "adapter_model.safetensors")["sha256"]})
    # Replace the mapping itself so scrubbed inherited variables are restored.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.update(CG_SOURCE=str(source), CG_SOURCE_SHA="a" * 64, CG_TASK="mis",
                      CG_OUTPUT=str(output), CG_STOP="100",
                      CG_DIAGNOSTIC="0", CG_RESUME="", CG_COMPARE_WITH="", SLURM_JOB_ID="12345",
                      WORLD_SIZE="4", RESUME_FROM_CHECKPOINT="/poisoned/resume",
                      CLASP_KL_REFERENCE_MODEL_PATH="/poisoned/reference",
                      CLASP_KL_REFERENCE_RECEIPT="/poisoned/receipt", GRPO_GRADED_REWARD="1",
                      GRPO_STOP_AT_GLOBAL_STEP="999", GRPO_VERIFIED_IMPROVEMENT="1",
                      FOARL_UNDECLARED_CONTROL="poison", PD_UNDECLARED_CONTROL="poison")
    monkeypatch.setattr(sys, "argv", ["chunk", "--executed-slurm", str(spool)])
    source_checks, validations, commands = [], [], []
    seals = []

    def verify(path, digest):
        source_checks.append((Path(path), digest))
        return binding

    def inspect(checkpoint, **kwargs):
        validations.append((Path(checkpoint), kwargs))
        contract = json.loads(kwargs["run_contract_path"].read_text())
        if contract != kwargs["expected_contract"]:
            raise ValueError("existing arm contract differs")
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
        return {"status": "synthetic_checkpoint_inspected", "step": kwargs["expected_step"]}

    def execute(argv, **kwargs):
        # Independent descriptor cannot enter the arm while a child runs.
        with (output / ".chunk.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands.append({"argv": argv, "environment": dict(os.environ), "kwargs": kwargs})
        return SimpleNamespace(returncode=0)

    def seal(checkpoint, **kwargs):
        stop = kwargs["expected_step"]
        assert any(c["argv"][0] == "srun" for c in commands)
        assert json.loads((output / f"chunk-{stop}-exit.json").read_text())["returncode"] == 0
        seals.append(Path(checkpoint))
        return inspect(checkpoint, **kwargs)

    monkeypatch.setattr(source_guard, "verify_dev_source_manifest", verify)
    monkeypatch.setattr(classic_grpo_resume, "validate_classic_grpo_resume", inspect)
    monkeypatch.setattr(classic_grpo_resume, "seal_classic_grpo_checkpoint", seal)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 4)
    monkeypatch.setattr(module.subprocess, "run", execute)
    return SimpleNamespace(module=module, source=source, spool=spool, sft=sft, data=data,
                           output=output, binding=binding, source_checks=source_checks,
                           validations=validations, seals=seals, commands=commands)


@pytest.mark.parametrize("diagnostic,stop", [(False, 100), (True, 5), (True, 10)])
def test_actual_chunk_argv_passes_profile_without_kl_or_teacher(chunk, diagnostic, stop):
    os.environ.update(CG_DIAGNOSTIC=str(int(diagnostic)), CG_STOP=str(stop))
    chunk.module.main()
    train = next(c for c in chunk.commands if c["argv"][0] == "srun")
    argv = train["argv"]
    entry = argv.index("clasp.entrypoints.train_classic_grpo_400_dev")
    args = finalize_grpo_args(build_parser().parse_args(argv[entry + 1:]))
    runtime = build_runtime(chunk.binding, "mis", diagnostic)
    runtime.validate_args(args)
    assert (args.max_steps, args.generation_batch_size, args.gradient_accumulation_steps) == (400, 36, 3)
    assert args.grpo_beta == 0. and args.no_reason and args.verify_gt_rewards
    assert args.per_device_train_batch_size == 3 and args.per_device_eval_batch_size == 9
    assert args.save_steps == (5 if diagnostic else 100)
    assert args.min_train_samples_per_task == 1900 and args.max_samples_per_task == 0
    env = train["environment"]
    assert not any(name.startswith("CLASP_KL_REFERENCE_") for name in env)
    assert all(env[name] == value for name, value in REWARD_ENVIRONMENT.items())
    assert env["CLASP_MIS_JIANG_RESPONSE_OUTPUT"] == "0"
    assert env["GRPO_STOP_AT_GLOBAL_STEP"] == str(stop)
    assert env["GRPO_STRICT_OPTIMIZER_RESUME"] == "1"
    assert env["CLASP_GRADIENT_CONSISTENCY_CHECKS"] == str(int(diagnostic))
    assert "RESUME_FROM_CHECKPOINT" not in env
    assert "FOARL_UNDECLARED_CONTROL" not in env and "PD_UNDECLARED_CONTROL" not in env
    assert chunk.source_checks == [(chunk.source / "SOURCE_MANIFEST.sha256", "a" * 64)]
    assert (chunk.output / f"chunk-{stop}-complete.json").is_file()
    assert chunk.seals == [chunk.output / "checkpoints" / f"checkpoint-{stop}"]
    assert chunk.validations[-2:] == [chunk.validations[-1]] * 2  # Seal then independent revalidation.
    contract = json.loads((chunk.output / "local_development_run_contract.json").read_text())
    assert contract["method"] == "classic_grpo" and contract["beta"] == contract["teacher_weight"] == 0.
    assert sum("probe_foarl_teacher_tokenizer.py" in " ".join(c["argv"]) for c in chunk.commands) == int(diagnostic)


def test_same_arm_resume_retains_horizon_data_and_original_sft_without_kl(chunk):
    chunk.module.main()
    contract = (chunk.output / "local_development_run_contract.json").read_bytes()
    checkpoint = chunk.output / "checkpoints/checkpoint-100"
    chunk.commands.clear()
    os.environ.update(CG_STOP="200", CG_RESUME=str(checkpoint))
    chunk.module.main()
    train = chunk.commands[0]
    assert train["environment"]["RESUME_FROM_CHECKPOINT"] == str(checkpoint)
    assert "CLASP_KL_REFERENCE_MODEL_PATH" not in train["environment"]
    assert train["argv"][train["argv"].index("--sft_checkpoint_path") + 1] == str(chunk.sft)
    assert (chunk.output / "local_development_run_contract.json").read_bytes() == contract
    assert chunk.validations[-3][0] == checkpoint
    assert chunk.validations[-1][1]["expected_step"] == 200
    # A second attempt at the same admitted boundary cannot rerun training.
    before = len(chunk.commands)
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(chunk.commands) == before


@pytest.mark.parametrize("change", ["other_arm", "wrong_step", "task", "data"])
def test_resume_rejects_cross_arm_nonadjacent_or_changed_contract_before_training(chunk, change):
    chunk.module.main()
    checkpoint = chunk.output / "checkpoints/checkpoint-100"
    if change == "other_arm":
        checkpoint = chunk.output.parent / "other/checkpoints/checkpoint-100"
        checkpoint.mkdir(parents=True)
    elif change == "wrong_step":
        checkpoint = chunk.output / "checkpoints/checkpoint-5"
        checkpoint.mkdir()
    elif change == "task":
        os.environ["CG_TASK"] = "cvrp"
    else:
        (chunk.data / "mis/train/train.json").write_text('[{"changed":true}]\n')
    os.environ.update(CG_STOP="200", CG_RESUME=str(checkpoint))
    chunk.commands.clear()
    with pytest.raises(ValueError):
        chunk.module.main()
    assert chunk.commands == []
    assert not (chunk.output / "chunk-200-admission.json").exists()


@pytest.mark.parametrize("failure", ["manifest", "slurm", "gpu", "sft"])
def test_source_and_hardware_fail_closed_before_output_or_child_process(chunk, monkeypatch, failure):
    if failure == "manifest":
        def reject(*args, **kwargs):
            raise ValueError("frozen source digest mismatch")
        monkeypatch.setattr(source_guard, "verify_dev_source_manifest", reject)
    elif failure == "slurm":
        chunk.spool.write_text("modified Slurm bytes")
    elif failure == "gpu":
        monkeypatch.setattr(chunk.module.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 3)
    else:
        (chunk.sft / "adapter_model.safetensors").write_text("wrong initialization")
    with pytest.raises(ValueError):
        chunk.module.main()
    assert not chunk.output.exists()
    assert chunk.commands == []


def test_failed_training_retains_one_attempt_and_never_creates_completion(chunk, monkeypatch):
    calls = []
    def fail(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(chunk.module.subprocess, "run", fail)
    with pytest.raises(SystemExit) as first:
        chunk.module.main()
    assert first.value.code == 7
    assert json.loads((chunk.output / "chunk-100-exit.json").read_text())["returncode"] == 7
    assert not (chunk.output / "chunk-100-complete.json").exists()
    assert chunk.seals == []
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(calls) == 1


def test_diagnostic_five_plus_five_keeps_same_contract_and_invokes_only_classic_comparator(chunk, tmp_path):
    os.environ.update(CG_DIAGNOSTIC="1", CG_STOP="5")
    chunk.module.main()
    contract = (chunk.output / "local_development_run_contract.json").read_bytes()
    full = tmp_path / "full/checkpoints/checkpoint-10"
    full.mkdir(parents=True)
    os.environ.update(CG_STOP="10", CG_RESUME=str(chunk.output / "checkpoints/checkpoint-5"), CG_COMPARE_WITH=str(full))
    chunk.commands.clear()
    chunk.module.main()
    assert (chunk.output / "local_development_run_contract.json").read_bytes() == contract
    assert len(chunk.commands) == 2 and chunk.commands[0]["argv"][0] == "srun"
    comparison = chunk.commands[1]
    assert comparison["argv"] == [sys.executable, "-B", str(chunk.source / "scripts/compare_classic_grpo_resume.py"),
        "--uninterrupted", str(full), "--resumed", str(chunk.output / "checkpoints/checkpoint-10"),
        "--output", str(chunk.output / "resume_equivalence.json")]
    assert comparison["kwargs"] == {"check": True}


def test_sealed_checkpoint_drift_prevents_complete_receipt(chunk, monkeypatch):
    monkeypatch.setattr(classic_grpo_resume, "seal_classic_grpo_checkpoint", lambda *a, **k: {"tampered": True})
    with pytest.raises(ValueError, match="changed after sealing"):
        chunk.module.main()
    assert (chunk.output / "chunk-100-exit.json").is_file()
    assert not (chunk.output / "chunk-100-complete.json").exists()


def test_slurm_requests_exact_resource_and_nonrequeue_profile():
    text = (ROOT / "scripts/train_classic_grpo_400_dev_a100.slurm").read_text()
    for directive in ("--account=vas@a100", "--partition=gpu_p5", "--constraint=a100", "--gres=gpu:4",
                      "--nodes=1", "--cpus-per-task=40", "--hint=nomultithread", "--time=02:00:00",
                      "--qos=qos_gpu_a100-dev", "--no-requeue"):
        assert text.splitlines().count("#SBATCH " + directive) == 1
    assert 'export PYTHONPATH="${CG_SOURCE:?}/src"' in text
    assert 'scripts/run_classic_grpo_400_dev_chunk.py --executed-slurm "${BASH_SOURCE[0]}"' in text
