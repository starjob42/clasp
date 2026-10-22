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
from clasp.baselines.foarl_teacher_dev import build_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_teacher_dev import build_parser
from clasp.models import foarl_teacher_resume

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chunk(tmp_path, monkeypatch):
    module = load_script("run_foarl_teacher_dev_chunk")
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    slurm = source / "scripts/train_foarl_teacher_dev_a100.slurm"
    slurm.write_bytes((ROOT / "scripts/train_foarl_teacher_dev_a100.slurm").read_bytes())
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
    os.environ.update(FT_SOURCE=str(source), FT_SOURCE_SHA="a" * 64, FT_TASK="mis",
                      FT_METHOD="foarl_teacher", FT_OUTPUT=str(output), FT_STOP="100",
                      FT_DIAGNOSTIC="0", FT_RESUME="", FT_COMPARE_WITH="", SLURM_JOB_ID="12345",
                      WORLD_SIZE="4", RESUME_FROM_CHECKPOINT="/poisoned/resume",
                      CLASP_KL_REFERENCE_MODEL_PATH="/poisoned/reference",
                      GRPO_STOP_AT_GLOBAL_STEP="999", GRPO_VERIFIED_IMPROVEMENT="1",
                      FOARL_UNDECLARED_CONTROL="poison", PD_UNDECLARED_CONTROL="poison")
    monkeypatch.setattr(sys, "argv", ["chunk", "--executed-slurm", str(spool)])
    source_checks, validations, commands = [], [], []

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
        commands.append({"argv": argv, "environment": dict(os.environ), "kwargs": kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(source_guard, "verify_dev_source_manifest", verify)
    monkeypatch.setattr(foarl_teacher_resume, "validate_foarl_teacher_resume", inspect)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "NVIDIA A100\n" * 4)
    monkeypatch.setattr(module.subprocess, "run", execute)
    return SimpleNamespace(module=module, source=source, spool=spool, sft=sft, data=data,
                           output=output, binding=binding, source_checks=source_checks,
                           validations=validations, commands=commands)


@pytest.mark.parametrize("method,weight", [("foarl_control", 0.0), ("foarl_teacher", 0.05)])
@pytest.mark.parametrize("diagnostic,stop", [(False, 100), (True, 5), (True, 10)])
def test_actual_chunk_argv_passes_profile_and_pins_original_reference(chunk, method, weight, diagnostic, stop):
    os.environ.update(FT_METHOD=method, FT_DIAGNOSTIC=str(int(diagnostic)), FT_STOP=str(stop))
    chunk.module.main()
    train = next(c for c in chunk.commands if c["argv"][0] == "srun")
    argv = train["argv"]
    entry = argv.index("clasp.entrypoints.train_foarl_teacher_dev")
    args = finalize_grpo_args(build_parser().parse_args(argv[entry + 1:]))
    runtime = build_runtime(chunk.binding, weight, diagnostic)
    runtime.validate_args(args)
    assert (args.max_steps, args.generation_batch_size, args.gradient_accumulation_steps) == (400, 36, 3)
    assert args.grpo_beta == 0.05 and args.no_reason and args.foarl_scale_rewards == "group"
    assert args.per_device_train_batch_size == 3 and args.per_device_eval_batch_size == 9
    assert args.save_steps == (5 if diagnostic else 100)
    assert args.min_train_samples_per_task == 1900 and args.max_samples_per_task == 0
    env = train["environment"]
    assert env["CLASP_KL_REFERENCE_MODEL_PATH"] == str(chunk.sft)
    assert env["CLASP_KL_REFERENCE_RECEIPT"] == str(chunk.sft / "checkpoint_fidelity.json")
    assert env["CLASP_MIS_JIANG_RESPONSE_OUTPUT"] == "0"
    assert env["GRPO_STOP_AT_GLOBAL_STEP"] == str(stop)
    assert env["GRPO_STRICT_OPTIMIZER_RESUME"] == "1"
    assert env["GRPO_VERIFIED_IMPROVEMENT"] == "0"
    assert env["CLASP_GRADIENT_CONSISTENCY_CHECKS"] == str(int(diagnostic))
    assert "RESUME_FROM_CHECKPOINT" not in env
    assert "FOARL_UNDECLARED_CONTROL" not in env and "PD_UNDECLARED_CONTROL" not in env
    assert chunk.source_checks == [(chunk.source / "SOURCE_MANIFEST.sha256", "a" * 64)]
    assert (chunk.output / f"chunk-{stop}-complete.json").is_file()
    assert sum("probe_foarl_teacher_tokenizer.py" in " ".join(c["argv"]) for c in chunk.commands) == int(diagnostic)


def test_same_arm_resume_retains_horizon_data_and_original_sft_reference(chunk):
    chunk.module.main()
    contract = (chunk.output / "local_development_run_contract.json").read_bytes()
    checkpoint = chunk.output / "checkpoints/checkpoint-100"
    chunk.commands.clear()
    os.environ.update(FT_STOP="200", FT_RESUME=str(checkpoint))
    chunk.module.main()
    train = chunk.commands[0]
    assert train["environment"]["RESUME_FROM_CHECKPOINT"] == str(checkpoint)
    assert train["environment"]["CLASP_KL_REFERENCE_MODEL_PATH"] == str(chunk.sft)
    assert train["argv"][train["argv"].index("--sft_checkpoint_path") + 1] == str(chunk.sft)
    assert (chunk.output / "local_development_run_contract.json").read_bytes() == contract
    assert chunk.validations[-2][0] == checkpoint
    assert chunk.validations[-1][1]["expected_step"] == 200
    # A second attempt at the same admitted boundary cannot rerun training.
    before = len(chunk.commands)
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(chunk.commands) == before


@pytest.mark.parametrize("change", ["other_arm", "wrong_step", "method", "data"])
def test_resume_rejects_cross_arm_nonadjacent_or_changed_contract_before_training(chunk, change):
    chunk.module.main()
    checkpoint = chunk.output / "checkpoints/checkpoint-100"
    if change == "other_arm":
        checkpoint = chunk.output.parent / "other/checkpoints/checkpoint-100"
        checkpoint.mkdir(parents=True)
    elif change == "wrong_step":
        checkpoint = chunk.output / "checkpoints/checkpoint-5"
        checkpoint.mkdir()
    elif change == "method":
        os.environ["FT_METHOD"] = "foarl_control"
    else:
        (chunk.data / "mis/train/train.json").write_text('[{"changed":true}]\n')
    os.environ.update(FT_STOP="200", FT_RESUME=str(checkpoint))
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
    with pytest.raises(FileExistsError):
        chunk.module.main()
    assert len(calls) == 1


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "run_foarl_teacher_dev_chunk", load_script("run_foarl_teacher_dev_chunk"))
    module = load_script("foarl_teacher_diagnostics")
    root = tmp_path / "diagnostic"
    root.mkdir()
    calls = []
    monkeypatch.setattr(module, "active_jobs", lambda: [str(i) for i in range(1, 10)])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "")

    def submit(argv, **kwargs):
        assert (root / "new-attempt.json").is_file()
        # A genuinely separate descriptor cannot take the admission lock.
        with (root / ".submission.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", submit)
    return SimpleNamespace(module=module, root=root, calls=calls)


def test_diagnostic_submission_is_locked_durable_and_once(diagnostic):
    diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    assert json.loads((diagnostic.root / "new-submission.json").read_text())["stdout"] == "12345\n"
    with pytest.raises((FileExistsError, RuntimeError)):
        diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    assert len(diagnostic.calls) == 1


@pytest.mark.parametrize("reservation", ["invisible", "unresolved", "ambiguous", "queued"])
def test_global_cap_includes_prior_invisible_or_uncertain_attempts(diagnostic, monkeypatch, reservation):
    if reservation == "queued":
        monkeypatch.setattr(diagnostic.module, "active_jobs", lambda: [str(i) for i in range(1, 11)])
    else:
        diagnostic.module.write_once(diagnostic.root / "old-attempt.json", {"command": ["sbatch"]})
        if reservation != "unresolved":
            diagnostic.module.write_once(diagnostic.root / "old-submission.json", {
                "returncode": 1 if reservation == "ambiguous" else 0,
                "stdout": "99999\n", "stderr": ""})
    with pytest.raises(RuntimeError, match="slot"):
        diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    assert not (diagnostic.root / "new-attempt.json").exists()
    assert diagnostic.calls == []


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_conclusive_terminal_old_job_releases_only_its_reservation(diagnostic, monkeypatch, terminal):
    diagnostic.module.write_once(diagnostic.root / "old-attempt.json", {"command": ["sbatch"]})
    diagnostic.module.write_once(diagnostic.root / "old-submission.json", {
        "returncode": 0, "stdout": "99999\n", "stderr": ""})
    monkeypatch.setattr(diagnostic.module.subprocess, "check_output", lambda *a, **kw: f"99999|{terminal}\n")
    diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    assert len(diagnostic.calls) == 1


def test_sbatch_ambiguity_keeps_attempt_and_does_not_retry(diagnostic, monkeypatch):
    calls = []
    def fail(argv, **kwargs):
        assert (diagnostic.root / "new-attempt.json").exists()
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="not one job ID", stderr="")
    monkeypatch.setattr(diagnostic.module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    with pytest.raises((FileExistsError, RuntimeError)):
        diagnostic.module.submit(diagnostic.root, "new", ["sbatch", "--parsable"])
    assert len(calls) == 1
    assert (diagnostic.root / "new-submission.json").exists()


def test_diagnostic_command_is_argv_bound_no_requeue_and_rejects_export_injection(diagnostic):
    argv = diagnostic.module.command(Path("/source"), "a" * 64, diagnostic.root,
                                     "mis_teacher_staged", "mis", "foarl_teacher", 10,
                                     diagnostic.root / "mis_teacher_staged/checkpoints/checkpoint-5")
    assert "--no-requeue" in argv and "--time=02:00:00" in argv
    env = dict(piece.split("=", 1) for piece in next(x for x in argv if x.startswith("--export=")).split(",")[1:])
    assert env["FT_SOURCE_SHA"] == "a" * 64
    assert env["FT_STOP"] == "10" and env["FT_METHOD"] == "foarl_teacher"
    assert env["FT_COMPARE_WITH"] == str(diagnostic.root / "mis_teacher_full/checkpoints/checkpoint-10")
    with pytest.raises(ValueError, match="export"):
        diagnostic.module.command(Path("/source,FT_METHOD=other"), "a" * 64,
                                  diagnostic.root, "x", "mis", "foarl_teacher", 5)


def test_resume_plan_tampering_is_rejected_before_any_scheduler_call(diagnostic, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    diagnostic.module.write_once(diagnostic.root / "protocol.json", {
        "source": str(source), "source_manifest_sha256": "a" * 64,
        "resume_plan": ["sbatch", "different_unreviewed_script"]})
    monkeypatch.setattr(source_guard, "verify_dev_source_manifest", lambda *a, **kw: {})
    # Map only the native project root used by the placement guard to this
    # synthetic filesystem. No native paths or scheduler commands are touched.
    monkeypatch.setattr(diagnostic.module, "Path", lambda p: tmp_path if str(p) ==
                        "/lustre/fsn1/projects/rech/vas/uag72kz" else Path(p))
    monkeypatch.setattr(sys, "argv", ["diagnostic", "resume", "--source", str(source),
                                     "--source-sha", "a" * 64, "--output", str(diagnostic.root)])
    with pytest.raises(ValueError, match="Resume plan"):
        diagnostic.module.main()
    assert diagnostic.calls == []
