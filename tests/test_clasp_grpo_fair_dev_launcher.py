"""Submission plumbing tests: no Slurm or GPUs are used."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fair_artifacts", ROOT / "scripts/clasp_grpo_fair_dev_artifacts.py"
)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def mock_environment(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/resume_clasp_grpo_fair_dev.py", source / "scripts/resume_clasp_grpo_fair_dev.py")
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("mock-preflight-source")
    runtime = tmp_path / "runtime/bin"
    runtime.mkdir(parents=True)
    python = runtime / "python"
    python.write_text("#!/bin/bash\n"
                      "if [[ \"$1\" == */clasp_grpo_fair_dev_artifacts.py ]]; then\n"
                      "  cat \"${MOCK_PROTOCOL}\"\n"
                      "else\n"
                      f"  exec {sys.executable} \"$@\"\n"
                      "fi\n")
    python.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "p=pathlib.Path(os.environ['MOCK_SUBMISSIONS'])\n"
        "rows=p.read_text().splitlines() if p.exists() else []\n"
        "if os.environ.get('MOCK_FAIL_AT')==str(len(rows)+1): sys.exit(1)\n"
        "with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(1001+len(rows))\n"
    )
    sbatch.chmod(0o755)
    for name, body in {
        "sacct": "print('')",
        "squeue": "print('\\n'.join(f'879217_{i}|PENDING|qos_gpu_a100-dev' for i in range(int(os.environ.get('MOCK_OCCUPIED', '0')))))",
    }.items():
        path = bindir / name
        path.write_text(f"#!{sys.executable}\nimport os\n{body}\n")
        path.chmod(0o755)
    for name in ("data", "sft"):
        (tmp_path / name).mkdir()
    protocol = {
        "source_manifest": artifacts.identity(manifest), "created_at_utc": "2026-09-05T17:00:00Z",
        "runtime": str(runtime.parent), "data_root": str(tmp_path / "data"),
        "training": {"seed": 43, "steps": 100},
        "initializations": {task: {"adapter_model.safetensors": {"path": f"/sft/{task}/checkpoint-31250/adapter_model.safetensors"}} for task in ("cvrp", "mis")},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return dict(os.environ, FAIR_SOURCE_SNAPSHOT=str(source), FAIR_OUTPUT_ROOT=str(tmp_path / "output"),
                FAIR_RUNTIME_ENV=str(runtime.parent), FAIR_DATA_ROOT=str(tmp_path / "data"),
                FAIR_SFT_ROOT=str(tmp_path / "sft"), MOCK_SUBMISSIONS=str(tmp_path / "calls.jsonl"), MOCK_PROTOCOL=str(protocol_path),
                PATH=str(bindir) + os.pathsep + os.environ["PATH"], DEV_RESUME_CHECKPOINT="bad-inherited-resume",
                DEV_GROUP_STD_CORRECTION="0", DEV_ROLLOUT_BATCH_SIZE="18", MAX_PROMPT_LENGTH="0")


def submit(env):
    return subprocess.run(["bash", str(ROOT / "scripts/submit_clasp_grpo_fair_dev_a100.sh")],
                          env=env, text=True, capture_output=True)


def test_initial_submission_fills_available_slots_without_dependent_evals(tmp_path):
    env = mock_environment(tmp_path)
    env["MOCK_OCCUPIED"] = "7"
    result = submit(env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 3
    for train in calls:
        exported = next(value for value in train if value.startswith("--export="))
        assert "DEV_SEED=43" in exported
        assert "DEV_MAX_STEPS=100" in exported
        assert "DEV_GROUP_STD_CORRECTION=1" in exported
        values = dict(part.split("=", 1) for part in exported.split(",")[1:])
        assert values["DEV_ROLLOUT_BATCH_SIZE"] == ""
        assert "DEV_RESUME_CHECKPOINT=," in exported
        assert "checkpoint-31250" in exported
        assert not any(value.startswith("--dependency=") for value in train)
        assert train[-1].endswith("train_clasp_canonical_checkpoint_dev_a100.slurm")
    ledger = tmp_path / "output/ledgers/submission.tsv"
    assert len(ledger.read_text().splitlines()) == 1
    assert ledger.stat().st_mode & 0o222 == 0
    assert submit(env).returncode != 0
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 3
    events = [json.loads(path.read_text()) for path in (ledger.parent / "orchestration_events").glob("*.json")]
    assert sum(item["kind"] == "submitted" for item in events) == 3
    assert "FAIR_DEV_RESUME_COMMAND=" in result.stdout


def test_initial_submission_uses_all_six_training_slots_when_queue_empty(tmp_path):
    env = mock_environment(tmp_path)
    assert submit(env).returncode == 0
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 6
    assert all("--dependency=" not in " ".join(command) for command in calls)


def test_initial_submission_at_capacity_creates_resumable_protocol_only(tmp_path):
    env = mock_environment(tmp_path)
    env["MOCK_OCCUPIED"] = "10"
    result = submit(env)
    assert result.returncode == 0
    assert not (tmp_path / "calls.jsonl").exists()
    assert (tmp_path / "output/ledgers/protocol.json").is_file()
    assert "FAIR_DEV_RESUME_COMMAND=" in result.stdout


def test_preflight_only_does_not_submit_or_create_output(tmp_path):
    env = mock_environment(tmp_path)
    env["FAIR_VALIDATE_ONLY"] = "1"
    assert submit(env).returncode == 0
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "calls.jsonl").exists()


def test_partial_submission_keeps_immutable_ledger(tmp_path):
    env = mock_environment(tmp_path)
    env["MOCK_FAIL_AT"] = "4"
    result = submit(env)
    assert result.returncode == 0
    assert "submission_rejected" in result.stdout
    ledger = tmp_path / "output/ledgers/submission.tsv"
    assert len(ledger.read_text().splitlines()) == 1
    assert ledger.stat().st_mode & 0o222 == 0
    events = [json.loads(path.read_text()) for path in (ledger.parent / "orchestration_events").glob("*.json")]
    assert sum(item["kind"] == "submitted" for item in events) == 3
    assert sum(item["kind"] == "rejected" for item in events) == 1


def test_source_manifest_detects_tampering(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    names = ["train_clasp_canonical_checkpoint_dev_a100.slurm", "submit_clasp_grpo_fair_dev_a100.sh",
             "evaluate_clasp_grpo_fair_dev_a100.slurm", "clasp_grpo_fair_dev_artifacts.py",
             "evaluate_a100.slurm", "compare_development.py", "resume_clasp_grpo_fair_dev.py"]
    entries = []
    for name in names:
        path = scripts / name
        path.write_text("clasp_grpo_anchor_v4)\n")
        entries.append(artifacts.identity(path)["sha256"] + "  scripts/" + name)
    (tmp_path / "SOURCE_MANIFEST.sha256").write_text("\n".join(entries) + "\n")
    assert artifacts.validate_source(tmp_path)["sha256"]
    (scripts / names[0]).write_text("tampered")
    with pytest.raises(SystemExit, match="source hash mismatch"):
        artifacts.validate_source(tmp_path)


def test_fresh_training_contract_and_resumption_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "validate_source", lambda _: {"sha256": "manifest"})
    train = tmp_path / "train"
    checkpoint = train / "checkpoints/checkpoint-100"
    checkpoint.mkdir(parents=True)
    contract = {
        "task": "cvrp", "method": "clasp_grpo_anchor_v4", "seed": 43,
        "fixed_horizon": {"max_steps": 100, "world_size": 4},
        "continuation": {"resume_checkpoint": None, "resume_global_step": 0, "strict_optimizer_resume": False},
        "initialization": {"adapter_model_sha256": artifacts.SFT_HASHES["cvrp"]},
        "source": {"root": str(tmp_path), "manifest_sha256": "manifest"},
        "production_preserved": {"rollout_batch_b": 9, "global_generation_batch_size": 36, "learning_rate": 1e-6,
                                 "max_completion_length": 3000, "max_prompt_length": 29768,
                                 "per_device_train_batch_size": 3, "gradient_accumulation_steps": 3,
                                 "beta": 0.0, "deployment_set_k": 8, "prompt_style": "jiang_f638", "no_reason": True},
        "policy_estimator": {"group_std_correction": 1, "importance_sampling_level": "token", "policy_length_alpha": 0.0,
                             "canonical_set": True, "use_clasp": True, "canonical_grpo_anchor_scale": 1.0,
                             "canonical_quality_anchor_ratio": 0.5, "canonical_dense_feasibility_scale": 0.5},
    }
    path = train / "local_development_run_contract.json"
    path.write_text(json.dumps(contract))
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
        (checkpoint / name).write_text("placeholder")
    (checkpoint / "trainer_state.json").write_text('{"global_step":100}')
    assert artifacts.validate_training(train, "cvrp", "clasp_grpo_anchor_v4", tmp_path)["status"] == "training_validated"
    contract["continuation"]["resume_global_step"] = 50
    path.write_text(json.dumps(contract))
    with pytest.raises(SystemExit, match="fresh SFT"):
        artifacts.validate_training(train, "cvrp", "clasp_grpo_anchor_v4", tmp_path)
