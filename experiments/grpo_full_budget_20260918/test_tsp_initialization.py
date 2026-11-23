"""The TSP gate waits for existing jobs and rejects contradictory provenance."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tsp_initialization as module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run/checkpoints/checkpoint-31250"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(module, "EXPECTED_CHECKPOINT", checkpoint)
    root = tmp_path / "source"
    names = ("src/clasp/entrypoints/train_sft_jiang_f638.py",
             "src/clasp/entrypoints/train_sft_jiang_f638_tsp_stable.py",
             "scripts/train_tsp_sft_stability_h100.slurm",
             "scripts/audit_sft_reload_equivalence.py")
    lines = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + name)
    manifest = root / "SOURCE_MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(module, "SOURCE_SHA256", module.identity(manifest)["sha256"])
    hyper = checkpoint.parent.parent / "hyperparams.json"
    hyper.write_text(json.dumps({"tasks": ["tsp"], "stage": "sft",
        "output_dir": str(checkpoint.parent.parent), "model_path": str(tmp_path / "base"),
        "stage_hyperparameters": {"learning_rate": 0.0001,
         "expected_terminal_global_step": 31250, "effective_train_batch_size": 16}}))
    state = {"global_step": 31250, "max_steps": 31250, "epoch": 1.0}
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json",
                 "sft_reload_equivalence.json"):
        (checkpoint / name).write_text("certificate fixture")
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    rows = {job: {"job_id": job, "state": "COMPLETED", "exit_code": "0:0",
                   "job_name": module.JOB_NAMES[job], "account": "vas@h100", "partition": "gpu_p6"}
            for job in module.JOBS}
    monkeypatch.setattr(module, "_accounting", lambda: rows)
    monkeypatch.setattr(module, "_verify_certificates", lambda checkpoint, base: ("fidelity", {"status": "verified"}))
    pending = {"checkpoint": str(checkpoint), "jobs": list(module.JOBS),
               "source": {"root": str(root), "manifest_sha256": module.SOURCE_SHA256},
               "hyperparameters": module.identity(hyper)}
    return SimpleNamespace(checkpoint=checkpoint, root=root, rows=rows, hyper=hyper,
                           declaration={"tasks": {"tsp": {"initialization_pending": pending}}})


def test_completed_existing_chain_returns_bound_initializer_without_writes(fixture):
    before = {p: module.identity(p) for p in fixture.checkpoint.iterdir()}
    result = module.gate(fixture.declaration)
    assert result["checkpoint"] == str(fixture.checkpoint)
    assert result["sft_reload_equivalence"] == {"status": "verified"}
    assert len(result["completed_sft_jobs"]) == 4
    assert {p: module.identity(p) for p in fixture.checkpoint.iterdir()} == before


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "COMPLETING", "REQUEUED"])
def test_pending_original_chain_waits(fixture, state):
    fixture.rows[module.JOBS[0]]["state"] = state
    assert module.gate(fixture.declaration) is None


def test_missing_accounting_or_terminal_artifact_waits(fixture):
    row = fixture.rows.pop(module.JOBS[0])
    assert module.gate(fixture.declaration) is None
    fixture.rows[module.JOBS[0]] = row
    (fixture.checkpoint / "sft_reload_equivalence.json").unlink()
    assert module.gate(fixture.declaration) is None


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED", "0:0"),
                                      ("TIMEOUT", "0:0"), ("COMPLETED", "1:0")])
def test_terminal_failure_refuses_no_retry(fixture, state, code):
    fixture.rows[module.JOBS[0]].update(state=state, exit_code=code)
    with pytest.raises(ValueError, match="TSP SFT"):
        module.gate(fixture.declaration)


def test_modified_hyperparameters_refused_even_while_pending(fixture):
    fixture.rows[module.JOBS[0]]["state"] = "PENDING"
    fixture.hyper.write_text("{}")
    with pytest.raises(ValueError, match="Input binding changed"):
        module.gate(fixture.declaration)


def test_modified_source_refused(fixture):
    (fixture.root / "scripts/train_tsp_sft_stability_h100.slurm").write_text("changed")
    with pytest.raises(ValueError, match="source bytes changed"):
        module.gate(fixture.declaration)


def test_changed_terminal_step_refused(fixture):
    (fixture.checkpoint / "trainer_state.json").write_text(json.dumps(
        {"global_step": 15000, "max_steps": 31250, "epoch": 0.48}))
    with pytest.raises(ValueError, match="terminal trainer state"):
        module.gate(fixture.declaration)


def test_certificate_verifier_failure_is_not_treated_as_waiting(fixture, monkeypatch):
    def invalid(*args):
        raise ValueError("Reload identity changed")
    monkeypatch.setattr(module, "_verify_certificates", invalid)
    with pytest.raises(ValueError, match="Reload identity changed"):
        module.gate(fixture.declaration)


def test_unknown_job_or_wrong_checkpoint_refused(fixture):
    pending = fixture.declaration["tasks"]["tsp"]["initialization_pending"]
    pending["jobs"][0] = "99"
    with pytest.raises(ValueError, match="job graph"):
        module.gate(fixture.declaration)
    pending["jobs"] = list(module.JOBS)
    pending["checkpoint"] = str(fixture.checkpoint.parent / "checkpoint-15000")
    with pytest.raises(ValueError, match="repaired terminal"):
        module.gate(fixture.declaration)


def test_accounting_parser_requires_exact_known_identity(monkeypatch):
    def run(command, **kwargs):
        assert command[0] == "sacct" and kwargs["timeout"] == 45
        return SimpleNamespace(stdout="1703800|COMPLETED|0:0|wrong-name|vas@h100|gpu_p6|\n")
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError, match="scheduler identity"):
        module._accounting()
