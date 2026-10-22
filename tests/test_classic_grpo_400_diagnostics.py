"""No scheduler/model calls: exact plans, source admission and one-attempt behavior."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("classic_diagnostics", REPO / "scripts/classic_grpo_400_diagnostics.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
SHA = "a" * 64


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    DIAG.write_once(path, value)


def exported(argv):
    value = next(item.removeprefix("--export=ALL,") for item in argv if item.startswith("--export=ALL,"))
    return dict(item.split("=", 1) for item in value.split(","))


def test_proven_submission_primitives_are_ast_identical():
    def node(path, name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    current = REPO / "scripts/classic_grpo_400_diagnostics.py"
    for name in ("active_jobs", "submit"):
        assert node(current, name) == node(REPO / "scripts/foarl_teacher_diagnostics.py", name)
    assert node(current, "write_once") == node(REPO / "scripts/run_foarl_teacher_dev_chunk.py", "write_once")


def test_exact_two_initial_plans_and_one_continuation(tmp_path):
    source, root = tmp_path / "source", tmp_path / "output"
    plans = DIAG.declared_plans(source, SHA, root)
    assert [p["key"] for p in plans] == ["mis_classic_full", "mis_classic_staged"]
    for plan, stop in zip(plans, (10, 5), strict=True):
        argv = plan["command"]
        assert argv[:4] == ["sbatch", "--parsable", "--no-requeue", "--time=02:00:00"]
        assert argv[-1] == str(source / "scripts/train_classic_grpo_400_dev_a100.slurm")
        assert exported(argv) == {"CG_SOURCE": str(source), "CG_SOURCE_SHA": SHA, "CG_TASK": "mis",
                                  "CG_OUTPUT": str(root / plan["key"]), "CG_STOP": str(stop),
                                  "CG_DIAGNOSTIC": "1", "CG_RESUME": "", "CG_COMPARE_WITH": ""}
    values = exported(DIAG.declared_resume(source, SHA, root))
    assert values["CG_RESUME"] == str(root / "mis_classic_staged/checkpoints/checkpoint-5")
    assert values["CG_COMPARE_WITH"] == str(root / "mis_classic_full/checkpoints/checkpoint-10")
    assert values["CG_STOP"] == "10"
    slurm = (REPO / "scripts/train_classic_grpo_400_dev_a100.slurm").read_text()
    for directive in ("--account=vas@a100", "--partition=gpu_p5", "--constraint=a100", "--gres=gpu:4",
                      "--qos=qos_gpu_a100-dev", "--time=02:00:00", "--no-requeue"):
        assert "#SBATCH " + directive in slurm


@pytest.mark.parametrize("bad", ["with space", "comma,path", "with\nnewline"])
def test_unsupported_export_paths_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="export"):
        DIAG.command(tmp_path / bad, SHA, tmp_path, "mis_classic_full", 10)


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    source = tmp_path / "source"
    entries = {}
    for name in DIAG.REQUIRED:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen file\n")
        entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    helper = source / "scripts/classic_grpo_400_diagnostics.py"
    helper.write_bytes((REPO / "scripts/classic_grpo_400_diagnostics.py").read_bytes())
    entries[str(helper.relative_to(source))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(f"{sha}  {name}\n" for name, sha in sorted(entries.items())))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(DIAG, "__file__", str(helper))
    return source, sha


def test_source_full_inventory_verified_before_project_import(source_tree):
    source, sha = source_tree
    assert DIAG.verify_source(source, sha) == source


@pytest.mark.parametrize("mutation", ["digest", "file", "extra_python", "source_alias", "manifest_alias", "omission", "duplicate"])
def test_source_mutations_rejected(source_tree, tmp_path, monkeypatch, mutation):
    source, sha = source_tree
    manifest = source / "SOURCE_MANIFEST.sha256"
    if mutation == "digest":
        sha = "f" * 64
    elif mutation == "file":
        (source / "src/clasp/models/classic_grpo_resume.py").write_text("raise AssertionError('must never import')\n")
    elif mutation == "extra_python":
        (source / "src/clasp/new.py").write_text("raise AssertionError('must never import')\n")
    elif mutation == "source_alias":
        path = source / "src/clasp/models/classic_grpo_resume.py"
        payload = path.read_bytes()
        path.unlink()
        target = tmp_path / "outside.py"
        target.write_bytes(payload)
        path.symlink_to(target)
    elif mutation == "manifest_alias":
        payload = manifest.read_bytes()
        manifest.unlink()
        target = tmp_path / "manifest"
        target.write_bytes(payload)
        manifest.symlink_to(target)
    else:
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(lines[1:] if mutation == "omission" else lines + lines[:1]) + "\n")
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        DIAG.verify_source(source, sha)


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    root = tmp_path / "diagnostics"
    root.mkdir()
    state = {"queue": [], "accounting": {}, "calls": [], "returncode": 0, "stdout": "101\n"}
    monkeypatch.setattr(DIAG, "active_jobs", lambda: state["queue"])

    def read(argv, **kwargs):
        assert argv[0] == "sacct"
        return state["accounting"].get(argv[argv.index("-j") + 1], "")

    def run(argv, **kwargs):
        # Every sbatch runs under a real held flock and after a durable intent.
        with (root / ".submission.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert list(root.glob("*-attempt.json"))
        state["calls"].append(argv)
        if isinstance(state["stdout"], Exception):
            raise state["stdout"]
        return SimpleNamespace(returncode=state["returncode"], stdout=state["stdout"], stderr="")

    monkeypatch.setattr(DIAG.subprocess, "check_output", read)
    monkeypatch.setattr(DIAG.subprocess, "run", run)
    return root, state


def test_one_attempt_lock_and_duplicate_rejection(scheduler):
    root, state = scheduler
    DIAG.submit(root, "first", ["sbatch", "synthetic"])
    with pytest.raises(FileExistsError):
        DIAG.submit(root, "first", ["sbatch", "synthetic"])
    assert len(state["calls"]) == 1
    assert (root / "first-attempt.json").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("failure", ["error", "ambiguous", "transport"])
def test_submission_failure_never_retries(scheduler, failure):
    root, state = scheduler
    if failure == "error":
        state["returncode"] = 1
    elif failure == "ambiguous":
        state["stdout"] = "unrecognized response\n"
    else:
        state["stdout"] = subprocess.TimeoutExpired(["sbatch"], 1)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        DIAG.submit(root, "one", ["sbatch"])
    with pytest.raises(FileExistsError):
        DIAG.submit(root, "one", ["sbatch"])
    assert len(state["calls"]) == 1
    assert (root / "one-attempt.json").is_file()


@pytest.mark.parametrize("previous", ["missing", "invisible", "failed_receipt", "unknown_state"])
def test_invisible_and_uncertain_attempts_reserve_slots(scheduler, previous):
    root, state = scheduler
    state["queue"] = [str(i) for i in range(10, 19)]
    save(root / "prior-attempt.json", {"command": ["sbatch"]})
    if previous != "missing":
        save(root / "prior-submission.json", {"returncode": int(previous == "failed_receipt"), "stdout": "200\n", "stderr": ""})
    state["accounting"]["200"] = "200|RUNNING\n" if previous == "invisible" else ""
    with pytest.raises(RuntimeError, match="slot"):
        DIAG.submit(root, "next", ["sbatch"])
    assert state["calls"] == []
    assert not (root / "next-attempt.json").exists()


def test_terminal_known_previous_does_not_reserve_slot(scheduler):
    root, state = scheduler
    state["queue"] = [str(i) for i in range(10, 19)]
    save(root / "prior-attempt.json", {"command": ["sbatch"]})
    save(root / "prior-submission.json", {"returncode": 0, "stdout": "200\n", "stderr": ""})
    state["accounting"]["200"] = "200|COMPLETED\n"
    DIAG.submit(root, "next", ["sbatch"])
    assert len(state["calls"]) == 1


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root, source = tmp_path / "run", tmp_path / "source"
    root.mkdir()
    protocol = {"protocol": DIAG.PROTOCOL, "scope": "correctness_only_no_performance_claim", "source": str(source),
                "source_manifest_sha256": SHA, "plans": DIAG.declared_plans(source, SHA, root),
                "resume_plan": DIAG.declared_resume(source, SHA, root), "fixed_horizon": 400,
                "native_gpus_per_job": 4, "walltime": "02:00:00", "no_automatic_retry": True,
                "cross_campaign_submission_authority": "one_sequential_operator"}
    save(root / "protocol.json", protocol)
    contract = {"source": {"root": str(source), "manifest_sha256": SHA}, "method": "classic_grpo", "diagnostic": True,
                "fixed_horizon": {"max_steps": 400, "world_size": 4}, "beta": 0, "teacher_weight": 0}
    inspections, accounting = [], {}
    for (name, step), job in zip(DIAG.CASES, ("101", "102"), strict=True):
        save(root / f"{name}-attempt.json", {"command": DIAG.command(source, SHA, root, name, step)})
        save(root / f"{name}-submission.json", {"returncode": 0, "stdout": job + "\n", "stderr": ""})
        save(root / name / "local_development_run_contract.json", contract)
        save(root / name / f"chunk-{step}-admission.json", {"job_id": job, "stop": step})
        save(root / name / f"chunk-{step}-exit.json", {"returncode": 0})
        save(root / name / f"chunk-{step}-complete.json", {"synthetic_verified_step": step})
        accounting[job] = job + "|COMPLETED|0:0\n"

    def inspect(checkpoint, step, contract, contract_path, source):
        inspections.append((checkpoint, step, contract, contract_path, source))
        return {"synthetic_verified_step": step}

    monkeypatch.setattr(DIAG, "validate_checkpoint", inspect)
    monkeypatch.setattr(DIAG.subprocess, "check_output", lambda argv, **kwargs: accounting[argv[argv.index("-j") + 1]])
    return root, source, accounting, inspections


def replace_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.chmod(0o640)
    path.write_text(json.dumps(value))
    path.chmod(0o440)


def test_both_checkpoint_receipts_replayed_before_resume(completed):
    root, source, accounting, inspections = completed
    assert DIAG.admit_resume(root, source, SHA) == DIAG.declared_resume(source, SHA, root)
    assert [row[1] for row in inspections] == [10, 5]
    assert inspections[0][2] == inspections[1][2]


@pytest.mark.parametrize("failure", ["full_running", "staged_failed", "full_missing_complete", "staged_missing_complete",
                                      "changed_source", "changed_plan", "changed_contract", "changed_completion",
                                      "wrong_job", "wrong_boundary", "failed_subprocess", "duplicate_job", "changed_attempt"])
def test_resume_admission_rejects_incomplete_or_changed_inputs(completed, failure):
    root, source, accounting, inspections = completed
    full = root / "mis_classic_full"
    staged = root / "mis_classic_staged"
    if failure == "full_running":
        accounting["101"] = "101|RUNNING|0:0\n"
    elif failure == "staged_failed":
        accounting["102"] = "102|FAILED|1:0\n"
    elif failure == "full_missing_complete":
        (full / "chunk-10-complete.json").unlink()
    elif failure == "staged_missing_complete":
        (staged / "chunk-5-complete.json").unlink()
    elif failure == "changed_source":
        replace_json(root / "protocol.json", lambda v: v.update(source_manifest_sha256="b" * 64))
    elif failure == "changed_plan":
        replace_json(root / "protocol.json", lambda v: v["resume_plan"].append("--undeclared"))
    elif failure == "changed_contract":
        replace_json(staged / "local_development_run_contract.json", lambda v: v.update(beta=.05))
    elif failure == "changed_completion":
        replace_json(full / "chunk-10-complete.json", lambda v: v.update(synthetic_verified_step=5))
    elif failure == "wrong_job":
        replace_json(staged / "chunk-5-admission.json", lambda v: v.update(job_id="999"))
    elif failure == "wrong_boundary":
        replace_json(staged / "chunk-5-admission.json", lambda v: v.update(stop=10))
    elif failure == "failed_subprocess":
        replace_json(staged / "chunk-5-exit.json", lambda v: v.update(returncode=1))
    elif failure == "duplicate_job":
        replace_json(root / "mis_classic_staged-submission.json", lambda v: v.update(stdout="101\n"))
    else:
        replace_json(root / "mis_classic_full-attempt.json", lambda v: v["command"].append("--undeclared"))
    with pytest.raises(ValueError):
        DIAG.admit_resume(root, source, SHA)


def test_init_creates_only_two_plans_and_does_not_repeat(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "new-diagnostics"
    calls = []
    monkeypatch.setattr(DIAG, "PROJECT", tmp_path)
    monkeypatch.setattr(DIAG, "verify_source", lambda *args: source)
    monkeypatch.setattr(DIAG, "active_jobs", lambda: [str(i) for i in range(8)])
    monkeypatch.setattr(DIAG, "submit", lambda *args: calls.append(args))
    argv = ["init", "--source", str(source), "--source-sha", SHA, "--output", str(root)]
    DIAG.main(argv)
    assert len(calls) == 2
    protocol = json.loads((root / "protocol.json").read_text())
    DIAG.validate_protocol(protocol, source, SHA, root)
    assert protocol["fixed_horizon"] == 400
    with pytest.raises(FileExistsError):
        DIAG.main(argv)
    assert len(calls) == 2


def test_init_quota_and_wrong_output_fail_before_creation(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "new-diagnostics"
    monkeypatch.setattr(DIAG, "PROJECT", tmp_path)
    monkeypatch.setattr(DIAG, "verify_source", lambda *args: source)
    monkeypatch.setattr(DIAG, "active_jobs", lambda: [str(i) for i in range(9)])
    argv = ["init", "--source", str(source), "--source-sha", SHA, "--output", str(root)]
    with pytest.raises(RuntimeError, match="Two diagnostic"):
        DIAG.main(argv)
    assert not root.exists()
    with pytest.raises(ValueError, match="direct unlinked child"):
        DIAG.main(argv[:-1] + [str(root / "nested")])


def test_resume_main_submits_only_declared_continuation(completed, monkeypatch):
    root, source, accounting, inspections = completed
    calls = []
    monkeypatch.setattr(DIAG, "PROJECT", root.parent)
    monkeypatch.setattr(DIAG, "verify_source", lambda *args: source)
    monkeypatch.setattr(DIAG, "submit", lambda *args: calls.append(args))
    DIAG.main(["resume", "--source", str(source), "--source-sha", SHA, "--output", str(root)])
    assert calls == [(root, "mis_classic_staged-resume10", DIAG.declared_resume(source, SHA, root))]
    assert [row[1] for row in inspections] == [10, 5]
