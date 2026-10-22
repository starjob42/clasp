import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fair_resume", ROOT / "scripts/resume_clasp_grpo_fair_dev.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def setup_run(tmp_path):
    output = tmp_path / "run"
    (output / "ledgers").mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("frozen-source")
    protocol = {
        "source_manifest": {"path": str(manifest), "sha256": resume.hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "created_at_utc": "2026-09-05T17:00:00Z", "runtime": "/runtime", "data_root": "/data",
        "training": {"seed": 43, "steps": 100},
        "initializations": {task: {"adapter_model.safetensors": {"path": f"/sft/{task}/checkpoint-31250/adapter_model.safetensors"}} for task in resume.TASKS},
    }
    (output / "ledgers/protocol.json").write_text(json.dumps(protocol))
    (output / "ledgers/submission.tsv").write_text("role\ttask\tmethod\tjob_id\tdependency\tcheckpoint_or_output\n")
    return output


class Scheduler:
    def __init__(self, output, occupied=0, fail=False):
        self.output, self.occupied, self.fail = output, occupied, fail
        self.accounting, self.commands = [], []

    def add(self, job, role, task, method, state):
        stdout = resume.paths(self.output, role, task, method)[2]
        self.accounting.append(f"{job}|{state}|job-name|{stdout}")

    def __call__(self, command):
        if command[0] == "sacct":
            return "\n".join(self.accounting)
        if command[0] == "squeue":
            assert "-r" in command
            return "\n".join(f"879217_{index}|PENDING|{resume.QOS}" for index in range(self.occupied))
        assert command[0] == "sbatch"
        self.commands.append(command)
        if self.fail:
            raise subprocess.CalledProcessError(1, command, stderr="QOSMaxSubmitJobPerUserLimit")
        return str(2000 + len(self.commands))


def test_full_qos_counts_array_tasks_and_submits_nothing(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output, occupied=10)
    result = resume.reconcile(output, "user", submit=True, runner=scheduler)
    assert result["occupied_slots"] == 10
    assert not scheduler.commands
    assert not (output / "ledgers/orchestration_events").exists()


def test_manual_jobs_discovered_and_only_missing_train_submitted(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output, occupied=9)
    index = 1000
    for task in resume.TASKS:
        for method in resume.METHODS:
            if task == "mis" and method == "clasp_grpo_anchor_v3":
                continue
            scheduler.add(index, "train", task, method, "PENDING")
            index += 1
    result = resume.reconcile(output, "user", submit=True, runner=scheduler)
    assert len(result["submitted"]) == 1
    assert result["submitted"][0]["key"] == ("train", "mis", "clasp_grpo_anchor_v3")
    assert "DEV_GROUP_STD_CORRECTION=1" in " ".join(scheduler.commands[0])
    # Accepted-job event prevents duplication even while sacct has not updated.
    resume.reconcile(output, "user", submit=True, runner=scheduler)
    assert len(scheduler.commands) == 1
    assert all(path.stat().st_mode & 0o222 == 0 for path in (output / "ledgers/orchestration_events").glob("*.json"))


def test_cancelled_eval_recreated_only_after_training_completes(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output)
    for index, (task, method) in enumerate((task, method) for task in resume.TASKS for method in resume.METHODS):
        scheduler.add(1000 + index, "train", task, method, "COMPLETED")
        checkpoint = resume.paths(output, "train", task, method)[0] / "checkpoints/checkpoint-100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint_fidelity.json").write_text("{}")
    scheduler.add(999, "eval", "cvrp", "grpo_z", "CANCELLED by 304433")
    result = resume.reconcile(output, "user", submit=True, max_new=1, runner=scheduler)
    assert result["submitted"][0]["key"] == ("eval", "cvrp", "grpo_z")
    assert "--dependency=afterok:1000" in scheduler.commands[0]


def test_existing_failed_output_is_never_deleted_or_reused(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output)
    scheduler.add(1000, "train", "cvrp", "grpo_z", "FAILED")
    destination = resume.paths(output, "train", "cvrp", "grpo_z")[1]
    destination.mkdir(parents=True)
    result = resume.reconcile(output, "user", submit=False, runner=scheduler)
    first = result["observations"][0]
    assert first["status"] == "existing_output_requires_review"
    assert destination.exists()


def test_rejected_submission_records_cause_and_stops(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output, fail=True)
    result = resume.reconcile(output, "user", submit=True, runner=scheduler)
    assert not result["submitted"]
    assert len(scheduler.commands) == 1
    assert result["observations"][-1]["status"] == "submission_rejected"
    events = [json.loads(path.read_text()) for path in (output / "ledgers/orchestration_events").glob("*.json")]
    assert {item["kind"] for item in events} == {"intent", "rejected"}


def test_unresolved_intent_fails_closed_during_accounting_lag(tmp_path):
    output = setup_run(tmp_path)
    directory = output / "ledgers/orchestration_events"
    directory.mkdir()
    resume.event(directory, {"kind": "intent", "transaction": "crash-before-response", "key": ["train", "cvrp", "grpo_z"]})
    result = resume.reconcile(output, "user", submit=False, runner=Scheduler(output))
    assert result["observations"][0]["status"] == "awaiting_accounting_reconciliation"


def test_unresolved_retry_intent_does_not_use_old_cancelled_job_as_acceptance(tmp_path):
    output = setup_run(tmp_path)
    directory = output / "ledgers/orchestration_events"
    directory.mkdir()
    resume.event(directory, {"kind": "intent", "transaction": "crash-during-retry", "key": ["train", "cvrp", "grpo_z"],
                            "known_job_ids": ["1000"]})
    scheduler = Scheduler(output)
    scheduler.add(1000, "train", "cvrp", "grpo_z", "CANCELLED")
    result = resume.reconcile(output, "user", submit=False, runner=scheduler)
    assert result["observations"][0]["status"] == "awaiting_accounting_reconciliation"


def test_cancelled_receipted_job_missing_from_date_query_reconciles_by_explicit_id(tmp_path):
    output = setup_run(tmp_path)
    ledger = output / "ledgers/submission.tsv"
    with ledger.open("a") as handle:
        handle.write("eval\tcvrp\tgrpo_z\t1001\tafterok:1000\t/result\n")
    scheduler = Scheduler(output)
    scheduler.add(1000, "train", "cvrp", "grpo_z", "PENDING")
    queried = []

    def runner(command):
        if command[0] == "sacct" and "-j" in command:
            queried.append(command[command.index("-j") + 1])
            # Trusted job-ID binding handles missing StdOut without mapping
            # the arbitrary unknown job returned beside it.
            return "1001|CANCELLED by 304433|eval-cvrp-grpo_z-s43|\n9999|PENDING|mis-grpo_z-s43|"
        return scheduler(command)

    result = resume.reconcile(output, "user", submit=False, runner=runner)
    assert queried == ["1001"]
    cvrp_eval = next(item for item in result["observations"] if item["key"] == ("eval", "cvrp", "grpo_z"))
    assert cvrp_eval["status"] == "waiting_for_training_completion"
    mis_train = next(item for item in result["observations"] if item["key"] == ("train", "mis", "grpo_z"))
    assert mis_train["status"] == "ready"


def test_trusted_job_id_with_conflicting_nonempty_stdout_is_not_adopted(tmp_path):
    output = setup_run(tmp_path)
    result = resume.discover_jobs(output, "1001|PENDING|job|/unrelated/output.out", {"1001": ("train", "cvrp", "grpo_z")})
    assert result == {}


def test_training_only_never_schedules_public_evaluations_after_training_finishes(tmp_path):
    output = setup_run(tmp_path)
    scheduler = Scheduler(output)
    for index, (task, method) in enumerate((task, method) for task in resume.TASKS for method in resume.METHODS):
        scheduler.add(1000 + index, "train", task, method, "COMPLETED")
        checkpoint = resume.paths(output, "train", task, method)[0] / "checkpoints/checkpoint-100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint_fidelity.json").write_text("{}")
    result = resume.reconcile(output, "user", submit=True, training_only=True, runner=scheduler)
    assert not scheduler.commands
    assert len(result["observations"]) == 6
    assert all(item["key"][0] == "train" for item in result["observations"])
