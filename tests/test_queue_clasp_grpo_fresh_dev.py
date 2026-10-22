import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("fresh_queue", ROOT / "scripts/queue_clasp_grpo_fresh_dev.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def setup_plan(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    (output / "ledgers").mkdir(parents=True)
    training = tmp_path / "training"
    evaluations = []
    for task in queue.TASKS:
        for method in queue.METHODS:
            train = None if method == "sft" else training / "train" / f"{task}_{queue.fresh.METHODS[method]}_seed43_step100"
            checkpoint = tmp_path / "sft" / task if train is None else train / "checkpoints/checkpoint-100"
            checkpoint.mkdir(parents=True)
            (checkpoint / "checkpoint_fidelity.json").write_text("{}")
            evaluations.append({"task": task, "method": method, "checkpoint": str(checkpoint),
                                "training_root": str(train) if train else None,
                                "output": str(output / "evaluation" / f"{task}_{method}"),
                                "stdout": str(output / "logs" / f"{task}_{method}_%j.out")})
    plan = {"created_at_utc": "2026-09-05T17:00:00Z", "training_campaign_root": str(training),
            "evaluations": evaluations, "source_manifest": {"path": "/frozen/SOURCE_MANIFEST.sha256"},
            "management_root": "/management", "runtime": "/runtime", "data_root": "/data_v3", "addendum_sha256": "a" * 64}
    monkeypatch.setattr(queue, "validate_addendum", lambda _: plan)
    monkeypatch.setattr(queue.fresh, "validate_checkpoint", lambda *args: None)
    return output, plan


class Scheduler:
    def __init__(self, plan, occupied=0, fail=False):
        self.plan, self.occupied, self.fail = plan, occupied, fail
        self.accounting, self.calls = [], []

    def complete_training(self):
        for index, item in enumerate(self.plan["evaluations"]):
            if item["method"] == "sft":
                continue
            stdout = queue.scheduler.paths(Path(self.plan["training_campaign_root"]), "train", item["task"], queue.fresh.METHODS[item["method"]])[2]
            self.accounting.append(f"{1000+index}|COMPLETED|train|{stdout}")

    def __call__(self, command):
        if command[0] == "sacct":
            return "\n".join(self.accounting)
        if command[0] == "squeue":
            assert "-r" in command
            return "\n".join(f"879217_{index}|PENDING|{queue.scheduler.QOS}" for index in range(self.occupied))
        assert command[0] == "sbatch"
        self.calls.append(command)
        if self.fail:
            raise subprocess.CalledProcessError(1, command, stderr="QOSMaxSubmitJobPerUserLimit")
        return str(2000 + len(self.calls))


def test_only_sft_ready_while_terminal_files_exist_but_training_not_completed(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    scheduler = Scheduler(plan)
    result = queue.manage(output, "user", submit=True, max_new=8, runner=scheduler)
    assert len(result["submitted"]) == 2
    assert all(item["key"][1] == "sft" for item in result["submitted"])
    assert sum(item["status"] == "waiting_for_training_completion" for item in result["observations"]) == 6
    assert all("FRESH_DATA_ROOT=/data_v3" in " ".join(command) for command in scheduler.calls)
    assert all("FRESH_TRAIN=" in " ".join(command) for command in scheduler.calls)


def test_expanded_qos_capacity_and_per_call_bound_limit_submission(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    scheduler = Scheduler(plan, occupied=9)
    scheduler.complete_training()
    result = queue.manage(output, "user", submit=True, max_new=2, runner=scheduler)
    assert result["occupied_slots"] == 9 and len(result["submitted"]) == 1
    second = queue.manage(output, "user", submit=True, max_new=1, runner=scheduler)
    assert second["observations"][0]["status"] == "awaiting_accounting_reconciliation"
    assert second["submitted"][0]["key"] != result["submitted"][0]["key"]
    assert all(path.stat().st_mode & 0o222 == 0 for path in (output / "ledgers/fresh_evaluation_events").glob("*.json"))


def test_active_and_completed_evaluations_are_not_submitted_again(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    scheduler = Scheduler(plan, occupied=10)
    first = plan["evaluations"][0]
    scheduler.accounting.append(f"1234|RUNNING|fresh|{first['stdout']}")
    second = plan["evaluations"][1]
    destination = Path(second["output"])
    destination.mkdir(parents=True)
    queue.fresh.write_new(destination / "result_index.json", {"task": second["task"], "method": second["method"],
                                                              "status": "complete_fresh_development_only"}, "index_sha256")
    result = queue.manage(output, "user", submit=True, runner=scheduler)
    assert not scheduler.calls
    assert result["observations"][0]["status"] == "active:1234"
    assert result["observations"][1]["status"] == "complete"


def test_rejection_is_recorded_without_unrelated_actions(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    scheduler = Scheduler(plan, fail=True)
    result = queue.manage(output, "user", submit=True, runner=scheduler)
    assert len(scheduler.calls) == 1 and not result["submitted"]
    assert result["observations"][-1]["status"] == "submission_rejected"
    events = [json.loads(path.read_text()) for path in (output / "ledgers/fresh_evaluation_events").glob("*.json")]
    assert {item["kind"] for item in events} == {"intent", "rejected"}


def test_full_quota_skips_expensive_checkpoint_hashing(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    scheduler = Scheduler(plan, occupied=10)
    scheduler.complete_training()
    validated = []
    monkeypatch.setattr(queue.fresh, "validate_checkpoint", lambda *args: validated.append(args))
    result = queue.manage(output, "user", submit=True, runner=scheduler)
    assert not validated and not scheduler.calls
    assert len(result["observations"]) == 8
    assert all(item["status"] == "ready_for_slot" for item in result["observations"])


def test_existing_failed_output_is_preserved_for_review(tmp_path, monkeypatch):
    output, plan = setup_plan(tmp_path, monkeypatch)
    destination = Path(plan["evaluations"][0]["output"])
    destination.mkdir(parents=True)
    result = queue.manage(output, "user", runner=Scheduler(plan))
    assert result["observations"][0]["status"] == "existing_output_requires_review"
    assert destination.exists()


def test_freeze_preserves_training_protocol_and_binds_all_eight_plans_and_metrics(tmp_path, monkeypatch):
    training = tmp_path / "training"
    (training / "ledgers").mkdir(parents=True)
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("frozen")
    management = tmp_path / "management"
    (management / "scripts").mkdir(parents=True)
    for name in queue.MANAGEMENT_FILES:
        (management / "scripts" / name).write_text("implementation")
    (source / "scripts/compare_development.py").write_text("implementation")
    data = tmp_path / "data_v3"
    data.mkdir()
    protocol = {"training": {"seed": 43, "steps": 100}, "source_manifest": queue.fresh.identity(manifest), "runtime": "/runtime",
                "initializations": {task: {"adapter_model.safetensors": {"path": f"/sft/{task}/checkpoint-31250/adapter_model.safetensors"}} for task in queue.TASKS}}
    path = training / "ledgers/protocol.json"
    queue.fresh.write_new(path, protocol, "receipt_sha256")
    original = path.read_bytes()
    monkeypatch.setattr(queue.fresh, "SOURCE_MANIFEST_SHA256", queue.fresh.identity(manifest)["sha256"])
    monkeypatch.setattr(queue.fresh, "validate_benchmark", lambda _: {"files": {}, "test_validated": True})
    monkeypatch.setattr(queue.fresh, "validate_checkpoint", lambda *args: None)
    output = tmp_path / "fresh"
    result = queue.freeze(argparse.Namespace(output=output, training_root=training, data=data, management=management))
    assert result["evaluations"] == 8 and result["jobs_submitted"] == 0
    addendum = queue.fresh.checked_json(output / "protocol_addendum.json", "addendum_sha256")
    assert addendum["analysis"]["quality_bootstrap"]["seed"] == 20260905
    assert addendum["analysis"]["quality_bootstrap"]["replicates"] == 20000
    assert "summarize_clasp_grpo_fresh_dev.py" in addendum["management_files"]
    assert path.read_bytes() == original
    assert (output / "protocol_addendum.json").stat().st_mode & 0o222 == 0
