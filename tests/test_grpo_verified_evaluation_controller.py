"""Failure/identity gates for the independent four-arm evaluation controller."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verified_evaluation", ROOT / "scripts/evaluate_grpo_verified_ddpfix.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def scheduler_rows(raw):
    rows = {}
    for line in raw.splitlines():
        job, state, code, stdout, elapsed = line.split("|")
        rows[job] = {"job_id": job, "state": state, "exit_code": code,
                     "stdout": stdout, "elapsed_seconds": int(elapsed)}
    return rows


training_control = SimpleNamespace(scheduler_rows=scheduler_rows,
                                   expanded_queue=lambda raw: {line.split("|")[0] for line in raw.splitlines()})


class TrainingGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "ledgers/terminal").mkdir(parents=True)
        self.items = [{"key": t + "/" + m} for t in evaluation.TASKS for m in evaluation.METHODS]
        self.protocol = {"receipt_sha256": "training-pin", "planned_train_commands": self.items}
        self.events = {item["key"]: [{"kind": "submitted", "job_id": str(1800000 + i)}] for i, item in enumerate(self.items)}
        self.control = SimpleNamespace(read_events=lambda *args: self.events,
                                       scheduler_rows=training_control.scheduler_rows,
                                       terminal_validation=lambda item, *args: {"status": "terminal_training_verified"})
        self.rows = "".join(f"{1800000+i}|COMPLETED|0:0|Unknown|3500\n" for i in range(4))
        for i, item in enumerate(self.items):
            self.write_terminal(item, str(1800000 + i))

    def write_terminal(self, item, job, **overrides):
        path = self.root / "ledgers/terminal" / (item["key"].replace("/", "_") + ".json")
        if path.exists():
            path.unlink()
        evaluation.write_new(path, {"status": "terminal_training_verified", "global_step": 100,
                                   "job_id": job, "key": item["key"], "protocol_sha256": "training-pin", **overrides})

    def ready(self, **kwargs):
        return evaluation.training_ready(self.root, self.protocol, self.root / "source", self.control,
                                         runner=lambda command: self.rows, **kwargs)

    def test_failed_job_is_rejected_even_with_complete_checkpoint_receipt(self):
        self.rows = self.rows.replace("1800001|COMPLETED|0:0", "1800001|FAILED|1:0")
        with self.assertRaisesRegex(ValueError, "failed training is comparison-ineligible"):
            self.ready()

    def test_completed_nonzero_exit_is_rejected(self):
        self.rows = self.rows.replace("1800001|COMPLETED|0:0", "1800001|COMPLETED|1:0")
        with self.assertRaises(ValueError):
            self.ready()

    def test_running_arm_waits(self):
        self.rows = self.rows.replace("1800001|COMPLETED", "1800001|RUNNING")
        self.assertFalse(self.ready())

    def test_missing_terminal_receipt_waits(self):
        next((self.root / "ledgers/terminal").iterdir()).unlink()
        self.assertFalse(self.ready())

    def test_stale_source_protocol_binding_is_rejected(self):
        self.write_terminal(self.items[0], "1800000", protocol_sha256="old-source-protocol")
        with self.assertRaisesRegex(ValueError, "terminal receipt binding mismatch"):
            self.ready()

    def test_changed_terminal_policy_is_rejected(self):
        self.control.terminal_validation = lambda *args: {"global_step": 99}
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            self.ready(revalidate=True)

    def test_all_four_complete_receipts_allow_evaluation(self):
        self.assertTrue(self.ready(revalidate=True))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "ledgers/evaluation_events").mkdir(parents=True)
        self.plans = [{"key": t + "/" + m, "task": t, "method": m,
                       "output": str(self.root / "evaluation" / (t + "_" + m)), "command": ["sbatch", t, m]}
                      for t in evaluation.TASKS for m in evaluation.METHODS]
        self.p = {"receipt_sha256": "evaluation-pin", "plans": self.plans}
        patcher = patch.object(evaluation, "protocol_for", return_value=(self.p, {}, self.root, training_control, None))
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(evaluation, "training_ready", return_value=True)
        patcher.start(); self.addCleanup(patcher.stop)
        self.commands, self.occupied, self.rows = [], 5, ""

    def runner(self, command):
        if command[0] == "squeue":
            return "".join(f"1790000_{i}|PENDING\n" for i in range(self.occupied))
        if command[0] == "sacct":
            return self.rows
        if command[0] == "sbatch":
            self.commands.append(command)
            return str(1810000 + len(self.commands)) + "\n"
        raise AssertionError(command)

    def test_declared_four_jobs_submit_once_during_scheduler_lag(self):
        result = evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(len(self.commands), 4)
        self.assertFalse(result["complete"])
        evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(len(self.commands), 4)

    def test_one_slot_reserves_only_one_job(self):
        self.occupied = 9
        evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(len(self.commands), 1)

    def test_readonly_run_does_not_submit(self):
        evaluation.reconcile(self.root, runner=self.runner)
        self.assertEqual(self.commands, [])
        self.assertEqual(list((self.root / "ledgers/evaluation_events").iterdir()), [])

    def test_failed_evaluation_blocks_summary_and_retry(self):
        evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.rows = "1810001|FAILED|1:0|Unknown|10\n"
        with self.assertRaisesRegex(ValueError, "failed evaluation"):
            evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(len(self.commands), 4)

    def test_training_not_ready_prevents_all_submissions(self):
        with patch.object(evaluation, "training_ready", return_value=False):
            result = evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(result["status"], "waiting_for_all_four_validated_training_arms")
        self.assertEqual(self.commands, [])

    def test_ambiguous_submission_is_never_retried(self):
        def runner(command):
            if command[0] == "sbatch":
                return "accepted but no unique ID"
            return self.runner(command)
        with self.assertRaisesRegex(ValueError, "invalid job ID"):
            evaluation.reconcile(self.root, submit=True, runner=runner)
        with self.assertRaisesRegex(ValueError, "no automatic retry"):
            evaluation.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(self.commands, [])


class WrapperTests(unittest.TestCase):
    def test_only_new_method_and_helper_change_in_existing_wrapper(self):
        old = (ROOT / evaluation.WRAPPER).read_text()
        new = evaluation.patched_wrapper(old)
        restored = new.replace("grpo_z|grpo_verified|v3|v4)", "grpo_z|v3|v4)").replace(
            'HELPER="${MANAGEMENT}/scripts/evaluate_grpo_verified_ddpfix.py"',
            'HELPER="${MANAGEMENT}/scripts/clasp_grpo_fresh_dev_artifacts.py"')
        self.assertEqual(restored, old)
        self.assertIn("SEED=271828", new)
        self.assertIn("GENERATION_BATCH_SIZE=8", new)
        self.assertIn("#SBATCH --time=02:00:00", new)

    def test_unknown_wrapper_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interface changed"):
            evaluation.patched_wrapper("different wrapper")


if __name__ == "__main__":
    unittest.main()
