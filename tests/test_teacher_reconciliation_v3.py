"""CPU-only authority and real owned-process cleanup checks."""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/rloo_one_shot_20261016/cpu_preparation_reconciliation_v3/run_teacher_reconciliation_v3.py"
LEGACY = ROOT / "experiments/rloo_one_shot_20261016/teacher_execution_v2/pipeline/run_teacher_followup_v2.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class RecoveryAuthority(unittest.TestCase):
    def setUp(self):
        self.m = load(SOURCE, "recovery_authority_tests")

    def budget(self):
        return {"maximum_total_retained_GPU_hours": 20.5, "already_launched_GPU_hours": 1.5,
            "maximum_remaining_GPU_hours": 19.0, "additional_GPU_hours": 0,
            "historical_requested_GPU_hours_retained": 239.5, "reservations_reclaimed": False,
            "automatic_retry": False, "collection_repeated": False, "numerical_recipe_changed": False,
            "training_or_evaluation_data_changed": False, "quality_selection_used": False,
            "publication_eligible": False, "superiority_established": False,
            "execution_deadline_utc": self.m.DEADLINE.isoformat(), "remaining_phases": list(self.m.PHASES)}

    def test_exact_remaining_reservations_are_admitted(self):
        self.m.validate_budget(self.budget())
        self.assertEqual(2 + .5 + .5 + 4 * 4, self.budget()["maximum_remaining_GPU_hours"])

    def test_scope_changes_are_rejected(self):
        for key, before in self.budget().items():
            with self.subTest(key=key):
                value = self.budget()
                if type(before) is bool:
                    value[key] = not before
                elif type(before) in (int, float):
                    value[key] = before + 1
                elif isinstance(before, list):
                    value[key] = list(reversed(before))
                else:
                    value[key] = "changed"
                with self.assertRaisesRegex(ValueError, "recovery scope"):
                    self.m.validate_budget(value)

    def test_collection_and_failed_CPU_record_remain_in_registration(self):
        value = {"master_packet": {"old": "master"}, "failed_controller_status": {"failed": "status"},
            "failed_CPU_launch": {"failed": "launch"}, "failed_CPU_log": {"failed": "log"},
            "old_collection": {"completed": "collection"}, "adapter": {"new": "guard"},
            "runtime_bootstrap": {"new": "bootstrap"}, "runtime_context": {"new": "context"},
            "canonical_downstream_root": "/canonical/original"}
        result = self.m.registration({"new": "packet"}, value)
        self.assertEqual(result["master_packet"], value["master_packet"])
        self.assertEqual(result["old_collection"], value["old_collection"])
        self.assertEqual(result["old_failure_bindings"], {key: value[key] for key in ("failed_controller_status", "failed_CPU_launch", "failed_CPU_log")})
        self.assertEqual(result["receipt_sha256"], self.m.sha(self.m.encoded({key: val for key, val in result.items() if key != "receipt_sha256"})))
        self.assertFalse(result["collection_repeated"])

    def test_default_command_is_inert(self):
        with patch("sys.argv", [str(SOURCE)]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.m.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "prepared_reconciliation_only_no_actions")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["GPU_calls"], 0)

    def test_bad_byte_bindings_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "evidence.json"; p.write_text("{}")
            row = self.m.binding(p); self.assertEqual(self.m.read_bound(row), b"{}")
            for delta in ({"sha256": "0" * 64}, {"size_bytes": True}, {"extra": "unbound"}):
                with self.subTest(delta=delta), self.assertRaises(ValueError):
                    self.m.read_bound({**row, **delta})
            alias = Path(directory) / "alias.json"; alias.symlink_to(p)
            with self.assertRaises(ValueError):
                self.m.read_bound({**row, "path": str(alias)})

    def test_write_once_never_replaces_failure_history(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "failure.json"; self.m.write_once(p, {"status": "failed"})
            before = p.read_bytes()
            with self.assertRaises(FileExistsError):
                self.m.write_once(p, {"status": "success"})
            self.assertEqual(p.read_bytes(), before)

    def test_invalid_master_cannot_reach_execution(self):
        value = self.m.seal({"protocol": self.m.PROTOCOL, **self.budget(),
            "master_packet": {"sha256": "0" * 64}, "prior_failed_screen": {"sha256": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "Original teacher master"):
            self.m.validate_packet(value)


class OwnedCPUProcess(unittest.TestCase):
    def setUp(self):
        self.m = load(SOURCE, "recovery_process_tests")
        self.old = load(LEGACY, "recovery_original_cleanup")
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.children = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for process in self.children:
            if process.poll() is None:
                self.old.stop_own_group(process)

    def launch(self, *args, **kwargs):
        self.assertTrue(kwargs["start_new_session"])
        process = subprocess.Popen(*args, **kwargs); self.children.append(process)
        return process

    def run_phase(self, code, *, cap=1, on_started=lambda p: None, on_wait=lambda: None, **kwargs):
        return self.m.run_owned_phase(["/usr/bin/python3", "-B", "-c", code], Path(self.tmp.name) / "phase.log",
            {**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cap, on_started=on_started, on_wait=on_wait,
            stop_group=self.old.stop_own_group, launch=self.launch, **kwargs)

    def assert_reaped(self):
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())
        with self.assertRaises(ProcessLookupError):
            os.killpg(self.children[0].pid, 0)

    def test_normal_owned_phase(self):
        result = self.run_phase("print('closed engineering fixture')")
        self.assertEqual(result["returncode"], 0); self.assert_reaped()

    def test_first_postspawn_heartbeat_failure_reaps_group(self):
        def fail(process):
            raise RuntimeError("first heartbeat failed")
        with self.assertRaisesRegex(RuntimeError, "heartbeat"):
            self.run_phase("import time;time.sleep(30)", on_started=fail)
        self.assert_reaped()

    def test_timeout_reaps_group(self):
        with self.assertRaisesRegex(ValueError, "transport bound"):
            self.run_phase("import time;time.sleep(30)", cap=.03)
        self.assert_reaped()

    def test_slow_first_heartbeat_cannot_admit_late_zero_exit(self):
        import time
        def slow(process):
            process.wait(timeout=2)
            time.sleep(.04)
        with self.assertRaisesRegex(ValueError, "transport bound before successful admission"):
            self.run_phase("pass", cap=.02, on_started=slow)
        self.assert_reaped()

    def test_wait_heartbeat_failure_reaps_group(self):
        def fail():
            raise RuntimeError("wait heartbeat failed")
        with self.assertRaisesRegex(RuntimeError, "wait heartbeat"):
            self.run_phase("import time;time.sleep(30)", cap=.03, on_wait=fail)
        self.assert_reaped()

    def test_nonzero_exit_is_not_retried(self):
        with self.assertRaisesRegex(ValueError, "without retry"):
            self.run_phase("raise SystemExit(7)")
        self.assert_reaped()

    def test_absolute_deadline_limits_larger_transport_bound(self):
        with self.assertRaisesRegex(ValueError, "transport bound"):
            self.run_phase("import time;time.sleep(30)", cap=20,
                absolute_deadline=datetime.now(timezone.utc) + timedelta(seconds=.04))
        self.assert_reaped()

    def test_deadline_reached_while_spawning_reaps_new_group(self):
        with self.assertRaisesRegex(ValueError, "after spawn"):
            self.run_phase("import time;time.sleep(30)", cap=20,
                absolute_deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
        self.assert_reaped()

    def test_base_exception_from_first_heartbeat_reaps_group(self):
        def fail(process):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_phase("import time;time.sleep(30)", on_started=fail)
        self.assert_reaped()

    def test_nonfinite_limit_cannot_launch(self):
        for cap in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.run_phase("pass", cap=cap)
        self.assertEqual(self.children, [])


if __name__ == "__main__":
    unittest.main()
