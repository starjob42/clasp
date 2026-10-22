"""CPU-only failure/order checks for the proposed native standby."""
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/standby_timing_controller_v2.py"


def load():
    spec = importlib.util.spec_from_file_location("tested_native_timing_standby", SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class StandbyGuards(unittest.TestCase):
    def setUp(self):
        self.m = load(); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.m.ROOT = self.root
        self.clock = lambda: datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
        self.spec = {"fixed": "prospective_spec"}
        self.owner = {"pid": 12345, "uid": 77, "hostname": "fixture", "proc_start_ticks": "123"}
        self.helper = types.SimpleNamespace(OWNER=self.owner, process_snapshot=lambda pid: {"pid": pid, "alive": False})
        self.events = []
        self.context = {}
        self.operator = types.SimpleNamespace(verify_inputs=lambda context: self.events.append("verify"), load_handoff=self.handoff)

    def handoff(self, context):
        p = self.root / "handoff/complete.json"; raw = p.read_bytes()
        return {"path": str(p), "sha256": self.m.sha(raw), "size_bytes": len(raw)}

    def complete(self, **updates):
        value = {"spec_sha256": self.m.sha(self.m.encoded(self.spec)), "old_owner": self.owner,
                 "signals": ["SIGSTOP", "SIGTERM", "SIGCONT"]}
        value.update(updates); p = self.root / "handoff/complete.json"; p.parent.mkdir(exist_ok=True)
        p.write_bytes(self.m.encoded(value) + b"\n")

    def wait(self, **kwargs):
        return self.m.wait_for_handoff(self.context, self.operator, self.helper, self.spec, clock=self.clock,
                                      sleep=lambda seconds: self.fail("unexpected waiting"), **kwargs)

    def test_default_cli_is_inert(self):
        with patch("sys.argv", [str(SOURCE)]), contextlib.redirect_stdout(io.StringIO()) as out:
            self.m.main()
        self.assertEqual(json.loads(out.getvalue())["status"], "prepared_cpu_standby_only_no_actions")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_actual_execute_requires_native_owner_before_any_reads(self):
        with self.assertRaisesRegex(ValueError, "native login host"):
            self.m.execute("/missing", "a" * 64, "/missing", "b" * 64)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_no_handoff_waits_without_executing_controller(self):
        def sleep(seconds):
            self.events.append(("sleep", seconds)); self.complete()
        emitted = []
        row = self.m.wait_for_handoff(self.context, self.operator, self.helper, self.spec, clock=self.clock,
                                      sleep=sleep, emit=emitted.append)
        self.assertEqual(emitted[0]["status"], "waiting_for_exact_cpu_handoff")
        self.assertEqual(self.events, ["verify", ("sleep", 10), "verify", "verify"])
        self.assertEqual(row["path"], str(self.root / "handoff/complete.json"))

    def test_exact_closed_handoff_is_rechecked(self):
        self.complete(); self.wait(); self.assertEqual(self.events, ["verify", "verify"])

    def test_requires_investigation_prevents_activation(self):
        self.complete(); (self.root / "handoff/requires_investigation.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "requires investigation"):
            self.wait()

    def test_deadline_prevents_wait_or_activation(self):
        self.complete(); self.clock = lambda: datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "waiting deadline"):
            self.wait()
        self.assertEqual(self.events, [])

    def test_different_spec_rejected(self):
        self.complete(spec_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "external spec"):
            self.wait()

    def test_different_owner_rejected(self):
        self.complete(old_owner={**self.owner, "pid": 9})
        with self.assertRaisesRegex(ValueError, "external spec"):
            self.wait()

    def test_extra_signal_rejected(self):
        self.complete(signals=["SIGSTOP", "SIGTERM", "SIGCONT", "SIGKILL"])
        with self.assertRaisesRegex(ValueError, "external spec"):
            self.wait()

    def test_live_old_owner_prevents_controller(self):
        self.complete(); self.helper.process_snapshot = lambda pid: {**self.owner, "alive": True, "state": "S"}
        with self.assertRaisesRegex(ValueError, "still live"):
            self.wait()

    def test_terminal_same_identity_admitted(self):
        self.complete(); self.helper.process_snapshot = lambda pid: {**self.owner, "alive": True, "state": "Z"}
        self.wait()

    def test_terminal_reused_pid_rejected(self):
        self.complete(); self.helper.process_snapshot = lambda pid: {**self.owner, "alive": True, "state": "Z", "proc_start_ticks": "999"}
        with self.assertRaisesRegex(ValueError, "PID was reused"):
            self.wait()

    def test_source_drift_prevents_handoff_consumption(self):
        self.complete()
        def drift(context):
            raise ValueError("source drift")
        self.operator.verify_inputs = drift
        with self.assertRaisesRegex(ValueError, "source drift"):
            self.wait()

    def test_canonical_spec_and_byte_sha_required(self):
        spec = self.m.seal({"protocol": "sft-anchor-cvrp-controller-handoff-v2-spec", "new_registration_sha256": "a" * 64,
            "handoff_helper_sha256": "b" * 64, "synthetic_fixture": False, "automatic_retry": False,
            "scheduler_mutations_allowed": False})
        raw = self.m.encoded(spec) + b"\n"; digest = self.m.sha(raw)
        self.assertEqual(self.m.validate_spec(spec, digest, "a" * 64, "b" * 64), spec)
        for changed in ("spec", "registration", "helper", "synthetic", "retry", "mutation"):
            with self.subTest(changed=changed):
                v = dict(spec); args = [digest, "a" * 64, "b" * 64]
                if changed in {"spec", "registration", "helper"}:
                    args[{"spec": 0, "registration": 1, "helper": 2}[changed]] = "c" * 64
                else:
                    key = {"synthetic": "synthetic_fixture", "retry": "automatic_retry", "mutation": "scheduler_mutations_allowed"}[changed]
                    v[key] = True; v = self.m.seal(v); args[0] = self.m.sha(self.m.encoded(v) + b"\n")
                with self.assertRaises(ValueError):
                    self.m.validate_spec(v, *args)

    def test_compile_uses_captured_bound_buffer(self):
        raw = b"x=7\n"; row = {"path": "/fictional/captured.py", "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        value = self.m.compile_buffer(row, raw, "standby_compile_test")
        self.assertEqual(value.x, 7)
        with self.assertRaises(ValueError):
            self.m.compile_buffer(row, b"x=8\n", "standby_compile_test_changed")

    def test_symlink_source_rejected(self):
        target = self.root / "source"; target.write_text("test")
        alias = self.root / "alias"; alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.m.stable(alias)

    def test_write_once_preserves_first_attempt(self):
        p = self.root / "intent.json"; self.m.write_once(p, {"attempt": 1}); before = p.read_bytes()
        with self.assertRaises(FileExistsError):
            self.m.write_once(p, {"attempt": 2})
        self.assertEqual(p.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
