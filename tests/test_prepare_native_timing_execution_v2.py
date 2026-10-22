"""Local-only payload builder guards; no native execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/rloo_sft_anchor_20261013/resource_deadline_extension_v2/prepare_timing_execution_v2.py"


class PreparationGuards(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("tested_native_timing_preparation", SOURCE)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "new_payload"

    def test_default_cli_has_no_output_or_native_action(self):
        with patch("sys.argv", [str(SOURCE)]), contextlib.redirect_stdout(io.StringIO()) as stream:
            self.m.main()
        value = json.loads(stream.getvalue())
        self.assertEqual(value["status"], "prepared_local_builder_only_no_actions")
        self.assertEqual(value["native_mutations"], [])
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_replaced(self):
        self.output.mkdir(); marker = self.output / "existing"; marker.write_text("retained")
        with self.assertRaisesRegex(ValueError, "Fresh output"):
            self.m.prepare(self.output, "2026-10-16T12:00:00+00:00")
        self.assertEqual(marker.read_text(), "retained")

    def test_naive_or_noncanonical_clock_is_rejected_before_sources(self):
        for value in ("2026-10-16T12:00:00", "2026-10-16T12:00:00Z", "2026-10-16T12:00:00+01:00"):
            with self.subTest(clock=value):
                with self.assertRaisesRegex(ValueError, "UTC registration"):
                    self.m.prepare(self.output, value)
                self.assertFalse(self.output.exists())

    def test_command_round_trip_keeps_entire_bootstrap_literal(self):
        arguments = ["--registration", self.m.NATIVE + "/registration.json", "--registration-sha256", "a" * 64]
        command = self.m.command("cvrp_timing_operator_v2.py", "b" * 64, arguments)
        argv = shlex.split(command)
        self.assertEqual(argv[:4], ["/usr/bin/python3", "-I", "-B", "-c"])
        self.assertEqual(argv[4], self.m.BOOT)
        self.assertEqual(argv[5:], [self.m.NATIVE + "/cvrp_timing_operator_v2.py", "b" * 64, *arguments])

    def test_original_source_binding_and_symlink_guards(self):
        p = Path(self.tmp.name) / "original.py"; p.write_text("x=1\n")
        raw = p.read_bytes(); self.assertEqual(self.m.read_source(p, self.m.sha(raw)), raw)
        with self.assertRaisesRegex(ValueError, "frozen source changed"):
            self.m.read_source(p, "0" * 64)
        alias = Path(self.tmp.name) / "alias.py"; alias.symlink_to(p)
        with self.assertRaisesRegex(ValueError, "regular local source"):
            self.m.read_source(alias)

    def test_standby_uses_original_qualified_module_environment(self):
        command = self.m.command("standby_timing_controller_v2.py", "b" * 64, ["--execute"])
        outer = shlex.split(command)
        self.assertEqual(outer[:2], ["/bin/bash", "-lc"])
        lines = outer[2].splitlines()
        self.assertEqual(lines[:5], ["set -e", "module purge", "module load arch/a100", "module load miniforge/24.9.0",
                                    "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1"])
        argv = shlex.split(lines[5])
        self.assertEqual(argv[:5], ["exec", self.m.STANDBY_PYTHON, "-I", "-B", "-c"])
        self.assertEqual(argv[5], self.m.BOOT)
        self.assertEqual(argv[6:], [self.m.NATIVE + "/standby_timing_controller_v2.py", "b" * 64, "--execute"])

    def test_compile_has_no_main_entrypoint(self):
        module = self.m.compile_source(b"x=1\nif __name__=='__main__': raise RuntimeError('executed')\n", "/captured/file.py", "cpu_builder_test")
        self.assertEqual(module.x, 1)

    def test_native_binding_maps_bytes_without_native_read(self):
        self.assertEqual(self.m.native_binding("data.json", b"{}"),
            {"path": self.m.NATIVE + "/data.json", "sha256": self.m.sha(b"{}"), "size_bytes": 2})


if __name__ == "__main__":
    unittest.main()
