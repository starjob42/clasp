"""The new pilot preserves baseline arguments and rejects inherited variant knobs."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "scripts/train_grpo_verified_checkpoint_dev_a100.slurm"
OLD = ROOT / "scripts/train_clasp_canonical_checkpoint_dev_a100.slurm"


class VerifiedLauncherTests(unittest.TestCase):
    def run_header(self, method="grpo_z", **overrides):
        source = NEW.read_text()
        header = source[source.index("set -euo pipefail"):source.index('case "${TASK}" in')]
        scrub_start = source.index("while IFS='=' read -r inherited_name")
        scrub = source[scrub_start:source.index("unset RESUME_FROM_CHECKPOINT", scrub_start)]
        export_start = source.index('export GRPO_GRADED_REWARD=')
        exports = source[export_start:source.index('export GRPO_DATA_SEED=', export_start)]
        env = {**os.environ, "DEV_TASK": "mis", "DEV_METHOD": method,
               "DEV_SFT_CHECKPOINT": "/tmp", "DEV_OUTPUT_DIR": "/tmp/verified-launcher-test-unused",
               "DEV_PROJECT_ROOT": "/tmp", "DEV_SEALED_ENV": "/tmp", "DEV_DATA_ROOT": "/tmp",
               "DEV_SEED": "43", "DEV_MAX_STEPS": "100", "DEV_SAVE_STEPS": "100",
               "DEV_SAVE_TOTAL_LIMIT": "1", "DEV_GROUP_STD_CORRECTION": "1",
               "DEV_RESUME_CHECKPOINT": "", "DEV_ROLLOUT_BATCH_SIZE": "",
               "DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES": "0",
               "DEV_TEACHER_SELECTION": "per_completion",
               "GRPO_GRADED_REWARD": "99", "GRPO_VERIFIED_IMPROVEMENT": "99",
               "GRPO_VERIFIED_IMPROVEMENT_WEIGHT": "99", "GRPO_UNKNOWN_EXPERIMENT": "99",
               **overrides}
        code = header + scrub + exports + '\n env | sort\n'
        return subprocess.run(["bash", "-c", code], env=env, capture_output=True, text=True)

    def test_baseline_and_verified_flags_are_fixed_after_environment_scrub(self):
        for method, enabled, weight in (("grpo_z", "0", "0.0"), ("grpo_verified", "1", "0.05")):
            with self.subTest(method=method):
                result = self.run_header(method)
                self.assertEqual(result.returncode, 0, result.stderr)
                env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                self.assertEqual(env["GRPO_GRADED_REWARD"], enabled)
                self.assertEqual(env["GRPO_VERIFIED_IMPROVEMENT"], enabled)
                self.assertEqual(env["GRPO_VERIFIED_IMPROVEMENT_WEIGHT"], weight)
                self.assertEqual(env["GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS"], "3000")
                self.assertEqual(env["GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES"], "2")
                self.assertNotIn("GRPO_UNKNOWN_EXPERIMENT", env)

    def test_wrong_budget_or_resume_is_rejected(self):
        for change in ({"DEV_SEED": "44"}, {"DEV_MAX_STEPS": "101"}, {"DEV_GROUP_STD_CORRECTION": "0"},
                       {"DEV_RESUME_CHECKPOINT": "/tmp"}, {"DEV_ROLLOUT_BATCH_SIZE": "18"}, {"DEV_TASK": "tsp"}):
            with self.subTest(change=change):
                self.assertNotEqual(self.run_header(**change).returncode, 0)
        self.assertNotEqual(self.run_header("clasp_grpo_anchor_v3").returncode, 0)

    def test_common_optimizer_rollout_and_scalarization_arguments_match_baseline(self):
        old, new = OLD.read_text(), NEW.read_text()
        for name in ("COMMON_ARGS", "PD_ARGS"):
            a = old.split(name + "=(\n", 1)[1].split("\n)", 1)[0]
            b = new.split(name + "=(\n", 1)[1].split("\n)", 1)[0]
            self.assertEqual(a, b)
        a = old.split('if [[ "${METHOD}" == "grpo_z" ]]; then\n    METHOD_ARGS=(\n', 1)[1].split("\n    )", 1)[0]
        b = new.split("METHOD_ARGS=(\n", 1)[1].split("\n)", 1)[0]
        self.assertEqual(" ".join(a.split()), " ".join(b.split()))

    def test_short_distributed_diagnostic_requires_explicit_mode(self):
        self.assertNotEqual(self.run_header(DEV_MAX_STEPS="5", DEV_SAVE_STEPS="5").returncode, 0)
        result = self.run_header("grpo_verified", DEV_MAX_STEPS="5", DEV_SAVE_STEPS="5",
                                 DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES="5")
        self.assertEqual(result.returncode, 0, result.stderr)
        env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.assertEqual(env["CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL"], "1")
        self.assertEqual(env["CLASP_GRADIENT_CONSISTENCY_CHECKS"], "1")
        self.assertNotEqual(self.run_header(DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES="5").returncode, 0)

    def test_incumbent_selection_is_explicit_and_modified_only(self):
        result = self.run_header("grpo_verified", DEV_TEACHER_SELECTION="best_per_prompt")
        self.assertEqual(result.returncode, 0, result.stderr)
        env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.assertEqual(env["GRPO_VERIFIED_TEACHER_SELECTION"], "best_per_prompt")
        self.assertNotEqual(self.run_header(DEV_TEACHER_SELECTION="best_per_prompt").returncode, 0)
        self.assertNotEqual(self.run_header("grpo_verified", DEV_TEACHER_SELECTION="unknown").returncode, 0)


if __name__ == "__main__":
    unittest.main()
