"""Exercise the actual launcher's admission, argv, and immutable source gates."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_grpo_incumbent_dev_a100.slurm"
PREVIOUS = ROOT / "scripts/train_grpo_verified_checkpoint_dev_a100.slurm"


class IncumbentLauncherTests(unittest.TestCase):
    def resolved(self, method="grpo_z", **overrides):
        source = LAUNCHER.read_text()
        header = source[source.index("set -euo pipefail"):source.index('case "${TASK}" in')]
        scrub_start = source.index("while IFS='=' read -r inherited_name")
        scrub = source[scrub_start:source.index('CACHE_ROOT=', scrub_start)]
        exports = source[source.index('export CLASP_LORA_TARGET_MODULES='):source.index('# The fidelity verifier')]
        env = {key: value for key, value in os.environ.items() if not key.startswith("DEV_")}
        env.update({"DEV_TASK": "mis", "DEV_METHOD": method,
                    "DEV_SFT_CHECKPOINT": "/tmp", "DEV_OUTPUT_DIR": "/tmp/not-created-by-launcher-test",
                    "DEV_PROJECT_ROOT": "/tmp", "DEV_SEALED_ENV": "/tmp", "DEV_DATA_ROOT": "/tmp",
                    "DEV_SOURCE_MANIFEST_SHA256": "a" * 64,
                    "DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES": "0",
                    "GRPO_VERIFIED_IMPROVEMENT_WEIGHT": "99", "GRPO_STOP_AT_GLOBAL_STEP": "1",
                    "GRPO_VERIFIED_TEACHER_NORMALIZATION": "per_microbatch",
                    "CLASP_KL_REFERENCE_MODEL_PATH": "/wrong/reference",
                    "FOARL_UNDECLARED_EXPERIMENT": "99", "HF_TOKEN": "not-a-real-secret",
                    **overrides})
        script = (header + scrub + f"\nSEALED_PYTHON={shlex.quote(sys.executable)}\nCACHE_ROOT=/tmp/test-cache\n"
                  + exports + '\n"${SEALED_PYTHON}" - "${TRAIN_COMMAND[@]}" <<\'PY\'\n'
                  + "import json, os, sys\nprint(json.dumps({'argv':sys.argv[1:], 'env':{k:v for k,v in os.environ.items() if k.startswith(('CLASP_', 'GRPO_', 'PD_', 'FOARL_', 'HF_'))}}))\nPY\n")
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def result(self, method="grpo_z", **overrides):
        result = self.resolved(method, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_all_three_arms_share_sampling_optimizer_and_initialization_arguments(self):
        runs = {method: self.result(method) for method in ("grpo_z", "grpo_verified", "foarl_f638_dev")}
        shared = {
            "--model_path": "/tmp", "--tokenizer_path": "/tmp", "--tasks": "mis",
            "--max_steps": "100", "--seed": "43", "--num_generations": "9",
            "--generation_batch_size": "36", "--per_device_train_batch_size": "3",
            "--gradient_accumulation_steps": "3", "--learning_rate": "1e-6",
            "--optim": "adamw_8bit", "--weight_decay": "0.01", "--warmup_ratio": "0.03",
            "--lr_scheduler_type": "linear", "--max_prompt_length": "29768",
            "--max_completion_length": "3000", "--generation_temperature": "0.7",
            "--generation_top_p": "0.9", "--generation_top_k": "50",
            "--generation_repetition_penalty": "1.0", "--save_steps": "100",
            "--save_total_limit": "1", "--dtype": "bfloat16", "--lora_r": "64",
            "--lora_alpha": "64", "--lora_dropout": "0.0", "--max_eval_samples_per_task": "0",
            "--eval_ratio": "0", "--min_train_samples_per_task": "1900",
        }
        for method, run in runs.items():
            with self.subTest(method=method):
                argv = run["argv"]
                for flag, value in shared.items():
                    self.assertEqual(argv.count(flag), 1, flag)
                    self.assertEqual(argv[argv.index(flag) + 1], value, flag)
                self.assertIn("--nproc_per_node=4", argv)
                self.assertNotIn("GRPO_STOP_AT_GLOBAL_STEP", run["env"])
                self.assertNotIn("FOARL_UNDECLARED_EXPERIMENT", run["env"])
                self.assertNotIn("HF_TOKEN", run["env"])
                self.assertEqual(run["env"]["CLASP_POLICY_CONSISTENCY_CHECKS"], "1")

    def test_plain_grpo_algorithm_arguments_are_identical_to_previous_control(self):
        source = PREVIOUS.read_text()
        arrays = source[source.index("COMMON_ARGS=(\n"):source.index("\nsrun --ntasks=1")]
        # Expand the literal historical arrays without training or filesystem writes.
        code = "TASK=mis; SFT_CHECKPOINT=/tmp; DATA_ROOT=/tmp; OUTPUT_DIR=/tmp/not-created-by-launcher-test; TRAIN_SEED=43; MAX_STEPS=100; ROLLOUT_BATCH_SIZE=9; SAVE_STEPS=100; SAVE_TOTAL_LIMIT=1\n"
        code += arrays + "\nprintf '%s\\0' \"${COMMON_ARGS[@]}\" \"${METHOD_ARGS[@]}\"\n"
        old = subprocess.run(["bash", "-c", code], capture_output=True, check=True).stdout.decode().split("\0")[:-1]
        new = self.result()["argv"]
        new = new[new.index("clasp.entrypoints.train_grpo") + 1:]
        # no_reason/verify_gt_rewards moved from common to method arguments only.
        self.assertEqual(sorted(old), sorted(new))

    def test_modified_teacher_recipe_and_optional_ablation_are_explicit(self):
        run = self.result("grpo_verified")
        self.assertEqual(run["env"]["GRPO_VERIFIED_TEACHER_SELECTION"], "feasibility_incumbent")
        self.assertEqual(run["env"]["GRPO_VERIFIED_TEACHER_NORMALIZATION"], "optimizer_window")
        self.assertEqual(run["env"]["GRPO_VERIFIED_IMPROVEMENT_WEIGHT"], "0.05")
        self.assertEqual(run["env"]["GRPO_GRADED_REWARD"], "1")
        ablation = self.result("grpo_verified", DEV_TEACHER_SELECTION="best_per_prompt")
        self.assertEqual(ablation["env"]["GRPO_VERIFIED_TEACHER_SELECTION"], "best_per_prompt")
        for method in ("grpo_z", "foarl_f638_dev"):
            run = self.result(method)
            self.assertEqual(run["env"]["GRPO_VERIFIED_IMPROVEMENT"], "0")
            self.assertEqual(run["env"]["GRPO_VERIFIED_IMPROVEMENT_WEIGHT"], "0.0")
            self.assertEqual(run["env"]["GRPO_VERIFIED_TEACHER_NORMALIZATION"], "microbatch")

    def test_foarl_keeps_released_rewards_reference_and_declares_common_prompt(self):
        run = self.result("foarl_f638_dev")
        argv, env = run["argv"], run["env"]
        self.assertIn("clasp.entrypoints.train_foarl_f638_dev_20260907", argv)
        self.assertEqual(argv[argv.index("--grpo_beta") + 1], "0.05")
        self.assertEqual(argv[argv.index("--foarl_scale_rewards") + 1], "group")
        self.assertEqual(argv[argv.index("--foarl_dev_source_manifest_sha256") + 1], "a" * 64)
        self.assertIn("--no_reason", argv)
        self.assertIn("--no-verify_gt_rewards", argv)
        self.assertIn("--no-use_primal_dual", argv)
        self.assertIn("--pd_force_scale_rewards_none", argv)  # Inactive because use_primal_dual=False; runtime enforces scale=group.
        self.assertEqual(env["CLASP_KL_REFERENCE_MODEL_PATH"], "/tmp")
        self.assertEqual(env["CLASP_KL_REFERENCE_RECEIPT"], "/tmp/checkpoint_fidelity.json")
        self.assertEqual(env["CLASP_MIS_JIANG_RESPONSE_OUTPUT"], "0")
        self.assertNotIn("CLASP_KL_REFERENCE_MODEL_PATH", self.result()["env"])
        self.assertEqual(self.result("foarl_f638_dev", DEV_TASK="cvrp")["env"]["CLASP_MIS_JIANG_RESPONSE_OUTPUT"], "0")

    def test_invalid_overrides_are_rejected_before_any_runtime_or_output_operations(self):
        changes = ({"DEV_MAX_STEPS": "300"}, {"DEV_SAVE_STEPS": "50"},
                   {"DEV_SEED": "44"}, {"DEV_RESUME_CHECKPOINT": "/tmp"},
                   {"DEV_ROLLOUT_BATCH_SIZE": "18"}, {"DEV_TASK": "tsp"},
                   {"DEV_GROUP_STD_CORRECTION": "0"}, {"DEV_SAVE_TOTAL_LIMIT": "2"},
                   {"DEV_TEACHER_NORMALIZATION": "per_microbatch"},
                   {"DEV_SOURCE_MANIFEST_SHA256": "not-a-digest"},
                   {"DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES": "1"})
        for changeset in changes:
            with self.subTest(changeset=changeset):
                self.assertNotEqual(self.resolved(**changeset).returncode, 0)
        self.assertNotEqual(self.resolved("unknown").returncode, 0)
        self.assertNotEqual(self.resolved(DEV_TEACHER_NORMALIZATION="optimizer_window").returncode, 0)
        self.assertNotEqual(self.resolved("foarl_f638_dev", DEV_TEACHER_NORMALIZATION="optimizer_window").returncode, 0)
        self.assertNotEqual(self.resolved("grpo_verified", DEV_TEACHER_NORMALIZATION="microbatch").returncode, 0)
        self.assertNotEqual(self.resolved("grpo_verified", DEV_TEACHER_SELECTION="per_completion").returncode, 0)
        self.assertNotEqual(self.resolved(DEV_TEACHER_SELECTION="feasibility_incumbent").returncode, 0)

    def test_five_update_diagnostic_is_explicit_and_checks_every_step(self):
        self.assertNotEqual(self.resolved(DEV_MAX_STEPS="5", DEV_SAVE_STEPS="5").returncode, 0)
        for method in ("grpo_z", "grpo_verified", "foarl_f638_dev"):
            run = self.result(method, DEV_MAX_STEPS="5", DEV_SAVE_STEPS="5", DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES="5")
            self.assertEqual(run["env"]["CLASP_GRADIENT_CONSISTENCY_CHECKS"], "1")
            self.assertEqual(run["env"]["CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL"], "1")


class IncumbentSourceGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "snapshot"
        self.launcher = self.root / "scripts/train_grpo_incumbent_dev_a100.slurm"
        self.launcher.parent.mkdir(parents=True)
        self.launcher.write_bytes(LAUNCHER.read_bytes())
        self.spool = Path(self.temporary.name) / "slurm_script"
        self.spool.write_bytes(LAUNCHER.read_bytes())
        self.manifest = self.root / "SOURCE_MANIFEST.sha256"
        self.manifest.write_text(hashlib.sha256(self.launcher.read_bytes()).hexdigest() + "  scripts/train_grpo_incumbent_dev_a100.slurm\n")
        self.expected = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.code = LAUNCHER.read_text().split("<<'PY_SOURCE'\n", 1)[1].split("\nPY_SOURCE", 1)[0]

    def run_gate(self):
        return subprocess.run([sys.executable, "-", str(self.root), self.expected, str(self.spool)],
                              input=self.code, text=True, capture_output=True)

    def test_identical_slurm_spool_alias_is_accepted(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_different_spool_bytes_are_rejected(self):
        self.spool.write_text("changed")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_changed_frozen_file_is_rejected(self):
        self.launcher.write_text("changed")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_changed_manifest_is_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_extra_unmanifested_source_is_rejected(self):
        (self.root / "shadow.py").write_text("raise RuntimeError('shadow source')")
        self.assertNotEqual(self.run_gate().returncode, 0)


if __name__ == "__main__":
    unittest.main()
