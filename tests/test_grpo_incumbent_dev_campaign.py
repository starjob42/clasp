"""Synthetic scheduler, native guard, and raw provenance tests; no jobs or models."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("incumbent_campaign", ROOT / "scripts/grpo_incumbent_dev_campaign.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("train", "evaluation", "logs", "ledgers", "ledgers/events", "ledgers/terminal",
                     "ledgers/evaluation_events", "ledgers/evaluation_terminal", "source"):
            (self.root / name).mkdir()
        self.source = self.root / "source"
        (self.source / "SOURCE_MANIFEST.sha256").write_text("test manifest\n")
        initial = {task: {"adapter_model.safetensors": {"path": "/fixed/" + task + "/checkpoint-31250/adapter_model.safetensors"}}
                   for task in control.TASKS}
        train_plans = control.plans(self.root, self.source, initial)
        self.protocol = {"receipt_sha256": "fixed", "created_at_utc": "2026-09-07T08:00:00+00:00",
                         "planned_train_commands": train_plans,
                         "planned_eval_commands": control.evaluation_plans(self.root, self.source, train_plans)}
        for patcher in (patch.object(control, "protocol_for", return_value=(self.protocol, self.source)),
                        patch.object(control, "check_diagnostics", return_value={"passed": True})):
            patcher.start(); self.addCleanup(patcher.stop)
        self.commands, self.accounting, self.occupied = [], "", 5

    def runner(self, command):
        if command[0] == "sacct":
            return self.accounting
        if command[0] == "squeue":
            return "".join(f"879217_{i}|PENDING\n" for i in range(self.occupied))
        if command[0] == "sbatch":
            self.commands.append(command)
            return str(1900000 + len(self.commands)) + "\n"
        raise AssertionError(command)

    def bind(self, item, job="1900001", *, role="train", complete=True):
        payload = {"key": item["key"], "command": item["command"], "protocol_sha256": "fixed", "transaction": item["key"]}
        directory = self.root / "ledgers" / ("events" if role == "train" else "evaluation_events")
        control.write_new(directory / (job + "_intent.json"), {**payload, "kind": "intent"}, "event_sha256")
        if complete:
            control.write_new(directory / (job + "_submitted.json"), {**payload, "kind": "submitted", "job_id": job}, "event_sha256")

    def test_six_unique_train_and_evaluation_plans_bind_all_method_budgets(self):
        self.assertEqual(len(self.protocol["planned_train_commands"]), 6)
        self.assertEqual(len(self.protocol["planned_eval_commands"]), 6)
        for item in self.protocol["planned_train_commands"]:
            export = next(arg for arg in item["command"] if arg.startswith("--export="))
            env = dict(value.split("=", 1) for value in export.split(",")[1:])
            for name, value in (("DEV_MAX_STEPS", "100"), ("DEV_SEED", "43"),
                                ("DEV_SAVE_STEPS", "100"), ("DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES", "0"),
                                ("DEV_RESUME_CHECKPOINT", ""), ("DEV_GROUP_STD_CORRECTION", "1")):
                self.assertEqual(env[name], value)
            self.assertEqual(env["DEV_SOURCE_MANIFEST_SHA256"], control.identity(self.source / "SOURCE_MANIFEST.sha256")["sha256"])
            self.assertEqual(env["DEV_TEACHER_NORMALIZATION"], "optimizer_window" if item["method"] == "grpo_verified" else "microbatch")

    def test_expanded_unrelated_array_and_scheduler_lag_limit_first_tick_to_five(self):
        result = control.reconcile(self.root, submit=True, record=True, runner=self.runner)
        self.assertEqual(len(self.commands), 5)
        self.assertEqual([row["state"] for row in result["observations"]], ["submitted"] * 5 + ["ready"])
        self.accounting = "".join(f"{1900001+i}|PENDING|0:0|{plan['stdout']}|0\n"
                                  for i, plan in enumerate(self.protocol["planned_train_commands"][:5]))
        control.reconcile(self.root, submit=True, record=True, runner=self.runner)
        self.assertEqual(len(self.commands), 5)

    def test_explicit_per_tick_cap_and_one_free_slot_are_obeyed(self):
        control.reconcile(self.root, submit=True, runner=self.runner, max_new_jobs=2)
        self.assertEqual(len(self.commands), 2)

    def test_unknown_job_with_matching_stdout_cannot_use_an_orphan_intent(self):
        item = self.protocol["planned_train_commands"][0]
        self.bind(item, complete=False)
        self.accounting = f"1900099|RUNNING|0:0|{item['stdout']}|10\n"
        with self.assertRaisesRegex(ValueError, "submitted-event job ID"):
            control.reconcile(self.root, submit=True, runner=self.runner)
        self.assertFalse(self.commands)

    def test_orphan_intent_never_retries(self):
        item = self.protocol["planned_train_commands"][0]
        self.bind(item, complete=False)
        result = control.reconcile(self.root, submit=True, runner=self.runner)
        self.assertEqual(result["observations"][0]["state"], "prior_attempt_awaiting_accounting_no_retry")
        self.assertNotIn(item["command"], self.commands)

    def test_other_roles_unresolved_attempt_reserves_capacity(self):
        self.occupied = 9
        self.bind(self.protocol["planned_eval_commands"][0], role="eval", complete=False)
        control.reconcile(self.root, submit=True, runner=self.runner)
        self.assertFalse(self.commands)

    def test_missing_or_failed_native_prerequisite_blocks_every_submission(self):
        for observations in ([], [{"passed": False, "state": "FAILED"}] * 4):
            with self.subTest(observations=observations):
                with patch.object(control, "check_diagnostics", return_value={"passed": False, "observations": observations}):
                    with self.assertRaisesRegex(ValueError, "correctness prerequisites"):
                        control.reconcile(self.root, submit=True, runner=self.runner)
        self.assertFalse(self.commands)

    def test_pending_native_prerequisite_waits_without_any_submission_or_ledger(self):
        observations = [{"passed": True, "state": "COMPLETED"}, {"passed": False, "state": "RUNNING"},
                        {"passed": False, "state": "PENDING"}, {"passed": False, "state": "PENDING"}]
        with patch.object(control, "check_diagnostics", return_value={"passed": False, "observations": observations}):
            result = control.reconcile(self.root, submit=True, record=True, runner=self.runner)
        self.assertTrue(all(row["state"] == "waiting_for_native_correctness_prerequisites" for row in result["observations"]))
        self.assertFalse(result["all_training_verified"])
        self.assertFalse(self.commands)
        self.assertFalse(list((self.root / "ledgers/events").iterdir()))

    def test_one_job_cannot_bind_both_training_and_evaluation_roles(self):
        self.bind(self.protocol["planned_train_commands"][0])
        self.bind(self.protocol["planned_eval_commands"][0], role="eval")
        with self.assertRaisesRegex(ValueError, "multiple campaign roles"):
            control.reconcile(self.root, runner=self.runner)

    def test_readonly_status_creates_no_events_and_eval_waits_for_training(self):
        result = control.reconcile(self.root, runner=self.runner, role="eval")
        self.assertTrue(all(row["state"] == "waiting_for_verified_training" for row in result["observations"]))
        self.assertFalse(self.commands)
        self.assertFalse(list((self.root / "ledgers/events").iterdir()))
        self.assertFalse(list((self.root / "ledgers/evaluation_events").iterdir()))

    def test_failed_training_endpoint_is_not_validated_or_retried(self):
        item = self.protocol["planned_train_commands"][0]
        self.bind(item)
        self.accounting = f"1900001|FAILED|1:0|{item['stdout']}|100\n"
        validator = unittest.mock.Mock(side_effect=AssertionError("must not validate failed job"))
        result = control.reconcile(self.root, runner=self.runner, validate=validator)
        self.assertEqual(result["observations"][0]["state"], "FAILED")
        validator.assert_not_called()


class NativeDiagnosticAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [{"key": key, "job_id": str(1950000 + i), "output": "/diagnostic/" + key,
                      "stdout": "/diagnostic/logs/" + key + "_%j.out"}
                     for i, key in enumerate(control.DIAGNOSTIC_JOBS)]
        self.expected = {"protocol": {"path": "/diagnostic/protocol.json"}, "jobs": self.jobs}
        self.protocol = {"diagnostic_prerequisite": self.expected}

    def check(self, raw):
        with patch.object(control, "declare_diagnostics", return_value=self.expected):
            return control.check_diagnostics(self.protocol, Path("/source"), runner=lambda command: raw)

    def rows(self, *, state="PENDING", overrides=None):
        result = []
        for job in self.jobs:
            fields = [job["job_id"], state, "0:0", job["stdout"], "400" if state == "COMPLETED" else "0",
                      "25", "cpu=32,gres/gpu=4", control.QOS, "vas@a100", "gpu_p5"]
            for index, value in (overrides or {}).items():
                fields[index] = value
            result.append("|".join(fields))
        return "\n".join(result)

    def test_four_task_method_gates_wait_when_pending(self):
        self.assertEqual(set(control.DIAGNOSTIC_JOBS), {task + "_" + method for task in control.TASKS
                         for method in ("grpo_verified", "foarl_f638_dev")})
        result = self.check(self.rows())
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["observations"]), 4)

    def test_unfilled_source_or_job_pin_fails_before_filesystem_admission(self):
        with patch.object(control, "DIAGNOSTIC_SOURCE_SHA256", "PENDING"):
            with self.assertRaisesRegex(ValueError, "pins are not yet declared"):
                control.declare_diagnostics(Path("/nonexistent"), Path("/nonexistent"))

    def test_wrong_native_walltime_gpu_queue_or_stdout_fails_before_model_reads(self):
        for index, value in ((5, "120"), (6, "cpu=32,gres/gpu=1"), (7, "other-qos"),
                              (8, "other-account"), (9, "gpu_p6"), (3, "/other/log"), (4, "1501")):
            with self.subTest(index=index, value=value):
                with self.assertRaisesRegex(ValueError, "actual native diagnostic allocation"):
                    self.check(self.rows(state="COMPLETED", overrides={index: value}))

    def test_unknown_or_duplicate_native_job_is_rejected(self):
        for raw in (self.rows(overrides={0: "1999999"}), self.rows() + "\n" + self.rows().splitlines()[0]):
            with self.assertRaisesRegex(ValueError, "invalid native diagnostic accounting row"):
                self.check(raw)


class ExactReferenceLineageTests(unittest.TestCase):
    """Strict inspector results still need complete sibling lineage binding."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.train = Path(temporary.name)
        self.checkpoint = self.train / "checkpoints/checkpoint-100"
        self.checkpoint.mkdir(parents=True)
        for root in (self.train, self.checkpoint):
            for name in ("pre.json", "step1.json", "post.json"):
                (root / name).write_text(name)
        self.contract = {"initialization": {"checkpoint": "/sft", "base_model": "/base",
                         "adapter_config_sha256": "config", "checkpoint_fidelity_sha256": "authority"}}
        self.hp = {"stage_hyperparameters": {"training_run_contract_identity": {"file_sha256": "run"}}}
        self.pre = {"protocol": "pre", "status": "passed", "receipt_sha256": "pre-self",
                    "source_binding": {"adapter_path": "/sft", "tokenizer_path": "/sft", "base_model_path": "/base",
                                       "adapter_config_sha256": "config", "authoritative_receipt_file_sha256": "authority"},
                    "training_contract": {"beta": .05, "dtype": "bfloat16"}, "distributed": {"world_size": 4},
                    "runtime_pair": {"tensor_count": 3, "parameter_count": 5, "tensor_inventory_sha256": "reference",
                                     "model_class": "Model", "active_adapter": "default", "model_config_sha256": "model-config",
                                     "prepared_runtime": {"reference_prepared_class": "Prepared", "reference_unwrapped_class": "Model",
                                                          "device_types": ["cuda"]}}}
        self.expected_pre = {"protocol": "pre", "status": "passed", "receipt_sha256": "pre-self",
                             "receipt_file_sha256": control.identity(self.train / "pre.json")["sha256"],
                             "reference_tensor_count": 3, "reference_parameter_count": 5,
                             "reference_tensor_inventory_sha256": "reference", "model_class": "Model",
                             "prepared_class": "Prepared", "unwrapped_class": "Model", "active_adapter": "default",
                             "model_config_sha256": "model-config", "device_types": ["cuda"]}
        self.expected_step1 = {"protocol": "post", "status": "passed", "receipt_sha256": "step1-self",
                               "receipt_file_sha256": control.identity(self.train / "step1.json")["sha256"],
                               "global_step": 1, "reference_tensor_inventory_sha256": "reference",
                               "policy_trainable_tensor_inventory_sha256": "policy-at-step1"}
        self.step1 = {"receipt_sha256": "step1-self", "pre_step_reference": copy.deepcopy(self.expected_pre),
                      "training_run_contract": copy.deepcopy(self.hp["stage_hyperparameters"]["training_run_contract_identity"])}
        self.terminal = {**copy.deepcopy(self.step1), "receipt_sha256": "post-self",
                         "step1_reference": copy.deepcopy(self.expected_step1)}
        self.reference = SimpleNamespace(RECEIPT_FILENAME="pre.json",
                    inspect_runtime_receipt=unittest.mock.Mock(side_effect=lambda *a, **kw: copy.deepcopy(self.pre)))
        self.post = SimpleNamespace(STEP1_RECEIPT_FILENAME="step1.json", POST_STEP_RECEIPT_FILENAME="post.json",
                    inspect_post_step_receipt=unittest.mock.Mock(side_effect=lambda p, **kw:
                        copy.deepcopy(self.terminal if Path(p).name == "post.json" else self.step1)),
                    step1_reference_binding=unittest.mock.Mock(side_effect=lambda *a, **kw: copy.deepcopy(self.expected_step1)))

    def check(self):
        with patch.object(control, "exact_reference_modules", return_value=(self.reference, self.post)):
            return control.exact_reference_evidence(self.train, self.hp, self.contract, self.train)

    def test_exact_lineage_passes_and_retains_all_five_file_identities(self):
        result = self.check()
        self.assertEqual(len(result["files"]), 5)
        self.reference.inspect_runtime_receipt.assert_any_call(self.train / "pre.json", claim_scope="development", rehash_sources=True)
        self.post.step1_reference_binding.assert_called_once_with(self.train / "step1.json", claim_scope="development", rehash_sources=False)
        before = result["files"]["pre.json"]
        (self.train / "pre.json").write_text("changed")
        self.assertNotEqual(before, control.identity(self.train / "pre.json"))

    def test_same_receipt_hashes_cannot_hide_changed_pre_reference_projection(self):
        for target in (self.step1, self.terminal):
            for key, value in (("reference_tensor_inventory_sha256", "other"), ("model_class", "Other"),
                               ("reference_tensor_count", 4), ("active_adapter", "other")):
                with self.subTest(target=target["receipt_sha256"], key=key):
                    previous = target["pre_step_reference"][key]
                    target["pre_step_reference"][key] = value
                    with self.assertRaisesRegex(ValueError, "source/run lineage"):
                        self.check()
                    target["pre_step_reference"][key] = previous

    def test_same_step1_receipt_hashes_cannot_hide_other_policy_inventory(self):
        self.terminal["step1_reference"]["policy_trainable_tensor_inventory_sha256"] = "other-policy"
        with self.assertRaisesRegex(ValueError, "another step-one policy"):
            self.check()

    def test_malformed_pre_receipt_cannot_skip_strict_inspection(self):
        self.reference.inspect_runtime_receipt.side_effect = ValueError("malformed exact-reference receipt")
        with self.assertRaisesRegex(ValueError, "malformed exact-reference"):
            self.check()
        self.post.inspect_post_step_receipt.assert_not_called()


class ConfigurationProjectionTests(unittest.TestCase):
    """Pairing must compare unknown fields after each arm's separate validation."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = {"receipt_sha256": "protocol", "planned_train_commands": []}
        self.hp, self.audits = {}, {}
        for method in control.METHODS:
            root = self.root / method; root.mkdir()
            self.hp[method] = {
                "trainer_config": {"output_dir": str(root), "logging_dir": str(root / "logs"), "run_name": method,
                                   "learning_rate": 1e-6, "beta": .05 if method == "foarl_f638_dev" else 0.,
                                   "scale_rewards": "group" if method == "foarl_f638_dev" else "none",
                                   "reward_weights": [1., 1.] if method == "foarl_f638_dev" else None,
                                   "loss_type": "grpo", "future_numeric_setting": 7},
                "stage_hyperparameters": {"training_run_contract_path": str(root / "contract.json"),
                                          "training_run_contract_identity": method, "max_completion_length": 3000},
                "args": {"output_dir": str(root), "generation_top_p": .9},
                "dataset": {"train_size": 2000}, "context": {"window": 32768}, "distributed": {"world_size": 4}}
            contract = {"runtime": {"torch": "same"}, "compute": {"gpus": 4}, "initialization": {"weights": "same"}, "source": {"hash": "same"}}
            (root / "contract.json").write_text(json.dumps(contract))
            self.audits[method] = {"key": "cvrp/" + method, "protocol_sha256": "protocol",
                                   "contract": control.identity(root / "contract.json")}
            self.protocol["planned_train_commands"].append({"key": "cvrp/" + method})

    def compare(self, method):
        for name, hp in self.hp.items():
            path = self.root / name / "hp.json"; path.write_text(json.dumps(hp))
            self.audits[name]["hyperparams"] = control.identity(path)
        with patch.object(control, "validate_actual_arm"):
            return control.compare_actual_configuration("cvrp", self.audits["grpo_z"], self.audits[method], self.protocol, self.root)

    def test_matching_all_fields_and_declared_foarl_differences_pass(self):
        self.compare("grpo_verified")
        self.compare("foarl_f638_dev")

    def test_unknown_trainer_field_mismatch_fails_for_each_pair(self):
        for method in control.METHODS[1:]:
            with self.subTest(method=method):
                self.hp[method]["trainer_config"]["future_numeric_setting"] = 8
                with self.assertRaisesRegex(ValueError, "future_numeric_setting"):
                    self.compare(method)
                self.hp[method]["trainer_config"]["future_numeric_setting"] = 7

    def test_prompt_token_budget_stage_mismatch_fails(self):
        self.hp["foarl_f638_dev"]["stage_hyperparameters"]["max_completion_length"] = 1000
        with self.assertRaisesRegex(ValueError, "max_completion_length"):
            self.compare("foarl_f638_dev")

    def test_sampler_argument_mismatch_fails(self):
        self.hp["grpo_verified"]["args"]["generation_top_p"] = .7
        with self.assertRaisesRegex(ValueError, "generation_top_p"):
            self.compare("grpo_verified")


class WrapperContractTests(unittest.TestCase):
    def test_raw_generation_assignments_match_the_completed_comparison(self):
        previous = (ROOT / "scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm").read_text()
        current = (ROOT / control.EVAL_WRAPPER).read_text()
        prefixes = ("export EVAL_START_INDEX=", "export SEED=", "export TEMPERATURE=", "export MAX_NEW_TOKENS=")
        for prefix in prefixes:
            self.assertEqual(next(line for line in current.splitlines() if line.startswith(prefix)),
                             next(line for line in previous.splitlines() if line.startswith(prefix)))
        self.assertIn("EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp", current)
        self.assertIn('bash "${SOURCE}/scripts/evaluate_a100.slurm"', current)
        self.assertIn("#SBATCH --gres=gpu:1", current)
        self.assertIn("#SBATCH --time=02:00:00", current)

    def test_all_three_comparisons_are_declared_in_summary(self):
        import inspect
        code = inspect.getsource(control.summarize_campaign)
        for pair in ('("grpo_z", "grpo_verified")', '("grpo_z", "foarl_f638_dev")', '("foarl_f638_dev", "grpo_verified")'):
            self.assertIn(pair, code)
        self.assertLess(code.index("evaluation_validation("), code.index("directory.mkdir()"))
        self.assertIn("summary.paired_quality(a, b)", code)

class PolicyConsistencyReceiptTests(unittest.TestCase):
    def setUp(self):
        self.steps = [1, 2, 5, *range(10, 101, 10)]
        self.inventory = "a" * 64
        self.consensus_sha = "b" * 64
        self.receipt = {
            "schema_version": 1,
            "protocol": "clasp-optimizer-step-exact-policy-consistency-v1",
            "claim_scope": "development", "status": "passed",
            "settings": {"enabled": True, "gradient_enabled": False, "initial_optimizer_steps": [1, 2, 5], "interval": 10},
            "verified_optimizer_steps": self.steps,
            "checks": [{"optimizer_step": step, "tensor_inventory_sha256": self.inventory,
                        "distributed": {"phase": f"optimizer_step_{step}_policy_consistency", "world_size": 4,
                                        "consensus_digest": self.consensus_sha,
                                        "rank_digests": [{"rank": rank, "digest": self.consensus_sha} for rank in range(4)]}}
                       for step in self.steps],
        }
        self.fidelity = {"trainable_closure": {"tensor_inventory_sha256": self.inventory}}

    def check(self, receipt=None, *, resign=True):
        receipt = copy.deepcopy(self.receipt if receipt is None else receipt)
        if resign:
            receipt.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = control.digest(receipt)
        control.validate_policy_consistency(receipt, self.fidelity)

    def test_complete_four_rank_terminal_evidence_passes(self):
        self.check()

    def test_step_list_without_underlying_checks_is_rejected(self):
        self.receipt["checks"] = []
        with self.assertRaisesRegex(ValueError, "checks are incomplete"):
            self.check()

    def test_single_rank_check_cannot_certify_distributed_training(self):
        self.receipt["checks"][0]["distributed"]["world_size"] = 1
        with self.assertRaisesRegex(ValueError, "four-rank"):
            self.check()

    def test_missing_rank_is_rejected(self):
        self.receipt["checks"][0]["distributed"]["rank_digests"].pop()
        with self.assertRaisesRegex(ValueError, "missing distributed rank digests"):
            self.check()

    def test_duplicate_rank_is_rejected(self):
        self.receipt["checks"][0]["distributed"]["rank_digests"][3]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "do not agree"):
            self.check()

    def test_different_rank_policy_digest_is_rejected(self):
        self.receipt["checks"][0]["distributed"]["rank_digests"][2]["digest"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "do not agree"):
            self.check()

    def test_missing_terminal_optimizer_step_is_rejected(self):
        self.receipt["verified_optimizer_steps"] = self.steps[:-1]
        with self.assertRaisesRegex(ValueError, "declared optimizer steps"):
            self.check()

    def test_reordered_underlying_checks_are_rejected(self):
        self.receipt["checks"][0], self.receipt["checks"][1] = self.receipt["checks"][1], self.receipt["checks"][0]
        with self.assertRaisesRegex(ValueError, "order/step"):
            self.check()

    def test_phase_must_match_optimizer_step(self):
        self.receipt["checks"][0]["distributed"]["phase"] = "pre_step"
        with self.assertRaisesRegex(ValueError, "phase"):
            self.check()

    def test_final_inventory_must_match_exact_saved_policy(self):
        self.receipt["checks"][-1]["tensor_inventory_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "certified checkpoint"):
            self.check()

    def test_invalid_consensus_or_inventory_hash_is_rejected(self):
        for field in ("tensor_inventory_sha256", "consensus_digest"):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                target = receipt["checks"][0] if field == "tensor_inventory_sha256" else receipt["checks"][0]["distributed"]
                target[field] = "not-a-sha256"
                with self.assertRaises(ValueError):
                    self.check(receipt)

    def test_wrong_protocol_scope_version_and_status_are_rejected(self):
        for field, value in (("protocol", "gradient-check"), ("claim_scope", "publication"),
                             ("schema_version", 2), ("schema_version", True), ("status", "failed")):
            with self.subTest(field=field, value=value):
                receipt = {**self.receipt, field: value}
                with self.assertRaisesRegex(ValueError, "protocol/status/scope"):
                    self.check(receipt)

    def test_disabled_wrong_interval_or_unexpected_gradient_mode_is_rejected(self):
        for field, value in (("enabled", False), ("gradient_enabled", True), ("interval", 20), ("initial_optimizer_steps", [1, 5])):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt["settings"][field] = value
                with self.assertRaisesRegex(ValueError, "fixed production guard"):
                    self.check(receipt)

    def test_bool_cannot_masquerade_as_optimizer_step_or_rank(self):
        for location in ("step", "rank"):
            with self.subTest(location=location):
                receipt = copy.deepcopy(self.receipt)
                if location == "step":
                    receipt["verified_optimizer_steps"][0] = True
                else:
                    receipt["checks"][0]["distributed"]["rank_digests"][0]["rank"] = False
                with self.assertRaises(ValueError):
                    self.check(receipt)

    def test_passed_receipt_cannot_retain_failure_diagnostics(self):
        for field, value in (("failed_optimizer_step", 100), ("error", "mismatch"), ("policy_mismatch_details", {})):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "failure evidence"):
                    self.check({**self.receipt, field: value})

    def test_invalid_self_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self hash"):
            self.check({**self.receipt, "receipt_sha256": "d" * 64}, resign=False)

    def test_missing_and_malformed_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy_consistency_checks.json"
            with self.assertRaises(FileNotFoundError):
                control.signed(path)
            path.write_text("{not JSON}")
            with self.assertRaises(json.JSONDecodeError):
                control.signed(path)



class CrossBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        train, output, source, data = [root / name for name in ("train", "output", "source", "data")]
        checkpoint = train / "checkpoints/checkpoint-100"
        for path in (checkpoint, output, source / "src/clasp/entrypoints", data):
            path.mkdir(parents=True)
        files = {}
        for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json"):
            (checkpoint / name).write_text(name)
            files[name] = control.identity(checkpoint / name)
        details, config_path = output / "details.jsonl", output / "config.json"
        details.write_text("synthetic\n")
        config_path.write_text("synthetic\n")
        evaluator = source / "src/clasp/entrypoints/evaluate.py"
        evaluator.write_text("synthetic\n")
        code_file = control.identity(evaluator)
        code_files = [{"path": "entrypoints/evaluate.py", "sha256": code_file["sha256"], "size": code_file["size_bytes"]}]
        code = {"files": code_files, "manifest_sha256": control.digest(code_files)}
        artifact_files = [{"path": k, "sha256": v["sha256"], "size": v["size_bytes"]} for k, v in files.items()]
        artifact = {"kind": "local_directory", "basename": "checkpoint-100", "files": artifact_files,
                    "manifest_sha256": control.digest(artifact_files)}
        generation = {"seed": 271828, "num_return_sequences": 8}
        extra = {"seed_policy": "per_example_sha256_v1", "temperature_schedule": [], "top_p_schedule": [],
                 "op_size_band_top_p_schedule": [], "op_size_band_top_p_min_size": 0, "op_size_band_top_p_max_size": 0}
        benchmark = {"files": {"cvrp": {"dataset": {"sha256": "dataset", "size_bytes": 20}}},
                     "ordered_instances": {"cvrp": [{"example_index": i} for i in range(100)]}}
        training = {"source_manifest": {"sha256": "source"}, "evaluation": {"data_root": str(data)}}
        evaluation_p = {"management": {"helper": "helper", "wrapper": "wrapper"}, "generation": generation}
        terminal = {"contract": {"sha256": "contract"}, "checkpoint": files}
        plan = {"key": "cvrp/grpo_z", "task": "cvrp", "method": "grpo_z", "checkpoint": str(checkpoint),
                "training_root": str(train), "output": str(output)}
        slice_ = {"split": "eval", "start": 0, "count": 100}
        launch = {"slurm_job_id": "1826000", "task": "cvrp", "method": "grpo_z", "stage": "grpo",
                  "source_manifest": training["source_manifest"], "training_contract": terminal["contract"],
                  "checkpoint": files, "benchmark": benchmark, "management": evaluation_p["management"],
                  "generation": generation, "repair": False, "evaluation_slice": slice_,
                  "reference_semantics": {"cvrp": "gap_to_verified_heuristic_reference"}, "launch_sha256": "launch"}
        index = {"task": "cvrp", "method": "grpo_z", "repair": False, "evaluation_slice": slice_,
                 "reference_semantics": "gap_to_verified_heuristic_reference", "index_sha256": "index",
                 "run_config": control.identity(config_path), "details": control.identity(details)}
        config = {"stage": "grpo", "tasks": ["cvrp"], "split": "eval", "run_status": "complete",
                  "evaluation_purpose": "development", "resolved_model_path": str(checkpoint),
                  "resolved_run_dir": str(train), "data_root": str(data), "results_dir": str(output),
                  "files": {"run_config": str(config_path), "details": str(details)},
                  "generation": {**generation, **extra, "best_of_n": 8},
                  "prompting": {"prompt_style_resolved": "jiang_f638", "no_reason": True,
                                "clasp_mis_jiang_response_output": "0", "clasp_mis_relabel_order": "", "tokenizer_path": str(checkpoint)},
                  "limits": {"eval_start_index": 0, "max_eval_samples_per_task": 100, "prefer_subset_files": False},
                  "code_identity": code, "run_config_sha256": "config",
                  "experiment_provenance": {"generation": {**generation, **extra}, "code": code,
                      "model": {"checkpoint_artifact": artifact}, "data": {"files": [{"path": "cvrp/eval/test.json",
                      "sha256": "dataset", "size": 20, "split": "eval", "task": "cvrp"}]}}}
        records = {i: SimpleNamespace(example_index=i, raw={"example_index": i, "task": "cvrp", "split": "eval",
                   "num_return_sequences": 8, "candidates": [{}] * 8}) for i in range(100)}
        self.args = [plan, launch, config, index, terminal]
        self.kwargs = dict(job_id="1826000", training=training, evaluation=evaluation_p, benchmark=benchmark, source=source, records=records)

    def test_exact_cross_bindings_pass(self):
        self.assertEqual(control.audit_evaluation_arm(*self.args, **self.kwargs)["status"], "all_cross_bindings_verified")

    def test_mismatch_cases_are_rejected(self):
        cases = [(1, ("slurm_job_id",), "1825999", "launch job"),
                 (1, ("method",), "grpo_verified", "method/task"),
                 (2, ("resolved_model_path",), self.args[0]["training_root"], "raw run model"),
                 (2, ("data_root",), self.args[0]["training_root"], "data root"),
                 (2, ("generation", "seed"), 42, "generation"),
                 (2, ("files", "details"), self.args[3]["run_config"]["path"], "artifact path"),
                 (3, ("reference_semantics",), "gap_to_certified_optimum", "reference semantics")]
        for index, path, value, message in cases:
            with self.subTest(path=path):
                args = copy.deepcopy(self.args)
                node = args[index]
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    control.audit_evaluation_arm(*args, **self.kwargs)
