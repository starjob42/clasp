"""Management-only evaluator changes; all inputs are tiny local fixtures."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


evaluation = module("evaluation_v2", ROOT / "scripts/evaluate_grpo_verified_ddpfix_v2.py")
audit = module("evaluation_binding_audit", ROOT / "analysis_outputs/grpo_verified_ddpfix_20260906_v2/audit_final_evaluation_bindings.py")


class WrapperAndDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_identical_slurm_spool_bytes_are_accepted_despite_different_path(self):
        managed, spool = self.root / "managed.sh", self.root / "slurm_script"
        managed.write_text("#!/bin/bash\ntrue\n")
        spool.write_bytes(managed.read_bytes())
        self.assertNotEqual(evaluation.identity(managed), evaluation.identity(spool))
        evaluation.validate_running_wrapper(spool, evaluation.identity(managed))

    def test_changed_spool_bytes_fail_even_with_same_size(self):
        managed, spool = self.root / "managed.sh", self.root / "slurm_script"
        managed.write_text("true\n")
        spool.write_text("oops\n")
        with self.assertRaisesRegex(ValueError, "running management wrapper changed"):
            evaluation.validate_running_wrapper(spool, evaluation.identity(managed))

    def test_wrapper_changes_only_method_and_helper(self):
        old = (ROOT / evaluation.WRAPPER).read_text()
        new = evaluation.patched_wrapper(old)
        restored = new.replace("grpo_z|grpo_verified|v3|v4)", "grpo_z|v3|v4)").replace(
            'HELPER="${MANAGEMENT}/scripts/evaluate_grpo_verified_ddpfix_v2.py"',
            'HELPER="${MANAGEMENT}/scripts/clasp_grpo_fresh_dev_artifacts.py"')
        self.assertEqual(restored, old)

    def test_new_declaration_preserves_original_and_pins_textual_delta(self):
        campaign, source = self.root / "campaign", self.root / "source"
        (campaign / "ledgers").mkdir(parents=True)
        (campaign / "evaluation").mkdir()
        sentinel = campaign / "ledgers/evaluation_protocol.json"
        sentinel.write_text("retained original declaration\n")
        (source / "scripts").mkdir(parents=True)
        for relative in (evaluation.ORIGIN, evaluation.WRAPPER):
            (source / relative).write_bytes((ROOT / relative).read_bytes())
        training = {"receipt_sha256": "training", "runtime": str(self.root / "runtime"),
                    "evaluation": {"data_root": str(self.root / "data")}, "source_manifest": {"sha256": "source"},
                    "planned_train_commands": [{"key": task + "/" + method, "task": task, "method": method,
                    "training_root": str(campaign / "train" / (task + "_" + method))}
                    for task in evaluation.TASKS for method in evaluation.METHODS]}
        evaluation.write_new(campaign / "ledgers/protocol.json", training)
        fresh = SimpleNamespace(GENERATION={"seed": 271828})
        benchmark = {"manifest_sha256": "benchmark", "audit_sha256": "overlap"}
        auditor = ROOT / "analysis_outputs/grpo_verified_ddpfix_20260906_v2/audit_final_evaluation_bindings.py"
        with patch.object(evaluation, "environment", return_value=(training, source, None, fresh, benchmark)):
            declared = evaluation.initialize(campaign, auditor)
            evaluation.protocol_for(campaign)
            self.assertTrue(all("/evaluation_v2/" in p["output"] for p in declared["plans"]))
            self.assertEqual(sentinel.read_text(), "retained original declaration\n")
            self.assertEqual(declared["controller_patch"]["origin"]["sha256"], evaluation.ORIGIN_SHA256)
            delta = Path(declared["controller_patch"]["delta"]["path"])
            self.assertIn("validate_running_wrapper", delta.read_text())
            delta.chmod(0o640)
            delta.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "textual delta"):
                evaluation.protocol_for(campaign)


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
            files[name] = audit.identity(checkpoint / name)
        details, config_path = output / "details.jsonl", output / "config.json"
        details.write_text("synthetic\n")
        config_path.write_text("synthetic\n")
        evaluator = source / "src/clasp/entrypoints/evaluate.py"
        evaluator.write_text("synthetic\n")
        code_file = audit.identity(evaluator)
        code_files = [{"path": "entrypoints/evaluate.py", "sha256": code_file["sha256"], "size": code_file["size_bytes"]}]
        code = {"files": code_files, "manifest_sha256": audit.digest(code_files)}
        artifact_files = [{"path": k, "sha256": v["sha256"], "size": v["size_bytes"]} for k, v in files.items()]
        artifact = {"kind": "local_directory", "basename": "checkpoint-100", "files": artifact_files,
                    "manifest_sha256": audit.digest(artifact_files)}
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
                 "run_config": audit.identity(config_path), "details": audit.identity(details)}
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
        self.assertEqual(audit.audit_arm(*self.args, **self.kwargs)["status"], "all_cross_bindings_verified")

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
                    audit.audit_arm(*args, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
