from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import clasp.entrypoints.evaluate as evaluate_module
from clasp.config import _apply_clasp_strict_defaults
from clasp.entrypoints.evaluate import (
    _aggregate_records,
    _build_experiment_provenance,
    _build_instance_record,
    _dataset_content_identity,
    _example_sampling_seed,
    _experiment_fingerprint,
    _nearest_run_dir,
    _limit_prompt_length,
    _protocol_fingerprint_material,
    _publication_exact_runtime_identity,
    _read_detail_records_for_resume,
    _resolve_prompt_no_reason,
    _resolve_max_new_tokens,
    _restore_stage_representation_env,
    _results_label,
    _resume_record_matches_example,
    _resume_records_by_key,
    _run_selection_score,
    _self_hash_payload,
    _stage_prompt_style,
    _training_run_contract_identity,
    _validate_resume_run_config,
    _write_json_atomic,
    build_eval_parser,
)


class _FakeConfig:
    _commit_hash = "f" * 40

    def to_dict(self):
        return {"model_type": "fake", "hidden_size": 4}


class _FakeModel:
    config = _FakeConfig()


class _FakeTokenizer:
    special_tokens_map = {"eos_token": "</s>"}
    chat_template = "{{ messages }}"
    init_kwargs = {"_commit_hash": "e" * 40}

    def get_vocab(self):
        return {"a": 0, "b": 1}


class EvaluateContractTests(unittest.TestCase):
    def test_jiang_evaluation_uses_released_generation_budget(self):
        stage = {
            "max_seq_length": 20_000,
            "evaluation_max_new_tokens": 3_000,
            "evaluation_max_prompt_length": 128_072,
        }
        args = SimpleNamespace(max_new_tokens=0)
        self.assertEqual(_resolve_max_new_tokens(args, stage), 3_000)
        self.assertEqual(_limit_prompt_length(0, stage, 131_072, 3_000), 128_072)

    def test_evaluation_prompt_variant_can_differ_from_foarl_training_variant(self):
        stage = {"no_reason": False, "evaluation_no_reason": True}
        self.assertTrue(
            _resolve_prompt_no_reason(
                stage,
                force_reasoning_prompt=False,
                force_no_reason_prompt=False,
            )
        )

    def test_jiang_stage_prompt_style_never_falls_back_to_generic_alpaca(self):
        self.assertEqual(_stage_prompt_style({"prompt_style": "jiang_f638"}), "jiang_f638")
        self.assertEqual(
            _stage_prompt_style({"prompt_template": "jiang_f638_no_reason"}),
            "jiang_f638",
        )
        parser = build_eval_parser()
        prompt_action = next(action for action in parser._actions if action.dest == "prompt_style")
        self.assertIn("jiang_f638", prompt_action.choices)

    def test_detail_and_summary_retain_compute_denominators(self):
        candidates = [
            {"is_feasible": True, "gap_to_ground_truth": 0.01, "completion_tokens": 7},
            {"is_feasible": False, "gap_to_ground_truth": None, "completion_tokens": 5},
        ]
        record = _build_instance_record(
            task="tsp",
            split="eval",
            example_index=0,
            source_json=None,
            source_json_files=[],
            instruction="instruction",
            input_text="input",
            answer="answer",
            instance_size=3,
            prompt_tokens=11,
            prompt_was_truncated=False,
            generation_time_sec=1.5,
            num_return_sequences=2,
            candidates=candidates,
            peak_gpu_memory_allocated_bytes=123,
            peak_gpu_memory_reserved_bytes=456,
        )
        self.assertEqual(record["generated_tokens"], 12)
        summary = _aggregate_records([record])
        self.assertEqual(summary["generated_tokens"], 12)
        self.assertEqual(summary["peak_gpu_memory_allocated_bytes"], 123)
        self.assertEqual(summary["peak_gpu_memory_reserved_bytes"], 456)

    def test_prompt_override_is_symmetric_and_mutually_exclusive(self):
        parser = build_eval_parser()
        no_reason = parser.parse_args(["--force_no_reason_prompt"])
        reasoning = parser.parse_args(["--force_reasoning_prompt"])
        self.assertTrue(no_reason.force_no_reason_prompt)
        self.assertFalse(no_reason.force_reasoning_prompt)
        self.assertTrue(reasoning.force_reasoning_prompt)
        self.assertFalse(reasoning.force_no_reason_prompt)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--force_no_reason_prompt", "--force_reasoning_prompt"])

    def test_prompt_contract_fails_closed_without_metadata_or_override(self):
        with self.assertRaisesRegex(RuntimeError, "refuses to guess"):
            _resolve_prompt_no_reason(
                {},
                force_reasoning_prompt=False,
                force_no_reason_prompt=False,
            )
        self.assertTrue(
            _resolve_prompt_no_reason(
                {},
                force_reasoning_prompt=False,
                force_no_reason_prompt=True,
            )
        )
        self.assertFalse(
            _resolve_prompt_no_reason(
                {},
                force_reasoning_prompt=True,
                force_no_reason_prompt=False,
            )
        )

    def test_prompt_contract_uses_explicit_stage_metadata(self):
        self.assertTrue(
            _resolve_prompt_no_reason(
                {"no_reason": True},
                force_reasoning_prompt=False,
                force_no_reason_prompt=False,
            )
        )
        self.assertFalse(
            _resolve_prompt_no_reason(
                {"no_reason": False},
                force_reasoning_prompt=False,
                force_no_reason_prompt=False,
            )
        )
        self.assertFalse(
            _resolve_prompt_no_reason(
                {"no_reason": "0"},
                force_reasoning_prompt=False,
                force_no_reason_prompt=False,
            )
        )

    def test_experiment_fingerprint_is_canonical_and_sensitive(self):
        first = {"model": {"sha256": "a"}, "generation": {"seed": 1}}
        reordered = {"generation": {"seed": 1}, "model": {"sha256": "a"}}
        changed = {"model": {"sha256": "a"}, "generation": {"seed": 2}}
        self.assertEqual(_experiment_fingerprint(first), _experiment_fingerprint(reordered))
        self.assertNotEqual(_experiment_fingerprint(first), _experiment_fingerprint(changed))

    def test_protocol_excludes_model_specific_checkpoint_selection(self):
        shared = {
            "schema_version": 1,
            "model": {"manifest_sha256": "a" * 64},
            "training_run_contract": {"method": "clasp_dense"},
            "training_run_contract_sha256": "b" * 64,
            "selection": {
                "tasks": ["tsp"],
                "benchmark_manifest": {"manifest_sha256": "c" * 64},
                "checkpoint_selection_receipt": {"receipt_sha256": "d" * 64},
                "checkpoint_selection_binding": {"checkpoint_step": 400},
            },
            "generation": {"seed": 271828},
        }
        challenger = json.loads(json.dumps(shared))
        challenger["model"] = {"manifest_sha256": "e" * 64}
        challenger["training_run_contract"] = {"method": "grpo"}
        challenger["training_run_contract_sha256"] = "f" * 64
        challenger["selection"]["checkpoint_selection_receipt"] = {"receipt_sha256": "1" * 64}
        challenger["selection"]["checkpoint_selection_binding"] = {"checkpoint_step": 800}
        self.assertEqual(
            _protocol_fingerprint_material(shared),
            _protocol_fingerprint_material(challenger),
        )
        self.assertNotEqual(
            _experiment_fingerprint(shared),
            _experiment_fingerprint(challenger),
        )

    def test_protocol_retains_tokenizer_identity(self):
        first = {
            "schema_version": 1,
            "model": {
                "checkpoint_artifact": {"manifest_sha256": "a" * 64},
                "tokenizer_artifact": {"manifest_sha256": "b" * 64},
                "loaded_tokenizer": {"vocab_sha256": "c" * 64},
            },
            "selection": {"tasks": ["tsp"]},
        }
        changed = json.loads(json.dumps(first))
        changed["model"]["tokenizer_artifact"]["manifest_sha256"] = "d" * 64
        self.assertNotEqual(
            _protocol_fingerprint_material(first),
            _protocol_fingerprint_material(changed),
        )

    def test_per_example_sampling_seed_is_stable_and_row_specific(self):
        first = _example_sampling_seed(42, "tsp", "eval", 7)
        self.assertEqual(first, _example_sampling_seed(42, "tsp", "eval", 7))
        self.assertNotEqual(first, _example_sampling_seed(42, "tsp", "eval", 8))
        self.assertNotEqual(first, _example_sampling_seed(43, "tsp", "eval", 7))

    def test_experiment_provenance_hashes_model_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"_name_or_path": str(model_dir)}),
                encoding="utf-8",
            )
            weight_path = model_dir / "model.safetensors"
            weight_path.write_bytes(b"weights-a")
            (model_dir / "tokenizer.json").write_text(
                json.dumps({"version": "1.0"}), encoding="utf-8"
            )

            kwargs = {
                "model_path": str(model_dir),
                "tokenizer_path": str(model_dir),
                "model": _FakeModel(),
                "tokenizer": _FakeTokenizer(),
                "data_identity": {"manifest_sha256": "data", "files": []},
                "prompting": {"no_reason": True},
                "generation": {"seed": 42},
                "selection": {"tasks": ["tsp"]},
            }
            _material, first, _protocol = _build_experiment_provenance(**kwargs)
            weight_path.write_bytes(b"weights-b")
            _material, second, _protocol = _build_experiment_provenance(**kwargs)
            self.assertNotEqual(first, second)

    def test_data_identity_hashes_content_not_only_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_file = root / "tsp" / "eval" / "eval.json"
            data_file.parent.mkdir(parents=True)
            data_file.write_text('[{"input":"a"}]', encoding="utf-8")
            first = _dataset_content_identity(
                data_root=str(root),
                task_json_files={"tsp": [str(data_file)]},
                split="eval",
            )
            data_file.write_text('[{"input":"b"}]', encoding="utf-8")
            second = _dataset_content_identity(
                data_root=str(root),
                task_json_files={"tsp": [str(data_file)]},
                split="eval",
            )
            self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_result_label_includes_experiment_fingerprint(self):
        args = SimpleNamespace(
            stage="grpo",
            split="eval",
            num_return_sequences=8,
            eval_name="sealed",
        )
        label = _results_label(
            args,
            "checkpoint-10",
            experiment_fingerprint="a" * 64,
        )
        self.assertIn("fpaaaaaaaaaaaa", label)

    def test_resume_rejects_missing_or_mismatched_fingerprint(self):
        args = SimpleNamespace(num_return_sequences=2)
        record = {
            "task": "tsp",
            "split": "eval",
            "example_index": 0,
            "num_return_sequences": 2,
            "candidates": [{}, {}],
        }
        with self.assertRaisesRegex(RuntimeError, "incompatible detail record"):
            _resume_records_by_key(
                [record],
                args,
                experiment_fingerprint="expected",
            )
        record["experiment_fingerprint"] = "expected"
        record = _self_hash_payload(record, "record_sha256")
        by_key = _resume_records_by_key(
            [record],
            args,
            experiment_fingerprint="expected",
        )
        self.assertEqual(by_key[("tsp", "eval", 0)], record)

    def test_resume_rejects_incompatible_config(self):
        with self.assertRaisesRegex(RuntimeError, "incompatible experiment"):
            _validate_resume_run_config(
                {"experiment_fingerprint": "old"},
                experiment_fingerprint="new",
                run_config_path="run_config.json",
            )

    def test_training_run_contract_is_content_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint = run_dir / "checkpoints" / "checkpoint-400"
            checkpoint.mkdir(parents=True)
            file_receipt = {
                "path": "artifact.bin",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
            aggregate = hashlib.sha256()
            aggregate.update(b"artifact.bin\0")
            aggregate.update(b"1\0")
            aggregate.update(("a" * 64).encode("ascii"))
            aggregate.update(b"\n")
            tree_receipt = {
                "canonical_path": "/immutable/artifact",
                "sha256": aggregate.hexdigest(),
                "file_count": 1,
                "size_bytes": 1,
                "files": [file_receipt],
            }
            contract = {
                "schema_version": 2,
                "stage": "grpo",
                "method": "clasp_canonical",
                "task": "tsp",
                "seeds": {"train": 42, "data": 42, "sft": 42},
                "fixed_horizon": {"max_steps": 3200, "world_size": 4},
                "data_lineage": {"registry_sha256": "1" * 64},
                "model_initialization_registry": {
                    "registry_sha256": "2" * 64,
                    "specialist_sft_by_task": {},
                },
                "fixed_kl_reference": None,
                "fixed_null_receipt": None,
                "fingerprints": {
                    name: tree_receipt for name in ("initial_model", "base_model", "data", "source")
                },
                "generation": {"num_generations": 8},
                "training_config": {
                    "learning_rate": 1e-6,
                    "beta": 0.0,
                    "checkpoint_fidelity": "publication_exact_required",
                },
                "tracked_environment": {
                    "CLASP_BON_ALIGNED": "1",
                    "CLASP_PUBLICATION_FIDELITY": "1",
                },
            }
            contract_path = run_dir / "run_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            payload, file_sha256, identity = _training_run_contract_identity(
                model_path=str(checkpoint),
                run_dir=str(run_dir),
                required=True,
            )
            self.assertEqual(payload, contract)
            self.assertEqual(file_sha256, identity["sha256"])
            self.assertEqual(
                identity["payload_sha256"],
                _experiment_fingerprint(contract),
            )

    def test_legacy_development_contract_remains_nonpublication_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "legacy-run"
            checkpoint = run_dir / "checkpoint-40"
            checkpoint.mkdir(parents=True)
            legacy = {"schema_version": 1, "stage": "grpo", "legacy": True}
            (run_dir / "run_contract.json").write_text(json.dumps(legacy), encoding="utf-8")

            payload, file_sha256, identity = _training_run_contract_identity(
                model_path=str(checkpoint),
                run_dir=str(run_dir),
                required=False,
            )
            self.assertEqual(payload, legacy)
            self.assertEqual(file_sha256, identity["sha256"])
            with self.assertRaisesRegex(RuntimeError, "full official schema"):
                _training_run_contract_identity(
                    model_path=str(checkpoint),
                    run_dir=str(run_dir),
                    required=True,
                )

    def test_publication_development_fidelity_closes_model_load_before_provenance(self):
        source = Path(evaluate_module.__file__).read_text(encoding="utf-8")
        publication_gate = source.index(
            "publication_checkpoint_required = is_confirmation or publication_contract"
        )
        pre_load_verify = source.index(
            "publication_checkpoint_fidelity_before_load = publication_fidelity_identity(",
            publication_gate,
        )
        pre_load_runtime = source.index(
            "publication_exact_runtime_before_load = _publication_exact_runtime_identity(",
            pre_load_verify,
        )
        model_load = source.index(
            "model, tokenizer, _ = load_model_and_tokenizer(", pre_load_runtime
        )
        post_load_verify = source.index(
            "publication_checkpoint_fidelity = publication_fidelity_identity(",
            model_load,
        )
        post_load_runtime = source.index(
            "publication_exact_runtime = _publication_exact_runtime_identity(",
            post_load_verify,
        )
        provenance = source.index("_build_experiment_provenance(", post_load_runtime)
        self.assertLess(publication_gate, pre_load_verify)
        self.assertLess(pre_load_verify, pre_load_runtime)
        self.assertLess(pre_load_runtime, model_load)
        self.assertLess(model_load, post_load_verify)
        self.assertLess(post_load_verify, post_load_runtime)
        self.assertLess(post_load_runtime, provenance)

    def test_beta_zero_publication_rejects_stale_exact_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint = run_dir / "checkpoint-40"
            checkpoint.mkdir(parents=True)
            (run_dir / "run_contract.json").write_text("{}", encoding="utf-8")
            contract = {
                "training_config": {"beta": 0.0},
                "fixed_kl_reference": None,
            }
            self.assertIsNone(
                _publication_exact_runtime_identity(
                    model_path=str(checkpoint),
                    run_dir=str(run_dir),
                    training_run_contract=contract,
                )
            )
            (checkpoint / "exact_peft_kl_reference.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "beta=0.*unexpected"):
                _publication_exact_runtime_identity(
                    model_path=str(checkpoint),
                    run_dir=str(run_dir),
                    training_run_contract=contract,
                )

    def test_positive_beta_publication_requires_root_and_checkpoint_runtime_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint = run_dir / "checkpoint-40"
            checkpoint.mkdir(parents=True)
            (run_dir / "run_contract.json").write_text("{}", encoding="utf-8")
            contract = {
                "training_config": {"beta": 0.01, "dtype": "bfloat16"},
                "fixed_horizon": {"world_size": 4},
                "fixed_kl_reference": {"identity_sha256": "a" * 64},
            }
            with patch.object(
                evaluate_module,
                "_validate_fixed_kl_projection",
                return_value=(0.01, contract["fixed_kl_reference"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "missing.*runtime receipt"):
                    _publication_exact_runtime_identity(
                        model_path=str(checkpoint),
                        run_dir=str(run_dir),
                        training_run_contract=contract,
                    )

    def test_resume_uses_content_locator_after_data_root_moves(self):
        record = {
            "task": "tsp",
            "split": "eval",
            "example_index": 3,
            "source_json_files": ["/old/root/tsp/eval/eval.json"],
            "instruction": "instruction",
            "input_problem": "input",
            "ground_truth_output": "answer",
            "experiment_fingerprint": "fingerprint",
            "source_row_locator": {
                "data_manifest_sha256": "data-manifest",
                "task": "tsp",
                "split": "eval",
                "merged_dataset_index": 3,
            },
        }
        self.assertTrue(
            _resume_record_matches_example(
                record,
                task="tsp",
                split="eval",
                example_index=3,
                json_files=["/new/root/tsp/eval/eval.json"],
                instruction="instruction",
                input_text="input",
                answer="answer",
                experiment_fingerprint="fingerprint",
                data_manifest_sha256="data-manifest",
            )
        )

    def test_atomic_json_write_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"state":"old"}\n', encoding="utf-8")
            with patch("clasp.entrypoints.evaluate.json.dump", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    _write_json_atomic(str(path), {"state": "new"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"state": "old"})
            self.assertEqual(list(path.parent.glob(".config.json.*.tmp")), [])

    def test_interrupted_trailing_detail_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "details.jsonl"
            path.write_text('{"example_index":0}\n{"example_', encoding="utf-8")
            self.assertEqual(
                _read_detail_records_for_resume(str(path)),
                [{"example_index": 0}],
            )

    def test_interior_malformed_or_nonobject_detail_line_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "details.jsonl"
            path.write_text(
                '{"example_index":0}\n{"broken"\n{"example_index":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "malformed interior"):
                _read_detail_records_for_resume(str(path))
            path.write_text(
                '{"example_index":0}\n[]\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "non-object"):
                _read_detail_records_for_resume(str(path))

    def test_raw_feasibility_preset_disables_incompatible_bon_mode(self):
        args = SimpleNamespace(
            clasp_raw_feasibility_targeting=True,
            clasp_strict_defaults=True,
            use_clasp=False,
            use_primal_dual=False,
            pd_component_profile="none",
            clasp_feasibility_warmup_steps=0,
            clasp_self_replay=None,
            clasp_validation_gate_checkpoints=None,
            num_generations_train=None,
        )
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["clasp-grpo"]):
            _apply_clasp_strict_defaults(args, num_generations_attr="num_generations")
            self.assertEqual(os.environ["CLASP_BON_ALIGNED"], "0")
            self.assertEqual(os.environ["CLASP_FEASIBILITY_GATED"], "1")
            self.assertEqual(args.clasp_feasibility_warmup_steps, 500)

    def test_best_run_selection_is_feasibility_first(self):
        high_feasibility = {
            "overall": {
                "best_of_n_instance_feasibility_rate": 0.95,
                "mean_best_feasible_gap": 0.10,
            }
        }
        low_feasibility_low_gap = {
            "overall": {
                "best_of_n_instance_feasibility_rate": 0.50,
                "mean_best_feasible_gap": 0.01,
            }
        }
        self.assertLess(
            _run_selection_score(high_feasibility),
            _run_selection_score(low_feasibility_low_gap),
        )

    def test_best_run_selection_uses_gap_as_tiebreaker(self):
        better_gap = {
            "overall": {
                "best_of_n_instance_feasibility_rate": 0.95,
                "mean_best_feasible_gap": 0.02,
            }
        }
        worse_gap = {
            "overall": {
                "best_of_n_instance_feasibility_rate": 0.95,
                "mean_best_feasible_gap": 0.03,
            }
        }
        self.assertLess(_run_selection_score(better_gap), _run_selection_score(worse_gap))

    def test_representation_metadata_restores_missing_env_only(self):
        stage_hparams = {
            "clasp_mis_jiang_response_output": "1",
            "clasp_tsp_prompt_knn": "2",
        }
        with patch.dict(
            os.environ,
            {"CLASP_TSP_PROMPT_KNN": "9"},
            clear=True,
        ):
            restored = _restore_stage_representation_env(stage_hparams)
            self.assertEqual(os.environ["CLASP_MIS_JIANG_RESPONSE_OUTPUT"], "1")
            self.assertEqual(os.environ["CLASP_TSP_PROMPT_KNN"], "9")
            self.assertEqual(restored, {"CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1"})

    def test_jiang_representation_metadata_rejects_inherited_conflict(self):
        stage_hparams = {
            "prompt_style": "jiang_f638",
            "clasp_mis_jiang_response_output": "0",
        }
        with patch.dict(
            os.environ,
            {"CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1"},
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "representation contract"):
            _restore_stage_representation_env(stage_hparams)

    def test_nearest_run_dir_follows_checkpoint_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "real_run"
            checkpoint = run_dir / "checkpoints" / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            alias = root / "checkpoint-alias"
            alias.symlink_to(checkpoint, target_is_directory=True)
            self.assertEqual(
                _nearest_run_dir(str(alias), str(root / "unused"), "sft"),
                str(run_dir),
            )


if __name__ == "__main__":
    unittest.main()
