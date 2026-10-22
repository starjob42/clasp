"""Synthetic collection custody/physics fixtures; no actor or GPU is loaded."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/rloo_one_shot_20261016"
PATH = HERE / "admit_training_collection_v1.py"


def load(path, name):
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


M = load(PATH, "training_collection_admission_tests")
C = load(HERE / "collect_training_sft_v1.py", "original_training_collector_for_fixture")


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0
    bos_token_id = None
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}

    def __init__(self, task="op"):
        self.task = task

    def __len__(self):
        return 10

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return [4, 5]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "Route: [0]" if self.task == "op" else "Routes: [[0]]"

    def get_vocab(self):
        return {str(i): i for i in range(10)}


class GenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_dict(self):
        return copy.deepcopy(self.kwargs)


class Fixture:
    def __init__(self, directory, task="op"):
        self.output = directory / "once"
        self.output.mkdir()
        self.contract_path = directory / "contract.json"
        self.tokenizer = Tokenizer(task)
        self.contract = C.make_contract(M.binding(HERE / "execution_v1/execution_roster.json"), task,
                                       M.binding(HERE / "collect_training_sft_v1.py"), output=self.output)
        self.contract["synthetic_fixture"] = True
        self.contract["contract_sha256"] = M.digest({k: v for k, v in self.contract.items() if k != "contract_sha256"})
        self.save(self.contract_path, self.contract)
        self.roster, self.anchor, self.rows, _, self.runner = C.validate_contract(self.contract)
        source, buffers = self.runner.source_buffers(self.roster["source_manifest"])
        self.core = M._core_replay(source, buffers, GenerationConfig)
        self.greedy = [self.record(row, call=index) for index, row in enumerate(self.rows)]
        by_id = {row["instance_id"]: row for row in self.rows}
        self.calibration = [self.record(by_id[identifier], group=group, index=index, call=128 + group)
                            for group, identifier in enumerate(self.contract["calibration_instance_ids"]) for index in range(9)]
        self.save_lines("greedy_training.jsonl", self.greedy)
        self.save_lines("calibration_training.jsonl", self.calibration)
        init = self.anchor["initializer"]
        self.save(self.output / "input_admission.json", {
            "synthetic_fixture": True, "contract_sha256": self.contract["contract_sha256"],
            "initializer_files": [{**row, "path": str(Path(init[kind + "_root"]) / row["path"])}
                                  for kind in ("base", "sft") for row in init[kind + "_files"]],
            "training_binding": self.contract["training"], "validation_inputs_used_for_policy": False, "model_calls": 0})
        self.save(self.output / "model_admission.json", {
            "synthetic_fixture": True, "runtime": M.PACKAGES, "whole_actor_sha256": "1" * 64,
            "parameter_counts_by_dtype": {"torch.bfloat16": 6_000_000_000, "torch.float32": 32},
            "trainable_parameters": 0, "devices": ["cuda:0"], "actual_BF16_base_weights_verified": True,
            "FP32_adapter_parameter_names": ["base_model.model.layers.0.lora_A.default.weight"]})
        self.save(self.output / "gpu_owner.json", {
            "synthetic_fixture": True, "protocol": "one-shot-exclusive-gpu-owner-v1", "pid": 123456, "uid": os.getuid(),
            "gpu_uuid": self.runner.GPU_UUID, "lock_path": "/tmp/rloo_one_shot_20261016_" + self.runner.GPU_UUID + ".lock",
            "all_locks_held_until_GPU_lifetime_closed": True,
            "cooperating_legacy_lock_paths": ["/tmp/graphopt_iclr_prospective_ourasi_gpu3.lock", "/tmp/graphopt_iclr_prospective_factorial_ourasi_gpu3.lock"],
            "gpu_query_raw": "3, " + self.runner.GPU_UUID + ", NVIDIA RTX A6000, 49140, 48000, 0\n",
            "recorded_utc": "2026-10-16T10:00:05+00:00"})
        self.save(self.output / "attempt.json", {
            "synthetic_fixture": True, "contract": M.binding(self.contract_path), "requested_gpu_hours": 1.5,
            "maximum_wall_seconds": 5400, "automatic_retry": False, "recorded_utc": "2026-10-16T10:00:00+00:00"})
        self.result = {"synthetic_fixture": True, "protocol": C.PROTOCOL, "status": "complete_training_only_collection",
            "contract": self.contract, "whole_actor_sha256": "1" * 64, "whole_actor_unchanged": True,
            "optimizer_updates": 0, "greedy_responses": 128, "sampled_calibration_responses": 72,
            "total_policy_responses": 200, "generation_calls": 136, "global_calibration_stream_seed": 100073,
            "worker_collection_seconds_excluding_initializer_hashing": 20., "publication_eligible": False,
            "superiority_established": False, "heldout_policy_calls": 0}
        self.refresh_result_bindings()
        self.save(self.output / "completion.json", {"synthetic_fixture": True, "status": "completed", "error": None,
            "actual_wall_seconds": 30., "maximum_wall_seconds": 5400, "requested_gpu_hours": 1.5,
            "contract_file_sha256": M.binding(self.contract_path)["sha256"], "automatic_retry": False,
            "heldout_policy_calls": 0, "superiority_established": False})

    @staticmethod
    def save(path, value):
        path.write_bytes(M.encoded(value) + b"\n")

    def save_lines(self, name, rows):
        (self.output / name).write_bytes(b"".join(M.encoded(row) + b"\n" for row in rows))

    def refresh_result_bindings(self):
        for field, filename in (("greedy_binding", "greedy_training.jsonl"), ("calibration_binding", "calibration_training.jsonl"),
                                ("input_admission_binding", "input_admission.json"), ("model_admission_binding", "model_admission.json"),
                                ("gpu_owner_binding", "gpu_owner.json")):
            self.result[field] = M.binding(self.output / filename)
        self.save(self.output / "result.json", self.result)

    def record(self, row, *, group=None, index=0, call):
        sampled = group is not None
        text = self.tokenizer.decode([3, 9], skip_special_tokens=True)
        return {"instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
                "prompt_sha256": M.sha(row["prompt"].encode()), "split": "training",
                "role": "gradient_calibration" if sampled else "teacher_incumbent", "sample_index": index,
                "generation_calls_for_group": 1, "group_size": 9 if sampled else 1, "generation_call_index": call,
                "calibration_group_index": group, "global_stream_seed": 100073 if sampled else 73,
                "RNG_before_group": {"CPU_sha256": "a" * 64, "CUDA_sha256": "b" * 64},
                "RNG_after_group": {"CPU_sha256": "a" * 64, "CUDA_sha256": "b" * 64},
                "prompt_token_ids": [4, 5], "prompt_attention_mask": [1, 1], "prompt_token_count": 2,
                "padded_completion_token_ids": [3, 9, 0], "completion_mask": [1, 1, 0],
                "completion_token_ids": [3, 9], "text": text, "generated_tokens": 2,
                "eos_observed": True, "truncated": False,
                "actual_generation_config": self.core.generation_config(self.tokenizer, 3000, sample=sampled).to_dict(),
                "use_model_defaults": False, "extra_generation_kwargs": {}, "group_wall_seconds": .01,
                "strict_score": self.core.score_completion(self.contract["task"], text, row["instance_json"], row["output"])}

    def change(self, filename, fn):
        path = self.output / filename
        value = json.loads(path.read_bytes())
        fn(value)
        self.save(path, value)
        if filename in {"input_admission.json", "model_admission.json", "gpu_owner.json"}:
            self.refresh_result_bindings()

    def admit(self, **kwargs):
        return M.admit_training_collection(M.binding(self.contract_path), M.binding(self.output / "completion.json"),
            synthetic_fixture=True, fixture_tokenizer=self.tokenizer, fixture_generation_config=GenerationConfig, **kwargs)


@pytest.fixture
def bundle(tmp_path):
    return Fixture(tmp_path)


def no_raw_reads(monkeypatch):
    original = M.stable_read

    def checked(path):
        if Path(path).name in {"greedy_training.jsonl", "calibration_training.jsonl", "forbidden_raw.jsonl"}:
            pytest.fail("Opened raw training data before metadata admission")
        return original(path)

    monkeypatch.setattr(M, "stable_read", checked)


def test_complete_synthetic_cache_replays_physics_without_models(bundle, monkeypatch):
    imported = set(sys.modules)
    result = bundle.admit()
    assert result["kind"] == M.DIAGNOSTIC_KIND and result["synthetic_fixture"] is True
    assert result["actual_7b_production_collection_admitted"] is False
    assert result["training_records_replayed"] == 200 and result["model_calls_by_admission"] == result["GPU_calls_by_admission"] == 0
    assert result["common_content"]["SFT_identity"]["actual_optimizer_inventory"] is None
    assert result["common_content"]["SFT_identity"]["actual_model_config_semantic_sha256"] is None
    assert len(result["common_content"]["greedy_training"]) == 128
    assert len(result["common_content"]["calibration_groups"]) == 8
    assert result["common_content"]["logical_contract"]["scope"] == "prospective_teacher_development"
    assert "local_execution_contract_sha256" not in result["common_content"]["logical_contract"]
    assert all(len(group["records"]) == 9 for group in result["common_content"]["calibration_groups"])
    assert all(record["completion_mask"] == [1, 1] for record in result["common_content"]["greedy_training"])
    assert result["admission_sha256"] == M.digest({k: v for k, v in result.items() if k != "admission_sha256"})
    assert not ({"torch", "transformers", "clasp"} & (set(sys.modules) - imported))
    role = result["common_content"]["training_role_admission"]
    assert len(role["allowed_training_semantic_ids"]) == 128 and len(role["excluded_semantic_ids_by_split"]["validation"]) == 96
    assert len(role["excluded_semantic_ids_by_split"]["engineering_eval"]) == 6


def test_cvrp_task_is_independently_replayed(tmp_path):
    result = Fixture(tmp_path, "cvrp").admit()
    assert result["common_content"]["task"] == "cvrp" and result["kind"] == M.DIAGNOSTIC_KIND


@pytest.mark.parametrize("filename,field,value", [
    ("completion.json", "status", "failed"), ("completion.json", "actual_wall_seconds", 5400.1),
    ("completion.json", "requested_gpu_hours", 2.), ("completion.json", "heldout_policy_calls", True),
    ("completion.json", "error", "timeout"), ("attempt.json", "automatic_retry", True),
    ("model_admission.json", "trainable_parameters", 1), ("model_admission.json", "whole_actor_sha256", "2" * 64),
    ("model_admission.json", "devices", ["cpu"]), ("model_admission.json", "actual_BF16_base_weights_verified", False),
    ("input_admission.json", "validation_inputs_used_for_policy", True),
    ("gpu_owner.json", "all_locks_held_until_GPU_lifetime_closed", False), ("gpu_owner.json", "pid", True),
    ("gpu_owner.json", "recorded_utc", "2026-10-16T10:00:05"),
])
def test_metadata_denials_happen_before_raw(bundle, monkeypatch, filename, field, value):
    bundle.change(filename, lambda row: row.__setitem__(field, value))
    no_raw_reads(monkeypatch)
    with pytest.raises(ValueError):
        bundle.admit()


@pytest.mark.parametrize("field,value", [("optimizer_updates", 1), ("generation_calls", 135),
                                        ("whole_actor_unchanged", False), ("worker_collection_seconds_excluding_initializer_hashing", 31.),
                                        ("model_admission_binding", None)])
def test_result_denials_happen_before_raw(bundle, monkeypatch, field, value):
    bundle.result[field] = value
    bundle.save(bundle.output / "result.json", bundle.result)
    no_raw_reads(monkeypatch)
    with pytest.raises(ValueError):
        bundle.admit()


def test_bad_initializer_row_rejected_before_raw(bundle, monkeypatch):
    bundle.change("input_admission.json", lambda row: row["initializer_files"][3].__setitem__("sha256", "0" * 64))
    no_raw_reads(monkeypatch)
    with pytest.raises(ValueError, match="initializer"):
        bundle.admit()


def test_raw_escape_denied_without_reading_escaped_file(bundle, monkeypatch):
    escaped = bundle.output.parent / "forbidden_raw.jsonl"
    escaped.write_bytes(b"do not read")
    bundle.result["greedy_binding"] = M.binding(escaped)
    bundle.save(bundle.output / "result.json", bundle.result)
    no_raw_reads(monkeypatch)
    with pytest.raises(ValueError, match="escaped"):
        bundle.admit()


def test_linked_model_admission_is_not_accepted(bundle):
    path = bundle.output / "model_admission.json"
    target = bundle.output.parent / "copied_model.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Canonical"):
        bundle.admit()


@pytest.mark.parametrize("field,value", [("split", "validation"), ("prompt_token_ids", [1, 4, 5]),
                                        ("completion_mask", [1, 1, 1]), ("generated_tokens", True),
                                        ("global_stream_seed", 100074), ("group_wall_seconds", float("inf"))])
def test_corrupt_fourth_calibration_group_is_rejected(bundle, field, value):
    if field == "group_wall_seconds":
        # JSON itself refuses a nonfinite producer timer.
        with pytest.raises(ValueError):
            bundle.calibration[27][field] = value
            bundle.save_lines("calibration_training.jsonl", bundle.calibration)
        return
    bundle.calibration[27][field] = value
    bundle.save_lines("calibration_training.jsonl", bundle.calibration)
    bundle.refresh_result_bindings()
    with pytest.raises(ValueError):
        bundle.admit()


def test_calibration_global_stream_reset_is_rejected(bundle):
    for row in bundle.calibration[27:36]:
        row["RNG_before_group"]["CPU_sha256"] = "c" * 64
    bundle.save_lines("calibration_training.jsonl", bundle.calibration)
    bundle.refresh_result_bindings()
    with pytest.raises(ValueError, match="stream"):
        bundle.admit()


def test_typed_strict_score_replay_rejects_boolean_alias(bundle):
    # False==0 in Python is not a byte-identical physical score.
    assert bundle.greedy[0]["strict_score"]["u10"] == 0.
    bundle.greedy[0]["strict_score"]["u10"] = False
    bundle.save_lines("greedy_training.jsonl", bundle.greedy)
    bundle.refresh_result_bindings()
    with pytest.raises(ValueError, match="Typed independent strict"):
        bundle.admit()


def test_production_never_accepts_custom_tokenizer_or_synthetic_cache(bundle, monkeypatch):
    with pytest.raises(ValueError, match="cannot be replaced"):
        M.admit_training_collection(M.binding(bundle.contract_path), fixture_tokenizer=bundle.tokenizer)
    monkeypatch.setattr(M, "_production_tokenizer", lambda *_: pytest.fail("Loaded production tokenizer for synthetic metadata"))
    no_raw_reads(monkeypatch)
    with pytest.raises(ValueError, match="Synthetic evidence"):
        M.admit_training_collection(M.binding(bundle.contract_path))


def test_diagnostic_envelope_cannot_cross_v2_production_import(bundle):
    value = bundle.admit()
    with pytest.raises(ValueError, match="production7B"):
        M.validate_v2_import(value["common_content"]["logical_contract"], value, value["admission_sha256"], bundle.tokenizer)
    value["kind"] = M.PRODUCTION_KIND
    value["synthetic_fixture"] = False
    value["actual_7b_production_collection_admitted"] = True
    value["admission_sha256"] = M.digest({k: v for k, v in value.items() if k != "admission_sha256"})
    with pytest.raises(ValueError, match="Synthetic evidence"):
        M.validate_v2_import(value["common_content"]["logical_contract"], value, value["admission_sha256"], bundle.tokenizer)


def test_executed_source_cannot_be_rebound_to_changed_file(tmp_path):
    copied = tmp_path / "admit_training_collection_v1.py"
    copied.write_bytes(PATH.read_bytes())
    module = load(copied, "old_executed_admission")
    copied.write_bytes(copied.read_bytes() + b"\n# later disk bytes\n")
    with pytest.raises(ValueError, match="exact externally pinned"):
        module._self_binding()


def test_consumed_metadata_mutation_after_replay_is_rejected(bundle, monkeypatch):
    original = M.compile_buffer

    def compile_and_change(raw, path, name):
        module = original(raw, path, name)
        if name == "verified_training_only_collector":
            replay = module.validate_collection_records

            def replay_then_change(*args):
                replay(*args)
                p = bundle.output / "model_admission.json"
                p.write_bytes(p.read_bytes() + b"\n")

            module.validate_collection_records = replay_then_change
        return module

    monkeypatch.setattr(M, "compile_buffer", compile_and_change)
    with pytest.raises(ValueError, match="binding differs"):
        bundle.admit()


@pytest.mark.parametrize("raw_name", ["greedy_training.jsonl", "calibration_training.jsonl"])
def test_raw_synthetic_marker_never_promotes_even_if_metadata_is_resealed(bundle, monkeypatch, raw_name):
    # Adversarial diagnostic: strip the metadata labels to exercise the raw
    # boundary specifically. The private tokenizer substitute exists ONLY in
    # this test; the public production API never accepts such a callback.
    bundle.contract.pop("synthetic_fixture")
    bundle.contract["contract_sha256"] = M.digest({k: v for k, v in bundle.contract.items() if k != "contract_sha256"})
    bundle.save(bundle.contract_path, bundle.contract)
    for name in ("completion.json", "attempt.json", "input_admission.json", "model_admission.json", "gpu_owner.json"):
        value = json.loads((bundle.output / name).read_bytes())
        value.pop("synthetic_fixture")
        if name == "completion.json":
            value["contract_file_sha256"] = M.binding(bundle.contract_path)["sha256"]
        elif name == "attempt.json":
            value["contract"] = M.binding(bundle.contract_path)
        elif name == "input_admission.json":
            value["contract_sha256"] = bundle.contract["contract_sha256"]
        bundle.save(bundle.output / name, value)
    bundle.result.pop("synthetic_fixture")
    records = bundle.greedy if raw_name.startswith("greedy") else bundle.calibration
    records[4]["synthetic_fixture"] = True
    bundle.save_lines(raw_name, records)
    bundle.refresh_result_bindings()
    monkeypatch.setattr(M, "_production_tokenizer", lambda *_: (bundle.tokenizer, GenerationConfig))
    with pytest.raises(ValueError, match="Synthetic raw training"):
        M.admit_training_collection(M.binding(bundle.contract_path))
