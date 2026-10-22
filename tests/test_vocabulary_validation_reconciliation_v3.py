"""Pure CPU token-validation equivalence; no actors or policy outputs."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/rloo_one_shot_20261016"
PATH = HERE / "cpu_preparation_reconciliation_v3/vocabulary_validation_v3.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


@pytest.fixture
def adapter():
    raw = PATH.read_bytes()
    m = types.ModuleType("validation_adapter_fixture")
    m.__file__ = str(PATH)
    m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(PATH), "exec", dont_inherit=True), m.__dict__)
    return m


class Tokenizer:
    eos_token_id = pad_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}
    def __init__(self):
        self.calls = 0
        self.vocab = {str(i): i for i in range(100)}
    def __len__(self):
        self.calls += 1
        return len(self.vocab)
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [3, 4]
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "Route: [0, 1]"
    def get_vocab(self):
        return dict(self.vocab)


class Config:
    max_new_tokens = 3000
    def __init__(self, sampled=False):
        self.sampled = sampled
    def to_dict(self):
        return {"max_new_tokens": 3000, "do_sample": self.sampled,
                "num_return_sequences": 1, "num_beams": 1}


def pure_module(adapter, filename):
    path = HERE / filename
    raw = path.read_bytes()
    target = adapter.TARGETS[filename][1]
    tree = ast.parse(raw)
    names = {target, "_record"} if filename == "teacher_candidate_v2.py" else {target}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    m = types.ModuleType("original_pure_record_fixture")
    m.__file__ = str(path)
    m.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    m.__dict__.update(need=need, digest=digest, sha=lambda z: hashlib.sha256(z).hexdigest(),
                      require=need, same=lambda a, b: digest(a) == digest(b),
                      encode=lambda z: json.dumps(z, sort_keys=True, separators=(",", ":")).encode())
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path),
                 "exec", dont_inherit=True), m.__dict__)
    return m, raw, path


def record(module, filename, length, sampled=False):
    row = {"instance_id": "synthetic_training_row", "semantic_id": "a" * 64, "prompt": "training prompt"}
    ids = [5] * (length - 1) + [0]
    tok, cfg = Tokenizer(), Config(sampled)
    role, count, group, index = ("gradient_calibration", 9, 7, 8) if sampled else ("teacher_incumbent", 1, None, 0)
    kwargs = {"role": role, "index": index, "count": count, "group_index": group}
    if filename == "run_local_7b_v1.py":
        row.update(task_type="op", instance_json=[[[0, 0], [1, 0]], [0, 1], 2], output="Route: [0, 1]")
        cfg = Config(False)
        value = {"instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
            "completion_token_ids": ids, "generated_tokens": length, "generation_calls": 1,
            "text": tok.decode(ids, skip_special_tokens=True), "actual_generation_config": cfg.to_dict(),
            "use_model_defaults": False, "extra_generation_kwargs": {}, "eos_observed": True,
            "truncated": False, "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "strict_score": {"synthetic_physical_score": True}}
        core = types.SimpleNamespace(generation_config=lambda tokenizer, cap, sample: cfg,
            score_completion=lambda *args: {"synthetic_physical_score": True})
        call = lambda fn, token: fn(value, row, token, core)
    elif filename == "teacher_candidate_v2.py":
        value = module._record(row, ids, tok, cfg, **kwargs)
        call = lambda fn, token: fn(value, row, token, cfg, **kwargs)
    else:
        padded = ids + [0] * min(3, 3000 - len(ids))
        value = {"instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(), "split": "training",
            "completion_token_ids": ids, "padded_completion_token_ids": padded,
            "prompt_token_ids": [3, 4], "prompt_attention_mask": [1, 1], "prompt_token_count": 2,
            "completion_mask": [1] * len(ids) + [0] * (len(padded) - len(ids)),
            "group_size": count, "generation_calls_for_group": 1,
            "generation_call_index": 135 if sampled else 0,
            "calibration_group_index": group, "global_stream_seed": 100073 if sampled else 73,
            "RNG_before_group": {"CPU_sha256": "b" * 64, "CUDA_sha256": "c" * 64},
            "RNG_after_group": {"CPU_sha256": "d" * 64, "CUDA_sha256": "e" * 64},
            "role": role, "sample_index": index, "text": tok.decode(ids, skip_special_tokens=True),
            "generated_tokens": length, "eos_observed": True, "truncated": False,
            "actual_generation_config": cfg.to_dict(), "use_model_defaults": False, "extra_generation_kwargs": {}}
        call = lambda fn, token: fn(value, row, token, cfg, sample_index=index, role=role)
    return value, call


@pytest.mark.parametrize("filename", ["collect_training_sft_v1.py", "teacher_candidate_v2.py", "run_local_7b_v1.py"])
@pytest.mark.parametrize("length", [1, 12, 150, 3000])
@pytest.mark.parametrize("sampled", [False, True])
def test_exact_valid_return_and_bounds_with_one_len_per_record(adapter, filename, length, sampled):
    m, raw, path = pure_module(adapter, filename)
    name = adapter.TARGETS[filename][1]
    value, call = record(m, filename, length, sampled)
    before = copy.deepcopy(value)
    original = getattr(m, name)
    slow = Tokenizer()
    expected_return = True if filename == "run_local_7b_v1.py" else None
    assert call(original, slow) is expected_return
    proof = adapter.apply_known_module(m, raw, str(path))
    fast = Tokenizer()
    assert call(getattr(m, name), fast) is expected_return
    assert value == before
    assert fast.calls == 1
    assert slow.calls == (length + min(length + 3, 3000) if filename == "collect_training_sft_v1.py" else length)
    assert proof["inverse_AST_restores_entire_original_function"] is True
    assert proof["all_other_module_AST_nodes_unchanged"] is True
    assert proof["teacher_search_or_numerical_training_changed"] is False


@pytest.mark.parametrize("filename,fault", [
    (filename, fault) for filename in ("collect_training_sft_v1.py", "teacher_candidate_v2.py", "run_local_7b_v1.py")
    for fault in (["bool", "float", "negative", "upper_bound", "eos_inside", "text", "config"] +
                  ([] if filename == "run_local_7b_v1.py" else ["mask", "role"]))])
def test_original_and_derived_reject_same_corrupted_record(adapter, filename, fault):
    m, raw, path = pure_module(adapter, filename)
    name = adapter.TARGETS[filename][1]
    value, call = record(m, filename, 12)
    if fault in {"bool", "float", "negative", "upper_bound"}:
        value["completion_token_ids"][2] = {"bool": True, "float": 1.0, "negative": -1, "upper_bound": 100}[fault]
    elif fault == "eos_inside": value["completion_token_ids"][2] = 0
    elif fault == "text": value["text"] += " altered"
    elif fault == "mask": value["completion_mask"][0] = 0
    elif fault == "role": value["role"] = "validation"
    else: value["actual_generation_config"]["do_sample"] = True
    original = getattr(m, name)
    with pytest.raises(ValueError): call(original, Tokenizer())
    adapter.apply_known_module(m, raw, str(path))
    with pytest.raises(ValueError): call(getattr(m, name), Tokenizer())


def test_complete_module_inverse_proof_covers_exact_four_expressions_only(adapter):
    assert sum(v[2] for v in adapter.TARGETS.values()) == 4
    for filename, (expected, name, count) in adapter.TARGETS.items():
        path = HERE / filename
        original, derived, proof = adapter.derive_validator(path.read_bytes(), path)
        assert proof["original_source"]["sha256"] == expected
        assert proof["function"] == name and proof["replaced_per_token_len_expressions"] == count
        assert proof["original_full_module_AST_sha256"] != proof["derived_full_module_AST_sha256"]
        assert original.co_argcount == derived.co_argcount and original.co_kwonlyargcount == derived.co_kwonlyargcount


def test_actual_slice_namespace_has_truthful_source_and_live_same_globals(adapter):
    path = HERE / "run_local_7b_v1.py"
    raw = path.read_bytes()
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "validate_raw_completion")
    namespace = {"require": need, "sha": lambda z: hashlib.sha256(z).hexdigest(),
                 "same": lambda a, b: digest(a) == digest(b)}
    # The legacy metadata selector intentionally omits the future import.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    assert "__file__" not in namespace and "__executed_source_sha256__" not in namespace
    m = types.SimpleNamespace(**namespace)
    value, call = record(m, path.name, 12)
    proof = adapter.apply_known_namespace(namespace, raw, str(path), original_full_buffer_executed=False)
    assert proof["original_buffer_was_executed_before_declared_function_replacement"] is False
    assert proof["known_full_source_function_matched_in_pure_AST_slice"] is True
    assert namespace["validate_raw_completion"].__globals__ is namespace
    assert call(namespace["validate_raw_completion"], Tokenizer()) is True
    def changed(condition, message):
        raise ValueError("live namespace guard changed")
    namespace["require"] = changed
    with pytest.raises(ValueError, match="live namespace"):
        call(namespace["validate_raw_completion"], Tokenizer())


def test_full_buffer_cannot_be_fabricated_from_unlabelled_slice(adapter):
    m, raw, path = pure_module(adapter, "run_local_7b_v1.py")
    del m.__dict__["__executed_source_sha256__"]
    with pytest.raises(ValueError, match="captured module identity"):
        adapter.apply_known_namespace(m.__dict__, raw, str(path), original_full_buffer_executed=True)


def test_copied_slice_globals_are_denied(adapter):
    m, raw, path = pure_module(adapter, "run_local_7b_v1.py")
    copied = dict(m.__dict__)
    with pytest.raises(ValueError, match="actual function-global"):
        adapter.apply_known_namespace(copied, raw, str(path), original_full_buffer_executed=False)


@pytest.mark.parametrize("fault", ["raw", "metadata", "path", "already_modified"])
def test_source_or_function_rebinding_denied_before_install(adapter, fault):
    m, raw, path = pure_module(adapter, "collect_training_sft_v1.py")
    if fault == "raw": raw += b"\n"
    elif fault == "metadata": m.__executed_source_sha256__ = "0" * 64
    elif fault == "path": path = path.with_name("other.py")
    else: m.validate_record = lambda *args, **kwargs: None
    with pytest.raises(ValueError): adapter.apply_known_module(m, raw, str(path))


def test_idempotence_rechecks_derived_code_and_denies_later_tamper(adapter):
    m, raw, path = pure_module(adapter, "teacher_candidate_v2.py")
    first = adapter.apply_known_module(m, raw, str(path))
    assert adapter.apply_known_module(m, raw, str(path)) == first
    m._validate_record = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        adapter.apply_known_module(m, raw, str(path))


@pytest.mark.parametrize("bad", [True, -1, 100])
def test_padded_token_bounds_remain_typed_and_complete(adapter, bad):
    m, raw, path = pure_module(adapter, "collect_training_sft_v1.py")
    value, call = record(m, path.name, 12)
    value["padded_completion_token_ids"][-1] = bad
    with pytest.raises(ValueError): call(m.validate_record, Tokenizer())
    adapter.apply_known_module(m, raw, str(path))
    with pytest.raises(ValueError): call(m.validate_record, Tokenizer())


def test_immutable_vocabulary_guard_measures_then_rejects_mutation(adapter):
    t = Tokenizer()
    with adapter.immutable_vocabulary(t) as proof:
        assert proof["size"] == 100
    assert t.calls == 2
    with pytest.raises(ValueError, match="vocabulary changed"):
        with adapter.immutable_vocabulary(t): t.vocab["new"] = 100


def test_adapter_itself_requires_captured_source_metadata(adapter):
    adapter.__executed_source_sha256__ = "0" * 64
    with pytest.raises(ValueError, match="exact captured"):
        adapter.source_binding()
