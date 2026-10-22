"""CPU replay of actual teacher methods, tokenizer and TRL row transforms."""
import ast
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clasp.rewards.co_tasks import (
    _check_cvrp_constraints,
    _check_mis_constraints,
    _mis_meta,
    _parse_cvrp_solution,
    _parse_mis_solution,
)

ROOT = Path(__file__).resolve().parents[1]


def extracted_trainer():
    """Compile repository methods directly, without constructing a GPU trainer."""
    source = ast.parse((ROOT / "src/clasp/trainers/primal_dual_grpo.py").read_text())
    names = {
        "_verified_improvement_target", "_tokenize_verified_target",
        "_attach_verified_improvement_tensors", "_improvement_distillation_specs",
        "_extract_batch_raw_values", "_maybe_json_load", "_safe_pad_token_id",
        "_sanitize_token_ids", "_model_token_vocab_size",
    }
    trainer = next(n for n in source.body if isinstance(n, ast.ClassDef)
                   and n.name == "PrimalDualGRPOTrainer")
    methods = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    cls = ast.ClassDef(name="ReplayTrainer", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {"torch": torch, "json": json, "time": time}
    exec(compile(ast.fix_missing_locations(module), "<actual trainer methods>", "exec"), namespace)
    return namespace["ReplayTrainer"]()


@pytest.fixture
def teacher():
    tokenizers = pytest.importorskip("tokenizers")
    path = ROOT / "models/local_tsp_sft500k_seed42_instruct/tokenizer.json"
    if not path.exists():
        pytest.skip("Local real Qwen tokenizer artifact unavailable")
    backend = tokenizers.Tokenizer.from_file(str(path))

    class Tokenizer:
        eos_token_id = backend.token_to_id("<|im_end|>")
        pad_token_id = backend.token_to_id("<|endoftext|>")

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return backend.encode(text, add_special_tokens=False).ids

        def batch_decode(self, rows, **kwargs):
            return backend.decode_batch(rows, **kwargs)

    obj = extracted_trainer()
    obj.clasp_enabled = False
    obj.grpo_verified_improvement = True
    obj.grpo_verified_improvement_weight = 0.05
    obj.grpo_verified_improvement_max_tokens = 3000
    obj.grpo_verified_improvement_max_passes = 2
    obj.processing_class = Tokenizer()
    obj.model = SimpleNamespace(training=True, config=SimpleNamespace(vocab_size=backend.get_vocab_size()))
    return obj


def test_real_tokenizer_roundtrip_keeps_complete_solution_and_eos(teacher):
    text = "Set: [1, 2], Objective: 2"
    ids = teacher._tokenize_verified_target(text, 3000)
    assert ids[-1] == teacher.processing_class.eos_token_id
    assert teacher.processing_class.batch_decode([ids], skip_special_tokens=True) == [text]
    assert teacher._tokenize_verified_target(text, len(ids) - 1) == []


def test_actual_development_metadata_yields_verified_teacher_text(teacher):
    base = ROOT / "data/development_fair_cvrp_mis_20260905_v3"
    if not base.exists():
        pytest.skip("Retained development metadata unavailable")
    cvrp = json.loads((base / "cvrp/eval/test.json").read_text())[0]
    result = teacher._verified_improvement_target(
        "cvrp", "Routes: [[0, 1, 1, 0]], Objective: 999999",
        cvrp["instance_json"], cvrp["input"],
    )
    assert result is not None and result[3]
    routes = _parse_cvrp_solution(result[0])["routes"]
    assert _check_cvrp_constraints(routes, *cvrp["instance_json"])[2]
    assert teacher._tokenize_verified_target(result[0], 3000)
    mis = json.loads((base / "mis/eval/test.json").read_text())[0]
    result = teacher._verified_improvement_target(
        "mis", "Set: [0, 0, -1], Objective: 999999", mis["instance_json"], mis["input"],
    )
    assert result is not None and result[3]
    nodes = _parse_mis_solution(result[0])["node_set"]
    assert _check_mis_constraints(nodes, *_mis_meta(mis["instance_json"], mis["input"]))[2]
    assert teacher._tokenize_verified_target(result[0], 3000)


def test_malformed_completion_has_no_unconditional_teacher(teacher):
    instance = {"num_nodes": 3, "edges": [[0, 1]]}
    assert teacher._verified_improvement_target("mis", "nonsense", instance, "") is None
    assert teacher._verified_improvement_target("mis", "Set: [], Objective: 0", instance, "") is None


def runtime_row_transforms():
    trl = importlib.util.find_spec("trl")
    if trl is not None:
        path = Path(trl.origin).parent / "trainer/utils.py"
    else:
        path = ROOT / "analysis_outputs/clasp_grpo_fair_dev_20260905/b18_telemetry_all_terminal100_v1/inputs/runtime/trl/trainer/utils.py"
    if not path.exists():
        pytest.skip("Actual TRL runtime source unavailable")
    source = ast.parse(path.read_text())
    names = {"shuffle_sequence_dict", "split_tensor_dict"}
    funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *funcs], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["shuffle_sequence_dict"], namespace["split_tensor_dict"]


def test_actual_trl_shuffle_and_split_preserve_each_teacher_prompt(teacher):
    instances = [
        {"num_nodes": 3, "edges": [[0, 1], [0, 2]]},
        {"num_nodes": 5, "edges": []},
        {"num_nodes": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
    ]
    inputs = [{"instance_json": json.dumps(instance), "task_type": "mis", "input": "graph"}
              for instance in instances]
    tokens = teacher.processing_class.encode("Set: [0, 0], Objective: 2", add_special_tokens=False, truncation=False)
    output = {"completion_ids": torch.tensor([tokens] * 3), "row_identity": torch.arange(3)}
    teacher._attach_verified_improvement_tensors(inputs, output, ["mis"] * 3)
    shuffle, split = runtime_row_transforms()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6062026)
        chunks = split(shuffle(output), 3)
    assert sorted(int(c["row_identity"].item()) for c in chunks) == [0, 1, 2]
    for chunk in chunks:
        index = int(chunk["row_identity"].item())
        ids = chunk["grpo_verified_mis_completion_ids"]
        mask = chunk["grpo_verified_mis_completion_mask"].bool()
        text = teacher.processing_class.batch_decode([ids[mask].tolist()], skip_special_tokens=True)[0]
        nodes = _parse_mis_solution(text)["node_set"]
        assert _check_mis_constraints(nodes, *_mis_meta(instances[index], ""))[2]
        assert bool(chunk["grpo_verified_mis_repairs"].item())
        assert chunk["grpo_verified_cvrp_completion_mask"].sum().item() == 0
