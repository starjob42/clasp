"""Exercise native TRL collection/configuration with an in-memory tiny Qwen.

These tests require the declared Torch/Transformers/TRL runtime. They use no
network, downloaded checkpoint, accelerator allocation, or external dataset.
The numerical tracker tests in test_external_baseline.py also run without it.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import GRPOConfig

from clasp.trainers.external_baseline_grpo import (
    ExternalBaselineGRPOConfig,
    ExternalBaselineGRPOTrainer,
)


def row(index, *, reward=-0.5, task="mis", size=8):
    if task == "mis":
        instance = {"num_nodes": size, "edges": [[0, 1]]}
    else:
        instance = [[[float(i), 0.0] for i in range(size + 1)], [0] + [1] * size, 5]
    return {"prompt": f"solve {task} {index}", "task_type": task,
            "instance_json": json.dumps(instance), "input": f"instance {index}",
            "expected_reward": reward}


def config(tmp_path, *, generations=1, batch=2, accumulation=1, **overrides):
    arguments = {
        "output_dir": str(tmp_path), "num_generations": generations,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": accumulation, "steps_per_generation": accumulation,
        "max_completion_length": 4, "max_prompt_length": 32, "max_steps": 2,
        "bf16": False, "fp16": False, "use_cpu": True,
        "gradient_checkpointing": False, "report_to": [], "disable_tqdm": True,
        "logging_steps": 1, "save_strategy": "no", "eval_strategy": "no",
        "scale_rewards": "none", "loss_type": "dr_grpo", "beta": 0.0,
        "num_iterations": 1, "importance_sampling_level": "token",
        "use_vllm": False, "use_liger_loss": False,
        "mask_truncated_completions": False, "dataloader_num_workers": 0,
        "dataloader_drop_last": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    }
    arguments.update(overrides)
    return ExternalBaselineGRPOConfig(**arguments)


def tiny_trainer(tmp_path, *, estimator="external", generations=1, batch=2,
                 accumulation=1, initial=0.0, scale=1.0, reward_contract=None,
                 reward_funcs=None, dataset=None):
    vocabulary = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3,
                  "solve": 4, "mis": 5, "cvrp": 6,
                  **{str(i): 7 + i for i in range(16)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        eos_token="<eos>", bos_token="<bos>", padding_side="left",
        model_max_length=64,
    )
    # Match the real Qwen tokenizer: the generic fast-tokenizer default also
    # emits token_type_ids, which Qwen2.generate correctly rejects.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=2, bos_token_id=3,
        attention_dropout=0.0, attn_implementation="eager",
    ))

    def fixture_reward(prompts, completions, expected_reward, **kwargs):
        return list(expected_reward)

    trainer = ExternalBaselineGRPOTrainer(
        model=model,
        args=config(tmp_path, generations=generations, batch=batch, accumulation=accumulation),
        reward_funcs=[fixture_reward] if reward_funcs is None else reward_funcs,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([row(i) for i in range(16)]) if dataset is None else dataset,
        advantage_estimator=estimator, baseline_initial=initial, baseline_scale=scale,
        reward_contract=reward_contract,
    )
    trainer.model.train()
    return trainer


def deterministic_generation(trainer, monkeypatch):
    observed_revisions = []

    def generate(prompts, images):
        assert trainer._baseline_generation_active
        observed_revisions.append(trainer.external_baseline.revision)
        count = len(prompts)
        # Native _generate returns the sum of a gathered tensor of lengths;
        # buffer shuffling/splitting preserves this scalar tensor unchanged.
        token_count = torch.tensor(2 * count, device=trainer.accelerator.device)
        return [[4, 5]] * count, [[7, 2]] * count, token_count, None, {}

    monkeypatch.setattr(trainer, "_generate", generate)
    return observed_revisions


def test_single_generation_config_preserves_parent_derivations_and_other_guards(tmp_path):
    arguments = config(tmp_path, generations=1, batch=2, accumulation=2)
    assert arguments.num_generations == 1
    assert arguments.generation_batch_size == 4
    assert arguments.steps_per_generation == 2
    with pytest.raises(ValueError, match="at least 2 generations"):
        GRPOConfig(output_dir=str(tmp_path), num_generations=1, bf16=False, use_cpu=True,
                   report_to=[], per_device_train_batch_size=2)
    with pytest.raises(ValueError, match="both configured"):
        config(tmp_path, generation_batch_size=4)
    with pytest.raises(ValueError, match="Liger loss does not support"):
        config(tmp_path, delta=2.0, use_liger_loss=True)


@pytest.mark.parametrize("generations", [1, 2])
def test_native_repeat_sampler_allocates_distinct_prompts_then_repeats_buffer(
    tmp_path, generations
):
    trainer = tiny_trainer(tmp_path, generations=generations, batch=2, accumulation=2)
    indices = list(trainer._get_train_sampler())
    width = trainer.args.generation_batch_size
    first, repeated = indices[:width], indices[width:2 * width]
    assert first == repeated
    assert len(set(first)) == width // generations
    for start in range(0, width, generations):
        assert len(set(first[start:start + generations])) == 1


def test_real_tiny_qwen_g1_generation_and_nonzero_fixed_horizon_policy_gradient(tmp_path):
    trainer = tiny_trainer(tmp_path)
    trainer.current_gradient_accumulation_steps = 1
    batch = trainer._generate_and_score_completions([row(0), row(1)])
    assert trainer.external_baseline.revision == 1
    assert trainer.external_baseline.seen_samples == 2
    assert torch.equal(batch["advantages"], torch.tensor([-0.5, -0.5]))
    assert batch["completion_mask"].sum().item() > 0
    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    # Stock G1 sample-STD logging is replaced with finite population diagnostics.
    assert trainer._metrics["train"]["reward_std"] == [0.0]
    assert trainer._metrics["train"]["frac_reward_zero_std"] == [1.0]


@pytest.mark.parametrize("estimator,expected", [("external", [-0.5, 1.5]),
                                               ("rloo", [-2.0, 2.0])])
def test_native_reward_calculation_receives_frozen_baseline_and_keeps_raw_rewards(
    tmp_path, monkeypatch, estimator, expected
):
    trainer = tiny_trainer(tmp_path, estimator=estimator, generations=2)
    seen = deterministic_generation(trainer, monkeypatch)
    rows = [row(0, reward=-0.5), row(0, reward=1.5)]
    pre_hash = trainer.external_baseline.state_sha256
    batch = trainer._generate_and_score_completions(rows)
    assert seen == [0]
    assert batch["advantages"].tolist() == expected
    assert trainer.external_baseline.snapshot([("mis", 8)]).values == (0.5,)
    assert list(trainer._logs["advantages"]) == expected
    assert list(trainer._logs["rewards"]["fixture_reward"]) == [-0.5, 1.5]
    records = [json.loads(line) for line in
               (tmp_path / "external_baseline_batches.jsonl").read_text().splitlines()]
    assert records[0]["pre_batch_state_sha256"] == pre_hash
    assert records[0]["post_batch_state_sha256"] == trainer.external_baseline.state_sha256


def test_native_accumulation_buffer_updates_baseline_once_and_keeps_its_old_advantages(
    tmp_path, monkeypatch
):
    trainer = tiny_trainer(tmp_path, generations=1, batch=2, accumulation=2)
    seen = deterministic_generation(trainer, monkeypatch)
    loader = iter(trainer.get_train_dataloader())
    first = trainer._prepare_inputs(next(loader))
    assert seen == [0]
    assert trainer.external_baseline.revision == 1
    second = trainer._prepare_inputs(next(loader))
    assert seen == [0]  # Native TRL consumes the other half of the same buffer.
    assert trainer.external_baseline.revision == 1
    assert torch.equal(first["advantages"], torch.full((2,), -0.5))
    assert torch.equal(second["advantages"], torch.full((2,), -0.5))
    # A later buffer uses the now-updated baseline, despite unique prompt IDs.
    third = trainer._prepare_inputs(next(loader))
    assert seen == [0, 1]
    assert trainer.external_baseline.revision == 2
    assert trainer.external_baseline.seen_samples == 8
    assert torch.equal(third["advantages"], torch.zeros(2))


def test_failed_generation_does_not_advance_or_leave_active_tracker(tmp_path, monkeypatch):
    trainer = tiny_trainer(tmp_path)
    before = trainer.external_baseline.state_dict()

    def fail(prompts, images):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(trainer, "_generate", fail)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        trainer._generate_and_score_completions([row(0), row(1)])
    assert not trainer._baseline_generation_active
    assert trainer.external_baseline.state_dict() == before


def test_nonfinite_reward_rejects_entire_global_buffer_without_baseline_update(
    tmp_path, monkeypatch
):
    trainer = tiny_trainer(tmp_path)
    deterministic_generation(trainer, monkeypatch)
    before = trainer.external_baseline.state_dict()
    with pytest.raises(RuntimeError, match="nonfinite"):
        trainer._generate_and_score_completions([row(0, reward=math.nan), row(1)])
    assert trainer.external_baseline.state_dict() == before


def test_grouped_input_identity_failure_is_detected_before_generation(tmp_path, monkeypatch):
    trainer = tiny_trainer(tmp_path, generations=2)
    seen = deterministic_generation(trainer, monkeypatch)
    with pytest.raises(RuntimeError, match="sibling prompt identity"):
        trainer._generate_and_score_completions([row(0), row(1)])
    assert seen == []
    assert trainer.external_baseline.revision == 0


def test_cvrp_size_features_exclude_depot_at_bucket_boundary(tmp_path):
    trainer = tiny_trainer(tmp_path)
    metadata = trainer._input_metadata([row(0, task="cvrp", size=25)])
    assert metadata[0][:2] == ("cvrp", 25)


def test_rloo_constructor_still_rejects_single_sibling(tmp_path):
    with pytest.raises(ValueError, match="RLOO requires at least two siblings"):
        tiny_trainer(tmp_path, estimator="rloo", generations=1)


def metadata_trainer(gather, *, generations=1):
    trainer = object.__new__(ExternalBaselineGRPOTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=2, process_index=0)
    trainer.num_generations = generations
    trainer._gather_objects = gather
    return trainer


def test_rank_local_invalid_metadata_still_enters_collective_with_no_partial_rows():
    calls = []

    def gather(payload):
        calls.append(payload)
        return [payload, {"rows": [("mis", 8, "remote_prompt")], "error": None}]

    trainer = metadata_trainer(gather)
    with pytest.raises(RuntimeError, match="validation failed across ranks"):
        trainer._input_metadata([row(0), {"task_type": "unknown"}])
    assert len(calls) == 1
    assert calls[0]["rows"] == []
    assert calls[0]["error"].startswith("ValueError:")


def test_rank_with_valid_metadata_raises_the_other_ranks_error_after_collective():
    calls = []

    def gather(payload):
        calls.append(payload)
        return [payload, {"rows": [], "error": "ValueError: rank1 bad coordinates"}]

    trainer = metadata_trainer(gather)
    with pytest.raises(RuntimeError, match="rank1 bad coordinates"):
        trainer._input_metadata([row(0)])
    assert len(calls) == 1
    assert calls[0]["error"] is None
    assert len(calls[0]["rows"]) == 1


@pytest.mark.parametrize("inputs", [[], None, [None]])
def test_even_empty_or_malformed_local_batches_take_the_collective_error_path(inputs):
    calls = []

    def gather(payload):
        calls.append(payload)
        return [payload, {"rows": [], "error": "ValueError: remote bad batch"}]

    with pytest.raises(RuntimeError, match="validation failed across ranks"):
        metadata_trainer(gather)._input_metadata(inputs)
    assert len(calls) == 1 and calls[0]["rows"] == []
    assert calls[0]["error"] is not None


def test_global_group_identity_and_lengths_are_validated_after_collective():
    valid_remote = {"rows": [("mis", 8, "different_prompt")], "error": None}
    trainer = metadata_trainer(lambda local: [local, valid_remote], generations=2)
    with pytest.raises(RuntimeError, match="sibling prompt identity"):
        trainer._input_metadata([row(0)])
    trainer = metadata_trainer(lambda local: [local, {"rows": [], "error": None}])
    with pytest.raises(RuntimeError, match="Rank rollout sizes differ"):
        trainer._input_metadata([row(0)])


def test_global_metadata_preserves_rank_order_for_reward_alignment():
    remote = {"rows": [("mis", 12, "remote_prompt")], "error": None}
    trainer = metadata_trainer(lambda local: [local, remote])
    metadata = trainer._input_metadata([row(0)])
    assert metadata[0][:2] == ("mis", 8)
    assert metadata[1] == ("mis", 12, "remote_prompt")


@pytest.mark.parametrize("task,instance,expected_size", [
    ("tsp", [[0, 0], [1, 0], [0, 1]], 3),
    ("op", [[[0, 0], [1, 0], [0, 1]], [0, 1, 1], 4], 3),
    ("jssp", [[0, 1, 1, 2], [1, 1, 0, 2]], 4),
])
def test_new_task_metadata_stays_collective_and_input_only(task, instance, expected_size):
    calls = []

    def gather(payload):
        calls.append(payload)
        return [payload, payload]

    trainer = metadata_trainer(gather, generations=2)
    item = {"task_type": task, "instance_json": json.dumps(instance),
            "input": "the instance", "prompt": "solve", "answer": "ignored"}
    before = trainer._input_metadata([item])
    assert before[0][:2] == (task, expected_size)
    assert before[0] == before[1]
    item.update(answer="999999", expected_reward=-1, completion="ignored")
    assert trainer._input_metadata([item]) == before
    assert len(calls) == 2 and all(call["error"] is None for call in calls)


@pytest.mark.parametrize("task,instance", [("tsp", []), ("op", [[], [], 1]),
                                           ("jssp", [[0, 1, 2]])])
def test_new_task_bad_metadata_enters_collective_before_failing(task, instance):
    calls = []

    def gather(payload):
        calls.append(payload)
        return [payload, {"rows": [], "error": None}]

    with pytest.raises(RuntimeError, match="validation failed across ranks"):
        metadata_trainer(gather)._input_metadata([
            {"task_type": task, "instance_json": json.dumps(instance)}])
    assert len(calls) == 1
    assert calls[0]["rows"] == [] and calls[0]["error"].startswith("ValueError:")


def test_malformed_collective_envelope_is_rejected():
    trainer = metadata_trainer(lambda local: [local, {"rows": []}])
    with pytest.raises(RuntimeError, match="Invalid gathered"):
        trainer._input_metadata([row(0)])
