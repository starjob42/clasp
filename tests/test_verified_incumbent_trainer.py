"""Behavioral checks for feasibility-triggered, optimizer-window teachers."""
from types import SimpleNamespace

import pytest
import torch

from test_verified_teacher_selection_integration import _attach, _trainer, _Tokenizer


def incumbent_trainer():
    obj = _trainer()
    obj.grpo_verified_teacher_selection = "feasibility_incumbent"
    return obj


def selected_texts(output, task="mis"):
    ids, mask = [output[f"grpo_verified_{task}_completion_{suffix}"] for suffix in ("ids", "mask")]
    return _Tokenizer.batch_decode([row[keep.bool()].tolist() for row, keep in zip(ids, mask) if keep.any()])


def test_optimal_incumbent_teaches_invalid_sibling_without_local_search_gain():
    obj = incumbent_trainer()
    out = _attach(obj, ["Set: [0, 1, 2, 3, 4], Objective: 999", "Set: [99]"], [None, "Set: [0]"])
    assert selected_texts(out) == ["Set: [0, 1, 2, 3, 4], Objective: 5"]
    assert out["grpo_verified_mis_fallbacks"].sum().item() == 0


def test_feasible_groups_without_improvement_do_not_distill_themselves():
    out = _attach(incumbent_trainer(), ["Set: [0]", "Set: [1]"], [None, None])
    assert selected_texts(out) == []


def test_best_improvement_still_wins_over_raw_incumbents_in_mixed_group():
    out = _attach(incumbent_trainer(), ["Set: [0, 1]", "Set: [99]"], ["Set: [0, 1, 2]", "Set: [0]"])
    assert selected_texts(out) == ["Set: [0, 1, 2]"]


@pytest.mark.parametrize("task,instance", [
    ("mis", {"num_nodes": 5, "edges": [[0, 1], [1, 2], [3, 4]]}),
    ("cvrp", [[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 2, 2, 2], 3]),
])
def test_all_malformed_group_gets_one_independently_verified_bounded_fallback(task, instance):
    obj = incumbent_trainer()
    calls = []
    original = obj._verified_constructive_target

    def construct(*args):
        calls.append(args)
        return original(*args)

    obj._verified_constructive_target = construct
    out = _attach(obj, ["nonsense", "nonsense again"], [None, None], task=task, instances=[instance] * 2)
    texts = selected_texts(out, task)
    assert len(calls) == len(texts) == 1
    assert obj._verified_completion_objective(task, texts[0], instance, "") is not None
    assert out[f"grpo_verified_{task}_fallbacks"].sum().item() == 1


def test_existing_repair_avoids_constructive_fallback():
    obj = incumbent_trainer()
    obj._verified_constructive_target = lambda *a: pytest.fail("existing verified repair must be reused")
    out = _attach(obj, ["Set: [99]", "bad"], ["Set: [0]", None])
    assert selected_texts(out) == ["Set: [0]"]


def test_overlong_targets_do_not_train_on_a_truncated_or_inferior_solution():
    obj = incumbent_trainer()
    obj.grpo_verified_improvement_max_tokens = 25
    out = _attach(obj, ["Set: [0, 1, 2, 3, 4]", "Set: [99]"], [None, "Set: [0]"])
    assert selected_texts(out) == []


def window_trainer(batch=1, accumulation=3):
    obj = incumbent_trainer()
    obj.grpo_verified_teacher_normalization = "optimizer_window"
    obj.args = SimpleNamespace(gradient_accumulation_steps=accumulation,
                               steps_per_generation=accumulation, per_device_train_batch_size=batch,
                               num_iterations=1)
    obj.current_gradient_accumulation_steps = accumulation
    obj.accelerator.gather = lambda value: value
    return obj


def test_window_count_survives_real_trl_shuffle_and_split():
    from test_verified_teacher_integration import runtime_row_transforms

    obj = window_trainer()
    out = _attach(obj, ["Set: [0, 1, 2, 3, 4]", "Set: [99]", "Set: [98]"], [None] * 3)
    shuffle, split = runtime_row_transforms()
    assert out["grpo_verified_mis_window_count"].tolist() == [1, 1, 1]
    assert out["grpo_verified_cvrp_window_count"].tolist() == [0, 0, 0]
    chunks = split(shuffle(out), 3)
    assert [chunk["grpo_verified_mis_window_count"].item() for chunk in chunks] == [1, 1, 1]
    assert sum(chunk["grpo_verified_mis_completion_mask"].any().item() for chunk in chunks) == 1


@pytest.mark.parametrize("field,value", [("steps_per_generation", 1), ("num_iterations", 2)])
def test_window_mode_rejects_misaligned_or_replayed_generation(field, value):
    obj = window_trainer()
    setattr(obj.args, field, value)
    with pytest.raises(ValueError, match="Optimizer-window"):
        obj._validate_verified_teacher_window(3)


def test_window_mode_rejects_partial_generation_buffer():
    with pytest.raises(ValueError, match="complete local generation buffer"):
        window_trainer()._validate_verified_teacher_window(2)


def test_sparse_window_has_exact_selected_target_mean_gradient():
    # Microbatches have 1, 0, and 2 teachers with different per-target gradients.
    # A mean of occupied-microbatch means would yield a different answer.
    obj = window_trainer(batch=2)
    spec = obj._improvement_distillation_specs()[1]
    obj._improvement_distillation_specs = lambda: [spec]
    parameter = torch.tensor(0.4, requires_grad=True)
    obj._get_per_token_logps_and_entropies = lambda model, ids, mask, length, **kw: (
        -parameter * ids[:, -length:].float(), None)
    for mask, targets in [([[1, 0], [0, 0]], [[2, 0], [0, 0]]),
                          ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
                          ([[1, 1], [1, 0]], [[4, 6], [8, 0]])]:
        inputs = {"grpo_verified_mis_completion_ids": torch.tensor(targets),
                  "grpo_verified_mis_completion_mask": torch.tensor(mask),
                  "grpo_verified_mis_window_count": torch.full((2,), 3, dtype=torch.long)}
        prompt = torch.ones(2, 2, dtype=torch.long)
        loss, _ = obj._compute_improvement_distillation_losses(
            obj.model, inputs, prompt, prompt, prompt, parameter.expand(2, 2))
        (loss / obj.current_gradient_accumulation_steps).backward()
    assert parameter.grad.item() == pytest.approx(0.05 * (2 + 5 + 8) / 3)


def test_fast_tokenizer_backend_state_is_restored_even_for_rejected_target():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "Set": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.backend_tokenizer.enable_padding(direction="left", pad_id=1, pad_token="[PAD]")
    tokenizer.backend_tokenizer.enable_truncation(max_length=123)
    before = tokenizer.backend_tokenizer.to_str()
    obj = incumbent_trainer()
    obj.processing_class = tokenizer
    assert obj._tokenize_verified_target("Set", 1) == []
    assert tokenizer.backend_tokenizer.to_str() == before
