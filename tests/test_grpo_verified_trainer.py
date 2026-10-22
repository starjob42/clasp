from types import SimpleNamespace

import pytest
import torch

from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        assert not kwargs.get("truncation", False)
        return [ord(c) + 3 for c in text]

    def batch_decode(self, rows, **kwargs):
        return ["sample" for _ in rows]


def trainer():
    obj = object.__new__(PrimalDualGRPOTrainer)
    obj.clasp_enabled = False
    obj.grpo_verified_improvement = True
    obj.grpo_verified_improvement_weight = 0.05
    obj.grpo_verified_improvement_max_tokens = 3000
    obj.grpo_verified_improvement_max_passes = 2
    obj.num_generations = 9
    obj.processing_class = Tokenizer()
    obj.model = SimpleNamespace(training=True)
    obj.accelerator = SimpleNamespace(gather=lambda x: x, num_processes=1)
    return obj


def test_default_plain_grpo_has_no_teacher_specs():
    obj = object.__new__(PrimalDualGRPOTrainer)
    obj.clasp_enabled = False
    assert obj._improvement_distillation_specs() == []
    assert len(trainer()._improvement_distillation_specs()) == 2


def test_teacher_rejects_truncation_and_preserves_eos():
    obj = trainer()
    assert obj._tokenize_verified_target("abc", 4)[-1] == 2
    assert obj._tokenize_verified_target("abc", 3) == []


def test_partial_groups_and_shuffled_instances_stay_row_aligned():
    obj = trainer()
    seen = []

    def improve(task, text, instance, prompt):
        seen.append(instance["id"])
        return str(instance["id"]), 0.1, 1, False

    obj._verified_improvement_target = improve
    inputs = [{"instance": {"id": i}, "task_type": "mis"} for i in [5, 1, 8]]
    raw = torch.tensor([[10], [11], [12]])
    advantages = torch.tensor([1.0, 0.0, -1.0])
    output = {"completion_ids": raw.clone(), "advantages": advantages.clone()}
    obj._attach_improvement_distillation_tensors(inputs, output, ["mis"] * 3)
    assert seen == [5, 1, 8]
    assert output["grpo_verified_mis_completion_ids"][:, 0].tolist() == [ord(str(i)) + 3 for i in seen]
    assert torch.equal(output["completion_ids"], raw)
    assert torch.equal(output["advantages"], advantages)


def test_metadata_mismatch_raises_instead_of_training_wrong_prompt():
    obj = trainer()
    with pytest.raises(ValueError, match="completion-aligned"):
        obj._attach_improvement_distillation_tensors(
            [{"instance": {}}], {"completion_ids": torch.ones(3, 1, dtype=torch.long)}, ["mis"]
        )


def test_loss_uses_global_teacher_count_and_backpropagates():
    obj = trainer()
    spec = obj._improvement_distillation_specs()[1]
    obj._improvement_distillation_specs = lambda: [spec]
    # Local rank has two teachers, the other rank has one.
    obj.accelerator = SimpleNamespace(gather=lambda x: torch.cat([x, torch.ones_like(x)]), num_processes=2)
    parameter = torch.tensor(-2.0, requires_grad=True)
    obj._get_per_token_logps_and_entropies = lambda model, ids, mask, length, **kw: (
        parameter.expand(ids.size(0), length), None
    )
    inputs = {"grpo_verified_mis_completion_ids": torch.ones(3, 2, dtype=torch.long),
              "grpo_verified_mis_completion_mask": torch.tensor([[1, 1], [0, 0], [1, 0]])}
    prompt = torch.ones(3, 2, dtype=torch.long)
    loss, stats = obj._compute_improvement_distillation_losses(
        obj.model, inputs, prompt, prompt, prompt, parameter.expand(3, 2)
    )
    assert loss.item() == pytest.approx(0.05 * 4 * 2 / 3)
    loss.backward()
    assert parameter.grad.item() == pytest.approx(-0.05 * 2 * 2 / 3)
    assert stats["clasp/grpo_verified_mis_valid_row_rate"] == pytest.approx(2 / 3)


def test_empty_rank_participates_with_zero_gradient_when_remote_has_teacher():
    obj = trainer()
    spec = obj._improvement_distillation_specs()[1]
    obj._improvement_distillation_specs = lambda: [spec]
    obj.accelerator = SimpleNamespace(gather=lambda x: torch.cat([x, torch.ones_like(x)]), num_processes=2)
    parameter = torch.tensor(-2.0, requires_grad=True)
    forwards = []

    def forward(model, ids, mask, length, **kw):
        forwards.append(ids.size(0))
        return parameter.expand(ids.size(0), length), None

    obj._get_per_token_logps_and_entropies = forward
    prompt = torch.ones(3, 2, dtype=torch.long)
    loss, _ = obj._compute_improvement_distillation_losses(
        obj.model, {}, prompt, prompt, prompt, parameter.expand(3, 2)
    )
    assert forwards == [1]
    assert loss.item() == 0
    loss.backward()
    assert parameter.grad.item() == 0
