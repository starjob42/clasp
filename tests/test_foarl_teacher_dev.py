"""Stock-loss identity and independently calculated sparse teacher gradients."""
import copy
import os
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer

from clasp.baselines.foarl_teacher_dev import build_runtime
from clasp.config import finalize_grpo_args
from clasp.entrypoints.train_foarl_teacher_dev import build_parser
from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer
from test_verified_teacher_selection_integration import _Tokenizer, _attach


def make_trainer(weight):
    obj = object.__new__(VerifiedTeacherGRPOTrainer)
    obj.model = SimpleNamespace(training=True)
    obj.args = SimpleNamespace(report_to=[], delta=None, gradient_accumulation_steps=3,
                               steps_per_generation=3, per_device_train_batch_size=2,
                               num_iterations=1)
    obj.accelerator = SimpleNamespace(gather=lambda x: x, num_processes=1, process_index=0)
    obj._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    obj.use_liger_loss = obj.use_vllm = False
    obj.beta = .05
    obj.top_entropy_quantile = 1.0
    obj.importance_sampling_level = "token"
    obj.epsilon_low = obj.epsilon_high = .2
    obj.loss_type = "grpo"
    obj.current_gradient_accumulation_steps = 3
    obj.processing_class = _Tokenizer()
    obj.grpo_verified_improvement = weight > 0
    obj.grpo_verified_improvement_weight = weight
    obj.grpo_verified_improvement_max_tokens = 3000
    obj.grpo_verified_improvement_max_passes = 2
    obj.grpo_verified_teacher_selection = "feasibility_incumbent"
    obj.grpo_verified_teacher_normalization = "optimizer_window"
    return obj


def inputs():
    return {"prompt_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "prompt_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "completion_ids": torch.tensor([[2, 3], [4, 0]]),
            "completion_mask": torch.tensor([[1, 1], [1, 0]]),
            "advantages": torch.tensor([1.2, -.8], dtype=torch.float64),
            "ref_per_token_logps": torch.tensor([[-.8, -.9], [-1., -.2]], dtype=torch.float64)}


def bind_logps(trainer, parameter):
    def logps(model, ids, mask, length, **kwargs):
        values = -parameter * ids[:, -length:].double()
        return values, torch.ones_like(values)
    trainer._get_per_token_logps_and_entropies = logps


@pytest.mark.parametrize("old", [None, [[-1., -1.], [-1., -1.]]])
def test_weight_zero_loss_and_gradient_are_bitwise_stock_trl(old):
    obj = make_trainer(0)
    batch = inputs()
    batch["old_per_token_logps"] = None if old is None else torch.tensor(old, dtype=torch.float64)
    p = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    bind_logps(obj, p)
    stock = GRPOTrainer.compute_loss(obj, obj.model, batch)
    expected = torch.autograd.grad(stock, p)[0]
    modified = obj.compute_loss(obj.model, batch)
    observed = torch.autograd.grad(modified, p)[0]
    assert torch.equal(stock, modified)
    assert torch.equal(expected, observed)


def test_teacher_adds_exact_window_mean_without_changing_foarl_loss():
    obj = make_trainer(.05)
    p = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    bind_logps(obj, p)
    accumulated, expected = [], []
    for mask, target in [([[1, 0], [0, 0]], [[2, 0], [0, 0]]),
                         ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
                         ([[1, 1], [1, 0]], [[4, 6], [8, 0]])]:
        batch = inputs()
        batch.update(grpo_verified_mis_completion_ids=torch.tensor(target),
                     grpo_verified_mis_completion_mask=torch.tensor(mask),
                     grpo_verified_mis_window_count=torch.full((2,), 3, dtype=torch.long),
                     grpo_verified_cvrp_window_count=torch.zeros(2, dtype=torch.long))
        unchanged = copy.deepcopy(batch)
        expected.append(GRPOTrainer.compute_loss(obj, obj.model, batch))
        accumulated.append(obj.compute_loss(obj.model, batch))
        assert all(torch.equal(batch[k], v) for k, v in unchanged.items())
    oracle = sum(expected) + .05 * p * (2 + 5 + 8) / 3
    torch.testing.assert_close(sum(accumulated), oracle, atol=1e-15, rtol=0)
    observed_grad = torch.autograd.grad(sum(accumulated), p)[0]
    oracle_grad = torch.autograd.grad(oracle, p)[0]
    torch.testing.assert_close(observed_grad, oracle_grad, atol=1e-15, rtol=0)


def test_teacher_attachment_preserves_stock_rewards_and_full_target():
    obj = make_trainer(.05)
    # This attachment-only test uses a single complete two-row window.
    obj.args.gradient_accumulation_steps = obj.args.steps_per_generation = 1
    obj.current_gradient_accumulation_steps = 1
    out = _attach(obj, ["Set: [0, 1, 2, 3, 4]", "Set: [99]"], [None, None])
    ids, mask = out["grpo_verified_mis_completion_ids"], out["grpo_verified_mis_completion_mask"]
    selected = [row[keep.bool()].tolist() for row, keep in zip(ids, mask) if keep.any()]
    assert _Tokenizer.batch_decode(selected) == ["Set: [0, 1, 2, 3, 4], Objective: 5"]
    assert selected[0][-1] == _Tokenizer.eos_token_id


@pytest.mark.parametrize("weight", [0., .05])
@pytest.mark.parametrize("diagnostic,stop", [(False, 100), (False, 200), (False, 400), (True, 5), (True, 10)])
def test_fixed_horizon_profile(weight, diagnostic, stop, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("GRPO_STOP_AT_GLOBAL_STEP", str(stop))
    monkeypatch.setenv("GRPO_STRICT_OPTIMIZER_RESUME", "1")
    monkeypatch.setenv("GRPO_DATA_SEED", "43")
    monkeypatch.setenv("GRPO_VERIFIED_IMPROVEMENT", "0")
    argv = ["--tasks", "mis", "--sft_checkpoint_path", "/tmp/mis",
            "--foarl_teacher_weight", str(weight), "--foarl_dev_source_manifest", "/tmp/SOURCE_MANIFEST.sha256",
            "--foarl_dev_source_manifest_sha256", "a" * 64]
    if diagnostic:
        argv += ["--foarl_teacher_diagnostic", "--save_steps", "5"]
    args = finalize_grpo_args(build_parser().parse_args(argv))
    binding = {"status": "passed_frozen_manifest_successor_source",
               "required_files": {"src/clasp/entrypoints/train_grpo.py": {"sha256": "a" * 64}}}
    runtime = build_runtime(binding, weight, diagnostic)
    runtime.validate_args(args)
    assert args.max_steps == 400
    assert runtime.metadata()["beta"] == .05
    args.max_steps = 100
    with pytest.raises(ValueError, match="recipe mismatch"):
        runtime.validate_args(args)
