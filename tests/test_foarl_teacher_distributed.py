"""Real DDP full-loss oracle for stock FOARL plus sparse verified teachers.

Only synthetic integer tokens/tiny randomly initialized models are used. The
oracle uses full causal logits, independently reduces PPO/KL/teacher terms, and
never calls either trainer's likelihood or loss helpers.
"""
import copy
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl import GRPOTrainer
from trl.models.utils import unwrap_model_for_generation

from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer
from test_verified_incumbent_distributed import _Accelerator, _batch
from test_verified_teacher_selection_integration import _Tokenizer


class _FullLossAccelerator(_Accelerator):
    @staticmethod
    def gather(value):
        # Stock TRL's logging gathers scalar KL/entropy/clip metrics too.
        return _Accelerator.gather(value.reshape(1) if value.ndim == 0 else value)


def _model():
    torch.manual_seed(113)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=43, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        # Float64 eager attention casts fully masked left-padding scores to
        # float32 -inf; its NaNs poison later layers. SDPA handles those rows
        # while retaining float64 precision for this independent CPU oracle.
        attn_implementation="sdpa", attention_dropout=0,
    )).to(dtype=torch.float64)
    policy = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0, bias="all", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    return policy


def _trainer(policy, reference, rank, weight):
    obj = object.__new__(VerifiedTeacherGRPOTrainer)
    obj.model, obj.ref_model = policy, reference
    obj.accelerator = _FullLossAccelerator()
    obj.accelerator.process_index = rank
    obj.args = SimpleNamespace(
        report_to=[], delta=None, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=3, steps_per_generation=3,
        per_device_train_batch_size=2, num_iterations=1,
    )
    obj.current_gradient_accumulation_steps = 3
    obj.model_kwarg_keys = ["logits_to_keep"]
    obj.temperature, obj.beta = 0.7, 0.05
    obj.top_entropy_quantile = 1.0
    obj.importance_sampling_level, obj.loss_type = "token", "grpo"
    obj.epsilon_low = obj.epsilon_high = 0.2
    obj.use_liger_loss = obj.use_vllm = False
    obj.processing_class = _Tokenizer()
    obj.grpo_verified_improvement = weight > 0
    obj.grpo_verified_improvement_weight = weight
    obj.grpo_verified_improvement_max_tokens = 3000
    obj.grpo_verified_improvement_max_passes = 2
    obj.grpo_verified_teacher_selection = "feasibility_incumbent"
    obj.grpo_verified_teacher_normalization = "optimizer_window"
    obj._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return obj


def _masked_batch(window, rank, micro):
    batch = _batch(window, rank, micro)
    batch["prompt_mask"] = torch.ones_like(batch["prompt"])
    batch["prompt_mask"][0, :1 + (micro % 2)] = 0
    batch["prompt_mask"][1, :rank] = 0
    batch["prompt"] *= batch["prompt_mask"]
    batch["completion_mask"] = torch.ones_like(batch["completion"])
    batch["completion_mask"][0, 2:] = micro % 2
    batch["completion_mask"][1, 1 + ((window + micro) % 3):] = 0
    return batch


def _full_logps(model, prompt, prompt_mask, completion, completion_mask, temperature):
    inputs = torch.cat((prompt, completion), dim=1)
    attention = torch.cat((prompt_mask, completion_mask), dim=1)
    logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
    # Target token zero is predicted at the last prompt position, including
    # when prompts are left padded or selected teacher rows have different lengths.
    predictions = logits[:, prompt.shape[1] - 1:-1] / temperature
    return predictions.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)


def _prepare_logps(policy, frozen_reference, batches, temperature, clipped):
    with torch.no_grad():
        for batch in batches:
            arguments = (batch["prompt"], batch["prompt_mask"], batch["completion"],
                         batch["completion_mask"], temperature)
            batch["reference_logps"] = _full_logps(frozen_reference, *arguments)
            assert torch.isfinite(batch["reference_logps"]).all()
            if clipped:
                # Both active and clipped tokens exist for each advantage sign.
                # These are fixed old-policy scores, not current differentiable tensors.
                ratios = torch.tensor([[1.5, 1.0, 0.5], [0.5, 1.0, 1.5]], dtype=torch.float64)
                batch["old_logps"] = _full_logps(policy, *arguments) - ratios.log()
            else:
                batch["old_logps"] = None


def _oracle(policy, batches, temperature, beta, teacher_weight):
    def cat(name):
        return torch.cat([batch[name] for batch in batches])

    prompt, prompt_mask = cat("prompt"), cat("prompt_mask")
    completion, mask = cat("completion"), cat("completion_mask")
    logps = _full_logps(policy, prompt, prompt_mask, completion, mask, temperature)
    old = logps.detach() if batches[0]["old_logps"] is None else cat("old_logps")
    ratio = (logps - old).exp()
    advantage = cat("advantages").unsqueeze(1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(0.8, 1.2) * advantage
    policy_terms = -torch.minimum(unclipped, clipped)
    difference = cat("reference_logps") - logps
    kl_terms = difference.exp() - difference - 1
    loss = (((policy_terms + beta * kl_terms) * mask).sum(1) / mask.sum(1)).mean()
    assert (kl_terms * mask).sum().item() > 1e-10, "KL oracle must have a nonzero signal"
    if teacher_weight:
        target, target_mask = cat("target"), cat("target_mask")
        selected = target_mask.any(1)
        if selected.any():
            teacher_logps = _full_logps(policy, prompt[selected], prompt_mask[selected],
                                       target[selected], target_mask[selected], temperature)
            teacher_nll = -(teacher_logps * target_mask[selected]).sum(1) / target_mask[selected].sum(1)
            loss = loss + teacher_weight * teacher_nll.mean()
    return loss


def _check_generation_restore(obj, ddp, policy, fail=False):
    marker = {"original_stock_output": torch.tensor([7])}

    def generation(_self, _inputs):
        with unwrap_model_for_generation(ddp, obj.accelerator):
            assert not policy.is_gradient_checkpointing
        # Exercise the actual installed TRL context that causes the regression.
        assert policy.base_model.model.model.layers[0]._gradient_checkpointing_func.keywords["use_reentrant"] is True
        if fail:
            raise RuntimeError("synthetic generation failure")
        return marker

    def attach(_inputs, _output, _task_types):
        assert policy.base_model.model.model.layers[0]._gradient_checkpointing_func.keywords["use_reentrant"] is False

    with patch.object(GRPOTrainer, "_generate_and_score_completions", generation), \
            patch.object(obj, "_attach_verified_improvement_tensors", side_effect=attach) as attachment:
        if fail:
            with pytest.raises(RuntimeError, match="synthetic generation failure"):
                obj._generate_and_score_completions([{"task_type": "mis"}])
            attachment.assert_not_called()
        else:
            assert obj._generate_and_score_completions([{"task_type": "mis"}]) is marker
            assert attachment.call_count == int(obj.grpo_verified_improvement)
    assert policy.base_model.model.model.layers[0]._gradient_checkpointing_func.keywords["use_reentrant"] is False


def _worker(rank, rendezvous, weight):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        policy = _model()
        frozen_reference = copy.deepcopy(policy).eval()
        frozen_reference.requires_grad_(False)
        reference_state = {name: value.detach().clone() for name, value in frozen_reference.state_dict().items()}
        # Simulate a policy that has already moved from its immutable SFT reference;
        # otherwise KL has zero gradient and omitting beta could escape this test.
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if parameter.requires_grad and "lora_B" in name:
                    parameter.add_((torch.arange(parameter.numel()).reshape(parameter.shape) % 7 - 3) * 0.007)
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.enable_input_require_grads()
        policy.train()
        oracle_policy = copy.deepcopy(policy)
        ddp = torch.nn.parallel.DistributedDataParallel(policy, find_unused_parameters=False, bucket_cap_mb=0.01)
        obj = _trainer(policy, frozen_reference, rank, weight)
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        oracle_optimizer = torch.optim.SGD([p for p in oracle_policy.parameters() if p.requires_grad], lr=0.01)
        forward_shapes = []

        def record_forward(_model, _args, kwargs):
            forward_shapes.append((tuple(kwargs["input_ids"].shape), kwargs["attention_mask"].detach().clone()))

        hook = ddp.register_forward_pre_hook(record_forward, with_kwargs=True)
        dummy_forwards = 0
        for window in range(4):
            _check_generation_restore(obj, ddp, policy)
            optimizer.zero_grad()
            oracle_optimizer.zero_grad()
            batches = [_masked_batch(window, other_rank, micro) for other_rank in range(2) for micro in range(3)]
            window_count = sum(int(batch["target_mask"].any(1).sum()) for batch in batches)
            assert window_count == (3, 1, 0, 6)[window]
            _prepare_logps(policy, frozen_reference, batches, obj.temperature, clipped=window % 2 == 1)
            local_loss = torch.zeros((), dtype=torch.float64)
            for micro in range(3):
                batch = batches[rank * 3 + micro]
                global_micro_count = sum(int(batches[r * 3 + micro]["target_mask"].any(1).sum()) for r in range(2))
                inputs = {
                    "prompt_ids": batch["prompt"], "prompt_mask": batch["prompt_mask"],
                    "completion_ids": batch["completion"], "completion_mask": batch["completion_mask"],
                    "advantages": batch["advantages"], "old_per_token_logps": batch["old_logps"],
                    "ref_per_token_logps": batch["reference_logps"],
                    "grpo_verified_cvrp_window_count": torch.zeros(2, dtype=torch.long),
                    "grpo_verified_mis_window_count": torch.full((2,), window_count, dtype=torch.long),
                    "grpo_verified_mis_completion_ids": batch["target"],
                    "grpo_verified_mis_completion_mask": batch["target_mask"],
                }
                retained_inputs = copy.deepcopy(inputs)
                forward_shapes.clear()
                with ddp.no_sync() if micro < 2 else nullcontext():
                    loss = obj.compute_loss(ddp, inputs)
                    local_loss += loss.detach()
                    assert len(forward_shapes) == 1 + int(weight > 0 and global_micro_count > 0)
                    if weight > 0 and global_micro_count > 0 and not batch["target_mask"].any():
                        shape, attention = forward_shapes[1]
                        assert shape == (1, batch["prompt"].shape[1] + 1)
                        assert attention[0, -1].item() == 0
                        dummy_forwards += 1
                    for key, value in retained_inputs.items():
                        if value is None:
                            assert inputs[key] is None, key
                        else:
                            assert torch.equal(inputs[key], value), (key, value, inputs[key])
                    loss.backward()
            expected_loss = _oracle(oracle_policy, batches, obj.temperature, obj.beta, weight)
            dist.all_reduce(local_loss)
            torch.testing.assert_close(local_loss / 2, expected_loss.detach(), atol=2e-11, rtol=2e-9)
            expected_loss.backward()
            expected_parameters = dict(oracle_policy.named_parameters())
            nonzero = 0
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    continue
                expected = expected_parameters[name]
                assert parameter.grad is not None and expected.grad is not None, name
                torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-9, rtol=2e-7, msg=name)
                nonzero += int(torch.count_nonzero(parameter.grad))
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                assert torch.equal(gathered[0].contiguous().view(torch.uint8),
                                   gathered[1].contiguous().view(torch.uint8)), name
            assert nonzero > 0
            if window % 2:
                assert obj._metrics["train"]["clip_ratio/low_mean"][-1] > 0
                assert obj._metrics["train"]["clip_ratio/high_mean"][-1] > 0
            optimizer.step()
            oracle_optimizer.step()
            for name, parameter in policy.named_parameters():
                torch.testing.assert_close(parameter, expected_parameters[name], atol=2e-9, rtol=2e-7, msg=name)
        assert (dummy_forwards > 0) == (weight > 0)
        _check_generation_restore(obj, ddp, policy, fail=True)
        hook.remove()
        for name, value in frozen_reference.state_dict().items():
            assert torch.equal(value, reference_state[name]), name
        assert all(parameter.grad is None for parameter in frozen_reference.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("teacher_weight", [0.0, 0.05])
def test_full_foarl_teacher_loss_matches_independent_accumulated_ddp_oracle(tmp_path, monkeypatch, teacher_weight):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), teacher_weight), nprocs=2, join=True)
