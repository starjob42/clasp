"""Compare real accumulated DDP teacher gradients with a whole-window objective."""
import copy
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl.models.utils import unwrap_model_for_generation

from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer
from test_verified_teacher_selection_integration import _attach, _Tokenizer, _trainer


class _Accelerator:
    num_processes = 2
    state = SimpleNamespace(deepspeed_plugin=None)

    @staticmethod
    def unwrap_model(model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    @staticmethod
    def gather(value):
        gathered = [torch.empty_like(value) for _ in range(2)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered)


def _check_distributed_attachment(rank):
    obj = _trainer(rank, 2)
    obj.grpo_verified_teacher_selection = "feasibility_incumbent"
    raw = ["Set: [0]", "Set: [99]"] if rank == 0 else ["Set: [0, 1, 2, 3, 4]"]
    teachers = ["Set: [0, 1]", None] if rank == 0 else [None]
    out = _attach(obj, raw, teachers)
    ids, mask = out["grpo_verified_mis_completion_ids"], out["grpo_verified_mis_completion_mask"]
    chosen = [values[keep.bool()].tolist() for values, keep in zip(ids, mask) if keep.any()]
    assert _Tokenizer.batch_decode(chosen) == (["Set: [0, 1, 2, 3, 4], Objective: 5"] if rank == 1 else [])
    selected = torch.tensor([len(chosen)])
    dist.all_reduce(selected)
    assert selected.item() == 1

    for task, instance in (
        ("mis", {"num_nodes": 5, "edges": [[0, 1], [1, 2], [3, 4]]}),
        ("cvrp", [[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 2, 2, 2], 3]),
    ):
        obj = _trainer(rank, 2)
        obj.grpo_verified_teacher_selection = "feasibility_incumbent"
        construct = obj._verified_constructive_target
        calls = []

        def construct_once(*args):
            calls.append(args)
            return construct(*args)

        obj._verified_constructive_target = construct_once
        raw = ["malformed", "also malformed"] if rank == 0 else ["malformed elsewhere"]
        out = _attach(obj, raw, [None] * len(raw), task=task, instances=[instance] * len(raw))
        ids, mask = [out[f"grpo_verified_{task}_completion_{suffix}"] for suffix in ("ids", "mask")]
        chosen = [values[keep.bool()].tolist() for values, keep in zip(ids, mask) if keep.any()]
        assert len(calls) == len(chosen) == (1 if rank == 0 else 0)
        for text in _Tokenizer.batch_decode(chosen):
            assert obj._verified_completion_objective(task, text, instance, "") is not None
        totals = torch.tensor([len(calls), len(chosen), int(out[f"grpo_verified_{task}_fallbacks"].sum())])
        dist.all_reduce(totals)
        assert totals.tolist() == [1, 1, 1]


def _batch(window, rank, micro):
    # Entries are valid target lengths for the two rows. Layouts cover an
    # uneven window with a globally empty middle microbatch, one teacher in
    # the entire window, no teachers anywhere, and uneven dense participation.
    layouts = (
        (((3, 1), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (2, 0))),
        (((0, 0), (0, 1)), ((0, 0), (0, 0)), ((0, 0), (0, 0))),
        (((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0))),
        (((1, 3), (2, 0)), ((0, 0), (3, 1)), ((2, 0), (0, 0))),
    )
    prompt = (torch.arange(8).reshape(2, 4) + 2 + rank * 5 + micro + window) % 43
    completion = (prompt[:, :3] + 11 + window) % 43
    target = (completion + 7 + rank) % 43
    target_mask = torch.zeros_like(target)
    for index, length in enumerate(layouts[window][micro][rank]):
        target_mask[index, :length] = 1
    return {
        "prompt": prompt, "completion": completion, "target": target, "target_mask": target_mask,
        "advantages": torch.tensor([1.0 + rank * 0.2, -0.7 - micro * 0.1], dtype=torch.float64),
    }


def _manual_logps(model, prompt, completion, completion_mask, temperature):
    """Independent full-logit likelihood, without the trainer's loss helper."""
    ids = torch.cat((prompt, completion), dim=1)
    attention = torch.cat((torch.ones_like(prompt), completion_mask), dim=1)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    prediction = logits[:, prompt.shape[1] - 1:-1] / temperature
    return prediction.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)


def _whole_window_reference(policy, batches, temperature, teacher_weight):
    prompt = torch.cat([batch["prompt"] for batch in batches])
    completion = torch.cat([batch["completion"] for batch in batches])
    advantages = torch.cat([batch["advantages"] for batch in batches])
    logps = _manual_logps(policy, prompt, completion, torch.ones_like(completion), temperature)
    # All microbatches have two policy rows; concatenating the complete window
    # exactly reproduces the policy mean over two ranks and three microbatches.
    loss = -(logps.mean(1) * advantages).mean()
    target = torch.cat([batch["target"] for batch in batches])
    mask = torch.cat([batch["target_mask"] for batch in batches])
    selected = mask.any(1)
    if selected.any():
        teacher_logps = _manual_logps(policy, prompt[selected], target[selected], mask[selected], temperature)
        selected_nll = -(teacher_logps * mask[selected]).sum(1) / mask[selected].sum(1)
        loss = loss + teacher_weight * selected_nll.mean()
    return loss


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        _check_distributed_attachment(rank)
        torch.manual_seed(19)
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=43, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            attn_implementation="eager", attention_dropout=0,
        )).to(dtype=torch.float64)
        policy = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=4, lora_dropout=0, bias="all", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.enable_input_require_grads()
        policy.train()
        reference = copy.deepcopy(policy)
        model = torch.nn.parallel.DistributedDataParallel(policy, find_unused_parameters=False, bucket_cap_mb=0.01)
        obj = object.__new__(PrimalDualGRPOTrainer)
        obj.model = policy
        obj.accelerator = _Accelerator()
        obj.processing_class = _Tokenizer()
        obj.args = SimpleNamespace(
            report_to=[], gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=3, steps_per_generation=3, per_device_train_batch_size=2, num_iterations=1,
        )
        obj.current_gradient_accumulation_steps = 3
        obj.model_kwarg_keys = ["logits_to_keep"]
        obj.temperature = 0.7
        obj.grpo_verified_improvement = True
        obj.grpo_verified_improvement_weight = 0.05
        obj.grpo_verified_improvement_max_tokens = 3000
        obj.grpo_verified_improvement_max_passes = 2
        obj.grpo_verified_teacher_selection = "feasibility_incumbent"
        obj.grpo_verified_teacher_normalization = "optimizer_window"
        obj._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        reference_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.01)
        for window in range(4):
            # Exercise the installed TRL lifecycle and the actual shared fix.
            with unwrap_model_for_generation(model, obj.accelerator):
                assert not policy.is_gradient_checkpointing
            obj._restore_configured_gradient_checkpointing()
            assert policy.base_model.model.model.layers[0]._gradient_checkpointing_func.keywords["use_reentrant"] is False
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            batches = [_batch(window, other_rank, micro) for other_rank in range(2) for micro in range(3)]
            window_count = sum(int(batch["target_mask"].any(1).sum()) for batch in batches)
            assert window_count == (3, 1, 0, 6)[window]
            for micro in range(3):
                batch = _batch(window, rank, micro)
                prompt, completion = batch["prompt"], batch["completion"]
                with model.no_sync() if micro < 2 else nullcontext():
                    logps, _ = obj._get_per_token_logps_and_entropies(
                        model, torch.cat((prompt, completion), dim=1),
                        torch.ones(2, 7, dtype=torch.long), 3,
                    )
                    inputs = {
                        "grpo_verified_cvrp_window_count": torch.zeros(2, dtype=torch.long),
                        "grpo_verified_mis_window_count": torch.full((2,), window_count, dtype=torch.long),
                        "grpo_verified_mis_completion_ids": batch["target"],
                        "grpo_verified_mis_completion_mask": batch["target_mask"],
                    }
                    teacher_loss, stats = obj._compute_improvement_distillation_losses(
                        model, inputs, prompt, torch.ones_like(prompt), completion, logps,
                    )
                    assert stats["grpo/grpo_verified_mis_window_teacher_count"] == window_count
                    if window_count == 0:
                        assert teacher_loss.item() == 0
                    policy_loss = -(logps.mean(1) * batch["advantages"]).mean()
                    ((policy_loss + teacher_loss) / 3).backward()
            _whole_window_reference(reference, batches, obj.temperature, obj.grpo_verified_improvement_weight).backward()
            reference_parameters = dict(reference.named_parameters())
            total_nonzero = 0
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    continue
                expected = reference_parameters[name]
                assert parameter.grad is not None and expected.grad is not None, name
                torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-9, rtol=2e-7, msg=name)
                total_nonzero += int(torch.count_nonzero(parameter.grad))
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                assert torch.equal(gathered[0].contiguous().view(torch.uint8),
                                   gathered[1].contiguous().view(torch.uint8)), name
            assert total_nonzero > 0
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in policy.named_parameters():
                torch.testing.assert_close(parameter, reference_parameters[name], atol=2e-9, rtol=2e-7, msg=name)
    finally:
        dist.destroy_process_group()


def test_real_ddp_optimizer_window_matches_concatenated_selected_teacher_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
