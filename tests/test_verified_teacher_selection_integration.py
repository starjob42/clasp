"""Verify teacher selection against real text and cross-rank partial groups."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    @staticmethod
    def encode(text, **kwargs):
        return [ord(character) + 3 for character in text]

    @staticmethod
    def batch_decode(rows, **kwargs):
        return ["".join(chr(value - 3) for value in row if value > 2) for row in rows]


def _trainer(rank=0, world_size=1):
    obj = object.__new__(PrimalDualGRPOTrainer)
    obj.clasp_enabled = False
    obj.grpo_verified_improvement = True
    obj.grpo_verified_improvement_weight = 0.05
    obj.grpo_verified_improvement_max_tokens = 3000
    obj.grpo_verified_improvement_max_passes = 2
    obj.grpo_verified_teacher_selection = "best_per_prompt"
    obj.num_generations = 9
    obj.processing_class = _Tokenizer()
    obj.model = SimpleNamespace(training=True)
    obj.accelerator = SimpleNamespace(num_processes=world_size, process_index=rank)
    return obj


def _attach(obj, completions, teachers, *, instances=None, prompts=None, task="mis"):
    if instances is None:
        instances = [{"num_nodes": 5, "edges": []}] * len(completions)
    if prompts is None:
        prompts = ["Solve this graph."] * len(completions)
    inputs = [{"instance": instance, "prompt": prompt, "task_type": task}
              for instance, prompt in zip(instances, prompts)]
    encoded = [obj.processing_class.encode(text) for text in completions]
    width = max(map(len, encoded))
    raw_ids = torch.tensor([row + [0] * (width - len(row)) for row in encoded])
    advantages = torch.arange(len(encoded), dtype=torch.float32)
    old_logps = torch.full(raw_ids.shape, -0.5)
    output = {"completion_ids": raw_ids.clone(), "advantages": advantages.clone(),
              "old_per_token_logps": old_logps.clone()}
    teacher_iterator = iter(teachers)

    def improve(task, completion, instance, prompt):
        text = next(teacher_iterator)
        return None if text is None else (text, 0.1, 1, False)

    obj._verified_improvement_target = improve
    obj._attach_verified_improvement_tensors(inputs, output, [task] * len(encoded))
    assert torch.equal(output["completion_ids"], raw_ids)
    assert torch.equal(output["advantages"], advantages)
    assert torch.equal(output["old_per_token_logps"], old_logps)
    return output


def test_only_best_strict_group_improvement_is_retained_and_eos_survives():
    output = _attach(_trainer(), ["Set: [0]", "Set: [0, 1, 2]", "Set: [1]"],
                     ["Set: [0, 1]", "Set: [0, 1, 2, 3]", "Set: [0, 1, 2, 3]"])
    mask = output["grpo_verified_mis_completion_mask"]
    assert (mask.sum(1) > 0).tolist() == [False, True, False]
    assert output["grpo_verified_mis_eligible"].tolist() == [1, 1, 1]
    tokens = output["grpo_verified_mis_completion_ids"][1][mask[1].bool()].tolist()
    assert tokens[-1] == 2
    assert _Tokenizer.batch_decode([tokens]) == ["Set: [0, 1, 2, 3]"]


def test_raw_row_without_teacher_still_blocks_inferior_supervision():
    output = _attach(_trainer(), ["Set: [0]", "Set: [0, 1, 2, 3]"],
                     ["Set: [0, 1, 2]", None])
    assert output["grpo_verified_mis_completion_mask"].sum() == 0


def test_distinct_prompt_instances_do_not_share_teacher_competition():
    output = _attach(_trainer(), ["Set: [0]", "Set: [0]"],
                     ["Set: [0, 1]", "Set: [0, 1, 2]"],
                     instances=[{"num_nodes": 3, "edges": []}, {"num_nodes": 5, "edges": []}])
    assert (output["grpo_verified_mis_completion_mask"].sum(1) > 0).tolist() == [True, True]


def test_legacy_per_completion_preserves_all_targets():
    obj = _trainer()
    obj.grpo_verified_teacher_selection = "per_completion"
    obj._gather_verified_teacher_rows = lambda rows: pytest.fail("legacy path must not gather selection metadata")
    output = _attach(obj, ["Set: [0]", "Set: [0, 1, 2, 3]"], ["Set: [0, 1]", None])
    assert (output["grpo_verified_mis_completion_mask"].sum(1) > 0).tolist() == [True, False]
    assert "grpo_verified_mis_eligible" not in output


def test_selection_reverifies_targets_and_ignores_claimed_objective():
    obj = _trainer()
    with pytest.raises(RuntimeError, match="independent selection verification"):
        _attach(obj, ["Set: [0]"], ["Set: [0, 0], Objective: 9999"])
    output = _attach(obj, ["Set: [0, 1], Objective: 1"],
                     ["Set: [0, 1], Objective: 9999"])
    assert output["grpo_verified_mis_completion_mask"].sum() == 0


def test_cvrp_selection_recomputes_distance_and_gates_against_every_raw_route():
    instance = [[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 1, 1], 3]
    bad = "Routes: [[0, 1, 3, 2, 0]], Objective: 0"
    best = "Routes: [[0, 1, 2, 3, 0]], Objective: 99999"
    reverse_best = "Routes: [[0, 3, 2, 1, 0]], Objective: 99999"
    output = _attach(_trainer(), [bad, reverse_best], [best, None],
                     instances=[instance, instance], task="cvrp")
    assert output["grpo_verified_cvrp_completion_mask"].sum() == 0
    output = _attach(_trainer(), [bad, bad], [best, reverse_best],
                     instances=[instance, instance], task="cvrp")
    assert (output["grpo_verified_cvrp_completion_mask"].sum(1) > 0).tolist() == [True, False]


def test_distributed_selection_refuses_missing_process_group():
    obj = _trainer(world_size=2)
    with pytest.raises(RuntimeError, match="initialized process group"):
        obj._gather_verified_teacher_rows([])


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for case in ("remote_winner", "remote_incumbent", "all_invalid"):
            obj = _trainer(rank, 2)
            if case == "all_invalid":
                raw = ["Set: [99]", "Set: [98]"] if rank == 0 else ["Set: [97]"]
            else:
                raw = ["Set: [0]", "Set: [0, 1, 2]"] if rank == 0 else [
                    "Set: [0, 1, 2, 3, 4]" if case == "remote_incumbent" else "Set: [0, 1, 2, 3]"]
            teachers = ["Set: [0, 1]", "Set: [0, 1, 2, 3]"] if rank == 0 else [
                None if case == "remote_incumbent" else "Set: [0, 1, 2, 3, 4]"]
            output = _attach(obj, raw, teachers)
            mask = output["grpo_verified_mis_completion_mask"]
            selected = (mask.sum(1) > 0).sum()
            assert int(selected.item()) == (int(case != "remote_incumbent") if rank == 1 else 0)
            dist.all_reduce(selected)
            assert int(selected.item()) == int(case != "remote_incumbent")
    finally:
        dist.destroy_process_group()


def test_real_cross_rank_selection_with_unequal_partial_prompt_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
