"""Offline checks of the prospective pilot's common schedule and I/O guards."""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from datasets import Dataset

from clasp.baselines.sft_anchor_pilot import make_contract
from clasp.entrypoints import train_sft_anchor_pilot as entry


def contract(task="op"):
    train, validation = entry.smoke_rows(task)
    return make_contract(task, {"id": "local fixture", "sha256": "a" * 64},
                         train, validation, scope="system_smoke", horizon=4,
                         max_completion_tokens=1, train_seed=61)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("task", ["op", "cvrp"])
def test_sampler_exactly_preserves_declared_global_draws_through_native_buffer_reuse(world, task):
    pilot = contract(task)
    by_id = {row["instance_id"]: row for row in pilot["training_rows"]}
    draw_rows = [by_id[identifier] for draw in pilot["training"]["prompt_draws"] for identifier in draw]
    dataset = Dataset.from_list([{**row, "instance_json": json.dumps(row["instance_json"])}
                                 for row in draw_rows])
    accumulation = 36 // world
    trainer = SimpleNamespace(train_dataset=dataset,
                              args=SimpleNamespace(steps_per_generation=accumulation, seed=61))
    sampler = list(entry.OrderedObservationMixin._get_train_sampler(trainer))
    assert len(sampler) == 4 * 36 * accumulation
    local_width = 36 // world
    for step, draw in enumerate(pilot["training"]["prompt_draws"]):
        expected = [identifier for identifier in draw for _ in range(9)]
        # Accelerate shards contiguous local-generation batches across ranks.
        # Each accumulation microstep consumes another repeated sampler block;
        # GRPO reuses the first generated buffer until the optimizer step ends.
        for micro in range(accumulation):
            offset = step * 36 * accumulation + micro * 36
            parts = []
            for rank in range(world):
                indices = sampler[offset + rank * local_width:offset + (rank + 1) * local_width]
                parts.extend(dataset[index]["instance_id"] for index in indices)
            assert parts == expected


class ForwardStub:
    def _generate_and_score_completions(self, inputs):
        self.forward_calls += 1
        return {"completion_mask": torch.ones(len(inputs), 1, dtype=torch.long)}

    def compute_loss(self, *args, **kwargs):
        return self.fixture_loss


class ObservationHarness(entry.OrderedObservationMixin, ForwardStub):
    pass


def observation(tmp_path, pilot):
    trainer = object.__new__(ObservationHarness)
    trainer.pilot_contract = pilot
    trainer.actual_windows = []
    trainer.actual_generated_tokens = 0
    trainer.forward_calls = 0
    trainer.state = SimpleNamespace(global_step=0)
    trainer.accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), is_main_process=True)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.fixture_loss = torch.tensor(1.0)
    return trainer


def test_observation_records_exact_window_and_actual_generated_tokens(tmp_path):
    pilot = contract()
    trainer = observation(tmp_path, pilot)
    expected = [identifier for identifier in pilot["training"]["prompt_draws"][0] for _ in range(9)]
    trainer._generate_and_score_completions([{"instance_id": identifier} for identifier in expected])
    assert trainer.forward_calls == 1 and trainer.actual_windows == [expected]
    assert trainer.actual_generated_tokens == 36
    assert (tmp_path / "actual_prompt_windows.jsonl").is_file()


def test_schedule_mismatch_fails_before_reward_or_generation(tmp_path):
    trainer = observation(tmp_path, contract())
    with pytest.raises(RuntimeError, match="sealed common prompt schedule"):
        trainer._generate_and_score_completions([{"instance_id": "unexpected"}] * 36)
    assert trainer.forward_calls == 0 and trainer.actual_windows == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_rejected_before_backward(tmp_path, bad):
    trainer = observation(tmp_path, contract())
    trainer.fixture_loss = torch.tensor(bad)
    with pytest.raises(FloatingPointError, match="before backward"):
        trainer.compute_loss()


@pytest.mark.parametrize("which", ["missing", "nonfinite", "finite"])
def test_pre_optimizer_callback_checks_real_parameter_gradients(which):
    model = torch.nn.Linear(2, 1)
    trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(device=torch.device("cpu")))
    callback = entry.FiniteUpdates(trainer)
    if which != "missing":
        model(torch.ones(1, 2)).sum().backward()
    if which == "nonfinite":
        model.weight.grad[0, 0] = float("nan")
    if which == "finite":
        callback.on_pre_optimizer_step(None, None, None)
    else:
        with pytest.raises(FloatingPointError, match="before optimizer"):
            callback.on_pre_optimizer_step(None, None, None)


def test_other_rank_nonfinite_gradient_cannot_admit_a_local_finite_update(monkeypatch):
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(device=torch.device("cpu")))
    callback = entry.FiniteUpdates(trainer)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = []

    def reduce(flag, op):
        calls.append(op)
        flag.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(FloatingPointError, match="before optimizer"):
        callback.on_pre_optimizer_step(None, None, None)
    assert calls == [torch.distributed.ReduceOp.MIN]


def test_other_rank_nonfinite_loss_stops_all_ranks_before_backward(tmp_path, monkeypatch):
    trainer = observation(tmp_path, contract())
    assert torch.isfinite(trainer.fixture_loss)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = []

    def reduce(flag, op):
        calls.append(op)
        flag.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(FloatingPointError, match="before backward"):
        trainer.compute_loss()
    assert calls == [torch.distributed.ReduceOp.MIN]


def test_terminal_parameter_nonfinite_guard_detects_real_corruption():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(device=torch.device("cpu")))
    with pytest.raises(FloatingPointError, match="after optimizer"):
        entry.FiniteUpdates(trainer).on_step_end(None, None, None)


def test_receipt_write_preserves_previous_attempt_and_binds_exact_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    entry.write_json(path, {"first": True})
    identity = entry.file_identity(path)
    with pytest.raises(FileExistsError, match="Preserve previous attempt"):
        entry.write_json(path, {"replacement": True})
    assert entry.file_identity(path) == identity


def test_uninitialized_secondary_sft_rank_does_not_write_terminal_artifacts(tmp_path, monkeypatch):
    # SFT has no Trainer, so torchrun rank metadata must remain authoritative
    # even before any distributed process group has been initialized.
    monkeypatch.setattr(entry.os, "environ", {**os.environ, "RANK": "1", "WORLD_SIZE": "4"})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    output = tmp_path / "secondary"
    result = entry.run_arm(contract(), "sft", output, tokenizer=None,
                           initializer=torch.nn.Linear(2, 1), device=torch.device("cpu"))
    assert result is None and not output.exists()
