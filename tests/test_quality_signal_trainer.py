"""Real CPU TRL/Qwen tests of reward dispatch and checkpoint-contract admission."""
# ruff: noqa: E402 -- optional runtime dependencies are skipped before imports.
from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from datasets import Dataset

from clasp.baselines.external_baseline_dev import seal_reward_environment
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from test_external_baseline_trainer import deterministic_generation, row, tiny_trainer


def contract(mode="reference_sigmoid_v2"):
    return {"profile_id": PROFILE_ID, "task": "mis", "mode": mode, "quality_slope": 8.,
            "source_manifest_sha256": "a" * 64,
            "reward_head": REWARD_HEADS["mis", mode].__name__}


def new_trainer(tmp_path, mode="reference_sigmoid_v2"):
    records = [{**row(i), "answer": "Set: [0, 2], Objective: 999"} for i in range(16)]
    seal_reward_environment()
    return tiny_trainer(tmp_path, generations=2, reward_contract=contract(mode),
                        reward_funcs=[REWARD_HEADS["mis", mode]], dataset=Dataset.from_list(records))


def test_real_trainer_runs_versioned_reward_and_finite_gradient(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path)
    deterministic_generation(trainer, monkeypatch)
    trainer.current_gradient_accumulation_steps = 1
    item = {**row(0), "answer": "Set: [0, 2], Objective: 999"}
    batch = trainer._generate_and_score_completions([item, item])
    # The deterministic token strings are malformed sets; the verified training
    # answer is admitted and both raw actions keep the invalid reward -1.
    assert batch["advantages"].tolist() == [-1., -1.]
    assert trainer.reward_funcs == [REWARD_HEADS["mis", "reference_sigmoid_v2"]]
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert trainer._baseline_checkpoint_payload()["reward_contract"] == contract()


def test_real_save_and_optimizer_resume_preserve_reward_and_tracker(tmp_path):
    trainer = new_trainer(tmp_path / "original")
    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    # Take an actual optimizer step and advance the baseline at the same boundary.
    loss = sum(p.square().sum() for p in trainer.model.parameters() if p.requires_grad)
    loss.backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    snapshot = trainer.external_baseline.snapshot([("mis", 8), ("mis", 8)])
    trainer.external_baseline.update(snapshot, [-1., 1.5])
    trainer.state.global_step = 1
    trainer._save_checkpoint(trainer.model, trial=None)
    checkpoint = tmp_path / "original/checkpoint-1"
    payload = json.loads((checkpoint / "external_baseline_state.json").read_text())
    assert payload["reward_contract"] == contract()
    assert payload["tracker"]["revision"] == payload["global_step"] == 1
    restored = new_trainer(tmp_path / "resumed")
    restored.create_optimizer_and_scheduler(num_training_steps=2)
    restored._load_optimizer_and_scheduler(str(checkpoint))
    assert restored.external_baseline.state_dict() == trainer.external_baseline.state_dict()
    assert restored.reward_contract == trainer.reward_contract
    assert (tmp_path / "resumed/external_resume_1_rank0.json").is_file()


@pytest.mark.parametrize("change", ["missing", "mode", "task", "slope", "source", "to_legacy"])
def test_real_resume_rejects_cross_recipe_before_optimizer_restore(tmp_path, change):
    trainer = new_trainer(tmp_path / "new")
    payload = trainer._baseline_checkpoint_payload()
    bad = copy.deepcopy(payload)
    if change == "missing":
        bad.pop("reward_contract")
    elif change == "mode":
        bad["reward_contract"] = contract("graded_v2")
    elif change == "task":
        bad["reward_contract"].update(task="op", reward_head=REWARD_HEADS["op", "reference_sigmoid_v2"].__name__)
    elif change == "slope":
        bad["reward_contract"]["quality_slope"] = 10.
    elif change == "source":
        bad["reward_contract"]["source_manifest_sha256"] = "b" * 64
    else:
        trainer = tiny_trainer(tmp_path / "legacy", generations=2)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "external_baseline_state.json").write_text(json.dumps(bad))
    # No optimizer files exist: reward admission must reject before reading them.
    with pytest.raises(ValueError, match="reward contract"):
        trainer._load_optimizer_and_scheduler(str(checkpoint))
