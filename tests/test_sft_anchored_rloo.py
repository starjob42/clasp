"""Native offline TRL/Qwen checks for the prospective anchored RLOO pilot.

The tiny model validates credit, KL, training and resume mechanics. It provides
no evidence about pretrained OP/CVRP quality or superiority over FOARL.
"""
# ruff: noqa: E402
from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from datasets import Dataset
from trl import GRPOTrainer

from clasp.baselines.external_baseline_dev import seal_reward_environment
from clasp.baselines.quality_signal_dev import REWARD_HEADS
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards.feasible_quality import make_advantage_contract
from clasp.trainers.sft_anchored_rloo import (
    CONTRACT_FILENAME, SFTAnchoredRLOOTrainer, SFTReferenceGRPOTrainer,
)
from test_external_baseline_trainer import row, tiny_trainer
from test_quality_signal_trainer import contract as reward_contract


REFERENCE = {"source_kind": "original_sft", "source_sha256": "a" * 64, "immutable": True}
RESPONSES = ["invalid", "Set : [ 0 ]", "Set : [ 0 , 2 ]", "Set : [ 0 , 2 , 4 ]",
             "Set : [ 0 , 2 ]", "Set : [ 0 , 2 , 4 ]", "invalid", "Set : [ 0 ]", "Set : [ 0 , 2 ]"]


def training_row(index=0):
    return {**row(index), "answer": "Set: [0, 2, 4], Objective: 999"}


def carrier(path):
    seal_reward_environment()
    dataset = Dataset.from_list([training_row(i) for i in range(18)])
    trainer = tiny_trainer(path, estimator="rloo", generations=9, batch=9,
                           reward_contract=reward_contract(),
                           reward_funcs=[REWARD_HEADS["mis", "reference_sigmoid_v2"]], dataset=dataset)
    trainer.processing_class.add_tokens(["Set", "Objective", ":", "[", "]", ",", "invalid"])
    trainer.model.resize_token_embeddings(len(trainer.processing_class), mean_resizing=False)
    trainer.args.max_completion_length = trainer.max_completion_length = 32
    return trainer


def new_trainer(path, *, beta=0.0, method="control", baseline=False, original=None):
    source = carrier(path)
    if original is not None:
        source.model.load_state_dict(original)
    source.args.beta = beta
    reference = copy.deepcopy(source.model).requires_grad_(False) if beta else None
    common = dict(model=source.model, args=source.args, reward_funcs=source.reward_funcs,
                  processing_class=source.processing_class, train_dataset=source.train_dataset,
                  reference_model=reference, reference_identity=copy.deepcopy(REFERENCE) if beta else None,
                  recipe_contract={"task": "mis", "source": "tiny fixture", "seed": 43, "horizon": 2})
    if baseline:
        source.args.loss_type = "grpo"
        source.args.scale_rewards = "group"
        trainer = SFTReferenceGRPOTrainer(**common)
    else:
        trainer = SFTAnchoredRLOOTrainer(**common, advantage_contract=make_advantage_contract(method),
                                        reward_contract=reward_contract())
    trainer.model.train()
    return trainer


def fixed_generation(trainer, monkeypatch):
    tokens = [trainer.processing_class.encode(text, add_special_tokens=False)
              + [trainer.processing_class.eos_token_id] for text in RESPONSES]

    def generate(prompts, images):
        count = len(prompts)
        assert count == 9
        return [[4, 5]] * count, tokens, torch.tensor(sum(map(len, tokens))), None, {}

    monkeypatch.setattr(trainer, "_generate", generate)
    trainer.current_gradient_accumulation_steps = 1


def test_beta_zero_control_exactly_preserves_old_rloo_credit_loss_and_gradients(tmp_path, monkeypatch):
    old = carrier(tmp_path / "old")
    new = new_trainer(tmp_path / "new", original=old.model.state_dict())
    assert new.ref_model is None and new.args.beta == 0.0
    for trainer in (old, new):
        fixed_generation(trainer, monkeypatch)
    batches = [trainer._generate_and_score_completions([training_row()] * 9) for trainer in (old, new)]
    assert torch.equal(batches[0]["advantages"], batches[1]["advantages"])
    assert not batches[1]["advantages"].requires_grad
    losses = []
    for trainer, batch in zip((old, new), batches):
        loss = trainer.compute_loss(trainer.model, batch)
        loss.backward()
        losses.append(loss.detach())
    assert torch.equal(*losses)
    gradients = 0
    for p, q in zip(old.model.parameters(), new.model.parameters()):
        if p.grad is not None:
            assert q.grad is not None and torch.equal(p.grad, q.grad)
            gradients += int(torch.count_nonzero(p.grad))
    assert gradients > 0
    record = json.loads((tmp_path / "new/sft_anchored_rloo_batches.jsonl").read_text())
    assert record["base"] == record["combined"] and record["quality"] == [0.0] * 9


@pytest.mark.parametrize("beta", [0.0, 0.05])
def test_cardinal_adds_only_declared_feasible_coefficients_and_detaches_them(tmp_path, monkeypatch, beta):
    trainer = new_trainer(tmp_path, beta=beta, method="cardinal")
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    record = json.loads((tmp_path / "sft_anchored_rloo_batches.jsonl").read_text())
    expected = torch.tensor(record["base"]) + .25 * torch.tensor(record["quality"])
    assert torch.equal(batch["advantages"], expected)
    assert record["quality"][0] == record["quality"][6] == 0.0
    assert any(v != 0.0 for v in record["quality"])
    assert record["beta"] == beta and record["quality_weight"] == .25
    assert not batch["advantages"].requires_grad
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
    if beta:
        assert trainer.initial_reference_audit["bitwise_equal"]
        assert trainer.prepared_reference_audit["bitwise_equal"]
        assert all(p.grad is None for p in trainer.ref_model.parameters())


def test_positive_kl_has_nonzero_gradient_after_actor_moves_reference_stays_fixed(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path, beta=.05)
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    batch["advantages"] = torch.zeros_like(batch["advantages"])
    before = trainer.reference_state_sha256()
    with torch.no_grad():
        weights = trainer.model.lm_head.weight
        weights.add_(.15 * torch.sin(torch.arange(weights.numel()).reshape(weights.shape)))
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert sum(p.grad.abs().sum().item() for p in trainer.model.parameters() if p.grad is not None) > 0
    assert trainer._metrics["train"]["kl"][-1] > 0
    assert all(p.grad is None for p in trainer.ref_model.parameters())
    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    trainer._assert_reference_integrity(full=True)
    assert before == trainer.reference_state_sha256()
    # Actor/reference equality is a step-zero admission condition, not a
    # condition imposed on a successfully changing actor.
    trainer._generate_and_score_completions([training_row()] * 9)


@pytest.mark.parametrize("which", ["same_object", "shared_storage", "trainable", "different_state", "attention"])
def test_invalid_reference_rejected_before_training(tmp_path, which):
    source = carrier(tmp_path)
    source.args.beta = .05
    reference = copy.deepcopy(source.model).requires_grad_(False)
    if which == "same_object":
        reference = source.model
        reference.requires_grad_(False)
    elif which == "shared_storage":
        reference.lm_head.weight.data = source.model.lm_head.weight.data
    elif which == "trainable":
        next(reference.parameters()).requires_grad_(True)
    elif which == "different_state":
        with torch.no_grad():
            next(reference.parameters()).add_(.01)
    else:
        reference.config._attn_implementation = "sdpa"
    with pytest.raises((ValueError, RuntimeError), match="reference|attention"):
        SFTAnchoredRLOOTrainer(model=source.model, args=source.args, reward_funcs=source.reward_funcs,
                              processing_class=source.processing_class, train_dataset=source.train_dataset,
                              reference_model=reference, reference_identity=REFERENCE,
                              recipe_contract={"fixture": True})


@pytest.mark.parametrize("mutation", ["weights", "raw_data", "train_mode", "grad", "optimizer"])
def test_reference_drift_or_optimizer_contamination_is_rejected(tmp_path, monkeypatch, mutation):
    trainer = new_trainer(tmp_path, beta=.05)
    parameter = next(trainer.ref_model.parameters())
    if mutation == "weights":
        with torch.no_grad():
            parameter.add_(.01)
    elif mutation == "raw_data":
        # Tensor.data can bypass version counters; full checkpoint hashing
        # must still reject this silent state mutation.
        parameter.data.add_(.01)
    elif mutation == "train_mode":
        trainer.ref_model.train()
    elif mutation == "grad":
        parameter.grad = torch.ones_like(parameter)
    else:
        trainer.create_optimizer_and_scheduler(num_training_steps=2)
        trainer.optimizer.add_param_group({"params": [parameter]})
    with pytest.raises((ValueError, RuntimeError), match="reference|Optimizer"):
        trainer._assert_reference_integrity(full=True)


@pytest.mark.parametrize("change", ["beta", "quality", "reward", "recipe", "reference"])
def test_recipe_mutation_is_rejected_before_generation(tmp_path, monkeypatch, change):
    trainer = new_trainer(tmp_path, beta=.05)
    if change == "beta":
        trainer.beta = 0.0
    elif change == "quality":
        trainer.advantage_contract = make_advantage_contract("cardinal")
    elif change == "reward":
        trainer.reward_contract["source_manifest_sha256"] = "b" * 64
    elif change == "recipe":
        trainer.recipe_contract["seed"] = 44
    else:
        trainer.reference_identity["source_sha256"] = "b" * 64
    generated = []
    monkeypatch.setattr(trainer, "_generate", lambda *a: generated.append(True))
    with pytest.raises(RuntimeError, match="contract"):
        trainer._generate_and_score_completions([training_row()] * 9)
    assert generated == []


def test_same_prompt_cannot_mix_training_references(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path)
    fixed_generation(trainer, monkeypatch)
    rows = [training_row() for _ in range(9)]
    rows[1]["answer"] = "Set: [0, 2], Objective: 999"
    with pytest.raises(RuntimeError, match="references"):
        trainer._generate_and_score_completions(rows)
    assert not trainer._generation_active


def checkpoint_after_real_update(trainer, monkeypatch):
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    trainer.compute_loss(trainer.model, batch).backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    trainer.state.global_step = 1
    trainer._save_checkpoint(trainer.model, trial=None)
    return trainer.args.output_dir + "/checkpoint-1"


def test_native_checkpoint_restore_preserves_recipe_ref_optimizer_and_allows_changed_actor(tmp_path, monkeypatch):
    original = new_trainer(tmp_path / "original", beta=.05, method="cardinal")
    sft = copy.deepcopy(original.ref_model.state_dict())
    checkpoint = checkpoint_after_real_update(original, monkeypatch)
    saved = json.loads(open(checkpoint + "/" + CONTRACT_FILENAME).read())
    assert saved["contract"]["beta"] == .05 and saved["contract"]["quality_weight"] == .25
    resumed = new_trainer(tmp_path / "resumed", beta=.05, method="cardinal", original=sft)
    resumed._load_from_checkpoint(checkpoint)
    resumed.create_optimizer_and_scheduler(num_training_steps=2)
    resumed._load_optimizer_and_scheduler(checkpoint)
    assert resumed._resuming
    assert semantic_state_sha256(original.optimizer.state_dict()) == semantic_state_sha256(resumed.optimizer.state_dict())
    assert semantic_state_sha256(original.lr_scheduler.state_dict()) == semantic_state_sha256(resumed.lr_scheduler.state_dict())
    assert resumed.reference_state_sha256() == original.reference_state_sha256()
    assert any(not torch.equal(p, q) for p, q in zip(resumed.model.parameters(), resumed.ref_model.parameters()))
    fixed_generation(resumed, monkeypatch)
    resumed.compute_loss(resumed.model, resumed._generate_and_score_completions([training_row()] * 9)).backward()
    assert all(p.grad is None for p in resumed.ref_model.parameters())


@pytest.mark.parametrize("change", ["missing", "step", "world", "beta", "reference", "credit", "source", "horizon"])
def test_resume_rejects_changed_checkpoint_contract_before_optimizer_load(tmp_path, monkeypatch, change):
    trainer = new_trainer(tmp_path, beta=.05)
    checkpoint = checkpoint_after_real_update(trainer, monkeypatch)
    path = tmp_path / "checkpoint-1" / CONTRACT_FILENAME
    payload = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "step":
            payload["global_step"] = 2
        elif change == "world":
            payload["world_size"] = 2
        elif change == "beta":
            payload["contract"]["beta"] = 0.0
        elif change == "reference":
            payload["contract"]["reference_state_sha256"] = "b" * 64
        elif change == "credit":
            payload["contract"]["advantage_contract"] = make_advantage_contract("cardinal")
        elif change == "source":
            payload["contract"]["recipe_contract"]["source"] = "different"
        else:
            payload["contract"]["max_steps"] = 3
        path.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(GRPOTrainer, "_load_optimizer_and_scheduler", lambda *a: calls.append(True))
    with pytest.raises(RuntimeError, match="resume recipe"):
        trainer._load_optimizer_and_scheduler(checkpoint)
    assert calls == []


def test_stock_foarl_plumbing_keeps_native_group_credit_and_response_mean_loss(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path, beta=.05, baseline=True)
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    rewards = torch.tensor([row["anchor_reward"] for row in __import__(
        "clasp.rewards.feasible_quality", fromlist=["verified_reward_metadata"]
    ).verified_reward_metadata("mis", ["prompt"] * 9, RESPONSES, [training_row()] * 9)])
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    assert torch.equal(batch["advantages"], expected)
    assert trainer.loss_type == "grpo" and trainer.scale_rewards == "group"
    assert trainer.beta == trainer.args.beta == .05
    native = GRPOTrainer._compute_loss(trainer, trainer.model, batch)
    actual = trainer.compute_loss(trainer.model, batch)
    assert torch.equal(native, actual)
    actual.backward()
    assert all(p.grad is None for p in trainer.ref_model.parameters())


@pytest.mark.parametrize("baseline", [False, True])
def test_peer_reference_corruption_stops_healthy_rank_before_native_actor_forward(tmp_path, monkeypatch, baseline):
    trainer = new_trainer(tmp_path, beta=.05, baseline=baseline)
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    forwarded, gathered = [], []

    def peer_failure(local_error):
        gathered.append(local_error)
        # This rank is healthy; its peer rejected a mutated reference.
        return [local_error, "RuntimeError: Original SFT reference tensors changed after initialization"]

    monkeypatch.setattr(trainer, "_gather_objects", peer_failure)
    monkeypatch.setattr(trainer.model, "forward", lambda *args, **kwargs: forwarded.append(True))
    with pytest.raises(RuntimeError, match="pre-forward.*loss admission.*across ranks"):
        trainer.compute_loss(trainer.model, batch)
    assert gathered == [None] and forwarded == []


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("fault", ["reference", "advantages", "reference_scores"])
def test_local_loss_admission_failure_still_joins_peer_collective_before_forward(tmp_path, monkeypatch, baseline, fault):
    trainer = new_trainer(tmp_path, beta=.05, baseline=baseline)
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    if fault == "reference":
        trainer.ref_model.train()
    elif fault == "advantages":
        batch["advantages"].requires_grad_(True)
    else:
        batch["ref_per_token_logps"].requires_grad_(True)
    forwarded, gathered = [], []

    def gather_failure(local_error):
        gathered.append(local_error)
        return [None, local_error]

    monkeypatch.setattr(trainer, "_gather_objects", gather_failure)
    monkeypatch.setattr(trainer.model, "forward", lambda *args, **kwargs: forwarded.append(True))
    with pytest.raises(RuntimeError, match="pre-forward.*loss admission.*across ranks"):
        trainer.compute_loss(trainer.model, batch)
    assert len(gathered) == 1 and isinstance(gathered[0], str) and forwarded == []
