"""Native CPU TRL tests of the opt-in quality trainer and its checkpoint mixin.

No downloads, GPU allocation or external data are used. Missing local Torch/TRL
skips this module; the declared training environment must execute it before use.
"""
# ruff: noqa: E402
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from datasets import Dataset

from clasp.baselines.external_baseline_dev import seal_reward_environment
from clasp.baselines.quality_signal_dev import REWARD_HEADS
from clasp.rewards.feasible_quality import make_advantage_contract, verified_reward_metadata
from clasp.trainers import feasible_quality_grpo as fq
from clasp.trainers.budget_matched_grpo import budget_matched_trainer_class
from clasp.trainers.external_baseline_grpo import ExternalBaselineGRPOTrainer
from test_external_baseline_trainer import row, tiny_trainer
from test_quality_signal_trainer import contract as reward_contract


def training_row(index=0):
    return {**row(index), "answer": "Set: [0, 2, 4], Objective: 999"}


def new_trainer(path, method="cardinal", *, anchor=False):
    seal_reward_environment()
    rows = Dataset.from_list([training_row(i) for i in range(18)])
    carrier = tiny_trainer(path, estimator="rloo", generations=9, batch=9,
                           reward_contract=reward_contract(),
                           reward_funcs=[REWARD_HEADS["mis", "reference_sigmoid_v2"]], dataset=rows)
    tokenizer = carrier.processing_class
    tokenizer.add_tokens(["Set", "Objective", ":", "[", "]", ",", "invalid"])
    carrier.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    carrier.args.max_completion_length = carrier.max_completion_length = 32
    if anchor:
        return carrier
    advantage = make_advantage_contract(method)
    contract = {"protocol": "feasible-quality-grpo-development-20260919-v1",
                "method": method, "task": "mis", "seed": 43, "horizon": 2,
                "source": {"root": "/tmp/fixture-source", "manifest_sha256": "a" * 64},
                "advantage_contract": advantage}
    wrapped = budget_matched_trainer_class(fq.FeasibleQualityGRPOTrainer, external_baseline=True)
    trainer = wrapped(model=carrier.model, args=carrier.args, processing_class=tokenizer,
                      reward_funcs=carrier.reward_funcs, train_dataset=rows,
                      advantage_estimator="rloo", baseline_scale=1.0,
                      reward_contract=reward_contract(), advantage_contract=advantage,
                      budget_resume_contract=contract)
    trainer.model.train()
    return trainer


RESPONSES = ["invalid", "Set : [ 0 ]", "Set : [ 0 , 2 ]", "Set : [ 0 , 2 , 4 ]",
             "Set : [ 0 , 2 ]", "Set : [ 0 , 2 , 4 ]", "invalid", "Set : [ 0 ]", "Set : [ 0 , 2 ]"]


def fixed_generation(trainer, monkeypatch):
    tokens = [trainer.processing_class.encode(text, add_special_tokens=False)
              + [trainer.processing_class.eos_token_id] for text in RESPONSES]

    def generate(prompts, images):
        assert len(prompts) == 9 and trainer._baseline_generation_active
        return [[4, 5]] * 9, tokens, torch.tensor(sum(map(len, tokens))), None, {}

    monkeypatch.setattr(trainer, "_generate", generate)
    trainer.current_gradient_accumulation_steps = 1


def test_control_preserves_native_advantages_loss_and_parameter_gradients(tmp_path, monkeypatch):
    anchor = new_trainer(tmp_path / "anchor", anchor=True)
    control = new_trainer(tmp_path / "control", "control")
    control.model.load_state_dict(anchor.model.state_dict())
    for trainer in (anchor, control):
        fixed_generation(trainer, monkeypatch)
    rows = [training_row()] * 9
    a, c = [trainer._generate_and_score_completions(rows) for trainer in (anchor, control)]
    assert torch.equal(a["advantages"], c["advantages"])
    assert torch.count_nonzero(a["advantages"]) > 0
    assert anchor.external_baseline.state_dict() == control.external_baseline.state_dict()
    losses = []
    for trainer, batch in ((anchor, a), (control, c)):
        loss = trainer.compute_loss(trainer.model, batch)
        loss.backward()
        losses.append(loss.detach())
    assert torch.equal(*losses)
    gradients = 0
    for p, q in zip(anchor.model.parameters(), control.model.parameters()):
        if p.grad is not None:
            assert q.grad is not None and torch.equal(p.grad, q.grad)
            gradients += int(torch.count_nonzero(p.grad))
    assert gradients > 0
    record = json.loads((tmp_path / "control/feasible_quality_batches.jsonl").read_text())
    assert record["quality"] == [0.0] * 9
    assert record["base"] == record["combined"]
    assert [row["objective"] for row in record["metadata"]] == [None, 1, 2, 3, 2, 3, None, 1, 2]
    assert {row["reference"] for row in record["metadata"]} == {3.0}


@pytest.mark.parametrize("method", ["cardinal", "scalar_reweight", "midrank"])
def test_active_ablations_add_only_declared_credit_and_keep_anchor_tracker(tmp_path, monkeypatch, method):
    trainer = new_trainer(tmp_path, method)
    fixed_generation(trainer, monkeypatch)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    record = json.loads((tmp_path / "feasible_quality_batches.jsonl").read_text())
    expected = torch.tensor(record["base"]) + .25 * torch.tensor(record["quality"])
    assert torch.equal(batch["advantages"], expected)
    assert any(value != 0 for value in record["quality"])
    assert trainer.external_baseline.revision == 1 and trainer.external_baseline.seen_samples == 9
    assert not trainer._quality_generation_active and trainer._quality_metadata_buffer is None
    assert list(trainer._logs["advantages"])[-9:] == record["combined"]
    if method != "scalar_reweight":
        assert record["quality"][0] == record["quality"][6] == 0.0
    else:
        assert record["quality"][0] != 0.0
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)


def collection_fixture(monkeypatch, *, change=None, rank=0):
    trainer = object.__new__(fq.FeasibleQualityGRPOTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=2, process_index=rank)
    trainer.num_generations = 9
    trainer.reward_contract = reward_contract()
    trainer._quality_generation_active = True
    trainer._quality_metadata_buffer = trainer._quality_reward_buffer = None
    batches = [[training_row(i)] * 9 for i in range(2)]
    metadata = [verified_reward_metadata("mis", ["prompt"] * 9, RESPONSES, rows) for rows in batches]
    trainer._quality_prompt_rows = [("mis", 8, row["prompt_key"]) for part in metadata for row in part]
    expected = torch.tensor([row["anchor_reward"] for part in metadata for row in part]).reshape(-1, 1)
    called = []

    def inherited(*args):
        called.append(True)
        return expected

    monkeypatch.setattr(ExternalBaselineGRPOTrainer, "_calculate_rewards", inherited)

    def gather(local):
        parts = [{"rank": i, "rows": copy.deepcopy(metadata[i]), "error": None} for i in range(2)]
        parts[rank] = local
        if change is not None:
            change(parts)
        return parts

    trainer._gather_objects = gather
    return trainer, batches[rank], metadata, expected, called


@pytest.mark.parametrize("rank", [0, 1])
def test_global_metadata_retains_rank_then_candidate_order(monkeypatch, rank):
    trainer, rows, metadata, expected, called = collection_fixture(monkeypatch, rank=rank)
    result = trainer._calculate_rewards(rows, ["prompt"] * 9, RESPONSES, [[7, 2]] * 9)
    assert torch.equal(result, expected) and called == [True]
    observed = trainer._quality_metadata_buffer
    assert [row["completion_sha256"] for row in observed] == [row["completion_sha256"] for part in metadata for row in part]
    assert [row["prompt_key"] for row in observed] == [row["prompt_key"] for part in metadata for row in part]


@pytest.mark.parametrize("change,pattern", [
    (lambda parts: parts.reverse(), "envelope"),
    (lambda parts: parts[1]["rows"].pop(), "sizes differ"),
    (lambda parts: parts[1].update(error="ValueError: remote invalid reference"), "across ranks"),
    (lambda parts: parts[1]["rows"][0].update(prompt_key="wrong"), "prompt order"),
    (lambda parts: parts[1].pop("error"), "envelope"),
])
def test_bad_global_metadata_fails_before_native_reward_collective(monkeypatch, change, pattern):
    trainer, rows, _, _, called = collection_fixture(monkeypatch, change=change)
    with pytest.raises(RuntimeError, match=pattern):
        trainer._calculate_rewards(rows, ["prompt"] * 9, RESPONSES, [[7, 2]] * 9)
    assert called == [] and trainer._quality_metadata_buffer is None


def test_local_verifier_failure_still_enters_metadata_collective(monkeypatch):
    trainer, rows, _, _, called = collection_fixture(monkeypatch)
    rows = copy.deepcopy(rows)
    rows[0]["answer"] = "invalid reference"
    with pytest.raises(RuntimeError, match="across ranks"):
        trainer._calculate_rewards(rows, ["prompt"] * 9, RESPONSES, [[7, 2]] * 9)
    assert called == []


def test_same_prompt_siblings_cannot_mix_training_references(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path)
    fixed_generation(trainer, monkeypatch)
    rows = [training_row() for _ in range(9)]
    rows[1]["answer"] = "Set: [0, 2], Objective: 999"
    with pytest.raises(RuntimeError, match="reference"):
        trainer._generate_and_score_completions(rows)
    assert trainer.external_baseline.revision == 0
    assert not trainer._quality_generation_active


def test_candidate_reordering_with_different_rewards_is_rejected(monkeypatch):
    def change(parts):
        parts[1]["rows"][0], parts[1]["rows"][1] = parts[1]["rows"][1], parts[1]["rows"][0]

    trainer, rows, _, _, called = collection_fixture(monkeypatch, change=change)
    with pytest.raises(RuntimeError, match="anchor rewards"):
        trainer._calculate_rewards(rows, ["prompt"] * 9, RESPONSES, [[7, 2]] * 9)
    assert called == [True] and trainer._quality_metadata_buffer is None


@pytest.mark.parametrize("which", ["advantage", "reward"])
def test_active_contract_mutation_fails_before_generation(tmp_path, monkeypatch, which):
    trainer = new_trainer(tmp_path)
    if which == "advantage":
        trainer.advantage_contract = make_advantage_contract("midrank")
    else:
        trainer.reward_contract["source_manifest_sha256"] = "b" * 64
    invoked = []
    monkeypatch.setattr(trainer, "_generate", lambda *args: invoked.append(True))
    with pytest.raises((ValueError, RuntimeError), match="contract"):
        trainer._generate_and_score_completions([training_row()] * 9)
    assert invoked == [] and trainer.external_baseline.revision == 0


def checkpoint_contracts(trainer, path):
    path.mkdir()
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 1, "max_steps": 2}))
    (path / "budget_resume_contract.json").write_text(json.dumps(trainer._budget_payload(1)))
    (path / fq.CONTRACT_FILENAME).write_text(json.dumps(trainer._quality_checkpoint_payload(1)))


@pytest.mark.parametrize("change", ["missing", "method", "weight", "source", "horizon", "reward", "world", "step"])
def test_resume_rejects_changed_identity_before_optimizer_load(tmp_path, monkeypatch, change):
    trainer = new_trainer(tmp_path / "new")
    checkpoint = tmp_path / "checkpoint"
    checkpoint_contracts(trainer, checkpoint)
    target = checkpoint / fq.CONTRACT_FILENAME
    payload = json.loads(target.read_text())
    if change == "missing":
        target.unlink()
    elif change in ("source", "horizon"):
        target = checkpoint / "budget_resume_contract.json"
        payload = json.loads(target.read_text())
        if change == "source":
            payload["contract"]["source"]["manifest_sha256"] = "b" * 64
        else:
            payload["contract"]["horizon"] = 3
        target.write_text(json.dumps(payload))
    else:
        if change == "method":
            payload["contract"] = make_advantage_contract("midrank")
        elif change == "weight":
            payload["contract"]["weight"] = .5
        elif change == "reward":
            payload["reward_contract"]["source_manifest_sha256"] = "b" * 64
        elif change == "world":
            payload["world_size"] = 4
        elif change == "step":
            payload["global_step"] = 2
        target.write_text(json.dumps(payload))
    invoked = []
    monkeypatch.setattr(ExternalBaselineGRPOTrainer, "_load_optimizer_and_scheduler",
                        lambda *args: invoked.append(True))
    with pytest.raises(RuntimeError, match="resume.*validation"):
        trainer._load_optimizer_and_scheduler(str(checkpoint))
    assert invoked == []


def test_real_optimizer_checkpoint_roundtrip_retains_both_contracts_and_baseline(tmp_path, monkeypatch):
    trainer = new_trainer(tmp_path / "original")
    fixed_generation(trainer, monkeypatch)
    trainer.create_optimizer_and_scheduler(num_training_steps=2)
    batch = trainer._generate_and_score_completions([training_row()] * 9)
    trainer.compute_loss(trainer.model, batch).backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    trainer.state.global_step = 1
    trainer.state.max_steps = 2
    trainer._save_checkpoint(trainer.model, trial=None)
    checkpoint = tmp_path / "original/checkpoint-1"
    restored = new_trainer(tmp_path / "restored")
    restored.create_optimizer_and_scheduler(num_training_steps=2)
    restored._load_optimizer_and_scheduler(str(checkpoint))
    assert restored.external_baseline.state_dict() == trainer.external_baseline.state_dict()
    assert restored._quality_contract_json == trainer._quality_contract_json
    assert restored.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()
    assert (tmp_path / "restored/budget_resume_1_rank0.json").is_file()
    assert (tmp_path / "restored/external_resume_1_rank0.json").is_file()
    assert json.loads((checkpoint / fq.CONTRACT_FILENAME).read_text()) == trainer._quality_checkpoint_payload(1)
