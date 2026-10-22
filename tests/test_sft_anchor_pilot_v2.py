"""Actual PEFT/GC regression and durable v2 operational-proof checks."""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
from datasets import Dataset

from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from clasp.baselines.sft_anchor_pilot import evaluate_arm, make_contract
from clasp.entrypoints import train_sft_anchor_pilot as v1
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.rewards.feasible_quality import make_advantage_contract


@pytest.fixture
def production_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    for key, value in reward_environment("rloo_quality").items():
        monkeypatch.setenv(key, value)
    base, sft = tmp_path / "base", tmp_path / "sft"
    policy, tokenizer, train, validation = v2.tiny_peft_initializer("op", base, sft, torch.device("cpu"))
    del policy
    contract = make_contract("op", {"id": "test-only-tiny-PEFT-SFT",
                                    "sha256": v2.file_identity(sft / "adapter_model.safetensors")["sha256"]},
                             train, validation, scope="system_smoke", horizon=4, max_completion_tokens=1, train_seed=61)
    return base, sft, tokenizer, contract


def trainer_fixture(production_fixture, path, *, repaired, beta=.05):
    base, sft, tokenizer, contract = production_fixture
    arm = "rloo_cardinal_beta005" if beta else "rloo_cardinal_beta0"
    actor = v2.loaded_model(base, sft, torch.device("cpu"), trainable=True, dtype=torch.float32)
    reference = (v2.loaded_model(base, sft, torch.device("cpu"), trainable=False, dtype=torch.float32)
                 if beta else None)
    args = v2._training_args(contract, arm, path, torch.device("cpu"), production_peft=True)
    rows = [{**row, "answer": row["output"], "instance_json": json.dumps(row["instance_json"])}
            for row in contract["training_rows"]]
    reward = REWARD_HEADS["op", "reference_sigmoid_v2"]
    trainer_class = v2.ObservedRLOOV2 if repaired else v1.ObservedRLOO
    trainer = trainer_class(
        model=actor, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        reward_funcs=[reward], reference_model=reference,
        reference_identity={"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"],
                            "immutable": True} if beta else None,
        recipe_contract={"fixture": True}, pilot_contract=contract,
        advantage_contract=make_advantage_contract("cardinal"),
        reward_contract={"profile_id": PROFILE_ID, "task": "op", "mode": "reference_sigmoid_v2",
                         "quality_slope": 8.0, "reward_head": reward.__name__, "source_manifest_sha256": "a" * 64},
    )
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = 1
    by_id = {row["instance_id"]: row for row in rows}
    inputs = [by_id[identifier] for identifier in contract["training"]["prompt_draws"][0] for _ in range(9)]
    return trainer, inputs


def checkpoint_modes(model):
    base = model.get_base_model()
    return [module._gradient_checkpointing_func.keywords.get("use_reentrant")
            for module in base.modules() if getattr(module, "gradient_checkpointing", False)]


def test_unrepaired_v1_actual_peft_generation_loses_kwargs_and_first_backward_fails(production_fixture, tmp_path):
    trainer, inputs = trainer_fixture(production_fixture, tmp_path / "v1", repaired=False, beta=0.0)
    assert not trainer.model.get_input_embeddings().weight.requires_grad
    # Match Trainer's declared setup immediately before its first rollout.
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert checkpoint_modes(trainer.model) and all(mode is False for mode in checkpoint_modes(trainer.model))
    batch = trainer._generate_and_score_completions(inputs)
    assert all(mode is True for mode in checkpoint_modes(trainer.model))
    with pytest.warns(UserWarning, match="None of the inputs have requires_grad"):
        loss = trainer.compute_loss(trainer.model, batch)
    assert not loss.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


@pytest.mark.parametrize("beta", [0.0, .05])
def test_repaired_actual_peft_generation_backpropagates_and_updates_with_frozen_embeddings(production_fixture, tmp_path, beta):
    trainer, inputs = trainer_fixture(production_fixture, tmp_path / "v2", repaired=True, beta=beta)
    embedding = trainer.model.get_input_embeddings().weight.detach().clone()
    assert not trainer.model.get_input_embeddings().weight.requires_grad
    assert all(mode is False for mode in checkpoint_modes(trainer.model))
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
              if parameter.requires_grad}
    ref_sha = trainer.reference_state_sha256()
    batch = trainer._generate_and_score_completions(inputs)
    assert all(mode is False for mode in checkpoint_modes(trainer.model))
    loss = trainer.compute_loss(trainer.model, batch)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    trainer.create_optimizer_and_scheduler(num_training_steps=4)
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    assert any(not torch.equal(parameter.detach(), before[name]) for name, parameter in trainer.model.named_parameters()
               if parameter.requires_grad)
    assert torch.equal(embedding, trainer.model.get_input_embeddings().weight)
    assert trainer.checkpointing_stage_counts == {"construction": 1, "after_generation": 1, "before_actor_loss": 1}
    trainer._assert_reference_integrity(full=True)
    assert trainer.reference_state_sha256() == ref_sha
    if beta:
        assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())


def test_loss_restores_nonreentrant_mode_after_an_external_reentrant_toggle(production_fixture, tmp_path):
    trainer, inputs = trainer_fixture(production_fixture, tmp_path / "restoration", repaired=True)
    batch = trainer._generate_and_score_completions(inputs)
    trainer.model.gradient_checkpointing_enable()
    assert all(mode is True for mode in checkpoint_modes(trainer.model))
    loss = trainer.compute_loss(trainer.model, batch)
    assert loss.requires_grad and all(mode is False for mode in checkpoint_modes(trainer.model))
    loss.backward()


def test_effective_checkpoint_contract_mutation_is_collectively_rejected_before_actor_forward(production_fixture, tmp_path, monkeypatch):
    trainer, inputs = trainer_fixture(production_fixture, tmp_path / "contract", repaired=True)
    batch = trainer._generate_and_score_completions(inputs)
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    forwarded = []
    monkeypatch.setattr(trainer.model, "forward", lambda *args, **kwargs: forwarded.append(True))
    with pytest.raises(RuntimeError, match="effective checkpointing before_actor_loss.*across ranks"):
        trainer.compute_loss(trainer.model, batch)
    assert forwarded == []


def test_durable_proofs_exist_before_evaluation_and_raw_groups_replay_exactly(production_fixture, tmp_path, monkeypatch):
    base, sft, tokenizer, contract = production_fixture
    output = tmp_path / "actual"
    actual_collect = v2.collect_outputs
    observed = []

    def admitted_collect(*args, **kwargs):
        training = json.loads((output / "training_complete.json").read_text())
        reload = json.loads((output / "checkpoint_reload_exact.json").read_text())
        assert training["checkpoint"]["state_sha256"] == reload["reloaded_actor_state_sha256"]
        assert reload["bitwise_state_equal"] and training["reference"]["full_integrity_verified"]
        assert training["training"]["optimizer_steps"] == 4 and training["training"]["responses"] == 144
        assert training["distributed"]["rank_closure"][0]["effective_checkpointing"]["effective_use_reentrant"] is False
        observed.append(True)
        return actual_collect(*args, **kwargs)

    monkeypatch.setattr(v2, "collect_outputs", admitted_collect)
    result = v2.run_arm(contract, "rloo_cardinal_beta005", output, tokenizer, None, torch.device("cpu"),
                        base=base, sft=sft, production_fixture=True)
    assert observed == [True]
    groups = [json.loads(line) for line in (output / "evaluation_groups.jsonl").read_text().splitlines()]
    raw = [row for group in groups for row in group["raw_completions"]]
    assert len(groups) == 8 and len(raw) == 36 and result["resources"]["evaluation_groups_closed"]
    assert evaluate_arm(contract, "rloo_cardinal_beta005", raw, result["checkpoint"], result["resources"]) == result
    assert result["resources"]["training_complete_sha256"] == v2.file_identity(output / "training_complete.json")["sha256"]
    assert result["resources"]["checkpoint_reload_exact_sha256"] == v2.file_identity(output / "checkpoint_reload_exact.json")["sha256"]


def test_completed_training_and_reload_proofs_survive_evaluation_failure(production_fixture, tmp_path, monkeypatch):
    base, sft, tokenizer, contract = production_fixture
    output = tmp_path / "interrupted"
    monkeypatch.setattr(v2, "collect_outputs", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated evaluation failure")))
    with pytest.raises(RuntimeError, match="simulated evaluation failure"):
        v2.run_arm(contract, "rloo_cardinal_beta005", output, tokenizer, None, torch.device("cpu"),
                   base=base, sft=sft, production_fixture=True)
    assert json.loads((output / "training_complete.json").read_text())["training"]["optimizer_steps"] == 4
    assert json.loads((output / "checkpoint_reload_exact.json").read_text())["bitwise_state_equal"]
    assert not (output / "result.json").exists()


def test_atomic_receipt_preserves_existing_file_and_never_exposes_invalid_json(tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        v2.write_json_atomic(path, {"bad": float("nan")})
    assert not path.exists()
    v2.write_json_atomic(path, {"valid": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        v2.write_json_atomic(path, {"replacement": True})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".receipt.json.*"))


def test_new_cli_production_fixture_uses_actual_peft_checkpointed_branch(tmp_path):
    output = tmp_path / "cli"
    process = subprocess.run([sys.executable, "-m", "clasp.entrypoints.train_sft_anchor_pilot_v2",
                              "--production-fixture", "--task", "op", "--arm", "rloo_cardinal_beta005",
                              "--device", "cpu", "--output", str(output)], capture_output=True, text=True,
                             env={**{key: value for key, value in os.environ.items() if key != "LOCAL_RANK"},
                                  "OMP_NUM_THREADS": "1", "WORLD_SIZE": "1", "RANK": "0"},
                             timeout=90)
    assert process.returncode == 0, process.stderr[-5000:]
    complete = json.loads((output / "complete.json").read_text())
    assert complete["status"] == "completed_system_smoke" and complete["production_fixture"]
    assert complete["actual_RL_optimizer_steps"] == 4 and complete["actual_RL_responses"] == 144
    assert not complete["superiority_established"]
    training = json.loads((output / "op/rloo_cardinal_beta005/training_complete.json").read_text())
    closure = training["distributed"]["rank_closure"][0]
    assert closure["effective_checkpointing"]["frozen_embedding_weights"]
    assert closure["effective_checkpointing"]["effective_use_reentrant"] is False
    assert closure["checkpointing_stage_counts"]["after_generation"] == 4
    assert closure["checkpointing_stage_counts"]["before_actor_loss"] == 144
