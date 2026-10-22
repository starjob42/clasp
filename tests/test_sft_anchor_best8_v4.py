"""Prospective source-only strict/best8 integration, actual tiny CPU PEFT.

Synthetic replies test gradients and control semantics; no protected outcomes,
fresh study cohorts, native CUDA claims or empirical FOARL superiority.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
from clasp.entrypoints import train_sft_anchor_best8_v4 as v4
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.rewards import best8_from9, strict_raw_training


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    keys = set(v4.reward_environment("foarl")) | set(v4.reward_environment("rloo_quality"))
    previous = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in previous.items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = value


def fixture(path, task="op"):
    base, sft = path / "base", path / "sft"
    model, tokenizer, train, validation = v2.tiny_peft_initializer(task, base, sft, torch.device("cpu"))
    del model; gc.collect()
    contract = v4.make_smoke_contract(task, {"id": "actual-tiny-PEFT-successor-regression",
        "sha256": v2.file_identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation)
    return dict(contract=contract, tokenizer=tokenizer, base=base, sft=sft)


def first_window(contract):
    by_id = {row["instance_id"]: row for row in contract["training_rows"]}
    return [{**by_id[value], "answer": by_id[value]["output"],
             "instance_json": json.dumps(by_id[value]["instance_json"])}
            for value in contract["training"]["prompt_draws"][0] for _ in range(9)]


def fixed_replies(trainer, monkeypatch, replies):
    assert len(replies) == 9
    tokens = [trainer.processing_class.encode(text, add_special_tokens=False) for text in replies] * 4
    assert all(len(row) == 1 for row in tokens)
    def generate(prompts, images):
        assert len(prompts) == 36
        return [trainer.processing_class.encode(prompt, add_special_tokens=False) for prompt in prompts], tokens, torch.tensor(36), None, {}
    monkeypatch.setattr(trainer, "_generate", generate)
    trainer.current_gradient_accumulation_steps = 1
    trainer.model.train()


def snapshot(trainer):
    trainer._assert_reference_integrity(full=True)
    return {"actor": semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            "optimizer": semantic_state_sha256(trainer.optimizer.state_dict()),
            "scheduler": semantic_state_sha256(trainer.lr_scheduler.state_dict()), "rng": v3.rng_sha256(),
            "reference": trainer.reference_state_sha256(), "rows": copy.deepcopy(trainer.recovery_rows),
            "tokens": trainer.actual_generated_tokens}


def train(trainer):
    trainer.recovery_started = v3.time.monotonic()
    trainer.train(resume_from_checkpoint=str(trainer.recovery_checkpoint) if trainer.recovery_checkpoint else None)
    return trainer


def test_template_and_smoke_contract_cannot_register_or_launch_production(tmp_path):
    template = v4.prospective_template()
    assert len(template["recipes"]) == 6 and not template["actual_fresh_cohorts_created"]
    assert not template["production_registration_ready"] and not template["production_launch_enabled"] and not template["superiority_established"]
    assert (template["horizon"], template["global_responses_per_update"], template["siblings_per_group"]) == (100, 36, 9)
    assert (template["temperature"], template["top_p"], template["top_k"]) == (.7, 1., 0)
    common = fixture(tmp_path)
    assert v4.validate_smoke_contract(common["contract"]) == common["contract"]
    modified = copy.deepcopy(common["contract"]); modified["scope"] = "production"
    with pytest.raises(ValueError, match="only explicit synthetic"):
        v4.validate_smoke_contract(modified)
    modified = copy.deepcopy(common["contract"]); modified["training"]["top_p"] = .9
    with pytest.raises(ValueError, match="contract changed"):
        v4.validate_smoke_contract(modified)


@pytest.mark.parametrize("interval,stop", [(0, None), (3, None), (2, 1), (2, 4), (True, None)])
def test_checkpoint_controls_preserve_fixed_terminal(tmp_path, interval, stop):
    common = fixture(tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        v4.make_successor_trainer(recipe="rloo_strict_best8", output=tmp_path / "out",
            checkpoint_interval=interval, stop_after=stop, **common)


@pytest.mark.parametrize("task", ["op", "cvrp"])
def test_actual_peft_best8_gradient_is_exact_declared_detached_increment(tmp_path, monkeypatch, task):
    common = fixture(tmp_path, task)
    trainer = v4.make_successor_trainer(recipe="rloo_strict_best8", output=tmp_path / "out", **common)
    good, lower = (("Route: [0, 1, 2]", "Route: [0, 1]") if task == "op" else
                   ("Routes: [[0, 1, 2, 0]]", "Routes: [[0, 1, 0], [0, 2, 0]]"))
    fixed_replies(trainer, monkeypatch, [good] + [lower] * 8)
    batch = trainer._generate_and_score_completions(first_window(common["contract"]))
    record = trainer.last_coefficient_record
    best = torch.tensor(record["best8"]); base = torch.tensor(record["base"])
    assert torch.equal(batch["advantages"], base + .25 * best)
    assert torch.count_nonzero(best) == 36 and record["best8_active_group_fraction"] == 1.
    assert all(not batch[key].requires_grad for key in ("advantages", "ref_per_token_logps"))
    assert list(trainer._logs["advantages"])[-36:] == record["combined"]
    params = tuple(p for p in trainer.model.parameters() if p.requires_grad)
    main_loss = trainer.compute_loss(trainer.model, batch)
    control_loss = trainer.compute_loss(trainer.model, {**batch, "advantages": base})
    actual = torch.autograd.grad(main_loss - control_loss, params, allow_unused=True)
    logps, _ = trainer._get_per_token_logps_and_entropies(trainer.model,
        torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1),
        torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1), 1)
    expected = torch.autograd.grad(-.25 * (logps * best.unsqueeze(1) * batch["completion_mask"]).sum() / 36,
                                   params, allow_unused=True)
    assert sum(g.abs().sum().item() for g in actual if g is not None) > 0
    for a, b in zip(actual, expected):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6)
    assert trainer.initial_reference_audit["bitwise_equal"] and trainer.prepared_reference_audit["bitwise_equal"]
    trainer._assert_reference_integrity(full=True)
    assert all(p.grad is None for p in trainer.ref_model.parameters())
    assert trainer.last_checkpointing_audit["effective_use_reentrant"] is False


@pytest.mark.parametrize("replies", [
    ["Route: [0, 1, 2]"] * 9,
    ["Route: [0, 1, 2]"] * 2 + ["Route: [0, 1]"] * 7,
    ["Route: [0, 1, 1]"] * 9,
    ["Route: [0, 1, 1]"] * 4 + ["invalid"] * 5,
])
def test_ties_and_all_invalid_have_zero_best8_credit_without_fake_winner(tmp_path, monkeypatch, replies):
    common = fixture(tmp_path)
    trainer = v4.make_successor_trainer(recipe="rloo_strict_best8", output=tmp_path / "out", **common)
    fixed_replies(trainer, monkeypatch, replies)
    batch = trainer._generate_and_score_completions(first_window(common["contract"]))
    record = trainer.last_coefficient_record
    assert record["best8"] == [0.] * 36 and record["combined"] == record["base"]
    assert record["best8_active_group_fraction"] == 0.
    if replies[0] == "Route: [0, 1, 1]":
        assert not any(row["feasible"] for row in record["strict_metadata"])
        assert all(row[0] < 0 for row in record["gathered_reward_heads"])
    loss = trainer.compute_loss(trainer.model, batch); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)


@pytest.mark.parametrize("recipe", ["foarl_released", "foarl_strict_aligned"])
def test_actual_foarl_control_retains_native_two_head_group_std_and_response_mean_loss(tmp_path, monkeypatch, recipe):
    common = fixture(tmp_path)
    trainer = v4.make_successor_trainer(recipe=recipe, output=tmp_path / "out", **common)
    fixed_replies(trainer, monkeypatch, ["Route: [0, 1, 2]"] + ["Route: [0, 1]"] * 8)
    batch = trainer._generate_and_score_completions(first_window(common["contract"]))
    record = trainer.last_coefficient_record
    heads = torch.tensor(record["gathered_reward_heads"])
    assert heads.shape == (36, 2) and trainer.loss_type == "grpo" and trainer.scale_rewards == "group"
    scalar = heads.sum(1).reshape(-1, 9)
    expected = ((scalar - scalar.mean(1, keepdim=True)) / (scalar.std(1, keepdim=True) + 1e-4)).reshape(-1)
    assert torch.equal(batch["advantages"], expected)
    assert record["combined"] == record["base"] and record["best8_eta"] == 0.
    loss = trainer.compute_loss(trainer.model, batch); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in trainer.ref_model.parameters())


def test_released_and_changed_strict_heads_keep_distinct_valid_grammar_and_false_positive_behavior(monkeypatch):
    instance = [[[0, 0], [1, 0], [0, 1]], [0, 7, 8], 5]
    row = {"task_type": "op", "instance_json": json.dumps(instance), "input": "", "answer": "Route: [0, 1, 2], Objective: 15"}
    # Released literal parser requires comma+space; changed shared strict head accepts typed literal grammar.
    kwargs = {key: [value] for key, value in row.items()}
    compact = "Route: [0,1]"
    old = [head(prompts=["x"], completions=[compact], **kwargs)[0] for head in v4.FOARL_F638_REWARD_FUNCS]
    new = [head(prompts=["x"], completions=[compact], **kwargs)[0] for head in (v4.strict_foarl_optimality, v4.strict_foarl_fraction)]
    assert old == [0., 0.] and new == [2 * 7 / 15, 1.]
    forged = "Route: [0, 1.9, 2]"
    for key, value in v4.reward_environment("rloo_quality").items():
        monkeypatch.setenv(key, value)
    strict = strict_raw_training.strict_rloo_reward_batch("op", ["x"], [forged], [{**row, "prompt": "x"}])
    assert strict["rewards"][0] < 0 and strict["metadata"][0]["quality_utility"] == 0.


@pytest.mark.parametrize("change", ["top_p", "top_k", "temperature", "eta", "beta", "token_penalty", "typical_filter", "minimum_length"])
def test_successor_recipe_drift_is_collectively_rejected_before_native_generation(tmp_path, monkeypatch, change):
    common = fixture(tmp_path)
    trainer = v4.make_successor_trainer(recipe="rloo_strict_best8", output=tmp_path / "out", **common)
    if change == "eta": trainer.successor_settings["best8_eta"] = .5
    elif change == "beta": trainer.beta = 0.
    elif change == "token_penalty": trainer.generation_config.repetition_penalty = 1.1
    elif change == "typical_filter": trainer.generation_config.typical_p = .9
    elif change == "minimum_length": trainer.generation_config.min_new_tokens = 1
    else: setattr(trainer.generation_config, change, {"top_p": .9, "top_k": 50, "temperature": 1.}[change])
    generated = []; monkeypatch.setattr(trainer, "_generate", lambda *args: generated.append(True))
    with pytest.raises(RuntimeError, match="successor active recipe"):
        trainer._generate_and_score_completions(first_window(common["contract"]))
    assert generated == []


@pytest.mark.parametrize("change", ["cross_prompt", "wrong_task", "forged_utility"])
def test_global_strict_identity_and_utility_admission_stops_generation_or_actor_loss(tmp_path, monkeypatch, change):
    common = fixture(tmp_path)
    trainer = v4.make_successor_trainer(recipe="rloo_strict_best8", output=tmp_path / "out", **common)
    inputs = first_window(common["contract"])
    if change == "cross_prompt": inputs[1]["prompt"] = "x99"
    elif change == "wrong_task": inputs[1]["task_type"] = "cvrp"
    else:
        original = strict_raw_training.strict_rloo_reward_batch
        def forged(*args, **kwargs):
            result = original(*args, **kwargs); result["metadata"][0]["quality_utility"] = .123
            return result
        monkeypatch.setattr(strict_raw_training, "strict_rloo_reward_batch", forged)
    fixed_replies(trainer, monkeypatch, ["Route: [0, 1, 2]"] * 9)
    with pytest.raises((ValueError, RuntimeError), match="same-prompt|task/row|utility"):
        trainer._generate_and_score_completions(inputs)


@pytest.mark.parametrize("recipe", ["rloo_strict_best8", "foarl_released", "rloo_strict_best8_beta0"])
def test_actual_peft_periodic_resume_is_exact_and_reference_stays_original(tmp_path, recipe):
    common = fixture(tmp_path)
    full = train(v4.make_successor_trainer(recipe=recipe, output=tmp_path / "full", **common))
    expected = snapshot(full); initial_audit = full.initial_reference_audit
    assert full.state.global_step == 4 and len(full.actual_windows) == 4 and expected["tokens"] == 144
    assert full.last_checkpointing_audit["effective_use_reentrant"] is False
    del full; gc.collect()
    partial = train(v4.make_successor_trainer(recipe=recipe, output=tmp_path / "partial", stop_after=2, **common))
    checkpoint = tmp_path / "partial/training/checkpoint-2"
    assert partial.state.global_step == 2
    del partial; gc.collect()
    resumed = train(v4.make_successor_trainer(recipe=recipe, output=tmp_path / "resumed", resume_from=checkpoint, **common))
    assert snapshot(resumed) == expected and resumed.resume_rng_audit["bitwise_equal"]
    assert resumed.initial_reference_audit == initial_audit
    payload = json.loads((tmp_path / "resumed/training/checkpoint-4/sft_anchored_rloo_contract.json").read_text())
    assert payload["contract"]["protocol"] == v4.PROTOCOL
    assert payload["contract"]["successor_recipe"] == recipe
    assert payload["contract"]["training_sampling"] == {"temperature": .7, "top_p": 1., "top_k": 0}
    assert payload["contract"]["statistical_source"] == v4.file_identity(Path(v4.__file__))


@pytest.mark.parametrize("recipe", ["rloo_strict_best8", "foarl_released", "sft"])
def test_actual_tiny_training_terminal_reload_and_fixed_three_stream_eval(tmp_path, recipe):
    common = fixture(tmp_path)
    result = v4.run_successor_arm(recipe=recipe, output=tmp_path / "out", **common)
    assert result["scope"] == "system_smoke" and not result["superiority_established"]
    assert result["trained_updates"] == (0 if recipe == "sft" else 4)
    assert result["RL_responses"] == (0 if recipe == "sft" else 144)
    assert result["evaluation_responses"] == 68 and result["checkpoint_reload_exact"]
    groups = [json.loads(line) for line in (tmp_path / "out/successor_evaluation_groups.jsonl").read_text().splitlines()]
    assert len(groups) == 12
    assert {row["stream"] for row in groups} == set(v4.DECODE_STREAMS)
    assert all(len(row["raw_completions"]) == v4.DECODE_STREAMS[row["stream"]]["count"] for row in groups)
    proof = json.loads((tmp_path / "out/successor_training_complete.json").read_text())
    assert proof["actor_changed"] == (recipe != "sft")
    assert proof["reference_full_integrity_verified"] == (recipe != "sft")
    assert proof["terminal_artifact_tree"] == v2.artifact_tree(tmp_path / "out/terminal")
    assert not proof["superiority_established"]


def test_optional_classic_control_is_named_separate_and_original_recipe_is_preserved():
    from clasp.baselines import classic_grpo_400_dev as classic
    info = v4.optional_classic_comparator_api("cvrp")
    assert info["reward_callable"] is classic.classic_cvrp_strict_objective_reward
    assert info["original_config"] == classic.CONFIG and info["reward_environment"] == classic.REWARD_ENVIRONMENT
    assert info["resource_protocol_extension_required"] and not info["appended_to_six_recipes"]
    assert not info["original_runtime_compatible"] and "classic" not in v4.RECIPES
    with pytest.raises(ValueError, match="no OP"):
        v4.optional_classic_comparator_api("op")


def test_serial_reload_evaluation_api_rejects_distributed_environment_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    loaded = []
    monkeypatch.setattr(v2, "loaded_model", lambda *args, **kwargs: loaded.append(True))
    with pytest.raises(ValueError, match="one rank only"):
        v4.run_successor_arm({}, "sft", tmp_path / "out", None, None, None)
    assert loaded == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("recipe,fault", [("rloo_strict_best8", False), ("foarl_released", False), ("rloo_strict_best8", True)])
def test_actual_cpu_gloo_two_rank_trainer_global_credit_and_peer_fault(tmp_path, recipe, fault):
    common = fixture(tmp_path)
    v4.write_json_atomic(tmp_path / "contract.json", common["contract"])
    output = tmp_path / "ddp"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(Path(__file__).parent / "fixtures/best8_successor_ddp_worker.py"), "--fixture", str(tmp_path),
        "--output", str(output), "--recipe", recipe]
    if fault: command.append("--fault")
    env = {key: value for key, value in os.environ.items() if key not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")}
    env.update(OMP_NUM_THREADS="1", PYTHONPATH=str(Path(v4.__file__).resolve().parents[2]))
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [json.loads((output / f"rank{rank}_trainer_only.json").read_text()) for rank in range(2)]
    if fault:
        assert all(row["peer_fault_collectively_rejected_before_generation"] for row in reports)
    else:
        assert all(row["steps"] == 4 and row["RL_responses"] == row["tokens"] == 144 for row in reports)
        for key in ("actor", "reference", "optimizer", "scheduler", "windows", "strict_prompt_keys", "combined"):
            assert reports[0][key] == reports[1][key]
        assert all(row["reference_frozen"] and row["effective_checkpointing"]["effective_use_reentrant"] is False for row in reports)
        keys = reports[0]["strict_prompt_keys"]
        assert len(keys) == 36 and all(len(set(keys[start:start + 9])) == 1 for start in range(0, 36, 9))
        seal = json.loads((output / "training/checkpoint-4/recovery_checkpoint_complete.json").read_text())
        assert seal["world_size"] == 2 and len(seal["rank_closure"]) == 2
        assert not (output / "successor_evaluation_groups.jsonl").exists()
