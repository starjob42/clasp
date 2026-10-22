"""Actual PEFT resume/evaluation-rescue regressions; integration evidence only."""
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
from clasp.baselines.sft_anchor_pilot import make_contract
from clasp.entrypoints import train_sft_anchor_pilot_v2 as v2
from clasp.entrypoints import train_sft_anchor_recovery_v3 as v3
from clasp.models.external_baseline_state import semantic_state_sha256
from clasp.baselines.budget_matched_dev import reward_environment
from clasp.baselines.quality_signal_dev import PROFILE_ID, REWARD_HEADS
from clasp.rewards.feasible_quality import make_advantage_contract
from datasets import Dataset
from transformers import set_seed


def fixture(tmp_path, task="op"):
    base, sft = tmp_path / "base", tmp_path / "sft"
    policy, tokenizer, train, validation = v2.tiny_peft_initializer(task, base, sft, torch.device("cpu"))
    del policy; gc.collect()
    contract = make_contract(task, {"id": "actual-tiny-PEFT-resume-regression",
        "sha256": v2.file_identity(sft / "adapter_model.safetensors")["sha256"]}, train, validation,
        scope="system_smoke", horizon=4, max_completion_tokens=1, train_seed=61)
    return base, sft, tokenizer, contract


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def snapshot(trainer):
    trainer._assert_reference_integrity(full=True)
    return {"actor": semantic_state_sha256(trainer.accelerator.unwrap_model(trainer.model).state_dict()),
            "optimizer": semantic_state_sha256(trainer.optimizer.state_dict()),
            "scheduler": semantic_state_sha256(trainer.lr_scheduler.state_dict()),
            "rng": v3.rng_sha256(), "reference": trainer.reference_state_sha256(),
            "rows": copy.deepcopy(trainer.recovery_rows), "tokens": trainer.actual_generated_tokens}


def test_periodic_v3_preserves_actual_uninterrupted_v2_numerics(tmp_path):
    base, sft, tokenizer, contract = fixture(tmp_path)
    class TracedV2(v2.ObservedRLOOV2):
        def _generate_and_score_completions(self, inputs):
            before = v3.rng_sha256()
            output = super()._generate_and_score_completions(inputs)
            tokens, masks = output["completion_ids"].detach().cpu().tolist(), output["completion_mask"].detach().cpu().tolist()
            self.recovery_rows.append({"step": int(self.state.global_step),
                "instance_ids": list(self.actual_windows[-1]),
                "generated_tokens": self.actual_generated_tokens - sum(row["generated_tokens"] for row in self.recovery_rows),
                "completion_fingerprints": [v3.digest([ids, mask]) for ids, mask in zip(tokens, masks)],
                "rank_rng_before_generation": [before]})
            return output
    set_seed(contract["train_seed"])
    os.environ.update(reward_environment("rloo_quality"))
    actor = v2.loaded_model(base, sft, torch.device("cpu"), trainable=True, dtype=torch.float32)
    ref = v2.loaded_model(base, sft, torch.device("cpu"), trainable=False, dtype=torch.float32)
    args = v2._training_args(contract, "rloo_cardinal_beta005", tmp_path / "v2", torch.device("cpu"), production_peft=True)
    by_id = {row["instance_id"]: row for row in contract["training_rows"]}
    rows = [{**by_id[value], "answer": by_id[value]["output"], "instance_json": json.dumps(by_id[value]["instance_json"])}
            for draw in contract["training"]["prompt_draws"] for value in draw]
    reward = REWARD_HEADS["op", "reference_sigmoid_v2"]
    native = TracedV2(model=actor, args=args, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        reward_funcs=[reward], reference_model=ref,
        reference_identity={"source_kind": "original_sft", "source_sha256": contract["initializer"]["sha256"], "immutable": True, "scope": contract["scope"]},
        recipe_contract={"pilot_sha256": contract["contract_sha256"], "arm": "rloo_cardinal_beta005", "operational_version": 2,
                         "entrypoint_sha256": v2.file_identity(Path(v2.__file__))["sha256"]}, pilot_contract=contract,
        advantage_contract=make_advantage_contract("cardinal"),
        reward_contract={"profile_id": PROFILE_ID, "task": "op", "mode": "reference_sigmoid_v2", "quality_slope": 8.,
                         "reward_head": reward.__name__, "source_manifest_sha256": v2.file_identity(Path(v2.__file__))["sha256"]})
    native.recovery_rows = []
    native.train(); expected = snapshot(native)
    assert (tmp_path / "v2/training/checkpoint-4").is_dir()
    assert not (tmp_path / "v2/training/checkpoint-2").exists()
    del native, actor, ref; gc.collect()
    periodic, _ = v3.run_training_segment(contract, "rloo_cardinal_beta005", tmp_path / "v3", tokenizer,
        base, sft, torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    assert snapshot(periodic) == expected
    assert (tmp_path / "v3/training/checkpoint-2").is_dir()


@pytest.mark.parametrize("task,arm", [("op", "rloo_cardinal_beta005"), ("op", "rloo_scalar_beta0"),
                                     ("cvrp", "rloo_cardinal_beta005"), ("cvrp", "foarl")])
def test_actual_peft_native_resume_is_bitwise_equal_to_uninterrupted_cpu(tmp_path, task, arm):
    base, sft, tokenizer, contract = fixture(tmp_path, task)
    common = dict(contract=contract, arm=arm, tokenizer=tokenizer, base=base, sft=sft,
                  device=torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    full, _ = v3.run_training_segment(output=tmp_path / "full", **common)
    expected = snapshot(full); del full; gc.collect()
    partial, report = v3.run_training_segment(output=tmp_path / "partial", stop_after=2, **common)
    first = snapshot(partial)
    assert partial.state.global_step == 2 and not (tmp_path / "partial/training_complete.json").exists()
    assert not (tmp_path / "partial/terminal").exists()
    del partial; gc.collect()
    resumed, final = v3.run_training_segment(output=tmp_path / "resumed", resume_from=report["checkpoint"], **common)
    actual = snapshot(resumed)
    assert actual == expected
    assert resumed.resume_rng_audit["bitwise_equal"]
    assert resumed.state.global_step == final["fixed_terminal_selection"] == 4
    assert actual["tokens"] == 144 and actual["reference"] == first["reference"]
    assert resumed.last_checkpointing_audit["effective_use_reentrant"] is False
    assert resumed.last_checkpointing_audit["frozen_embedding_weights"]
    if resumed.beta:
        assert any(not torch.equal(a, b) for a, b in zip(resumed.model.parameters(), resumed.ref_model.parameters()))
    proof = json.loads((tmp_path / "resumed/training_complete.json").read_text())
    assert proof["checkpoint"]["state_sha256"] == expected["actor"]
    assert proof["training"]["responses"] == 144 and proof["training"]["rollout_windows_count"] == 4
    assert len(proof["lineage"]) == 1
    assert proof["accounting"]["segment_start_step"] == 2
    assert proof["accounting"]["accepted_training_seconds"] > proof["accounting"]["segment_training_seconds"]


def test_corrupt_rank_optimizer_or_incomplete_seal_is_rejected(tmp_path):
    base, sft, tokenizer, contract = fixture(tmp_path)
    common = dict(contract=contract, arm="rloo_cardinal_beta005", tokenizer=tokenizer, base=base, sft=sft,
                  device=torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    trainer, report = v3.run_training_segment(output=tmp_path / "partial", stop_after=2, **common)
    del trainer; gc.collect()
    directory = Path(report["checkpoint"])
    state = directory / "optimizer_rank0.pt"
    original = state.read_bytes(); state.write_bytes(original + b"corrupt")
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        v3.make_recovery_trainer(output=tmp_path / "corrupt", resume_from=directory, **common)
    state.write_bytes(original)
    (directory / v3.CHECKPOINT_SEAL).unlink()
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        v3.make_recovery_trainer(output=tmp_path / "incomplete", resume_from=directory, **common)


def test_native_optimizer_pickle_safety_guard_cannot_be_bypassed(tmp_path, monkeypatch):
    base, sft, tokenizer, contract = fixture(tmp_path)
    common = dict(contract=contract, arm="rloo_cardinal_beta005", tokenizer=tokenizer, base=base, sft=sft,
                  device=torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    partial, report = v3.run_training_segment(output=tmp_path / "partial", stop_after=2, **common)
    del partial; gc.collect()
    resumed = v3.make_recovery_trainer(output=tmp_path / "guard", resume_from=report["checkpoint"], **common)
    resumed.create_optimizer_and_scheduler(num_training_steps=4)
    import transformers.trainer
    restored = []
    def unsafe():
        raise ValueError("simulated native Torch<2.6 pickle safety guard")
    monkeypatch.setattr(transformers.trainer, "check_torch_load_is_safe", unsafe)
    monkeypatch.setattr(v3, "restore_rank_state", lambda *args, **kwargs: restored.append(True))
    with pytest.raises(RuntimeError, match="Torch<2.6 pickle safety guard"):
        resumed._load_optimizer_and_scheduler(report["checkpoint"])
    assert restored == []


def test_resume_discards_uncommitted_tail_without_double_counting(tmp_path):
    base, sft, tokenizer, contract = fixture(tmp_path)
    common = dict(contract=contract, arm="rloo_cardinal_beta005", tokenizer=tokenizer, base=base, sft=sft,
                  device=torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    trainer, report = v3.run_training_segment(output=tmp_path / "partial", stop_after=2, **common)
    prefix = copy.deepcopy(trainer.recovery_rows); del trainer; gc.collect()
    trailing = copy.deepcopy(prefix[-1]); trailing["step"] = 2
    trailing["instance_ids"] = [identifier for identifier in contract["training"]["prompt_draws"][2] for _ in range(9)]
    observations = tmp_path / "partial/training/recovery_rollout_observations.jsonl"
    v2._append_durable(observations, trailing)
    with observations.open("ab") as stream:
        stream.write(b'{"step":3')
    resumed, final = v3.run_training_segment(output=tmp_path / "resumed", resume_from=report["checkpoint"], **common)
    assert resumed.actual_generated_tokens == 144
    assert len(resumed.recovery_rows) == 4 and resumed.recovery_rows[:2] == prefix
    assert final["discarded_work"]["responses"] == 36
    assert final["discarded_work"]["generated_tokens"] == 36
    assert final["discarded_work"]["truncated_progress_tail"]["size_bytes"] == 9


def test_eval_only_v2_rescue_reuses_closed_groups_and_matches_full_seeded_decode(tmp_path, monkeypatch):
    base, sft, tokenizer, contract = fixture(tmp_path)
    original = v2.collect_outputs
    def interrupted(model, *args, **kwargs):
        generate = model.generate; calls = 0
        def fail(*a, **k):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated walltime during third modegroup")
            return generate(*a, **k)
        model.generate = fail
        return original(model, *args, **kwargs)
    monkeypatch.setattr(v2, "collect_outputs", interrupted)
    training = tmp_path / "v2_partial_eval"
    with pytest.raises(RuntimeError, match="simulated walltime"):
        v2.run_arm(contract, "rloo_cardinal_beta005", training, tokenizer, None, torch.device("cpu"),
                   base=base, sft=sft, production_fixture=True)
    prior = training / "evaluation_groups.jsonl"
    old = [json.loads(line) for line in prior.read_text().splitlines()]
    assert len(old) == 2 and (training / "training_complete.json").exists()
    prior_hash = v2.file_identity(prior)["sha256"]
    result, recovery = v3.rescue_evaluation(contract, "rloo_cardinal_beta005", training, tmp_path / "rescue", tokenizer,
        base, torch.device("cpu"), production_fixture=True,
        allocation_records=[{"id": "tiny-failed", "gpus": 0, "requested_wall_seconds": 0, "elapsed_seconds": 1},
                            {"id": "tiny-rescue", "gpus": 0, "requested_wall_seconds": 0, "elapsed_seconds": 1}])
    groups = [json.loads(line) for line in (tmp_path / "rescue/evaluation_groups.jsonl").read_text().splitlines()]
    assert groups[:2] == old and v2.file_identity(prior)["sha256"] == prior_hash
    assert len(recovery["reused_modegroups"]) == 2 and len(recovery["regenerated_modegroups"]) == 6
    assert recovery["reused_evaluation_responses"] == 9 and recovery["new_evaluation_responses"] == 27
    assert recovery["allocation_accounting"]["allocation_accounting_complete"] is False
    assert recovery["allocation_accounting"]["accounting_certificate_verified"] is False
    reloaded = v2.loaded_model(base, training / "terminal", torch.device("cpu"), trainable=False, dtype=torch.float32)
    full, _ = original(reloaded, tokenizer, contract, torch.device("cpu"), progress_path=tmp_path / "full_groups.jsonl")
    assert [row for group in groups for row in group["raw_completions"]] == full
    assert result["resources"]["evaluation_responses"] == 36
    assert not result["superiority_established"]


@pytest.mark.parametrize("field", ["completion_token_ids", "text", "generation_seed", "completion_id"])
def test_eval_group_corruption_is_rejected(field, tmp_path):
    base, sft, tokenizer, contract = fixture(tmp_path)
    model = v2.loaded_model(base, sft, torch.device("cpu"), trainable=False, dtype=torch.float32)
    group = v3._generate_group(model, tokenizer, contract, contract["validation_rows"][0], "greedy", torch.device("cpu"))
    v3._validate_group(group, contract, tokenizer)
    row = group["raw_completions"][0]
    if field == "completion_token_ids":
        row[field] = [999999]
    elif field == "generation_seed":
        row[field] += 1
    else:
        row[field] += "corrupt"
    with pytest.raises(ValueError, match="text/tokens/identity"):
        v3._validate_group(group, contract, tokenizer)


def test_allocations_cannot_exceed_current_authority_or_hide_duplicate_attempts():
    rows = [{"id": "prior-failed", "gpus": 4, "requested_wall_seconds": 7200, "elapsed_seconds": 459},
            {"id": "rescue", "gpus": 1, "requested_wall_seconds": 7200, "elapsed_seconds": 10}]
    accounting = v3.allocation_accounting(rows)
    assert accounting["total_requested_gpu_hours"] == 10
    assert accounting["total_allocated_gpu_seconds"] == 1846
    assert accounting["new_job_authorized"] is False
    assert accounting["allocation_accounting_complete"] is False
    with pytest.raises(ValueError, match="duplicate|Invalid allocation"):
        v3.allocation_accounting(rows + [rows[0]])
    with pytest.raises(ValueError, match="exceeds"):
        v3.allocation_accounting(rows, authorized_requested_gpu_hours=9)


@pytest.fixture(scope="module")
def sealed_v3(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual_sealed_v3")
    base, sft, tokenizer, contract = fixture(directory)
    trainer, report = v3.run_training_segment(contract, "rloo_cardinal_beta005", directory / "training_run", tokenizer,
        base, sft, torch.device("cpu"), checkpoint_interval=2, production_fixture=True)
    del trainer; gc.collect()
    return directory, base, tokenizer, contract, report


@pytest.mark.parametrize("field", ["rank_actor", "rank_reference", "reference_integrity", "prepared_equivalence", "checkpointing", "windows"])
def test_original_producer_proof_admission_rejects_missing_or_changed_closure(sealed_v3, tmp_path, field):
    source, base, tokenizer, contract, report = sealed_v3
    original = source / "training_run"
    # Copy model/metadata artifacts solely to simulate a damaged recovery source.
    import shutil
    copied = tmp_path / "damaged"
    shutil.copytree(original, copied)
    path = copied / "training_complete.json"
    proof = json.loads(path.read_text())
    if field == "rank_actor":
        proof["distributed"]["rank_closure"][0]["actor_state_sha256"] = "f" * 64
    elif field == "rank_reference":
        proof["distributed"]["rank_closure"][0]["reference_state_sha256"] = "f" * 64
    elif field == "reference_integrity":
        proof["reference"]["full_integrity_verified"] = False
    elif field == "prepared_equivalence":
        proof["reference"]["prepared_equivalence"]["bitwise_equal"] = False
    elif field == "checkpointing":
        proof["distributed"]["rank_closure"][0]["effective_checkpointing"]["effective_use_reentrant"] = True
    else:
        with (copied / "training/actual_prompt_windows.jsonl").open("a") as stream:
            stream.write('{}\n')
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        v3.admit_original_training_proof(contract, "rloo_cardinal_beta005", copied)
    assert not (tmp_path / "rescue").exists()


def test_intermediate_checkpoint_selection_is_forbidden_when_terminal_is_sealed(sealed_v3):
    source, _, _, contract, _ = sealed_v3
    with pytest.raises(ValueError, match="newest sealed"):
        v3.read_sealed_checkpoint(source / "training_run/training/checkpoint-2", contract, "rloo_cardinal_beta005")


def test_v3_eval_rescue_handles_truncated_tail_and_duplicate_group_rejection(sealed_v3, tmp_path):
    source, base, tokenizer, contract, _ = sealed_v3
    training = source / "training_run"
    model = v2.loaded_model(base, training / "terminal", torch.device("cpu"), trainable=False, dtype=torch.float32)
    group = v3._generate_group(model, tokenizer, contract, contract["validation_rows"][0], "greedy", torch.device("cpu"))
    del model; gc.collect()
    partial = tmp_path / "partial.jsonl"
    v2._append_durable(partial, group)
    with partial.open("ab") as stream:
        stream.write(b'{"schema_version":2')
    result, recovery = v3.rescue_evaluation(contract, "rloo_cardinal_beta005", training, tmp_path / "rescued", tokenizer,
        base, torch.device("cpu"), production_fixture=True, progress_paths=[partial],
        expected_training_complete_sha256=v2.file_identity(training / "training_complete.json")["sha256"])
    assert recovery["reused_evaluation_responses"] == 1 and recovery["new_evaluation_responses"] == 35
    assert len(recovery["discarded_partial_tails"]) == 1
    assert recovery["wasted_inflight_generation_seconds"] is None
    duplicate = tmp_path / "duplicate.jsonl"
    v2._append_durable(duplicate, group); v2._append_durable(duplicate, group)
    with pytest.raises(ValueError, match="Duplicate/conflicting"):
        v3.rescue_evaluation(contract, "rloo_cardinal_beta005", training, tmp_path / "duplicate_rescue", tokenizer,
            base, torch.device("cpu"), production_fixture=True, progress_paths=[duplicate])
    assert result["resources"]["training_optimizer_steps"] == 4


def test_rescue_requires_bound_training_bytes_and_original_terminal_tree(sealed_v3, tmp_path):
    source, base, tokenizer, contract, _ = sealed_v3
    training = source / "training_run"
    with pytest.raises(ValueError, match="byte binding"):
        v3.rescue_evaluation(contract, "rloo_cardinal_beta005", training, tmp_path / "bad_binding", tokenizer,
            base, torch.device("cpu"), production_fixture=True, expected_training_complete_sha256="f" * 64)
    import shutil
    damaged = tmp_path / "damaged"
    shutil.copytree(training, damaged)
    with (damaged / "terminal/adapter_model.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="terminal model/config bytes"):
        v3.rescue_evaluation(contract, "rloo_cardinal_beta005", damaged, tmp_path / "bad_model", tokenizer,
            base, torch.device("cpu"), production_fixture=True)


def test_rescue_integrates_externally_pinned_accounting_certificate_without_claiming_final_cost(sealed_v3, tmp_path, monkeypatch):
    """Fabricated scheduler fixture tests wiring only; those jobs never existed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("certificate_fixtures", Path(__file__).with_name("test_sft_anchor_recovery_accounting.py"))
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    source, base, tokenizer, contract, _ = sealed_v3
    training = source / "training_run"
    proof = json.loads((training / "training_complete.json").read_text())
    certificate = fixtures.fixture()
    certificate["training_proof_binding"].update(
        training_complete_sha256=v2.file_identity(training / "training_complete.json")["sha256"],
        pilot_contract_sha256=contract["contract_sha256"], source_sha256=proof["source"]["sha256"])
    path = tmp_path / "fabricated_unit_certificate.json"
    path.write_bytes(fixtures.encoded(certificate))
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    result, recovery = v3.rescue_evaluation(contract, "rloo_cardinal_beta005", training, tmp_path / "rescue", tokenizer,
        base, torch.device("cpu"), production_fixture=True, accounting_certificate=path,
        accounting_certificate_sha256=v2.file_identity(path)["sha256"])
    accounting = recovery["allocation_accounting"]
    assert accounting["accounting_certificate_verified"] is True
    assert accounting["total_requested_gpu_hours"] == 92
    assert accounting["allocation_accounting_complete"] is False
    assert accounting["allocated_cost_is_lower_bound_running_snapshot"] is True
    assert accounting["new_job_authorized"] is False
    assert result["scope"] == "system_smoke" and not result["superiority_established"]


@pytest.mark.parametrize("task,arm", [("op", "rloo_cardinal_beta005"), ("cvrp", "foarl")])
def test_actual_two_rank_gloo_peft_resume_and_collective_peer_fault(tmp_path, task, arm):
    helper = Path(__file__).with_name("helpers") / "sft_anchor_recovery_v3_worker.py"
    env = {key: value for key, value in os.environ.items() if key not in
           {"LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE"}}
    env.update(PYTHONPATH="src", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             str(helper), "--output", str(tmp_path / "ddp"), "--task", task, "--arm", arm],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=120)
    assert result.returncode == 0, result.stdout[-12000:]
    receipts = [json.loads((tmp_path / f"ddp/rank{rank}_trajectory_receipt.json").read_text()) for rank in range(2)]
    assert all(row["whole_actor_optimizer_scheduler_rng_prefix_equal"] and
               row["peer_fault_rejected_collectively"] and row["resumed"]["step"] == 4 for row in receipts)
    assert receipts[0]["resumed"]["actor"] == receipts[1]["resumed"]["actor"]
