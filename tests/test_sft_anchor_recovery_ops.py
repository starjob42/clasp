"""Synthetic guards for separate v2 lineage, budget and no-retry operations.

Fixtures never train models, contact Slurm or establish a quality result.
"""
import base64
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"


def load(name):
    spec = importlib.util.spec_from_file_location("test_recovery_" + name, EXP / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("launch_study_v2")
controller = load("run_campaign_v2")
collector = load("collect_campaign_v2")
PRIOR = json.loads((EXP / "validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json").read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def manifest():
    value = json.loads((EXP / "control_v1/launch_manifest.json").read_text())
    value.update(schema_version=2, study=launcher.STUDY,
                 native_source_root=str(launcher.NATIVE_NAMESPACE / "source_v2"),
                 native_control_root=str(launcher.NATIVE_NAMESPACE / "control_v2"),
                 slots=[launcher.slot_spec(slot) for slot in launcher.SLOTS],
                 combined_maximum_requested_gpu_hours=92,
                 prior_failed_phase=launcher.prior_phase_binding("recovery/prior.json", collector.sha(encoded(PRIOR))))
    value.pop("semantic_sha256", None)
    value["semantic_sha256"] = launcher.canonical(value)
    return value


def reseal(value):
    value.pop("semantic_sha256", None)
    value["semantic_sha256"] = launcher.canonical(value)
    return value


def test_exact_prior_failed_phase_is_accepted_by_both_independent_validators():
    assert launcher.validate_prior_failure(PRIOR) == PRIOR
    assert collector.validate_prior_failure(PRIOR) == PRIOR


@pytest.mark.parametrize("field,value", [("prior_job_id", "1030930"), ("prior_requested_gpu_hours", 0),
                                          ("prior_actual_gpu_hours", 0), ("prior_source_manifest_sha256", "b" * 64),
                                          ("complete_exists", True), ("terminal_checkpoint_exists", True)])
def test_prior_identity_or_erased_cost_or_completed_claim_is_rejected(field, value):
    proof = copy.deepcopy(PRIOR)
    proof[field] = value
    for tool in (launcher, collector):
        with pytest.raises(ValueError):
            tool.validate_prior_failure(proof)


def test_rehashed_prior_attempt_cannot_point_to_another_job():
    proof = copy.deepcopy(PRIOR)
    item = proof["attempt_ledger"]
    attempt = json.loads(base64.b64decode(item["content_base64"]))
    attempt["job_id"] = "1030930"
    raw = encoded(attempt)
    item.update(content_base64=base64.b64encode(raw).decode(), sha256=collector.sha(raw), size_bytes=len(raw))
    with pytest.raises(ValueError, match="attempt ledger"):
        launcher.validate_prior_failure(proof)


@pytest.mark.parametrize("replacement", ["TIMEOUT", "RUNNING", "COMPLETED"])
def test_prior_scheduler_terminal_failure_cannot_be_rewritten(replacement):
    proof = copy.deepcopy(PRIOR)
    proof["accounting"]["stdout"] = proof["accounting"]["stdout"].replace("FAILED", replacement)
    with pytest.raises(ValueError, match="terminal failure"):
        launcher.validate_prior_failure(proof)


def test_phase2_exact_roots_and_combined_budget_and_new_entrypoint():
    value = manifest()
    assert launcher.validate_manifest(value) == value
    assert collector.validate_manifest(value) == value
    script = launcher.script_text(value, "a" * 64, launcher.ORDER[0])
    assert "clasp.entrypoints.train_sft_anchor_pilot_v2" in script
    assert "--nproc_per_node=4" in script and "#SBATCH --no-requeue" in script
    assert "control_v1" not in script and "source_v1" not in script
    sft = launcher.script_text(value, "a" * 64, "op.sft")
    assert "torch.distributed.run" not in sft and "#SBATCH --gres=gpu:1" in sft


@pytest.mark.parametrize("field,value", [("schema_version", 1), ("combined_maximum_requested_gpu_hours", 84),
        ("native_source_root", str(launcher.NATIVE_NAMESPACE / "source_v1")),
        ("native_control_root", str(launcher.NATIVE_NAMESPACE / "control_v1")),
        ("native_control_root", str(launcher.NATIVE_NAMESPACE / "source_v2/../control_v2"))])
def test_manifest_cannot_alias_v1_or_hide_reserved_failed_attempt(field, value):
    packet = manifest()
    packet[field] = value
    reseal(packet)
    for tool in (launcher, collector):
        with pytest.raises(ValueError):
            tool.validate_manifest(packet)


def test_source_manifest_requires_v1_and_v2_entrypoints_and_all_operational_tools(tmp_path):
    paths = [launcher.LAUNCHER, launcher.CONTROLLER, launcher.COLLECTOR,
             launcher.ENTRYPOINT, launcher.V1_ENTRYPOINT]
    entries = []
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("pass\n")
        entries.append({"path": name, "sha256": launcher.sha(p), "size_bytes": p.stat().st_size})
    assert launcher.verify_source(tmp_path, {"files": entries}) == set(paths)
    with pytest.raises(ValueError, match="both entrypoints"):
        launcher.verify_source(tmp_path, {"files": entries[:-1]})


@pytest.mark.parametrize("prefix", ["asft-", "asft2-"])
def test_active_job_from_either_phase_blocks_a_second_job(prefix):
    row = {"qos": "qos_gpu_a100-dev", "gpus": 4, "comment": "", "name": prefix + "op-3"}
    with pytest.raises(ValueError, match="already pending"):
        launcher.check_scheduler_capacity([row], launcher.slot_spec(launcher.ORDER[0]))


@pytest.mark.parametrize("stdout", ["8|FAILED|1:0\n", "8|TIMEOUT|0:0\n", "8|RUNNING|0:0\n",
        "8|COMPLETED|0:0\n8.batch|FAILED|1:0\n", "8|COMPLETED|0:0\n8|COMPLETED|0:0\n", ""])
def test_previous_failed_or_ambiguous_job_blocks_manual_later_slot_without_retry(stdout):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")
    with pytest.raises(ValueError):
        launcher.previous_attempts_completed([{"job_id": "8"}], run=run)
    assert len(calls) == 1 and calls[0][0] == "sacct"


def test_exact_completed_previous_accounting_is_read_only():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "8|COMPLETED|0:0\n8.batch|COMPLETED|0:0\n", "")
    launcher.previous_attempts_completed([{"job_id": "8"}], run=run)
    assert len(calls) == 1 and calls[0][0] == "sacct"


def test_failed_controller_accounting_stops_without_submission():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "8|FAILED\n8.batch|FAILED\n", "")
    with pytest.raises(ValueError, match="step failed"):
        controller.sacct_state("8", run=run)
    assert len(calls) == 1 and calls[0][0] == "sacct"


def test_authority_preserves_prior_failure_and_lifetime_on_resume(tmp_path):
    plan = {"prior_failed_phase": manifest()["prior_failed_phase"],
            "combined_maximum_requested_gpu_hours": 92, "operational_version": 2}
    authority = controller.campaign_authority(tmp_path, plan, launcher, lambda: 100)
    assert authority["prior_failed_phase"] == plan["prior_failed_phase"]
    assert authority["combined_maximum_requested_gpu_hours"] == 92
    assert controller.campaign_authority(tmp_path, plan, launcher, lambda: 999) == authority
    with pytest.raises(ValueError, match="lifetime exhausted"):
        controller.deadlines(tmp_path, authority, lambda: authority["absolute_deadline_unix"])


def test_no_execute_defaults_to_plan_and_makes_no_scheduler_calls(monkeypatch, capsys):
    monkeypatch.setattr(controller, "source_and_plan", lambda *args: ({}, ROOT, EXP, {"combined_maximum_requested_gpu_hours": 92}))
    monkeypatch.setattr(controller, "run_campaign", lambda *a, **k: pytest.fail("execution attempted"))
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: pytest.fail("scheduler called"))
    monkeypatch.setattr(controller.sys, "argv", ["run_campaign_v2.py", "--prepared-manifest", "/fixture.json",
                                               "--prepared-manifest-sha256", "a" * 64])
    controller.main()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_plan_only_no_scheduler_calls"


def test_collector_remote_reader_is_read_only_and_collects_durable_proofs():
    assert "sbatch" not in collector.REMOTE_READER and "scancel" not in collector.REMOTE_READER
    for name in ("training_complete.json", "checkpoint_reload_exact.json", "evaluation_groups.jsonl",
                 "actual_prompt_windows.jsonl", "terminal_artifact_proofs"):
        assert name in collector.REMOTE_READER


def durable_fixture():
    """Tiny synthetic evidence for proof guards; cannot pass full cohort admission."""
    slot = "op.rloo_cardinal_beta005"
    contract = {"contract_sha256": "a" * 64, "eval_seed": 9,
                "training": {"prompt_draw_sha256": "b" * 64,
                             "prompt_draws": [["fixture-train"] * 4 for _ in range(100)]},
                "validation_rows": [{"instance_id": "fixture-val", "semantic_id": "c" * 64}]}
    packet = manifest()
    entry = {"sha256": "d" * 64, "size_bytes": 99}
    source = {"path": packet["native_source_root"] + "/" + collector.ENTRYPOINT, **entry}
    entries = {collector.ENTRYPOINT: entry}
    windows = [{"step": i, "instance_ids": ["fixture-train"] * 36, "generated_tokens": 36} for i in range(100)]
    window_raw = b"".join((json.dumps(row) + "\n").encode() for row in windows)
    tree = {"files": [{"path": "adapter.safetensors", "size_bytes": 16, "sha256": "e" * 64}]}
    tree["tree_sha256"] = collector.canonical(tree["files"])
    ranks = [{"rank": rank, "status": "passed", "optimizer_steps": 100,
              "actor_state_sha256": "f" * 64, "reference_state_sha256": "1" * 64,
              "reference_full_integrity_verified": True,
              "effective_checkpointing": {"effective_use_reentrant": False, "frozen_embedding_weights": True},
              "checkpointing_stage_counts": {"construction": 1, "after_generation": 100, "before_actor_loss": 900}}
             for rank in range(4)]
    training = {"schema_version": 2, "protocol": "sft-anchor-training-complete-v2", "status": "passed",
                "operational_version": 2, "task": "op", "arm": "rloo_cardinal_beta005", "scope": "production",
                "pilot_contract_sha256": contract["contract_sha256"], "checkpoint": {"state_sha256": "f" * 64},
                "source": source, "terminal_artifact_tree": tree,
                "training": {"optimizer_steps": 100, "responses": 3600, "rollout_windows_count": 100,
                             "generated_tokens": 3600, "wall_seconds": 50,
                             "prompt_draw_sha256": "b" * 64,
                             "actual_prompt_windows_sha256": collector.sha(window_raw)},
                "distributed": {"world_size": 4, "rank_closure": ranks},
                "reference": {"beta": .05, "state_sha256": "1" * 64, "full_integrity_verified": True,
                              "initial_equivalence": {"bitwise_equal": True}, "prepared_equivalence": {"bitwise_equal": True}}}
    training_raw = encoded(training)
    reload = {"schema_version": 2, "protocol": "sft-anchor-checkpoint-reload-v2", "status": "passed",
              "source": source, "training_complete_sha256": collector.sha(training_raw),
              "expected_actor_state_sha256": "f" * 64, "reloaded_actor_state_sha256": "f" * 64,
              "bitwise_state_equal": True, "distributed_group_closed_before_evaluation": True,
              "checkpoint_path": packet["native_control_root"] + f"/outputs/{slot}/op/rloo_cardinal_beta005/terminal"}
    groups, raw = [], []
    row = contract["validation_rows"][0]
    for mode, count in (("greedy", 1), ("k8", 8)):
        seed = int.from_bytes(hashlib.sha256(f'9/{row["semantic_id"]}/{mode}'.encode()).digest()[:4], "big")
        completions = [{"instance_id": row["instance_id"], "decode": mode, "sample_index": i,
                        "completion_id": f'{row["instance_id"]}/{mode}/{i}', "generated_tokens": 1,
                        "text": "synthetic fixture only", "completion_token_ids": [1],
                        "runtime_complete": True, "generation_seed": seed} for i in range(count)]
        groups.append({"schema_version": 2, "instance_id": row["instance_id"], "semantic_id": row["semantic_id"],
                       "decode": mode, "seed": seed, "generated_tokens": count, "wall_seconds": .1,
                       "raw_completions": completions})
        raw.extend(completions)
    group_raw = b"".join((json.dumps(row) + "\n").encode() for row in groups)
    reload_raw = encoded(reload)
    resources = {"operational_version": 2, "production_fixture": False, "evaluation_groups_closed": True,
                 "training_generated_tokens": 3600, "training_wall_seconds": 50,
                 "training_complete_sha256": collector.sha(training_raw),
                 "checkpoint_reload_exact_sha256": collector.sha(reload_raw),
                 "evaluation_groups_sha256": collector.sha(group_raw),
                 "distributed_group_closed_before_evaluation": True}
    result = {"resources": resources, "checkpoint": training["checkpoint"], "raw_completions": raw}
    decoded = {}
    for filename, payload in (("training_complete.json", training_raw), ("checkpoint_reload_exact.json", reload_raw),
                              ("evaluation_groups.jsonl", group_raw), ("actual_prompt_windows.jsonl", window_raw)):
        native_name = "training/" + filename if filename == "actual_prompt_windows.jsonl" else filename
        decoded[f"results/{slot}/{filename}"] = ({"sha256": collector.sha(payload), "native_path":
                packet["native_control_root"] + f"/outputs/{slot}/op/rloo_cardinal_beta005/" + native_name}, payload)
    return slot, result, contract, packet, entries, decoded, tree


def test_exact_synthetic_durable_proof_checks_without_quality_or_publication_claim():
    collector.validate_durable_proofs(*durable_fixture())


@pytest.mark.parametrize("mutation", ["fixture", "unclosed", "rank_missing", "actor_changed", "reentrant",
    "generation_count", "loss_count", "reference_changed", "reload_changed", "group_not_destroyed",
    "missing_group", "group_seed", "runtime_incomplete", "training_window_changed", "wrong_source"])
def test_rehashed_semantically_wrong_durable_proofs_are_rejected(mutation):
    slot, result, contract, packet, entries, decoded, tree = durable_fixture()
    prefix = f"results/{slot}/"
    training = json.loads(decoded[prefix + "training_complete.json"][1])
    reload = json.loads(decoded[prefix + "checkpoint_reload_exact.json"][1])
    groups = [json.loads(line) for line in decoded[prefix + "evaluation_groups.jsonl"][1].splitlines()]
    if mutation == "fixture": result["resources"]["production_fixture"] = True
    elif mutation == "unclosed": result["resources"]["evaluation_groups_closed"] = False
    elif mutation == "rank_missing": training["distributed"]["rank_closure"].pop()
    elif mutation == "actor_changed": training["distributed"]["rank_closure"][0]["actor_state_sha256"] = "2" * 64
    elif mutation == "reentrant": training["distributed"]["rank_closure"][0]["effective_checkpointing"]["effective_use_reentrant"] = True
    elif mutation == "generation_count": training["distributed"]["rank_closure"][0]["checkpointing_stage_counts"]["after_generation"] = 99
    elif mutation == "loss_count": training["distributed"]["rank_closure"][0]["checkpointing_stage_counts"]["before_actor_loss"] = 899
    elif mutation == "reference_changed": training["reference"]["prepared_equivalence"]["bitwise_equal"] = False
    elif mutation == "reload_changed": reload["reloaded_actor_state_sha256"] = "2" * 64
    elif mutation == "group_not_destroyed": reload["distributed_group_closed_before_evaluation"] = False
    elif mutation == "missing_group": groups.pop()
    elif mutation == "group_seed": groups[0]["seed"] += 1
    elif mutation == "runtime_incomplete": groups[0]["raw_completions"][0]["runtime_complete"] = False
    elif mutation == "wrong_source": training["source"]["path"] = packet["native_source_root"] + "/src/clasp/entrypoints/train_sft_anchor_pilot.py"
    elif mutation == "training_window_changed":
        record, raw = decoded[prefix + "actual_prompt_windows.jsonl"]
        windows = [json.loads(line) for line in raw.splitlines()]
        windows[0]["instance_ids"][0] = "another-case"
        raw = b"".join((json.dumps(row) + "\n").encode() for row in windows)
        record["sha256"] = collector.sha(raw)
        decoded[prefix + "actual_prompt_windows.jsonl"] = (record, raw)
        training["training"]["actual_prompt_windows_sha256"] = record["sha256"]
    for filename, raw, key in (("training_complete.json", encoded(training), "training_complete_sha256"),
            ("evaluation_groups.jsonl", b"".join((json.dumps(row) + "\n").encode() for row in groups), "evaluation_groups_sha256")):
        record = decoded[prefix + filename][0]
        record["sha256"] = collector.sha(raw)
        decoded[prefix + filename] = (record, raw)
        result["resources"][key] = record["sha256"]
    reload["training_complete_sha256"] = result["resources"]["training_complete_sha256"]
    raw = encoded(reload)
    record = decoded[prefix + "checkpoint_reload_exact.json"][0]
    record["sha256"] = collector.sha(raw)
    decoded[prefix + "checkpoint_reload_exact.json"] = (record, raw)
    result["resources"]["checkpoint_reload_exact_sha256"] = record["sha256"]
    with pytest.raises(ValueError):
        collector.validate_durable_proofs(slot, result, contract, packet, entries, decoded, tree)

