"""Offline synthetic guards for immutable phase-two recovery collection.

The frozen 128/96 contracts and historical failure receipt are real inputs.
Every completion, checkpoint, rank, source stub and scheduler response below
is manufactured test evidence. These fixtures never represent model outcomes,
contact a scheduler, train a model or publish paper results.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test_sft_anchor_collection import FROZEN_RAW, encoded, record, replace_json, scorer
from test_sft_anchor_recovery_ops import collector, launcher, manifest as recovery_manifest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
PRIOR_PATH = "validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json"


def make_contract(task):
    """Load the exact frozen production input, with no invented cohort rows."""
    return scorer.validate_contract(json.loads((EXP / f"prepared_v1/contracts/{task}_seed43.json").read_bytes()))


def jsonl(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def _synthetic_groups(contract):
    groups = []
    for row in contract["validation_rows"]:
        for mode, count in (("greedy", 1), ("k8", 8)):
            seed = int.from_bytes(hashlib.sha256(
                f'{contract["eval_seed"]}/{row["semantic_id"]}/{mode}'.encode()).digest()[:4], "big")
            completions = [{
                "instance_id": row["instance_id"], "decode": mode, "sample_index": index,
                "completion_id": f'{row["instance_id"]}/{mode}/{index}',
                "text": row["output"], "generated_tokens": 1, "completion_token_ids": [1],
                "runtime_complete": True, "generation_seed": seed, "synthetic_fixture": True,
                "semantic_id": row["semantic_id"], "reference_id": row["reference_id"],
                "reference_answer": row["reference_answer"],
                "score": scorer.score_completion(contract["task"], row["output"],
                                                   row["instance_json"], row["reference_answer"]),
            } for index in range(count)]
            groups.append({"schema_version": 2, "instance_id": row["instance_id"],
                           "semantic_id": row["semantic_id"], "decode": mode, "seed": seed,
                           "generated_tokens": count, "wall_seconds": .1,
                           "raw_completions": completions})
    return groups


def _durable_slot(slot, contract, manifest, entry):
    """Manufacture the exact durable schema, including all four RL ranks."""
    task, arm = slot.split(".")
    steps, world = (0, 1) if arm == "sft" else (100, 4)
    actor_sha = collector.sha(("synthetic actor " + slot).encode())
    reference_sha = collector.sha(("synthetic frozen SFT " + task).encode())
    checkpoint = {"id": "synthetic_" + slot, "state_sha256": actor_sha,
                  "initializer_sha256": contract["initializer_sha256"],
                  "reference_initializer_sha256": contract["initializer_sha256"],
                  "train_seed": contract["train_seed"], "step": steps}
    source = {"path": manifest["native_source_root"] + "/" + collector.ENTRYPOINT,
              "sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
    windows = [{"step": index, "instance_ids": [identifier for identifier in draw for _ in range(9)],
                "generated_tokens": 36}
               for index, draw in enumerate(contract["training"]["prompt_draws"][:steps])]
    window_raw = jsonl(windows)
    tree = {"files": [{"path": "synthetic_adapter.safetensors", "size_bytes": 17,
                        "sha256": collector.sha(("synthetic tree " + slot).encode())}]}
    tree["tree_sha256"] = collector.canonical(tree["files"])
    ranks = [{"rank": rank, "status": "passed", "optimizer_steps": steps,
              "actor_state_sha256": actor_sha, "reference_state_sha256": reference_sha,
              "reference_full_integrity_verified": True,
              "effective_checkpointing": {"effective_use_reentrant": False,
                                           "frozen_embedding_weights": True},
              "checkpointing_stage_counts": {"construction": 1, "after_generation": steps,
                                              "before_actor_loss": steps * 9}}
             for rank in range(world)]
    beta = .05 if arm.endswith("beta005") or arm == "foarl" else 0.0
    training = {"schema_version": 2, "protocol": "sft-anchor-training-complete-v2",
                "status": "passed", "operational_version": 2, "synthetic_fixture": True,
                "task": task, "arm": arm, "scope": "production",
                "pilot_contract_sha256": contract["contract_sha256"], "checkpoint": checkpoint,
                "source": source, "terminal_artifact_tree": tree,
                "training": {"optimizer_steps": steps, "responses": steps * 36,
                             "rollout_windows_count": steps, "generated_tokens": steps * 36,
                             "wall_seconds": steps * .1,
                             "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
                             "actual_prompt_windows_sha256": collector.sha(window_raw)},
                "distributed": {"world_size": world, "rank_closure": ranks},
                "reference": {"beta": beta, "state_sha256": reference_sha,
                              "full_integrity_verified": True,
                              "initial_equivalence": {"bitwise_equal": True},
                              "prepared_equivalence": {"bitwise_equal": True}}}
    training_raw = encoded(training)
    reload = {"schema_version": 2, "protocol": "sft-anchor-checkpoint-reload-v2", "status": "passed",
              "synthetic_fixture": True, "source": source,
              "training_complete_sha256": collector.sha(training_raw), "bitwise_state_equal": True,
              "expected_actor_state_sha256": actor_sha, "reloaded_actor_state_sha256": actor_sha,
              "distributed_group_closed_before_evaluation": bool(steps),
              "checkpoint_path": manifest["native_control_root"] + f"/outputs/{slot}/{task}/{arm}/terminal"}
    reload_raw = encoded(reload)
    groups = _synthetic_groups(contract)
    group_raw = jsonl(groups)
    resources = {"training_optimizer_steps": steps, "training_responses": steps * 36,
                 "training_generated_tokens": steps * 36, "training_wall_seconds": steps * .1,
                 "evaluation_wall_seconds": .5, "checkpoint_reload_exact": True,
                 "prompt_draw_sha256": contract["training"]["prompt_draw_sha256"],
                 "operational_version": 2, "production_fixture": False, "synthetic_fixture": True,
                 "evaluation_groups_closed": True,
                 "distributed_group_closed_before_evaluation": bool(steps),
                 "training_complete_sha256": collector.sha(training_raw),
                 "checkpoint_reload_exact_sha256": collector.sha(reload_raw),
                 "evaluation_groups_sha256": collector.sha(group_raw)}
    result = scorer.evaluate_arm(contract, arm,
                                 [item for group in groups for item in group["raw_completions"]],
                                 checkpoint, resources)
    complete = {"status": "completed_single_arm_model_evaluation", "operational_version": 2,
                "synthetic_fixture": True, "superiority_established": False,
                "checkpoint_reload_exact": True, "actual_RL_optimizer_steps": steps,
                "actual_RL_responses": steps * 36, "actual_evaluation_responses": 864,
                "device": "cuda:0", "source": source}
    payloads = {"complete.json": encoded(complete), "result.json": encoded(result),
                "training_complete.json": training_raw, "checkpoint_reload_exact.json": reload_raw,
                "evaluation_groups.jsonl": group_raw}
    if steps:
        payloads["actual_prompt_windows.jsonl"] = window_raw
    return payloads, tree


def fixture_bundle(completed_count=0, attempts_count=1, campaign_report=False):
    """Return (manifest, bytes, SHA, bundle) of wholly synthetic v2 outcomes.

    This helper is shared with offline paper-gate tests only. Its launch SHA is
    synthetic and must never be admitted through the live publication allowlist.
    """
    assert 0 <= completed_count <= attempts_count <= 12
    assert not campaign_report or completed_count == attempts_count == 12
    manifest = recovery_manifest()
    source, control = manifest["native_source_root"], manifest["native_control_root"]
    contract_raws = {task: (ROOT / binding["path"]).read_bytes()
                     for task, binding in manifest["contracts"].items()}
    contracts = {task: scorer.validate_contract(json.loads(raw)) for task, raw in contract_raws.items()}
    prior_raw = (EXP / PRIOR_PATH).read_bytes()
    prior_relative = "recovery/synthetic_fixture_prior_failed_phase.json"
    manifest["prior_failed_phase"] = collector.prior_phase_binding(prior_relative, collector.sha(prior_raw))
    files = {collector.SCORER: FROZEN_RAW, prior_relative: prior_raw}
    for path in (collector.ENTRYPOINT, collector.CONTROLLER, launcher.LAUNCHER,
                 launcher.COLLECTOR, launcher.V1_ENTRYPOINT):
        files[path] = ("# SYNTHETIC TEST SOURCE STUB ONLY: " + path + "\n").encode()
    files.update({manifest["contracts"][task]["path"]: raw for task, raw in contract_raws.items()})
    entries = [{"path": path, "sha256": collector.sha(raw), "size_bytes": len(raw)}
               for path, raw in files.items()]
    source_raw = encoded({"synthetic_fixture": True, "files": entries})
    manifest["source_manifest"] = {"path": "source_manifest.json", "sha256": collector.sha(source_raw)}
    manifest.pop("semantic_sha256", None)
    manifest["semantic_sha256"] = collector.canonical(manifest)
    manifest_raw = encoded(manifest)
    manifest_sha = collector.sha(manifest_raw)
    collector.validate_manifest(manifest)
    records = [record(control + "/launch_manifest.json", "status/launch_manifest.json", manifest_raw,
                      "bound_launch_manifest"),
               record(source + "/source_manifest.json", "source/source_manifest.json", source_raw,
                      "bound_source_manifest"),
               record(source + "/" + collector.SCORER, "source/frozen_scorer.py", FROZEN_RAW,
                      "bound_frozen_scorer"),
               record(source + "/" + prior_relative, "source/prior_failed_phase.json", prior_raw,
                      "bound_prior_failed_phase")]
    for task, raw in contract_raws.items():
        records.append(record(source + "/" + manifest["contracts"][task]["path"],
                              f"source/contracts/{task}.json", raw, "bound_task_contract"))
    attempts = {}
    for index, slot in enumerate(collector.ORDER[:attempts_count]):
        attempts[slot] = {"slot": slot, "study": collector.STUDY, "manifest_sha256": manifest_sha,
                          "requested_gpu_hours": 2 if slot.endswith(".sft") else 8,
                          "state": "submitted", "job_id": str(123 + index), "synthetic_fixture": True}
        records.append(record(control + f"/attempts/{slot}.json", f"attempts/{slot}.json",
                              encoded(attempts[slot]), "submission_attempt_snapshot"))
    status = {"state": "waiting_for_exact_job", "slot": collector.ORDER[0], "scheduler_state": "PENDING",
              "synthetic_fixture": True}
    records.append(record(control + "/campaign_status.json", "status/campaign_status.json", encoded(status)))
    owner = {"pid": 88, "proc_start_ticks": "1234", "proc_command": ["python", "run_campaign_v2.py"]}
    records.append(record(control + "/campaign_owners/88-1234.json", "status/campaign_owners/88-1234.json",
                          encoded(owner), "owner_identity_snapshot"))
    records.append(record(control + "/logs/op.rloo_cardinal_beta005-123.out",
                          "logs/op.rloo_cardinal_beta005-123.out.tail", b"synthetic pending log\n",
                          "current_log_snapshot", tail=True))
    job_ids = ",".join(item["job_id"] for item in attempts.values())
    accounting = "".join(f"{item['job_id']}|{'COMPLETED' if i < completed_count else 'PENDING'}|0:0\n"
                         for i, item in enumerate(attempts.values()))
    queries = [{"argv": ["sacct", "--noheader", "--parsable2", "--jobs", job_ids,
                          "--format=JobIDRaw,State%40,ExitCode"], "returncode": 0, "stdout": accounting, "stderr": ""},
               {"argv": ["squeue", "--noheader", "--jobs", job_ids, "--format=%i|%T|%R"],
                "returncode": 0, "stdout": "", "stderr": ""}]
    raw_bindings, terminal_trees = [], {}
    entry = next(item for item in entries if item["path"] == collector.ENTRYPOINT)
    for slot in collector.ORDER[:completed_count]:
        task, arm = slot.split(".")
        payloads, terminal_trees[slot] = _durable_slot(slot, contracts[task], manifest, entry)
        binding = {"slot": slot, "job_id": attempts[slot]["job_id"]}
        for filename, raw in payloads.items():
            native = (f"outputs/{slot}/complete.json" if filename == "complete.json" else
                      f"outputs/{slot}/{task}/{arm}/" +
                      ("training/" if filename == "actual_prompt_windows.jsonl" else "") + filename)
            purpose = ("completed_job_receipt" if filename == "complete.json" else
                       "completed_raw_result" if filename == "result.json" else "completed_durable_proof")
            records.append(record(control + "/" + native, f"results/{slot}/{filename}", raw, purpose))
            if filename in ("complete.json", "result.json"):
                binding[filename.removesuffix(".json")] = {
                    "path": control + "/" + native, "sha256": collector.sha(raw), "size_bytes": len(raw)}
        raw_bindings.append(binding)
    if campaign_report:
        authority = {"synthetic_fixture": True,
                     "plan": {"prepared_manifest_sha256": manifest_sha,
                              "source_manifest_sha256": manifest["source_manifest"]["sha256"],
                              "controller_sha256": next(item["sha256"] for item in entries
                                                        if item["path"] == collector.CONTROLLER),
                              "order": list(collector.ORDER), "operational_version": 2,
                              "prior_failed_phase": manifest["prior_failed_phase"],
                              "combined_maximum_requested_gpu_hours": 92},
                     "prior_failed_phase": manifest["prior_failed_phase"],
                     "combined_maximum_requested_gpu_hours": 92}
        authority["semantic_sha256"] = collector.canonical(authority)
        report = {"status": "completed_fixed_order_development_campaign", "scope": "production_development_pilot",
                  "synthetic_fixture": True, "superiority_established": False,
                  "no_heldout_or_five_seed_superiority_claim": True, "fixed_order": list(collector.ORDER),
                  "actual_complete_slots": 12, "requested_gpu_hours": 84, "operational_version": 2,
                  "combined_requested_gpu_hours": 92, "prior_actual_gpu_hours": .51,
                  "prior_failed_phase": manifest["prior_failed_phase"],
                  "authority_sha256": authority["semantic_sha256"], "raw_result_bindings": raw_bindings}
        records.append(record(control + "/campaign_authority.json", "status/campaign_authority.json", encoded(authority)))
        records.append(record(control + "/campaign_report.json", "campaign_report.json", encoded(report),
                              "completed_campaign_report"))
    bundle = {"schema_version": 2, "study": collector.STUDY, "synthetic_fixture": True,
              "collected_utc": "2026-10-14T10:00:00+00:00", "prepared_manifest_sha256": manifest_sha,
              "source_proofs": entries, "completed_slots": list(collector.ORDER[:completed_count]),
              "owner_probes": [{"pid": 88, "proc_exists": True}], "read_only_queries": queries,
              "terminal_artifact_proofs": terminal_trees, "records": records,
              "native_mutations": False, "superiority_established": False}
    return manifest, manifest_raw, manifest_sha, bundle


def test_partial_recovery_collection_is_read_only_unique_and_immutable(tmp_path):
    manifest, raw, expected, bundle = fixture_bundle()
    packet = tmp_path / "launch_manifest.json"
    packet.write_bytes(raw)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(bundle), "")
    output = tmp_path / "synthetic_snapshot_v2"
    receipt = collector.collect_campaign(packet, expected, output, run=run)
    assert receipt["schema_version"] == 2 and receipt["study"] == collector.STUDY
    assert receipt["admission"]["status"] == "status_only_campaign_incomplete"
    assert not receipt["admission"]["completed_slots"] and not (output / "results").exists()
    assert not receipt["native_mutations"] and not receipt["superiority_established"]
    assert (output / "source/prior_failed_phase.json").read_bytes() == (EXP / PRIOR_PATH).read_bytes()
    assert (output / "status/campaign_owners/88-1234.json").is_file()
    assert (output / "logs/op.rloo_cardinal_beta005-123.out.tail").read_bytes() == b"synthetic pending log\n"
    assert all(path.stat().st_mode & 0o222 == 0 for path in output.rglob("*") if path.is_file())
    assert calls[0][0][0] == "ssh" and calls[0][0][-2] == "jean-zay"
    assert collector.NATIVE_PYTHON + " -I -S -B -c " in calls[0][0][-1]
    assert "sbatch" not in calls[0][0][-1] and "scancel" not in calls[0][0][-1]
    assert manifest["native_source_root"].endswith("/source_v2")
    assert manifest["native_control_root"].endswith("/control_v2")
    with pytest.raises(ValueError, match="unique output"):
        collector.collect_campaign(packet, expected, output, run=run)
    assert len(calls) == 1


def test_exact_completed_recovery_slot_replays_full_frozen_cohort():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    admission, decoded = collector.validate_bundle(bundle, manifest, expected)
    assert admission["status"] == "validated_partial_completed_slots"
    slot = admission["completed_slots"][0]
    assert slot["actual_optimizer_steps"] == 100 and slot["actual_training_responses"] == 3600
    assert slot["actual_evaluation_responses"] == 864 and slot["raw_result_replay_exact"]
    result = json.loads(decoded[f"results/{collector.ORDER[0]}/result.json"][1])
    assert len(result["raw_completions"]) == 864
    assert all(row["synthetic_fixture"] for row in result["raw_completions"])
    assert not admission["superiority_established"] and not admission["campaign_report_admitted"]


def test_full_twelve_slot_recovery_campaign_and_durable_evidence_are_admitted():
    manifest, _, expected, bundle = fixture_bundle(12, 12, True)
    admission, decoded = collector.validate_bundle(bundle, manifest, expected)
    assert admission["status"] == "validated_completed_campaign" and admission["campaign_report_admitted"]
    assert len(admission["completed_slots"]) == 12 and admission["raw_results_replayed_exactly"]
    assert not admission["superiority_established"] and admission["benchmark_superiority_claim"] == "not_established"
    for slot in collector.ORDER:
        steps, world = (0, 1) if slot.endswith(".sft") else (100, 4)
        training = json.loads(decoded[f"results/{slot}/training_complete.json"][1])
        reload = json.loads(decoded[f"results/{slot}/checkpoint_reload_exact.json"][1])
        assert training["distributed"]["world_size"] == world
        assert training["training"]["optimizer_steps"] == steps
        assert reload["distributed_group_closed_before_evaluation"] is bool(steps)
        assert len(json.loads(decoded[f"results/{slot}/result.json"][1])["raw_completions"]) == 864
        for rank in training["distributed"]["rank_closure"]:
            assert rank["effective_checkpointing"]["effective_use_reentrant"] is False
            assert rank["checkpointing_stage_counts"]["after_generation"] == steps
            assert rank["checkpointing_stage_counts"]["before_actor_loss"] == steps * 9
    report = json.loads(decoded["campaign_report.json"][1])
    assert report["requested_gpu_hours"] + report["prior_failed_phase"]["requested_gpu_hours"] == 92
    assert report["prior_actual_gpu_hours"] == .51 and report["no_heldout_or_five_seed_superiority_claim"]


@pytest.mark.parametrize("filename", ["training_complete.json", "checkpoint_reload_exact.json",
                                      "evaluation_groups.jsonl", "actual_prompt_windows.jsonl"])
def test_completed_slot_without_durable_artifact_is_rejected(filename):
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    missing = f"results/{collector.ORDER[0]}/{filename}"
    bundle["records"] = [item for item in bundle["records"] if item["local_path"] != missing]
    with pytest.raises((ValueError, KeyError)):
        collector.validate_bundle(bundle, manifest, expected)


def test_rehashed_wrong_terminal_tree_cannot_replace_native_proof():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    tree = bundle["terminal_artifact_proofs"][collector.ORDER[0]]
    tree["files"][0]["sha256"] = "b" * 64
    tree["tree_sha256"] = collector.canonical(tree["files"])
    with pytest.raises(ValueError, match="terminal artifact tree"):
        collector.validate_bundle(bundle, manifest, expected)


def test_missing_native_tree_proof_cannot_admit_completed_slot():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    bundle["terminal_artifact_proofs"] = {}
    with pytest.raises(ValueError, match="terminal artifact tree"):
        collector.validate_bundle(bundle, manifest, expected)


def test_rehashed_forged_metrics_fail_frozen_full_cohort_replay():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    def forge(item):
        value = item["metrics"]["k8"]["feasibility_percent"]
        item["metrics"]["k8"]["feasibility_percent"] = 99 if value != 99 else 98
        item["result_sha256"] = collector.canonical({key: value for key, value in item.items()
                                                   if key != "result_sha256"})
    replace_json(bundle, f"results/{collector.ORDER[0]}/result.json", forge)
    with pytest.raises(ValueError, match="raw-result replay"):
        collector.validate_bundle(bundle, manifest, expected)


def test_rehashed_result_with_missing_validation_completion_is_rejected():
    manifest, _, expected, bundle = fixture_bundle(completed_count=1)
    def remove(item):
        item["raw_completions"].pop()
        item["result_sha256"] = collector.canonical({key: value for key, value in item.items()
                                                   if key != "result_sha256"})
    replace_json(bundle, f"results/{collector.ORDER[0]}/result.json", remove)
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        collector.validate_bundle(bundle, manifest, expected)


@pytest.mark.parametrize("mutation", ["missing_slot", "wrong_binding", "erase_prior_cost", "superiority"])
def test_full_recovery_report_cannot_hide_incomplete_or_unbound_evidence(mutation):
    manifest, _, expected, bundle = fixture_bundle(12, 12, True)
    if mutation == "missing_slot":
        bundle["completed_slots"].pop()
        message = "exactly the completed slots"
    else:
        def forge(item):
            if mutation == "wrong_binding": item["raw_result_bindings"][0]["result"]["sha256"] = "b" * 64
            elif mutation == "erase_prior_cost": item["combined_requested_gpu_hours"] = 84
            else: item["superiority_established"] = True
        replace_json(bundle, "campaign_report.json", forge)
        message = "artifact byte binding" if mutation == "wrong_binding" else "no-claim binding"
    with pytest.raises(ValueError, match=message):
        collector.validate_bundle(bundle, manifest, expected)


def test_historical_failure_receipt_is_validated_inside_status_only_collection():
    manifest, _, expected, bundle = fixture_bundle()
    replace_json(bundle, "source/prior_failed_phase.json", lambda item: item.update(prior_actual_gpu_hours=0))
    with pytest.raises(ValueError, match="Prior failed-phase frozen byte/path binding"):
        collector.validate_bundle(bundle, manifest, expected)


def test_partial_campaign_may_not_fetch_unfinished_raw_results():
    manifest, _, expected, bundle = fixture_bundle()
    slot = collector.ORDER[0]
    bundle["records"].append(record(manifest["native_control_root"] + f"/outputs/{slot}/complete.json",
                                     f"results/{slot}/complete.json", encoded({"synthetic_fixture": True}),
                                     "completed_job_receipt"))
    with pytest.raises(ValueError, match="unfinished or unadmitted"):
        collector.validate_bundle(bundle, manifest, expected)


def test_local_manifest_sha_rejected_before_any_remote_call(tmp_path):
    _, raw, _, _ = fixture_bundle()
    packet = tmp_path / "launch_manifest.json"
    packet.write_bytes(raw)
    def prohibited(*args, **kwargs):
        pytest.fail("Unexpected remote call")
    with pytest.raises(ValueError, match="Local prepared manifest byte SHA"):
        collector.collect_campaign(packet, "f" * 64, tmp_path / "snapshot", run=prohibited)
