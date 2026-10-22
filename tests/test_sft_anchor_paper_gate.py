"""Synthetic/offline publication-gate fixtures; never write live paper results.

No scheduler, model, GPU or protected-data operation occurs. The fake launch
digest is eligible only through the pure loader's explicit test parameter.
"""
import base64
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from test_sft_anchor_collection import collector, encoded, fixture_bundle, record, scorer
import test_sft_anchor_collection as collection_fixture

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "paper/rloo9_20261008/sft_anchor_results.py"
SPEC = importlib.util.spec_from_file_location("offline_sft_anchor_paper_gate", MODULE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def fixture_contract(task):
    rows = []
    for i in range(1, 225):
        instance = ([[[0, 0], [i, 0], [2 * i, 0]], [0, 4, 6], 3 * i] if task == "op"
                    else [[[0, 0], [i, 0], [0, i]], [0, 1, 1], 2])
        rows.append({"instance_id": f"synthetic_offline_{task}_{i}", "instance_json": instance,
                     "instruction": "synthetic offline gate fixture", "input": json.dumps(instance),
                     "output": "Route: [0,1,2]" if task == "op" else "Routes: [[0,1,2,0]]"})
    return scorer.make_contract(task, {"id": "synthetic_offline_original_sft", "sha256": "a" * 64},
                                rows[:128], rows[128:], scope="production", horizon=100)


def make_snapshot(tmp_path, monkeypatch, *, completed=12, full_cohort=True, invalid_arm=False):
    if full_cohort:
        monkeypatch.setattr(collection_fixture, "make_contract", fixture_contract)
    original_evaluate = scorer.evaluate_arm
    def unique_fixture_ids(contract, arm, completions, checkpoint, resources):
        completions = copy.deepcopy(completions)
        for row in completions:
            prefix = row["instance_id"] + "/"
            if not row["completion_id"].startswith(prefix):
                row["completion_id"] = prefix + row["completion_id"]
        return original_evaluate(contract, arm, completions, checkpoint, resources)
    monkeypatch.setattr(scorer, "evaluate_arm", unique_fixture_ids)
    manifest, _, expected_sha, bundle = fixture_bundle(
        completed_count=completed, attempts_count=12, campaign_report=completed == 12)
    decoded = collector.decode_records(bundle)
    contracts = {task: json.loads(decoded[f"source/contracts/{task}.json"][1]) for task in gate.TASKS}
    for slot in collector.ORDER[:completed]:
        path = f"results/{slot}/complete.json"
        item, raw = decoded[path]
        complete = json.loads(raw)
        complete["actual_evaluation_responses"] = len(contracts[slot.split('.')[0]]["validation_rows"]) * 9
        decoded[path] = (item, encoded(complete))
    if invalid_arm:
        path = "results/op.rloo_scalar_beta0/result.json"
        item, raw = decoded[path]
        result = json.loads(raw)
        for candidate in result["raw_completions"]:
            candidate["text"] = "Route: [0,0]"
        result = scorer.evaluate_arm(contracts["op"], "rloo_scalar_beta0", result["raw_completions"],
                                     result["checkpoint"], result["resources"])
        decoded[path] = item, encoded(result)
    if completed == 12:
        item, raw = decoded["campaign_report.json"]
        report = json.loads(raw)
        for binding in report["raw_result_bindings"]:
            for kind in ("complete", "result"):
                _, content = decoded[f"results/{binding['slot']}/{kind}.json"]
                binding[kind].update(sha256=collector.sha(content), size_bytes=len(content))
        report.update(bootstrap_seed=20261013, bootstrap_replicates=50000,
                      training_seeds_by_task={"op": [43], "cvrp": [43]}, groups=[])
        for task in gate.TASKS:
            group = {"task": task, "train_seed": 43, "matched": True, "missing_arms": [],
                     "completed_arms": list(gate.ARMS), "comparisons": []}
            for candidate in gate.ARMS:
                if not candidate.startswith("rloo_"):
                    continue
                for control in ("foarl", "sft"):
                    for mode in ("greedy", "k8"):
                        left = json.loads(decoded[f"results/{task}.{candidate}/result.json"][1])["selected"][mode]
                        right = json.loads(decoded[f"results/{task}.{control}/result.json"][1])["selected"][mode]
                        joint = [(a, b) for a, b in zip(left, right) if a["feasible"] and b["feasible"]]
                        row = {"left": candidate, "right": control, "decoder": mode,
                               "N": len(left), "n_joint": len(joint), "bootstrap_seed": 20261013,
                               "bootstrap_replicates": 50000}
                        for field, values in (
                            ("feasibility_delta_pp", [100 * (int(a["feasible"]) - int(b["feasible"])) for a, b in zip(left, right)]),
                            ("u10_delta_pp", [100 * (a["u10"] - b["u10"]) for a, b in zip(left, right)]),
                            ("joint_signed_gap_delta_pp", [100 * (a["signed_gap"] - b["signed_gap"]) for a, b in joint]),
                            ("joint_clipped_gap_delta_pp", [100 * (a["clipped_gap"] - b["clipped_gap"]) for a, b in joint])):
                            require_constant = len(set(values)) <= 1
                            assert require_constant, "Offline fixture must have mathematically degenerate resampling"
                            value = math.fsum(values) / len(values) if values else None
                            row[field] = {"estimate": value, "ci95": [value, value] if values else None}
                        group["comparisons"].append(row)
            report["groups"].append(group)
        decoded["campaign_report.json"] = item, encoded(report)
    bundle["records"] = [record(item["native_path"], path, raw, item["purpose"], item["tail_only"])
                         for path, (item, raw) in decoded.items()]
    admission, decoded = collector.validate_bundle(bundle, manifest, expected_sha)
    folder = tmp_path / "synthetic_offline_immutable_collection"
    folder.mkdir()
    for path, (_, raw) in decoded.items():
        collector.write_bytes_once(folder / path, raw)
    receipt = {"schema_version": 1, "study": collector.STUDY, "synthetic_offline_fixture": True,
               "collected_utc": bundle["collected_utc"], "prepared_manifest_sha256": expected_sha,
               "admission": admission, "source_proofs": bundle["source_proofs"],
               "read_only_queries": bundle["read_only_queries"], "owner_probes": [],
               "collector": {"path": str(ROOT / gate.STUDY / "collect_campaign.py"),
                             "sha256": collector.sha((ROOT / gate.STUDY / "collect_campaign.py").read_bytes())},
               "snapshots": [{k: v for k, v in r.items() if k != "content_base64"} for r in bundle["records"]],
               "native_mutations": False, "superiority_established": False}
    receipt["collection_sha256"] = gate.digest(receipt)
    collector.write_bytes_once(folder / "collection.json", encoded(receipt))
    return folder / "collection.json", expected_sha


def load_fixture(path, expected):
    return gate.load_completed_collection(path, root=ROOT, expected_launch_sha256=expected)


def test_no_collection_has_no_bindings_or_outputs():
    called = []
    assert gate.build_layer(called.append) is None
    assert called == []


def test_pending_snapshot_never_publishes_partial_performance(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch, completed=1)
    assert load_fixture(path, expected) is None


def test_fixture_digest_cannot_enter_live_publication_builder(tmp_path, monkeypatch):
    path, _ = make_snapshot(tmp_path, monkeypatch)
    called = []
    before = set((gate.HERE / "generated").glob("sft_anchor_*"))
    with pytest.raises(ValueError, match="fixed production launch"):
        gate.build_layer(called.append, collection=path)
    assert not called and before == set((gate.HERE / "generated").glob("sft_anchor_*"))


def test_one_row_collector_fixture_is_rejected_by_exact96_gate(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch, full_cohort=False)
    with pytest.raises(ValueError, match="128 training and96"):
        load_fixture(path, expected)


def test_complete_offline96_gate_replays_all_arms_and_renders_separate_tables(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    evidence = load_fixture(path, expected)
    summaries, contrasts = gate.checked_summaries(evidence)
    tables = gate.rendered_tables(summaries, contrasts)
    assert len(summaries) == 12 and len(contrasts) == 32 and len(tables) == 6
    assert all(s["greedy"]["N"] == 96 and s["k8"]["n"] == 96 for s in summaries.values())
    assert all("development" in text and "superiority" in text for text in tables.values())
    for name, text in tables.items():
        (tmp_path / ("synthetic_offline_" + name)).write_text(text)


def test_all_invalid_support_stays_undefined_and_all_instance_outcomes_remain(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch, invalid_arm=True)
    summaries, contrasts = gate.checked_summaries(load_fixture(path, expected))
    summary = summaries[("op", "rloo_scalar_beta0")]["k8"]
    assert summary["n"] == 0 and summary["signed_gap_percent"] is None and summary["u10_percent"] == 0
    contrast = contrasts[("op", "rloo_scalar_beta0", "foarl", "k8")]
    assert contrast["N"] == 96 and contrast["n_joint"] == 0
    assert contrast["feasibility_delta_pp"]["estimate"] == -100
    assert contrast["joint_signed_gap_delta_pp"] == {"estimate": None, "ci95": None}
    assert "---" in gate.rendered_tables(summaries, contrasts)["sft_anchor_op_marginal_table.tex"]


def test_mutated_raw_snapshot_is_rejected(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    raw = path.parent / "results/op.foarl/result.json"
    raw.chmod(0o640)
    raw.write_bytes(raw.read_bytes() + b" ")
    raw.chmod(0o440)
    with pytest.raises(ValueError, match="Snapshot bytes"):
        load_fixture(path, expected)


def test_writable_completed_receipt_is_rejected(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    path.chmod(0o640)
    with pytest.raises(ValueError, match="not immutable"):
        load_fixture(path, expected)


@pytest.mark.parametrize("change", ["estimate", "ci95"])
def test_forged_paired_points_or_intervals_are_rejected(tmp_path, monkeypatch, change):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    evidence = load_fixture(path, expected)
    metric = evidence["report"]["groups"][0]["comparisons"][0]["u10_delta_pp"]
    metric[change] = 1 if change == "estimate" else [1, 1]
    with pytest.raises(ValueError, match="Metric differs"):
        gate.checked_summaries(evidence)


def test_pending_checker_has_no_published_layers():
    assert gate.check_layer(manifest={}) == {
        "passed": True, "status": "pending_no_pilot_performance_tables", "published_layers": 0}


def test_prespecified_signed_quality_retains_negative_gap_information_and_failure_denominator():
    equal_reference = gate.signed_sensitive_quality({"feasible": True, "signed_gap": 0.0})
    better_reference = gate.signed_sensitive_quality({"feasible": True, "signed_gap": -.1})
    even_better = gate.signed_sensitive_quality({"feasible": True, "signed_gap": -.2})
    assert 0.5 == equal_reference < better_reference < even_better < 1
    assert gate.signed_sensitive_quality({"feasible": False, "signed_gap": None}) == 0
    assert gate.signed_sensitive_quality({"feasible": True, "signed_gap": 1e300}) == 0
    assert gate.signed_sensitive_quality({"feasible": True, "signed_gap": -1e300}) == 1
    with pytest.raises(ValueError, match="verified boolean"):
        gate.signed_sensitive_quality({"feasible": 1, "signed_gap": 0})
    with pytest.raises(ValueError, match="finite"):
        gate.signed_sensitive_quality({"feasible": True, "signed_gap": float("nan")})


def test_registry_preserves_frozen_v1_and_exact_v2_identities():
    original = gate.PACKETS[gate.EXPECTED_LAUNCH_SHA256]
    assert original["version"] == "v1" and original["source_root_suffix"] == "/source_v1"
    assert all(len(key) == 64 and set(key) <= set("0123456789abcdef") for key in gate.PACKETS)
    assert all(len(value["entrypoint_sha256"]) == 64 for value in gate.PACKETS.values())
    corrected = gate.PACKETS[gate.V2_LAUNCH_SHA256]
    assert corrected["version"] == "v2" and corrected["source_root_suffix"] == "/source_v2"
    packet = ROOT / gate.STUDY / "control_v2/launch_manifest.json"
    assert gate.identity(packet)["sha256"] == gate.V2_LAUNCH_SHA256


def test_nested_synthetic_outcome_marker_is_barred_from_production_admission():
    with pytest.raises(ValueError, match="Synthetic outcome"):
        gate.require_no_synthetic_outcomes({"raw_completions": [{"synthetic_fixture": True}]})
    gate.require_no_synthetic_outcomes({"raw_completions": [{"synthetic_fixture": False}]})


@pytest.mark.parametrize("endpoints", [[], [0], [0, 0, 0]])
def test_defined_interval_requires_exactly_two_endpoints(tmp_path, monkeypatch, endpoints):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    evidence = load_fixture(path, expected)
    evidence["report"]["groups"][0]["comparisons"][0]["feasibility_delta_pp"]["ci95"] = endpoints
    with pytest.raises(ValueError, match="exactly two endpoints"):
        gate.checked_summaries(evidence)


def test_nondegenerate_intervals_match_frozen_random_stream(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    evidence = load_fixture(path, expected)
    candidate = evidence["results"][("op", "rloo_scalar_beta0")]
    for mode in ("greedy", "k8"):
        # Synthetic qualified-vector arithmetic only: one five-percent gap.
        row = candidate["selected"][mode][0]
        row.update(signed_gap=.05, clipped_gap=.05, u10=.5)
        candidate["metrics"][mode].update(signed_gap_percent=5 / 96,
            clipped_gap_percent=5 / 96, u10_percent=100 * 95.5 / 96)
    left = candidate["selected"]["greedy"]
    right = evidence["results"][("op", "foarl")]["selected"]["greedy"]
    independently_frozen = scorer.paired_bootstrap(left, right, seed=20261013, replicates=50000)
    for group in evidence["report"]["groups"]:
        if group["task"] != "op":
            continue
        for row in group["comparisons"]:
            if row["left"] == "rloo_scalar_beta0":
                row.update(independently_frozen)
    _, contrasts = gate.checked_summaries(evidence)
    checked = contrasts[("op", "rloo_scalar_beta0", "foarl", "greedy")]
    assert checked["joint_clipped_gap_delta_pp"]["ci95"] == independently_frozen["joint_clipped_gap_delta_pp"]["ci95"]
