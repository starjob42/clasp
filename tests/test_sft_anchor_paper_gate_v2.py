"""Synthetic phase-two publication fixtures; no model/scheduler/publication actions."""
import json
import math

import pytest

from test_sft_anchor_paper_gate import gate, ROOT
from test_sft_anchor_recovery_collection import collector, encoded, fixture_bundle, record


def make_snapshot(tmp_path, *, completed=12):
    manifest, _, expected, bundle = fixture_bundle(completed, 12, completed == 12)
    decoded = collector.decode_records(bundle)
    if completed == 12:
        metadata, raw = decoded["campaign_report.json"]
        report = json.loads(raw)
        report.update(bootstrap_replicates=50000, bootstrap_seed=20261013,
                      training_seeds_by_task={"op": [43], "cvrp": [43]}, groups=[])
        for task in gate.TASKS:
            group = {"task": task, "train_seed": 43, "matched": True,
                     "missing_arms": [], "completed_arms": list(gate.ARMS), "comparisons": []}
            for candidate in gate.ARMS:
                if not candidate.startswith("rloo_"):
                    continue
                for control in ("foarl", "sft"):
                    for mode in ("greedy", "k8"):
                        left = json.loads(decoded[f"results/{task}.{candidate}/result.json"][1])["selected"][mode]
                        right = json.loads(decoded[f"results/{task}.{control}/result.json"][1])["selected"][mode]
                        joint = [(a, b) for a, b in zip(left, right) if a["feasible"] and b["feasible"]]
                        contrast = {"left": candidate, "right": control, "decoder": mode, "N": 96,
                                    "n_joint": len(joint), "bootstrap_replicates": 50000, "bootstrap_seed": 20261013}
                        for name, values in (
                            ("feasibility_delta_pp", [100 * (int(a["feasible"]) - int(b["feasible"])) for a, b in zip(left, right)]),
                            ("u10_delta_pp", [100 * (a["u10"] - b["u10"]) for a, b in zip(left, right)]),
                            ("joint_signed_gap_delta_pp", [100 * (a["signed_gap"] - b["signed_gap"]) for a, b in joint]),
                            ("joint_clipped_gap_delta_pp", [100 * (a["clipped_gap"] - b["clipped_gap"]) for a, b in joint])):
                            assert len(set(values)) <= 1, "Synthetic equal-arm contrasts must be degenerate"
                            value = math.fsum(values) / len(values) if values else None
                            contrast[name] = {"estimate": value, "ci95": [value, value] if values else None}
                        group["comparisons"].append(contrast)
            report["groups"].append(group)
        decoded["campaign_report.json"] = metadata, encoded(report)
        bundle["records"] = [record(item["native_path"], path, raw, item["purpose"], item["tail_only"])
                             for path, (item, raw) in decoded.items()]
    admission, decoded = collector.validate_bundle(bundle, manifest, expected)
    folder = tmp_path / "synthetic_offline_v2_immutable_collection"
    folder.mkdir()
    for path, (_, raw) in decoded.items():
        collector.write_bytes_once(folder / path, raw)
    collector_path = ROOT / gate.STUDY / "collect_campaign_v2.py"
    receipt = {"schema_version": 2, "study": collector.STUDY, "synthetic_offline_fixture": True,
               "collected_utc": bundle["collected_utc"], "prepared_manifest_sha256": expected,
               "admission": admission, "source_proofs": bundle["source_proofs"],
               "read_only_queries": bundle["read_only_queries"], "owner_probes": bundle["owner_probes"],
               "terminal_artifact_proofs": bundle["terminal_artifact_proofs"],
               "collector": {"path": str(collector_path), "sha256": gate.identity(collector_path)["sha256"]},
               "snapshots": [{k: v for k, v in r.items() if k != "content_base64"} for r in bundle["records"]],
               "native_mutations": False, "superiority_established": False}
    receipt["collection_sha256"] = gate.digest(receipt)
    collector.write_bytes_once(folder / "collection.json", encoded(receipt))
    return folder / "collection.json", expected


def load_fixture(path, expected):
    return gate.load_completed_collection(path, root=ROOT, expected_launch_sha256=expected,
                                          offline_collector_filename="collect_campaign_v2.py")


def test_v2_pending_fixture_never_enters_performance_layer(tmp_path):
    path, expected = make_snapshot(tmp_path, completed=1)
    assert load_fixture(path, expected) is None


def test_v2_complete_fixture_preserves_terminal_proofs_and_replays_exact_full_cohort(tmp_path):
    path, expected = make_snapshot(tmp_path)
    evidence = load_fixture(path, expected)
    summaries, contrasts = gate.checked_summaries(evidence)
    assert len(summaries) == 12 and len(contrasts) == 32
    assert all(row["N"] == 96 and 0 <= row["signed_sensitive_quality_percent"] <= 100
               for modes in summaries.values() for row in modes.values())
    assert len(evidence["receipt"]["terminal_artifact_proofs"]) == 12
    assert len(gate.rendered_tables(summaries, contrasts)) == 6
    assert evidence["packet_spec"] is None and evidence["receipt"]["synthetic_offline_fixture"] is True


def test_v2_fixture_cannot_publish_through_live_builder(tmp_path):
    path, _ = make_snapshot(tmp_path)
    called = []
    before = set((gate.HERE / "generated").glob("sft_anchor_*"))
    with pytest.raises(ValueError, match="fixed production launch"):
        gate.build_layer(called.append, collection=path)
    assert not called and before == set((gate.HERE / "generated").glob("sft_anchor_*"))


def test_v2_full_fixture_requires_every_independent_terminal_tree(tmp_path):
    path, expected = make_snapshot(tmp_path)
    receipt = json.loads(path.read_bytes())
    receipt["terminal_artifact_proofs"].pop(collector.ORDER[0])
    receipt["collection_sha256"] = gate.digest({k: v for k, v in receipt.items() if k != "collection_sha256"})
    path.chmod(0o640)
    path.write_bytes(encoded(receipt))
    path.chmod(0o440)
    with pytest.raises(ValueError, match="[Tt]erminal"):
        load_fixture(path, expected)


def test_known_production_collector_cannot_be_overridden():
    # This check occurs before any completed evidence is read or rendered.
    path = ROOT / gate.STUDY / "collected_status_v1/collection.json"
    with pytest.raises(ValueError, match="cannot be overridden"):
        gate.load_completed_collection(path, offline_collector_filename="collect_campaign_v2.py")


def test_actual_frozen_v2_provenance_is_complete_and_is_not_policy_evidence():
    provenance = gate.frozen_v2_provenance()
    assert provenance["source_files_verified"] == 121 and provenance["source_bytes_verified"] == 15536549
    assert provenance["completed_7B_policy_results_included"] is False
    assert provenance["benchmark_superiority_established"] is False
    assert provenance["combined_requested_GPU_hour_cap"] == 92 and provenance["preserved_prior_failed_job"] == "1030929"
    manifest = {"sft_anchor_corrected_launch_provenance": provenance, "sources": provenance["source_bindings"]}
    assert gate.check_frozen_v2_provenance(manifest)["passed"] is True


def test_forged_frozen_v2_recipe_metadata_is_rejected_without_file_changes():
    provenance = gate.frozen_v2_provenance()
    provenance["fixed_RL_updates_per_trained_arm"] = 99
    manifest = {"sft_anchor_corrected_launch_provenance": provenance, "sources": provenance["source_bindings"]}
    with pytest.raises(ValueError, match="provenance metadata differs"):
        gate.check_frozen_v2_provenance(manifest)
