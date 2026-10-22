"""Unscored raw-generation admission fixtures; never production model results."""
import json

import pytest

import test_sft_anchor_paper_gate_v2 as old_fixture
from test_sft_anchor_paper_gate import gate, ROOT
from test_sft_anchor_admission_v3_collection import collector, encoded, fixture_bundle


def make_snapshot(tmp_path, monkeypatch, *, completed=12):
    monkeypatch.setattr(old_fixture, "collector", collector)
    monkeypatch.setattr(old_fixture, "fixture_bundle", fixture_bundle)
    path, expected = old_fixture.make_snapshot(tmp_path, completed=completed)
    receipt = json.loads(path.read_bytes())
    collector_path = ROOT / gate.STUDY / "collect_campaign_admission_v3.py"
    receipt.update(synthetic_fixture=True, collector={"path": str(collector_path),
        "sha256": gate.identity(collector_path)["sha256"]})
    receipt["collection_sha256"] = gate.digest({k: v for k, v in receipt.items() if k != "collection_sha256"})
    path.chmod(0o640)
    path.write_bytes(encoded(receipt))
    path.chmod(0o440)
    return path, expected


def load_fixture(path, expected):
    return gate.load_completed_collection(path, expected_launch_sha256=expected,
        offline_collector_filename="collect_campaign_admission_v3.py")


def test_v3_unscored_full_fixture_replays_twelve_exact_cohorts(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch)
    evidence = load_fixture(path, expected)
    summaries, contrasts = gate.checked_summaries(evidence)
    assert len(summaries) == 12 and len(contrasts) == 32
    assert len(gate.rendered_tables(summaries, contrasts)) == 6
    raw = json.loads((path.parent / "results/op.rloo_cardinal_beta005/evaluation_groups.jsonl").read_bytes().splitlines()[0])["raw_completions"][0]
    scored = evidence["results"]["op", "rloo_cardinal_beta005"]["raw_completions"][0]
    assert set(scored) - set(raw) == collector.SCORER_ENRICHMENT_FIELDS
    assert collector.raw_generation_projection(scored) == raw
    assert evidence["packet_spec"] is None and evidence["admission_spec"] is None


def test_v3_partial_fixture_never_creates_performance_layer(tmp_path, monkeypatch):
    path, expected = make_snapshot(tmp_path, monkeypatch, completed=1)
    assert load_fixture(path, expected) is None


def test_v3_synthetic_fixture_cannot_publish_live_tables(tmp_path, monkeypatch):
    path, _ = make_snapshot(tmp_path, monkeypatch)
    called = []
    before = set((gate.HERE / "generated").glob("sft_anchor_*"))
    with pytest.raises(ValueError, match="fixed production launch"):
        gate.build_layer(called.append, collection=path)
    assert not called and before == set((gate.HERE / "generated").glob("sft_anchor_*"))


def test_closed_admission_patch_keeps_original_producer_and_four_field_projection():
    patch = gate.admission_v3_provenance()
    assert patch["preserved_launch_manifest_sha256"] == gate.V2_LAUNCH_SHA256
    assert patch["preserved_source_manifest_sha256"] == gate.PACKETS[gate.V2_LAUNCH_SHA256]["source_manifest_sha256"]
    assert patch["model_reward_credit_sampler_checkpoint_producer_changed"] is False
    assert patch["completed_7B_policy_results_included"] is False
    assert patch["removed_only_four_scorer_derived_fields_for_raw_join"] is True
    manifest = {"sft_anchor_admission_v3_provenance": patch, "sources": patch["source_bindings"]}
    assert gate.check_admission_v3_provenance(manifest)["passed"] is True


def test_live_v3_collector_requires_exact_operational_patch_before_any_admission(tmp_path):
    path = tmp_path / "missing_patch_collection.json"
    value = {"prepared_manifest_sha256": gate.V2_LAUNCH_SHA256,
        "collector": {"sha256": gate.ADMISSION_V3["collector_sha256"]},
        "admission": {"status": "status_only_campaign_incomplete"}, "operational_patch": None}
    value["collection_sha256"] = gate.digest(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact registered operational patch"):
        gate.load_completed_collection(path)
