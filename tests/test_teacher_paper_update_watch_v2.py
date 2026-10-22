"""Only metadata fixtures and an owned stdlib process; never paper edits."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/rloo_one_shot_20261016/watch_teacher_paper_update_v2.py"
spec = importlib.util.spec_from_file_location("teacher_paper_watch_under_test", SOURCE)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def fixture(tmp_path, status):
    owner = {"pid": 123, "source": {"sha256": "a"*64}}
    p = tmp_path / "owner.json"; p.write_text(json.dumps(owner))
    state = {"owner": owner, "status": status}
    (tmp_path / "status.json").write_text(json.dumps(state))
    return {"producer_owner": mod.binding(p)}, owner, state


@pytest.mark.parametrize("status", ["waiting_for_original_complete_OP_screen", "running", "waiting_for_idle_GPU"])
def test_waiting_never_parses_quality(tmp_path, status):
    packet, owner, _ = fixture(tmp_path, status)
    (tmp_path / "teacher_OP_analysis.json").write_bytes(b"deliberately not JSON")
    assert mod.decision(packet, owner) == ("wait", None)


@pytest.mark.parametrize("status", sorted(mod.STOP_STATES))
def test_inapplicable_or_failed_study_leaves_paper_alone(tmp_path, status):
    packet, owner, _ = fixture(tmp_path, status)
    assert mod.decision(packet, owner) == ("stop_without_paper_changes", None)


@pytest.mark.parametrize("status", sorted(mod.SUCCESS_STATES))
def test_only_complete_producer_binds_analysis_for_independent_publisher(tmp_path, status):
    packet, owner, state = fixture(tmp_path, status)
    p = tmp_path / "teacher_OP_analysis.json"; p.write_text("report bytes validated by publisher")
    state["analysis"] = mod.binding(p); (tmp_path / "status.json").write_text(json.dumps(state))
    assert mod.decision(packet, owner) == ("publish", state["analysis"])
    p.write_text("drift")
    with pytest.raises(ValueError, match="changed"):
        mod.decision(packet, owner)


def test_producer_owner_drift_denied_before_report(tmp_path):
    packet, owner, state = fixture(tmp_path, "running")
    state["owner"] = {"pid": 456}; (tmp_path / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="owner changed"):
        mod.decision(packet, owner)


def test_complete_state_cannot_select_another_report(tmp_path):
    packet, owner, state = fixture(tmp_path, next(iter(mod.SUCCESS_STATES)))
    state["analysis"] = {"path": str(tmp_path / "other.json")}
    (tmp_path / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="own complete analysis"):
        mod.decision(packet, owner)


def test_default_invocation_never_opens_request_or_writes(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--packet", str(tmp_path / "missing")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert json.loads(result.stdout)["paper_writes"] == 0
    assert not list(tmp_path.iterdir())


def test_symlink_evidence_denied(tmp_path):
    p = tmp_path / "real"; p.write_text("bytes"); row = mod.binding(p)
    link = tmp_path / "link"; link.symlink_to(p); row["path"] = str(link)
    with pytest.raises(ValueError, match="Canonical"):
        mod.read_bound(row)


def test_zero_exit_without_durable_publication_is_insufficient(tmp_path):
    packet = {"request_template": {"output": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        mod.verify_publication(packet, {"sha256": "a"*64}, {"sha256": "b"*64})


def test_publication_must_match_the_exact_parent_request(tmp_path):
    packet = {"request_template": {"output": str(tmp_path)}, "renderer": {"sha256": "a"*64},
              "publisher": {"sha256": "b"*64}}
    report = {"status": "published_verified_actual_teacher_development", "request_binding": {"sha256": "c"*64},
              "analysis": {"sha256": "d"*64}, "renderer": packet["renderer"], "publisher": packet["publisher"],
              "superiority_established": False, "publication_eligible": False}
    report["receipt_sha256"] = mod.sha(mod.encoded(report))
    (tmp_path / "publication.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source/request/analysis"):
        mod.verify_publication(packet, {"sha256": "different"}, report["analysis"])


@pytest.mark.parametrize("field,value", [("deadline_utc", "later"), ("publisher_timeout_seconds", 7200.0),
    ("publisher_timeout_seconds", 7201), ("automatic_retry", True)])
def test_time_retry_authority_denied_before_file_reads(field, value):
    p = {"protocol": mod.PROTOCOL, "deadline_utc": mod.DEADLINE, "publisher_timeout_seconds": 7200,
         "automatic_retry": False, "producer_packet": {"sha256": mod.PACKET_SHA}, "renderer": {"sha256": mod.RENDERER_SHA}}
    p[field] = value; p["packet_sha256"] = mod.sha(mod.encoded(p))
    with pytest.raises(ValueError, match="finite CPU"):
        mod.validate(p)
