"""Metadata fixtures and one bounded owned CPU process group; no paper edits."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

SOURCE = Path(__file__).parents[1]/"experiments/rloo_one_shot_20261016/watch_confirmation_paper_update_v3.py"
spec = importlib.util.spec_from_file_location("confirmation_paper_watch_guards", SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(M.encoded(value)+b"\n"); return M.binding(path)


def fixture(tmp_path, producer_status, prior_status="completed_admitted_development_paper_update"):
    producer = {"pid": 123, "source": {"sha256": "a"*64}}
    prior = {"pid": 456, "source": {"sha256": "b"*64}}
    packet = {"producer_owner": save(tmp_path/"producer/owner.json", producer),
              "prior_paper_owner": save(tmp_path/"prior/owner.json", prior)}
    analysis = tmp_path/"producer/confirmation_analysis.json"; analysis.write_text("Deliberately not JSON; watcher reads no metrics")
    state = {"owner": producer, "status": producer_status, "confirmation_analysis": M.binding(analysis)}
    save(tmp_path/"producer/status.json", state)
    save(tmp_path/"prior/status.json", {"owner": prior, "status": prior_status})
    return packet, producer, prior, state


def test_default_is_inert_with_missing_packet(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--packet", str(tmp_path/"missing")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and json.loads(result.stdout)["paper_writes"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", ["waiting_for_selected_development", "running", "waiting_for_exclusive_A6000"])
def test_incomplete_confirmation_never_reads_analysis_or_prior(tmp_path, status):
    packet, producer, prior, _ = fixture(tmp_path, status)
    (tmp_path/"producer/confirmation_analysis.json").unlink()
    (tmp_path/"prior/status.json").write_text("not JSON")
    assert M.decision(packet, producer, prior) == ("wait_for_confirmation", None)


@pytest.mark.parametrize("status", sorted(M.STOP))
def test_no_complete_comparison_does_not_edit_paper(tmp_path, status):
    packet, producer, prior, _ = fixture(tmp_path, status)
    assert M.decision(packet, producer, prior) == ("stop_without_paper_changes", None)


@pytest.mark.parametrize("status", sorted(M.SUCCESS))
def test_both_pass_and_fail_wait_for_prior_paper_writer(tmp_path, status):
    packet, producer, prior, _ = fixture(tmp_path, status, "independent_replay_and_staged_publication")
    (tmp_path/"producer/confirmation_analysis.json").unlink()
    assert M.decision(packet, producer, prior) == ("wait_for_previous_paper_update", None)


@pytest.mark.parametrize("status", sorted(M.SUCCESS))
@pytest.mark.parametrize("prior_status", sorted(M.PRIOR_DONE))
def test_all_producers_closed_before_snapshot(tmp_path, status, prior_status):
    packet, producer, prior, state = fixture(tmp_path, status, prior_status)
    assert M.decision(packet, producer, prior) == ("snapshot_then_publish", state["confirmation_analysis"])


def test_ambiguous_prior_publication_blocks_new_snapshot(tmp_path):
    packet, producer, prior, _ = fixture(tmp_path, next(iter(M.SUCCESS)), "failed_requires_reconciliation")
    with pytest.raises(ValueError, match="reconciliation"): M.decision(packet, producer, prior)


@pytest.mark.parametrize("which", ["producer", "prior"])
def test_foreign_owner_blocks_paper(tmp_path, which):
    packet, producer, prior, state = fixture(tmp_path, next(iter(M.SUCCESS)))
    p = tmp_path/which/"status.json"; value = json.loads(p.read_text()); value["owner"] = {"pid": 999}; save(p, value)
    with pytest.raises(ValueError, match="owner changed"): M.decision(packet, producer, prior)


def test_complete_producer_cannot_point_to_different_analysis(tmp_path):
    packet, producer, prior, state = fixture(tmp_path, next(iter(M.SUCCESS)))
    state["confirmation_analysis"]["path"] = str(tmp_path/"elsewhere")
    save(tmp_path/"producer/status.json", state)
    with pytest.raises(ValueError, match="own all-slot"): M.decision(packet, producer, prior)


@pytest.mark.parametrize("field,value", [("publisher_timeout_seconds", True), ("publisher_timeout_seconds", 7201),
    ("snapshot_timeout_seconds", 121), ("deadline_utc", "later"), ("automatic_retry", True)])
def test_no_unbounded_or_retried_publication(field, value):
    packet = {"protocol": M.PROTOCOL, "deadline_utc": M.DEADLINE, "publisher_timeout_seconds": 7200,
              "snapshot_timeout_seconds": 120, "automatic_retry": False, "superiority_established": False}
    packet[field] = value; packet = M.seal(packet, "packet_sha256")
    with pytest.raises(ValueError, match="finite"): M.validate(packet)


def publication_fixture(tmp_path):
    paper = tmp_path/"synthetic_paper"; paper.mkdir()
    snap = save(tmp_path/"snapshot.json", {"paper": str(paper), "fixture_only": True})
    pub = tmp_path/"publication"; pub.mkdir()
    packet = {"publication_output": str(pub), "renderer": {"sha256": "a"*64}, "publisher": {"sha256": "b"*64}}
    request = {"sha256": "c"*64}; analysis = {"sha256": "d"*64}
    files = {}
    for name in ("main.tex", "main.pdf", "main.log", "evidence_manifest.json", "compile_receipt.json", "artifact_verification.json",
        "verify_confirmation_artifacts_v3.py", "generated/confirmation_op_summary_table.tex",
        "generated/confirmation_op_contrast_table.tex", "generated/confirmation_op_section.tex"):
        p = paper/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic guard fixture; no publication")
        files[name] = M.binding(p)
    packet["artifact_verifier"] = files["verify_confirmation_artifacts_v3.py"]
    report = {"protocol": M.PUBLISHER_PROTOCOL, "status": "published_verified_actual_OP_confirmation",
        "request_binding": request, "analysis": analysis, "renderer": packet["renderer"], "publisher": packet["publisher"],
        "before_snapshot": json.loads(M.read_bound(snap)), "published_files": files,
        "scoped_confirmation_established": False, "registered_global_IUT_passed": False,
        "superiority_established": False, "publication_eligible": False}
    save(pub/"publication.json", M.seal(report, "receipt_sha256"))
    return packet, request, analysis, snap, report


def test_verify_actual_published_file_hashes(tmp_path):
    packet, request, analysis, snap, _ = publication_fixture(tmp_path)
    assert M.verify_publication(packet, request, analysis, snap)["path"].endswith("publication.json")
    (tmp_path/"synthetic_paper/main.pdf").write_text("foreign edit")
    with pytest.raises(ValueError, match="changed"): M.verify_publication(packet, request, analysis, snap)


@pytest.mark.parametrize("field,value", [("request_binding", {"sha256": "e"*64}), ("status", "started"),
    ("superiority_established", True), ("scoped_confirmation_established", True), ("scoped_confirmation_established", None)])
def test_exit_zero_or_resealed_wrong_receipt_is_insufficient(tmp_path, field, value):
    packet, request, analysis, snap, report = publication_fixture(tmp_path); report[field] = value
    save(tmp_path/"publication/publication.json", M.seal(report, "receipt_sha256"))
    with pytest.raises(ValueError): M.verify_publication(packet, request, analysis, snap)


@pytest.mark.parametrize("name", ["verify_confirmation_artifacts_v3.py", "generated/confirmation_op_summary_table.tex",
    "generated/confirmation_op_contrast_table.tex", "generated/confirmation_op_section.tex"])
def test_receipt_cannot_omit_the_actual_new_scientific_layer(tmp_path, name):
    packet, request, analysis, snap, report = publication_fixture(tmp_path)
    report["published_files"].pop(name)
    save(tmp_path/"publication/publication.json", M.seal(report, "receipt_sha256"))
    with pytest.raises(ValueError, match="artifact closure"): M.verify_publication(packet, request, analysis, snap)


def test_snapshot_is_captured_at_firing_then_bound_to_publication(tmp_path, monkeypatch):
    out = tmp_path/"watch"; packet = {"source": {"sha256": "a"*64}, "output": str(out),
        "renderer": {"sha256": "b"*64}, "publisher": {"sha256": "c"*64}, "publication_output": str(tmp_path/"pub")}
    pb = save(tmp_path/"packet.json", packet); analysis = {"sha256": "d"*64}; calls = []
    monkeypatch.setattr(M, "validate", lambda p: ({"pid": 1}, {"pid": 2}))
    monkeypatch.setattr(M, "__executed_source_sha256__", "a"*64, raising=False)
    monkeypatch.setattr(M, "decision", lambda *args: ("snapshot_then_publish", analysis))
    def child(p, args, name, timeout, output, state):
        calls.append(name)
        if name == "capture_actual_paper_baseline":
            assert not (out/"publication_request.json").exists()
            save(Path(args[-1]), {"paper": str(tmp_path/"fixture_paper")})
        else:
            request = json.loads((out/"publication_request.json").read_text())
            assert request["analysis_binding"] == analysis
            assert json.loads(M.read_bound(request["expected_paper_snapshot"]))["paper"].endswith("fixture_paper")
    monkeypatch.setattr(M, "run_cpu_child", child)
    monkeypatch.setattr(M, "verify_publication", lambda *args: {"fixture": "no publication"})
    M.run(pb)
    assert calls == ["capture_actual_paper_baseline", "independent_replay_and_staged_publication"]
    assert json.loads((out/"status.json").read_text())["status"] == "completed_admitted_confirmation_paper_update"


def test_owned_cpu_timeout_reaps_only_its_group(tmp_path, monkeypatch):
    marker = tmp_path/"own_child.json"
    source = tmp_path/"owned_fixture.py"
    source.write_text("import json,os,subprocess,sys,time\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'parent':os.getpid(),'child':p.pid}}))\n"
        "time.sleep(60)\n")
    row = M.binding(source); monkeypatch.setattr(M, "PYTHON", sys.executable)
    packet = {"publisher": row, "renderer": row}; state = {}
    with pytest.raises(ValueError, match="expired"):
        M.run_cpu_child(packet, [], "owned_timeout_fixture", 1, tmp_path, state)
    pids = json.loads(marker.read_text())
    for pid in pids.values():
        p = Path(f"/proc/{pid}/stat")
        assert not p.exists() or p.read_text().rsplit(")", 1)[1].split()[0] in ("Z", "X")


def test_post_spawn_heartbeat_failure_cleans_own_child(tmp_path, monkeypatch):
    source = tmp_path/"fixture.py"; source.write_text("# owned mocked child only\n")
    row = M.binding(source); process = types.SimpleNamespace(pid=123); stopped = []
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: process)
    def failed(*args): raise OSError("synthetic heartbeat disk failure")
    monkeypatch.setattr(M, "heartbeat", failed)
    monkeypatch.setattr(M, "stop_group", lambda child: stopped.append(child))
    with pytest.raises(OSError, match="heartbeat"):
        M.run_cpu_child({"publisher": row, "renderer": row}, [], "owned_mocked_heartbeat_failure", 1, tmp_path, {})
    assert stopped == [process]
