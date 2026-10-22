"""Synthetic transaction/stage guards. Never mutate the real manuscript."""
import json
import os
from pathlib import Path
import signal
import sys
import types

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/rloo_one_shot_20261016/publish_teacher_paper_v2.py"


def module():
    raw = SOURCE.read_bytes();value = types.ModuleType("publisher_synthetic_guard")
    value.__file__ = str(SOURCE)
    import hashlib
    value.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(SOURCE), "exec", dont_inherit=True), value.__dict__)
    return value


def fake_paper(tmp_path, monkeypatch):
    m = module();root = tmp_path / "project";paper = root / "paper/manuscript";paper.mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", root);monkeypatch.setattr(m, "PAPER", paper)
    assets = {"main.tex": b"prefix\n\\input{sections/discussion}\n\\bibliographystyle{plainnat}\n",
              "main.pdf": b"SYNTHETIC PDF BYTES", "main.log": b"SYNTHETIC LOG", "compile_receipt.json": b"{}\n",
              "artifact_verification.json": b"{}\n", "presentation_revision_verification.json": b"{}\n",
              "generated/old_table.tex": b"SYNTHETIC OLD TABLE\n", "sections/discussion.tex": b"SYNTHETIC DISCUSSION\n",
              "planning/static_science.json": b"{}\n", "planning/untouched_history.json": b"old history\n"}
    for name, raw in assets.items():
        p = paper / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(raw)
    prefix = paper.relative_to(root).as_posix() + "/"
    manifest = m.sealed({"sources": {prefix + name: {"path": prefix + name, **m.identity(raw)} for name, raw in assets.items()
                                    if name in ("main.tex", "sections/discussion.tex", "planning/static_science.json")},
                         "outputs": {"generated/old_table.tex": m.identity(assets["generated/old_table.tex"])},
                         "generated_tables": 1, "rendered_tables": 1, "figures": 0,
                         "separate_development_layers": [{"unchanged": True}]})
    (paper / "evidence_manifest.json").write_bytes(m.encoded(manifest) + b"\n")
    return m, paper, m.snapshot_paper()


def fake_layer(m):
    return {"protocol": "synthetic_diagnostic_only", "status": "synthetic_layout_only", "texts": {name: "SYNTHETIC " + name for name in m.NAMES},
            "source_bindings": [], "superiority_established": False, "publication_eligible": False}


def test_default_cli_no_writes_or_renderer_calls(tmp_path, monkeypatch, capsys):
    m = module();monkeypatch.setattr(sys, "argv", [str(SOURCE)])
    monkeypatch.setattr(m, "snapshot_paper", lambda: pytest.fail("Dryrun read the paper"))
    monkeypatch.setattr(m, "publish", lambda *a: pytest.fail("Dryrun published"))
    m.main();assert json.loads(capsys.readouterr().out)["status"] == "prepared_only_no_paper_changes"
    assert list(tmp_path.iterdir()) == []


def test_operational_lock_is_not_a_manuscript_change(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch)
    (paper / ".teacher_publish.lock").write_bytes(b"")
    assert m.verify_snapshot(snapshot)


@pytest.mark.parametrize("name", ["main.tex", "main.pdf", "generated/old_table.tex", "sections/discussion.tex", "planning/static_science.json"])
def test_original_source_pdf_and_numeric_changes_reject_snapshot(tmp_path, monkeypatch, name):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch)
    (paper / name).write_bytes(b"concurrent user edit")
    with pytest.raises(ValueError, match="Concurrent"):
        m.verify_snapshot(snapshot)


def test_generated_inventory_cannot_be_resealed_to_skip_checks(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch)
    snapshot["old_generated_TeX"] = {};snapshot.pop("receipt_sha256");snapshot = m.sealed(snapshot)
    with pytest.raises(ValueError, match="generated TeX"):
        m.verify_snapshot(snapshot)


def test_stage_copies_only_needed_history_and_does_not_edit_original(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch);out = tmp_path / "proof";out.mkdir()
    stage, proof = m.stage_layer(snapshot, fake_layer(m), out, diagnostic=True)
    assert stage.parent == m.ROOT / "paper"
    assert (stage / "planning/static_science.json").exists()
    assert not (stage / "planning/untouched_history.json").exists()
    assert m.verify_snapshot(snapshot)
    assert len(proof["old_generated_TeX_preserved"]) == 1
    manifest = json.loads((stage / "evidence_manifest.json").read_bytes())
    prefix = stage.relative_to(m.ROOT).as_posix() + "/"
    assert manifest["sources"][prefix + "main.tex"] == {"path": prefix + "main.tex", **m.identity((stage / "main.tex").read_bytes())}
    assert all(row["path"] == key for key, row in manifest["sources"].items())
    assert manifest["generated_tables"] == manifest["rendered_tables"] == 3
    assert manifest["separate_development_layers"] == [{"unchanged": True}]


def test_final_alias_rebind_uses_revised_main_and_new_sources(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch);out = tmp_path / "proof";out.mkdir()
    stage, proof = m.stage_layer(snapshot, fake_layer(m), out, diagnostic=True)
    manifest = m.rebind_final_manifest(stage, proof)
    prefix = paper.relative_to(m.ROOT).as_posix() + "/"
    assert manifest["sources"][prefix + "main.tex"] == {"path": prefix + "main.tex", **m.identity((stage / "main.tex").read_bytes())}
    assert manifest["sources"][prefix + "main.tex"]["sha256"] != snapshot["files"]["main.tex"]["sha256"]
    assert all(row["path"] == key for key, row in manifest["sources"].items())
    assert all(".rloo9_teacher_stage_" not in key for key in manifest["sources"])
    assert all(prefix + "generated/" + name in manifest["sources"] for name in m.NAMES)


def test_synthetic_diagnostic_requires_visible_markers(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch);out = tmp_path / "proof";out.mkdir();layer = fake_layer(m)
    layer["texts"][m.NAMES[0]] = "unmarked"
    with pytest.raises(ValueError, match="visibly mark"):
        m.stage_layer(snapshot, layer, out, diagnostic=True)


def test_public_renderer_rejects_synthetic_without_analysis_or_stage(tmp_path):
    m = module();renderer = SOURCE.parent / "render_teacher_op_paper_v2.py"
    analysis = tmp_path / "synthetic.json";analysis.write_bytes(b'{"synthetic_fixture":true}')
    with pytest.raises(ValueError, match="actual development"):
        m.admitted_layer(m.binding(analysis), m.binding(renderer))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.json"]


def test_unknown_renderer_rejected_before_open(tmp_path):
    m = module()
    with pytest.raises(ValueError, match="reviewed"):
        m.admitted_layer({}, {"path": str(tmp_path / "absent"), "sha256": "f" * 64})


@pytest.mark.parametrize("text", ["no discussion", "\\input{sections/discussion}\\input{sections/discussion}",
                                 "\\input{sections/discussion} teacher_op_development"])
def test_main_marker_ambiguity_rejected(text):
    m = module()
    with pytest.raises(ValueError):
        m.patch_main(text.encode())


def test_transaction_failure_rolls_back_all_own_changes(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"old")
    def fail(name):
        if name == "new":
            raise RuntimeError("forced write-stage failure")
    with pytest.raises(RuntimeError, match="forced"):
        m.transaction(paper, {"a": b"new", "new": b"newfile"}, {"a": m.identity(b"old")}, tmp_path / "backup", hook=fail)
    assert (paper / "a").read_bytes() == b"old" and not (paper / "new").exists()
    assert (tmp_path / "backup/a").read_bytes() == b"old"


def test_final_receipt_failure_is_inside_rollback(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"old")
    def fail():
        raise OSError("receipt unavailable")
    with pytest.raises(OSError, match="receipt"):
        m.transaction(paper, {"a": b"new"}, {"a": m.identity(b"old")}, tmp_path / "backup", finish=fail)
    assert (paper / "a").read_bytes() == b"old"


def test_concurrent_edit_before_transaction_is_never_replaced(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"user")
    with pytest.raises(ValueError, match="Concurrent"):
        m.transaction(paper, {"a": b"ours"}, {"a": m.identity(b"old")}, tmp_path / "backup")
    assert (paper / "a").read_bytes() == b"user"


def test_concurrent_edit_during_failure_is_preserved(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"old")
    def edit(name):
        (paper / name).write_bytes(b"concurrent user edit")
        raise RuntimeError("forced error after user edit")
    with pytest.raises(RuntimeError, match="concurrent edits retained"):
        m.transaction(paper, {"a": b"ours"}, {"a": m.identity(b"old")}, tmp_path / "backup", hook=edit)
    assert (paper / "a").read_bytes() == b"concurrent user edit"


def test_untouched_numeric_edit_fails_closure_and_rolls_back_own_tex(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"old");(paper / "oldtable").write_bytes(b"numbers")
    m.source_inventory = lambda paper: (["a", "oldtable"], {})
    snapshot = {"files": {"a": m.identity(b"old"), "oldtable": m.identity(b"numbers")}}
    replacements = {"a": b"ours"}
    def edit(name):
        (paper / "oldtable").write_bytes(b"user numbers")
    with pytest.raises(ValueError, match="Untouched"):
        m.transaction(paper, replacements, snapshot["files"], tmp_path / "backup", hook=edit,
                      finish=lambda: m.verify_retained_files(paper, snapshot, replacements))
    assert (paper / "a").read_bytes() == b"old"
    assert (paper / "oldtable").read_bytes() == b"user numbers"


def test_request_cli_needs_no_redundant_output_flag(tmp_path, monkeypatch, capsys):
    m = module();request = m.sealed({"protocol": m.PROTOCOL + "-request", "analysis_binding": {"kind": "synthetic"},
        "renderer_binding": {"kind": "synthetic"}, "expected_paper_snapshot": {"kind": "synthetic"}, "output": "fake", "superiority_established": False})
    p = tmp_path / "request.json";p.write_bytes(m.encoded(request))
    seen = [];monkeypatch.setattr(m, "publish", lambda *a, **kw: seen.append((a, kw)) or {"status": "test_only"})
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--request", str(p), "--request-sha256", m.binding(p)["sha256"], "--execute"])
    m.main();assert len(seen) == 1 and seen[0][0][-1] == "fake" and seen[0][1]["request_binding"] == m.binding(p)
    assert json.loads(capsys.readouterr().out)["status"] == "test_only"


def test_stage_alias_does_not_duplicate_observer_provenance(tmp_path, monkeypatch):
    m, paper, _ = fake_paper(tmp_path, monkeypatch)
    manifest = json.loads((paper / "evidence_manifest.json").read_bytes())
    name = paper.relative_to(m.ROOT).as_posix() + "/planning/static_science.json"
    manifest["sources"][name]["observer_origin_path"] = "frozen/original/decoder.json"
    manifest.pop("receipt_sha256");(paper / "evidence_manifest.json").write_bytes(m.encoded(m.sealed(manifest)))
    snapshot = m.snapshot_paper();out = tmp_path / "proof";out.mkdir()
    stage, proof = m.stage_layer(snapshot, fake_layer(m), out, diagnostic=True)
    staged = json.loads((stage / "evidence_manifest.json").read_bytes())
    assert sum(row.get("observer_origin_path") == "frozen/original/decoder.json" for row in staged["sources"].values()) == 1
    final = m.rebind_final_manifest(stage, proof)
    assert final["sources"][name]["observer_origin_path"] == "frozen/original/decoder.json"
    assert sum(row.get("observer_origin_path") == "frozen/original/decoder.json" for row in final["sources"].values()) == 1


def test_pdf_probe_uses_explicit_system_tools_without_ml_import(tmp_path, monkeypatch):
    m = module();pdf = tmp_path / "main.pdf";pdf.write_bytes(b"synthetic pdf")
    def run(argv, **kwargs):
        assert argv[:2] == ["/usr/bin/python3", "-B"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONPATH"] == "/tmp/rloo-paper-tools/python"
        return types.SimpleNamespace(returncode=0, stdout=b'{"pages":2,"references":[2],"page_captions":[[1,2],[]]}')
    monkeypatch.setattr(m.subprocess, "run", run)
    assert m.pdf_layout(tmp_path)["page_captions"] == [[1, 2], []]


def test_new_unexpected_asset_rejects_closure(tmp_path, monkeypatch):
    m, paper, snapshot = fake_paper(tmp_path, monkeypatch)
    (paper / "user_new_section.tex").write_bytes(b"user text")
    with pytest.raises(ValueError, match="Unexpected"):
        m.verify_retained_files(paper, snapshot, {})


def test_own_SIGTERM_triggers_rollback_and_restores_handler(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "a").write_bytes(b"old")
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match="rolling back"):
        m.transaction(paper, {"a": b"ours"}, {"a": m.identity(b"old")}, tmp_path / "backup",
                      hook=lambda name: os.kill(os.getpid(), signal.SIGTERM))
    assert (paper / "a").read_bytes() == b"old"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_compile_commands_use_same_verified_buffer_bootstrap(tmp_path, monkeypatch):
    m = module();stage = tmp_path / "stage";stage.mkdir();out = tmp_path / "proof";out.mkdir()
    for name in ("compile_paper.py", "verify_artifacts.py"):
        (stage / name).write_bytes(("# synthetic " + name).encode())
    compiler = tmp_path / "tectonic";compiler.write_bytes(b"synthetic executable, not run")
    monkeypatch.setattr(m, "TECTONIC", compiler);monkeypatch.setattr(m, "TECTONIC_SHA", m.sha(compiler.read_bytes()))
    monkeypatch.setattr(m, "COMPILER_SHA", m.sha((stage / "compile_paper.py").read_bytes()))
    monkeypatch.setattr(m, "VERIFIER_SHA", m.sha((stage / "verify_artifacts.py").read_bytes()))
    seen = []
    def run(argv, **kwargs):
        seen.append(argv)
        assert argv[:4] == ["/usr/bin/python3", "-B", "-c", m.BOOT]
        assert argv[5] == m.sha(Path(argv[4]).read_bytes())
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return types.SimpleNamespace(returncode=0, stdout=b"synthetic mock output", stderr=b"")
    monkeypatch.setattr(m.subprocess, "run", run)
    assert len(m.compile_stage(stage, out)) == 2 and len(seen) == 2


def test_transaction_retains_existing_file_permissions(tmp_path):
    m = module();paper = tmp_path / "paper";paper.mkdir();p = paper / "a";p.write_bytes(b"old");p.chmod(0o640)
    m.transaction(paper, {"a": b"new"}, {"a": m.identity(b"old")}, tmp_path / "backup")
    assert p.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("rename_first", [False, True])
def test_rename_interruption_boundary_is_rolled_back(tmp_path, monkeypatch, rename_first):
    m = module();paper = tmp_path / "paper";paper.mkdir();(paper / "main.tex").write_bytes(b"ORIGINAL")
    replace = m.os.replace
    def interrupted(src, dst):
        if rename_first:
            replace(src, dst)
        raise InterruptedError("at the actual rename syscall boundary")
    monkeypatch.setattr(m.os, "replace", interrupted)
    with pytest.raises(InterruptedError, match="syscall boundary"):
        m.transaction(paper, {"main.tex": b"NEW"}, {"main.tex": m.identity(b"ORIGINAL")}, tmp_path / "backup")
    assert (paper / "main.tex").read_bytes() == b"ORIGINAL"
