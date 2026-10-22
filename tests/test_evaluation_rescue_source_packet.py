"""Wholly synthetic local source closure/import guards; never native admission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


m = load("packet_builder", "package_evaluation_rescue_source_v1.py")
b = load("packet_bootstrap", "evaluation_rescue_bootstrap_v1.py")


def write_manifest(tmp_path):
    root = tmp_path / "files"; root.mkdir()
    content = {"src/clasp/__init__.py": b"", "src/clasp/foo.py": b"value=5\n",
               "entry.py": b'import clasp.foo,json\nprint(json.dumps({"value":clasp.foo.value,"executed":clasp.foo.__executed_source_sha256__}))\n'}
    rows = []
    for path, raw in content.items():
        target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        rows.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
    value = {"schema_version": 1, "protocol": b.PROTOCOL, "files": rows, "allowed_entrypoints": ["entry.py"],
             "production_admitted": False, "synthetic_fixture": True}
    value["semantic_sha256"] = m.sha(m.encoded(value))
    path = tmp_path / "source_manifest.json"; path.write_bytes(m.encoded(value))
    return path, m.sha(path.read_bytes()), root


def test_isolated_bootstrap_imports_only_verified_source_not_external_pythonpath(tmp_path):
    path, sha, _ = write_manifest(tmp_path)
    completed = subprocess.run([sys.executable, "-I", "-B", str(EXP / "evaluation_rescue_bootstrap_v1.py"),
        "--manifest", str(path), "--manifest-sha256", sha, "--entrypoint", "entry.py"],
        check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout)
    assert value == {"value": 5, "executed": m.sha(b"value=5\n")}


@pytest.mark.parametrize("change", ["pyc", "unbound_python", "linked", "mutated", "manifest"])
def test_stale_cache_unbound_code_and_source_drift_rejected(tmp_path, change):
    path, sha, root = write_manifest(tmp_path)
    if change == "pyc": (root / "stale.pyc").write_bytes(b"not trusted bytecode")
    if change == "unbound_python": (root / "new.py").write_bytes(b"value=7\n")
    if change == "linked": (root / "alias.py").symlink_to(root / "entry.py")
    if change == "mutated": (root / "src/clasp/foo.py").write_bytes(b"value=7\n")
    if change == "manifest": path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError): b.verify_packet(path, sha)


def test_import_loader_executes_the_verified_buffer_and_records_its_identity(tmp_path):
    path, sha, root = write_manifest(tmp_path)
    _, _, indexed = b.verify_packet(path, sha)
    source = root / "src/clasp/foo.py"; source.write_bytes(b"value=7\n")
    spec = b.BoundClaspFinder(root, indexed).find_spec("clasp.foo")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.value == 5 and module.__executed_source_sha256__ == m.sha(b"value=5\n")


def fixture_repository(tmp_path, monkeypatch):
    root = tmp_path / "repository"; root.mkdir()
    original = root / "original"; original.mkdir()
    rows = []
    for index in range(121):
        name = f"src/original{index}.py"; raw = f"# unchanged original{index}\n".encode()
        path = original / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        rows.append({"path": name, "sha256": m.sha(raw), "size_bytes": len(raw)})
    lineage = m.encoded({"files": rows, "synthetic_fixture": True})
    (original / "source_manifest.json").write_bytes(lineage); monkeypatch.setattr(m, "ORIGINAL_SHA", m.sha(lineage))
    for name in m.REQUIRED:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"# synthetic source\n")
    registry = root / m.PREFIX / "planning/evaluation_rescue_source_registry_v1.json"
    registry.write_bytes(m.encoded({"schema_version": 1, "protocol": "sft-anchor-terminal-evaluation-operator-v1-source-registry", "entries": []}))
    sources = [{"path": name, "sha256": m.sha((root / name).read_bytes()), "size_bytes": (root / name).stat().st_size}
               for name in m.CORE_VALIDATION]
    receipt = {"status": "passed_source_only_fixture", "tests": 3, "failures": 0, "errors": 0, "skips": 0,
               "production_admitted": False, "native_queries_or_mutations": False, "sources": sources, "synthetic_fixture": True}
    receipt_path = root / "receipt.json"; receipt_path.write_bytes(m.encoded(receipt))
    receipts = [{"path": "receipt.json", "sha256": m.sha(receipt_path.read_bytes())}]
    return root, original, receipts


def test_packet_preserves_all121_lineage_and_exclusive_output(tmp_path, monkeypatch):
    root, original, receipts = fixture_repository(tmp_path, monkeypatch)
    output = tmp_path / "closed"
    value = m.close_packet(output, receipts, root=root, original_source=original)
    assert not value["production_admitted"] and not value["native_queries_or_mutations"]
    manifest, _, _ = b.verify_packet(output / "source_manifest.json", value["source_manifest_sha256"])
    assert manifest["original_source2_files_preserved"] == 121
    assert (output / "files" / (m.PREFIX + "original_source2_manifest.json")).read_bytes() == (original / "source_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="exclusive"): m.close_packet(output, receipts, root=root, original_source=original)


@pytest.mark.parametrize("change", ["original", "tested_source", "receipt", "live_registry", "missing_core"])
def test_only_current_tested_source_with_empty_registry_can_be_closed(tmp_path, monkeypatch, change):
    root, original, receipts = fixture_repository(tmp_path, monkeypatch)
    if change == "original": (original / "src/original100.py").write_bytes(b"changed")
    if change == "tested_source": (root / sorted(m.CORE_VALIDATION)[0]).write_bytes(b"changed")
    if change == "receipt": receipts[0]["sha256"] = "0" * 64
    if change == "live_registry":
        path = root / m.PREFIX / "planning/evaluation_rescue_source_registry_v1.json"
        path.write_bytes(m.encoded({"schema_version": 1, "protocol": "sft-anchor-terminal-evaluation-operator-v1-source-registry", "entries": [{}]}))
    if change == "missing_core":
        path = root / "receipt.json"; value = json.loads(path.read_bytes()); value["sources"].pop()
        path.write_bytes(m.encoded(value)); receipts[0]["sha256"] = m.sha(path.read_bytes())
    with pytest.raises(ValueError): m.close_packet(tmp_path / "closed", receipts, root=root, original_source=original)
