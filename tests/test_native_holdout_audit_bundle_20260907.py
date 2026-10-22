"""Management-only provenance guards for the corpus-local holdout audit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).resolve().parents[1] / (
        "analysis_outputs/grpo_incumbent_20260907_v1/run_native_holdout_audit_v1.py"
    )
    spec = importlib.util.spec_from_file_location("test_native_holdout_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, wrapper, monkeypatch):
    required = {wrapper.BUILDER, "run_native_holdout_audit_v1.py", "audit_native_holdout_v1.slurm",
                "forbidden_sources.native.json", "data/plan.json", "data/build_manifest.json",
                "data/build_declaration.json", *wrapper.MODULE_PATHS.values()}
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    manifest = {"protocol": wrapper.PROTOCOL, "runtime_executable": "/native/python",
                "files": {name: wrapper.identity(tmp_path / name) for name in sorted(required)}}
    monkeypatch.setattr(wrapper, "sys", SimpleNamespace(
        executable="/native/python", version_info=(3, 10, 8),
        flags=SimpleNamespace(isolated=1, no_site=1, dont_write_bytecode=1)))
    monkeypatch.setattr(wrapper, "__file__", str(tmp_path / "run_native_holdout_audit_v1.py"))
    return tmp_path, manifest


def write_manifest(wrapper, bundle):
    root, manifest = bundle
    path = root / "bundle_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return wrapper.identity(path)["sha256"]


def test_exact_bundle_and_runtime_are_admitted(wrapper, bundle):
    digest = write_manifest(wrapper, bundle)
    assert wrapper.verified_manifest(bundle[0], digest) == bundle[1]


def test_changed_manifest_bytes_are_rejected(wrapper, bundle):
    digest = write_manifest(wrapper, bundle)
    path = bundle[0] / "bundle_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest differs"):
        wrapper.verified_manifest(bundle[0], digest)


def test_changed_live_helper_is_rejected(wrapper, bundle):
    digest = write_manifest(wrapper, bundle)
    (bundle[0] / "source/src/clasp/rewards/co_tasks.py").write_text("changed\n")
    with pytest.raises(ValueError, match="input differs"):
        wrapper.verified_manifest(bundle[0], digest)


def test_missing_import_closure_is_rejected(wrapper, bundle):
    del bundle[1]["files"]["source/src/clasp/runtime.py"]
    digest = write_manifest(wrapper, bundle)
    with pytest.raises(ValueError, match="inventory is incomplete"):
        wrapper.verified_manifest(bundle[0], digest)


@pytest.mark.parametrize("relative", ["/tmp/outside.py", "../outside.py"])
def test_unsafe_inventory_paths_are_rejected(wrapper, bundle, relative):
    bundle[1]["files"][relative] = {"sha256": "0" * 64, "size_bytes": 1}
    digest = write_manifest(wrapper, bundle)
    with pytest.raises(ValueError, match="unsafe native bundle path"):
        wrapper.verified_manifest(bundle[0], digest)


def test_same_byte_symlink_is_rejected(wrapper, bundle):
    target = bundle[0] / "source/src/clasp/runtime.py"
    other = bundle[0] / "other.py"
    other.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(other)
    digest = write_manifest(wrapper, bundle)
    with pytest.raises(ValueError, match="input differs"):
        wrapper.verified_manifest(bundle[0], digest)


def test_wrong_runtime_is_rejected(wrapper, bundle, monkeypatch):
    digest = write_manifest(wrapper, bundle)
    monkeypatch.setattr(wrapper.sys, "version_info", (3, 11, 2))
    with pytest.raises(ValueError, match="Python version"):
        wrapper.verified_manifest(bundle[0], digest)


@pytest.mark.parametrize("flag", ["isolated", "no_site", "dont_write_bytecode"])
def test_runtime_isolation_flags_are_required(wrapper, bundle, monkeypatch, flag):
    digest = write_manifest(wrapper, bundle)
    monkeypatch.setattr(wrapper.sys.flags, flag, 0)
    with pytest.raises(ValueError, match="Python flags"):
        wrapper.verified_manifest(bundle[0], digest)


def test_login_node_cannot_start_corpus_audit(wrapper, bundle, monkeypatch):
    digest = write_manifest(wrapper, bundle)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="allocated Slurm job"):
        wrapper.run(bundle[0], digest)
    assert not (bundle[0] / "execution").exists()


def test_receipt_writes_are_create_only(wrapper, tmp_path):
    path = tmp_path / "receipt.json"
    wrapper.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        wrapper.write_new(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_loaded_module_must_be_exact_bundled_file(wrapper, bundle, monkeypatch):
    root, manifest = bundle
    modules = {name: SimpleNamespace(__file__=str(root / relative))
               for name, relative in wrapper.MODULE_PATHS.items()}
    monkeypatch.setattr(wrapper.sys, "modules", modules, raising=False)
    assert len(wrapper.loaded_modules(root, manifest)) == len(modules)
    modules["clasp.runtime"] = SimpleNamespace(__file__="/site-packages/clasp/runtime.py")
    with pytest.raises(ValueError, match="outside frozen bundle"):
        wrapper.loaded_modules(root, manifest)
