"""Read-only transport mechanics; no native round/cohort/registry/job creation."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"

def load(name, path):
    import types
    body = path.read_bytes(); value = types.ModuleType(name); value.__file__ = str(path)
    value.__executed_source_sha256__ = hashlib.sha256(body).hexdigest()
    exec(compile(body, str(path), "exec"), value.__dict__); return value

m = load("readonly_remote_chunks", EXP / "remote_chunk_provider_v1.py")
c = load("unchanged_chunk_collector", EXP / "collect_confirmation_round_v1.py")

def raw(value): return (json.dumps(value, sort_keys=True) + "\n").encode()

def fixture_username():
    return "synthetic-uid:" + str(os.getuid())

def fixture(tmp_path, *, factory=None, idle=2):
    folder = tmp_path / "owned"; folder.mkdir()
    anchor = folder / "root_authority.json"
    anchor.write_bytes(raw({"synthetic_fixture": True, "allowed_roots": [str(folder)]}))
    binding = {"protocol": m.PROTOCOL, "schema_version": 1, "scope": "synthetic_fixture", "synthetic_fixture": True,
        "ssh_host": "local-fixture", "native_python": sys.executable, "native_username": fixture_username(),
        "allowed_roots": [str(folder)], "roots_authority": {"native_path": str(anchor), "size_bytes": anchor.stat().st_size,
            "sha256": m.sha(anchor.read_bytes())}, "remote_program_sha256": m.REMOTE_SOURCE_SHA256,
        "provider_source_sha256": m.__executed_source_sha256__,
        "native_mutations": False, "production_round_admission": False, "overall_timeout_seconds": 20, "idle_timeout_seconds": idle}
    path = tmp_path / "binding.json"; path.write_bytes(raw(binding))
    provider = m.RegisteredSSHReadOnlyProvider(path, expected_binding_sha256=m.sha(path.read_bytes()), process_factory=factory)
    return provider, folder, path, binding

def process(code):
    return subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def fp(size): return {"device": 1, "inode": 2, "size_bytes": size, "mtime_ns": 3, "ctime_ns": 4}

def frames(request, data=b"abcdef"):
    return [{"type": "header", "protocol": m.WIRE_PROTOCOL, "source_sha256": m.REMOTE_SOURCE_SHA256,
        "action": request["action"], "path": request["path"], "reader_hostname": "synthetic_host", "reader_uid": os.getuid(),
        "reader_username": request["native_username"], "native_mutations": False},
        {"type": "fingerprint", "fingerprint": fp(len(data))},
        {"type": "chunk", "index": 0, "offset": 0, "size_bytes": len(data), "sha256": m.sha(data),
            "content_base64": base64.b64encode(data).decode()},
        {"type": "complete", "action": "stream", "fingerprint": fp(len(data)), "size_bytes": len(data), "chunks": 1, "sha256": m.sha(data)}]

def framed_factory(change, *, code=0):
    def factory(argv, payload):
        request = json.loads(payload); rows = frames(request); rows = change(rows) or rows
        text = b"".join(raw(row) if not isinstance(row, bytes) else row for row in rows)
        return process("import sys;sys.stdout.buffer.write(" + repr(text) + ");sys.stdout.flush();sys.exit(" + repr(code) + ")")
    return factory

def test_actual_saved_cpu_producer_actor_chunk_identity_and_safe_tree(tmp_path):
    provider, folder, _, _ = fixture(tmp_path)
    original = EXP / "validation/v2_peft_cpu_ddp4_v1/op/rloo_cardinal_beta005/terminal/adapter_model.safetensors"
    file = folder / "terminal/adapter_model.safetensors"; file.parent.mkdir(); shutil.copyfile(original, file)
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    store = c.ChunkStore(tmp_path / "chunks", chunk_bytes=113)
    binding = store.ingest(provider, str(file), expected={"sha256": original_hash, "size_bytes": original.stat().st_size})
    reader = c.ChunkReader(store.root)
    assert reader.native_identity(binding)["sha256"] == original_hash
    assert b"".join(reader.iter_bytes(binding)) == original.read_bytes()
    assert [row["action"] for row in provider.evidence] == ["fingerprint", "stream", "fingerprint"]
    assert all(row["transport_complete"] and row["native_mutations"] is False for row in provider.evidence)
    assert provider.tree_entries(str(file.parent)) == [(file.name, provider.fingerprint(str(file)))]
    assert c.KNOWN_REGISTRATIONS == {} and c.A.PRODUCTION_ADMISSION_READY is False

def test_whole_artifact_over64mb_has_one_stream_process_and_exact_full_sha(tmp_path):
    provider, folder, _, _ = fixture(tmp_path, idle=5)
    file = folder / "large.bin"; whole = hashlib.sha256()
    block = hashlib.sha256(b"actual bounded chunk").digest() * (1024 * 1024 // 32)
    with file.open("wb") as output:
        for _ in range(65): output.write(block); whole.update(block)
        output.write(b"final123"); whole.update(b"final123")
    store = c.ChunkStore(tmp_path / "chunks", chunk_bytes=1024 * 1024)
    binding = store.ingest(provider, str(file), expected={"sha256": whole.hexdigest(), "size_bytes": 65 * 1024 * 1024 + 8})
    reader = c.ChunkReader(store.root); manifest = reader.manifest(binding)
    assert manifest["size_bytes"] > 64 * 1024 * 1024 and len(manifest["chunks"]) == 66
    assert sum(row["action"] == "stream" for row in provider.evidence) == 1
    assert sum(len(part) for part in reader.iter_bytes(binding)) == manifest["size_bytes"]

@pytest.mark.parametrize("fault", ["source", "owner", "path", "action", "mutations", "fp_type", "zero_chunk", "oversized", "offset", "index", "base64", "chunk_hash", "full_hash", "footer_fingerprint", "footer_bool", "missing_footer", "truncated", "malformed", "duplicate_key", "extra_record", "exit_network"])
def test_malformed_changed_corrupt_truncated_and_network_streams_reject(tmp_path, fault):
    def change(rows):
        if fault == "source": rows[0]["source_sha256"] = "0" * 64
        elif fault == "owner": rows[0]["reader_username"] = "other"
        elif fault == "path": rows[0]["path"] += "other"
        elif fault == "action": rows[0]["action"] = "write"
        elif fault == "mutations": rows[0]["native_mutations"] = True
        elif fault == "fp_type": rows[1]["fingerprint"]["size_bytes"] = 6.0
        elif fault == "zero_chunk": rows[2]["size_bytes"] = 0
        elif fault == "oversized": rows[2]["size_bytes"] = 7
        elif fault == "offset": rows[2]["offset"] = 1
        elif fault == "index": rows[2]["index"] = 1
        elif fault == "base64": rows[2]["content_base64"] = "@@@"
        elif fault == "chunk_hash": rows[2]["sha256"] = "0" * 64
        elif fault == "full_hash": rows[-1]["sha256"] = "0" * 64
        elif fault == "footer_fingerprint": rows[-1]["fingerprint"]["inode"] += 1
        elif fault == "footer_bool": rows[-1]["chunks"] = True
        elif fault == "missing_footer": rows.pop()
        elif fault == "truncated": rows[-1] = b'{"incomplete":'
        elif fault == "malformed": rows[-1] = b"not json\n"
        elif fault == "duplicate_key": rows[-1] = b'{"type":"complete","type":"complete"}\n'
        elif fault == "extra_record": rows.append({"unexpected": True})
        return rows
    provider, folder, _, _ = fixture(tmp_path, factory=framed_factory(change, code=255 if fault == "exit_network" else 0))
    with pytest.raises((ValueError, m.RemoteTransportError)):
        list(provider.iter_bytes(str(folder / "fake.bin"), 6))
    assert not provider.evidence[-1]["transport_complete"] and not provider.evidence[-1]["scheduler_state_inferred"]
    if fault == "exit_network": assert provider.evidence[-1]["returncode"] == 255

@pytest.mark.parametrize("fault", ["host", "python", "namespace", "root", "native_callback", "binding_sha", "remote_source", "provider_source", "overlap", "timeouts"])
def test_transport_configuration_cannot_widen_native_scope(tmp_path, fault):
    _, folder, path, value = fixture(tmp_path)
    value.update(scope="existing_development_transport", ssh_host=m.HOST, native_python=m.NATIVE_PYTHON,
        native_username="uag72kz", native_namespace=m.NAMESPACE, allowed_roots=[m.NAMESPACE + "/source_v2", m.NAMESPACE + "/control_v2"],
        roots_authority={"native_path": m.NAMESPACE + "/control_v2/launch_manifest.json", "sha256": m.LAUNCH_SHA256, "size_bytes": 6109})
    value.pop("synthetic_fixture")
    callback = None
    if fault == "host": value["ssh_host"] = "arbitrary.example"
    elif fault == "python": value["native_python"] = "/bin/python"
    elif fault == "namespace": value["native_namespace"] = "/etc"
    elif fault == "root": value["allowed_roots"] = ["/etc"]
    elif fault == "native_callback": callback = lambda *args: None
    elif fault == "remote_source": value["remote_program_sha256"] = "0" * 64
    elif fault == "provider_source": value["provider_source_sha256"] = "0" * 64
    elif fault == "overlap": value["allowed_roots"][1] = value["allowed_roots"][0] + "/child"
    elif fault == "timeouts": value["overall_timeout_seconds"] = 0
    path.write_bytes(raw(value)); expected = "0" * 64 if fault == "binding_sha" else m.sha(path.read_bytes())
    with pytest.raises(ValueError): m.RegisteredSSHReadOnlyProvider(path, expected_binding_sha256=expected, process_factory=callback)

@pytest.mark.parametrize("fault", ["final_symlink", "ancestor_symlink", "unscoped", "parent_escape", "double_slash", "anchor_changed", "fifo"])
def test_actual_remote_program_rejects_symlinks_unscoped_and_nonregular_paths(tmp_path, fault):
    provider, folder, _, _ = fixture(tmp_path)
    other = tmp_path / "outside"; other.mkdir(); other.joinpath("secret").write_text("outside")
    target = folder / "data"; target.write_text("owned")
    if fault == "final_symlink": target.unlink(); target.symlink_to(other / "secret")
    elif fault == "ancestor_symlink": target = folder / "alias/secret"; folder.joinpath("alias").symlink_to(other, target_is_directory=True)
    elif fault == "unscoped": target = other / "secret"
    elif fault == "parent_escape": target = str(folder) + "/../outside/secret"
    elif fault == "double_slash": target = str(folder) + "//data"
    elif fault == "anchor_changed": folder.joinpath("root_authority.json").write_text("changed")
    else: target.unlink(); os.mkfifo(target)
    with pytest.raises((ValueError, m.RemoteTransportError)):
        provider.fingerprint(str(target))

def test_explicit_idle_timeout_is_transport_only(tmp_path):
    provider, folder, _, _ = fixture(tmp_path, factory=lambda argv, payload: process("import time;time.sleep(4)"), idle=1)
    with pytest.raises(m.RemoteTransportError, match="timeout") as caught:
        provider.fingerprint(str(folder / "fake.bin"))
    assert caught.value.phase == "timeout" and not provider.evidence[-1]["scheduler_state_inferred"]

def test_mutation_during_actual_stream_cannot_commit_chunk_manifest(tmp_path):
    provider, folder, _, _ = fixture(tmp_path)
    file = folder / "changing.bin"; file.write_bytes(b"a" * (8 * 1024 * 1024))
    original = provider.iter_bytes
    def mutating(path, chunk_bytes):
        for index, block in enumerate(original(path, chunk_bytes)):
            if index == 0:
                with file.open("r+b") as stream: stream.write(b"x")
            yield block
    provider.iter_bytes = mutating
    store = c.ChunkStore(tmp_path / "chunks")
    with pytest.raises((ValueError, m.RemoteTransportError)): store.ingest(provider, str(file))
    assert not list((store.root / "receipts").glob("*.json")) if (store.root / "receipts").exists() else True

def test_changed_file_between_stat_and_stream_rejects_before_bytes(tmp_path):
    provider, folder, _, _ = fixture(tmp_path)
    file = folder / "data"; file.write_text("before"); provider.fingerprint(str(file)); file.write_text("after")
    with pytest.raises(ValueError, match="between fingerprint"):
        list(provider.iter_bytes(str(file), 1024))

def test_empty_regular_file_and_empty_safe_inventory_are_supported(tmp_path):
    provider, folder, _, _ = fixture(tmp_path)
    empty = folder / "empty"; empty.write_bytes(b"")
    store = c.ChunkStore(tmp_path / "chunks")
    receipt = store.ingest(provider, str(empty), expected={"sha256": m.sha(b""), "size_bytes": 0})
    assert c.ChunkReader(store.root).manifest(receipt)["chunks"] == []
    directory = folder / "empty_tree"; directory.mkdir()
    assert provider.tree_entries(str(directory)) == []

def test_oversized_record_is_bounded_before_json_materialization(tmp_path):
    factory = framed_factory(lambda rows: [b"a" * (m.METADATA_LINE_LIMIT + 1) + b"\n"])
    provider, folder, _, _ = fixture(tmp_path, factory=factory)
    with pytest.raises(m.RemoteTransportError) as caught:
        provider.fingerprint(str(folder / "fake"))
    assert caught.value.phase == "record_limit"

def test_launch_oserror_is_explicit_transport_failure(tmp_path):
    def failed(argv, payload): raise OSError("synthetic network launch failure")
    provider, folder, _, _ = fixture(tmp_path, factory=failed)
    with pytest.raises(m.RemoteTransportError) as caught:
        provider.fingerprint(str(folder / "fake"))
    assert caught.value.phase == "launch" and caught.value.returncode is None

def test_safe_inventory_rejects_linked_tree_member(tmp_path):
    provider, folder, _, _ = fixture(tmp_path)
    tree = folder / "tree"; tree.mkdir(); tree.joinpath("linked").symlink_to(folder / "root_authority.json")
    with pytest.raises(m.RemoteTransportError): provider.tree_entries(str(tree))

def test_formal_transport_boolean_cannot_replace_bound_prepolicy_registration(tmp_path):
    _, folder, path, value = fixture(tmp_path)
    value.update(scope="registered_confirmation_transport", ssh_host=m.HOST, native_python=m.NATIVE_PYTHON,
        native_username="uag72kz", native_namespace=m.NAMESPACE,
        allowed_roots=[m.NAMESPACE + "/future_source"] + [m.NAMESPACE + "/future_control" + str(i) for i in range(20)],
        roots_authority={"native_path": m.NAMESPACE + "/future_source/contract.json", "size_bytes": 1, "sha256": "1" * 64},
        independent_prepolicy_registration=True, registration_sha256="2" * 64)
    value.pop("synthetic_fixture"); path.write_bytes(raw(value))
    with pytest.raises(ValueError, match="boolean cannot create"):
        m.RegisteredSSHReadOnlyProvider(path, expected_binding_sha256=m.sha(path.read_bytes()))

