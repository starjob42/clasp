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

m = load("readonly_remote_chunks", EXP / "remote_chunk_provider_v2.py")
c = load("unchanged_chunk_collector", EXP / "collect_confirmation_round_v1.py")
r = load("readonly_remote_reader2", EXP / "remote_chunk_reader_v2.py")

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

@pytest.mark.parametrize("fault", ["binding_file", "binding_memory", "python", "username", "host", "roots", "fixture", "factory", "overall", "idle"])
def test_changed_binding_or_routing_rejects_before_process_start(tmp_path, fault, monkeypatch):
    provider, folder, binding_path, binding = fixture(tmp_path)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("Changed routing cannot start a process"))
    if fault == "binding_file": binding_path.write_text("changed")
    elif fault == "binding_memory": provider.binding["native_mutations"] = True
    elif fault == "python": provider.python = "/other/python"
    elif fault == "username": provider.username = "other"
    elif fault == "host": provider.host = "other"
    elif fault == "roots": provider.roots = (m.canonical_path("/other"),)
    elif fault == "fixture": provider.fixture = False
    elif fault == "factory": provider.process_factory = lambda *a: pytest.fail("Altered fixture factory cannot run")
    elif fault == "overall": provider.overall_seconds += 1
    elif fault == "idle": provider.idle_seconds += 1
    with pytest.raises(ValueError): provider.fingerprint(str(folder / "data"))
    assert provider.evidence == []

def future_transport(tmp_path):
    """Offline constructor metadata; never a registration or native read."""
    namespace, source, controls = r.confirmation_roots(1)
    binding = {"protocol": m.PROTOCOL, "schema_version": 1,
        "scope": "registered_confirmation_transport_v2", "round_number": 1,
        "native_namespace": namespace, "native_mutations": False,
        "production_round_admission": False, "ssh_host": m.HOST,
        "native_username": "uag72kz", "native_python": m.NATIVE_PYTHON,
        "allowed_roots": [source, *controls.values()],
        "provider_source_sha256": m.__executed_source_sha256__, "remote_program_sha256": m.REMOTE_SOURCE_SHA256,
        "roots_authority": {"native_path": source + "/registered_contract.json", "size_bytes": 100, "sha256": "1" * 64},
        "registration_authority": {"native_path": source + "/independent_prepolicy.json", "size_bytes": 100, "sha256": "2" * 64},
        "source_manifest_sha256": "3" * 64, "contract_sha256": "4" * 64}
    bpath = tmp_path / "binding2.json"; bpath.write_bytes(raw(binding))
    entry = {key: binding[key] for key in ("round_number", "native_namespace", "allowed_roots", "roots_authority",
        "registration_authority", "contract_sha256", "source_manifest_sha256", "provider_source_sha256")}
    entry.update(binding_sha256=m.sha(bpath.read_bytes()), reader_source_sha256=m.REMOTE_SOURCE_SHA256)
    registry = {"protocol": m.REGISTRY_PROTOCOL, "schema_version": 1, "native_mutations": False,
        "production_round_admission": False, "registrations": {entry["binding_sha256"]: entry}}
    rpath = tmp_path / "registry2.json"; rpath.write_bytes(raw(registry))
    return bpath, binding, rpath, registry

def open_future(bpath, rpath):
    return m.RegisteredSSHReadOnlyProvider(bpath, expected_binding_sha256=m.sha(bpath.read_bytes()),
        registry_path=rpath, expected_registry_sha256=m.sha(rpath.read_bytes()))

def test_separate_scope_constructor_uses_closed_registry_without_native_calls(tmp_path, monkeypatch):
    bpath, binding, rpath, registry = future_transport(tmp_path)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("No constructor network calls"))
    provider = open_future(bpath, rpath)
    assert provider.path(binding["allowed_roots"][1] + "/outputs/op.sft/op/sft/terminal/trainer_state.json")
    assert provider.evidence == []
    assert json.loads((EXP / m.DEFAULT_REGISTRY).read_bytes())["registrations"] == {}
    assert c.KNOWN_REGISTRATIONS == {} and c.A.PRODUCTION_ADMISSION_READY is False

@pytest.mark.parametrize("fault", ["factory", "fixture", "both", "overall", "idle"])
def test_native_route_cannot_be_changed_to_local_fixture_after_construction(tmp_path, monkeypatch, fault):
    bpath, binding, rpath, registry = future_transport(tmp_path)
    provider = open_future(bpath, rpath)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("Changed production route cannot run"))
    if fault in ("factory", "both"):
        provider.process_factory = lambda *a: pytest.fail("Forged native callback cannot run")
    if fault in ("fixture", "both"): provider.fixture = True
    if fault == "overall": provider.overall_seconds = 1801
    if fault == "idle": provider.idle_seconds = 121
    with pytest.raises(ValueError): provider.fingerprint(binding["roots_authority"]["native_path"])
    assert provider.evidence == []

@pytest.mark.parametrize("fault", ["no_registry", "empty_registry", "binding", "provider", "reader", "round", "namespace",
    "roots", "contract", "source", "authority", "receipt", "synthetic", "mutations", "admission", "registry_hash"])
def test_external_registry_identity_cannot_be_replaced_by_flags(tmp_path, fault):
    bpath, binding, rpath, registry = future_transport(tmp_path)
    entry = next(iter(registry["registrations"].values()))
    if fault == "no_registry":
        with pytest.raises(ValueError): m.RegisteredSSHReadOnlyProvider(bpath, expected_binding_sha256=m.sha(bpath.read_bytes()))
        return
    if fault == "empty_registry": registry["registrations"] = {}
    elif fault == "binding": entry["binding_sha256"] = "0" * 64
    elif fault == "provider": entry["provider_source_sha256"] = "0" * 64
    elif fault == "reader": entry["reader_source_sha256"] = "0" * 64
    elif fault == "round": entry["round_number"] = 2
    elif fault == "namespace": entry["native_namespace"] += "other"
    elif fault == "roots": entry["allowed_roots"] = []
    elif fault == "contract": entry["contract_sha256"] = "0" * 64
    elif fault == "source": entry["source_manifest_sha256"] = "0" * 64
    elif fault == "authority": entry["roots_authority"] = {}
    elif fault == "receipt": entry["registration_authority"] = {}
    elif fault == "synthetic": registry["synthetic_fixture"] = True
    elif fault == "mutations": registry["native_mutations"] = True
    elif fault == "admission": registry["production_round_admission"] = True
    rpath.write_bytes(raw(registry))
    with pytest.raises(ValueError):
        m.RegisteredSSHReadOnlyProvider(bpath, expected_binding_sha256=m.sha(bpath.read_bytes()), registry_path=rpath,
            expected_registry_sha256="0" * 64 if fault == "registry_hash" else m.sha(rpath.read_bytes()))

@pytest.mark.parametrize("fault", ["old_namespace", "arbitrary_sibling", "round_bool", "round_zero", "round_high", "missing_draw",
    "reused_draw", "escaping_receipt", "escaping_authority", "host", "python", "owner", "callback", "old_scope"])
def test_native_binding_scope_is_deterministic_even_with_resigned_metadata(tmp_path, fault):
    bpath, binding, rpath, registry = future_transport(tmp_path)
    if fault == "old_namespace": binding["native_namespace"] = m.PROJECT_ROOT + "/rloo_sft_anchor_20261013"
    elif fault == "arbitrary_sibling": binding["native_namespace"] = m.PROJECT_ROOT + "/arbitrary"
    elif fault == "round_bool": binding["round_number"] = True
    elif fault == "round_zero": binding["round_number"] = 0
    elif fault == "round_high": binding["round_number"] = 101
    elif fault == "missing_draw": binding["allowed_roots"].pop()
    elif fault == "reused_draw": binding["allowed_roots"][-1] = binding["allowed_roots"][-2]
    elif fault == "escaping_receipt": binding["registration_authority"]["native_path"] = "/etc/passwd"
    elif fault == "escaping_authority": binding["roots_authority"]["native_path"] = "/etc/passwd"
    elif fault == "host": binding["ssh_host"] = "other"
    elif fault == "python": binding["native_python"] = "/bin/python"
    elif fault == "owner": binding["native_username"] = "other"
    elif fault == "old_scope": binding["scope"] = "registered_confirmation_transport"
    bpath.write_bytes(raw(binding))
    old_entry = next(iter(registry["registrations"].values()))
    for key in list(old_entry):
        if key in binding: old_entry[key] = binding[key]
    old_entry["binding_sha256"] = m.sha(bpath.read_bytes())
    registry["registrations"] = {old_entry["binding_sha256"]: old_entry}; rpath.write_bytes(raw(registry))
    with pytest.raises(ValueError):
        m.RegisteredSSHReadOnlyProvider(bpath, expected_binding_sha256=m.sha(bpath.read_bytes()), registry_path=rpath,
            expected_registry_sha256=m.sha(rpath.read_bytes()), process_factory=(lambda *a: None) if fault == "callback" else None)

def authority_objects():
    namespace, source, controls = r.confirmation_roots(1)
    value = {"protocol": r.FORMAL_CONTRACT_PROTOCOL, "round_number": 1,
        "native_source_root": source, "native_control_roots": controls,
        "source_manifest_binding": {"sha256": "3" * 64}}
    value["contract_sha256"] = r.canonical(value)
    proof = {"protocol": r.INDEPENDENT_PROTOCOL, "independent_read_only_custody": True,
        "confirmation_policy_outcomes_read": False, "registered_contract_sha256": value["contract_sha256"],
        "round_number": 1, "native_source_root": source, "native_control_roots": controls,
        "source_manifest_sha256": "3" * 64}
    request = {"scope": "registered_confirmation_transport_v2", "native_namespace": namespace,
        "round_number": 1, "contract_sha256": value["contract_sha256"], "source_manifest_sha256": "3" * 64}
    return request, [r.path(source), *map(r.path, controls.values())], value, proof

def test_remote_replays_contract_and_independent_receipt_before_stream():
    request, roots, value, proof = authority_objects()
    r.validate_confirmation_authority(request, roots, value, proof)

@pytest.mark.parametrize("fault", ["seal", "round", "old_source", "draw", "receipt_protocol", "self_reported",
    "outcomes_read", "receipt_contract", "receipt_source", "receipt_round", "receipt_roots", "fixture", "extra_root"])
def test_remote_authority_rejects_drift_and_producer_only_flags(fault):
    request, roots, value, proof = authority_objects()
    if fault == "seal": value["unexpected"] = True
    elif fault == "round": request["round_number"] = 2
    elif fault == "old_source": value["native_source_root"] = m.PROJECT_ROOT + "/rloo_sft_anchor_20261013/source_v2"
    elif fault == "draw": value["native_control_roots"] = {}
    elif fault == "receipt_protocol": proof["protocol"] = "producer_telemetry"
    elif fault == "self_reported": proof["independent_read_only_custody"] = False
    elif fault == "outcomes_read": proof["confirmation_policy_outcomes_read"] = True
    elif fault == "receipt_contract": proof["registered_contract_sha256"] = "0" * 64
    elif fault == "receipt_source": proof["source_manifest_sha256"] = "0" * 64
    elif fault == "receipt_round": proof["round_number"] = 2
    elif fault == "receipt_roots": proof["native_control_roots"] = {}
    elif fault == "fixture": proof["synthetic_fixture"] = True
    elif fault == "extra_root": roots.append(r.path(m.PROJECT_ROOT + "/other"))
    with pytest.raises(ValueError): r.validate_confirmation_authority(request, roots, value, proof)

