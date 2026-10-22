"""Actual maintained Ed25519 implementation; every key/packet is fixture-only."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import types
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/rloo_sft_anchor_20261013/verify_registration_promotion_v1.py"
RAW = PATH.read_bytes()
m = types.ModuleType("fixture_promotion_verifier"); m.__file__ = str(PATH)
exec(compile(RAW, str(PATH), "exec"), m.__dict__)

def binding(path, raw):
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}

@pytest.fixture(scope="module")
def signed_packet():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    keys = {"fixture-independent-custodian": {"purpose": "independent_prepolicy_registration", "scope": "fixture_only",
        "public_key_base64": base64.b64encode(public).decode(), "crypto_environment": m.crypto_environment()}}
    # The key-bearing verifier freezes FIRST. Neither it nor the core embeds
    # registry/manifest hashes. External signed promotion is emitted LAST.
    key_source = RAW.replace(b"TRUSTED_PUBLIC_KEYS = {}", b"TRUSTED_PUBLIC_KEYS = " + repr(keys).encode())
    core = (ROOT / "experiments/rloo_sft_anchor_20261013/analyze_chunked_confirmation_v1.py").read_bytes()
    core = core.replace(hashlib.sha256(RAW).hexdigest().encode(), hashlib.sha256(key_source).hexdigest().encode())
    manifest = {"files": [binding("fixed_core.py", core), binding("fixed_key_verifier.py", key_source)], "synthetic_fixture": True}
    manifest_raw = m.encoded(manifest); manifest_binding = binding("source_manifest.json", manifest_raw)
    registry = {"entries": [{"contract_sha256": "c" * 64, "consumer_core_source_sha256": hashlib.sha256(core).hexdigest(),
        "effective_source_manifest_binding": manifest_binding, "verifier_source_sha256": hashlib.sha256(key_source).hexdigest()}],
        "synthetic_fixture": True}
    registry_binding = binding("registry.json", m.encoded(registry))
    expected = {"consumer_protocol": "fixture-realistic-closed-packet", "consumer_core_source_sha256": hashlib.sha256(core).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(key_source).hexdigest(), "registry_binding": registry_binding,
        "registered_contract_sha256": "c" * 64, "effective_source_manifest_binding": manifest_binding,
        "promotion_frozen_before_confirmation_policy_calls": True}
    payload = {"protocol": m.PROTOCOL, "schema_version": 1, "synthetic_fixture": True,
        "performance_admission": False, **expected}
    certificate = {"payload": payload, "signer_key_id": "fixture-independent-custodian",
        "signature_base64": base64.b64encode(private.sign(m.encoded(payload))).decode()}
    return keys, certificate, expected, key_source, core, manifest_raw, registry

def test_actual_ed25519_acyclic_closed_whole_packet_and_stable_core(signed_packet):
    keys, certificate, expected, key_source, core, manifest_raw, registry = signed_packet
    proof = m.verify_promotion(certificate, expected, synthetic_fixture=True, fixture_public_keys=keys)
    assert proof["synthetic_fixture"] and not proof["production_registration_authenticated"]
    manifest = json.loads(manifest_raw)
    assert manifest["files"][0]["sha256"] == hashlib.sha256(core).hexdigest()
    assert manifest["files"][1]["sha256"] == hashlib.sha256(key_source).hexdigest()
    assert hashlib.sha256(key_source).hexdigest().encode() in core
    assert registry["entries"][0]["effective_source_manifest_binding"]["sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert hashlib.sha256(core).hexdigest().encode() not in key_source
    assert expected["registry_binding"]["sha256"].encode() not in core + key_source
    assert m.TRUSTED_PUBLIC_KEYS == {}
    keyed = types.ModuleType("fixture_independently_frozen_key_source")
    exec(compile(key_source, "fixture-frozen-verifier.py", "exec"), keyed.__dict__)
    with pytest.raises(ValueError, match="Trusted signer"):
        changed = copy.deepcopy(certificate); changed["payload"].pop("synthetic_fixture"); changed["payload"]["performance_admission"] = True
        keyed.verify_promotion(changed, expected)

@pytest.mark.parametrize("fault", ["signature", "payload", "unknown_signer", "key", "crypto_version", "crypto_file", "purpose", "registry", "core", "manifest", "fixture_admission"])
def test_signature_and_external_identity_guards(signed_packet, fault):
    keys, cert, expected, *_ = signed_packet
    keys, cert, expected = copy.deepcopy(keys), copy.deepcopy(cert), copy.deepcopy(expected)
    if fault == "signature": cert["signature_base64"] = base64.b64encode(b"0" * 64).decode()
    elif fault == "payload": cert["payload"]["extra"] = "forged"
    elif fault == "unknown_signer": cert["signer_key_id"] = "producer"
    elif fault == "key": keys[cert["signer_key_id"]]["public_key_base64"] = base64.b64encode(b"0" * 32).decode()
    elif fault == "crypto_version": keys[cert["signer_key_id"]]["crypto_environment"]["cryptography_version"] = "forged"
    elif fault == "crypto_file": keys[cert["signer_key_id"]]["crypto_environment"]["implementation_files"][0]["sha256"] = "0" * 64
    elif fault == "purpose": keys[cert["signer_key_id"]]["purpose"] = "producer"
    elif fault == "registry": expected["registry_binding"]["sha256"] = "0" * 64
    elif fault == "core": expected["consumer_core_source_sha256"] = "0" * 64
    elif fault == "manifest": expected["effective_source_manifest_binding"]["sha256"] = "0" * 64
    else: cert["payload"]["performance_admission"] = True
    with pytest.raises(ValueError): m.verify_promotion(cert, expected, synthetic_fixture=True, fixture_public_keys=keys)

def test_fixture_key_even_valid_signature_and_caller_true_flag_cannot_promote_production(signed_packet):
    keys, cert, expected, *_ = signed_packet
    with pytest.raises(ValueError, match="Fixture"):
        m.verify_promotion(cert, expected, synthetic_fixture=False, fixture_public_keys=keys)
    cert = copy.deepcopy(cert); cert["payload"].pop("synthetic_fixture"); cert["payload"]["performance_admission"] = True
    with pytest.raises(ValueError, match="registry is empty"):
        m.verify_promotion(cert, expected)
