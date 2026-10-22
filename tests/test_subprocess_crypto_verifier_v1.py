"""Isolated signature transport fixtures, never model/admission evidence."""
import base64
from pathlib import Path
import sys
import types

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"


def module(filename):
    path = EXP / filename; raw = path.read_bytes()
    value = types.ModuleType(filename.removesuffix(".py")); value.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


m = module("subprocess_crypto_verifier_v1.py")
v = module("verify_registration_promotion_v1.py")
VERIFIER = (EXP / "verify_registration_promotion_v1.py").read_bytes()


@pytest.fixture
def inputs():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    expected = {"consumer": "isolated_crypto_transport_fixture_only", "source_sha256": "1" * 64}
    payload = {"protocol": v.PROTOCOL, "schema_version": 1, "synthetic_fixture": True,
        "performance_admission": False, **expected}
    certificate = {"payload": payload, "signer_key_id": "ephemeral_fixture_only",
        "signature_base64": base64.b64encode(private.sign(v.encoded(payload))).decode()}
    keys = {"ephemeral_fixture_only": {"scope": "fixture_only", "purpose": "independent_prepolicy_registration",
        "public_key_base64": base64.b64encode(public).decode(), "crypto_environment": v.crypto_environment()}}
    return {"verifier_raw": VERIFIER, "certificate": certificate, "expected": expected,
        "synthetic_fixture": True, "fixture_public_keys": keys,
        "fixture_interpreter_binding": m.interpreter_identity(sys.executable)}


def test_real_fixture_signature_verified_in_fresh_isolated_interpreter(inputs):
    result = m.verify_promotion(**inputs)
    assert result["synthetic_fixture"] is True and result["production_registration_authenticated"] is False
    assert result["certificate_sha256"] == m.sha(m.encoded(inputs["certificate"]))
    assert result["signed_payload_sha256"] == m.sha(m.encoded(inputs["certificate"]["payload"]))
    assert result["interpreter"] == inputs["fixture_interpreter_binding"]
    assert result["verifier_sha256"] == m.FIXTURE_VERIFIER_SHA
    assert result["verification_transport"] == m.PROTOCOL
    assert not result["native_mutations"] and result["model_calls"] == 0
    assert m.PRODUCTION_VERIFIERS == {} and v.TRUSTED_PUBLIC_KEYS == {}


@pytest.mark.parametrize("fault", ["unknown_code", "mutable_buffer", "bool_flag", "fixture_performance",
    "fixture_marker", "interpreter_sha", "interpreter_size", "interpreter_resolved", "extra_interpreter_key",
    "missing_keys", "native_with_fixture", "native_empty_registry"])
def test_unregistered_or_changed_source_scope_interpreter_denied_before_process(inputs, monkeypatch, fault):
    if fault == "unknown_code": inputs["verifier_raw"] += b"\n# unreviewed\n"
    elif fault == "mutable_buffer": inputs["verifier_raw"] = bytearray(inputs["verifier_raw"])
    elif fault == "bool_flag": inputs["synthetic_fixture"] = 1
    elif fault == "fixture_performance": inputs["certificate"]["payload"]["performance_admission"] = True
    elif fault == "fixture_marker": inputs["certificate"]["payload"]["synthetic_fixture"] = False
    elif fault.startswith("interpreter_"):
        key, value = {"interpreter_sha": ("sha256", "0" * 64), "interpreter_size": ("size_bytes", 1),
            "interpreter_resolved": ("resolved_path", "/unregistered/python")}[fault]
        inputs["fixture_interpreter_binding"][key] = value
    elif fault == "extra_interpreter_key": inputs["fixture_interpreter_binding"]["unbound"] = True
    elif fault == "missing_keys": inputs["fixture_public_keys"] = None
    elif fault == "native_with_fixture": inputs["synthetic_fixture"] = False
    else:
        inputs.update(synthetic_fixture=False, fixture_public_keys=None, fixture_interpreter_binding=None)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("Denied packet cannot start a verification process"))
    with pytest.raises(ValueError): m.verify_promotion(**inputs)


@pytest.mark.parametrize("fault", ["signature", "expected", "key", "environment"])
def test_actual_signature_expected_key_or_installed_environment_mismatch_denied(inputs, fault):
    if fault == "signature": inputs["certificate"]["signature_base64"] = base64.b64encode(b"0" * 64).decode()
    elif fault == "expected": inputs["expected"]["source_sha256"] = "2" * 64
    elif fault == "key": inputs["fixture_public_keys"]["ephemeral_fixture_only"]["public_key_base64"] = base64.b64encode(b"1" * 32).decode()
    else: inputs["fixture_public_keys"]["ephemeral_fixture_only"]["crypto_environment"]["python_version"] = "0.0.0"
    with pytest.raises(ValueError):
        m.verify_promotion(**inputs)


@pytest.mark.parametrize("signed_value,expected_value", [(3.0, 3), (0, False), (1, True)])
def test_authentic_signed_numeric_alias_cannot_bind_typed_expected_value(inputs, monkeypatch, signed_value, expected_value):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    inputs["certificate"]["payload"]["typed_field"] = signed_value
    inputs["certificate"]["signature_base64"] = base64.b64encode(
        private.sign(v.encoded(inputs["certificate"]["payload"]))).decode()
    inputs["fixture_public_keys"]["ephemeral_fixture_only"]["public_key_base64"] = base64.b64encode(public).decode()
    inputs["expected"]["typed_field"] = expected_value
    # The actual frozen verifier accepts Python numeric equality. The new
    # transport enforces exact types without changing that historical source.
    weaker = v.verify_promotion(inputs["certificate"], inputs["expected"], synthetic_fixture=True,
        fixture_public_keys=inputs["fixture_public_keys"])
    assert weaker["synthetic_fixture"] is True
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("Typed alias must fail before child execution"))
    with pytest.raises(ValueError, match="Typed canonical"):
        m.verify_promotion(**inputs)


@pytest.mark.parametrize("fault", ["response_sha", "response_protocol", "certificate_identity", "native_flag", "stderr", "exit", "extra_field"])
def test_wrong_or_ambiguous_response_cannot_supply_verification(inputs, monkeypatch, fault):
    verification = v.verify_promotion(inputs["certificate"], inputs["expected"], synthetic_fixture=True,
        fixture_public_keys=inputs["fixture_public_keys"])
    response = {"protocol": m.PROTOCOL + "-response", "verifier_sha256": m.FIXTURE_VERIFIER_SHA,
        "verification": verification}
    exit_code, stderr = 0, b""
    if fault == "response_sha": response["verifier_sha256"] = "0" * 64
    elif fault == "response_protocol": response["protocol"] = "unregistered"
    elif fault == "certificate_identity": verification["certificate_sha256"] = "0" * 64
    elif fault == "native_flag": verification["production_registration_authenticated"] = True
    elif fault == "stderr": stderr = b"unexpected diagnostic"
    elif fault == "exit": exit_code = 1
    else: response["unbound"] = True
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=exit_code, stdout=m.encoded(response), stderr=stderr))
    with pytest.raises(ValueError): m.verify_promotion(**inputs)


def test_interpreter_change_across_process_rejected(inputs, monkeypatch):
    original = m.interpreter_identity
    calls = []
    def observed(invocation):
        result = original(invocation)
        calls.append(invocation)
        if len(calls) == 2: result["sha256"] = "0" * 64
        return result
    monkeypatch.setattr(m, "interpreter_identity", observed)
    with pytest.raises(ValueError, match="changed across verification"):
        m.verify_promotion(**inputs)
