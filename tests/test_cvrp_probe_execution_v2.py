"""Execution2 source/crypto/bootstrap checks; no actual native allocation."""
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/rloo_sft_anchor_20261013"
EXEC = EXP / "cvrp_probe_execution_v2"
OVERRIDE = EXEC / "overrides"
WORKER = OVERRIDE / "experiments/rloo_sft_anchor_20261013/run_cvrp_timing_probe_v1.py"
BOOT = EXEC / "execute_worker_from_buffer_v2.py"
FIXTURE = EXP / "validation/v2_peft_cpu_all_arms_v1/cvrp"


def load(path, name):
    raw = path.read_bytes(); module = types.ModuleType(name); module.__file__ = str(path)
    module.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


m = load(WORKER, "probe_exec2_test_worker")


def raw(value): return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
def seal(value, key): value[key] = m.sha(m.encoded({k: v for k, v in value.items() if k != key})); return value

def put(path, value):
    body = value if isinstance(value, bytes) else raw(value); path.write_bytes(body); return m.binding(path, body)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    parent = tmp_path_factory.mktemp("probe-execution2-source")
    source = parent / "source"; m.package_source(source, repo=ROOT)
    spec = m.make_spec(source); spec_path = parent / "spec.json"; spec_path.write_bytes(raw(spec))
    return source, spec_path, m.load_core(source, spec_path, m.sha(spec_path.read_bytes()))


def tree(path): return ast.parse(path.read_bytes())
def classes_functions(path): return {n.name: n for n in tree(path).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
def dump(node): return ast.dump(node, include_attributes=False)


def test_kernel_all_numeric_and_sampler_math_AST_is_original_a1_source():
    old = classes_functions(ROOT / "src/clasp/entrypoints/profile_cvrp_timing_probe_v1.py")
    new = classes_functions(OVERRIDE / "src/clasp/entrypoints/profile_cvrp_timing_probe_v1.py")
    for name in ("fingerprint", "rng_fingerprint", "environment", "StopAtThree", "PassiveTiming", "admit_contract", "make_trainer"):
        assert dump(old[name]) == dump(new[name]), name
    # The only train-body delta is a truthful output version stamp after all
    # actor/optimizer/reference/GC/clock/window closure and object release.
    result = copy.deepcopy(new["run_probe"])
    for node in ast.walk(result):
        if isinstance(node, (ast.If, ast.For, ast.Try, ast.FunctionDef)):
            for attr in ("body", "orelse", "finalbody"):
                if hasattr(node, attr):
                    setattr(node, attr, [x for x in getattr(node, attr) if not (isinstance(x, ast.Assign) and
                        any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "result" and
                            isinstance(target.slice, ast.Constant) and target.slice.value == "execution_revision" for target in x.targets))])
    assert dump(old["run_probe"]) == dump(result)


def test_worker_actual_loader_data_and_context_functions_AST_preserved():
    old = classes_functions(EXP / "run_cvrp_timing_probe_v1.py"); new = classes_functions(WORKER)
    for name in ("file_inventory", "validate_initializer", "validate_job", "validate_process", "execution_environment",
        "original_native_tokenizer", "BoundProjectLoader", "BoundProjectFinder", "install_project"):
        assert dump(old[name]) == dump(new[name]), name
    # Native flow changes only the authenticated external-bootstrap admission.
    old_native = copy.deepcopy(old["native_run"]); new_native = copy.deepcopy(new["native_run"])
    old_native.body = old_native.body[1:]; new_native.body = new_native.body[1:]
    assert dump(old_native) == dump(new_native)


def test_old_v1_sources_and_historical_packets_not_rewritten():
    assert m.sha((ROOT / "src/clasp/entrypoints/profile_cvrp_timing_probe_v1.py").read_bytes()) == "a1d6a20a19d6f9c38ec912bb1f5e33dd560a3ffb41f924a1e114105f8ce46d7a"
    assert m.sha((EXP / "run_cvrp_timing_probe_v1.py").read_bytes()) == "8856d6c41871365d0c67aea84dbdd8a8ea83016fb5ff02b7d641071d759227eb"
    assert m.sha((EXP / "validation/cvrp_timing_probe_v1/validation.json").read_bytes()) == "afd4bce746455f1cd4047e62822d1019aad8b2bc1fa62e47e0752d06dc8b2258"
    assert m.sha((EXP / "validation/cvrp_timing_probe_native_parity_v2/validation.json").read_bytes()) == "9c7b8cbd37effec6e23f2d3d1efbb7953edb7baaeceee15bed96583506c86300"


def test_effective130_rows_keep_all_original121_and_exact_new_crypto_dependencies(packet):
    core = packet[2]
    assert len(core["buffers"]) == 130
    old = json.loads(core["buffers"]["original_source2_manifest.json"])
    assert len(old["files"]) == 121 and all(m.sha(core["buffers"][r["path"]]) == r["sha256"] for r in old["files"])
    assert core["spec"]["execution_revision"] == 2
    assert m.sha(core["buffers"][m.PROMOTION_PATH]) == m.PROMOTION_SHA
    assert m.sha(core["buffers"][m.PROXY_PATH]) == m.PROXY_SHA
    assert m.sha(core["buffers"][m.BOOTSTRAP_PATH]) == m.BOOTSTRAP_SHA
    proxy = m.module_buffer("reviewed_production_proxy", core["root"] / m.PROXY_PATH, core["buffers"][m.PROXY_PATH])
    assert set(proxy.PRODUCTION_VERIFIERS) == {m.PROMOTION_SHA}
    assert proxy.PRODUCTION_VERIFIERS[m.PROMOTION_SHA]["allowed_consumers"] == ["sft-anchor-cvrp-prefix-timing-probe-v1-result"]


@pytest.mark.parametrize("fault", ["proxy", "verifier", "bootstrap", "v1_revision", "float_revision", "binding"])
def test_admission_dependency_or_revision_tamper_denied_before_models(packet, tmp_path, fault):
    import shutil
    source, spec_path, core = packet; clone = tmp_path / "source"; shutil.copytree(source, clone)
    spec = copy.deepcopy(core["spec"])
    if fault in ("proxy", "verifier", "bootstrap"):
        relative = {"proxy": m.PROXY_PATH, "verifier": m.PROMOTION_PATH, "bootstrap": m.BOOTSTRAP_PATH}[fault]
        (clone / relative).write_bytes((clone / relative).read_bytes() + b"\n")
    elif fault == "binding": spec["crypto_proxy_binding"]["sha256"] = "0" * 64
    else: spec["execution_revision"] = 1 if fault == "v1_revision" else 2.0
    seal(spec, "spec_sha256"); path = tmp_path / "spec.json"; path.write_bytes(raw(spec))
    with pytest.raises(ValueError): m.load_core(clone, path, m.sha(path.read_bytes()))


def bootstrap_command(packet, *extra):
    source, spec_path, core = packet
    return [sys.executable, str(source / m.BOOTSTRAP_PATH), "--expected-bootstrap-sha256", m.BOOTSTRAP_SHA,
        "--expected-worker-sha256", core["worker_sha256"], "--source-root", str(source), "--spec", str(spec_path),
        "--expected-spec-sha256", m.sha(core["spec_raw"]), *extra]


def test_actual_external_bootstrap_executes_closed_worker_source_only(packet):
    result = subprocess.run(bootstrap_command(packet, "--validate-only"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["source_files"] == 130 and not record["native_registration_or_model_calls"]
    assert not list(packet[0].rglob("*.pyc"))


def test_source_buffer_bootstrap_rejects_wrong_worker_SHA_before_exec(packet):
    command = bootstrap_command(packet, "--validate-only")
    command[command.index("--expected-worker-sha256") + 1] = "0" * 64
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "before exec" in result.stderr


def test_direct_native_entry_without_external_bootstrap_cannot_import_torch_or_load_models(packet, monkeypatch):
    # It closes source then fails the external bootstrap guard before attempting
    # authority/signature/native-runtime imports. No CUDA device is touched.
    monkeypatch.delitem(m.__dict__, "__external_probe_bootstrap_sha256__", raising=False)
    source, spec, core = packet
    args = ["--expected-worker-sha256", core["worker_sha256"], "--source-root", str(source), "--spec", str(spec),
        "--expected-spec-sha256", m.sha(core["spec_raw"])]
    with pytest.raises(ValueError, match="external same-buffer"): m.main(args)


def fixture_registration(packet, tmp_path):
    # Evidence templates are independently shared with already-closed V1 tests;
    # all new scope/source identities are explicitly signed with ephemeral keys.
    from test_cvrp_timing_probe_worker_v1 import signed_fixture
    reg_raw, _, _ = signed_fixture(packet, tmp_path)
    reg = json.loads(reg_raw); reg["execution_revision"] = 2
    authority = json.loads(Path(reg["authority_binding"]["path"]).read_bytes())
    authority["execution_revision"] = 2; seal(authority, "authority_sha256")
    reg["authority_binding"] = put(Path(reg["authority_binding"]["path"]), authority)
    intent = json.loads(Path(reg["submission_intent_binding"]["path"]).read_bytes())
    intent["authority_sha256"] = reg["authority_binding"]["sha256"]
    reg["submission_intent_binding"] = put(Path(reg["submission_intent_binding"]["path"]), intent)
    attempt = json.loads(Path(reg["attempt_binding"]["path"]).read_bytes())
    attempt["authority_sha256"] = reg["authority_binding"]["sha256"]
    attempt["submission_intent_sha256"] = reg["submission_intent_binding"]["sha256"]
    reg["attempt_binding"] = put(Path(reg["attempt_binding"]["path"]), attempt)
    action = json.loads(Path(reg["submission_action_binding"]["path"]).read_bytes())
    action["intent_sha256"] = reg["submission_intent_binding"]["sha256"]
    reg["submission_action_binding"] = put(Path(reg["submission_action_binding"]["path"]), action)
    return reg


def fixture_sign(core, reg, *, payload_changes=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    verifier = load(EXP / "verify_registration_promotion_v1.py", "test_exec2_legacy_verifier")
    proxy = load(OVERRIDE / m.PROXY_PATH, "test_exec2_proxy")
    reg_raw = raw(reg); spec = core["spec"]
    expected = {"consumer": "sft-anchor-cvrp-prefix-timing-probe-v1-result", "operator_protocol": m.PROTOCOL,
        "worker_sha256": core["worker_sha256"], "kernel_sha256": m.sha(core["buffers"][m.KERNEL_PATH]),
        "source2_sha256": core["preparer"].TRAINER_SHA, "source_manifest_sha256": m.sha(core["manifest_raw"]),
        "original_source_manifest_sha256": core["preparer"].ORIGINAL_MANIFEST_SHA,
        "contract_semantic_sha256": spec["contract_semantic_sha256"], "base": spec["base"], "sft": spec["sft"],
        "output": spec["output"], "native_source_root": spec["source_root"], "spec_sha256": spec["spec_sha256"],
        "spec_file_sha256": m.sha(core["spec_raw"]), "authority_sha256": reg["authority_binding"]["sha256"],
        "attempt_sha256": reg["attempt_binding"]["sha256"], "registration_sha256": m.sha(reg_raw),
        "initializer_inventory_sha256": reg["initializer_inventory_binding"]["sha256"], "job_id": "9990001",
        "committed_prefix_updates": 3, "original_horizon": 100, "study_admission": False, "policy_outcomes_read": False,
        "registered_utc": reg["registered_utc"], "execution_revision": 2, "external_bootstrap_sha256": m.BOOTSTRAP_SHA,
        "crypto_proxy_sha256": m.PROXY_SHA, "trusted_verifier_sha256": m.PROMOTION_SHA}
    payload = {"protocol": verifier.PROTOCOL, "schema_version": 1, "synthetic_fixture": True, "performance_admission": False, **expected}
    payload.update(payload_changes or {})
    private = Ed25519PrivateKey.generate()
    cert = {"payload": payload, "signer_key_id": "diagnostic-fixture-key", "signature_base64": base64.b64encode(private.sign(verifier.encoded(payload))).decode()}
    keys = {"diagnostic-fixture-key": {"scope": "fixture_only", "purpose": "independent_prepolicy_registration", "crypto_environment": verifier.crypto_environment(),
        "public_key_base64": base64.b64encode(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}}
    return reg_raw, raw(cert), keys, proxy.interpreter_identity("/usr/bin/python3")


def test_real_isolated_crypto_fixture_all_packet_bytes_stay_diagnostic(packet, tmp_path):
    reg = fixture_registration(packet, tmp_path); reg_raw, cert_raw, keys, interpreter = fixture_sign(packet[2], reg)
    proof = m.admit_registration(packet[2], reg_raw, cert_raw, fixture_public_keys=keys, fixture_interpreter_binding=interpreter)
    assert proof["synthetic_fixture"] and not proof["native_operator_admitted"]
    assert proof["execution_revision"] == 2 and proof["signed_promotion"]["verification_transport"] == "sft-anchor-isolated-crypto-verification-v1"
    assert proof["signed_promotion"]["interpreter"] == interpreter
    assert proof["history"]["combined_reserved_requested_gpu_hours"] == 94
    with pytest.raises(ValueError): m.native_run(packet[2], proof)


@pytest.mark.parametrize("field,value", [("study_admission", 0), ("policy_outcomes_read", 0), ("original_horizon", 100.0),
    ("committed_prefix_updates", 3.0), ("execution_revision", 2.0), ("crypto_proxy_sha256", "0" * 64),
    ("external_bootstrap_sha256", "0" * 64), ("trusted_verifier_sha256", "0" * 64)])
def test_actual_signed_wrong_typed_or_unbound_payload_rejected_before_model_load(packet, tmp_path, field, value):
    reg = fixture_registration(packet, tmp_path)
    reg_raw, cert_raw, keys, interpreter = fixture_sign(packet[2], reg, payload_changes={field: value})
    with pytest.raises(ValueError): m.admit_registration(packet[2], reg_raw, cert_raw, fixture_public_keys=keys, fixture_interpreter_binding=interpreter)


def test_real_fixture_keys_or_local_interpreter_cannot_become_native_admission(packet, tmp_path):
    reg = fixture_registration(packet, tmp_path); reg_raw, cert_raw, keys, interpreter = fixture_sign(packet[2], reg)
    with pytest.raises(ValueError): m.admit_registration(packet[2], reg_raw, cert_raw)
    with pytest.raises(ValueError): m.admit_registration(packet[2], reg_raw, cert_raw, fixture_interpreter_binding=interpreter)
    proxy = load(OVERRIDE / m.PROXY_PATH, "public_only_proxy_admission")
    with pytest.raises(ValueError): proxy.verify_promotion(packet[2]["buffers"][m.PROMOTION_PATH], json.loads(cert_raw), {"consumer": m.PROTOCOL})


def test_actual_external_bootstrap_CPU_four_rank_prefix_matches_original_control(packet, tmp_path):
    from test_cvrp_timing_probe_kernel_v1 import contract, same_numerics
    contract_path = tmp_path / "tiny_fixture.json"; contract_path.write_bytes(raw(contract()))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    control = tmp_path / "control"; profile = tmp_path / "profile"
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    commands = [prefix + [str(ROOT / "tests/test_cvrp_timing_probe_kernel_v1.py"), "--worker", "--mode", "control", "--contract", str(contract_path), "--output", str(control)],
        prefix + bootstrap_command(packet, "--cpu-fixture", "--fixture-contract", str(contract_path),
            "--fixture-base", str(FIXTURE / "fixture_base"), "--fixture-sft", str(FIXTURE / "fixture_sft"), "--fixture-output", str(profile))[1:]]
    for number, command in enumerate(commands):
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, text=True)
        (tmp_path / f"execution{number}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-15000:]
    a, b = [json.loads((path / "timing_probe.json").read_bytes()) for path in (control, profile)]
    same_numerics(a, b)
    assert b["execution_revision"] == b["source_admission"]["execution_revision"] == 2
    assert b["source_admission"]["worker_sha256"] == packet[2]["worker_sha256"]
    assert b["reference_discarded"] and b["heldout_policy_calls"] == 0
