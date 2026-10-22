"""Closed-source/operator diagnostics; never register or submit a native probe."""
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
WORKER = EXP / "run_cvrp_timing_probe_v1.py"
FIXTURE = EXP / "validation/v2_peft_cpu_all_arms_v1/cvrp"


def load(path, name):
    raw = path.read_bytes(); module = types.ModuleType(name); module.__file__ = str(path)
    module.__executed_source_sha256__ = hashlib.sha256(raw).hexdigest()
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


m = load(WORKER, "probe_test_worker")


def raw(value): return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
def seal(value, key):
    value[key] = m.sha(m.encoded({k: v for k, v in value.items() if k != key})); return value

def save(path, value):
    body = value if isinstance(value, bytes) else raw(value); path.write_bytes(body)
    return m.binding(path, body)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    parent = tmp_path_factory.mktemp("probe-closed-source")
    source = parent / "source"; m.package_source(source, repo=ROOT)
    spec = m.make_spec(source); spec_path = parent / "spec.json"; spec_path.write_bytes(raw(spec))
    return source, spec_path, m.load_core(source, spec_path, m.sha(spec_path.read_bytes()))


def test_source_packet_preserves_actual_original121_and_executes_exact_buffers(packet):
    source, path, core = packet
    assert len(core["buffers"]) == 127
    old = json.loads(core["buffers"]["original_source2_manifest.json"])
    assert len(old["files"]) == 121
    assert all(m.sha(core["buffers"][row["path"]]) == row["sha256"] for row in old["files"])
    assert core["worker_sha256"] == m.__executed_source_sha256__
    assert not core["spec"]["heldout_evaluation"] and not core["spec"]["study_admission"]
    assert core["spec"]["recipe"]["original_horizon"] == 100
    assert core["spec"]["recipe"]["committed_prefix_updates"] == 3


def test_actual_source_buffer_bootstrap_read_only_validation(packet):
    source, spec_path, core = packet
    command = [sys.executable, str(source / m.WORKER_PATH), "--expected-worker-sha256", core["worker_sha256"],
        "--source-root", str(source), "--spec", str(spec_path), "--expected-spec-sha256", m.sha(core["spec_raw"]), "--validate-only"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["status"] == "passed_source_only" and record["source_files"] == 127
    assert not record["native_registration_or_model_calls"] and not record["native_submission_authorized"]
    assert not list(source.rglob("*.pyc"))


@pytest.mark.parametrize("fault", ["old_source", "new_source", "cache", "unbound", "symlink", "spec", "worker_metadata"])
def test_closed_source_rejects_drift_or_import_override(packet, tmp_path, fault):
    import shutil
    source, spec_path, _ = packet; clone = tmp_path / "source"; shutil.copytree(source, clone)
    spec_body = spec_path.read_bytes(); local_spec = tmp_path / "spec.json"; local_spec.write_bytes(spec_body)
    if fault == "old_source": (clone / "src/clasp/entrypoints/train_sft_anchor_pilot_v2.py").write_text("raise RuntimeError('changed')\n")
    elif fault == "new_source": (clone / m.KERNEL_PATH).write_text("raise RuntimeError('changed')\n")
    elif fault in ("cache", "unbound"): (clone / ("override.pyc" if fault == "cache" else "override.py")).write_bytes(b"unbound")
    elif fault == "symlink": (clone / "linked").symlink_to(clone / "source_manifest.json")
    elif fault == "spec": local_spec.write_bytes(spec_body + b" ")
    else:
        alternate = load(WORKER, "wrong_executed_metadata"); alternate.__executed_source_sha256__ = "0" * 64
        with pytest.raises(ValueError): alternate.load_core(clone, local_spec, m.sha(spec_body))
        return
    with pytest.raises(ValueError): m.load_core(clone, local_spec, m.sha(spec_body))


def test_unbound_cached_project_import_is_rejected_before_model_load(packet, monkeypatch):
    monkeypatch.setitem(sys.modules, "clasp.injected", types.ModuleType("clasp.injected"))
    with pytest.raises(ValueError, match="cached"): m.install_project(packet[2])


def initializer(packet):
    core = packet[2]; spec = core["spec"]
    contract = json.loads((FIXTURE / "contract.json").read_bytes())
    return {"protocol": m.PROTOCOL + "-initializer-inventory", "base": spec["base"], "sft": spec["sft"],
        "initializer_identity_sha256": contract["initializer"]["sha256"],
        "base_files": m.file_inventory(FIXTURE / "fixture_base"), "sft_files": m.file_inventory(FIXTURE / "fixture_sft")}, contract


def test_actual_tiny_base_and_peft_config_tokenizer_complete_inventory(packet):
    inventory, contract = initializer(packet)
    proof = m.validate_initializer(inventory, spec=packet[2]["spec"], contract=contract,
        actual_roots={"base": FIXTURE / "fixture_base", "sft": FIXTURE / "fixture_sft"})
    assert proof["complete_initializer_inventory_verified"]


@pytest.mark.parametrize("fault", ["config", "generation", "tokenizer", "weights", "sft_path", "initializer"])
def test_initializer_custody_rejects_actual_tiny_config_generation_tokens_weights_changes(packet, tmp_path, fault):
    import shutil
    inventory, contract = initializer(packet)
    base = tmp_path / "base"; sft = tmp_path / "sft"
    shutil.copytree(FIXTURE / "fixture_base", base); shutil.copytree(FIXTURE / "fixture_sft", sft)
    target = {"config": base / "config.json", "generation": base / "generation_config.json", "tokenizer": sft / "tokenizer.json",
        "weights": sft / "adapter_model.safetensors"}.get(fault)
    if target: target.write_bytes(target.read_bytes() + b" ")
    elif fault == "sft_path": inventory["sft"] = "/changed"
    else: inventory["initializer_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError): m.validate_initializer(inventory, spec=packet[2]["spec"], contract=contract,
        actual_roots={"base": base, "sft": sft})


def job(spec):
    return {"JobId": "9990001", "JobState": "RUNNING", "UserId": "uag72kz(9999)",
        "Account": "vas@a100", "Partition": "gpu_p5", "QOS": "qos_gpu_a100-t3", "TimeLimit": "00:30:00",
        "NumNodes": "1", "NumCPUs": "80", "ReqB:S:C:T": "0:0:*:1", "Command": spec["control_root"] + "/job.slurm",
        "AllocTRES": "billing=5,cpu=80,gres/gpu=4,gres/gpu:a100=4,mem=300000M,node=1", "MinMemoryNode": "300000M",
        "ReqTRES": "billing=4,cpu=40,gres/gpu=4,mem=300000M,node=1", "Features": "a100", "TresPerTask": "cpu=40"}


def test_exact_known_native_cpu80_bill5_allocation_alias_admitted_as_fixture_only(packet):
    value = job(packet[2]["spec"])
    proof = m.validate_job(m.parse_job(" ".join(k + "=" + v for k, v in value.items())), spec=packet[2]["spec"], job_id="9990001")
    assert proof["allocated_tres"]["cpu"] == "80" and proof["allocated_tres"]["billing"] == "5"


@pytest.mark.parametrize("fault", ["gpus", "wall", "cpu", "memory", "owner", "qos", "state", "script", "rank_geometry"])
def test_scheduler_context_change_rejects_before_model_loading(packet, fault):
    value = job(packet[2]["spec"])
    if fault == "gpus": value["AllocTRES"] = value["AllocTRES"].replace("gpu=4", "gpu=8")
    elif fault == "memory": value["AllocTRES"] = value["AllocTRES"].replace("300000M", "400000M")
    else:
        key, replacement = {"wall": ("TimeLimit", "02:00:00"), "cpu": ("NumCPUs", "40"), "owner": ("UserId", "other(9999)"),
            "qos": ("QOS", "qos_gpu_a100-dev"), "state": ("JobState", "PENDING"), "script": ("Command", "/copied/job.slurm"),
            "rank_geometry": ("TresPerTask", "cpu=80")}[fault]
        value[key] = replacement
    with pytest.raises(ValueError): m.validate_job(value, spec=packet[2]["spec"], job_id="9990001")


def test_duplicate_scheduler_fields_are_not_silently_overwritten():
    with pytest.raises(ValueError): m.parse_job("JobId=1 JobId=2")


def signed_fixture(packet, tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    core = packet[2]; p = core["preparer"]; spec = core["spec"]
    pilot = (EXP / "collected_native_first_complete_v1/status/campaign_authority.json").read_bytes()
    failed = (EXP / "validation/native_monitor/prior_failed_1030929_reverification_v2_bound.json").read_bytes()
    authority = p.make_authority_proposal(spec["spec_sha256"]); authority["synthetic_fixture"] = True; seal(authority, "authority_sha256")
    authority_binding = save(tmp_path / "authority.json", authority)
    script_binding = save(tmp_path / "job.slurm", b"#!/bin/bash\n# synthetic fixture only\n")
    intent = {"protocol": p.PROTOCOL + "-intent", "reservation_id": "diagnostic/cvrp-first3-v1",
        "spec_sha256": spec["spec_sha256"], "authority_sha256": authority_binding["sha256"],
        "source_manifest_sha256": m.sha(core["manifest_raw"]), "script_sha256": script_binding["sha256"], "automatic_retry": False}
    intent_binding = save(tmp_path / "submission_intent.json", intent)
    attempt = {"protocol": p.PROTOCOL + "-attempt", "state": "submitted", "job_id": "9990001", "attempt_number": 1,
        "reservation_id": "diagnostic/cvrp-first3-v1", "replacement_or_retry": False, "spec_sha256": spec["spec_sha256"],
        "authority_sha256": authority_binding["sha256"], "source_manifest_sha256": m.sha(core["manifest_raw"]),
        "script_path": spec["control_root"] + "/job.slurm", "script_sha256": script_binding["sha256"],
        "submission_argv": ["sbatch", "--parsable", spec["control_root"] + "/job.slurm"], "step_strategy": "torchrun4_in_batch",
        "requested_gpus": 4, "requested_wall_seconds": 1800, "submission_intent_sha256": intent_binding["sha256"]}
    action = {"protocol": p.PROTOCOL + "-submission-action", "intent_sha256": intent_binding["sha256"], "argv": attempt["submission_argv"],
        "returncode": 0, "stdout": "9990001\n", "stderr": ""}
    inventory, _ = initializer(packet)
    registration = {"protocol": m.PROTOCOL + "-registration", "schema_version": 1, "synthetic_fixture": True,
        "spec_binding": m.binding(packet[1], core["spec_raw"]), "source_manifest_binding": spec["source_manifest_binding"],
        "original_source_manifest_binding": spec["original_source_manifest_binding"], "source_root": spec["source_root"],
        "control_root": spec["control_root"], "reservation_id": "diagnostic/cvrp-first3-v1", "first_attempt_only": True,
        "study_admission": False, "heldout_policy_calls": 0, "policy_outcomes_read": False, "registered_utc": "2026-10-14T00:00:00+00:00",
        "authority_binding": authority_binding, "script_binding": script_binding, "attempt_binding": save(tmp_path / "attempt.json", attempt),
        "pilot_authority_binding": save(tmp_path / "pilot.json", pilot), "prior_failure_binding": save(tmp_path / "failed.json", failed),
        "initializer_inventory_binding": save(tmp_path / "initializer.json", inventory), "submission_intent_binding": intent_binding,
        "submission_action_binding": save(tmp_path / "submission_action.json", action),
        "attempt_directory_inventory_binding": save(tmp_path / "directory.json", {"protocol": m.PROTOCOL + "-attempt-directory-inventory",
            "independent_read_only_collection": True, "control_root": spec["control_root"], "attempt_files": ["attempt.json"],
            "intent_files": ["submission_intent.json"], "action_files": ["submission_action.json"], "unresolved_intents": []})}
    registration_raw = raw(registration)
    expected = {"consumer": "sft-anchor-cvrp-prefix-timing-probe-v1-result", "operator_protocol": m.PROTOCOL,
        "worker_sha256": core["worker_sha256"], "kernel_sha256": m.sha(core["buffers"][m.KERNEL_PATH]), "source2_sha256": p.TRAINER_SHA,
        "source_manifest_sha256": m.sha(core["manifest_raw"]), "original_source_manifest_sha256": p.ORIGINAL_MANIFEST_SHA,
        "contract_semantic_sha256": spec["contract_semantic_sha256"], "base": spec["base"], "sft": spec["sft"],
        "output": spec["output"], "native_source_root": spec["source_root"],
        "spec_sha256": spec["spec_sha256"], "spec_file_sha256": m.sha(core["spec_raw"]), "authority_sha256": authority_binding["sha256"],
        "attempt_sha256": registration["attempt_binding"]["sha256"], "registration_sha256": m.sha(registration_raw),
        "initializer_inventory_sha256": registration["initializer_inventory_binding"]["sha256"], "job_id": "9990001",
        "committed_prefix_updates": 3, "original_horizon": 100, "study_admission": False, "policy_outcomes_read": False,
        "registered_utc": registration["registered_utc"]}
    verifier = load(EXP / "verify_registration_promotion_v1.py", "fixture_probe_crypto")
    payload = {"protocol": verifier.PROTOCOL, "schema_version": 1, "synthetic_fixture": True, "performance_admission": False, **expected}
    private = Ed25519PrivateKey.generate()
    certificate = {"payload": payload, "signer_key_id": "fixture-only",
        "signature_base64": base64.b64encode(private.sign(verifier.encoded(payload))).decode()}
    keys = {"fixture-only": {"scope": "fixture_only", "purpose": "independent_prepolicy_registration", "crypto_environment": verifier.crypto_environment(),
        "public_key_base64": base64.b64encode(private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}}
    return registration_raw, raw(certificate), keys


def test_real_ed25519_fixture_whole_packet_admission_remains_diagnostic(packet, tmp_path):
    reg, cert, keys = signed_fixture(packet, tmp_path)
    proof = m.admit_registration(packet[2], reg, cert, fixture_public_keys=keys)
    assert proof["synthetic_fixture"] and not proof["native_operator_admitted"]
    assert proof["history"]["combined_reserved_requested_gpu_hours"] == 94
    assert proof["signed_expectations"]["spec_file_sha256"] != proof["signed_expectations"]["spec_sha256"]
    with pytest.raises(ValueError): m.native_run(packet[2], proof)


@pytest.mark.parametrize("fault", ["bool_zero", "retry", "prefix", "registration_bytes", "signature", "future", "source_identity", "native_boolean"])
def test_signed_fixture_cannot_hide_bad_scope_or_custody(packet, tmp_path, fault):
    reg_raw, cert_raw, keys = signed_fixture(packet, tmp_path)
    reg, cert = json.loads(reg_raw), json.loads(cert_raw)
    if fault == "bool_zero": reg["heldout_policy_calls"] = False
    elif fault == "retry": reg["first_attempt_only"] = False
    elif fault == "prefix": cert["payload"]["committed_prefix_updates"] = 4
    elif fault == "registration_bytes": reg["extra_unregistered_field"] = 1
    elif fault == "signature": cert["signature_base64"] = base64.b64encode(bytes(64)).decode()
    elif fault == "future": reg["registered_utc"] = "2099-01-01T00:00:00+00:00"
    elif fault == "source_identity": cert["payload"]["worker_sha256"] = "0" * 64
    else: reg["synthetic_fixture"] = False; reg["native_operator_admitted"] = True
    with pytest.raises(ValueError): m.admit_registration(packet[2], raw(reg), raw(cert), fixture_public_keys=keys)


def test_default_production_keys_empty_and_no_caller_registry_can_install_one(packet, tmp_path):
    reg_raw, cert_raw, _ = signed_fixture(packet, tmp_path)
    with pytest.raises(ValueError): m.admit_registration(packet[2], reg_raw, cert_raw)


def test_actual_closed_buffer_four_cpu_rank_prefix_equals_original_v2_control(packet, tmp_path):
    # Original control and operator each run the same tiny PEFT first3/H100
    # schedule in a fresh four-rank Gloo world; no evaluation or checkpoint.
    from test_cvrp_timing_probe_kernel_v1 import contract, same_numerics
    contract_path = tmp_path / "fixture.json"; contract_path.write_bytes(raw(contract()))
    source, spec_path, core = packet
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    control = tmp_path / "control"; profile = tmp_path / "profile"
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    commands = [prefix + [str(ROOT / "tests/test_cvrp_timing_probe_kernel_v1.py"), "--worker", "--mode", "control",
        "--contract", str(contract_path), "--output", str(control)], prefix + [str(source / m.WORKER_PATH),
        "--expected-worker-sha256", core["worker_sha256"], "--source-root", str(source), "--spec", str(spec_path),
        "--expected-spec-sha256", m.sha(core["spec_raw"]), "--cpu-fixture", "--fixture-contract", str(contract_path),
        "--fixture-base", str(FIXTURE / "fixture_base"), "--fixture-sft", str(FIXTURE / "fixture_sft"), "--fixture-output", str(profile)]]
    for number, command in enumerate(commands):
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, text=True)
        (tmp_path / f"worker{number}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-15000:]
    a, b = [json.loads((path / "timing_probe.json").read_bytes()) for path in (control, profile)]
    same_numerics(a, b)
    assert b["source_admission"]["worker_sha256"] == core["worker_sha256"]
    assert b["source_admission"]["source_manifest_sha256"] == m.sha(core["manifest_raw"])
    assert b["reference_discarded"] and b["actor_discarded"] and not b["study_admission"]
    assert not list(profile.rglob("checkpoint-*"))

@pytest.mark.parametrize("fault", ["uid", "host", "rank", "local_rank", "world", "pid"])
def test_native_process_custody_rejects_other_host_owner_or_rank(packet, fault):
    fields = job(packet[2]["spec"]); fields.update(NodeList="fixture-node", BatchHost="fixture-node")
    kwargs = {"host": "fixture-node", "uid": 9999, "pid": 123, "rank": 1, "local_rank": 1,
        "distributed_rank": 1, "distributed_world": 4}
    changed = {"uid": ("uid", 1000), "host": ("host", "other-node"), "rank": ("distributed_rank", 2),
        "local_rank": ("local_rank", 2), "world": ("distributed_world", 2), "pid": ("pid", 0)}
    key, value = changed[fault]; kwargs[key] = value
    with pytest.raises(ValueError): m.validate_process(fields, **kwargs)


def test_native_process_custody_retains_actual_host_uid_pid_and_rank_as_diagnostic(packet):
    fields = job(packet[2]["spec"]); fields.update(NodeList="fixture-node", BatchHost="fixture-node")
    proof = m.validate_process(fields, host="fixture-node.example", uid=9999, pid=123, rank=2, local_rank=2,
        distributed_rank=2, distributed_world=4)
    assert proof["uid"] == 9999 and proof["rank"] == 2 and proof["pid"] == 123


def test_initializer_full_directory_membership_cannot_change_during_stream(tmp_path, monkeypatch):
    (tmp_path / "first.json").write_bytes(b"original")
    original = Path.rglob; calls = 0
    def changed(self, pattern):
        nonlocal calls
        if self == tmp_path:
            calls += 1
            if calls == 2: (tmp_path / "added.json").write_bytes(b"unadmitted")
        return original(self, pattern)
    monkeypatch.setattr(Path, "rglob", changed)
    with pytest.raises(ValueError, match="full directory"): m.file_inventory(tmp_path)


def test_actual_tiny_native_BASE_tokenizer_setup_matches_original_v2_production(packet):
    from transformers import AutoTokenizer
    contract = json.loads((FIXTURE / "contract.json").read_bytes())
    actual, proof = m.original_native_tokenizer(FIXTURE / "fixture_base", contract, AutoTokenizer)
    original = AutoTokenizer.from_pretrained(FIXTURE / "fixture_base", local_files_only=True, trust_remote_code=False)
    original.padding_side = "left"
    if original.pad_token_id is None: original.pad_token = original.eos_token
    assert actual.get_vocab() == original.get_vocab()
    assert actual.model_input_names == original.model_input_names
    assert actual.pad_token_id == original.pad_token_id and actual.eos_token_id == original.eos_token_id
    assert actual.padding_side == original.padding_side == "left"
    assert proof["source"] == "original_base" and proof["heldout_model_or_reward_calls"] == 0
    assert proof["training_prompts_checked"] == len(contract["training_rows"])
    assert proof["validation_prompts_length_checked"] == len(contract["validation_rows"])


@pytest.mark.parametrize("pad_present", [False, True])
def test_native_BASE_tokenizer_padEOS_and_model_input_names_are_exact_original_behavior(pad_present):
    seen = []; calls = []
    class Tokenizer:
        model_input_names = ["input_ids", "attention_mask", "preserve_original_extra_field"]
        eos_token = "EOS"; eos_token_id = 7
        pad_token_id = 3 if pad_present else None
        padding_side = "right"
        @property
        def pad_token(self): return "PAD" if self.pad_token_id == 3 else "EOS"
        @pad_token.setter
        def pad_token(self, value):
            assert value == "EOS"; self.pad_token_id = self.eos_token_id
        def __call__(self, prompt, truncation):
            assert truncation is False; calls.append(prompt)
            return types.SimpleNamespace(input_ids=[1] * len(prompt))
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs): seen.append((args, kwargs)); return Tokenizer()
    contract = {"training_rows": [{"prompt": "train"}], "validation_rows": [{"prompt": "validate"}]}
    tokenizer, proof = m.original_native_tokenizer("/original/BASE", contract, Loader)
    assert seen == [(('/original/BASE',), {"local_files_only": True, "trust_remote_code": False})]
    assert tokenizer.pad_token_id == (3 if pad_present else 7)
    assert tokenizer.model_input_names == Tokenizer.model_input_names
    assert calls == ["train", "validate"] and proof["maximum_input_prompt_tokens"] == 8


@pytest.mark.parametrize("partition", ["training_rows", "validation_rows"])
def test_original_input_cap_is_checked_before_actor_or_reference_load(partition):
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return types.SimpleNamespace(pad_token_id=0, eos_token_id=1, eos_token="EOS", model_input_names=["input_ids"],
                __call__=lambda *args, **kwargs: None)
    class Tokenizer:
        pad_token_id=0; eos_token_id=1; eos_token="EOS"; model_input_names=["input_ids"]
        def __call__(self, prompt, truncation): return types.SimpleNamespace(input_ids=[1] * len(prompt))
    Loader.from_pretrained = staticmethod(lambda *args, **kwargs: Tokenizer())
    contract = {"training_rows": [{"prompt": "short"}], "validation_rows": [{"prompt": "short"}]}
    contract[partition][0]["prompt"] = "x" * 16385
    with pytest.raises(ValueError, match="prompt cap"): m.original_native_tokenizer("/original/BASE", contract, Loader)


def test_native_thread1_and_BASE_loader_source_composition_matches_frozen_V2():
    import ast
    old_tree = ast.parse((EXP / "source_v2/src/clasp/entrypoints/train_sft_anchor_pilot_v2.py").read_bytes())
    worker_tree = ast.parse(WORKER.read_bytes())
    main = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    native = next(node for node in worker_tree.body if isinstance(node, ast.FunctionDef) and node.name == "native_run")
    def sets_threads1(node):
        return [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and
            call.func.attr == "set_num_threads" and len(call.args) == 1 and isinstance(call.args[0], ast.Constant) and call.args[0].value == 1]
    assert len(sets_threads1(main)) == len(sets_threads1(native)) == 1
    assert ast.unparse(native).index("torch.set_num_threads(1)") < ast.unparse(native).index("kernel.run_probe(")
    assert 'original_native_tokenizer(core[\'spec\'][\'base\']' in ast.unparse(native)
    assert 'original_native_tokenizer(core[\'spec\'][\'sft\']' not in ast.unparse(native)


def test_adapter_bytes_identity_cannot_be_only_a_declared_inventory_field(packet):
    inventory, contract = initializer(packet)
    contract["initializer"]["sha256"] = "0" * 64
    inventory["initializer_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter bytes"):
        m.validate_initializer(inventory, spec=packet[2]["spec"], contract=contract,
            actual_roots={"base": FIXTURE / "fixture_base", "sft": FIXTURE / "fixture_sft"})
