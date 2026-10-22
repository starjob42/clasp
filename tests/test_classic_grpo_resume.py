"""Beta-zero lineage admission using real tiny safetensors/fidelity bundles."""
from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.models import checkpoint_fidelity as fidelity
from clasp.models import classic_grpo_resume as resume
from test_checkpoint_fidelity import _closure, _record, _write_adapter


def write(path, value, readonly=False):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    if readonly:
        path.chmod(0o440)


def adapter(path, value):
    key = "base_model.model.layer.lora_A.weight"
    encoded = struct.pack("<f", value)
    _write_adapter(path, {key: ("F32", [1], encoded)})
    closure = _closure([_record(key, "F32", [1], encoded)])
    fidelity.write_checkpoint_fidelity_receipt(path, closure)
    return closure


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # Tiny real safetensors exercise serialization/fidelity. Only the separate
    # 7B structural boundary is stubbed, avoiding a 600MB synthetic adapter.
    monkeypatch.setattr(resume, "_qwen_publication_invariant",
                        lambda config, tensors: {"status": "synthetic_qwen_boundary",
                                                 "tensor_count": len(tensors)})
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    checkpoint = output / "checkpoints/checkpoint-100"
    closure = adapter(checkpoint, 2.)
    sft = tmp_path / "sft"
    adapter(sft, 1.)
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("synthetic source declaration\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.json").write_text('[{"synthetic":true}]\n')
    contract = {"schema_version": 1, "stage": "grpo", "status": "development_only_not_publication_eligible",
                "method": "classic_grpo", "source": {"root": str(source), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "initialization": {"checkpoint": str(sft), "files": {n: resume._identity(sft / n) for n in
                    ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json")}},
                "fixed_horizon": {"max_steps": 400, "world_size": 4}, "teacher_weight": 0., "beta": 0., "seed": 43,
                "data_root": str(data), "data_files": {"train.json": resume._identity(data / "train.json")},
                "schedule": {"generation_batch_size": 36, "gradient_accumulation_steps": 3, "steps_per_generation": 3, "save_steps": 100},
                "diagnostic": False}
    contract_path = output / "local_development_run_contract.json"
    write(contract_path, contract, True)
    write(checkpoint / "trainer_state.json", {"global_step": 100, "max_steps": 400})
    # No pickle is executed by this admission test. Native equality tests own
    # deserialization and loaded optimizer/scheduler/RNG semantic checks.
    for name in ("optimizer.pt", "scheduler.pt", *(f"rng_state_{r}.pth" for r in range(4))):
        (checkpoint / name).write_bytes(b"opaque synthetic state\n")
    digest = closure["tensor_inventory_sha256"]
    composite = resume._digest({"closure": closure, "qwen_structural_invariant":
        resume._qwen_publication_invariant({}, closure["tensors"]), "claim_scope": "development_only"})
    check = {"optimizer_step": 100, "tensor_inventory_sha256": digest,
             "distributed": {"phase": "optimizer_step_100_policy_consistency", "world_size": 4,
                 "rank_digests": [{"rank": r, "digest": composite} for r in range(4)], "consensus_digest": composite}}
    policy = {"schema_version": 1, "protocol": "clasp-optimizer-step-exact-policy-consistency-v1", "status": "passed",
              "claim_scope": "development", "settings": {}, "verified_optimizer_steps": [100], "checks": [check]}
    policy["receipt_sha256"] = resume._digest(policy)
    write(checkpoint.parent / "policy_consistency_checks.json", policy)
    return checkpoint, contract_path, contract


def call(bundle, seal=False):
    root, path, contract = bundle
    func = resume.seal_classic_grpo_checkpoint if seal else resume.validate_classic_grpo_resume
    return func(root, expected_step=100, expected_contract=contract, run_contract_path=path)


def test_real_fidelity_bundle_seals_complete_beta_zero_state_once(bundle):
    root = bundle[0]
    result = call(bundle, seal=True)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert call(bundle) == result
    assert result["no_kl_sidecars"] and result["all_rank_rng_files_present"]
    assert len(result["files"]) == 12
    assert result["live_optimizer_scheduler_rng_restoration_verified"] is False
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match="already sealed"):
        call(bundle, seal=True)


def test_checkpoint_policy_copy_survives_next_chunks_mutable_telemetry(bundle):
    expected = call(bundle, seal=True)
    write(bundle[0].parent / "policy_consistency_checks.json", {"later_chunk": True})
    assert call(bundle) == expected


@pytest.mark.parametrize("name", resume.STATE_NAMES)
def test_every_optimizer_scheduler_rng_and_policy_input_is_required(bundle, name):
    (bundle[0] / name).unlink()
    with pytest.raises((ValueError, OSError)):
        call(bundle, seal=True)


@pytest.mark.parametrize("name", resume.KL_NAMES)
@pytest.mark.parametrize("location", ["checkpoint", "checkpoints_parent", "run_root"])
def test_beta_zero_rejects_every_kl_sidecar_even_dangling_links(bundle, name, location):
    root = {"checkpoint": bundle[0], "checkpoints_parent": bundle[0].parent, "run_root": bundle[1].parent}[location]
    (root / name).symlink_to(root / "missing")
    with pytest.raises(ValueError, match="KL reference sidecars"):
        call(bundle, seal=True)


@pytest.mark.parametrize("mutation", ["rank", "digest", "step", "scope", "schema_bool"])
def test_self_signed_but_wrong_policy_consensus_is_rejected(bundle, mutation):
    path = bundle[0].parent / "policy_consistency_checks.json"
    p = json.loads(path.read_text())
    if mutation == "rank": p["checks"][0]["distributed"]["rank_digests"][0]["rank"] = False
    if mutation == "digest": p["checks"][0]["tensor_inventory_sha256"] = "0" * 64
    if mutation == "step": p["checks"][0]["optimizer_step"] = 99
    if mutation == "scope": p["claim_scope"] = "publication"
    if mutation == "schema_bool": p["schema_version"] = True
    p.pop("receipt_sha256"); p["receipt_sha256"] = resume._digest(p)
    write(path, p)
    with pytest.raises(ValueError): call(bundle, seal=True)


@pytest.mark.parametrize("what", ["optimizer", "rng", "contract", "source", "data", "copied_consensus", "lineage"])
def test_post_seal_mutation_is_rejected(bundle, what):
    call(bundle, seal=True)
    root, contract_path, contract = bundle
    path = {"optimizer": root / "optimizer.pt", "rng": root / "rng_state_3.pth", "contract": contract_path,
            "source": Path(contract["source"]["root"]) / "SOURCE_MANIFEST.sha256",
            "data": Path(contract["data_root"]) / "train.json", "copied_consensus": root / resume.POLICY_NAME,
            "lineage": root / resume.LINEAGE_NAME}[what]
    path.chmod(0o640)
    with path.open("ab") as stream: stream.write(b"drift")
    with pytest.raises((ValueError, OSError)):
        call(bundle)


def test_another_original_contract_cannot_admit_copied_lineage(bundle):
    call(bundle, seal=True)
    changed = copy.deepcopy(bundle[2]); changed["beta"] = .05
    with pytest.raises(ValueError, match="Original contract differs"):
        resume.validate_classic_grpo_resume(bundle[0], expected_step=100, expected_contract=changed, run_contract_path=bundle[1])


@pytest.mark.parametrize("kind", ["empty", "symlink"])
def test_opaque_inputs_must_be_nonempty_unlinked_files(bundle, kind):
    path = bundle[0] / "optimizer.pt"
    path.unlink()
    if kind == "empty": path.touch()
    else: path.symlink_to(bundle[0] / "scheduler.pt")
    with pytest.raises(ValueError): call(bundle, seal=True)


def actual_callback_receipt(bundle, monkeypatch):
    """Execute the real preflight, collective wrapper and callback writer."""
    import torch.distributed as dist

    from clasp.models import training_fidelity as training

    root = bundle[0]
    closure = fidelity.verify_checkpoint_fidelity(root)["trainable_closure"]
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    monkeypatch.setenv(training.PUBLICATION_FIDELITY_ENV, "0")
    monkeypatch.setenv(training.POLICY_CONSISTENCY_CHECKS_ENV, "1")
    monkeypatch.setenv(training.POLICY_CONSISTENCY_CHECK_INTERVAL_ENV, "100")
    monkeypatch.setenv(training.GRADIENT_CONSISTENCY_CHECKS_ENV, "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(training, "audit_trainable_state_closure", lambda model, name: closure)
    monkeypatch.setattr(training, "assert_qwen25_7b_publication_invariant",
                        lambda model, observed, name: resume._qwen_publication_invariant(
                            json.loads((root / "adapter_config.json").read_text()), observed["tensors"]))
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def gather(records, local):
        records[:] = [{**local, "rank": rank} for rank in range(4)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    trainer = SimpleNamespace(model=object())
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args = SimpleNamespace(output_dir=str(root.parent), should_save=True)
    state = SimpleNamespace(global_step=100, is_world_process_zero=True)
    callback.on_step_end(args, state, object())
    path = root.parent / "policy_consistency_checks.json"
    return path, json.loads(path.read_text()), closure


def test_actual_native_callback_composite_digest_seals_saved_policy(bundle, monkeypatch):
    _, receipt, _ = actual_callback_receipt(bundle, monkeypatch)
    record = receipt["checks"][0]
    assert record["distributed"]["consensus_digest"] != record["tensor_inventory_sha256"]
    result = call(bundle, seal=True)
    assert result["policy_tensor_inventory_sha256"] == record["tensor_inventory_sha256"]
    assert result["policy_preflight_consensus_sha256"] == record["distributed"]["consensus_digest"]
    assert call(bundle) == result


@pytest.mark.parametrize("mutation", ["tensor_only", "structure", "scope", "rank_disagreement"])
def test_actual_callback_composite_requires_structure_scope_and_all_ranks(bundle, monkeypatch, mutation):
    path, receipt, closure = actual_callback_receipt(bundle, monkeypatch)
    record = receipt["checks"][0]
    if mutation == "rank_disagreement":
        record["distributed"]["rank_digests"][3]["digest"] = "0" * 64
    else:
        payload = {"closure": closure, "qwen_structural_invariant":
                   resume._qwen_publication_invariant({}, closure["tensors"]), "claim_scope": "development_only"}
        if mutation == "structure": payload["qwen_structural_invariant"]["tensor_count"] += 1
        elif mutation == "scope": payload["claim_scope"] = "publication"
        digest = record["tensor_inventory_sha256"] if mutation == "tensor_only" else resume._digest(payload)
        record["distributed"]["consensus_digest"] = digest
        for rank in record["distributed"]["rank_digests"]: rank["digest"] = digest
    receipt.pop("receipt_sha256"); receipt["receipt_sha256"] = resume._digest(receipt)
    write(path, receipt)
    with pytest.raises(ValueError, match="Four-rank policy consensus"):
        call(bundle, seal=True)
    assert not (bundle[0] / resume.POLICY_NAME).exists()
    assert not (bundle[0] / resume.LINEAGE_NAME).exists()
