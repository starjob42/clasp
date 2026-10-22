"""Early rank divergence must fail with the original exact policy hash guard."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from clasp.models import training_fidelity as training


@pytest.fixture(autouse=True)
def clean_modes(monkeypatch):
    for key in (
        training.PUBLICATION_FIDELITY_ENV,
        training.DEVELOPMENT_FIDELITY_ENV,
        training.POLICY_CONSISTENCY_CHECKS_ENV,
        training.POLICY_CONSISTENCY_CHECK_INTERVAL_ENV,
        training.GRADIENT_CONSISTENCY_CHECKS_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")


def enable(monkeypatch, interval="0"):
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    monkeypatch.setenv(training.POLICY_CONSISTENCY_CHECKS_ENV, "1")
    monkeypatch.setenv(training.POLICY_CONSISTENCY_CHECK_INTERVAL_ENV, interval)


def args_state(tmp_path, step=1, rank=0):
    return (
        SimpleNamespace(output_dir=str(tmp_path), should_save=rank == 0),
        SimpleNamespace(global_step=step, is_world_process_zero=rank == 0),
    )


def install_small_closure(monkeypatch):
    record = {
        "key": "adapter.lora_B.weight", "shape": [1, 1], "dtype": "F32",
        "numel": 1, "value_sha256": "a" * 64,
    }
    closure = {"tensor_inventory_sha256": "b" * 64, "tensors": [record]}
    monkeypatch.setattr(training, "audit_trainable_state_closure", lambda model, name: closure)
    monkeypatch.setattr(
        training, "assert_qwen25_7b_publication_invariant",
        lambda model, observed, name: {"status": "exact"},
    )
    return SimpleNamespace(model=object()), closure


def read_receipt(tmp_path):
    receipt = json.loads((tmp_path / training.POLICY_CONSISTENCY_RECEIPT_FILENAME).read_text())
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == training._sha256_json(unsigned)
    return receipt


def test_default_off_preserves_beta_zero_callback_without_extra_checks(monkeypatch, tmp_path):
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    monkeypatch.setattr(
        training, "run_publication_fidelity_preflight",
        lambda *args, **kwargs: pytest.fail("default-off callback must not audit optimizer steps"),
    )
    callback = training.PublicationFidelityCheckpointCallback(SimpleNamespace())
    args, state = args_state(tmp_path)
    control = object()
    assert callback.on_step_end(args, state, control) is control
    assert not (tmp_path / training.POLICY_CONSISTENCY_RECEIPT_FILENAME).exists()


def test_opt_in_requires_fidelity_before_callback_installation(monkeypatch):
    monkeypatch.setenv(training.POLICY_CONSISTENCY_CHECKS_ENV, "1")
    with pytest.raises(training.PublicationFidelityError, match="requires development or publication"):
        training.install_publication_fidelity_callback(SimpleNamespace())


@pytest.mark.parametrize("interval", ["-1", "1.5", "true", "", "many"])
def test_bad_enabled_interval_fails_before_training(monkeypatch, interval):
    enable(monkeypatch, interval)
    with pytest.raises(training.PublicationFidelityError, match="integer >= 0"):
        training.PublicationFidelityCheckpointCallback(SimpleNamespace())


def test_early_and_periodic_checks_run_for_beta_zero_and_write_exact_evidence(monkeypatch, tmp_path):
    enable(monkeypatch, "4")
    trainer, closure = install_small_closure(monkeypatch)
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args, state = args_state(tmp_path)
    control = object()
    for step in range(1, 10):
        state.global_step = step
        assert callback.on_step_end(args, state, control) is control
    # Duplicate callback delivery must not add a second record.
    state.global_step = 8
    callback.on_step_end(args, state, control)
    receipt = read_receipt(tmp_path)
    assert receipt["status"] == "passed"
    assert receipt["verified_optimizer_steps"] == [1, 2, 4, 5, 8]
    assert receipt["claim_scope"] == "development"
    for record in receipt["checks"]:
        assert record["tensor_inventory_sha256"] == closure["tensor_inventory_sha256"]
        agreement = record["distributed"]
        assert agreement["world_size"] == 1
        assert agreement["rank_digests"][0]["digest"] == agreement["consensus_digest"]
    assert trainer._clasp_policy_consistency_checks == receipt


def test_rank_disagreement_aborts_at_step_two_and_retains_named_tensor_evidence(monkeypatch, tmp_path):
    import torch.distributed as dist

    enable(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    trainer, _ = install_small_closure(monkeypatch)
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args, state = args_state(tmp_path)

    def gather(values, local):
        remote = copy.deepcopy(local)
        remote["rank"] = 1
        if state.global_step == 2:
            if "digest" in remote:
                remote["digest"] = "f" * 64
            else:
                remote["tensor_inventory_sha256"] = "c" * 64
                remote["tensors"][0]["value_sha256"] = "d" * 64
        values[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    callback.on_step_end(args, state, object())
    state.global_step = 2
    with pytest.raises(training.PublicationFidelityError, match="differs across ranks during optimizer_step_2"):
        callback.on_step_end(args, state, object())
    receipt = read_receipt(tmp_path)
    assert receipt["status"] == "failed"
    assert receipt["verified_optimizer_steps"] == [1]
    assert receipt["failed_optimizer_step"] == 2
    details = receipt["policy_mismatch_details"]
    assert details["world_size"] == 2
    assert details["differing_tensor_count"] == 1
    mismatch = details["differing_tensors"][0]
    assert mismatch["key"] == "adapter.lora_B.weight"
    assert [r["tensor"]["dtype"] for r in mismatch["rank_records"]] == ["F32", "F32"]
    assert [r["tensor"]["value_sha256"] for r in mismatch["rank_records"]] == ["a" * 64, "d" * 64]


def test_nonzero_rank_checks_but_does_not_write_receipt(monkeypatch, tmp_path):
    enable(monkeypatch)
    trainer, _ = install_small_closure(monkeypatch)
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args, state = args_state(tmp_path, rank=1)
    callback.on_step_end(args, state, object())
    assert trainer._clasp_policy_consistency_checks["verified_optimizer_steps"] == [1]
    assert not (tmp_path / training.POLICY_CONSISTENCY_RECEIPT_FILENAME).exists()


def test_gradient_diagnostic_requires_policy_guard(monkeypatch):
    monkeypatch.setenv(training.DEVELOPMENT_FIDELITY_ENV, "1")
    monkeypatch.setenv(training.GRADIENT_CONSISTENCY_CHECKS_ENV, "1")
    with pytest.raises(training.PublicationFidelityError, match="requires CLASP_POLICY_CONSISTENCY_CHECKS"):
        training.install_publication_fidelity_callback(SimpleNamespace())


def test_gradient_inventory_records_absent_gradients_without_changing_tensors():
    import torch

    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[0.5, -0.25]])
    before_parameter = model.weight.detach().clone()
    before_gradient = model.weight.grad.clone()
    inventory = training._audit_trainable_gradient_inventory(SimpleNamespace(model=model))
    records = {record["key"]: record for record in inventory["tensors"]}
    assert records["weight"]["gradient_present"] is True
    assert records["weight"]["dtype"] == "F32"
    assert records["bias"]["gradient_present"] is False
    assert records["bias"]["value_sha256"] is None
    torch.testing.assert_close(model.weight, before_parameter, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, before_gradient, rtol=0, atol=0)
    model.weight.grad[0, 0] = 0.75
    changed = training._audit_trainable_gradient_inventory(SimpleNamespace(model=model))
    assert changed["tensor_inventory_sha256"] != inventory["tensor_inventory_sha256"]


def test_gradient_diagnostic_uses_upcoming_step_and_separate_receipt(monkeypatch, tmp_path):
    import torch

    enable(monkeypatch)
    monkeypatch.setenv(training.GRADIENT_CONSISTENCY_CHECKS_ENV, "1")
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.ones_like(model.weight)
    trainer = SimpleNamespace(model=model)
    callback = training.PublicationFidelityCheckpointCallback(trainer)
    args, state = args_state(tmp_path, step=0)
    control = object()
    assert callback.on_pre_optimizer_step(args, state, control) is control
    receipt = json.loads((tmp_path / training.GRADIENT_CONSISTENCY_RECEIPT_FILENAME).read_text())
    assert receipt["protocol"] == "clasp-pre-optimizer-exact-gradient-consistency-v1"
    assert receipt["verified_gradient_optimizer_steps"] == [1]
    assert receipt["checks"][0]["gradient_tensor_count"] == 1
    assert not (tmp_path / training.POLICY_CONSISTENCY_RECEIPT_FILENAME).exists()
    assert trainer._clasp_gradient_consistency_checks == receipt


def test_gradient_disagreement_fails_before_update_with_tensor_diagnostics(monkeypatch, tmp_path):
    import torch
    import torch.distributed as dist

    enable(monkeypatch)
    monkeypatch.setenv(training.GRADIENT_CONSISTENCY_CHECKS_ENV, "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def gather(values, local):
        remote = copy.deepcopy(local)
        remote["rank"] = 1
        if "digest" in remote:
            remote["digest"] = "f" * 64
        else:
            remote["tensors"][0]["value_sha256"] = "d" * 64
        values[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    before_parameter, before_gradient = model.weight.detach().clone(), model.weight.grad.clone()
    callback = training.PublicationFidelityCheckpointCallback(SimpleNamespace(model=model))
    args, state = args_state(tmp_path, step=0)
    with pytest.raises(training.PublicationFidelityError, match="before_optimizer_step_1_gradient"):
        callback.on_pre_optimizer_step(args, state, object())
    receipt = json.loads((tmp_path / training.GRADIENT_CONSISTENCY_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "failed"
    assert receipt["failed_optimizer_step"] == 1
    assert receipt["verified_gradient_optimizer_steps"] == []
    assert receipt["policy_mismatch_details"]["differing_tensors"][0]["key"] == "weight"
    torch.testing.assert_close(model.weight, before_parameter, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, before_gradient, rtol=0, atol=0)
