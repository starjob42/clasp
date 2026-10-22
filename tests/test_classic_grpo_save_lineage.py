"""Save-time lineage, callback ordering and real four-rank failure publication."""
# ruff: noqa: F811 - pytest injects the imported bundle fixture into test arguments
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from clasp.models import classic_grpo_resume as resume
from clasp.models import classic_grpo_save_lineage as lineage
from test_classic_grpo_resume import bundle, write  # noqa: F401


@pytest.fixture
def collectives(monkeypatch):
    state = {"events": [], "bad_rank": None}
    monkeypatch.setattr(lineage, "validate_strict_determinism", lambda: {})
    monkeypatch.setattr(lineage.dist, "is_available", lambda: True)
    monkeypatch.setattr(lineage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lineage.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(lineage.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lineage.dist, "barrier", lambda: state["events"].append("barrier"))
    def gather(values, local):
        state["events"].append("gather")
        values[:] = [{**local, "rank": rank} for rank in range(4)]
        if state["bad_rank"] is not None:
            values[state["bad_rank"]]["error"] = "ValueError"
    def broadcast(values, src):
        assert src == 0
        state["events"].append(("broadcast", copy.deepcopy(values[0])))
    monkeypatch.setattr(lineage.dist, "all_gather_object", gather)
    monkeypatch.setattr(lineage.dist, "broadcast_object_list", broadcast)
    return state


def invocation(checkpoint, contract_path, step, monkeypatch):
    monkeypatch.setenv("CLASP_TRAINING_RUN_CONTRACT_PATH", str(contract_path))
    return (SimpleNamespace(output_dir=str(checkpoint.parent), max_steps=400),
            SimpleNamespace(global_step=step, max_steps=400, is_world_process_zero=True), object())


def set_diagnostic_step(checkpoint, contract_path, contract, step):
    changed = checkpoint.with_name(f"checkpoint-{step}")
    checkpoint.rename(changed)
    contract["diagnostic"] = True
    contract["schedule"]["save_steps"] = 5
    contract_path.chmod(0o640)
    write(contract_path, contract, True)
    write(changed / "trainer_state.json", {"global_step": step, "max_steps": 400})
    policy_path = changed.parent / "policy_consistency_checks.json"
    policy = json.loads(policy_path.read_text())
    policy["verified_optimizer_steps"] = [step]
    policy["checks"][0]["optimizer_step"] = step
    policy["checks"][0]["distributed"]["phase"] = f"optimizer_step_{step}_policy_consistency"
    policy.pop("receipt_sha256")
    policy["receipt_sha256"] = resume._digest(policy)
    write(policy_path, policy)
    return changed


def test_real_step5_fidelity_and_current_report_are_sealed_before_later_report(bundle, collectives, monkeypatch):
    checkpoint, contract_path, contract = bundle
    checkpoint = set_diagnostic_step(checkpoint, contract_path, contract, 5)
    original_report = (checkpoint.parent / "policy_consistency_checks.json").read_bytes()
    original_files = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    rng = torch.random.get_rng_state().clone()
    callback = lineage.ClassicSaveLineageCallback(object())
    args, state, control = invocation(checkpoint, contract_path, 5, monkeypatch)
    assert callback.on_save(args, state, control) is control
    assert callback.saved_steps == [5]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert {n: (checkpoint / n).read_bytes() for n in original_files} == original_files
    assert (checkpoint / resume.POLICY_NAME).read_bytes() == original_report
    receipt = resume.validate_classic_grpo_resume(checkpoint, expected_step=5,
                    expected_contract=contract, run_contract_path=contract_path)
    # Step10 telemetry may replace the root report without changing saved5.
    write(checkpoint.parent / "policy_consistency_checks.json", {"later_step": 10})
    assert resume.validate_classic_grpo_resume(checkpoint, expected_step=5,
                    expected_contract=contract, run_contract_path=contract_path) == receipt
    assert collectives["events"][:2] == ["barrier", "gather"]
    assert collectives["events"][-2:] == [("broadcast", {"status": "sealed_provisional_save", "step": 5}), "barrier"]


@pytest.mark.parametrize("failure", ["missing_rng", "missing_fidelity", "wrong_policy_step", "other_rank", "duplicate"])
def test_failed_save_publishes_failure_and_finishes_barrier_before_any_return(bundle, collectives, monkeypatch, failure):
    checkpoint, contract_path, contract = bundle
    callback = lineage.ClassicSaveLineageCallback(object())
    args, state, control = invocation(checkpoint, contract_path, 100, monkeypatch)
    if failure == "missing_rng":
        (checkpoint / "rng_state_3.pth").unlink()
    elif failure == "missing_fidelity":
        (checkpoint / "checkpoint_fidelity.json").unlink()
    elif failure == "wrong_policy_step":
        write(checkpoint.parent / "policy_consistency_checks.json", {"later_step": 200})
    elif failure == "other_rank":
        collectives["bad_rank"] = 2
    else:
        callback.saved_steps.append(100)
    with pytest.raises(RuntimeError, match="provisional save failed"):
        callback.on_save(args, state, control)
    assert collectives["events"][-1] == "barrier"
    assert collectives["events"][-2][1]["status"] == "failed_provisional_save"
    assert not (checkpoint / resume.LINEAGE_NAME).exists()


def test_callback_requires_four_rank_group_before_any_save(bundle, monkeypatch):
    checkpoint, contract_path, contract = bundle
    monkeypatch.setattr(lineage.dist, "is_initialized", lambda: False)
    callback = lineage.ClassicSaveLineageCallback(object())
    with pytest.raises(RuntimeError, match="four-rank"):
        callback.on_save(*invocation(checkpoint, contract_path, 100, monkeypatch))
    assert not (checkpoint / resume.LINEAGE_NAME).exists()


@pytest.fixture
def installation(monkeypatch):
    class Fidelity:
        def __init__(self, trainer):
            self._trainer = trainer
    monkeypatch.setattr(lineage, "PublicationFidelityCheckpointCallback", Fidelity)
    callbacks = []
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=callbacks), add_callback=callbacks.append)
    calls = []
    def original(value):
        calls.append(value)
        value.add_callback(Fidelity(value))
        return True
    monkeypatch.setattr(lineage, "install_publication_fidelity_callback", original)
    module = SimpleNamespace(install_publication_fidelity_callback=original)
    return module, trainer, original, callbacks, calls


def test_scoped_installer_delegates_once_then_appends_after_fidelity_and_restores(installation):
    module, trainer, original, callbacks, calls = installation
    with lineage.install_classic_save_lineage(module, "classic_grpo"):
        assert module.install_publication_fidelity_callback is not original
        assert module.install_publication_fidelity_callback(trainer) is True
        assert type(callbacks[0]) is lineage.PublicationFidelityCheckpointCallback
        assert type(callbacks[1]) is lineage.ClassicSaveLineageCallback
        assert callbacks[1].trainer is trainer
        with pytest.raises(ValueError, match="exactly once"):
            module.install_publication_fidelity_callback(trainer)
    assert module.install_publication_fidelity_callback is original
    assert calls == [trainer]


@pytest.mark.parametrize("method", ["foarl_control", "foarl_teacher"])
def test_foarl_installation_is_completely_unchanged(installation, method):
    module, trainer, original, callbacks, calls = installation
    with lineage.install_classic_save_lineage(module, method):
        assert module.install_publication_fidelity_callback is original
        module.install_publication_fidelity_callback(trainer)
    assert len(callbacks) == 1 and calls == [trainer]


def test_installer_restores_on_training_failure_and_rejects_missing_install(installation):
    module, trainer, original, callbacks, calls = installation
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        with lineage.install_classic_save_lineage(module, "classic_grpo"):
            module.install_publication_fidelity_callback(trainer)
            raise RuntimeError("synthetic training failure")
    assert module.install_publication_fidelity_callback is original
    with pytest.raises(ValueError, match="never installed"):
        with lineage.install_classic_save_lineage(module, "classic_grpo"):
            pass
    assert module.install_publication_fidelity_callback is original


def test_installer_rejects_wrong_order_and_disabled_fidelity(installation, monkeypatch):
    module, trainer, original, callbacks, calls = installation
    for result in (False, True):
        def broken(value):
            value.add_callback(object())
            return result
        module.install_publication_fidelity_callback = broken
        monkeypatch.setattr(lineage, "install_publication_fidelity_callback", broken)
        with pytest.raises(ValueError, match="original enabled fidelity"):
            with lineage.install_classic_save_lineage(module, "classic_grpo"):
                module.install_publication_fidelity_callback(trainer)
        assert module.install_publication_fidelity_callback is broken


def _distributed_worker(rank, rendezvous, output, contract_path, mode):
    distributed.init_process_group("gloo", rank=rank, world_size=4,
                                   init_method="file://" + rendezvous, timeout=dt.timedelta(seconds=30))
    try:
        lineage.validate_strict_determinism = lambda: {}
        os.environ["CLASP_TRAINING_RUN_CONTRACT_PATH"] = contract_path
        def seal(checkpoint, **kwargs):
            assert rank == 0
            if mode == "seal_failure":
                raise OSError("synthetic save failure")
            with (Path(output) / "only_rank_zero_sealed").open("x") as handle:
                handle.write(str(kwargs["expected_step"]))
        lineage.seal_classic_grpo_checkpoint = seal
        callback = lineage.ClassicSaveLineageCallback(object())
        state = SimpleNamespace(global_step=6 if mode == "rank2_failure" and rank == 2 else 5,
                                max_steps=400, is_world_process_zero=rank == 0)
        args = SimpleNamespace(output_dir=str(Path(output) / "checkpoints"), max_steps=400)
        before = torch.random.get_rng_state().clone()
        try:
            callback.on_save(args, state, object())
            status = "success"
        except RuntimeError:
            status = "failed_on_all_ranks"
        report = {"status": status, "saved_steps": callback.saved_steps,
                  "rng_unchanged": torch.equal(before, torch.random.get_rng_state())}
        (Path(output) / f"rank{rank}.json").write_text(json.dumps(report))
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["success", "seal_failure", "rank2_failure"])
def test_real_four_rank_gloo_success_or_failure_reaches_every_rank_without_rng_draws(tmp_path, mode):
    output = tmp_path / mode
    (output / "checkpoints").mkdir(parents=True)
    contract_path = output / "contract.json"
    contract_path.write_text(json.dumps({"method": "classic_grpo", "beta": 0, "teacher_weight": 0,
                                        "diagnostic": True, "schedule": {"save_steps": 5}}))
    multiprocessing.start_processes(_distributed_worker,
        args=(str(tmp_path / (mode + ".rendezvous")), str(output), str(contract_path), mode),
        nprocs=4, join=True, start_method="fork")
    expected = {"status": "success" if mode == "success" else "failed_on_all_ranks",
                "saved_steps": [5] if mode == "success" else [], "rng_unchanged": True}
    assert [json.loads((output / f"rank{rank}.json").read_text()) for rank in range(4)] == [expected] * 4
    assert (output / "only_rank_zero_sealed").exists() == (mode == "success")
