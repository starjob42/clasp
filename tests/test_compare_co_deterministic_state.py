"""Real tiny checkpoint state comparisons at both diagnostic boundaries."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from clasp.models import classic_grpo_resume as classic
from clasp.models import exact_peft_kl_post_step as post
from test_classic_grpo_resume import bundle as classic_bundle  # noqa: F401
from test_co_deterministic_launcher import chunk as runtime_chunk  # noqa: F401
from test_foarl_teacher_resume import bundle as foarl_bundle  # noqa: F401
from test_foarl_teacher_resume import reseal_post, write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_co_deterministic_state.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("compare_co_state_tested", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def signed(path, value):
    value = {**value, "receipt_sha256": classic._digest(value)}
    write_json(path, value, readonly=True)
    return value


def reseal_classic(checkpoint, step):
    for name in (classic.POLICY_NAME, classic.LINEAGE_NAME):
        (checkpoint / name).unlink(missing_ok=True)
    path = checkpoint.parent.parent / "local_development_run_contract.json"
    return classic.seal_classic_grpo_checkpoint(checkpoint, expected_step=step,
        expected_contract=json.loads(path.read_text()), run_contract_path=path)


@pytest.fixture(params=[(m, s) for m in compare.METHODS for s in (5, 10)])
def pair(request, tmp_path, monkeypatch):
    method, step = request.param
    base = request.getfixturevalue("classic_bundle" if method == "classic_grpo" else "foarl_bundle")
    old, contract_path, contract = base[:3]
    checkpoint = old.with_name(f"checkpoint-{step}")
    old.rename(checkpoint)
    contract.update(method=method, diagnostic=True, teacher_weight=.05 if method == "foarl_teacher" else 0.)
    contract["schedule"]["save_steps"] = 5
    contract_path.chmod(0o640); write_json(contract_path, contract, readonly=True)
    write_json(checkpoint / "trainer_state.json", {"global_step": step, "max_steps": 400, "epoch": .01})
    optimizer = {"state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(1)}},
                 "param_groups": [{"params": [0], "lr": 1e-6}]}
    torch.save(optimizer, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": step, "_step_count": step + 1, "_last_lr": [1e-6]}, checkpoint / "scheduler.pt")
    for rank in range(4):
        torch.save({"python": random.Random(rank).getstate(), "numpy": np.random.RandomState(rank).get_state(),
                    "cpu": torch.Generator().manual_seed(rank).get_state(),
                    "cuda": [torch.Generator().manual_seed(rank * 4 + gpu).get_state() for gpu in range(4)]},
                   checkpoint / f"rng_state_{rank}.pth")
    if method == "classic_grpo":
        policy_path = checkpoint.parent / "policy_consistency_checks.json"
        policy = json.loads(policy_path.read_text()); policy.pop("receipt_sha256")
        policy["verified_optimizer_steps"] = [step]
        policy["checks"][0]["optimizer_step"] = step
        policy["checks"][0]["distributed"]["phase"] = f"optimizer_step_{step}_policy_consistency"
        signed(policy_path, policy)
    else:
        values = base[3]
        values["global_step"] = step
        values["training_run_contract"] = post.training_run_contract_identity(contract_path, claim_scope="development")
        p = checkpoint / post.POST_STEP_RECEIPT_FILENAME
        p.unlink(); write_json(p, reseal_post(values), readonly=True)
    second = tmp_path / "staged"
    shutil.copytree(checkpoint.parent.parent, second)
    roots = (checkpoint, second / "checkpoints" / f"checkpoint-{step}")
    for root in roots:
        if method == "classic_grpo": reseal_classic(root, step)
        run = root.parent.parent
        signed(run / "deterministic_runtime_declaration.json", {
            "profile_id": "synthetic_deterministic_runtime_boundary", "method": method,
            "source": contract["source"], "requirements": {"strict": True},
            "run_contract": classic._identity(run / "local_development_run_contract.json")})
        signed(root / "co_deterministic_runtime_evidence.json", {"step": step, "strict": True})

    # Runtime snapshot validation has its own real producer/helper tests. Here
    # retain actual runtime-file identities so post-load mutations remain tested.
    def runtime_inputs(root, contract, expected_step):
        return {"declaration": compare._load_json(root.parent.parent / "deterministic_runtime_declaration.json"),
                "declaration_file": classic._identity(root.parent.parent / "deterministic_runtime_declaration.json"),
                "applied_runtime": compare._load_json(root / "co_deterministic_runtime_evidence.json"),
                "applied_runtime_file": classic._identity(root / "co_deterministic_runtime_evidence.json")}
    monkeypatch.setattr(compare, "_runtime_inputs", runtime_inputs)
    return roots, method, step


def test_all_methods_and_both_boundaries_compare_actual_saved_values(pair):
    roots, method, step = pair
    result = compare.compare_checkpoints(*roots, expected_step=step)
    assert result["status"] == compare.SUCCESS[step] and result["method"] == method
    assert result["differing_field_count"] == 0 and result["compared_leaf_count"] > 2500
    assert result["parameter_rng_values_exported"] is False
    assert result["live_loaded_optimizer_scheduler_rng_restoration_verified"] is False
    assert compare.compare_checkpoints(*(p.parent.parent for p in roots), expected_step=step) == result


def test_old_checkpoint_is_independent_of_later_root_telemetry(pair):
    roots, _, step = pair
    expected = compare.compare_checkpoints(*roots, expected_step=step)
    for root in roots:
        path = root.parent / "policy_consistency_checks.json"
        if path.exists(): path.chmod(0o640)
        write_json(path, {"later_step": 10, "not_endpoint_policy": True})
        write_json(root.parent.parent / "hyperparams.json", {"later_runtime": True})
    assert compare.compare_checkpoints(*roots, expected_step=step) == expected


def test_different_rng_serialization_with_equal_saved_state_is_admitted(pair):
    roots, method, step = pair
    path = roots[1] / "rng_state_1.pth"
    original_bytes = path.read_bytes()
    state = torch.load(path, map_location="cpu", weights_only=False)
    # The legacy container stores the same state with different framing bytes.
    # File hashes remain independently bound, but are not semantic equality.
    torch.save(state, path, _use_new_zipfile_serialization=False)
    assert path.read_bytes() != original_bytes
    if method == "classic_grpo":
        reseal_classic(roots[1], step)
    result = compare.compare_checkpoints(*roots, expected_step=step)
    assert result["status"] == compare.SUCCESS[step]
    assert result["differing_fields"] == []


@pytest.mark.parametrize("component", ["optimizer", "python", "numpy", "cpu", "cuda"])
def test_all_state_components_detect_differences_without_values(pair, component):
    roots, method, step = pair
    name = "optimizer.pt" if component == "optimizer" else "rng_state_3.pth"
    path = roots[1] / name
    state = torch.load(path, map_location="cpu", weights_only=False)
    if component == "optimizer": state["state"][0]["exp_avg"].fill_(123456.125)
    elif component == "python": state["python"] = random.Random(7711).getstate()
    elif component == "numpy": state["numpy"][1][0] ^= np.uint32(1)
    elif component == "cpu": state["cpu"][0] ^= 1
    else: state["cuda"][2][0] ^= 1
    torch.save(state, path)
    if method == "classic_grpo": reseal_classic(roots[1], step)
    result = compare.compare_checkpoints(*roots, expected_step=step)
    assert result["status"] == "failed_state_difference" and result["differing_field_count"] > 0
    assert "123456.125" not in json.dumps(result)


@pytest.mark.parametrize("kind", ["optimizer_step", "scheduler_step", "empty_rng"])
def test_matching_but_wrong_boundary_or_incomplete_states_fail(pair, kind):
    roots, method, step = pair
    for root in roots:
        name = {"optimizer_step": "optimizer.pt", "scheduler_step": "scheduler.pt", "empty_rng": "rng_state_0.pth"}[kind]
        state = torch.load(root / name, map_location="cpu", weights_only=False)
        if kind == "optimizer_step": state["state"][0]["step"] = 10 if step == 5 else 5
        elif kind == "scheduler_step": state["_step_count"] = step
        else: state["cpu"] = torch.empty(0, dtype=torch.uint8)
        torch.save(state, root / name)
        if method == "classic_grpo": reseal_classic(root, step)
    with pytest.raises(ValueError): compare.compare_checkpoints(*roots, expected_step=step)


def test_both_complete_checkpoint_and_runtime_admissions_precede_pickle(pair, monkeypatch):
    roots, _, step = pair
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unadmitted state loaded"))
    (roots[1] / "rng_state_3.pth").unlink()
    with pytest.raises(FileNotFoundError): compare.compare_checkpoints(*roots, expected_step=step)


def test_runtime_admission_failure_also_precedes_pickle(pair, monkeypatch):
    roots, _, step = pair
    def reject(*args): raise ValueError("synthetic runtime gate")
    monkeypatch.setattr(compare, "_runtime_inputs", reject)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unadmitted runtime state loaded"))
    with pytest.raises(ValueError, match="runtime gate"): compare.compare_checkpoints(*roots, expected_step=step)


def test_changed_runtime_evidence_during_comparison_is_rejected(pair, monkeypatch):
    roots, _, step = pair
    load = torch.load
    changed = False
    def changing_load(*args, **kwargs):
        nonlocal changed
        if not changed:
            path = roots[1] / "co_deterministic_runtime_evidence.json"
            path.chmod(0o640); write_json(path, {"changed_after_admission": True})
            changed = True
        return load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", changing_load)
    with pytest.raises(ValueError, match="changed during"): compare.compare_checkpoints(*roots, expected_step=step)


def test_different_declared_source_or_method_fails_before_deserialization(pair, monkeypatch):
    roots, _, step = pair
    path = roots[1].parent.parent / "local_development_run_contract.json"
    contract = json.loads(path.read_text()); contract["source"]["manifest_sha256"] = "0" * 64
    path.chmod(0o640); write_json(path, contract, readonly=True)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("different contract loaded"))
    with pytest.raises(ValueError, match="contracts differ"): compare.compare_checkpoints(*roots, expected_step=step)


def test_runtime_adapter_uses_checkpoint_local_evidence(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints/checkpoint-5"; root.mkdir(parents=True)
    declaration = signed(tmp_path / "deterministic_runtime_declaration.json", {"synthetic": True})
    proof = signed(root / "co_deterministic_runtime_evidence.json", {"saved_step": 5})
    calls = []
    module = SimpleNamespace(
        read_only_runtime_declaration=lambda path, contract: calls.append(("declaration", path)) or declaration,
        read_only_runtime_evidence=lambda path, contract: calls.append(("evidence", path)) or proof)
    monkeypatch.setitem(sys.modules, "run_co_deterministic_dev_chunk", module)
    result = compare._runtime_inputs(root, {}, 5)
    assert calls == [("declaration", tmp_path), ("evidence", root)]
    assert result["applied_runtime_file"] == classic._identity(root / "co_deterministic_runtime_evidence.json")


@pytest.mark.parametrize("method", compare.METHODS)
@pytest.mark.parametrize("saved_step", [5, 10])
def test_runtime_adapter_admits_real_producer_proofs_and_rejects_rank_drift(
        request, monkeypatch, method, saved_step):
    # The launcher fixture substitutes machine/training interfaces only. The
    # actual runner creates and revalidates all declaration, snapshot, admission,
    # exit and four-rank proof files consumed by this comparator adapter.
    setup = request.getfixturevalue("runtime_chunk")
    runner = setup.module
    monkeypatch.setitem(sys.modules, "run_co_deterministic_dev_chunk", runner)
    os.environ.update(CO_METHOD=method, CO_DIAGNOSTIC="1", CO_STOP="10")
    runner.main()
    checkpoint = setup.output / "checkpoints" / f"checkpoint-{saved_step}"
    contract = runner.read_json(setup.output / "local_development_run_contract.json")
    admitted = compare._runtime_inputs(checkpoint, contract, saved_step)
    proof = admitted["applied_runtime"]
    assert proof["checkpoint_step"] == saved_step and proof["producer_stop"] == 10
    assert [item["rank"] for item in proof["postrun_per_rank_checks"]] == list(range(4))
    assert admitted["applied_runtime_file"] == classic._identity(checkpoint / runner.EVIDENCE_NAME)
    # A full run's saved5 remains valid after root metadata has advanced.
    (setup.output / "hyperparams.json").write_text('{"later_root_metadata":true}')
    assert compare._runtime_inputs(checkpoint, contract, saved_step) == admitted
    path = setup.output / "postrun_determinism_step10_rank2.json"
    record = runner.read_json(path)
    record.pop("receipt_sha256")
    record["training_flags"]["deterministic_warn_only"] = True
    path.chmod(0o640)
    write_json(path, runner.signed(record), readonly=True)
    with pytest.raises(ValueError, match="flags/version differ"):
        compare._runtime_inputs(checkpoint, contract, saved_step)


def test_optimizer_check_does_not_rewrite_original_values():
    optimizer = {"state": {0: {"step": torch.tensor(5.), "exp_avg": torch.tensor([2.])}},
                 "param_groups": [{"params": [0], "lr": 1e-6}]}
    scheduler = {"last_epoch": 5, "_step_count": 6, "_last_lr": [1e-6]}
    original = copy.deepcopy((optimizer, scheduler))
    compare._check_optimizer(optimizer, scheduler, 5)
    oracle = compare.Comparison(); oracle.compare(original, (optimizer, scheduler), "unchanged")
    assert oracle.differences == []


def test_cli_suppresses_nested_state_diagnostics_and_is_create_only(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        print("PRIVATE_STATE_VALUE_123456")
        print("PRIVATE_STATE_VALUE_123456", file=sys.stderr)
        raise ValueError("PRIVATE_STATE_VALUE_123456")
    monkeypatch.setattr(compare, "compare_checkpoints", fail)
    output = tmp_path / "result.json"
    argv = ["--uninterrupted", "/first", "--resumed", "/second", "--step", "5", "--output", str(output)]
    assert compare.main(argv) == 1
    captured = capsys.readouterr()
    assert "PRIVATE" not in output.read_text() + captured.out + captured.err
    assert json.loads(output.read_text())["status"] == "failed_admission_or_comparison"
    with pytest.raises(FileExistsError): compare.main(argv)
