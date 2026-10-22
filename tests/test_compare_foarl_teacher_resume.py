"""Exact state comparison and value-free failure output for the resume probe."""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from clasp.models import exact_peft_kl_post_step as post
from test_foarl_teacher_resume import bundle, reseal_post, write_json  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_foarl_teacher_resume.py"
SPEC = importlib.util.spec_from_file_location("compare_foarl_teacher_resume_tested", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


@pytest.fixture
def pair(bundle, tmp_path):  # noqa: F811 - imported pytest fixture
    old, contract_path, contract, values = bundle
    checkpoint = old.with_name("checkpoint-10")
    old.rename(checkpoint)
    contract["diagnostic"] = True
    contract["schedule"]["save_steps"] = 5
    contract_path.chmod(0o640)
    write_json(contract_path, contract, readonly=True)
    write_json(checkpoint / "trainer_state.json", {"global_step": 10, "max_steps": 400})
    optimizer = {"state": {0: {"step": torch.tensor(10.), "exp_avg": torch.ones(1)}},
                 "param_groups": [{"params": [0], "lr": 1e-6}]}
    torch.save(optimizer, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 10, "_step_count": 11, "_last_lr": [1e-6]}, checkpoint / "scheduler.pt")
    for rank in range(4):
        torch.save({"python": random.Random(rank).getstate(),
                    "numpy": np.random.RandomState(rank).get_state(),
                    "cpu": torch.get_rng_state(),
                    "cuda": [torch.Generator().manual_seed(rank * 4 + gpu).get_state() for gpu in range(4)]},
                   checkpoint / f"rng_state_{rank}.pth")
    values["global_step"] = 10
    values["training_run_contract"] = post.training_run_contract_identity(contract_path, claim_scope="development")
    path = checkpoint / post.POST_STEP_RECEIPT_FILENAME
    path.unlink()
    write_json(path, reseal_post(values), readonly=True)
    second = tmp_path / "resumed"
    shutil.copytree(checkpoint.parent.parent, second)
    return checkpoint, second / "checkpoints/checkpoint-10"


def test_complete_actual_tensor_optimizer_scheduler_four_rank_rng_comparison(pair):
    result = compare.compare_checkpoints(*pair)
    assert result["status"] == "passed_exact_step10_continuation"
    assert result["differing_field_count"] == 0
    assert result["compared_leaf_count"] > 2500
    assert result["parameter_rng_values_exported"] is False
    # Either a run directory or the exact step-10 directory is accepted.
    other = compare.compare_checkpoints(pair[0].parent.parent, pair[1].parent.parent)
    assert other == result


@pytest.mark.parametrize("which", ["optimizer", "python", "numpy", "cpu", "cuda"])
def test_each_state_difference_is_detected_without_returning_values(pair, which):
    name = "optimizer.pt" if which == "optimizer" else "rng_state_2.pth"
    path = pair[1] / name
    state = torch.load(path, map_location="cpu", weights_only=False)
    if which == "optimizer":
        state["state"][0]["exp_avg"].fill_(123456.125)
    elif which == "python":
        state["python"] = random.Random(7654).getstate()
    elif which == "numpy":
        state["numpy"][1][0] ^= np.uint32(1)
    elif which == "cpu":
        state["cpu"][0] ^= 1
    else:
        state["cuda"][3][0] ^= 1
    torch.save(state, path)
    result = compare.compare_checkpoints(*pair)
    assert result["status"] == "failed_state_difference"
    assert result["differing_field_count"] > 0
    assert "123456.125" not in json.dumps(result)
    assert any(("optimizer" if which == "optimizer" else "rng_rank_2") in field
               for field in result["differing_fields"])


@pytest.mark.parametrize("which", ["optimizer_step", "scheduler_step", "missing_cuda"])
def test_equal_but_incomplete_or_wrong_step_states_never_pass(pair, which):
    for checkpoint in pair:
        name = {"optimizer_step": "optimizer.pt", "scheduler_step": "scheduler.pt",
                "missing_cuda": "rng_state_0.pth"}[which]
        path = checkpoint / name
        state = torch.load(path, map_location="cpu", weights_only=False)
        if which == "optimizer_step":
            state["state"][0]["step"] = torch.tensor(0.)
        elif which == "scheduler_step":
            state["last_epoch"] = 0
        else:
            del state["cuda"]
        torch.save(state, path)
    with pytest.raises(ValueError):
        compare.compare_checkpoints(*pair)


def test_both_checkpoints_admitted_before_any_pickle_is_loaded(pair, monkeypatch):
    (pair[1] / "rng_state_3.pth").unlink()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle read before admission"))
    with pytest.raises(FileNotFoundError):
        compare.compare_checkpoints(*pair)


def test_byte_exact_comparison_catches_signed_zero_and_preserves_nan_bits():
    result = compare.Comparison()
    result.compare(torch.tensor([0.]), torch.tensor([-0.]), "signed_zero")
    result.compare(np.array([0.]), np.array([-0.]), "numpy_zero")
    result.compare(0., -0., "float_zero")
    nan = torch.tensor([float("nan")])
    result.compare(nan, nan.clone(), "identical_nan")
    assert result.differences == ["signed_zero.tensor_bytes", "numpy_zero.array_bytes", "float_zero"]


def test_output_is_create_only_and_exception_values_are_suppressed(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise ValueError("PRIVATE_PARAMETER_VALUE_123456")

    monkeypatch.setattr(compare, "compare_checkpoints", fail)
    output = tmp_path / "result.json"
    argv = ["--uninterrupted", "/first", "--resumed", "/second", "--output", str(output)]
    assert compare.main(argv) == 1
    assert "PRIVATE" not in output.read_text() + capsys.readouterr().out
    assert json.loads(output.read_text())["exception_class"] == "ValueError"
    with pytest.raises(FileExistsError):
        compare.main(argv)
