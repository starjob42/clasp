"""Real process-group coverage for the online teacher loss and accumulation."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


def test_tensor_comparison_detects_signed_zero_that_torch_equal_accepts():
    import torch

    path = Path(__file__).resolve().parents[1] / "scripts/probe_verified_grpo_distributed.py"
    spec = importlib.util.spec_from_file_location("verified_ddp_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    for dtype in (torch.float32, torch.bfloat16):
        positive = torch.tensor([0.0, 1.0], dtype=dtype)
        negative = torch.tensor([-0.0, 1.0], dtype=dtype)
        assert torch.equal(positive, negative)
        difference = probe._tensor_difference(positive, negative)
        assert difference["numerically_equal"]
        assert difference["max_abs_difference"] == 0.0
        assert difference["different_elements"] == 0
        assert difference["different_bytes"] == 1
        assert probe._tensor_difference(positive, positive) is None


@pytest.mark.parametrize("inspect_every_step", [0, 1])
def test_real_ddp_teacher_loss_keeps_gradients_parameters_and_optimizer_equal(
    tmp_path, inspect_every_step
):
    assert dist.is_available() and dist.is_gloo_available()
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    environment = os.environ.copy()
    environment["GLOO_SOCKET_IFNAME"] = "lo"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/probe_verified_grpo_distributed.py"),
         "--device", "cpu", "--optimizer", "adamw", "--steps", "3",
         "--inspect-every-step", str(inspect_every_step),
         "--output", str(output)],
        cwd=root, env=environment, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["exact_rank_consensus"]
    assert len(report["ranks"]) == 2
    for rank in report["ranks"]:
        assert set(rank["modes"]) == {
            "grpo", "existing_double_forward", "fused_single_forward"}
        for observations in rank["modes"].values():
            assert len(observations) == 3
            counts = [count for step in observations for count in step["teacher_counts"]]
            assert any(count["global"] == 0 for count in counts)
            assert any(count["local"] == 0 and count["global"] > 0 for count in counts)
            assert any(0 < count["local"] < count["global"] for count in counts)
            for step in observations:
                for field in ("gradients", "parameters", "optimizer_states"):
                    if inspect_every_step or step["step"] == 3:
                        assert step[field]["different_tensor_count"] == 0
                    else:
                        assert step[field] is None
                if inspect_every_step:
                    assert step["gradient_inspection_phase"] == "before_clipping"
                elif step["step"] == 3:
                    assert step["gradient_inspection_phase"] == "after_final_optimizer_clipped"
                else:
                    assert step["gradient_inspection_phase"] == "not_inspected"


def test_real_trl_generation_checkpointing_regression_and_shared_fix(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["GLOO_SOCKET_IFNAME"] = "lo"
    reports = {}
    for lifecycle in ("trl", "restored"):
        output = tmp_path / f"{lifecycle}.json"
        result = subprocess.run(
            [sys.executable, str(root / "scripts/probe_verified_grpo_distributed.py"),
             "--device", "cpu", "--optimizer", "adamw", "--steps", "2",
             "--generation-lifecycle", lifecycle, "--bucket-cap-mb", "0.01",
             "--output", str(output)],
            cwd=root, env=environment, text=True, capture_output=True, timeout=180,
        )
        assert result.returncode == (1 if lifecycle == "trl" else 0), result.stdout + result.stderr
        reports[lifecycle] = json.loads(output.read_text())
    assert not reports["trl"]["exact_rank_consensus"]
    assert reports["restored"]["exact_rank_consensus"]
    bad_rank = reports["trl"]["ranks"][1]
    assert bad_rank["modes"]["existing_double_forward"][0]["gradients"]["different_tensor_count"] > 0
    for observations in reports["restored"]["ranks"][1]["modes"].values():
        for step in observations:
            assert step["checkpointing"] == {
                "before": {"use_reentrant": False}, "after_trl": {"use_reentrant": True},
                "before_loss": {"use_reentrant": False}}
            for field in ("gradients", "parameters", "optimizer_states"):
                assert step[field]["different_tensor_count"] == 0
