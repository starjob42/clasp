from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pytest

from clasp.entrypoints import train_sft_jiang_f638 as parent
from clasp.entrypoints import train_sft_jiang_f638_tsp_stable as stable


def _args(*, mode: str, stop: int, task: str = "tsp", resume=None) -> argparse.Namespace:
    return argparse.Namespace(
        task=task,
        stability_mode=mode,
        stop_after_step=stop,
        resume_from_checkpoint=resume,
    )


def test_profile_changes_only_the_declared_optimizer_knob() -> None:
    original = dict(parent.MATCHED_SFT_STAGE_HPARAMETERS)
    projected = stable._stable_stage_hyperparameters(original)

    assert original["learning_rate"] == 2e-4
    assert projected["learning_rate"] == 1e-4
    assert projected["tsp_stability_single_changed_optimization_knob"] == "learning_rate"
    assert projected["parent_recipe"] == original["recipe"]
    ignored = {
        "recipe",
        "learning_rate",
        "parent_recipe",
        "tsp_stability_profile",
        "tsp_stability_single_changed_optimization_knob",
        "tsp_stability_parent_learning_rate",
        "tsp_stability_reason",
        "tsp_stability_fail_fast",
    }
    for key, value in original.items():
        if key not in ignored:
            assert projected[key] == value


def test_modes_fail_closed_on_task_boundary_and_resume() -> None:
    stable._validate_mode(_args(mode="smoke", stop=250))
    stable._validate_mode(_args(mode="production", stop=5_000))

    with pytest.raises(stable.TSPStabilityRecoveryError, match="TSP-only"):
        stable._validate_mode(_args(mode="production", stop=5_000, task="cvrp"))
    with pytest.raises(stable.TSPStabilityRecoveryError, match="requires --stop-after-step"):
        stable._validate_mode(_args(mode="smoke", stop=5_000))
    with pytest.raises(stable.TSPStabilityRecoveryError, match="must start fresh"):
        stable._validate_mode(
            _args(mode="smoke", stop=250, resume="checkpoint-250")
        )
    with pytest.raises(stable.TSPStabilityRecoveryError, match="5,000-step"):
        stable._validate_mode(_args(mode="production", stop=500))


def test_parser_defaults_to_first_production_boundary() -> None:
    parser = stable.build_parser()
    task_action = next(action for action in parser._actions if action.dest == "task")
    stop_action = next(
        action for action in parser._actions if action.dest == "stop_after_step"
    )
    assert tuple(task_action.choices) == ("tsp",)
    assert tuple(stop_action.choices) == (250, *stable.PRODUCTION_CHECKPOINT_STEPS)
    assert stop_action.default == 5_000


def test_monitor_accepts_healthy_window_and_reports_clipping() -> None:
    monitor = stable.TSPStabilityMonitor()
    record = None
    for step in range(2_901, 3_101):
        record = monitor.observe(
            step=step,
            logs={"loss": 0.18, "grad_norm": 0.2, "learning_rate": 8e-5},
        )
    assert record is not None
    assert record["rolling_median_loss_100"] == pytest.approx(0.18)
    assert record["rolling_pre_clip_fraction_gt_1_100"] == 0.0
    assert record["failure_reason"] is None


def test_monitor_stops_on_sustained_loss_regime_shift() -> None:
    monitor = stable.TSPStabilityMonitor()
    record = None
    for step in range(3_000, 3_100):
        record = monitor.observe(
            step=step,
            logs={"loss": 0.4, "grad_norm": 2.0, "learning_rate": 9e-5},
        )
    assert record is not None
    assert record["rolling_median_loss_100"] == pytest.approx(0.4)
    assert record["rolling_pre_clip_fraction_gt_1_100"] == 1.0
    assert record["failure_reason"] is not None


def test_callback_propagates_monitor_failure_as_a_hard_error() -> None:
    class Callback:
        def on_save(self, args, state, control, **kwargs):
            control.parent_save_observed = True
            return control

    callback = stable._attach_stability_monitor(Callback())
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    for step in range(3_000, 3_025):
        callback.on_log(
            SimpleNamespace(),
            SimpleNamespace(global_step=step),
            control,
            logs={"loss": 0.6, "grad_norm": 2.0, "learning_rate": 9e-5},
        )

    assert control.should_save is True
    assert control.should_training_stop is True
    assert callback.stability_failure is not None
    with pytest.raises(stable.TSPStabilityRecoveryError, match="stability monitor"):
        callback.on_save(
            SimpleNamespace(),
            SimpleNamespace(global_step=3_024),
            control,
        )
    assert not hasattr(control, "parent_save_observed")


@pytest.mark.parametrize("field", ["loss", "grad_norm", "learning_rate"])
def test_monitor_rejects_nonfinite_metrics(field: str) -> None:
    monitor = stable.TSPStabilityMonitor()
    logs = {"loss": 0.1, "grad_norm": 0.2, "learning_rate": 1e-4}
    logs[field] = math.nan
    with pytest.raises(stable.TSPStabilityRecoveryError, match="non-finite"):
        monitor.observe(step=10, logs=logs)
