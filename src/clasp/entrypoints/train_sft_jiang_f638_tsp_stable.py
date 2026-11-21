"""Stability-controlled TSP recovery for the Jiang-f638 SFT recipe.

The exact Jiang-f638 TSP run irreversibly changed loss regime at optimizer
step 3,145 while its learning rate was still approximately 1.8e-4.  This
entrypoint restarts from the frozen Qwen2.5-7B base and changes exactly one
optimization knob: the peak learning rate is reduced from 2e-4 to 1e-4.

Everything else is delegated to the audited Jiang-f638 trainer: dataset and
sampler seeds, BF16, AdamW-8bit, gradient clipping, batch shape, response-only
masking, LoRA closure, serialization audits, and the 31,250-step horizon.
The deviation is embedded in the immutable run hyperparameters.  A health
monitor stops the process on non-finite metrics or a sustained loss-regime
shift so a second collapsed policy cannot be mistaken for a valid checkpoint.

``smoke`` mode is an implementation-only 250-step boundary suitable for a
two-hour H100 development allocation.  It cannot validate the historical
step-3,145 failure window.  ``production`` mode retains the audited 5,000-step
chunk boundaries and must cross that window before the first checkpoint is
accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

from clasp.entrypoints import train_sft_jiang_f638 as parent

STABILITY_PROFILE = "tsp_lr1e4_failfast_v1"
PROTOCOL = "clasp-jiang-llmcosolver-f638-sft-tsp-stable-v1"
PARENT_RECIPE = str(parent.MATCHED_SFT_STAGE_HPARAMETERS["recipe"])
PARENT_LEARNING_RATE = 2e-4
STABLE_LEARNING_RATE = 1e-4
SMOKE_STOP_STEP = 250
PRODUCTION_CHECKPOINT_STEPS = tuple(parent.EXPECTED_TRAINER_CHECKPOINT_STEPS)
PRODUCTION_SAVE_STEPS = int(parent.SAVE_STEPS)
PRODUCTION_SAVE_TOTAL_LIMIT = int(parent.SAVE_TOTAL_LIMIT)

LOSS_WINDOW = 100
LOSS_GATE_START_STEP = 3_000
ROLLING_MEDIAN_LOSS_LIMIT = 0.25
CONSECUTIVE_HIGH_LOSS_START_STEP = 1_000
CONSECUTIVE_HIGH_LOSS_LIMIT = 0.5
CONSECUTIVE_HIGH_LOSS_PATIENCE = 25
GRADIENT_CLIP_NORM = 1.0


class TSPStabilityRecoveryError(parent.JiangF638SFTError):
    """Raised when a recovery launch violates its declared profile."""


class TSPStabilityMonitor:
    """Track logged optimizer-step health without changing optimization."""

    def __init__(self) -> None:
        self.losses: deque[float] = deque(maxlen=LOSS_WINDOW)
        self.grad_norms: deque[float] = deque(maxlen=LOSS_WINDOW)
        self.consecutive_high_loss = 0
        self.failure_reason: str | None = None

    def observe(self, *, step: int, logs: Mapping[str, Any]) -> dict[str, Any]:
        loss = _finite_metric(logs.get("loss"), name="loss", step=step)
        grad_norm = _finite_metric(logs.get("grad_norm"), name="grad_norm", step=step)
        learning_rate = _finite_metric(
            logs.get("learning_rate"), name="learning_rate", step=step
        )
        if loss is not None:
            self.losses.append(loss)
            if step >= CONSECUTIVE_HIGH_LOSS_START_STEP and loss > CONSECUTIVE_HIGH_LOSS_LIMIT:
                self.consecutive_high_loss += 1
            else:
                self.consecutive_high_loss = 0
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

        rolling_loss = (
            statistics.median(self.losses) if len(self.losses) == LOSS_WINDOW else None
        )
        rolling_grad = (
            statistics.median(self.grad_norms)
            if len(self.grad_norms) == LOSS_WINDOW
            else None
        )
        clipping_fraction = (
            sum(value > GRADIENT_CLIP_NORM for value in self.grad_norms)
            / len(self.grad_norms)
            if self.grad_norms
            else None
        )

        if (
            self.failure_reason is None
            and self.consecutive_high_loss >= CONSECUTIVE_HIGH_LOSS_PATIENCE
        ):
            self.failure_reason = (
                f"loss exceeded {CONSECUTIVE_HIGH_LOSS_LIMIT} for "
                f"{self.consecutive_high_loss} consecutive logged steps at step {step}"
            )
        if (
            self.failure_reason is None
            and step >= LOSS_GATE_START_STEP
            and rolling_loss is not None
            and rolling_loss > ROLLING_MEDIAN_LOSS_LIMIT
        ):
            self.failure_reason = (
                f"{LOSS_WINDOW}-step rolling median loss {rolling_loss:.6f} exceeded "
                f"{ROLLING_MEDIAN_LOSS_LIMIT} at step {step}"
            )

        return {
            "step": int(step),
            "loss": loss,
            "grad_norm_pre_clip": grad_norm,
            "grad_norm_post_clip_upper_bound": (
                min(grad_norm, GRADIENT_CLIP_NORM) if grad_norm is not None else None
            ),
            "learning_rate": learning_rate,
            "rolling_median_loss_100": rolling_loss,
            "rolling_median_grad_norm_100": rolling_grad,
            "rolling_pre_clip_fraction_gt_1_100": clipping_fraction,
            "consecutive_loss_gt_0_5": self.consecutive_high_loss,
            "failure_reason": self.failure_reason,
        }


def _finite_metric(value: Any, *, name: str, step: int) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TSPStabilityRecoveryError(
            f"non-numeric logged {name} at step {step}: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise TSPStabilityRecoveryError(
            f"non-finite logged {name} at step {step}: {result!r}"
        )
    return result


def _stable_stage_hyperparameters(source: Mapping[str, Any]) -> dict[str, Any]:
    stage = dict(source)
    observed_learning_rate = float(stage.get("learning_rate", float("nan")))
    if observed_learning_rate != PARENT_LEARNING_RATE:
        raise TSPStabilityRecoveryError(
            "parent Jiang-f638 learning rate drifted before recovery configuration: "
            f"{observed_learning_rate!r}"
        )
    stage.update(
        {
            "recipe": "jiang-llmcosolver-f638-corrected-tsp-stable-v1",
            "learning_rate": STABLE_LEARNING_RATE,
            "parent_recipe": PARENT_RECIPE,
            "tsp_stability_profile": STABILITY_PROFILE,
            "tsp_stability_single_changed_optimization_knob": "learning_rate",
            "tsp_stability_parent_learning_rate": PARENT_LEARNING_RATE,
            "tsp_stability_reason": (
                "fresh exact run sustained an irreversible loss/gradient regime change "
                "beginning at optimizer step 3145"
            ),
            "tsp_stability_fail_fast": {
                "loss_window": LOSS_WINDOW,
                "gate_start_step": LOSS_GATE_START_STEP,
                "rolling_median_loss_limit": ROLLING_MEDIAN_LOSS_LIMIT,
                "consecutive_high_loss_start_step": CONSECUTIVE_HIGH_LOSS_START_STEP,
                "consecutive_high_loss_limit": CONSECUTIVE_HIGH_LOSS_LIMIT,
                "consecutive_high_loss_patience": CONSECUTIVE_HIGH_LOSS_PATIENCE,
            },
        }
    )
    return stage


def _validate_mode(args: argparse.Namespace) -> None:
    if args.task != "tsp":
        raise TSPStabilityRecoveryError("the stability recovery entrypoint is TSP-only")
    stop_step = int(args.stop_after_step)
    if args.stability_mode == "smoke":
        if stop_step != SMOKE_STOP_STEP:
            raise TSPStabilityRecoveryError(
                f"smoke mode requires --stop-after-step {SMOKE_STOP_STEP}"
            )
        if args.resume_from_checkpoint is not None:
            raise TSPStabilityRecoveryError("smoke mode must start fresh")
        return
    if args.stability_mode != "production":
        raise TSPStabilityRecoveryError(
            f"unsupported stability mode: {args.stability_mode!r}"
        )
    if stop_step not in PRODUCTION_CHECKPOINT_STEPS:
        raise TSPStabilityRecoveryError(
            "production mode requires an audited 5,000-step chunk boundary"
        )


def _configure_parent(args: argparse.Namespace) -> None:
    """Install the declared process-local profile before parent.run()."""

    _validate_mode(args)
    parent.PROTOCOL = PROTOCOL
    parent.LEARNING_RATE = STABLE_LEARNING_RATE
    parent.MATCHED_SFT_STAGE_HPARAMETERS = _stable_stage_hyperparameters(
        parent.MATCHED_SFT_STAGE_HPARAMETERS
    )

    if args.stability_mode == "smoke":
        parent.SAVE_STEPS = SMOKE_STOP_STEP
        parent.SAVE_TOTAL_LIMIT = 2
        parent.EXPECTED_TRAINER_CHECKPOINT_STEPS = (SMOKE_STOP_STEP,)
        parent.ALLOWED_CHUNK_STOP_STEPS = (SMOKE_STOP_STEP,)
    else:
        parent.SAVE_STEPS = PRODUCTION_SAVE_STEPS
        parent.SAVE_TOTAL_LIMIT = PRODUCTION_SAVE_TOTAL_LIMIT
        parent.EXPECTED_TRAINER_CHECKPOINT_STEPS = PRODUCTION_CHECKPOINT_STEPS
        parent.ALLOWED_CHUNK_STOP_STEPS = PRODUCTION_CHECKPOINT_STEPS
    parent.FIXED_RECORDED_CLI_ARGS = {
        **parent.FIXED_RECORDED_CLI_ARGS,
        "save_steps": parent.SAVE_STEPS,
        "save_total_limit": parent.SAVE_TOTAL_LIMIT,
    }

    original_payload_builder = parent._build_hyperparams_payload

    def recovery_payload_builder(**kwargs: Any) -> dict[str, Any]:
        payload = original_payload_builder(**kwargs)
        payload["protocol"] = PROTOCOL
        payload["intentional_corrections"]["tsp_stability_learning_rate"] = {
            "parent": PARENT_LEARNING_RATE,
            "recovery": STABLE_LEARNING_RATE,
            "only_changed_optimization_knob": True,
        }
        payload["tsp_stability_recovery"] = {
            "profile": STABILITY_PROFILE,
            "mode": args.stability_mode,
            "clean_restart_required": True,
            "forbidden_resume_steps_from_failed_run": [5_000, 10_000, 15_000, 20_000],
            "historical_precursor_step": 3_103,
            "historical_durable_collapse_step": 3_145,
            "historical_max_loss": 45.5315,
            "historical_max_grad_norm": 921.6256,
            "two_hour_smoke_does_not_validate_collapse_window": True,
        }
        payload["args"]["tsp_stability_profile"] = STABILITY_PROFILE
        payload["args"]["tsp_stability_mode"] = args.stability_mode
        return payload

    parent._build_hyperparams_payload = recovery_payload_builder

    original_chunk_builder = parent.build_chunk_stop_callback

    def monitored_chunk_builder(
        stop_after_step: int,
        *,
        trainer_callback_class: type[Any] | None = None,
    ) -> Any:
        callback = original_chunk_builder(
            stop_after_step,
            trainer_callback_class=trainer_callback_class,
        )
        return _attach_stability_monitor(callback)

    parent.build_chunk_stop_callback = monitored_chunk_builder


def _attach_stability_monitor(callback: Any) -> Any:
    """Add monitoring while preserving the parent's exact-boundary callback."""

    monitor = TSPStabilityMonitor()
    original_on_save = callback.on_save

    def on_log(
        self: Any,
        training_args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not logs or "loss" not in logs:
            return control
        step = int(getattr(state, "global_step", -1))
        try:
            record = monitor.observe(step=step, logs=logs)
        except TSPStabilityRecoveryError as exc:
            monitor.failure_reason = str(exc)
            record = {
                "step": step,
                "failure_reason": monitor.failure_reason,
            }
        if step % 100 == 0 or record["failure_reason"] is not None:
            print(
                "[TSPStability] " + json.dumps(record, sort_keys=True),
                flush=True,
            )
        if record["failure_reason"] is not None:
            self.stability_failure = str(record["failure_reason"])
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(
        self: Any,
        training_args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if self.stability_failure is not None:
            raise TSPStabilityRecoveryError(
                "TSP stability monitor stopped a collapsed/non-finite run: "
                f"{self.stability_failure}"
            )
        return original_on_save(training_args, state, control, **kwargs)

    callback.stability_failure = None
    callback.on_log = MethodType(on_log, callback)
    callback.on_save = MethodType(on_save, callback)
    return callback


def build_parser() -> argparse.ArgumentParser:
    parser = parent.build_parser()
    parser.description = __doc__
    task_action = next(action for action in parser._actions if action.dest == "task")
    task_action.choices = ("tsp",)
    stop_action = next(
        action for action in parser._actions if action.dest == "stop_after_step"
    )
    stop_action.choices = (SMOKE_STOP_STEP, *PRODUCTION_CHECKPOINT_STEPS)
    stop_action.default = PRODUCTION_CHECKPOINT_STEPS[0]
    parser.add_argument(
        "--stability-mode",
        choices=("smoke", "production"),
        default="production",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    _configure_parent(args)
    return parent.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (parent.JiangF638SFTError, OSError, ValueError) as exc:
        print(f"[TSPStabilityRecoveryError] {exc}", file=sys.stderr)
        return 2
    print(
        f"[TSPStabilityRecovery] {args.stability_mode} complete at step "
        f"{args.stop_after_step}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
