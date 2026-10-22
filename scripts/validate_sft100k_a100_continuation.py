#!/usr/bin/env python3
"""Validate the fixed unified-100k SFT continuation contract.

The validator is intentionally dependency-free so it can run on a Jean Zay
login or compute node before the Conda environment is activated.  It prints
one machine-readable line:

* ``RESUME:<checkpoint>`` for the newest complete Trainer checkpoint; or
* ``COMPLETE:<checkpoint>`` once the five-epoch model was finalized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
ROWS_PER_TASK = 100_000
WORLD_SIZE = 4
INITIAL_STEP = 31_250
TARGET_STEP = 156_250
CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")
MODEL_ARTIFACTS = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def _validate_run_contract(run_dir: Path, model_path: Path, data_root: Path) -> None:
    if not model_path.is_dir():
        raise ValueError(f"base model directory does not exist: {model_path}")
    if not data_root.is_dir():
        raise ValueError(f"SFT data directory does not exist: {data_root}")

    payload = _load_object(run_dir / "hyperparams.json")
    _require_equal(payload.get("stage"), "sft", "stage")
    _require_equal(payload.get("tasks"), list(TASKS), "tasks")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("hyperparams.json has no dataset object")
    _require_equal(dataset.get("train_size"), ROWS_PER_TASK * len(TASKS), "dataset.train_size")
    _require_equal(dataset.get("eval_size"), 0, "dataset.eval_size")
    _require_equal(
        dataset.get("train_task_counts"),
        {task: ROWS_PER_TASK for task in TASKS},
        "dataset.train_task_counts",
    )

    stage = payload.get("stage_hyperparameters")
    if not isinstance(stage, dict):
        raise ValueError("hyperparams.json has no stage_hyperparameters object")
    expected = {
        "full_finetune": False,
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_train_batch_size": 16,
        "learning_rate": 0.0002,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_steps": 20,
        "lr_scheduler_type": "linear",
        "max_steps": -1,
        "max_seq_length": 20_000,
        "deduplicate_train_rows": False,
        "no_reason": True,
        "prompt_style": "alpaca",
        "completion_only_loss": True,
        "memory_safe_loss": True,
        "supervised_logits_only": True,
    }
    for key, value in expected.items():
        _require_equal(stage.get(key), value, f"stage_hyperparameters.{key}")
    if float(stage.get("num_train_epochs", -1)) not in {1.0, 5.0}:
        raise ValueError(
            "stage_hyperparameters.num_train_epochs must describe the original "
            "one-epoch run or the five-epoch continuation"
        )

    distributed = payload.get("distributed")
    if not isinstance(distributed, dict):
        raise ValueError("hyperparams.json has no distributed object")
    _require_equal(distributed.get("world_size"), WORLD_SIZE, "distributed.world_size")


def _validate_adapter(checkpoint: Path, model_path: Path) -> None:
    config = _load_object(checkpoint / "adapter_config.json")
    _require_equal(config.get("peft_type"), "LORA", "adapter.peft_type")
    _require_equal(config.get("task_type"), "CAUSAL_LM", "adapter.task_type")
    _require_equal(config.get("r"), 64, "adapter.r")
    _require_equal(config.get("lora_alpha"), 64, "adapter.lora_alpha")
    _require_equal(float(config.get("lora_dropout", -1)), 0.0, "adapter.lora_dropout")
    _require_equal(config.get("bias"), "lora_only", "adapter.bias")
    _require_equal(
        set(config.get("target_modules") or []),
        {"q_proj", "k_proj", "v_proj", "o_proj"},
        "adapter.target_modules",
    )
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    _require_equal(configured_base, model_path.resolve(), "adapter.base_model_name_or_path")


def _checkpoint_step_if_strict(checkpoint: Path, model_path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        return None
    step = int(match.group(1))
    if step < INITIAL_STEP or step > TARGET_STEP:
        return None
    if any(not (checkpoint / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
        return None
    if not any((checkpoint / name).is_file() for name in MODEL_ARTIFACTS):
        return None
    if any(not (checkpoint / f"rng_state_{rank}.pth").is_file() for rank in range(WORLD_SIZE)):
        return None
    try:
        state = _load_object(checkpoint / "trainer_state.json")
        if int(state.get("global_step", -1)) != step:
            return None
        # The source checkpoint was completed under a one-epoch horizon.  Every
        # checkpoint produced after resumption uses the extended five-epoch horizon.
        if int(state.get("max_steps", -1)) not in {INITIAL_STEP, TARGET_STEP}:
            return None
        _validate_adapter(checkpoint, model_path)
    except (TypeError, ValueError):
        return None
    return step


def _is_finalized(run_dir: Path, target_checkpoint: Path) -> bool:
    if not any((run_dir / name).is_file() for name in MODEL_ARTIFACTS):
        return False
    try:
        status = _load_object(run_dir / "stage_status.json")
    except ValueError:
        return False
    return (
        status.get("completed") is True
        and int(status.get("global_step", -1)) == TARGET_STEP
        and int(status.get("max_steps", -1)) == TARGET_STEP
        and Path(str(status.get("latest_checkpoint_dir", ""))).resolve()
        == target_checkpoint.resolve()
    )


def resolve_state(run_dir: Path, model_path: Path, data_root: Path) -> str:
    run_dir = run_dir.resolve()
    model_path = model_path.resolve()
    data_root = data_root.resolve()
    _validate_run_contract(run_dir, model_path, data_root)

    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_root}")
    candidates: list[tuple[int, Path]] = []
    for checkpoint in checkpoint_root.iterdir():
        step = _checkpoint_step_if_strict(checkpoint, model_path)
        if step is not None:
            candidates.append((step, checkpoint.resolve()))
    if not candidates:
        raise ValueError("no complete optimizer-resumable checkpoint satisfies the run contract")

    step, checkpoint = max(candidates)
    if step == TARGET_STEP and _is_finalized(run_dir, checkpoint):
        return f"COMPLETE:{checkpoint}"
    return f"RESUME:{checkpoint}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(resolve_state(args.run_dir, args.model_path, args.data_root))
    except ValueError as exc:
        print(f"[SFT100kContinuationError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
