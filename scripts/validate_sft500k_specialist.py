#!/usr/bin/env python3
"""Validate or resolve a Jiang-matched 500k specialist SFT run.

The script is dependency-free so it can run on Jean Zay login and compute
nodes.  It prints exactly one state line on success:

* ``FRESH`` when no run exists yet;
* ``RESUME:<checkpoint>`` for the newest complete optimizer checkpoint; or
* ``COMPLETE:<checkpoint>`` after the one-epoch target is finalized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROWS = 500_000
WORLD_SIZE = 4
TARGET_STEP = 31_250
TASKS = {"op", "mis", "jssp"}
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")
MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_FILES = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}={actual!r}, expected {expected!r}")


def validate_adapter(checkpoint: Path, model_path: Path) -> None:
    config = load_object(checkpoint / "adapter_config.json")
    require_equal(config.get("peft_type"), "LORA", "adapter.peft_type")
    require_equal(config.get("task_type"), "CAUSAL_LM", "adapter.task_type")
    require_equal(config.get("r"), 64, "adapter.r")
    require_equal(config.get("lora_alpha"), 64, "adapter.lora_alpha")
    require_equal(float(config.get("lora_dropout", -1)), 0.0, "adapter.lora_dropout")
    require_equal(config.get("bias"), "lora_only", "adapter.bias")
    require_equal(set(config.get("target_modules") or []), TARGET_MODULES, "adapter.target_modules")
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).resolve()
    require_equal(configured_base, model_path.resolve(), "adapter.base_model_name_or_path")


def validate_base_model(model_path: Path) -> None:
    config = load_object(model_path / "config.json")
    require_equal(config.get("model_type"), "qwen2", "base_model.model_type")
    require_equal(config.get("num_hidden_layers"), 28, "base_model.num_hidden_layers")


def validate_run(run_dir: Path, task: str, model_path: Path, data_root: Path) -> None:
    if not model_path.is_dir():
        raise ValueError(f"base model directory does not exist: {model_path}")
    validate_base_model(model_path)
    if not data_root.is_dir():
        raise ValueError(f"SFT data root does not exist: {data_root}")
    payload = load_object(run_dir / "hyperparams.json")
    require_equal(payload.get("stage"), "sft", "stage")
    require_equal(payload.get("tasks"), [task], "tasks")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("hyperparams.json has no dataset object")
    require_equal(dataset.get("train_size"), ROWS, "dataset.train_size")
    require_equal(dataset.get("eval_size"), 0, "dataset.eval_size")
    require_equal(dataset.get("train_task_counts"), {task: ROWS}, "dataset.train_task_counts")

    stage = payload.get("stage_hyperparameters")
    if not isinstance(stage, dict):
        raise ValueError("hyperparams.json has no stage_hyperparameters object")
    expected = {
        "full_finetune": False,
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_train_batch_size": 16,
        "learning_rate": 0.0002,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_steps": 20,
        "lr_scheduler_type": "linear",
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "max_seq_length": 20_000,
        "min_train_samples_per_task": ROWS,
        "deduplicate_train_rows": False,
        "no_reason": True,
        "prompt_style": "alpaca",
        "completion_only_loss": True,
        "memory_safe_loss": True,
        "supervised_logits_only": True,
    }
    for key, value in expected.items():
        require_equal(stage.get(key), value, f"stage_hyperparameters.{key}")

    distributed = payload.get("distributed")
    if not isinstance(distributed, dict):
        raise ValueError("hyperparams.json has no distributed object")
    require_equal(distributed.get("world_size"), WORLD_SIZE, "distributed.world_size")


def strict_checkpoint_step(checkpoint: Path, model_path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        return None
    step = int(match.group(1))
    if step > TARGET_STEP:
        return None
    if any(not (checkpoint / name).is_file() for name in REQUIRED_FILES):
        return None
    if not any((checkpoint / name).is_file() for name in MODEL_FILES):
        return None
    if any(not (checkpoint / f"rng_state_{rank}.pth").is_file() for rank in range(WORLD_SIZE)):
        return None
    try:
        state = load_object(checkpoint / "trainer_state.json")
        require_equal(int(state.get("global_step", -1)), step, "trainer_state.global_step")
        require_equal(int(state.get("max_steps", -1)), TARGET_STEP, "trainer_state.max_steps")
        validate_adapter(checkpoint, model_path)
    except (TypeError, ValueError):
        return None
    return step


def is_finalized(run_dir: Path, checkpoint: Path) -> bool:
    if not any((run_dir / name).is_file() for name in MODEL_FILES):
        return False
    try:
        status = load_object(run_dir / "stage_status.json")
    except ValueError:
        return False
    return (
        status.get("completed") is True
        and int(status.get("global_step", -1)) == TARGET_STEP
        and int(status.get("max_steps", -1)) == TARGET_STEP
        and Path(str(status.get("latest_checkpoint_dir", ""))).resolve() == checkpoint.resolve()
    )


def resolve(run_dir: Path, task: str, model_path: Path, data_root: Path) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}, got {task!r}")
    model_path = model_path.resolve()
    data_root = data_root.resolve()
    if not model_path.is_dir():
        raise ValueError(f"base model directory does not exist: {model_path}")
    validate_base_model(model_path)
    if not data_root.is_dir():
        raise ValueError(f"SFT data root does not exist: {data_root}")
    if not (data_root / task / "train").is_dir():
        raise ValueError(f"training split directory does not exist for {task}")
    if not run_dir.exists():
        return "FRESH"
    if not (run_dir / "hyperparams.json").is_file():
        raise ValueError(f"run directory exists without hyperparams.json: {run_dir}")

    validate_run(run_dir, task, model_path, data_root)
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_root}")
    candidates: list[tuple[int, Path]] = []
    for checkpoint in checkpoint_root.iterdir():
        step = strict_checkpoint_step(checkpoint, model_path)
        if step is not None:
            candidates.append((step, checkpoint.resolve()))
    if not candidates:
        raise ValueError("no complete optimizer-resumable checkpoint satisfies the run contract")
    step, checkpoint = max(candidates)
    if step == TARGET_STEP and is_finalized(run_dir, checkpoint):
        return f"COMPLETE:{checkpoint}"
    if step >= TARGET_STEP:
        raise ValueError("target checkpoint exists but the final model/status receipt is incomplete")
    return f"RESUME:{checkpoint}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(resolve(args.run_dir, args.task, args.model_path, args.data_root))
    except ValueError as exc:
        print(f"[SFT500kSpecialistError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
