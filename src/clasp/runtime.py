"""Cross-stage runtime infrastructure.

Run-name construction, Weights & Biases environment setup (with writable-directory
fallbacks), context-window clamping, batch and generation-schedule arithmetic
(:func:`effective_optimization_batch_size`, :func:`resolve_generation_schedule`),
checkpoint discovery and resume (:func:`find_latest_resumable_checkpoint_dir`), and
stage-status persistence. Imported by every entry point.
"""
from __future__ import annotations

import errno
import gc
import json
import os
import tempfile
import time
from collections import Counter
from typing import Any

SUPPORTED_TASKS = ("tsp", "cvrp", "op", "mis", "jssp")
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
DEFAULT_SFT_ROOT = os.path.join(_REPO_ROOT, "data", "data_sft")
DEFAULT_RL_ROOT = os.path.join(_REPO_ROOT, "data", "data_rl")
HYPERPARAMS_FILENAME = "hyperparams.json"
STAGE_STATUS_FILENAME = "stage_status.json"
CHECKPOINTS_DIRNAME = "checkpoints"


def parse_task_list(raw: str) -> list[str]:
    tasks = [task.strip().lower() for task in str(raw).split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks cannot be empty")
    invalid = [task for task in tasks if task not in SUPPORTED_TASKS]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}. Supported: {SUPPORTED_TASKS}")
    return tasks


def env_world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def env_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def is_primary_process() -> bool:
    return env_rank() == 0


def print_primary(message: str) -> None:
    if is_primary_process():
        print(message, flush=True)


def sanitize_run_name_component(value: Any) -> str:
    sanitized = "".join(
        ch if (ch.isalnum() or ch in {"-", "_", "."}) else "-"
        for ch in str(value).strip()
    )
    sanitized = sanitized.strip("-")
    return sanitized or "run"


def format_tag_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0.0:
            text = "0"
        elif abs(value) < 1e-3 or abs(value) >= 1e3:
            text = f"{value:.0e}"
        else:
            text = f"{value:g}"
    else:
        text = str(value)
    return sanitize_run_name_component(text)


def format_training_budget(num_train_epochs: float, max_steps: int) -> str:
    if int(max_steps) > 0:
        return f"ms{int(max_steps)}"
    return f"ep{format_tag_value(num_train_epochs)}"


def resolve_base_model_name(model_path: str) -> str:
    candidate_paths = (
        ("adapter_config.json", "base_model_name_or_path"),
        ("config.json", "_name_or_path"),
    )
    for file_name, field_name in candidate_paths:
        file_path = os.path.join(model_path, file_name)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        candidate = payload.get(field_name)
        if candidate:
            return sanitize_run_name_component(os.path.basename(str(candidate).rstrip(os.sep)))

    fallback = os.path.basename(model_path.rstrip(os.sep)) or model_path
    return sanitize_run_name_component(fallback)


def resolve_model_artifact_name(model_path: str) -> str:
    """Return a run-stable tag for the concrete model artifact at `model_path`.

    Unlike `resolve_base_model_name`, this keeps stage provenance. In
    particular, merged SFT handoff directories and checkpoint directories
    receive distinct names so separate GRPO runs do not accidentally resume
    into the same output folder.
    """

    normalized_path = os.path.normpath(str(model_path))
    base_name = os.path.basename(normalized_path)
    parent_name = os.path.basename(os.path.dirname(normalized_path))

    if base_name == "merged_for_grpo":
        if parent_name:
            return sanitize_run_name_component(f"{parent_name}-merged")
        return "merged-for-grpo"

    if base_name.startswith("checkpoint-"):
        if parent_name:
            return sanitize_run_name_component(f"{parent_name}-{base_name}")
        return sanitize_run_name_component(base_name)

    if base_name:
        return sanitize_run_name_component(base_name)

    return resolve_base_model_name(model_path)


def normalize_context_candidate(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0 or normalized >= 1_000_000_000:
        return None
    return normalized


def infer_model_context_window(model, tokenizer=None) -> int | None:
    candidates: list[int] = []
    model_config = getattr(model, "config", None)
    if model_config is not None:
        for attr_name in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
            "model_max_length",
        ):
            candidate = normalize_context_candidate(getattr(model_config, attr_name, None))
            if candidate is not None:
                candidates.append(candidate)

    if tokenizer is not None:
        tokenizer_limit = normalize_context_candidate(getattr(tokenizer, "model_max_length", None))
        if tokenizer_limit is not None:
            candidates.append(tokenizer_limit)

    return min(candidates) if candidates else None


def enforce_sft_context_limit(max_seq_length: int, model, tokenizer) -> tuple[int, dict]:
    context_window = infer_model_context_window(model, tokenizer)
    if context_window is None:
        return max_seq_length, {"stage": "sft", "model_context_window": None, "max_seq_length": max_seq_length}

    resolved_max_seq_length = min(int(max_seq_length), int(context_window))
    if resolved_max_seq_length != int(max_seq_length):
        print_primary(
            "[Config] Clamped SFT max_seq_length to fit the loaded model context window: "
            f"{max_seq_length} -> {resolved_max_seq_length} (context_window={context_window})"
        )

    return resolved_max_seq_length, {
        "stage": "sft",
        "model_context_window": context_window,
        "max_seq_length": resolved_max_seq_length,
    }


def enforce_grpo_context_limits(
    max_prompt_length: int,
    max_completion_length: int,
    model,
    tokenizer,
) -> tuple[int, int, dict]:
    context_window = infer_model_context_window(model, tokenizer)
    if context_window is None:
        return (
            max_prompt_length,
            max_completion_length,
            {
                "stage": "grpo",
                "model_context_window": None,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
                "total_generation_window": max_prompt_length + max_completion_length,
            },
        )

    resolved_completion = int(max_completion_length)
    if resolved_completion >= context_window:
        old_value = resolved_completion
        resolved_completion = max(256, min(resolved_completion, context_window // 2))
        print_primary(
            "[Config] Clamped GRPO max_completion_length to keep it strictly below the model "
            f"context window: {old_value} -> {resolved_completion} (context_window={context_window})"
        )

    max_prompt_budget = context_window - resolved_completion
    if max_prompt_budget <= 0:
        raise ValueError(
            "Resolved GRPO sequence lengths exceed the loaded model context window: "
            f"context_window={context_window}, "
            f"max_prompt_length={max_prompt_length}, "
            f"max_completion_length={max_completion_length}."
        )

    resolved_prompt = min(int(max_prompt_length), int(max_prompt_budget))
    if resolved_prompt != int(max_prompt_length):
        print_primary(
            "[Config] Clamped GRPO max_prompt_length to fit the loaded model context window: "
            f"{max_prompt_length} -> {resolved_prompt} "
            f"(context_window={context_window}, max_completion_length={resolved_completion})"
        )

    return resolved_prompt, resolved_completion, {
        "stage": "grpo",
        "model_context_window": context_window,
        "max_prompt_length": resolved_prompt,
        "max_completion_length": resolved_completion,
        "total_generation_window": resolved_prompt + resolved_completion,
    }


def build_sft_run_name(
    base_model_name: str,
    *,
    prompt_style: str = "alpaca",
    no_reason: bool,
    num_tasks: int,
    dtype: str,
    full_finetune: bool,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    max_steps: int,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
) -> str:
    normalized_prompt_style = str(prompt_style).strip().lower() or "alpaca"
    if normalized_prompt_style == "auto":
        normalized_prompt_style = "alpaca"
    parts = [
        f"ps{normalized_prompt_style}" if normalized_prompt_style != "alpaca" else None,
        "no_reason" if no_reason else None,
        base_model_name,
        f"{num_tasks}tasks",
        format_tag_value(dtype),
        "full" if full_finetune else f"lora{lora_r}a{lora_alpha}",
        f"bs{per_device_train_batch_size}x{gradient_accumulation_steps}",
        f"lr{format_tag_value(learning_rate)}",
        format_training_budget(num_train_epochs, max_steps),
        f"seq{max_seq_length}",
    ]
    return "_".join(sanitize_run_name_component(part) for part in parts if part)


def build_grpo_run_name(
    base_model_name: str,
    *,
    prompt_style: str = "alpaca",
    no_reason: bool,
    num_tasks: int,
    dtype: str,
    full_finetune: bool,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    generation_batch_size: int,
    learning_rate: float,
    num_train_epochs: float,
    max_steps: int,
    max_prompt_length: int,
    max_completion_length: int,
    lora_r: int,
    lora_alpha: int,
    use_primal_dual: bool,
    epsilon: float | None,
    pd_variant_tag: str | None = None,
) -> str:
    normalized_prompt_style = str(prompt_style).strip().lower() or "alpaca"
    if normalized_prompt_style == "auto":
        normalized_prompt_style = "alpaca"
    parts = [
        f"ps{normalized_prompt_style}" if normalized_prompt_style != "alpaca" else None,
        "no_reason" if no_reason else None,
        base_model_name,
        f"{num_tasks}tasks",
        format_tag_value(dtype),
        "full" if full_finetune else f"lora{lora_r}a{lora_alpha}",
        f"bs{per_device_train_batch_size}x{gradient_accumulation_steps}",
        f"g{num_generations}",
        f"gb{generation_batch_size}",
        f"lr{format_tag_value(learning_rate)}",
        format_training_budget(num_train_epochs, max_steps),
        f"p{max_prompt_length}",
        f"c{max_completion_length}",
    ]
    if use_primal_dual:
        parts.append(f"eps{format_tag_value(epsilon if epsilon is not None else 0.0)}")
        if pd_variant_tag:
            parts.append(pd_variant_tag)
    return "_".join(sanitize_run_name_component(part) for part in parts if part)


def global_batch_size(per_device_batch_size: int) -> int:
    return max(1, int(per_device_batch_size)) * env_world_size()


def effective_optimization_batch_size(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    return global_batch_size(per_device_train_batch_size) * max(1, int(gradient_accumulation_steps))


def resolve_generation_schedule(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    generation_batch_size: int | None,
    steps_per_generation: int | None,
) -> tuple[int, int | None]:
    if int(num_generations) < 1:
        raise ValueError(f"num_generations must be >= 1, got {num_generations}")
    if generation_batch_size is not None and steps_per_generation is not None:
        raise ValueError(
            "generation_batch_size and steps_per_generation are mutually exclusive in GRPO. "
            "Specify only one of them."
        )

    if steps_per_generation is not None and int(steps_per_generation) < 1:
        raise ValueError(f"steps_per_generation must be >= 1, got {steps_per_generation}")

    resolved_steps_per_generation = steps_per_generation
    if generation_batch_size is None:
        if resolved_steps_per_generation is None:
            resolved_steps_per_generation = max(1, int(gradient_accumulation_steps))
        resolved_generation_batch_size = (
            global_batch_size(per_device_train_batch_size) * int(resolved_steps_per_generation)
        )
    else:
        resolved_generation_batch_size = int(generation_batch_size)
        if resolved_generation_batch_size < 1:
            raise ValueError(
                f"generation_batch_size must be >= 1 when provided, got {generation_batch_size}"
            )

    if resolved_generation_batch_size % int(num_generations) != 0:
        raise ValueError(
            "The resolved GRPO generation batch size must be divisible by num_generations. "
            f"Resolved generation batch size={resolved_generation_batch_size}, "
            f"num_generations={num_generations}."
        )

    return resolved_generation_batch_size, resolved_steps_per_generation


def report_to_uses_wandb(report_to: str) -> bool:
    return "wandb" in str(report_to).lower()


def resolve_wandb_run_name(run_name: str) -> str:
    run_name_override = str(os.environ.get("WANDB_RUN_NAME_OVERRIDE", "")).strip()
    if run_name_override:
        return run_name_override
    run_name_prefix = str(os.environ.get("WANDB_NAME_PREFIX", "")).strip()
    if run_name_prefix:
        return f"{run_name_prefix}{run_name}"
    return str(run_name)


def _wandb_probe_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    probe_file = None
    try:
        fd, probe_file = tempfile.mkstemp(prefix=".wandb_probe_", dir=path)
        os.close(fd)
    finally:
        if probe_file and os.path.exists(probe_file):
            os.unlink(probe_file)


def _wandb_fallback_roots() -> list[str]:
    user_name = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("SLURM_JOB_ID") or "local"
    suffix = os.path.join("wandb", f"{user_name}-{job_id}")
    candidates = [
        os.environ.get("WANDB_FALLBACK_DIR"),
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("JOBSCRATCH"),
        os.environ.get("TMPDIR"),
        tempfile.gettempdir(),
    ]
    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.join(str(candidate), suffix))
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _select_wandb_dir(preferred_dir: str) -> str:
    candidates = [os.path.abspath(preferred_dir), *_wandb_fallback_roots()]
    last_error: OSError | None = None
    seen: set[str] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            _wandb_probe_dir(candidate)
            if candidate != os.path.abspath(preferred_dir):
                print_primary(
                    f"[W&B] Falling back to {candidate} because "
                    f"{os.path.abspath(preferred_dir)} is not writable for run logs."
                )
            return candidate
        except OSError as exc:
            last_error = exc
            if exc.errno not in {
                errno.EDQUOT,
                errno.ENOSPC,
                errno.EACCES,
                errno.EROFS,
                errno.EPERM,
            }:
                raise

    if last_error is not None:
        raise last_error
    raise OSError(f"Unable to prepare a W&B directory from candidates: {candidates}")


def _resolve_wandb_subdir(env_name: str, root_dir: str, leaf_name: str) -> str:
    explicit_value = os.environ.get(env_name)
    candidate = os.path.abspath(explicit_value) if explicit_value else os.path.join(root_dir, leaf_name)
    try:
        _wandb_probe_dir(candidate)
        return candidate
    except OSError as exc:
        if explicit_value and exc.errno in {
            errno.EDQUOT,
            errno.ENOSPC,
            errno.EACCES,
            errno.EROFS,
            errno.EPERM,
        }:
            fallback = os.path.join(root_dir, leaf_name)
            _wandb_probe_dir(fallback)
            print_primary(
                f"[W&B] Falling back to {fallback} for {env_name} because "
                f"{candidate} is not writable."
            )
            return fallback
        raise


def _resolve_wandb_dir(preferred_dir: str) -> str:
    strict_dir = str(os.environ.get("WANDB_STRICT_DIR", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict_dir:
        resolved = os.path.abspath(preferred_dir)
        _wandb_probe_dir(resolved)
        return resolved
    return _select_wandb_dir(preferred_dir)


def configure_wandb_env(
    *,
    report_to: str,
    project: str,
    group: str | None,
    stage: str,
    run_name: str,
) -> None:
    if not report_to_uses_wandb(report_to):
        return

    # In the unified SFT->GRPO pipeline, both stages run in the same Python process.
    # Explicitly closing any active run here prevents stage metrics from being merged.
    try:
        import wandb  # type: ignore[import-not-found]

        active_run = getattr(wandb, "run", None)
        if active_run is not None:
            wandb.finish(quiet=True)
    except Exception:
        pass

    # Keep offline/online run files in a deterministic location even when launched via schedulers.
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        project_root = os.path.abspath(project_root)
        if os.path.basename(project_root) == "clasp":
            default_wandb_dir = os.path.join(project_root, "wandb")
        else:
            default_wandb_dir = os.path.join(project_root, "clasp", "wandb")
    else:
        default_wandb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb")
    wandb_dir = _resolve_wandb_dir(os.environ.get("WANDB_DIR", default_wandb_dir))
    os.environ["WANDB_DIR"] = wandb_dir
    os.environ["WANDB_DATA_DIR"] = _resolve_wandb_subdir("WANDB_DATA_DIR", wandb_dir, "data")
    os.environ["WANDB_CACHE_DIR"] = _resolve_wandb_subdir("WANDB_CACHE_DIR", wandb_dir, "cache")
    os.environ["WANDB_ARTIFACT_DIR"] = _resolve_wandb_subdir("WANDB_ARTIFACT_DIR", wandb_dir, "artifacts")

    os.environ["WANDB_PROJECT"] = str(project or "grpo_primal_dual")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    if group:
        os.environ["WANDB_RUN_GROUP"] = str(group)
    else:
        os.environ.setdefault("WANDB_RUN_GROUP", f"{stage}-{os.environ.get('SLURM_JOB_ID', 'local')}")

    os.environ["WANDB_NAME"] = str(run_name)


def task_distribution(dataset) -> dict[str, int]:
    if dataset is None or "task_type" not in getattr(dataset, "column_names", []):
        return {}
    counts = Counter(str(task) for task in dataset["task_type"])
    return {task: int(count) for task, count in sorted(counts.items())}


def resolve_output_dir(
    *,
    output_root: str,
    stage: str,
    run_name: str,
    explicit_output_dir: str | None,
) -> str:
    if explicit_output_dir:
        normalized_explicit_output_dir = os.path.normpath(explicit_output_dir)
        if os.path.basename(normalized_explicit_output_dir) == stage:
            explicit_output_dir = os.path.join(explicit_output_dir, run_name)
        os.makedirs(explicit_output_dir, exist_ok=True)
        return explicit_output_dir
    output_dir = os.path.join(output_root, stage, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolve_checkpoint_output_dir(output_dir: str) -> str:
    checkpoint_output_dir = os.path.join(output_dir, CHECKPOINTS_DIRNAME)
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    return checkpoint_output_dir


def write_hyperparams(output_dir: str, payload: dict) -> None:
    if not is_primary_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, HYPERPARAMS_FILENAME)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def serialize_trainer_args(trainer_args) -> dict[str, Any] | None:
    if trainer_args is None:
        return None

    to_dict = getattr(trainer_args, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass

    try:
        return dict(vars(trainer_args))
    except Exception:
        return None


def build_run_payload(
    *,
    stage: str,
    run_name: str,
    base_model_name: str,
    model_path: str,
    output_dir: str,
    tasks: list[str],
    report_to: str,
    train_dataset,
    eval_dataset,
    stage_hyperparameters: dict,
    context_info: dict,
    args_namespace,
    trainer_args=None,
    checkpoint_output_dir: str | None = None,
) -> dict:
    payload = {
        "stage": stage,
        "run_name": run_name,
        "base_model_name": base_model_name,
        "model_path": model_path,
        "output_dir": output_dir,
        "tasks": list(tasks),
        "num_datasets": len(tasks),
        "dataset": {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "train_task_counts": task_distribution(train_dataset),
            "eval_task_counts": task_distribution(eval_dataset),
        },
        "distributed": {
            "world_size": env_world_size(),
            "rank": env_rank(),
        },
        "report_to": report_to,
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT") if report_to_uses_wandb(report_to) else None,
            "group": os.environ.get("WANDB_RUN_GROUP") if report_to_uses_wandb(report_to) else None,
            "mode": os.environ.get("WANDB_MODE"),
            "dir": os.environ.get("WANDB_DIR") if report_to_uses_wandb(report_to) else None,
        },
        "context": context_info,
        "stage_hyperparameters": stage_hyperparameters,
        "args": dict(vars(args_namespace)),
    }

    trainer_config = serialize_trainer_args(trainer_args)
    if trainer_config is not None:
        payload["trainer_config"] = trainer_config

    if checkpoint_output_dir is not None:
        payload["artifacts"] = {
            "model_dir": output_dir,
            "checkpoint_dir": checkpoint_output_dir,
        }

    return payload


def _checkpoint_search_roots(output_dir: str) -> list[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return []

    search_roots = []
    if os.path.basename(os.path.normpath(output_dir)) == CHECKPOINTS_DIRNAME:
        search_roots.append(output_dir)
    else:
        checkpoint_root = os.path.join(output_dir, CHECKPOINTS_DIRNAME)
        if os.path.isdir(checkpoint_root):
            search_roots.append(checkpoint_root)
        search_roots.append(output_dir)
    return search_roots


def _iter_checkpoint_dirs(output_dir: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    seen_paths: set[str] = set()

    for search_root in _checkpoint_search_roots(output_dir):
        try:
            entries = os.listdir(search_root)
        except OSError:
            continue

        for name in entries:
            path = os.path.join(search_root, name)
            if not (os.path.isdir(path) and name.startswith("checkpoint-")):
                continue
            try:
                step = int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue

            normalized_path = os.path.normpath(path)
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            candidates.append((step, path))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def find_latest_checkpoint_dir(output_dir: str) -> str | None:
    checkpoints = _iter_checkpoint_dirs(output_dir)
    if not checkpoints:
        return None

    return checkpoints[0][1]


def checkpoint_has_model_artifacts(checkpoint_dir: str) -> bool:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return False

    root_candidates = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    if any(os.path.isfile(os.path.join(checkpoint_dir, name)) for name in root_candidates):
        return True

    # Some PEFT saves place adapter files under adapter-name subdirectories.
    try:
        children = os.listdir(checkpoint_dir)
    except OSError:
        return False

    for child in children:
        child_path = os.path.join(checkpoint_dir, child)
        if not os.path.isdir(child_path):
            continue
        if (
            os.path.isfile(os.path.join(child_path, "adapter_model.bin"))
            or os.path.isfile(os.path.join(child_path, "adapter_model.safetensors"))
        ):
            return True

    return False


def checkpoint_has_valid_trainer_state(checkpoint_dir: str) -> bool:
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    payload = load_json_dict(trainer_state_path)
    return isinstance(payload, dict) and bool(payload)


def checkpoint_has_required_files(checkpoint_dir: str, required_files: tuple[str, ...] = ()) -> bool:
    for relative_path in required_files:
        if not os.path.exists(os.path.join(checkpoint_dir, relative_path)):
            return False
    return True


def list_resumable_checkpoint_dirs(
    output_dir: str,
    *,
    require_trainer_state: bool = True,
    required_files: tuple[str, ...] = (),
) -> list[str]:
    resumable: list[str] = []
    for _step, path in _iter_checkpoint_dirs(output_dir):
        if not checkpoint_has_model_artifacts(path):
            continue
        if require_trainer_state and not checkpoint_has_valid_trainer_state(path):
            continue
        if not checkpoint_has_required_files(path, required_files=required_files):
            continue
        resumable.append(path)
    return resumable


def find_latest_resumable_checkpoint_dir(output_dir: str) -> str | None:
    resumable = list_resumable_checkpoint_dirs(output_dir)
    if not resumable:
        return None
    return resumable[0]


def _iter_exception_chain(exc: BaseException):
    seen_ids: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen_ids:
        yield current
        seen_ids.add(id(current))
        current = current.__cause__ or current.__context__


def is_likely_resume_load_error(exc: BaseException) -> bool:
    message_markers = (
        "state_dict",
        "trainer_state",
        "optimizer",
        "scheduler",
        "rng_state",
        "safetensor",
        "pytorchstreamreader",
        "pickle data was truncated",
        "invalid load key",
        "unexpected eof",
        "file is too short",
        "failed finding central directory",
        "unable to load",
        "cannot load",
    )

    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, (OSError, EOFError, json.JSONDecodeError)):
            return True

        message = str(candidate).strip().lower()
        if not message:
            continue
        if any(marker in message for marker in message_markers):
            return True

    return False


def has_checkpoint_dir(output_dir: str) -> bool:
    return find_latest_checkpoint_dir(output_dir) is not None


def output_dir_has_saved_model(output_dir: str) -> bool:
    if not output_dir or not os.path.isdir(output_dir):
        return False
    expected_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any(os.path.exists(os.path.join(output_dir, name)) for name in expected_files)


def write_stage_status(output_dir: str, stage: str, trainer_state) -> None:
    if not is_primary_process():
        return
    payload = {
        "stage": stage,
        "completed": True,
        "global_step": safe_int(getattr(trainer_state, "global_step", None)),
        "max_steps": safe_int(getattr(trainer_state, "max_steps", None)),
        "epoch": safe_float(getattr(trainer_state, "epoch", None)),
        "has_saved_model": output_dir_has_saved_model(output_dir),
        "latest_checkpoint_dir": find_latest_checkpoint_dir(output_dir),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, STAGE_STATUS_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_prefixed_trainer_logs(output_dir: str, stage: str, trainer_state) -> None:
    """Write stage-prefixed trainer state/log artifacts for easier SFT/GRPO separation."""
    if not is_primary_process():
        return

    os.makedirs(output_dir, exist_ok=True)
    stage_prefix = sanitize_run_name_component(stage)
    state_payload = serialize_trainer_args(trainer_state) or {}

    state_path = os.path.join(output_dir, f"{stage_prefix}_trainer_state.json")
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state_payload, handle, indent=2, ensure_ascii=True, default=str)

    log_history = state_payload.get("log_history")
    if isinstance(log_history, list):
        log_path = os.path.join(output_dir, f"{stage_prefix}_log_history.json")
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(log_history, handle, indent=2, ensure_ascii=True, default=str)


def load_json_dict(file_path: str | None) -> dict | None:
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_trainer_state(output_dir: str | None) -> dict | None:
    if not output_dir:
        return None
    return load_json_dict(os.path.join(output_dir, "trainer_state.json"))


def is_stage_complete_from_state(state: dict | None) -> bool:
    if not state:
        return False
    global_step = safe_int(state.get("global_step"))
    max_steps = safe_int(state.get("max_steps"))
    return (
        global_step is not None
        and max_steps is not None
        and max_steps > 0
        and global_step >= max_steps
    )


def inspect_stage_progress(output_dir: str, stage: str) -> dict:
    latest_checkpoint_dir = find_latest_checkpoint_dir(output_dir)
    stage_status = load_json_dict(os.path.join(output_dir, STAGE_STATUS_FILENAME)) or {}
    root_state = load_trainer_state(output_dir)
    checkpoint_state = load_trainer_state(latest_checkpoint_dir)

    global_step = None
    max_steps = None
    epoch = None
    for candidate in (stage_status, root_state, checkpoint_state):
        if global_step is None:
            global_step = safe_int(candidate.get("global_step")) if candidate else None
        if max_steps is None:
            max_steps = safe_int(candidate.get("max_steps")) if candidate else None
        if epoch is None:
            epoch = safe_float(candidate.get("epoch")) if candidate else None

    has_saved_model = output_dir_has_saved_model(output_dir)
    completed = bool(stage_status.get("completed")) or is_stage_complete_from_state(root_state)
    completed = completed or is_stage_complete_from_state(checkpoint_state)

    return {
        "stage": stage,
        "output_dir": output_dir,
        "latest_checkpoint_dir": latest_checkpoint_dir,
        "has_checkpoint": latest_checkpoint_dir is not None,
        "has_saved_model": has_saved_model,
        "completed": completed,
        "global_step": global_step,
        "max_steps": max_steps,
        "epoch": epoch,
    }


def format_stage_progress(progress: dict) -> str:
    parts = [f"completed={progress['completed']}"]
    if progress["global_step"] is not None:
        if progress["max_steps"] is not None:
            parts.append(f"steps={progress['global_step']}/{progress['max_steps']}")
        else:
            parts.append(f"steps={progress['global_step']}")
    elif progress["max_steps"] is not None:
        parts.append(f"max_steps={progress['max_steps']}")
    if progress["epoch"] is not None:
        parts.append(f"epoch={progress['epoch']:.4f}")
    parts.append(f"checkpoint={progress['has_checkpoint']}")
    parts.append(f"saved_model={progress['has_saved_model']}")
    return ", ".join(parts)


def cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def maybe_torch_distributed_barrier() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        return


def wait_for_condition(
    predicate,
    *,
    description: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> None:
    if is_primary_process():
        return
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out while waiting for {description}")


def safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
