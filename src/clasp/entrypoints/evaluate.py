"""Entry point: checkpoint evaluation (``clasp-eval``).

Generates candidates for each task's eval split and scores them with the same reward and
feasibility code used in training, so eval and training stay aligned. The protocol is raw
best-of-N with no inference-time repair or reranking. Per-task CSV, a JSON run summary, and
a streaming JSONL detail file are written under ``<inference_root>/<stage>/<checkpoint>/``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from statistics import mean, median, stdev
from typing import Any

from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    FOARL_MATCHED_B9_CONFIRMATION_PAIR,
    FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST,
    is_foarl_matched_b9_confirmation_pair,
)
from clasp.data.benchmark import (
    benchmark_row_sha256,
    reference_output_sha256,
    semantic_instance_sha256,
    validate_forbidden_source_registry,
    validate_frozen_benchmark_manifest,
)
from clasp.data.loaders import (
    active_tsp_representation_settings,
    list_split_json_files,
    load_and_merge_task_datasets,
    transform_example_for_task,
    transform_tsp_example_for_label_tta_variant,
    tsp_label_tta_schedule_from_env,
)
from clasp.data.prompts import build_formatted_prompt, normalize_prompt_style, resolve_prompt_style
from clasp.models.checkpoint_fidelity import RECEIPT_FILENAME as CHECKPOINT_FIDELITY_FILENAME
from clasp.models.exact_peft_kl_post_step import (
    POST_STEP_RECEIPT_FILENAME as EXACT_PEFT_KL_POST_STEP_FILENAME,
)
from clasp.models.exact_peft_kl_post_step import (
    STEP1_RECEIPT_FILENAME as EXACT_PEFT_KL_STEP1_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_REFERENCE_FILENAME,
)
from clasp.models.loading import load_model_and_tokenizer, resolve_tokenizer_path
from clasp.models.publication_fidelity import (
    exact_peft_kl_post_step_lineage_identity,
    exact_peft_kl_runtime_identity,
    fixed_kl_reference_identity,
    publication_fidelity_identity,
    validate_exact_peft_kl_post_step_lineage,
    validate_exact_peft_kl_runtime_identity,
    validate_fixed_kl_reference_identity,
    validate_publication_fidelity_identity,
)
from clasp.rewards.co_tasks import (
    TASK_COST_COMPONENTS,
    _check_cvrp_constraints,
    _check_jssp_constraints,
    _check_mis_constraints,
    _check_op_constraints,
    _check_tsp_constraints,
    _combined_context,
    _cvrp_meta,
    _cvrp_total_distance,
    _distance,
    _jssp_jobs,
    _mis_meta,
    _op_meta,
    _op_prize,
    _parse_cvrp_solution,
    _parse_jssp_solution,
    _parse_mis_solution,
    _parse_op_solution,
    _parse_tsp_solution,
    _schedule_makespan,
    _tsp_coords_from_instance,
)
from clasp.runtime import (
    DEFAULT_RL_ROOT,
    DEFAULT_SFT_ROOT,
    HYPERPARAMS_FILENAME,
    cleanup_torch_memory,
    env_world_size,
    find_latest_checkpoint_dir,
    infer_model_context_window,
    inspect_stage_progress,
    output_dir_has_saved_model,
    parse_task_list,
    print_primary,
    resolve_base_model_name,
    resolve_model_artifact_name,
    sanitize_run_name_component,
)

TASK_DIRECTIONS = {
    "tsp": "min",
    "cvrp": "min",
    "op": "max",
    "mis": "max",
    "jssp": "min",
}

_RUN_EVAL_FILE = os.path.abspath(__file__)
_RUN_EVAL_DIR = os.path.dirname(_RUN_EVAL_FILE)
_RUN_EVAL_PARENT = os.path.dirname(_RUN_EVAL_DIR)

EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 1
STRICT_VERIFIER_SCHEMA_VERSION = 1

_INFERENCE_ARTIFACT_EXACT_NAMES = {
    "adapter_config.json",
    "added_tokens.json",
    "chat_template.jinja",
    CHECKPOINT_FIDELITY_FILENAME,
    EXACT_PEFT_KL_REFERENCE_FILENAME,
    EXACT_PEFT_KL_STEP1_FILENAME,
    EXACT_PEFT_KL_POST_STEP_FILENAME,
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
_INFERENCE_WEIGHT_PREFIXES = (
    "adapter_model",
    "model",
    "pytorch_model",
)
_INFERENCE_ARTIFACT_SUFFIXES = (
    ".jinja",
    ".model",
    ".py",
    ".safetensors",
    ".tiktoken",
)
_TOKENIZER_ARTIFACT_EXACT_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_ARTIFACT_SUFFIXES = (".jinja", ".model", ".tiktoken")

OP_STRICT_BUDGET_HINT = (
    "Important OP feasibility rule: the route is invalid if its total travel "
    "length exceeds the maximum route length. Prefer a lower-prize feasible "
    "route over any over-budget route, and internally double-check the route "
    "length before writing the final answer."
)


def _op_prompt_budget_scale() -> float:
    raw = str(os.environ.get("CLASP_OP_PROMPT_BUDGET_SCALE", "1.0")).strip()
    try:
        scale = float(raw)
    except ValueError as exc:
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE must be a float") from exc
    if not (0.0 < scale <= 1.0):
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE must be in (0, 1]")
    return scale


def _op_prompt_budget_scale_min_size() -> int:
    raw = str(os.environ.get("CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE", "0")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE must be an integer") from exc
    if value < 0:
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE_MIN_SIZE must be >= 0")
    return value


def _op_prompt_budget_scale_max_size() -> int:
    raw = str(os.environ.get("CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE", "0")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE must be an integer") from exc
    if value < 0:
        raise ValueError("CLASP_OP_PROMPT_BUDGET_SCALE_MAX_SIZE must be >= 0")
    return value


def _op_prompt_budget_scale_size_ranges() -> list[tuple[int, int]]:
    raw = str(os.environ.get("CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES", "")).strip()
    if not raw:
        return []
    ranges: list[tuple[int, int]] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
        else:
            lo_raw = hi_raw = item
        try:
            lo = int(lo_raw.strip())
            hi = int(hi_raw.strip())
        except ValueError as exc:
            raise ValueError(
                "CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES must contain "
                "integer sizes or inclusive ranges like '20-24,80-95'"
            ) from exc
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError(
                "CLASP_OP_PROMPT_BUDGET_SCALE_SIZE_RANGES ranges must satisfy 0 <= lo <= hi"
            )
        ranges.append((lo, hi))
    return ranges


def _op_size_band_top_p_min_size() -> int:
    raw = str(os.environ.get("CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE", "0")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE must be an integer") from exc
    if value < 0:
        raise ValueError("CLASP_OP_SIZE_BAND_TOP_P_MIN_SIZE must be >= 0")
    return value


def _op_size_band_top_p_max_size() -> int:
    raw = str(os.environ.get("CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE", "0")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE must be an integer") from exc
    if value < 0:
        raise ValueError("CLASP_OP_SIZE_BAND_TOP_P_MAX_SIZE must be >= 0")
    return value


def _op_instruction_size(instruction: str) -> int | None:
    size_match = re.search(
        r"\bwith\s+(\d+)\s+(?:locations|nodes)\b",
        str(instruction),
        re.IGNORECASE,
    )
    if not size_match:
        size_match = re.search(
            r"\b(\d+)\s+(?:locations|nodes)\b",
            str(instruction),
            re.IGNORECASE,
        )
    if not size_match:
        return None
    return int(size_match.group(1))


def _op_size_in_band(size: int | None, min_size: int, max_size: int) -> bool:
    if size is None:
        return False
    if min_size > 0 and int(size) < min_size:
        return False
    if max_size > 0 and int(size) > max_size:
        return False
    return True


def _format_budget_value(value: float) -> str:
    text = f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _op_strict_budget_hint_enabled() -> bool:
    return _env_flag("CLASP_OP_STRICT_BUDGET_HINT", "0")


def _apply_op_prompt_budget_scale(instruction: str, input_text: str) -> tuple[str, str]:
    scale = _op_prompt_budget_scale()
    if abs(scale - 1.0) <= 1e-12:
        return str(instruction), str(input_text)

    min_size = _op_prompt_budget_scale_min_size()
    max_size = _op_prompt_budget_scale_max_size()
    size_ranges = _op_prompt_budget_scale_size_ranges()
    if size_ranges:
        size = _op_instruction_size(instruction)
        if size is None or not any(lo <= int(size) <= hi for lo, hi in size_ranges):
            return str(instruction), str(input_text)
    elif min_size > 0 or max_size > 0:
        size = _op_instruction_size(instruction)
        if not _op_size_in_band(size, min_size, max_size):
            return str(instruction), str(input_text)

    pattern = re.compile(r"(maximum route length\s+T\s*=\s*)(-?\d+(?:\.\d+)?)", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        original = float(match.group(2))
        return match.group(1) + _format_budget_value(original * scale)

    updated_instruction, count = pattern.subn(replace, str(instruction), count=1)
    if count > 0:
        return updated_instruction, str(input_text)

    fallback = re.compile(r"(\bT\s*=\s*)(-?\d+(?:\.\d+)?)")
    updated_instruction, count = fallback.subn(replace, str(instruction), count=1)
    if count > 0:
        return updated_instruction, str(input_text)

    return str(instruction), str(input_text)


def _apply_task_prompt_hints(task: str, instruction: str, input_text: str) -> tuple[str, str]:
    if str(task).lower() == "op":
        instruction, input_text = _apply_op_prompt_budget_scale(instruction, input_text)
        if _op_strict_budget_hint_enabled():
            instruction = str(instruction).rstrip() + "\n\n" + OP_STRICT_BUDGET_HINT
    return str(instruction), str(input_text)


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate latest or explicit SFT/GRPO checkpoints on CO eval sets."
    )
    parser.add_argument("--tasks", type=str, default="tsp,cvrp,op,mis,jssp")
    parser.add_argument("--stage", choices=["sft", "grpo"], default="grpo")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer source path. Defaults to the resolved model path. "
            "Useful for evaluating Gemma 4 base checkpoints with a matching *-it tokenizer."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Alias of --model_path for explicitly selecting a checkpoint/model directory.",
    )
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["auto", "alpaca", "alpaca_chat", "alpaca_hint", "jiang_f638"],
        default="auto",
        help=(
            "Prompt rendering strategy. "
            "When set to auto, evaluation follows the stage hyperparameters when available, "
            "otherwise falls back to Gemma-aware automatic resolution."
        ),
    )
    prompt_override_group = parser.add_mutually_exclusive_group()
    prompt_override_group.add_argument(
        "--force_reasoning_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate with the reasoning-capable version of the resolved prompt template "
            "even when the checkpoint stage metadata says no_reason. Default is off."
        ),
    )
    prompt_override_group.add_argument(
        "--force_no_reason_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate with the no-reason version of the resolved prompt template even "
            "when checkpoint metadata is missing or says reasoning. Default is off."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Defaults to the SFT data root when --stage sft, otherwise the RL data root.",
    )
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument(
        "--evaluation_purpose",
        choices=["development", "replication", "confirmation"],
        default="development",
        help=(
            "Declare how outputs may be used. Confirmation requires a frozen benchmark "
            "manifest and disables all best-run/checkpoint-selection side effects."
        ),
    )
    parser.add_argument(
        "--benchmark_manifest",
        type=str,
        default=None,
        help=(
            "Required for --evaluation_purpose confirmation. The self-hashed frozen "
            "manifest is content-bound into the experiment fingerprint."
        ),
    )
    parser.add_argument(
        "--checkpoint_selection_receipt",
        type=str,
        default=None,
        help=(
            "Required for confirmation of a GRPO checkpoint. Must be the self-hashed "
            "receipt produced from frozen development-only checkpoint-selection evidence."
        ),
    )
    parser.add_argument(
        "--campaign_manifest",
        type=str,
        default=None,
        help=(
            "Required for confirmation. Must be the self-hashed campaign frozen before "
            "inference and contain this exact task/method/seed checkpoint plus its unique "
            "confirmation_attempt_id."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch sampling seed recorded with the raw evaluation run.",
    )
    parser.add_argument(
        "--prefer_subset_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer *_subset.json files for evaluation when available.",
    )
    parser.add_argument(
        "--max_eval_samples_per_task",
        type=int,
        default=0,
        help="Cap the number of loaded examples per CO task. Use 0 for all available eval examples.",
    )
    parser.add_argument(
        "--eval_start_index",
        type=int,
        default=0,
        help=(
            "Skip this many examples in the deterministic selected eval ordering before "
            "applying the per-task cap. Use a nonzero offset for a checkpoint-selection "
            "slice disjoint from the final first-N comparison slice."
        ),
    )
    parser.add_argument(
        "--eval_samples_per_task",
        type=str,
        default="",
        help=(
            "Optional per-task eval caps. Format: 'tsp:50,cvrp:25,op:40,mis:30,jssp:20'. "
            "Listed tasks override --max_eval_samples_per_task."
        ),
    )
    parser.add_argument(
        "--size_bucket",
        choices=["all", "small", "medium", "large", "hardest"],
        default="all",
        help=(
            "Filter eval instances by inferred CO problem size before applying sample caps. "
            "Use 'hardest' for the largest 20%% of each task, or small/medium/large thirds."
        ),
    )
    parser.add_argument(
        "--min_instance_size",
        type=int,
        default=0,
        help="Optional inclusive lower bound on inferred problem size. Use 0 to disable.",
    )
    parser.add_argument(
        "--max_instance_size",
        type=int,
        default=0,
        help="Optional inclusive upper bound on inferred problem size. Use 0 to disable.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=1,
        help=(
            "How many return sequences to generate in one model.generate call. "
            "Keep this small for long prompts/best-of-N inference to avoid OOM kills."
        ),
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sampling defaults to True when num_return_sequences > 1, otherwise False.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--temperature_schedule",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-candidate temperatures for fair diversified BoN. "
            "When set, the number of values must equal --num_return_sequences; default "
            "--temperature is used only when the schedule is empty."
        ),
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--top_p_schedule",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-candidate top-p values for fair diversified BoN. "
            "When set, the number of values must equal --num_return_sequences; default "
            "--top_p is used only when the schedule is empty."
        ),
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--inference_root",
        type=str,
        default=None,
        help=(
            "Root directory for inference outputs. Results are written under "
            "<root>/<stage>/<checkpoint_name>/."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=("Deprecated alias of --inference_root."),
    )
    parser.add_argument("--eval_name", type=str, default=None)
    parser.add_argument(
        "--update_best_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to the legacy mutable best-run snapshot for development only. "
            "Publication checkpoint selection uses the separate frozen selector."
        ),
    )
    parser.add_argument(
        "--is_print",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-instance metrics plus the input problem, generated candidates, and ground-truth solution.",
    )
    parser.add_argument(
        "--save_details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-example candidate details to details.jsonl.",
    )
    parser.add_argument(
        "--save_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write inference artifacts to disk. Disable with --no-save_results for print-only eval.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume partially completed saved inference runs by reusing completed "
            "records in run_details__*.jsonl. Disable with --no-resume to start over."
        ),
    )
    return parser


def _parse_per_task_eval_caps(raw: str, tasks: list[str]) -> dict[str, int]:
    if not str(raw).strip():
        return {}

    result: dict[str, int] = {}
    allowed = set(tasks)
    entries = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"--eval_samples_per_task entries must use 'task:value' format, got {entry!r}."
            )
        task_name, value_text = [part.strip().lower() for part in entry.split(":", 1)]
        if task_name not in allowed:
            raise ValueError(
                f"--eval_samples_per_task contains unsupported task {task_name!r}. "
                f"Active tasks are: {sorted(allowed)}."
            )
        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid eval cap for task {task_name!r}: {value_text!r}. "
                "Expected a non-negative integer."
            ) from exc
        if value < 0:
            raise ValueError(f"Invalid eval cap for task {task_name!r}: {value}. Must be >= 0.")
        result[task_name] = value
    return result


def _parse_temperature_schedule(raw: str, num_return_sequences: int) -> list[float]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    values: list[float] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(
                f"--temperature_schedule must be a comma-separated list of floats; got {item!r}."
            ) from exc
        if value <= 0:
            raise ValueError(f"--temperature_schedule values must be positive; got {value}.")
        values.append(value)
    if len(values) != int(num_return_sequences):
        raise ValueError(
            "--temperature_schedule length must equal --num_return_sequences. "
            f"Got {len(values)} values for num_return_sequences={num_return_sequences}."
        )
    return values


def _parse_top_p_schedule(raw: str, num_return_sequences: int) -> list[float]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    values: list[float] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(
                f"--top_p_schedule must be a comma-separated list of floats; got {item!r}."
            ) from exc
        if value <= 0 or value > 1:
            raise ValueError(
                f"--top_p_schedule values must be in the interval (0, 1]; got {value}."
            )
        values.append(value)
    if len(values) != int(num_return_sequences):
        raise ValueError(
            "--top_p_schedule length must equal --num_return_sequences. "
            f"Got {len(values)} values for num_return_sequences={num_return_sequences}."
        )
    return values


def parse_eval_args():
    args = build_eval_parser().parse_args()
    args.tasks = parse_task_list(args.tasks)
    args.tokenizer_path = (
        (str(args.tokenizer_path).strip() or None) if args.tokenizer_path is not None else None
    )
    args.prompt_style = str(args.prompt_style).strip().lower()
    if args.checkpoint_path and args.model_path:
        if os.path.abspath(args.checkpoint_path) != os.path.abspath(args.model_path):
            raise ValueError(
                "--model_path and --checkpoint_path were both provided with different values. "
                "Use only one of them or provide identical paths."
            )
    if args.checkpoint_path and not args.model_path:
        args.model_path = args.checkpoint_path
    if args.inference_root and args.output_dir:
        if os.path.abspath(args.inference_root) != os.path.abspath(args.output_dir):
            raise ValueError(
                "--inference_root and --output_dir were both provided with different values. "
                "Use only one of them or provide identical paths."
            )
    if args.inference_root and not args.output_dir:
        args.output_dir = args.inference_root
    args.explicit_data_root = args.data_root is not None
    if int(args.num_return_sequences) < 1:
        raise ValueError("--num_return_sequences must be >= 1")
    if int(args.generation_batch_size) < 1:
        raise ValueError("--generation_batch_size must be >= 1")
    if args.max_prompt_length is not None and int(args.max_prompt_length) < 0:
        raise ValueError("--max_prompt_length must be >= 0")
    if args.max_new_tokens is not None and int(args.max_new_tokens) < 0:
        raise ValueError("--max_new_tokens must be >= 0")
    if int(args.max_eval_samples_per_task) < 0:
        raise ValueError("--max_eval_samples_per_task must be >= 0")
    if int(args.eval_start_index) < 0:
        raise ValueError("--eval_start_index must be >= 0")
    if int(args.min_instance_size) < 0:
        raise ValueError("--min_instance_size must be >= 0")
    if int(args.max_instance_size) < 0:
        raise ValueError("--max_instance_size must be >= 0")
    if int(args.max_instance_size) > 0 and int(args.min_instance_size) > int(
        args.max_instance_size
    ):
        raise ValueError("--min_instance_size cannot be larger than --max_instance_size")
    args.eval_samples_per_task_map = _parse_per_task_eval_caps(
        args.eval_samples_per_task,
        list(args.tasks),
    )

    purpose = str(args.evaluation_purpose)
    if bool(args.update_best_run) and purpose != "development":
        raise ValueError("--update_best_run is allowed only for development evaluation")
    if purpose == "confirmation":
        if not args.benchmark_manifest:
            raise ValueError("--evaluation_purpose confirmation requires --benchmark_manifest")
        if not args.campaign_manifest:
            raise ValueError("--evaluation_purpose confirmation requires --campaign_manifest")
        if not bool(args.save_results) or not bool(args.save_details):
            raise ValueError("confirmation requires --save_results and --save_details")
        if not bool(args.resume):
            raise ValueError(
                "confirmation requires --resume so an existing fingerprinted run cannot "
                "be cleared and selectively regenerated"
            )
        if str(args.split) != "eval":
            raise ValueError("confirmation requires --split eval")
        if int(args.eval_start_index) != 0:
            raise ValueError("confirmation requires --eval_start_index 0")
        if int(args.max_eval_samples_per_task) != 0 or args.eval_samples_per_task_map:
            raise ValueError(
                "confirmation must evaluate every frozen row; use "
                "--max_eval_samples_per_task 0 and no per-task caps"
            )
        if (
            str(args.size_bucket) != "all"
            or int(args.min_instance_size) != 0
            or int(args.max_instance_size) != 0
        ):
            raise ValueError("confirmation forbids instance-size filtering")
        if bool(args.prefer_subset_files):
            raise ValueError("confirmation requires --no-prefer_subset_files")
        if str(args.stage) == "grpo" and not args.checkpoint_selection_receipt:
            raise ValueError("confirmation of a GRPO model requires --checkpoint_selection_receipt")
    elif args.benchmark_manifest:
        raise ValueError("--benchmark_manifest is reserved for --evaluation_purpose confirmation")
    elif args.checkpoint_selection_receipt:
        raise ValueError("--checkpoint_selection_receipt is reserved for confirmation evaluation")
    elif args.campaign_manifest:
        raise ValueError("--campaign_manifest is reserved for confirmation evaluation")

    args.output_root = os.path.abspath(args.output_root)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    if args.model_path:
        args.model_path = os.path.abspath(args.model_path)
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        args.tokenizer_path = os.path.abspath(args.tokenizer_path)

    if args.data_root is not None:
        args.data_root = os.path.abspath(args.data_root)
    if args.benchmark_manifest is not None:
        args.benchmark_manifest = os.path.abspath(args.benchmark_manifest)
        (
            args.benchmark_manifest_payload,
            args.benchmark_manifest_identity,
        ) = _load_benchmark_manifest_snapshot(args.benchmark_manifest)
        manifest_tasks = set(args.benchmark_manifest_identity["tasks"])
        missing_tasks = sorted(set(args.tasks) - manifest_tasks)
        if missing_tasks:
            raise ValueError(
                f"confirmation tasks are absent from benchmark manifest: {missing_tasks}"
            )
    else:
        args.benchmark_manifest_identity = None
        args.benchmark_manifest_payload = None
    if args.checkpoint_selection_receipt is not None:
        args.checkpoint_selection_receipt = os.path.abspath(args.checkpoint_selection_receipt)
        (
            args.checkpoint_selection_receipt_identity,
            args.checkpoint_selection_receipt_payload,
        ) = _checkpoint_selection_receipt_identity(args.checkpoint_selection_receipt)
    else:
        args.checkpoint_selection_receipt_identity = None
        args.checkpoint_selection_receipt_payload = None
    if args.campaign_manifest is not None:
        args.campaign_manifest = os.path.abspath(args.campaign_manifest)
        (
            args.campaign_manifest_payload,
            args.campaign_manifest_identity,
        ) = _load_campaign_manifest_snapshot(args.campaign_manifest)
    else:
        args.campaign_manifest_payload = None
        args.campaign_manifest_identity = None
    return args


def _progress_iter(iterable, *, total: int, desc: str):
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
    except Exception:
        if total <= 0:
            return iterable

        update_every = max(1, total // 20)

        def _fallback():
            print_primary(f"[Eval][Progress] {desc}: 0/{total}")
            for index, item in enumerate(iterable, start=1):
                yield item
                if index == total or index % update_every == 0:
                    print_primary(f"[Eval][Progress] {desc}: {index}/{total}")

        return _fallback()


def _candidate_stage_data_roots(stage: str, explicit_data_root: str | None) -> list[str]:
    stage_dirname = "data_sft" if stage == "sft" else "data_rl"
    candidates = []

    if explicit_data_root:
        candidates.append(os.path.abspath(explicit_data_root))

    project_root_env = os.environ.get("PROJECT_ROOT")
    if project_root_env:
        project_root_env = os.path.abspath(project_root_env)
        candidates.append(os.path.join(project_root_env, "data", stage_dirname))
        candidates.append(os.path.join(project_root_env, "clasp", "data", stage_dirname))

    candidates.extend(
        [
            os.path.join(_RUN_EVAL_DIR, "data", stage_dirname),
            os.path.join(_RUN_EVAL_PARENT, "data", stage_dirname),
            os.path.join(os.getcwd(), "data", stage_dirname),
            DEFAULT_SFT_ROOT if stage == "sft" else DEFAULT_RL_ROOT,
        ]
    )

    normalized = []
    seen = set()
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        normalized.append(norm)
    return normalized


def _candidate_split_requirements(
    root: str,
    *,
    tasks: list[str],
    split: str,
) -> list[str]:
    missing = []
    for task in tasks:
        split_dir = os.path.join(root, task, split)
        if not os.path.isdir(split_dir):
            missing.append(split_dir)
    return missing


def _resolve_existing_data_root(args) -> str:
    candidates = _candidate_stage_data_roots(
        args.stage,
        args.data_root if args.explicit_data_root else None,
    )
    print_primary("[Eval] Resolving data root candidates...")
    for candidate in candidates:
        if not os.path.isdir(candidate):
            print_primary(f"[Eval] Skipping missing data root candidate: {candidate}")
            continue

        missing_requirements = _candidate_split_requirements(
            candidate,
            tasks=list(args.tasks),
            split=str(args.split),
        )
        if missing_requirements:
            print_primary(
                "[Eval] Skipping incomplete data root candidate: "
                f"{candidate} "
                f"(missing {len(missing_requirements)} split dirs, "
                f"for example {missing_requirements[0]})"
            )
            continue

        if candidate != os.path.abspath(args.data_root or candidate):
            print_primary(f"[Eval] Data root fallback: {args.data_root or 'auto'} -> {candidate}")
        return candidate

    if args.explicit_data_root:
        raise FileNotFoundError(
            f"--data_root is missing one or more required task/split directories: {args.data_root}. "
            f"Expected tasks={list(args.tasks)} split={args.split}."
        )

    raise FileNotFoundError(
        "Could not resolve an evaluation data root automatically. "
        f"Tried: {candidates}. "
        f"Expected tasks={list(args.tasks)} split={args.split}."
    )


def _json_load(file_path: str | None) -> dict | None:
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _canonical_json_bytes(payload: Any) -> bytes:
    """Serialize fingerprint material in a stable, locale-independent form."""

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_file(file_path: str) -> tuple[str, int]:
    """Hash a stable file snapshot and reject concurrent modification."""

    before = os.stat(file_path)
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = os.stat(file_path)
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"File changed while computing its experiment identity: {file_path}")
    return digest.hexdigest(), int(after.st_size)


def _read_file_snapshot(file_path: str) -> tuple[bytes, dict[str, Any]]:
    """Read and hash one immutable file snapshot without a parse/hash split."""

    normalized = os.path.realpath(os.path.abspath(file_path))
    path_before = os.stat(normalized)
    with open(normalized, "rb") as handle:
        descriptor_before = os.fstat(handle.fileno())
        payload = handle.read()
        descriptor_after = os.fstat(handle.fileno())
    path_after = os.stat(normalized)

    def signature(stat_result) -> tuple[int, int, int, int]:
        return (
            int(stat_result.st_dev),
            int(stat_result.st_ino),
            int(stat_result.st_size),
            int(stat_result.st_mtime_ns),
        )

    expected = signature(descriptor_before)
    if (
        expected != signature(descriptor_after)
        or expected != signature(path_before)
        or expected != signature(path_after)
        or len(payload) != int(descriptor_after.st_size)
    ):
        raise RuntimeError(f"File changed while reading its content snapshot: {normalized}")
    return payload, {
        "path": normalized,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_json_snapshot(file_path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, identity = _read_file_snapshot(file_path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {identity['path']}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {identity['path']}")
    return payload, identity


_CONFIRMATION_TRAIN_SEEDS = list(range(42, 50))
_CONFIRMATION_CAMPAIGN_SCHEMA_VERSION = 2
_CONFIRMATION_METHODS = frozenset(
    {
        "grpo_z",
        "dual_scalar_z",
        "dual_separate_z",
        "clasp_canonical",
        "clasp_canonical_fixed_lambda",
        "clasp_canonical_no_recovery",
        CLASP_FOARL_MATCHED_B9,
        "set_null",
        "foarl_f638_native",
        "foarl_f638_matched_b9",
    }
)
_LEGACY_CONFIRMATION_METHOD_ALIASES = frozenset(
    {"grpo", "dual_grpo", "clasp_dense", "clasp_exact", "foarl_f638_corrected"}
)
_CONFIRMATION_PRIMARY_FAMILY = "normalized_feasible_gap_auc_seed_randomization"
_CONFIRMATION_NONINFERIORITY_FAMILY = "pass_at_k_noninferiority_seed_randomization"
_APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK = {
    "cvrp": "d4c7f8e4924bee2bf54b3f85c5eb39f69293d858dda855685d2bacff703df217",
    "jssp": "f67cc9befc71086ab1e5bcf419af731ba6e9a62647fc268684b83b997954fac5",
    "mis": "ab9b34ec5d97854489a3a7bf10d2875c918565ff1d6e486a542d35304cd256d2",
    "op": "e5bc6baa29ba685fd4806599d9221fe81a3c5b68e63b7dbf11a8bc7d6002aeb4",
    "tsp": "26d59b60957f7cc866947bdb99baaf0c5aaddeaf68720465e76e352213b30dbf",
}
_CONFIRMATION_METHOD_SPECIFIC_ALLOWLIST = (
    "training_config.entrypoint",
    "training_config.policy_credit_formula",
    "training_config.component_controller",
    "training_config.set_controller",
    "training_config.use_primal_dual",
    "training_config.use_clasp",
    "training_config.clasp_canonical_set",
    "training_config.clasp_fixed_null_set",
    "training_config.pd_group_scalarization",
    "training_config.pd_separate_z",
    "training_config.pd_dual_lr",
    "training_config.pd_dual_lr_up",
    "training_config.pd_dual_lr_down",
    "training_config.clasp_bon_dual_lr",
    "training_config.clasp_bon_lambda_init",
    "training_config.clasp_canonical_recovery_scale",
    "tracked_environment.CLASP_BON_ALIGNED",
    "tracked_environment.CLASP_FEASIBILITY_GATED",
    "tracked_environment.CLASP_CANONICAL_SET",
    "tracked_environment.CLASP_FIXED_NULL_SET",
    "tracked_environment.CLASP_BON_DUAL_LR",
    "tracked_environment.CLASP_BON_LAMBDA_INIT",
    "tracked_environment.CLASP_CANONICAL_RECOVERY_SCALE",
)
_CONFIRMATION_ATTEMPT_PROTOCOL = "clasp-confirmation-attempt-v2"


def _validate_confirmation_analysis_contract(manifest: dict[str, Any]) -> None:
    analysis = manifest.get("analysis")
    expected_primary = {
        "metric": "normalized_feasible_gap_success_curve_auc",
        "gap_range": [0.0, 0.10],
        "infeasible_score": 0.0,
        "estimand": "paired_seed_mean_challenger_minus_baseline",
        "inference_unit": "paired_training_seed",
        "test": "exact_two_sided_paired_seed_sign_flip",
        "multiplicity": "holm_across_five_tasks",
        "direction": "higher_is_better",
        "decision_rule": "holm_adjusted_p <= alpha and effect_mean > 0",
        "negative_direction_label": "difference_detected_against_clasp",
        "family": _CONFIRMATION_PRIMARY_FAMILY,
    }
    expected_noninferiority = {
        "metric": "strict_shared_verifier_pass_at_8",
        "margin": 0.01,
        "null_hypothesis": "challenger_minus_baseline <= -0.01",
        "test": "exact_one_sided_paired_seed_sign_flip_at_margin_boundary",
        "multiplicity": "holm_across_five_tasks",
        "observed_mean_requirement": "challenger_minus_baseline >= -0.01",
        "overall_rule": "all_five_tasks_pass",
        "family": _CONFIRMATION_NONINFERIORITY_FAMILY,
    }
    expected_secondary = [
        "candidate_feasibility",
        "conditional_verified_gap",
        "lexicographic_win_score",
        "success_at_external_target_gap_when_eligible",
        "generation_and_repair_compute",
    ]
    if (
        not isinstance(analysis, dict)
        or set(analysis)
        != {
            "bootstrap_repetitions",
            "bootstrap_seed",
            "alpha",
            "primary_outcome",
            "pass_at_k_noninferiority_gate",
            "secondary_metrics",
        }
        or isinstance(analysis.get("bootstrap_repetitions"), bool)
        or not isinstance(analysis.get("bootstrap_repetitions"), int)
        or int(analysis["bootstrap_repetitions"]) < 10_000
        or isinstance(analysis.get("bootstrap_seed"), bool)
        or not isinstance(analysis.get("bootstrap_seed"), int)
        or int(analysis["bootstrap_seed"]) < 0
        or analysis.get("alpha") != 0.05
        or analysis.get("primary_outcome") != expected_primary
        or analysis.get("pass_at_k_noninferiority_gate") != expected_noninferiority
        or analysis.get("secondary_metrics") != expected_secondary
    ):
        raise ValueError("campaign manifest has no exact eight-seed confirmation analysis contract")
    sign_assignments = 1 << len(_CONFIRMATION_TRAIN_SEEDS)
    if len(TASK_DIRECTIONS) * (2.0 / sign_assignments) > 0.05:
        raise ValueError("campaign seed design cannot resolve its primary Holm family")


def _validate_confirmation_training_design_contract(manifest: dict[str, Any]) -> None:
    policy = manifest.get("training_design")
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "fixed_horizon",
        "method_specific_allowlist",
        "shared_design_sha256_by_task_seed",
        "fixed_null_receipt_sha256_by_task",
    }:
        raise ValueError("campaign manifest has no exact matched training-design contract")
    horizon = policy.get("fixed_horizon")
    methods = manifest.get("methods")
    foarl_comparison = (
        isinstance(methods, list)
        and is_foarl_matched_b9_confirmation_pair(methods)
    )
    expected_allowlist = (
        list(FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST)
        if foarl_comparison
        else list(_CONFIRMATION_METHOD_SPECIFIC_ALLOWLIST)
    )
    if (
        policy.get("schema_version") != 1
        or not isinstance(horizon, dict)
        or set(horizon) != {"max_steps", "world_size"}
        or any(
            isinstance(horizon.get(field), bool)
            or not isinstance(horizon.get(field), int)
            or int(horizon[field]) <= 0
            for field in ("max_steps", "world_size")
        )
        or policy.get("method_specific_allowlist") != expected_allowlist
    ):
        raise ValueError("campaign manifest matched training-design policy is invalid")
    tasks = manifest.get("tasks")
    seeds = manifest.get("training_seeds")
    expected_groups = {
        f"{task}/seed{seed}"
        for task in tasks if isinstance(task, str)
        for seed in seeds if isinstance(seed, int) and not isinstance(seed, bool)
    } if isinstance(tasks, list) and isinstance(seeds, list) else set()
    shared_hashes = policy.get("shared_design_sha256_by_task_seed")
    if (
        not isinstance(shared_hashes, dict)
        or set(shared_hashes) != expected_groups
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in shared_hashes.values())
    ):
        raise ValueError("campaign shared training-design hashes do not cover the exact grid")
    fixed_null = policy.get("fixed_null_receipt_sha256_by_task")
    expected_fixed_null_tasks = set(tasks) if isinstance(methods, list) and "set_null" in methods else set()
    if (
        not isinstance(fixed_null, dict)
        or set(fixed_null) != expected_fixed_null_tasks
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in fixed_null.values())
        or any(
            fixed_null.get(task) != _APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK[task]
            for task in expected_fixed_null_tasks
        )
    ):
        raise ValueError("campaign fixed-null receipt preregistration is invalid")

_CHECKPOINT_SELECTION_FINGERPRINT_FIELDS = (
    "evaluation_fingerprint",
    "protocol_fingerprint",
    "development_evaluation_sha256",
    "checkpoint_artifact_sha256",
    "checkpoint_fidelity_identity_sha256",
    "checkpoint_fidelity_receipt_sha256",
    "checkpoint_fidelity_receipt_file_sha256",
    "run_contract_sha256",
    "data_manifest_sha256",
    "strict_verifier_manifest_sha256",
)
_CHECKPOINT_SELECTION_SELECTED_FIELDS = {
    "task",
    "method",
    "train_seed",
    "checkpoint_step",
    "pass_at_k",
    "feasible_instances",
    "conditional_gap_mean",
    "conditional_gap_n",
    "instances",
    "candidate_budget",
    "grpo_beta",
    "fixed_kl_reference",
    "exact_peft_kl_runtime",
    "exact_peft_kl_post_step",
    *_CHECKPOINT_SELECTION_FINGERPRINT_FIELDS,
}
_CHECKPOINT_SELECTION_CANDIDATE_FIELDS = {
    "checkpoint_step",
    "candidate_budget",
    "instances",
    "feasible_instances",
    "pass_at_k",
    "conditional_gap_n",
    "conditional_gap_mean",
    "grpo_beta",
    "fixed_kl_reference",
    "exact_peft_kl_runtime",
    "exact_peft_kl_post_step",
    *_CHECKPOINT_SELECTION_FINGERPRINT_FIELDS,
    "disposition",
}


def _validate_fixed_kl_projection(
    beta: object,
    fixed_kl_reference: object,
    *,
    label: str,
) -> tuple[float, dict[str, Any] | None]:
    """Validate the portable beta/reference pair embedded in publication receipts."""

    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        raise ValueError(f"{label} has invalid grpo_beta")
    normalized_beta = float(beta)
    if normalized_beta == 0.0:
        if fixed_kl_reference is not None:
            raise ValueError(f"{label} beta=0 must not bind a fixed-KL reference")
        return normalized_beta, None
    try:
        normalized_reference = validate_fixed_kl_reference_identity(fixed_kl_reference)
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 fixed-KL identity is invalid: {exc}") from exc
    if normalized_reference != fixed_kl_reference:
        raise ValueError(f"{label} fixed-KL identity is not canonical")
    return normalized_beta, normalized_reference


def _validate_exact_peft_kl_runtime_projection(
    beta: object,
    fixed_kl_reference: object,
    exact_peft_kl_runtime: object,
    *,
    label: str,
) -> tuple[float, dict[str, Any] | None, dict[str, Any] | None]:
    """Validate the source/runtime chronology embedded in publication receipts."""

    normalized_beta, normalized_reference = _validate_fixed_kl_projection(
        beta,
        fixed_kl_reference,
        label=label,
    )
    if normalized_beta == 0.0:
        if exact_peft_kl_runtime is not None:
            raise ValueError(f"{label} beta=0 must not bind an exact PEFT KL runtime")
        return normalized_beta, None, None
    try:
        normalized_runtime = validate_exact_peft_kl_runtime_identity(exact_peft_kl_runtime)
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 exact PEFT KL runtime is invalid: {exc}") from exc
    if normalized_runtime != exact_peft_kl_runtime:
        raise ValueError(f"{label} exact PEFT KL runtime identity is not canonical")
    if normalized_runtime["fixed_kl_reference"] != normalized_reference:
        raise ValueError(
            f"{label} exact PEFT KL runtime differs from its prelaunch source commitment"
        )
    return normalized_beta, normalized_reference, normalized_runtime


def _validate_exact_peft_kl_post_step_projection(
    beta: object,
    fixed_kl_reference: object,
    exact_peft_kl_runtime: object,
    exact_peft_kl_post_step: object,
    *,
    label: str,
    expected_run_contract_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
) -> tuple[
    float,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Validate step-zero and post-optimization chronology as one unit."""

    normalized_beta, normalized_reference, normalized_runtime = (
        _validate_exact_peft_kl_runtime_projection(
            beta,
            fixed_kl_reference,
            exact_peft_kl_runtime,
            label=label,
        )
    )
    if normalized_beta == 0.0:
        if exact_peft_kl_post_step is not None:
            raise ValueError(f"{label} beta=0 must not bind exact PEFT KL post-step evidence")
        return normalized_beta, None, None, None
    try:
        normalized_post_step = validate_exact_peft_kl_post_step_lineage(
            exact_peft_kl_post_step,
            expected_fixed_kl_reference=normalized_reference,
            expected_runtime_identity=normalized_runtime,
            expected_run_contract_file_sha256=expected_run_contract_sha256,
            expected_selected_checkpoint_step=expected_checkpoint_step,
        )
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 post-step lineage is invalid: {exc}") from exc
    if normalized_post_step != exact_peft_kl_post_step:
        raise ValueError(f"{label} exact PEFT KL post-step lineage is not canonical")
    return normalized_beta, normalized_reference, normalized_runtime, normalized_post_step


def _confirmation_attempt_id(manifest: dict[str, Any], entry: dict[str, Any]) -> str:
    """Derive the preregistered identity of one confirmation grid cell."""

    material = {
        "protocol": _CONFIRMATION_ATTEMPT_PROTOCOL,
        "campaign_id": manifest.get("campaign_id"),
        "method": entry.get("method"),
        "training_seed": entry.get("training_seed"),
        "task": entry.get("task"),
        "training_run_contract_sha256": entry.get("training_run_contract_sha256"),
        "checkpoint_step": entry.get("checkpoint_step"),
        "checkpoint_artifact_sha256": entry.get("checkpoint_artifact_sha256"),
        "checkpoint_fidelity_identity_sha256": entry.get("checkpoint_fidelity_identity_sha256"),
        "checkpoint_fidelity_receipt_sha256": entry.get("checkpoint_fidelity_receipt_sha256"),
        "checkpoint_fidelity_receipt_file_sha256": entry.get(
            "checkpoint_fidelity_receipt_file_sha256"
        ),
        "grpo_beta": entry.get("grpo_beta"),
        "fixed_kl_reference": entry.get("fixed_kl_reference"),
        "exact_peft_kl_runtime": entry.get("exact_peft_kl_runtime"),
        "exact_peft_kl_post_step": entry.get("exact_peft_kl_post_step"),
        "benchmark_manifest_sha256": manifest.get("benchmark_manifest_sha256"),
        "checkpoint_selection_receipt_sha256": entry.get("checkpoint_selection_receipt_sha256"),
    }
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _load_campaign_manifest_snapshot(
    file_path: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Validate the exact campaign frozen before any confirmation inference."""

    manifest, file_identity = _load_json_snapshot(file_path, label="campaign manifest")
    unsigned = dict(manifest)
    stored_hash = str(unsigned.pop("campaign_manifest_sha256", "") or "")
    observed_hash = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored_hash != observed_hash:
        raise ValueError(
            "campaign manifest self-hash mismatch: "
            f"stored={stored_hash or '<missing>'}, observed={observed_hash}"
        )
    campaign_id = manifest.get("campaign_id")
    baseline = manifest.get("baseline")
    challenger = manifest.get("challenger")
    methods = manifest.get("methods")
    tasks = manifest.get("tasks")
    seeds = manifest.get("training_seeds")
    method_pair = {baseline, challenger}
    if (
        manifest.get("schema_version") != _CONFIRMATION_CAMPAIGN_SCHEMA_VERSION
        or manifest.get("status") != "frozen_before_confirmation"
        or not isinstance(manifest.get("frozen_at_utc"), str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(manifest.get("frozen_at_utc", "")),
        )
        is None
        or not isinstance(campaign_id, str)
        or not campaign_id
        or manifest.get("lane") != "raw_n8"
        or not isinstance(baseline, str)
        or not baseline
        or not isinstance(challenger, str)
        or not challenger
        or baseline == challenger
        or methods != [baseline, challenger]
        or seeds != _CONFIRMATION_TRAIN_SEEDS
        or tasks != sorted(TASK_DIRECTIONS)
    ):
        raise ValueError("campaign manifest is not the frozen publication design")
    if any(method in _LEGACY_CONFIRMATION_METHOD_ALIASES for method in method_pair):
        raise ValueError("campaign manifest uses a prohibited legacy method alias")
    if not all(isinstance(method, str) and method in _CONFIRMATION_METHODS for method in method_pair):
        raise ValueError("campaign manifest uses an unsupported confirmation method ID")
    if "foarl_f638_native" in method_pair:
        raise ValueError(
            "foarl_f638_native is descriptive-only and cannot enter a matched confirmation grid"
        )
    ordered_pair = (baseline, challenger)
    foarl_matched_ids = set(FOARL_MATCHED_B9_CONFIRMATION_PAIR)
    if foarl_matched_ids.intersection(method_pair) and not (
        is_foarl_matched_b9_confirmation_pair(ordered_pair)
    ):
        raise ValueError(
            "foarl_f638_matched_b9 confirmation requires the exact ordered pair "
            "baseline=foarl_f638_matched_b9 and "
            f"challenger={CLASP_FOARL_MATCHED_B9}"
        )
    if manifest.get("approved_fixed_null_receipt_sha256_by_task") != (
        _APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
    ):
        raise ValueError("campaign manifest does not bind the approved fixed-null receipts")
    _validate_confirmation_analysis_contract(manifest)
    _validate_confirmation_training_design_contract(manifest)
    for field in (
        "lane_contract_registry_sha256",
        "target_registry_sha256",
        "benchmark_manifest_sha256",
        "forbidden_source_registry_sha256",
        "forbidden_source_tree_manifest_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(field, ""))) is None:
            raise ValueError(f"campaign manifest has invalid {field}")
    selection = manifest.get("checkpoint_selection_receipt")
    selection_identity = selection.get("identity") if isinstance(selection, dict) else None
    selection_payload = selection.get("payload") if isinstance(selection, dict) else None
    if (
        not isinstance(selection_identity, dict)
        or not isinstance(selection_payload, dict)
        or selection_identity.get("campaign_id") != campaign_id
        or re.fullmatch(r"[0-9a-f]{64}", str(selection_identity.get("receipt_sha256", ""))) is None
    ):
        raise ValueError("campaign manifest has no valid common selection receipt")
    attempt_ledger = manifest.get("confirmation_attempt_ledger")
    ledger_root = attempt_ledger.get("canonical_root") if isinstance(attempt_ledger, dict) else None
    if (
        not isinstance(attempt_ledger, dict)
        or set(attempt_ledger) != {"protocol", "canonical_root"}
        or attempt_ledger.get("protocol") != "posix-create-once-lock-v1"
        or not isinstance(ledger_root, str)
        or not os.path.isabs(ledger_root)
        or os.path.normpath(ledger_root) != ledger_root
        or os.path.realpath(ledger_root) != ledger_root
        or ledger_root in {"/", os.path.expanduser("~")}
        or len([part for part in ledger_root.split(os.sep) if part]) < 3
    ):
        raise ValueError("campaign manifest has no safe shared confirmation attempt ledger")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("campaign manifest has no runs list")
    expected_grid = {(method, seed, task) for method in methods for seed in seeds for task in tasks}
    runs_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    seen_attempts: set[str] = set()
    expected_run_fields = {
        "method",
        "training_seed",
        "task",
        "training_run_contract_sha256",
        "checkpoint_step",
        "checkpoint_artifact_sha256",
        "checkpoint_fidelity_identity_sha256",
        "checkpoint_fidelity_receipt_sha256",
        "checkpoint_fidelity_receipt_file_sha256",
        "grpo_beta",
        "fixed_kl_reference",
        "exact_peft_kl_runtime",
        "exact_peft_kl_post_step",
        "checkpoint_selection_receipt_sha256",
        "confirmation_attempt_id",
    }
    for entry in runs:
        if not isinstance(entry, dict) or set(entry) != expected_run_fields:
            raise ValueError("campaign run entry has an invalid exact schema")
        raw_seed = entry.get("training_seed")
        key = (entry.get("method"), raw_seed, entry.get("task"))
        if (
            not isinstance(key[0], str)
            or isinstance(raw_seed, bool)
            or not isinstance(raw_seed, int)
            or not isinstance(key[2], str)
            or key in runs_by_key
        ):
            raise ValueError("campaign run grid contains an invalid or duplicate cell")
        for field in (
            "training_run_contract_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_fidelity_identity_sha256",
            "checkpoint_fidelity_receipt_sha256",
            "checkpoint_fidelity_receipt_file_sha256",
            "checkpoint_selection_receipt_sha256",
            "confirmation_attempt_id",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))) is None:
                raise ValueError(f"campaign run entry has invalid {field}")
        step = entry.get("checkpoint_step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("campaign run entry has invalid checkpoint_step")
        _validate_exact_peft_kl_post_step_projection(
            entry.get("grpo_beta"),
            entry.get("fixed_kl_reference"),
            entry.get("exact_peft_kl_runtime"),
            entry.get("exact_peft_kl_post_step"),
            label="campaign run entry",
            expected_run_contract_sha256=entry.get("training_run_contract_sha256"),
            expected_checkpoint_step=step,
        )
        attempt_id = str(entry["confirmation_attempt_id"])
        if attempt_id != _confirmation_attempt_id(manifest, entry):
            raise ValueError("campaign run confirmation_attempt_id is not deterministic")
        if attempt_id in seen_attempts:
            raise ValueError("campaign run confirmation_attempt_id is not unique")
        seen_attempts.add(attempt_id)
        runs_by_key[key] = entry
    if set(runs_by_key) != expected_grid:
        raise ValueError("campaign manifest runs do not equal its full publication grid")
    identity: dict[str, object] = {
        "basename": os.path.basename(str(file_identity["path"])),
        "file_sha256": file_identity["sha256"],
        "size_bytes": file_identity["size_bytes"],
        "campaign_manifest_sha256": stored_hash,
        "campaign_id": campaign_id,
    }
    return manifest, identity


def _load_benchmark_manifest_snapshot(
    file_path: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Validate and content-bind one exact frozen benchmark-manifest snapshot."""

    manifest, file_identity = _load_json_snapshot(
        file_path,
        label="benchmark manifest",
    )
    validated = validate_frozen_benchmark_manifest(manifest)
    identity: dict[str, object] = {
        "basename": os.path.basename(str(file_identity["path"])),
        "file_sha256": file_identity["sha256"],
        "size_bytes": file_identity["size_bytes"],
        **validated,
    }
    return manifest, identity


def _benchmark_manifest_identity(file_path: str) -> dict[str, object]:
    """Return the identity of a validated frozen confirmation manifest."""

    _manifest, identity = _load_benchmark_manifest_snapshot(file_path)
    return identity


def _checkpoint_selection_receipt_identity(
    file_path: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Validate a frozen development-only checkpoint selection receipt."""

    normalized = os.path.realpath(os.path.abspath(file_path))
    payload, file_identity = _load_json_snapshot(
        normalized,
        label="checkpoint selection receipt",
    )
    unsigned = dict(payload)
    stored_hash = str(unsigned.pop("receipt_sha256", "") or "")
    observed_hash = _experiment_fingerprint(unsigned)
    if stored_hash != observed_hash:
        raise ValueError(
            "checkpoint selection receipt self-hash mismatch: "
            f"stored={stored_hash or '<missing>'}, observed={observed_hash}"
        )
    expected_payload_fields = {
        "schema_version",
        "selector",
        "generated_at_utc",
        "campaign_id",
        "stage_id",
        "decision",
        "evidence_scope",
        "confirmation_evidence_used",
        "policy",
        "evidence",
        "expected_group_count",
        "selected_group_count",
        "selections",
        "errors",
        "side_effects",
        "receipt_sha256",
    }
    if (
        set(payload) != expected_payload_fields
        or payload.get("schema_version") != 2
        or payload.get("selector") != "clasp_development_checkpoint_selection"
        or payload.get("decision") != "SELECTED"
        or payload.get("evidence_scope") != "development_only"
        or payload.get("confirmation_evidence_used") is not False
        or not isinstance(payload.get("generated_at_utc"), str)
        or not payload.get("generated_at_utc")
    ):
        raise ValueError(
            "checkpoint selection receipt is not an accepted development-only SELECTED decision"
        )
    required_sha_fields = _CHECKPOINT_SELECTION_FINGERPRINT_FIELDS
    policy = payload.get("policy")
    evidence = payload.get("evidence")
    side_effects = payload.get("side_effects")
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {
            "path",
            "file_sha256",
            "policy_sha256",
            "frozen_at_utc",
            "selection_rule",
            "pass_at_k_absolute_band",
            "frozen_policy",
        }
        or not isinstance(policy.get("path"), str)
        or not policy.get("path")
        or not isinstance(policy.get("frozen_at_utc"), str)
        or not policy.get("frozen_at_utc")
        or re.fullmatch(r"[0-9a-f]{64}", str(policy.get("file_sha256", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(policy.get("policy_sha256", ""))) is None
        or policy.get("selection_rule")
        != [
            "maximize_pass_at_k",
            "retain_within_0.01_absolute_of_best_pass_at_k",
            "minimize_conditional_verified_gap",
            "tie_break_earlier_checkpoint_step",
        ]
        or policy.get("pass_at_k_absolute_band") != 0.01
    ):
        raise ValueError("checkpoint selection receipt has an invalid frozen policy binding")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"path", "size_bytes", "file_sha256", "evidence_sha256", "record_count"}
        or not isinstance(evidence.get("path"), str)
        or not evidence.get("path")
        or isinstance(evidence.get("size_bytes"), bool)
        or not isinstance(evidence.get("size_bytes"), int)
        or int(evidence["size_bytes"]) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("file_sha256", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("evidence_sha256", ""))) is None
        or isinstance(evidence.get("record_count"), bool)
        or not isinstance(evidence.get("record_count"), int)
        or int(evidence["record_count"]) <= 0
    ):
        raise ValueError("checkpoint selection receipt has an invalid development evidence binding")
    expected_groups, expected_steps, candidate_budget, development_contracts = (
        _validate_embedded_checkpoint_selection_policy(
            descriptor=policy,
            evidence=evidence,
        )
    )
    frozen_policy = policy["frozen_policy"]
    if (
        not isinstance(payload.get("campaign_id"), str)
        or not payload["campaign_id"]
        or not isinstance(payload.get("stage_id"), str)
        or not payload["stage_id"]
        or frozen_policy.get("campaign_id") != payload["campaign_id"]
        or frozen_policy.get("stage_id") != payload["stage_id"]
    ):
        raise ValueError(
            "checkpoint selection receipt campaign/stage does not match its frozen policy"
        )
    if side_effects != {
        "checkpoints_modified": False,
        "evaluator_state_modified": False,
        "scheduler_actions": [],
    }:
        raise ValueError("checkpoint selection receipt reports forbidden side effects")
    if payload.get("errors") != []:
        raise ValueError("checkpoint selection receipt contains errors")
    selections = payload.get("selections")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("checkpoint selection receipt has no selected groups")
    if set(selections) != expected_groups:
        raise ValueError(
            "checkpoint selection receipt does not cover its complete frozen policy grid"
        )
    expected_count = payload.get("expected_group_count")
    selected_count = payload.get("selected_group_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(selections)
        or selected_count != expected_count
    ):
        raise ValueError("checkpoint selection receipt group counts are inconsistent")
    pass_band = float(policy["pass_at_k_absolute_band"])
    audit_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for group_id, group in selections.items():
        if (
            not isinstance(group_id, str)
            or not isinstance(group, dict)
            or group.get("status") != "PASS"
            or group.get("errors") != []
            or not isinstance(group.get("selected"), dict)
            or not isinstance(group.get("selection_audit"), dict)
        ):
            raise ValueError("checkpoint selection receipt contains an invalid group decision")
        selected = group["selected"]
        expected_group_id = (
            f"{selected.get('task')}/{selected.get('method')}/seed{selected.get('train_seed')}"
        )
        if set(group) != {"status", "selected", "selection_audit", "errors"}:
            raise ValueError("checkpoint selection receipt group has an invalid exact schema")
        if set(selected) != _CHECKPOINT_SELECTION_SELECTED_FIELDS:
            raise ValueError("checkpoint selection receipt selected row has an invalid schema")
        if group_id != expected_group_id or any(
            re.fullmatch(r"[0-9a-f]{64}", str(selected.get(field, ""))) is None
            for field in required_sha_fields
        ):
            raise ValueError("checkpoint selection receipt group provenance is invalid")
        if (
            isinstance(selected.get("checkpoint_step"), bool)
            or not isinstance(selected.get("checkpoint_step"), int)
            or int(selected["checkpoint_step"]) <= 0
        ):
            raise ValueError("checkpoint selection receipt checkpoint step is invalid")
        if selected.get("candidate_budget") != candidate_budget:
            raise ValueError(
                "checkpoint selection receipt selected row has the wrong candidate budget"
            )
        selected_beta, selected_fixed_kl, selected_exact_runtime, selected_exact_post = (
            _validate_exact_peft_kl_post_step_projection(
                selected.get("grpo_beta"),
                selected.get("fixed_kl_reference"),
                selected.get("exact_peft_kl_runtime"),
                selected.get("exact_peft_kl_post_step"),
                label=f"checkpoint selection receipt selected row {group_id}",
                expected_run_contract_sha256=selected.get("run_contract_sha256"),
                expected_checkpoint_step=selected.get("checkpoint_step"),
            )
        )
        if (
            selected_beta != selected.get("grpo_beta")
            or selected_fixed_kl != selected.get("fixed_kl_reference")
            or selected_exact_runtime != selected.get("exact_peft_kl_runtime")
            or selected_exact_post != selected.get("exact_peft_kl_post_step")
        ):
            raise ValueError("checkpoint selection receipt selected KL projection is not canonical")
        development_contract = development_contracts.get(str(selected["task"]))
        if (
            development_contract is None
            or selected.get("data_manifest_sha256") != development_contract["data_manifest_sha256"]
            or selected.get("development_evaluation_sha256")
            != development_contract["development_evaluation_sha256"]
            or selected.get("protocol_fingerprint") != development_contract["protocol_fingerprint"]
            or selected.get("strict_verifier_manifest_sha256")
            != development_contract["strict_verifier_manifest_sha256"]
            or selected.get("instances") != development_contract["expected_instances"]
        ):
            raise ValueError(
                "checkpoint selection receipt selected row is not bound to its frozen "
                "development corpus/slice"
            )
        audit_rows = _validate_checkpoint_selection_audit(
            group_id=group_id,
            selected=selected,
            audit=group["selection_audit"],
            pass_band=pass_band,
            expected_steps=expected_steps,
            candidate_budget=candidate_budget,
        )
        audit_rows_by_task.setdefault(str(selected["task"]), []).extend(audit_rows)
    for task, task_rows in audit_rows_by_task.items():
        for field in (
            "protocol_fingerprint",
            "development_evaluation_sha256",
            "data_manifest_sha256",
            "strict_verifier_manifest_sha256",
            "candidate_budget",
            "instances",
            "grpo_beta",
        ):
            if len({row[field] for row in task_rows}) != 1:
                raise ValueError(
                    f"checkpoint selection receipt uses mixed development {field} for task {task}"
                )
        if len({_canonical_json_bytes(row["fixed_kl_reference"]) for row in task_rows}) != 1:
            raise ValueError(
                f"checkpoint selection receipt uses mixed fixed-KL identity for task {task}"
            )
        if len({_canonical_json_bytes(row["exact_peft_kl_runtime"]) for row in task_rows}) != 1:
            raise ValueError(
                f"checkpoint selection receipt uses mixed exact PEFT KL runtime for task {task}"
            )
        positive_post = [
            row["exact_peft_kl_post_step"]
            for row in task_rows
            if row["exact_peft_kl_post_step"] is not None
        ]
        if positive_post and len({row["source_reference_sha256"] for row in positive_post}) != 1:
            raise ValueError(
                f"checkpoint selection receipt uses mixed exact-KL source/reference for {task}"
            )
    identity = {
        "basename": os.path.basename(normalized),
        "file_sha256": file_identity["sha256"],
        "size_bytes": file_identity["size_bytes"],
        "receipt_sha256": stored_hash,
        "campaign_id": str(payload.get("campaign_id", "") or ""),
        "stage_id": str(payload.get("stage_id", "") or ""),
    }
    return identity, payload


def _validate_checkpoint_selection_audit(
    *,
    group_id: str,
    selected: dict[str, Any],
    audit: dict[str, Any],
    pass_band: float,
    expected_steps: list[int],
    candidate_budget: int,
) -> list[dict[str, Any]]:
    """Recompute the prespecified checkpoint rule from its development audit."""

    if set(audit) != {
        "best_pass_at_k",
        "eligibility_threshold",
        "eligible_checkpoint_steps",
        "candidates",
    }:
        raise ValueError(f"checkpoint selection receipt group {group_id} has invalid audit schema")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"checkpoint selection receipt group {group_id} has no candidate audit")
    normalized: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != (
            _CHECKPOINT_SELECTION_CANDIDATE_FIELDS
        ):
            raise ValueError(
                f"checkpoint selection receipt group {group_id} candidate {index} "
                "has an invalid schema"
            )
        step = candidate.get("checkpoint_step")
        pass_at_k = candidate.get("pass_at_k")
        gap = candidate.get("conditional_gap_mean")
        instances = candidate.get("instances")
        feasible = candidate.get("feasible_instances")
        gap_n = candidate.get("conditional_gap_n")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0 or step in seen_steps:
            raise ValueError(
                f"checkpoint selection receipt group {group_id} has invalid candidate steps"
            )
        try:
            pass_value = float(pass_at_k)
        except (TypeError, ValueError):
            pass_value = math.nan
        if not math.isfinite(pass_value) or not 0.0 <= pass_value <= 1.0:
            raise ValueError(f"checkpoint selection receipt group {group_id} has invalid pass@K")
        if (
            candidate.get("candidate_budget") != candidate_budget
            or isinstance(instances, bool)
            or not isinstance(instances, int)
            or instances <= 0
            or isinstance(feasible, bool)
            or not isinstance(feasible, int)
            or not 0 <= feasible <= instances
            or gap_n != feasible
            or not math.isclose(
                pass_value,
                feasible / instances,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"checkpoint selection receipt group {group_id} has invalid denominators"
            )
        gap_value: float | None
        if gap is None:
            gap_value = None
        else:
            try:
                gap_value = float(gap)
            except (TypeError, ValueError):
                gap_value = math.nan
            if not math.isfinite(gap_value) or gap_value < 0.0:
                raise ValueError(f"checkpoint selection receipt group {group_id} has invalid gap")
        if feasible == 0 and gap is not None:
            raise ValueError(
                f"checkpoint selection receipt group {group_id} has gap without feasible rows"
            )
        if feasible > 0 and gap_value is None:
            raise ValueError(
                f"checkpoint selection receipt group {group_id} has no gap for feasible rows"
            )
        beta, fixed_kl, exact_runtime, exact_post = _validate_exact_peft_kl_post_step_projection(
            candidate.get("grpo_beta"),
            candidate.get("fixed_kl_reference"),
            candidate.get("exact_peft_kl_runtime"),
            candidate.get("exact_peft_kl_post_step"),
            label=f"checkpoint selection receipt group {group_id} candidate {index}",
            expected_run_contract_sha256=candidate.get("run_contract_sha256"),
            expected_checkpoint_step=step,
        )
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(candidate.get(field, ""))) is None
            for field in _CHECKPOINT_SELECTION_FINGERPRINT_FIELDS
        ):
            raise ValueError(
                f"checkpoint selection receipt group {group_id} has invalid candidate provenance"
            )
        seen_steps.add(step)
        normalized.append(
            {
                **candidate,
                "checkpoint_step": step,
                "instances": instances,
                "feasible_instances": feasible,
                "conditional_gap_n": gap_n,
                "pass_at_k": pass_value,
                "conditional_gap_mean": gap_value,
                "grpo_beta": beta,
                "fixed_kl_reference": fixed_kl,
                "exact_peft_kl_runtime": exact_runtime,
                "exact_peft_kl_post_step": exact_post,
            }
        )

    best_pass = max(candidate["pass_at_k"] for candidate in normalized)
    if sorted(seen_steps) != expected_steps:
        raise ValueError(
            f"checkpoint selection receipt group {group_id} omits a scheduled checkpoint"
        )
    if len({candidate["run_contract_sha256"] for candidate in normalized}) != 1:
        raise ValueError(
            f"checkpoint selection receipt group {group_id} mixes training run contracts"
        )
    if len({candidate["grpo_beta"] for candidate in normalized}) != 1:
        raise ValueError(f"checkpoint selection receipt group {group_id} mixes GRPO beta")
    if (
        len({_canonical_json_bytes(candidate["fixed_kl_reference"]) for candidate in normalized})
        != 1
    ):
        raise ValueError(f"checkpoint selection receipt group {group_id} mixes fixed-KL identity")
    if (
        len({_canonical_json_bytes(candidate["exact_peft_kl_runtime"]) for candidate in normalized})
        != 1
    ):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} mixes exact PEFT KL runtime"
        )
    positive_post = [
        candidate["exact_peft_kl_post_step"]
        for candidate in normalized
        if candidate["exact_peft_kl_post_step"] is not None
    ]
    if positive_post and (
        len({candidate["run_invariant_sha256"] for candidate in positive_post}) != 1
        or len({candidate["source_reference_sha256"] for candidate in positive_post}) != 1
        or len({_canonical_json_bytes(candidate["root_final"]) for candidate in positive_post}) != 1
    ):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} mixes post-step run lineage"
        )
    if len({candidate["evaluation_fingerprint"] for candidate in normalized}) != len(normalized):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} duplicates an evaluation fingerprint"
        )
    if len({candidate["checkpoint_artifact_sha256"] for candidate in normalized}) != len(
        normalized
    ):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} duplicates a checkpoint artifact"
        )
    if len({candidate["checkpoint_fidelity_identity_sha256"] for candidate in normalized}) != len(
        normalized
    ):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} duplicates checkpoint fidelity"
        )
    threshold = best_pass - pass_band
    eligible = [
        candidate for candidate in normalized if candidate["pass_at_k"] >= threshold - 1e-15
    ]
    if any(candidate["conditional_gap_mean"] is None for candidate in eligible):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} has undefined eligible gap"
        )
    chosen = min(
        eligible,
        key=lambda candidate: (
            float(candidate["conditional_gap_mean"]),
            int(candidate["checkpoint_step"]),
        ),
    )
    eligible_steps = sorted(int(candidate["checkpoint_step"]) for candidate in eligible)
    try:
        stored_best = float(audit.get("best_pass_at_k"))
        stored_threshold = float(audit.get("eligibility_threshold"))
    except (TypeError, ValueError):
        stored_best = stored_threshold = math.nan
    if (
        not math.isclose(stored_best, best_pass, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(stored_threshold, threshold, rel_tol=0.0, abs_tol=1e-12)
        or audit.get("eligible_checkpoint_steps") != eligible_steps
    ):
        raise ValueError(
            f"checkpoint selection receipt group {group_id} audit summary is inconsistent"
        )
    for candidate in normalized:
        step = int(candidate["checkpoint_step"])
        if step not in eligible_steps:
            expected_disposition = "outside_pass_at_k_band"
        elif candidate is chosen:
            expected_disposition = "selected"
        else:
            expected_disposition = "eligible_but_higher_gap_or_later_exact_tie"
        if candidate.get("disposition") != expected_disposition:
            raise ValueError(
                f"checkpoint selection receipt group {group_id} candidate disposition is stale"
            )
    selected_projection = {
        field: selected.get(field)
        for field in (
            "checkpoint_step",
            "candidate_budget",
            "instances",
            "feasible_instances",
            "pass_at_k",
            "conditional_gap_n",
            "conditional_gap_mean",
            "grpo_beta",
            "fixed_kl_reference",
            "exact_peft_kl_runtime",
            "exact_peft_kl_post_step",
            "evaluation_fingerprint",
            "protocol_fingerprint",
            "development_evaluation_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_fidelity_identity_sha256",
            "checkpoint_fidelity_receipt_sha256",
            "checkpoint_fidelity_receipt_file_sha256",
            "run_contract_sha256",
            "data_manifest_sha256",
            "strict_verifier_manifest_sha256",
        )
    }
    chosen_projection = {
        field: chosen.get(field)
        for field in (
            "checkpoint_step",
            "candidate_budget",
            "instances",
            "feasible_instances",
            "pass_at_k",
            "conditional_gap_n",
            "conditional_gap_mean",
            "grpo_beta",
            "fixed_kl_reference",
            "exact_peft_kl_runtime",
            "exact_peft_kl_post_step",
            "evaluation_fingerprint",
            "protocol_fingerprint",
            "development_evaluation_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_fidelity_identity_sha256",
            "checkpoint_fidelity_receipt_sha256",
            "checkpoint_fidelity_receipt_file_sha256",
            "run_contract_sha256",
            "data_manifest_sha256",
            "strict_verifier_manifest_sha256",
        )
    }
    if selected_projection != chosen_projection:
        raise ValueError(
            f"checkpoint selection receipt group {group_id} selected row violates frozen rule"
        )
    return normalized


def _validate_embedded_checkpoint_selection_policy(
    *,
    descriptor: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[set[str], list[int], int, dict[str, dict[str, Any]]]:
    frozen = descriptor.get("frozen_policy")
    expected_policy_fields = {
        "campaign_id",
        "candidate_budget",
        "checkpoint_steps",
        "development_evaluation_by_task",
        "evidence_scope",
        "expected_methods",
        "expected_tasks",
        "forbidden_source_registry",
        "frozen_at_utc",
        "pass_at_k_absolute_band",
        "policy_sha256",
        "require_complete_checkpoint_grid",
        "schema_version",
        "selection_rule",
        "stage_id",
        "status",
        "train_seeds",
    }
    if not isinstance(frozen, dict) or set(frozen) != expected_policy_fields:
        raise ValueError("checkpoint selection receipt omits its frozen policy payload")
    unsigned = dict(frozen)
    stored_hash = str(unsigned.pop("policy_sha256", "") or "")
    if (
        re.fullmatch(r"[0-9a-f]{64}", stored_hash) is None
        or stored_hash != _experiment_fingerprint(unsigned)
        or stored_hash != descriptor.get("policy_sha256")
    ):
        raise ValueError("checkpoint selection receipt embedded policy hash is invalid")
    if (
        frozen.get("schema_version") != 1
        or frozen.get("status") != "frozen_before_development_results"
        or frozen.get("evidence_scope") != "development_only"
        or frozen.get("require_complete_checkpoint_grid") is not True
        or frozen.get("selection_rule") != descriptor.get("selection_rule")
        or frozen.get("pass_at_k_absolute_band") != descriptor.get("pass_at_k_absolute_band")
    ):
        raise ValueError("checkpoint selection receipt embedded policy is not frozen")
    lists: dict[str, list[Any]] = {}
    for field in ("expected_tasks", "expected_methods", "train_seeds", "checkpoint_steps"):
        value = frozen.get(field)
        if not isinstance(value, list) or not value or len(value) != len(set(value)):
            raise ValueError(f"checkpoint selection receipt embedded policy has invalid {field}")
        lists[field] = value
    try:
        seeds = [int(value) for value in lists["train_seeds"]]
        steps = [int(value) for value in lists["checkpoint_steps"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint selection receipt embedded grid is invalid") from exc
    if (
        any(isinstance(value, bool) or value < 0 for value in lists["train_seeds"])
        or any(isinstance(value, bool) or value <= 0 for value in lists["checkpoint_steps"])
        or steps != sorted(steps)
    ):
        raise ValueError("checkpoint selection receipt embedded grid values are invalid")
    candidate_budget = frozen.get("candidate_budget")
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("checkpoint selection receipt embedded candidate budget is invalid")
    registry = frozen.get("forbidden_source_registry")
    if not isinstance(registry, dict):
        raise ValueError(
            "checkpoint selection receipt embedded policy has no forbidden-source registry"
        )
    registry_identity = validate_forbidden_source_registry(
        registry,
        expected_tasks=[str(task) for task in lists["expected_tasks"]],
    )
    sources_by_task = registry_identity.get("sources_by_task_label")
    if not isinstance(sources_by_task, dict):
        raise ValueError("checkpoint selection receipt embedded registry has no source projections")
    development_contracts = frozen.get("development_evaluation_by_task")
    expected_development_fields = {
        "data_manifest_sha256",
        "evaluation_data_files",
        "forbidden_source_label",
        "forbidden_source_identity_sha256",
        "protocol_fingerprint",
        "strict_verifier_manifest_sha256",
        "split",
        "eval_start_index",
        "max_eval_samples_per_task",
        "eval_samples_per_task",
        "size_bucket",
        "min_instance_size",
        "max_instance_size",
        "prefer_subset_files",
        "expected_instances",
        "development_evaluation_sha256",
    }
    if not isinstance(development_contracts, dict) or set(development_contracts) != {
        str(task) for task in lists["expected_tasks"]
    }:
        raise ValueError(
            "checkpoint selection receipt embedded policy has no exact development-data grid"
        )
    validated_development: dict[str, dict[str, Any]] = {}
    for task in lists["expected_tasks"]:
        task_name = str(task)
        contract = development_contracts.get(task_name)
        if not isinstance(contract, dict) or set(contract) != expected_development_fields:
            raise ValueError(
                "checkpoint selection receipt embedded development contract is invalid"
            )
        unsigned_contract = dict(contract)
        contract_hash = str(unsigned_contract.pop("development_evaluation_sha256", "") or "")
        if (
            re.fullmatch(r"[0-9a-f]{64}", contract_hash) is None
            or contract_hash != _experiment_fingerprint(unsigned_contract)
            or re.fullmatch(r"[0-9a-f]{64}", str(contract.get("data_manifest_sha256", ""))) is None
            or contract.get("forbidden_source_label") != "historical-eval"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(contract.get("forbidden_source_identity_sha256", "")),
            )
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(contract.get("protocol_fingerprint", ""))) is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(contract.get("strict_verifier_manifest_sha256", "")),
            )
            is None
            or contract.get("split") != "eval"
            or contract.get("eval_start_index") != 0
            or isinstance(contract.get("max_eval_samples_per_task"), bool)
            or not isinstance(contract.get("max_eval_samples_per_task"), int)
            or int(contract["max_eval_samples_per_task"]) <= 0
            or contract.get("eval_samples_per_task") != {}
            or contract.get("size_bucket") != "all"
            or contract.get("min_instance_size") != 0
            or contract.get("max_instance_size") != 0
            or contract.get("prefer_subset_files") is not False
            or contract.get("expected_instances") != contract.get("max_eval_samples_per_task")
        ):
            raise ValueError(
                "checkpoint selection receipt embedded development contract is not frozen"
            )
        evaluation_files = contract.get("evaluation_data_files")
        expected_file_keys = {"task", "split", "path", "size", "sha256"}
        if (
            not isinstance(evaluation_files, list)
            or not evaluation_files
            or any(
                not isinstance(item, dict)
                or set(item) != expected_file_keys
                or item.get("task") != task_name
                or item.get("split") != contract["split"]
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or isinstance(item.get("size"), bool)
                or not isinstance(item.get("size"), int)
                or int(item["size"]) <= 0
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
                for item in evaluation_files
            )
        ):
            raise ValueError("checkpoint selection receipt embedded development files are invalid")
        paths = [str(item["path"]) for item in evaluation_files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError(
                "checkpoint selection receipt embedded development files are not sorted/unique"
            )
        if contract["data_manifest_sha256"] != _experiment_fingerprint(evaluation_files):
            raise ValueError(
                "checkpoint selection receipt embedded development manifest is invalid"
            )
        task_sources = sources_by_task.get(task_name)
        historical = task_sources.get("historical-eval") if isinstance(task_sources, dict) else None
        if not isinstance(historical, dict) or contract[
            "forbidden_source_identity_sha256"
        ] != historical.get("source_identity_sha256"):
            raise ValueError(
                "checkpoint selection receipt development data is not registered historical-eval"
            )
        registered_files = historical.get("files")
        registered_projection = (
            [
                {
                    "task": task_name,
                    "split": contract["split"],
                    "path": item.get("relative_path"),
                    "size": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in registered_files
                if isinstance(item, dict)
            ]
            if isinstance(registered_files, list)
            else []
        )
        if evaluation_files != registered_projection:
            raise ValueError(
                "checkpoint selection receipt development bytes differ from the "
                "registered historical-eval source"
            )
        validated_development[task_name] = contract
    expected_groups = {
        f"{task}/{method}/seed{seed}"
        for task in lists["expected_tasks"]
        for method in lists["expected_methods"]
        for seed in seeds
    }
    expected_records = len(expected_groups) * len(steps)
    if evidence.get("record_count") != expected_records:
        raise ValueError(
            "checkpoint selection receipt evidence count does not cover the frozen grid"
        )
    return expected_groups, steps, candidate_budget, validated_development


def _validate_confirmation_checkpoint_selection(
    *,
    receipt: dict[str, Any],
    training_run_contract: dict[str, Any],
    training_run_contract_sha256: str,
    checkpoint_path: str,
    checkpoint_artifact: dict[str, Any],
    checkpoint_fidelity: dict[str, Any],
    exact_peft_kl_runtime: dict[str, Any] | None,
    exact_peft_kl_post_step: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind the resolved checkpoint to its prespecified development decision."""

    task = str(training_run_contract["task"])
    method = str(training_run_contract["method"])
    train_seed = int(training_run_contract["seeds"]["train"])
    group_id = f"{task}/{method}/seed{train_seed}"
    group = receipt.get("selections", {}).get(group_id)
    if not isinstance(group, dict) or group.get("status") != "PASS":
        raise RuntimeError(f"Checkpoint selection receipt has no passing decision for {group_id}")
    selected = group.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"Checkpoint selection receipt has no selected row for {group_id}")
    expected_identity = {
        "task": task,
        "method": method,
        "train_seed": train_seed,
    }
    if any(selected.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(
            "Checkpoint selection identity disagrees with the immutable training contract"
        )
    if selected.get("run_contract_sha256") != training_run_contract_sha256:
        raise RuntimeError(
            "Checkpoint selection receipt references a different training run contract"
        )
    try:
        selected_step = int(selected["checkpoint_step"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Checkpoint selection receipt has an invalid checkpoint step") from None
    checkpoint_basename = os.path.basename(os.path.realpath(checkpoint_path))
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_basename)
    if match is None or int(match.group(1)) != selected_step:
        raise RuntimeError(
            "Resolved model path is not the checkpoint selected before confirmation: "
            f"selected=checkpoint-{selected_step}, resolved={checkpoint_basename}"
        )
    artifact_sha256 = str(
        checkpoint_artifact.get("manifest_sha256") or checkpoint_artifact.get("sha256") or ""
    )
    if selected.get("checkpoint_artifact_sha256") != artifact_sha256:
        raise RuntimeError(
            "Resolved checkpoint bytes do not match the selected checkpoint artifact"
        )
    try:
        live_fidelity = validate_publication_fidelity_identity(checkpoint_fidelity)
    except ValueError as exc:
        raise RuntimeError(f"Resolved checkpoint fidelity is invalid: {exc}") from exc
    artifact_files = checkpoint_artifact.get("files")
    files_by_path = {
        str(item.get("path")): item
        for item in artifact_files or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    expected_artifact_files = {
        CHECKPOINT_FIDELITY_FILENAME: live_fidelity["receipt_file_sha256"],
        "adapter_config.json": live_fidelity["adapter_config_sha256"],
        "adapter_model.safetensors": live_fidelity["adapter_weights_sha256"],
    }
    if not isinstance(artifact_files, list) or any(
        not isinstance(files_by_path.get(path), dict) or files_by_path[path].get("sha256") != digest
        for path, digest in expected_artifact_files.items()
    ):
        raise RuntimeError(
            "Resolved checkpoint artifact does not close over its live fidelity receipt"
        )
    fidelity_projection = {
        "checkpoint_fidelity_identity_sha256": live_fidelity["identity_sha256"],
        "checkpoint_fidelity_receipt_sha256": live_fidelity["receipt_sha256"],
        "checkpoint_fidelity_receipt_file_sha256": live_fidelity["receipt_file_sha256"],
    }
    fidelity_mismatches = {
        field: {"selected": selected.get(field), "live": observed}
        for field, observed in fidelity_projection.items()
        if selected.get(field) != observed
    }
    if fidelity_mismatches:
        raise RuntimeError(
            "Resolved checkpoint fidelity does not match the selected checkpoint: "
            f"{fidelity_mismatches}"
        )
    try:
        contract_beta, contract_fixed_kl, live_exact_runtime, live_exact_post_step = (
            _validate_exact_peft_kl_post_step_projection(
                training_run_contract.get("training_config", {}).get("beta"),
                training_run_contract.get("fixed_kl_reference"),
                exact_peft_kl_runtime,
                exact_peft_kl_post_step,
                label="Training run contract",
                expected_run_contract_sha256=training_run_contract_sha256,
                expected_checkpoint_step=selected_step,
            )
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if live_exact_runtime is not None:
        expected_artifact_files[EXACT_PEFT_KL_REFERENCE_FILENAME] = live_exact_runtime[
            "receipt_file_sha256"
        ]
        runtime_artifact = files_by_path.get(EXACT_PEFT_KL_REFERENCE_FILENAME)
        if (
            not isinstance(runtime_artifact, dict)
            or runtime_artifact.get("sha256") != live_exact_runtime["receipt_file_sha256"]
        ):
            raise RuntimeError(
                "Resolved checkpoint artifact does not close over its exact PEFT KL runtime"
            )
        expected_artifact_files[EXACT_PEFT_KL_STEP1_FILENAME] = live_exact_post_step[
            "selected_checkpoint_step1"
        ]["receipt_file_sha256"]
        expected_artifact_files[EXACT_PEFT_KL_POST_STEP_FILENAME] = live_exact_post_step[
            "selected_checkpoint_post_step"
        ]["receipt_file_sha256"]
        for name in (EXACT_PEFT_KL_STEP1_FILENAME, EXACT_PEFT_KL_POST_STEP_FILENAME):
            if (
                not isinstance(files_by_path.get(name), dict)
                or files_by_path[name].get("sha256") != expected_artifact_files[name]
            ):
                raise RuntimeError(
                    "Resolved checkpoint artifact does not close over post-step exact KL"
                )
    if (
        selected.get("grpo_beta") != contract_beta
        or selected.get("fixed_kl_reference") != contract_fixed_kl
        or selected.get("exact_peft_kl_runtime") != live_exact_runtime
        or selected.get("exact_peft_kl_post_step") != live_exact_post_step
    ):
        raise RuntimeError(
            "Checkpoint selection KL projection disagrees with the live training runtime"
        )
    return {
        "group_id": group_id,
        "checkpoint_step": selected_step,
        "checkpoint_artifact_sha256": artifact_sha256,
        **fidelity_projection,
        "grpo_beta": contract_beta,
        "fixed_kl_reference": contract_fixed_kl,
        "exact_peft_kl_runtime": live_exact_runtime,
        "exact_peft_kl_post_step": live_exact_post_step,
        "run_contract_sha256": training_run_contract_sha256,
        "checkpoint_selection_receipt_sha256": receipt.get("receipt_sha256"),
        "selection_evaluation_fingerprint": selected.get("evaluation_fingerprint"),
    }


def _validate_confirmation_campaign_binding(
    *,
    manifest: dict[str, Any],
    manifest_identity: dict[str, object],
    benchmark_identity: dict[str, object],
    selection_receipt_identity: dict[str, object],
    selection_receipt_payload: dict[str, Any],
    training_run_contract: dict[str, Any],
    training_run_contract_sha256: str,
    checkpoint_selection_binding: dict[str, Any],
    decode_seed: int,
) -> dict[str, Any]:
    """Bind one model evaluation to its predeclared confirmation grid cell."""

    task = str(training_run_contract["task"])
    method = str(training_run_contract["method"])
    train_seed = int(training_run_contract["seeds"]["train"])
    if manifest.get("decode_seed") != decode_seed:
        raise RuntimeError("Confirmation decode seed disagrees with the frozen campaign")
    if manifest.get("benchmark_manifest_sha256") != benchmark_identity.get("manifest_sha256"):
        raise RuntimeError("Confirmation campaign is bound to another benchmark")
    training_design = manifest.get("training_design")
    fixed_null_by_task = (
        training_design.get("fixed_null_receipt_sha256_by_task")
        if isinstance(training_design, dict)
        else None
    )
    fixed_null_receipt = training_run_contract.get("fixed_null_receipt")
    if method == "set_null":
        observed_fixed_null_sha256 = (
            fixed_null_receipt.get("receipt_sha256")
            if isinstance(fixed_null_receipt, dict)
            else None
        )
        if (
            not isinstance(fixed_null_by_task, dict)
            or fixed_null_by_task.get(task) != observed_fixed_null_sha256
        ):
            raise RuntimeError(
                "SET-NULL run receipt disagrees with the frozen campaign preregistration"
            )
    elif isinstance(fixed_null_by_task, dict) and fixed_null_by_task:
        raise RuntimeError("Non-SET-NULL campaign unexpectedly preregisters fixed-null receipts")
    common_receipt = manifest.get("checkpoint_selection_receipt")
    common_identity = common_receipt.get("identity") if isinstance(common_receipt, dict) else None
    common_payload = common_receipt.get("payload") if isinstance(common_receipt, dict) else None
    if common_identity != selection_receipt_identity or common_payload != selection_receipt_payload:
        raise RuntimeError(
            "Confirmation campaign is not bound to the supplied checkpoint-selection receipt"
        )
    entries = manifest.get("runs")
    matches = (
        [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("method") == method
            and entry.get("training_seed") == train_seed
            and entry.get("task") == task
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Confirmation campaign has no unique run cell for {method}/seed{train_seed}/{task}"
        )
    entry = matches[0]
    try:
        contract_beta, contract_fixed_kl, contract_exact_runtime, contract_exact_post_step = (
            _validate_exact_peft_kl_post_step_projection(
                training_run_contract.get("training_config", {}).get("beta"),
                training_run_contract.get("fixed_kl_reference"),
                checkpoint_selection_binding.get("exact_peft_kl_runtime"),
                checkpoint_selection_binding.get("exact_peft_kl_post_step"),
                label="Training run contract",
                expected_run_contract_sha256=training_run_contract_sha256,
                expected_checkpoint_step=checkpoint_selection_binding.get("checkpoint_step"),
            )
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if (
        checkpoint_selection_binding.get("grpo_beta") != contract_beta
        or checkpoint_selection_binding.get("fixed_kl_reference") != contract_fixed_kl
        or checkpoint_selection_binding.get("exact_peft_kl_runtime") != contract_exact_runtime
        or checkpoint_selection_binding.get("exact_peft_kl_post_step") != contract_exact_post_step
    ):
        raise RuntimeError(
            "Confirmation checkpoint binding KL projection disagrees with its contract/runtime"
        )
    expected = {
        "training_run_contract_sha256": training_run_contract_sha256,
        "checkpoint_step": checkpoint_selection_binding.get("checkpoint_step"),
        "checkpoint_artifact_sha256": checkpoint_selection_binding.get(
            "checkpoint_artifact_sha256"
        ),
        "checkpoint_fidelity_identity_sha256": checkpoint_selection_binding.get(
            "checkpoint_fidelity_identity_sha256"
        ),
        "checkpoint_fidelity_receipt_sha256": checkpoint_selection_binding.get(
            "checkpoint_fidelity_receipt_sha256"
        ),
        "checkpoint_fidelity_receipt_file_sha256": checkpoint_selection_binding.get(
            "checkpoint_fidelity_receipt_file_sha256"
        ),
        "grpo_beta": contract_beta,
        "fixed_kl_reference": contract_fixed_kl,
        "exact_peft_kl_runtime": contract_exact_runtime,
        "exact_peft_kl_post_step": contract_exact_post_step,
        "checkpoint_selection_receipt_sha256": selection_receipt_identity.get("receipt_sha256"),
    }
    mismatches = {
        field: {"campaign": entry.get(field), "observed": value}
        for field, value in expected.items()
        if entry.get(field) != value
    }
    if mismatches:
        raise RuntimeError(
            "Confirmation campaign run cell disagrees with resolved immutable artifacts: "
            f"{mismatches}"
        )
    attempt_id = str(entry["confirmation_attempt_id"])
    if attempt_id != _confirmation_attempt_id(manifest, entry):
        raise RuntimeError("Confirmation campaign run cell has an invalid attempt identity")
    return {
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": manifest_identity["campaign_manifest_sha256"],
        "method": method,
        "training_seed": train_seed,
        "task": task,
        "confirmation_attempt_id": attempt_id,
        **expected,
        "benchmark_manifest_sha256": manifest["benchmark_manifest_sha256"],
    }


def _claim_confirmation_attempt(
    *,
    attempt_ledger_root: str,
    campaign_binding: dict[str, Any],
    experiment_fingerprint: str,
    results_dir: str,
    required_resume_artifacts: tuple[str, str],
):
    """Create and exclusively lock the campaign's one operational attempt receipt."""

    campaign_sha256 = str(campaign_binding["campaign_manifest_sha256"])
    attempt_id = str(campaign_binding["confirmation_attempt_id"])
    ledger_dir = os.path.join(
        os.path.realpath(attempt_ledger_root),
        campaign_sha256,
    )
    os.makedirs(ledger_dir, mode=0o750, exist_ok=True)
    claim_path = os.path.join(ledger_dir, f"{attempt_id}.json")
    core = {
        "schema_version": 2,
        "protocol": _CONFIRMATION_ATTEMPT_PROTOCOL,
        "campaign_manifest_sha256": campaign_sha256,
        "campaign_id": campaign_binding["campaign_id"],
        "confirmation_attempt_id": attempt_id,
        "experiment_fingerprint": experiment_fingerprint,
        "results_dir": os.path.realpath(results_dir),
    }
    payload = {
        **core,
        "claim_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    temporary_path = None
    claim_created = False
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{attempt_id}.", suffix=".tmp", dir=ledger_dir
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o440)
        try:
            os.link(temporary_path, claim_path)
            claim_created = True
        except FileExistsError:
            existing, _identity = _load_json_snapshot(
                claim_path, label="confirmation attempt claim"
            )
            existing_unsigned = dict(existing)
            existing_hash = existing_unsigned.pop("claim_sha256", None)
            if (
                existing_unsigned != core
                or existing_hash
                != hashlib.sha256(_canonical_json_bytes(existing_unsigned)).hexdigest()
            ):
                raise RuntimeError(
                    "Confirmation attempt was already claimed for a different output or "
                    "experiment fingerprint"
                )
        finally:
            os.unlink(temporary_path)
            temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    claim_handle = open(claim_path, "rb")
    try:
        fcntl.flock(claim_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        claim_handle.close()
        raise RuntimeError(
            "Confirmation attempt is already active in another evaluator process"
        ) from exc
    marker_path = os.path.join(os.path.realpath(results_dir), ".confirmation_attempt.json")
    if claim_created:
        os.makedirs(os.path.realpath(results_dir), mode=0o750, exist_ok=True)
        if os.path.lexists(marker_path):
            claim_handle.close()
            raise RuntimeError("Fresh confirmation attempt found a pre-existing output marker")
        marker_temporary = None
        try:
            marker_descriptor, marker_temporary = tempfile.mkstemp(
                prefix=".confirmation_attempt.",
                suffix=".tmp",
                dir=os.path.realpath(results_dir),
            )
            with os.fdopen(marker_descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(marker_temporary, 0o440)
            os.link(marker_temporary, marker_path)
        finally:
            if marker_temporary is not None and os.path.exists(marker_temporary):
                os.unlink(marker_temporary)
    else:
        marker, _marker_identity = _load_json_snapshot(
            marker_path, label="confirmation output attempt marker"
        )
        if marker != payload:
            claim_handle.close()
            raise RuntimeError(
                "Confirmation output marker disagrees with its durable attempt claim"
            )
        missing = [path for path in required_resume_artifacts if not os.path.isfile(path)]
        if missing:
            claim_handle.close()
            raise RuntimeError(
                "Existing confirmation claim cannot restart after output artifacts were "
                f"removed or never committed: {missing}"
            )
    return claim_path, claim_handle


def _validate_confirmation_dataset(
    *,
    manifest_path: str,
    tasks: list[str],
    data_identity: dict[str, object],
    manifest_payload: dict[str, Any] | None = None,
) -> None:
    """Fail before generation unless task files and verifier equal the frozen manifest."""

    if manifest_payload is None:
        manifest, _identity = _load_benchmark_manifest_snapshot(manifest_path)
    else:
        manifest = manifest_payload
    test_sets = manifest.get("test_sets")
    if not isinstance(test_sets, dict):
        raise ValueError("benchmark manifest test_sets is invalid")
    observed_files = data_identity.get("files")
    if not isinstance(observed_files, list):
        raise ValueError("evaluation data identity has no file manifest")
    for task in tasks:
        frozen = test_sets.get(task)
        if not isinstance(frozen, dict):
            raise ValueError(f"benchmark manifest has no frozen test set for task={task}")
        frozen_files = frozen.get("files")
        if not isinstance(frozen_files, list) or not frozen_files:
            raise ValueError(f"benchmark manifest has no frozen files for task={task}")
        expected = sorted(
            (str(item.get("sha256", "")), int(item.get("size_bytes", -1)))
            for item in frozen_files
            if isinstance(item, dict)
        )
        observed = sorted(
            (str(item.get("sha256", "")), int(item.get("size", -1)))
            for item in observed_files
            if isinstance(item, dict) and str(item.get("task", "")) == task
        )
        if not expected or observed != expected:
            raise ValueError(
                f"evaluation data files do not match frozen benchmark for task={task}: "
                f"expected={expected}, observed={observed}"
            )
    frozen_verifier = manifest.get("strict_verifier")
    active_verifier = _strict_verifier_identity()
    if frozen_verifier != active_verifier:
        raise ValueError("active strict verifier does not match the frozen benchmark manifest")


def _is_inference_artifact(relative_path: str) -> bool:
    name = os.path.basename(relative_path)
    if name in _INFERENCE_ARTIFACT_EXACT_NAMES:
        return True
    if name.endswith(_INFERENCE_ARTIFACT_SUFFIXES):
        return True
    return name.endswith(".bin") and name.startswith(_INFERENCE_WEIGHT_PREFIXES)


def _is_tokenizer_artifact(relative_path: str) -> bool:
    name = os.path.basename(relative_path)
    return name in _TOKENIZER_ARTIFACT_EXACT_NAMES or name.endswith(_TOKENIZER_ARTIFACT_SUFFIXES)


def _directory_content_identity(directory: str, *, inference_artifacts_only: bool) -> dict:
    """Return a relocatable content manifest for a local model/tokenizer directory."""

    root = os.path.realpath(directory)
    files: list[dict[str, Any]] = []
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name
            not in {
                ".cache",
                ".git",
                "__pycache__",
                "checkpoints",
                "output",
                "outputs",
                "runs",
                "wandb",
            }
        )
        for name in sorted(file_names):
            file_path = os.path.join(current_root, name)
            relative_path = os.path.relpath(file_path, root).replace(os.sep, "/")
            if inference_artifacts_only and not _is_inference_artifact(relative_path):
                continue
            sha256, size = _sha256_file(file_path)
            files.append(
                {
                    "path": relative_path,
                    "size": size,
                    "sha256": sha256,
                }
            )
    if not files:
        raise RuntimeError(f"No fingerprintable files found under: {directory}")
    manifest_sha256 = hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
    return {
        "kind": "local_directory",
        "basename": os.path.basename(root),
        "manifest_sha256": manifest_sha256,
        "files": files,
    }


def _reference_content_identity(reference: str, *, inference_artifacts_only: bool) -> dict:
    normalized_reference = str(reference or "").strip()
    if normalized_reference and os.path.isdir(normalized_reference):
        return _directory_content_identity(
            normalized_reference,
            inference_artifacts_only=inference_artifacts_only,
        )
    if normalized_reference and os.path.isfile(normalized_reference):
        sha256, size = _sha256_file(normalized_reference)
        return {
            "kind": "local_file",
            "basename": os.path.basename(normalized_reference),
            "size": size,
            "sha256": sha256,
        }
    return {
        "kind": "external_reference",
        "reference": normalized_reference,
    }


def _tokenizer_content_identity(reference: str) -> dict:
    """Fingerprint tokenizer bytes without accidentally including model weights."""

    normalized_reference = str(reference or "").strip()
    if normalized_reference and os.path.isdir(normalized_reference):
        root = os.path.realpath(normalized_reference)
        files: list[dict[str, Any]] = []
        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = sorted(
                name for name in dir_names if name not in {".cache", ".git", "__pycache__"}
            )
            for name in sorted(file_names):
                file_path = os.path.join(current_root, name)
                relative_path = os.path.relpath(file_path, root).replace(os.sep, "/")
                if not _is_tokenizer_artifact(relative_path):
                    continue
                sha256, size = _sha256_file(file_path)
                files.append({"path": relative_path, "size": size, "sha256": sha256})
        if not files:
            raise RuntimeError(
                f"No fingerprintable tokenizer files found under: {normalized_reference}"
            )
        return {
            "kind": "local_directory",
            "basename": os.path.basename(root),
            "manifest_sha256": _experiment_fingerprint(files),
            "files": files,
        }
    if normalized_reference and os.path.isfile(normalized_reference):
        sha256, size = _sha256_file(normalized_reference)
        return {
            "kind": "local_file",
            "basename": os.path.basename(normalized_reference),
            "size": size,
            "sha256": sha256,
        }
    return {"kind": "external_reference", "reference": normalized_reference}


def _raw_base_model_reference(model_path: str) -> str:
    for file_name, field_name in (
        ("adapter_config.json", "base_model_name_or_path"),
        ("config.json", "_name_or_path"),
    ):
        payload = _json_load(os.path.join(model_path, file_name)) or {}
        candidate = str(payload.get(field_name, "") or "").strip()
        if candidate:
            return candidate
    return str(model_path)


def _loaded_model_revision(model) -> str | None:
    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ]
    for config in candidates:
        revision = str(getattr(config, "_commit_hash", "") or "").strip()
        if revision:
            return revision
    return None


def _loaded_model_config_identity(model) -> dict:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    payload = to_dict() if callable(to_dict) else {"class": type(config).__name__}
    return {
        "class": type(config).__name__,
        "sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def _loaded_tokenizer_identity(tokenizer) -> dict:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "class": type(tokenizer).__name__,
        "vocab_sha256": hashlib.sha256(_canonical_json_bytes(vocab)).hexdigest(),
        "special_tokens_sha256": hashlib.sha256(
            _canonical_json_bytes(special_tokens_map)
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "commit_hash": str(init_kwargs.get("_commit_hash", "") or "") or None,
    }


def _dataset_content_identity(
    *,
    data_root: str,
    task_json_files: dict[str, list[str]],
    split: str,
) -> dict:
    files: list[dict[str, Any]] = []
    for task in sorted(task_json_files):
        for file_path in task_json_files[task]:
            sha256, size = _sha256_file(file_path)
            files.append(
                {
                    "task": task,
                    "split": str(split),
                    "path": os.path.relpath(file_path, data_root).replace(os.sep, "/"),
                    "size": size,
                    "sha256": sha256,
                }
            )
    return {
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
        "files": files,
    }


def _source_code_identity() -> dict:
    relative_paths = (
        "entrypoints/evaluate.py",
        "data/loaders.py",
        "data/prompts.py",
        "models/loading.py",
        "rewards/co_tasks.py",
        "runtime.py",
    )
    files: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        source_path = os.path.join(_RUN_EVAL_PARENT, *relative_path.split("/"))
        sha256, size = _sha256_file(source_path)
        files.append({"path": relative_path, "size": size, "sha256": sha256})
    return {
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
        "files": files,
    }


def _strict_verifier_identity() -> dict:
    implementations: list[dict[str, Any]] = []
    for relative_path in ("rewards/co_tasks.py", "entrypoints/evaluate.py"):
        verifier_path = os.path.join(_RUN_EVAL_PARENT, *relative_path.split("/"))
        sha256, size = _sha256_file(verifier_path)
        implementations.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": sha256,
            }
        )
    return {
        "name": "clasp_strict_shared_verifier",
        "schema_version": STRICT_VERIFIER_SCHEMA_VERSION,
        "implementation_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(implementations)
        ).hexdigest(),
        "implementations": implementations,
        "tasks": sorted(TASK_EVALUATORS),
    }


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for distribution_name in ("datasets", "peft", "safetensors", "torch", "transformers"):
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution_name] = "not-installed"
    try:
        import torch  # type: ignore[import-not-found]

        versions["cuda_runtime"] = str(torch.version.cuda or "none")
        versions["cudnn"] = int(torch.backends.cudnn.version() or 0)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu_name"] = str(torch.cuda.get_device_name(0))
            versions["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        versions["hardware_probe_error"] = type(exc).__name__
    return versions


def _experiment_fingerprint(material: dict) -> str:
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _protocol_fingerprint_material(material: dict) -> dict:
    """Remove model/checkpoint-selection identity while retaining the eval protocol."""

    protocol = {
        key: value
        for key, value in material.items()
        if key
        not in {
            "model",
            "training_run_contract",
            "training_run_contract_sha256",
        }
    }
    model = material.get("model")
    if isinstance(model, dict):
        protocol["tokenizer"] = {
            "artifact": model.get("tokenizer_artifact"),
            "loaded": model.get("loaded_tokenizer"),
        }
    selection = protocol.get("selection")
    if isinstance(selection, dict):
        shared_selection = dict(selection)
        shared_selection.pop("checkpoint_selection_receipt", None)
        shared_selection.pop("checkpoint_selection_binding", None)
        shared_selection.pop("campaign_run_binding", None)
        protocol["selection"] = shared_selection
    return protocol


def _self_hash_payload(payload: dict, field: str) -> dict:
    """Return a copy carrying a canonical SHA-256 over every other field."""

    sealed = dict(payload)
    sealed.pop(field, None)
    sealed[field] = _experiment_fingerprint(sealed)
    return sealed


def _validate_self_hash(payload: dict, field: str, *, label: str) -> str:
    """Fail closed when a content-addressed JSON object has been altered."""

    stored = str(payload.get(field, "") or "")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    observed = _experiment_fingerprint(unsigned)
    if stored != observed:
        raise RuntimeError(
            f"{label} self-hash mismatch: stored={stored or '<missing>'}, observed={observed}"
        )
    return stored


def _details_artifact_identity(path: str, *, record_count: int) -> dict[str, Any]:
    sha256, size_bytes = _sha256_file(path)
    return {
        "basename": os.path.basename(path),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "record_count": int(record_count),
    }


def _validate_training_tree_receipt(receipt: Any, *, label: str) -> None:
    if not isinstance(receipt, dict):
        raise RuntimeError(f"Training run contract has no {label} fingerprint object")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Training run contract {label} fingerprint has no files")
    aggregate = hashlib.sha256()
    total_size = 0
    previous_path = ""
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise RuntimeError(f"Training run contract {label} file {index} is invalid")
        relative = str(item.get("path", "") or "")
        digest = str(item.get("sha256", "") or "")
        try:
            size = int(item.get("size_bytes", -1))
        except (TypeError, ValueError):
            size = -1
        if (
            not relative
            or relative <= previous_path
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or size < 0
        ):
            raise RuntimeError(
                f"Training run contract {label} file manifest is malformed or unsorted"
            )
        previous_path = relative
        total_size += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if (
        receipt.get("file_count") != len(files)
        or receipt.get("size_bytes") != total_size
        or receipt.get("sha256") != aggregate.hexdigest()
        or not str(receipt.get("canonical_path", "") or "")
    ):
        raise RuntimeError(f"Training run contract {label} fingerprint is inconsistent")


def _validate_training_run_contract(contract: dict[str, Any]) -> None:
    """Require the complete immutable schema emitted by prepare_sota_run.py."""

    expected_contract_fields = {
        "schema_version",
        "stage",
        "method",
        "task",
        "seeds",
        "fixed_horizon",
        "data_lineage",
        "model_initialization_registry",
        "fixed_kl_reference",
        "fixed_null_receipt",
        "fingerprints",
        "generation",
        "training_config",
        "tracked_environment",
    }
    if set(contract) != expected_contract_fields:
        raise RuntimeError("Training run contract does not use the full official schema")
    if contract.get("schema_version") != 2 or contract.get("stage") != "grpo":
        raise RuntimeError("Training run contract has an unsupported schema or stage")
    method = str(contract.get("method", "") or "")
    task = str(contract.get("task", "") or "")
    if method in _LEGACY_CONFIRMATION_METHOD_ALIASES:
        raise RuntimeError("Training run contract uses a prohibited legacy method alias")
    if method not in _CONFIRMATION_METHODS:
        raise RuntimeError("Training run contract has an unsupported method")
    if task not in TASK_DIRECTIONS:
        raise RuntimeError("Training run contract has an unsupported task")
    seeds = contract.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"train", "data", "sft"}:
        raise RuntimeError("Training run contract must bind train/data/SFT seeds")
    if any(
        isinstance(seeds[name], bool) or not isinstance(seeds[name], int) or int(seeds[name]) < 0
        for name in ("train", "data", "sft")
    ) or int(seeds["train"]) != int(seeds["data"]):
        raise RuntimeError("Training run contract seeds are invalid or unmatched")
    horizon = contract.get("fixed_horizon")
    if (
        not isinstance(horizon, dict)
        or set(horizon) != {"max_steps", "world_size"}
        or isinstance(horizon.get("max_steps"), bool)
        or not isinstance(horizon.get("max_steps"), int)
        or int(horizon["max_steps"]) <= 0
        or isinstance(horizon.get("world_size"), bool)
        or not isinstance(horizon.get("world_size"), int)
        or int(horizon["world_size"]) <= 0
    ):
        raise RuntimeError("Training run contract fixed horizon is invalid")
    fingerprints = contract.get("fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "initial_model",
        "base_model",
        "data",
        "source",
    }:
        raise RuntimeError("Training run contract has no artifact fingerprints")
    for name in ("initial_model", "base_model", "data", "source"):
        _validate_training_tree_receipt(fingerprints.get(name), label=name)
    if not isinstance(contract.get("generation"), dict) or not contract["generation"]:
        raise RuntimeError("Training run contract has no generation configuration")
    if not isinstance(contract.get("training_config"), dict) or not contract["training_config"]:
        raise RuntimeError("Training run contract has no training configuration")
    if (
        not isinstance(contract.get("tracked_environment"), dict)
        or not contract["tracked_environment"]
    ):
        raise RuntimeError("Training run contract has no tracked environment")
    if not isinstance(contract.get("data_lineage"), dict) or not contract["data_lineage"]:
        raise RuntimeError("Training run contract has no registered training-data lineage")
    initialization = contract.get("model_initialization_registry")
    if not isinstance(initialization, dict) or not initialization:
        raise RuntimeError("Training run contract has no model-initialization registry")

    training_config = contract["training_config"]
    tracked_environment = contract["tracked_environment"]
    if training_config.get("checkpoint_fidelity") != "publication_exact_required":
        raise RuntimeError("Training run contract does not require publication checkpoint fidelity")
    if tracked_environment.get("CLASP_PUBLICATION_FIDELITY") != "1":
        raise RuntimeError("Training run contract did not enable publication checkpoint fidelity")
    fixed_null_receipt = contract.get("fixed_null_receipt")
    fixed_null_path = tracked_environment.get("CLASP_FIXED_NULL_RECEIPT_PATH")
    if method == "set_null":
        expected_fixed_null_fields = {
            "schema_version",
            "protocol",
            "status",
            "receipt_sha256",
            "receipt_file_sha256",
            "null_candidate_sha256",
            "evaluator_identity_sha256",
            "record_count",
            "tasks",
        }
        if (
            not isinstance(fixed_null_receipt, dict)
            or set(fixed_null_receipt) != expected_fixed_null_fields
            or fixed_null_receipt.get("schema_version") != 1
            or fixed_null_receipt.get("protocol") != "clasp-fixed-null-construction-v1"
            or fixed_null_receipt.get("status") != "frozen_before_training"
            or fixed_null_receipt.get("tasks") != [task]
            or isinstance(fixed_null_receipt.get("record_count"), bool)
            or not isinstance(fixed_null_receipt.get("record_count"), int)
            or int(fixed_null_receipt["record_count"]) <= 0
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(fixed_null_receipt.get(field, ""))) is None
                for field in (
                    "receipt_sha256",
                    "receipt_file_sha256",
                    "null_candidate_sha256",
                    "evaluator_identity_sha256",
                )
            )
            or not isinstance(fixed_null_path, str)
            or not os.path.isabs(fixed_null_path)
            or os.path.realpath(fixed_null_path) != fixed_null_path
        ):
            raise RuntimeError("set_null training run contract has no valid fixed-null binding")
    elif fixed_null_receipt is not None or fixed_null_path not in {None, ""}:
        raise RuntimeError("Non-set_null training run contract carries a fixed-null binding")
    beta = training_config.get("beta")
    try:
        normalized_beta, fixed_kl = _validate_fixed_kl_projection(
            beta,
            contract.get("fixed_kl_reference"),
            label="Training run contract",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    kl_model_path = tracked_environment.get("CLASP_KL_REFERENCE_MODEL_PATH")
    kl_receipt_path = tracked_environment.get("CLASP_KL_REFERENCE_RECEIPT")
    if normalized_beta == 0.0:
        if kl_model_path is not None or kl_receipt_path is not None:
            raise RuntimeError(
                "Training run contract beta=0 must not carry fixed-KL reference paths"
            )
        return
    if (
        not isinstance(kl_model_path, str)
        or not os.path.isabs(kl_model_path)
        or os.path.realpath(kl_model_path) != kl_model_path
        or not isinstance(kl_receipt_path, str)
        or not os.path.isabs(kl_receipt_path)
        or os.path.realpath(kl_receipt_path) != kl_receipt_path
    ):
        raise RuntimeError(
            "Training run contract beta>0 has no canonical fixed-KL model/receipt paths"
        )
    specialists = initialization.get("specialist_sft_by_task")
    specialist = specialists.get(task) if isinstance(specialists, dict) else None
    initial_model = fingerprints.get("initial_model")
    expected_model_path = (
        initial_model.get("canonical_path") if isinstance(initial_model, dict) else None
    )
    if (
        not isinstance(specialist, dict)
        or not isinstance(fixed_kl, dict)
        or fixed_kl["source_checkpoint_fidelity"] != specialist.get("checkpoint_fidelity")
    ):
        raise RuntimeError(
            "Training run contract fixed-KL source differs from its approved specialist"
        )
    if (
        not isinstance(expected_model_path, str)
        or os.path.realpath(expected_model_path) != kl_model_path
        or kl_receipt_path != os.path.join(kl_model_path, CHECKPOINT_FIDELITY_FILENAME)
    ):
        raise RuntimeError(
            "Training run contract exact KL source is not the policy initialization adapter"
        )
    try:
        live_fixed_kl = fixed_kl_reference_identity(kl_model_path, kl_receipt_path)
    except ValueError as exc:
        raise RuntimeError(
            f"Training run contract fixed-KL live verification failed: {exc}"
        ) from exc
    if live_fixed_kl != fixed_kl:
        raise RuntimeError(
            "Training run contract live fixed-KL reference differs from its embedded identity"
        )


def _locate_training_run_contract(model_path: str, run_dir: str | None) -> str | None:
    """Locate the nearest immutable run contract using the evaluator's fixed search."""

    candidates: list[str] = []
    if run_dir:
        candidates.append(os.path.join(run_dir, "run_contract.json"))
    current = os.path.realpath(model_path)
    for _ in range(4):
        candidates.append(os.path.join(current, "run_contract.json"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return next((path for path in candidates if os.path.isfile(path)), None)


def _training_run_contract_identity(
    *,
    model_path: str,
    run_dir: str | None,
    required: bool,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Locate and content-bind the immutable GRPO training contract.

    The contract's canonical payload is embedded in evaluation provenance while the
    byte-level digest protects its exact on-disk receipt.  Absolute paths are excluded
    from fingerprint material so an immutable run can be relocated without changing its
    scientific identity.
    """

    contract_path = _locate_training_run_contract(model_path, run_dir)
    if contract_path is None:
        if required:
            raise RuntimeError(
                "Confirmation of a GRPO checkpoint requires its immutable "
                "run_contract.json; none was found at the resolved run/model path."
            )
        return None, None, None

    try:
        contract, file_identity = _load_json_snapshot(
            contract_path,
            label="training run contract",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    # Historical development contracts predate the publication schema and remain
    # inspectable as development-only provenance.  Schema-v2 contracts, and every
    # contract used for confirmation, must satisfy the exact publication validator.
    if required or contract.get("schema_version") == 2:
        try:
            _validate_training_run_contract(contract)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}: {contract_path}") from exc
    identity = {
        "basename": os.path.basename(contract_path),
        "sha256": file_identity["sha256"],
        "size_bytes": file_identity["size_bytes"],
        "payload_sha256": _experiment_fingerprint(contract),
    }
    return contract, str(file_identity["sha256"]), identity


def _publication_exact_runtime_identity(
    *,
    model_path: str,
    run_dir: str | None,
    training_run_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Live-bind root and selected-checkpoint exact-pair receipts without mutating contract."""

    beta, fixed_kl = _validate_fixed_kl_projection(
        training_run_contract.get("training_config", {}).get("beta"),
        training_run_contract.get("fixed_kl_reference"),
        label="Training run contract",
    )
    contract_path = _locate_training_run_contract(model_path, run_dir)
    if contract_path is None:
        raise RuntimeError("Publication exact-runtime verification cannot locate run_contract.json")
    root = os.path.dirname(os.path.realpath(contract_path))
    checkpoint = os.path.realpath(model_path)
    receipt_paths = {
        os.path.join(root, EXACT_PEFT_KL_REFERENCE_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_REFERENCE_FILENAME),
    }
    post_step_paths = {
        os.path.join(root, EXACT_PEFT_KL_STEP1_FILENAME),
        os.path.join(root, EXACT_PEFT_KL_POST_STEP_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_STEP1_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_POST_STEP_FILENAME),
    }
    if beta == 0.0:
        stale = sorted(path for path in receipt_paths | post_step_paths if os.path.lexists(path))
        if stale:
            raise RuntimeError(
                "Publication beta=0 run carries an unexpected exact PEFT KL runtime receipt: "
                + ", ".join(stale)
            )
        return None
    if not isinstance(fixed_kl, dict):
        raise RuntimeError("Publication beta>0 run has no fixed-KL source commitment")
    missing = sorted(path for path in receipt_paths if not os.path.isfile(path))
    if missing:
        raise RuntimeError(
            "Publication beta>0 run is missing an exact PEFT KL runtime receipt: "
            + ", ".join(missing)
        )
    identities: list[dict[str, Any]] = []
    for path in sorted(receipt_paths):
        try:
            identities.append(
                exact_peft_kl_runtime_identity(
                    path,
                    expected_fixed_kl_reference=fixed_kl,
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Publication exact PEFT KL runtime verification failed for {path}: {exc}"
            ) from exc
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("Publication root and checkpoint exact PEFT KL runtime receipts differ")
    runtime = identities[0]
    horizon = training_run_contract.get("fixed_horizon", {})
    training = training_run_contract.get("training_config", {})
    if runtime["distributed_world_size"] != horizon.get("world_size"):
        raise RuntimeError(
            "Publication exact PEFT KL runtime world size differs from the run contract"
        )
    if (
        float(runtime["training_beta"]) != beta
        or runtime["training_dtype"] != training.get("dtype")
        or runtime["training_sync_ref_model"] is not False
        or runtime["training_use_vllm"] is not False
    ):
        raise RuntimeError(
            "Publication exact PEFT KL runtime training settings differ from the run contract"
        )
    return runtime


def _publication_exact_post_step_identity(
    *,
    model_path: str,
    run_dir: str | None,
    training_run_contract: dict[str, Any],
    training_run_contract_sha256: str,
    exact_runtime: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Live-bind root/selected post-optimization reference immutability evidence."""

    beta, fixed = _validate_fixed_kl_projection(
        training_run_contract.get("training_config", {}).get("beta"),
        training_run_contract.get("fixed_kl_reference"),
        label="Training run contract",
    )
    if beta == 0.0:
        if exact_runtime is not None:
            raise RuntimeError("Publication beta=0 run carries an exact step-zero runtime")
        return None
    if not isinstance(fixed, dict) or not isinstance(exact_runtime, dict):
        raise RuntimeError("Publication beta>0 run lacks exact step-zero authorization")
    contract_path = _locate_training_run_contract(model_path, run_dir)
    if contract_path is None:
        raise RuntimeError("Publication post-step verification cannot locate run_contract.json")
    root = os.path.dirname(os.path.realpath(contract_path))
    terminal_step = int(training_run_contract["fixed_horizon"]["max_steps"])
    try:
        return exact_peft_kl_post_step_lineage_identity(
            root,
            os.path.realpath(model_path),
            expected_fixed_kl_reference=fixed,
            expected_runtime_identity=exact_runtime,
            training_run_contract_file_sha256=training_run_contract_sha256,
            terminal_global_step=terminal_step,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Publication exact PEFT KL post-step verification failed: {exc}"
        ) from exc


def _example_sampling_seed(global_seed: int, task: str, split: str, example_index: int) -> int:
    """Derive a stable per-row seed so resume/skips cannot shift later samples."""

    payload = {
        "schema_version": 1,
        "global_seed": int(global_seed),
        "task": str(task),
        "split": str(split),
        "example_index": int(example_index),
    }
    raw_value = int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(payload)).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return int(raw_value % (2**63 - 1))


def _build_experiment_provenance(
    *,
    model_path: str,
    tokenizer_path: str,
    model,
    tokenizer,
    data_identity: dict,
    prompting: dict,
    generation: dict,
    selection: dict,
    training_run_contract: dict[str, Any] | None = None,
    training_run_contract_sha256: str | None = None,
    checkpoint_artifact: dict[str, Any] | None = None,
    checkpoint_fidelity: dict[str, Any] | None = None,
    exact_peft_kl_runtime: dict[str, Any] | None = None,
    exact_peft_kl_post_step: dict[str, Any] | None = None,
) -> tuple[dict, str, str]:
    """Build complete experiment material plus experiment/protocol fingerprints."""

    model_artifact = checkpoint_artifact or _reference_content_identity(
        model_path,
        inference_artifacts_only=True,
    )
    base_reference = _raw_base_model_reference(model_path)
    if os.path.realpath(base_reference) == os.path.realpath(model_path):
        base_identity = model_artifact
    else:
        base_identity = _reference_content_identity(
            base_reference,
            inference_artifacts_only=True,
        )
    loaded_revision = _loaded_model_revision(model)
    is_adapter = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    if is_adapter and base_identity.get("kind") == "external_reference" and not loaded_revision:
        raise RuntimeError(
            "Cannot establish an immutable content identity for the adapter's base model "
            f"reference {base_reference!r}. Use a local base-model snapshot or a Hub snapshot "
            "that exposes its commit hash; evaluation artifacts would otherwise be ambiguous."
        )

    tokenizer_artifact = _tokenizer_content_identity(tokenizer_path)
    normalized_checkpoint_fidelity = (
        validate_publication_fidelity_identity(checkpoint_fidelity)
        if checkpoint_fidelity is not None
        else None
    )
    normalized_exact_runtime = (
        validate_exact_peft_kl_runtime_identity(exact_peft_kl_runtime)
        if exact_peft_kl_runtime is not None
        else None
    )
    normalized_exact_post_step = (
        validate_exact_peft_kl_post_step_lineage(
            exact_peft_kl_post_step,
            expected_runtime_identity=normalized_exact_runtime,
            expected_run_contract_file_sha256=training_run_contract_sha256,
        )
        if exact_peft_kl_post_step is not None
        else None
    )

    material = {
        "schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "model": {
            "checkpoint_artifact": model_artifact,
            "checkpoint_fidelity": normalized_checkpoint_fidelity,
            "exact_peft_kl_runtime": normalized_exact_runtime,
            "exact_peft_kl_post_step": normalized_exact_post_step,
            "base_model_reference": base_reference,
            "base_model_artifact": base_identity,
            "loaded_revision": loaded_revision,
            "loaded_config": _loaded_model_config_identity(model),
            "tokenizer_artifact": tokenizer_artifact,
            "loaded_tokenizer": _loaded_tokenizer_identity(tokenizer),
        },
        "training_run_contract": training_run_contract,
        "training_run_contract_sha256": training_run_contract_sha256,
        "data": data_identity,
        "prompting": prompting,
        "generation": generation,
        "selection": selection,
        "verifier": _strict_verifier_identity(),
        "code": _source_code_identity(),
        "runtime_versions": _runtime_versions(),
    }
    experiment_fingerprint = _experiment_fingerprint(material)
    protocol_material = _protocol_fingerprint_material(material)
    protocol_fingerprint = _experiment_fingerprint(protocol_material)
    return material, experiment_fingerprint, protocol_fingerprint


def _artifact_mtime(model_path: str) -> float:
    candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
    )
    mtimes = []
    for name in candidates:
        file_path = os.path.join(model_path, name)
        if os.path.isfile(file_path):
            mtimes.append(os.path.getmtime(file_path))
    if mtimes:
        return max(mtimes)
    return os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0


def _resolve_model_from_run_dir(run_dir: str, stage: str) -> str | None:
    progress = inspect_stage_progress(run_dir, stage)
    latest_checkpoint_dir = progress["latest_checkpoint_dir"]
    if progress["completed"] and progress["has_saved_model"]:
        return run_dir
    if latest_checkpoint_dir:
        return latest_checkpoint_dir
    if progress["has_saved_model"]:
        return run_dir
    return None


def _resolve_latest_model_path(output_root: str, stage: str) -> tuple[str, str]:
    stage_root = os.path.join(output_root, stage)
    if not os.path.isdir(stage_root):
        raise FileNotFoundError(f"Stage directory does not exist: {stage_root}")

    best_model_path = None
    best_run_dir = None
    best_mtime = -1.0

    for name in sorted(os.listdir(stage_root)):
        run_dir = os.path.join(stage_root, name)
        if not os.path.isdir(run_dir):
            continue
        model_path = _resolve_model_from_run_dir(run_dir, stage)
        if not model_path:
            continue
        artifact_mtime = _artifact_mtime(model_path)
        if artifact_mtime > best_mtime:
            best_mtime = artifact_mtime
            best_model_path = model_path
            best_run_dir = run_dir

    if not best_model_path or not best_run_dir:
        raise FileNotFoundError(
            f"No saved model or checkpoint could be resolved under {stage_root}"
        )
    return best_model_path, best_run_dir


def _nearest_run_dir(model_path: str, output_root: str, stage: str) -> str | None:
    # Resolve checkpoint aliases before walking to the run directory.  An abspath-only
    # walk stops beside a convenience symlink and loses the run's hyperparams.json,
    # which can silently select the wrong prompt/representation at evaluation time.
    normalized = os.path.realpath(model_path)
    if os.path.basename(normalized).startswith("checkpoint-"):
        candidate = os.path.dirname(normalized)
        if os.path.basename(candidate) == "checkpoints":
            parent = os.path.dirname(candidate)
            if os.path.isdir(parent):
                return parent
        return candidate if os.path.isdir(candidate) else None

    if os.path.basename(normalized) == "merged_for_grpo":
        candidate = os.path.dirname(normalized)
        return candidate if os.path.isdir(candidate) else None

    stage_root = os.path.abspath(os.path.join(output_root, stage))
    if normalized.startswith(stage_root + os.sep):
        return normalized
    if output_dir_has_saved_model(normalized) or find_latest_checkpoint_dir(normalized):
        return normalized
    return None


def _resolve_model_and_run_dir(args) -> tuple[str, str | None]:
    if args.model_path:
        model_path = os.path.abspath(args.model_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return model_path, _nearest_run_dir(model_path, args.output_root, args.stage)
    return _resolve_latest_model_path(args.output_root, args.stage)


def _load_stage_hyperparams(run_dir: str | None) -> dict:
    if not run_dir:
        return {}
    payload = _json_load(os.path.join(run_dir, HYPERPARAMS_FILENAME)) or {}
    stage_hparams = payload.get("stage_hyperparameters")
    return stage_hparams if isinstance(stage_hparams, dict) else {}


_REPRESENTATION_HPARAM_TO_ENV = {
    "clasp_tsp_prompt_knn": "CLASP_TSP_PROMPT_KNN",
    "clasp_tsp_prompt_knn_distances": "CLASP_TSP_PROMPT_KNN_DISTANCES",
    "clasp_tsp_input_order": "CLASP_TSP_INPUT_ORDER",
    "clasp_tsp_relabel_order": "CLASP_TSP_RELABEL_ORDER",
    "clasp_tsp_route_canonicalize": "CLASP_TSP_ROUTE_CANONICALIZE",
    "clasp_tsp_route_only_output": "CLASP_TSP_ROUTE_ONLY_OUTPUT",
    "clasp_tsp_label_tta_schedule": "CLASP_TSP_LABEL_TTA_SCHEDULE",
    "clasp_mis_relabel_order": "CLASP_MIS_RELABEL_ORDER",
    "clasp_mis_jiang_response_output": "CLASP_MIS_JIANG_RESPONSE_OUTPUT",
    "clasp_mis_prompt_node_degrees": "CLASP_MIS_PROMPT_NODE_DEGREES",
    "clasp_mis_prompt_low_degree_k": "CLASP_MIS_PROMPT_LOW_DEGREE_K",
}


def _restore_stage_representation_env(stage_hparams: dict) -> dict[str, str]:
    """Restore recorded task representation switches and protect Jiang replication.

    Training-time prompt transforms are part of the model contract.  New SFT runs persist
    them in ``hyperparams.json``. Jiang-f638 specialists reject a conflicting
    inherited environment instead of silently changing their instruction and
    label grammar. Older checkpoints retain the explicit-override behavior.
    """

    restored: dict[str, str] = {}
    if not isinstance(stage_hparams, dict):
        return restored
    strict_jiang_contract = str(stage_hparams.get("prompt_style", "")) == "jiang_f638"
    for hyperparam_name, env_name in _REPRESENTATION_HPARAM_TO_ENV.items():
        if hyperparam_name not in stage_hparams:
            continue
        raw_value = stage_hparams.get(hyperparam_name)
        if raw_value is None:
            continue
        value = str(raw_value)
        if env_name in os.environ:
            if strict_jiang_contract and os.environ[env_name] != value:
                raise RuntimeError(
                    "Jiang-f638 evaluation environment conflicts with the recorded "
                    f"representation contract: {env_name}={os.environ[env_name]!r}, "
                    f"expected {value!r}"
                )
            continue
        os.environ[env_name] = value
        restored[env_name] = value
    return restored


def _parse_instance_value(row: dict) -> Any:
    value = row.get("instance_json", row.get("instance"))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _mean_or_none(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(mean(kept)) if kept else None


def _median_or_none(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(median(kept)) if kept else None


def _stdev_or_none(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return None
    if len(kept) == 1:
        return 0.0
    return float(stdev(kept))


def _objective_gap(task: str, pred_value: float | None, gt_value: float | None) -> float | None:
    if pred_value is None or gt_value is None or gt_value <= 0.0:
        return None
    if TASK_DIRECTIONS[task] == "min":
        return float(max(0.0, (pred_value - gt_value) / gt_value))
    return float(max(0.0, (gt_value - pred_value) / gt_value))


def _evaluate_tsp(
    prompt: str, input_text: str, answer: str, instance: Any, completion: str
) -> dict:
    parsed = _parse_tsp_solution(completion)
    ref = _parse_tsp_solution(answer)
    coords = _tsp_coords_from_instance(instance, input_text)
    pred_value = _distance(coords, parsed["route"]) if coords else None
    gt_value = _distance(coords, ref["route"]) if coords else None
    if coords:
        passed, total, feasible = _check_tsp_constraints(parsed["route"], len(coords))
        constraint_fraction = passed / total
    else:
        feasible = False
        constraint_fraction = 0.0
    return {
        "parsed_solution": parsed["route"],
        "declared_objective": parsed["declared_objective"],
        "objective_value": pred_value,
        "ground_truth_objective": gt_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(constraint_fraction),
        "gap_to_ground_truth": _objective_gap("tsp", pred_value, gt_value) if feasible else None,
    }


def _evaluate_cvrp(
    prompt: str, input_text: str, answer: str, instance: Any, completion: str
) -> dict:
    combined_context = _combined_context(prompt, input_text)
    parsed = _parse_cvrp_solution(completion)
    ref = _parse_cvrp_solution(answer)
    coords, demands, capacity = _cvrp_meta(instance, combined_context)
    pred_value = _cvrp_total_distance(coords, parsed["routes"]) if coords else None
    gt_value = _cvrp_total_distance(coords, ref["routes"]) if coords else None
    if coords and demands:
        passed, total, feasible = _check_cvrp_constraints(
            parsed["routes"], coords, demands, capacity
        )
        constraint_fraction = passed / total
    else:
        feasible = False
        constraint_fraction = 0.0
    return {
        "parsed_solution": parsed["routes"],
        "declared_objective": parsed["declared_objective"],
        "objective_value": pred_value,
        "ground_truth_objective": gt_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(constraint_fraction),
        "gap_to_ground_truth": _objective_gap("cvrp", pred_value, gt_value) if feasible else None,
    }


def _evaluate_op(prompt: str, input_text: str, answer: str, instance: Any, completion: str) -> dict:
    combined_context = _combined_context(prompt, input_text)
    parsed = _parse_op_solution(completion)
    ref = _parse_op_solution(answer)
    coords, prizes, max_len = _op_meta(instance, combined_context)
    pred_value = _op_prize(parsed["route"], prizes) if prizes else None
    gt_value = _op_prize(ref["route"], prizes) if prizes else None
    if coords and max_len is not None:
        passed, total, feasible = _check_op_constraints(parsed["route"], coords, max_len)
        constraint_fraction = passed / total
    else:
        feasible = False
        constraint_fraction = 0.0
    return {
        "parsed_solution": parsed["route"],
        "declared_objective": parsed["declared_objective"],
        "objective_value": pred_value,
        "ground_truth_objective": gt_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(constraint_fraction),
        "gap_to_ground_truth": _objective_gap("op", pred_value, gt_value) if feasible else None,
    }


def _evaluate_mis(
    prompt: str, input_text: str, answer: str, instance: Any, completion: str
) -> dict:
    combined_context = _combined_context(prompt, input_text)
    parsed = _parse_mis_solution(completion)
    ref = _parse_mis_solution(answer)
    num_nodes, edges = _mis_meta(instance, combined_context)
    passed, total, feasible = _check_mis_constraints(parsed["node_set"], num_nodes, edges)
    pred_value = float(len(parsed["node_set"])) if parsed["node_set"] else None
    gt_value = float(len(ref["node_set"])) if ref["node_set"] else None
    return {
        "parsed_solution": parsed["node_set"],
        "declared_objective": parsed["declared_objective"],
        "objective_value": pred_value,
        "ground_truth_objective": gt_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(passed / total),
        "gap_to_ground_truth": _objective_gap("mis", pred_value, gt_value) if feasible else None,
    }


def _evaluate_jssp(
    prompt: str, input_text: str, answer: str, instance: Any, completion: str
) -> dict:
    combined_context = _combined_context(prompt, input_text)
    parsed = _parse_jssp_solution(completion)
    ref = _parse_jssp_solution(answer)
    jobs = _jssp_jobs(instance, combined_context)
    pred_value = _schedule_makespan(jobs, parsed["schedule"]) if jobs else None
    gt_value = _schedule_makespan(jobs, ref["schedule"]) if jobs else None
    if jobs:
        passed, total, feasible = _check_jssp_constraints(jobs, parsed["schedule"])
        constraint_fraction = passed / total
    else:
        feasible = False
        constraint_fraction = 0.0
    return {
        "parsed_solution": parsed["schedule"],
        "declared_objective": parsed["declared_makespan"],
        "objective_value": pred_value,
        "ground_truth_objective": gt_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(constraint_fraction),
        "gap_to_ground_truth": _objective_gap("jssp", pred_value, gt_value) if feasible else None,
    }


TASK_EVALUATORS = {
    "tsp": _evaluate_tsp,
    "cvrp": _evaluate_cvrp,
    "op": _evaluate_op,
    "mis": _evaluate_mis,
    "jssp": _evaluate_jssp,
}


def _constraint_diagnostics(
    task: str,
    *,
    prompt: str,
    input_text: str,
    answer: str,
    instance: Any,
    completion: str,
) -> dict[str, object]:
    """Return named active constraint severities from the shared reward verifier."""

    result = TASK_COST_COMPONENTS[task](
        [prompt],
        [completion],
        input=[input_text],
        answer=[answer],
        instance=[instance],
        instance_json=[instance],
    )
    names = result.get("cost_names")
    vectors = result.get("cost_vectors")
    masks = result.get("cost_masks")
    if not (
        isinstance(names, list)
        and isinstance(vectors, list)
        and len(vectors) == 1
        and isinstance(vectors[0], list)
        and isinstance(masks, list)
        and len(masks) == 1
        and isinstance(masks[0], list)
        and len(names) == len(vectors[0]) == len(masks[0])
    ):
        raise RuntimeError(f"invalid named constraint diagnostics for task={task}")
    active = {
        str(name): float(value)
        for name, value, mask in zip(names, vectors[0], masks[0], strict=True)
        if float(mask) > 0.0
    }
    return {
        "constraint_violations": active,
        "parse_failure": bool(float(active.get("parse_failure", 0.0)) > 0.0),
    }


def _select_best_candidate_index(task: str, candidates: list[dict]) -> int | None:
    feasible = [
        (idx, candidate) for idx, candidate in enumerate(candidates) if candidate.get("is_feasible")
    ]
    if not feasible:
        return None

    direction = TASK_DIRECTIONS[task]

    def sort_key(item):
        idx, candidate = item
        gap = candidate.get("gap_to_ground_truth")
        objective = candidate.get("objective_value")
        gap_key = float(gap) if gap is not None else float("inf")
        if direction == "min":
            objective_key = float(objective) if objective is not None else float("inf")
        else:
            objective_key = -float(objective) if objective is not None else float("inf")
        return (gap_key, objective_key, idx)

    return min(feasible, key=sort_key)[0]


def _stage_label(stage: str) -> str:
    return "sft_only" if stage == "sft" else "grpo"


def _stage_dirname(stage: str) -> str:
    return "sft" if stage == "sft" else "grpo"


def _bounded_filesystem_label(label: str, *, max_length: int = 160) -> str:
    """Keep generated artifact filenames below common filesystem limits."""
    clean = sanitize_run_name_component(label)
    if len(clean) <= max_length:
        return clean

    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:10]
    prefix_length = max(1, max_length - len(digest) - 1)
    prefix = clean[:prefix_length].rstrip("_-.")
    if not prefix:
        prefix = clean[:prefix_length]
    return f"{prefix}-{digest}"


def _resolve_checkpoint_name(model_path: str, run_dir: str | None) -> str:
    model_path = os.path.abspath(model_path)
    model_base = os.path.basename(model_path)
    if model_base.startswith("checkpoint-") and run_dir:
        return sanitize_run_name_component(f"{os.path.basename(run_dir)}-{model_base}")
    if run_dir and os.path.abspath(run_dir) == model_path:
        return sanitize_run_name_component(os.path.basename(run_dir))
    return resolve_model_artifact_name(model_path)


def _resolve_inference_root(args) -> str:
    root = (
        args.output_dir if args.output_dir else os.path.join(_RUN_EVAL_PARENT, "inference_results")
    )
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def _default_results_dir(args, checkpoint_name: str) -> str:
    base_dir = os.path.join(
        _resolve_inference_root(args),
        _stage_dirname(args.stage),
        checkpoint_name,
    )
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _results_label(
    args,
    checkpoint_name: str,
    *,
    experiment_fingerprint: str | None = None,
) -> str:
    base = f"{_stage_label(args.stage)}_{checkpoint_name}_{args.split}_k{args.num_return_sequences}"
    if args.eval_name:
        base = f"{base}_{args.eval_name}"
    if experiment_fingerprint:
        base = f"{base}_fp{experiment_fingerprint[:12]}"
    return _bounded_filesystem_label(base)


def _best_run_scope_label(
    args,
    *,
    do_sample: bool,
    protocol_fingerprint: str | None = None,
) -> str:
    max_eval = int(args.max_eval_samples_per_task)
    max_eval_label = str(max_eval) if max_eval > 0 else "all"
    per_task_caps = dict(getattr(args, "eval_samples_per_task_map", {}) or {})
    caps_label = (
        "caps-" + "-".join(f"{task}{int(per_task_caps[task])}" for task in sorted(per_task_caps))
        if per_task_caps
        else "caps-none"
    )
    task_label = "-".join(args.tasks)
    mode_label = "sample" if do_sample else "greedy"
    start_label = f"start{int(getattr(args, 'eval_start_index', 0) or 0)}"
    seed_label = f"seed{int(getattr(args, 'seed', 42))}"
    scope = sanitize_run_name_component(
        f"{args.split}_{task_label}_{mode_label}_k{args.num_return_sequences}_"
        f"{seed_label}_{start_label}_max{max_eval_label}_{caps_label}"
    )
    if protocol_fingerprint:
        scope = sanitize_run_name_component(f"{scope}_protocol{protocol_fingerprint[:12]}")
    return scope


def _limit_prompt_length(
    requested_max_prompt_length: int,
    stage_hparams: dict,
    context_window: int | None,
    max_new_tokens: int,
) -> int | None:
    prompt_limit = (
        int(requested_max_prompt_length) if int(requested_max_prompt_length) > 0 else None
    )

    if prompt_limit is None:
        if "evaluation_max_prompt_length" in stage_hparams:
            prompt_limit = int(stage_hparams["evaluation_max_prompt_length"])
        elif "max_prompt_length" in stage_hparams:
            prompt_limit = int(stage_hparams["max_prompt_length"])
        elif "max_seq_length" in stage_hparams:
            prompt_limit = max(1, int(stage_hparams["max_seq_length"]) - int(max_new_tokens))

    if context_window is not None:
        context_based_limit = max(1, int(context_window) - int(max_new_tokens))
        prompt_limit = (
            min(prompt_limit, context_based_limit)
            if prompt_limit is not None
            else context_based_limit
        )

    return prompt_limit


def _resolve_max_new_tokens(args, stage_hparams: dict) -> int:
    if int(args.max_new_tokens) > 0:
        return int(args.max_new_tokens)
    if "evaluation_max_new_tokens" in stage_hparams:
        return int(stage_hparams["evaluation_max_new_tokens"])
    if "max_completion_length" in stage_hparams:
        return int(stage_hparams["max_completion_length"])
    return 1024


def _requested_eval_examples_for_task(args, task: str) -> int:
    task_caps = getattr(args, "eval_samples_per_task_map", {}) or {}
    if task in task_caps:
        return int(task_caps[task])
    return int(args.max_eval_samples_per_task)


def _instance_size_from_text(task: str, input_text: str) -> int | None:
    task = str(task).lower()
    text = str(input_text or "")
    patterns_by_task = {
        "tsp": (r"\b(\d+)\s+(?:cities|nodes)\b",),
        "cvrp": (r"\b(\d+)\s+(?:customers|nodes)\b",),
        "op": (r"\b(\d+)\s+(?:locations|nodes)\b",),
        "mis": (r"graph\s+with\s+(\d+)\s+nodes", r"\b(\d+)\s+nodes\b"),
        "jssp": (
            r"\b(\d+)\s+jobs?\b.*?\b(\d+)\s+machines?\b",
            r"\b(\d+)\s+machines?\b.*?\b(\d+)\s+jobs?\b",
        ),
    }
    for pattern in patterns_by_task.get(task, (r"\b(\d+)\s+nodes\b",)):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            values = [int(group) for group in match.groups() if group is not None]
        except ValueError:
            continue
        if task == "jssp" and len(values) >= 2:
            return int(values[0] * values[1])
        if values:
            return int(values[0])
    return None


def _infer_instance_size(task: str, instance: Any, input_text: str) -> int | None:
    task = str(task).lower()
    try:
        if task == "tsp":
            coords = _tsp_coords_from_instance(instance, input_text)
            if coords:
                return int(len(coords))
        elif task == "cvrp":
            coords, demands, _capacity = _cvrp_meta(instance, input_text)
            if coords:
                return int(len(coords))
            if demands:
                return int(len(demands))
        elif task == "op":
            coords, prizes, _max_len = _op_meta(instance, input_text)
            if coords:
                return int(len(coords))
            if prizes:
                return int(len(prizes))
        elif task == "mis":
            num_nodes, edges = _mis_meta(instance, input_text)
            if num_nodes is not None:
                return int(num_nodes)
            if edges:
                return int(max(max(edge) for edge in edges) + 1)
        elif task == "jssp":
            jobs = _jssp_jobs(instance, input_text)
            if jobs:
                return int(sum(len(job) for job in jobs))
    except Exception:
        pass
    return _instance_size_from_text(task, input_text)


def _mean_number(values: list[int]) -> float | None:
    return float(mean(values)) if values else None


def _median_number(values: list[int]) -> float | None:
    return float(median(values)) if values else None


def _size_summary(size_records: list[tuple[int, int | None]]) -> dict:
    known = [int(size) for _idx, size in size_records if size is not None]
    if not known:
        return {
            "known_examples": 0,
            "unknown_examples": len(size_records),
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "known_examples": len(known),
        "unknown_examples": len(size_records) - len(known),
        "min": int(min(known)),
        "median": _median_number(known),
        "mean": _mean_number(known),
        "max": int(max(known)),
    }


def _slice_known_size_bucket(
    records: list[tuple[int, int | None]],
    bucket: str,
) -> list[tuple[int, int | None]]:
    known_records = [(idx, int(size)) for idx, size in records if size is not None]
    if bucket == "all":
        return list(records)
    if not known_records:
        return []

    ordered = sorted(known_records, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    if bucket == "small":
        return ordered[: max(1, (n + 2) // 3)]
    if bucket == "medium":
        start = n // 3
        end = max(start + 1, (2 * n + 2) // 3)
        return ordered[start : min(end, n)]
    if bucket == "large":
        return ordered[(2 * n) // 3 :]
    if bucket == "hardest":
        count = max(1, (n + 4) // 5)
        return ordered[-count:]
    return list(records)


def _select_eval_indices_for_task(
    dataset, args, task: str
) -> tuple[list[int], dict, dict[int, int | None]]:
    requested_examples = _requested_eval_examples_for_task(args, task)
    start_index = int(getattr(args, "eval_start_index", 0) or 0)
    size_bucket = str(getattr(args, "size_bucket", "all") or "all").lower()
    min_size = int(getattr(args, "min_instance_size", 0) or 0)
    max_size = int(getattr(args, "max_instance_size", 0) or 0)

    size_by_index: dict[int, int | None] = {}
    all_records: list[tuple[int, int | None]] = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        instance = _parse_instance_value(row)
        input_text = str(row.get("input", ""))
        instance_size = _infer_instance_size(task, instance, input_text)
        size_by_index[idx] = instance_size
        all_records.append((idx, instance_size))

    filter_active = size_bucket != "all" or min_size > 0 or max_size > 0
    if filter_active:
        filtered_records = [
            (idx, size)
            for idx, size in all_records
            if size is not None
            and (min_size <= 0 or int(size) >= min_size)
            and (max_size <= 0 or int(size) <= max_size)
        ]
    else:
        filtered_records = list(all_records)

    bucket_records = _slice_known_size_bucket(filtered_records, size_bucket)
    bucket_records_sorted_by_index = sorted(bucket_records, key=lambda item: item[0])

    if size_bucket in {"large", "hardest"}:
        ordered_records = sorted(
            bucket_records,
            key=lambda item: (-(int(item[1]) if item[1] is not None else -1), item[0]),
        )
    elif size_bucket == "small":
        ordered_records = sorted(
            bucket_records,
            key=lambda item: (int(item[1]) if item[1] is not None else 10**18, item[0]),
        )
    else:
        ordered_records = bucket_records_sorted_by_index
    sliced_records = ordered_records[start_index:]
    if requested_examples > 0:
        sliced_records = sliced_records[:requested_examples]
    selected_indices = sorted(idx for idx, _size in sliced_records)

    selected_records = [(idx, size_by_index.get(idx)) for idx in selected_indices]
    selection_info = {
        "size_bucket": size_bucket,
        "min_instance_size": min_size,
        "max_instance_size": max_size,
        "size_filter_active": bool(filter_active),
        "available_examples": int(len(dataset)),
        "bucket_examples_before_cap": int(len(bucket_records)),
        "requested_examples": int(requested_examples),
        "eval_start_index": int(start_index),
        "loaded_examples": int(len(selected_indices)),
        "all_size_summary": _size_summary(all_records),
        "selected_size_summary": _size_summary(selected_records),
    }
    return selected_indices, selection_info, size_by_index


def _infer_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _build_instance_record(
    *,
    task: str,
    split: str,
    example_index: int,
    source_json: str | None,
    source_json_files: list[str],
    instruction: str,
    input_text: str,
    answer: str,
    instance_size: int | None,
    prompt_tokens: int,
    prompt_was_truncated: bool,
    generation_time_sec: float,
    num_return_sequences: int,
    candidates: list[dict],
    peak_gpu_memory_allocated_bytes: int | None = None,
    peak_gpu_memory_reserved_bytes: int | None = None,
) -> dict:
    feasible_gaps = [
        float(candidate["gap_to_ground_truth"])
        for candidate in candidates
        if candidate.get("is_feasible") and candidate.get("gap_to_ground_truth") is not None
    ]
    feasible_count = sum(int(candidate.get("is_feasible")) for candidate in candidates)
    best_candidate_index = _select_best_candidate_index(task, candidates)
    best_candidate = candidates[best_candidate_index] if best_candidate_index is not None else None

    return {
        "task": task,
        "split": split,
        "example_index": int(example_index),
        "source_json": source_json,
        "source_json_files": list(source_json_files),
        "instruction": instruction,
        "input_problem": input_text,
        "ground_truth_output": answer,
        "instance_size": instance_size,
        "ground_truth_objective": (
            candidates[0].get("ground_truth_objective") if candidates else None
        ),
        "prompt_tokens": int(prompt_tokens),
        "prompt_was_truncated": bool(prompt_was_truncated),
        "generation_time_sec": float(generation_time_sec),
        "generation_time_sec_per_candidate": (float(generation_time_sec / max(len(candidates), 1))),
        "generated_tokens": int(
            sum(int(candidate.get("completion_tokens", 0) or 0) for candidate in candidates)
        ),
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
        "num_return_sequences": int(num_return_sequences),
        "feasible_candidates": int(feasible_count),
        "candidate_feasibility_rate": (
            float(feasible_count / len(candidates)) if candidates else 0.0
        ),
        "best_candidate_index": best_candidate_index,
        "best_of_n_feasible": bool(best_candidate_index is not None),
        "best_feasible_gap": (
            best_candidate.get("gap_to_ground_truth") if best_candidate is not None else None
        ),
        "best_objective_value": (
            best_candidate.get("objective_value") if best_candidate is not None else None
        ),
        "feasible_gap_mean_in_batch": _mean_or_none(feasible_gaps),
        "feasible_gap_std_in_batch": _stdev_or_none(feasible_gaps),
        "candidates": candidates,
    }


def _aggregate_records(records: list[dict]) -> dict:
    total_examples = len(records)
    total_candidates = sum(len(record["candidates"]) for record in records)
    feasible_candidates = sum(
        int(candidate["is_feasible"]) for record in records for candidate in record["candidates"]
    )
    examples_with_feasible = sum(int(record["best_of_n_feasible"]) for record in records)
    best_gaps = [
        float(record["best_feasible_gap"])
        for record in records
        if record["best_feasible_gap"] is not None
    ]
    feasible_gap_stds = [
        float(record["feasible_gap_std_in_batch"])
        for record in records
        if record["feasible_gap_std_in_batch"] is not None
    ]
    generation_times = [float(record["generation_time_sec"]) for record in records]
    generation_times_per_candidate = [
        float(record["generation_time_sec_per_candidate"]) for record in records
    ]
    generated_tokens = [int(record.get("generated_tokens", 0) or 0) for record in records]
    peak_allocated = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_allocated_bytes")) is not None
    ]
    peak_reserved = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_reserved_bytes")) is not None
    ]
    return {
        "examples": total_examples,
        "candidates": total_candidates,
        "feasible_candidates": feasible_candidates,
        "candidate_feasibility_rate": (
            float(feasible_candidates / total_candidates) if total_candidates else 0.0
        ),
        "best_of_n_feasible_examples": examples_with_feasible,
        "best_of_n_instance_feasibility_rate": (
            float(examples_with_feasible / total_examples) if total_examples else 0.0
        ),
        "mean_best_feasible_gap": _mean_or_none(best_gaps),
        "median_best_feasible_gap": _median_or_none(best_gaps),
        "std_best_feasible_gap": _stdev_or_none(best_gaps),
        "mean_feasible_gap_std_in_batch": _mean_or_none(feasible_gap_stds),
        "std_feasible_gap_std_in_batch": _stdev_or_none(feasible_gap_stds),
        "total_generation_time_sec": float(sum(generation_times)),
        "mean_generation_time_sec": float(mean(generation_times)) if generation_times else 0.0,
        "std_generation_time_sec": _stdev_or_none(generation_times),
        "mean_generation_time_sec_per_candidate": (
            float(mean(generation_times_per_candidate)) if generation_times_per_candidate else 0.0
        ),
        "std_generation_time_sec_per_candidate": _stdev_or_none(generation_times_per_candidate),
        "generated_tokens": int(sum(generated_tokens)),
        "mean_generated_tokens_per_instance": (
            float(mean(generated_tokens)) if generated_tokens else 0.0
        ),
        "peak_gpu_memory_allocated_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_gpu_memory_reserved_bytes": max(peak_reserved) if peak_reserved else None,
    }


def _format_metric(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fsync_directory(directory: str) -> None:
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl_atomic(path: str, records: list[dict]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _write_summary(results_dir: str, results_label: str, payload: dict) -> str:
    path = os.path.join(results_dir, f"run_summary__{results_label}.json")
    sealed = _self_hash_payload(payload, "summary_sha256")
    _write_json_atomic(path, sealed)
    latest_path = os.path.join(results_dir, "inference_summary.json")
    _write_json_atomic(latest_path, sealed)
    return path


def _write_details(results_dir: str, results_label: str, records: list[dict]) -> str:
    path = os.path.join(results_dir, f"run_details__{results_label}.jsonl")
    _write_jsonl_atomic(path, records)
    return path


def _read_detail_records_for_resume(details_path: str) -> list[dict]:
    if not os.path.isfile(details_path):
        return []

    raw_payload, _identity = _read_file_snapshot(details_path)
    raw_lines = raw_payload.splitlines()
    nonempty_indices = [index for index, raw_line in enumerate(raw_lines) if raw_line.strip()]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else None
    records: list[dict] = []
    skipped_trailing_line = False
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        line_number = index + 1
        try:
            line = raw_line.decode("utf-8")
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == last_nonempty and not raw_payload.endswith(b"\n"):
                skipped_trailing_line = True
                print_primary(
                    "[Eval][Resume] Ignoring one truncated final details line "
                    f"{line_number} in {details_path}."
                )
                continue
            raise RuntimeError(
                f"Cannot resume malformed interior details line {line_number}: {details_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Cannot resume non-object details line {line_number}: {details_path}"
            )
        records.append(payload)

    if skipped_trailing_line:
        print_primary(
            "[Eval][Resume] Loaded "
            f"{len(records)} valid detail records and skipped one truncated final line."
        )
    return records


def _rewrite_detail_records_atomic(details_path: str, records: list[dict]) -> None:
    _write_jsonl_atomic(details_path, records)


def _append_detail_record(details_path: str, record: dict) -> None:
    with open(details_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _detail_record_resume_key(record: dict) -> tuple[str, str, int] | None:
    try:
        example_index = int(record.get("example_index"))
    except (TypeError, ValueError):
        return None
    return (
        str(record.get("task", "")),
        str(record.get("split", "")),
        example_index,
    )


def _validate_resume_run_config(
    run_config: dict | None,
    *,
    experiment_fingerprint: str,
    run_config_path: str,
) -> dict:
    if not isinstance(run_config, dict):
        raise RuntimeError(
            "Cannot safely resume because the matching run config is missing or malformed: "
            f"{run_config_path}. Use --no-resume to start the fingerprinted run over."
        )
    stored_fingerprint = str(run_config.get("experiment_fingerprint", "") or "")
    if stored_fingerprint != experiment_fingerprint:
        raise RuntimeError(
            "Refusing to resume generations produced by an incompatible experiment. "
            f"Expected fingerprint {experiment_fingerprint}, found "
            f"{stored_fingerprint or '<missing>'} in {run_config_path}. "
            "The existing artifacts were left untouched."
        )
    _validate_self_hash(run_config, "run_config_sha256", label="run config")
    provenance = run_config.get("experiment_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Cannot safely resume: run config has no experiment_provenance")
    observed_fingerprint = _experiment_fingerprint(provenance)
    if stored_fingerprint != observed_fingerprint:
        raise RuntimeError(
            "Refusing to resume a run config whose experiment fingerprint does not "
            "match its embedded provenance."
        )
    return run_config


def _resume_records_by_key(
    records: list[dict],
    args,
    *,
    experiment_fingerprint: str,
) -> dict[tuple[str, str, int], dict]:
    expected_num_return_sequences = int(args.num_return_sequences)
    by_key: dict[tuple[str, str, int], dict] = {}
    for record_index, record in enumerate(records):
        record_fingerprint = str(record.get("experiment_fingerprint", "") or "")
        if record_fingerprint != experiment_fingerprint:
            raise RuntimeError(
                "Refusing to resume an incompatible detail record at zero-based record "
                f"index {record_index}: expected fingerprint {experiment_fingerprint}, "
                f"found {record_fingerprint or '<missing>'}. Existing artifacts were left "
                "untouched."
            )
        try:
            _validate_self_hash(
                record,
                "record_sha256",
                label=f"detail record {record_index}",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Refusing to resume a corrupted detail record: {exc}") from exc
        key = _detail_record_resume_key(record)
        if key is None:
            raise RuntimeError(
                f"Cannot resume malformed detail record {record_index}: invalid task/split/index."
            )
        try:
            record_num_return_sequences = int(record.get("num_return_sequences"))
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Cannot resume malformed detail record {record_index}: invalid candidate count."
            ) from None
        candidates = record.get("candidates")
        if record_num_return_sequences != expected_num_return_sequences:
            raise RuntimeError(
                "Refusing to resume a detail record with a different candidate count: "
                f"expected {expected_num_return_sequences}, found "
                f"{record_num_return_sequences} at record {record_index}."
            )
        if not isinstance(candidates, list) or len(candidates) != expected_num_return_sequences:
            raise RuntimeError(
                f"Cannot resume incomplete detail record {record_index}: expected "
                f"{expected_num_return_sequences} candidates."
            )
        if key in by_key:
            raise RuntimeError(f"Cannot resume duplicate detail record key: {key}")
        by_key[key] = record
    return by_key


def _resume_record_matches_example(
    record: dict,
    *,
    task: str,
    split: str,
    example_index: int,
    json_files: list[str],
    instruction: str,
    input_text: str,
    answer: str,
    experiment_fingerprint: str,
    data_manifest_sha256: str | None = None,
    expected_sampling_seed: int | None = None,
    expected_semantic_instance_sha256: str | None = None,
    expected_benchmark_row_sha256: str | None = None,
    expected_reference_output_sha256: str | None = None,
) -> bool:
    if str(record.get("experiment_fingerprint", "") or "") != experiment_fingerprint:
        return False
    if expected_sampling_seed is not None:
        try:
            if int(record.get("sampling_seed")) != int(expected_sampling_seed):
                return False
        except (TypeError, ValueError):
            return False
    if _detail_record_resume_key(record) != (str(task), str(split), int(example_index)):
        return False
    locator = record.get("source_row_locator")
    if isinstance(locator, dict) and data_manifest_sha256:
        if locator != {
            "data_manifest_sha256": data_manifest_sha256,
            "task": str(task),
            "split": str(split),
            "merged_dataset_index": int(example_index),
        }:
            return False
    elif list(record.get("source_json_files", []) or []) != list(json_files):
        return False
    for field, expected in (
        ("semantic_instance_sha256", expected_semantic_instance_sha256),
        ("benchmark_row_sha256", expected_benchmark_row_sha256),
        ("reference_output_sha256", expected_reference_output_sha256),
    ):
        if expected is not None and record.get(field) != expected:
            return False
    return (
        str(record.get("instruction", "")) == instruction
        and str(record.get("input_problem", "")) == input_text
        and str(record.get("ground_truth_output", "")) == answer
    )


def _write_run_config(results_dir: str, results_label: str, payload: dict) -> str:
    path = os.path.join(results_dir, f"run_config__{results_label}.json")
    sealed = _self_hash_payload(payload, "run_config_sha256")
    _write_json_atomic(path, sealed)
    latest_path = os.path.join(results_dir, "inference_hyperparams.json")
    _write_json_atomic(latest_path, sealed)
    return path


def _run_selection_score(summary_payload: dict) -> tuple[float, float]:
    """Return a feasibility-first score suitable for ascending tuple comparison."""

    overall = summary_payload.get("overall", {}) if isinstance(summary_payload, dict) else {}
    feasibility = (
        overall.get("best_of_n_instance_feasibility_rate") if isinstance(overall, dict) else None
    )
    gap = overall.get("mean_best_feasible_gap") if isinstance(overall, dict) else None
    try:
        feasibility_score = -float(feasibility)
    except (TypeError, ValueError):
        feasibility_score = float("inf")
    try:
        gap_score = float(gap)
    except (TypeError, ValueError):
        gap_score = float("inf")
    return feasibility_score, gap_score


def _copy_run_artifacts_to_best_run(
    *,
    args,
    scope_label: str,
    run_summary: dict,
    run_summary_path: str,
    run_config: dict,
    run_config_path: str,
    details_path: str | None,
    csv_by_task: dict[str, str],
) -> str | None:
    best_run_dir = os.path.join(
        _resolve_inference_root(args),
        "best_run",
        _stage_dirname(args.stage),
        scope_label,
    )
    existing_summary_path = os.path.join(best_run_dir, "best_run_summary.json")
    existing_summary = (
        _json_load(existing_summary_path) if os.path.isfile(existing_summary_path) else None
    )

    current_score = _run_selection_score(run_summary)
    existing_score = _run_selection_score(existing_summary or {})

    if existing_summary is not None and current_score > existing_score:
        return None

    shutil.rmtree(best_run_dir, ignore_errors=True)
    os.makedirs(best_run_dir, exist_ok=True)

    run_summary_for_best = dict(run_summary)
    run_summary_for_best["best_run_source"] = {
        "run_summary_path": run_summary_path,
        "run_config_path": run_config_path,
        "details_path": details_path,
        "csv_by_task": dict(csv_by_task),
    }
    run_summary_for_best["best_run_selection_metric"] = {
        "metric": (
            "lexicographic(overall.best_of_n_instance_feasibility_rate desc, "
            "overall.mean_best_feasible_gap asc)"
        ),
        "score": {
            "best_of_n_instance_feasibility_rate": -current_score[0],
            "mean_best_feasible_gap": current_score[1],
        },
        "scope_label": scope_label,
    }

    with open(os.path.join(best_run_dir, "best_run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary_for_best, handle, indent=2, ensure_ascii=True)

    with open(os.path.join(best_run_dir, "best_run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=True)

    shutil.copy2(run_summary_path, os.path.join(best_run_dir, os.path.basename(run_summary_path)))
    shutil.copy2(run_config_path, os.path.join(best_run_dir, os.path.basename(run_config_path)))
    source_results_dir = os.path.dirname(run_summary_path)
    fixed_summary = os.path.join(source_results_dir, "inference_summary.json")
    fixed_hparams = os.path.join(source_results_dir, "inference_hyperparams.json")
    if os.path.isfile(fixed_summary):
        shutil.copy2(fixed_summary, os.path.join(best_run_dir, "inference_summary.json"))
    if os.path.isfile(fixed_hparams):
        shutil.copy2(fixed_hparams, os.path.join(best_run_dir, "inference_hyperparams.json"))
    if details_path and os.path.isfile(details_path):
        shutil.copy2(details_path, os.path.join(best_run_dir, os.path.basename(details_path)))
    for csv_path in csv_by_task.values():
        if os.path.isfile(csv_path):
            shutil.copy2(csv_path, os.path.join(best_run_dir, os.path.basename(csv_path)))

    return best_run_dir


def _write_task_csv(
    *,
    results_dir: str,
    results_label: str,
    task: str,
    model_stage: str,
    checkpoint_name: str,
    model_path: str,
    split: str,
    available_examples: int,
    requested_examples: int,
    loaded_examples: int,
    records: list[dict],
    summary: dict,
) -> str:
    path = os.path.join(results_dir, f"{task}__{results_label}.csv")
    fieldnames = [
        "row_type",
        "model_stage",
        "task",
        "split",
        "checkpoint_name",
        "model_path",
        "example_index",
        "instance_size",
        "available_examples",
        "requested_examples",
        "loaded_examples",
        "num_candidates",
        "feasible_candidates",
        "candidate_feasibility_rate",
        "best_of_n_feasible",
        "best_candidate_index",
        "best_feasible_gap",
        "feasible_gap_mean_in_batch",
        "feasible_gap_std_in_batch",
        "best_objective_value",
        "ground_truth_objective",
        "generation_time_sec",
        "generation_time_sec_per_candidate",
        "prompt_tokens",
        "prompt_was_truncated",
        "best_of_n_instance_feasibility_rate",
        "mean_best_feasible_gap",
        "median_best_feasible_gap",
        "std_best_feasible_gap",
        "mean_feasible_gap_std_in_batch",
        "std_feasible_gap_std_in_batch",
        "mean_generation_time_sec",
        "std_generation_time_sec",
        "mean_generation_time_sec_per_candidate",
        "std_generation_time_sec_per_candidate",
        "source_json",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "row_type": "instance",
                    "model_stage": model_stage,
                    "task": task,
                    "split": split,
                    "checkpoint_name": checkpoint_name,
                    "model_path": model_path,
                    "example_index": record["example_index"],
                    "instance_size": record.get("instance_size", ""),
                    "available_examples": available_examples,
                    "requested_examples": requested_examples,
                    "loaded_examples": loaded_examples,
                    "num_candidates": len(record["candidates"]),
                    "feasible_candidates": record["feasible_candidates"],
                    "candidate_feasibility_rate": record["candidate_feasibility_rate"],
                    "best_of_n_feasible": int(record["best_of_n_feasible"]),
                    "best_candidate_index": (
                        ""
                        if record["best_candidate_index"] is None
                        else record["best_candidate_index"]
                    ),
                    "best_feasible_gap": record["best_feasible_gap"],
                    "feasible_gap_mean_in_batch": record["feasible_gap_mean_in_batch"],
                    "feasible_gap_std_in_batch": record["feasible_gap_std_in_batch"],
                    "best_objective_value": record["best_objective_value"],
                    "ground_truth_objective": record["ground_truth_objective"],
                    "generation_time_sec": record["generation_time_sec"],
                    "generation_time_sec_per_candidate": record[
                        "generation_time_sec_per_candidate"
                    ],
                    "prompt_tokens": record["prompt_tokens"],
                    "prompt_was_truncated": int(record["prompt_was_truncated"]),
                    "best_of_n_instance_feasibility_rate": "",
                    "mean_best_feasible_gap": "",
                    "median_best_feasible_gap": "",
                    "std_best_feasible_gap": "",
                    "mean_feasible_gap_std_in_batch": "",
                    "std_feasible_gap_std_in_batch": "",
                    "mean_generation_time_sec": "",
                    "std_generation_time_sec": "",
                    "mean_generation_time_sec_per_candidate": "",
                    "std_generation_time_sec_per_candidate": "",
                    "source_json": record["source_json"] or "",
                }
            )

        writer.writerow(
            {
                "row_type": "summary",
                "model_stage": model_stage,
                "task": task,
                "split": split,
                "checkpoint_name": checkpoint_name,
                "model_path": model_path,
                "example_index": "__summary__",
                "instance_size": "",
                "available_examples": available_examples,
                "requested_examples": requested_examples,
                "loaded_examples": loaded_examples,
                "num_candidates": summary["candidates"],
                "feasible_candidates": summary["feasible_candidates"],
                "candidate_feasibility_rate": summary["candidate_feasibility_rate"],
                "best_of_n_feasible": summary["best_of_n_feasible_examples"],
                "best_candidate_index": "",
                "best_feasible_gap": summary["mean_best_feasible_gap"],
                "feasible_gap_mean_in_batch": summary["mean_feasible_gap_std_in_batch"],
                "feasible_gap_std_in_batch": summary["std_feasible_gap_std_in_batch"],
                "best_objective_value": "",
                "ground_truth_objective": "",
                "generation_time_sec": summary["mean_generation_time_sec"],
                "generation_time_sec_per_candidate": summary[
                    "mean_generation_time_sec_per_candidate"
                ],
                "prompt_tokens": "",
                "prompt_was_truncated": "",
                "best_of_n_instance_feasibility_rate": summary[
                    "best_of_n_instance_feasibility_rate"
                ],
                "mean_best_feasible_gap": summary["mean_best_feasible_gap"],
                "median_best_feasible_gap": summary["median_best_feasible_gap"],
                "std_best_feasible_gap": summary["std_best_feasible_gap"],
                "mean_feasible_gap_std_in_batch": summary["mean_feasible_gap_std_in_batch"],
                "std_feasible_gap_std_in_batch": summary["std_feasible_gap_std_in_batch"],
                "mean_generation_time_sec": summary["mean_generation_time_sec"],
                "std_generation_time_sec": summary["std_generation_time_sec"],
                "mean_generation_time_sec_per_candidate": summary[
                    "mean_generation_time_sec_per_candidate"
                ],
                "std_generation_time_sec_per_candidate": summary[
                    "std_generation_time_sec_per_candidate"
                ],
                "source_json": "",
            }
        )
    return path


def _print_verbose_instance(record: dict) -> None:
    def _format_block(text: Any, *, is_input: bool) -> str:
        formatted = str(text).strip()
        if is_input:
            formatted = formatted.replace(";Node ", ";\nNode ")
        return formatted

    print_primary("")
    print_primary("=" * 80)
    print_primary(
        "[Eval][Instance] "
        f"task={record['task']} "
        f"index={record['example_index']} "
        f"best_of_n_feasible={int(record['best_of_n_feasible'])} "
        f"candidate_feasibility_rate={_format_metric(record['candidate_feasibility_rate'])} "
        f"best_gap={_format_metric(record['best_feasible_gap'])} "
        f"feasible_gap_std={_format_metric(record['feasible_gap_std_in_batch'])} "
        f"time_sec={_format_metric(record['generation_time_sec'])}"
    )
    print_primary("-" * 80)
    print_primary("[Eval][Input]")
    print_primary("")
    print_primary(_format_block(record["input_problem"], is_input=True))
    print_primary("")
    print_primary("[Eval][GroundTruth]")
    print_primary("")
    print_primary(_format_block(record["ground_truth_output"], is_input=False))
    for candidate in record["candidates"]:
        print_primary("")
        print_primary("-" * 80)
        print_primary(
            "[Eval][Candidate] "
            f"task={record['task']} "
            f"index={record['example_index']} "
            f"candidate_index={candidate['candidate_index']} "
            f"feasible={int(candidate['is_feasible'])} "
            f"constraint_fraction={_format_metric(candidate['constraint_fraction'])} "
            f"gap={_format_metric(candidate['gap_to_ground_truth'])} "
            f"objective={_format_metric(candidate['objective_value'])}"
        )
        print_primary("")
        print_primary(_format_block(candidate["completion"], is_input=False))
    if record["best_candidate_index"] is not None:
        print_primary("")
        print_primary(f"[Eval][BestCandidate] candidate_index={record['best_candidate_index']}")
    print_primary("=" * 80)
    print_primary("")


def _print_task_summary(
    task: str,
    summary: dict,
    *,
    available_examples: int,
    requested_examples: int,
    loaded_examples: int,
) -> None:
    print_primary(
        "[Eval][Summary] "
        f"task={task} "
        f"available_examples={available_examples} "
        f"requested_examples={requested_examples} "
        f"loaded_examples={loaded_examples} "
        f"candidate_feasibility_rate={_format_metric(summary['candidate_feasibility_rate'])} "
        f"best_of_n_instance_feasibility_rate={_format_metric(summary['best_of_n_instance_feasibility_rate'])} "
        f"mean_best_gap={_format_metric(summary['mean_best_feasible_gap'])} "
        f"std_best_gap={_format_metric(summary['std_best_feasible_gap'])} "
        f"mean_feasible_gap_std_in_batch={_format_metric(summary['mean_feasible_gap_std_in_batch'])} "
        f"mean_time_sec={_format_metric(summary['mean_generation_time_sec'])} "
        f"std_time_sec={_format_metric(summary['std_generation_time_sec'])}"
    )


def _resolve_prompt_no_reason(
    stage_hparams: dict,
    *,
    force_reasoning_prompt: bool,
    force_no_reason_prompt: bool,
) -> bool:
    """Resolve the prompt contract without silently guessing for metadata-free models."""

    if force_reasoning_prompt and force_no_reason_prompt:
        raise ValueError(
            "--force_reasoning_prompt and --force_no_reason_prompt are mutually exclusive."
        )
    if force_reasoning_prompt:
        return False
    if force_no_reason_prompt:
        return True
    if not isinstance(stage_hparams, dict):
        stage_hparams = {}
    prompt_variant_field = (
        "evaluation_no_reason"
        if "evaluation_no_reason" in stage_hparams
        else "no_reason"
    )
    if prompt_variant_field in stage_hparams:
        raw_no_reason = stage_hparams[prompt_variant_field]
        if isinstance(raw_no_reason, bool):
            return raw_no_reason
        if isinstance(raw_no_reason, int) and raw_no_reason in {0, 1}:
            return bool(raw_no_reason)
        normalized_no_reason = str(raw_no_reason).strip().lower()
        if normalized_no_reason in {"1", "true", "yes", "on"}:
            return True
        if normalized_no_reason in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(
            f"Checkpoint metadata contains an invalid {prompt_variant_field} value: "
            f"{raw_no_reason!r}. Use an explicit prompt override only after verifying the "
            "training contract."
        )
    template = str(stage_hparams.get("prompt_template", "") or "").strip().lower()
    if template:
        return template.endswith("_no_reason")
    raise RuntimeError(
        "Checkpoint metadata does not declare whether training used a reasoning or "
        "no-reason prompt. Evaluation refuses to guess because this changes the model "
        "contract. Re-run with exactly one of --force_reasoning_prompt or "
        "--force_no_reason_prompt, or evaluate from a run directory containing "
        f"{HYPERPARAMS_FILENAME}."
    )


def _stage_prompt_style(stage_hparams: dict) -> str:
    if not isinstance(stage_hparams, dict):
        return "auto"
    explicit_style = str(stage_hparams.get("prompt_style", "")).strip().lower()
    if explicit_style in {"alpaca", "alpaca_chat", "jiang_f638"}:
        return explicit_style
    template = str(stage_hparams.get("prompt_template", "")).strip().lower()
    if template.startswith("alpaca_chat"):
        return "alpaca_chat"
    if template.startswith("alpaca"):
        return "alpaca"
    if template.startswith("jiang_f638"):
        return "jiang_f638"
    return "auto"


def _print_grouped_final_summary(tasks: list[str], per_task: dict[str, dict]) -> None:
    if not tasks:
        return

    print_primary("")
    print_primary("=" * 80)
    print_primary("[Eval][FinalSummary]")
    for task in tasks:
        summary = per_task.get(task)
        if not isinstance(summary, dict):
            continue

        print_primary("")
        print_primary(task.upper())
        print_primary(f"available_examples: {_format_metric(summary.get('available_examples'))}")
        print_primary(f"requested_examples: {_format_metric(summary.get('requested_examples'))}")
        print_primary(f"loaded_examples: {_format_metric(summary.get('loaded_examples'))}")
        print_primary(
            "mean_best_gap: "
            f"{_format_metric(summary.get('mean_best_feasible_gap'))} +- "
            f"{_format_metric(summary.get('std_best_feasible_gap'))}"
        )
        print_primary(f"median_best_gap: {_format_metric(summary.get('median_best_feasible_gap'))}")
        print_primary(
            "candidate_feasibility_rate: "
            f"{_format_metric(summary.get('candidate_feasibility_rate'))}"
        )
        print_primary(
            "best_of_n_instance_feasibility_rate: "
            f"{_format_metric(summary.get('best_of_n_instance_feasibility_rate'))}"
        )
        print_primary(
            "mean_feasible_gap_std_in_batch: "
            f"{_format_metric(summary.get('mean_feasible_gap_std_in_batch'))} +- "
            f"{_format_metric(summary.get('std_feasible_gap_std_in_batch'))}"
        )
        print_primary(
            "mean_time_sec: "
            f"{_format_metric(summary.get('mean_generation_time_sec'))} +- "
            f"{_format_metric(summary.get('std_generation_time_sec'))}"
        )
        print_primary(
            "mean_time_sec_per_candidate: "
            f"{_format_metric(summary.get('mean_generation_time_sec_per_candidate'))} +- "
            f"{_format_metric(summary.get('std_generation_time_sec_per_candidate'))}"
        )
    print_primary("=" * 80)
    print_primary("")


def run_eval(args) -> str:
    if env_world_size() > 1:
        raise RuntimeError(
            "clasp.entrypoints.evaluate currently supports single-process evaluation only."
        )

    print_primary("[Eval] Resolving checkpoint/model path...")
    explicit_or_latest_model_path, run_dir = _resolve_model_and_run_dir(args)
    checkpoint_name = _resolve_checkpoint_name(explicit_or_latest_model_path, run_dir)
    model_stage = _stage_label(args.stage)
    save_results = bool(args.save_results)
    resume_enabled = bool(getattr(args, "resume", True))
    args.data_root = _resolve_existing_data_root(args)

    stage_hparams = _load_stage_hyperparams(run_dir)
    restored_representation_env = _restore_stage_representation_env(stage_hparams)
    prompt_no_reason = _resolve_prompt_no_reason(
        stage_hparams,
        force_reasoning_prompt=bool(getattr(args, "force_reasoning_prompt", False)),
        force_no_reason_prompt=bool(getattr(args, "force_no_reason_prompt", False)),
    )
    stage_prompt_style = _stage_prompt_style(stage_hparams)
    requested_prompt_style = normalize_prompt_style(args.prompt_style)
    if requested_prompt_style == "auto":
        requested_prompt_style = stage_prompt_style
    max_new_tokens = _resolve_max_new_tokens(args, stage_hparams)
    do_sample = args.do_sample if args.do_sample is not None else int(args.num_return_sequences) > 1
    if int(args.num_return_sequences) > 1 and not do_sample:
        raise ValueError("num_return_sequences > 1 requires sampling. Use --do_sample.")
    generation_batch_size = min(int(args.generation_batch_size), int(args.num_return_sequences))
    temperature_schedule = _parse_temperature_schedule(
        args.temperature_schedule,
        int(args.num_return_sequences),
    )
    top_p_schedule = _parse_top_p_schedule(
        args.top_p_schedule,
        int(args.num_return_sequences),
    )
    op_size_band_top_p_schedule = _parse_top_p_schedule(
        os.environ.get("CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE", ""),
        int(args.num_return_sequences),
    )
    op_size_band_top_p_min_size = _op_size_band_top_p_min_size()
    op_size_band_top_p_max_size = _op_size_band_top_p_max_size()
    if temperature_schedule and not do_sample:
        raise ValueError("--temperature_schedule requires sampling. Use --do_sample.")
    if top_p_schedule and not do_sample:
        raise ValueError("--top_p_schedule requires sampling. Use --do_sample.")
    if op_size_band_top_p_schedule and not do_sample:
        raise ValueError("CLASP_OP_SIZE_BAND_TOP_P_SCHEDULE requires sampling. Use --do_sample.")
    tsp_label_tta_schedule = tsp_label_tta_schedule_from_env()
    if tsp_label_tta_schedule and len(tsp_label_tta_schedule) != int(args.num_return_sequences):
        raise ValueError(
            "CLASP_TSP_LABEL_TTA_SCHEDULE must contain exactly "
            f"--num_return_sequences entries ({args.num_return_sequences}); got "
            f"{len(tsp_label_tta_schedule)}."
        )
    print_primary(f"[Eval] Stage: {args.stage} ({model_stage})")
    print_primary(f"[Eval] Resolved model path: {explicit_or_latest_model_path}")
    print_primary(f"[Eval] Checkpoint label: {checkpoint_name}")
    if run_dir:
        print_primary(f"[Eval] Resolved run directory: {run_dir}")
    if restored_representation_env:
        print_primary(
            "[Eval] Restored task representation from training metadata: "
            + ", ".join(
                f"{name}={value!r}" for name, value in sorted(restored_representation_env.items())
            )
        )
    print_primary(f"[Eval] Data root: {args.data_root}")

    resolved_tokenizer_path = resolve_tokenizer_path(
        explicit_or_latest_model_path, args.tokenizer_path
    )
    is_confirmation = str(getattr(args, "evaluation_purpose", "development")) == "confirmation"
    (
        training_run_contract,
        training_run_contract_sha256,
        training_run_contract_identity,
    ) = _training_run_contract_identity(
        model_path=explicit_or_latest_model_path,
        run_dir=run_dir,
        required=bool(is_confirmation and str(args.stage) == "grpo"),
    )
    if is_confirmation and training_run_contract is not None:
        contracted_task = str(training_run_contract["task"])
        if list(args.tasks) != [contracted_task]:
            raise RuntimeError(
                "Confirmation must evaluate exactly the specialist task declared by "
                f"run_contract.json: expected [{contracted_task!r}], got {list(args.tasks)!r}."
            )
    publication_contract = (
        isinstance(training_run_contract, dict)
        and training_run_contract.get("schema_version") == 2
        and isinstance(training_run_contract.get("training_config"), dict)
        and training_run_contract["training_config"].get("checkpoint_fidelity")
        == "publication_exact_required"
    )
    publication_checkpoint_required = is_confirmation or publication_contract
    publication_checkpoint_artifact_before_load = None
    publication_checkpoint_fidelity_before_load = None
    publication_exact_runtime_before_load = None
    publication_exact_post_step_before_load = None
    if publication_checkpoint_required:
        contracted_tokenizer_path = resolve_tokenizer_path(explicit_or_latest_model_path, None)
        if _tokenizer_content_identity(resolved_tokenizer_path) != _tokenizer_content_identity(
            contracted_tokenizer_path
        ):
            raise RuntimeError(
                "Publication tokenizer bytes differ from the checkpoint/run tokenizer; "
                "post-selection tokenizer overrides are forbidden."
            )
        print_primary("[Eval] Hashing publication checkpoint before model load...")
        publication_checkpoint_artifact_before_load = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        print_primary("[Eval] Live-verifying publication checkpoint before model load...")
        try:
            publication_checkpoint_fidelity_before_load = publication_fidelity_identity(
                explicit_or_latest_model_path
            )
        except ValueError as exc:
            raise RuntimeError(f"Publication checkpoint is not publication-exact: {exc}") from exc
        checkpoint_artifact_after_preload_fidelity = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        if (
            checkpoint_artifact_after_preload_fidelity
            != publication_checkpoint_artifact_before_load
        ):
            raise RuntimeError(
                "Publication checkpoint changed during pre-load fidelity verification"
            )
        if not isinstance(training_run_contract, dict):
            raise RuntimeError("Publication checkpoint has no immutable training run contract")
        print_primary("[Eval] Live-verifying exact PEFT KL runtime chronology...")
        publication_exact_runtime_before_load = _publication_exact_runtime_identity(
            model_path=explicit_or_latest_model_path,
            run_dir=run_dir,
            training_run_contract=training_run_contract,
        )
        publication_exact_post_step_before_load = _publication_exact_post_step_identity(
            model_path=explicit_or_latest_model_path,
            run_dir=run_dir,
            training_run_contract=training_run_contract,
            training_run_contract_sha256=str(training_run_contract_sha256),
            exact_runtime=publication_exact_runtime_before_load,
        )
        checkpoint_artifact_after_runtime = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        if checkpoint_artifact_after_runtime != publication_checkpoint_artifact_before_load:
            raise RuntimeError(
                "Publication checkpoint changed during exact PEFT KL runtime verification"
            )
    print_primary("[Eval] Loading model and tokenizer...")
    model, tokenizer, _ = load_model_and_tokenizer(
        explicit_or_latest_model_path,
        args.dtype,
        padding_side="left",
        tokenizer_path=resolved_tokenizer_path,
    )
    model.eval()
    resolved_prompt_style = resolve_prompt_style(
        requested_prompt_style,
        model_path=explicit_or_latest_model_path,
        tokenizer=tokenizer,
    )
    strict_chat_prompt = requested_prompt_style == "alpaca_chat"
    prompt_template = (
        f"{resolved_prompt_style}_no_reason" if prompt_no_reason else resolved_prompt_style
    )
    print_primary(f"[Eval] Prompt template: {prompt_template}")
    print_primary(
        "[Eval] Prompt style (stage/default -> requested -> resolved): "
        f"{stage_prompt_style} -> {requested_prompt_style} -> {resolved_prompt_style}"
    )
    print_primary(f"[Eval] Tokenizer source: {resolved_tokenizer_path}")
    print_primary("[Eval] Model and tokenizer loaded.")

    publication_checkpoint_artifact = None
    publication_checkpoint_fidelity = None
    publication_exact_runtime = None
    publication_exact_post_step = None
    if publication_checkpoint_artifact_before_load is not None:
        print_primary("[Eval] Re-hashing publication checkpoint after model load...")
        publication_checkpoint_artifact = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        if publication_checkpoint_artifact != publication_checkpoint_artifact_before_load:
            raise RuntimeError(
                "Publication checkpoint changed while the model was loading; refusing "
                "to bind generated outputs to ambiguous model bytes."
            )
        print_primary("[Eval] Re-opening publication checkpoint fidelity after model load...")
        try:
            publication_checkpoint_fidelity = publication_fidelity_identity(
                explicit_or_latest_model_path
            )
        except ValueError as exc:
            raise RuntimeError(f"Publication checkpoint is not publication-exact: {exc}") from exc
        if publication_checkpoint_fidelity != publication_checkpoint_fidelity_before_load:
            raise RuntimeError(
                "Publication checkpoint fidelity changed while the model was loading"
            )
        checkpoint_artifact_after_fidelity = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        if checkpoint_artifact_after_fidelity != publication_checkpoint_artifact:
            raise RuntimeError("Publication checkpoint changed during live fidelity verification")
        if not isinstance(training_run_contract, dict):
            raise RuntimeError("Publication checkpoint has no immutable training run contract")
        print_primary("[Eval] Re-opening exact PEFT KL runtime after model load...")
        publication_exact_runtime = _publication_exact_runtime_identity(
            model_path=explicit_or_latest_model_path,
            run_dir=run_dir,
            training_run_contract=training_run_contract,
        )
        if publication_exact_runtime != publication_exact_runtime_before_load:
            raise RuntimeError(
                "Publication exact PEFT KL runtime changed while the model was loading"
            )
        publication_exact_post_step = _publication_exact_post_step_identity(
            model_path=explicit_or_latest_model_path,
            run_dir=run_dir,
            training_run_contract=training_run_contract,
            training_run_contract_sha256=str(training_run_contract_sha256),
            exact_runtime=publication_exact_runtime,
        )
        if publication_exact_post_step != publication_exact_post_step_before_load:
            raise RuntimeError(
                "Publication exact PEFT KL post-step lineage changed while the model loaded"
            )
        checkpoint_artifact_after_runtime = _reference_content_identity(
            explicit_or_latest_model_path,
            inference_artifacts_only=True,
        )
        if checkpoint_artifact_after_runtime != publication_checkpoint_artifact:
            raise RuntimeError(
                "Publication checkpoint changed during live exact-runtime verification"
            )

    context_window = infer_model_context_window(model, tokenizer)
    if context_window is not None and max_new_tokens >= int(context_window):
        old_value = max_new_tokens
        max_new_tokens = max(1, int(context_window) - 1)
        print_primary(
            "[Eval] Clamped max_new_tokens to fit the loaded model context window: "
            f"{old_value} -> {max_new_tokens} (context_window={context_window})"
        )
    prompt_limit = _limit_prompt_length(
        args.max_prompt_length,
        stage_hparams,
        context_window,
        max_new_tokens,
    )
    model_device = _infer_model_device(model)

    print_primary("[Eval] Fingerprinting model, data, prompt, decoder, verifier, and code...")
    task_json_files = {
        task: list_split_json_files(
            args.data_root,
            task,
            args.split,
            prefer_subset_files=args.prefer_subset_files,
            required=True,
        )
        for task in args.tasks
    }
    data_identity = _dataset_content_identity(
        data_root=args.data_root,
        task_json_files=task_json_files,
        split=args.split,
    )
    if str(getattr(args, "evaluation_purpose", "development")) == "confirmation":
        _validate_confirmation_dataset(
            manifest_path=str(args.benchmark_manifest),
            tasks=list(args.tasks),
            data_identity=data_identity,
            manifest_payload=getattr(args, "benchmark_manifest_payload", None),
        )
    prompting_contract = {
        "no_reason": bool(prompt_no_reason),
        "prompt_template": prompt_template,
        "prompt_style_stage_default": stage_prompt_style,
        "prompt_style_requested": requested_prompt_style,
        "prompt_style_resolved": resolved_prompt_style,
        "strict_chat_prompt": bool(strict_chat_prompt),
        "op_strict_budget_hint": _op_strict_budget_hint_enabled(),
        "op_prompt_budget_scale": float(_op_prompt_budget_scale()),
        "op_prompt_budget_scale_min_size": int(_op_prompt_budget_scale_min_size()),
        "op_prompt_budget_scale_max_size": int(_op_prompt_budget_scale_max_size()),
        "op_prompt_budget_scale_size_ranges": list(_op_prompt_budget_scale_size_ranges()),
        **active_tsp_representation_settings(),
    }
    generation_contract = {
        "seed": int(args.seed),
        "seed_policy": "per_example_sha256_v1",
        "dtype": args.dtype,
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": prompt_limit,
        "num_return_sequences": int(args.num_return_sequences),
        "generation_batch_size": int(generation_batch_size),
        "do_sample": bool(do_sample),
        "temperature": float(args.temperature),
        "temperature_schedule": list(temperature_schedule),
        "top_p": float(args.top_p),
        "top_p_schedule": list(top_p_schedule),
        "op_size_band_top_p_schedule": list(op_size_band_top_p_schedule),
        "op_size_band_top_p_min_size": int(op_size_band_top_p_min_size),
        "op_size_band_top_p_max_size": int(op_size_band_top_p_max_size),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "model_context_window": context_window,
    }
    selection_contract = {
        "evaluation_purpose": str(getattr(args, "evaluation_purpose", "development")),
        "benchmark_manifest": getattr(args, "benchmark_manifest_identity", None),
        "campaign_manifest": getattr(args, "campaign_manifest_identity", None),
        "checkpoint_selection_receipt": getattr(
            args, "checkpoint_selection_receipt_identity", None
        ),
        "tasks": list(args.tasks),
        "split": args.split,
        "max_eval_samples_per_task": int(args.max_eval_samples_per_task),
        "eval_start_index": int(args.eval_start_index),
        "eval_samples_per_task": dict(getattr(args, "eval_samples_per_task_map", {}) or {}),
        "size_bucket": str(getattr(args, "size_bucket", "all")),
        "min_instance_size": int(getattr(args, "min_instance_size", 0) or 0),
        "max_instance_size": int(getattr(args, "max_instance_size", 0) or 0),
        "prefer_subset_files": bool(args.prefer_subset_files),
    }
    checkpoint_selection_binding = None
    if is_confirmation and str(args.stage) == "grpo":
        receipt_payload = getattr(args, "checkpoint_selection_receipt_payload", None)
        if not isinstance(receipt_payload, dict) or training_run_contract is None:
            raise RuntimeError(
                "Confirmation checkpoint selection or training contract is unavailable"
            )
        checkpoint_selection_binding = _validate_confirmation_checkpoint_selection(
            receipt=receipt_payload,
            training_run_contract=training_run_contract,
            training_run_contract_sha256=str(training_run_contract_sha256),
            checkpoint_path=explicit_or_latest_model_path,
            checkpoint_artifact=publication_checkpoint_artifact,
            checkpoint_fidelity=publication_checkpoint_fidelity,
            exact_peft_kl_runtime=publication_exact_runtime,
            exact_peft_kl_post_step=publication_exact_post_step,
        )
        selection_contract["checkpoint_selection_binding"] = checkpoint_selection_binding
    campaign_run_binding = None
    if is_confirmation:
        campaign_payload = getattr(args, "campaign_manifest_payload", None)
        campaign_identity = getattr(args, "campaign_manifest_identity", None)
        benchmark_identity = getattr(args, "benchmark_manifest_identity", None)
        receipt_identity = getattr(args, "checkpoint_selection_receipt_identity", None)
        receipt_payload = getattr(args, "checkpoint_selection_receipt_payload", None)
        if (
            not isinstance(campaign_payload, dict)
            or not isinstance(campaign_identity, dict)
            or not isinstance(benchmark_identity, dict)
            or not isinstance(receipt_identity, dict)
            or not isinstance(receipt_payload, dict)
            or training_run_contract is None
            or training_run_contract_sha256 is None
            or checkpoint_selection_binding is None
        ):
            raise RuntimeError(
                "Confirmation requires a campaign-bound GRPO training contract and "
                "checkpoint-selection decision"
            )
        campaign_run_binding = _validate_confirmation_campaign_binding(
            manifest=campaign_payload,
            manifest_identity=campaign_identity,
            benchmark_identity=benchmark_identity,
            selection_receipt_identity=receipt_identity,
            selection_receipt_payload=receipt_payload,
            training_run_contract=training_run_contract,
            training_run_contract_sha256=str(training_run_contract_sha256),
            checkpoint_selection_binding=checkpoint_selection_binding,
            decode_seed=int(args.seed),
        )
        selection_contract["campaign_run_binding"] = campaign_run_binding
    experiment_provenance, experiment_fingerprint, protocol_fingerprint = (
        _build_experiment_provenance(
            model_path=explicit_or_latest_model_path,
            tokenizer_path=resolved_tokenizer_path,
            model=model,
            tokenizer=tokenizer,
            data_identity=data_identity,
            prompting=prompting_contract,
            generation=generation_contract,
            selection=selection_contract,
            training_run_contract=training_run_contract,
            training_run_contract_sha256=training_run_contract_sha256,
            checkpoint_artifact=publication_checkpoint_artifact,
            checkpoint_fidelity=publication_checkpoint_fidelity,
            exact_peft_kl_runtime=publication_exact_runtime,
            exact_peft_kl_post_step=publication_exact_post_step,
        )
    )
    results_label = _results_label(
        args,
        checkpoint_name,
        experiment_fingerprint=experiment_fingerprint,
    )
    inference_root = _resolve_inference_root(args) if save_results else None
    results_dir = _default_results_dir(args, checkpoint_name) if save_results else None
    confirmation_attempt_claim_path = None
    _confirmation_attempt_lock = None
    best_run_scope = _best_run_scope_label(
        args,
        do_sample=bool(do_sample),
        protocol_fingerprint=protocol_fingerprint,
    )
    print_primary(f"[Eval] Experiment fingerprint: {experiment_fingerprint}")
    print_primary(f"[Eval] Protocol fingerprint: {protocol_fingerprint}")
    print_primary(
        f"[Eval] Evaluation purpose: {getattr(args, 'evaluation_purpose', 'development')}"
    )
    if save_results:
        print_primary(f"[Eval] Inference root: {inference_root}")
        print_primary(f"[Eval] Results directory: {results_dir}")
    else:
        print_primary("[Eval] Result saving disabled (--no-save_results).")
    if bool(getattr(args, "update_best_run", False)):
        print_primary(f"[Eval] Mutable development best-run scope: {best_run_scope}")
    else:
        print_primary("[Eval] Mutable best-run selection: disabled")

    print_primary(
        "[Eval] Generation config: "
        f"max_new_tokens={max_new_tokens}, max_prompt_length={prompt_limit}, "
        f"num_return_sequences={args.num_return_sequences}, "
        f"generation_batch_size={generation_batch_size}, do_sample={do_sample}, "
        f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}"
    )
    if _op_strict_budget_hint_enabled():
        print_primary("[Eval] OP strict budget prompt hint: enabled")
    op_prompt_budget_scale = _op_prompt_budget_scale()
    if abs(op_prompt_budget_scale - 1.0) > 1e-12:
        print_primary(
            f"[Eval] OP prompt budget scale: {op_prompt_budget_scale:g} "
            "(generation prompt only; evaluation uses original instance budget)"
        )
        op_prompt_budget_scale_min_size = _op_prompt_budget_scale_min_size()
        if op_prompt_budget_scale_min_size > 0:
            print_primary(
                f"[Eval] OP prompt budget scale applies only to size >= "
                f"{op_prompt_budget_scale_min_size}"
            )
        op_prompt_budget_scale_max_size = _op_prompt_budget_scale_max_size()
        if op_prompt_budget_scale_max_size > 0:
            print_primary(
                f"[Eval] OP prompt budget scale applies only to size <= "
                f"{op_prompt_budget_scale_max_size}"
            )
        op_prompt_budget_scale_size_ranges = _op_prompt_budget_scale_size_ranges()
        if op_prompt_budget_scale_size_ranges:
            print_primary(
                "[Eval] OP prompt budget scale applies only to size ranges: "
                + ",".join(f"{lo}-{hi}" for lo, hi in op_prompt_budget_scale_size_ranges)
            )
    if temperature_schedule:
        print_primary(
            "[Eval] Temperature schedule: "
            + ",".join(f"{value:g}" for value in temperature_schedule)
        )
    if top_p_schedule:
        print_primary(
            "[Eval] Top-p schedule: " + ",".join(f"{value:g}" for value in top_p_schedule)
        )
    if op_size_band_top_p_schedule:
        print_primary(
            "[Eval] OP size-band top-p schedule: "
            + ",".join(f"{value:g}" for value in op_size_band_top_p_schedule)
        )
        if op_size_band_top_p_min_size > 0:
            print_primary(
                f"[Eval] OP size-band top-p schedule applies only to size >= "
                f"{op_size_band_top_p_min_size}"
            )
        if op_size_band_top_p_max_size > 0:
            print_primary(
                f"[Eval] OP size-band top-p schedule applies only to size <= "
                f"{op_size_band_top_p_max_size}"
            )
    if tsp_label_tta_schedule:
        print_primary("[Eval] TSP label-TTA schedule: " + ",".join(tsp_label_tta_schedule))
    if int(args.num_return_sequences) > generation_batch_size:
        print_primary(
            "[Eval] Best-of-N generation will be chunked to reduce peak memory: "
            f"{args.num_return_sequences} return sequences in batches of {generation_batch_size}."
        )
    if getattr(args, "eval_samples_per_task_map", None):
        print_primary(
            f"[Eval] Per-task eval caps: {dict(sorted(args.eval_samples_per_task_map.items()))}"
        )

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("Evaluation requires torch to be installed.") from exc
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    print_primary(f"[Eval] Sampling seed: {int(args.seed)} (deterministic per-example derivation)")

    all_records: list[dict] = []
    details_artifact: dict[str, Any] | None = None
    per_task = {}
    csv_by_task = {}
    task_dataset_info = {}
    generated_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_complete = False
    details_path = (
        os.path.join(results_dir, f"run_details__{results_label}.jsonl")
        if save_results and args.save_details and results_dir is not None
        else None
    )
    run_config_output_path = (
        os.path.join(results_dir, f"run_config__{results_label}.json")
        if save_results and results_dir is not None
        else None
    )
    summary_output_path = (
        os.path.join(results_dir, f"run_summary__{results_label}.json")
        if save_results and results_dir is not None
        else None
    )
    if is_confirmation:
        campaign_payload = getattr(args, "campaign_manifest_payload", None)
        attempt_ledger = (
            campaign_payload.get("confirmation_attempt_ledger")
            if isinstance(campaign_payload, dict)
            else None
        )
        if (
            results_dir is None
            or run_config_output_path is None
            or details_path is None
            or campaign_run_binding is None
            or not isinstance(attempt_ledger, dict)
        ):
            raise RuntimeError("Confirmation has no immutable output or campaign binding")
        (
            confirmation_attempt_claim_path,
            _confirmation_attempt_lock,
        ) = _claim_confirmation_attempt(
            attempt_ledger_root=str(attempt_ledger["canonical_root"]),
            campaign_binding=campaign_run_binding,
            experiment_fingerprint=experiment_fingerprint,
            results_dir=results_dir,
            required_resume_artifacts=(run_config_output_path, details_path),
        )

    def _current_run_config_payload(best_run_dir: str | None = None) -> dict:
        return {
            "stage": args.stage,
            "evaluation_purpose": str(getattr(args, "evaluation_purpose", "development")),
            "benchmark_manifest": getattr(args, "benchmark_manifest_identity", None),
            "campaign_manifest": getattr(args, "campaign_manifest_identity", None),
            "campaign_run_binding": campaign_run_binding,
            "confirmation_attempt_claim": (
                {
                    "basename": os.path.basename(confirmation_attempt_claim_path),
                    "confirmation_attempt_id": campaign_run_binding["confirmation_attempt_id"],
                }
                if confirmation_attempt_claim_path is not None and campaign_run_binding is not None
                else None
            ),
            "best_run_update_requested": bool(getattr(args, "update_best_run", False)),
            "stage_dirname": _stage_dirname(args.stage),
            "model_stage": model_stage,
            "checkpoint_name": checkpoint_name,
            "run_label": results_label,
            "run_status": "complete" if run_complete else "in_progress",
            "experiment_fingerprint": experiment_fingerprint,
            "experiment_fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "experiment_provenance": experiment_provenance,
            "training_run_contract": training_run_contract,
            "training_run_contract_sha256": training_run_contract_sha256,
            "training_run_contract_identity": training_run_contract_identity,
            "checkpoint_selection_binding": checkpoint_selection_binding,
            "code_identity": experiment_provenance["code"],
            "verifier_identity": experiment_provenance["verifier"],
            "generated_at_utc": generated_at_utc,
            "resolved_model_path": explicit_or_latest_model_path,
            "resolved_run_dir": run_dir,
            "resolved_hyperparams_path": (
                os.path.join(run_dir, HYPERPARAMS_FILENAME)
                if run_dir and os.path.isfile(os.path.join(run_dir, HYPERPARAMS_FILENAME))
                else None
            ),
            "base_model_name": resolve_base_model_name(explicit_or_latest_model_path),
            "data_root": args.data_root,
            "inference_root": inference_root,
            "results_dir": results_dir,
            "best_run_scope": best_run_scope,
            "tasks": list(args.tasks),
            "split": args.split,
            "generation": {
                **generation_contract,
                "best_of_n": int(args.num_return_sequences),
            },
            "prompting": {
                **prompting_contract,
                "tokenizer_path": resolved_tokenizer_path,
            },
            "limits": {
                **selection_contract,
                "save_details": bool(args.save_details),
                "is_print": bool(args.is_print),
                "resume": bool(resume_enabled),
            },
            "per_task_dataset_info": task_dataset_info,
            "details_artifact": details_artifact,
            "files": {
                "run_config": run_config_output_path,
                "summary": summary_output_path,
                "details": details_path,
                "csv_by_task": csv_by_task,
                "best_run_dir": best_run_dir,
            },
        }

    def _current_summary_payload(best_run_dir: str | None = None) -> dict:
        return {
            "stage": args.stage,
            "evaluation_purpose": str(getattr(args, "evaluation_purpose", "development")),
            "benchmark_manifest": getattr(args, "benchmark_manifest_identity", None),
            "campaign_manifest": getattr(args, "campaign_manifest_identity", None),
            "campaign_run_binding": campaign_run_binding,
            "confirmation_attempt_claim": (
                {
                    "basename": os.path.basename(confirmation_attempt_claim_path),
                    "confirmation_attempt_id": campaign_run_binding["confirmation_attempt_id"],
                }
                if confirmation_attempt_claim_path is not None and campaign_run_binding is not None
                else None
            ),
            "best_run_update_requested": bool(getattr(args, "update_best_run", False)),
            "stage_dirname": _stage_dirname(args.stage),
            "model_stage": model_stage,
            "checkpoint_name": checkpoint_name,
            "run_label": results_label,
            "run_status": "complete" if run_complete else "in_progress",
            "experiment_fingerprint": experiment_fingerprint,
            "experiment_fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "experiment_provenance": experiment_provenance,
            "training_run_contract": training_run_contract,
            "training_run_contract_sha256": training_run_contract_sha256,
            "training_run_contract_identity": training_run_contract_identity,
            "checkpoint_selection_binding": checkpoint_selection_binding,
            "code_identity": experiment_provenance["code"],
            "verifier_identity": experiment_provenance["verifier"],
            "generated_at_utc": generated_at_utc,
            "tasks": list(args.tasks),
            "split": args.split,
            "resolved_model_path": explicit_or_latest_model_path,
            "resolved_run_dir": run_dir,
            "resolved_hyperparams_path": (
                os.path.join(run_dir, HYPERPARAMS_FILENAME)
                if run_dir and os.path.isfile(os.path.join(run_dir, HYPERPARAMS_FILENAME))
                else None
            ),
            "base_model_name": resolve_base_model_name(explicit_or_latest_model_path),
            "data_root": args.data_root,
            "inference_root": inference_root,
            "results_dir": results_dir,
            "best_run_scope": best_run_scope,
            "generation": dict(generation_contract),
            "prompting": {
                **prompting_contract,
                "tokenizer_path": resolved_tokenizer_path,
            },
            "limits": {
                **selection_contract,
                "resume": bool(resume_enabled),
            },
            "per_task_dataset_info": task_dataset_info,
            "overall": _aggregate_records(all_records),
            "per_task": per_task,
            "details_artifact": details_artifact,
            "files": {
                "run_config": run_config_output_path,
                "summary": summary_output_path,
                "details": details_path,
                "csv_by_task": csv_by_task,
                "best_run_dir": best_run_dir,
            },
        }

    existing_detail_records: list[dict] = []
    resume_record_by_key: dict[tuple[str, str, int], dict] = {}
    if save_results and results_dir is not None:
        existing_config = None
        existing_config_file = bool(
            run_config_output_path and os.path.isfile(run_config_output_path)
        )
        existing_details_file = bool(details_path and os.path.isfile(details_path))
        if resume_enabled and (existing_config_file or existing_details_file):
            existing_config = _validate_resume_run_config(
                _json_load(run_config_output_path),
                experiment_fingerprint=experiment_fingerprint,
                run_config_path=str(run_config_output_path),
            )
            generated_at_utc = str(
                existing_config.get("generated_at_utc", generated_at_utc) or generated_at_utc
            )
        if details_path and resume_enabled and existing_details_file:
            existing_detail_records = _read_detail_records_for_resume(details_path)
            resume_record_by_key = _resume_records_by_key(
                existing_detail_records,
                args,
                experiment_fingerprint=experiment_fingerprint,
            )
            _rewrite_detail_records_atomic(details_path, existing_detail_records)
            if resume_record_by_key:
                print_primary(
                    "[Eval][Resume] Found "
                    f"{len(resume_record_by_key)} reusable completed detail record(s): {details_path}"
                )
        elif details_path:
            _rewrite_detail_records_atomic(details_path, [])

    config_path = None
    summary_path = None
    if save_results and results_dir is not None:
        config_path = _write_run_config(results_dir, results_label, _current_run_config_payload())
        summary_path = _write_summary(results_dir, results_label, _current_summary_payload())
        print_primary(f"[Eval] Run config path: {config_path}")
        print_primary(f"[Eval] Summary path: {summary_path}")
        if details_path:
            print_primary(f"[Eval] Details path (streaming): {details_path}")

    with torch.inference_mode():
        for task in args.tasks:
            print_primary(f"[Eval] Loading dataset for task={task}, split={args.split}...")
            json_files = task_json_files[task]
            dataset = load_and_merge_task_datasets(json_files)
            source_desc = (
                json_files[0]
                if len(json_files) == 1
                else f"{len(json_files)} files under {os.path.dirname(json_files[0])}"
            )
            available_examples = len(dataset)
            requested_examples = _requested_eval_examples_for_task(args, task)
            eval_indices, size_selection_info, size_by_index = _select_eval_indices_for_task(
                dataset,
                args,
                task,
            )
            loaded_examples = len(eval_indices)

            print_primary(
                f"[Eval] {task}: source={source_desc}, "
                f"available_examples={available_examples}, requested_examples={requested_examples}, "
                f"loaded_examples={loaded_examples}"
            )
            if size_selection_info.get("size_filter_active") or size_selection_info.get(
                "all_size_summary", {}
            ).get("known_examples"):
                print_primary(
                    f"[Eval] {task}: size_bucket={size_selection_info['size_bucket']}, "
                    f"bucket_examples_before_cap={size_selection_info['bucket_examples_before_cap']}, "
                    f"all_size={size_selection_info['all_size_summary']}, "
                    f"selected_size={size_selection_info['selected_size_summary']}"
                )
            task_dataset_info[task] = {
                "source": source_desc,
                "json_files": list(json_files),
                "available_examples": int(available_examples),
                "requested_examples": int(requested_examples),
                "loaded_examples": int(loaded_examples),
                "size_selection": size_selection_info,
                "num_candidates_per_instance": int(args.num_return_sequences),
                "expected_candidates_total": int(loaded_examples * int(args.num_return_sequences)),
            }
            if save_results and results_dir is not None:
                config_path = _write_run_config(
                    results_dir, results_label, _current_run_config_payload()
                )

            task_records: list[dict] = []
            resumed_records_for_task = 0
            evaluator = TASK_EVALUATORS[task]
            for idx in _progress_iter(
                eval_indices,
                total=loaded_examples,
                desc=f"Eval {task}",
            ):
                row = dataset[idx]
                raw_instruction = str(row.get("instruction", ""))
                raw_input_text = str(row.get("input", ""))
                raw_answer = str(row.get("output", ""))
                raw_instance_json = row.get("instance_json")
                source_identity_row = {
                    "instruction": raw_instruction,
                    "input": raw_input_text,
                    "output": raw_answer,
                    "instance_json": raw_instance_json,
                }
                semantic_instance_digest = semantic_instance_sha256(task, source_identity_row)
                benchmark_row_digest = benchmark_row_sha256(source_identity_row)
                reference_output_digest = reference_output_sha256(raw_answer)
                instruction = raw_instruction
                transformed = transform_example_for_task(
                    task,
                    instruction,
                    raw_input_text,
                    raw_answer,
                    raw_instance_json,
                )
                instruction = str(transformed["instruction"])
                input_text = str(transformed["input"])
                answer = str(transformed["output"])
                instance_json = transformed["instance_json"]
                instruction, input_text = _apply_task_prompt_hints(
                    task,
                    instruction,
                    input_text,
                )
                example_sampling_seed = _example_sampling_seed(
                    int(args.seed),
                    str(task),
                    str(args.split),
                    int(idx),
                )
                resume_key = (str(task), str(args.split), int(idx))
                resumed_record = resume_record_by_key.get(resume_key)
                if resumed_record is not None:
                    if not _resume_record_matches_example(
                        resumed_record,
                        task=task,
                        split=args.split,
                        example_index=int(idx),
                        json_files=list(json_files),
                        instruction=instruction,
                        input_text=input_text,
                        answer=answer,
                        experiment_fingerprint=experiment_fingerprint,
                        data_manifest_sha256=data_identity["manifest_sha256"],
                        expected_sampling_seed=example_sampling_seed,
                        expected_semantic_instance_sha256=semantic_instance_digest,
                        expected_benchmark_row_sha256=benchmark_row_digest,
                        expected_reference_output_sha256=reference_output_digest,
                    ):
                        raise RuntimeError(
                            "Refusing to resume a detail record whose row content does not "
                            "match the fingerprinted dataset/prompt transformation: "
                            f"task={task}, split={args.split}, index={idx}. Existing artifacts "
                            "were left untouched."
                        )
                    task_records.append(resumed_record)
                    all_records.append(resumed_record)
                    resumed_records_for_task += 1
                    continue
                torch.manual_seed(example_sampling_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(example_sampling_seed)
                should_sync_cuda = bool(
                    torch.cuda.is_available()
                    and model_device is not None
                    and str(model_device).startswith("cuda")
                )
                if should_sync_cuda:
                    torch.cuda.reset_peak_memory_stats(model_device)
                use_tsp_label_tta = bool(tsp_label_tta_schedule and task == "tsp")
                example_top_p_schedule = top_p_schedule
                op_size_band_top_p_applied = False
                if str(task).lower() == "op" and op_size_band_top_p_schedule:
                    op_size = size_by_index.get(int(idx))
                    if op_size is None:
                        op_size = _op_instruction_size(instruction)
                    if _op_size_in_band(
                        op_size,
                        op_size_band_top_p_min_size,
                        op_size_band_top_p_max_size,
                    ):
                        example_top_p_schedule = op_size_band_top_p_schedule
                        op_size_band_top_p_applied = True
                candidate_payloads: list[dict[str, Any]] = []
                prompt_token_counts: list[int] = []
                prompt_was_truncated_values: list[bool] = []
                generation_time_sec = 0.0
                generated_sequences = 0
                remaining_sequences = int(args.num_return_sequences)
                while remaining_sequences > 0:
                    variant = "identity"
                    if use_tsp_label_tta:
                        variant = tsp_label_tta_schedule[generated_sequences]
                    current_temperature = float(args.temperature)
                    if temperature_schedule:
                        current_temperature = float(temperature_schedule[generated_sequences])
                    current_top_p = float(args.top_p)
                    if example_top_p_schedule:
                        current_top_p = float(example_top_p_schedule[generated_sequences])
                    if temperature_schedule or example_top_p_schedule:
                        same_temperature_span = 1
                        while (
                            same_temperature_span < int(generation_batch_size)
                            and same_temperature_span < remaining_sequences
                            and (
                                not temperature_schedule
                                or float(
                                    temperature_schedule[
                                        generated_sequences + same_temperature_span
                                    ]
                                )
                                == current_temperature
                            )
                            and (
                                not example_top_p_schedule
                                or float(
                                    example_top_p_schedule[
                                        generated_sequences + same_temperature_span
                                    ]
                                )
                                == current_top_p
                            )
                            and (
                                not use_tsp_label_tta
                                or tsp_label_tta_schedule[
                                    generated_sequences + same_temperature_span
                                ]
                                == variant
                            )
                        ):
                            same_temperature_span += 1
                        current_batch_size = same_temperature_span
                    else:
                        current_batch_size = 1
                        if use_tsp_label_tta:
                            while (
                                current_batch_size < int(generation_batch_size)
                                and current_batch_size < remaining_sequences
                                and tsp_label_tta_schedule[generated_sequences + current_batch_size]
                                == variant
                            ):
                                current_batch_size += 1
                        else:
                            current_batch_size = min(
                                int(generation_batch_size), remaining_sequences
                            )

                    if use_tsp_label_tta:
                        candidate_transformed = transform_tsp_example_for_label_tta_variant(
                            raw_instruction,
                            raw_input_text,
                            raw_answer,
                            raw_instance_json,
                            variant=variant,
                        )
                        candidate_instruction = str(candidate_transformed["instruction"])
                        candidate_input_text = str(candidate_transformed["input"])
                        candidate_answer = str(candidate_transformed["output"])
                        candidate_instance_json = candidate_transformed["instance_json"]
                    else:
                        candidate_instruction = instruction
                        candidate_input_text = input_text
                        candidate_answer = answer
                        candidate_instance_json = instance_json
                    candidate_instance = _parse_instance_value(
                        {"instance_json": candidate_instance_json}
                    )
                    candidate_prompt = build_formatted_prompt(
                        candidate_instruction,
                        candidate_input_text,
                        no_reason=prompt_no_reason,
                        prompt_style=resolved_prompt_style,
                        tokenizer=tokenizer,
                        strict_chat_prompt=strict_chat_prompt,
                    )
                    raw_prompt_token_count = len(tokenizer(candidate_prompt)["input_ids"])
                    if (
                        bool(stage_hparams.get("evaluation_requires_zero_prompt_truncation"))
                        and prompt_limit is not None
                        and raw_prompt_token_count > prompt_limit
                    ):
                        raise RuntimeError(
                            "Jiang-f638 evaluation prompt exceeds the live context budget; "
                            "source-exact evaluation refuses truncation: "
                            f"tokens={raw_prompt_token_count}, limit={prompt_limit}"
                        )
                    tokenized = tokenizer(
                        candidate_prompt,
                        return_tensors="pt",
                        truncation=prompt_limit is not None,
                        max_length=prompt_limit,
                    )
                    if model_device is not None:
                        tokenized = {
                            key: value.to(model_device) for key, value in tokenized.items()
                        }
                    prompt_length = int(tokenized["input_ids"].shape[1])
                    prompt_token_counts.append(prompt_length)
                    prompt_was_truncated_values.append(
                        bool(prompt_limit is not None and raw_prompt_token_count > prompt_limit)
                    )

                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "repetition_penalty": args.repetition_penalty,
                        "num_return_sequences": current_batch_size,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    if do_sample:
                        generate_kwargs.update(
                            {
                                "temperature": current_temperature,
                                "top_p": current_top_p,
                                "top_k": args.top_k,
                            }
                        )

                    if should_sync_cuda:
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    generated = model.generate(
                        **tokenized,
                        **generate_kwargs,
                    )
                    if should_sync_cuda:
                        torch.cuda.synchronize()
                    generation_time_sec += float(time.perf_counter() - start_time)

                    completion_ids = generated[:, prompt_length:]
                    decoded_completions = tokenizer.batch_decode(
                        completion_ids,
                        skip_special_tokens=True,
                    )
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        completion_token_counts = [
                            int(completion_ids.shape[1]) for _ in decoded_completions
                        ]
                    else:
                        completion_token_counts = [
                            int(value)
                            for value in (completion_ids != int(pad_token_id))
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                    for completion, completion_tokens in zip(
                        decoded_completions,
                        completion_token_counts,
                        strict=True,
                    ):
                        candidate_payloads.append(
                            {
                                "prompt": candidate_prompt,
                                "instruction": candidate_instruction,
                                "input_text": candidate_input_text,
                                "answer": candidate_answer,
                                "instance": candidate_instance,
                                "instance_json": candidate_instance_json,
                                "completion": completion,
                                "completion_tokens": completion_tokens,
                                "temperature": current_temperature,
                                "top_p": current_top_p,
                                "op_size_band_top_p_applied": op_size_band_top_p_applied,
                                "tsp_label_tta_variant": variant,
                            }
                        )
                    remaining_sequences -= current_batch_size
                    generated_sequences += current_batch_size
                    del generated
                    del completion_ids

                candidates = []
                for candidate_idx, payload in enumerate(candidate_payloads):
                    completion = str(payload["completion"])
                    metrics = evaluator(
                        payload["prompt"],
                        payload["input_text"],
                        payload["answer"],
                        payload["instance"],
                        completion,
                    )
                    metrics.update(
                        _constraint_diagnostics(
                            str(task).lower(),
                            prompt=str(payload["prompt"]),
                            input_text=str(payload["input_text"]),
                            answer=str(payload["answer"]),
                            instance=payload["instance"],
                            completion=completion,
                        )
                    )
                    metrics["candidate_index"] = candidate_idx
                    metrics["completion_tokens"] = int(payload["completion_tokens"])
                    metrics["decoding_temperature"] = payload["temperature"]
                    metrics["decoding_top_p"] = payload["top_p"]
                    if str(task).lower() == "op":
                        metrics["op_size_band_top_p_applied"] = bool(
                            payload.get("op_size_band_top_p_applied", False)
                        )
                    if use_tsp_label_tta:
                        metrics["tsp_label_tta_variant"] = payload["tsp_label_tta_variant"]
                        verification_context = {
                            "instruction": payload["instruction"],
                            "input_problem": payload["input_text"],
                            "ground_truth_output": payload["answer"],
                            "instance_json": payload["instance_json"],
                        }
                        metrics["verification_context"] = verification_context
                        metrics["verification_context_sha256"] = hashlib.sha256(
                            _canonical_json_bytes(verification_context)
                        ).hexdigest()
                    metrics["completion"] = completion
                    candidates.append(metrics)
                prompt_tokens = max(prompt_token_counts) if prompt_token_counts else 0
                peak_gpu_memory_allocated_bytes = None
                peak_gpu_memory_reserved_bytes = None
                if should_sync_cuda:
                    peak_gpu_memory_allocated_bytes = int(
                        torch.cuda.max_memory_allocated(model_device)
                    )
                    peak_gpu_memory_reserved_bytes = int(
                        torch.cuda.max_memory_reserved(model_device)
                    )

                record = _build_instance_record(
                    task=task,
                    split=args.split,
                    example_index=idx,
                    source_json=(json_files[0] if len(json_files) == 1 else None),
                    source_json_files=list(json_files),
                    instruction=instruction,
                    input_text=input_text,
                    answer=answer,
                    instance_size=size_by_index.get(int(idx)),
                    prompt_tokens=prompt_tokens,
                    prompt_was_truncated=any(prompt_was_truncated_values),
                    generation_time_sec=generation_time_sec,
                    num_return_sequences=int(args.num_return_sequences),
                    candidates=candidates,
                    peak_gpu_memory_allocated_bytes=peak_gpu_memory_allocated_bytes,
                    peak_gpu_memory_reserved_bytes=peak_gpu_memory_reserved_bytes,
                )
                record["instance_json"] = instance_json
                record["instance_sha256"] = hashlib.sha256(
                    _canonical_json_bytes(instance_json)
                ).hexdigest()
                record["semantic_instance_sha256"] = semantic_instance_digest
                record["benchmark_row_sha256"] = benchmark_row_digest
                record["reference_output_sha256"] = reference_output_digest
                record["transformed_example_sha256"] = hashlib.sha256(
                    _canonical_json_bytes(
                        {
                            "instruction": instruction,
                            "input": input_text,
                            "output": answer,
                            "instance_json": instance_json,
                        }
                    )
                ).hexdigest()
                record["source_row_locator"] = {
                    "data_manifest_sha256": data_identity["manifest_sha256"],
                    "task": str(task),
                    "split": str(args.split),
                    "merged_dataset_index": int(idx),
                }
                record["sampling_seed"] = example_sampling_seed
                record["experiment_fingerprint"] = experiment_fingerprint
                record["experiment_fingerprint_schema_version"] = (
                    EXPERIMENT_FINGERPRINT_SCHEMA_VERSION
                )
                record["protocol_fingerprint"] = protocol_fingerprint
                record["strict_verifier"] = experiment_provenance["verifier"]
                record["training_run_contract_sha256"] = training_run_contract_sha256
                if campaign_run_binding is not None:
                    record["campaign_manifest_sha256"] = campaign_run_binding[
                        "campaign_manifest_sha256"
                    ]
                    record["confirmation_attempt_id"] = campaign_run_binding[
                        "confirmation_attempt_id"
                    ]
                if str(task).lower() == "op":
                    record["op_size_band_top_p_applied"] = bool(op_size_band_top_p_applied)
                    record["effective_top_p_schedule"] = list(example_top_p_schedule)
                record = _self_hash_payload(record, "record_sha256")
                task_records.append(record)
                all_records.append(record)
                if details_path:
                    _append_detail_record(details_path, record)

                partial_task_summary = _aggregate_records(task_records)
                partial_task_summary["available_examples"] = int(available_examples)
                partial_task_summary["requested_examples"] = int(requested_examples)
                partial_task_summary["loaded_examples"] = int(loaded_examples)
                partial_task_summary["source"] = source_desc
                per_task[task] = partial_task_summary
                if save_results and results_dir is not None:
                    csv_path = _write_task_csv(
                        results_dir=results_dir,
                        results_label=results_label,
                        task=task,
                        model_stage=model_stage,
                        checkpoint_name=checkpoint_name,
                        model_path=explicit_or_latest_model_path,
                        split=args.split,
                        available_examples=available_examples,
                        requested_examples=requested_examples,
                        loaded_examples=loaded_examples,
                        records=task_records,
                        summary=partial_task_summary,
                    )
                    csv_by_task[task] = csv_path
                    summary_path = _write_summary(
                        results_dir, results_label, _current_summary_payload()
                    )
                    config_path = _write_run_config(
                        results_dir, results_label, _current_run_config_payload()
                    )

                if args.is_print:
                    _print_verbose_instance(record)

            print_primary(f"[Eval] Finished task={task}. Aggregating results...")
            if resumed_records_for_task:
                print_primary(
                    "[Eval][Resume] "
                    f"task={task} reused {resumed_records_for_task}/{loaded_examples} completed instance(s)."
                )
            task_summary = _aggregate_records(task_records)
            task_summary["available_examples"] = int(available_examples)
            task_summary["requested_examples"] = int(requested_examples)
            task_summary["loaded_examples"] = int(loaded_examples)
            task_summary["source"] = source_desc
            per_task[task] = task_summary

            if save_results and results_dir is not None:
                csv_path = _write_task_csv(
                    results_dir=results_dir,
                    results_label=results_label,
                    task=task,
                    model_stage=model_stage,
                    checkpoint_name=checkpoint_name,
                    model_path=explicit_or_latest_model_path,
                    split=args.split,
                    available_examples=available_examples,
                    requested_examples=requested_examples,
                    loaded_examples=loaded_examples,
                    records=task_records,
                    summary=task_summary,
                )
                csv_by_task[task] = csv_path
                print_primary(f"[Eval] CSV written to: {csv_path}")
            _print_task_summary(
                task,
                task_summary,
                available_examples=available_examples,
                requested_examples=requested_examples,
                loaded_examples=loaded_examples,
            )
            if save_results and results_dir is not None:
                summary_path = _write_summary(
                    results_dir, results_label, _current_summary_payload()
                )
                config_path = _write_run_config(
                    results_dir, results_label, _current_run_config_payload()
                )

    run_complete = True
    best_run_dir = None
    if save_results and results_dir is not None:
        if details_path:
            _rewrite_detail_records_atomic(details_path, all_records)
            details_artifact = _details_artifact_identity(
                details_path,
                record_count=len(all_records),
            )
        summary = _current_summary_payload()
        run_config = _current_run_config_payload()
        summary_path = _write_summary(results_dir, results_label, summary)
        config_path = _write_run_config(results_dir, results_label, run_config)
        if bool(getattr(args, "update_best_run", False)):
            best_run_dir = _copy_run_artifacts_to_best_run(
                args=args,
                scope_label=best_run_scope,
                run_summary=summary,
                run_summary_path=summary_path,
                run_config=run_config,
                run_config_path=config_path,
                details_path=details_path,
                csv_by_task=csv_by_task,
            )
        run_config = _current_run_config_payload(best_run_dir)
        summary = _current_summary_payload(best_run_dir)
        config_path = _write_run_config(results_dir, results_label, run_config)
        summary_path = _write_summary(results_dir, results_label, summary)

        if best_run_dir:
            print_primary(f"[Eval] Best-run snapshot updated: {best_run_dir}")
        elif bool(getattr(args, "update_best_run", False)):
            print_primary("[Eval] Best-run snapshot unchanged for this evaluation scope.")
        else:
            print_primary("[Eval] Best-run mutation disabled; use the frozen development selector.")

        print_primary(f"[Eval] Summary written to: {summary_path}")
        print_primary(f"[Eval] Run config written to: {config_path}")
        if details_path:
            print_primary(f"[Eval] Details written to: {details_path}")
    else:
        summary = _current_summary_payload()
        run_config = _current_run_config_payload()
        print_primary("[Eval] Result saving disabled; no files were written.")

    _print_grouped_final_summary(list(args.tasks), per_task)

    del model
    del tokenizer
    cleanup_torch_memory()
    return summary_path or ""


def main() -> None:
    args = parse_eval_args()
    run_eval(args)


if __name__ == "__main__":
    main()
    (validate_exact_peft_kl_post_step_lineage,)
