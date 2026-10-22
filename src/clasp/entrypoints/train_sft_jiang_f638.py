"""Fresh, serialization-safe reproduction of Jiang et al.'s f638 SFT recipe.

This entrypoint deliberately targets the dependency/API surface used by
LLMCoSolver commit ``f6388991d243c0aa91d9ee7e2fa18ed1c9e72477``.  It keeps
the released Unsloth ``FastLanguageModel`` path, single physical dataset
shuffle, and completion-only objective. Corrections are intentional and fail
closed:

* fresh jobs call ``trainer.train()`` because Transformers 4.50.3 rejects the
  released unconditional ``resume_from_checkpoint=True`` in an empty output;
  continuation jobs accept only a complete checkpoint at one of the released
  save boundaries and pass its canonical path explicitly;
* the released legacy ``SFTTrainer`` call shape is retained.  Importing
  Unsloth first installs its compatibility wrapper, which promotes the four
  SFT-only keywords into ``SFTConfig`` and maps ``tokenizer`` to
  ``processing_class``; the resulting effective config is audited before the
  first optimizer step;
* ``bias="all"`` is used because Qwen2.5-7B's only base biases are the 84
  targeted Q/K/V biases; those biases are explicitly re-enabled after
  Unsloth's post-patch freezes them, and PEFT 0.17.0 serializes them;
* response masking anchors on the contextual indented header without its
  colon, because the released standalone marker IDs never occur in the exact
  Qwen-formatted text and silently trigger full-example fallback loss.

Chunk stops are operational boundaries only: Trainer's horizon remains the
fixed 31,250 steps.  A run either satisfies the fixed recipe below or does not
run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from clasp.baselines.llmcosolver_f638_public_data import iter_json_array
from clasp.data.prompts import (
    JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY,
    JIANG_F638_RESPONSE_HEADER,
    JIANG_F638_SFT_TEMPLATE,
    JIANG_F638_SUPERVISED_BOUNDARY,
    JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
    build_jiang_f638_sft_text,
    jiang_f638_response_header_token_ids,
    locate_jiang_f638_supervised_start,
)
from clasp.models.jiang_f638_base import audit_jiang_f638_base_snapshot

UPSTREAM_REPOSITORY = "https://github.com/Summer142857/LLMCoSolver"
UPSTREAM_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
UPSTREAM_TREE = "4017f8b4c78bf057eeadc3f5b0e2ee1c0000a77d"
UPSTREAM_MODEL_ID = "unsloth/Qwen2.5-7B"
UPSTREAM_MODEL_REVISION = "86e6e1eccea424c7fb77d59f7d2a9125c0673bfd"

TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
# The released standalone marker does not occur in the exact Qwen tokenization.
# Anchor on the contextual, indented header and supervise from its next token;
# that token can span the colon and the first output subword.
RESPONSE_MARKER = JIANG_F638_RESPONSE_HEADER
REQUIRED_TEXT_FIELDS = ("instruction", "input", "output")
ALLOWED_FRESH_RUN_FILES = frozenset({"base_sft_run_contract.json", "run_contract.json"})

MAX_SEQ_LENGTH = 20_000
EXPECTED_TRAIN_ROWS = 500_000
TARGET_STEP = 31_250
DATASET_SHUFFLE_SEED = 42
MODEL_RANDOM_STATE = 42
TRAINER_SEED = 42
LORA_R = 64
LORA_ALPHA = 64
LORA_BIAS = "all"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_TRAIN_BATCH_SIZE = 16
WARMUP_STEPS = 20
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
LOGGING_STEPS = 1
SAVE_STEPS = 5_000
SAVE_TOTAL_LIMIT = 50
DATASET_NUM_PROC = 16
EXPECTED_TRAINER_CHECKPOINT_STEPS = (
    *range(SAVE_STEPS, TARGET_STEP, SAVE_STEPS),
    TARGET_STEP,
)
ALLOWED_CHUNK_STOP_STEPS = EXPECTED_TRAINER_CHECKPOINT_STEPS
REQUIRED_CHUNK_CHECKPOINT_FILES = frozenset(
    {
        "README.md",
        "added_tokens.json",
        "adapter_config.json",
        "checkpoint_fidelity.json",
        "merges.txt",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "vocab.json",
    }
)
CHUNK_ROOT_FILES = frozenset(
    {
        *ALLOWED_FRESH_RUN_FILES,
        "checkpoints",
        "hyperparams.json",
    }
)
TERMINAL_ROOT_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "checkpoint_fidelity.json",
        "stage_status.json",
        "sft_reload_equivalence.json",
    }
)

# This projection is byte-for-value compatible with
# scripts.prepare_sota_run.MATCHED_SFT_STAGE_HPARAMETERS. Keep the complete
# contract here so the standalone trainer cannot silently omit a fixed field.
MATCHED_SFT_STAGE_HPARAMETERS: Mapping[str, Any] = {
    "recipe": "jiang-llmcosolver-f638-corrected-v1",
    "jiang_source_commit": UPSTREAM_COMMIT,
    "model_loader": "unsloth.FastLanguageModel",
    "load_in_4bit": False,
    "full_finetune": False,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.0,
    "lora_bias": LORA_BIAS,
    "explicit_bias_trainability_restore_after_unsloth": True,
    "lora_target_modules": list(LORA_TARGET_MODULES),
    "use_gradient_checkpointing": "unsloth",
    "use_rslora": False,
    "loftq_config": None,
    "train_lm_head": False,
    "train_embed_tokens": False,
    "seed": TRAINER_SEED,
    "dataset_shuffle_seed": DATASET_SHUFFLE_SEED,
    "dataset_shuffle_passes": 1,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_train_batch_size": EFFECTIVE_TRAIN_BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optim": OPTIMIZER,
    "weight_decay": WEIGHT_DECAY,
    "warmup_steps": WARMUP_STEPS,
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "max_seq_length": MAX_SEQ_LENGTH,
    "packing": False,
    "min_train_samples_per_task": EXPECTED_TRAIN_ROWS,
    "deduplicate_train_rows": False,
    "no_reason": True,
    "prompt_style": "jiang_f638",
    "released_response_template": JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
    "response_boundary_prefix": JIANG_F638_RESPONSE_HEADER,
    "response_boundary_policy": "supervise_colon_output_crossing_token",
    "append_eos_token": True,
    "completion_only_loss": True,
    "memory_safe_loss": False,
    "supervised_logits_only": False,
    "evaluation_max_new_tokens": 3_000,
    "evaluation_max_prompt_length": 128_072,
    "evaluation_requires_zero_prompt_truncation": True,
    "evaluation_no_reason": True,
    "clasp_tsp_prompt_knn": "0",
    "clasp_tsp_prompt_knn_distances": "",
    "clasp_tsp_input_order": "",
    "clasp_tsp_relabel_order": "",
    "clasp_tsp_route_canonicalize": "",
    "clasp_tsp_route_only_output": "",
    "clasp_tsp_label_tta_schedule": "",
    "clasp_mis_relabel_order": "",
    "clasp_mis_jiang_response_output": "0",
    "clasp_mis_prompt_node_degrees": "0",
    "clasp_mis_prompt_low_degree_k": "32",
}

FIXED_RECORDED_CLI_ARGS: Mapping[str, Any] = {
    "prompt_style": "jiang_f638",
    "no_reason": True,
    "dtype": "bfloat16",
    "eval_ratio": 0.0,
    "eval_from_train_split": False,
    "max_samples_per_task": EXPECTED_TRAIN_ROWS,
    "max_eval_samples_per_task": 0,
    "min_train_samples_per_task": EXPECTED_TRAIN_ROWS,
    "min_eval_samples_per_task": 0,
    "prefer_subset_files": False,
    "deduplicate_train_rows": False,
    "logging_steps": LOGGING_STEPS,
    "save_steps": SAVE_STEPS,
    "eval_steps": None,
    "save_total_limit": SAVE_TOTAL_LIMIT,
}

# Exact versions in the official repository's requirements.txt.  Enforcing
# these before importing Unsloth is important: importing it mutates TRL's
# trainer behavior, and newer TRL exposes a materially different SFT API.
PINNED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "accelerate": "1.7.0",
    "bitsandbytes": "0.46.1",
    "datasets": "3.6.0",
    "peft": "0.17.0",
    "safetensors": "0.6.1",
    "torch": "2.7.1",
    "transformers": "4.50.3",
    "trl": "0.15.2",
    "unsloth": "2025.3.19",
    "unsloth_zoo": "2025.3.17",
    "wandb": "0.18.0",
    "xformers": "0.0.31.post1",
}

UPSTREAM_FILE_SHA256: Mapping[str, str] = {
    "main_train.py": "a37901e49fcea759f9fb960e67a9c38a6f73aad4da8f79df222aa8e1bde84f13",
    "requirements.txt": "b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6",
    "utils.py": "88b5c9c06fad5025e8d3caf6ed24d0a0b8f97dacefc537b3a60d27c0bd512db1",
}

PROTOCOL = "clasp-jiang-llmcosolver-f638-sft-corrected-v2"


class JiangF638SFTError(RuntimeError):
    """Raised when the fixed reproduction contract cannot be satisfied."""


def audit_runtime_versions(
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Require the exact package versions published with f638."""

    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for distribution, expected in PINNED_RUNTIME_VERSIONS.items():
        try:
            actual = str(version_getter(distribution))
        except importlib.metadata.PackageNotFoundError:
            actual = "<missing>"
        except Exception as exc:
            raise JiangF638SFTError(
                f"cannot identify runtime distribution {distribution!r}: {exc}"
            ) from exc
        observed[distribution] = actual
        if actual != expected:
            mismatches.append(f"{distribution}={actual!r} (expected {expected!r})")
    if mismatches:
        raise JiangF638SFTError(
            "Jiang-f638 runtime version mismatch: " + "; ".join(mismatches)
        )
    return observed


def _marker_offsets(input_ids: Sequence[int], marker_ids: Sequence[int]) -> list[int]:
    if not marker_ids:
        raise JiangF638SFTError("response marker tokenization is empty")
    width = len(marker_ids)
    return [
        index
        for index in range(len(input_ids) - width + 1)
        if list(input_ids[index : index + width]) == list(marker_ids)
    ]


def _as_token_rows(value: Any, expected_rows: int) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise JiangF638SFTError("tokenizer did not return input_ids as a list")
    if expected_rows == 1 and (not value or isinstance(value[0], int)):
        value = [value]
    if len(value) != expected_rows:
        raise JiangF638SFTError(
            f"tokenizer returned {len(value)} rows for a batch of {expected_rows}"
        )
    rows: list[list[int]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list) or any(type(token) is not int for token in row):
            raise JiangF638SFTError("tokenizer returned a non-integer input_ids row")
        rows.append(row)
    return rows


def _batch_text(dataset: Any, start: int, stop: int) -> list[str]:
    batch = dataset[start:stop]
    if isinstance(batch, Mapping):
        values = batch.get("text")
    elif isinstance(batch, list):
        values = [row.get("text") if isinstance(row, Mapping) else None for row in batch]
    else:
        values = None
    if not isinstance(values, list) or len(values) != stop - start:
        raise JiangF638SFTError("formatted dataset slicing did not return a text batch")
    if any(not isinstance(value, str) for value in values):
        raise JiangF638SFTError("formatted dataset contains a non-string text field")
    return values


def audit_formatted_dataset(
    dataset: Any,
    tokenizer: Any,
    *,
    expected_rows: int = EXPECTED_TRAIN_ROWS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Audit every row before Trainer tokenization, including marker survival.

    The official collator silently fell back to full-example LM loss when its
    marker was absent.  Here every raw and tokenized row must contain exactly
    one marker, the marker must have a non-empty supervised suffix, and no row
    may rely on right truncation at 20k.
    """

    row_count = len(dataset)
    if row_count != expected_rows:
        raise JiangF638SFTError(
            f"training corpus has {row_count} rows; expected exactly {expected_rows}"
        )
    if batch_size <= 0:
        raise JiangF638SFTError("marker audit batch size must be positive")
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise JiangF638SFTError("tokenizer has no non-empty EOS token")
    try:
        marker_ids = jiang_f638_response_header_token_ids(tokenizer)
    except ValueError as exc:
        raise JiangF638SFTError(f"cannot tokenize corrected response header: {exc}") from exc

    maximum_tokens = 0
    minimum_completion_tokens: int | None = None
    for start in range(0, row_count, batch_size):
        stop = min(row_count, start + batch_size)
        texts = _batch_text(dataset, start, stop)
        for offset, text in enumerate(texts):
            index = start + offset
            if not text.endswith(eos_token):
                raise JiangF638SFTError(f"row {index} does not end in the tokenizer EOS token")
        try:
            encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
            raw_input_ids = encoded["input_ids"]
        except Exception as exc:
            raise JiangF638SFTError(
                f"cannot tokenize marker-audit rows {start}:{stop}: {exc}"
            ) from exc
        token_rows = _as_token_rows(raw_input_ids, len(texts))
        for offset, input_ids in enumerate(token_rows):
            index = start + offset
            token_count = len(input_ids)
            maximum_tokens = max(maximum_tokens, token_count)
            if token_count > max_seq_length:
                raise JiangF638SFTError(
                    f"row {index} has {token_count} tokens and would truncate at {max_seq_length}"
                )
            try:
                completion_start = locate_jiang_f638_supervised_start(
                    texts[offset],
                    input_ids,
                    marker_ids,
                )
            except ValueError as exc:
                raise JiangF638SFTError(
                    f"row {index} has an invalid response boundary: {exc}"
                ) from exc
            completion_tokens = token_count - completion_start
            minimum_completion_tokens = (
                completion_tokens
                if minimum_completion_tokens is None
                else min(minimum_completion_tokens, completion_tokens)
            )
    return {
        "status": "all_rows_exactly_one_contextual_header_no_truncation",
        "rows_checked": row_count,
        "max_seq_length": max_seq_length,
        "maximum_observed_tokens": maximum_tokens,
        "minimum_completion_tokens": minimum_completion_tokens,
        "response_header": RESPONSE_MARKER,
        "response_header_ids": marker_ids,
        "upstream_nonmatching_template": JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
        "supervised_boundary": JIANG_F638_SUPERVISED_BOUNDARY,
        "first_supervised_token_policy": JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY,
        "raw_header_mismatches": 0,
        "tokenized_header_mismatches": 0,
        "truncated_rows": 0,
        "empty_completion_rows": 0,
    }


def _format_batch(examples: Mapping[str, Any], *, eos_token: str) -> dict[str, list[str]]:
    columns: dict[str, list[Any]] = {}
    for field in REQUIRED_TEXT_FIELDS:
        values = examples.get(field)
        if not isinstance(values, list):
            raise JiangF638SFTError(f"dataset batch has no list-valued {field!r} column")
        columns[field] = values
    widths = {len(values) for values in columns.values()}
    if len(widths) != 1:
        raise JiangF638SFTError("dataset text columns have inconsistent batch lengths")
    texts: list[str] = []
    for row_index, values in enumerate(zip(*(columns[field] for field in REQUIRED_TEXT_FIELDS))):
        for field, value in zip(REQUIRED_TEXT_FIELDS, values):
            if not isinstance(value, str) or not value.strip():
                raise JiangF638SFTError(
                    f"dataset batch row {row_index} has invalid required field {field!r}"
                )
        instruction, input_text, output = values
        texts.append(
            build_jiang_f638_sft_text(
                instruction,
                input_text,
                output,
                eos_token=eos_token,
            )
        )
    return {"text": texts}


def prepare_training_dataset(
    dataset: Any,
    tokenizer: Any,
    *,
    expected_rows: int = EXPECTED_TRAIN_ROWS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> tuple[Any, dict[str, Any]]:
    """Perform f638's one explicit shuffle, format, then audit all rows."""

    if len(dataset) != expected_rows:
        raise JiangF638SFTError(
            f"training corpus has {len(dataset)} rows; expected exactly {expected_rows}"
        )
    columns = set(getattr(dataset, "column_names", ()) or ())
    missing = sorted(set(REQUIRED_TEXT_FIELDS) - columns)
    if missing:
        raise JiangF638SFTError(f"training corpus is missing required columns: {missing}")
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise JiangF638SFTError("tokenizer has no non-empty EOS token")

    # This is the sole physical Dataset.shuffle in the entrypoint.  Trainer's
    # seeded sampler remains unchanged, as in the released program.
    shuffled = dataset.shuffle(seed=DATASET_SHUFFLE_SEED)
    formatted = shuffled.map(
        _format_batch,
        batched=True,
        fn_kwargs={"eos_token": eos_token},
        desc="Formatting exact Jiang-f638 SFT text",
    )
    audit = audit_formatted_dataset(
        formatted,
        tokenizer,
        expected_rows=expected_rows,
        max_seq_length=max_seq_length,
    )
    fingerprint = getattr(formatted, "_fingerprint", None)
    if isinstance(fingerprint, str) and fingerprint:
        audit["formatted_dataset_fingerprint"] = fingerprint
    return formatted, audit


def _strict_mask_batch(batch: Mapping[str, Any], marker_ids: Sequence[int]) -> Mapping[str, Any]:
    input_matrix = batch.get("input_ids")
    labels = batch.get("labels")
    if input_matrix is None or labels is None:
        raise JiangF638SFTError("language-model collator returned no input_ids or labels")
    batch_size = len(input_matrix)
    for index in range(batch_size):
        input_row = input_matrix[index]
        if hasattr(input_row, "tolist"):
            input_row = input_row.tolist()
        offsets = _marker_offsets(input_row, marker_ids)
        if len(offsets) != 1:
            raise JiangF638SFTError(
                f"collated row {index} has {len(offsets)} contextual response headers; "
                "refusing fallback loss"
            )
        response_start = offsets[0] + len(marker_ids)
        label_row = labels[index]
        comparable = label_row.tolist() if hasattr(label_row, "tolist") else list(label_row)
        if response_start >= len(comparable) or not any(
            token != -100 for token in comparable[response_start:]
        ):
            raise JiangF638SFTError(
                f"collated row {index} has no supervised completion token"
            )
        try:
            labels[index, :response_start] = -100
        except (IndexError, TypeError):
            try:
                label_row[:response_start] = [-100] * response_start
            except Exception as exc:
                raise JiangF638SFTError(
                    f"cannot apply response-only mask to collated row {index}: {exc}"
                ) from exc
    return batch


def build_strict_response_collator(
    tokenizer: Any,
    *,
    base_collator_class: type[Any] | None = None,
) -> Any:
    """Build the old f638 LM collator with a fail-closed marker policy."""

    if base_collator_class is None:
        from transformers import DataCollatorForLanguageModeling

        base_collator_class = DataCollatorForLanguageModeling
    try:
        marker_ids = jiang_f638_response_header_token_ids(tokenizer)
    except ValueError as exc:
        raise JiangF638SFTError(f"cannot tokenize corrected response header: {exc}") from exc

    class StrictResponseOnlyCollator(base_collator_class):
        def __init__(self) -> None:
            super().__init__(tokenizer=tokenizer, mlm=False)
            self.response_header = RESPONSE_MARKER
            self.response_header_ids = tuple(marker_ids)

        def torch_call(self, examples: Any) -> Any:
            batch = super().torch_call(examples)
            return _strict_mask_batch(batch, self.response_header_ids)

    return StrictResponseOnlyCollator()


def assert_fresh_output(output_dir: Path) -> Path:
    """Allow an absent run dir or only immutable pre-training contract files."""

    raw = output_dir.expanduser().absolute()
    if raw.is_symlink():
        raise JiangF638SFTError(f"output directory must not be a symlink: {raw}")
    if not raw.exists():
        return raw
    if not raw.is_dir():
        raise JiangF638SFTError(f"output path is not a directory: {raw}")
    unexpected = sorted(
        entry.name for entry in raw.iterdir() if entry.name not in ALLOWED_FRESH_RUN_FILES
    )
    if unexpected:
        raise JiangF638SFTError(
            "fresh Jiang-f638 training refuses existing run artifacts: " + ", ".join(unexpected)
        )
    for entry in raw.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise JiangF638SFTError(f"pre-existing run contract is not a regular file: {entry}")
        if entry.stat().st_mode & 0o222:
            raise JiangF638SFTError(f"pre-existing run contract must be read-only: {entry}")
    return raw.resolve(strict=True)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise JiangF638SFTError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JiangF638SFTError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise JiangF638SFTError(f"{label} is not a JSON object: {path}")
    return value


def _validated_chunk_stop_step(value: Any) -> int:
    if type(value) is not int or value not in ALLOWED_CHUNK_STOP_STEPS:
        raise JiangF638SFTError(
            "--stop-after-step must be one of "
            + ",".join(str(step) for step in ALLOWED_CHUNK_STOP_STEPS)
        )
    return value


def _checkpoint_step(checkpoint: Path) -> int:
    prefix = "checkpoint-"
    if not checkpoint.name.startswith(prefix):
        raise JiangF638SFTError(
            f"resume checkpoint has an invalid directory name: {checkpoint.name!r}"
        )
    suffix = checkpoint.name[len(prefix) :]
    if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0"):
        raise JiangF638SFTError(
            f"resume checkpoint has an invalid directory name: {checkpoint.name!r}"
        )
    step = int(suffix)
    if step not in ALLOWED_CHUNK_STOP_STEPS:
        raise JiangF638SFTError(
            f"resume checkpoint step {step} is not an allowed Jiang-f638 save boundary"
        )
    return step


def _require_chunk_checkpoint(checkpoint: Path, *, expected_step: int) -> Path:
    raw = checkpoint.expanduser()
    if not raw.is_absolute():
        raise JiangF638SFTError(f"resume checkpoint path must be absolute: {raw}")
    raw = raw.absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise JiangF638SFTError(f"resume checkpoint is not a regular directory: {raw}")
    canonical = raw.resolve(strict=True)
    if canonical != raw:
        raise JiangF638SFTError(
            f"resume checkpoint path must be canonical: {raw} != {canonical}"
        )
    step = _checkpoint_step(canonical)
    if step != expected_step:
        raise JiangF638SFTError(
            f"resume checkpoint step {step} differs from expected step {expected_step}"
        )
    missing = sorted(
        name for name in REQUIRED_CHUNK_CHECKPOINT_FILES if not (canonical / name).is_file()
    )
    if missing:
        raise JiangF638SFTError(
            f"resume checkpoint {canonical.name} is incomplete: {missing}"
        )
    model_files = [
        canonical / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (canonical / name).is_file() and not (canonical / name).is_symlink()
    ]
    if len(model_files) != 1:
        raise JiangF638SFTError(
            f"resume checkpoint {canonical.name} must contain exactly one adapter model"
        )
    for name in REQUIRED_CHUNK_CHECKPOINT_FILES:
        candidate = canonical / name
        if candidate.is_symlink() or not candidate.is_file():
            raise JiangF638SFTError(
                f"resume checkpoint contains an unsafe required artifact: {candidate}"
            )
    trainer_state = _load_json_object(
        canonical / "trainer_state.json",
        label="resume Trainer state",
    )
    if (
        int(trainer_state.get("global_step", -1)) != step
        or int(trainer_state.get("max_steps", -1)) != TARGET_STEP
    ):
        raise JiangF638SFTError(
            "resume Trainer state does not preserve the fixed 31,250-step horizon"
        )
    return canonical


def resolve_chunk_execution(
    output_dir: Path,
    *,
    stop_after_step: int,
    resume_from_checkpoint: Path | None,
) -> tuple[Path, Path | None, int]:
    """Resolve a fresh or exact-prefix continuation before importing Unsloth."""

    stop_step = _validated_chunk_stop_step(stop_after_step)
    if resume_from_checkpoint is None:
        return assert_fresh_output(output_dir), None, 0

    output = _canonical_existing_directory(output_dir, label="resume output directory")
    checkpoint_root = output / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise JiangF638SFTError(
            f"resume output has no regular checkpoint directory: {checkpoint_root}"
        )
    resume = _require_chunk_checkpoint(
        resume_from_checkpoint,
        expected_step=_checkpoint_step(resume_from_checkpoint),
    )
    expected_resume = checkpoint_root / resume.name
    if resume != expected_resume:
        raise JiangF638SFTError(
            f"resume checkpoint is outside the requested run: {resume} != {expected_resume}"
        )
    resume_step = _checkpoint_step(resume)
    if resume_step >= stop_step:
        raise JiangF638SFTError(
            f"resume step {resume_step} must be strictly before stop step {stop_step}"
        )
    root_entries = {entry.name for entry in output.iterdir()}
    unexpected_root = sorted(root_entries - CHUNK_ROOT_FILES)
    if unexpected_root:
        raise JiangF638SFTError(
            "resume output contains terminal or unexpected root artifacts: "
            + ", ".join(unexpected_root)
        )
    missing_root = sorted({"checkpoints", "hyperparams.json"} - root_entries)
    if missing_root:
        raise JiangF638SFTError(
            "resume output is missing required chunk artifacts: " + ", ".join(missing_root)
        )
    if any((output / name).exists() or (output / name).is_symlink() for name in TERMINAL_ROOT_FILES):
        raise JiangF638SFTError("a terminalized run cannot be resumed")

    expected_names = {
        f"checkpoint-{step}"
        for step in ALLOWED_CHUNK_STOP_STEPS
        if step <= resume_step
    }
    observed_names = {entry.name for entry in checkpoint_root.iterdir()}
    if observed_names != expected_names:
        raise JiangF638SFTError(
            "resume checkpoint inventory is not the exact saved prefix: "
            f"observed={sorted(observed_names)}, expected={sorted(expected_names)}"
        )
    for step in ALLOWED_CHUNK_STOP_STEPS:
        if step > resume_step:
            break
        _require_chunk_checkpoint(
            checkpoint_root / f"checkpoint-{step}",
            expected_step=step,
        )
    return output, resume, resume_step


def _write_json_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True, default=str)
            handle.write("\n")
    except FileExistsError as exc:
        raise JiangF638SFTError(f"refusing to replace run artifact: {path}") from exc


def write_or_validate_hyperparams(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if not resume:
        _write_json_create_once(path, payload)
        return
    observed = _load_json_object(path, label="existing chunk hyperparameters")
    if observed != dict(payload):
        raise JiangF638SFTError(
            "resume invocation differs from the immutable fresh-chunk hyperparameters"
        )


def write_final_stage_status(
    output_dir: Path,
    trainer_state: Any,
) -> dict[str, Any]:
    """Bind terminal state to Trainer's forced step-31,250 checkpoint."""

    global_step = int(getattr(trainer_state, "global_step", -1))
    max_steps = int(getattr(trainer_state, "max_steps", -1))
    if global_step != TARGET_STEP or max_steps != TARGET_STEP:
        raise JiangF638SFTError(
            f"cannot finalize unexpected trainer state: global={global_step}, max={max_steps}"
        )
    checkpoint = output_dir / "checkpoints" / f"checkpoint-{TARGET_STEP}"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise JiangF638SFTError(f"latest periodic checkpoint is unavailable: {checkpoint}")
    required_checkpoint = {
        "adapter_config.json",
        "checkpoint_fidelity.json",
        "trainer_state.json",
    }
    missing_checkpoint = sorted(
        name for name in required_checkpoint if not (checkpoint / name).is_file()
    )
    if missing_checkpoint:
        raise JiangF638SFTError(
            f"latest periodic checkpoint is incomplete: {missing_checkpoint}"
        )
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise JiangF638SFTError("latest periodic checkpoint has no adapter weights")
    if not (output_dir / "adapter_config.json").is_file() or not any(
        (output_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise JiangF638SFTError("terminal RUN_DIR adapter is incomplete")
    if not (output_dir / "checkpoint_fidelity.json").is_file():
        raise JiangF638SFTError("terminal RUN_DIR adapter has no checkpoint-fidelity receipt")
    epoch_value = getattr(trainer_state, "epoch", None)
    epoch = float(epoch_value) if epoch_value is not None else None
    payload = {
        "stage": "sft",
        "completed": True,
        "global_step": TARGET_STEP,
        "max_steps": TARGET_STEP,
        "epoch": epoch,
        "has_saved_model": True,
        "latest_checkpoint_dir": str(checkpoint.resolve(strict=True)),
    }
    _write_json_create_once(output_dir / "stage_status.json", payload)
    return payload


def _unwrap_trainer_model(trainer: Any) -> Any:
    model = trainer.model
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    return unwrap(model) if callable(unwrap) else model


def build_checkpoint_fidelity_callback(
    trainer: Any,
    *,
    trainer_callback_class: type[Any] | None = None,
    auditor: Callable[..., Mapping[str, Any]] | None = None,
) -> Any:
    """Attach the existing exact adapter-closure receipt to every checkpoint."""

    if trainer_callback_class is None:
        from transformers import TrainerCallback

        trainer_callback_class = TrainerCallback
    if auditor is None:
        from clasp.models.checkpoint_fidelity import audit_and_write_checkpoint_fidelity

        auditor = audit_and_write_checkpoint_fidelity

    class JiangCheckpointFidelityCallback(trainer_callback_class):
        def __init__(self, owner: Any) -> None:
            super().__init__()
            self._trainer = owner

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if not bool(getattr(state, "is_world_process_zero", True)):
                return control
            step = int(getattr(state, "global_step", -1))
            checkpoint = Path(str(args.output_dir)) / f"checkpoint-{step}"
            if step <= 0 or not checkpoint.is_dir():
                raise JiangF638SFTError(
                    f"Trainer save callback cannot resolve checkpoint at step {step}: {checkpoint}"
                )
            auditor(
                _unwrap_trainer_model(self._trainer),
                checkpoint,
                require_qwen25_7b_publication=True,
            )
            return control

    return JiangCheckpointFidelityCallback(trainer)


def build_chunk_stop_callback(
    stop_after_step: int,
    *,
    trainer_callback_class: type[Any] | None = None,
) -> Any:
    """Stop only from ``on_save`` after the exact target checkpoint is complete."""

    target = _validated_chunk_stop_step(stop_after_step)
    if trainer_callback_class is None:
        from transformers import TrainerCallback

        trainer_callback_class = TrainerCallback

    class JiangChunkStopCallback(trainer_callback_class):
        def __init__(self) -> None:
            super().__init__()
            self.saved_target = False

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(getattr(state, "global_step", -1))
            horizon = int(getattr(state, "max_steps", -1))
            if horizon != TARGET_STEP:
                raise JiangF638SFTError(
                    f"chunk callback observed max_steps={horizon}, expected {TARGET_STEP}"
                )
            if step > target:
                raise JiangF638SFTError(
                    f"Trainer saved step {step} after requested chunk boundary {target}"
                )
            if step != target:
                return control
            checkpoint = Path(str(args.output_dir)) / f"checkpoint-{target}"
            _require_chunk_checkpoint(checkpoint, expected_step=target)
            self.saved_target = True
            control.should_training_stop = True
            return control

    return JiangChunkStopCallback()


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise JiangF638SFTError(f"{label} is not a regular directory: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise JiangF638SFTError(f"{label} path must be canonical: {raw} != {resolved}")
    return resolved


def resolve_training_source(data_root: Path, task: str) -> tuple[Path, Path]:
    """Accept either the corpus root or the task's concrete ``train`` directory."""

    canonical_root = _canonical_existing_directory(data_root, label="SFT data root")
    nested = canonical_root / task / "train"
    if nested.is_dir() and not nested.is_symlink():
        return canonical_root, nested.resolve(strict=True)
    direct_files = [
        entry
        for entry in canonical_root.iterdir()
        if entry.is_file()
        and not entry.is_symlink()
        and entry.suffix == ".json"
    ]
    if canonical_root.name == "train" and direct_files:
        return canonical_root, canonical_root
    raise JiangF638SFTError(
        f"SFT data root contains neither {task}/train nor a direct train dataset: {canonical_root}"
    )


def resolve_training_json_file(train_source: Path) -> Path:
    """Resolve the one JSON array whose text columns Jiang-f638 consumes."""

    candidates = sorted(
        (
            entry
            for entry in train_source.iterdir()
            if entry.suffix == ".json" and entry.is_file() and not entry.is_symlink()
        ),
        key=lambda entry: entry.name,
    )
    if len(candidates) != 1:
        raise JiangF638SFTError(
            f"training source must contain exactly one regular .json file, found "
            f"{len(candidates)}: {train_source}"
        )
    raw = candidates[0].absolute()
    canonical = raw.resolve(strict=True)
    if canonical != raw:
        raise JiangF638SFTError(
            f"training JSON path must be canonical: supplied={raw}, resolved={canonical}"
        )
    return canonical


def _projected_training_rows(
    training_file: str,
    *,
    expected_rows: int,
    expected_identity: tuple[int, ...],
) -> Any:
    """Yield only exact training strings from a bounded-memory JSON parser.

    ``instance_json`` is useful corpus metadata but is not consumed by the
    Jiang-f638 SFT objective. CVRP and OP encode that metadata as a
    heterogeneous list (nested lists plus a scalar), which Arrow cannot infer
    as one column type. Project before Arrow inference while preserving the
    three strings byte-for-value.
    """

    path = Path(training_file)
    def observed_identity() -> tuple[int, ...]:
        if path.is_symlink() or not path.is_file():
            raise JiangF638SFTError(f"training JSON is not a regular file: {path}")
        value = path.stat(follow_symlinks=False)
        return tuple(
            int(getattr(value, field))
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )

    if observed_identity() != expected_identity:
        raise JiangF638SFTError(f"training JSON identity changed before projection: {path}")

    raw_digest = hashlib.sha256()
    row_count = 0
    try:
        rows = iter_json_array(path, raw_digest)
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise JiangF638SFTError(
                    f"training JSON row {row_index} is not an object: {path}"
                )
            projected: dict[str, str] = {}
            for field in REQUIRED_TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise JiangF638SFTError(
                        f"training JSON row {row_index} has invalid required field {field!r}"
                    )
                projected[field] = value
            yield projected
            row_count += 1
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, JiangF638SFTError):
            raise
        raise JiangF638SFTError(f"cannot stream training JSON {path}: {exc}") from exc

    if row_count != expected_rows:
        raise JiangF638SFTError(
            f"training JSON has {row_count} rows; expected exactly {expected_rows}: {path}"
        )
    if observed_identity() != expected_identity:
        raise JiangF638SFTError(f"training JSON identity changed during projection: {path}")


def load_projected_training_dataset(
    train_source: Path,
    *,
    cache_dir: Path | None,
    expected_rows: int = EXPECTED_TRAIN_ROWS,
    dataset_class: Any = None,
    features_class: Any = None,
    value_class: Any = None,
) -> Any:
    """Materialize the exact three-column SFT view with an explicit schema."""

    if dataset_class is None or features_class is None or value_class is None:
        from datasets import Dataset, Features, Value

        dataset_class = Dataset
        features_class = Features
        value_class = Value

    training_file = resolve_training_json_file(train_source)
    source_stat = training_file.stat(follow_symlinks=False)
    source_identity = tuple(
        int(getattr(source_stat, field))
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )
    features = features_class(
        {field: value_class("string") for field in REQUIRED_TEXT_FIELDS}
    )
    dataset = dataset_class.from_generator(
        _projected_training_rows,
        features=features,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        keep_in_memory=False,
        num_proc=1,
        gen_kwargs={
            "training_file": str(training_file),
            "expected_rows": expected_rows,
            "expected_identity": source_identity,
        },
    )
    if len(dataset) != expected_rows:
        raise JiangF638SFTError(
            f"projected training corpus has {len(dataset)} rows; expected {expected_rows}"
        )
    observed_columns = tuple(getattr(dataset, "column_names", ()) or ())
    if observed_columns != REQUIRED_TEXT_FIELDS:
        raise JiangF638SFTError(
            "projected training corpus has unexpected columns/order: "
            f"{observed_columns!r}"
        )
    return dataset


def _audit_single_h100(
    torch_module: Any,
    is_bfloat16_supported: Callable[[], bool],
) -> dict[str, Any]:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    except ValueError as exc:
        raise JiangF638SFTError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise JiangF638SFTError(f"Jiang-f638 SFT requires WORLD_SIZE=1, got {world_size}")
    if not torch_module.cuda.is_available() or int(torch_module.cuda.device_count()) != 1:
        raise JiangF638SFTError("Jiang-f638 SFT requires exactly one visible CUDA accelerator")
    name = str(torch_module.cuda.get_device_name(0))
    if "H100" not in name.upper():
        raise JiangF638SFTError(f"Jiang-f638 SFT requires an NVIDIA H100, got {name!r}")
    if not bool(is_bfloat16_supported()):
        raise JiangF638SFTError("the visible H100 does not report bfloat16 support")
    return {"world_size": 1, "device_count": 1, "device_name": name, "dtype": "bfloat16"}


def _trainer_config_payload(training_args: Any) -> dict[str, Any]:
    to_dict = getattr(training_args, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(vars(training_args))


def audit_effective_sft_config(training_args: Any) -> dict[str, Any]:
    """Require the exact effective config produced by Unsloth's f638 wrapper."""

    def observed(name: str, *aliases: str) -> Any:
        for candidate in (name, *aliases):
            if hasattr(training_args, candidate):
                value = getattr(training_args, candidate)
                return getattr(value, "value", value)
        raise JiangF638SFTError(f"effective SFT config is missing {name!r}")

    expected = {
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "num_train_epochs": float(NUM_TRAIN_EPOCHS),
        "learning_rate": LEARNING_RATE,
        "bf16": True,
        "logging_steps": LOGGING_STEPS,
        "optim": OPTIMIZER,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "seed": TRAINER_SEED,
        "eval_strategy": "no",
        "metric_for_best_model": "eval_loss",
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "save_steps": SAVE_STEPS,
        "dataset_text_field": "text",
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset_num_proc": DATASET_NUM_PROC,
        "packing": False,
        # These two parser arguments are not passed by f638.  The values below
        # are the effective TrainingArguments/SFTConfig defaults, not the
        # unused parser defaults (4 and 1000).
        "per_device_eval_batch_size": 8,
        "eval_steps": None,
    }
    aliases = {"eval_strategy": ("evaluation_strategy",)}
    actual = {
        name: observed(name, *aliases.get(name, ()))
        for name in expected
    }
    drift = {
        name: {"expected": value, "observed": actual[name]}
        for name, value in expected.items()
        if actual[name] != value
    }
    if drift:
        raise JiangF638SFTError(
            "Unsloth's legacy SFTTrainer wrapper produced unexpected effective defaults: "
            + json.dumps(drift, sort_keys=True)
        )
    return actual


def _build_hyperparams_payload(
    *,
    args: argparse.Namespace,
    model_path: Path,
    tokenizer_path: Path,
    data_root: Path,
    train_source: Path,
    output_dir: Path,
    runtime_versions: Mapping[str, str],
    hardware: Mapping[str, Any],
    base_snapshot: Mapping[str, Any],
    marker_audit: Mapping[str, Any],
    policy_preflight: Mapping[str, Any],
    training_args: Any,
) -> dict[str, Any]:
    stage_hyperparameters = {
        **MATCHED_SFT_STAGE_HPARAMETERS,
        "dtype": "bfloat16",
        "checkpoint_fidelity_mode": "publication",
        "load_in_4bit": False,
        "full_finetune": False,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_bias": LORA_BIAS,
        "use_gradient_checkpointing": "unsloth",
        "random_state": MODEL_RANDOM_STATE,
        "per_device_eval_batch_size": 8,
        "min_eval_samples_per_task": 0,
        "numeric_token_loss_weight": 1.0,
        "expected_terminal_global_step": TARGET_STEP,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "eval_steps": None,
        "evaluation_strategy": "no",
        "dataset_text_field": "text",
        "dataset_num_proc": DATASET_NUM_PROC,
        "response_header": RESPONSE_MARKER,
        "upstream_response_template": JIANG_F638_UPSTREAM_RESPONSE_TEMPLATE,
        "supervised_boundary": JIANG_F638_SUPERVISED_BOUNDARY,
        "first_supervised_token_policy": JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY,
        "response_marker_fallback": None,
        "single_explicit_dataset_shuffle_seed": DATASET_SHUFFLE_SEED,
        "tokenizer_path": str(tokenizer_path),
        "tokenization_alignment_train_mismatches": 0,
        "tokenization_alignment_eval_mismatches": 0,
    }
    cli = {
        "task": args.task,
        "tasks": [args.task],
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "report_to": args.report_to,
        "post_save_reload_check": bool(args.post_save_reload_check),
        **FIXED_RECORDED_CLI_ARGS,
    }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "stage": "sft",
        "tasks": [args.task],
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "dataset": {
            "train_size": EXPECTED_TRAIN_ROWS,
            "eval_size": 0,
            "train_task_counts": {args.task: EXPECTED_TRAIN_ROWS},
            "eval_task_counts": {},
            "train_source": str(train_source),
        },
        "distributed": {"world_size": 1, "rank": 0},
        "artifacts": {
            "model_dir": str(output_dir),
            "checkpoint_dir": str(output_dir / "checkpoints"),
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "tree": UPSTREAM_TREE,
            "model_id": UPSTREAM_MODEL_ID,
            "model_revision": UPSTREAM_MODEL_REVISION,
            "file_sha256": dict(UPSTREAM_FILE_SHA256),
            "prompt_template_sha256": hashlib.sha256(
                JIANG_F638_SFT_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "base_snapshot": dict(base_snapshot),
        },
        "intentional_corrections": {
            "fresh_train_call": "trainer.train()",
            "released_fresh_call": "trainer.train(resume_from_checkpoint=True)",
        "bias_policy": "all",
        "released_bias_policy": "lora_only",
        "bias_trainability_restore": "explicitly re-enable exact 84 Q/K/V base biases",
        "response_boundary": "contextual indented header without colon",
        "released_response_boundary": "standalone ### Response: token IDs (never match)",
        "trainer_api": "released legacy SFTTrainer kwargs through Unsloth compatibility wrapper",
        "effective_trainer_config_audit": "required before first optimizer step",
        "reason": (
            "fresh-start/API correctness, explicit QKV-bias training and serialization, "
            "and intended response-only supervision"
        ),
        },
        "runtime_versions": dict(runtime_versions),
        "hardware": dict(hardware),
        "marker_audit": dict(marker_audit),
        "policy_preflight": dict(policy_preflight),
        "stage_hyperparameters": stage_hyperparameters,
        "args": cli,
        "trainer_config": _trainer_config_payload(training_args),
        "effective_sft_config": audit_effective_sft_config(training_args),
    }


def _run_reload_audit(*, model_path: Path, output_dir: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "scripts" / "audit_sft_reload_equivalence.py"
    if not script.is_file() or script.is_symlink():
        raise JiangF638SFTError(f"reload-equivalence auditor is unavailable: {script}")
    command = [
        sys.executable,
        str(script),
        "--base-model-path",
        str(model_path),
        "--adapter-path",
        str(output_dir),
        "--runtime-dtype",
        "bfloat16",
        "--device",
        "cuda:0",
        "--output",
        str(output_dir / "sft_reload_equivalence.json"),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise JiangF638SFTError(
            f"post-save reload-equivalence audit failed with exit code {exc.returncode}"
        ) from exc


def _iclr_wandb_identity(environment: Mapping[str, str]) -> dict[str, str]:
    """Require the worker-controlled ICLR identity used by offline W&B logs."""

    identity: dict[str, str] = {}
    for key in ("WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_NAME"):
        value = str(environment.get(key, ""))
        if not value or "_iclr" not in value or "\n" in value or "\r" in value:
            raise JiangF638SFTError(
                f"{key} must be a nonempty, single-line _iclr identity"
            )
        identity[key] = value
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--post-save-reload-check", action="store_true")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        choices=ALLOWED_CHUNK_STOP_STEPS,
        default=TARGET_STEP,
        help="Stop normally only after this exact scheduled checkpoint has been saved.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Canonical checkpoint-N path from this same run; omitted only for a fresh chunk.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    """Execute one fresh or exact-checkpoint 500k specialist chunk."""

    runtime_versions = audit_runtime_versions()
    stop_after_step = _validated_chunk_stop_step(
        getattr(args, "stop_after_step", TARGET_STEP)
    )
    model_path = _canonical_existing_directory(args.model_path, label="base model")
    tokenizer_path = _canonical_existing_directory(
        args.tokenizer_path or args.model_path,
        label="tokenizer",
    )
    data_root, train_source = resolve_training_source(args.data_root, args.task)
    output_dir, resume_checkpoint, _ = resolve_chunk_execution(
        args.output_dir,
        stop_after_step=stop_after_step,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
    )
    base_snapshot = audit_jiang_f638_base_snapshot(model_path)
    if args.post_save_reload_check:
        script = Path(__file__).resolve().parents[3] / "scripts" / "audit_sft_reload_equivalence.py"
        if not script.is_file() or script.is_symlink():
            raise JiangF638SFTError(f"reload-equivalence auditor is unavailable: {script}")

    # Import order intentionally matches upstream: Unsloth patches TRL before
    # SFTTrainer is imported.
    import torch
    import wandb
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset, Features, Value
    from trl import SFTTrainer
    from transformers import AutoTokenizer, TrainingArguments

    from clasp.models.checkpoint_fidelity import (
        assert_qwen25_7b_publication_invariant,
        audit_and_write_checkpoint_fidelity,
        audit_trainable_state_closure,
        enable_qwen25_targeted_bias_training,
    )
    from clasp.runtime import write_prefixed_trainer_logs

    hardware = _audit_single_h100(torch, is_bfloat16_supported)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    if resume_checkpoint is None:
        checkpoint_dir.mkdir(exist_ok=False)
    elif checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise JiangF638SFTError(
            f"resume checkpoint root is no longer a regular directory: {checkpoint_dir}"
        )

    raw_train_dataset = load_projected_training_dataset(
        train_source,
        cache_dir=args.cache_dir,
        dataset_class=Dataset,
        features_class=Features,
        value_class=Value,
    )

    model_kwargs: dict[str, Any] = {
        "model_name": str(model_path),
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": torch.bfloat16,
        "load_in_4bit": False,
    }
    if args.cache_dir is not None:
        model_kwargs["cache_dir"] = str(args.cache_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    if tokenizer_path != model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
            local_files_only=True,
        )
    if not isinstance(getattr(tokenizer, "pad_token", None), str):
        raise JiangF638SFTError("tokenizer has no pad token required by the f638 LM collator")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias=LORA_BIAS,
        use_gradient_checkpointing="unsloth",
        random_state=MODEL_RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )
    bias_trainability_restore = enable_qwen25_targeted_bias_training(model)
    initial_closure = audit_trainable_state_closure(model)
    initial_invariant = assert_qwen25_7b_publication_invariant(model, initial_closure)
    policy_preflight = {
        "status": "exact_trainable_peft_state_closure",
        "tensor_count": initial_closure["tensor_count"],
        "parameter_count": initial_closure["parameter_count"],
        "tensor_inventory_sha256": initial_closure["tensor_inventory_sha256"],
        "publication_invariant": initial_invariant,
        "bias_trainability_restore": bias_trainability_restore,
    }

    train_dataset, marker_audit = prepare_training_dataset(raw_train_dataset, tokenizer)
    collator = build_strict_response_collator(tokenizer)

    if args.report_to == "wandb":
        wandb_identity = _iclr_wandb_identity(os.environ)
        wandb.init(
            project=wandb_identity["WANDB_PROJECT"],
            group=wandb_identity["WANDB_RUN_GROUP"],
            name=wandb_identity["WANDB_NAME"],
        )
    training_args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=TRAINER_SEED,
        output_dir=str(checkpoint_dir),
        # TrainingArguments otherwise embeds a per-process timestamp/hostname
        # here.  A later allocation would then differ from the create-once
        # hyperparams contract even though every training parameter is equal.
        logging_dir=str(checkpoint_dir / "logging"),
        report_to=args.report_to,
        evaluation_strategy="no",
        metric_for_best_model="eval_loss",
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_steps=SAVE_STEPS,
    )
    # Retain the released constructor surface.  The required Unsloth-first
    # import patches SFTTrainer to map tokenizer -> processing_class and move
    # the four legacy SFT keywords into an SFTConfig.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=False,
        data_collator=collator,
        args=training_args,
    )
    effective_sft_config = audit_effective_sft_config(trainer.args)
    post_trainer_closure = audit_trainable_state_closure(_unwrap_trainer_model(trainer))
    post_trainer_invariant = assert_qwen25_7b_publication_invariant(
        _unwrap_trainer_model(trainer),
        post_trainer_closure,
    )
    if (
        post_trainer_closure != initial_closure
        or post_trainer_invariant != initial_invariant
    ):
        raise JiangF638SFTError(
            "SFTTrainer initialization changed the exact LoRA/QKV-bias trainable closure"
        )
    policy_preflight["post_trainer_trainable_closure"] = post_trainer_closure
    policy_preflight["post_trainer_publication_invariant"] = post_trainer_invariant
    hyperparams = _build_hyperparams_payload(
        args=args,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        data_root=data_root,
        train_source=train_source,
        output_dir=output_dir,
        runtime_versions=runtime_versions,
        hardware=hardware,
        base_snapshot=base_snapshot,
        marker_audit=marker_audit,
        policy_preflight=policy_preflight,
        training_args=trainer.args,
    )
    if hyperparams["effective_sft_config"] != effective_sft_config:
        raise JiangF638SFTError("effective SFT config changed during run-contract creation")
    write_or_validate_hyperparams(
        output_dir / "hyperparams.json",
        hyperparams,
        resume=resume_checkpoint is not None,
    )
    checkpoint_callback = build_checkpoint_fidelity_callback(trainer)
    trainer.add_callback(checkpoint_callback)
    chunk_stop_callback = build_chunk_stop_callback(stop_after_step)
    trainer.add_callback(chunk_stop_callback)

    if resume_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    if (
        int(trainer.state.global_step) != stop_after_step
        or int(trainer.state.max_steps) != TARGET_STEP
    ):
        raise JiangF638SFTError(
            "fixed-horizon Trainer did not stop at the requested saved boundary: "
            f"global_step={trainer.state.global_step}, max_steps={trainer.state.max_steps}"
        )
    if not chunk_stop_callback.saved_target:
        raise JiangF638SFTError(
            f"Trainer stopped at step {stop_after_step} without a verified saved checkpoint"
        )
    observed_checkpoint_names = sorted(
        entry.name for entry in checkpoint_dir.iterdir()
    )
    expected_checkpoint_names = sorted(
        f"checkpoint-{step}"
        for step in EXPECTED_TRAINER_CHECKPOINT_STEPS
        if step <= stop_after_step
    )
    if observed_checkpoint_names != expected_checkpoint_names:
        raise JiangF638SFTError(
            "Trainer checkpoint inventory differs from the f638/Transformers-4.50.3 "
            f"schedule: observed={observed_checkpoint_names}, "
            f"expected={expected_checkpoint_names}"
        )

    if stop_after_step < TARGET_STEP:
        checkpoint_callback._trainer = None  # type: ignore[attr-defined]
        del chunk_stop_callback
        del checkpoint_callback
        del trainer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if args.report_to == "wandb":
            wandb.finish()
        return output_dir

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    audit_and_write_checkpoint_fidelity(
        _unwrap_trainer_model(trainer),
        output_dir,
        require_qwen25_7b_publication=True,
    )
    write_prefixed_trainer_logs(str(output_dir), "sft", trainer.state)

    # Release the training allocation before the optional independent two-load
    # check.  Its receipt is exactly the one consumed by existing certification.
    checkpoint_callback._trainer = None  # type: ignore[attr-defined]
    trainer_state = trainer.state
    del chunk_stop_callback
    del checkpoint_callback
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    write_final_stage_status(output_dir, trainer_state)
    if args.post_save_reload_check:
        _run_reload_audit(model_path=model_path, output_dir=output_dir)
    if args.report_to == "wandb":
        wandb.finish()
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (JiangF638SFTError, OSError, ValueError) as exc:
        print(f"[JiangF638SFTError] {exc}", file=sys.stderr)
        return 2
    stop_after_step = _validated_chunk_stop_step(args.stop_after_step)
    status = "complete" if stop_after_step == TARGET_STEP else "chunk complete"
    print(f"[JiangF638SFT] {status} at step {stop_after_step}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
