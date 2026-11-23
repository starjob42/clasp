"""Dedicated executable entry point for FOARL-f638-corrected.

This is a common-contract reimplementation, not an official reproduction.  It
continues the corrected Jiang f638 SFT adapter, uses the exact two released
training reward heads, and delegates model loading, checkpointing, and the
independent frozen KL reference to the shared GRPO runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from clasp.baselines.llmcosolver_f638_training import (
    FOARL_F638_ADVANTAGE_EPSILON,
    FOARL_F638_BETA,
    FOARL_F638_CLIP_EPSILON,
    FOARL_F638_CORRECTED_RUNTIME,
    FOARL_F638_NUM_GENERATIONS,
    FOARL_F638_NUM_ITERATIONS,
    FOARL_F638_PROMPT_STYLE,
    FOARL_F638_SCALE_REWARDS,
    FOARL_F638_SYNC_REF_MODEL,
    FOARL_F638_TOKEN_AGGREGATION,
    validate_foarl_f638_prompt_template,
)
from clasp.config import build_grpo_parser, finalize_grpo_args
from clasp.runtime import print_primary
from clasp.training_source_identity import (
    FOARL_F638_CORRECTED_TRAINER_VERSION,
    require_train_grpo_source,
)

try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except ImportError:

    def _elastic_record(fn):
        return fn


def build_foarl_f638_corrected_parser() -> argparse.ArgumentParser:
    parser = build_grpo_parser()
    # The shared parser requires --model_path.  This lane has a stronger,
    # unambiguous initialization argument and resolves it before shared runtime
    # code sees the namespace.
    for action in parser._actions:
        if action.dest == "model_path":
            action.required = False
            action.default = None
        elif action.dest == "tasks":
            action.required = True
            action.default = None
    parser.description = (
        "Run FOARL-f638-corrected with exact released rewards and a frozen corrected-SFT reference."
    )
    parser.add_argument(
        "--sft_checkpoint_path",
        type=str,
        required=True,
        help="Corrected Jiang f638 bias=all SFT output or checkpoint-* directory.",
    )
    parser.add_argument(
        "--foarl_advantage_epsilon",
        type=float,
        default=FOARL_F638_ADVANTAGE_EPSILON,
        help="Immutable group-normalization epsilon; noncanonical values are rejected.",
    )
    parser.add_argument(
        "--foarl_scale_rewards",
        type=str,
        default=FOARL_F638_SCALE_REWARDS,
        help="Immutable reward scaling scope; must be 'group'.",
    )
    parser.add_argument(
        "--foarl_num_iterations",
        type=int,
        default=FOARL_F638_NUM_ITERATIONS,
        help="Immutable number of updates per rollout group; must be 1.",
    )
    parser.add_argument(
        "--foarl_token_aggregation",
        type=str,
        default=FOARL_F638_TOKEN_AGGREGATION,
        help="Immutable loss reduction identifier.",
    )
    parser.add_argument(
        "--foarl_sync_ref_model",
        action=argparse.BooleanOptionalAction,
        default=FOARL_F638_SYNC_REF_MODEL,
        help="Must remain false so the corrected step-zero SFT reference is immutable.",
    )
    parser.set_defaults(
        use_primal_dual=False,
        use_clasp=False,
        full_finetune=False,
        verify_gt_rewards=False,
        no_reason=False,
        prompt_style=FOARL_F638_PROMPT_STYLE,
        num_generations=FOARL_F638_NUM_GENERATIONS,
        num_generations_train=None,
        num_return_sequences=None,
        grpo_beta=FOARL_F638_BETA,
        # These are PPO clipping epsilons, not the 1e-4 advantage epsilon.
        grpo_epsilon=FOARL_F638_CLIP_EPSILON,
        grpo_epsilon_high=FOARL_F638_CLIP_EPSILON,
        grpo_loss_type="grpo",
        # Released f638 RL defaults.  A later matched-compute launcher may
        # override budget fields, but direct execution should not inherit the
        # unrelated generic-GRPO recipe accidentally.
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        weight_decay=0.02,
        num_train_epochs=1.0,
        max_prompt_length=20_000,
        max_completion_length=1_000,
        lora_r=64,
        lora_alpha=64,
        lora_dropout=0.0,
        save_steps=50,
        wandb_project="FOARL-f638-corrected",
    )
    return parser


_REPRESENTATION_ENV_DEFAULTS = {
    "CLASP_LORA_TARGET_MODULES": "jiang",
    "CLASP_LORA_BIAS": "all",
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_INPUT_ORDER": "original",
    "CLASP_TSP_RELABEL_ORDER": "original",
    "CLASP_TSP_ROUTE_CANONICALIZE": "original",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
    "CLASP_TSP_LABEL_TTA_SCHEDULE": "",
    "CLASP_MIS_RELABEL_ORDER": "original",
    "CLASP_MIS_PROMPT_NODE_DEGREES": "0",
    "CLASP_MIS_PROMPT_LOW_DEGREE_K": "32",
}


def seal_foarl_f638_loader_environment(task: str) -> dict[str, str]:
    """Reject inherited representation drift, then expose explicit defaults."""

    expected = {
        **_REPRESENTATION_ENV_DEFAULTS,
        "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1" if task == "mis" else "0",
    }
    drift = {
        name: {"expected": value, "observed": os.environ[name]}
        for name, value in expected.items()
        if name in os.environ and os.environ[name] != value
    }
    if drift:
        raise ValueError(
            "FOARL-f638-corrected rejects inherited representation-altering loader "
            f"environment: {json.dumps(drift, sort_keys=True)}"
        )
    for name, value in expected.items():
        os.environ[name] = value
    return dict(expected)


def _task_values(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [piece.strip().lower() for piece in value.split(",") if piece.strip()]
    elif isinstance(value, (list, tuple)):
        values = [str(piece).strip().lower() for piece in value if str(piece).strip()]
    else:
        raise ValueError(f"checkpoint task identity has unsupported type {type(value).__name__}")
    return tuple(values)


def verify_foarl_f638_checkpoint_task_binding(model_path: str, task: str) -> dict[str, Any]:
    """Bind an exposed SFT receipt task to the one requested specialist lane."""

    model_dir = Path(model_path).expanduser().resolve()
    roots = [model_dir]
    if model_dir.name.startswith("checkpoint-"):
        roots.append(model_dir.parent)
        if model_dir.parent.name == "checkpoints":
            roots.append(model_dir.parent.parent)
    receipt_paths: list[Path] = []
    for root in roots:
        candidate = root / "hyperparams.json"
        if candidate.is_file() and candidate not in receipt_paths:
            receipt_paths.append(candidate)

    bindings: list[dict[str, Any]] = []
    for receipt_path in receipt_paths:
        try:
            receipt_bytes = receipt_path.read_bytes()
            payload = json.loads(receipt_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read checkpoint task receipt {receipt_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint task receipt must be a JSON object: {receipt_path}")
        args_payload = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        exposed = [
            values
            for values in (
                _task_values(payload.get("tasks")),
                _task_values(args_payload.get("tasks")),
                _task_values(args_payload.get("task")),
            )
            if values is not None
        ]
        if not exposed:
            continue
        if any(values != (task,) for values in exposed):
            raise ValueError(
                "FOARL checkpoint task identity disagrees with requested specialist: "
                f"requested={task}, exposed={exposed}, receipt={receipt_path}"
            )
        stage = payload.get("stage")
        if stage is not None and str(stage).strip().lower() != "sft":
            raise ValueError(
                f"FOARL initialization receipt must describe stage=sft, got {stage!r}"
            )
        bindings.append(
            {
                "path": str(receipt_path),
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "task": task,
            }
        )
    return {
        "status": "BOUND" if bindings else "TASK_IDENTITY_NOT_EXPOSED",
        "requested_task": task,
        "receipts": bindings,
    }


def parse_foarl_f638_corrected_args(argv: list[str] | None = None):
    parser = build_foarl_f638_corrected_parser()
    args = finalize_grpo_args(parser.parse_args(argv))
    return finalize_foarl_f638_args_namespace(
        args,
        runtime_profile=FOARL_F638_CORRECTED_RUNTIME,
        purpose="FOARL-f638-corrected",
    )


def finalize_foarl_f638_args_namespace(
    args: Any,
    *,
    runtime_profile=FOARL_F638_CORRECTED_RUNTIME,
    purpose: str = "FOARL-f638-corrected",
):
    """Apply the common corrected-SFT/task/source gates to a resolved namespace."""

    runtime_profile.validate_args(args)

    # Keep training-stack imports lazy so the semantic parser and contract tests
    # remain usable in lightweight audit environments.
    from clasp.entrypoints.train_grpo_standard import _resolve_sft_init_model_path
    from clasp.models.loading import (
        JIANG_LORA_TARGET_MODULES,
        lora_target_modules_from_env,
        peft_checkpoint_bias_policy,
    )
    from clasp.models.training_fidelity import verify_publication_input_adapter

    args.model_path = _resolve_sft_init_model_path(args.sft_checkpoint_path)
    task = str(args.tasks[0]).lower()
    args.foarl_task_identity_binding = verify_foarl_f638_checkpoint_task_binding(
        args.model_path,
        task,
    )
    args.foarl_loader_environment = seal_foarl_f638_loader_environment(task)
    args.foarl_prompt_template_validation = validate_foarl_f638_prompt_template()
    bias_policy = peft_checkpoint_bias_policy(args.model_path)
    if bias_policy != "all":
        raise ValueError(
            f"{purpose} requires the corrected bias=all SFT adapter; "
            f"resolved bias policy is {bias_policy!r} at {args.model_path}"
        )
    if lora_target_modules_from_env() != list(JIANG_LORA_TARGET_MODULES):
        raise ValueError(
            f"{purpose} requires the exact seven Jiang LoRA target modules"
        )
    verify_publication_input_adapter(
        args.model_path,
        purpose=f"{purpose} initialization adapter",
    )
    return args


@_elastic_record
def main() -> None:
    args = parse_foarl_f638_corrected_args()
    trainer_source = require_train_grpo_source(FOARL_F638_CORRECTED_TRAINER_VERSION)
    from clasp.entrypoints.train_grpo import run_grpo

    print_primary(
        "[FOARL-f638-corrected] Exact runtime active: "
        f"profile={FOARL_F638_CORRECTED_RUNTIME.semantics.profile_id}, "
        f"trainer_source={trainer_source['version']}@{trainer_source['sha256']}, "
        f"output_root={args.output_root}, report_to={args.report_to}"
    )
    run_grpo(args, runtime_profile=FOARL_F638_CORRECTED_RUNTIME)


if __name__ == "__main__":
    main()
