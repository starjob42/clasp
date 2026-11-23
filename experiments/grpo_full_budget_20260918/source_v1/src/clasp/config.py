"""Command-line interface: argparse parsers, cross-validation, and preset logic.

Defines every ``build_*_parser`` and ``finalize_*_args`` pair. This is where arguments are
cross-validated against TRL's batch/generation constraints, and where the CLASP presets
live: :func:`_apply_clasp_strict_defaults` translates the high-level ``--use_clasp`` and
``--clasp_raw_feasibility_targeting`` flags into concrete ``pd_*`` settings *and* exported
``CLASP_*`` environment variables.

Much runtime behaviour is driven by those ``CLASP_*`` env vars rather than by the parsed
arguments, so when tracing CLASP behaviour, grep for both a flag and its ``CLASP_*`` twin.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from clasp.runtime import (
    DEFAULT_RL_ROOT,
    DEFAULT_SFT_ROOT,
    effective_optimization_batch_size,
    global_batch_size,
    parse_task_list,
    resolve_generation_schedule,
)


def add_shared_arguments(parser: argparse.ArgumentParser, *, data_root_default: str) -> None:
    parser.add_argument("--tasks", type=str, default="tsp,cvrp,op,mis,jssp")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer source path. Defaults to --model_path. "
            "For Gemma 4 base checkpoints, you can point this to a matching *-it tokenizer."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["auto", "alpaca", "alpaca_chat", "alpaca_hint", "jiang_f638"],
        default="auto",
        help=(
            "Prompt rendering strategy. "
            "'alpaca' keeps the existing plain instruction format, "
            "'alpaca_chat' wraps the Alpaca content with tokenizer.apply_chat_template, "
            "'alpaca_hint' adds a no-repair internal construction/checking instruction, "
            "'auto' enables alpaca_chat for Gemma when supported and otherwise falls back to alpaca."
        ),
    )
    parser.add_argument("--data_root", type=str, default=data_root_default)
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--full_finetune", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument(
        "--sft_eval_ratio",
        type=float,
        default=None,
        help="Stage-specific override for SFT eval split ratio. Defaults to --eval_ratio.",
    )
    parser.add_argument(
        "--grpo_eval_ratio",
        type=float,
        default=None,
        help="Stage-specific override for GRPO eval split ratio. Defaults to --eval_ratio.",
    )
    parser.add_argument(
        "--eval-from-train-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Derive the training-time eval split by slicing each task's train set "
            "instead of loading task eval files."
        ),
    )
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--max_eval_samples_per_task", type=int, default=0)
    parser.add_argument("--min_train_samples_per_task", type=int, default=0)
    parser.add_argument("--min_eval_samples_per_task", type=int, default=0)
    parser.add_argument(
        "--prefer_subset_files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer *_subset.json files when present under task split directories.",
    )

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)


def add_context_window_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--context_window",
        type=int,
        default=0,
        help=(
            "Shared total context-window budget. "
            "For SFT this overrides --max_seq_length. "
            "For GRPO this sets max_prompt_length = context_window - max_completion_length. "
            "Set to 0 to use the explicit max_* arguments."
        ),
    )


def build_sft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-GPU TRL SFT for mixed combinatorial optimization.")
    add_shared_arguments(parser, data_root_default=DEFAULT_SFT_ROOT)
    add_context_window_argument(parser)
    parser.add_argument(
        "--no_reason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the no-reasoning Alpaca prompt variant during SFT.",
    )
    parser.add_argument(
        "--deduplicate_train_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove exact duplicate SFT train rows before selection. Disable only when "
            "an explicitly source-matched recipe must preserve the released sampling distribution."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optional Transformers optimizer name (for example adamw_8bit).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optional optimizer weight decay override passed to SFTConfig.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Optional fixed number of LR warmup steps passed to SFTConfig.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Optional Transformers LR scheduler name (for example linear).",
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=0,
        help=(
            "Optional distributed-training timeout in seconds for SFT. "
            "Set to 0 to keep the TRL/Transformers default."
        ),
    )
    parser.add_argument("--train_size_bucket", type=str, default="all")
    parser.add_argument("--train_min_instance_size", type=int, default=0)
    parser.add_argument("--train_max_instance_size", type=int, default=0)
    parser.add_argument(
        "--numeric_token_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Default-off SFT loss multiplier for completion label tokens whose decoded "
            "token piece contains a digit or list punctuation. Values <= 1.0 keep the "
            "standard SFT loss."
        ),
    )
    return parser


def build_grpo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-GPU TRL GRPO for mixed combinatorial optimization.")
    add_shared_arguments(parser, data_root_default=DEFAULT_RL_ROOT)
    add_context_window_argument(parser)
    parser.add_argument(
        "--no_reason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the no-reasoning Alpaca prompt variant during GRPO.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optional Transformers optimizer name (for example adamw_8bit).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optional optimizer weight decay override passed to GRPOConfig.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Optional LR warmup ratio override passed to GRPOConfig.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Optional Transformers LR scheduler type passed to GRPOConfig.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument(
        "--num_generations_train",
        type=int,
        default=None,
        help=(
            "Training-only alias for --num_generations. Evaluation scripts keep their "
            "own num_return_sequences setting, so this can be 16/32 for CLASP training "
            "without changing raw benchmark sampling."
        ),
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=None,
        help="Deprecated alias for --num_generations.",
    )
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--steps_per_generation", type=int, default=None)
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=None,
        help="Optional training rollout temperature override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--generation_top_p",
        type=float,
        default=None,
        help="Optional training rollout top-p override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--generation_top_k",
        type=int,
        default=None,
        help="Optional training rollout top-k override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--generation_repetition_penalty",
        type=float,
        default=None,
        help="Optional training rollout repetition penalty override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--grpo_beta",
        type=float,
        default=None,
        help="Optional GRPO KL beta override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--grpo_epsilon",
        type=float,
        default=None,
        help="Optional GRPO lower clipping epsilon override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--grpo_epsilon_high",
        type=float,
        default=None,
        help="Optional GRPO upper clipping epsilon override passed to GRPOConfig when supported.",
    )
    parser.add_argument(
        "--grpo_loss_type",
        type=str,
        default=None,
        help="Optional GRPO loss_type override passed to GRPOConfig when supported.",
    )

    parser.add_argument("--verify_gt_rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_check_samples_per_task", type=int, default=5)

    parser.add_argument("--use_primal_dual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_constraint_target", type=float, default=0.98)
    parser.add_argument("--pd_cost_limit", type=float, default=None)
    parser.add_argument("--pd_lambda_init", type=float, default=0.0)
    parser.add_argument("--pd_dual_lr", type=float, default=0.05)
    parser.add_argument("--pd_dual_lr_up", type=float, default=None)
    parser.add_argument("--pd_dual_lr_down", type=float, default=None)
    parser.add_argument("--pd_dual_update_interval", type=int, default=1)
    parser.add_argument("--pd_violation_ema_beta", type=float, default=1.0)
    parser.add_argument("--pd_lambda_max", type=float, default=10.0)
    parser.add_argument("--pd_violation_tolerance", type=float, default=0.01)
    parser.add_argument("--pd_objective_weight", type=float, default=1.0)
    parser.add_argument("--pd_structure_weight", type=float, default=0.2)
    parser.add_argument("--pd_sync_across_ranks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_require_state_on_resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_strict_constraint_range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_force_scale_rewards_none", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_log_every", type=int, default=50)
    parser.add_argument(
        "--pd_soft_objective_for_infeasible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable partial objective credit on near-feasible outputs during primal-dual GRPO. "
            "This usually improves early-stage reward density."
        ),
    )
    parser.add_argument(
        "--pd_improved_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the opt-in constrained-GRPO improvements: normalized primal advantage, "
            "lexicographic feasible-first advantage, and CO component weighting. Use the "
            "individual pd_* flags for ablations when not using the bundled preset."
        ),
    )
    parser.add_argument(
        "--pd_normalize_primal_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize only the primal reward advantage within each GRPO prompt group.",
    )
    parser.add_argument("--pd_primal_advantage_scale", type=float, default=1.0)
    parser.add_argument("--pd_penalty_scale", type=float, default=1.0)
    parser.add_argument(
        "--pd_group_scalarization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the strong Dual-Primal GRPO baseline A=standardize_group(r-lambda*v). "
            "This differs from the legacy constrained trainer, which standardizes the "
            "primal term before subtracting an unstandardized constraint penalty."
        ),
    )
    parser.add_argument(
        "--pd_separate_z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the exact DUAL-SEPARATE-Z comparator A=z_G(r)-z_G(p), where "
            "p=sum_j lambda_j*w_j*max(0,c_j-limit_j) is the positive aggregate "
            "component penalty. Mutually exclusive with --pd_group_scalarization "
            "and --use_clasp."
        ),
    )
    parser.add_argument(
        "--pd_penalty_scale_start",
        type=float,
        default=None,
        help="Optional starting penalty scale for a linear warmup toward --pd_penalty_scale.",
    )
    parser.add_argument("--pd_penalty_scale_warmup_fraction", type=float, default=0.0)
    parser.add_argument(
        "--pd_positive_constraint_penalty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use only positive constraint violations in the primal-dual penalty term. "
            "This avoids rewarding already-feasible samples for being far below a constraint limit."
        ),
    )
    parser.add_argument(
        "--pd_lexicographic_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use feasible-first CO ranking: feasible completions are ranked by primal objective; "
            "infeasible completions are ranked by constraint severity."
        ),
    )
    parser.add_argument("--pd_lexicographic_feasible_margin", type=float, default=0.5)
    parser.add_argument(
        "--pd_component_profile",
        choices=[
            "none",
            "co_default",
            "co_op_strict_length",
            "co_op_strict_length_prizecap",
            "co_op_strict_length_margin",
            "co_jssp_valid_target_gap",
            "co_feasibility_lock_all",
            "co_jssp_tail_feas_lock_all",
            "co_mis_strict_edges",
            "co_mis_conflict_nodes",
            "co_mis_conflict_target_gap",
            "co_mis_conflict_target_gap_sizecap",
            "co_mis_conflict_target_gap_maximality",
            "co_mis_conflict_repair_target_gap",
            "co_mis_target_gap",
            "co_mis_repair_target",
        ],
        default="none",
        help="Optional per-task/per-cost tolerance and penalty-weight profile for primal-dual GRPO.",
    )
    parser.add_argument("--pd_enable_schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pd_stage_a_fraction", type=float, default=0.25)
    parser.add_argument("--pd_stage_b_fraction", type=float, default=0.70)
    parser.add_argument("--pd_stage_a_cost_limit", type=float, default=0.05)
    parser.add_argument("--pd_stage_b_cost_limit", type=float, default=0.02)
    parser.add_argument("--pd_stage_c_cost_limit", type=float, default=0.01)
    parser.add_argument("--pd_stage_a_dual_lr_scale", type=float, default=0.20)
    parser.add_argument("--pd_stage_b_dual_lr_scale", type=float, default=0.50)
    parser.add_argument("--pd_stage_c_dual_lr_scale", type=float, default=1.00)
    parser.add_argument("--pd_stage_c_constraint_threshold", type=float, default=0.95)

    # ----------------------------------------------------------------------
    # CLASP-GRPO (Constrained Lagrangian Anchored Saddle-Point GRPO).
    # All flags are gated behind --use_clasp. When --use_clasp is False the
    # remaining flags are recorded but NOT consulted by the trainer.
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--use_clasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CLASP-GRPO. When on, overrides the primal-dual advantage "
            "combiner, dual-ascent rule, and (optionally) the completion-length "
            "budget. Requires --use_primal_dual."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_set",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the experimental canonical finite-batch CLASP branch. "
            "--num_generations is the rollout count B and "
            "--clasp_deployment_set_size is the deployment count K, with K < B."
        ),
    )
    parser.add_argument(
        "--clasp_fixed_null_set",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use formula-matched non-centered fixed-null credit for the exact "
            "canonical K-set utility. Requires an explicit frozen construction "
            "receipt; no null candidate is inferred by the trainer."
        ),
    )
    parser.add_argument(
        "--clasp_fixed_null_receipt_path",
        type=str,
        default=None,
        help=(
            "Frozen prompt-keyed evaluator receipt for the exact fixed null. "
            "Required with --clasp_fixed_null_set and rejected otherwise."
        ),
    )
    parser.add_argument(
        "--clasp_deployment_set_size",
        type=int,
        default=0,
        help=(
            "Deployment candidate count K for canonical-set CLASP. A positive "
            "value is required when --clasp_canonical_set is enabled; B=K+1 is "
            "the recommended exact-enumeration schedule."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_recovery_scale",
        type=float,
        default=1.0,
        help="Fixed rho multiplying bounded recovery in failed size-K subsets.",
    )
    parser.add_argument(
        "--clasp_canonical_dual_recovery_coupling",
        type=float,
        default=0.0,
        help=(
            "Experimental non-negative gamma for the named failed-set recovery "
            "pressure arm: effective rho is rho + gamma * frozen lambda. Zero "
            "preserves the canonical binary-failure objective exactly."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_recovery_temperature",
        type=float,
        default=0.15,
        help="Positive soft-min temperature for canonical failed-set recovery.",
    )
    parser.add_argument(
        "--clasp_canonical_max_subsets",
        type=int,
        default=1024,
        help=(
            "Fail-closed cap on exact size-K subsets per rollout group. The "
            "recommended B=K+1 schedule needs only B subsets."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_exact_scale",
        type=float,
        default=1.0,
        help="Positive weight on the exact K-set canonical policy coefficient.",
    )
    parser.add_argument(
        "--clasp_canonical_grpo_anchor_scale",
        type=float,
        default=0.0,
        help=(
            "Non-negative weight on the normalized feasible-only candidate GRPO "
            "advantage added to canonical set credit. Zero preserves historical "
            "canonical CLASP; a positive value is defined only for canonical "
            "leave-out credit, not fixed-null credit."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_quality_anchor_ratio",
        type=float,
        default=0.0,
        help=(
            "Non-negative batch-RMS target for canonical best-of-K quality "
            "credit relative to the feasible-only GRPO anchor. Zero preserves "
            "the raw exact coefficient. A positive value replaces only the raw "
            "quality component by a detached norm-matched residual (rescale "
            "capped at 4x); recovery and dual failure credit remain in their "
            "original utility units."
        ),
    )
    parser.add_argument(
        "--clasp_canonical_dense_quality_scale",
        type=float,
        default=0.0,
        help="Non-negative dense feasible-candidate quality-midrank weight.",
    )
    parser.add_argument(
        "--clasp_canonical_dense_feasibility_scale",
        type=float,
        default=0.0,
        help="Non-negative dense mixed-batch feasibility-contrast weight.",
    )
    parser.add_argument(
        "--clasp_canonical_dense_violation_scale",
        type=float,
        default=0.0,
        help="Non-negative lower-violation midrank weight in mixed batches.",
    )
    parser.add_argument(
        "--clasp_canonical_dense_recovery_scale",
        type=float,
        default=0.0,
        help="Non-negative lower-violation midrank weight in failed batches.",
    )
    parser.add_argument(
        "--clasp_policy_length_alpha",
        type=float,
        default=0.0,
        help=(
            "Exact-set policy-gradient response-length exponent in [0,1]. "
            "0 preserves the historical per-response token mean; 1 uses a "
            "fixed max-completion-horizon sequence-score reduction. Values "
            "above 0 require sequence-level importance sampling."
        ),
    )
    parser.add_argument(
        "--clasp_strict_defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --use_clasp is enabled, force evaluator-aligned feasibility defaults: "
            "co_feasibility_lock_all, target=1.0, cost_limit=0.0, no infeasible objective "
            "credit, and CLASP_FEASIBILITY_GATED=1."
        ),
    )
    parser.add_argument(
        "--clasp_raw_feasibility_targeting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the strict raw-feasibility CLASP preset: hard feasibility gates, "
            "feasibility-only warmup defaults, and self-generated feasible replay as "
            "auxiliary losses only. Inference remains raw decoding."
        ),
    )
    parser.add_argument(
        "--clasp_bon_success_credit",
        type=str,
        choices=("dense_rank", "exact_loo"),
        default="dense_rank",
        help=(
            "Successful-group BoN credit estimator. dense_rank is the tie-safe, "
            "variance-reduced lexicographic default; exact_loo preserves the exact "
            "best-of-K leave-one-out estimator as an ablation."
        ),
    )
    parser.add_argument(
        "--clasp_component_dual_recovery_scale",
        type=float,
        default=1.0,
        help=(
            "Interpolation scale between the fixed and dual-aware bounded "
            "lexicographic recovery scores. Zero is the explicit no-component-dual "
            "ablation."
        ),
    )
    parser.add_argument(
        "--clasp_feasibility_warmup_steps",
        type=int,
        default=0,
        help=(
            "Minimum optimizer steps for feasibility-only CLASP advantages. During "
            "warmup, objective ranking is disabled and dense constraint violation "
            "reduction supplies the learning signal."
        ),
    )
    parser.add_argument(
        "--clasp_feasibility_warmup_targets",
        type=str,
        default="tsp:0.995,mis:0.995,op:0.99,cvrp:0.985,jssp:0.985",
        help=(
            "Comma-separated task:threshold validation pass@K gates used after "
            "--clasp_feasibility_warmup_steps, e.g. cvrp:0.985,jssp:0.985. "
            "These are group-success probabilities, not candidate feasibility."
        ),
    )
    parser.add_argument(
        "--clasp_feasibility_warmup_hold_until_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep feasibility-only warmup active after the minimum step count until "
            "observed validation pass@K reaches the per-task threshold."
        ),
    )
    parser.add_argument(
        "--clasp_self_replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Store self-generated feasible completions that pass the exact reward "
            "manager and use them for auxiliary imitation/contrastive losses. "
            "Defaults on only under --clasp_raw_feasibility_targeting."
        ),
    )
    parser.add_argument("--clasp_self_replay_imitation_weight", type=float, default=0.0)
    parser.add_argument("--clasp_self_replay_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--clasp_self_replay_sample_rate", type=float, default=0.5)
    parser.add_argument("--clasp_self_replay_max_tokens", type=int, default=512)
    parser.add_argument("--clasp_self_replay_max_keys", type=int, default=20000)
    parser.add_argument("--clasp_self_replay_max_per_key", type=int, default=4)
    parser.add_argument("--clasp_self_replay_min_buffer_size", type=int, default=8)
    parser.add_argument(
        "--clasp_validation_gate_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When validation pass@K metrics are available, suppress checkpoint "
            "saves that fail the --clasp_feasibility_warmup_targets gates."
        ),
    )
    parser.add_argument(
        "--clasp_pareto_anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the legacy pre-BoN Pareto-anchor ablation. It is bypassed by "
            "the default CLASP_BON_ALIGNED estimator and is not a claimed main-method "
            "contribution."
        ),
    )
    parser.add_argument(
        "--clasp_pareto_tau",
        type=float,
        default=4.0,
        help=(
            "Temperature of the sigmoid controlling the primal/dual trade-off "
            "alpha_G in the Pareto-anchored advantage. Ignored unless "
            "--clasp_pareto_anchor is on."
        ),
    )
    parser.add_argument(
        "--clasp_pareto_center",
        type=float,
        default=0.5,
        help=(
            "Feasibility-rate crossover c in alpha_G=sigmoid(tau*(f_G-c)). "
            "Must lie in [0, 1]. Ignored unless --clasp_pareto_anchor is on."
        ),
    )
    parser.add_argument(
        "--clasp_proper_dual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use proper stochastic dual ascent with feasibility-conditional "
            "step size (Contribution B). Ignored unless --use_clasp is on."
        ),
    )
    parser.add_argument(
        "--clasp_dual_update_interval",
        type=int,
        default=4,
        help=(
            "Number of primal steps between successive dual updates under "
            "proper dual ascent. Ignored unless --clasp_proper_dual is on."
        ),
    )
    parser.add_argument(
        "--clasp_proper_dual_gradient_mode",
        choices=("signed", "positive_excess"),
        default="signed",
        help=(
            "Per-component gradient estimator for CLASP projected dual ascent. "
            "signed is the proper masked mean (cost-limit) and can decrease lambda; "
            "positive_excess is the explicitly one-sided ablation."
        ),
    )
    parser.add_argument(
        "--clasp_adaptive_length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use task-adaptive completion-length budgeting (Contribution C). "
            "Ignored unless --use_clasp is on."
        ),
    )
    parser.add_argument(
        "--clasp_adaptive_length_quantile",
        type=float,
        default=0.99,
        help=(
            "Quantile of per-task completion length used as the per-task "
            "completion-length budget. Ignored unless --clasp_adaptive_length is on."
        ),
    )
    parser.add_argument(
        "--clasp_adaptive_length_floor",
        type=int,
        default=128,
        help="Lower bound on the per-task completion-length budget.",
    )
    parser.add_argument(
        "--clasp_adaptive_length_window",
        type=int,
        default=1000,
        help="Rolling window size (samples per task) for the adaptive budget estimator.",
    )
    parser.add_argument(
        "--clasp_adaptive_length_refresh_steps",
        type=int,
        default=50,
        help="Refresh the adaptive budget every N optimizer steps.",
    )
    parser.add_argument(
        "--clasp_adaptive_length_margin",
        type=int,
        default=32,
        help=(
            "Non-negative safety margin added to the rolling completion-length quantile. "
            "The paper default is 32 tokens."
        ),
    )
    parser.add_argument(
        "--clasp_mixed_adaptive_length_mode",
        choices=("error", "global_max"),
        default="error",
        help=(
            "Behavior when one mixed-task generation batch requests unequal per-task "
            "length budgets. error fails closed because the backend accepts only one "
            "batch-wide limit; global_max is the explicit longest-task ablation."
        ),
    )

    parser.add_argument(
        "--train_size_bucket",
        choices=["all", "small", "medium", "large", "hardest"],
        default="all",
        help="Optional GRPO train-set instance-size bucket, using the same semantics as evaluation.",
    )
    parser.add_argument("--train_min_instance_size", type=int, default=0)
    parser.add_argument("--train_max_instance_size", type=int, default=0)
    parser.add_argument(
        "--train_task_multipliers",
        type=str,
        default="",
        help=(
            "Optional comma-separated task sampling multipliers, e.g. 'cvrp:2,jssp:1.5,mis:0.5'. "
            "Applied after size filtering and before mixed-task concatenation."
        ),
    )
    return parser


def build_pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified multi-GPU TRL SFT -> GRPO pipeline for mixed combinatorial optimization."
    )
    parser.add_argument("--tasks", type=str, default="tsp,cvrp,op,mis,jssp")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer source path. Defaults to --model_path. "
            "For Gemma 4 base checkpoints, you can point this to a matching *-it tokenizer."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["auto", "alpaca", "alpaca_chat", "alpaca_hint", "jiang_f638"],
        default="auto",
        help=(
            "Prompt rendering strategy shared by SFT and GRPO. "
            "'alpaca' keeps the existing plain instruction format, "
            "'alpaca_chat' wraps Alpaca content with tokenizer.apply_chat_template, "
            "'alpaca_hint' adds a no-repair internal construction/checking instruction, "
            "'auto' enables alpaca_chat for Gemma when supported and otherwise falls back to alpaca."
        ),
    )
    parser.add_argument("--sft_data_root", type=str, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--grpo_data_root", type=str, default=DEFAULT_RL_ROOT)
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--sft_output_dir", type=str, default=None)
    parser.add_argument("--grpo_output_dir", type=str, default=None)

    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--full_finetune", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument(
        "--sft_eval_ratio",
        type=float,
        default=None,
        help="Stage-specific override for SFT eval split ratio. Defaults to --eval_ratio.",
    )
    parser.add_argument(
        "--grpo_eval_ratio",
        type=float,
        default=None,
        help="Stage-specific override for GRPO eval split ratio. Defaults to --eval_ratio.",
    )
    parser.add_argument(
        "--eval-from-train-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Derive the training-time eval split by slicing each task's train set "
            "instead of loading task eval files."
        ),
    )
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--max_eval_samples_per_task", type=int, default=0)
    parser.add_argument("--min_train_samples_per_task", type=int, default=0)
    parser.add_argument("--min_eval_samples_per_task", type=int, default=0)
    parser.add_argument(
        "--sft_max_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for SFT train cap per task. Defaults to --max_samples_per_task.",
    )
    parser.add_argument(
        "--sft_max_eval_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for SFT eval cap per task. Defaults to --max_eval_samples_per_task.",
    )
    parser.add_argument(
        "--sft_min_train_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for minimum SFT train split size per task. Defaults to --min_train_samples_per_task.",
    )
    parser.add_argument(
        "--sft_min_eval_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for minimum SFT eval split size per task. Defaults to --min_eval_samples_per_task.",
    )
    parser.add_argument(
        "--grpo_max_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for GRPO train cap per task. Defaults to --max_samples_per_task.",
    )
    parser.add_argument(
        "--grpo_max_eval_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for GRPO eval cap per task. Defaults to --max_eval_samples_per_task.",
    )
    parser.add_argument(
        "--grpo_min_train_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for minimum GRPO train split size per task. Defaults to --min_train_samples_per_task.",
    )
    parser.add_argument(
        "--grpo_min_eval_samples_per_task",
        type=int,
        default=None,
        help="Stage-specific override for minimum GRPO eval split size per task. Defaults to --min_eval_samples_per_task.",
    )
    parser.add_argument(
        "--prefer_subset_files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer *_subset.json files when present under task split directories.",
    )

    parser.add_argument("--sft_per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--sft_per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--sft_learning_rate", type=float, default=2e-5)
    parser.add_argument("--sft_num_train_epochs", type=float, default=1.0)
    parser.add_argument("--sft_max_steps", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    parser.add_argument("--grpo_per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--grpo_per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--sft_no_reason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the no-reasoning Alpaca prompt variant during the SFT stage.",
    )
    parser.add_argument(
        "--grpo_no_reason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the no-reasoning Alpaca prompt variant during the GRPO stage.",
    )
    parser.add_argument("--grpo_learning_rate", type=float, default=1e-6)
    parser.add_argument("--grpo_num_train_epochs", type=float, default=1.0)
    parser.add_argument("--grpo_max_steps", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--grpo_num_generations", type=int, default=8)
    parser.add_argument(
        "--grpo_num_return_sequences",
        type=int,
        default=None,
        help="Deprecated alias for --grpo_num_generations.",
    )
    parser.add_argument("--grpo_generation_batch_size", type=int, default=None)
    parser.add_argument("--grpo_steps_per_generation", type=int, default=None)

    parser.add_argument("--verify_gt_rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_check_samples_per_task", type=int, default=5)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--sft_logging_steps", type=int, default=None)
    parser.add_argument("--sft_save_steps", type=int, default=None)
    parser.add_argument("--grpo_logging_steps", type=int, default=None)
    parser.add_argument("--grpo_save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    add_context_window_argument(parser)

    parser.add_argument("--use_primal_dual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_constraint_target", type=float, default=0.98)
    parser.add_argument("--pd_cost_limit", type=float, default=None)
    parser.add_argument("--pd_lambda_init", type=float, default=0.0)
    parser.add_argument("--pd_dual_lr", type=float, default=0.05)
    parser.add_argument("--pd_dual_lr_up", type=float, default=None)
    parser.add_argument("--pd_dual_lr_down", type=float, default=None)
    parser.add_argument("--pd_dual_update_interval", type=int, default=1)
    parser.add_argument("--pd_violation_ema_beta", type=float, default=1.0)
    parser.add_argument("--pd_lambda_max", type=float, default=10.0)
    parser.add_argument("--pd_violation_tolerance", type=float, default=0.01)
    parser.add_argument("--pd_objective_weight", type=float, default=1.0)
    parser.add_argument("--pd_structure_weight", type=float, default=0.2)
    parser.add_argument("--pd_sync_across_ranks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_require_state_on_resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_strict_constraint_range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_force_scale_rewards_none", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pd_log_every", type=int, default=50)
    parser.add_argument(
        "--pd_soft_objective_for_infeasible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable partial objective credit on near-feasible outputs during primal-dual GRPO. "
            "This usually improves early-stage reward density."
        ),
    )
    parser.add_argument(
        "--pd_improved_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the opt-in constrained-GRPO improvements: normalized primal advantage, "
            "lexicographic feasible-first advantage, and CO component weighting. Use the "
            "individual pd_* flags for ablations when not using the bundled preset."
        ),
    )
    parser.add_argument(
        "--pd_normalize_primal_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize only the primal reward advantage within each GRPO prompt group.",
    )
    parser.add_argument("--pd_primal_advantage_scale", type=float, default=1.0)
    parser.add_argument("--pd_penalty_scale", type=float, default=1.0)
    parser.add_argument(
        "--pd_group_scalarization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Dual-Primal baseline A=z_G(r-p).",
    )
    parser.add_argument(
        "--pd_separate_z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the exact DUAL-SEPARATE-Z comparator A=z_G(r)-z_G(p), with "
            "the positive aggregate component penalty p."
        ),
    )
    parser.add_argument(
        "--pd_penalty_scale_start",
        type=float,
        default=None,
        help="Optional starting penalty scale for a linear warmup toward --pd_penalty_scale.",
    )
    parser.add_argument("--pd_penalty_scale_warmup_fraction", type=float, default=0.0)
    parser.add_argument(
        "--pd_positive_constraint_penalty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use only positive constraint violations in the primal-dual penalty term. "
            "This avoids rewarding already-feasible samples for being far below a constraint limit."
        ),
    )
    parser.add_argument(
        "--pd_lexicographic_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use feasible-first CO ranking: feasible completions are ranked by primal objective; "
            "infeasible completions are ranked by constraint severity."
        ),
    )
    parser.add_argument("--pd_lexicographic_feasible_margin", type=float, default=0.5)
    parser.add_argument(
        "--pd_component_profile",
        choices=[
            "none",
            "co_default",
            "co_op_strict_length",
            "co_op_strict_length_prizecap",
            "co_op_strict_length_margin",
            "co_jssp_valid_target_gap",
            "co_feasibility_lock_all",
            "co_jssp_tail_feas_lock_all",
            "co_mis_strict_edges",
            "co_mis_conflict_nodes",
            "co_mis_conflict_target_gap",
            "co_mis_conflict_target_gap_sizecap",
            "co_mis_conflict_target_gap_maximality",
            "co_mis_conflict_repair_target_gap",
            "co_mis_target_gap",
            "co_mis_repair_target",
        ],
        default="none",
        help="Optional per-task/per-cost tolerance and penalty-weight profile for primal-dual GRPO.",
    )
    parser.add_argument("--pd_enable_schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pd_stage_a_fraction", type=float, default=0.25)
    parser.add_argument("--pd_stage_b_fraction", type=float, default=0.70)
    parser.add_argument("--pd_stage_a_cost_limit", type=float, default=0.05)
    parser.add_argument("--pd_stage_b_cost_limit", type=float, default=0.02)
    parser.add_argument("--pd_stage_c_cost_limit", type=float, default=0.01)
    parser.add_argument("--pd_stage_a_dual_lr_scale", type=float, default=0.20)
    parser.add_argument("--pd_stage_b_dual_lr_scale", type=float, default=0.50)
    parser.add_argument("--pd_stage_c_dual_lr_scale", type=float, default=1.00)
    parser.add_argument("--pd_stage_c_constraint_threshold", type=float, default=0.95)
    parser.add_argument(
        "--grpo_train_size_bucket",
        choices=["all", "small", "medium", "large", "hardest"],
        default="all",
        help="Optional GRPO train-set instance-size bucket, using the same semantics as evaluation.",
    )
    parser.add_argument("--grpo_train_min_instance_size", type=int, default=0)
    parser.add_argument("--grpo_train_max_instance_size", type=int, default=0)
    parser.add_argument(
        "--grpo_train_task_multipliers",
        type=str,
        default="",
        help=(
            "Optional comma-separated GRPO task sampling multipliers, e.g. "
            "'cvrp:2,jssp:1.5,mis:0.5'."
        ),
    )
    return parser


def build_data_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit overlap between the SFT and GRPO corpora for mixed combinatorial optimization."
    )
    parser.add_argument("--tasks", type=str, default="tsp,cvrp,op,mis,jssp")
    parser.add_argument("--sft_data_root", type=str, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--grpo_data_root", type=str, default=DEFAULT_RL_ROOT)
    parser.add_argument(
        "--prefer_subset_files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer *_subset.json files when present under task split directories.",
    )
    parser.add_argument(
        "--fail_on_overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit with status 1 when any exact-row or instance-level overlap is detected.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional path to write the full audit report as JSON.",
    )
    return parser


def finalize_sft_args(args):
    args.tasks = parse_task_list(args.tasks)
    args.report_to = str(args.report_to).lower()
    args.wandb_project = (
        str(args.wandb_project).strip()
        if args.wandb_project
        else "grpo_primal_dual"
    )
    args.tokenizer_path = (str(args.tokenizer_path).strip() or None) if args.tokenizer_path is not None else None
    args.prompt_style = str(args.prompt_style).strip().lower()
    if int(args.context_window) < 0:
        raise ValueError("--context_window must be >= 0")
    if int(args.context_window) > 0:
        args.max_seq_length = int(args.context_window)
    if int(args.ddp_timeout) < 0:
        raise ValueError("--ddp_timeout must be >= 0")
    if args.weight_decay is not None and float(args.weight_decay) < 0.0:
        raise ValueError("--weight_decay must be >= 0")
    if args.warmup_steps is not None and int(args.warmup_steps) < 0:
        raise ValueError("--warmup_steps must be >= 0")
    if int(args.min_train_samples_per_task) < 0:
        raise ValueError("--min_train_samples_per_task must be >= 0")
    if int(args.min_eval_samples_per_task) < 0:
        raise ValueError("--min_eval_samples_per_task must be >= 0")
    _validate_train_selection_args(args)
    if int(args.per_device_train_batch_size) < 1:
        raise ValueError("--per_device_train_batch_size must be >= 1")
    args.effective_train_batch_size = effective_optimization_batch_size(
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
    )
    return args


def _validate_grpo_settings(
    *,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    num_generations,
    generation_batch_size,
    steps_per_generation,
    pd_cost_limit,
):
    resolved_generation_batch_size, resolved_steps_per_generation = resolve_generation_schedule(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        steps_per_generation=steps_per_generation,
    )
    effective_batch_size = effective_optimization_batch_size(
        per_device_train_batch_size,
        gradient_accumulation_steps,
    )
    resolved_global_batch_size = global_batch_size(per_device_train_batch_size)
    if effective_batch_size % int(num_generations) != 0:
        raise ValueError(
            "TRL GRPO requires the effective train batch size to be divisible by num_generations. "
            f"effective_train_batch_size={effective_batch_size}, num_generations={num_generations}"
        )
    if resolved_generation_batch_size % int(resolved_global_batch_size) != 0:
        raise ValueError(
            "TRL GRPO requires generation_batch_size to be divisible by the global batch size. "
            f"generation_batch_size={resolved_generation_batch_size}, "
            f"global_batch_size={resolved_global_batch_size} "
            f"(per_device_train_batch_size={per_device_train_batch_size}, world_size is inferred at runtime)."
        )
    if pd_cost_limit is not None and not (0.0 <= pd_cost_limit <= 1.0):
        raise ValueError(f"--pd_cost_limit must be in [0, 1], got {pd_cost_limit}")
    return resolved_generation_batch_size, resolved_steps_per_generation


def _validate_pd_schedule_args(args) -> None:
    if not (0.0 <= float(args.pd_stage_a_fraction) <= 1.0):
        raise ValueError(f"--pd_stage_a_fraction must be in [0, 1], got {args.pd_stage_a_fraction}")
    if not (0.0 <= float(args.pd_stage_b_fraction) <= 1.0):
        raise ValueError(f"--pd_stage_b_fraction must be in [0, 1], got {args.pd_stage_b_fraction}")
    if float(args.pd_stage_a_fraction) > float(args.pd_stage_b_fraction):
        raise ValueError(
            "--pd_stage_a_fraction must be <= --pd_stage_b_fraction. "
            f"Got {args.pd_stage_a_fraction} > {args.pd_stage_b_fraction}."
        )

    for name in (
        "pd_stage_a_cost_limit",
        "pd_stage_b_cost_limit",
        "pd_stage_c_cost_limit",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name} must be in [0, 1], got {value}")

    for name in (
        "pd_stage_a_dual_lr_scale",
        "pd_stage_b_dual_lr_scale",
        "pd_stage_c_dual_lr_scale",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be >= 0, got {value}")

    if not (0.0 <= float(args.pd_stage_c_constraint_threshold) <= 1.0):
        raise ValueError(
            "--pd_stage_c_constraint_threshold must be in [0, 1], "
            f"got {args.pd_stage_c_constraint_threshold}"
        )


def _parse_task_multiplier_map(spec: str | None, *, allowed_tasks: list[str], arg_name: str) -> dict[str, float]:
    if spec is None or not str(spec).strip():
        return {}

    out: dict[str, float] = {}
    allowed = set(allowed_tasks)
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--{arg_name} entries must use task:multiplier format, got {item!r}")
        task_name, raw_value = item.split(":", 1)
        task_name = task_name.strip().lower()
        if task_name not in allowed:
            raise ValueError(
                f"--{arg_name} references task {task_name!r}, but active tasks are {sorted(allowed)!r}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"--{arg_name} multiplier for {task_name!r} is not numeric: {raw_value!r}") from exc
        if value < 0.0:
            raise ValueError(f"--{arg_name} multiplier for {task_name!r} must be >= 0, got {value}")
        out[task_name] = value
    return out


def _apply_pd_improved_preset(args) -> None:
    improved = bool(getattr(args, "pd_improved_mode", False))
    if improved:
        if getattr(args, "pd_normalize_primal_advantage", None) is None:
            args.pd_normalize_primal_advantage = True
        if getattr(args, "pd_lexicographic_advantage", None) is None:
            args.pd_lexicographic_advantage = True
        if getattr(args, "pd_positive_constraint_penalty", None) is None:
            args.pd_positive_constraint_penalty = True
        if getattr(args, "pd_component_profile", "none") == "none":
            args.pd_component_profile = "co_default"

    if getattr(args, "pd_normalize_primal_advantage", None) is None:
        args.pd_normalize_primal_advantage = False
    if getattr(args, "pd_lexicographic_advantage", None) is None:
        args.pd_lexicographic_advantage = False
    if getattr(args, "pd_positive_constraint_penalty", None) is None:
        args.pd_positive_constraint_penalty = False


def _validate_pd_improved_args(args) -> None:
    _apply_pd_improved_preset(args)
    for name in (
        "pd_primal_advantage_scale",
        "pd_penalty_scale",
        "pd_lexicographic_feasible_margin",
        "pd_penalty_scale_warmup_fraction",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be >= 0, got {value}")
    if float(args.pd_penalty_scale_warmup_fraction) > 1.0:
        raise ValueError(
            "--pd_penalty_scale_warmup_fraction must be in [0, 1], "
            f"got {args.pd_penalty_scale_warmup_fraction}"
        )
    if getattr(args, "pd_penalty_scale_start", None) is not None:
        start_value = float(args.pd_penalty_scale_start)
        if start_value < 0.0:
            raise ValueError(f"--pd_penalty_scale_start must be >= 0, got {start_value}")

    if bool(getattr(args, "pd_separate_z", False)):
        if not bool(getattr(args, "use_primal_dual", False)):
            raise ValueError("--pd_separate_z requires --use_primal_dual")
        if bool(getattr(args, "pd_group_scalarization", False)):
            raise ValueError(
                "--pd_separate_z is mutually exclusive with --pd_group_scalarization"
            )
        if bool(getattr(args, "use_clasp", False)):
            raise ValueError("--pd_separate_z is mutually exclusive with --use_clasp")
        if bool(getattr(args, "pd_lexicographic_advantage", False)):
            raise ValueError(
                "--pd_separate_z is mutually exclusive with "
                "--pd_lexicographic_advantage"
            )
        if (
            abs(float(getattr(args, "pd_primal_advantage_scale", 1.0)) - 1.0) > 1e-12
            or abs(float(getattr(args, "pd_penalty_scale", 1.0)) - 1.0) > 1e-12
            or getattr(args, "pd_penalty_scale_start", None) is not None
            or abs(float(getattr(args, "pd_penalty_scale_warmup_fraction", 0.0))) > 1e-12
        ):
            raise ValueError(
                "--pd_separate_z requires unit primal/penalty scales and no penalty "
                "warmup so its formula remains exactly z_G(r)-z_G(p)"
            )


def _set_env_default(name: str, value: object, *, force: bool = False) -> None:
    if force or not str(os.environ.get(name, "") or "").strip():
        os.environ[name] = str(value)


def _strict_boolean_env(name: str) -> bool | None:
    raw_value = str(os.environ.get(name, "") or "").strip()
    if not raw_value:
        return None
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, or off/on; "
        f"got {raw_value!r}"
    )


def _canonical_env_override(
    args,
    *,
    env_name: str,
    arg_name: str,
    parser,
) -> None:
    """Apply one canonical experiment env override without hiding conflicts."""

    raw_value = str(os.environ.get(env_name, "") or "").strip()
    if not raw_value:
        return
    try:
        parsed = parser(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} has invalid value {raw_value!r}") from exc
    cli_name = f"--{arg_name}"
    current = getattr(args, arg_name)
    if _cli_option_present(cli_name) and parsed != current:
        raise ValueError(
            f"Conflicting canonical CLASP settings: {cli_name}={current!r} "
            f"but {env_name}={parsed!r}."
        )
    setattr(args, arg_name, parsed)


def _append_env_csv_values(name: str, values: list[str]) -> None:
    existing = [
        item.strip()
        for item in str(os.environ.get(name, "") or "").replace(";", ",").split(",")
        if item.strip()
    ]
    seen = {item for item in existing}
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            existing.append(value)
            seen.add(value)
    if existing:
        os.environ[name] = ",".join(existing)


def _cli_option_present(option_name: str) -> bool:
    """Return True when an argparse option was provided on the process argv."""
    for item in sys.argv[1:]:
        if item == option_name or item.startswith(f"{option_name}="):
            return True
    return False


def _validate_task_thresholds(spec: str, *, allowed_tasks: list[str], arg_name: str) -> dict[str, float]:
    if not str(spec or "").strip():
        return {}
    thresholds: dict[str, float] = {}
    allowed = set(allowed_tasks)
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--{arg_name} entries must use task:threshold format, got {item!r}")
        task_name, raw_value = item.split(":", 1)
        task_name = task_name.strip().lower()
        if task_name not in allowed:
            raise ValueError(
                f"--{arg_name} references task {task_name!r}, but active tasks are {sorted(allowed)!r}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"--{arg_name} threshold for {task_name!r} is not numeric: {raw_value!r}") from exc
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{arg_name} threshold for {task_name!r} must be in [0, 1], got {value}")
        thresholds[task_name] = value
    return thresholds


def _apply_clasp_strict_defaults(args, *, num_generations_attr: str) -> None:
    raw_targeting = bool(getattr(args, "clasp_raw_feasibility_targeting", False))
    canonical_env = _strict_boolean_env("CLASP_CANONICAL_SET")
    canonical_cli = bool(getattr(args, "clasp_canonical_set", False))
    canonical_cli_explicit = _cli_option_present(
        "--clasp_canonical_set"
    ) or _cli_option_present("--no-clasp_canonical_set")
    if canonical_env is not None and canonical_cli_explicit and canonical_env != canonical_cli:
        raise ValueError(
            "Conflicting canonical CLASP switches: --clasp_canonical_set and "
            f"CLASP_CANONICAL_SET={canonical_env!r}."
        )
    canonical_set = canonical_cli if canonical_env is None else canonical_env
    args.clasp_canonical_set = bool(canonical_set)
    fixed_null_env = _strict_boolean_env("CLASP_FIXED_NULL_SET")
    fixed_null_cli = bool(getattr(args, "clasp_fixed_null_set", False))
    fixed_null_cli_explicit = _cli_option_present(
        "--clasp_fixed_null_set"
    ) or _cli_option_present("--no-clasp_fixed_null_set")
    if (
        fixed_null_env is not None
        and fixed_null_cli_explicit
        and fixed_null_env != fixed_null_cli
    ):
        raise ValueError(
            "Conflicting fixed-null CLASP switches: --clasp_fixed_null_set and "
            f"CLASP_FIXED_NULL_SET={fixed_null_env!r}."
        )
    fixed_null_set = (
        fixed_null_cli if fixed_null_env is None else fixed_null_env
    )
    args.clasp_fixed_null_set = bool(fixed_null_set)
    if canonical_set and fixed_null_set:
        raise ValueError(
            "--clasp_canonical_set and --clasp_fixed_null_set are mutually exclusive."
        )
    if raw_targeting:
        args.use_clasp = True
        args.use_primal_dual = True
    if canonical_set or fixed_null_set:
        args.use_clasp = True
        args.use_primal_dual = True
    if not bool(getattr(args, "use_clasp", False)):
        return

    strict_defaults = bool(getattr(args, "clasp_strict_defaults", True)) or raw_targeting
    if not strict_defaults:
        return

    args.use_primal_dual = True
    if str(getattr(args, "pd_component_profile", "none") or "none").lower() == "none":
        args.pd_component_profile = "co_feasibility_lock_all"
    args.pd_constraint_target = 1.0
    args.pd_cost_limit = 0.0
    args.pd_soft_objective_for_infeasible = False
    args.pd_positive_constraint_penalty = True
    args.pd_lexicographic_advantage = True
    _set_env_default("CLASP_FEASIBILITY_GATED", "1", force=True)

    if raw_targeting:
        # The raw-feasibility preset deliberately uses the strict three-regime
        # advantage and a feasibility-only warmup.  BoN-aligned mode has a separate
        # controller and explicitly rejects warmup_steps > 0, so leaving its default
        # enabled makes this documented preset abort during trainer construction.
        _set_env_default("CLASP_BON_ALIGNED", "0", force=True)
        warmup_was_supplied = _cli_option_present("--clasp_feasibility_warmup_steps") or bool(
            str(os.environ.get("CLASP_FEASIBILITY_WARMUP_STEPS", "") or "").strip()
        )
        if not warmup_was_supplied and int(getattr(args, "clasp_feasibility_warmup_steps", 0) or 0) <= 0:
            args.clasp_feasibility_warmup_steps = 500
        if getattr(args, "clasp_self_replay", None) is None:
            args.clasp_self_replay = True
        if bool(getattr(args, "clasp_self_replay", False)):
            if float(getattr(args, "clasp_self_replay_imitation_weight", 0.0) or 0.0) <= 0.0:
                args.clasp_self_replay_imitation_weight = 0.05
            if float(getattr(args, "clasp_self_replay_contrastive_weight", 0.0) or 0.0) <= 0.0:
                args.clasp_self_replay_contrastive_weight = 0.02
        if getattr(args, "clasp_validation_gate_checkpoints", None) is None:
            args.clasp_validation_gate_checkpoints = True
        _set_env_default("CLASP_FEASIBILITY_GATE_MARGIN", "3.0")
        _set_env_default("CLASP_FEASIBILITY_GATE_ALL_INFEASIBLE_WEIGHT", "2.0")
        _set_env_default("CLASP_FEASIBILITY_GATE_MIXED_WEIGHT", "1.5")
        _set_env_default("CLASP_FEASIBILITY_WARMUP_FEASIBLE_MARGIN", "3.0")
        _set_env_default("CLASP_FEASIBILITY_WARMUP_VIOLATION_SCALE", "2.0")

    if getattr(args, "clasp_self_replay", None) is None:
        args.clasp_self_replay = False

    _set_env_default("CLASP_FEASIBILITY_WARMUP_STEPS", getattr(args, "clasp_feasibility_warmup_steps", 0), force=True)
    _set_env_default(
        "CLASP_FEASIBILITY_WARMUP_TARGETS",
        getattr(args, "clasp_feasibility_warmup_targets", ""),
        force=True,
    )
    _set_env_default(
        "CLASP_FEASIBILITY_WARMUP_HOLD_UNTIL_EVAL",
        "1" if bool(getattr(args, "clasp_feasibility_warmup_hold_until_eval", True)) else "0",
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_ENABLED",
        "1" if bool(getattr(args, "clasp_self_replay", False)) else "0",
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT",
        getattr(args, "clasp_self_replay_imitation_weight", 0.0),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT",
        getattr(args, "clasp_self_replay_contrastive_weight", 0.0),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_SAMPLE_RATE",
        getattr(args, "clasp_self_replay_sample_rate", 0.5),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_MAX_TOKENS",
        getattr(args, "clasp_self_replay_max_tokens", 512),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_MAX_KEYS",
        getattr(args, "clasp_self_replay_max_keys", 20000),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_MAX_PER_KEY",
        getattr(args, "clasp_self_replay_max_per_key", 4),
        force=True,
    )
    _set_env_default(
        "CLASP_SELF_REPLAY_MIN_BUFFER_SIZE",
        getattr(args, "clasp_self_replay_min_buffer_size", 8),
        force=True,
    )
    _set_env_default(
        "CLASP_VALIDATION_GATE_CHECKPOINTS",
        "1" if bool(getattr(args, "clasp_validation_gate_checkpoints", False)) else "0",
        force=True,
    )

    train_generations = getattr(args, "num_generations_train", None)
    if train_generations is not None:
        setattr(args, num_generations_attr, int(train_generations))


def _finalize_canonical_set_args(args) -> None:
    """Resolve and fail-closed validate exact canonical-utility set credit."""

    canonical_set = bool(getattr(args, "clasp_canonical_set", False))
    fixed_null_set = bool(getattr(args, "clasp_fixed_null_set", False))
    raw_receipt_path = str(
        getattr(args, "clasp_fixed_null_receipt_path", "") or ""
    ).strip()
    receipt_env = str(os.environ.get("CLASP_FIXED_NULL_RECEIPT_PATH", "") or "").strip()
    if raw_receipt_path and receipt_env:
        if os.path.realpath(raw_receipt_path) != os.path.realpath(receipt_env):
            raise ValueError(
                "--clasp_fixed_null_receipt_path conflicts with "
                "CLASP_FIXED_NULL_RECEIPT_PATH."
            )
    receipt_path = raw_receipt_path or receipt_env
    args.clasp_fixed_null_receipt_path = receipt_path or None

    _canonical_env_override(
        args,
        env_name="CLASP_POLICY_LENGTH_ALPHA",
        arg_name="clasp_policy_length_alpha",
        parser=float,
    )
    _canonical_env_override(
        args,
        env_name="CLASP_CANONICAL_DUAL_RECOVERY_COUPLING",
        arg_name="clasp_canonical_dual_recovery_coupling",
        parser=float,
    )
    for env_name, arg_name in (
        ("CLASP_CANONICAL_EXACT_SCALE", "clasp_canonical_exact_scale"),
        (
            "CLASP_CANONICAL_GRPO_ANCHOR_SCALE",
            "clasp_canonical_grpo_anchor_scale",
        ),
        (
            "CLASP_CANONICAL_QUALITY_ANCHOR_RATIO",
            "clasp_canonical_quality_anchor_ratio",
        ),
        (
            "CLASP_CANONICAL_DENSE_QUALITY_SCALE",
            "clasp_canonical_dense_quality_scale",
        ),
        (
            "CLASP_CANONICAL_DENSE_FEASIBILITY_SCALE",
            "clasp_canonical_dense_feasibility_scale",
        ),
        (
            "CLASP_CANONICAL_DENSE_VIOLATION_SCALE",
            "clasp_canonical_dense_violation_scale",
        ),
        (
            "CLASP_CANONICAL_DENSE_RECOVERY_SCALE",
            "clasp_canonical_dense_recovery_scale",
        ),
    ):
        _canonical_env_override(
            args,
            env_name=env_name,
            arg_name=arg_name,
            parser=float,
        )
    policy_length_alpha = float(args.clasp_policy_length_alpha)
    dual_recovery_coupling = float(args.clasp_canonical_dual_recovery_coupling)
    exact_scale = float(args.clasp_canonical_exact_scale)
    grpo_anchor_scale = float(args.clasp_canonical_grpo_anchor_scale)
    quality_anchor_ratio = float(args.clasp_canonical_quality_anchor_ratio)
    dense_scales = (
        float(args.clasp_canonical_dense_quality_scale),
        float(args.clasp_canonical_dense_feasibility_scale),
        float(args.clasp_canonical_dense_violation_scale),
        float(args.clasp_canonical_dense_recovery_scale),
    )
    if not math.isfinite(policy_length_alpha) or not 0.0 <= policy_length_alpha <= 1.0:
        raise ValueError("--clasp_policy_length_alpha must be finite and in [0, 1]")
    if not math.isfinite(dual_recovery_coupling) or dual_recovery_coupling < 0.0:
        raise ValueError(
            "--clasp_canonical_dual_recovery_coupling must be finite and >= 0"
        )
    if not math.isfinite(exact_scale) or exact_scale <= 0.0:
        raise ValueError("--clasp_canonical_exact_scale must be finite and > 0")
    if not math.isfinite(grpo_anchor_scale) or grpo_anchor_scale < 0.0:
        raise ValueError(
            "--clasp_canonical_grpo_anchor_scale must be finite and >= 0"
        )
    if not math.isfinite(quality_anchor_ratio) or quality_anchor_ratio < 0.0:
        raise ValueError(
            "--clasp_canonical_quality_anchor_ratio must be finite and >= 0"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in dense_scales):
        raise ValueError("canonical dense auxiliary scales must be finite and >= 0")

    if not (canonical_set or fixed_null_set):
        if receipt_path:
            raise ValueError(
                "--clasp_fixed_null_receipt_path is valid only with "
                "--clasp_fixed_null_set."
            )
        if abs(policy_length_alpha) > 1e-12:
            raise ValueError(
                "--clasp_policy_length_alpha is defined only for exact canonical "
                "or fixed-null set credit."
            )
        if dual_recovery_coupling > 0.0:
            raise ValueError(
                "--clasp_canonical_dual_recovery_coupling is defined only for "
                "exact canonical or fixed-null set credit."
            )
        if grpo_anchor_scale > 0.0:
            raise ValueError(
                "--clasp_canonical_grpo_anchor_scale requires "
                "--clasp_canonical_set."
            )
        if quality_anchor_ratio > 0.0:
            raise ValueError(
                "--clasp_canonical_quality_anchor_ratio requires "
                "--clasp_canonical_set."
            )
        if any(value > 0.0 for value in dense_scales):
            raise ValueError(
                "canonical dense auxiliary scales require --clasp_canonical_set."
            )
        return
    if canonical_set and fixed_null_set:
        raise ValueError(
            "--clasp_canonical_set and --clasp_fixed_null_set are mutually exclusive."
        )
    if canonical_set and receipt_path:
        raise ValueError(
            "Canonical leave-out credit rejects fixed-null receipt inputs."
        )
    if fixed_null_set:
        if grpo_anchor_scale > 0.0:
            raise ValueError(
                "GRPO-anchor credit is not defined for fixed-null set credit."
            )
        if quality_anchor_ratio > 0.0:
            raise ValueError(
                "Norm-matched canonical quality credit is not defined for "
                "fixed-null set credit."
            )
        if any(value > 0.0 for value in dense_scales):
            raise ValueError(
                "canonical dense auxiliary scales are not defined for fixed-null credit."
            )
        if not receipt_path:
            raise ValueError(
                "--clasp_fixed_null_set requires an explicit "
                "--clasp_fixed_null_receipt_path."
            )
        from clasp.set_grpo import load_fixed_null_score_receipt

        receipt = load_fixed_null_score_receipt(receipt_path)
        args.clasp_fixed_null_receipt_path = os.path.realpath(receipt_path)
        args.clasp_fixed_null_receipt_identity = dict(receipt.identity)
    else:
        args.clasp_fixed_null_receipt_identity = None
        if quality_anchor_ratio > 0.0 and grpo_anchor_scale <= 0.0:
            raise ValueError(
                "--clasp_canonical_quality_anchor_ratio requires a positive "
                "--clasp_canonical_grpo_anchor_scale."
            )
    if bool(getattr(args, "clasp_raw_feasibility_targeting", False)):
        raise ValueError(
            "Exact set credit is incompatible with "
            "--clasp_raw_feasibility_targeting."
        )

    _canonical_env_override(
        args,
        env_name="CLASP_DEPLOYMENT_SET_SIZE",
        arg_name="clasp_deployment_set_size",
        parser=int,
    )
    _canonical_env_override(
        args,
        env_name="CLASP_CANONICAL_RECOVERY_SCALE",
        arg_name="clasp_canonical_recovery_scale",
        parser=float,
    )
    _canonical_env_override(
        args,
        env_name="CLASP_CANONICAL_RECOVERY_TEMPERATURE",
        arg_name="clasp_canonical_recovery_temperature",
        parser=float,
    )
    _canonical_env_override(
        args,
        env_name="CLASP_CANONICAL_MAX_SUBSETS",
        arg_name="clasp_canonical_max_subsets",
        parser=int,
    )

    B = int(args.num_generations)
    K = int(args.clasp_deployment_set_size)
    if K <= 0:
        raise ValueError(
            "--clasp_deployment_set_size must be explicitly positive when "
            "exact set credit is enabled."
        )
    if K >= B:
        raise ValueError(
            "Exact CLASP set credit requires deployment K < rollout B: "
            f"K={K}, B={B}. The recommended schedule is B=K+1."
        )
    max_subsets = int(args.clasp_canonical_max_subsets)
    if max_subsets <= 0:
        raise ValueError("--clasp_canonical_max_subsets must be > 0")
    subset_count = math.comb(B, K)
    if subset_count > max_subsets:
        raise ValueError(
            "Canonical CLASP exact subset enumeration exceeds the configured cap: "
            f"comb({B}, {K})={subset_count} > {max_subsets}. Prefer B=K+1 or "
            "raise --clasp_canonical_max_subsets deliberately."
        )
    recovery_scale = float(args.clasp_canonical_recovery_scale)
    recovery_temperature = float(args.clasp_canonical_recovery_temperature)
    if not math.isfinite(recovery_scale) or recovery_scale < 0.0:
        raise ValueError("--clasp_canonical_recovery_scale must be finite and >= 0")
    if not math.isfinite(recovery_temperature) or recovery_temperature <= 0.0:
        raise ValueError(
            "--clasp_canonical_recovery_temperature must be finite and > 0"
        )

    bon_aligned_env = _strict_boolean_env("CLASP_BON_ALIGNED")
    if bon_aligned_env is False:
        raise ValueError("Exact CLASP set credit requires CLASP_BON_ALIGNED=1")
    frontier_env = _strict_boolean_env("CLASP_FRONTIER_CDF")
    if frontier_env is True:
        raise ValueError("Exact CLASP set credit is incompatible with CLASP_FRONTIER_CDF=1")
    if int(getattr(args, "clasp_feasibility_warmup_steps", 0) or 0) != 0:
        raise ValueError(
            "Exact CLASP set credit does not permit advantage replacement; set "
            "--clasp_feasibility_warmup_steps=0."
        )
    if bool(getattr(args, "clasp_self_replay", False)):
        raise ValueError(
            "Exact CLASP set-credit runs cannot bundle self replay; use "
            "--no-clasp_self_replay."
        )
    if args.grpo_loss_type is None:
        args.grpo_loss_type = "grpo"
    elif str(args.grpo_loss_type).strip().lower() != "grpo":
        raise ValueError(
            "Exact CLASP set credit requires --grpo_loss_type=grpo for an outer "
            "mean-reduced sequence loss."
        )

    # These optional system components are deliberately outside the core arm.
    # Disable their default-on presets rather than requiring noisy negative
    # flags for every canonical experiment.
    args.clasp_adaptive_length = False
    args.clasp_proper_dual = False
    args.clasp_self_replay = False

    _set_env_default(
        "CLASP_CANONICAL_SET",
        "1" if canonical_set else "0",
        force=True,
    )
    _set_env_default(
        "CLASP_FIXED_NULL_SET",
        "1" if fixed_null_set else "0",
        force=True,
    )
    if fixed_null_set:
        _set_env_default(
            "CLASP_FIXED_NULL_RECEIPT_PATH",
            args.clasp_fixed_null_receipt_path,
            force=True,
        )
    _set_env_default("CLASP_BON_ALIGNED", "1", force=True)
    _set_env_default("CLASP_FRONTIER_CDF", "0", force=True)
    _set_env_default("CLASP_DEPLOYMENT_SET_SIZE", K, force=True)
    _set_env_default(
        "CLASP_CANONICAL_RECOVERY_SCALE",
        recovery_scale,
        force=True,
    )
    _set_env_default(
        "CLASP_CANONICAL_DUAL_RECOVERY_COUPLING",
        dual_recovery_coupling,
        force=True,
    )
    _set_env_default(
        "CLASP_CANONICAL_RECOVERY_TEMPERATURE",
        recovery_temperature,
        force=True,
    )
    _set_env_default("CLASP_CANONICAL_MAX_SUBSETS", max_subsets, force=True)
    _set_env_default(
        "CLASP_POLICY_LENGTH_ALPHA",
        policy_length_alpha,
        force=True,
    )


def _validate_train_selection_args(args, *, prefix: str = "") -> None:
    min_name = f"{prefix}train_min_instance_size"
    max_name = f"{prefix}train_max_instance_size"
    min_value = int(getattr(args, min_name))
    max_value = int(getattr(args, max_name))
    if min_value < 0:
        raise ValueError(f"--{min_name} must be >= 0")
    if max_value < 0:
        raise ValueError(f"--{max_name} must be >= 0")
    if max_value > 0 and min_value > max_value:
        raise ValueError(f"--{min_name} cannot be larger than --{max_name}")


def finalize_grpo_args(args):
    args.tasks = parse_task_list(args.tasks)
    args.report_to = str(args.report_to).lower()
    args.wandb_project = (
        str(args.wandb_project).strip()
        if args.wandb_project
        else ("CLASP" if bool(getattr(args, "use_clasp", False)) else "grpo_primal_dual")
    )
    args.tokenizer_path = (str(args.tokenizer_path).strip() or None) if args.tokenizer_path is not None else None
    args.prompt_style = str(args.prompt_style).strip().lower()
    if int(args.context_window) < 0:
        raise ValueError("--context_window must be >= 0")
    if int(args.context_window) > 0:
        if int(args.max_completion_length) >= int(args.context_window):
            raise ValueError(
                "--max_completion_length must be strictly smaller than --context_window. "
                f"Got max_completion_length={args.max_completion_length}, context_window={args.context_window}."
            )
        args.max_prompt_length = int(args.context_window) - int(args.max_completion_length)
    args.max_seq_length = int(args.max_prompt_length) + int(args.max_completion_length)
    if int(args.min_train_samples_per_task) < 0:
        raise ValueError("--min_train_samples_per_task must be >= 0")
    if int(args.min_eval_samples_per_task) < 0:
        raise ValueError("--min_eval_samples_per_task must be >= 0")
    if len(args.tasks) > 1 and int(args.per_device_train_batch_size) < 2:
        raise ValueError(
            "Mixed-task GRPO requires per_device_train_batch_size >= 2 so every local batch can contain "
            "multiple task types."
        )
    if args.num_return_sequences is not None:
        if int(args.num_generations) != 8 and int(args.num_generations) != int(args.num_return_sequences):
            raise ValueError(
                "--num_return_sequences and --num_generations were both provided with different values. "
                "Use only one of them."
            )
        args.num_generations = int(args.num_return_sequences)
    if args.num_generations_train is not None and int(args.num_generations_train) < 1:
        raise ValueError("--num_generations_train must be >= 1")
    if args.num_generations_train is not None and args.num_return_sequences is not None:
        if int(args.num_generations_train) != int(args.num_return_sequences):
            raise ValueError(
                "--num_generations_train and --num_return_sequences both target training generation count; "
                "use only one value."
            )
    warmup_targets_explicit = _cli_option_present(
        "--clasp_feasibility_warmup_targets"
    ) or bool(str(os.environ.get("CLASP_FEASIBILITY_WARMUP_TARGETS", "") or "").strip())
    _apply_clasp_strict_defaults(args, num_generations_attr="num_generations")
    _finalize_canonical_set_args(args)
    _set_env_default(
        "CLASP_BON_SUCCESS_CREDIT",
        getattr(args, "clasp_bon_success_credit", "dense_rank"),
        force=True,
    )
    component_dual_recovery_scale = float(
        getattr(args, "clasp_component_dual_recovery_scale", 1.0)
    )
    if not math.isfinite(component_dual_recovery_scale) or component_dual_recovery_scale < 0.0:
        raise ValueError(
            "--clasp_component_dual_recovery_scale must be finite and >= 0"
        )
    _set_env_default(
        "CLASP_COMPONENT_DUAL_RECOVERY_SCALE",
        component_dual_recovery_scale,
        force=True,
    )
    _set_env_default(
        "CLASP_PROPER_DUAL_GRADIENT_MODE",
        getattr(args, "clasp_proper_dual_gradient_mode", "signed"),
        force=True,
    )
    _set_env_default(
        "CLASP_MIXED_ADAPTIVE_LENGTH_MODE",
        getattr(args, "clasp_mixed_adaptive_length_mode", "error"),
        force=True,
    )
    raw_warmup_targets = getattr(args, "clasp_feasibility_warmup_targets", "")
    if warmup_targets_explicit:
        args.clasp_feasibility_warmup_target_map = _validate_task_thresholds(
            raw_warmup_targets,
            allowed_tasks=args.tasks,
            arg_name="clasp_feasibility_warmup_targets",
        )
    else:
        # The bundled defaults cover all tasks, but publication runs are often
        # single-task specialists. Filter only the implicit default; an
        # explicitly supplied inactive task remains an error so typos cannot be
        # silently ignored.
        default_target_map = _validate_task_thresholds(
            raw_warmup_targets,
            allowed_tasks=("tsp", "cvrp", "op", "mis", "jssp"),
            arg_name="clasp_feasibility_warmup_targets",
        )
        active_tasks = {str(task).lower() for task in args.tasks}
        args.clasp_feasibility_warmup_target_map = {
            task: target
            for task, target in default_target_map.items()
            if task in active_tasks
        }
        args.clasp_feasibility_warmup_targets = ",".join(
            f"{task}:{target:g}"
            for task, target in args.clasp_feasibility_warmup_target_map.items()
        )
        _set_env_default(
            "CLASP_FEASIBILITY_WARMUP_TARGETS",
            args.clasp_feasibility_warmup_targets,
            force=True,
        )
    if int(getattr(args, "clasp_feasibility_warmup_steps", 0) or 0) < 0:
        raise ValueError("--clasp_feasibility_warmup_steps must be >= 0")
    for name in (
        "clasp_self_replay_imitation_weight",
        "clasp_self_replay_contrastive_weight",
        "clasp_self_replay_sample_rate",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be >= 0, got {value}")
    if float(args.clasp_self_replay_sample_rate) > 1.0:
        raise ValueError("--clasp_self_replay_sample_rate must be in [0, 1]")
    for name in (
        "clasp_self_replay_max_tokens",
        "clasp_self_replay_max_keys",
        "clasp_self_replay_max_per_key",
        "clasp_self_replay_min_buffer_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be >= 0")
    (
        args.resolved_generation_batch_size,
        args.resolved_steps_per_generation,
    ) = _validate_grpo_settings(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        steps_per_generation=args.steps_per_generation,
        pd_cost_limit=args.pd_cost_limit,
    )
    args.effective_train_batch_size = effective_optimization_batch_size(
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
    )
    _validate_pd_schedule_args(args)
    _validate_pd_improved_args(args)
    _validate_train_selection_args(args)
    args.train_task_multiplier_map = _parse_task_multiplier_map(
        args.train_task_multipliers,
        allowed_tasks=args.tasks,
        arg_name="train_task_multipliers",
    )
    return args


def finalize_pipeline_args(args):
    args.tasks = parse_task_list(args.tasks)
    args.report_to = str(args.report_to).lower()
    args.wandb_project = (
        str(args.wandb_project).strip()
        if args.wandb_project
        else "grpo_primal_dual"
    )
    args.tokenizer_path = (str(args.tokenizer_path).strip() or None) if args.tokenizer_path is not None else None
    args.prompt_style = str(args.prompt_style).strip().lower()
    args.sft_eval_ratio = float(args.eval_ratio if args.sft_eval_ratio is None else args.sft_eval_ratio)
    args.grpo_eval_ratio = float(args.eval_ratio if args.grpo_eval_ratio is None else args.grpo_eval_ratio)
    for ratio_name in ("sft_eval_ratio", "grpo_eval_ratio"):
        ratio_value = float(getattr(args, ratio_name))
        if not (0.0 < ratio_value < 1.0):
            raise ValueError(f"--{ratio_name} must be in (0, 1), got {ratio_value}")
    for name in (
        "max_samples_per_task",
        "max_eval_samples_per_task",
        "min_train_samples_per_task",
        "min_eval_samples_per_task",
    ):
        shared_value = int(getattr(args, name))
        sft_value = getattr(args, f"sft_{name}")
        grpo_value = getattr(args, f"grpo_{name}")
        setattr(args, f"sft_{name}", shared_value if sft_value is None else int(sft_value))
        setattr(args, f"grpo_{name}", shared_value if grpo_value is None else int(grpo_value))
    if args.sft_logging_steps is None:
        args.sft_logging_steps = args.logging_steps
    if args.sft_save_steps is None:
        args.sft_save_steps = args.save_steps
    if args.grpo_logging_steps is None:
        args.grpo_logging_steps = args.logging_steps
    if args.grpo_save_steps is None:
        args.grpo_save_steps = args.save_steps
    if int(args.context_window) < 0:
        raise ValueError("--context_window must be >= 0")
    if int(args.context_window) > 0:
        if int(args.max_completion_length) >= int(args.context_window):
            raise ValueError(
                "--max_completion_length must be strictly smaller than --context_window. "
                f"Got max_completion_length={args.max_completion_length}, context_window={args.context_window}."
            )
        args.max_seq_length = int(args.context_window)
        args.max_prompt_length = int(args.context_window) - int(args.max_completion_length)
    for name in (
        "min_train_samples_per_task",
        "min_eval_samples_per_task",
        "sft_max_samples_per_task",
        "sft_max_eval_samples_per_task",
        "sft_min_train_samples_per_task",
        "sft_min_eval_samples_per_task",
        "grpo_max_samples_per_task",
        "grpo_max_eval_samples_per_task",
        "grpo_min_train_samples_per_task",
        "grpo_min_eval_samples_per_task",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if len(args.tasks) > 1 and int(args.sft_per_device_train_batch_size) < 2:
        raise ValueError(
            "Mixed-task SFT requires sft_per_device_train_batch_size >= 2 so every local batch can contain "
            "multiple task types."
        )
    if len(args.tasks) > 1 and int(args.grpo_per_device_train_batch_size) < 2:
        raise ValueError(
            "Mixed-task GRPO requires grpo_per_device_train_batch_size >= 2 so every local batch can contain "
            "multiple task types."
        )
    if args.grpo_num_return_sequences is not None:
        if (
            int(args.grpo_num_generations) != 8
            and int(args.grpo_num_generations) != int(args.grpo_num_return_sequences)
        ):
            raise ValueError(
                "--grpo_num_return_sequences and --grpo_num_generations were both provided with different values. "
                "Use only one of them."
        )
        args.grpo_num_generations = int(args.grpo_num_return_sequences)
    total_grpo_window = int(args.max_prompt_length) + int(args.max_completion_length)
    args.max_seq_length = int(total_grpo_window)
    (
        args.grpo_resolved_generation_batch_size,
        args.grpo_resolved_steps_per_generation,
    ) = _validate_grpo_settings(
        per_device_train_batch_size=args.grpo_per_device_train_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        num_generations=args.grpo_num_generations,
        generation_batch_size=args.grpo_generation_batch_size,
        steps_per_generation=args.grpo_steps_per_generation,
        pd_cost_limit=args.pd_cost_limit,
    )
    args.sft_effective_train_batch_size = effective_optimization_batch_size(
        args.sft_per_device_train_batch_size,
        args.sft_gradient_accumulation_steps,
    )
    args.grpo_effective_train_batch_size = effective_optimization_batch_size(
        args.grpo_per_device_train_batch_size,
        args.grpo_gradient_accumulation_steps,
    )
    _validate_pd_schedule_args(args)
    _validate_pd_improved_args(args)
    _validate_train_selection_args(args, prefix="grpo_")
    args.grpo_train_task_multiplier_map = _parse_task_multiplier_map(
        args.grpo_train_task_multipliers,
        allowed_tasks=args.tasks,
        arg_name="grpo_train_task_multipliers",
    )
    return args


def parse_sft_args():
    return finalize_sft_args(build_sft_parser().parse_args())


def parse_grpo_args():
    return finalize_grpo_args(build_grpo_parser().parse_args())


def parse_pipeline_args():
    return finalize_pipeline_args(build_pipeline_parser().parse_args())


def parse_data_audit_args():
    args = build_data_audit_parser().parse_args()
    args.tasks = parse_task_list(args.tasks)
    return args
