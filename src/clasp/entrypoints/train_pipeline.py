"""Entry point: chained SFT then GRPO in one process (``clasp-pipeline``).

Runs supervised fine-tuning, merges the resulting adapter if needed, and starts GRPO from
it, resolving and reusing checkpoints between the two stages.
"""
from __future__ import annotations

import json
import os
from argparse import SUPPRESS, ArgumentParser, Namespace
from collections.abc import Callable

from clasp.config import build_grpo_parser, build_sft_parser, parse_pipeline_args
from clasp.entrypoints.train_grpo import run_grpo
from clasp.entrypoints.train_sft import run_sft
from clasp.models.loading import (
    load_model_and_tokenizer,
    merge_peft_checkpoint,
    path_looks_like_peft_checkpoint,
)
from clasp.models.training_fidelity import (
    PublicationFidelityError,
    checkpoint_fidelity_enabled,
    verify_publication_input_adapter,
)
from clasp.runtime import (
    build_grpo_run_name,
    build_sft_run_name,
    cleanup_torch_memory,
    find_latest_checkpoint_dir,
    format_stage_progress,
    inspect_stage_progress,
    is_primary_process,
    list_resumable_checkpoint_dirs,
    output_dir_has_saved_model,
    print_primary,
    resolve_base_model_name,
    resolve_model_artifact_name,
    resolve_output_dir,
    wait_for_condition,
    write_hyperparams,
)

try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except ImportError:
    def _elastic_record(fn):
        return fn


def _materialize_stage_output_from_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    *,
    dtype: str,
    padding_side: str,
) -> None:
    if not checkpoint_dir or checkpoint_dir == output_dir:
        return
    if checkpoint_fidelity_enabled():
        raise PublicationFidelityError(
            "Publication mode cannot post-hoc materialize a completed stage: the live "
            "terminal trainable policy is unavailable for checkpoint_fidelity.json. "
            "Resume/finalize the original training stage instead."
        )
    if is_primary_process():
        print_primary(
            "[Pipeline] Finalizing completed stage from checkpoint "
            f"{checkpoint_dir} into {output_dir}"
        )
        model, tokenizer, _ = load_model_and_tokenizer(
            checkpoint_dir,
            dtype,
            padding_side=padding_side,
        )
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        del model
        del tokenizer
        cleanup_torch_memory()
        return

    wait_for_condition(
        lambda: output_dir_has_saved_model(output_dir)
        and os.path.isfile(os.path.join(output_dir, "tokenizer_config.json")),
        description=f"materialized model artifacts in {output_dir}",
    )


def _resolve_best_stage_model_checkpoint(output_dir: str) -> str | None:
    resumable_checkpoint_dirs = list_resumable_checkpoint_dirs(
        output_dir,
        require_trainer_state=False,
    )
    if resumable_checkpoint_dirs:
        return resumable_checkpoint_dirs[0]
    return find_latest_checkpoint_dir(output_dir)


def _ensure_completed_stage_has_saved_model(progress: dict, *, dtype: str, padding_side: str) -> None:
    if not progress["completed"]:
        return
    if progress["has_saved_model"]:
        return
    checkpoint_dir = _resolve_best_stage_model_checkpoint(progress["output_dir"])
    if not checkpoint_dir:
        return
    _materialize_stage_output_from_checkpoint(
        checkpoint_dir,
        progress["output_dir"],
        dtype=dtype,
        padding_side=padding_side,
    )


def _primary_artifact_mtime(model_path: str) -> float:
    candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
    )
    mtimes = []
    for name in candidates:
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            mtimes.append(os.path.getmtime(path))
    return max(mtimes) if mtimes else 0.0


def _resolve_sft_source_model_path(sft_output_dir: str) -> str:
    if output_dir_has_saved_model(sft_output_dir):
        return sft_output_dir
    checkpoint_dir = _resolve_best_stage_model_checkpoint(sft_output_dir)
    if checkpoint_dir:
        return checkpoint_dir
    raise RuntimeError("SFT completed but no saved model or checkpoint is available.")


def _prepare_grpo_init_model(
    *,
    sft_output_dir: str,
    sft_source_model_path: str,
    dtype: str,
) -> str:
    if not path_looks_like_peft_checkpoint(sft_source_model_path):
        return sft_source_model_path

    if checkpoint_fidelity_enabled():
        verify_publication_input_adapter(
            sft_source_model_path,
            purpose="pipeline SFT-to-GRPO initialization adapter",
        )
        print_primary(
            "[Pipeline][PublicationFidelity] Reusing the receipted SFT adapter "
            "directly; an unreceipted merged initialization is forbidden."
        )
        return sft_source_model_path

    merged_dir = os.path.join(sft_output_dir, "merged_for_grpo")
    metadata_path = os.path.join(merged_dir, "merge_metadata.json")
    source_mtime = _primary_artifact_mtime(sft_source_model_path)
    expected_metadata = {
        "source_model_path": sft_source_model_path,
        "source_mtime": source_mtime,
        "dtype": dtype,
    }

    metadata = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metadata = None

    def merged_artifacts_match_expected() -> bool:
        if not output_dir_has_saved_model(merged_dir):
            return False
        if not os.path.isfile(os.path.join(merged_dir, "tokenizer_config.json")):
            return False
        if not os.path.isfile(metadata_path):
            return False
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                current_metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        return current_metadata == expected_metadata

    merged_ready = metadata == expected_metadata and merged_artifacts_match_expected()
    if merged_ready:
        print_primary(f"[Pipeline] Reusing merged SFT model for GRPO: {merged_dir}")
        return merged_dir

    if is_primary_process():
        print_primary(
            "[Pipeline] Merging SFT PEFT weights for GRPO initialization from "
            f"{sft_source_model_path} into {merged_dir}"
        )
        merge_peft_checkpoint(sft_source_model_path, merged_dir, dtype=dtype)
        write_hyperparams(merged_dir, {"merge_metadata": expected_metadata})
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(expected_metadata, handle, indent=2, ensure_ascii=True)
        cleanup_torch_memory()
    else:
        wait_for_condition(
            merged_artifacts_match_expected,
            description=f"merged SFT model artifacts in {merged_dir}",
        )
    return merged_dir


def _stage_defaults(build_parser: Callable[[], ArgumentParser]) -> dict[str, object]:
    """Default value for every option a stage parser declares.

    The pipeline exposes a subset of each stage's flags. Seeding a stage namespace with these
    defaults means a stage can read any of its own arguments, whether or not the pipeline
    surfaces it, instead of raising AttributeError. Explicit pipeline values override these.
    """
    return {
        action.dest: action.default
        for action in build_parser()._actions
        if action.dest not in {"help", SUPPRESS}
    }


def _make_sft_args(args, *, output_dir: str) -> Namespace:
    values: dict[str, object] = _stage_defaults(build_sft_parser)
    # Explicit pipeline values override the stage defaults.
    values.update(
        tasks=list(args.tasks),
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        prompt_style=args.prompt_style,
        data_root=args.sft_data_root,
        output_root=args.output_root,
        output_dir=output_dir,
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        no_reason=args.sft_no_reason,
        seed=args.seed,
        eval_ratio=args.sft_eval_ratio,
        eval_from_train_split=args.eval_from_train_split,
        max_samples_per_task=args.sft_max_samples_per_task,
        max_eval_samples_per_task=args.sft_max_eval_samples_per_task,
        min_train_samples_per_task=args.sft_min_train_samples_per_task,
        min_eval_samples_per_task=args.sft_min_eval_samples_per_task,
        prefer_subset_files=args.prefer_subset_files,
        per_device_train_batch_size=args.sft_per_device_train_batch_size,
        per_device_eval_batch_size=args.sft_per_device_eval_batch_size,
        gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        effective_train_batch_size=args.sft_effective_train_batch_size,
        learning_rate=args.sft_learning_rate,
        num_train_epochs=args.sft_num_train_epochs,
        max_steps=args.sft_max_steps,
        max_seq_length=args.max_seq_length,
        logging_steps=args.sft_logging_steps,
        save_steps=args.sft_save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )
    return Namespace(**values)


def _make_grpo_args(args, *, model_path: str, output_dir: str) -> Namespace:
    values: dict[str, object] = _stage_defaults(build_grpo_parser)
    # Explicit pipeline values override the stage defaults.
    values.update(
        tasks=list(args.tasks),
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        prompt_style=args.prompt_style,
        data_root=args.grpo_data_root,
        output_root=args.output_root,
        output_dir=output_dir,
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        no_reason=args.grpo_no_reason,
        seed=args.seed,
        eval_ratio=args.grpo_eval_ratio,
        eval_from_train_split=args.eval_from_train_split,
        max_samples_per_task=args.grpo_max_samples_per_task,
        max_eval_samples_per_task=args.grpo_max_eval_samples_per_task,
        min_train_samples_per_task=args.grpo_min_train_samples_per_task,
        min_eval_samples_per_task=args.grpo_min_eval_samples_per_task,
        prefer_subset_files=args.prefer_subset_files,
        per_device_train_batch_size=args.grpo_per_device_train_batch_size,
        per_device_eval_batch_size=args.grpo_per_device_eval_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        effective_train_batch_size=args.grpo_effective_train_batch_size,
        learning_rate=args.grpo_learning_rate,
        num_train_epochs=args.grpo_num_train_epochs,
        max_steps=args.grpo_max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.grpo_num_generations,
        generation_batch_size=args.grpo_generation_batch_size,
        resolved_generation_batch_size=args.grpo_resolved_generation_batch_size,
        steps_per_generation=args.grpo_steps_per_generation,
        resolved_steps_per_generation=args.grpo_resolved_steps_per_generation,
        verify_gt_rewards=args.verify_gt_rewards,
        reward_check_samples_per_task=args.reward_check_samples_per_task,
        logging_steps=args.grpo_logging_steps,
        save_steps=args.grpo_save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        use_primal_dual=args.use_primal_dual,
        pd_constraint_target=args.pd_constraint_target,
        pd_cost_limit=args.pd_cost_limit,
        pd_lambda_init=args.pd_lambda_init,
        pd_dual_lr=args.pd_dual_lr,
        pd_dual_lr_up=args.pd_dual_lr_up,
        pd_dual_lr_down=args.pd_dual_lr_down,
        pd_dual_update_interval=args.pd_dual_update_interval,
        pd_violation_ema_beta=args.pd_violation_ema_beta,
        pd_lambda_max=args.pd_lambda_max,
        pd_violation_tolerance=args.pd_violation_tolerance,
        pd_objective_weight=args.pd_objective_weight,
        pd_structure_weight=args.pd_structure_weight,
        pd_sync_across_ranks=args.pd_sync_across_ranks,
        pd_require_state_on_resume=args.pd_require_state_on_resume,
        pd_strict_constraint_range=args.pd_strict_constraint_range,
        pd_force_scale_rewards_none=args.pd_force_scale_rewards_none,
        pd_log_every=args.pd_log_every,
        pd_soft_objective_for_infeasible=args.pd_soft_objective_for_infeasible,
        pd_improved_mode=args.pd_improved_mode,
        pd_normalize_primal_advantage=args.pd_normalize_primal_advantage,
        pd_primal_advantage_scale=args.pd_primal_advantage_scale,
        pd_penalty_scale=args.pd_penalty_scale,
        pd_penalty_scale_start=args.pd_penalty_scale_start,
        pd_penalty_scale_warmup_fraction=args.pd_penalty_scale_warmup_fraction,
        pd_positive_constraint_penalty=args.pd_positive_constraint_penalty,
        pd_group_scalarization=args.pd_group_scalarization,
        pd_separate_z=args.pd_separate_z,
        pd_lexicographic_advantage=args.pd_lexicographic_advantage,
        pd_lexicographic_feasible_margin=args.pd_lexicographic_feasible_margin,
        pd_component_profile=args.pd_component_profile,
        pd_enable_schedule=args.pd_enable_schedule,
        pd_stage_a_fraction=args.pd_stage_a_fraction,
        pd_stage_b_fraction=args.pd_stage_b_fraction,
        pd_stage_a_cost_limit=args.pd_stage_a_cost_limit,
        pd_stage_b_cost_limit=args.pd_stage_b_cost_limit,
        pd_stage_c_cost_limit=args.pd_stage_c_cost_limit,
        pd_stage_a_dual_lr_scale=args.pd_stage_a_dual_lr_scale,
        pd_stage_b_dual_lr_scale=args.pd_stage_b_dual_lr_scale,
        pd_stage_c_dual_lr_scale=args.pd_stage_c_dual_lr_scale,
        pd_stage_c_constraint_threshold=args.pd_stage_c_constraint_threshold,
        train_size_bucket=args.grpo_train_size_bucket,
        train_min_instance_size=args.grpo_train_min_instance_size,
        train_max_instance_size=args.grpo_train_max_instance_size,
        train_task_multipliers=args.grpo_train_task_multipliers,
        train_task_multiplier_map=args.grpo_train_task_multiplier_map,
    )
    return Namespace(**values)


def _resolve_sft_output_dir(args) -> str:
    run_name = build_sft_run_name(
        resolve_base_model_name(args.model_path),
        prompt_style=args.prompt_style,
        no_reason=args.sft_no_reason,
        num_tasks=len(args.tasks),
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        per_device_train_batch_size=args.sft_per_device_train_batch_size,
        gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        num_train_epochs=args.sft_num_train_epochs,
        max_steps=args.sft_max_steps,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    return resolve_output_dir(
        output_root=args.output_root,
        stage="sft",
        run_name=run_name,
        explicit_output_dir=args.sft_output_dir,
    )


def _pd_variant_tag(args) -> str | None:
    if not bool(getattr(args, "use_primal_dual", False)):
        return None

    tags: list[str] = []
    if bool(getattr(args, "pd_improved_mode", False)):
        tags.append("pdimp")
    if bool(getattr(args, "pd_normalize_primal_advantage", False)):
        tags.append("normadv")
    if bool(getattr(args, "pd_positive_constraint_penalty", False)):
        tags.append("pospen")
    if bool(getattr(args, "pd_lexicographic_advantage", False)):
        tags.append("lex")
    if bool(getattr(args, "pd_group_scalarization", False)):
        tags.append("groupscalar")
    if bool(getattr(args, "pd_separate_z", False)):
        tags.append("separatez")
    component_profile = str(getattr(args, "pd_component_profile", "none") or "none").lower()
    if component_profile != "none":
        tags.append(f"cp{component_profile}")
    penalty_scale = float(getattr(args, "pd_penalty_scale", 1.0))
    penalty_scale_start = getattr(args, "pd_penalty_scale_start", None)
    if penalty_scale_start is not None:
        tags.append(f"ps{float(penalty_scale_start):g}to{penalty_scale:g}")
    elif abs(penalty_scale - 1.0) > 1e-12:
        tags.append(f"ps{penalty_scale:g}")
    return "-".join(tags) if tags else None


def _resolve_grpo_output_dir(args, *, model_path: str) -> str:
    epsilon = args.pd_cost_limit if args.pd_cost_limit is not None else args.pd_violation_tolerance
    run_name = build_grpo_run_name(
        resolve_model_artifact_name(model_path),
        prompt_style=args.prompt_style,
        no_reason=args.grpo_no_reason,
        num_tasks=len(args.tasks),
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        per_device_train_batch_size=args.grpo_per_device_train_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        num_generations=args.grpo_num_generations,
        generation_batch_size=args.grpo_resolved_generation_batch_size,
        learning_rate=args.grpo_learning_rate,
        num_train_epochs=args.grpo_num_train_epochs,
        max_steps=args.grpo_max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_primal_dual=args.use_primal_dual,
        epsilon=epsilon,
        pd_variant_tag=_pd_variant_tag(args),
    )
    return resolve_output_dir(
        output_root=args.output_root,
        stage="grpo",
        run_name=run_name,
        explicit_output_dir=args.grpo_output_dir,
    )


@_elastic_record
def main() -> None:
    args = parse_pipeline_args()
    print_primary(
        "[Runtime] clasp.entrypoints.train_pipeline active: "
        f"output_root={args.output_root}, report_to={args.report_to}"
    )
    print_primary(
        "[Pipeline] Unified sequence budget: "
        f"context_window={args.max_seq_length}, "
        f"SFT max_seq_length={args.max_seq_length}, "
        f"GRPO total_window={args.max_prompt_length + args.max_completion_length} "
        f"(prompt={args.max_prompt_length}, completion={args.max_completion_length})"
    )
    print_primary(
        "[Pipeline] Prompting config: "
        f"prompt_style={args.prompt_style}, "
        f"tokenizer_path={args.tokenizer_path or args.model_path}"
    )

    sft_output_dir = _resolve_sft_output_dir(args)
    sft_progress = inspect_stage_progress(sft_output_dir, "sft")
    print_primary(f"[Pipeline] SFT progress at {sft_output_dir}: {format_stage_progress(sft_progress)}")

    if sft_progress["completed"]:
        _ensure_completed_stage_has_saved_model(sft_progress, dtype=args.dtype, padding_side="right")
        print_primary("[Pipeline] SFT already complete; continuing with GRPO.")
        sft_source_model_path = _resolve_sft_source_model_path(sft_output_dir)
    else:
        print_primary("[Pipeline] SFT is incomplete; running or resuming SFT.")
        sft_source_model_path = run_sft(_make_sft_args(args, output_dir=sft_output_dir))

    sft_model_path = _prepare_grpo_init_model(
        sft_output_dir=sft_output_dir,
        sft_source_model_path=sft_source_model_path,
        dtype=args.dtype,
    )
    print_primary(f"[Pipeline] GRPO will initialize from: {sft_model_path}")

    cleanup_torch_memory()

    grpo_output_dir = _resolve_grpo_output_dir(args, model_path=sft_model_path)
    grpo_progress = inspect_stage_progress(grpo_output_dir, "grpo")
    print_primary(f"[Pipeline] GRPO progress at {grpo_output_dir}: {format_stage_progress(grpo_progress)}")

    if grpo_progress["completed"]:
        _ensure_completed_stage_has_saved_model(grpo_progress, dtype=args.dtype, padding_side="left")
        verify_publication_input_adapter(
            grpo_output_dir,
            purpose="completed pipeline GRPO adapter",
        )
        print_primary("[Pipeline] GRPO already complete; nothing left to run.")
        return

    print_primary("[Pipeline] GRPO is incomplete; running or resuming GRPO.")
    run_grpo(_make_grpo_args(args, model_path=sft_model_path, output_dir=grpo_output_dir))


if __name__ == "__main__":
    main()
