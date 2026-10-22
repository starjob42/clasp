"""Entry point for GRPO, constrained baselines, and canonical CLASP.

``--use_primal_dual`` selects a per-candidate constrained-GRPO baseline.
``--pd_separate_z`` selects the separately standardized comparator
``z_G(r)-z_G(p)`` with the positive aggregate component penalty ``p``.
``--use_clasp --clasp_canonical_set`` selects deployment-aligned canonical
leave-out credit with an explicit deployment set size smaller than the sampled
rollout batch; ``--clasp_fixed_null_set`` selects its receipted non-centered
same-utility comparator. Legacy Pareto, dense-rank, and systems-component paths remain
available only as labeled ablations.  Without either method flag this is plain
GRPO, although :mod:`clasp.entrypoints.train_grpo_standard` is the clearer
entry point for that configuration.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import warnings
from pathlib import Path

from transformers import TrainerCallback  # type: ignore[import-not-found]
from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

from clasp.config import parse_grpo_args
from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer
from clasp.data.loaders import build_mixed_task_sampler_for_trainer, load_mixed_grpo_dataset
from clasp.data.prompts import prompt_template_tag, resolve_prompt_style
from clasp.models.exact_peft_kl_post_step import reference_has_input_grad_hook
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_RECEIPT_FILENAME,
)
from clasp.models.loading import (
    _align_special_token_ids_with_tokenizer,
    assert_exact_paired_special_token_ids,
    build_lora_config,
    load_model_and_tokenizer,
    load_peft_causal_lm_with_explicit_adapter_autocast,
    lora_bias_from_env,
    lora_target_modules_from_env,
    peft_checkpoint_bias_policy,
    resolve_tokenizer_path,
)
from clasp.models.training_fidelity import (
    checkpoint_fidelity_enabled,
    checkpoint_fidelity_mode,
    install_publication_fidelity_callback,
    reject_beta_zero_exact_kl_sidecars,
    require_publication_lora_configuration,
    resolve_explicit_fixed_kl_reference,
    resolve_training_run_contract_identity,
    run_exact_peft_kl_post_step_audit,
    run_exact_peft_kl_runtime_preflight,
    run_publication_fidelity_preflight,
    verify_exact_peft_kl_source,
    verify_fixed_kl_reference_distributed,
    verify_publication_input_adapter,
    verify_publication_resume_checkpoint,
    write_final_publication_fidelity,
)
from clasp.runtime import (
    build_grpo_run_name,
    build_run_payload,
    cleanup_torch_memory,
    configure_wandb_env,
    enforce_grpo_context_limits,
    env_world_size,
    find_latest_checkpoint_dir,
    has_checkpoint_dir,
    is_likely_resume_load_error,
    is_primary_process,
    list_resumable_checkpoint_dirs,
    maybe_torch_distributed_barrier,
    print_primary,
    resolve_base_model_name,
    resolve_checkpoint_output_dir,
    resolve_model_artifact_name,
    resolve_output_dir,
    resolve_wandb_run_name,
    write_hyperparams,
    write_prefixed_trainer_logs,
    write_stage_status,
)

try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except ImportError:

    def _elastic_record(fn):
        return fn


from clasp.rewards.heads import (
    MultiTaskPrimalDualScheduleCallback,
    MultiTaskPrimalDualStateCallback,
    MultiTaskPrimalDualUpdateCallback,
    PrimalDualGRPOTrainer,
    build_mixed_reward_head,
    build_primal_dual_manager,
    build_task_triplets,
    maybe_load_primal_dual_state,
    save_multitask_primal_dual_state,
    verify_ground_truth_rewards,
)
from clasp.trainers.primal_dual_grpo import DUAL_SEPARATE_Z_FORMULA_ID


class ClaspPassAtKValidationGateCheckpointCallback(TrainerCallback):
    """Suppress gated saves unless every task meets deployment pass@K.

    Candidate feasibility is intentionally not accepted here: with K=8 even a
    modest per-candidate success rate can imply near-perfect group success, and
    gating the wrong random variable can hold feasibility-only training long
    after the deployment contract is satisfied.
    """

    def __init__(self, targets: dict[str, float]):
        self.targets = {
            str(task).lower(): float(target)
            for task, target in (targets or {}).items()
            if 0.0 <= float(target) <= 1.0
        }

    def _find_latest_metric(self, state, task_name: str) -> float | None:
        history = getattr(state, "log_history", None)
        if not isinstance(history, list):
            return None
        task_suffix = f"task/{task_name}/pass_at_k_feasibility_rate"
        for record in reversed(history):
            if not isinstance(record, dict):
                continue
            task_matches = [
                value
                for key, value in record.items()
                if str(key).startswith("eval") and str(key).endswith(task_suffix)
            ]
            if task_matches:
                try:
                    return float(task_matches[-1])
                except (TypeError, ValueError):
                    return None
        return None

    def _gate_save(self, state, control):
        if not self.targets or not bool(getattr(control, "should_save", False)):
            return control
        failures = []
        for task_name, target in sorted(self.targets.items()):
            observed = self._find_latest_metric(state, task_name)
            if observed is None or observed < target:
                failures.append((task_name, target, observed))
        if failures:
            control.should_save = False
            details = ", ".join(
                f"{task} pass@K={observed if observed is not None else 'missing'} "
                f"target={target:.4f}"
                for task, target, observed in failures
            )
            print_primary(f"[CLASP][PassAtKValidationGate] Suppressing checkpoint save: {details}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._gate_save(state, control)

    def on_save(self, args, state, control, **kwargs):
        return self._gate_save(state, control)


class StopAtGlobalStepCallback(TrainerCallback):
    """Stop a continuation cleanly at an absolute step and force a checkpoint.

    ``TrainingArguments.max_steps`` remains the final experiment horizon, so
    the optimizer and LR scheduler are identical across chained jobs.  This
    callback only creates clean scheduler boundaries for short wall-time jobs.
    """

    def __init__(self, stop_step: int):
        if int(stop_step) <= 0:
            raise ValueError("stop_step must be a positive integer")
        self.stop_step = int(stop_step)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


class _MixedBatchGRPOSamplerMixin:
    def _get_train_sampler(self, train_dataset=None):
        dataset = (
            train_dataset if train_dataset is not None else getattr(self, "train_dataset", None)
        )
        # GRPO needs its native repeated-prompt sampler when num_generations > 1.
        # Overriding it with a plain mixed-task sampler can break prompt grouping.
        num_generations = int(getattr(getattr(self, "args", None), "num_generations", 1) or 1)
        if num_generations > 1:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()

        sampler = build_mixed_task_sampler_for_trainer(dataset, self.args)
        if sampler is None:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()
        return sampler


class _OptimizerResumeFallbackMixin:
    def _rebase_loaded_scheduler_for_extension(self, checkpoint):
        """Install the declared phase-2 schedule without resetting the optimizer.

        A normal chained job keeps one fixed ``max_steps`` value, so loading the
        saved optimizer and scheduler is sufficient.  A post-horizon extension is
        different: the source checkpoint was saved with a zero LR at its former
        final step.  The extension therefore supplies its own low-LR warmup and
        linear decay.  The lambda must be reconstructed on every resume because
        PyTorch does not serialize LambdaLR callables; later resumes additionally
        prove that the reconstructed LR exactly matches the saved optimizer LR.

        This behavior is fail-closed and opt-in.
        """

        expected_raw = str(os.environ.get("GRPO_REBASE_LOADED_SCHEDULER_AT_STEP", "")).strip()
        if not expected_raw:
            return
        if not expected_raw.isdigit() or int(expected_raw) <= 0:
            raise RuntimeError("GRPO_REBASE_LOADED_SCHEDULER_AT_STEP must be a positive integer")
        expected_step = int(expected_raw)
        schedule_mode = (
            str(os.environ.get("GRPO_EXTENSION_SCHEDULE_MODE", "warmup_linear")).strip().lower()
        )
        try:
            extension_base_step = int(os.environ["GRPO_EXTENSION_BASE_STEP"])
            extension_final_step = int(os.environ["GRPO_EXTENSION_FINAL_STEP"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Extension scheduler requires valid BASE_STEP and FINAL_STEP settings"
            ) from exc
        extension_steps = extension_final_step - extension_base_step
        if (
            extension_base_step <= 0
            or extension_final_step <= extension_base_step
            or not (extension_base_step <= expected_step < extension_final_step)
        ):
            raise RuntimeError(
                "Invalid extension schedule: "
                f"base={extension_base_step}, resume={expected_step}, "
                f"final={extension_final_step}, mode={schedule_mode}"
            )
        extension_warmup_steps = None
        extension_peak_lr = None
        extension_start_lr = None
        if schedule_mode == "warmup_linear":
            try:
                extension_warmup_steps = int(os.environ["GRPO_EXTENSION_WARMUP_STEPS"])
                extension_peak_lr = float(os.environ["GRPO_EXTENSION_PEAK_LR"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "warmup_linear requires valid WARMUP_STEPS and PEAK_LR settings"
                ) from exc
            if (
                not (0 < extension_warmup_steps < extension_steps)
                or not math.isfinite(extension_peak_lr)
                or extension_peak_lr <= 0.0
            ):
                raise RuntimeError(
                    "Invalid warmup_linear extension schedule: "
                    f"warmup={extension_warmup_steps}, peak_lr={extension_peak_lr}"
                )
        elif schedule_mode == "anchored_linear_decay":
            try:
                extension_start_lr = float(os.environ["GRPO_EXTENSION_START_LR"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "anchored_linear_decay requires a valid START_LR setting"
                ) from exc
            if not math.isfinite(extension_start_lr) or extension_start_lr <= 0.0:
                raise RuntimeError(f"Invalid anchored extension start LR: {extension_start_lr}")
        else:
            raise RuntimeError(f"Unsupported extension schedule mode: {schedule_mode}")
        trainer_state_path = os.path.join(str(checkpoint), "trainer_state.json")
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as handle:
                trainer_state_payload = json.load(handle)
            checkpoint_step = int(trainer_state_payload.get("global_step", -1))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot validate extension checkpoint state: {trainer_state_path}"
            ) from exc
        if checkpoint_step != expected_step:
            raise RuntimeError(
                "Extension scheduler rebase step does not match checkpoint: "
                f"expected={expected_step}, checkpoint={checkpoint_step}"
            )

        final_horizon = int(getattr(getattr(self, "args", None), "max_steps", -1))
        if final_horizon != extension_final_step:
            raise RuntimeError(
                "Extension scheduler final step does not match max_steps: "
                f"extension_final={extension_final_step}, max_steps={final_horizon}"
            )
        loaded_scheduler = getattr(self, "lr_scheduler", None)
        if loaded_scheduler is None:
            raise RuntimeError("Extension scheduler rebase requested without a scheduler")
        scheduler = getattr(loaded_scheduler, "scheduler", loaded_scheduler)
        optimizer = getattr(scheduler, "optimizer", None)
        if optimizer is None:
            optimizer = getattr(self, "optimizer", None)
        if optimizer is None or not getattr(optimizer, "param_groups", None):
            raise RuntimeError("Extension scheduler rebase requested without an optimizer")
        if int(getattr(scheduler, "last_epoch", -1)) != expected_step:
            raise RuntimeError(
                "Loaded scheduler step does not match extension checkpoint: "
                f"last_epoch={getattr(scheduler, 'last_epoch', None)}, "
                f"checkpoint={expected_step}"
            )

        base_lrs = [float(value) for value in getattr(scheduler, "base_lrs", [])]
        if len(base_lrs) != len(optimizer.param_groups) or any(
            not math.isfinite(value) or value <= 0.0 for value in base_lrs
        ):
            raise RuntimeError(f"Invalid extension scheduler base LRs: {base_lrs}")
        if not hasattr(scheduler, "lr_lambdas"):
            raise RuntimeError("Extension schedule requires a LambdaLR-compatible scheduler")

        def make_extension_lambda(base_lr):
            if schedule_mode == "anchored_linear_decay":
                start_scale = extension_start_lr / base_lr

                def anchored_linear_decay(global_step):
                    remaining = max(0, extension_final_step - int(global_step))
                    return start_scale * float(remaining) / float(extension_steps)

                return anchored_linear_decay

            peak_scale = extension_peak_lr / base_lr

            def extension_lambda(global_step):
                relative_step = int(global_step) - extension_base_step
                if relative_step <= 0:
                    return 0.0
                if relative_step < extension_warmup_steps:
                    return peak_scale * float(relative_step) / float(extension_warmup_steps)
                remaining = max(0, extension_final_step - int(global_step))
                decay_steps = extension_steps - extension_warmup_steps
                return peak_scale * float(remaining) / float(decay_steps)

            return extension_lambda

        scheduler.lr_lambdas = [make_extension_lambda(base_lr) for base_lr in base_lrs]
        before_lrs = [float(group.get("lr", float("nan"))) for group in optimizer.param_groups]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scheduler.step(expected_step)
            except TypeError as exc:
                raise RuntimeError(
                    "Loaded scheduler does not support an absolute-step extension rebase"
                ) from exc
        # Match the state convention in checkpoints produced by Trainer:
        # scheduler.last_epoch == global_step and _step_count == global_step + 1.
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = expected_step + 1
        after_lrs = [float(group.get("lr", float("nan"))) for group in optimizer.param_groups]
        if int(getattr(scheduler, "last_epoch", -1)) != expected_step:
            raise RuntimeError("Scheduler moved to the wrong step during extension rebase")
        if any(not math.isfinite(value) or value < 0.0 for value in after_lrs):
            raise RuntimeError(
                f"Extension scheduler produced an invalid learning rate: {after_lrs}"
            )
        for before, after in zip(before_lrs, after_lrs):
            if not math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(
                    "Extension scheduler continuity mismatch on resume: "
                    f"before={before_lrs}, reconstructed={after_lrs}"
                )
        print_primary(
            "[GRPO] Installed explicit continuation scheduler "
            f"base={extension_base_step}, resume={expected_step}, "
            f"final={extension_final_step}, mode={schedule_mode}, "
            f"warmup={extension_warmup_steps}, peak_lr={extension_peak_lr}, "
            f"start_lr={extension_start_lr}, lr_before={before_lrs}, "
            f"lr_after={after_lrs}; optimizer moments preserved"
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint and checkpoint_fidelity_enabled():
            run_publication_fidelity_preflight(
                self,
                phase=f"post_resume_model_load:{os.path.abspath(str(checkpoint))}",
            )
        try:
            result = super()._load_optimizer_and_scheduler(checkpoint)
            self._rebase_loaded_scheduler_for_extension(checkpoint)
            return result
        except ValueError as exc:
            message = str(exc)
            if "parameter group that doesn't match the size of optimizer's group" not in message:
                raise
            strict_resume = str(
                os.environ.get("GRPO_STRICT_OPTIMIZER_RESUME", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if strict_resume or checkpoint_fidelity_enabled():
                raise RuntimeError(
                    "Optimizer state is incompatible with the current trainable "
                    "parameter groups and GRPO_STRICT_OPTIMIZER_RESUME=1; refusing "
                    "to silently restart the optimizer/scheduler."
                ) from exc

            global_step = 0
            if checkpoint:
                trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
                try:
                    with open(trainer_state_path, "r", encoding="utf-8") as handle:
                        trainer_state_payload = json.load(handle)
                    global_step = int(trainer_state_payload.get("global_step", 0) or 0)
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    global_step = 0

            print_primary(
                "[GRPO][Warn] Optimizer state is incompatible with current trainable parameter groups; "
                "resuming with fresh optimizer/scheduler state."
            )
            if global_step > 0 and getattr(self, "lr_scheduler", None) is not None:
                try:
                    self.lr_scheduler.last_epoch = global_step - 1
                    if hasattr(self.lr_scheduler, "_step_count"):
                        self.lr_scheduler._step_count = max(1, global_step)
                except Exception:
                    pass
            return


class MixedBatchGRPOTrainer(
    _OptimizerResumeFallbackMixin, _MixedBatchGRPOSamplerMixin, GRPOTrainer
):
    pass


class MixedBatchVerifiedTeacherGRPOTrainer(
    _OptimizerResumeFallbackMixin, _MixedBatchGRPOSamplerMixin, VerifiedTeacherGRPOTrainer
):
    pass


def _external_baseline_trainer_class():
    # The native TRL sampler is necessary at G=1 too: it repeats rollout
    # buffers for accumulation/reuse. The generic mixed-task sampler does not.
    from clasp.trainers.external_baseline_grpo import ExternalBaselineGRPOTrainer

    class NativeExternalBaselineGRPOTrainer(
        _OptimizerResumeFallbackMixin, ExternalBaselineGRPOTrainer
    ):
        pass

    return NativeExternalBaselineGRPOTrainer


def _feasible_quality_trainer_class():
    from clasp.trainers.feasible_quality_grpo import FeasibleQualityGRPOTrainer

    class NativeFeasibleQualityGRPOTrainer(
        _OptimizerResumeFallbackMixin, FeasibleQualityGRPOTrainer
    ):
        pass

    return NativeFeasibleQualityGRPOTrainer


class MixedBatchPrimalDualGRPOTrainer(
    _OptimizerResumeFallbackMixin,
    _MixedBatchGRPOSamplerMixin,
    PrimalDualGRPOTrainer,
):
    pass


def _pd_variant_tag(args) -> str | None:
    if not bool(getattr(args, "use_primal_dual", False)):
        return None

    tags: list[str] = []
    if bool(getattr(args, "use_clasp", False)):
        tags.append("clasp")
        if bool(getattr(args, "clasp_raw_feasibility_targeting", False)):
            tags.append("rawfeas")
        if bool(getattr(args, "clasp_pareto_anchor", True)):
            tags.append(f"pa{getattr(args, 'clasp_pareto_tau', 4.0)}")
            tags.append(f"pc{getattr(args, 'clasp_pareto_center', 0.5)}")
        else:
            tags.append("nopareto")
        warmup_steps = int(getattr(args, "clasp_feasibility_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            tags.append(f"fw{warmup_steps}")
        if bool(getattr(args, "clasp_self_replay", False)):
            tags.append("selfreplay")
        if bool(getattr(args, "clasp_proper_dual", True)):
            tags.append(f"dual{getattr(args, 'clasp_dual_update_interval', 4)}")
        else:
            tags.append("legacydual")
        if bool(getattr(args, "clasp_adaptive_length", True)):
            tags.append(
                "alen"
                f"q{getattr(args, 'clasp_adaptive_length_quantile', 0.99)}"
                f"f{getattr(args, 'clasp_adaptive_length_floor', 128)}"
            )
        else:
            tags.append("fixedlen")
        raw_gap_penalty = os.environ.get("CLASP_PRIMAL_SCORE_GAP_PENALTY", "0.0")
        try:
            raw_gap_penalty_value = float(raw_gap_penalty)
        except ValueError:
            raw_gap_penalty_value = 0.0
        if raw_gap_penalty_value > 0.0:
            tags.append(f"rgap{raw_gap_penalty_value:g}")
        raw_hard_group_weight = os.environ.get("CLASP_HARD_GROUP_WEIGHT_SCALE", "0.0")
        try:
            hard_group_weight_value = float(raw_hard_group_weight)
        except ValueError:
            hard_group_weight_value = 0.0
        if hard_group_weight_value > 0.0:
            raw_hard_group_threshold = os.environ.get("CLASP_HARD_GROUP_WEIGHT_THRESHOLD", "0.0")
            try:
                hard_group_threshold_value = float(raw_hard_group_threshold)
            except ValueError:
                hard_group_threshold_value = 0.0
            tags.append(f"hgw{hard_group_weight_value:g}t{hard_group_threshold_value:g}")
        raw_tail_enabled = (
            str(os.environ.get("CLASP_TSP_TAIL_GAP_CONSTRAINT", "") or "").strip().lower()
        )
        if raw_tail_enabled in {"1", "true", "yes", "on"}:
            raw_tail_gap = os.environ.get(
                "CLASP_TSP_TAIL_GAP_CONSTRAINT_GAP", ""
            ) or os.environ.get(
                "CLASP_TSP_TAIL_GAP",
                "0.0",
            )
            try:
                tail_gap_value = float(raw_tail_gap)
            except ValueError:
                tail_gap_value = 0.0
            if tail_gap_value > 0.0:
                tags.append(f"tspTail{tail_gap_value:g}")
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


def resolve_grpo_data_seed(policy_seed: int) -> int:
    """Resolve an optional data-row seed independently of policy randomness."""

    raw_value = os.environ.get("GRPO_DATA_SEED")
    if raw_value is None or not raw_value.strip():
        return int(policy_seed)
    try:
        data_seed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"GRPO_DATA_SEED must be an integer, got {raw_value!r}") from exc
    if data_seed < 0:
        raise ValueError(f"GRPO_DATA_SEED must be nonnegative, got {data_seed}")
    return data_seed


def fixed_horizon_exit_is_terminal(global_step: int, max_steps: int) -> bool:
    """Classify a chunk exit against the immutable full training horizon."""

    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise RuntimeError(f"invalid GRPO global_step: {global_step!r}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise RuntimeError(f"invalid GRPO max_steps: {max_steps!r}")
    if global_step > max_steps:
        raise RuntimeError(
            "GRPO advanced beyond its contracted fixed horizon: "
            f"step={global_step}, max_steps={max_steps}"
        )
    return global_step == max_steps


def validate_fixed_horizon_preflight(
    *,
    max_steps: int,
    runtime_world_size: int,
    contract_identity: dict[str, object] | None,
) -> None:
    """Fail before Trainer construction when runtime and sealed horizon differ."""

    if contract_identity is None:
        return
    contracted_steps = contract_identity.get("fixed_horizon_max_steps")
    contracted_world_size = contract_identity.get("fixed_horizon_world_size")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= 0
        or isinstance(runtime_world_size, bool)
        or not isinstance(runtime_world_size, int)
        or runtime_world_size <= 0
        or isinstance(contracted_steps, bool)
        or not isinstance(contracted_steps, int)
        or contracted_steps <= 0
        or isinstance(contracted_world_size, bool)
        or not isinstance(contracted_world_size, int)
        or contracted_world_size <= 0
    ):
        raise RuntimeError("invalid immutable GRPO fixed-horizon preflight values")
    if max_steps != contracted_steps or runtime_world_size != contracted_world_size:
        raise RuntimeError(
            "runtime GRPO horizon differs from the immutable run contract: "
            f"max_steps={max_steps}/{contracted_steps}, "
            f"world_size={runtime_world_size}/{contracted_world_size}"
        )


def _load_frozen_exact_peft_reference(
    source_binding: dict[str, object],
    dtype: str,
    *,
    tokenizer,
):
    """Independently load the same base+adapter without changing rollout RNG."""

    import numpy as np
    import torch

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        resolved_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = None
        if torch.cuda.is_available() and env_world_size() <= 1:
            device_map = {"": f"cuda:{local_rank}"}
        base_path = str(source_binding["base_model_path"])
        reference = load_peft_causal_lm_with_explicit_adapter_autocast(
            str(source_binding["adapter_path"]),
            dtype=resolved_dtype,
            device_map=device_map,
            is_trainable=False,
            trust_remote_code=False,
            local_files_only=True,
            expected_base_model_path=base_path,
        )
        _align_special_token_ids_with_tokenizer(reference, tokenizer)
        # PEFT 0.19 leaves bias="all" base-layer biases trainable even when
        # is_trainable=False.  Freeze the complete object and verify below.
        reference.requires_grad_(False)
        reference.eval()
        if any(parameter.requires_grad for parameter in reference.parameters()):
            raise RuntimeError("exact PEFT KL reference could not be fully frozen")
        if reference_has_input_grad_hook(reference):
            raise RuntimeError("exact PEFT KL reference has an input-requires-grad hook")
        return reference
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def run_grpo(args, *, runtime_profile=None) -> str:
    runtime_profile_receipt = None
    if runtime_profile is not None:
        runtime_profile.validate_args(args)
        runtime_profile_receipt = runtime_profile.metadata()
        runtime_profile_receipt["trainer_implementation"] = (
            runtime_profile.validate_trainer_implementation(GRPOTrainer)
        )

    data_seed = resolve_grpo_data_seed(args.seed)
    if data_seed != int(args.seed):
        print_primary(
            "[GRPO] Separating policy and data randomness: "
            f"policy_seed={args.seed}, data_seed={data_seed}"
        )

    if args.verify_gt_rewards:
        verify_ground_truth_rewards(
            tasks=args.tasks,
            data_root=args.data_root,
            max_samples_per_task=args.reward_check_samples_per_task,
            prefer_subset_files=args.prefer_subset_files,
            no_reason=args.no_reason,
        )

    initialization_fidelity = verify_publication_input_adapter(
        args.model_path,
        purpose="GRPO initialization adapter",
    )

    resolved_tokenizer_path = resolve_tokenizer_path(args.model_path, args.tokenizer_path)
    model, tokenizer, model_is_peft = load_model_and_tokenizer(
        args.model_path,
        args.dtype,
        padding_side="left",
        tokenizer_path=args.tokenizer_path,
    )
    requested_prompt_style = str(args.prompt_style).strip().lower()
    prompt_style = resolve_prompt_style(
        requested_prompt_style,
        model_path=args.model_path,
        tokenizer=tokenizer,
    )
    strict_chat_prompt = requested_prompt_style == "alpaca_chat"
    prompt_template = prompt_template_tag(prompt_style, no_reason=bool(args.no_reason))
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = True
        print_primary(
            "[GRPO] Enabling generation KV cache for rollout speed while keeping "
            "model.config.use_cache disabled for training forwards."
        )

    train_dataset, eval_dataset = load_mixed_grpo_dataset(
        tasks=args.tasks,
        data_root=args.data_root,
        eval_ratio=args.eval_ratio,
        eval_from_train_split=args.eval_from_train_split,
        seed=data_seed,
        max_samples_per_task=max(0, int(args.max_samples_per_task)),
        max_eval_samples_per_task=max(0, int(args.max_eval_samples_per_task)),
        min_train_samples_per_task=max(0, int(args.min_train_samples_per_task)),
        min_eval_samples_per_task=max(0, int(args.min_eval_samples_per_task)),
        prefer_subset_files=args.prefer_subset_files,
        no_reason=args.no_reason,
        prompt_style=prompt_style,
        tokenizer=tokenizer,
        strict_chat_prompt=strict_chat_prompt,
        train_size_bucket=getattr(args, "train_size_bucket", "all"),
        train_min_instance_size=getattr(args, "train_min_instance_size", 0),
        train_max_instance_size=getattr(args, "train_max_instance_size", 0),
        train_task_multiplier_map=getattr(args, "train_task_multiplier_map", {}),
    )
    if runtime_profile is not None:
        runtime_profile_receipt["dataset_validation"] = {
            "train": runtime_profile.validate_dataset(train_dataset, label="train"),
            "eval": runtime_profile.validate_dataset(eval_dataset, label="eval"),
        }

    max_prompt_length, max_completion_length, context_info = enforce_grpo_context_limits(
        args.max_prompt_length,
        args.max_completion_length,
        model,
        tokenizer,
    )

    base_model_name = resolve_base_model_name(args.model_path)
    init_model_artifact_name = resolve_model_artifact_name(args.model_path)
    epsilon = args.pd_cost_limit if args.pd_cost_limit is not None else args.pd_violation_tolerance
    run_name = build_grpo_run_name(
        init_model_artifact_name,
        prompt_style=requested_prompt_style,
        no_reason=args.no_reason,
        num_tasks=len(args.tasks),
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.resolved_generation_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_primal_dual=args.use_primal_dual,
        epsilon=epsilon,
        pd_variant_tag=_pd_variant_tag(args),
    )
    wandb_run_name = resolve_wandb_run_name(run_name)
    output_dir = resolve_output_dir(
        output_root=args.output_root,
        stage="grpo",
        run_name=run_name,
        explicit_output_dir=args.output_dir,
    )
    checkpoint_output_dir = resolve_checkpoint_output_dir(output_dir)

    configure_wandb_env(
        report_to=args.report_to,
        project=args.wandb_project,
        group=args.wandb_group,
        stage="grpo",
        run_name=wandb_run_name,
    )

    print_primary(f"[GRPO] Output directory: {output_dir}")
    print_primary(f"[GRPO] Run name: {run_name}")
    if wandb_run_name != run_name:
        print_primary(f"[GRPO] W&B run name: {wandb_run_name}")
    print_primary(f"[GRPO] Prompt template: {prompt_template}")
    print_primary(
        f"[GRPO] Prompt style (requested -> resolved): {requested_prompt_style} -> {prompt_style}"
    )
    print_primary(f"[GRPO] Tokenizer source: {resolved_tokenizer_path}")
    lora_target_modules = lora_target_modules_from_env()
    requested_lora_bias = lora_bias_from_env()
    checkpoint_lora_bias = peft_checkpoint_bias_policy(args.model_path)
    effective_lora_bias = checkpoint_lora_bias or requested_lora_bias
    require_publication_lora_configuration(
        full_finetune=bool(args.full_finetune),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=lora_target_modules,
        effective_bias=effective_lora_bias,
    )
    print_primary(f"[GRPO] LoRA target modules: {','.join(lora_target_modules)}")
    print_primary(
        "[GRPO] LoRA bias policy: "
        f"effective={effective_lora_bias}, requested={requested_lora_bias}, "
        f"checkpoint={checkpoint_lora_bias or '<full-model>'}"
    )
    print_primary(
        "[GRPO] Batch config: "
        f"per_device_train_batch_size={args.per_device_train_batch_size}, "
        f"per_device_eval_batch_size={args.per_device_eval_batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"num_generations={args.num_generations}, "
        f"canonical_set={bool(getattr(args, 'clasp_canonical_set', False))}, "
        f"fixed_null_set={bool(getattr(args, 'clasp_fixed_null_set', False))}, "
        f"deployment_set_size={int(getattr(args, 'clasp_deployment_set_size', 0))}, "
        f"policy_length_alpha={float(getattr(args, 'clasp_policy_length_alpha', 0.0))}, "
        f"dual_recovery_coupling={float(getattr(args, 'clasp_canonical_dual_recovery_coupling', 0.0))}, "
        f"canonical_exact_scale={float(getattr(args, 'clasp_canonical_exact_scale', 1.0))}, "
        f"canonical_grpo_anchor_scale={float(getattr(args, 'clasp_canonical_grpo_anchor_scale', 0.0))}, "
        f"canonical_quality_anchor_ratio={float(getattr(args, 'clasp_canonical_quality_anchor_ratio', 0.0))}, "
        f"canonical_dense_scales="
        f"{float(getattr(args, 'clasp_canonical_dense_quality_scale', 0.0))}/"
        f"{float(getattr(args, 'clasp_canonical_dense_feasibility_scale', 0.0))}/"
        f"{float(getattr(args, 'clasp_canonical_dense_violation_scale', 0.0))}/"
        f"{float(getattr(args, 'clasp_canonical_dense_recovery_scale', 0.0))}, "
        f"generation_batch_size={args.generation_batch_size}, "
        f"resolved_generation_batch_size={args.resolved_generation_batch_size}, "
        f"steps_per_generation={args.steps_per_generation}, "
        f"resolved_steps_per_generation={args.resolved_steps_per_generation}, "
        f"effective_train_batch_size={args.effective_train_batch_size}, "
        f"world_size={env_world_size()}"
    )

    resolved_generation_batch_size = (
        args.generation_batch_size
        if args.generation_batch_size is not None
        else args.resolved_generation_batch_size
    )
    if (
        int(max_prompt_length) >= 8192
        or int(max_completion_length) >= 1024
        or int(resolved_generation_batch_size) >= 32
        or int(args.gradient_accumulation_steps) >= 16
    ):
        print_primary(
            "[GRPO][Warn] Heavy generation configuration detected; long first-step latency is expected. "
            f"max_prompt_length={max_prompt_length}, "
            f"max_completion_length={max_completion_length}, "
            f"generation_batch_size={resolved_generation_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"world_size={env_world_size()}"
        )

    generation_defaults: dict[str, float | int] = {}
    if generation_config is not None:
        for field_name in ("temperature", "top_p", "repetition_penalty"):
            field_value = getattr(generation_config, field_name, None)
            if field_value is not None:
                try:
                    generation_defaults[field_name] = float(field_value)
                except (TypeError, ValueError):
                    continue
    generation_overrides = {
        "temperature": args.generation_temperature,
        "top_p": args.generation_top_p,
        "top_k": args.generation_top_k,
        "repetition_penalty": args.generation_repetition_penalty,
    }
    explicit_generation_overrides: set[str] = set()
    for field_name, field_value in generation_overrides.items():
        if field_value is None:
            continue
        explicit_generation_overrides.add(field_name)
        if field_name == "top_k":
            generation_defaults[field_name] = int(field_value)
        else:
            generation_defaults[field_name] = float(field_value)

    scale_rewards_supported = "scale_rewards" in inspect.signature(GRPOConfig.__init__).parameters
    importance_sampling_supported = (
        "importance_sampling_level" in inspect.signature(GRPOConfig.__init__).parameters
    )
    constrained_importance_level = (
        os.environ.get(
            "PD_IMPORTANCE_SAMPLING_LEVEL",
            "sequence",
        )
        .strip()
        .lower()
    )
    allow_token_importance_ablation = os.environ.get(
        "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION", "0"
    ).strip() in {"1", "true", "yes", "on"}
    if constrained_importance_level not in {"sequence", "token"}:
        raise ValueError(
            "PD_IMPORTANCE_SAMPLING_LEVEL must be 'sequence' or 'token', got "
            f"{constrained_importance_level!r}"
        )
    if (
        args.use_primal_dual
        and constrained_importance_level != "sequence"
        and not allow_token_importance_ablation
    ):
        raise RuntimeError(
            "Token-level constrained importance sampling is permitted only "
            "for the explicit matched ablation. Set "
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1 to acknowledge this."
        )
    exact_set_credit = bool(
        getattr(args, "clasp_canonical_set", False)
        or getattr(args, "clasp_fixed_null_set", False)
    )
    if (
        exact_set_credit
        and float(getattr(args, "clasp_policy_length_alpha", 0.0)) > 0.0
        and constrained_importance_level != "sequence"
    ):
        raise RuntimeError(
            "--clasp_policy_length_alpha>0 requires "
            "PD_IMPORTANCE_SAMPLING_LEVEL=sequence."
        )
    setattr(
        args,
        "pd_importance_sampling_level",
        constrained_importance_level,
    )
    setattr(
        args,
        "pd_allow_token_importance_ablation",
        allow_token_importance_ablation,
    )
    if args.use_primal_dual:
        if not args.pd_force_scale_rewards_none:
            raise RuntimeError(
                "Constrained-GRPO requires reward scaling to be disabled exactly. "
                "Run with --pd_force_scale_rewards_none."
            )
        if not scale_rewards_supported:
            raise RuntimeError(
                "Constrained-GRPO requires GRPOConfig(scale_rewards='none'), "
                "but this TRL build does not support scale_rewards."
            )
        if not importance_sampling_supported:
            raise RuntimeError(
                "Constrained-GRPO requires GRPOConfig(importance_sampling_level='sequence'), "
                "but this TRL build does not support importance_sampling_level."
            )
        print_primary(
            "[ConstrainedGRPO] Enforcing GRPO scale_rewards='none' and "
            f"importance_sampling_level={constrained_importance_level!r} "
            "for the constrained objective."
        )

    eval_strategy = "steps" if len(eval_dataset) > 0 else "no"
    if eval_strategy != "no":
        # TRL requires the global eval batch to hold whole prompt groups, but only checks it
        # once evaluation is enabled -- and it raises from inside GRPOConfig without naming the
        # flag to change. Check it here, where eval_strategy is finally known.
        global_eval_batch_size = args.per_device_eval_batch_size * env_world_size()
        if global_eval_batch_size % int(args.num_generations) != 0:
            raise ValueError(
                "TRL GRPO requires the global eval batch size to be divisible by "
                f"num_generations. global_eval_batch_size={global_eval_batch_size} "
                f"(per_device_eval_batch_size={args.per_device_eval_batch_size} x "
                f"world_size={env_world_size()}), num_generations={args.num_generations}. "
                f"Set --per_device_eval_batch_size to a multiple of {args.num_generations}."
            )
    grpo_signature = inspect.signature(GRPOConfig.__init__).parameters
    optional_grpo_kwargs = {}
    if "max_prompt_length" in grpo_signature:
        optional_grpo_kwargs["max_prompt_length"] = max_prompt_length
    # Only `generation_batch_size` is forwarded to GRPOConfig. TRL treats
    # `generation_batch_size` and `steps_per_generation` as mutually exclusive and raises if
    # both are set, and it recovers steps_per_generation from generation_batch_size using the
    # same formula `resolve_generation_schedule` applies, so the schedule is identical.
    unsupported_generation_overrides: list[str] = []
    for field_name, field_value in generation_defaults.items():
        if field_name in grpo_signature:
            optional_grpo_kwargs[field_name] = field_value
        elif field_name in explicit_generation_overrides:
            unsupported_generation_overrides.append(field_name)
    if unsupported_generation_overrides:
        print_primary(
            "[GRPO][Warn] This TRL GRPOConfig does not support explicit generation fields: "
            + ", ".join(sorted(unsupported_generation_overrides))
        )
    active_generation_kwargs = {
        key: optional_grpo_kwargs[key]
        for key in ("temperature", "top_p", "top_k", "repetition_penalty")
        if key in optional_grpo_kwargs
    }
    if active_generation_kwargs:
        print_primary(f"[GRPO] Training rollout generation kwargs: {active_generation_kwargs}")
    trust_region_overrides = {
        "beta": args.grpo_beta,
        "epsilon": args.grpo_epsilon,
        "epsilon_high": args.grpo_epsilon_high,
        "loss_type": args.grpo_loss_type,
    }
    unsupported_trust_region_overrides: list[str] = []
    for field_name, field_value in trust_region_overrides.items():
        if field_value is None:
            continue
        if field_name in grpo_signature:
            optional_grpo_kwargs[field_name] = field_value
        else:
            unsupported_trust_region_overrides.append(field_name)
    if unsupported_trust_region_overrides:
        print_primary(
            "[GRPO][Warn] This TRL GRPOConfig does not support explicit trust-region fields: "
            + ", ".join(sorted(unsupported_trust_region_overrides))
        )
    active_trust_region_kwargs = {
        key: optional_grpo_kwargs[key]
        for key in ("beta", "epsilon", "epsilon_high", "loss_type")
        if key in optional_grpo_kwargs
    }
    if active_trust_region_kwargs:
        print_primary(f"[GRPO] Trust-region kwargs: {active_trust_region_kwargs}")
    optimizer_overrides = {
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
    }
    unsupported_optimizer_overrides: list[str] = []
    for field_name, field_value in optimizer_overrides.items():
        if field_value is None:
            continue
        if field_name in grpo_signature:
            optional_grpo_kwargs[field_name] = field_value
        else:
            unsupported_optimizer_overrides.append(field_name)
    if unsupported_optimizer_overrides:
        raise RuntimeError(
            "This TRL GRPOConfig cannot honor the explicitly requested optimizer fields: "
            + ", ".join(sorted(unsupported_optimizer_overrides))
        )
    active_optimizer_kwargs = {
        key: optional_grpo_kwargs[key]
        for key in ("optim", "weight_decay", "warmup_ratio", "lr_scheduler_type")
        if key in optional_grpo_kwargs
    }
    if active_optimizer_kwargs:
        print_primary(f"[GRPO] Optimizer/schedule kwargs: {active_optimizer_kwargs}")
    if "seed" in grpo_signature:
        optional_grpo_kwargs["seed"] = args.seed
    if "data_seed" in grpo_signature:
        optional_grpo_kwargs["data_seed"] = data_seed
    if "sync_ref_model" in grpo_signature:
        # A step-zero KL reference is immutable.  TRL's moving-reference
        # callback would silently invalidate that scientific contract.
        optional_grpo_kwargs["sync_ref_model"] = False
    if "gradient_checkpointing" in grpo_signature:
        if "gradient_checkpointing_kwargs" in grpo_signature:
            optional_grpo_kwargs["gradient_checkpointing"] = True
            optional_grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        else:
            optional_grpo_kwargs["gradient_checkpointing"] = False
            print_primary(
                "[GRPO][Warn] This TRL/Transformers build does not expose "
                "gradient_checkpointing_kwargs; disabling gradient checkpointing to avoid "
                "DDP reentrant-backward failures."
            )
    if "ddp_static_graph" in grpo_signature:
        optional_grpo_kwargs["ddp_static_graph"] = True

    if runtime_profile is not None:
        profile_grpo_kwargs = runtime_profile.grpo_config_overrides(grpo_signature)
        for field_name, field_value in profile_grpo_kwargs.items():
            if field_name in optional_grpo_kwargs and optional_grpo_kwargs[field_name] != field_value:
                raise RuntimeError(
                    "GRPO runtime profile conflicts with a resolved config value: "
                    f"{field_name}={optional_grpo_kwargs[field_name]!r}, "
                    f"profile={field_value!r}"
                )
            optional_grpo_kwargs[field_name] = field_value

    config_cls = GRPOConfig
    if runtime_profile is not None and getattr(runtime_profile, "external_baseline_extension", False):
        from clasp.trainers.external_baseline_grpo import ExternalBaselineGRPOConfig

        config_cls = ExternalBaselineGRPOConfig
    grpo_args = config_cls(
        output_dir=checkpoint_output_dir,
        eval_strategy=eval_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=wandb_run_name,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        num_generations=args.num_generations,
        generation_batch_size=resolved_generation_batch_size,
        max_completion_length=max_completion_length,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **(
            {
                "scale_rewards": "none",
                "importance_sampling_level": constrained_importance_level,
            }
            if args.use_primal_dual
            else {}
        ),
        **optional_grpo_kwargs,
    )
    if runtime_profile is not None:
        runtime_profile.validate_grpo_config(grpo_args)
        runtime_profile_receipt["resolved_config_validation"] = "PASS"

    # TRL normally computes PEFT KL by disabling the adapter. That is only a
    # fixed reference when the adapter leaves base biases frozen. With
    # bias=lora_only/all, disable_adapter() retains the trained bias values and
    # silently makes the beta reference drift.  The preferred exact contract is
    # two independent loads of the same immutable base+SFT adapter: continue the
    # policy adapter and inject the other, fully frozen PEFT model as ref_model.
    resolved_beta = float(getattr(grpo_args, "beta", 0.0) or 0.0)
    biased_lora_fixed_reference_required = (
        not args.full_finetune and resolved_beta > 0.0 and effective_lora_bias != "none"
    )
    runtime_profile_fixed_reference_required = bool(
        runtime_profile is not None
        and getattr(runtime_profile, "requires_explicit_fixed_reference", False)
    )
    explicit_kl_reference_path, explicit_kl_reference_receipt = resolve_explicit_fixed_kl_reference(
        required=(
            biased_lora_fixed_reference_required or runtime_profile_fixed_reference_required
        ),
    )
    verified_kl_reference = None
    exact_peft_source_binding = None
    explicit_kl_reference_model = None
    kl_reference_mode = "none_beta_zero" if resolved_beta == 0.0 else "trl_disable_adapter"
    exact_kl_runtime_receipt = None
    if explicit_kl_reference_path is not None:
        if explicit_kl_reference_receipt is None:
            raise RuntimeError("Explicit KL reference receipt unexpectedly unresolved.")
        reference_is_peft = peft_checkpoint_bias_policy(explicit_kl_reference_path) is not None
        if reference_is_peft:
            if not model_is_peft:
                raise RuntimeError(
                    "exact PEFT KL mode requires continuing the same loaded policy adapter"
                )
            if bool(getattr(grpo_args, "sync_ref_model", False)):
                raise RuntimeError("exact PEFT KL mode requires sync_ref_model=False")
            if bool(getattr(grpo_args, "use_vllm", False)):
                raise RuntimeError(
                    "exact PEFT KL mode forbids vLLM: pinned TRL merges/unmerges the live "
                    "adapter in place and can round BF16 policy weights"
                )
            exact_peft_source_binding = verify_exact_peft_kl_source(
                policy_adapter_path=args.model_path,
                reference_adapter_path=explicit_kl_reference_path,
                authoritative_receipt_path=explicit_kl_reference_receipt,
                tokenizer_path=resolved_tokenizer_path,
            )
            explicit_kl_reference_model = _load_frozen_exact_peft_reference(
                exact_peft_source_binding,
                args.dtype,
                tokenizer=tokenizer,
            )
            verified_kl_reference = {
                "protocol": exact_peft_source_binding["authoritative_receipt_protocol"],
                "receipt_sha256": exact_peft_source_binding["authoritative_receipt_sha256"],
            }
            kl_reference_mode = "explicit_frozen_exact_peft_step_zero"
            print_primary(
                "[GRPO] Independently loaded the exact frozen base+adapter KL reference: "
                f"{explicit_kl_reference_path}"
            )
        else:
            # Retain the strict merged-v2 verifier for existing non-PEFT inputs.
            # It is not a fallback for a failed exact pair and its numerical gate
            # is intentionally not tuned to the observed 7B merge discrepancy.
            if checkpoint_fidelity_enabled():
                reference_realpath = os.path.realpath(explicit_kl_reference_path)
                policy_realpath = os.path.realpath(str(args.model_path))
                tokenizer_realpath = os.path.realpath(str(resolved_tokenizer_path))
                if policy_realpath != reference_realpath:
                    raise RuntimeError(
                        "merged-v2 mode requires the verified full model as both policy "
                        f"initialization and KL reference: policy={policy_realpath}, "
                        f"reference={reference_realpath}"
                    )
                if tokenizer_realpath != reference_realpath:
                    raise RuntimeError(
                        "merged-v2 mode requires its bound tokenizer directory: "
                        f"tokenizer={tokenizer_realpath}, reference={reference_realpath}"
                    )
            verified_kl_reference = verify_fixed_kl_reference_distributed(
                explicit_kl_reference_path,
                explicit_kl_reference_receipt,
            )
            kl_reference_mode = "explicit_frozen_full_model_for_biased_lora"
            print_primary(
                "[GRPO] Loading a verified frozen full-model KL reference because "
                f"beta={resolved_beta:g} and LoRA bias={effective_lora_bias}: "
                f"{explicit_kl_reference_path}"
            )

    training_run_contract_identity = resolve_training_run_contract_identity(output_dir)
    validate_fixed_horizon_preflight(
        max_steps=args.max_steps,
        runtime_world_size=env_world_size(),
        contract_identity=training_run_contract_identity,
    )
    write_hyperparams(
        output_dir,
        build_run_payload(
            stage="grpo",
            run_name=run_name,
            base_model_name=base_model_name,
            model_path=args.model_path,
            output_dir=output_dir,
            tasks=args.tasks,
            report_to=args.report_to,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            stage_hyperparameters={
                "runtime_profile": runtime_profile_receipt,
                "dtype": args.dtype,
                "init_model_artifact_name": init_model_artifact_name,
                "full_finetune": args.full_finetune,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": lora_target_modules,
                "lora_bias": effective_lora_bias,
                "lora_bias_requested": requested_lora_bias,
                "lora_bias_checkpoint": checkpoint_lora_bias,
                "kl_reference_mode": kl_reference_mode,
                "kl_reference_model_path": explicit_kl_reference_path,
                "kl_reference_receipt_path": explicit_kl_reference_receipt,
                "kl_reference_runtime_receipt_path": (
                    os.path.join(output_dir, EXACT_PEFT_KL_RECEIPT_FILENAME)
                    if exact_peft_source_binding is not None
                    else None
                ),
                "checkpoint_fidelity_mode": checkpoint_fidelity_mode(),
                "training_run_contract_path": (
                    str(os.environ.get("CLASP_TRAINING_RUN_CONTRACT_PATH", "")).strip() or None
                ),
                "training_run_contract_identity": training_run_contract_identity,
                "initialization_fidelity_protocol": (
                    initialization_fidelity.get("protocol")
                    if initialization_fidelity is not None
                    else None
                ),
                "initialization_fidelity_status": (
                    initialization_fidelity.get("status")
                    if initialization_fidelity is not None
                    else None
                ),
                "initialization_fidelity_receipt_sha256": (
                    initialization_fidelity.get("receipt_sha256")
                    if initialization_fidelity is not None
                    else None
                ),
                "seed": args.seed,
                "data_seed": data_seed,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_train_batch_size": args.effective_train_batch_size,
                "learning_rate": args.learning_rate,
                "optim": args.optim,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type,
                "resolved_optimizer_kwargs": active_optimizer_kwargs,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "generation_batch_size": args.generation_batch_size,
                "resolved_generation_batch_size": args.resolved_generation_batch_size,
                "steps_per_generation": args.steps_per_generation,
                "resolved_steps_per_generation": args.resolved_steps_per_generation,
                "generation_temperature": args.generation_temperature,
                "generation_top_p": args.generation_top_p,
                "generation_top_k": args.generation_top_k,
                "generation_repetition_penalty": args.generation_repetition_penalty,
                "resolved_generation_kwargs": active_generation_kwargs,
                "grpo_beta": args.grpo_beta,
                "grpo_epsilon": args.grpo_epsilon,
                "grpo_epsilon_high": args.grpo_epsilon_high,
                "grpo_loss_type": args.grpo_loss_type,
                "resolved_trust_region_kwargs": active_trust_region_kwargs,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
                "min_train_samples_per_task": args.min_train_samples_per_task,
                "min_eval_samples_per_task": args.min_eval_samples_per_task,
                "min_task_types_per_local_batch": 2 if len(args.tasks) > 1 else 1,
                "use_primal_dual": args.use_primal_dual,
                "pd_constraint_target": args.pd_constraint_target,
                "pd_cost_limit": args.pd_cost_limit,
                "pd_resolved_cost_limit": epsilon,
                "pd_lambda_init": args.pd_lambda_init,
                "pd_dual_lr": args.pd_dual_lr,
                "pd_dual_lr_up": args.pd_dual_lr_up,
                "pd_dual_lr_down": args.pd_dual_lr_down,
                "pd_dual_update_interval": args.pd_dual_update_interval,
                "pd_violation_ema_beta": args.pd_violation_ema_beta,
                "pd_lambda_max": args.pd_lambda_max,
                "pd_violation_tolerance": args.pd_violation_tolerance,
                "pd_objective_weight": args.pd_objective_weight,
                "pd_structure_weight": args.pd_structure_weight,
                "pd_sync_across_ranks": args.pd_sync_across_ranks,
                "pd_require_state_on_resume": args.pd_require_state_on_resume,
                "pd_strict_constraint_range": args.pd_strict_constraint_range,
                "pd_force_scale_rewards_none": args.pd_force_scale_rewards_none,
                "pd_log_every": args.pd_log_every,
                "pd_soft_objective_for_infeasible": args.pd_soft_objective_for_infeasible,
                "pd_improved_mode": args.pd_improved_mode,
                "pd_normalize_primal_advantage": args.pd_normalize_primal_advantage,
                "pd_primal_advantage_scale": args.pd_primal_advantage_scale,
                "pd_penalty_scale": args.pd_penalty_scale,
                "pd_penalty_scale_start": args.pd_penalty_scale_start,
                "pd_penalty_scale_warmup_fraction": args.pd_penalty_scale_warmup_fraction,
                "pd_positive_constraint_penalty": args.pd_positive_constraint_penalty,
                "pd_lexicographic_advantage": args.pd_lexicographic_advantage,
                "pd_lexicographic_feasible_margin": args.pd_lexicographic_feasible_margin,
                "pd_group_scalarization": args.pd_group_scalarization,
                "pd_separate_z": args.pd_separate_z,
                "pd_separate_z_formula_id": (
                    DUAL_SEPARATE_Z_FORMULA_ID if args.pd_separate_z else None
                ),
                "pd_component_profile": args.pd_component_profile,
                "use_clasp": getattr(args, "use_clasp", False),
                "clasp_strict_defaults": getattr(args, "clasp_strict_defaults", True),
                "clasp_raw_feasibility_targeting": getattr(
                    args, "clasp_raw_feasibility_targeting", False
                ),
                "clasp_canonical_set": getattr(args, "clasp_canonical_set", False),
                "clasp_fixed_null_set": getattr(
                    args, "clasp_fixed_null_set", False
                ),
                "clasp_fixed_null_receipt_path": getattr(
                    args, "clasp_fixed_null_receipt_path", None
                ),
                "clasp_fixed_null_receipt_identity": getattr(
                    args, "clasp_fixed_null_receipt_identity", None
                ),
                "clasp_deployment_set_size": getattr(
                    args, "clasp_deployment_set_size", 0
                ),
                "clasp_canonical_recovery_scale": getattr(
                    args, "clasp_canonical_recovery_scale", 1.0
                ),
                "clasp_canonical_dual_recovery_coupling": getattr(
                    args, "clasp_canonical_dual_recovery_coupling", 0.0
                ),
                "clasp_canonical_recovery_temperature": getattr(
                    args, "clasp_canonical_recovery_temperature", 0.15
                ),
                "clasp_canonical_max_subsets": getattr(
                    args, "clasp_canonical_max_subsets", 1024
                ),
                "clasp_canonical_exact_scale": getattr(
                    args, "clasp_canonical_exact_scale", 1.0
                ),
                "clasp_canonical_grpo_anchor_scale": getattr(
                    args, "clasp_canonical_grpo_anchor_scale", 0.0
                ),
                "clasp_canonical_quality_anchor_ratio": getattr(
                    args, "clasp_canonical_quality_anchor_ratio", 0.0
                ),
                "clasp_canonical_dense_quality_scale": getattr(
                    args, "clasp_canonical_dense_quality_scale", 0.0
                ),
                "clasp_canonical_dense_feasibility_scale": getattr(
                    args, "clasp_canonical_dense_feasibility_scale", 0.0
                ),
                "clasp_canonical_dense_violation_scale": getattr(
                    args, "clasp_canonical_dense_violation_scale", 0.0
                ),
                "clasp_canonical_dense_recovery_scale": getattr(
                    args, "clasp_canonical_dense_recovery_scale", 0.0
                ),
                "clasp_policy_length_alpha": getattr(
                    args, "clasp_policy_length_alpha", 0.0
                ),
                "clasp_pareto_anchor": getattr(args, "clasp_pareto_anchor", True),
                "clasp_pareto_tau": getattr(args, "clasp_pareto_tau", 4.0),
                "clasp_pareto_center": getattr(args, "clasp_pareto_center", 0.5),
                "clasp_feasibility_gated": os.environ.get("CLASP_FEASIBILITY_GATED", "0"),
                "clasp_lexicographic_gate": os.environ.get(
                    "CLASP_LEXICOGRAPHIC_GATE",
                    "0",
                ),
                "clasp_bon_aligned": os.environ.get("CLASP_BON_ALIGNED", "1"),
                "clasp_frontier_cdf": os.environ.get("CLASP_FRONTIER_CDF", "0"),
                "clasp_bon_target_failure_rate": os.environ.get(
                    "CLASP_BON_TARGET_FAILURE_RATE",
                    "0.01",
                ),
                "clasp_bon_target_failure_rates": os.environ.get(
                    "CLASP_BON_TARGET_FAILURE_RATES",
                    "",
                ),
                "clasp_bon_dual_lr": os.environ.get("CLASP_BON_DUAL_LR", "0.05"),
                "clasp_bon_dual_ema_beta": os.environ.get(
                    "CLASP_BON_DUAL_EMA_BETA",
                    "0.9",
                ),
                "clasp_bon_lambda_init": os.environ.get("CLASP_BON_LAMBDA_INIT", "0.0"),
                "clasp_bon_lambda_max": os.environ.get("CLASP_BON_LAMBDA_MAX", "100.0"),
                "clasp_bon_quality_temperature": os.environ.get(
                    "CLASP_BON_QUALITY_TEMPERATURE",
                    "0.0",
                ),
                "clasp_bon_success_credit": os.environ.get(
                    "CLASP_BON_SUCCESS_CREDIT",
                    "dense_rank",
                ),
                "clasp_bon_dense_quality_scale": os.environ.get(
                    "CLASP_BON_DENSE_QUALITY_SCALE",
                    "1.0",
                ),
                "clasp_bon_dense_feasibility_scale": os.environ.get(
                    "CLASP_BON_DENSE_FEASIBILITY_SCALE",
                    "1.0",
                ),
                "clasp_bon_dense_violation_mix": os.environ.get(
                    "CLASP_BON_DENSE_VIOLATION_MIX",
                    "0.5",
                ),
                "clasp_bon_recovery_scale": os.environ.get(
                    "CLASP_BON_RECOVERY_SCALE",
                    "1.0",
                ),
                "clasp_component_dual_recovery_scale": os.environ.get(
                    "CLASP_COMPONENT_DUAL_RECOVERY_SCALE",
                    "1.0",
                ),
                "clasp_frontier_cdf_gap_targets": os.environ.get(
                    "CLASP_FRONTIER_CDF_GAP_TARGETS",
                    "",
                ),
                "clasp_frontier_cdf_quality_temperature_initial": os.environ.get(
                    "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_INITIAL",
                    "0.5",
                ),
                "clasp_frontier_cdf_quality_temperature_final": os.environ.get(
                    "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_FINAL",
                    "0.15",
                ),
                "clasp_frontier_cdf_quality_anneal_fraction": os.environ.get(
                    "CLASP_FRONTIER_CDF_QUALITY_ANNEAL_FRACTION",
                    "0.6",
                ),
                "clasp_frontier_cdf_violation_temperature": os.environ.get(
                    "CLASP_FRONTIER_CDF_VIOLATION_TEMPERATURE",
                    "0.15",
                ),
                "clasp_frontier_cdf_recovery_scale": os.environ.get(
                    "CLASP_FRONTIER_CDF_RECOVERY_SCALE",
                    "1.0",
                ),
                "clasp_frontier_cdf_relative_failed_recovery": os.environ.get(
                    "CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY",
                    "0",
                ),
                "clasp_frontier_cdf_relative_recovery_scale": os.environ.get(
                    "CLASP_FRONTIER_CDF_RELATIVE_RECOVERY_SCALE",
                    "1.0",
                ),
                "clasp_frontier_cdf_objective_score_scale": os.environ.get(
                    "CLASP_FRONTIER_CDF_OBJECTIVE_SCORE_SCALE",
                    "2.0",
                ),
                "clasp_frontier_cdf_gap_slope": os.environ.get(
                    "CLASP_FRONTIER_CDF_GAP_SLOPE",
                    "3.0",
                ),
                "clasp_frontier_cdf_feasibility_violation_tolerance": os.environ.get(
                    "CLASP_FRONTIER_CDF_FEASIBILITY_VIOLATION_TOLERANCE",
                    "1e-8",
                ),
                "clasp_declared_objective_match_weight": os.environ.get(
                    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT",
                    "0.0",
                ),
                "clasp_feasibility_warmup_steps": getattr(
                    args, "clasp_feasibility_warmup_steps", 0
                ),
                "clasp_feasibility_warmup_targets": getattr(
                    args, "clasp_feasibility_warmup_targets", ""
                ),
                "clasp_feasibility_warmup_hold_until_eval": getattr(
                    args,
                    "clasp_feasibility_warmup_hold_until_eval",
                    True,
                ),
                "clasp_feasibility_warmup_margin": os.environ.get(
                    "CLASP_FEASIBILITY_WARMUP_FEASIBLE_MARGIN",
                    "3.0",
                ),
                "clasp_feasibility_warmup_violation_scale": os.environ.get(
                    "CLASP_FEASIBILITY_WARMUP_VIOLATION_SCALE",
                    "2.0",
                ),
                "clasp_feasibility_warmup_rank_scale": os.environ.get(
                    "CLASP_FEASIBILITY_WARMUP_RANK_SCALE",
                    "1.0",
                ),
                "clasp_feasibility_gate_margin": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_MARGIN",
                    "2.0",
                ),
                "clasp_feasibility_gate_feasible_rank_scale": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_FEASIBLE_RANK_SCALE",
                    "0.5",
                ),
                "clasp_feasibility_gate_infeasible_rank_scale": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_INFEASIBLE_RANK_SCALE",
                    "0.5",
                ),
                "clasp_feasibility_gate_all_infeasible_weight": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_ALL_INFEASIBLE_WEIGHT",
                    "1.5",
                ),
                "clasp_feasibility_gate_mixed_weight": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_MIXED_WEIGHT",
                    "1.25",
                ),
                "clasp_feasibility_gate_all_feasible_weight": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_ALL_FEASIBLE_WEIGHT",
                    "1.0",
                ),
                "clasp_feasibility_gate_feasible_z_mix": os.environ.get(
                    "CLASP_FEASIBILITY_GATE_FEASIBLE_Z_MIX",
                    "0.5",
                ),
                "clasp_proper_dual": getattr(args, "clasp_proper_dual", True),
                "clasp_dual_update_interval": getattr(args, "clasp_dual_update_interval", 4),
                "clasp_proper_dual_gradient_mode": os.environ.get(
                    "CLASP_PROPER_DUAL_GRADIENT_MODE",
                    "signed",
                ),
                "clasp_adaptive_length": getattr(args, "clasp_adaptive_length", True),
                "clasp_adaptive_length_quantile": getattr(
                    args, "clasp_adaptive_length_quantile", 0.99
                ),
                "clasp_adaptive_length_floor": getattr(args, "clasp_adaptive_length_floor", 128),
                "clasp_adaptive_length_window": getattr(args, "clasp_adaptive_length_window", 1000),
                "clasp_adaptive_length_refresh_steps": getattr(
                    args, "clasp_adaptive_length_refresh_steps", 50
                ),
                "clasp_adaptive_length_margin": getattr(args, "clasp_adaptive_length_margin", 32),
                "clasp_mixed_adaptive_length_mode": os.environ.get(
                    "CLASP_MIXED_ADAPTIVE_LENGTH_MODE",
                    "error",
                ),
                "clasp_pairwise_loss_weight": os.environ.get("CLASP_PAIRWISE_LOSS_WEIGHT", "0.0"),
                "clasp_aux_grouping_strict": os.environ.get(
                    "CLASP_AUX_GROUPING_STRICT",
                    "1",
                ),
                "clasp_pairwise_margin": os.environ.get("CLASP_PAIRWISE_MARGIN", "0.0"),
                "clasp_pairwise_advantage_delta": os.environ.get(
                    "CLASP_PAIRWISE_ADVANTAGE_DELTA", "0.0"
                ),
                "clasp_elite_imitation_loss_weight": os.environ.get(
                    "CLASP_ELITE_IMITATION_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_elite_imitation_advantage_delta": os.environ.get(
                    "CLASP_ELITE_IMITATION_ADVANTAGE_DELTA",
                    "0.0",
                ),
                "clasp_multi_elite_imitation_loss_weight": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_multi_elite_imitation_fraction": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_FRACTION",
                    "0.25",
                ),
                "clasp_multi_elite_imitation_max_raw_gap": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_MAX_RAW_GAP",
                    "0.0",
                ),
                "clasp_multi_elite_imitation_advantage_delta": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_ADVANTAGE_DELTA",
                    "0.0",
                ),
                "clasp_multi_elite_imitation_temperature": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_TEMPERATURE",
                    "1.0",
                ),
                "clasp_multi_elite_imitation_require_feasible": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_REQUIRE_FEASIBLE",
                    "1",
                ),
                "clasp_multi_elite_imitation_fallback_to_best_feasible": os.environ.get(
                    "CLASP_MULTI_ELITE_IMITATION_FALLBACK_TO_BEST_FEASIBLE",
                    "1",
                ),
                "clasp_reference_imitation_loss_weight": os.environ.get(
                    "CLASP_REFERENCE_IMITATION_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_reference_imitation_max_tokens": os.environ.get(
                    "CLASP_REFERENCE_IMITATION_MAX_TOKENS",
                    "256",
                ),
                "clasp_reference_imitation_tasks": os.environ.get(
                    "CLASP_REFERENCE_IMITATION_TASKS",
                    "",
                ),
                "clasp_reference_contrastive_loss_weight": os.environ.get(
                    "CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_reference_contrastive_margin": os.environ.get(
                    "CLASP_REFERENCE_CONTRASTIVE_MARGIN",
                    "0.02",
                ),
                "clasp_reference_contrastive_max_tokens": os.environ.get(
                    "CLASP_REFERENCE_CONTRASTIVE_MAX_TOKENS",
                    "256",
                ),
                "clasp_reference_contrastive_tasks": os.environ.get(
                    "CLASP_REFERENCE_CONTRASTIVE_TASKS",
                    "",
                ),
                "clasp_self_replay": getattr(args, "clasp_self_replay", False),
                "clasp_self_replay_state_schema_version": 1,
                "clasp_self_replay_checkpoint_state_key": "__clasp_trainer__",
                "clasp_self_replay_enabled": os.environ.get("CLASP_SELF_REPLAY_ENABLED", "0"),
                "clasp_self_replay_imitation_weight": os.environ.get(
                    "CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_self_replay_contrastive_weight": os.environ.get(
                    "CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_self_replay_sample_rate": os.environ.get(
                    "CLASP_SELF_REPLAY_SAMPLE_RATE",
                    "0.5",
                ),
                "clasp_self_replay_max_tokens": os.environ.get(
                    "CLASP_SELF_REPLAY_MAX_TOKENS",
                    "512",
                ),
                "clasp_self_replay_max_keys": os.environ.get(
                    "CLASP_SELF_REPLAY_MAX_KEYS",
                    "20000",
                ),
                "clasp_self_replay_max_per_key": os.environ.get(
                    "CLASP_SELF_REPLAY_MAX_PER_KEY",
                    "4",
                ),
                "clasp_self_replay_min_buffer_size": os.environ.get(
                    "CLASP_SELF_REPLAY_MIN_BUFFER_SIZE",
                    "8",
                ),
                "clasp_validation_gate_checkpoints": getattr(
                    args,
                    "clasp_validation_gate_checkpoints",
                    False,
                ),
                "clasp_tsp_two_opt_distill_loss_weight": os.environ.get(
                    "CLASP_TSP_TWO_OPT_DISTILL_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_tsp_two_opt_distill_max_tokens": os.environ.get(
                    "CLASP_TSP_TWO_OPT_DISTILL_MAX_TOKENS",
                    "512",
                ),
                "clasp_tsp_two_opt_distill_max_passes": os.environ.get(
                    "CLASP_TSP_TWO_OPT_DISTILL_MAX_PASSES",
                    "1",
                ),
                "clasp_tsp_two_opt_distill_min_gain": os.environ.get(
                    "CLASP_TSP_TWO_OPT_DISTILL_MIN_GAIN",
                    "0.0",
                ),
                "clasp_cvrp_local_search_distill_loss_weight": os.environ.get(
                    "CLASP_CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_cvrp_local_search_distill_max_tokens": os.environ.get(
                    "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MAX_TOKENS",
                    "768",
                ),
                "clasp_cvrp_local_search_distill_max_passes": os.environ.get(
                    "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MAX_PASSES",
                    "2",
                ),
                "clasp_cvrp_local_search_distill_min_gain": os.environ.get(
                    "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MIN_GAIN",
                    "0.0",
                ),
                "clasp_jssp_active_schedule_distill_loss_weight": os.environ.get(
                    "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_jssp_active_schedule_distill_max_tokens": os.environ.get(
                    "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MAX_TOKENS",
                    "768",
                ),
                "clasp_jssp_active_schedule_distill_max_passes": os.environ.get(
                    "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MAX_PASSES",
                    "1",
                ),
                "clasp_jssp_active_schedule_distill_min_gain": os.environ.get(
                    "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MIN_GAIN",
                    "0.0",
                ),
                "clasp_mis_maximalize_distill_loss_weight": os.environ.get(
                    "CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT",
                    "0.0",
                ),
                "clasp_mis_maximalize_distill_max_tokens": os.environ.get(
                    "CLASP_MIS_MAXIMALIZE_DISTILL_MAX_TOKENS",
                    "512",
                ),
                "clasp_mis_maximalize_distill_min_gain": os.environ.get(
                    "CLASP_MIS_MAXIMALIZE_DISTILL_MIN_GAIN",
                    "0.0",
                ),
                "clasp_primal_score_gap_penalty": os.environ.get(
                    "CLASP_PRIMAL_SCORE_GAP_PENALTY",
                    "0.0",
                ),
                "clasp_hard_group_weight_scale": os.environ.get(
                    "CLASP_HARD_GROUP_WEIGHT_SCALE",
                    "0.0",
                ),
                "clasp_hard_group_weight_threshold": os.environ.get(
                    "CLASP_HARD_GROUP_WEIGHT_THRESHOLD",
                    "0.0",
                ),
                "clasp_hard_group_weight_max": os.environ.get(
                    "CLASP_HARD_GROUP_WEIGHT_MAX",
                    "3.0",
                ),
                "clasp_tsp_objective_gap_cost": os.environ.get(
                    "CLASP_TSP_OBJECTIVE_GAP_COST",
                    "0",
                ),
                "clasp_tsp_objective_gap_cost_cap": os.environ.get(
                    "CLASP_TSP_OBJECTIVE_GAP_COST_CAP",
                    "0.05",
                ),
                "clasp_tsp_objective_gap_cost_scale": os.environ.get(
                    "CLASP_TSP_OBJECTIVE_GAP_COST_SCALE",
                    "1.0",
                ),
                "clasp_tsp_target_gap": os.environ.get("CLASP_TSP_TARGET_GAP", "0.0"),
                "clasp_tsp_target_gap_constraint": os.environ.get(
                    "CLASP_TSP_TARGET_GAP_CONSTRAINT",
                    "0",
                ),
                "clasp_tsp_target_gap_constraint_gap": os.environ.get(
                    "CLASP_TSP_TARGET_GAP_CONSTRAINT_GAP",
                    "",
                ),
                "clasp_tsp_target_gap_constraint_scale": os.environ.get(
                    "CLASP_TSP_TARGET_GAP_CONSTRAINT_SCALE",
                    "1.0",
                ),
                "clasp_cvrp_target_gap": os.environ.get("CLASP_CVRP_TARGET_GAP", "0.0"),
                "clasp_cvrp_target_gap_constraint": os.environ.get(
                    "CLASP_CVRP_TARGET_GAP_CONSTRAINT",
                    "0",
                ),
                "clasp_cvrp_target_gap_constraint_gap": os.environ.get(
                    "CLASP_CVRP_TARGET_GAP_CONSTRAINT_GAP",
                    "",
                ),
                "clasp_cvrp_target_gap_constraint_scale": os.environ.get(
                    "CLASP_CVRP_TARGET_GAP_CONSTRAINT_SCALE",
                    "1.0",
                ),
                "clasp_cvrp_objective_gap_cost": os.environ.get(
                    "CLASP_CVRP_OBJECTIVE_GAP_COST",
                    "0",
                ),
                "clasp_cvrp_objective_gap_cost_cap": os.environ.get(
                    "CLASP_CVRP_OBJECTIVE_GAP_COST_CAP",
                    "0.20",
                ),
                "clasp_cvrp_objective_gap_cost_scale": os.environ.get(
                    "CLASP_CVRP_OBJECTIVE_GAP_COST_SCALE",
                    "1.0",
                ),
                "clasp_tsp_tail_gap": os.environ.get("CLASP_TSP_TAIL_GAP", "0.0"),
                "clasp_tsp_tail_gap_constraint": os.environ.get(
                    "CLASP_TSP_TAIL_GAP_CONSTRAINT",
                    "0",
                ),
                "clasp_tsp_tail_gap_constraint_gap": os.environ.get(
                    "CLASP_TSP_TAIL_GAP_CONSTRAINT_GAP",
                    "",
                ),
                "clasp_tsp_tail_gap_constraint_scale": os.environ.get(
                    "CLASP_TSP_TAIL_GAP_CONSTRAINT_SCALE",
                    "1.0",
                ),
                "clasp_feasibility_ignore_cost_names": os.environ.get(
                    "CLASP_FEASIBILITY_IGNORE_COST_NAMES",
                    "",
                ),
                "clasp_tsp_route_only_output": os.environ.get(
                    "CLASP_TSP_ROUTE_ONLY_OUTPUT",
                    "0",
                ),
                "pd_enable_schedule": args.pd_enable_schedule,
                "pd_stage_a_fraction": args.pd_stage_a_fraction,
                "pd_stage_b_fraction": args.pd_stage_b_fraction,
                "pd_stage_a_cost_limit": args.pd_stage_a_cost_limit,
                "pd_stage_b_cost_limit": args.pd_stage_b_cost_limit,
                "pd_stage_c_cost_limit": args.pd_stage_c_cost_limit,
                "pd_stage_a_dual_lr_scale": args.pd_stage_a_dual_lr_scale,
                "pd_stage_b_dual_lr_scale": args.pd_stage_b_dual_lr_scale,
                "pd_stage_c_dual_lr_scale": args.pd_stage_c_dual_lr_scale,
                "pd_stage_c_constraint_threshold": args.pd_stage_c_constraint_threshold,
                "train_size_bucket": getattr(args, "train_size_bucket", "all"),
                "train_min_instance_size": getattr(args, "train_min_instance_size", 0),
                "train_max_instance_size": getattr(args, "train_max_instance_size", 0),
                "train_task_multipliers": getattr(args, "train_task_multipliers", ""),
                "no_reason": bool(args.no_reason),
                "grpo_verified_train_reference_teacher": os.environ.get("GRPO_VERIFIED_TRAIN_REFERENCE_TEACHER", "0") == "1",
                "prompt_style": prompt_style,
                "prompt_style_requested": requested_prompt_style,
                "prompt_template": prompt_template,
                "tokenizer_path": resolved_tokenizer_path,
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing": optional_grpo_kwargs.get("gradient_checkpointing"),
                "gradient_checkpointing_kwargs": optional_grpo_kwargs.get(
                    "gradient_checkpointing_kwargs"
                ),
                "ddp_static_graph": optional_grpo_kwargs.get("ddp_static_graph"),
            },
            context_info=context_info,
            args_namespace=args,
            trainer_args=grpo_args,
            checkpoint_output_dir=checkpoint_output_dir,
        ),
    )

    peft_config = None
    if not model_is_peft and not args.full_finetune:
        peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    callbacks = []
    stop_step_raw = str(os.environ.get("GRPO_STOP_AT_GLOBAL_STEP", "")).strip()
    if stop_step_raw:
        if not stop_step_raw.isdigit() or int(stop_step_raw) <= 0:
            raise ValueError("GRPO_STOP_AT_GLOBAL_STEP must be a positive integer")
        stop_step = int(stop_step_raw)
        if int(args.max_steps) <= 0 or stop_step > int(args.max_steps):
            raise ValueError(
                "GRPO_STOP_AT_GLOBAL_STEP must not exceed the final positive "
                f"max_steps horizon: stop={stop_step}, max_steps={args.max_steps}"
            )
        callbacks.append(StopAtGlobalStepCallback(stop_step))
        print_primary(
            f"[GRPO] Clean continuation boundary enabled at global_step={stop_step}; "
            f"final scheduler horizon remains max_steps={int(args.max_steps)}"
        )
    reward_manager_initial_state = None
    task_triplets = build_task_triplets(args.tasks) if runtime_profile is None else None
    if args.use_primal_dual:
        reward_manager = build_primal_dual_manager(args, args.tasks)
        reward_manager_initial_state = reward_manager.state_dict()
        reward_funcs = [reward_manager]
        trainer_cls = MixedBatchPrimalDualGRPOTrainer
        trainer_extra_kwargs = {
            "primal_dual_reward_manager": reward_manager,
            "normalize_primal_advantage": getattr(args, "pd_normalize_primal_advantage", False),
            "primal_advantage_scale": getattr(args, "pd_primal_advantage_scale", 1.0),
            "penalty_scale": getattr(args, "pd_penalty_scale", 1.0),
            "penalty_scale_start": getattr(args, "pd_penalty_scale_start", None),
            "penalty_scale_warmup_fraction": getattr(args, "pd_penalty_scale_warmup_fraction", 0.0),
            "positive_constraint_penalty": getattr(args, "pd_positive_constraint_penalty", False),
            "lexicographic_advantage": getattr(args, "pd_lexicographic_advantage", False),
            "lexicographic_feasible_margin": getattr(args, "pd_lexicographic_feasible_margin", 0.5),
            "group_scalarization": getattr(args, "pd_group_scalarization", False),
            "separate_z": getattr(args, "pd_separate_z", False),
        }

        # ------------------------------------------------------------------
        # CLASP-GRPO wiring. Strictly additive; no effect unless --use_clasp.
        # ------------------------------------------------------------------
        if getattr(args, "use_clasp", False):
            from clasp.rewards.clasp_math import AdaptiveLengthEstimator

            clasp_length_estimator = None
            if bool(getattr(args, "clasp_adaptive_length", True)):
                clasp_length_estimator = AdaptiveLengthEstimator(
                    tasks=args.tasks,
                    global_cap=int(max_completion_length),
                    floor=int(getattr(args, "clasp_adaptive_length_floor", 128)),
                    window=int(getattr(args, "clasp_adaptive_length_window", 1000)),
                    quantile=float(getattr(args, "clasp_adaptive_length_quantile", 0.99)),
                    refresh_steps=int(getattr(args, "clasp_adaptive_length_refresh_steps", 50)),
                    margin=int(getattr(args, "clasp_adaptive_length_margin", 32)),
                )

            trainer_extra_kwargs.update(
                {
                    "clasp_enabled": True,
                    "clasp_pareto_anchor": bool(getattr(args, "clasp_pareto_anchor", True)),
                    "clasp_pareto_tau": float(getattr(args, "clasp_pareto_tau", 4.0)),
                    "clasp_pareto_center": float(getattr(args, "clasp_pareto_center", 0.5)),
                    "clasp_adaptive_length": clasp_length_estimator,
                    "clasp_canonical_set": bool(
                        getattr(args, "clasp_canonical_set", False)
                    ),
                    "clasp_fixed_null_set": bool(
                        getattr(args, "clasp_fixed_null_set", False)
                    ),
                    "clasp_deployment_set_size": int(
                        getattr(args, "clasp_deployment_set_size", 0)
                    ),
                    "clasp_canonical_recovery_scale": float(
                        getattr(args, "clasp_canonical_recovery_scale", 1.0)
                    ),
                    "clasp_canonical_dual_recovery_coupling": float(
                        getattr(
                            args,
                            "clasp_canonical_dual_recovery_coupling",
                            0.0,
                        )
                    ),
                    "clasp_canonical_recovery_temperature": float(
                        getattr(args, "clasp_canonical_recovery_temperature", 0.15)
                    ),
                    "clasp_canonical_max_subsets": int(
                        getattr(args, "clasp_canonical_max_subsets", 1024)
                    ),
                    "clasp_canonical_exact_scale": float(
                        getattr(args, "clasp_canonical_exact_scale", 1.0)
                    ),
                    "clasp_canonical_grpo_anchor_scale": float(
                        getattr(args, "clasp_canonical_grpo_anchor_scale", 0.0)
                    ),
                    "clasp_canonical_quality_anchor_ratio": float(
                        getattr(args, "clasp_canonical_quality_anchor_ratio", 0.0)
                    ),
                    "clasp_canonical_dense_quality_scale": float(
                        getattr(args, "clasp_canonical_dense_quality_scale", 0.0)
                    ),
                    "clasp_canonical_dense_feasibility_scale": float(
                        getattr(args, "clasp_canonical_dense_feasibility_scale", 0.0)
                    ),
                    "clasp_canonical_dense_violation_scale": float(
                        getattr(args, "clasp_canonical_dense_violation_scale", 0.0)
                    ),
                    "clasp_canonical_dense_recovery_scale": float(
                        getattr(args, "clasp_canonical_dense_recovery_scale", 0.0)
                    ),
                    "clasp_policy_length_alpha": float(
                        getattr(args, "clasp_policy_length_alpha", 0.0)
                    ),
                }
            )

            if bool(getattr(args, "clasp_proper_dual", True)):
                wrappers = getattr(reward_manager, "wrappers", None)
                if isinstance(wrappers, dict):
                    for wrapper in wrappers.values():
                        if hasattr(wrapper, "enable_clasp_proper_dual"):
                            wrapper.enable_clasp_proper_dual(
                                base_dual_lr=float(args.pd_dual_lr),
                                dual_update_interval=int(
                                    getattr(args, "clasp_dual_update_interval", 4)
                                ),
                                gradient_mode=str(
                                    getattr(
                                        args,
                                        "clasp_proper_dual_gradient_mode",
                                        "signed",
                                    )
                                ),
                            )

            print_primary(
                "[CLASP] enabled | "
                f"bon_aligned={os.environ.get('CLASP_BON_ALIGNED', '1')}, "
                f"bon_success_credit={os.environ.get('CLASP_BON_SUCCESS_CREDIT', 'dense_rank')}, "
                f"frontier_cdf={os.environ.get('CLASP_FRONTIER_CDF', '0')}, "
                f"raw_feasibility_targeting={bool(getattr(args, 'clasp_raw_feasibility_targeting', False))}, "
                f"strict_defaults={bool(getattr(args, 'clasp_strict_defaults', True))}, "
                f"legacy_pareto_anchor_ablation={trainer_extra_kwargs['clasp_pareto_anchor']}, "
                f"pareto_tau={trainer_extra_kwargs['clasp_pareto_tau']}, "
                f"pareto_center={trainer_extra_kwargs['clasp_pareto_center']}, "
                f"proper_dual={bool(getattr(args, 'clasp_proper_dual', True))}, "
                f"proper_dual_gradient={getattr(args, 'clasp_proper_dual_gradient_mode', 'signed')}, "
                f"component_dual_recovery_scale={os.environ.get('CLASP_COMPONENT_DUAL_RECOVERY_SCALE', '1.0')}, "
                f"dual_update_interval={int(getattr(args, 'clasp_dual_update_interval', 4))}, "
                f"adaptive_length={bool(getattr(args, 'clasp_adaptive_length', True))}, "
                f"mixed_adaptive_length_mode={getattr(args, 'clasp_mixed_adaptive_length_mode', 'error')}, "
                f"warmup_steps={int(getattr(args, 'clasp_feasibility_warmup_steps', 0) or 0)}, "
                f"self_replay={bool(getattr(args, 'clasp_self_replay', False))}"
            )

        # Capture the reset baseline after all controller modes are wired so an
        # explicit positive_excess ablation does not conflict with a stale
        # pre-wiring "signed" controller snapshot during resume fallback.
        reward_manager_initial_state = reward_manager.state_dict()
        callbacks.append(MultiTaskPrimalDualStateCallback(reward_manager))
        if bool(getattr(args, "clasp_validation_gate_checkpoints", False)):
            callbacks.append(
                ClaspPassAtKValidationGateCheckpointCallback(
                    getattr(args, "clasp_feasibility_warmup_target_map", {})
                )
            )

        schedule_enabled = bool(args.pd_enable_schedule)
        if bool(getattr(args, "use_clasp", False)) and schedule_enabled:
            print_primary(
                "[CLASP] Disabling pd_enable_schedule; CLASP uses its own "
                "feasibility-conditioned dual rule and fixed cost limits."
            )
            schedule_enabled = False

        if schedule_enabled:
            callbacks.append(
                MultiTaskPrimalDualScheduleCallback(
                    reward_manager,
                    stage_a_fraction=float(args.pd_stage_a_fraction),
                    stage_b_fraction=float(args.pd_stage_b_fraction),
                    stage_a_cost_limit=float(args.pd_stage_a_cost_limit),
                    stage_b_cost_limit=float(args.pd_stage_b_cost_limit),
                    stage_c_cost_limit=float(args.pd_stage_c_cost_limit),
                    stage_a_dual_lr_scale=float(args.pd_stage_a_dual_lr_scale),
                    stage_b_dual_lr_scale=float(args.pd_stage_b_dual_lr_scale),
                    stage_c_dual_lr_scale=float(args.pd_stage_c_dual_lr_scale),
                    stage_c_constraint_threshold=float(args.pd_stage_c_constraint_threshold),
                    base_dual_lr_up=float(
                        args.pd_dual_lr_up if args.pd_dual_lr_up is not None else args.pd_dual_lr
                    ),
                    base_dual_lr_down=float(
                        args.pd_dual_lr_down
                        if args.pd_dual_lr_down is not None
                        else args.pd_dual_lr
                    ),
                )
            )
        if (
            bool(getattr(args, "clasp_canonical_set", False))
            or bool(getattr(args, "clasp_fixed_null_set", False))
        ):
            # Exact set credit has exactly one operative controller: the
            # task-level failure@K multiplier committed by the trainer.  The
            # reward manager is retained as a strict evaluator, but its legacy
            # per-component multipliers must remain frozen.
            reward_manager.reset_pending()
            reward_manager.set_collect_stats(False)
            print_primary(
                "[CLASP][SetCredit] Per-component primal-dual updates disabled; "
                "only the signed failure@K set controller is active."
            )
        else:
            callbacks.append(MultiTaskPrimalDualUpdateCallback(reward_manager))
    elif runtime_profile is not None:
        reward_funcs = list(runtime_profile.reward_functions())
        trainer_cls = MixedBatchGRPOTrainer
        trainer_extra_kwargs = {}
        if getattr(runtime_profile, "external_baseline_extension", False):
            trainer_cls = _external_baseline_trainer_class()
            trainer_extra_kwargs = {
                "advantage_estimator": runtime_profile.advantage_estimator,
                "baseline_initial": runtime_profile.baseline_initial,
                "baseline_scale": runtime_profile.baseline_scale,
            }
            if callable(getattr(runtime_profile, "reward_contract", None)):
                trainer_extra_kwargs["reward_contract"] = runtime_profile.reward_contract()
            if getattr(runtime_profile, "feasible_quality_extension", False):
                trainer_cls = _feasible_quality_trainer_class()
                trainer_extra_kwargs["advantage_contract"] = runtime_profile.advantage_contract()
        elif getattr(runtime_profile, "verified_teacher_extension", False):
            trainer_cls = MixedBatchVerifiedTeacherGRPOTrainer
            trainer_extra_kwargs = {"verified_teacher_weight": runtime_profile.teacher_weight}
        if getattr(runtime_profile, "budget_matched_extension", False):
            from clasp.trainers.budget_matched_grpo import budget_matched_trainer_class

            trainer_cls = budget_matched_trainer_class(
                trainer_cls,
                external_baseline=bool(runtime_profile.external_baseline_extension),
            )
            trainer_extra_kwargs["budget_resume_contract"] = runtime_profile.resume_contract()
    else:
        reward_funcs = [
            build_mixed_reward_head(task_triplets, 0),
            build_mixed_reward_head(task_triplets, 1),
            build_mixed_reward_head(task_triplets, 2),
        ]
        trainer_cls = MixedBatchGRPOTrainer
        trainer_extra_kwargs = {}

    trainer = trainer_cls(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        **trainer_extra_kwargs,
    )
    if runtime_profile is not None:
        expected_reward_funcs = tuple(runtime_profile.reward_functions())
        if tuple(getattr(trainer, "reward_funcs", ())) != expected_reward_funcs:
            raise RuntimeError("GRPOTrainer did not preserve the ordered runtime-profile reward heads")

    fidelity_preflight = run_publication_fidelity_preflight(
        trainer,
        phase="pre_train",
    )
    if fidelity_preflight is not None:
        install_publication_fidelity_callback(trainer)
        print_primary(
            "[GRPO][PublicationFidelity] Exact distributed trainable-policy closure: "
            f"{fidelity_preflight['distributed']['consensus_digest']}"
        )

    if explicit_kl_reference_path is not None:
        if verified_kl_reference is None:
            raise RuntimeError("Explicit KL reference was not live-verified before Trainer setup.")
        if exact_peft_source_binding is None:
            (
                explicit_kl_reference_model,
                explicit_kl_reference_tokenizer,
                explicit_reference_is_peft,
            ) = load_model_and_tokenizer(
                explicit_kl_reference_path,
                args.dtype,
                padding_side="left",
                tokenizer_path=args.tokenizer_path,
            )
            if explicit_reference_is_peft:
                raise RuntimeError(
                    "merged-v2 KL reference unexpectedly loaded as a PEFT model: "
                    f"{explicit_kl_reference_path}"
                )
            del explicit_kl_reference_tokenizer
        if explicit_kl_reference_model is None:
            raise RuntimeError("explicit KL reference model was not independently loaded")
        explicit_kl_reference_model.requires_grad_(False)
        explicit_kl_reference_model.eval()
        if hasattr(explicit_kl_reference_model, "config") and hasattr(
            explicit_kl_reference_model.config,
            "use_cache",
        ):
            explicit_kl_reference_model.config.use_cache = False
        if bool(getattr(trainer, "is_deepspeed_enabled", False)) or bool(
            getattr(trainer, "is_fsdp_enabled", False)
        ):
            raise RuntimeError(
                "Explicit biased-LoRA KL references are currently supported only by the "
                "DDP/non-sharded CLASP launcher."
            )
        if getattr(trainer, "ref_model", None) is not None:
            raise RuntimeError("TRL unexpectedly constructed a second KL reference model")
        prepared_kl_reference = trainer.accelerator.prepare_model(
            explicit_kl_reference_model,
            evaluation_mode=True,
        )
        trainer.ref_model = prepared_kl_reference
        if exact_peft_source_binding is not None:
            policy_model = trainer.accelerator.unwrap_model(trainer.model)
            policy_embeddings = policy_model.get_input_embeddings()
            reference_embeddings = prepared_kl_reference.get_input_embeddings()
            if int(policy_embeddings.num_embeddings) != int(reference_embeddings.num_embeddings):
                raise RuntimeError(
                    "tokenizer-driven embedding resize differs across exact paired loads; "
                    "random embedding initialization is forbidden"
                )
            assert_exact_paired_special_token_ids(
                policy_model,
                prepared_kl_reference,
                tokenizer,
            )
            exact_kl_runtime_receipt = run_exact_peft_kl_runtime_preflight(
                trainer,
                prepared_kl_reference,
                exact_peft_source_binding,
                output_dir,
                dtype=args.dtype,
                beta=resolved_beta,
                sync_ref_model=bool(getattr(grpo_args, "sync_ref_model", False)),
                use_vllm=bool(getattr(grpo_args, "use_vllm", False)),
            )
            print_primary(
                "[GRPO][ExactKL] Persisted exact paired-reference consensus: "
                f"{exact_kl_runtime_receipt['receipt_sha256']}"
            )
        print_primary(
            "[GRPO] Explicit KL reference is frozen and device-prepared; "
            "adapter disabling will not be used for beta regularization."
        )

    if checkpoint_fidelity_enabled() and exact_kl_runtime_receipt is None:
        reject_beta_zero_exact_kl_sidecars(output_dir, checkpoint_output_dir)

    for callback in callbacks:
        trainer.add_callback(callback)

    resume_required_files = ()
    if args.use_primal_dual and args.pd_require_state_on_resume:
        resume_required_files = ("multitask_primal_dual_state.json",)

    explicit_resume_checkpoint = str(os.environ.get("RESUME_FROM_CHECKPOINT", "")).strip()
    if explicit_resume_checkpoint:
        if not os.path.isdir(explicit_resume_checkpoint):
            raise RuntimeError(
                "RESUME_FROM_CHECKPOINT was set, but it is not an existing checkpoint directory: "
                f"{explicit_resume_checkpoint}"
            )
        if checkpoint_fidelity_enabled():
            latest_checkpoint_dir = find_latest_checkpoint_dir(output_dir)
            if latest_checkpoint_dir is not None and (
                os.path.realpath(latest_checkpoint_dir)
                != os.path.realpath(explicit_resume_checkpoint)
            ):
                raise RuntimeError(
                    "Checkpoint-fidelity mode refuses an explicit resume that bypasses "
                    "a newer output checkpoint. "
                    f"latest={latest_checkpoint_dir}, explicit={explicit_resume_checkpoint}"
                )
        verify_publication_resume_checkpoint(
            explicit_resume_checkpoint,
            expected_kl_reference=exact_kl_runtime_receipt,
            trainer=trainer,
        )
        if args.use_primal_dual:
            maybe_load_primal_dual_state(
                output_dir,
                reward_manager,
                required=args.pd_require_state_on_resume,
                checkpoint_dir=explicit_resume_checkpoint,
                default_state=reward_manager_initial_state,
            )
        print_primary(f"[GRPO] Resuming from RESUME_FROM_CHECKPOINT={explicit_resume_checkpoint}")
        trainer.train(resume_from_checkpoint=explicit_resume_checkpoint)
        resumed = True
    else:
        resumed = False

    resumable_checkpoint_dirs = list_resumable_checkpoint_dirs(
        output_dir,
        required_files=resume_required_files,
    )
    if not resumed and checkpoint_fidelity_enabled():
        latest_checkpoint_dir = find_latest_checkpoint_dir(output_dir)
        newest_resumable = resumable_checkpoint_dirs[0] if resumable_checkpoint_dirs else None
        if latest_checkpoint_dir is not None and (
            newest_resumable is None
            or os.path.realpath(latest_checkpoint_dir) != os.path.realpath(newest_resumable)
        ):
            raise RuntimeError(
                "Checkpoint-fidelity mode found a newer checkpoint that is not the "
                "strict resume candidate; refusing fallback to an older checkpoint or fresh run. "
                f"latest={latest_checkpoint_dir}, candidate={newest_resumable}"
            )
    if not resumed and resumable_checkpoint_dirs:
        for checkpoint_dir in resumable_checkpoint_dirs:
            try:
                verify_publication_resume_checkpoint(
                    checkpoint_dir,
                    expected_kl_reference=exact_kl_runtime_receipt,
                    trainer=trainer,
                )
                if args.use_primal_dual:
                    maybe_load_primal_dual_state(
                        output_dir,
                        reward_manager,
                        required=args.pd_require_state_on_resume,
                        checkpoint_dir=checkpoint_dir,
                        default_state=reward_manager_initial_state,
                    )
                print_primary(
                    f"[GRPO] Found existing checkpoint(s); resuming training from: {checkpoint_dir}"
                )
                trainer.train(resume_from_checkpoint=checkpoint_dir)
                resumed = True
                break
            except Exception as exc:
                if checkpoint_fidelity_enabled():
                    raise RuntimeError(
                        "Checkpoint-fidelity mode forbids fallback after a resume "
                        f"failure at the newest eligible checkpoint: {checkpoint_dir}"
                    ) from exc
                if not is_likely_resume_load_error(exc):
                    raise
                print_primary(
                    "[GRPO][Warn] Failed to resume from checkpoint; "
                    f"trying an older checkpoint if available. "
                    f"checkpoint={checkpoint_dir} error={exc.__class__.__name__}: {exc}"
                )

        if not resumed:
            if args.use_primal_dual:
                reward_manager.load_state_dict(reward_manager_initial_state)
            print_primary(
                "[GRPO][Warn] Existing checkpoints were found, but none resumed cleanly. "
                "Starting fresh training."
            )
            trainer.train()
    elif not resumed and has_checkpoint_dir(output_dir):
        if checkpoint_fidelity_enabled():
            raise RuntimeError(
                "Checkpoint-fidelity mode found checkpoint directories without a complete "
                "resumable state; refusing to start fresh."
            )
        print_primary(
            "[GRPO][Warn] Checkpoint directories were found, but none contained a complete "
            "resumable trainer state. Starting fresh training."
        )
        trainer.train()
    elif not resumed:
        print_primary("[GRPO] No existing checkpoint found; starting fresh training.")
        trainer.train()

    final_global_step = int(getattr(trainer.state, "global_step", -1))
    trainer_max_steps = int(getattr(trainer.args, "max_steps", -1))
    contracted_max_steps = (
        int(training_run_contract_identity["fixed_horizon_max_steps"])
        if training_run_contract_identity is not None
        else trainer_max_steps
    )
    if trainer_max_steps != contracted_max_steps:
        raise RuntimeError(
            "Trainer max_steps differs from the immutable run-contract fixed horizon: "
            f"trainer={trainer_max_steps}, contract={contracted_max_steps}"
        )
    terminal_horizon = fixed_horizon_exit_is_terminal(
        final_global_step,
        contracted_max_steps,
    )
    if not terminal_horizon:
        retained_checkpoint = Path(str(trainer.args.output_dir)) / (
            f"checkpoint-{final_global_step}"
        )
        if not retained_checkpoint.is_dir():
            raise RuntimeError(
                "nonterminal GRPO chunk stopped without its exact retained checkpoint: "
                f"{retained_checkpoint}"
            )
        verify_publication_resume_checkpoint(
            retained_checkpoint,
            expected_kl_reference=exact_kl_runtime_receipt,
            trainer=trainer,
            expect_live_policy_matches_checkpoint=True,
        )
        print_primary(
            "[GRPO] Nonterminal fixed-horizon chunk completed at "
            f"step={final_global_step}/{contracted_max_steps}; retained checkpoint verified, "
            "root finalization intentionally skipped."
        )
        del trainer
        del model
        del tokenizer
        cleanup_torch_memory()
        return output_dir

    final_post_step = run_exact_peft_kl_post_step_audit(
        trainer,
        output_dir,
        artifact_role="final",
        global_step=final_global_step,
    )
    final_fidelity_preflight = run_publication_fidelity_preflight(
        trainer,
        phase="pre_final_save",
    )
    maybe_torch_distributed_barrier()
    if is_primary_process():
        trainer.save_model(output_dir)
        write_final_publication_fidelity(
            trainer,
            output_dir,
            expected_preflight=final_fidelity_preflight,
            expected_post_step=final_post_step,
        )
        trainer.save_state()
        tokenizer.save_pretrained(output_dir)
        if args.use_primal_dual:
            save_multitask_primal_dual_state(output_dir, reward_manager)
        write_prefixed_trainer_logs(output_dir, "grpo", trainer.state)
        write_stage_status(output_dir, "grpo", trainer.state)
    maybe_torch_distributed_barrier()
    print_primary(f"[GRPO] Saved model to: {output_dir}")
    del trainer
    del model
    del tokenizer
    cleanup_torch_memory()
    return output_dir


@_elastic_record
def main() -> None:
    args = parse_grpo_args()
    print_primary(
        "[Runtime] clasp.entrypoints.train_grpo active: "
        f"output_root={args.output_root}, report_to={args.report_to}"
    )
    run_grpo(args)


if __name__ == "__main__":
    main()
