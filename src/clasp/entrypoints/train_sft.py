"""Entry point: supervised fine-tuning (``clasp-sft``).

Wraps the TRL ``SFTTrainer`` with mixed-task batch sampling and optimizer-resume fallback.
This is the initialization stage for every RL variant.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import torch
from transformers import Trainer  # type: ignore[import-not-found]
from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

from clasp.config import parse_sft_args
from clasp.data.loaders import (
    active_tsp_representation_settings,
    build_mixed_task_sampler_for_trainer,
    convert_sft_dataset_to_text_only,
    inspect_sft_tokenization_alignment,
    load_mixed_sft_dataset,
)
from clasp.data.prompts import prompt_template_tag, resolve_prompt_style
from clasp.models.loading import (
    build_lora_config,
    load_model_and_tokenizer,
    lora_bias_from_env,
    lora_target_modules_from_env,
    peft_checkpoint_bias_policy,
    resolve_tokenizer_path,
)
from clasp.models.training_fidelity import (
    checkpoint_fidelity_enabled,
    checkpoint_fidelity_mode,
    install_publication_fidelity_callback,
    require_publication_lora_configuration,
    run_publication_fidelity_preflight,
    verify_publication_input_adapter,
    verify_publication_resume_checkpoint,
    write_final_publication_fidelity,
)
from clasp.runtime import (
    build_run_payload,
    build_sft_run_name,
    cleanup_torch_memory,
    configure_wandb_env,
    enforce_sft_context_limit,
    find_latest_checkpoint_dir,
    has_checkpoint_dir,
    is_likely_resume_load_error,
    is_primary_process,
    list_resumable_checkpoint_dirs,
    maybe_torch_distributed_barrier,
    print_primary,
    resolve_base_model_name,
    resolve_checkpoint_output_dir,
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


def strict_sft_resume_required() -> bool:
    """Return whether this process must resume exact trainer state.

    Long SFT jobs can opt into this at continuation boundaries.  It remains
    default-off so existing fresh launches retain their historical behavior.
    """

    value = str(os.environ.get("CLASP_SFT_REQUIRE_RESUME", "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def strict_sft_completion_only_required() -> bool:
    """Return whether prompt/completion masking is a mandatory run contract."""

    value = str(os.environ.get("CLASP_SFT_REQUIRE_COMPLETION_ONLY", "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sft_memory_safe_loss_enabled() -> bool:
    """Return whether to skip TRL's full-logit diagnostic copies.

    TRL computes the ordinary Transformers training loss first, then copies the
    complete shifted-logits tensor to report entropy and token accuracy.  That
    diagnostic copy can exceed H100 memory at the Jiang 20k context.  This
    opt-in path preserves the same model loss and gradients while omitting only
    those auxiliary metrics.
    """

    value = str(os.environ.get("CLASP_SFT_MEMORY_SAFE_LOSS", "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sft_supervised_logits_only_enabled() -> bool:
    """Return whether the LM head should run only at supervised positions.

    Completion-only SFT masks every prompt label with ``-100``.  Computing
    vocabulary logits for those prompt positions is therefore unnecessary,
    and at a 20k context it dominates memory even when only LoRA parameters are
    trainable.  Qwen2 can select sequence positions before its LM head via
    ``logits_to_keep`` and accepts the corresponding pre-shifted labels.
    """

    value = str(os.environ.get("CLASP_SFT_SUPERVISED_LOGITS_ONLY", "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def completion_supervised_logit_selection(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select causal predictor positions having any non-ignored target.

    ``labels[:, token]`` is predicted by hidden state ``token - 1``.  A single
    sequence-position index is shared across the batch, so for batch sizes
    above one we retain the union of supervised positions and preserve
    per-example ``-100`` labels inside that union.
    """

    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise RuntimeError(
            "Supervised-logits-only SFT requires a rank-2 labels tensor shaped [batch, sequence]."
        )
    if labels.shape[1] < 2:
        raise RuntimeError(
            "Supervised-logits-only SFT requires sequences with at least two tokens."
        )

    causal_labels = labels[:, 1:]
    active_positions = torch.nonzero(
        causal_labels.ne(-100).any(dim=0),
        as_tuple=False,
    ).flatten()
    if active_positions.numel() == 0:
        raise RuntimeError(
            "Completion-only SFT batch has no supervised target tokens after truncation; "
            "refusing a zero-signal optimizer step."
        )
    selected_shift_labels = causal_labels.index_select(1, active_positions).contiguous()
    return active_positions, selected_shift_labels


def model_supports_supervised_logits_only(model) -> bool:
    """Check the exact Qwen/Transformers forward-loss contract fail-closed."""

    candidate = model
    get_base_model = getattr(candidate, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidate = get_base_model()
        except Exception:
            return False

    try:
        forward_parameters = inspect.signature(candidate.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    if "logits_to_keep" not in forward_parameters:
        return False

    try:
        loss_function = candidate.loss_function
        loss_parameters = inspect.signature(loss_function).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return "shift_labels" in loss_parameters


def list_strict_sft_resume_checkpoint_dirs(output_dir: str) -> list[str]:
    """Return checkpoints that can preserve the full optimization trajectory."""

    candidates = list_resumable_checkpoint_dirs(
        output_dir,
        required_files=("optimizer.pt", "scheduler.pt"),
    )
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
    except ValueError:
        world_size = 1
    expected_max_steps_raw = str(os.environ.get("CLASP_SFT_EXPECTED_MAX_STEPS", "") or "").strip()
    expected_max_steps = None
    if expected_max_steps_raw:
        try:
            expected_max_steps = int(expected_max_steps_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"CLASP_SFT_EXPECTED_MAX_STEPS must be an integer; got {expected_max_steps_raw!r}."
            ) from exc

    strict: list[str] = []
    for checkpoint_dir in candidates:
        try:
            names = set(os.listdir(checkpoint_dir))
        except OSError:
            continue
        match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", os.path.basename(checkpoint_dir))
        if match is None or "adapter_config.json" not in names:
            continue
        try:
            with open(
                os.path.join(checkpoint_dir, "trainer_state.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                trainer_state = json.load(handle)
            if int(trainer_state.get("global_step", -1)) != int(match.group(1)):
                continue
            if (
                expected_max_steps is not None
                and int(trainer_state.get("max_steps", -1)) != expected_max_steps
            ):
                continue
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if world_size > 1:
            rng_state_complete = all(f"rng_state_{rank}.pth" in names for rank in range(world_size))
        else:
            rng_state_complete = "rng_state.pth" in names
        if rng_state_complete:
            strict.append(checkpoint_dir)
    return strict


def select_pinned_strict_sft_resume_checkpoint(output_dir: str) -> str:
    """Return the exact checkpoint approved by the external run contract."""

    requested = str(os.environ.get("CLASP_SFT_RESUME_CHECKPOINT", "") or "").strip()
    if not requested:
        raise RuntimeError(
            "CLASP_SFT_REQUIRE_RESUME=1 requires CLASP_SFT_RESUME_CHECKPOINT to "
            "name the checkpoint approved by the continuation contract."
        )
    requested_real = os.path.realpath(requested)
    by_realpath = {
        os.path.realpath(candidate): candidate
        for candidate in list_strict_sft_resume_checkpoint_dirs(output_dir)
    }
    if requested_real not in by_realpath:
        raise RuntimeError(
            "The contract-pinned SFT checkpoint is not a complete strict checkpoint "
            f"for this output directory: {requested}."
        )
    return by_realpath[requested_real]


class _OptimizerResumeFallbackMixin:
    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint and checkpoint_fidelity_enabled():
            run_publication_fidelity_preflight(
                self,
                phase=f"post_resume_model_load:{os.path.abspath(str(checkpoint))}",
            )
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        except ValueError as exc:
            message = str(exc)
            if "parameter group that doesn't match the size of optimizer's group" not in message:
                raise
            if strict_sft_resume_required() or checkpoint_fidelity_enabled():
                raise RuntimeError(
                    "Strict SFT continuation requires the saved optimizer and scheduler state; "
                    "refusing the fresh-optimizer compatibility fallback."
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
                "[SFT][Warn] Optimizer state is incompatible with current trainable parameter groups; "
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


class MixedBatchSFTTrainer(_OptimizerResumeFallbackMixin, SFTTrainer):
    def __init__(
        self,
        *args,
        numeric_token_loss_weight: float = 1.0,
        memory_safe_loss: bool = False,
        supervised_logits_only: bool = False,
        **kwargs,
    ):
        tokenizer = kwargs.get("processing_class")
        super().__init__(*args, **kwargs)
        self.numeric_token_loss_weight = max(1.0, float(numeric_token_loss_weight or 1.0))
        self.memory_safe_loss = bool(memory_safe_loss)
        self.supervised_logits_only = bool(supervised_logits_only)
        self._logged_supervised_logits_selection = False
        self._numeric_token_mask_cpu = None
        if self.numeric_token_loss_weight > 1.0:
            tokenizer = (
                tokenizer
                or getattr(self, "processing_class", None)
                or getattr(self, "tokenizer", None)
            )
            if tokenizer is None:
                print_primary(
                    "[SFT][Warn] numeric_token_loss_weight requested but no tokenizer is available; "
                    "falling back to standard SFT loss."
                )
                self.numeric_token_loss_weight = 1.0
            else:
                vocab_size = int(len(tokenizer))
                mask = torch.zeros(vocab_size, dtype=torch.bool)
                for token_id in range(vocab_size):
                    piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    if re.search(r"\d|[\[\],]", piece):
                        mask[token_id] = True
                self._numeric_token_mask_cpu = mask
                print_primary(
                    "[SFT] Numeric/list-token weighted loss enabled: "
                    f"weight={self.numeric_token_loss_weight}, weighted_tokens={int(mask.sum())}/{vocab_size}"
                )

    def _get_train_sampler(self, train_dataset=None):
        dataset = (
            train_dataset if train_dataset is not None else getattr(self, "train_dataset", None)
        )
        sampler = build_mixed_task_sampler_for_trainer(dataset, self.args)
        if sampler is None:
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()
        return sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.memory_safe_loss:
            if self.numeric_token_loss_weight > 1.0:
                raise RuntimeError(
                    "CLASP_SFT_MEMORY_SAFE_LOSS=1 is incompatible with "
                    "numeric_token_loss_weight > 1."
                )
            model_inputs = dict(inputs)
            model_inputs["use_cache"] = False
            if self.supervised_logits_only:
                labels = model_inputs.get("labels")
                active_positions, selected_shift_labels = completion_supervised_logit_selection(
                    labels
                )
                model_inputs["logits_to_keep"] = active_positions
                model_inputs["shift_labels"] = selected_shift_labels
                if not self._logged_supervised_logits_selection:
                    sequence_length = int(labels.shape[1])
                    print_primary(
                        "[SFT] Supervised-logits-only first batch: "
                        f"sequence_positions={sequence_length}, "
                        f"lm_head_positions={int(active_positions.numel())}, "
                        f"avoided_positions={sequence_length - int(active_positions.numel())}"
                    )
                    self._logged_supervised_logits_selection = True
            try:
                return Trainer.compute_loss(
                    self,
                    model,
                    model_inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return Trainer.compute_loss(
                    self,
                    model,
                    model_inputs,
                    return_outputs=return_outputs,
                )

        if (
            self.numeric_token_loss_weight <= 1.0
            or self._numeric_token_mask_cpu is None
            or "labels" not in inputs
        ):
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        per_token_loss = per_token_loss.view_as(shift_labels)

        label_mask = shift_labels.ne(-100)
        token_mask = self._numeric_token_mask_cpu.to(device=shift_labels.device)
        safe_labels = shift_labels.clamp(min=0, max=token_mask.numel() - 1)
        numeric_mask = token_mask[safe_labels] & label_mask
        weights = torch.ones_like(per_token_loss)
        weights = torch.where(
            numeric_mask,
            torch.full_like(weights, float(self.numeric_token_loss_weight)),
            weights,
        )
        weighted_mask = label_mask.to(per_token_loss.dtype) * weights
        loss = (per_token_loss * weighted_mask).sum() / weighted_mask.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def run_sft(args) -> str:
    initialization_fidelity = verify_publication_input_adapter(
        args.model_path,
        purpose="SFT initialization adapter",
    )
    resolved_tokenizer_path = resolve_tokenizer_path(args.model_path, args.tokenizer_path)
    model, tokenizer, model_is_peft = load_model_and_tokenizer(
        args.model_path,
        args.dtype,
        padding_side="right",
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
        generation_config.use_cache = False

    train_dataset, eval_dataset = load_mixed_sft_dataset(
        tasks=args.tasks,
        data_root=args.data_root,
        eval_ratio=args.eval_ratio,
        eval_from_train_split=args.eval_from_train_split,
        seed=args.seed,
        eos_token=tokenizer.eos_token or "",
        max_samples_per_task=max(0, int(args.max_samples_per_task)),
        max_eval_samples_per_task=max(0, int(args.max_eval_samples_per_task)),
        min_train_samples_per_task=max(0, int(args.min_train_samples_per_task)),
        min_eval_samples_per_task=max(0, int(args.min_eval_samples_per_task)),
        prefer_subset_files=args.prefer_subset_files,
        deduplicate_train_rows=args.deduplicate_train_rows,
        no_reason=args.no_reason,
        prompt_style=prompt_style,
        tokenizer=tokenizer,
        strict_chat_prompt=strict_chat_prompt,
        train_size_bucket=args.train_size_bucket,
        train_min_instance_size=args.train_min_instance_size,
        train_max_instance_size=args.train_max_instance_size,
    )

    train_alignment = inspect_sft_tokenization_alignment(train_dataset, tokenizer, max_samples=256)
    eval_alignment = (
        inspect_sft_tokenization_alignment(eval_dataset, tokenizer, max_samples=256)
        if len(eval_dataset) > 0
        else {"checked": 0, "mismatches": 0, "examples": []}
    )
    train_mismatches = int(train_alignment["mismatches"])
    eval_mismatches = int(eval_alignment["mismatches"])
    use_text_only_sft = (train_mismatches + eval_mismatches) > 0
    require_completion_only = strict_sft_completion_only_required()
    memory_safe_loss = sft_memory_safe_loss_enabled()
    supervised_logits_only = sft_supervised_logits_only_enabled()

    if memory_safe_loss and float(args.numeric_token_loss_weight) > 1.0:
        raise RuntimeError("CLASP_SFT_MEMORY_SAFE_LOSS=1 requires numeric_token_loss_weight <= 1.")
    if supervised_logits_only and not memory_safe_loss:
        raise RuntimeError(
            "CLASP_SFT_SUPERVISED_LOGITS_ONLY=1 requires CLASP_SFT_MEMORY_SAFE_LOSS=1."
        )
    if supervised_logits_only and not require_completion_only:
        raise RuntimeError(
            "CLASP_SFT_SUPERVISED_LOGITS_ONLY=1 requires the fail-closed "
            "CLASP_SFT_REQUIRE_COMPLETION_ONLY=1 contract."
        )
    if supervised_logits_only and not model_supports_supervised_logits_only(model):
        raise RuntimeError(
            "CLASP_SFT_SUPERVISED_LOGITS_ONLY=1 requires a model forward method with "
            "logits_to_keep and a causal loss accepting shift_labels; the loaded model "
            f"{model.__class__.__name__} does not expose that contract."
        )
    if memory_safe_loss:
        print_primary(
            "[SFT] Memory-safe loss enabled: using the standard Transformers model loss "
            "without TRL full-logit entropy/token-accuracy diagnostics."
        )
    if supervised_logits_only:
        print_primary(
            "[SFT] Supervised-logits-only loss enabled: the LM head and causal loss "
            "will run only at completion-supervised sequence positions."
        )

    print_primary(
        "[SFT] Tokenization alignment check: "
        f"train={train_mismatches}/{int(train_alignment['checked'])} mismatches, "
        f"eval={eval_mismatches}/{int(eval_alignment['checked'])} mismatches"
    )

    if require_completion_only and use_text_only_sft:
        raise RuntimeError(
            "CLASP_SFT_REQUIRE_COMPLETION_ONLY=1, but prompt/completion tokenization "
            "alignment has mismatches; refusing the text-only fallback."
        )

    if use_text_only_sft:
        sample_examples = list(train_alignment["examples"]) + list(eval_alignment["examples"])
        if sample_examples:
            first_example = sample_examples[0]
            print_primary(
                "[SFT][Warn] prompt/completion tokenization mismatch detected; "
                "switching to text-only SFT dataset mode. "
                f"Example idx={first_example.get('index')}, "
                f"prompt_tail={first_example.get('prompt_tail')!r}, "
                f"completion_head={first_example.get('completion_head')!r}"
            )
        else:
            print_primary(
                "[SFT][Warn] prompt/completion tokenization mismatch detected; "
                "switching to text-only SFT dataset mode."
            )
        train_dataset = convert_sft_dataset_to_text_only(train_dataset)
        if len(eval_dataset) > 0:
            eval_dataset = convert_sft_dataset_to_text_only(eval_dataset)

    max_seq_length, context_info = enforce_sft_context_limit(args.max_seq_length, model, tokenizer)
    base_model_name = resolve_base_model_name(args.model_path)
    run_name = build_sft_run_name(
        base_model_name,
        prompt_style=requested_prompt_style,
        no_reason=args.no_reason,
        num_tasks=len(args.tasks),
        dtype=args.dtype,
        full_finetune=args.full_finetune,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_seq_length=max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    output_dir = resolve_output_dir(
        output_root=args.output_root,
        stage="sft",
        run_name=run_name,
        explicit_output_dir=args.output_dir,
    )
    checkpoint_output_dir = resolve_checkpoint_output_dir(output_dir)
    wandb_run_name = resolve_wandb_run_name(run_name)

    configure_wandb_env(
        report_to=args.report_to,
        project=args.wandb_project,
        group=args.wandb_group,
        stage="sft",
        run_name=wandb_run_name,
    )

    print_primary(f"[SFT] Output directory: {output_dir}")
    print_primary(f"[SFT] Run name: {run_name}")
    if wandb_run_name != run_name:
        print_primary(f"[SFT] W&B run name: {wandb_run_name}")
    print_primary(f"[SFT] Prompt template: {prompt_template}")
    print_primary(
        f"[SFT] Prompt style (requested -> resolved): {requested_prompt_style} -> {prompt_style}"
    )
    print_primary(f"[SFT] Tokenizer source: {resolved_tokenizer_path}")
    lora_target_modules = lora_target_modules_from_env()
    checkpoint_lora_bias = peft_checkpoint_bias_policy(args.model_path)
    effective_lora_bias = checkpoint_lora_bias or lora_bias_from_env()
    require_publication_lora_configuration(
        full_finetune=bool(args.full_finetune),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=lora_target_modules,
        effective_bias=effective_lora_bias,
    )
    print_primary(f"[SFT] LoRA target modules: {','.join(lora_target_modules)}")
    print_primary(f"[SFT] Effective LoRA bias policy: {effective_lora_bias}")

    optional_kwargs = {}
    sft_signature = inspect.signature(SFTConfig.__init__).parameters
    completion_only_loss_enabled = False
    if "completion_only_loss" in sft_signature:
        completion_only_loss_enabled = not use_text_only_sft
        optional_kwargs["completion_only_loss"] = completion_only_loss_enabled
    elif require_completion_only:
        raise RuntimeError(
            "CLASP_SFT_REQUIRE_COMPLETION_ONLY=1, but this TRL build does not expose "
            "SFTConfig.completion_only_loss."
        )
    if "seed" in sft_signature:
        optional_kwargs["seed"] = args.seed
    if "data_seed" in sft_signature:
        optional_kwargs["data_seed"] = args.seed
    if "ddp_find_unused_parameters" in sft_signature:
        optional_kwargs["ddp_find_unused_parameters"] = False
    if "gradient_checkpointing" in sft_signature:
        if "gradient_checkpointing_kwargs" in sft_signature:
            optional_kwargs["gradient_checkpointing"] = True
            optional_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        else:
            optional_kwargs["gradient_checkpointing"] = False
            print_primary(
                "[SFT][Warn] This TRL/Transformers build does not expose "
                "gradient_checkpointing_kwargs; disabling gradient checkpointing to avoid "
                "DDP reentrant-backward failures."
            )
    if "ddp_static_graph" in sft_signature:
        optional_kwargs["ddp_static_graph"] = True
    if "ddp_timeout" in sft_signature and int(args.ddp_timeout or 0) > 0:
        optional_kwargs["ddp_timeout"] = int(args.ddp_timeout)
    requested_sft_controls = {
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
    }
    for control_name, control_value in requested_sft_controls.items():
        if control_value is None:
            continue
        if control_name not in sft_signature:
            raise RuntimeError(
                f"This TRL build cannot pin requested SFT control {control_name!r}; "
                "refusing to inherit a version-dependent default."
            )
        optional_kwargs[control_name] = control_value

    eval_strategy = "steps" if len(eval_dataset) > 0 else "no"
    sft_args = SFTConfig(
        output_dir=checkpoint_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_length=max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=eval_strategy,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=wandb_run_name,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        **optional_kwargs,
    )

    write_hyperparams(
        output_dir,
        build_run_payload(
            stage="sft",
            run_name=run_name,
            base_model_name=base_model_name,
            model_path=args.model_path,
            output_dir=output_dir,
            tasks=args.tasks,
            report_to=args.report_to,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            stage_hyperparameters={
                "dtype": args.dtype,
                "full_finetune": args.full_finetune,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": lora_target_modules,
                "lora_bias": effective_lora_bias,
                "checkpoint_fidelity_mode": checkpoint_fidelity_mode(),
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
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_train_batch_size": args.effective_train_batch_size,
                "learning_rate": args.learning_rate,
                "optim": args.optim,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "lr_scheduler_type": args.lr_scheduler_type,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "max_seq_length": max_seq_length,
                "min_train_samples_per_task": args.min_train_samples_per_task,
                "min_eval_samples_per_task": args.min_eval_samples_per_task,
                "deduplicate_train_rows": bool(args.deduplicate_train_rows),
                "min_task_types_per_local_batch": (
                    min(2, int(args.per_device_train_batch_size)) if len(args.tasks) > 1 else 1
                ),
                "no_reason": bool(args.no_reason),
                "prompt_style": prompt_style,
                "prompt_style_requested": requested_prompt_style,
                "prompt_template": prompt_template,
                "tokenizer_path": resolved_tokenizer_path,
                # Evaluation must reconstruct the exact task representation used to
                # build SFT prompts and labels.  Despite the legacy helper name this
                # includes both TSP and MIS representation switches.
                **active_tsp_representation_settings(),
                "tokenization_alignment_train_mismatches": train_mismatches,
                "tokenization_alignment_train_checked": int(train_alignment["checked"]),
                "tokenization_alignment_eval_mismatches": eval_mismatches,
                "tokenization_alignment_eval_checked": int(eval_alignment["checked"]),
                "sft_text_only_mode": use_text_only_sft,
                "completion_only_loss": completion_only_loss_enabled,
                "numeric_token_loss_weight": args.numeric_token_loss_weight,
                "memory_safe_loss": memory_safe_loss,
                "supervised_logits_only": supervised_logits_only,
                "ddp_find_unused_parameters": optional_kwargs.get("ddp_find_unused_parameters"),
                "gradient_checkpointing": optional_kwargs.get("gradient_checkpointing"),
                "gradient_checkpointing_kwargs": optional_kwargs.get(
                    "gradient_checkpointing_kwargs"
                ),
                "ddp_static_graph": optional_kwargs.get("ddp_static_graph"),
                "ddp_timeout": optional_kwargs.get("ddp_timeout"),
            },
            context_info=context_info,
            args_namespace=args,
            trainer_args=sft_args,
            checkpoint_output_dir=checkpoint_output_dir,
        ),
    )

    peft_config = None
    if not model_is_peft and not args.full_finetune:
        peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    trainer = MixedBatchSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        numeric_token_loss_weight=args.numeric_token_loss_weight,
        memory_safe_loss=memory_safe_loss,
        supervised_logits_only=supervised_logits_only,
    )

    fidelity_preflight = run_publication_fidelity_preflight(
        trainer,
        phase="pre_train",
    )
    base_sft_resume_fidelity = str(
        os.environ.get("CLASP_BASE_SFT_RESUME_FIDELITY", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if base_sft_resume_fidelity and fidelity_preflight is None:
        raise RuntimeError("CLASP_BASE_SFT_RESUME_FIDELITY=1 requires checkpoint-fidelity mode")
    if fidelity_preflight is not None:
        if base_sft_resume_fidelity:
            from base_sft_resume_fidelity import (  # type: ignore[import-not-found]
                install_base_sft_resume_fidelity_callback,
            )

            install_base_sft_resume_fidelity_callback(trainer)
        else:
            install_publication_fidelity_callback(trainer)
        print_primary(
            "[SFT][PublicationFidelity] Exact distributed trainable-policy closure: "
            f"{fidelity_preflight['distributed']['consensus_digest']}"
        )

    require_resume = strict_sft_resume_required()
    resumable_checkpoint_dirs = (
        [select_pinned_strict_sft_resume_checkpoint(output_dir)]
        if require_resume
        else list_resumable_checkpoint_dirs(output_dir)
    )
    if checkpoint_fidelity_enabled():
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
    resumed = False
    last_resume_error = None
    if resumable_checkpoint_dirs:
        for checkpoint_dir in resumable_checkpoint_dirs:
            try:
                verify_publication_resume_checkpoint(checkpoint_dir)
                resume_load_path = checkpoint_dir
                if base_sft_resume_fidelity:
                    from base_sft_resume_fidelity import (  # type: ignore[import-not-found]
                        finalize_pinned_resume_for_trainer,
                        pin_verified_resume_for_trainer,
                        release_pinned_resume_for_trainer,
                    )

                    resume_load_path = pin_verified_resume_for_trainer(
                        trainer,
                        Path(checkpoint_dir),
                    )
                print_primary(
                    f"[SFT] Found existing checkpoint(s); resuming training from: {checkpoint_dir}"
                )
                try:
                    trainer.train(resume_from_checkpoint=resume_load_path)
                except Exception:
                    if base_sft_resume_fidelity:
                        release_pinned_resume_for_trainer(trainer)
                    raise
                if base_sft_resume_fidelity:
                    finalize_pinned_resume_for_trainer(trainer)
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
                last_resume_error = exc
                print_primary(
                    "[SFT][Warn] Failed to resume from checkpoint; "
                    f"trying an older checkpoint if available. "
                    f"checkpoint={checkpoint_dir} error={exc.__class__.__name__}: {exc}"
                )

        if not resumed:
            if require_resume:
                raise RuntimeError(
                    "CLASP_SFT_REQUIRE_RESUME=1, but no existing SFT checkpoint resumed "
                    "cleanly; refusing to start fresh."
                ) from last_resume_error
            print_primary(
                "[SFT][Warn] Existing checkpoints were found, but none resumed cleanly. "
                "Starting fresh training."
            )
            trainer.train()
    elif has_checkpoint_dir(output_dir):
        if require_resume or checkpoint_fidelity_enabled():
            raise RuntimeError(
                "Strict checkpoint fidelity was requested, but checkpoint directories "
                "lack a complete resumable trainer state; refusing to start fresh."
            )
        print_primary(
            "[SFT][Warn] Checkpoint directories were found, but none contained a complete "
            "resumable trainer state. Starting fresh training."
        )
        trainer.train()
    else:
        if require_resume:
            raise RuntimeError(
                "CLASP_SFT_REQUIRE_RESUME=1, but no existing checkpoint was found; "
                "refusing to start fresh."
            )
        print_primary("[SFT] No existing checkpoint found; starting fresh training.")
        trainer.train()

    trainer.save_state()
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
        )
        tokenizer.save_pretrained(output_dir)
        write_prefixed_trainer_logs(output_dir, "sft", trainer.state)
        write_stage_status(output_dir, "sft", trainer.state)
    maybe_torch_distributed_barrier()
    print_primary(f"[SFT] Saved model to: {output_dir}")
    del trainer
    del model
    del tokenizer
    cleanup_torch_memory()
    return output_dir


@_elastic_record
def main() -> None:
    args = parse_sft_args()
    print_primary(
        "[Runtime] clasp.entrypoints.train_sft active: "
        f"output_root={args.output_root}, report_to={args.report_to}"
    )
    run_sft(args)


if __name__ == "__main__":
    main()
