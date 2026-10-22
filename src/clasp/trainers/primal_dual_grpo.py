"""The active constrained GRPO trainer.

:class:`PrimalDualGRPOTrainer` extends the TRL ``GRPOTrainer`` to replace plain
group-centred advantages with constraint-aware ones: it scores each sampled group with the
per-task reward heads, applies the selected CLASP advantage transform from
:mod:`clasp.rewards.clasp_math`, drives dual (lambda) ascent from the observed group
failure rate, and adapts the per-task completion-length budget.

Advantage math is kept in :mod:`clasp.rewards.clasp_math` as pure tensor functions so it
stays unit-testable without instantiating a trainer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict, deque

import torch  # type: ignore[import-not-found]
from trl import GRPOTrainer  # type: ignore[import-not-found]

_FRONTIER_CDF_DEFAULT_GAP_TARGETS = {
    "tsp": 0.0107,
    "cvrp": 0.0453,
    "op": 0.0185,
    "mis": 0.0104,
    "jssp": 0.0820,
}

DUAL_SEPARATE_Z_FORMULA_ID = (
    "group_z_reward_minus_group_z_positive_component_penalty_v1"
)


def _log_gpu_memory(label: str) -> None:
    """Log GPU memory usage for OOM debugging. Enable via CONSTRAINED_GRPO_DEBUG_MEMORY=1."""
    if not os.environ.get("CONSTRAINED_GRPO_DEBUG_MEMORY"):
        return
    if not torch.cuda.is_available():
        return
    try:
        rank = int(os.environ.get("RANK", "0"))
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_mem / (1024 ** 3)
        print(
            f"[GPU-Mem][rank={rank}] {label}: "
            f"allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB, total={total_gb:.2f}GB, "
            f"free≈{total_gb - reserved_gb:.2f}GB",
            flush=True,
        )
    except Exception:
        pass


def _gather_tensor(trainer: GRPOTrainer, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    try:
        from trl.trainer.grpo_trainer import gather  # type: ignore[import-not-found]

        return gather(tensor)
    except Exception:
        return trainer.accelerator.gather(tensor)


def _nonnegative_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return float(default)
    try:
        value = float(raw_value)
    except ValueError:
        return float(default)
    if not torch.isfinite(torch.tensor(value)).item() or value < 0.0:
        return float(default)
    return float(value)


def _finite_float_env(name: str, default: float) -> float:
    """Parse a controller-critical float without silently masking typos."""

    raw_value = os.environ.get(name, "")
    if not raw_value:
        return float(default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float, got {raw_value!r}") from exc
    if not torch.isfinite(torch.tensor(value)).item():
        raise ValueError(f"{name} must be finite, got {raw_value!r}")
    return float(value)


def _strict_nonnegative_float_env(name: str, default: float) -> float:
    value = _finite_float_env(name, default)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def _nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return int(default)
    try:
        value = int(raw_value)
    except ValueError:
        return int(default)
    if value < 0:
        return int(default)
    return int(value)


def _strict_positive_int_env(name: str, default: int) -> int:
    """Parse an experiment-critical positive integer without typo fallback."""

    raw_value = os.environ.get(name, "")
    if not raw_value:
        value = int(default)
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return int(value)


def _env_name_set(name: str) -> set[str] | None:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return None
    normalized = raw_value
    for sep in ("+", ";", ":", ","):
        normalized = normalized.replace(sep, " ")
    values = {part.strip().lower() for part in normalized.split() if part.strip()}
    return values or None


def _env_task_thresholds(name: str) -> dict[str, float]:
    raw_value = os.environ.get(name, "")
    thresholds: dict[str, float] = {}
    if not raw_value:
        return thresholds
    for raw_item in str(raw_value).split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            continue
        task_name, raw_threshold = item.split(":", 1)
        task_name = task_name.strip().lower()
        try:
            threshold = float(raw_threshold)
        except ValueError:
            continue
        if task_name and 0.0 <= threshold <= 1.0:
            thresholds[task_name] = threshold
    return thresholds


def _strict_env_task_thresholds(name: str) -> dict[str, float]:
    raw_value = os.environ.get(name, "")
    thresholds: dict[str, float] = {}
    if not raw_value:
        return thresholds
    for raw_item in str(raw_value).split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            raise ValueError(f"{name} entries must use task:threshold, got {item!r}")
        task_name, raw_threshold = item.split(":", 1)
        task_name = task_name.strip().lower()
        if not task_name:
            raise ValueError(f"{name} contains an empty task name")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(
                f"{name} threshold for {task_name!r} is not numeric: {raw_threshold!r}"
            ) from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"{name} threshold for {task_name!r} must be finite and in [0, 1]"
            )
        if task_name in thresholds:
            raise ValueError(f"{name} contains duplicate task {task_name!r}")
        thresholds[task_name] = threshold
    return thresholds


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return bool(default)
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _strict_env_flag(name: str, default: bool = False) -> bool:
    """Parse a new experiment switch without silently disabling a typo."""

    raw_value = os.environ.get(name, "")
    if not raw_value:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, or off/on; "
        f"got {raw_value!r}"
    )


def _strict_env_choice(name: str, default: str, choices: set[str]) -> str:
    """Parse an experiment-critical categorical switch without typo fallback."""

    raw_value = os.environ.get(name, "")
    normalized = str(raw_value or default).strip().lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{allowed}}}, got {raw_value!r}")
    return normalized


def _compose_clasp_recovery_gaps(
    raw_positive_gaps: torch.Tensor,
    dual_aware_recovery_gaps: torch.Tensor,
    *,
    component_dual_scale: float,
) -> torch.Tensor:
    """Combine fixed violation geometry with learned per-component priorities.

    Both inputs use the same bounded, deduplicated lexicographic geometry.
    ``dual_aware_recovery_gaps`` differs only through effective family weights
    ``component_weight * (1 + lambda)``.  Convex interpolation makes learned
    multipliers causal while preserving the hard parsed-before-unparseable tier
    for every finite non-negative scale.
    """

    if raw_positive_gaps.dim() != 1 or dual_aware_recovery_gaps.dim() != 1:
        raise ValueError("CLASP recovery gap tensors must be one-dimensional")
    if raw_positive_gaps.numel() != dual_aware_recovery_gaps.numel():
        raise ValueError("CLASP raw/component-dual recovery gap size mismatch")
    if not math.isfinite(component_dual_scale) or component_dual_scale < 0.0:
        raise ValueError("component_dual_scale must be finite and >= 0")
    if not bool(torch.isfinite(raw_positive_gaps).all().item()):
        raise ValueError("raw_positive_gaps contains non-finite values")
    if not bool(torch.isfinite(dual_aware_recovery_gaps).all().item()):
        raise ValueError("dual_aware_recovery_gaps contains non-finite values")
    if bool((raw_positive_gaps < 0.0).any().item()):
        raise ValueError("raw_positive_gaps must be non-negative")
    if bool((dual_aware_recovery_gaps < 0.0).any().item()):
        raise ValueError("dual_aware_recovery_gaps must be non-negative")
    scale = float(component_dual_scale)
    return (raw_positive_gaps + scale * dual_aware_recovery_gaps) / (1.0 + scale)


def _canonical_task_order(
    local_tasks: list[str],
    *,
    gathered_task_lists: list[list[str]] | None = None,
    configured_tasks: list[str] | None = None,
) -> tuple[str, ...]:
    """Return one deterministic task axis for distributed reductions.

    A rank-local ``sorted(set(tasks))`` is not a valid all-reduce axis: ranks
    can observe different task subsets and silently add unrelated entries.
    This helper is intentionally pure so the ordering invariant can be tested
    without initializing torch.distributed.
    """

    tasks = {str(task).strip().lower() for task in local_tasks if str(task).strip()}
    for task in configured_tasks or []:
        normalized = str(task).strip().lower()
        if normalized:
            tasks.add(normalized)
    for task_list in gathered_task_lists or []:
        for task in task_list:
            normalized = str(task).strip().lower()
            if normalized:
                tasks.add(normalized)
    return tuple(sorted(tasks))


def _complete_group_permutation(
    group_ids: torch.Tensor | None,
    group_positions: torch.Tensor | None,
    *,
    batch_size: int,
    num_generations: int,
) -> tuple[torch.Tensor | None, str]:
    """Validate and order complete prompt groups after TRL shuffling.

    TRL shuffles the generated rows before splitting them into optimization
    microbatches.  Reshaping an arbitrary microbatch to ``(-1, K)`` can then
    compare completions from unrelated prompts.  Group auxiliaries may run only
    when IDs prove that the current microbatch contains complete groups.
    """

    if not (torch.is_tensor(group_ids) and torch.is_tensor(group_positions)):
        return None, "missing_group_metadata"
    if group_ids.ndim != 1 or group_positions.ndim != 1:
        return None, "non_vector_group_metadata"
    if int(group_ids.numel()) != int(batch_size) or int(group_positions.numel()) != int(batch_size):
        return None, "group_metadata_length_mismatch"
    if num_generations <= 0 or batch_size % num_generations != 0:
        return None, "microbatch_not_divisible_by_num_generations"

    ids = group_ids.detach().to(device="cpu", dtype=torch.long)
    positions = group_positions.detach().to(device="cpu", dtype=torch.long)
    unique_ids = torch.unique(ids, sorted=True)
    ordered_rows: list[int] = []
    expected_positions = list(range(int(num_generations)))
    for group_id in unique_ids.tolist():
        rows = torch.nonzero(ids == int(group_id), as_tuple=False).reshape(-1)
        if int(rows.numel()) != int(num_generations):
            return None, "incomplete_prompt_group"
        row_positions = positions[rows]
        sorted_positions, order = torch.sort(row_positions)
        if sorted_positions.tolist() != expected_positions:
            return None, "invalid_or_duplicate_group_positions"
        ordered_rows.extend(rows[order].tolist())

    if len(ordered_rows) != int(batch_size):
        return None, "group_permutation_length_mismatch"
    return torch.tensor(ordered_rows, dtype=torch.long, device=group_ids.device), "ok"


def compute_scalarized_group_advantages(
    primal_scores: torch.Tensor,
    penalty_scores: torch.Tensor,
    num_generations: int,
    *,
    eps: float = 1e-4,
    std_correction: int = 0,
) -> torch.Tensor:
    """Return ``z_G(r-lambda*v)`` with an explicit standard-deviation convention.

    Correction zero preserves historical constrained runs. Correction one
    matches stock TRL GRPO's sample standard deviation for a matched baseline.
    """
    if primal_scores.dim() != 1 or penalty_scores.dim() != 1:
        raise ValueError("scalarized group advantages require 1-D tensors")
    if primal_scores.numel() != penalty_scores.numel():
        raise ValueError("primal/penalty tensor size mismatch")
    if num_generations <= 1 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"sample count {primal_scores.numel()} must be divisible by "
            f"num_generations={num_generations}, with num_generations > 1"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")
    if type(std_correction) is not int or std_correction not in (0, 1):
        raise ValueError("std_correction must be the integer 0 or 1")

    scalarized = (
        primal_scores.to(dtype=torch.float32)
        - penalty_scores.to(device=primal_scores.device, dtype=torch.float32)
    ).view(-1, int(num_generations))
    centered = scalarized - scalarized.mean(dim=1, keepdim=True)
    scale = scalarized.std(dim=1, keepdim=True, correction=std_correction)
    return (centered / (scale + float(eps))).reshape(-1)


def compute_separate_z_group_advantages(
    primal_scores: torch.Tensor,
    positive_component_penalties: torch.Tensor,
    num_generations: int,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return the DUAL-SEPARATE-Z advantage ``z_G(r) - z_G(p)``.

    ``positive_component_penalties[i]`` is the already aggregated component
    penalty

    ``p_i = sum_j lambda_j * w_j * max(0, c_ij - limit_j)``.

    Reward and penalty are standardized *separately within each prompt group*.
    This is intentionally distinct from :func:`compute_scalarized_group_advantages`,
    which computes ``z_G(r-p)`` after scalarization.
    """
    if primal_scores.dim() != 1 or positive_component_penalties.dim() != 1:
        raise ValueError("separate-z group advantages require 1-D tensors")
    if primal_scores.numel() != positive_component_penalties.numel():
        raise ValueError("primal/positive-penalty tensor size mismatch")
    if num_generations <= 1 or primal_scores.numel() % num_generations != 0:
        raise ValueError(
            f"sample count {primal_scores.numel()} must be divisible by "
            f"num_generations={num_generations}, with num_generations > 1"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")

    rewards = primal_scores.to(dtype=torch.float32).view(-1, int(num_generations))
    penalties = positive_component_penalties.to(
        device=primal_scores.device,
        dtype=torch.float32,
    ).view(-1, int(num_generations))
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("primal_scores contains non-finite values")
    if not bool(torch.isfinite(penalties).all().item()):
        raise ValueError("positive_component_penalties contains non-finite values")
    if bool((penalties < 0.0).any().item()):
        raise ValueError("positive_component_penalties must be non-negative")

    reward_centered = rewards - rewards.mean(dim=1, keepdim=True)
    penalty_centered = penalties - penalties.mean(dim=1, keepdim=True)
    reward_z = reward_centered / (
        rewards.std(dim=1, keepdim=True, unbiased=False) + float(eps)
    )
    penalty_z = penalty_centered / (
        penalties.std(dim=1, keepdim=True, unbiased=False) + float(eps)
    )
    return (reward_z - penalty_z).reshape(-1)


def reduce_clasp_policy_loss(
    per_token_policy_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    length_alpha: float,
    max_completion_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce exact-set policy loss with an explicit length-bias trade-off.

    For completion ``i`` of length ``L_i`` and configured fixed horizon ``C``,
    the reduction is

    ``mean_i[token_sum_i / (L_i**(1-alpha) * C**alpha)]``.

    Equivalently, it is the usual per-response token mean multiplied by
    ``(L_i / C)**alpha``.  ``alpha=0`` is the historical GRPO reduction;
    ``alpha=1`` is the fixed-horizon sequence-score reduction used by the
    Dr.GRPO endpoint.  Intermediate values make the otherwise implicit
    response-length choice an auditable experiment parameter.

    The helper reduces only the policy term.  KL regularization, when enabled,
    remains token-mean normalized and is added separately by ``compute_loss``.
    """

    if per_token_policy_loss.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("CLASP policy loss and completion mask must be rank-2 tensors")
    if per_token_policy_loss.size(0) != completion_mask.size(0):
        raise ValueError("CLASP policy loss and completion mask batch sizes must match")
    if per_token_policy_loss.size(1) == 1:
        # TRL represents a sequence-level likelihood ratio as [batch, 1].
        # Its historical reduction relies on broadcasting that value over the
        # completion mask; preserve exactly that intentional broadcast here.
        effective_policy_loss = per_token_policy_loss.expand_as(completion_mask)
    elif per_token_policy_loss.shape == completion_mask.shape:
        effective_policy_loss = per_token_policy_loss
    else:
        raise ValueError(
            "CLASP policy loss must have one value per sequence or one per "
            "completion token"
        )
    alpha = float(length_alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("CLASP policy length alpha must be finite and in [0, 1]")
    horizon = int(max_completion_length)
    if horizon <= 0:
        raise ValueError("CLASP max completion length must be a positive fixed horizon")
    if not bool(torch.isfinite(per_token_policy_loss).all().item()):
        raise ValueError("CLASP per-token policy loss contains non-finite values")
    if not bool(torch.isfinite(completion_mask).all().item()):
        raise ValueError("CLASP completion mask contains non-finite values")
    if bool(((completion_mask < 0.0) | (completion_mask > 1.0)).any().item()):
        raise ValueError("CLASP completion mask values must lie in [0, 1]")

    # TRL commonly supplies an integer attention mask. Keep counts in float32:
    # integer reductions cannot be averaged by torch, while bf16 would lose
    # exact unit resolution well below the supported 3000-token horizon.
    lengths = completion_mask.to(dtype=torch.float32).sum(dim=-1)
    if bool((lengths <= 0.0).any().item()):
        raise ValueError("CLASP exact-set completions must contain at least one token")
    if bool((lengths > float(horizon) + 1e-6).any().item()):
        raise ValueError(
            "CLASP completion length exceeds the configured fixed completion horizon"
        )

    token_sums = (effective_policy_loss * completion_mask).sum(dim=-1)
    length_weights = (lengths / float(horizon)).pow(alpha)
    sequence_losses = (token_sums / lengths) * length_weights
    return sequence_losses.mean(), length_weights, lengths


class PrimalDualGRPOTrainer(GRPOTrainer):
    """
    Constrained GRPO trainer with a clipped Lagrangian advantage.

    Let

        A_i^r = primal_i - mean_group(primal)
        g_i   = masked constraint gap
        A_i^L = A_i^r - <lambda, g_i>

    The trainer optimizes the sequence-level clipped GRPO surrogate

        L = -E[min(rho A^L, clip(rho) A^L)] + beta * KL

    so only the primal reward is group-centered, while the absolute constraint
    gap remains inside the optimized advantage.
    """

    def __init__(
        self,
        *args,
        primal_dual_reward_manager=None,
        normalize_primal_advantage: bool = False,
        primal_advantage_scale: float = 1.0,
        penalty_scale: float = 1.0,
        penalty_scale_start: float | None = None,
        penalty_scale_warmup_fraction: float = 0.0,
        positive_constraint_penalty: bool = False,
        lexicographic_advantage: bool = False,
        lexicographic_feasible_margin: float = 0.5,
        group_scalarization: bool = False,
        separate_z: bool = False,
        # ------------------- CLASP-GRPO additions -------------------
        # When clasp_enabled is False the trainer keeps the baseline constrained
        # GRPO path. When enabled, corrected BoN-aligned CLASP is the default;
        # CLASP_BON_ALIGNED=0 explicitly selects legacy strict/Pareto ablations.
        clasp_enabled: bool = False,
        clasp_pareto_anchor: bool = True,
        clasp_pareto_tau: float = 4.0,
        clasp_pareto_center: float = 0.5,
        clasp_adaptive_length=None,  # AdaptiveLengthEstimator or None
        clasp_canonical_set: bool = False,
        clasp_fixed_null_set: bool = False,
        clasp_deployment_set_size: int | None = None,
        clasp_canonical_recovery_scale: float = 1.0,
        clasp_canonical_dual_recovery_coupling: float = 0.0,
        clasp_canonical_recovery_temperature: float = 0.15,
        clasp_canonical_max_subsets: int = 1024,
        clasp_canonical_exact_scale: float = 1.0,
        clasp_canonical_grpo_anchor_scale: float = 0.0,
        clasp_canonical_quality_anchor_ratio: float = 0.0,
        clasp_canonical_dense_quality_scale: float = 0.0,
        clasp_canonical_dense_feasibility_scale: float = 0.0,
        clasp_canonical_dense_violation_scale: float = 0.0,
        clasp_canonical_dense_recovery_scale: float = 0.0,
        clasp_policy_length_alpha: float = 0.0,
        **kwargs,
    ):
        if primal_dual_reward_manager is None:
            raise ValueError("primal_dual_reward_manager must be provided for PrimalDualGRPOTrainer")

        self.primal_dual_reward_manager = primal_dual_reward_manager
        self.normalize_primal_advantage = bool(normalize_primal_advantage)
        self.primal_advantage_scale = float(primal_advantage_scale)
        self.penalty_scale = float(penalty_scale)
        self.penalty_scale_start = None if penalty_scale_start is None else float(penalty_scale_start)
        self.penalty_scale_warmup_fraction = float(penalty_scale_warmup_fraction)
        self.positive_constraint_penalty = bool(positive_constraint_penalty)
        self.lexicographic_advantage = bool(lexicographic_advantage)
        self.lexicographic_feasible_margin = float(lexicographic_feasible_margin)
        self.group_scalarization = bool(group_scalarization)
        self.group_std_correction = int(
            _strict_env_choice("PD_GROUP_STD_CORRECTION", "0", {"0", "1"})
        )
        self.separate_z = bool(separate_z)
        # CLASP attributes are always set so attribute lookups never fail,
        # but are only consulted in the gated branches below.
        self.clasp_enabled = bool(clasp_enabled)
        if self.separate_z and self.group_scalarization:
            raise ValueError(
                "DUAL-SEPARATE-Z is mutually exclusive with group_scalarization "
                "(z_G(r-p))."
            )
        if self.separate_z and self.clasp_enabled:
            raise ValueError("DUAL-SEPARATE-Z is mutually exclusive with CLASP.")
        if self.separate_z and self.lexicographic_advantage:
            raise ValueError(
                "DUAL-SEPARATE-Z is mutually exclusive with lexicographic_advantage."
            )
        if self.separate_z and (
            abs(self.primal_advantage_scale - 1.0) > 1e-12
            or abs(self.penalty_scale - 1.0) > 1e-12
            or self.penalty_scale_start is not None
            or abs(self.penalty_scale_warmup_fraction) > 1e-12
        ):
            raise ValueError(
                "DUAL-SEPARATE-Z requires unit primal/penalty scales and no penalty "
                "warmup so the optimized formula remains exactly z_G(r)-z_G(p)."
            )
        self.clasp_pareto_anchor = bool(clasp_pareto_anchor)
        self.clasp_pareto_tau = float(clasp_pareto_tau)
        self.clasp_pareto_center = float(clasp_pareto_center)
        self.clasp_adaptive_length = clasp_adaptive_length
        canonical_env_requested = _strict_env_flag("CLASP_CANONICAL_SET", False)
        canonical_requested = bool(clasp_canonical_set) or canonical_env_requested
        fixed_null_env_requested = _strict_env_flag("CLASP_FIXED_NULL_SET", False)
        fixed_null_requested = bool(clasp_fixed_null_set) or fixed_null_env_requested
        if canonical_requested and fixed_null_requested:
            raise ValueError(
                "Canonical leave-out and fixed-null set credit are mutually exclusive."
            )
        set_credit_requested = bool(canonical_requested or fixed_null_requested)
        if set_credit_requested and not self.clasp_enabled:
            raise ValueError("Exact CLASP set credit requires clasp_enabled=True")
        self.clasp_canonical_set = bool(canonical_requested)
        self.clasp_fixed_null_set = bool(fixed_null_requested)
        deployment_default = int(clasp_deployment_set_size or 0)
        if set_credit_requested:
            self.clasp_deployment_set_size = _strict_positive_int_env(
                "CLASP_DEPLOYMENT_SET_SIZE",
                deployment_default,
            )
            self.clasp_canonical_max_subsets = _strict_positive_int_env(
                "CLASP_CANONICAL_MAX_SUBSETS",
                int(clasp_canonical_max_subsets),
            )
        else:
            self.clasp_deployment_set_size = deployment_default
            self.clasp_canonical_max_subsets = int(clasp_canonical_max_subsets)
        self.clasp_canonical_recovery_scale = _finite_float_env(
            "CLASP_CANONICAL_RECOVERY_SCALE",
            float(clasp_canonical_recovery_scale),
        )
        self.clasp_canonical_dual_recovery_coupling = _finite_float_env(
            "CLASP_CANONICAL_DUAL_RECOVERY_COUPLING",
            float(clasp_canonical_dual_recovery_coupling),
        )
        self.clasp_canonical_recovery_temperature = _finite_float_env(
            "CLASP_CANONICAL_RECOVERY_TEMPERATURE",
            float(clasp_canonical_recovery_temperature),
        )
        self.clasp_canonical_exact_scale = _finite_float_env(
            "CLASP_CANONICAL_EXACT_SCALE",
            float(clasp_canonical_exact_scale),
        )
        self.clasp_canonical_grpo_anchor_scale = _finite_float_env(
            "CLASP_CANONICAL_GRPO_ANCHOR_SCALE",
            float(clasp_canonical_grpo_anchor_scale),
        )
        self.clasp_canonical_quality_anchor_ratio = _finite_float_env(
            "CLASP_CANONICAL_QUALITY_ANCHOR_RATIO",
            float(clasp_canonical_quality_anchor_ratio),
        )
        self.clasp_canonical_dense_quality_scale = _finite_float_env(
            "CLASP_CANONICAL_DENSE_QUALITY_SCALE",
            float(clasp_canonical_dense_quality_scale),
        )
        self.clasp_canonical_dense_feasibility_scale = _finite_float_env(
            "CLASP_CANONICAL_DENSE_FEASIBILITY_SCALE",
            float(clasp_canonical_dense_feasibility_scale),
        )
        self.clasp_canonical_dense_violation_scale = _finite_float_env(
            "CLASP_CANONICAL_DENSE_VIOLATION_SCALE",
            float(clasp_canonical_dense_violation_scale),
        )
        self.clasp_canonical_dense_recovery_scale = _finite_float_env(
            "CLASP_CANONICAL_DENSE_RECOVERY_SCALE",
            float(clasp_canonical_dense_recovery_scale),
        )
        self.clasp_policy_length_alpha = _finite_float_env(
            "CLASP_POLICY_LENGTH_ALPHA",
            float(clasp_policy_length_alpha),
        )
        self.clasp_mixed_adaptive_length_mode = _strict_env_choice(
            "CLASP_MIXED_ADAPTIVE_LENGTH_MODE",
            "error",
            {"error", "global_max"},
        )
        self._clasp_last_generation_budget_stats: dict[str, float] = {}
        self.clasp_feasibility_gated = _env_flag("CLASP_FEASIBILITY_GATED", False)
        self.clasp_lexicographic_gate = _env_flag(
            "CLASP_LEXICOGRAPHIC_GATE",
            False,
        )
        clasp_switch_parser = _strict_env_flag if set_credit_requested else _env_flag
        self.clasp_bon_aligned = bool(
            self.clasp_enabled and clasp_switch_parser("CLASP_BON_ALIGNED", True)
        )
        self.clasp_frontier_cdf = bool(
            self.clasp_enabled and clasp_switch_parser("CLASP_FRONTIER_CDF", False)
        )
        if set_credit_requested:
            self.clasp_bon_target_failure_rate = _strict_nonnegative_float_env(
                "CLASP_BON_TARGET_FAILURE_RATE",
                0.01,
            )
            if self.clasp_bon_target_failure_rate > 1.0:
                raise ValueError("CLASP_BON_TARGET_FAILURE_RATE must be in [0, 1]")
            self.clasp_bon_target_failure_rates = _strict_env_task_thresholds(
                "CLASP_BON_TARGET_FAILURE_RATES"
            )
            self.clasp_bon_dual_lr = _strict_nonnegative_float_env(
                "CLASP_BON_DUAL_LR",
                0.05,
            )
        else:
            self.clasp_bon_target_failure_rate = min(
                1.0,
                _nonnegative_float_env("CLASP_BON_TARGET_FAILURE_RATE", 0.01),
            )
            self.clasp_bon_target_failure_rates = _env_task_thresholds(
                "CLASP_BON_TARGET_FAILURE_RATES",
            )
            self.clasp_bon_dual_lr = _nonnegative_float_env(
                "CLASP_BON_DUAL_LR",
                0.05,
            )
        self.clasp_bon_dual_ema_beta = _finite_float_env(
            "CLASP_BON_DUAL_EMA_BETA",
            0.9,
        )
        self.clasp_bon_lambda_init = (
            _strict_nonnegative_float_env("CLASP_BON_LAMBDA_INIT", 0.0)
            if set_credit_requested
            else _nonnegative_float_env("CLASP_BON_LAMBDA_INIT", 0.0)
        )
        self.clasp_bon_lambda_max = _finite_float_env("CLASP_BON_LAMBDA_MAX", 100.0)
        self.clasp_bon_quality_temperature = _finite_float_env(
            "CLASP_BON_QUALITY_TEMPERATURE",
            0.0,
        )
        self.clasp_bon_success_credit = _strict_env_choice(
            "CLASP_BON_SUCCESS_CREDIT",
            "dense_rank",
            {"dense_rank", "exact_loo"},
        )
        self.clasp_bon_dense_quality_scale = _nonnegative_float_env(
            "CLASP_BON_DENSE_QUALITY_SCALE",
            1.0,
        )
        self.clasp_bon_dense_feasibility_scale = _nonnegative_float_env(
            "CLASP_BON_DENSE_FEASIBILITY_SCALE",
            1.0,
        )
        self.clasp_bon_dense_violation_mix = min(
            1.0,
            _nonnegative_float_env("CLASP_BON_DENSE_VIOLATION_MIX", 0.5),
        )
        self.clasp_bon_recovery_scale = _nonnegative_float_env(
            "CLASP_BON_RECOVERY_SCALE",
            1.0,
        )
        self.clasp_component_dual_recovery_scale = _nonnegative_float_env(
            "CLASP_COMPONENT_DUAL_RECOVERY_SCALE",
            1.0,
        )
        self.clasp_frontier_cdf_gap_targets = dict(_FRONTIER_CDF_DEFAULT_GAP_TARGETS)
        self.clasp_frontier_cdf_gap_targets.update(
            _env_task_thresholds("CLASP_FRONTIER_CDF_GAP_TARGETS")
        )
        self.clasp_frontier_cdf_quality_temperature_initial = _finite_float_env(
            "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_INITIAL",
            0.5,
        )
        self.clasp_frontier_cdf_quality_temperature_final = _finite_float_env(
            "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_FINAL",
            0.15,
        )
        self.clasp_frontier_cdf_quality_anneal_fraction = _finite_float_env(
            "CLASP_FRONTIER_CDF_QUALITY_ANNEAL_FRACTION",
            0.6,
        )
        self.clasp_frontier_cdf_violation_temperature = _finite_float_env(
            "CLASP_FRONTIER_CDF_VIOLATION_TEMPERATURE",
            0.15,
        )
        self.clasp_frontier_cdf_recovery_scale = _nonnegative_float_env(
            "CLASP_FRONTIER_CDF_RECOVERY_SCALE",
            1.0,
        )
        self.clasp_frontier_cdf_relative_failed_recovery = _strict_env_flag(
            "CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY",
            False,
        )
        self.clasp_frontier_cdf_relative_recovery_scale = _finite_float_env(
            "CLASP_FRONTIER_CDF_RELATIVE_RECOVERY_SCALE",
            1.0,
        )
        self.clasp_frontier_cdf_objective_score_scale = _finite_float_env(
            "CLASP_FRONTIER_CDF_OBJECTIVE_SCORE_SCALE",
            2.0,
        )
        self.clasp_frontier_cdf_gap_slope = _finite_float_env(
            "CLASP_FRONTIER_CDF_GAP_SLOPE",
            3.0,
        )
        self.clasp_frontier_cdf_feasibility_violation_tolerance = _finite_float_env(
            "CLASP_FRONTIER_CDF_FEASIBILITY_VIOLATION_TOLERANCE",
            1e-8,
        )
        # A shuffled local optimization microbatch often contains fragments of
        # several global K-sample prompt groups. Running a group loss after a
        # blind reshape is scientifically invalid, so fail closed by default.
        # The explicit false setting is retained only for legacy runs that want
        # to skip an enabled auxiliary with a warning.
        self.clasp_aux_grouping_strict = _strict_env_flag(
            "CLASP_AUX_GROUPING_STRICT",
            True,
        )
        self._clasp_aux_grouping_warning_emitted = False
        self._clasp_objective_fallback_warning_emitted = False
        self._clasp_generation_serial = 0
        self._clasp_bon_lambdas: dict[str, float] = {}
        self._clasp_bon_failure_ema: dict[str, float] = {}
        self._clasp_bon_observed_groups: dict[str, int] = {}
        # Generation and optimization batching are distinct in TRL.  A single
        # optimizer step can therefore observe several generated K-sample
        # batches.  Keep their sufficient statistics here and commit one dual
        # update only at the gradient-synchronization boundary; otherwise the
        # later microbatches in the same optimizer step are optimized with a
        # different lambda from the earlier ones.
        self._clasp_bon_pending_failure_sums: dict[str, float] = {}
        self._clasp_bon_pending_group_counts: dict[str, int] = {}
        self.clasp_pairwise_loss_weight = _nonnegative_float_env("CLASP_PAIRWISE_LOSS_WEIGHT", 0.0)
        self.clasp_pairwise_margin = _nonnegative_float_env("CLASP_PAIRWISE_MARGIN", 0.0)
        self.clasp_pairwise_advantage_delta = _nonnegative_float_env(
            "CLASP_PAIRWISE_ADVANTAGE_DELTA",
            0.0,
        )
        self.clasp_elite_imitation_loss_weight = _nonnegative_float_env(
            "CLASP_ELITE_IMITATION_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_elite_imitation_advantage_delta = _nonnegative_float_env(
            "CLASP_ELITE_IMITATION_ADVANTAGE_DELTA",
            0.0,
        )
        self.clasp_multi_elite_imitation_loss_weight = _nonnegative_float_env(
            "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_multi_elite_imitation_fraction = _nonnegative_float_env(
            "CLASP_MULTI_ELITE_IMITATION_FRACTION",
            0.25,
        )
        self.clasp_multi_elite_imitation_max_raw_gap = _nonnegative_float_env(
            "CLASP_MULTI_ELITE_IMITATION_MAX_RAW_GAP",
            0.0,
        )
        self.clasp_multi_elite_imitation_advantage_delta = _nonnegative_float_env(
            "CLASP_MULTI_ELITE_IMITATION_ADVANTAGE_DELTA",
            0.0,
        )
        self.clasp_multi_elite_imitation_temperature = _nonnegative_float_env(
            "CLASP_MULTI_ELITE_IMITATION_TEMPERATURE",
            1.0,
        )
        self.clasp_multi_elite_imitation_require_feasible = _env_flag(
            "CLASP_MULTI_ELITE_IMITATION_REQUIRE_FEASIBLE",
            True,
        )
        self.clasp_multi_elite_imitation_fallback_to_best_feasible = _env_flag(
            "CLASP_MULTI_ELITE_IMITATION_FALLBACK_TO_BEST_FEASIBLE",
            True,
        )
        self.clasp_reference_imitation_loss_weight = _nonnegative_float_env(
            "CLASP_REFERENCE_IMITATION_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_reference_imitation_max_tokens = _nonnegative_int_env(
            "CLASP_REFERENCE_IMITATION_MAX_TOKENS",
            256,
        )
        self.clasp_reference_imitation_tasks = _env_name_set(
            "CLASP_REFERENCE_IMITATION_TASKS",
        )
        self.clasp_reference_contrastive_loss_weight = _nonnegative_float_env(
            "CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_reference_contrastive_margin = _nonnegative_float_env(
            "CLASP_REFERENCE_CONTRASTIVE_MARGIN",
            0.02,
        )
        self.clasp_reference_contrastive_max_tokens = _nonnegative_int_env(
            "CLASP_REFERENCE_CONTRASTIVE_MAX_TOKENS",
            256,
        )
        self.clasp_reference_contrastive_tasks = _env_name_set(
            "CLASP_REFERENCE_CONTRASTIVE_TASKS",
        )
        self.clasp_tsp_two_opt_distill_loss_weight = _nonnegative_float_env(
            "CLASP_TSP_TWO_OPT_DISTILL_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_tsp_two_opt_distill_max_tokens = _nonnegative_int_env(
            "CLASP_TSP_TWO_OPT_DISTILL_MAX_TOKENS",
            512,
        )
        self.clasp_tsp_two_opt_distill_max_passes = _nonnegative_int_env(
            "CLASP_TSP_TWO_OPT_DISTILL_MAX_PASSES",
            1,
        )
        self.clasp_tsp_two_opt_distill_min_gain = _nonnegative_float_env(
            "CLASP_TSP_TWO_OPT_DISTILL_MIN_GAIN",
            0.0,
        )
        self.clasp_cvrp_local_search_distill_loss_weight = _nonnegative_float_env(
            "CLASP_CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_cvrp_local_search_distill_max_tokens = _nonnegative_int_env(
            "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MAX_TOKENS",
            768,
        )
        self.clasp_cvrp_local_search_distill_max_passes = _nonnegative_int_env(
            "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MAX_PASSES",
            2,
        )
        self.clasp_cvrp_local_search_distill_min_gain = _nonnegative_float_env(
            "CLASP_CVRP_LOCAL_SEARCH_DISTILL_MIN_GAIN",
            0.0,
        )
        self.clasp_jssp_active_schedule_distill_loss_weight = _nonnegative_float_env(
            "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_jssp_active_schedule_distill_max_tokens = _nonnegative_int_env(
            "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MAX_TOKENS",
            768,
        )
        self.clasp_jssp_active_schedule_distill_max_passes = _nonnegative_int_env(
            "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MAX_PASSES",
            1,
        )
        self.clasp_jssp_active_schedule_distill_min_gain = _nonnegative_float_env(
            "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_MIN_GAIN",
            0.0,
        )
        self.clasp_mis_maximalize_distill_loss_weight = _nonnegative_float_env(
            "CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_mis_maximalize_distill_max_tokens = _nonnegative_int_env(
            "CLASP_MIS_MAXIMALIZE_DISTILL_MAX_TOKENS",
            512,
        )
        self.clasp_mis_maximalize_distill_min_gain = _nonnegative_float_env(
            "CLASP_MIS_MAXIMALIZE_DISTILL_MIN_GAIN",
            0.0,
        )
        self.grpo_verified_improvement = _strict_env_flag("GRPO_VERIFIED_IMPROVEMENT", False)
        self.grpo_verified_improvement_weight = _strict_nonnegative_float_env(
            "GRPO_VERIFIED_IMPROVEMENT_WEIGHT", 0.05
        )
        self.grpo_verified_improvement_max_tokens = int(
            os.environ.get("GRPO_VERIFIED_IMPROVEMENT_MAX_TOKENS", "3000")
        )
        self.grpo_verified_improvement_max_passes = int(
            os.environ.get("GRPO_VERIFIED_IMPROVEMENT_MAX_PASSES", "2")
        )
        self.grpo_verified_teacher_selection = _strict_env_choice(
            "GRPO_VERIFIED_TEACHER_SELECTION", "per_completion",
            {"per_completion", "best_per_prompt", "feasibility_incumbent"},
        )
        self.grpo_verified_teacher_normalization = _strict_env_choice(
            "GRPO_VERIFIED_TEACHER_NORMALIZATION", "microbatch",
            {"microbatch", "optimizer_window"},
        )
        self.grpo_verified_train_reference_teacher = _strict_env_flag(
            "GRPO_VERIFIED_TRAIN_REFERENCE_TEACHER", False
        )
        if self.grpo_verified_train_reference_teacher and (
            not self.grpo_verified_improvement
            or self.grpo_verified_teacher_selection != "feasibility_incumbent"
            or self.grpo_verified_teacher_normalization != "optimizer_window"
        ):
            raise ValueError("Training-reference extension requires verified incumbent/window teachers")
        if self.grpo_verified_improvement:
            if self.clasp_enabled:
                raise ValueError("GRPO_VERIFIED_IMPROVEMENT requires plain GRPO (CLASP disabled)")
            if (self.grpo_verified_improvement_weight <= 0
                    or self.grpo_verified_improvement_max_tokens <= 0
                    or not 0 <= self.grpo_verified_improvement_max_passes <= 8):
                raise ValueError("Invalid GRPO verified improvement weight, token cap, or passes")
        self.clasp_primal_score_gap_penalty = _nonnegative_float_env(
            "CLASP_PRIMAL_SCORE_GAP_PENALTY",
            0.0,
        )
        self.clasp_hard_group_weight_scale = _nonnegative_float_env(
            "CLASP_HARD_GROUP_WEIGHT_SCALE",
            0.0,
        )
        self.clasp_hard_group_weight_threshold = _nonnegative_float_env(
            "CLASP_HARD_GROUP_WEIGHT_THRESHOLD",
            0.0,
        )
        self.clasp_hard_group_weight_max = _nonnegative_float_env(
            "CLASP_HARD_GROUP_WEIGHT_MAX",
            3.0,
        )
        self.clasp_feasibility_warmup_steps = _nonnegative_int_env(
            "CLASP_FEASIBILITY_WARMUP_STEPS",
            0,
        )
        self.clasp_feasibility_warmup_targets = _env_task_thresholds(
            "CLASP_FEASIBILITY_WARMUP_TARGETS",
        )
        self.clasp_feasibility_warmup_hold_until_eval = _env_flag(
            "CLASP_FEASIBILITY_WARMUP_HOLD_UNTIL_EVAL",
            True,
        )
        self.clasp_feasibility_warmup_margin = _nonnegative_float_env(
            "CLASP_FEASIBILITY_WARMUP_FEASIBLE_MARGIN",
            3.0,
        )
        self.clasp_feasibility_warmup_violation_scale = _nonnegative_float_env(
            "CLASP_FEASIBILITY_WARMUP_VIOLATION_SCALE",
            2.0,
        )
        self.clasp_feasibility_warmup_rank_scale = _nonnegative_float_env(
            "CLASP_FEASIBILITY_WARMUP_RANK_SCALE",
            1.0,
        )
        self.clasp_self_replay_enabled = _env_flag("CLASP_SELF_REPLAY_ENABLED", False)
        self.clasp_self_replay_imitation_loss_weight = _nonnegative_float_env(
            "CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_self_replay_contrastive_loss_weight = _nonnegative_float_env(
            "CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT",
            0.0,
        )
        self.clasp_self_replay_contrastive_margin = _nonnegative_float_env(
            "CLASP_SELF_REPLAY_CONTRASTIVE_MARGIN",
            0.02,
        )
        self.clasp_self_replay_sample_rate = min(
            1.0,
            _nonnegative_float_env("CLASP_SELF_REPLAY_SAMPLE_RATE", 0.5),
        )
        self.clasp_self_replay_max_tokens = _nonnegative_int_env(
            "CLASP_SELF_REPLAY_MAX_TOKENS",
            512,
        )
        self.clasp_self_replay_max_keys = _nonnegative_int_env(
            "CLASP_SELF_REPLAY_MAX_KEYS",
            20000,
        )
        self.clasp_self_replay_max_per_key = _nonnegative_int_env(
            "CLASP_SELF_REPLAY_MAX_PER_KEY",
            4,
        )
        self.clasp_self_replay_min_buffer_size = _nonnegative_int_env(
            "CLASP_SELF_REPLAY_MIN_BUFFER_SIZE",
            8,
        )
        self._clasp_validation_task_feasibility: dict[str, float] = {}
        self._clasp_validation_task_candidate_feasibility: dict[str, float] = {}
        self._clasp_validation_group_success_sums: dict[str, float] = {}
        self._clasp_validation_group_counts: dict[str, int] = {}
        self._clasp_validation_candidate_success_sums: dict[str, float] = {}
        self._clasp_validation_candidate_counts: dict[str, int] = {}
        self._clasp_last_generation_mode: str | None = None
        self._clasp_self_replay_buffer: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
        self._clasp_self_replay_rng = random.Random(1337)
        self._clasp_self_replay_state_loaded = False
        self._clasp_self_replay_resume_load_serial = -1
        self._clasp_self_replay_lookup_attempts = 0
        self._clasp_self_replay_lookup_hits = 0
        if self.clasp_pareto_tau <= 0.0:
            raise ValueError(f"clasp_pareto_tau must be > 0, got {self.clasp_pareto_tau}")
        if not 0.0 <= self.clasp_pareto_center <= 1.0:
            raise ValueError(
                "clasp_pareto_center must lie in [0, 1], "
                f"got {self.clasp_pareto_center}"
            )
        if self.clasp_bon_aligned and self.clasp_bon_quality_temperature < 0.0:
            raise ValueError("CLASP_BON_QUALITY_TEMPERATURE must be >= 0")
        if self.clasp_frontier_cdf and not self.clasp_bon_aligned:
            raise ValueError("CLASP_FRONTIER_CDF=1 requires CLASP_BON_ALIGNED=1")
        if (
            self.clasp_frontier_cdf_relative_failed_recovery
            and not self.clasp_frontier_cdf
        ):
            raise ValueError(
                "CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY=1 requires "
                "CLASP_FRONTIER_CDF=1"
            )
        if self.clasp_frontier_cdf_relative_recovery_scale < 0.0:
            raise ValueError(
                "CLASP_FRONTIER_CDF_RELATIVE_RECOVERY_SCALE must be >= 0"
            )
        if (
            self.clasp_frontier_cdf_relative_failed_recovery
            and self.clasp_frontier_cdf_relative_recovery_scale <= 0.0
        ):
            raise ValueError(
                "CLASP_FRONTIER_CDF_RELATIVE_RECOVERY_SCALE must be > 0 when "
                "relative failed-group recovery is enabled"
            )
        if self.clasp_frontier_cdf:
            for name, value in {
                "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_INITIAL": (
                    self.clasp_frontier_cdf_quality_temperature_initial
                ),
                "CLASP_FRONTIER_CDF_QUALITY_TEMPERATURE_FINAL": (
                    self.clasp_frontier_cdf_quality_temperature_final
                ),
                "CLASP_FRONTIER_CDF_VIOLATION_TEMPERATURE": (
                    self.clasp_frontier_cdf_violation_temperature
                ),
                "CLASP_FRONTIER_CDF_OBJECTIVE_SCORE_SCALE": (
                    self.clasp_frontier_cdf_objective_score_scale
                ),
                "CLASP_FRONTIER_CDF_GAP_SLOPE": self.clasp_frontier_cdf_gap_slope,
            }.items():
                if value <= 0.0:
                    raise ValueError(f"{name} must be > 0, got {value}")
            if not 0.0 < self.clasp_frontier_cdf_quality_anneal_fraction <= 1.0:
                raise ValueError(
                    "CLASP_FRONTIER_CDF_QUALITY_ANNEAL_FRACTION must be in (0, 1]"
                )
            if self.clasp_frontier_cdf_feasibility_violation_tolerance < 0.0:
                raise ValueError(
                    "CLASP_FRONTIER_CDF_FEASIBILITY_VIOLATION_TOLERANCE must be >= 0"
                )
            quality_mode = str(
                os.environ.get("CLASP_OBJECTIVE_QUALITY_MODE", "exp") or "exp"
            ).strip().lower()
            if quality_mode not in {"exp", "exponential", "default"}:
                raise ValueError(
                    "Frontier-CDF objective inversion requires CLASP_OBJECTIVE_QUALITY_MODE=exp"
                )
            declared_weight = _finite_float_env(
                "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT",
                0.0,
            )
            if abs(declared_weight) > 1e-12:
                raise ValueError(
                    "Frontier-CDF objective inversion requires "
                    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT=0"
                )
            if _env_flag("PD_SOFT_OBJECTIVE_FOR_INFEASIBLE", False):
                raise ValueError(
                    "Frontier-CDF requires PD_SOFT_OBJECTIVE_FOR_INFEASIBLE=0"
                )
            declared_slope = _finite_float_env(
                "CLASP_OBJECTIVE_GAP_SLOPE",
                3.0,
            )
            if abs(declared_slope - self.clasp_frontier_cdf_gap_slope) > 1e-12:
                raise ValueError(
                    "Frontier-CDF inversion slope must equal the objective reward slope: "
                    f"{self.clasp_frontier_cdf_gap_slope} != {declared_slope}"
                )
            if abs(self.clasp_frontier_cdf_objective_score_scale - 2.0) > 1e-12:
                raise ValueError(
                    "The current objective heads have a fixed leading scale of 2; set "
                    "CLASP_FRONTIER_CDF_OBJECTIVE_SCORE_SCALE=2"
                )
            forbidden_reward_modifiers = {
                "CLASP_TSP_TWO_OPT_PENALTY": _finite_float_env(
                    "CLASP_TSP_TWO_OPT_PENALTY", 0.0
                ),
                "CLASP_TSP_EDGE_OVERLAP_MIX": _finite_float_env(
                    "CLASP_TSP_EDGE_OVERLAP_MIX", 0.0
                ),
                "CLASP_TSP_TARGET_GAP_BONUS": _finite_float_env(
                    "CLASP_TSP_TARGET_GAP_BONUS", 0.0
                ),
                "CLASP_TSP_TARGET_GAP_EXCESS_PENALTY": _finite_float_env(
                    "CLASP_TSP_TARGET_GAP_EXCESS_PENALTY", 0.0
                ),
            }
            active_reward_modifiers = sorted(
                name
                for name, value in forbidden_reward_modifiers.items()
                if abs(value) > 1e-12
            )
            if active_reward_modifiers:
                raise ValueError(
                    "Frontier-CDF requires an invertible unmodified exponential objective; "
                    "disable: " + ", ".join(active_reward_modifiers)
                )
            forbidden_cost_heads = (
                "CLASP_TSP_TARGET_GAP_CONSTRAINT",
                "CLASP_TSP_TAIL_GAP_CONSTRAINT",
                "CLASP_TSP_OBJECTIVE_GAP_COST",
                "CLASP_CVRP_TARGET_GAP_CONSTRAINT",
                "CLASP_CVRP_OBJECTIVE_GAP_COST",
                "CLASP_OP_PRIZE_SURPLUS_CONSTRAINT",
                "CLASP_OP_PRIZE_CAP_CONSTRAINT",
                "CLASP_MIS_TARGET_GAP_CONSTRAINT",
                "CLASP_MIS_REPAIR_TARGET_GAP_CONSTRAINT",
                "CLASP_MIS_SIZE_EXCESS_CONSTRAINT",
                "CLASP_MIS_MAXIMALITY_CONSTRAINT",
                "CLASP_JSSP_TARGET_GAP_CONSTRAINT",
            )
            active_cost_heads = sorted(
                name for name in forbidden_cost_heads if _env_flag(name, False)
            )
            if active_cost_heads:
                raise ValueError(
                    "Frontier-CDF recovery must use evaluator-feasibility costs only; "
                    "disable: " + ", ".join(active_cost_heads)
                )
        if self.clasp_bon_lambda_max < self.clasp_bon_lambda_init:
            raise ValueError(
                "CLASP_BON_LAMBDA_MAX must be >= CLASP_BON_LAMBDA_INIT, got "
                f"{self.clasp_bon_lambda_max} < {self.clasp_bon_lambda_init}"
            )
        if self.clasp_bon_aligned and self.clasp_bon_lambda_max <= 0.0:
            raise ValueError("CLASP_BON_LAMBDA_MAX must be > 0")
        if not 0.0 <= self.clasp_bon_dual_ema_beta < 1.0:
            raise ValueError("CLASP_BON_DUAL_EMA_BETA must be in [0, 1)")
        if self.clasp_bon_aligned and self.clasp_feasibility_warmup_steps > 0:
            raise ValueError(
                "CLASP_BON_ALIGNED is incompatible with feasibility-only warmup, which would "
                "overwrite the set-level advantages. Set CLASP_FEASIBILITY_WARMUP_STEPS=0."
            )
        if self.clasp_bon_aligned and self.clasp_hard_group_weight_scale > 0.0:
            raise ValueError(
                "CLASP_BON_ALIGNED already handles failed-group recovery; set "
                "CLASP_HARD_GROUP_WEIGHT_SCALE=0 to avoid changing its constrained objective."
            )
        for name, value in {
            "primal_advantage_scale": self.primal_advantage_scale,
            "penalty_scale": self.penalty_scale,
            "lexicographic_feasible_margin": self.lexicographic_feasible_margin,
            "penalty_scale_warmup_fraction": self.penalty_scale_warmup_fraction,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.penalty_scale_start is not None and self.penalty_scale_start < 0.0:
            raise ValueError(f"penalty_scale_start must be >= 0, got {self.penalty_scale_start}")
        if self.penalty_scale_warmup_fraction > 1.0:
            raise ValueError(
                "penalty_scale_warmup_fraction must be in [0, 1], "
                f"got {self.penalty_scale_warmup_fraction}"
            )
        super().__init__(*args, **kwargs)
        self._validate_canonical_set_configuration()
        self._validate_group_auxiliary_schedule()
        self._validate_verified_teacher_window()
        self._clasp_self_replay_rng.seed(int(getattr(getattr(self, "args", None), "seed", 0) or 0) + 1337)

        importance_sampling_level = getattr(
            self,
            "importance_sampling_level",
            getattr(self.args, "importance_sampling_level", None),
        )
        allow_token_importance_ablation = (
            os.environ.get(
                "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if importance_sampling_level not in {"sequence", "token"}:
            raise RuntimeError(
                "PrimalDualGRPOTrainer importance_sampling_level must be "
                "'sequence' or 'token'."
            )
        if (
            importance_sampling_level == "token"
            and not allow_token_importance_ablation
        ):
            raise RuntimeError(
                "Token-level importance weights are permitted only for the "
                "explicit matched surrogate ablation; set "
                "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1."
            )

        scale_rewards = getattr(self, "scale_rewards", getattr(self.args, "scale_rewards", None))
        if scale_rewards not in {None, False, "none"}:
            raise RuntimeError(
                "PrimalDualGRPOTrainer requires reward scaling to be disabled. "
                "The constrained objective assumes reward advantages are group-centered only, "
                "without additional variance normalization."
            )

    def _validate_canonical_set_configuration(self) -> None:
        """Reject configurations that would silently change exact set credit."""

        canonical = bool(getattr(self, "clasp_canonical_set", False))
        fixed_null = bool(getattr(self, "clasp_fixed_null_set", False))
        length_alpha = float(getattr(self, "clasp_policy_length_alpha", 0.0))
        if not math.isfinite(length_alpha) or not 0.0 <= length_alpha <= 1.0:
            raise ValueError("CLASP_POLICY_LENGTH_ALPHA must be finite and in [0, 1]")
        if not (canonical or fixed_null):
            if abs(length_alpha) > 1e-12:
                raise ValueError(
                    "CLASP_POLICY_LENGTH_ALPHA is defined only for exact canonical "
                    "or fixed-null set credit."
                )
            return
        if canonical and fixed_null:
            raise ValueError(
                "Canonical leave-out and fixed-null set credit are mutually exclusive."
            )
        B = int(getattr(self, "num_generations", 0) or 0)
        K = int(getattr(self, "clasp_deployment_set_size", 0) or 0)
        if B <= 1 or K <= 0 or K >= B:
            raise ValueError(
                "Exact CLASP set credit requires 1 <= deployment K < rollout B, "
                f"got K={K}, B={B}. Prefer B=K+1."
            )
        subset_count = math.comb(B, K)
        max_subsets = int(getattr(self, "clasp_canonical_max_subsets", 0) or 0)
        if max_subsets <= 0 or subset_count > max_subsets:
            raise ValueError(
                "Exact CLASP set-credit enumeration is outside its configured cap: "
                f"comb({B}, {K})={subset_count}, cap={max_subsets}."
            )
        if not math.isfinite(self.clasp_canonical_recovery_scale) or (
            self.clasp_canonical_recovery_scale < 0.0
        ):
            raise ValueError("CLASP_CANONICAL_RECOVERY_SCALE must be finite and >= 0")
        if not math.isfinite(self.clasp_canonical_dual_recovery_coupling) or (
            self.clasp_canonical_dual_recovery_coupling < 0.0
        ):
            raise ValueError(
                "CLASP_CANONICAL_DUAL_RECOVERY_COUPLING must be finite and >= 0"
            )
        if not math.isfinite(self.clasp_canonical_recovery_temperature) or (
            self.clasp_canonical_recovery_temperature <= 0.0
        ):
            raise ValueError("CLASP_CANONICAL_RECOVERY_TEMPERATURE must be finite and > 0")
        dense_scales = {
            "CLASP_CANONICAL_EXACT_SCALE": float(
                getattr(self, "clasp_canonical_exact_scale", 1.0)
            ),
            "CLASP_CANONICAL_GRPO_ANCHOR_SCALE": float(
                getattr(self, "clasp_canonical_grpo_anchor_scale", 0.0)
            ),
            "CLASP_CANONICAL_QUALITY_ANCHOR_RATIO": float(
                getattr(self, "clasp_canonical_quality_anchor_ratio", 0.0)
            ),
            "CLASP_CANONICAL_DENSE_QUALITY_SCALE": float(
                getattr(self, "clasp_canonical_dense_quality_scale", 0.0)
            ),
            "CLASP_CANONICAL_DENSE_FEASIBILITY_SCALE": float(
                getattr(self, "clasp_canonical_dense_feasibility_scale", 0.0)
            ),
            "CLASP_CANONICAL_DENSE_VIOLATION_SCALE": float(
                getattr(self, "clasp_canonical_dense_violation_scale", 0.0)
            ),
            "CLASP_CANONICAL_DENSE_RECOVERY_SCALE": float(
                getattr(self, "clasp_canonical_dense_recovery_scale", 0.0)
            ),
        }
        for name, value in dense_scales.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if dense_scales["CLASP_CANONICAL_EXACT_SCALE"] <= 0.0:
            raise ValueError("CLASP_CANONICAL_EXACT_SCALE must be > 0")
        if fixed_null and dense_scales["CLASP_CANONICAL_GRPO_ANCHOR_SCALE"] > 0.0:
            raise ValueError("GRPO-anchor credit is not defined for fixed-null credit")
        quality_anchor_ratio = dense_scales["CLASP_CANONICAL_QUALITY_ANCHOR_RATIO"]
        if fixed_null and quality_anchor_ratio > 0.0:
            raise ValueError(
                "Norm-matched canonical quality credit is not defined for fixed-null credit"
            )
        if quality_anchor_ratio > 0.0 and (
            not canonical
            or dense_scales["CLASP_CANONICAL_GRPO_ANCHOR_SCALE"] <= 0.0
        ):
            raise ValueError(
                "CLASP_CANONICAL_QUALITY_ANCHOR_RATIO requires canonical credit "
                "and a positive GRPO anchor"
            )
        auxiliary_enabled = any(
            value > 0.0
            for value in (
                dense_scales["CLASP_CANONICAL_DENSE_QUALITY_SCALE"],
                dense_scales["CLASP_CANONICAL_DENSE_FEASIBILITY_SCALE"],
                dense_scales["CLASP_CANONICAL_DENSE_VIOLATION_SCALE"],
                dense_scales["CLASP_CANONICAL_DENSE_RECOVERY_SCALE"],
            )
        )
        if fixed_null and auxiliary_enabled:
            raise ValueError("Dense canonical auxiliary credit is not defined for fixed-null credit")
        if not bool(getattr(self, "clasp_bon_aligned", False)):
            raise ValueError("Exact CLASP set credit requires CLASP_BON_ALIGNED=1")
        if bool(getattr(self, "clasp_frontier_cdf", False)):
            raise ValueError("Exact CLASP set credit is incompatible with CLASP_FRONTIER_CDF=1")
        if str(getattr(self, "loss_type", "")).strip().lower() != "grpo":
            raise ValueError(
                "Exact CLASP set credit requires loss_type='grpo' so K*A is "
                "paired with an outer mean over sequences."
            )
        importance_sampling_level = getattr(
            self,
            "importance_sampling_level",
            getattr(getattr(self, "args", None), "importance_sampling_level", None),
        )
        if importance_sampling_level not in {"sequence", "token"}:
            raise ValueError(
                "Exact CLASP set-credit importance_sampling_level must be sequence or token."
            )
        if length_alpha > 0.0 and importance_sampling_level != "sequence":
            raise ValueError(
                "CLASP_POLICY_LENGTH_ALPHA>0 requires sequence-level importance "
                "sampling; token importance is retained only as an alpha=0 ablation."
            )
        max_completion_length_value = getattr(self, "max_completion_length", None)
        if max_completion_length_value is None:
            max_completion_length_value = getattr(
                getattr(self, "args", None), "max_completion_length", 0
            )
        max_completion_length = int(max_completion_length_value or 0)
        if max_completion_length <= 0:
            raise ValueError(
                "Exact CLASP set credit requires a positive fixed max_completion_length."
            )
        if self.clasp_adaptive_length is not None:
            raise ValueError(
                "Exact CLASP set-credit runs cannot bundle adaptive completion length."
            )
        if fixed_null:
            receipt_identity = getattr(
                self.primal_dual_reward_manager,
                "fixed_null_receipt_identity",
                None,
            )
            if not isinstance(receipt_identity, dict):
                raise ValueError(
                    "Fixed-null SET-GRPO requires a validated fixed-null construction "
                    "receipt on the reward manager."
                )
        wrappers = getattr(self.primal_dual_reward_manager, "wrappers", None)
        if isinstance(wrappers, dict) and wrappers:
            active_tasks = {str(task).strip().lower() for task in wrappers}
            unknown_targets = sorted(
                set(self.clasp_bon_target_failure_rates) - active_tasks
            )
            if unknown_targets:
                raise ValueError(
                    "CLASP_BON_TARGET_FAILURE_RATES references inactive tasks: "
                    + ", ".join(unknown_targets)
                )

        incompatible_weights = {
            "CLASP_PAIRWISE_LOSS_WEIGHT": self.clasp_pairwise_loss_weight,
            "CLASP_ELITE_IMITATION_LOSS_WEIGHT": self.clasp_elite_imitation_loss_weight,
            "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT": (
                self.clasp_multi_elite_imitation_loss_weight
            ),
            "CLASP_REFERENCE_IMITATION_LOSS_WEIGHT": (
                self.clasp_reference_imitation_loss_weight
            ),
            "CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT": (
                self.clasp_reference_contrastive_loss_weight
            ),
            "CLASP_TSP_TWO_OPT_DISTILL_LOSS_WEIGHT": (
                self.clasp_tsp_two_opt_distill_loss_weight
            ),
            "CLASP_CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT": (
                self.clasp_cvrp_local_search_distill_loss_weight
            ),
            "CLASP_JSSP_ACTIVE_SCHEDULE_DISTILL_LOSS_WEIGHT": (
                self.clasp_jssp_active_schedule_distill_loss_weight
            ),
            "CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT": (
                self.clasp_mis_maximalize_distill_loss_weight
            ),
            "CLASP_SELF_REPLAY_IMITATION_LOSS_WEIGHT": (
                self.clasp_self_replay_imitation_loss_weight
            ),
            "CLASP_SELF_REPLAY_CONTRASTIVE_LOSS_WEIGHT": (
                self.clasp_self_replay_contrastive_loss_weight
            ),
            "CLASP_HARD_GROUP_WEIGHT_SCALE": self.clasp_hard_group_weight_scale,
        }
        active_weights = sorted(
            name for name, value in incompatible_weights.items() if float(value) > 0.0
        )
        if active_weights:
            raise ValueError(
                "Exact CLASP set-credit runs cannot bundle auxiliary losses or "
                "advantage rescaling; disable: " + ", ".join(active_weights)
            )
        if int(self.clasp_feasibility_warmup_steps) != 0:
            raise ValueError(
                "Exact CLASP set credit forbids feasibility-warmup advantage replacement."
            )

    def _get_local_primal_dual_components(self) -> dict[str, list[float]]:
        if not hasattr(self.primal_dual_reward_manager, "get_last_components"):
            raise RuntimeError(
                "primal_dual_reward_manager must expose get_last_components() for "
                "PrimalDualGRPOTrainer."
            )

        components = self.primal_dual_reward_manager.get_last_components()
        required_keys = (
            "primal_scores",
            "constraint_scores",
            "combined_scores",
        )
        for key in required_keys:
            values = components.get(key)
            if not isinstance(values, list):
                raise RuntimeError(
                    f"Missing constrained-GRPO component '{key}' from reward manager. "
                    "Cannot recompute the constrained Lagrangian advantage."
                )
        weighted_gap_scores = components.get("weighted_gap_scores", components.get("weighted_cost_scores"))
        if not isinstance(weighted_gap_scores, list):
            raise RuntimeError(
                "Missing constrained-GRPO component 'weighted_gap_scores' from reward manager. "
                "Cannot recompute the Lagrangian advantage."
            )
        return components

    def _standardize_group_values(self, values: torch.Tensor) -> torch.Tensor:
        grouped_values = values.view(-1, self.num_generations)
        mean_values = grouped_values.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = values - mean_values
        if self.normalize_primal_advantage:
            std_values = grouped_values.std(dim=1, unbiased=False).repeat_interleave(self.num_generations, dim=0)
            advantages = advantages / (std_values + 1e-4)
        return advantages

    def _current_penalty_scale(self) -> float:
        if self.penalty_scale_start is None or self.penalty_scale_warmup_fraction <= 0.0:
            return self.penalty_scale

        state = getattr(self, "state", None)
        global_step = float(getattr(state, "global_step", 0) or 0)
        max_steps = float(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0.0:
            return self.penalty_scale

        warmup_steps = max(1.0, max_steps * self.penalty_scale_warmup_fraction)
        progress = max(0.0, min(1.0, global_step / warmup_steps))
        return float(
            self.penalty_scale_start
            + progress * (self.penalty_scale - self.penalty_scale_start)
        )

    def _compute_lexicographic_advantages(
        self,
        primal_scores: torch.Tensor,
        violation_scores: torch.Tensor,
        feasibility_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        grouped_primal = primal_scores.view(-1, self.num_generations)
        grouped_violation = violation_scores.view(-1, self.num_generations)
        grouped_feasible = feasibility_flags.view(-1, self.num_generations) > 0.5
        grouped_advantages = torch.zeros_like(grouped_primal)

        groups_with_feasible = 0
        mixed_groups = 0
        all_infeasible_groups = 0
        for group_idx in range(grouped_primal.size(0)):
            feasible_mask = grouped_feasible[group_idx]
            has_feasible = bool(feasible_mask.any().item())
            all_feasible = bool(feasible_mask.all().item())

            if has_feasible:
                groups_with_feasible += 1
                feasible_primal = grouped_primal[group_idx][feasible_mask]
                if feasible_primal.numel() > 1:
                    feasible_centered = feasible_primal - feasible_primal.mean()
                    if self.normalize_primal_advantage:
                        feasible_centered = feasible_centered / (feasible_primal.std(unbiased=False) + 1e-4)
                else:
                    feasible_centered = torch.zeros_like(feasible_primal)

                if all_feasible:
                    grouped_advantages[group_idx][feasible_mask] = (
                        self.primal_advantage_scale * feasible_centered
                    )
                    continue

                mixed_groups += 1
                margin = self.lexicographic_feasible_margin
                objective_term = self.primal_advantage_scale * feasible_centered
                objective_term = torch.clamp(objective_term, min=-0.5 * margin, max=0.5 * margin)
                grouped_advantages[group_idx][feasible_mask] = margin + objective_term

                infeasible_mask = ~feasible_mask
                infeasible_violation = grouped_violation[group_idx][infeasible_mask].clamp(min=0.0)
                penalty_scale = self._current_penalty_scale()
                grouped_advantages[group_idx][infeasible_mask] = -margin - (
                    penalty_scale * infeasible_violation
                )
                grouped_advantages[group_idx] = (
                    grouped_advantages[group_idx] - grouped_advantages[group_idx].mean()
                )
                continue

            all_infeasible_groups += 1
            violations = grouped_violation[group_idx].clamp(min=0.0)
            violation_advantages = violations.mean() - violations
            if self.normalize_primal_advantage and violations.numel() > 1:
                violation_advantages = violation_advantages / (violations.std(unbiased=False) + 1e-4)
            grouped_advantages[group_idx] = self._current_penalty_scale() * violation_advantages

        num_groups = max(int(grouped_primal.size(0)), 1)
        stats = {
            "lexicographic_group_feasible_rate": float(groups_with_feasible / num_groups),
            "lexicographic_mixed_group_rate": float(mixed_groups / num_groups),
            "lexicographic_all_infeasible_group_rate": float(all_infeasible_groups / num_groups),
        }
        return grouped_advantages.reshape(-1), stats

    def _replace_logged_advantages(self, advantages_global: torch.Tensor):
        logs = getattr(self, "_logs", None)
        if not isinstance(logs, dict):
            return

        recorded = logs.get("advantages")
        if not isinstance(recorded, (list, deque)):
            return

        new_values = advantages_global.detach().cpu().tolist()
        replace_count = len(new_values)
        if replace_count <= 0:
            return
        if len(recorded) < replace_count:
            return

        if isinstance(recorded, deque):
            for _ in range(replace_count):
                recorded.pop()
        else:
            del recorded[-replace_count:]
        recorded.extend(new_values)

    def _append_metric(self, mode: str, key: str, value: float):
        metrics = getattr(self, "_metrics", None)
        if not isinstance(metrics, dict):
            return

        bucket = metrics.get(mode)
        if bucket is None:
            return
        bucket[key].append(float(value))

    def _gather_task_types(self, task_types: list[str], expected_global_len: int) -> list[str] | None:
        local_tasks = [str(task) for task in task_types]
        gathered: list[list[str]] | None = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
                dist.all_gather_object(gathered, local_tasks)
        except Exception:
            gathered = None

        if gathered is None:
            all_tasks = local_tasks
        else:
            all_tasks = []
            for part in gathered:
                if isinstance(part, list):
                    all_tasks.extend(str(task) for task in part)

        if len(all_tasks) != int(expected_global_len):
            return None
        return all_tasks

    def _distributed_task_order(self, task_types: list[str]) -> tuple[str, ...]:
        """Build the same task reduction axis on every rank."""

        configured_tasks: list[str] = []
        wrappers = getattr(self.primal_dual_reward_manager, "wrappers", None)
        if isinstance(wrappers, dict):
            configured_tasks = [str(task) for task in wrappers]

        gathered: list[list[str]] | None = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
                dist.all_gather_object(gathered, [str(task) for task in task_types])
        except Exception as exc:
            # Falling back to configured wrapper names is safe because every
            # rank constructs the same manager. Without either source, a
            # distributed reduction would be ambiguous and must not continue.
            if not configured_tasks:
                raise RuntimeError(
                    "Unable to establish a global CLASP task order for distributed reduction"
                ) from exc

        order = _canonical_task_order(
            [str(task) for task in task_types],
            gathered_task_lists=gathered,
            configured_tasks=configured_tasks,
        )
        if not order:
            raise RuntimeError("CLASP task reduction received no task names")
        return order

    def _clasp_task_warmup_active(self, task_name: str | None) -> bool:
        if not (self.clasp_enabled and self.clasp_feasibility_warmup_steps > 0):
            return False
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        if step < int(self.clasp_feasibility_warmup_steps):
            return True
        if not self.clasp_feasibility_warmup_hold_until_eval:
            return False
        task = str(task_name or "").lower()
        if not task:
            return True
        target = float(self.clasp_feasibility_warmup_targets.get(task, 1.0))
        observed = self._clasp_validation_task_feasibility.get(task)
        if observed is None:
            return True
        return float(observed) < target

    def _clasp_warmup_flags_for_tasks(
        self,
        task_types: list[str] | None,
        total_len: int,
        device,
    ) -> torch.Tensor | None:
        if not (self.clasp_enabled and self.clasp_feasibility_warmup_steps > 0):
            return None
        if not task_types or len(task_types) != int(total_len):
            active = self._clasp_task_warmup_active(None)
            return torch.full((int(total_len),), bool(active), dtype=torch.bool, device=device)
        flags = [
            bool(self._clasp_task_warmup_active(str(task_name)))
            for task_name in task_types
        ]
        return torch.tensor(flags, dtype=torch.bool, device=device)

    def _update_clasp_validation_feasibility(self, mode: str, components: dict) -> None:
        previous_mode = getattr(self, "_clasp_last_generation_mode", None)
        self._clasp_last_generation_mode = str(mode)
        if mode != "eval" or not self.clasp_enabled:
            return
        task_types = components.get("task_types") if isinstance(components, dict) else None
        flags = components.get("feasibility_flags") if isinstance(components, dict) else None
        if not (isinstance(task_types, list) and isinstance(flags, list) and len(task_types) == len(flags)):
            return

        # A rollout prompt group can span ranks (the shipped B=8, four-GPU
        # schedule places two candidates on each rank). Candidate averaging is
        # therefore not a valid deployment gate. Gather the ordered candidates
        # first, then reduce exact contiguous prompt groups to pass@K.
        device = getattr(getattr(self, "accelerator", None), "device", None)
        local_flags = torch.tensor(flags, dtype=torch.float32, device=device)
        gathered_flags = _gather_tensor(self, local_flags)
        gathered_tasks = self._gather_task_types(
            [str(task) for task in task_types],
            int(gathered_flags.numel()),
        )
        if gathered_tasks is None:
            raise RuntimeError(
                "CLASP validation pass@K gating requires globally gathered task labels."
            )
        if int(gathered_flags.numel()) % int(self.num_generations) != 0:
            raise RuntimeError(
                "CLASP validation candidate count is not divisible by rollout B: "
                f"count={int(gathered_flags.numel())}, B={int(self.num_generations)}"
            )

        if previous_mode != "eval":
            self._clasp_validation_group_success_sums.clear()
            self._clasp_validation_group_counts.clear()
            self._clasp_validation_candidate_success_sums.clear()
            self._clasp_validation_candidate_counts.clear()

        group_tasks = self._bon_group_task_names(
            gathered_tasks,
            int(gathered_flags.numel()),
        )
        grouped_flags = gathered_flags.view(-1, int(self.num_generations)) > 0.5
        if (
            bool(getattr(self, "clasp_canonical_set", False))
            or bool(getattr(self, "clasp_fixed_null_set", False))
        ):
            from clasp.set_grpo import deployment_failure_u_statistic

            deployment_k = int(self.clasp_deployment_set_size)
            group_success = 1.0 - deployment_failure_u_statistic(
                grouped_flags,
                set_size=deployment_k,
            )
        else:
            deployment_k = int(self.num_generations)
            group_success = grouped_flags.any(dim=1).to(dtype=torch.float32)

        batch_group_sums: dict[str, float] = {}
        batch_group_counts: dict[str, int] = {}
        for task, success in zip(group_tasks, group_success.tolist()):
            normalized_task = str(task).lower()
            batch_group_sums[normalized_task] = float(
                batch_group_sums.get(normalized_task, 0.0) + float(success)
            )
            batch_group_counts[normalized_task] = int(
                batch_group_counts.get(normalized_task, 0) + 1
            )

        batch_candidate_sums: dict[str, float] = {}
        batch_candidate_counts: dict[str, int] = {}
        for task, flag in zip(gathered_tasks, gathered_flags.tolist()):
            normalized_task = str(task).lower()
            batch_candidate_sums[normalized_task] = float(
                batch_candidate_sums.get(normalized_task, 0.0) + float(flag)
            )
            batch_candidate_counts[normalized_task] = int(
                batch_candidate_counts.get(normalized_task, 0) + 1
            )

        for task in sorted(batch_group_counts):
            group_count = int(batch_group_counts[task])
            group_sum = float(batch_group_sums[task])
            candidate_count = int(batch_candidate_counts.get(task, 0))
            candidate_sum = float(batch_candidate_sums.get(task, 0.0))
            self._clasp_validation_group_success_sums[task] = float(
                self._clasp_validation_group_success_sums.get(task, 0.0) + group_sum
            )
            self._clasp_validation_group_counts[task] = int(
                self._clasp_validation_group_counts.get(task, 0) + group_count
            )
            self._clasp_validation_candidate_success_sums[task] = float(
                self._clasp_validation_candidate_success_sums.get(task, 0.0)
                + candidate_sum
            )
            self._clasp_validation_candidate_counts[task] = int(
                self._clasp_validation_candidate_counts.get(task, 0)
                + candidate_count
            )

            cumulative_group_count = self._clasp_validation_group_counts[task]
            cumulative_candidate_count = self._clasp_validation_candidate_counts[task]
            pass_at_k = (
                self._clasp_validation_group_success_sums[task]
                / max(cumulative_group_count, 1)
            )
            candidate_rate = (
                self._clasp_validation_candidate_success_sums[task]
                / max(cumulative_candidate_count, 1)
            )
            # This state drives warmup release. It deliberately stores pass@K,
            # while candidate feasibility remains available only as a diagnostic.
            self._clasp_validation_task_feasibility[task] = float(pass_at_k)
            self._clasp_validation_task_candidate_feasibility[task] = float(
                candidate_rate
            )

            batch_pass_at_k = group_sum / max(group_count, 1)
            self._append_metric(
                mode,
                f"constrained_grpo/task/{task}/pass_at_k_feasibility_rate",
                float(batch_pass_at_k),
            )
            self._append_metric(
                mode,
                f"constrained_grpo/task/{task}/pass_at_{deployment_k}_feasibility_rate",
                float(batch_pass_at_k),
            )
            self._append_metric(
                mode,
                f"constrained_grpo/task/{task}/validation_candidate_feasibility_rate",
                candidate_sum / max(candidate_count, 1),
            )

    def _bon_group_task_names(
        self,
        gathered_task_types: list[str] | None,
        total_len: int,
    ) -> list[str]:
        num_groups = int(total_len) // int(self.num_generations)
        if gathered_task_types is None:
            wrappers = getattr(self.primal_dual_reward_manager, "wrappers", None)
            configured = (
                sorted(str(task).strip().lower() for task in wrappers if str(task).strip())
                if isinstance(wrappers, dict)
                else []
            )
            if len(configured) == 1:
                return [configured[0]] * num_groups
            if len(configured) > 1:
                raise RuntimeError(
                    "CLASP BoN cannot assign per-task controller pressure because gathered "
                    "task labels are missing for a multi-task reward manager."
                )
            return ["__all__"] * num_groups
        if len(gathered_task_types) != int(total_len):
            raise RuntimeError(
                "CLASP BoN task labels do not match the gathered completion count: "
                f"tasks={len(gathered_task_types)}, completions={total_len}"
            )

        group_tasks: list[str] = []
        for group_idx in range(num_groups):
            start = group_idx * int(self.num_generations)
            end = start + int(self.num_generations)
            names = {
                str(task).strip().lower()
                for task in gathered_task_types[start:end]
                if str(task).strip()
            }
            if len(names) != 1:
                raise RuntimeError(
                    "CLASP BoN requires every K-sample prompt group to have one task label; "
                    f"group={group_idx}, labels={sorted(names)}"
                )
            group_tasks.append(next(iter(names)))
        return group_tasks

    def _load_bon_lambdas(self) -> None:
        state_getter = getattr(
            self.primal_dual_reward_manager,
            "get_clasp_bon_controller_state",
            None,
        )
        if callable(state_getter):
            state = state_getter()
            if isinstance(state, dict):
                loaded_ema = state.get("failure_ema", {})
                loaded_counts = state.get("observed_groups", {})
                if isinstance(loaded_ema, dict):
                    self._clasp_bon_failure_ema.update(
                        {str(task).lower(): float(value) for task, value in loaded_ema.items()}
                    )
                if isinstance(loaded_counts, dict):
                    self._clasp_bon_observed_groups.update(
                        {str(task).lower(): int(value) for task, value in loaded_counts.items()}
                    )
        getter = getattr(self.primal_dual_reward_manager, "get_clasp_bon_lambdas", None)
        if callable(getter):
            loaded = getter()
            if isinstance(loaded, dict):
                self._clasp_bon_lambdas.update(
                    {
                        str(task).lower(): float(value)
                        for task, value in loaded.items()
                    }
                )

    def _store_bon_lambdas(self) -> None:
        state_setter = getattr(
            self.primal_dual_reward_manager,
            "set_clasp_bon_controller_state",
            None,
        )
        if callable(state_setter):
            state_setter(
                {
                    "lambdas": dict(self._clasp_bon_lambdas),
                    "failure_ema": dict(self._clasp_bon_failure_ema),
                    "observed_groups": dict(self._clasp_bon_observed_groups),
                }
            )
            return
        setter = getattr(self.primal_dual_reward_manager, "set_clasp_bon_lambdas", None)
        if callable(setter):
            setter(dict(self._clasp_bon_lambdas))

    def _frontier_cdf_quality_temperature(self) -> float:
        """Anneal from broad SFT-frontier coverage to target-gap resolution."""

        initial = float(self.clasp_frontier_cdf_quality_temperature_initial)
        final = float(self.clasp_frontier_cdf_quality_temperature_final)
        state = getattr(self, "state", None)
        global_step = max(0, int(getattr(state, "global_step", 0) or 0))
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = int(getattr(getattr(self, "args", None), "max_steps", 0) or 0)
        if max_steps <= 0:
            return initial
        anneal_steps = max(
            1,
            int(round(max_steps * float(self.clasp_frontier_cdf_quality_anneal_fraction))),
        )
        progress = min(1.0, float(global_step) / float(anneal_steps))
        return float(initial * ((final / initial) ** progress))

    def _compute_exact_set_advantages(
        self,
        objective_scores: torch.Tensor,
        violation_scores: torch.Tensor,
        feasibility_flags: torch.Tensor,
        gathered_task_types: list[str] | None,
        *,
        dense_violation_scores: torch.Tensor | None = None,
        null_objective_scores: torch.Tensor | None = None,
        null_violation_scores: torch.Tensor | None = None,
        null_feasibility_flags: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute canonical or receipted fixed-null credit for one exact utility."""

        fixed_null_inputs = (
            null_objective_scores,
            null_violation_scores,
            null_feasibility_flags,
        )
        fixed_null = any(value is not None for value in fixed_null_inputs)
        if fixed_null and not all(value is not None for value in fixed_null_inputs):
            raise RuntimeError(
                "Fixed-null SET-GRPO requires objective, violation, and feasibility "
                "scores from one construction receipt."
            )
        credit_name = "fixed_null" if fixed_null else "canonical"

        B = int(self.num_generations)
        K = int(self.clasp_deployment_set_size)
        total_len = int(objective_scores.numel())
        group_tasks = self._bon_group_task_names(gathered_task_types, total_len)
        task_order = sorted(set(group_tasks))
        task_to_index = {task: idx for idx, task in enumerate(task_order)}
        group_task_indices = torch.tensor(
            [task_to_index[task] for task in group_tasks],
            dtype=torch.long,
            device=objective_scores.device,
        )

        self._load_bon_lambdas()
        for task in task_order:
            self._clasp_bon_lambdas.setdefault(task, float(self.clasp_bon_lambda_init))
        current_lambdas = torch.tensor(
            [self._clasp_bon_lambdas[task] for task in task_order],
            dtype=torch.float32,
            device=objective_scores.device,
        )
        target_rates = torch.tensor(
            [
                self.clasp_bon_target_failure_rates.get(
                    task,
                    self.clasp_bon_target_failure_rate,
                )
                for task in task_order
            ],
            dtype=torch.float32,
            device=objective_scores.device,
        )
        group_lambdas = current_lambdas.detach()[group_task_indices]
        group_targets = target_rates.detach()[group_task_indices]
        # The exact binary failure term has zero leave-out credit in an
        # all-infeasible group: every size-K subset fails.  Keep the canonical
        # utility unchanged at coupling=0, but expose a named recovery-pressure
        # arm whose frozen per-group coefficient is rho + gamma * lambda.  This
        # remains exact credit for the resulting symmetric set utility because
        # lambda is committed only at optimizer boundaries; it is deliberately
        # reported as auxiliary shaping rather than as the binary-risk dual.
        dual_recovery_coupling = float(
            getattr(self, "clasp_canonical_dual_recovery_coupling", 0.0)
        )
        effective_recovery_scales = (
            float(self.clasp_canonical_recovery_scale)
            + dual_recovery_coupling * group_lambdas
        )
        if fixed_null:
            from clasp.set_grpo import compute_fixed_null_clasp_set_advantages

            result = compute_fixed_null_clasp_set_advantages(
                objective_scores,
                violation_scores,
                feasibility_flags,
                null_objective_scores,
                null_violation_scores,
                null_feasibility_flags,
                rollout_batch_size=B,
                deployment_set_size=K,
                group_lambdas=group_lambdas,
                recovery_scale=effective_recovery_scales,
                recovery_temperature=self.clasp_canonical_recovery_temperature,
                target_failure_rates=group_targets,
            )
        else:
            from clasp.set_grpo import compute_canonical_clasp_set_advantages

            result = compute_canonical_clasp_set_advantages(
                objective_scores,
                violation_scores,
                feasibility_flags,
                rollout_batch_size=B,
                deployment_set_size=K,
                group_lambdas=group_lambdas,
                recovery_scale=effective_recovery_scales,
                recovery_temperature=self.clasp_canonical_recovery_temperature,
                target_failure_rates=group_targets,
            )
        exact_scale = float(getattr(self, "clasp_canonical_exact_scale", 1.0))
        grpo_anchor_scale = float(
            getattr(self, "clasp_canonical_grpo_anchor_scale", 0.0)
        )
        quality_anchor_ratio = float(
            getattr(self, "clasp_canonical_quality_anchor_ratio", 0.0)
        )
        dense_quality_scale = float(
            getattr(self, "clasp_canonical_dense_quality_scale", 0.0)
        )
        dense_feasibility_scale = float(
            getattr(self, "clasp_canonical_dense_feasibility_scale", 0.0)
        )
        dense_violation_scale = float(
            getattr(self, "clasp_canonical_dense_violation_scale", 0.0)
        )
        dense_recovery_scale = float(
            getattr(self, "clasp_canonical_dense_recovery_scale", 0.0)
        )
        # Preserve classic GRPO's strongest property on these tasks: every
        # feasible candidate contributes a normalized, dense objective signal.
        # Exact best-of-K credit is then a residual instead of a replacement.
        # Because the reward head hard-gates objective_scores to zero on
        # infeasible candidates, this anchor is exactly the launcher's classic
        # feasible-only group-z advantage. In all-infeasible groups it is zero,
        # leaving canonical violation recovery to supply the missing signal.
        grpo_anchor_coefficients = torch.zeros_like(
            result.mean_reduced_policy_coefficients
        )
        if grpo_anchor_scale > 0.0:
            grpo_anchor_coefficients = grpo_anchor_scale * compute_scalarized_group_advantages(
                objective_scores,
                torch.zeros_like(objective_scores),
                B,
                std_correction=int(getattr(self, "group_std_correction", 0)),
            )

        grouped_quality = result.quality_advantages.view(-1, B)
        grouped_recovery = result.recovery_advantages.view(-1, B)
        grouped_failure = result.failure_advantages.view(-1, B)
        grouped_grpo_anchor_coefficients = grpo_anchor_coefficients.view(-1, B)
        raw_quality_coefficients = float(K) * grouped_quality
        raw_quality_rms = torch.sqrt(raw_quality_coefficients.square().mean())
        grpo_anchor_rms = torch.sqrt(
            grouped_grpo_anchor_coefficients.square().mean()
        )
        quality_rescale = 1.0
        quality_norm_active = False
        if quality_anchor_ratio > 0.0:
            # Raw canonical utilities and group-z GRPO advantages have
            # incompatible units. Match the set-quality residual to a declared
            # fraction of the anchor's batch RMS, while retaining its exact
            # centered direction. The detached, capped scalar prevents a tiny
            # best/runner-up margin from exploding into an arbitrary update.
            # Recovery and failure coefficients intentionally remain in raw
            # Lagrangian units so the controller's magnitude keeps its meaning.
            raw_rms_value = float(raw_quality_rms.detach().item())
            anchor_rms_value = float(grpo_anchor_rms.detach().item())
            if raw_rms_value > 1e-8 and anchor_rms_value > 1e-8:
                quality_rescale = min(4.0, anchor_rms_value / raw_rms_value)
                quality_norm_active = True
            quality_coefficients = (
                quality_anchor_ratio
                * quality_rescale
                * raw_quality_coefficients
            )
        else:
            quality_coefficients = exact_scale * raw_quality_coefficients
        quality_coefficient_rms = torch.sqrt(quality_coefficients.square().mean())
        quality_to_anchor_rms_ratio = (
            float((quality_coefficient_rms / grpo_anchor_rms).item())
            if float(grpo_anchor_rms.detach().item()) > 1e-8
            else 0.0
        )
        recovery_coefficients = (
            exact_scale
            * float(K)
            * effective_recovery_scales.unsqueeze(1)
            * grouped_recovery
        )
        base_recovery_coefficients = (
            exact_scale
            * float(K)
            * float(self.clasp_canonical_recovery_scale)
            * grouped_recovery
        )
        dual_recovery_coefficients = (
            recovery_coefficients - base_recovery_coefficients
        )
        failure_coefficients = (
            -exact_scale * float(K) * group_lambdas.unsqueeze(1) * grouped_failure
        )
        exact_coefficients = (
            quality_coefficients + recovery_coefficients + failure_coefficients
        ).reshape(-1)
        dense_auxiliary = None
        dense_coefficients = torch.zeros_like(exact_coefficients)
        if not fixed_null and any(
            value > 0.0
            for value in (
                dense_quality_scale,
                dense_feasibility_scale,
                dense_violation_scale,
                dense_recovery_scale,
            )
        ):
            from clasp.set_grpo import compute_dense_clasp_auxiliary_advantages

            if dense_violation_scores is None:
                raise RuntimeError(
                    "Dense canonical CLASP requires component-aware violation scores"
                )
            dense_auxiliary = compute_dense_clasp_auxiliary_advantages(
                objective_scores,
                dense_violation_scores,
                feasibility_flags,
                rollout_batch_size=B,
                group_lambdas=group_lambdas,
                quality_scale=dense_quality_scale,
                feasibility_scale=dense_feasibility_scale,
                violation_scale=dense_violation_scale,
                recovery_scale=dense_recovery_scale,
            )
            dense_coefficients = dense_auxiliary.advantages
        coefficients = (
            grpo_anchor_coefficients + exact_coefficients + dense_coefficients
        )
        if not torch.isfinite(coefficients).all().item():
            raise RuntimeError(
                f"{credit_name} CLASP set credit produced non-finite policy coefficients"
            )

        grouped_coefficients = coefficients.view(-1, B)
        grouped_dense_coefficients = dense_coefficients.view(-1, B)
        decomposition_residual = (
            grouped_coefficients
            - grouped_grpo_anchor_coefficients
            - quality_coefficients
            - recovery_coefficients
            - failure_coefficients
            - grouped_dense_coefficients
        )
        direction_denominator = torch.linalg.vector_norm(
            raw_quality_coefficients
        ) * torch.linalg.vector_norm(grouped_grpo_anchor_coefficients)
        quality_anchor_cosine = (
            float(
                (
                    (raw_quality_coefficients * grouped_grpo_anchor_coefficients).sum()
                    / direction_denominator
                ).item()
            )
            if float(direction_denominator.detach().item()) > 1e-8
            else 0.0
        )
        failure_at_k = result.deployment_failure_estimates
        full_rollout_failure = (
            ~(feasibility_flags.view(-1, B) > 0.5).any(dim=1)
        ).to(dtype=torch.float32)

        is_training = bool(getattr(getattr(self, "model", None), "training", False))
        failure_ema_values: list[float] = []
        task_lambda_stats: dict[str, float] = {}
        for task, task_idx in task_to_index.items():
            task_mask = group_task_indices == int(task_idx)
            batch_count = int(task_mask.sum().item())
            batch_failure_sum = float(failure_at_k[task_mask].sum().item())
            if is_training:
                self._clasp_bon_pending_failure_sums[task] = float(
                    self._clasp_bon_pending_failure_sums.get(task, 0.0)
                    + batch_failure_sum
                )
                self._clasp_bon_pending_group_counts[task] = int(
                    self._clasp_bon_pending_group_counts.get(task, 0)
                    + batch_count
                )
            failure_ema = float(self._clasp_bon_failure_ema.get(task, 0.0))
            failure_ema_values.append(failure_ema)
            pending_count = int(self._clasp_bon_pending_group_counts.get(task, 0))
            pending_sum = float(self._clasp_bon_pending_failure_sums.get(task, 0.0))
            task_lambda_stats[f"clasp/bon_lambda/{task}"] = float(
                current_lambdas[task_idx].item()
            )
            task_lambda_stats[f"clasp/bon_lambda_used/{task}"] = float(
                current_lambdas[task_idx].item()
            )
            task_lambda_stats[f"clasp/bon_lambda_next/{task}"] = float(
                current_lambdas[task_idx].item()
            )
            task_lambda_stats[f"clasp/bon_failure_ema/{task}"] = failure_ema
            task_lambda_stats[f"clasp/bon_pending_groups/{task}"] = float(
                pending_count
            )
            task_lambda_stats[f"clasp/bon_pending_failure_rate/{task}"] = (
                pending_sum / pending_count if pending_count > 0 else 0.0
            )
            task_lambda_stats[f"clasp/{credit_name}_failure_at_k/{task}"] = (
                batch_failure_sum / batch_count if batch_count > 0 else 0.0
            )

        failed_subset_mask = result.subset_failures > 0.5
        failed_recovery_values = result.subset_recovery_utilities[failed_subset_mask]
        grouped_feasible = feasibility_flags.view(-1, B) > 0.5
        signal_tolerance = 1e-8
        informative_groups = (
            grouped_coefficients.abs().amax(dim=1) > signal_tolerance
        )
        quality_signal_groups = (
            quality_coefficients.abs().amax(dim=1) > signal_tolerance
        )
        recovery_signal_groups = (
            recovery_coefficients.abs().amax(dim=1) > signal_tolerance
        )
        failure_signal_groups = (
            failure_coefficients.abs().amax(dim=1) > signal_tolerance
        )
        grpo_anchor_signal_groups = (
            grouped_grpo_anchor_coefficients.abs().amax(dim=1) > signal_tolerance
        )
        realized_advantage_sum_abs_max = float(
            grouped_coefficients.sum(dim=1).abs().max().item()
        )
        failure_pressure_abs_mean = float(failure_coefficients.abs().mean().item())
        stats = {
            **task_lambda_stats,
            "clasp/canonical_set": float(not fixed_null),
            "clasp/fixed_null_set": float(fixed_null),
            f"clasp/{credit_name}_rollout_batch_size": float(B),
            f"clasp/{credit_name}_deployment_set_size": float(K),
            f"clasp/{credit_name}_subset_count": float(result.subset_indices.size(0)),
            f"clasp/{credit_name}_b_equals_k_plus_one": float(B == K + 1),
            f"clasp/{credit_name}_sequence_importance_sampling": float(
                getattr(self, "importance_sampling_level", None) == "sequence"
            ),
            f"clasp/{credit_name}_token_importance_sampling_ablation": float(
                getattr(self, "importance_sampling_level", None) == "token"
            ),
            f"clasp/{credit_name}_recovery_scale": float(
                self.clasp_canonical_recovery_scale
            ),
            f"clasp/{credit_name}_dual_recovery_coupling": dual_recovery_coupling,
            f"clasp/{credit_name}_effective_recovery_scale_mean": float(
                effective_recovery_scales.mean().item()
            ),
            f"clasp/{credit_name}_effective_recovery_scale_min": float(
                effective_recovery_scales.min().item()
            ),
            f"clasp/{credit_name}_effective_recovery_scale_max": float(
                effective_recovery_scales.max().item()
            ),
            f"clasp/{credit_name}_recovery_temperature": float(
                self.clasp_canonical_recovery_temperature
            ),
            f"clasp/{credit_name}_exact_scale": exact_scale,
            f"clasp/{credit_name}_grpo_anchor_scale": grpo_anchor_scale,
            f"clasp/{credit_name}_grpo_anchor_std_correction": float(
                getattr(self, "group_std_correction", 0)
            ),
            f"clasp/{credit_name}_quality_anchor_ratio": quality_anchor_ratio,
            f"clasp/{credit_name}_quality_norm_active": float(
                quality_norm_active
            ),
            f"clasp/{credit_name}_quality_norm_rescale": quality_rescale,
            f"clasp/{credit_name}_quality_norm_rescale_cap": 4.0,
            f"clasp/{credit_name}_quality_norm_rescale_cap_hit": float(
                quality_norm_active and quality_rescale >= 4.0 - 1e-8
            ),
            f"clasp/{credit_name}_raw_quality_rms": float(
                raw_quality_rms.item()
            ),
            f"clasp/{credit_name}_grpo_anchor_rms": float(
                grpo_anchor_rms.item()
            ),
            f"clasp/{credit_name}_quality_anchor_cosine": quality_anchor_cosine,
            f"clasp/{credit_name}_quality_to_anchor_rms_ratio": (
                quality_to_anchor_rms_ratio
            ),
            f"clasp/{credit_name}_quality_coefficient_is_exact_raw": float(
                quality_anchor_ratio == 0.0
            ),
            f"clasp/{credit_name}_grpo_anchor_abs_mean": float(
                grouped_grpo_anchor_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_grpo_anchor_signal_group_fraction": float(
                grpo_anchor_signal_groups.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_dense_quality_scale": dense_quality_scale,
            f"clasp/{credit_name}_dense_feasibility_scale": dense_feasibility_scale,
            f"clasp/{credit_name}_dense_violation_scale": dense_violation_scale,
            f"clasp/{credit_name}_dense_recovery_scale": dense_recovery_scale,
            f"clasp/{credit_name}_dense_auxiliary_abs_mean": float(
                grouped_dense_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_failure_at_k": float(failure_at_k.mean().item()),
            f"clasp/{credit_name}_full_rollout_failure": float(
                full_rollout_failure.mean().item()
            ),
            f"clasp/{credit_name}_failure_k_minus_failure_b": float(
                (failure_at_k - full_rollout_failure).mean().item()
            ),
            f"clasp/{credit_name}_subset_quality_mean": float(
                result.subset_quality_utilities.mean().item()
            ),
            f"clasp/{credit_name}_failed_recovery_mean": (
                float(failed_recovery_values.mean().item())
                if failed_recovery_values.numel()
                else 0.0
            ),
            f"clasp/{credit_name}_quality_coefficient_abs_mean": float(
                quality_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_raw_quality_coefficient_abs_mean": float(
                raw_quality_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_recovery_coefficient_abs_mean": float(
                recovery_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_dual_recovery_coefficient_abs_mean": float(
                dual_recovery_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_failure_coefficient_abs_mean": (
                failure_pressure_abs_mean
            ),
            f"clasp/{credit_name}_coefficient_abs_mean": float(
                grouped_coefficients.abs().mean().item()
            ),
            f"clasp/{credit_name}_coefficient_nonzero_fraction": float(
                (grouped_coefficients.abs() > signal_tolerance)
                .to(dtype=torch.float32)
                .mean()
                .item()
            ),
            f"clasp/{credit_name}_informative_group_fraction": float(
                informative_groups.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_zero_signal_group_fraction": float(
                (~informative_groups).to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_quality_signal_group_fraction": float(
                quality_signal_groups.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_recovery_signal_group_fraction": float(
                recovery_signal_groups.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_failure_signal_group_fraction": float(
                failure_signal_groups.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_feasible_candidate_fraction": float(
                grouped_feasible.to(dtype=torch.float32).mean().item()
            ),
            f"clasp/{credit_name}_feasible_candidates_per_group_mean": float(
                grouped_feasible.to(dtype=torch.float32).sum(dim=1).mean().item()
            ),
            f"clasp/{credit_name}_coefficient_group_mean_abs_max": float(
                grouped_coefficients.mean(dim=1).abs().max().item()
            ),
            f"clasp/{credit_name}_realized_advantage_sum_abs_max": (
                realized_advantage_sum_abs_max
            ),
            f"clasp/{credit_name}_component_decomposition_residual_abs_max": float(
                decomposition_residual.abs().max().item()
            ),
            "clasp/bon_controller_advantage_delta": failure_pressure_abs_mean,
            "clasp/bon_controller_pressure_observed": float(
                bool((group_lambdas <= 1e-8).all().item())
                or failure_pressure_abs_mean > 0.0
                or bool((grouped_failure.abs() <= 1e-8).all().item())
            ),
            "clasp/bon_target_failure_rate_mean": float(target_rates.mean().item()),
            "clasp/bon_lambda_used_mean": float(current_lambdas.mean().item()),
            "clasp/bon_lambda_next_mean": float(current_lambdas.mean().item()),
            "clasp/bon_failure_ema_mean": float(
                sum(failure_ema_values) / max(len(failure_ema_values), 1)
            ),
            "clasp/bon_frontier_cdf": 0.0,
        }
        if not fixed_null:
            stats["clasp/canonical_advantage_centering_residual_abs_max"] = (
                realized_advantage_sum_abs_max
            )
        if dense_auxiliary is not None:
            stats.update(
                {
                    "clasp/canonical_dense_auxiliary": 1.0,
                    "clasp/canonical_dense_all_infeasible_group_rate": float(
                        dense_auxiliary.all_infeasible_groups.float().mean().item()
                    ),
                    "clasp/canonical_dense_mixed_group_rate": float(
                        dense_auxiliary.mixed_groups.float().mean().item()
                    ),
                    "clasp/canonical_dense_all_feasible_group_rate": float(
                        dense_auxiliary.all_feasible_groups.float().mean().item()
                    ),
                    "clasp/canonical_dense_quality_abs_mean": float(
                        dense_auxiliary.quality_advantages.abs().mean().item()
                    ),
                    "clasp/canonical_dense_feasibility_abs_mean": float(
                        dense_auxiliary.feasibility_advantages.abs().mean().item()
                    ),
                    "clasp/canonical_dense_violation_abs_mean": float(
                        dense_auxiliary.violation_advantages.abs().mean().item()
                    ),
                    "clasp/canonical_dense_recovery_abs_mean": float(
                        dense_auxiliary.recovery_advantages.abs().mean().item()
                    ),
                    "clasp/canonical_dense_centering_residual_abs_max": float(
                        dense_auxiliary.centering_residuals.abs().max().item()
                    ),
                }
            )
        else:
            stats["clasp/canonical_dense_auxiliary"] = 0.0
        return coefficients, stats

    def _compute_canonical_set_advantages(
        self,
        objective_scores: torch.Tensor,
        violation_scores: torch.Tensor,
        feasibility_flags: torch.Tensor,
        gathered_task_types: list[str] | None,
        dense_violation_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._compute_exact_set_advantages(
            objective_scores,
            violation_scores,
            feasibility_flags,
            gathered_task_types,
            dense_violation_scores=dense_violation_scores,
        )

    def _compute_fixed_null_set_advantages(
        self,
        objective_scores: torch.Tensor,
        violation_scores: torch.Tensor,
        feasibility_flags: torch.Tensor,
        null_objective_scores: torch.Tensor,
        null_violation_scores: torch.Tensor,
        null_feasibility_flags: torch.Tensor,
        gathered_task_types: list[str] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._compute_exact_set_advantages(
            objective_scores,
            violation_scores,
            feasibility_flags,
            gathered_task_types,
            null_objective_scores=null_objective_scores,
            null_violation_scores=null_violation_scores,
            null_feasibility_flags=null_feasibility_flags,
        )

    def _compute_bon_advantages(
        self,
        primal_scores: torch.Tensor,
        violation_scores: torch.Tensor,
        feasibility_flags: torch.Tensor,
        gathered_task_types: list[str] | None,
        quality_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_len = int(primal_scores.numel())
        group_tasks = self._bon_group_task_names(gathered_task_types, total_len)
        task_order = sorted(set(group_tasks))
        task_to_index = {task: idx for idx, task in enumerate(task_order)}
        group_task_indices = torch.tensor(
            [task_to_index[task] for task in group_tasks],
            dtype=torch.long,
            device=primal_scores.device,
        )

        self._load_bon_lambdas()
        for task in task_order:
            self._clasp_bon_lambdas.setdefault(task, float(self.clasp_bon_lambda_init))
        current_lambdas = torch.tensor(
            [self._clasp_bon_lambdas[task] for task in task_order],
            dtype=torch.float32,
            device=primal_scores.device,
        )
        target_rates = torch.tensor(
            [
                self.clasp_bon_target_failure_rates.get(
                    task,
                    self.clasp_bon_target_failure_rate,
                )
                for task in task_order
            ],
            dtype=torch.float32,
            device=primal_scores.device,
        )
        grouped_feasible = feasibility_flags.view(-1, self.num_generations) > 0.5
        group_failure_flags = (~grouped_feasible.any(dim=1)).to(dtype=torch.float32)

        # Every generation call contributing to one accumulated optimizer step
        # must use the same detached controller state.  This method only queues
        # sufficient statistics; compute_loss commits them once when Accelerate
        # reaches its gradient-synchronization boundary.
        group_lambdas = current_lambdas.detach()[group_task_indices]
        effective_quality_scores = primal_scores if quality_scores is None else quality_scores
        frontier_cdf_enabled = bool(getattr(self, "clasp_frontier_cdf", False))

        if frontier_cdf_enabled:
            from clasp.set_grpo import compute_frontier_cdf_advantages

            missing_targets = sorted(
                task for task in task_order if task not in self.clasp_frontier_cdf_gap_targets
            )
            if missing_targets:
                raise RuntimeError(
                    "Frontier-CDF has no positive gap target for tasks: "
                    + ", ".join(missing_targets)
                )
            group_target_gaps = torch.tensor(
                [self.clasp_frontier_cdf_gap_targets[task] for task in group_tasks],
                dtype=torch.float32,
                device=primal_scores.device,
            )
            quality_temperature = self._frontier_cdf_quality_temperature()

            def compute_with_lambdas(lambdas: torch.Tensor):
                result = compute_frontier_cdf_advantages(
                    effective_quality_scores,
                    violation_scores,
                    feasibility_flags,
                    num_generations=self.num_generations,
                    group_lambdas=lambdas,
                    group_target_gaps=group_target_gaps,
                    quality_temperature=quality_temperature,
                    violation_temperature=self.clasp_frontier_cdf_violation_temperature,
                    recovery_scale=self.clasp_frontier_cdf_recovery_scale,
                    objective_score_scale=self.clasp_frontier_cdf_objective_score_scale,
                    gap_slope=self.clasp_frontier_cdf_gap_slope,
                    feasibility_violation_tolerance=(
                        self.clasp_frontier_cdf_feasibility_violation_tolerance
                    ),
                )
                stats = dict(result.stats)
                relative_failed_recovery = bool(
                    getattr(
                        self,
                        "clasp_frontier_cdf_relative_failed_recovery",
                        False,
                    )
                )
                stats["frontier_cdf/relative_failed_recovery"] = float(
                    relative_failed_recovery
                )
                stats["frontier_cdf/relative_recovery_scale"] = float(
                    getattr(
                        self,
                        "clasp_frontier_cdf_relative_recovery_scale",
                        1.0,
                    )
                )
                if not relative_failed_recovery:
                    return result.advantages, stats

                # The fixed-null FCDF difference pays -lambda in both the full
                # and counterfactual set when every completion is infeasible,
                # so its learned feasibility multiplier cancels. Replace only
                # those failed-group credits with the already-tested signed,
                # centered exact-BoN recovery:
                #
                #   A_i = scale * (1 + lambda_g) * [r_i - mean_j(r_j)]
                #
                # where r blends a robust lower-violation score and tie-aware
                # lower-violation rank. Lambda now multiplies the policy signal
                # and therefore cannot cancel. Successful groups retain FCDF's
                # quality/failure null-difference credit exactly.
                from clasp.rewards.clasp_math import compute_bon_aligned_advantages

                relative_advantages, _ = compute_bon_aligned_advantages(
                    primal_scores,
                    violation_scores,
                    feasibility_flags,
                    num_generations=self.num_generations,
                    group_lambdas=lambdas,
                    quality_scores=effective_quality_scores,
                    quality_temperature=0.0,
                    success_credit_mode="exact_loo",
                    dense_violation_mix=self.clasp_bon_dense_violation_mix,
                    recovery_scale=float(
                        getattr(
                            self,
                            "clasp_frontier_cdf_relative_recovery_scale",
                            1.0,
                        )
                    ),
                )
                relative_advantages = relative_advantages.to(
                    device=result.advantages.device,
                    dtype=result.advantages.dtype,
                )
                failed_rows = result.group_failures.repeat_interleave(
                    self.num_generations
                )
                hybrid_advantages = torch.where(
                    failed_rows,
                    relative_advantages,
                    result.advantages,
                )

                failed_relative = relative_advantages[failed_rows]
                failed_fixed_null = result.advantages[failed_rows]
                grouped_relative = relative_advantages.view(
                    -1,
                    self.num_generations,
                )
                failed_groups = result.group_failures
                signed_failed_groups = (
                    (grouped_relative > 1e-8).any(dim=1)
                    & (grouped_relative < -1e-8).any(dim=1)
                    & failed_groups
                )
                failed_count = int(failed_groups.sum().item())
                stats.update(
                    {
                        "frontier_cdf/fixed_null_advantage_mean": float(
                            result.advantages.mean().item()
                        ),
                        "frontier_cdf/fixed_null_advantage_abs_max": float(
                            result.advantages.abs().max().item()
                        ),
                        "frontier_cdf/relative_recovery_abs_mean_failed": (
                            float(failed_relative.abs().mean().item())
                            if failed_relative.numel()
                            else 0.0
                        ),
                        "frontier_cdf/relative_recovery_positive_rate_failed": (
                            float((failed_relative > 1e-8).float().mean().item())
                            if failed_relative.numel()
                            else 0.0
                        ),
                        "frontier_cdf/relative_recovery_negative_rate_failed": (
                            float((failed_relative < -1e-8).float().mean().item())
                            if failed_relative.numel()
                            else 0.0
                        ),
                        "frontier_cdf/relative_recovery_zero_rate_failed": (
                            float((failed_relative.abs() <= 1e-8).float().mean().item())
                            if failed_relative.numel()
                            else 0.0
                        ),
                        "frontier_cdf/relative_recovery_signed_group_rate_failed": (
                            float(signed_failed_groups.float().sum().item())
                            / float(failed_count)
                            if failed_count
                            else 0.0
                        ),
                        "frontier_cdf/relative_recovery_group_mean_abs_max_failed": (
                            float(
                                grouped_relative[failed_groups]
                                .mean(dim=1)
                                .abs()
                                .max()
                                .item()
                            )
                            if failed_count
                            else 0.0
                        ),
                        "frontier_cdf/fixed_null_recovery_abs_mean_replaced": (
                            float(failed_fixed_null.abs().mean().item())
                            if failed_fixed_null.numel()
                            else 0.0
                        ),
                        "frontier_cdf/advantage_mean": float(
                            hybrid_advantages.mean().item()
                        ),
                        "frontier_cdf/advantage_abs_max": float(
                            hybrid_advantages.abs().max().item()
                        ),
                        "frontier_cdf/zero_advantage_rate": float(
                            (hybrid_advantages.abs() <= 1e-8).float().mean().item()
                        ),
                    }
                )
                return hybrid_advantages, stats

        else:
            from clasp.rewards.clasp_math import compute_bon_aligned_advantages

            def compute_with_lambdas(lambdas: torch.Tensor):
                return compute_bon_aligned_advantages(
                    primal_scores,
                    violation_scores,
                    feasibility_flags,
                    num_generations=self.num_generations,
                    group_lambdas=lambdas,
                    quality_scores=effective_quality_scores,
                    quality_temperature=self.clasp_bon_quality_temperature,
                    success_credit_mode=getattr(
                        self,
                        "clasp_bon_success_credit",
                        "dense_rank",
                    ),
                    dense_quality_scale=float(
                        getattr(self, "clasp_bon_dense_quality_scale", 1.0)
                    ),
                    dense_feasibility_scale=float(
                        getattr(self, "clasp_bon_dense_feasibility_scale", 1.0)
                    ),
                    dense_violation_mix=self.clasp_bon_dense_violation_mix,
                    recovery_scale=self.clasp_bon_recovery_scale,
                )

        advantages, advantage_stats = compute_with_lambdas(group_lambdas)

        # Diagnose the exact regression that made the old strict branch ignore
        # learned controller pressure. The executable unit test enforces the
        # wiring at a numerically meaningful lambda; runtime only logs the
        # delta because tiny projected lambdas can round away in float32.
        feasibility_counts = grouped_feasible.sum(dim=1)
        grouped_violations = violation_scores.view(-1, self.num_generations)
        failed_group_has_recovery_signal = (
            grouped_violations.max(dim=1).values
            - grouped_violations.min(dim=1).values
        ) > 1e-8
        relative_failed_recovery_enabled = bool(
            getattr(
                self,
                "clasp_frontier_cdf_relative_failed_recovery",
                False,
            )
        )
        if frontier_cdf_enabled and relative_failed_recovery_enabled:
            pressure_relevant = (group_lambdas > 1e-8) & (
                (group_failure_flags.bool() & failed_group_has_recovery_signal)
                | (feasibility_counts == 1)
            )
        elif frontier_cdf_enabled:
            # In failed groups the exact lambda term cancels between the full
            # set and its fixed-null counterfactual. Recovery—not lambda—moves
            # the policy until a sole feasible completion appears.
            pressure_relevant = (group_lambdas > 1e-8) & (feasibility_counts == 1)
        else:
            success_credit_mode = str(
                getattr(self, "clasp_bon_success_credit", "dense_rank")
            ).strip().lower()
            if success_credit_mode == "dense_rank":
                mixed_success = (feasibility_counts > 0) & (
                    feasibility_counts < int(self.num_generations)
                )
                pressure_relevant = (group_lambdas > 1e-8) & (
                    (group_failure_flags.bool() & failed_group_has_recovery_signal)
                    | mixed_success
                )
            else:
                pressure_relevant = (group_lambdas > 1e-8) & (
                    (group_failure_flags.bool() & failed_group_has_recovery_signal)
                    | (feasibility_counts == 1)
                )
        pressure_delta = 0.0
        if bool(pressure_relevant.any().item()):
            zero_pressure_advantages, _ = compute_with_lambdas(
                torch.zeros_like(group_lambdas)
            )
            relevant_rows = pressure_relevant.repeat_interleave(self.num_generations)
            pressure_delta = float(
                (advantages[relevant_rows] - zero_pressure_advantages[relevant_rows])
                .abs()
                .mean()
                .item()
            )
        if not torch.isfinite(advantages).all().item():
            raise RuntimeError("CLASP BoN produced non-finite optimized advantages")

        next_lambdas = current_lambdas.detach()
        failure_ema_values: list[float] = []
        is_training = bool(getattr(getattr(self, "model", None), "training", False))
        for task, task_idx in task_to_index.items():
            task_mask = group_task_indices == int(task_idx)
            batch_count = int(task_mask.sum().item())
            batch_failure_sum = float(group_failure_flags[task_mask].sum().item())
            if is_training:
                self._clasp_bon_pending_failure_sums[task] = float(
                    self._clasp_bon_pending_failure_sums.get(task, 0.0)
                    + batch_failure_sum
                )
                self._clasp_bon_pending_group_counts[task] = int(
                    self._clasp_bon_pending_group_counts.get(task, 0)
                    + batch_count
                )
            failure_ema_values.append(
                float(self._clasp_bon_failure_ema.get(task, 0.0))
            )

        task_lambda_stats: dict[str, float] = {}
        for task, idx in task_to_index.items():
            task_lambda_stats[f"clasp/bon_lambda/{task}"] = float(current_lambdas[idx].item())
            task_lambda_stats[f"clasp/bon_lambda_used/{task}"] = float(
                current_lambdas[idx].item()
            )
            task_lambda_stats[f"clasp/bon_lambda_next/{task}"] = float(next_lambdas[idx].item())
            task_lambda_stats[f"clasp/bon_failure_ema/{task}"] = float(
                failure_ema_values[idx]
            )
            pending_count = int(self._clasp_bon_pending_group_counts.get(task, 0))
            pending_sum = float(self._clasp_bon_pending_failure_sums.get(task, 0.0))
            task_lambda_stats[f"clasp/bon_pending_groups/{task}"] = float(pending_count)
            task_lambda_stats[f"clasp/bon_pending_failure_rate/{task}"] = (
                pending_sum / pending_count if pending_count > 0 else 0.0
            )
        return advantages, {
            **advantage_stats,
            **task_lambda_stats,
            "clasp/bon_controller_advantage_delta": pressure_delta,
            "clasp/bon_controller_pressure_observed": float(
                (not bool(pressure_relevant.any().item())) or pressure_delta > 0.0
            ),
            "clasp/bon_target_failure_rate_mean": float(target_rates.mean().item()),
            "clasp/bon_lambda_used_mean": float(current_lambdas.mean().item()),
            "clasp/bon_lambda_next_mean": float(next_lambdas.mean().item()),
            "clasp/bon_failure_ema_mean": float(
                sum(failure_ema_values) / max(len(failure_ema_values), 1)
            ),
            "clasp/bon_frontier_cdf": float(frontier_cdf_enabled),
        }

    def _commit_pending_bon_controller_update(
        self,
        device: torch.device | str | None = None,
    ) -> dict[str, float]:
        """Commit exactly one signed BoN dual update for an optimizer step.

        ``_compute_bon_advantages`` is called at generation cadence, which can
        be more frequent than optimizer cadence under gradient accumulation.
        It records globally gathered failure sums/counts.  This method combines
        every queued call, updates the EMA once from the combined rate, persists
        the resulting controller state, and clears the queue atomically after a
        successful update.
        """

        from clasp.rewards.clasp_math import update_signed_bon_lambdas

        task_order = sorted(
            task
            for task, count in self._clasp_bon_pending_group_counts.items()
            if int(count) > 0
        )
        if not task_order:
            return {}

        self._load_bon_lambdas()
        for task in task_order:
            self._clasp_bon_lambdas.setdefault(task, float(self.clasp_bon_lambda_init))

        tensor_device = torch.device("cpu") if device is None else torch.device(device)
        current_lambdas = torch.tensor(
            [self._clasp_bon_lambdas[task] for task in task_order],
            dtype=torch.float32,
            device=tensor_device,
        )
        batch_failure_rates: list[float] = []
        next_failure_ema: list[float] = []
        for task in task_order:
            batch_count = int(self._clasp_bon_pending_group_counts[task])
            batch_rate = float(self._clasp_bon_pending_failure_sums[task]) / batch_count
            batch_failure_rates.append(batch_rate)

            previous_count = int(self._clasp_bon_observed_groups.get(task, 0))
            previous_ema = self._clasp_bon_failure_ema.get(task)
            if previous_ema is None or previous_count <= 0:
                updated_ema = batch_rate
            else:
                beta = float(self.clasp_bon_dual_ema_beta)
                updated_ema = beta * float(previous_ema) + (1.0 - beta) * batch_rate
            next_failure_ema.append(float(updated_ema))

        target_rates = torch.tensor(
            [
                self.clasp_bon_target_failure_rates.get(
                    task,
                    self.clasp_bon_target_failure_rate,
                )
                for task in task_order
            ],
            dtype=torch.float32,
            device=tensor_device,
        )
        controller_observations = torch.tensor(
            next_failure_ema,
            dtype=torch.float32,
            device=tensor_device,
        )
        if bool(getattr(self, "clasp_frontier_cdf", False)):
            from clasp.set_grpo import update_signed_bucket_lambdas_from_counts

            # Each task contributes one globally aggregated EMA observation at
            # this optimizer boundary. Treat it as one fractional Bernoulli
            # sufficient statistic; this is algebraically the signed update
            # lambda <- project(lambda + eta*(EMA_failure-target)).
            controller_result = update_signed_bucket_lambdas_from_counts(
                current_lambdas.detach(),
                controller_observations,
                torch.ones_like(controller_observations),
                target_failure_rates=target_rates,
                learning_rates=self.clasp_bon_dual_lr,
                lambda_max=self.clasp_bon_lambda_max,
            )
            next_lambdas = controller_result.lambdas
            dual_stats = controller_result.stats
        else:
            next_lambdas, dual_stats = update_signed_bon_lambdas(
                current_lambdas.detach(),
                controller_observations,
                group_task_indices=torch.arange(
                    len(task_order),
                    dtype=torch.long,
                    device=tensor_device,
                ),
                target_failure_rates=target_rates,
                learning_rates=self.clasp_bon_dual_lr,
                lambda_max=self.clasp_bon_lambda_max,
            )

        commit_stats: dict[str, float] = {
            **dual_stats,
            "clasp/bon_controller_commit": 1.0,
            "clasp/bon_committed_group_count": float(
                sum(self._clasp_bon_pending_group_counts[task] for task in task_order)
            ),
            "clasp/bon_committed_failure_rate_mean": float(
                sum(batch_failure_rates) / len(batch_failure_rates)
            ),
        }
        next_lambda_values = next_lambdas.detach().cpu().tolist()
        for idx, task in enumerate(task_order):
            batch_count = int(self._clasp_bon_pending_group_counts[task])
            self._clasp_bon_failure_ema[task] = float(next_failure_ema[idx])
            self._clasp_bon_observed_groups[task] = int(
                self._clasp_bon_observed_groups.get(task, 0) + batch_count
            )
            self._clasp_bon_lambdas[task] = float(next_lambda_values[idx])
            commit_stats[f"clasp/bon_committed_lambda_used/{task}"] = float(
                current_lambdas[idx].item()
            )
            commit_stats[f"clasp/bon_committed_lambda_next/{task}"] = float(
                next_lambda_values[idx]
            )
            commit_stats[f"clasp/bon_committed_failure_rate/{task}"] = float(
                batch_failure_rates[idx]
            )
            commit_stats[f"clasp/bon_committed_failure_ema/{task}"] = float(
                next_failure_ema[idx]
            )
            commit_stats[f"clasp/bon_committed_group_count/{task}"] = float(batch_count)
            residual = float(next_failure_ema[idx]) - float(target_rates[idx].item())
            requested_delta = float(self.clasp_bon_dual_lr) * residual
            actual_delta = float(next_lambda_values[idx]) - float(
                current_lambdas[idx].item()
            )
            commit_stats[f"clasp/bon_committed_signed_residual/{task}"] = residual
            commit_stats[f"clasp/bon_committed_requested_delta/{task}"] = (
                requested_delta
            )
            commit_stats[f"clasp/bon_committed_actual_delta/{task}"] = actual_delta
            commit_stats[f"clasp/bon_committed_at_zero/{task}"] = float(
                next_lambda_values[idx] == 0.0
            )
            commit_stats[f"clasp/bon_committed_at_max/{task}"] = float(
                next_lambda_values[idx] == float(self.clasp_bon_lambda_max)
            )

        self._store_bon_lambdas()
        self._clasp_bon_pending_failure_sums.clear()
        self._clasp_bon_pending_group_counts.clear()
        return commit_stats

    def _recompute_constrained_terms(
        self, components: dict[str, list[float]]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, float],
    ]:
        device = self.accelerator.device

        primal_scores = torch.tensor(components["primal_scores"], dtype=torch.float32, device=device)
        objective_quality_fallback = not isinstance(components.get("objective_scores"), list)
        raw_objective_scores = components.get("objective_scores", components["primal_scores"])
        if self.clasp_bon_aligned and objective_quality_fallback and not self._clasp_objective_fallback_warning_emitted:
            process_index = int(getattr(getattr(self, "accelerator", None), "process_index", 0))
            if process_index == 0:
                print(
                    "[CLASP][QualitySafety] objective_scores missing; falling back to mixed "
                    "primal_scores for legacy compatibility.",
                    flush=True,
                )
            self._clasp_objective_fallback_warning_emitted = True
        objective_scores = torch.tensor(
            raw_objective_scores,
            dtype=torch.float32,
            device=device,
        )
        raw_weighted_gap_scores = components.get("weighted_gap_scores", components.get("weighted_cost_scores"))
        weighted_gap_scores = torch.tensor(
            raw_weighted_gap_scores, dtype=torch.float32, device=device
        )
        positive_weighted_gap_scores = torch.tensor(
            components.get("positive_weighted_gap_scores", raw_weighted_gap_scores),
            dtype=torch.float32,
            device=device,
        )
        raw_positive_gap_scores = torch.tensor(
            components.get(
                "raw_positive_gap_scores",
                components.get("positive_weighted_gap_scores", raw_weighted_gap_scores),
            ),
            dtype=torch.float32,
            device=device,
        )
        dual_aware_api_available = isinstance(
            components.get("dual_aware_recovery_gap_scores"),
            list,
        )
        dual_aware_recovery_gap_scores = torch.tensor(
            components.get(
                "dual_aware_recovery_gap_scores",
                components.get(
                    "raw_positive_gap_scores",
                    components.get("positive_weighted_gap_scores", raw_weighted_gap_scores),
                ),
            ),
            dtype=torch.float32,
            device=device,
        )
        violation_scores = torch.tensor(
            components.get("constraint_violation_scores", [0.0] * len(components["primal_scores"])),
            dtype=torch.float32,
            device=device,
        )
        max_constraint_violation_scores = torch.tensor(
            components.get(
                "max_constraint_violation_scores",
                components.get(
                    "constraint_violation_scores",
                    [0.0] * len(components["primal_scores"]),
                ),
            ),
            dtype=torch.float32,
            device=device,
        )
        feasibility_flags = torch.tensor(
            components.get("feasibility_flags", [0.0] * len(components["primal_scores"])),
            dtype=torch.float32,
            device=device,
        )
        fixed_null_component_names = (
            "fixed_null_objective_scores",
            "fixed_null_max_constraint_violation_scores",
            "fixed_null_feasibility_flags",
        )
        fixed_null_components: list[torch.Tensor] = []
        if bool(getattr(self, "clasp_fixed_null_set", False)):
            for component_name in fixed_null_component_names:
                values = components.get(component_name)
                if not isinstance(values, list) or len(values) != len(
                    components["primal_scores"]
                ):
                    raise RuntimeError(
                        "Fixed-null SET-GRPO did not receive a complete construction "
                        f"receipt component {component_name!r}. Refusing an implicit null."
                    )
                fixed_null_components.append(
                    torch.tensor(values, dtype=torch.float32, device=device)
                )
        constraint_scores = torch.tensor(
            components["constraint_scores"], dtype=torch.float32, device=device
        )

        local_len = int(primal_scores.numel())
        if local_len == 0:
            raise RuntimeError("Constrained-GRPO reward manager returned no per-sample components.")
        if int(objective_scores.numel()) != local_len:
            raise RuntimeError(
                "Constrained-GRPO objective quality count does not match primal count: "
                f"objective={int(objective_scores.numel())}, primal={local_len}"
            )

        gathered_primal_scores = _gather_tensor(self, primal_scores)
        gathered_objective_scores = _gather_tensor(self, objective_scores)
        gathered_weighted_gap_scores = _gather_tensor(self, weighted_gap_scores)
        gathered_positive_weighted_gap_scores = _gather_tensor(self, positive_weighted_gap_scores)
        gathered_raw_positive_gap_scores = _gather_tensor(self, raw_positive_gap_scores)
        gathered_dual_aware_recovery_gap_scores = _gather_tensor(
            self,
            dual_aware_recovery_gap_scores,
        )
        gathered_violation_scores = _gather_tensor(self, violation_scores)
        gathered_max_constraint_violation_scores = _gather_tensor(
            self,
            max_constraint_violation_scores,
        )
        gathered_feasibility_flags = _gather_tensor(self, feasibility_flags)
        gathered_fixed_null_components = [
            _gather_tensor(self, value) for value in fixed_null_components
        ]
        gathered_constraint_scores = _gather_tensor(self, constraint_scores)
        active_recovery_gap_scores = _compose_clasp_recovery_gaps(
            gathered_raw_positive_gap_scores,
            gathered_dual_aware_recovery_gap_scores,
            component_dual_scale=float(
                getattr(self, "clasp_component_dual_recovery_scale", 1.0)
            ),
        )

        if gathered_primal_scores.numel() % self.num_generations != 0:
            raise RuntimeError(
                "Constrained-GRPO gathered sample count is not divisible by num_generations. "
                f"count={gathered_primal_scores.numel()}, num_generations={self.num_generations}"
            )
        gathered_task_types = None
        local_task_types = components.get("task_types")
        if isinstance(local_task_types, list):
            gathered_task_types = self._gather_task_types(
                [str(task_name) for task_name in local_task_types],
                int(gathered_primal_scores.numel()),
            )

        if self.scale_rewards not in {None, False, "none"}:
            raise RuntimeError(
                f"Unsupported scale_rewards setting for constrained GRPO: {self.scale_rewards!r}"
            )

        lexicographic_stats: dict[str, float] = {}
        clasp_stats: dict[str, float] = {}
        reward_advantages_global = self._standardize_group_values(gathered_primal_scores)
        reward_advantages_global = self.primal_advantage_scale * reward_advantages_global
        current_penalty_scale = self._current_penalty_scale()
        penalty_source_scores = (
            gathered_positive_weighted_gap_scores
            if self.positive_constraint_penalty or self.separate_z
            else gathered_weighted_gap_scores
        )
        penalty_weighted_gap_scores = current_penalty_scale * penalty_source_scores
        if self.clasp_enabled and bool(getattr(self, "clasp_fixed_null_set", False)):
            if len(gathered_fixed_null_components) != 3:
                raise RuntimeError(
                    "Fixed-null SET-GRPO has no complete gathered construction receipt."
                )
            combined_advantages_global, clasp_stats = (
                self._compute_fixed_null_set_advantages(
                    gathered_objective_scores,
                    gathered_max_constraint_violation_scores,
                    gathered_feasibility_flags,
                    gathered_fixed_null_components[0],
                    gathered_fixed_null_components[1],
                    gathered_fixed_null_components[2],
                    gathered_task_types,
                )
            )
        elif self.clasp_enabled and bool(getattr(self, "clasp_canonical_set", False)):
            combined_advantages_global, clasp_stats = (
                self._compute_canonical_set_advantages(
                    gathered_objective_scores,
                    gathered_max_constraint_violation_scores,
                    gathered_feasibility_flags,
                    gathered_task_types,
                    active_recovery_gap_scores,
                )
            )
        elif self.clasp_enabled and self.clasp_bon_aligned:
            combined_advantages_global, clasp_stats = self._compute_bon_advantages(
                gathered_primal_scores,
                (
                    gathered_max_constraint_violation_scores
                    if getattr(self, "clasp_frontier_cdf", False)
                    else active_recovery_gap_scores
                ),
                gathered_feasibility_flags,
                gathered_task_types,
                quality_scores=gathered_objective_scores,
            )
        elif self.clasp_enabled and self.clasp_pareto_anchor:
            # Legacy pre-BoN Pareto-anchor ablation. The default BoN-aligned
            # estimator bypasses this branch.
            if self.clasp_lexicographic_gate:
                from clasp.rewards.clasp_math import compute_lexicographic_clasp_advantages

                combined_advantages_global, clasp_stats = (
                    compute_lexicographic_clasp_advantages(
                        gathered_primal_scores,
                        active_recovery_gap_scores,
                        gathered_feasibility_flags,
                        num_generations=self.num_generations,
                        raw_violation_scores=active_recovery_gap_scores,
                    )
                )
            elif self.clasp_feasibility_gated:
                from clasp.rewards.clasp_math import compute_feasibility_gated_advantages

                combined_advantages_global, clasp_stats = compute_feasibility_gated_advantages(
                    gathered_primal_scores,
                    active_recovery_gap_scores,
                    gathered_feasibility_flags,
                    num_generations=self.num_generations,
                    rank_violation_scores=active_recovery_gap_scores,
                )
            else:
                from clasp.rewards.clasp_math import compute_pareto_anchored_advantages

                combined_advantages_global, clasp_stats = compute_pareto_anchored_advantages(
                    gathered_primal_scores,
                    active_recovery_gap_scores,
                    gathered_feasibility_flags,
                    num_generations=self.num_generations,
                    rank_violation_scores=active_recovery_gap_scores,
                    tau=self.clasp_pareto_tau,
                    feasibility_center=self.clasp_pareto_center,
                )
        elif self.group_scalarization and not self.clasp_enabled:
            combined_advantages_global = compute_scalarized_group_advantages(
                self.primal_advantage_scale * gathered_primal_scores,
                penalty_weighted_gap_scores,
                num_generations=self.num_generations,
                std_correction=int(getattr(self, "group_std_correction", 0)),
            )
            scalarized_scores = (
                self.primal_advantage_scale * gathered_primal_scores
                - penalty_weighted_gap_scores
            )
            lexicographic_stats = {
                "dual_primal/group_scalarization": 1.0,
                "dual_primal/group_std_correction": float(
                    getattr(self, "group_std_correction", 0)
                ),
                "dual_primal/scalarized_score_mean": float(
                    scalarized_scores.mean().item()
                ),
                "dual_primal/scalarized_score_std": float(
                    scalarized_scores.std(unbiased=False).item()
                ),
            }
        elif self.separate_z and not self.clasp_enabled:
            combined_advantages_global = compute_separate_z_group_advantages(
                gathered_primal_scores,
                gathered_positive_weighted_gap_scores,
                num_generations=self.num_generations,
            )
            reward_advantages_global = compute_separate_z_group_advantages(
                gathered_primal_scores,
                torch.zeros_like(gathered_positive_weighted_gap_scores),
                num_generations=self.num_generations,
            )
            grouped_penalties = gathered_positive_weighted_gap_scores.view(
                -1, self.num_generations
            )
            centered_penalties = grouped_penalties - grouped_penalties.mean(
                dim=1, keepdim=True
            )
            penalty_z = centered_penalties / (
                grouped_penalties.std(dim=1, keepdim=True, unbiased=False) + 1e-4
            )
            lexicographic_stats = {
                "dual_primal/separate_z": 1.0,
                "dual_primal/separate_z_reward_z_std": float(
                    reward_advantages_global.std(unbiased=False).item()
                ),
                "dual_primal/separate_z_penalty_z_std": float(
                    penalty_z.std(unbiased=False).item()
                ),
            }
        elif self.lexicographic_advantage and not self.clasp_enabled:
            combined_advantages_global, lexicographic_stats = self._compute_lexicographic_advantages(
                gathered_primal_scores,
                gathered_violation_scores,
                gathered_feasibility_flags,
            )
        else:
            combined_advantages_global = reward_advantages_global - penalty_weighted_gap_scores

        warmup_flags = self._clasp_warmup_flags_for_tasks(
            gathered_task_types,
            int(gathered_primal_scores.numel()),
            device,
        )
        if (
            self.clasp_enabled
            and warmup_flags is not None
            and bool(warmup_flags.any().item())
        ):
            from clasp.rewards.clasp_math import compute_feasibility_warmup_advantages

            warmup_advantages_global, warmup_stats = compute_feasibility_warmup_advantages(
                gathered_raw_positive_gap_scores,
                gathered_feasibility_flags,
                num_generations=self.num_generations,
                sample_warmup_flags=warmup_flags,
                feasible_margin=self.clasp_feasibility_warmup_margin,
                violation_scale=self.clasp_feasibility_warmup_violation_scale,
                rank_scale=self.clasp_feasibility_warmup_rank_scale,
            )
            combined_advantages_global = torch.where(
                warmup_flags,
                warmup_advantages_global,
                combined_advantages_global,
            )
            released_tasks = 0
            active_tasks = 0
            if gathered_task_types:
                for task_name in sorted({str(task).lower() for task in gathered_task_types}):
                    if self._clasp_task_warmup_active(task_name):
                        active_tasks += 1
                    else:
                        released_tasks += 1
            clasp_stats = {
                **clasp_stats,
                **warmup_stats,
                "clasp/feasibility_warmup_task_active_count": float(active_tasks),
                "clasp/feasibility_warmup_task_released_count": float(released_tasks),
            }

        if self.clasp_enabled and self.clasp_hard_group_weight_scale > 0.0:
            from clasp.rewards.clasp_math import compute_hard_group_weighted_advantages

            combined_advantages_global, hard_group_stats = compute_hard_group_weighted_advantages(
                combined_advantages_global,
                gathered_raw_positive_gap_scores,
                num_generations=self.num_generations,
                scale=self.clasp_hard_group_weight_scale,
                threshold=self.clasp_hard_group_weight_threshold,
                max_weight=self.clasp_hard_group_weight_max,
            )
            clasp_stats = {**clasp_stats, **hard_group_stats}

        stats = {
            "mean_primal": float(gathered_primal_scores.mean().item()),
            "mean_objective_quality": float(gathered_objective_scores.mean().item()),
            "objective_quality_fallback": float(objective_quality_fallback),
            "mean_weighted_gap": float(gathered_weighted_gap_scores.mean().item()),
            "mean_positive_weighted_gap": float(gathered_positive_weighted_gap_scores.mean().item()),
            "mean_raw_positive_gap": float(gathered_raw_positive_gap_scores.mean().item()),
            "mean_dual_aware_recovery_gap": float(
                gathered_dual_aware_recovery_gap_scores.mean().item()
            ),
            "mean_active_recovery_gap": float(active_recovery_gap_scores.mean().item()),
            "component_dual_recovery_scale": float(
                getattr(self, "clasp_component_dual_recovery_scale", 1.0)
            ),
            "component_dual_recovery_contribution": float(
                (
                    active_recovery_gap_scores - gathered_raw_positive_gap_scores
                ).abs().mean().item()
            ),
            "component_dual_recovery_active": float(
                self.clasp_enabled
                and not (
                    bool(getattr(self, "clasp_canonical_set", False))
                    or bool(getattr(self, "clasp_fixed_null_set", False))
                )
                and not (
                    self.clasp_bon_aligned
                    and bool(getattr(self, "clasp_frontier_cdf", False))
                )
                and float(getattr(self, "clasp_component_dual_recovery_scale", 1.0)) > 0.0
                and dual_aware_api_available
            ),
            "component_dual_recovery_api_available": float(dual_aware_api_available),
            "frontier_cdf_uses_max_violation_gap": float(
                self.clasp_enabled
                and (
                    bool(getattr(self, "clasp_canonical_set", False))
                    or bool(getattr(self, "clasp_fixed_null_set", False))
                    or (
                        self.clasp_bon_aligned
                        and bool(getattr(self, "clasp_frontier_cdf", False))
                    )
                )
            ),
            "mean_penalty_weighted_gap": float(penalty_weighted_gap_scores.mean().item()),
            "effective_penalty_pressure": float(
                penalty_weighted_gap_scores.abs().mean().item()
                / (gathered_primal_scores.abs().mean().item() + 1e-6)
            ),
            "penalty_scale": float(current_penalty_scale),
            "positive_constraint_penalty": float(
                self.positive_constraint_penalty or self.separate_z
            ),
            "mean_violation": float(gathered_violation_scores.mean().item()),
            "candidate_feasibility_rate": float(gathered_feasibility_flags.mean().item()),
            "mean_constraint": float(gathered_constraint_scores.mean().item()),
            "mean_reward_advantage": float(reward_advantages_global.mean().item()),
            "std_reward_advantage": float(reward_advantages_global.std(unbiased=False).item()),
            "mean_combined_advantage": float(combined_advantages_global.mean().item()),
            **lexicographic_stats,
            **clasp_stats,
        }
        return (
            reward_advantages_global,
            combined_advantages_global,
            penalty_weighted_gap_scores,
            gathered_raw_positive_gap_scores,
            gathered_feasibility_flags,
            local_len,
            stats,
        )

    def _append_local_per_task_metrics(self, mode: str, components: dict[str, list[float]]) -> None:
        task_types = components.get("task_types")
        if not isinstance(task_types, list):
            return
        if len(task_types) != len(components.get("primal_scores", [])):
            return

        for task_name in sorted({str(task) for task in task_types}):
            indices = [idx for idx, task in enumerate(task_types) if str(task) == task_name]
            if not indices:
                continue

            def mean_for(key: str) -> float | None:
                values = components.get(key)
                if not isinstance(values, list) or len(values) <= max(indices):
                    return None
                return float(sum(float(values[idx]) for idx in indices) / len(indices))

            for metric_key, source_key in (
                ("mean_primal_reward", "primal_scores"),
                ("mean_objective_quality", "objective_scores"),
                ("mean_structure_reward", "structure_scores"),
                ("mean_weighted_gap", "weighted_gap_scores"),
                ("mean_positive_weighted_gap", "positive_weighted_gap_scores"),
                ("mean_raw_positive_gap", "raw_positive_gap_scores"),
                ("mean_dual_aware_recovery_gap", "dual_aware_recovery_gap_scores"),
                ("mean_constraint", "constraint_scores"),
                ("mean_violation", "constraint_violation_scores"),
                ("candidate_feasibility_rate", "feasibility_flags"),
            ):
                value = mean_for(source_key)
                if value is not None:
                    self._append_metric(mode, f"constrained_grpo/task/{task_name}/{metric_key}", value)

    def _extract_task_types(self, inputs) -> list[str] | None:
        raw_task_types = None
        if isinstance(inputs, dict):
            raw_task_types = inputs.get("task_type") or inputs.get("task_types")
        elif isinstance(inputs, list):
            values = []
            for item in inputs:
                if not isinstance(item, dict):
                    return None
                if "task_type" in item:
                    values.append(item["task_type"])
                elif "task_types" in item:
                    values.append(item["task_types"])
                else:
                    return None
            raw_task_types = values
        if raw_task_types is None:
            return None
        if isinstance(raw_task_types, torch.Tensor):
            raw_task_types = raw_task_types.detach().cpu().tolist()
        if not isinstance(raw_task_types, list):
            return None
        return [str(task) for task in raw_task_types]

    @staticmethod
    def _extract_batch_values(inputs, *keys: str) -> list[str | None] | None:
        if isinstance(inputs, dict):
            raw_values = None
            for key in keys:
                if key in inputs:
                    raw_values = inputs.get(key)
                    break
            if raw_values is None:
                return None
            if isinstance(raw_values, torch.Tensor):
                raw_values = raw_values.detach().cpu().tolist()
            if isinstance(raw_values, tuple):
                raw_values = list(raw_values)
            if not isinstance(raw_values, list):
                return None
            return [None if value is None else str(value) for value in raw_values]

        if isinstance(inputs, list):
            values: list[str | None] = []
            for item in inputs:
                if not isinstance(item, dict):
                    return None
                value = None
                for key in keys:
                    if key in item:
                        value = item.get(key)
                        break
                values.append(None if value is None else str(value))
            return values

        return None

    def _reference_imitation_enabled(self) -> bool:
        return (
            self.clasp_enabled
            and self.clasp_reference_imitation_loss_weight > 0.0
            and self.clasp_reference_imitation_max_tokens > 0
        )

    def _reference_contrastive_enabled(self) -> bool:
        return (
            self.clasp_enabled
            and self.clasp_reference_contrastive_loss_weight > 0.0
            and self.clasp_reference_contrastive_max_tokens > 0
        )

    def _reference_task_allowed(self, task_name: str | None) -> bool:
        if self.clasp_reference_imitation_tasks is None:
            return True
        if task_name is None:
            return False
        return str(task_name).lower() in self.clasp_reference_imitation_tasks

    def _reference_contrastive_task_allowed(self, task_name: str | None) -> bool:
        if self.clasp_reference_contrastive_tasks is None:
            return True
        if task_name is None:
            return False
        return str(task_name).lower() in self.clasp_reference_contrastive_tasks

    def _tokenize_completion_text(self, text: str, max_tokens: int) -> list[int]:
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            return []
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            return []
        try:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
        except TypeError:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
            token_ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._sanitize_token_ids([int(token_id) for token_id in token_ids[:max_tokens]])

    def _model_token_vocab_size(self) -> int | None:
        candidates: list[int] = []
        model = getattr(self, "model", None)
        base_model = getattr(model, "module", model)
        for candidate_model in (model, base_model):
            if candidate_model is None:
                continue
            get_embeddings = getattr(candidate_model, "get_input_embeddings", None)
            if callable(get_embeddings):
                try:
                    embeddings = get_embeddings()
                    num_embeddings = int(getattr(embeddings, "num_embeddings", 0) or 0)
                    if num_embeddings > 0:
                        candidates.append(num_embeddings)
                except Exception:
                    pass
            config = getattr(candidate_model, "config", None)
            vocab_size = int(getattr(config, "vocab_size", 0) or 0) if config is not None else 0
            if vocab_size > 0:
                candidates.append(vocab_size)
        return min(candidates) if candidates else None

    def _sanitize_token_ids(self, token_ids: list[int]) -> list[int]:
        vocab_size = self._model_token_vocab_size()
        if vocab_size is None:
            return [int(token_id) for token_id in token_ids if int(token_id) >= 0]
        return [
            int(token_id)
            for token_id in token_ids
            if 0 <= int(token_id) < int(vocab_size)
        ]

    def _safe_pad_token_id(self) -> int:
        vocab_size = self._model_token_vocab_size()
        candidates = [
            getattr(self.processing_class, "pad_token_id", None),
            getattr(self.processing_class, "eos_token_id", None),
            0,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            token_id = int(candidate)
            if token_id < 0:
                continue
            if vocab_size is None or token_id < int(vocab_size):
                return token_id
        return 0

    def _tokenize_reference_text(self, text: str) -> list[int]:
        return self._tokenize_completion_text(
            text,
            int(self.clasp_reference_imitation_max_tokens),
        )

    def _tokenize_reference_contrastive_text(self, text: str) -> list[int]:
        return self._tokenize_completion_text(
            text,
            int(self.clasp_reference_contrastive_max_tokens),
        )

    def _self_replay_enabled(self) -> bool:
        return (
            self.clasp_enabled
            and self.clasp_self_replay_enabled
            and self.clasp_self_replay_max_tokens > 0
            and self.clasp_self_replay_max_keys > 0
            and self.clasp_self_replay_max_per_key > 0
        )

    def _self_replay_loss_enabled(self) -> bool:
        return (
            self._self_replay_enabled()
            and (
                self.clasp_self_replay_imitation_loss_weight > 0.0
                or self.clasp_self_replay_contrastive_loss_weight > 0.0
            )
        )

    @staticmethod
    def _self_replay_tuple_tree(value):
        if isinstance(value, list):
            return tuple(
                PrimalDualGRPOTrainer._self_replay_tuple_tree(item)
                for item in value
            )
        if isinstance(value, tuple):
            return tuple(
                PrimalDualGRPOTrainer._self_replay_tuple_tree(item)
                for item in value
            )
        return value

    def _self_replay_runtime_state(self) -> dict:
        return {
            "schema_version": 1,
            "self_replay": {
                "schema_version": 1,
                "enabled": True,
                "max_tokens": int(self.clasp_self_replay_max_tokens),
                "max_keys": int(self.clasp_self_replay_max_keys),
                "max_per_key": int(self.clasp_self_replay_max_per_key),
                "buffer": [
                    {
                        "key": str(key),
                        "entries": [
                            {
                                "text": str(entry["text"]),
                                "score": float(entry["score"]),
                            }
                            for entry in entries
                        ],
                    }
                    for key, entries in self._clasp_self_replay_buffer.items()
                ],
                "rng_state": self._clasp_self_replay_rng.getstate(),
            },
        }

    def _restore_self_replay_runtime_state(self, runtime_state: dict) -> None:
        if not isinstance(runtime_state, dict) or runtime_state.get("schema_version") != 1:
            raise RuntimeError(
                "Incompatible CLASP trainer-state schema while restoring self replay; "
                "expected schema_version=1."
            )
        replay_state = runtime_state.get("self_replay")
        if not isinstance(replay_state, dict) or replay_state.get("schema_version") != 1:
            raise RuntimeError(
                "Checkpoint is missing a compatible versioned CLASP self-replay state."
            )
        if replay_state.get("enabled") is not True:
            raise RuntimeError(
                "Checkpoint self-replay state is not marked enabled; refusing to resume "
                "with CLASP self replay enabled."
            )
        expected_config = {
            "max_tokens": int(self.clasp_self_replay_max_tokens),
            "max_keys": int(self.clasp_self_replay_max_keys),
            "max_per_key": int(self.clasp_self_replay_max_per_key),
        }
        incompatible = {
            name: (replay_state.get(name), expected)
            for name, expected in expected_config.items()
            if replay_state.get(name) != expected
        }
        if incompatible:
            raise RuntimeError(
                "CLASP self-replay checkpoint limits do not match the active run: "
                f"{incompatible!r}. Start a fresh run or use the checkpoint's replay limits."
            )

        raw_buffer = replay_state.get("buffer")
        if not isinstance(raw_buffer, list):
            raise RuntimeError("CLASP self-replay checkpoint buffer must be a list")
        if len(raw_buffer) > int(self.clasp_self_replay_max_keys):
            raise RuntimeError("CLASP self-replay checkpoint exceeds max_keys")

        restored: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
        for raw_bucket in raw_buffer:
            if not isinstance(raw_bucket, dict):
                raise RuntimeError("CLASP self-replay checkpoint contains an invalid bucket")
            key = raw_bucket.get("key")
            raw_entries = raw_bucket.get("entries")
            if not isinstance(key, str) or not key or key in restored:
                raise RuntimeError("CLASP self-replay checkpoint contains an invalid/duplicate key")
            if not isinstance(raw_entries, list):
                raise RuntimeError("CLASP self-replay checkpoint entries must be a list")
            if len(raw_entries) > int(self.clasp_self_replay_max_per_key):
                raise RuntimeError("CLASP self-replay checkpoint exceeds max_per_key")
            entries: list[dict[str, object]] = []
            seen_texts: set[str] = set()
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    raise RuntimeError("CLASP self-replay checkpoint contains an invalid entry")
                text = raw_entry.get("text")
                try:
                    score = float(raw_entry.get("score"))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "CLASP self-replay checkpoint contains a non-numeric score"
                    ) from exc
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or text in seen_texts
                    or not math.isfinite(score)
                ):
                    raise RuntimeError(
                        "CLASP self-replay checkpoint contains an invalid/duplicate entry"
                    )
                seen_texts.add(text)
                entries.append({"text": text, "score": score})
            restored[key] = entries

        raw_rng_state = replay_state.get("rng_state")
        if raw_rng_state is None:
            raise RuntimeError("CLASP self-replay checkpoint is missing RNG state")
        try:
            self._clasp_self_replay_rng.setstate(
                self._self_replay_tuple_tree(raw_rng_state)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("CLASP self-replay checkpoint RNG state is invalid") from exc
        self._clasp_self_replay_buffer = restored

    def _sync_self_replay_state_to_manager(self) -> None:
        setter = getattr(
            self.primal_dual_reward_manager,
            "set_clasp_trainer_state",
            None,
        )
        if callable(setter):
            setter(self._self_replay_runtime_state())

    def _ensure_self_replay_state_loaded(self) -> None:
        if not self._self_replay_enabled():
            return
        status_getter = getattr(
            self.primal_dual_reward_manager,
            "get_clasp_resume_state_status",
            None,
        )
        if callable(status_getter):
            resume_requested, checkpoint_has_state, load_serial = status_getter()
        else:
            resume_requested, checkpoint_has_state, load_serial = False, False, 0
        if (
            bool(getattr(self, "_clasp_self_replay_state_loaded", False))
            and int(getattr(self, "_clasp_self_replay_resume_load_serial", -1))
            == int(load_serial)
        ):
            return

        state_getter = getattr(
            self.primal_dual_reward_manager,
            "get_clasp_trainer_state",
            None,
        )
        runtime_state = state_getter() if callable(state_getter) else None
        if resume_requested and (not checkpoint_has_state or runtime_state is None):
            raise RuntimeError(
                "Resuming with CLASP self replay enabled requires the versioned "
                "__clasp_trainer__ state in multitask_primal_dual_state.json; refusing "
                "to silently reset the replay buffer."
            )
        if runtime_state is not None:
            self._restore_self_replay_runtime_state(runtime_state)
        else:
            self._clasp_self_replay_buffer = OrderedDict()
            self._sync_self_replay_state_to_manager()
        self._clasp_self_replay_state_loaded = True
        self._clasp_self_replay_resume_load_serial = int(load_serial)

    @staticmethod
    def _gather_self_replay_records(
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                gathered: list[list[dict[str, object]] | None] = [
                    None for _ in range(dist.get_world_size())
                ]
                dist.all_gather_object(gathered, records)
                return [
                    record
                    for rank_records in gathered
                    for record in (rank_records or [])
                ]
        except Exception as exc:
            try:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
            except ValueError:
                world_size = 1
            if world_size > 1:
                raise RuntimeError(
                    "Distributed CLASP self-replay record synchronization failed; "
                    "refusing rank-divergent replay state."
                ) from exc
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        if world_size > 1:
            raise RuntimeError(
                "CLASP self replay is enabled in a multi-process run, but "
                "torch.distributed is not initialized."
            )
        return list(records)

    def _self_replay_tokenize_text(self, text: str) -> list[int]:
        return self._tokenize_completion_text(
            text,
            int(self.clasp_self_replay_max_tokens),
        )

    @staticmethod
    def _self_replay_key(task_name: str | None, instance_value, input_value, prompt_value) -> str | None:
        task = str(task_name or "").strip().lower()
        if not task:
            return None
        def is_empty(value) -> bool:
            return value is None or str(value) == ""

        source = instance_value if not is_empty(instance_value) else input_value
        if is_empty(source):
            source = prompt_value
        if is_empty(source):
            return None
        payload = str(source)
        digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
        return f"{task}:{digest}"

    def _self_replay_group_keys(
        self,
        inputs,
        task_types: list[str] | None,
        batch_size: int,
        num_groups: int,
    ) -> list[str | None]:
        if task_types is not None and len(task_types) == batch_size:
            group_tasks = task_types[:: self.num_generations]
        elif task_types is not None and len(task_types) == num_groups:
            group_tasks = task_types
        else:
            group_tasks = [None] * num_groups

        instance_values = self._normalize_group_values(
            self._extract_batch_values(inputs, "instance_json", "instance"),
            num_groups,
            batch_size,
            self.num_generations,
        )
        input_values = self._normalize_group_values(
            self._extract_batch_values(inputs, "input"),
            num_groups,
            batch_size,
            self.num_generations,
        )
        prompt_values = self._normalize_group_values(
            self._extract_batch_values(inputs, "prompt", "prompts"),
            num_groups,
            batch_size,
            self.num_generations,
        )

        return [
            self._self_replay_key(task, instance_value, input_value, prompt_value)
            for task, instance_value, input_value, prompt_value in zip(
                group_tasks,
                instance_values,
                input_values,
                prompt_values,
            )
        ]

    def _self_replay_total_entries(self) -> int:
        return sum(len(entries) for entries in self._clasp_self_replay_buffer.values())

    def _self_replay_pick(
        self,
        key: str | None,
        *,
        draw: float | None = None,
    ) -> dict[str, object] | None:
        if key is None:
            return None
        entries = self._clasp_self_replay_buffer.get(key)
        if not entries:
            return None
        if draw is None:
            return dict(self._clasp_self_replay_rng.choice(entries))
        index = min(int(max(0.0, min(float(draw), 1.0)) * len(entries)), len(entries) - 1)
        return dict(entries[index])

    def _tokens_to_padded_tensor_pair(
        self,
        tokenized_rows: list[list[int]],
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sanitized_rows = [self._sanitize_token_ids(tokens) for tokens in tokenized_rows]
        max_len = max([len(tokens) for tokens in sanitized_rows] + [1])
        pad_id = self._safe_pad_token_id()
        ids = torch.full(
            (len(sanitized_rows), max_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros(
            (len(sanitized_rows), max_len),
            dtype=torch.long,
            device=device,
        )
        for row_idx, token_ids in enumerate(sanitized_rows):
            if not token_ids:
                continue
            row_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            ids[row_idx, : row_ids.numel()] = row_ids
            mask[row_idx, : row_ids.numel()] = 1
        return ids, mask

    @staticmethod
    def _repeat_group_tensor_for_generation_batch(
        tensor: torch.Tensor,
        num_generations: int,
        batch_size: int,
    ) -> torch.Tensor:
        if tensor.size(0) == batch_size:
            return tensor
        repeated = tensor.repeat_interleave(int(num_generations), dim=0)
        if int(repeated.size(0)) != int(batch_size):
            raise RuntimeError(
                "CLASP self-replay tensor row mismatch after group expansion. "
                f"rows={int(tensor.size(0))}, num_generations={int(num_generations)}, "
                f"expanded_rows={int(repeated.size(0))}, batch_size={int(batch_size)}."
            )
        return repeated

    @staticmethod
    def _validate_trl_generation_batch_rows(output: dict) -> None:
        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids) or completion_ids.dim() == 0:
            return
        batch_size = int(completion_ids.size(0))
        mismatches: list[str] = []
        for key, value in output.items():
            if not torch.is_tensor(value) or value.dim() == 0:
                continue
            rows = int(value.size(0))
            if rows != batch_size:
                mismatches.append(f"{key}: rows={rows}, expected={batch_size}")
        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(
                "CLASP produced tensors with non-batch leading dimensions in TRL's "
                "generation_batch. TRL shuffles every tensor with a per-completion "
                "permutation, so these mismatches would otherwise become a CUDA "
                f"gather/index assert. {details}"
            )

    def _attach_clasp_group_metadata(self, output: dict, components: dict, local_len: int) -> None:
        """Attach row-level prompt-group identity before TRL shuffles the batch."""

        if not self.clasp_enabled:
            return
        num_processes = int(getattr(getattr(self, "accelerator", None), "num_processes", 1))
        global_len = int(local_len) * num_processes
        if local_len <= 0 or global_len % int(self.num_generations) != 0:
            raise RuntimeError(
                "CLASP global generation batch must contain complete prompt groups before shuffle: "
                f"local_len={local_len}, num_processes={num_processes}, "
                f"global_len={global_len}, num_generations={self.num_generations}"
            )
        completion_ids = output.get("completion_ids")
        if not torch.is_tensor(completion_ids) or int(completion_ids.size(0)) != int(local_len):
            raise RuntimeError("CLASP cannot attach group IDs without row-aligned completion_ids")
        device = completion_ids.device
        process_index = int(getattr(getattr(self, "accelerator", None), "process_index", 0))
        global_indices = process_index * int(local_len) + torch.arange(
            int(local_len),
            dtype=torch.long,
            device=device,
        )
        local_group_indices = torch.div(
            global_indices,
            int(self.num_generations),
            rounding_mode="floor",
        )

        prompt_ids = output.get("prompt_ids")
        if torch.is_tensor(prompt_ids):
            # A K-sample prompt group may span ranks. Validate every same-group
            # subset visible locally; global task consistency is checked on the
            # gathered pre-shuffle rows in _bon_group_task_names.
            for local_group_id in torch.unique(local_group_indices).tolist():
                rows = torch.nonzero(
                    local_group_indices == int(local_group_id),
                    as_tuple=False,
                ).reshape(-1)
                if rows.numel() <= 1:
                    continue
                group_prompts = prompt_ids[rows]
                if not torch.equal(group_prompts, group_prompts[:1].expand_as(group_prompts)):
                    raise RuntimeError(
                        "CLASP found different prompt token IDs within a rank-local subset of "
                        f"global group {local_group_id}."
                    )

        task_types = components.get("task_types") if isinstance(components, dict) else None
        if isinstance(task_types, list) and len(task_types) == int(local_len):
            normalized_tasks = [str(task).strip().lower() for task in task_types]
            for local_group_id in torch.unique(local_group_indices).tolist():
                rows = torch.nonzero(
                    local_group_indices == int(local_group_id),
                    as_tuple=False,
                ).reshape(-1).tolist()
                names = {normalized_tasks[row] for row in rows if normalized_tasks[row]}
                if len(names) != 1:
                    raise RuntimeError(
                        "CLASP found mixed task labels within a rank-local subset of global "
                        f"group {local_group_id}: {sorted(names)}"
                    )

        serial = int(self._clasp_generation_serial)
        group_ids = torch.div(
            global_indices,
            int(self.num_generations),
            rounding_mode="floor",
        ).add(serial * 1_000_000_000)
        group_positions = torch.remainder(global_indices, int(self.num_generations))
        output["clasp_group_ids"] = group_ids
        output["clasp_group_positions"] = group_positions
        self._clasp_generation_serial += 1

    def _group_auxiliary_permutation(
        self,
        inputs: dict,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, str]:
        permutation, reason = _complete_group_permutation(
            inputs.get("clasp_group_ids"),
            inputs.get("clasp_group_positions"),
            batch_size=int(batch_size),
            num_generations=int(self.num_generations),
        )
        if permutation is not None:
            return permutation, reason
        if self.clasp_aux_grouping_strict:
            raise RuntimeError(
                "CLASP group auxiliary loss cannot prove prompt-group integrity after TRL "
                f"shuffle/split: {reason}. Disable pairwise/elite auxiliaries or use complete "
                "group-aligned optimization microbatches."
            )
        if not self._clasp_aux_grouping_warning_emitted:
            process_index = int(getattr(getattr(self, "accelerator", None), "process_index", 0))
            if process_index == 0:
                print(
                    "[CLASP][GroupSafety] Disabling pairwise/elite auxiliary loss for "
                    f"incomplete shuffled microbatch ({reason}).",
                    flush=True,
                )
            self._clasp_aux_grouping_warning_emitted = True
        return None, reason

    def _validate_group_auxiliary_schedule(self) -> None:
        """Reject schedules that cannot present intact K-groups to aux losses.

        TRL shuffles each local generation buffer before splitting it into
        optimization microbatches. Therefore a group auxiliary is supported
        only when one optimization microbatch is itself the whole generation
        buffer (``steps_per_generation == 1``) and its local size is a multiple
        of K. The normal CLASP policy loss does not have this restriction
        because its advantages are computed before that shuffle.
        """

        auxiliary_weights = {
            "CLASP_PAIRWISE_LOSS_WEIGHT": float(self.clasp_pairwise_loss_weight),
            "CLASP_ELITE_IMITATION_LOSS_WEIGHT": float(
                self.clasp_elite_imitation_loss_weight
            ),
            "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT": float(
                self.clasp_multi_elite_imitation_loss_weight
            ),
        }
        enabled = sorted(name for name, value in auxiliary_weights.items() if value > 0.0)
        if not enabled:
            return

        args = getattr(self, "args", None)
        local_batch_size = int(
            getattr(args, "per_device_train_batch_size", 0) or 0
        )
        steps_per_generation = int(
            getattr(args, "steps_per_generation", 0) or 0
        )
        schedule_supported = (
            local_batch_size > 0
            and local_batch_size % int(self.num_generations) == 0
            and steps_per_generation == 1
        )
        if schedule_supported:
            return

        details = (
            f"local_batch={local_batch_size}, K={int(self.num_generations)}, "
            f"steps_per_generation={steps_per_generation}"
        )
        message = (
            "Enabled CLASP group auxiliary losses require complete prompt groups in every "
            f"local optimization microbatch ({details}). TRL shuffles the generation buffer "
            "before splitting it, so use --steps_per_generation 1 and a "
            "--per_device_train_batch_size that is a positive multiple of K, or set these "
            f"auxiliary weights to zero: {', '.join(enabled)}. The dense_rank CLASP policy "
            "estimator does not require group-local optimization batches."
        )
        if self.clasp_aux_grouping_strict:
            raise ValueError(message)
        process_index = int(getattr(getattr(self, "accelerator", None), "process_index", 0))
        if process_index == 0:
            print(f"[CLASP][GroupSafety] {message} Enabled auxiliaries will be skipped.", flush=True)

    def _attach_self_replay_tensors(
        self,
        inputs,
        output: dict,
        components: dict,
        task_types: list[str] | None,
    ) -> None:
        if not (self.model.training and self._self_replay_loss_enabled()):
            return
        self._ensure_self_replay_state_loaded()

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return
        batch_size = int(completion_ids.size(0))
        if self.num_generations <= 0 or batch_size % self.num_generations != 0:
            return
        if self._self_replay_total_entries() < int(self.clasp_self_replay_min_buffer_size):
            return

        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
            return

        num_groups = batch_size // self.num_generations
        group_keys = self._self_replay_group_keys(inputs, task_types, batch_size, num_groups)
        raw_gaps = components.get("raw_positive_gap_scores")
        flags = components.get("feasibility_flags")
        if not (isinstance(raw_gaps, list) and isinstance(flags, list) and len(raw_gaps) == batch_size):
            raw_gaps = [0.0] * batch_size
            flags = [0.0] * batch_size

        try:
            completion_texts = tokenizer.batch_decode(
                completion_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except TypeError:
            completion_texts = tokenizer.batch_decode(completion_ids.detach().cpu().tolist())

        positive_rows: list[list[int]] = []
        negative_rows: list[list[int]] = []
        selected_count = 0
        for group_idx, key in enumerate(group_keys):
            positive_tokens: list[int] = []
            negative_tokens: list[int] = []
            sample_draw = self._clasp_self_replay_rng.random()
            choice_draw = self._clasp_self_replay_rng.random()
            if key is not None:
                self._clasp_self_replay_lookup_attempts = int(
                    getattr(self, "_clasp_self_replay_lookup_attempts", 0)
                ) + 1
                if self._clasp_self_replay_buffer.get(key):
                    self._clasp_self_replay_lookup_hits = int(
                        getattr(self, "_clasp_self_replay_lookup_hits", 0)
                    ) + 1
            if (
                key is not None
                and sample_draw <= float(self.clasp_self_replay_sample_rate)
            ):
                entry = self._self_replay_pick(key, draw=choice_draw)
                if entry is not None:
                    positive_tokens = self._self_replay_tokenize_text(str(entry.get("text", "")))

            if positive_tokens:
                selected_count += 1
                if self.clasp_self_replay_contrastive_loss_weight > 0.0:
                    start = group_idx * self.num_generations
                    end = start + self.num_generations
                    candidate_indices = [
                        idx
                        for idx in range(start, end)
                        if idx < len(flags) and float(flags[idx]) <= 0.5
                    ]
                    if candidate_indices:
                        worst_idx = max(
                            candidate_indices,
                            key=lambda idx: float(raw_gaps[idx]) if idx < len(raw_gaps) else 0.0,
                        )
                        negative_tokens = self._self_replay_tokenize_text(str(completion_texts[worst_idx]))

            positive_rows.append(positive_tokens)
            negative_rows.append(negative_tokens)

        if selected_count <= 0:
            return

        positive_ids, positive_mask = self._tokens_to_padded_tensor_pair(
            positive_rows,
            completion_ids.device,
        )
        positive_ids = self._repeat_group_tensor_for_generation_batch(
            positive_ids,
            self.num_generations,
            batch_size,
        )
        positive_mask = self._repeat_group_tensor_for_generation_batch(
            positive_mask,
            self.num_generations,
            batch_size,
        )
        output["self_replay_completion_ids"] = positive_ids
        output["self_replay_completion_mask"] = positive_mask
        if self.clasp_self_replay_contrastive_loss_weight > 0.0:
            negative_ids, negative_mask = self._tokens_to_padded_tensor_pair(
                negative_rows,
                completion_ids.device,
            )
            negative_ids = self._repeat_group_tensor_for_generation_batch(
                negative_ids,
                self.num_generations,
                batch_size,
            )
            negative_mask = self._repeat_group_tensor_for_generation_batch(
                negative_mask,
                self.num_generations,
                batch_size,
            )
            output["self_replay_negative_ids"] = negative_ids
            output["self_replay_negative_mask"] = negative_mask

    def _observe_self_replay(
        self,
        inputs,
        output: dict,
        components: dict,
        task_types: list[str] | None,
    ) -> dict[str, float]:
        lookup_attempts = int(getattr(self, "_clasp_self_replay_lookup_attempts", 0))
        lookup_hits = int(getattr(self, "_clasp_self_replay_lookup_hits", 0))
        self._clasp_self_replay_lookup_attempts = 0
        self._clasp_self_replay_lookup_hits = 0
        stats = {
            "clasp/self_replay_enabled": float(self._self_replay_enabled()),
            "clasp/self_replay_buffer_keys": float(len(self._clasp_self_replay_buffer)),
            "clasp/self_replay_buffer_entries": float(self._self_replay_total_entries()),
            "clasp/self_replay_added": 0.0,
            "clasp/self_replay_lookup_attempts": float(lookup_attempts),
            "clasp/self_replay_lookup_hits": float(lookup_hits),
            "clasp/self_replay_lookup_hit_rate": float(
                lookup_hits / max(lookup_attempts, 1)
            ),
            "clasp/self_replay_insert_candidates": 0.0,
            "clasp/self_replay_insert_rate": 0.0,
            "clasp/self_replay_evicted": 0.0,
            "clasp/self_replay_eviction_rate": 0.0,
        }
        if not (self.model.training and self._self_replay_enabled()):
            return stats
        self._ensure_self_replay_state_loaded()
        stats["clasp/self_replay_buffer_keys"] = float(
            len(self._clasp_self_replay_buffer)
        )
        stats["clasp/self_replay_buffer_entries"] = float(
            self._self_replay_total_entries()
        )

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return stats
        batch_size = int(completion_ids.size(0))
        if self.num_generations <= 0 or batch_size % self.num_generations != 0:
            return stats
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
            return stats

        flags = components.get("feasibility_flags")
        primal_scores = components.get("primal_scores")
        if not (isinstance(flags, list) and len(flags) == batch_size):
            return stats
        if not (isinstance(primal_scores, list) and len(primal_scores) == batch_size):
            primal_scores = [0.0] * batch_size

        try:
            completion_texts = tokenizer.batch_decode(
                completion_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except TypeError:
            completion_texts = tokenizer.batch_decode(completion_ids.detach().cpu().tolist())

        num_groups = batch_size // self.num_generations
        group_keys = self._self_replay_group_keys(inputs, task_types, batch_size, num_groups)
        local_records: list[dict[str, object]] = []
        for sample_idx, (flag, text) in enumerate(zip(flags, completion_texts)):
            if float(flag) <= 0.5:
                continue
            text = str(text).strip()
            if not text:
                continue
            key = group_keys[sample_idx // self.num_generations]
            if key is None:
                continue
            score = float(primal_scores[sample_idx])
            if not math.isfinite(score):
                raise RuntimeError("CLASP self replay received a non-finite primal score")
            local_records.append({"key": key, "text": text, "score": score})

        records = self._gather_self_replay_records(local_records)
        inserted = 0
        evicted = 0
        for record in records:
            key = str(record["key"])
            text = str(record["text"])
            score = float(record["score"])
            entries = self._clasp_self_replay_buffer.get(key)
            if entries is None:
                entries = []
                self._clasp_self_replay_buffer[key] = entries
            self._clasp_self_replay_buffer.move_to_end(key)
            if any(str(entry.get("text", "")) == text for entry in entries):
                continue
            entries.append(
                {
                    "text": text,
                    "score": score,
                }
            )
            entries.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)
            overflow = max(0, len(entries) - int(self.clasp_self_replay_max_per_key))
            if overflow:
                del entries[int(self.clasp_self_replay_max_per_key) :]
                evicted += overflow
            inserted += 1
            while len(self._clasp_self_replay_buffer) > int(self.clasp_self_replay_max_keys):
                _, removed_entries = self._clasp_self_replay_buffer.popitem(last=False)
                evicted += len(removed_entries)

        self._sync_self_replay_state_to_manager()
        stats["clasp/self_replay_buffer_keys"] = float(len(self._clasp_self_replay_buffer))
        stats["clasp/self_replay_buffer_entries"] = float(self._self_replay_total_entries())
        stats["clasp/self_replay_added"] = float(inserted)
        stats["clasp/self_replay_insert_candidates"] = float(len(records))
        stats["clasp/self_replay_insert_rate"] = float(
            inserted / max(len(records), 1)
        )
        stats["clasp/self_replay_evicted"] = float(evicted)
        stats["clasp/self_replay_eviction_rate"] = float(
            evicted / max(inserted, 1)
        )
        return stats

    def _attach_reference_imitation_tensors(
        self,
        inputs,
        output: dict,
        task_types: list[str] | None,
    ) -> None:
        if not self._reference_imitation_enabled():
            return

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return
        if self.num_generations <= 0 or completion_ids.size(0) % self.num_generations != 0:
            return

        num_groups = completion_ids.size(0) // self.num_generations
        reference_answers = self._extract_batch_values(
            inputs,
            "reference_answer",
            "reference_imitation_output",
            "imitation_output",
        )
        fallback_answers = self._extract_batch_values(inputs, "answer", "output")
        reference_answers = self._normalize_group_values(
            reference_answers,
            num_groups,
            int(completion_ids.size(0)),
            self.num_generations,
        )
        fallback_answers = self._normalize_group_values(
            fallback_answers,
            num_groups,
            int(completion_ids.size(0)),
            self.num_generations,
        )
        answers = [
            ref if ref not in {None, ""} else fallback
            for ref, fallback in zip(reference_answers, fallback_answers)
        ]

        if task_types is not None and len(task_types) == completion_ids.size(0):
            task_types = task_types[:: self.num_generations]
        if task_types is None or len(task_types) != num_groups:
            task_types = [None] * num_groups

        tokenized_answers: list[list[int]] = []
        for answer, task_name in zip(answers, task_types):
            if answer is None or not self._reference_task_allowed(task_name):
                tokenized_answers.append([])
                continue
            tokenized_answers.append(self._tokenize_reference_text(answer))

        max_len = max([len(tokens) for tokens in tokenized_answers] + [1])
        device = completion_ids.device
        pad_id = self._safe_pad_token_id()

        reference_ids = torch.full(
            (num_groups, max_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        reference_mask = torch.zeros(
            (num_groups, max_len),
            dtype=torch.long,
            device=device,
        )
        for row_idx, token_ids in enumerate(tokenized_answers):
            if not token_ids:
                continue
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            reference_ids[row_idx, : ids.numel()] = ids
            reference_mask[row_idx, : ids.numel()] = 1

        output["reference_completion_ids"] = reference_ids.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        output["reference_completion_mask"] = reference_mask.repeat_interleave(
            self.num_generations,
            dim=0,
        )

    def _attach_reference_contrastive_tensors(
        self,
        inputs,
        output: dict,
        task_types: list[str] | None,
    ) -> None:
        if not self._reference_contrastive_enabled():
            return

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return
        if self.num_generations <= 0 or completion_ids.size(0) % self.num_generations != 0:
            return

        num_groups = completion_ids.size(0) // self.num_generations
        positive_answers = self._extract_batch_values(
            inputs,
            "reference_answer",
            "reference_imitation_output",
            "imitation_output",
        )
        negative_answers = self._extract_batch_values(
            inputs,
            "negative_reference_answer",
            "negative_imitation_output",
        )
        positive_answers = self._normalize_group_values(
            positive_answers,
            num_groups,
            int(completion_ids.size(0)),
            self.num_generations,
        )
        negative_answers = self._normalize_group_values(
            negative_answers,
            num_groups,
            int(completion_ids.size(0)),
            self.num_generations,
        )

        if task_types is not None and len(task_types) == completion_ids.size(0):
            task_types = task_types[:: self.num_generations]
        if task_types is None or len(task_types) != num_groups:
            task_types = [None] * num_groups

        positive_tokenized: list[list[int]] = []
        negative_tokenized: list[list[int]] = []
        for positive, negative, task_name in zip(positive_answers, negative_answers, task_types):
            if (
                positive in {None, ""}
                or negative in {None, ""}
                or not self._reference_contrastive_task_allowed(task_name)
            ):
                positive_tokenized.append([])
                negative_tokenized.append([])
                continue
            positive_tokenized.append(self._tokenize_reference_contrastive_text(str(positive)))
            negative_tokenized.append(self._tokenize_reference_contrastive_text(str(negative)))

        max_len = max(
            [len(tokens) for tokens in positive_tokenized]
            + [len(tokens) for tokens in negative_tokenized]
            + [1]
        )
        device = completion_ids.device
        pad_id = self._safe_pad_token_id()

        positive_ids = torch.full(
            (num_groups, max_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        positive_mask = torch.zeros(
            (num_groups, max_len),
            dtype=torch.long,
            device=device,
        )
        negative_ids = torch.full(
            (num_groups, max_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        negative_mask = torch.zeros(
            (num_groups, max_len),
            dtype=torch.long,
            device=device,
        )
        for row_idx, token_ids in enumerate(positive_tokenized):
            if not token_ids:
                continue
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            positive_ids[row_idx, : ids.numel()] = ids
            positive_mask[row_idx, : ids.numel()] = 1
        for row_idx, token_ids in enumerate(negative_tokenized):
            if not token_ids:
                continue
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            negative_ids[row_idx, : ids.numel()] = ids
            negative_mask[row_idx, : ids.numel()] = 1

        output["reference_contrastive_positive_ids"] = positive_ids.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        output["reference_contrastive_positive_mask"] = positive_mask.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        output["reference_contrastive_negative_ids"] = negative_ids.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        output["reference_contrastive_negative_mask"] = negative_mask.repeat_interleave(
            self.num_generations,
            dim=0,
        )

    def _tsp_two_opt_distillation_enabled(self) -> bool:
        return (
            self.clasp_enabled
            and self.clasp_tsp_two_opt_distill_loss_weight > 0.0
            and self.clasp_tsp_two_opt_distill_max_tokens > 0
            and self.clasp_tsp_two_opt_distill_max_passes > 0
        )

    @staticmethod
    def _extract_batch_raw_values(inputs, *keys: str) -> list | None:
        if isinstance(inputs, dict):
            raw_values = None
            for key in keys:
                if key in inputs:
                    raw_values = inputs.get(key)
                    break
            if raw_values is None:
                return None
            if isinstance(raw_values, torch.Tensor):
                raw_values = raw_values.detach().cpu().tolist()
            if isinstance(raw_values, tuple):
                raw_values = list(raw_values)
            if isinstance(raw_values, list):
                return raw_values
            return None

        if isinstance(inputs, list):
            values: list = []
            for item in inputs:
                if not isinstance(item, dict):
                    return None
                value = None
                for key in keys:
                    if key in item:
                        value = item.get(key)
                        break
                values.append(value)
            return values

        return None

    @staticmethod
    def _normalize_group_values(
        values: list | None,
        num_groups: int,
        batch_size: int,
        num_generations: int,
    ) -> list:
        if values is None:
            return [None] * num_groups
        if len(values) == num_groups:
            return list(values)
        if num_generations > 0 and len(values) == batch_size:
            return [values[idx * num_generations] for idx in range(num_groups)]
        return [None] * num_groups

    @staticmethod
    def _maybe_json_load(value):
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value

    @staticmethod
    def _format_tsp_distillation_target(route: list[int], objective: float) -> str:
        route_text = ", ".join(str(int(node)) for node in route)
        if str(os.environ.get("CLASP_TSP_ROUTE_ONLY_OUTPUT", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return f"Route: [{route_text}]"
        return f"Route: [{route_text}], Objective: {objective:.6f}"

    def _attach_tsp_two_opt_distillation_tensors(
        self,
        inputs,
        output: dict,
        task_types: list[str] | None,
    ) -> None:
        if not self._tsp_two_opt_distillation_enabled():
            return

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return
        if self.num_generations <= 0 or completion_ids.size(0) % self.num_generations != 0:
            return

        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
            return

        batch_size = int(completion_ids.size(0))
        num_groups = batch_size // self.num_generations
        device = completion_ids.device

        if task_types is not None and len(task_types) == batch_size:
            group_task_types = task_types[:: self.num_generations]
        elif task_types is not None and len(task_types) == num_groups:
            group_task_types = task_types
        else:
            group_task_types = [None] * num_groups

        raw_inputs = self._extract_batch_raw_values(inputs, "input")
        raw_prompts = self._extract_batch_raw_values(inputs, "prompt", "prompts")
        raw_instances = self._extract_batch_raw_values(inputs, "instance", "instance_json")
        group_inputs = self._normalize_group_values(
            raw_inputs,
            num_groups,
            batch_size,
            self.num_generations,
        )
        group_prompts = self._normalize_group_values(
            raw_prompts,
            num_groups,
            batch_size,
            self.num_generations,
        )
        group_instances = self._normalize_group_values(
            raw_instances,
            num_groups,
            batch_size,
            self.num_generations,
        )

        try:
            completion_texts = tokenizer.batch_decode(
                completion_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except TypeError:
            completion_texts = tokenizer.batch_decode(completion_ids.detach().cpu().tolist())
        except Exception:
            return

        from clasp.rewards.co_tasks import (
            _as_text,
            _check_tsp_constraints,
            _parse_tsp_solution,
            _tsp_coords_from_instance,
            _tsp_two_opt_improved_route,
        )

        tokenized_targets: list[list[int]] = []
        gain_values: list[float] = []
        pass_values: list[float] = []
        min_gain = float(self.clasp_tsp_two_opt_distill_min_gain)
        for row_idx, completion_text in enumerate(completion_texts):
            group_idx = row_idx // self.num_generations
            task_name = group_task_types[group_idx] if group_idx < len(group_task_types) else None
            if task_name is not None and str(task_name).lower() != "tsp":
                tokenized_targets.append([])
                gain_values.append(0.0)
                pass_values.append(0.0)
                continue

            input_text = _as_text(group_inputs[group_idx]) if group_idx < len(group_inputs) else ""
            if not input_text and group_idx < len(group_prompts):
                input_text = _as_text(group_prompts[group_idx])
            instance = (
                self._maybe_json_load(group_instances[group_idx])
                if group_idx < len(group_instances)
                else None
            )
            coords = _tsp_coords_from_instance(instance, input_text)
            if not coords:
                tokenized_targets.append([])
                gain_values.append(0.0)
                pass_values.append(0.0)
                continue

            parsed = _parse_tsp_solution(str(completion_text))
            route = parsed["route"]
            _, _, feasible = _check_tsp_constraints(route, len(coords))
            if not feasible:
                tokenized_targets.append([])
                gain_values.append(0.0)
                pass_values.append(0.0)
                continue

            improved = _tsp_two_opt_improved_route(
                coords,
                route,
                max_passes=int(self.clasp_tsp_two_opt_distill_max_passes),
            )
            if improved is None:
                tokenized_targets.append([])
                gain_values.append(0.0)
                pass_values.append(0.0)
                continue

            improved_route, _, improved_total, gain_ratio, num_passes = improved
            if gain_ratio < min_gain:
                tokenized_targets.append([])
                gain_values.append(0.0)
                pass_values.append(0.0)
                continue

            target_text = self._format_tsp_distillation_target(improved_route, improved_total)
            token_ids = self._tokenize_completion_text(
                target_text,
                int(self.clasp_tsp_two_opt_distill_max_tokens),
            )
            tokenized_targets.append(token_ids)
            gain_values.append(float(gain_ratio))
            pass_values.append(float(num_passes))

        if len(tokenized_targets) != batch_size:
            return

        max_len = max([len(tokens) for tokens in tokenized_targets] + [1])
        pad_id = self._safe_pad_token_id()

        target_ids = torch.full(
            (batch_size, max_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        target_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )
        for row_idx, token_ids in enumerate(tokenized_targets):
            if not token_ids:
                continue
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            target_ids[row_idx, : ids.numel()] = ids
            target_mask[row_idx, : ids.numel()] = 1

        output["tsp_two_opt_distill_completion_ids"] = target_ids
        output["tsp_two_opt_distill_completion_mask"] = target_mask
        output["tsp_two_opt_distill_gains"] = torch.tensor(
            gain_values,
            dtype=torch.float32,
            device=device,
        )
        output["tsp_two_opt_distill_passes"] = torch.tensor(
            pass_values,
            dtype=torch.float32,
            device=device,
        )

    def _improvement_distillation_specs(self) -> list[dict]:
        """Enabled per-task improvement-operator distillation specs.

        Each spec turns a feasible model completion into a strictly better
        feasible target via a cheap classical operator (training-time only;
        inference stays raw). Spec lists are env-driven, so they are identical
        on every rank.
        """
        if getattr(self, "grpo_verified_improvement", False):
            extra_tasks = ("tsp", "op", "jssp") if getattr(self, "grpo_verified_train_reference_teacher", False) else ()
            return [
                {
                    "prefix": f"grpo_verified_{task}", "task": task,
                    "weight": self.grpo_verified_improvement_weight,
                    "max_tokens": self.grpo_verified_improvement_max_tokens,
                    "min_gain": 0.0, "verified_grpo": True,
                    **({"training_reference": True} if task in extra_tasks else {}),
                }
                for task in ("cvrp", "mis") + extra_tasks
            ]
        if not self.clasp_enabled:
            return []
        specs: list[dict] = []
        if (
            self.clasp_cvrp_local_search_distill_loss_weight > 0.0
            and self.clasp_cvrp_local_search_distill_max_tokens > 0
            and self.clasp_cvrp_local_search_distill_max_passes > 0
        ):
            specs.append(
                {
                    "prefix": "cvrp_local_search_distill",
                    "task": "cvrp",
                    "weight": float(self.clasp_cvrp_local_search_distill_loss_weight),
                    "max_tokens": int(self.clasp_cvrp_local_search_distill_max_tokens),
                    "min_gain": float(self.clasp_cvrp_local_search_distill_min_gain),
                    "improve": self._improve_cvrp_completion,
                }
            )
        if (
            self.clasp_jssp_active_schedule_distill_loss_weight > 0.0
            and self.clasp_jssp_active_schedule_distill_max_tokens > 0
            and self.clasp_jssp_active_schedule_distill_max_passes > 0
        ):
            specs.append(
                {
                    "prefix": "jssp_active_schedule_distill",
                    "task": "jssp",
                    "weight": float(self.clasp_jssp_active_schedule_distill_loss_weight),
                    "max_tokens": int(self.clasp_jssp_active_schedule_distill_max_tokens),
                    "min_gain": float(self.clasp_jssp_active_schedule_distill_min_gain),
                    "improve": self._improve_jssp_completion,
                }
            )
        if (
            self.clasp_mis_maximalize_distill_loss_weight > 0.0
            and self.clasp_mis_maximalize_distill_max_tokens > 0
        ):
            specs.append(
                {
                    "prefix": "mis_maximalize_distill",
                    "task": "mis",
                    "weight": float(self.clasp_mis_maximalize_distill_loss_weight),
                    "max_tokens": int(self.clasp_mis_maximalize_distill_max_tokens),
                    "min_gain": float(self.clasp_mis_maximalize_distill_min_gain),
                    "improve": self._improve_mis_completion,
                }
            )
        return specs

    def _verified_improvement_target(self, task, completion_text, instance, input_text):
        """Construct and verify an external teacher; never change the rollout reward."""
        from clasp.rewards.co_tasks import (
            _cvrp_meta,
            _format_cvrp_distillation_target,
            _format_mis_distillation_target,
            _mis_meta,
            _parse_cvrp_solution,
            _parse_mis_solution,
        )
        if task == "cvrp":
            from clasp.rewards.cvrp_verified_improvement import improve_cvrp_routes

            coords, demands, capacity = _cvrp_meta(instance, input_text)
            result = improve_cvrp_routes(
                _parse_cvrp_solution(completion_text)["routes"], coords, demands, capacity,
                max_passes=self.grpo_verified_improvement_max_passes,
            )
            if result is None:
                return None
            return (_format_cvrp_distillation_target(result.routes, result.objective),
                    result.gain, result.passes, result.repaired)
        if task == "mis":
            from clasp.rewards.verified_improvement import improve_mis_set

            num_nodes, edges = _mis_meta(instance, input_text)
            nodes = _parse_mis_solution(completion_text)["node_set"]
            # Malformed/empty output supplies graded-reward feedback, not an
            # unconditional solver-generated label unrelated to sampled decisions.
            if not nodes:
                return None
            result = improve_mis_set(
                nodes, num_nodes, edges,
                max_exchanges=self.grpo_verified_improvement_max_passes,
            )
            if result is None:
                return None
            gain = result.gain / max(result.original_size, 1) if result.source_feasible else 0.0
            return (_format_mis_distillation_target(result.node_set), gain,
                    result.passes, result.repaired)
        return None

    def _tokenize_verified_target(self, text, max_tokens):
        """Reject overlong/invalid targets instead of supervising a truncated solution."""
        tokenizer = self.processing_class
        # Fast tokenization mutates backend padding/truncation caches. Preserve
        # them so training-only teacher encoding cannot change the tokenizer
        # artifact written into otherwise comparable checkpoints.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        padding = backend.padding if backend is not None else None
        truncation = backend.truncation if backend is not None else None
        try:
            ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        finally:
            if backend is not None:
                if padding is None:
                    backend.no_padding()
                else:
                    backend.enable_padding(**padding)
                if truncation is None:
                    backend.no_truncation()
                else:
                    backend.enable_truncation(**truncation)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(int(eos))
        if not ids or len(ids) > max_tokens:
            return []
        if self._sanitize_token_ids(ids) != ids:
            return []
        return ids

    @staticmethod
    def _verified_completion_objective(task, text, instance, input_text):
        """Recompute a feasible raw/teacher objective without any reference label."""
        if task in ("tsp", "op", "jssp"):
            from clasp.rewards.verified_train_reference import canonical_verified_solution
            result = canonical_verified_solution(task, text, instance, input_text)
            return result[1] if result else None
        from clasp.rewards.co_tasks import (
            _check_cvrp_constraints,
            _check_mis_constraints,
            _cvrp_meta,
            _cvrp_total_distance,
            _mis_meta,
            _parse_cvrp_solution,
            _parse_mis_solution,
        )

        if task == "cvrp":
            coords, demands, capacity = _cvrp_meta(instance, input_text)
            routes = _parse_cvrp_solution(text)["routes"]
            if not _check_cvrp_constraints(routes, coords, demands, capacity)[2]:
                return None
            value = _cvrp_total_distance(coords, routes)
        elif task == "mis":
            num_nodes, edges = _mis_meta(instance, input_text)
            nodes = _parse_mis_solution(text)["node_set"]
            if not _check_mis_constraints(nodes, num_nodes, edges)[2]:
                return None
            value = len(nodes)
        else:
            raise ValueError(f"Unsupported verified teacher task: {task}")
        return value if value is not None and math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _verified_teacher_prompt_key(task, instance, prompt, input_text):
        if instance is None and not prompt and not input_text:
            raise ValueError("Verified teacher selection requires prompt or instance identity")
        payload = json.dumps(
            {"task": task, "instance": instance, "prompt": prompt or input_text},
            sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _gather_verified_teacher_rows(self, rows):
        """Gather metadata before TRL shuffles/splits the completion batch."""
        import torch.distributed as dist

        world_size = int(self.accelerator.num_processes)
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size() != world_size:
                raise RuntimeError("Verified teacher selection distributed world size mismatch")
            gathered = [None] * world_size
            dist.all_gather_object(gathered, rows)
            return [row for rank_rows in gathered for row in rank_rows]
        if world_size != 1:
            raise RuntimeError("Distributed verified teacher selection requires an initialized process group")
        return list(rows)

    def _validate_verified_teacher_window(self, batch_size=None):
        """One generation buffer must cover one complete optimizer update."""
        if getattr(self, "grpo_verified_teacher_normalization", "microbatch") != "optimizer_window":
            return
        args = self.args
        accumulation = int(args.gradient_accumulation_steps)
        if (not getattr(self, "grpo_verified_improvement", False)
                or getattr(self, "grpo_verified_teacher_selection", "per_completion") == "per_completion"
                or accumulation < 1
                or int(args.steps_per_generation) != accumulation
                or int(getattr(self, "num_iterations", getattr(args, "num_iterations", 1))) != 1):
            raise ValueError("Optimizer-window teachers require prompt selection, one iteration, "
                             "and steps_per_generation == gradient_accumulation_steps")
        if batch_size is not None and batch_size != int(args.per_device_train_batch_size) * accumulation:
            raise ValueError("Optimizer-window teachers require a complete local generation buffer")

    def _verified_raw_target(self, task, text, instance, input_text):
        """Canonicalize a verified sampled solution, discarding its claimed objective."""
        if task in ("tsp", "op", "jssp"):
            from clasp.rewards.verified_train_reference import canonical_verified_solution
            result = canonical_verified_solution(task, text, instance, input_text)
            return (result[0], 0.0, 0, False) if result else None
        from clasp.rewards.co_tasks import (
            _format_cvrp_distillation_target, _format_mis_distillation_target,
            _parse_cvrp_solution, _parse_mis_solution,
        )

        objective = self._verified_completion_objective(task, text, instance, input_text)
        if objective is None:
            return None
        if task == "cvrp":
            target = _format_cvrp_distillation_target(_parse_cvrp_solution(text)["routes"], objective)
        else:
            target = _format_mis_distillation_target(_parse_mis_solution(text)["node_set"])
        return target, 0.0, 0, False

    def _verified_constructive_target(self, task, instance, input_text):
        """Bounded, reference-free fallback only for groups with no feasible target."""
        from clasp.rewards.co_tasks import (
            _cvrp_meta, _mis_meta, _format_cvrp_distillation_target,
            _format_mis_distillation_target,
        )

        if task == "cvrp":
            from clasp.rewards.cvrp_verified_improvement import improve_cvrp_routes

            result = improve_cvrp_routes(
                [[0, 0]], *_cvrp_meta(instance, input_text),
                max_passes=self.grpo_verified_improvement_max_passes,
            )
            if result is not None:
                return _format_cvrp_distillation_target(result.routes, result.objective), 0.0, result.passes, True
        elif task == "mis":
            from clasp.rewards.verified_improvement import improve_mis_set

            result = improve_mis_set(
                [], *_mis_meta(instance, input_text),
                max_exchanges=self.grpo_verified_improvement_max_passes,
            )
            if result is not None:
                return _format_mis_distillation_target(result.node_set), 0.0, result.passes, True
        return None

    def _attach_verified_improvement_tensors(self, inputs, output, task_types):
        """Per-completion teachers remain aligned through TRL row shuffling/splitting."""
        ids = output["completion_ids"]
        batch_size, device = ids.size(0), ids.device
        specs = self._improvement_distillation_specs()

        def rows(values):
            if values is None:
                return [None] * batch_size
            if len(values) != batch_size:
                raise ValueError("Verified GRPO requires completion-aligned input metadata")
            return values

        tasks = rows(task_types)
        instances = rows(self._extract_batch_raw_values(inputs, "instance", "instance_json"))
        texts = rows(self._extract_batch_raw_values(inputs, "input"))
        prompts = rows(self._extract_batch_raw_values(inputs, "prompt", "prompts"))
        answers = (rows(self._extract_batch_raw_values(inputs, "answer"))
                   if getattr(self, "grpo_verified_train_reference_teacher", False) else None)
        completions = self.processing_class.batch_decode(
            ids.detach().cpu().tolist(), skip_special_tokens=True
        )
        from clasp.rewards.co_tasks import _as_text

        selection_mode = getattr(self, "grpo_verified_teacher_selection", "per_completion")
        select_best = selection_mode in {"best_per_prompt", "feasibility_incumbent"}
        feasibility_incumbent = selection_mode == "feasibility_incumbent"
        window_normalization = (getattr(self, "grpo_verified_teacher_normalization", "microbatch")
                                == "optimizer_window" and self.model.training)
        if window_normalization:
            self._validate_verified_teacher_window(batch_size)
        rank = int(getattr(self.accelerator, "process_index", 0)) if select_best else 0
        if select_best and (rank < 0 or batch_size >= 2 ** 32):
            raise ValueError("Invalid rank or row count for verified teacher identity")
        for spec in specs:
            targets, gains, passes, repairs, seconds = [], [], [], [], []
            fallbacks = [0.0] * batch_size
            selection_rows = []
            for row in range(batch_size):
                start = time.perf_counter()
                result = None
                if str(tasks[row]).lower() == spec["task"]:
                    instance = self._maybe_json_load(instances[row])
                    input_text = _as_text(texts[row] or prompts[row] or "")
                    if spec.get("training_reference", False):
                        from clasp.rewards.verified_train_reference import training_reference_target
                        result = training_reference_target(spec["task"], answers[row], instance, input_text)
                    else:
                        result = self._verified_improvement_target(
                            spec["task"], completions[row], instance, input_text,
                        )
                    if feasibility_incumbent:
                        raw_target = self._verified_raw_target(spec["task"], completions[row], instance, input_text)
                        if raw_target is not None:
                            raw_value = self._verified_completion_objective(
                                spec["task"], raw_target[0], instance, input_text)
                            improved_value = self._verified_completion_objective(
                                spec["task"], result[0], instance, input_text) if result else None
                            if result is not None and improved_value is None:
                                raise RuntimeError("Verified teacher failed independent selection verification")
                            if (improved_value is None or (raw_value <= improved_value if spec["task"] in ("tsp", "cvrp", "jssp")
                                                          else raw_value >= improved_value)):
                                result = raw_target
                token_ids = self._tokenize_verified_target(result[0], spec["max_tokens"]) if result else []
                if select_best and str(tasks[row]).lower() == spec["task"]:
                    raw_objective = self._verified_completion_objective(
                        spec["task"], completions[row], instance, input_text)
                    target_objective = self._verified_completion_objective(
                        spec["task"], result[0], instance, input_text) if token_ids else None
                    if token_ids and target_objective is None:
                        raise RuntimeError("Verified teacher failed independent selection verification")
                    selection_rows.append({
                        "row_id": (rank << 32) + row, "task": spec["task"],
                        "prompt_key": self._verified_teacher_prompt_key(
                            spec["task"], instance, prompts[row], texts[row]),
                        "source_feasible": raw_objective is not None,
                        "source_objective": raw_objective,
                        "target_text": result[0] if token_ids else None,
                        "target_objective": target_objective,
                    })
                targets.append(token_ids)
                gains.append(result[1] if token_ids else 0.0)
                passes.append(result[2] if token_ids else 0.0)
                repairs.append(float(result[3]) if token_ids else 0.0)
                seconds.append(time.perf_counter() - start)
            if select_best:
                from clasp.rewards.verified_teacher_selection import select_verified_teachers

                global_rows = self._gather_verified_teacher_rows(selection_rows)
                if feasibility_incumbent:
                    groups = {}
                    for item in global_rows:
                        groups.setdefault(item["prompt_key"], []).append(item)
                    fallback_owners = {
                        min(item["row_id"] for item in group)
                        for group in groups.values()
                        if not any(item["source_feasible"] or item["target_text"] is not None for item in group)
                    }
                    if fallback_owners:
                        for item in selection_rows:
                            if item["row_id"] not in fallback_owners:
                                continue
                            row = item["row_id"] - (rank << 32)
                            start = time.perf_counter()
                            instance = self._maybe_json_load(instances[row])
                            input_text = _as_text(texts[row] or prompts[row] or "")
                            result = self._verified_constructive_target(spec["task"], instance, input_text)
                            tokens = self._tokenize_verified_target(result[0], spec["max_tokens"]) if result else []
                            if tokens:
                                objective = self._verified_completion_objective(spec["task"], result[0], instance, input_text)
                                if objective is None:
                                    raise RuntimeError("Constructive teacher failed independent verification")
                                item.update(target_text=result[0], target_objective=objective)
                                targets[row], gains[row], passes[row], repairs[row] = tokens, result[1], result[2], float(result[3])
                                fallbacks[row] = 1.0
                            seconds[row] += time.perf_counter() - start
                        # All ranks see the same owner set and enter this collective.
                        global_rows = self._gather_verified_teacher_rows(selection_rows)
                eligible = [float(bool(tokens)) for tokens in targets]
                selection = select_verified_teachers(
                    global_rows, selection_mode=("feasibility_or_improvement" if feasibility_incumbent
                                                 else "strict_improvement"))
                selected = set(selection["selected_row_ids"])
                for row in range(batch_size):
                    if (rank << 32) + row not in selected:
                        targets[row] = []
                        gains[row] = passes[row] = repairs[row] = 0.0
                        fallbacks[row] = 0.0
                prefix = spec["prefix"]
                output[f"{prefix}_eligible"] = torch.tensor(eligible, dtype=torch.float32, device=device)
                if window_normalization:
                    # A row-aligned constant survives TRL's shuffle and split;
                    # never infer an optimizer-wide denominator from one microbatch.
                    output[f"{prefix}_window_count"] = torch.full(
                        (batch_size,), len(selected), dtype=torch.long, device=device)
                mode = "train" if self.model.training else "eval"
                for name, value in selection["diagnostics"].items():
                    if value is not None:
                        self._append_metric(mode, f"grpo/{prefix}_selection_{name}", value)
            width = max([len(x) for x in targets] + [1])
            target_ids = torch.full((batch_size, width), self._safe_pad_token_id(),
                                    dtype=torch.long, device=device)
            mask = torch.zeros_like(target_ids)
            for row, tokens in enumerate(targets):
                if tokens:
                    target_ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
                    mask[row, :len(tokens)] = 1
            prefix = spec["prefix"]
            output[f"{prefix}_completion_ids"] = target_ids
            output[f"{prefix}_completion_mask"] = mask
            for suffix, values in (("gains", gains), ("passes", passes),
                                   ("repairs", repairs), ("seconds", seconds), ("fallbacks", fallbacks)):
                output[f"{prefix}_{suffix}"] = torch.tensor(values, dtype=torch.float32, device=device)

    def _improve_cvrp_completion(
        self,
        completion_text: str,
        instance,
        input_text: str,
    ) -> tuple[str, float, int] | None:
        from clasp.rewards.co_tasks import (
            _check_cvrp_constraints,
            _cvrp_local_search_improved_routes,
            _cvrp_meta,
            _format_cvrp_distillation_target,
            _parse_cvrp_solution,
        )

        coords, demands, capacity = _cvrp_meta(instance, input_text)
        if not coords or not demands:
            return None
        routes = _parse_cvrp_solution(str(completion_text))["routes"]
        if not routes:
            return None
        _, _, feasible = _check_cvrp_constraints(routes, coords, demands, capacity)
        if not feasible:
            return None
        improved = _cvrp_local_search_improved_routes(
            coords,
            demands,
            capacity,
            routes,
            max_passes=int(self.clasp_cvrp_local_search_distill_max_passes),
        )
        if improved is None:
            return None
        improved_routes, _, improved_total, gain_ratio, passes = improved
        return (
            _format_cvrp_distillation_target(improved_routes, improved_total),
            float(gain_ratio),
            int(passes),
        )

    def _improve_jssp_completion(
        self,
        completion_text: str,
        instance,
        input_text: str,
    ) -> tuple[str, float, int] | None:
        from clasp.rewards.co_tasks import (
            _check_jssp_constraints,
            _format_jssp_distillation_target,
            _jssp_jobs,
            _jssp_order_descent_improved_schedule,
            _parse_jssp_solution,
        )

        jobs = _jssp_jobs(instance, input_text)
        if not jobs:
            return None
        schedule = _parse_jssp_solution(str(completion_text))["schedule"]
        if not schedule:
            return None
        _, _, feasible = _check_jssp_constraints(jobs, schedule)
        if not feasible:
            return None
        improved = _jssp_order_descent_improved_schedule(
            jobs,
            schedule,
            max_passes=int(self.clasp_jssp_active_schedule_distill_max_passes),
        )
        if improved is None:
            return None
        improved_schedule, _, improved_makespan, gain_ratio, passes = improved
        return (
            _format_jssp_distillation_target(improved_schedule, improved_makespan),
            float(gain_ratio),
            int(passes),
        )

    def _improve_mis_completion(
        self,
        completion_text: str,
        instance,
        input_text: str,
    ) -> tuple[str, float, int] | None:
        from clasp.rewards.co_tasks import (
            _check_mis_constraints,
            _format_mis_distillation_target,
            _mis_maximalized_set,
            _mis_meta,
            _parse_mis_solution,
        )

        num_nodes, edges = _mis_meta(instance, input_text)
        node_set = _parse_mis_solution(str(completion_text))["node_set"]
        if not node_set:
            return None
        _, _, feasible = _check_mis_constraints(node_set, num_nodes, edges)
        if not feasible:
            return None
        improved = _mis_maximalized_set(node_set, num_nodes, edges)
        if improved is None:
            return None
        improved_set, _, _, added_fraction, added_count = improved
        return (
            _format_mis_distillation_target(improved_set),
            float(added_fraction),
            int(added_count),
        )

    def _attach_improvement_distillation_tensors(
        self,
        inputs,
        output: dict,
        task_types: list[str] | None,
    ) -> None:
        specs = self._improvement_distillation_specs()
        if not specs:
            return

        if getattr(self, "grpo_verified_improvement", False):
            self._attach_verified_improvement_tensors(inputs, output, task_types)
            return

        completion_ids = output.get("completion_ids") if isinstance(output, dict) else None
        if not torch.is_tensor(completion_ids):
            return
        if self.num_generations <= 0 or completion_ids.size(0) % self.num_generations != 0:
            return

        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
            return

        batch_size = int(completion_ids.size(0))
        num_groups = batch_size // self.num_generations
        device = completion_ids.device

        if task_types is not None and len(task_types) == batch_size:
            group_task_types = task_types[:: self.num_generations]
        elif task_types is not None and len(task_types) == num_groups:
            group_task_types = task_types
        else:
            group_task_types = [None] * num_groups

        raw_inputs = self._extract_batch_raw_values(inputs, "input")
        raw_prompts = self._extract_batch_raw_values(inputs, "prompt", "prompts")
        raw_instances = self._extract_batch_raw_values(inputs, "instance", "instance_json")
        group_inputs = self._normalize_group_values(
            raw_inputs,
            num_groups,
            batch_size,
            self.num_generations,
        )
        group_prompts = self._normalize_group_values(
            raw_prompts,
            num_groups,
            batch_size,
            self.num_generations,
        )
        group_instances = self._normalize_group_values(
            raw_instances,
            num_groups,
            batch_size,
            self.num_generations,
        )

        try:
            completion_texts = tokenizer.batch_decode(
                completion_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except TypeError:
            completion_texts = tokenizer.batch_decode(completion_ids.detach().cpu().tolist())
        except Exception:
            return

        from clasp.rewards.co_tasks import _as_text

        pad_id = self._safe_pad_token_id()
        for spec in specs:
            tokenized_targets: list[list[int]] = []
            gain_values: list[float] = []
            pass_values: list[float] = []
            for row_idx, completion_text in enumerate(completion_texts):
                group_idx = row_idx // self.num_generations
                task_name = (
                    group_task_types[group_idx] if group_idx < len(group_task_types) else None
                )
                if task_name is not None and str(task_name).lower() != spec["task"]:
                    tokenized_targets.append([])
                    gain_values.append(0.0)
                    pass_values.append(0.0)
                    continue

                input_text = (
                    _as_text(group_inputs[group_idx]) if group_idx < len(group_inputs) else ""
                )
                if not input_text and group_idx < len(group_prompts):
                    input_text = _as_text(group_prompts[group_idx])
                instance = (
                    self._maybe_json_load(group_instances[group_idx])
                    if group_idx < len(group_instances)
                    else None
                )
                try:
                    improved = spec["improve"](str(completion_text), instance, input_text)
                except Exception:
                    improved = None
                if improved is None:
                    tokenized_targets.append([])
                    gain_values.append(0.0)
                    pass_values.append(0.0)
                    continue
                target_text, gain_ratio, num_passes = improved
                if gain_ratio < spec["min_gain"]:
                    tokenized_targets.append([])
                    gain_values.append(0.0)
                    pass_values.append(0.0)
                    continue
                token_ids = self._tokenize_completion_text(target_text, spec["max_tokens"])
                tokenized_targets.append(token_ids)
                gain_values.append(float(gain_ratio))
                pass_values.append(float(num_passes))

            if len(tokenized_targets) != batch_size:
                continue

            max_len = max([len(tokens) for tokens in tokenized_targets] + [1])
            target_ids = torch.full(
                (batch_size, max_len),
                int(pad_id),
                dtype=torch.long,
                device=device,
            )
            target_mask = torch.zeros(
                (batch_size, max_len),
                dtype=torch.long,
                device=device,
            )
            for row_idx, token_ids in enumerate(tokenized_targets):
                if not token_ids:
                    continue
                ids = torch.tensor(token_ids, dtype=torch.long, device=device)
                target_ids[row_idx, : ids.numel()] = ids
                target_mask[row_idx, : ids.numel()] = 1

            output[f"{spec['prefix']}_completion_ids"] = target_ids
            output[f"{spec['prefix']}_completion_mask"] = target_mask
            output[f"{spec['prefix']}_gains"] = torch.tensor(
                gain_values,
                dtype=torch.float32,
                device=device,
            )
            output[f"{spec['prefix']}_passes"] = torch.tensor(
                pass_values,
                dtype=torch.float32,
                device=device,
            )

    def _compute_improvement_distillation_losses(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        specs = self._improvement_distillation_specs()
        if not (self.model.training and specs):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        total_loss = zero
        stats: dict[str, float] = {}
        batch_size = int(completion_ids.size(0))
        for spec in specs:
            prefix = spec["prefix"]
            verified_grpo = spec.get("verified_grpo", False)
            target_ids = inputs.get(f"{prefix}_completion_ids")
            target_mask = inputs.get(f"{prefix}_completion_mask")
            gains = inputs.get(f"{prefix}_gains")
            passes = inputs.get(f"{prefix}_passes")
            stats[f"clasp/{prefix}_enabled"] = 1.0
            stats[f"clasp/{prefix}_has_tensors"] = float(
                torch.is_tensor(target_ids) and torch.is_tensor(target_mask)
            )
            stats[f"clasp/{prefix}_valid_row_rate"] = 0.0
            stats[f"clasp/{prefix}_nll"] = 0.0
            stats[f"clasp/{prefix}_length"] = 0.0
            stats[f"clasp/{prefix}_gain"] = 0.0
            stats[f"clasp/{prefix}_passes"] = 0.0
            has_local_tensors = bool(
                torch.is_tensor(target_ids)
                and torch.is_tensor(target_mask)
                and target_ids.shape == target_mask.shape
                and target_ids.size(0) == batch_size
            )
            if has_local_tensors:
                valid = target_mask.sum(dim=1) > 0
            else:
                valid = torch.zeros(batch_size, dtype=torch.bool, device=per_token_logps.device)

            # Rank-uniform participation: a forward through the (DDP-wrapped)
            # model is NOT collective-free, so every rank must run the same
            # number of forwards or the NCCL sequences desync (the fable dev
            # shakeout 1249436 failure mode). All ranks agree first; if any
            # rank has a valid target, every rank forwards, empty ranks
            # contribute an exact zero.
            local_any = valid.any().to(device=per_token_logps.device, dtype=torch.float32)
            global_count = None
            window_count = None
            accumulation = 1
            if verified_grpo:
                # Pooled valid-row mean: averaging each rank's mean would
                # overweight ranks with few teachers. Do not swallow a failed
                # collective, which could desynchronize forward participation.
                global_count = self.accelerator.gather(valid.sum().reshape(1)).sum()
                global_any = bool(global_count.item() > 0)
                if getattr(self, "grpo_verified_teacher_normalization", "microbatch") == "optimizer_window":
                    self._validate_verified_teacher_window()
                    accumulation = int(self.args.gradient_accumulation_steps)
                    if int(self.current_gradient_accumulation_steps) != accumulation:
                        raise ValueError("Optimizer-window teachers do not support partial accumulation updates")
                    counts = inputs.get(f"{prefix}_window_count")
                    if (not torch.is_tensor(counts) or counts.shape != (batch_size,)
                            or counts.dtype != torch.long or bool((counts < 0).any())
                            or not bool((counts == counts[0]).all())):
                        raise ValueError("Missing or inconsistent row-aligned optimizer-window teacher count")
                    all_counts = self.accelerator.gather(counts[:1])
                    if not bool((all_counts == all_counts[0]).all()):
                        raise ValueError("Optimizer-window teacher count differs across ranks")
                    window_count = all_counts[0]
                    if int(window_count.item()) < int(global_count.item()):
                        raise ValueError("Optimizer-window teacher count is smaller than its microbatch count")
                    stats[f"grpo/{prefix}_window_teacher_count"] = float(window_count.item())
                    stats[f"grpo/{prefix}_microbatch_teacher_count"] = float(global_count.item())
                for suffix in ("repairs", "seconds", "eligible", "fallbacks"):
                    values = inputs.get(f"{prefix}_{suffix}")
                    if torch.is_tensor(values):
                        stats[f"grpo/{prefix}_{suffix}_per_completion"] = float(values.float().mean().item())
            else:
                try:
                    global_any = bool(
                        self.accelerator.gather(local_any.reshape(1)).sum().item() > 0
                    )
                except Exception:
                    global_any = bool(valid.any().item())
            if not global_any:
                continue

            # Memory: forward ONLY the valid-row subset (shakeout 1251747 hit
            # CUDA OOM forwarding the full padded batch per spec while every
            # spec's graph stays retained until the single backward). Ranks
            # without valid rows forward a 1-row dummy so the forward COUNT
            # stays rank-uniform; per-rank shapes may differ (no shape-coupled
            # collectives in a DDP forward).
            local_valid = bool(valid.any().item())
            if local_valid:
                sub_prompt_ids = prompt_ids[valid]
                sub_prompt_mask = prompt_mask[valid]
                sub_ids = target_ids[valid]
                sub_mask = target_mask[valid]
                width = max(int(sub_mask.sum(dim=1).max().item()), 1)
                sub_ids = sub_ids[:, :width]
                sub_mask = sub_mask[:, :width]
            else:
                sub_prompt_ids = prompt_ids[:1]
                sub_prompt_mask = prompt_mask[:1]
                sub_ids = torch.full(
                    (1, 1),
                    int(self._safe_pad_token_id()),
                    dtype=torch.long,
                    device=per_token_logps.device,
                )
                sub_mask = torch.zeros(
                    (1, 1), dtype=torch.long, device=per_token_logps.device
                )

            target_input_ids = torch.cat([sub_prompt_ids, sub_ids], dim=1)
            target_attention_mask = torch.cat([sub_prompt_mask, sub_mask], dim=1)
            target_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                target_input_ids,
                target_attention_mask,
                sub_ids.size(1),
                compute_entropy=not verified_grpo,
            )
            target_mask_float = sub_mask.to(dtype=target_per_token_logps.dtype)
            token_counts = target_mask_float.sum(dim=1).clamp(min=1.0)
            sequence_nll = -(
                (target_per_token_logps * target_mask_float).sum(dim=1) / token_counts
            )
            if local_valid:
                if verified_grpo:
                    if window_count is not None:
                        # DDP averages W ranks; compute_loss subsequently divides
                        # by A microbatches. The optimizer therefore sees sum(NLL)/M.
                        spec_loss = (sequence_nll.sum() * self.accelerator.num_processes
                                     * accumulation / window_count)
                    else:
                        spec_loss = sequence_nll.sum() * self.accelerator.num_processes / global_count
                else:
                    spec_loss = sequence_nll.mean()
                stats[f"clasp/{prefix}_valid_row_rate"] = float(valid.float().mean().item())
                stats[f"clasp/{prefix}_nll"] = float(sequence_nll.detach().mean().item())
                stats[f"clasp/{prefix}_length"] = float(token_counts.detach().mean().item())
                if torch.is_tensor(gains) and gains.numel() == batch_size:
                    stats[f"clasp/{prefix}_gain"] = float(gains[valid].detach().mean().item())
                if torch.is_tensor(passes) and passes.numel() == batch_size:
                    stats[f"clasp/{prefix}_passes"] = float(
                        passes[valid].detach().mean().item()
                    )
            else:
                # Participate in the forward, contribute nothing to the loss.
                spec_loss = target_per_token_logps.sum() * 0.0
            total_loss = total_loss + spec["weight"] * spec_loss
        return total_loss, stats

    def _set_clasp_generation_budget(self, task_types: list[str] | None):
        self._clasp_last_generation_budget_stats = {
            "clasp/adaptive_budget_enabled": 0.0,
            "clasp/adaptive_budget_requested_min": 0.0,
            "clasp/adaptive_budget_requested_max": 0.0,
            "clasp/adaptive_budget_effective": 0.0,
            "clasp/adaptive_budget_mixed_unequal": 0.0,
            "clasp/adaptive_budget_global_max_ablation": 0.0,
        }
        if not (self.clasp_enabled and self.clasp_adaptive_length is not None and task_types):
            return []

        normalized_tasks = [str(task).strip().lower() for task in task_types]
        task_budgets = {
            task: int(self.clasp_adaptive_length.budget(task))
            for task in dict.fromkeys(normalized_tasks)
        }
        budgets = [task_budgets[task] for task in normalized_tasks]
        if not budgets:
            return []
        requested_min = int(min(budgets))
        requested_max = int(max(budgets))
        mixed_unequal = len(task_budgets) > 1 and requested_min != requested_max
        if mixed_unequal and self.clasp_mixed_adaptive_length_mode == "error":
            requested = ", ".join(
                f"{task}={budget}" for task, budget in sorted(task_budgets.items())
            )
            raise RuntimeError(
                "Task-adaptive generation cannot apply unequal per-row limits through "
                f"the current batch-wide generation backend ({requested}). Use "
                "homogeneous/specialist batches, align the task budgets, or select "
                "--clasp_mixed_adaptive_length_mode global_max as an explicit ablation."
            )
        batch_budget = requested_max
        if batch_budget <= 0:
            return []

        configured_caps = []
        for obj, attr in (
            (self, "max_completion_length"),
            (getattr(self, "args", None), "max_completion_length"),
        ):
            if obj is None or not hasattr(obj, attr):
                continue
            value = getattr(obj, attr)
            if value is not None and int(value) > 0:
                configured_caps.append(int(value))
        effective_budget = min([batch_budget, *configured_caps])
        self._clasp_last_generation_budget_stats = {
            "clasp/adaptive_budget_enabled": 1.0,
            "clasp/adaptive_budget_requested_min": float(requested_min),
            "clasp/adaptive_budget_requested_max": float(requested_max),
            "clasp/adaptive_budget_effective": float(effective_budget),
            "clasp/adaptive_budget_mixed_unequal": float(mixed_unequal),
            "clasp/adaptive_budget_global_max_ablation": float(
                mixed_unequal
                and self.clasp_mixed_adaptive_length_mode == "global_max"
            ),
        }

        restorers = []
        for obj, attr in (
            (self, "max_completion_length"),
            (getattr(self, "args", None), "max_completion_length"),
        ):
            if obj is None or not hasattr(obj, attr):
                continue
            old_value = getattr(obj, attr)
            if old_value is None:
                new_value = effective_budget
            else:
                new_value = min(int(old_value), effective_budget)
            if old_value is not None and int(old_value) == int(new_value):
                continue
            setattr(obj, attr, int(new_value))
            restorers.append((obj, attr, old_value))

        generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = getattr(getattr(self, "model", None), "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "max_new_tokens"):
            old_value = getattr(generation_config, "max_new_tokens", None)
            if old_value is None or int(old_value) != effective_budget:
                setattr(generation_config, "max_new_tokens", effective_budget)
                restorers.append((generation_config, "max_new_tokens", old_value))

        return restorers

    @staticmethod
    def _restore_generation_budget(restorers) -> None:
        for obj, attr, old_value in reversed(restorers):
            setattr(obj, attr, old_value)

    def _observe_clasp_completion_lengths(self, components: dict, output: dict) -> None:
        if not (self.clasp_enabled and self.clasp_adaptive_length is not None):
            return
        task_types = components.get("task_types") if isinstance(components, dict) else None
        completions = output.get("completion_ids") if isinstance(output, dict) else None
        if task_types is None or completions is None or len(task_types) != completions.size(0):
            return

        pad_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_id is None:
            lengths = [int(completions.size(1))] * completions.size(0)
        else:
            lengths = completions.ne(pad_id).sum(dim=1).tolist()
        for task_name, length in zip(task_types, lengths):
            self.clasp_adaptive_length.observe(str(task_name), int(length))

    def _compute_reference_imitation_loss(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not (self.model.training and self._reference_imitation_enabled()):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        reference_ids = inputs.get("reference_completion_ids")
        reference_mask = inputs.get("reference_completion_mask")
        batch_size = int(completion_ids.size(0))
        stats = {
            "clasp/reference_imitation_enabled": 1.0,
            "clasp/reference_imitation_has_tensors": float(
                torch.is_tensor(reference_ids) and torch.is_tensor(reference_mask)
            ),
            "clasp/reference_imitation_shape_ok": 0.0,
            "clasp/reference_imitation_tensor_rows": float(
                int(reference_ids.size(0)) if torch.is_tensor(reference_ids) and reference_ids.dim() > 0 else 0
            ),
            "clasp/reference_imitation_batch_rows": float(batch_size),
            "clasp/reference_imitation_valid_group_rate": 0.0,
            "clasp/reference_imitation_nll": 0.0,
            "clasp/reference_imitation_length": 0.0,
        }
        if not (torch.is_tensor(reference_ids) and torch.is_tensor(reference_mask)):
            return zero, stats
        if reference_ids.shape != reference_mask.shape:
            return zero, stats

        if batch_size <= 0:
            return zero, stats

        if reference_ids.size(0) == batch_size:
            group_reference_ids = reference_ids
            group_reference_mask = reference_mask
            group_prompt_ids = prompt_ids
            group_prompt_mask = prompt_mask
        elif (
            self.num_generations > 0
            and batch_size % self.num_generations == 0
            and reference_ids.size(0) == batch_size // self.num_generations
        ):
            group_reference_ids = reference_ids
            group_reference_mask = reference_mask
            group_prompt_ids = prompt_ids[:: self.num_generations]
            group_prompt_mask = prompt_mask[:: self.num_generations]
        else:
            return zero, stats

        if group_prompt_ids.size(0) != group_reference_ids.size(0):
            return zero, stats

        stats["clasp/reference_imitation_shape_ok"] = 1.0
        valid = group_reference_mask.sum(dim=1) > 0
        if not bool(valid.any().item()):
            return zero, stats

        ref_input_ids = torch.cat([group_prompt_ids, group_reference_ids], dim=1)
        ref_attention_mask = torch.cat([group_prompt_mask, group_reference_mask], dim=1)
        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            ref_input_ids,
            ref_attention_mask,
            group_reference_ids.size(1),
            compute_entropy=True,
        )
        ref_mask_float = group_reference_mask.to(dtype=ref_per_token_logps.dtype)
        token_counts = ref_mask_float.sum(dim=1).clamp(min=1.0)
        sequence_nll = -((ref_per_token_logps * ref_mask_float).sum(dim=1) / token_counts)
        selected_nll = sequence_nll[valid]
        selected_lengths = token_counts[valid]
        loss = selected_nll.mean()
        stats["clasp/reference_imitation_valid_group_rate"] = float(valid.float().mean().item())
        stats["clasp/reference_imitation_nll"] = float(selected_nll.detach().mean().item())
        stats["clasp/reference_imitation_length"] = float(selected_lengths.detach().mean().item())
        return loss, stats

    def _compute_reference_contrastive_loss(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not (self.model.training and self._reference_contrastive_enabled()):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        positive_ids = inputs.get("reference_contrastive_positive_ids")
        positive_mask = inputs.get("reference_contrastive_positive_mask")
        negative_ids = inputs.get("reference_contrastive_negative_ids")
        negative_mask = inputs.get("reference_contrastive_negative_mask")
        batch_size = int(completion_ids.size(0))
        stats = {
            "clasp/reference_contrastive_enabled": 1.0,
            "clasp/reference_contrastive_has_tensors": float(
                torch.is_tensor(positive_ids)
                and torch.is_tensor(positive_mask)
                and torch.is_tensor(negative_ids)
                and torch.is_tensor(negative_mask)
            ),
            "clasp/reference_contrastive_shape_ok": 0.0,
            "clasp/reference_contrastive_tensor_rows": float(
                int(positive_ids.size(0)) if torch.is_tensor(positive_ids) and positive_ids.dim() > 0 else 0
            ),
            "clasp/reference_contrastive_batch_rows": float(batch_size),
            "clasp/reference_contrastive_valid_group_rate": 0.0,
            "clasp/reference_contrastive_pos_nll": 0.0,
            "clasp/reference_contrastive_neg_nll": 0.0,
            "clasp/reference_contrastive_nll_gap": 0.0,
            "clasp/reference_contrastive_length": 0.0,
        }
        tensors_ok = (
            torch.is_tensor(positive_ids)
            and torch.is_tensor(positive_mask)
            and torch.is_tensor(negative_ids)
            and torch.is_tensor(negative_mask)
        )
        if not tensors_ok:
            return zero, stats
        if positive_ids.shape != positive_mask.shape or negative_ids.shape != negative_mask.shape:
            return zero, stats
        if positive_ids.shape != negative_ids.shape:
            return zero, stats

        if batch_size <= 0:
            return zero, stats

        if positive_ids.size(0) == batch_size:
            group_positive_ids = positive_ids
            group_positive_mask = positive_mask
            group_negative_ids = negative_ids
            group_negative_mask = negative_mask
            group_prompt_ids = prompt_ids
            group_prompt_mask = prompt_mask
        elif (
            self.num_generations > 0
            and batch_size % self.num_generations == 0
            and positive_ids.size(0) == batch_size // self.num_generations
        ):
            group_positive_ids = positive_ids
            group_positive_mask = positive_mask
            group_negative_ids = negative_ids
            group_negative_mask = negative_mask
            group_prompt_ids = prompt_ids[:: self.num_generations]
            group_prompt_mask = prompt_mask[:: self.num_generations]
        else:
            return zero, stats

        if group_prompt_ids.size(0) != group_positive_ids.size(0):
            return zero, stats

        stats["clasp/reference_contrastive_shape_ok"] = 1.0
        valid = (group_positive_mask.sum(dim=1) > 0) & (group_negative_mask.sum(dim=1) > 0)
        if not bool(valid.any().item()):
            return zero, stats

        positive_input_ids = torch.cat([group_prompt_ids, group_positive_ids], dim=1)
        positive_attention_mask = torch.cat([group_prompt_mask, group_positive_mask], dim=1)
        negative_input_ids = torch.cat([group_prompt_ids, group_negative_ids], dim=1)
        negative_attention_mask = torch.cat([group_prompt_mask, group_negative_mask], dim=1)
        positive_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            positive_input_ids,
            positive_attention_mask,
            group_positive_ids.size(1),
            compute_entropy=True,
        )
        negative_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            negative_input_ids,
            negative_attention_mask,
            group_negative_ids.size(1),
            compute_entropy=True,
        )
        positive_mask_float = group_positive_mask.to(dtype=positive_per_token_logps.dtype)
        negative_mask_float = group_negative_mask.to(dtype=negative_per_token_logps.dtype)
        positive_token_counts = positive_mask_float.sum(dim=1).clamp(min=1.0)
        negative_token_counts = negative_mask_float.sum(dim=1).clamp(min=1.0)
        positive_logp = (positive_per_token_logps * positive_mask_float).sum(dim=1) / positive_token_counts
        negative_logp = (negative_per_token_logps * negative_mask_float).sum(dim=1) / negative_token_counts
        margin = float(self.clasp_reference_contrastive_margin)
        pair_losses = torch.nn.functional.softplus(negative_logp - positive_logp + margin)
        selected_loss = pair_losses[valid]
        selected_pos_nll = -positive_logp[valid]
        selected_neg_nll = -negative_logp[valid]
        selected_lengths = positive_token_counts[valid]
        loss = selected_loss.mean()
        stats["clasp/reference_contrastive_valid_group_rate"] = float(valid.float().mean().item())
        stats["clasp/reference_contrastive_pos_nll"] = float(selected_pos_nll.detach().mean().item())
        stats["clasp/reference_contrastive_neg_nll"] = float(selected_neg_nll.detach().mean().item())
        stats["clasp/reference_contrastive_nll_gap"] = float(
            (selected_neg_nll - selected_pos_nll).detach().mean().item()
        )
        stats["clasp/reference_contrastive_length"] = float(selected_lengths.detach().mean().item())
        return loss, stats

    def _compute_self_replay_imitation_loss(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not (
            self.model.training
            and self._self_replay_enabled()
            and self.clasp_self_replay_imitation_loss_weight > 0.0
        ):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        replay_ids = inputs.get("self_replay_completion_ids")
        replay_mask = inputs.get("self_replay_completion_mask")
        batch_size = int(completion_ids.size(0))
        stats = {
            "clasp/self_replay_imitation_enabled": 1.0,
            "clasp/self_replay_imitation_has_tensors": float(
                torch.is_tensor(replay_ids) and torch.is_tensor(replay_mask)
            ),
            "clasp/self_replay_imitation_valid_group_rate": 0.0,
            "clasp/self_replay_imitation_nll": 0.0,
            "clasp/self_replay_imitation_length": 0.0,
        }
        if not (torch.is_tensor(replay_ids) and torch.is_tensor(replay_mask)):
            return zero, stats
        if replay_ids.shape != replay_mask.shape:
            return zero, stats
        if self.num_generations <= 0 or batch_size % self.num_generations != 0:
            return zero, stats

        num_groups = batch_size // self.num_generations
        if replay_ids.size(0) == batch_size:
            group_replay_ids = replay_ids
            group_replay_mask = replay_mask
            group_prompt_ids = prompt_ids
            group_prompt_mask = prompt_mask
        elif replay_ids.size(0) == num_groups:
            group_replay_ids = replay_ids
            group_replay_mask = replay_mask
            group_prompt_ids = prompt_ids[:: self.num_generations]
            group_prompt_mask = prompt_mask[:: self.num_generations]
        else:
            return zero, stats

        valid = group_replay_mask.sum(dim=1) > 0
        if not bool(valid.any().item()):
            return zero, stats

        replay_input_ids = torch.cat([group_prompt_ids, group_replay_ids], dim=1)
        replay_attention_mask = torch.cat([group_prompt_mask, group_replay_mask], dim=1)
        replay_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            replay_input_ids,
            replay_attention_mask,
            group_replay_ids.size(1),
            compute_entropy=True,
        )
        replay_mask_float = group_replay_mask.to(dtype=replay_per_token_logps.dtype)
        token_counts = replay_mask_float.sum(dim=1).clamp(min=1.0)
        sequence_nll = -((replay_per_token_logps * replay_mask_float).sum(dim=1) / token_counts)
        selected_nll = sequence_nll[valid]
        selected_lengths = token_counts[valid]
        loss = selected_nll.mean()
        stats["clasp/self_replay_imitation_valid_group_rate"] = float(valid.float().mean().item())
        stats["clasp/self_replay_imitation_nll"] = float(selected_nll.detach().mean().item())
        stats["clasp/self_replay_imitation_length"] = float(selected_lengths.detach().mean().item())
        return loss, stats

    def _compute_self_replay_contrastive_loss(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not (
            self.model.training
            and self._self_replay_enabled()
            and self.clasp_self_replay_contrastive_loss_weight > 0.0
        ):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        positive_ids = inputs.get("self_replay_completion_ids")
        positive_mask = inputs.get("self_replay_completion_mask")
        negative_ids = inputs.get("self_replay_negative_ids")
        negative_mask = inputs.get("self_replay_negative_mask")
        batch_size = int(completion_ids.size(0))
        stats = {
            "clasp/self_replay_contrastive_enabled": 1.0,
            "clasp/self_replay_contrastive_has_tensors": float(
                torch.is_tensor(positive_ids)
                and torch.is_tensor(positive_mask)
                and torch.is_tensor(negative_ids)
                and torch.is_tensor(negative_mask)
            ),
            "clasp/self_replay_contrastive_valid_group_rate": 0.0,
            "clasp/self_replay_contrastive_pos_nll": 0.0,
            "clasp/self_replay_contrastive_neg_nll": 0.0,
            "clasp/self_replay_contrastive_nll_gap": 0.0,
        }
        tensors_ok = (
            torch.is_tensor(positive_ids)
            and torch.is_tensor(positive_mask)
            and torch.is_tensor(negative_ids)
            and torch.is_tensor(negative_mask)
        )
        if not tensors_ok:
            return zero, stats
        if positive_ids.shape != positive_mask.shape or negative_ids.shape != negative_mask.shape:
            return zero, stats
        if positive_ids.shape != negative_ids.shape:
            return zero, stats
        if self.num_generations <= 0 or batch_size % self.num_generations != 0:
            return zero, stats

        num_groups = batch_size // self.num_generations
        if positive_ids.size(0) == batch_size:
            group_positive_ids = positive_ids
            group_positive_mask = positive_mask
            group_negative_ids = negative_ids
            group_negative_mask = negative_mask
            group_prompt_ids = prompt_ids
            group_prompt_mask = prompt_mask
        elif positive_ids.size(0) == num_groups:
            group_positive_ids = positive_ids
            group_positive_mask = positive_mask
            group_negative_ids = negative_ids
            group_negative_mask = negative_mask
            group_prompt_ids = prompt_ids[:: self.num_generations]
            group_prompt_mask = prompt_mask[:: self.num_generations]
        else:
            return zero, stats

        valid = (group_positive_mask.sum(dim=1) > 0) & (group_negative_mask.sum(dim=1) > 0)
        if not bool(valid.any().item()):
            return zero, stats

        positive_input_ids = torch.cat([group_prompt_ids, group_positive_ids], dim=1)
        positive_attention_mask = torch.cat([group_prompt_mask, group_positive_mask], dim=1)
        negative_input_ids = torch.cat([group_prompt_ids, group_negative_ids], dim=1)
        negative_attention_mask = torch.cat([group_prompt_mask, group_negative_mask], dim=1)
        positive_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            positive_input_ids,
            positive_attention_mask,
            group_positive_ids.size(1),
            compute_entropy=True,
        )
        negative_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            negative_input_ids,
            negative_attention_mask,
            group_negative_ids.size(1),
            compute_entropy=True,
        )
        positive_mask_float = group_positive_mask.to(dtype=positive_per_token_logps.dtype)
        negative_mask_float = group_negative_mask.to(dtype=negative_per_token_logps.dtype)
        positive_token_counts = positive_mask_float.sum(dim=1).clamp(min=1.0)
        negative_token_counts = negative_mask_float.sum(dim=1).clamp(min=1.0)
        positive_logp = (positive_per_token_logps * positive_mask_float).sum(dim=1) / positive_token_counts
        negative_logp = (negative_per_token_logps * negative_mask_float).sum(dim=1) / negative_token_counts
        margin = float(self.clasp_self_replay_contrastive_margin)
        pair_losses = torch.nn.functional.softplus(negative_logp - positive_logp + margin)
        selected_loss = pair_losses[valid]
        selected_pos_nll = -positive_logp[valid]
        selected_neg_nll = -negative_logp[valid]
        loss = selected_loss.mean()
        stats["clasp/self_replay_contrastive_valid_group_rate"] = float(valid.float().mean().item())
        stats["clasp/self_replay_contrastive_pos_nll"] = float(selected_pos_nll.detach().mean().item())
        stats["clasp/self_replay_contrastive_neg_nll"] = float(selected_neg_nll.detach().mean().item())
        stats["clasp/self_replay_contrastive_nll_gap"] = float(
            (selected_neg_nll - selected_pos_nll).detach().mean().item()
        )
        return loss, stats

    def _compute_tsp_two_opt_distillation_loss(
        self,
        model,
        inputs: dict,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not (self.model.training and self._tsp_two_opt_distillation_enabled()):
            return None, {}

        zero = per_token_logps.sum() * 0.0
        target_ids = inputs.get("tsp_two_opt_distill_completion_ids")
        target_mask = inputs.get("tsp_two_opt_distill_completion_mask")
        gains = inputs.get("tsp_two_opt_distill_gains")
        passes = inputs.get("tsp_two_opt_distill_passes")
        batch_size = int(completion_ids.size(0))
        stats = {
            "clasp/tsp_two_opt_distill_enabled": 1.0,
            "clasp/tsp_two_opt_distill_has_tensors": float(
                torch.is_tensor(target_ids) and torch.is_tensor(target_mask)
            ),
            "clasp/tsp_two_opt_distill_shape_ok": 0.0,
            "clasp/tsp_two_opt_distill_tensor_rows": float(
                int(target_ids.size(0)) if torch.is_tensor(target_ids) and target_ids.dim() > 0 else 0
            ),
            "clasp/tsp_two_opt_distill_batch_rows": float(batch_size),
            "clasp/tsp_two_opt_distill_valid_row_rate": 0.0,
            "clasp/tsp_two_opt_distill_nll": 0.0,
            "clasp/tsp_two_opt_distill_length": 0.0,
            "clasp/tsp_two_opt_distill_gain": 0.0,
            "clasp/tsp_two_opt_distill_passes": 0.0,
        }
        has_local_tensors = bool(
            torch.is_tensor(target_ids)
            and torch.is_tensor(target_mask)
            and target_ids.shape == target_mask.shape
            and target_ids.size(0) == batch_size
        )
        if has_local_tensors:
            stats["clasp/tsp_two_opt_distill_shape_ok"] = 1.0
            valid = target_mask.sum(dim=1) > 0
        else:
            valid = torch.zeros(batch_size, dtype=torch.bool, device=per_token_logps.device)

        # Rank-uniform participation (see _compute_improvement_distillation_losses):
        # a DDP forward is not collective-free, so all ranks must agree before
        # anyone runs the extra forward.
        local_any = valid.any().to(device=per_token_logps.device, dtype=torch.float32)
        try:
            global_any = bool(self.accelerator.gather(local_any.reshape(1)).sum().item() > 0)
        except Exception:
            global_any = bool(valid.any().item())
        if not global_any:
            return zero, stats

        # Memory: forward only the valid-row subset (see
        # _compute_improvement_distillation_losses); dummy 1-row forward keeps
        # the forward count rank-uniform when this rank has no targets.
        local_valid = bool(valid.any().item())
        if local_valid:
            sub_prompt_ids = prompt_ids[valid]
            sub_prompt_mask = prompt_mask[valid]
            sub_ids = target_ids[valid]
            sub_mask = target_mask[valid]
            width = max(int(sub_mask.sum(dim=1).max().item()), 1)
            sub_ids = sub_ids[:, :width]
            sub_mask = sub_mask[:, :width]
        else:
            sub_prompt_ids = prompt_ids[:1]
            sub_prompt_mask = prompt_mask[:1]
            sub_ids = torch.full(
                (1, 1),
                int(self._safe_pad_token_id()),
                dtype=torch.long,
                device=per_token_logps.device,
            )
            sub_mask = torch.zeros((1, 1), dtype=torch.long, device=per_token_logps.device)

        target_input_ids = torch.cat([sub_prompt_ids, sub_ids], dim=1)
        target_attention_mask = torch.cat([sub_prompt_mask, sub_mask], dim=1)
        target_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            target_input_ids,
            target_attention_mask,
            sub_ids.size(1),
            compute_entropy=True,
        )
        target_mask_float = sub_mask.to(dtype=target_per_token_logps.dtype)
        token_counts = target_mask_float.sum(dim=1).clamp(min=1.0)
        sequence_nll = -((target_per_token_logps * target_mask_float).sum(dim=1) / token_counts)
        if not local_valid:
            # Participate in the forward, contribute nothing to the loss.
            return target_per_token_logps.sum() * 0.0, stats
        loss = sequence_nll.mean()
        stats["clasp/tsp_two_opt_distill_valid_row_rate"] = float(valid.float().mean().item())
        stats["clasp/tsp_two_opt_distill_nll"] = float(sequence_nll.detach().mean().item())
        stats["clasp/tsp_two_opt_distill_length"] = float(token_counts.detach().mean().item())
        if torch.is_tensor(gains) and gains.numel() == batch_size:
            stats["clasp/tsp_two_opt_distill_gain"] = float(gains[valid].detach().mean().item())
        if torch.is_tensor(passes) and passes.numel() == batch_size:
            stats["clasp/tsp_two_opt_distill_passes"] = float(passes[valid].detach().mean().item())
        return loss, stats

    def _record_clasp_dual_feasibility(self, components: dict, fallback_rate: float) -> None:
        if not self.clasp_enabled:
            return
        wrappers = getattr(self.primal_dual_reward_manager, "wrappers", None)
        if not isinstance(wrappers, dict) or not wrappers:
            return

        task_names = sorted(str(task).lower() for task in wrappers.keys())
        index_by_task = {str(task).lower(): idx for idx, task in enumerate(task_names)}
        device = getattr(getattr(self, "accelerator", None), "device", None)
        sums = torch.zeros(len(task_names), dtype=torch.float32, device=device)
        counts = torch.zeros(len(task_names), dtype=torch.float32, device=device)

        task_types = components.get("task_types") if isinstance(components, dict) else None
        flags = components.get("feasibility_flags") if isinstance(components, dict) else None
        if isinstance(task_types, list) and isinstance(flags, list) and len(task_types) == len(flags):
            for task, flag in zip(task_types, flags):
                idx = index_by_task.get(str(task).lower())
                if idx is None:
                    continue
                sums[idx] += float(flag)
                counts[idx] += 1.0

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        except Exception:
            pass

        for idx, task_name in enumerate(task_names):
            count = float(counts[idx].item())
            rate = float(sums[idx].item() / count) if count > 0.0 else float(fallback_rate)
            wrapper = wrappers.get(task_name)
            if wrapper is None:
                wrapper = next(
                    (
                        candidate
                        for key, candidate in wrappers.items()
                        if str(key).lower() == task_name
                    ),
                    None,
                )
            if hasattr(wrapper, "record_batch_feasibility_rate"):
                wrapper.record_batch_feasibility_rate(rate)

    def _validate_fixed_kl_reference(self) -> None:
        """Reject TRL's drifting disable-adapter reference for biased LoRA."""

        if float(getattr(self, "beta", 0.0) or 0.0) == 0.0:
            return
        if getattr(self, "ref_model", None) is not None:
            return
        policy_model = getattr(self, "model", None)
        unwrap_model = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
        if callable(unwrap_model) and policy_model is not None:
            try:
                policy_model = unwrap_model(policy_model)
            except Exception:
                pass
        peft_configs = getattr(policy_model, "peft_config", None)
        if not isinstance(peft_configs, dict):
            return
        biased = []
        for adapter_name, config in peft_configs.items():
            bias = str(getattr(config, "bias", "none") or "none").strip().lower()
            if bias != "none":
                biased.append(f"{adapter_name}:{bias}")
        if biased:
            raise RuntimeError(
                "beta > 0 with a biased LoRA adapter requires an explicit frozen full-model "
                "KL reference. disable_adapter() does not restore trained base biases. "
                f"Biased adapters: {', '.join(biased)}"
            )

    def _generate_and_score_completions(self, inputs):
        self._validate_fixed_kl_reference()
        task_types_for_budget = self._extract_task_types(inputs)
        generation_budget_restore = self._set_clasp_generation_budget(task_types_for_budget)
        _log_gpu_memory("before_generate_and_score")
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            try:
                self._restore_configured_gradient_checkpointing()
            finally:
                self._restore_generation_budget(generation_budget_restore)
                _log_gpu_memory("after_generate_and_score")

        components = self._get_local_primal_dual_components()
        self._attach_reference_imitation_tensors(inputs, output, task_types_for_budget)
        self._attach_reference_contrastive_tensors(inputs, output, task_types_for_budget)
        self._attach_tsp_two_opt_distillation_tensors(inputs, output, task_types_for_budget)
        self._attach_improvement_distillation_tensors(inputs, output, task_types_for_budget)
        self._attach_self_replay_tensors(inputs, output, components, task_types_for_budget)
        replay_stats = self._observe_self_replay(inputs, output, components, task_types_for_budget)
        (
            reward_advantages_global,
            combined_advantages_global,
            gathered_weighted_gap_scores,
            gathered_raw_positive_gap_scores,
            gathered_feasibility_flags,
            local_len,
            stats,
        ) = self._recompute_constrained_terms(components)

        process_index = int(self.accelerator.process_index)
        global_len = int(combined_advantages_global.numel())
        expected_global_len = local_len * self.accelerator.num_processes
        if global_len != expected_global_len:
            raise RuntimeError(
                "Constrained-GRPO gathered tensor size mismatch across ranks. "
                f"global_len={global_len}, expected={expected_global_len} "
                f"(local_len={local_len} x num_processes={self.accelerator.num_processes}). "
                "This indicates uneven sample counts across ranks."
            )
        process_slice = slice(process_index * local_len, (process_index + 1) * local_len)
        output["reward_advantages"] = reward_advantages_global[process_slice]
        output["weighted_gap_scores"] = gathered_weighted_gap_scores[process_slice]
        output["raw_positive_gap_scores"] = gathered_raw_positive_gap_scores[process_slice]
        output["feasibility_flags"] = gathered_feasibility_flags[process_slice]
        output["advantages"] = combined_advantages_global[process_slice]
        self._attach_clasp_group_metadata(output, components, local_len)

        self._replace_logged_advantages(combined_advantages_global)

        mode = "train" if self.model.training else "eval"
        for metric_name, value in self._clasp_last_generation_budget_stats.items():
            self._append_metric(
                mode,
                f"constrained_grpo/{metric_name}",
                value,
            )
        self._append_metric(mode, "constrained_grpo/mean_primal_reward", stats["mean_primal"])
        self._append_metric(
            mode,
            "constrained_grpo/mean_objective_quality",
            stats["mean_objective_quality"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/objective_quality_fallback",
            stats["objective_quality_fallback"],
        )
        self._append_metric(mode, "constrained_grpo/mean_weighted_gap", stats["mean_weighted_gap"])
        self._append_metric(
            mode,
            "constrained_grpo/mean_positive_weighted_gap",
            stats["mean_positive_weighted_gap"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/mean_raw_positive_gap",
            stats["mean_raw_positive_gap"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/mean_dual_aware_recovery_gap",
            stats["mean_dual_aware_recovery_gap"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/mean_active_recovery_gap",
            stats["mean_active_recovery_gap"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/component_dual_recovery_scale",
            stats["component_dual_recovery_scale"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/component_dual_recovery_contribution",
            stats["component_dual_recovery_contribution"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/component_dual_recovery_active",
            stats["component_dual_recovery_active"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/component_dual_recovery_api_available",
            stats["component_dual_recovery_api_available"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/frontier_cdf_uses_max_violation_gap",
            stats["frontier_cdf_uses_max_violation_gap"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/mean_penalty_weighted_gap",
            stats["mean_penalty_weighted_gap"],
        )
        self._append_metric(mode, "constrained_grpo/penalty_scale", stats["penalty_scale"])
        self._append_metric(
            mode,
            "constrained_grpo/positive_constraint_penalty",
            stats["positive_constraint_penalty"],
        )
        self._append_metric(mode, "constrained_grpo/mean_violation", stats["mean_violation"])
        self._append_metric(
            mode,
            "constrained_grpo/candidate_feasibility_rate",
            stats["candidate_feasibility_rate"],
        )
        self._append_metric(mode, "constrained_grpo/mean_constraint", stats["mean_constraint"])
        self._append_metric(
            mode,
            "constrained_grpo/mean_reward_advantage",
            stats["mean_reward_advantage"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/std_reward_advantage",
            stats["std_reward_advantage"],
        )
        self._append_metric(
            mode,
            "constrained_grpo/mean_combined_advantage",
            stats["mean_combined_advantage"],
        )
        self._record_clasp_dual_feasibility(components, stats["candidate_feasibility_rate"])
        self._update_clasp_validation_feasibility(mode, components)
        for metric_name, value in replay_stats.items():
            self._append_metric(mode, f"constrained_grpo/{metric_name}", value)
        for metric_name in (
            "lexicographic_group_feasible_rate",
            "lexicographic_mixed_group_rate",
            "lexicographic_all_infeasible_group_rate",
            "dual_primal/group_scalarization",
            "dual_primal/group_std_correction",
            "dual_primal/scalarized_score_mean",
            "dual_primal/scalarized_score_std",
            "dual_primal/separate_z",
            "dual_primal/separate_z_reward_z_std",
            "dual_primal/separate_z_penalty_z_std",
        ):
            if metric_name in stats:
                self._append_metric(mode, f"constrained_grpo/{metric_name}", stats[metric_name])
        for metric_name, value in stats.items():
            if (
                str(metric_name).startswith("clasp/bon_")
                or str(metric_name).startswith("clasp/bon_lambda/")
                or str(metric_name).startswith("clasp/canonical_")
                or str(metric_name).startswith("frontier_cdf/")
            ):
                self._append_metric(mode, f"constrained_grpo/{metric_name}", value)
        # ------------------- CLASP metrics (no-op when disabled) -------------------
        for metric_name in (
            "clasp/mean_alpha",
            "clasp/std_alpha",
            "clasp/mean_frontier_size",
            "clasp/all_infeasible_group_rate",
            "clasp/fully_feasible_group_rate",
            "clasp/advantage_l2",
            "clasp/advantage_abs_mean",
            "clasp/primal_score_gap_penalty",
            "clasp/primal_rank_gap_mean",
            "clasp/hard_group_weight_scale",
            "clasp/hard_group_weight_threshold",
            "clasp/hard_group_weight_max",
            "clasp/hard_group_weight_mean",
            "clasp/hard_group_weight_max_observed",
            "clasp/hard_group_active_rate",
            "clasp/hard_group_min_raw_gap_mean",
            "clasp/feasibility_warmup",
            "clasp/feasibility_warmup_active_rate",
            "clasp/feasibility_warmup_group_active_rate",
            "clasp/feasibility_warmup_all_infeasible_group_rate",
            "clasp/feasibility_warmup_mixed_group_rate",
            "clasp/feasibility_warmup_all_feasible_group_rate",
            "clasp/feasibility_warmup_margin",
            "clasp/feasibility_warmup_violation_scale",
            "clasp/feasibility_warmup_rank_scale",
            "clasp/feasibility_warmup_advantage_abs_mean",
            "clasp/feasibility_warmup_task_active_count",
            "clasp/feasibility_warmup_task_released_count",
        ):
            if metric_name in stats:
                self._append_metric(mode, f"constrained_grpo/{metric_name}", stats[metric_name])
        self._append_metric(
            mode,
            "constrained_grpo/effective_penalty_pressure",
            stats["effective_penalty_pressure"],
        )
        # Per-constraint lambda traces when the manager exposes wrappers.
        try:
            wrappers = getattr(self.primal_dual_reward_manager, "wrappers", None)
            if isinstance(wrappers, dict):
                for task_name, wrapper in wrappers.items():
                    names = getattr(wrapper, "cost_names", None) or []
                    lams = getattr(wrapper, "dual_lambdas", None)
                    if not names or lams is None:
                        continue
                    for name, lam in zip(names, lams):
                        self._append_metric(
                            mode,
                            f"PD/lambda/{task_name}/{name}",
                            float(lam),
                        )
        except Exception:
            pass
        # Feed the adaptive length estimator when CLASP is active.
        if self.clasp_enabled and self.clasp_adaptive_length is not None:
            try:
                self._observe_clasp_completion_lengths(components, output)
                step = getattr(getattr(self, "state", None), "global_step", 0)
                self.clasp_adaptive_length.maybe_refresh(int(step))
                for task_name, budget in self.clasp_adaptive_length.snapshot().items():
                    self._append_metric(
                        mode,
                        f"clasp/adaptive_budget/{task_name}",
                        float(budget),
                    )
            except Exception:
                pass

        self._append_local_per_task_metrics(mode, components)

        self._validate_trl_generation_batch_rows(output)
        return output

    def _restore_configured_gradient_checkpointing(self) -> None:
        """Undo TRL 0.24's loss of checkpointing kwargs after generation.

        ``unwrap_model_for_generation`` temporarily disables checkpointing,
        then calls ``gradient_checkpointing_enable()`` without the configured
        kwargs. Transformers consequently installs ``use_reentrant=True``.
        With two loss forwards before backward, that changes DDP reduction
        behavior and can silently leave rank-local gradients. Reapply the
        declared training configuration for both ordinary and modified GRPO,
        including when generation exits exceptionally.
        """
        args = getattr(self, "args", None)
        if not bool(getattr(args, "gradient_checkpointing", False)):
            return
        model = self.accelerator.unwrap_model(self.model)
        kwargs = getattr(args, "gradient_checkpointing_kwargs", None)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=None if kwargs is None else dict(kwargs)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _log_gpu_memory("compute_loss_start")
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        if "weighted_gap_scores" not in inputs:
            raise RuntimeError(
                "PrimalDualGRPOTrainer.compute_loss: 'weighted_gap_scores' missing from inputs. "
                "This means _generate_and_score_completions did not inject the constrained terms, "
                "or TRL's internal batching dropped the custom key. "
                f"Available keys: {sorted(inputs.keys())}"
            )
        weighted_gap_scores = inputs["weighted_gap_scores"]
        lagrangian_advantages = inputs["advantages"]

        old_per_token_logps = inputs.get("old_per_token_logps")
        has_old_policy_logps = old_per_token_logps is not None
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        lagrangian_loss_1 = coef_1 * lagrangian_advantages.unsqueeze(1)
        lagrangian_loss_2 = coef_2 * lagrangian_advantages.unsqueeze(1)
        lagrangian_per_token_loss = -torch.min(lagrangian_loss_1, lagrangian_loss_2)
        if entropy_mask is not None:
            lagrangian_per_token_loss = lagrangian_per_token_loss * entropy_mask

        policy_per_token_loss = lagrangian_per_token_loss

        if self.use_vllm and self.vllm_importance_sampling_correction:
            policy_per_token_loss = (
                policy_per_token_loss * inputs["importance_sampling_ratio"]
            )

        per_token_loss = policy_per_token_loss
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        set_credit_enabled = bool(
            getattr(self, "clasp_canonical_set", False)
            or getattr(self, "clasp_fixed_null_set", False)
        )
        clasp_policy_length_weights = None
        clasp_completion_lengths = None
        if self.loss_type == "grpo" and set_credit_enabled:
            fixed_horizon_value = getattr(self, "max_completion_length", None)
            if fixed_horizon_value is None:
                fixed_horizon_value = getattr(self.args, "max_completion_length", 0)
            fixed_horizon = int(fixed_horizon_value or 0)
            loss, clasp_policy_length_weights, clasp_completion_lengths = (
                reduce_clasp_policy_loss(
                    policy_per_token_loss,
                    completion_mask,
                    length_alpha=float(self.clasp_policy_length_alpha),
                    max_completion_length=fixed_horizon,
                )
            )
            if self.beta != 0.0:
                # Length alpha changes only the policy-gradient estimator.  Keep
                # the KL contract identical to historical GRPO token means.
                sequence_kl = (per_token_kl * completion_mask).sum(-1) / (
                    completion_mask.sum(-1).clamp(min=1.0)
                )
                loss = loss + self.beta * sequence_kl.mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            fixed_horizon = getattr(self, "max_completion_length", None)
            if fixed_horizon is None:
                fixed_horizon = getattr(self.args, "max_completion_length", 0)
            if int(fixed_horizon) <= 0:
                raise ValueError("Dr.GRPO requires a positive fixed max_completion_length")
            # The configured horizon must not vary with the longest sampled
            # response or with how TRL pads a minibatch.
            loss_denominator = per_token_loss.size(0) * int(fixed_horizon)
            loss = (per_token_loss * completion_mask).sum() / loss_denominator
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        clasp_pairwise_loss = None
        clasp_pairwise_stats: dict[str, float] = {}
        clasp_elite_imitation_loss = None
        clasp_elite_imitation_stats: dict[str, float] = {}
        clasp_multi_elite_imitation_loss = None
        clasp_multi_elite_imitation_stats: dict[str, float] = {}
        clasp_reference_imitation_loss = None
        clasp_reference_imitation_stats: dict[str, float] = {}
        clasp_reference_contrastive_loss = None
        clasp_reference_contrastive_stats: dict[str, float] = {}
        clasp_tsp_two_opt_distillation_loss = None
        clasp_tsp_two_opt_distillation_stats: dict[str, float] = {}
        clasp_self_replay_imitation_loss = None
        clasp_self_replay_imitation_stats: dict[str, float] = {}
        clasp_self_replay_contrastive_loss = None
        clasp_self_replay_contrastive_stats: dict[str, float] = {}
        group_auxiliary_enabled = bool(
            self.model.training
            and self.clasp_enabled
            and (
                self.clasp_pairwise_loss_weight > 0.0
                or self.clasp_elite_imitation_loss_weight > 0.0
                or self.clasp_multi_elite_imitation_loss_weight > 0.0
            )
        )
        group_permutation = None
        group_auxiliary_reason = "disabled"
        if group_auxiliary_enabled:
            group_permutation, group_auxiliary_reason = self._group_auxiliary_permutation(
                inputs,
                int(completion_ids.size(0)),
            )
        grouped_advantages = (
            lagrangian_advantages[group_permutation]
            if group_permutation is not None
            else None
        )
        grouped_per_token_logps = (
            per_token_logps[group_permutation]
            if group_permutation is not None
            else None
        )
        grouped_completion_mask = (
            completion_mask[group_permutation]
            if group_permutation is not None
            else None
        )
        if (
            self.model.training
            and self.clasp_enabled
            and self.clasp_pairwise_loss_weight > 0.0
            and group_permutation is not None
        ):
            from clasp.rewards.clasp_math import compute_pairwise_preference_loss

            sequence_log_ratios = log_importance_weights.squeeze(-1)[group_permutation]
            clasp_pairwise_loss, clasp_pairwise_stats = compute_pairwise_preference_loss(
                sequence_log_ratios,
                grouped_advantages,
                num_generations=self.num_generations,
                margin=self.clasp_pairwise_margin,
                advantage_delta=self.clasp_pairwise_advantage_delta,
            )
            loss = loss + (
                self.clasp_pairwise_loss_weight
                * clasp_pairwise_loss
                / self.current_gradient_accumulation_steps
            )
        if (
            self.model.training
            and self.clasp_enabled
            and self.clasp_elite_imitation_loss_weight > 0.0
            and group_permutation is not None
        ):
            from clasp.rewards.clasp_math import compute_elite_imitation_loss

            clasp_elite_imitation_loss, clasp_elite_imitation_stats = compute_elite_imitation_loss(
                grouped_per_token_logps,
                grouped_completion_mask,
                grouped_advantages,
                num_generations=self.num_generations,
                advantage_delta=self.clasp_elite_imitation_advantage_delta,
            )
            loss = loss + (
                self.clasp_elite_imitation_loss_weight
                * clasp_elite_imitation_loss
                / self.current_gradient_accumulation_steps
            )
        if (
            self.model.training
            and self.clasp_enabled
            and self.clasp_multi_elite_imitation_loss_weight > 0.0
            and group_permutation is not None
        ):
            from clasp.rewards.clasp_math import compute_multi_elite_imitation_loss

            raw_positive_gap_scores = inputs.get("raw_positive_gap_scores")
            feasibility_flags = inputs.get("feasibility_flags")
            if not (torch.is_tensor(raw_positive_gap_scores) and torch.is_tensor(feasibility_flags)):
                raise RuntimeError(
                    "CLASP multi-elite imitation requires raw_positive_gap_scores and "
                    "feasibility_flags in the GRPO batch."
                )
            clasp_multi_elite_imitation_loss, clasp_multi_elite_imitation_stats = (
                compute_multi_elite_imitation_loss(
                    grouped_per_token_logps,
                    grouped_completion_mask,
                    grouped_advantages,
                    raw_positive_gap_scores[group_permutation],
                    feasibility_flags[group_permutation],
                    num_generations=self.num_generations,
                    elite_fraction=self.clasp_multi_elite_imitation_fraction,
                    max_raw_gap=self.clasp_multi_elite_imitation_max_raw_gap,
                    advantage_delta=self.clasp_multi_elite_imitation_advantage_delta,
                    temperature=self.clasp_multi_elite_imitation_temperature,
                    require_feasible=self.clasp_multi_elite_imitation_require_feasible,
                    fallback_to_best_feasible=(
                        self.clasp_multi_elite_imitation_fallback_to_best_feasible
                    ),
                )
            )
            loss = loss + (
                self.clasp_multi_elite_imitation_loss_weight
                * clasp_multi_elite_imitation_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_reference_imitation_loss, clasp_reference_imitation_stats = (
            self._compute_reference_imitation_loss(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_reference_imitation_loss is not None:
            loss = loss + (
                self.clasp_reference_imitation_loss_weight
                * clasp_reference_imitation_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_reference_contrastive_loss, clasp_reference_contrastive_stats = (
            self._compute_reference_contrastive_loss(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_reference_contrastive_loss is not None:
            loss = loss + (
                self.clasp_reference_contrastive_loss_weight
                * clasp_reference_contrastive_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_self_replay_imitation_loss, clasp_self_replay_imitation_stats = (
            self._compute_self_replay_imitation_loss(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_self_replay_imitation_loss is not None:
            loss = loss + (
                self.clasp_self_replay_imitation_loss_weight
                * clasp_self_replay_imitation_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_self_replay_contrastive_loss, clasp_self_replay_contrastive_stats = (
            self._compute_self_replay_contrastive_loss(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_self_replay_contrastive_loss is not None:
            loss = loss + (
                self.clasp_self_replay_contrastive_loss_weight
                * clasp_self_replay_contrastive_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_tsp_two_opt_distillation_loss, clasp_tsp_two_opt_distillation_stats = (
            self._compute_tsp_two_opt_distillation_loss(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_tsp_two_opt_distillation_loss is not None:
            loss = loss + (
                self.clasp_tsp_two_opt_distill_loss_weight
                * clasp_tsp_two_opt_distillation_loss
                / self.current_gradient_accumulation_steps
            )
        clasp_improvement_distillation_loss, clasp_improvement_distillation_stats = (
            self._compute_improvement_distillation_losses(
                model,
                inputs,
                prompt_ids,
                prompt_mask,
                completion_ids,
                per_token_logps,
            )
        )
        if clasp_improvement_distillation_loss is not None:
            # Per-spec weights are already applied inside the loss.
            loss = loss + (
                clasp_improvement_distillation_loss
                / self.current_gradient_accumulation_steps
            )

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if clasp_policy_length_weights is not None and clasp_completion_lengths is not None:
            gathered_weights = self.accelerator.gather(
                clasp_policy_length_weights.detach()
            )
            gathered_lengths = self.accelerator.gather(
                clasp_completion_lengths.detach()
            )
            fixed_horizon_value = getattr(self, "max_completion_length", None)
            if fixed_horizon_value is None:
                fixed_horizon_value = getattr(self.args, "max_completion_length", 0)
            fixed_horizon = float(int(fixed_horizon_value))
            self._metrics[mode][
                "constrained_grpo/clasp_policy_length_alpha"
            ].append(float(self.clasp_policy_length_alpha))
            self._metrics[mode][
                "constrained_grpo/clasp_policy_length_weight_mean"
            ].append(gathered_weights.mean().item())
            self._metrics[mode][
                "constrained_grpo/clasp_policy_length_weight_min"
            ].append(gathered_weights.min().item())
            self._metrics[mode][
                "constrained_grpo/clasp_policy_length_weight_max"
            ].append(gathered_weights.max().item())
            self._metrics[mode][
                "constrained_grpo/clasp_completion_length_mean"
            ].append(gathered_lengths.mean().item())
            self._metrics[mode][
                "constrained_grpo/clasp_completion_horizon_fraction_mean"
            ].append((gathered_lengths / fixed_horizon).mean().item())

        if group_auxiliary_enabled:
            group_aux_valid = per_token_logps.new_tensor(
                1.0 if group_permutation is not None else 0.0
            )
            self._metrics[mode]["constrained_grpo/clasp_group_aux_complete"].append(
                self.accelerator.gather(group_aux_valid).nanmean().item()
            )
            if group_permutation is None:
                skipped = per_token_logps.new_tensor(1.0)
                self._metrics[mode]["constrained_grpo/clasp_group_aux_skipped"].append(
                    self.accelerator.gather(skipped).nanmean().item()
                )

        if clasp_pairwise_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_pairwise_loss"].append(
                self.accelerator.gather(clasp_pairwise_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_pairwise_stats.items():
                metric_tensor = sequence_log_ratios.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_elite_imitation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_elite_imitation_loss"].append(
                self.accelerator.gather(clasp_elite_imitation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_elite_imitation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_multi_elite_imitation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_multi_elite_imitation_loss"].append(
                self.accelerator.gather(clasp_multi_elite_imitation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_multi_elite_imitation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_reference_imitation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_reference_imitation_loss"].append(
                self.accelerator.gather(clasp_reference_imitation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_reference_imitation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_reference_contrastive_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_reference_contrastive_loss"].append(
                self.accelerator.gather(clasp_reference_contrastive_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_reference_contrastive_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_self_replay_imitation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_self_replay_imitation_loss"].append(
                self.accelerator.gather(clasp_self_replay_imitation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_self_replay_imitation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_self_replay_contrastive_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_self_replay_contrastive_loss"].append(
                self.accelerator.gather(clasp_self_replay_contrastive_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_self_replay_contrastive_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_tsp_two_opt_distillation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_tsp_two_opt_distillation_loss"].append(
                self.accelerator.gather(clasp_tsp_two_opt_distillation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_tsp_two_opt_distillation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )
        if clasp_improvement_distillation_loss is not None:
            self._metrics[mode]["constrained_grpo/clasp_improvement_distillation_loss"].append(
                self.accelerator.gather(clasp_improvement_distillation_loss.detach()).nanmean().item()
            )
            for metric_name, value in clasp_improvement_distillation_stats.items():
                metric_tensor = per_token_logps.new_tensor(float(value))
                self._metrics[mode][f"constrained_grpo/{metric_name}"].append(
                    self.accelerator.gather(metric_tensor).nanmean().item()
                )

        mean_weighted_gap = weighted_gap_scores.mean()
        self._metrics[mode]["constrained_grpo/mean_weighted_gap_term"].append(
            self.accelerator.gather(mean_weighted_gap).nanmean().item()
        )

        if has_old_policy_logps:
            # These are policy-ratio clipping metrics for the constrained GRPO surrogate.
            # They are distinct from TRL's completions/clipped_ratio, which tracks generation truncation.
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (lagrangian_advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (lagrangian_advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["policy_clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["policy_clip_ratio/low_min"].append(
                gathered_low_clip.nan_to_num(nan=float("inf")).min().item()
            )
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["policy_clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["policy_clip_ratio/high_max"].append(
                gathered_high_clip.nan_to_num(nan=float("-inf")).max().item()
            )
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["policy_clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        if (
            mode == "train"
            and self.clasp_enabled
            and self.clasp_bon_aligned
            and bool(getattr(self.accelerator, "sync_gradients", False))
        ):
            commit_stats = self._commit_pending_bon_controller_update(
                device=per_token_logps.device,
            )
            for metric_name, value in commit_stats.items():
                self._append_metric(
                    "train",
                    f"constrained_grpo/{metric_name}",
                    value,
                )

        return loss
