"""Immutable campaign profiles for the two approved corrected-FOARL lanes.

Neither profile is an official Jiang et al. reproduction.  The native lane
keeps the released f638 hyperparameters and the row-derived, single-process
horizon.  The matched-B9 lane is a common-attribution port: it changes the
rollout batch and compute/data schedule to the CLASP matched lane while keeping
the corrected f638 SFT initialization, prompt, rewards, and loss semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from .llmcosolver_f638_training import (
    FOARL_F638_SEMANTICS,
    FoarlF638CorrectedRuntimeProfile,
    FoarlF638Semantics,
)


FOARL_F638_NATIVE = "foarl_f638_native"
FOARL_F638_MATCHED_B9 = "foarl_f638_matched_b9"
FOARL_F638_CAMPAIGN_METHODS = (FOARL_F638_NATIVE, FOARL_F638_MATCHED_B9)

NATIVE_SOURCE_ROWS_BY_TASK = MappingProxyType({
    "cvrp": 2_000,
    "jssp": 3_200,
    "mis": 3_200,
    "op": 5_000,
    "tsp": 3_200,
})
NATIVE_FINAL_STEP_BY_TASK = MappingProxyType({
    "cvrp": 250,
    "jssp": 400,
    "mis": 400,
    "op": 625,
    "tsp": 400,
})


@dataclass(frozen=True)
class FoarlF638CampaignProfile:
    """One fail-closed scheduler/advancement contract."""

    method_id: str
    arm_contract_schema: str
    attribution: str
    semantics: FoarlF638Semantics
    world_size: int
    generation_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    optimizer: str
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    eval_from_train_split: bool
    max_eval_samples_per_task: int
    min_train_samples_per_task: int
    min_eval_samples_per_task: int
    fixed_final_step: int | None
    native_final_step_by_task: Mapping[str, int]
    native_source_rows_by_task: Mapping[str, int]
    required_train_seed: int | None

    @property
    def runtime(self) -> FoarlF638CorrectedRuntimeProfile:
        return FoarlF638CorrectedRuntimeProfile(semantics=self.semantics)

    def final_step(self, task: str) -> int:
        normalized = str(task).strip().lower()
        if normalized not in self.semantics.supported_tasks:
            raise ValueError(f"unsupported FOARL task {task!r}")
        if self.fixed_final_step is not None:
            return self.fixed_final_step
        return int(self.native_final_step_by_task[normalized])

    def source_rows(self, task: str) -> int | None:
        normalized = str(task).strip().lower()
        if normalized not in self.semantics.supported_tasks:
            raise ValueError(f"unsupported FOARL task {task!r}")
        value = self.native_source_rows_by_task.get(normalized)
        return int(value) if value is not None else None

    def generation_contract(self) -> dict[str, Any]:
        return {
            "num_generations": self.semantics.num_generations,
            "generation_batch_size": self.generation_batch_size,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "max_prompt_length": 20_000,
            "max_completion_length": 1_000,
        }

    def training_contract(self, *, task: str, train_seed: int) -> dict[str, Any]:
        normalized = str(task).strip().lower()
        self.final_step(normalized)
        if self.required_train_seed is not None and train_seed != self.required_train_seed:
            raise ValueError(
                f"{self.method_id} requires released seed={self.required_train_seed}, "
                f"got {train_seed}"
            )
        source_rows = self.source_rows(normalized)
        return {
            "entrypoint": "clasp.entrypoints.train_foarl_f638_campaign",
            "arm_contract_schema": self.arm_contract_schema,
            "attribution": self.attribution,
            "reproduction_claim": "not_an_official_reproduction",
            "prompt_style": "jiang_f638",
            "prompt_template_sha256": self.semantics.prompt_sha256,
            "no_reason": False,
            "reward_heads": "optimality,feasibility",
            "reward_weights": "1.0,1.0",
            "reward_aggregation": "sum",
            "loss_type": "grpo",
            "token_aggregation": self.semantics.token_aggregation,
            "scale_rewards": "group",
            "std_estimator": "sample_unbiased",
            "advantage_epsilon": 0.0001,
            "num_iterations": 1,
            "sync_ref_model": False,
            "importance_sampling_level": "token",
            "beta": 0.05,
            "epsilon": 0.2,
            "epsilon_high": 0.2,
            "reference_policy": "fixed_corrected_sft_step_zero",
            "use_primal_dual": False,
            "use_clasp": False,
            "dtype": "bfloat16",
            "lora_r": 64,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "lora_bias": "all",
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": 0.000001,
            "optim": self.optimizer,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "eval_from_train_split": self.eval_from_train_split,
            "max_samples_per_task": 0,
            "max_eval_samples_per_task": self.max_eval_samples_per_task,
            "min_train_samples_per_task": self.min_train_samples_per_task,
            "min_eval_samples_per_task": self.min_eval_samples_per_task,
            "eval_ratio": 0.05,
            "prefer_subset_files": False,
            "train_size_bucket": "all",
            "source_rows": source_rows if source_rows is not None else "registry_bound",
            "data_order": "registered_rows_seeded_shuffle",
            "data_seed": train_seed,
            "hardware_contract": (
                "sealed_a100_single_process"
                if self.world_size == 1
                else "sealed_4xa100_clasp_matched"
            ),
            "token_budget_contract": "max_prompt_20000_max_completion_1000",
        }

    def tracked_environment_contract(self, *, task: str, train_seed: int) -> dict[str, str]:
        normalized = str(task).strip().lower()
        self.final_step(normalized)
        return {
            "CLASP_PUBLICATION_FIDELITY": "1",
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
            "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1" if normalized == "mis" else "0",
            "CLASP_AUX_GROUPING_STRICT": "1",
            "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
            "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
            "GRPO_DATA_SEED": str(train_seed),
            "GRPO_STRICT_OPTIMIZER_RESUME": "1",
        }


FOARL_F638_NATIVE_SEMANTICS = replace(
    FOARL_F638_SEMANTICS,
    method_id=FOARL_F638_NATIVE,
    profile_id="foarl-f638-native-corrected-campaign-v1",
    publication_label="FOARL-f638-native corrected port (not an official reproduction)",
)
FOARL_F638_MATCHED_B9_SEMANTICS = replace(
    FOARL_F638_SEMANTICS,
    method_id=FOARL_F638_MATCHED_B9,
    profile_id="foarl-f638-matched-b9-common-attribution-v1",
    publication_label="FOARL-f638 matched-B9 common-attribution port",
    num_generations=9,
)

FOARL_F638_CAMPAIGN_PROFILES: Mapping[str, FoarlF638CampaignProfile] = MappingProxyType({
    FOARL_F638_NATIVE: FoarlF638CampaignProfile(
        method_id=FOARL_F638_NATIVE,
        arm_contract_schema="foarl-f638-native-corrected-a100-v1",
        attribution="corrected_f638_native_hyperparameter_port",
        semantics=FOARL_F638_NATIVE_SEMANTICS,
        world_size=1,
        generation_batch_size=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        optimizer="adamw_torch",
        weight_decay=0.02,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        eval_from_train_split=False,
        max_eval_samples_per_task=300,
        min_train_samples_per_task=1,
        min_eval_samples_per_task=300,
        fixed_final_step=None,
        native_final_step_by_task=NATIVE_FINAL_STEP_BY_TASK,
        native_source_rows_by_task=NATIVE_SOURCE_ROWS_BY_TASK,
        required_train_seed=42,
    ),
    FOARL_F638_MATCHED_B9: FoarlF638CampaignProfile(
        method_id=FOARL_F638_MATCHED_B9,
        arm_contract_schema="foarl-f638-matched-b9-common-attribution-a100-v1",
        attribution="common_attribution_port_never_reproduction",
        semantics=FOARL_F638_MATCHED_B9_SEMANTICS,
        world_size=4,
        generation_batch_size=36,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=9,
        gradient_accumulation_steps=3,
        optimizer="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        eval_from_train_split=True,
        max_eval_samples_per_task=100,
        min_train_samples_per_task=1_900,
        min_eval_samples_per_task=100,
        fixed_final_step=3_200,
        native_final_step_by_task=MappingProxyType({}),
        native_source_rows_by_task=MappingProxyType({}),
        required_train_seed=None,
    ),
})

FOARL_F638_NATIVE_RUNTIME = FOARL_F638_CAMPAIGN_PROFILES[FOARL_F638_NATIVE].runtime
FOARL_F638_MATCHED_B9_RUNTIME = FOARL_F638_CAMPAIGN_PROFILES[FOARL_F638_MATCHED_B9].runtime


def get_foarl_f638_campaign_profile(method: str) -> FoarlF638CampaignProfile:
    try:
        return FOARL_F638_CAMPAIGN_PROFILES[str(method).strip()]
    except KeyError as exc:
        raise ValueError(
            f"FOARL campaign method must be one of {FOARL_F638_CAMPAIGN_METHODS}, "
            f"got {method!r}"
        ) from exc


def _require_fields(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    differences = {
        key: {"expected": expected_value, "observed": observed.get(key)}
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    }
    if differences:
        raise ValueError(f"{label} differs from immutable FOARL lane: {differences}")


def validate_foarl_f638_campaign_contract(
    *,
    method: str,
    task: str,
    train_seed: int,
    final_step: int,
    world_size: int,
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    environment: Mapping[str, str],
) -> FoarlF638CampaignProfile:
    """Validate a static run specification before any scheduler/model action."""

    profile = get_foarl_f638_campaign_profile(method)
    normalized_task = str(task).strip().lower()
    expected_final = profile.final_step(normalized_task)
    if final_step != expected_final:
        raise ValueError(
            f"{method}/{normalized_task} final_step must equal {expected_final}, got {final_step}"
        )
    if world_size != profile.world_size:
        raise ValueError(
            f"{method} world_size must equal {profile.world_size}, got {world_size}"
        )
    _require_fields(
        generation_config,
        profile.generation_contract(),
        label=f"{method} generation configuration",
    )
    _require_fields(
        config,
        profile.training_contract(task=normalized_task, train_seed=train_seed),
        label=f"{method} training configuration",
    )
    _require_fields(
        environment,
        profile.tracked_environment_contract(task=normalized_task, train_seed=train_seed),
        label=f"{method} tracked environment",
    )
    beta = config.get("beta")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isclose(
        float(beta), 0.05, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{method} requires beta=0.05 and a fixed corrected-SFT reference")
    return profile


__all__ = [
    "FOARL_F638_CAMPAIGN_METHODS",
    "FOARL_F638_CAMPAIGN_PROFILES",
    "FOARL_F638_MATCHED_B9",
    "FOARL_F638_MATCHED_B9_RUNTIME",
    "FOARL_F638_NATIVE",
    "FOARL_F638_NATIVE_RUNTIME",
    "NATIVE_FINAL_STEP_BY_TASK",
    "NATIVE_SOURCE_ROWS_BY_TASK",
    "FoarlF638CampaignProfile",
    "get_foarl_f638_campaign_profile",
    "validate_foarl_f638_campaign_contract",
]
