"""Fail-closed shared-design contract for the corrected-FOARL comparison.

The native corrected-FOARL lane is descriptive only.  Confirmatory comparison
is permitted only between ``foarl_f638_matched_b9`` and the distinct CLASP
comparator defined here.  Both lanes share the data, initialization, hardware,
batch, optimization-schedule, and 20k/1k token budget; their algorithm-specific
prompt, reward, credit, controller, and reference-policy fields remain explicit
and are *not* described as an official Jiang et al. reproduction.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .foarl_f638_campaign import FOARL_F638_MATCHED_B9


CLASP_FOARL_MATCHED_B9 = "clasp_canonical_foarl_matched_b9"
FOARL_MATCHED_B9_CONFIRMATION_PAIR = (
    FOARL_F638_MATCHED_B9,
    CLASP_FOARL_MATCHED_B9,
)
FOARL_MATCHED_B9_COMPARISON_PROTOCOL = (
    "foarl-f638-clasp-shared-compute-data-token-design-v1"
)
CLASP_FOARL_MATCHED_B9_ARM_SCHEMA = (
    "clasp-canonical-foarl-matched-b9-20k1k-a100-v1"
)


SHARED_GENERATION_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "num_generations": 9,
        "generation_batch_size": 36,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_prompt_length": 20_000,
        "max_completion_length": 1_000,
    }
)

SHARED_TRAINING_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "dtype": "bfloat16",
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_bias": "all",
        "per_device_train_batch_size": 3,
        "per_device_eval_batch_size": 9,
        "gradient_accumulation_steps": 3,
        "learning_rate": 0.000001,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "eval_from_train_split": True,
        "max_samples_per_task": 0,
        "max_eval_samples_per_task": 100,
        "min_train_samples_per_task": 1_900,
        "min_eval_samples_per_task": 100,
        "eval_ratio": 0.05,
        "prefer_subset_files": False,
        "train_size_bucket": "all",
        "source_rows": "registry_bound",
        "data_order": "registered_rows_seeded_shuffle",
        "hardware_contract": "sealed_4xa100_clasp_matched",
        "token_budget_contract": "max_prompt_20000_max_completion_1000",
        "logging_steps": 1,
        "save_steps": 400,
        "eval_steps": 100,
        "save_total_limit": 8,
        "final_max_steps": 3_200,
        "checkpoint_fidelity": "publication_exact_required",
    }
)

SHARED_ENVIRONMENT_CONTRACT: Mapping[str, str] = MappingProxyType(
    {
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
        "CLASP_AUX_GROUPING_STRICT": "1",
        "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
        "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
        "GRPO_STRICT_OPTIMIZER_RESUME": "1",
    }
)

# This is recorded in a frozen campaign manifest when the FOARL comparison is
# selected.  Unlike the canonical-arm allowlist, it names the exact projection
# whose equality is hashed after both complete lane validators have run.
FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST = (
    "projection:algorithm_specific_fields_excluded_only_after_lane_validation",
    "training_config:shared_fields="
    + ",".join(SHARED_TRAINING_CONTRACT),
    "tracked_environment:shared_fields="
    + ",".join(SHARED_ENVIRONMENT_CONTRACT)
    + ",CLASP_MIS_JIANG_RESPONSE_OUTPUT,GRPO_DATA_SEED",
    "generation:all_fields_shared",
    "fixed_kl_reference:algorithm_specific_but_source_must_equal_initial_model",
)


def is_foarl_matched_b9_confirmation_pair(methods: Sequence[str]) -> bool:
    return (
        len(methods) == 2
        and tuple(methods) == FOARL_MATCHED_B9_CONFIRMATION_PAIR
    )


def _require_fields(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    differences = {
        key: {"expected": expected_value, "observed": observed.get(key)}
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    }
    if differences:
        raise ValueError(f"{label} differs from the matched-B9 shared design: {differences}")


def validate_foarl_matched_b9_shared_contract(
    *,
    method: str,
    task: str,
    train_seed: int,
    fixed_horizon: Mapping[str, Any],
    generation: Mapping[str, Any],
    training_config: Mapping[str, Any],
    tracked_environment: Mapping[str, str],
) -> None:
    """Validate the fields that must be identical in both comparison lanes."""

    if method not in FOARL_MATCHED_B9_CONFIRMATION_PAIR:
        raise ValueError(f"unsupported matched-B9 comparison method {method!r}")
    if str(task).strip().lower() not in {"cvrp", "jssp", "mis", "op", "tsp"}:
        raise ValueError(f"unsupported matched-B9 task {task!r}")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int) or train_seed < 0:
        raise ValueError("matched-B9 train seed must be a nonnegative integer")
    _require_fields(
        fixed_horizon,
        {"max_steps": 3_200, "world_size": 4},
        label="fixed horizon",
    )
    _require_fields(generation, SHARED_GENERATION_CONTRACT, label="generation")
    _require_fields(
        training_config,
        {**SHARED_TRAINING_CONTRACT, "data_seed": train_seed},
        label="training configuration",
    )
    _require_fields(
        tracked_environment,
        {
            **SHARED_ENVIRONMENT_CONTRACT,
            "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1" if task == "mis" else "0",
            "GRPO_DATA_SEED": str(train_seed),
        },
        label="tracked environment",
    )


def validate_clasp_foarl_matched_b9_method_contract(
    *,
    task: str,
    train_seed: int,
    fixed_horizon: Mapping[str, Any],
    generation: Mapping[str, Any],
    training_config: Mapping[str, Any],
    tracked_environment: Mapping[str, str],
) -> None:
    """Validate both the shared projection and CLASP-specific identity."""

    validate_foarl_matched_b9_shared_contract(
        method=CLASP_FOARL_MATCHED_B9,
        task=task,
        train_seed=train_seed,
        fixed_horizon=fixed_horizon,
        generation=generation,
        training_config=training_config,
        tracked_environment=tracked_environment,
    )
    _require_fields(
        training_config,
        {
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
            "comparison_protocol": FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
            "comparison_baseline": FOARL_F638_MATCHED_B9,
            "attribution": "common_compute_data_token_schedule_only",
            "reproduction_claim": "not_an_official_jiang_reproduction",
            "prompt_style": "alpaca",
            "no_reason": True,
            "loss_type": "grpo",
            "beta": 0.05,
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "clasp_fixed_null_set": False,
        },
        label="CLASP matched-B9 method identity",
    )


def foarl_matched_b9_shared_design_material(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the path-free projection hashed by confirmation statistics."""

    method = str(contract.get("method", ""))
    task = str(contract.get("task", ""))
    seeds = contract.get("seeds")
    fixed_horizon = contract.get("fixed_horizon")
    generation = contract.get("generation")
    training_config = contract.get("training_config")
    tracked_environment = contract.get("tracked_environment")
    if not all(
        isinstance(value, Mapping)
        for value in (
            seeds,
            fixed_horizon,
            generation,
            training_config,
            tracked_environment,
        )
    ):
        raise ValueError("matched-B9 run contract lacks a shared-design object")
    train_seed = seeds.get("train")  # type: ignore[union-attr]
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise ValueError("matched-B9 run contract has no integer train seed")
    validate_foarl_matched_b9_shared_contract(
        method=method,
        task=task,
        train_seed=train_seed,
        fixed_horizon=fixed_horizon,  # type: ignore[arg-type]
        generation=generation,  # type: ignore[arg-type]
        training_config=training_config,  # type: ignore[arg-type]
        tracked_environment=tracked_environment,  # type: ignore[arg-type]
    )
    if method == CLASP_FOARL_MATCHED_B9:
        validate_clasp_foarl_matched_b9_method_contract(
            task=task,
            train_seed=train_seed,
            fixed_horizon=fixed_horizon,  # type: ignore[arg-type]
            generation=generation,  # type: ignore[arg-type]
            training_config=training_config,  # type: ignore[arg-type]
            tracked_environment=tracked_environment,  # type: ignore[arg-type]
        )

    fingerprint_source = contract.get("fingerprints")
    data_lineage_source = contract.get("data_lineage")
    if not isinstance(fingerprint_source, Mapping) or not isinstance(
        data_lineage_source, Mapping
    ):
        raise ValueError("matched-B9 run contract lacks fingerprints/data lineage")
    environment_fields = {
        **SHARED_ENVIRONMENT_CONTRACT,
        "CLASP_MIS_JIANG_RESPONSE_OUTPUT": (
            "1" if task == "mis" else "0"
        ),
        "GRPO_DATA_SEED": str(train_seed),
    }
    material = {
        "schema_version": 1,
        "protocol": FOARL_MATCHED_B9_COMPARISON_PROTOCOL,
        "sft_seed": seeds.get("sft"),  # type: ignore[union-attr]
        "fixed_horizon": dict(fixed_horizon),  # type: ignore[arg-type]
        "data_lineage": data_lineage_source,
        "fingerprints": fingerprint_source,
        "generation": {
            key: generation[key]  # type: ignore[index]
            for key in SHARED_GENERATION_CONTRACT
        },
        "training_config": {
            **{
                key: training_config[key]  # type: ignore[index]
                for key in SHARED_TRAINING_CONTRACT
            },
            "data_seed": train_seed,
        },
        "tracked_environment": environment_fields,
    }
    # Round-trip strips mapping proxies/custom mapping subclasses and prevents a
    # caller from mutating the source contract through the returned material.
    return json.loads(json.dumps(material, sort_keys=True, ensure_ascii=True))


def foarl_matched_b9_shared_design_sha256(contract: Mapping[str, Any]) -> str:
    material = foarl_matched_b9_shared_design_material(contract)
    payload = json.dumps(
        material,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CLASP_FOARL_MATCHED_B9",
    "CLASP_FOARL_MATCHED_B9_ARM_SCHEMA",
    "FOARL_MATCHED_B9_COMPARISON_PROTOCOL",
    "FOARL_MATCHED_B9_CONFIRMATION_PAIR",
    "FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST",
    "SHARED_ENVIRONMENT_CONTRACT",
    "SHARED_GENERATION_CONTRACT",
    "SHARED_TRAINING_CONTRACT",
    "foarl_matched_b9_shared_design_material",
    "foarl_matched_b9_shared_design_sha256",
    "is_foarl_matched_b9_confirmation_pair",
    "validate_clasp_foarl_matched_b9_method_contract",
    "validate_foarl_matched_b9_shared_contract",
]
