#!/usr/bin/env python3
"""Freeze the development-only TSP Frontier-CDF x KL-beta pilot.

This module is intentionally independent of the training entry point.  It
validates immutable inputs, proves that the development slice is disjoint from
RL training data, binds the relevant source bytes, and creates one read-only
contract.  It never starts training or evaluation.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from clasp.models.exact_peft_kl_post_step import (
    DEVELOPMENT_PROTOCOL as EXACT_PEFT_POST_DEVELOPMENT_PROTOCOL,
)
from clasp.models.exact_peft_kl_post_step import (
    DEVELOPMENT_STATUS as EXACT_PEFT_POST_DEVELOPMENT_STATUS,
)
from clasp.models.exact_peft_kl_post_step import (
    POST_STEP_RECEIPT_FILENAME as EXACT_PEFT_POST_RECEIPT,
)
from clasp.models.exact_peft_kl_post_step import (
    SCHEMA_VERSION as EXACT_PEFT_POST_SCHEMA_VERSION,
)
from clasp.models.exact_peft_kl_post_step import (
    STEP1_RECEIPT_FILENAME as EXACT_PEFT_STEP1_RECEIPT,
)
from clasp.models.exact_peft_kl_post_step import (
    VERIFIER_PROTOCOL as EXACT_PEFT_POST_VERIFIER_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    DEVELOPMENT_PROTOCOL as EXACT_PEFT_DEVELOPMENT_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    DEVELOPMENT_STATUS as EXACT_PEFT_DEVELOPMENT_STATUS,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_RUNTIME_RECEIPT,
)
from clasp.models.exact_peft_kl_reference import (
    SCHEMA_VERSION as EXACT_PEFT_RUNTIME_SCHEMA_VERSION,
)
from clasp.models.exact_peft_kl_reference import (
    VERIFIER_PROTOCOL as EXACT_PEFT_VERIFIER_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    ExactPeftKLReferenceError,
    build_source_binding,
    verify_development_relocation_authority,
)
from clasp.training_source_identity import (
    FOARL_F638_CORRECTED_TRAINER_SOURCE,
    TSP_FCDF_BETA_PILOT_TRAINER_SHA256,
    identify_train_grpo_source,
)

SCHEMA_VERSION = 4
SPEC_PROTOCOL = "clasp-tsp-fcdf-beta-development-pilot-spec-v4"
CONTRACT_PROTOCOL = "clasp-tsp-fcdf-beta-development-pilot-v4"
CONVERSION_PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
CONVERSION_RECEIPT = "legacy_saved_policy_conversion_receipt.json"
RELOCATION_SCHEMA_VERSION = 1
RELOCATION_PROTOCOL = "clasp-development-adapter-base-relocation-v1"
RELOCATION_STATUS = "development_saved_policy_base_relocation_verified"
RELOCATION_VERIFIER_PROTOCOL = "clasp-development-adapter-base-relocation-verifier-v1"
RELOCATION_RECEIPT = "adapter_base_relocation_receipt.json"
FROZEN_RELOCATION_IMPLEMENTATION_SHA256 = (
    "29ab2608db680d0a2510b28e4a07d0ea2f133ed6c606c8f2579d1a6f1b7ce640"
)
FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256 = (
    "9cffcbe6abfaac35e9d4f7ac2c74ed86f91aee63258c95bb8204673744ea9c06"
)
FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256 = (
    "fedf7042386c33848cb7a734d75c409709031472396b68016615e3b68f19d9f9"
)
FROZEN_MODEL_LOADING_SHA256 = "3acdec37e991a4c1c184920193d7c4771a4a41888bf9f1dc9d1401ca8625e382"
FROZEN_TRAINING_FIDELITY_SHA256 = "936adebbdb5f25f3ae5bdb030587d87d450aefcecc5e87235c30466a2c2df5da"
FROZEN_TRAIN_GRPO_SHA256 = "a35e4a4cb0882bc221469bccb49b60a9e5717b4275169b8b5d13412975923c49"
if FROZEN_TRAIN_GRPO_SHA256 != TSP_FCDF_BETA_PILOT_TRAINER_SHA256:
    raise RuntimeError("historical TSP FCDF trainer identity registry drift")
NVIDIA_SMI_ENTRYPOINT = Path("/usr/bin/nvidia-smi")
FROZEN_NVIDIA_SMI_RESOLVED_PATH = Path("/usr/lib/nvidia/current/nvidia-smi")
FROZEN_NVIDIA_SMI_RESOLVED_SHA256 = (
    "d2ce2e23e90b1044e8cbf9e60c93baefc7abbc2faa43275528916d652e983db3"
)
FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES = 686_384
FROZEN_NVIDIA_SMI_RESOLVED_MODE = 0o755
FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS = frozenset({(0, 0), (65_534, 65_534)})
TRUSTED_SYSTEM_OWNER_IDENTITY = "trusted_system_owner_root_or_user_namespace_overflow"
FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY = {
    "entrypoint_path": str(NVIDIA_SMI_ENTRYPOINT),
    "resolved_path": str(FROZEN_NVIDIA_SMI_RESOLVED_PATH),
    "resolved_sha256": FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
    "resolved_size_bytes": FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
    "resolved_mode": FROZEN_NVIDIA_SMI_RESOLVED_MODE,
    "resolved_owner_identity": TRUSTED_SYSTEM_OWNER_IDENTITY,
}
FROZEN_RELOCATED_ADAPTER_RELATIVE_PATH = (
    "experiments/sota_20260814/local_development/tsp_sft_saved_policy_bias_all_frozen_base_v1"
)
FROZEN_RELOCATION_RECEIPT_SHA256 = (
    "2030fd314dae6857999f24de8103f2051a28a6ed908779ac2c21218ebec8d000"
)
FROZEN_RELOCATION_RECEIPT_FILE_SHA256 = (
    "3b03875fcb2729c75b147d2fa6ef66a4dc747d48e38614ca51e4a0b8ca1ab52e"
)
FROZEN_RELOCATED_OUTPUT_TREE_SHA256 = (
    "36100d4a24a754863049866f570dcf528e3bd5c64e2d936848bbcdbdd54e7f10"
)
FROZEN_CONVERTED_SOURCE_TREE_SHA256 = (
    "47e638fc2a7293fad7fc8bb4910eb2aee3859943c1d8554c9df800deaf5c985d"
)
FROZEN_CONVERTED_PAYLOAD_TREE_SHA256 = (
    "a106cbf0ede0bb8bcfb0691bb0918116d763ef92466cac625dca1f5f0cc21c84"
)
FROZEN_BASE_TREE_SHA256 = "ace0432b498998365629c8a4a71042b8f117ff5e782fb368a76fe7e39df0e5a0"
FROZEN_RELOCATION_STATE_INVENTORY_SHA256 = (
    "92d518ed996424f39b3909c4a7936a6a4d01a557bd2a31db9292f3226973879a"
)
FROZEN_RELOCATION_LOGITS_SHA256 = "adc7f9e0e5ae91328650803d2bcfd687320cdb967060329b57809fc233dd1753"
RELOCATION_PROOF_DTYPE = "float32"
TRAINING_LOAD_DTYPE = "bfloat16"
EXPECTED_RUNTIME = {
    "python": "3.11.2",
    "torch": "2.10.0+cu128",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "peft": "0.19.1",
    "datasets": "4.3.0",
    "bitsandbytes": "0.49.2",
    "cuda": "12.8",
}
EXPECTED_LOCAL_HARDWARE = {
    "physical_index": 3,
    "uuid": "GPU-025e10e6-263f-d814-6dd5-added86fc8af",
    "name": "NVIDIA RTX A6000",
    "memory_total_mib": 49_140,
    "torch_total_memory_bytes": 51_041_271_808,
    "driver_version": "535.261.03",
    "visible_device_count": 1,
    "visible_runtime_index": 0,
}
CLOSED_ENVIRONMENT_PATH = (
    "/home/lamsade/habgaryan/.local/bin:/usr/local/bin:/usr/bin:/bin:/opt/dell/srvadmin/bin"
)
MIN_WORKSPACE_FREE_BYTES = 100 * 1024**3
CHECKPOINT_FIDELITY_PROTOCOL = "clasp-checkpoint-fidelity-v1"
MEMORY_GATE_DIRNAME = "_prerequisite_beta005_memory_gate_seed42"
MEMORY_GATE_PASS_FILENAME = "tsp_fcdf_beta_memory_gate_pass.json"
MEMORY_GATE_PASS_PROTOCOL = "clasp-tsp-fcdf-beta-memory-gate-pass-v1"
MATRIX_AUTHORIZATION_FILENAME = "tsp_fcdf_beta_matrix_authorization.json"
MATRIX_AUTHORIZATION_PROTOCOL = "clasp-tsp-fcdf-beta-matrix-authorization-v1"
ERROR_PREFIX = "[TspFcdfBetaPilotContractError]"
HEX64 = re.compile(r"[0-9a-f]{64}")
CAMPAIGN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{7,95}")

EXPECTED_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
EXPECTED_INIT_COUNTS = {
    "lora_tensor_count": 392,
    "bias_tensor_count": 84,
    "trainable_tensor_count": 476,
    "trainable_parameter_count": 161_609_728,
}

RELOCATION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "development_only",
        "relocation_only",
        "recovers_historical_terminal_biases",
        "converted_saved_policy_fidelity",
        "terminal_sft_policy_fidelity",
        "publication_eligible",
        "sota_eligible",
        "publication_sota_eligible",
        "claim_boundary",
        "verifier",
        "upstream_conversion",
        "frozen_base",
        "relocated_output",
        "transformation",
        "verification",
        "software",
        "receipt_sha256",
    }
)

# These files determine data loading, policy loading, the two credit rules,
# fidelity enforcement, optimization, and terminal evaluation.  A frozen
# contract is invalid as soon as any one of these bytes changes.
SOURCE_FILES = (
    "src/clasp/config.py",
    "src/clasp/data/benchmark.py",
    "src/clasp/data/loaders.py",
    "src/clasp/data/prompts.py",
    "src/clasp/entrypoints/evaluate.py",
    "src/clasp/entrypoints/train_grpo.py",
    "src/clasp/models/checkpoint_fidelity.py",
    "src/clasp/models/exact_peft_kl_reference.py",
    "src/clasp/models/exact_peft_kl_post_step.py",
    "src/clasp/models/loading.py",
    "src/clasp/models/training_fidelity.py",
    "src/clasp/rewards/clasp_math.py",
    "src/clasp/rewards/co_tasks.py",
    "src/clasp/rewards/heads.py",
    "src/clasp/rewards/primal_dual.py",
    "src/clasp/rewards/registry.py",
    "src/clasp/runtime.py",
    "src/clasp/set_grpo/controller.py",
    "src/clasp/set_grpo/frontier_cdf.py",
    "src/clasp/trainers/primal_dual_grpo.py",
    "scripts/relocate_development_adapter_base.py",
    "scripts/dry_run_tsp_fcdf_beta_pilot.py",
    "scripts/freeze_tsp_fcdf_beta_pilot.py",
)

FROZEN_CORE_SOURCE_SHA256 = {
    "scripts/relocate_development_adapter_base.py": (FROZEN_RELOCATION_IMPLEMENTATION_SHA256),
    "src/clasp/entrypoints/train_grpo.py": FROZEN_TRAIN_GRPO_SHA256,
    "src/clasp/models/exact_peft_kl_reference.py": (FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256),
    "src/clasp/models/exact_peft_kl_post_step.py": (FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256),
    "src/clasp/models/loading.py": FROZEN_MODEL_LOADING_SHA256,
    "src/clasp/models/training_fidelity.py": FROZEN_TRAINING_FIDELITY_SHA256,
}

METHODS = (
    {
        "code": "D",
        "name": "dense_rank",
        "frontier_cdf": False,
        "relative_failed_recovery": False,
        "interpretation": "rank-only successful-set credit",
    },
    {
        "code": "F",
        "name": "frontier_cdf_relative_failed_recovery",
        "frontier_cdf": True,
        "relative_failed_recovery": True,
        "interpretation": "target-aware fixed-K set credit with relative failed recovery",
    },
)
BETAS = (0.0, 0.05)
SEEDS = (42, 43, 44)

FIXED_DESIGN: dict[str, Any] = {
    "task": "tsp",
    "factorial": "2x2",
    "methods": list(METHODS),
    "grpo_betas": list(BETAS),
    "training_seeds": list(SEEDS),
    "trajectory_count": len(METHODS) * len(BETAS) * len(SEEDS),
    "terminal_step": 400,
    "num_generations_train": 8,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "effective_train_batch_size": 32,
    "learning_rate": 1e-6,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "linear",
    "num_return_sequences_eval": 8,
    "evaluation_seed": 271828,
    "development_slice": {"start_index": 0, "count": 200},
    "frontier_gap_target": 0.0107,
    "group_failure_target": 0.005,
    "relocation_verification_dtype": RELOCATION_PROOF_DTYPE,
    "training_load_dtype": TRAINING_LOAD_DTYPE,
    "runtime_versions": dict(EXPECTED_RUNTIME),
    "local_hardware": dict(EXPECTED_LOCAL_HARDWARE),
    "closed_environment_path": CLOSED_ENVIRONMENT_PATH,
    "pytorch_cuda_alloc_conf": "expandable_segments:True",
    "workspace_bound_per_attempt_tmpdir": True,
    "minimum_workspace_free_bytes_before_every_exec": MIN_WORKSPACE_FREE_BYTES,
    "root_tmp_fallback_forbidden": True,
    "common_bfloat16_load_not_exact_fp32_source_policy_equivalence": True,
    "policy_parameterization": "continue_same_bias_all_sft_adapter",
    "fresh_rl_lora_attached": False,
    "checkpoint_rule": "terminal_checkpoint_400_only",
    "metric_early_stopping": False,
}

ANALYSIS_PLAN: dict[str, Any] = {
    "primary_endpoint": (
        "two-part terminal endpoint: raw pass-at-8 feasibility, then mean raw best-of-8 "
        "objective gap among feasible groups; infeasible groups are failures, not repaired "
        "observations"
    ),
    "estimand": (
        "seed-blocked mean paired objective-gap difference on jointly feasible instances "
        "after the feasibility gate, using identical frozen instances and evaluation seed"
    ),
    "missing_quality_rule": {
        "primary": "pairwise gap contrasts use only instances feasible in both compared arms",
        "required_reporting": "report joint-feasible denominator and each arm's failure rows",
        "sensitivity": "repeat with every all-infeasible group assigned objective_gap=1.0",
        "instability_action": "if the factor-effect direction changes, that factor cannot advance",
    },
    "planned_contrasts": {
        "frontier_at_beta_0": "F0-D0",
        "frontier_at_beta_0.05": "F5-D5",
        "anchor_at_dense": "D5-D0",
        "anchor_at_frontier": "F5-F0",
        "factorial_interaction": "(F5-F0)-(D5-D0)",
    },
    "uncertainty": (
        "report every seed contrast and a seed-blocked hierarchical bootstrap over "
        "training seeds then paired instances; three seeds imply development uncertainty only"
    ),
    "matrix_completion_rule": (
        "all 12 preregistered trajectories remain in the accounting denominator; a missing, "
        "failed, or invalid trajectory is reported at its reserved cell/seed and cannot be "
        "silently replaced, rerun, or dropped from factor advancement"
    ),
    "health_stop_only": [
        "NaN_or_infinite_loss_reward_advantage_or_KL",
        "immutable_input_or_source_hash_drift",
        "missing_or_invalid_checkpoint_fidelity_at_checkpoint_400_or_final_root",
        "global_step_not_exactly_400",
        "optimizer_or_scheduler_resume_detected",
    ],
    "advancement_gate": {
        "scope": "selects_only_a_later_preregistered_development_or_publication_pilot",
        "cell_eligibility": {
            "required_mean_gap": "<=0.0107",
            "required_pass_at_8_feasibility": ">=0.995",
        },
        "factor_decisions": {
            "frontier_cdf": {
                "estimand": "0.5*((F0-D0)+(F5-D5))",
                "required_paired_improvement": ">=0.0005 absolute gap",
                "required_seed_direction": "pooled improvement in at least 2 of 3 seeds",
            },
            "beta_0.05_anchor": {
                "estimand": "0.5*((D5-D0)+(F5-F0))",
                "required_paired_improvement": ">=0.0005 absolute gap",
                "required_seed_direction": "pooled improvement in at least 2 of 3 seeds",
            },
            "interaction": "reported_with_uncertainty_but_not_used_as_a_selection_gate",
        },
        "winner": (
            "lowest terminal mean raw gap among eligible cells whose nonbaseline factor "
            "levels each pass their factor decision"
        ),
        "tie_break": "within 0.0001 absolute gap choose lower beta, then dense_rank",
        "target_saturation_override": (
            "if the prespecified F0 or F5 cell saturation flag fires, no nonbaseline "
            "factor advances and D0 remains control"
        ),
        "if_no_nonbaseline_factor_passes": "no new mechanism advances; D0 remains control",
    },
}

SEALED_SFT_SUMMARY_SHA256 = "a065aeadbf0811462492e79233b3d11edbd745b92799236db492041213512967"
SEALED_SFT_DETAILS_SHA256 = "5d6b56e64092be597d1fe4fe389b30c471cd2a2892cc6a033fd0758e319a0046"
FRONTIER_THRESHOLD_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
SEALED_SFT_EXPECTED = {
    "example_count": 200,
    "best_of_8_feasible_count": 200,
    "best_of_8_mean_gap": 0.008826100659264306,
    "best_of_8_zero_gap_count": 35,
    "best_of_8_at_or_below_target_count": 128,
    "feasible_candidate_count": 1434,
    "feasible_candidate_mean_gap": 0.02246338141132295,
    "feasible_candidate_zero_gap_count": 106,
    "feasible_candidate_at_or_below_target_count": 455,
}

TARGET_SATURATION_PREREGISTRATION = {
    "target_gap": 0.0107,
    "target_status": "retained_without_outcome_retuning",
    "sealed_sft_limitation": {
        "mean_best_of_8_gap": 0.008826100659264306,
        "mean_to_target_ratio": 0.8248692204919913,
        "target_minus_mean": 0.0018738993407356932,
        "best_of_8_at_or_below_target": "128/200=0.64",
        "best_of_8_exact_zero": "35/200=0.175",
        "feasible_candidates_at_or_below_target": "455/1434=0.3172942817294282",
        "feasible_candidates_exact_zero": "106/1434=0.07391910739191074",
        "feasible_candidate_mean_gap": 0.02246338141132295,
    },
    "retention_rationale": (
        "0.0107 remains the pre-existing raw-gap normalization and eligibility scale; "
        "the seven fixed thresholds span 0.5x through 3x, 36% of sealed SFT best-of-8 "
        "sets and 68.2706% of feasible candidates remain above 1x, and paired factor "
        "improvement rather than target attainment controls advancement"
    ),
    "terminal_reporting": {
        "threshold_multipliers": list(FRONTIER_THRESHOLD_MULTIPLIERS),
        "required_for_every_arm": (
            "report feasible-candidate and best-of-8 empirical fractions at or below each "
            "fixed target multiplier, plus exact-zero fractions"
        ),
        "status": "descriptive_fixed_grid_sensitivity_not_a_second_training_target",
    },
    "training_signal_saturation_guard": {
        "frontier_cells_only": True,
        "window_optimizer_steps_inclusive": [301, 400],
        "required_complete_steps": 100,
        "trajectory_flag": (
            "median quality_utility_success_mean >= 0.99 and "
            "median quality_credit_group_mass_mean <= 0.0001"
        ),
        "cell_flag": "trajectory_flag in at least 2 of 3 seeds",
        "decision": (
            "if F0 or F5 has a cell flag, the pilot is target-saturated/underidentified; "
            "no nonbaseline factor advances and the result cannot reject magnitude-aware credit"
        ),
        "not_a_health_stop": True,
    },
    "forbidden": [
        "retune_0.0107_after_observing_any_pilot_trajectory",
        "substitute_a_new_training_target_inside_this_2x2",
        "treat_baseline_target_attainment_as_factor_improvement",
    ],
}

REQUIRED_TELEMETRY = {
    "cadence": "every_optimizer_step",
    "common": [
        "loss",
        "learning_rate",
        "grad_norm",
        "constrained_grpo/candidate_feasibility_rate",
        "constrained_grpo/clasp/advantage_l2",
        "constrained_grpo/clasp/advantage_abs_mean",
        "constrained_grpo/clasp/bon_group_failure_rate",
    ],
    "frontier_cells": [
        "constrained_grpo/frontier_cdf/group_failure_rate",
        "constrained_grpo/frontier_cdf/quality_temperature",
        "constrained_grpo/frontier_cdf/mean_target_gap",
        "constrained_grpo/frontier_cdf/feasible_normalized_gap_p50",
        "constrained_grpo/frontier_cdf/feasible_normalized_gap_p90",
        "constrained_grpo/frontier_cdf/best_feasible_normalized_gap_p50",
        "constrained_grpo/frontier_cdf/best_feasible_normalized_gap_p90",
        "constrained_grpo/frontier_cdf/quality_utility_success_mean",
        "constrained_grpo/frontier_cdf/quality_credit_group_mass_mean",
        "constrained_grpo/frontier_cdf/quality_credit_group_nonzero_rate",
        "constrained_grpo/frontier_cdf/zero_advantage_rate",
        "constrained_grpo/frontier_cdf/null_decomposition_error_max",
        "constrained_grpo/frontier_cdf/relative_recovery_abs_mean_failed",
        "constrained_grpo/frontier_cdf/relative_recovery_signed_group_rate_failed",
        "constrained_grpo/frontier_cdf/dual_lambda_mean",
        "constrained_grpo/frontier_cdf/dual_at_zero_rate",
    ],
    "beta_positive_cells": ["kl"],
    "missing_required_metric_action": "invalidate_trajectory_before_arm_selection",
}

EVIDENCE_BOUNDARY = {
    "development_only": True,
    "publication_eligible": False,
    "sota_eligible": False,
    "confirmation_eligible": False,
    "initialization_claim": "common_relocated_bias_all_adapter_loaded_bfloat16",
    "relocation_verification_dtype": RELOCATION_PROOF_DTYPE,
    "training_load_dtype": TRAINING_LOAD_DTYPE,
    "common_bfloat16_load_not_exact_fp32_source_policy_equivalence": True,
    "continued_lora_optimization_geometry": True,
    "fresh_rl_lora_attached": False,
    "merged_reference_or_policy_used": False,
    "exact_source_policy_equivalence_claim_allowed": False,
    "historical_terminal_sft_policy_claim_allowed": False,
    "publication_claim_allowed": False,
    "sota_claim_allowed": False,
    "terminal_sft_policy_fidelity": False,
    "prohibited_claim": "publication_or_state_of_the_art_evidence",
}

EXECUTION_CONTRACT = {
    "generator": "scripts/dry_run_tsp_fcdf_beta_pilot.py",
    "mode": "audited_lifecycle_with_memory_gate_supervisor_only",
    "executes_processes": True,
    "general_execute_flag": False,
    "process_execution_scope": "one_create_once_memory_gate_child_only",
    "memory_gate_supervision": {
        "claim_bound_precreated_transcript_inode": True,
        "single_claim_bound_precreated_outcome_inode": True,
        "exact_two_link_generation_guard_for_every_execution_anchor": True,
        "extra_or_missing_anchor_hardlinks_rejected": True,
        "sealed_anchor_directory_full_stat_identity_bound": True,
        "post_execution_containing_gate_full_stat_identity_bound": True,
        "combined_stdout_stderr_transcript": True,
        "exact_payload_environment_and_cwd_rechecked_before_spawn": True,
        "isolated_process_group_and_session": True,
        "linux_parent_death_wrapper": True,
        "parent_death_group_termination": True,
        "pid_start_time_pgid_bound_immediately_after_spawn": True,
        "normal_zero_nonzero_signal_and_spawn_error_captured": True,
        "post_execution_process_group_payload_and_gpu_absence_required": True,
        "zero_exit_receipt_required_before_pass": True,
        "nonzero_signal_or_unknown_recursively_seals_failure_tree": True,
        "operator_unknown_only_after_released_lock_and_process_gpu_absence": True,
        "retry_after_any_claim": False,
    },
    "relocation_live_authorization": (
        "one successful strict relocation verify-only rehash is a prerequisite for the "
        "one-step beta-positive memory gate; every beta-positive entrypoint verifies the "
        "same source again"
    ),
    "one_step_beta_memory_gate_required_before_any_400_step_cell": True,
    "resume": (
        "first start is create-once; a repeated training invocation is allowed only from "
        "the same surviving newest nonterminal checkpoint with exact fidelity/post-step "
        "evidence and one create-once checkpoint-specific resume claim"
    ),
    "overwrite": False,
    "adopt_existing_outputs": False,
    "development_fidelity_env": "CLASP_DEVELOPMENT_FIDELITY=1",
    "publication_fidelity_env": "CLASP_PUBLICATION_FIDELITY=0",
    "kl_reference_env": [
        "CLASP_KL_REFERENCE_MODEL_PATH",
        "CLASP_KL_REFERENCE_RECEIPT",
    ],
    "training_run_contract_env": "CLASP_TRAINING_RUN_CONTRACT_PATH",
    "training_run_contract_chronology": {
        "memory_gate": "frozen pilot contract bytes",
        "scientific_cells": "create-once pre-outcome full-matrix authorization ledger",
    },
    "cell_commands_before_gate_pass": "forbidden",
    "matrix_authorization_regeneration_after_any_cell_or_eval_attempt": "forbidden",
    "matrix_reservation_closure": (
        "one create-once pre-outcome closure binds the inode and receipt identity of all "
        "12 training plus 12 evaluation reservations and every precreated single-use "
        "attempt-claim anchor; missing/deleted slots or anchors are never recreated"
    ),
    "training_attempt_claim": (
        "consume_one_precreated_closure_bound_inode_before_exec; exact live env/argv must "
        "match the ledger before claim and again immediately before exec; retry sets only "
        "the exact newest surviving claimed RESUME_FROM_CHECKPOINT"
    ),
    "evaluation_attempt_claim": (
        "consume_one_precreated_closure_bound_inode_before_exec_no_retry_or_resume; exact "
        "live env/argv must match the ledger before claim and immediately before exec"
    ),
    "terminal_audit": (
        "create_once before evaluation; exact-crossbind root/checkpoint-400 trainable closure, "
        "hyperparameters, training run-contract ledger, telemetry, and beta-specific KL evidence"
    ),
    "closed_execution_environment": {
        "env_i": True,
        "cuda_visible_devices": "3",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "workspace_bound_tmpdir_and_caches": True,
        "minimum_workspace_free_bytes": MIN_WORKSPACE_FREE_BYTES,
        "tempfile_gettempdir_must_equal_bound_tmpdir": True,
        "home_and_all_named_caches_must_be_descendants_of_bound_tmpdir": True,
        "root_tmp_fallback": False,
        "live_runtime_hardware_contract_and_source_rehash_before_every_exec": True,
        "exact_live_environment_and_payload_argv_rechecked_before_claim_and_exec": True,
        "nvidia_smi_entrypoint": str(NVIDIA_SMI_ENTRYPOINT),
        "nvidia_smi_resolved_path": str(FROZEN_NVIDIA_SMI_RESOLVED_PATH),
        "nvidia_smi_resolved_sha256": FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
        "nvidia_smi_alternatives_chain_and_resolved_inode_rehashed": True,
    },
    "all_twelve_trajectories_required_in_denominator": True,
    "beta_zero_forbids_kl_reference_env": True,
}

REQUIRED_OUTPUT_EVIDENCE = {
    "every_trajectory": [
        "checkpoints/checkpoint-400/checkpoint_fidelity.json",
        "checkpoint_fidelity.json",
    ],
    "receipt_protocol": CHECKPOINT_FIDELITY_PROTOCOL,
    "receipt_status": "exact_trainable_policy_persisted",
    "publication_invariant": None,
    "exact_qwen_structure_preflight": True,
    "beta_positive_trajectory": [
        EXACT_PEFT_RUNTIME_RECEIPT,
        EXACT_PEFT_STEP1_RECEIPT,
        EXACT_PEFT_POST_RECEIPT,
        f"checkpoints/checkpoint-400/{EXACT_PEFT_RUNTIME_RECEIPT}",
        f"checkpoints/checkpoint-400/{EXACT_PEFT_STEP1_RECEIPT}",
        f"checkpoints/checkpoint-400/{EXACT_PEFT_POST_RECEIPT}",
    ],
    "beta_zero_trajectory_forbids_exact_peft_runtime_receipt": True,
    "beta_zero_trajectory_forbids_all_pre_step_step1_and_post_step_sidecars": True,
    "exact_peft_runtime": {
        "schema_version": EXACT_PEFT_RUNTIME_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_DEVELOPMENT_PROTOCOL,
        "status": EXACT_PEFT_DEVELOPMENT_STATUS,
        "verifier_protocol": EXACT_PEFT_VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256,
        "parameterization": "continue_same_bias_all_sft_adapter",
        "sync_ref_model": False,
        "use_vllm": False,
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
        "reference_trainable_tensor_count": 0,
        "independent_storage": True,
        "step_zero_state_and_three_probe_outputs_bitwise_equal": True,
    },
    "exact_peft_post_step": {
        "schema_version": EXACT_PEFT_POST_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_POST_DEVELOPMENT_PROTOCOL,
        "status": EXACT_PEFT_POST_DEVELOPMENT_STATUS,
        "verifier_protocol": EXACT_PEFT_POST_VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": (FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256),
        "step1_filename": EXACT_PEFT_STEP1_RECEIPT,
        "checkpoint_or_final_filename": EXACT_PEFT_POST_RECEIPT,
        "reference_trainable_tensor_count": 0,
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
        "live_reference_bitwise_equal_to_pre_step": True,
        "policy_reference_storage_independent": True,
        "optimizer_excludes_reference_storage": True,
        "root_and_checkpoint_400_run_contract": "immutable_pre_outcome_matrix_ledger",
    },
    "missing_receipt_action": "invalidate_trajectory_and_stop_before_evaluation",
}

MEMORY_GATE = {
    "required": True,
    "phase": "one_step_beta_0.05_memory_and_exactness_gate",
    "method": "dense_rank",
    "grpo_beta": 0.05,
    "seed": 42,
    "scheduler_horizon_max_steps": 400,
    "hard_stop_global_step": 1,
    "warmup_ratio": 0.0,
    "pilot_warmup_ratio": 0.03,
    "nonzero_lr_engineering_override": True,
    "never_resume_or_extend_gate_artifact": True,
    "num_generations": 8,
    "training_load_dtype": TRAINING_LOAD_DTYPE,
    "parameterization": "continue_same_bias_all_sft_adapter",
    "policy_adapter_autocast_dtype": True,
    "reference_adapter_autocast_dtype": True,
    "not_an_analysis_cell": True,
    "output_reuse_for_400_step_cells": False,
    "required_before_any_400_step_cell": True,
    "required_artifacts": [
        (
            "_prerequisite_beta005_memory_gate_seed42/_claim_anchors/"
            "memory_gate_attempt_started.json"
        ),
        (
            "_prerequisite_beta005_memory_gate_seed42/_claim_anchors/"
            "memory_gate_execution.transcript"
        ),
        (
            "_prerequisite_beta005_memory_gate_seed42/_claim_anchors/"
            "memory_gate_execution_outcome.json"
        ),
        "_prerequisite_beta005_memory_gate_seed42/checkpoints/checkpoint-1/checkpoint_fidelity.json",
        f"_prerequisite_beta005_memory_gate_seed42/{EXACT_PEFT_RUNTIME_RECEIPT}",
        f"_prerequisite_beta005_memory_gate_seed42/{EXACT_PEFT_STEP1_RECEIPT}",
        (
            "_prerequisite_beta005_memory_gate_seed42/checkpoints/checkpoint-1/"
            f"{EXACT_PEFT_RUNTIME_RECEIPT}"
        ),
        (
            "_prerequisite_beta005_memory_gate_seed42/checkpoints/checkpoint-1/"
            f"{EXACT_PEFT_STEP1_RECEIPT}"
        ),
        (
            "_prerequisite_beta005_memory_gate_seed42/checkpoints/checkpoint-1/"
            f"{EXACT_PEFT_POST_RECEIPT}"
        ),
    ],
    "success_evidence": [
        "create_once_gate_attempt_receipt_binds_exact_live_environment_and_payload_argv",
        "claim_bound_combined_stream_transcript_and_exact_zero_exit_receipt",
        "isolated_child_process_group_and_parent_death_wrapper_identity",
        "post_child_process_group_payload_and_target_gpu_process_absence",
        "actual_optimizer_global_step_exactly_1",
        "checkpoint_1_checkpoint_fidelity_valid",
        "root_and_checkpoint_1_pre_step_runtime_receipts_valid_and_identical",
        "checkpoint_1_post_step_reference_receipt_valid",
        "at_least_one_policy_trainable_tensor_digest_changed_from_initial_adapter",
        "reference_trainable_tensor_count_zero",
        "live_reference_tensor_inventory_bitwise_equal_to_pre_step",
        "reference_object_identity_storage_disjoint_and_optimizer_excluded",
        "reference_source_binding_unchanged_and_runtime_receipts_copied_without_drift",
        "no_cuda_oom_or_nan_inf",
        "per_rank_peak_allocated_and_reserved_cuda_bytes_recorded",
    ],
    "claim_limit": (
        "engineering feasibility only; the frozen post-step receipt independently proves "
        "live reference non-mutation, but the gate is not a scientific trajectory and can "
        "never enter development selection, publication, benchmark, or SOTA evidence"
    ),
    "acceptance_lifecycle": {
        "gate_subtree": MEMORY_GATE_DIRNAME,
        "pass_receipt": MEMORY_GATE_PASS_FILENAME,
        "pass_protocol": MEMORY_GATE_PASS_PROTOCOL,
        "matrix_authorization": MATRIX_AUTHORIZATION_FILENAME,
        "matrix_protocol": MATRIX_AUTHORIZATION_PROTOCOL,
        "matrix_reservation_closure": ("tsp_fcdf_beta_matrix_reservation_closure.json"),
        "gate_tree_sealed_read_only_and_fully_rehashed": True,
        "exact_nonterminal_gate_root_allowlist_and_attempt_receipt_reopened": True,
        "cell_commands_emitted_only_after_pass": True,
        "one_pre_outcome_ledger_reserves_all_training_and_evaluation_attempts": True,
        "ledger_regeneration_after_any_outcome": False,
        "training_and_evaluation_attempts_are_create_once": True,
        "precreated_claim_anchor_inodes_bound_before_any_matrix_outcome": True,
        "terminal_audit_required_before_evaluation": True,
    },
    "current_readiness": "CORE_FROZEN_REQUIRES_UNRUN_ONE_UPDATE_GATE",
    "nonterminal_root_finalization_forbidden": True,
    "root_final_model_checkpoint_fidelity_and_post_step_receipt_expected": False,
}


class ContractError(ValueError):
    """An immutable preregistration condition was not met."""


def fail(message: str) -> NoReturn:
    raise ContractError(f"{ERROR_PREFIX} {message}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def clone_json(value: object) -> Any:
    """Detach contract values from module-level preregistration constants."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"expected a regular, non-symlink file: {path}")
    try:
        before = path.lstat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in identity):
        fail(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular, non-symlink JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value


def load_json_lines(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular, non-symlink JSONL file: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    fail(f"{label} contains a blank line at {line_number}")
                value = json.loads(line, object_pairs_hook=_strict_json_object)
                if not isinstance(value, dict):
                    fail(f"{label} line {line_number} is not a JSON object")
                records.append(value)
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} JSONL from {path}: {exc}")
    if not records:
        fail(f"{label} contains no records: {path}")
    return records


def canonical_path(raw: object, *, label: str, kind: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        fail(f"{label} must be a non-empty absolute path")
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        fail(f"{label} must be absolute: {supplied}")
    if supplied.is_symlink():
        fail(f"{label} must not be a symlink: {supplied}")
    try:
        path = supplied.resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve {label} {supplied}: {exc}")
    if kind == "file" and not path.is_file():
        fail(f"{label} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        fail(f"{label} is not a directory: {path}")
    return path


def executable_path(raw: object, *, label: str) -> Path:
    """Keep a virtualenv entry-point path while validating its resolved target."""

    if not isinstance(raw, str) or not raw.strip():
        fail(f"{label} must be a non-empty absolute path")
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        fail(f"{label} must be absolute: {supplied}")
    supplied = supplied.absolute()
    if not supplied.is_file():
        fail(f"{label} must be an absolute executable file: {supplied}")
    try:
        supplied.resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve {label} {supplied}: {exc}")
    if not os.access(supplied, os.X_OK):
        fail(f"{label} is not executable: {supplied}")
    return supplied


def probe_runtime_identity(python_executable: Path) -> dict[str, Any]:
    code = (
        "import json,platform,torch,transformers,trl,peft,datasets,bitsandbytes;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'transformers':transformers.__version__,'trl':trl.__version__,"
        "'peft':peft.__version__,'datasets':datasets.__version__,"
        "'bitsandbytes':bitsandbytes.__version__,'cuda':torch.version.cuda or 'none'},"
        "sort_keys=True))"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="clasp_runtime_probe_") as temporary:
            environment = {
                "HOME": "/home/lamsade/habgaryan",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "LOGNAME": "habgaryan",
                "PATH": CLOSED_ENVIRONMENT_PATH,
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": temporary,
                "TORCHINDUCTOR_CACHE_DIR": f"{temporary}/torchinductor",
                "USER": "habgaryan",
                "XDG_CACHE_HOME": f"{temporary}/xdg",
            }
            completed = subprocess.run(
                [str(python_executable), "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"cannot execute frozen runtime probe: {exc}")
    if completed.returncode != 0:
        fail(f"frozen runtime probe failed: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        fail(f"frozen runtime probe returned invalid JSON: {exc}")
    if observed != EXPECTED_RUNTIME:
        fail(f"runtime version drift: expected={EXPECTED_RUNTIME}, observed={observed}")
    resolved = python_executable.resolve(strict=True)
    return {
        "versions": dict(observed),
        "python_executable": str(python_executable),
        "python_resolved_executable": str(resolved),
        "python_resolved_executable_sha256": sha256_file(resolved),
        "probe_method": "closed_environment_imported_package_versions_v1",
    }


def _executable_symlink_chain(entrypoint: Path, *, label: str) -> list[dict[str, Any]]:
    """Inspect an alternatives chain without silently accepting retargeting."""

    if not entrypoint.is_absolute():
        fail(f"{label} entrypoint must be absolute: {entrypoint}")
    current = Path(os.path.abspath(entrypoint))
    visited: set[str] = set()
    records: list[dict[str, Any]] = []
    for _ in range(32):
        key = str(current)
        if key in visited:
            fail(f"{label} symlink cycle detected at {current}")
        visited.add(key)
        try:
            observed = current.lstat()
        except OSError as exc:
            fail(f"{label} symlink chain is broken at {current}: {exc}")
        common = {
            "path": str(current),
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
            "uid": observed.st_uid,
            "gid": observed.st_gid,
            "size_bytes": observed.st_size,
            "mtime_ns": observed.st_mtime_ns,
            "ctime_ns": observed.st_ctime_ns,
        }
        if stat.S_ISLNK(observed.st_mode):
            try:
                target = os.readlink(current)
            except OSError as exc:
                fail(f"cannot read {label} symlink {current}: {exc}")
            records.append({**common, "kind": "symlink", "link_target": target})
            target_path = Path(target)
            unresolved = target_path if target_path.is_absolute() else current.parent / target_path
            current = Path(os.path.abspath(unresolved))
            continue
        if not stat.S_ISREG(observed.st_mode):
            fail(f"{label} resolved target is not a regular file: {current}")
        records.append({**common, "kind": "regular_file", "link_target": None})
        return records
    fail(f"{label} symlink chain exceeds 32 entries")


def normalize_executable_owner(
    uid: int,
    gid: int,
    *,
    accepted_owner_ids: frozenset[tuple[int, int]] | None,
    label: str,
) -> str:
    owner = (uid, gid)
    if accepted_owner_ids is None:
        return f"uid:{uid}:gid:{gid}"
    if owner not in accepted_owner_ids:
        fail(f"{label} alternatives chain has untrusted owner mapping {owner}")
    return TRUSTED_SYSTEM_OWNER_IDENTITY


def resolve_executable_identity(
    entrypoint: Path,
    *,
    label: str,
    expected_resolved_path: Path | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    expected_mode: int | None = None,
    accepted_owner_ids: frozenset[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Bind a symlink entrypoint to one stable, non-user-writable executable."""

    before = _executable_symlink_chain(entrypoint, label=label)
    resolved = Path(before[-1]["path"])
    try:
        canonical = entrypoint.resolve(strict=True)
    except OSError as exc:
        fail(f"cannot resolve {label} entrypoint {entrypoint}: {exc}")
    if resolved != canonical or resolved.is_symlink() or not resolved.is_file():
        fail(f"{label} resolved identity is inconsistent: {entrypoint} -> {resolved}")
    if expected_resolved_path is not None and resolved != expected_resolved_path:
        fail(f"{label} resolved path drift: expected={expected_resolved_path}, observed={resolved}")
    observed = resolved.lstat()
    mode = stat.S_IMODE(observed.st_mode)
    if not os.access(resolved, os.X_OK):
        fail(f"{label} resolved target is not executable: {resolved}")
    if os.access(resolved, os.W_OK):
        fail(f"{label} resolved target is writable by the current user: {resolved}")
    observed_owners = {(record["uid"], record["gid"]) for record in before}
    if accepted_owner_ids is not None:
        for uid, gid in observed_owners:
            normalize_executable_owner(
                uid,
                gid,
                accepted_owner_ids=accepted_owner_ids,
                label=label,
            )
    if expected_size_bytes is not None and observed.st_size != expected_size_bytes:
        fail(
            f"{label} resolved size drift: expected={expected_size_bytes}, "
            f"observed={observed.st_size}"
        )
    if expected_mode is not None and mode != expected_mode:
        fail(f"{label} resolved mode drift: expected={oct(expected_mode)}, observed={oct(mode)}")
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        fail(
            f"{label} resolved executable hash drift: expected={expected_sha256}, observed={digest}"
        )
    after = _executable_symlink_chain(entrypoint, label=label)
    if after != before:
        fail(f"{label} entrypoint or resolved target changed during inspection")
    normalized_chain = []
    for record in before:
        normalized = dict(record)
        owner = (normalized.pop("uid"), normalized.pop("gid"))
        normalized["owner_identity"] = normalize_executable_owner(
            owner[0],
            owner[1],
            accepted_owner_ids=accepted_owner_ids,
            label=label,
        )
        normalized_chain.append(normalized)
    return {
        "entrypoint_path": str(entrypoint),
        "entrypoint_is_symlink": before[0]["kind"] == "symlink",
        "symlink_chain": normalized_chain,
        "resolved_path": str(resolved),
        "resolved_sha256": digest,
        "resolved_size_bytes": observed.st_size,
        "resolved_mode": mode,
        "resolved_owner_identity": normalize_executable_owner(
            observed.st_uid,
            observed.st_gid,
            accepted_owner_ids=accepted_owner_ids,
            label=label,
        ),
        "accepted_owner_ids": (
            [list(value) for value in sorted(accepted_owner_ids)]
            if accepted_owner_ids is not None
            else None
        ),
        "resolved_executable": True,
        "resolved_writable_by_current_user": False,
    }


def normalized_executable_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    fields = tuple(FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY)
    return {field: identity.get(field) for field in fields}


def probe_local_hardware(python_executable: Path) -> dict[str, Any]:
    nvidia_smi = resolve_executable_identity(
        NVIDIA_SMI_ENTRYPOINT,
        label="required hardware probe",
        expected_resolved_path=FROZEN_NVIDIA_SMI_RESOLVED_PATH,
        expected_sha256=FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
        expected_size_bytes=FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
        expected_mode=FROZEN_NVIDIA_SMI_RESOLVED_MODE,
        accepted_owner_ids=FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
    )
    if normalized_executable_identity(nvidia_smi) != FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY:
        fail("required hardware probe normalized executable identity drifted")
    command = [
        str(nvidia_smi["resolved_path"]),
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
        "-i",
        str(EXPECTED_LOCAL_HARDWARE["physical_index"]),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": CLOSED_ENVIRONMENT_PATH, "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"cannot execute local hardware probe: {exc}")
    if completed.returncode != 0:
        fail(f"local hardware probe failed: {completed.stderr.strip()}")
    rows = [value.strip() for value in completed.stdout.splitlines() if value.strip()]
    if len(rows) != 1:
        fail(f"local hardware probe returned {len(rows)} rows")
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) != 5:
        fail(f"local hardware probe returned malformed row: {rows[0]!r}")
    torch_code = (
        "import json,torch;"
        "p=torch.cuda.get_device_properties(0);"
        "print(json.dumps({'visible_device_count':torch.cuda.device_count(),"
        "'visible_runtime_index':0,'visible_name':p.name,"
        "'torch_total_memory_bytes':p.total_memory,'cuda':torch.version.cuda},sort_keys=True))"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="clasp_hardware_probe_") as temporary:
            torch_environment = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(EXPECTED_LOCAL_HARDWARE["physical_index"]),
                "HOME": "/home/lamsade/habgaryan",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "LOGNAME": "habgaryan",
                "PATH": CLOSED_ENVIRONMENT_PATH,
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": temporary,
                "TORCHINDUCTOR_CACHE_DIR": f"{temporary}/torchinductor",
                "USER": "habgaryan",
                "XDG_CACHE_HOME": f"{temporary}/xdg",
            }
            torch_completed = subprocess.run(
                [str(python_executable), "-c", torch_code],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                env=torch_environment,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"cannot execute torch-visible hardware probe: {exc}")
    if torch_completed.returncode != 0:
        fail(f"torch-visible hardware probe failed: {torch_completed.stderr.strip()}")
    try:
        torch_observed = json.loads(torch_completed.stdout.strip())
        observed = {
            "physical_index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": int(fields[3]),
            "torch_total_memory_bytes": int(torch_observed["torch_total_memory_bytes"]),
            "driver_version": fields[4],
            "visible_device_count": int(torch_observed["visible_device_count"]),
            "visible_runtime_index": int(torch_observed["visible_runtime_index"]),
        }
    except ValueError as exc:
        fail(f"local hardware probe returned non-integer index/memory: {exc}")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"torch-visible hardware probe returned invalid JSON: {exc}")
    if (
        torch_observed.get("visible_name") != EXPECTED_LOCAL_HARDWARE["name"]
        or torch_observed.get("cuda") != EXPECTED_RUNTIME["cuda"]
    ):
        fail(f"torch-visible CUDA identity drift: {torch_observed}")
    if observed != EXPECTED_LOCAL_HARDWARE:
        fail(f"local hardware drift: expected={EXPECTED_LOCAL_HARDWARE}, observed={observed}")
    final_nvidia_smi = resolve_executable_identity(
        NVIDIA_SMI_ENTRYPOINT,
        label="required hardware probe",
        expected_resolved_path=FROZEN_NVIDIA_SMI_RESOLVED_PATH,
        expected_sha256=FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
        expected_size_bytes=FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
        expected_mode=FROZEN_NVIDIA_SMI_RESOLVED_MODE,
        accepted_owner_ids=FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
    )
    if final_nvidia_smi != nvidia_smi:
        fail("required hardware-probe alternatives chain drifted during GPU inspection")
    return {
        **observed,
        "probe_executable": nvidia_smi,
        "query": command[1:],
    }


def canonical_output_path(
    raw: object,
    *,
    require_absent: bool,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        fail("output_root must be a non-empty absolute path")
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        fail(f"output_root must be absolute: {supplied}")
    if supplied.exists() or supplied.is_symlink():
        if require_absent:
            fail(f"output_root must not exist: {supplied}")
        if supplied.is_symlink() or not supplied.is_dir():
            fail(f"existing output_root is not a regular directory: {supplied}")
        return supplied.resolve(strict=True)
    parent = canonical_path(
        str(supplied.parent),
        label="output_root parent",
        kind="directory",
    )
    canonical = parent / supplied.name
    if canonical != supplied:
        fail("output_root must not contain symlinked ancestors or lexical '..' components")
    return canonical


def canonical_contract_path(raw: object, *, require_absent: bool) -> Path:
    """Resolve the preregistered contract filename without following links."""

    if not isinstance(raw, str) or not raw.strip():
        fail("frozen_contract_path must be a non-empty absolute path")
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        fail(f"frozen_contract_path must be absolute: {supplied}")
    supplied = supplied.absolute()
    if supplied.exists() or supplied.is_symlink():
        if require_absent:
            fail(f"frozen_contract_path must not exist: {supplied}")
        if supplied.is_symlink() or not supplied.is_file():
            fail(f"frozen_contract_path is not a regular file: {supplied}")
        resolved = supplied.resolve(strict=True)
        if resolved != supplied:
            fail("frozen_contract_path must not contain symlinked ancestors")
        return resolved
    parent = canonical_path(
        str(supplied.parent),
        label="frozen_contract_path parent",
        kind="directory",
    )
    canonical = parent / supplied.name
    if canonical != supplied:
        fail("frozen_contract_path must not contain symlinked ancestors or lexical '..'")
    return canonical


def require_read_only(path: Path, *, label: str, recursive: bool = False) -> None:
    candidates = [path]
    if recursive:
        if not path.is_dir():
            fail(f"cannot recursively audit non-directory {label}: {path}")
        candidates.extend(sorted(path.rglob("*"), key=lambda item: item.as_posix()))
    writable = []
    for candidate in candidates:
        if candidate.is_symlink():
            fail(f"{label} contains a symlink: {candidate}")
        try:
            mode = stat.S_IMODE(candidate.lstat().st_mode)
        except OSError as exc:
            fail(f"cannot stat {label} entry {candidate}: {exc}")
        if mode & 0o222:
            writable.append(str(candidate))
    if writable:
        fail(f"{label} must be immutable; writable entries={writable[:8]}")


def tree_manifest(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
    include_size_total: bool = False,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"manifest root must be a regular directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            fail(f"manifest root contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            fail(f"manifest root contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        if Path(relative).name in excluded_names:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        fail(f"manifest root contains no files: {root}")
    result: dict[str, Any] = {
        "file_count": len(files),
        "files": files,
        "tree_sha256": sha256_json(files),
    }
    if include_size_total:
        result["size_bytes"] = sum(item["size_bytes"] for item in files)
        # Match the fixed-KL schema's stable key order only semantically; JSON
        # object order is deliberately irrelevant to hashing and comparison.
        result = {
            "file_count": result["file_count"],
            "size_bytes": result["size_bytes"],
            "tree_sha256": result["tree_sha256"],
            "files": result["files"],
        }
    return result


def selected_file_manifest(root: Path, names: Sequence[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        if not name or Path(name).name != name:
            fail(f"invalid source-artifact filename: {name!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            fail(f"missing regular source artifact: {path}")
        records.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "tree_sha256": sha256_json(records),
    }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def verify_self_hash(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256_json(unhashed):
        fail(f"{label} self-hash mismatch")


def _validate_qwen_bias_all_inspection(inspection: object) -> None:
    if not isinstance(inspection, dict):
        fail("conversion receipt lacks adapter inspection")
    config = inspection.get("adapter_config")
    tensors = inspection.get("tensors")
    if not isinstance(config, dict) or not isinstance(tensors, list):
        fail("conversion adapter inspection schema is incomplete")
    targets = config.get("target_modules")
    if (
        config.get("peft_type") != "LORA"
        or config.get("bias") != "all"
        or config.get("r") != 64
        or config.get("lora_alpha") != 64
        or not isinstance(targets, list)
        or set(targets) != EXPECTED_TARGET_MODULES
    ):
        fail("converted initialization is not exact Jiang r64/a64 bias=all LoRA")
    keys = [item.get("key") for item in tensors if isinstance(item, dict)]
    if len(keys) != len(tensors) or not all(isinstance(key, str) for key in keys):
        fail("converted initialization has an invalid tensor inventory")
    lora_count = sum(".lora_" in key for key in keys)
    bias_count = sum(key.endswith(".base_layer.bias") for key in keys)
    parameter_count = sum(item.get("numel", -1) for item in tensors if isinstance(item, dict))
    observed = {
        "lora_tensor_count": lora_count,
        "bias_tensor_count": bias_count,
        "trainable_tensor_count": len(tensors),
        "trainable_parameter_count": parameter_count,
    }
    if observed != EXPECTED_INIT_COUNTS:
        fail(f"converted initialization tensor scope mismatch: {observed}")


def validate_conversion(
    initialization_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    expected_receipt = initialization_path / CONVERSION_RECEIPT
    if receipt_path != expected_receipt:
        fail(f"conversion receipt must be exactly {expected_receipt}")
    require_read_only(initialization_path, label="converted initialization", recursive=True)
    receipt = load_json_object(receipt_path, label="conversion receipt")
    require_read_only(receipt_path, label="conversion receipt")
    verify_self_hash(receipt, label="conversion receipt")
    required_claims = {
        "schema_version": 1,
        "protocol": CONVERSION_PROTOCOL,
        "status": "converted_saved_policy_verified",
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
    }
    mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required_claims.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        fail(f"conversion claim boundary mismatch: {mismatches}")

    recorded_base = receipt.get("base")
    if not isinstance(recorded_base, dict):
        fail("conversion receipt lacks frozen base-model lineage")
    base_path = canonical_path(
        recorded_base.get("path"),
        label="conversion base model",
        kind="directory",
    )
    base_manifest = tree_manifest(base_path)
    if recorded_base.get("manifest") != base_manifest:
        fail("conversion base-model bytes differ from receipt")
    config_hash = sha256_file(base_path / "config.json")
    if recorded_base.get("config_sha256") != config_hash:
        fail("conversion base-model config hash mismatch")

    converted = receipt.get("converted_output")
    if not isinstance(converted, dict):
        fail("conversion receipt lacks converted_output")
    if Path(str(converted.get("path", ""))).expanduser().absolute() != initialization_path:
        fail("conversion receipt initialization path mismatch")
    if converted.get("serialization_bias_policy") != "all":
        fail("converted initialization receipt does not bind bias=all")
    live_manifest = tree_manifest(
        initialization_path,
        excluded_names=frozenset({CONVERSION_RECEIPT}),
    )
    if converted.get("manifest_excluding_receipt") != live_manifest:
        fail("converted initialization bytes differ from conversion receipt")
    _validate_qwen_bias_all_inspection(converted.get("checkpoint_fidelity_adapter_inspection"))
    verification = receipt.get("verification")
    required_verification = {
        "canonical_adapter_state_bitwise_equal": True,
        "qkv_biases_equal_frozen_base": True,
        "deterministic_logits_bitwise_equal": True,
        "deterministic_log_probabilities_bitwise_equal": True,
    }
    if not isinstance(verification, dict) or any(
        verification.get(key) != expected for key, expected in required_verification.items()
    ):
        fail("conversion receipt lacks exact saved-policy equivalence")
    return {
        "evidence_kind": "development_legacy_saved_policy_conversion",
        "checkpoint_path": str(initialization_path),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "base_model_lineage": {
            "path": str(base_path),
            "manifest": base_manifest,
            "config_sha256": config_hash,
        },
        "converted_tree": live_manifest,
        "claims": {
            "converted_saved_policy_fidelity": True,
            "terminal_sft_policy_fidelity": False,
            "publication_eligible": False,
            "sota_eligible": False,
        },
    }


def validate_relocated_adapter(
    adapter_path: Path,
    receipt_path: Path,
    tokenizer_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the exact no-merge development policy and its upstream lineage."""

    expected_adapter = (project_root / FROZEN_RELOCATED_ADAPTER_RELATIVE_PATH).resolve(strict=True)
    if adapter_path != expected_adapter:
        fail(f"pilot requires the frozen relocated adapter exactly at {expected_adapter}")
    expected_receipt = adapter_path / RELOCATION_RECEIPT
    if receipt_path != expected_receipt:
        fail(f"relocation receipt must be exactly {expected_receipt}")
    require_read_only(adapter_path, label="relocated development adapter", recursive=True)
    receipt = load_json_object(receipt_path, label="relocation receipt")
    if set(receipt) != RELOCATION_TOP_LEVEL_KEYS:
        fail("relocation receipt top-level schema mismatch")
    verify_self_hash(receipt, label="relocation receipt")
    required_claims = {
        "schema_version": RELOCATION_SCHEMA_VERSION,
        "protocol": RELOCATION_PROTOCOL,
        "status": RELOCATION_STATUS,
        "claim_scope": "development_only",
        "development_only": True,
        "relocation_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
        "receipt_sha256": FROZEN_RELOCATION_RECEIPT_SHA256,
    }
    mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required_claims.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        fail(f"relocation claim or identity mismatch: {mismatches}")
    if sha256_file(receipt_path) != FROZEN_RELOCATION_RECEIPT_FILE_SHA256:
        fail("relocation receipt file hash differs from the frozen 7B artifact")

    exact_source = project_root / "src/clasp/models/exact_peft_kl_reference.py"
    relocation_source = project_root / "scripts/relocate_development_adapter_base.py"
    training_fidelity_source = project_root / "src/clasp/models/training_fidelity.py"
    train_entrypoint_source = project_root / "src/clasp/entrypoints/train_grpo.py"
    frozen_sources = {
        exact_source: FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256,
        relocation_source: FROZEN_RELOCATION_IMPLEMENTATION_SHA256,
        training_fidelity_source: FROZEN_TRAINING_FIDELITY_SHA256,
        train_entrypoint_source: FROZEN_TRAIN_GRPO_SHA256,
    }
    drift = {
        str(path): {"actual": sha256_file(path), "expected": expected}
        for path, expected in frozen_sources.items()
        if sha256_file(path) != expected
    }
    if drift:
        fail(f"frozen exact-PEFT training implementation drift: {drift}")
    verifier = receipt.get("verifier")
    if verifier != {
        "protocol": RELOCATION_VERIFIER_PROTOCOL,
        "implementation": "scripts.relocate_development_adapter_base",
        "implementation_sha256": FROZEN_RELOCATION_IMPLEMENTATION_SHA256,
    }:
        fail("relocation receipt verifier identity mismatch")

    upstream = receipt.get("upstream_conversion")
    frozen_base = receipt.get("frozen_base")
    relocated = receipt.get("relocated_output")
    verification = receipt.get("verification")
    if not all(
        isinstance(value, dict) for value in (upstream, frozen_base, relocated, verification)
    ):
        fail("relocation receipt lacks source/base/output/verification evidence")
    base_path = canonical_path(
        frozen_base.get("path"), label="frozen relocated base", kind="directory"
    )
    require_read_only(base_path, label="frozen relocated base", recursive=True)
    if tokenizer_path != base_path:
        fail("tokenizer_path must be the immutable frozen base/tokenizer, not the adapter")
    base_manifest = tree_manifest(base_path, include_size_total=True)
    if (
        frozen_base.get("manifest") != base_manifest
        or base_manifest.get("tree_sha256") != FROZEN_BASE_TREE_SHA256
    ):
        fail("frozen relocated base bytes differ from the bound 7B artifact")
    if Path(str(relocated.get("path", ""))).expanduser().absolute() != adapter_path:
        fail("relocation output path mismatch")
    if (
        relocated.get("manifest_excluding_receipt", {}).get("tree_sha256")
        != FROZEN_RELOCATED_OUTPUT_TREE_SHA256
        or relocated.get("serialization_bias_policy") != "all"
    ):
        fail("relocated adapter tree or bias policy differs from the frozen artifact")
    full_state = verification.get("full_model_state")
    functional = verification.get("functional_equivalence")
    if (
        verification.get("requested_dtype") != RELOCATION_PROOF_DTYPE
        or verification.get("full_model_state_exact") is not True
        or verification.get("functional_equivalence_exact") is not True
        or not isinstance(full_state, dict)
        or full_state.get("tensor_inventory_sha256") != FROZEN_RELOCATION_STATE_INVENTORY_SHA256
        or not isinstance(functional, dict)
        or functional.get("logits", {}).get("value_sha256") != FROZEN_RELOCATION_LOGITS_SHA256
        or functional.get("bitwise_equal") is not True
        or functional.get("max_absolute_error") != 0.0
    ):
        fail("relocation FP32 state/probe proof differs from the frozen 7B evidence")

    source_adapter = canonical_path(
        upstream.get("source_adapter_path"),
        label="relocation upstream converted adapter",
        kind="directory",
    )
    conversion_receipt_path = canonical_path(
        upstream.get("authoritative_receipt_path"),
        label="relocation upstream conversion receipt",
        kind="file",
    )
    conversion = validate_conversion(source_adapter, conversion_receipt_path)
    if (
        upstream.get("source_adapter_manifest", {}).get("tree_sha256")
        != FROZEN_CONVERTED_SOURCE_TREE_SHA256
        or conversion["converted_tree"].get("tree_sha256") != FROZEN_CONVERTED_PAYLOAD_TREE_SHA256
        or upstream.get("authoritative_receipt_sha256") != conversion["receipt_sha256"]
        or upstream.get("authoritative_receipt_file_sha256") != conversion["receipt_file_sha256"]
    ):
        fail("relocation upstream converted-policy lineage differs from frozen evidence")

    try:
        verified = verify_development_relocation_authority(
            adapter_path,
            authoritative_receipt=receipt,
        )
        source_binding = build_source_binding(
            policy_adapter_path=adapter_path,
            reference_adapter_path=adapter_path,
            authoritative_receipt_path=receipt_path,
            authoritative_receipt=receipt,
            tokenizer_path=tokenizer_path,
            claim_scope="development",
        )
    except ExactPeftKLReferenceError as exc:
        fail(f"frozen exact-PEFT verifier rejected relocated adapter: {exc}")
    if verified != receipt:
        fail("strict relocation verifier result differs from authoritative receipt")
    if (
        source_binding.get("adapter_path") != str(adapter_path)
        or source_binding.get("base_model_path") != str(base_path)
        or source_binding.get("tokenizer_path") != str(tokenizer_path)
        or source_binding.get("authoritative_receipt_protocol") != RELOCATION_PROTOCOL
        or source_binding.get("authoritative_receipt_sha256") != FROZEN_RELOCATION_RECEIPT_SHA256
        or source_binding.get("authoritative_receipt_file_sha256")
        != FROZEN_RELOCATION_RECEIPT_FILE_SHA256
        or source_binding.get("saved_lora_dtype") != "F32"
        or source_binding.get("policy_adapter_autocast_dtype") is not True
        or source_binding.get("reference_adapter_autocast_dtype") is not True
    ):
        fail("exact-PEFT source binding does not match the relocated adapter contract")

    relocation = {
        "schema_version": RELOCATION_SCHEMA_VERSION,
        "protocol": RELOCATION_PROTOCOL,
        "status": RELOCATION_STATUS,
        "adapter_path": str(adapter_path),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": FROZEN_RELOCATION_RECEIPT_FILE_SHA256,
        "receipt_sha256": FROZEN_RELOCATION_RECEIPT_SHA256,
        "frozen_base_path": str(base_path),
        "tokenizer_path": str(tokenizer_path),
        "relocation_verification_dtype": RELOCATION_PROOF_DTYPE,
        "relocated_output_tree_sha256": FROZEN_RELOCATED_OUTPUT_TREE_SHA256,
        "converted_source_tree_sha256": FROZEN_CONVERTED_SOURCE_TREE_SHA256,
        "converted_payload_tree_sha256": FROZEN_CONVERTED_PAYLOAD_TREE_SHA256,
        "base_tree_sha256": FROZEN_BASE_TREE_SHA256,
        "state_inventory_sha256": FROZEN_RELOCATION_STATE_INVENTORY_SHA256,
        "probe_logits_sha256": FROZEN_RELOCATION_LOGITS_SHA256,
        "producer": dict(verifier),
        "exact_peft_source_binding": source_binding,
        "claims": {
            "converted_saved_policy_fidelity": True,
            "terminal_sft_policy_fidelity": False,
            "publication_eligible": False,
            "sota_eligible": False,
        },
    }
    return relocation, conversion


def audit_sealed_sft_target_saturation(
    summary_path: Path,
    details_path: Path,
) -> dict[str, Any]:
    """Recompute the sealed pre-pilot SFT target-occupancy diagnostics."""

    summary_file_sha256 = sha256_file(summary_path)
    details_file_sha256 = sha256_file(details_path)
    if summary_file_sha256 != SEALED_SFT_SUMMARY_SHA256:
        fail("sealed SFT summary identity differs from the preregistered artifact")
    if details_file_sha256 != SEALED_SFT_DETAILS_SHA256:
        fail("sealed SFT details identity differs from the preregistered artifact")
    summary = load_json_object(summary_path, label="sealed SFT summary")
    details_artifact = summary.get("details_artifact")
    overall = summary.get("overall")
    if not isinstance(details_artifact, dict) or not isinstance(overall, dict):
        fail("sealed SFT summary lacks overall/details evidence")
    if (
        details_artifact.get("sha256") != details_file_sha256
        or details_artifact.get("record_count") != SEALED_SFT_EXPECTED["example_count"]
        or details_artifact.get("basename") != details_path.name
    ):
        fail("sealed SFT summary does not bind the supplied details artifact")

    rows = load_json_lines(details_path, label="sealed SFT details")
    if len(rows) != SEALED_SFT_EXPECTED["example_count"]:
        fail("sealed SFT details do not contain exactly 200 examples")
    example_indices = [row.get("example_index") for row in rows]
    if example_indices != list(range(SEALED_SFT_EXPECTED["example_count"])):
        fail("sealed SFT details are not the ordered development slice [0, 200)")

    best_gaps: list[float] = []
    feasible_candidate_gaps: list[float] = []
    for index, row in enumerate(rows):
        if row.get("best_of_n_feasible") is not True:
            fail(f"sealed SFT row {index} is unexpectedly all-infeasible")
        best_gap = row.get("best_feasible_gap")
        candidates = row.get("candidates")
        if (
            not isinstance(best_gap, (int, float))
            or isinstance(best_gap, bool)
            or not math.isfinite(float(best_gap))
            or float(best_gap) < 0.0
            or not isinstance(candidates, list)
            or len(candidates) != 8
        ):
            fail(f"sealed SFT row {index} has invalid K=8 gap evidence")
        best_gaps.append(float(best_gap))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                fail(f"sealed SFT row {index} has an invalid candidate record")
            if candidate.get("is_feasible") is not True:
                continue
            gap = candidate.get("gap_to_ground_truth")
            if (
                not isinstance(gap, (int, float))
                or isinstance(gap, bool)
                or not math.isfinite(float(gap))
                or float(gap) < 0.0
            ):
                fail(f"sealed SFT row {index} has an invalid feasible-candidate gap")
            feasible_candidate_gaps.append(float(gap))

    target = float(FIXED_DESIGN["frontier_gap_target"])

    def occupancy(values: Sequence[float]) -> dict[str, Any]:
        return {
            format(multiplier, "g"): {
                "threshold_gap": target * multiplier,
                "count": sum(value <= target * multiplier for value in values),
                "rate": sum(value <= target * multiplier for value in values) / len(values),
            }
            for multiplier in FRONTIER_THRESHOLD_MULTIPLIERS
        }

    observed = {
        "example_count": len(rows),
        "best_of_8_feasible_count": len(best_gaps),
        "best_of_8_mean_gap": sum(best_gaps) / len(best_gaps),
        "best_of_8_zero_gap_count": sum(value == 0.0 for value in best_gaps),
        "best_of_8_at_or_below_target_count": sum(value <= target for value in best_gaps),
        "feasible_candidate_count": len(feasible_candidate_gaps),
        "feasible_candidate_mean_gap": (
            sum(feasible_candidate_gaps) / len(feasible_candidate_gaps)
        ),
        "feasible_candidate_zero_gap_count": sum(value == 0.0 for value in feasible_candidate_gaps),
        "feasible_candidate_at_or_below_target_count": sum(
            value <= target for value in feasible_candidate_gaps
        ),
    }
    for key, expected in SEALED_SFT_EXPECTED.items():
        actual = observed[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
                fail(f"sealed SFT saturation statistic changed: {key}={actual!r}")
        elif actual != expected:
            fail(f"sealed SFT saturation statistic changed: {key}={actual!r}")
    if (
        overall.get("examples") != observed["example_count"]
        or overall.get("best_of_n_feasible_examples") != observed["best_of_8_feasible_count"]
        or overall.get("feasible_candidates") != observed["feasible_candidate_count"]
        or not math.isclose(
            float(overall.get("mean_best_feasible_gap", float("nan"))),
            float(observed["best_of_8_mean_gap"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        fail("sealed SFT summary and recomputed target audit disagree")
    return {
        "development_only": True,
        "publication_eligible": False,
        "sota_eligible": False,
        "summary_path": str(summary_path),
        "summary_file_sha256": summary_file_sha256,
        "details_path": str(details_path),
        "details_file_sha256": details_file_sha256,
        "target_gap": target,
        "threshold_multipliers": list(FRONTIER_THRESHOLD_MULTIPLIERS),
        "observed": observed,
        "best_of_8_threshold_occupancy": occupancy(best_gaps),
        "feasible_candidate_threshold_occupancy": occupancy(feasible_candidate_gaps),
        "limitation": (
            "sealed SFT mean best-of-8 gap already clears 0.0107 and 64% of sets meet it; "
            "target attainment alone is not evidence of improvement"
        ),
    }


def _load_json_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {label} dataset {path}: {exc}")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        fail(f"{label} dataset must be a JSON list of objects: {path}")
    return rows


def _instance_digest(row: Mapping[str, Any], *, label: str) -> str:
    if "instance" not in row:
        fail(f"{label} row lacks canonical instance coordinates")
    return sha256_json({"task": "tsp", "instance": row["instance"]})


def audit_tsp_references(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    pattern = re.compile(r"\s*Route:\s*(\[[^\]]*\])\s*,\s*Objective:\s*([-+0-9.eE]+)\s*")
    objective_errors = []
    output_digests = []
    for index, row in enumerate(rows):
        coordinates = row.get("instance")
        output = row.get("output")
        if not isinstance(coordinates, list) or not isinstance(output, str):
            fail(f"{label} reference row {index} lacks coordinates or output")
        if not coordinates or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in point
            )
            for point in coordinates
        ):
            fail(f"{label} reference row {index} has invalid coordinates")
        matched = pattern.fullmatch(output)
        if matched is None:
            fail(f"{label} reference row {index} has an unsupported output format")
        try:
            route = ast.literal_eval(matched.group(1))
            declared_objective = float(matched.group(2))
        except (SyntaxError, ValueError) as exc:
            fail(f"{label} reference row {index} cannot be parsed: {exc}")
        node_count = len(coordinates)
        if (
            not isinstance(route, list)
            or len(route) != node_count + 1
            or route[0] != route[-1]
            or any(isinstance(node, bool) or not isinstance(node, int) for node in route)
            or len(set(route[:-1])) != node_count
            or set(route[:-1]) != set(range(node_count))
        ):
            fail(f"{label} reference row {index} is not a feasible TSP tour")
        recomputed = sum(
            math.hypot(
                float(coordinates[left][0]) - float(coordinates[right][0]),
                float(coordinates[left][1]) - float(coordinates[right][1]),
            )
            for left, right in zip(route, route[1:])
        )
        if not math.isfinite(declared_objective) or declared_objective <= 0.0:
            fail(f"{label} reference row {index} has an invalid declared objective")
        error = abs(recomputed - declared_objective)
        # Dataset objectives are serialized to three decimal places.
        if error > 0.00051:
            fail(
                f"{label} reference row {index} objective replay mismatch: "
                f"declared={declared_objective}, recomputed={recomputed}, error={error}"
            )
        objective_errors.append(error)
        output_digests.append(sha256_json({"row_index": index, "output": output}))
    return {
        "status": "all_references_strictly_replayed",
        "rows": len(rows),
        "feasible_rate": 1.0,
        "objective_rounding_tolerance": 0.00051,
        "objective_replay_error_max": max(objective_errors, default=0.0),
        "reference_output_set_sha256": sha256_json(output_digests),
        "reference_class": "uncertified_dataset_heuristic",
        "optimality_certified": False,
        "gap_interpretation": "gap_to_frozen_feasible_heuristic_reference",
    }


def validate_data(
    train_path: Path,
    development_path: Path,
) -> dict[str, Any]:
    if not (
        train_path.parent.name == "train"
        and train_path.parent.parent.name == "tsp"
        and development_path.parent.name == "eval"
        and development_path.parent.parent.name == "tsp"
    ):
        fail(
            "data paths must use distinct <root>/tsp/train/<file>.json and "
            "<root>/tsp/eval/<file>.json layouts"
        )
    if train_path.parents[2] == development_path.parents[2]:
        fail("RL training and frozen terminal development data must use distinct roots")
    require_read_only(train_path, label="RL train data")
    require_read_only(development_path, label="development data")
    train_rows = _load_json_rows(train_path, label="RL train")
    development_rows = _load_json_rows(development_path, label="development")
    start = FIXED_DESIGN["development_slice"]["start_index"]
    count = FIXED_DESIGN["development_slice"]["count"]
    selected = development_rows[start : start + count]
    if len(train_rows) != 3_200:
        fail(f"pilot requires exactly 3200 frozen RL train rows; observed={len(train_rows)}")
    if len(selected) != count:
        fail(
            f"development source lacks frozen [{start}, {start + count}) slice; "
            f"source_rows={len(development_rows)}"
        )
    train_ids = [_instance_digest(row, label="RL train") for row in train_rows]
    development_ids = [_instance_digest(row, label="development") for row in selected]
    if len(set(train_ids)) != len(train_ids):
        fail("RL train data contains duplicate semantic TSP instances")
    if len(set(development_ids)) != len(development_ids):
        fail("development slice contains duplicate semantic TSP instances")
    overlap = sorted(set(train_ids) & set(development_ids))
    if overlap:
        fail(f"development slice overlaps RL training data: {overlap[:8]}")
    train_reference_audit = audit_tsp_references(train_rows, label="RL train")
    development_reference_audit = audit_tsp_references(
        selected,
        label="development",
    )
    return {
        "identity_method": "sha256(canonical_json({task:'tsp',instance:row.instance}))",
        "train": {
            "path": str(train_path),
            "size_bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
            "rows": len(train_rows),
            "identity_set_sha256": sha256_json(sorted(train_ids)),
            "reference_audit": train_reference_audit,
        },
        "development": {
            "source_path": str(development_path),
            "source_size_bytes": development_path.stat().st_size,
            "source_sha256": sha256_file(development_path),
            "source_rows": len(development_rows),
            "selected_interval": [start, start + count],
            "selected_rows": count,
            "selected_slice_sha256": sha256_json(selected),
            "identity_set_sha256": sha256_json(sorted(development_ids)),
            "reference_audit": development_reference_audit,
        },
        "disjointness": {"overlap_count": 0, "duplicate_count": 0},
    }


def source_manifest(project_root: Path) -> dict[str, Any]:
    records = []
    for relative in SOURCE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            fail(f"required pilot source file is missing or linked: {path}")
        digest = sha256_file(path)
        frozen = FROZEN_CORE_SOURCE_SHA256.get(relative)
        if frozen is not None and digest != frozen:
            if relative == "src/clasp/entrypoints/train_grpo.py":
                active = identify_train_grpo_source(project_root)
                if active == FOARL_F638_CORRECTED_TRAINER_SOURCE:
                    fail(
                        "historical trainer source is unavailable from this mutable tree: "
                        f"the TSP FCDF pilot remains frozen at {frozen}, while the live file is "
                        f"the explicitly versioned FOARL successor {active.version} at {digest}. "
                        "The old pilot must not be rehashed or executed under successor code"
                    )
            fail(
                f"frozen fidelity/training implementation drift for {relative}: "
                f"expected={frozen}, observed={digest}"
            )
        records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest})
    return {"files": records, "manifest_sha256": sha256_json(records)}


def _path_from_spec(paths: Mapping[str, Any], key: str, kind: str) -> Path:
    return canonical_path(paths.get(key), label=key, kind=kind)


def _fixed_cells() -> list[dict[str, Any]]:
    cells = []
    for method in METHODS:
        for beta in BETAS:
            beta_code = "0" if beta == 0.0 else "5"
            for seed in SEEDS:
                cells.append(
                    {
                        "cell": f"{method['code']}{beta_code}",
                        "trajectory_id": f"{method['code']}{beta_code}-seed{seed}",
                        "method": method["name"],
                        "grpo_beta": beta,
                        "seed": seed,
                    }
                )
    return cells


def build_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    if (
        spec.get("schema_version") != SCHEMA_VERSION
        or spec.get("protocol") != SPEC_PROTOCOL
        or spec.get("status") != "READY_TO_FREEZE"
    ):
        fail("spec is a HOLD template or has an unsupported schema/status")
    campaign_id = spec.get("campaign_id")
    if not isinstance(campaign_id, str) or CAMPAIGN_ID.fullmatch(campaign_id) is None:
        fail("campaign_id must be 8-96 lowercase alphanumeric/underscore/hyphen characters")
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        fail("spec.paths must be an object")
    project_root = _path_from_spec(paths, "project_root", "directory")
    python_executable = executable_path(
        paths.get("python_executable"),
        label="python_executable",
    )
    relocated_adapter = _path_from_spec(paths, "relocated_development_adapter", "directory")
    relocation_receipt = _path_from_spec(paths, "relocation_receipt", "file")
    tokenizer_path = _path_from_spec(paths, "tokenizer_path", "directory")
    train_path = _path_from_spec(paths, "rl_train_file", "file")
    development_path = _path_from_spec(paths, "development_source_file", "file")
    sealed_sft_summary = _path_from_spec(paths, "sealed_sft_summary", "file")
    sealed_sft_details = _path_from_spec(paths, "sealed_sft_details", "file")

    output_root = canonical_output_path(paths.get("output_root"), require_absent=True)
    frozen_contract_path = canonical_contract_path(
        paths.get("frozen_contract_path"),
        require_absent=True,
    )
    if frozen_contract_path == output_root or output_root in frozen_contract_path.parents:
        fail("frozen_contract_path must be outside the mutable pilot output_root")

    relocation, conversion = validate_relocated_adapter(
        relocated_adapter,
        relocation_receipt,
        tokenizer_path,
        project_root=project_root,
    )
    data = validate_data(train_path, development_path)
    saturation_audit = audit_sealed_sft_target_saturation(
        sealed_sft_summary,
        sealed_sft_details,
    )
    runtime_identity = probe_runtime_identity(python_executable)
    hardware_identity = probe_local_hardware(python_executable)
    source = source_manifest(project_root)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": CONTRACT_PROTOCOL,
        "status": "frozen_before_development_results",
        "stage": "grpo",
        "fixed_horizon": {"max_steps": 400, "world_size": 1},
        "campaign_id": campaign_id,
        "frozen_at_utc": now,
        "evidence_boundary": clone_json(EVIDENCE_BOUNDARY),
        "paths": {
            "project_root": str(project_root),
            "python_executable": str(python_executable),
            "frozen_contract_path": str(frozen_contract_path),
            "output_root": str(output_root),
        },
        "runtime_execution": {
            "runtime": runtime_identity,
            "hardware": hardware_identity,
            "closed_environment": {
                "inherit_parent_environment": False,
                "env_i_required": True,
                "path": CLOSED_ENVIRONMENT_PATH,
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_visible_devices": "3",
                "pytorch_cuda_alloc_conf": "expandable_segments:True",
                "workspace_bound_per_attempt_tmpdir": True,
                "minimum_workspace_free_bytes": MIN_WORKSPACE_FREE_BYTES,
                "tempfile_gettempdir_must_equal_bound_tmpdir": True,
                "home_and_all_named_caches_must_be_descendants_of_bound_tmpdir": True,
                "root_tmp_fallback": False,
            },
        },
        "converted_source_lineage": conversion,
        "policy_initialization": {
            "kind": "continue_common_relocated_bias_all_saved_policy_adapter",
            "model_path": relocation["adapter_path"],
            "tokenizer_path": relocation["tokenizer_path"],
            "relocation_verification_dtype": RELOCATION_PROOF_DTYPE,
            "training_load_dtype": TRAINING_LOAD_DTYPE,
            "common_bfloat16_load_not_exact_fp32_source_policy_equivalence": True,
            "exact_source_policy_equivalence_claim_allowed": False,
            "relocation_receipt_sha256": relocation["receipt_sha256"],
            "relocation_receipt_file_sha256": relocation["receipt_file_sha256"],
            "converted_source_receipt_sha256": conversion["receipt_file_sha256"],
            "strict_relocation_reauthorization_required_before_memory_gate": True,
            "attach_fresh_rl_lora": False,
            "continue_existing_sft_lora": True,
            "rl_lora_bias_policy": "all",
            "optimization_geometry": (
                "continue_nonzero_SFT_LoRA_A_B_and_saved_base_layer_biases_in_place"
            ),
            "identical_across_all_cells": True,
        },
        "relocated_saved_policy": relocation,
        "exact_peft_kl_contract": {
            "beta_zero_reference": None,
            "beta_positive_reference_adapter_path": relocation["adapter_path"],
            "beta_positive_reference_receipt_path": relocation["receipt_path"],
            "policy_and_reference_are_separate_runtime_loads": True,
            "runtime_protocol": EXACT_PEFT_DEVELOPMENT_PROTOCOL,
            "runtime_status": EXACT_PEFT_DEVELOPMENT_STATUS,
            "runtime_receipt_filename": EXACT_PEFT_RUNTIME_RECEIPT,
            "verifier_protocol": EXACT_PEFT_VERIFIER_PROTOCOL,
            "verifier_implementation_sha256": FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256,
            "post_step_schema_version": EXACT_PEFT_POST_SCHEMA_VERSION,
            "post_step_protocol": EXACT_PEFT_POST_DEVELOPMENT_PROTOCOL,
            "post_step_status": EXACT_PEFT_POST_DEVELOPMENT_STATUS,
            "post_step_verifier_protocol": EXACT_PEFT_POST_VERIFIER_PROTOCOL,
            "post_step_verifier_implementation_sha256": (
                FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256
            ),
            "step1_receipt_filename": EXACT_PEFT_STEP1_RECEIPT,
            "post_step_receipt_filename": EXACT_PEFT_POST_RECEIPT,
            "parameterization": "continue_same_bias_all_sft_adapter",
            "sync_ref_model": False,
            "use_vllm": False,
            "policy_adapter_autocast_dtype": True,
            "reference_adapter_autocast_dtype": True,
        },
        "memory_gate": clone_json(MEMORY_GATE),
        "data": data,
        "target_saturation_preregistration": clone_json(TARGET_SATURATION_PREREGISTRATION),
        "sealed_sft_target_audit": saturation_audit,
        "source": source,
        "design": clone_json({**FIXED_DESIGN, "cells": _fixed_cells()}),
        "analysis": clone_json(ANALYSIS_PLAN),
        "required_telemetry": clone_json(REQUIRED_TELEMETRY),
        "execution_contract": clone_json(EXECUTION_CONTRACT),
        "required_output_evidence": clone_json(REQUIRED_OUTPUT_EVIDENCE),
    }
    payload["contract_sha256"] = sha256_json(payload)
    return payload


def verify_contract(
    contract: Mapping[str, Any],
    *,
    require_fresh_output: bool = True,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    expected_top_level = {
        "schema_version",
        "protocol",
        "status",
        "stage",
        "fixed_horizon",
        "campaign_id",
        "frozen_at_utc",
        "evidence_boundary",
        "paths",
        "converted_source_lineage",
        "policy_initialization",
        "relocated_saved_policy",
        "exact_peft_kl_contract",
        "memory_gate",
        "data",
        "target_saturation_preregistration",
        "sealed_sft_target_audit",
        "source",
        "design",
        "analysis",
        "required_telemetry",
        "execution_contract",
        "required_output_evidence",
        "runtime_execution",
        "contract_sha256",
    }
    if set(contract) != expected_top_level:
        fail("pilot contract top-level schema mismatch")
    claimed = contract.get("contract_sha256")
    unhashed = dict(contract)
    unhashed.pop("contract_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256_json(unhashed):
        fail("pilot contract self-hash mismatch")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("protocol") != CONTRACT_PROTOCOL
        or contract.get("status") != "frozen_before_development_results"
        or contract.get("stage") != "grpo"
        or contract.get("fixed_horizon") != {"max_steps": 400, "world_size": 1}
    ):
        fail("unsupported or non-frozen pilot contract")
    campaign_id = contract.get("campaign_id")
    if not isinstance(campaign_id, str) or CAMPAIGN_ID.fullmatch(campaign_id) is None:
        fail("pilot contract campaign_id is invalid")
    frozen_at = contract.get("frozen_at_utc")
    try:
        parsed_frozen_at = dt.datetime.fromisoformat(str(frozen_at))
    except ValueError:
        fail("pilot contract frozen_at_utc is invalid")
    if parsed_frozen_at.tzinfo is None or parsed_frozen_at.utcoffset() != dt.timedelta(0):
        fail("pilot contract frozen_at_utc must be timezone-aware UTC")
    if contract.get("evidence_boundary") != EVIDENCE_BOUNDARY:
        fail("pilot evidence boundary was weakened")
    design = contract.get("design")
    expected_design = {**FIXED_DESIGN, "cells": _fixed_cells()}
    if design != expected_design:
        fail("pilot fixed design differs from preregistered 2x2 constants")
    if contract.get("analysis") != ANALYSIS_PLAN:
        fail("pilot analysis or selection plan differs from preregistration")
    if contract.get("target_saturation_preregistration") != (TARGET_SATURATION_PREREGISTRATION):
        fail("pilot target-saturation plan differs from preregistration")
    if contract.get("required_telemetry") != REQUIRED_TELEMETRY:
        fail("pilot required telemetry differs from preregistration")
    if contract.get("execution_contract") != EXECUTION_CONTRACT:
        fail("pilot dry-run execution boundary differs from preregistration")
    if contract.get("required_output_evidence") != REQUIRED_OUTPUT_EVIDENCE:
        fail("pilot checkpoint-fidelity postcondition differs from preregistration")
    if contract.get("memory_gate") != MEMORY_GATE:
        fail("pilot one-step beta memory gate differs from preregistration")

    paths = contract.get("paths")
    runtime_execution = contract.get("runtime_execution")
    conversion_lineage = contract.get("converted_source_lineage")
    policy_initialization = contract.get("policy_initialization")
    relocation = contract.get("relocated_saved_policy")
    exact_kl = contract.get("exact_peft_kl_contract")
    data = contract.get("data")
    saturation_audit = contract.get("sealed_sft_target_audit")
    if not isinstance(runtime_execution, dict) or not all(
        isinstance(value, dict)
        for value in (
            paths,
            conversion_lineage,
            policy_initialization,
            relocation,
            exact_kl,
            data,
            saturation_audit,
        )
    ):
        fail("pilot contract lacks path/artifact/data objects")
    project_root = canonical_path(paths.get("project_root"), label="project_root", kind="directory")
    observed_python = executable_path(
        paths.get("python_executable"),
        label="python_executable",
    )
    if str(observed_python) != paths.get("python_executable"):
        fail("pilot Python executable path changed")
    expected_runtime_execution = {
        "runtime": probe_runtime_identity(observed_python),
        "hardware": probe_local_hardware(observed_python),
        "closed_environment": {
            "inherit_parent_environment": False,
            "env_i_required": True,
            "path": CLOSED_ENVIRONMENT_PATH,
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "3",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "workspace_bound_per_attempt_tmpdir": True,
            "minimum_workspace_free_bytes": MIN_WORKSPACE_FREE_BYTES,
            "tempfile_gettempdir_must_equal_bound_tmpdir": True,
            "home_and_all_named_caches_must_be_descendants_of_bound_tmpdir": True,
            "root_tmp_fallback": False,
        },
    }
    if runtime_execution != expected_runtime_execution:
        fail("live runtime/hardware identity differs from frozen pilot contract")
    output_root = canonical_output_path(
        paths.get("output_root"),
        require_absent=require_fresh_output,
    )
    if str(output_root) != paths.get("output_root"):
        fail("contract output_root path changed")
    recorded_contract_path = paths.get("frozen_contract_path")
    recorded_exists = (
        isinstance(recorded_contract_path, str) and Path(recorded_contract_path).exists()
    )
    expected_contract_path = canonical_contract_path(
        recorded_contract_path,
        require_absent=not recorded_exists,
    )
    if expected_contract_path == output_root or output_root in expected_contract_path.parents:
        fail("frozen contract must remain outside the mutable pilot output_root")
    if contract_path is not None:
        supplied_contract_path = contract_path.expanduser().absolute()
        if supplied_contract_path != expected_contract_path:
            fail("loaded pilot contract path differs from preregistered frozen_contract_path")
        if supplied_contract_path.exists():
            require_read_only(supplied_contract_path, label="frozen pilot contract")

    init_path = canonical_path(
        conversion_lineage.get("checkpoint_path"),
        label="converted source-lineage checkpoint",
        kind="directory",
    )
    conversion_receipt = canonical_path(
        conversion_lineage.get("receipt_path"),
        label="conversion receipt",
        kind="file",
    )
    observed_init = validate_conversion(init_path, conversion_receipt)
    if observed_init != conversion_lineage:
        fail("live converted source-lineage evidence differs from frozen contract")
    relocated_adapter = canonical_path(
        relocation.get("adapter_path"),
        label="relocated development adapter",
        kind="directory",
    )
    relocation_receipt = canonical_path(
        relocation.get("receipt_path"),
        label="relocation receipt",
        kind="file",
    )
    tokenizer_path = canonical_path(
        relocation.get("tokenizer_path"),
        label="frozen base/tokenizer",
        kind="directory",
    )
    observed_relocation, relocation_conversion = validate_relocated_adapter(
        relocated_adapter,
        relocation_receipt,
        tokenizer_path,
        project_root=project_root,
    )
    if observed_relocation != relocation:
        fail("live relocated saved-policy evidence differs from frozen contract")
    if relocation_conversion != observed_init:
        fail("relocation upstream lineage differs from frozen converted source")
    expected_policy_initialization = {
        "kind": "continue_common_relocated_bias_all_saved_policy_adapter",
        "model_path": observed_relocation["adapter_path"],
        "tokenizer_path": observed_relocation["tokenizer_path"],
        "relocation_verification_dtype": RELOCATION_PROOF_DTYPE,
        "training_load_dtype": TRAINING_LOAD_DTYPE,
        "common_bfloat16_load_not_exact_fp32_source_policy_equivalence": True,
        "exact_source_policy_equivalence_claim_allowed": False,
        "relocation_receipt_sha256": observed_relocation["receipt_sha256"],
        "relocation_receipt_file_sha256": observed_relocation["receipt_file_sha256"],
        "converted_source_receipt_sha256": observed_init["receipt_file_sha256"],
        "strict_relocation_reauthorization_required_before_memory_gate": True,
        "attach_fresh_rl_lora": False,
        "continue_existing_sft_lora": True,
        "rl_lora_bias_policy": "all",
        "optimization_geometry": (
            "continue_nonzero_SFT_LoRA_A_B_and_saved_base_layer_biases_in_place"
        ),
        "identical_across_all_cells": True,
    }
    if policy_initialization != expected_policy_initialization:
        fail("policy initialization does not bind continued relocated-adapter geometry")
    expected_exact_kl = {
        "beta_zero_reference": None,
        "beta_positive_reference_adapter_path": observed_relocation["adapter_path"],
        "beta_positive_reference_receipt_path": observed_relocation["receipt_path"],
        "policy_and_reference_are_separate_runtime_loads": True,
        "runtime_protocol": EXACT_PEFT_DEVELOPMENT_PROTOCOL,
        "runtime_status": EXACT_PEFT_DEVELOPMENT_STATUS,
        "runtime_receipt_filename": EXACT_PEFT_RUNTIME_RECEIPT,
        "verifier_protocol": EXACT_PEFT_VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256,
        "post_step_schema_version": EXACT_PEFT_POST_SCHEMA_VERSION,
        "post_step_protocol": EXACT_PEFT_POST_DEVELOPMENT_PROTOCOL,
        "post_step_status": EXACT_PEFT_POST_DEVELOPMENT_STATUS,
        "post_step_verifier_protocol": EXACT_PEFT_POST_VERIFIER_PROTOCOL,
        "post_step_verifier_implementation_sha256": (FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256),
        "step1_receipt_filename": EXACT_PEFT_STEP1_RECEIPT,
        "post_step_receipt_filename": EXACT_PEFT_POST_RECEIPT,
        "parameterization": "continue_same_bias_all_sft_adapter",
        "sync_ref_model": False,
        "use_vllm": False,
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
    }
    if exact_kl != expected_exact_kl:
        fail("exact no-merge PEFT KL runtime contract differs from frozen constants")
    train_path = canonical_path(
        data.get("train", {}).get("path") if isinstance(data.get("train"), dict) else None,
        label="RL train file",
        kind="file",
    )
    development_path = canonical_path(
        data.get("development", {}).get("source_path")
        if isinstance(data.get("development"), dict)
        else None,
        label="development source file",
        kind="file",
    )
    if validate_data(train_path, development_path) != data:
        fail("live data evidence differs from frozen contract")
    summary_path = canonical_path(
        saturation_audit.get("summary_path"),
        label="sealed SFT summary",
        kind="file",
    )
    details_path = canonical_path(
        saturation_audit.get("details_path"),
        label="sealed SFT details",
        kind="file",
    )
    if audit_sealed_sft_target_saturation(summary_path, details_path) != saturation_audit:
        fail("live sealed SFT target-saturation evidence differs from frozen contract")
    if source_manifest(project_root) != contract.get("source"):
        fail("pilot source bytes differ from frozen contract")
    return dict(contract)


def write_contract_create_once(path: Path, contract: Mapping[str, Any]) -> None:
    output = path.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite pilot contract: {output}")
    if not output.parent.is_dir():
        fail(f"pilot contract parent must already exist: {output.parent}")
    encoded = (
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        output.chmod(stat.S_IRUSR | stat.S_IRGRP)
    except Exception:
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an immutable development-only TSP Frontier-CDF x beta pilot; "
            "this command never launches training or evaluation."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_json_object(args.spec.expanduser().absolute(), label="pilot spec")
        contract = build_contract(spec)
        expected_output = Path(contract["paths"]["frozen_contract_path"])
        supplied_output = args.output.expanduser().absolute()
        if supplied_output != expected_output:
            fail(
                "--output must equal spec.paths.frozen_contract_path exactly: "
                f"expected={expected_output} actual={supplied_output}"
            )
        verify_contract(
            contract,
            require_fresh_output=True,
            contract_path=supplied_output,
        )
        write_contract_create_once(supplied_output, contract)
    except (ContractError, FileExistsError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": contract["status"],
                "development_only": True,
                "trajectory_count": contract["design"]["trajectory_count"],
                "contract": str(args.output.expanduser().absolute()),
                "contract_sha256": contract["contract_sha256"],
                "training_or_evaluation_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
