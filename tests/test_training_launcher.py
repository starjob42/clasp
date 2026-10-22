from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_clasp_sota_a100.slurm"
POLICY = ROOT / "experiments" / "sota_20260814" / "advancement_policy_40_step_v3.json"
WRAPPER = ROOT / "scripts" / "run_grpo_in_sealed_environment.py"
REWARD_HEADS = ROOT / "src" / "clasp" / "rewards" / "heads.py"
CO_TASKS = ROOT / "src" / "clasp" / "rewards" / "co_tasks.py"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_publication_launcher_requires_new_explicit_storage_and_frozen_source() -> None:
    text = _text()
    assert "CLASP_STORAGE_ROOT:?CLASP_STORAGE_ROOT is required" in text
    assert "SOTA_CAMPAIGN_ID:?SOTA_CAMPAIGN_ID is required" in text
    assert "SOTA_ADVANCEMENT_POLICY:?SOTA_ADVANCEMENT_POLICY is required" in text
    assert "SOTA_PRIOR_STAGE_RECEIPTS_JSON:-[]" in text
    assert "SOTA_FORBIDDEN_SOURCE_REGISTRY:?SOTA_FORBIDDEN_SOURCE_REGISTRY is required" in text
    assert (
        "SOTA_MODEL_INITIALIZATION_REGISTRY:?SOTA_MODEL_INITIALIZATION_REGISTRY is required" in text
    )
    assert "SOTA_BASE_MODEL_PATH:?SOTA_BASE_MODEL_PATH is required" in text
    assert "SOTA_REMOTE_ENVIRONMENT_RECEIPT:?SOTA_REMOTE_ENVIRONMENT_RECEIPT is required" in text
    assert "SOTA_SEALED_ENV:?SOTA_SEALED_ENV is required" in text
    assert "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" in text
    assert '--source-manifest "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256"' in text
    assert '--forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}"' in text
    assert '--model-initialization-registry "${MODEL_INITIALIZATION_REGISTRY}"' in text
    assert '--sft-data-path "${SFT_DATA_ROOT}/${TASK}/train"' in text
    assert '--data-path "${TASK_DATA_ROOT}/${TASK}/train"' in text
    assert "clasp_sota_20260814_v1" not in text
    assert "models/Qwen2.5-7B-Instruct" not in text


def test_wandb_identity_is_unique_to_the_campaign_and_trajectory() -> None:
    text = _text()
    assert 'WANDB_RUN_NAME_OVERRIDE="${CAMPAIGN_ID}-${TASK}-${METHOD}-seed${TRAIN_SEED}"' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_NAME_OVERRIDE}"' in text
    assert '--config "campaign_id=${CAMPAIGN_ID}"' in text


def test_publication_launcher_retains_every_scheduled_checkpoint() -> None:
    text = _text()
    assert 'SAVE_STEPS="${SOTA_SAVE_STEPS:-400}"' in text
    assert 'SAVE_TOTAL_LIMIT="${SOTA_SAVE_TOTAL_LIMIT:-$((FINAL_STEP / SAVE_STEPS))}"' in text
    assert "SAVE_TOTAL_LIMIT >= FINAL_STEP / SAVE_STEPS" in text
    assert '--save_total_limit "${SAVE_TOTAL_LIMIT}"' in text
    assert '--config "logging_steps=1" --config "save_total_limit=${SAVE_TOTAL_LIMIT}"' in text


def test_publication_launcher_stops_only_on_saved_evaluated_boundaries() -> None:
    text = _text()
    assert "CHUNK_STEPS % SAVE_STEPS == 0" in text
    assert "CHUNK_STEPS % EVAL_STEPS == 0" in text
    assert "stop_step % SAVE_STEPS == 0 && stop_step % EVAL_STEPS == 0" in text
    assert "SOTA_STOP_AT_GLOBAL_STEP" not in text
    assert 'export CLASP_TRAINING_RUN_CONTRACT_PATH="${RUN_DIR}/run_contract.json"' in text
    assert '--max_steps "${FINAL_STEP}"' in text
    assert 'export GRPO_STOP_AT_GLOBAL_STEP="${stop_step}"' in text


def test_publication_launcher_binds_the_preregistered_matrix() -> None:
    text = _text()
    for field in (
        "policy_sha256",
        "frozen_before_results",
        "campaign_id",
        "expected_tasks",
        "expected_methods",
        "train_seeds",
        "target_step",
        "contract_requirements",
        "chunk_steps",
        "save_total_limit",
        "source_snapshot_id",
        "artifact_root",
        "model_initialization_registry_sha256",
        "stage_index",
        "prior_stage_receipt_chain",
    ):
        assert field in text
    assert '--config "advancement_policy_sha256=${ADVANCEMENT_POLICY_SHA256}"' in text
    assert '--config "advancement_stage_index=${ADVANCEMENT_STAGE_INDEX}"' in text
    assert (
        '--config "advancement_prior_stage_receipt_chain_sha256='
        '${ADVANCEMENT_PRIOR_STAGE_RECEIPT_CHAIN_SHA256}"' in text
    )
    assert "_prior_stage_authorization_audit" in text
    assert 'prior_authorization["errors"]' in text
    assert '"clasp-model-initialization-registry-v2"' in text
    assert '"frozen_before_training"' in text
    assert "CLASP_PUBLICATION_FIDELITY=1" in text
    assert '--config "checkpoint_fidelity=publication_exact_required"' in text


def test_publication_launcher_binds_positive_beta_to_an_exact_sft_adapter_pair() -> None:
    text = _text()
    assert 'print("1" if beta > 0.0 else "0")' in text
    assert "beta>0 publication is blocked" not in text
    assert 'TRAINING_MODEL_PATH="${SFT_MODEL_PATH}"' in text
    assert '[[ "${KL_REFERENCE_MODEL_PATH}" == "${SFT_MODEL_PATH}" ]]' in text
    assert (
        '[[ "${KL_REFERENCE_RECEIPT}" == "${SFT_MODEL_PATH}/checkpoint_fidelity.json" ]]'
        in text
    )
    assert '--model_path "${TRAINING_MODEL_PATH}"' in text
    assert '--tokenizer_path "${TRAINING_MODEL_PATH}"' in text
    assert '--grpo_beta "${BETA}"' in text
    assert '--fixed-kl-reference-model-path "${KL_REFERENCE_MODEL_PATH}"' in text
    assert '--fixed-kl-reference-receipt-path "${KL_REFERENCE_RECEIPT}"' in text
    assert "ALLOWED_ENV+=(CLASP_KL_REFERENCE_MODEL_PATH CLASP_KL_REFERENCE_RECEIPT)" in text
    assert 'ENTRYPOINT=clasp.entrypoints.train_grpo' in text
    assert "train_grpo_standard" not in text


def test_environment_scrub_cannot_unset_resolved_internal_paths_or_sizes() -> None:
    text = _text()
    pre_scrub = text.split("# sbatch --export=ALL must not inject", 1)[0]
    assert 'TASK_DATA_ROOT="${GRPO_DATA_ROOT:-' in pre_scrub
    assert "ROLLOUT_BATCH_SIZE" not in pre_scrub
    post_scrub = text.split("done < <(env)", 1)[1]
    assert "NUM_GENERATIONS=9" in post_scrub
    assert "DEPLOYMENT_SET_SIZE=8" in post_scrub
    assert "ROLLOUT_BATCH_SIZE=36" in post_scrub
    assert "${TASK_DATA_ROOT}/${TASK}" in text
    for line in pre_scrub.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name = line.split("=", 1)[0].strip()
        assert not name.startswith(("CLASP_", "PD_", "GENERATION_", "GRPO_"))


def test_v3_smoke_policy_is_self_hashed_and_matches_the_launch_namespace() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    stored = policy.pop("policy_sha256")
    observed = hashlib.sha256(
        json.dumps(
            policy,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert stored == observed == "1aaab2942e118bb28c1ffcde03c159debd4d53cd391d1bd5dbf9684feebd0298"
    assert policy["status"] == "frozen_before_results"
    assert policy["campaign_id"] == "clasp_sota_smoke_20260814_v3"
    assert policy["source_snapshot_id"] == "correct_clasp_sota_20260814_v3"
    assert policy["artifact_root"].endswith("/clasp_sota_smoke_20260814_v3")
    assert policy["chunk_steps"] == 40
    assert policy["save_total_limit"] == 1
    assert "model_initialization_registry_sha256" not in policy
    assert "stage_index" not in policy
    assert "prior_stage_receipt_chain" not in policy
    # The launcher delegates to the auditor's strict current policy validator,
    # so this immutable historical policy remains non-runnable.
    assert "advancement_audit._validate_policy" in _text()


def test_canonical_publication_arm_inventory_is_explicit_and_legacy_is_retired() -> None:
    text = _text()
    for method in (
        "grpo_z",
        "dual_scalar_z",
        "dual_separate_z",
        "clasp_canonical",
        "clasp_canonical_fixed_lambda",
        "clasp_canonical_no_recovery",
        "set_null",
    ):
        assert method in text
    assert "is a retired legacy arm" in text
    assert "SOTA_FIXED_NULL_RECEIPT is required for set_null" in text
    assert "fixed_null_k8_same_set_utility_v1" in text
    assert "--fixed-null-receipt" in text
    assert "--clasp_fixed_null_set" in text
    assert 'policy.get("fixed_null_receipt_sha256_by_task")' in text
    assert 'fixed_null_identity["receipt_sha256"] == registered_receipt_sha256' in text
    assert "group_z_reward_minus_group_z_positive_component_penalty_v1" in text
    assert "--pd_separate_z" in text


def test_all_new_arms_share_the_canonical_matched_protocol() -> None:
    text = _text()
    for required in (
        "NUM_GENERATIONS=9",
        "DEPLOYMENT_SET_SIZE=8",
        "ROLLOUT_BATCH_SIZE=36",
        "PER_DEVICE_TRAIN_BATCH_SIZE=3",
        "PER_DEVICE_EVAL_BATCH_SIZE=9",
        "GRADIENT_ACCUMULATION_STEPS=3",
        "PD_IMPORTANCE_SAMPLING_LEVEL=token",
        "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1",
        "--grpo_loss_type grpo",
        "--pd_structure_weight 0.0",
        "--no-pd_soft_objective_for_infeasible",
        "--pd_component_profile co_feasibility_lock_all",
        "--clasp_feasibility_warmup_steps 0",
        "--no-clasp_self_replay",
        "--no-clasp_adaptive_length",
        "--verify_gt_rewards",
        "ARM_CONTRACT_SCHEMA=clasp-canonical-matched-a100-v1",
        "verifier_gate=strict_recomputed_feasible_only_quality",
    ):
        assert required in text
    assert "ARM_CONTRACT_SCHEMA=clasp-canonical-foarl-matched-b9-20k1k-a100-v1" in text
    assert "MAX_PROMPT_LENGTH=20000" in text
    assert "MAX_COMPLETION_LENGTH=1000" in text
    assert "--grpo_loss_type dr_grpo" not in text
    assert "PD_IMPORTANCE_SAMPLING_LEVEL=sequence" not in text


def test_canonical_method_diffs_are_closed_and_receipted() -> None:
    text = _text()
    assert 'SOTA_CANONICAL_FIXED_LAMBDA:?SOTA_CANONICAL_FIXED_LAMBDA must be preregistered' in text
    assert 'BON_DUAL_LR=0.0' in text
    assert 'CANONICAL_RECOVERY_SCALE=0.0' in text
    assert 'CANONICAL_RECOVERY_SCALE=1.0' in text
    assert '--config "intentional_method_diff=' in text
    assert '--config "policy_credit_formula=${POLICY_CREDIT_FORMULA}"' in text
    assert '--config "component_controller=${COMPONENT_CONTROLLER}"' in text
    assert '--config "set_controller=${SET_CONTROLLER}"' in text
    assert '--remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}"' in text
    assert 'scripts/run_grpo_in_sealed_environment.py' in text
    assert '--receipt "${REMOTE_ENVIRONMENT_RECEIPT}" --' in text
    assert not any(line.lstrip().startswith("sbatch ") for line in text.splitlines())


def test_formula_level_arm_profiles_do_not_enable_hidden_controllers() -> None:
    text = _text()
    profile = text.split('case "${METHOD}" in', 2)[2].split("esac", 1)[0]
    dual = profile.split("dual_scalar_z)", 1)[1].split(";;", 1)[0]
    separate = profile.split("dual_separate_z)", 1)[1].split(";;", 1)[0]
    canonical = profile.split("clasp_canonical)", 1)[1].split(";;", 1)[0]
    fixed = profile.split("clasp_canonical_fixed_lambda)", 1)[1].split(";;", 1)[0]
    no_recovery = profile.split("clasp_canonical_no_recovery)", 1)[1].split(";;", 1)[0]
    set_null = profile.split("set_null)", 1)[1].split(";;", 1)[0]
    assert "COMPONENT_DUAL_LR=0.05" in dual
    assert "COMPONENT_DUAL_LR=0.05" in separate
    assert "COMPONENT_DUAL_LR=0.05" not in canonical
    assert "COMPONENT_DUAL_LR=0.05" not in fixed
    assert "COMPONENT_DUAL_LR=0.05" not in no_recovery
    assert "BON_DUAL_LR=0.05" in canonical
    assert "BON_DUAL_LR=0.05" not in fixed
    assert "BON_DUAL_LR=0.05" in no_recovery
    assert "CANONICAL_RECOVERY_SCALE=1.0" in canonical
    assert "CANONICAL_RECOVERY_SCALE=1.0" in fixed
    assert "CANONICAL_RECOVERY_SCALE=1.0" not in no_recovery
    assert "FIXED_NULL_SET=1" in set_null
    assert "CANONICAL_SET=1" not in set_null
    assert "BON_DUAL_LR=0.05" in set_null
    assert "CANONICAL_RECOVERY_SCALE=1.0" in set_null
    assert "POLICY_CREDIT_FORMULA=fixed_null_k8_same_set_utility_v1" in set_null
    assert "SET_CONTROLLER=signed_failure_at_k_ema" in set_null
    assert "POLICY_CREDIT_FORMULA=group_z_scalarized_positive_component_penalty_v1" in dual
    assert (
        "POLICY_CREDIT_FORMULA="
        "group_z_reward_minus_group_z_positive_component_penalty_v1"
        in separate
    )
    assert "COMPONENT_CONTROLLER=legacy_projected_ema_component_gap" in dual
    assert "COMPONENT_CONTROLLER=legacy_projected_ema_component_gap" in separate
    assert "PD_GROUP_SCALARIZATION=0" in separate
    assert "PD_SEPARATE_Z=1" in separate
    assert "SET_CONTROLLER=signed_failure_at_k_ema" in canonical
    assert "SET_CONTROLLER=frozen" in fixed
    assert "SET_CONTROLLER=signed_failure_at_k_ema" in no_recovery
    assert "CLASP_BON_SUCCESS_CREDIT" not in text
    assert "CLASP_PROPER_DUAL_GRADIENT_MODE" not in text
    assert '"canonical_target_failure_rate"' in text
    assert '"canonical_fixed_lambda_by_task"' in text


def test_grpo_z_strict_reward_gate_is_independent_of_legacy_clasp_gate() -> None:
    launcher = _text()
    reward_heads = REWARD_HEADS.read_text(encoding="utf-8")
    co_tasks = CO_TASKS.read_text(encoding="utf-8")
    assert "--no-pd_soft_objective_for_infeasible" in launcher
    assert 'export CLASP_FEASIBILITY_GATED="${USE_CLASP}"' in launcher
    assert "apply_constraint_gate=True" in reward_heads
    assert 'getattr(args, "pd_soft_objective_for_infeasible", True)' in reward_heads
    assert "if not allow_infeasible_partial_credit:" in co_tasks
    assert "return 0.0" in co_tasks


def test_sealed_wrapper_verifies_inside_the_slurm_step_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("run_grpo_in_sealed_environment", WRAPPER)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    receipt = tmp_path / "remote_environment.json"
    receipt.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[Path, bool]] = []
    executed: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        wrapper,
        "environment_identity",
        lambda path, *, require_accelerator: calls.append((path, require_accelerator)),
    )

    def capture_exec(executable: str, argv: list[str]) -> None:
        executed.append((executable, argv))
        raise RuntimeError("exec captured")

    monkeypatch.setattr(wrapper.os, "execv", capture_exec)
    with pytest.raises(RuntimeError, match="exec captured"):
        wrapper.main(
            [
                "--receipt",
                str(receipt),
                "--",
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=4",
            ]
        )
    assert calls == [(receipt, True)]
    assert executed[0][1][1:] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
    ]
