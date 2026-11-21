from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_clasp_canonical_checkpoint_dev_a100.slurm"


def _arm(name: str) -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    start = text.index(f"\n{name})") + 1
    end = text.index(";;", start)
    return text[start:end]


def test_v9_isolates_the_controller_fix() -> None:
    arm = _arm("clasp_v9")
    assert 'BON_DUAL_LR="0.5"' in arm
    assert 'BON_DUAL_EMA_BETA="0.5"' in arm
    assert 'BON_TARGET_FAILURE_RATE="0.02"' in arm
    assert 'BON_SUCCESS_CREDIT="dense_rank"' in arm
    assert (
        'POLICY_CREDIT_FORMULA="bon_dense_foarlinverse_maxdistill10_k8_v9_controller"'
        in arm
    )


def test_v10_adds_best8_credit_on_top_of_v9() -> None:
    arm = _arm("clasp_v10")
    assert 'BON_DUAL_LR="0.5"' in arm
    assert 'BON_DUAL_EMA_BETA="0.5"' in arm
    assert 'BON_TARGET_FAILURE_RATE="0.02"' in arm
    assert 'BON_SUCCESS_CREDIT="best8_from9"' in arm
    assert (
        'POLICY_CREDIT_FORMULA="bon_best8_from9_foarlinverse_maxdistill10_k8_v10"'
        in arm
    )


def test_v11_cvrp_adds_local_search_teacher_to_v9_controller() -> None:
    arm = _arm("clasp_v11_cvrp")
    assert 'BON_DUAL_LR="0.5"' in arm
    assert 'BON_DUAL_EMA_BETA="0.5"' in arm
    assert 'BON_SUCCESS_CREDIT="dense_rank"' in arm
    assert 'CVRP_LOCAL_SEARCH_DISTILL_LOSS_WEIGHT="0.1"' in arm
    assert 'CVRP_LOCAL_SEARCH_DISTILL_MAX_TOKENS="768"' in arm
    assert 'CVRP_LOCAL_SEARCH_DISTILL_MAX_PASSES="4"' in arm
    assert (
        'POLICY_CREDIT_FORMULA="bon_dense_cvrplocal_foarlinverse_maxdistill10_k8_v11_cvrp"'
        in arm
    )


def test_v12_mis_adds_swap_teacher_to_v9_controller() -> None:
    arm = _arm("clasp_v12_mis")
    assert 'BON_DUAL_LR="0.5"' in arm
    assert 'BON_DUAL_EMA_BETA="0.5"' in arm
    assert 'BON_SUCCESS_CREDIT="dense_rank"' in arm
    assert 'MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT="0.0"' in arm
    assert 'MIS_SWAP_DISTILL_LOSS_WEIGHT="0.1"' in arm
    assert 'MIS_SWAP_DISTILL_MAX_TOKENS="1024"' in arm
    assert 'MIS_SWAP_DISTILL_MAX_PASSES="2"' in arm
    assert (
        'POLICY_CREDIT_FORMULA="bon_dense_misswap_foarlinverse_maxdistill10_k8_v12_mis"'
        in arm
    )


def test_v14_op_margin_uses_training_only_safety_margin() -> None:
    arm = _arm("clasp_v14_op_margin")
    assert 'BON_DUAL_LR="0.5"' in arm
    assert 'BON_DUAL_EMA_BETA="0.5"' in arm
    assert 'BON_TARGET_FAILURE_RATE="0.01"' in arm
    assert 'BON_SUCCESS_CREDIT="dense_rank"' in arm
    assert 'PD_COMPONENT_PROFILE="co_op_strict_length_margin"' in arm
    assert 'OP_LENGTH_MARGIN_RATIO="0.985"' in arm
    assert (
        'POLICY_CREDIT_FORMULA="bon_dense_opmargin_foarlinverse_maxdistill10_k8_v14_op"'
        in arm
    )


def test_v9_v10_v11_v12_v14_are_distinct_arms() -> None:
    assert (
        len(
            {
                _arm("clasp_v9"),
                _arm("clasp_v10"),
                _arm("clasp_v11_cvrp"),
                _arm("clasp_v12_mis"),
                _arm("clasp_v14_op_margin"),
            }
        )
        == 5
    )


def test_default_component_profile_stays_eval_aligned() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'PD_COMPONENT_PROFILE="co_feasibility_lock_all"' in text
    assert 'OP_LENGTH_MARGIN_RATIO="0.0"' in text
    assert '--pd_component_profile "${PD_COMPONENT_PROFILE}"' in text
    assert 'export CLASP_OP_LENGTH_MARGIN_RATIO="${OP_LENGTH_MARGIN_RATIO}"' in text
    assert (
        'if [[ "${METHOD}" == "clasp_v14_op_margin" && "${TASK}" != "op" ]]' in text
    )


def test_launcher_keeps_one_final_horizon_across_chunks() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'FINAL_MAX_STEPS="${DEV_FINAL_MAX_STEPS:-${DEV_MAX_STEPS:-2}}"' in text
    assert 'STOP_STEP="${DEV_STOP_STEP:-${FINAL_MAX_STEPS}}"' in text
    assert 'export GRPO_STOP_AT_GLOBAL_STEP="${STOP_STEP}"' in text
    assert '--max_steps "${MAX_STEPS}"' in text
    assert 'checkpoint-${STOP_STEP}/adapter_model.safetensors' in text
    assert '"stop_step": int(stop_step)' in text
