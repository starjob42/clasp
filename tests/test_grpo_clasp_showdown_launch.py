from pathlib import Path

from clasp.config import build_grpo_parser, finalize_grpo_args


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_showdown_methods_are_closed_and_matched() -> None:
    launcher = _read("scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")
    assert 'grpo_z)' in launcher
    assert 'clasp_sequence_a05)' in launcher
    assert 'POLICY_LENGTH_ALPHA="0.5"' in launcher
    assert 'IMPORTANCE_SAMPLING_LEVEL="sequence"' in launcher
    assert 'CANONICAL_RECOVERY_SCALE="1.0"' in launcher
    assert 'DUAL_RECOVERY_COUPLING="0.0"' in launcher
    assert '--pd_group_scalarization --no-pd_separate_z' in launcher
    assert '--no-pd_group_scalarization --no-pd_separate_z' in launcher
    assert '--no-use_clasp --no-clasp_canonical_set' in launcher
    assert '--clasp_feasibility_warmup_targets "${TASK}:0.995"' in launcher
    for shared in (
        '--learning_rate 1e-6',
        '--num_generations "${ROLLOUT_BATCH_SIZE}"',
        '--generation_batch_size 36',
        '--max_prompt_length 29768',
        '--max_completion_length 3000',
        '--generation_temperature 0.7',
        '--generation_top_p 0.9',
        '--generation_top_k 50',
        '--grpo_beta 0.0',
        '--lora_r 64 --lora_alpha 64',
    ):
        assert shared in launcher


def test_strong_clasp_arm_has_dense_b12_k8_contract() -> None:
    launcher = _read("scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")
    assert "clasp_strong_v1)" in launcher
    for contract_line in (
        'ROLLOUT_BATCH_SIZE="12"',
        'CANONICAL_MAX_SUBSETS="495"',
        'CANONICAL_EXACT_SCALE="0.25"',
        'DENSE_QUALITY_SCALE="0.5"',
        'DENSE_FEASIBILITY_SCALE="1.5"',
        'DENSE_VIOLATION_SCALE="0.5"',
        'DENSE_RECOVERY_SCALE="1.5"',
        'DUAL_RECOVERY_COUPLING="1.0"',
        'COMPONENT_DUAL_RECOVERY_SCALE="1.0"',
        'POLICY_LENGTH_ALPHA="0.0"',
        'IMPORTANCE_SAMPLING_LEVEL="sequence"',
    ):
        assert contract_line in launcher
    assert '--clasp_canonical_exact_scale "${CANONICAL_EXACT_SCALE}"' in launcher
    assert (
        '--clasp_canonical_dense_feasibility_scale "${DENSE_FEASIBILITY_SCALE}"'
        in launcher
    )
    assert '--clasp_component_dual_recovery_scale "${COMPONENT_DUAL_RECOVERY_SCALE}"' in launcher


def test_grpo_anchor_v2_preserves_classic_credit_with_small_exact_residual() -> None:
    launcher = _read("scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")
    assert "clasp_grpo_anchor_v2)" in launcher
    for contract_line in (
        'ESTIMATOR_ARM="grpo_anchor1_exact025_b9_k8_v2"',
        'ROLLOUT_BATCH_SIZE="9"',
        'CANONICAL_MAX_SUBSETS="9"',
        'CANONICAL_EXACT_SCALE="0.25"',
        'CANONICAL_GRPO_ANCHOR_SCALE="1.0"',
        'POLICY_LENGTH_ALPHA="0.0"',
        'IMPORTANCE_SAMPLING_LEVEL="token"',
        'POLICY_CREDIT_FORMULA="group_z_feasible_objective_plus_exact025_canonical_k8_residual_v2"',
    ):
        assert contract_line in launcher
    assert (
        '--clasp_canonical_grpo_anchor_scale "${CANONICAL_GRPO_ANCHOR_SCALE}"'
        in launcher
    )


def test_grpo_anchor_v3_norm_matches_only_tail_quality_credit() -> None:
    launcher = _read("scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")
    assert "clasp_grpo_anchor_v3)" in launcher
    for contract_line in (
        'ESTIMATOR_ARM="grpo_anchor1_normtail05_exact1_b9_k8_v3"',
        'ROLLOUT_BATCH_SIZE="9"',
        'CANONICAL_MAX_SUBSETS="9"',
        'CANONICAL_EXACT_SCALE="1.0"',
        'CANONICAL_GRPO_ANCHOR_SCALE="1.0"',
        'CANONICAL_QUALITY_ANCHOR_RATIO="0.5"',
        'POLICY_LENGTH_ALPHA="0.0"',
        'IMPORTANCE_SAMPLING_LEVEL="token"',
        'POLICY_CREDIT_FORMULA="group_z_feasible_objective_plus_normmatched05_canonical_k8_quality_raw_constraint_v3"',
    ):
        assert contract_line in launcher
    assert (
        '--clasp_canonical_quality_anchor_ratio "${CANONICAL_QUALITY_ANCHOR_RATIO}"'
        in launcher
    )


def test_strong_clasp_submitter_uses_two_hour_dev_launcher() -> None:
    submitter = _read("scripts/submit_clasp_strong_v1_a100.sh")
    assert "DEV_METHOD=clasp_strong_v1" in submitter
    assert "STRONG_CLASP_MAX_STEPS:-100" in submitter
    assert "checkpoint-20000" in submitter
    assert "train_clasp_canonical_checkpoint_dev_a100.slurm" in submitter
    assert "DEV_SAVE_TOTAL_LIMIT=1" in submitter


def test_grpo_anchor_submitter_is_matched_on_final_sft_for_cvrp_and_mis() -> None:
    submitter = _read("scripts/submit_clasp_grpo_anchor_v2_a100.sh")
    assert "checkpoint-31250" in submitter
    assert 'TASKS=("cvrp" "mis")' in submitter
    assert 'METHODS=("grpo_z" "clasp_grpo_anchor_v2")' in submitter
    assert "ANCHOR_MAX_STEPS:-100" in submitter
    assert "train_clasp_canonical_checkpoint_dev_a100.slurm" in submitter
    assert "DEV_SAVE_TOTAL_LIMIT=1" in submitter


def test_showdown_evaluation_is_fresh_paired_bon8() -> None:
    evaluator = _read("scripts/evaluate_grpo_clasp_showdown_a100.slurm")
    assert 'SHOWDOWN_EVAL_START_INDEX:-100' in evaluator
    assert 'SHOWDOWN_EVAL_SAMPLES:-100' in evaluator
    assert 'EVALUATION_PURPOSE=development' in evaluator
    assert 'EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp' in evaluator
    assert 'NUM_RETURN_SEQUENCES=8' in evaluator
    assert 'GENERATION_BATCH_SIZE=8' in evaluator
    assert 'SEED=271828' in evaluator
    assert 'MAX_PROMPT_LENGTH=29768' in evaluator
    assert 'sft_vs_grpo_z.json' in evaluator
    assert 'sft_vs_clasp_sequence_a05.json' in evaluator
    assert 'grpo_z_vs_clasp_sequence_a05.json' in evaluator


def test_showdown_submitter_uses_final_sft_and_dependency() -> None:
    submitter = _read("scripts/submit_grpo_clasp_showdown_a100.sh")
    assert 'checkpoint-31250' in submitter
    assert 'SHOWDOWN_MAX_STEPS:-100' in submitter
    assert 'submit_train grpo_z' in submitter
    assert 'submit_train clasp_sequence_a05' in submitter
    assert 'afterok:${GRPO_JOB_ID}:${CLASP_JOB_ID}' in submitter
    assert 'DEV_SAVE_STEPS=${MAX_STEPS}' in submitter
    assert 'DEV_SAVE_TOTAL_LIMIT=1' in submitter


def test_generic_evaluator_defaults_to_replication_but_allows_development() -> None:
    evaluator = _read("scripts/evaluate_a100.slurm")
    assert 'EVALUATION_PURPOSE="${EVALUATION_PURPOSE:-replication}"' in evaluator
    assert '--evaluation_purpose "${EVALUATION_PURPOSE}"' in evaluator


def test_showdown_recovery_reuses_completed_clasp_and_replaces_only_grpo() -> None:
    submitter = _read("scripts/submit_grpo_clasp_showdown_recovery_a100.sh")
    assert 'SHOWDOWN_EXISTING_CLASP_CHECKPOINT' in submitter
    assert 'DEV_METHOD=grpo_z' in submitter
    assert 'DEV_MAX_STEPS=${MAX_STEPS}' in submitter
    assert 'afterok:${GRPO_JOB_ID}' in submitter
    assert 'SHOWDOWN_CLASP_CHECKPOINT=${CLASP_CHECKPOINT}' in submitter
    assert 'DEV_METHOD=clasp_sequence_a05' not in submitter


def test_classic_grpo_cli_overrides_inactive_tsp_warmup_environment(monkeypatch) -> None:
    # This reproduces the pre-step-1 failure from job 1699613. The explicit
    # task-local CLI value must win even when the parser sees a stale TSP env.
    monkeypatch.setenv("CLASP_FEASIBILITY_WARMUP_TARGETS", "tsp:0.995")
    args = finalize_grpo_args(
        build_grpo_parser().parse_args(
            [
                "--tasks",
                "cvrp",
                "--model_path",
                "checkpoint",
                "--no-use_clasp",
                "--no-clasp_canonical_set",
                "--clasp_feasibility_warmup_steps",
                "0",
                "--clasp_feasibility_warmup_targets",
                "cvrp:0.995",
                "--no-clasp_raw_feasibility_targeting",
                "--no-clasp_self_replay",
                "--no-clasp_validation_gate_checkpoints",
                "--no-clasp_pareto_anchor",
                "--no-clasp_proper_dual",
                "--no-clasp_adaptive_length",
                "--use_primal_dual",
                "--pd_group_scalarization",
                "--no-pd_separate_z",
            ]
        )
    )
    assert args.tasks == ["cvrp"]
    assert args.clasp_feasibility_warmup_target_map == {"cvrp": 0.995}
    assert not args.use_clasp
    assert args.pd_group_scalarization


def test_pair_validation_compares_only_grpo_and_clasp() -> None:
    evaluator = _read("scripts/evaluate_grpo_clasp_pair_a100.slurm")
    assert 'GRPO_LABEL="${PAIR_EVAL_GRPO_LABEL:-grpo_z_step100}"' in evaluator
    assert 'CLASP_LABEL="${PAIR_EVAL_CLASP_LABEL:-clasp_sequence_a05_step100}"' in evaluator
    assert 'run_eval "${GRPO_LABEL}" "${GRPO_CHECKPOINT}"' in evaluator
    assert 'run_eval "${CLASP_LABEL}" "${CLASP_CHECKPOINT}"' in evaluator
    assert "run_eval sft" not in evaluator
    assert "EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp" in evaluator
    assert 'PAIR_EVAL_START_INDEX:-100' in evaluator
    assert 'PAIR_EVAL_SAMPLES:-100' in evaluator
    assert "NUM_RETURN_SEQUENCES=8" in evaluator
    assert "GENERATION_BATCH_SIZE=8" in evaluator
    assert "SEED=271828" in evaluator
    assert '${GRPO_LABEL}_vs_${CLASP_LABEL}.json' in evaluator
    assert "compare_development.py" in evaluator


def test_anchor_v2_continuation_is_strict_and_evaluation_is_dependent() -> None:
    launcher = _read("scripts/train_clasp_canonical_checkpoint_dev_a100.slurm")
    submitter = _read("scripts/submit_clasp_grpo_anchor_v2_continuation_a100.sh")
    assert 'export RESUME_FROM_CHECKPOINT="${RESUME_CHECKPOINT}"' in launcher
    for required in (
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "multitask_primal_dual_state.json",
        "rng_state_0.pth",
        "rng_state_3.pth",
    ):
        assert required in launcher
        assert required in submitter
    assert 'FINAL_STEP=$((resume_step + ADDITIONAL_STEPS))' in submitter
    assert 'DEPENDENCY="afterok:${GRPO_JOB}:${CLASP_JOB}"' in submitter
    assert 'PAIR_EVAL_GRPO_LABEL=grpo_z_step${FINAL_STEP}' in submitter
    assert 'PAIR_EVAL_CLASP_LABEL=clasp_grpo_anchor_v2_step${FINAL_STEP}' in submitter


def test_anchor_v3_branches_at_step100_and_runs_fresh_triplet_evaluation() -> None:
    submitter = _read("scripts/submit_clasp_grpo_anchor_v3_a100.sh")
    evaluator = _read("scripts/evaluate_grpo_clasp_v3_triplet_a100.slurm")
    assert "DEV_METHOD=clasp_grpo_anchor_v3" in submitter
    assert "clasp_grpo_anchor_v2_seed42_step100/checkpoints/checkpoint-100" in submitter
    assert 'FINAL_STEP=$((RESUME_STEP + ADDITIONAL_STEPS))' in submitter
    assert 'DEPENDENCY="afterok:${TRAIN_JOB}"' in submitter
    assert "#SBATCH --constraint=a100" in evaluator
    assert "#SBATCH --time=02:00:00" in evaluator
    assert "NUM_RETURN_SEQUENCES=8" in evaluator
    assert "EVAL_START_INDEX=\"${EVAL_START_INDEX}\"" in evaluator
    for label in (
        "grpo_z_step200",
        "clasp_grpo_anchor_v2_step200",
        "clasp_grpo_anchor_v3_step200",
    ):
        assert f"run_eval {label}" in evaluator
    assert "grpo_z_step200_vs_clasp_grpo_anchor_v3_step200.json" in evaluator
    assert (
        "clasp_grpo_anchor_v2_step200_vs_clasp_grpo_anchor_v3_step200.json"
        in evaluator
    )


def test_pair_validation_submitter_binds_completed_checkpoints() -> None:
    submitter = _read("scripts/submit_grpo_clasp_pair_eval_a100.sh")
    assert "cvrp_grpo_z_seed42_step100_recovery/checkpoints/checkpoint-100" in submitter
    assert "cvrp_clasp_sequence_a05_seed42_step100/checkpoints/checkpoint-100" in submitter
    assert "PAIR_EVAL_START_INDEX=${EVAL_START_INDEX}" in submitter
    assert "PAIR_EVAL_SAMPLES=${EVAL_SAMPLES}" in submitter
