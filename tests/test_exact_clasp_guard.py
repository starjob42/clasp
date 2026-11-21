"""Regression tests for the exact CLASP set-credit configuration guard.

The v6 dev run (Slurm 3167) died at trainer init because the exact
`CLASP_CANONICAL_SET=1` path forbids auxiliary losses.  The launcher dry-run
only prints environment values, so it cannot catch this; these tests exercise
the trainer's guard directly, without loading a model or a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.trainers.primal_dual_grpo import PrimalDualGRPOTrainer


AUXILIARY_WEIGHT_ATTRIBUTES = (
    "clasp_pairwise_loss_weight",
    "clasp_elite_imitation_loss_weight",
    "clasp_multi_elite_imitation_loss_weight",
    "clasp_reference_imitation_loss_weight",
    "clasp_reference_contrastive_loss_weight",
    "clasp_tsp_two_opt_distill_loss_weight",
    "clasp_cvrp_local_search_distill_loss_weight",
    "clasp_jssp_active_schedule_distill_loss_weight",
    "clasp_mis_maximalize_distill_loss_weight",
    "clasp_mis_swap_distill_loss_weight",
    "clasp_self_replay_imitation_loss_weight",
    "clasp_self_replay_contrastive_loss_weight",
    "clasp_hard_group_weight_scale",
)


def _canonical_config(**overrides: object) -> SimpleNamespace:
    """A minimal valid exact-CLASP configuration, matching the B9/K8 arm."""

    config = SimpleNamespace(
        clasp_canonical_set=True,
        clasp_fixed_null_set=False,
        clasp_policy_length_alpha=0.0,
        num_generations=9,
        clasp_deployment_set_size=8,
        clasp_canonical_max_subsets=4096,
        clasp_canonical_recovery_scale=0.0,
        clasp_canonical_dual_recovery_coupling=0.0,
        clasp_canonical_recovery_temperature=1.0,
        clasp_canonical_exact_scale=1.0,
        clasp_canonical_grpo_anchor_scale=1.0,
        clasp_canonical_quality_anchor_ratio=1.0,
        clasp_canonical_dense_quality_scale=1.0,
        clasp_canonical_dense_feasibility_scale=0.0,
        clasp_canonical_dense_violation_scale=0.0,
        clasp_canonical_dense_recovery_scale=0.0,
        clasp_bon_aligned=True,
        clasp_frontier_cdf=False,
        loss_type="grpo",
        importance_sampling_level="sequence",
        max_completion_length=1000,
        clasp_adaptive_length=None,
        clasp_bon_target_failure_rates={"mis": 0.005},
        primal_dual_reward_manager=SimpleNamespace(wrappers={"mis": object()}),
        clasp_feasibility_warmup_steps=0,
    )
    for attribute in AUXILIARY_WEIGHT_ATTRIBUTES:
        setattr(config, attribute, 0.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _validate(config: SimpleNamespace) -> None:
    PrimalDualGRPOTrainer._validate_canonical_set_configuration(config)


def test_exact_canonical_config_accepts_zero_auxiliary_weights() -> None:
    _validate(_canonical_config())


def test_v7_reward_shaping_config_passes_exact_guard() -> None:
    """The v7 arm changes only sanctioned dense quality/feasibility credit."""

    _validate(
        _canonical_config(
            clasp_canonical_dense_quality_scale=1.5,
            clasp_canonical_dense_feasibility_scale=0.25,
            clasp_canonical_quality_anchor_ratio=1.0,
        )
    )


def test_v8_nonexact_bon_config_allows_maximalize_and_multi_elite() -> None:
    """Recipe B keeps exact set credit off, so its quality teachers are legal."""

    _validate(
        _canonical_config(
            clasp_canonical_set=False,
            clasp_policy_length_alpha=0.0,
            clasp_canonical_dual_recovery_coupling=0.0,
            clasp_canonical_grpo_anchor_scale=0.0,
            clasp_canonical_quality_anchor_ratio=0.0,
            clasp_canonical_dense_quality_scale=0.0,
            clasp_canonical_dense_feasibility_scale=0.0,
            clasp_canonical_dense_violation_scale=0.0,
            clasp_canonical_dense_recovery_scale=0.0,
            clasp_mis_maximalize_distill_loss_weight=0.1,
            clasp_multi_elite_imitation_loss_weight=0.05,
        )
    )


def test_v8_launcher_arm_selects_nonexact_bon_with_teachers() -> None:
    """Guard the v8 arm wiring against a silent exact/BON config regression."""

    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_clasp_canonical_checkpoint_dev_a100.slurm"
    )
    text = launcher.read_text(encoding="utf-8")
    block = text.split("clasp_v8)", 1)[1].split(";;", 1)[0]

    assert 'CANONICAL_SET="0"' in block
    assert 'USE_CLASP="1"' in block
    assert 'CANONICAL_GRPO_ANCHOR_SCALE="0.0"' in block
    assert 'CANONICAL_QUALITY_ANCHOR_RATIO="0.0"' in block
    assert 'DENSE_QUALITY_SCALE="0.0"' in block
    assert 'DENSE_FEASIBILITY_SCALE="0.0"' in block
    assert 'BON_DENSE_QUALITY_SCALE="1.5"' in block
    assert 'BON_RECOVERY_SCALE="0.5"' in block
    assert 'MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT="0.1"' in block
    # The pinned batch contract resolves to steps_per_generation=3, which the
    # group-complete-microbatch guard forbids for multi-elite (Slurm 134195).
    assert 'MULTI_ELITE_IMITATION_LOSS_WEIGHT="0.0"' in block
    assert 'POLICY_CREDIT_FORMULA="bon_dense_foarlinverse_maxdistill10_k8_v8"' in block
    # The arm must drive the CLI switch from CANONICAL_SET, otherwise the
    # strict config guard rejects the v8 CLI/env disagreement at trainer init.
    assert "CANONICAL_SET_ARGS=(--clasp_canonical_set)" in text
    assert "CANONICAL_SET_ARGS=(--no-clasp_canonical_set)" in text
    assert '"${CANONICAL_SET_ARGS[@]}"' in text
    assert '--clasp_canonical_set "${PD_ARGS[@]}"' not in text


def _canonical_flag_config(**overrides: object) -> SimpleNamespace:
    config = SimpleNamespace(
        clasp_canonical_set=False,
        clasp_fixed_null_set=False,
        clasp_raw_feasibility_targeting=False,
        use_clasp=True,
        use_primal_dual=False,
        clasp_strict_defaults=False,
        num_generations_train=None,
        clasp_feasibility_warmup_steps=0,
        clasp_feasibility_warmup_targets="",
        clasp_feasibility_warmup_hold_until_eval=True,
        clasp_self_replay=False,
        clasp_self_replay_imitation_weight=0.0,
        clasp_self_replay_contrastive_weight=0.0,
        clasp_validation_gate_checkpoints=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_strict_defaults_reject_cli_env_canonical_conflict(monkeypatch) -> None:
    """Slurm 117265: unconditional --clasp_canonical_set with v8 env 0."""

    from clasp.config import _apply_clasp_strict_defaults

    monkeypatch.setenv("CLASP_CANONICAL_SET", "0")
    monkeypatch.setattr(sys, "argv", ["train_grpo", "--clasp_canonical_set"])

    with pytest.raises(ValueError, match="Conflicting canonical CLASP switches"):
        _apply_clasp_strict_defaults(
            _canonical_flag_config(clasp_canonical_set=True),
            num_generations_attr="num_generations",
        )


def test_strict_defaults_accept_nonexact_cli_env_agreement(monkeypatch) -> None:
    from clasp.config import _apply_clasp_strict_defaults

    monkeypatch.setenv("CLASP_CANONICAL_SET", "0")
    monkeypatch.setattr(sys, "argv", ["train_grpo", "--no-clasp_canonical_set"])
    config = _canonical_flag_config(clasp_canonical_set=False)

    _apply_clasp_strict_defaults(config, num_generations_attr="num_generations")

    assert config.clasp_canonical_set is False


@pytest.mark.parametrize("attribute", AUXILIARY_WEIGHT_ATTRIBUTES)
def test_exact_canonical_rejects_each_auxiliary_weight(attribute: str) -> None:
    config = _canonical_config(**{attribute: 0.1})
    with pytest.raises(ValueError, match="cannot bundle auxiliary losses"):
        _validate(config)


def test_exact_canonical_rejects_v6_maximalize_and_multi_elite_pair() -> None:
    """Exact reproduction of the v6 Slurm 3167 failure."""

    config = _canonical_config(
        clasp_mis_maximalize_distill_loss_weight=0.1,
        clasp_multi_elite_imitation_loss_weight=0.05,
    )
    with pytest.raises(ValueError) as excinfo:
        _validate(config)
    message = str(excinfo.value)
    assert "CLASP_MIS_MAXIMALIZE_DISTILL_LOSS_WEIGHT" in message
    assert "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT" in message


def test_exact_canonical_rejects_feasibility_warmup() -> None:
    config = _canonical_config(clasp_feasibility_warmup_steps=1)
    with pytest.raises(ValueError, match="feasibility-warmup"):
        _validate(config)
