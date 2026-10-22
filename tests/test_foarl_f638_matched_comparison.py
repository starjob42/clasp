from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
    FOARL_MATCHED_B9_CONFIRMATION_PAIR,
    FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST,
    SHARED_ENVIRONMENT_CONTRACT,
    SHARED_GENERATION_CONTRACT,
    SHARED_TRAINING_CONTRACT,
    foarl_matched_b9_shared_design_sha256,
    is_foarl_matched_b9_confirmation_pair,
    validate_clasp_foarl_matched_b9_method_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract(method: str, *, seed: int = 42, task: str = "tsp") -> dict:
    training = {**SHARED_TRAINING_CONTRACT, "data_seed": seed}
    environment = {
        **SHARED_ENVIRONMENT_CONTRACT,
        "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1" if task == "mis" else "0",
        "GRPO_DATA_SEED": str(seed),
    }
    if method == CLASP_FOARL_MATCHED_B9:
        training.update(
            {
                "entrypoint": "clasp.entrypoints.train_grpo",
                "arm_contract_schema": CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
                "comparison_protocol": (
                    "foarl-f638-clasp-shared-compute-data-token-design-v1"
                ),
                "comparison_baseline": "foarl_f638_matched_b9",
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
            }
        )
    else:
        training.update(
            {
                "entrypoint": "clasp.entrypoints.train_foarl_f638_campaign",
                "prompt_style": "jiang_f638",
                "no_reason": False,
                "loss_type": "grpo",
                "beta": 0.05,
                "use_primal_dual": False,
                "use_clasp": False,
            }
        )
    return {
        "method": method,
        "task": task,
        "seeds": {"train": seed, "data": seed, "sft": 42},
        "fixed_horizon": {"max_steps": 3_200, "world_size": 4},
        "data_lineage": {"same_registered_sources": "identity"},
        "fingerprints": {
            "initial_model": {"sha256": "a" * 64},
            "base_model": {"sha256": "b" * 64},
            "data": {"sha256": "c" * 64},
            "source": {"sha256": "d" * 64},
        },
        "generation": dict(SHARED_GENERATION_CONTRACT),
        "training_config": training,
        "tracked_environment": environment,
    }


def test_confirmation_pair_is_exact_ordered_and_native_is_never_aliased() -> None:
    assert FOARL_MATCHED_B9_CONFIRMATION_PAIR == (
        "foarl_f638_matched_b9",
        "clasp_canonical_foarl_matched_b9",
    )
    assert is_foarl_matched_b9_confirmation_pair(FOARL_MATCHED_B9_CONFIRMATION_PAIR)
    assert not is_foarl_matched_b9_confirmation_pair(
        tuple(reversed(FOARL_MATCHED_B9_CONFIRMATION_PAIR))
    )
    assert not is_foarl_matched_b9_confirmation_pair(
        ("foarl_f638_native", CLASP_FOARL_MATCHED_B9)
    )
    assert FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST


def test_clasp_comparator_and_foarl_lane_have_identical_shared_digest() -> None:
    foarl = _contract("foarl_f638_matched_b9")
    clasp = _contract(CLASP_FOARL_MATCHED_B9)
    validate_clasp_foarl_matched_b9_method_contract(
        task="tsp",
        train_seed=42,
        fixed_horizon=clasp["fixed_horizon"],
        generation=clasp["generation"],
        training_config=clasp["training_config"],
        tracked_environment=clasp["tracked_environment"],
    )
    assert foarl_matched_b9_shared_design_sha256(foarl) == (
        foarl_matched_b9_shared_design_sha256(clasp)
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("generation", "max_prompt_length", 29_768),
        ("generation", "max_completion_length", 3_000),
        ("training_config", "optim", "adamw_torch"),
        ("training_config", "final_max_steps", 3_201),
        ("tracked_environment", "CLASP_TSP_INPUT_ORDER", ""),
    ],
)
def test_shared_design_drift_fails_before_hashing(
    section: str, field: str, value: object
) -> None:
    contract = deepcopy(_contract(CLASP_FOARL_MATCHED_B9))
    contract[section][field] = value
    with pytest.raises(ValueError, match="matched-B9 shared design"):
        foarl_matched_b9_shared_design_sha256(contract)


def test_launcher_exposes_only_a_distinct_20k1k_comparator_profile() -> None:
    text = (ROOT / "scripts/train_clasp_sota_a100.slurm").read_text(encoding="utf-8")
    assert "clasp_canonical_foarl_matched_b9" in text
    assert "MAX_PROMPT_LENGTH=20000" in text
    assert "MAX_COMPLETION_LENGTH=1000" in text
    assert "matched-B9 CLASP comparator requires beta=0.05" in text
    assert "comparison_baseline=foarl_f638_matched_b9" in text
    assert "reproduction_claim=not_an_official_jiang_reproduction" in text
    assert "MAX_PROMPT_LENGTH=29768" in text
    assert "MAX_COMPLETION_LENGTH=3000" in text


def test_frozen_comparison_contract_self_hashes_and_binds_live_producers() -> None:
    path = (
        ROOT
        / "experiments/sota_20260814/foarl_f638_matched_b9_comparison_contract.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    stored = unsigned.pop("contract_sha256")
    observed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert stored == observed
    assert payload["ordered_confirmation_pair"] == list(
        FOARL_MATCHED_B9_CONFIRMATION_PAIR
    )
    assert payload["native_foarl_policy"]["confirmation_eligible"] is False
    assert payload["launch_state"]["jobs_submitted"] is False
    for relative, digest in payload["producer_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
