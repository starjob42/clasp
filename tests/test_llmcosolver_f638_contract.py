from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/sota_20260814/llmcosolver_f638_semantic_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _load_contract() -> dict:
    return json.loads(CONTRACT.read_text())


def test_contract_is_self_hashed_and_non_executable():
    contract = _load_contract()
    expected = contract["contract_sha256"]
    contract["contract_sha256"] = None
    assert hashlib.sha256(_canonical(contract)).hexdigest() == expected
    assert contract["status"] == "HOLD_NON_EXECUTABLE"
    assert contract["classification_guards"]["no_training_or_evaluation_authorized"] is True
    assert contract["classification_guards"]["both_reference_variants_hold"] is True


def test_contract_seals_every_local_semantic_dependency():
    contract = _load_contract()
    projection = contract["repository_source_projection"]
    assert projection
    for relative, expected_hash in projection.items():
        assert _sha256(ROOT / relative) == expected_hash, relative
    assert "scripts/freeze_llmcosolver_f638_contract.py" in projection
    assert "third_party/llmcosolver_f6388991/rewards.py" in projection
    assert "src/clasp/baselines/llmcosolver_f638_rewards.py" in projection


def test_reward_loss_and_scope_are_not_conflated_with_current_clasp_grpo():
    contract = _load_contract()
    semantics = contract["reward_and_loss_semantics"]
    assert semantics["reward_heads"] == ["optimality", "feasibility"]
    assert semantics["head_count"] == 2
    assert semantics["beta"] == 0.05
    assert semantics["num_generations"] == 8
    assert "loss_type=grpo" in semantics["current_trl_compatibility"]
    assert "not dr_grpo" in semantics["current_trl_compatibility"]
    assert contract["scope"]["clasp_publication_intersection"] == [
        "tsp",
        "cvrp",
        "op",
        "mis",
        "jssp",
    ]
    assert (
        "three strict reward heads"
        in contract["classification_guards"]["current_clasp_method_grpo"]
    )
    assert "non-FOARL" in contract["classification_guards"]["current_clasp_method_grpo"]


def test_reference_variants_are_explicitly_distinct_and_neither_is_selected():
    variants = _load_contract()["adapter_and_reference_variants"]
    official = variants["A_official_code_operational"]
    corrected = variants["B_corrected_frozen_step_zero_sft"]
    assert official["executable"] is False
    assert official["adapter_bias"] == "lora_only"
    assert official["fixed_reference"] is False
    assert "live trainable" in official["qkv_bias_reference"]
    assert official["resume_equivalence"] is False
    assert corrected["executable"] is False
    assert corrected["fixed_reference"] is True
    assert "bias=all" not in official["adapter_bias"]
    assert corrected["adapter_bias"].startswith("all")
    assert corrected["resume_equivalence"] is True
    assert "not byte-identical" in corrected["classification"]


def test_missing_data_and_budget_provenance_blocks_matched_claim():
    evidence = _load_contract()["model_data_and_budget_evidence"]
    assert evidence["matched_baseline_status"] == "UNRESOLVED_DATA_AND_BUDGET_PROVENANCE"
    unresolved = " ".join(evidence["unresolved_publication_requirements"])
    for phrase in ("row hashes", "checkpoint-31250", "max_steps", "compute/token"):
        assert phrase in unresolved


def test_contract_freezer_is_create_once_and_pins_upstream_and_stack_hashes():
    contract = _load_contract()
    assert contract["upstream"]["commit"] == ("f6388991d243c0aa91d9ee7e2fa18ed1c9e72477")
    assert contract["upstream"]["archive_sha256"] == (
        "35a6ffdd5b165cdd5b35976a63596d41c818b6994bef09e3d09e9645c97caaf5"
    )
    assert contract["pinned_stack"]["trl"]["version"] == "0.15.2"
    assert contract["pinned_stack"]["unsloth"]["version"] == "2025.3.19"
    assert contract["pinned_stack"]["unsloth_zoo"]["version"] == "2025.3.17"
    assert contract["pinned_stack"]["peft"]["version"] == "0.17.0"
    freezer = (ROOT / "scripts/freeze_llmcosolver_f638_contract.py").read_text()
    assert 'args.output.open("x"' in freezer
