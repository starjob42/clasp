from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments/sota_20260814/llmcosolver_f638_semantic_contract.json"
V2 = ROOT / "experiments/sota_20260814/llmcosolver_f638_semantic_contract_v2.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_contract_v2_is_self_hashed_read_only_and_supersedes_v1():
    contract = _load(V2)
    expected = contract["contract_sha256"]
    contract["contract_sha256"] = None
    assert hashlib.sha256(_canonical(contract)).hexdigest() == expected
    assert stat.S_IMODE(V1.stat().st_mode) == 0o444
    assert stat.S_IMODE(V2.stat().st_mode) == 0o444
    assert contract["status"] == "HOLD_NON_EXECUTABLE"
    assert contract["supersedes"]["path"].endswith("llmcosolver_f638_semantic_contract.json")
    assert contract["supersedes"]["contract_sha256"] == _load(V1)["contract_sha256"]


def test_contract_v2_binds_data_manifest_self_and_file_hashes():
    contract = _load(V2)
    data = contract["public_data_evidence"]
    assert data["manifest_sha256"] == (
        "4b4896ae9d627f8306ca505bb245075d2af784dbe1b420c66cf5cf9fab5af06c"
    )
    assert data["file_sha256"] == (
        "6bc23d3cd0686385101b5d0fbbba7a8ee783986b57211958f256b644bc278f28"
    )
    assert data["rl_total_bytes"] == 266_602_952
    assert data["sft_drive_byte_identity_proven"] is False
    assert data["separate_data_license_observed"] is False


def test_contract_keeps_exact_two_head_loss_semantics_separate_from_reference_choice():
    contract = _load(V2)
    semantics = contract["reward_and_loss_semantics"]
    assert semantics["reward_heads"] == ["optimality", "feasibility"]
    assert semantics["head_count"] == 2
    assert semantics["beta"] == 0.05
    assert semantics["num_generations"] == 8
    assert "loss_type=grpo" in semantics["current_trl_compatibility"]
    assert "not dr_grpo" in semantics["current_trl_compatibility"]
    variants = contract["adapter_and_reference_variants"]
    assert variants["A_official_code_operational"]["executable"] is False
    assert variants["A_official_code_operational"]["fixed_reference"] is False
    assert variants["B_corrected_frozen_step_zero_sft"]["executable"] is False
    assert variants["B_corrected_frozen_step_zero_sft"]["fixed_reference"] is True
    assert "not byte-identical" in variants["B_corrected_frozen_step_zero_sft"]["classification"]


def test_contract_prohibits_claims_on_contaminated_author_eval():
    contract = _load(V2)
    assert contract["split_integrity"]["author_training_union_to_rl_eval"] == {
        "cvrp": 299,
        "jssp": 0,
        "mis": 300,
        "op": 299,
        "tsp": 0,
    }
    guards = contract["classification_guards"]
    assert guards["author_eval_held_out_claim_allowed"] is False
    assert guards["author_eval_leakage_free_claim_allowed"] is False
    assert guards["fresh_preregistered_disjoint_iid_required"] is True
    assert guards["no_training_or_evaluation_authorized"] is True
    assert "non-FOARL" in guards["current_clasp_method_grpo"]


def test_contract_records_known_horizons_without_inventing_compute_equivalence():
    evidence = _load(V2)["model_data_and_budget_evidence"]
    assert evidence["rl_rows"] == {
        "cvrp": 2000,
        "jssp": 3200,
        "mis": 3200,
        "op": 5000,
        "tsp": 3200,
    }
    assert evidence["conditional_single_process_rl_optimizer_steps"] == {
        "cvrp": 250,
        "jssp": 400,
        "mis": 400,
        "op": 625,
        "tsp": 400,
    }
    assert evidence["conditional_single_process_sft_optimizer_steps_per_task"] == 31_250
    unresolved = " ".join(evidence["unresolved_publication_requirements"])
    for phrase in ("checkpoint", "base-model revision", "token", "world size"):
        assert phrase in unresolved


def test_contract_seals_v2_implementation_tests_and_addendum():
    projection = _load(V2)["repository_source_projection"]
    required = {
        "src/clasp/baselines/llmcosolver_f638_public_data.py",
        "scripts/freeze_llmcosolver_f638_public_data.py",
        "scripts/freeze_llmcosolver_f638_public_data_v2.py",
        "scripts/freeze_llmcosolver_f638_contract_v2.py",
        "tests/test_llmcosolver_f638_public_data.py",
        "tests/test_llmcosolver_f638_public_data_manifest_v2.py",
        "tests/test_llmcosolver_f638_contract_v2.py",
        "docs/llmcosolver_f638_public_data_addendum_v2.md",
    }
    assert required <= set(projection)
    for relative, expected in projection.items():
        assert _sha256(ROOT / relative) == expected, relative
