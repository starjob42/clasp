from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_sota_checkpoint", ROOT / "scripts" / "select_sota_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _registry_and_development() -> tuple[dict, dict]:
    evaluation_files = [
        {
            "task": "tsp",
            "split": "eval",
            "path": "tsp/eval/test.json",
            "size": 3,
            "sha256": _digest("historical-eval-bytes"),
        }
    ]
    sources = []
    for label in ("historical-eval", "rl-train", "sft-train"):
        relative_path = (
            evaluation_files[0]["path"] if label == "historical-eval" else f"tsp/{label}/data.json"
        )
        file_sha256 = (
            evaluation_files[0]["sha256"]
            if label == "historical-eval"
            else _digest(f"{label}-bytes")
        )
        files = [
            {
                "relative_path": relative_path,
                "sha256": file_sha256,
                "size_bytes": 3,
                "rows": 1000,
            }
        ]
        source = {
            "task": "tsp",
            "label": label,
            "files": files,
            "file_manifest_sha256": hashlib.sha256(
                selector._canonical_json_bytes(files)
            ).hexdigest(),
            "rows": 1000,
            "unique_instance_hashes": 1000,
        }
        source["source_identity_sha256"] = hashlib.sha256(
            selector._canonical_json_bytes(source)
        ).hexdigest()
        sources.append(source)
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": hashlib.sha256(
            selector._canonical_json_bytes(sources)
        ).hexdigest(),
    }
    registry["registry_sha256"] = hashlib.sha256(
        selector._canonical_json_bytes(registry)
    ).hexdigest()
    historical = next(source for source in sources if source["label"] == "historical-eval")
    development = {
        "data_manifest_sha256": hashlib.sha256(
            selector._canonical_json_bytes(evaluation_files)
        ).hexdigest(),
        "evaluation_data_files": evaluation_files,
        "forbidden_source_label": "historical-eval",
        "forbidden_source_identity_sha256": historical["source_identity_sha256"],
        "protocol_fingerprint": _digest("protocol"),
        "strict_verifier_manifest_sha256": _digest("verifier"),
        "split": "eval",
        "eval_start_index": 0,
        "max_eval_samples_per_task": 1000,
        "eval_samples_per_task": {},
        "size_bucket": "all",
        "min_instance_size": 0,
        "max_instance_size": 0,
        "prefer_subset_files": False,
        "expected_instances": 1000,
    }
    development["development_evaluation_sha256"] = hashlib.sha256(
        selector._canonical_json_bytes(development)
    ).hexdigest()
    return registry, development


def _development_contract() -> dict:
    return _registry_and_development()[1]


def _write_policy(tmp_path: Path) -> Path:
    registry, development = _registry_and_development()
    payload = {
        "schema_version": 1,
        "campaign_id": "campaign",
        "stage_id": "3200_step_selection",
        "status": "frozen_before_development_results",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "evidence_scope": "development_only",
        "expected_tasks": ["tsp"],
        "expected_methods": ["clasp_dense"],
        "train_seeds": [42],
        "checkpoint_steps": [100, 200, 300, 400],
        "candidate_budget": 8,
        "forbidden_source_registry": registry,
        "development_evaluation_by_task": {"tsp": development},
        "require_complete_checkpoint_grid": True,
        "pass_at_k_absolute_band": 0.01,
        "selection_rule": selector.SELECTION_RULE,
    }
    payload["policy_sha256"] = hashlib.sha256(selector._canonical_json_bytes(payload)).hexdigest()
    path = tmp_path / "selection_policy.json"
    path.write_text(json.dumps(payload))
    return path


def _record(step: int, feasible: int, gap: float) -> dict:
    return {
        "task": "tsp",
        "method": "clasp_dense",
        "train_seed": 42,
        "checkpoint_step": step,
        "candidate_budget": 8,
        "instances": 1000,
        "feasible_instances": feasible,
        "pass_at_k": feasible / 1000,
        "conditional_gap_n": feasible,
        "conditional_gap_mean": gap,
        "evaluation_fingerprint": _digest(f"eval-{step}"),
        "protocol_fingerprint": _digest("protocol"),
        "development_evaluation_sha256": _development_contract()["development_evaluation_sha256"],
        "checkpoint_artifact_sha256": _digest(f"checkpoint-{step}"),
        "checkpoint_fidelity_identity_sha256": _digest(f"fidelity-{step}"),
        "checkpoint_fidelity_receipt_sha256": _digest(f"receipt-{step}"),
        "checkpoint_fidelity_receipt_file_sha256": _digest(f"receipt-file-{step}"),
        "grpo_beta": 0.0,
        "fixed_kl_reference": None,
        "exact_peft_kl_runtime": None,
        "exact_peft_kl_post_step": None,
        "run_contract_sha256": _digest("contract"),
        "data_manifest_sha256": _development_contract()["data_manifest_sha256"],
        "strict_verifier_manifest_sha256": _digest("verifier"),
    }


def _write_evidence(
    tmp_path: Path, records: list[dict], *, scope: str = "development_only"
) -> Path:
    payload = {
        "schema_version": 2,
        "campaign_id": "campaign",
        "status": "complete",
        "evidence_scope": scope,
        "candidate_budget": 8,
        "records": records,
    }
    payload["evidence_sha256"] = hashlib.sha256(selector._canonical_json_bytes(payload)).hexdigest()
    path = tmp_path / "checkpoint_evidence.json"
    path.write_text(json.dumps(payload))
    return path


def test_lexicographic_selection_uses_pass_band_then_gap(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    evidence = _write_evidence(
        tmp_path,
        [
            _record(100, 820, 0.12),
            _record(200, 825, 0.11),
            _record(300, 816, 0.09),
            _record(400, 814, 0.01),
        ],
    )
    receipt = selector.select_checkpoints(policy_path=policy, evidence_path=evidence)
    assert receipt["decision"] == "SELECTED"
    row = receipt["selections"]["tsp/clasp_dense/seed42"]
    assert row["selected"]["checkpoint_step"] == 300
    assert row["selection_audit"]["eligible_checkpoint_steps"] == [100, 200, 300]
    assert receipt["confirmation_evidence_used"] is False


def test_exact_gap_tie_selects_earlier_checkpoint(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    evidence = _write_evidence(
        tmp_path,
        [
            _record(100, 820, 0.12),
            _record(200, 825, 0.09),
            _record(300, 816, 0.09),
            _record(400, 814, 0.01),
        ],
    )
    receipt = selector.select_checkpoints(policy_path=policy, evidence_path=evidence)
    selected = receipt["selections"]["tsp/clasp_dense/seed42"]["selected"]
    assert selected["checkpoint_step"] == 200


def test_missing_scheduled_checkpoint_fails_closed(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    evidence = _write_evidence(
        tmp_path,
        [_record(100, 820, 0.12), _record(200, 825, 0.11), _record(300, 816, 0.09)],
    )
    with pytest.raises(ValueError, match="missing 1 scheduled rows"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)


def test_missing_post_step_chronology_field_fails_closed_even_for_beta_zero(
    tmp_path: Path,
) -> None:
    policy = _write_policy(tmp_path)
    records = [_record(step, 820, 0.12) for step in (100, 200, 300, 400)]
    records[0].pop("exact_peft_kl_post_step")
    evidence = _write_evidence(tmp_path, records)
    with pytest.raises(ValueError, match="exact_peft_kl_post_step is missing"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)


def test_confirmation_evidence_is_forbidden(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    evidence = _write_evidence(
        tmp_path,
        [
            _record(100, 820, 0.12),
            _record(200, 825, 0.11),
            _record(300, 816, 0.09),
            _record(400, 814, 0.01),
        ],
        scope="independent_confirmation",
    )
    with pytest.raises(ValueError, match="confirmation/test evidence is forbidden"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)


def test_mixed_protocol_or_corrupt_denominator_is_rejected(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    records = [
        _record(100, 820, 0.12),
        _record(200, 825, 0.11),
        _record(300, 816, 0.09),
        _record(400, 814, 0.01),
    ]
    records[1]["protocol_fingerprint"] = _digest("other")
    evidence = _write_evidence(tmp_path, records)
    with pytest.raises(ValueError, match="mixed protocol_fingerprint"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)

    records[1]["protocol_fingerprint"] = _digest("protocol")
    records[0]["pass_at_k"] = 0.999
    evidence = _write_evidence(tmp_path, records)
    with pytest.raises(ValueError, match="exact denominator"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)


def test_cross_method_development_protocol_must_match(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    policy_payload = json.loads(policy.read_text())
    policy_payload["expected_methods"] = ["clasp_dense", "grpo"]
    policy_payload.pop("policy_sha256")
    policy_payload["policy_sha256"] = hashlib.sha256(
        selector._canonical_json_bytes(policy_payload)
    ).hexdigest()
    policy.write_text(json.dumps(policy_payload))

    records = [_record(step, 820, 0.1) for step in (100, 200, 300, 400)]
    for source in list(records):
        challenger = dict(source)
        challenger["method"] = "grpo"
        challenger["evaluation_fingerprint"] = _digest(f"grpo-eval-{source['checkpoint_step']}")
        challenger["checkpoint_artifact_sha256"] = _digest(
            f"grpo-checkpoint-{source['checkpoint_step']}"
        )
        challenger["run_contract_sha256"] = _digest("grpo-contract")
        challenger["data_manifest_sha256"] = _digest("different-data")
        records.append(challenger)
    evidence = _write_evidence(tmp_path, records)

    with pytest.raises(ValueError, match="mixed data_manifest_sha256 across methods"):
        selector.select_checkpoints(policy_path=policy, evidence_path=evidence)


def test_development_data_must_equal_registered_historical_eval(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    payload = json.loads(policy.read_text())
    development = payload["development_evaluation_by_task"]["tsp"]
    development["evaluation_data_files"][0]["sha256"] = _digest("hidden-confirmation-bytes")
    development["data_manifest_sha256"] = hashlib.sha256(
        selector._canonical_json_bytes(development["evaluation_data_files"])
    ).hexdigest()
    development.pop("development_evaluation_sha256")
    development["development_evaluation_sha256"] = hashlib.sha256(
        selector._canonical_json_bytes(development)
    ).hexdigest()
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(selector._canonical_json_bytes(payload)).hexdigest()
    policy.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="registered historical-eval"):
        selector._validate_policy(policy)


def test_cli_emits_atomic_hold_receipt(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    records = [_record(step, 0, 0.0) for step in (100, 200, 300, 400)]
    for record in records:
        record["conditional_gap_n"] = 0
        record["conditional_gap_mean"] = None
    evidence = _write_evidence(tmp_path, records)
    output = tmp_path / "receipts" / "selection.json"
    assert (
        selector.main(
            [
                "--policy",
                str(policy),
                "--evidence",
                str(evidence),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    receipt = json.loads(output.read_text())
    assert receipt["decision"] == "HOLD"
    assert selector._is_sha256(receipt["receipt_sha256"])
    assert output.stat().st_mode & 0o222 == 0
    assert not list(output.parent.glob(".selection.json.*.tmp"))
    original = output.read_bytes()
    assert (
        selector.main(
            [
                "--policy",
                str(policy),
                "--evidence",
                str(evidence),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == original
