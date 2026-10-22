from __future__ import annotations

import json
from pathlib import Path

import pytest

from clasp.set_grpo.fixed_null_construction import (
    CONSTRUCTION_SPEC_PROTOCOL,
    FROZEN_INPUT_STATUS,
    NULL_CANDIDATE_PROTOCOL,
    behavior_environment_names,
    build_fixed_null_receipt,
    canonical_json_bytes,
    sha256_bytes,
)
from clasp.set_grpo.fixed_null_replay import verify_fixed_null_receipt


def _write_frozen_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o440)
    return path


def _self_hash(payload: dict, field: str) -> dict:
    unhashed = dict(payload)
    unhashed.pop(field, None)
    payload[field] = sha256_bytes(canonical_json_bytes(unhashed))
    return payload


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    spec = _self_hash(
        {
            "schema_version": 1,
            "protocol": CONSTRUCTION_SPEC_PROTOCOL,
            "status": FROZEN_INPUT_STATUS,
            "task": "tsp",
            "representation": {"prompt_style": "alpaca", "no_reason": True},
            "evaluator": {
                "pd_component_profile": "co_feasibility_lock_all",
                "pd_soft_objective_for_infeasible": False,
                "pd_constraint_target": 1.0,
                "pd_cost_limit": 0.0,
                "pd_violation_tolerance": 0.0,
            },
            "behavior_environment": {name: "" for name in behavior_environment_names()},
        },
        "spec_sha256",
    )
    candidate = _self_hash(
        {
            "schema_version": 1,
            "protocol": NULL_CANDIDATE_PROTOCOL,
            "status": FROZEN_INPUT_STATUS,
            "task": "tsp",
            "completion": "",
            "scientific_parameters": {"test_fixture_rule": "explicit_empty_completion"},
        },
        "candidate_sha256",
    )
    rows = [
        {
            "instruction": "Return a closed traveling-salesperson route and objective.",
            "input": "Nodes: 0=(0,0), 1=(1,0), 2=(0,1)",
            "output": "Route: [0, 1, 2, 0], Objective: 3.41421356237",
            "instance_json": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        }
    ]
    return (
        _write_frozen_json(tmp_path / "spec.json", spec),
        _write_frozen_json(tmp_path / "candidate.json", candidate),
        _write_frozen_json(tmp_path / "train.json", rows),
    )


def test_build_and_independently_replay_every_fixed_null_prompt(tmp_path: Path) -> None:
    spec, candidate, rows = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    payload = build_fixed_null_receipt(
        construction_spec_path=spec,
        null_candidate_path=candidate,
        training_rows_path=rows,
        output_path=receipt,
    )

    assert receipt.stat().st_mode & 0o222 == 0
    assert len(payload["records"]) == 1
    assert payload["records"][0]["feasibility_flag"] == 0.0
    replay = verify_fixed_null_receipt(
        receipt_path=receipt,
        training_rows_path=rows,
        expected_task="tsp",
        expected_prompt_style="alpaca",
        expected_no_reason=True,
        expected_config=payload["construction_spec"]["evaluator"],
        expected_environment=payload["construction_spec"]["behavior_environment"],
    )
    assert replay["status"] == "PASS"
    assert replay["verified_record_count"] == 1


def test_independent_replay_rejects_rehashed_but_fabricated_scores(tmp_path: Path) -> None:
    spec, candidate, rows = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    build_fixed_null_receipt(
        construction_spec_path=spec,
        null_candidate_path=candidate,
        training_rows_path=rows,
        output_path=receipt,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["records"][0]["objective_score"] = 123.0
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    receipt.chmod(0o640)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt.chmod(0o440)

    with pytest.raises(ValueError, match="independent replay differs"):
        verify_fixed_null_receipt(receipt_path=receipt, training_rows_path=rows)


def test_replay_rejects_active_prompt_or_environment_drift(tmp_path: Path) -> None:
    spec, candidate, rows = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    payload = build_fixed_null_receipt(
        construction_spec_path=spec,
        null_candidate_path=candidate,
        training_rows_path=rows,
        output_path=receipt,
    )
    with pytest.raises(ValueError, match="prompt style differs"):
        verify_fixed_null_receipt(
            receipt_path=receipt,
            training_rows_path=rows,
            expected_prompt_style="jiang_f638",
        )
    drifted_environment = dict(payload["construction_spec"]["behavior_environment"])
    drifted_environment["CLASP_OBJECTIVE_GAP_SLOPE"] = "9.0"
    with pytest.raises(ValueError, match="behavior environment differs"):
        verify_fixed_null_receipt(
            receipt_path=receipt,
            training_rows_path=rows,
            expected_environment=drifted_environment,
        )


def test_builder_rejects_hold_candidate_and_mutable_rows(tmp_path: Path) -> None:
    spec, candidate, rows = _inputs(tmp_path)
    rows.chmod(0o640)
    with pytest.raises(ValueError, match="training rows must be frozen read-only"):
        build_fixed_null_receipt(
            construction_spec_path=spec,
            null_candidate_path=candidate,
            training_rows_path=rows,
            output_path=tmp_path / "mutable.json",
        )
    rows.chmod(0o440)

    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_payload["status"] = "HOLD_NEEDS_SCIENTIFIC_PARAMETERS"
    candidate_payload.pop("candidate_sha256")
    _self_hash(candidate_payload, "candidate_sha256")
    candidate.chmod(0o640)
    candidate.write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    candidate.chmod(0o440)
    with pytest.raises(ValueError, match="remains on HOLD"):
        build_fixed_null_receipt(
            construction_spec_path=spec,
            null_candidate_path=candidate,
            training_rows_path=rows,
            output_path=tmp_path / "hold.json",
        )
