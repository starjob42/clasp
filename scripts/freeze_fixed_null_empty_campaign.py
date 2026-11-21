#!/usr/bin/env python3
"""Freeze the approved empty-completion SET-NULL inputs for all five tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from clasp.set_grpo.fixed_null_construction import (
    CONSTRUCTION_SPEC_PROTOCOL,
    FROZEN_INPUT_STATUS,
    NULL_CANDIDATE_PROTOCOL,
    behavior_environment_names,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
AUTHORITATIVE_RL_SHA256 = {
    "cvrp": "4a1c3e97ce4408e722348d893b88ee58a0c1ca65ff0c6db66739901b8a540ff6",
    "jssp": "35c2aeed27e885491a234787855012b9f890e2a970c86ee910a639fc72d0c27a",
    "mis": "65cc0ea9bad6a0d3a4a5b5d8a37c2b49cc56227bf696bfd79388e6f935308c39",
    "op": "bf5319112b68fba04ed41dd96b268d8e22d98620b9266aa9aab69fba2de0b1b4",
    "tsp": "90c5a48909ac9df191c727237474ec262db4a9c654f3555a19b744304dc0a288",
}
AUTHORITATIVE_RESERVED_REGISTRY_SHA256 = (
    "90c5fe2b58bffcd190249d341cedd565a10a251fc3d24c9ac92caae604cb1d32"
)

# Exact source-derived behavior variables set by the matched SET-NULL launcher.
# Every other active behavior variable is frozen as unset (the empty string).
MATCHED_NONEMPTY_BEHAVIOR_ENVIRONMENT = {
    "CLASP_DECLARED_OBJECTIVE_MATCH_WEIGHT": "0.0",
    "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "1",
    "CLASP_OBJECTIVE_GAP_SLOPE": "3.0",
    "CLASP_OBJECTIVE_QUALITY_MODE": "exp",
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "0",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "0",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/sota_20260814/fixed_null_empty_v1"),
    )
    result.add_argument(
        "--rl-root",
        type=Path,
        default=Path("third_party/llmcosolver_f6388991/data_rl"),
    )
    return result


def _self_hash(payload: dict[str, object], field: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return sha256_bytes(canonical_json_bytes(unhashed))


def _write_create_once(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while freezing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    rl_root = args.rl_root.expanduser().resolve()
    active_names = behavior_environment_names()
    unknown = set(MATCHED_NONEMPTY_BEHAVIOR_ENVIRONMENT) - set(active_names)
    if unknown:
        raise SystemExit(f"matched environment names are not active: {sorted(unknown)}")
    behavior_environment = {name: "" for name in active_names}
    behavior_environment.update(MATCHED_NONEMPTY_BEHAVIOR_ENVIRONMENT)

    bindings: dict[str, object] = {}
    for task in TASKS:
        supplied_training_rows = rl_root / task / "train" / "train_rl.json"
        if supplied_training_rows.is_symlink():
            raise SystemExit(f"RL training rows must not be a symlink: {supplied_training_rows}")
        training_rows = supplied_training_rows.resolve()
        if not training_rows.is_file():
            raise SystemExit(f"unsafe/missing RL training rows: {training_rows}")
        mode = training_rows.stat().st_mode & 0o777
        if mode & 0o222:
            raise SystemExit(f"RL training rows are not frozen read-only: {training_rows}")
        observed_sha256 = sha256_file(training_rows)
        if observed_sha256 != AUTHORITATIVE_RL_SHA256[task]:
            raise SystemExit(
                f"authoritative RL hash mismatch for {task}: {observed_sha256}"
            )

        candidate: dict[str, object] = {
            "schema_version": 1,
            "protocol": NULL_CANDIDATE_PROTOCOL,
            "status": FROZEN_INPUT_STATUS,
            "task": task,
            "completion": "",
            "scientific_parameters": {
                "construction_rule": "empty_string_completion_for_every_prompt_v1",
                "independence_evidence": (
                    "user_approved_2026-08-26_before_any_set_null_training_or_rollout"
                ),
                "null_semantics": "no_generated_action_parse_failure_baseline",
                "rollout_or_confirmation_outcomes_used": False,
            },
            "candidate_sha256": "",
        }
        candidate["candidate_sha256"] = _self_hash(candidate, "candidate_sha256")

        spec: dict[str, object] = {
            "schema_version": 1,
            "protocol": CONSTRUCTION_SPEC_PROTOCOL,
            "status": FROZEN_INPUT_STATUS,
            "task": task,
            "representation": {"prompt_style": "alpaca", "no_reason": True},
            "evaluator": {
                "pd_component_profile": "co_feasibility_lock_all",
                "pd_soft_objective_for_infeasible": False,
                "pd_constraint_target": 1.0,
                "pd_cost_limit": 0.0,
                "pd_violation_tolerance": 0.0,
            },
            "behavior_environment": behavior_environment,
            "spec_sha256": "",
        }
        spec["spec_sha256"] = _self_hash(spec, "spec_sha256")

        task_root = output_root / task
        candidate_path = task_root / "fixed_null_candidate.json"
        spec_path = task_root / "fixed_null_construction_spec.json"
        _write_create_once(candidate_path, candidate)
        _write_create_once(spec_path, spec)
        bindings[task] = {
            "training_rows": str(training_rows),
            "training_rows_sha256": observed_sha256,
            "candidate_relative_path": str(candidate_path.relative_to(output_root)),
            "candidate_file_sha256": sha256_file(candidate_path),
            "candidate_sha256": candidate["candidate_sha256"],
            "construction_spec_relative_path": str(spec_path.relative_to(output_root)),
            "construction_spec_file_sha256": sha256_file(spec_path),
            "spec_sha256": spec["spec_sha256"],
        }

    campaign: dict[str, object] = {
        "schema_version": 1,
        "protocol": "clasp-fixed-null-empty-campaign-freeze-v1",
        "status": FROZEN_INPUT_STATUS,
        "approved_at_date": "2026-08-26",
        "confirmation_or_rollout_outcomes_used": False,
        "authoritative_reserved_registry_sha256": AUTHORITATIVE_RESERVED_REGISTRY_SHA256,
        "tasks": list(TASKS),
        "task_bindings": bindings,
        "campaign_sha256": "",
    }
    campaign["campaign_sha256"] = _self_hash(campaign, "campaign_sha256")
    _write_create_once(output_root / "fixed_null_campaign_freeze.json", campaign)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(output_root),
                "campaign_sha256": campaign["campaign_sha256"],
                "tasks": list(TASKS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
