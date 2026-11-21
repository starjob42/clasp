#!/usr/bin/env python3
"""Select development checkpoints by the frozen CLASP lexicographic rule.

For every declared ``(task, method, training seed)`` group, the selector:

1. finds the maximum development pass@K;
2. retains checkpoints within 0.01 absolute pass@K of that maximum;
3. minimizes conditional verified gap within that band; and
4. breaks exact gap ties in favor of the earlier checkpoint.

The input must be complete, fingerprinted *development-only* evidence for the
entire checkpoint grid declared before results were viewed.  The only write is
an atomic selection receipt.  This script never changes model checkpoints and
refuses confirmation/test evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from clasp.data.benchmark import validate_forbidden_source_registry
from clasp.models.publication_fidelity import (
    validate_exact_peft_kl_post_step_lineage,
    validate_exact_peft_kl_runtime_identity,
    validate_fixed_kl_reference_identity,
)

SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SELECTION_RULE = [
    "maximize_pass_at_k",
    "retain_within_0.01_absolute_of_best_pass_at_k",
    "minimize_conditional_verified_gap",
    "tie_break_earlier_checkpoint_step",
]
FINGERPRINT_FIELDS = (
    "evaluation_fingerprint",
    "protocol_fingerprint",
    "development_evaluation_sha256",
    "checkpoint_artifact_sha256",
    "checkpoint_fidelity_identity_sha256",
    "checkpoint_fidelity_receipt_sha256",
    "checkpoint_fidelity_receipt_file_sha256",
    "run_contract_sha256",
    "data_manifest_sha256",
    "strict_verifier_manifest_sha256",
)
SHARED_GROUP_FIELDS = (
    "protocol_fingerprint",
    "run_contract_sha256",
    "data_manifest_sha256",
    "strict_verifier_manifest_sha256",
    "grpo_beta",
)
SHARED_TASK_PROTOCOL_FIELDS = (
    "protocol_fingerprint",
    "development_evaluation_sha256",
    "data_manifest_sha256",
    "strict_verifier_manifest_sha256",
    "candidate_budget",
    "instances",
)

DEVELOPMENT_EVALUATION_FIELDS = {
    "data_manifest_sha256",
    "evaluation_data_files",
    "forbidden_source_label",
    "forbidden_source_identity_sha256",
    "protocol_fingerprint",
    "strict_verifier_manifest_sha256",
    "split",
    "eval_start_index",
    "max_eval_samples_per_task",
    "eval_samples_per_task",
    "size_bucket",
    "min_instance_size",
    "max_instance_size",
    "prefer_subset_files",
    "expected_instances",
    "development_evaluation_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_self_hash(payload: dict[str, Any], *, field: str, label: str) -> str:
    stored = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored != observed:
        raise ValueError(f"{label} self-hash mismatch: stored={stored!r}, observed={observed}")
    return observed


def _unique_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        raise ValueError(f"selection policy {key} must be a nonempty unique list")
    return value


def _validate_development_evaluation_policy(
    policy: dict[str, Any], *, tasks: list[Any]
) -> dict[str, dict[str, Any]]:
    registry = policy.get("forbidden_source_registry")
    if not isinstance(registry, dict):
        raise ValueError("selection policy has no frozen forbidden-source registry")
    registry_identity = validate_forbidden_source_registry(
        registry, expected_tasks=[str(task) for task in tasks]
    )
    sources_by_task = registry_identity.get("sources_by_task_label")
    if not isinstance(sources_by_task, dict):
        raise ValueError("selection policy registry has no source projections")
    contracts = policy.get("development_evaluation_by_task")
    if not isinstance(contracts, dict) or set(contracts) != {str(task) for task in tasks}:
        raise ValueError("selection policy development_evaluation_by_task must exactly cover tasks")
    validated: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_name = str(task)
        contract = contracts.get(task_name)
        if not isinstance(contract, dict) or set(contract) != DEVELOPMENT_EVALUATION_FIELDS:
            raise ValueError(
                f"selection policy has an invalid development contract for {task_name}"
            )
        unsigned = dict(contract)
        stored_hash = unsigned.pop("development_evaluation_sha256", None)
        observed_hash = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        if stored_hash != observed_hash:
            raise ValueError(
                f"selection policy development contract self-hash mismatch for {task_name}"
            )
        if (
            not _is_sha256(contract.get("data_manifest_sha256"))
            or not _is_sha256(contract.get("protocol_fingerprint"))
            or not _is_sha256(contract.get("strict_verifier_manifest_sha256"))
            or contract.get("forbidden_source_label") != "historical-eval"
            or not _is_sha256(contract.get("forbidden_source_identity_sha256"))
            or contract.get("split") != "eval"
            or contract.get("eval_start_index") != 0
            or isinstance(contract.get("max_eval_samples_per_task"), bool)
            or not isinstance(contract.get("max_eval_samples_per_task"), int)
            or int(contract["max_eval_samples_per_task"]) <= 0
            or contract.get("eval_samples_per_task") != {}
            or contract.get("size_bucket") != "all"
            or contract.get("min_instance_size") != 0
            or contract.get("max_instance_size") != 0
            or contract.get("prefer_subset_files") is not False
            or contract.get("expected_instances") != contract.get("max_eval_samples_per_task")
        ):
            raise ValueError(
                f"selection policy development contract is not the full frozen slice for {task_name}"
            )
        evaluation_files = contract.get("evaluation_data_files")
        if not isinstance(evaluation_files, list) or not evaluation_files:
            raise ValueError(
                f"selection policy development contract has no data files for {task_name}"
            )
        expected_file_keys = {"task", "split", "path", "size", "sha256"}
        if any(
            not isinstance(item, dict)
            or set(item) != expected_file_keys
            or item.get("task") != task_name
            or item.get("split") != contract["split"]
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int)
            or int(item["size"]) <= 0
            or not _is_sha256(item.get("sha256"))
            for item in evaluation_files
        ):
            raise ValueError(f"selection policy development data files are invalid for {task_name}")
        paths = [str(item["path"]) for item in evaluation_files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError(
                f"selection policy development data files are not sorted/unique for {task_name}"
            )
        if (
            contract["data_manifest_sha256"]
            != hashlib.sha256(_canonical_json_bytes(evaluation_files)).hexdigest()
        ):
            raise ValueError(
                f"selection policy development data manifest is invalid for {task_name}"
            )
        task_sources = sources_by_task.get(task_name)
        historical = task_sources.get("historical-eval") if isinstance(task_sources, dict) else None
        if not isinstance(historical, dict) or contract[
            "forbidden_source_identity_sha256"
        ] != historical.get("source_identity_sha256"):
            raise ValueError(
                f"selection policy development data is not registered historical-eval for {task_name}"
            )
        registered_files = historical.get("files")
        registered_projection = (
            [
                {
                    "task": task_name,
                    "split": contract["split"],
                    "path": item.get("relative_path"),
                    "size": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in registered_files
                if isinstance(item, dict)
            ]
            if isinstance(registered_files, list)
            else []
        )
        if evaluation_files != registered_projection:
            raise ValueError(
                f"selection policy development bytes differ from registered historical-eval for {task_name}"
            )
        validated[task_name] = contract
    return validated


def _validate_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-selection policy schema")
    _validate_self_hash(policy, field="policy_sha256", label="selection policy")
    if policy.get("status") != "frozen_before_development_results":
        raise ValueError("selection policy is not frozen_before_development_results")
    if policy.get("evidence_scope") != "development_only":
        raise ValueError("checkpoint selection is permitted only for development evidence")
    if policy.get("selection_rule") != SELECTION_RULE:
        raise ValueError("selection policy changed the fixed lexicographic rule")
    tolerance = policy.get("pass_at_k_absolute_band")
    if not _is_finite_number(tolerance) or float(tolerance) != 0.01:
        raise ValueError("selection pass@K band must be exactly 0.01 absolute")
    for key in ("campaign_id", "stage_id", "frozen_at_utc"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise ValueError(f"selection policy has no valid {key}")
    tasks = _unique_list(policy, "expected_tasks")
    _unique_list(policy, "expected_methods")
    seeds = _unique_list(policy, "train_seeds")
    steps = _unique_list(policy, "checkpoint_steps")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds):
        raise ValueError("selection policy train_seeds must be nonnegative integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in steps):
        raise ValueError("selection policy checkpoint_steps must be positive integers")
    if steps != sorted(steps):
        raise ValueError("selection policy checkpoint_steps must be sorted")
    if policy.get("require_complete_checkpoint_grid") is not True:
        raise ValueError("selection policy must require the complete checkpoint grid")
    _validate_development_evaluation_policy(policy, tasks=tasks)
    candidate_budget = policy.get("candidate_budget")
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("selection policy candidate_budget must be positive")
    return policy


def _expected_keys(policy: dict[str, Any]) -> list[tuple[str, str, int, int]]:
    return [
        (str(task), str(method), int(seed), int(step))
        for task in policy["expected_tasks"]
        for method in policy["expected_methods"]
        for seed in policy["train_seeds"]
        for step in policy["checkpoint_steps"]
    ]


def _validate_record(
    record: object,
    *,
    candidate_budget: int,
    row_index: int,
) -> tuple[tuple[str, str, int, int] | None, list[str]]:
    label = f"records[{row_index}]"
    if not isinstance(record, dict):
        return None, [f"{label} is not an object"]
    errors: list[str] = []
    try:
        task = str(record["task"])
        method = str(record["method"])
        seed = int(record["train_seed"])
        step = int(record["checkpoint_step"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"{label} has an invalid identity: {exc}"]
    if isinstance(record.get("train_seed"), bool) or isinstance(
        record.get("checkpoint_step"), bool
    ):
        errors.append(f"{label} seed/step cannot be boolean")
    if seed < 0 or step <= 0 or not task or not method:
        errors.append(f"{label} has an out-of-range identity")
    if record.get("candidate_budget") != candidate_budget:
        errors.append(f"{label}.candidate_budget disagrees with policy")
    instances = record.get("instances")
    feasible = record.get("feasible_instances")
    if (
        isinstance(instances, bool)
        or not isinstance(instances, int)
        or instances <= 0
        or isinstance(feasible, bool)
        or not isinstance(feasible, int)
        or feasible < 0
        or feasible > instances
    ):
        errors.append(f"{label} has invalid instance denominators")
    pass_at_k = record.get("pass_at_k")
    if not _is_finite_number(pass_at_k) or not 0.0 <= float(pass_at_k) <= 1.0:
        errors.append(f"{label}.pass_at_k is invalid")
    elif isinstance(instances, int) and isinstance(feasible, int) and instances > 0:
        expected_pass = feasible / instances
        if abs(float(pass_at_k) - expected_pass) > 1e-12:
            errors.append(f"{label}.pass_at_k disagrees with its exact denominator")
    gap_n = record.get("conditional_gap_n")
    gap = record.get("conditional_gap_mean")
    if not isinstance(feasible, bool) and isinstance(feasible, int):
        if gap_n != feasible:
            errors.append(f"{label}.conditional_gap_n must equal feasible_instances")
        if feasible == 0:
            if gap is not None:
                errors.append(f"{label}.conditional_gap_mean must be null with no feasible rows")
        elif not _is_finite_number(gap) or float(gap) < 0.0:
            errors.append(f"{label}.conditional_gap_mean is invalid")
    for field in FINGERPRINT_FIELDS:
        if not _is_sha256(record.get(field)):
            errors.append(f"{label}.{field} is not a SHA-256 digest")
    beta = record.get("grpo_beta")
    fixed_kl = record.get("fixed_kl_reference")
    exact_runtime = record.get("exact_peft_kl_runtime")
    exact_post_step = record.get("exact_peft_kl_post_step")
    if "exact_peft_kl_runtime" not in record:
        errors.append(f"{label}.exact_peft_kl_runtime is missing")
    if "exact_peft_kl_post_step" not in record:
        errors.append(f"{label}.exact_peft_kl_post_step is missing")
    if not _is_finite_number(beta) or float(beta) < 0.0:
        errors.append(f"{label}.grpo_beta is invalid")
    elif float(beta) == 0.0:
        if fixed_kl is not None:
            errors.append(f"{label} beta=0 must not bind a fixed-KL reference")
        if exact_runtime is not None:
            errors.append(f"{label} beta=0 must not bind an exact PEFT KL runtime")
        if exact_post_step is not None:
            errors.append(f"{label} beta=0 must not bind exact PEFT KL post-step evidence")
    else:
        try:
            normalized_fixed = validate_fixed_kl_reference_identity(fixed_kl)
        except ValueError as exc:
            errors.append(f"{label} beta>0 fixed-KL identity is invalid: {exc}")
        else:
            try:
                normalized_runtime = validate_exact_peft_kl_runtime_identity(exact_runtime)
            except ValueError as exc:
                errors.append(f"{label} beta>0 exact PEFT KL runtime is invalid: {exc}")
            else:
                if normalized_runtime["fixed_kl_reference"] != normalized_fixed:
                    errors.append(f"{label} exact PEFT KL runtime differs from its fixed-KL source")
                if float(normalized_runtime["training_beta"]) != float(beta):
                    errors.append(f"{label} exact PEFT KL runtime beta differs from evidence")
                try:
                    normalized_post = validate_exact_peft_kl_post_step_lineage(
                        exact_post_step,
                        expected_fixed_kl_reference=normalized_fixed,
                        expected_runtime_identity=normalized_runtime,
                        expected_run_contract_file_sha256=record.get("run_contract_sha256"),
                        expected_selected_checkpoint_step=step,
                    )
                except ValueError as exc:
                    errors.append(f"{label} beta>0 post-step lineage is invalid: {exc}")
                else:
                    selected_fidelity = normalized_post["selected_checkpoint_fidelity"]
                    if (
                        selected_fidelity["identity_sha256"]
                        != record.get("checkpoint_fidelity_identity_sha256")
                        or selected_fidelity["receipt_sha256"]
                        != record.get("checkpoint_fidelity_receipt_sha256")
                        or selected_fidelity["receipt_file_sha256"]
                        != record.get("checkpoint_fidelity_receipt_file_sha256")
                    ):
                        errors.append(f"{label} post-step selected fidelity differs from evidence")
    return (task, method, seed, step), errors


def _validate_evidence(
    path: Path,
    *,
    policy: dict[str, Any],
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], dict[str, object]]:
    development_contracts = _validate_development_evaluation_policy(
        policy, tasks=policy["expected_tasks"]
    )
    evidence = _load_json(path)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-evidence schema")
    _validate_self_hash(evidence, field="evidence_sha256", label="checkpoint evidence")
    if evidence.get("status") != "complete":
        raise ValueError("checkpoint evidence is not complete")
    if evidence.get("evidence_scope") != "development_only":
        raise ValueError("confirmation/test evidence is forbidden for checkpoint selection")
    if evidence.get("campaign_id") != policy["campaign_id"]:
        raise ValueError("checkpoint evidence campaign_id disagrees with policy")
    if evidence.get("candidate_budget") != policy["candidate_budget"]:
        raise ValueError("checkpoint evidence candidate budget disagrees with policy")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("checkpoint evidence has no records list")
    by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for index, record in enumerate(records):
        key, record_errors = _validate_record(
            record,
            candidate_budget=policy["candidate_budget"],
            row_index=index,
        )
        errors.extend(record_errors)
        if key is not None:
            if key in by_key:
                errors.append(f"duplicate checkpoint evidence key: {key}")
            elif isinstance(record, dict):
                by_key[key] = record
    expected = set(_expected_keys(policy))
    observed = set(by_key)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        errors.append(
            f"checkpoint evidence is missing {len(missing)} scheduled rows: {missing[:8]}"
        )
    if unexpected:
        errors.append(
            f"checkpoint evidence has {len(unexpected)} unscheduled rows: {unexpected[:8]}"
        )
    for task in policy["expected_tasks"]:
        task_records = [record for key, record in by_key.items() if key[0] == str(task)]
        contract = development_contracts[str(task)]
        for record in task_records:
            if (
                record.get("data_manifest_sha256") != contract["data_manifest_sha256"]
                or record.get("protocol_fingerprint") != contract["protocol_fingerprint"]
                or record.get("strict_verifier_manifest_sha256")
                != contract["strict_verifier_manifest_sha256"]
                or record.get("development_evaluation_sha256")
                != contract["development_evaluation_sha256"]
                or record.get("instances") != contract["expected_instances"]
            ):
                errors.append(
                    f"task {task} evidence does not match its frozen development corpus/slice"
                )
        for field in SHARED_TASK_PROTOCOL_FIELDS:
            values = {record.get(field) for record in task_records}
            if len(values) != 1:
                errors.append(f"task {task} uses mixed {field} across methods/seeds/checkpoints")
    if errors:
        raise ValueError("invalid checkpoint evidence: " + " | ".join(errors))
    return by_key, {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
        "evidence_sha256": evidence["evidence_sha256"],
        "record_count": len(records),
    }


def _group_key(key: tuple[str, str, int, int]) -> tuple[str, str, int]:
    return key[0], key[1], key[2]


def _validate_group_provenance(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for field in SHARED_GROUP_FIELDS:
        values = {record[field] for record in records}
        if len(values) != 1:
            errors.append(f"mixed {field} across checkpoints")
    fixed_kl_values = {_canonical_json_bytes(record["fixed_kl_reference"]) for record in records}
    if len(fixed_kl_values) != 1:
        errors.append("mixed fixed_kl_reference across checkpoints")
    exact_runtime_values = {
        _canonical_json_bytes(record.get("exact_peft_kl_runtime")) for record in records
    }
    if len(exact_runtime_values) != 1:
        errors.append("mixed exact_peft_kl_runtime across checkpoints")
    post_values = [record.get("exact_peft_kl_post_step") for record in records]
    if any(value is None for value in post_values):
        if any(value is not None for value in post_values):
            errors.append("mixed exact_peft_kl_post_step presence across checkpoints")
    else:
        for field in ("run_invariant_sha256", "source_reference_sha256", "root_final"):
            values = {_canonical_json_bytes(value[field]) for value in post_values}
            if len(values) != 1:
                errors.append(f"mixed exact PEFT KL {field} across checkpoints")
    evaluation_fingerprints = {record["evaluation_fingerprint"] for record in records}
    if len(evaluation_fingerprints) != len(records):
        errors.append("duplicate evaluation_fingerprint across distinct checkpoints")
    checkpoint_hashes = {record["checkpoint_artifact_sha256"] for record in records}
    if len(checkpoint_hashes) != len(records):
        errors.append("duplicate checkpoint artifact hash across distinct checkpoints")
    fidelity_hashes = {record["checkpoint_fidelity_identity_sha256"] for record in records}
    if len(fidelity_hashes) != len(records):
        errors.append("duplicate checkpoint fidelity identity across distinct checkpoints")
    instance_counts = {record["instances"] for record in records}
    if len(instance_counts) != 1:
        errors.append("checkpoint evaluations use different instance denominators")
    return errors


def _select_group(
    records: list[dict[str, Any]],
    *,
    pass_band: float,
) -> tuple[dict[str, Any], dict[str, object]]:
    best_pass = max(float(record["pass_at_k"]) for record in records)
    threshold = best_pass - pass_band
    eligible = [record for record in records if float(record["pass_at_k"]) >= threshold - 1e-15]
    if not eligible:
        raise ValueError("internal selection error: pass@K band is empty")
    if any(record["conditional_gap_mean"] is None for record in eligible):
        raise ValueError(
            "conditional gap is undefined for a checkpoint in the best-pass band; "
            "selection fails closed"
        )
    selected = min(
        eligible,
        key=lambda record: (
            float(record["conditional_gap_mean"]),
            int(record["checkpoint_step"]),
        ),
    )
    explanation_rows = []
    eligible_steps = {int(record["checkpoint_step"]) for record in eligible}
    for record in sorted(records, key=lambda item: int(item["checkpoint_step"])):
        step = int(record["checkpoint_step"])
        if step not in eligible_steps:
            disposition = "outside_pass_at_k_band"
        elif record is selected:
            disposition = "selected"
        else:
            disposition = "eligible_but_higher_gap_or_later_exact_tie"
        explanation_rows.append(
            {
                "checkpoint_step": step,
                "candidate_budget": record["candidate_budget"],
                "instances": record["instances"],
                "feasible_instances": record["feasible_instances"],
                "pass_at_k": record["pass_at_k"],
                "conditional_gap_n": record["conditional_gap_n"],
                "conditional_gap_mean": record["conditional_gap_mean"],
                "evaluation_fingerprint": record["evaluation_fingerprint"],
                "protocol_fingerprint": record["protocol_fingerprint"],
                "development_evaluation_sha256": record["development_evaluation_sha256"],
                "checkpoint_artifact_sha256": record["checkpoint_artifact_sha256"],
                "checkpoint_fidelity_identity_sha256": record[
                    "checkpoint_fidelity_identity_sha256"
                ],
                "checkpoint_fidelity_receipt_sha256": record["checkpoint_fidelity_receipt_sha256"],
                "checkpoint_fidelity_receipt_file_sha256": record[
                    "checkpoint_fidelity_receipt_file_sha256"
                ],
                "grpo_beta": record["grpo_beta"],
                "fixed_kl_reference": record["fixed_kl_reference"],
                "exact_peft_kl_runtime": record["exact_peft_kl_runtime"],
                "exact_peft_kl_post_step": record["exact_peft_kl_post_step"],
                "run_contract_sha256": record["run_contract_sha256"],
                "data_manifest_sha256": record["data_manifest_sha256"],
                "strict_verifier_manifest_sha256": record["strict_verifier_manifest_sha256"],
                "disposition": disposition,
            }
        )
    selected_fields = {
        key: selected[key]
        for key in (
            "task",
            "method",
            "train_seed",
            "checkpoint_step",
            "pass_at_k",
            "feasible_instances",
            "conditional_gap_mean",
            "conditional_gap_n",
            "instances",
            "candidate_budget",
            "grpo_beta",
            "fixed_kl_reference",
            "exact_peft_kl_runtime",
            "exact_peft_kl_post_step",
            *FINGERPRINT_FIELDS,
        )
    }
    audit = {
        "best_pass_at_k": best_pass,
        "eligibility_threshold": threshold,
        "eligible_checkpoint_steps": sorted(eligible_steps),
        "candidates": explanation_rows,
    }
    return selected_fields, audit


def select_checkpoints(*, policy_path: Path, evidence_path: Path) -> dict[str, object]:
    policy_path = policy_path.expanduser().resolve()
    evidence_path = evidence_path.expanduser().resolve()
    policy = _validate_policy(policy_path)
    by_key, evidence_receipt = _validate_evidence(evidence_path, policy=policy)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for key, record in by_key.items():
        grouped.setdefault(_group_key(key), []).append(record)
    selections: dict[str, object] = {}
    errors: list[str] = []
    for group in sorted(grouped):
        task, method, seed = group
        group_id = f"{task}/{method}/seed{seed}"
        records = grouped[group]
        provenance_errors = _validate_group_provenance(records)
        if provenance_errors:
            selections[group_id] = {
                "status": "FAIL",
                "errors": provenance_errors,
            }
            errors.extend(f"{group_id}: {error}" for error in provenance_errors)
            continue
        try:
            selected, selection_audit = _select_group(
                records,
                pass_band=float(policy["pass_at_k_absolute_band"]),
            )
        except ValueError as exc:
            selections[group_id] = {"status": "FAIL", "errors": [str(exc)]}
            errors.append(f"{group_id}: {exc}")
            continue
        selections[group_id] = {
            "status": "PASS",
            "selected": selected,
            "selection_audit": selection_audit,
            "errors": [],
        }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "selector": "clasp_development_checkpoint_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "campaign_id": policy["campaign_id"],
        "stage_id": policy["stage_id"],
        "decision": "SELECTED" if not errors else "HOLD",
        "evidence_scope": "development_only",
        "confirmation_evidence_used": False,
        "policy": {
            "path": str(policy_path),
            "file_sha256": _sha256_file(policy_path),
            "policy_sha256": policy["policy_sha256"],
            "frozen_at_utc": policy["frozen_at_utc"],
            "selection_rule": SELECTION_RULE,
            "pass_at_k_absolute_band": policy["pass_at_k_absolute_band"],
            "frozen_policy": policy,
        },
        "evidence": evidence_receipt,
        "expected_group_count": (
            len(policy["expected_tasks"])
            * len(policy["expected_methods"])
            * len(policy["train_seeds"])
        ),
        "selected_group_count": sum(
            row["status"] == "PASS"
            for row in selections.values()  # type: ignore[index]
        ),
        "selections": selections,
        "errors": errors,
        "side_effects": {
            "checkpoints_modified": False,
            "evaluator_state_modified": False,
            "scheduler_actions": [],
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing selection receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing selection receipt: {path}"
            ) from exc
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = select_checkpoints(
            policy_path=args.policy,
            evidence_path=args.evidence,
        )
    except (OSError, TypeError, ValueError) as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "selector": "clasp_development_checkpoint_selection",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "decision": "HOLD",
            "evidence_scope": "development_only",
            "confirmation_evidence_used": False,
            "errors": [str(exc)],
            "side_effects": {
                "checkpoints_modified": False,
                "evaluator_state_modified": False,
                "scheduler_actions": [],
            },
        }
        payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    try:
        _atomic_json(args.output, payload)
    except FileExistsError as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 2
    print(f"{payload['decision']}: wrote {args.output}")
    if payload["decision"] != "SELECTED":
        for error in payload.get("errors", [])[:20]:
            print(f"  - {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
