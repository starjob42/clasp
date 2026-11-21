"""Independent, fail-closed replay of fixed-null construction receipts."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from clasp.data import loaders as data_loaders
from clasp.data import prompts as prompt_renderers
from clasp.rewards import heads
from clasp.set_grpo.fixed_null_construction import (
    LIVE_CONSTRUCTION_PROTOCOL,
    _read_training_rows,
    behavior_environment_names,
    canonical_json_bytes,
    evaluator_identity,
    implementation_file_identities,
    representation_identity,
    sealed_behavior_environment,
    sha256_bytes,
    sha256_file,
    validate_construction_spec,
    validate_null_candidate,
)
from clasp.set_grpo.fixed_null_receipt import load_fixed_null_score_receipt, prompt_sha256

_REQUIRED_LIVE_FIELDS = {
    "construction_protocol",
    "construction_spec",
    "construction_spec_file_sha256",
    "null_candidate",
    "null_candidate_file_sha256",
    "training_rows_identity",
    "representation_identity",
    "representation_identity_sha256",
    "evaluator_identity",
    "construction_implementation_files",
    "records",
}
_EVALUATOR_CONFIG_KEYS = {
    "pd_component_profile",
    "pd_soft_objective_for_infeasible",
    "pd_constraint_target",
    "pd_cost_limit",
    "pd_violation_tolerance",
}


def resolve_single_training_rows(path: Path) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError("fixed-null training rows path must not be a symbolic link")
    resolved = supplied.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".json":
            raise ValueError("fixed-null training rows must be a JSON file")
        return resolved
    if not resolved.is_dir():
        raise ValueError(f"fixed-null training rows path is missing: {resolved}")
    candidates = sorted(
        candidate
        for candidate in resolved.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".json"
    )
    if len(candidates) != 1:
        raise ValueError(
            "fixed-null construction requires exactly one frozen training JSON in the "
            f"contracted train directory; found {len(candidates)} under {resolved}"
        )
    if candidates[0].is_symlink():
        raise ValueError("fixed-null training JSON must not be a symbolic link")
    return candidates[0].resolve()


def _read_receipt_payload(path: Path) -> dict[str, Any]:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError("fixed-null receipt must not be a symbolic link")
    resolved = supplied.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixed-null receipt is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError("fixed-null receipt must be a JSON object")
    return payload


def _render_rows_independently(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    prompt_style: str,
    no_reason: bool,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        transformed = data_loaders.transform_example_for_task(
            task,
            str(row.get("instruction", "")),
            str(row.get("input", "")),
            str(row.get("output", "")),
            row.get("instance_json"),
        )
        prompt = prompt_renderers.build_formatted_prompt(
            str(transformed["instruction"]),
            str(transformed["input"]),
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=None,
            strict_chat_prompt=True,
        )
        prompt_digest = prompt_sha256(prompt)
        if prompt_digest in seen:
            raise ValueError("independent replay found duplicate rendered prompt identities")
        seen.add(prompt_digest)
        rendered.append(
            {
                "prompt": prompt,
                "answer": str(transformed["output"]),
                "instruction": str(transformed["instruction"]),
                "input": str(transformed["input"]),
                "instance_json": data_loaders.coerce_json_string(transformed["instance_json"]),
                "task_type": task,
                "row_sha256": sha256_bytes(canonical_json_bytes(dict(row))),
            }
        )
    return rendered


def _replay_scores_independently(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    completion: str,
) -> list[dict[str, Any]]:
    evaluator = spec["evaluator"]
    args = SimpleNamespace(
        pd_component_profile=evaluator["pd_component_profile"],
        pd_soft_objective_for_infeasible=evaluator["pd_soft_objective_for_infeasible"],
        pd_constraint_target=float(evaluator["pd_constraint_target"]),
        pd_cost_limit=float(evaluator["pd_cost_limit"]),
        pd_lambda_init=0.0,
        pd_dual_lr=0.0,
        pd_dual_lr_up=0.0,
        pd_dual_lr_down=0.0,
        pd_dual_update_interval=1,
        pd_violation_ema_beta=1.0,
        pd_lambda_max=0.0,
        pd_violation_tolerance=float(evaluator["pd_violation_tolerance"]),
        pd_objective_weight=1.0,
        pd_structure_weight=0.0,
        pd_sync_across_ranks=False,
        pd_strict_constraint_range=True,
        pd_log_every=0,
        clasp_fixed_null_set=False,
    )
    task = str(spec["task"])
    manager = heads.build_primal_dual_manager(args, [task])
    manager.set_collect_stats(False)
    prompts = [str(row["prompt"]) for row in rows]
    kwargs = {
        key: [row[key] for row in rows]
        for key in ("answer", "instruction", "input", "instance_json", "task_type")
    }
    manager(prompts, [completion] * len(rows), **kwargs)
    components = manager.get_last_components()
    objective = components.get("objective_scores")
    violation = components.get("max_constraint_violation_scores")
    feasibility = components.get("feasibility_flags")
    if not all(isinstance(values, list) and len(values) == len(rows) for values in (objective, violation, feasibility)):
        raise RuntimeError("independent replay did not receive one strict score triplet per row")
    replayed: list[dict[str, Any]] = []
    for row, objective_value, violation_value, feasibility_value in zip(
        rows, objective, violation, feasibility
    ):
        values = (float(objective_value), float(violation_value), float(feasibility_value))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("independent fixed-null replay produced non-finite scores")
        replayed.append(
            {
                "task": task,
                "prompt_sha256": prompt_sha256(row["prompt"]),
                "row_sha256": str(row["row_sha256"]),
                "objective_score": values[0],
                "max_constraint_violation_score": values[1],
                "feasibility_flag": values[2],
            }
        )
    return replayed


def _normalized_active_evaluator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(_EVALUATOR_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"active SET-NULL config lacks evaluator keys: {missing}")
    return {
        "pd_component_profile": str(config["pd_component_profile"]),
        "pd_soft_objective_for_infeasible": bool(config["pd_soft_objective_for_infeasible"]),
        "pd_constraint_target": float(config["pd_constraint_target"]),
        "pd_cost_limit": float(config["pd_cost_limit"]),
        "pd_violation_tolerance": float(config["pd_violation_tolerance"]),
    }


def verify_fixed_null_receipt(
    *,
    receipt_path: Path,
    training_rows_path: Path,
    expected_task: str | None = None,
    expected_prompt_style: str | None = None,
    expected_no_reason: bool | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reopen source rows and replay every prompt/score using active code."""

    loaded = load_fixed_null_score_receipt(receipt_path)
    payload = _read_receipt_payload(receipt_path)
    missing_live = sorted(_REQUIRED_LIVE_FIELDS - set(payload))
    if missing_live or payload.get("construction_protocol") != LIVE_CONSTRUCTION_PROTOCOL:
        raise ValueError(
            "fixed-null receipt has no replayable live-construction contract; "
            f"missing={missing_live}"
        )
    spec_raw = payload.get("construction_spec")
    if not isinstance(spec_raw, dict):
        raise ValueError("fixed-null receipt construction_spec must be an object")
    spec = validate_construction_spec(spec_raw, expected_task=expected_task)
    task = str(spec["task"])
    candidate_raw = payload.get("null_candidate")
    if not isinstance(candidate_raw, dict):
        raise ValueError("fixed-null receipt null_candidate must be an object")
    candidate = validate_null_candidate(candidate_raw, expected_task=task)

    if expected_prompt_style is not None and str(spec["representation"]["prompt_style"]) != str(expected_prompt_style):
        raise ValueError("fixed-null receipt prompt style differs from the active run")
    if expected_no_reason is not None and bool(spec["representation"]["no_reason"]) != bool(expected_no_reason):
        raise ValueError("fixed-null receipt reasoning mode differs from the active run")
    if expected_config is not None:
        active_evaluator = _normalized_active_evaluator_config(expected_config)
        if spec["evaluator"] != active_evaluator:
            raise ValueError("fixed-null receipt evaluator config differs from the active run")
    if expected_environment is not None:
        active_behavior_environment = {
            name: str(expected_environment.get(name, ""))
            for name in behavior_environment_names()
        }
        if spec["behavior_environment"] != active_behavior_environment:
            raise ValueError("fixed-null receipt behavior environment differs from the active run")

    rows_path = resolve_single_training_rows(training_rows_path)
    rows, rows_identity = _read_training_rows(rows_path, expected_task=task)
    if payload.get("training_rows_identity") != rows_identity:
        raise ValueError("fixed-null receipt training-row identity differs from reopened rows")

    current_files = implementation_file_identities(include_construction=True)
    if payload.get("construction_implementation_files") != current_files:
        raise ValueError("fixed-null construction implementation differs from active source")
    current_evaluator = evaluator_identity(spec)
    if payload.get("evaluator_identity") != current_evaluator:
        raise ValueError("fixed-null evaluator identity differs from active source/config")
    if payload.get("evaluator_identity_sha256") != sha256_bytes(canonical_json_bytes(current_evaluator)):
        raise ValueError("fixed-null evaluator identity digest does not match its structured identity")
    current_representation = representation_identity(spec)
    if payload.get("representation_identity") != current_representation:
        raise ValueError("fixed-null representation identity differs from active source/config")
    if payload.get("representation_identity_sha256") != sha256_bytes(canonical_json_bytes(current_representation)):
        raise ValueError("fixed-null representation identity digest does not match")

    with sealed_behavior_environment(spec["behavior_environment"]):
        rendered = _render_rows_independently(
            rows,
            task=task,
            prompt_style=str(spec["representation"]["prompt_style"]),
            no_reason=bool(spec["representation"]["no_reason"]),
        )
        replayed_records = _replay_scores_independently(
            rendered,
            spec=spec,
            completion=str(candidate["completion"]),
        )
    if payload.get("records") != replayed_records:
        raise ValueError("fixed-null independent replay differs from frozen prompt scores")
    expected_keys = {(task, prompt_sha256(row["prompt"])) for row in rendered}
    loaded_keys = {
        (str(record.get("task", "")).strip().lower(), str(record.get("prompt_sha256", "")))
        for record in payload["records"]
        if isinstance(record, dict)
    }
    if loaded_keys != expected_keys:
        raise ValueError("fixed-null receipt does not exactly cover every frozen training prompt")

    replay_source = Path(inspect.getsourcefile(verify_fixed_null_receipt) or "").resolve()
    return {
        "schema_version": 1,
        "protocol": "clasp-fixed-null-independent-replay-v1",
        "status": "PASS",
        "task": task,
        "receipt_sha256": loaded.identity["receipt_sha256"],
        "training_rows_file_sha256": rows_identity["file_sha256"],
        "verified_record_count": len(replayed_records),
        "replay_implementation_sha256": sha256_file(replay_source),
    }


__all__ = ["resolve_single_training_rows", "verify_fixed_null_receipt"]
