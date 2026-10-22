"""Deterministic construction of prompt-keyed fixed-null score receipts.

This module deliberately does not select a null candidate.  It accepts one
explicit, frozen candidate artifact and evaluates its exact completion against
every row in one frozen single-task training JSON file.  The construction spec
closes every CLASP environment variable referenced by the active row
representation and reward implementations, so inherited research knobs cannot
silently change the artifact.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

from clasp.data import loaders as data_loaders
from clasp.data import prompts as prompt_renderers
from clasp.rewards import co_tasks, heads, primal_dual
from clasp.set_grpo.fixed_null_receipt import PROTOCOL, SCHEMA_VERSION, STATUS, prompt_sha256

CONSTRUCTION_SPEC_PROTOCOL = "clasp-fixed-null-construction-spec-v1"
NULL_CANDIDATE_PROTOCOL = "clasp-fixed-null-candidate-v1"
LIVE_CONSTRUCTION_PROTOCOL = "clasp-fixed-null-live-construction-v1"
FROZEN_INPUT_STATUS = "frozen_before_training"

_SPEC_KEYS = {
    "schema_version",
    "protocol",
    "status",
    "task",
    "representation",
    "evaluator",
    "behavior_environment",
    "spec_sha256",
}
_REPRESENTATION_KEYS = {"prompt_style", "no_reason"}
_EVALUATOR_KEYS = {
    "pd_component_profile",
    "pd_soft_objective_for_infeasible",
    "pd_constraint_target",
    "pd_cost_limit",
    "pd_violation_tolerance",
}
_CANDIDATE_KEYS = {
    "schema_version",
    "protocol",
    "status",
    "task",
    "completion",
    "scientific_parameters",
    "candidate_sha256",
}
_SUPPORTED_TASKS = frozenset(co_tasks.TASK_REWARD_TRIPLETS)
_NONDETERMINISTIC_PROMPT_STYLES = frozenset({"auto", "alpaca_chat"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BEHAVIOR_ENV_RE = re.compile(r"\bCLASP_[A-Z0-9_]+\b")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(module: object) -> Path:
    raw = inspect.getsourcefile(module)
    if not raw:
        raise RuntimeError(f"cannot resolve source path for {module!r}")
    path = Path(raw).resolve()
    if not path.is_file():
        raise RuntimeError(f"resolved implementation source is missing: {path}")
    return path


def implementation_file_identities(*, include_construction: bool) -> dict[str, str]:
    modules: list[tuple[str, object]] = [
        ("clasp.data.loaders", data_loaders),
        ("clasp.data.prompts", prompt_renderers),
        ("clasp.rewards.co_tasks", co_tasks),
        ("clasp.rewards.heads", heads),
        ("clasp.rewards.primal_dual", primal_dual),
    ]
    if include_construction:
        import clasp.set_grpo.fixed_null_construction as construction_module

        modules.append(("clasp.set_grpo.fixed_null_construction", construction_module))
    return {name: sha256_file(_source_path(module)) for name, module in modules}


def behavior_environment_names() -> tuple[str, ...]:
    """Return every CLASP behavior variable referenced by active implementations.

    This is intentionally source-derived.  Adding a new environment-dependent
    branch makes old specs fail their exact-key check until it is reviewed and
    frozen explicitly.
    """

    names: set[str] = set()
    for module in (data_loaders, co_tasks, primal_dual):
        names.update(_BEHAVIOR_ENV_RE.findall(_source_path(module).read_text(encoding="utf-8")))
    return tuple(sorted(names))


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_frozen_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = supplied.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    mode = resolved.stat().st_mode & 0o777
    if mode & 0o222:
        raise ValueError(f"{label} must be frozen read-only; mode is {mode:04o}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, sha256_file(resolved)


def _validate_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> str:
    stored = _require_sha256(payload.get(field), field=f"{label}.{field}")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    observed = sha256_bytes(canonical_json_bytes(unhashed))
    if observed != stored:
        raise ValueError(f"{label}.{field} does not match the canonical payload")
    return stored


def validate_construction_spec(payload: Mapping[str, Any], *, expected_task: str | None = None) -> dict[str, Any]:
    if set(payload) != _SPEC_KEYS:
        raise ValueError(
            "fixed-null construction spec must have the exact key set; "
            f"missing={sorted(_SPEC_KEYS - set(payload))}, extra={sorted(set(payload) - _SPEC_KEYS)}"
        )
    if payload.get("schema_version") != 1 or payload.get("protocol") != CONSTRUCTION_SPEC_PROTOCOL:
        raise ValueError("fixed-null construction spec schema/protocol is invalid")
    if payload.get("status") != FROZEN_INPUT_STATUS:
        raise ValueError(
            "fixed-null construction spec remains on HOLD; freeze all scientific and evaluator "
            "parameters before construction"
        )
    task = str(payload.get("task", "")).strip().lower()
    if task not in _SUPPORTED_TASKS or (expected_task is not None and task != expected_task):
        raise ValueError(f"fixed-null construction spec has invalid task {task!r}")

    representation = payload.get("representation")
    if not isinstance(representation, dict) or set(representation) != _REPRESENTATION_KEYS:
        raise ValueError("construction spec representation must contain exactly prompt_style/no_reason")
    prompt_style = prompt_renderers.normalize_prompt_style(representation.get("prompt_style"))
    if prompt_style in _NONDETERMINISTIC_PROMPT_STYLES:
        raise ValueError(
            "fixed-null construction requires an explicit tokenizer-independent prompt style; "
            "auto and alpaca_chat are not accepted"
        )
    if not isinstance(representation.get("no_reason"), bool):
        raise ValueError("construction spec representation.no_reason must be boolean")

    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict) or set(evaluator) != _EVALUATOR_KEYS:
        raise ValueError(
            "construction spec evaluator has a non-exact key set; "
            f"expected={sorted(_EVALUATOR_KEYS)}"
        )
    profile = str(evaluator.get("pd_component_profile", "")).strip()
    if not profile:
        raise ValueError("construction spec evaluator.pd_component_profile must be non-empty")
    if not isinstance(evaluator.get("pd_soft_objective_for_infeasible"), bool):
        raise ValueError("construction spec evaluator.pd_soft_objective_for_infeasible must be boolean")
    for name in ("pd_constraint_target", "pd_cost_limit", "pd_violation_tolerance"):
        value = evaluator.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"construction spec evaluator.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not (0.0 <= number <= 1.0):
            raise ValueError(f"construction spec evaluator.{name} must lie in [0, 1]")

    behavior_environment = payload.get("behavior_environment")
    expected_environment_names = set(behavior_environment_names())
    if not isinstance(behavior_environment, dict) or set(behavior_environment) != expected_environment_names:
        observed = set(behavior_environment) if isinstance(behavior_environment, dict) else set()
        raise ValueError(
            "construction spec behavior_environment must close every active CLASP behavior variable; "
            f"missing={sorted(expected_environment_names - observed)}, "
            f"extra={sorted(observed - expected_environment_names)}"
        )
    if not all(isinstance(value, str) for value in behavior_environment.values()):
        raise ValueError("construction spec behavior_environment values must all be strings")

    _validate_self_hash(payload, field="spec_sha256", label="fixed-null construction spec")
    return json.loads(json.dumps(payload))


def validate_null_candidate(payload: Mapping[str, Any], *, expected_task: str) -> dict[str, Any]:
    if set(payload) != _CANDIDATE_KEYS:
        raise ValueError(
            "fixed-null candidate must have the exact key set; "
            f"missing={sorted(_CANDIDATE_KEYS - set(payload))}, extra={sorted(set(payload) - _CANDIDATE_KEYS)}"
        )
    if payload.get("schema_version") != 1 or payload.get("protocol") != NULL_CANDIDATE_PROTOCOL:
        raise ValueError("fixed-null candidate schema/protocol is invalid")
    if payload.get("status") != FROZEN_INPUT_STATUS:
        raise ValueError(
            "fixed-null candidate remains on HOLD; the scientific null completion must be "
            "chosen and frozen explicitly"
        )
    task = str(payload.get("task", "")).strip().lower()
    if task != expected_task:
        raise ValueError(f"fixed-null candidate task {task!r} does not equal {expected_task!r}")
    if not isinstance(payload.get("completion"), str):
        raise ValueError("fixed-null candidate completion must be an exact string")
    if not isinstance(payload.get("scientific_parameters"), dict) or not payload.get(
        "scientific_parameters"
    ):
        raise ValueError(
            "fixed-null candidate scientific_parameters must be an explicit non-empty object"
        )
    _validate_self_hash(payload, field="candidate_sha256", label="fixed-null candidate")
    return json.loads(json.dumps(payload))


@contextmanager
def sealed_behavior_environment(values: Mapping[str, str]) -> Iterator[None]:
    names = behavior_environment_names()
    if set(values) != set(names):
        raise ValueError("sealed behavior environment does not have the active exact key set")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
            if values[name] != "":
                os.environ[name] = values[name]
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
            if previous[name] is not None:
                os.environ[name] = str(previous[name])


def _read_training_rows(
    path: Path, *, expected_task: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError("fixed-null training rows must not be a symbolic link")
    resolved = supplied.resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ValueError(f"fixed-null training rows must be one JSON file: {resolved}")
    mode = resolved.stat().st_mode & 0o777
    if mode & 0o222:
        raise ValueError(f"fixed-null training rows must be frozen read-only; mode is {mode:04o}")
    raw_rows = data_loaders.read_json_rows(str(resolved))
    if expected_task is not None:
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                continue
            declared = raw_row.get("task_type", raw_row.get("task"))
            if declared is not None and str(declared).strip().lower() != expected_task:
                raise ValueError(
                    "fixed-null training rows are not single-task: "
                    f"row {row_index} declares {declared!r}, expected {expected_task!r}"
                )
    normalized = data_loaders.normalize_rows(raw_rows, str(resolved))
    deduplicated = data_loaders.deduplicate_rows(normalized, label=str(resolved))
    if not deduplicated:
        raise ValueError("fixed-null training rows are empty after normalization")
    identity = {
        "file_sha256": sha256_file(resolved),
        "source_record_count": len(raw_rows),
        "normalized_record_count": len(deduplicated),
        "normalized_rows_sha256": sha256_bytes(canonical_json_bytes(deduplicated)),
    }
    return deduplicated, identity


def _prepared_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    prompt_style: str,
    no_reason: bool,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_prompts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
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
        digest = prompt_sha256(prompt)
        if digest in seen_prompts:
            raise ValueError(
                "fixed-null training rows render a duplicate prompt identity at rows "
                f"{seen_prompts[digest]} and {row_index}; prompt-keyed null scores would be ambiguous"
            )
        seen_prompts[digest] = row_index
        prepared.append(
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
    return prepared


def _manager_for_spec(spec: Mapping[str, Any]):
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
    manager = heads.build_primal_dual_manager(args, [str(spec["task"])])
    manager.set_collect_stats(False)
    return manager


def _score_rows(
    prepared: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    completion: str,
) -> list[dict[str, Any]]:
    manager = _manager_for_spec(spec)
    prompts = [str(row["prompt"]) for row in prepared]
    completions = [completion] * len(prepared)
    kwargs = {
        key: [row[key] for row in prepared]
        for key in ("answer", "instruction", "input", "instance_json", "task_type")
    }
    manager(prompts, completions, **kwargs)
    components = manager.get_last_components()
    objective = components.get("objective_scores")
    violation = components.get("max_constraint_violation_scores")
    feasibility = components.get("feasibility_flags")
    if not all(isinstance(values, list) and len(values) == len(prepared) for values in (objective, violation, feasibility)):
        raise RuntimeError("active strict evaluator did not return one null score triplet per row")
    records: list[dict[str, Any]] = []
    for row, objective_value, violation_value, feasibility_value in zip(
        prepared, objective, violation, feasibility
    ):
        values = (float(objective_value), float(violation_value), float(feasibility_value))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("active strict evaluator returned a non-finite fixed-null score")
        records.append(
            {
                "task": str(spec["task"]),
                "prompt_sha256": prompt_sha256(row["prompt"]),
                "row_sha256": str(row["row_sha256"]),
                "objective_score": values[0],
                "max_constraint_violation_score": values[1],
                "feasibility_flag": values[2],
            }
        )
    return records


def evaluator_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    task = str(spec["task"])
    triplet = co_tasks.TASK_REWARD_TRIPLETS[task]
    cost_fn = co_tasks.TASK_COST_COMPONENTS[task]
    return {
        "schema_version": 1,
        "protocol": "clasp-active-strict-null-evaluator-v1",
        "task": task,
        "reward_callables": [f"{fn.__module__}.{fn.__qualname__}" for fn in triplet],
        "cost_callable": f"{cost_fn.__module__}.{cost_fn.__qualname__}",
        "evaluator": json.loads(json.dumps(spec["evaluator"])),
        "behavior_environment": json.loads(json.dumps(spec["behavior_environment"])),
        "implementation_files": implementation_file_identities(include_construction=False),
    }


def representation_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    source_identities = implementation_file_identities(include_construction=False)
    return {
        "schema_version": 1,
        "protocol": "clasp-fixed-null-row-representation-v1",
        "task": str(spec["task"]),
        "representation": json.loads(json.dumps(spec["representation"])),
        "behavior_environment": json.loads(json.dumps(spec["behavior_environment"])),
        "implementation_files": {
            name: source_identities[name]
            for name in ("clasp.data.loaders", "clasp.data.prompts")
        },
    }


def build_fixed_null_receipt(
    *,
    construction_spec_path: Path,
    null_candidate_path: Path,
    training_rows_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    spec_payload, spec_file_sha256 = _read_frozen_json(
        construction_spec_path, label="fixed-null construction spec"
    )
    spec = validate_construction_spec(spec_payload)
    task = str(spec["task"])
    candidate_payload, candidate_file_sha256 = _read_frozen_json(
        null_candidate_path, label="fixed-null candidate"
    )
    candidate = validate_null_candidate(candidate_payload, expected_task=task)
    rows, rows_identity = _read_training_rows(training_rows_path, expected_task=task)

    with sealed_behavior_environment(spec["behavior_environment"]):
        prepared = _prepared_rows(
            rows,
            task=task,
            prompt_style=str(spec["representation"]["prompt_style"]),
            no_reason=bool(spec["representation"]["no_reason"]),
        )
        records = _score_rows(prepared, spec=spec, completion=str(candidate["completion"]))
        evaluator = evaluator_identity(spec)
        representation = representation_identity(spec)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "construction_protocol": LIVE_CONSTRUCTION_PROTOCOL,
        "construction_spec": spec,
        "construction_spec_file_sha256": spec_file_sha256,
        "null_candidate": candidate,
        "null_candidate_file_sha256": candidate_file_sha256,
        "null_candidate_sha256": sha256_bytes(canonical_json_bytes(candidate)),
        "training_rows_identity": rows_identity,
        "representation_identity": representation,
        "representation_identity_sha256": sha256_bytes(canonical_json_bytes(representation)),
        "evaluator_identity": evaluator,
        "evaluator_identity_sha256": sha256_bytes(canonical_json_bytes(evaluator)),
        "construction_implementation_files": implementation_file_identities(
            include_construction=True
        ),
        "records": records,
    }
    payload["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))

    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o440)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite fixed-null receipt: {destination}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o440)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return payload


__all__ = [
    "CONSTRUCTION_SPEC_PROTOCOL",
    "FROZEN_INPUT_STATUS",
    "LIVE_CONSTRUCTION_PROTOCOL",
    "NULL_CANDIDATE_PROTOCOL",
    "behavior_environment_names",
    "build_fixed_null_receipt",
    "canonical_json_bytes",
    "evaluator_identity",
    "implementation_file_identities",
    "representation_identity",
    "sealed_behavior_environment",
    "sha256_bytes",
    "sha256_file",
    "validate_construction_spec",
    "validate_null_candidate",
]
