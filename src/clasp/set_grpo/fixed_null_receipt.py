"""Strict receipts for externally constructed fixed-null evaluator scores.

The publication contract does not choose a null candidate.  This module does
not choose one either.  It only validates a frozen construction receipt and
provides prompt-keyed evaluator outputs to the trainer.  Whoever constructs a
receipt remains responsible for showing that its ``null_candidate`` and
``evaluator_identity_sha256`` implement the preregistered scientific null.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL = "clasp-fixed-null-construction-v1"
STATUS = "frozen_before_training"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("fixed-null values must be finite canonical JSON") from exc
    return text.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"fixed-null receipt {field} must be a lowercase SHA-256 digest")
    return normalized


def prompt_sha256(prompt: Any) -> str:
    """Return the stable lookup key used by construction and training."""

    return _sha256_bytes(_canonical_json_bytes(prompt))


@dataclass(frozen=True)
class FixedNullEvaluatorScores:
    objective_score: float
    max_constraint_violation_score: float
    feasibility_flag: float


class FixedNullScoreReceipt:
    """Validated immutable lookup keyed by ``(task, prompt_sha256)``."""

    def __init__(
        self,
        *,
        identity: dict[str, Any],
        records: dict[tuple[str, str], FixedNullEvaluatorScores],
    ) -> None:
        self.identity = dict(identity)
        self._records = dict(records)

    def scores_for(self, task: str, prompt: Any) -> FixedNullEvaluatorScores:
        normalized_task = str(task).strip().lower()
        digest = prompt_sha256(prompt)
        try:
            return self._records[(normalized_task, digest)]
        except KeyError as exc:
            raise RuntimeError(
                "Fixed-null construction receipt has no evaluator scores for "
                f"task={normalized_task!r}, prompt_sha256={digest}. Refusing to "
                "substitute an implicit null."
            ) from exc


def _numeric_score(value: Any, *, field: str, nonnegative: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"fixed-null receipt {field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"fixed-null receipt {field} must be {qualifier}")
    return numeric


def load_fixed_null_score_receipt(path: str | os.PathLike[str]) -> FixedNullScoreReceipt:
    """Load and validate a frozen fixed-null construction receipt.

    The receipt intentionally stores prompt hashes rather than training rows.
    Every prompt used by a fixed-null run must be covered; lookup never falls
    back to a default candidate or default score.
    """

    supplied_path = Path(path).expanduser()
    if supplied_path.is_symlink():
        raise ValueError("fixed-null construction receipt must not be a symbolic link")
    receipt_path = supplied_path.resolve()
    if not receipt_path.is_file():
        raise ValueError(f"fixed-null construction receipt is missing: {receipt_path}")
    mode = receipt_path.stat().st_mode & 0o777
    if mode & 0o222:
        raise ValueError(
            "fixed-null construction receipt must be frozen read-only before training; "
            f"mode is {mode:04o}"
        )
    try:
        with receipt_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"fixed-null construction receipt is not valid JSON: {receipt_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("fixed-null construction receipt must be a JSON object")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"fixed-null receipt schema_version must equal {SCHEMA_VERSION}"
        )
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"fixed-null receipt protocol must equal {PROTOCOL!r}")
    if payload.get("status") != STATUS:
        raise ValueError(f"fixed-null receipt status must equal {STATUS!r}")

    if "null_candidate" not in payload:
        raise ValueError("fixed-null receipt must contain the exact null_candidate")
    null_candidate_sha256 = _require_sha256(
        payload.get("null_candidate_sha256"),
        field="null_candidate_sha256",
    )
    observed_null_sha256 = _sha256_bytes(
        _canonical_json_bytes(payload["null_candidate"])
    )
    if observed_null_sha256 != null_candidate_sha256:
        raise ValueError("fixed-null receipt null_candidate_sha256 does not match")
    evaluator_identity_sha256 = _require_sha256(
        payload.get("evaluator_identity_sha256"),
        field="evaluator_identity_sha256",
    )

    receipt_sha256 = _require_sha256(
        payload.get("receipt_sha256"),
        field="receipt_sha256",
    )
    unhashed_payload = dict(payload)
    unhashed_payload.pop("receipt_sha256", None)
    observed_receipt_sha256 = _sha256_bytes(_canonical_json_bytes(unhashed_payload))
    if observed_receipt_sha256 != receipt_sha256:
        raise ValueError("fixed-null receipt self-hash does not match")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("fixed-null receipt records must be a non-empty list")
    records: dict[tuple[str, str], FixedNullEvaluatorScores] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"fixed-null receipt record {index} must be an object")
        task = str(record.get("task", "")).strip().lower()
        if not task:
            raise ValueError(f"fixed-null receipt record {index} has no task")
        prompt_digest = _require_sha256(
            record.get("prompt_sha256"),
            field=f"records[{index}].prompt_sha256",
        )
        objective = _numeric_score(
            record.get("objective_score"),
            field=f"records[{index}].objective_score",
            nonnegative=True,
        )
        violation = _numeric_score(
            record.get("max_constraint_violation_score"),
            field=f"records[{index}].max_constraint_violation_score",
            nonnegative=True,
        )
        feasibility = _numeric_score(
            record.get("feasibility_flag"),
            field=f"records[{index}].feasibility_flag",
            nonnegative=True,
        )
        if feasibility not in {0.0, 1.0}:
            raise ValueError(
                f"fixed-null receipt record {index} feasibility_flag must be 0 or 1"
            )
        expected_feasibility = 1.0 if violation <= 1e-8 else 0.0
        if feasibility != expected_feasibility:
            raise ValueError(
                f"fixed-null receipt record {index} feasibility_flag disagrees with "
                "max_constraint_violation_score"
            )
        key = (task, prompt_digest)
        if key in records:
            raise ValueError(
                "fixed-null receipt contains a duplicate task/prompt record: "
                f"{task}/{prompt_digest}"
            )
        records[key] = FixedNullEvaluatorScores(
            objective_score=objective,
            max_constraint_violation_score=violation,
            feasibility_flag=feasibility,
        )

    tasks = sorted({task for task, _ in records})
    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "receipt_sha256": receipt_sha256,
        "receipt_file_sha256": _sha256_file(receipt_path),
        "null_candidate_sha256": null_candidate_sha256,
        "evaluator_identity_sha256": evaluator_identity_sha256,
        "record_count": len(records),
        "tasks": tasks,
    }
    return FixedNullScoreReceipt(identity=identity, records=records)


__all__ = [
    "FixedNullEvaluatorScores",
    "FixedNullScoreReceipt",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "STATUS",
    "load_fixed_null_score_receipt",
    "prompt_sha256",
]
