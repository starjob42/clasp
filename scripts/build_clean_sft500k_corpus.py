#!/usr/bin/env python3
"""Audit and build a clean, leakage-disjoint 500k specialist SFT corpus.

The released CVRP corpus intentionally contains multiple demonstrations for
some identical instances.  Different, strictly valid solutions are useful
multi-reference supervision and are retained.  Byte-for-byte equivalent
training demonstrations, rows colliding with a reserved evaluation instance,
and labels rejected by the shared strict verifier are removed.  Replacement
rows must be strictly valid and semantically new relative to every retained
source row and every reserved evaluation row.

The implementation streams top-level JSON arrays.  It never materializes a
multi-gigabyte task file and it writes a create-once, hash-bound receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
PROTOCOL = "clasp-clean-specialist-sft500k-v2"
TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
EXPECTED_ROWS = 500_000
RESERVED_SOURCE_CLASSES = ("sft_eval", "rl_eval", "rl_train", "iid_confirmation")
REPLACEMENT_PROTOCOL = "clasp-solver-backed-sft-replacement-shard-v1"
JSSP_RELABEL_PROTOCOL = "clasp-strict-jssp-terminal-makespan-relabel-v1"
INTEGER_OBJECTIVE_TASKS = frozenset({"jssp", "mis", "op"})
FLOAT_OBJECTIVE_TASKS = frozenset({"cvrp", "tsp"})

# Match the released Jiang-f638 data representation. All optional CLASP
# representation rewrites are disabled; in particular, released MIS ``Set:``
# labels are preserved byte-for-byte for SFT.
REPRESENTATION_ENVIRONMENT = {
    "CLASP_TSP_PROMPT_KNN": "0",
    "CLASP_TSP_PROMPT_KNN_DISTANCES": "",
    "CLASP_TSP_INPUT_ORDER": "",
    "CLASP_TSP_RELABEL_ORDER": "",
    "CLASP_TSP_ROUTE_CANONICALIZE": "",
    "CLASP_TSP_ROUTE_ONLY_OUTPUT": "",
    "CLASP_TSP_LABEL_TTA_SCHEDULE": "",
    "CLASP_MIS_RELABEL_ORDER": "",
    "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0",
    "CLASP_MIS_PROMPT_NODE_DEGREES": "0",
    "CLASP_MIS_PROMPT_LOW_DEGREE_K": "32",
}

_OBJECTIVE_PATTERN = re.compile(
    r"(?:Objective|Makespan)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE,
)
_JSSP_TERMINAL_MAKESPAN_PATTERN = re.compile(
    r"(?i)(Makespan\s*:\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(\s*)\Z"
)


class CleanSFTError(ValueError):
    """Raised when a clean-corpus invariant cannot be certified."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat(follow_symlinks=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise CleanSFTError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CleanSFTError(f"artifact is not a regular file: {path}")
    return {
        "canonical_path": str(path.resolve(strict=True)),
        "size_bytes": path.stat(follow_symlinks=False).st_size,
        "sha256": sha256_file(path),
    }


def implementation_identity() -> dict[str, object]:
    """Bind the builder and every semantic/strict-verifier implementation."""

    repository_root = Path(__file__).resolve().parents[1]
    relatives = (
        "scripts/build_clean_sft500k_corpus.py",
        "src/clasp/data/benchmark.py",
        "src/clasp/data/loaders.py",
        "src/clasp/entrypoints/evaluate.py",
        "src/clasp/rewards/co_tasks.py",
    )
    files = []
    for relative in relatives:
        identity = file_identity(repository_root / relative)
        identity["relative_path"] = relative
        files.append(identity)
    compact = [
        {
            "relative_path": record["relative_path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in files
    ]
    return {
        "files": files,
        "implementation_manifest_sha256": sha256_json(compact),
    }


def iter_json_array(
    path: Path, *, chunk_size: int = 1024 * 1024
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield objects from a top-level JSON array using bounded memory."""

    if chunk_size < 64:
        raise CleanSFTError("JSON chunk size must be at least 64 bytes")
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        refill()
        while not eof and not buffer.strip():
            refill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise CleanSFTError(f"dataset is not a top-level JSON array: {path}")
        position += 1
        index = 0
        need_comma = False
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise CleanSFTError(f"unterminated JSON array: {path}")
                refill()
            if need_comma:
                if buffer[position] == "]":
                    position += 1
                    break
                if buffer[position] != ",":
                    raise CleanSFTError(f"expected comma before row {index}: {path}")
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        break
                    if eof:
                        raise CleanSFTError(f"unterminated JSON array: {path}")
                    refill()
                if buffer[position] == "]":
                    raise CleanSFTError(f"trailing comma in JSON array: {path}")
            elif buffer[position] == "]":
                position += 1
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise CleanSFTError(f"malformed JSON in {path}: {exc}") from exc
                    refill()
            if not isinstance(value, dict):
                raise CleanSFTError(f"row {index} is not an object: {path}")
            yield index, value
            index += 1
            need_comma = True
            if position > chunk_size:
                refill()
        trailing = buffer[position:] + handle.read()
        if trailing.strip():
            raise CleanSFTError(f"data follows the top-level JSON array: {path}")


@contextlib.contextmanager
def fixed_representation_environment() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in REPRESENTATION_ENVIRONMENT}
    try:
        os.environ.update(REPRESENTATION_ENVIRONMENT)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _coerce_instance(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _required_text(row: Mapping[str, Any], field_name: str) -> str:
    value = str(row.get(field_name, ""))
    if not value.strip():
        raise CleanSFTError(f"empty required field {field_name!r}")
    return value


def default_transform(task: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact task transform used by specialist SFT."""

    # The shared OP transform is deliberately representation-neutral: unlike
    # TSP and MIS it returns the four required fields unchanged.  Keep that
    # exact behavior here without importing ``clasp.data.loaders``, whose
    # module-level training dependencies include datasets and torch.  On Jean
    # Zay those packages live on Lustre, and importing them after a completed
    # solve can block the shard in client I/O before strict replay is written.
    if task == "op":
        return {
            "instruction": _required_text(row, "instruction"),
            "input": _required_text(row, "input"),
            "output": _required_text(row, "output"),
            "instance_json": _coerce_instance(
                row.get("instance_json", row.get("instance"))
            ),
        }

    from clasp.data.loaders import transform_example_for_task

    transformed = transform_example_for_task(
        task,
        _required_text(row, "instruction"),
        _required_text(row, "input"),
        _required_text(row, "output"),
        row.get("instance_json", row.get("instance")),
    )
    return {
        "instruction": str(transformed["instruction"]),
        "input": str(transformed["input"]),
        "output": str(transformed["output"]),
        "instance_json": _coerce_instance(transformed.get("instance_json")),
    }


def default_forbidden_transform(
    _task: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Project a reserved instance without requiring a reference output.

    The frozen IID confirmation tree is intentionally unlabeled.  Instance
    leakage checks therefore cannot route reserved rows through the supervised
    training transform, which requires a completion.  All active input-side
    representation switches are disabled by ``REPRESENTATION_ENVIRONMENT``.
    """

    return {
        "instruction": _required_text(row, "instruction"),
        "input": _required_text(row, "input"),
        "instance_json": _coerce_instance(row.get("instance_json", row.get("instance"))),
    }


def demonstration_sha256(transformed: Mapping[str, Any]) -> str:
    """Hash exactly the instruction/input/completion seen by SFT."""

    return sha256_json(
        {
            "instruction": str(transformed["instruction"]),
            "input": str(transformed["input"]),
            "output": str(transformed["output"]),
        }
    )


def default_semantic_hash(task: str, transformed: Mapping[str, Any]) -> str:
    from clasp.data.benchmark import semantic_instance_sha256

    return semantic_instance_sha256(task, dict(transformed), require_structured=False)


def _default_op_evaluator(
    prompt: str,
    input_text: str,
    answer: str,
    instance: Any,
    completion: str,
) -> dict[str, Any]:
    """Dependency-light equivalent of ``entrypoints.evaluate._evaluate_op``.

    The parsing, instance recovery, objective, and constraint implementations
    remain the shared reward functions.  Only the heavyweight evaluation CLI
    module (which imports PEFT and Transformers at module load) is bypassed.
    """

    from clasp.rewards.co_tasks import (
        _check_op_constraints,
        _combined_context,
        _op_meta,
        _op_prize,
        _parse_op_solution,
    )

    combined_context = _combined_context(prompt, input_text)
    parsed = _parse_op_solution(completion)
    reference = _parse_op_solution(answer)
    coordinates, prizes, route_limit = _op_meta(instance, combined_context)
    predicted_value = _op_prize(parsed["route"], prizes) if prizes else None
    reference_value = _op_prize(reference["route"], prizes) if prizes else None
    if coordinates and route_limit is not None:
        passed, total, feasible = _check_op_constraints(
            parsed["route"], coordinates, route_limit
        )
        constraint_fraction = passed / total
    else:
        feasible = False
        constraint_fraction = 0.0
    if (
        feasible
        and predicted_value is not None
        and reference_value is not None
        and reference_value > 0.0
    ):
        gap_to_ground_truth: float | None = float(
            max(0.0, (reference_value - predicted_value) / reference_value)
        )
    else:
        gap_to_ground_truth = None
    return {
        "parsed_solution": parsed["route"],
        "declared_objective": parsed["declared_objective"],
        "objective_value": predicted_value,
        "ground_truth_objective": reference_value,
        "is_feasible": bool(feasible),
        "constraint_fraction": float(constraint_fraction),
        "gap_to_ground_truth": gap_to_ground_truth,
    }


def _declared_literal(output: str) -> tuple[float, str]:
    matches = list(_OBJECTIVE_PATTERN.finditer(output))
    if not matches:
        raise CleanSFTError("completion has no declared Objective/Makespan")
    literal = matches[-1].group(1)
    value = float(literal)
    if not math.isfinite(value):
        raise CleanSFTError("completion declares a non-finite objective")
    return value, literal


def _float_display_tolerance(literal: str) -> float:
    """Return half a printed unit plus a small raw-distance roundoff margin."""

    mantissa = literal.lower().split("e", 1)[0]
    exponent = int(literal.lower().split("e", 1)[1]) if "e" in literal.lower() else 0
    decimals = len(mantissa.partition(".")[2])
    printed_unit = 10.0 ** (exponent - decimals)
    return (0.5000001 * printed_unit) + 1e-8


def default_strict_verify(task: str, transformed: Mapping[str, Any]) -> dict[str, Any]:
    """Replay a label through the shared strict verifier and its objective semantics."""

    output = str(transformed["output"])
    if task == "op":
        metrics = _default_op_evaluator(
            str(transformed["instruction"]),
            str(transformed["input"]),
            output,
            transformed.get("instance_json"),
            output,
        )
    else:
        from clasp.entrypoints.evaluate import TASK_EVALUATORS

        metrics = TASK_EVALUATORS[task](
            str(transformed["instruction"]),
            str(transformed["input"]),
            output,
            transformed.get("instance_json"),
            output,
        )
    if metrics.get("is_feasible") is not True:
        raise CleanSFTError("shared strict verifier rejected the label")
    constraint_fraction = float(metrics.get("constraint_fraction", 0.0))
    if not math.isfinite(constraint_fraction) or not math.isclose(
        constraint_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CleanSFTError(
            f"strict verifier did not pass every constraint: {constraint_fraction}"
        )
    recomputed_raw = metrics.get("objective_value")
    if isinstance(recomputed_raw, bool):
        raise CleanSFTError("strict verifier returned a boolean objective")
    try:
        recomputed = float(recomputed_raw)
    except (TypeError, ValueError) as exc:
        raise CleanSFTError("strict verifier returned no finite objective") from exc
    if not math.isfinite(recomputed):
        raise CleanSFTError("strict verifier returned no finite objective")
    declared, literal = _declared_literal(output)
    error = abs(declared - recomputed)
    if task in INTEGER_OBJECTIVE_TASKS:
        if abs(declared - round(declared)) > 1e-9 or error > 1e-9:
            raise CleanSFTError(
                f"declared integer objective {declared} != recomputed {recomputed}"
            )
        tolerance = 1e-9
    elif task in FLOAT_OBJECTIVE_TASKS:
        tolerance = _float_display_tolerance(literal)
        if error > tolerance:
            raise CleanSFTError(
                f"declared objective {declared} != recomputed {recomputed} "
                f"within printed tolerance {tolerance}"
            )
    else:  # pragma: no cover - guarded by CLI and direct caller checks
        raise CleanSFTError(f"unsupported task: {task}")
    return {
        "declared_objective": declared,
        "recomputed_objective": recomputed,
        "absolute_objective_error": error,
        "declared_objective_tolerance": tolerance,
        "constraint_fraction": constraint_fraction,
    }


def default_jssp_terminal_relabel(
    transformed: Mapping[str, Any], *, original_error: Exception
) -> tuple[dict[str, Any], dict[str, object]]:
    """Repair only a stale terminal JSSP makespan on a fully feasible schedule."""

    if not isinstance(original_error, CleanSFTError) or not str(original_error).startswith(
        "declared integer objective "
    ):
        raise CleanSFTError("JSSP failure is not exclusively an objective mismatch")
    from clasp.entrypoints.evaluate import TASK_EVALUATORS

    output = str(transformed["output"])
    terminal = _JSSP_TERMINAL_MAKESPAN_PATTERN.search(output)
    objective_matches = list(_OBJECTIVE_PATTERN.finditer(output))
    if (
        terminal is None
        or not objective_matches
        or objective_matches[-1].span(1) != terminal.span(2)
    ):
        raise CleanSFTError("JSSP output has no unique terminal Makespan numeric span")
    metrics = TASK_EVALUATORS["jssp"](
        str(transformed["instruction"]),
        str(transformed["input"]),
        output,
        transformed.get("instance_json"),
        output,
    )
    if metrics.get("is_feasible") is not True:
        raise CleanSFTError("JSSP schedule is not strictly feasible")
    constraint_fraction = float(metrics.get("constraint_fraction", 0.0))
    if not math.isfinite(constraint_fraction) or not math.isclose(
        constraint_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CleanSFTError("JSSP schedule does not satisfy every constraint")
    recomputed_raw = metrics.get("objective_value")
    if isinstance(recomputed_raw, bool):
        raise CleanSFTError("JSSP evaluator returned a boolean makespan")
    try:
        recomputed = float(recomputed_raw)
    except (TypeError, ValueError) as exc:
        raise CleanSFTError("JSSP evaluator returned no finite makespan") from exc
    if not math.isfinite(recomputed) or abs(recomputed - round(recomputed)) > 1e-9:
        raise CleanSFTError("JSSP evaluator returned a noninteger makespan")
    old_declared, _old_literal = _declared_literal(output)
    if (
        abs(old_declared - round(old_declared)) > 1e-9
        or math.isclose(old_declared, recomputed, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise CleanSFTError("JSSP terminal makespan is not a stale integer")
    new_literal = str(int(round(recomputed)))
    repaired_output = (
        output[: terminal.start(2)] + new_literal + output[terminal.end(2) :]
    )
    repaired = dict(transformed)
    repaired["output"] = repaired_output
    strict = default_strict_verify("jssp", repaired)
    evidence: dict[str, object] = {
        "old_makespan": int(round(old_declared)),
        "new_makespan": int(round(recomputed)),
        "terminal_numeric_span": [terminal.start(2), terminal.end(2)],
        "before_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "after_output_sha256": hashlib.sha256(
            repaired_output.encode("utf-8")
        ).hexdigest(),
        "before_demonstration_sha256": demonstration_sha256(transformed),
        "after_demonstration_sha256": demonstration_sha256(repaired),
        "strict_replay": strict,
    }
    return repaired, evidence


@dataclass
class AuditState:
    task: str
    source_rows: int = 0
    accepted_source_rows: int = 0
    replacement_rows: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    semantic_seen: set[str] = field(default_factory=set)
    demonstration_seen: set[str] = field(default_factory=set)
    retained_semantic_counts: Counter[str] = field(default_factory=Counter)
    accepted_semantic_digest: Any = field(default_factory=hashlib.sha256)
    accepted_demonstration_digest: Any = field(default_factory=hashlib.sha256)
    max_objective_error: float = 0.0
    first_rejections: list[dict[str, object]] = field(default_factory=list)
    replacement_rows_considered: int = 0
    replacement_skip_counts: Counter[str] = field(default_factory=Counter)
    selected_replacement_digest: Any = field(default_factory=hashlib.sha256)
    source_repairs: list[dict[str, object]] = field(default_factory=list)
    source_repair_digest: Any = field(default_factory=hashlib.sha256)

    def record_accept(self, semantic_hash: str, demonstration_hash: str) -> None:
        self.semantic_seen.add(semantic_hash)
        self.demonstration_seen.add(demonstration_hash)
        self.retained_semantic_counts[semantic_hash] += 1
        self.accepted_semantic_digest.update((semantic_hash + "\n").encode("ascii"))
        self.accepted_demonstration_digest.update(
            (demonstration_hash + "\n").encode("ascii")
        )

    def record_rejection(self, index: int, reasons: Sequence[str]) -> None:
        self.reason_counts.update(reasons)
        self.reason_counts["rejected_union"] += 1
        if len(self.first_rejections) < 32:
            self.first_rejections.append({"row_index": index, "reasons": list(reasons)})

    def record_source_repair(self, index: int, evidence: Mapping[str, object]) -> None:
        record = {"source_row_index": index, **dict(evidence)}
        self.source_repairs.append(record)
        self.source_repair_digest.update(canonical_json_bytes(record) + b"\n")


Transform = Callable[[str, Mapping[str, Any]], dict[str, Any]]
SemanticHasher = Callable[[str, Mapping[str, Any]], str]
StrictVerifier = Callable[[str, Mapping[str, Any]], dict[str, Any]]
JSSPRelabeler = Callable[..., tuple[dict[str, Any], dict[str, object]]]


def load_forbidden_hashes(
    task: str,
    sources: Sequence[tuple[str, Path]],
    *,
    transform: Transform = default_forbidden_transform,
    semantic_hasher: SemanticHasher = default_semantic_hash,
) -> tuple[set[str], list[dict[str, object]]]:
    hashes: set[str] = set()
    identities: list[dict[str, object]] = []
    for source_class, path in sources:
        rows = 0
        before = len(hashes)
        for _index, row in iter_json_array(path):
            transformed = transform(task, row)
            hashes.add(semantic_hasher(task, transformed))
            rows += 1
        identities.append(
            {
                "source_class": source_class,
                **file_identity(path),
                "rows": rows,
                "new_unique_semantic_instances": len(hashes) - before,
            }
        )
    return hashes, identities


def _ordered_reserved_sources(
    *,
    reserved_sources: Sequence[tuple[str, Path]],
    legacy_forbidden_paths: Sequence[Path],
    publication: bool,
) -> list[tuple[str, Path]]:
    """Normalize source labels and fail closed for a publication corpus build."""

    if reserved_sources and legacy_forbidden_paths:
        raise CleanSFTError(
            "cannot combine labeled reserved sources with legacy forbidden paths"
        )
    if reserved_sources:
        by_class: dict[str, Path] = {}
        for source_class, path in reserved_sources:
            if source_class not in RESERVED_SOURCE_CLASSES:
                raise CleanSFTError(f"unknown reserved source class: {source_class!r}")
            if source_class in by_class:
                raise CleanSFTError(
                    f"duplicate reserved source class: {source_class!r}"
                )
            by_class[source_class] = path
        if publication and set(by_class) != set(RESERVED_SOURCE_CLASSES):
            missing = sorted(set(RESERVED_SOURCE_CLASSES) - set(by_class))
            extra = sorted(set(by_class) - set(RESERVED_SOURCE_CLASSES))
            raise CleanSFTError(
                f"publication build requires exactly {RESERVED_SOURCE_CLASSES}; "
                f"missing={missing}, extra={extra}"
            )
        ordered = [
            (source_class, by_class[source_class])
            for source_class in RESERVED_SOURCE_CLASSES
            if source_class in by_class
        ]
    else:
        if publication:
            raise CleanSFTError(
                "publication build requires four labeled reserved sources: "
                + ", ".join(RESERVED_SOURCE_CLASSES)
            )
        ordered = [
            (f"legacy_unlabeled_{index}", path)
            for index, path in enumerate(legacy_forbidden_paths)
        ]
    canonical_paths = [str(path.resolve(strict=True)) for _label, path in ordered]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise CleanSFTError("reserved source paths must be four distinct canonical files")
    return ordered


def _compact_reserved_source_manifest(
    identities: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "source_class": identity["source_class"],
            "canonical_path": identity["canonical_path"],
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
            "rows": identity["rows"],
        }
        for identity in identities
    ]


def _load_self_hashed_receipt(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CleanSFTError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanSFTError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CleanSFTError(f"{label} is not a JSON object: {path}")
    stored = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    observed = sha256_json(unsigned)
    if stored != observed:
        raise CleanSFTError(
            f"{label} self-hash mismatch: stored={stored!r}, observed={observed}"
        )
    return value


def _validate_relative_file_record(
    root: Path, value: object, *, label: str
) -> tuple[Path, dict[str, object]]:
    expected_keys = {"relative_path", "size_bytes", "sha256"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CleanSFTError(f"{label} has a noncanonical file record")
    relative = Path(str(value["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CleanSFTError(f"{label} has an unsafe relative path")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CleanSFTError(f"{label} escapes its replacement shard") from exc
    live = file_identity(path)
    for field in ("size_bytes", "sha256"):
        if value[field] != live[field]:
            raise CleanSFTError(f"{label} live {field} mismatch")
    _require_read_only(path, label=label)
    return path, dict(value)


def _validate_producer_implementation(value: object) -> str:
    expected_keys = {"producer", "generator", "reference_runner", "strict_verifier"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CleanSFTError("replacement receipt has invalid producer implementation")
    for role in ("producer", "generator", "reference_runner"):
        record = value[role]
        if (
            not isinstance(record, dict)
            or set(record) != {"canonical_path", "size_bytes", "sha256"}
        ):
            raise CleanSFTError(f"replacement producer has invalid {role} identity")
        live = file_identity(Path(str(record["canonical_path"])))
        if any(record[field] != live[field] for field in live):
            raise CleanSFTError(f"replacement producer {role} source changed")
    strict = value["strict_verifier"]
    if not isinstance(strict, dict) or set(strict) != {
        "files",
        "implementation_manifest_sha256",
    }:
        raise CleanSFTError("replacement receipt has invalid strict-verifier identity")
    files = strict["files"]
    if not isinstance(files, list) or not files:
        raise CleanSFTError("replacement strict-verifier file manifest is empty")
    compact: list[dict[str, object]] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "canonical_path",
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise CleanSFTError("replacement strict-verifier file record is invalid")
        live = file_identity(Path(str(record["canonical_path"])))
        if any(record[field] != live[field] for field in live):
            raise CleanSFTError("replacement strict-verifier source changed")
        compact.append(
            {
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    if strict["implementation_manifest_sha256"] != sha256_json(compact):
        raise CleanSFTError("replacement strict-verifier manifest digest mismatch")
    return sha256_json(value)


def _replacement_provenance(
    task: str,
    replacement_paths: Sequence[Path],
    receipt_paths: Sequence[Path],
) -> list[dict[str, object]]:
    if len(replacement_paths) != len(receipt_paths):
        raise CleanSFTError(
            "every replacement JSON must have one ordered --replacement-receipt"
        )
    records: list[dict[str, object]] = []
    for replacement_path, receipt_path in zip(
        replacement_paths, receipt_paths, strict=True
    ):
        receipt = _load_self_hashed_receipt(receipt_path, label="replacement receipt")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("protocol") != REPLACEMENT_PROTOCOL
            or receipt.get("status")
            != "strict_solver_backed_replacement_shard_complete"
            or receipt.get("task") != task
        ):
            raise CleanSFTError("replacement receipt is not an official matching-task shard")
        _require_read_only(receipt_path, label="replacement receipt")
        shard_root = receipt_path.resolve(strict=True).parent
        output = receipt.get("output")
        replacement_identity = file_identity(replacement_path)
        output_path, _output_record = _validate_relative_file_record(
            shard_root, output, label="replacement output"
        )
        if output_path != replacement_path.resolve(strict=True):
            raise CleanSFTError("replacement receipt resolves to a different output path")
        for field in ("size_bytes", "sha256"):
            if output.get(field) != replacement_identity[field]:
                raise CleanSFTError(
                    f"replacement receipt does not bind {field}: {receipt_path}"
                )
        declared_rows = receipt.get("rows")
        if not isinstance(declared_rows, int) or declared_rows <= 0:
            raise CleanSFTError("replacement receipt has invalid row count")
        if receipt.get("unique_semantic_instances") != declared_rows:
            raise CleanSFTError("replacement receipt does not certify semantic uniqueness")
        observed_rows = sum(1 for _index, _row in iter_json_array(replacement_path))
        if observed_rows != declared_rows:
            raise CleanSFTError("replacement output row count disagrees with receipt")
        evidence_path, _evidence_record = _validate_relative_file_record(
            shard_root,
            receipt.get("evidence_manifest"),
            label="replacement evidence manifest",
        )
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CleanSFTError("replacement evidence manifest is unreadable") from exc
        if not isinstance(evidence, list) or len(evidence) != declared_rows:
            raise CleanSFTError("replacement evidence manifest has wrong row count")
        for expected_index, row_evidence in enumerate(evidence):
            if (
                not isinstance(row_evidence, dict)
                or row_evidence.get("row_index") != expected_index
                or not _is_sha256(row_evidence.get("semantic_instance_sha256"))
                or not _is_sha256(row_evidence.get("demonstration_sha256"))
                or not isinstance(row_evidence.get("native_status"), str)
                or not row_evidence["native_status"]
            ):
                raise CleanSFTError("replacement evidence row is malformed")
            _validate_relative_file_record(
                shard_root,
                row_evidence.get("strict_replay"),
                label=f"replacement strict replay row {expected_index}",
            )
        native_root = shard_root / "native_evidence"
        if native_root.is_symlink() or not native_root.is_dir():
            raise CleanSFTError("replacement native-evidence tree is missing")
        for path in (native_root, *native_root.rglob("*")):
            if path.is_symlink():
                raise CleanSFTError("replacement native-evidence tree contains a symlink")
            _require_read_only(path, label="replacement native evidence")
        solver_identity = receipt.get("solver_identity")
        solver_parameters = receipt.get("solver_parameters")
        if not isinstance(solver_identity, dict) or not solver_identity:
            raise CleanSFTError("replacement receipt has no solver identity")
        if not isinstance(solver_parameters, dict) or not solver_parameters:
            raise CleanSFTError("replacement receipt has no solver parameters")
        producer_implementation_sha256 = _validate_producer_implementation(
            receipt.get("implementation")
        )
        records.append(
            {
                "replacement": replacement_identity,
                "producer_receipt": file_identity(receipt_path),
                "producer_receipt_sha256": receipt["receipt_sha256"],
                "producer_protocol": REPLACEMENT_PROTOCOL,
                "task": task,
                "rows": declared_rows,
                "evidence_manifest": file_identity(evidence_path),
                "producer_implementation_sha256": producer_implementation_sha256,
                "solver_identity": solver_identity,
                "solver_parameters": solver_parameters,
            }
        )
    return records


def _write_array_row(handle: Any, row: Mapping[str, Any], *, first: bool) -> bool:
    if not first:
        handle.write(",\n")
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return False


def audit_or_build(
    *,
    task: str,
    source_path: Path,
    forbidden_paths: Sequence[Path] = (),
    reserved_sources: Sequence[tuple[str, Path]] = (),
    replacement_paths: Sequence[Path] = (),
    replacement_receipt_paths: Sequence[Path] = (),
    output_path: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
    transform: Transform = default_transform,
    forbidden_transform: Transform = default_forbidden_transform,
    semantic_hasher: SemanticHasher = default_semantic_hash,
    strict_verifier: StrictVerifier = default_strict_verify,
    repair_jssp_terminal_makespan: bool = False,
    jssp_relabeler: JSSPRelabeler = default_jssp_terminal_relabel,
) -> dict[str, Any]:
    """Audit *source_path* and optionally assemble an exact-size clean corpus."""

    if task not in TASKS:
        raise CleanSFTError(f"unsupported task: {task}")
    if expected_rows <= 0:
        raise CleanSFTError("expected_rows must be positive")
    if repair_jssp_terminal_makespan and task != "jssp":
        raise CleanSFTError("terminal-makespan relabeling is JSSP-only")
    ordered_reserved_sources = _ordered_reserved_sources(
        reserved_sources=reserved_sources,
        legacy_forbidden_paths=forbidden_paths,
        publication=output_path is not None,
    )
    with fixed_representation_environment():
        forbidden, forbidden_identities = load_forbidden_hashes(
            task,
            ordered_reserved_sources,
            transform=forbidden_transform,
            semantic_hasher=semantic_hasher,
        )
    reserved_source_manifest = _compact_reserved_source_manifest(forbidden_identities)
    if output_path is not None:
        if [item["source_class"] for item in reserved_source_manifest] != list(
            RESERVED_SOURCE_CLASSES
        ):
            raise CleanSFTError("publication reserved-source manifest is noncanonical")
        if any(int(item["rows"]) <= 0 for item in reserved_source_manifest):
            raise CleanSFTError("every publication reserved source must contain rows")
    replacement_provenance = _replacement_provenance(
        task, replacement_paths, replacement_receipt_paths
    )
    state = AuditState(task=task)
    temporary_path: Path | None = None
    output_handle: Any = contextlib.nullcontext(None)
    if output_path is not None:
        output_path = output_path.resolve()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite output corpus: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        output_handle = temporary_path.open("w", encoding="utf-8")

    clean_count = 0
    first_output = True
    try:
        with output_handle as handle, fixed_representation_environment():
            if handle is not None:
                handle.write("[\n")
            for index, row in iter_json_array(source_path):
                state.source_rows += 1
                reasons: list[str] = []
                row_for_output: Mapping[str, Any] = row
                repair_evidence_to_record: Mapping[str, object] | None = None
                try:
                    transformed = transform(task, row)
                    semantic_hash = semantic_hasher(task, transformed)
                except Exception as exc:
                    state.reason_counts[f"identity_error:{type(exc).__name__}"] += 1
                    state.record_rejection(index, ["identity_error"])
                    continue

                if semantic_hash in forbidden:
                    reasons.append("reserved_evaluation_collision")
                try:
                    strict = strict_verifier(task, transformed)
                    state.max_objective_error = max(
                        state.max_objective_error,
                        float(strict.get("absolute_objective_error", 0.0)),
                    )
                except Exception as exc:
                    repaired = False
                    if repair_jssp_terminal_makespan and not reasons:
                        try:
                            repaired_transformed, repair_evidence = jssp_relabeler(
                                transformed, original_error=exc
                            )
                            repaired_semantic_hash = semantic_hasher(
                                task, repaired_transformed
                            )
                            if repaired_semantic_hash != semantic_hash:
                                raise CleanSFTError(
                                    "JSSP terminal relabel changed semantic identity"
                                )
                            transformed = repaired_transformed
                            row_for_output = dict(row)
                            row_for_output["output"] = transformed["output"]
                            strict = dict(repair_evidence["strict_replay"])
                            repair_evidence_to_record = repair_evidence
                            state.max_objective_error = max(
                                state.max_objective_error,
                                float(strict.get("absolute_objective_error", 0.0)),
                            )
                            repaired = True
                        except Exception as repair_exc:
                            state.reason_counts[
                                f"jssp_relabel_error:{type(repair_exc).__name__}"
                            ] += 1
                    if not repaired:
                        reasons.append("strict_label_rejection")
                        state.reason_counts[f"strict_error:{type(exc).__name__}"] += 1

                try:
                    demonstration_hash = demonstration_sha256(transformed)
                except Exception as exc:
                    state.reason_counts[f"identity_error:{type(exc).__name__}"] += 1
                    reasons.append("identity_error")
                    demonstration_hash = ""
                if demonstration_hash in state.demonstration_seen:
                    reasons.append("exact_demonstration_duplicate")

                if reasons:
                    state.record_rejection(index, reasons)
                    continue
                if semantic_hash in state.semantic_seen:
                    state.reason_counts["retained_alternative_label_semantic_repeat"] += 1
                if repair_evidence_to_record is not None:
                    state.record_source_repair(index, repair_evidence_to_record)
                state.record_accept(semantic_hash, demonstration_hash)
                state.accepted_source_rows += 1
                clean_count += 1
                if handle is not None:
                    first_output = _write_array_row(
                        handle, row_for_output, first=first_output
                    )

            for replacement_path in replacement_paths:
                for replacement_index, row in iter_json_array(replacement_path):
                    if clean_count >= expected_rows:
                        break
                    state.replacement_rows_considered += 1
                    try:
                        transformed = transform(task, row)
                        semantic_hash = semantic_hasher(task, transformed)
                        demonstration_hash = demonstration_sha256(transformed)
                    except Exception as exc:
                        state.replacement_skip_counts[
                            f"identity_error:{type(exc).__name__}"
                        ] += 1
                        continue
                    if semantic_hash in forbidden:
                        state.replacement_skip_counts["reserved_source_collision"] += 1
                        continue
                    if semantic_hash in state.semantic_seen:
                        state.replacement_skip_counts["retained_semantic_collision"] += 1
                        continue
                    if demonstration_hash in state.demonstration_seen:
                        state.replacement_skip_counts[
                            "retained_demonstration_collision"
                        ] += 1
                        continue
                    try:
                        strict = strict_verifier(task, transformed)
                    except Exception as exc:
                        state.replacement_skip_counts[
                            f"strict_error:{type(exc).__name__}"
                        ] += 1
                        continue
                    state.max_objective_error = max(
                        state.max_objective_error,
                        float(strict.get("absolute_objective_error", 0.0)),
                    )
                    state.record_accept(semantic_hash, demonstration_hash)
                    state.replacement_rows += 1
                    state.selected_replacement_digest.update(
                        (
                            f"{replacement_path.resolve(strict=True)}:"
                            f"{replacement_index}:{semantic_hash}:{demonstration_hash}\n"
                        ).encode("utf-8")
                    )
                    clean_count += 1
                    if handle is not None:
                        first_output = _write_array_row(handle, row, first=first_output)
                if clean_count >= expected_rows:
                    break
            if handle is not None:
                handle.write("\n]\n")
                handle.flush()
                os.fsync(handle.fileno())
        if state.source_rows != expected_rows:
            raise CleanSFTError(
                f"source has {state.source_rows} rows; expected exactly {expected_rows}"
            )
        if output_path is not None and clean_count != expected_rows:
            raise CleanSFTError(
                f"clean corpus has {clean_count} rows; expected exactly {expected_rows}; "
                f"provide {expected_rows - clean_count} additional replacements"
            )
        if temporary_path is not None and output_path is not None:
            # A hard-link publication is atomic and fails if a racing process
            # created the target.  Unlike os.replace it can never overwrite an
            # existing corpus.
            os.link(temporary_path, output_path)
            os.chmod(output_path, 0o440)
            temporary_path.unlink()
            directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    multi_reference_instances = sum(
        1 for count in state.retained_semantic_counts.values() if count > 1
    )
    extra_multi_reference_rows = sum(
        count - 1 for count in state.retained_semantic_counts.values() if count > 1
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": (
            "clean_exact_500k_corpus_built"
            if output_path is not None
            else "all_source_rows_audited"
        ),
        "task": task,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": file_identity(source_path),
        "reserved_sources": forbidden_identities,
        "reserved_source_manifest": reserved_source_manifest,
        "reserved_source_manifest_sha256": sha256_json(reserved_source_manifest),
        "reserved_source_unique_semantic_instances": len(forbidden),
        "replacement_provenance": replacement_provenance,
        "expected_rows": expected_rows,
        "source_rows": state.source_rows,
        "accepted_source_rows": state.accepted_source_rows,
        "replacement_rows": state.replacement_rows,
        "replacement_rows_considered": state.replacement_rows_considered,
        "replacement_skip_counts": dict(sorted(state.replacement_skip_counts.items())),
        "selected_replacement_rows_sha256": state.selected_replacement_digest.hexdigest(),
        "source_repairs": {
            "protocol": JSSP_RELABEL_PROTOCOL,
            "enabled": repair_jssp_terminal_makespan,
            "task_scope": "jssp_only",
            "repair_scope": "terminal_makespan_numeric_span_only",
            "repaired_rows": len(state.source_repairs),
            "ordered_evidence_sha256": state.source_repair_digest.hexdigest(),
            "evidence": state.source_repairs,
        },
        "clean_rows": clean_count,
        "additional_replacements_required": max(0, expected_rows - clean_count),
        "reason_counts": dict(sorted(state.reason_counts.items())),
        "retained_unique_demonstrations": len(state.demonstration_seen),
        "retained_unique_semantic_instances": len(state.semantic_seen),
        "retained_multi_reference_instances": multi_reference_instances,
        "retained_extra_multi_reference_rows": extra_multi_reference_rows,
        "max_declared_objective_absolute_error": state.max_objective_error,
        "first_rejections": state.first_rejections,
        "ordered_clean_semantic_hashes_sha256": state.accepted_semantic_digest.hexdigest(),
        "ordered_clean_demonstration_hashes_sha256": (
            state.accepted_demonstration_digest.hexdigest()
        ),
        "representation_environment": dict(REPRESENTATION_ENVIRONMENT),
        "implementation": implementation_identity(),
        "strict_policy": {
            "shared_task_evaluator_feasible": True,
            "integer_objectives_exact": sorted(INTEGER_OBJECTIVE_TASKS),
            "float_objectives_match_displayed_precision": sorted(FLOAT_OBJECTIVE_TASKS),
        },
    }
    if output_path is not None:
        receipt["output"] = file_identity(output_path)
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o440)


def _require_read_only(path: Path, *, label: str) -> None:
    mode = path.stat(follow_symlinks=False).st_mode & 0o777
    if mode & 0o222:
        raise CleanSFTError(f"{label} must be read-only, observed mode {mode:04o}: {path}")


RESERVED_SOURCE_MANIFEST_KEYS = frozenset(
    {"source_class", "canonical_path", "size_bytes", "sha256", "rows"}
)


def validate_reserved_source_manifest(
    value: object, *, reopen: bool, deep: bool
) -> list[dict[str, object]]:
    """Validate the exact four-class reservation boundary.

    With ``reopen=True`` the path, size, and content hash are checked against
    live files.  ``deep=True`` additionally streams each JSON array to verify
    its recorded row count.
    """

    if not isinstance(value, list) or len(value) != len(RESERVED_SOURCE_CLASSES):
        raise CleanSFTError("reserved-source manifest must contain exactly four records")
    checked: list[dict[str, object]] = []
    for expected_class, raw in zip(RESERVED_SOURCE_CLASSES, value, strict=True):
        if not isinstance(raw, dict) or set(raw) != RESERVED_SOURCE_MANIFEST_KEYS:
            raise CleanSFTError("reserved-source manifest has a noncanonical record")
        if raw.get("source_class") != expected_class:
            raise CleanSFTError("reserved-source manifest class order is noncanonical")
        canonical_path = raw.get("canonical_path")
        if not isinstance(canonical_path, str) or not canonical_path.startswith("/"):
            raise CleanSFTError("reserved-source manifest path is not canonical absolute")
        if not isinstance(raw.get("size_bytes"), int) or int(raw["size_bytes"]) <= 0:
            raise CleanSFTError("reserved-source manifest has invalid size")
        if not _is_sha256(raw.get("sha256")):
            raise CleanSFTError("reserved-source manifest has invalid SHA256")
        if not isinstance(raw.get("rows"), int) or int(raw["rows"]) <= 0:
            raise CleanSFTError("reserved-source manifest must bind a nonempty JSON array")
        record = dict(raw)
        if reopen:
            path = Path(canonical_path)
            live = file_identity(path)
            for field in ("canonical_path", "size_bytes", "sha256"):
                if record[field] != live[field]:
                    raise CleanSFTError(
                        f"reserved-source live identity mismatch for {expected_class}:{field}"
                    )
            if deep:
                rows = sum(1 for _index, _row in iter_json_array(path))
                if rows != record["rows"]:
                    raise CleanSFTError(
                        f"reserved-source row-count mismatch for {expected_class}"
                    )
        checked.append(record)
    paths = [str(record["canonical_path"]) for record in checked]
    if len(set(paths)) != len(paths):
        raise CleanSFTError("reserved-source manifest paths are not unique")
    return checked


SOURCE_REPAIR_KEYS = frozenset(
    {
        "protocol",
        "enabled",
        "task_scope",
        "repair_scope",
        "repaired_rows",
        "ordered_evidence_sha256",
        "evidence",
    }
)
SOURCE_REPAIR_EVIDENCE_KEYS = frozenset(
    {
        "source_row_index",
        "old_makespan",
        "new_makespan",
        "terminal_numeric_span",
        "before_output_sha256",
        "after_output_sha256",
        "before_demonstration_sha256",
        "after_demonstration_sha256",
        "strict_replay",
    }
)


def validate_source_repairs(
    value: object, *, task: str, source_rows: int
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != SOURCE_REPAIR_KEYS:
        raise CleanSFTError("clean SFT source-repair evidence has a noncanonical schema")
    enabled = value.get("enabled")
    if (
        value.get("protocol") != JSSP_RELABEL_PROTOCOL
        or type(enabled) is not bool
        or value.get("task_scope") != "jssp_only"
        or value.get("repair_scope") != "terminal_makespan_numeric_span_only"
    ):
        raise CleanSFTError("clean SFT source-repair policy is invalid")
    evidence = value.get("evidence")
    repaired_rows = value.get("repaired_rows")
    if (
        not isinstance(evidence, list)
        or type(repaired_rows) is not int
        or repaired_rows < 0
        or repaired_rows != len(evidence)
    ):
        raise CleanSFTError("clean SFT source-repair count is invalid")
    if (task != "jssp" or not enabled) and repaired_rows != 0:
        raise CleanSFTError("source repairs are present outside the enabled JSSP lane")
    digest = hashlib.sha256()
    previous_index = -1
    for record in evidence:
        if not isinstance(record, dict) or set(record) != SOURCE_REPAIR_EVIDENCE_KEYS:
            raise CleanSFTError("clean SFT source-repair evidence row is invalid")
        index = record.get("source_row_index")
        if type(index) is not int or not previous_index < index < source_rows:
            raise CleanSFTError("source-repair row indices are not strictly increasing")
        previous_index = index
        old_makespan = record.get("old_makespan")
        new_makespan = record.get("new_makespan")
        if (
            type(old_makespan) is not int
            or type(new_makespan) is not int
            or old_makespan == new_makespan
        ):
            raise CleanSFTError("source-repair makespan pair is invalid")
        span = record.get("terminal_numeric_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(item) is not int for item in span)
            or not 0 <= span[0] < span[1]
        ):
            raise CleanSFTError("source-repair terminal numeric span is invalid")
        for field in (
            "before_output_sha256",
            "after_output_sha256",
            "before_demonstration_sha256",
            "after_demonstration_sha256",
        ):
            if not _is_sha256(record.get(field)):
                raise CleanSFTError(f"source-repair evidence has invalid {field}")
        if (
            record["before_output_sha256"] == record["after_output_sha256"]
            or record["before_demonstration_sha256"]
            == record["after_demonstration_sha256"]
        ):
            raise CleanSFTError("source-repair before/after hashes did not change")
        strict = record.get("strict_replay")
        if (
            not isinstance(strict, dict)
            or strict.get("constraint_fraction") != 1.0
            or strict.get("declared_objective") != float(new_makespan)
            or strict.get("recomputed_objective") != float(new_makespan)
            or strict.get("absolute_objective_error") != 0.0
        ):
            raise CleanSFTError("source-repair strict replay is invalid")
        digest.update(canonical_json_bytes(record) + b"\n")
    if value.get("ordered_evidence_sha256") != digest.hexdigest():
        raise CleanSFTError("source-repair ordered evidence digest mismatch")
    return dict(value)


def deep_validate_jssp_source_repairs(
    source_repairs: Mapping[str, object], *, source_path: Path, output_path: Path
) -> None:
    """Reopen source/output together and reproduce every claimed JSSP repair."""

    evidence = source_repairs.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return
    evidence_by_index = {
        int(record["source_row_index"]): record
        for record in evidence
        if isinstance(record, dict)
    }
    if len(evidence_by_index) != len(evidence):
        raise CleanSFTError("JSSP repair evidence has duplicate source indices")
    observed_repairs = 0
    missing = object()
    with fixed_representation_environment():
        for source_item, output_item in zip_longest(
            iter_json_array(source_path),
            iter_json_array(output_path),
            fillvalue=missing,
        ):
            if source_item is missing or output_item is missing:
                raise CleanSFTError("JSSP source/output row counts differ")
            assert isinstance(source_item, tuple) and isinstance(output_item, tuple)
            source_index, source_row = source_item
            output_index, output_row = output_item
            if source_index != output_index:
                raise CleanSFTError("JSSP source/output row indices diverged")
            record = evidence_by_index.get(source_index)
            if record is None:
                if source_row != output_row:
                    raise CleanSFTError(
                        f"unrecorded JSSP source mutation at row {source_index}"
                    )
                continue
            observed_repairs += 1
            source_nonoutput = dict(source_row)
            output_nonoutput = dict(output_row)
            source_output = source_nonoutput.pop("output", None)
            repaired_output = output_nonoutput.pop("output", None)
            if source_nonoutput != output_nonoutput:
                raise CleanSFTError(
                    f"JSSP repair changed a non-output field at row {source_index}"
                )
            if not isinstance(source_output, str) or not isinstance(
                repaired_output, str
            ):
                raise CleanSFTError("JSSP repair row has a non-text output")
            if hashlib.sha256(source_output.encode("utf-8")).hexdigest() != record.get(
                "before_output_sha256"
            ):
                raise CleanSFTError("JSSP repair before-output hash mismatch")
            if hashlib.sha256(
                repaired_output.encode("utf-8")
            ).hexdigest() != record.get("after_output_sha256"):
                raise CleanSFTError("JSSP repair after-output hash mismatch")
            source_transformed = default_transform("jssp", source_row)
            output_transformed = default_transform("jssp", output_row)
            if default_semantic_hash(
                "jssp", source_transformed
            ) != default_semantic_hash("jssp", output_transformed):
                raise CleanSFTError("JSSP repair changed semantic identity")
            if demonstration_sha256(source_transformed) != record.get(
                "before_demonstration_sha256"
            ):
                raise CleanSFTError("JSSP repair before-demonstration hash mismatch")
            if demonstration_sha256(output_transformed) != record.get(
                "after_demonstration_sha256"
            ):
                raise CleanSFTError("JSSP repair after-demonstration hash mismatch")
            try:
                default_strict_verify("jssp", source_transformed)
            except Exception as original_error:
                expected_transformed, expected_evidence = (
                    default_jssp_terminal_relabel(
                        source_transformed, original_error=original_error
                    )
                )
            else:
                raise CleanSFTError(
                    f"claimed JSSP repair row {source_index} was already strict-valid"
                )
            if expected_transformed != output_transformed:
                raise CleanSFTError(
                    f"JSSP repair output mismatch at row {source_index}"
                )
            expected_record = {"source_row_index": source_index, **expected_evidence}
            if expected_record != record:
                raise CleanSFTError(
                    f"JSSP repair evidence mismatch at row {source_index}"
                )
    if observed_repairs != len(evidence):
        raise CleanSFTError("not every JSSP repair evidence row was reopened")


def validate_task_receipt(
    receipt_path: Path,
    *,
    expected_task: str | None = None,
    expected_rows: int = EXPECTED_ROWS,
    deep: bool = True,
) -> dict[str, object]:
    """Dependency-light reopen validation and compact task identity."""

    receipt = _load_self_hashed_receipt(receipt_path, label="clean SFT task receipt")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise CleanSFTError("clean SFT task receipt has unsupported schema")
    if receipt.get("protocol") != PROTOCOL:
        raise CleanSFTError("clean SFT task receipt has unsupported protocol")
    if receipt.get("status") != "clean_exact_500k_corpus_built":
        raise CleanSFTError("clean SFT task receipt does not certify a built corpus")
    task = str(receipt.get("task", ""))
    if task not in TASKS or (expected_task is not None and task != expected_task):
        raise CleanSFTError(
            f"clean SFT task mismatch: expected={expected_task!r}, observed={task!r}"
        )
    for field in ("expected_rows", "source_rows", "clean_rows"):
        if receipt.get(field) != expected_rows:
            raise CleanSFTError(f"clean SFT receipt has invalid {field}: {receipt.get(field)!r}")
    if receipt.get("additional_replacements_required") != 0:
        raise CleanSFTError("clean SFT task receipt still requires replacements")
    if receipt.get("retained_unique_demonstrations") != expected_rows:
        raise CleanSFTError("clean SFT corpus does not contain unique demonstrations")
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise CleanSFTError("clean SFT task receipt has no source identity")
    observed_source = file_identity(Path(str(source.get("canonical_path", ""))))
    for field in ("canonical_path", "size_bytes", "sha256"):
        if source.get(field) != observed_source[field]:
            raise CleanSFTError(f"clean SFT source identity mismatch for {field}")
    output = receipt.get("output")
    if not isinstance(output, dict):
        raise CleanSFTError("clean SFT task receipt has no output identity")
    output_path = Path(str(output.get("canonical_path", "")))
    observed_output = file_identity(output_path)
    for field in ("canonical_path", "size_bytes", "sha256"):
        if output.get(field) != observed_output[field]:
            raise CleanSFTError(f"clean SFT output identity mismatch for {field}")
    _require_read_only(output_path, label="clean SFT task output")
    if deep:
        reopened_rows = sum(1 for _index, _row in iter_json_array(output_path))
        if reopened_rows != expected_rows:
            raise CleanSFTError(
                f"reopened clean SFT output has {reopened_rows} rows, expected {expected_rows}"
            )
    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        raise CleanSFTError("clean SFT task receipt has no implementation identity")
    manifest_sha = implementation.get("implementation_manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise CleanSFTError("clean SFT implementation manifest identity is invalid")
    replacement_provenance = receipt.get("replacement_provenance")
    if not isinstance(replacement_provenance, list):
        raise CleanSFTError("clean SFT task receipt has invalid replacement provenance")
    replacement_rows = receipt.get("replacement_rows")
    if not isinstance(replacement_rows, int) or replacement_rows < 0:
        raise CleanSFTError("clean SFT task receipt has invalid replacement_rows")
    if replacement_rows > 0 and not replacement_provenance:
        raise CleanSFTError("clean SFT task receipt has no replacement provenance")
    if replacement_rows == 0 and replacement_provenance:
        raise CleanSFTError("replacement provenance is present but no replacements were used")
    if replacement_provenance:
        try:
            live_replacement_provenance = _replacement_provenance(
                task,
                [
                    Path(str(record["replacement"]["canonical_path"]))
                    for record in replacement_provenance
                ],
                [
                    Path(str(record["producer_receipt"]["canonical_path"]))
                    for record in replacement_provenance
                ],
            )
        except (KeyError, TypeError) as exc:
            raise CleanSFTError("clean SFT replacement provenance is malformed") from exc
        if replacement_provenance != live_replacement_provenance:
            raise CleanSFTError("clean SFT replacement provenance changed on reopen")
    source_repairs = validate_source_repairs(
        receipt.get("source_repairs"), task=task, source_rows=expected_rows
    )
    reason_counts = receipt.get("reason_counts")
    if not isinstance(reason_counts, dict):
        raise CleanSFTError("clean SFT task receipt has invalid reason counts")
    if source_repairs["repaired_rows"] and (
        receipt.get("accepted_source_rows") != expected_rows
        or replacement_rows != 0
        or reason_counts.get("rejected_union", 0) != 0
    ):
        raise CleanSFTError(
            "JSSP relabel lane must recover every source row without replacements"
        )
    if deep and source_repairs["repaired_rows"]:
        deep_validate_jssp_source_repairs(
            source_repairs,
            source_path=Path(str(source["canonical_path"])),
            output_path=output_path,
        )
    reserved_source_manifest = validate_reserved_source_manifest(
        receipt.get("reserved_source_manifest"), reopen=True, deep=deep
    )
    reserved_manifest_sha = receipt.get("reserved_source_manifest_sha256")
    if reserved_manifest_sha != sha256_json(reserved_source_manifest):
        raise CleanSFTError("reserved-source manifest digest mismatch")
    live_implementation = implementation_identity()["implementation_manifest_sha256"]
    if manifest_sha != live_implementation:
        raise CleanSFTError(
            "clean SFT task receipt verifier implementation differs from the live source"
        )
    _require_read_only(receipt_path, label="clean SFT task receipt")
    receipt_identity = file_identity(receipt_path)
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "task": task,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_canonical_path": receipt_identity["canonical_path"],
        "receipt_file_sha256": receipt_identity["sha256"],
        "output_sha256": observed_output["sha256"],
        "output_size_bytes": observed_output["size_bytes"],
        "clean_rows": expected_rows,
        "unique_demonstrations": receipt["retained_unique_demonstrations"],
        "unique_semantic_instances": receipt["retained_unique_semantic_instances"],
        "reserved_source_manifest": reserved_source_manifest,
        "reserved_source_manifest_sha256": reserved_manifest_sha,
        "implementation_manifest_sha256": manifest_sha,
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


MERGED_PROTOCOL = "clasp-clean-specialist-sft500k-five-task-audit-v2"
MERGED_RECEIPT_FILENAME = "sft_cleanliness_audit.json"


TASK_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "task",
        "receipt_sha256",
        "receipt_canonical_path",
        "receipt_file_sha256",
        "output_sha256",
        "output_size_bytes",
        "clean_rows",
        "unique_demonstrations",
        "unique_semantic_instances",
        "reserved_source_manifest",
        "reserved_source_manifest_sha256",
        "implementation_manifest_sha256",
        "identity_sha256",
    }
)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_task_identity(value: object, *, expected_task: str | None = None) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != TASK_IDENTITY_KEYS:
        raise CleanSFTError("clean SFT task identity has a noncanonical key set")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("protocol") != PROTOCOL:
        raise CleanSFTError("clean SFT task identity has unsupported protocol")
    task = str(value.get("task", ""))
    if task not in TASKS or (expected_task is not None and task != expected_task):
        raise CleanSFTError("clean SFT task identity has an invalid task")
    receipt_path_value = value.get("receipt_canonical_path")
    if not isinstance(receipt_path_value, str):
        raise CleanSFTError("clean SFT task identity has no canonical receipt path")
    receipt_path = Path(receipt_path_value)
    if (
        not receipt_path.is_absolute()
        or str(receipt_path) != receipt_path_value
        or ".." in receipt_path.parts
        or receipt_path.name != f"{task}.json"
        or receipt_path.parent.name != "cleanliness_receipts"
    ):
        raise CleanSFTError("clean SFT task identity has an invalid receipt path")
    for field in (
        "receipt_sha256",
        "receipt_file_sha256",
        "output_sha256",
        "reserved_source_manifest_sha256",
        "implementation_manifest_sha256",
        "identity_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise CleanSFTError(f"clean SFT task identity has invalid {field}")
    if value.get("clean_rows") != EXPECTED_ROWS:
        raise CleanSFTError("clean SFT task identity has wrong clean_rows")
    if value.get("unique_demonstrations") != EXPECTED_ROWS:
        raise CleanSFTError("clean SFT task identity has duplicate demonstrations")
    semantic_count = value.get("unique_semantic_instances")
    if not isinstance(semantic_count, int) or not 0 < semantic_count <= EXPECTED_ROWS:
        raise CleanSFTError("clean SFT task identity has invalid semantic count")
    reserved_source_manifest = validate_reserved_source_manifest(
        value.get("reserved_source_manifest"), reopen=False, deep=False
    )
    if value.get("reserved_source_manifest_sha256") != sha256_json(
        reserved_source_manifest
    ):
        raise CleanSFTError("clean SFT task identity reserved-source digest mismatch")
    output_size = value.get("output_size_bytes")
    if not isinstance(output_size, int) or output_size <= 0:
        raise CleanSFTError("clean SFT task identity has invalid output size")
    unsigned = dict(value)
    stored = unsigned.pop("identity_sha256")
    if stored != sha256_json(unsigned):
        raise CleanSFTError("clean SFT task identity self-hash mismatch")
    return dict(value)


def _reopen_bound_task_receipt(
    task_identity: Mapping[str, object], *, data_root: Path
) -> None:
    """Require the detailed task receipt at its canonical data-root location."""

    task = str(task_identity["task"])
    expected_path = data_root / "cleanliness_receipts" / f"{task}.json"
    recorded_path = task_identity.get("receipt_canonical_path")
    if recorded_path != str(expected_path):
        raise CleanSFTError(
            f"{task} task receipt is not at its canonical data-root path"
        )
    live_identity = file_identity(expected_path)
    if live_identity["canonical_path"] != str(expected_path):
        raise CleanSFTError(f"{task} task receipt path resolves outside its canonical path")
    if live_identity["sha256"] != task_identity.get("receipt_file_sha256"):
        raise CleanSFTError(f"{task} task receipt file hash mismatch")
    live_receipt = _load_self_hashed_receipt(
        expected_path, label=f"{task} clean SFT task receipt"
    )
    if (
        live_receipt.get("schema_version") != SCHEMA_VERSION
        or live_receipt.get("protocol") != PROTOCOL
        or live_receipt.get("status") != "clean_exact_500k_corpus_built"
        or live_receipt.get("task") != task
        or live_receipt.get("receipt_sha256")
        != task_identity.get("receipt_sha256")
    ):
        raise CleanSFTError(f"{task} detailed task receipt identity mismatch")
    _require_read_only(expected_path, label=f"{task} clean SFT task receipt")


def merge_task_receipts(
    *,
    data_root: Path,
    task_receipt_paths: Sequence[Path],
    deep: bool = True,
) -> dict[str, object]:
    """Validate and merge the five task receipts into the campaign gate."""

    root = data_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise CleanSFTError(f"data root is not a regular directory: {data_root}")
    if len(task_receipt_paths) != len(TASKS):
        raise CleanSFTError("exactly five task receipts are required")
    task_identities: dict[str, object] = {}
    implementation_hashes: set[str] = set()
    for receipt_path in task_receipt_paths:
        raw = _load_self_hashed_receipt(receipt_path, label="clean SFT task receipt")
        task = str(raw.get("task", ""))
        if task not in TASKS or task in task_identities:
            raise CleanSFTError(f"duplicate or invalid task receipt: {task!r}")
        expected_receipt_path = root / "cleanliness_receipts" / f"{task}.json"
        observed_receipt_identity = file_identity(receipt_path)
        if observed_receipt_identity["canonical_path"] != str(expected_receipt_path):
            raise CleanSFTError(
                f"{task} task receipt must be exactly {expected_receipt_path}"
            )
        output = raw.get("output")
        if not isinstance(output, dict):
            raise CleanSFTError(f"{task} task receipt has no output")
        output_path = Path(str(output.get("canonical_path", ""))).resolve(strict=True)
        try:
            output_path.relative_to(root)
        except ValueError as exc:
            raise CleanSFTError(f"{task} output is outside frozen data root") from exc
        identity = validate_task_receipt(
            receipt_path,
            expected_task=task,
            expected_rows=EXPECTED_ROWS,
            deep=deep,
        )
        validate_task_identity(identity, expected_task=task)
        _reopen_bound_task_receipt(identity, data_root=root)
        task_identities[task] = identity
        implementation_hashes.add(str(identity["implementation_manifest_sha256"]))
    if set(task_identities) != set(TASKS):
        raise CleanSFTError("merged cleanliness receipt does not cover all five tasks")
    if len(implementation_hashes) != 1:
        raise CleanSFTError("task cleanliness receipts used different verifier implementations")
    merged: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": MERGED_PROTOCOL,
        "status": "all_five_clean_exact_500k_corpora_verified",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(root),
        "tasks": {task: task_identities[task] for task in TASKS},
        "rows_per_task": EXPECTED_ROWS,
        "total_rows": EXPECTED_ROWS * len(TASKS),
        "implementation_manifest_sha256": next(iter(implementation_hashes)),
    }
    merged["receipt_sha256"] = sha256_json(merged)
    return merged


def validate_merged_receipt(
    receipt_path: Path, *, expected_data_root: Path | None = None
) -> dict[str, object]:
    """Reopen a canonical five-task receipt without ML dependencies."""

    receipt = _load_self_hashed_receipt(receipt_path, label="merged SFT cleanliness receipt")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise CleanSFTError("merged SFT cleanliness receipt has unsupported schema")
    if receipt.get("protocol") != MERGED_PROTOCOL:
        raise CleanSFTError("merged SFT cleanliness receipt has unsupported protocol")
    if receipt.get("status") != "all_five_clean_exact_500k_corpora_verified":
        raise CleanSFTError("merged SFT cleanliness receipt is not complete")
    if receipt.get("rows_per_task") != EXPECTED_ROWS:
        raise CleanSFTError("merged SFT cleanliness receipt has wrong rows_per_task")
    if receipt.get("total_rows") != EXPECTED_ROWS * len(TASKS):
        raise CleanSFTError("merged SFT cleanliness receipt has wrong total_rows")
    data_root_value = receipt.get("data_root")
    if not isinstance(data_root_value, str) or not data_root_value.startswith("/"):
        raise CleanSFTError("merged SFT cleanliness receipt has no canonical data root")
    data_root = Path(data_root_value)
    if (
        data_root.is_symlink()
        or not data_root.is_dir()
        or data_root.resolve(strict=True) != data_root
    ):
        raise CleanSFTError("merged SFT cleanliness receipt data root is not canonical")
    tasks = receipt.get("tasks")
    if not isinstance(tasks, dict) or list(tasks) != list(TASKS):
        raise CleanSFTError("merged SFT cleanliness receipt has noncanonical tasks")
    checked_tasks = {
        task: validate_task_identity(tasks[task], expected_task=task) for task in TASKS
    }
    for task in TASKS:
        _reopen_bound_task_receipt(checked_tasks[task], data_root=data_root)
    task_implementation_hashes = {
        str(checked_tasks[task]["implementation_manifest_sha256"]) for task in TASKS
    }
    if len(task_implementation_hashes) != 1:
        raise CleanSFTError("merged SFT task identities use different implementations")
    receipt_implementation_hash = receipt.get("implementation_manifest_sha256")
    if receipt_implementation_hash != next(iter(task_implementation_hashes)):
        raise CleanSFTError("merged SFT implementation hash is not task-bound")
    live_implementation_hash = implementation_identity()["implementation_manifest_sha256"]
    if receipt_implementation_hash != live_implementation_hash:
        raise CleanSFTError("merged SFT verifier implementation differs from live source")
    if expected_data_root is not None:
        expected = str(expected_data_root.resolve(strict=True))
        if data_root_value != expected:
            raise CleanSFTError(
                f"merged SFT data root mismatch: expected={expected}, "
                f"observed={receipt.get('data_root')}"
            )
    _require_read_only(receipt_path, label="merged SFT cleanliness receipt")
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": MERGED_PROTOCOL,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": sha256_file(receipt_path),
        "data_root": receipt["data_root"],
        "rows_per_task": receipt["rows_per_task"],
        "total_rows": receipt["total_rows"],
        "implementation_manifest_sha256": receipt["implementation_manifest_sha256"],
        "task_identity_sha256": {
            task: checked_tasks[task]["identity_sha256"] for task in TASKS
        },
        "task_receipt_canonical_path": {
            task: checked_tasks[task]["receipt_canonical_path"] for task in TASKS
        },
        "task_output_sha256": {
            task: checked_tasks[task]["output_sha256"] for task in TASKS
        },
        "task_unique_demonstrations": {
            task: checked_tasks[task]["unique_demonstrations"] for task in TASKS
        },
        "task_unique_semantic_instances": {
            task: checked_tasks[task]["unique_semantic_instances"] for task in TASKS
        },
        "task_reserved_source_manifest": {
            task: checked_tasks[task]["reserved_source_manifest"] for task in TASKS
        },
        "task_reserved_source_manifest_sha256": {
            task: checked_tasks[task]["reserved_source_manifest_sha256"] for task in TASKS
        },
    }
    identity["identity_sha256"] = sha256_json(identity)
    return validate_merged_identity(identity)


MERGED_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "receipt_sha256",
        "receipt_file_sha256",
        "data_root",
        "rows_per_task",
        "total_rows",
        "implementation_manifest_sha256",
        "task_identity_sha256",
        "task_receipt_canonical_path",
        "task_output_sha256",
        "task_unique_demonstrations",
        "task_unique_semantic_instances",
        "task_reserved_source_manifest",
        "task_reserved_source_manifest_sha256",
        "identity_sha256",
    }
)


def validate_merged_identity(value: object) -> dict[str, object]:
    """Validate a portable merged identity without reopening external paths."""

    if not isinstance(value, dict) or set(value) != MERGED_IDENTITY_KEYS:
        raise CleanSFTError("merged SFT compact identity has a noncanonical key set")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("protocol") != MERGED_PROTOCOL:
        raise CleanSFTError("merged SFT compact identity has unsupported protocol")
    if value.get("rows_per_task") != EXPECTED_ROWS:
        raise CleanSFTError("merged SFT compact identity has wrong rows_per_task")
    if value.get("total_rows") != EXPECTED_ROWS * len(TASKS):
        raise CleanSFTError("merged SFT compact identity has wrong total_rows")
    data_root = value.get("data_root")
    if (
        not isinstance(data_root, str)
        or not data_root.startswith("/")
        or str(Path(data_root)) != data_root
        or ".." in Path(data_root).parts
    ):
        raise CleanSFTError("merged SFT compact identity has no canonical absolute data root")
    for field in (
        "receipt_sha256",
        "receipt_file_sha256",
        "implementation_manifest_sha256",
        "identity_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise CleanSFTError(f"merged SFT compact identity has invalid {field}")
    hash_maps = (
        "task_identity_sha256",
        "task_output_sha256",
        "task_reserved_source_manifest_sha256",
    )
    for field in hash_maps:
        mapping = value.get(field)
        if not isinstance(mapping, dict) or list(mapping) != list(TASKS):
            raise CleanSFTError(f"merged SFT compact identity has noncanonical {field}")
        if any(not _is_sha256(mapping[task]) for task in TASKS):
            raise CleanSFTError(f"merged SFT compact identity has invalid hashes in {field}")
    receipt_paths = value.get("task_receipt_canonical_path")
    if not isinstance(receipt_paths, dict) or list(receipt_paths) != list(TASKS):
        raise CleanSFTError(
            "merged SFT compact identity has noncanonical task receipt paths"
        )
    for task in TASKS:
        expected_receipt_path = str(
            Path(data_root) / "cleanliness_receipts" / f"{task}.json"
        )
        if receipt_paths[task] != expected_receipt_path:
            raise CleanSFTError(
                f"merged SFT compact identity has invalid {task} receipt path"
            )
    demonstrations = value.get("task_unique_demonstrations")
    semantics = value.get("task_unique_semantic_instances")
    if not isinstance(demonstrations, dict) or list(demonstrations) != list(TASKS):
        raise CleanSFTError("merged SFT compact identity has invalid demonstration counts")
    if any(demonstrations[task] != EXPECTED_ROWS for task in TASKS):
        raise CleanSFTError("merged SFT compact identity contains duplicate demonstrations")
    if not isinstance(semantics, dict) or list(semantics) != list(TASKS):
        raise CleanSFTError("merged SFT compact identity has invalid semantic counts")
    if any(
        not isinstance(semantics[task], int)
        or not 0 < semantics[task] <= EXPECTED_ROWS
        for task in TASKS
    ):
        raise CleanSFTError("merged SFT compact identity has out-of-range semantic counts")
    reserved_manifests = value.get("task_reserved_source_manifest")
    if not isinstance(reserved_manifests, dict) or list(reserved_manifests) != list(TASKS):
        raise CleanSFTError("merged SFT compact identity has invalid reserved manifests")
    reserved_digests = value["task_reserved_source_manifest_sha256"]
    assert isinstance(reserved_digests, dict)
    for task in TASKS:
        manifest = validate_reserved_source_manifest(
            reserved_manifests[task], reopen=False, deep=False
        )
        if reserved_digests[task] != sha256_json(manifest):
            raise CleanSFTError(
                f"merged SFT compact identity reserved digest mismatch for {task}"
            )
    unsigned = dict(value)
    stored = unsigned.pop("identity_sha256")
    if stored != sha256_json(unsigned):
        raise CleanSFTError("merged SFT compact identity self-hash mismatch")
    return dict(value)


def _parse_reserved_source_arg(value: str) -> tuple[str, Path]:
    source_class, separator, raw_path = value.partition(",")
    if not separator or source_class not in RESERVED_SOURCE_CLASSES or not raw_path:
        raise argparse.ArgumentTypeError(
            "reserved source must be CLASS,PATH with CLASS in "
            + ",".join(RESERVED_SOURCE_CLASSES)
        )
    return source_class, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument(
        "--reserved-eval-json",
        type=Path,
        action="append",
        default=[],
        help="Legacy unlabeled diagnostic-only reservation; forbidden for corpus builds.",
    )
    parser.add_argument(
        "--reserved-source",
        type=_parse_reserved_source_arg,
        action="append",
        default=[],
        metavar="CLASS,PATH",
        help=(
            "Publication reservation. Supply exactly once for sft_eval, rl_eval, "
            "rl_train, and iid_confirmation."
        ),
    )
    parser.add_argument(
        "--replacement-json",
        type=Path,
        action="append",
        default=[],
        help="Strictly valid, semantically new rows to append after source cleaning.",
    )
    parser.add_argument(
        "--replacement-receipt",
        type=Path,
        action="append",
        default=[],
        help="Repeat in the same order as --replacement-json.",
    )
    parser.add_argument(
        "--merge-task-receipt",
        type=Path,
        action="append",
        default=[],
        help="Merge five completed per-task receipts instead of auditing one task.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--repair-jssp-terminal-makespan",
        action="store_true",
        help=(
            "JSSP-only: repair a stale final Makespan integer after proving the "
            "unchanged schedule strictly feasible and replaying the repaired label."
        ),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args(argv)
    if args.merge_task_receipt:
        if args.data_root is None:
            parser.error("receipt merging requires --data-root")
        if (
            args.task is not None
            or args.source_json is not None
            or args.output_json is not None
            or args.repair_jssp_terminal_makespan
        ):
            parser.error("receipt merging cannot be combined with task audit/build flags")
        return args
    if args.task is None or args.source_json is None:
        parser.error("task audit/build requires --task and --source-json")
    if args.repair_jssp_terminal_makespan and args.task != "jssp":
        parser.error("--repair-jssp-terminal-makespan is valid only for --task jssp")
    if args.output_json is not None:
        if args.reserved_eval_json:
            parser.error("corpus builds require labeled --reserved-source, not legacy flags")
        labels = [source_class for source_class, _path in args.reserved_source]
        if len(labels) != len(RESERVED_SOURCE_CLASSES) or set(labels) != set(
            RESERVED_SOURCE_CLASSES
        ):
            parser.error(
                "corpus builds require exactly one --reserved-source for each of: "
                + ", ".join(RESERVED_SOURCE_CLASSES)
            )
    elif args.reserved_source and args.reserved_eval_json:
        parser.error("cannot combine labeled and legacy reserved sources")
    if len(args.replacement_json) != len(args.replacement_receipt):
        parser.error("every --replacement-json requires one ordered --replacement-receipt")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_task_receipt:
        receipt = merge_task_receipts(
            data_root=args.data_root,
            task_receipt_paths=args.merge_task_receipt,
        )
        _write_json_new(args.receipt, receipt)
        print(json.dumps(validate_merged_receipt(args.receipt), indent=2, sort_keys=True))
        return 0
    receipt = audit_or_build(
        task=args.task,
        source_path=args.source_json,
        forbidden_paths=args.reserved_eval_json,
        reserved_sources=args.reserved_source,
        replacement_paths=args.replacement_json,
        replacement_receipt_paths=args.replacement_receipt,
        output_path=args.output_json,
        expected_rows=args.expected_rows,
        repair_jssp_terminal_makespan=args.repair_jssp_terminal_makespan,
    )
    _write_json_new(args.receipt, receipt)
    if args.output_json is not None:
        identity = validate_task_receipt(
            args.receipt,
            expected_task=args.task,
            expected_rows=args.expected_rows,
        )
        print(json.dumps(identity, indent=2, sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
