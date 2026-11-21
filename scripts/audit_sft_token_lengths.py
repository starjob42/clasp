#!/usr/bin/env python3
"""Stream-audit every row in the frozen five-task Jiang-f638 SFT corpus.

The audit renders the released ``instruction``/``input``/``output`` columns
directly with Jiang et al.'s byte-exact formatter.  It then mirrors the f638
SFT tokenizer contract plus a deterministic correction to its broken response
boundary: the complete ``text`` is tokenized with special tokens and without
truncation, while the contextual ``    ### Response`` header (without its
colon) is tokenized separately without special tokens.  Supervision starts at
the next token, which preserves a token spanning the colon and output prefix.

Only bounded tokenization batches and a length histogram are retained in
memory.  The audit fails immediately if any of the 500,000 rows per task would
be truncated at 20,000 tokens, if the response marker is missing or ambiguous,
or if no token would remain supervised after the marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from freeze_forbidden_source_registry import (  # type: ignore[import-not-found]
    validate_base_sft_source_registry as validate_forbidden_source_registry,
)

SCHEMA_VERSION = 2
PROTOCOL = "clasp-jiang-f638-sft-all-row-token-length-audit-v2"
STATUS = "all_rows_fit_without_truncation"
RECEIPT_FILENAME = "sft_token_length_audit.json"
SHARD_PROTOCOL = "clasp-jiang-f638-sft-all-row-token-length-task-shard-v2"
SHARD_STATUS = "task_all_rows_fit_without_truncation"
TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
MAX_SEQ_LENGTH = 20_000
EXPECTED_ROWS_PER_TASK = 500_000
PROMPT_STYLE = "jiang_f638"
NO_REASON = True
RESPONSE_HEADER = "    ### Response"
UPSTREAM_RESPONSE_TEMPLATE = "### Response:"
UPSTREAM_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
BOUNDARY_CONTRACT = {
    "upstream_commit": UPSTREAM_COMMIT,
    "upstream_response_template": UPSTREAM_RESPONSE_TEMPLATE,
    "corrected_contextual_response_header": RESPONSE_HEADER,
    "supervised_boundary": "contextual_response_header_next_token",
    "first_supervised_token_policy": ("include_response_colon_token_which_may_span_output_prefix"),
    "required_raw_header_occurrences": 1,
    "required_header_token_occurrences": 1,
    "missing_or_ambiguous_header_policy": "fail_closed",
    "audit_dataset_shuffle_passes": 0,
    "training_dataset_shuffle_passes": 1,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The Jiang-f638 trainer consumes the released instruction/input/output fields
# directly. Pin every optional CLASP representation rewrite off here.
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


class SFTTokenLengthAuditError(ValueError):
    """Raised when the full-context claim cannot be certified."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SFTTokenLengthAuditError(f"required artifact is not a regular file: {path}")
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise SFTTokenLengthAuditError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise SFTTokenLengthAuditError(f"{label} is not a regular directory: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise SFTTokenLengthAuditError(
            f"{label} must be supplied canonically: supplied={raw}, resolved={resolved}"
        )
    return resolved


def _canonical_file(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink() or not raw.is_file():
        raise SFTTokenLengthAuditError(f"{label} is not a regular file: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise SFTTokenLengthAuditError(
            f"{label} must be supplied canonically: supplied={raw}, resolved={resolved}"
        )
    return resolved


def tokenizer_artifact_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash every local file that can affect Qwen tokenizer construction."""

    root = _canonical_directory(path, label="tokenizer root")
    candidates: list[Path] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise SFTTokenLengthAuditError(f"tokenizer root contains a symlink: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise SFTTokenLengthAuditError(f"tokenizer root contains a special entry: {candidate}")
        name = candidate.name
        suffix = candidate.suffix.lower()
        if suffix in {".json", ".model", ".txt", ".tiktoken", ".jinja", ".py"} or name.startswith(
            ("tokenizer", "vocab", "merges", "special_tokens", "added_tokens")
        ):
            candidates.append(candidate)
    if not candidates:
        raise SFTTokenLengthAuditError(f"tokenizer root has no tokenizer artifacts: {root}")
    files = [
        {
            "relative_path": candidate.relative_to(root).as_posix(),
            "size_bytes": candidate.stat(follow_symlinks=False).st_size,
            "sha256": _sha256_file(candidate),
        }
        for candidate in candidates
    ]
    identity: dict[str, Any] = {
        "canonical_path": str(root),
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return identity


def _iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[tuple[int, dict]]:
    """Yield a top-level JSON array without materializing the complete file."""

    if type(chunk_size) is not int or chunk_size < 64:
        raise SFTTokenLengthAuditError("JSON streaming chunk size must be at least 64 bytes")
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def compact_and_read() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        compact_and_read()
        while not eof and not buffer.strip():
            compact_and_read()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise SFTTokenLengthAuditError(f"dataset must be a top-level JSON list: {path}")
        position += 1
        row_index = 0
        need_comma = False
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise SFTTokenLengthAuditError(f"unterminated JSON list: {path}")
                compact_and_read()
            if need_comma:
                if buffer[position] == "]":
                    position += 1
                    break
                if buffer[position] != ",":
                    raise SFTTokenLengthAuditError(
                        f"malformed JSON {path}: expected comma before row {row_index}"
                    )
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        break
                    if eof:
                        raise SFTTokenLengthAuditError(f"unterminated JSON list: {path}")
                    compact_and_read()
                if buffer[position] == "]":
                    raise SFTTokenLengthAuditError(f"malformed JSON {path}: trailing comma")
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
                        raise SFTTokenLengthAuditError(f"malformed JSON {path}: {exc}") from exc
                    compact_and_read()
            if not isinstance(value, dict):
                raise SFTTokenLengthAuditError(f"non-object row at {path}:{row_index}")
            yield row_index, value
            row_index += 1
            need_comma = True
            if position > chunk_size:
                compact_and_read()
        trailing = buffer[position:] + handle.read()
        if trailing.strip():
            raise SFTTokenLengthAuditError(f"data follows top-level JSON list: {path}")


def _render_training_row(
    task: str,
    row: Mapping[str, Any],
    *,
    eos_token: str,
    tokenizer: Any,
) -> str:
    """Render one row exactly as ``main_train.py`` at ``UPSTREAM_COMMIT``.

    ``task`` and ``tokenizer`` remain explicit arguments so callers cannot
    accidentally use a separate task formatter or tokenizer-owned chat
    template.  Neither value changes the released raw-column formatting.
    """

    del tokenizer
    if task not in TASKS:
        raise SFTTokenLengthAuditError(f"unsupported SFT task: {task}")
    missing = [name for name in ("instruction", "input", "output") if name not in row]
    if missing:
        raise SFTTokenLengthAuditError(f"row is missing released dataset fields: {missing}")

    # Import only in the expensive producer lane.  Registry and campaign
    # validators remain usable without importing the CLASP package.
    from clasp.data.prompts import build_jiang_f638_sft_text

    return build_jiang_f638_sft_text(
        row["instruction"],
        row["input"],
        row["output"],
        eos_token=eos_token,
    )


def _tokenize_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    try:
        # Jiang's TRL 0.15.2 SFTTrainer tokenizes the ``text`` field with the
        # tokenizer's special-token policy. Pin that policy here; using
        # ``add_special_tokens=False`` could under-count a tokenizer that
        # injects BOS/EOS tokens and would no longer prove no truncation.
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
    except Exception as exc:
        raise SFTTokenLengthAuditError(f"tokenizer failed during all-row audit: {exc}") from exc
    if not isinstance(input_ids, list) or len(input_ids) != len(texts):
        raise SFTTokenLengthAuditError("tokenizer returned an invalid batch shape")
    normalized: list[list[int]] = []
    for row_index, values in enumerate(input_ids):
        if not isinstance(values, list) or any(
            type(value) is not int or value < 0 for value in values
        ):
            raise SFTTokenLengthAuditError(
                f"tokenizer returned invalid token IDs for batch row {row_index}"
            )
        normalized.append(values)
    return normalized


def _response_header_token_ids(tokenizer: Any) -> list[int]:
    """Tokenize the shared deterministic boundary used by trainer and audit."""

    try:
        from clasp.data.prompts import (
            JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY,
            JIANG_F638_RESPONSE_HEADER,
            JIANG_F638_SUPERVISED_BOUNDARY,
            jiang_f638_response_header_token_ids,
        )

        if (
            JIANG_F638_RESPONSE_HEADER != RESPONSE_HEADER
            or JIANG_F638_SUPERVISED_BOUNDARY != BOUNDARY_CONTRACT["supervised_boundary"]
            or JIANG_F638_FIRST_SUPERVISED_TOKEN_POLICY
            != BOUNDARY_CONTRACT["first_supervised_token_policy"]
        ):
            raise SFTTokenLengthAuditError("Jiang-f638 response-boundary constants disagree")
        token_ids = jiang_f638_response_header_token_ids(tokenizer)
    except Exception as exc:
        raise SFTTokenLengthAuditError(
            f"tokenizer failed to encode {RESPONSE_HEADER!r}: {exc}"
        ) from exc
    return token_ids


def _quantile_from_histogram(histogram: Mapping[int, int], total: int, q: float) -> int:
    target = max(1, int(math.ceil(float(q) * total)))
    cumulative = 0
    for length, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return int(length)
    raise SFTTokenLengthAuditError("cannot derive quantile from incomplete histogram")


def _source_binding(
    *,
    registry_identity: Mapping[str, Any],
    data_root: Path,
    task: str,
) -> tuple[dict[str, Any], list[tuple[str, Path, Mapping[str, Any]]]]:
    sources = registry_identity.get("sources_by_task_label")
    registered = (
        sources.get(task, {}).get("sft-train")
        if isinstance(sources, dict) and isinstance(sources.get(task), dict)
        else None
    )
    if not isinstance(registered, dict):
        raise SFTTokenLengthAuditError(f"registry has no {task}/sft-train source")
    source_root = data_root / task / "train"
    if source_root.is_symlink() or not source_root.is_dir():
        raise SFTTokenLengthAuditError(f"SFT source is not a regular directory: {source_root}")
    files = registered.get("files")
    if not isinstance(files, list) or not files:
        raise SFTTokenLengthAuditError(f"registry {task}/sft-train has no files")
    expected: list[tuple[str, Path, Mapping[str, Any]]] = []
    for record in files:
        if not isinstance(record, dict):
            raise SFTTokenLengthAuditError(f"registry {task}/sft-train file is invalid")
        relative = str(record.get("relative_path", ""))
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            raise SFTTokenLengthAuditError(f"registry {task}/sft-train path is unsafe: {relative}")
        path = source_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise SFTTokenLengthAuditError(f"registered SFT file is missing: {path}")
        if path.stat(follow_symlinks=False).st_size != record.get("size_bytes"):
            raise SFTTokenLengthAuditError(f"registered SFT file size changed: {path}")
        if _sha256_file(path) != record.get("sha256"):
            raise SFTTokenLengthAuditError(f"registered SFT file bytes changed: {path}")
        expected.append((relative, path, record))
    observed = sorted(
        candidate.relative_to(source_root).as_posix()
        for candidate in source_root.rglob("*.json")
        if candidate.is_file()
    )
    if observed != [relative for relative, _path, _record in expected]:
        raise SFTTokenLengthAuditError(f"{task} SFT JSON inventory differs from the registry")
    binding = {
        "canonical_path": str(source_root.resolve(strict=True)),
        "source_identity_sha256": registered["source_identity_sha256"],
        "file_manifest_sha256": registered["file_manifest_sha256"],
        "rows": registered["rows"],
        "unique_instance_hashes": registered["unique_instance_hashes"],
    }
    return binding, expected


def audit_task(
    *,
    task: str,
    source_binding: Mapping[str, Any],
    files: Sequence[tuple[str, Path, Mapping[str, Any]]],
    tokenizer: Any,
    eos_token: str,
    tokenizer_identity_sha256: str,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Stream and tokenize one task using bounded batches."""

    if task not in TASKS:
        raise SFTTokenLengthAuditError(f"unsupported SFT task: {task}")
    if type(batch_size) is not int or batch_size <= 0:
        raise SFTTokenLengthAuditError("tokenization batch size must be positive")
    if _SHA256_RE.fullmatch(tokenizer_identity_sha256) is None:
        raise SFTTokenLengthAuditError("tokenizer identity is invalid")
    response_header_ids = _response_header_token_ids(tokenizer)
    histogram: Counter[int] = Counter()
    ordered = hashlib.sha256()
    ordered.update(b"clasp-jiang-f638-sft-row-length-evidence-v2\0")
    ordered.update(task.encode("ascii"))
    ordered.update(b"\0")
    ordered.update(str(source_binding.get("source_identity_sha256", "")).encode("ascii"))
    ordered.update(b"\0")
    ordered.update(tokenizer_identity_sha256.encode("ascii"))
    ordered.update(b"\0")
    prompt_max = 0
    completion_min: int | None = None
    completion_max = 0
    full_min: int | None = None
    full_max = 0
    at_limit = 0
    over_limit = 0
    alignment_mismatches = 0
    zero_completion = 0
    row_count = 0
    batch: list[tuple[str, int, str]] = []

    def consume() -> None:
        nonlocal prompt_max, completion_min, completion_max, full_min, full_max
        nonlocal at_limit, over_limit, alignment_mismatches, zero_completion, row_count
        if not batch:
            return
        full_texts = [record[2] for record in batch]
        full_ids_batch = _tokenize_batch(tokenizer, full_texts)
        from clasp.data.prompts import locate_jiang_f638_supervised_start

        for (relative, file_row, full_text), full_ids in zip(
            batch,
            full_ids_batch,
            strict=True,
        ):
            full_length = len(full_ids)
            row_label = f"{task}/{relative}:{file_row}"
            if full_length > max_seq_length:
                over_limit += 1
                raise SFTTokenLengthAuditError(
                    f"{row_label}: {full_length} tokens exceed max_seq_length={max_seq_length}"
                )
            try:
                prompt_length = locate_jiang_f638_supervised_start(
                    full_text,
                    full_ids,
                    response_header_ids,
                )
            except ValueError as exc:
                alignment_mismatches += 1
                if "no supervised token" in str(exc):
                    zero_completion += 1
                raise SFTTokenLengthAuditError(f"{row_label}: {exc}") from exc
            completion_length = full_length - prompt_length
            if completion_length <= 0:
                zero_completion += 1
                raise SFTTokenLengthAuditError(
                    f"{row_label}: no supervised token follows the response template"
                )
            prompt_max = max(prompt_max, prompt_length)
            completion_min = (
                completion_length
                if completion_min is None
                else min(completion_min, completion_length)
            )
            completion_max = max(completion_max, completion_length)
            full_min = full_length if full_min is None else min(full_min, full_length)
            full_max = max(full_max, full_length)
            histogram[full_length] += 1
            at_limit += int(full_length == max_seq_length)
            relative_bytes = relative.encode("utf-8")
            ordered.update(len(relative_bytes).to_bytes(4, "big", signed=False))
            ordered.update(relative_bytes)
            ordered.update(file_row.to_bytes(8, "big", signed=False))
            ordered.update(prompt_length.to_bytes(8, "big", signed=False))
            ordered.update(full_length.to_bytes(8, "big", signed=False))
            ordered.update(completion_length.to_bytes(8, "big", signed=True))
            ordered.update(
                bytes(
                    (
                        1,
                        int(completion_length > 0),
                        int(full_length <= max_seq_length),
                    )
                )
            )
            row_count += 1
        batch.clear()

    for relative, path, file_record in files:
        before = path.stat(follow_symlinks=False)
        file_rows = 0
        for file_row, row in _iter_json_array(path):
            try:
                full_text = _render_training_row(
                    task,
                    row,
                    eos_token=eos_token,
                    tokenizer=tokenizer,
                )
            except SFTTokenLengthAuditError as exc:
                raise SFTTokenLengthAuditError(f"{path}:{file_row}: {exc}") from exc
            batch.append((relative, file_row, full_text))
            file_rows += 1
            if len(batch) >= batch_size:
                consume()
        consume()
        after = path.stat(follow_symlinks=False)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity):
            raise SFTTokenLengthAuditError(f"SFT file changed during token audit: {path}")
        if file_rows != file_record.get("rows"):
            raise SFTTokenLengthAuditError(
                f"{path} streamed {file_rows} rows, registry declares {file_record.get('rows')}"
            )
    if row_count <= 0 or sum(histogram.values()) != row_count:
        raise SFTTokenLengthAuditError(f"{task} token audit has an incomplete histogram")
    histogram_rows = [[length, histogram[length]] for length in sorted(histogram)]
    return {
        "source_binding": dict(source_binding),
        "row_count": row_count,
        "prompt_token_max": prompt_max,
        "completion_token_min": int(completion_min or 0),
        "completion_token_max": completion_max,
        "full_token_min": int(full_min or 0),
        "full_token_max": full_max,
        "full_token_p50": _quantile_from_histogram(histogram, row_count, 0.50),
        "full_token_p95": _quantile_from_histogram(histogram, row_count, 0.95),
        "full_token_p99": _quantile_from_histogram(histogram, row_count, 0.99),
        "full_token_length_histogram": histogram_rows,
        "at_max_seq_length_count": at_limit,
        "over_max_seq_length_count": over_limit,
        "prompt_completion_alignment_mismatch_count": alignment_mismatches,
        "zero_supervised_completion_count": zero_completion,
        "ordered_row_length_evidence_sha256": ordered.hexdigest(),
    }


def _validate_task_record(
    value: object,
    *,
    task: str,
    expected_source: Mapping[str, Any],
    expected_rows: int,
    max_seq_length: int,
) -> dict[str, Any]:
    expected_keys = {
        "source_binding",
        "row_count",
        "prompt_token_max",
        "completion_token_min",
        "completion_token_max",
        "full_token_min",
        "full_token_max",
        "full_token_p50",
        "full_token_p95",
        "full_token_p99",
        "full_token_length_histogram",
        "at_max_seq_length_count",
        "over_max_seq_length_count",
        "prompt_completion_alignment_mismatch_count",
        "zero_supervised_completion_count",
        "ordered_row_length_evidence_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SFTTokenLengthAuditError(f"{task} token audit has an invalid schema")
    histogram = value.get("full_token_length_histogram")
    if not isinstance(histogram, list) or not histogram:
        raise SFTTokenLengthAuditError(f"{task} token audit has no length histogram")
    lengths: list[int] = []
    counts: list[int] = []
    for pair in histogram:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or pair[0] <= 0
            or type(pair[1]) is not int
            or pair[1] <= 0
        ):
            raise SFTTokenLengthAuditError(f"{task} token audit histogram is invalid")
        lengths.append(pair[0])
        counts.append(pair[1])
    if lengths != sorted(set(lengths)) or sum(counts) != expected_rows:
        raise SFTTokenLengthAuditError(f"{task} token audit histogram is incomplete")
    hist_map = dict(zip(lengths, counts, strict=True))
    integer_fields = expected_keys - {
        "source_binding",
        "full_token_length_histogram",
        "ordered_row_length_evidence_sha256",
    }
    if any(type(value.get(field)) is not int for field in integer_fields):
        raise SFTTokenLengthAuditError(f"{task} token audit has non-integer counts")
    if (
        value.get("source_binding") != dict(expected_source)
        or value.get("row_count") != expected_rows
        or value.get("completion_token_min", 0) <= 0
        or value.get("full_token_min") != lengths[0]
        or value.get("full_token_max") != lengths[-1]
        or value.get("full_token_max", max_seq_length + 1) > max_seq_length
        or value.get("full_token_p50") != _quantile_from_histogram(hist_map, expected_rows, 0.50)
        or value.get("full_token_p95") != _quantile_from_histogram(hist_map, expected_rows, 0.95)
        or value.get("full_token_p99") != _quantile_from_histogram(hist_map, expected_rows, 0.99)
        or value.get("at_max_seq_length_count") != hist_map.get(max_seq_length, 0)
        or value.get("over_max_seq_length_count") != 0
        or value.get("prompt_completion_alignment_mismatch_count") != 0
        or value.get("zero_supervised_completion_count") != 0
        or _SHA256_RE.fullmatch(str(value.get("ordered_row_length_evidence_sha256", ""))) is None
    ):
        raise SFTTokenLengthAuditError(
            f"{task} does not prove all {expected_rows} rows fit untruncated at {max_seq_length}"
        )
    return value


def _validate_tokenizer_runtime(value: object) -> dict[str, Any]:
    runtime_keys = {"tokenizer_class", "is_fast", "eos_token", "eos_token_id", "vocab_size"}
    if (
        not isinstance(value, dict)
        or set(value) != runtime_keys
        or not isinstance(value.get("tokenizer_class"), str)
        or not value["tokenizer_class"]
        or value.get("is_fast") is not True
        or not isinstance(value.get("eos_token"), str)
        or not value["eos_token"]
        or type(value.get("eos_token_id")) is not int
        or value["eos_token_id"] < 0
        or type(value.get("vocab_size")) is not int
        or value["vocab_size"] <= 0
    ):
        raise SFTTokenLengthAuditError("tokenizer runtime evidence is invalid")
    return value


def build_receipt(
    *,
    data_root: Path,
    tokenizer_path: Path,
    forbidden_registry_identity: Mapping[str, Any],
    task_audits: Mapping[str, Mapping[str, Any]],
    tokenizer_artifacts: Mapping[str, Any],
    tokenizer_runtime: Mapping[str, Any],
    max_seq_length: int = MAX_SEQ_LENGTH,
    expected_rows_per_task: int = EXPECTED_ROWS_PER_TASK,
) -> dict[str, Any]:
    if tuple(task_audits) != TASKS:
        raise SFTTokenLengthAuditError("token audit must cover all five tasks in canonical order")
    sources = forbidden_registry_identity.get("sources_by_task_label")
    expected_sources: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        registered = (
            sources.get(task, {}).get("sft-train")
            if isinstance(sources, dict) and isinstance(sources.get(task), dict)
            else None
        )
        if not isinstance(registered, dict):
            raise SFTTokenLengthAuditError(f"registry lacks {task}/sft-train")
        expected_sources[task] = {
            "canonical_path": str((data_root / task / "train").resolve(strict=True)),
            "source_identity_sha256": registered["source_identity_sha256"],
            "file_manifest_sha256": registered["file_manifest_sha256"],
            "rows": registered["rows"],
            "unique_instance_hashes": registered["unique_instance_hashes"],
        }
        _validate_task_record(
            task_audits[task],
            task=task,
            expected_source=expected_sources[task],
            expected_rows=expected_rows_per_task,
            max_seq_length=max_seq_length,
        )
    _validate_tokenizer_runtime(dict(tokenizer_runtime))
    live_tokenizer_artifacts = tokenizer_artifact_identity(tokenizer_path)
    if dict(tokenizer_artifacts) != live_tokenizer_artifacts:
        raise SFTTokenLengthAuditError("tokenizer artifacts changed during the all-row audit")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "data_root": str(data_root),
        "tokenizer_path": str(tokenizer_path),
        "tasks": list(TASKS),
        "rows_per_task": expected_rows_per_task,
        "total_rows": expected_rows_per_task * len(TASKS),
        "max_seq_length": max_seq_length,
        "prompt_style": PROMPT_STYLE,
        "no_reason": NO_REASON,
        "boundary_contract": dict(BOUNDARY_CONTRACT),
        "representation_environment": dict(REPRESENTATION_ENVIRONMENT),
        "forbidden_source_registry": {
            "registry_sha256": forbidden_registry_identity["registry_sha256"],
            "source_tree_manifest_sha256": forbidden_registry_identity[
                "source_tree_manifest_sha256"
            ],
        },
        "tokenizer_artifacts": dict(tokenizer_artifacts),
        "tokenizer_runtime": dict(tokenizer_runtime),
        "task_audits": {task: dict(task_audits[task]) for task in TASKS},
        "producer_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def _receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace token-length audit receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_receipt_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace token-length audit receipt: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_task_shard_receipt(
    *,
    task: str,
    data_root: Path,
    tokenizer_path: Path,
    forbidden_registry_identity: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    tokenizer_artifacts: Mapping[str, Any],
    tokenizer_runtime: Mapping[str, Any],
    task_audit: Mapping[str, Any],
    max_seq_length: int = MAX_SEQ_LENGTH,
    expected_rows: int = EXPECTED_ROWS_PER_TASK,
) -> dict[str, Any]:
    """Build one independently producible task receipt for later five-way merge."""

    if task not in TASKS:
        raise SFTTokenLengthAuditError(f"unsupported SFT task shard: {task}")
    _validate_task_record(
        task_audit,
        task=task,
        expected_source=source_binding,
        expected_rows=expected_rows,
        max_seq_length=max_seq_length,
    )
    _validate_tokenizer_runtime(dict(tokenizer_runtime))
    live_tokenizer_artifacts = tokenizer_artifact_identity(tokenizer_path)
    if dict(tokenizer_artifacts) != live_tokenizer_artifacts:
        raise SFTTokenLengthAuditError("tokenizer artifacts changed during the task audit")
    registry_binding = {
        "registry_sha256": forbidden_registry_identity.get("registry_sha256"),
        "source_tree_manifest_sha256": forbidden_registry_identity.get(
            "source_tree_manifest_sha256"
        ),
    }
    if any(_SHA256_RE.fullmatch(str(value or "")) is None for value in registry_binding.values()):
        raise SFTTokenLengthAuditError("task shard has an invalid forbidden-source binding")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": SHARD_PROTOCOL,
        "status": SHARD_STATUS,
        "task": task,
        "data_root": str(data_root),
        "tokenizer_path": str(tokenizer_path),
        "rows": expected_rows,
        "max_seq_length": max_seq_length,
        "prompt_style": PROMPT_STYLE,
        "no_reason": NO_REASON,
        "boundary_contract": dict(BOUNDARY_CONTRACT),
        "representation_environment": dict(REPRESENTATION_ENVIRONMENT),
        "forbidden_source_registry": registry_binding,
        "tokenizer_artifacts": dict(tokenizer_artifacts),
        "tokenizer_runtime": dict(tokenizer_runtime),
        "task_audit": dict(task_audit),
        "producer_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def validate_task_shard_receipt(
    path: str | os.PathLike[str],
    *,
    task: str,
    data_root: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    expected_source: Mapping[str, Any],
    expected_registry_identity: Mapping[str, Any],
    tokenizer_artifacts: Mapping[str, Any] | None = None,
    expected_rows: int = EXPECTED_ROWS_PER_TASK,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """Validate a read-only task shard and all live data/tokenizer bindings."""

    data = _canonical_directory(data_root, label="SFT data root")
    tokenizer = _canonical_directory(tokenizer_path, label="tokenizer root")
    shard_path = _canonical_file(path, label=f"{task} token-length task receipt")
    if stat.S_IMODE(shard_path.stat(follow_symlinks=False).st_mode) & 0o222:
        raise SFTTokenLengthAuditError(f"task shard must be read-only: {shard_path}")
    try:
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTTokenLengthAuditError(f"cannot read task shard {shard_path}: {exc}") from exc
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "task",
        "data_root",
        "tokenizer_path",
        "rows",
        "max_seq_length",
        "prompt_style",
        "no_reason",
        "boundary_contract",
        "representation_environment",
        "forbidden_source_registry",
        "tokenizer_artifacts",
        "tokenizer_runtime",
        "task_audit",
        "producer_sha256",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise SFTTokenLengthAuditError(f"{task} token-length task receipt has an invalid schema")
    if shard_path.read_bytes() != _receipt_bytes(payload):
        raise SFTTokenLengthAuditError(f"{task} token-length task receipt is not canonical JSON")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("receipt_sha256", None)
    registry_binding = {
        "registry_sha256": expected_registry_identity.get("registry_sha256"),
        "source_tree_manifest_sha256": expected_registry_identity.get(
            "source_tree_manifest_sha256"
        ),
    }
    live_tokenizer_artifacts = tokenizer_artifact_identity(tokenizer)
    if tokenizer_artifacts is not None and dict(tokenizer_artifacts) != live_tokenizer_artifacts:
        raise SFTTokenLengthAuditError("tokenizer changed while task shards were being merged")
    if (
        stored_hash != _sha256_json(unsigned)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != SHARD_PROTOCOL
        or payload.get("status") != SHARD_STATUS
        or payload.get("task") != task
        or payload.get("data_root") != str(data)
        or payload.get("tokenizer_path") != str(tokenizer)
        or payload.get("rows") != expected_rows
        or payload.get("max_seq_length") != max_seq_length
        or payload.get("prompt_style") != PROMPT_STYLE
        or payload.get("no_reason") is not NO_REASON
        or payload.get("boundary_contract") != BOUNDARY_CONTRACT
        or payload.get("representation_environment") != REPRESENTATION_ENVIRONMENT
        or payload.get("forbidden_source_registry") != registry_binding
        or payload.get("tokenizer_artifacts") != live_tokenizer_artifacts
        or payload.get("producer_sha256") != _sha256_file(Path(__file__).resolve(strict=True))
    ):
        raise SFTTokenLengthAuditError(f"{task} task shard disagrees with the audit contract")
    _validate_tokenizer_runtime(payload.get("tokenizer_runtime"))
    _validate_task_record(
        payload.get("task_audit"),
        task=task,
        expected_source=expected_source,
        expected_rows=expected_rows,
        max_seq_length=max_seq_length,
    )
    return payload


def validate_token_length_receipt(
    path: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    expected_sources: Mapping[str, Mapping[str, Any]],
    expected_registry_identity: Mapping[str, Any],
    expected_rows_per_task: int = EXPECTED_ROWS_PER_TASK,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """Validate the receipt and its live tokenizer/source identities."""

    data = _canonical_directory(data_root, label="SFT data root")
    tokenizer = _canonical_directory(tokenizer_path, label="tokenizer root")
    receipt_path = Path(path).expanduser().absolute()
    if receipt_path != data / RECEIPT_FILENAME:
        raise SFTTokenLengthAuditError(
            f"token-length receipt must be the data root's {RECEIPT_FILENAME}"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SFTTokenLengthAuditError(
            f"token-length receipt is not a regular file: {receipt_path}"
        )
    if stat.S_IMODE(receipt_path.stat(follow_symlinks=False).st_mode) & 0o222:
        raise SFTTokenLengthAuditError(f"token-length receipt must be read-only: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTTokenLengthAuditError(f"cannot read token-length receipt: {exc}") from exc
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "data_root",
        "tokenizer_path",
        "tasks",
        "rows_per_task",
        "total_rows",
        "max_seq_length",
        "prompt_style",
        "no_reason",
        "boundary_contract",
        "representation_environment",
        "forbidden_source_registry",
        "tokenizer_artifacts",
        "tokenizer_runtime",
        "task_audits",
        "producer_sha256",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise SFTTokenLengthAuditError("token-length receipt has an invalid schema")
    if receipt_path.read_bytes() != _receipt_bytes(payload):
        raise SFTTokenLengthAuditError("token-length receipt is not canonical JSON")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("receipt_sha256", None)
    if stored_hash != _sha256_json(unsigned):
        raise SFTTokenLengthAuditError("token-length receipt self-hash mismatch")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("status") != STATUS
        or payload.get("data_root") != str(data)
        or payload.get("tokenizer_path") != str(tokenizer)
        or payload.get("tasks") != list(TASKS)
        or payload.get("rows_per_task") != expected_rows_per_task
        or payload.get("total_rows") != expected_rows_per_task * len(TASKS)
        or payload.get("max_seq_length") != max_seq_length
        or payload.get("prompt_style") != PROMPT_STYLE
        or payload.get("no_reason") is not NO_REASON
        or payload.get("boundary_contract") != BOUNDARY_CONTRACT
        or payload.get("representation_environment") != REPRESENTATION_ENVIRONMENT
        or payload.get("producer_sha256") != _sha256_file(Path(__file__).resolve(strict=True))
    ):
        raise SFTTokenLengthAuditError("token-length receipt disagrees with the SFT contract")
    expected_registry = {
        "registry_sha256": expected_registry_identity.get("registry_sha256"),
        "source_tree_manifest_sha256": expected_registry_identity.get(
            "source_tree_manifest_sha256"
        ),
    }
    if payload.get("forbidden_source_registry") != expected_registry or any(
        _SHA256_RE.fullmatch(str(value or "")) is None for value in expected_registry.values()
    ):
        raise SFTTokenLengthAuditError(
            "token-length receipt belongs to a different forbidden-source registry"
        )
    if payload.get("tokenizer_artifacts") != tokenizer_artifact_identity(tokenizer):
        raise SFTTokenLengthAuditError("tokenizer artifacts changed after the all-row audit")
    _validate_tokenizer_runtime(payload.get("tokenizer_runtime"))
    audits = payload.get("task_audits")
    if (
        not isinstance(audits, dict)
        or tuple(audits) != TASKS
        or set(expected_sources) != set(TASKS)
    ):
        raise SFTTokenLengthAuditError("token-length receipt does not cover exactly five tasks")
    for task in TASKS:
        _validate_task_record(
            audits[task],
            task=task,
            expected_source=expected_sources[task],
            expected_rows=expected_rows_per_task,
            max_seq_length=max_seq_length,
        )
    return payload


def token_length_audit_identity(
    receipt: Mapping[str, Any], *, receipt_path: Path
) -> dict[str, Any]:
    audits = receipt["task_audits"]
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": _sha256_file(receipt_path),
        "tokenizer_identity_sha256": receipt["tokenizer_artifacts"]["identity_sha256"],
        "task_audits_sha256": _sha256_json(audits),
        "rows_per_task": receipt["rows_per_task"],
        "total_rows": receipt["total_rows"],
        "max_seq_length": receipt["max_seq_length"],
        "maximum_observed_tokens": max(audits[task]["full_token_max"] for task in TASKS),
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return identity


def validate_token_length_audit_identity(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "receipt_sha256",
        "receipt_file_sha256",
        "tokenizer_identity_sha256",
        "task_audits_sha256",
        "rows_per_task",
        "total_rows",
        "max_seq_length",
        "maximum_observed_tokens",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SFTTokenLengthAuditError("token-length audit identity has an invalid schema")
    unsigned = dict(value)
    identity_hash = unsigned.pop("identity_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL
        or value.get("status") != STATUS
        or any(
            _SHA256_RE.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "receipt_sha256",
                "receipt_file_sha256",
                "tokenizer_identity_sha256",
                "task_audits_sha256",
            )
        )
        or value.get("rows_per_task") != EXPECTED_ROWS_PER_TASK
        or value.get("total_rows") != EXPECTED_ROWS_PER_TASK * len(TASKS)
        or value.get("max_seq_length") != MAX_SEQ_LENGTH
        or type(value.get("maximum_observed_tokens")) is not int
        or not 0 < value["maximum_observed_tokens"] <= MAX_SEQ_LENGTH
        or identity_hash != _sha256_json(unsigned)
    ):
        raise SFTTokenLengthAuditError("token-length audit identity is invalid")
    return value


def _load_registry_identity(path: Path) -> dict[str, Any]:
    registry_path = _canonical_file(path, label="forbidden-source registry")
    if stat.S_IMODE(registry_path.stat(follow_symlinks=False).st_mode) & 0o222:
        raise SFTTokenLengthAuditError("forbidden-source registry must be read-only")
    try:
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTTokenLengthAuditError(f"cannot read forbidden-source registry: {exc}") from exc
    return validate_forbidden_source_registry(
        registry_payload,
        expected_tasks=list(TASKS),
    )


def _load_tokenizer_context(tokenizer_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer_root = _canonical_directory(tokenizer_path, label="tokenizer root")
    artifacts = tokenizer_artifact_identity(tokenizer_root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root),
        trust_remote_code=True,
        local_files_only=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise SFTTokenLengthAuditError("all-row audit requires the exact fast tokenizer")
    eos_token = str(getattr(tokenizer, "eos_token", "") or "")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not eos_token or type(eos_token_id) is not int or eos_token_id < 0:
        raise SFTTokenLengthAuditError("tokenizer has no valid EOS token")
    try:
        vocab_size = len(tokenizer)
    except TypeError as exc:
        raise SFTTokenLengthAuditError("tokenizer has no finite vocabulary") from exc
    runtime = {
        "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "is_fast": True,
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "vocab_size": int(vocab_size),
    }
    if tokenizer_artifact_identity(tokenizer_root) != artifacts:
        raise SFTTokenLengthAuditError("tokenizer artifacts changed while loading the tokenizer")
    return tokenizer, runtime, artifacts


def run_audit(
    *,
    data_root: Path,
    tokenizer_path: Path,
    forbidden_source_registry: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Run all five tasks serially; task shards are preferred for the 2.5M-row campaign."""

    data = _canonical_directory(data_root, label="SFT data root")
    tokenizer_root = _canonical_directory(tokenizer_path, label="tokenizer root")
    registry_identity = _load_registry_identity(forbidden_source_registry)
    tokenizer, runtime, artifacts = _load_tokenizer_context(tokenizer_root)
    eos_token = runtime["eos_token"]
    task_audits: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        binding, files = _source_binding(
            registry_identity=registry_identity,
            data_root=data,
            task=task,
        )
        task_audits[task] = audit_task(
            task=task,
            source_binding=binding,
            files=files,
            tokenizer=tokenizer,
            eos_token=eos_token,
            tokenizer_identity_sha256=artifacts["identity_sha256"],
            max_seq_length=MAX_SEQ_LENGTH,
            batch_size=batch_size,
        )
    for task in TASKS:
        current_binding, _current_files = _source_binding(
            registry_identity=registry_identity,
            data_root=data,
            task=task,
        )
        if current_binding != task_audits[task]["source_binding"]:
            raise SFTTokenLengthAuditError(f"{task} SFT source changed during the serial audit")
    return build_receipt(
        data_root=data,
        tokenizer_path=tokenizer_root,
        forbidden_registry_identity=registry_identity,
        task_audits=task_audits,
        tokenizer_artifacts=artifacts,
        tokenizer_runtime=runtime,
    )


def run_task_shard_audit(
    *,
    task: str,
    data_root: Path,
    tokenizer_path: Path,
    forbidden_source_registry: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Run one task so the five full-corpus audits can execute independently."""

    if task not in TASKS:
        raise SFTTokenLengthAuditError(f"unsupported SFT task shard: {task}")
    data = _canonical_directory(data_root, label="SFT data root")
    tokenizer_root = _canonical_directory(tokenizer_path, label="tokenizer root")
    registry_identity = _load_registry_identity(forbidden_source_registry)
    tokenizer, runtime, artifacts = _load_tokenizer_context(tokenizer_root)
    binding, files = _source_binding(
        registry_identity=registry_identity,
        data_root=data,
        task=task,
    )
    task_audit = audit_task(
        task=task,
        source_binding=binding,
        files=files,
        tokenizer=tokenizer,
        eos_token=runtime["eos_token"],
        tokenizer_identity_sha256=artifacts["identity_sha256"],
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=batch_size,
    )
    current_binding, _current_files = _source_binding(
        registry_identity=registry_identity,
        data_root=data,
        task=task,
    )
    if current_binding != binding:
        raise SFTTokenLengthAuditError(f"{task} SFT source changed during the task audit")
    return build_task_shard_receipt(
        task=task,
        data_root=data,
        tokenizer_path=tokenizer_root,
        forbidden_registry_identity=registry_identity,
        source_binding=binding,
        tokenizer_artifacts=artifacts,
        tokenizer_runtime=runtime,
        task_audit=task_audit,
    )


def merge_task_shards(
    *,
    shard_paths: Mapping[str, Path],
    data_root: Path,
    tokenizer_path: Path,
    forbidden_source_registry: Path,
) -> dict[str, Any]:
    """Merge exactly five independently verified task receipts without retokenizing."""

    if set(shard_paths) != set(TASKS):
        raise SFTTokenLengthAuditError("task-shard merge must provide exactly all five tasks")
    data = _canonical_directory(data_root, label="SFT data root")
    tokenizer_root = _canonical_directory(tokenizer_path, label="tokenizer root")
    registry_identity = _load_registry_identity(forbidden_source_registry)
    artifacts = tokenizer_artifact_identity(tokenizer_root)
    task_audits: dict[str, dict[str, Any]] = {}
    runtime: dict[str, Any] | None = None
    for task in TASKS:
        binding, _files = _source_binding(
            registry_identity=registry_identity,
            data_root=data,
            task=task,
        )
        shard = validate_task_shard_receipt(
            Path(shard_paths[task]).expanduser().absolute(),
            task=task,
            data_root=data,
            tokenizer_path=tokenizer_root,
            expected_source=binding,
            expected_registry_identity=registry_identity,
            tokenizer_artifacts=artifacts,
        )
        shard_runtime = shard["tokenizer_runtime"]
        if runtime is None:
            runtime = dict(shard_runtime)
        elif shard_runtime != runtime:
            raise SFTTokenLengthAuditError("task shards used different tokenizer runtimes")
        task_audits[task] = dict(shard["task_audit"])
    if runtime is None:
        raise SFTTokenLengthAuditError("task-shard merge has no tokenizer runtime")
    return build_receipt(
        data_root=data,
        tokenizer_path=tokenizer_root,
        forbidden_registry_identity=registry_identity,
        task_audits=task_audits,
        tokenizer_artifacts=artifacts,
        tokenizer_runtime=runtime,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-seq-length", type=int, choices=[MAX_SEQ_LENGTH], default=MAX_SEQ_LENGTH
    )
    parser.add_argument(
        "--expected-rows-per-task",
        type=int,
        choices=[EXPECTED_ROWS_PER_TASK],
        default=EXPECTED_ROWS_PER_TASK,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--task",
        choices=TASKS,
        help="produce one read-only task shard instead of the final five-task receipt",
    )
    mode.add_argument(
        "--merge-shard-dir",
        type=Path,
        help="merge TASK.json receipts from this directory without retokenizing",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = _canonical_directory(args.data_root, label="SFT data root")
        output = args.output.expanduser().absolute()
        if output.parent.is_symlink() or not output.parent.is_dir():
            raise SFTTokenLengthAuditError(
                f"--output parent is not a regular directory: {output.parent}"
            )
        if output.parent.resolve(strict=True) != output.parent:
            raise SFTTokenLengthAuditError("--output parent must be supplied canonically")
        if args.task is not None:
            payload = run_task_shard_audit(
                task=args.task,
                data_root=data,
                tokenizer_path=args.tokenizer_path,
                forbidden_source_registry=args.forbidden_source_registry,
                batch_size=args.batch_size,
            )
        else:
            if output != data / RECEIPT_FILENAME:
                raise SFTTokenLengthAuditError(f"--output must be {data / RECEIPT_FILENAME}")
            if args.merge_shard_dir is not None:
                shard_root = _canonical_directory(
                    args.merge_shard_dir,
                    label="token-length task-shard directory",
                )
                payload = merge_task_shards(
                    shard_paths={task: shard_root / f"{task}.json" for task in TASKS},
                    data_root=data,
                    tokenizer_path=args.tokenizer_path,
                    forbidden_source_registry=args.forbidden_source_registry,
                )
            else:
                payload = run_audit(
                    data_root=data,
                    tokenizer_path=args.tokenizer_path,
                    forbidden_source_registry=args.forbidden_source_registry,
                    batch_size=args.batch_size,
                )
        _write_create_once(output, payload)
        if args.task is not None:
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "protocol": SHARD_PROTOCOL,
                        "receipt_sha256": payload["receipt_sha256"],
                        "receipt_file_sha256": _sha256_file(output),
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    token_length_audit_identity(payload, receipt_path=output), sort_keys=True
                )
            )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[SFTTokenLengthAuditError] {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
