"""Create-once provenance and semantic analysis for public LLMCoSolver data.

The official repository points to Google Drive folders but publishes no file
checksums.  This module therefore keeps three claims separate:

* a Drive item identifier and observed HTTP metadata;
* the bytes actually frozen in this workspace; and
* deterministic semantic identities derived by parsing every JSON row.

Downloads are published with a hard-link create operation.  An existing file
is validated, never replaced.  Large SFT files are inventoried in place and
are never acquired by this module.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import urllib.request
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

UPSTREAM_COMMIT = "f6388991d243c0aa91d9ee7e2fa18ed1c9e72477"
OFFICIAL_RL_FOLDER_ID = "1VN9crftdW7DTsMQupbc06u6PzRT-Bwnx"
OFFICIAL_SFT_FOLDER_ID = "1bE1coGUa00gfuMkPXnfvldi1-WHGNnEb"
PUBLICATION_TASKS = ("cvrp", "jssp", "mis", "op", "tsp")


@dataclass(frozen=True)
class DriveFile:
    task: str
    split: str
    file_id: str
    byte_length: int
    filename: str
    relative_path: str
    last_modified: str | None = None

    @property
    def download_url(self) -> str:
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={self.file_id}&export=download&confirm=t"
        )


RL_FILES = (
    DriveFile(
        "cvrp",
        "eval",
        "11y_mH0dcKjwOwNjGAc9B5madaSgp9Y71",
        3_490_120,
        "test.json",
        "cvrp/eval/test.json",
    ),
    DriveFile(
        "cvrp",
        "train",
        "1l_tnARZRVIzJrWOZ7QyupSXrHBGkUP_r",
        23_563_607,
        "train_rl.json",
        "cvrp/train/train_rl.json",
    ),
    DriveFile(
        "jssp",
        "eval",
        "19hCmNa0WQL_k_FOUyh09dIZKUjXd-NIr",
        3_358_206,
        "test.json",
        "jssp/eval/test.json",
    ),
    DriveFile(
        "jssp",
        "train",
        "1ZTPJCnJWqJ0HCKOmX0BqO8L1vaP-ZTAG",
        35_695_814,
        "train_rl.json",
        "jssp/train/train_rl.json",
    ),
    DriveFile(
        "mis",
        "eval",
        "1u6FA8rFkFkVUD0rTLQzFGAui-p8M1nVv",
        8_118_895,
        "test.json",
        "mis/eval/test.json",
    ),
    DriveFile(
        "mis",
        "train",
        "1InVqKB9QIj4D9SQarkKLLyYdQotPnN3m",
        95_368_471,
        "train_rl.json",
        "mis/train/train_rl.json",
    ),
    DriveFile(
        "op",
        "eval",
        "1LFz_o5rO9QS5O8PXM1SYA933-n2In01R",
        3_428_819,
        "test.json",
        "op/eval/test.json",
    ),
    DriveFile(
        "op",
        "train",
        "1WaPxOg1a4lyPk7eZ_7Ldf2E3NtvXK7CB",
        57_061_566,
        "train_rl.json",
        "op/train/train_rl.json",
    ),
    DriveFile(
        "tsp",
        "eval",
        "1RlYdsvSSSKDBEj_q98ONL2TXTmB055w5",
        3_140_444,
        "test.json",
        "tsp/eval/test.json",
    ),
    DriveFile(
        "tsp",
        "train",
        "1SgpbRP8iBgOmfcd_nZrvGoChKkiXS_M8",
        33_377_010,
        "train_rl.json",
        "tsp/train/train_rl.json",
    ),
)

RL_SHA256_BY_PATH = {
    "cvrp/eval/test.json": "09d37d3becd158f38ea4fdbf70075a157f2721ce8d4d60cf7d36e714cd34ef8b",
    "cvrp/train/train_rl.json": (
        "4a1c3e97ce4408e722348d893b88ee58a0c1ca65ff0c6db66739901b8a540ff6"
    ),
    "jssp/eval/test.json": "bff60bcf176b0ae43fa5cc899eb6bdc4f0c895874e325b5fcb8d08be0b257b31",
    "jssp/train/train_rl.json": (
        "35c2aeed27e885491a234787855012b9f890e2a970c86ee910a639fc72d0c27a"
    ),
    "mis/eval/test.json": "6c0e8ded09acc0e7cadd42f88f96caff562ae3d8d62445756679f5ce4bdcd933",
    "mis/train/train_rl.json": ("65cc0ea9bad6a0d3a4a5b5d8a37c2b49cc56227bf696bfd79388e6f935308c39"),
    "op/eval/test.json": "b4d5093760e70ceaa9e0b05501057fc1496cfa46c9eb0ea458094a441d341e3f",
    "op/train/train_rl.json": ("bf5319112b68fba04ed41dd96b268d8e22d98620b9266aa9aab69fba2de0b1b4"),
    "tsp/eval/test.json": "f169771f0ccfe0e74cf247067edcea8f89a357ab4823352e3ff55201cf8b9597",
    "tsp/train/train_rl.json": ("90c5a48909ac9df191c727237474ec262db4a9c654f3555a19b744304dc0a288"),
}


SFT_FILES = (
    DriveFile(
        "cvrp",
        "instances",
        "1Bc6YeI32WOQA15hqP4urwefzAj9Ld_-F",
        296_004,
        "instances.pkl",
        "cvrp/instances.pkl",
        "Fri, 26 Sep 2025 18:25:09 GMT",
    ),
    DriveFile(
        "cvrp",
        "eval",
        "1u4JW7RopnbBq3w_hUFtl6aezt7irENcA",
        1_615_617,
        "test.json",
        "cvrp/eval/test.json",
        "Fri, 26 Sep 2025 18:36:07 GMT",
    ),
    DriveFile(
        "cvrp",
        "train",
        "1GY8xT6swZlBN3_FKBTD2tGqjtg0goaCw",
        2_709_936_196,
        "train_cvrp.json",
        "cvrp/train/train_cvrp.json",
        "Fri, 26 Sep 2025 18:29:07 GMT",
    ),
    DriveFile(
        "jssp",
        "instances",
        "1jdRhdVXYcgqNq6a0ukqYYRIi56D1gBGK",
        383_458,
        "instances.pkl",
        "jssp/instances.pkl",
        "Fri, 26 Sep 2025 18:25:10 GMT",
    ),
    DriveFile(
        "jssp",
        "eval",
        "1iErXNi3J0MklMhB9TAq2BcS1zKVORN55",
        1_302_790,
        "eval.json",
        "jssp/eval/eval.json",
        "Fri, 26 Sep 2025 18:36:15 GMT",
    ),
    DriveFile(
        "jssp",
        "train",
        "1r2Wfet7fzOoY21s0te4braPUtQRhTNB4",
        2_180_079_958,
        "training_jssp.json",
        "jssp/train/training_jssp.json",
        "Fri, 26 Sep 2025 18:50:01 GMT",
    ),
    DriveFile(
        "mis",
        "instances",
        "19JV27w-QHJEGqDA0raG0aFO5kSbkMobM",
        493_573,
        "instances.pkl",
        "mis/instances.pkl",
        "Fri, 26 Sep 2025 18:25:09 GMT",
    ),
    DriveFile(
        "mis",
        "eval",
        "1HTNn131St-DPWZ-xROGXuoLLs1SEz9Cl",
        1_108_331,
        "test.json",
        "mis/eval/test.json",
        "Fri, 26 Sep 2025 18:36:09 GMT",
    ),
    DriveFile(
        "mis",
        "train",
        "10r_OZ22iSxsWnTwzabN6JKVtg_HXzwq9",
        1_975_922_309,
        "mis_train.json",
        "mis/train/mis_train.json",
        "Fri, 26 Sep 2025 18:32:02 GMT",
    ),
    DriveFile(
        "op",
        "instances",
        "1wl707BypmY06DWPjZsc5aGexGdms1OyL",
        368_689,
        "instances.pkl",
        "op/instances.pkl",
        "Fri, 26 Sep 2025 18:25:09 GMT",
    ),
    DriveFile(
        "op",
        "eval",
        "1U3WvITyPHIzb8JaBi99hEBgQRz-aA5iA",
        1_588_436,
        "eval.json",
        "op/eval/eval.json",
        "Fri, 26 Sep 2025 18:36:05 GMT",
    ),
    DriveFile(
        "op",
        "train",
        "102bWCorXDUbtEaxcqY0MCpA9DbQf43na",
        2_618_731_138,
        "train_op.json",
        "op/train/train_op.json",
        "Fri, 26 Sep 2025 18:53:58 GMT",
    ),
    DriveFile(
        "tsp",
        "instances",
        "1QnCBzH2dsyS0-mVIEtwBGx49CSvuOW2L",
        249_598,
        "instances.pkl",
        "tsp/instances.pkl",
        "Fri, 26 Sep 2025 18:25:09 GMT",
    ),
    DriveFile(
        "tsp",
        "eval",
        "11W8x-x3Nf2dxdcAFXaIx8cZVFbFOpdvh",
        3_140_444,
        "test.json",
        "tsp/eval/test.json",
        "Fri, 26 Sep 2025 18:36:13 GMT",
    ),
    DriveFile(
        "tsp",
        "train",
        "1QaM1aV10wXXtTFFUXeoody1DmAaWGqPP",
        3_451_489_028,
        "train_tsp.json",
        "tsp/train/train_tsp.json",
        "Fri, 26 Sep 2025 18:46:48 GMT",
    ),
)

OBJECTIVE_PATTERN = re.compile(
    r"(?P<label>Objective|Makespan)\s*:\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> bytes:
    """Return the exact semantic-identity encoding used by the manifest."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_regular_file_stable(
    path: Path,
    *,
    require_read_only: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[str, os.stat_result]:
    """Hash one non-symlink regular file and reject changes during the read."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        if require_read_only and before.st_mode & 0o222:
            raise ValueError(f"Frozen file remains writable: {path}")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"File changed while being hashed: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def seal_regular_file_read_only(path: Path) -> None:
    """Set a regular non-symlink file to 0444 and verify the resulting mode."""

    _, observed = sha256_regular_file_stable(path)
    if observed.st_mode & 0o222:
        os.chmod(path, 0o444, follow_symlinks=False)
    _, sealed = sha256_regular_file_stable(path, require_read_only=True)
    if stat.S_IMODE(sealed.st_mode) != 0o444:
        raise ValueError(f"Read-only file seal is not mode 0444: {path}")


def seal_directory_tree_read_only(root: Path) -> None:
    """Seal all regular files to 0444 and directories to 0555, fail closed."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Data root is not a non-symlink directory: {root}")
    directories: list[Path] = []
    for directory, child_directories, filenames in os.walk(root):
        directory_path = Path(directory)
        directories.append(directory_path)
        for name in child_directories:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"Symlink forbidden in frozen data tree: {child}")
        for name in filenames:
            seal_regular_file_read_only(directory_path / name)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(directory, 0o555, follow_symlinks=False)
        mode = stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode)
        if mode != 0o555:
            raise ValueError(f"Read-only directory seal is not mode 0555: {directory}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON numeric constant is forbidden: {value}")


def iter_json_array(
    path: Path,
    raw_digest: Any,
    *,
    chunk_size: int = 4 * 1024 * 1024,
) -> Iterator[Any]:
    """Stream a top-level JSON array while hashing its exact UTF-8 bytes."""

    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    utf8 = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    position = 0
    eof = False

    with path.open("rb") as handle:

        def fill() -> None:
            nonlocal buffer, eof
            raw = handle.read(chunk_size)
            if raw:
                raw_digest.update(raw)
                buffer += utf8.decode(raw, final=False)
            else:
                buffer += utf8.decode(b"", final=True)
                eof = True

        def skip_space() -> None:
            nonlocal position
            while position < len(buffer) and buffer[position].isspace():
                position += 1

        fill()
        if buffer.startswith("\ufeff"):
            position = 1
        while True:
            skip_space()
            if position < len(buffer) or eof:
                break
            fill()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1
        require_comma = False

        while True:
            while True:
                skip_space()
                if position < len(buffer) or eof:
                    break
                fill()
            if require_comma:
                if position < len(buffer) and buffer[position] == ",":
                    position += 1
                    while True:
                        skip_space()
                        if position < len(buffer) or eof:
                            break
                        fill()
                    if position < len(buffer) and buffer[position] == "]":
                        raise ValueError(f"Trailing comma before ']' in {path}")
                elif position < len(buffer) and buffer[position] == "]":
                    position += 1
                    break
                else:
                    raise ValueError(f"Expected ',' or ']' at character {position} in {path}")
                require_comma = False
                continue
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"Invalid or truncated JSON in {path}: {error}") from error
                    if position:
                        buffer = buffer[position:]
                        position = 0
                    fill()
            yield value
            position = end
            require_comma = True
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0

        while not eof:
            fill()
        if buffer[position:].strip():
            raise ValueError(f"Trailing content after top-level JSON array in {path}")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _shape_signature(value: Any, depth: int = 0) -> str:
    kind = _type_name(value)
    if depth >= 4 or kind not in {"array", "object"}:
        return kind
    if isinstance(value, list):
        children = sorted({_shape_signature(item, depth + 1) for item in value})
        child = "|".join(children) if children else "empty"
        return f"array[{len(value)}]<{child}>"
    fields = ",".join(f"{key}:{_shape_signature(value[key], depth + 1)}" for key in sorted(value))
    return "object{" + fields + "}"


def _digest_sequence(values: list[bytes], *, unique: bool = False) -> str:
    sequence = sorted(set(values)) if unique else values
    digest = hashlib.sha256()
    for value in sequence:
        digest.update(value)
    return digest.hexdigest()


def _decimal_bounds(values: list[str]) -> tuple[str | None, str | None]:
    parsed: list[Decimal] = []
    for value in values:
        try:
            parsed.append(Decimal(value))
        except InvalidOperation as error:
            raise ValueError(f"Invalid objective decimal {value!r}") from error
    if not parsed:
        return None, None
    return str(min(parsed)), str(max(parsed))


def analyze_json_rows(path: Path, *, source_kind: str) -> tuple[dict[str, Any], set[bytes]]:
    """Hash and parse every row, returning a compact semantic inventory."""

    if source_kind not in {"rl", "sft"}:
        raise ValueError(f"Unknown source kind: {source_kind}")
    required = (
        ("instruction", "input", "output", "instance")
        if source_kind == "rl"
        else (
            "instruction",
            "input",
            "output",
        )
    )
    raw_digest = hashlib.sha256()
    ordered_rows = hashlib.sha256()
    ordered_instances = hashlib.sha256()
    ordered_references = hashlib.sha256()
    row_hashes: list[bytes] = []
    instance_hashes: list[bytes] = []
    reference_hashes: list[bytes] = []
    common_keys: set[str] | None = None
    union_keys: set[str] = set()
    field_types: dict[str, Counter[str]] = {}
    instance_shapes: Counter[str] = Counter()
    objective_labels: Counter[str] = Counter()
    objective_values: list[str] = []
    missing_required: Counter[str] = Counter()
    non_string_prompt_fields: Counter[str] = Counter()
    row_count = 0

    for row_index, row in enumerate(iter_json_array(path, raw_digest)):
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            raise ValueError(f"Row {row_index} in {path} is not a string-keyed object")
        row_count += 1
        keys = set(row)
        common_keys = keys if common_keys is None else common_keys & keys
        union_keys.update(keys)
        for key, value in row.items():
            field_types.setdefault(key, Counter())[_type_name(value)] += 1
        for key in required:
            if key not in row:
                missing_required[key] += 1
        for key in ("instruction", "input", "output"):
            if key in row and not isinstance(row[key], str):
                non_string_prompt_fields[key] += 1

        row_hash = hashlib.sha256(canonical_json(row)).digest()
        row_hashes.append(row_hash)
        ordered_rows.update(row_hash)

        if source_kind == "rl" and "instance" in row:
            instance_projection: Any = row["instance"]
        else:
            instance_projection = {
                key: value for key, value in row.items() if key not in {"instruction", "output"}
            }
        instance_hash = hashlib.sha256(canonical_json(instance_projection)).digest()
        instance_hashes.append(instance_hash)
        ordered_instances.update(instance_hash)
        instance_shapes[_shape_signature(instance_projection)] += 1

        if "output" in row:
            reference_hash = hashlib.sha256(canonical_json(row["output"])).digest()
            reference_hashes.append(reference_hash)
            ordered_references.update(reference_hash)
            if isinstance(row["output"], str):
                matches = list(OBJECTIVE_PATTERN.finditer(row["output"]))
                if matches:
                    match = matches[-1]
                    objective_labels[match.group("label").lower()] += 1
                    objective_values.append(match.group("value"))

    if row_count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    if missing_required:
        raise ValueError(f"Missing required fields in {path}: {dict(missing_required)}")
    if non_string_prompt_fields:
        raise ValueError(f"Non-string prompt fields in {path}: {dict(non_string_prompt_fields)}")

    objective_min, objective_max = _decimal_bounds(objective_values)
    instance_set = set(instance_hashes)
    return (
        {
            "bytes": path.stat().st_size,
            "sha256": raw_digest.hexdigest(),
            "top_level": "array",
            "row_count": row_count,
            "common_keys": sorted(common_keys or ()),
            "union_keys": sorted(union_keys),
            "field_type_counts": {
                key: dict(sorted(counts.items())) for key, counts in sorted(field_types.items())
            },
            "row_identity": {
                "definition": "sha256(canonical_json(full_row))",
                "ordered_sha256": ordered_rows.hexdigest(),
                "multiset_sha256": _digest_sequence(sorted(row_hashes)),
                "unique_set_sha256": _digest_sequence(row_hashes, unique=True),
                "unique_count": len(set(row_hashes)),
                "duplicate_count": row_count - len(set(row_hashes)),
            },
            "instance_identity": {
                "definition": (
                    "sha256(canonical_json(row['instance']))"
                    if source_kind == "rl"
                    else "sha256(canonical_json(row excluding instruction and output))"
                ),
                "ordered_sha256": ordered_instances.hexdigest(),
                "multiset_sha256": _digest_sequence(sorted(instance_hashes)),
                "unique_set_sha256": _digest_sequence(instance_hashes, unique=True),
                "unique_count": len(instance_set),
                "duplicate_count": row_count - len(instance_set),
                "shape_counts": dict(sorted(instance_shapes.items())),
            },
            "reference_fields": {
                "field": "output",
                "identity_definition": "sha256(canonical_json(row['output']))",
                "ordered_sha256": ordered_references.hexdigest(),
                "multiset_sha256": _digest_sequence(sorted(reference_hashes)),
                "present_count": len(reference_hashes),
                "parsed_terminal_objective_count": len(objective_values),
                "unparsed_terminal_objective_count": row_count - len(objective_values),
                "parsed_labels": dict(sorted(objective_labels.items())),
                "parsed_lexeme_ordered_sha256": hashlib.sha256(
                    canonical_json(objective_values)
                ).hexdigest(),
                "parsed_decimal_min": objective_min,
                "parsed_decimal_max": objective_max,
            },
        },
        instance_set,
    )


def analyze_split_pair(
    train_path: Path,
    eval_path: Path,
    *,
    source_kind: str,
) -> dict[str, Any]:
    train, train_instances = analyze_json_rows(train_path, source_kind=source_kind)
    evaluation, eval_instances = analyze_json_rows(eval_path, source_kind=source_kind)
    overlap = sorted(train_instances & eval_instances)
    return {
        "train": train,
        "eval": evaluation,
        "train_eval_instance_overlap": {
            "count": len(overlap),
            "sha256": _digest_sequence(overlap),
        },
    }


def observe_drive_metadata(specification: DriveFile, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        specification.download_url,
        method="HEAD",
        headers={"User-Agent": "CLASP-public-data-freezer/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        length = int(response.headers["Content-Length"])
        observed = {
            "http_status": response.status,
            "content_length": length,
            "content_disposition": response.headers.get("Content-Disposition"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    if length != specification.byte_length:
        raise ValueError(
            f"Drive size mismatch for {specification.file_id}: "
            f"expected {specification.byte_length}, got {length}"
        )
    return observed


def acquire_create_once(
    specification: DriveFile,
    destination_root: Path,
    *,
    timeout: int = 300,
) -> tuple[Path, str]:
    """Acquire one public RL file without ever replacing an existing target."""

    target = destination_root / specification.relative_path
    expected_sha256 = RL_SHA256_BY_PATH.get(specification.relative_path)
    if expected_sha256 is None:
        raise ValueError(f"No frozen byte hash for RL item: {specification.relative_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.stat().st_size != specification.byte_length:
            raise ValueError(f"Existing target does not match frozen size: {target}")
        actual_sha256, _ = sha256_regular_file_stable(target)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Existing target does not match frozen SHA256: {target}: {actual_sha256}"
            )
        return target, "validated_existing"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.partial.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = urllib.request.Request(
                specification.download_url,
                headers={"User-Agent": "CLASP-public-data-freezer/1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != specification.byte_length:
            raise ValueError(
                f"Downloaded size mismatch for {specification.file_id}: "
                f"expected {specification.byte_length}, got {temporary.stat().st_size}"
            )
        actual_sha256, _ = sha256_regular_file_stable(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded SHA256 mismatch for {specification.file_id}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            os.link(temporary, target)
            outcome = "created"
        except FileExistsError:
            if not target.is_file() or target.stat().st_size != specification.byte_length:
                raise ValueError(f"Concurrent target does not match frozen size: {target}")
            concurrent_sha256, _ = sha256_regular_file_stable(target)
            if concurrent_sha256 != expected_sha256:
                raise ValueError(f"Concurrent target does not match frozen SHA256: {target}")
            outcome = "validated_concurrent_existing"
        return target, outcome
    finally:
        temporary.unlink(missing_ok=True)


def drive_file_projection(specification: DriveFile) -> dict[str, Any]:
    return {
        "task": specification.task,
        "split": specification.split,
        "file_id": specification.file_id,
        "filename": specification.filename,
        "relative_path": specification.relative_path,
        "content_length": specification.byte_length,
        "last_modified": specification.last_modified,
    }


@dataclass
class _IdentityProjectionV2:
    summary: dict[str, Any]
    row_hashes: list[bytes]
    references_by_instance: dict[bytes, bytes | set[bytes]]


def canonical_rendered_input(task: str, rendered_input: str) -> dict[str, str]:
    """Return the one cross-stage identity projection used by schema v2."""

    if task not in PUBLICATION_TASKS:
        raise ValueError(f"Task is outside the publication intersection: {task}")
    normalized = unicodedata.normalize("NFC", rendered_input)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return {"task": task, "rendered_input": normalized}


def _add_reference(
    mapping: dict[bytes, bytes | set[bytes]],
    instance_hash: bytes,
    reference_hash: bytes,
) -> None:
    previous = mapping.get(instance_hash)
    if previous is None:
        mapping[instance_hash] = reference_hash
    elif isinstance(previous, bytes):
        if previous != reference_hash:
            mapping[instance_hash] = {previous, reference_hash}
    else:
        previous.add(reference_hash)


def analyze_identity_projection_v2(
    path: Path,
    *,
    task: str,
    require_read_only: bool = True,
) -> _IdentityProjectionV2:
    """Parse all rows using the task-aware rendered-input identity."""

    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Identity source is not a non-symlink regular file: {path}")
    if require_read_only and before.st_mode & 0o222:
        raise ValueError(f"Identity source is writable rather than sealed: {path}")
    raw_digest = hashlib.sha256()
    ordered_instances = hashlib.sha256()
    ordered_rows = hashlib.sha256()
    instance_hashes: list[bytes] = []
    row_hashes: list[bytes] = []
    references: dict[bytes, bytes | set[bytes]] = {}
    row_count = 0
    for row_index, row in enumerate(iter_json_array(path, raw_digest)):
        if not isinstance(row, dict):
            raise ValueError(f"Row {row_index} in {path} is not an object")
        if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), str):
            raise ValueError(f"Row {row_index} in {path} lacks string input/output fields")
        instance_hash = hashlib.sha256(
            canonical_json(canonical_rendered_input(task, row["input"]))
        ).digest()
        row_hash = hashlib.sha256(canonical_json(row)).digest()
        reference_hash = hashlib.sha256(canonical_json(row["output"])).digest()
        instance_hashes.append(instance_hash)
        row_hashes.append(row_hash)
        ordered_instances.update(instance_hash)
        ordered_rows.update(row_hash)
        _add_reference(references, instance_hash, reference_hash)
        row_count += 1
    if row_count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    after = os.lstat(path)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"Identity source changed while it was parsed: {path}")
    sorted_instances = sorted(instance_hashes)
    sorted_rows = sorted(row_hashes)
    return _IdentityProjectionV2(
        summary={
            "raw_sha256": raw_digest.hexdigest(),
            "row_count": row_count,
            "instance_identity": {
                "definition": (
                    "sha256(canonical_json({'task': task, 'rendered_input': "
                    "NFC(input) with every Unicode whitespace run collapsed to one space}))"
                ),
                "ordered_sha256": ordered_instances.hexdigest(),
                "multiset_sha256": _digest_sequence(sorted_instances),
                "unique_set_sha256": _digest_sequence(instance_hashes, unique=True),
                "unique_count": len(references),
                "duplicate_count": row_count - len(references),
            },
            "exact_row_identity": {
                "definition": "sha256(canonical_json(full_row))",
                "ordered_sha256": ordered_rows.hexdigest(),
                "multiset_sha256": _digest_sequence(sorted_rows),
                "unique_count": len(set(row_hashes)),
                "duplicate_count": row_count - len(set(row_hashes)),
            },
        },
        row_hashes=sorted_rows,
        references_by_instance=references,
    )


def _reference_set(value: bytes | set[bytes]) -> set[bytes]:
    return {value} if isinstance(value, bytes) else value


def _sorted_multiset_intersection(left: list[bytes], right: list[bytes]) -> tuple[int, int, str]:
    left_index = right_index = 0
    multiset_count = 0
    unique_count = 0
    digest = hashlib.sha256()
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            left_index += 1
            continue
        if right[right_index] < left[left_index]:
            right_index += 1
            continue
        value = left[left_index]
        left_end = left_index + 1
        right_end = right_index + 1
        while left_end < len(left) and left[left_end] == value:
            left_end += 1
        while right_end < len(right) and right[right_end] == value:
            right_end += 1
        multiset_count += min(left_end - left_index, right_end - right_index)
        unique_count += 1
        digest.update(value)
        left_index, right_index = left_end, right_end
    return unique_count, multiset_count, digest.hexdigest()


def compare_identity_projections_v2(
    left: _IdentityProjectionV2,
    right: _IdentityProjectionV2,
) -> dict[str, Any]:
    """Separate exact rows from shared instances and reference agreement."""

    left_references = left.references_by_instance
    right_references = right.references_by_instance
    smaller, larger = (
        (left_references, right_references)
        if len(left_references) <= len(right_references)
        else (right_references, left_references)
    )
    common_instances = sorted(identity for identity in smaller if identity in larger)
    shared_reference_count = 0
    disjoint_reference_count = 0
    for identity in common_instances:
        if _reference_set(left_references[identity]) & _reference_set(right_references[identity]):
            shared_reference_count += 1
        else:
            disjoint_reference_count += 1
    exact_unique, exact_multiset, exact_digest = _sorted_multiset_intersection(
        left.row_hashes, right.row_hashes
    )
    instance_digest = hashlib.sha256()
    for identity in common_instances:
        instance_digest.update(identity)
    return {
        "shared_unique_instance_count": len(common_instances),
        "shared_unique_instance_sha256": instance_digest.hexdigest(),
        "shared_instance_with_any_identical_reference_count": shared_reference_count,
        "shared_instance_with_disjoint_reference_sets_count": disjoint_reference_count,
        "exact_duplicate_unique_row_count": exact_unique,
        "exact_duplicate_row_multiset_count": exact_multiset,
        "exact_duplicate_unique_row_sha256": exact_digest,
    }


def _compare_training_union_to_eval_v2(
    training: tuple[_IdentityProjectionV2, ...],
    evaluation: _IdentityProjectionV2,
) -> dict[str, Any]:
    shared_instances: list[bytes] = []
    shared_reference = 0
    disjoint_reference = 0
    for identity, eval_references in evaluation.references_by_instance.items():
        matching = [
            source.references_by_instance[identity]
            for source in training
            if identity in source.references_by_instance
        ]
        if not matching:
            continue
        shared_instances.append(identity)
        training_references: set[bytes] = set()
        for value in matching:
            training_references.update(_reference_set(value))
        if training_references & _reference_set(eval_references):
            shared_reference += 1
        else:
            disjoint_reference += 1
    shared_instances.sort()
    instance_digest = hashlib.sha256()
    for identity in shared_instances:
        instance_digest.update(identity)

    exact_eval_rows: set[bytes] = set()
    for row_hash in evaluation.row_hashes:
        if any(
            (index := bisect_left(source.row_hashes, row_hash)) < len(source.row_hashes)
            and source.row_hashes[index] == row_hash
            for source in training
        ):
            exact_eval_rows.add(row_hash)
    exact_digest = hashlib.sha256()
    for row_hash in sorted(exact_eval_rows):
        exact_digest.update(row_hash)
    return {
        "shared_unique_instance_count": len(shared_instances),
        "shared_unique_instance_sha256": instance_digest.hexdigest(),
        "shared_instance_with_any_identical_reference_count": shared_reference,
        "shared_instance_with_disjoint_reference_sets_count": disjoint_reference,
        "exact_duplicate_unique_eval_row_count": len(exact_eval_rows),
        "exact_duplicate_unique_eval_row_sha256": exact_digest.hexdigest(),
    }


def analyze_task_identity_matrix_v2(
    task: str,
    rl_train: Path,
    rl_eval: Path,
    sft_train: Path,
    sft_eval: Path,
) -> dict[str, Any]:
    """Build every within-stage and cross-stage split comparison for one task."""

    projections = {
        "rl_train": analyze_identity_projection_v2(rl_train, task=task),
        "rl_eval": analyze_identity_projection_v2(rl_eval, task=task),
        "sft_train": analyze_identity_projection_v2(sft_train, task=task, require_read_only=False),
        "sft_eval": analyze_identity_projection_v2(sft_eval, task=task, require_read_only=False),
    }
    within = {
        "rl_train__rl_eval": compare_identity_projections_v2(
            projections["rl_train"], projections["rl_eval"]
        ),
        "sft_train__sft_eval": compare_identity_projections_v2(
            projections["sft_train"], projections["sft_eval"]
        ),
    }
    cross = {
        left + "__" + right: compare_identity_projections_v2(projections[left], projections[right])
        for left in ("sft_train", "sft_eval")
        for right in ("rl_train", "rl_eval")
    }
    author_training = (projections["sft_train"], projections["rl_train"])
    return {
        "splits": {name: projection.summary for name, projection in projections.items()},
        "within_stage": within,
        "cross_stage": cross,
        "author_training_union_to_evaluation": {
            "sft_train_union_rl_train__rl_eval": _compare_training_union_to_eval_v2(
                author_training, projections["rl_eval"]
            ),
            "sft_train_union_rl_train__sft_eval": _compare_training_union_to_eval_v2(
                author_training, projections["sft_eval"]
            ),
        },
        "raw_file_equality": {
            "rl_eval_equals_sft_eval": (
                projections["rl_eval"].summary["raw_sha256"]
                == projections["sft_eval"].summary["raw_sha256"]
            ),
            "rl_train_equals_sft_train": (
                projections["rl_train"].summary["raw_sha256"]
                == projections["sft_train"].summary["raw_sha256"]
            ),
        },
    }


def self_hash(value: dict[str, Any], field: str = "manifest_sha256") -> str:
    payload = dict(value)
    payload[field] = None
    return hashlib.sha256(canonical_json(payload)).hexdigest()
