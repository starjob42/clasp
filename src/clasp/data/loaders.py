"""Dataset loading, validation, and the mixed-task batch sampler.

Loads per-task JSON from ``<data_root>/<task>/<train|eval>/*.json``, validates that
``instruction``, ``input``, and ``output`` are non-empty, deduplicates exact duplicate rows,
and removes train/eval overlap within each task. The structured task instance is read from
``instance_json``, falling back to ``instance``.

Also provides the mixed-batch sampler that guarantees each local batch contains more than
one task type -- which is why multi-task runs require
``per_device_train_batch_size >= 2``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict, deque
from typing import Iterator

from datasets import Dataset, DatasetDict, concatenate_datasets  # type: ignore[import-not-found]
from torch.utils.data import Sampler  # type: ignore[import-not-found]

from clasp.data.prompts import build_formatted_prompt
from clasp.runtime import env_rank, env_world_size, print_primary


def coerce_json_string(value) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except Exception:
            return json.dumps(value, ensure_ascii=True)
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return json.dumps(str(value), ensure_ascii=True)


def read_json_rows(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {file_path}")
    return data


def list_split_json_files(
    data_root: str,
    task: str,
    split: str,
    *,
    prefer_subset_files: bool = True,
    required: bool = True,
) -> list[str]:
    split_dir = os.path.join(data_root, task, split)
    if not os.path.isdir(split_dir):
        if required:
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        return []

    candidates = sorted(
        os.path.join(split_dir, name) for name in os.listdir(split_dir) if name.endswith(".json")
    )
    if not candidates:
        if required:
            raise FileNotFoundError(f"No .json files found under: {split_dir}")
        return []

    if prefer_subset_files:
        by_name = {os.path.basename(path): path for path in candidates}
        preferred_names = (
            ("train_subset.json", "train_small.json")
            if split == "train"
            else ("eval_subset.json", "test_subset.json", "eval_small.json", "test_small.json")
        )
        for name in preferred_names:
            if name in by_name:
                return [by_name[name]]

        subset_candidates = [
            path for path in candidates if os.path.basename(path).endswith("_subset.json")
        ]
        if subset_candidates:
            return subset_candidates

    return candidates


def find_split_json(
    data_root: str,
    task: str,
    split: str,
    *,
    prefer_subset_files: bool = True,
    required: bool = True,
) -> str | None:
    files = list_split_json_files(
        data_root,
        task,
        split,
        prefer_subset_files=prefer_subset_files,
        required=required,
    )
    return files[0] if files else None


def stable_row_signature(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def stable_instance_signature(row: dict) -> str:
    """Hash problem identity independently of labels and formatting.

    Full-row hashes miss leakage when the same optimization instance has a
    different solver output, instruction, or JSON whitespace.  Prefer the
    structured instance payload and fall back to ``input`` only when no usable
    structured value is present.  Normalized legacy rows encode a missing
    instance as the JSON string ``"null"``; treat that as missing rather than
    collapsing every such row onto one signature.
    """

    instance = None
    for key in ("instance_json", "instance", "input"):
        if key not in row:
            continue
        raw_value = row.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                continue
            try:
                parsed_value = json.loads(stripped)
            except Exception:
                parsed_value = raw_value
        else:
            parsed_value = raw_value
        if parsed_value is None and key != "input":
            continue
        instance = parsed_value
        break

    payload = json.dumps(
        instance,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_rows(rows: list[dict], json_file: str) -> list[dict]:
    normalized_rows = []
    invalid_rows: list[dict[str, object]] = []
    invalid_count = 0
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_count += 1
            if len(invalid_rows) < 5:
                invalid_rows.append(
                    {
                        "row_index": row_idx,
                        "issue": f"expected dict, found {type(row).__name__}",
                    }
                )
            continue
        instruction = str(row.get("instruction", ""))
        input_text = str(row.get("input", ""))
        output_text = str(row.get("output", ""))

        missing_fields = []
        if not instruction.strip():
            missing_fields.append("instruction")
        if not input_text.strip():
            missing_fields.append("input")
        if not output_text.strip():
            missing_fields.append("output")

        if missing_fields:
            invalid_count += 1
            if len(invalid_rows) < 5:
                invalid_rows.append(
                    {
                        "row_index": row_idx,
                        "missing_fields": list(missing_fields),
                    }
                )
            continue

        instance_source = row.get("instance_json", row.get("instance"))
        normalized_row = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "instance_json": coerce_json_string(instance_source),
        }
        for optional_key in (
            "reference_imitation_output",
            "imitation_output",
            "negative_imitation_output",
        ):
            optional_value = row.get(optional_key)
            if optional_value is not None:
                normalized_row[optional_key] = str(optional_value)
        normalized_rows.append(normalized_row)

    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} invalid row(s) in {json_file}. "
            "Every example must contain non-empty 'instruction', 'input', and 'output' fields. "
            f"Examples: {invalid_rows!r}"
        )

    if not normalized_rows:
        raise ValueError(f"No valid dict rows found in JSON dataset: {json_file}")
    return normalized_rows


def deduplicate_rows(rows: list[dict], *, label: str) -> list[dict]:
    unique_rows = []
    seen_signatures: set[str] = set()
    dropped = 0
    for row in rows:
        signature = stable_row_signature(row)
        if signature in seen_signatures:
            dropped += 1
            continue
        seen_signatures.add(signature)
        unique_rows.append(row)

    if dropped > 0:
        print_primary(f"[DataLoad][Warn] Removed {dropped} duplicate row(s) while loading {label}.")

    return unique_rows


def dataset_row_signatures(dataset: Dataset) -> set[str]:
    return {stable_row_signature(dict(row)) for row in dataset}


def dataset_instance_signatures(dataset: Dataset) -> set[str]:
    return {stable_instance_signature(dict(row)) for row in dataset}


def ensure_no_instance_overlap(
    *,
    task: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Dataset:
    """Remove train rows whose underlying instance occurs in external eval."""

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        return train_dataset

    eval_signatures = dataset_instance_signatures(eval_dataset)

    def is_not_in_eval(row):
        return stable_instance_signature(dict(row)) not in eval_signatures

    initial_len = len(train_dataset)
    cleaned_train = train_dataset.filter(
        is_not_in_eval,
        desc=f"Decontaminating {task} train against external eval instances",
    )
    overlap_count = initial_len - len(cleaned_train)
    if overlap_count > 0:
        print_primary(
            f"[DataLoad][Warn] Removed {overlap_count} {task} train row(s) whose "
            "optimization instance occurs in the external eval split."
        )
    return cleaned_train


def ensure_no_split_overlap(*, task: str, train_dataset: Dataset, eval_dataset: Dataset) -> Dataset:
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        return train_dataset

    eval_signatures = dataset_row_signatures(eval_dataset)

    def is_not_in_eval(row):
        return stable_row_signature(dict(row)) not in eval_signatures

    initial_len = len(train_dataset)
    cleaned_train = train_dataset.filter(
        is_not_in_eval, desc=f"Deduplicating {task} train against eval"
    )
    overlap_count = initial_len - len(cleaned_train)

    if overlap_count > 0:
        print_primary(
            f"[DataLoad][Warn] Detected and removed {overlap_count} overlapping example(s) "
            f"from {task} train split to prevent data leakage."
        )

    return cleaned_train


def manual_dataset_from_rows(rows: list[dict], json_file: str) -> Dataset:
    normalized_rows = normalize_rows(rows, json_file)
    return Dataset.from_list(normalized_rows)


def load_task_dataset(json_file: str) -> Dataset:
    print_primary(f"[DataLoad] Using manual JSON loader for {json_file}")
    return manual_dataset_from_rows(read_json_rows(json_file), json_file)


def load_and_merge_task_datasets(
    json_files: list[str],
    *,
    deduplicate_exact_rows: bool = True,
) -> Dataset:
    if not json_files:
        raise ValueError("Expected at least one JSON file to load a dataset split.")

    merged_rows = []
    for json_file in json_files:
        merged_rows.extend(normalize_rows(read_json_rows(json_file), json_file))

    if not deduplicate_exact_rows:
        return Dataset.from_list(merged_rows)
    deduplicated_rows = deduplicate_rows(merged_rows, label=summarize_json_sources(json_files))
    return Dataset.from_list(deduplicated_rows)


def summarize_json_sources(json_files: list[str]) -> str:
    if not json_files:
        return "none"
    if len(json_files) == 1:
        return json_files[0]
    directory = os.path.dirname(json_files[0])
    return f"{len(json_files)} files under {directory}"


def shuffle_and_cap_dataset(dataset: Dataset, seed: int, cap: int) -> Dataset:
    shuffled = dataset.shuffle(seed=seed)
    if cap <= 0:
        return shuffled
    keep = min(len(shuffled), int(cap))
    return shuffled.select(range(keep))


def cap_dataset_preserving_order(dataset: Dataset, cap: int) -> Dataset:
    """Apply an explicit row cap without perturbing an already shuffled split."""

    if cap <= 0 or len(dataset) <= int(cap):
        return dataset
    return dataset.select(range(int(cap)))


def parse_instance_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _env_positive_int(name: str, default: int = 0) -> int:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return int(default)
    try:
        value = int(raw_value)
    except ValueError:
        return int(default)
    return max(0, value)


def _tsp_prompt_knn_k() -> int:
    return min(_env_positive_int("CLASP_TSP_PROMPT_KNN", 0), 16)


def _tsp_prompt_knn_include_distances() -> bool:
    raw_value = str(os.environ.get("CLASP_TSP_PROMPT_KNN_DISTANCES", "") or "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _tsp_input_order_mode() -> str:
    mode = str(os.environ.get("CLASP_TSP_INPUT_ORDER", "") or "").strip().lower()
    if mode in {"", "none", "default", "original", "0", "false"}:
        return ""
    if mode in {"morton", "xy", "yx"}:
        return mode
    return ""


def _tsp_relabel_order_mode() -> str:
    mode = str(os.environ.get("CLASP_TSP_RELABEL_ORDER", "") or "").strip().lower()
    if mode in {"", "none", "default", "original", "0", "false"}:
        return ""
    if mode in {"morton", "xy", "yx"}:
        return mode
    return ""


def _tsp_route_canonicalize_mode() -> str:
    mode = str(os.environ.get("CLASP_TSP_ROUTE_CANONICALIZE", "") or "").strip().lower()
    if mode in {"", "none", "default", "original", "0", "false"}:
        return ""
    if mode in {"1", "true", "yes", "start0", "start0_minlex"}:
        return "start0_minlex"
    return ""


def _tsp_route_only_output_enabled() -> bool:
    return _env_truthy("CLASP_TSP_ROUTE_ONLY_OUTPUT")


def tsp_label_tta_schedule_from_env() -> list[str]:
    raw_value = str(os.environ.get("CLASP_TSP_LABEL_TTA_SCHEDULE", "") or "").strip()
    if not raw_value:
        return []
    schedule = [piece.strip().lower() for piece in raw_value.split(",") if piece.strip()]
    return schedule


def _coords_from_tsp_instance(instance) -> list[list[float]]:
    if isinstance(instance, list):
        if all(
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
            for item in instance
        ):
            return [[float(item[0]), float(item[1])] for item in instance]
    if isinstance(instance, dict):
        for key in ("coords", "coordinates", "nodes", "locations"):
            value = instance.get(key)
            if isinstance(value, list):
                coords = _coords_from_tsp_instance(value)
                if coords:
                    return coords
    return []


def _coords_from_tsp_text(input_text: str) -> list[list[float]]:
    coords: dict[int, list[float]] = {}
    for match in re.finditer(
        r"Node\s+(\d+)\s*,\s*coordinates:\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
        str(input_text),
        flags=re.IGNORECASE,
    ):
        try:
            idx = int(match.group(1))
            coords[idx] = [float(match.group(2)), float(match.group(3))]
        except ValueError:
            continue
    if not coords:
        return []
    ordered_indices = sorted(coords)
    if ordered_indices != list(range(len(ordered_indices))):
        return []
    return [coords[idx] for idx in ordered_indices]


def _format_tsp_knn_hints(
    coords: list[list[float]],
    k: int,
    *,
    include_distances: bool = False,
) -> str:
    neighbor_count = min(max(0, int(k)), max(0, len(coords) - 1))
    if neighbor_count <= 0:
        return ""
    parts: list[str] = []
    for idx, (x_i, y_i) in enumerate(coords):
        distances: list[tuple[float, int]] = []
        for other_idx, (x_j, y_j) in enumerate(coords):
            if other_idx == idx:
                continue
            distance = math.hypot(float(x_i) - float(x_j), float(y_i) - float(y_j))
            distances.append((distance, other_idx))
        nearest = sorted(distances)[:neighbor_count]
        if include_distances:
            neighbors = ", ".join(
                f"({other_idx}, {distance:.4f})" for distance, other_idx in nearest
            )
        else:
            neighbors = ", ".join(str(other_idx) for _distance, other_idx in nearest)
        parts.append(f"{idx}: [{neighbors}]")
    if include_distances:
        prefix = "Additional nearest-neighbor node IDs with Euclidean distances"
    else:
        prefix = "Additional nearest-neighbor node IDs"
    return (
        f"{prefix} (computed from coordinates; {neighbor_count} per node, hints only): "
        + "; ".join(parts)
        + "."
    )


def _morton_code(x_value: float, y_value: float, *, bits: int = 16) -> int:
    scale = (1 << bits) - 1
    x_int = max(0, min(scale, int(round(float(x_value) * scale))))
    y_int = max(0, min(scale, int(round(float(y_value) * scale))))
    code = 0
    for bit in range(bits):
        code |= ((x_int >> bit) & 1) << (2 * bit)
        code |= ((y_int >> bit) & 1) << (2 * bit + 1)
    return code


def _normalize_coords_for_order(
    coords: list[tuple[int, float, float]],
) -> list[tuple[int, float, float, float, float]]:
    xs = [x for _idx, x, _y in coords]
    ys = [y for _idx, _x, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    return [(idx, x, y, (x - min_x) / span_x, (y - min_y) / span_y) for idx, x, y in coords]


def _sort_tsp_node_segments(input_text: str, mode: str) -> str:
    text = str(input_text or "")
    if not mode:
        return text

    segments = [segment.strip() for segment in text.split(";") if segment.strip()]
    if len(segments) < 3:
        return text

    parsed_segments: list[tuple[int, float, float, str]] = []
    for segment in segments:
        match = re.match(
            r"^Node\s+(\d+)\s*,\s*coordinates:\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
            segment,
            flags=re.IGNORECASE,
        )
        if not match:
            return text
        try:
            parsed_segments.append(
                (
                    int(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    segment,
                )
            )
        except ValueError:
            return text

    normalized = _normalize_coords_for_order(
        [(idx, x, y) for idx, x, y, _segment in parsed_segments]
    )
    normalized_by_idx = {idx: (norm_x, norm_y) for idx, _x, _y, norm_x, norm_y in normalized}

    ParsedSegment = tuple[int, float, float, str]

    if mode == "xy":

        def key_fn(item: ParsedSegment) -> tuple[float, ...]:
            """Sort by x, then y, then original index."""
            return (item[1], item[2], item[0])

    elif mode == "yx":

        def key_fn(item: ParsedSegment) -> tuple[float, ...]:
            """Sort by y, then x, then original index."""
            return (item[2], item[1], item[0])

    else:

        def key_fn(item: ParsedSegment) -> tuple[float, ...]:
            """Sort by Morton (Z-order) code of the normalized coordinates."""
            return (_morton_code(*normalized_by_idx[item[0]]), item[0])

    ordered_segments = [segment for _idx, _x, _y, segment in sorted(parsed_segments, key=key_fn)]
    suffix = ";" if text.rstrip().endswith(";") else ""
    return ";".join(ordered_segments) + suffix


def _format_coord_value(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return str(int(rounded))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _format_tsp_input_from_coords(coords: list[list[float]], *, neighbor_count: int = 2) -> str:
    if not coords:
        return ""
    k = min(max(0, int(neighbor_count)), max(0, len(coords) - 1))
    segments: list[str] = []
    for idx, (x_i, y_i) in enumerate(coords):
        distances: list[tuple[float, int]] = []
        for other_idx, (x_j, y_j) in enumerate(coords):
            if other_idx == idx:
                continue
            distances.append(
                (
                    math.hypot(float(x_i) - float(x_j), float(y_i) - float(y_j)),
                    other_idx,
                )
            )
        nearest = sorted(distances)[:k]
        neighbor_text = ", ".join(f"{other_idx}: {distance:.1f}" for distance, other_idx in nearest)
        segments.append(
            f"Node {idx}, coordinates: [{_format_coord_value(x_i)}, {_format_coord_value(y_i)}], "
            f"neighbors: [{neighbor_text}]"
        )
    return ";".join(segments) + "."


def _parse_tsp_route_labels(output_text: str) -> list[int] | None:
    match = re.search(r"Route\s*:\s*\[([^\]]*)\]", str(output_text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    labels: list[int] = []
    for value in re.findall(r"-?\d+", match.group(1)):
        try:
            labels.append(int(value))
        except ValueError:
            return None
    return labels if labels else None


def _remap_tsp_output_route(output_text: str, old_to_new: dict[int, int]) -> str:
    route = _parse_tsp_route_labels(output_text)
    if not route:
        return str(output_text or "")
    if any(label not in old_to_new for label in route):
        return str(output_text or "")

    remapped = [old_to_new[label] for label in route]
    return _format_tsp_output_route(str(output_text or ""), remapped)


def _format_tsp_output_route(output_text: str, route: list[int]) -> str:
    route_text = ", ".join(str(label) for label in route)
    objective_match = re.search(
        r"Objective\s*:\s*([-+]?\d+(?:\.\d+)?)",
        str(output_text or ""),
        flags=re.IGNORECASE,
    )
    if objective_match:
        return f"Route: [{route_text}], Objective: {objective_match.group(1)}"
    return f"Route: [{route_text}]"


def _format_tsp_route_only_output(output_text: str) -> str:
    route = _parse_tsp_route_labels(output_text)
    if not route:
        return str(output_text or "")
    route_text = ", ".join(str(label) for label in route)
    return f"Route: [{route_text}]"


def _route_only_tsp_instruction(instruction_text: str) -> str:
    if not _tsp_route_only_output_enabled():
        return str(instruction_text or "")
    text = str(instruction_text or "")
    marker = "Route-only output requirement:"
    if marker in text:
        return text
    suffix = (
        "\n\n"
        f"{marker} return only the route in the format `Route: [node_ids]`. "
        "Do not include an Objective field or any extra text."
    )
    return text.rstrip() + suffix


def _route_only_tsp_output(output_text: str) -> str:
    if not _tsp_route_only_output_enabled():
        return str(output_text or "")
    return _format_tsp_route_only_output(output_text)


def _canonicalize_tsp_output_route(output_text: str) -> str:
    if not _tsp_route_canonicalize_mode():
        return str(output_text or "")
    route = _parse_tsp_route_labels(output_text)
    if not route:
        return str(output_text or "")

    closed = len(route) > 1 and route[0] == route[-1]
    core = list(route[:-1] if closed else route)
    if 0 not in core or len(core) < 3:
        return str(output_text or "")

    start = core.index(0)
    forward = core[start:] + core[:start]
    backward = [0] + list(reversed(forward[1:]))
    canonical_core = min(forward, backward, key=lambda values: tuple(values[1:]))
    canonical_route = canonical_core + [0]
    return _format_tsp_output_route(str(output_text or ""), canonical_route)


def _canonicalize_tsp_instruction(instruction_text: str) -> str:
    if not _tsp_route_canonicalize_mode():
        return str(instruction_text or "")
    text = str(instruction_text or "")
    marker = "Route canonicalization requirement:"
    if marker in text:
        return text
    suffix = (
        "\n\n"
        f"{marker} return the TSP route starting and ending at node 0. "
        "If the same cycle can be written in two traversal directions, use the direction "
        "whose node sequence after node 0 is lexicographically smaller."
    )
    return text.rstrip() + suffix


def _tsp_relabel_order_indices(coords: list[list[float]], mode: str) -> list[int]:
    indexed = [(idx, float(coord[0]), float(coord[1])) for idx, coord in enumerate(coords)]
    if mode == "xy":
        return [
            idx for idx, _x, _y in sorted(indexed, key=lambda item: (item[1], item[2], item[0]))
        ]
    if mode == "yx":
        return [
            idx for idx, _x, _y in sorted(indexed, key=lambda item: (item[2], item[1], item[0]))
        ]

    normalized = _normalize_coords_for_order(indexed)
    normalized_by_idx = {idx: (norm_x, norm_y) for idx, _x, _y, norm_x, norm_y in normalized}
    return [
        idx
        for idx, _x, _y in sorted(
            indexed,
            key=lambda item: (_morton_code(*normalized_by_idx[item[0]]), item[0]),
        )
    ]


def _tsp_label_tta_start_index(coords: list[list[float]], variant: str) -> int | None:
    if len(coords) < 3:
        return None
    indexed = [(idx, float(coord[0]), float(coord[1])) for idx, coord in enumerate(coords)]
    if variant in {"swap:minx", "start:minx"}:
        return min(indexed, key=lambda item: (item[1], item[2], item[0]))[0]
    if variant in {"swap:maxx", "start:maxx"}:
        return max(indexed, key=lambda item: (item[1], -item[2], -item[0]))[0]
    if variant in {"swap:miny", "start:miny"}:
        return min(indexed, key=lambda item: (item[2], item[1], item[0]))[0]
    if variant in {"swap:maxy", "start:maxy"}:
        return max(indexed, key=lambda item: (item[2], -item[1], -item[0]))[0]
    if variant in {"swap:center", "start:center"}:
        xs = [item[1] for item in indexed]
        ys = [item[2] for item in indexed]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        return min(
            indexed,
            key=lambda item: (
                math.hypot(item[1] - center_x, item[2] - center_y),
                item[0],
            ),
        )[0]
    if variant in {"swap:farthest0", "start:farthest0"}:
        x0, y0 = float(coords[0][0]), float(coords[0][1])
        return max(
            indexed,
            key=lambda item: (math.hypot(item[1] - x0, item[2] - y0), -item[0]),
        )[0]
    if variant in {"swap:nearest0", "swap:nn0", "start:nearest0", "start:nn0"}:
        x0, y0 = float(coords[0][0]), float(coords[0][1])
        candidates = [item for item in indexed if item[0] != 0]
        if not candidates:
            return 0
        return min(
            candidates,
            key=lambda item: (math.hypot(item[1] - x0, item[2] - y0), item[0]),
        )[0]
    if variant.startswith("swap:") or variant.startswith("start:"):
        _, raw_idx = variant.split(":", 1)
        if raw_idx.isdigit():
            idx = int(raw_idx)
            if 0 <= idx < len(coords):
                return idx
    return None


def _tsp_label_tta_order_indices(coords: list[list[float]], variant: str) -> list[int] | None:
    variant = str(variant or "").strip().lower()
    if variant in {"", "identity", "default", "original", "none", "0", "false"}:
        return list(range(len(coords)))
    if variant in {"morton", "xy", "yx"}:
        return _tsp_relabel_order_indices(coords, variant)

    start_idx = _tsp_label_tta_start_index(coords, variant)
    if start_idx is None:
        return None

    order = list(range(len(coords)))
    if variant.startswith("start:"):
        order.pop(start_idx)
        return [start_idx] + order

    order[0], order[start_idx] = order[start_idx], order[0]
    return order


def _env_truthy(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _mis_prompt_node_degrees_enabled() -> bool:
    return _env_truthy("CLASP_MIS_PROMPT_NODE_DEGREES")


def _mis_jiang_response_output_enabled() -> bool:
    """Use the literal MIS field consumed by Jiang's released BoN selector.

    The released SFT labels say ``Set: [...]`` but the released
    ``optimality_reward_func_mis`` selects candidates only when it sees
    ``Response: [...]``.  This opt-in changes the learned output grammar at
    training/prompt construction time; it does not rewrite generated text.
    """

    return _env_truthy("CLASP_MIS_JIANG_RESPONSE_OUTPUT")


def _canonicalize_mis_jiang_response_output(
    instruction: str,
    output_text: str,
) -> tuple[str, str]:
    if not _mis_jiang_response_output_enabled():
        return str(instruction or ""), str(output_text or "")

    instruction_value = re.sub(
        r"\bSet\s*:",
        "Response:",
        str(instruction or ""),
        count=1,
        flags=re.IGNORECASE,
    )
    output_value = re.sub(
        r"\bSet\s*:",
        "Response:",
        str(output_text or ""),
        count=1,
        flags=re.IGNORECASE,
    )
    return instruction_value, output_value


def _mis_prompt_low_degree_k() -> int:
    raw = str(os.environ.get("CLASP_MIS_PROMPT_LOW_DEGREE_K", "32") or "").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 32


def _mis_relabel_order_mode() -> str:
    mode = str(os.environ.get("CLASP_MIS_RELABEL_ORDER", "") or "").strip().lower()
    if mode in {"", "none", "default", "original", "0", "false"}:
        return ""
    if mode in {"degree_asc", "degree", "low_degree_first"}:
        return "degree_asc"
    if mode in {"degree_desc", "high_degree_first"}:
        return "degree_desc"
    return ""


def _mis_num_nodes_from_instance(instance) -> int | None:
    parsed = parse_instance_value(instance)
    if isinstance(parsed, dict):
        for key in ("num_nodes", "n"):
            value = parsed.get(key)
            if isinstance(value, int):
                return int(value)
        nodes = parsed.get("nodes")
        if isinstance(nodes, list):
            return int(len(nodes))
    return None


def _mis_edges_from_instance(instance) -> list[tuple[int, int]]:
    parsed = parse_instance_value(instance)
    edges = parsed.get("edges") if isinstance(parsed, dict) else None
    if not isinstance(edges, list):
        return []

    parsed_edges: list[tuple[int, int]] = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            a = int(edge[0])
            b = int(edge[1])
        except (TypeError, ValueError):
            continue
        if a != b:
            parsed_edges.append((min(a, b), max(a, b)))
    return parsed_edges


def _mis_num_nodes_from_text(input_text: str) -> int | None:
    text = str(input_text or "")
    for pattern in (
        r"undirected graph with\s+(\d+)\s+nodes",
        r"graph with\s+(\d+)\s+nodes",
        r"\b(\d+)\s+nodes\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _mis_edges_from_text(input_text: str) -> list[tuple[int, int]]:
    text = str(input_text or "")
    parsed_edges: list[tuple[int, int]] = []
    for a_raw, b_raw in re.findall(r"\((\d+)\s*,\s*(\d+)\)", text):
        try:
            a = int(a_raw)
            b = int(b_raw)
        except ValueError:
            continue
        if a != b:
            parsed_edges.append((min(a, b), max(a, b)))
    return parsed_edges


def _format_mis_degree_hints(input_text: str, instance_json=None) -> str:
    num_nodes = _mis_num_nodes_from_instance(instance_json)
    edges = _mis_edges_from_instance(instance_json)
    if not edges:
        edges = _mis_edges_from_text(input_text)
    if num_nodes is None:
        num_nodes = _mis_num_nodes_from_text(input_text)
    if num_nodes is None and edges:
        num_nodes = max(max(a, b) for a, b in edges) + 1
    if num_nodes is None or num_nodes <= 0 or not edges:
        return ""

    degrees = [0 for _ in range(num_nodes)]
    for a, b in set(edges):
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            degrees[a] += 1
            degrees[b] += 1

    degree_pairs = ", ".join(f"{idx}:{degree}" for idx, degree in enumerate(degrees))
    low_k = min(_mis_prompt_low_degree_k(), num_nodes)
    low_degree_nodes = sorted(range(num_nodes), key=lambda idx: (degrees[idx], idx))[:low_k]
    low_degree_text = ", ".join(f"{idx}(deg={degrees[idx]})" for idx in low_degree_nodes)

    lines = [
        "MIS node-degree hints:",
        f"Degrees (node:degree): {degree_pairs}",
    ]
    if low_k > 0:
        lines.append(f"Lowest-degree nodes to consider first: [{low_degree_text}]")
    return "\n".join(lines)


def _mis_graph_from_instance_or_text(
    input_text: str, instance_json=None
) -> tuple[int | None, list[tuple[int, int]]]:
    num_nodes = _mis_num_nodes_from_instance(instance_json)
    edges = _mis_edges_from_instance(instance_json)
    if not edges:
        edges = _mis_edges_from_text(input_text)
    if num_nodes is None:
        num_nodes = _mis_num_nodes_from_text(input_text)
    if num_nodes is None and edges:
        num_nodes = max(max(a, b) for a, b in edges) + 1
    return num_nodes, edges


def _mis_relabel_order_indices(
    num_nodes: int, edges: list[tuple[int, int]], mode: str
) -> list[int]:
    degrees = [0 for _ in range(num_nodes)]
    for a, b in set(edges):
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            degrees[a] += 1
            degrees[b] += 1
    if mode == "degree_desc":
        return sorted(range(num_nodes), key=lambda idx: (-degrees[idx], idx))
    return sorted(range(num_nodes), key=lambda idx: (degrees[idx], idx))


def _format_mis_input_from_edges(edges: list[tuple[int, int]]) -> str:
    edge_text = ",".join(f"({a},{b})" for a, b in sorted(set(edges)))
    return f"Edges: [{edge_text}]"


def _remap_mis_output_set(output_text: str, old_to_new: dict[int, int]) -> str:
    text = str(output_text or "")

    def replace(match: re.Match) -> str:
        raw_values = re.findall(r"-?\d+", match.group(2))
        remapped = []
        for raw in raw_values:
            try:
                old_idx = int(raw)
            except ValueError:
                continue
            if old_idx in old_to_new:
                remapped.append(old_to_new[old_idx])
        remapped_text = ", ".join(str(node) for node in sorted(set(remapped)))
        return f"{match.group(1)}[{remapped_text}]"

    return re.sub(r"(Set\s*:\s*)\[(.*?)\]", replace, text, count=1, flags=re.IGNORECASE | re.DOTALL)


def _relabel_mis_example(
    input_text: str,
    output_text: str,
    instance_json=None,
) -> tuple[str, str, object]:
    mode = _mis_relabel_order_mode()
    if not mode:
        return str(input_text or ""), str(output_text or ""), instance_json

    num_nodes, edges = _mis_graph_from_instance_or_text(str(input_text or ""), instance_json)
    if num_nodes is None or num_nodes <= 0 or not edges:
        return str(input_text or ""), str(output_text or ""), instance_json

    order = _mis_relabel_order_indices(num_nodes, edges, mode)
    if order == list(range(num_nodes)):
        return str(input_text or ""), str(output_text or ""), instance_json

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    relabeled_edges = []
    for a, b in edges:
        if 0 <= a < num_nodes and 0 <= b < num_nodes and a in old_to_new and b in old_to_new:
            new_a = old_to_new[a]
            new_b = old_to_new[b]
            if new_a != new_b:
                relabeled_edges.append((min(new_a, new_b), max(new_a, new_b)))

    parsed_instance = parse_instance_value(instance_json)
    relabeled_instance = dict(parsed_instance) if isinstance(parsed_instance, dict) else {}
    relabeled_instance["num_nodes"] = int(num_nodes)
    relabeled_instance["edges"] = [[a, b] for a, b in sorted(set(relabeled_edges))]

    return (
        _format_mis_input_from_edges(relabeled_edges),
        _remap_mis_output_set(str(output_text or ""), old_to_new),
        relabeled_instance,
    )


def transform_tsp_example_for_label_tta_variant(
    instruction: str,
    input_text: str,
    output_text: str,
    instance_json=None,
    *,
    variant: str,
) -> dict[str, object]:
    instruction_value = str(instruction or "")
    input_value = str(input_text or "")
    output_value = str(output_text or "")
    transformed_instance = instance_json
    variant_value = str(variant or "").strip().lower()

    instance = parse_instance_value(instance_json)
    coords = _coords_from_tsp_instance(instance)
    if not coords:
        coords = _coords_from_tsp_text(input_value)
    order = _tsp_label_tta_order_indices(coords, variant_value) if len(coords) >= 3 else None
    if order and order != list(range(len(coords))):
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        relabeled_coords = [coords[old_idx] for old_idx in order]
        input_value = _format_tsp_input_from_coords(relabeled_coords)
        output_value = _remap_tsp_output_route(output_value, old_to_new)
        transformed_instance = relabeled_coords

    input_value = augment_input_for_task("tsp", input_value, transformed_instance)
    instruction_value = _route_only_tsp_instruction(instruction_value)
    output_value = _route_only_tsp_output(output_value)
    return {
        "instruction": instruction_value,
        "input": input_value,
        "output": output_value,
        "instance_json": transformed_instance,
        "tsp_label_tta_variant": variant_value or "identity",
    }


def _relabel_tsp_example(
    input_text: str,
    output_text: str,
    instance_json=None,
) -> tuple[str, str, object]:
    mode = _tsp_relabel_order_mode()
    if not mode:
        return str(input_text or ""), str(output_text or ""), instance_json

    instance = parse_instance_value(instance_json)
    coords = _coords_from_tsp_instance(instance)
    if not coords:
        coords = _coords_from_tsp_text(str(input_text or ""))
    if len(coords) < 3:
        return str(input_text or ""), str(output_text or ""), instance_json

    order = _tsp_relabel_order_indices(coords, mode)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    relabeled_coords = [coords[old_idx] for old_idx in order]
    relabeled_input = _format_tsp_input_from_coords(relabeled_coords)
    relabeled_output = _remap_tsp_output_route(str(output_text or ""), old_to_new)
    return relabeled_input, relabeled_output, relabeled_coords


def augment_input_for_task(task: str, input_text: str, instance_json=None) -> str:
    task_name = str(task).lower()
    input_value = str(input_text or "")
    if task_name == "mis":
        if not _mis_prompt_node_degrees_enabled() or "MIS node-degree hints:" in input_value:
            return input_value
        hints = _format_mis_degree_hints(input_value, instance_json)
        if not hints:
            return input_value
        return input_value.rstrip() + "\n\n" + hints

    if task_name != "tsp":
        return input_value

    input_value = _sort_tsp_node_segments(input_value, _tsp_input_order_mode())

    knn_k = _tsp_prompt_knn_k()
    if knn_k <= 0 or "Additional nearest-neighbor" in input_value:
        return input_value

    instance = parse_instance_value(instance_json)
    coords = _coords_from_tsp_instance(instance)
    if not coords:
        coords = _coords_from_tsp_text(input_value)
    if len(coords) < 3:
        return input_value

    hints = _format_tsp_knn_hints(
        coords,
        knn_k,
        include_distances=_tsp_prompt_knn_include_distances(),
    )
    if not hints:
        return input_value
    return input_value.rstrip() + "\n\n" + hints


def transform_example_for_task(
    task: str,
    instruction: str,
    input_text: str,
    output_text: str,
    instance_json=None,
) -> dict[str, object]:
    task_name = str(task).lower()
    instruction_value = str(instruction or "")
    input_value = str(input_text or "")
    output_value = str(output_text or "")
    transformed_instance = instance_json

    if task_name == "tsp":
        input_value, output_value, transformed_instance = _relabel_tsp_example(
            input_value,
            output_value,
            transformed_instance,
        )
        instruction_value = _canonicalize_tsp_instruction(instruction_value)
        output_value = _canonicalize_tsp_output_route(output_value)
        instruction_value = _route_only_tsp_instruction(instruction_value)
        output_value = _route_only_tsp_output(output_value)
    elif task_name == "mis":
        input_value, output_value, transformed_instance = _relabel_mis_example(
            input_value,
            output_value,
            transformed_instance,
        )
        instruction_value, output_value = _canonicalize_mis_jiang_response_output(
            instruction_value,
            output_value,
        )

    input_value = augment_input_for_task(task_name, input_value, transformed_instance)
    return {
        "instruction": instruction_value,
        "input": input_value,
        "output": output_value,
        "instance_json": transformed_instance,
    }


def active_tsp_representation_settings() -> dict[str, str]:
    return {
        "clasp_tsp_prompt_knn": str(os.environ.get("CLASP_TSP_PROMPT_KNN", "0")),
        "clasp_tsp_prompt_knn_distances": str(os.environ.get("CLASP_TSP_PROMPT_KNN_DISTANCES", "")),
        "clasp_tsp_input_order": str(os.environ.get("CLASP_TSP_INPUT_ORDER", "")),
        "clasp_tsp_relabel_order": str(os.environ.get("CLASP_TSP_RELABEL_ORDER", "")),
        "clasp_tsp_route_canonicalize": str(os.environ.get("CLASP_TSP_ROUTE_CANONICALIZE", "")),
        "clasp_tsp_route_only_output": str(os.environ.get("CLASP_TSP_ROUTE_ONLY_OUTPUT", "")),
        "clasp_tsp_label_tta_schedule": str(os.environ.get("CLASP_TSP_LABEL_TTA_SCHEDULE", "")),
        "clasp_mis_relabel_order": str(os.environ.get("CLASP_MIS_RELABEL_ORDER", "")),
        "clasp_mis_jiang_response_output": str(
            os.environ.get("CLASP_MIS_JIANG_RESPONSE_OUTPUT", "0")
        ),
        "clasp_mis_prompt_node_degrees": str(os.environ.get("CLASP_MIS_PROMPT_NODE_DEGREES", "0")),
        "clasp_mis_prompt_low_degree_k": str(os.environ.get("CLASP_MIS_PROMPT_LOW_DEGREE_K", "32")),
    }


def instance_size_from_text(task: str, input_text: str) -> int | None:
    task = str(task).lower()
    text = str(input_text or "")
    patterns_by_task = {
        "tsp": (r"\b(\d+)\s+(?:cities|nodes)\b",),
        "cvrp": (r"\b(\d+)\s+(?:customers|nodes)\b",),
        "op": (r"\b(\d+)\s+(?:locations|nodes)\b",),
        "mis": (r"graph\s+with\s+(\d+)\s+nodes", r"\b(\d+)\s+nodes\b"),
        "jssp": (
            r"\b(\d+)\s+jobs?\b.*?\b(\d+)\s+machines?\b",
            r"\b(\d+)\s+machines?\b.*?\b(\d+)\s+jobs?\b",
        ),
    }
    for pattern in patterns_by_task.get(task, (r"\b(\d+)\s+nodes\b",)):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            values = [int(group) for group in match.groups() if group is not None]
        except ValueError:
            continue
        if task == "jssp" and len(values) >= 2:
            return int(values[0] * values[1])
        if values:
            return int(values[0])
    return None


def infer_instance_size(task: str, row: dict) -> int | None:
    instance = parse_instance_value(row.get("instance_json"))
    input_text = str(row.get("input", ""))
    instruction_text = str(row.get("instruction", ""))
    task = str(task).lower()

    if isinstance(instance, list):
        try:
            if task == "tsp":
                return int(len(instance))
            if task == "op":
                if (
                    instance
                    and isinstance(instance[0], list)
                    and instance[0]
                    and isinstance(instance[0][0], list)
                ):
                    return int(len(instance[0]))
                return int(len(instance))
            if task == "cvrp":
                coords = instance[0] if instance and isinstance(instance[0], list) else instance
                if isinstance(coords, list):
                    return int(len(coords))
        except Exception:
            pass

    if isinstance(instance, dict):
        try:
            if task in {"tsp", "cvrp", "op"}:
                for key in ("coords", "coordinates", "nodes", "locations"):
                    value = instance.get(key)
                    if isinstance(value, list) and value:
                        return int(len(value))
            if task == "mis":
                for key in ("num_nodes", "n", "nodes"):
                    value = instance.get(key)
                    if isinstance(value, int):
                        return int(value)
                    if isinstance(value, list):
                        return int(len(value))
            if task == "jssp":
                jobs = instance.get("jobs")
                if isinstance(jobs, list) and jobs:
                    return int(sum(len(job) for job in jobs if isinstance(job, list)))
                n_jobs = instance.get("num_jobs", instance.get("n_jobs"))
                n_machines = instance.get("num_machines", instance.get("n_machines"))
                if isinstance(n_jobs, int) and isinstance(n_machines, int):
                    return int(n_jobs * n_machines)
        except Exception:
            pass

    return instance_size_from_text(task, f"{instruction_text}\n{input_text}")


def size_summary(size_records: list[tuple[int, int | None]]) -> dict:
    known = [int(size) for _idx, size in size_records if size is not None]
    if not known:
        return {
            "known_examples": 0,
            "unknown_examples": len(size_records),
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    sorted_known = sorted(known)
    median_idx = len(sorted_known) // 2
    if len(sorted_known) % 2 == 0:
        median_value = (sorted_known[median_idx - 1] + sorted_known[median_idx]) / 2.0
    else:
        median_value = float(sorted_known[median_idx])
    return {
        "known_examples": len(known),
        "unknown_examples": len(size_records) - len(known),
        "min": int(min(known)),
        "median": median_value,
        "mean": float(sum(known) / len(known)),
        "max": int(max(known)),
    }


def slice_known_size_bucket(
    records: list[tuple[int, int | None]], bucket: str
) -> list[tuple[int, int | None]]:
    known_records = [(idx, int(size)) for idx, size in records if size is not None]
    if bucket == "all":
        return list(records)
    if not known_records:
        return []

    ordered = sorted(known_records, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    if bucket == "small":
        return ordered[: max(1, (n + 2) // 3)]
    if bucket == "medium":
        start = n // 3
        end = max(start + 1, (2 * n + 2) // 3)
        return ordered[start : min(end, n)]
    if bucket == "large":
        return ordered[(2 * n) // 3 :]
    if bucket == "hardest":
        count = max(1, (n + 4) // 5)
        return ordered[-count:]
    return list(records)


def select_dataset_by_size(
    dataset: Dataset,
    *,
    task: str,
    size_bucket: str,
    min_instance_size: int,
    max_instance_size: int,
) -> tuple[Dataset, dict]:
    bucket = str(size_bucket or "all").lower()
    min_size = int(min_instance_size or 0)
    max_size = int(max_instance_size or 0)
    if bucket == "all" and min_size <= 0 and max_size <= 0:
        return dataset, {
            "size_bucket": bucket,
            "min_instance_size": min_size,
            "max_instance_size": max_size,
            "size_filter_active": False,
            "all_size": size_summary([]),
            "selected_size": size_summary([]),
        }

    records: list[tuple[int, int | None]] = []
    for idx in range(len(dataset)):
        records.append((idx, infer_instance_size(task, dict(dataset[idx]))))

    filtered_records = [
        (idx, size)
        for idx, size in records
        if size is not None
        and (min_size <= 0 or int(size) >= min_size)
        and (max_size <= 0 or int(size) <= max_size)
    ]
    selected_records = slice_known_size_bucket(filtered_records, bucket)
    selected_indices = sorted(idx for idx, _size in selected_records)
    selected_dataset = dataset.select(selected_indices) if selected_indices else dataset.select([])
    return selected_dataset, {
        "size_bucket": bucket,
        "min_instance_size": min_size,
        "max_instance_size": max_size,
        "size_filter_active": True,
        "all_size": size_summary(records),
        "selected_size": size_summary(selected_records),
    }


def apply_task_multiplier(dataset: Dataset, *, multiplier: float, seed: int, task: str) -> Dataset:
    value = float(multiplier)
    if len(dataset) == 0 or math.isclose(value, 1.0):
        return dataset
    if value <= 0.0:
        print_primary(
            f"[GRPO] {task}: task multiplier={value:.3f}; dropping all selected train examples."
        )
        return dataset.select([])

    full_repeats = int(math.floor(value))
    fractional = value - full_repeats
    parts = [dataset] * full_repeats
    if fractional > 1e-9:
        sample_count = max(1, int(round(len(dataset) * fractional)))
        sample_count = min(sample_count, len(dataset))
        parts.append(dataset.shuffle(seed=seed + 7919).select(range(sample_count)))

    if not parts:
        return dataset.select([])
    if len(parts) == 1:
        return parts[0].shuffle(seed=seed + 1543)
    return concatenate_datasets(parts).shuffle(seed=seed + 1543)


def split_dataset(dataset: Dataset, eval_ratio: float, seed: int) -> DatasetDict:
    if float(eval_ratio) <= 0.0:
        return DatasetDict({"train": dataset, "test": dataset.select([])})
    if len(dataset) < 2:
        return DatasetDict({"train": dataset, "test": dataset.select([])})
    ratio = min(max(float(eval_ratio), 1.0 / len(dataset)), 0.5)
    return DatasetDict(dataset.train_test_split(test_size=ratio, seed=seed))


def resolve_eval_cap(max_train_samples_per_task: int, max_eval_samples_per_task: int) -> int:
    train_cap = max(0, int(max_train_samples_per_task))
    eval_cap = int(max_eval_samples_per_task)
    if eval_cap > 0:
        return eval_cap
    if train_cap > 0:
        return max(train_cap // 10, 1)
    return 0


def validate_task_split_sizes(
    *,
    task: str,
    train_size: int,
    eval_size: int,
    min_train_samples_per_task: int,
    min_eval_samples_per_task: int,
) -> None:
    if int(min_train_samples_per_task) > 0 and int(train_size) < int(min_train_samples_per_task):
        raise ValueError(
            f"{task} train split has only {train_size} examples, below the required minimum of "
            f"{min_train_samples_per_task}."
        )
    if int(min_eval_samples_per_task) > 0 and int(eval_size) < int(min_eval_samples_per_task):
        raise ValueError(
            f"{task} eval split has only {eval_size} examples, below the required minimum of "
            f"{min_eval_samples_per_task}."
        )


def _normalize_sft_boundary(prompt: str, completion: str) -> tuple[str, str]:
    normalized_prompt = str(prompt)
    if not normalized_prompt.endswith("\n"):
        normalized_prompt = normalized_prompt + "\n"

    normalized_completion = str(completion)
    while normalized_completion.startswith("\n"):
        normalized_completion = normalized_completion[1:]

    return normalized_prompt, normalized_completion


def inspect_sft_tokenization_alignment(
    dataset: Dataset,
    tokenizer,
    *,
    max_samples: int = 256,
) -> dict[str, object]:
    checked = min(len(dataset), max(0, int(max_samples)))
    mismatches = 0
    examples: list[dict[str, object]] = []

    for idx in range(checked):
        row = dataset[idx]
        prompt = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        full_text = prompt + completion

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if full_ids[: len(prompt_ids)] == prompt_ids:
            continue

        mismatches += 1
        if len(examples) < 3:
            examples.append(
                {
                    "index": idx,
                    "prompt_tail": prompt[-80:],
                    "completion_head": completion[:80],
                    "prompt_token_count": len(prompt_ids),
                    "full_token_count": len(full_ids),
                }
            )

    return {
        "checked": checked,
        "mismatches": mismatches,
        "examples": examples,
    }


def convert_sft_dataset_to_text_only(dataset: Dataset) -> Dataset:
    def map_row(example):
        text = str(example.get("text", ""))
        if not text:
            text = str(example.get("prompt", "")) + str(example.get("completion", ""))
        return {
            "text": text,
            "task_type": str(example.get("task_type", "")),
        }

    return dataset.map(map_row, remove_columns=dataset.column_names)


class MixedTaskDistributedSampler(Sampler[int]):
    """Sampler that guarantees each full local batch spans multiple task types."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        task_column: str = "task_type",
        min_task_types_per_batch: int = 2,
        seed: int = 42,
        drop_last: bool = False,
        rank: int | None = None,
        num_replicas: int | None = None,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        column_names = getattr(dataset, "column_names", [])
        if task_column not in column_names:
            raise ValueError(f"Dataset is missing required task column '{task_column}'")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.task_column = str(task_column)
        self.min_task_types_per_batch = max(1, int(min_task_types_per_batch))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.rank = env_rank() if rank is None else int(rank)
        self.num_replicas = env_world_size() if num_replicas is None else int(num_replicas)
        self.epoch = 0

        self.task_labels = [str(label) for label in dataset[self.task_column]]
        self.global_task_counts: dict[str, int] = defaultdict(int)
        for task_name in self.task_labels:
            self.global_task_counts[task_name] += 1

        if (
            len(self.global_task_counts) > 1
            and self.batch_size < 2
            and self.min_task_types_per_batch > 1
        ):
            raise ValueError(
                "Mixed-task batching requires batch_size >= 2 when more than one task is present."
            )

        self.local_task_counts = {
            task_name: math.ceil(count / max(self.num_replicas, 1))
            for task_name, count in self.global_task_counts.items()
        }
        self.total_local_examples = sum(self.local_task_counts.values())
        active_task_count = len([count for count in self.local_task_counts.values() if count > 0])

        if self.drop_last:
            base_num_batches = self.total_local_examples // self.batch_size
        else:
            base_num_batches = math.ceil(self.total_local_examples / self.batch_size)

        required_distinct_tasks = min(
            self.min_task_types_per_batch, self.batch_size, active_task_count
        )
        max_examples_per_task_per_batch = (
            self.batch_size - max(required_distinct_tasks - 1, 0)
            if active_task_count > 1
            else self.batch_size
        )
        batches_needed_for_coverage = 0
        if max_examples_per_task_per_batch > 0:
            batches_needed_for_coverage = max(
                math.ceil(count / max_examples_per_task_per_batch)
                for count in self.local_task_counts.values()
            )

        self.num_batches = max(base_num_batches, batches_needed_for_coverage)
        self.num_samples = self.num_batches * self.batch_size

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_local_task_indices(self) -> dict[str, list[int]]:
        indices_by_task: dict[str, list[int]] = defaultdict(list)
        for idx, task_name in enumerate(self.task_labels):
            indices_by_task[task_name].append(idx)

        local_task_indices: dict[str, list[int]] = {}
        for task_offset, task_name in enumerate(sorted(indices_by_task)):
            task_indices = list(indices_by_task[task_name])
            rng = random.Random(self.seed + (1009 * self.epoch) + task_offset)
            rng.shuffle(task_indices)

            if len(task_indices) % self.num_replicas != 0:
                pad = self.num_replicas - (len(task_indices) % self.num_replicas)
                task_indices.extend(task_indices[:pad])

            local_task_indices[task_name] = task_indices[self.rank :: self.num_replicas]
        return local_task_indices

    def _pick_task(
        self,
        rotation: deque[str],
        remaining: dict[str, int],
        reusable_tasks: list[str],
        exclude: set[str],
    ) -> str:
        preferred_exists = any(
            task not in exclude and remaining.get(task, 0) > 0 for task in rotation
        )
        for _ in range(len(rotation)):
            task_name = rotation[0]
            rotation.rotate(-1)
            if task_name in exclude:
                continue
            if preferred_exists and remaining.get(task_name, 0) <= 0:
                continue
            return task_name

        for task_name in reusable_tasks:
            if task_name not in exclude:
                return task_name
        return reusable_tasks[0]

    def _build_task_schedule(self, local_task_indices: dict[str, list[int]]) -> list[str]:
        reusable_tasks = [task_name for task_name, indices in local_task_indices.items() if indices]
        if not reusable_tasks or self.num_samples <= 0:
            return []

        required_distinct_tasks = min(
            self.min_task_types_per_batch,
            self.batch_size,
            len(reusable_tasks),
        )
        remaining = {task_name: len(local_task_indices[task_name]) for task_name in reusable_tasks}
        rotation_list = list(reusable_tasks)
        random.Random(self.seed + (2003 * self.epoch) + 17).shuffle(rotation_list)
        if self.batch_size == 1 and len(rotation_list) > 1:
            # Phase-shift rank-local schedules so a DDP microstep still spans
            # several tasks globally even though each rank holds one example.
            rank_offset = self.rank % len(rotation_list)
            rotation_list = rotation_list[rank_offset:] + rotation_list[:rank_offset]
        rotation = deque(rotation_list)

        schedule: list[str] = []
        for _ in range(self.num_batches):
            batch_tasks: list[str] = []
            for _ in range(required_distinct_tasks):
                task_name = self._pick_task(rotation, remaining, reusable_tasks, set(batch_tasks))
                batch_tasks.append(task_name)
                if remaining.get(task_name, 0) > 0:
                    remaining[task_name] -= 1

            while len(batch_tasks) < self.batch_size:
                task_name = self._pick_task(rotation, remaining, reusable_tasks, set())
                batch_tasks.append(task_name)
                if remaining.get(task_name, 0) > 0:
                    remaining[task_name] -= 1

            schedule.extend(batch_tasks)

        return schedule

    def __iter__(self) -> Iterator[int]:
        local_task_indices = self._build_local_task_indices()
        task_schedule = self._build_task_schedule(local_task_indices)
        if not task_schedule:
            return iter(())

        refill_rng = random.Random(self.seed + (3001 * self.epoch) + self.rank)
        task_queues = {
            task_name: deque(indices)
            for task_name, indices in local_task_indices.items()
            if indices
        }
        task_refills = {
            task_name: list(indices) for task_name, indices in local_task_indices.items() if indices
        }

        ordered_indices: list[int] = []
        for task_name in task_schedule:
            queue = task_queues[task_name]
            if not queue:
                refill = list(task_refills[task_name])
                refill_rng.shuffle(refill)
                queue.extend(refill)
            ordered_indices.append(queue.popleft())

        return iter(ordered_indices)


def build_mixed_task_sampler_for_trainer(dataset, trainer_args):
    if dataset is None or "task_type" not in getattr(dataset, "column_names", []):
        return None

    task_types = dataset["task_type"]
    unique_tasks = {str(task_name) for task_name in task_types}
    batch_size = int(trainer_args.per_device_train_batch_size)
    # A shared adapter does not require every individual microbatch to contain
    # multiple tasks.  With batch size 1 the deterministic round-robin schedule
    # still balances tasks across gradient accumulation steps, while avoiding
    # the doubled logits allocation that can OOM at long sequence lengths.
    min_task_types = min(2, batch_size) if len(unique_tasks) > 1 else 1
    return MixedTaskDistributedSampler(
        dataset,
        batch_size=batch_size,
        task_column="task_type",
        min_task_types_per_batch=min_task_types,
        seed=int(getattr(trainer_args, "data_seed", None) or trainer_args.seed),
        drop_last=bool(getattr(trainer_args, "dataloader_drop_last", False)),
    )


def prepare_sft_dataset_for_task(
    dataset: Dataset,
    task: str,
    eos_token: str,
    *,
    no_reason: bool = False,
    prompt_style: str = "alpaca",
    tokenizer=None,
    strict_chat_prompt: bool = False,
) -> Dataset:
    def map_row(example):
        transformed = transform_example_for_task(
            task,
            str(example.get("instruction", "")),
            str(example.get("input", "")),
            str(example.get("output", "")),
            example.get("instance_json"),
        )
        prompt = build_formatted_prompt(
            transformed["instruction"],
            str(transformed["input"]),
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            strict_chat_prompt=strict_chat_prompt,
        )
        completion = str(transformed["output"])
        prompt, completion = _normalize_sft_boundary(prompt, completion)
        completion_with_eos = completion + eos_token
        text = prompt + completion_with_eos
        return {
            "prompt": prompt,
            "completion": completion_with_eos,
            "text": text,
            "task_type": task,
        }

    return dataset.map(map_row, remove_columns=dataset.column_names)


def prepare_grpo_dataset_for_task(
    dataset: Dataset,
    task: str,
    *,
    no_reason: bool = False,
    prompt_style: str = "alpaca",
    tokenizer=None,
    strict_chat_prompt: bool = False,
) -> Dataset:
    def map_row(example):
        instruction = str(example.get("instruction", ""))
        imitation_source = (
            example.get("reference_imitation_output")
            if example.get("reference_imitation_output") is not None
            else example.get("imitation_output")
        )
        negative_imitation_source = example.get("negative_imitation_output")
        transformed = transform_example_for_task(
            task,
            instruction,
            str(example.get("input", "")),
            str(example.get("output", "")),
            example.get("instance_json"),
        )
        row = {
            "prompt": build_formatted_prompt(
                str(transformed["instruction"]),
                str(transformed["input"]),
                no_reason=no_reason,
                prompt_style=prompt_style,
                tokenizer=tokenizer,
                strict_chat_prompt=strict_chat_prompt,
            ),
            "answer": str(transformed["output"]),
            "instruction": str(transformed["instruction"]),
            "input": str(transformed["input"]),
            "instance_json": coerce_json_string(transformed["instance_json"]),
            "task_type": task,
        }
        if imitation_source is not None:
            imitation_transformed = transform_example_for_task(
                task,
                instruction,
                str(example.get("input", "")),
                str(imitation_source),
                example.get("instance_json"),
            )
            row["reference_answer"] = str(imitation_transformed["output"])
        if negative_imitation_source is not None:
            negative_transformed = transform_example_for_task(
                task,
                instruction,
                str(example.get("input", "")),
                str(negative_imitation_source),
                example.get("instance_json"),
            )
            row["negative_reference_answer"] = str(negative_transformed["output"])
        return row

    return dataset.map(map_row, remove_columns=dataset.column_names)


def load_mixed_sft_dataset(
    *,
    tasks: list[str],
    data_root: str,
    eval_ratio: float,
    eval_from_train_split: bool,
    seed: int,
    eos_token: str,
    max_samples_per_task: int,
    max_eval_samples_per_task: int,
    min_train_samples_per_task: int,
    min_eval_samples_per_task: int,
    prefer_subset_files: bool,
    deduplicate_train_rows: bool = True,
    no_reason: bool = False,
    prompt_style: str = "alpaca",
    tokenizer=None,
    strict_chat_prompt: bool = False,
    train_size_bucket: str = "all",
    train_min_instance_size: int = 0,
    train_max_instance_size: int = 0,
) -> tuple[Dataset, Dataset]:
    train_parts = []
    eval_parts = []
    train_cap = max(0, int(max_samples_per_task))
    eval_cap = resolve_eval_cap(train_cap, int(max_eval_samples_per_task))

    if eval_from_train_split:
        print_primary(
            "[SFT] eval-from-train-split is enabled: training eval samples will be carved "
            "from each task train split; task eval files are reserved for post-training evaluation."
        )

    for task in tasks:
        train_json_files = list_split_json_files(
            data_root,
            task,
            "train",
            prefer_subset_files=prefer_subset_files,
            required=True,
        )
        train_source = summarize_json_sources(train_json_files)
        raw_train_dataset = load_and_merge_task_datasets(
            train_json_files,
            deduplicate_exact_rows=bool(deduplicate_train_rows),
        )
        print_primary(f"[SFT] {task}: deduplicate_train_rows={bool(deduplicate_train_rows)}")
        raw_train_dataset, size_info = select_dataset_by_size(
            raw_train_dataset,
            task=task,
            size_bucket=train_size_bucket,
            min_instance_size=train_min_instance_size,
            max_instance_size=train_max_instance_size,
        )
        if size_info["size_filter_active"]:
            print_primary(
                f"[SFT] {task}: train_size_bucket={size_info['size_bucket']}, "
                f"selected={len(raw_train_dataset)}, all_size={size_info['all_size']}, "
                f"selected_size={size_info['selected_size']}"
            )
        train_dataset = prepare_sft_dataset_for_task(
            shuffle_and_cap_dataset(
                raw_train_dataset,
                seed=seed,
                cap=train_cap,
            ),
            task,
            eos_token,
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            strict_chat_prompt=strict_chat_prompt,
        )

        if eval_from_train_split:
            split = split_dataset(train_dataset, eval_ratio=eval_ratio, seed=seed)
            eval_split = cap_dataset_preserving_order(split["test"], eval_cap)
            validate_task_split_sizes(
                task=task,
                train_size=len(split["train"]),
                eval_size=len(eval_split),
                min_train_samples_per_task=min_train_samples_per_task,
                min_eval_samples_per_task=min_eval_samples_per_task,
            )
            train_parts.append(split["train"])
            eval_parts.append(eval_split)
            print_primary(
                f"[SFT] {task}: train={len(split['train'])}, eval={len(eval_split)}, "
                f"source_train={train_source}, source_eval=auto_split"
            )
            continue

        eval_json_files = list_split_json_files(
            data_root,
            task,
            "eval",
            prefer_subset_files=prefer_subset_files,
            required=False,
        )
        if not eval_json_files:
            split = split_dataset(train_dataset, eval_ratio=eval_ratio, seed=seed)
            eval_split = cap_dataset_preserving_order(split["test"], eval_cap)
            validate_task_split_sizes(
                task=task,
                train_size=len(split["train"]),
                eval_size=len(eval_split),
                min_train_samples_per_task=min_train_samples_per_task,
                min_eval_samples_per_task=min_eval_samples_per_task,
            )
            train_parts.append(split["train"])
            eval_parts.append(eval_split)
            print_primary(
                f"[SFT] {task}: train={len(split['train'])}, eval={len(eval_split)}, "
                f"source_train={train_source}, source_eval=auto_split_fallback"
            )
            continue

        eval_source = summarize_json_sources(eval_json_files)
        eval_dataset = prepare_sft_dataset_for_task(
            shuffle_and_cap_dataset(
                load_and_merge_task_datasets(eval_json_files),
                seed=seed,
                cap=eval_cap,
            ),
            task,
            eos_token,
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            strict_chat_prompt=strict_chat_prompt,
        )
        validate_task_split_sizes(
            task=task,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            min_train_samples_per_task=min_train_samples_per_task,
            min_eval_samples_per_task=min_eval_samples_per_task,
        )
        train_dataset = ensure_no_split_overlap(
            task=task, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        train_parts.append(train_dataset)
        eval_parts.append(eval_dataset)
        print_primary(
            f"[SFT] {task}: train={len(train_dataset)}, eval={len(eval_dataset)}, "
            f"source_train={train_source}, source_eval={eval_source}"
        )

    train_dataset = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_dataset = (
        concatenate_datasets(eval_parts).shuffle(seed=seed)
        if eval_parts
        else train_dataset.select([])
    )
    print_primary(f"[SFT] Mixed dataset: train={len(train_dataset)}, eval={len(eval_dataset)}")
    return train_dataset, eval_dataset


def load_mixed_grpo_dataset(
    *,
    tasks: list[str],
    data_root: str,
    eval_ratio: float,
    eval_from_train_split: bool,
    seed: int,
    max_samples_per_task: int,
    max_eval_samples_per_task: int,
    min_train_samples_per_task: int,
    min_eval_samples_per_task: int,
    prefer_subset_files: bool,
    no_reason: bool = False,
    prompt_style: str = "alpaca",
    tokenizer=None,
    strict_chat_prompt: bool = False,
    train_size_bucket: str = "all",
    train_min_instance_size: int = 0,
    train_max_instance_size: int = 0,
    train_task_multiplier_map: dict[str, float] | None = None,
) -> tuple[Dataset, Dataset]:
    train_parts = []
    eval_parts = []
    train_cap = max(0, int(max_samples_per_task))
    eval_cap = resolve_eval_cap(train_cap, int(max_eval_samples_per_task))
    task_multipliers = train_task_multiplier_map or {}

    def maybe_apply_train_multiplier(dataset: Dataset, task: str) -> Dataset:
        multiplier = float(task_multipliers.get(task, 1.0))
        if math.isclose(multiplier, 1.0):
            return dataset
        before_multiplier = len(dataset)
        multiplied = apply_task_multiplier(
            dataset,
            multiplier=multiplier,
            seed=seed,
            task=task,
        )
        print_primary(
            f"[GRPO] {task}: train_task_multiplier={multiplier:.3f}, "
            f"selected_before_multiplier={before_multiplier}, selected_after_multiplier={len(multiplied)}"
        )
        return multiplied

    if eval_from_train_split:
        print_primary(
            "[GRPO] eval-from-train-split is enabled: training eval samples will be carved "
            "from each task train split; task eval files are reserved for post-training evaluation."
        )

    for task in tasks:
        train_json_files = list_split_json_files(
            data_root,
            task,
            "train",
            prefer_subset_files=prefer_subset_files,
            required=True,
        )
        train_source = summarize_json_sources(train_json_files)
        raw_train_dataset = load_and_merge_task_datasets(train_json_files)
        raw_train_dataset, size_info = select_dataset_by_size(
            raw_train_dataset,
            task=task,
            size_bucket=train_size_bucket,
            min_instance_size=train_min_instance_size,
            max_instance_size=train_max_instance_size,
        )
        if size_info["size_filter_active"]:
            print_primary(
                f"[GRPO] {task}: train_size_bucket={size_info['size_bucket']}, "
                f"selected={len(raw_train_dataset)}, all_size={size_info['all_size']}, "
                f"selected_size={size_info['selected_size']}"
            )

        # External benchmark data remains evaluation-only even when training
        # uses an internal split (or disables internal evaluation entirely).
        # Decontaminate before shuffling/capping so a removed row is refilled
        # from the remaining public training pool instead of shrinking the
        # requested one-pass sample budget.  Load every external eval JSON here,
        # not merely a preferred subset, to exclude the complete benchmark.
        external_eval_json_files = list_split_json_files(
            data_root,
            task,
            "eval",
            prefer_subset_files=False,
            required=False,
        )
        if external_eval_json_files:
            external_eval_for_exclusion = load_and_merge_task_datasets(external_eval_json_files)
            raw_train_dataset = ensure_no_instance_overlap(
                task=task,
                train_dataset=raw_train_dataset,
                eval_dataset=external_eval_for_exclusion,
            )
            print_primary(
                f"[GRPO] {task}: external-eval decontamination used "
                f"{len(external_eval_for_exclusion)} instance row(s) from "
                f"{summarize_json_sources(external_eval_json_files)} before train shuffle/cap."
            )
        train_dataset = prepare_grpo_dataset_for_task(
            shuffle_and_cap_dataset(
                raw_train_dataset,
                seed=seed,
                cap=train_cap,
            ),
            task,
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            strict_chat_prompt=strict_chat_prompt,
        )

        if eval_from_train_split:
            split = split_dataset(train_dataset, eval_ratio=eval_ratio, seed=seed)
            train_split = maybe_apply_train_multiplier(split["train"], task)
            eval_split = cap_dataset_preserving_order(split["test"], eval_cap)
            validate_task_split_sizes(
                task=task,
                train_size=len(train_split),
                eval_size=len(eval_split),
                min_train_samples_per_task=min_train_samples_per_task,
                min_eval_samples_per_task=min_eval_samples_per_task,
            )
            train_parts.append(train_split)
            eval_parts.append(eval_split)
            print_primary(
                f"[GRPO] {task}: train={len(train_split)}, eval={len(eval_split)}, "
                f"source_train={train_source}, source_eval=auto_split"
            )
            continue

        eval_json_files = list_split_json_files(
            data_root,
            task,
            "eval",
            prefer_subset_files=prefer_subset_files,
            required=False,
        )
        if not eval_json_files:
            split = split_dataset(train_dataset, eval_ratio=eval_ratio, seed=seed)
            train_split = maybe_apply_train_multiplier(split["train"], task)
            eval_split = cap_dataset_preserving_order(split["test"], eval_cap)
            validate_task_split_sizes(
                task=task,
                train_size=len(train_split),
                eval_size=len(eval_split),
                min_train_samples_per_task=min_train_samples_per_task,
                min_eval_samples_per_task=min_eval_samples_per_task,
            )
            train_parts.append(train_split)
            eval_parts.append(eval_split)
            print_primary(
                f"[GRPO] {task}: train={len(train_split)}, eval={len(eval_split)}, "
                f"source_train={train_source}, source_eval=auto_split_fallback"
            )
            continue

        eval_source = summarize_json_sources(eval_json_files)
        eval_dataset = prepare_grpo_dataset_for_task(
            shuffle_and_cap_dataset(
                load_and_merge_task_datasets(eval_json_files),
                seed=seed,
                cap=eval_cap,
            ),
            task,
            no_reason=no_reason,
            prompt_style=prompt_style,
            tokenizer=tokenizer,
            strict_chat_prompt=strict_chat_prompt,
        )
        train_dataset = ensure_no_split_overlap(
            task=task, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
        train_dataset = maybe_apply_train_multiplier(train_dataset, task)
        validate_task_split_sizes(
            task=task,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            min_train_samples_per_task=min_train_samples_per_task,
            min_eval_samples_per_task=min_eval_samples_per_task,
        )
        train_parts.append(train_dataset)
        eval_parts.append(eval_dataset)
        print_primary(
            f"[GRPO] {task}: train={len(train_dataset)}, eval={len(eval_dataset)}, "
            f"source_train={train_source}, source_eval={eval_source}"
        )

    train_dataset = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_dataset = (
        concatenate_datasets(eval_parts).shuffle(seed=seed)
        if eval_parts
        else train_dataset.select([])
    )
    print_primary(f"[GRPO] Mixed dataset: train={len(train_dataset)}, eval={len(eval_dataset)}")
    return train_dataset, eval_dataset
