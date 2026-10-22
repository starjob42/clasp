#!/usr/bin/env python3
"""Create a small, leakage-separated TSP development split for CLASP pilots.

This dataset is deliberately marked development-only.  It is useful for fast
algorithm and systems checks with sub-billion-parameter models on local GPUs;
it is not a substitute for the frozen publication benchmark or 7B experiments.

Rows are selected from the existing RL training corpus by instance size, then
ordered by a seed-bound SHA-256 key.  The final holdout is written under the
normal ``eval`` directory so the GRPO loader excludes those instances from all
training draws, including when an internal validation split is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "data_rl" / "tsp" / "train" / "train_rl.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "development_small_tsp_canonical_v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def instance_payload(row: dict[str, Any]) -> object:
    value = row.get("instance_json", row.get("instance"))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def instance_digest(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(instance_payload(row)))


def stable_order_key(row: dict[str, Any], seed: int) -> str:
    payload = f"small-tsp-canonical-v1:{seed}:{instance_digest(row)}".encode("ascii")
    return sha256_bytes(payload)


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-nodes", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--train-count", type=int, default=90)
    parser.add_argument("--eval-count", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_root = args.output_root.absolute()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if args.min_nodes <= 1 or args.max_nodes < args.min_nodes:
        raise ValueError("invalid node-size bounds")
    if args.train_count <= 0 or args.eval_count <= 0:
        raise ValueError("train and eval counts must be positive")

    with source.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("source must be a JSON list of objects")

    selected = []
    seen_instances: set[str] = set()
    for row in rows:
        try:
            size = int(row["num_nodes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every source row must have an integer num_nodes") from exc
        if not args.min_nodes <= size <= args.max_nodes:
            continue
        missing = [key for key in ("instruction", "input", "output") if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"selected row is missing required fields: {missing}")
        digest = instance_digest(row)
        if digest in seen_instances:
            continue
        seen_instances.add(digest)
        selected.append(dict(row))

    required = args.train_count + args.eval_count
    if len(selected) != required:
        raise ValueError(
            "the frozen default split expects exactly "
            f"{required} unique rows in the requested size range; found {len(selected)}"
        )
    selected.sort(key=lambda row: stable_order_key(row, args.seed))
    train_rows = selected[: args.train_count]
    eval_rows = selected[args.train_count :]
    train_digests = {instance_digest(row) for row in train_rows}
    eval_digests = {instance_digest(row) for row in eval_rows}
    if train_digests & eval_digests:
        raise AssertionError("train/eval instance overlap")

    parent = output_root.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=parent))
    try:
        train_path = temporary / "tsp" / "train" / "train_small.json"
        eval_path = temporary / "tsp" / "eval" / "test_small.json"
        write_json_exclusive(train_path, train_rows)
        write_json_exclusive(eval_path, eval_rows)
        manifest = {
            "schema_version": 1,
            "protocol": "small-tsp-canonical-development-v1",
            "development_only": True,
            "publication_eligible": False,
            "reason": (
                "Fast local algorithm/systems pilot derived from the historical RL training "
                "corpus; not the frozen independent ICLR confirmation benchmark."
            ),
            "source": {
                "path": str(source),
                "sha256": sha256_file(source),
                "rows": len(rows),
            },
            "selection": {
                "min_nodes": args.min_nodes,
                "max_nodes": args.max_nodes,
                "seed": args.seed,
                "stable_key": "sha256('small-tsp-canonical-v1:' || seed || instance_sha256)",
                "unique_selected_rows": len(selected),
            },
            "train": {
                "relative_path": str(train_path.relative_to(temporary)),
                "rows": len(train_rows),
                "size_counts": dict(sorted(Counter(int(row["num_nodes"]) for row in train_rows).items())),
                "sha256": sha256_file(train_path),
            },
            "eval": {
                "relative_path": str(eval_path.relative_to(temporary)),
                "rows": len(eval_rows),
                "size_counts": dict(sorted(Counter(int(row["num_nodes"]) for row in eval_rows).items())),
                "sha256": sha256_file(eval_path),
            },
            "instance_overlap_count": len(train_digests & eval_digests),
            "all_instance_sha256": sorted(train_digests | eval_digests),
        }
        manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
        write_json_exclusive(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
    except BaseException:
        # Preserve a failed staging directory for diagnosis, but never expose it
        # under the contracted destination name.
        raise

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
