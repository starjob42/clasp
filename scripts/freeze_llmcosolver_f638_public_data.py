#!/usr/bin/env python3
"""Acquire public RL bytes once and freeze an all-row data manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from clasp.baselines.llmcosolver_f638_public_data import (  # noqa: E402
    OFFICIAL_RL_FOLDER_ID,
    OFFICIAL_SFT_FOLDER_ID,
    PUBLICATION_TASKS,
    RL_FILES,
    SFT_FILES,
    UPSTREAM_COMMIT,
    acquire_create_once,
    analyze_split_pair,
    canonical_json,
    drive_file_projection,
    observe_drive_metadata,
    self_hash,
    sha256_file,
)

UPSTREAM_README_SHA256 = "09b2bfea2a6af3d56457d72d4d34c8039d61319fa5ff5d0e17436fe34eda4050"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--rl-root",
        type=Path,
        default=Path("third_party/llmcosolver_f6388991/data_rl"),
    )
    parser.add_argument(
        "--sft-root",
        type=Path,
        default=Path("../hybrid_gnn_llm/data_sft"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json"),
    )
    parser.add_argument("--observed-on", default="2026-08-15")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--acquire-rl",
        action="store_true",
        help="Acquire missing RL files from their exact official Drive item IDs.",
    )
    parser.add_argument(
        "--verify-rl-headers",
        action="store_true",
        help="Issue HEAD requests and require the live Content-Length to match.",
    )
    return parser.parse_args()


def _resolved(root: Path, repository_root: Path) -> Path:
    return root.resolve() if root.is_absolute() else (repository_root / root).resolve()


def _relative(path: Path, repository_root: Path) -> str:
    return os.path.relpath(path, repository_root)


def _specification(task: str, split: str, source: tuple) -> Any:
    matches = [item for item in source if item.task == task and item.split == split]
    if len(matches) != 1:
        raise ValueError(f"Expected one {task}/{split} file, found {len(matches)}")
    return matches[0]


def _acquire_and_observe_rl(
    rl_root: Path,
    *,
    acquire: bool,
    verify_headers: bool,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    metadata: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, str] = {}

    if verify_headers:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(observe_drive_metadata, specification): specification
                for specification in RL_FILES
            }
            for future in as_completed(futures):
                specification = futures[future]
                metadata[specification.relative_path] = future.result()
                print(f"verified RL header: {specification.relative_path}", flush=True)

    if acquire:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(acquire_create_once, specification, rl_root): specification
                for specification in RL_FILES
            }
            for future in as_completed(futures):
                specification = futures[future]
                _, outcome = future.result()
                outcomes[specification.relative_path] = outcome
                print(
                    f"RL {outcome}: {specification.relative_path} "
                    f"({specification.byte_length} bytes)",
                    flush=True,
                )
    else:
        for specification in RL_FILES:
            path = rl_root / specification.relative_path
            if not path.is_file() or path.stat().st_size != specification.byte_length:
                raise FileNotFoundError(f"Missing exact RL file {path}; rerun with --acquire-rl")
            outcomes[specification.relative_path] = "validated_existing"
    return metadata, outcomes


def _analysis_jobs(
    rl_root: Path,
    sft_root: Path,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    jobs: dict[Any, tuple[str, str]] = {}
    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for source_kind, root, specifications in (
            ("rl", rl_root, RL_FILES),
            ("sft", sft_root, SFT_FILES),
        ):
            for task in PUBLICATION_TASKS:
                train = _specification(task, "train", specifications)
                evaluation = _specification(task, "eval", specifications)
                future = executor.submit(
                    analyze_split_pair,
                    root / train.relative_path,
                    root / evaluation.relative_path,
                    source_kind=source_kind,
                )
                jobs[future] = (source_kind, task)
        for future in as_completed(jobs):
            key = jobs[future]
            analyses[key] = future.result()
            print(f"parsed every {key[0]} row: {key[1]}", flush=True)
    return analyses


def _local_file_inventory(
    root: Path,
    specifications: tuple,
    repository_root: Path,
    *,
    sft: bool,
    acquisition_outcomes: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for specification in specifications:
        path = root / specification.relative_path
        projection = drive_file_projection(specification)
        projection["workspace_path"] = _relative(path, repository_root)
        if not path.is_file():
            projection.update(
                {
                    "local_availability": "MISSING",
                    "local_bytes": None,
                    "local_sha256": None,
                    "size_matches_drive_header": False,
                }
            )
        else:
            size = path.stat().st_size
            if size != specification.byte_length:
                raise ValueError(
                    f"Local size mismatch for {path}: {size} != {specification.byte_length}"
                )
            projection.update(
                {
                    "local_availability": "PRESENT",
                    "local_bytes": size,
                    "local_sha256": sha256_file(path),
                    "size_matches_drive_header": True,
                }
            )
        if sft:
            projection["drive_byte_identity"] = (
                "NOT_PROVEN_NO_OFFICIAL_CHECKSUM"
                if path.is_file()
                else "NOT_APPLICABLE_LOCAL_FILE_MISSING"
            )
            projection["downloaded_by_this_workflow"] = False
        else:
            projection["acquisition_outcome"] = (acquisition_outcomes or {}).get(
                specification.relative_path
            )
            projection["drive_byte_identity"] = "DIRECT_OFFICIAL_ITEM_ACQUISITION_LOCALLY_HASHED"
        inventory[specification.relative_path] = projection
    return dict(sorted(inventory.items()))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.resolve()
    rl_root = _resolved(args.rl_root, repository_root)
    sft_root = _resolved(args.sft_root, repository_root)
    output = _resolved(args.output, repository_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    if args.workers < 1:
        raise ValueError("--workers must be at least one")

    print(
        "SFT inspection only: "
        f"{sum(item.byte_length for item in SFT_FILES):,} published bytes; "
        "large files will not be downloaded",
        flush=True,
    )
    rl_metadata, acquisition_outcomes = _acquire_and_observe_rl(
        rl_root,
        acquire=args.acquire_rl,
        verify_headers=args.verify_rl_headers,
        workers=args.workers,
    )
    analyses = _analysis_jobs(rl_root, sft_root, args.workers)
    rl_inventory = _local_file_inventory(
        rl_root,
        RL_FILES,
        repository_root,
        sft=False,
        acquisition_outcomes=acquisition_outcomes,
    )
    sft_inventory = _local_file_inventory(
        sft_root,
        SFT_FILES,
        repository_root,
        sft=True,
    )

    for task in PUBLICATION_TASKS:
        for source_kind, inventory in (("rl", rl_inventory), ("sft", sft_inventory)):
            for split in ("train", "eval"):
                specification = _specification(
                    task, split, RL_FILES if source_kind == "rl" else SFT_FILES
                )
                observed_hash = analyses[(source_kind, task)][split]["sha256"]
                if observed_hash != inventory[specification.relative_path]["local_sha256"]:
                    raise ValueError(
                        f"Analysis/raw inventory hash mismatch: {source_kind}/{task}/{split}"
                    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract_type": "llmcosolver_f638_public_data_manifest",
        "manifest_sha256": None,
        "status": "FROZEN_DATA_EVIDENCE_NON_EXECUTABLE",
        "observed_on": args.observed_on,
        "upstream": {
            "repository": "https://github.com/Summer142857/LLMCoSolver",
            "commit": UPSTREAM_COMMIT,
            "readme_sha256": UPSTREAM_README_SHA256,
            "readme_data_claim": "data generated in the original paper",
            "root_repository_license": "MIT",
            "data_license_scope": (
                "NO_SEPARATE_LICENSE_OBSERVED_FOR_GOOGLE_DRIVE_DATA; do not infer that "
                "the repository MIT license governs externally hosted bytes"
            ),
        },
        "identity_algorithm": {
            "canonical_json": (
                "UTF-8 json.dumps(sort_keys=True,separators=(',',':'),"
                "ensure_ascii=False,allow_nan=False)"
            ),
            "sequence": "SHA256 of concatenated 32-byte per-row SHA256 digests",
            "parser": "strict UTF-8 streaming top-level-array parser; every row parsed",
        },
        "rl": {
            "official_drive_folder_id": OFFICIAL_RL_FOLDER_ID,
            "classification": "DIRECT_OFFICIAL_LINK_ACQUISITION_LOCALLY_FROZEN",
            "workspace_root": _relative(rl_root, repository_root),
            "total_bytes": sum(item.byte_length for item in RL_FILES),
            "file_count": len(RL_FILES),
            "acquisition_policy": (
                "temporary file plus atomic hard-link create; existing targets validated; "
                "no overwrite path exists"
            ),
            "live_header_observations": dict(sorted(rl_metadata.items())),
            "files": rl_inventory,
            "task_semantics": {task: analyses[("rl", task)] for task in PUBLICATION_TASKS},
        },
        "sft": {
            "official_drive_folder_id": OFFICIAL_SFT_FOLDER_ID,
            "classification": "PROBABLE_RELEASED_BYTES_SIZE_AND_FILENAME_MATCH_ONLY",
            "workspace_root": _relative(sft_root, repository_root),
            "official_total_bytes": sum(item.byte_length for item in SFT_FILES),
            "official_train_bytes": sum(
                item.byte_length for item in SFT_FILES if item.split == "train"
            ),
            "download_policy": "INSPECTION_ONLY_NO_SFT_DOWNLOADS_PERFORMED",
            "lineage_evidence": (
                "Files are untracked under a local clone whose origin is the official "
                "repository and whose HEAD is the pinned commit. Names and byte lengths "
                "match Drive HEAD metadata. No local acquisition receipt or official "
                "checksum was found, so byte equality to Drive is not proven."
            ),
            "drive_header_observation": {
                "method": "direct HEAD to each official Drive item ID",
                "observed_on": args.observed_on,
                "checksum_header_available": False,
                "content_length_and_last_modified_embedded_per_file": True,
            },
            "files": sft_inventory,
            "task_semantics": {task: analyses[("sft", task)] for task in PUBLICATION_TASKS},
            "pickle_policy": (
                "raw bytes hashed only; pickle is not deserialized because it can execute "
                "code and official main_train.py leaves instances=None"
            ),
        },
        "classification_guards": {
            "authorizes_training": False,
            "proves_sft_drive_byte_identity": False,
            "proves_per_task_sft_checkpoint_identity": False,
            "all_rl_json_rows_semantically_parsed": True,
            "all_present_sft_json_rows_semantically_parsed": True,
            "current_clasp_method_grpo": "three-head dr_grpo; non-FOARL and unmatched",
        },
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    return manifest


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    output = _resolved(args.output, repository_root)
    manifest = build_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(f"created {output}", flush=True)
    print(f"manifest_sha256={manifest['manifest_sha256']}", flush=True)
    print(f"canonical_bytes={len(canonical_json(manifest))}", flush=True)


if __name__ == "__main__":
    main()
