#!/usr/bin/env python3
"""Supersede the v1 data receipt with universal cross-stage identities."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from clasp.baselines.llmcosolver_f638_public_data import (  # noqa: E402
    PUBLICATION_TASKS,
    RL_FILES,
    SFT_FILES,
    analyze_task_identity_matrix_v2,
    canonical_json,
    seal_directory_tree_read_only,
    seal_regular_file_read_only,
    self_hash,
    sha256_regular_file_stable,
)

V1_MANIFEST_RELATIVE = "experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json"
V1_MANIFEST_FILE_SHA256 = "e9c99490abf0f27d2ef0b2d6941b3053116dfa080876a3eded28fe06be5eb713"
V1_MANIFEST_SELF_SHA256 = "e5a0edefcc07a8ab8760c724274dd02a15af15267388e59542de8b2a90b25442"


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
    parser.add_argument("--v1-manifest", type=Path, default=Path(V1_MANIFEST_RELATIVE))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json"),
    )
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def _resolved(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _relative(path: Path, repository_root: Path) -> str:
    return os.path.relpath(path, repository_root)


def _specification(task: str, split: str, source: tuple) -> Any:
    matches = [item for item in source if item.task == task and item.split == split]
    if len(matches) != 1:
        raise ValueError(f"Expected one {task}/{split} item, found {len(matches)}")
    return matches[0]


def _load_v1(path: Path) -> dict[str, Any]:
    observed_sha256, _ = sha256_regular_file_stable(path)
    if observed_sha256 != V1_MANIFEST_FILE_SHA256:
        raise ValueError(f"v1 manifest file SHA256 mismatch: {observed_sha256}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_sha256") != V1_MANIFEST_SELF_SHA256:
        raise ValueError("v1 manifest declared self-hash mismatch")
    if self_hash(value) != V1_MANIFEST_SELF_SHA256:
        raise ValueError("v1 manifest canonical self-hash mismatch")
    return value


def _validate_and_seal_rl_tree(
    root: Path,
    v1: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    for specification in RL_FILES:
        path = root / specification.relative_path
        record = v1["rl"]["files"][specification.relative_path]
        observed, metadata = sha256_regular_file_stable(path)
        if observed != record["local_sha256"] or metadata.st_size != record["local_bytes"]:
            raise ValueError(f"RL byte mismatch before read-only seal: {path}")
    seal_directory_tree_read_only(root)
    file_modes: dict[str, str] = {}
    for specification in RL_FILES:
        path = root / specification.relative_path
        observed, metadata = sha256_regular_file_stable(path, require_read_only=True)
        expected = v1["rl"]["files"][specification.relative_path]["local_sha256"]
        if observed != expected:
            raise ValueError(f"RL byte mismatch after read-only seal: {path}")
        file_modes[specification.relative_path] = oct(stat.S_IMODE(metadata.st_mode))
    directory_modes = {}
    for directory, _, _ in os.walk(root):
        path = Path(directory)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Invalid directory in sealed RL tree: {path}")
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode & 0o222:
            raise ValueError(f"Writable directory in sealed RL tree: {path}")
        directory_modes[_relative(path, repository_root)] = oct(mode)
    return {
        "file_mode": "0o444",
        "directory_mode": "0o555",
        "files_verified": len(file_modes),
        "directories": dict(sorted(directory_modes.items())),
        "symlinks_allowed": False,
        "regular_file_stable_hash_required": True,
    }


def _run_matrices(
    rl_root: Path,
    sft_root: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for task in PUBLICATION_TASKS:
            rl_train = _specification(task, "train", RL_FILES)
            rl_eval = _specification(task, "eval", RL_FILES)
            sft_train = _specification(task, "train", SFT_FILES)
            sft_eval = _specification(task, "eval", SFT_FILES)
            future = executor.submit(
                analyze_task_identity_matrix_v2,
                task,
                rl_root / rl_train.relative_path,
                rl_root / rl_eval.relative_path,
                sft_root / sft_train.relative_path,
                sft_root / sft_eval.relative_path,
            )
            futures[future] = task
        for future in as_completed(futures):
            task = futures[future]
            matrices[task] = future.result()
            print(f"schema-v2 identities complete: {task}", flush=True)
    return {task: matrices[task] for task in PUBLICATION_TASKS}


def _v1_split(v1: dict[str, Any], stage: str, task: str, split: str) -> dict[str, Any]:
    return v1[stage]["task_semantics"][task][split]


def _verify_matrix_sources(v1: dict[str, Any], matrices: dict[str, Any]) -> None:
    for task, matrix in matrices.items():
        for stage in ("rl", "sft"):
            for split in ("train", "eval"):
                name = f"{stage}_{split}"
                observed = matrix["splits"][name]
                previous = _v1_split(v1, stage, task, split)
                if observed["raw_sha256"] != previous["sha256"]:
                    raise ValueError(f"Raw SHA changed since v1: {task}/{name}")
                if observed["row_count"] != previous["row_count"]:
                    raise ValueError(f"Row count changed since v1: {task}/{name}")


def _schema_reference_projection(v1: dict[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    retained = (
        "bytes",
        "sha256",
        "top_level",
        "row_count",
        "common_keys",
        "union_keys",
        "field_type_counts",
        "row_identity",
        "reference_fields",
    )
    for stage in ("rl", "sft"):
        projection[stage] = {}
        for task in PUBLICATION_TASKS:
            projection[stage][task] = {}
            for split in ("train", "eval"):
                source = _v1_split(v1, stage, task, split)
                projection[stage][task][split] = {key: source[key] for key in retained}
    return projection


def _contamination_summary(matrices: dict[str, Any]) -> dict[str, Any]:
    rl_within = {}
    sft_within = {}
    pipeline_to_rl_eval = {}
    for task, matrix in matrices.items():
        rl_overlap = matrix["within_stage"]["rl_train__rl_eval"]
        sft_overlap = matrix["within_stage"]["sft_train__sft_eval"]
        union_to_rl_eval = matrix["author_training_union_to_evaluation"][
            "sft_train_union_rl_train__rl_eval"
        ]
        rl_within[task] = rl_overlap["shared_unique_instance_count"]
        sft_within[task] = sft_overlap["shared_unique_instance_count"]
        pipeline_to_rl_eval[task] = {
            "exact_union_shared_instance_count": union_to_rl_eval["shared_unique_instance_count"],
            "eval_row_count": matrix["splits"]["rl_eval"]["row_count"],
            "exact_duplicate_unique_eval_row_count": union_to_rl_eval[
                "exact_duplicate_unique_eval_row_count"
            ],
        }
    return {
        "rl_train_to_rl_eval_shared_instances": rl_within,
        "sft_train_to_sft_eval_shared_instances": sft_within,
        "author_pipeline_train_sources_to_rl_eval": pipeline_to_rl_eval,
        "claim": (
            "The linked author evaluation files are not globally held-out or leakage-free. "
            "They are context-reproduction artifacts only. A publication comparison requires "
            "a preregistered fresh disjoint IID split, which is no longer exact-paper-data "
            "reproduction."
        ),
    }


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    repository_root = args.repository_root.resolve()
    rl_root = _resolved(args.rl_root, repository_root)
    sft_root = _resolved(args.sft_root, repository_root)
    v1_path = _resolved(args.v1_manifest, repository_root)
    output = _resolved(args.output, repository_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite schema-v2 manifest: {output}")
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    v1 = _load_v1(v1_path)
    rl_seal = _validate_and_seal_rl_tree(rl_root, v1, repository_root)
    seal_regular_file_read_only(v1_path)
    matrices = _run_matrices(rl_root, sft_root, args.workers)
    _verify_matrix_sources(v1, matrices)

    rl_projection = {key: value for key, value in v1["rl"].items() if key != "task_semantics"}
    sft_projection = {key: value for key, value in v1["sft"].items() if key != "task_semantics"}
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "contract_type": "llmcosolver_f638_public_data_manifest",
        "manifest_sha256": None,
        "status": "SEALED_DATA_EVIDENCE_NON_EXECUTABLE_SUPERSEDES_V1",
        "supersedes": {
            "path": _relative(v1_path, repository_root),
            "file_sha256": V1_MANIFEST_FILE_SHA256,
            "manifest_sha256": V1_MANIFEST_SELF_SHA256,
            "preserved_read_only": True,
            "reason": (
                "schema v1 used source-specific whole input-side objects for SFT identity; "
                "TSP eval alone carries an extra `instance` field, so its cross-split overlap "
                "was not comparable. Raw hashes, schemas, row identities, and reference-field "
                "evidence remain valid; all overlap claims are superseded by schema v2."
            ),
        },
        "upstream": v1["upstream"],
        "identity_algorithm": {
            "cross_split_and_stage": (
                "task-prefixed rendered input after Unicode NFC normalization and collapse of "
                "every Unicode whitespace run to one ASCII space"
            ),
            "exact_row": "sha256 of the schema-v1 canonical JSON full row",
            "reference": "sha256 of the schema-v1 canonical JSON output string",
            "scope": "same algorithm for RL/SFT train/eval; every JSON row parsed",
        },
        "rl": rl_projection,
        "sft": sft_projection,
        "schema_and_reference_evidence": _schema_reference_projection(v1),
        "task_identity_matrices": matrices,
        "contamination": _contamination_summary(matrices),
        "filesystem_seal": {
            "rl_tree": rl_seal,
            "v1_manifest_mode": "0o444",
            "v2_manifest_required_mode": "0o444",
            "external_sft_tree": (
                "not chmod-mutated; it is neighboring probable-lineage evidence, not vendored "
                "or claimed immutable; non-symlink regular files are stable-checked during parse"
            ),
        },
        "classification_guards": {
            "authorizes_training_or_evaluation": False,
            "author_eval_is_held_out": False,
            "author_eval_is_leakage_free": False,
            "exact_paper_data_is_valid_publication_denominator": False,
            "fresh_preregistered_disjoint_iid_required_for_claims": True,
            "fresh_iid_is_exact_paper_data_reproduction": False,
            "current_clasp_method_grpo": "three-head dr_grpo; non-FOARL and unmatched",
            "sft_drive_byte_identity_proven": False,
            "per_task_sft_checkpoint_identity_proven": False,
        },
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    return manifest, output, v1_path


def main() -> None:
    args = parse_args()
    manifest, output, _ = build_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    seal_regular_file_read_only(output)
    file_sha256, metadata = sha256_regular_file_stable(output, require_read_only=True)
    print(f"created read-only {output}", flush=True)
    print(f"manifest_sha256={manifest['manifest_sha256']}", flush=True)
    print(f"file_sha256={file_sha256}", flush=True)
    print(f"bytes={metadata.st_size}", flush=True)
    print(f"canonical_bytes={len(canonical_json(manifest))}", flush=True)


if __name__ == "__main__":
    main()
