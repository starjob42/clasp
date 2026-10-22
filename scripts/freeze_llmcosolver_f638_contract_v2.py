#!/usr/bin/env python3
"""Freeze the superseding LLMCoSolver f638 semantic/data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from clasp.baselines.llmcosolver_f638_public_data import (  # noqa: E402
    canonical_json,
    seal_regular_file_read_only,
    sha256_regular_file_stable,
)

V1_CONTRACT_RELATIVE = "experiments/sota_20260814/llmcosolver_f638_semantic_contract.json"
V1_CONTRACT_FILE_SHA256 = "25cfa5898d1418aefde2a84989986a4584fd1dce0d4a51d0eb6479bccb037d3a"
V1_CONTRACT_SELF_SHA256 = "0c94f1e6bd987714f9c39adfc03a632f037e23bb512e5edab2f2bf5004cfaa78"
DATA_V2_RELATIVE = "experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json"
DATA_V2_FILE_SHA256 = "6bc23d3cd0686385101b5d0fbbba7a8ee783986b57211958f256b644bc278f28"
DATA_V2_SELF_SHA256 = "4b4896ae9d627f8306ca505bb245075d2af784dbe1b420c66cf5cf9fab5af06c"

REPOSITORY_SEAL_PATHS = (
    "src/clasp/baselines/llmcosolver_f638_public_data.py",
    "scripts/freeze_llmcosolver_f638_public_data.py",
    "scripts/freeze_llmcosolver_f638_public_data_v2.py",
    "scripts/freeze_llmcosolver_f638_contract_v2.py",
    "tests/test_llmcosolver_f638_public_data.py",
    "tests/test_llmcosolver_f638_public_data_manifest_v2.py",
    "tests/test_llmcosolver_f638_contract_v2.py",
    "docs/llmcosolver_f638_public_data_addendum_v2.md",
    "experiments/sota_20260814/llmcosolver_f638_semantic_contract.json",
    "experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json",
    "experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v1-contract", type=Path, default=Path(V1_CONTRACT_RELATIVE))
    parser.add_argument("--data-manifest", type=Path, default=Path(DATA_V2_RELATIVE))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/sota_20260814/llmcosolver_f638_semantic_contract_v2.json"),
    )
    return parser.parse_args()


def _resolved(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def _self_hash(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = None
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _load_self_hashed(
    path: Path,
    *,
    file_sha256: str,
    self_field: str,
    self_sha256: str,
    require_read_only: bool,
) -> dict[str, Any]:
    observed, _ = sha256_regular_file_stable(path, require_read_only=require_read_only)
    if observed != file_sha256:
        raise ValueError(f"File SHA256 mismatch for {path}: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(self_field) != self_sha256 or _self_hash(value, self_field) != self_sha256:
        raise ValueError(f"Canonical self-hash mismatch for {path}")
    return value


def _overlap_counts(
    data: dict[str, Any],
    section: str,
    pair: str,
) -> dict[str, int]:
    return {
        task: data["task_identity_matrices"][task][section][pair]["shared_unique_instance_count"]
        for task in ("cvrp", "jssp", "mis", "op", "tsp")
    }


def _row_counts(data: dict[str, Any], stage: str, split: str) -> dict[str, int]:
    return {
        task: data["task_identity_matrices"][task]["splits"][f"{stage}_{split}"]["row_count"]
        for task in ("cvrp", "jssp", "mis", "op", "tsp")
    }


def _source_projection(repository_root: Path) -> dict[str, str]:
    projection = {}
    for relative in REPOSITORY_SEAL_PATHS:
        path = repository_root / relative
        observed, _ = sha256_regular_file_stable(path)
        projection[relative] = observed
    return projection


def build_contract(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repository_root = args.repository_root.resolve()
    v1_path = _resolved(args.v1_contract, repository_root)
    data_path = _resolved(args.data_manifest, repository_root)
    output = _resolved(args.output, repository_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite semantic contract v2: {output}")

    v1 = _load_self_hashed(
        v1_path,
        file_sha256=V1_CONTRACT_FILE_SHA256,
        self_field="contract_sha256",
        self_sha256=V1_CONTRACT_SELF_SHA256,
        require_read_only=False,
    )
    seal_regular_file_read_only(v1_path)
    data = _load_self_hashed(
        data_path,
        file_sha256=DATA_V2_FILE_SHA256,
        self_field="manifest_sha256",
        self_sha256=DATA_V2_SELF_SHA256,
        require_read_only=True,
    )
    rl_rows = _row_counts(data, "rl", "train")
    sft_rows = _row_counts(data, "sft", "train")
    if set(sft_rows.values()) != {500_000}:
        raise ValueError(f"Unexpected per-task SFT row counts: {sft_rows}")
    if any(rows % 8 for rows in rl_rows.values()):
        raise ValueError(f"RL rows do not imply an integral single-process horizon: {rl_rows}")

    within_rl = _overlap_counts(data, "within_stage", "rl_train__rl_eval")
    within_sft = _overlap_counts(data, "within_stage", "sft_train__sft_eval")
    sft_train_to_rl_eval = _overlap_counts(data, "cross_stage", "sft_train__rl_eval")
    author_union_to_rl_eval = {
        task: data["task_identity_matrices"][task]["author_training_union_to_evaluation"][
            "sft_train_union_rl_train__rl_eval"
        ]["shared_unique_instance_count"]
        for task in rl_rows
    }
    exact_duplicate_rl = {
        task: data["task_identity_matrices"][task]["within_stage"]["rl_train__rl_eval"][
            "exact_duplicate_unique_row_count"
        ]
        for task in rl_rows
    }

    contract: dict[str, Any] = {
        "schema_version": 2,
        "contract_type": "llmcosolver_f638_source_data_and_semantics",
        "contract_sha256": None,
        "status": "HOLD_NON_EXECUTABLE",
        "supersedes": {
            "path": _relative(v1_path, repository_root),
            "file_sha256": V1_CONTRACT_FILE_SHA256,
            "contract_sha256": V1_CONTRACT_SELF_SHA256,
            "preserved_read_only": True,
            "scope": (
                "updates public-data availability, exact row/schema/reference bindings, "
                "budget bounds, and split-integrity claims; reward/loss and reference-policy "
                "findings are inherited unchanged"
            ),
        },
        "upstream": v1["upstream"],
        "pinned_stack": v1["pinned_stack"],
        "vendored_oracle": v1["vendored_oracle"],
        "scope": v1["scope"],
        "reward_and_loss_semantics": v1["reward_and_loss_semantics"],
        "adapter_and_reference_variants": v1["adapter_and_reference_variants"],
        "pinned_stack_probe": v1["pinned_stack_probe"],
        "public_data_evidence": {
            "path": _relative(data_path, repository_root),
            "file_sha256": DATA_V2_FILE_SHA256,
            "manifest_sha256": DATA_V2_SELF_SHA256,
            "status": data["status"],
            "rl_total_bytes": data["rl"]["total_bytes"],
            "rl_file_count": data["rl"]["file_count"],
            "rl_files_read_only": True,
            "rl_direct_official_item_acquisition": True,
            "sft_official_total_bytes": data["sft"]["official_total_bytes"],
            "sft_official_train_bytes": data["sft"]["official_train_bytes"],
            "sft_classification": data["sft"]["classification"],
            "sft_drive_byte_identity_proven": False,
            "sft_missing_local_items": ["tsp/instances.pkl"],
            "separate_data_license_observed": False,
            "data_redistribution_authorized_by_this_contract": False,
        },
        "split_integrity": {
            "identity": data["identity_algorithm"]["cross_split_and_stage"],
            "rl_train_to_rl_eval": within_rl,
            "sft_train_to_sft_eval": within_sft,
            "sft_train_to_rl_eval": sft_train_to_rl_eval,
            "author_training_union_to_rl_eval": author_union_to_rl_eval,
            "exact_duplicate_rl_rows": exact_duplicate_rl,
            "same_instance_and_reference_relationships": (
                "full matrices in the bound data manifest distinguish exact rows, shared "
                "instances with any identical reference, and shared instances with disjoint "
                "reference sets"
            ),
            "classification": (
                "linked author evaluation is not globally held-out or leakage-free; context "
                "reproduction only"
            ),
        },
        "model_data_and_budget_evidence": {
            "matched_baseline_status": (
                "UNMATCHED_HOLD_CHECKPOINT_COMPUTE_REFERENCE_AND_SPLIT_COMPARABILITY"
            ),
            "sft_defaults": v1["model_data_and_budget_evidence"]["sft_defaults"],
            "rl_defaults": v1["model_data_and_budget_evidence"]["rl_defaults"],
            "evaluation_defaults": v1["model_data_and_budget_evidence"]["evaluation_defaults"],
            "sft_rows_per_task": sft_rows,
            "rl_rows": rl_rows,
            "rl_eval_rows": _row_counts(data, "rl", "eval"),
            "rl_completions_per_one_epoch": {task: rows * 8 for task, rows in rl_rows.items()},
            "conditional_single_process_rl_optimizer_steps": {
                task: rows // 8 for task, rows in rl_rows.items()
            },
            "conditional_single_process_sft_optimizer_steps_per_task": 31_250,
            "horizon_condition": (
                "optimizer-step counts assume one process, source-default batch/accumulation, "
                "one epoch, and pinned TRL RepeatRandomSampler semantics; not an executed receipt"
            ),
            "seed_evidence": {
                "sft_shuffle_seed": 42,
                "sft_trainer_seed": 42,
                "sft_lora_random_state": 42,
                "rl_sampler_and_trainer_seed": 42,
                "rl_lora_random_state": 42,
            },
            "resolved_since_v1": [
                "exact official RL item IDs, bytes, SHA256s, schemas, rows, instances, and references",
                "probable-lineage SFT byte hashes, schemas, rows, input identities, and references",
                "conditional one-process row-derived SFT and RL optimizer-step horizons",
                "all within-stage and cross-stage split overlap matrices",
            ],
            "unresolved_publication_requirements": [
                "exact per-task released SFT checkpoint identities and hashes remain unavailable",
                "the base-model revision and immutable base-model bytes are not pinned upstream",
                "actual paper world size, hardware, wall-clock, and executed training receipt are unavailable",
                "actual prompt/completion token counts and a matched compute/token budget remain unavailable",
                "official lora_only resume loses live q/k/v bias state and cannot reproduce uninterrupted training",
                "corrected frozen-reference variant intentionally differs from official evolving-bias KL semantics",
                "linked author eval is contaminated for CVRP, MIS, and OP; fresh disjoint IID is required",
                "the external Drive data has no separately observed redistribution license",
            ],
        },
        "classification_guards": {
            "both_reference_variants_hold": True,
            "no_training_or_evaluation_authorized": True,
            "no_sota_claim_from_this_contract": True,
            "author_eval_held_out_claim_allowed": False,
            "author_eval_leakage_free_claim_allowed": False,
            "fresh_preregistered_disjoint_iid_required": True,
            "fresh_iid_is_exact_paper_data_reproduction": False,
            "current_clasp_method_grpo": (
                "three strict reward heads with dr_grpo; non-FOARL and unmatched"
            ),
        },
        "repository_source_projection": _source_projection(repository_root),
    }
    if any(
        variant["executable"] for variant in contract["adapter_and_reference_variants"].values()
    ):
        raise ValueError("A reference variant was accidentally made executable")
    contract["contract_sha256"] = _self_hash(contract, "contract_sha256")
    return contract, output


def main() -> None:
    args = parse_args()
    contract, output = build_contract(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    seal_regular_file_read_only(output)
    file_sha256, metadata = sha256_regular_file_stable(output, require_read_only=True)
    print(f"created read-only {output}")
    print(f"contract_sha256={contract['contract_sha256']}")
    print(f"file_sha256={file_sha256}")
    print(f"bytes={metadata.st_size}")


if __name__ == "__main__":
    main()
