#!/usr/bin/env python3
"""Build immutable development checkpoint evidence from completed eval artifacts.

Identity is derived only from self-hashed evaluator output, the embedded/on-disk
``run_contract.json``, and the content-addressed checkpoint directory. CLI values supply
paths, never method/task/seed/step labels. The full frozen policy grid is required before an
atomic, read-only evidence receipt is created. No checkpoint or job state is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_sota as aggregate  # noqa: E402
import select_sota_checkpoint as selector  # noqa: E402

from clasp.models.checkpoint_fidelity import (  # noqa: E402
    RECEIPT_FILENAME as CHECKPOINT_FIDELITY_FILENAME,
)
from clasp.models.exact_peft_kl_post_step import (  # noqa: E402
    POST_STEP_RECEIPT_FILENAME as EXACT_PEFT_KL_POST_STEP_FILENAME,
)
from clasp.models.exact_peft_kl_post_step import (  # noqa: E402
    STEP1_RECEIPT_FILENAME as EXACT_PEFT_KL_STEP1_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (  # noqa: E402
    RECEIPT_FILENAME as EXACT_PEFT_KL_REFERENCE_FILENAME,
)
from clasp.models.publication_fidelity import (  # noqa: E402
    exact_peft_kl_post_step_lineage_identity,
    exact_peft_kl_runtime_identity,
    publication_fidelity_identity,
)

SCHEMA_VERSION = 2
EXCLUDED_MODEL_DIRECTORIES = {
    ".cache",
    ".git",
    "__pycache__",
    "checkpoints",
    "output",
    "outputs",
    "runs",
    "wandb",
}
INFERENCE_EXACT_NAMES = {
    "adapter_config.json",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    CHECKPOINT_FIDELITY_FILENAME,
    EXACT_PEFT_KL_REFERENCE_FILENAME,
    EXACT_PEFT_KL_STEP1_FILENAME,
    EXACT_PEFT_KL_POST_STEP_FILENAME,
}
INFERENCE_SUFFIXES = (".jinja", ".model", ".py", ".safetensors", ".tiktoken")
INFERENCE_WEIGHT_PREFIXES = ("adapter_model", "model", "pytorch_model")


def _is_inference_artifact(relative_path: str) -> bool:
    name = os.path.basename(relative_path)
    return (
        name in INFERENCE_EXACT_NAMES
        or name.endswith(INFERENCE_SUFFIXES)
        or (name.endswith(".bin") and name.startswith(INFERENCE_WEIGHT_PREFIXES))
    )


def _checkpoint_identity(path: Path) -> dict[str, object]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"resolved checkpoint directory does not exist: {root}")
    files: list[dict[str, object]] = []
    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_MODEL_DIRECTORIES
        )
        for name in sorted(file_names):
            file_path = Path(current_root) / name
            relative = file_path.relative_to(root).as_posix()
            if not _is_inference_artifact(relative):
                continue
            _raw_bytes, snapshot = aggregate._read_stable_file_bytes(str(file_path))
            files.append(
                {
                    "path": relative,
                    "size": snapshot.size_bytes,
                    "sha256": snapshot.sha256,
                }
            )
    if not files:
        raise ValueError(f"checkpoint has no inference artifacts: {root}")
    return {
        "kind": "local_directory",
        "basename": root.name,
        "manifest_sha256": hashlib.sha256(aggregate._canonical_json_bytes(files)).hexdigest(),
        "files": files,
    }


def _native_paths(details_path: Path) -> tuple[Path, Path]:
    name = details_path.name
    if not (name.startswith("run_details__") and name.endswith(".jsonl")):
        raise ValueError(
            f"checkpoint evidence accepts native run_details__LABEL.jsonl only: {details_path}"
        )
    label = name[len("run_details__") : -len(".jsonl")]
    return (
        details_path.with_name(f"run_config__{label}.json"),
        details_path.with_name(f"run_summary__{label}.json"),
    )


def _matching_run_contract(
    *,
    config: dict[str, object],
    expected_payload: dict[str, object],
    expected_sha256: str,
) -> tuple[Path, aggregate.FileSnapshot]:
    candidates: list[Path] = []
    run_dir = config.get("resolved_run_dir")
    if isinstance(run_dir, str) and run_dir:
        candidates.append(Path(run_dir).expanduser().resolve() / "run_contract.json")
    model_path = config.get("resolved_model_path")
    if isinstance(model_path, str) and model_path:
        current = Path(model_path).expanduser().resolve()
        for _ in range(5):
            candidates.append(current / "run_contract.json")
            if current.parent == current:
                break
            current = current.parent
    matching: list[tuple[Path, aggregate.FileSnapshot]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        raw_bytes, snapshot = aggregate._read_stable_file_bytes(str(candidate))
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"run contract is not valid UTF-8 JSON: {candidate}") from exc
        if snapshot.sha256 == expected_sha256 and payload == expected_payload:
            matching.append((candidate, snapshot))
    if len(matching) != 1:
        raise ValueError(
            "evaluation must resolve exactly one on-disk run_contract.json matching its "
            f"embedded payload/hash; observed={[path for path, _snapshot in matching]}"
        )
    return matching[0]


def _derive_row(
    details_path: Path,
    *,
    candidate_budget: int,
    development_contracts: dict[str, dict[str, Any]],
) -> dict[str, object]:
    details_path = details_path.expanduser().resolve()
    config_path, summary_path = _native_paths(details_path)
    if not summary_path.is_file():
        raise ValueError(f"native evaluation summary is missing: {summary_path}")

    evidence: list[aggregate.ArtifactEvidence] = []
    records = aggregate.load_run(
        aggregate.RunSpec("artifact-derived", 0, str(details_path)),
        expected_k=candidate_budget,
        publication_mode=True,
        evidence_sink=evidence,
    )
    if len(evidence) != 1 or evidence[0].artifact_type != "native":
        raise ValueError(f"details do not resolve to exactly one native artifact: {details_path}")
    artifact = evidence[0]
    config = aggregate._load_json_object(str(config_path))
    summary = aggregate._load_json_object(str(summary_path))
    provenance = artifact.experiment_provenance
    selection = provenance.get("selection")
    if (
        config.get("evaluation_purpose") != "development"
        or summary.get("evaluation_purpose") != "development"
        or not isinstance(selection, dict)
        or selection.get("evaluation_purpose") != "development"
        or selection.get("benchmark_manifest") is not None
        or selection.get("checkpoint_selection_binding") is not None
    ):
        raise ValueError("checkpoint selection evidence must be development-only")

    contract = artifact.training_run_contract
    contract_sha256 = artifact.training_run_contract_sha256
    if not isinstance(contract, dict) or not aggregate._is_sha256(contract_sha256):
        raise ValueError("evaluation has no immutable training contract")
    method = str(contract.get("method") or "")
    task = str(contract.get("task") or "")
    seeds = contract.get("seeds")
    if not isinstance(seeds, dict) or isinstance(seeds.get("train"), bool):
        raise ValueError("training contract has no integer train seed")
    train_seed = int(seeds.get("train", -1))
    development_contract = development_contracts.get(task)
    if development_contract is None:
        raise ValueError(f"selection policy has no development contract for task {task}")
    aggregate._validate_training_contract(
        contract,
        contract_sha256,
        method=method,
        seed=train_seed,
        task=task,
        label=str(details_path),
    )
    run_contract_path, run_contract_snapshot = _matching_run_contract(
        config=config,
        expected_payload=contract,
        expected_sha256=str(contract_sha256),
    )

    model_path_raw = config.get("resolved_model_path")
    if not isinstance(model_path_raw, str) or not model_path_raw:
        raise ValueError("evaluation config has no resolved_model_path")
    model_path = Path(model_path_raw).expanduser().resolve()
    model_match = re.fullmatch(r"checkpoint-(\d+)", model_path.name)
    if model_match is None:
        raise ValueError("resolved model artifact is not a checkpoint-STEP directory")
    checkpoint_step = int(model_match.group(1))
    checkpoint_name = str(config.get("checkpoint_name") or "")
    label_match = re.search(r"(?:^|-)checkpoint-(\d+)$", checkpoint_name)
    if label_match is None or int(label_match.group(1)) != checkpoint_step:
        raise ValueError("evaluation checkpoint label disagrees with resolved model artifact")
    observed_checkpoint = _checkpoint_identity(model_path)
    checkpoint_fidelity = publication_fidelity_identity(model_path)
    beta = float(contract["training_config"]["beta"])
    fixed_kl_reference = contract["fixed_kl_reference"]
    runtime_paths = {
        "root": run_contract_path.parent / EXACT_PEFT_KL_REFERENCE_FILENAME,
        "checkpoint": model_path / EXACT_PEFT_KL_REFERENCE_FILENAME,
    }
    post_step_paths = {
        "root_step1": run_contract_path.parent / EXACT_PEFT_KL_STEP1_FILENAME,
        "root_final": run_contract_path.parent / EXACT_PEFT_KL_POST_STEP_FILENAME,
        "checkpoint_step1": model_path / EXACT_PEFT_KL_STEP1_FILENAME,
        "checkpoint_post_step": model_path / EXACT_PEFT_KL_POST_STEP_FILENAME,
    }
    exact_runtime: dict[str, Any] | None
    exact_post_step: dict[str, Any] | None
    runtime_source_artifacts: dict[str, object] | None
    if beta == 0.0:
        stale = sorted(
            str(path)
            for path in (*runtime_paths.values(), *post_step_paths.values())
            if os.path.lexists(path)
        )
        if stale:
            raise ValueError(
                "beta=0 checkpoint evidence carries an unexpected exact PEFT KL runtime: "
                + ", ".join(stale)
            )
        exact_runtime = None
        exact_post_step = None
        runtime_source_artifacts = None
    else:
        missing = sorted(str(path) for path in runtime_paths.values() if not path.is_file())
        if missing:
            raise ValueError(
                "beta>0 checkpoint evidence is missing an exact PEFT KL runtime receipt: "
                + ", ".join(missing)
            )
        runtime_identities = {
            label: exact_peft_kl_runtime_identity(
                path,
                expected_fixed_kl_reference=fixed_kl_reference,
            )
            for label, path in runtime_paths.items()
        }
        if runtime_identities["root"] != runtime_identities["checkpoint"]:
            raise ValueError(
                "root and checkpoint exact PEFT KL runtime receipts are not byte-identical"
            )
        exact_runtime = runtime_identities["root"]
        if (
            exact_runtime["training_beta"] != beta
            or exact_runtime["training_dtype"] != contract["training_config"].get("dtype")
            or exact_runtime["distributed_world_size"]
            != contract["fixed_horizon"].get("world_size")
        ):
            raise ValueError(
                "exact PEFT KL runtime training settings disagree with the run contract"
            )
        runtime_source_artifacts = {
            label: {
                "path": str(path),
                "receipt_file_sha256": exact_runtime["receipt_file_sha256"],
            }
            for label, path in runtime_paths.items()
        }
        exact_post_step = exact_peft_kl_post_step_lineage_identity(
            run_contract_path.parent,
            model_path,
            expected_fixed_kl_reference=fixed_kl_reference,
            expected_runtime_identity=exact_runtime,
            training_run_contract_file_sha256=run_contract_snapshot.sha256,
            terminal_global_step=int(contract["fixed_horizon"]["max_steps"]),
        )
        runtime_source_artifacts["post_step"] = {
            label: {
                "path": str(path),
                "receipt_file_sha256": (
                    exact_post_step[
                        {
                            "root_step1": "root_step1",
                            "root_final": "root_final",
                            "checkpoint_step1": "selected_checkpoint_step1",
                            "checkpoint_post_step": "selected_checkpoint_post_step",
                        }[label]
                    ]["receipt_file_sha256"]
                ),
            }
            for label, path in post_step_paths.items()
        }
    checkpoint_files = {
        str(item["path"]): item
        for item in observed_checkpoint["files"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    required_fidelity_files = {
        CHECKPOINT_FIDELITY_FILENAME: checkpoint_fidelity["receipt_file_sha256"],
        "adapter_config.json": checkpoint_fidelity["adapter_config_sha256"],
        "adapter_model.safetensors": checkpoint_fidelity["adapter_weights_sha256"],
    }
    if any(
        not isinstance(checkpoint_files.get(name), dict)
        or checkpoint_files[name].get("sha256") != expected
        for name, expected in required_fidelity_files.items()
    ):
        raise ValueError("checkpoint artifact manifest disagrees with checkpoint fidelity")
    runtime_artifact = checkpoint_files.get(EXACT_PEFT_KL_REFERENCE_FILENAME)
    step1_artifact = checkpoint_files.get(EXACT_PEFT_KL_STEP1_FILENAME)
    post_step_artifact = checkpoint_files.get(EXACT_PEFT_KL_POST_STEP_FILENAME)
    if beta > 0.0:
        if (
            not isinstance(runtime_artifact, dict)
            or runtime_artifact.get("sha256") != exact_runtime["receipt_file_sha256"]
        ):
            raise ValueError("checkpoint artifact manifest disagrees with exact PEFT KL runtime")
        if (
            not isinstance(step1_artifact, dict)
            or step1_artifact.get("sha256")
            != exact_post_step["selected_checkpoint_step1"]["receipt_file_sha256"]
            or not isinstance(post_step_artifact, dict)
            or post_step_artifact.get("sha256")
            != exact_post_step["selected_checkpoint_post_step"]["receipt_file_sha256"]
        ):
            raise ValueError("checkpoint artifact manifest disagrees with post-step exact KL")
    elif any(value is not None for value in (runtime_artifact, step1_artifact, post_step_artifact)):
        raise ValueError("beta=0 checkpoint artifact contains exact PEFT KL evidence")
    model = provenance.get("model")
    expected_checkpoint = model.get("checkpoint_artifact") if isinstance(model, dict) else None
    if expected_checkpoint != observed_checkpoint:
        raise ValueError("resolved checkpoint bytes disagree with experiment provenance")
    if not isinstance(model, dict) or model.get("checkpoint_fidelity") != checkpoint_fidelity:
        raise ValueError(
            "evaluator provenance does not bind the live publication checkpoint fidelity"
        )
    if "exact_peft_kl_runtime" not in model or model.get("exact_peft_kl_runtime") != exact_runtime:
        raise ValueError("evaluator provenance does not bind the live exact PEFT KL runtime")
    if (
        "exact_peft_kl_post_step" not in model
        or model.get("exact_peft_kl_post_step") != exact_post_step
    ):
        raise ValueError("evaluator provenance does not bind post-step exact PEFT KL lineage")

    observed_tasks = sorted({key[0] for key in records})
    if observed_tasks != [task] or selection.get("tasks") != [task]:
        raise ValueError("evaluation task set disagrees with the embedded training contract")
    observed_indices = sorted(key[2] for key in records)
    if observed_indices != list(range(len(records))):
        raise ValueError("checkpoint evaluation details have skipped/reordered indices")

    feasible_records = [record for record in records.values() if aggregate._is_feasible(record)]
    gaps: list[float] = []
    for record in feasible_records:
        gap = aggregate._gap(record)
        if gap is None or gap < 0.0:
            raise ValueError("feasible checkpoint record has no finite nonnegative gap")
        gaps.append(gap)
    feasible_instances = len(feasible_records)
    instances = len(records)
    data = provenance.get("data")
    verifier = provenance.get("verifier")
    if not isinstance(data, dict) or not isinstance(verifier, dict):
        raise ValueError("evaluation data/verifier provenance is incomplete")
    selection_projection = {
        field: selection.get(field)
        for field in (
            "split",
            "eval_start_index",
            "max_eval_samples_per_task",
            "eval_samples_per_task",
            "size_bucket",
            "min_instance_size",
            "max_instance_size",
            "prefer_subset_files",
        )
    }
    expected_selection = {field: development_contract[field] for field in selection_projection}
    if (
        data.get("manifest_sha256") != development_contract["data_manifest_sha256"]
        or config.get("protocol_fingerprint") != development_contract["protocol_fingerprint"]
        or verifier.get("implementation_manifest_sha256")
        != development_contract["strict_verifier_manifest_sha256"]
        or selection_projection != expected_selection
        or instances != development_contract["expected_instances"]
    ):
        raise ValueError("development evaluation does not match the pre-frozen corpus/slice policy")
    row: dict[str, object] = {
        "task": task,
        "method": method,
        "train_seed": train_seed,
        "checkpoint_step": checkpoint_step,
        "candidate_budget": candidate_budget,
        "instances": instances,
        "feasible_instances": feasible_instances,
        "pass_at_k": feasible_instances / instances,
        "conditional_gap_n": feasible_instances,
        "conditional_gap_mean": (sum(gaps) / len(gaps) if gaps else None),
        "evaluation_fingerprint": config.get("experiment_fingerprint"),
        "protocol_fingerprint": config.get("protocol_fingerprint"),
        "development_evaluation_sha256": development_contract["development_evaluation_sha256"],
        "checkpoint_artifact_sha256": observed_checkpoint["manifest_sha256"],
        "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
        "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
        "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity["receipt_file_sha256"],
        "grpo_beta": beta,
        "fixed_kl_reference": fixed_kl_reference,
        "exact_peft_kl_runtime": exact_runtime,
        "exact_peft_kl_post_step": exact_post_step,
        "run_contract_sha256": contract_sha256,
        "data_manifest_sha256": data.get("manifest_sha256"),
        "strict_verifier_manifest_sha256": verifier.get("implementation_manifest_sha256"),
        "source_artifacts": {
            "details": {
                "path": str(details_path),
                **(config.get("details_artifact") or {}),
            },
            "config": {
                "path": str(config_path),
                "run_config_sha256": config.get("run_config_sha256"),
            },
            "summary": {
                "path": str(summary_path),
                "summary_sha256": summary.get("summary_sha256"),
            },
            "run_contract": {
                "path": str(run_contract_path),
                "sha256": contract_sha256,
                "size_bytes": run_contract_snapshot.size_bytes,
            },
            "checkpoint": {"path": str(model_path), **observed_checkpoint},
            "checkpoint_fidelity": checkpoint_fidelity,
            "exact_peft_kl_runtime": {
                "identity": exact_runtime,
                "receipts": runtime_source_artifacts,
            },
            "exact_peft_kl_post_step": exact_post_step,
        },
    }
    key, errors = selector._validate_record(
        row,
        candidate_budget=candidate_budget,
        row_index=0,
    )
    if key is None or errors:
        raise ValueError("derived checkpoint evidence row is invalid: " + " | ".join(errors))
    return row


def build_evidence(*, policy_path: Path, detail_paths: Sequence[Path]) -> dict[str, object]:
    policy_path = policy_path.expanduser().resolve()
    policy = selector._validate_policy(policy_path)
    development_contracts = selector._validate_development_evaluation_policy(
        policy, tasks=policy["expected_tasks"]
    )
    if not detail_paths:
        raise ValueError("at least one --details artifact is required")
    rows = [
        _derive_row(
            path,
            candidate_budget=int(policy["candidate_budget"]),
            development_contracts=development_contracts,
        )
        for path in detail_paths
    ]
    rows.sort(
        key=lambda row: (
            str(row["task"]),
            str(row["method"]),
            int(row["train_seed"]),
            int(row["checkpoint_step"]),
        )
    )
    keys = [
        (
            str(row["task"]),
            str(row["method"]),
            int(row["train_seed"]),
            int(row["checkpoint_step"]),
        )
        for row in rows
    ]
    expected = sorted(selector._expected_keys(policy))
    if keys != expected:
        raise ValueError(
            "derived checkpoint artifacts do not equal the frozen policy grid: "
            f"missing={sorted(set(expected) - set(keys))[:8]}, "
            f"unexpected={sorted(set(keys) - set(expected))[:8]}"
        )
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate derived checkpoint evidence identity")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        group = (str(row["task"]), str(row["method"]), int(row["train_seed"]))
        grouped.setdefault(group, []).append(row)
    group_errors = {
        group: selector._validate_group_provenance(group_rows)
        for group, group_rows in grouped.items()
    }
    group_errors = {group: errors for group, errors in group_errors.items() if errors}
    if group_errors:
        raise ValueError(f"derived checkpoint groups have inconsistent provenance: {group_errors}")

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": policy["campaign_id"],
        "status": "complete",
        "evidence_scope": "development_only",
        "candidate_budget": policy["candidate_budget"],
        "policy": {
            "path": str(policy_path),
            "policy_sha256": policy["policy_sha256"],
            "file_sha256": selector._sha256_file(policy_path),
        },
        "records": rows,
        "builder_side_effects": {
            "checkpoints_modified": False,
            "evaluator_state_modified": False,
            "scheduler_actions": [],
        },
    }
    payload["evidence_sha256"] = hashlib.sha256(selector._canonical_json_bytes(payload)).hexdigest()
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace checkpoint evidence: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace checkpoint evidence: {path}") from exc
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--details",
        action="append",
        required=True,
        type=Path,
        help="Completed native development run_details__LABEL.jsonl; repeat for the grid.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_evidence(policy_path=args.policy, detail_paths=args.details)
        _atomic_json(args.output, payload)
    except (OSError, TypeError, ValueError) as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 2
    print(f"COMPLETE: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
