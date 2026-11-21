#!/usr/bin/env python3
"""Aggregate strict CLASP evaluations without hiding seeds or missing records.

Example::

    python scripts/aggregate_sota.py \
      --run grpo_z,42,tsp,inference/grpo_z/seed42/tsp \
      --run clasp_canonical,42,tsp,inference/clasp_canonical/seed42/tsp \
      --baseline grpo_z --challenger clasp_canonical \
      --campaign-manifest experiments/sota_20260814/campaign_manifest.json \
      --targets experiments/sota_20260814/targets.json \
      --lane raw_n8 --output results/sota/aggregate.json

Each run path may be one JSONL file or a directory containing
``run_details__*.jsonl`` files.  A run must contain at most one record for a
``(task, split, example_index)`` key.  This fail-closed rule prevents resumed or
copied shards from silently changing the denominator.  The example shows syntax
only; confirmation requires the frozen campaign's complete two-method,
eight-seed, five-task Cartesian grid.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    validate_foarl_f638_campaign_contract,
)
from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    FOARL_MATCHED_B9_CONFIRMATION_PAIR,
    FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST,
    foarl_matched_b9_shared_design_sha256,
    is_foarl_matched_b9_confirmation_pair,
    validate_clasp_foarl_matched_b9_method_contract,
)
from clasp.data.benchmark import (
    benchmark_row_sha256,
    forbidden_source_registry_identity,
    reference_output_sha256,
    semantic_instance_sha256,
    structured_instance,
    validate_forbidden_source_registry,
    validate_frozen_benchmark_manifest,
)
from clasp.data.loaders import (
    transform_example_for_task,
    transform_tsp_example_for_label_tta_variant,
    tsp_label_tta_schedule_from_env,
)
from clasp.entrypoints.evaluate import TASK_EVALUATORS, _constraint_diagnostics
from clasp.entrypoints.repair import (
    REPAIR_ALGORITHM_VERSION,
    REPAIR_SCHEMA_VERSION,
    repair_completion,
)
from clasp.entrypoints.repair import (
    _implementation_identity as _repair_implementation_identity,
)
from clasp.models.checkpoint_fidelity import RECEIPT_FILENAME as CHECKPOINT_FIDELITY_FILENAME
from clasp.models.exact_peft_kl_post_step import (
    POST_STEP_RECEIPT_FILENAME as EXACT_PEFT_KL_POST_STEP_FILENAME,
)
from clasp.models.exact_peft_kl_post_step import (
    STEP1_RECEIPT_FILENAME as EXACT_PEFT_KL_STEP1_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_REFERENCE_FILENAME,
)
from clasp.models.publication_fidelity import (
    exact_peft_kl_post_step_lineage_identity,
    exact_peft_kl_runtime_identity,
    fixed_kl_reference_identity,
    validate_derived_source_config_snapshot,
    validate_exact_peft_kl_post_step_lineage,
    validate_exact_peft_kl_runtime_identity,
    validate_fixed_kl_reference_identity,
    validate_publication_fidelity_identity,
)
from audit_sft_reload_equivalence import validate_reload_equivalence_identity

Record = dict[str, object]
RecordKey = tuple[str, str, int]

STRICT_VERIFIER_NAME = "clasp_strict_shared_verifier"
PUBLICATION_ARTIFACT_TYPES = frozenset({"native", "repair", "rescore"})
CANONICAL_TRAINING_METHODS = frozenset(
    {
        "grpo_z",
        "dual_scalar_z",
        "dual_separate_z",
        "clasp_canonical",
        "clasp_canonical_fixed_lambda",
        "clasp_canonical_no_recovery",
        CLASP_FOARL_MATCHED_B9,
        "set_null",
        "foarl_f638_native",
        "foarl_f638_matched_b9",
    }
)
LEGACY_TRAINING_METHOD_ALIASES = frozenset(
    {"grpo", "dual_grpo", "clasp_dense", "clasp_exact", "foarl_f638_corrected"}
)
# Kept as the public allowlist name used by the aggregation validators/tests.
# Legacy aliases are deliberately absent: confirmation artifacts must name the
# exact scientific arm that produced them.
TRAINING_METHODS = CANONICAL_TRAINING_METHODS
PUBLICATION_TASKS = ("cvrp", "jssp", "mis", "op", "tsp")
CONFIRMATION_TRAINING_SEEDS = tuple(range(42, 50))
CONFIRMATION_CAMPAIGN_SCHEMA_VERSION = 2
PRIMARY_GAP_RANGE = (0.0, 0.10)
PASS_AT_K_NONINFERIORITY_MARGIN = 0.01
CONFIRMATION_ALPHA = 0.05
MATCHED_SFT_CONTRACT_SHA256 = "dd294083e1643d08e833bca35be1ad275544b1bc59e4c0ddac44deb874f5bf48"
PRIOR_STAGE_RECEIPT_IDENTITY_KEYS = {
    "campaign_id",
    "stage_id",
    "policy_sha256",
    "receipt_sha256",
    "receipt_file_sha256",
}
TASK_DIRECTIONS = {
    "tsp": "min",
    "cvrp": "min",
    "op": "max",
    "mis": "max",
    "jssp": "min",
}
CHECKPOINT_SELECTION_RULE = [
    "maximize_pass_at_k",
    "retain_within_0.01_absolute_of_best_pass_at_k",
    "minimize_conditional_verified_gap",
    "tie_break_earlier_checkpoint_step",
]
CHECKPOINT_SELECTION_FINGERPRINT_FIELDS = (
    "evaluation_fingerprint",
    "protocol_fingerprint",
    "development_evaluation_sha256",
    "checkpoint_artifact_sha256",
    "checkpoint_fidelity_identity_sha256",
    "checkpoint_fidelity_receipt_sha256",
    "checkpoint_fidelity_receipt_file_sha256",
    "run_contract_sha256",
    "data_manifest_sha256",
    "strict_verifier_manifest_sha256",
)
_FIXED_KL_LIVE_CACHE: dict[tuple[str, str, str], dict[str, object]] = {}
DEVELOPMENT_EVALUATION_FIELDS = {
    "data_manifest_sha256",
    "evaluation_data_files",
    "forbidden_source_label",
    "forbidden_source_identity_sha256",
    "protocol_fingerprint",
    "strict_verifier_manifest_sha256",
    "split",
    "eval_start_index",
    "max_eval_samples_per_task",
    "eval_samples_per_task",
    "size_bucket",
    "min_instance_size",
    "max_instance_size",
    "prefer_subset_files",
    "expected_instances",
    "development_evaluation_sha256",
}
METHOD_SPECIFIC_TRAINING_ALLOWLIST = (
    "training_config.entrypoint",
    "training_config.policy_credit_formula",
    "training_config.component_controller",
    "training_config.set_controller",
    "training_config.use_primal_dual",
    "training_config.use_clasp",
    "training_config.clasp_canonical_set",
    "training_config.clasp_fixed_null_set",
    "training_config.pd_group_scalarization",
    "training_config.pd_separate_z",
    "training_config.pd_dual_lr",
    "training_config.pd_dual_lr_up",
    "training_config.pd_dual_lr_down",
    "training_config.clasp_bon_dual_lr",
    "training_config.clasp_bon_lambda_init",
    "training_config.clasp_canonical_recovery_scale",
    "tracked_environment.CLASP_BON_ALIGNED",
    "tracked_environment.CLASP_FEASIBILITY_GATED",
    "tracked_environment.CLASP_CANONICAL_SET",
    "tracked_environment.CLASP_FIXED_NULL_SET",
    "tracked_environment.CLASP_BON_DUAL_LR",
    "tracked_environment.CLASP_BON_LAMBDA_INIT",
    "tracked_environment.CLASP_CANONICAL_RECOVERY_SCALE",
)
CONFIRMATION_OUTCOME_FAMILIES = (
    "normalized_feasible_gap_auc_seed_randomization",
)
CONFIRMATION_NONINFERIORITY_FAMILY = "pass_at_k_noninferiority_seed_randomization"
APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK = {
    "cvrp": "d4c7f8e4924bee2bf54b3f85c5eb39f69293d858dda855685d2bacff703df217",
    "jssp": "f67cc9befc71086ab1e5bcf419af731ba6e9a62647fc268684b83b997954fac5",
    "mis": "ab9b34ec5d97854489a3a7bf10d2875c918565ff1d6e486a542d35304cd256d2",
    "op": "e5bc6baa29ba685fd4806599d9221fe81a3c5b68e63b7dbf11a8bc7d6002aeb4",
    "tsp": "26d59b60957f7cc866947bdb99baaf0c5aaddeaf68720465e76e352213b30dbf",
}
# Trust anchor for the one authoritative external-context registry.  Numeric
# points and task semantics intentionally live only in targets.json; this hash
# prevents a caller from resealing easier values and presenting them as the
# audited registry.  Update it only with a reviewed registry revision.
APPROVED_TARGET_REGISTRY_SHA256 = "b0753d8befe3beb87ae983e8b4b82b41b2c8511b056356109a4f2b0955cae706"
APPROVED_REPAIR_SYSTEM_NAME = "clasp_repair_v1"
APPROVED_REPAIR_SYSTEM_LANE = "deterministic_repair_and_improve_hybrid"
APPROVED_REPAIR_CONFIG = {
    "schema_version": REPAIR_SCHEMA_VERSION,
    "algorithm_version": REPAIR_ALGORITHM_VERSION,
    "tsp_max_passes": 12,
    "cvrp_max_passes": 8,
    "op_max_passes": 8,
    "jssp_max_passes": 2,
    "selection_policy": "strict_feasibility_then_task_objective",
    "ground_truth_access_policy": "verifier_only",
}
# These are the always-active structural diagnostics once parsing succeeds and
# the frozen benchmark supplies the task metadata. Optional objective-shaping
# diagnostics may also be present, but publication artifacts may not omit these.
TASK_CORE_CONSTRAINTS = {
    "tsp": frozenset(
        {
            "parse_failure",
            "depot_boundary",
            "route_length_mismatch",
            "invalid_node",
            "duplicate_customer",
            "missing_customer",
            "interior_depot",
        }
    ),
    "cvrp": frozenset(
        {
            "parse_failure",
            "start_at_depot",
            "end_at_depot",
            "depot_boundary",
            "route_shape_invalid",
            "invalid_node",
            "duplicate_customer",
            "missing_customer",
            "customer_count_mismatch",
            "interior_depot",
            "capacity_overflow",
            "over_capacity_route_fraction",
            "max_route_capacity_overflow",
        }
    ),
    "op": frozenset(
        {
            "parse_failure",
            "start_at_depot",
            "invalid_node",
            "duplicate_visit",
            "length_overflow",
        }
    ),
    "mis": frozenset({"parse_failure", "invalid_node", "duplicate_node", "edge_conflict"}),
    "jssp": frozenset(
        {
            "parse_failure",
            "machine_row_mismatch",
            "row_length_mismatch",
            "invalid_job",
            "duplicate_job",
            "missing_job",
            "row_set_mismatch",
            "schedule_invalid",
        }
    ),
}
DEFAULT_LANE_CONTRACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "sota_20260814",
    "lane_contracts.json",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: str) -> tuple[str, int]:
    """Hash one stable file snapshot, rejecting concurrent replacement."""

    before = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = os.stat(path)
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError(f"artifact changed while it was being hashed: {path}")
    return digest.hexdigest(), int(after.st_size)


def _aggregation_implementation_identity(path: str | None = None) -> dict[str, object]:
    """Bind the sealed result to the exact analysis implementation/runtime."""

    implementation_path = os.path.realpath(path or __file__)
    digest, size = _sha256_file(implementation_path)
    return {
        "schema_version": 1,
        "implementation": {
            "basename": os.path.basename(implementation_path),
            "size_bytes": size,
            "sha256": digest,
        },
        "runtime": {
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "python_cache_tag": sys.implementation.cache_tag,
        },
    }


def _current_strict_verifier_identity() -> dict[str, object]:
    source_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "clasp",
    )
    implementations: list[dict[str, object]] = []
    for relative_path in ("rewards/co_tasks.py", "entrypoints/evaluate.py"):
        digest, size = _sha256_file(os.path.join(source_root, *relative_path.split("/")))
        implementations.append({"path": relative_path, "size": size, "sha256": digest})
    return {
        "name": STRICT_VERIFIER_NAME,
        "schema_version": 1,
        "implementation_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(implementations)
        ).hexdigest(),
        "implementations": implementations,
        "tasks": sorted(TASK_EVALUATORS),
    }


def _validate_fixed_kl_projection(
    beta: object,
    value: object,
    *,
    label: str,
) -> tuple[float, dict[str, object] | None]:
    parsed_beta = _finite_float(beta)
    if parsed_beta is None or parsed_beta < 0.0:
        raise ValueError(f"{label} has invalid grpo_beta")
    if parsed_beta == 0.0:
        if value is not None:
            raise ValueError(f"{label} beta=0 must not bind a fixed-KL reference")
        return parsed_beta, None
    try:
        identity = validate_fixed_kl_reference_identity(value)
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 fixed-KL identity is invalid: {exc}") from exc
    return parsed_beta, identity


def _validate_exact_runtime_projection(
    beta: object,
    fixed_kl_reference: object,
    runtime_value: object,
    *,
    label: str,
) -> tuple[float, dict[str, object] | None, dict[str, object] | None]:
    parsed_beta, fixed = _validate_fixed_kl_projection(
        beta,
        fixed_kl_reference,
        label=label,
    )
    if parsed_beta == 0.0:
        if runtime_value is not None:
            raise ValueError(f"{label} beta=0 must not bind an exact PEFT KL runtime")
        return parsed_beta, None, None
    try:
        runtime = validate_exact_peft_kl_runtime_identity(runtime_value)
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 exact PEFT KL runtime is invalid: {exc}") from exc
    if runtime["fixed_kl_reference"] != fixed:
        raise ValueError(f"{label} exact PEFT KL runtime differs from its fixed-KL source")
    if float(runtime["training_beta"]) != parsed_beta:
        raise ValueError(f"{label} exact PEFT KL runtime beta differs from its projection")
    return parsed_beta, fixed, runtime


def _validate_exact_post_step_projection(
    beta: object,
    fixed_kl_reference: object,
    runtime_value: object,
    post_step_value: object,
    *,
    label: str,
    expected_run_contract_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
) -> tuple[
    float,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Validate the complete pre-step and post-optimization KL chronology."""

    parsed_beta, fixed, runtime = _validate_exact_runtime_projection(
        beta,
        fixed_kl_reference,
        runtime_value,
        label=label,
    )
    if parsed_beta == 0.0:
        if post_step_value is not None:
            raise ValueError(f"{label} beta=0 must not bind exact PEFT KL post-step evidence")
        return parsed_beta, None, None, None
    try:
        post_step = validate_exact_peft_kl_post_step_lineage(
            post_step_value,
            expected_fixed_kl_reference=fixed,
            expected_runtime_identity=runtime,
            expected_run_contract_file_sha256=expected_run_contract_sha256,
            expected_selected_checkpoint_step=expected_checkpoint_step,
        )
    except ValueError as exc:
        raise ValueError(f"{label} beta>0 post-step lineage is invalid: {exc}") from exc
    if post_step != post_step_value:
        raise ValueError(f"{label} exact PEFT KL post-step lineage is not canonical")
    return parsed_beta, fixed, runtime, post_step


def _validate_self_hash(payload: Mapping[str, object], field: str, *, label: str) -> str:
    stored = str(payload.get(field) or "")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored != observed:
        raise ValueError(
            f"{label} {field} mismatch: stored={stored or '<missing>'}, observed={observed}"
        )
    return stored


def _assert_finite_json(value: object, *, label: str) -> None:
    """Reject NaN/Inf anywhere in publication evidence."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, label=f"{label}[{index}]")


def _require_manifest_digest(
    identity: object,
    *,
    items_field: str,
    digest_field: str,
    label: str,
) -> None:
    if not isinstance(identity, dict) or not isinstance(identity.get(items_field), list):
        raise ValueError(f"{label} has no {items_field} manifest")
    observed = hashlib.sha256(_canonical_json_bytes(identity[items_field])).hexdigest()
    if identity.get(digest_field) != observed:
        raise ValueError(f"{label} {digest_field} is internally inconsistent")


def _training_contract_file_sha256(contract: Mapping[str, object]) -> str:
    """Recreate prepare_sota_run.py's immutable on-disk JSON encoding."""

    payload = _training_contract_file_bytes(contract)
    return hashlib.sha256(payload).hexdigest()


def _training_contract_file_bytes(contract: Mapping[str, object]) -> bytes:
    return (json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _validate_training_tree_receipt(receipt: object, *, label: str) -> None:
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} is not a tree fingerprint receipt")
    expected_keys = {"canonical_path", "sha256", "file_count", "size_bytes", "files"}
    if set(receipt) != expected_keys:
        raise ValueError(f"{label} does not use the official tree-receipt schema")
    canonical_path = receipt.get("canonical_path")
    files = receipt.get("files")
    file_count = receipt.get("file_count")
    size_bytes = receipt.get("size_bytes")
    if (
        not isinstance(canonical_path, str)
        or not canonical_path
        or not isinstance(files, list)
        or not files
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(files)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ValueError(f"{label} has invalid path/count/size fields")
    aggregate_digest = hashlib.sha256()
    observed_total = 0
    observed_paths: list[str] = []
    for index, raw_file in enumerate(files):
        if not isinstance(raw_file, dict) or set(raw_file) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"{label}.files[{index}] has an invalid schema")
        path = raw_file.get("path")
        file_size = raw_file.get("size_bytes")
        file_sha256 = raw_file.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or os.path.isabs(path)
            or path.startswith("../")
            or isinstance(file_size, bool)
            or not isinstance(file_size, int)
            or file_size < 0
            or not _is_sha256(file_sha256)
        ):
            raise ValueError(f"{label}.files[{index}] has invalid identity fields")
        observed_paths.append(path)
        observed_total += file_size
        aggregate_digest.update(path.encode("utf-8"))
        aggregate_digest.update(b"\0")
        aggregate_digest.update(str(file_size).encode("ascii"))
        aggregate_digest.update(b"\0")
        aggregate_digest.update(str(file_sha256).encode("ascii"))
        aggregate_digest.update(b"\n")
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(set(observed_paths)):
        raise ValueError(f"{label} file paths are not sorted and unique")
    if observed_total != size_bytes or aggregate_digest.hexdigest() != receipt.get("sha256"):
        raise ValueError(f"{label} aggregate hash/count/size is inconsistent")


def _publication_checkpoint_fidelity(
    provenance: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Require the evaluator's portable identity and its checkpoint-file closure."""

    model = provenance.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{label} model identity is missing")
    if "exact_peft_kl_runtime" not in model or "exact_peft_kl_post_step" not in model:
        raise ValueError(f"{label} model identity omits exact PEFT KL chronology")
    try:
        identity = validate_publication_fidelity_identity(model.get("checkpoint_fidelity"))
    except ValueError as exc:
        raise ValueError(f"{label} checkpoint fidelity is not publication-exact: {exc}") from exc
    artifact = model.get("checkpoint_artifact")
    files = artifact.get("files") if isinstance(artifact, dict) else None
    if not isinstance(files, list):
        raise ValueError(f"{label} checkpoint artifact has no file manifest")
    by_path = {
        str(item.get("path")): item
        for item in files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    expected = {
        CHECKPOINT_FIDELITY_FILENAME: identity["receipt_file_sha256"],
        "adapter_config.json": identity["adapter_config_sha256"],
        "adapter_model.safetensors": identity["adapter_weights_sha256"],
    }
    if any(
        not isinstance(by_path.get(path), dict) or by_path[path].get("sha256") != digest
        for path, digest in expected.items()
    ):
        raise ValueError(f"{label} checkpoint artifact disagrees with fidelity identity")
    return identity


def _publication_exact_peft_kl_runtime(
    provenance: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    label: str,
    live_config: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Validate embedded chronology and optionally reopen root/checkpoint receipts live."""

    model = provenance.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{label} model identity is missing")
    training = contract.get("training_config")
    horizon = contract.get("fixed_horizon")
    if not isinstance(training, dict) or not isinstance(horizon, dict):
        raise ValueError(f"{label} training contract has no runtime settings")
    beta, fixed, runtime = _validate_exact_runtime_projection(
        training.get("beta"),
        contract.get("fixed_kl_reference"),
        model.get("exact_peft_kl_runtime"),
        label=label,
    )
    artifact = model.get("checkpoint_artifact")
    files = artifact.get("files") if isinstance(artifact, dict) else None
    by_path = {
        str(item.get("path")): item
        for item in files or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    runtime_file = by_path.get(EXACT_PEFT_KL_REFERENCE_FILENAME)
    if beta == 0.0:
        if runtime_file is not None:
            raise ValueError(f"{label} beta=0 checkpoint artifact contains exact KL evidence")
    elif (
        not isinstance(runtime, dict)
        or not isinstance(runtime_file, dict)
        or runtime_file.get("sha256") != runtime["receipt_file_sha256"]
        or runtime["training_dtype"] != training.get("dtype")
        or runtime["distributed_world_size"] != horizon.get("world_size")
    ):
        raise ValueError(f"{label} exact PEFT KL runtime disagrees with artifact/run settings")
    if live_config is None:
        return runtime

    raw_checkpoint = live_config.get("resolved_model_path")
    raw_root = live_config.get("resolved_run_dir")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError(f"{label} has no resolved checkpoint path for live KL verification")
    checkpoint = os.path.realpath(os.path.abspath(raw_checkpoint))
    if isinstance(raw_root, str) and raw_root:
        root = os.path.realpath(os.path.abspath(raw_root))
    else:
        current = checkpoint
        root = ""
        for _ in range(5):
            if os.path.isfile(os.path.join(current, "run_contract.json")):
                root = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if not root:
            raise ValueError(f"{label} cannot locate live run root for exact KL verification")
    receipt_paths = {
        os.path.join(root, EXACT_PEFT_KL_REFERENCE_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_REFERENCE_FILENAME),
    }
    if beta == 0.0:
        stale = sorted(path for path in receipt_paths if os.path.lexists(path))
        if stale:
            raise ValueError(f"{label} beta=0 live artifacts carry exact PEFT KL receipts: {stale}")
        return None
    if not isinstance(fixed, dict) or not isinstance(runtime, dict):
        raise ValueError(f"{label} beta>0 lacks portable exact PEFT KL identities")
    identities = [
        exact_peft_kl_runtime_identity(path, expected_fixed_kl_reference=fixed)
        for path in sorted(receipt_paths)
    ]
    if any(identity != runtime for identity in identities):
        raise ValueError(
            f"{label} live root/checkpoint exact PEFT KL receipts differ from provenance"
        )
    return runtime


def _publication_exact_peft_kl_post_step(
    provenance: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    label: str,
    live_config: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Validate and, when possible, live-reopen the post-step reference lineage."""

    model = provenance.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{label} model identity is missing")
    training = contract.get("training_config")
    horizon = contract.get("fixed_horizon")
    if not isinstance(training, dict) or not isinstance(horizon, dict):
        raise ValueError(f"{label} training contract has no post-step settings")
    run_contract_sha256 = _training_contract_file_sha256(contract)
    selection = provenance.get("selection")
    selection_binding = (
        selection.get("checkpoint_selection_binding") if isinstance(selection, dict) else None
    )
    checkpoint_step = (
        selection_binding.get("checkpoint_step") if isinstance(selection_binding, dict) else None
    )
    beta, fixed, runtime, post_step = _validate_exact_post_step_projection(
        training.get("beta"),
        contract.get("fixed_kl_reference"),
        model.get("exact_peft_kl_runtime"),
        model.get("exact_peft_kl_post_step"),
        label=label,
        expected_run_contract_sha256=run_contract_sha256,
        expected_checkpoint_step=(checkpoint_step if isinstance(checkpoint_step, int) else None),
    )
    artifact = model.get("checkpoint_artifact")
    files = artifact.get("files") if isinstance(artifact, dict) else None
    by_path = {
        str(item.get("path")): item
        for item in files or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    step1_file = by_path.get(EXACT_PEFT_KL_STEP1_FILENAME)
    post_file = by_path.get(EXACT_PEFT_KL_POST_STEP_FILENAME)
    if beta == 0.0:
        if step1_file is not None or post_file is not None:
            raise ValueError(f"{label} beta=0 checkpoint artifact contains post-step KL evidence")
    elif (
        not isinstance(post_step, dict)
        or not isinstance(step1_file, dict)
        or not isinstance(post_file, dict)
        or step1_file.get("sha256") != post_step["selected_checkpoint_step1"]["receipt_file_sha256"]
        or post_file.get("sha256")
        != post_step["selected_checkpoint_post_step"]["receipt_file_sha256"]
        or post_step["run_invariant"]["training_run_contract"]["fixed_horizon_max_steps"]
        != horizon.get("max_steps")
        or post_step["run_invariant"]["training_run_contract"]["fixed_horizon_world_size"]
        != horizon.get("world_size")
    ):
        raise ValueError(f"{label} post-step KL lineage disagrees with artifact/run settings")
    if live_config is None:
        return post_step

    raw_checkpoint = live_config.get("resolved_model_path")
    raw_root = live_config.get("resolved_run_dir")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError(f"{label} has no resolved checkpoint path for live post-step verification")
    checkpoint = os.path.realpath(os.path.abspath(raw_checkpoint))
    if isinstance(raw_root, str) and raw_root:
        root = os.path.realpath(os.path.abspath(raw_root))
    else:
        current = checkpoint
        root = ""
        for _ in range(5):
            if os.path.isfile(os.path.join(current, "run_contract.json")):
                root = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if not root:
            raise ValueError(f"{label} cannot locate live run root for post-step verification")
    live_paths = {
        os.path.join(root, EXACT_PEFT_KL_STEP1_FILENAME),
        os.path.join(root, EXACT_PEFT_KL_POST_STEP_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_STEP1_FILENAME),
        os.path.join(checkpoint, EXACT_PEFT_KL_POST_STEP_FILENAME),
    }
    if beta == 0.0:
        stale = sorted(path for path in live_paths if os.path.lexists(path))
        if stale:
            raise ValueError(f"{label} beta=0 live artifacts carry post-step KL receipts: {stale}")
        return None
    if not isinstance(fixed, dict) or not isinstance(runtime, dict):
        raise ValueError(f"{label} beta>0 lacks exact pre-step authorization")
    max_steps = horizon.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"{label} fixed horizon is invalid")
    try:
        live = exact_peft_kl_post_step_lineage_identity(
            root,
            checkpoint,
            expected_fixed_kl_reference=fixed,
            expected_runtime_identity=runtime,
            training_run_contract_file_sha256=run_contract_sha256,
            terminal_global_step=max_steps,
        )
    except ValueError as exc:
        raise ValueError(f"{label} live post-step KL verification failed: {exc}") from exc
    if live != post_step:
        raise ValueError(f"{label} live post-step KL receipts differ from provenance")
    return post_step


def _protocol_fingerprint_material(provenance: Mapping[str, object]) -> dict[str, object]:
    """Mirror evaluator normalization for a method-independent shared protocol."""

    protocol = {
        key: value
        for key, value in provenance.items()
        if key not in {"model", "training_run_contract", "training_run_contract_sha256"}
    }
    model = provenance.get("model")
    if isinstance(model, dict):
        protocol["tokenizer"] = {
            "artifact": model.get("tokenizer_artifact"),
            "loaded": model.get("loaded_tokenizer"),
        }
    selection = protocol.get("selection")
    if isinstance(selection, dict):
        shared_selection = dict(selection)
        shared_selection.pop("checkpoint_selection_receipt", None)
        shared_selection.pop("checkpoint_selection_binding", None)
        shared_selection.pop("campaign_run_binding", None)
        protocol["selection"] = shared_selection
    return protocol


def _validate_training_contract(
    contract: object,
    contract_sha256: object,
    *,
    method: str,
    seed: int,
    task: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(contract, dict):
        raise ValueError(f"{label} has no embedded immutable training_run_contract")
    expected_contract_keys = {
        "schema_version",
        "stage",
        "method",
        "task",
        "seeds",
        "fixed_horizon",
        "data_lineage",
        "model_initialization_registry",
        "fixed_kl_reference",
        "fixed_null_receipt",
        "fingerprints",
        "generation",
        "training_config",
        "tracked_environment",
    }
    if set(contract) != expected_contract_keys:
        raise ValueError(f"{label} training run contract does not use the full official schema")
    if contract.get("schema_version") != 2 or contract.get("stage") != "grpo":
        raise ValueError(f"{label} has an unsupported training run contract")
    seeds = contract.get("seeds")
    if (
        not isinstance(seeds, dict)
        or set(seeds) != {"train", "data", "sft"}
        or any(
            isinstance(seeds.get(name), bool)
            or not isinstance(seeds.get(name), int)
            or int(seeds[name]) < 0
            for name in ("train", "data", "sft")
        )
        or seeds.get("data") != seeds.get("train")
    ):
        raise ValueError(f"{label} training run contract has invalid train/data/SFT seeds")
    horizon = contract.get("fixed_horizon")
    if (
        not isinstance(horizon, dict)
        or set(horizon) != {"max_steps", "world_size"}
        or any(
            isinstance(horizon.get(name), bool)
            or not isinstance(horizon.get(name), int)
            or int(horizon[name]) <= 0
            for name in ("max_steps", "world_size")
        )
    ):
        raise ValueError(f"{label} training run contract has an invalid fixed horizon")
    fingerprints = contract.get("fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "initial_model",
        "base_model",
        "data",
        "source",
    }:
        raise ValueError(f"{label} training run contract has incomplete fingerprints")
    for fingerprint_name in ("initial_model", "base_model", "data", "source"):
        _validate_training_tree_receipt(
            fingerprints[fingerprint_name],
            label=f"{label}.fingerprints.{fingerprint_name}",
        )
    _validate_training_data_lineage(
        contract.get("data_lineage"),
        task=str(contract.get("task") or ""),
        data_fingerprint=fingerprints.get("data"),
        label=label,
    )
    _validate_model_initialization_identity(
        contract.get("model_initialization_registry"),
        task=str(contract.get("task") or ""),
        seeds=seeds,
        fingerprints=fingerprints,
        data_lineage=contract.get("data_lineage"),
        label=label,
    )
    for object_name in ("generation", "training_config", "tracked_environment"):
        value = contract.get(object_name)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{label} training run contract has no {object_name}")
    training_config = contract["training_config"]
    tracked_environment = contract["tracked_environment"]
    if training_config.get("checkpoint_fidelity") != "publication_exact_required":
        raise ValueError(
            f"{label} training run contract does not require publication checkpoint fidelity"
        )
    if tracked_environment.get("CLASP_PUBLICATION_FIDELITY") != "1":
        raise ValueError(
            f"{label} training run contract did not enable publication checkpoint fidelity"
        )
    fixed_null_receipt = contract.get("fixed_null_receipt")
    fixed_null_path = tracked_environment.get("CLASP_FIXED_NULL_RECEIPT_PATH")
    if method == "set_null":
        expected_fixed_null_keys = {
            "schema_version",
            "protocol",
            "status",
            "receipt_sha256",
            "receipt_file_sha256",
            "null_candidate_sha256",
            "evaluator_identity_sha256",
            "record_count",
            "tasks",
        }
        if (
            not isinstance(fixed_null_receipt, dict)
            or set(fixed_null_receipt) != expected_fixed_null_keys
            or fixed_null_receipt.get("schema_version") != 1
            or fixed_null_receipt.get("protocol") != "clasp-fixed-null-construction-v1"
            or fixed_null_receipt.get("status") != "frozen_before_training"
            or fixed_null_receipt.get("tasks") != [task]
            or isinstance(fixed_null_receipt.get("record_count"), bool)
            or not isinstance(fixed_null_receipt.get("record_count"), int)
            or int(fixed_null_receipt["record_count"]) <= 0
            or any(
                not _is_sha256(fixed_null_receipt.get(field))
                for field in (
                    "receipt_sha256",
                    "receipt_file_sha256",
                    "null_candidate_sha256",
                    "evaluator_identity_sha256",
                )
            )
            or not isinstance(fixed_null_path, str)
            or not os.path.isabs(fixed_null_path)
            or os.path.realpath(fixed_null_path) != fixed_null_path
        ):
            raise ValueError(f"{label} set_null contract has no valid fixed-null receipt binding")
    elif fixed_null_receipt is not None or fixed_null_path not in {None, ""}:
        raise ValueError(f"{label} non-set_null contract carries a fixed-null receipt binding")
    beta = training_config.get("beta")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or float(beta) < 0.0
    ):
        raise ValueError(f"{label} training run contract has invalid beta")
    fixed_kl = contract.get("fixed_kl_reference")
    kl_model_path = tracked_environment.get("CLASP_KL_REFERENCE_MODEL_PATH")
    kl_receipt_path = tracked_environment.get("CLASP_KL_REFERENCE_RECEIPT")
    if float(beta) == 0.0:
        if fixed_kl is not None or kl_model_path is not None or kl_receipt_path is not None:
            raise ValueError(f"{label} beta=0 contract must not bind a fixed-KL reference")
    else:
        try:
            fixed_kl = validate_fixed_kl_reference_identity(fixed_kl)
        except ValueError as exc:
            raise ValueError(f"{label} beta>0 fixed-KL identity is invalid: {exc}") from exc
        if (
            not isinstance(kl_model_path, str)
            or not os.path.isabs(kl_model_path)
            or os.path.realpath(kl_model_path) != kl_model_path
            or not isinstance(kl_receipt_path, str)
            or not os.path.isabs(kl_receipt_path)
            or os.path.realpath(kl_receipt_path) != kl_receipt_path
        ):
            raise ValueError(f"{label} beta>0 contract has no canonical fixed-KL paths")
        initial_model = fingerprints.get("initial_model")
        expected_model_path = (
            initial_model.get("canonical_path") if isinstance(initial_model, dict) else None
        )
        if (
            not isinstance(expected_model_path, str)
            or os.path.realpath(expected_model_path) != kl_model_path
            or kl_receipt_path != os.path.join(kl_model_path, CHECKPOINT_FIDELITY_FILENAME)
        ):
            raise ValueError(f"{label} exact KL source is not the policy initialization adapter")
        source_identity = fixed_kl["source_checkpoint_fidelity"]
        initialization = contract.get("model_initialization_registry")
        specialists = (
            initialization.get("specialist_sft_by_task")
            if isinstance(initialization, dict)
            else None
        )
        specialist = specialists.get(task) if isinstance(specialists, dict) else None
        if not isinstance(specialist, dict) or source_identity != specialist.get(
            "checkpoint_fidelity"
        ):
            raise ValueError(
                f"{label} fixed-KL source differs from its approved specialist initialization"
            )
        cache_key = (kl_model_path, kl_receipt_path, str(fixed_kl["identity_sha256"]))
        live_identity = _FIXED_KL_LIVE_CACHE.get(cache_key)
        if live_identity is None:
            live_identity = fixed_kl_reference_identity(kl_model_path, kl_receipt_path)
            _FIXED_KL_LIVE_CACHE[cache_key] = live_identity
        if live_identity != fixed_kl:
            raise ValueError(f"{label} live fixed-KL reference differs from its run contract")
    if method in FOARL_F638_CAMPAIGN_METHODS:
        try:
            validate_foarl_f638_campaign_contract(
                method=method,
                task=task,
                train_seed=seed,
                final_step=int(horizon["max_steps"]),
                world_size=int(horizon["world_size"]),
                config=training_config,
                generation_config=contract["generation"],
                environment=tracked_environment,
            )
        except ValueError as exc:
            raise ValueError(f"{label} FOARL immutable lane validation failed: {exc}") from exc
    if method == CLASP_FOARL_MATCHED_B9:
        try:
            validate_clasp_foarl_matched_b9_method_contract(
                task=task,
                train_seed=seed,
                fixed_horizon=horizon,
                generation=contract["generation"],
                training_config=training_config,
                tracked_environment=tracked_environment,
            )
        except ValueError as exc:
            raise ValueError(
                f"{label} CLASP/FOARL matched-B9 lane validation failed: {exc}"
            ) from exc
    observed_identity = (
        str(contract.get("method") or ""),
        int(seeds.get("train", -1)),
        str(contract.get("task") or ""),
    )
    expected_identity = (method, int(seed), task)
    if observed_identity != expected_identity:
        raise ValueError(
            f"{label} CLI identity disagrees with embedded training contract: "
            f"expected={expected_identity}, embedded={observed_identity}"
        )
    if method in LEGACY_TRAINING_METHOD_ALIASES:
        raise ValueError(
            f"{label} uses prohibited legacy method alias {method!r}; "
            "use an exact canonical confirmation method ID"
        )
    if method not in TRAINING_METHODS or task not in PUBLICATION_TASKS:
        raise ValueError(f"{label} has an unsupported publication method/task")
    stored = str(contract_sha256 or "")
    observed = _training_contract_file_sha256(contract)
    if stored != observed:
        raise ValueError(
            f"{label} training_run_contract_sha256 mismatch: "
            f"stored={stored or '<missing>'}, observed={observed}"
        )
    return contract


def _validate_training_data_lineage(
    lineage: object,
    *,
    task: str,
    data_fingerprint: object,
    label: str,
) -> None:
    if not isinstance(lineage, dict) or set(lineage) != {
        "forbidden_source_registry",
        "sft_train",
        "rl_train",
    }:
        raise ValueError(f"{label} training contract has no exact data lineage")
    registry = lineage.get("forbidden_source_registry")
    expected_registry_keys = {
        "schema_version",
        "protocol",
        "status",
        "registry_sha256",
        "source_tree_manifest_sha256",
        "tasks",
        "required_labels",
        "source_count",
        "sources_by_task_label",
    }
    if (
        not isinstance(registry, dict)
        or set(registry) != expected_registry_keys
        or registry.get("schema_version") != 1
        or registry.get("protocol") != "clasp-forbidden-source-registry-v1"
        or registry.get("status") != "frozen_before_confirmation_generation"
        or not _is_sha256(registry.get("registry_sha256"))
        or not _is_sha256(registry.get("source_tree_manifest_sha256"))
        or task not in (registry.get("tasks") or [])
    ):
        raise ValueError(f"{label} training contract registry lineage is invalid")
    sources = registry.get("sources_by_task_label")
    task_sources = sources.get(task) if isinstance(sources, dict) else None
    if not isinstance(task_sources, dict):
        raise ValueError(f"{label} training contract registry lacks task lineage")
    binding_keys = {
        "task",
        "label",
        "canonical_path",
        "registry_sha256",
        "source_tree_manifest_sha256",
        "source_identity_sha256",
        "file_manifest_sha256",
        "rows",
        "unique_instance_hashes",
        "files",
    }
    for contract_field, source_label in (
        ("sft_train", "sft-train"),
        ("rl_train", "rl-train"),
    ):
        binding = lineage.get(contract_field)
        registered = task_sources.get(source_label)
        if (
            not isinstance(binding, dict)
            or set(binding) != binding_keys
            or not isinstance(registered, dict)
            or binding.get("task") != task
            or binding.get("label") != source_label
            or not isinstance(binding.get("canonical_path"), str)
            or not os.path.isabs(str(binding.get("canonical_path")))
            or binding.get("registry_sha256") != registry.get("registry_sha256")
            or binding.get("source_tree_manifest_sha256")
            != registry.get("source_tree_manifest_sha256")
            or any(
                binding.get(field) != registered.get(field)
                for field in (
                    "source_identity_sha256",
                    "file_manifest_sha256",
                    "rows",
                    "unique_instance_hashes",
                    "files",
                )
            )
        ):
            raise ValueError(f"{label} training contract {source_label} lineage is invalid")
    data_receipt = data_fingerprint if isinstance(data_fingerprint, dict) else {}
    rl_binding = lineage["rl_train"]
    data_files = data_receipt.get("files")
    registered_files = rl_binding.get("files") if isinstance(rl_binding, dict) else None
    expected_data_files = [
        {
            "path": row.get("relative_path"),
            "size_bytes": row.get("size_bytes"),
            "sha256": row.get("sha256"),
        }
        for row in (registered_files or [])
        if isinstance(row, dict)
    ]
    if data_files != expected_data_files:
        raise ValueError(f"{label} training data fingerprint differs from registered RL data")


def _validate_model_initialization_identity(
    identity: object,
    *,
    task: str,
    seeds: Mapping[str, object],
    fingerprints: Mapping[str, object],
    data_lineage: object,
    label: str,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "registry_sha256",
        "model_family",
        "base_model_tree_sha256",
        "forbidden_source_registry_sha256",
        "forbidden_source_tree_manifest_sha256",
        "specialist_sft_by_task",
    }
    specialists = identity.get("specialist_sft_by_task") if isinstance(identity, dict) else None
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_keys
        or identity.get("schema_version") != 2
        or identity.get("protocol") != "clasp-model-initialization-registry-v2"
        or identity.get("status") != "frozen_before_training"
        or identity.get("model_family") != "Qwen2.5-7B"
        or any(
            not _is_sha256(identity.get(field))
            for field in (
                "registry_sha256",
                "base_model_tree_sha256",
                "forbidden_source_registry_sha256",
                "forbidden_source_tree_manifest_sha256",
            )
        )
        or not isinstance(specialists, dict)
        or set(specialists) != set(PUBLICATION_TASKS)
    ):
        raise ValueError(f"{label} has an invalid model-initialization registry identity")
    for specialist_task, specialist in specialists.items():
        if (
            not isinstance(specialist, dict)
            or set(specialist)
            != {
                "initial_model_tree_sha256",
                "sft_train_source_identity_sha256",
                "sft_seed",
                "matched_sft_contract_sha256",
                "checkpoint_fidelity",
                "reload_equivalence",
            }
            or any(
                not _is_sha256(specialist.get(field))
                for field in (
                    "initial_model_tree_sha256",
                    "sft_train_source_identity_sha256",
                    "matched_sft_contract_sha256",
                )
            )
            or isinstance(specialist.get("sft_seed"), bool)
            or not isinstance(specialist.get("sft_seed"), int)
            or specialist.get("sft_seed") != 42
            or specialist.get("matched_sft_contract_sha256") != MATCHED_SFT_CONTRACT_SHA256
        ):
            raise ValueError(
                f"{label} has an invalid specialist initialization for {specialist_task}"
            )
        try:
            validate_publication_fidelity_identity(specialist.get("checkpoint_fidelity"))
        except ValueError as exc:
            raise ValueError(
                f"{label} has invalid specialist fidelity for {specialist_task}: {exc}"
            ) from exc
        try:
            reload_equivalence = validate_reload_equivalence_identity(
                specialist.get("reload_equivalence")
            )
        except ValueError as exc:
            raise ValueError(
                f"{label} specialist reload equivalence is invalid for {specialist_task}: {exc}"
            ) from exc
        checkpoint_fidelity = specialist["checkpoint_fidelity"]
        if (
            reload_equivalence["checkpoint_fidelity_identity_sha256"]
            != checkpoint_fidelity["identity_sha256"]
        ):
            raise ValueError(
                f"{label} specialist reload equivalence targets another checkpoint for {specialist_task}"
            )
    specialist = specialists.get(task)
    lineage = data_lineage if isinstance(data_lineage, dict) else {}
    registry = lineage.get("forbidden_source_registry")
    sft_binding = lineage.get("sft_train")
    initial = fingerprints.get("initial_model")
    base = fingerprints.get("base_model")
    if (
        not isinstance(specialist, dict)
        or not isinstance(registry, dict)
        or not isinstance(sft_binding, dict)
        or not isinstance(initial, dict)
        or not isinstance(base, dict)
        or identity.get("forbidden_source_registry_sha256") != registry.get("registry_sha256")
        or identity.get("forbidden_source_tree_manifest_sha256")
        != registry.get("source_tree_manifest_sha256")
        or identity.get("base_model_tree_sha256") != base.get("sha256")
        or specialist.get("initial_model_tree_sha256") != initial.get("sha256")
        or specialist.get("sft_train_source_identity_sha256")
        != sft_binding.get("source_identity_sha256")
        or specialist.get("sft_seed") != seeds.get("sft")
    ):
        raise ValueError(f"{label} model initialization differs from its data/model fingerprints")
    initial_files = initial.get("files")
    initial_by_path = {
        str(item.get("path")): item
        for item in initial_files
        if isinstance(initial_files, list)
        and isinstance(item, dict)
        and isinstance(item.get("path"), str)
    }
    checkpoint_fidelity = validate_publication_fidelity_identity(
        specialist.get("checkpoint_fidelity")
    )
    expected_fidelity_files = {
        CHECKPOINT_FIDELITY_FILENAME: checkpoint_fidelity["receipt_file_sha256"],
        "sft_reload_equivalence.json": reload_equivalence["receipt_file_sha256"],
        "adapter_config.json": checkpoint_fidelity["adapter_config_sha256"],
        "adapter_model.safetensors": checkpoint_fidelity["adapter_weights_sha256"],
    }
    if any(
        not isinstance(initial_by_path.get(path), dict)
        or initial_by_path[path].get("sha256") != digest
        for path, digest in expected_fidelity_files.items()
    ):
        raise ValueError(f"{label} specialist fidelity differs from initial-model fingerprint")


def _validate_experiment_provenance(
    provenance: object,
    *,
    experiment_fingerprint: object,
    protocol_fingerprint: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} has no experiment_provenance")
    _assert_finite_json(provenance, label=f"{label}.experiment_provenance")
    expected_experiment = hashlib.sha256(_canonical_json_bytes(provenance)).hexdigest()
    if experiment_fingerprint != expected_experiment:
        raise ValueError(f"{label} experiment fingerprint does not match provenance")
    protocol_material = _protocol_fingerprint_material(provenance)
    expected_protocol = hashlib.sha256(_canonical_json_bytes(protocol_material)).hexdigest()
    if protocol_fingerprint != expected_protocol:
        raise ValueError(f"{label} protocol fingerprint does not match provenance")

    _require_manifest_digest(
        provenance.get("data"),
        items_field="files",
        digest_field="manifest_sha256",
        label=f"{label} data identity",
    )
    _require_manifest_digest(
        provenance.get("code"),
        items_field="files",
        digest_field="manifest_sha256",
        label=f"{label} code identity",
    )
    verifier = provenance.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("name") != STRICT_VERIFIER_NAME:
        raise ValueError(f"{label} does not use the strict shared verifier")
    _require_manifest_digest(
        verifier,
        items_field="implementations",
        digest_field="implementation_manifest_sha256",
        label=f"{label} verifier identity",
    )
    if sorted(str(task) for task in (verifier.get("tasks") or [])) != sorted(PUBLICATION_TASKS):
        raise ValueError(f"{label} verifier task registry is incomplete")
    current_verifier = _current_strict_verifier_identity()
    if verifier != current_verifier:
        raise ValueError(
            f"{label} strict verifier identity does not match the code executing replay"
        )

    model = provenance.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{label} model identity is missing")
    for field in ("checkpoint_artifact", "base_model_artifact", "tokenizer_artifact"):
        artifact = model.get(field)
        if isinstance(artifact, dict) and artifact.get("kind") == "local_directory":
            _require_manifest_digest(
                artifact,
                items_field="files",
                digest_field="manifest_sha256",
                label=f"{label} model.{field}",
            )
    _publication_checkpoint_fidelity(provenance, label=label)

    contract = provenance.get("training_run_contract")
    contract_sha256 = provenance.get("training_run_contract_sha256")
    if (contract is None) != (contract_sha256 is None):
        raise ValueError(f"{label} training-contract provenance is only partially present")
    if isinstance(contract, dict):
        if _training_contract_file_sha256(contract) != contract_sha256:
            raise ValueError(f"{label} training contract bytes do not match provenance")
        _publication_exact_peft_kl_runtime(provenance, contract, label=label)
        _publication_exact_peft_kl_post_step(provenance, contract, label=label)
    return provenance


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    path: str
    task: str | None = None


@dataclass(frozen=True)
class ArtifactEvidence:
    artifact_type: str
    details_path: str
    companion_path: str
    experiment_provenance: dict[str, object]
    training_run_contract: dict[str, object] | None
    training_run_contract_sha256: str | None
    checkpoint_selection_binding: dict[str, object] | None
    checkpoint_selection_receipt: dict[str, object] | None
    checkpoint_fidelity: dict[str, object]
    exact_peft_kl_runtime: dict[str, object] | None = None
    exact_peft_kl_post_step: dict[str, object] | None = None
    details_sha256: str = ""
    details_size_bytes: int = 0
    details_record_count: int = 0
    run_config_sha256: str | None = None
    summary_sha256: str | None = None
    campaign_manifest: dict[str, object] | None = None
    campaign_run_binding: dict[str, object] | None = None
    confirmation_attempt_claim: dict[str, object] | None = None


@dataclass(frozen=True)
class FileSnapshot:
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _read_stable_file_bytes(path: str) -> tuple[bytes, FileSnapshot]:
    """Read and hash one inode snapshot, rejecting in-place or pathname replacement."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = os.stat(path)
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(after) != _stat_identity(
        path_after
    ):
        raise ValueError(f"artifact changed while its immutable snapshot was read: {path}")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"artifact size changed while it was read: {path}")
    return data, FileSnapshot(
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
    )


def _assert_path_matches_snapshot(path: str, snapshot: FileSnapshot) -> None:
    observed = os.stat(path)
    expected_identity = (
        snapshot.device,
        snapshot.inode,
        snapshot.size_bytes,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if _stat_identity(observed) != expected_identity:
        raise ValueError(f"artifact pathname changed after snapshot validation: {path}")


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record_key(record: Record) -> RecordKey:
    try:
        return (
            str(record["task"]),
            str(record.get("split", "eval")),
            int(record["example_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"detail record is missing a valid task/split/example_index: {record}"
        ) from exc


def _detail_paths(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    paths = sorted(glob.glob(os.path.join(path, "**", "run_details__*.jsonl"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no run_details__*.jsonl under {path}")
    return paths


def _record_protocol_fingerprint(record: Record) -> str:
    return str(
        record.get("protocol_fingerprint") or record.get("source_protocol_fingerprint") or ""
    )


def _record_artifact_fingerprint(record: Record) -> tuple[str, str]:
    for key in ("repair_fingerprint", "rescore_fingerprint", "experiment_fingerprint"):
        value = str(record.get(key) or "")
        if value:
            return key, value
    return "", ""


def _record_artifact_type(record: Record, *, label: str) -> str:
    present = [
        artifact_type
        for artifact_type, field in (
            ("native", "experiment_fingerprint"),
            ("repair", "repair_fingerprint"),
            ("rescore", "rescore_fingerprint"),
        )
        if record.get(field)
    ]
    if len(present) != 1:
        raise ValueError(
            f"{label} must identify exactly one native/repair/rescore artifact type; "
            f"observed={present}"
        )
    return present[0]


def _candidate_verification_context(
    record: Record,
    candidate: Mapping[str, object],
    index: int,
    *,
    label: str,
) -> tuple[str, str, str, object]:
    context = candidate.get("verification_context")
    context_hash = candidate.get("verification_context_sha256")
    if context is None:
        raw_record = record.get("raw_record")
        raw_candidates = raw_record.get("candidates") if isinstance(raw_record, dict) else None
        raw_candidate = (
            raw_candidates[index]
            if isinstance(raw_candidates, list) and index < len(raw_candidates)
            else None
        )
        if isinstance(raw_candidate, dict):
            context = raw_candidate.get("verification_context")
            context_hash = raw_candidate.get("verification_context_sha256")
    if context is not None:
        if not isinstance(context, dict):
            raise ValueError(f"{label} verification_context is not an object")
        observed_hash = hashlib.sha256(_canonical_json_bytes(context)).hexdigest()
        if context_hash != observed_hash:
            raise ValueError(f"{label} verification_context hash mismatch")
        required = {"instruction", "input_problem", "ground_truth_output", "instance_json"}
        if not required.issubset(context):
            raise ValueError(f"{label} verification_context is incomplete")
        return (
            str(context["instruction"]),
            str(context["input_problem"]),
            str(context["ground_truth_output"]),
            structured_instance({"instance_json": context["instance_json"]}),
        )
    required_record_fields = (
        "instruction",
        "input_problem",
        "ground_truth_output",
        "instance_json",
    )
    if any(field not in record for field in required_record_fields):
        raise ValueError(f"{label} has no complete verifier context")
    return (
        str(record["instruction"]),
        str(record["input_problem"]),
        str(record["ground_truth_output"]),
        structured_instance({"instance_json": record["instance_json"]}),
    )


def _validate_replayed_candidate_metrics(record: Record, *, label: str) -> None:
    task = str(record.get("task") or "")
    candidates = record.get("candidates")
    if task not in TASK_EVALUATORS or not isinstance(candidates, list):
        raise ValueError(f"{label} cannot replay strict candidate verification")
    for index, candidate in enumerate(candidates):
        candidate_label = f"{label}.candidates[{index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{candidate_label} is not an object")
        if "completion" not in candidate or not isinstance(candidate.get("completion"), str):
            raise ValueError(f"{candidate_label} has no exact completion for verifier replay")
        instruction, input_text, answer, instance = _candidate_verification_context(
            record,
            candidate,
            index,
            label=candidate_label,
        )
        completion = str(candidate["completion"])
        replayed = TASK_EVALUATORS[task](
            instruction,
            input_text,
            answer,
            instance,
            completion,
        )
        replayed.update(
            _constraint_diagnostics(
                task,
                prompt=instruction,
                input_text=input_text,
                answer=answer,
                instance=instance,
                completion=completion,
            )
        )
        for field in (
            "parsed_solution",
            "declared_objective",
            "objective_value",
            "ground_truth_objective",
            "is_feasible",
            "constraint_fraction",
            "gap_to_ground_truth",
            "parse_failure",
        ):
            expected = replayed.get(field)
            observed = candidate.get(field)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                matches = _numbers_equal(observed, expected)
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(
                    f"{candidate_label}.{field} disagrees with strict verifier replay: "
                    f"expected={expected!r}, observed={observed!r}"
                )
        expected_violations = replayed.get("constraint_violations")
        observed_violations = candidate.get("constraint_violations")
        if (
            not isinstance(expected_violations, dict)
            or not isinstance(observed_violations, dict)
            or set(expected_violations) != set(observed_violations)
        ):
            raise ValueError(
                f"{candidate_label}.constraint_violations disagrees with strict verifier replay"
            )
        for name, expected in expected_violations.items():
            if not _numbers_equal(observed_violations.get(name), expected):
                raise ValueError(
                    f"{candidate_label}.constraint_violations[{name!r}] disagrees with "
                    "strict verifier replay"
                )


def _validate_headline_metrics(
    record: Record,
    *,
    label: str,
    replay_verifier: bool = True,
) -> None:
    candidates = record.get("candidates")
    task = str(record.get("task") or "")
    if not isinstance(candidates, list) or not candidates or task not in TASK_DIRECTIONS:
        raise ValueError(f"{label} cannot recompute headline candidate metrics")
    if replay_verifier:
        _validate_replayed_candidate_metrics(record, label=label)
    ground_truth_objective = _finite_float(record.get("ground_truth_objective"))
    if ground_truth_objective is None or ground_truth_objective <= 0.0:
        raise ValueError(f"{label} has no finite positive ground_truth_objective")
    feasible: list[tuple[int, dict[str, object], float, float]] = []
    completion_tokens = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"{label}.candidates[{index}] is not an object")
        if candidate.get("candidate_index") != index:
            raise ValueError(f"{label}.candidates[{index}] has a mismatched candidate_index")
        is_feasible = candidate.get("is_feasible")
        if not isinstance(is_feasible, bool):
            raise ValueError(f"{label}.candidates[{index}].is_feasible is not boolean")
        parse_failure = candidate.get("parse_failure")
        if not isinstance(parse_failure, bool):
            raise ValueError(f"{label}.candidates[{index}].parse_failure is not boolean")
        violations = candidate.get("constraint_violations")
        if not isinstance(violations, dict) or "parse_failure" not in violations:
            raise ValueError(f"{label}.candidates[{index}] has incomplete constraint_violations")
        for name, raw_violation in violations.items():
            violation = _finite_float(raw_violation)
            if (
                not isinstance(name, str)
                or not name
                or violation is None
                or violation < 0.0
                or violation > 1.0
            ):
                raise ValueError(f"{label}.candidates[{index}].constraint_violations is invalid")
        if parse_failure != (float(violations["parse_failure"]) > 0.0):
            raise ValueError(
                f"{label}.candidates[{index}].parse_failure disagrees with diagnostics"
            )
        if not parse_failure:
            required_constraints = set(TASK_CORE_CONSTRAINTS[task])
            if task == "cvrp" and re.search(
                r"\b\d+\s*vehicle",
                f"{record.get('instruction', '')}\n{record.get('input_problem', '')}",
                flags=re.IGNORECASE,
            ):
                required_constraints.add("route_count_mismatch")
            missing_constraints = sorted(required_constraints - set(violations))
            if missing_constraints:
                raise ValueError(
                    f"{label}.candidates[{index}] has incomplete constraint_violations: "
                    f"missing={missing_constraints}"
                )
        constraint_fraction = _finite_float(candidate.get("constraint_fraction"))
        if (
            constraint_fraction is None
            or constraint_fraction < 0.0
            or constraint_fraction > 1.0
            or (is_feasible and not _numbers_equal(constraint_fraction, 1.0))
        ):
            raise ValueError(f"{label}.candidates[{index}] has invalid constraint_fraction")
        candidate_reference = _finite_float(candidate.get("ground_truth_objective"))
        if candidate_reference is None or not _numbers_equal(
            candidate_reference, ground_truth_objective
        ):
            raise ValueError(
                f"{label}.candidates[{index}] ground-truth objective disagrees with record"
            )
        token_count = candidate.get("completion_tokens")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"{label}.candidates[{index}] has invalid completion_tokens")
        completion_tokens += token_count
        if not is_feasible:
            if candidate.get("gap_to_ground_truth") is not None:
                raise ValueError(
                    f"{label}.candidates[{index}] is infeasible but carries an objective gap"
                )
            continue
        gap = _finite_float(candidate.get("gap_to_ground_truth"))
        objective = _finite_float(candidate.get("objective_value"))
        if gap is None or gap < 0.0 or objective is None:
            raise ValueError(
                f"{label}.candidates[{index}] is feasible without finite gap/objective"
            )
        if TASK_DIRECTIONS[task] == "min":
            recomputed_gap = max(
                0.0,
                (objective - ground_truth_objective) / ground_truth_objective,
            )
        else:
            recomputed_gap = max(
                0.0,
                (ground_truth_objective - objective) / ground_truth_objective,
            )
        if not _numbers_equal(gap, recomputed_gap):
            raise ValueError(
                f"{label}.candidates[{index}] gap disagrees with objective/reference/direction: "
                f"expected={recomputed_gap!r}, observed={gap!r}"
            )
        feasible.append((index, candidate, gap, objective))

    direction = TASK_DIRECTIONS[task]
    best = (
        min(
            feasible,
            key=lambda item: (
                item[2],
                item[3] if direction == "min" else -item[3],
                item[0],
            ),
        )
        if feasible
        else None
    )
    feasible_count = len(feasible)
    expected_exact = {
        "num_return_sequences": len(candidates),
        "generated_tokens": completion_tokens,
        "feasible_candidates": feasible_count,
        "best_candidate_index": best[0] if best is not None else None,
        "best_of_n_feasible": best is not None,
    }
    for field, expected in expected_exact.items():
        if record.get(field) != expected:
            raise ValueError(
                f"{label}.{field} disagrees with candidates: "
                f"expected={expected!r}, observed={record.get(field)!r}"
            )
    gaps = [item[2] for item in feasible]
    expected_numeric = {
        "candidate_feasibility_rate": feasible_count / len(candidates),
        "best_feasible_gap": best[2] if best is not None else None,
        "best_objective_value": best[3] if best is not None else None,
        "feasible_gap_mean_in_batch": _mean(gaps),
        "feasible_gap_std_in_batch": (
            statistics.stdev(gaps) if len(gaps) > 1 else (0.0 if gaps else None)
        ),
        "generation_time_sec_per_candidate": (
            float(record["generation_time_sec"]) / len(candidates)
            if _finite_float(record.get("generation_time_sec")) is not None
            else None
        ),
    }
    for field, expected in expected_numeric.items():
        if not _numbers_equal(record.get(field), expected):
            raise ValueError(
                f"{label}.{field} disagrees with candidates: "
                f"expected={expected!r}, observed={record.get(field)!r}"
            )


def _record_training_contract_sha256(record: Record) -> object:
    if record.get("training_run_contract_sha256") is not None:
        return record.get("training_run_contract_sha256")
    row_identity = record.get("row_identity")
    if isinstance(row_identity, dict):
        return row_identity.get("training_run_contract_sha256")
    return None


def _validate_record_self_hash(record: Record, *, artifact_type: str, label: str) -> None:
    field = {
        "native": "record_sha256",
        "repair": "repair_record_sha256",
        "rescore": "rescore_record_sha256",
    }[artifact_type]
    _validate_self_hash(record, field, label=label)
    if artifact_type == "repair":
        raw_record = record.get("raw_record")
        if not isinstance(raw_record, dict):
            raise ValueError(f"{label} repair record has no embedded raw_record")
        expected_raw_hash = hashlib.sha256(_canonical_json_bytes(raw_record)).hexdigest()
        if record.get("raw_record_sha256") != expected_raw_hash:
            raise ValueError(f"{label} raw_record_sha256 mismatch")
        _validate_self_hash(raw_record, "record_sha256", label=f"{label}.raw_record")


def _validate_repair_views(record: Record, *, label: str) -> None:
    """Close every repair metric view over the same strict candidate evidence."""

    raw_record = record.get("raw_record")
    raw_view = record.get("raw_strict_view")
    repaired_view = record.get("repaired_strict_view")
    wrapped_candidates = record.get("candidates")
    if not all(
        isinstance(value, dict) for value in (raw_record, raw_view, repaired_view)
    ) or not isinstance(wrapped_candidates, list):
        raise ValueError(f"{label} repair record has incomplete strict views")
    assert isinstance(raw_record, dict)
    assert isinstance(raw_view, dict)
    assert isinstance(repaired_view, dict)
    raw_candidates = raw_record.get("candidates")
    raw_view_candidates = raw_view.get("candidates")
    repaired_view_candidates = repaired_view.get("candidates")
    if not all(
        isinstance(value, list)
        for value in (raw_candidates, raw_view_candidates, repaired_view_candidates)
    ):
        raise ValueError(f"{label} repair strict views have no candidate arrays")
    assert isinstance(raw_candidates, list)
    assert isinstance(raw_view_candidates, list)
    assert isinstance(repaired_view_candidates, list)
    candidate_count = len(wrapped_candidates)
    if not (
        len(raw_candidates)
        == len(raw_view_candidates)
        == len(repaired_view_candidates)
        == candidate_count
    ):
        raise ValueError(f"{label} repair candidate views have different lengths")

    for view_name, view in (("raw_strict_view", raw_view), ("repaired_strict_view", repaired_view)):
        _validate_headline_metrics(
            view,
            label=f"{label}.{view_name}",
            replay_verifier=False,
        )
    _validate_headline_metrics(raw_record, label=f"{label}.raw_record")

    # The repair record is the repaired strict view plus provenance/repair
    # metadata. Every shared scalar/list field must therefore be identical.
    for field, expected in repaired_view.items():
        if field != "candidates" and record.get(field) != expected:
            raise ValueError(
                f"{label} top-level repair field {field!r} disagrees with repaired_strict_view"
            )
    for field, expected in raw_view.items():
        if field != "candidates" and raw_record.get(field) != expected:
            raise ValueError(f"{label} raw_strict_view field {field!r} disagrees with raw_record")

    for index, wrapped in enumerate(wrapped_candidates):
        if not isinstance(wrapped, dict):
            raise ValueError(f"{label}.candidates[{index}] is not an object")
        raw_metrics = wrapped.get("raw_metrics")
        repaired_metrics = wrapped.get("repaired_metrics")
        if not isinstance(raw_metrics, dict) or not isinstance(repaired_metrics, dict):
            raise ValueError(f"{label}.candidates[{index}] has incomplete repair metrics")
        if raw_view_candidates[index] != raw_metrics:
            raise ValueError(
                f"{label}.candidates[{index}].raw_metrics disagrees with raw_strict_view"
            )
        if repaired_view_candidates[index] != repaired_metrics:
            raise ValueError(
                f"{label}.candidates[{index}].repaired_metrics disagrees with repaired_strict_view"
            )
        raw_candidate = raw_candidates[index]
        if not isinstance(raw_candidate, dict) or any(
            raw_candidate.get(field) != expected for field, expected in raw_metrics.items()
        ):
            raise ValueError(f"{label}.candidates[{index}].raw_metrics disagrees with raw artifact")
        if any(wrapped.get(field) != expected for field, expected in repaired_metrics.items()):
            raise ValueError(
                f"{label}.candidates[{index}] top-level metrics disagree with repaired_metrics"
            )


def _validate_deterministic_repair_record(
    record: Record,
    *,
    config: Mapping[str, object],
    label: str,
) -> None:
    """Re-run the approved ground-truth-blind repair on every raw completion."""

    raw_record = record.get("raw_record")
    wrapped_candidates = record.get("candidates")
    raw_candidates = raw_record.get("candidates") if isinstance(raw_record, dict) else None
    if (
        not isinstance(raw_record, dict)
        or not isinstance(raw_candidates, list)
        or not isinstance(wrapped_candidates, list)
        or len(raw_candidates) != len(wrapped_candidates)
    ):
        raise ValueError(f"{label} cannot replay deterministic repair")
    task = str(raw_record.get("task") or "")
    for index, (raw_candidate, wrapped_candidate) in enumerate(
        zip(raw_candidates, wrapped_candidates, strict=True)
    ):
        candidate_label = f"{label}.candidates[{index}]"
        if not isinstance(raw_candidate, dict) or not isinstance(wrapped_candidate, dict):
            raise ValueError(f"{candidate_label} cannot replay deterministic repair")
        raw_completion = raw_candidate.get("completion")
        if (
            not isinstance(raw_completion, str)
            or wrapped_candidate.get("raw_completion") != raw_completion
        ):
            raise ValueError(f"{candidate_label} raw completion changed before repair")
        _instruction, _input_text, _answer, instance = _candidate_verification_context(
            raw_record,
            raw_candidate,
            index,
            label=candidate_label,
        )
        expected_completion, expected_actions = repair_completion(
            task,
            instance=instance,
            completion=raw_completion,
            config=dict(config),
        )
        if (
            wrapped_candidate.get("repaired_completion") != expected_completion
            or wrapped_candidate.get("completion") != expected_completion
            or wrapped_candidate.get("repair_actions") != expected_actions
        ):
            raise ValueError(f"{candidate_label} disagrees with the approved deterministic repair")
        raw_metrics = wrapped_candidate.get("raw_metrics")
        repaired_metrics = wrapped_candidate.get("repaired_metrics")
        if not isinstance(raw_metrics, dict) or not isinstance(repaired_metrics, dict):
            raise ValueError(f"{candidate_label} repair metrics are missing")
        if not bool(raw_metrics.get("is_feasible")):
            expected_status = "repaired"
        elif raw_metrics.get("parsed_solution") != repaired_metrics.get("parsed_solution"):
            expected_status = "improved"
        else:
            expected_status = "validated_or_reformatted"
        if wrapped_candidate.get("repair_status") != expected_status:
            raise ValueError(f"{candidate_label} repair status is inconsistent")


def _validate_details_artifact(
    identity: object,
    *,
    details_path: str,
    record_count: int,
    label: str,
    snapshot: FileSnapshot,
) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} has no details_artifact identity")
    expected = {
        "basename": os.path.basename(details_path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "record_count": int(record_count),
    }
    observed = {key: identity.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"{label} details_artifact mismatch: expected={expected}, observed={observed}"
        )


def _numbers_equal(first: object, second: object) -> bool:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is None or second_number is None:
        return first is None and second is None
    return math.isclose(first_number, second_number, rel_tol=1e-12, abs_tol=1e-12)


def _validate_aggregate_projection(
    summary: object,
    records: Sequence[Record],
    *,
    label: str,
) -> None:
    """Verify all deterministic aggregate fields shared by emitted schemas."""

    if not isinstance(summary, dict):
        raise ValueError(f"{label} aggregate is missing")
    candidate_count = sum(_candidate_counts(record)[1] for record in records)
    feasible_candidates = sum(_candidate_counts(record)[0] for record in records)
    feasible_examples = sum(_is_feasible(record) for record in records)
    gaps = [gap for record in records if (gap := _gap(record)) is not None]
    generated_tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in records)
    generation_times = [
        float(_finite_float(record.get("generation_time_sec")) or 0.0) for record in records
    ]
    generation_time = sum(generation_times)
    generation_times_per_candidate = [
        value
        for record in records
        if (value := _finite_float(record.get("generation_time_sec_per_candidate"))) is not None
    ]
    feasible_gap_stds = [
        value
        for record in records
        if (value := _finite_float(record.get("feasible_gap_std_in_batch"))) is not None
    ]
    peak_allocated = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_allocated_bytes")) is not None
    ]
    peak_reserved = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_reserved_bytes")) is not None
    ]
    expected_exact = {
        "examples": len(records),
        "candidates": candidate_count,
        "feasible_candidates": feasible_candidates,
        "best_of_n_feasible_examples": feasible_examples,
        "generated_tokens": generated_tokens,
    }
    for field, expected in expected_exact.items():
        if field in summary and summary.get(field) != expected:
            raise ValueError(
                f"{label}.{field} disagrees with details: "
                f"expected={expected}, observed={summary.get(field)!r}"
            )
    expected_numeric = {
        "candidate_feasibility_rate": (
            feasible_candidates / candidate_count if candidate_count else 0.0
        ),
        "best_of_n_instance_feasibility_rate": (
            feasible_examples / len(records) if records else 0.0
        ),
        "mean_best_feasible_gap": _mean(gaps),
        "median_best_feasible_gap": statistics.median(gaps) if gaps else None,
        "std_best_feasible_gap": (
            statistics.stdev(gaps) if len(gaps) > 1 else (0.0 if gaps else None)
        ),
        "mean_feasible_gap_std_in_batch": _mean(feasible_gap_stds),
        "std_feasible_gap_std_in_batch": (
            statistics.stdev(feasible_gap_stds)
            if len(feasible_gap_stds) > 1
            else (0.0 if feasible_gap_stds else None)
        ),
        "total_generation_time_sec": generation_time,
        "mean_generation_time_sec": _mean(generation_times),
        "std_generation_time_sec": (
            statistics.stdev(generation_times)
            if len(generation_times) > 1
            else (0.0 if generation_times else None)
        ),
        "mean_generation_time_sec_per_candidate": _mean(generation_times_per_candidate),
        "std_generation_time_sec_per_candidate": (
            statistics.stdev(generation_times_per_candidate)
            if len(generation_times_per_candidate) > 1
            else (0.0 if generation_times_per_candidate else None)
        ),
        "mean_generated_tokens_per_instance": (generated_tokens / len(records) if records else 0.0),
        "peak_gpu_memory_allocated_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_gpu_memory_reserved_bytes": max(peak_reserved) if peak_reserved else None,
    }
    for field, expected in expected_numeric.items():
        if field in summary and not _numbers_equal(summary.get(field), expected):
            raise ValueError(
                f"{label}.{field} disagrees with details: "
                f"expected={expected}, observed={summary.get(field)!r}"
            )


def _validate_summary_integrity(
    summary: Mapping[str, object],
    *,
    artifact_type: str,
    records: Sequence[Record],
    label: str,
) -> None:
    if int(summary.get("record_count", len(records)) or -1) != len(records):
        raise ValueError(f"{label} record_count disagrees with details")
    if artifact_type in {"native", "rescore"}:
        _validate_aggregate_projection(summary.get("overall"), records, label=f"{label}.overall")
    elif artifact_type == "repair":
        repaired_views = [
            record.get("repaired_strict_view")
            if isinstance(record.get("repaired_strict_view"), dict)
            else record
            for record in records
        ]
        raw_views = [record.get("raw_strict_view") for record in records]
        if not all(isinstance(record, dict) for record in raw_views):
            raise ValueError(f"{label} repair details have no raw_strict_view")
        _validate_aggregate_projection(
            summary.get("repaired_strict"),
            repaired_views,  # type: ignore[arg-type]
            label=f"{label}.repaired_strict",
        )
        _validate_aggregate_projection(
            summary.get("raw_strict"),
            raw_views,  # type: ignore[arg-type]
            label=f"{label}.raw_strict",
        )
        expected_repair_time = sum(
            float(_finite_float(record.get("total_repair_time_sec")) or 0.0) for record in records
        )
        if not _numbers_equal(summary.get("total_repair_time_sec"), expected_repair_time):
            raise ValueError(f"{label} total_repair_time_sec disagrees with details")
        if summary.get("candidate_count") != sum(
            len(record.get("candidates", []))
            for record in records
            if isinstance(record.get("candidates"), list)
        ):
            raise ValueError(f"{label} candidate_count disagrees with details")

    per_task = summary.get("per_task")
    tasks = sorted({_record_key(record)[0] for record in records})
    if not isinstance(per_task, dict) or sorted(per_task) != tasks:
        raise ValueError(f"{label}.per_task key set disagrees with details")
    for task in tasks:
        task_records = [record for record in records if _record_key(record)[0] == task]
        task_summary = per_task[task]
        if artifact_type in {"native", "rescore"}:
            _validate_aggregate_projection(
                task_summary,
                task_records,
                label=f"{label}.per_task.{task}",
            )
        else:
            if not isinstance(task_summary, dict):
                raise ValueError(f"{label}.per_task.{task} is missing")
            raw_views = [record.get("raw_strict_view") for record in task_records]
            repaired_views = [
                record.get("repaired_strict_view")
                if isinstance(record.get("repaired_strict_view"), dict)
                else record
                for record in task_records
            ]
            _validate_aggregate_projection(
                task_summary.get("raw_strict"),
                raw_views,  # type: ignore[arg-type]
                label=f"{label}.per_task.{task}.raw_strict",
            )
            _validate_aggregate_projection(
                task_summary.get("repaired_strict"),
                repaired_views,  # type: ignore[arg-type]
                label=f"{label}.per_task.{task}.repaired_strict",
            )


def _validate_record_provenance(
    record: Record,
    *,
    path: str,
    line_number: int,
    publication_mode: bool = False,
) -> None:
    location = f"{path}:{line_number}"
    artifact_type = _record_artifact_type(record, label=location)
    protocol = _record_protocol_fingerprint(record)
    artifact_key, artifact = _record_artifact_fingerprint(record)
    if not _is_sha256(protocol):
        raise ValueError(f"missing or invalid protocol fingerprint at {location}")
    if not artifact_key or not _is_sha256(artifact):
        raise ValueError(f"missing or invalid experiment/rescore/repair fingerprint at {location}")

    verifier = record.get("strict_verifier")
    if not isinstance(verifier, dict):
        raise ValueError(f"missing strict verifier identity at {location}")
    if verifier.get("name") != STRICT_VERIFIER_NAME:
        raise ValueError(f"unexpected strict verifier at {location}")
    if not _is_sha256(verifier.get("implementation_manifest_sha256")):
        raise ValueError(f"invalid strict verifier implementation hash at {location}")

    key = _record_key(record)
    locator = record.get("source_row_locator")
    if not isinstance(locator, dict):
        raise ValueError(f"missing source_row_locator at {location}")
    expected_locator = {
        "task": key[0],
        "split": key[1],
        "merged_dataset_index": key[2],
    }
    observed_locator = {
        "task": str(locator.get("task", "")),
        "split": str(locator.get("split", "")),
        "merged_dataset_index": int(locator.get("merged_dataset_index", -1)),
    }
    if observed_locator != expected_locator:
        raise ValueError(f"source_row_locator disagrees with detail key at {location}")
    if not _is_sha256(locator.get("data_manifest_sha256")):
        raise ValueError(f"invalid data manifest hash at {location}")

    instance_sha256 = record.get("instance_sha256")
    if not _is_sha256(instance_sha256):
        raise ValueError(f"missing or invalid instance_sha256 at {location}")
    if "instance_json" not in record or record.get("instance_json") is None:
        raise ValueError(f"confirmatory record has no embedded instance_json at {location}")
    observed_instance_sha256 = hashlib.sha256(
        _canonical_json_bytes(record.get("instance_json"))
    ).hexdigest()
    if observed_instance_sha256 != instance_sha256:
        raise ValueError(f"instance_json hash mismatch at {location}")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"confirmatory record has no candidate details at {location}")
    observed_k = record.get("num_return_sequences")
    if (
        isinstance(observed_k, bool)
        or not isinstance(observed_k, int)
        or observed_k != len(candidates)
    ):
        raise ValueError(f"record candidate count is internally inconsistent at {location}")
    completion_tokens: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"non-object candidate {candidate_index} at {location}")
        value = candidate.get("completion_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"missing completion-token denominator for candidate {candidate_index} "
                f"at {location}"
            )
        completion_tokens.append(value)
    if int(record.get("generated_tokens", -1)) != sum(completion_tokens):
        raise ValueError(f"record generated-token total is inconsistent at {location}")
    generation_time = _finite_float(record.get("generation_time_sec"))
    if generation_time is None or generation_time < 0.0:
        raise ValueError(f"record has no valid generation-time denominator at {location}")
    if publication_mode:
        _assert_finite_json(record, label=location)
        _validate_record_self_hash(record, artifact_type=artifact_type, label=location)
        _validate_headline_metrics(record, label=location)
        if artifact_type == "repair":
            _validate_repair_views(record, label=location)
        if generation_time <= 0.0 or sum(completion_tokens) <= 0:
            raise ValueError(f"publication record has zero generation compute at {location}")
        feasible = record.get("best_of_n_feasible")
        gap = _finite_float(record.get("best_feasible_gap"))
        if not isinstance(feasible, bool):
            raise ValueError(f"publication record has no boolean feasibility at {location}")
        if feasible and (gap is None or gap < 0.0):
            raise ValueError(
                f"feasible publication record has no finite nonnegative gap at {location}"
            )
        if not feasible and record.get("best_feasible_gap") is not None:
            raise ValueError(
                f"infeasible publication record carries a conditional gap at {location}"
            )
        if artifact_type == "repair":
            repair_time = _finite_float(record.get("total_repair_time_sec"))
            if repair_time is None or repair_time < 0.0:
                raise ValueError(f"repair record has no finite repair runtime at {location}")


def _load_json_object(path: str) -> dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read provenance companion {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"provenance companion is not a JSON object: {path}")
    return value


def _validate_companion(path: str, file_records: Sequence[Record]) -> None:
    basename = os.path.basename(path)
    directory = os.path.dirname(path)
    fingerprints = {_record_artifact_fingerprint(record) for record in file_records}
    protocols = {_record_protocol_fingerprint(record) for record in file_records}

    if basename.startswith("run_details__") and basename.endswith(".jsonl"):
        label = basename[len("run_details__") : -len(".jsonl")]
        companion_path = os.path.join(directory, f"run_config__{label}.json")
        companion = _load_json_object(companion_path)
        if companion.get("run_status") != "complete":
            raise ValueError(f"evaluation run is not complete: {companion_path}")
        expected = {("experiment_fingerprint", str(companion.get("experiment_fingerprint") or ""))}
        if fingerprints != expected:
            raise ValueError(f"detail/config experiment fingerprint mismatch: {path}")
        if protocols != {str(companion.get("protocol_fingerprint") or "")}:
            raise ValueError(f"detail/config protocol fingerprint mismatch: {path}")
        verifier = companion.get("verifier_identity")
        record_verifiers = {
            json.dumps(record["strict_verifier"], sort_keys=True) for record in file_records
        }
        if not isinstance(verifier, dict) or record_verifiers != {
            json.dumps(verifier, sort_keys=True)
        }:
            raise ValueError(f"detail/config verifier identity mismatch: {path}")
        return

    summary_path = f"{os.path.splitext(path)[0]}__summary.json"
    companion = _load_json_object(summary_path)
    if int(companion.get("record_count", -1)) != len(file_records):
        raise ValueError(f"detail/summary record-count mismatch: {path}")
    companion_fingerprints = {
        (key, str(companion.get(key) or ""))
        for key in ("repair_fingerprint", "rescore_fingerprint", "experiment_fingerprint")
        if companion.get(key)
    }
    if fingerprints != companion_fingerprints:
        raise ValueError(f"detail/summary artifact fingerprint mismatch: {path}")


def _summary_path_for_details(path: str) -> str:
    return f"{os.path.splitext(path)[0]}__summary.json"


def _validate_native_publication_companions(
    path: str,
    records: Sequence[Record],
    snapshot: FileSnapshot,
) -> ArtifactEvidence:
    basename = os.path.basename(path)
    if not (basename.startswith("run_details__") and basename.endswith(".jsonl")):
        raise ValueError(f"native details must use run_details__LABEL.jsonl: {path}")
    label = basename[len("run_details__") : -len(".jsonl")]
    directory = os.path.dirname(path)
    config_path = os.path.join(directory, f"run_config__{label}.json")
    summary_path = os.path.join(directory, f"run_summary__{label}.json")
    config = _load_json_object(config_path)
    summary = _load_json_object(summary_path)
    _assert_finite_json(config, label=config_path)
    _assert_finite_json(summary, label=summary_path)
    _validate_self_hash(config, "run_config_sha256", label=config_path)
    _validate_self_hash(summary, "summary_sha256", label=summary_path)
    if config.get("run_status") != "complete" or summary.get("run_status") != "complete":
        raise ValueError(f"native evaluation is not complete: {path}")
    for field in (
        "experiment_fingerprint",
        "protocol_fingerprint",
        "experiment_provenance",
        "training_run_contract",
        "training_run_contract_sha256",
        "verifier_identity",
        "campaign_manifest",
        "campaign_run_binding",
        "confirmation_attempt_claim",
    ):
        if config.get(field) != summary.get(field):
            raise ValueError(f"native config/summary {field} mismatch: {path}")
    provenance = _validate_experiment_provenance(
        config.get("experiment_provenance"),
        experiment_fingerprint=config.get("experiment_fingerprint"),
        protocol_fingerprint=config.get("protocol_fingerprint"),
        label=config_path,
    )
    if config.get("code_identity") != provenance.get("code"):
        raise ValueError(f"native config code identity mismatch: {path}")
    if config.get("verifier_identity") != provenance.get("verifier"):
        raise ValueError(f"native config verifier identity mismatch: {path}")
    contract = config.get("training_run_contract")
    contract_sha256 = config.get("training_run_contract_sha256")
    identity = config.get("training_run_contract_identity")
    if not isinstance(contract, dict) or not isinstance(identity, dict):
        raise ValueError(f"native publication artifact lacks training contract identity: {path}")
    if (
        identity.get("sha256") != contract_sha256
        or identity.get("size_bytes") != len(_training_contract_file_bytes(contract))
        or identity.get("payload_sha256")
        != hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()
        or _training_contract_file_sha256(contract) != contract_sha256
    ):
        raise ValueError(f"native training contract identity is inconsistent: {path}")
    contract_seeds = contract.get("seeds")
    if not isinstance(contract_seeds, dict):
        raise ValueError(f"native training contract has no seeds: {path}")
    _validate_training_contract(
        contract,
        contract_sha256,
        method=str(contract.get("method") or ""),
        seed=int(contract_seeds.get("train", -1)),
        task=str(contract.get("task") or ""),
        label=path,
    )
    _validate_details_artifact(
        config.get("details_artifact"),
        details_path=path,
        record_count=len(records),
        label=config_path,
        snapshot=snapshot,
    )
    _validate_details_artifact(
        summary.get("details_artifact"),
        details_path=path,
        record_count=len(records),
        label=summary_path,
        snapshot=snapshot,
    )
    _validate_summary_integrity(
        summary,
        artifact_type="native",
        records=records,
        label=summary_path,
    )
    for index, record in enumerate(records):
        record_label = f"{path}:record[{index}]"
        if record.get("experiment_fingerprint") != config.get("experiment_fingerprint"):
            raise ValueError(f"native record experiment fingerprint mismatch at {record_label}")
        if record.get("protocol_fingerprint") != config.get("protocol_fingerprint"):
            raise ValueError(f"native record protocol fingerprint mismatch at {record_label}")
        if record.get("strict_verifier") != provenance.get("verifier"):
            raise ValueError(f"native record verifier mismatch at {record_label}")
        if record.get("training_run_contract_sha256") != contract_sha256:
            raise ValueError(f"native record training contract mismatch at {record_label}")

    selection = provenance.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    if config.get("campaign_manifest") != selection.get("campaign_manifest"):
        raise ValueError(f"native campaign manifest provenance mismatch: {path}")
    if config.get("campaign_run_binding") != selection.get("campaign_run_binding"):
        raise ValueError(f"native campaign run provenance mismatch: {path}")
    exact_runtime = _publication_exact_peft_kl_runtime(
        provenance,
        contract,
        label=config_path,
        live_config=config,
    )
    exact_post_step = _publication_exact_peft_kl_post_step(
        provenance,
        contract,
        label=config_path,
        live_config=config,
    )
    return ArtifactEvidence(
        artifact_type="native",
        details_path=path,
        companion_path=config_path,
        experiment_provenance=provenance,
        training_run_contract=contract,
        training_run_contract_sha256=str(contract_sha256),
        checkpoint_selection_binding=(
            selection.get("checkpoint_selection_binding")
            if isinstance(selection.get("checkpoint_selection_binding"), dict)
            else None
        ),
        checkpoint_selection_receipt=(
            selection.get("checkpoint_selection_receipt")
            if isinstance(selection.get("checkpoint_selection_receipt"), dict)
            else None
        ),
        checkpoint_fidelity=_publication_checkpoint_fidelity(provenance, label=config_path),
        exact_peft_kl_runtime=exact_runtime,
        exact_peft_kl_post_step=exact_post_step,
        details_sha256=snapshot.sha256,
        details_size_bytes=snapshot.size_bytes,
        details_record_count=len(records),
        run_config_sha256=str(config.get("run_config_sha256") or ""),
        summary_sha256=str(summary.get("summary_sha256") or ""),
        campaign_manifest=(
            config.get("campaign_manifest")
            if isinstance(config.get("campaign_manifest"), dict)
            else None
        ),
        campaign_run_binding=(
            config.get("campaign_run_binding")
            if isinstance(config.get("campaign_run_binding"), dict)
            else None
        ),
        confirmation_attempt_claim=(
            config.get("confirmation_attempt_claim")
            if isinstance(config.get("confirmation_attempt_claim"), dict)
            else None
        ),
    )


def _validate_derived_publication_companion(
    path: str,
    records: Sequence[Record],
    *,
    artifact_type: str,
    snapshot: FileSnapshot,
) -> ArtifactEvidence:
    summary_path = _summary_path_for_details(path)
    summary = _load_json_object(summary_path)
    _assert_finite_json(summary, label=summary_path)
    _validate_self_hash(summary, "summary_sha256", label=summary_path)
    _validate_details_artifact(
        summary.get("details_artifact"),
        details_path=path,
        record_count=len(records),
        label=summary_path,
        snapshot=snapshot,
    )
    expected_lane = {
        "repair": "deterministic_repair_and_improve_hybrid",
        "rescore": "strict_rescore_raw_generation",
    }[artifact_type]
    if summary.get("system_lane") != expected_lane:
        raise ValueError(f"{artifact_type} summary has an unexpected system_lane: {path}")
    fingerprint_field = f"{artifact_type}_fingerprint"
    material = summary.get("fingerprint_material")
    if not isinstance(material, dict):
        raise ValueError(f"{artifact_type} summary has no fingerprint_material: {path}")
    fingerprint = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    if summary.get(fingerprint_field) != fingerprint:
        raise ValueError(f"{artifact_type} fingerprint does not match its material: {path}")
    if artifact_type == "repair":
        expected_material_keys = {
            "schema_version",
            "system_name",
            "source_experiment_fingerprint",
            "source_protocol_fingerprint",
            "source_experiment_provenance",
            "training_run_contract",
            "training_run_contract_sha256",
            "source_campaign_manifest",
            "source_campaign_run_binding",
            "source_confirmation_attempt_claim",
            "source_config",
            "source_details",
            "repair_config",
            "repair_implementation",
            "repair_verifier",
        }
        current_implementation = _repair_implementation_identity()
        if (
            set(material) != expected_material_keys
            or summary.get("system_lane") != APPROVED_REPAIR_SYSTEM_LANE
            or summary.get("system_name") != APPROVED_REPAIR_SYSTEM_NAME
            or summary.get("repair_schema_version") != REPAIR_SCHEMA_VERSION
            or summary.get("repair_algorithm_version") != REPAIR_ALGORITHM_VERSION
            or material.get("schema_version") != REPAIR_SCHEMA_VERSION
            or material.get("system_name") != APPROVED_REPAIR_SYSTEM_NAME
            or material.get("repair_config") != APPROVED_REPAIR_CONFIG
            or material.get("repair_implementation") != current_implementation
        ):
            raise ValueError(
                "repair artifact does not use the frozen approved algorithm/config/implementation"
            )
    else:
        expected_material_keys = {
            "schema_version",
            "system_name",
            "sources",
            "source_config",
            "source_experiment_fingerprint",
            "source_protocol_fingerprint",
            "source_experiment_provenance",
            "training_run_contract",
            "training_run_contract_sha256",
            "source_campaign_manifest",
            "source_campaign_run_binding",
            "source_confirmation_attempt_claim",
            "verifier",
            "rescore_implementation",
        }
        if set(material) != expected_material_keys or material.get("schema_version") != 1:
            raise ValueError("rescore artifact does not use the exact native-source schema")
    provenance = _validate_experiment_provenance(
        summary.get("source_experiment_provenance"),
        experiment_fingerprint=summary.get("source_experiment_fingerprint"),
        protocol_fingerprint=summary.get("source_protocol_fingerprint"),
        label=summary_path,
    )
    if material.get("source_experiment_provenance") != provenance:
        raise ValueError(f"{artifact_type} fingerprint omits/changes source provenance: {path}")
    source_selection = provenance.get("selection")
    source_selection = source_selection if isinstance(source_selection, dict) else {}
    derived_campaign_fields = {
        "source_campaign_manifest": source_selection.get("campaign_manifest"),
        "source_campaign_run_binding": source_selection.get("campaign_run_binding"),
    }
    for field, expected in derived_campaign_fields.items():
        if material.get(field) != expected or summary.get(field) != expected:
            raise ValueError(
                f"{artifact_type} {field} differs from source experiment provenance: {path}"
            )
    if material.get("source_confirmation_attempt_claim") != summary.get(
        "source_confirmation_attempt_claim"
    ):
        raise ValueError(
            f"{artifact_type} source confirmation claim is not fingerprint-bound: {path}"
        )
    contract = summary.get("training_run_contract")
    contract_sha256 = summary.get("training_run_contract_sha256")
    if contract != provenance.get("training_run_contract") or contract_sha256 != provenance.get(
        "training_run_contract_sha256"
    ):
        raise ValueError(f"{artifact_type} training contract provenance mismatch: {path}")
    if (
        not isinstance(contract, dict)
        or _training_contract_file_sha256(contract) != contract_sha256
    ):
        raise ValueError(f"{artifact_type} has no valid embedded training contract: {path}")
    contract_seeds = contract.get("seeds")
    if not isinstance(contract_seeds, dict):
        raise ValueError(f"{artifact_type} training contract has no seeds: {path}")
    _validate_training_contract(
        contract,
        contract_sha256,
        method=str(contract.get("method") or ""),
        seed=int(contract_seeds.get("train", -1)),
        task=str(contract.get("task") or ""),
        label=path,
    )

    verifier_material_field = "repair_verifier" if artifact_type == "repair" else "verifier"
    verifier = material.get(verifier_material_field)
    if verifier != provenance.get("verifier"):
        raise ValueError(f"{artifact_type} verifier differs from the native verifier: {path}")
    _validate_summary_integrity(
        summary,
        artifact_type=artifact_type,
        records=records,
        label=summary_path,
    )
    for index, record in enumerate(records):
        record_label = f"{path}:record[{index}]"
        if record.get(fingerprint_field) != fingerprint:
            raise ValueError(f"{artifact_type} record fingerprint mismatch at {record_label}")
        if record.get("source_experiment_fingerprint") != summary.get(
            "source_experiment_fingerprint"
        ):
            raise ValueError(f"{artifact_type} record source experiment mismatch at {record_label}")
        source_schema = record.get("source_experiment_fingerprint_schema_version")
        if source_schema is None and isinstance(record.get("raw_record"), dict):
            source_schema = record["raw_record"].get("experiment_fingerprint_schema_version")
        if source_schema != provenance.get("schema_version"):
            raise ValueError(
                f"{artifact_type} record source experiment schema mismatch at {record_label}"
            )
        if _record_protocol_fingerprint(record) != summary.get("source_protocol_fingerprint"):
            raise ValueError(f"{artifact_type} record protocol mismatch at {record_label}")
        if _record_training_contract_sha256(record) != contract_sha256:
            raise ValueError(f"{artifact_type} record training contract mismatch at {record_label}")
        if record.get("strict_verifier") != provenance.get("verifier"):
            raise ValueError(f"{artifact_type} record verifier mismatch at {record_label}")
        if artifact_type == "repair":
            raw_record = record.get("raw_record")
            if not isinstance(raw_record, dict):
                raise ValueError(f"repair record lacks raw artifact at {record_label}")
            if raw_record.get("experiment_fingerprint") != summary.get(
                "source_experiment_fingerprint"
            ):
                raise ValueError(f"repair raw artifact provenance mismatch at {record_label}")
            if (
                record.get("repair_schema_version") != REPAIR_SCHEMA_VERSION
                or record.get("repair_algorithm_version") != REPAIR_ALGORITHM_VERSION
                or record.get("repair_verifier") != provenance.get("verifier")
            ):
                raise ValueError(f"repair record algorithm/verifier mismatch at {record_label}")
            _validate_deterministic_repair_record(
                record,
                config=APPROVED_REPAIR_CONFIG,
                label=record_label,
            )

    source_config_identity = material.get("source_config")
    try:
        portable_snapshot = validate_derived_source_config_snapshot(
            summary.get("source_config_snapshot"),
            expected_source_config_identity=source_config_identity,
        )
    except ValueError as exc:
        raise ValueError(
            f"{artifact_type} source-config snapshot is invalid: {path}: {exc}"
        ) from exc
    embedded_source_config = portable_snapshot["source_config_payload"]
    if (
        not isinstance(embedded_source_config, dict)
        or embedded_source_config.get("experiment_provenance") != provenance
        or embedded_source_config.get("training_run_contract") != contract
        or embedded_source_config.get("training_run_contract_sha256") != contract_sha256
    ):
        raise ValueError(
            f"{artifact_type} embedded source config differs from bound provenance: {path}"
        )

    # Any still-live source config is an additional operational check, never the
    # portable artifact's sole authority.  Derived results remain valid after an
    # exact relocation that omits the original evaluation directory.
    normalized_identity = portable_snapshot["source_config_identity"]
    source_files = embedded_source_config.get("files")
    declared_live_path = source_files.get("run_config") if isinstance(source_files, dict) else None
    live_candidates = {
        os.path.join(
            os.path.dirname(summary_path),
            str(normalized_identity["basename"]),
        )
    }
    if isinstance(declared_live_path, str) and os.path.isabs(declared_live_path):
        live_candidates.add(declared_live_path)
    for source_config_path in sorted(path for path in live_candidates if os.path.lexists(path)):
        if os.path.islink(source_config_path) or not os.path.isfile(source_config_path):
            raise ValueError(f"{artifact_type} live source config is not a regular file: {path}")
        source_config_bytes, live_snapshot = _read_stable_file_bytes(source_config_path)
        if (
            live_snapshot.sha256 != normalized_identity["sha256"]
            or live_snapshot.size_bytes != normalized_identity["size_bytes"]
        ):
            raise ValueError(
                f"{artifact_type} live source config differs from its identity: {path}"
            )
        try:
            live_source_config = json.loads(source_config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{artifact_type} live source config is invalid: {path}") from exc
        if live_source_config != embedded_source_config:
            raise ValueError(
                f"{artifact_type} live source config differs from its snapshot: {path}"
            )
    exact_runtime = _publication_exact_peft_kl_runtime(
        provenance,
        contract,
        label=summary_path,
    )
    receipt_snapshot = portable_snapshot["exact_peft_kl_runtime_receipt"]
    if exact_runtime is None:
        if receipt_snapshot is not None:
            raise ValueError(f"{artifact_type} beta=0 carries an exact KL receipt snapshot")
    elif (
        not isinstance(receipt_snapshot, dict)
        or receipt_snapshot.get("runtime_identity") != exact_runtime
    ):
        raise ValueError(
            f"{artifact_type} exact KL receipt snapshot differs from source provenance"
        )
    exact_post_step = _publication_exact_peft_kl_post_step(
        provenance,
        contract,
        label=summary_path,
    )
    post_step_snapshot = portable_snapshot["exact_peft_kl_post_step_receipts"]
    if exact_post_step is None:
        if post_step_snapshot is not None:
            raise ValueError(f"{artifact_type} beta=0 carries a post-step KL snapshot")
    elif (
        not isinstance(post_step_snapshot, dict)
        or post_step_snapshot.get("lineage") != exact_post_step
    ):
        raise ValueError(f"{artifact_type} post-step KL snapshot differs from source provenance")
    selection = source_selection
    return ArtifactEvidence(
        artifact_type=artifact_type,
        details_path=path,
        companion_path=summary_path,
        experiment_provenance=provenance,
        training_run_contract=contract,
        training_run_contract_sha256=str(contract_sha256),
        checkpoint_selection_binding=(
            selection.get("checkpoint_selection_binding")
            if isinstance(selection.get("checkpoint_selection_binding"), dict)
            else None
        ),
        checkpoint_selection_receipt=(
            selection.get("checkpoint_selection_receipt")
            if isinstance(selection.get("checkpoint_selection_receipt"), dict)
            else None
        ),
        checkpoint_fidelity=_publication_checkpoint_fidelity(provenance, label=summary_path),
        exact_peft_kl_runtime=exact_runtime,
        exact_peft_kl_post_step=exact_post_step,
        details_sha256=snapshot.sha256,
        details_size_bytes=snapshot.size_bytes,
        details_record_count=len(records),
        summary_sha256=str(summary.get("summary_sha256") or ""),
        campaign_manifest=(
            summary.get("source_campaign_manifest")
            if isinstance(summary.get("source_campaign_manifest"), dict)
            else None
        ),
        campaign_run_binding=(
            summary.get("source_campaign_run_binding")
            if isinstance(summary.get("source_campaign_run_binding"), dict)
            else None
        ),
        confirmation_attempt_claim=(
            summary.get("source_confirmation_attempt_claim")
            if isinstance(summary.get("source_confirmation_attempt_claim"), dict)
            else None
        ),
    )


def _validate_publication_companion(
    path: str,
    records: Sequence[Record],
    snapshot: FileSnapshot,
) -> ArtifactEvidence:
    artifact_types = {
        _record_artifact_type(record, label=f"{path}:record[{index}]")
        for index, record in enumerate(records)
    }
    if len(artifact_types) != 1:
        raise ValueError(f"details file mixes artifact types: {path}")
    artifact_type = artifact_types.pop()
    if artifact_type == "native":
        return _validate_native_publication_companions(path, records, snapshot)
    return _validate_derived_publication_companion(
        path,
        records,
        artifact_type=artifact_type,
        snapshot=snapshot,
    )


def load_run(
    spec: RunSpec,
    *,
    expected_k: int | None,
    allow_legacy_provenance: bool = False,
    publication_mode: bool = False,
    evidence_sink: list[ArtifactEvidence] | None = None,
) -> dict[RecordKey, Record]:
    records: dict[RecordKey, Record] = {}
    for path in _detail_paths(spec.path):
        file_records: list[Record] = []
        raw_bytes, snapshot = _read_stable_file_bytes(path)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"details artifact is not UTF-8: {path}: {exc}") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object detail record at {path}:{line_number}")
            if not allow_legacy_provenance:
                _validate_record_provenance(
                    record,
                    path=path,
                    line_number=line_number,
                    publication_mode=publication_mode,
                )
            key = _record_key(record)
            if key in records:
                raise ValueError(
                    f"duplicate detail key for method={spec.method} seed={spec.seed}: {key}; "
                    "provide a single canonical resume output"
                )
            if expected_k is not None:
                observed_k = record.get("num_return_sequences")
                candidates = record.get("candidates")
                if observed_k is None and isinstance(candidates, list):
                    observed_k = len(candidates)
                if int(observed_k or -1) != expected_k:
                    raise ValueError(
                        f"candidate-budget mismatch at {path}:{line_number}: "
                        f"expected K={expected_k}, observed {observed_k!r}"
                    )
            records[key] = record
            file_records.append(record)
        if file_records and not allow_legacy_provenance:
            if publication_mode:
                evidence = _validate_publication_companion(
                    path,
                    file_records,
                    snapshot,
                )
                _assert_path_matches_snapshot(path, snapshot)
                if evidence_sink is not None:
                    evidence_sink.append(evidence)
            else:
                _validate_companion(path, file_records)
    if not records:
        raise ValueError(f"run contains no detail records: {spec.path}")
    if not allow_legacy_provenance:
        protocol_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
        verifier_manifests: set[str] = set()
        for key, record in records.items():
            protocol_groups[key[:2]].add(_record_protocol_fingerprint(record))
            verifier = record["strict_verifier"]
            assert isinstance(verifier, dict)
            verifier_manifests.add(str(verifier["implementation_manifest_sha256"]))
        mixed = [group for group, values in protocol_groups.items() if len(values) != 1]
        if mixed:
            raise ValueError(
                f"mixed protocol fingerprints within method={spec.method} seed={spec.seed}: {mixed}"
            )
        if len(verifier_manifests) != 1:
            raise ValueError(
                f"mixed strict-verifier implementations in method={spec.method} seed={spec.seed}"
            )
    return records


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _sample_std(values: Iterable[float]) -> float | None:
    """Sample standard deviation across independent training-seed estimates."""

    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else None


def _is_feasible(record: Record) -> bool:
    return bool(record.get("best_of_n_feasible"))


def _gap(record: Record) -> float | None:
    if not _is_feasible(record):
        return None
    return _finite_float(record.get("best_feasible_gap"))


def normalized_feasible_gap_auc_score(
    record: Record,
    *,
    gap_range: tuple[float, float] = PRIMARY_GAP_RANGE,
) -> float:
    """Per-instance normalized AUC of feasible success over a frozen gap range.

    For threshold ``t``, the success curve is one exactly when the record is
    feasible and its verified gap is at most ``t``.  Integrating that step
    function over [0%, 10%] and dividing by the range width gives a score in
    [0, 1].  Infeasible records, missing gaps, and feasible gaps at or above 10%
    score zero, so feasibility is never conditioned away.
    """

    lower, upper = (float(gap_range[0]), float(gap_range[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or upper <= lower:
        raise ValueError("normalized feasible-gap AUC requires a finite increasing gap range")
    gap = _gap(record)
    if gap is None:
        return 0.0
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError("normalized feasible-gap AUC received an invalid feasible gap")
    if gap <= lower:
        return 1.0
    if gap >= upper:
        return 0.0
    return (upper - gap) / (upper - lower)


def normalized_feasible_gap_auc(
    records: Sequence[Record],
    *,
    gap_range: tuple[float, float] = PRIMARY_GAP_RANGE,
) -> float | None:
    if not records:
        return None
    return statistics.fmean(
        normalized_feasible_gap_auc_score(record, gap_range=gap_range) for record in records
    )


def _published_gap_target_for_task(
    task: str,
    task_target: Mapping[str, object],
    *,
    external_context: Mapping[str, object] | None = None,
) -> float | None:
    """Return an external threshold only when its registry explicitly permits it.

    The current audited sources are context-only.  In particular, the published
    values do not share a certified, per-instance denominator with the new IID
    benchmark.  Keeping this gate registry-driven prevents task-classification
    drift in the aggregation implementation.
    """

    del task
    if not isinstance(external_context, Mapping):
        return None
    if external_context.get("aggregate_point_comparison_eligible") is not True:
        return None
    if external_context.get("strict_beat_claim_eligible") is not True:
        return None
    if task_target.get("strict_beat_claim_eligible") is not True:
        return None
    return _finite_float(task_target.get("reported_gap"))


def _published_point_descriptive_comparison(
    *,
    task: str,
    observed_feasibility: float | None,
    observed_gap: float | None,
    published_feasibility: float | None,
    published_gap: float | None,
    external_context: Mapping[str, object] | None = None,
    task_target: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del task, observed_feasibility, observed_gap, published_feasibility, published_gap
    context = external_context if isinstance(external_context, Mapping) else {}
    target = task_target if isinstance(task_target, Mapping) else {}
    comparable = bool(
        context.get("aggregate_point_comparison_eligible") is True
        and context.get("strict_beat_claim_eligible") is True
        and target.get("strict_beat_claim_eligible") is True
    )
    if comparable:
        raise ValueError(
            "target registry unexpectedly enables external strict-beat comparison; "
            "the current aggregate schema supports context-only external values"
        )
    reason = str(
        target.get("reason_not_claim_eligible")
        or context.get("reason_not_claim_eligible")
        or "external point estimate is context only"
    )
    return {
        "external_point_comparison_eligible": False,
        "strict_beat_claim_eligible": False,
        "comparison_status": "CONTEXT_ONLY_NOT_COMPARABLE",
        "reason_not_claim_eligible": reason,
    }


def _candidate_counts(record: Record) -> tuple[int, int]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        total = int(record.get("num_return_sequences") or 0)
        feasible = int(record.get("feasible_candidates") or 0)
        return feasible, total
    return sum(
        bool(candidate.get("is_feasible"))
        for candidate in candidates
        if isinstance(candidate, dict)
    ), len(candidates)


def summarize_records(
    records: Sequence[Record], target_gap: float | None = None
) -> dict[str, object]:
    n = len(records)
    feasible_n = sum(_is_feasible(record) for record in records)
    gaps = [gap for record in records if (gap := _gap(record)) is not None]
    candidate_feasible = 0
    candidate_total = 0
    for record in records:
        feasible, total = _candidate_counts(record)
        candidate_feasible += feasible
        candidate_total += total
    target_success_n = None
    if target_gap is not None:
        target_success_n = sum(
            gap is not None and gap <= target_gap for record in records for gap in [_gap(record)]
        )
    generated_tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in records)
    peak_allocated = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_allocated_bytes")) is not None
    ]
    peak_reserved = [
        int(value)
        for record in records
        if (value := record.get("peak_gpu_memory_reserved_bytes")) is not None
    ]
    named_active: dict[str, int] = defaultdict(int)
    named_failures: dict[str, int] = defaultdict(int)
    named_severity: dict[str, float] = defaultdict(float)
    parse_failures = 0
    for record in records:
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            parse_failures += int(bool(candidate.get("parse_failure")))
            violations = candidate.get("constraint_violations")
            if not isinstance(violations, dict):
                continue
            for name, raw_value in violations.items():
                value = _finite_float(raw_value)
                if value is None:
                    continue
                constraint_name = str(name)
                named_active[constraint_name] += 1
                named_failures[constraint_name] += int(value > 0.0)
                named_severity[constraint_name] += value
    named_constraint_rates = {
        name: {
            "active_candidates": named_active[name],
            "failures": named_failures[name],
            "failure_rate": named_failures[name] / named_active[name],
            "mean_severity": named_severity[name] / named_active[name],
        }
        for name in sorted(named_active)
    }
    repair_time_sec = sum(
        _finite_float(record.get("total_repair_time_sec")) or 0.0 for record in records
    )
    repair_status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("repair_status"):
                repair_status_counts[str(candidate["repair_status"])] += 1
    generation_time_sec = sum(
        _finite_float(record.get("generation_time_sec")) or 0.0 for record in records
    )
    return {
        "instances": n,
        "feasible_instances": feasible_n,
        "pass_at_k": feasible_n / n if n else None,
        "pass_at_k_wilson95": wilson(feasible_n, n),
        "normalized_feasible_gap_auc": normalized_feasible_gap_auc(records),
        "normalized_feasible_gap_auc_range": list(PRIMARY_GAP_RANGE),
        "candidate_feasibility": candidate_feasible / candidate_total if candidate_total else None,
        "candidate_count": candidate_total,
        "conditional_gap_mean": _mean(gaps),
        "conditional_gap_median": quantile(gaps, 0.5),
        "conditional_gap_p90": quantile(gaps, 0.9),
        "conditional_gap_n": len(gaps),
        "success_at_target_gap": (
            target_success_n / n if target_success_n is not None and n else None
        ),
        "success_at_target_gap_wilson95": (
            wilson(target_success_n, n) if target_success_n is not None else None
        ),
        "target_gap": target_gap,
        "generation_time_sec": generation_time_sec,
        "repair_time_sec": repair_time_sec,
        "generation_plus_repair_time_sec": generation_time_sec + repair_time_sec,
        "repair_status_counts": dict(sorted(repair_status_counts.items())),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_instance": generated_tokens / n if n else None,
        "peak_gpu_memory_allocated_bytes": max(peak_allocated) if peak_allocated else None,
        "peak_gpu_memory_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        "parse_failure_rate": parse_failures / candidate_total if candidate_total else None,
        "named_constraint_failures": named_constraint_rates,
        "prompt_truncated": sum(bool(record.get("prompt_was_truncated")) for record in records),
    }


def _bootstrap_ci(
    values: Sequence[object],
    metric: Callable[[list[object]], float | None],
    *,
    repetitions: int,
    rng: random.Random,
) -> list[float] | None:
    if not values or repetitions <= 0:
        return None
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimate = metric(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    return [float(quantile(estimates, 0.025)), float(quantile(estimates, 0.975))]


def _hierarchical_paired_ci(
    pairs_by_seed: dict[int, list[tuple[Record, Record]]],
    metric: Callable[[list[object]], float | None],
    *,
    repetitions: int,
    rng: random.Random,
) -> list[float] | None:
    """Resample training seeds, then benchmark instances within each seed."""

    seeds = sorted(pairs_by_seed)
    if not seeds or repetitions <= 0:
        return None
    estimates: list[float] = []
    for _ in range(repetitions):
        sample: list[object] = []
        for _seed_slot in seeds:
            selected_seed = seeds[rng.randrange(len(seeds))]
            seed_pairs = pairs_by_seed[selected_seed]
            sample.extend(seed_pairs[rng.randrange(len(seed_pairs))] for _ in seed_pairs)
        estimate = metric(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    return [float(quantile(estimates, 0.025)), float(quantile(estimates, 0.975))]


def _hierarchical_seed_cluster_ci(
    pairs_by_seed: dict[int, list[tuple[Record, Record]]],
    metric: Callable[[list[tuple[Record, Record]]], float | None],
    *,
    repetitions: int,
    rng: random.Random,
) -> list[float] | None:
    """Resample training seeds, then paired instances, weighting seeds equally."""

    seeds = sorted(pairs_by_seed)
    if not seeds or repetitions <= 0:
        return None
    estimates: list[float] = []
    for _ in range(repetitions):
        seed_estimates: list[float] = []
        for _seed_slot in seeds:
            selected_seed = seeds[rng.randrange(len(seeds))]
            seed_pairs = pairs_by_seed[selected_seed]
            sample = [seed_pairs[rng.randrange(len(seed_pairs))] for _ in seed_pairs]
            estimate = metric(sample)
            if estimate is not None and math.isfinite(estimate):
                seed_estimates.append(float(estimate))
        if seed_estimates:
            estimates.append(statistics.fmean(seed_estimates))
    if not estimates:
        return None
    return [float(quantile(estimates, 0.025)), float(quantile(estimates, 0.975))]


def exact_two_sided_binomial(discordant_a: int, discordant_b: int) -> float:
    """Exact two-sided sign/McNemar p-value under p=0.5."""

    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    observed = min(discordant_a, discordant_b)
    lower_tail = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def exact_two_sided_seed_sign_flip(effects: Sequence[float]) -> float:
    """Exact paired randomization test over independent training-seed effects.

    Under the sharp null, each matched seed-level method difference may exchange
    sign.  Enumerating all sign assignments keeps benchmark instances inside
    their training-seed cluster and prevents pseudo-replication.
    """

    if not effects:
        raise ValueError("seed-level randomization test requires at least one effect")
    normalized = [float(value) for value in effects]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("seed-level randomization test received a non-finite effect")
    observed = abs(statistics.fmean(normalized))
    extreme = 0
    assignments = 1 << len(normalized)
    for mask in range(assignments):
        permuted = statistics.fmean(
            value if mask & (1 << index) else -value for index, value in enumerate(normalized)
        )
        if abs(permuted) >= observed - 1e-15:
            extreme += 1
    return extreme / assignments


def exact_one_sided_seed_sign_flip_greater(effects: Sequence[float]) -> float:
    """Exact paired randomization p-value for a positive seed-level effect.

    Callers shift effects to the null boundary before invoking this function.
    For pass@8 noninferiority with margin ``m``, the supplied values are
    ``challenger_minus_baseline + m`` and the alternative is positive.
    """

    if not effects:
        raise ValueError("seed-level randomization test requires at least one effect")
    normalized = [float(value) for value in effects]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("seed-level randomization test received a non-finite effect")
    observed = statistics.fmean(normalized)
    extreme = 0
    assignments = 1 << len(normalized)
    for mask in range(assignments):
        permuted = statistics.fmean(
            value if mask & (1 << index) else -value for index, value in enumerate(normalized)
        )
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / assignments


def compare_paired(
    baseline: dict[tuple[int, RecordKey], Record],
    challenger: dict[tuple[int, RecordKey], Record],
    *,
    task: str,
    target_gap: float | None,
    repetitions: int,
    rng: random.Random,
) -> dict[str, object]:
    baseline_keys = {key for key in baseline if key[1][0] == task}
    challenger_keys = {key for key in challenger if key[1][0] == task}
    if baseline_keys != challenger_keys:
        baseline_only_keys = sorted(baseline_keys - challenger_keys)
        challenger_only_keys = sorted(challenger_keys - baseline_keys)
        raise ValueError(
            f"paired key sets differ for task={task}: "
            f"baseline_only={baseline_only_keys[:8]}, "
            f"challenger_only={challenger_only_keys[:8]}"
        )
    common = sorted(baseline_keys)
    if not common:
        raise ValueError(f"no paired records for task={task}")
    pairs = [(baseline[key], challenger[key]) for key in common]
    pairs_by_seed: dict[int, list[tuple[Record, Record]]] = defaultdict(list)
    for key in common:
        pairs_by_seed[key[0]].append((baseline[key], challenger[key]))

    def pass_difference(sample: list[tuple[Record, Record]]) -> float:
        return statistics.fmean(float(_is_feasible(b)) - float(_is_feasible(a)) for a, b in sample)

    def normalized_auc_difference(sample: list[tuple[Record, Record]]) -> float:
        return statistics.fmean(
            normalized_feasible_gap_auc_score(b) - normalized_feasible_gap_auc_score(a)
            for a, b in sample
        )

    def target_difference(sample: list[tuple[Record, Record]]) -> float | None:
        if target_gap is None:
            return None
        return statistics.fmean(
            float((_gap(b) is not None) and float(_gap(b)) <= target_gap)
            - float((_gap(a) is not None) and float(_gap(a)) <= target_gap)
            for a, b in sample
        )

    def jointly_feasible_gap_difference(
        sample: list[tuple[Record, Record]],
    ) -> float | None:
        differences = [
            float(b_gap) - float(a_gap)
            for a, b in sample
            if (a_gap := _gap(a)) is not None and (b_gap := _gap(b)) is not None
        ]
        return _mean(differences)

    def counts(sample: list[tuple[Record, Record]]) -> dict[str, int]:
        baseline_only = challenger_only = wins = ties = losses = 0
        target_baseline_only = target_challenger_only = 0
        gap_improvements = gap_regressions = gap_ties = 0
        for a, b in sample:
            af, bf = _is_feasible(a), _is_feasible(b)
            if af and not bf:
                baseline_only += 1
                losses += 1
            elif bf and not af:
                challenger_only += 1
                wins += 1
            elif not af and not bf:
                ties += 1
            else:
                delta = float(_gap(b)) - float(_gap(a))
                if delta < -1e-12:
                    wins += 1
                    gap_improvements += 1
                elif delta > 1e-12:
                    losses += 1
                    gap_regressions += 1
                else:
                    ties += 1
                    gap_ties += 1
            if target_gap is not None:
                a_target = _gap(a) is not None and float(_gap(a)) <= target_gap
                b_target = _gap(b) is not None and float(_gap(b)) <= target_gap
                target_baseline_only += int(a_target and not b_target)
                target_challenger_only += int(b_target and not a_target)
        return {
            "baseline_only": baseline_only,
            "challenger_only": challenger_only,
            "challenger_wins": wins,
            "ties": ties,
            "baseline_wins": losses,
            "target_baseline_only": target_baseline_only,
            "target_challenger_only": target_challenger_only,
            "gap_improvements": gap_improvements,
            "gap_regressions": gap_regressions,
            "gap_ties": gap_ties,
        }

    per_seed: dict[str, object] = {}
    seed_pass_effects: list[float] = []
    seed_normalized_auc_effects: list[float] = []
    seed_gap_effects: list[float] = []
    seed_target_effects: list[float] = []
    for seed, seed_pairs in sorted(pairs_by_seed.items()):
        seed_counts = counts(seed_pairs)
        pass_effect = pass_difference(seed_pairs)
        normalized_auc_effect = normalized_auc_difference(seed_pairs)
        gap_effect = jointly_feasible_gap_difference(seed_pairs)
        target_effect = target_difference(seed_pairs)
        lexicographic_effect = (
            seed_counts["challenger_wins"] - seed_counts["baseline_wins"]
        ) / len(seed_pairs)
        seed_pass_effects.append(pass_effect)
        seed_normalized_auc_effects.append(normalized_auc_effect)
        if gap_effect is not None:
            seed_gap_effects.append(gap_effect)
        if target_effect is not None:
            seed_target_effects.append(target_effect)
        per_seed[str(seed)] = {
            "paired_observations": len(seed_pairs),
            "challenger_minus_baseline_normalized_feasible_gap_auc": normalized_auc_effect,
            "normalized_feasible_gap_auc_range": list(PRIMARY_GAP_RANGE),
            "normalized_feasible_gap_auc_instance_cluster_bootstrap95": _bootstrap_ci(
                list(seed_pairs),
                lambda sample: normalized_auc_difference(sample),  # type: ignore[arg-type]
                repetitions=repetitions,
                rng=rng,
            ),
            "challenger_minus_baseline_pass_at_k": pass_effect,
            "pass_at_k_instance_cluster_bootstrap95": _bootstrap_ci(
                list(seed_pairs),
                lambda sample: pass_difference(sample),  # type: ignore[arg-type]
                repetitions=repetitions,
                rng=rng,
            ),
            "mcnemar": {
                "baseline_only": seed_counts["baseline_only"],
                "challenger_only": seed_counts["challenger_only"],
                "exact_two_sided_p": exact_two_sided_binomial(
                    seed_counts["baseline_only"], seed_counts["challenger_only"]
                ),
            },
            "challenger_minus_baseline_jointly_feasible_gap": gap_effect,
            "jointly_feasible_gap_n": (
                seed_counts["gap_improvements"]
                + seed_counts["gap_regressions"]
                + seed_counts["gap_ties"]
            ),
            "gap_difference_instance_cluster_bootstrap95": _bootstrap_ci(
                list(seed_pairs),
                lambda sample: jointly_feasible_gap_difference(sample),  # type: ignore[arg-type]
                repetitions=repetitions,
                rng=rng,
            ),
            "jointly_feasible_gap_sign_test": {
                "challenger_improvements": seed_counts["gap_improvements"],
                "challenger_regressions": seed_counts["gap_regressions"],
                "ties": seed_counts["gap_ties"],
                "exact_two_sided_p": exact_two_sided_binomial(
                    seed_counts["gap_improvements"],
                    seed_counts["gap_regressions"],
                ),
            },
            "challenger_minus_baseline_success_at_target_gap": target_effect,
            "challenger_minus_baseline_lexicographic_win_score": lexicographic_effect,
            "target_success_instance_cluster_bootstrap95": _bootstrap_ci(
                list(seed_pairs),
                lambda sample: target_difference(sample),  # type: ignore[arg-type]
                repetitions=repetitions,
                rng=rng,
            ),
            "target_success_mcnemar": (
                {
                    "baseline_only": seed_counts["target_baseline_only"],
                    "challenger_only": seed_counts["target_challenger_only"],
                    "exact_two_sided_p": exact_two_sided_binomial(
                        seed_counts["target_baseline_only"],
                        seed_counts["target_challenger_only"],
                    ),
                }
                if target_gap is not None
                else None
            ),
            "lexicographic_exact_test": {
                "challenger_wins": seed_counts["challenger_wins"],
                "ties": seed_counts["ties"],
                "baseline_wins": seed_counts["baseline_wins"],
                "exact_two_sided_p": exact_two_sided_binomial(
                    seed_counts["challenger_wins"], seed_counts["baseline_wins"]
                ),
            },
        }

    pooled_counts = counts(pairs)
    return {
        "paired_observations": len(pairs),
        "paired_training_seeds": sorted(pairs_by_seed),
        "primary_inference_unit": "training_seed_with_instance_cluster_resampling",
        "per_training_seed": per_seed,
        "challenger_minus_baseline_normalized_feasible_gap_auc": statistics.fmean(
            seed_normalized_auc_effects
        ),
        "challenger_minus_baseline_normalized_feasible_gap_auc_seed_sample_std": _sample_std(
            seed_normalized_auc_effects
        ),
        "normalized_feasible_gap_auc_hierarchical_bootstrap95": _hierarchical_seed_cluster_ci(
            pairs_by_seed,
            normalized_auc_difference,
            repetitions=repetitions,
            rng=rng,
        ),
        "challenger_minus_baseline_pass_at_k": statistics.fmean(seed_pass_effects),
        "challenger_minus_baseline_pass_at_k_seed_sample_std": _sample_std(seed_pass_effects),
        "pass_at_k_hierarchical_bootstrap95": _hierarchical_seed_cluster_ci(
            pairs_by_seed,
            pass_difference,
            repetitions=repetitions,
            rng=rng,
        ),
        "challenger_minus_baseline_jointly_feasible_gap": _mean(seed_gap_effects),
        "challenger_minus_baseline_jointly_feasible_gap_seed_sample_std": _sample_std(
            seed_gap_effects
        ),
        "gap_difference_hierarchical_bootstrap95": _hierarchical_seed_cluster_ci(
            pairs_by_seed,
            jointly_feasible_gap_difference,
            repetitions=repetitions,
            rng=rng,
        ),
        "challenger_minus_baseline_success_at_target_gap": _mean(seed_target_effects),
        "challenger_minus_baseline_success_at_target_gap_seed_sample_std": _sample_std(
            seed_target_effects
        ),
        "success_at_target_hierarchical_bootstrap95": _hierarchical_seed_cluster_ci(
            pairs_by_seed,
            target_difference,
            repetitions=repetitions,
            rng=rng,
        ),
        "pooled_for_descriptive_use_only": {
            "challenger_minus_baseline_normalized_feasible_gap_auc": normalized_auc_difference(
                pairs
            ),
            "challenger_minus_baseline_pass_at_k": pass_difference(pairs),
            "challenger_minus_baseline_jointly_feasible_gap": (
                jointly_feasible_gap_difference(pairs)
            ),
            "challenger_minus_baseline_success_at_target_gap": target_difference(pairs),
            "mcnemar_discordant_counts": {
                "baseline_only": pooled_counts["baseline_only"],
                "challenger_only": pooled_counts["challenger_only"],
            },
            "lexicographic": {
                "challenger_wins": pooled_counts["challenger_wins"],
                "ties": pooled_counts["ties"],
                "baseline_wins": pooled_counts["baseline_wins"],
            },
            "note": "No pooled p-value is reported because instances across training seeds are pseudo-replicates.",
        },
    }


def _validate_paired_provenance(
    baseline: dict[tuple[int, RecordKey], Record],
    challenger: dict[tuple[int, RecordKey], Record],
) -> None:
    if set(baseline) != set(challenger):
        raise ValueError(
            "paired method key sets differ: "
            f"baseline_only={sorted(set(baseline) - set(challenger))[:8]}, "
            f"challenger_only={sorted(set(challenger) - set(baseline))[:8]}"
        )
    for seed_and_key in sorted(baseline):
        baseline_record = baseline[seed_and_key]
        challenger_record = challenger[seed_and_key]
        label = f"training_seed={seed_and_key[0]}, key={seed_and_key[1]}"
        if _record_protocol_fingerprint(baseline_record) != _record_protocol_fingerprint(
            challenger_record
        ):
            raise ValueError(f"paired protocol mismatch at {label}")
        if baseline_record.get("source_row_locator") != challenger_record.get("source_row_locator"):
            raise ValueError(f"paired source-row mismatch at {label}")
        if baseline_record.get("instance_sha256") != challenger_record.get("instance_sha256"):
            raise ValueError(f"paired instance mismatch at {label}")
        for identity_field in (
            "semantic_instance_sha256",
            "benchmark_row_sha256",
            "reference_output_sha256",
        ):
            if baseline_record.get(identity_field) != challenger_record.get(identity_field):
                raise ValueError(f"paired {identity_field} mismatch at {label}")
        baseline_verifier = baseline_record.get("strict_verifier")
        challenger_verifier = challenger_record.get("strict_verifier")
        if baseline_verifier != challenger_verifier:
            raise ValueError(f"paired strict-verifier mismatch at {label}")


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _apply_seed_family_holm(comparisons: Mapping[str, object]) -> dict[str, object]:
    """Apply the sole primary test family over the five publication tasks."""

    effect_fields = {
        "normalized_feasible_gap_auc_seed_randomization": (
            "challenger_minus_baseline_normalized_feasible_gap_auc"
        ),
    }
    if set(effect_fields) != set(CONFIRMATION_OUTCOME_FAMILIES):
        raise ValueError("analysis implementation disagrees with frozen outcome families")
    if set(comparisons) != set(PUBLICATION_TASKS):
        raise ValueError("task-level inference requires the exact five-task suite")
    raw_by_family: dict[str, dict[str, float]] = {family: {} for family in effect_fields}
    tests_by_task: dict[str, dict[str, object]] = {}
    expected_seeds = [str(seed) for seed in CONFIRMATION_TRAINING_SEEDS]
    for task in PUBLICATION_TASKS:
        raw_comparison = comparisons.get(task)
        per_seed = (
            raw_comparison.get("per_training_seed") if isinstance(raw_comparison, dict) else None
        )
        if not isinstance(per_seed, dict) or sorted(per_seed) != expected_seeds:
            raise ValueError(
                f"task-level inference requires exact seeds {expected_seeds} for {task}"
            )
        task_tests: dict[str, object] = {}
        for family, effect_field in effect_fields.items():
            seed_effects: dict[str, float] = {}
            for seed in expected_seeds:
                seed_row = per_seed.get(seed)
                effect = _finite_float(
                    seed_row.get(effect_field) if isinstance(seed_row, dict) else None
                )
                if effect is None:
                    raise ValueError(f"task-level {family} is undefined for {task}/seed{seed}")
                seed_effects[seed] = effect
            p_value = exact_two_sided_seed_sign_flip(list(seed_effects.values()))
            raw_by_family[family][task] = p_value
            task_tests[family] = {
                "claim_role": "primary",
                "inference_unit": "paired_training_seed",
                "training_seed_effects": seed_effects,
                "effect_mean": statistics.fmean(seed_effects.values()),
                "exact_two_sided_p": p_value,
                "method": "exhaustive_sign_flip_of_seed_level_effects",
                "sign_assignments": 1 << len(seed_effects),
            }
        assert isinstance(raw_comparison, dict)
        raw_comparison["task_level_exact_tests"] = task_tests
        tests_by_task[task] = task_tests

    adjusted_by_family = {
        family: holm_adjust(raw_values) for family, raw_values in raw_by_family.items()
    }
    result: dict[str, object] = {}
    for family in CONFIRMATION_OUTCOME_FAMILIES:
        if len(raw_by_family[family]) != len(PUBLICATION_TASKS):
            raise ValueError(f"{family} does not contain exactly five task hypotheses")
        hypotheses: dict[str, object] = {}
        for task in PUBLICATION_TASKS:
            raw_p = raw_by_family[family][task]
            adjusted_p = adjusted_by_family[family][task]
            task_test = tests_by_task[task][family]
            assert isinstance(task_test, dict)
            detected = adjusted_p <= CONFIRMATION_ALPHA
            positive = float(task_test["effect_mean"]) > 0.0
            directional_conclusion = (
                "primary_superiority_support"
                if detected and positive
                else "difference_detected_against_clasp"
                if detected
                else "no_primary_difference_detected"
            )
            task_test["holm_adjusted_p_across_tasks"] = adjusted_p
            task_test["difference_detected_at_alpha"] = detected
            task_test["primary_superiority_support"] = detected and positive
            task_test["directional_conclusion"] = directional_conclusion
            hypotheses[task] = {
                "raw_exact_p": raw_p,
                "holm_adjusted_p": adjusted_p,
                "difference_detected_at_alpha": detected,
                "primary_superiority_support": detected and positive,
                "directional_conclusion": directional_conclusion,
            }
        result[family] = {
            "claim_role": "primary",
            "family_size": len(PUBLICATION_TASKS),
            "alpha": CONFIRMATION_ALPHA,
            "hypotheses": hypotheses,
        }
    return result


def _apply_pass_at_k_noninferiority_holm(
    comparisons: Mapping[str, object],
    *,
    margin: float = PASS_AT_K_NONINFERIORITY_MARGIN,
    alpha: float = CONFIRMATION_ALPHA,
) -> dict[str, object]:
    """Evaluate the secondary pass@8 noninferiority gate across five tasks."""

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("pass@8 noninferiority margin must be finite and positive")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("pass@8 noninferiority alpha must be in (0, 1)")
    if set(comparisons) != set(PUBLICATION_TASKS):
        raise ValueError("pass@8 noninferiority requires the exact five-task suite")
    expected_seeds = [str(seed) for seed in CONFIRMATION_TRAINING_SEEDS]
    raw_p_values: dict[str, float] = {}
    task_rows: dict[str, dict[str, object]] = {}
    for task in PUBLICATION_TASKS:
        comparison = comparisons.get(task)
        per_seed = comparison.get("per_training_seed") if isinstance(comparison, dict) else None
        if not isinstance(per_seed, dict) or sorted(per_seed) != expected_seeds:
            raise ValueError(
                f"pass@8 noninferiority requires exact seeds {expected_seeds} for {task}"
            )
        seed_effects: dict[str, float] = {}
        for seed in expected_seeds:
            seed_row = per_seed.get(seed)
            effect = _finite_float(
                seed_row.get("challenger_minus_baseline_pass_at_k")
                if isinstance(seed_row, dict)
                else None
            )
            if effect is None:
                raise ValueError(f"pass@8 noninferiority is undefined for {task}/seed{seed}")
            seed_effects[seed] = effect
        effect_mean = statistics.fmean(seed_effects.values())
        shifted_effects = {seed: effect + margin for seed, effect in seed_effects.items()}
        raw_p = exact_one_sided_seed_sign_flip_greater(list(shifted_effects.values()))
        raw_p_values[task] = raw_p
        task_rows[task] = {
            "claim_role": "secondary_noninferiority_gate",
            "metric": "strict_shared_verifier_pass_at_8",
            "inference_unit": "paired_training_seed",
            "training_seed_effects": seed_effects,
            "margin": margin,
            "effect_mean": effect_mean,
            "null_hypothesis": "challenger_minus_baseline <= -margin",
            "boundary_shifted_training_seed_effects": shifted_effects,
            "exact_one_sided_p": raw_p,
            "method": "exhaustive_one_sided_sign_flip_at_noninferiority_boundary",
            "sign_assignments": 1 << len(seed_effects),
        }

    adjusted = holm_adjust(raw_p_values)
    for task in PUBLICATION_TASKS:
        row = task_rows[task]
        adjusted_p = adjusted[task]
        effect_requirement_pass = float(row["effect_mean"]) >= -margin - 1e-15
        inference_requirement_pass = adjusted_p <= alpha
        row["holm_adjusted_p_across_tasks"] = adjusted_p
        row["observed_mean_requirement_pass"] = effect_requirement_pass
        row["holm_inference_requirement_pass"] = inference_requirement_pass
        row["task_noninferiority_pass"] = (
            effect_requirement_pass and inference_requirement_pass
        )
        comparison = comparisons[task]
        assert isinstance(comparison, dict)
        comparison["pass_at_k_noninferiority_test"] = row

    return {
        "family": CONFIRMATION_NONINFERIORITY_FAMILY,
        "claim_role": "secondary_noninferiority_gate",
        "family_size": len(PUBLICATION_TASKS),
        "alpha": alpha,
        "margin": margin,
        "overall_pass": all(
            bool(task_rows[task]["task_noninferiority_pass"]) for task in PUBLICATION_TASKS
        ),
        "tasks": task_rows,
    }


def _provenance_summary(records: dict[RecordKey, Record]) -> dict[str, object]:
    artifact_fingerprints = sorted(
        {
            f"{key}:{value}"
            for record in records.values()
            for key, value in [_record_artifact_fingerprint(record)]
            if key
        }
    )
    protocol_fingerprints = sorted(
        {
            _record_protocol_fingerprint(record)
            for record in records.values()
            if _record_protocol_fingerprint(record)
        }
    )
    verifier_manifests = sorted(
        {
            str(verifier.get("implementation_manifest_sha256"))
            for record in records.values()
            for verifier in [record.get("strict_verifier")]
            if isinstance(verifier, dict) and verifier.get("implementation_manifest_sha256")
        }
    )
    data_manifests = sorted(
        {
            str(locator.get("data_manifest_sha256"))
            for record in records.values()
            for locator in [record.get("source_row_locator")]
            if isinstance(locator, dict) and locator.get("data_manifest_sha256")
        }
    )
    return {
        "artifact_fingerprints": artifact_fingerprints,
        "protocol_fingerprints": protocol_fingerprints,
        "strict_verifier_manifests": verifier_manifests,
        "data_manifests": data_manifests,
        "records": len(records),
    }


def _load_lane_contract_registry(path: str) -> tuple[dict[str, object], str]:
    registry = _load_json_object(path)
    registry_hash = _validate_self_hash(registry, "registry_sha256", label=path)
    if registry.get("schema_version") != 1 or registry.get("status") != "frozen_policy":
        raise ValueError("lane contract registry is not a frozen schema-v1 policy")
    lanes = registry.get("lanes")
    required_lanes = {"raw_n8", "repair_n8", "repair_adaptive_n8_64"}
    if not isinstance(lanes, dict) or set(lanes) != required_lanes:
        raise ValueError("lane contract registry does not define the exact publication lanes")
    required_confirmation = {
        name
        for name, lane in lanes.items()
        if isinstance(lane, dict) and lane.get("required_for_confirmation") is True
    }
    if required_confirmation != {"raw_n8", "repair_n8"}:
        raise ValueError("frozen policy must require exactly raw_n8 and repair_n8 for confirmation")
    expected_repair_contract = {
        "system_lane": APPROVED_REPAIR_SYSTEM_LANE,
        "system_name": APPROVED_REPAIR_SYSTEM_NAME,
        "schema_version": REPAIR_SCHEMA_VERSION,
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "repair_config_exact": APPROVED_REPAIR_CONFIG,
        "repair_implementation": _repair_implementation_identity(),
    }
    for lane_name in ("repair_n8", "repair_adaptive_n8_64"):
        lane = lanes.get(lane_name)
        if not isinstance(lane, dict) or lane.get("repair_contract") != expected_repair_contract:
            raise ValueError(f"lane {lane_name} does not freeze the approved deterministic repair")
    return registry, registry_hash


def _validate_target_registry(
    registry: Mapping[str, object],
    *,
    lane_registry: Mapping[str, object],
    label: str,
) -> str:
    registry_sha256 = _validate_self_hash(registry, "registry_sha256", label=label)
    if registry_sha256 != APPROVED_TARGET_REGISTRY_SHA256:
        raise ValueError(
            "target registry is self-consistent but is not the reviewed external-context "
            f"registry: observed={registry_sha256}"
        )
    expected_top_level = {
        "schema_version",
        "status",
        "frozen_at_utc",
        "metric_units",
        "tasks",
        "new_iid_reference_contract",
        "lanes",
        "context_only",
        "registry_sha256",
    }
    if (
        set(registry) != expected_top_level
        or registry.get("schema_version") != 3
        or (registry.get("status") != "frozen_external_context_registry")
    ):
        raise ValueError("target registry does not use the exact frozen schema-v3 contract")
    if registry.get("tasks") != list(PUBLICATION_TASKS):
        raise ValueError("target registry does not cover the exact five-task suite")
    metric_units = registry.get("metric_units")
    expected_metric_units = {
        "feasibility": ("fraction of instances with at least one strictly feasible candidate"),
        "conditional_gap": (
            "mean conditional gap-to-reference over instances with a feasible candidate; "
            "optimality-gap wording is reserved for certified references"
        ),
        "external_reported_gap": (
            "author-reported aggregate table value, retained verbatim as a fraction and "
            "not reinterpreted as sharing CLASP's denominator or certification semantics"
        ),
    }
    if metric_units != expected_metric_units:
        raise ValueError("target registry metric semantics are invalid")

    iid_contract = registry.get("new_iid_reference_contract")
    if (
        not isinstance(iid_contract, dict)
        or set(iid_contract) != {"status", "contract_path", "separation_rule", "tasks"}
        or iid_contract.get("status") != "separate_blocked_reference_contract"
        or not str(iid_contract.get("contract_path") or "")
        or not str(iid_contract.get("separation_rule") or "")
    ):
        raise ValueError("target registry does not separate the new IID reference contract")
    iid_tasks = iid_contract.get("tasks")
    if not isinstance(iid_tasks, dict) or set(iid_tasks) != set(PUBLICATION_TASKS):
        raise ValueError("new IID reference contract does not cover the five-task suite")
    iid_reference_kinds = {
        "best_known_heuristic",
        "certified_optimum_required",
    }
    for task, task_semantics in iid_tasks.items():
        if (
            not isinstance(task_semantics, dict)
            or set(task_semantics)
            != {
                "intended_reference_kind",
                "permitted_gap_label_if_approved",
                "certification_requirement",
            }
            or task_semantics.get("intended_reference_kind") not in iid_reference_kinds
            or not str(task_semantics.get("permitted_gap_label_if_approved") or "")
            or not str(task_semantics.get("certification_requirement") or "")
        ):
            raise ValueError(f"new IID reference semantics are invalid for {task}")
    if (
        iid_tasks["op"].get("intended_reference_kind") != "certified_optimum_required"
        or iid_tasks["op"].get("permitted_gap_label_if_approved") != "conditional_optimality_gap"
    ):
        raise ValueError("new IID OP must remain a distinct exact-reference contract")

    target_lanes = registry.get("lanes")
    lane_policies = lane_registry.get("lanes")
    if (
        not isinstance(target_lanes, dict)
        or not isinstance(lane_policies, dict)
        or set(target_lanes) != set(lane_policies)
    ):
        raise ValueError("target/lane registries do not define the same exact lanes")
    allowed_external_reference_semantics = {
        "heuristic_reference",
        "cross_protocol_invalid",
        "author_reported_gurobi_uncertified",
        "time_limited_near_optimal_reference",
    }
    historical_contexts = 0
    for lane_name, target_lane in target_lanes.items():
        lane_policy = lane_policies.get(lane_name)
        if not isinstance(target_lane, dict) or not isinstance(lane_policy, dict):
            raise ValueError(f"target lane {lane_name} is invalid")
        if set(target_lane) != {
            "candidate_budget",
            "postprocessing",
            "local_lane_policy_projection",
            "external_context",
        }:
            raise ValueError(f"target lane {lane_name} has an invalid schema")
        policy_projection = target_lane.get("local_lane_policy_projection")
        expected_policy_projection = {
            "publication_status": lane_policy.get("publication_status"),
            "claim_eligible": lane_policy.get("claim_eligible"),
            "required_for_confirmation": lane_policy.get("required_for_confirmation"),
        }
        if (
            target_lane.get("candidate_budget") != lane_policy.get("reported_candidate_budget")
            or policy_projection != expected_policy_projection
            or not str(target_lane.get("postprocessing") or "")
        ):
            raise ValueError(f"target lane {lane_name} disagrees with frozen lane policy")

        reference = target_lane.get("external_context")
        if (
            not isinstance(reference, dict)
            or set(reference)
            != {
                "context_id",
                "name",
                "status",
                "sources",
                "evidence_kind",
                "comparison_scope",
                "aggregate_point_comparison_eligible",
                "strict_beat_claim_eligible",
                "reason_not_claim_eligible",
                "scope",
                "reported_point_estimates",
            }
            or any(
                not str(reference.get(field) or "")
                for field in (
                    "context_id",
                    "name",
                    "status",
                    "evidence_kind",
                    "comparison_scope",
                    "reason_not_claim_eligible",
                )
            )
            or reference.get("aggregate_point_comparison_eligible") is not False
            or reference.get("strict_beat_claim_eligible") is not False
        ):
            raise ValueError(f"target lane {lane_name} external values are not fail-closed context")
        sources = reference.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"target lane {lane_name} has no primary-source disclosure")
        for source in sources:
            if (
                not isinstance(source, dict)
                or set(source) != {"kind", "url"}
                or not str(source.get("kind") or "")
                or not str(source.get("url") or "").startswith("https://")
            ):
                raise ValueError(f"target lane {lane_name} has an invalid source row")
        scope = reference.get("scope")
        if not isinstance(scope, dict) or not scope:
            raise ValueError(f"target lane {lane_name} has no explicit context scope")
        if reference.get("comparison_scope") == "historical_first_100_author_protocol_only":
            historical_contexts += 1
            if scope != {
                "dataset": "released_historical_evaluation",
                "row_interval": [0, 100],
                "instances_per_task": 100,
                "candidate_budget": 8,
                "decoding": "author_protocol_paper_top_p_0.7",
                "reference_values": "author_supplied_task_solver_outputs",
            }:
                raise ValueError(
                    "LLMCoSolver context must be restricted to the historical first-100 "
                    "author protocol"
                )
        results = reference.get("reported_point_estimates")
        if not isinstance(results, dict) or set(results) != set(PUBLICATION_TASKS):
            raise ValueError(f"target lane {lane_name} has an incomplete context table")
        for task in PUBLICATION_TASKS:
            row = results.get(task)
            feasibility = _finite_float(row.get("feasibility")) if isinstance(row, dict) else None
            gap = _finite_float(row.get("reported_gap")) if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "feasibility",
                    "reported_gap",
                    "reference_semantics",
                    "comparison_status",
                    "strict_beat_claim_eligible",
                    "reason_not_claim_eligible",
                }
                or feasibility is None
                or not 0.0 <= feasibility <= 1.0
                or gap is None
                or gap < 0.0
                or row.get("reference_semantics") not in allowed_external_reference_semantics
                or row.get("comparison_status") != "context_only"
                or row.get("strict_beat_claim_eligible") is not False
                or not str(row.get("reason_not_claim_eligible") or "")
            ):
                raise ValueError(
                    f"target lane {lane_name}/{task} is not a valid context-only point"
                )
    if historical_contexts != 1:
        raise ValueError(
            "target registry must contain exactly one historical first-100 author-protocol context"
        )
    context_only = registry.get("context_only")
    if not isinstance(context_only, list) or not context_only:
        raise ValueError("target registry has no context-only source disclosures")
    for index, row in enumerate(context_only):
        if not isinstance(row, dict) or any(
            not str(row.get(field) or "")
            for field in ("name", "source", "code", "status", "reason_not_in_numeric_table")
        ):
            raise ValueError(f"target context-only source {index} is invalid")
    return registry_sha256


def _load_target_registry(
    path: str,
    *,
    lane_registry: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    raw_bytes, snapshot = _read_stable_file_bytes(path)
    try:
        registry = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"target registry is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(registry, dict):
        raise ValueError("target registry must be a JSON object")
    registry_sha256 = _validate_target_registry(
        registry,
        lane_registry=lane_registry,
        label=path,
    )
    _assert_path_matches_snapshot(path, snapshot)
    return registry, registry_sha256


def _require_supported_lane_claim(
    lane_name: str,
    lane: Mapping[str, object],
) -> None:
    if lane.get("claim_eligible") is not True:
        status = str(lane.get("publication_status") or "unsupported")
        raise ValueError(
            f"lane {lane_name} is not eligible for a publication/SOTA claim under the "
            f"frozen policy ({status}); a future prespecified producer and policy receipt "
            "are required"
        )
    if lane_name == "repair_adaptive_n8_64" and not isinstance(lane.get("producer_contract"), dict):
        raise ValueError("adaptive publication claims require a prespecified producer_contract")


def _require_exact_contract_fields(
    actual: object,
    expected: object,
    *,
    label: str,
) -> None:
    """Recursively require every policy field, while permitting bound dynamic fields."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{label} must be an object")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"{label}.{key} is missing")
            _require_exact_contract_fields(actual[key], expected_value, label=f"{label}.{key}")
        return
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, observed={actual!r}")


def _validate_native_lane_contract(
    provenance: Mapping[str, object],
    *,
    registry: Mapping[str, object],
    lane_name: str,
    task: str,
    confirmation: bool,
) -> None:
    common = registry.get("native_contract")
    lanes = registry.get("lanes")
    if not isinstance(common, dict) or not isinstance(lanes, dict):
        raise ValueError("lane contract registry has no native contract")
    lane = lanes.get(lane_name)
    if not isinstance(lane, dict):
        raise ValueError(f"unknown lane contract {lane_name!r}")
    prompting = provenance.get("prompting")
    generation = provenance.get("generation")
    selection = provenance.get("selection")
    _require_exact_contract_fields(
        prompting,
        common.get("prompting_exact"),
        label="native prompt contract",
    )
    _require_exact_contract_fields(
        generation,
        common.get("generation_exact"),
        label="native decoding contract",
    )
    if not isinstance(generation, dict):
        raise ValueError("native generation provenance is missing")
    budget = lane.get("source_candidate_budget")
    if isinstance(budget, int):
        if generation.get("num_return_sequences") != budget:
            raise ValueError(
                f"lane {lane_name} requires native K={budget}, observed "
                f"{generation.get('num_return_sequences')!r}"
            )
        if generation.get("generation_batch_size") != budget:
            raise ValueError(f"lane {lane_name} requires K candidates in one generate call")
    elif isinstance(budget, dict):
        maximum = int(budget.get("maximum", -1))
        batch = int(budget.get("generation_batch_size", -1))
        if generation.get("num_return_sequences") != maximum:
            raise ValueError(f"adaptive lane source must expose the prespecified max K={maximum}")
        if generation.get("generation_batch_size") != batch:
            raise ValueError("adaptive lane source has the wrong generation batch size")
    else:
        raise ValueError(f"lane {lane_name} has an invalid source candidate budget")

    if confirmation:
        _require_exact_contract_fields(
            selection,
            common.get("confirmation_selection_exact"),
            label="confirmation selection contract",
        )
        if not isinstance(selection, dict) or selection.get("tasks") != [task]:
            raise ValueError(
                "confirmation must evaluate exactly the specialist task bound by the run contract"
            )
        if not isinstance(selection.get("benchmark_manifest"), dict):
            raise ValueError("confirmation provenance has no frozen benchmark binding")
        if not isinstance(selection.get("checkpoint_selection_binding"), dict):
            raise ValueError("confirmation provenance has no checkpoint-selection binding")
        if not isinstance(selection.get("checkpoint_selection_receipt"), dict):
            raise ValueError("confirmation provenance has no checkpoint-selection receipt identity")


def _require_positive_resource(record: Record, field: str, *, label: str) -> float:
    value = _finite_float(record.get(field))
    if value is None or value <= 0.0:
        raise ValueError(f"{label} requires positive {field}")
    return value


def _checkpoint_artifact_sha256(provenance: Mapping[str, object]) -> str:
    model = provenance.get("model")
    artifact = model.get("checkpoint_artifact") if isinstance(model, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError("experiment provenance has no checkpoint artifact identity")
    digest = str(artifact.get("manifest_sha256") or artifact.get("sha256") or "")
    if not _is_sha256(digest):
        raise ValueError("experiment provenance checkpoint artifact digest is invalid")
    return digest


def _validate_lane_evidence(
    spec: RunSpec,
    records: Mapping[RecordKey, Record],
    evidences: Sequence[ArtifactEvidence],
    *,
    lane_name: str,
    lane_registry: Mapping[str, object],
    confirmation: bool,
) -> None:
    if not evidences:
        raise ValueError(f"run has no publication companion evidence: {spec.path}")
    observed_tasks = sorted({key[0] for key in records})
    if spec.task is not None and observed_tasks != [spec.task]:
        raise ValueError(
            f"--run task label does not match its exact detail task set: "
            f"declared={spec.task!r}, observed={observed_tasks}"
        )
    expected_artifact_type = "native" if lane_name == "raw_n8" else "repair"
    artifact_types = {evidence.artifact_type for evidence in evidences}
    if artifact_types != {expected_artifact_type}:
        raise ValueError(
            f"lane {lane_name} requires {expected_artifact_type} artifacts; "
            f"observed={sorted(artifact_types)}. Rescore and repair outputs cannot be "
            "relabeled as raw generation."
        )

    for evidence in evidences:
        contract = evidence.training_run_contract
        if not isinstance(contract, dict):
            raise ValueError(
                f"publication evidence has no training contract: {evidence.details_path}"
            )
        contract_task = str(contract.get("task") or "")
        _validate_training_contract(
            contract,
            evidence.training_run_contract_sha256,
            method=spec.method,
            seed=spec.seed,
            task=contract_task,
            label=evidence.details_path,
        )
        if contract_task not in observed_tasks:
            raise ValueError(
                f"artifact training contract task {contract_task!r} is absent from its details"
            )
        if spec.task is not None and contract_task != spec.task:
            raise ValueError(
                f"--run task {spec.task!r} disagrees with embedded contract task {contract_task!r}"
            )
        _validate_native_lane_contract(
            evidence.experiment_provenance,
            registry=lane_registry,
            lane_name=lane_name,
            task=contract_task,
            confirmation=confirmation,
        )
        if confirmation:
            binding = evidence.checkpoint_selection_binding
            receipt = evidence.checkpoint_selection_receipt
            if not isinstance(binding, dict) or not isinstance(receipt, dict):
                raise ValueError("confirmation lacks immutable checkpoint selection evidence")
            expected_group = f"{contract_task}/{spec.method}/seed{spec.seed}"
            checkpoint_artifact_sha256 = _checkpoint_artifact_sha256(evidence.experiment_provenance)
            receipt_size = receipt.get("size_bytes")
            checkpoint_step = binding.get("checkpoint_step")
            horizon = contract.get("fixed_horizon")
            max_steps = horizon.get("max_steps") if isinstance(horizon, dict) else None
            if (
                binding.get("group_id") != expected_group
                or binding.get("run_contract_sha256") != evidence.training_run_contract_sha256
                or binding.get("checkpoint_artifact_sha256") != checkpoint_artifact_sha256
                or binding.get("grpo_beta") != contract.get("training_config", {}).get("beta")
                or binding.get("fixed_kl_reference") != contract.get("fixed_kl_reference")
                or binding.get("exact_peft_kl_runtime") != evidence.exact_peft_kl_runtime
                or binding.get("exact_peft_kl_post_step") != evidence.exact_peft_kl_post_step
                or isinstance(checkpoint_step, bool)
                or not isinstance(checkpoint_step, int)
                or checkpoint_step <= 0
                or not isinstance(max_steps, int)
                or checkpoint_step > max_steps
                or not _is_sha256(receipt.get("receipt_sha256"))
                or not _is_sha256(receipt.get("file_sha256"))
                or isinstance(receipt_size, bool)
                or not isinstance(receipt_size, int)
                or receipt_size <= 0
                or not str(receipt.get("campaign_id") or "")
                or not str(receipt.get("stage_id") or "")
                or binding.get("checkpoint_selection_receipt_sha256")
                != receipt.get("receipt_sha256")
                or not _is_sha256(binding.get("selection_evaluation_fingerprint"))
            ):
                raise ValueError("checkpoint selection evidence disagrees with training contract")

    lanes = lane_registry.get("lanes")
    lane = lanes.get(lane_name) if isinstance(lanes, dict) else None
    if not isinstance(lane, dict):
        raise ValueError(f"lane policy is missing: {lane_name}")
    _require_supported_lane_claim(lane_name, lane)
    budget = lane.get("reported_candidate_budget")
    for key, record in records.items():
        label = f"{spec.method}/seed{spec.seed}/{key}"
        artifact_type = _record_artifact_type(record, label=label)
        if lane_name == "raw_n8":
            forbidden = {
                "repair_fingerprint",
                "rescore_fingerprint",
                "repair_record_sha256",
                "rescore_record_sha256",
                "raw_record",
                "repaired_strict_view",
                "system_lane",
            }
            present = sorted(field for field in forbidden if field in record)
            if artifact_type != "native" or present:
                raise ValueError(
                    f"raw lane record carries repair/rescore ambiguity: {label}: {present}"
                )
        elif artifact_type != "repair" or not isinstance(record.get("raw_record"), dict):
            raise ValueError(f"repair lane requires a content-bound repair artifact: {label}")

        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"lane record has no candidate list: {label}")
        observed_k = len(candidates)
        if isinstance(budget, int) and observed_k != budget:
            raise ValueError(
                f"lane {lane_name} requires reported K={budget}; observed K={observed_k} at {label}"
            )
        _require_positive_resource(record, "generated_tokens", label=label)
        generation_time = _require_positive_resource(record, "generation_time_sec", label=label)
        _require_positive_resource(record, "peak_gpu_memory_allocated_bytes", label=label)
        _require_positive_resource(record, "peak_gpu_memory_reserved_bytes", label=label)

        if lane_name == "repair_adaptive_n8_64":
            if not isinstance(budget, dict):
                raise ValueError("adaptive lane has no candidate-budget range")
            minimum = int(budget.get("minimum", -1))
            maximum = int(budget.get("maximum", -1))
            if observed_k < minimum or observed_k > maximum:
                raise ValueError(
                    f"adaptive lane K must be in [{minimum}, {maximum}], observed {observed_k}"
                )
            adaptive = record.get("adaptive_compute")
            if not isinstance(adaptive, dict):
                raise ValueError(f"adaptive lane record has no adaptive_compute receipt: {label}")
            expected_adaptive = {
                "schema_version": 1,
                "policy": str(lane.get("adaptive_policy")),
                "initial_k": minimum,
                "maximum_k": maximum,
                "candidates_generated": observed_k,
            }
            _require_exact_contract_fields(
                adaptive,
                expected_adaptive,
                label=f"{label}.adaptive_compute",
            )
            if not str(adaptive.get("stop_reason") or ""):
                raise ValueError(f"adaptive lane has no stop_reason: {label}")
            wall_time = _finite_float(adaptive.get("wall_time_sec"))
            repair_time = _finite_float(record.get("total_repair_time_sec"))
            if wall_time is None or wall_time <= 0.0 or repair_time is None or repair_time < 0.0:
                raise ValueError(f"adaptive lane lacks finite wall/repair time: {label}")
            if wall_time + 1e-12 < generation_time + repair_time:
                raise ValueError(
                    f"adaptive wall time is smaller than measured component time: {label}"
                )


def _load_campaign_manifest(
    path: str,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_bytes, snapshot = _read_stable_file_bytes(path)
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"campaign manifest is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must contain a JSON object")
    manifest_hash = _validate_self_hash(
        manifest,
        "campaign_manifest_sha256",
        label=path,
    )
    if manifest.get("schema_version") != CONFIRMATION_CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            "campaign manifest must use schema_version "
            f"{CONFIRMATION_CAMPAIGN_SCHEMA_VERSION}"
        )
    if manifest.get("status") != "frozen_before_confirmation":
        raise ValueError("campaign manifest was not frozen before confirmation")
    frozen_at = manifest.get("frozen_at_utc")
    if (
        not isinstance(frozen_at, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            frozen_at,
        )
        is None
    ):
        raise ValueError("campaign manifest has no canonical pre-confirmation freeze time")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign manifest has no campaign_id")
    identity = {
        "basename": os.path.basename(os.path.realpath(path)),
        "file_sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "campaign_manifest_sha256": manifest_hash,
        "campaign_id": campaign_id,
    }
    _assert_path_matches_snapshot(path, snapshot)
    return manifest, identity


def _validate_benchmark_provenance_bindings(
    evidence_by_spec: Mapping[RunSpec, Sequence[ArtifactEvidence]],
    *,
    benchmark_path: str,
    benchmark_manifest: Mapping[str, object],
) -> None:
    raw_bytes, snapshot = _read_stable_file_bytes(benchmark_path)
    try:
        snapshot_manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen benchmark snapshot is not valid UTF-8 JSON") from exc
    if not isinstance(snapshot_manifest, dict) or snapshot_manifest != benchmark_manifest:
        raise ValueError("frozen benchmark pathname no longer matches the validated manifest")
    validated_identity = validate_frozen_benchmark_manifest(snapshot_manifest)
    expected_identity = {
        "basename": os.path.basename(os.path.realpath(benchmark_path)),
        "file_sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        **validated_identity,
    }
    for spec, evidences in evidence_by_spec.items():
        for evidence in evidences:
            selection = evidence.experiment_provenance.get("selection")
            observed = selection.get("benchmark_manifest") if isinstance(selection, dict) else None
            if observed != expected_identity:
                raise ValueError(
                    "evaluation provenance is bound to another benchmark artifact: "
                    f"method={spec.method}, seed={spec.seed}, task={spec.task}; "
                    f"expected={expected_identity}, observed={observed}"
                )
    _assert_path_matches_snapshot(benchmark_path, snapshot)


@contextlib.contextmanager
def _bound_representation_environment(
    prompting: Mapping[str, object],
) -> Iterable[None]:
    field_to_environment = {
        "clasp_tsp_prompt_knn": "CLASP_TSP_PROMPT_KNN",
        "clasp_tsp_prompt_knn_distances": "CLASP_TSP_PROMPT_KNN_DISTANCES",
        "clasp_tsp_input_order": "CLASP_TSP_INPUT_ORDER",
        "clasp_tsp_relabel_order": "CLASP_TSP_RELABEL_ORDER",
        "clasp_tsp_route_canonicalize": "CLASP_TSP_ROUTE_CANONICALIZE",
        "clasp_tsp_route_only_output": "CLASP_TSP_ROUTE_ONLY_OUTPUT",
        "clasp_tsp_label_tta_schedule": "CLASP_TSP_LABEL_TTA_SCHEDULE",
        "clasp_mis_relabel_order": "CLASP_MIS_RELABEL_ORDER",
        "clasp_mis_jiang_response_output": "CLASP_MIS_JIANG_RESPONSE_OUTPUT",
        "clasp_mis_prompt_node_degrees": "CLASP_MIS_PROMPT_NODE_DEGREES",
        "clasp_mis_prompt_low_degree_k": "CLASP_MIS_PROMPT_LOW_DEGREE_K",
    }
    previous = {
        environment_name: os.environ.get(environment_name)
        for environment_name in field_to_environment.values()
    }
    try:
        for field, environment_name in field_to_environment.items():
            if field not in prompting:
                raise ValueError(f"experiment prompting lacks representation field {field}")
            os.environ[environment_name] = str(prompting[field])
        yield
    finally:
        for environment_name, value in previous.items():
            if value is None:
                os.environ.pop(environment_name, None)
            else:
                os.environ[environment_name] = value


def _load_frozen_source_rows(
    benchmark_manifest: Mapping[str, object],
    *,
    benchmark_path: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    test_sets = benchmark_manifest.get("test_sets")
    if not isinstance(test_sets, dict):
        raise ValueError("frozen benchmark has no source-row manifests")
    rows_by_task: dict[str, list[dict[str, object]]] = {}
    for task, raw_task_manifest in sorted(test_sets.items()):
        if not isinstance(raw_task_manifest, dict):
            raise ValueError(f"frozen benchmark task {task} source manifest is invalid")
        source_locator = raw_task_manifest.get("source_locator")
        files = raw_task_manifest.get("files")
        if not isinstance(source_locator, str) or not source_locator or not isinstance(files, list):
            raise ValueError(f"frozen benchmark task {task} has no exact source locator")
        if os.path.isabs(source_locator):
            resolved_locator = os.path.realpath(source_locator)
        else:
            if benchmark_path is None:
                raise ValueError(
                    "relative frozen source locator requires the benchmark manifest path"
                )
            benchmark_root = os.path.dirname(os.path.realpath(os.path.abspath(benchmark_path)))
            resolved_locator = os.path.realpath(os.path.join(benchmark_root, source_locator))
            if os.path.commonpath([benchmark_root, resolved_locator]) != benchmark_root:
                raise ValueError("frozen benchmark source locator escaped its bundle")
        if os.path.isfile(resolved_locator):
            source_root = os.path.dirname(resolved_locator)
        elif os.path.isdir(resolved_locator):
            source_root = resolved_locator
        else:
            raise ValueError(f"frozen benchmark source locator is unavailable: {resolved_locator}")
        task_rows: list[dict[str, object]] = []
        for index, file_identity in enumerate(files):
            if not isinstance(file_identity, dict):
                raise ValueError(f"frozen benchmark task {task} file {index} is invalid")
            relative_path = str(file_identity.get("relative_path") or "")
            source_path = os.path.realpath(os.path.join(source_root, relative_path))
            if os.path.commonpath([source_root, source_path]) != source_root:
                raise ValueError(f"frozen benchmark task {task} has an unsafe source path")
            raw_bytes, snapshot = _read_stable_file_bytes(source_path)
            if snapshot.sha256 != file_identity.get(
                "sha256"
            ) or snapshot.size_bytes != file_identity.get("size_bytes"):
                raise ValueError(f"frozen benchmark source file identity mismatch: {source_path}")
            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"frozen source file is not valid UTF-8 JSON: {source_path}"
                ) from exc
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise ValueError(f"frozen source file is not a JSON row array: {source_path}")
            if len(payload) != file_identity.get("rows"):
                raise ValueError(f"frozen source file row count mismatch: {source_path}")
            task_rows.extend(payload)
            _assert_path_matches_snapshot(source_path, snapshot)
        if len(task_rows) != raw_task_manifest.get("rows"):
            raise ValueError(f"frozen benchmark task {task} merged source row count mismatch")
        rows_by_task[str(task)] = task_rows
    return rows_by_task


def _validate_frozen_source_row_closure(
    *,
    specs: Sequence[RunSpec],
    by_method_seed: Mapping[tuple[str, int], Mapping[RecordKey, Record]],
    evidence_by_spec: Mapping[RunSpec, Sequence[ArtifactEvidence]],
    benchmark_manifest: Mapping[str, object],
    benchmark_path: str | None = None,
) -> None:
    source_rows = _load_frozen_source_rows(
        benchmark_manifest,
        benchmark_path=benchmark_path,
    )
    for spec in specs:
        if spec.task is None:
            raise ValueError("confirmation source-row closure requires task-specific runs")
        evidences = evidence_by_spec.get(spec, [])
        if len(evidences) != 1:
            raise ValueError("confirmation source-row closure requires one artifact per run")
        prompting = evidences[0].experiment_provenance.get("prompting")
        if not isinstance(prompting, dict):
            raise ValueError("confirmation artifact has no prompting transformation contract")
        records = by_method_seed.get((spec.method, spec.seed), {})
        selected_records = [record for key, record in records.items() if key[0] == spec.task]
        with _bound_representation_environment(prompting):
            tta_schedule = tsp_label_tta_schedule_from_env()
            for record in selected_records:
                locator = record.get("source_row_locator")
                if not isinstance(locator, dict):
                    raise ValueError("confirmation record has no source row locator")
                merged_index = locator.get("merged_dataset_index")
                task_rows = source_rows.get(spec.task, [])
                if (
                    isinstance(merged_index, bool)
                    or not isinstance(merged_index, int)
                    or merged_index < 0
                    or merged_index >= len(task_rows)
                ):
                    raise ValueError("confirmation source row locator is out of range")
                source_row = task_rows[merged_index]
                expected_hashes = {
                    "semantic_instance_sha256": semantic_instance_sha256(
                        spec.task,
                        source_row,
                        require_structured=True,
                    ),
                    "benchmark_row_sha256": benchmark_row_sha256(source_row),
                    "reference_output_sha256": reference_output_sha256(source_row.get("output")),
                }
                for field, expected in expected_hashes.items():
                    if record.get(field) != expected:
                        raise ValueError(
                            f"confirmation record {field} disagrees with frozen source row"
                        )
                raw_instance = structured_instance(source_row)
                transformed = transform_example_for_task(
                    spec.task,
                    str(source_row.get("instruction", "")),
                    str(source_row.get("input", "")),
                    str(source_row.get("output", "")),
                    raw_instance,
                )
                expected_transformed = {
                    "instruction": str(transformed["instruction"]),
                    "input_problem": str(transformed["input"]),
                    "ground_truth_output": str(transformed["output"]),
                    "instance_json": transformed["instance_json"],
                }
                for field, expected in expected_transformed.items():
                    if record.get(field) != expected:
                        raise ValueError(
                            f"confirmation transformed field {field} disagrees with frozen source row"
                        )
                transformed_identity = {
                    "instruction": expected_transformed["instruction"],
                    "input": expected_transformed["input_problem"],
                    "output": expected_transformed["ground_truth_output"],
                    "instance_json": expected_transformed["instance_json"],
                }
                expected_transformed_sha = hashlib.sha256(
                    _canonical_json_bytes(transformed_identity)
                ).hexdigest()
                if record.get("transformed_example_sha256") != expected_transformed_sha:
                    raise ValueError(
                        "confirmation transformed_example_sha256 disagrees with frozen source row"
                    )
                candidates = record.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError("confirmation record has no candidates")
                if tta_schedule and spec.task == "tsp":
                    if len(tta_schedule) != len(candidates):
                        raise ValueError("TSP label-TTA schedule/candidate count mismatch")
                    raw_record = record.get("raw_record")
                    raw_candidates = (
                        raw_record.get("candidates") if isinstance(raw_record, dict) else None
                    )
                    for candidate_index, candidate in enumerate(candidates):
                        if not isinstance(candidate, dict):
                            raise ValueError("TSP candidate is not an object")
                        context_candidate = candidate
                        if (
                            candidate.get("verification_context") is None
                            and isinstance(raw_candidates, list)
                            and candidate_index < len(raw_candidates)
                            and isinstance(raw_candidates[candidate_index], dict)
                        ):
                            context_candidate = raw_candidates[candidate_index]
                        variant = context_candidate.get("tsp_label_tta_variant")
                        if variant != tta_schedule[candidate_index]:
                            raise ValueError("TSP candidate label-TTA variant changed")
                        expected_context_row = transform_tsp_example_for_label_tta_variant(
                            str(source_row.get("instruction", "")),
                            str(source_row.get("input", "")),
                            str(source_row.get("output", "")),
                            raw_instance,
                            variant=str(variant),
                        )
                        expected_context = {
                            "instruction": str(expected_context_row["instruction"]),
                            "input_problem": str(expected_context_row["input"]),
                            "ground_truth_output": str(expected_context_row["output"]),
                            "instance_json": expected_context_row["instance_json"],
                        }
                        if context_candidate.get("verification_context") != expected_context:
                            raise ValueError("TSP verification context disagrees with frozen row")


def _validate_confirmation_comparison_design(
    manifest: Mapping[str, object],
    *,
    baseline: str | None,
    challenger: str | None,
) -> None:
    if not baseline or not challenger:
        raise ValueError("confirmation requires both a matched baseline and challenger")
    if baseline == challenger:
        raise ValueError("confirmation baseline and challenger must be distinct")
    for role, method in (("baseline", baseline), ("challenger", challenger)):
        if method in LEGACY_TRAINING_METHOD_ALIASES:
            raise ValueError(
                f"confirmation {role} uses prohibited legacy alias {method!r}; "
                "use an exact canonical method ID"
            )
        if method not in TRAINING_METHODS:
            raise ValueError(f"confirmation {role} has unsupported method ID {method!r}")
    if "foarl_f638_native" in {baseline, challenger}:
        raise ValueError(
            "foarl_f638_native is an unmatched descriptive lane and is not eligible "
            "for the matched confirmatory comparison; use foarl_f638_matched_b9 only "
            "after shared-design equality is proven"
        )
    comparison_methods = (baseline, challenger)
    foarl_matched_ids = set(FOARL_MATCHED_B9_CONFIRMATION_PAIR)
    if foarl_matched_ids.intersection(comparison_methods) and not (
        is_foarl_matched_b9_confirmation_pair(comparison_methods)
    ):
        raise ValueError(
            "foarl_f638_matched_b9 confirmation requires the exact ordered pair "
            "baseline=foarl_f638_matched_b9 and "
            f"challenger={CLASP_FOARL_MATCHED_B9}"
        )
    if manifest.get("approved_fixed_null_receipt_sha256_by_task") != (
        APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
    ):
        raise ValueError("confirmation campaign does not bind the approved fixed-null receipts")
    frozen_baseline = manifest.get("baseline")
    frozen_challenger = manifest.get("challenger")
    if frozen_baseline != baseline or frozen_challenger != challenger:
        raise ValueError(
            "CLI comparison roles disagree with the frozen campaign baseline/challenger"
        )
    methods = manifest.get("methods")
    if methods != [frozen_baseline, frozen_challenger]:
        raise ValueError(
            "confirmation campaign methods must equal the prespecified ordered "
            "[baseline, challenger] pair"
        )
    if manifest.get("training_seeds") != list(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError(
            "confirmation requires the exact prespecified training seeds "
            f"{list(CONFIRMATION_TRAINING_SEEDS)}"
        )


def _validate_confirmation_task_suite(
    manifest: Mapping[str, object],
    benchmark_manifest: Mapping[str, object],
) -> None:
    expected_tasks = list(PUBLICATION_TASKS)
    if manifest.get("tasks") != expected_tasks:
        raise ValueError("confirmation campaign must freeze the exact five-task publication suite")
    benchmark_tasks = benchmark_manifest.get("tasks")
    benchmark_sets = benchmark_manifest.get("test_sets")
    if (
        benchmark_tasks != expected_tasks
        or not isinstance(benchmark_sets, dict)
        or (list(benchmark_sets) != expected_tasks)
    ):
        raise ValueError("confirmation benchmark must contain the exact ordered five-task suite")


def _validate_campaign_leakage_registry_binding(
    manifest: Mapping[str, object],
    benchmark_manifest: Mapping[str, object],
) -> None:
    validated = validate_frozen_benchmark_manifest(dict(benchmark_manifest))
    if validated.get("benchmark_mode") != "publication":
        raise ValueError("confirmation requires a publication-mode frozen benchmark")
    for campaign_field, identity_field in (
        ("forbidden_source_registry_sha256", "forbidden_source_registry_sha256"),
        (
            "forbidden_source_tree_manifest_sha256",
            "forbidden_source_tree_manifest_sha256",
        ),
    ):
        expected = validated.get(identity_field)
        if not _is_sha256(expected) or manifest.get(campaign_field) != expected:
            raise ValueError(f"campaign {campaign_field} does not match the validated benchmark")


def _validate_confirmation_analysis(
    manifest: Mapping[str, object],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> None:
    analysis = manifest.get("analysis")
    if not isinstance(analysis, dict) or set(analysis) != {
        "bootstrap_repetitions",
        "bootstrap_seed",
        "alpha",
        "primary_outcome",
        "pass_at_k_noninferiority_gate",
        "secondary_metrics",
    }:
        raise ValueError("campaign has no exact frozen confirmation analysis contract")
    repetitions = analysis.get("bootstrap_repetitions")
    seed = analysis.get("bootstrap_seed")
    expected_primary = {
        "metric": "normalized_feasible_gap_success_curve_auc",
        "gap_range": list(PRIMARY_GAP_RANGE),
        "infeasible_score": 0.0,
        "estimand": "paired_seed_mean_challenger_minus_baseline",
        "inference_unit": "paired_training_seed",
        "test": "exact_two_sided_paired_seed_sign_flip",
        "multiplicity": "holm_across_five_tasks",
        "direction": "higher_is_better",
        "decision_rule": "holm_adjusted_p <= alpha and effect_mean > 0",
        "negative_direction_label": "difference_detected_against_clasp",
        "family": CONFIRMATION_OUTCOME_FAMILIES[0],
    }
    expected_noninferiority = {
        "metric": "strict_shared_verifier_pass_at_8",
        "margin": PASS_AT_K_NONINFERIORITY_MARGIN,
        "null_hypothesis": "challenger_minus_baseline <= -0.01",
        "test": "exact_one_sided_paired_seed_sign_flip_at_margin_boundary",
        "multiplicity": "holm_across_five_tasks",
        "observed_mean_requirement": "challenger_minus_baseline >= -0.01",
        "overall_rule": "all_five_tasks_pass",
        "family": CONFIRMATION_NONINFERIORITY_FAMILY,
    }
    expected_secondary = [
        "candidate_feasibility",
        "conditional_verified_gap",
        "lexicographic_win_score",
        "success_at_external_target_gap_when_eligible",
        "generation_and_repair_compute",
    ]
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 10_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or analysis.get("alpha") != CONFIRMATION_ALPHA
        or analysis.get("primary_outcome") != expected_primary
        or analysis.get("pass_at_k_noninferiority_gate") != expected_noninferiority
        or analysis.get("secondary_metrics") != expected_secondary
    ):
        raise ValueError("campaign confirmation analysis contract is invalid")
    if repetitions != bootstrap_repetitions or seed != bootstrap_seed:
        raise ValueError("CLI bootstrap settings disagree with the frozen campaign")
    sign_assignments = 1 << len(CONFIRMATION_TRAINING_SEEDS)
    minimum_two_sided_p = 2.0 / sign_assignments
    if len(PUBLICATION_TASKS) * minimum_two_sided_p > CONFIRMATION_ALPHA:
        raise ValueError("confirmation seed design cannot resolve its primary Holm family")


def _mask_method_specific_training_fields(design: dict[str, object]) -> None:
    for dotted_path in METHOD_SPECIFIC_TRAINING_ALLOWLIST:
        parts = dotted_path.split(".")
        cursor: object = design
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"training design lacks allowlisted method field {dotted_path}")
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ValueError(f"training design lacks allowlisted method field {dotted_path}")
        cursor[parts[-1]] = {"method_specific_field": dotted_path}


def _shared_training_design_sha256(contract: Mapping[str, object]) -> str:
    seeds = contract.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("training contract has no seed design")
    design: dict[str, object] = {
        "schema_version": 1,
        "sft_seed": seeds.get("sft"),
        "data_lineage": json.loads(json.dumps(contract.get("data_lineage"))),
        "fingerprints": json.loads(json.dumps(contract.get("fingerprints"))),
        "generation": json.loads(json.dumps(contract.get("generation"))),
        "training_config": json.loads(json.dumps(contract.get("training_config"))),
        "tracked_environment": json.loads(json.dumps(contract.get("tracked_environment"))),
    }
    tracked_environment = design.get("tracked_environment")
    if isinstance(tracked_environment, dict):
        # The canonical SET-NULL receipt path is machine-local method evidence,
        # not a shared optimization-design field.  It may be removed here only
        # because _validate_training_contract has already required its immutable
        # path-free identity, and _validate_matched_training_design binds that
        # identity per task across every seed.
        tracked_environment.pop("CLASP_FIXED_NULL_RECEIPT_PATH", None)
    _mask_method_specific_training_fields(design)
    return hashlib.sha256(_canonical_json_bytes(design)).hexdigest()


def _validate_matched_training_design(
    manifest: Mapping[str, object],
    *,
    contracts_by_group: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    methods: Sequence[str],
    tasks: Sequence[str],
    seeds: Sequence[int],
) -> None:
    policy = manifest.get("training_design")
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "fixed_horizon",
        "method_specific_allowlist",
        "shared_design_sha256_by_task_seed",
        "fixed_null_receipt_sha256_by_task",
    }:
        raise ValueError("campaign has no exact matched training-design contract")
    horizon = policy.get("fixed_horizon")
    foarl_comparison = is_foarl_matched_b9_confirmation_pair(methods)
    expected_allowlist = (
        list(FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST)
        if foarl_comparison
        else list(METHOD_SPECIFIC_TRAINING_ALLOWLIST)
    )
    if (
        policy.get("schema_version") != 1
        or not isinstance(horizon, dict)
        or set(horizon) != {"max_steps", "world_size"}
        or any(
            isinstance(horizon.get(field), bool)
            or not isinstance(horizon.get(field), int)
            or int(horizon[field]) <= 0
            for field in ("max_steps", "world_size")
        )
        or policy.get("method_specific_allowlist") != expected_allowlist
    ):
        raise ValueError("campaign matched training-design policy is invalid")
    declared_hashes = policy.get("shared_design_sha256_by_task_seed")
    expected_group_ids = {f"{task}/seed{seed}" for task in tasks for seed in seeds}
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != expected_group_ids:
        raise ValueError("campaign shared training-design hashes do not cover the exact grid")
    declared_fixed_null = policy.get("fixed_null_receipt_sha256_by_task")
    expected_fixed_null_tasks = set(tasks) if "set_null" in methods else set()
    if (
        not isinstance(declared_fixed_null, dict)
        or set(declared_fixed_null) != expected_fixed_null_tasks
        or any(not _is_sha256(value) for value in declared_fixed_null.values())
        or any(
            declared_fixed_null.get(task) != APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK[task]
            for task in expected_fixed_null_tasks
        )
    ):
        raise ValueError("campaign fixed-null receipt preregistration is invalid")
    for task in tasks:
        for seed in seeds:
            group = contracts_by_group.get((task, seed))
            if not isinstance(group, dict) or set(group) != set(methods):
                raise ValueError(
                    f"matched training design lacks the exact method pair for {task}/seed{seed}"
                )
            observed_hashes: set[str] = set()
            for method in methods:
                contract = group[method]
                if contract.get("fixed_horizon") != horizon:
                    raise ValueError(
                        f"unequal training horizon/resource budget at {task}/{method}/seed{seed}"
                    )
                observed_hashes.add(
                    foarl_matched_b9_shared_design_sha256(contract)
                    if foarl_comparison
                    else _shared_training_design_sha256(contract)
                )
                if method == "set_null":
                    fixed_null_receipt = contract.get("fixed_null_receipt")
                    observed_receipt_sha256 = (
                        fixed_null_receipt.get("receipt_sha256")
                        if isinstance(fixed_null_receipt, dict)
                        else None
                    )
                    if declared_fixed_null.get(task) != observed_receipt_sha256:
                        raise ValueError(
                            f"set_null receipt disagrees with campaign preregistration at "
                            f"{task}/seed{seed}"
                        )
            if len(observed_hashes) != 1:
                raise ValueError(
                    f"baseline/challenger shared training design differs at {task}/seed{seed}"
                )
            observed_hash = next(iter(observed_hashes))
            group_id = f"{task}/seed{seed}"
            if declared_hashes.get(group_id) != observed_hash:
                raise ValueError(f"campaign shared training-design hash mismatch at {group_id}")


def _validate_campaign_training_lineage(
    *,
    contracts_by_group: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    benchmark_manifest: Mapping[str, object],
) -> None:
    expected_registry = benchmark_manifest.get("forbidden_source_registry_identity")
    if not isinstance(expected_registry, dict):
        raise ValueError("publication benchmark has no training-data registry identity")
    for (task, seed), methods in contracts_by_group.items():
        for method, contract in methods.items():
            lineage = contract.get("data_lineage")
            observed_registry = (
                lineage.get("forbidden_source_registry") if isinstance(lineage, dict) else None
            )
            if observed_registry != expected_registry:
                raise ValueError(
                    "training data lineage disagrees with the benchmark leakage registry: "
                    f"{task}/{method}/seed{seed}"
                )


def _validate_campaign_training_authorization(
    manifest: Mapping[str, object],
    *,
    campaign_id: str,
    contracts_by_group: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
) -> None:
    """Bind every training run to the one predeclared campaign/stage/policy."""

    authorization = manifest.get("training_authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "campaign_id",
        "advancement_policy_sha256",
        "advancement_stage_id",
        "advancement_stage_index",
        "prior_stage_receipt_chain",
        "prior_stage_receipt_chain_sha256",
    }:
        raise ValueError("campaign has no exact training authorization")
    stage_index = authorization.get("advancement_stage_index")
    prior_chain = authorization.get("prior_stage_receipt_chain")
    if (
        authorization.get("campaign_id") != campaign_id
        or not _is_sha256(authorization.get("advancement_policy_sha256"))
        or not isinstance(authorization.get("advancement_stage_id"), str)
        or not authorization.get("advancement_stage_id")
        or isinstance(stage_index, bool)
        or not isinstance(stage_index, int)
        or stage_index < 0
        or not isinstance(prior_chain, list)
        or len(prior_chain) != stage_index
        or authorization.get("prior_stage_receipt_chain_sha256")
        != hashlib.sha256(_canonical_json_bytes(prior_chain)).hexdigest()
    ):
        raise ValueError("campaign training authorization is invalid")
    prior_stage_ids: list[str] = []
    prior_receipt_hashes: list[str] = []
    for index, entry in enumerate(prior_chain):
        label = f"campaign training authorization prior receipt {index}"
        if not isinstance(entry, dict) or set(entry) != PRIOR_STAGE_RECEIPT_IDENTITY_KEYS:
            raise ValueError(f"{label} has an invalid identity schema")
        stage_id = entry.get("stage_id")
        if (
            entry.get("campaign_id") != campaign_id
            or not isinstance(stage_id, str)
            or not stage_id
            or any(
                not _is_sha256(entry.get(field))
                for field in (
                    "policy_sha256",
                    "receipt_sha256",
                    "receipt_file_sha256",
                )
            )
        ):
            raise ValueError(f"{label} has invalid identity fields")
        prior_stage_ids.append(stage_id)
        prior_receipt_hashes.append(str(entry["receipt_sha256"]))
    if (
        len(prior_stage_ids) != len(set(prior_stage_ids))
        or len(prior_receipt_hashes) != len(set(prior_receipt_hashes))
        or authorization["advancement_stage_id"] in prior_stage_ids
    ):
        raise ValueError("campaign training authorization prior receipt chain is not unique")
    expected_config = {
        "campaign_id": campaign_id,
        "advancement_policy_sha256": authorization["advancement_policy_sha256"],
        "advancement_stage_id": authorization["advancement_stage_id"],
        "advancement_stage_index": stage_index,
        "advancement_prior_stage_receipt_chain_sha256": authorization[
            "prior_stage_receipt_chain_sha256"
        ],
    }
    for (task, seed), methods in contracts_by_group.items():
        for method, contract in methods.items():
            config = contract.get("training_config")
            if not isinstance(config, dict) or any(
                config.get(field) != expected for field, expected in expected_config.items()
            ):
                raise ValueError(
                    "training contract belongs to another campaign/stage/policy: "
                    f"{task}/{method}/seed{seed}"
                )


def _validate_campaign_model_initialization(
    manifest: Mapping[str, object],
    *,
    contracts_by_group: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
) -> None:
    """Enforce the pre-frozen base model and task-specialist SFT starts."""

    registry = manifest.get("model_initialization_registry")
    expected_registry_keys = {
        "schema_version",
        "protocol",
        "status",
        "registry_sha256",
        "model_family",
        "base_model_tree_sha256",
        "forbidden_source_registry_sha256",
        "forbidden_source_tree_manifest_sha256",
        "specialist_sft_by_task",
    }
    if not isinstance(registry, dict) or set(registry) != expected_registry_keys:
        raise ValueError("campaign has no exact model-initialization registry")
    specialists = registry.get("specialist_sft_by_task")
    if (
        registry.get("schema_version") != 2
        or registry.get("protocol") != "clasp-model-initialization-registry-v2"
        or registry.get("status") != "frozen_before_training"
        or registry.get("model_family") != "Qwen2.5-7B"
        or any(
            not _is_sha256(registry.get(field))
            for field in (
                "registry_sha256",
                "base_model_tree_sha256",
                "forbidden_source_registry_sha256",
                "forbidden_source_tree_manifest_sha256",
            )
        )
        or not isinstance(specialists, dict)
        or set(specialists) != set(PUBLICATION_TASKS)
    ):
        raise ValueError("campaign model-initialization registry is invalid")
    for task in PUBLICATION_TASKS:
        specialist = specialists.get(task)
        if not isinstance(specialist, dict) or set(specialist) != {
            "initial_model_tree_sha256",
            "sft_train_source_identity_sha256",
            "sft_seed",
            "matched_sft_contract_sha256",
            "checkpoint_fidelity",
            "reload_equivalence",
        }:
            raise ValueError(f"campaign specialist initialization is invalid for {task}")
        if (
            not _is_sha256(specialist.get("initial_model_tree_sha256"))
            or not _is_sha256(specialist.get("sft_train_source_identity_sha256"))
            or specialist.get("sft_seed") != 42
            or specialist.get("matched_sft_contract_sha256") != MATCHED_SFT_CONTRACT_SHA256
        ):
            raise ValueError(f"campaign specialist initialization is invalid for {task}")
        try:
            validate_publication_fidelity_identity(specialist.get("checkpoint_fidelity"))
        except ValueError as exc:
            raise ValueError(
                f"campaign specialist checkpoint fidelity is invalid for {task}: {exc}"
            ) from exc
        try:
            reload_equivalence = validate_reload_equivalence_identity(
                specialist.get("reload_equivalence")
            )
        except ValueError as exc:
            raise ValueError(
                f"campaign specialist reload equivalence is invalid for {task}: {exc}"
            ) from exc
        if (
            reload_equivalence["checkpoint_fidelity_identity_sha256"]
            != specialist["checkpoint_fidelity"]["identity_sha256"]
        ):
            raise ValueError(
                f"campaign specialist reload equivalence targets another checkpoint for {task}"
            )
    for (task, seed), methods in contracts_by_group.items():
        for method, contract in methods.items():
            fingerprints = contract.get("fingerprints")
            seeds = contract.get("seeds")
            lineage = contract.get("data_lineage")
            if (
                contract.get("model_initialization_registry") != registry
                or not isinstance(fingerprints, dict)
                or not isinstance(seeds, dict)
                or not isinstance(lineage, dict)
            ):
                raise ValueError(
                    "training contract uses an unapproved base/SFT initialization: "
                    f"{task}/{method}/seed{seed}"
                )
            _validate_model_initialization_identity(
                registry,
                task=task,
                seeds=seeds,
                fingerprints=fingerprints,
                data_lineage=lineage,
                label=f"campaign {task}/{method}/seed{seed}",
            )


def _selector_receipt_file_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _validate_selection_development_policy(
    frozen_policy: Mapping[str, object],
    *,
    tasks: Sequence[str],
    expected_forbidden_source_registry_identity: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    """Validate the selector's exact frozen development-corpus contracts."""

    registry = frozen_policy.get("forbidden_source_registry")
    if not isinstance(registry, dict):
        raise ValueError("campaign checkpoint-selection policy has no forbidden-source registry")
    validate_forbidden_source_registry(
        registry,
        expected_tasks=list(tasks),
    )
    registry_identity = forbidden_source_registry_identity(registry)
    if (
        expected_forbidden_source_registry_identity is not None
        and registry_identity != expected_forbidden_source_registry_identity
    ):
        raise ValueError(
            "checkpoint-selection forbidden-source registry disagrees with the "
            "confirmation benchmark"
        )
    sources_by_task = registry_identity.get("sources_by_task_label")
    if not isinstance(sources_by_task, dict):
        raise ValueError("campaign checkpoint-selection registry has no source projections")

    raw_contracts = frozen_policy.get("development_evaluation_by_task")
    if not isinstance(raw_contracts, dict) or set(raw_contracts) != set(tasks):
        raise ValueError("campaign checkpoint-selection policy has no exact development-data grid")
    validated: dict[str, dict[str, object]] = {}
    for task in tasks:
        contract = raw_contracts.get(task)
        if not isinstance(contract, dict) or set(contract) != DEVELOPMENT_EVALUATION_FIELDS:
            raise ValueError(
                f"campaign checkpoint-selection development contract is invalid for {task}"
            )
        _validate_self_hash(
            contract,
            "development_evaluation_sha256",
            label=f"campaign checkpoint-selection development contract for {task}",
        )
        maximum = contract.get("max_eval_samples_per_task")
        if (
            not _is_sha256(contract.get("data_manifest_sha256"))
            or not _is_sha256(contract.get("protocol_fingerprint"))
            or not _is_sha256(contract.get("strict_verifier_manifest_sha256"))
            or contract.get("forbidden_source_label") != "historical-eval"
            or not _is_sha256(contract.get("forbidden_source_identity_sha256"))
            or contract.get("split") != "eval"
            or contract.get("eval_start_index") != 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
            or contract.get("eval_samples_per_task") != {}
            or contract.get("size_bucket") != "all"
            or contract.get("min_instance_size") != 0
            or contract.get("max_instance_size") != 0
            or contract.get("prefer_subset_files") is not False
            or contract.get("expected_instances") != maximum
        ):
            raise ValueError(
                "campaign checkpoint-selection development contract is not the "
                f"full frozen slice for {task}"
            )

        evaluation_files = contract.get("evaluation_data_files")
        expected_file_keys = {"task", "split", "path", "size", "sha256"}
        if (
            not isinstance(evaluation_files, list)
            or not evaluation_files
            or any(
                not isinstance(item, dict)
                or set(item) != expected_file_keys
                or item.get("task") != task
                or item.get("split") != contract["split"]
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or isinstance(item.get("size"), bool)
                or not isinstance(item.get("size"), int)
                or int(item["size"]) <= 0
                or not _is_sha256(item.get("sha256"))
                for item in evaluation_files
            )
        ):
            raise ValueError(
                f"campaign checkpoint-selection development files are invalid for {task}"
            )
        paths = [str(item["path"]) for item in evaluation_files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError(
                f"campaign checkpoint-selection development files are not sorted/unique for {task}"
            )
        if (
            contract["data_manifest_sha256"]
            != hashlib.sha256(_canonical_json_bytes(evaluation_files)).hexdigest()
        ):
            raise ValueError(
                f"campaign checkpoint-selection development manifest is invalid for {task}"
            )

        task_sources = sources_by_task.get(task)
        historical = task_sources.get("historical-eval") if isinstance(task_sources, dict) else None
        if not isinstance(historical, dict) or contract[
            "forbidden_source_identity_sha256"
        ] != historical.get("source_identity_sha256"):
            raise ValueError(
                "campaign checkpoint-selection development data is not registered "
                f"historical-eval for {task}"
            )
        registered_files = historical.get("files")
        registered_projection = (
            [
                {
                    "task": task,
                    "split": contract["split"],
                    "path": item.get("relative_path"),
                    "size": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in registered_files
                if isinstance(item, dict)
            ]
            if isinstance(registered_files, list)
            else []
        )
        if evaluation_files != registered_projection:
            raise ValueError(
                "campaign checkpoint-selection development bytes differ from "
                f"registered historical-eval for {task}"
            )
        validated[task] = contract
    return validated


def _validate_selection_audit(
    *,
    group_id: str,
    selected: Mapping[str, object],
    audit: object,
    expected_steps: Sequence[int],
    pass_band: float,
    candidate_budget: int,
) -> list[dict[str, object]]:
    if not isinstance(audit, dict) or set(audit) != {
        "best_pass_at_k",
        "eligibility_threshold",
        "eligible_checkpoint_steps",
        "candidates",
    }:
        raise ValueError(f"selection receipt {group_id} has an invalid audit schema")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"selection receipt {group_id} has no checkpoint audit")
    normalized: list[dict[str, object]] = []
    seen_steps: set[int] = set()
    candidate_fields = {
        "checkpoint_step",
        "candidate_budget",
        "instances",
        "feasible_instances",
        "pass_at_k",
        "conditional_gap_n",
        "conditional_gap_mean",
        "grpo_beta",
        "fixed_kl_reference",
        "exact_peft_kl_runtime",
        "exact_peft_kl_post_step",
        *CHECKPOINT_SELECTION_FINGERPRINT_FIELDS,
        "disposition",
    }
    for index, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, dict) or set(raw_candidate) != candidate_fields:
            raise ValueError(
                f"selection receipt {group_id} candidate {index} has an invalid schema"
            )
        step = raw_candidate.get("checkpoint_step")
        pass_at_k = _finite_float(raw_candidate.get("pass_at_k"))
        raw_gap = raw_candidate.get("conditional_gap_mean")
        gap = _finite_float(raw_gap)
        instances = raw_candidate.get("instances")
        feasible_instances = raw_candidate.get("feasible_instances")
        gap_n = raw_candidate.get("conditional_gap_n")
        try:
            grpo_beta, fixed_kl_reference, exact_runtime, exact_post_step = (
                _validate_exact_post_step_projection(
                    raw_candidate.get("grpo_beta"),
                    raw_candidate.get("fixed_kl_reference"),
                    raw_candidate.get("exact_peft_kl_runtime"),
                    raw_candidate.get("exact_peft_kl_post_step"),
                    label=f"selection receipt {group_id} candidate {index}",
                    expected_run_contract_sha256=raw_candidate.get("run_contract_sha256"),
                    expected_checkpoint_step=raw_candidate.get("checkpoint_step"),
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"selection receipt {group_id} candidate {index} is invalid: {exc}"
            ) from exc
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step in seen_steps
            or pass_at_k is None
            or not 0.0 <= pass_at_k <= 1.0
            or raw_candidate.get("candidate_budget") != candidate_budget
            or isinstance(instances, bool)
            or not isinstance(instances, int)
            or instances <= 0
            or isinstance(feasible_instances, bool)
            or not isinstance(feasible_instances, int)
            or not 0 <= feasible_instances <= instances
            or gap_n != feasible_instances
            or not _numbers_equal(pass_at_k, feasible_instances / instances)
            or (feasible_instances == 0 and raw_gap is not None)
            or (feasible_instances > 0 and (gap is None or gap < 0.0))
            or any(
                not _is_sha256(raw_candidate.get(field))
                for field in CHECKPOINT_SELECTION_FINGERPRINT_FIELDS
            )
        ):
            raise ValueError(f"selection receipt {group_id} candidate {index} is invalid")
        seen_steps.add(step)
        normalized.append(
            {
                **raw_candidate,
                "checkpoint_step": step,
                "instances": instances,
                "feasible_instances": feasible_instances,
                "conditional_gap_n": gap_n,
                "pass_at_k": pass_at_k,
                "conditional_gap_mean": gap,
                "grpo_beta": grpo_beta,
                "fixed_kl_reference": fixed_kl_reference,
                "exact_peft_kl_runtime": exact_runtime,
                "exact_peft_kl_post_step": exact_post_step,
            }
        )
    if sorted(seen_steps) != list(expected_steps):
        raise ValueError(f"selection receipt {group_id} omits a scheduled checkpoint")
    for field in (
        "protocol_fingerprint",
        "run_contract_sha256",
        "data_manifest_sha256",
        "strict_verifier_manifest_sha256",
        "candidate_budget",
        "instances",
        "grpo_beta",
    ):
        if len({row[field] for row in normalized}) != 1:
            raise ValueError(
                f"selection receipt {group_id} mixes development {field} across checkpoints"
            )
    if len({_canonical_json_bytes(row["fixed_kl_reference"]) for row in normalized}) != 1:
        raise ValueError(f"selection receipt {group_id} mixes fixed-KL identity across checkpoints")
    if len({_canonical_json_bytes(row["exact_peft_kl_runtime"]) for row in normalized}) != 1:
        raise ValueError(
            f"selection receipt {group_id} mixes exact PEFT KL runtime across checkpoints"
        )
    positive_post = [
        row["exact_peft_kl_post_step"]
        for row in normalized
        if row["exact_peft_kl_post_step"] is not None
    ]
    if positive_post and (
        len({row["run_invariant_sha256"] for row in positive_post}) != 1
        or len({row["source_reference_sha256"] for row in positive_post}) != 1
        or len({_canonical_json_bytes(row["root_final"]) for row in positive_post}) != 1
    ):
        raise ValueError(f"selection receipt {group_id} mixes post-step run lineage")
    for field in (
        "evaluation_fingerprint",
        "checkpoint_artifact_sha256",
        "checkpoint_fidelity_identity_sha256",
    ):
        if len({row[field] for row in normalized}) != len(normalized):
            raise ValueError(f"selection receipt {group_id} repeats {field} across checkpoints")
    best_pass = max(float(row["pass_at_k"]) for row in normalized)
    threshold = best_pass - pass_band
    eligible = [row for row in normalized if float(row["pass_at_k"]) >= threshold - 1e-15]
    if any(row["conditional_gap_mean"] is None for row in eligible):
        raise ValueError(f"selection receipt {group_id} has undefined eligible gap")
    chosen = min(
        eligible,
        key=lambda row: (
            float(row["conditional_gap_mean"]),
            int(row["checkpoint_step"]),
        ),
    )
    eligible_steps = sorted(int(row["checkpoint_step"]) for row in eligible)
    if (
        not _numbers_equal(audit.get("best_pass_at_k"), best_pass)
        or not _numbers_equal(audit.get("eligibility_threshold"), threshold)
        or audit.get("eligible_checkpoint_steps") != eligible_steps
    ):
        raise ValueError(f"selection receipt {group_id} audit summary is inconsistent")
    for candidate in normalized:
        step = int(candidate["checkpoint_step"])
        if step not in eligible_steps:
            expected_disposition = "outside_pass_at_k_band"
        elif candidate is chosen:
            expected_disposition = "selected"
        else:
            expected_disposition = "eligible_but_higher_gap_or_later_exact_tie"
        if candidate.get("disposition") != expected_disposition:
            raise ValueError(f"selection receipt {group_id} candidate disposition is inconsistent")
    chosen_projection = {
        field: chosen[field]
        for field in (
            "checkpoint_step",
            "candidate_budget",
            "instances",
            "feasible_instances",
            "pass_at_k",
            "conditional_gap_n",
            "conditional_gap_mean",
            "grpo_beta",
            "fixed_kl_reference",
            "exact_peft_kl_runtime",
            "exact_peft_kl_post_step",
            *CHECKPOINT_SELECTION_FINGERPRINT_FIELDS,
        )
    }
    if any(selected.get(field) != value for field, value in chosen_projection.items()):
        raise ValueError(f"selection receipt {group_id} selected row violates the frozen rule")
    return normalized


def _validate_campaign_selection_receipt(
    manifest: Mapping[str, object],
    *,
    expected_grid: Sequence[tuple[str, int, str]],
    campaign_id: str,
    expected_candidate_budget: int,
    expected_forbidden_source_registry_identity: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    frozen = manifest.get("checkpoint_selection_receipt")
    if not isinstance(frozen, dict) or set(frozen) != {"identity", "payload"}:
        raise ValueError("campaign has no embedded full-grid checkpoint-selection receipt")
    identity = frozen.get("identity")
    payload = frozen.get("payload")
    if not isinstance(identity, dict) or not isinstance(payload, dict):
        raise ValueError("campaign checkpoint-selection receipt is incomplete")
    expected_payload_keys = {
        "schema_version",
        "selector",
        "generated_at_utc",
        "campaign_id",
        "stage_id",
        "decision",
        "evidence_scope",
        "confirmation_evidence_used",
        "policy",
        "evidence",
        "expected_group_count",
        "selected_group_count",
        "selections",
        "errors",
        "side_effects",
        "receipt_sha256",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("campaign checkpoint-selection receipt has an invalid schema")
    _validate_self_hash(payload, "receipt_sha256", label="campaign selection receipt")
    if (
        payload.get("schema_version") != 2
        or payload.get("selector") != "clasp_development_checkpoint_selection"
        or payload.get("decision") != "SELECTED"
        or payload.get("evidence_scope") != "development_only"
        or payload.get("confirmation_evidence_used") is not False
        or payload.get("campaign_id") != campaign_id
        or not isinstance(payload.get("generated_at_utc"), str)
        or not payload.get("generated_at_utc")
        or not str(payload.get("stage_id") or "")
        or payload.get("errors") != []
        or payload.get("side_effects")
        != {
            "checkpoints_modified": False,
            "evaluator_state_modified": False,
            "scheduler_actions": [],
        }
    ):
        raise ValueError("campaign checkpoint-selection receipt is not an accepted decision")
    policy = payload.get("policy")
    evidence = payload.get("evidence")
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {
            "path",
            "file_sha256",
            "policy_sha256",
            "frozen_at_utc",
            "selection_rule",
            "pass_at_k_absolute_band",
            "frozen_policy",
        }
        or not _is_sha256(policy.get("file_sha256"))
        or not _is_sha256(policy.get("policy_sha256"))
        or not isinstance(policy.get("path"), str)
        or not policy.get("path")
        or not isinstance(policy.get("frozen_at_utc"), str)
        or not policy.get("frozen_at_utc")
        or policy.get("selection_rule") != CHECKPOINT_SELECTION_RULE
        or policy.get("pass_at_k_absolute_band") != 0.01
        or not isinstance(evidence, dict)
        or set(evidence) != {"path", "size_bytes", "file_sha256", "evidence_sha256", "record_count"}
        or not _is_sha256(evidence.get("file_sha256"))
        or not _is_sha256(evidence.get("evidence_sha256"))
        or not isinstance(evidence.get("path"), str)
        or not evidence.get("path")
        or isinstance(evidence.get("size_bytes"), bool)
        or not isinstance(evidence.get("size_bytes"), int)
        or int(evidence["size_bytes"]) <= 0
        or isinstance(evidence.get("record_count"), bool)
        or not isinstance(evidence.get("record_count"), int)
        or int(evidence["record_count"]) <= 0
    ):
        raise ValueError("campaign selection receipt policy/evidence binding is invalid")
    frozen_policy = policy.get("frozen_policy")
    expected_policy_keys = {
        "campaign_id",
        "candidate_budget",
        "checkpoint_steps",
        "development_evaluation_by_task",
        "evidence_scope",
        "expected_methods",
        "expected_tasks",
        "forbidden_source_registry",
        "frozen_at_utc",
        "pass_at_k_absolute_band",
        "policy_sha256",
        "require_complete_checkpoint_grid",
        "schema_version",
        "selection_rule",
        "stage_id",
        "status",
        "train_seeds",
    }
    if not isinstance(frozen_policy, dict) or set(frozen_policy) != expected_policy_keys:
        raise ValueError("campaign selection receipt omits its exact frozen policy")
    _validate_self_hash(
        frozen_policy,
        "policy_sha256",
        label="campaign checkpoint-selection policy",
    )
    declared_methods = manifest.get("methods")
    if not isinstance(declared_methods, list):
        raise ValueError("campaign has no ordered method pair for checkpoint selection")
    grid_methods = declared_methods
    grid_seeds = sorted({seed for _method, seed, _task in expected_grid})
    grid_tasks = sorted({task for _method, _seed, task in expected_grid})
    checkpoint_steps = frozen_policy.get("checkpoint_steps")
    if (
        frozen_policy.get("schema_version") != 1
        or frozen_policy.get("status") != "frozen_before_development_results"
        or frozen_policy.get("evidence_scope") != "development_only"
        or frozen_policy.get("require_complete_checkpoint_grid") is not True
        or frozen_policy.get("campaign_id") != campaign_id
        or frozen_policy.get("stage_id") != payload.get("stage_id")
        or frozen_policy.get("expected_methods") != grid_methods
        or frozen_policy.get("expected_tasks") != grid_tasks
        or frozen_policy.get("train_seeds") != grid_seeds
        or frozen_policy.get("candidate_budget") != expected_candidate_budget
        or frozen_policy.get("selection_rule") != CHECKPOINT_SELECTION_RULE
        or frozen_policy.get("pass_at_k_absolute_band") != 0.01
        or policy.get("policy_sha256") != frozen_policy.get("policy_sha256")
        or policy.get("frozen_at_utc") != frozen_policy.get("frozen_at_utc")
        or not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or checkpoint_steps != sorted(checkpoint_steps)
        or len(checkpoint_steps) != len(set(checkpoint_steps))
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in checkpoint_steps
        )
    ):
        raise ValueError("campaign checkpoint-selection frozen policy is invalid")
    development_contracts = _validate_selection_development_policy(
        frozen_policy,
        tasks=grid_tasks,
        expected_forbidden_source_registry_identity=(expected_forbidden_source_registry_identity),
    )
    expected_group_ids = {f"{task}/{method}/seed{seed}" for method, seed, task in expected_grid}
    selections = payload.get("selections")
    if (
        not isinstance(selections, dict)
        or set(selections) != expected_group_ids
        or payload.get("expected_group_count") != len(expected_group_ids)
        or payload.get("selected_group_count") != len(expected_group_ids)
        or evidence.get("record_count") != len(expected_group_ids) * len(checkpoint_steps)
    ):
        raise ValueError("checkpoint-selection receipt does not cover the exact campaign grid")
    per_task_fairness: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    selected_fields = {
        "task",
        "method",
        "train_seed",
        "checkpoint_step",
        "pass_at_k",
        "feasible_instances",
        "conditional_gap_mean",
        "conditional_gap_n",
        "instances",
        "candidate_budget",
        "grpo_beta",
        "fixed_kl_reference",
        "exact_peft_kl_runtime",
        "exact_peft_kl_post_step",
        *CHECKPOINT_SELECTION_FINGERPRINT_FIELDS,
    }
    for group_id, group in selections.items():
        selected = group.get("selected") if isinstance(group, dict) else None
        if (
            not isinstance(group, dict)
            or set(group) != {"status", "selected", "selection_audit", "errors"}
            or group.get("status") != "PASS"
            or group.get("errors") != []
            or not isinstance(group.get("selection_audit"), dict)
            or not isinstance(selected, dict)
            or set(selected) != selected_fields
            or not isinstance(selected.get("task"), str)
            or not isinstance(selected.get("method"), str)
            or isinstance(selected.get("train_seed"), bool)
            or not isinstance(selected.get("train_seed"), int)
            or group_id
            != f"{selected.get('task')}/{selected.get('method')}/seed{selected.get('train_seed')}"
        ):
            raise ValueError("campaign selection receipt has an invalid group decision")
        if (
            selected.get("candidate_budget") != expected_candidate_budget
            or any(
                not _is_sha256(selected.get(field))
                for field in CHECKPOINT_SELECTION_FINGERPRINT_FIELDS
            )
            or isinstance(selected.get("instances"), bool)
            or not isinstance(selected.get("instances"), int)
            or int(selected["instances"]) <= 0
            or isinstance(selected.get("conditional_gap_n"), bool)
            or not isinstance(selected.get("conditional_gap_n"), int)
            or not 0 < int(selected["conditional_gap_n"]) <= int(selected["instances"])
        ):
            raise ValueError("campaign selection receipt selected evidence is invalid")
        selected_beta, selected_fixed_kl, selected_exact_runtime, selected_exact_post = (
            _validate_exact_post_step_projection(
                selected.get("grpo_beta"),
                selected.get("fixed_kl_reference"),
                selected.get("exact_peft_kl_runtime"),
                selected.get("exact_peft_kl_post_step"),
                label=f"campaign selection receipt {group_id}",
                expected_run_contract_sha256=selected.get("run_contract_sha256"),
                expected_checkpoint_step=selected.get("checkpoint_step"),
            )
        )
        if (
            selected_beta != selected.get("grpo_beta")
            or selected_fixed_kl != selected.get("fixed_kl_reference")
            or selected_exact_runtime != selected.get("exact_peft_kl_runtime")
            or selected_exact_post != selected.get("exact_peft_kl_post_step")
        ):
            raise ValueError("campaign selection receipt KL projection is not canonical")
        development_contract = development_contracts.get(str(selected["task"]))
        if (
            development_contract is None
            or selected.get("data_manifest_sha256") != development_contract["data_manifest_sha256"]
            or selected.get("development_evaluation_sha256")
            != development_contract["development_evaluation_sha256"]
            or selected.get("protocol_fingerprint") != development_contract["protocol_fingerprint"]
            or selected.get("strict_verifier_manifest_sha256")
            != development_contract["strict_verifier_manifest_sha256"]
            or selected.get("instances") != development_contract["expected_instances"]
        ):
            raise ValueError(
                "campaign selection receipt selected row is not bound to its "
                "frozen development corpus/slice"
            )
        audit_rows = _validate_selection_audit(
            group_id=group_id,
            selected=selected,
            audit=group.get("selection_audit"),
            expected_steps=checkpoint_steps,
            pass_band=0.01,
            candidate_budget=expected_candidate_budget,
        )
        for audit_row in audit_rows:
            if (
                audit_row.get("protocol_fingerprint")
                != development_contract["protocol_fingerprint"]
                or audit_row.get("strict_verifier_manifest_sha256")
                != development_contract["strict_verifier_manifest_sha256"]
                or audit_row.get("data_manifest_sha256")
                != development_contract["data_manifest_sha256"]
                or audit_row.get("development_evaluation_sha256")
                != development_contract["development_evaluation_sha256"]
                or audit_row.get("instances") != development_contract["expected_instances"]
            ):
                raise ValueError(
                    "checkpoint selection audit row disagrees with the frozen development protocol"
                )
            per_task_fairness[str(selected["task"])].add(
                (
                    audit_row.get("protocol_fingerprint"),
                    audit_row.get("development_evaluation_sha256"),
                    audit_row.get("data_manifest_sha256"),
                    audit_row.get("strict_verifier_manifest_sha256"),
                    audit_row.get("candidate_budget"),
                    audit_row.get("instances"),
                    audit_row.get("grpo_beta"),
                    (
                        audit_row.get("fixed_kl_reference", {}).get("identity_sha256")
                        if isinstance(audit_row.get("fixed_kl_reference"), dict)
                        else None
                    ),
                    (
                        audit_row.get("exact_peft_kl_post_step", {}).get("source_reference_sha256")
                        if isinstance(audit_row.get("exact_peft_kl_post_step"), dict)
                        else None
                    ),
                )
            )
    for task, identities in per_task_fairness.items():
        if len(identities) != 1:
            raise ValueError(
                f"development checkpoint evidence is not comparable across methods/seeds for {task}"
            )
    serialized = _selector_receipt_file_bytes(payload)
    expected_identity = {
        "basename": identity.get("basename"),
        "file_sha256": hashlib.sha256(serialized).hexdigest(),
        "size_bytes": len(serialized),
        "receipt_sha256": payload.get("receipt_sha256"),
        "campaign_id": payload.get("campaign_id"),
        "stage_id": payload.get("stage_id"),
    }
    if (
        identity != expected_identity
        or not isinstance(identity.get("basename"), str)
        or not identity.get("basename")
        or not _is_sha256(identity.get("file_sha256"))
        or (
            isinstance(identity.get("size_bytes"), bool)
            or not isinstance(identity.get("size_bytes"), int)
            or int(identity["size_bytes"]) <= 0
        )
    ):
        raise ValueError("campaign selection receipt identity/payload mismatch")
    return identity, payload


def _campaign_run_key(value: Mapping[str, object]) -> tuple[str, int, str]:
    try:
        raw_seed = value["training_seed"]
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError("training_seed must be an integer")
        return (
            str(value["method"]),
            raw_seed,
            str(value["task"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"campaign run entry has an invalid identity: {value}") from exc


def _confirmation_attempt_id(manifest: Mapping[str, object], entry: Mapping[str, object]) -> str:
    material = {
        "protocol": "clasp-confirmation-attempt-v2",
        "campaign_id": manifest.get("campaign_id"),
        "method": entry.get("method"),
        "training_seed": entry.get("training_seed"),
        "task": entry.get("task"),
        "training_run_contract_sha256": entry.get("training_run_contract_sha256"),
        "checkpoint_step": entry.get("checkpoint_step"),
        "checkpoint_artifact_sha256": entry.get("checkpoint_artifact_sha256"),
        "checkpoint_fidelity_identity_sha256": entry.get("checkpoint_fidelity_identity_sha256"),
        "checkpoint_fidelity_receipt_sha256": entry.get("checkpoint_fidelity_receipt_sha256"),
        "checkpoint_fidelity_receipt_file_sha256": entry.get(
            "checkpoint_fidelity_receipt_file_sha256"
        ),
        "grpo_beta": entry.get("grpo_beta"),
        "fixed_kl_reference": entry.get("fixed_kl_reference"),
        "exact_peft_kl_runtime": entry.get("exact_peft_kl_runtime"),
        "exact_peft_kl_post_step": entry.get("exact_peft_kl_post_step"),
        "benchmark_manifest_sha256": manifest.get("benchmark_manifest_sha256"),
        "checkpoint_selection_receipt_sha256": entry.get("checkpoint_selection_receipt_sha256"),
    }
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _validate_confirmation_attempt_claim(
    *,
    evidence: ArtifactEvidence,
    ledger_root: str,
    campaign_id: str,
    campaign_manifest_sha256: str,
    confirmation_attempt_id: str,
) -> None:
    expected_identity = {
        "basename": f"{confirmation_attempt_id}.json",
        "confirmation_attempt_id": confirmation_attempt_id,
    }
    if evidence.confirmation_attempt_claim != expected_identity:
        raise ValueError("confirmation artifact has no exact operational attempt claim identity")
    ledger_directory = os.path.realpath(os.path.join(ledger_root, campaign_manifest_sha256))
    claim_path = os.path.realpath(os.path.join(ledger_directory, f"{confirmation_attempt_id}.json"))
    if os.path.commonpath([ledger_directory, claim_path]) != ledger_directory:
        raise ValueError("confirmation attempt claim escaped the frozen ledger")
    if os.path.islink(claim_path) or not os.path.isfile(claim_path):
        raise ValueError("confirmation attempt claim is missing or not a regular file")
    if stat.S_IMODE(os.stat(claim_path).st_mode) & 0o222:
        raise ValueError("confirmation attempt claim is writable")
    raw_bytes, snapshot = _read_stable_file_bytes(claim_path)
    try:
        claim = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("confirmation attempt claim is not valid UTF-8 JSON") from exc
    expected_claim_fields = {
        "schema_version",
        "protocol",
        "campaign_manifest_sha256",
        "campaign_id",
        "confirmation_attempt_id",
        "experiment_fingerprint",
        "results_dir",
        "claim_sha256",
    }
    if not isinstance(claim, dict) or set(claim) != expected_claim_fields:
        raise ValueError("confirmation attempt claim has an invalid schema")
    unsigned_claim = dict(claim)
    stored_claim_sha = unsigned_claim.pop("claim_sha256", None)
    results_dir = claim.get("results_dir")
    expected_experiment = hashlib.sha256(
        _canonical_json_bytes(evidence.experiment_provenance)
    ).hexdigest()
    if (
        claim.get("schema_version") != 2
        or claim.get("protocol") != "clasp-confirmation-attempt-v2"
        or claim.get("campaign_manifest_sha256") != campaign_manifest_sha256
        or claim.get("campaign_id") != campaign_id
        or claim.get("confirmation_attempt_id") != confirmation_attempt_id
        or claim.get("experiment_fingerprint") != expected_experiment
        or not isinstance(results_dir, str)
        or not os.path.isabs(results_dir)
        or os.path.realpath(results_dir) != results_dir
        or stored_claim_sha != hashlib.sha256(_canonical_json_bytes(unsigned_claim)).hexdigest()
    ):
        raise ValueError("confirmation attempt claim disagrees with immutable evidence")
    if (
        evidence.artifact_type == "native"
        and os.path.realpath(os.path.dirname(evidence.details_path)) != results_dir
    ):
        raise ValueError("native confirmation output is outside its claimed results_dir")
    marker_path = os.path.join(results_dir, ".confirmation_attempt.json")
    if os.path.islink(marker_path) or not os.path.isfile(marker_path):
        raise ValueError("confirmation output attempt marker is missing")
    if stat.S_IMODE(os.stat(marker_path).st_mode) & 0o222:
        raise ValueError("confirmation output attempt marker is writable")
    marker_bytes, marker_snapshot = _read_stable_file_bytes(marker_path)
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("confirmation output attempt marker is invalid") from exc
    if marker != claim:
        raise ValueError("confirmation output marker disagrees with durable attempt claim")
    _assert_path_matches_snapshot(claim_path, snapshot)
    _assert_path_matches_snapshot(marker_path, marker_snapshot)


def _validate_campaign_manifest(
    manifest: Mapping[str, object],
    *,
    campaign_manifest_identity: Mapping[str, object],
    specs: Sequence[RunSpec],
    by_method_seed: Mapping[tuple[str, int], Mapping[RecordKey, Record]],
    evidence_by_spec: Mapping[RunSpec, Sequence[ArtifactEvidence]],
    lane_name: str,
    lane_registry_sha256: str,
    target_registry_sha256: str,
    benchmark_manifest: Mapping[str, object],
    baseline: str | None = None,
    challenger: str | None = None,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_814,
) -> None:
    _validate_confirmation_comparison_design(
        manifest,
        baseline=baseline,
        challenger=challenger,
    )
    _validate_confirmation_task_suite(manifest, benchmark_manifest)
    _validate_campaign_leakage_registry_binding(manifest, benchmark_manifest)
    _validate_confirmation_analysis(
        manifest,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    raw_methods = manifest.get("methods")
    raw_seeds = manifest.get("training_seeds")
    raw_tasks = manifest.get("tasks")
    if (
        not isinstance(raw_methods, list)
        or not raw_methods
        or not all(isinstance(value, str) and value in TRAINING_METHODS for value in raw_methods)
        or not isinstance(raw_seeds, list)
        or not raw_seeds
        or not all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in raw_seeds
        )
        or not isinstance(raw_tasks, list)
        or not raw_tasks
        or not all(isinstance(value, str) and value in PUBLICATION_TASKS for value in raw_tasks)
    ):
        raise ValueError("campaign methods/training_seeds/tasks are invalid")
    methods = sorted(raw_methods)
    seeds = sorted(raw_seeds)
    tasks = sorted(raw_tasks)
    if (
        len(methods) != len(set(methods))
        or len(seeds) != len(set(seeds))
        or len(tasks) != len(set(tasks))
    ):
        raise ValueError("campaign methods/training_seeds/tasks contain duplicates")
    if manifest.get("lane") != lane_name:
        raise ValueError("campaign lane does not match --lane")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign manifest has no campaign_id")
    expected_campaign_identity = {
        "basename": campaign_manifest_identity.get("basename"),
        "file_sha256": campaign_manifest_identity.get("file_sha256"),
        "size_bytes": campaign_manifest_identity.get("size_bytes"),
        "campaign_manifest_sha256": manifest.get("campaign_manifest_sha256"),
        "campaign_id": campaign_id,
    }
    if (
        campaign_manifest_identity != expected_campaign_identity
        or not isinstance(expected_campaign_identity["basename"], str)
        or not expected_campaign_identity["basename"]
        or not _is_sha256(expected_campaign_identity["file_sha256"])
        or isinstance(expected_campaign_identity["size_bytes"], bool)
        or not isinstance(expected_campaign_identity["size_bytes"], int)
        or int(expected_campaign_identity["size_bytes"]) <= 0
        or not _is_sha256(expected_campaign_identity["campaign_manifest_sha256"])
    ):
        raise ValueError("campaign manifest portable identity is invalid")
    if manifest.get("lane_contract_registry_sha256") != lane_registry_sha256:
        raise ValueError("campaign is bound to another lane contract registry")
    if manifest.get("target_registry_sha256") != target_registry_sha256:
        raise ValueError("campaign is bound to another target registry")
    if manifest.get("benchmark_manifest_sha256") != benchmark_manifest.get("manifest_sha256"):
        raise ValueError("campaign is bound to another confirmation benchmark")
    attempt_ledger = manifest.get("confirmation_attempt_ledger")
    ledger_root = attempt_ledger.get("canonical_root") if isinstance(attempt_ledger, dict) else None
    if (
        not isinstance(attempt_ledger, dict)
        or set(attempt_ledger) != {"protocol", "canonical_root"}
        or attempt_ledger.get("protocol") != "posix-create-once-lock-v1"
        or not isinstance(ledger_root, str)
        or not os.path.isabs(ledger_root)
        or os.path.normpath(ledger_root) != ledger_root
        or os.path.realpath(ledger_root) != ledger_root
        or ledger_root in {"/", os.path.expanduser("~")}
        or len([part for part in ledger_root.split(os.sep) if part]) < 3
    ):
        raise ValueError("campaign has no safe shared confirmation attempt ledger")
    benchmark_sets = benchmark_manifest.get("test_sets")
    if not isinstance(benchmark_sets, dict) or tasks != sorted(benchmark_sets):
        raise ValueError("campaign task set does not equal the frozen benchmark task set")

    spec_keys: list[tuple[str, int, str]] = []
    for spec in specs:
        if spec.task is None:
            raise ValueError("confirmation --run entries must use METHOD,SEED,TASK,PATH")
        spec_keys.append((spec.method, spec.seed, spec.task))
    expected_grid = sorted(
        (method, seed, task) for method in methods for seed in seeds for task in tasks
    )
    if sorted(spec_keys) != expected_grid:
        raise ValueError(
            "confirmation --run entries do not equal the campaign Cartesian grid: "
            f"expected={expected_grid}, observed={sorted(spec_keys)}"
        )
    benchmark_registry_identity = benchmark_manifest.get("forbidden_source_registry_identity")
    if not isinstance(benchmark_registry_identity, dict):
        raise ValueError("publication benchmark has no forbidden-source registry identity")
    common_selection_identity, common_selection_payload = _validate_campaign_selection_receipt(
        manifest,
        expected_grid=expected_grid,
        campaign_id=str(campaign_id),
        expected_candidate_budget=8,
        expected_forbidden_source_registry_identity=benchmark_registry_identity,
    )

    entries = manifest.get("runs")
    if not isinstance(entries, list):
        raise ValueError("campaign manifest has no runs list")
    entries_by_key: dict[tuple[str, int, str], Mapping[str, object]] = {}
    seen_attempt_ids: set[str] = set()
    expected_entry_fields = {
        "method",
        "training_seed",
        "task",
        "training_run_contract_sha256",
        "checkpoint_step",
        "checkpoint_artifact_sha256",
        "checkpoint_fidelity_identity_sha256",
        "checkpoint_fidelity_receipt_sha256",
        "checkpoint_fidelity_receipt_file_sha256",
        "grpo_beta",
        "fixed_kl_reference",
        "exact_peft_kl_runtime",
        "exact_peft_kl_post_step",
        "checkpoint_selection_receipt_sha256",
        "confirmation_attempt_id",
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_entry_fields:
            raise ValueError("campaign run entry must be an object")
        key = _campaign_run_key(entry)
        if key in entries_by_key:
            raise ValueError(f"duplicate campaign run identity: {key}")
        for field in (
            "training_run_contract_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_fidelity_identity_sha256",
            "checkpoint_fidelity_receipt_sha256",
            "checkpoint_fidelity_receipt_file_sha256",
            "checkpoint_selection_receipt_sha256",
            "confirmation_attempt_id",
        ):
            if not _is_sha256(entry.get(field)):
                raise ValueError(f"campaign run entry has invalid {field}")
        step = entry.get("checkpoint_step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("campaign run entry has invalid checkpoint_step")
        _validate_exact_post_step_projection(
            entry.get("grpo_beta"),
            entry.get("fixed_kl_reference"),
            entry.get("exact_peft_kl_runtime"),
            entry.get("exact_peft_kl_post_step"),
            label="campaign run entry",
            expected_run_contract_sha256=entry.get("training_run_contract_sha256"),
            expected_checkpoint_step=entry.get("checkpoint_step"),
        )
        attempt_id = str(entry["confirmation_attempt_id"])
        if attempt_id != _confirmation_attempt_id(manifest, entry):
            raise ValueError("campaign run confirmation_attempt_id is not deterministic")
        if attempt_id in seen_attempt_ids:
            raise ValueError("campaign confirmation_attempt_id values are not unique")
        seen_attempt_ids.add(attempt_id)
        entries_by_key[key] = entry
    if sorted(entries_by_key) != expected_grid:
        raise ValueError("campaign manifest run entries do not equal its declared grid")

    raw_decode_seed = manifest.get("decode_seed")
    if (
        isinstance(raw_decode_seed, bool)
        or not isinstance(raw_decode_seed, int)
        or raw_decode_seed < 0
    ):
        raise ValueError("campaign decode_seed must be a nonnegative integer")
    decode_seed = raw_decode_seed
    contracts_by_group: dict[tuple[str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for spec in specs:
        assert spec.task is not None
        entry = entries_by_key[(spec.method, spec.seed, spec.task)]
        evidences = evidence_by_spec.get(spec, [])
        if len(evidences) != 1:
            raise ValueError(
                "each confirmation --run must resolve to exactly one task artifact; "
                f"{spec.method}/seed{spec.seed}/{spec.task} resolved {len(evidences)}"
            )
        evidence = evidences[0]
        contract = evidence.training_run_contract
        _validate_training_contract(
            contract,
            evidence.training_run_contract_sha256,
            method=spec.method,
            seed=spec.seed,
            task=spec.task,
            label=evidence.details_path,
        )
        assert isinstance(contract, dict)
        contracts_by_group[(spec.task, spec.seed)][spec.method] = contract
        generation = evidence.experiment_provenance.get("generation")
        if not isinstance(generation, dict) or generation.get("seed") != decode_seed:
            raise ValueError("artifact decode seed disagrees with frozen campaign manifest")
        binding = evidence.checkpoint_selection_binding
        receipt = evidence.checkpoint_selection_receipt
        if not isinstance(binding, dict) or not isinstance(receipt, dict):
            raise ValueError("campaign run lacks checkpoint selection binding")
        if receipt != common_selection_identity:
            raise ValueError("campaign artifacts do not share the one frozen selection receipt")
        if receipt.get("campaign_id") != campaign_id:
            raise ValueError("checkpoint-selection receipt belongs to another campaign")
        expected_fields = {
            "training_run_contract_sha256": evidence.training_run_contract_sha256,
            "checkpoint_step": binding.get("checkpoint_step"),
            "checkpoint_artifact_sha256": binding.get("checkpoint_artifact_sha256"),
            "checkpoint_fidelity_identity_sha256": binding.get(
                "checkpoint_fidelity_identity_sha256"
            ),
            "checkpoint_fidelity_receipt_sha256": binding.get("checkpoint_fidelity_receipt_sha256"),
            "checkpoint_fidelity_receipt_file_sha256": binding.get(
                "checkpoint_fidelity_receipt_file_sha256"
            ),
            "grpo_beta": binding.get("grpo_beta"),
            "fixed_kl_reference": binding.get("fixed_kl_reference"),
            "exact_peft_kl_runtime": binding.get("exact_peft_kl_runtime"),
            "exact_peft_kl_post_step": binding.get("exact_peft_kl_post_step"),
            "checkpoint_selection_receipt_sha256": receipt.get("receipt_sha256"),
        }
        contract_beta = contract["training_config"].get("beta")
        contract_fixed_kl = contract.get("fixed_kl_reference")
        if (
            binding.get("grpo_beta") != contract_beta
            or binding.get("fixed_kl_reference") != contract_fixed_kl
            or binding.get("exact_peft_kl_runtime") != evidence.exact_peft_kl_runtime
            or binding.get("exact_peft_kl_post_step") != evidence.exact_peft_kl_post_step
        ):
            raise ValueError("campaign KL binding disagrees with training runtime/contract")
        if binding.get("checkpoint_artifact_sha256") != _checkpoint_artifact_sha256(
            evidence.experiment_provenance
        ):
            raise ValueError("campaign checkpoint binding disagrees with evaluated model artifact")
        fidelity_binding = {
            "checkpoint_fidelity_identity_sha256": evidence.checkpoint_fidelity.get(
                "identity_sha256"
            ),
            "checkpoint_fidelity_receipt_sha256": evidence.checkpoint_fidelity.get(
                "receipt_sha256"
            ),
            "checkpoint_fidelity_receipt_file_sha256": evidence.checkpoint_fidelity.get(
                "receipt_file_sha256"
            ),
        }
        if any(binding.get(field) != expected for field, expected in fidelity_binding.items()):
            raise ValueError(
                "campaign checkpoint binding disagrees with publication fidelity identity"
            )
        if binding.get("checkpoint_selection_receipt_sha256") != receipt.get("receipt_sha256"):
            raise ValueError("campaign checkpoint binding is not bound to its selection receipt")
        group_id = f"{spec.task}/{spec.method}/seed{spec.seed}"
        selection_group = common_selection_payload["selections"].get(group_id)
        selected = selection_group.get("selected") if isinstance(selection_group, dict) else None
        selection_evaluation_fingerprint = binding.get("selection_evaluation_fingerprint")
        expected_selected = {
            "task": spec.task,
            "method": spec.method,
            "train_seed": spec.seed,
            "checkpoint_step": binding.get("checkpoint_step"),
            "evaluation_fingerprint": selection_evaluation_fingerprint,
            "checkpoint_artifact_sha256": binding.get("checkpoint_artifact_sha256"),
            "checkpoint_fidelity_identity_sha256": binding.get(
                "checkpoint_fidelity_identity_sha256"
            ),
            "checkpoint_fidelity_receipt_sha256": binding.get("checkpoint_fidelity_receipt_sha256"),
            "checkpoint_fidelity_receipt_file_sha256": binding.get(
                "checkpoint_fidelity_receipt_file_sha256"
            ),
            "grpo_beta": binding.get("grpo_beta"),
            "fixed_kl_reference": binding.get("fixed_kl_reference"),
            "exact_peft_kl_runtime": binding.get("exact_peft_kl_runtime"),
            "exact_peft_kl_post_step": binding.get("exact_peft_kl_post_step"),
            "run_contract_sha256": evidence.training_run_contract_sha256,
        }
        if (
            not _is_sha256(selection_evaluation_fingerprint)
            or not isinstance(selected, dict)
            or any(selected.get(field) != expected for field, expected in expected_selected.items())
            or any(
                not _is_sha256(selected.get(field))
                for field in (
                    "protocol_fingerprint",
                    "data_manifest_sha256",
                    "strict_verifier_manifest_sha256",
                )
            )
        ):
            raise ValueError(
                f"checkpoint-selection receipt selected provenance mismatch at {group_id}"
            )
        for field, expected in expected_fields.items():
            if field == "checkpoint_step":
                if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
                    raise ValueError("campaign artifact has invalid checkpoint_step")
            elif field == "grpo_beta":
                if _finite_float(expected) is None or float(expected) < 0.0:
                    raise ValueError("campaign artifact has invalid grpo_beta")
            elif field == "fixed_kl_reference":
                _validate_exact_post_step_projection(
                    expected_fields["grpo_beta"],
                    expected,
                    expected_fields["exact_peft_kl_runtime"],
                    expected_fields["exact_peft_kl_post_step"],
                    label="campaign artifact",
                    expected_run_contract_sha256=expected_fields["training_run_contract_sha256"],
                    expected_checkpoint_step=expected_fields["checkpoint_step"],
                )
            elif field == "exact_peft_kl_runtime":
                # Validated together with the beta/source chronology above.
                pass
            elif field == "exact_peft_kl_post_step":
                # Validated together with the beta/source chronology above.
                pass
            elif not _is_sha256(expected):
                raise ValueError(f"campaign artifact has invalid {field}")
            if entry.get(field) != expected:
                raise ValueError(
                    f"campaign run {spec.method}/seed{spec.seed}/{spec.task} {field} mismatch"
                )
        attempt_id = str(entry["confirmation_attempt_id"])
        expected_campaign_run_binding = {
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": manifest["campaign_manifest_sha256"],
            "method": spec.method,
            "training_seed": spec.seed,
            "task": spec.task,
            "confirmation_attempt_id": attempt_id,
            **expected_fields,
            "benchmark_manifest_sha256": manifest["benchmark_manifest_sha256"],
        }
        if evidence.campaign_manifest != expected_campaign_identity:
            raise ValueError("confirmation artifact is bound to another campaign manifest")
        if evidence.campaign_run_binding != expected_campaign_run_binding:
            raise ValueError(
                "confirmation artifact campaign run binding disagrees with its grid cell"
            )
        _validate_confirmation_attempt_claim(
            evidence=evidence,
            ledger_root=str(ledger_root),
            campaign_id=str(campaign_id),
            campaign_manifest_sha256=str(manifest["campaign_manifest_sha256"]),
            confirmation_attempt_id=attempt_id,
        )
        matching_records = [
            record
            for (task, _split, _index), record in by_method_seed.get(
                (spec.method, spec.seed), {}
            ).items()
            if task == spec.task
        ]
        if not matching_records:
            raise ValueError("confirmation campaign cell has no records")
        for record in matching_records:
            source_record = (
                record.get("raw_record") if evidence.artifact_type == "repair" else record
            )
            if not isinstance(source_record, dict) or (
                source_record.get("campaign_manifest_sha256")
                != manifest["campaign_manifest_sha256"]
                or source_record.get("confirmation_attempt_id") != attempt_id
            ):
                raise ValueError("confirmation record is not bound to its campaign attempt")
            for field, expected in (
                ("campaign_manifest_sha256", manifest["campaign_manifest_sha256"]),
                ("confirmation_attempt_id", attempt_id),
            ):
                if record.get(field) is not None and record.get(field) != expected:
                    raise ValueError(f"derived confirmation record has conflicting {field}")
    _validate_campaign_training_lineage(
        contracts_by_group=contracts_by_group,
        benchmark_manifest=benchmark_manifest,
    )
    _validate_campaign_training_authorization(
        manifest,
        campaign_id=str(campaign_id),
        contracts_by_group=contracts_by_group,
    )
    _validate_campaign_model_initialization(
        manifest,
        contracts_by_group=contracts_by_group,
    )
    _validate_matched_training_design(
        manifest,
        contracts_by_group=contracts_by_group,
        methods=methods,
        tasks=tasks,
        seeds=seeds,
    )


def _load_frozen_benchmark(path: str) -> dict[str, object]:
    manifest = _load_json_object(path)
    validate_frozen_benchmark_manifest(manifest)
    return manifest


def _validate_frozen_benchmark_records(
    by_method_seed: dict[tuple[str, int], dict[RecordKey, Record]],
    manifest: dict[str, object],
) -> None:
    test_sets = manifest.get("test_sets")
    if not isinstance(test_sets, dict):
        raise ValueError("benchmark manifest has no test_sets object")
    expected_verifier = manifest.get("strict_verifier")
    if not isinstance(expected_verifier, dict):
        raise ValueError("benchmark manifest has no strict verifier identity")
    for (method, seed), records in sorted(by_method_seed.items()):
        tasks = sorted({key[0] for key in records})
        if tasks != sorted(test_sets):
            raise ValueError(
                f"run task set does not match frozen benchmark for method={method} "
                f"seed={seed}: expected={sorted(test_sets)}, observed={tasks}"
            )
        for task in tasks:
            task_manifest = test_sets.get(task)
            if not isinstance(task_manifest, dict):
                raise ValueError(f"run contains task absent from frozen benchmark: {task}")
            expected = task_manifest.get("ordered_semantic_instance_sha256")
            expected_rows = task_manifest.get("ordered_benchmark_row_sha256")
            expected_references = task_manifest.get("ordered_reference_output_sha256")
            if not isinstance(expected, list) or not expected:
                raise ValueError(
                    f"frozen benchmark task {task} has no ordered semantic instance hashes"
                )
            if task_manifest.get("ordered_instance_sha256") != expected:
                raise ValueError(
                    f"frozen benchmark task {task} has conflicting legacy instance hashes"
                )
            if not isinstance(expected_rows, list) or len(expected_rows) != len(expected):
                raise ValueError(
                    f"frozen benchmark task {task} has invalid ordered benchmark row hashes"
                )
            if not isinstance(expected_references, list) or len(expected_references) != len(
                expected
            ):
                raise ValueError(
                    f"frozen benchmark task {task} has invalid ordered reference hashes"
                )
            for values, aggregate_field in (
                (expected, "instance_hashes_sha256"),
                (expected_rows, "row_hashes_sha256"),
                (expected_references, "reference_hashes_sha256"),
            ):
                observed_aggregate = hashlib.sha256(_canonical_json_bytes(values)).hexdigest()
                if task_manifest.get(aggregate_field) != observed_aggregate:
                    raise ValueError(f"frozen benchmark task {task} has invalid {aggregate_field}")
            task_records = sorted(
                (key, record) for key, record in records.items() if key[0] == task
            )
            observed_indices = [key[2] for key, _record in task_records]
            expected_indices = list(range(len(expected)))
            if observed_indices != expected_indices:
                raise ValueError(
                    f"incomplete or reordered frozen benchmark for method={method} seed={seed} "
                    f"task={task}: expected indices 0..{len(expected) - 1}, "
                    f"observed {len(observed_indices)} rows"
                )
            for key, record in task_records:
                index = key[2]
                if record.get("semantic_instance_sha256") != expected[index]:
                    raise ValueError(
                        f"record semantic instance does not match frozen benchmark for "
                        f"method={method} seed={seed}, key={key}"
                    )
                if record.get("benchmark_row_sha256") != expected_rows[index]:
                    raise ValueError(
                        f"record source row does not match frozen benchmark for "
                        f"method={method} seed={seed}, key={key}"
                    )
                if record.get("reference_output_sha256") != expected_references[index]:
                    raise ValueError(
                        f"record reference output does not match frozen benchmark for "
                        f"method={method} seed={seed}, key={key}"
                    )
                if record.get("strict_verifier") != expected_verifier:
                    raise ValueError(
                        f"record verifier does not match frozen benchmark for "
                        f"method={method} seed={seed}, key={key}"
                    )


def _parse_run(value: str) -> RunSpec:
    parts = value.split(",", 3)
    if len(parts) not in {3, 4} or not parts[0].strip():
        raise argparse.ArgumentTypeError(
            "--run must be METHOD,SEED,PATH (development) or METHOD,SEED,TASK,PATH (confirmation)"
        )
    try:
        seed = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--run SEED must be an integer") from exc
    if len(parts) == 3:
        if not parts[2].strip():
            raise argparse.ArgumentTypeError("--run PATH must be non-empty")
        return RunSpec(parts[0].strip(), seed, parts[2].strip())
    task = parts[2].strip().lower()
    path = parts[3].strip()
    if task not in PUBLICATION_TASKS or not path:
        raise argparse.ArgumentTypeError(
            f"--run TASK must be one of {PUBLICATION_TASKS} and PATH must be non-empty"
        )
    return RunSpec(parts[0].strip(), seed, path, task)


def _atomic_json(path: str, payload: object) -> None:
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to replace aggregate receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".aggregate-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace aggregate receipt: {path}") from exc
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--baseline")
    parser.add_argument("--challenger")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--lane", default="raw_n8")
    parser.add_argument(
        "--lane-contracts",
        default=DEFAULT_LANE_CONTRACTS,
        help="Self-hashed frozen machine-readable lane policy.",
    )
    parser.add_argument(
        "--campaign-manifest",
        help=(
            "Self-hashed campaign grid frozen before confirmation. Required with "
            "--benchmark-manifest and forbidden for development-only aggregation."
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_814)
    evidence_group = parser.add_mutually_exclusive_group(required=True)
    evidence_group.add_argument(
        "--benchmark-manifest",
        help="Hash-frozen, leakage-audited independent-test manifest.",
    )
    evidence_group.add_argument(
        "--development-only",
        action="store_true",
        help="Label this aggregation as development/replication evidence, never confirmatory.",
    )
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help=(
            "Permit details without complete fingerprint/config/instance provenance. "
            "This is exploratory-only and disables confirmatory paired identity checks."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.allow_legacy_provenance and not args.development_only:
        parser.error("--allow-legacy-provenance is permitted only with --development-only")
    if args.benchmark_manifest and not args.campaign_manifest:
        parser.error("--campaign-manifest is required with --benchmark-manifest")
    if args.benchmark_manifest and (not args.baseline or not args.challenger):
        parser.error("confirmation requires both --baseline and --challenger")
    if args.benchmark_manifest and args.baseline == args.challenger:
        parser.error("confirmation baseline and challenger must be distinct")
    if args.development_only and args.campaign_manifest:
        parser.error("--campaign-manifest is reserved for independent confirmation")

    lane_registry, lane_registry_sha256 = _load_lane_contract_registry(args.lane_contracts)
    target_registry, target_registry_sha256 = _load_target_registry(
        args.targets,
        lane_registry=lane_registry,
    )
    try:
        lane = target_registry["lanes"][args.lane]
        external_context = lane["external_context"]
        target_results = external_context["reported_point_estimates"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid target lane {args.lane!r}") from exc
    expected_k = lane.get("candidate_budget")
    expected_k = expected_k if isinstance(expected_k, int) else None

    lane_policies = lane_registry.get("lanes")
    lane_policy = lane_policies.get(args.lane) if isinstance(lane_policies, dict) else None
    if not isinstance(lane_policy, dict):
        raise ValueError(f"lane contract registry has no lane {args.lane!r}")
    _require_supported_lane_claim(args.lane, lane_policy)
    target_budget = lane.get("candidate_budget")
    policy_budget = lane_policy.get("reported_candidate_budget")
    if isinstance(target_budget, int):
        if target_budget != policy_budget:
            raise ValueError("target registry and lane policy candidate budgets disagree")
    elif not isinstance(policy_budget, dict):
        raise ValueError("adaptive target lane has no machine-readable budget range")

    by_method_seed: dict[tuple[str, int], dict[RecordKey, Record]] = {}
    evidence_by_spec: dict[RunSpec, list[ArtifactEvidence]] = {}
    seen_spec_identities: set[tuple[str, int, str | None]] = set()
    for spec in args.run:
        key = (spec.method, spec.seed)
        spec_identity = (spec.method, spec.seed, spec.task)
        if spec_identity in seen_spec_identities:
            raise ValueError(f"duplicate --run identity: {spec_identity}")
        seen_spec_identities.add(spec_identity)
        evidence: list[ArtifactEvidence] = []
        loaded = load_run(
            spec,
            expected_k=expected_k,
            allow_legacy_provenance=bool(args.allow_legacy_provenance),
            publication_mode=not bool(args.allow_legacy_provenance),
            evidence_sink=evidence,
        )
        if not args.allow_legacy_provenance:
            _validate_lane_evidence(
                spec,
                loaded,
                evidence,
                lane_name=args.lane,
                lane_registry=lane_registry,
                confirmation=bool(args.benchmark_manifest),
            )
        evidence_by_spec[spec] = evidence
        destination = by_method_seed.setdefault(key, {})
        collisions = sorted(set(destination) & set(loaded))
        if collisions:
            raise ValueError(
                f"duplicate detail keys across --run entries for method={spec.method} "
                f"seed={spec.seed}: {collisions[:8]}"
            )
        destination.update(loaded)

    benchmark_manifest = None
    if args.benchmark_manifest:
        benchmark_manifest = _load_frozen_benchmark(args.benchmark_manifest)
        _validate_frozen_benchmark_records(by_method_seed, benchmark_manifest)
        _validate_benchmark_provenance_bindings(
            evidence_by_spec,
            benchmark_path=args.benchmark_manifest,
            benchmark_manifest=benchmark_manifest,
        )
        _validate_frozen_source_row_closure(
            specs=args.run,
            by_method_seed=by_method_seed,
            evidence_by_spec=evidence_by_spec,
            benchmark_manifest=benchmark_manifest,
            benchmark_path=args.benchmark_manifest,
        )
        campaign_manifest, campaign_manifest_identity = _load_campaign_manifest(
            args.campaign_manifest
        )
        _validate_campaign_manifest(
            campaign_manifest,
            campaign_manifest_identity=campaign_manifest_identity,
            specs=args.run,
            by_method_seed=by_method_seed,
            evidence_by_spec=evidence_by_spec,
            lane_name=args.lane,
            lane_registry_sha256=lane_registry_sha256,
            target_registry_sha256=target_registry_sha256,
            benchmark_manifest=benchmark_manifest,
            baseline=args.baseline,
            challenger=args.challenger,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        campaign_manifest = None
        campaign_manifest_identity = None

    methods = sorted({method for method, _ in by_method_seed})
    tasks = sorted({key[0] for records in by_method_seed.values() for key in records})
    summaries: dict[str, dict[str, object]] = {}
    summary_rng = random.Random(args.bootstrap_seed ^ 0xC1A5)
    for method in methods:
        summaries[method] = {}
        method_runs = {
            seed: records
            for (candidate, seed), records in by_method_seed.items()
            if candidate == method
        }
        for task in tasks:
            task_target = target_results.get(task, {})
            if not isinstance(task_target, dict):
                raise ValueError(f"target registry has no external context row for {task}")
            published_gap = _finite_float(task_target.get("reported_gap"))
            target_gap = _published_gap_target_for_task(
                task,
                task_target,
                external_context=external_context,
            )
            per_seed: dict[str, object] = {}
            pooled: list[Record] = []
            for seed, records in sorted(method_runs.items()):
                selected = [record for key, record in records.items() if key[0] == task]
                if not selected:
                    continue
                pooled.extend(selected)
                seed_summary = summarize_records(selected, target_gap)
                seed_summary["conditional_gap_bootstrap95"] = _bootstrap_ci(
                    list(selected),
                    lambda sample: _mean(
                        gap
                        for record in sample
                        if (gap := _gap(record)) is not None  # type: ignore[arg-type]
                    ),
                    repetitions=args.bootstrap_repetitions,
                    rng=summary_rng,
                )
                per_seed[str(seed)] = seed_summary
            if not pooled:
                continue
            aggregate = summarize_records(pooled, target_gap)
            seed_rows = [row for row in per_seed.values() if isinstance(row, dict)]
            seed_pass = [
                float(row["pass_at_k"])
                for row in seed_rows
                if _finite_float(row.get("pass_at_k")) is not None
            ]
            seed_normalized_auc = [
                float(row["normalized_feasible_gap_auc"])
                for row in seed_rows
                if _finite_float(row.get("normalized_feasible_gap_auc")) is not None
            ]
            seed_gaps = [
                float(row["conditional_gap_mean"])
                for row in seed_rows
                if _finite_float(row.get("conditional_gap_mean")) is not None
            ]
            seed_target = [
                float(row["success_at_target_gap"])
                for row in seed_rows
                if _finite_float(row.get("success_at_target_gap")) is not None
            ]
            training_seed_aggregate = {
                "training_seed_count": len(seed_rows),
                "primary_inference_unit": "training_seed",
                "normalized_feasible_gap_auc_seed_mean": _mean(seed_normalized_auc),
                "normalized_feasible_gap_auc_seed_sample_std": _sample_std(seed_normalized_auc),
                "normalized_feasible_gap_auc_seed_min": (
                    min(seed_normalized_auc) if seed_normalized_auc else None
                ),
                "normalized_feasible_gap_auc_seed_max": (
                    max(seed_normalized_auc) if seed_normalized_auc else None
                ),
                "normalized_feasible_gap_auc_range": list(PRIMARY_GAP_RANGE),
                "pass_at_k_seed_mean": _mean(seed_pass),
                "pass_at_k_seed_sample_std": _sample_std(seed_pass),
                "pass_at_k_seed_min": min(seed_pass) if seed_pass else None,
                "pass_at_k_seed_max": max(seed_pass) if seed_pass else None,
                "conditional_gap_seed_mean": _mean(seed_gaps),
                "conditional_gap_seed_sample_std": _sample_std(seed_gaps),
                "conditional_gap_seed_min": min(seed_gaps) if seed_gaps else None,
                "conditional_gap_seed_max": max(seed_gaps) if seed_gaps else None,
                "success_at_target_gap_seed_mean": _mean(seed_target),
                "success_at_target_gap_seed_sample_std": _sample_std(seed_target),
            }
            published_feasibility = _finite_float(task_target.get("feasibility"))
            observed_feasibility = _finite_float(training_seed_aggregate["pass_at_k_seed_mean"])
            observed_gap = _finite_float(training_seed_aggregate["conditional_gap_seed_mean"])
            descriptive_comparison = _published_point_descriptive_comparison(
                task=task,
                observed_feasibility=observed_feasibility,
                observed_gap=observed_gap,
                published_feasibility=published_feasibility,
                published_gap=published_gap,
                external_context=external_context,
                task_target=task_target,
            )
            summaries[method][task] = {
                "per_seed": per_seed,
                "training_seed_aggregate": training_seed_aggregate,
                "pooled_for_descriptive_use_only": aggregate,
                "external_context": {
                    "context_id": external_context["context_id"],
                    "comparison_scope": external_context["comparison_scope"],
                    "reported_point_estimate": task_target,
                    **descriptive_comparison,
                },
            }

    comparisons: dict[str, object] = {}
    if bool(args.baseline) != bool(args.challenger):
        parser.error("--baseline and --challenger must be supplied together")
    if args.baseline and args.challenger:
        if args.baseline not in methods or args.challenger not in methods:
            parser.error("baseline and challenger must name supplied --run methods")
        baseline_records = {
            (seed, key): record
            for (method, seed), records in by_method_seed.items()
            if method == args.baseline
            for key, record in records.items()
        }
        challenger_records = {
            (seed, key): record
            for (method, seed), records in by_method_seed.items()
            if method == args.challenger
            for key, record in records.items()
        }
        if not args.allow_legacy_provenance:
            _validate_paired_provenance(baseline_records, challenger_records)
        rng = random.Random(args.bootstrap_seed)
        for task in tasks:
            task_target = target_results.get(task, {})
            if not isinstance(task_target, dict):
                raise ValueError(f"target registry has no external context row for {task}")
            target_gap = _published_gap_target_for_task(
                task,
                task_target,
                external_context=external_context,
            )
            comparison = compare_paired(
                baseline_records,
                challenger_records,
                task=task,
                target_gap=target_gap,
                repetitions=args.bootstrap_repetitions,
                rng=rng,
            )
            comparisons[task] = comparison
        multiple_testing = _apply_seed_family_holm(comparisons)
        if set(multiple_testing) != set(CONFIRMATION_OUTCOME_FAMILIES):
            raise ValueError(
                "computed hypothesis-test families disagree with the frozen analysis contract"
            )
        pass_at_k_noninferiority_gate = _apply_pass_at_k_noninferiority_holm(comparisons)
        if pass_at_k_noninferiority_gate.get("family") != CONFIRMATION_NONINFERIORITY_FAMILY:
            raise ValueError("computed noninferiority family disagrees with frozen analysis")
    else:
        multiple_testing = {}
        pass_at_k_noninferiority_gate = None

    payload = {
        "schema_version": 3,
        "aggregation_implementation": _aggregation_implementation_identity(),
        "claim_status": "NO_SOTA_CLAIM",
        "claim_status_reason": (
            "External published/preprint values in the target registry are context-only: "
            "LLMCoSolver values are restricted to its historical first-100 author protocol, "
            "and FALCON is an unreproduced cross-system preprint context. This aggregate "
            "therefore cannot emit a SOTA or strict-beat claim from those values."
        ),
        "lane": args.lane,
        "expected_candidate_budget": expected_k,
        "targets_path": os.path.abspath(args.targets),
        "targets_sha256": target_registry_sha256,
        "external_context_registry_projection": {
            field: external_context[field]
            for field in (
                "context_id",
                "name",
                "status",
                "sources",
                "evidence_kind",
                "comparison_scope",
                "aggregate_point_comparison_eligible",
                "strict_beat_claim_eligible",
                "reason_not_claim_eligible",
                "scope",
            )
        },
        "lane_contracts": {
            "path": os.path.abspath(args.lane_contracts),
            "registry_sha256": lane_registry_sha256,
        },
        "confirmatory_provenance_enforced": not bool(args.allow_legacy_provenance),
        "evidence_scope": "independent_confirmation" if benchmark_manifest else "development_only",
        "benchmark_manifest": (
            {
                "path": os.path.abspath(args.benchmark_manifest),
                "manifest_sha256": benchmark_manifest["manifest_sha256"],
            }
            if benchmark_manifest is not None
            else None
        ),
        "campaign_manifest": (
            {
                "path": os.path.abspath(args.campaign_manifest),
                **campaign_manifest_identity,
            }
            if campaign_manifest is not None
            else None
        ),
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "note": "Paired intervals resample training seeds, then paired benchmark instances within seed; per-seed rows remain visible and pooled rows are descriptive only.",
        },
        "metric_semantics": {
            "feasibility": "strict shared-verifier pass@K",
            "primary_normalized_feasible_gap_auc": (
                "normalized area under S(t)=I(strictly feasible and verified gap<=t) "
                "for t in [0,0.10]; infeasible records score zero"
            ),
            "gap": (
                "conditional gap to the artifact-bound evaluation reference; the term "
                "optimality gap is permitted only when that benchmark row has an accepted "
                "certificate under its separate reference-solver contract"
            ),
            "external_context_comparison": (
                "disabled: author-reported point estimates retain their source protocol and "
                "denominator semantics and cannot establish a SOTA or strict-beat claim"
            ),
        },
        "runs": [spec.__dict__ for spec in args.run],
        "run_artifact_receipts": [
            {
                "method": spec.method,
                "training_seed": spec.seed,
                "task": spec.task,
                "details_path": os.path.abspath(evidence.details_path),
                "details_sha256": evidence.details_sha256,
                "details_size_bytes": evidence.details_size_bytes,
                "details_record_count": evidence.details_record_count,
                "run_config_sha256": evidence.run_config_sha256,
                "summary_sha256": evidence.summary_sha256,
                "artifact_type": evidence.artifact_type,
                "checkpoint_fidelity": evidence.checkpoint_fidelity,
                "grpo_beta": (
                    evidence.training_run_contract.get("training_config", {}).get("beta")
                    if isinstance(evidence.training_run_contract, dict)
                    else None
                ),
                "fixed_kl_reference": (
                    evidence.training_run_contract.get("fixed_kl_reference")
                    if isinstance(evidence.training_run_contract, dict)
                    else None
                ),
                "exact_peft_kl_runtime": evidence.exact_peft_kl_runtime,
                "exact_peft_kl_post_step": evidence.exact_peft_kl_post_step,
                "confirmation_attempt_id": (
                    evidence.campaign_run_binding.get("confirmation_attempt_id")
                    if isinstance(evidence.campaign_run_binding, dict)
                    else None
                ),
            }
            for spec in args.run
            for evidence in evidence_by_spec.get(spec, [])
        ],
        "run_provenance": {
            f"{method}/seed{seed}": _provenance_summary(records)
            for (method, seed), records in sorted(by_method_seed.items())
        },
        "summaries": summaries,
        "comparison": {
            "baseline": args.baseline,
            "challenger": args.challenger,
            "per_task": comparisons,
            "multiple_testing": multiple_testing,
            "pass_at_k_noninferiority_gate": pass_at_k_noninferiority_gate,
            "note": (
                "The sole primary outcome is normalized feasible-gap success-curve AUC "
                "over the frozen 0-10% gap range. Primary exact two-sided tests exhaustively "
                "sign-flip paired task effects over eight independent training seeds and "
                "Holm-adjust exactly five task hypotheses. Pass@8 noninferiority is a "
                "separate secondary one-sided sign-flip family with margin 0.01 and an "
                "all-five-task gate. Other within-seed tests and pooled rows are descriptive only."
            ),
        },
    }
    payload["aggregate_receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    _atomic_json(args.output, payload)

    print(f"wrote {args.output}")
    for method in methods:
        for task, row in summaries[method].items():
            aggregate = row["pooled_for_descriptive_use_only"]
            seed_aggregate = row["training_seed_aggregate"]
            pass_value = _finite_float(seed_aggregate["pass_at_k_seed_mean"])
            gap_value = _finite_float(seed_aggregate["conditional_gap_seed_mean"])
            print(
                f"{method:16s} {task:5s} n={aggregate['instances']:4d} "
                f"seed-mean-pass@K={pass_value:.1%} "
                f"seed-mean-gap={f'{gap_value:.3%}' if gap_value is not None else 'n/a'} "
                "external_context=NOT_COMPARABLE"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
