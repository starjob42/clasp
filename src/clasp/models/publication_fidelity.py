"""Portable publication identity for an exact checkpoint-fidelity receipt.

The full :mod:`clasp.models.checkpoint_fidelity` receipt contains a tensor-level
closure and is intentionally stored next to the adapter.  Publication contracts
need a smaller, relocatable identity that can be embedded repeatedly without
copying hundreds of tensor records.  This module is the sole definition of that
identity.

Creating an identity always verifies the live adapter/config/receipt first.
Validating an embedded identity is dependency-light and path-free, but does not
replace live verification at artifact-producing boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from clasp.models import exact_peft_kl_post_step as exact_peft_kl_post_step
from clasp.models import exact_peft_kl_reference as exact_peft_kl
from clasp.models.checkpoint_fidelity import (
    PROTOCOL as CHECKPOINT_FIDELITY_PROTOCOL,
)
from clasp.models.checkpoint_fidelity import (
    QWEN_PUBLICATION_PROTOCOL,
    RECEIPT_FILENAME,
    CheckpointFidelityError,
    verify_checkpoint_fidelity,
)
from clasp.models.checkpoint_fidelity import (
    SCHEMA_VERSION as CHECKPOINT_FIDELITY_SCHEMA_VERSION,
)

IDENTITY_SCHEMA_VERSION = 1
IDENTITY_PROTOCOL = "clasp-checkpoint-fidelity-identity-v1"
IDENTITY_STATUS = "publication_exact"
IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "receipt_sha256",
        "receipt_file_sha256",
        "adapter_config_sha256",
        "adapter_weights_sha256",
        "trainable_tensor_inventory_sha256",
        "publication_invariant_sha256",
        "identity_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FIXED_KL_IDENTITY_SCHEMA_VERSION = 4
FIXED_KL_IDENTITY_PROTOCOL = "clasp-exact-peft-kl-source-identity-v4"
FIXED_KL_IDENTITY_STATUS = "publication_exact"
FIXED_KL_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "runtime_schema_version",
        "runtime_protocol",
        "runtime_status",
        "runtime_receipt_filename",
        "verifier_protocol",
        "verifier_implementation_sha256",
        "post_step_schema_version",
        "post_step_protocol",
        "post_step_status",
        "post_step_step1_receipt_filename",
        "post_step_receipt_filename",
        "post_step_verifier_protocol",
        "post_step_verifier_implementation_sha256",
        "parameterization",
        "authoritative_receipt_sha256",
        "authoritative_receipt_file_sha256",
        "adapter_config_sha256",
        "adapter_tree_sha256",
        "base_model_tree_sha256",
        "tokenizer_tree_sha256",
        "saved_adapter_tensor_inventory_sha256",
        "saved_adapter_dtypes",
        "saved_lora_dtype",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
        "source_checkpoint_fidelity",
        "source_binding_sha256",
        "identity_sha256",
    }
)

EXACT_PEFT_KL_SOURCE_REFERENCE_SCHEMA_VERSION = 2
EXACT_PEFT_KL_SOURCE_REFERENCE_PROTOCOL = "clasp-exact-peft-kl-source-reference-v2"
EXACT_PEFT_KL_SOURCE_REFERENCE_STATUS = "publication_exact"
EXACT_PEFT_KL_SOURCE_REFERENCE_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "fixed_kl_reference_identity_sha256",
        "source_checkpoint_fidelity_identity_sha256",
        "source_binding_sha256",
        "pre_step_schema_version",
        "pre_step_protocol",
        "pre_step_status",
        "pre_step_receipt_filename",
        "pre_step_verifier_protocol",
        "pre_step_verifier_implementation_sha256",
        "post_step_schema_version",
        "post_step_protocol",
        "post_step_status",
        "post_step_step1_receipt_filename",
        "post_step_receipt_filename",
        "post_step_verifier_protocol",
        "post_step_verifier_implementation_sha256",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
        "projection_sha256",
    }
)

EXACT_PEFT_KL_POST_STEP_LINEAGE_SCHEMA_VERSION = 1
EXACT_PEFT_KL_POST_STEP_LINEAGE_PROTOCOL = "clasp-exact-peft-kl-post-step-lineage-v1"
EXACT_PEFT_KL_POST_STEP_LINEAGE_STATUS = "publication_exact"
EXACT_PEFT_KL_POST_STEP_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "terminal_global_step",
        "selected_checkpoint_step",
        "selected_checkpoint_is_terminal",
        "training_run_contract_file_sha256",
        "exact_peft_kl_runtime_identity_sha256",
        "source_reference",
        "source_reference_sha256",
        "run_invariant",
        "run_invariant_sha256",
        "root_checkpoint_fidelity",
        "selected_checkpoint_fidelity",
        "root_step1",
        "selected_checkpoint_step1",
        "root_final",
        "selected_checkpoint_post_step",
        "lineage_sha256",
    }
)

EXACT_PEFT_KL_RUNTIME_IDENTITY_SCHEMA_VERSION = 2
EXACT_PEFT_KL_RUNTIME_IDENTITY_PROTOCOL = "clasp-exact-peft-kl-runtime-identity-v2"
EXACT_PEFT_KL_RUNTIME_IDENTITY_STATUS = "publication_exact"
EXACT_PEFT_KL_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "runtime_schema_version",
        "runtime_protocol",
        "runtime_status",
        "receipt_sha256",
        "receipt_file_sha256",
        "verifier_protocol",
        "verifier_implementation_sha256",
        "fixed_kl_reference",
        "training_dtype",
        "training_beta",
        "training_sync_ref_model",
        "training_use_vllm",
        "training_parameterization",
        "training_policy_adapter_autocast_dtype",
        "training_reference_adapter_autocast_dtype",
        "training_contract_sha256",
        "runtime_pair_tensor_inventory_sha256",
        "runtime_pair_sha256",
        "functional_equivalence_sha256",
        "distributed_world_size",
        "distributed_consensus_digest",
        "distributed_sha256",
        "identity_sha256",
    }
)

DERIVED_SOURCE_CONFIG_SNAPSHOT_SCHEMA_VERSION = 2
DERIVED_SOURCE_CONFIG_SNAPSHOT_PROTOCOL = "clasp-derived-source-config-snapshot-v2"
DERIVED_SOURCE_CONFIG_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "source_config_identity",
        "source_config_payload",
        "source_config_payload_sha256",
        "exact_peft_kl_runtime_receipt",
        "exact_peft_kl_post_step_receipts",
        "snapshot_sha256",
    }
)
EXACT_PEFT_KL_RECEIPT_SNAPSHOT_SCHEMA_VERSION = 1
EXACT_PEFT_KL_RECEIPT_SNAPSHOT_PROTOCOL = "clasp-exact-peft-kl-receipt-snapshot-v1"
EXACT_PEFT_KL_RECEIPT_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "receipt_filename",
        "receipt_size_bytes",
        "receipt_file_sha256",
        "receipt_payload",
        "receipt_payload_sha256",
        "runtime_identity",
        "verified_source_roles",
        "snapshot_sha256",
    }
)
_EXACT_PEFT_KL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "development_only",
        "publication_eligible",
        "sota_eligible",
        "purpose",
        "verifier",
        "source_binding",
        "training_contract",
        "runtime_pair",
        "distributed",
        "receipt_sha256",
    }
)
_EXACT_PEFT_KL_SOURCE_BINDING_KEYS = frozenset(
    {
        "method",
        "claim_scope",
        "adapter_path",
        "adapter_tree",
        "base_model_path",
        "base_model_tree",
        "tokenizer_path",
        "tokenizer_tree",
        "authoritative_receipt_path",
        "authoritative_receipt_protocol",
        "authoritative_receipt_sha256",
        "authoritative_receipt_file_sha256",
        "adapter_config_sha256",
        "saved_adapter_tensor_inventory_sha256",
        "saved_adapter_dtypes",
        "saved_lora_dtype",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
    }
)
EXACT_PEFT_KL_POST_STEP_SNAPSHOT_SCHEMA_VERSION = 1
EXACT_PEFT_KL_POST_STEP_SNAPSHOT_PROTOCOL = "clasp-exact-peft-kl-post-step-snapshot-v1"
EXACT_PEFT_KL_POST_STEP_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "lineage",
        "receipts",
        "snapshot_sha256",
    }
)
EXACT_PEFT_KL_POST_STEP_SNAPSHOT_ENTRY_KEYS = frozenset(
    {
        "receipt_filename",
        "receipt_size_bytes",
        "receipt_file_sha256",
        "receipt_payload",
        "receipt_payload_sha256",
        "identity",
        "verified_source_roles",
    }
)


class PublicationFidelityError(ValueError):
    """A checkpoint is not eligible for publication evidence."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _stable_regular_file_sha256(path: Path) -> str:
    if path.is_symlink():
        raise PublicationFidelityError(f"publication receipt must not be a symlink: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise PublicationFidelityError(f"cannot open publication receipt {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationFidelityError(f"publication receipt must be a regular file: {path}")
        digest = hashlib.sha256()
        try:
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                digest.update(chunk)
        except OSError as exc:
            raise PublicationFidelityError(
                f"cannot hash publication receipt {path}: {exc}"
            ) from exc
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PublicationFidelityError(f"cannot restat publication receipt {path}: {exc}") from exc
    identities = {
        (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )
        for value in (before, after, path_after)
    }
    if len(identities) != 1:
        raise PublicationFidelityError(
            f"publication receipt changed while it was fingerprinted: {path}"
        )
    return digest.hexdigest()


def _stable_regular_file_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise PublicationFidelityError(f"publication receipt must not be a symlink: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise PublicationFidelityError(f"cannot open publication receipt {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationFidelityError(f"publication receipt must be a regular file: {path}")
        chunks: list[bytes] = []
        try:
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                chunks.append(chunk)
        except OSError as exc:
            raise PublicationFidelityError(
                f"cannot read publication receipt {path}: {exc}"
            ) from exc
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PublicationFidelityError(f"cannot restat publication receipt {path}: {exc}") from exc
    identities = {
        (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )
        for value in (before, after, path_after)
    }
    if len(identities) != 1:
        raise PublicationFidelityError(f"publication receipt changed while it was read: {path}")
    return b"".join(chunks)


def validate_publication_fidelity_identity(value: object) -> dict[str, Any]:
    """Validate and return one canonical portable publication identity."""

    if not isinstance(value, dict) or set(value) != IDENTITY_KEYS:
        raise PublicationFidelityError("checkpoint-fidelity identity has an invalid schema")
    unsigned = dict(value)
    stored_identity = unsigned.pop("identity_sha256", None)
    if not isinstance(stored_identity, str) or stored_identity != _sha256_json(unsigned):
        raise PublicationFidelityError("checkpoint-fidelity identity self-hash mismatch")
    if (
        isinstance(value.get("schema_version"), bool)
        or not isinstance(value.get("schema_version"), int)
        or value.get("schema_version") != IDENTITY_SCHEMA_VERSION
        or value.get("protocol") != IDENTITY_PROTOCOL
        or value.get("status") != IDENTITY_STATUS
    ):
        raise PublicationFidelityError("checkpoint-fidelity identity is not publication-exact")
    for field in IDENTITY_KEYS - {"schema_version", "protocol", "status"}:
        if not isinstance(value.get(field), str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise PublicationFidelityError(f"checkpoint-fidelity identity has invalid {field}")
    return json.loads(json.dumps(value, sort_keys=True))


def _identity_from_verified_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
) -> dict[str, Any]:
    publication = receipt.get("publication_invariant")
    artifacts = receipt.get("artifacts")
    closure = receipt.get("trainable_closure")
    adapter_config = artifacts.get("adapter_config") if isinstance(artifacts, Mapping) else None
    adapter_weights = artifacts.get("adapter_weights") if isinstance(artifacts, Mapping) else None
    if (
        isinstance(receipt.get("schema_version"), bool)
        or not isinstance(receipt.get("schema_version"), int)
        or receipt.get("schema_version") != CHECKPOINT_FIDELITY_SCHEMA_VERSION
        or receipt.get("protocol") != CHECKPOINT_FIDELITY_PROTOCOL
        or receipt.get("status") != "exact_trainable_policy_persisted"
        or not isinstance(publication, Mapping)
        or publication.get("protocol") != QWEN_PUBLICATION_PROTOCOL
        or publication.get("status") != "exact"
        or not isinstance(artifacts, Mapping)
        or not isinstance(adapter_config, Mapping)
        or not isinstance(adapter_weights, Mapping)
        or not isinstance(closure, Mapping)
    ):
        raise PublicationFidelityError(
            "checkpoint-fidelity receipt lacks the exact Qwen2.5-7B publication invariant"
        )
    fields = {
        "receipt_sha256": receipt.get("receipt_sha256"),
        "receipt_file_sha256": receipt_file_sha256,
        "adapter_config_sha256": adapter_config.get("sha256"),
        "adapter_weights_sha256": adapter_weights.get("sha256"),
        "trainable_tensor_inventory_sha256": closure.get("tensor_inventory_sha256"),
        "publication_invariant_sha256": _sha256_json(publication),
    }
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in fields.values()
    ):
        raise PublicationFidelityError(
            "checkpoint-fidelity receipt has incomplete publication artifact hashes"
        )
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "protocol": IDENTITY_PROTOCOL,
        "status": IDENTITY_STATUS,
        **fields,
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return validate_publication_fidelity_identity(identity)


def publication_fidelity_identity(
    checkpoint_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Live-verify a checkpoint and derive its path-free publication identity."""

    root = Path(checkpoint_dir).expanduser().resolve()
    receipt_path = root / RECEIPT_FILENAME
    try:
        receipt_file_sha256 = _stable_regular_file_sha256(receipt_path)
        receipt = verify_checkpoint_fidelity(root)
        # Refuse a pathname swap after the verifier inspected the live receipt.
        if _stable_regular_file_sha256(receipt_path) != receipt_file_sha256:
            raise PublicationFidelityError(
                "checkpoint-fidelity receipt changed during live verification"
            )
        return _identity_from_verified_receipt(
            receipt,
            receipt_file_sha256=receipt_file_sha256,
        )
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(str(exc)) from exc


def validate_fixed_kl_reference_identity(value: object) -> dict[str, Any]:
    """Validate the path-free prelaunch commitment for an exact PEFT reference."""

    if not isinstance(value, dict) or set(value) != FIXED_KL_IDENTITY_KEYS:
        raise PublicationFidelityError("fixed-KL reference identity has an invalid schema")
    unsigned = dict(value)
    stored_identity = unsigned.pop("identity_sha256", None)
    if not isinstance(stored_identity, str) or stored_identity != _sha256_json(unsigned):
        raise PublicationFidelityError("fixed-KL reference identity self-hash mismatch")
    if (
        isinstance(value.get("schema_version"), bool)
        or not isinstance(value.get("schema_version"), int)
        or value.get("schema_version") != FIXED_KL_IDENTITY_SCHEMA_VERSION
        or value.get("protocol") != FIXED_KL_IDENTITY_PROTOCOL
        or value.get("status") != FIXED_KL_IDENTITY_STATUS
        or isinstance(value.get("runtime_schema_version"), bool)
        or value.get("runtime_schema_version") != exact_peft_kl.SCHEMA_VERSION
        or value.get("runtime_protocol") != exact_peft_kl.PUBLICATION_PROTOCOL
        or value.get("runtime_status") != exact_peft_kl.PUBLICATION_STATUS
        or value.get("runtime_receipt_filename") != exact_peft_kl.RECEIPT_FILENAME
        or value.get("verifier_protocol") != exact_peft_kl.VERIFIER_PROTOCOL
        or value.get("post_step_schema_version") != exact_peft_kl_post_step.SCHEMA_VERSION
        or value.get("post_step_protocol") != exact_peft_kl_post_step.PUBLICATION_PROTOCOL
        or value.get("post_step_status") != exact_peft_kl_post_step.PUBLICATION_STATUS
        or value.get("post_step_step1_receipt_filename")
        != exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME
        or value.get("post_step_receipt_filename")
        != exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME
        or value.get("post_step_verifier_protocol") != exact_peft_kl_post_step.VERIFIER_PROTOCOL
        or value.get("parameterization") != "continue_same_bias_all_sft_adapter"
        or value.get("policy_adapter_autocast_dtype") is not True
        or value.get("reference_adapter_autocast_dtype") is not True
    ):
        raise PublicationFidelityError(
            "fixed-KL reference identity is not the publication-exact PEFT source contract"
        )
    digest_fields = FIXED_KL_IDENTITY_KEYS - {
        "schema_version",
        "protocol",
        "status",
        "runtime_schema_version",
        "runtime_protocol",
        "runtime_status",
        "runtime_receipt_filename",
        "verifier_protocol",
        "post_step_schema_version",
        "post_step_protocol",
        "post_step_status",
        "post_step_step1_receipt_filename",
        "post_step_receipt_filename",
        "post_step_verifier_protocol",
        "parameterization",
        "saved_adapter_dtypes",
        "saved_lora_dtype",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
        "source_checkpoint_fidelity",
    }
    for field in digest_fields:
        if not isinstance(value.get(field), str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise PublicationFidelityError(f"fixed-KL reference identity has invalid {field}")
    if (
        not isinstance(value.get("saved_adapter_dtypes"), list)
        or not value["saved_adapter_dtypes"]
        or any(not isinstance(dtype, str) or not dtype for dtype in value["saved_adapter_dtypes"])
        or value["saved_adapter_dtypes"] != sorted(set(value["saved_adapter_dtypes"]))
        or value.get("saved_lora_dtype") != "F32"
        or "F32" not in value["saved_adapter_dtypes"]
    ):
        raise PublicationFidelityError(
            "fixed-KL reference identity lacks exact saved adapter dtype evidence"
        )
    source_identity = validate_publication_fidelity_identity(
        value.get("source_checkpoint_fidelity")
    )
    canonical = dict(value)
    canonical["source_checkpoint_fidelity"] = source_identity
    return json.loads(json.dumps(canonical, sort_keys=True))


def _portable_exact_source_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    adapter_tree = binding.get("adapter_tree")
    base_tree = binding.get("base_model_tree")
    tokenizer_tree = binding.get("tokenizer_tree")
    if not all(isinstance(value, Mapping) for value in (adapter_tree, base_tree, tokenizer_tree)):
        raise PublicationFidelityError("exact PEFT KL source binding lacks immutable trees")
    projection = {
        "method": binding.get("method"),
        "claim_scope": binding.get("claim_scope"),
        "adapter_tree_sha256": adapter_tree.get("tree_sha256"),
        "base_model_tree_sha256": base_tree.get("tree_sha256"),
        "tokenizer_tree_sha256": tokenizer_tree.get("tree_sha256"),
        "authoritative_receipt_protocol": binding.get("authoritative_receipt_protocol"),
        "authoritative_receipt_sha256": binding.get("authoritative_receipt_sha256"),
        "authoritative_receipt_file_sha256": binding.get("authoritative_receipt_file_sha256"),
        "adapter_config_sha256": binding.get("adapter_config_sha256"),
        "saved_adapter_tensor_inventory_sha256": binding.get(
            "saved_adapter_tensor_inventory_sha256"
        ),
        "saved_adapter_dtypes": binding.get("saved_adapter_dtypes"),
        "saved_lora_dtype": binding.get("saved_lora_dtype"),
        "policy_adapter_autocast_dtype": binding.get("policy_adapter_autocast_dtype"),
        "reference_adapter_autocast_dtype": binding.get("reference_adapter_autocast_dtype"),
    }
    if (
        projection["method"] != "immutable_local_base_adapter_tokenizer_trees_v2"
        or projection["claim_scope"] != "publication"
    ):
        raise PublicationFidelityError("exact PEFT KL source binding is not publication-exact")
    for field, value in projection.items():
        if field in {
            "method",
            "claim_scope",
            "authoritative_receipt_protocol",
            "saved_adapter_dtypes",
            "saved_lora_dtype",
            "policy_adapter_autocast_dtype",
            "reference_adapter_autocast_dtype",
        }:
            continue
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise PublicationFidelityError(f"exact PEFT KL source binding has invalid {field}")
    if (
        not isinstance(projection["saved_adapter_dtypes"], list)
        or projection["saved_adapter_dtypes"] != sorted(set(projection["saved_adapter_dtypes"]))
        or projection["saved_lora_dtype"] != "F32"
        or "F32" not in projection["saved_adapter_dtypes"]
        or projection["policy_adapter_autocast_dtype"] is not True
        or projection["reference_adapter_autocast_dtype"] is not True
    ):
        raise PublicationFidelityError(
            "exact PEFT KL source binding lacks exact saved adapter dtype evidence"
        )
    return projection


def _exact_verifier_implementation_sha256() -> str:
    module_path = Path(str(exact_peft_kl.__file__)).resolve(strict=True)
    return _stable_regular_file_sha256(module_path)


def _post_step_verifier_implementation_sha256() -> str:
    module_path = Path(str(exact_peft_kl_post_step.__file__)).resolve(strict=True)
    return _stable_regular_file_sha256(module_path)


def _fixed_kl_identity_from_source_binding(
    binding: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    portable = _portable_exact_source_binding(binding)
    source = validate_publication_fidelity_identity(dict(source_identity))
    if (
        portable["authoritative_receipt_protocol"] != CHECKPOINT_FIDELITY_PROTOCOL
        or portable["authoritative_receipt_sha256"] != source["receipt_sha256"]
        or portable["authoritative_receipt_file_sha256"] != source["receipt_file_sha256"]
        or portable["adapter_config_sha256"] != source["adapter_config_sha256"]
    ):
        raise PublicationFidelityError(
            "exact PEFT KL source binding disagrees with checkpoint fidelity"
        )
    identity: dict[str, Any] = {
        "schema_version": FIXED_KL_IDENTITY_SCHEMA_VERSION,
        "protocol": FIXED_KL_IDENTITY_PROTOCOL,
        "status": FIXED_KL_IDENTITY_STATUS,
        "runtime_schema_version": exact_peft_kl.SCHEMA_VERSION,
        "runtime_protocol": exact_peft_kl.PUBLICATION_PROTOCOL,
        "runtime_status": exact_peft_kl.PUBLICATION_STATUS,
        "runtime_receipt_filename": exact_peft_kl.RECEIPT_FILENAME,
        "verifier_protocol": exact_peft_kl.VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": _exact_verifier_implementation_sha256(),
        "post_step_schema_version": exact_peft_kl_post_step.SCHEMA_VERSION,
        "post_step_protocol": exact_peft_kl_post_step.PUBLICATION_PROTOCOL,
        "post_step_status": exact_peft_kl_post_step.PUBLICATION_STATUS,
        "post_step_step1_receipt_filename": exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        "post_step_receipt_filename": exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
        "post_step_verifier_protocol": exact_peft_kl_post_step.VERIFIER_PROTOCOL,
        "post_step_verifier_implementation_sha256": (_post_step_verifier_implementation_sha256()),
        "parameterization": "continue_same_bias_all_sft_adapter",
        "authoritative_receipt_sha256": portable["authoritative_receipt_sha256"],
        "authoritative_receipt_file_sha256": portable["authoritative_receipt_file_sha256"],
        "adapter_config_sha256": portable["adapter_config_sha256"],
        "adapter_tree_sha256": portable["adapter_tree_sha256"],
        "base_model_tree_sha256": portable["base_model_tree_sha256"],
        "tokenizer_tree_sha256": portable["tokenizer_tree_sha256"],
        "saved_adapter_tensor_inventory_sha256": portable["saved_adapter_tensor_inventory_sha256"],
        "saved_adapter_dtypes": portable["saved_adapter_dtypes"],
        "saved_lora_dtype": portable["saved_lora_dtype"],
        "policy_adapter_autocast_dtype": portable["policy_adapter_autocast_dtype"],
        "reference_adapter_autocast_dtype": portable["reference_adapter_autocast_dtype"],
        "source_checkpoint_fidelity": source,
        "source_binding_sha256": _sha256_json(portable),
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return validate_fixed_kl_reference_identity(identity)


def fixed_kl_reference_identity(
    model_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Live-authorize the immutable same-adapter source used by an exact PEFT pair."""

    model = Path(model_path).expanduser().absolute()
    receipt_file = Path(receipt_path).expanduser().absolute()
    try:
        receipt_file_sha256 = _stable_regular_file_sha256(receipt_file)
        receipt = verify_checkpoint_fidelity(model)
        source_identity = _identity_from_verified_receipt(
            receipt,
            receipt_file_sha256=receipt_file_sha256,
        )
        binding = exact_peft_kl.build_source_binding(
            policy_adapter_path=model,
            reference_adapter_path=model,
            authoritative_receipt_path=receipt_file,
            authoritative_receipt=receipt,
            tokenizer_path=model,
            claim_scope="publication",
        )
        exact_peft_kl.verify_source_binding_unchanged(binding)
        if _stable_regular_file_sha256(receipt_file) != receipt_file_sha256:
            raise PublicationFidelityError(
                "exact PEFT KL source receipt changed during live verification"
            )
        return _fixed_kl_identity_from_source_binding(
            binding,
            source_identity=source_identity,
        )
    except (CheckpointFidelityError, exact_peft_kl.ExactPeftKLReferenceError) as exc:
        raise PublicationFidelityError(str(exc)) from exc


def validate_exact_peft_kl_source_reference(value: object) -> dict[str, Any]:
    """Validate the comparison-common exact source and verifier projection."""

    if not isinstance(value, dict) or set(value) != EXACT_PEFT_KL_SOURCE_REFERENCE_KEYS:
        raise PublicationFidelityError("exact PEFT KL source/reference schema mismatch")
    unsigned = dict(value)
    stored = unsigned.pop("projection_sha256", None)
    if stored != _sha256_json(unsigned):
        raise PublicationFidelityError("exact PEFT KL source/reference self-hash mismatch")
    if (
        value.get("schema_version") != EXACT_PEFT_KL_SOURCE_REFERENCE_SCHEMA_VERSION
        or value.get("protocol") != EXACT_PEFT_KL_SOURCE_REFERENCE_PROTOCOL
        or value.get("status") != EXACT_PEFT_KL_SOURCE_REFERENCE_STATUS
        or value.get("pre_step_schema_version") != exact_peft_kl.SCHEMA_VERSION
        or value.get("pre_step_protocol") != exact_peft_kl.PUBLICATION_PROTOCOL
        or value.get("pre_step_status") != exact_peft_kl.PUBLICATION_STATUS
        or value.get("pre_step_receipt_filename") != exact_peft_kl.RECEIPT_FILENAME
        or value.get("pre_step_verifier_protocol") != exact_peft_kl.VERIFIER_PROTOCOL
        or value.get("pre_step_verifier_implementation_sha256")
        != _exact_verifier_implementation_sha256()
        or value.get("post_step_schema_version") != exact_peft_kl_post_step.SCHEMA_VERSION
        or value.get("post_step_protocol") != exact_peft_kl_post_step.PUBLICATION_PROTOCOL
        or value.get("post_step_status") != exact_peft_kl_post_step.PUBLICATION_STATUS
        or value.get("post_step_step1_receipt_filename")
        != exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME
        or value.get("post_step_receipt_filename")
        != exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME
        or value.get("post_step_verifier_protocol") != exact_peft_kl_post_step.VERIFIER_PROTOCOL
        or value.get("post_step_verifier_implementation_sha256")
        != _post_step_verifier_implementation_sha256()
        or value.get("policy_adapter_autocast_dtype") is not True
        or value.get("reference_adapter_autocast_dtype") is not True
    ):
        raise PublicationFidelityError(
            "exact PEFT KL source/reference projection is not publication-exact"
        )
    for field in EXACT_PEFT_KL_SOURCE_REFERENCE_KEYS - {
        "schema_version",
        "protocol",
        "status",
        "pre_step_schema_version",
        "pre_step_protocol",
        "pre_step_status",
        "pre_step_receipt_filename",
        "pre_step_verifier_protocol",
        "post_step_schema_version",
        "post_step_protocol",
        "post_step_status",
        "post_step_step1_receipt_filename",
        "post_step_receipt_filename",
        "post_step_verifier_protocol",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
    }:
        if not isinstance(value.get(field), str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise PublicationFidelityError(
                f"exact PEFT KL source/reference projection has invalid {field}"
            )
    return json.loads(json.dumps(value, sort_keys=True))


def exact_peft_kl_source_reference(
    fixed_kl_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the outcome-independent source/reference verifier commitment."""

    fixed = validate_fixed_kl_reference_identity(dict(fixed_kl_reference))
    source_fidelity = fixed["source_checkpoint_fidelity"]
    projection: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_SOURCE_REFERENCE_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_SOURCE_REFERENCE_PROTOCOL,
        "status": EXACT_PEFT_KL_SOURCE_REFERENCE_STATUS,
        "fixed_kl_reference_identity_sha256": fixed["identity_sha256"],
        "source_checkpoint_fidelity_identity_sha256": source_fidelity["identity_sha256"],
        "source_binding_sha256": fixed["source_binding_sha256"],
        "pre_step_schema_version": fixed["runtime_schema_version"],
        "pre_step_protocol": fixed["runtime_protocol"],
        "pre_step_status": fixed["runtime_status"],
        "pre_step_receipt_filename": fixed["runtime_receipt_filename"],
        "pre_step_verifier_protocol": fixed["verifier_protocol"],
        "pre_step_verifier_implementation_sha256": fixed["verifier_implementation_sha256"],
        "post_step_schema_version": fixed["post_step_schema_version"],
        "post_step_protocol": fixed["post_step_protocol"],
        "post_step_status": fixed["post_step_status"],
        "post_step_step1_receipt_filename": fixed["post_step_step1_receipt_filename"],
        "post_step_receipt_filename": fixed["post_step_receipt_filename"],
        "post_step_verifier_protocol": fixed["post_step_verifier_protocol"],
        "post_step_verifier_implementation_sha256": fixed[
            "post_step_verifier_implementation_sha256"
        ],
        "policy_adapter_autocast_dtype": fixed["policy_adapter_autocast_dtype"],
        "reference_adapter_autocast_dtype": fixed["reference_adapter_autocast_dtype"],
    }
    projection["projection_sha256"] = _sha256_json(projection)
    return validate_exact_peft_kl_source_reference(projection)


def validate_exact_peft_kl_runtime_identity(value: object) -> dict[str, Any]:
    """Validate one portable post-Trainer exact-pair runtime identity."""

    if not isinstance(value, dict) or set(value) != EXACT_PEFT_KL_RUNTIME_IDENTITY_KEYS:
        raise PublicationFidelityError("exact PEFT KL runtime identity has an invalid schema")
    unsigned = dict(value)
    stored_identity = unsigned.pop("identity_sha256", None)
    if stored_identity != _sha256_json(unsigned):
        raise PublicationFidelityError("exact PEFT KL runtime identity self-hash mismatch")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != EXACT_PEFT_KL_RUNTIME_IDENTITY_SCHEMA_VERSION
        or value.get("protocol") != EXACT_PEFT_KL_RUNTIME_IDENTITY_PROTOCOL
        or value.get("status") != EXACT_PEFT_KL_RUNTIME_IDENTITY_STATUS
        or isinstance(value.get("runtime_schema_version"), bool)
        or value.get("runtime_schema_version") != exact_peft_kl.SCHEMA_VERSION
        or value.get("runtime_protocol") != exact_peft_kl.PUBLICATION_PROTOCOL
        or value.get("runtime_status") != exact_peft_kl.PUBLICATION_STATUS
        or value.get("verifier_protocol") != exact_peft_kl.VERIFIER_PROTOCOL
        or value.get("training_dtype") not in {"float16", "bfloat16", "float32"}
        or isinstance(value.get("training_beta"), bool)
        or not isinstance(value.get("training_beta"), (int, float))
        or not math.isfinite(float(value["training_beta"]))
        or float(value["training_beta"]) <= 0.0
        or value.get("training_sync_ref_model") is not False
        or value.get("training_use_vllm") is not False
        or value.get("training_parameterization") != "continue_same_bias_all_sft_adapter"
        or value.get("training_policy_adapter_autocast_dtype") is not True
        or value.get("training_reference_adapter_autocast_dtype") is not True
        or isinstance(value.get("distributed_world_size"), bool)
        or not isinstance(value.get("distributed_world_size"), int)
        or int(value["distributed_world_size"]) <= 0
    ):
        raise PublicationFidelityError("exact PEFT KL runtime identity is not publication-exact")
    digest_fields = EXACT_PEFT_KL_RUNTIME_IDENTITY_KEYS - {
        "schema_version",
        "protocol",
        "status",
        "runtime_schema_version",
        "runtime_protocol",
        "runtime_status",
        "verifier_protocol",
        "fixed_kl_reference",
        "training_dtype",
        "training_beta",
        "training_sync_ref_model",
        "training_use_vllm",
        "training_parameterization",
        "training_policy_adapter_autocast_dtype",
        "training_reference_adapter_autocast_dtype",
        "distributed_world_size",
    }
    for field in digest_fields:
        if not isinstance(value.get(field), str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise PublicationFidelityError(f"exact PEFT KL runtime identity has invalid {field}")
    source = validate_fixed_kl_reference_identity(value.get("fixed_kl_reference"))
    if (
        value.get("verifier_implementation_sha256") != source["verifier_implementation_sha256"]
        or value.get("training_policy_adapter_autocast_dtype")
        is not source["policy_adapter_autocast_dtype"]
        or value.get("training_reference_adapter_autocast_dtype")
        is not source["reference_adapter_autocast_dtype"]
    ):
        raise PublicationFidelityError(
            "exact PEFT KL runtime/source loader or verifier commitments differ"
        )
    canonical = dict(value)
    canonical["fixed_kl_reference"] = source
    return json.loads(json.dumps(canonical, sort_keys=True))


def exact_peft_kl_runtime_identity(
    receipt_path: str | os.PathLike[str],
    *,
    expected_fixed_kl_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Live-reopen a runtime receipt and derive its path-free publication identity."""

    path = Path(receipt_path).expanduser().absolute()
    try:
        file_sha256 = _stable_regular_file_sha256(path)
        receipt = exact_peft_kl.inspect_runtime_receipt(
            path,
            claim_scope="publication",
            rehash_sources=True,
        )
        if _stable_regular_file_sha256(path) != file_sha256:
            raise PublicationFidelityError(
                "exact PEFT KL runtime receipt changed during live verification"
            )
    except exact_peft_kl.ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(str(exc)) from exc
    source_binding = receipt.get("source_binding")
    verifier = receipt.get("verifier")
    training = receipt.get("training_contract")
    runtime_pair = receipt.get("runtime_pair")
    functional = (
        runtime_pair.get("functional_equivalence") if isinstance(runtime_pair, Mapping) else None
    )
    distributed = receipt.get("distributed")
    if not all(
        isinstance(value, Mapping)
        for value in (source_binding, verifier, training, runtime_pair, functional, distributed)
    ):
        raise PublicationFidelityError("exact PEFT KL runtime receipt is incomplete")
    fixed = fixed_kl_reference_identity(
        str(source_binding.get("adapter_path", "")),
        str(source_binding.get("authoritative_receipt_path", "")),
    )
    if expected_fixed_kl_reference is not None and fixed != validate_fixed_kl_reference_identity(
        dict(expected_fixed_kl_reference)
    ):
        raise PublicationFidelityError(
            "exact PEFT KL runtime receipt differs from its prelaunch source commitment"
        )
    if (
        verifier.get("implementation_sha256") != fixed["verifier_implementation_sha256"]
        or _sha256_json(_portable_exact_source_binding(source_binding))
        != fixed["source_binding_sha256"]
    ):
        raise PublicationFidelityError(
            "exact PEFT KL runtime receipt disagrees with its source/verifier commitment"
        )
    identity: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_RUNTIME_IDENTITY_PROTOCOL,
        "status": EXACT_PEFT_KL_RUNTIME_IDENTITY_STATUS,
        "runtime_schema_version": exact_peft_kl.SCHEMA_VERSION,
        "runtime_protocol": exact_peft_kl.PUBLICATION_PROTOCOL,
        "runtime_status": exact_peft_kl.PUBLICATION_STATUS,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "receipt_file_sha256": file_sha256,
        "verifier_protocol": verifier.get("protocol"),
        "verifier_implementation_sha256": verifier.get("implementation_sha256"),
        "fixed_kl_reference": fixed,
        "training_dtype": training.get("dtype"),
        "training_beta": float(training.get("beta")),
        "training_sync_ref_model": training.get("sync_ref_model"),
        "training_use_vllm": training.get("use_vllm"),
        "training_parameterization": training.get("parameterization"),
        "training_policy_adapter_autocast_dtype": training.get("policy_adapter_autocast_dtype"),
        "training_reference_adapter_autocast_dtype": training.get(
            "reference_adapter_autocast_dtype"
        ),
        "training_contract_sha256": _sha256_json(training),
        "runtime_pair_tensor_inventory_sha256": runtime_pair.get("tensor_inventory_sha256"),
        "runtime_pair_sha256": _sha256_json(runtime_pair),
        "functional_equivalence_sha256": _sha256_json(functional),
        "distributed_world_size": distributed.get("world_size"),
        "distributed_consensus_digest": distributed.get("consensus_digest"),
        "distributed_sha256": _sha256_json(distributed),
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return validate_exact_peft_kl_runtime_identity(identity)


def exact_peft_kl_runtime_identity_from_receipt_payload(
    receipt_payload: object,
    *,
    receipt_file_sha256: object,
    expected_runtime_identity: object,
) -> dict[str, Any]:
    """Integrity-check an embedded receipt without treating it as live authorization.

    Publication entry points must still call :func:`exact_peft_kl_runtime_identity`
    against the live root and selected-checkpoint files.  This projection exists so
    repair/rescore artifacts can remain relocatable while carrying the exact receipt
    bytes that their already-authorized native evaluation consumed.
    """

    expected = validate_exact_peft_kl_runtime_identity(expected_runtime_identity)
    if not isinstance(receipt_payload, dict) or set(receipt_payload) != _EXACT_PEFT_KL_RECEIPT_KEYS:
        raise PublicationFidelityError("embedded exact PEFT KL receipt has an invalid schema")
    if (
        not isinstance(receipt_file_sha256, str)
        or _SHA256_RE.fullmatch(receipt_file_sha256) is None
    ):
        raise PublicationFidelityError("embedded exact PEFT KL receipt file hash is invalid")
    encoded = (
        json.dumps(receipt_payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    if hashlib.sha256(encoded).hexdigest() != receipt_file_sha256:
        raise PublicationFidelityError("embedded exact PEFT KL receipt file hash mismatch")
    unsigned_receipt = dict(receipt_payload)
    stored_receipt_sha256 = unsigned_receipt.pop("receipt_sha256", None)
    if stored_receipt_sha256 != _sha256_json(unsigned_receipt):
        raise PublicationFidelityError("embedded exact PEFT KL receipt self-hash mismatch")
    if (
        receipt_payload.get("schema_version") != exact_peft_kl.SCHEMA_VERSION
        or receipt_payload.get("protocol") != exact_peft_kl.PUBLICATION_PROTOCOL
        or receipt_payload.get("status") != exact_peft_kl.PUBLICATION_STATUS
        or receipt_payload.get("claim_scope") != "publication"
        or receipt_payload.get("development_only") is not False
        or receipt_payload.get("publication_eligible") is not True
        or receipt_payload.get("sota_eligible") is not True
        or receipt_payload.get("purpose") != "immutable_exact_step_zero_peft_kl_reference"
    ):
        raise PublicationFidelityError("embedded exact PEFT KL receipt is not publication eligible")
    verifier = receipt_payload.get("verifier")
    if verifier != {
        "protocol": exact_peft_kl.VERIFIER_PROTOCOL,
        "implementation": "clasp.models.exact_peft_kl_reference",
        "implementation_sha256": expected["verifier_implementation_sha256"],
    }:
        raise PublicationFidelityError("embedded exact PEFT KL verifier identity mismatch")

    source_binding = receipt_payload.get("source_binding")
    training = receipt_payload.get("training_contract")
    runtime_pair = receipt_payload.get("runtime_pair")
    distributed = receipt_payload.get("distributed")
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != _EXACT_PEFT_KL_SOURCE_BINDING_KEYS
        or source_binding.get("method") != "immutable_local_base_adapter_tokenizer_trees_v2"
        or source_binding.get("claim_scope") != "publication"
    ):
        raise PublicationFidelityError("embedded exact PEFT KL source binding is invalid")
    for path_field in (
        "adapter_path",
        "base_model_path",
        "tokenizer_path",
        "authoritative_receipt_path",
    ):
        raw_path = source_binding.get(path_field)
        if not isinstance(raw_path, str) or not raw_path or not os.path.isabs(raw_path):
            raise PublicationFidelityError(
                f"embedded exact PEFT KL source binding has invalid {path_field}"
            )
    try:
        exact_peft_kl._validate_tree(source_binding.get("adapter_tree"), label="adapter")
        exact_peft_kl._validate_tree(source_binding.get("base_model_tree"), label="base model")
        exact_peft_kl._validate_tree(source_binding.get("tokenizer_tree"), label="tokenizer")
        exact_peft_kl._validate_training_contract(training)
        exact_peft_kl._validate_runtime_pair(runtime_pair)
        exact_peft_kl._validate_distributed(
            distributed,
            source_binding=source_binding,
            runtime_pair=runtime_pair,
            claim_scope="publication",
        )
    except exact_peft_kl.ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(str(exc)) from exc
    assert isinstance(training, Mapping)
    assert isinstance(runtime_pair, Mapping)
    assert isinstance(distributed, Mapping)
    functional = runtime_pair.get("functional_equivalence")
    if not isinstance(functional, Mapping):
        raise PublicationFidelityError("embedded exact PEFT KL functional evidence is missing")
    if functional.get("runtime_dtype") != training.get("dtype"):
        raise PublicationFidelityError(
            "embedded exact PEFT KL runtime probe dtype differs from its training contract"
        )
    fixed = validate_fixed_kl_reference_identity(expected["fixed_kl_reference"])
    portable_source = _portable_exact_source_binding(source_binding)
    if (
        _sha256_json(portable_source) != fixed["source_binding_sha256"]
        or portable_source["authoritative_receipt_sha256"] != fixed["authoritative_receipt_sha256"]
        or portable_source["authoritative_receipt_file_sha256"]
        != fixed["authoritative_receipt_file_sha256"]
        or portable_source["adapter_config_sha256"] != fixed["adapter_config_sha256"]
    ):
        raise PublicationFidelityError(
            "embedded exact PEFT KL receipt differs from its fixed source identity"
        )
    projected: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_RUNTIME_IDENTITY_PROTOCOL,
        "status": EXACT_PEFT_KL_RUNTIME_IDENTITY_STATUS,
        "runtime_schema_version": receipt_payload.get("schema_version"),
        "runtime_protocol": receipt_payload.get("protocol"),
        "runtime_status": receipt_payload.get("status"),
        "receipt_sha256": stored_receipt_sha256,
        "receipt_file_sha256": receipt_file_sha256,
        "verifier_protocol": verifier["protocol"],
        "verifier_implementation_sha256": verifier["implementation_sha256"],
        "fixed_kl_reference": fixed,
        "training_dtype": training.get("dtype"),
        "training_beta": float(training.get("beta")),
        "training_sync_ref_model": training.get("sync_ref_model"),
        "training_use_vllm": training.get("use_vllm"),
        "training_parameterization": training.get("parameterization"),
        "training_policy_adapter_autocast_dtype": training.get("policy_adapter_autocast_dtype"),
        "training_reference_adapter_autocast_dtype": training.get(
            "reference_adapter_autocast_dtype"
        ),
        "training_contract_sha256": _sha256_json(training),
        "runtime_pair_tensor_inventory_sha256": runtime_pair.get("tensor_inventory_sha256"),
        "runtime_pair_sha256": _sha256_json(runtime_pair),
        "functional_equivalence_sha256": _sha256_json(functional),
        "distributed_world_size": distributed.get("world_size"),
        "distributed_consensus_digest": distributed.get("consensus_digest"),
        "distributed_sha256": _sha256_json(distributed),
    }
    projected["identity_sha256"] = _sha256_json(projected)
    projected = validate_exact_peft_kl_runtime_identity(projected)
    if projected != expected:
        raise PublicationFidelityError(
            "embedded exact PEFT KL receipt differs from its portable runtime identity"
        )
    return projected


def validate_exact_peft_kl_post_step_lineage(
    value: object,
    *,
    expected_fixed_kl_reference: object | None = None,
    expected_runtime_identity: object | None = None,
    expected_run_contract_file_sha256: str | None = None,
    expected_terminal_global_step: int | None = None,
    expected_selected_checkpoint_step: int | None = None,
    expected_root_checkpoint_fidelity: object | None = None,
    expected_selected_checkpoint_fidelity: object | None = None,
) -> dict[str, Any]:
    """Validate one path-free, full-run post-optimization exact-KL lineage."""

    if not isinstance(value, dict) or set(value) != EXACT_PEFT_KL_POST_STEP_LINEAGE_KEYS:
        raise PublicationFidelityError("exact PEFT KL post-step lineage schema mismatch")
    unsigned = dict(value)
    stored = unsigned.pop("lineage_sha256", None)
    if stored != _sha256_json(unsigned):
        raise PublicationFidelityError("exact PEFT KL post-step lineage self-hash mismatch")
    terminal_step = value.get("terminal_global_step")
    selected_step = value.get("selected_checkpoint_step")
    if (
        value.get("schema_version") != EXACT_PEFT_KL_POST_STEP_LINEAGE_SCHEMA_VERSION
        or value.get("protocol") != EXACT_PEFT_KL_POST_STEP_LINEAGE_PROTOCOL
        or value.get("status") != EXACT_PEFT_KL_POST_STEP_LINEAGE_STATUS
        or isinstance(terminal_step, bool)
        or not isinstance(terminal_step, int)
        or terminal_step <= 0
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step <= 0
        or selected_step > terminal_step
        or value.get("selected_checkpoint_is_terminal") is not (selected_step == terminal_step)
    ):
        raise PublicationFidelityError("exact PEFT KL post-step chronology mismatch")
    for field in (
        "training_run_contract_file_sha256",
        "exact_peft_kl_runtime_identity_sha256",
        "source_reference_sha256",
        "run_invariant_sha256",
        "lineage_sha256",
    ):
        if not isinstance(value.get(field), str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise PublicationFidelityError(f"exact PEFT KL post-step lineage has invalid {field}")

    source = validate_exact_peft_kl_source_reference(value.get("source_reference"))
    if value.get("source_reference_sha256") != source["projection_sha256"]:
        raise PublicationFidelityError("exact PEFT KL source/reference lineage hash mismatch")
    root_fidelity = validate_publication_fidelity_identity(value.get("root_checkpoint_fidelity"))
    selected_fidelity = validate_publication_fidelity_identity(
        value.get("selected_checkpoint_fidelity")
    )
    try:
        root_step1 = exact_peft_kl_post_step.validate_post_step_reference_identity(
            value.get("root_step1"), claim_scope="publication"
        )
        selected_step1 = exact_peft_kl_post_step.validate_post_step_reference_identity(
            value.get("selected_checkpoint_step1"), claim_scope="publication"
        )
        root_final = exact_peft_kl_post_step.validate_post_step_reference_identity(
            value.get("root_final"), claim_scope="publication"
        )
        selected_post = exact_peft_kl_post_step.validate_post_step_reference_identity(
            value.get("selected_checkpoint_post_step"), claim_scope="publication"
        )
    except exact_peft_kl_post_step.ExactPeftKLPostStepError as exc:
        raise PublicationFidelityError(str(exc)) from exc
    if (
        root_step1["artifact_role"] != "step1"
        or root_step1["global_step"] != 1
        or selected_step1 != root_step1
        or root_final["artifact_role"] != "final"
        or root_final["global_step"] != terminal_step
        or selected_post["artifact_role"] != "checkpoint"
        or selected_post["global_step"] != selected_step
    ):
        raise PublicationFidelityError("exact PEFT KL post-step role/step mismatch")
    identities = (root_step1, selected_step1, root_final, selected_post)
    invariant = value.get("run_invariant")
    if (
        not isinstance(invariant, dict)
        or value.get("run_invariant_sha256") != _sha256_json(invariant)
        or any(identity["invariant"] != invariant for identity in identities)
        or any(
            identity["invariant_sha256"] != value["run_invariant_sha256"] for identity in identities
        )
    ):
        raise PublicationFidelityError("exact PEFT KL post-step within-run invariant mismatch")
    if any(
        identity["verifier_protocol"] != source["post_step_verifier_protocol"]
        or identity["verifier_implementation_sha256"]
        != source["post_step_verifier_implementation_sha256"]
        or identity["training_run_contract_file_sha256"]
        != value["training_run_contract_file_sha256"]
        or identity["training_run_contract_max_steps"] != terminal_step
        for identity in identities
    ):
        raise PublicationFidelityError(
            "exact PEFT KL post-step source/verifier/run-contract mismatch"
        )
    if (
        root_final["policy_trainable_tensor_inventory_sha256"]
        != root_fidelity["trainable_tensor_inventory_sha256"]
        or selected_post["policy_trainable_tensor_inventory_sha256"]
        != selected_fidelity["trainable_tensor_inventory_sha256"]
        or selected_post["changed_trainable_tensor_count"] <= 0
    ):
        raise PublicationFidelityError(
            "exact PEFT KL post-step receipts differ from persisted policy closure"
        )
    if value["selected_checkpoint_is_terminal"] and (
        selected_fidelity != root_fidelity
        or selected_post["policy_trainable_tensor_inventory_sha256"]
        != root_final["policy_trainable_tensor_inventory_sha256"]
    ):
        raise PublicationFidelityError(
            "terminal selected checkpoint differs from terminal run-root policy"
        )

    runtime: dict[str, Any] | None = None
    if expected_runtime_identity is not None:
        runtime = validate_exact_peft_kl_runtime_identity(expected_runtime_identity)
        if value["exact_peft_kl_runtime_identity_sha256"] != runtime["identity_sha256"]:
            raise PublicationFidelityError("post-step lineage differs from exact step-zero runtime")
        if any(
            identity["pre_step_receipt_sha256"] != runtime["receipt_sha256"]
            or identity["pre_step_receipt_file_sha256"] != runtime["receipt_file_sha256"]
            or float(identity["beta"]) != float(runtime["training_beta"])
            or identity["training_dtype"] != runtime["training_dtype"]
            or identity["world_size"] != runtime["distributed_world_size"]
            for identity in identities
        ):
            raise PublicationFidelityError(
                "post-step lineage is not cross-bound to the exact step-zero runtime"
            )
        if source != exact_peft_kl_source_reference(runtime["fixed_kl_reference"]):
            raise PublicationFidelityError(
                "post-step source/reference differs from the step-zero runtime"
            )
    if expected_fixed_kl_reference is not None:
        fixed = validate_fixed_kl_reference_identity(expected_fixed_kl_reference)
        if source != exact_peft_kl_source_reference(fixed):
            raise PublicationFidelityError("post-step lineage differs from fixed source contract")
        if runtime is not None and runtime["fixed_kl_reference"] != fixed:
            raise PublicationFidelityError("step-zero runtime differs from fixed source contract")
    if (
        expected_run_contract_file_sha256 is not None
        and value["training_run_contract_file_sha256"] != expected_run_contract_file_sha256
    ):
        raise PublicationFidelityError("post-step lineage run-contract file hash mismatch")
    if expected_terminal_global_step is not None and terminal_step != expected_terminal_global_step:
        raise PublicationFidelityError("post-step lineage terminal step mismatch")
    if (
        expected_selected_checkpoint_step is not None
        and selected_step != expected_selected_checkpoint_step
    ):
        raise PublicationFidelityError("post-step lineage selected checkpoint step mismatch")
    if expected_root_checkpoint_fidelity is not None and root_fidelity != (
        validate_publication_fidelity_identity(expected_root_checkpoint_fidelity)
    ):
        raise PublicationFidelityError("post-step lineage root checkpoint fidelity mismatch")
    if expected_selected_checkpoint_fidelity is not None and selected_fidelity != (
        validate_publication_fidelity_identity(expected_selected_checkpoint_fidelity)
    ):
        raise PublicationFidelityError("post-step lineage selected checkpoint fidelity mismatch")
    canonical = dict(value)
    canonical.update(
        {
            "source_reference": source,
            "root_checkpoint_fidelity": root_fidelity,
            "selected_checkpoint_fidelity": selected_fidelity,
            "root_step1": root_step1,
            "selected_checkpoint_step1": selected_step1,
            "root_final": root_final,
            "selected_checkpoint_post_step": selected_post,
        }
    )
    return json.loads(json.dumps(canonical, sort_keys=True))


def exact_peft_kl_post_step_lineage_identity(
    run_root: str | os.PathLike[str],
    selected_checkpoint: str | os.PathLike[str],
    *,
    expected_fixed_kl_reference: Mapping[str, Any],
    expected_runtime_identity: Mapping[str, Any],
    training_run_contract_file_sha256: str,
    terminal_global_step: int,
) -> dict[str, Any]:
    """Live-authorize root and selected-checkpoint post-step exact-KL evidence."""

    root = Path(run_root).expanduser().absolute()
    selected = Path(selected_checkpoint).expanduser().absolute()
    match = re.fullmatch(r"checkpoint-(\d+)", selected.name)
    if match is None:
        raise PublicationFidelityError("selected exact-KL artifact is not checkpoint-STEP")
    selected_step = int(match.group(1))
    fixed = validate_fixed_kl_reference_identity(dict(expected_fixed_kl_reference))
    runtime = validate_exact_peft_kl_runtime_identity(dict(expected_runtime_identity))
    root_fidelity = publication_fidelity_identity(root)
    selected_fidelity = publication_fidelity_identity(selected)
    try:
        root_step1 = exact_peft_kl_post_step.post_step_reference_identity(
            root / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_policy_artifact=False,
        )
        selected_step1 = exact_peft_kl_post_step.post_step_reference_identity(
            selected / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=1,
            expected_artifact_role="step1",
            rehash_policy_artifact=False,
        )
        root_final = exact_peft_kl_post_step.post_step_reference_identity(
            root / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=terminal_global_step,
            expected_artifact_role="final",
            expected_policy_trainable_tensor_inventory_sha256=root_fidelity[
                "trainable_tensor_inventory_sha256"
            ],
        )
        selected_post = exact_peft_kl_post_step.post_step_reference_identity(
            selected / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            claim_scope="publication",
            expected_global_step=selected_step,
            expected_artifact_role="checkpoint",
            expected_policy_trainable_tensor_inventory_sha256=selected_fidelity[
                "trainable_tensor_inventory_sha256"
            ],
        )
    except exact_peft_kl_post_step.ExactPeftKLPostStepError as exc:
        raise PublicationFidelityError(str(exc)) from exc
    source = exact_peft_kl_source_reference(fixed)
    payload: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_POST_STEP_LINEAGE_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_POST_STEP_LINEAGE_PROTOCOL,
        "status": EXACT_PEFT_KL_POST_STEP_LINEAGE_STATUS,
        "terminal_global_step": terminal_global_step,
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_is_terminal": selected_step == terminal_global_step,
        "training_run_contract_file_sha256": training_run_contract_file_sha256,
        "exact_peft_kl_runtime_identity_sha256": runtime["identity_sha256"],
        "source_reference": source,
        "source_reference_sha256": source["projection_sha256"],
        "run_invariant": root_step1["invariant"],
        "run_invariant_sha256": root_step1["invariant_sha256"],
        "root_checkpoint_fidelity": root_fidelity,
        "selected_checkpoint_fidelity": selected_fidelity,
        "root_step1": root_step1,
        "selected_checkpoint_step1": selected_step1,
        "root_final": root_final,
        "selected_checkpoint_post_step": selected_post,
    }
    payload["lineage_sha256"] = _sha256_json(payload)
    return validate_exact_peft_kl_post_step_lineage(
        payload,
        expected_fixed_kl_reference=fixed,
        expected_runtime_identity=runtime,
        expected_run_contract_file_sha256=training_run_contract_file_sha256,
        expected_terminal_global_step=terminal_global_step,
        expected_selected_checkpoint_step=selected_step,
        expected_root_checkpoint_fidelity=root_fidelity,
        expected_selected_checkpoint_fidelity=selected_fidelity,
    )


def _normalize_source_config_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationFidelityError("source-config file identity is missing")
    allowed = {"basename", "sha256", "size", "size_bytes"}
    if not set(value).issubset(allowed) or set(value) not in (
        {"basename", "sha256", "size"},
        {"basename", "sha256", "size_bytes"},
    ):
        raise PublicationFidelityError("source-config file identity has an invalid schema")
    basename = value.get("basename")
    size = value.get("size_bytes", value.get("size"))
    sha256 = value.get("sha256")
    if (
        not isinstance(basename, str)
        or not basename
        or os.path.basename(basename) != basename
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
    ):
        raise PublicationFidelityError("source-config file identity is invalid")
    return {"basename": basename, "size_bytes": size, "sha256": sha256}


def _exact_runtime_projection_from_source_config(
    source_config: Mapping[str, Any],
) -> tuple[
    float | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    provenance = source_config.get("experiment_provenance")
    contract = source_config.get("training_run_contract")
    model = provenance.get("model") if isinstance(provenance, Mapping) else None
    training = contract.get("training_config") if isinstance(contract, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or not isinstance(training, Mapping)
        or "beta" not in training
    ):
        return None, None, None, None
    try:
        beta = float(training["beta"])
    except (TypeError, ValueError) as exc:
        raise PublicationFidelityError("source config has invalid exact-KL beta") from exc
    if not math.isfinite(beta) or beta < 0.0:
        raise PublicationFidelityError("source config has invalid exact-KL beta")
    fixed_value = contract.get("fixed_kl_reference")
    runtime_value = model.get("exact_peft_kl_runtime")
    post_step_value = model.get("exact_peft_kl_post_step")
    if beta == 0.0:
        if fixed_value is not None or runtime_value is not None or post_step_value is not None:
            raise PublicationFidelityError("source config beta=0 carries exact PEFT KL identities")
        return beta, None, None, None
    if fixed_value is None and runtime_value is None and post_step_value is None:
        return beta, None, None, None
    fixed = validate_fixed_kl_reference_identity(fixed_value)
    runtime = validate_exact_peft_kl_runtime_identity(runtime_value)
    if runtime["fixed_kl_reference"] != fixed or float(runtime["training_beta"]) != beta:
        raise PublicationFidelityError(
            "source config exact PEFT KL runtime differs from its training contract"
        )
    raw_checkpoint = source_config.get("resolved_model_path")
    match = (
        re.fullmatch(r"checkpoint-(\d+)", Path(raw_checkpoint).name)
        if isinstance(raw_checkpoint, str) and raw_checkpoint
        else None
    )
    if match is None:
        raise PublicationFidelityError("source config exact-KL model is not checkpoint-STEP")
    post_step = validate_exact_peft_kl_post_step_lineage(
        post_step_value,
        expected_fixed_kl_reference=fixed,
        expected_runtime_identity=runtime,
        expected_run_contract_file_sha256=source_config.get("training_run_contract_sha256"),
        expected_selected_checkpoint_step=int(match.group(1)),
    )
    return beta, fixed, runtime, post_step


def _build_exact_peft_kl_receipt_snapshot(
    source_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    beta, fixed, runtime, _post_step = _exact_runtime_projection_from_source_config(source_config)
    if beta is None:
        return None
    raw_root = source_config.get("resolved_run_dir")
    raw_checkpoint = source_config.get("resolved_model_path")
    receipt_paths: list[Path] = []
    if isinstance(raw_root, str) and raw_root:
        receipt_paths.append(
            Path(raw_root).expanduser().absolute() / exact_peft_kl.RECEIPT_FILENAME
        )
    if isinstance(raw_checkpoint, str) and raw_checkpoint:
        receipt_paths.append(
            Path(raw_checkpoint).expanduser().absolute() / exact_peft_kl.RECEIPT_FILENAME
        )
    if beta == 0.0:
        stale = sorted(
            str(path) for path in set(receipt_paths) if path.exists() or path.is_symlink()
        )
        if stale:
            raise PublicationFidelityError(
                f"source config beta=0 live artifacts carry exact PEFT KL receipts: {stale}"
            )
        return None
    if fixed is None and runtime is None:
        # Legacy/development artifacts remain usable for development transforms, but
        # publication aggregation will reject them because no exact snapshot exists.
        return None
    if not isinstance(fixed, dict) or not isinstance(runtime, dict):
        raise PublicationFidelityError("source config beta>0 lacks exact PEFT KL identities")
    if (
        not isinstance(raw_root, str)
        or not raw_root
        or not isinstance(raw_checkpoint, str)
        or not raw_checkpoint
    ):
        raise PublicationFidelityError(
            "source config beta>0 lacks live root/checkpoint paths for exact KL snapshotting"
        )
    unique_paths = sorted(set(receipt_paths))
    payloads: list[bytes] = []
    identities: list[dict[str, Any]] = []
    for path in unique_paths:
        before = _stable_regular_file_bytes(path)
        identity = exact_peft_kl_runtime_identity(
            path,
            expected_fixed_kl_reference=fixed,
        )
        after = _stable_regular_file_bytes(path)
        if before != after or hashlib.sha256(before).hexdigest() != identity["receipt_file_sha256"]:
            raise PublicationFidelityError(
                f"exact PEFT KL receipt changed while snapshotting: {path}"
            )
        payloads.append(before)
        identities.append(identity)
    if not payloads or any(payload != payloads[0] for payload in payloads[1:]):
        raise PublicationFidelityError(
            "root/checkpoint exact PEFT KL receipt bytes are not identical"
        )
    if any(identity != runtime for identity in identities):
        raise PublicationFidelityError(
            "root/checkpoint exact PEFT KL receipts differ from source provenance"
        )
    try:
        receipt_payload = json.loads(payloads[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationFidelityError("exact PEFT KL receipt snapshot is invalid JSON") from exc
    snapshot: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_RECEIPT_SNAPSHOT_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_RECEIPT_SNAPSHOT_PROTOCOL,
        "status": "publication_root_and_checkpoint_verified",
        "receipt_filename": exact_peft_kl.RECEIPT_FILENAME,
        "receipt_size_bytes": len(payloads[0]),
        "receipt_file_sha256": hashlib.sha256(payloads[0]).hexdigest(),
        "receipt_payload": receipt_payload,
        "receipt_payload_sha256": _sha256_json(receipt_payload),
        "runtime_identity": runtime,
        "verified_source_roles": ["run_root", "selected_checkpoint"],
    }
    snapshot["snapshot_sha256"] = _sha256_json(snapshot)
    return validate_exact_peft_kl_receipt_snapshot(
        snapshot,
        expected_runtime_identity=runtime,
    )


def validate_exact_peft_kl_receipt_snapshot(
    value: object,
    *,
    expected_runtime_identity: object,
) -> dict[str, Any]:
    """Validate one portable copy of a live-authorized root/checkpoint receipt."""

    expected = validate_exact_peft_kl_runtime_identity(expected_runtime_identity)
    if not isinstance(value, dict) or set(value) != EXACT_PEFT_KL_RECEIPT_SNAPSHOT_KEYS:
        raise PublicationFidelityError("exact PEFT KL receipt snapshot has an invalid schema")
    unsigned = dict(value)
    stored = unsigned.pop("snapshot_sha256", None)
    if stored != _sha256_json(unsigned):
        raise PublicationFidelityError("exact PEFT KL receipt snapshot self-hash mismatch")
    payload = value.get("receipt_payload")
    if (
        value.get("schema_version") != EXACT_PEFT_KL_RECEIPT_SNAPSHOT_SCHEMA_VERSION
        or value.get("protocol") != EXACT_PEFT_KL_RECEIPT_SNAPSHOT_PROTOCOL
        or value.get("status") != "publication_root_and_checkpoint_verified"
        or value.get("receipt_filename") != exact_peft_kl.RECEIPT_FILENAME
        or value.get("verified_source_roles") != ["run_root", "selected_checkpoint"]
        or not isinstance(payload, dict)
        or value.get("receipt_payload_sha256") != _sha256_json(payload)
    ):
        raise PublicationFidelityError("exact PEFT KL receipt snapshot is incomplete")
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    )
    if (
        isinstance(value.get("receipt_size_bytes"), bool)
        or value.get("receipt_size_bytes") != len(encoded)
        or value.get("receipt_file_sha256") != hashlib.sha256(encoded).hexdigest()
        or value.get("runtime_identity") != expected
    ):
        raise PublicationFidelityError("exact PEFT KL receipt snapshot byte identity mismatch")
    exact_peft_kl_runtime_identity_from_receipt_payload(
        payload,
        receipt_file_sha256=value["receipt_file_sha256"],
        expected_runtime_identity=expected,
    )
    return json.loads(json.dumps(value, sort_keys=True))


def _build_post_step_snapshot_entry(
    paths: list[Path],
    *,
    expected_identity: Mapping[str, Any],
    verified_source_roles: list[str],
) -> dict[str, Any]:
    payloads: list[bytes] = []
    for path in paths:
        before = _stable_regular_file_bytes(path)
        try:
            identity = exact_peft_kl_post_step.post_step_reference_identity(
                path,
                claim_scope="publication",
                expected_global_step=int(expected_identity["global_step"]),
                expected_artifact_role=str(expected_identity["artifact_role"]),
                rehash_policy_artifact=expected_identity["artifact_role"] != "step1",
                expected_policy_trainable_tensor_inventory_sha256=(
                    expected_identity["policy_trainable_tensor_inventory_sha256"]
                    if expected_identity["artifact_role"] != "step1"
                    else None
                ),
            )
        except exact_peft_kl_post_step.ExactPeftKLPostStepError as exc:
            raise PublicationFidelityError(str(exc)) from exc
        after = _stable_regular_file_bytes(path)
        if before != after or identity != dict(expected_identity):
            raise PublicationFidelityError(
                f"post-step exact-KL receipt changed or differs from provenance: {path}"
            )
        payloads.append(before)
    if not payloads or any(payload != payloads[0] for payload in payloads[1:]):
        raise PublicationFidelityError("equivalent post-step source roles differ bytewise")
    try:
        payload = json.loads(payloads[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationFidelityError("post-step exact-KL snapshot is invalid JSON") from exc
    entry = {
        "receipt_filename": paths[0].name,
        "receipt_size_bytes": len(payloads[0]),
        "receipt_file_sha256": hashlib.sha256(payloads[0]).hexdigest(),
        "receipt_payload": payload,
        "receipt_payload_sha256": _sha256_json(payload),
        "identity": dict(expected_identity),
        "verified_source_roles": verified_source_roles,
    }
    return entry


def build_exact_peft_kl_post_step_snapshot(
    source_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Copy the exact native post-step receipts needed by relocatable transforms."""

    beta, _fixed, _runtime, lineage = _exact_runtime_projection_from_source_config(source_config)
    if beta is None or beta == 0.0 or lineage is None:
        return None
    raw_root = source_config.get("resolved_run_dir")
    raw_checkpoint = source_config.get("resolved_model_path")
    if (
        not isinstance(raw_root, str)
        or not raw_root
        or not isinstance(raw_checkpoint, str)
        or not raw_checkpoint
    ):
        raise PublicationFidelityError("post-step snapshot lacks live root/checkpoint paths")
    root = Path(raw_root).expanduser().absolute()
    checkpoint = Path(raw_checkpoint).expanduser().absolute()
    receipts = {
        "step1": _build_post_step_snapshot_entry(
            [
                root / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
                checkpoint / exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            ],
            expected_identity=lineage["root_step1"],
            verified_source_roles=["run_root", "selected_checkpoint"],
        ),
        "root_final": _build_post_step_snapshot_entry(
            [root / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME],
            expected_identity=lineage["root_final"],
            verified_source_roles=["run_root"],
        ),
        "selected_checkpoint": _build_post_step_snapshot_entry(
            [checkpoint / exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME],
            expected_identity=lineage["selected_checkpoint_post_step"],
            verified_source_roles=["selected_checkpoint"],
        ),
    }
    snapshot: dict[str, Any] = {
        "schema_version": EXACT_PEFT_KL_POST_STEP_SNAPSHOT_SCHEMA_VERSION,
        "protocol": EXACT_PEFT_KL_POST_STEP_SNAPSHOT_PROTOCOL,
        "status": "publication_root_and_selected_checkpoint_verified",
        "lineage": lineage,
        "receipts": receipts,
    }
    snapshot["snapshot_sha256"] = _sha256_json(snapshot)
    return validate_exact_peft_kl_post_step_snapshot(snapshot, expected_lineage=lineage)


def validate_exact_peft_kl_post_step_snapshot(
    value: object,
    *,
    expected_lineage: object,
) -> dict[str, Any]:
    """Integrity-check embedded post-step payloads without live authorization."""

    lineage = validate_exact_peft_kl_post_step_lineage(expected_lineage)
    if not isinstance(value, dict) or set(value) != EXACT_PEFT_KL_POST_STEP_SNAPSHOT_KEYS:
        raise PublicationFidelityError("post-step exact-KL snapshot schema mismatch")
    unsigned = dict(value)
    stored = unsigned.pop("snapshot_sha256", None)
    receipts = value.get("receipts")
    if (
        stored != _sha256_json(unsigned)
        or value.get("schema_version") != EXACT_PEFT_KL_POST_STEP_SNAPSHOT_SCHEMA_VERSION
        or value.get("protocol") != EXACT_PEFT_KL_POST_STEP_SNAPSHOT_PROTOCOL
        or value.get("status") != "publication_root_and_selected_checkpoint_verified"
        or value.get("lineage") != lineage
        or not isinstance(receipts, dict)
        or set(receipts) != {"step1", "root_final", "selected_checkpoint"}
    ):
        raise PublicationFidelityError("post-step exact-KL snapshot is incomplete")
    expectations = {
        "step1": (
            lineage["root_step1"],
            exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
            ["run_root", "selected_checkpoint"],
        ),
        "root_final": (
            lineage["root_final"],
            exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            ["run_root"],
        ),
        "selected_checkpoint": (
            lineage["selected_checkpoint_post_step"],
            exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
            ["selected_checkpoint"],
        ),
    }
    for role, (expected_identity, filename, sources) in expectations.items():
        entry = receipts.get(role)
        if not isinstance(entry, dict) or set(entry) != EXACT_PEFT_KL_POST_STEP_SNAPSHOT_ENTRY_KEYS:
            raise PublicationFidelityError(f"post-step exact-KL snapshot {role} schema mismatch")
        payload = entry.get("receipt_payload")
        if not isinstance(payload, dict):
            raise PublicationFidelityError(f"post-step exact-KL snapshot {role} lacks payload")
        encoded = (
            json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
        )
        if (
            entry.get("receipt_filename") != filename
            or entry.get("verified_source_roles") != sources
            or entry.get("receipt_size_bytes") != len(encoded)
            or entry.get("receipt_file_sha256") != hashlib.sha256(encoded).hexdigest()
            or entry.get("receipt_payload_sha256") != _sha256_json(payload)
            or entry.get("identity") != expected_identity
        ):
            raise PublicationFidelityError(
                f"post-step exact-KL snapshot {role} byte identity mismatch"
            )
        try:
            projected = exact_peft_kl_post_step.post_step_reference_identity_from_receipt_payload(
                payload,
                receipt_file_sha256=entry["receipt_file_sha256"],
                claim_scope="publication",
                expected_policy_trainable_tensor_inventory_sha256=(
                    expected_identity["policy_trainable_tensor_inventory_sha256"]
                    if expected_identity["artifact_role"] != "step1"
                    else None
                ),
            )
        except exact_peft_kl_post_step.ExactPeftKLPostStepError as exc:
            raise PublicationFidelityError(str(exc)) from exc
        if projected != expected_identity:
            raise PublicationFidelityError(
                f"post-step exact-KL snapshot {role} differs from lineage"
            )
    return json.loads(json.dumps(value, sort_keys=True))


def build_derived_source_config_snapshot(
    source_config: Mapping[str, Any],
    source_config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a path-independent source-config snapshot for repair/rescore output."""

    normalized_identity = _normalize_source_config_identity(source_config_identity)
    payload = json.loads(json.dumps(dict(source_config), sort_keys=True))
    stored_config_sha256 = payload.get("run_config_sha256")
    unsigned_config = dict(payload)
    unsigned_config.pop("run_config_sha256", None)
    if stored_config_sha256 != _sha256_json(unsigned_config):
        raise PublicationFidelityError("source config snapshot has an invalid self-hash")
    snapshot: dict[str, Any] = {
        "schema_version": DERIVED_SOURCE_CONFIG_SNAPSHOT_SCHEMA_VERSION,
        "protocol": DERIVED_SOURCE_CONFIG_SNAPSHOT_PROTOCOL,
        "source_config_identity": normalized_identity,
        "source_config_payload": payload,
        "source_config_payload_sha256": _sha256_json(payload),
        "exact_peft_kl_runtime_receipt": _build_exact_peft_kl_receipt_snapshot(payload),
        "exact_peft_kl_post_step_receipts": build_exact_peft_kl_post_step_snapshot(payload),
    }
    snapshot["snapshot_sha256"] = _sha256_json(snapshot)
    return validate_derived_source_config_snapshot(
        snapshot,
        expected_source_config_identity=normalized_identity,
    )


def validate_derived_source_config_snapshot(
    value: object,
    *,
    expected_source_config_identity: object | None = None,
) -> dict[str, Any]:
    """Validate a portable, summary-self-hashed native source-config snapshot."""

    if not isinstance(value, dict) or set(value) != DERIVED_SOURCE_CONFIG_SNAPSHOT_KEYS:
        raise PublicationFidelityError("derived source-config snapshot has an invalid schema")
    unsigned = dict(value)
    stored = unsigned.pop("snapshot_sha256", None)
    if stored != _sha256_json(unsigned):
        raise PublicationFidelityError("derived source-config snapshot self-hash mismatch")
    if (
        value.get("schema_version") != DERIVED_SOURCE_CONFIG_SNAPSHOT_SCHEMA_VERSION
        or value.get("protocol") != DERIVED_SOURCE_CONFIG_SNAPSHOT_PROTOCOL
    ):
        raise PublicationFidelityError("derived source-config snapshot protocol mismatch")
    identity = _normalize_source_config_identity(value.get("source_config_identity"))
    if (
        expected_source_config_identity is not None
        and identity != _normalize_source_config_identity(expected_source_config_identity)
    ):
        raise PublicationFidelityError("derived source-config snapshot file identity mismatch")
    payload = value.get("source_config_payload")
    if not isinstance(payload, dict) or value.get("source_config_payload_sha256") != _sha256_json(
        payload
    ):
        raise PublicationFidelityError("derived source-config snapshot payload hash mismatch")
    unsigned_config = dict(payload)
    stored_config_sha256 = unsigned_config.pop("run_config_sha256", None)
    if stored_config_sha256 != _sha256_json(unsigned_config):
        raise PublicationFidelityError("derived source-config payload self-hash mismatch")
    _beta, _fixed, runtime, post_step = _exact_runtime_projection_from_source_config(payload)
    receipt_snapshot = value.get("exact_peft_kl_runtime_receipt")
    if runtime is None:
        if receipt_snapshot is not None:
            raise PublicationFidelityError(
                "derived source-config snapshot carries unexpected exact KL evidence"
            )
    else:
        validate_exact_peft_kl_receipt_snapshot(
            receipt_snapshot,
            expected_runtime_identity=runtime,
        )
    post_step_snapshot = value.get("exact_peft_kl_post_step_receipts")
    if post_step is None:
        if post_step_snapshot is not None:
            raise PublicationFidelityError(
                "derived source-config snapshot carries unexpected post-step evidence"
            )
    else:
        validate_exact_peft_kl_post_step_snapshot(
            post_step_snapshot,
            expected_lineage=post_step,
        )
    return json.loads(json.dumps(value, sort_keys=True))
