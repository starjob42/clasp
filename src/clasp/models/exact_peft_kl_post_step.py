"""Post-step immutability evidence for an explicit exact PEFT KL reference.

The step-zero receipt proves that independently loaded policy and reference
models start from the same numerical object.  This module closes the temporal
gap: after optimization begins it proves that the *actual prepared reference*
still has the exact step-zero tensor inventory and values, is fully frozen,
shares no storage with the policy or optimizer, and belongs to the same
authorized run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clasp.models.checkpoint_fidelity import (
    CheckpointFidelityError,
    _TORCH_TO_SAFETENSORS_DTYPE,
    canonicalize_peft_adapter_key,
    verify_checkpoint_fidelity,
)
from clasp.models.exact_peft_kl_reference import (
    DEVELOPMENT_PROTOCOL as PRE_DEVELOPMENT_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    DEVELOPMENT_STATUS as PRE_DEVELOPMENT_STATUS,
)
from clasp.models.exact_peft_kl_reference import (
    PUBLICATION_PROTOCOL as PRE_PUBLICATION_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    PUBLICATION_STATUS as PRE_PUBLICATION_STATUS,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as PRE_STEP_RECEIPT_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (
    ExactPeftKLReferenceError,
    inspect_runtime_receipt,
)

SCHEMA_VERSION = 2
VERIFIER_PROTOCOL = "clasp-exact-peft-kl-reference-post-step-verifier-v2"
PUBLICATION_PROTOCOL = "clasp-publication-exact-peft-kl-reference-post-step-v2"
DEVELOPMENT_PROTOCOL = "clasp-development-exact-peft-kl-reference-post-step-v2"
PUBLICATION_STATUS = "publication_exact_peft_reference_immutability_verified"
DEVELOPMENT_STATUS = "development_exact_peft_reference_immutability_verified"
STEP1_RECEIPT_FILENAME = "exact_peft_kl_reference_step1.json"
POST_STEP_RECEIPT_FILENAME = "exact_peft_kl_reference_post_step.json"
TRAINING_RUN_CONTRACT_IDENTITY_PROTOCOL = "clasp-training-run-contract-file-identity-v1"
TRAINING_RUN_CONTRACT_IDENTITY_STATUS = "immutable_verified"

_SHA256_CHARS = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "development_only",
        "publication_eligible",
        "sota_eligible",
        "artifact_role",
        "global_step",
        "verifier",
        "pre_step_reference",
        "step1_reference",
        "training_contract",
        "training_run_contract",
        "reference_audit",
        "policy_context",
        "source_binding_unchanged",
        "distributed",
        "receipt_sha256",
    }
)
_TENSOR_KEYS = frozenset({"key", "shape", "dtype", "numel", "value_sha256"})
_INVENTORY_KEYS = frozenset(
    {"tensor_count", "parameter_count", "tensor_inventory_sha256", "tensors"}
)
_PRE_STEP_KEYS = frozenset(
    {
        "protocol",
        "status",
        "receipt_sha256",
        "receipt_file_sha256",
        "reference_tensor_count",
        "reference_parameter_count",
        "reference_tensor_inventory_sha256",
        "model_class",
        "prepared_class",
        "unwrapped_class",
        "active_adapter",
        "model_config_sha256",
        "device_types",
    }
)
_STEP1_KEYS = frozenset(
    {
        "protocol",
        "status",
        "receipt_sha256",
        "receipt_file_sha256",
        "global_step",
        "reference_tensor_inventory_sha256",
        "policy_trainable_tensor_inventory_sha256",
    }
)
_TRAINING_CONTRACT_KEYS = frozenset(
    {
        "dtype",
        "beta",
        "sync_ref_model",
        "use_vllm",
        "parameterization",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
    }
)
_RUN_CONTRACT_KEYS = frozenset(
    {
        "protocol",
        "status",
        "file_sha256",
        "payload_sha256",
        "size_bytes",
        "payload_schema_version",
        "payload_stage",
        "fixed_horizon_max_steps",
        "fixed_horizon_world_size",
    }
)
_REFERENCE_AUDIT_KEYS = frozenset(
    {
        "method",
        "trainer_ref_model_identity",
        "reference_prepared_object_identity",
        "reference_unwrapped_identity",
        "prepared_class",
        "unwrapped_class",
        "active_adapter",
        "model_config_sha256",
        "reference_trainable_tensor_count",
        "reference_eval_mode",
        "reference_input_grad_hook_absent",
        "baseline_inventory",
        "live_inventory",
        "live_equals_pre_step_bitwise",
        "policy_reference_storage_independent",
        "optimizer_excludes_reference_storage",
        "optimizer_matches_policy_trainables",
        "optimizer_parameter_count",
        "device_types",
    }
)
_POLICY_CONTEXT_KEYS = frozenset(
    {
        "method",
        "adapter_name",
        "full_inventory",
        "pre_step_tensor_inventory_sha256",
        "changed_from_pre_step",
        "changed_tensor_count",
        "changed_tensor_keys",
        "changed_tensor_keys_sha256",
        "changed_trainable_tensor_count",
        "changed_trainable_tensor_keys",
        "changed_trainable_tensor_keys_sha256",
        "policy_trainable_tensor_count",
        "policy_trainable_parameter_count",
        "policy_trainable_tensor_inventory_sha256",
        "trainable_inventory",
    }
)

_DISTRIBUTED_KEYS = frozenset(
    {
        "phase",
        "world_size",
        "rank_digests",
        "consensus_digest",
        "rank_cuda_memory",
        "distributed_evidence_sha256",
    }
)
_CUDA_MEMORY_KEYS = frozenset(
    {
        "rank",
        "cuda_available",
        "device_index",
        "physical_device_index",
        "visible_device_token",
        "device_uuid",
        "device_name",
        "driver_version",
        "cuda_runtime_version",
        "total_memory_bytes",
        "nvidia_smi_total_memory_mib",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "artifact_role",
        "global_step",
        "receipt_sha256",
        "receipt_file_sha256",
        "verifier_protocol",
        "verifier_implementation_sha256",
        "pre_step_receipt_sha256",
        "pre_step_receipt_file_sha256",
        "step1_receipt_sha256",
        "step1_receipt_file_sha256",
        "training_dtype",
        "beta",
        "sync_ref_model",
        "use_vllm",
        "parameterization",
        "policy_adapter_autocast_dtype",
        "reference_adapter_autocast_dtype",
        "training_run_contract_file_sha256",
        "training_run_contract_payload_sha256",
        "training_run_contract_size_bytes",
        "training_run_contract_max_steps",
        "training_run_contract_world_size",
        "world_size",
        "distributed_consensus_digest",
        "distributed_evidence_sha256",
        "rank_cuda_memory",
        "reference_model_class",
        "reference_prepared_class",
        "reference_unwrapped_class",
        "reference_active_adapter",
        "reference_model_config_sha256",
        "reference_device_types",
        "reference_tensor_count",
        "reference_parameter_count",
        "reference_tensor_inventory_sha256",
        "reference_trainable_tensor_count",
        "reference_eval_mode",
        "reference_input_grad_hook_absent",
        "trainer_ref_model_identity",
        "reference_prepared_object_identity",
        "reference_unwrapped_identity",
        "policy_reference_storage_independent",
        "optimizer_excludes_reference_storage",
        "optimizer_matches_policy_trainables",
        "policy_trainable_tensor_count",
        "policy_trainable_parameter_count",
        "policy_trainable_tensor_inventory_sha256",
        "policy_changed_from_pre_step",
        "changed_trainable_tensor_count",
        "changed_trainable_tensor_keys_sha256",
        "invariant",
        "invariant_sha256",
        "identity_sha256",
    }
)
_INVARIANT_KEYS = frozenset(
    {
        "schema_version",
        "claim_scope",
        "verifier",
        "pre_step_reference",
        "step1_receipt_sha256",
        "step1_receipt_file_sha256",
        "training_contract",
        "training_run_contract",
        "world_size",
        "rank_cuda_identity",
        "reference_method",
        "reference_prepared_class",
        "reference_unwrapped_class",
        "reference_active_adapter",
        "reference_model_config_sha256",
        "reference_device_types",
        "reference_tensor_count",
        "reference_parameter_count",
        "reference_tensor_inventory_sha256",
        "reference_trainable_tensor_count",
        "reference_eval_mode",
        "reference_input_grad_hook_absent",
        "trainer_ref_model_identity",
        "reference_prepared_object_identity",
        "reference_unwrapped_identity",
        "policy_reference_storage_independent",
        "optimizer_excludes_reference_storage",
        "optimizer_matches_policy_trainables",
        "optimizer_parameter_count",
    }
)


class ExactPeftKLPostStepError(ValueError):
    """Post-step exact-reference evidence is absent or inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExactPeftKLPostStepError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _stable_regular_bytes(path: Path, *, require_read_only: bool) -> bytes:
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExactPeftKLPostStepError(f"cannot open required file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExactPeftKLPostStepError(f"required file is not regular: {path}")
        if require_read_only and stat.S_IMODE(before.st_mode) & 0o222:
            raise ExactPeftKLPostStepError(f"required receipt must be read-only: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            live_path = path.lstat()
        except OSError as exc:
            raise ExactPeftKLPostStepError(f"required file path vanished: {path}: {exc}") from exc
    except OSError as exc:
        raise ExactPeftKLPostStepError(f"cannot read receipt {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity) or any(
        getattr(before, field) != getattr(live_path, field) for field in identity
    ):
        raise ExactPeftKLPostStepError(f"required file changed while read: {path}")
    return b"".join(chunks)


def _regular_read_only_bytes(path: Path) -> bytes:
    return _stable_regular_bytes(path, require_read_only=True)


def _decode_object(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactPeftKLPostStepError(f"invalid JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExactPeftKLPostStepError(f"receipt is not a JSON object: {path}")
    return value


def _implementation_sha256() -> str:
    return _sha256_bytes(
        _stable_regular_bytes(Path(__file__).resolve(strict=True), require_read_only=False)
    )


def training_run_contract_identity(
    path: str | os.PathLike[str],
    *,
    claim_scope: str,
) -> dict[str, Any]:
    """Hash a read-only GRPO run contract using verifier-owned identity labels."""

    contract_path = Path(path).expanduser().absolute()
    raw = _regular_read_only_bytes(contract_path)
    payload = _decode_object(raw, path=contract_path)
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
        or payload.get("stage") != "grpo"
    ):
        raise ExactPeftKLPostStepError("training run contract must be a versioned GRPO contract")
    if claim_scope == "publication" and schema_version != 2:
        raise ExactPeftKLPostStepError("publication GRPO run contract must use schema_version=2")
    if claim_scope not in {"publication", "development"}:
        raise ExactPeftKLPostStepError(f"unsupported claim scope: {claim_scope!r}")
    horizon = payload.get("fixed_horizon")
    if not isinstance(horizon, Mapping) or set(horizon) != {"max_steps", "world_size"}:
        raise ExactPeftKLPostStepError("training run contract lacks an exact fixed horizon")
    for field in ("max_steps", "world_size"):
        observed = horizon.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            raise ExactPeftKLPostStepError(f"training fixed horizon has invalid {field}")
    return {
        "protocol": TRAINING_RUN_CONTRACT_IDENTITY_PROTOCOL,
        "status": TRAINING_RUN_CONTRACT_IDENTITY_STATUS,
        "file_sha256": _sha256_bytes(raw),
        "payload_sha256": _sha256_json(payload),
        "size_bytes": len(raw),
        "payload_schema_version": schema_version,
        "payload_stage": "grpo",
        "fixed_horizon_max_steps": horizon["max_steps"],
        "fixed_horizon_world_size": horizon["world_size"],
    }


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach()
    if value.layout != torch.strided or value.device.type == "meta":
        raise ExactPeftKLPostStepError("live model contains a non-materialized tensor")
    value = value.resolve_conj().resolve_neg().to(device="cpu").contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _tensor_record(key: str, tensor: Any) -> dict[str, Any]:
    return {
        "key": key,
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "value_sha256": _sha256_bytes(_tensor_bytes(tensor)),
    }


def _trainable_matches_full_tensor(full: Mapping[str, Any], trainable: Mapping[str, Any]) -> bool:
    """Compare exact values across the two established dtype encodings.

    Live full-state receipts use e.g. ``torch.float32``; the persistable
    checkpoint closure uses ``F32``. Preserve both inventories and their hashes,
    converting only the dtype tag for this cross-inventory comparison. Older
    torch-tagged closure receipts remain readable. Unknown or genuinely
    different dtypes never compare equal, including equal-width F16/BF16.
    """
    full_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(full["dtype"])
    closure_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(trainable["dtype"], trainable["dtype"])
    return (full_dtype is not None and full_dtype == closure_dtype
            and all(full[field] == trainable[field] for field in ("shape", "numel", "value_sha256")))


def _storage_pointer(tensor: Any) -> int:
    value = tensor.detach()
    if int(value.numel()) == 0:
        return 0
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover
        return int(value.storage().data_ptr())


def _state_inventory(model: Any) -> dict[str, Any]:
    records = [_tensor_record(key, tensor) for key, tensor in sorted(model.state_dict().items())]
    if not records:
        raise ExactPeftKLPostStepError("live model state inventory is empty")
    return {
        "tensor_count": len(records),
        "parameter_count": sum(int(record["numel"]) for record in records),
        "tensor_inventory_sha256": _sha256_json(records),
        "tensors": records,
    }


def _validate_tensor(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _TENSOR_KEYS:
        raise ExactPeftKLPostStepError(f"{label} tensor schema mismatch")
    if (
        not isinstance(record.get("key"), str)
        or not record["key"]
        or not isinstance(record.get("shape"), list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in record["shape"]
        )
        or isinstance(record.get("numel"), bool)
        or not isinstance(record.get("numel"), int)
        or record["numel"] < 0
        or not isinstance(record.get("dtype"), str)
        or not record["dtype"]
        or not _valid_sha256(record.get("value_sha256"))
    ):
        raise ExactPeftKLPostStepError(f"{label} tensor record is invalid")
    product = 1
    for dimension in record["shape"]:
        product *= int(dimension)
    if product != record["numel"]:
        raise ExactPeftKLPostStepError(f"{label} tensor numel/shape mismatch")


def _validate_inventory(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INVENTORY_KEYS:
        raise ExactPeftKLPostStepError(f"{label} inventory schema mismatch")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExactPeftKLPostStepError(f"{label} tensor inventory is empty")
    previous = ""
    for record in tensors:
        _validate_tensor(record, label=label)
        key = str(record["key"])
        if key <= previous:
            raise ExactPeftKLPostStepError(f"{label} tensor keys are not unique/sorted")
        previous = key
    if (
        value.get("tensor_count") != len(tensors)
        or value.get("parameter_count") != sum(int(record["numel"]) for record in tensors)
        or value.get("tensor_inventory_sha256") != _sha256_json(tensors)
    ):
        raise ExactPeftKLPostStepError(f"{label} inventory aggregate mismatch")
    return value


def _model_config_sha256(model: Any) -> str:
    to_dict = getattr(getattr(model, "config", None), "to_dict", None)
    if not callable(to_dict):
        raise ExactPeftKLPostStepError("live reference lacks a serializable model config")
    return _sha256_json(to_dict())


def _active_adapter(model: Any) -> str:
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and len(value) == 1:
        return str(value[0])
    raise ExactPeftKLPostStepError("live reference does not expose one active adapter")


def reference_has_input_grad_hook(model: Any) -> bool:
    """Return whether Transformers installed its input-requires-grad hook."""

    modules = getattr(model, "modules", None)
    candidates = list(modules()) if callable(modules) else [model]
    return any(getattr(module, "_require_grads_hook", None) is not None for module in candidates)


def pre_step_reference_identity(
    path: str | os.PathLike[str],
    *,
    claim_scope: str,
) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().absolute()
    try:
        receipt = inspect_runtime_receipt(receipt_path, claim_scope=claim_scope)
    except ExactPeftKLReferenceError as exc:
        raise ExactPeftKLPostStepError(f"invalid pre-step reference receipt: {exc}") from exc
    raw = _regular_read_only_bytes(receipt_path)
    runtime = receipt["runtime_pair"]
    prepared = runtime["prepared_runtime"]
    return {
        "protocol": receipt["protocol"],
        "status": receipt["status"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": _sha256_bytes(raw),
        "reference_tensor_count": runtime["tensor_count"],
        "reference_parameter_count": runtime["parameter_count"],
        "reference_tensor_inventory_sha256": runtime["tensor_inventory_sha256"],
        "model_class": runtime["model_class"],
        "prepared_class": prepared["reference_prepared_class"],
        "unwrapped_class": prepared["reference_unwrapped_class"],
        "active_adapter": runtime["active_adapter"],
        "model_config_sha256": runtime["model_config_sha256"],
        "device_types": list(prepared["device_types"]),
    }


def audit_live_post_step_reference(
    policy: Any,
    reference: Any,
    *,
    pre_step_receipt: Mapping[str, Any],
    trainable_closure: Mapping[str, Any],
    optimizer: Any,
    runtime_identity: Mapping[str, Any],
    require_policy_update: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit exact frozen-reference state and the contemporaneous live policy."""

    if policy is reference:
        raise ExactPeftKLPostStepError("policy and reference are the same Python object")
    expected_runtime_identity = {
        "trainer_ref_model_identity": True,
        "reference_prepared_object_identity": True,
        "reference_unwrapped_identity": True,
    }
    if any(
        runtime_identity.get(key) is not value for key, value in expected_runtime_identity.items()
    ):
        raise ExactPeftKLPostStepError("prepared reference object identity changed")
    prepared_class = runtime_identity.get("prepared_class")
    unwrapped_class = runtime_identity.get("unwrapped_class")
    if (
        not isinstance(prepared_class, str)
        or not prepared_class
        or not isinstance(unwrapped_class, str)
        or not unwrapped_class
    ):
        raise ExactPeftKLPostStepError("prepared reference classes are missing")
    actual_unwrapped_class = f"{reference.__class__.__module__}.{reference.__class__.__qualname__}"
    if unwrapped_class != actual_unwrapped_class:
        raise ExactPeftKLPostStepError("unwrapped reference class identity is inconsistent")
    reference_trainable = [
        name for name, value in reference.named_parameters() if value.requires_grad
    ]
    if reference_trainable:
        raise ExactPeftKLPostStepError(
            f"post-step reference has trainable tensors: {reference_trainable[:12]}"
        )
    modules = getattr(reference, "modules", None)
    reference_modules = list(modules()) if callable(modules) else [reference]
    if any(bool(getattr(module, "training", True)) for module in reference_modules):
        raise ExactPeftKLPostStepError("post-step reference is not in eval mode")
    if reference_has_input_grad_hook(reference):
        raise ExactPeftKLPostStepError("post-step reference has an input-requires-grad hook")
    baseline_records = pre_step_receipt.get("runtime_pair", {}).get("tensors")
    if not isinstance(baseline_records, list) or not baseline_records:
        raise ExactPeftKLPostStepError("pre-step receipt lacks baseline reference tensors")
    baseline_inventory = {
        "tensor_count": pre_step_receipt["runtime_pair"]["tensor_count"],
        "parameter_count": pre_step_receipt["runtime_pair"]["parameter_count"],
        "tensor_inventory_sha256": pre_step_receipt["runtime_pair"]["tensor_inventory_sha256"],
        "tensors": baseline_records,
    }
    _validate_inventory(baseline_inventory, label="pre-step reference")
    live_reference = _state_inventory(reference)
    if live_reference != baseline_inventory:
        baseline_by_key = {record["key"]: record for record in baseline_records}
        live_by_key = {record["key"]: record for record in live_reference["tensors"]}
        changed = sorted(
            key
            for key in set(baseline_by_key) | set(live_by_key)
            if baseline_by_key.get(key) != live_by_key.get(key)
        )
        raise ExactPeftKLPostStepError(
            f"post-step reference differs from step zero: changed={changed[:12]}"
        )

    policy_state = policy.state_dict()
    reference_state = reference.state_dict()
    policy_storage = {
        (str(value.device), _storage_pointer(value))
        for value in policy_state.values()
        if int(value.numel()) > 0
    }
    reference_state_storage = {
        (str(value.device), _storage_pointer(value))
        for value in reference_state.values()
        if int(value.numel()) > 0
    }
    shared_storage = policy_storage & reference_state_storage
    if shared_storage:
        raise ExactPeftKLPostStepError("post-step policy/reference share tensor storage")

    reference_parameters = list(reference.parameters())
    reference_ids = {id(value) for value in reference_parameters}
    reference_storage = {
        _storage_pointer(value) for value in reference_parameters if int(value.numel()) > 0
    }
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise ExactPeftKLPostStepError("post-step optimizer has no parameter groups")
    optimizer_parameters = [value for group in groups for value in group.get("params", [])]
    if not optimizer_parameters:
        raise ExactPeftKLPostStepError("post-step optimizer has no parameters")
    leaked = [
        value
        for value in optimizer_parameters
        if id(value) in reference_ids
        or (int(value.numel()) > 0 and _storage_pointer(value) in reference_storage)
    ]
    if leaked:
        raise ExactPeftKLPostStepError("optimizer contains frozen-reference storage")
    optimizer_ids = [id(value) for value in optimizer_parameters]
    optimizer_storage = [
        (str(value.device), _storage_pointer(value))
        for value in optimizer_parameters
        if int(value.numel()) > 0
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)) or len(optimizer_storage) != len(
        set(optimizer_storage)
    ):
        raise ExactPeftKLPostStepError("optimizer contains duplicate parameter/storage entries")
    policy_trainable_parameters = [value for value in policy.parameters() if value.requires_grad]
    policy_trainable_ids = {id(value) for value in policy_trainable_parameters}
    policy_trainable_storage = {
        (str(value.device), _storage_pointer(value))
        for value in policy_trainable_parameters
        if int(value.numel()) > 0
    }
    if (
        set(optimizer_ids) != policy_trainable_ids
        or set(optimizer_storage) != policy_trainable_storage
    ):
        raise ExactPeftKLPostStepError(
            "optimizer parameter groups are not the exact policy requires_grad closure"
        )

    policy_inventory = _state_inventory(policy)
    baseline_by_key = {record["key"]: record for record in baseline_records}
    policy_by_key = {record["key"]: record for record in policy_inventory["tensors"]}
    if set(policy_by_key) != set(baseline_by_key):
        raise ExactPeftKLPostStepError("post-step policy state keys differ from step zero")
    raw_changed_keys = sorted(
        key for key in policy_by_key if policy_by_key[key] != baseline_by_key[key]
    )
    tensors = trainable_closure.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExactPeftKLPostStepError("post-step trainable closure is empty")
    adapter_name = str(trainable_closure.get("adapter_name", "default"))
    canonical_policy: dict[str, dict[str, Any]] = {}
    canonical_baseline: dict[str, dict[str, Any]] = {}
    for raw_key, record in policy_by_key.items():
        canonical = canonicalize_peft_adapter_key(raw_key, adapter_name)
        if canonical in canonical_policy:
            raise ExactPeftKLPostStepError(f"canonical policy state collision: {canonical}")
        canonical_policy[canonical] = record
        canonical_baseline[canonical] = baseline_by_key[raw_key]
    changed_keys = sorted(
        canonicalize_peft_adapter_key(raw_key, adapter_name) for raw_key in raw_changed_keys
    )
    trainable_keys = [str(record.get("key", "")) for record in tensors]
    if len(trainable_keys) != len(set(trainable_keys)):
        raise ExactPeftKLPostStepError("post-step trainable closure has duplicate keys")
    missing_trainable = sorted(set(trainable_keys) - set(canonical_policy))
    if missing_trainable:
        raise ExactPeftKLPostStepError(
            f"trainable closure is absent from full policy state: {missing_trainable[:12]}"
        )
    changed_trainable = sorted(
        key
        for key in trainable_keys
        if canonical_policy[key]["value_sha256"] != canonical_baseline[key]["value_sha256"]
    )
    if require_policy_update and not changed_trainable:
        raise ExactPeftKLPostStepError("post-step gate observed no changed trainable policy tensor")
    closure_digest = trainable_closure.get("tensor_inventory_sha256")
    if closure_digest != _sha256_json(tensors):
        raise ExactPeftKLPostStepError("post-step trainable closure aggregate mismatch")
    trainable_inventory = {
        "tensor_count": trainable_closure.get("tensor_count"),
        "parameter_count": trainable_closure.get("parameter_count"),
        "tensor_inventory_sha256": closure_digest,
        "tensors": [dict(record) for record in tensors],
    }
    _validate_inventory(trainable_inventory, label="policy trainable")
    for record in tensors:
        full = canonical_policy[str(record["key"])]
        if not _trainable_matches_full_tensor(full, record):
            raise ExactPeftKLPostStepError(
                f"trainable closure differs from full live policy: {record['key']}"
            )
    if set(changed_keys) != set(changed_trainable):
        raise ExactPeftKLPostStepError(
            "live policy changed outside the persistable trainable tensor closure"
        )

    reference_audit = {
        "method": "live_prepared_reference_immutability_v1",
        **expected_runtime_identity,
        "prepared_class": prepared_class,
        "unwrapped_class": unwrapped_class,
        "active_adapter": _active_adapter(reference),
        "model_config_sha256": _model_config_sha256(reference),
        "reference_trainable_tensor_count": 0,
        "reference_eval_mode": True,
        "reference_input_grad_hook_absent": True,
        "baseline_inventory": baseline_inventory,
        "live_inventory": live_reference,
        "live_equals_pre_step_bitwise": True,
        "policy_reference_storage_independent": True,
        "optimizer_excludes_reference_storage": True,
        "optimizer_matches_policy_trainables": True,
        "optimizer_parameter_count": len(optimizer_parameters),
        "device_types": sorted({str(value.device.type) for value in reference_state.values()}),
    }
    policy_context = {
        "method": "full_live_policy_state_and_trainable_closure_v1",
        "adapter_name": adapter_name,
        "full_inventory": policy_inventory,
        "pre_step_tensor_inventory_sha256": baseline_inventory["tensor_inventory_sha256"],
        "changed_from_pre_step": bool(changed_keys),
        "changed_tensor_count": len(changed_keys),
        "changed_tensor_keys": changed_keys,
        "changed_tensor_keys_sha256": _sha256_json(changed_keys),
        "changed_trainable_tensor_count": len(changed_trainable),
        "changed_trainable_tensor_keys": changed_trainable,
        "changed_trainable_tensor_keys_sha256": _sha256_json(changed_trainable),
        "policy_trainable_tensor_count": int(trainable_closure["tensor_count"]),
        "policy_trainable_parameter_count": int(trainable_closure["parameter_count"]),
        "policy_trainable_tensor_inventory_sha256": closure_digest,
        "trainable_inventory": trainable_inventory,
    }
    return reference_audit, policy_context


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _local_digest_fields(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: receipt[key]
        for key in (
            "claim_scope",
            "artifact_role",
            "global_step",
            "pre_step_reference",
            "step1_reference",
            "training_contract",
            "training_run_contract",
            "reference_audit",
            "policy_context",
            "source_binding_unchanged",
        )
    }


def build_post_step_receipt(
    *,
    artifact_role: str,
    global_step: int,
    pre_step_reference: Mapping[str, Any],
    step1_reference: Mapping[str, Any] | None,
    training_contract: Mapping[str, Any],
    training_run_contract: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
    policy_context: Mapping[str, Any],
    distributed: Mapping[str, Any],
    claim_scope: str,
) -> dict[str, Any]:
    """Build canonical post-step evidence after every live distributed audit."""

    if artifact_role not in {"step1", "checkpoint", "final"}:
        raise ExactPeftKLPostStepError(f"unsupported artifact role: {artifact_role!r}")
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 1:
        raise ExactPeftKLPostStepError("post-step global_step must be a positive integer")
    if artifact_role == "step1" and global_step != 1:
        raise ExactPeftKLPostStepError("step1 receipt must record global_step=1")
    if artifact_role == "step1" and step1_reference is not None:
        raise ExactPeftKLPostStepError("step1 receipt cannot bind itself")
    if artifact_role != "step1" and not isinstance(step1_reference, Mapping):
        raise ExactPeftKLPostStepError("checkpoint/final receipt lacks step1 binding")
    if claim_scope == "publication":
        contract = {
            "protocol": PUBLICATION_PROTOCOL,
            "status": PUBLICATION_STATUS,
            "claim_scope": "publication",
            "development_only": False,
            "publication_eligible": True,
            "sota_eligible": True,
        }
    elif claim_scope == "development":
        contract = {
            "protocol": DEVELOPMENT_PROTOCOL,
            "status": DEVELOPMENT_STATUS,
            "claim_scope": "development_only",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
        }
    else:
        raise ExactPeftKLPostStepError(f"unsupported claim scope: {claim_scope!r}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **contract,
        "artifact_role": artifact_role,
        "global_step": global_step,
        "verifier": {
            "protocol": VERIFIER_PROTOCOL,
            "implementation": "clasp.models.exact_peft_kl_post_step",
            "implementation_sha256": _implementation_sha256(),
        },
        "pre_step_reference": dict(pre_step_reference),
        "step1_reference": dict(step1_reference) if step1_reference is not None else None,
        "training_contract": dict(training_contract),
        "training_run_contract": dict(training_run_contract),
        "reference_audit": dict(reference_audit),
        "policy_context": dict(policy_context),
        "source_binding_unchanged": True,
        "distributed": dict(distributed),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return _validate_receipt(payload, claim_scope=claim_scope)


def _expected_claim(claim_scope: str) -> dict[str, Any]:
    if claim_scope == "publication":
        return {
            "protocol": PUBLICATION_PROTOCOL,
            "status": PUBLICATION_STATUS,
            "claim_scope": "publication",
            "development_only": False,
            "publication_eligible": True,
            "sota_eligible": True,
        }
    if claim_scope == "development":
        return {
            "protocol": DEVELOPMENT_PROTOCOL,
            "status": DEVELOPMENT_STATUS,
            "claim_scope": "development_only",
            "development_only": True,
            "publication_eligible": False,
            "sota_eligible": False,
        }
    raise ExactPeftKLPostStepError(f"unsupported claim scope: {claim_scope!r}")


def _validate_pre_binding(pre: object, *, claim_scope: str) -> Mapping[str, Any]:
    if not isinstance(pre, Mapping) or set(pre) != _PRE_STEP_KEYS:
        raise ExactPeftKLPostStepError("pre-step binding schema mismatch")
    expected_protocol = (
        PRE_PUBLICATION_PROTOCOL if claim_scope == "publication" else PRE_DEVELOPMENT_PROTOCOL
    )
    expected_status = (
        PRE_PUBLICATION_STATUS if claim_scope == "publication" else PRE_DEVELOPMENT_STATUS
    )
    if pre.get("protocol") != expected_protocol or pre.get("status") != expected_status:
        raise ExactPeftKLPostStepError("pre-step protocol/status mismatch")
    for field in (
        "receipt_sha256",
        "receipt_file_sha256",
        "reference_tensor_inventory_sha256",
        "model_config_sha256",
    ):
        if not _valid_sha256(pre.get(field)):
            raise ExactPeftKLPostStepError(f"pre-step binding has invalid {field}")
    for field in ("reference_tensor_count", "reference_parameter_count"):
        observed = pre.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            raise ExactPeftKLPostStepError(f"pre-step binding has invalid {field}")
    for field in ("model_class", "prepared_class", "unwrapped_class", "active_adapter"):
        if not isinstance(pre.get(field), str) or not pre[field]:
            raise ExactPeftKLPostStepError(f"pre-step binding has invalid {field}")
    devices = pre.get("device_types")
    if not isinstance(devices, list) or not devices or devices != sorted(set(devices)):
        raise ExactPeftKLPostStepError("pre-step device types are invalid")
    return pre


def _validate_cuda_memory(value: object, *, rank: int) -> None:
    if not isinstance(value, Mapping) or set(value) != _CUDA_MEMORY_KEYS:
        raise ExactPeftKLPostStepError("CUDA memory evidence schema mismatch")
    if value.get("rank") != rank or not isinstance(value.get("cuda_available"), bool):
        raise ExactPeftKLPostStepError("CUDA memory rank/availability mismatch")
    numeric = (
        "total_memory_bytes",
        "nvidia_smi_total_memory_mib",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    if value["cuda_available"]:
        if (
            isinstance(value.get("device_index"), bool)
            or not isinstance(value.get("device_index"), int)
            or value["device_index"] < 0
            or isinstance(value.get("physical_device_index"), bool)
            or not isinstance(value.get("physical_device_index"), int)
            or value["physical_device_index"] < 0
            or not isinstance(value.get("visible_device_token"), str)
            or not value["visible_device_token"]
            or not isinstance(value.get("device_uuid"), str)
            or not value["device_uuid"]
            or not isinstance(value.get("device_name"), str)
            or not value["device_name"]
            or not isinstance(value.get("driver_version"), str)
            or not value["driver_version"]
            or not isinstance(value.get("cuda_runtime_version"), str)
            or not value["cuda_runtime_version"]
        ):
            raise ExactPeftKLPostStepError("CUDA device/runtime identity is invalid")
        for field in numeric:
            observed = value.get(field)
            if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
                raise ExactPeftKLPostStepError(f"CUDA memory {field} is invalid")
        if (
            value["total_memory_bytes"] <= 0
            or value["peak_reserved_bytes"] < value["peak_allocated_bytes"]
            or value["nvidia_smi_total_memory_mib"] <= 0
        ):
            raise ExactPeftKLPostStepError("CUDA memory totals are inconsistent")
    else:
        nullable = (
            "device_index",
            "physical_device_index",
            "visible_device_token",
            "device_uuid",
            "device_name",
            "driver_version",
            "cuda_runtime_version",
            "total_memory_bytes",
            "nvidia_smi_total_memory_mib",
        )
        if any(value.get(field) is not None for field in nullable) or any(
            value.get(field) != 0 for field in ("peak_allocated_bytes", "peak_reserved_bytes")
        ):
            raise ExactPeftKLPostStepError("CPU rank carries fabricated CUDA evidence")


def _validate_receipt(receipt: object, *, claim_scope: str) -> dict[str, Any]:
    if not isinstance(receipt, dict) or set(receipt) != _TOP_LEVEL_KEYS:
        raise ExactPeftKLPostStepError("post-step receipt top-level schema mismatch")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256_json(unhashed):
        raise ExactPeftKLPostStepError("post-step receipt self-hash mismatch")
    expected = _expected_claim(claim_scope)
    if receipt.get("schema_version") != SCHEMA_VERSION or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ExactPeftKLPostStepError("post-step schema/claim mismatch")
    if receipt.get("verifier") != {
        "protocol": VERIFIER_PROTOCOL,
        "implementation": "clasp.models.exact_peft_kl_post_step",
        "implementation_sha256": _implementation_sha256(),
    }:
        raise ExactPeftKLPostStepError("post-step verifier identity mismatch")

    role = receipt.get("artifact_role")
    step = receipt.get("global_step")
    if (
        role not in {"step1", "checkpoint", "final"}
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 1
    ):
        raise ExactPeftKLPostStepError("post-step role/global_step mismatch")
    if role == "step1" and (step != 1 or receipt.get("step1_reference") is not None):
        raise ExactPeftKLPostStepError("step1 chronology mismatch")
    pre = _validate_pre_binding(receipt.get("pre_step_reference"), claim_scope=claim_scope)
    step1 = receipt.get("step1_reference")
    if role != "step1":
        if not isinstance(step1, Mapping) or set(step1) != _STEP1_KEYS:
            raise ExactPeftKLPostStepError("post-step receipt lacks exact step1 binding")
        expected_status = PUBLICATION_STATUS if claim_scope == "publication" else DEVELOPMENT_STATUS
        expected_protocol = (
            PUBLICATION_PROTOCOL if claim_scope == "publication" else DEVELOPMENT_PROTOCOL
        )
        if (
            step1.get("protocol") != expected_protocol
            or step1.get("status") != expected_status
            or step1.get("global_step") != 1
        ):
            raise ExactPeftKLPostStepError("step1 binding protocol/status/step mismatch")
        for field in (
            "receipt_sha256",
            "receipt_file_sha256",
            "reference_tensor_inventory_sha256",
            "policy_trainable_tensor_inventory_sha256",
        ):
            if not _valid_sha256(step1.get(field)):
                raise ExactPeftKLPostStepError(f"step1 binding has invalid {field}")

    training = receipt.get("training_contract")
    try:
        beta = float(training.get("beta")) if isinstance(training, Mapping) else float("nan")
    except (TypeError, ValueError):
        beta = float("nan")
    if (
        not isinstance(training, Mapping)
        or set(training) != _TRAINING_CONTRACT_KEYS
        or isinstance(training.get("beta"), bool)
        or not isinstance(training.get("beta"), (int, float))
        or not math.isfinite(beta)
        or beta <= 0
        or training.get("dtype") not in {"float16", "bfloat16", "float32"}
        or training.get("sync_ref_model") is not False
        or training.get("use_vllm") is not False
        or training.get("parameterization") != "continue_same_bias_all_sft_adapter"
        or training.get("policy_adapter_autocast_dtype") is not True
        or training.get("reference_adapter_autocast_dtype") is not True
    ):
        raise ExactPeftKLPostStepError("post-step training contract mismatch")
    run = receipt.get("training_run_contract")
    if (
        not isinstance(run, Mapping)
        or set(run) != _RUN_CONTRACT_KEYS
        or run.get("protocol") != TRAINING_RUN_CONTRACT_IDENTITY_PROTOCOL
        or run.get("status") != TRAINING_RUN_CONTRACT_IDENTITY_STATUS
        or not _valid_sha256(run.get("file_sha256"))
        or not _valid_sha256(run.get("payload_sha256"))
        or isinstance(run.get("size_bytes"), bool)
        or not isinstance(run.get("size_bytes"), int)
        or run["size_bytes"] <= 0
        or isinstance(run.get("payload_schema_version"), bool)
        or not isinstance(run.get("payload_schema_version"), int)
        or run["payload_schema_version"] <= 0
        or run.get("payload_stage") != "grpo"
        or (claim_scope == "publication" and run["payload_schema_version"] != 2)
        or isinstance(run.get("fixed_horizon_max_steps"), bool)
        or not isinstance(run.get("fixed_horizon_max_steps"), int)
        or run["fixed_horizon_max_steps"] <= 0
        or isinstance(run.get("fixed_horizon_world_size"), bool)
        or not isinstance(run.get("fixed_horizon_world_size"), int)
        or run["fixed_horizon_world_size"] <= 0
    ):
        raise ExactPeftKLPostStepError("training run-contract binding mismatch")
    if step > run["fixed_horizon_max_steps"] or (
        role == "final" and step != run["fixed_horizon_max_steps"]
    ):
        raise ExactPeftKLPostStepError("post-step role/global_step exceeds fixed horizon")

    reference = receipt.get("reference_audit")
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_AUDIT_KEYS:
        raise ExactPeftKLPostStepError("reference-audit schema mismatch")
    true_fields = (
        "trainer_ref_model_identity",
        "reference_prepared_object_identity",
        "reference_unwrapped_identity",
        "reference_eval_mode",
        "reference_input_grad_hook_absent",
        "live_equals_pre_step_bitwise",
        "policy_reference_storage_independent",
        "optimizer_excludes_reference_storage",
        "optimizer_matches_policy_trainables",
    )
    if (
        reference.get("method") != "live_prepared_reference_immutability_v1"
        or any(reference.get(field) is not True for field in true_fields)
        or reference.get("reference_trainable_tensor_count") != 0
        or reference.get("prepared_class") != pre["prepared_class"]
        or reference.get("unwrapped_class") != pre["unwrapped_class"]
        or reference.get("active_adapter") != pre["active_adapter"]
        or reference.get("model_config_sha256") != pre["model_config_sha256"]
        or reference.get("device_types") != pre["device_types"]
        or isinstance(reference.get("optimizer_parameter_count"), bool)
        or not isinstance(reference.get("optimizer_parameter_count"), int)
        or reference["optimizer_parameter_count"] <= 0
    ):
        raise ExactPeftKLPostStepError("reference-audit invariant mismatch")
    baseline = _validate_inventory(reference.get("baseline_inventory"), label="baseline")
    live = _validate_inventory(reference.get("live_inventory"), label="live reference")
    if (
        baseline != live
        or baseline["tensor_count"] != pre["reference_tensor_count"]
        or baseline["parameter_count"] != pre["reference_parameter_count"]
        or baseline["tensor_inventory_sha256"] != pre["reference_tensor_inventory_sha256"]
    ):
        raise ExactPeftKLPostStepError("live reference differs from pre-step baseline")

    policy = receipt.get("policy_context")
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_CONTEXT_KEYS:
        raise ExactPeftKLPostStepError("policy-context schema mismatch")
    full = _validate_inventory(policy.get("full_inventory"), label="live policy")
    trainable = _validate_inventory(policy.get("trainable_inventory"), label="policy trainable")
    adapter_name = policy.get("adapter_name")
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ExactPeftKLPostStepError("policy adapter name is invalid")
    baseline_by_key = {record["key"]: record for record in baseline["tensors"]}
    full_by_key = {record["key"]: record for record in full["tensors"]}
    if set(baseline_by_key) != set(full_by_key):
        raise ExactPeftKLPostStepError("policy/baseline full-state keys differ")
    changed = sorted(
        canonicalize_peft_adapter_key(key, adapter_name)
        for key in full_by_key
        if full_by_key[key] != baseline_by_key[key]
    )
    canonical_full: dict[str, Mapping[str, Any]] = {}
    for key, record in full_by_key.items():
        canonical = canonicalize_peft_adapter_key(key, adapter_name)
        if canonical in canonical_full:
            raise ExactPeftKLPostStepError("canonical full-policy key collision")
        canonical_full[canonical] = record
    trainable_by_key = {record["key"]: record for record in trainable["tensors"]}
    if not set(trainable_by_key).issubset(canonical_full):
        raise ExactPeftKLPostStepError("trainable inventory is absent from full policy")
    for key, record in trainable_by_key.items():
        if not _trainable_matches_full_tensor(canonical_full[key], record):
            raise ExactPeftKLPostStepError("trainable/full policy inventory mismatch")
    changed_trainable = policy.get("changed_trainable_tensor_keys")
    require_change = role == "final" or (role == "checkpoint" and step > 1)
    if (
        policy.get("method") != "full_live_policy_state_and_trainable_closure_v1"
        or policy.get("pre_step_tensor_inventory_sha256")
        != pre["reference_tensor_inventory_sha256"]
        or policy.get("changed_from_pre_step") is not bool(changed)
        or policy.get("changed_tensor_keys") != changed
        or policy.get("changed_tensor_count") != len(changed)
        or policy.get("changed_tensor_keys_sha256") != _sha256_json(changed)
        or not isinstance(changed_trainable, list)
        or changed_trainable != sorted(set(changed_trainable))
        or set(changed_trainable) != set(changed)
        or policy.get("changed_trainable_tensor_count") != len(changed_trainable)
        or policy.get("changed_trainable_tensor_keys_sha256") != _sha256_json(changed_trainable)
        or (require_change and not changed_trainable)
        or policy.get("policy_trainable_tensor_count") != trainable["tensor_count"]
        or policy.get("policy_trainable_parameter_count") != trainable["parameter_count"]
        or policy.get("policy_trainable_tensor_inventory_sha256")
        != trainable["tensor_inventory_sha256"]
        or trainable["tensor_count"] <= 0
        or trainable["parameter_count"] <= 0
        or reference["optimizer_parameter_count"] != trainable["tensor_count"]
    ):
        raise ExactPeftKLPostStepError("policy-context update/closure evidence mismatch")
    if receipt.get("source_binding_unchanged") is not True:
        raise ExactPeftKLPostStepError("post-step source binding was not reverified")

    distributed = receipt.get("distributed")
    if not isinstance(distributed, Mapping) or set(distributed) != _DISTRIBUTED_KEYS:
        raise ExactPeftKLPostStepError("post-step distributed schema mismatch")
    expected_local = _sha256_json(_local_digest_fields(receipt))
    world_size = distributed.get("world_size")
    ranks = distributed.get("rank_digests")
    memory = distributed.get("rank_cuda_memory")
    if (
        distributed.get("phase") != f"exact_peft_kl_{role}_{step}"
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or world_size != run["fixed_horizon_world_size"]
        or ranks != [{"rank": rank, "digest": expected_local} for rank in range(world_size)]
        or distributed.get("consensus_digest") != expected_local
        or not isinstance(memory, list)
        or len(memory) != world_size
    ):
        raise ExactPeftKLPostStepError("post-step distributed consensus mismatch")
    for rank, record in enumerate(memory):
        _validate_cuda_memory(record, rank=rank)
    evidence = {"rank_digests": ranks, "rank_cuda_memory": memory}
    if distributed.get("distributed_evidence_sha256") != _sha256_json(evidence):
        raise ExactPeftKLPostStepError("distributed evidence hash mismatch")
    return receipt


def _find_training_run_contract(receipt_path: Path, *, claim_scope: str) -> dict[str, Any]:
    explicit = str(os.environ.get("CLASP_TRAINING_RUN_CONTRACT_PATH", "")).strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().absolute())
    for root in (
        receipt_path.parent,
        receipt_path.parent.parent,
        receipt_path.parent.parent.parent,
    ):
        candidates.extend(
            (root / "run_contract.json", root / "local_development_run_contract.json")
        )
    observed: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        observed.append(training_run_contract_identity(candidate, claim_scope=claim_scope))
    unique = {_sha256_json(value): value for value in observed}
    if len(unique) != 1:
        raise ExactPeftKLPostStepError(
            "cannot resolve exactly one immutable training run-contract identity"
        )
    return next(iter(unique.values()))


def step1_reference_binding(
    path: str | os.PathLike[str],
    *,
    claim_scope: str,
    rehash_sources: bool = True,
) -> dict[str, Any]:
    """Return the exact compact lineage binding embedded by later receipts."""

    receipt_path = Path(path).expanduser().absolute()
    receipt = inspect_post_step_receipt(
        receipt_path,
        claim_scope=claim_scope,
        expected_global_step=1,
        expected_artifact_role="step1",
        rehash_sources=rehash_sources,
        rehash_policy_artifact=False,
    )
    raw = _regular_read_only_bytes(receipt_path)
    return {
        "protocol": receipt["protocol"],
        "status": receipt["status"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": _sha256_bytes(raw),
        "global_step": 1,
        "reference_tensor_inventory_sha256": receipt["reference_audit"]["live_inventory"][
            "tensor_inventory_sha256"
        ],
        "policy_trainable_tensor_inventory_sha256": receipt["policy_context"][
            "policy_trainable_tensor_inventory_sha256"
        ],
    }


def inspect_post_step_receipt(
    path: str | os.PathLike[str],
    *,
    claim_scope: str,
    expected_global_step: int | None = None,
    expected_artifact_role: str | None = None,
    rehash_sources: bool = True,
    rehash_policy_artifact: bool = True,
    expected_policy_trainable_tensor_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen canonical post-step bytes and optionally their sibling lineage."""

    receipt_path = Path(path).expanduser().absolute()
    if receipt_path.name not in {STEP1_RECEIPT_FILENAME, POST_STEP_RECEIPT_FILENAME}:
        raise ExactPeftKLPostStepError(f"unexpected post-step receipt name: {receipt_path}")
    raw = _regular_read_only_bytes(receipt_path)
    receipt = _decode_object(raw, path=receipt_path)
    if raw != _receipt_bytes(receipt):
        raise ExactPeftKLPostStepError("post-step receipt is not canonically serialized")
    validated = _validate_receipt(receipt, claim_scope=claim_scope)
    role = validated["artifact_role"]
    if (receipt_path.name == STEP1_RECEIPT_FILENAME) is not (role == "step1"):
        raise ExactPeftKLPostStepError("post-step filename/artifact-role mismatch")
    if expected_global_step is not None and validated["global_step"] != expected_global_step:
        raise ExactPeftKLPostStepError("post-step receipt global_step differs from expected")
    if expected_artifact_role is not None and validated["artifact_role"] != expected_artifact_role:
        raise ExactPeftKLPostStepError("post-step receipt artifact role differs from expected")
    if rehash_sources:
        pre_path = receipt_path.parent / PRE_STEP_RECEIPT_FILENAME
        pre = pre_step_reference_identity(pre_path, claim_scope=claim_scope)
        if pre != validated["pre_step_reference"]:
            raise ExactPeftKLPostStepError("post-step pre-step sibling identity mismatch")
        run_identity = _find_training_run_contract(receipt_path, claim_scope=claim_scope)
        if run_identity != validated["training_run_contract"]:
            raise ExactPeftKLPostStepError("training run-contract live identity mismatch")
        if role != "step1":
            step1_path = receipt_path.parent / STEP1_RECEIPT_FILENAME
            step1 = step1_reference_binding(
                step1_path,
                claim_scope=claim_scope,
                rehash_sources=True,
            )
            if step1 != validated["step1_reference"]:
                raise ExactPeftKLPostStepError("post-step step1 sibling identity mismatch")
    expected_digest = expected_policy_trainable_tensor_inventory_sha256
    if rehash_policy_artifact and role != "step1":
        try:
            fidelity = verify_checkpoint_fidelity(receipt_path.parent)
        except CheckpointFidelityError as exc:
            raise ExactPeftKLPostStepError(
                f"post-step sibling checkpoint fidelity is invalid: {exc}"
            ) from exc
        observed_digest = fidelity["trainable_closure"]["tensor_inventory_sha256"]
        if expected_digest is not None and expected_digest != observed_digest:
            raise ExactPeftKLPostStepError("expected checkpoint-fidelity closure mismatch")
        expected_digest = observed_digest
    if (
        expected_digest is not None
        and expected_digest
        != validated["policy_context"]["policy_trainable_tensor_inventory_sha256"]
    ):
        raise ExactPeftKLPostStepError("post-step policy/checkpoint closure mismatch")
    return validated


def write_post_step_receipt(
    output_dir: str | os.PathLike[str],
    receipt: Mapping[str, Any],
    *,
    claim_scope: str,
    filename: str,
    rehash_sources: bool = True,
) -> dict[str, Any]:
    """Create one read-only post-step receipt or exactly reverify it."""

    if filename not in {STEP1_RECEIPT_FILENAME, POST_STEP_RECEIPT_FILENAME}:
        raise ExactPeftKLPostStepError(f"unsupported post-step filename: {filename!r}")
    validated = _validate_receipt(dict(receipt), claim_scope=claim_scope)
    role = validated["artifact_role"]
    if (filename == STEP1_RECEIPT_FILENAME) != (role == "step1"):
        raise ExactPeftKLPostStepError("post-step filename/artifact-role mismatch")
    root = Path(output_dir).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    encoded = _receipt_bytes(validated)
    if path.exists() or path.is_symlink():
        existing = inspect_post_step_receipt(
            path,
            claim_scope=claim_scope,
            rehash_sources=rehash_sources,
            rehash_policy_artifact=False,
        )
        if existing != validated or _regular_read_only_bytes(path) != encoded:
            raise ExactPeftKLPostStepError(f"existing post-step receipt differs: {path}")
        return existing
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExactPeftKLPostStepError(
                f"refusing to replace post-step receipt: {path}"
            ) from exc
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return inspect_post_step_receipt(
        path,
        claim_scope=claim_scope,
        rehash_sources=rehash_sources,
        rehash_policy_artifact=False,
    )


def post_step_invariant_projection(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Project all step-independent reference/runtime invariants.

    This projection is integrity evidence only; it never replaces live source
    rehashing by :func:`inspect_post_step_receipt`.
    """

    step1 = receipt.get("step1_reference")
    if receipt.get("artifact_role") == "step1":
        step1_semantic = receipt.get("receipt_sha256")
        step1_file = receipt_file_sha256
    else:
        step1_semantic = step1.get("receipt_sha256") if isinstance(step1, Mapping) else None
        step1_file = step1.get("receipt_file_sha256") if isinstance(step1, Mapping) else None
    reference = receipt.get("reference_audit", {})
    distributed = receipt.get("distributed", {})
    return {
        "schema_version": receipt.get("schema_version"),
        "claim_scope": receipt.get("claim_scope"),
        "verifier": receipt.get("verifier"),
        "pre_step_reference": receipt.get("pre_step_reference"),
        "step1_receipt_sha256": step1_semantic,
        "step1_receipt_file_sha256": step1_file,
        "training_contract": receipt.get("training_contract"),
        "training_run_contract": receipt.get("training_run_contract"),
        "world_size": distributed.get("world_size"),
        "rank_cuda_identity": [
            {
                key: record.get(key)
                for key in (
                    "rank",
                    "cuda_available",
                    "device_index",
                    "physical_device_index",
                    "visible_device_token",
                    "device_uuid",
                    "device_name",
                    "driver_version",
                    "cuda_runtime_version",
                    "total_memory_bytes",
                    "nvidia_smi_total_memory_mib",
                )
            }
            for record in distributed.get("rank_cuda_memory", [])
            if isinstance(record, Mapping)
        ],
        "reference_method": reference.get("method"),
        "reference_prepared_class": reference.get("prepared_class"),
        "reference_unwrapped_class": reference.get("unwrapped_class"),
        "reference_active_adapter": reference.get("active_adapter"),
        "reference_model_config_sha256": reference.get("model_config_sha256"),
        "reference_device_types": reference.get("device_types"),
        "reference_tensor_count": reference.get("live_inventory", {}).get("tensor_count"),
        "reference_parameter_count": reference.get("live_inventory", {}).get("parameter_count"),
        "reference_tensor_inventory_sha256": reference.get("live_inventory", {}).get(
            "tensor_inventory_sha256"
        ),
        "reference_trainable_tensor_count": reference.get("reference_trainable_tensor_count"),
        "reference_eval_mode": reference.get("reference_eval_mode"),
        "reference_input_grad_hook_absent": reference.get("reference_input_grad_hook_absent"),
        "trainer_ref_model_identity": reference.get("trainer_ref_model_identity"),
        "reference_prepared_object_identity": reference.get("reference_prepared_object_identity"),
        "reference_unwrapped_identity": reference.get("reference_unwrapped_identity"),
        "policy_reference_storage_independent": reference.get(
            "policy_reference_storage_independent"
        ),
        "optimizer_excludes_reference_storage": reference.get(
            "optimizer_excludes_reference_storage"
        ),
        "optimizer_matches_policy_trainables": reference.get("optimizer_matches_policy_trainables"),
        "optimizer_parameter_count": reference.get("optimizer_parameter_count"),
    }


def _identity_from_validated_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
) -> dict[str, Any]:
    reference = receipt["reference_audit"]
    policy = receipt["policy_context"]
    training = receipt["training_contract"]
    run = receipt["training_run_contract"]
    distributed = receipt["distributed"]
    step1 = receipt.get("step1_reference")
    step1_semantic = (
        receipt["receipt_sha256"]
        if receipt["artifact_role"] == "step1"
        else step1["receipt_sha256"]
    )
    step1_file = (
        receipt_file_sha256 if receipt["artifact_role"] == "step1" else step1["receipt_file_sha256"]
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": receipt["protocol"],
        "status": receipt["status"],
        "claim_scope": receipt["claim_scope"],
        "artifact_role": receipt["artifact_role"],
        "global_step": receipt["global_step"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": receipt_file_sha256,
        "verifier_protocol": receipt["verifier"]["protocol"],
        "verifier_implementation_sha256": receipt["verifier"]["implementation_sha256"],
        "pre_step_receipt_sha256": receipt["pre_step_reference"]["receipt_sha256"],
        "pre_step_receipt_file_sha256": receipt["pre_step_reference"]["receipt_file_sha256"],
        "step1_receipt_sha256": step1_semantic,
        "step1_receipt_file_sha256": step1_file,
        "training_dtype": training["dtype"],
        "beta": training["beta"],
        "sync_ref_model": training["sync_ref_model"],
        "use_vllm": training["use_vllm"],
        "parameterization": training["parameterization"],
        "policy_adapter_autocast_dtype": training["policy_adapter_autocast_dtype"],
        "reference_adapter_autocast_dtype": training["reference_adapter_autocast_dtype"],
        "training_run_contract_file_sha256": run["file_sha256"],
        "training_run_contract_payload_sha256": run["payload_sha256"],
        "training_run_contract_size_bytes": run["size_bytes"],
        "training_run_contract_max_steps": run["fixed_horizon_max_steps"],
        "training_run_contract_world_size": run["fixed_horizon_world_size"],
        "world_size": distributed["world_size"],
        "distributed_consensus_digest": distributed["consensus_digest"],
        "distributed_evidence_sha256": distributed["distributed_evidence_sha256"],
        "rank_cuda_memory": [dict(record) for record in distributed["rank_cuda_memory"]],
        "reference_model_class": receipt["pre_step_reference"]["model_class"],
        "reference_prepared_class": reference["prepared_class"],
        "reference_unwrapped_class": reference["unwrapped_class"],
        "reference_active_adapter": reference["active_adapter"],
        "reference_model_config_sha256": reference["model_config_sha256"],
        "reference_device_types": list(reference["device_types"]),
        "reference_tensor_count": reference["live_inventory"]["tensor_count"],
        "reference_parameter_count": reference["live_inventory"]["parameter_count"],
        "reference_tensor_inventory_sha256": reference["live_inventory"]["tensor_inventory_sha256"],
        "reference_trainable_tensor_count": reference["reference_trainable_tensor_count"],
        "reference_eval_mode": reference["reference_eval_mode"],
        "reference_input_grad_hook_absent": reference["reference_input_grad_hook_absent"],
        "trainer_ref_model_identity": reference["trainer_ref_model_identity"],
        "reference_prepared_object_identity": reference["reference_prepared_object_identity"],
        "reference_unwrapped_identity": reference["reference_unwrapped_identity"],
        "policy_reference_storage_independent": reference["policy_reference_storage_independent"],
        "optimizer_excludes_reference_storage": reference["optimizer_excludes_reference_storage"],
        "optimizer_matches_policy_trainables": reference["optimizer_matches_policy_trainables"],
        "policy_trainable_tensor_count": policy["policy_trainable_tensor_count"],
        "policy_trainable_parameter_count": policy["policy_trainable_parameter_count"],
        "policy_trainable_tensor_inventory_sha256": policy[
            "policy_trainable_tensor_inventory_sha256"
        ],
        "policy_changed_from_pre_step": policy["changed_from_pre_step"],
        "changed_trainable_tensor_count": policy["changed_trainable_tensor_count"],
        "changed_trainable_tensor_keys_sha256": policy["changed_trainable_tensor_keys_sha256"],
    }
    invariant = post_step_invariant_projection(
        receipt,
        receipt_file_sha256=receipt_file_sha256,
    )
    payload["invariant"] = invariant
    payload["invariant_sha256"] = _sha256_json(invariant)
    payload["identity_sha256"] = _sha256_json(payload)
    return payload


def validate_post_step_reference_identity(
    value: object,
    *,
    claim_scope: str | None = None,
) -> dict[str, Any]:
    """Strictly validate a portable post-step identity and its self-hash."""

    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise ExactPeftKLPostStepError("post-step portable identity schema mismatch")
    unhashed = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != _sha256_json(unhashed):
        raise ExactPeftKLPostStepError("post-step portable identity self-hash mismatch")
    for field in (
        "receipt_sha256",
        "receipt_file_sha256",
        "verifier_implementation_sha256",
        "pre_step_receipt_sha256",
        "pre_step_receipt_file_sha256",
        "step1_receipt_sha256",
        "step1_receipt_file_sha256",
        "training_run_contract_file_sha256",
        "training_run_contract_payload_sha256",
        "distributed_consensus_digest",
        "distributed_evidence_sha256",
        "reference_model_config_sha256",
        "reference_tensor_inventory_sha256",
        "policy_trainable_tensor_inventory_sha256",
        "changed_trainable_tensor_keys_sha256",
        "invariant_sha256",
        "identity_sha256",
    ):
        if not _valid_sha256(value.get(field)):
            raise ExactPeftKLPostStepError(f"post-step portable identity has invalid {field}")
    observed_claim = value.get("claim_scope")
    if observed_claim == "publication":
        expected_protocol = PUBLICATION_PROTOCOL
        expected_status = PUBLICATION_STATUS
        normalized_scope = "publication"
    elif observed_claim == "development_only":
        expected_protocol = DEVELOPMENT_PROTOCOL
        expected_status = DEVELOPMENT_STATUS
        normalized_scope = "development"
    else:
        raise ExactPeftKLPostStepError("post-step portable identity claim scope is invalid")
    if claim_scope is not None and claim_scope != normalized_scope:
        raise ExactPeftKLPostStepError("post-step portable identity claim scope differs")
    beta_value = value.get("beta")
    beta = (
        float(beta_value)
        if isinstance(beta_value, (int, float)) and not isinstance(beta_value, bool)
        else float("nan")
    )
    role = value.get("artifact_role")
    step = value.get("global_step")
    invariant = value.get("invariant")
    boolean_true = (
        "reference_eval_mode",
        "reference_input_grad_hook_absent",
        "trainer_ref_model_identity",
        "reference_prepared_object_identity",
        "reference_unwrapped_identity",
        "policy_reference_storage_independent",
        "optimizer_excludes_reference_storage",
        "optimizer_matches_policy_trainables",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != expected_protocol
        or value.get("status") != expected_status
        or value.get("verifier_protocol") != VERIFIER_PROTOCOL
        or value.get("verifier_implementation_sha256") != _implementation_sha256()
        or role not in {"step1", "checkpoint", "final"}
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 1
        or (role == "step1" and step != 1)
        or isinstance(value.get("beta"), bool)
        or not isinstance(value.get("beta"), (int, float))
        or not math.isfinite(beta)
        or beta <= 0
        or value.get("training_dtype") not in {"float16", "bfloat16", "float32"}
        or value.get("sync_ref_model") is not False
        or value.get("use_vllm") is not False
        or value.get("parameterization") != "continue_same_bias_all_sft_adapter"
        or value.get("policy_adapter_autocast_dtype") is not True
        or value.get("reference_adapter_autocast_dtype") is not True
        or isinstance(value.get("world_size"), bool)
        or not isinstance(value.get("world_size"), int)
        or value["world_size"] <= 0
        or any(value.get(field) is not True for field in boolean_true)
        or value.get("reference_trainable_tensor_count") != 0
        or isinstance(value.get("reference_tensor_count"), bool)
        or not isinstance(value.get("reference_tensor_count"), int)
        or value["reference_tensor_count"] <= 0
        or isinstance(value.get("reference_parameter_count"), bool)
        or not isinstance(value.get("reference_parameter_count"), int)
        or value["reference_parameter_count"] <= 0
        or isinstance(value.get("policy_trainable_tensor_count"), bool)
        or not isinstance(value.get("policy_trainable_tensor_count"), int)
        or value["policy_trainable_tensor_count"] <= 0
        or isinstance(value.get("policy_trainable_parameter_count"), bool)
        or not isinstance(value.get("policy_trainable_parameter_count"), int)
        or value["policy_trainable_parameter_count"] <= 0
        or isinstance(value.get("changed_trainable_tensor_count"), bool)
        or not isinstance(value.get("changed_trainable_tensor_count"), int)
        or value["changed_trainable_tensor_count"] < 0
        or value.get("policy_changed_from_pre_step")
        is not bool(value["changed_trainable_tensor_count"])
        or (
            (role == "final" or (role == "checkpoint" and step > 1))
            and value["changed_trainable_tensor_count"] <= 0
        )
        or not isinstance(value.get("reference_device_types"), list)
        or not value["reference_device_types"]
        or value["reference_device_types"] != sorted(set(value["reference_device_types"]))
        or any(
            not isinstance(value.get(field), str) or not value[field]
            for field in (
                "reference_model_class",
                "reference_prepared_class",
                "reference_unwrapped_class",
                "reference_active_adapter",
            )
        )
        or value.get("reference_model_class") != value.get("reference_unwrapped_class")
        or not isinstance(value.get("training_run_contract_size_bytes"), int)
        or isinstance(value.get("training_run_contract_size_bytes"), bool)
        or value["training_run_contract_size_bytes"] <= 0
        or isinstance(value.get("training_run_contract_max_steps"), bool)
        or not isinstance(value.get("training_run_contract_max_steps"), int)
        or value["training_run_contract_max_steps"] <= 0
        or isinstance(value.get("training_run_contract_world_size"), bool)
        or not isinstance(value.get("training_run_contract_world_size"), int)
        or value["training_run_contract_world_size"] != value["world_size"]
        or not isinstance(invariant, Mapping)
        or set(invariant) != _INVARIANT_KEYS
        or value.get("invariant_sha256") != _sha256_json(invariant)
        or (role == "step1" and value.get("step1_receipt_sha256") != value.get("receipt_sha256"))
        or (
            role == "step1"
            and value.get("step1_receipt_file_sha256") != value.get("receipt_file_sha256")
        )
    ):
        raise ExactPeftKLPostStepError("post-step portable identity invariant mismatch")
    if step > value["training_run_contract_max_steps"] or (
        role == "final" and step != value["training_run_contract_max_steps"]
    ):
        raise ExactPeftKLPostStepError(
            "post-step portable identity role/global_step exceeds fixed horizon"
        )

    expected_verifier = {
        "protocol": VERIFIER_PROTOCOL,
        "implementation": "clasp.models.exact_peft_kl_post_step",
        "implementation_sha256": _implementation_sha256(),
    }
    expected_training = {
        "dtype": value["training_dtype"],
        "beta": value["beta"],
        "sync_ref_model": value["sync_ref_model"],
        "use_vllm": value["use_vllm"],
        "parameterization": value["parameterization"],
        "policy_adapter_autocast_dtype": value["policy_adapter_autocast_dtype"],
        "reference_adapter_autocast_dtype": value["reference_adapter_autocast_dtype"],
    }
    invariant_pre = _validate_pre_binding(
        invariant.get("pre_step_reference"),
        claim_scope=normalized_scope,
    )
    invariant_training = invariant.get("training_contract")
    invariant_run = invariant.get("training_run_contract")
    if (
        invariant.get("schema_version") != SCHEMA_VERSION
        or invariant.get("verifier") != expected_verifier
        or invariant.get("reference_method") != "live_prepared_reference_immutability_v1"
        or not isinstance(invariant_training, Mapping)
        or set(invariant_training) != _TRAINING_CONTRACT_KEYS
        or dict(invariant_training) != expected_training
        or not isinstance(invariant_run, Mapping)
        or set(invariant_run) != _RUN_CONTRACT_KEYS
        or invariant_run.get("protocol") != TRAINING_RUN_CONTRACT_IDENTITY_PROTOCOL
        or invariant_run.get("status") != TRAINING_RUN_CONTRACT_IDENTITY_STATUS
        or not _valid_sha256(invariant_run.get("file_sha256"))
        or not _valid_sha256(invariant_run.get("payload_sha256"))
        or isinstance(invariant_run.get("size_bytes"), bool)
        or not isinstance(invariant_run.get("size_bytes"), int)
        or invariant_run["size_bytes"] <= 0
        or isinstance(invariant_run.get("payload_schema_version"), bool)
        or not isinstance(invariant_run.get("payload_schema_version"), int)
        or invariant_run["payload_schema_version"] <= 0
        or invariant_run.get("payload_stage") != "grpo"
        or (normalized_scope == "publication" and invariant_run["payload_schema_version"] != 2)
        or isinstance(invariant_run.get("fixed_horizon_max_steps"), bool)
        or not isinstance(invariant_run.get("fixed_horizon_max_steps"), int)
        or invariant_run["fixed_horizon_max_steps"] <= 0
        or isinstance(invariant_run.get("fixed_horizon_world_size"), bool)
        or not isinstance(invariant_run.get("fixed_horizon_world_size"), int)
        or invariant_run["fixed_horizon_world_size"] <= 0
    ):
        raise ExactPeftKLPostStepError("post-step portable nested invariant mismatch")
    rank_memory = value.get("rank_cuda_memory")
    if not isinstance(rank_memory, list) or len(rank_memory) != value["world_size"]:
        raise ExactPeftKLPostStepError("post-step portable CUDA evidence mismatch")
    for rank, record in enumerate(rank_memory):
        _validate_cuda_memory(record, rank=rank)
    expected_rank_digests = [
        {"rank": rank, "digest": value["distributed_consensus_digest"]}
        for rank in range(value["world_size"])
    ]
    if value["distributed_evidence_sha256"] != _sha256_json(
        {"rank_digests": expected_rank_digests, "rank_cuda_memory": rank_memory}
    ):
        raise ExactPeftKLPostStepError("post-step portable distributed evidence mismatch")
    expected_cuda_identity = [
        {
            key: record[key]
            for key in (
                "rank",
                "cuda_available",
                "device_index",
                "physical_device_index",
                "visible_device_token",
                "device_uuid",
                "device_name",
                "driver_version",
                "cuda_runtime_version",
                "total_memory_bytes",
                "nvidia_smi_total_memory_mib",
            )
        }
        for record in rank_memory
    ]
    if (
        invariant.get("claim_scope") != observed_claim
        or invariant_pre["receipt_sha256"] != value["pre_step_receipt_sha256"]
        or invariant_pre["receipt_file_sha256"] != value["pre_step_receipt_file_sha256"]
        or invariant_pre["reference_tensor_count"] != value["reference_tensor_count"]
        or invariant_pre["reference_parameter_count"] != value["reference_parameter_count"]
        or invariant_pre["reference_tensor_inventory_sha256"]
        != value["reference_tensor_inventory_sha256"]
        or invariant_pre["model_class"] != value["reference_model_class"]
        or invariant_pre["prepared_class"] != value["reference_prepared_class"]
        or invariant_pre["unwrapped_class"] != value["reference_unwrapped_class"]
        or invariant_pre["active_adapter"] != value["reference_active_adapter"]
        or invariant_pre["model_config_sha256"] != value["reference_model_config_sha256"]
        or invariant_pre["device_types"] != value["reference_device_types"]
        or invariant.get("step1_receipt_sha256") != value["step1_receipt_sha256"]
        or invariant.get("step1_receipt_file_sha256") != value["step1_receipt_file_sha256"]
        or invariant.get("world_size") != value["world_size"]
        or invariant.get("rank_cuda_identity") != expected_cuda_identity
        or invariant_run["file_sha256"] != value["training_run_contract_file_sha256"]
        or invariant_run["payload_sha256"] != value["training_run_contract_payload_sha256"]
        or invariant_run["size_bytes"] != value["training_run_contract_size_bytes"]
        or invariant_run["fixed_horizon_max_steps"] != value["training_run_contract_max_steps"]
        or invariant_run["fixed_horizon_world_size"] != value["training_run_contract_world_size"]
        or invariant.get("reference_prepared_class") != value["reference_prepared_class"]
        or invariant.get("reference_unwrapped_class") != value["reference_unwrapped_class"]
        or invariant.get("reference_active_adapter") != value["reference_active_adapter"]
        or invariant.get("reference_model_config_sha256") != value["reference_model_config_sha256"]
        or invariant.get("reference_device_types") != value["reference_device_types"]
        or invariant.get("reference_tensor_inventory_sha256")
        != value["reference_tensor_inventory_sha256"]
        or invariant.get("reference_tensor_count") != value["reference_tensor_count"]
        or invariant.get("reference_parameter_count") != value["reference_parameter_count"]
        or invariant.get("reference_trainable_tensor_count")
        != value["reference_trainable_tensor_count"]
        or invariant.get("reference_eval_mode") != value["reference_eval_mode"]
        or invariant.get("reference_input_grad_hook_absent")
        != value["reference_input_grad_hook_absent"]
        or invariant.get("trainer_ref_model_identity") != value["trainer_ref_model_identity"]
        or invariant.get("reference_prepared_object_identity")
        != value["reference_prepared_object_identity"]
        or invariant.get("reference_unwrapped_identity") != value["reference_unwrapped_identity"]
        or invariant.get("policy_reference_storage_independent")
        != value["policy_reference_storage_independent"]
        or invariant.get("optimizer_excludes_reference_storage")
        != value["optimizer_excludes_reference_storage"]
        or invariant.get("optimizer_matches_policy_trainables")
        != value["optimizer_matches_policy_trainables"]
        or invariant.get("optimizer_parameter_count") != value["policy_trainable_tensor_count"]
    ):
        raise ExactPeftKLPostStepError("post-step portable identity projection mismatch")
    return value


def post_step_reference_identity_from_receipt_payload(
    payload: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    claim_scope: str,
    expected_policy_trainable_tensor_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Integrity-only projection for embedded canonical receipt payloads.

    This validates schema, aggregates, and self-hashes but intentionally cannot
    authorize live training because it does not reopen source artifacts.
    """

    receipt = _validate_receipt(dict(payload), claim_scope=claim_scope)
    expected_file = _sha256_bytes(_receipt_bytes(receipt))
    if receipt_file_sha256 != expected_file:
        raise ExactPeftKLPostStepError("embedded post-step receipt file hash mismatch")
    if (
        expected_policy_trainable_tensor_inventory_sha256 is not None
        and receipt["policy_context"]["policy_trainable_tensor_inventory_sha256"]
        != expected_policy_trainable_tensor_inventory_sha256
    ):
        raise ExactPeftKLPostStepError("embedded post-step policy closure mismatch")
    return validate_post_step_reference_identity(
        _identity_from_validated_receipt(
            receipt,
            receipt_file_sha256=receipt_file_sha256,
        ),
        claim_scope=claim_scope,
    )


def post_step_reference_identity(
    path: str | os.PathLike[str],
    *,
    claim_scope: str,
    expected_global_step: int | None = None,
    expected_artifact_role: str | None = None,
    rehash_sources: bool = True,
    rehash_policy_artifact: bool = True,
    expected_policy_trainable_tensor_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the portable identity of a live-authorized post-step receipt."""

    receipt_path = Path(path).expanduser().absolute()
    receipt = inspect_post_step_receipt(
        receipt_path,
        claim_scope=claim_scope,
        expected_global_step=expected_global_step,
        expected_artifact_role=expected_artifact_role,
        rehash_sources=rehash_sources,
        rehash_policy_artifact=rehash_policy_artifact,
        expected_policy_trainable_tensor_inventory_sha256=(
            expected_policy_trainable_tensor_inventory_sha256
        ),
    )
    raw = _regular_read_only_bytes(receipt_path)
    return validate_post_step_reference_identity(
        _identity_from_validated_receipt(
            receipt,
            receipt_file_sha256=_sha256_bytes(raw),
        ),
        claim_scope=claim_scope,
    )
