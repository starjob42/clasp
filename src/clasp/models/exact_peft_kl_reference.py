"""Exact, no-merge PEFT reference evidence for beta-positive GRPO.

Merging a LoRA policy changes floating-point operation order and can therefore
change the policy even when the merged state is serialized losslessly.  This
module proves a stronger alternative: the trainable policy and frozen KL
reference are independent loads of the same immutable base and PEFT adapter.

The dependency-light receipt verifier binds the exact base, adapter, tokenizer,
and source-receipt byte trees.  The live auditor additionally requires exact
parameter/buffer values, disjoint storage, a fully frozen reference, and
bitwise-identical deterministic probe distributions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from clasp.models.checkpoint_fidelity import (
    CheckpointFidelityError,
    inspect_checkpoint_adapter,
    verify_checkpoint_fidelity,
)

SCHEMA_VERSION = 2
VERIFIER_PROTOCOL = "clasp-exact-peft-kl-reference-verifier-v2"
PUBLICATION_PROTOCOL = "clasp-publication-exact-peft-kl-reference-v2"
DEVELOPMENT_PROTOCOL = "clasp-development-exact-peft-kl-reference-v2"
PUBLICATION_STATUS = "publication_exact_step_zero_peft_reference_verified"
DEVELOPMENT_STATUS = "development_exact_step_zero_peft_reference_verified"
RECEIPT_FILENAME = "exact_peft_kl_reference.json"
FIXED_PROBES = (
    (7, 11, 13, 17, 19, 23),
    (23, 19, 17, 13, 11, 7),
    (2, 3, 5, 7, 11, 13),
)

_CHECKPOINT_PROTOCOL = "clasp-checkpoint-fidelity-v1"
_LEGACY_CONVERSION_PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
_DEVELOPMENT_RELOCATION_PROTOCOL = "clasp-development-adapter-base-relocation-v1"
_DEVELOPMENT_RELOCATION_VERIFIER_PROTOCOL = "clasp-development-adapter-base-relocation-verifier-v1"
_CHECKPOINT_RECEIPT = "checkpoint_fidelity.json"
_LEGACY_RECEIPT = "legacy_saved_policy_conversion_receipt.json"
_DEVELOPMENT_RELOCATION_RECEIPT = "adapter_base_relocation_receipt.json"
_DEVELOPMENT_RELOCATION_CLAIM_BOUNDARY = (
    "Config-only relocation of a faithfully re-encoded legacy saved policy; "
    "omitted historical biases remain unrecoverable and this artifact is never "
    "eligible for publication or SOTA claims."
)
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
        "purpose",
        "verifier",
        "source_binding",
        "training_contract",
        "runtime_pair",
        "distributed",
        "receipt_sha256",
    }
)
_SOURCE_BINDING_KEYS = frozenset(
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
_TENSOR_KEYS = frozenset({"key", "shape", "dtype", "numel", "value_sha256"})
_RUNTIME_PAIR_KEYS = frozenset(
    {
        "method",
        "model_class",
        "active_adapter",
        "model_config_sha256",
        "model_objects_distinct",
        "independent_storage",
        "reference_trainable_tensor_count",
        "policy_trainable_tensor_count",
        "tensor_count",
        "parameter_count",
        "tensor_inventory_sha256",
        "tensors",
        "state_bitwise_equal",
        "prepared_runtime",
        "functional_equivalence",
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


class ExactPeftKLReferenceError(ValueError):
    """The exact PEFT KL-reference contract was not satisfied."""


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


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExactPeftKLReferenceError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise ExactPeftKLReferenceError(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExactPeftKLReferenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _regular_read_only_file_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ExactPeftKLReferenceError(f"cannot stat required file {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExactPeftKLReferenceError(f"expected a regular file: {path}")
    if stat.S_IMODE(before.st_mode) & 0o222:
        raise ExactPeftKLReferenceError(f"required receipt must be read-only: {path}")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ExactPeftKLReferenceError(f"cannot read required file {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise ExactPeftKLReferenceError(f"required file changed while read: {path}")
    return raw


def _decode_json(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactPeftKLReferenceError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExactPeftKLReferenceError(f"expected a JSON object in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return _decode_json(_regular_read_only_file_bytes(path), path=path)


def _implementation_sha256() -> str:
    return _sha256_file(Path(__file__).resolve(strict=True))


def _relocation_implementation_path() -> Path:
    return (
        Path(__file__).resolve(strict=True).parents[3]
        / "scripts"
        / ("relocate_development_adapter_base.py")
    )


def verify_development_relocation_authority(
    checkpoint_dir: str | os.PathLike[str],
    *,
    authoritative_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen relocation verifier and require its exact live result."""

    checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
    implementation = _relocation_implementation_path().resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "_clasp_exact_peft_relocation_verifier",
        implementation,
    )
    if spec is None or spec.loader is None:
        raise ExactPeftKLReferenceError(
            f"cannot load development relocation verifier: {implementation}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        verifier = getattr(module, "verify_relocation_receipt", None)
        if not callable(verifier):
            raise ExactPeftKLReferenceError(
                "development relocation implementation lacks verify_relocation_receipt"
            )
        verified = verifier(checkpoint, rehash_upstream=True)
    except ExactPeftKLReferenceError:
        raise
    except Exception as exc:
        raise ExactPeftKLReferenceError(
            f"trusted development relocation verifier rejected the source: {exc}"
        ) from exc
    if not isinstance(verified, dict):
        raise ExactPeftKLReferenceError(
            "development relocation verifier returned a non-object receipt"
        )
    if authoritative_receipt is not None and verified != dict(authoritative_receipt):
        raise ExactPeftKLReferenceError(
            "development relocation verifier result differs from authoritative receipt"
        )
    return verified


def immutable_tree_manifest(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    """Hash a local, read-only, symlink-free tree."""

    root = Path(path).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ExactPeftKLReferenceError(f"{label} is not a regular local directory: {root}")
    entries = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    writable: list[str] = []
    files: list[dict[str, Any]] = []
    for entry in entries:
        if entry.is_symlink():
            raise ExactPeftKLReferenceError(f"{label} contains a symlink: {entry}")
        mode = stat.S_IMODE(entry.lstat().st_mode)
        if mode & 0o222:
            writable.append(str(entry))
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ExactPeftKLReferenceError(f"{label} contains a non-regular entry: {entry}")
        relative = entry.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": int(entry.stat().st_size),
                "sha256": _sha256_file(entry),
            }
        )
    if writable:
        raise ExactPeftKLReferenceError(
            f"{label} must be immutable; writable entries={writable[:8]}"
        )
    if not files:
        raise ExactPeftKLReferenceError(f"{label} contains no files")
    return {
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "tree_sha256": _sha256_json(files),
        "files": files,
    }


def saved_adapter_dtype_evidence(adapter_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Require source-precision LoRA for the symmetric pinned PEFT loaders."""

    adapter = Path(adapter_path).expanduser().absolute()
    try:
        inspection = inspect_checkpoint_adapter(adapter)
    except CheckpointFidelityError as exc:
        raise ExactPeftKLReferenceError(f"cannot inspect exact KL adapter dtype: {exc}") from exc
    tensors = inspection.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExactPeftKLReferenceError("exact KL adapter tensor inventory is empty")
    lora = [
        record for record in tensors if ".lora_A." in record["key"] or ".lora_B." in record["key"]
    ]
    lora_dtypes = sorted({str(record["dtype"]) for record in lora})
    if not lora or lora_dtypes != ["F32"]:
        raise ExactPeftKLReferenceError(
            "exact paired PEFT loading requires saved LoRA tensors in F32 so both the "
            "trainable policy and frozen reference can use the same explicit "
            "autocast_adapter_dtype=True, adapter low_cpu_mem_usage=False path "
            "without reducing source precision; "
            f"observed_lora_dtypes={lora_dtypes}"
        )
    return {
        "saved_adapter_tensor_inventory_sha256": _sha256_json(tensors),
        "saved_adapter_dtypes": sorted({str(record["dtype"]) for record in tensors}),
        "saved_lora_dtype": "F32",
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
    }


def build_source_binding(
    *,
    policy_adapter_path: str | os.PathLike[str],
    reference_adapter_path: str | os.PathLike[str],
    authoritative_receipt_path: str | os.PathLike[str],
    authoritative_receipt: Mapping[str, Any],
    tokenizer_path: str | os.PathLike[str],
    claim_scope: str,
) -> dict[str, Any]:
    """Bind the exact local base/adapter/tokenizer source before either forward."""

    policy = Path(policy_adapter_path).expanduser().resolve(strict=True)
    reference = Path(reference_adapter_path).expanduser().resolve(strict=True)
    if policy != reference:
        raise ExactPeftKLReferenceError(
            "policy and reference must independently load the same PEFT adapter: "
            f"policy={policy}, reference={reference}"
        )
    config_path = reference / "adapter_config.json"
    config = _load_json(config_path)
    if config.get("bias") != "all":
        raise ExactPeftKLReferenceError("exact PEFT KL reference requires adapter bias='all'")
    base_raw = config.get("base_model_name_or_path")
    if not isinstance(base_raw, str) or not base_raw.strip():
        raise ExactPeftKLReferenceError("adapter config lacks an explicit local base model path")
    base_candidate = Path(base_raw).expanduser()
    if not base_candidate.is_absolute():
        raise ExactPeftKLReferenceError(
            "adapter base_model_name_or_path must be an absolute local path for byte binding"
        )
    base = base_candidate.resolve(strict=True)
    tokenizer = Path(tokenizer_path).expanduser().resolve(strict=True)
    receipt_path = Path(authoritative_receipt_path).expanduser().resolve(strict=True)
    source_protocol = authoritative_receipt.get("protocol")
    if claim_scope == "publication":
        if source_protocol != _CHECKPOINT_PROTOCOL:
            raise ExactPeftKLReferenceError(
                "publication exact reference requires checkpoint-fidelity source evidence"
            )
        publication = authoritative_receipt.get("publication_invariant")
        if not isinstance(publication, Mapping) or publication.get("status") != "exact":
            raise ExactPeftKLReferenceError(
                "publication source lacks the exact Qwen publication invariant"
            )
        expected_receipt = reference / _CHECKPOINT_RECEIPT
    elif claim_scope == "development":
        if source_protocol == _CHECKPOINT_PROTOCOL:
            expected_receipt = reference / _CHECKPOINT_RECEIPT
        elif source_protocol == _LEGACY_CONVERSION_PROTOCOL:
            expected_receipt = reference / _LEGACY_RECEIPT
        elif source_protocol == _DEVELOPMENT_RELOCATION_PROTOCOL:
            expected_receipt = reference / _DEVELOPMENT_RELOCATION_RECEIPT
        else:
            raise ExactPeftKLReferenceError(
                f"unsupported development source receipt protocol: {source_protocol!r}"
            )
    else:
        raise ExactPeftKLReferenceError(
            f"claim_scope must be 'publication' or 'development', got {claim_scope!r}"
        )
    if receipt_path != expected_receipt.resolve(strict=True):
        raise ExactPeftKLReferenceError(
            "external KL receipt must be the source adapter's authoritative internal receipt: "
            f"expected={expected_receipt}, observed={receipt_path}"
        )
    receipt_self_hash = authoritative_receipt.get("receipt_sha256")
    if not isinstance(receipt_self_hash, str) or len(receipt_self_hash) != 64:
        raise ExactPeftKLReferenceError("source receipt lacks a valid semantic self-hash")
    base_tree = immutable_tree_manifest(base, label="exact KL base model")
    adapter_tree = immutable_tree_manifest(reference, label="exact KL source adapter")
    dtype_evidence = saved_adapter_dtype_evidence(reference)
    tokenizer_tree = (
        base_tree
        if tokenizer == base
        else immutable_tree_manifest(tokenizer, label="exact KL tokenizer")
    )
    return {
        "method": "immutable_local_base_adapter_tokenizer_trees_v2",
        "claim_scope": claim_scope,
        "adapter_path": str(reference),
        "adapter_tree": adapter_tree,
        "base_model_path": str(base),
        "base_model_tree": base_tree,
        "tokenizer_path": str(tokenizer),
        "tokenizer_tree": tokenizer_tree,
        "authoritative_receipt_path": str(receipt_path),
        "authoritative_receipt_protocol": source_protocol,
        "authoritative_receipt_sha256": receipt_self_hash,
        "authoritative_receipt_file_sha256": _sha256_file(receipt_path),
        "adapter_config_sha256": _sha256_file(config_path),
        **dtype_evidence,
    }


def verify_source_binding_unchanged(binding: Mapping[str, Any]) -> None:
    """Rehash all identity-bearing source trees after both model loads."""

    checks = (
        ("adapter_path", "adapter_tree", "exact KL source adapter"),
        ("base_model_path", "base_model_tree", "exact KL base model"),
        ("tokenizer_path", "tokenizer_tree", "exact KL tokenizer"),
    )
    cached: dict[str, dict[str, Any]] = {}
    for path_key, tree_key, label in checks:
        raw_path = str(binding.get(path_key, ""))
        observed = cached.get(raw_path)
        if observed is None:
            observed = immutable_tree_manifest(raw_path, label=label)
            cached[raw_path] = observed
        if observed != binding.get(tree_key):
            raise ExactPeftKLReferenceError(f"{label} changed during paired model loading")
    receipt_path = Path(str(binding.get("authoritative_receipt_path", "")))
    if _sha256_file(receipt_path) != binding.get("authoritative_receipt_file_sha256"):
        raise ExactPeftKLReferenceError("authoritative source receipt changed during loading")


def _tensor_bytes(tensor: Any) -> bytes:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ExactPeftKLReferenceError("torch is required for live pair auditing") from exc
    value = tensor.detach()
    if value.layout != torch.strided or value.device.type == "meta":
        raise ExactPeftKLReferenceError("live pair contains a non-materialized strided tensor")
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


def _storage_pointer(tensor: Any) -> int:
    value = tensor.detach()
    if int(value.numel()) == 0:
        return 0
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - old torch fallback
        return int(value.storage().data_ptr())


def _model_config_digest(model: Any) -> str:
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise ExactPeftKLReferenceError("paired model does not expose a serializable config")
    return _sha256_json(to_dict())


def _active_adapter(model: Any) -> list[str]:
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ExactPeftKLReferenceError("paired PEFT model exposes no active adapter")


def _functional_pair_evidence(
    policy: Any,
    reference: Any,
    *,
    runtime_dtype: str,
) -> dict[str, Any]:
    import torch

    try:
        policy_parameter = next(policy.parameters())
        reference_parameter = next(reference.parameters())
    except StopIteration as exc:
        raise ExactPeftKLReferenceError("paired model has no parameters") from exc
    if policy_parameter.device != reference_parameter.device:
        raise ExactPeftKLReferenceError(
            "policy/reference must be on the same device for bitwise probe evidence"
        )
    vocab_size = getattr(getattr(policy, "config", None), "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= max(max(row) for row in FIXED_PROBES):
        raise ExactPeftKLReferenceError(f"model vocabulary is too small: {vocab_size!r}")
    input_ids = torch.tensor(FIXED_PROBES, dtype=torch.long, device=policy_parameter.device)
    attention_mask = torch.ones_like(input_ids)
    runtime_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if runtime_dtype not in runtime_dtypes:
        raise ExactPeftKLReferenceError(f"unsupported exact-pair runtime dtype: {runtime_dtype!r}")
    autocast_enabled = runtime_dtype != "float32"
    autocast_context = (
        torch.autocast(
            device_type=policy_parameter.device.type,
            dtype=runtime_dtypes[runtime_dtype],
        )
        if autocast_enabled
        else nullcontext()
    )
    policy_was_training = bool(policy.training)
    reference_was_training = bool(reference.training)
    policy_config = getattr(policy, "config", None)
    reference_config = getattr(reference, "config", None)
    policy_use_cache = getattr(policy_config, "use_cache", None)
    reference_use_cache = getattr(reference_config, "use_cache", None)
    try:
        policy.eval()
        reference.eval()
        if policy_config is not None and hasattr(policy_config, "use_cache"):
            policy_config.use_cache = False
        if reference_config is not None and hasattr(reference_config, "use_cache"):
            reference_config.use_cache = False
        with torch.inference_mode(), autocast_context:
            policy_logits = policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits.detach()
            reference_logits = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits.detach()
            policy_logps = torch.log_softmax(policy_logits[:, :-1, :].float(), dim=-1)
            reference_logps = torch.log_softmax(reference_logits[:, :-1, :].float(), dim=-1)
            policy_argmax = policy_logits.argmax(dim=-1)
            reference_argmax = reference_logits.argmax(dim=-1)
        if not torch.equal(policy_logits, reference_logits):
            delta = (policy_logits.float() - reference_logits.float()).abs()
            raise ExactPeftKLReferenceError(
                "independent paired loads produced different probe logits: "
                f"max_absolute_error={float(delta.max().item())!r}, "
                f"differing_elements={int(torch.count_nonzero(delta).item())}/{delta.numel()}"
            )
        if not torch.equal(policy_logps, reference_logps):
            raise ExactPeftKLReferenceError(
                "independent paired loads produced different probe distributions"
            )
        if not torch.equal(policy_argmax, reference_argmax):
            raise ExactPeftKLReferenceError(
                "independent paired loads produced different probe argmax decisions"
            )
        return {
            "method": "three_fixed_probes_bitwise_runtime_dtype_full_distribution_v2",
            "runtime_dtype": runtime_dtype,
            "autocast_enabled": autocast_enabled,
            "autocast_device_type": policy_parameter.device.type,
            "input_ids": [list(row) for row in FIXED_PROBES],
            "input_ids_sha256": _sha256_json(FIXED_PROBES),
            "logits": _tensor_record("logits", policy_logits),
            "log_probabilities": _tensor_record("log_probabilities", policy_logps),
            "argmax": _tensor_record("argmax", policy_argmax),
            "bitwise_equal": True,
            "max_absolute_error": 0.0,
        }
    finally:
        if policy_config is not None and hasattr(policy_config, "use_cache"):
            policy_config.use_cache = policy_use_cache
        if reference_config is not None and hasattr(reference_config, "use_cache"):
            reference_config.use_cache = reference_use_cache
        policy.train(policy_was_training)
        reference.train(reference_was_training)


def audit_exact_runtime_pair(
    policy: Any,
    reference: Any,
    *,
    runtime_dtype: str = "float32",
) -> dict[str, Any]:
    """Prove equality and independence of the actual policy/reference objects."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ExactPeftKLReferenceError("torch is required for live pair auditing") from exc
    if policy is reference:
        raise ExactPeftKLReferenceError("policy and reference are the same Python object")
    if policy.__class__ is not reference.__class__:
        raise ExactPeftKLReferenceError(
            f"policy/reference class mismatch: {policy.__class__!r} != {reference.__class__!r}"
        )
    policy_active = _active_adapter(policy)
    reference_active = _active_adapter(reference)
    if policy_active != reference_active or len(policy_active) != 1:
        raise ExactPeftKLReferenceError(
            f"active adapter mismatch: policy={policy_active}, reference={reference_active}"
        )
    reference_trainable = [
        name for name, value in reference.named_parameters() if value.requires_grad
    ]
    if reference_trainable:
        raise ExactPeftKLReferenceError(
            f"frozen reference still has trainable tensors: {reference_trainable[:12]}"
        )
    policy_trainable = [name for name, value in policy.named_parameters() if value.requires_grad]
    if not policy_trainable:
        raise ExactPeftKLReferenceError("policy has no trainable tensors")
    policy_state = policy.state_dict()
    reference_state = reference.state_dict()
    missing = sorted(set(policy_state) - set(reference_state))
    unexpected = sorted(set(reference_state) - set(policy_state))
    if missing or unexpected:
        raise ExactPeftKLReferenceError(
            f"paired state key mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    records: list[dict[str, Any]] = []
    shared_storage: list[str] = []
    for key in sorted(policy_state):
        policy_tensor = policy_state[key]
        reference_tensor = reference_state[key]
        if tuple(policy_tensor.shape) != tuple(reference_tensor.shape):
            raise ExactPeftKLReferenceError(
                f"paired state shape mismatch for {key}: "
                f"policy={tuple(policy_tensor.shape)}, reference={tuple(reference_tensor.shape)}"
            )
        if policy_tensor.dtype != reference_tensor.dtype:
            raise ExactPeftKLReferenceError(
                f"paired state dtype mismatch for {key}: "
                f"policy={policy_tensor.dtype}, reference={reference_tensor.dtype}; "
                "both pinned PEFT loaders must use explicit autocast_adapter_dtype=True "
                "and adapter low_cpu_mem_usage=False"
            )
        if not torch.equal(policy_tensor.detach(), reference_tensor.detach()):
            raise ExactPeftKLReferenceError(f"paired state tensor differs: {key}")
        if (
            policy_tensor.device == reference_tensor.device
            and _storage_pointer(policy_tensor) != 0
            and _storage_pointer(policy_tensor) == _storage_pointer(reference_tensor)
        ):
            shared_storage.append(key)
        records.append(_tensor_record(key, policy_tensor))
    if shared_storage:
        raise ExactPeftKLReferenceError(
            f"policy/reference share tensor storage: {shared_storage[:12]}"
        )
    config_digest = _model_config_digest(policy)
    if config_digest != _model_config_digest(reference):
        raise ExactPeftKLReferenceError("policy/reference model configs differ")
    functional = _functional_pair_evidence(
        policy,
        reference,
        runtime_dtype=runtime_dtype,
    )
    model_class = f"{policy.__class__.__module__}.{policy.__class__.__qualname__}"
    device_types = sorted(
        {str(tensor.device.type) for tensor in (*policy_state.values(), *reference_state.values())}
    )
    return {
        "method": "independent_same_base_same_adapter_exact_runtime_pair_v2",
        "model_class": model_class,
        "active_adapter": policy_active[0],
        "model_config_sha256": config_digest,
        "model_objects_distinct": True,
        "independent_storage": True,
        "reference_trainable_tensor_count": 0,
        "policy_trainable_tensor_count": len(policy_trainable),
        "tensor_count": len(records),
        "parameter_count": sum(record["numel"] for record in records),
        "tensor_inventory_sha256": _sha256_json(records),
        "tensors": records,
        "state_bitwise_equal": True,
        "prepared_runtime": {
            "trainer_ref_model_identity": True,
            "reference_unwrapped_identity": True,
            "policy_unwrapped_class": model_class,
            "reference_prepared_class": model_class,
            "reference_unwrapped_class": model_class,
            "device_types": device_types,
        },
        "functional_equivalence": functional,
    }


def build_runtime_receipt(
    *,
    source_binding: Mapping[str, Any],
    runtime_pair: Mapping[str, Any],
    distributed: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    claim_scope: str,
) -> dict[str, Any]:
    """Build a self-hashed receipt for an all-rank exact runtime pair."""

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
        raise ExactPeftKLReferenceError(f"unsupported claim_scope: {claim_scope!r}")
    if source_binding.get("claim_scope") != claim_scope:
        raise ExactPeftKLReferenceError("source binding claim scope mismatch")
    if runtime_pair.get("state_bitwise_equal") is not True:
        raise ExactPeftKLReferenceError("runtime pair does not prove bitwise state equality")
    if runtime_pair.get("independent_storage") is not True:
        raise ExactPeftKLReferenceError("runtime pair does not prove independent storage")
    if runtime_pair.get("reference_trainable_tensor_count") != 0:
        raise ExactPeftKLReferenceError("runtime reference is not fully frozen")
    _validate_training_contract(training_contract)
    _validate_runtime_pair(runtime_pair)
    functional = runtime_pair.get("functional_equivalence")
    if not isinstance(functional, Mapping) or functional.get(
        "runtime_dtype"
    ) != training_contract.get("dtype"):
        raise ExactPeftKLReferenceError(
            "runtime probe dtype differs from the exact training contract"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **contract,
        "purpose": "immutable_exact_step_zero_peft_kl_reference",
        "verifier": {
            "protocol": VERIFIER_PROTOCOL,
            "implementation": "clasp.models.exact_peft_kl_reference",
            "implementation_sha256": _implementation_sha256(),
        },
        "source_binding": dict(source_binding),
        "training_contract": dict(training_contract),
        "runtime_pair": dict(runtime_pair),
        "distributed": dict(distributed),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _validate_tree(tree: object, *, label: str) -> None:
    if not isinstance(tree, Mapping) or set(tree) != {
        "file_count",
        "size_bytes",
        "tree_sha256",
        "files",
    }:
        raise ExactPeftKLReferenceError(f"{label} tree schema mismatch")
    files = tree.get("files")
    if not isinstance(files, list) or not files:
        raise ExactPeftKLReferenceError(f"{label} tree has no files")
    previous = ""
    total = 0
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ExactPeftKLReferenceError(f"{label} tree file schema mismatch")
        path = record.get("path")
        size = record.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or path <= previous
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _valid_sha256(record.get("sha256"))
        ):
            raise ExactPeftKLReferenceError(f"{label} tree contains an invalid file record")
        previous = path
        total += size
    if (
        tree.get("file_count") != len(files)
        or tree.get("size_bytes") != total
        or tree.get("tree_sha256") != _sha256_json(files)
    ):
        raise ExactPeftKLReferenceError(f"{label} tree aggregate mismatch")


def _validate_tensor_record(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _TENSOR_KEYS:
        raise ExactPeftKLReferenceError(f"{label} tensor record schema mismatch")
    shape = record.get("shape")
    if (
        not isinstance(record.get("key"), str)
        or not record.get("key")
        or not isinstance(shape, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        )
        or not isinstance(record.get("dtype"), str)
        or not isinstance(record.get("numel"), int)
        or isinstance(record.get("numel"), bool)
        or int(record.get("numel", -1)) < 0
        or not _valid_sha256(record.get("value_sha256"))
    ):
        raise ExactPeftKLReferenceError(f"{label} tensor record is invalid")
    product = 1
    for dimension in shape:
        product *= dimension
    if product != record.get("numel"):
        raise ExactPeftKLReferenceError(f"{label} tensor numel/shape mismatch")


def _validate_training_contract(contract: object) -> None:
    if not isinstance(contract, Mapping) or set(contract) != _TRAINING_CONTRACT_KEYS:
        raise ExactPeftKLReferenceError("exact-reference training contract schema mismatch")
    try:
        beta = float(contract.get("beta", 0.0))
    except (TypeError, ValueError) as exc:
        raise ExactPeftKLReferenceError("exact-reference beta is not numeric") from exc
    if not math.isfinite(beta) or beta <= 0.0:
        raise ExactPeftKLReferenceError("exact-reference beta must be finite and positive")
    if contract.get("dtype") not in {"float16", "bfloat16", "float32"}:
        raise ExactPeftKLReferenceError("exact-reference dtype is unsupported")
    if contract.get("sync_ref_model") is not False or contract.get("use_vllm") is not False:
        raise ExactPeftKLReferenceError("exact reference must be immutable and non-vLLM")
    if (
        contract.get("policy_adapter_autocast_dtype") is not True
        or contract.get("reference_adapter_autocast_dtype") is not True
    ):
        raise ExactPeftKLReferenceError(
            "exact reference requires symmetric explicit adapter dtype autocasting"
        )
    if contract.get("parameterization") != "continue_same_bias_all_sft_adapter":
        raise ExactPeftKLReferenceError("exact-reference parameterization mismatch")


def _validate_runtime_pair(pair: object) -> None:
    if not isinstance(pair, Mapping) or set(pair) != _RUNTIME_PAIR_KEYS:
        raise ExactPeftKLReferenceError("runtime-pair schema mismatch")
    if (
        pair.get("method") != "independent_same_base_same_adapter_exact_runtime_pair_v2"
        or pair.get("model_objects_distinct") is not True
        or pair.get("independent_storage") is not True
        or pair.get("state_bitwise_equal") is not True
        or pair.get("reference_trainable_tensor_count") != 0
        or not isinstance(pair.get("policy_trainable_tensor_count"), int)
        or int(pair.get("policy_trainable_tensor_count", 0)) <= 0
        or not _valid_sha256(pair.get("model_config_sha256"))
    ):
        raise ExactPeftKLReferenceError("runtime-pair exactness invariant mismatch")
    prepared = pair.get("prepared_runtime")
    if not isinstance(prepared, Mapping) or set(prepared) != {
        "trainer_ref_model_identity",
        "reference_unwrapped_identity",
        "policy_unwrapped_class",
        "reference_prepared_class",
        "reference_unwrapped_class",
        "device_types",
    }:
        raise ExactPeftKLReferenceError("prepared runtime identity schema mismatch")
    if (
        prepared.get("trainer_ref_model_identity") is not True
        or prepared.get("reference_unwrapped_identity") is not True
        or prepared.get("policy_unwrapped_class") != pair.get("model_class")
        or prepared.get("reference_unwrapped_class") != pair.get("model_class")
        or prepared.get("reference_prepared_class") != pair.get("model_class")
        or not isinstance(prepared.get("device_types"), list)
        or not prepared.get("device_types")
    ):
        raise ExactPeftKLReferenceError("prepared runtime identity invariant mismatch")
    tensors = pair.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ExactPeftKLReferenceError("runtime-pair tensor inventory is empty")
    previous = ""
    for record in tensors:
        _validate_tensor_record(record, label="runtime-pair")
        key = str(record["key"])
        if key <= previous:
            raise ExactPeftKLReferenceError("runtime-pair tensor keys are not unique/sorted")
        previous = key
    if (
        pair.get("tensor_count") != len(tensors)
        or pair.get("parameter_count") != sum(int(record["numel"]) for record in tensors)
        or pair.get("tensor_inventory_sha256") != _sha256_json(tensors)
    ):
        raise ExactPeftKLReferenceError("runtime-pair tensor aggregate mismatch")
    functional = pair.get("functional_equivalence")
    expected_functional_keys = {
        "method",
        "runtime_dtype",
        "autocast_enabled",
        "autocast_device_type",
        "input_ids",
        "input_ids_sha256",
        "logits",
        "log_probabilities",
        "argmax",
        "bitwise_equal",
        "max_absolute_error",
    }
    if not isinstance(functional, Mapping) or set(functional) != expected_functional_keys:
        raise ExactPeftKLReferenceError("runtime-pair functional evidence schema mismatch")
    if (
        functional.get("method") != "three_fixed_probes_bitwise_runtime_dtype_full_distribution_v2"
        or functional.get("runtime_dtype") not in {"float16", "bfloat16", "float32"}
        or functional.get("autocast_enabled") is not (functional.get("runtime_dtype") != "float32")
        or not isinstance(functional.get("autocast_device_type"), str)
        or not functional.get("autocast_device_type")
        or functional.get("input_ids") != [list(row) for row in FIXED_PROBES]
        or functional.get("input_ids_sha256") != _sha256_json(FIXED_PROBES)
        or functional.get("bitwise_equal") is not True
        or functional.get("max_absolute_error") != 0.0
    ):
        raise ExactPeftKLReferenceError("runtime-pair functional exactness mismatch")
    for field in ("logits", "log_probabilities", "argmax"):
        _validate_tensor_record(functional.get(field), label=f"functional {field}")


def _validate_development_relocation_claim(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_top_level = {
        "schema_version",
        "protocol",
        "status",
        "claim_scope",
        "development_only",
        "relocation_only",
        "recovers_historical_terminal_biases",
        "converted_saved_policy_fidelity",
        "terminal_sft_policy_fidelity",
        "publication_eligible",
        "sota_eligible",
        "publication_sota_eligible",
        "claim_boundary",
        "verifier",
        "upstream_conversion",
        "frozen_base",
        "relocated_output",
        "transformation",
        "verification",
        "software",
        "receipt_sha256",
    }
    if set(receipt) != expected_top_level:
        raise ExactPeftKLReferenceError("development relocation top-level schema mismatch")
    required = {
        "schema_version": 1,
        "protocol": _DEVELOPMENT_RELOCATION_PROTOCOL,
        "status": "development_saved_policy_base_relocation_verified",
        "claim_scope": "development_only",
        "development_only": True,
        "relocation_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ExactPeftKLReferenceError("development relocation claim boundary mismatch")
    if receipt.get("claim_boundary") != _DEVELOPMENT_RELOCATION_CLAIM_BOUNDARY:
        raise ExactPeftKLReferenceError("development relocation claim-boundary text mismatch")
    software = receipt.get("software")
    if (
        not isinstance(software, Mapping)
        or set(software) != {"python", "torch", "transformers", "peft", "safetensors"}
        or any(not isinstance(value, str) or not value for value in software.values())
    ):
        raise ExactPeftKLReferenceError("development relocation software schema mismatch")
    verification = receipt.get("verification")
    verification_keys = {
        "method",
        "upstream_receipt_verified",
        "path_independent_base_manifests_exact",
        "source_and_output_stable",
        "independent_model_loads",
        "full_model_state_exact",
        "full_model_state",
        "functional_equivalence_exact",
        "functional_equivalence",
        "device",
        "requested_dtype",
    }
    required_true = {
        "upstream_receipt_verified",
        "path_independent_base_manifests_exact",
        "source_and_output_stable",
        "independent_model_loads",
        "full_model_state_exact",
        "functional_equivalence_exact",
    }
    if (
        not isinstance(verification, Mapping)
        or set(verification) != verification_keys
        or any(verification.get(field) is not True for field in required_true)
    ):
        raise ExactPeftKLReferenceError("development relocation verification is incomplete")
    if (
        verification.get("method") != "independent_source_and_relocated_policy_reload_v1"
        or verification.get("requested_dtype") not in {"auto", "float32", "bfloat16", "float16"}
        or not isinstance(verification.get("device"), str)
    ):
        raise ExactPeftKLReferenceError("development relocation verification contract mismatch")
    return verification


def _validate_source_binding(binding: object, *, claim_scope: str) -> None:
    if not isinstance(binding, Mapping) or set(binding) != _SOURCE_BINDING_KEYS:
        raise ExactPeftKLReferenceError("source-binding schema mismatch")
    if (
        binding.get("method") != "immutable_local_base_adapter_tokenizer_trees_v2"
        or binding.get("claim_scope") != claim_scope
    ):
        raise ExactPeftKLReferenceError("source-binding claim mismatch")
    for field in (
        "authoritative_receipt_sha256",
        "authoritative_receipt_file_sha256",
        "adapter_config_sha256",
        "saved_adapter_tensor_inventory_sha256",
    ):
        if not _valid_sha256(binding.get(field)):
            raise ExactPeftKLReferenceError(f"source binding has invalid {field}")
    _validate_tree(binding.get("adapter_tree"), label="adapter")
    _validate_tree(binding.get("base_model_tree"), label="base model")
    _validate_tree(binding.get("tokenizer_tree"), label="tokenizer")
    receipt_path = Path(str(binding.get("authoritative_receipt_path", ""))).expanduser().absolute()
    authoritative = _decode_json(_regular_read_only_file_bytes(receipt_path), path=receipt_path)
    claimed_hash = authoritative.get("receipt_sha256")
    unhashed = {key: value for key, value in authoritative.items() if key != "receipt_sha256"}
    if claimed_hash != _sha256_json(unhashed) or claimed_hash != binding.get(
        "authoritative_receipt_sha256"
    ):
        raise ExactPeftKLReferenceError("authoritative source receipt self-hash mismatch")
    if authoritative.get("protocol") != binding.get("authoritative_receipt_protocol"):
        raise ExactPeftKLReferenceError("authoritative source receipt protocol mismatch")
    adapter_path = Path(str(binding.get("adapter_path", ""))).expanduser().absolute()
    if receipt_path.parent != adapter_path:
        raise ExactPeftKLReferenceError("authoritative source receipt is outside adapter tree")
    config = _load_json(adapter_path / "adapter_config.json")
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).expanduser()
    if (
        not configured_base.is_absolute()
        or configured_base.resolve(strict=True)
        != Path(str(binding.get("base_model_path", ""))).resolve(strict=True)
        or config.get("bias") != "all"
    ):
        raise ExactPeftKLReferenceError("adapter config differs from exact base/bias binding")
    dtype_evidence = saved_adapter_dtype_evidence(adapter_path)
    if any(binding.get(key) != value for key, value in dtype_evidence.items()):
        raise ExactPeftKLReferenceError("saved adapter dtype evidence differs from source binding")
    if claim_scope == "publication":
        try:
            verified = verify_checkpoint_fidelity(adapter_path)
        except CheckpointFidelityError as exc:
            raise ExactPeftKLReferenceError(
                f"publication source checkpoint fidelity failed: {exc}"
            ) from exc
        if verified != authoritative:
            raise ExactPeftKLReferenceError("publication source receipt differs from live evidence")
    else:
        protocol = authoritative.get("protocol")
        if protocol not in {
            _CHECKPOINT_PROTOCOL,
            _LEGACY_CONVERSION_PROTOCOL,
            _DEVELOPMENT_RELOCATION_PROTOCOL,
        }:
            raise ExactPeftKLReferenceError("development source receipt protocol mismatch")
        if protocol == _DEVELOPMENT_RELOCATION_PROTOCOL:
            verification = _validate_development_relocation_claim(authoritative)
            verifier = authoritative.get("verifier")
            relocation_implementation = _relocation_implementation_path().resolve(strict=True)
            if not isinstance(verifier, Mapping) or verifier != {
                "protocol": _DEVELOPMENT_RELOCATION_VERIFIER_PROTOCOL,
                "implementation": "scripts.relocate_development_adapter_base",
                "implementation_sha256": _sha256_file(relocation_implementation),
            }:
                raise ExactPeftKLReferenceError("development relocation verifier identity mismatch")
            verify_development_relocation_authority(
                adapter_path,
                authoritative_receipt=authoritative,
            )
            upstream = authoritative.get("upstream_conversion")
            upstream_keys = {
                "source_adapter_path",
                "source_adapter_manifest",
                "authoritative_receipt_path",
                "authoritative_receipt_protocol",
                "authoritative_receipt_sha256",
                "authoritative_receipt_file_sha256",
                "adapter_config_sha256",
                "adapter_model_sha256",
                "configured_base_path",
                "recorded_base_manifest",
                "live_base_manifest",
            }
            if not isinstance(upstream, Mapping) or set(upstream) != upstream_keys:
                raise ExactPeftKLReferenceError(
                    "development relocation upstream-conversion schema mismatch"
                )
            source_adapter = (
                Path(str(upstream.get("source_adapter_path", ""))).expanduser().resolve(strict=True)
            )
            source_manifest = immutable_tree_manifest(
                source_adapter,
                label="development relocation upstream adapter",
            )
            if upstream.get("source_adapter_manifest") != source_manifest:
                raise ExactPeftKLReferenceError(
                    "development relocation upstream adapter bytes changed"
                )
            upstream_receipt_path = (
                Path(str(upstream.get("authoritative_receipt_path", "")))
                .expanduser()
                .resolve(strict=True)
            )
            if upstream_receipt_path.parent != source_adapter:
                raise ExactPeftKLReferenceError(
                    "development relocation upstream receipt is outside source adapter"
                )
            upstream_receipt_raw = _regular_read_only_file_bytes(upstream_receipt_path)
            upstream_receipt = _decode_json(
                upstream_receipt_raw,
                path=upstream_receipt_path,
            )
            upstream_unhashed = {
                key: value for key, value in upstream_receipt.items() if key != "receipt_sha256"
            }
            if (
                upstream_receipt.get("protocol") != _LEGACY_CONVERSION_PROTOCOL
                or upstream.get("authoritative_receipt_protocol") != _LEGACY_CONVERSION_PROTOCOL
                or upstream_receipt.get("receipt_sha256") != _sha256_json(upstream_unhashed)
                or upstream.get("authoritative_receipt_sha256")
                != upstream_receipt.get("receipt_sha256")
                or upstream.get("authoritative_receipt_file_sha256")
                != _sha256_bytes(upstream_receipt_raw)
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation upstream receipt evidence mismatch"
                )
            upstream_claim = {
                "development_only": True,
                "converted_saved_policy_fidelity": True,
                "terminal_sft_policy_fidelity": False,
                "publication_eligible": False,
                "sota_eligible": False,
                "publication_sota_eligible": False,
            }
            if any(upstream_receipt.get(key) != value for key, value in upstream_claim.items()):
                raise ExactPeftKLReferenceError(
                    "development relocation upstream claim boundary mismatch"
                )
            if upstream.get("adapter_config_sha256") != _sha256_file(
                source_adapter / "adapter_config.json"
            ) or upstream.get("adapter_model_sha256") != _sha256_file(
                source_adapter / "adapter_model.safetensors"
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation upstream adapter hashes mismatch"
                )
            _validate_tree(
                upstream.get("recorded_base_manifest"),
                label="recorded upstream base",
            )
            _validate_tree(
                upstream.get("live_base_manifest"),
                label="live upstream base",
            )
            if upstream.get("recorded_base_manifest") != upstream.get("live_base_manifest"):
                raise ExactPeftKLReferenceError(
                    "development relocation upstream base changed before relocation"
                )
            relocated = authoritative.get("relocated_output")
            frozen_base = authoritative.get("frozen_base")
            transformation = authoritative.get("transformation")
            if (
                not isinstance(relocated, Mapping)
                or not isinstance(frozen_base, Mapping)
                or not isinstance(transformation, Mapping)
            ):
                raise ExactPeftKLReferenceError("development relocation lacks output/base evidence")
            if set(frozen_base) != {"path", "manifest", "config_sha256"}:
                raise ExactPeftKLReferenceError(
                    "development relocation frozen-base schema mismatch"
                )
            if set(relocated) != {
                "path",
                "manifest_excluding_receipt",
                "adapter_config_sha256",
                "adapter_model_sha256",
                "configured_base_path",
                "serialization_bias_policy",
                "checkpoint_fidelity_adapter_inspection",
            }:
                raise ExactPeftKLReferenceError("development relocation output schema mismatch")
            if set(transformation) != {
                "method",
                "changed_config_fields",
                "source_config",
                "relocated_config",
                "config_only_delta",
                "unchanged_file_paths",
                "non_config_files_bitwise_identical",
            }:
                raise ExactPeftKLReferenceError(
                    "development relocation transformation schema mismatch"
                )
            if (
                transformation.get("method") != "config_only_absolute_base_path_relocation_v1"
                or transformation.get("changed_config_fields") != ["base_model_name_or_path"]
                or transformation.get("config_only_delta") is not True
                or transformation.get("non_config_files_bitwise_identical") is not True
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation transformation invariant mismatch"
                )
            if Path(str(relocated.get("path", ""))).expanduser().absolute() != adapter_path:
                raise ExactPeftKLReferenceError("development relocation output path mismatch")
            adapter_tree = binding.get("adapter_tree")
            if not isinstance(adapter_tree, Mapping):
                raise ExactPeftKLReferenceError("development relocation adapter tree is invalid")
            files = [
                record
                for record in adapter_tree.get("files", [])
                if record.get("path") != _DEVELOPMENT_RELOCATION_RECEIPT
            ]
            manifest_without_receipt = {
                "file_count": len(files),
                "size_bytes": sum(int(record["size_bytes"]) for record in files),
                "tree_sha256": _sha256_json(files),
                "files": files,
            }
            if relocated.get("manifest_excluding_receipt") != manifest_without_receipt:
                raise ExactPeftKLReferenceError("development relocated adapter manifest mismatch")
            base_path = Path(str(binding.get("base_model_path", ""))).expanduser().absolute()
            if (
                Path(str(frozen_base.get("path", ""))).expanduser().absolute() != base_path
                or frozen_base.get("manifest") != binding.get("base_model_tree")
                or frozen_base.get("manifest") != upstream.get("live_base_manifest")
                or relocated.get("configured_base_path") != str(base_path)
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation frozen-base binding mismatch"
                )
            config_path = adapter_path / "adapter_config.json"
            weights_path = adapter_path / "adapter_model.safetensors"
            if (
                relocated.get("adapter_config_sha256") != _sha256_file(config_path)
                or relocated.get("adapter_model_sha256") != _sha256_file(weights_path)
                or relocated.get("serialization_bias_policy") != "all"
                or relocated.get("checkpoint_fidelity_adapter_inspection")
                != inspect_checkpoint_adapter(adapter_path)
            ):
                raise ExactPeftKLReferenceError("development relocated adapter evidence mismatch")
            full_state = verification.get("full_model_state")
            if not isinstance(full_state, Mapping) or set(full_state) != {
                "tensor_count",
                "parameter_count",
                "tensor_inventory_sha256",
                "tensors",
            }:
                raise ExactPeftKLReferenceError(
                    "development relocation full-state evidence schema mismatch"
                )
            state_tensors = full_state.get("tensors")
            if not isinstance(state_tensors, list) or not state_tensors:
                raise ExactPeftKLReferenceError(
                    "development relocation full-state inventory is empty"
                )
            previous_key = ""
            for record in state_tensors:
                _validate_tensor_record(record, label="development relocation state")
                key = str(record["key"])
                if key <= previous_key:
                    raise ExactPeftKLReferenceError(
                        "development relocation state keys are not unique/sorted"
                    )
                previous_key = key
            if (
                full_state.get("tensor_count") != len(state_tensors)
                or full_state.get("parameter_count")
                != sum(int(record["numel"]) for record in state_tensors)
                or full_state.get("tensor_inventory_sha256") != _sha256_json(state_tensors)
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation full-state aggregate mismatch"
                )
            functional = verification.get("functional_equivalence")
            expected_functional_keys = {
                "method",
                "input_ids",
                "input_ids_sha256",
                "logits",
                "log_probabilities",
                "argmax",
                "bitwise_equal",
                "max_absolute_error",
            }
            expected_inputs = [list(row) for row in FIXED_PROBES]
            if (
                not isinstance(functional, Mapping)
                or set(functional) != expected_functional_keys
                or functional.get("method") != "three_fixed_probes_bitwise_full_distribution_v1"
                or functional.get("input_ids") != expected_inputs
                or functional.get("input_ids_sha256") != _sha256_json(expected_inputs)
                or functional.get("bitwise_equal") is not True
                or functional.get("max_absolute_error") != 0.0
            ):
                raise ExactPeftKLReferenceError(
                    "development relocation functional evidence mismatch"
                )
            for field in ("logits", "log_probabilities", "argmax"):
                _validate_tensor_record(
                    functional.get(field),
                    label=f"development relocation {field}",
                )


def _validate_distributed(
    distributed: object,
    *,
    source_binding: Mapping[str, Any],
    runtime_pair: Mapping[str, Any],
    claim_scope: str,
) -> None:
    if not isinstance(distributed, Mapping) or set(distributed) != {
        "phase",
        "world_size",
        "rank_digests",
        "consensus_digest",
    }:
        raise ExactPeftKLReferenceError("distributed evidence schema mismatch")
    world_size = distributed.get("world_size")
    rank_digests = distributed.get("rank_digests")
    consensus = distributed.get("consensus_digest")
    if (
        distributed.get("phase") != "exact_peft_kl_step_zero"
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
        or not isinstance(rank_digests, list)
        or len(rank_digests) != world_size
        or not _valid_sha256(consensus)
    ):
        raise ExactPeftKLReferenceError("distributed evidence invariant mismatch")
    ranks: list[int] = []
    for record in rank_digests:
        if not isinstance(record, Mapping) or set(record) != {"rank", "digest"}:
            raise ExactPeftKLReferenceError("distributed rank record schema mismatch")
        ranks.append(record.get("rank"))
        if record.get("digest") != consensus:
            raise ExactPeftKLReferenceError("distributed ranks lack exact digest consensus")
    if ranks != list(range(world_size)):
        raise ExactPeftKLReferenceError("distributed rank set/order mismatch")
    expected = _sha256_json(
        {
            "source_binding": dict(source_binding),
            "runtime_pair": dict(runtime_pair),
            "claim_scope": claim_scope,
        }
    )
    if consensus != expected:
        raise ExactPeftKLReferenceError("distributed consensus digest is not reproducible")


def inspect_runtime_receipt(
    receipt_path: str | os.PathLike[str],
    *,
    claim_scope: str,
    rehash_sources: bool = True,
) -> dict[str, Any]:
    """Verify canonical receipt bytes, self-hash, schema, and bound source trees."""

    path = Path(receipt_path).expanduser().absolute()
    if path.name != RECEIPT_FILENAME:
        raise ExactPeftKLReferenceError(f"unexpected exact-reference receipt name: {path}")
    raw = _regular_read_only_file_bytes(path)
    receipt = _decode_json(raw, path=path)
    if raw != _receipt_bytes(receipt):
        raise ExactPeftKLReferenceError("exact-reference receipt is not canonically serialized")
    if set(receipt) != _TOP_LEVEL_KEYS:
        raise ExactPeftKLReferenceError("exact-reference receipt top-level schema mismatch")
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed_hash != _sha256_json(unhashed):
        raise ExactPeftKLReferenceError("exact-reference receipt self-hash mismatch")
    expected_protocol = (
        PUBLICATION_PROTOCOL if claim_scope == "publication" else DEVELOPMENT_PROTOCOL
    )
    expected_status = PUBLICATION_STATUS if claim_scope == "publication" else DEVELOPMENT_STATUS
    expected_claim = "publication" if claim_scope == "publication" else "development_only"
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != expected_protocol
        or receipt.get("status") != expected_status
        or receipt.get("claim_scope") != expected_claim
    ):
        raise ExactPeftKLReferenceError("exact-reference receipt claim boundary mismatch")
    expected_flags = (False, True, True) if claim_scope == "publication" else (True, False, False)
    if (
        receipt.get("development_only"),
        receipt.get("publication_eligible"),
        receipt.get("sota_eligible"),
    ) != expected_flags:
        raise ExactPeftKLReferenceError("exact-reference eligibility boundary mismatch")
    verifier = receipt.get("verifier")
    if not isinstance(verifier, Mapping) or verifier != {
        "protocol": VERIFIER_PROTOCOL,
        "implementation": "clasp.models.exact_peft_kl_reference",
        "implementation_sha256": _implementation_sha256(),
    }:
        raise ExactPeftKLReferenceError("exact-reference verifier identity mismatch")
    source_binding = receipt.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ExactPeftKLReferenceError("exact-reference receipt lacks source binding")
    runtime_pair = receipt.get("runtime_pair")
    if not isinstance(runtime_pair, Mapping):
        raise ExactPeftKLReferenceError("exact-reference receipt lacks runtime pair")
    _validate_source_binding(source_binding, claim_scope=claim_scope)
    training_contract = receipt.get("training_contract")
    _validate_training_contract(training_contract)
    _validate_runtime_pair(runtime_pair)
    functional = runtime_pair.get("functional_equivalence")
    if (
        not isinstance(training_contract, Mapping)
        or not isinstance(functional, Mapping)
        or functional.get("runtime_dtype") != training_contract.get("dtype")
    ):
        raise ExactPeftKLReferenceError(
            "runtime probe dtype differs from the exact training contract"
        )
    _validate_distributed(
        receipt.get("distributed"),
        source_binding=source_binding,
        runtime_pair=runtime_pair,
        claim_scope=claim_scope,
    )
    if rehash_sources:
        verify_source_binding_unchanged(source_binding)
    return receipt


def write_runtime_receipt(
    output_dir: str | os.PathLike[str],
    receipt: Mapping[str, Any],
    *,
    claim_scope: str,
) -> dict[str, Any]:
    """Atomically create, or exactly reverify, the root runtime receipt."""

    root = Path(output_dir).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    path = root / RECEIPT_FILENAME
    encoded = _receipt_bytes(receipt)
    if path.exists() or path.is_symlink():
        existing = inspect_runtime_receipt(path, claim_scope=claim_scope)
        if existing != dict(receipt) or path.read_bytes() != encoded:
            raise ExactPeftKLReferenceError(
                f"existing exact-reference receipt differs from runtime evidence: {path}"
            )
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
            raise ExactPeftKLReferenceError(f"refusing to replace runtime receipt: {path}") from exc
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
    return inspect_runtime_receipt(path, claim_scope=claim_scope)
