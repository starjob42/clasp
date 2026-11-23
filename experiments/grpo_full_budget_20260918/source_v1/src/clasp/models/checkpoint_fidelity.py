"""Fail-closed evidence that a PEFT checkpoint preserves its trained policy.

PEFT's adapter configuration is not proof that all trainable tensors were
serialized.  In particular, some PEFT releases mark LoRA-layer biases as
trainable for ``bias="lora_only"`` while omitting those biases from
``get_peft_model_state_dict``.  This module checks the stronger invariant:

* every ``requires_grad`` tensor has exactly one canonical PEFT state key;
* the PEFT state has no missing or additional tensors;
* shape, dtype, number of elements, and value digest agree for every tensor;
* the saved safetensors payload agrees with that in-memory closure; and
* a create-once receipt binds the exact adapter config and weights bytes.

The on-disk verifier deliberately does not import torch, transformers, PEFT,
or safetensors.  Publication artifacts can therefore be checked on a small
CPU-only machine after the expensive model has been released from memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROTOCOL = "clasp-checkpoint-fidelity-v1"
CLOSURE_PROTOCOL = "clasp-trainable-peft-state-closure-v1"
QWEN_PUBLICATION_PROTOCOL = "clasp-qwen25-7b-publication-lora-v1"
RECEIPT_FILENAME = "checkpoint_fidelity.json"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"

QWEN_NUM_LAYERS = 28
QWEN_HIDDEN_SIZE = 3_584
QWEN_INTERMEDIATE_SIZE = 18_944
QWEN_NUM_ATTENTION_HEADS = 28
QWEN_NUM_KEY_VALUE_HEADS = 4
QWEN_HEAD_DIM = QWEN_HIDDEN_SIZE // QWEN_NUM_ATTENTION_HEADS
QWEN_KV_SIZE = QWEN_NUM_KEY_VALUE_HEADS * QWEN_HEAD_DIM
QWEN_LORA_RANK = 64
QWEN_LORA_TARGETS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
QWEN_EXPECTED_LORA_TENSORS = 392
QWEN_EXPECTED_BIAS_TENSORS = 84
QWEN_EXPECTED_TRAINABLE_TENSORS = 476
QWEN_EXPECTED_LORA_PARAMETERS = 161_480_704
QWEN_EXPECTED_BIAS_PARAMETERS = 129_024
QWEN_EXPECTED_TRAINABLE_PARAMETERS = 161_609_728

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_TORCH_TO_SAFETENSORS_DTYPE = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.float8_e4m3fn": "F8_E4M3FN",
    "torch.float8_e5m2": "F8_E5M2",
    "torch.uint16": "U16",
    "torch.int16": "I16",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.uint32": "U32",
    "torch.int32": "I32",
    "torch.float32": "F32",
    "torch.uint64": "U64",
    "torch.int64": "I64",
    "torch.float64": "F64",
}
_TENSOR_RECORD_KEYS = frozenset({"key", "shape", "dtype", "numel", "value_sha256"})
_CLOSURE_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "adapter_name",
        "tensor_count",
        "parameter_count",
        "tensor_inventory_sha256",
        "tensors",
    }
)


class CheckpointFidelityError(ValueError):
    """The trainable policy and its serialized checkpoint are not identical."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointFidelityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json_object(value: bytes, *, source: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(value, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointFidelityError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise CheckpointFidelityError(f"expected a JSON object in {source}")
    return decoded


def _regular_file_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CheckpointFidelityError(f"cannot stat required file {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointFidelityError(f"required path is not a regular file: {path}")
    try:
        value = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise CheckpointFidelityError(f"cannot read required file {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise CheckpointFidelityError(f"file changed while it was read: {path}")
    return value


def _checkpoint_root(checkpoint_dir: str | os.PathLike[str]) -> Path:
    root = Path(checkpoint_dir).expanduser().absolute()
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise CheckpointFidelityError(f"cannot stat checkpoint directory {root}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointFidelityError(f"checkpoint path is not a regular directory: {root}")
    return root


def canonicalize_peft_adapter_key(key: str, adapter_name: str = "default") -> str:
    """Remove PEFT's runtime-only adapter-name component from a state key.

    This mirrors ``get_peft_model_state_dict`` for both module-owned parameters
    (``lora_A.default.weight``) and directly owned parameters
    (``lora_embedding_A.default``).  A known adapter name is removed only at
    the final or penultimate component, never from an ordinary module name.
    """

    if not isinstance(key, str) or not key:
        raise CheckpointFidelityError("PEFT state keys must be non-empty strings")
    if not isinstance(adapter_name, str) or not adapter_name or "." in adapter_name:
        raise CheckpointFidelityError("adapter_name must be a non-empty path component")
    adapter_suffix = f".{adapter_name}"
    if key.endswith(adapter_suffix):
        return key[: -len(adapter_suffix)]
    stem, separator, leaf = key.rpartition(".")
    if separator and stem.endswith(adapter_suffix):
        return f"{stem[: -len(adapter_suffix)]}.{leaf}"
    return key


def _canonicalize_mapping(
    values: Mapping[str, Any],
    *,
    adapter_name: str,
    source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    originals: dict[str, str] = {}
    for original, value in values.items():
        canonical = canonicalize_peft_adapter_key(original, adapter_name)
        if canonical in result:
            raise CheckpointFidelityError(
                f"canonical PEFT key collision in {source}: "
                f"{originals[canonical]!r} and {original!r} -> {canonical!r}"
            )
        result[canonical] = value
        originals[canonical] = original
    return result


def _tensor_record(key: str, tensor: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in dependency-light use
        raise CheckpointFidelityError("torch is required for in-memory fidelity auditing") from exc

    dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(str(getattr(tensor, "dtype", None)))
    if dtype is None:
        raise CheckpointFidelityError(
            f"tensor {key!r} has unsupported dtype {getattr(tensor, 'dtype', None)!r}"
        )
    try:
        detached = tensor.detach()
        if detached.layout != torch.strided:
            raise CheckpointFidelityError(f"tensor {key!r} is not strided")
        if detached.device.type == "meta":
            raise CheckpointFidelityError(f"tensor {key!r} is on the meta device")
        detached = detached.resolve_conj().resolve_neg().to(device="cpu").contiguous()
        shape = [int(dimension) for dimension in detached.shape]
        numel = int(detached.numel())
        raw = detached.view(torch.uint8).numpy().tobytes(order="C")
    except CheckpointFidelityError:
        raise
    except Exception as exc:
        raise CheckpointFidelityError(f"cannot materialize tensor {key!r}: {exc}") from exc
    expected_bytes = numel * _SAFETENSORS_DTYPE_BYTES[dtype]
    if len(raw) != expected_bytes:
        raise CheckpointFidelityError(
            f"tensor {key!r} byte length mismatch: {len(raw)} != {expected_bytes}"
        )
    return {
        "key": key,
        "shape": shape,
        "dtype": dtype,
        "numel": numel,
        "value_sha256": _sha256_bytes(raw),
    }


def _validated_tensor_inventory(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise CheckpointFidelityError("tensor inventory must be a list")
    records: list[dict[str, Any]] = []
    previous_key: str | None = None
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != _TENSOR_RECORD_KEYS:
            raise CheckpointFidelityError(f"invalid tensor inventory record at index {index}")
        key = value["key"]
        shape = value["shape"]
        dtype = value["dtype"]
        numel = value["numel"]
        digest = value["value_sha256"]
        if (
            not isinstance(key, str)
            or not key
            or (previous_key is not None and key <= previous_key)
        ):
            raise CheckpointFidelityError("tensor inventory keys must be unique and sorted")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            )
            or dtype not in _SAFETENSORS_DTYPE_BYTES
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel < 0
            or numel != _product(shape)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CheckpointFidelityError(f"invalid tensor metadata for {key!r}")
        records.append(
            {
                "key": key,
                "shape": list(shape),
                "dtype": dtype,
                "numel": numel,
                "value_sha256": digest,
            }
        )
        previous_key = key
    return records


def _build_closure(adapter_name: str, tensors: list[dict[str, Any]]) -> dict[str, Any]:
    validated = _validated_tensor_inventory(tensors)
    if not validated:
        raise CheckpointFidelityError("trainable tensor inventory is empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": CLOSURE_PROTOCOL,
        "status": "exact",
        "adapter_name": adapter_name,
        "tensor_count": len(validated),
        "parameter_count": sum(record["numel"] for record in validated),
        "tensor_inventory_sha256": _sha256_json(validated),
        "tensors": validated,
    }


def _validated_closure(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _CLOSURE_KEYS:
        raise CheckpointFidelityError("invalid trainable-state closure schema")
    adapter_name = raw.get("adapter_name")
    canonicalize_peft_adapter_key("probe", adapter_name)
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("protocol") != CLOSURE_PROTOCOL
        or raw.get("status") != "exact"
    ):
        raise CheckpointFidelityError("unsupported or unsuccessful trainable-state closure")
    expected = _build_closure(adapter_name, raw.get("tensors"))
    if raw != expected:
        raise CheckpointFidelityError("trainable-state closure summary is inconsistent")
    return expected


def _get_peft_model_state(model: Any, *, adapter_name: str) -> Mapping[str, Any]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - exercised in dependency-light use
        raise CheckpointFidelityError("PEFT is required for in-memory fidelity auditing") from exc
    try:
        result = get_peft_model_state_dict(
            model,
            state_dict=model.state_dict(),
            adapter_name=adapter_name,
            save_embedding_layers=False,
        )
    except Exception as exc:
        raise CheckpointFidelityError(f"PEFT state extraction failed: {exc}") from exc
    if not isinstance(result, Mapping):
        raise CheckpointFidelityError("get_peft_model_state_dict did not return a mapping")
    return result


def audit_trainable_state_closure(
    model: Any,
    adapter_name: str = "default",
) -> dict[str, Any]:
    """Prove exact closure between trainable parameters and PEFT's save state."""

    canonicalize_peft_adapter_key("probe", adapter_name)
    try:
        named_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    except Exception as exc:
        raise CheckpointFidelityError(f"cannot enumerate trainable parameters: {exc}") from exc
    if not named_parameters:
        raise CheckpointFidelityError("model has no requires_grad tensors")
    trainable = _canonicalize_mapping(
        named_parameters,
        adapter_name=adapter_name,
        source="model.named_parameters",
    )
    peft_state = _canonicalize_mapping(
        _get_peft_model_state(model, adapter_name=adapter_name),
        adapter_name=adapter_name,
        source="get_peft_model_state_dict",
    )
    missing = sorted(set(trainable) - set(peft_state))
    unexpected = sorted(set(peft_state) - set(trainable))
    if missing or unexpected:
        raise CheckpointFidelityError(
            "trainable/PEFT state key closure mismatch: "
            f"missing_from_peft={missing[:12]}, unexpected_in_peft={unexpected[:12]}"
        )

    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for key in sorted(trainable):
        trainable_record = _tensor_record(key, trainable[key])
        peft_record = _tensor_record(key, peft_state[key])
        if trainable_record != peft_record:
            differing = [
                field
                for field in ("shape", "dtype", "numel", "value_sha256")
                if trainable_record[field] != peft_record[field]
            ]
            mismatches.append(f"{key} ({','.join(differing)})")
        records.append(trainable_record)
    if mismatches:
        raise CheckpointFidelityError(
            "trainable/PEFT tensor fidelity mismatch: " + ", ".join(mismatches[:12])
        )
    return _build_closure(adapter_name, records)


def expected_qwen25_7b_publication_shapes() -> dict[str, tuple[int, ...]]:
    """Return the exact canonical trainable inventory for publication LoRA."""

    dimensions = {
        "q_proj": (QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE),
        "k_proj": (QWEN_HIDDEN_SIZE, QWEN_KV_SIZE),
        "v_proj": (QWEN_HIDDEN_SIZE, QWEN_KV_SIZE),
        "o_proj": (QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE),
        "gate_proj": (QWEN_HIDDEN_SIZE, QWEN_INTERMEDIATE_SIZE),
        "up_proj": (QWEN_HIDDEN_SIZE, QWEN_INTERMEDIATE_SIZE),
        "down_proj": (QWEN_INTERMEDIATE_SIZE, QWEN_HIDDEN_SIZE),
    }
    expected: dict[str, tuple[int, ...]] = {}
    for layer in range(QWEN_NUM_LAYERS):
        root = f"base_model.model.model.layers.{layer}"
        for module, (input_size, output_size) in dimensions.items():
            block = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"{root}.{block}.{module}"
            expected[f"{prefix}.lora_A.weight"] = (QWEN_LORA_RANK, input_size)
            expected[f"{prefix}.lora_B.weight"] = (output_size, QWEN_LORA_RANK)
        attention = f"{root}.self_attn"
        expected[f"{attention}.q_proj.base_layer.bias"] = (QWEN_HIDDEN_SIZE,)
        expected[f"{attention}.k_proj.base_layer.bias"] = (QWEN_KV_SIZE,)
        expected[f"{attention}.v_proj.base_layer.bias"] = (QWEN_KV_SIZE,)
    return expected


def enable_qwen25_targeted_bias_training(
    model: Any,
    adapter_name: str = "default",
) -> dict[str, Any]:
    """Restore the 84 Q/K/V biases frozen by Unsloth's PEFT post-patch.

    Unsloth 2025.3.19 calls its generic model-for-training preparation after
    PEFT has applied ``bias="all"``.  That preparation leaves only LoRA (and
    optional magnitude-vector) parameters trainable.  Publication training
    explicitly re-enables the exact Qwen2.5-7B Q/K/V bias inventory before the
    trainable-to-serialized closure audit.
    """

    expected = {
        key: shape
        for key, shape in expected_qwen25_7b_publication_shapes().items()
        if key.endswith(".base_layer.bias")
    }
    observed: dict[str, tuple[str, Any]] = {}
    try:
        named_parameters = list(model.named_parameters())
    except Exception as exc:
        raise CheckpointFidelityError(f"cannot enumerate Qwen bias parameters: {exc}") from exc
    for raw_name, parameter in named_parameters:
        canonical = canonicalize_peft_adapter_key(raw_name, adapter_name)
        if canonical not in expected:
            continue
        if canonical in observed:
            raise CheckpointFidelityError(
                f"duplicate canonical Qwen bias parameter: {canonical!r}"
            )
        shape = tuple(int(dimension) for dimension in getattr(parameter, "shape", ()))
        if shape != expected[canonical]:
            raise CheckpointFidelityError(
                f"Qwen bias shape mismatch for {canonical}: {shape} != {expected[canonical]}"
            )
        observed[canonical] = (raw_name, parameter)
    missing = sorted(set(expected) - set(observed))
    if missing or len(observed) != QWEN_EXPECTED_BIAS_TENSORS:
        raise CheckpointFidelityError(
            "cannot restore exact Qwen2.5-7B bias training scope: "
            f"missing={missing[:12]}, observed={len(observed)}"
        )
    for _raw_name, parameter in observed.values():
        try:
            parameter.requires_grad_(True)
        except Exception as exc:
            raise CheckpointFidelityError("cannot re-enable a targeted Qwen bias") from exc
    still_frozen = sorted(
        canonical
        for canonical, (_raw_name, parameter) in observed.items()
        if not bool(getattr(parameter, "requires_grad", False))
    )
    if still_frozen:
        raise CheckpointFidelityError(
            f"targeted Qwen biases remain frozen after restoration: {still_frozen[:12]}"
        )
    return {
        "protocol": "clasp-qwen25-7b-unsloth-bias-trainability-restore-v1",
        "status": "exact",
        "adapter_name": adapter_name,
        "bias_tensor_count": len(observed),
        "bias_parameter_count": sum(_product(shape) for shape in expected.values()),
        "bias_keys_sha256": _sha256_json(sorted(observed)),
    }


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _normalized_peft_type(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value).upper()


def _qwen_publication_invariant(
    adapter_config: Any,
    tensors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if _normalized_peft_type(_config_value(adapter_config, "peft_type")) != "LORA":
        raise CheckpointFidelityError("publication adapter must use PEFT LoRA")
    if _config_value(adapter_config, "bias") != "all":
        raise CheckpointFidelityError("publication adapter must use bias='all'")
    rank = _config_value(adapter_config, "r")
    if isinstance(rank, bool) or rank != QWEN_LORA_RANK:
        raise CheckpointFidelityError(f"publication adapter must use LoRA rank {QWEN_LORA_RANK}")
    raw_targets = _config_value(adapter_config, "target_modules")
    if isinstance(raw_targets, str) or not isinstance(raw_targets, (list, tuple, set, frozenset)):
        raise CheckpointFidelityError("publication target_modules must be an explicit collection")
    if set(raw_targets) != QWEN_LORA_TARGETS:
        raise CheckpointFidelityError(
            "publication adapter must target exactly the seven Jiang modules"
        )

    records = _validated_tensor_inventory(list(tensors))
    expected = expected_qwen25_7b_publication_shapes()
    observed = {record["key"]: tuple(record["shape"]) for record in records}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    wrong_shapes = sorted(
        key for key in set(expected) & set(observed) if expected[key] != observed[key]
    )
    if missing or unexpected or wrong_shapes:
        raise CheckpointFidelityError(
            "Qwen publication tensor inventory mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"wrong_shapes={wrong_shapes[:8]}"
        )
    bias_records = [record for record in records if record["key"].endswith("base_layer.bias")]
    lora_records = [record for record in records if ".lora_" in record["key"]]
    counts = {
        "bias_tensor_count": len(bias_records),
        "bias_parameter_count": sum(record["numel"] for record in bias_records),
        "lora_tensor_count": len(lora_records),
        "lora_parameter_count": sum(record["numel"] for record in lora_records),
        "trainable_tensor_count": len(records),
        "trainable_parameter_count": sum(record["numel"] for record in records),
    }
    expected_counts = {
        "bias_tensor_count": QWEN_EXPECTED_BIAS_TENSORS,
        "bias_parameter_count": QWEN_EXPECTED_BIAS_PARAMETERS,
        "lora_tensor_count": QWEN_EXPECTED_LORA_TENSORS,
        "lora_parameter_count": QWEN_EXPECTED_LORA_PARAMETERS,
        "trainable_tensor_count": QWEN_EXPECTED_TRAINABLE_TENSORS,
        "trainable_parameter_count": QWEN_EXPECTED_TRAINABLE_PARAMETERS,
    }
    if counts != expected_counts:
        raise CheckpointFidelityError(
            f"Qwen publication count invariant mismatch: {counts} != {expected_counts}"
        )
    return {
        "protocol": QWEN_PUBLICATION_PROTOCOL,
        "status": "exact",
        "architecture": "Qwen2.5-7B",
        "bias_policy": "all",
        "lora_rank": QWEN_LORA_RANK,
        "target_modules": sorted(QWEN_LORA_TARGETS),
        **counts,
        "expected_keys_and_shapes_sha256": _sha256_json(
            {key: list(shape) for key, shape in sorted(expected.items())}
        ),
    }


def assert_qwen25_7b_publication_invariant(
    model: Any,
    closure: Mapping[str, Any],
    adapter_name: str = "default",
) -> dict[str, Any]:
    """Require the exact 7B Jiang-LoRA publication trainable scope."""

    validated = _validated_closure(closure)
    if validated["adapter_name"] != adapter_name:
        raise CheckpointFidelityError("closure adapter_name does not match requested adapter")
    try:
        adapter_config = model.peft_config[adapter_name]
    except Exception as exc:
        raise CheckpointFidelityError(
            f"cannot resolve PEFT adapter {adapter_name!r}: {exc}"
        ) from exc
    model_config = getattr(model, "config", None)
    expected_architecture = {
        "model_type": "qwen2",
        "hidden_size": QWEN_HIDDEN_SIZE,
        "intermediate_size": QWEN_INTERMEDIATE_SIZE,
        "num_hidden_layers": QWEN_NUM_LAYERS,
        "num_attention_heads": QWEN_NUM_ATTENTION_HEADS,
        "num_key_value_heads": QWEN_NUM_KEY_VALUE_HEADS,
    }
    mismatches = {
        key: {"actual": _config_value(model_config, key), "expected": expected}
        for key, expected in expected_architecture.items()
        if _config_value(model_config, key) != expected
    }
    if mismatches:
        raise CheckpointFidelityError(f"model is not audited Qwen2.5-7B: {mismatches}")
    return _qwen_publication_invariant(adapter_config, validated["tensors"])


def _read_safetensors(path: Path) -> dict[str, Any]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CheckpointFidelityError(f"cannot stat adapter weights {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointFidelityError(f"adapter weights are not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise CheckpointFidelityError("safetensors file is truncated before its header")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise CheckpointFidelityError(f"unsafe safetensors header length: {header_length}")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise CheckpointFidelityError("safetensors header is truncated")
            header = _decode_json_object(header_bytes, source=path)
            metadata = header.pop("__metadata__", None)
            if metadata is not None and (
                not isinstance(metadata, dict)
                or any(
                    not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
                )
            ):
                raise CheckpointFidelityError("invalid safetensors __metadata__")
            payload_size = before.st_size - 8 - header_length
            if payload_size < 0:
                raise CheckpointFidelityError("safetensors file is shorter than its header")

            entries: list[dict[str, Any]] = []
            intervals: list[tuple[int, int, str]] = []
            for key, raw in header.items():
                if not isinstance(key, str) or not key or not isinstance(raw, dict):
                    raise CheckpointFidelityError("invalid safetensors tensor entry")
                if set(raw) != {"dtype", "shape", "data_offsets"}:
                    raise CheckpointFidelityError(f"invalid safetensors metadata for {key!r}")
                dtype = raw.get("dtype")
                shape = raw.get("shape")
                offsets = raw.get("data_offsets")
                if (
                    dtype not in _SAFETENSORS_DTYPE_BYTES
                    or not isinstance(shape, list)
                    or any(
                        isinstance(dimension, bool)
                        or not isinstance(dimension, int)
                        or dimension < 0
                        for dimension in shape
                    )
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(
                        isinstance(offset, bool) or not isinstance(offset, int)
                        for offset in offsets
                    )
                ):
                    raise CheckpointFidelityError(f"invalid safetensors metadata for {key!r}")
                start, end = offsets
                numel = _product(shape)
                expected_bytes = numel * _SAFETENSORS_DTYPE_BYTES[dtype]
                if start < 0 or end < start or end > payload_size or end - start != expected_bytes:
                    raise CheckpointFidelityError(f"invalid safetensors payload extent for {key!r}")
                entries.append(
                    {
                        "key": key,
                        "shape": list(shape),
                        "dtype": dtype,
                        "numel": numel,
                    }
                )
                intervals.append((start, end, key))
            if not entries:
                raise CheckpointFidelityError("adapter safetensors contains no tensors")
            cursor = 0
            for start, end, key in sorted(intervals):
                if start != cursor:
                    raise CheckpointFidelityError(
                        f"safetensors payload has a gap or overlap before {key!r}"
                    )
                cursor = end
            if cursor != payload_size:
                raise CheckpointFidelityError("safetensors payload has unreferenced trailing bytes")

            artifact_digest = hashlib.sha256(length_bytes + header_bytes)
            value_digests: dict[str, str] = {}
            for start, end, key in sorted(intervals):
                handle.seek(8 + header_length + start)
                remaining = end - start
                tensor_digest = hashlib.sha256()
                while remaining:
                    chunk = handle.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise CheckpointFidelityError(
                            f"safetensors payload is truncated for {key!r}"
                        )
                    artifact_digest.update(chunk)
                    tensor_digest.update(chunk)
                    remaining -= len(chunk)
                value_digests[key] = tensor_digest.hexdigest()
            after = path.lstat()
    except CheckpointFidelityError:
        raise
    except OSError as exc:
        raise CheckpointFidelityError(f"cannot read adapter weights {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise CheckpointFidelityError(f"adapter weights changed while read: {path}")
    records = [
        {**entry, "value_sha256": value_digests[entry["key"]]}
        for entry in sorted(entries, key=lambda item: item["key"])
    ]
    records = _validated_tensor_inventory(records)
    return {
        "filename": ADAPTER_WEIGHTS_FILENAME,
        "sha256": artifact_digest.hexdigest(),
        "size_bytes": before.st_size,
        "header_sha256": _sha256_bytes(header_bytes),
        "tensor_count": len(records),
        "parameter_count": sum(record["numel"] for record in records),
        "tensor_inventory_sha256": _sha256_json(records),
        "tensors": records,
    }


def inspect_checkpoint_adapter(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Inspect adapter bytes without importing any ML dependency."""

    root = _checkpoint_root(checkpoint_dir)
    config_path = root / ADAPTER_CONFIG_FILENAME
    config_bytes = _regular_file_bytes(config_path)
    config = _decode_json_object(config_bytes, source=config_path)
    weights = _read_safetensors(root / ADAPTER_WEIGHTS_FILENAME)
    return {
        "adapter_config": config,
        "artifacts": {
            "adapter_config": {
                "filename": ADAPTER_CONFIG_FILENAME,
                "sha256": _sha256_bytes(config_bytes),
                "size_bytes": len(config_bytes),
            },
            "adapter_weights": {key: value for key, value in weights.items() if key != "tensors"},
        },
        "tensors": weights["tensors"],
    }


def _compare_inventories(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> None:
    expected_by_key = {record["key"]: record for record in expected}
    observed_by_key = {record["key"]: record for record in observed}
    missing = sorted(set(expected_by_key) - set(observed_by_key))
    unexpected = sorted(set(observed_by_key) - set(expected_by_key))
    differing = sorted(
        key
        for key in set(expected_by_key) & set(observed_by_key)
        if expected_by_key[key] != observed_by_key[key]
    )
    if missing or unexpected or differing:
        raise CheckpointFidelityError(
            "in-memory/safetensors fidelity mismatch: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}, "
            f"metadata_or_value_mismatch={differing[:12]}"
        )


def build_checkpoint_fidelity_receipt(
    checkpoint_dir: str | os.PathLike[str],
    closure: Mapping[str, Any],
    publication_invariant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build (but do not write) an exact artifact-bound fidelity receipt."""

    validated_closure = _validated_closure(closure)
    inspected = inspect_checkpoint_adapter(checkpoint_dir)
    _compare_inventories(validated_closure["tensors"], inspected["tensors"])
    if publication_invariant is not None:
        expected_publication = _qwen_publication_invariant(
            inspected["adapter_config"],
            inspected["tensors"],
        )
        if publication_invariant != expected_publication:
            raise CheckpointFidelityError(
                "provided publication invariant does not match serialized adapter"
            )
    else:
        expected_publication = None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "exact_trainable_policy_persisted",
        "adapter_name": validated_closure["adapter_name"],
        "artifacts": inspected["artifacts"],
        "trainable_closure": validated_closure,
        "publication_invariant": expected_publication,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def verify_checkpoint_fidelity(
    checkpoint_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify a receipt and its live config/weights using only the stdlib."""

    root = _checkpoint_root(checkpoint_dir)
    receipt_path = root / RECEIPT_FILENAME
    receipt_bytes = _regular_file_bytes(receipt_path)
    try:
        mode = stat.S_IMODE(receipt_path.lstat().st_mode)
    except OSError as exc:
        raise CheckpointFidelityError(
            f"cannot stat fidelity receipt {receipt_path}: {exc}"
        ) from exc
    if mode & 0o222:
        raise CheckpointFidelityError(f"fidelity receipt must be read-only: {receipt_path}")
    receipt = _decode_json_object(receipt_bytes, source=receipt_path)
    if receipt_bytes != _receipt_bytes(receipt):
        raise CheckpointFidelityError("fidelity receipt is not in canonical serialized form")
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not isinstance(claimed_hash, str) or claimed_hash != _sha256_json(unhashed):
        raise CheckpointFidelityError("fidelity receipt self-hash mismatch")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != "exact_trainable_policy_persisted"
    ):
        raise CheckpointFidelityError("unsupported or unsuccessful fidelity receipt")
    closure = receipt.get("trainable_closure")
    publication = receipt.get("publication_invariant")
    if publication is not None and not isinstance(publication, dict):
        raise CheckpointFidelityError("publication_invariant must be an object or null")
    expected = build_checkpoint_fidelity_receipt(root, closure, publication)
    if receipt != expected:
        raise CheckpointFidelityError("fidelity receipt differs from live checkpoint evidence")
    return expected


def _atomic_create_read_only(path: Path, value: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CheckpointFidelityError(f"refusing to replace fidelity receipt: {path}") from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_checkpoint_fidelity_receipt(
    checkpoint_dir: str | os.PathLike[str],
    closure: Mapping[str, Any],
    publication_invariant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically create, or exactly verify, ``checkpoint_fidelity.json``."""

    root = _checkpoint_root(checkpoint_dir)
    receipt = build_checkpoint_fidelity_receipt(root, closure, publication_invariant)
    path = root / RECEIPT_FILENAME
    encoded = _receipt_bytes(receipt)
    if path.exists() or path.is_symlink():
        existing = verify_checkpoint_fidelity(root)
        if existing != receipt or _regular_file_bytes(path) != encoded:
            raise CheckpointFidelityError(
                f"existing fidelity receipt does not match requested evidence: {path}"
            )
        return existing
    _atomic_create_read_only(path, encoded)
    return verify_checkpoint_fidelity(root)


def audit_and_write_checkpoint_fidelity(
    model: Any,
    checkpoint_dir: str | os.PathLike[str],
    adapter_name: str = "default",
    *,
    require_qwen25_7b_publication: bool = False,
) -> dict[str, Any]:
    """Run the in-memory audit and bind it to an already-saved checkpoint."""

    closure = audit_trainable_state_closure(model, adapter_name)
    publication = (
        assert_qwen25_7b_publication_invariant(model, closure, adapter_name)
        if require_qwen25_7b_publication
        else None
    )
    return write_checkpoint_fidelity_receipt(checkpoint_dir, closure, publication)
