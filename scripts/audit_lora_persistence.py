#!/usr/bin/env python3
"""Audit the serialization-safe Qwen2.5-7B Jiang-LoRA bias policy.

PEFT 0.18.0 and 0.19.1 omit trained biases when an adapter configured with
``bias="lora_only"`` is saved.  For the frozen non-instruction Qwen2.5-7B
architecture, ``bias="all"`` has exactly the intended trainable scope because
the base model's only biases are the q/k/v projection biases targeted by LoRA.

This dependency-free audit fails closed on both parts of that equivalence:

* the base weight index must contain exactly the 84 expected q/k/v biases; and
* an adapter must contain exactly those biases plus the 392 expected Jiang
  LoRA tensors, with exact shapes and a valid safetensors payload layout.

The returned receipt includes a value hash for each serialized bias.  An
optional output receipt is create-once and read-only so publication launchers
can bind the deployable artifact rather than trusting ``adapter_config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL = "clasp-qwen25-7b-lora-persistence-audit-v1"
RECEIPT_FILENAME = "adapter_persistence_receipt.json"
LORA_BIAS_POLICY = "all"
NUM_LAYERS = 28
HIDDEN_SIZE = 3_584
INTERMEDIATE_SIZE = 18_944
NUM_ATTENTION_HEADS = 28
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
KV_SIZE = NUM_KEY_VALUE_HEADS * HEAD_DIM
LORA_RANK = 64
JIANG_LORA_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
EXPECTED_BIAS_TENSORS = 84
EXPECTED_BIAS_PARAMETERS = 129_024
EXPECTED_LORA_TENSORS = 392
EXPECTED_LORA_PARAMETERS = 161_480_704
EXPECTED_TRAINABLE_TENSORS = EXPECTED_BIAS_TENSORS + EXPECTED_LORA_TENSORS
EXPECTED_TRAINABLE_PARAMETERS = EXPECTED_BIAS_PARAMETERS + EXPECTED_LORA_PARAMETERS
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _require_qwen25_dimensions(config: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "qwen2",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"base model is not the audited Qwen2.5-7B architecture: {mismatches}")


def expected_base_bias_shapes() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        expected[f"{prefix}.q_proj.bias"] = (HIDDEN_SIZE,)
        expected[f"{prefix}.k_proj.bias"] = (KV_SIZE,)
        expected[f"{prefix}.v_proj.bias"] = (KV_SIZE,)
    return expected


def expected_adapter_bias_shapes() -> dict[str, tuple[int, ...]]:
    return {
        f"base_model.model.{key.removesuffix('.bias')}.base_layer.bias": shape
        for key, shape in expected_base_bias_shapes().items()
    }


def expected_adapter_lora_shapes() -> dict[str, tuple[int, ...]]:
    module_dimensions = {
        "q_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "k_proj": (HIDDEN_SIZE, KV_SIZE),
        "v_proj": (HIDDEN_SIZE, KV_SIZE),
        "o_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "gate_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "up_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "down_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    }
    expected: dict[str, tuple[int, ...]] = {}
    for layer in range(NUM_LAYERS):
        for module, (input_size, output_size) in module_dimensions.items():
            block = "self_attn" if module.endswith("_proj") and module in {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            } else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{block}.{module}"
            expected[f"{prefix}.lora_A.weight"] = (LORA_RANK, input_size)
            expected[f"{prefix}.lora_B.weight"] = (output_size, LORA_RANK)
    return expected


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], bytes, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"safetensors file is truncated before its header: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(
                    f"safetensors header length is unsafe in {path}: {header_length}"
                )
            header_bytes = handle.read(header_length)
    except OSError as exc:
        raise ValueError(f"cannot read safetensors artifact {path}: {exc}") from exc
    if len(header_bytes) != header_length or file_size < 8 + header_length:
        raise ValueError(f"safetensors header is truncated in {path}")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors header is invalid JSON in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object in {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"safetensors artifact contains no tensors: {path}")
    return tensors, header_bytes, file_size


def _validate_tensor_layout(
    *,
    tensors: Mapping[str, Any],
    payload_size: int,
    path: Path,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for key, raw in tensors.items():
        if not isinstance(key, str) or not key or not isinstance(raw, dict):
            raise ValueError(f"invalid safetensors tensor entry in {path}")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype for {key} in {path}: {dtype!r}")
        if (
            not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"invalid shape or data offsets for {key} in {path}")
        start, end = offsets
        expected_bytes = _product(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > payload_size or end - start != expected_bytes:
            raise ValueError(f"invalid payload extent for {key} in {path}")
        normalized[key] = {
            "dtype": dtype,
            "shape": tuple(shape),
            "data_offsets": (start, end),
        }
        ranges.append((start, end, key))
    cursor = 0
    for start, end, key in sorted(ranges):
        if start != cursor:
            raise ValueError(f"safetensors payload has a gap or overlap before {key} in {path}")
        cursor = end
    if cursor != payload_size:
        raise ValueError(f"safetensors payload has unreferenced trailing bytes in {path}")
    return normalized


def _hash_tensor_payloads(
    path: Path,
    *,
    data_start: int,
    tensors: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[str, str]:
    before = path.stat()
    hashes: dict[str, str] = {}
    try:
        with path.open("rb") as handle:
            for key in keys:
                start, end = tensors[key]["data_offsets"]
                handle.seek(data_start + int(start))
                remaining = int(end) - int(start)
                digest = hashlib.sha256()
                while remaining:
                    chunk = handle.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"tensor payload is truncated for {key} in {path}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                hashes[key] = digest.hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash tensor payloads in {path}: {exc}") from exc
    after = path.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ValueError(f"adapter changed while its bias payloads were audited: {path}")
    return hashes


def audit_qwen25_base_bias_scope(base_model_path: Path) -> dict[str, Any]:
    raw_root = base_model_path.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError(f"base model must not be a symbolic link: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"base model is not a regular directory: {root}")
    config = _load_object(root / "config.json")
    _require_qwen25_dimensions(config)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError(
            "publication bias audit requires model.safetensors.index.json; "
            f"cannot prove bias=all scope for {root}"
        )
    index = _load_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"base safetensors index has no weight_map: {index_path}")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError(f"base safetensors index has invalid weight_map entries: {index_path}")
    unsafe_shards = sorted(
        shard
        for shard in set(weight_map.values())
        if Path(shard).is_absolute()
        or ".." in Path(shard).parts
        or (root / shard).resolve().parent != root
    )
    if unsafe_shards:
        raise ValueError("base safetensors index has unsafe shard paths: " + ", ".join(unsafe_shards))
    missing_shards = sorted(
        shard
        for shard in set(weight_map.values())
        if not (root / shard).is_file() or (root / shard).is_symlink()
    )
    if missing_shards:
        raise ValueError("base safetensors index references missing shards: " + ", ".join(missing_shards))
    expected = expected_base_bias_shapes()
    observed_biases = {key for key in weight_map if key.endswith(".bias")}
    if observed_biases != set(expected):
        missing = sorted(set(expected) - observed_biases)
        unexpected = sorted(observed_biases - set(expected))
        raise ValueError(
            "bias=all is not scope-equivalent on this base model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    parameters = sum(_product(shape) for shape in expected.values())
    if len(expected) != EXPECTED_BIAS_TENSORS or parameters != EXPECTED_BIAS_PARAMETERS:
        raise ValueError("internal Qwen2.5-7B bias inventory constant mismatch")
    return {
        "policy": LORA_BIAS_POLICY,
        "scope_equivalence": "all_base_biases_equal_targeted_qkv_biases",
        "bias_tensor_count": len(expected),
        "bias_parameter_count": parameters,
        "bias_keys_sha256": _sha256_json(sorted(expected)),
        "base_weight_index_sha256": _sha256_bytes(index_path.read_bytes()),
    }


def audit_adapter_persistence(adapter_path: Path) -> dict[str, Any]:
    raw_root = adapter_path.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError(f"adapter must not be a symbolic link: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"adapter is not a regular directory: {root}")
    config = _load_object(root / "adapter_config.json")
    expected_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "lora_dropout": 0.0,
        "bias": LORA_BIAS_POLICY,
    }
    for key, expected_value in expected_config.items():
        actual = config.get(key)
        if key == "lora_dropout":
            try:
                actual = float(actual)
            except (TypeError, ValueError):
                pass
        if actual != expected_value:
            raise ValueError(f"adapter_config.{key}={actual!r}, expected {expected_value!r}")
    if set(config.get("target_modules") or []) != JIANG_LORA_MODULES:
        raise ValueError("adapter does not use the seven Jiang LoRA target modules")

    artifact = root / "adapter_model.safetensors"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"publication adapter lacks regular adapter_model.safetensors: {root}")
    tensors, header_bytes, file_size = _read_safetensors_header(artifact)
    data_start = 8 + len(header_bytes)
    normalized = _validate_tensor_layout(
        tensors=tensors,
        payload_size=file_size - data_start,
        path=artifact,
    )
    expected_bias = expected_adapter_bias_shapes()
    expected_lora = expected_adapter_lora_shapes()
    expected_all = {**expected_bias, **expected_lora}
    if set(normalized) != set(expected_all):
        missing = sorted(set(expected_all) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected_all))
        raise ValueError(
            "serialized adapter tensor inventory mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_mismatches = {
        key: {"actual": normalized[key]["shape"], "expected": expected_shape}
        for key, expected_shape in expected_all.items()
        if normalized[key]["shape"] != expected_shape
    }
    if shape_mismatches:
        preview = dict(list(sorted(shape_mismatches.items()))[:8])
        raise ValueError(f"serialized adapter tensor shape mismatch: {preview}")

    bias_parameters = sum(_product(shape) for shape in expected_bias.values())
    lora_parameters = sum(_product(shape) for shape in expected_lora.values())
    counts = (
        len(expected_bias),
        bias_parameters,
        len(expected_lora),
        lora_parameters,
    )
    expected_counts = (
        EXPECTED_BIAS_TENSORS,
        EXPECTED_BIAS_PARAMETERS,
        EXPECTED_LORA_TENSORS,
        EXPECTED_LORA_PARAMETERS,
    )
    if counts != expected_counts:
        raise ValueError(f"internal adapter inventory constant mismatch: {counts} != {expected_counts}")
    value_hashes = _hash_tensor_payloads(
        artifact,
        data_start=data_start,
        tensors=normalized,
        keys=sorted(expected_bias),
    )
    value_receipt = [
        {
            "key": key,
            "dtype": normalized[key]["dtype"],
            "shape": list(expected_bias[key]),
            "value_sha256": value_hashes[key],
        }
        for key in sorted(expected_bias)
    ]
    return {
        "adapter_config_sha256": _sha256_bytes((root / "adapter_config.json").read_bytes()),
        "adapter_file_size_bytes": file_size,
        "adapter_header_sha256": _sha256_bytes(header_bytes),
        "bias_tensor_count": len(expected_bias),
        "bias_parameter_count": bias_parameters,
        "lora_tensor_count": len(expected_lora),
        "lora_parameter_count": lora_parameters,
        "trainable_tensor_count": len(expected_all),
        "trainable_parameter_count": bias_parameters + lora_parameters,
        "bias_value_receipt": value_receipt,
        "bias_value_receipt_sha256": _sha256_json(value_receipt),
    }


def build_audit(base_model_path: Path, adapter_path: Path | None = None) -> dict[str, Any]:
    base_scope = audit_qwen25_base_bias_scope(base_model_path)
    base = base_model_path.expanduser().absolute().resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "persistence_safe",
        "base_model_path": str(base),
        "base_bias_scope": base_scope,
    }
    if adapter_path is not None:
        adapter_persistence = audit_adapter_persistence(adapter_path)
        adapter = adapter_path.expanduser().absolute().resolve()
        payload["adapter_path"] = str(adapter)
        payload["adapter_persistence"] = adapter_persistence
    payload["audit_sha256"] = _sha256_json(payload)
    return payload


def _write_or_verify_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"audit receipt is not a regular file: {destination}")
        if destination.read_bytes() != encoded:
            raise ValueError(f"existing audit receipt differs from live artifact: {destination}")
        if stat.S_IMODE(destination.stat().st_mode) & 0o222:
            raise ValueError(f"existing audit receipt is writable: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace audit receipt: {destination}") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_audit_receipt(
    path: Path,
    *,
    base_model_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    receipt = path.expanduser().absolute()
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError(f"adapter persistence receipt is not a regular file: {receipt}")
    if stat.S_IMODE(receipt.stat().st_mode) & 0o222:
        raise ValueError(f"adapter persistence receipt is writable: {receipt}")
    observed = _load_object(receipt)
    expected = build_audit(base_model_path, adapter_path)
    if observed != expected:
        raise ValueError(f"adapter persistence receipt differs from live artifact: {receipt}")
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.adapter_path is None:
            raise ValueError("--output requires --adapter-path")
        payload = build_audit(args.base_model_path, args.adapter_path)
        if args.output is not None:
            _write_or_verify_receipt(args.output, payload)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except (FileExistsError, ValueError) as exc:
        print(f"[LoRAPersistenceAuditError] {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
