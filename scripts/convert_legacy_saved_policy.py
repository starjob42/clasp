#!/usr/bin/env python3
"""Re-encode a legacy bias-dropping LoRA *saved policy* for development use.

This converter does not recover the trained Q/K/V biases that PEFT omitted
from a legacy ``bias="lora_only"`` adapter.  It loads the policy that the
legacy artifact actually deploys (frozen base biases plus its saved LoRA
tensors), then saves that same policy with ``bias="all"`` so future reloads do
not depend on the bias-dropping serialization path.

The output directory is create-once.  A successful conversion is reloaded on
the exact same local base and must have bitwise-identical canonical adapter
state, deterministic logits, and token log-probabilities.  The directory is
then made read-only and receives a self-hashed receipt that permanently marks
the artifact as development-only and ineligible for publication or SOTA
claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clasp.models.checkpoint_fidelity import (
    CheckpointFidelityError,
    canonicalize_peft_adapter_key,
    inspect_checkpoint_adapter,
)

PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
RECEIPT_FILENAME = "legacy_saved_policy_conversion_receipt.json"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
ADAPTER_NAME = "default"
FIXED_PROMPT_TOKEN_IDS = (7, 11, 13, 17, 19, 23)
QKV_MODULES = frozenset({"q_proj", "k_proj", "v_proj"})
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


class ConversionError(ValueError):
    """Raised when conversion evidence cannot be established fail-closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConversionError(f"expected regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"expected a JSON object in {path}")
    return value


def _regular_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ConversionError(f"{label} must not be a symbolic link: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ConversionError(f"cannot resolve {label} {raw}: {exc}") from exc
    if not resolved.is_dir():
        raise ConversionError(f"{label} is not a directory: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ConversionError(f"tree contains a non-regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ConversionError(f"cannot hash {path}: {exc}") from exc
    after = path.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise ConversionError(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def build_tree_manifest(root: Path, *, excluded_names: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Build a deterministic content manifest without following links."""

    root = _regular_directory(root, label="manifest root")
    files: list[dict[str, Any]] = []
    try:
        descendants = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    except OSError as exc:
        raise ConversionError(f"cannot enumerate manifest root {root}: {exc}") from exc
    for path in descendants:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ConversionError(f"manifest root contains symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConversionError(f"manifest root contains non-regular entry: {path}")
        if Path(relative).name in excluded_names:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ConversionError(f"manifest root contains no auditable files: {root}")
    return {
        "file_count": len(files),
        "files": files,
        "tree_sha256": _sha256_json(files),
    }


def _safetensor_keys(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise ConversionError(f"expected regular safetensors artifact: {path}")
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise ConversionError(f"cannot read safetensors inventory from {path}: {exc}") from exc
    if not keys:
        raise ConversionError(f"safetensors artifact contains no tensors: {path}")
    return sorted(keys)


def _validate_lora_pairs(keys: Sequence[str]) -> None:
    pairs: dict[str, set[str]] = {}
    unexpected: list[str] = []
    for key in keys:
        if key.endswith(".lora_A.weight"):
            pairs.setdefault(key.removesuffix(".lora_A.weight"), set()).add("A")
        elif key.endswith(".lora_B.weight"):
            pairs.setdefault(key.removesuffix(".lora_B.weight"), set()).add("B")
        else:
            unexpected.append(key)
    incomplete = sorted(prefix for prefix, members in pairs.items() if members != {"A", "B"})
    if unexpected or incomplete:
        raise ConversionError(
            "legacy adapter does not contain a complete LoRA-only inventory: "
            f"unexpected={unexpected[:8]}, incomplete_pairs={incomplete[:8]}"
        )


def _resolve_configured_base(configured: object) -> Path:
    if not isinstance(configured, str) or not configured:
        raise ConversionError("legacy adapter lacks base_model_name_or_path")
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        raise ConversionError(
            "legacy base_model_name_or_path must be an absolute local path; "
            f"got {configured!r}"
        )
    if configured_path.is_symlink():
        raise ConversionError(f"configured base must not be a symbolic link: {configured_path}")
    try:
        return configured_path.resolve(strict=True)
    except OSError as exc:
        raise ConversionError(f"configured legacy base cannot be resolved: {configured_path}") from exc


def audit_legacy_inputs(base_model_path: Path, legacy_adapter_path: Path) -> dict[str, Any]:
    """Fail closed unless the input is the known missing-bias artifact class."""

    base = _regular_directory(base_model_path, label="base model")
    legacy = _regular_directory(legacy_adapter_path, label="legacy adapter")
    config_path = legacy / ADAPTER_CONFIG_FILENAME
    config = _load_json_object(config_path)
    required = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "lora_only",
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ConversionError(f"legacy adapter contract mismatch: {mismatches}")
    if config.get("modules_to_save") not in (None, []):
        raise ConversionError("legacy modules_to_save are outside this converter's evidence scope")
    if config.get("use_dora") is True:
        raise ConversionError("DoRA adapters are outside this converter's evidence scope")
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or not all(
        isinstance(value, str) and value for value in target_modules
    ):
        raise ConversionError("legacy target_modules must be a non-empty string list")
    if not QKV_MODULES.issubset(target_modules):
        raise ConversionError("legacy target_modules must include q_proj, k_proj, and v_proj")
    configured_base = _resolve_configured_base(config.get("base_model_name_or_path"))
    if configured_base != base:
        raise ConversionError(
            "base mismatch: legacy adapter designates "
            f"{configured_base}, but conversion received {base}"
        )

    artifact = legacy / ADAPTER_WEIGHTS_FILENAME
    keys = _safetensor_keys(artifact)
    bias_keys = [key for key in keys if key.endswith("bias") or ".bias." in key]
    if bias_keys:
        raise ConversionError(
            "legacy artifact unexpectedly contains biases; this converter is only for the "
            f"known omitted-bias artifact class: {bias_keys[:8]}"
        )
    _validate_lora_pairs(keys)
    return {
        "base_path": str(base),
        "legacy_adapter_path": str(legacy),
        "legacy_config": config,
        "legacy_serialized_lora_keys": keys,
    }


def _canonicalize_peft_key(key: str, *, adapter_name: str = ADAPTER_NAME) -> str:
    try:
        return canonicalize_peft_adapter_key(key, adapter_name)
    except CheckpointFidelityError as exc:
        raise ConversionError(f"cannot canonicalize PEFT adapter key {key!r}: {exc}") from exc


def _logical_base_bias_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        key = key.removeprefix(prefix)
    return key.replace(".base_layer.bias", ".bias")


def _base_qkv_bias_state(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not name.endswith(".bias"):
            continue
        module = name.rsplit(".", 2)[-2]
        if module not in QKV_MODULES or ".self_attn." not in name:
            unexpected.append(name)
            continue
        result[name] = parameter.detach().cpu().clone()
    if unexpected:
        raise ConversionError(
            "bias=all is not scope-equivalent for this base; unexpected base biases: "
            f"{unexpected[:8]}"
        )
    counts = {
        module: sum(name.endswith(f".{module}.bias") for name in result)
        for module in sorted(QKV_MODULES)
    }
    if not result or len(set(counts.values())) != 1 or 0 in counts.values():
        raise ConversionError(f"base does not have a complete Q/K/V bias inventory: {counts}")
    return result


def _canonical_deployed_state(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        canonical = _canonicalize_peft_key(name)
        is_lora = canonical.endswith(LORA_SUFFIXES)
        is_qkv_bias = canonical.endswith(
            tuple(f".{module}.base_layer.bias" for module in sorted(QKV_MODULES))
        )
        if is_lora or is_qkv_bias:
            if canonical in state:
                raise ConversionError(f"duplicate canonical adapter-state key: {canonical}")
            state[canonical] = tensor.detach().cpu().clone()
    if not state:
        raise ConversionError("loaded model exposes no canonical adapter state")
    return dict(sorted(state.items()))


def _tensor_bytes(tensor: Any) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()


def _tensor_descriptor(tensor: Any) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "value_sha256": _sha256_bytes(_tensor_bytes(tensor)),
    }


def _state_inventory(state: Mapping[str, Any]) -> dict[str, Any]:
    tensors = [
        {"key": key, **_tensor_descriptor(tensor)} for key, tensor in sorted(state.items())
    ]
    return {
        "tensor_count": len(tensors),
        "parameter_count": sum(item["numel"] for item in tensors),
        "tensors": tensors,
        "state_sha256": _sha256_json(tensors),
    }


def _assert_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    import torch

    if set(left) != set(right):
        missing = sorted(set(left) - set(right))
        unexpected = sorted(set(right) - set(left))
        raise ConversionError(
            "reloaded adapter-state key mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key in sorted(left):
        if left[key].dtype != right[key].dtype or left[key].shape != right[key].shape:
            raise ConversionError(f"reloaded adapter-state metadata mismatch for {key}")
        if not torch.equal(left[key], right[key]):
            raise ConversionError(f"reloaded adapter-state value mismatch for {key}")


def _assert_base_biases_unchanged(base_biases: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    import torch

    deployed_biases = {
        _logical_base_bias_key(key): tensor
        for key, tensor in state.items()
        if key.endswith(".base_layer.bias")
    }
    if set(base_biases) != set(deployed_biases):
        missing = sorted(set(base_biases) - set(deployed_biases))
        unexpected = sorted(set(deployed_biases) - set(base_biases))
        raise ConversionError(
            "deployed Q/K/V bias inventory mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key in sorted(base_biases):
        if not torch.equal(base_biases[key], deployed_biases[key]):
            raise ConversionError(
                "legacy saved policy did not retain the exact frozen-base bias for " f"{key}"
            )


def _fixed_probe(model: Any, *, device: str) -> dict[str, Any]:
    import torch

    vocab_size = getattr(model.config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= max(FIXED_PROMPT_TOKEN_IDS):
        raise ConversionError(
            "model vocabulary is too small for the frozen deterministic probe: "
            f"vocab_size={vocab_size}"
        )
    input_ids = torch.tensor([FIXED_PROMPT_TOKEN_IDS], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits.detach().cpu()
        continuation_log_probs = (
            torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            .gather(-1, input_ids.detach().cpu()[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
    return {
        "input_ids": list(FIXED_PROMPT_TOKEN_IDS),
        "logits": logits,
        "continuation_log_probs": continuation_log_probs,
    }


def _probe_inventory(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": probe["input_ids"],
        "logits": _tensor_descriptor(probe["logits"]),
        "continuation_log_probs": _tensor_descriptor(probe["continuation_log_probs"]),
    }


def _assert_probe_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    import torch

    if left["input_ids"] != right["input_ids"]:
        raise ConversionError("deterministic probe input changed across reload")
    if not torch.equal(left["logits"], right["logits"]):
        raise ConversionError("deterministic logits are not bitwise equal across reload")
    if not torch.equal(left["continuation_log_probs"], right["continuation_log_probs"]):
        raise ConversionError("deterministic token log-probabilities are not bitwise equal across reload")


def _torch_dtype(name: str) -> Any:
    import torch

    values = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ConversionError(f"unsupported dtype: {name}") from exc


def _load_base_model(base: Path, *, device: str, dtype: str) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ConversionError(f"conversion dependencies are unavailable: {exc}") from exc
    if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
        raise ConversionError(f"requested device is unavailable: {device}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base),
            local_files_only=True,
            trust_remote_code=False,
            dtype=_torch_dtype(dtype),
            attn_implementation="eager",
        )
        return model.to(device)
    except Exception as exc:
        raise ConversionError(f"cannot load exact local base model {base}: {exc}") from exc


def _load_adapter(base_model: Any, adapter: Path) -> Any:
    try:
        from peft import PeftModel

        return PeftModel.from_pretrained(
            base_model,
            str(adapter),
            adapter_name=ADAPTER_NAME,
            is_trainable=False,
            autocast_adapter_dtype=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise ConversionError(f"cannot load saved adapter policy {adapter}: {exc}") from exc


def _assert_serialized_state(
    artifact: Path,
    expected_state: Mapping[str, Any],
    *,
    require_biases: bool,
) -> None:
    import torch
    from safetensors.torch import load_file

    keys = _safetensor_keys(artifact)
    expected_keys = sorted(expected_state)
    if keys != expected_keys:
        missing = sorted(set(expected_keys) - set(keys))
        unexpected = sorted(set(keys) - set(expected_keys))
        label = "missing converted biases" if any(key.endswith("bias") for key in missing) else "key"
        raise ConversionError(
            f"serialized adapter {label} mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if require_biases and not any(key.endswith(".base_layer.bias") for key in keys):
        raise ConversionError("serialized adapter is missing converted biases")
    try:
        serialized = load_file(artifact, device="cpu")
    except Exception as exc:
        raise ConversionError(f"cannot load serialized adapter tensors from {artifact}: {exc}") from exc
    for key in expected_keys:
        expected = expected_state[key]
        actual = serialized[key]
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise ConversionError(f"serialized adapter metadata mismatch for {key}")
        if not torch.equal(expected, actual):
            raise ConversionError(f"serialized adapter value mismatch for {key}")


def _software_versions() -> dict[str, str]:
    import peft
    import safetensors
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ConversionError(f"refusing to freeze output containing symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
            )
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _write_receipt_create_once(output: Path, payload: Mapping[str, Any]) -> Path:
    receipt = output / RECEIPT_FILENAME
    if receipt.exists() or receipt.is_symlink():
        raise ConversionError(f"refusing to replace existing conversion receipt: {receipt}")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(receipt, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            receipt.unlink()
        except FileNotFoundError:
            pass
        raise
    receipt.chmod(stat.S_IRUSR | stat.S_IRGRP)
    return receipt


def _build_receipt(
    *,
    base: Path,
    legacy: Path,
    output: Path,
    base_manifest: Mapping[str, Any],
    legacy_manifest: Mapping[str, Any],
    output_manifest: Mapping[str, Any],
    legacy_config: Mapping[str, Any],
    converted_config: Mapping[str, Any],
    state_inventory: Mapping[str, Any],
    serialized_lora_keys: Sequence[str],
    serialized_bias_keys: Sequence[str],
    probe_inventory: Mapping[str, Any],
    device: str,
    dtype: str,
) -> dict[str, Any]:
    try:
        checkpoint_inspection = inspect_checkpoint_adapter(output)
    except CheckpointFidelityError as exc:
        raise ConversionError(f"checkpoint-fidelity adapter inspection failed: {exc}") from exc
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "converted_saved_policy_verified",
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
        "claim_boundary": (
            "This artifact is a faithful re-encoding of the legacy deployable saved policy. "
            "It does not recover omitted trained biases and cannot support publication or SOTA claims."
        ),
        "base": {
            "path": str(base),
            "manifest": base_manifest,
            "config_sha256": _sha256_file(base / "config.json"),
        },
        "legacy_source": {
            "path": str(legacy),
            "manifest": legacy_manifest,
            "adapter_config_sha256": _sha256_file(legacy / ADAPTER_CONFIG_FILENAME),
            "adapter_model_sha256": _sha256_file(legacy / ADAPTER_WEIGHTS_FILENAME),
            "serialization_bias_policy": legacy_config.get("bias"),
            "serialized_lora_keys_sha256": _sha256_json(list(serialized_lora_keys)),
            "serialized_bias_tensor_count": 0,
        },
        "converted_output": {
            "path": str(output),
            "manifest_excluding_receipt": output_manifest,
            "adapter_config_sha256": _sha256_file(output / ADAPTER_CONFIG_FILENAME),
            "adapter_model_sha256": _sha256_file(output / ADAPTER_WEIGHTS_FILENAME),
            "serialization_bias_policy": converted_config.get("bias"),
            "serialized_lora_keys": list(serialized_lora_keys),
            "serialized_bias_keys": list(serialized_bias_keys),
            "checkpoint_fidelity_adapter_inspection": checkpoint_inspection,
        },
        "verification": {
            "canonical_adapter_state_bitwise_equal": True,
            "canonical_adapter_state": state_inventory,
            "qkv_biases_equal_frozen_base": True,
            "deterministic_logits_bitwise_equal": True,
            "deterministic_log_probabilities_bitwise_equal": True,
            "probe": probe_inventory,
            "device": device,
            "requested_dtype": dtype,
        },
        "software": _software_versions(),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def verify_conversion_receipt(
    output_dir: Path,
    *,
    base_model_path: Path | None = None,
    legacy_adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the self-hash, immutable claim flags, and all bound live bytes."""

    output = _regular_directory(output_dir, label="converted output")
    receipt_path = output / RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path)
    observed_hash = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    expected_hash = _sha256_json(unhashed)
    if observed_hash != expected_hash:
        raise ConversionError("conversion receipt self-hash mismatch (tamper detected)")
    required_flags = {
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
    }
    flag_mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required_flags.items()
        if receipt.get(key) is not expected
    }
    if flag_mismatches:
        raise ConversionError(f"conversion receipt claim-boundary mismatch: {flag_mismatches}")
    if receipt.get("protocol") != PROTOCOL:
        raise ConversionError("conversion receipt protocol mismatch")
    if stat.S_IMODE(receipt_path.stat().st_mode) & 0o222:
        raise ConversionError("conversion receipt is writable")
    writable_entries = [
        str(path)
        for path in [output, *output.rglob("*")]
        if stat.S_IMODE(path.stat().st_mode) & 0o222
    ]
    if writable_entries:
        raise ConversionError(
            "converted output is not immutable; writable entries: " f"{writable_entries[:8]}"
        )

    recorded_output = receipt.get("converted_output")
    if not isinstance(recorded_output, dict):
        raise ConversionError("conversion receipt lacks converted_output evidence")
    if Path(str(recorded_output.get("path"))).resolve() != output:
        raise ConversionError("conversion receipt output path mismatch")
    config = _load_json_object(output / ADAPTER_CONFIG_FILENAME)
    if config.get("bias") != "all":
        raise ConversionError("converted adapter_config.bias is not 'all'")
    expected_lora = recorded_output.get("serialized_lora_keys")
    expected_bias = recorded_output.get("serialized_bias_keys")
    if not isinstance(expected_lora, list) or not isinstance(expected_bias, list) or not expected_bias:
        raise ConversionError("conversion receipt lacks a complete serialized tensor inventory")
    observed_keys = _safetensor_keys(output / ADAPTER_WEIGHTS_FILENAME)
    expected_keys = sorted([*expected_lora, *expected_bias])
    if observed_keys != expected_keys:
        missing = sorted(set(expected_keys) - set(observed_keys))
        unexpected = sorted(set(observed_keys) - set(expected_keys))
        label = "missing converted biases" if any(key in expected_bias for key in missing) else "key"
        raise ConversionError(
            f"converted adapter {label} mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    live_output_manifest = build_tree_manifest(
        output,
        excluded_names=frozenset({RECEIPT_FILENAME}),
    )
    if live_output_manifest != recorded_output.get("manifest_excluding_receipt"):
        raise ConversionError("converted output bytes differ from the conversion receipt")
    try:
        live_inspection = inspect_checkpoint_adapter(output)
    except CheckpointFidelityError as exc:
        raise ConversionError(f"checkpoint-fidelity adapter inspection failed: {exc}") from exc
    if live_inspection != recorded_output.get("checkpoint_fidelity_adapter_inspection"):
        raise ConversionError("checkpoint-fidelity inspection differs from the conversion receipt")

    recorded_base = receipt.get("base")
    recorded_legacy = receipt.get("legacy_source")
    if not isinstance(recorded_base, dict) or not isinstance(recorded_legacy, dict):
        raise ConversionError("conversion receipt lacks source identities")
    base = _regular_directory(Path(str(recorded_base.get("path"))), label="recorded base")
    legacy = _regular_directory(
        Path(str(recorded_legacy.get("path"))), label="recorded legacy adapter"
    )
    if base_model_path is not None:
        supplied_base = _regular_directory(base_model_path, label="supplied base")
        if supplied_base != base:
            raise ConversionError("supplied base does not match conversion receipt")
    if legacy_adapter_path is not None:
        supplied_legacy = _regular_directory(legacy_adapter_path, label="supplied legacy adapter")
        if supplied_legacy != legacy:
            raise ConversionError("supplied legacy adapter does not match conversion receipt")
    if build_tree_manifest(base) != recorded_base.get("manifest"):
        raise ConversionError("base model bytes differ from the conversion receipt")
    if build_tree_manifest(legacy) != recorded_legacy.get("manifest"):
        raise ConversionError("legacy adapter bytes differ from the conversion receipt")
    return receipt


def convert_legacy_saved_policy(
    *,
    base_model_path: Path,
    legacy_adapter_path: Path,
    output_dir: Path,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    """Create and verify a faithful, explicitly non-publication re-encoding."""

    raw_output = output_dir.expanduser().absolute()
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite conversion output: {raw_output}")
    audit = audit_legacy_inputs(base_model_path, legacy_adapter_path)
    base = Path(audit["base_path"])
    legacy = Path(audit["legacy_adapter_path"])
    base_manifest = build_tree_manifest(base)
    legacy_manifest = build_tree_manifest(legacy)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw_output.mkdir(mode=stat.S_IRWXU)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite conversion output: {raw_output}") from exc
    output = raw_output.resolve(strict=True)

    legacy_model = None
    converted_model = None
    success = False
    try:
        legacy_base = _load_base_model(base, device=device, dtype=dtype)
        base_biases = _base_qkv_bias_state(legacy_base)
        legacy_model = _load_adapter(legacy_base, legacy)
        legacy_state = _canonical_deployed_state(legacy_model)
        _assert_base_biases_unchanged(base_biases, legacy_state)
        legacy_lora_state = {
            key: tensor for key, tensor in legacy_state.items() if key.endswith(LORA_SUFFIXES)
        }
        if sorted(legacy_lora_state) != audit["legacy_serialized_lora_keys"]:
            missing = sorted(set(audit["legacy_serialized_lora_keys"]) - set(legacy_lora_state))
            unexpected = sorted(set(legacy_lora_state) - set(audit["legacy_serialized_lora_keys"]))
            raise ConversionError(
                "loaded legacy LoRA inventory differs from serialized source: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        _assert_serialized_state(
            legacy / ADAPTER_WEIGHTS_FILENAME,
            legacy_lora_state,
            require_biases=False,
        )
        legacy_probe = _fixed_probe(legacy_model, device=device)

        peft_config = legacy_model.peft_config.get(ADAPTER_NAME)
        if peft_config is None or getattr(peft_config, "bias", None) != "lora_only":
            raise ConversionError("loaded legacy PEFT config does not retain bias='lora_only'")
        peft_config.bias = "all"
        legacy_model.save_pretrained(
            str(output),
            safe_serialization=True,
            selected_adapters=[ADAPTER_NAME],
        )
        converted_config = _load_json_object(output / ADAPTER_CONFIG_FILENAME)
        if converted_config.get("bias") != "all":
            raise ConversionError("PEFT did not serialize the converted bias='all' policy")
        if _resolve_configured_base(converted_config.get("base_model_name_or_path")) != base:
            raise ConversionError("converted adapter no longer designates the exact legacy base")
        _assert_serialized_state(
            output / ADAPTER_WEIGHTS_FILENAME,
            legacy_state,
            require_biases=True,
        )

        del legacy_model, legacy_base
        legacy_model = None
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

        converted_base = _load_base_model(base, device=device, dtype=dtype)
        converted_model = _load_adapter(converted_base, output)
        converted_state = _canonical_deployed_state(converted_model)
        _assert_state_equal(legacy_state, converted_state)
        _assert_base_biases_unchanged(base_biases, converted_state)
        converted_probe = _fixed_probe(converted_model, device=device)
        _assert_probe_equal(legacy_probe, converted_probe)
        _assert_serialized_state(
            output / ADAPTER_WEIGHTS_FILENAME,
            converted_state,
            require_biases=True,
        )

        final_base_manifest = build_tree_manifest(base)
        final_legacy_manifest = build_tree_manifest(legacy)
        if final_base_manifest != base_manifest:
            raise ConversionError("base model changed during conversion")
        if final_legacy_manifest != legacy_manifest:
            raise ConversionError("legacy adapter changed during conversion")
        output_manifest = build_tree_manifest(
            output,
            excluded_names=frozenset({RECEIPT_FILENAME}),
        )
        serialized_bias_keys = sorted(
            key for key in converted_state if key.endswith(".base_layer.bias")
        )
        serialized_lora_keys = sorted(key for key in converted_state if key.endswith(LORA_SUFFIXES))
        receipt = _build_receipt(
            base=base,
            legacy=legacy,
            output=output,
            base_manifest=base_manifest,
            legacy_manifest=legacy_manifest,
            output_manifest=output_manifest,
            legacy_config=audit["legacy_config"],
            converted_config=converted_config,
            state_inventory=_state_inventory(converted_state),
            serialized_lora_keys=serialized_lora_keys,
            serialized_bias_keys=serialized_bias_keys,
            probe_inventory=_probe_inventory(converted_probe),
            device=device,
            dtype=dtype,
        )
        _write_receipt_create_once(output, receipt)
        _make_read_only(output)
        verified = verify_conversion_receipt(
            output,
            base_model_path=base,
            legacy_adapter_path=legacy,
        )
        success = True
        return verified
    except Exception:
        if not success and output.exists():
            for path in output.rglob("*"):
                try:
                    if path.is_dir():
                        path.chmod(stat.S_IRWXU)
                    else:
                        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass
            output.chmod(stat.S_IRWXU)
            shutil.rmtree(output)
        raise
    finally:
        del legacy_model, converted_model
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--legacy-adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing conversion receipt and all content-bound inputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            payload = verify_conversion_receipt(
                args.output_dir,
                base_model_path=args.base_model_path,
                legacy_adapter_path=args.legacy_adapter_path,
            )
        else:
            if args.base_model_path is None or args.legacy_adapter_path is None:
                raise ConversionError(
                    "conversion requires --base-model-path and --legacy-adapter-path"
                )
            payload = convert_legacy_saved_policy(
                base_model_path=args.base_model_path,
                legacy_adapter_path=args.legacy_adapter_path,
                output_dir=args.output_dir,
                device=args.device,
                dtype=args.dtype,
            )
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except (ConversionError, FileExistsError) as exc:
        print(f"[LegacySavedPolicyConversionError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
