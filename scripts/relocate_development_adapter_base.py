#!/usr/bin/env python3
"""Relocate a development-only PEFT adapter to an immutable identical base.

This is an identity transformation of the *saved* policy, not recovery of the
historical SFT policy.  It rewrites only ``base_model_name_or_path`` and proves
that the old and relocated policies have identical full state, logits, and
token log-probabilities.  The create-once output is permanently ineligible for
publication or SOTA claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
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
    inspect_checkpoint_adapter,
)

SCHEMA_VERSION = 1
PROTOCOL = "clasp-development-adapter-base-relocation-v1"
STATUS = "development_saved_policy_base_relocation_verified"
VERIFIER_PROTOCOL = "clasp-development-adapter-base-relocation-verifier-v1"
RECEIPT_FILENAME = "adapter_base_relocation_receipt.json"
UPSTREAM_RECEIPT_FILENAME = "legacy_saved_policy_conversion_receipt.json"
UPSTREAM_PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
FIXED_PROBES = (
    (7, 11, 13, 17, 19, 23),
    (23, 19, 17, 13, 11, 7),
    (2, 3, 5, 7, 11, 13),
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_CLAIM_BOUNDARY = (
    "Config-only relocation of a faithfully re-encoded legacy saved policy; "
    "omitted historical biases remain unrecoverable and this artifact is never "
    "eligible for publication or SOTA claims."
)
_TOP_LEVEL_KEYS = frozenset(
    {
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
)
_VERIFIER_KEYS = frozenset({"protocol", "implementation", "implementation_sha256"})
_UPSTREAM_KEYS = frozenset(
    {
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
)
_FROZEN_BASE_KEYS = frozenset({"path", "manifest", "config_sha256"})
_RELOCATED_OUTPUT_KEYS = frozenset(
    {
        "path",
        "manifest_excluding_receipt",
        "adapter_config_sha256",
        "adapter_model_sha256",
        "configured_base_path",
        "serialization_bias_policy",
        "checkpoint_fidelity_adapter_inspection",
    }
)
_TRANSFORMATION_KEYS = frozenset(
    {
        "method",
        "changed_config_fields",
        "source_config",
        "relocated_config",
        "config_only_delta",
        "unchanged_file_paths",
        "non_config_files_bitwise_identical",
    }
)
_VERIFICATION_KEYS = frozenset(
    {
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
)
_SOFTWARE_KEYS = frozenset({"python", "torch", "transformers", "peft", "safetensors"})
_STATE_KEYS = frozenset({"tensor_count", "parameter_count", "tensor_inventory_sha256", "tensors"})
_TENSOR_KEYS = frozenset({"key", "shape", "dtype", "numel", "value_sha256"})
_FUNCTIONAL_KEYS = frozenset(
    {
        "method",
        "input_ids",
        "input_ids_sha256",
        "logits",
        "log_probabilities",
        "argmax",
        "bitwise_equal",
        "max_absolute_error",
    }
)


class RelocationError(ValueError):
    """The development adapter relocation contract was not satisfied."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RelocationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _regular_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser().absolute()
    try:
        metadata = raw.lstat()
    except OSError as exc:
        raise RelocationError(f"cannot stat {label} {raw}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RelocationError(f"{label} is not a regular directory: {raw}")
    return raw.resolve(strict=True)


def _regular_file_bytes(path: Path, *, read_only: bool = False) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RelocationError(f"cannot stat required file {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RelocationError(f"expected a regular file: {path}")
    if read_only and stat.S_IMODE(before.st_mode) & 0o222:
        raise RelocationError(f"required file must be read-only: {path}")
    try:
        value = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise RelocationError(f"cannot read {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in identity):
        raise RelocationError(f"file changed while read: {path}")
    return value


def _load_json(path: Path, *, read_only: bool = False) -> dict[str, Any]:
    raw = _regular_file_bytes(path, read_only=read_only)
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelocationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RelocationError(f"expected JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_regular_file_bytes(path))


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _validate_tensor_record(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _TENSOR_KEYS:
        raise RelocationError(f"{label} tensor schema mismatch")
    shape = record.get("shape")
    numel = record.get("numel")
    if (
        not isinstance(record.get("key"), str)
        or not record.get("key")
        or not isinstance(shape, list)
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        )
        or not isinstance(record.get("dtype"), str)
        or not record.get("dtype")
        or isinstance(numel, bool)
        or not isinstance(numel, int)
        or numel < 0
        or not _valid_sha256(record.get("value_sha256"))
    ):
        raise RelocationError(f"{label} tensor record is invalid")
    product = 1
    for dimension in shape:
        product *= dimension
    if product != numel:
        raise RelocationError(f"{label} tensor numel/shape mismatch")


def _validate_full_state(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _STATE_KEYS:
        raise RelocationError("full-model state schema mismatch")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RelocationError("full-model state inventory is empty")
    previous = ""
    for record in tensors:
        _validate_tensor_record(record, label="full-model state")
        key = str(record["key"])
        if key <= previous:
            raise RelocationError("full-model state tensor keys are not unique and sorted")
        previous = key
    if (
        value.get("tensor_count") != len(tensors)
        or value.get("parameter_count") != sum(int(record["numel"]) for record in tensors)
        or value.get("tensor_inventory_sha256") != _sha256_json(tensors)
    ):
        raise RelocationError("full-model state aggregate mismatch")


def _validate_functional_evidence(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _FUNCTIONAL_KEYS:
        raise RelocationError("functional-equivalence schema mismatch")
    expected_inputs = [list(row) for row in FIXED_PROBES]
    if (
        value.get("method") != "three_fixed_probes_bitwise_full_distribution_v1"
        or value.get("input_ids") != expected_inputs
        or value.get("input_ids_sha256") != _sha256_json(expected_inputs)
        or value.get("bitwise_equal") is not True
        or value.get("max_absolute_error") != 0.0
    ):
        raise RelocationError("functional-equivalence invariant mismatch")
    for field in ("logits", "log_probabilities", "argmax"):
        _validate_tensor_record(value.get(field), label=f"functional {field}")


def tree_manifest(
    root: Path,
    *,
    label: str,
    excluded_names: frozenset[str] = frozenset(),
    require_read_only: bool,
) -> dict[str, Any]:
    """Return a path-independent, symlink-free content manifest."""

    root = _regular_directory(root, label=label)
    files: list[dict[str, Any]] = []
    writable: list[str] = []
    entries = [root, *sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix())]
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RelocationError(f"{label} contains a symbolic link: {path}")
        if require_read_only and stat.S_IMODE(metadata.st_mode) & 0o222:
            writable.append(str(path))
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RelocationError(f"{label} contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        if Path(relative).name in excluded_names:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": int(metadata.st_size),
                "sha256": _sha256_file(path),
            }
        )
    if writable:
        raise RelocationError(f"{label} is writable: {writable[:8]}")
    if not files:
        raise RelocationError(f"{label} contains no auditable files")
    return {
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "tree_sha256": _sha256_json(files),
        "files": files,
    }


def _normalize_legacy_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"file_count", "files", "tree_sha256"}:
        raise RelocationError("upstream recorded base manifest schema mismatch")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise RelocationError("upstream recorded base manifest is empty")
    normalized = {
        "file_count": value.get("file_count"),
        "size_bytes": sum(
            int(record.get("size_bytes", -1)) for record in files if isinstance(record, Mapping)
        ),
        "tree_sha256": value.get("tree_sha256"),
        "files": files,
    }
    if (
        normalized["file_count"] != len(files)
        or normalized["tree_sha256"] != _sha256_json(files)
        or normalized["size_bytes"] < 0
    ):
        raise RelocationError("upstream recorded base manifest aggregate mismatch")
    return normalized


def _load_upstream_verifier() -> Any:
    path = Path(__file__).resolve().with_name("convert_legacy_saved_policy.py")
    spec = importlib.util.spec_from_file_location("_clasp_legacy_converter", path)
    if spec is None or spec.loader is None:
        raise RelocationError(f"cannot load upstream verifier implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_upstream(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = source / UPSTREAM_RECEIPT_FILENAME
    raw = _regular_file_bytes(receipt_path, read_only=True)
    receipt = _load_json(receipt_path, read_only=True)
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed_hash != _sha256_json(unhashed):
        raise RelocationError("upstream conversion receipt self-hash mismatch")
    required = {
        "protocol": UPSTREAM_PROTOCOL,
        "status": "converted_saved_policy_verified",
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
    }
    mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise RelocationError(f"upstream conversion claim mismatch: {mismatches}")
    try:
        verified = _load_upstream_verifier().verify_conversion_receipt(source)
    except Exception as exc:
        raise RelocationError(f"upstream conversion verification failed: {exc}") from exc
    if verified != receipt:
        raise RelocationError("upstream verifier result differs from receipt")
    return receipt, {
        "path": str(receipt_path),
        "protocol": receipt["protocol"],
        "receipt_sha256": claimed_hash,
        "file_sha256": _sha256_bytes(raw),
    }


def _configured_base(config: Mapping[str, Any]) -> Path:
    value = config.get("base_model_name_or_path")
    if not isinstance(value, str) or not value:
        raise RelocationError("adapter config lacks base_model_name_or_path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RelocationError("adapter base_model_name_or_path must be absolute")
    return _regular_directory(path, label="configured base model")


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().resolve_conj().resolve_neg().to("cpu").contiguous()
    if value.layout != torch.strided or value.device.type == "meta":
        raise RelocationError("cannot hash a non-materialized tensor")
    return value.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_record(key: str, tensor: Any) -> dict[str, Any]:
    return {
        "key": key,
        "shape": [int(dimension) for dimension in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "value_sha256": _sha256_bytes(_tensor_bytes(tensor)),
    }


def _state_inventory(model: Any) -> dict[str, Any]:
    tensors = [_tensor_record(key, value) for key, value in sorted(model.state_dict().items())]
    return {
        "tensor_count": len(tensors),
        "parameter_count": sum(item["numel"] for item in tensors),
        "tensor_inventory_sha256": _sha256_json(tensors),
        "tensors": tensors,
    }


def _probe(model: Any, *, device: str) -> dict[str, Any]:
    import torch

    vocab_size = getattr(model.config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= max(max(row) for row in FIXED_PROBES):
        raise RelocationError(f"model vocabulary is too small for fixed probes: {vocab_size!r}")
    input_ids = torch.tensor(FIXED_PROBES, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            .logits.detach()
            .cpu()
        )
        logps = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        argmax = logits.argmax(dim=-1)
    return {
        "input_ids": [list(row) for row in FIXED_PROBES],
        "logits": logits,
        "logps": logps,
        "argmax": argmax,
    }


def _probe_evidence(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": "three_fixed_probes_bitwise_full_distribution_v1",
        "input_ids": probe["input_ids"],
        "input_ids_sha256": _sha256_json(probe["input_ids"]),
        "logits": _tensor_record("logits", probe["logits"]),
        "log_probabilities": _tensor_record("log_probabilities", probe["logps"]),
        "argmax": _tensor_record("argmax", probe["argmax"]),
        "bitwise_equal": True,
        "max_absolute_error": 0.0,
    }


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {
            "auto": "auto",
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as exc:
        raise RelocationError(f"unsupported dtype: {name}") from exc


def _load_policy(adapter: Path, *, device: str, dtype: str) -> Any:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RelocationError(f"relocation dependencies unavailable: {exc}") from exc
    if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
        raise RelocationError(f"requested device is unavailable: {device}")
    config = _load_json(adapter / ADAPTER_CONFIG_FILENAME)
    if config.get("bias") != "all":
        raise RelocationError("relocation requires adapter bias='all'")
    base = _configured_base(config)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base),
            local_files_only=True,
            trust_remote_code=False,
            dtype=_torch_dtype(dtype),
            attn_implementation="eager",
        ).to(device)
        return PeftModel.from_pretrained(
            model,
            str(adapter),
            adapter_name="default",
            is_trainable=False,
            autocast_adapter_dtype=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise RelocationError(f"cannot independently load policy {adapter}: {exc}") from exc


def _assert_probe_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    import torch

    for field in ("logits", "logps", "argmax"):
        if not torch.equal(left[field], right[field]):
            if field == "logits":
                delta = (left[field].float() - right[field].float()).abs()
                detail = f"; max_absolute_error={float(delta.max().item())!r}"
            else:
                detail = ""
            raise RelocationError(f"relocated policy {field} are not bitwise equal{detail}")


def _copy_payload(
    source: Path, output: Path, frozen_base: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source_config = _load_json(source / ADAPTER_CONFIG_FILENAME, read_only=True)
    relocated_config = dict(source_config)
    relocated_config["base_model_name_or_path"] = str(frozen_base)
    if source_config == relocated_config:
        raise RelocationError("source adapter already designates the requested frozen base")
    source_without_base = dict(source_config)
    relocated_without_base = dict(relocated_config)
    source_without_base.pop("base_model_name_or_path", None)
    relocated_without_base.pop("base_model_name_or_path", None)
    if source_without_base != relocated_without_base:
        raise RelocationError("relocation config delta is not base_model_name_or_path-only")
    unchanged: list[str] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        if relative.as_posix() in {ADAPTER_CONFIG_FILENAME, UPSTREAM_RECEIPT_FILENAME}:
            continue
        if path.is_symlink():
            raise RelocationError(f"source adapter contains a symlink: {path}")
        target = output / relative
        if path.is_dir():
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            unchanged.append(relative.as_posix())
        else:
            raise RelocationError(f"source adapter contains non-regular entry: {path}")
    (output / ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(relocated_config, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source_config, relocated_config, unchanged


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RelocationError(f"refusing to freeze symlink: {path}")
        path.chmod(0o550 if path.is_dir() else 0o440)
    root.chmod(0o550)


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _write_receipt(output: Path, receipt: Mapping[str, Any]) -> None:
    path = output / RECEIPT_FILENAME
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_receipt_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


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


def _implementation_sha256() -> str:
    return _sha256_file(Path(__file__).resolve(strict=True))


def relocate_development_adapter_base(
    *,
    source_adapter_path: Path,
    frozen_base_model_path: Path,
    output_dir: Path,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    """Create and prove a config-only relocation to a byte-identical base."""

    source = _regular_directory(source_adapter_path, label="source adapter")
    frozen_base = _regular_directory(frozen_base_model_path, label="frozen base")
    raw_output = output_dir.expanduser().absolute()
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite relocation output: {raw_output}")
    for protected, label in ((source, "source adapter"), (frozen_base, "frozen base")):
        try:
            raw_output.resolve(strict=False).relative_to(protected)
        except ValueError:
            pass
        else:
            raise RelocationError(f"output must not be inside {label}")

    source_manifest = tree_manifest(source, label="source adapter", require_read_only=True)
    frozen_manifest = tree_manifest(frozen_base, label="frozen base", require_read_only=True)
    upstream, upstream_identity = _verify_upstream(source)
    upstream_base = upstream.get("base")
    if not isinstance(upstream_base, Mapping):
        raise RelocationError("upstream receipt lacks base evidence")
    recorded_base = _normalize_legacy_manifest(upstream_base.get("manifest"))
    original_base = _regular_directory(Path(str(upstream_base.get("path"))), label="upstream base")
    live_original_base = tree_manifest(
        original_base, label="upstream base", require_read_only=False
    )
    if recorded_base != live_original_base or live_original_base != frozen_manifest:
        raise RelocationError(
            "original, recorded, and frozen base manifests are not byte-identical"
        )
    source_config = _load_json(source / ADAPTER_CONFIG_FILENAME, read_only=True)
    if _configured_base(source_config) != original_base:
        raise RelocationError("source config base differs from upstream receipt")
    if source_config.get("bias") != "all":
        raise RelocationError("source adapter must use bias='all'")

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(mode=0o700)
    output = raw_output.resolve(strict=True)
    source_model = relocated_model = None
    success = False
    try:
        source_config, relocated_config, unchanged_files = _copy_payload(
            source, output, frozen_base
        )
        for relative in unchanged_files:
            if _sha256_file(source / relative) != _sha256_file(output / relative):
                raise RelocationError(f"non-config payload changed: {relative}")

        source_model = _load_policy(source, device=device, dtype=dtype)
        source_state = _state_inventory(source_model)
        source_probe = _probe(source_model, device=device)
        del source_model
        source_model = None
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

        relocated_model = _load_policy(output, device=device, dtype=dtype)
        relocated_state = _state_inventory(relocated_model)
        relocated_probe = _probe(relocated_model, device=device)
        if source_state != relocated_state:
            raise RelocationError("relocated full model state is not bitwise identical")
        _assert_probe_equal(source_probe, relocated_probe)

        if tree_manifest(source, label="source adapter", require_read_only=True) != source_manifest:
            raise RelocationError("source adapter changed during relocation")
        if (
            tree_manifest(original_base, label="upstream base", require_read_only=False)
            != live_original_base
        ):
            raise RelocationError("upstream base changed during relocation")
        if (
            tree_manifest(frozen_base, label="frozen base", require_read_only=True)
            != frozen_manifest
        ):
            raise RelocationError("frozen base changed during relocation")
        output_manifest = tree_manifest(
            output,
            label="relocated output staging",
            excluded_names=frozenset({RECEIPT_FILENAME}),
            require_read_only=False,
        )
        try:
            inspection = inspect_checkpoint_adapter(output)
        except CheckpointFidelityError as exc:
            raise RelocationError(f"relocated adapter inspection failed: {exc}") from exc
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": STATUS,
            "claim_scope": "development_only",
            "development_only": True,
            "relocation_only": True,
            "recovers_historical_terminal_biases": False,
            "converted_saved_policy_fidelity": True,
            "terminal_sft_policy_fidelity": False,
            "publication_eligible": False,
            "sota_eligible": False,
            "publication_sota_eligible": False,
            "claim_boundary": _CLAIM_BOUNDARY,
            "verifier": {
                "protocol": VERIFIER_PROTOCOL,
                "implementation": "scripts.relocate_development_adapter_base",
                "implementation_sha256": _implementation_sha256(),
            },
            "upstream_conversion": {
                "source_adapter_path": str(source),
                "source_adapter_manifest": source_manifest,
                "authoritative_receipt_path": upstream_identity["path"],
                "authoritative_receipt_protocol": upstream_identity["protocol"],
                "authoritative_receipt_sha256": upstream_identity["receipt_sha256"],
                "authoritative_receipt_file_sha256": upstream_identity["file_sha256"],
                "adapter_config_sha256": _sha256_file(source / ADAPTER_CONFIG_FILENAME),
                "adapter_model_sha256": _sha256_file(source / ADAPTER_WEIGHTS_FILENAME),
                "configured_base_path": str(original_base),
                "recorded_base_manifest": recorded_base,
                "live_base_manifest": live_original_base,
            },
            "frozen_base": {
                "path": str(frozen_base),
                "manifest": frozen_manifest,
                "config_sha256": _sha256_file(frozen_base / "config.json"),
            },
            "relocated_output": {
                "path": str(output),
                "manifest_excluding_receipt": output_manifest,
                "adapter_config_sha256": _sha256_file(output / ADAPTER_CONFIG_FILENAME),
                "adapter_model_sha256": _sha256_file(output / ADAPTER_WEIGHTS_FILENAME),
                "configured_base_path": str(frozen_base),
                "serialization_bias_policy": relocated_config.get("bias"),
                "checkpoint_fidelity_adapter_inspection": inspection,
            },
            "transformation": {
                "method": "config_only_absolute_base_path_relocation_v1",
                "changed_config_fields": ["base_model_name_or_path"],
                "source_config": source_config,
                "relocated_config": relocated_config,
                "config_only_delta": True,
                "unchanged_file_paths": unchanged_files,
                "non_config_files_bitwise_identical": True,
            },
            "verification": {
                "method": "independent_source_and_relocated_policy_reload_v1",
                "upstream_receipt_verified": True,
                "path_independent_base_manifests_exact": True,
                "source_and_output_stable": True,
                "independent_model_loads": True,
                "full_model_state_exact": True,
                "full_model_state": source_state,
                "functional_equivalence_exact": True,
                "functional_equivalence": _probe_evidence(relocated_probe),
                "device": device,
                "requested_dtype": dtype,
            },
            "software": _software_versions(),
        }
        payload["receipt_sha256"] = _sha256_json(payload)
        _write_receipt(output, payload)
        _make_read_only(output)
        verified = verify_relocation_receipt(output, rehash_upstream=True)
        success = True
        return verified
    finally:
        del source_model, relocated_model
        gc.collect()
        if not success and output.exists():
            for path in [output, *output.rglob("*")]:
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(output)


def verify_relocation_receipt(
    output_dir: Path,
    *,
    source_adapter_path: Path | None = None,
    frozen_base_model_path: Path | None = None,
    rehash_upstream: bool = True,
) -> dict[str, Any]:
    """Verify canonical receipt bytes, lineage, claims, and all bound live bytes."""

    output = _regular_directory(output_dir, label="relocated output")
    receipt_path = output / RECEIPT_FILENAME
    raw = _regular_file_bytes(receipt_path, read_only=True)
    receipt = _load_json(receipt_path, read_only=True)
    if raw != _receipt_bytes(receipt):
        raise RelocationError("relocation receipt is not canonically serialized")
    claimed = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed != _sha256_json(unhashed):
        raise RelocationError("relocation receipt self-hash mismatch")
    if set(receipt) != _TOP_LEVEL_KEYS:
        raise RelocationError("relocation receipt top-level schema mismatch")
    required = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
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
    mismatches = {
        key: {"actual": receipt.get(key), "expected": value}
        for key, value in required.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RelocationError(f"relocation receipt claim mismatch: {mismatches}")
    if receipt.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise RelocationError("relocation claim-boundary text mismatch")
    verifier = receipt.get("verifier")
    if (
        not isinstance(verifier, Mapping)
        or set(verifier) != _VERIFIER_KEYS
        or verifier
        != {
            "protocol": VERIFIER_PROTOCOL,
            "implementation": "scripts.relocate_development_adapter_base",
            "implementation_sha256": _implementation_sha256(),
        }
    ):
        raise RelocationError("relocation verifier identity mismatch")
    upstream = receipt.get("upstream_conversion")
    frozen = receipt.get("frozen_base")
    relocated = receipt.get("relocated_output")
    transformation = receipt.get("transformation")
    verification = receipt.get("verification")
    if not all(
        isinstance(value, Mapping)
        for value in (upstream, frozen, relocated, transformation, verification)
    ):
        raise RelocationError("relocation receipt evidence object missing")
    if set(upstream) != _UPSTREAM_KEYS:
        raise RelocationError("upstream-conversion schema mismatch")
    if set(frozen) != _FROZEN_BASE_KEYS:
        raise RelocationError("frozen-base schema mismatch")
    if set(relocated) != _RELOCATED_OUTPUT_KEYS:
        raise RelocationError("relocated-output schema mismatch")
    if set(transformation) != _TRANSFORMATION_KEYS:
        raise RelocationError("relocation-transformation schema mismatch")
    if set(verification) != _VERIFICATION_KEYS:
        raise RelocationError("relocation-verification schema mismatch")
    software = receipt.get("software")
    if (
        not isinstance(software, Mapping)
        or set(software) != _SOFTWARE_KEYS
        or any(not isinstance(value, str) or not value for value in software.values())
    ):
        raise RelocationError("relocation software schema mismatch")
    required_booleans = (
        "upstream_receipt_verified",
        "path_independent_base_manifests_exact",
        "source_and_output_stable",
        "independent_model_loads",
        "full_model_state_exact",
        "functional_equivalence_exact",
    )
    if any(verification.get(key) is not True for key in required_booleans):
        raise RelocationError("relocation verification invariant is false")
    if verification.get("method") != "independent_source_and_relocated_policy_reload_v1":
        raise RelocationError("relocation verification method mismatch")
    if verification.get("requested_dtype") not in {
        "auto",
        "float32",
        "bfloat16",
        "float16",
    } or not isinstance(verification.get("device"), str):
        raise RelocationError("relocation verification runtime contract mismatch")
    _validate_full_state(verification.get("full_model_state"))
    _validate_functional_evidence(verification.get("functional_equivalence"))
    if Path(str(relocated.get("path"))).resolve(strict=True) != output:
        raise RelocationError("relocation output path mismatch")
    live_output_manifest = tree_manifest(
        output,
        label="relocated output",
        require_read_only=True,
        excluded_names=frozenset({RECEIPT_FILENAME}),
    )
    if live_output_manifest != relocated.get("manifest_excluding_receipt"):
        raise RelocationError("relocated output bytes differ from receipt")
    config = _load_json(output / ADAPTER_CONFIG_FILENAME, read_only=True)
    if config.get("bias") != "all":
        raise RelocationError("relocated adapter bias is not 'all'")
    frozen_path = _regular_directory(Path(str(frozen.get("path"))), label="frozen base")
    if (
        frozen_base_model_path is not None
        and _regular_directory(frozen_base_model_path, label="supplied frozen base") != frozen_path
    ):
        raise RelocationError("supplied frozen base path mismatch")
    if _configured_base(config) != frozen_path or relocated.get("configured_base_path") != str(
        frozen_path
    ):
        raise RelocationError("relocated config does not designate frozen base")
    live_frozen = tree_manifest(frozen_path, label="frozen base", require_read_only=True)
    if live_frozen != frozen.get("manifest"):
        raise RelocationError("frozen base bytes differ from receipt")
    if _sha256_file(output / ADAPTER_CONFIG_FILENAME) != relocated.get(
        "adapter_config_sha256"
    ) or _sha256_file(output / ADAPTER_WEIGHTS_FILENAME) != relocated.get("adapter_model_sha256"):
        raise RelocationError("relocated adapter artifact hash mismatch")
    try:
        if inspect_checkpoint_adapter(output) != relocated.get(
            "checkpoint_fidelity_adapter_inspection"
        ):
            raise RelocationError("relocated adapter inspection mismatch")
    except CheckpointFidelityError as exc:
        raise RelocationError(f"relocated adapter inspection failed: {exc}") from exc
    source = _regular_directory(
        Path(str(upstream.get("source_adapter_path"))), label="source adapter"
    )
    if (
        source_adapter_path is not None
        and _regular_directory(source_adapter_path, label="supplied source adapter") != source
    ):
        raise RelocationError("supplied source adapter path mismatch")
    source_manifest = tree_manifest(source, label="source adapter", require_read_only=True)
    if source_manifest != upstream.get("source_adapter_manifest"):
        raise RelocationError("source adapter bytes differ from receipt")
    for field in (
        "authoritative_receipt_sha256",
        "authoritative_receipt_file_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
    ):
        if not _valid_sha256(upstream.get(field)):
            raise RelocationError(f"upstream conversion has invalid {field}")
    if not _valid_sha256(frozen.get("config_sha256")):
        raise RelocationError("frozen base has invalid config_sha256")
    source_config_live = _load_json(source / ADAPTER_CONFIG_FILENAME, read_only=True)
    if (
        _sha256_file(source / ADAPTER_CONFIG_FILENAME) != upstream.get("adapter_config_sha256")
        or _sha256_file(source / ADAPTER_WEIGHTS_FILENAME) != upstream.get("adapter_model_sha256")
        or source_config_live.get("base_model_name_or_path") != upstream.get("configured_base_path")
    ):
        raise RelocationError("live source adapter identity mismatch")
    if rehash_upstream:
        verified_upstream, identity = _verify_upstream(source)
        if (
            identity["path"] != upstream.get("authoritative_receipt_path")
            or identity["protocol"] != upstream.get("authoritative_receipt_protocol")
            or identity["receipt_sha256"] != upstream.get("authoritative_receipt_sha256")
            or identity["file_sha256"] != upstream.get("authoritative_receipt_file_sha256")
        ):
            raise RelocationError("live upstream receipt identity mismatch")
        recorded = _normalize_legacy_manifest(verified_upstream["base"]["manifest"])
        original = _regular_directory(
            Path(str(upstream.get("configured_base_path"))), label="upstream base"
        )
        live = tree_manifest(original, label="upstream base", require_read_only=False)
        if (
            recorded != upstream.get("recorded_base_manifest")
            or live != upstream.get("live_base_manifest")
            or recorded != live_frozen
        ):
            raise RelocationError("upstream/frozen base lineage mismatch")
    source_config = transformation.get("source_config")
    relocated_config = transformation.get("relocated_config")
    if not isinstance(source_config, Mapping) or not isinstance(relocated_config, Mapping):
        raise RelocationError("relocation config evidence missing")
    if (
        transformation.get("method") != "config_only_absolute_base_path_relocation_v1"
        or transformation.get("changed_config_fields") != ["base_model_name_or_path"]
        or transformation.get("config_only_delta") is not True
        or transformation.get("non_config_files_bitwise_identical") is not True
    ):
        raise RelocationError("relocation transformation invariant mismatch")
    if dict(source_config) != source_config_live:
        raise RelocationError("recorded source config differs from live source")
    expected = dict(source_config)
    expected["base_model_name_or_path"] = str(frozen_path)
    if dict(relocated_config) != expected or config != expected:
        raise RelocationError("adapter config delta is not exactly the base path")
    unchanged = transformation.get("unchanged_file_paths")
    if not isinstance(unchanged, list) or any(not isinstance(item, str) for item in unchanged):
        raise RelocationError("unchanged payload inventory is invalid")
    expected_unchanged = sorted(
        record["path"]
        for record in source_manifest["files"]
        if record["path"] not in {ADAPTER_CONFIG_FILENAME, UPSTREAM_RECEIPT_FILENAME}
    )
    expected_output_paths = sorted([ADAPTER_CONFIG_FILENAME, *expected_unchanged])
    observed_output_paths = [record["path"] for record in live_output_manifest["files"]]
    if unchanged != expected_unchanged or observed_output_paths != expected_output_paths:
        raise RelocationError(
            "unchanged payload inventory is incomplete or contains unexpected entries"
        )
    for relative in unchanged:
        if _sha256_file(source / relative) != _sha256_file(output / relative):
            raise RelocationError(f"unchanged payload hash mismatch: {relative}")
    if frozen.get("config_sha256") != _sha256_file(frozen_path / "config.json"):
        raise RelocationError("frozen base config hash mismatch")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter-path", type=Path)
    parser.add_argument("--frozen-base-model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-upstream-rehash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            receipt = verify_relocation_receipt(
                args.output_dir,
                source_adapter_path=args.source_adapter_path,
                frozen_base_model_path=args.frozen_base_model_path,
                rehash_upstream=not args.skip_upstream_rehash,
            )
        else:
            if args.source_adapter_path is None or args.frozen_base_model_path is None:
                raise RelocationError("relocation requires source adapter and frozen base paths")
            receipt = relocate_development_adapter_base(
                source_adapter_path=args.source_adapter_path,
                frozen_base_model_path=args.frozen_base_model_path,
                output_dir=args.output_dir,
                device=args.device,
                dtype=args.dtype,
            )
        print(json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    except (RelocationError, FileExistsError) as exc:
        print(f"[DevelopmentAdapterBaseRelocationError] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
