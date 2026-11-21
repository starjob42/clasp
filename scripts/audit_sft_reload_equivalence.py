#!/usr/bin/env python3
"""Certify an independent, exact reload of a publication SFT adapter.

``checkpoint_fidelity.json`` proves that PEFT serialized every trainable tensor
that existed immediately before ``save_pretrained``.  This audit closes the
other half of the round trip: it independently reloads the saved adapter twice,
requires the reloaded trainable closure to equal that pre-save closure exactly,
and requires bitwise equality of the two complete runtime states and fixed
probe distributions.

The resulting receipt is create-once, read-only, and intentionally verified
without importing Transformers or PEFT at later publication gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clasp.models.checkpoint_fidelity import (
    CheckpointFidelityError,
    audit_trainable_state_closure,
    inspect_checkpoint_adapter,
    verify_checkpoint_fidelity,
)
from clasp.models.exact_peft_kl_reference import FIXED_PROBES, audit_exact_runtime_pair
from clasp.models.publication_fidelity import (
    publication_fidelity_identity,
    validate_publication_fidelity_identity,
)

SCHEMA_VERSION = 2
PROTOCOL = "clasp-sft-post-save-reload-equivalence-v2"
STATUS = "exact_post_save_reload_verified"
RECEIPT_FILENAME = "sft_reload_equivalence.json"
RUNTIME_DTYPE = "bfloat16"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_KEYS = {"key", "shape", "dtype", "numel", "value_sha256"}


class SFTReloadEquivalenceError(ValueError):
    """Raised when the saved SFT cannot be proven reload-equivalent."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_file():
        raise SFTReloadEquivalenceError(f"required artifact is not a regular file: {path}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise SFTReloadEquivalenceError(f"required artifact changed while hashing: {path}")
    return digest.hexdigest()


def _module_file_sha256(module: Any) -> str:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        raise SFTReloadEquivalenceError(f"cannot locate verifier module {module!r}")
    return _sha256_file(Path(raw).resolve(strict=True))


def _canonical_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise SFTReloadEquivalenceError(f"{label} is not a regular directory: {raw}")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise SFTReloadEquivalenceError(
            f"{label} must be supplied canonically: supplied={raw}, resolved={resolved}"
        )
    return resolved


def _validate_tensor_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _TENSOR_KEYS:
        raise SFTReloadEquivalenceError(f"{label} has an invalid tensor-record schema")
    shape = value.get("shape")
    if (
        not isinstance(value.get("key"), str)
        or not value["key"]
        or not isinstance(shape, list)
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or not isinstance(value.get("dtype"), str)
        or not value["dtype"]
        or type(value.get("numel")) is not int
        or value["numel"] < 0
        or value["numel"] != math.prod(shape)
        or not isinstance(value.get("value_sha256"), str)
        or _SHA256_RE.fullmatch(value["value_sha256"]) is None
    ):
        raise SFTReloadEquivalenceError(f"{label} has invalid tensor-record values")
    return value


def _validate_runtime_pair(value: object) -> dict[str, Any]:
    """Validate the exact-pair evidence without importing a model runtime."""

    expected_keys = {
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
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SFTReloadEquivalenceError("reload runtime pair has an invalid schema")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise SFTReloadEquivalenceError("reload runtime pair has no tensor inventory")
    records = [
        _validate_tensor_record(record, label=f"runtime_pair.tensors[{index}]")
        for index, record in enumerate(tensors)
    ]
    keys = [record["key"] for record in records]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SFTReloadEquivalenceError("reload runtime tensor keys are not sorted and unique")
    if (
        value.get("method") != "independent_same_base_same_adapter_exact_runtime_pair_v2"
        or not isinstance(value.get("model_class"), str)
        or not value["model_class"]
        or not isinstance(value.get("active_adapter"), str)
        or not value["active_adapter"]
        or _SHA256_RE.fullmatch(str(value.get("model_config_sha256", ""))) is None
        or value.get("model_objects_distinct") is not True
        or value.get("independent_storage") is not True
        or value.get("reference_trainable_tensor_count") != 0
        or type(value.get("policy_trainable_tensor_count")) is not int
        or value["policy_trainable_tensor_count"] <= 0
        or value.get("tensor_count") != len(records)
        or value.get("parameter_count") != sum(record["numel"] for record in records)
        or value.get("tensor_inventory_sha256") != _sha256_json(records)
        or value.get("state_bitwise_equal") is not True
    ):
        raise SFTReloadEquivalenceError("reload runtime pair does not prove exact state equality")

    prepared = value.get("prepared_runtime")
    prepared_keys = {
        "trainer_ref_model_identity",
        "reference_unwrapped_identity",
        "policy_unwrapped_class",
        "reference_prepared_class",
        "reference_unwrapped_class",
        "device_types",
    }
    if (
        not isinstance(prepared, dict)
        or set(prepared) != prepared_keys
        or prepared.get("trainer_ref_model_identity") is not True
        or prepared.get("reference_unwrapped_identity") is not True
        or prepared.get("policy_unwrapped_class") != value["model_class"]
        or prepared.get("reference_prepared_class") != value["model_class"]
        or prepared.get("reference_unwrapped_class") != value["model_class"]
        or not isinstance(prepared.get("device_types"), list)
        or prepared["device_types"] != ["cuda"]
    ):
        raise SFTReloadEquivalenceError("reload runtime pair lacks independent prepared evidence")

    functional = value.get("functional_equivalence")
    functional_keys = {
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
    if not isinstance(functional, dict) or set(functional) != functional_keys:
        raise SFTReloadEquivalenceError("reload functional probe has an invalid schema")
    expected_probes = [list(row) for row in FIXED_PROBES]
    functional_tensors = {
        key: _validate_tensor_record(
            functional.get(key),
            label=f"functional_equivalence.{key}",
        )
        for key in ("logits", "log_probabilities", "argmax")
    }
    logits = functional_tensors["logits"]
    log_probabilities = functional_tensors["log_probabilities"]
    argmax = functional_tensors["argmax"]
    logits_shape = logits["shape"]
    vocabulary_size = logits_shape[2] if len(logits_shape) == 3 else 0
    if (
        functional.get("method")
        != "three_fixed_probes_bitwise_runtime_dtype_full_distribution_v2"
        or functional.get("runtime_dtype") != RUNTIME_DTYPE
        or functional.get("autocast_enabled") is not True
        or functional.get("autocast_device_type") != "cuda"
        or functional.get("input_ids") != expected_probes
        or functional.get("input_ids_sha256") != _sha256_json(FIXED_PROBES)
        or logits.get("key") != "logits"
        or logits.get("dtype") != "torch.bfloat16"
        or logits_shape[:2] != [len(FIXED_PROBES), len(FIXED_PROBES[0])]
        or vocabulary_size <= max(max(row) for row in FIXED_PROBES)
        or log_probabilities.get("key") != "log_probabilities"
        or log_probabilities.get("dtype") != "torch.float32"
        or log_probabilities.get("shape")
        != [len(FIXED_PROBES), len(FIXED_PROBES[0]) - 1, vocabulary_size]
        or argmax.get("key") != "argmax"
        or argmax.get("dtype") != "torch.int64"
        or argmax.get("shape") != [len(FIXED_PROBES), len(FIXED_PROBES[0])]
        or functional.get("bitwise_equal") is not True
        or functional.get("max_absolute_error") != 0.0
    ):
        raise SFTReloadEquivalenceError(
            "reload functional probe does not prove bitwise bfloat16 CUDA equality"
        )
    return value


def _adapter_artifact_identity(adapter: Path) -> dict[str, Any]:
    try:
        inspected = inspect_checkpoint_adapter(adapter)
    except CheckpointFidelityError as exc:
        raise SFTReloadEquivalenceError(str(exc)) from exc
    artifacts = inspected["artifacts"]
    tensors = inspected["tensors"]
    return {
        "adapter_config_sha256": artifacts["adapter_config"]["sha256"],
        "adapter_weights_sha256": artifacts["adapter_weights"]["sha256"],
        "saved_tensor_inventory_sha256": _sha256_json(tensors),
        "saved_tensor_count": len(tensors),
        "saved_parameter_count": sum(record["numel"] for record in tensors),
    }


def _implementation_identity() -> dict[str, str]:
    import clasp.models.checkpoint_fidelity as checkpoint_fidelity_module
    import clasp.models.exact_peft_kl_reference as exact_pair_module

    checkpoint_path = Path(checkpoint_fidelity_module.__file__).resolve(strict=True)
    loading_path = checkpoint_path.with_name("loading.py").resolve(strict=True)

    return {
        "producer_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "checkpoint_fidelity_sha256": _module_file_sha256(checkpoint_fidelity_module),
        "exact_pair_sha256": _module_file_sha256(exact_pair_module),
        # Resolve the sibling source without importing the PEFT-dependent loader;
        # later receipt validation therefore remains dependency-light.
        "peft_loader_sha256": _sha256_file(loading_path),
    }


def build_receipt(
    *,
    base_model_path: str | os.PathLike[str],
    adapter_path: str | os.PathLike[str],
    reloaded_trainable_closure: Mapping[str, Any],
    runtime_pair: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a receipt from evidence collected from two fresh model loads."""

    base = _canonical_directory(base_model_path, label="base model")
    adapter = _canonical_directory(adapter_path, label="SFT adapter")
    try:
        fidelity = verify_checkpoint_fidelity(adapter)
        fidelity_identity = publication_fidelity_identity(adapter)
    except (CheckpointFidelityError, OSError, ValueError) as exc:
        raise SFTReloadEquivalenceError(
            f"cannot verify pre-save checkpoint fidelity before reload audit: {exc}"
        ) from exc
    config = inspect_checkpoint_adapter(adapter)["adapter_config"]
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).expanduser().resolve()
    if configured_base != base:
        raise SFTReloadEquivalenceError(
            f"adapter reload resolves a different base: configured={configured_base}, expected={base}"
        )
    serialized_closure = fidelity.get("trainable_closure")
    if not isinstance(serialized_closure, dict):
        raise SFTReloadEquivalenceError("checkpoint fidelity has no trainable closure")
    reloaded = dict(reloaded_trainable_closure)
    if reloaded != serialized_closure:
        raise SFTReloadEquivalenceError(
            "freshly reloaded trainable tensors differ from the pre-save/serialized closure"
        )
    pair = _validate_runtime_pair(dict(runtime_pair))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "base_model_path": str(base),
        "adapter_path": str(adapter),
        "runtime_dtype": RUNTIME_DTYPE,
        "loader_contract": {
            "method": "two_independent_local_peft_loads",
            "policy_is_trainable": True,
            "reference_is_trainable": False,
            "autocast_adapter_dtype": True,
            "adapter_low_cpu_mem_usage": False,
            "local_files_only": True,
            "device_type": "cuda",
        },
        "implementation": _implementation_identity(),
        "checkpoint_fidelity": fidelity_identity,
        "adapter_artifacts": _adapter_artifact_identity(adapter),
        "serialized_trainable_closure_sha256": _sha256_json(serialized_closure),
        "reloaded_trainable_closure_sha256": _sha256_json(reloaded),
        "trainable_closure_exact_equal": True,
        "runtime_pair": pair,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


def _receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _receipt_bytes(payload)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace reload-equivalence receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace reload-equivalence receipt: {path}"
            ) from exc
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


def validate_reload_receipt(
    path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    adapter_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify a receipt and all inexpensive live artifact bindings."""

    base = _canonical_directory(base_model_path, label="base model")
    adapter = _canonical_directory(adapter_path, label="SFT adapter")
    receipt_path = Path(path).expanduser().absolute()
    if receipt_path != adapter / RECEIPT_FILENAME:
        raise SFTReloadEquivalenceError(
            f"reload receipt must be the adapter's authoritative {RECEIPT_FILENAME}"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SFTReloadEquivalenceError(
            f"reload-equivalence receipt is not a regular file: {receipt_path}"
        )
    if stat.S_IMODE(receipt_path.stat(follow_symlinks=False).st_mode) & 0o222:
        raise SFTReloadEquivalenceError(
            f"reload-equivalence receipt must be read-only: {receipt_path}"
        )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SFTReloadEquivalenceError(f"cannot read reload-equivalence receipt: {exc}") from exc
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "base_model_path",
        "adapter_path",
        "runtime_dtype",
        "loader_contract",
        "implementation",
        "checkpoint_fidelity",
        "adapter_artifacts",
        "serialized_trainable_closure_sha256",
        "reloaded_trainable_closure_sha256",
        "trainable_closure_exact_equal",
        "runtime_pair",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise SFTReloadEquivalenceError("reload-equivalence receipt has an invalid schema")
    if receipt_path.read_bytes() != _receipt_bytes(payload):
        raise SFTReloadEquivalenceError("reload-equivalence receipt is not canonical JSON")
    unsigned = dict(payload)
    stored_hash = unsigned.pop("receipt_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != _sha256_json(unsigned):
        raise SFTReloadEquivalenceError("reload-equivalence receipt self-hash mismatch")
    loader = payload.get("loader_contract")
    expected_loader = {
        "method": "two_independent_local_peft_loads",
        "policy_is_trainable": True,
        "reference_is_trainable": False,
        "autocast_adapter_dtype": True,
        "adapter_low_cpu_mem_usage": False,
        "local_files_only": True,
        "device_type": "cuda",
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("status") != STATUS
        or payload.get("base_model_path") != str(base)
        or payload.get("adapter_path") != str(adapter)
        or payload.get("runtime_dtype") != RUNTIME_DTYPE
        or loader != expected_loader
        or payload.get("trainable_closure_exact_equal") is not True
        or payload.get("serialized_trainable_closure_sha256")
        != payload.get("reloaded_trainable_closure_sha256")
    ):
        raise SFTReloadEquivalenceError("reload-equivalence receipt has invalid contract values")
    implementation = payload.get("implementation")
    if implementation != _implementation_identity():
        raise SFTReloadEquivalenceError("reload verifier implementation differs from the receipt")
    _validate_runtime_pair(payload.get("runtime_pair"))
    try:
        fidelity = verify_checkpoint_fidelity(adapter)
        live_fidelity_identity = publication_fidelity_identity(adapter)
        validate_publication_fidelity_identity(payload.get("checkpoint_fidelity"))
    except (CheckpointFidelityError, OSError, ValueError) as exc:
        raise SFTReloadEquivalenceError(f"live checkpoint fidelity verification failed: {exc}") from exc
    if payload["checkpoint_fidelity"] != live_fidelity_identity:
        raise SFTReloadEquivalenceError("reload receipt belongs to different checkpoint-fidelity bytes")
    closure = fidelity.get("trainable_closure")
    if (
        not isinstance(closure, dict)
        or payload["serialized_trainable_closure_sha256"] != _sha256_json(closure)
        or payload["adapter_artifacts"] != _adapter_artifact_identity(adapter)
    ):
        raise SFTReloadEquivalenceError("reload receipt differs from live adapter tensors")
    config = inspect_checkpoint_adapter(adapter)["adapter_config"]
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).expanduser().resolve()
    if configured_base != base:
        raise SFTReloadEquivalenceError("reload receipt adapter resolves a different base model")
    return payload


def reload_equivalence_identity(
    adapter_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return the compact identity embedded in the final five-task registry."""

    adapter = _canonical_directory(adapter_path, label="SFT adapter")
    path = adapter / RECEIPT_FILENAME
    receipt = validate_reload_receipt(
        path,
        base_model_path=base_model_path,
        adapter_path=adapter,
    )
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": STATUS,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": _sha256_file(path),
        "checkpoint_fidelity_identity_sha256": receipt["checkpoint_fidelity"][
            "identity_sha256"
        ],
        "serialized_trainable_closure_sha256": receipt[
            "serialized_trainable_closure_sha256"
        ],
        "runtime_pair_sha256": _sha256_json(receipt["runtime_pair"]),
        "functional_equivalence_sha256": _sha256_json(
            receipt["runtime_pair"]["functional_equivalence"]
        ),
        "producer_sha256": receipt["implementation"]["producer_sha256"],
    }
    identity["identity_sha256"] = _sha256_json(identity)
    return identity


def validate_reload_equivalence_identity(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "receipt_sha256",
        "receipt_file_sha256",
        "checkpoint_fidelity_identity_sha256",
        "serialized_trainable_closure_sha256",
        "runtime_pair_sha256",
        "functional_equivalence_sha256",
        "producer_sha256",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SFTReloadEquivalenceError("reload-equivalence identity has an invalid schema")
    unsigned = dict(value)
    identity_hash = unsigned.pop("identity_sha256", None)
    hash_fields = expected_keys - {"schema_version", "protocol", "status", "identity_sha256"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL
        or value.get("status") != STATUS
        or any(_SHA256_RE.fullmatch(str(value.get(field, ""))) is None for field in hash_fields)
        or identity_hash != _sha256_json(unsigned)
    ):
        raise SFTReloadEquivalenceError("reload-equivalence identity is invalid")
    return value


def _freeze_reference_parameters(reference: Any) -> None:
    """Make PEFT's independently loaded reference unambiguously immutable.

    PEFT 0.17 can retain ``requires_grad=True`` on saved ``bias="all"`` base
    biases even when ``from_pretrained(..., is_trainable=False)`` was used.
    That flag controls adapter trainability but is not, by itself, a complete
    frozen-reference guarantee for this policy.  Freeze every parameter after
    the independent load and fail closed if the runtime refuses the change.
    """

    for _, parameter in reference.named_parameters():
        parameter.requires_grad_(False)
    remaining = sorted(
        name for name, parameter in reference.named_parameters() if parameter.requires_grad
    )
    if remaining:
        raise SFTReloadEquivalenceError(
            "explicitly frozen reload reference still has trainable tensors: "
            + ", ".join(remaining)
        )


def audit_live_reload(
    *,
    base_model_path: str | os.PathLike[str],
    adapter_path: str | os.PathLike[str],
    device: str,
) -> dict[str, Any]:
    """Load two independent models and collect exact tensor/functional evidence."""

    import torch

    from clasp.models.loading import load_peft_causal_lm_with_explicit_adapter_autocast

    base = _canonical_directory(base_model_path, label="base model")
    adapter = _canonical_directory(adapter_path, label="SFT adapter")
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise SFTReloadEquivalenceError(
            "publication reload equivalence requires an explicit CUDA device"
        )
    target = torch.device(device)
    if target.index is not None and target.index >= torch.cuda.device_count():
        raise SFTReloadEquivalenceError(f"reload audit CUDA device does not exist: {device}")
    device_map = {"": str(target)}
    policy = load_peft_causal_lm_with_explicit_adapter_autocast(
        str(adapter),
        dtype=torch.bfloat16,
        device_map=device_map,
        is_trainable=True,
        trust_remote_code=True,
        local_files_only=True,
        expected_base_model_path=str(base),
    )
    reference = load_peft_causal_lm_with_explicit_adapter_autocast(
        str(adapter),
        dtype=torch.bfloat16,
        device_map=device_map,
        is_trainable=False,
        trust_remote_code=True,
        local_files_only=True,
        expected_base_model_path=str(base),
    )
    try:
        _freeze_reference_parameters(reference)
        reloaded_closure = audit_trainable_state_closure(policy, adapter_name="default")
        runtime_pair = audit_exact_runtime_pair(
            policy,
            reference,
            runtime_dtype=RUNTIME_DTYPE,
        )
        return build_receipt(
            base_model_path=base,
            adapter_path=adapter,
            reloaded_trainable_closure=reloaded_closure,
            runtime_pair=runtime_pair,
        )
    finally:
        del reference
        del policy
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--runtime-dtype", choices=[RUNTIME_DTYPE], default=RUNTIME_DTYPE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        adapter = _canonical_directory(args.adapter_path, label="SFT adapter")
        output = args.output.expanduser().absolute()
        if output != adapter / RECEIPT_FILENAME:
            raise SFTReloadEquivalenceError(
                f"--output must be {adapter / RECEIPT_FILENAME}"
            )
        payload = audit_live_reload(
            base_model_path=args.base_model_path,
            adapter_path=adapter,
            device=args.device,
        )
        _write_create_once(output, payload)
        validate_reload_receipt(
            output,
            base_model_path=args.base_model_path,
            adapter_path=adapter,
        )
        print(json.dumps(reload_equivalence_identity(adapter, base_model_path=args.base_model_path), sort_keys=True))
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[SFTReloadEquivalenceError] {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
