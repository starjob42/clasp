"""Create and verify an immutable full-model KL reference.

A JSON document containing two equal, user-supplied hashes is not evidence that
an adapter and a merged model implement the same policy.  This module therefore
has two deliberately different verification levels:

``inspect_fixed_kl_reference_artifact``
    A dependency-light integrity check.  It verifies the create-once receipt,
    exact source/base/tokenizer/output byte trees, and claim boundary, but is
    explicitly *not* sufficient to authorize beta regularization.

``verify_fixed_kl_reference``
    The authorization check.  It freshly loads the bound base and source
    adapter, merges the adapter again, freshly loads the frozen full model, and
    recomputes exact state plus deterministic functional evidence.  A forged
    receipt with plausible or equal random digests consequently fails.

The builder writes a new full-model directory and an external, self-hashed
receipt exactly once.  Development and publication receipts use distinct
protocols and cannot cross their claim boundary.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clasp.models.checkpoint_fidelity import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
    CheckpointFidelityError,
    inspect_checkpoint_adapter,
    verify_checkpoint_fidelity,
)
from clasp.models.checkpoint_fidelity import (
    RECEIPT_FILENAME as CHECKPOINT_FIDELITY_RECEIPT_FILENAME,
)

SCHEMA_VERSION = 2
BUILDER_PROTOCOL = "clasp-fixed-kl-reference-builder-v1"
PUBLICATION_PROTOCOL = "clasp-fixed-kl-reference-v2"
DEVELOPMENT_PROTOCOL = "clasp-development-fixed-kl-reference-v2"
PUBLICATION_STATUS = "publication_full_sft_step_zero_reference_verified"
DEVELOPMENT_STATUS = "development_full_saved_policy_step_zero_reference_verified"
LEGACY_CONVERSION_PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
LEGACY_CONVERSION_RECEIPT_FILENAME = "legacy_saved_policy_conversion_receipt.json"
FIXED_PROBE_TOKEN_IDS = (7, 11, 13, 17, 19, 23)
SUPPORTED_DTYPES = frozenset({"float32", "bfloat16", "float16"})

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
        "builder",
        "base_model",
        "tokenizer_source",
        "source_policy",
        "model_path",
        "model_tree",
        "tokenizer_equivalence",
        "state_equivalence",
        "functional_equivalence",
        "receipt_sha256",
    }
)


class FixedKLReferenceError(ValueError):
    """A fixed KL reference could not be proven equivalent and immutable."""


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
            raise FixedKLReferenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FixedKLReferenceError(f"expected a regular JSON file: {path}")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixedKLReferenceError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FixedKLReferenceError(f"expected a JSON object in {path}")
    return value


def _regular_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(path).expanduser().absolute()
    if raw.is_symlink():
        raise FixedKLReferenceError(f"{label} must not be a symbolic link: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise FixedKLReferenceError(f"cannot resolve {label} {raw}: {exc}") from exc
    if not resolved.is_dir():
        raise FixedKLReferenceError(f"{label} is not a directory: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise FixedKLReferenceError(f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FixedKLReferenceError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise FixedKLReferenceError(f"cannot hash {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise FixedKLReferenceError(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def _tree_manifest(
    root: Path,
    *,
    require_read_only: bool,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    root = _regular_directory(root, label="tree root")
    try:
        descendants = sorted(
            root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except OSError as exc:
        raise FixedKLReferenceError(f"cannot enumerate tree {root}: {exc}") from exc
    writable: list[str] = []
    files: list[dict[str, Any]] = []
    for path in [root, *descendants]:
        if path.is_symlink():
            raise FixedKLReferenceError(f"tree contains a symbolic link: {path}")
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise FixedKLReferenceError(f"cannot stat tree entry {path}: {exc}") from exc
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            writable.append(str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise FixedKLReferenceError(f"tree contains a non-regular entry: {path}")
        if path.name in excluded_names:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": int(metadata.st_size),
                "sha256": _sha256_file(path),
            }
        )
    if require_read_only and writable:
        raise FixedKLReferenceError(f"tree must be immutable; writable entries={writable[:8]}")
    if not files:
        raise FixedKLReferenceError(f"tree contains no auditable files: {root}")
    return {
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
        "tree_sha256": _sha256_json(files),
    }


def _compact_tree(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": tree["file_count"],
        "files": tree["files"],
        "tree_sha256": tree["tree_sha256"],
    }


def _selected_file_manifest(root: Path, names: Sequence[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        if not name or Path(name).name != name:
            raise FixedKLReferenceError(f"invalid artifact filename: {name!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FixedKLReferenceError(f"missing regular source artifact: {path}")
        records.append(
            {
                "path": name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "tree_sha256": _sha256_json(records),
    }


def _receipt_self_hash(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not _valid_sha256(claimed) or claimed != _sha256_json(unhashed):
        raise FixedKLReferenceError(f"{label} receipt self-hash mismatch")


def _configured_base(adapter_config: Mapping[str, Any]) -> Path:
    raw = adapter_config.get("base_model_name_or_path")
    if not isinstance(raw, str) or not raw:
        raise FixedKLReferenceError("source adapter lacks base_model_name_or_path")
    path = Path(raw).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise FixedKLReferenceError(
            "source adapter must designate an absolute, non-symlink local base model"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FixedKLReferenceError(
            f"cannot resolve configured adapter base {path}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise FixedKLReferenceError(f"configured adapter base is not a directory: {resolved}")
    return resolved


def _verify_development_source(source: Path, base: Path) -> dict[str, Any]:
    receipt_path = source / LEGACY_CONVERSION_RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path)
    if stat.S_IMODE(receipt_path.stat().st_mode) & 0o222:
        raise FixedKLReferenceError("development conversion receipt must be read-only")
    _receipt_self_hash(receipt, label="development conversion")
    required = {
        "schema_version": 1,
        "protocol": LEGACY_CONVERSION_PROTOCOL,
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
        raise FixedKLReferenceError(f"development conversion claim boundary mismatch: {mismatches}")
    converted = receipt.get("converted_output")
    recorded_base = receipt.get("base")
    if not isinstance(converted, dict) or not isinstance(recorded_base, dict):
        raise FixedKLReferenceError("development conversion lacks source/output evidence")
    if Path(str(converted.get("path", ""))).expanduser().absolute() != source:
        raise FixedKLReferenceError("development conversion output path mismatch")
    if Path(str(recorded_base.get("path", ""))).expanduser().absolute() != base:
        raise FixedKLReferenceError("development conversion base path mismatch")
    live_base = _tree_manifest(base, require_read_only=False)
    if recorded_base.get("manifest") != _compact_tree(live_base):
        raise FixedKLReferenceError("development conversion base bytes changed")
    live_output = _tree_manifest(
        source,
        require_read_only=True,
        excluded_names=frozenset({LEGACY_CONVERSION_RECEIPT_FILENAME}),
    )
    if converted.get("manifest_excluding_receipt") != _compact_tree(live_output):
        raise FixedKLReferenceError("development converted adapter bytes changed")
    config = _load_json_object(source / ADAPTER_CONFIG_FILENAME)
    if config.get("bias") != "all" or converted.get("serialization_bias_policy") != "all":
        raise FixedKLReferenceError("development source is not a bias='all' saved policy")
    try:
        inspection = inspect_checkpoint_adapter(source)
    except CheckpointFidelityError as exc:
        raise FixedKLReferenceError(f"cannot inspect development source adapter: {exc}") from exc
    if converted.get("checkpoint_fidelity_adapter_inspection") != inspection:
        raise FixedKLReferenceError("development source adapter inspection changed")
    return receipt


def _verify_publication_source(source: Path) -> dict[str, Any]:
    try:
        receipt = verify_checkpoint_fidelity(source)
    except CheckpointFidelityError as exc:
        raise FixedKLReferenceError(
            f"publication source lacks valid checkpoint fidelity: {exc}"
        ) from exc
    invariant = receipt.get("publication_invariant")
    if not isinstance(invariant, dict) or invariant.get("status") != "exact":
        raise FixedKLReferenceError("publication source lacks the exact Qwen invariant")
    return receipt


def _verify_source_policy(source: Path, base: Path, claim_scope: str) -> dict[str, Any]:
    if claim_scope == "publication":
        receipt_name = CHECKPOINT_FIDELITY_RECEIPT_FILENAME
        receipt = _verify_publication_source(source)
        evidence_kind = "publication_checkpoint_fidelity"
    elif claim_scope == "development":
        receipt_name = LEGACY_CONVERSION_RECEIPT_FILENAME
        receipt = _verify_development_source(source, base)
        evidence_kind = "development_legacy_saved_policy_conversion"
    else:
        raise FixedKLReferenceError(
            f"claim_scope must be 'publication' or 'development', got {claim_scope!r}"
        )
    config = _load_json_object(source / ADAPTER_CONFIG_FILENAME)
    configured_base = _configured_base(config)
    if configured_base != base:
        raise FixedKLReferenceError(
            f"source adapter base mismatch: configured={configured_base}, supplied={base}"
        )
    receipt_path = source / receipt_name
    return {
        "evidence_kind": evidence_kind,
        "checkpoint_path": str(source),
        "evidence_receipt_path": str(receipt_path),
        "evidence_receipt_file_sha256": _sha256_file(receipt_path),
        "evidence_receipt_claim_sha256": receipt.get("receipt_sha256"),
        "adapter_config_base_model_path": str(configured_base),
        "artifact_tree": _selected_file_manifest(
            source,
            (ADAPTER_CONFIG_FILENAME, ADAPTER_WEIGHTS_FILENAME, receipt_name),
        ),
    }


def _implementation_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _software_versions() -> dict[str, str]:
    try:
        import peft
        import safetensors
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - training dependency
        raise FixedKLReferenceError(f"fixed-reference dependencies unavailable: {exc}") from exc
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
    }


def _torch_dtype(name: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("torch is required for live verification") from exc
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise FixedKLReferenceError(
            f"dtype must be one of {sorted(SUPPORTED_DTYPES)}, got {name!r}"
        ) from exc


def _load_base(base: Path, *, dtype: str) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("transformers is required for live verification") from exc
    try:
        return AutoModelForCausalLM.from_pretrained(
            str(base),
            local_files_only=True,
            trust_remote_code=False,
            dtype=_torch_dtype(dtype),
            attn_implementation="eager",
        ).to("cpu")
    except Exception as exc:
        raise FixedKLReferenceError(f"cannot freshly load exact base model {base}: {exc}") from exc


def _load_source_adapter(base: Path, source: Path, *, dtype: str) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("PEFT is required for live verification") from exc
    base_model = _load_base(base, dtype=dtype)
    try:
        return PeftModel.from_pretrained(
            base_model,
            str(source),
            adapter_name="default",
            is_trainable=False,
            autocast_adapter_dtype=False,
            local_files_only=True,
        ).to("cpu")
    except Exception as exc:
        del base_model
        raise FixedKLReferenceError(f"cannot freshly load source adapter {source}: {exc}") from exc


def _load_full_model(model: Path, *, dtype: str) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("transformers is required for live verification") from exc
    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model),
            local_files_only=True,
            trust_remote_code=False,
            dtype=_torch_dtype(dtype),
            attn_implementation="eager",
        ).to("cpu")
    except Exception as exc:
        raise FixedKLReferenceError(
            f"cannot freshly load full fixed reference {model}: {exc}"
        ) from exc


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("transformers is required for tokenizer verification") from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise FixedKLReferenceError(f"cannot freshly load tokenizer {path}: {exc}") from exc


def _tensor_bytes(tensor: Any) -> bytes:
    try:
        import torch

        value = tensor.detach().resolve_conj().resolve_neg().to("cpu").contiguous()
        if value.layout != torch.strided or value.device.type == "meta":
            raise FixedKLReferenceError("cannot digest a non-strided or meta tensor")
        return value.view(torch.uint8).numpy().tobytes(order="C")
    except FixedKLReferenceError:
        raise
    except Exception as exc:
        raise FixedKLReferenceError(f"cannot materialize tensor bytes: {exc}") from exc


def _tensor_descriptor(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": int(tensor.numel()),
        "value_sha256": _sha256_bytes(_tensor_bytes(tensor)),
    }


def _state_inventory(model: Any) -> dict[str, Any]:
    try:
        state = model.state_dict()
    except Exception as exc:
        raise FixedKLReferenceError(f"cannot enumerate full model state: {exc}") from exc
    tensors = [{"key": key, **_tensor_descriptor(tensor)} for key, tensor in sorted(state.items())]
    if not tensors:
        raise FixedKLReferenceError("full model exposes no state tensors")
    return {
        "tensor_count": len(tensors),
        "parameter_count": sum(record["numel"] for record in tensors),
        "tensor_inventory_sha256": _sha256_json(tensors),
        "tensors": tensors,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception as exc:
        raise FixedKLReferenceError(f"cannot enumerate tokenizer vocabulary: {exc}") from exc
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise FixedKLReferenceError("tokenizer exposes no vocabulary")
    normalized_vocab = sorted((str(token), int(index)) for token, index in vocabulary.items())
    special_map = {
        str(key): str(value) for key, value in sorted(dict(tokenizer.special_tokens_map).items())
    }
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary_size": len(normalized_vocab),
        "vocabulary_sha256": _sha256_json(normalized_vocab),
        "special_tokens_map": special_map,
        "special_tokens_map_sha256": _sha256_json(special_map),
        "chat_template_sha256": (
            _sha256_bytes(str(chat_template).encode("utf-8")) if chat_template is not None else None
        ),
    }


def _fixed_probe(model: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FixedKLReferenceError("torch is required for live verification") from exc
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= max(FIXED_PROBE_TOKEN_IDS):
        raise FixedKLReferenceError(
            f"model vocabulary is too small for fixed probe: vocab_size={vocab_size}"
        )
    input_ids = torch.tensor([FIXED_PROBE_TOKEN_IDS], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    try:
        with torch.inference_mode():
            logits = (
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                .logits.detach()
                .to("cpu")
                .contiguous()
            )
            log_probabilities = (
                torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                .gather(-1, input_ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .contiguous()
            )
    except Exception as exc:
        raise FixedKLReferenceError(f"fixed deterministic probe failed: {exc}") from exc
    return {
        "logits": logits,
        "continuation_log_probabilities": log_probabilities,
    }


def _probe_inventory(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "logits": _tensor_descriptor(probe["logits"]),
        "continuation_log_probabilities": _tensor_descriptor(
            probe["continuation_log_probabilities"]
        ),
    }


def _tolerances(dtype: str) -> dict[str, float]:
    values = {
        "float32": {"absolute": 1e-5, "relative": 1e-5},
        "float16": {"absolute": 1e-2, "relative": 2e-3},
        "bfloat16": {"absolute": 1.25e-1, "relative": 1e-2},
    }
    try:
        return values[dtype]
    except KeyError as exc:
        raise FixedKLReferenceError(f"unsupported equivalence dtype: {dtype}") from exc


def _comparison(left: Any, right: Any, *, tolerance: Mapping[str, float]) -> dict[str, Any]:
    import torch

    if left.shape != right.shape:
        raise FixedKLReferenceError(
            f"functional output shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    delta = (left.float() - right.float()).abs()
    maximum = float(delta.max().item()) if delta.numel() else 0.0
    if not math.isfinite(maximum):
        raise FixedKLReferenceError("functional comparison produced a non-finite error")
    equal = bool(
        torch.allclose(
            left.float(),
            right.float(),
            atol=float(tolerance["absolute"]),
            rtol=float(tolerance["relative"]),
            equal_nan=False,
        )
    )
    return {"allclose": equal, "max_absolute_error": maximum}


def _comparison_failure_diagnostic(
    left: Any,
    right: Any,
    *,
    tolerance: Mapping[str, float],
) -> str:
    """Return actionable elementwise diagnostics without changing receipt schema."""

    import torch

    observed = left.float()
    expected = right.float()
    delta = (observed - expected).abs()
    allowed = float(tolerance["absolute"]) + float(tolerance["relative"]) * expected.abs()
    finite = torch.isfinite(delta) & torch.isfinite(allowed)
    failing = (~finite) | (delta > allowed)
    total = int(delta.numel())
    failing_count = int(failing.sum().item())
    if total:
        safe_delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("inf")))
        flat_index = int(torch.argmax(safe_delta.reshape(-1)).item())
        max_absolute = float(safe_delta.reshape(-1)[flat_index].item())
        observed_at_max = float(observed.reshape(-1)[flat_index].item())
        expected_at_max = float(expected.reshape(-1)[flat_index].item())
        allowed_at_max = float(allowed.reshape(-1)[flat_index].item())
        normalized = torch.where(
            finite,
            delta / allowed,
            torch.full_like(delta, float("inf")),
        )
        max_normalized = float(normalized.max().item())
    else:
        max_absolute = 0.0
        observed_at_max = 0.0
        expected_at_max = 0.0
        allowed_at_max = float(tolerance["absolute"])
        max_normalized = 0.0
    return (
        f"max_absolute_error={max_absolute!r}, "
        f"max_normalized_error={max_normalized!r}, "
        f"failing_elements={failing_count}/{total}, "
        f"atol={float(tolerance['absolute'])!r}, "
        f"rtol={float(tolerance['relative'])!r}, "
        f"observed_at_max={observed_at_max!r}, "
        f"expected_at_max={expected_at_max!r}, "
        f"allowed_at_max={allowed_at_max!r}"
    )


def _functional_evidence(
    source: Mapping[str, Any],
    merged: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    dtype: str,
) -> dict[str, Any]:
    import torch

    tolerance = _tolerances(dtype)
    comparisons: dict[str, Any] = {}
    for field in ("logits", "continuation_log_probabilities"):
        comparisons[field] = {
            "source_vs_merged": _comparison(source[field], merged[field], tolerance=tolerance),
            "source_vs_fresh_reference": _comparison(
                source[field], reference[field], tolerance=tolerance
            ),
            "merged_vs_fresh_reference_bitwise_equal": bool(
                torch.equal(merged[field], reference[field])
            ),
        }
        if not comparisons[field]["source_vs_merged"]["allclose"]:
            raise FixedKLReferenceError(
                f"merged policy is not numerically equivalent for probe {field}: "
                + _comparison_failure_diagnostic(
                    source[field],
                    merged[field],
                    tolerance=tolerance,
                )
            )
        if not comparisons[field]["source_vs_fresh_reference"]["allclose"]:
            raise FixedKLReferenceError(
                f"fresh reference is not numerically equivalent for probe {field}: "
                + _comparison_failure_diagnostic(
                    source[field],
                    reference[field],
                    tolerance=tolerance,
                )
            )
        if not comparisons[field]["merged_vs_fresh_reference_bitwise_equal"]:
            raise FixedKLReferenceError(f"fresh reference probe is not bitwise stable for {field}")
    return {
        "method": "fresh_source_merge_and_fresh_full_reload_fixed_probe_v1",
        "device": "cpu",
        "requested_dtype": dtype,
        "input_ids": list(FIXED_PROBE_TOKEN_IDS),
        "input_ids_sha256": _sha256_json(list(FIXED_PROBE_TOKEN_IDS)),
        "tolerances": tolerance,
        "source": _probe_inventory(source),
        "merged": _probe_inventory(merged),
        "fresh_reference": _probe_inventory(reference),
        "comparisons": comparisons,
    }


def _state_evidence(merged: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    if merged != reference:
        raise FixedKLReferenceError(
            "fresh full-model reload differs from the just-merged exact state inventory"
        )
    return {
        "method": "fresh_source_merge_vs_fresh_full_reload_exact_tensor_inventory_v1",
        "exact": True,
        **merged,
    }


def _reconstruct_live_evidence(
    *,
    base: Path,
    source: Path,
    reference: Path,
    tokenizer_source: Path,
    dtype: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_model = None
    merged_model = None
    reference_model = None
    source_tokenizer = None
    reference_tokenizer = None
    try:
        source_model = _load_source_adapter(base, source, dtype=dtype)
        source_probe = _fixed_probe(source_model)
        try:
            merged_model = source_model.merge_and_unload(safe_merge=True)
        except Exception as exc:
            raise FixedKLReferenceError(f"cannot safely merge source adapter: {exc}") from exc
        source_model = None
        merged_state = _state_inventory(merged_model)
        merged_probe = _fixed_probe(merged_model)
        merged_model = None
        gc.collect()

        reference_model = _load_full_model(reference, dtype=dtype)
        reference_state = _state_inventory(reference_model)
        reference_probe = _fixed_probe(reference_model)
        state_evidence = _state_evidence(merged_state, reference_state)
        functional_evidence = _functional_evidence(
            source_probe,
            merged_probe,
            reference_probe,
            dtype=dtype,
        )
        source_tokenizer = _load_tokenizer(tokenizer_source)
        reference_tokenizer = _load_tokenizer(reference)
        source_fingerprint = _tokenizer_fingerprint(source_tokenizer)
        reference_fingerprint = _tokenizer_fingerprint(reference_tokenizer)
        if source_fingerprint != reference_fingerprint:
            raise FixedKLReferenceError(
                "fresh reference tokenizer is not identical to the bound source tokenizer"
            )
        tokenizer_evidence = {
            "method": "fresh_reload_exact_vocabulary_special_tokens_and_chat_template_v1",
            "exact": True,
            "fingerprint": source_fingerprint,
        }
        return state_evidence, functional_evidence, tokenizer_evidence
    finally:
        # Clear the caller-owned references themselves before collecting.  Passing
        # a model to a helper and deleting only the helper's local argument leaves
        # this frame's reference alive, which can keep safetensors mmap handles
        # open and turn an otherwise ordinary cleanup into an NFS ``.nfs`` race.
        source_model = None
        merged_model = None
        reference_model = None
        source_tokenizer = None
        reference_tokenizer = None
        gc.collect()


def _claim_contract(claim_scope: str) -> dict[str, Any]:
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
    raise FixedKLReferenceError(
        f"claim_scope must be 'publication' or 'development', got {claim_scope!r}"
    )


def _builder_record(dtype: str) -> dict[str, Any]:
    return {
        "protocol": BUILDER_PROTOCOL,
        "implementation": "clasp.models.fixed_kl_reference",
        "implementation_sha256": _implementation_sha256(),
        "device": "cpu",
        "requested_dtype": dtype,
        "software": _software_versions(),
    }


def _validate_inventory(value: object) -> None:
    if not isinstance(value, dict):
        raise FixedKLReferenceError("fixed reference lacks exact state evidence")
    required = {
        "method",
        "exact",
        "tensor_count",
        "parameter_count",
        "tensor_inventory_sha256",
        "tensors",
    }
    if set(value) != required:
        raise FixedKLReferenceError("fixed reference state evidence schema mismatch")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise FixedKLReferenceError("fixed reference state inventory is empty")
    previous: str | None = None
    for record in tensors:
        if not isinstance(record, dict) or set(record) != {
            "key",
            "shape",
            "dtype",
            "numel",
            "value_sha256",
        }:
            raise FixedKLReferenceError("invalid fixed reference tensor record")
        key = record.get("key")
        shape = record.get("shape")
        numel = record.get("numel")
        if (
            not isinstance(key, str)
            or not key
            or (previous is not None and key <= previous)
            or not isinstance(shape, list)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
            )
            or not isinstance(numel, int)
            or isinstance(numel, bool)
            or numel != math.prod(shape)
            or not isinstance(record.get("dtype"), str)
            or not _valid_sha256(record.get("value_sha256"))
        ):
            raise FixedKLReferenceError(f"invalid fixed reference tensor metadata: {key!r}")
        previous = key
    if (
        value.get("method") != "fresh_source_merge_vs_fresh_full_reload_exact_tensor_inventory_v1"
        or value.get("exact") is not True
        or value.get("tensor_count") != len(tensors)
        or value.get("parameter_count") != sum(record["numel"] for record in tensors)
        or value.get("tensor_inventory_sha256") != _sha256_json(tensors)
    ):
        raise FixedKLReferenceError("fixed reference state summary is inconsistent")


def _validate_functional(value: object, *, dtype: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "method",
        "device",
        "requested_dtype",
        "input_ids",
        "input_ids_sha256",
        "tolerances",
        "source",
        "merged",
        "fresh_reference",
        "comparisons",
    }:
        raise FixedKLReferenceError("fixed reference functional evidence schema mismatch")
    if (
        value.get("method") != "fresh_source_merge_and_fresh_full_reload_fixed_probe_v1"
        or value.get("device") != "cpu"
        or value.get("requested_dtype") != dtype
        or value.get("input_ids") != list(FIXED_PROBE_TOKEN_IDS)
        or value.get("input_ids_sha256") != _sha256_json(list(FIXED_PROBE_TOKEN_IDS))
        or value.get("tolerances") != _tolerances(dtype)
    ):
        raise FixedKLReferenceError("fixed reference functional method/probe mismatch")
    for group in ("source", "merged", "fresh_reference"):
        output = value.get(group)
        if not isinstance(output, dict) or set(output) != {
            "logits",
            "continuation_log_probabilities",
        }:
            raise FixedKLReferenceError(f"invalid functional output group: {group}")
        for descriptor in output.values():
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"shape", "dtype", "numel", "value_sha256"}
                or not _valid_sha256(descriptor.get("value_sha256"))
            ):
                raise FixedKLReferenceError("invalid functional tensor descriptor")
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {
        "logits",
        "continuation_log_probabilities",
    }:
        raise FixedKLReferenceError("invalid functional comparison evidence")
    for comparison in comparisons.values():
        if not isinstance(comparison, dict) or set(comparison) != {
            "source_vs_merged",
            "source_vs_fresh_reference",
            "merged_vs_fresh_reference_bitwise_equal",
        }:
            raise FixedKLReferenceError("invalid functional comparison schema")
        for pair in ("source_vs_merged", "source_vs_fresh_reference"):
            record = comparison.get(pair)
            if (
                not isinstance(record, dict)
                or set(record) != {"allclose", "max_absolute_error"}
                or record.get("allclose") is not True
                or not isinstance(record.get("max_absolute_error"), (float, int))
                or not math.isfinite(float(record["max_absolute_error"]))
            ):
                raise FixedKLReferenceError("unsuccessful functional equivalence claim")
        if comparison.get("merged_vs_fresh_reference_bitwise_equal") is not True:
            raise FixedKLReferenceError("fresh reference functional reload is not bitwise exact")


def _validate_tokenizer(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"method", "exact", "fingerprint"}:
        raise FixedKLReferenceError("fixed reference tokenizer evidence schema mismatch")
    fingerprint = value.get("fingerprint")
    if (
        value.get("method") != "fresh_reload_exact_vocabulary_special_tokens_and_chat_template_v1"
        or value.get("exact") is not True
        or not isinstance(fingerprint, dict)
        or set(fingerprint)
        != {
            "tokenizer_class",
            "vocabulary_size",
            "vocabulary_sha256",
            "special_tokens_map",
            "special_tokens_map_sha256",
            "chat_template_sha256",
        }
        or not _valid_sha256(fingerprint.get("vocabulary_sha256"))
        or not _valid_sha256(fingerprint.get("special_tokens_map_sha256"))
        or (
            fingerprint.get("chat_template_sha256") is not None
            and not _valid_sha256(fingerprint.get("chat_template_sha256"))
        )
    ):
        raise FixedKLReferenceError("invalid fixed reference tokenizer evidence")


def inspect_fixed_kl_reference_artifact(
    model_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    claim_scope: str,
) -> dict[str, Any]:
    """Check exact bytes and schema without authorizing beta regularization."""

    model = _regular_directory(model_path, label="fixed KL reference model")
    receipt_file = Path(receipt_path).expanduser().absolute()
    if receipt_file == model or model in receipt_file.parents:
        raise FixedKLReferenceError("receipt must be outside the hashed model tree")
    receipt = _load_json_object(receipt_file)
    if stat.S_IMODE(receipt_file.stat().st_mode) & 0o222:
        raise FixedKLReferenceError("fixed KL reference receipt must be read-only")
    if set(receipt) != _TOP_LEVEL_KEYS:
        raise FixedKLReferenceError("fixed KL reference receipt schema mismatch")
    _receipt_self_hash(receipt, label="fixed KL reference")
    contract = _claim_contract(claim_scope)
    required = {
        "schema_version": SCHEMA_VERSION,
        **contract,
        "purpose": "frozen_step_zero_kl_reference",
    }
    mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise FixedKLReferenceError(f"fixed KL reference claim boundary mismatch: {mismatches}")
    if Path(str(receipt.get("model_path", ""))).expanduser().absolute() != model:
        raise FixedKLReferenceError("fixed KL reference model path mismatch")
    builder = receipt.get("builder")
    if not isinstance(builder, dict) or set(builder) != {
        "protocol",
        "implementation",
        "implementation_sha256",
        "device",
        "requested_dtype",
        "software",
    }:
        raise FixedKLReferenceError("fixed KL reference builder evidence schema mismatch")
    dtype = builder.get("requested_dtype")
    if (
        builder.get("protocol") != BUILDER_PROTOCOL
        or builder.get("implementation") != "clasp.models.fixed_kl_reference"
        or builder.get("implementation_sha256") != _implementation_sha256()
        or builder.get("device") != "cpu"
        or dtype not in SUPPORTED_DTYPES
        or not isinstance(builder.get("software"), dict)
    ):
        raise FixedKLReferenceError("fixed KL reference was not made by this trusted builder")

    base_record = receipt.get("base_model")
    tokenizer_record = receipt.get("tokenizer_source")
    if not isinstance(base_record, dict) or set(base_record) != {"path", "tree"}:
        raise FixedKLReferenceError("fixed KL reference lacks exact base-model binding")
    if not isinstance(tokenizer_record, dict) or set(tokenizer_record) != {"path", "tree"}:
        raise FixedKLReferenceError("fixed KL reference lacks exact tokenizer-source binding")
    base = _regular_directory(str(base_record["path"]), label="recorded base model")
    tokenizer_source = _regular_directory(
        str(tokenizer_record["path"]), label="recorded tokenizer source"
    )
    live_base_tree = _tree_manifest(base, require_read_only=False)
    if base_record.get("tree") != live_base_tree:
        raise FixedKLReferenceError("fixed KL reference base-model bytes changed")
    live_tokenizer_tree = (
        live_base_tree
        if tokenizer_source == base
        else _tree_manifest(tokenizer_source, require_read_only=False)
    )
    if tokenizer_record.get("tree") != live_tokenizer_tree:
        raise FixedKLReferenceError("fixed KL reference tokenizer-source bytes changed")

    source_record = receipt.get("source_policy")
    if not isinstance(source_record, dict):
        raise FixedKLReferenceError("fixed KL reference lacks source-policy binding")
    source = _regular_directory(
        str(source_record.get("checkpoint_path", "")), label="recorded source policy"
    )
    live_source = _verify_source_policy(source, base, claim_scope)
    if source_record != live_source:
        raise FixedKLReferenceError("fixed KL reference source-policy evidence changed")
    live_model_tree = _tree_manifest(model, require_read_only=True)
    if receipt.get("model_tree") != live_model_tree:
        raise FixedKLReferenceError("fixed KL reference model bytes changed")
    _validate_inventory(receipt.get("state_equivalence"))
    _validate_functional(receipt.get("functional_equivalence"), dtype=str(dtype))
    _validate_tokenizer(receipt.get("tokenizer_equivalence"))
    return receipt


def verify_fixed_kl_reference(
    model_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    claim_scope: str,
) -> dict[str, Any]:
    """Authorize beta only after recomputing merge, state, tokenizer, and logits."""

    receipt = inspect_fixed_kl_reference_artifact(
        model_path,
        receipt_path,
        claim_scope=claim_scope,
    )
    model = _regular_directory(model_path, label="fixed KL reference model")
    base = _regular_directory(receipt["base_model"]["path"], label="recorded base model")
    source = _regular_directory(
        receipt["source_policy"]["checkpoint_path"], label="recorded source policy"
    )
    tokenizer_source = _regular_directory(
        receipt["tokenizer_source"]["path"], label="recorded tokenizer source"
    )
    dtype = str(receipt["builder"]["requested_dtype"])
    state, functional, tokenizer = _reconstruct_live_evidence(
        base=base,
        source=source,
        reference=model,
        tokenizer_source=tokenizer_source,
        dtype=dtype,
    )
    if state != receipt.get("state_equivalence"):
        raise FixedKLReferenceError("freshly reconstructed exact merged state differs from receipt")
    if functional != receipt.get("functional_equivalence"):
        raise FixedKLReferenceError(
            "freshly reconstructed deterministic functional evidence differs from receipt"
        )
    if tokenizer != receipt.get("tokenizer_equivalence"):
        raise FixedKLReferenceError("freshly reconstructed tokenizer evidence differs from receipt")
    return receipt


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise FixedKLReferenceError(f"refusing to freeze symlinked output: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _thaw_owned_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in root.rglob("*"):
        try:
            if path.is_dir():
                path.chmod(stat.S_IRWXU)
            elif path.is_file():
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    try:
        root.chmod(stat.S_IRWXU)
    except OSError:
        pass


def _remove_owned_tree(root: Path, *, label: str) -> str | None:
    """Best-effort removal that reports, rather than raises, cleanup failures."""

    try:
        _thaw_owned_tree(root)
        shutil.rmtree(root)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{label} {root}: {exc.__class__.__name__}: {exc}"
    return None


def _remove_owned_file(path: Path, *, label: str) -> str | None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.unlink()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{label} {path}: {exc.__class__.__name__}: {exc}"
    return None


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"


def _write_create_once(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace fixed KL receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def build_fixed_kl_reference(
    *,
    base_model_path: str | os.PathLike[str],
    source_adapter_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    output_model_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    claim_scope: str,
    dtype: str,
) -> dict[str, Any]:
    """Create-once merge, fresh-reload it, and emit authorization evidence."""

    if dtype not in SUPPORTED_DTYPES:
        raise FixedKLReferenceError(
            f"dtype must be one of {sorted(SUPPORTED_DTYPES)}, got {dtype!r}"
        )
    base = _regular_directory(base_model_path, label="base model")
    source = _regular_directory(source_adapter_path, label="source adapter")
    tokenizer_source = _regular_directory(tokenizer_path, label="tokenizer source")
    raw_output = Path(output_model_path).expanduser().absolute()
    receipt_file = Path(receipt_path).expanduser().absolute()
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite fixed KL model: {raw_output}")
    if receipt_file.exists() or receipt_file.is_symlink():
        raise FileExistsError(f"refusing to overwrite fixed KL receipt: {receipt_file}")
    if receipt_file == raw_output or raw_output in receipt_file.parents:
        raise FixedKLReferenceError("receipt must be outside the hashed model tree")
    protected_roots = {base, source, tokenizer_source}
    if any(
        raw_output == protected
        or raw_output in protected.parents
        or protected in raw_output.parents
        for protected in protected_roots
    ):
        raise FixedKLReferenceError(
            "fixed KL output must be disjoint from base, source adapter, and tokenizer trees"
        )
    if any(
        receipt_file == protected or protected in receipt_file.parents
        for protected in protected_roots
    ):
        raise FixedKLReferenceError(
            "fixed KL receipt must be outside base, source adapter, and tokenizer trees"
        )
    source_policy = _verify_source_policy(source, base, claim_scope)
    base_tree = _tree_manifest(base, require_read_only=False)
    tokenizer_tree = (
        base_tree
        if tokenizer_source == base
        else _tree_manifest(tokenizer_source, require_read_only=False)
    )
    contract = _claim_contract(claim_scope)
    builder = _builder_record(dtype)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{raw_output.name}.staging.", dir=raw_output.parent)
    ).resolve(strict=True)
    source_model = None
    merged_model = None
    fresh_model = None
    source_tokenizer = None
    fresh_tokenizer = None
    success = False
    output_committed = False
    receipt_created = False
    try:
        source_model = _load_source_adapter(base, source, dtype=dtype)
        source_probe = _fixed_probe(source_model)
        try:
            merged_model = source_model.merge_and_unload(safe_merge=True)
        except Exception as exc:
            raise FixedKLReferenceError(f"cannot safely merge source adapter: {exc}") from exc
        source_model = None
        merged_state = _state_inventory(merged_model)
        merged_probe = _fixed_probe(merged_model)
        try:
            merged_model.save_pretrained(
                str(staging),
                safe_serialization=True,
                max_shard_size="5GB",
            )
        except Exception as exc:
            raise FixedKLReferenceError(f"cannot save merged full model: {exc}") from exc
        source_tokenizer = _load_tokenizer(tokenizer_source)
        source_tokenizer_fingerprint = _tokenizer_fingerprint(source_tokenizer)
        try:
            source_tokenizer.save_pretrained(str(staging))
        except Exception as exc:
            raise FixedKLReferenceError(f"cannot save bound tokenizer: {exc}") from exc
        merged_model = None
        gc.collect()

        fresh_model = _load_full_model(staging, dtype=dtype)
        fresh_state = _state_inventory(fresh_model)
        fresh_probe = _fixed_probe(fresh_model)
        state_evidence = _state_evidence(merged_state, fresh_state)
        functional_evidence = _functional_evidence(
            source_probe,
            merged_probe,
            fresh_probe,
            dtype=dtype,
        )
        fresh_tokenizer = _load_tokenizer(staging)
        fresh_tokenizer_fingerprint = _tokenizer_fingerprint(fresh_tokenizer)
        if source_tokenizer_fingerprint != fresh_tokenizer_fingerprint:
            raise FixedKLReferenceError("saved tokenizer differs from its exact source")
        tokenizer_evidence = {
            "method": "fresh_reload_exact_vocabulary_special_tokens_and_chat_template_v1",
            "exact": True,
            "fingerprint": source_tokenizer_fingerprint,
        }
        fresh_model = None
        gc.collect()

        final_base_tree = _tree_manifest(base, require_read_only=False)
        if final_base_tree != base_tree:
            raise FixedKLReferenceError("base model changed during fixed-reference build")
        final_tokenizer_tree = (
            final_base_tree
            if tokenizer_source == base
            else _tree_manifest(tokenizer_source, require_read_only=False)
        )
        if final_tokenizer_tree != tokenizer_tree:
            raise FixedKLReferenceError("tokenizer source changed during fixed-reference build")
        if _verify_source_policy(source, base, claim_scope) != source_policy:
            raise FixedKLReferenceError("source policy changed during fixed-reference build")
        _make_read_only(staging)
        model_tree = _tree_manifest(staging, require_read_only=True)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            **contract,
            "purpose": "frozen_step_zero_kl_reference",
            "builder": builder,
            "base_model": {"path": str(base), "tree": base_tree},
            "tokenizer_source": {
                "path": str(tokenizer_source),
                "tree": tokenizer_tree,
            },
            "source_policy": source_policy,
            "model_path": str(raw_output),
            "model_tree": model_tree,
            "tokenizer_equivalence": tokenizer_evidence,
            "state_equivalence": state_evidence,
            "functional_equivalence": functional_evidence,
        }
        payload["receipt_sha256"] = _sha256_json(payload)

        if raw_output.exists() or raw_output.is_symlink():
            raise FileExistsError(f"fixed KL model appeared during build: {raw_output}")
        os.rename(staging, raw_output)
        output_committed = True
        _write_create_once(receipt_file, _receipt_bytes(payload))
        receipt_created = True
        verified = verify_fixed_kl_reference(
            raw_output,
            receipt_file,
            claim_scope=claim_scope,
        )
        success = True
        return verified
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors: list[str] = []
        try:
            # These assignments must happen in this frame before rmtree.  A
            # helper-local ``del model`` does not release caller references and
            # can leave safetensors mmap handles behind as NFS ``.nfs`` files.
            source_model = None
            merged_model = None
            fresh_model = None
            source_tokenizer = None
            fresh_tokenizer = None
            gc.collect()
        except Exception as exc:  # pragma: no cover - defensive around third-party finalizers
            cleanup_errors.append(f"model release: {exc.__class__.__name__}: {exc}")
        if staging.exists():
            cleanup_error = _remove_owned_tree(staging, label="staging tree")
            if cleanup_error is not None:
                cleanup_errors.append(cleanup_error)
        if not success:
            if receipt_created and receipt_file.exists():
                cleanup_error = _remove_owned_file(receipt_file, label="receipt")
                if cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)
            if output_committed and raw_output.exists():
                # Remove only the known-new output owned by this failed build.
                cleanup_error = _remove_owned_tree(raw_output, label="committed output")
                if cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            message = (
                "fixed-KL builder cleanup was incomplete; owned paths may require later "
                "removal (for example after an NFS .nfs handle closes): "
                + "; ".join(cleanup_errors)
            )
            if primary_error is None:
                raise FixedKLReferenceError(message)
            try:
                warnings.warn(
                    message + "; the primary build failure was preserved",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                # Warning filters may promote warnings to exceptions.  Cleanup
                # reporting must still never replace the scientific failure.
                pass
