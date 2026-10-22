"""Training-only orchestration for publication checkpoint fidelity.

The core :mod:`clasp.models.checkpoint_fidelity` module stays dependency-light
for artifact verification.  This module owns the Trainer callback, distributed
preflight, environment gate, and explicit fixed-KL-reference contract used by
publication training.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import TrainerCallback  # type: ignore[import-not-found]

from clasp.models.checkpoint_fidelity import (
    ADAPTER_CONFIG_FILENAME,
    RECEIPT_FILENAME,
    CheckpointFidelityError,
    _tensor_record,
    assert_qwen25_7b_publication_invariant,
    audit_trainable_state_closure,
    canonicalize_peft_adapter_key,
    inspect_checkpoint_adapter,
    verify_checkpoint_fidelity,
    write_checkpoint_fidelity_receipt,
)
from clasp.models.exact_peft_kl_post_step import (
    POST_STEP_RECEIPT_FILENAME,
    STEP1_RECEIPT_FILENAME,
    ExactPeftKLPostStepError,
    audit_live_post_step_reference,
    build_post_step_receipt,
    inspect_post_step_receipt,
    pre_step_reference_identity,
    step1_reference_binding,
    training_run_contract_identity,
    write_post_step_receipt,
)
from clasp.models.exact_peft_kl_reference import (
    DEVELOPMENT_PROTOCOL as _DEVELOPMENT_EXACT_PEFT_KL_REFERENCE_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    PUBLICATION_PROTOCOL as _EXACT_PEFT_KL_REFERENCE_PROTOCOL,
)
from clasp.models.exact_peft_kl_reference import (
    RECEIPT_FILENAME as EXACT_PEFT_KL_RECEIPT_FILENAME,
)
from clasp.models.exact_peft_kl_reference import (
    ExactPeftKLReferenceError,
    audit_exact_runtime_pair,
    build_runtime_receipt,
    build_source_binding,
    inspect_runtime_receipt,
    verify_development_relocation_authority,
    verify_source_binding_unchanged,
    write_runtime_receipt,
)
from clasp.models.fixed_kl_reference import (
    DEVELOPMENT_PROTOCOL,
    PUBLICATION_PROTOCOL,
    FixedKLReferenceError,
)
from clasp.models.fixed_kl_reference import (
    verify_fixed_kl_reference as _verify_fixed_kl_reference_live,
)

PUBLICATION_FIDELITY_ENV = "CLASP_PUBLICATION_FIDELITY"
DEVELOPMENT_FIDELITY_ENV = "CLASP_DEVELOPMENT_FIDELITY"
KL_REFERENCE_MODEL_PATH_ENV = "CLASP_KL_REFERENCE_MODEL_PATH"
KL_REFERENCE_RECEIPT_ENV = "CLASP_KL_REFERENCE_RECEIPT"
TRAINING_RUN_CONTRACT_ENV = "CLASP_TRAINING_RUN_CONTRACT_PATH"
POLICY_CONSISTENCY_CHECKS_ENV = "CLASP_POLICY_CONSISTENCY_CHECKS"
POLICY_CONSISTENCY_CHECK_INTERVAL_ENV = "CLASP_POLICY_CONSISTENCY_CHECK_INTERVAL"
POLICY_CONSISTENCY_RECEIPT_FILENAME = "policy_consistency_checks.json"
GRADIENT_CONSISTENCY_CHECKS_ENV = "CLASP_GRADIENT_CONSISTENCY_CHECKS"
GRADIENT_CONSISTENCY_RECEIPT_FILENAME = "gradient_consistency_checks.json"
FIXED_KL_REFERENCE_PROTOCOL = PUBLICATION_PROTOCOL
DEVELOPMENT_FIXED_KL_REFERENCE_PROTOCOL = DEVELOPMENT_PROTOCOL
EXACT_PEFT_KL_REFERENCE_PROTOCOL = _EXACT_PEFT_KL_REFERENCE_PROTOCOL
DEVELOPMENT_EXACT_PEFT_KL_REFERENCE_PROTOCOL = _DEVELOPMENT_EXACT_PEFT_KL_REFERENCE_PROTOCOL
LEGACY_CONVERSION_PROTOCOL = "clasp-legacy-saved-policy-reencoding-v1"
LEGACY_CONVERSION_RECEIPT_FILENAME = "legacy_saved_policy_conversion_receipt.json"
DEVELOPMENT_RELOCATION_PROTOCOL = "clasp-development-adapter-base-relocation-v1"
DEVELOPMENT_RELOCATION_RECEIPT_FILENAME = "adapter_base_relocation_receipt.json"
DEVELOPMENT_RELOCATION_VERIFIER_PROTOCOL = "clasp-development-adapter-base-relocation-verifier-v1"
_VERIFIED_ADAPTER_CACHE: dict[str, tuple[tuple[tuple[int, ...], ...], dict[str, Any]]] = {}
_VERIFIED_CONVERSION_CACHE: dict[
    str,
    tuple[tuple[tuple[int, ...], ...], dict[str, Any]],
] = {}


class PublicationFidelityError(RuntimeError):
    """A publication training integrity requirement was not satisfied."""


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
    try:
        before = path.lstat()
    except OSError as exc:
        raise PublicationFidelityError(f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PublicationFidelityError(f"expected a regular file: {path}")
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
        raise PublicationFidelityError(f"cannot hash {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise PublicationFidelityError(f"file changed while it was hashed: {path}")
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublicationFidelityError(f"expected a regular JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationFidelityError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicationFidelityError(f"expected a JSON object in {path}")
    return payload


def _training_run_contract_identity(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Bind the immutable full run authorization before optimization."""

    root = Path(output_dir).expanduser().absolute()
    explicit = str(os.environ.get(TRAINING_RUN_CONTRACT_ENV, "")).strip()
    if explicit:
        path = Path(explicit).expanduser().absolute()
    else:
        candidates = [
            root / "run_contract.json",
            root / "local_development_run_contract.json",
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) != 1:
            raise PublicationFidelityError(
                "exact PEFT KL mode requires exactly one immutable training run contract; "
                f"set {TRAINING_RUN_CONTRACT_ENV} explicitly, candidates={existing}"
            )
        path = existing[0]
    try:
        return training_run_contract_identity(
            path,
            claim_scope="development" if development_fidelity_enabled() else "publication",
        )
    except ExactPeftKLPostStepError as exc:
        raise PublicationFidelityError(f"invalid immutable training run contract: {exc}") from exc


def resolve_training_run_contract_identity(
    output_dir: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Bind the immutable GRPO run contract for every fidelity-mode arm."""

    if not checkpoint_fidelity_enabled():
        return None
    return _training_run_contract_identity(output_dir)


def _strict_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PublicationFidelityError(
        f"{name} must be an explicit boolean (0/1, false/true, no/yes, off/on); got {raw!r}"
    )


def publication_fidelity_enabled() -> bool:
    """Return the strict opt-in publication-fidelity mode."""

    publication = _strict_flag(PUBLICATION_FIDELITY_ENV, False)
    development = _strict_flag(DEVELOPMENT_FIDELITY_ENV, False)
    if publication and development:
        raise PublicationFidelityError(
            f"{PUBLICATION_FIDELITY_ENV} and {DEVELOPMENT_FIDELITY_ENV} are mutually exclusive"
        )
    return publication


def development_fidelity_enabled() -> bool:
    """Return the explicit development-only saved-policy fidelity mode."""

    development = _strict_flag(DEVELOPMENT_FIDELITY_ENV, False)
    publication = _strict_flag(PUBLICATION_FIDELITY_ENV, False)
    if publication and development:
        raise PublicationFidelityError(
            f"{PUBLICATION_FIDELITY_ENV} and {DEVELOPMENT_FIDELITY_ENV} are mutually exclusive"
        )
    return development


def checkpoint_fidelity_enabled() -> bool:
    """Return whether exact saved-policy closure is mandatory for this run."""

    return publication_fidelity_enabled() or development_fidelity_enabled()


def checkpoint_fidelity_mode() -> str:
    """Return ``off``, ``development``, or ``publication``."""

    if publication_fidelity_enabled():
        return "publication"
    if development_fidelity_enabled():
        return "development"
    return "off"


def _policy_consistency_check_settings() -> dict[str, Any]:
    """Freeze the optional early optimizer-step checks at callback construction."""

    enabled = _strict_flag(POLICY_CONSISTENCY_CHECKS_ENV, False)
    gradient_enabled = _strict_flag(GRADIENT_CONSISTENCY_CHECKS_ENV, False)
    if gradient_enabled and not enabled:
        raise PublicationFidelityError(
            f"{GRADIENT_CONSISTENCY_CHECKS_ENV} requires {POLICY_CONSISTENCY_CHECKS_ENV}"
        )
    if not enabled:
        return {
            "enabled": False, "gradient_enabled": False,
            "initial_optimizer_steps": [1, 2, 5], "interval": 0,
        }
    if not checkpoint_fidelity_enabled():
        raise PublicationFidelityError(
            f"{POLICY_CONSISTENCY_CHECKS_ENV} requires development or publication fidelity"
        )
    raw_interval = os.environ.get(POLICY_CONSISTENCY_CHECK_INTERVAL_ENV, "0")
    try:
        interval = int(raw_interval)
    except (TypeError, ValueError) as exc:
        raise PublicationFidelityError(
            f"{POLICY_CONSISTENCY_CHECK_INTERVAL_ENV} must be an integer >= 0"
        ) from exc
    if interval < 0:
        raise PublicationFidelityError(
            f"{POLICY_CONSISTENCY_CHECK_INTERVAL_ENV} must be an integer >= 0"
        )
    return {
        "enabled": True, "gradient_enabled": gradient_enabled,
        "initial_optimizer_steps": [1, 2, 5], "interval": interval,
    }


def require_publication_lora_configuration(
    *,
    full_finetune: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    effective_bias: str,
) -> None:
    """Reject a publication run before allocating an optimizer or generating."""

    if not checkpoint_fidelity_enabled():
        return
    expected_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    mismatches = []
    if bool(full_finetune):
        mismatches.append("full_finetune must be false")
    if int(lora_r) != 64:
        mismatches.append(f"lora_r={lora_r!r}, expected 64")
    if int(lora_alpha) != 64:
        mismatches.append(f"lora_alpha={lora_alpha!r}, expected 64")
    if float(lora_dropout) != 0.0:
        mismatches.append(f"lora_dropout={lora_dropout!r}, expected 0.0")
    if set(target_modules) != expected_targets:
        mismatches.append("target_modules are not the exact seven Jiang modules")
    if str(effective_bias).strip().lower() != "all":
        mismatches.append(f"effective LoRA bias={effective_bias!r}, expected 'all'")
    if mismatches:
        raise PublicationFidelityError(
            "Checkpoint fidelity configuration mismatch: " + "; ".join(mismatches)
        )


def _artifact_cache_identity(checkpoint: Path) -> tuple[tuple[int, ...], ...]:
    identities = []
    for name in (RECEIPT_FILENAME, ADAPTER_CONFIG_FILENAME, "adapter_model.safetensors"):
        path = checkpoint / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise PublicationFidelityError(
                f"cannot stat publication adapter file {path}: {exc}"
            ) from exc
        identities.append(
            (
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
                int(stat.S_IMODE(metadata.st_mode)),
            )
        )
    return tuple(identities)


def verify_publication_adapter_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    purpose: str,
) -> dict[str, Any]:
    """Require a publication Qwen receipt and exact live adapter bytes."""

    checkpoint = Path(checkpoint_dir).expanduser().absolute()
    identity = _artifact_cache_identity(checkpoint)
    cached = _VERIFIED_ADAPTER_CACHE.get(str(checkpoint))
    if cached is not None and cached[0] == identity:
        return cached[1]
    try:
        receipt = verify_checkpoint_fidelity(checkpoint)
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(
            f"{purpose} lacks valid checkpoint-fidelity evidence: {checkpoint}: {exc}"
        ) from exc
    publication = receipt.get("publication_invariant")
    if not isinstance(publication, dict) or publication.get("status") != "exact":
        raise PublicationFidelityError(
            f"{purpose} receipt is not an exact Qwen publication receipt: {checkpoint}"
        )
    _VERIFIED_ADAPTER_CACHE[str(checkpoint)] = (identity, receipt)
    return receipt


def _conversion_cache_identity(checkpoint: Path) -> tuple[tuple[int, ...], ...]:
    identities = []
    for name in (
        LEGACY_CONVERSION_RECEIPT_FILENAME,
        ADAPTER_CONFIG_FILENAME,
        "adapter_model.safetensors",
    ):
        path = checkpoint / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise PublicationFidelityError(
                f"cannot stat development conversion file {path}: {exc}"
            ) from exc
        identities.append(
            (
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
                int(stat.S_IMODE(metadata.st_mode)),
            )
        )
    return tuple(identities)


def verify_development_conversion_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    purpose: str,
) -> dict[str, Any]:
    """Reopen the immutable legacy saved-policy conversion evidence."""

    checkpoint = Path(checkpoint_dir).expanduser().absolute()
    identity = _conversion_cache_identity(checkpoint)
    cached = _VERIFIED_CONVERSION_CACHE.get(str(checkpoint))
    if cached is not None and cached[0] == identity:
        return cached[1]
    receipt_path = checkpoint / LEGACY_CONVERSION_RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path)
    if stat.S_IMODE(receipt_path.stat().st_mode) & 0o222:
        raise PublicationFidelityError(f"{purpose} conversion receipt must be read-only")
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed_hash != _sha256_json(unhashed):
        raise PublicationFidelityError(f"{purpose} conversion receipt self-hash mismatch")
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
        raise PublicationFidelityError(
            f"{purpose} conversion claim boundary mismatch: {mismatches}"
        )
    converted = receipt.get("converted_output")
    if not isinstance(converted, dict):
        raise PublicationFidelityError(f"{purpose} conversion lacks output evidence")
    if Path(str(converted.get("path", ""))).expanduser().absolute() != checkpoint:
        raise PublicationFidelityError(f"{purpose} conversion output path mismatch")
    config = _load_json_object(checkpoint / ADAPTER_CONFIG_FILENAME)
    if config.get("bias") != "all" or converted.get("serialization_bias_policy") != "all":
        raise PublicationFidelityError(f"{purpose} conversion is not serialized with bias='all'")
    live_tree = _tree_manifest(
        checkpoint,
        excluded_names=frozenset({LEGACY_CONVERSION_RECEIPT_FILENAME}),
    )
    compact_tree = {
        "file_count": live_tree["file_count"],
        "files": live_tree["files"],
        "tree_sha256": live_tree["tree_sha256"],
    }
    if converted.get("manifest_excluding_receipt") != compact_tree:
        raise PublicationFidelityError(f"{purpose} conversion bytes differ from receipt")
    try:
        inspection = inspect_checkpoint_adapter(checkpoint)
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(
            f"{purpose} conversion adapter inspection failed: {exc}"
        ) from exc
    if converted.get("checkpoint_fidelity_adapter_inspection") != inspection:
        raise PublicationFidelityError(f"{purpose} conversion inspection differs from receipt")
    _VERIFIED_CONVERSION_CACHE[str(checkpoint)] = (identity, receipt)
    return receipt


def verify_development_relocation_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    purpose: str,
) -> dict[str, Any]:
    """Reopen a dev-only, independently proven adapter-base relocation."""

    checkpoint = Path(checkpoint_dir).expanduser().absolute()
    receipt_path = checkpoint / DEVELOPMENT_RELOCATION_RECEIPT_FILENAME
    receipt = _load_json_object(receipt_path)
    if stat.S_IMODE(receipt_path.stat().st_mode) & 0o222:
        raise PublicationFidelityError(f"{purpose} relocation receipt must be read-only")
    claimed_hash = receipt.get("receipt_sha256")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if claimed_hash != _sha256_json(unhashed):
        raise PublicationFidelityError(f"{purpose} relocation receipt self-hash mismatch")
    required = {
        "schema_version": 1,
        "protocol": DEVELOPMENT_RELOCATION_PROTOCOL,
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
    mismatches = {
        key: {"actual": receipt.get(key), "expected": expected}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise PublicationFidelityError(
            f"{purpose} relocation claim boundary mismatch: {mismatches}"
        )
    verifier = receipt.get("verifier")
    implementation_path = (
        Path(__file__).resolve(strict=True).parents[3]
        / "scripts"
        / "relocate_development_adapter_base.py"
    ).resolve(strict=True)
    if not isinstance(verifier, dict) or verifier != {
        "protocol": DEVELOPMENT_RELOCATION_VERIFIER_PROTOCOL,
        "implementation": "scripts.relocate_development_adapter_base",
        "implementation_sha256": _sha256_file(implementation_path),
    }:
        raise PublicationFidelityError(f"{purpose} relocation verifier identity mismatch")
    upstream = receipt.get("upstream_conversion")
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
    if not isinstance(upstream, dict) or set(upstream) != upstream_keys:
        raise PublicationFidelityError(f"{purpose} relocation upstream schema mismatch")
    source_adapter = Path(str(upstream["source_adapter_path"])).expanduser().absolute()
    upstream_receipt_path = (
        Path(str(upstream["authoritative_receipt_path"])).expanduser().absolute()
    )
    if upstream_receipt_path != source_adapter / LEGACY_CONVERSION_RECEIPT_FILENAME:
        raise PublicationFidelityError(f"{purpose} relocation upstream receipt path mismatch")
    verified_upstream = verify_development_conversion_checkpoint(
        source_adapter,
        purpose=f"{purpose} relocation upstream",
    )
    if (
        upstream["authoritative_receipt_protocol"] != LEGACY_CONVERSION_PROTOCOL
        or upstream["authoritative_receipt_sha256"] != verified_upstream.get("receipt_sha256")
        or upstream["authoritative_receipt_file_sha256"] != _sha256_file(upstream_receipt_path)
        or upstream["adapter_config_sha256"]
        != _sha256_file(source_adapter / ADAPTER_CONFIG_FILENAME)
        or upstream["adapter_model_sha256"]
        != _sha256_file(source_adapter / "adapter_model.safetensors")
        or upstream["source_adapter_manifest"] != _tree_manifest(source_adapter)
    ):
        raise PublicationFidelityError(f"{purpose} relocation upstream bytes mismatch")
    recorded_base = verified_upstream.get("base")
    recorded_manifest = recorded_base.get("manifest") if isinstance(recorded_base, dict) else None
    if isinstance(recorded_manifest, dict) and isinstance(recorded_manifest.get("files"), list):
        normalized_recorded_manifest = {
            **recorded_manifest,
            "size_bytes": sum(
                int(record.get("size_bytes", 0))
                for record in recorded_manifest["files"]
                if isinstance(record, dict)
            ),
        }
    else:
        normalized_recorded_manifest = None
    if (
        not isinstance(recorded_base, dict)
        or upstream["configured_base_path"] != recorded_base.get("path")
        or upstream["recorded_base_manifest"] != normalized_recorded_manifest
        or upstream["live_base_manifest"] != upstream["recorded_base_manifest"]
    ):
        raise PublicationFidelityError(f"{purpose} relocation upstream base evidence mismatch")
    relocated = receipt.get("relocated_output")
    frozen_base = receipt.get("frozen_base")
    if not isinstance(relocated, dict) or not isinstance(frozen_base, dict):
        raise PublicationFidelityError(f"{purpose} relocation lacks output/base evidence")
    if Path(str(relocated.get("path", ""))).expanduser().absolute() != checkpoint:
        raise PublicationFidelityError(f"{purpose} relocation output path mismatch")
    config = _load_json_object(checkpoint / ADAPTER_CONFIG_FILENAME)
    configured_base = Path(str(config.get("base_model_name_or_path", ""))).expanduser()
    if not configured_base.is_absolute():
        raise PublicationFidelityError(f"{purpose} relocated base path is not absolute")
    configured_base = configured_base.absolute()
    frozen_base_path = Path(str(frozen_base.get("path", ""))).expanduser().absolute()
    if configured_base != frozen_base_path or relocated.get("configured_base_path") != str(
        frozen_base_path
    ):
        raise PublicationFidelityError(f"{purpose} relocated config/frozen-base path mismatch")
    if config.get("bias") != "all" or relocated.get("serialization_bias_policy") != "all":
        raise PublicationFidelityError(f"{purpose} relocation is not serialized with bias='all'")
    live_output_tree = _tree_manifest(
        checkpoint,
        excluded_names=frozenset({DEVELOPMENT_RELOCATION_RECEIPT_FILENAME}),
    )
    if relocated.get("manifest_excluding_receipt") != live_output_tree:
        raise PublicationFidelityError(f"{purpose} relocated adapter bytes differ from receipt")
    if relocated.get("adapter_config_sha256") != _sha256_file(
        checkpoint / ADAPTER_CONFIG_FILENAME
    ) or relocated.get("adapter_model_sha256") != _sha256_file(
        checkpoint / "adapter_model.safetensors"
    ):
        raise PublicationFidelityError(f"{purpose} relocated config/weights hash mismatch")
    if (
        frozen_base.get("manifest") != _tree_manifest(frozen_base_path)
        or frozen_base.get("manifest") != upstream["live_base_manifest"]
    ):
        raise PublicationFidelityError(f"{purpose} frozen base bytes differ from receipt")
    if frozen_base.get("config_sha256") != _sha256_file(frozen_base_path / "config.json"):
        raise PublicationFidelityError(f"{purpose} frozen base config hash mismatch")
    try:
        inspection = inspect_checkpoint_adapter(checkpoint)
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(
            f"{purpose} relocated adapter inspection failed: {exc}"
        ) from exc
    if relocated.get("checkpoint_fidelity_adapter_inspection") != inspection:
        raise PublicationFidelityError(
            f"{purpose} relocated adapter inspection differs from receipt"
        )
    verification = receipt.get("verification")
    if not isinstance(verification, dict):
        raise PublicationFidelityError(f"{purpose} relocation lacks live verification evidence")
    required_true = {
        "upstream_receipt_verified",
        "path_independent_base_manifests_exact",
        "source_and_output_stable",
        "independent_model_loads",
        "full_model_state_exact",
        "functional_equivalence_exact",
    }
    if any(verification.get(field) is not True for field in required_true):
        raise PublicationFidelityError(f"{purpose} relocation verification is incomplete")
    try:
        strictly_verified = verify_development_relocation_authority(
            checkpoint,
            authoritative_receipt=receipt,
        )
    except ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(
            f"{purpose} strict relocation authority failed: {exc}"
        ) from exc
    if strictly_verified != receipt:
        raise PublicationFidelityError(
            f"{purpose} strict relocation verifier returned different evidence"
        )
    return receipt


def verify_publication_input_adapter(
    model_path: str | os.PathLike[str],
    *,
    purpose: str,
) -> dict[str, Any] | None:
    """Verify an adapter initialization before model loading in publication mode."""

    if not checkpoint_fidelity_enabled():
        return None
    candidate = Path(model_path).expanduser().absolute()
    if not (candidate / ADAPTER_CONFIG_FILENAME).is_file():
        return None
    if publication_fidelity_enabled():
        return verify_publication_adapter_checkpoint(candidate, purpose=purpose)
    if (candidate / RECEIPT_FILENAME).is_file():
        try:
            return verify_checkpoint_fidelity(candidate)
        except CheckpointFidelityError as exc:
            raise PublicationFidelityError(
                f"{purpose} lacks valid development checkpoint fidelity: {candidate}: {exc}"
            ) from exc
    if (candidate / DEVELOPMENT_RELOCATION_RECEIPT_FILENAME).is_file():
        return verify_development_relocation_checkpoint(candidate, purpose=purpose)
    return verify_development_conversion_checkpoint(candidate, purpose=purpose)


def _unwrap_trainer_model(trainer: Any) -> Any:
    model = getattr(trainer, "model", None)
    if model is None:
        raise PublicationFidelityError("trainer has no model for publication fidelity")
    accelerator = getattr(trainer, "accelerator", None)
    unwrap = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap):
        try:
            model = unwrap(model)
        except Exception as exc:
            raise PublicationFidelityError(
                f"cannot unwrap publication policy model: {exc}"
            ) from exc
    return model


def _distributed_digest_agreement(local_digest: str, *, phase: str) -> dict[str, Any]:
    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch is a training dependency
        raise PublicationFidelityError("torch.distributed is required for training") from exc

    raw_world_size = str(os.environ.get("WORLD_SIZE", "1") or "1").strip()
    try:
        environment_world_size = int(raw_world_size)
    except ValueError as exc:
        raise PublicationFidelityError(
            f"WORLD_SIZE must be an integer, got {raw_world_size!r}"
        ) from exc
    if environment_world_size <= 0:
        raise PublicationFidelityError("WORLD_SIZE must be positive")

    if not dist.is_available() or not dist.is_initialized():
        if environment_world_size != 1:
            raise PublicationFidelityError(
                "publication fidelity requires initialized torch.distributed when WORLD_SIZE > 1"
            )
        return {
            "phase": phase,
            "world_size": 1,
            "rank_digests": [{"rank": 0, "digest": local_digest}],
            "consensus_digest": local_digest,
        }

    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    if world_size != environment_world_size:
        raise PublicationFidelityError(
            "distributed world size disagrees with environment: "
            f"torch={world_size}, WORLD_SIZE={environment_world_size}"
        )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, {"rank": rank, "digest": local_digest})
    expected_ranks = list(range(world_size))
    observed_ranks = sorted(
        int(value.get("rank", -1)) if isinstance(value, dict) else -1 for value in gathered
    )
    digests = {
        str(value.get("digest", "")) if isinstance(value, dict) else "" for value in gathered
    }
    if observed_ranks != expected_ranks or digests != {local_digest}:
        raise PublicationFidelityError(
            f"publication policy differs across ranks during {phase}: {gathered}"
        )
    return {
        "phase": phase,
        "world_size": world_size,
        "rank_digests": sorted(gathered, key=lambda value: int(value["rank"])),
        "consensus_digest": local_digest,
    }


def _distributed_policy_mismatch_details(closure: Mapping[str, Any]) -> dict[str, Any]:
    """Collect tensor hashes only after every rank has rejected digest consensus."""

    import torch.distributed as dist

    world_size = int(dist.get_world_size())
    local = {
        "rank": int(dist.get_rank()),
        "tensor_inventory_sha256": closure["tensor_inventory_sha256"],
        "tensors": closure["tensors"],
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    inventories = sorted(gathered, key=lambda item: int(item["rank"]))
    by_rank = [
        (inventory["rank"], {record["key"]: record for record in inventory["tensors"]})
        for inventory in inventories
    ]
    keys = sorted({key for _, tensors in by_rank for key in tensors})
    differing = []
    for key in keys:
        records = [{"rank": rank, "tensor": tensors.get(key)} for rank, tensors in by_rank]
        if len({_sha256_json(record["tensor"]) for record in records}) > 1:
            differing.append({"key": key, "rank_records": records})
    return {
        "world_size": world_size,
        "differing_tensor_count": len(differing),
        "differing_tensors": differing,
        "rank_tensor_inventories": inventories,
    }


def _audit_trainable_gradient_inventory(trainer: Any) -> dict[str, Any]:
    """Hash gradients without modifying them or the optimizer's parameter state."""

    records = []
    try:
        for raw_name, parameter in _unwrap_trainer_model(trainer).named_parameters():
            if not parameter.requires_grad:
                continue
            key = canonicalize_peft_adapter_key(raw_name, "default")
            if parameter.grad is None:
                record = {
                    "key": key, "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype), "numel": int(parameter.numel()),
                    "value_sha256": None, "gradient_present": False,
                }
            else:
                record = {**_tensor_record(key, parameter.grad), "gradient_present": True}
            records.append(record)
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(f"cannot audit trainable gradients: {exc}") from exc
    records.sort(key=lambda record: record["key"])
    if not records or len({record["key"] for record in records}) != len(records):
        raise PublicationFidelityError("trainable gradient inventory is empty or has duplicate keys")
    return {"tensor_inventory_sha256": _sha256_json(records), "tensors": records}


def _local_cuda_memory_evidence(rank: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "rank": rank,
            "cuda_available": False,
            "device_index": None,
            "physical_device_index": None,
            "visible_device_token": None,
            "device_uuid": None,
            "device_name": None,
            "driver_version": None,
            "cuda_runtime_version": None,
            "total_memory_bytes": None,
            "nvidia_smi_total_memory_mib": None,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    visible_token = visible_tokens[device_index] if device_index < len(visible_tokens) else None
    if visible_token is None:
        raise PublicationFidelityError(
            "CUDA fidelity telemetry requires an explicit CUDA_VISIBLE_DEVICES mapping"
        )
    try:
        query = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PublicationFidelityError(f"cannot bind live NVIDIA device identity: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise PublicationFidelityError("unexpected nvidia-smi identity output")
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_mib": int(fields[3]),
                    "driver": fields[4],
                }
            )
        except ValueError as exc:
            raise PublicationFidelityError("invalid nvidia-smi identity output") from exc
    matches = [row for row in rows if visible_token in {str(row["index"]), str(row["uuid"])}]
    if len(matches) != 1:
        raise PublicationFidelityError(
            f"CUDA visible device does not resolve uniquely in nvidia-smi: {visible_token!r}"
        )
    identity = matches[0]
    if identity["name"] != str(properties.name):
        raise PublicationFidelityError("torch and nvidia-smi CUDA device names differ")
    return {
        "rank": rank,
        "cuda_available": True,
        "device_index": device_index,
        "physical_device_index": identity["index"],
        "visible_device_token": visible_token,
        "device_uuid": identity["uuid"],
        "device_name": str(properties.name),
        "driver_version": identity["driver"],
        "cuda_runtime_version": str(torch.version.cuda),
        "total_memory_bytes": int(properties.total_memory),
        "nvidia_smi_total_memory_mib": identity["memory_mib"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
    }


def _distributed_post_step_agreement(local_digest: str, *, phase: str) -> dict[str, Any]:
    """Gather exact rank consensus plus rank-local in-process CUDA peaks."""

    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover
        raise PublicationFidelityError("torch.distributed is required for fidelity") from exc
    try:
        environment_world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    except ValueError as exc:
        raise PublicationFidelityError("WORLD_SIZE must be an integer") from exc
    active = dist.is_available() and dist.is_initialized()
    if not active:
        if environment_world_size != 1:
            raise PublicationFidelityError(
                "post-step fidelity requires initialized distributed state for WORLD_SIZE > 1"
            )
        gathered = [
            {
                "rank": 0,
                "digest": local_digest,
                "cuda_memory": _local_cuda_memory_evidence(0),
            }
        ]
    else:
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())
        if world_size != environment_world_size:
            raise PublicationFidelityError("post-step distributed world-size mismatch")
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "digest": local_digest,
                "cuda_memory": _local_cuda_memory_evidence(rank),
            },
        )
    ordered = sorted(gathered, key=lambda value: int(value["rank"]))
    world_size = len(ordered)
    if [int(value["rank"]) for value in ordered] != list(range(world_size)) or {
        str(value["digest"]) for value in ordered
    } != {local_digest}:
        raise PublicationFidelityError(f"post-step rank consensus differs: {ordered}")
    rank_digests = [
        {"rank": int(value["rank"]), "digest": str(value["digest"])} for value in ordered
    ]
    rank_cuda_memory = [dict(value["cuda_memory"]) for value in ordered]
    evidence = {"rank_digests": rank_digests, "rank_cuda_memory": rank_cuda_memory}
    return {
        "phase": phase,
        "world_size": world_size,
        "rank_digests": rank_digests,
        "consensus_digest": local_digest,
        "rank_cuda_memory": rank_cuda_memory,
        "distributed_evidence_sha256": _sha256_json(evidence),
    }


def _exact_reference_root(trainer: Any) -> Path:
    raw = getattr(trainer, "_clasp_exact_peft_kl_receipt_path", None)
    if raw is None:
        raise PublicationFidelityError("trainer lacks the exact PEFT KL pre-step receipt path")
    path = Path(str(raw)).expanduser().absolute()
    if path.name != EXACT_PEFT_KL_RECEIPT_FILENAME:
        raise PublicationFidelityError(f"unexpected exact PEFT KL receipt path: {path}")
    return path.parent


def _copy_exact_reference_lineage(
    trainer: Any,
    destination_root: Path,
    *,
    claim_scope: str,
    include_step1: bool,
) -> None:
    source_root = _exact_reference_root(trainer)
    pre_receipt = inspect_runtime_receipt(
        source_root / EXACT_PEFT_KL_RECEIPT_FILENAME,
        claim_scope=claim_scope,
    )
    if destination_root != source_root:
        write_runtime_receipt(destination_root, pre_receipt, claim_scope=claim_scope)
    if include_step1:
        step1 = inspect_post_step_receipt(
            source_root / STEP1_RECEIPT_FILENAME,
            claim_scope=claim_scope,
            expected_global_step=1,
            expected_artifact_role="step1",
        )
        if destination_root != source_root:
            write_post_step_receipt(
                destination_root,
                step1,
                claim_scope=claim_scope,
                filename=STEP1_RECEIPT_FILENAME,
            )


def run_exact_peft_kl_post_step_audit(
    trainer: Any,
    output_dir: str | os.PathLike[str],
    *,
    artifact_role: str,
    global_step: int,
) -> dict[str, Any] | None:
    """Audit the live prepared reference on every rank and persist rank-zero evidence."""

    pre_receipt = getattr(trainer, "_clasp_exact_peft_kl_reference", None)
    if pre_receipt is None:
        return None
    if not checkpoint_fidelity_enabled():
        raise PublicationFidelityError("post-step exact-reference audit requires fidelity mode")
    if artifact_role not in {"step1", "checkpoint", "final"}:
        raise PublicationFidelityError(f"unsupported post-step artifact role: {artifact_role}")
    claim_scope = "development" if development_fidelity_enabled() else "publication"
    reference = getattr(trainer, "ref_model", None)
    expected_reference = getattr(trainer, "_clasp_exact_peft_prepared_reference", None)
    if reference is None or expected_reference is None:
        raise PublicationFidelityError("trainer lacks the tracked prepared exact KL reference")
    accelerator = getattr(trainer, "accelerator", None)
    unwrap = getattr(accelerator, "unwrap_model", None)
    if not callable(unwrap):
        raise PublicationFidelityError("trainer accelerator cannot unwrap post-step reference")
    try:
        policy = _unwrap_trainer_model(trainer)
        unwrapped_reference = unwrap(reference)
    except Exception as exc:
        raise PublicationFidelityError(f"cannot unwrap post-step model pair: {exc}") from exc
    expected_unwrapped = getattr(trainer, "_clasp_exact_peft_unwrapped_reference", None)
    runtime_identity = {
        "trainer_ref_model_identity": reference is expected_reference,
        "reference_prepared_object_identity": reference is expected_reference,
        "reference_unwrapped_identity": unwrapped_reference is expected_unwrapped,
        "prepared_class": f"{reference.__class__.__module__}.{reference.__class__.__qualname__}",
        "unwrapped_class": (
            f"{unwrapped_reference.__class__.__module__}."
            f"{unwrapped_reference.__class__.__qualname__}"
        ),
    }
    try:
        closure = audit_trainable_state_closure(policy)
        assert_qwen25_7b_publication_invariant(policy, closure)
        reference_audit, policy_context = audit_live_post_step_reference(
            policy,
            unwrapped_reference,
            pre_step_receipt=pre_receipt,
            trainable_closure=closure,
            optimizer=getattr(trainer, "optimizer", None),
            runtime_identity=runtime_identity,
            require_policy_update=(
                artifact_role == "final" or (artifact_role == "checkpoint" and global_step > 1)
            ),
        )
        verify_source_binding_unchanged(pre_receipt["source_binding"])
    except (CheckpointFidelityError, ExactPeftKLReferenceError, ExactPeftKLPostStepError) as exc:
        raise PublicationFidelityError(f"post-step exact KL live audit failed: {exc}") from exc
    root = _exact_reference_root(trainer)
    pre_path = root / EXACT_PEFT_KL_RECEIPT_FILENAME
    pre_identity = pre_step_reference_identity(pre_path, claim_scope=claim_scope)
    training_contract = dict(pre_receipt["training_contract"])
    run_contract = _training_run_contract_identity(root)
    expected_run_contract = getattr(trainer, "_clasp_training_run_contract_identity", None)
    if run_contract != expected_run_contract:
        raise PublicationFidelityError("training run contract changed after step zero")
    step1_identity = None
    if artifact_role != "step1":
        step1_identity = step1_reference_binding(
            root / STEP1_RECEIPT_FILENAME,
            claim_scope=claim_scope,
        )
    digest_fields = {
        "claim_scope": "development_only" if claim_scope == "development" else "publication",
        "artifact_role": artifact_role,
        "global_step": global_step,
        "pre_step_reference": pre_identity,
        "step1_reference": step1_identity,
        "training_contract": training_contract,
        "training_run_contract": run_contract,
        "reference_audit": reference_audit,
        "policy_context": policy_context,
        "source_binding_unchanged": True,
    }
    local_digest = _sha256_json(digest_fields)
    distributed = _distributed_post_step_agreement(
        local_digest,
        phase=f"exact_peft_kl_{artifact_role}_{global_step}",
    )
    try:
        receipt = build_post_step_receipt(
            artifact_role=artifact_role,
            global_step=global_step,
            pre_step_reference=pre_identity,
            step1_reference=step1_identity,
            training_contract=training_contract,
            training_run_contract=run_contract,
            reference_audit=reference_audit,
            policy_context=policy_context,
            distributed=distributed,
            claim_scope=claim_scope,
        )
    except ExactPeftKLPostStepError as exc:
        raise PublicationFidelityError(f"cannot build post-step exact KL receipt: {exc}") from exc

    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover
        raise PublicationFidelityError("torch.distributed is required for training") from exc
    distributed_active = dist.is_available() and dist.is_initialized()
    rank = int(dist.get_rank()) if distributed_active else 0
    result: list[object] = [None]
    target = root if artifact_role == "step1" else Path(output_dir).expanduser().absolute()
    if rank == 0:
        try:
            _copy_exact_reference_lineage(
                trainer,
                target,
                claim_scope=claim_scope,
                include_step1=artifact_role != "step1",
            )
            filename = (
                STEP1_RECEIPT_FILENAME if artifact_role == "step1" else POST_STEP_RECEIPT_FILENAME
            )
            written = write_post_step_receipt(
                target,
                receipt,
                claim_scope=claim_scope,
                filename=filename,
            )
            result[0] = {"ok": True, "receipt": written}
        except Exception as exc:
            result[0] = {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
    if distributed_active:
        dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise PublicationFidelityError(f"post-step exact KL receipt write failed: {payload}")
    written = payload.get("receipt")
    if not isinstance(written, dict) or written != receipt:
        raise PublicationFidelityError("rank zero returned mismatched post-step evidence")
    if artifact_role == "step1":
        setattr(trainer, "_clasp_exact_peft_kl_step1_receipt", written)
    else:
        setattr(trainer, "_clasp_exact_peft_kl_latest_post_step_receipt", written)
    return written


def run_publication_fidelity_preflight(
    trainer: Any,
    *,
    phase: str,
    adapter_name: str = "default",
) -> dict[str, Any] | None:
    """Audit every rank's live trainable policy and require digest consensus."""

    if not checkpoint_fidelity_enabled():
        return None
    model = _unwrap_trainer_model(trainer)
    try:
        closure = audit_trainable_state_closure(model, adapter_name)
        qwen_structure = assert_qwen25_7b_publication_invariant(
            model,
            closure,
            adapter_name,
        )
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(
            f"publication trainable-policy preflight failed during {phase}: {exc}"
        ) from exc
    local_digest = _sha256_json(
        {
            "closure": closure,
            "qwen_structural_invariant": qwen_structure,
            "claim_scope": (
                "publication" if publication_fidelity_enabled() else "development_only"
            ),
        }
    )
    try:
        distributed = _distributed_digest_agreement(local_digest, phase=phase)
    except PublicationFidelityError as exc:
        if _strict_flag(POLICY_CONSISTENCY_CHECKS_ENV, False) and str(exc).startswith(
            "publication policy differs across ranks during "
        ):
            # The digest comparison fails identically on all ranks, so each
            # participates in this extra diagnostic collective before raising.
            exc.policy_mismatch_details = _distributed_policy_mismatch_details(closure)
        raise
    report = {
        "closure": closure,
        "qwen_structural_invariant": qwen_structure,
        "publication_invariant": (qwen_structure if publication_fidelity_enabled() else None),
        "development_only": development_fidelity_enabled(),
        "distributed": distributed,
    }
    setattr(trainer, "_clasp_publication_fidelity", report)
    return report


def verify_publication_resume_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_kl_reference: Mapping[str, Any] | None = None,
    trainer: Any | None = None,
    expect_live_policy_matches_checkpoint: bool = False,
) -> dict[str, Any] | None:
    """Verify a resume artifact locally before Trainer creates/loads optimizer state."""

    if not checkpoint_fidelity_enabled():
        return None
    if publication_fidelity_enabled():
        fidelity = verify_publication_adapter_checkpoint(
            checkpoint_dir,
            purpose="publication resume checkpoint",
        )
    else:
        try:
            fidelity = verify_checkpoint_fidelity(checkpoint_dir)
        except CheckpointFidelityError as exc:
            raise PublicationFidelityError(
                "development resume checkpoint lacks valid fidelity evidence: "
                f"{checkpoint_dir}: {exc}"
            ) from exc
    checkpoint_root = Path(checkpoint_dir).expanduser().absolute()
    exact_sidecars = (
        EXACT_PEFT_KL_RECEIPT_FILENAME,
        STEP1_RECEIPT_FILENAME,
        POST_STEP_RECEIPT_FILENAME,
    )
    if expected_kl_reference is None:
        stale = [name for name in exact_sidecars if (checkpoint_root / name).exists()]
        if stale:
            raise PublicationFidelityError(
                f"beta-zero resume checkpoint contains stale exact-KL sidecars: {stale}"
            )
    else:
        claim_scope = "development" if development_fidelity_enabled() else "publication"
        sidecar = checkpoint_root / EXACT_PEFT_KL_RECEIPT_FILENAME
        try:
            observed = inspect_runtime_receipt(sidecar, claim_scope=claim_scope)
        except ExactPeftKLReferenceError as exc:
            raise PublicationFidelityError(
                f"resume checkpoint lacks valid exact PEFT KL lineage: {sidecar}: {exc}"
            ) from exc
        if observed != dict(expected_kl_reference):
            raise PublicationFidelityError(
                f"resume checkpoint exact PEFT KL lineage differs from run root: {sidecar}"
            )
        try:
            step = int(checkpoint_root.name.removeprefix("checkpoint-"))
        except ValueError as exc:
            raise PublicationFidelityError(
                f"cannot derive exact resume global step from {checkpoint_root.name!r}"
            ) from exc
        closure_digest = fidelity["trainable_closure"]["tensor_inventory_sha256"]
        try:
            inspect_post_step_receipt(
                checkpoint_root / STEP1_RECEIPT_FILENAME,
                claim_scope=claim_scope,
                expected_global_step=1,
                expected_artifact_role="step1",
                rehash_policy_artifact=False,
            )
            post = inspect_post_step_receipt(
                checkpoint_root / POST_STEP_RECEIPT_FILENAME,
                claim_scope=claim_scope,
                expected_global_step=step,
                expected_artifact_role="checkpoint",
                expected_policy_trainable_tensor_inventory_sha256=closure_digest,
            )
        except ExactPeftKLPostStepError as exc:
            raise PublicationFidelityError(
                f"resume checkpoint lacks exact post-step reference evidence: {exc}"
            ) from exc
        expected_pre_identity = pre_step_reference_identity(sidecar, claim_scope=claim_scope)
        if post["pre_step_reference"] != expected_pre_identity:
            raise PublicationFidelityError(
                "resume post-step evidence has different pre-step lineage"
            )
        if trainer is None:
            raise PublicationFidelityError(
                "exact-KL resume verification requires the live Trainer/reference objects"
            )
        reference = getattr(trainer, "ref_model", None)
        expected_reference = getattr(trainer, "_clasp_exact_peft_prepared_reference", None)
        accelerator = getattr(trainer, "accelerator", None)
        unwrap = getattr(accelerator, "unwrap_model", None)
        if reference is None or expected_reference is None or not callable(unwrap):
            raise PublicationFidelityError("Trainer lacks tracked exact-KL reference state")
        policy = _unwrap_trainer_model(trainer)
        unwrapped_reference = unwrap(reference)
        expected_unwrapped = getattr(trainer, "_clasp_exact_peft_unwrapped_reference", None)
        try:
            closure = audit_trainable_state_closure(policy)
            assert_qwen25_7b_publication_invariant(policy, closure)
            optimizer = getattr(trainer, "optimizer", None)
            if not expect_live_policy_matches_checkpoint:
                optimizer = SimpleNamespace(
                    param_groups=[
                        {
                            "params": [
                                parameter
                                for parameter in policy.parameters()
                                if parameter.requires_grad
                            ]
                        }
                    ]
                )
            live_reference, live_policy = audit_live_post_step_reference(
                policy,
                unwrapped_reference,
                pre_step_receipt=observed,
                trainable_closure=closure,
                optimizer=optimizer,
                runtime_identity={
                    "trainer_ref_model_identity": reference is expected_reference,
                    "reference_prepared_object_identity": reference is expected_reference,
                    "reference_unwrapped_identity": unwrapped_reference is expected_unwrapped,
                    "prepared_class": (
                        f"{reference.__class__.__module__}.{reference.__class__.__qualname__}"
                    ),
                    "unwrapped_class": (
                        f"{unwrapped_reference.__class__.__module__}."
                        f"{unwrapped_reference.__class__.__qualname__}"
                    ),
                },
                require_policy_update=(expect_live_policy_matches_checkpoint and step > 1),
            )
            verify_source_binding_unchanged(observed["source_binding"])
        except (
            CheckpointFidelityError,
            ExactPeftKLReferenceError,
            ExactPeftKLPostStepError,
        ) as exc:
            raise PublicationFidelityError(
                f"live exact-KL reference resume re-audit failed: {exc}"
            ) from exc
        if live_reference != post["reference_audit"]:
            raise PublicationFidelityError(
                "live resumed reference identity/state differs from saved post-step audit"
            )
        if expect_live_policy_matches_checkpoint:
            if closure["tensor_inventory_sha256"] != closure_digest:
                raise PublicationFidelityError(
                    "live retained policy closure differs from checkpoint fidelity"
                )
            if live_policy != post["policy_context"]:
                raise PublicationFidelityError(
                    "live retained policy differs from checkpoint post-step evidence"
                )
        else:
            setattr(trainer, "_clasp_pending_exact_kl_resume_checkpoint", str(checkpoint_root))
            setattr(trainer, "_clasp_pending_exact_kl_resume_post_receipt", post)
            setattr(trainer, "_clasp_pending_exact_kl_resume_closure_digest", closure_digest)
        _distributed_digest_agreement(
            _sha256_json(
                {
                    "phase": "exact_peft_kl_resume_live_reaudit",
                    "checkpoint_step": step,
                    "reference_audit": live_reference,
                    "policy_context": live_policy
                    if expect_live_policy_matches_checkpoint
                    else None,
                    "post_receipt_sha256": post["receipt_sha256"],
                }
            ),
            phase="exact_peft_kl_resume_live_reaudit",
        )
    return fidelity


def reject_beta_zero_exact_kl_sidecars(*roots: str | os.PathLike[str]) -> None:
    """Reject stale beta-positive lineage in a beta-zero fidelity run."""

    if not checkpoint_fidelity_enabled():
        return
    names = (
        EXACT_PEFT_KL_RECEIPT_FILENAME,
        STEP1_RECEIPT_FILENAME,
        POST_STEP_RECEIPT_FILENAME,
    )
    stale = [
        str(Path(root).expanduser().absolute() / name)
        for root in roots
        for name in names
        if (Path(root).expanduser().absolute() / name).exists()
    ]
    if stale:
        raise PublicationFidelityError(
            f"beta-zero fidelity run contains stale exact-KL sidecars: {stale}"
        )


def verify_loaded_exact_kl_resume_state(trainer: Any) -> dict[str, Any] | None:
    """Re-audit the restored policy, actual optimizer, and reference before step advance."""

    raw = getattr(trainer, "_clasp_pending_exact_kl_resume_checkpoint", None)
    if raw is None:
        return None
    checkpoint = Path(str(raw)).expanduser().absolute()
    try:
        checkpoint_step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise PublicationFidelityError(
            "pending exact-KL resume checkpoint has invalid name"
        ) from exc
    observed_step = int(getattr(getattr(trainer, "state", None), "global_step", -1))
    if observed_step != checkpoint_step:
        raise PublicationFidelityError(
            "Trainer callback ran before the checkpoint state was restored: "
            f"trainer={observed_step}, checkpoint={checkpoint_step}"
        )
    expected = getattr(trainer, "_clasp_exact_peft_kl_reference", None)
    if not isinstance(expected, Mapping):
        raise PublicationFidelityError("pending exact-KL resume lacks pre-step lineage")
    result = verify_publication_resume_checkpoint(
        checkpoint,
        expected_kl_reference=expected,
        trainer=trainer,
        expect_live_policy_matches_checkpoint=True,
    )
    delattr(trainer, "_clasp_pending_exact_kl_resume_checkpoint")
    delattr(trainer, "_clasp_pending_exact_kl_resume_post_receipt")
    delattr(trainer, "_clasp_pending_exact_kl_resume_closure_digest")
    return result


def _write_live_receipt(
    trainer: Any,
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_preflight: Mapping[str, Any] | None = None,
    expected_post_step: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = _unwrap_trainer_model(trainer)
    try:
        _propagate_exact_peft_kl_receipt(trainer, checkpoint_dir)
        closure = audit_trainable_state_closure(model)
        qwen_structure = assert_qwen25_7b_publication_invariant(model, closure)
        if expected_post_step is not None:
            post_digest = expected_post_step.get("policy_context", {}).get(
                "policy_trainable_tensor_inventory_sha256"
            )
            if post_digest != closure.get("tensor_inventory_sha256"):
                raise CheckpointFidelityError(
                    "post-step reference audit is not bound to the saved trainable policy"
                )
        if expected_preflight is not None:
            if (
                expected_preflight.get("closure") != closure
                or expected_preflight.get("qwen_structural_invariant") != qwen_structure
            ):
                raise CheckpointFidelityError(
                    "rank-zero live policy changed after distributed final-save consensus"
                )
        publication = qwen_structure if publication_fidelity_enabled() else None
        return write_checkpoint_fidelity_receipt(
            checkpoint_dir,
            closure,
            publication,
        )
    except CheckpointFidelityError as exc:
        raise PublicationFidelityError(
            f"saved checkpoint does not preserve the live trainable policy: {checkpoint_dir}: {exc}"
        ) from exc


def write_final_publication_fidelity(
    trainer: Any,
    output_dir: str | os.PathLike[str],
    *,
    expected_preflight: Mapping[str, Any] | None = None,
    expected_post_step: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bind the rank-zero final save to the still-live terminal policy."""

    if not checkpoint_fidelity_enabled():
        return None
    return _write_live_receipt(
        trainer,
        output_dir,
        expected_preflight=expected_preflight,
        expected_post_step=expected_post_step,
    )


class PublicationFidelityCheckpointCallback(TrainerCallback):
    """Write checkpoint evidence and optionally reject early rank divergence."""

    def __init__(self, trainer: Any):
        self._trainer = trainer
        self._policy_consistency_settings = _policy_consistency_check_settings()
        self._policy_consistency_records: list[dict[str, Any]] = []
        self._gradient_consistency_records: list[dict[str, Any]] = []

    def _record_policy_consistency(
        self, args, state, *, failed_step=None, error=None, gradients=False
    ):
        records = self._gradient_consistency_records if gradients else self._policy_consistency_records
        filename = (
            GRADIENT_CONSISTENCY_RECEIPT_FILENAME if gradients else POLICY_CONSISTENCY_RECEIPT_FILENAME
        )
        protocol = (
            "clasp-pre-optimizer-exact-gradient-consistency-v1" if gradients
            else "clasp-optimizer-step-exact-policy-consistency-v1"
        )
        steps_key = "verified_gradient_optimizer_steps" if gradients else "verified_optimizer_steps"
        receipt = {
            "schema_version": 1,
            "protocol": protocol,
            "status": "failed" if failed_step is not None else "passed",
            "claim_scope": checkpoint_fidelity_mode(),
            "settings": self._policy_consistency_settings,
            steps_key: [record["optimizer_step"] for record in records],
            "checks": list(records),
        }
        if failed_step is not None:
            receipt["failed_optimizer_step"] = int(failed_step)
            receipt["error"] = str(error)
            details = getattr(error, "policy_mismatch_details", None)
            if details is not None:
                receipt["policy_mismatch_details"] = details
        receipt["receipt_sha256"] = _sha256_json(receipt)
        attribute = "_clasp_gradient_consistency_checks" if gradients else "_clasp_policy_consistency_checks"
        setattr(self._trainer, attribute, receipt)
        world_zero = bool(
            getattr(state, "is_world_process_zero", getattr(args, "should_save", False))
        )
        if not world_zero:
            return
        root = Path(str(args.output_dir))
        root.mkdir(parents=True, exist_ok=True)
        # An interrupted write must preserve the previous successful checks.
        descriptor, temporary_name = tempfile.mkstemp(
            dir=root, prefix=f".{filename}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_name, root / filename)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _check_optimizer_step_policy_consistency(self, args, state):
        settings = self._policy_consistency_settings
        if not settings["enabled"]:
            return
        step = int(getattr(state, "global_step", -1))
        if step <= 0:
            raise PublicationFidelityError("cannot identify completed optimizer step")
        if any(record["optimizer_step"] == step for record in self._policy_consistency_records):
            return
        interval = settings["interval"]
        if step not in settings["initial_optimizer_steps"] and not (
            interval > 0 and step % interval == 0
        ):
            return
        try:
            # Reuse the complete existing closure/invariant/hash comparison.
            # This runs on every rank, including beta-zero GRPO runs.
            report = run_publication_fidelity_preflight(
                self._trainer, phase=f"optimizer_step_{step}_policy_consistency"
            )
            if report is None:
                raise PublicationFidelityError("optimizer-step fidelity unexpectedly disabled")
        except PublicationFidelityError as exc:
            self._record_policy_consistency(args, state, failed_step=step, error=exc)
            raise
        self._policy_consistency_records.append({
            "optimizer_step": step,
            "tensor_inventory_sha256": report["closure"]["tensor_inventory_sha256"],
            "distributed": report["distributed"],
        })
        self._record_policy_consistency(args, state)

    def on_train_begin(self, args, state, control, **kwargs):
        if checkpoint_fidelity_enabled():
            verify_loaded_exact_kl_resume_state(self._trainer)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        settings = self._policy_consistency_settings
        if not (settings["gradient_enabled"] and checkpoint_fidelity_enabled()):
            return control
        # Trainer increments global_step after optimizer.step; these gradients
        # belong to the next step and have already been unscaled/clipped.
        step = int(getattr(state, "global_step", -1)) + 1
        if step <= 0:
            raise PublicationFidelityError("cannot identify upcoming gradient optimizer step")
        if any(record["optimizer_step"] == step for record in self._gradient_consistency_records):
            return control
        interval = settings["interval"]
        if step not in settings["initial_optimizer_steps"] and not (
            interval > 0 and step % interval == 0
        ):
            return control
        phase = f"before_optimizer_step_{step}_gradient_consistency"
        try:
            inventory = _audit_trainable_gradient_inventory(self._trainer)
            try:
                distributed = _distributed_digest_agreement(
                    inventory["tensor_inventory_sha256"], phase=phase
                )
            except PublicationFidelityError as exc:
                if str(exc).startswith("publication policy differs across ranks during "):
                    exc.policy_mismatch_details = _distributed_policy_mismatch_details(inventory)
                raise
        except PublicationFidelityError as exc:
            self._record_policy_consistency(args, state, failed_step=step, error=exc, gradients=True)
            raise
        self._gradient_consistency_records.append({
            "optimizer_step": step,
            "tensor_inventory_sha256": inventory["tensor_inventory_sha256"],
            "gradient_tensor_count": sum(record["gradient_present"] for record in inventory["tensors"]),
            "distributed": distributed,
        })
        self._record_policy_consistency(args, state, gradients=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not checkpoint_fidelity_enabled():
            return control
        self._check_optimizer_step_policy_consistency(args, state)
        if getattr(self._trainer, "_clasp_exact_peft_kl_reference", None) is None:
            return control
        global_step = int(getattr(state, "global_step", -1))
        if global_step == 1:
            run_exact_peft_kl_post_step_audit(
                self._trainer,
                args.output_dir,
                artifact_role="step1",
                global_step=1,
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        if not checkpoint_fidelity_enabled():
            return control
        global_step = int(getattr(state, "global_step", -1))
        if global_step < 0:
            raise PublicationFidelityError("cannot identify saved checkpoint global step")
        checkpoint_dir = Path(str(args.output_dir)) / f"checkpoint-{global_step}"
        post_step = run_exact_peft_kl_post_step_audit(
            self._trainer,
            checkpoint_dir,
            artifact_role="checkpoint",
            global_step=global_step,
        )
        world_zero = bool(
            getattr(state, "is_world_process_zero", getattr(args, "should_save", False))
        )
        if not world_zero:
            return control
        if not checkpoint_dir.is_dir():
            raise PublicationFidelityError(
                f"Trainer on_save fired before checkpoint existed: {checkpoint_dir}"
            )
        _write_live_receipt(
            self._trainer,
            checkpoint_dir,
            expected_post_step=post_step,
        )
        return control


def install_publication_fidelity_callback(trainer: Any) -> bool:
    """Install the post-save callback when publication fidelity is enabled."""

    _policy_consistency_check_settings()
    if not checkpoint_fidelity_enabled():
        return False
    trainer.add_callback(PublicationFidelityCheckpointCallback(trainer))
    return True


def _tree_manifest(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise PublicationFidelityError(f"fixed KL reference is not a regular directory: {root}")
    paths = [root, *sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())]
    writable = []
    files = []
    for path in paths:
        if path.is_symlink():
            raise PublicationFidelityError(f"fixed KL reference contains a symlink: {path}")
        try:
            mode = stat.S_IMODE(path.lstat().st_mode)
        except OSError as exc:
            raise PublicationFidelityError(
                f"cannot stat fixed KL reference entry {path}: {exc}"
            ) from exc
        if mode & 0o222:
            writable.append(str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublicationFidelityError(
                f"fixed KL reference contains a non-regular entry: {path}"
            )
        relative = path.relative_to(root).as_posix()
        if path.name in excluded_names:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if writable:
        raise PublicationFidelityError(
            f"fixed KL reference must be immutable; writable entries={writable[:8]}"
        )
    if not files:
        raise PublicationFidelityError("fixed KL reference contains no files")
    return {
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "tree_sha256": _sha256_json(files),
        "files": files,
    }


def verify_fixed_kl_reference(
    model_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Reconstruct the merge and authorize the active mode's exact v2 receipt."""

    claim_scope = "development" if development_fidelity_enabled() else "publication"
    try:
        return _verify_fixed_kl_reference_live(
            model_path,
            receipt_path,
            claim_scope=claim_scope,
        )
    except FixedKLReferenceError as exc:
        raise PublicationFidelityError(
            f"fixed KL reference live equivalence verification failed: {exc}"
        ) from exc


def resolve_explicit_fixed_kl_reference(
    *,
    required: bool,
) -> tuple[str | None, str | None]:
    """Resolve explicit reference inputs; never infer a foundation checkpoint."""

    model_path = str(os.environ.get(KL_REFERENCE_MODEL_PATH_ENV, "") or "").strip()
    receipt_path = str(os.environ.get(KL_REFERENCE_RECEIPT_ENV, "") or "").strip()
    if not required:
        if model_path or receipt_path:
            raise PublicationFidelityError(
                f"{KL_REFERENCE_MODEL_PATH_ENV}/{KL_REFERENCE_RECEIPT_ENV} were set, "
                "but this run does not require a biased-LoRA fixed KL reference"
            )
        return None, None
    if not model_path or not receipt_path:
        raise PublicationFidelityError(
            "beta > 0 with biased LoRA requires both explicit "
            f"{KL_REFERENCE_MODEL_PATH_ENV} and {KL_REFERENCE_RECEIPT_ENV}; "
            "the adapter's foundation model is not the step-zero SFT policy"
        )
    model = Path(model_path).expanduser().absolute()
    receipt = Path(receipt_path).expanduser().absolute()
    if model.is_symlink() or not model.is_dir():
        raise PublicationFidelityError(f"explicit fixed KL model is not a directory: {model}")
    if receipt.is_symlink() or not receipt.is_file():
        raise PublicationFidelityError(f"explicit fixed KL receipt is not a file: {receipt}")
    return str(model), str(receipt)


def verify_fixed_kl_reference_distributed(
    model_path: str,
    receipt_path: str,
) -> dict[str, Any]:
    """Rehash the large reference once on rank zero and broadcast the result."""

    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover
        raise PublicationFidelityError("torch.distributed is required for training") from exc
    if not dist.is_available() or not dist.is_initialized():
        return verify_fixed_kl_reference(model_path, receipt_path)
    rank = int(dist.get_rank())
    result: list[object] = [None]
    if rank == 0:
        try:
            result[0] = {"ok": True, "receipt": verify_fixed_kl_reference(model_path, receipt_path)}
        except Exception as exc:
            result[0] = {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise PublicationFidelityError(f"fixed KL reference verification failed: {payload}")
    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        raise PublicationFidelityError("rank zero returned an invalid fixed KL receipt")
    return receipt


def verify_exact_peft_kl_source(
    *,
    policy_adapter_path: str | os.PathLike[str],
    reference_adapter_path: str | os.PathLike[str],
    authoritative_receipt_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authorize and byte-bind the immutable source of an exact paired load."""

    if not checkpoint_fidelity_enabled():
        raise PublicationFidelityError(
            "exact PEFT KL references require explicit publication or development fidelity mode"
        )
    claim_scope = "development" if development_fidelity_enabled() else "publication"
    authoritative = verify_publication_input_adapter(
        reference_adapter_path,
        purpose=f"{claim_scope} exact PEFT KL source",
    )
    if authoritative is None:
        raise PublicationFidelityError("exact PEFT KL source is not a verified adapter checkpoint")
    try:
        return build_source_binding(
            policy_adapter_path=policy_adapter_path,
            reference_adapter_path=reference_adapter_path,
            authoritative_receipt_path=authoritative_receipt_path,
            authoritative_receipt=authoritative,
            tokenizer_path=tokenizer_path,
            claim_scope=claim_scope,
        )
    except ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(f"exact PEFT KL source authorization failed: {exc}") from exc


def run_exact_peft_kl_runtime_preflight(
    trainer: Any,
    reference_model: Any,
    source_binding: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    dtype: str,
    beta: float,
    sync_ref_model: bool,
    use_vllm: bool,
) -> dict[str, Any]:
    """Audit actual paired objects on every rank and persist rank consensus."""

    if not checkpoint_fidelity_enabled():
        raise PublicationFidelityError("exact PEFT KL runtime proof requires fidelity mode")
    claim_scope = "development" if development_fidelity_enabled() else "publication"
    policy_model = _unwrap_trainer_model(trainer)
    accelerator = getattr(trainer, "accelerator", None)
    unwrap = getattr(accelerator, "unwrap_model", None)
    if not callable(unwrap):
        raise PublicationFidelityError("trainer accelerator cannot unwrap exact KL reference")
    try:
        unwrapped_reference = unwrap(reference_model)
    except Exception as exc:
        raise PublicationFidelityError(f"cannot unwrap prepared exact KL reference: {exc}") from exc
    if unwrapped_reference is not reference_model:
        raise PublicationFidelityError(
            "prepared exact KL reference acquired an unsupported distributed wrapper; "
            "only replicated, non-sharded DDP reference placement is authorized"
        )
    if getattr(trainer, "ref_model", None) is not reference_model:
        raise PublicationFidelityError(
            "runtime exact KL audit did not receive the object installed on trainer.ref_model"
        )
    run_contract_identity = _training_run_contract_identity(output_dir)
    try:
        runtime_pair = audit_exact_runtime_pair(
            policy_model,
            unwrapped_reference,
            runtime_dtype=str(dtype),
        )
        verify_source_binding_unchanged(source_binding)
    except ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(f"exact PEFT KL runtime audit failed: {exc}") from exc
    local_digest = _sha256_json(
        {
            "source_binding": dict(source_binding),
            "runtime_pair": runtime_pair,
            "claim_scope": claim_scope,
        }
    )
    distributed = _distributed_digest_agreement(
        local_digest,
        phase="exact_peft_kl_step_zero",
    )
    try:
        training_contract = {
            "dtype": str(dtype),
            "beta": float(beta),
            "sync_ref_model": bool(sync_ref_model),
            "use_vllm": bool(use_vllm),
            "parameterization": "continue_same_bias_all_sft_adapter",
            "policy_adapter_autocast_dtype": True,
            "reference_adapter_autocast_dtype": True,
        }
        receipt = build_runtime_receipt(
            source_binding=source_binding,
            runtime_pair=runtime_pair,
            distributed=distributed,
            training_contract=training_contract,
            claim_scope=claim_scope,
        )
    except ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(f"cannot build exact PEFT KL receipt: {exc}") from exc

    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover
        raise PublicationFidelityError("torch.distributed is required for training") from exc
    distributed_active = dist.is_available() and dist.is_initialized()
    rank = int(dist.get_rank()) if distributed_active else 0
    result: list[object] = [None]
    if rank == 0:
        try:
            result[0] = {
                "ok": True,
                "receipt": write_runtime_receipt(
                    output_dir,
                    receipt,
                    claim_scope=claim_scope,
                ),
            }
        except Exception as exc:
            result[0] = {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
    if distributed_active:
        dist.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise PublicationFidelityError(f"exact PEFT KL receipt write failed: {payload}")
    written = payload.get("receipt")
    if not isinstance(written, dict) or written != receipt:
        raise PublicationFidelityError("rank zero returned mismatched exact PEFT KL evidence")
    setattr(trainer, "_clasp_exact_peft_kl_reference", written)
    setattr(trainer, "_clasp_exact_peft_prepared_reference", reference_model)
    setattr(trainer, "_clasp_exact_peft_unwrapped_reference", unwrapped_reference)
    setattr(trainer, "_clasp_training_run_contract_identity", run_contract_identity)
    setattr(
        trainer,
        "_clasp_exact_peft_kl_receipt_path",
        str(Path(output_dir).expanduser().absolute() / EXACT_PEFT_KL_RECEIPT_FILENAME),
    )
    return written


def _propagate_exact_peft_kl_receipt(
    trainer: Any,
    checkpoint_dir: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Cross-bind step-zero KL evidence into every retained checkpoint tree."""

    source_raw = getattr(trainer, "_clasp_exact_peft_kl_receipt_path", None)
    if source_raw is None:
        return None
    claim_scope = "development" if development_fidelity_enabled() else "publication"
    source = Path(str(source_raw)).expanduser().absolute()
    destination_root = Path(checkpoint_dir).expanduser().absolute()
    destination = destination_root / EXACT_PEFT_KL_RECEIPT_FILENAME
    try:
        receipt = inspect_runtime_receipt(source, claim_scope=claim_scope)
        if destination == source:
            return receipt
        return write_runtime_receipt(
            destination_root,
            receipt,
            claim_scope=claim_scope,
        )
    except ExactPeftKLReferenceError as exc:
        raise PublicationFidelityError(
            f"cannot bind exact PEFT KL lineage into {destination_root}: {exc}"
        ) from exc
