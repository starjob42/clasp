from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from clasp.data.benchmark import forbidden_source_manifest_identity, sha256_json
from clasp.entrypoints import train_sft_jiang_f638 as jiang
from clasp.models import publication_fidelity

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FREEZER_SPEC = importlib.util.spec_from_file_location(
    "freeze_base_sft_campaign", SCRIPTS / "freeze_base_sft_campaign.py"
)
assert FREEZER_SPEC is not None and FREEZER_SPEC.loader is not None
freezer = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = freezer
FREEZER_SPEC.loader.exec_module(freezer)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_base_sft500k_specialist",
    SCRIPTS / "validate_base_sft500k_specialist.py",
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)
resume_fidelity = sys.modules["base_sft_resume_fidelity"]
single_h100_resume = sys.modules["base_sft_single_h100_resume_fidelity"]
completed_run_sealer = sys.modules["seal_completed_sft_run"]

MODEL_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "freeze_model_initialization_registry",
    SCRIPTS / "freeze_model_initialization_registry.py",
)
assert MODEL_REGISTRY_SPEC is not None and MODEL_REGISTRY_SPEC.loader is not None
model_registry_freezer = importlib.util.module_from_spec(MODEL_REGISTRY_SPEC)
sys.modules[MODEL_REGISTRY_SPEC.name] = model_registry_freezer
MODEL_REGISTRY_SPEC.loader.exec_module(model_registry_freezer)


@pytest.fixture
def sealed_run_cleanup() -> list[Path]:
    """Restore owner-write modes after tests that intentionally seal run fixtures."""

    roots: list[Path] = []
    yield roots
    for root in roots:
        if not root.exists():
            continue
        for path in (candidate for candidate in root.rglob("*") if candidate.is_file()):
            if not path.is_symlink():
                path.chmod(0o600)
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_dir()),
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            if not path.is_symlink():
                path.chmod(0o700)
        root.chmod(0o700)


def _checkpoint_fidelity_identity(path: Path) -> dict[str, object]:
    receipt_path = path / "checkpoint_fidelity.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity: dict[str, object] = {
        "schema_version": publication_fidelity.IDENTITY_SCHEMA_VERSION,
        "protocol": publication_fidelity.IDENTITY_PROTOCOL,
        "status": publication_fidelity.IDENTITY_STATUS,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "adapter_config_sha256": receipt["artifacts"]["adapter_config"]["sha256"],
        "adapter_weights_sha256": receipt["artifacts"]["adapter_weights"]["sha256"],
        "trainable_tensor_inventory_sha256": receipt["trainable_closure"][
            "tensor_inventory_sha256"
        ],
        "publication_invariant_sha256": freezer.sha256_json(receipt["publication_invariant"]),
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return identity


def _remote_environment_identity(path: Path) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": freezer.ENVIRONMENT_SCHEMA_VERSION,
        "protocol": freezer.ENVIRONMENT_PROTOCOL,
        "status": freezer.ENVIRONMENT_STATUS,
        "claim_scope": freezer.ENVIRONMENT_CLAIM_SCOPE,
        "canonical_path": str(path.resolve()),
        "receipt_sha256": "7" * 64,
        "receipt_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "producer_implementation_sha256": "8" * 64,
        "python_sha256": "9" * 64,
        "packages_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "accelerator_sha256": "c" * 64,
    }
    identity["identity_sha256"] = freezer.sha256_json(identity)
    return identity


@pytest.fixture(autouse=True)
def _stub_large_qwen_persistence_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact sparse-artifact parser has focused tests in its own module."""

    monkeypatch.setattr(validator, "audit_qwen25_base_bias_scope", lambda _path: {})
    monkeypatch.setattr(validator, "audit_adapter_persistence", lambda _path: {})
    monkeypatch.setattr(validator, "validate_audit_receipt", lambda _path, **_kwargs: {})

    def require_reload_receipt(path: Path, **_kwargs: object) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("SFT reload-equivalence receipt is missing")
        if path.stat().st_mode & 0o222:
            raise ValueError("SFT reload-equivalence receipt must be read-only")
        return {}

    monkeypatch.setattr(validator, "validate_reload_receipt", require_reload_receipt)
    # Exact terminal/periodic artifact closure has direct tests in the sealer
    # module.  These campaign fixtures retain legacy four-rank checkpoint
    # shapes so their historical resume-evidence tests stay independent.
    monkeypatch.setattr(validator, "validate_completed_run_artifacts", lambda *_args: {})

    def require_checkpoint_fidelity(path: Path) -> dict[str, object]:
        if not (path / "checkpoint_fidelity.json").is_file():
            raise ValueError("checkpoint_fidelity.json is missing")
        return _checkpoint_fidelity_identity(path)

    monkeypatch.setattr(validator, "publication_fidelity_identity", require_checkpoint_fidelity)
    monkeypatch.setattr(
        resume_fidelity,
        "publication_fidelity_identity",
        require_checkpoint_fidelity,
    )
    monkeypatch.setattr(
        single_h100_resume,
        "publication_fidelity_identity",
        require_checkpoint_fidelity,
    )

    def require_remote_environment(
        path: Path,
        *,
        require_accelerator: bool,
    ) -> dict[str, object]:
        del require_accelerator
        if not path.is_file() or path.is_symlink():
            raise ValueError("sealed remote-environment receipt is missing")
        if path.stat().st_mode & 0o222:
            raise ValueError("sealed remote-environment receipt must be read-only")
        return _remote_environment_identity(path)

    monkeypatch.setattr(freezer, "environment_identity", require_remote_environment)
    monkeypatch.setattr(
        resume_fidelity,
        "environment_identity",
        require_remote_environment,
    )
    monkeypatch.setattr(
        single_h100_resume,
        "environment_identity",
        require_remote_environment,
    )

    token_module = sys.modules["audit_sft_token_lengths"]

    def require_token_receipt(path: Path, **_kwargs: object) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("all-row token-length receipt is missing")
        if path.stat().st_mode & 0o222:
            raise ValueError("all-row token-length receipt must be read-only")
        return {}

    def token_identity(_receipt: object, *, receipt_path: Path) -> dict[str, object]:
        identity: dict[str, object] = {
            "schema_version": token_module.SCHEMA_VERSION,
            "protocol": token_module.PROTOCOL,
            "status": token_module.STATUS,
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "tokenizer_identity_sha256": "d" * 64,
            "task_audits_sha256": "e" * 64,
            "rows_per_task": 500_000,
            "total_rows": 2_500_000,
            "max_seq_length": 20_000,
            "maximum_observed_tokens": 1,
        }
        identity["identity_sha256"] = freezer.sha256_json(identity)
        return identity

    monkeypatch.setattr(freezer, "validate_token_length_receipt", require_token_receipt)
    monkeypatch.setattr(freezer, "token_length_audit_identity", token_identity)

    clean_module = sys.modules["build_clean_sft500k_corpus"]

    def require_cleanliness_receipt(
        path: Path,
        *,
        expected_data_root: Path | None = None,
    ) -> dict[str, object]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("merged SFT cleanliness receipt is missing")
        if path.stat().st_mode & 0o222:
            raise ValueError("merged SFT cleanliness receipt must be read-only")
        assert expected_data_root is not None
        root = expected_data_root.resolve()
        trust_root = root.parent
        reserved_manifests: dict[str, list[dict[str, object]]] = {}
        reserved_manifest_hashes: dict[str, str] = {}
        for task in freezer.TASKS:
            manifest = []
            for source_class in clean_module.RESERVED_SOURCE_CLASSES:
                source = (
                    trust_root
                    / "reserved_sources_v1"
                    / source_class
                    / task
                    / "rows.json"
                )
                manifest.append(
                    {
                        "source_class": source_class,
                        "canonical_path": str(source.resolve()),
                        "size_bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "rows": 1,
                    }
                )
            reserved_manifests[task] = manifest
            reserved_manifest_hashes[task] = clean_module.sha256_json(manifest)
        identity: dict[str, object] = {
            "schema_version": clean_module.SCHEMA_VERSION,
            "protocol": clean_module.MERGED_PROTOCOL,
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_root": str(root),
            "rows_per_task": 500_000,
            "total_rows": 2_500_000,
            "implementation_manifest_sha256": "f" * 64,
            "task_identity_sha256": {task: "1" * 64 for task in freezer.TASKS},
            "task_receipt_canonical_path": {
                task: str(root / "cleanliness_receipts" / f"{task}.json")
                for task in freezer.TASKS
            },
            "task_output_sha256": {
                task: hashlib.sha256(
                    (root / task / "train" / "rows.json").read_bytes()
                ).hexdigest()
                for task in freezer.TASKS
            },
            "task_unique_demonstrations": {
                task: 500_000 for task in freezer.TASKS
            },
            "task_unique_semantic_instances": {task: 1 for task in freezer.TASKS},
            "task_reserved_source_manifest": reserved_manifests,
            "task_reserved_source_manifest_sha256": reserved_manifest_hashes,
        }
        identity["identity_sha256"] = clean_module.sha256_json(identity)
        return identity

    monkeypatch.setattr(freezer, "validate_merged_receipt", require_cleanliness_receipt)
    monkeypatch.setattr(
        freezer,
        "cleanliness_implementation_identity",
        lambda: {"implementation_manifest_sha256": "f" * 64},
    )
    prepare_module = sys.modules["prepare_sota_run"]

    def fixture_base_fingerprint(path: Path) -> dict[str, object]:
        return prepare_module.fingerprint_model(path, label="fixture Qwen base")

    monkeypatch.setattr(freezer, "audit_jiang_f638_base_snapshot", fixture_base_fingerprint)
    monkeypatch.setattr(freezer, "validate_jiang_f638_base_receipt", lambda receipt: receipt)
    # Other test modules can import the freezer before this module is
    # collected.  `base_sft_resume_fidelity` retains that earlier
    # `load_campaign` function object, so patch its defining globals as well
    # as this module's local freezer instance.
    monkeypatch.setitem(
        resume_fidelity.load_campaign.__globals__,
        "validate_jiang_f638_base_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setitem(
        single_h100_resume.load_campaign.__globals__,
        "validate_jiang_f638_base_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        prepare_module,
        "audit_jiang_f638_base_snapshot",
        fixture_base_fingerprint,
    )
    monkeypatch.setattr(
        prepare_module,
        "validate_jiang_f638_base_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(prepare_module, "audit_qwen25_base_bias_scope", lambda _path: {})
    monkeypatch.setattr(prepare_module, "audit_adapter_persistence", lambda _path: {})
    monkeypatch.setattr(
        prepare_module,
        "validate_audit_receipt",
        lambda _path, **_kwargs: {},
    )
    monkeypatch.setattr(prepare_module, "validate_reload_receipt", require_reload_receipt)
    monkeypatch.setattr(
        prepare_module,
        "publication_fidelity_identity",
        require_checkpoint_fidelity,
    )

    def reload_identity(path: Path, *, base_model_path: Path) -> dict[str, object]:
        del base_model_path
        checkpoint = require_checkpoint_fidelity(path)
        identity: dict[str, object] = {
            "schema_version": 2,
            "protocol": "clasp-sft-post-save-reload-equivalence-v2",
            "status": "exact_post_save_reload_verified",
            "receipt_sha256": "1" * 64,
            "receipt_file_sha256": hashlib.sha256(
                (path / "sft_reload_equivalence.json").read_bytes()
            ).hexdigest(),
            "checkpoint_fidelity_identity_sha256": checkpoint["identity_sha256"],
            "serialized_trainable_closure_sha256": "2" * 64,
            "runtime_pair_sha256": "3" * 64,
            "functional_equivalence_sha256": "4" * 64,
            "producer_sha256": "5" * 64,
        }
        identity["identity_sha256"] = freezer.sha256_json(identity)
        return identity

    monkeypatch.setattr(prepare_module, "reload_equivalence_identity", reload_identity)
    registry_globals = model_registry_freezer.build_model_initialization_registry.__globals__
    monkeypatch.setitem(registry_globals, "audit_qwen25_base_bias_scope", lambda _path: {})
    monkeypatch.setitem(registry_globals, "audit_adapter_persistence", lambda _path: {})
    monkeypatch.setitem(
        registry_globals,
        "audit_jiang_f638_base_snapshot",
        fixture_base_fingerprint,
    )
    monkeypatch.setitem(
        registry_globals,
        "validate_jiang_f638_base_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setitem(
        registry_globals,
        "validate_audit_receipt",
        lambda _path, **_kwargs: {},
    )
    monkeypatch.setitem(registry_globals, "validate_reload_receipt", require_reload_receipt)
    monkeypatch.setitem(registry_globals, "reload_equivalence_identity", reload_identity)
    monkeypatch.setitem(
        registry_globals,
        "publication_fidelity_identity",
        require_checkpoint_fidelity,
    )


def _source_snapshot(root: Path) -> Path:
    source = root / "immutable_source"
    required = sorted(freezer.REQUIRED_SOURCE_FILES)
    for relative in required:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen test source: {relative}\n", encoding="utf-8")
    logs = source / "logs"
    logs.mkdir()
    manifest_lines = []
    for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        relative = path.relative_to(source).as_posix()
        manifest_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
        path.chmod(0o440)
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")
    manifest.chmod(0o440)
    for directory in sorted(
        (candidate for candidate in source.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        directory.chmod(0o770 if directory == logs else 0o550)
    source.chmod(0o550)
    return source


def _base_model(root: Path, *, instruction: bool = False, writable: bool = False) -> Path:
    base = root / ("instruct_base" if instruction else "base")
    base.mkdir(parents=True)
    (base / "config.json").write_text(
        json.dumps(freezer.QWEN25_7B_BASE_SIGNATURE), encoding="utf-8"
    )
    (base / "model.safetensors").write_bytes(b"qwen25-7b-base-test-weights")
    weight_map = {
        f"model.layers.{layer}.self_attn.{module}.bias": "model.safetensors"
        for layer in range(28)
        for module in ("q_proj", "k_proj", "v_proj")
    }
    weight_map["model.embed_tokens.weight"] = "model.safetensors"
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    tokenizer = {"chat_template": "{{ messages }}"} if instruction else {}
    (base / "tokenizer_config.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    if not writable:
        for path in base.iterdir():
            path.chmod(0o440)
        base.chmod(0o550)
    return base


def _registry_and_data(root: Path) -> tuple[Path, Path, Path]:
    data_root = root / "data_sft"
    reserved_root = root / "reserved_sources_v1"
    reserved_specs: list[tuple[str, str, Path]] = []
    iid_tasks: dict[str, object] = {}
    for task_index, task in enumerate(freezer.TASKS):
        for class_index, source_class in enumerate(
            sys.modules["build_clean_sft500k_corpus"].RESERVED_SOURCE_CLASSES
        ):
            source = reserved_root / source_class / task
            source.mkdir(parents=True)
            path = source / "rows.json"
            path.write_text(
                json.dumps([{"task": task, "id": task_index * 10 + class_index}]) + "\n",
                encoding="utf-8",
            )
            path.chmod(0o440)
            reserved_specs.append((task, source_class, path))
            if source_class == "iid_confirmation":
                ordered = [freezer.sha256_json({"task": task, "row": 0})]
                iid_tasks[task] = {
                    "attempts": 1,
                    "distribution_counts": {"fixture": 1},
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "ordered_instance_hashes_sha256": freezer.sha256_json(ordered),
                    "ordered_instance_sha256": ordered,
                    "relative_path": f"{task}/eval/test.json",
                    "rows": 1,
                    "seed": task_index + 1,
                }
    generation: dict[str, object] = {
        "schema_version": 1,
        "status": "instances_generated",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "contains_references": False,
        "publishable": False,
        "generator_code_sha256": "4" * 64,
        "generator_contract_sha256": "5" * 64,
        "runtime_audit": {
            "source_audit": {
                "root": str(root.resolve()),
                "verified_locally": True,
                "observed_files_sha256": {"generator.py": "6" * 64},
            },
            "version_mismatches": {},
            "versions": {"python": "3.11", "numpy": "2.3", "networkx": "3.5"},
        },
        "tasks": iid_tasks,
    }
    generation["receipt_sha256"] = freezer.sha256_json(generation)
    generation_path = reserved_root / "iid_confirmation" / "generation_receipt.json"
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    generation_path.chmod(0o440)
    reserved_module = sys.modules["freeze_sft_reserved_source_registry"]
    reserved_payload = reserved_module.build_registry(
        trust_root=root,
        sources=reserved_specs,
        iid_generation_receipt=generation_path,
        frozen_at_utc="2026-08-14T00:00:00Z",
    )
    reserved_registry = root / "sft_reserved_source_registry.json"
    reserved_module._write_create_once(reserved_registry, reserved_payload)

    records = []
    for task_index, task in enumerate(freezer.TASKS):
        for label_index, label in enumerate(("historical-eval", "rl-train", "sft-train")):
            if label == "sft-train":
                source = data_root / task / "train"
                source.mkdir(parents=True)
                path = source / "rows.json"
                path.write_text(
                    json.dumps([{"task": task, "id": task_index * 10 + label_index}]) + "\n",
                    encoding="utf-8",
                )
            elif label == "historical-eval":
                source = reserved_root / "rl_eval" / task
                path = source / "rows.json"
            else:
                source = reserved_root / "rl_train" / task
                path = source / "rows.json"
            rows = 500_000 if label == "sft-train" else 1
            files = [
                {
                    "relative_path": "rows.json",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                    "rows": rows,
                }
            ]
            record = {
                "task": task,
                "label": label,
                "registered_source_locator": str(source.resolve()),
                "files": files,
                "file_manifest_sha256": sha256_json(files),
                "rows": rows,
                "unique_instance_hashes": rows,
            }
            record["source_identity_sha256"] = forbidden_source_manifest_identity(record)[
                "source_identity_sha256"
            ]
            records.append(record)
    source_registry_module = sys.modules["freeze_forbidden_source_registry"]
    generation_identity = reserved_module.iid_generation_receipt_identity(generation_path)
    payload = {
        "schema_version": source_registry_module.BASE_SFT_SCHEMA_VERSION,
        "protocol": source_registry_module.BASE_SFT_PROTOCOL,
        "status": source_registry_module.BASE_SFT_STATUS,
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": list(freezer.TASKS),
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": records,
        "source_tree_manifest_sha256": sha256_json(
            [forbidden_source_manifest_identity(record) for record in records]
        ),
        "temporal_claim": source_registry_module.BASE_SFT_TEMPORAL_CLAIM,
        "iid_generation_receipt": generation_identity,
        "pre_iid_forbidden_registry_identity": None,
        "pre_iid_provenance_reason": (
            source_registry_module.BASE_SFT_PRE_IID_PROVENANCE_REASON
        ),
    }
    payload["registry_sha256"] = sha256_json(payload)
    registry = root / "forbidden_source_registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    registry.chmod(0o440)
    token_receipt = data_root / "sft_token_length_audit.json"
    token_receipt.write_text("{}\n", encoding="utf-8")
    token_receipt.chmod(0o440)
    cleanliness_receipt = data_root / "sft_cleanliness_audit.json"
    cleanliness_receipt.write_text("{}\n", encoding="utf-8")
    cleanliness_receipt.chmod(0o440)
    for path in sorted(
        [*data_root.rglob("*"), *reserved_root.rglob("*")],
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        path.chmod(0o550 if path.is_dir() else 0o440)
    data_root.chmod(0o550)
    reserved_root.chmod(0o550)
    return registry, data_root, reserved_registry


def _environment_receipt(root: Path) -> Path:
    receipt = root / "sealed_remote_environment.json"
    receipt.write_text('{"test":"sealed-environment"}\n', encoding="utf-8")
    receipt.chmod(0o440)
    return receipt


def _frozen_campaign(
    tmp_path: Path,
    *,
    publish: bool = True,
) -> dict[str, Path | dict]:
    trust_root = tmp_path / "immutable_inputs"
    trust_root.mkdir()
    source = _source_snapshot(trust_root)
    base = _base_model(trust_root)
    registry, data_root, reserved_registry = _registry_and_data(trust_root)
    environment = _environment_receipt(tmp_path)
    output_root = tmp_path / "base_sft_outputs"
    authority_root = tmp_path / "base_sft_authority"
    trust_root.chmod(0o550)
    payload = freezer.build_campaign(
        campaign_id="qwen25-base-sft500k-test-v1",
        run_contract_authority_root=authority_root,
        immutable_input_trust_root=trust_root,
        source_root=source,
        base_model_path=base,
        forbidden_source_registry=registry,
        reserved_source_registry=reserved_registry,
        remote_environment_receipt=environment,
        sft_data_root=data_root,
        output_root=output_root,
        frozen_at_utc="2026-08-14T00:00:00Z",
    )
    campaign = authority_root / "base_sft_campaign.json"
    if publish:
        freezer._write_create_once(campaign, payload)
    return {
        "source": source,
        "trust_root": trust_root,
        "base": base,
        "registry": registry,
        "reserved_registry": reserved_registry,
        "environment": environment,
        "data_root": data_root,
        "output_root": output_root,
        "campaign": campaign,
        "authority_root": authority_root,
        "payload": payload,
    }


def _runtime_kwargs(assets: dict[str, Path | dict]) -> dict[str, Path]:
    return {
        "campaign_path": assets["campaign"],
        "immutable_input_trust_root": assets["trust_root"],
        "source_root": assets["source"],
        "base_model_path": assets["base"],
        "forbidden_source_registry": assets["registry"],
        "reserved_source_registry": assets["reserved_registry"],
        "remote_environment_receipt": assets["environment"],
        "sft_data_root": assets["data_root"],
        "output_root": assets["output_root"],
    }


def test_campaign_freezes_all_five_base_specialists_before_output_exists(
    tmp_path: Path,
) -> None:
    assets = _frozen_campaign(tmp_path)
    payload = assets["payload"]
    assert isinstance(payload, dict)
    assert payload["tasks"] == ["cvrp", "jssp", "mis", "op", "tsp"]
    assert payload["schema_version"] == 9
    assert payload["protocol"] == "clasp-jiang-f638-base-specialist-sft-campaign-v9"
    assert payload["base_model_signature"] == freezer.QWEN25_7B_BASE_SIGNATURE
    assert payload["base_bias_scope"]["policy"] == "all"
    assert payload["base_bias_scope"]["bias_tensor_count"] == 84
    assert payload["training_design"]["base_model_requirement"]["non_instruction"] is True
    assert payload["training_design"]["environment_exact"] == {
        "CLASP_PUBLICATION_FIDELITY": "1",
        "receipt_protocol": freezer.ENVIRONMENT_PROTOCOL,
        "receipt_claim_scope": freezer.ENVIRONMENT_CLAIM_SCOPE,
        "accelerator_count": 1,
        "accelerator_family": "NVIDIA H100 compute capability 9.0",
        "requirements_lock": "requirements.jiang-f638-sft.txt",
        "requirements_lock_sha256": (
            "b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6"
        ),
        "full_python_prefix_tree_sha256": True,
        "requires_nonwritable_prefix_and_parent": True,
        "requires_live_revalidation_every_allocation": True,
    }
    assert payload["training_design"]["stage_hyperparameters"]["save_total_limit"] == 50
    assert payload["training_design"]["checkpoint_retention"] == {
        "save_total_limit": 50,
        "trainer_checkpoint_count": 7,
        "forced_terminal_checkpoint_step": 31_250,
        "resume_supported": True,
        "fresh_run_only": False,
        "resume_world_size": 1,
        "resume_boundaries": [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 31_250],
        "resume_dependency": "afterok",
        "resume_receipt": "base_sft_single_h100_resume_fidelity.json",
        "resume_receipt_protocol": (
            "clasp-publication-base-sft-single-h100-resume-state-v1"
        ),
        "exact_resume_receipt_every_checkpoint": True,
        "completed_run_seal_receipt": "completed_sft_run_seal.json",
        "requires_recursive_exact_modes_for_complete": True,
        "checkpoint_file_mode": "0440",
        "checkpoint_directory_mode": "0550",
        "minimum_matrix_free_bytes_before_submission": freezer.MINIMUM_MATRIX_FREE_BYTES,
        "minimum_cell_free_bytes_before_allocation": freezer.MINIMUM_CELL_FREE_BYTES,
    }
    assert payload["sft_token_length_audit"]["total_rows"] == 2_500_000
    assert payload["sft_token_length_audit"]["max_seq_length"] == 20_000
    assert payload["sft_cleanliness_audit"]["total_rows"] == 2_500_000
    assert len(payload["sft_reserved_source_registry"]["sources"]) == 20
    assert payload["sft_reserved_source_registry"]["iid_generation_receipt"] is not None
    assert all(
        count == 500_000
        for count in payload["sft_cleanliness_audit"][
            "task_unique_demonstrations"
        ].values()
    )
    assert payload["training_design"]["clean_corpus_audit"][
        "requires_zero_reserved_evaluation_overlap"
    ] is True
    assert payload["training_design"]["full_context_audit"][
        "requires_zero_truncated_rows"
    ] is True
    assert payload["training_design"]["post_save_reload_equivalence"][
        "requires_pre_save_to_reloaded_trainable_tensor_equality"
    ] is True
    assert payload["remote_environment"] == _remote_environment_identity(
        Path(assets["environment"])
    )
    assert (
        payload["training_design"]["stage_hyperparameters"]["checkpoint_fidelity_mode"]
        == "publication"
    )
    assert payload["training_design"]["adapter_persistence"] == {
        "bias_policy": "all",
        "architecture_scope": "all_base_biases_equal_targeted_qkv_biases",
        "expected_bias_tensors": 84,
        "expected_bias_parameters": 129_024,
        "expected_lora_tensors": 392,
        "expected_lora_parameters": 161_480_704,
        "requires_exact_safetensors_inventory": True,
    }
    assert payload["campaign_sha256"] == freezer.sha256_json(
        {key: value for key, value in payload.items() if key != "campaign_sha256"}
    )
    observed, file_sha = freezer.verify_campaign_runtime(
        **_runtime_kwargs(assets),
        tasks=freezer.TASKS,
        require_output_absent=True,
    )
    assert observed == payload
    assert file_sha == hashlib.sha256(Path(assets["campaign"]).read_bytes()).hexdigest()


def test_campaign_rejects_instruction_or_writable_base(tmp_path: Path) -> None:
    for name in ("instruction", "writable"):
        case = tmp_path / name
        trust_root = case / "immutable_inputs"
        trust_root.mkdir(parents=True)
        source = _source_snapshot(trust_root)
        registry, data_root, reserved_registry = _registry_and_data(trust_root)
        environment = _environment_receipt(case)
        base = _base_model(
            trust_root,
            instruction=name == "instruction",
            writable=name == "writable",
        )
        trust_root.chmod(0o550)
        with pytest.raises(ValueError, match="chat template|writable"):
            freezer.build_campaign(
                campaign_id=f"reject-{name}",
                run_contract_authority_root=case / "authority",
                immutable_input_trust_root=trust_root,
                source_root=source,
                base_model_path=base,
                forbidden_source_registry=registry,
                reserved_source_registry=reserved_registry,
                remote_environment_receipt=environment,
                sft_data_root=data_root,
                output_root=tmp_path / f"out-{name}",
                frozen_at_utc="2026-08-14T00:00:00Z",
            )


def test_campaign_rejects_cleanliness_output_hash_not_in_registered_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = freezer.validate_merged_receipt

    def mismatched(path: Path, **kwargs: object) -> dict[str, object]:
        identity = original(path, **kwargs)
        outputs = dict(identity["task_output_sha256"])
        outputs["cvrp"] = "0" * 64
        identity["task_output_sha256"] = outputs
        identity["identity_sha256"] = sys.modules["build_clean_sft500k_corpus"].sha256_json(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
        return identity

    monkeypatch.setattr(freezer, "validate_merged_receipt", mismatched)
    with pytest.raises(ValueError, match="not the exact audited clean output"):
        _frozen_campaign(tmp_path)


def test_campaign_rejects_clean_reservation_not_in_authority(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    payload = assets["payload"]
    assert isinstance(payload, dict)
    cleanliness = json.loads(json.dumps(payload["sft_cleanliness_audit"]))
    cleanliness["task_reserved_source_manifest"]["cvrp"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the authoritative registry"):
        freezer._require_clean_reserved_source_equality(
            cleanliness_identity=cleanliness,
            reserved_registry_identity=payload["sft_reserved_source_registry"],
        )


def test_campaign_requires_nonnull_iid_generation_provenance(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    payload = json.loads(json.dumps(assets["payload"]))
    reserved = payload["sft_reserved_source_registry"]
    reserved["iid_generation_receipt"] = None
    unsigned_reserved = dict(reserved)
    unsigned_reserved.pop("identity_sha256")
    reserved["identity_sha256"] = freezer.sha256_json(unsigned_reserved)
    unsigned_campaign = dict(payload)
    unsigned_campaign.pop("campaign_sha256")
    payload["campaign_sha256"] = freezer.sha256_json(unsigned_campaign)
    with pytest.raises(ValueError, match="IID generation-receipt provenance"):
        freezer.validate_campaign(payload)


def test_campaign_rejects_rl_reservation_not_in_forbidden_registry(
    tmp_path: Path,
) -> None:
    assets = _frozen_campaign(tmp_path)
    payload = assets["payload"]
    assert isinstance(payload, dict)
    reserved = freezer.reserved_sources_by_task_class(
        payload["sft_reserved_source_registry"]
    )
    reserved["cvrp"]["rl_train"]["sha256"] = "0" * 64
    registry_identity = freezer.validate_forbidden_source_registry(
        json.loads(Path(assets["registry"]).read_text(encoding="utf-8")),
        expected_tasks=list(freezer.TASKS),
    )
    with pytest.raises(ValueError, match="does not equal registered rl-train bytes"):
        freezer._require_registered_rl_reserved_sources(
            reserved=reserved,
            forbidden_registry_identity=registry_identity,
        )


def test_campaign_rejects_environment_receipt_inside_source_snapshot(
    tmp_path: Path,
) -> None:
    trust_root = tmp_path / "immutable_inputs"
    trust_root.mkdir()
    source = _source_snapshot(trust_root)
    base = _base_model(trust_root)
    registry, data_root, reserved_registry = _registry_and_data(trust_root)
    trust_root.chmod(0o550)
    nested_receipt = source / "scripts" / "freeze_remote_environment.py"
    with pytest.raises(ValueError, match="must live outside the source snapshot tree"):
        freezer.build_campaign(
            campaign_id="reject-nested-environment",
            run_contract_authority_root=tmp_path / "authority",
            immutable_input_trust_root=trust_root,
            source_root=source,
            base_model_path=base,
            forbidden_source_registry=registry,
            reserved_source_registry=reserved_registry,
            remote_environment_receipt=nested_receipt,
            sft_data_root=data_root,
            output_root=tmp_path / "out-nested-environment",
            frozen_at_utc="2026-08-14T00:00:00Z",
        )


def test_runtime_rejects_changed_registered_sft_bytes(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    changed = Path(assets["data_root"]) / "tsp" / "train" / "rows.json"
    changed.chmod(0o640)
    changed.write_text('[{"task":"tsp","id":999}]\n', encoding="utf-8")
    changed.chmod(0o440)
    with pytest.raises(ValueError, match="differs from registry"):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=["tsp"],
            require_output_absent=True,
        )


def test_runtime_requires_bound_cleanliness_receipt(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    data_root = Path(assets["data_root"])
    receipt = data_root / "sft_cleanliness_audit.json"
    data_root.chmod(0o750)
    receipt.unlink()
    data_root.chmod(0o550)
    with pytest.raises(ValueError, match="cleanliness receipt is missing"):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=["tsp"],
            require_output_absent=True,
        )


@pytest.mark.parametrize(
    "role",
    ["trust_root", "source", "base", "registry", "data_root", "data_file"],
)
def test_runtime_rejects_writable_bound_input_or_ancestor(
    tmp_path: Path,
    role: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    targets = {
        "trust_root": Path(assets["trust_root"]),
        "source": Path(assets["source"]),
        "base": Path(assets["base"]),
        "registry": Path(assets["registry"]),
        "data_root": Path(assets["data_root"]),
        "data_file": Path(assets["data_root"]) / "tsp" / "train" / "rows.json",
    }
    target = targets[role]
    target.chmod(0o750 if target.is_dir() else 0o640)
    with pytest.raises(ValueError, match="writable"):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=["tsp"],
            require_output_absent=True,
        )


def test_runtime_rejects_fd_bound_input_inode_replacement(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    trust_root = Path(assets["trust_root"])
    registry = Path(assets["registry"])
    original = registry.read_bytes()
    trust_root.chmod(0o750)
    replacement = trust_root / "replacement-registry.json"
    replacement.write_bytes(original)
    registry.unlink()
    replacement.rename(registry)
    registry.chmod(0o440)
    trust_root.chmod(0o550)
    with pytest.raises(ValueError, match="sealed input path identity differs"):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=["tsp"],
            require_output_absent=True,
        )


def test_runtime_rejects_fifo_leaf_without_blocking(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    trust_root = Path(assets["trust_root"])
    registry = Path(assets["registry"])
    trust_root.chmod(0o750)
    registry.unlink()
    os.mkfifo(registry, mode=0o440)
    trust_root.chmod(0o550)
    with pytest.raises(ValueError, match="unexpected entry type"):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=["tsp"],
            require_output_absent=True,
        )


@pytest.mark.parametrize("mutation", ["missing", "writable", "changed"])
def test_runtime_reopens_exact_sealed_environment_receipt(
    tmp_path: Path,
    mutation: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    receipt = Path(assets["environment"])
    if mutation == "missing":
        receipt.unlink()
        expected = "No such file"
    elif mutation == "writable":
        receipt.chmod(0o640)
        expected = "read-only"
    else:
        receipt.chmod(0o640)
        receipt.write_text('{"test":"changed"}\n', encoding="utf-8")
        receipt.chmod(0o440)
        expected = "differs from the campaign"
    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        freezer.verify_campaign_runtime(
            **_runtime_kwargs(assets),
            tasks=freezer.TASKS,
            require_output_absent=True,
        )


def test_runtime_accelerator_requirement_reaches_live_environment_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path)
    calls: list[bool] = []

    def observe(path: Path, *, require_accelerator: bool) -> dict[str, object]:
        calls.append(require_accelerator)
        return _remote_environment_identity(path)

    monkeypatch.setattr(freezer, "environment_identity", observe)
    freezer.verify_campaign_runtime(
        **_runtime_kwargs(assets),
        tasks=["tsp"],
        require_output_absent=True,
        require_accelerator=True,
    )
    assert calls == [True]


def test_input_only_allocation_gate_does_not_inspect_or_initialize_run(
    tmp_path: Path,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    unsafe_run = output / "runs" / "tsp" / "seed42"
    unsafe_run.mkdir(parents=True)
    (unsafe_run / "unexpected.partial").write_text("must not be inspected\n")
    args = Namespace(
        campaign_contract=Path(assets["campaign"]),
        immutable_input_trust_root=Path(assets["trust_root"]),
        source_root=Path(assets["source"]),
        base_model_path=Path(assets["base"]),
        forbidden_source_registry=Path(assets["registry"]),
        reserved_source_registry=Path(assets["reserved_registry"]),
        remote_environment_receipt=Path(assets["environment"]),
        data_root=Path(assets["data_root"]),
        output_root=output,
        task="tsp",
        require_live_accelerator=True,
    )
    result = validator.verify_inputs_only(args)
    assert result.startswith("INPUTS_VALID:")
    assert (unsafe_run / "unexpected.partial").is_file()


def _adapter(path: Path, base: Path, *, persistence_receipt: bool = True) -> None:
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 64,
                "lora_alpha": 64,
                "lora_dropout": 0.0,
                "bias": "all",
                "target_modules": sorted(validator.JIANG_LORA_MODULES),
                "base_model_name_or_path": str(base),
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    if persistence_receipt:
        receipt = path / "adapter_persistence_receipt.json"
        receipt.write_text("{}\n", encoding="utf-8")
        receipt.chmod(0o440)
        reload_receipt = path / "sft_reload_equivalence.json"
        reload_receipt.write_text("{}\n", encoding="utf-8")
        reload_receipt.chmod(0o440)
    fidelity_payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "fixture-checkpoint-fidelity",
        "status": "exact_trainable_policy_persisted",
        "artifacts": {
            "adapter_config": {
                "sha256": hashlib.sha256((path / "adapter_config.json").read_bytes()).hexdigest()
            },
            "adapter_weights": {
                "sha256": hashlib.sha256(
                    (path / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
            },
        },
        "trainable_closure": {"tensor_inventory_sha256": "5" * 64},
        "publication_invariant": {"protocol": "fixture", "status": "exact"},
    }
    fidelity_payload["receipt_sha256"] = freezer.sha256_json(fidelity_payload)
    fidelity_receipt = path / "checkpoint_fidelity.json"
    fidelity_receipt.write_text(
        json.dumps(fidelity_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fidelity_receipt.chmod(0o440)


def _seal_completed_fixture(run: Path, base: Path) -> None:
    completed_run_sealer.seal_completed_run(
        run,
        base,
        artifact_validator=lambda _run, _base: {"fixture": True},
    )


def _hyperparams(run: Path, *, task: str, base: Path, data_root: Path) -> None:
    stage = dict(freezer.BASE_SFT_TRAINING_DESIGN["stage_hyperparameters"])
    stage.update(
        {
            "tokenizer_path": str(base),
            "tokenization_alignment_train_mismatches": 0,
            "tokenization_alignment_eval_mismatches": 0,
        }
    )
    args = {
        "tasks": [task],
        "model_path": str(base),
        "tokenizer_path": str(base),
        "data_root": str(data_root),
        "prompt_style": "jiang_f638",
        "no_reason": True,
        "dtype": "bfloat16",
        "eval_ratio": 0.0,
        "eval_from_train_split": False,
        "max_samples_per_task": 500_000,
        "max_eval_samples_per_task": 0,
        "min_train_samples_per_task": 500_000,
        "min_eval_samples_per_task": 0,
        "prefer_subset_files": False,
        "deduplicate_train_rows": False,
        "logging_steps": 1,
        "save_steps": 5_000,
        "eval_steps": None,
        "save_total_limit": 50,
    }
    (run / "hyperparams.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "tasks": [task],
                "model_path": str(base),
                "dataset": {
                    "train_size": 500_000,
                    "eval_size": 0,
                    "train_task_counts": {task: 500_000},
                },
                "stage_hyperparameters": stage,
                "args": args,
                "distributed": {"world_size": 1},
            }
        ),
        encoding="utf-8",
    )


def test_trainer_hyperparams_satisfy_frozen_campaign_validator_schema(
    tmp_path: Path,
) -> None:
    """Exercise the producer payload through the production stage validator."""

    base = tmp_path / "base"
    data_root = tmp_path / "data"
    run = tmp_path / "run"
    for path in (base, data_root, run):
        path.mkdir()
    args = Namespace(
        task="jssp",
        cache_dir=None,
        report_to="none",
        post_save_reload_check=True,
    )

    class TrainingArgs:
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 4
        warmup_steps = 20
        num_train_epochs = 1.0
        learning_rate = 0.0002
        bf16 = True
        logging_steps = 1
        optim = "adamw_8bit"
        weight_decay = 0.01
        lr_scheduler_type = "linear"
        seed = 42
        eval_strategy = "no"
        metric_for_best_model = "eval_loss"
        save_total_limit = 50
        save_steps = 5_000
        dataset_text_field = "text"
        max_seq_length = 20_000
        dataset_num_proc = 16
        packing = False
        per_device_eval_batch_size = 8
        eval_steps = None

        def to_dict(self) -> dict[str, object]:
            return {"seed": 42, "output_dir": str(run / "checkpoints")}

    payload = jiang._build_hyperparams_payload(
        args=args,
        model_path=base,
        tokenizer_path=base,
        data_root=data_root,
        train_source=data_root / "jssp" / "train",
        output_dir=run,
        runtime_versions=jiang.PINNED_RUNTIME_VERSIONS,
        hardware={"world_size": 1},
        base_snapshot={"sha256": "a" * 64},
        marker_audit={"rows_checked": 500_000},
        policy_preflight={"status": "exact"},
        training_args=TrainingArgs(),
    )
    (run / "hyperparams.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validator._validate_hyperparams(
        run_dir=run,
        task="jssp",
        base_model_path=base,
        data_root=data_root,
    )


def _checkpoint(run: Path, *, step: int, base: Path) -> Path:
    import random

    import numpy as np
    import torch

    checkpoint = run / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _adapter(checkpoint, base, persistence_receipt=False)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": 31_250}), encoding="utf-8"
    )
    torch.save(
        {
            "state": {},
            "param_groups": [{"params": [0], "lr": 2e-4, "weight_decay": 0.01}],
        },
        checkpoint / "optimizer.pt",
    )
    torch.save(
        {"last_epoch": step, "_step_count": step + 1, "_last_lr": [2e-4]},
        checkpoint / "scheduler.pt",
    )
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    for rank in range(4):
        torch.save(rng_state, checkpoint / f"rng_state_{rank}.pth")
    for name in (
        "README.md",
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.json",
    ):
        (checkpoint / name).write_bytes(f"fixture:{name}".encode())
    _seal_checkpoint(run, checkpoint)
    return checkpoint


def _single_h100_checkpoint(run: Path, *, step: int, base: Path) -> Path:
    """Build the exact v9 one-process Trainer checkpoint and seal its receipt."""

    import random

    import numpy as np
    import torch

    checkpoint = run / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _adapter(checkpoint, base, persistence_receipt=False)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": 31_250}), encoding="utf-8"
    )
    torch.save(
        {"state": {}, "param_groups": [{"params": [0], "lr": 2e-4, "weight_decay": 0.01}]},
        checkpoint / "optimizer.pt",
    )
    torch.save(
        {"last_epoch": step, "_step_count": step + 1, "_last_lr": [2e-4]},
        checkpoint / "scheduler.pt",
    )
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        },
        checkpoint / "rng_state.pth",
    )
    for name in (
        "README.md",
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.json",
    ):
        (checkpoint / name).write_bytes(f"fixture:{name}".encode())
    campaign = run.parents[3] / "base_sft_authority" / "base_sft_campaign.json"
    single_h100_resume.seal_resume_state(
        campaign_path=campaign,
        run_contract_path=run / validator.RUN_CONTRACT_FILENAME,
        checkpoint=checkpoint,
        require_accelerator=False,
        verify_live_environment=False,
    )
    return checkpoint


def _single_h100_checkpoint_prefix(run: Path, *, through: int, base: Path) -> Path:
    latest: Path | None = None
    for step in validator.CHUNK_BOUNDARIES:
        if step > through:
            break
        candidate = run / "checkpoints" / f"checkpoint-{step}"
        latest = candidate if candidate.is_dir() else _single_h100_checkpoint(
            run, step=step, base=base
        )
    assert latest is not None
    return latest


def _rewrite_single_h100_resume_receipt(checkpoint: Path, mutate: object) -> None:
    receipt = checkpoint / single_h100_resume.RECEIPT_FILENAME
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    mutate(payload)
    payload["receipt_sha256"] = freezer.sha256_json(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    receipt.chmod(0o640)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt.chmod(0o440)


def _distributed_receipt(checkpoint: Path, *, step: int) -> dict[str, object]:
    resume_fidelity._prepare_checkpoint_for_seal(checkpoint)
    root = checkpoint.stat(follow_symlinks=False)
    records = []
    for rank in range(4):
        rng = resume_fidelity._state_stat(checkpoint / f"rng_state_{rank}.pth")
        rng["sha256"] = hashlib.sha256(
            (checkpoint / f"rng_state_{rank}.pth").read_bytes()
        ).hexdigest()
        record = {
            "rank": rank,
            "world_size": 4,
            "global_step": step,
            "checkpoint_device": root.st_dev,
            "checkpoint_inode": root.st_ino,
            "shared_state": [
                resume_fidelity._state_stat(checkpoint / name)
                for name in resume_fidelity.REQUIRED_STATE_FILES
            ],
            "rank_rng": rng,
        }
        record["record_sha256"] = freezer.sha256_json(record)
        records.append(record)
    evidence = {
        "phase": "post_checkpoint_durable_state",
        "world_size": 4,
        "ordered_rank_records": records,
    }
    evidence["consensus_sha256"] = freezer.sha256_json(evidence)
    return evidence


def _seal_checkpoint(run: Path, checkpoint: Path) -> None:
    campaign = run.parents[3] / "base_sft_authority" / "base_sft_campaign.json"
    evidence = _distributed_receipt(
        checkpoint,
        step=int(checkpoint.name.removeprefix("checkpoint-")),
    )
    resume_fidelity.seal_resume_state(
        campaign_path=campaign,
        run_contract_path=run / validator.RUN_CONTRACT_FILENAME,
        checkpoint=checkpoint,
        require_accelerator=False,
        distributed=evidence,
        verify_live_environment=False,
    )


def _rewrite_resume_receipt(checkpoint: Path, mutate: object) -> None:
    receipt = checkpoint / resume_fidelity.RECEIPT_FILENAME
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    mutate(payload)
    payload["receipt_sha256"] = freezer.sha256_json(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    receipt.chmod(0o640)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    receipt.chmod(0o440)


def test_run_binding_resume_and_completion_are_campaign_bound(
    tmp_path: Path, sealed_run_cleanup: list[Path]
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    task = "tsp"
    run = output / "runs" / task / "seed42"
    sealed_run_cleanup.append(run)
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": task,
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    run_contract = run / validator.RUN_CONTRACT_FILENAME
    assert run_contract.is_file()
    assert run_contract.stat().st_mode & 0o222 == 0
    run_contract_payload = json.loads(run_contract.read_text(encoding="utf-8"))
    authority_contract = Path(run_contract_payload["authoritative_path"])
    assert authority_contract.parent == Path(assets["authority_root"])
    assert run_contract.read_bytes() == authority_contract.read_bytes()
    assert (
        run_contract_payload["remote_environment_identity_sha256"]
        == assets["payload"]["remote_environment"]["identity_sha256"]
    )
    assert (
        run_contract_payload["remote_environment_receipt_sha256"]
        == assets["payload"]["remote_environment"]["receipt_sha256"]
    )
    assert (
        run_contract_payload["remote_environment_receipt_file_sha256"]
        == assets["payload"]["remote_environment"]["receipt_file_sha256"]
    )

    _hyperparams(run, task=task, base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint_5000 = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))
    assert validator.resolve_state(
        **kwargs, initialize_run_binding=False
    ) == f"RESUME:{checkpoint_5000}"

    checkpoint_final = _single_h100_checkpoint_prefix(
        run, through=31_250, base=Path(assets["base"])
    )
    _adapter(run, Path(assets["base"]))
    (run / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "completed": True,
                "has_saved_model": True,
                "global_step": 31_250,
                "max_steps": 31_250,
                "latest_checkpoint_dir": str(checkpoint_final),
            }
        ),
        encoding="utf-8",
    )
    _seal_completed_fixture(run, Path(assets["base"]))
    assert validator.resolve_state(**kwargs, initialize_run_binding=False) == f"COMPLETE:{run}"
    (run / "stage_status.json").chmod(0o640)
    with pytest.raises(ValueError, match="terminal or stray root artifacts"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


def test_run_contract_cannot_be_rebound_to_another_campaign(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "mis" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "mis",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    contract = run / validator.RUN_CONTRACT_FILENAME
    contract.chmod(0o640)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["campaign_id"] = "different-campaign"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    contract.chmod(0o440)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


@pytest.mark.parametrize("mutation", ["writable_root", "changed_contract", "extra_entry"])
def test_sealed_five_run_authority_rejects_live_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    root = Path(assets["authority_root"])
    if mutation == "writable_root":
        root.chmod(0o750)
    elif mutation == "changed_contract":
        contract = root / "tsp.seed42.run_contract.json"
        contract.chmod(0o640)
        contract.write_bytes(contract.read_bytes() + b"changed")
        contract.chmod(0o440)
    else:
        root.chmod(0o750)
        (root / "extra.json").write_text("{}\n", encoding="utf-8")
        (root / "extra.json").chmod(0o440)
        root.chmod(0o550)
    with pytest.raises(ValueError):
        freezer.load_campaign(Path(assets["campaign"]))


def test_authority_publish_does_not_overwrite_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path, publish=False)
    authority = Path(assets["authority_root"])
    original = freezer._rename_noreplace_at

    def race_destination(**kwargs: object) -> None:
        os.mkdir(str(kwargs["destination_name"]), mode=0o700, dir_fd=kwargs["parent_descriptor"])
        original(**kwargs)

    monkeypatch.setattr(freezer, "_rename_noreplace_at", race_destination)
    with pytest.raises(ValueError, match="cannot atomically publish"):
        freezer._write_create_once(Path(assets["campaign"]), assets["payload"])
    assert authority.is_dir()
    assert list(authority.iterdir()) == []


def test_authority_publish_rejects_parent_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = tmp_path / "original"
    case.mkdir()
    assets = _frozen_campaign(case, publish=False)
    moved = tmp_path / "moved"
    original = freezer._require_canonical_directory_identity

    def relocate_parent(path: Path, identity: tuple[int, int], *, label: str) -> None:
        case.rename(moved)
        case.mkdir()
        original(path, identity, label=label)

    monkeypatch.setattr(freezer, "_require_canonical_directory_identity", relocate_parent)
    with pytest.raises(ValueError, match="no longer names the pinned directory"):
        freezer._write_create_once(Path(assets["campaign"]), assets["payload"])
    assert not Path(assets["authority_root"]).exists()
    assert not (moved / "base_sft_authority").exists()


def test_authority_publish_rejects_final_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path, publish=False)
    authority = Path(assets["authority_root"])
    displaced = authority.with_name("displaced_authority")
    original = freezer._verify_canonical_published_tree

    def swap_final(**kwargs: object) -> None:
        final = Path(kwargs["final"])
        final.rename(displaced)
        final.mkdir(mode=0o550)
        original(**kwargs)

    monkeypatch.setattr(freezer, "_verify_canonical_published_tree", swap_final)
    try:
        with pytest.raises(ValueError):
            freezer._write_create_once(Path(assets["campaign"]), assets["payload"])
    finally:
        for root in (authority, displaced):
            if root.exists():
                root.chmod(0o750)
                for candidate in root.iterdir():
                    candidate.chmod(0o640)
    assert authority.is_dir()
    assert list(authority.iterdir()) == []
    assert displaced.is_dir()


def test_any_checkpoint_with_missing_serialized_bias_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "tsp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "tsp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(
        run,
        task="tsp",
        base=Path(assets["base"]),
        data_root=Path(assets["data_root"]),
    )
    checkpoint = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))

    def reject_adapter(path: Path) -> dict[str, object]:
        if path == checkpoint:
            raise ValueError("serialized adapter tensor inventory mismatch: missing bias")
        return {}

    monkeypatch.setattr(validator, "audit_adapter_persistence", reject_adapter)
    with pytest.raises(ValueError, match="missing bias"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


def test_any_resumable_checkpoint_requires_live_publication_fidelity(tmp_path: Path) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "cvrp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "cvrp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(
        run,
        task="cvrp",
        base=Path(assets["base"]),
        data_root=Path(assets["data_root"]),
    )
    checkpoint = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))
    checkpoint.chmod(0o750)
    (checkpoint / "checkpoint_fidelity.json").unlink()
    with pytest.raises(ValueError, match="lacks live publication checkpoint fidelity"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


@pytest.mark.parametrize(
    "artifact",
    [
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ],
)
def test_resume_receipt_rejects_every_changed_resume_component(
    tmp_path: Path,
    artifact: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "tsp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "tsp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="tsp", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))
    target = checkpoint / artifact
    target.chmod(0o640)
    target.write_bytes(target.read_bytes() + b"tamper")
    target.chmod(0o440)
    with pytest.raises(ValueError):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


@pytest.mark.parametrize("mutation", ["extra", "missing_receipt", "writable_mode"])
def test_resume_receipt_rejects_inventory_and_mode_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "mis" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "mis",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="mis", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))
    if mutation == "extra":
        checkpoint.chmod(0o750)
        (checkpoint / "unbound.pt").write_bytes(b"extra")
        (checkpoint / "unbound.pt").chmod(0o440)
        checkpoint.chmod(0o550)
    elif mutation == "missing_receipt":
        checkpoint.chmod(0o750)
        (checkpoint / single_h100_resume.RECEIPT_FILENAME).unlink()
        checkpoint.chmod(0o550)
    else:
        (checkpoint / "optimizer.pt").chmod(0o640)
    with pytest.raises(ValueError):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


@pytest.mark.parametrize("mutation", ["wrong_step", "world_size_mismatch", "environment_mismatch"])
def test_resume_receipt_rejects_rehashed_semantic_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "op" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "op",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="op", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint = _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))

    def mutate(payload: dict[str, object]) -> None:
        if mutation == "wrong_step":
            payload["global_step"] = 501
        elif mutation == "world_size_mismatch":
            payload["world_size"] = 4
        else:
            payload["environment"]["identity_sha256"] = "f" * 64

    _rewrite_single_h100_resume_receipt(checkpoint, mutate)
    with pytest.raises(ValueError):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


def test_newer_incomplete_checkpoint_forbids_fallback_to_sealed_older_state(
    tmp_path: Path,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "jssp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "jssp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="jssp", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    _single_h100_checkpoint(run, step=5_000, base=Path(assets["base"]))
    (run / "checkpoints" / "checkpoint-10000").mkdir()
    with pytest.raises(ValueError, match="missing"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


@pytest.mark.parametrize(
    "missing_receipt",
    [
        "adapter_persistence_receipt.json",
        "checkpoint_fidelity.json",
        "sft_reload_equivalence.json",
    ],
)
def test_complete_root_requires_bound_read_only_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_run_cleanup: list[Path],
    missing_receipt: str,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "op" / "seed42"
    sealed_run_cleanup.append(run)
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "op",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(
        run,
        task="op",
        base=Path(assets["base"]),
        data_root=Path(assets["data_root"]),
    )
    checkpoint = _single_h100_checkpoint_prefix(
        run, through=31_250, base=Path(assets["base"])
    )
    _adapter(run, Path(assets["base"]))
    (run / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "completed": True,
                "has_saved_model": True,
                "global_step": 31_250,
                "max_steps": 31_250,
                "latest_checkpoint_dir": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    (run / missing_receipt).unlink()
    _seal_completed_fixture(run, Path(assets["base"]))

    def require_receipt(path: Path, **_kwargs: object) -> dict[str, object]:
        if not path.is_file():
            raise ValueError("adapter persistence receipt is not a regular file")
        return {}

    monkeypatch.setattr(validator, "validate_audit_receipt", require_receipt)
    with pytest.raises(ValueError, match="terminal or stray root artifacts"):
        validator.resolve_state(**kwargs, initialize_run_binding=False)


def test_model_registry_freezer_requires_and_binds_completed_base_campaign(
    tmp_path: Path,
    sealed_run_cleanup: list[Path],
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    specialist_args: list[str] = []
    for task in freezer.TASKS:
        run = output / "runs" / task / "seed42"
        sealed_run_cleanup.append(run)
        kwargs = {
            "campaign_path": Path(assets["campaign"]),
            "immutable_input_trust_root": Path(assets["trust_root"]),
            "source_root": Path(assets["source"]),
            "base_model_path": Path(assets["base"]),
            "forbidden_source_registry": Path(assets["registry"]),
            "remote_environment_receipt": Path(assets["environment"]),
            "data_root": Path(assets["data_root"]),
            "output_root": output,
            "task": task,
            "run_dir": run,
        }
        assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
        _hyperparams(
            run,
            task=task,
            base=Path(assets["base"]),
            data_root=Path(assets["data_root"]),
        )
        checkpoint = _single_h100_checkpoint_prefix(
            run, through=31_250, base=Path(assets["base"])
        )
        _adapter(run, Path(assets["base"]))
        (run / "stage_status.json").write_text(
            json.dumps(
                {
                    "stage": "sft",
                    "completed": True,
                    "has_saved_model": True,
                    "global_step": 31_250,
                    "max_steps": 31_250,
                    "latest_checkpoint_dir": str(checkpoint),
                }
            ),
            encoding="utf-8",
        )
        _seal_completed_fixture(run, Path(assets["base"]))
        specialist_args.extend(
            [
                "--specialist",
                f"{task},{run},{Path(assets['data_root']) / task / 'train'},42",
            ]
        )
    destination = tmp_path / "model_initialization_registry.json"
    argv = [
        "--base-model-path",
        str(assets["base"]),
        "--forbidden-source-registry",
        str(assets["registry"]),
        "--base-sft-campaign",
        str(assets["campaign"]),
        *specialist_args,
        "--frozen-at-utc",
        "2026-08-14T01:00:00Z",
        "--output",
        str(destination),
    ]
    assert model_registry_freezer.main(argv) == 0
    frozen = json.loads(destination.read_text(encoding="utf-8"))
    assert frozen["model_family"] == "Qwen2.5-7B"
    for task in freezer.TASKS:
        paths = {
            item["path"]
            for item in frozen["specialist_sft_by_task"][task]["initial_model"]["files"]
        }
        assert validator.RUN_CONTRACT_FILENAME in paths
        assert "adapter_persistence_receipt.json" in paths
        assert "sft_reload_equivalence.json" in paths
        assert "completed_sft_run_seal.json" in paths
        assert (
            frozen["specialist_sft_by_task"][task]["reload_equivalence"]["status"]
            == "exact_post_save_reload_verified"
        )


def test_new_launchers_are_isolated_from_historical_instruct_lane() -> None:
    worker = (SCRIPTS / "train_sft500k_base_specialist_h100.slurm").read_text(encoding="utf-8")
    submitter = (SCRIPTS / "submit_sft500k_base_specialists_h100.sh").read_text(encoding="utf-8")
    for required in (
        "BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT:?",
        "BASE_SFT_CAMPAIGN_CONTRACT:?",
        "BASE_SFT_FORBIDDEN_SOURCE_REGISTRY:?",
        "BASE_SFT_RESERVED_SOURCE_REGISTRY:?",
        "BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT:?",
        "BASE_SFT_BASE_MODEL_PATH:?",
        "BASE_SFT_DATA_ROOT:?",
        "BASE_SFT_OUTPUT_ROOT:?",
    ):
        assert required in worker
        assert required in submitter
    assert "Qwen2.5-7B-Instruct" not in worker
    assert "Qwen2.5-7B-Instruct" not in submitter
    assert "jiangmatched_task_sft500k_20260713_v2" not in worker
    assert "jiangmatched_task_sft500k_20260713_v2" not in submitter
    assert "use the separate dev smoke gate first" in submitter
    assert '[[ "${LAUNCH_PROFILE}" == "regular20h_iclr" ]]' in submitter
    assert "TASKS=(cvrp jssp mis op tsp)" in submitter
    assert "BOUNDARIES=(5000 10000 15000 20000 25000 30000 31250)" in submitter
    assert "WALL_TIME=20:00:00" in submitter
    assert "QOS=qos_gpu_h100-t4" in submitter
    assert '--dependency="afterok:${previous_job}"' in submitter
    assert '--time="${WALL_TIME}"' in submitter
    assert '--qos="${QOS}"' in submitter
    assert "BASE_SFT_PYTHON:?" in submitter
    assert submitter.startswith("#!/bin/bash -l\n")
    assert 'module load "${GPU_ARCH_MODULE:-arch/h100}"' in submitter
    assert 'conda activate "${SEALED_PREFIX}"' in submitter
    assert submitter.index('conda activate "${SEALED_PREFIX}"') < submitter.index(
        'identity="$("${SEALED_PYTHON}"'
    )
    assert "BASE_SFT_CONDA_ENV:?" in worker
    assert 'mktemp -d "/tmp/csft_iclr_${SLURM_JOB_ID:?}.XXXXXX"' in worker
    assert 'export TMPDIR="${node_tmpdir}"' in worker
    assert 'trap \'rm -rf -- "${node_tmpdir}"\' EXIT' in worker
    assert '${CACHE_ROOT}/tmp' not in worker
    assert "BASE_SFT_CONDA_ENV=${SEALED_PREFIX}" in submitter
    assert '--export="CLASP_PROJECT_ROOT=' in submitter
    assert "--export=ALL" not in submitter
    assert "export CLASP_LORA_BIAS=all" in worker
    assert "export CLASP_PUBLICATION_FIDELITY=1" in worker
    assert "clasp_jiang_f638_qwen25_base_sft500k_iclr" in worker
    assert '[[ "${OUTPUT_ROOT}" == *_iclr* ]]' in worker
    assert '[[ "${campaign_id}" == *_iclr* ]]' in worker
    assert "BASE_SFT_LAUNCH_PROFILE=${LAUNCH_PROFILE}" in submitter
    assert "#SBATCH --account=vas@h100" in worker
    assert "#SBATCH --constraint=h100" in worker
    assert "#SBATCH --gres=gpu:1" in worker
    assert "#SBATCH --time=20:00:00" in worker
    assert "#SBATCH --qos=qos_gpu_h100-t4" in worker
    assert "scripts/audit_sft_reload_equivalence.py" in worker
    assert "scripts/seal_completed_sft_run.py" in worker
    assert '--output "${RUN_DIR}/sft_reload_equivalence.json"' in worker
    assert "BASE_SFT_TARGET_STEP=${target_step}" in submitter
    assert "scripts/base_sft_single_h100_resume_fidelity.py" in worker
    assert '--stop-after-step "${TARGET_STEP}"' in worker
    assert '--resume-from-checkpoint "${RESUME_CHECKPOINT}"' in worker
    recovery_start = worker.index("# Complete certification interrupted")
    state_resolution = worker.index('state="$(')
    recovery_reload = worker.index(
        "scripts/audit_sft_reload_equivalence.py",
        recovery_start,
        state_resolution,
    )
    recovery_seal = worker.index("seal_completed_run", recovery_reload, state_resolution)
    assert recovery_start < recovery_reload < recovery_seal < state_resolution
    assert '[[ ! -e "${RUN_DIR}/sft_reload_equivalence.json"' in worker
    assert worker.count('"${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py') == 2
    assert worker.count('"${SEALED_PYTHON}" scripts/seal_completed_sft_run.py') == 1
    assert 'TOKEN_AUDIT_RECEIPT="${DATA_ROOT}/sft_token_length_audit.json"' in submitter
    assert 'CLEANLINESS_AUDIT_RECEIPT="${DATA_ROOT}/sft_cleanliness_audit.json"' in submitter
    assert worker.count("--require-live-accelerator") >= 3
    assert "BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT=${REMOTE_ENVIRONMENT_RECEIPT}" in submitter
    assert "BASE_SFT_RESERVED_SOURCE_REGISTRY=${RESERVED_SOURCE_REGISTRY}" in submitter
    assert "BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT=${IMMUTABLE_INPUT_TRUST_ROOT}" in submitter
    assert "--require-matrix-quota" in submitter
    assert "--require-cell-quota" in worker
    assert "RESUME_FROM_CHECKPOINT" in worker
    train_sft = (
        ROOT / "src" / "clasp" / "entrypoints" / "train_sft_jiang_f638.py"
    ).read_text(
        encoding="utf-8"
    )
    assert "trainer.train()" in train_sft
    train_tree = ast.parse(train_sft)
    assert any(
        isinstance(node, ast.keyword) and node.arg == "resume_from_checkpoint"
        for node in ast.walk(train_tree)
    )
    assert 'LORA_BIAS = "all"' in train_sft
    assert "PER_DEVICE_TRAIN_BATCH_SIZE = 4" in train_sft
    assert "GRADIENT_ACCUMULATION_STEPS = 4" in train_sft
    assert "adapter_persistence_receipt.json" in worker
    assert 'export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"' in worker
    assert "${PYTHONPATH:+" not in worker
    assert "scripts/run_jiang_f638_sft_in_sealed_environment.py" in worker
    first_gate = worker.index("validate_live_inputs\n")
    recovery = worker.index("# Complete certification interrupted")
    state_gate = worker.index('state="$("${SEALED_PYTHON}"')
    assert first_gate < recovery < state_gate
    training_step = worker.index('"${SRUN_BIN}" --ntasks=1')
    final_gate = worker.index("validate_live_inputs", training_step)
    final_audit = worker.index('"${SEALED_PYTHON}" scripts/audit_lora_persistence.py', final_gate)
    final_reload = worker.index(
        '"${SEALED_PYTHON}" scripts/audit_sft_reload_equivalence.py', final_audit
    )
    final_seal = worker.index("seal_completed_run", final_reload)
    final_validator = worker.index('final_state="$(')
    assert training_step < final_gate < final_audit < final_reload < final_seal < final_validator
    assert worker.count("validate_live_inputs") == 3
    assert "torch.distributed.run" not in worker
    historical = (SCRIPTS / "train_sft500k_specialist_a100.slurm").read_text(encoding="utf-8")
    assert "Qwen2.5-7B-Instruct" in historical
    assert 'case "${TASK}" in op|mis|jssp)' in historical


def _install_fake_rank_zero_collectives(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str] | None = None,
    remote_error_rank: int | None = None,
) -> None:
    import torch.distributed as dist

    log = events if events is not None else []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", lambda: log.append("barrier"))

    def gather(output: list[object], local: dict[str, object]) -> None:
        log.append("all_gather")
        for rank in range(4):
            envelope = dict(local)
            envelope["rank"] = rank
            if remote_error_rank == rank:
                envelope = {"rank": rank, "ok": False, "error": "injected rank error"}
            output[rank] = envelope

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda _value, src: log.append(f"broadcast:{src}")
    )


def test_rank_local_resume_error_is_gathered_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_fake_rank_zero_collectives(monkeypatch, events=events)

    def fail_local() -> dict[str, object]:
        events.append("local")
        raise OSError("rank-local read failed")

    with pytest.raises(ValueError, match="rank0=OSError: rank-local read failed"):
        resume_fidelity._rank_synchronized_operation(
            "resume test",
            fail_local,
            require_equal_evidence=True,
        )
    assert events == ["local", "all_gather"]


def test_remote_rank_resume_error_is_raised_by_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rank_zero_collectives(monkeypatch, remote_error_rank=2)
    with pytest.raises(ValueError, match="rank2=injected rank error"):
        resume_fidelity._rank_synchronized_operation(
            "resume test",
            lambda: {"ok": "local"},
            require_equal_evidence=True,
        )


def test_publication_callback_failure_is_synchronized_before_resume_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_fake_rank_zero_collectives(monkeypatch, events=events)
    trainer = Namespace()
    callback = resume_fidelity.SynchronizedBaseSFTPublicationCallback(trainer)

    class FailingPublication:
        def on_save(self, *_args: object, **_kwargs: object) -> None:
            events.append("publication")
            raise OSError("receipt write failed")

    callback._publication = FailingPublication()
    with pytest.raises(ValueError, match="receipt write failed"):
        callback.on_save(
            Namespace(output_dir=str(tmp_path)),
            Namespace(global_step=500),
            object(),
        )
    assert events == ["publication", "all_gather"]


def test_resume_callback_barrier_precedes_rank_zero_checkpoint_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_fake_rank_zero_collectives(monkeypatch, events=events)
    monkeypatch.setenv(resume_fidelity.ENABLE_ENV, "1")
    trainer = Namespace(_clasp_exact_peft_kl_reference=None)
    callback = resume_fidelity.SynchronizedBaseSFTPublicationCallback(trainer)
    callback._publication = Namespace(on_save=lambda *_args, **_kwargs: object())

    def fail_prepare(_checkpoint: Path) -> None:
        events.append("prepare")
        assert events[-2] == "barrier"
        raise OSError("injected prepare failure")

    monkeypatch.setattr(resume_fidelity, "_prepare_checkpoint_for_seal", fail_prepare)
    with pytest.raises(ValueError, match="durable-save preparation failed"):
        callback.on_save(
            Namespace(output_dir=str(tmp_path)),
            Namespace(global_step=500),
            object(),
        )
    assert events[:3] == ["all_gather", "barrier", "prepare"]
    assert events[-1] == "broadcast:0"


def test_resume_callback_installation_binds_retention_and_data_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rank_zero_collectives(monkeypatch)
    monkeypatch.setenv(resume_fidelity.ENABLE_ENV, "1")
    monkeypatch.setenv(resume_fidelity.EXPECTED_MAX_STEPS_ENV, "31250")
    callbacks: list[object] = []
    args = Namespace(
        save_strategy="steps",
        save_steps=500,
        save_total_limit=64,
        save_only_model=False,
        save_safetensors=True,
        ignore_data_skip=False,
        max_steps=-1,
    )
    trainer = Namespace(
        args=args,
        add_callback=callbacks.append,
        _clasp_exact_peft_kl_reference=None,
    )
    assert resume_fidelity.install_base_sft_resume_fidelity_callback(trainer) is True
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], resume_fidelity.SynchronizedBaseSFTPublicationCallback)
    args.ignore_data_skip = True
    with pytest.raises(ValueError, match="ignore_data_skip=False"):
        resume_fidelity.install_base_sft_resume_fidelity_callback(trainer)
    args.ignore_data_skip = False
    args.save_total_limit = 5
    with pytest.raises(ValueError, match="save_total_limit=64"):
        resume_fidelity.install_base_sft_resume_fidelity_callback(trainer)


def test_sixth_checkpoint_is_not_rotated_under_bound_limit(tmp_path: Path) -> None:
    from transformers import Trainer

    output = tmp_path / "rotation"
    output.mkdir()
    for step in range(500, 3_001, 500):
        (output / f"checkpoint-{step}").mkdir()
    trainer = object.__new__(Trainer)
    trainer.args = Namespace(save_total_limit=64, output_dir=str(output))
    trainer.state = Namespace(best_model_checkpoint=None)
    trainer._rotate_checkpoints(output_dir=str(output))
    assert sorted(path.name for path in output.iterdir()) == [
        "checkpoint-1000",
        "checkpoint-1500",
        "checkpoint-2000",
        "checkpoint-2500",
        "checkpoint-3000",
        "checkpoint-500",
    ]


def test_semantic_tensor_hash_supports_scalars() -> None:
    import torch

    digest = resume_fidelity._semantic_sha256(torch.tensor(7.0))
    assert isinstance(digest, str) and len(digest) == 64


def test_resume_pin_keeps_exact_inode_and_rng_after_canonical_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    import numpy as np
    import torch

    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "tsp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "tsp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="tsp", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint = _checkpoint(run, step=500, base=Path(assets["base"]))
    monkeypatch.setenv(resume_fidelity.ENABLE_ENV, "1")

    def local_collective(
        _label: str,
        operation: object,
        *,
        require_equal_evidence: bool,
    ) -> list[dict[str, object]]:
        del require_equal_evidence
        evidence = operation()
        return [evidence, evidence, evidence, evidence]

    monkeypatch.setattr(resume_fidelity, "_rank_synchronized_operation", local_collective)

    def verified_identity(
        _checkpoint: Path,
        *,
        checkpoint_fd: int | None = None,
    ) -> dict[str, object]:
        assert checkpoint_fd is not None
        payload = resume_fidelity._checkpoint_payload_fd(checkpoint_fd)
        observed = os.fstat(checkpoint_fd)
        identity: dict[str, object] = {
            "checkpoint_device": observed.st_dev,
            "checkpoint_inode": observed.st_ino,
            "receipt_sha256": payload["receipt_sha256"],
            "checkpoint_tree_sha256": payload["checkpoint_tree"]["tree_sha256"],
            "global_step": payload["global_step"],
        }
        identity["identity_sha256"] = freezer.sha256_json(identity)
        return identity

    monkeypatch.setattr(
        resume_fidelity,
        "verify_resume_before_trainer_load",
        verified_identity,
    )

    def load_rng(path: str) -> None:
        expected = torch.load(
            Path(path) / "rng_state_0.pth",
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(expected["python"])
        np.random.set_state(expected["numpy"])
        torch.random.set_rng_state(expected["cpu"])

    trainer = Namespace(_load_rng_state=load_rng)
    proc_path = resume_fidelity.pin_verified_resume_for_trainer(trainer, checkpoint)
    retained = checkpoint.with_name("retained-checkpoint-500")
    checkpoint.rename(retained)
    checkpoint.mkdir()
    (checkpoint / "attacker.txt").write_text("replacement", encoding="utf-8")
    try:
        trainer._load_rng_state(proc_path)
        assert trainer._clasp_base_sft_rng_restored is True
        assert trainer._clasp_base_sft_rng_restore_calls == 1
        assert (
            os.fstat(trainer._clasp_base_sft_pinned_resume["descriptor"]).st_ino
            == retained.stat().st_ino
        )
    finally:
        resume_fidelity.release_pinned_resume_for_trainer(trainer)


def test_post_load_audit_matches_full_adapter_optimizer_scheduler_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "mis" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "mis",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="mis", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    checkpoint = _checkpoint(run, step=500, base=Path(assets["base"]))
    descriptor = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    payload = resume_fidelity._checkpoint_payload_fd(descriptor)
    checkpoint_receipt = resume_fidelity._bound_adapter_fidelity_identity(descriptor)[1]
    optimizer_state = resume_fidelity._torch_load_fd(
        descriptor,
        "optimizer.pt",
        weights_only=True,
    )
    scheduler_state = resume_fidelity._torch_load_fd(
        descriptor,
        "scheduler.pt",
        weights_only=True,
    )

    class FixtureOptimizer:
        def state_dict(self) -> dict[str, object]:
            return optimizer_state

    class OptimizerWrapper:
        def __init__(self) -> None:
            self.optimizer = FixtureOptimizer()

        def state_dict(self) -> dict[str, object]:
            return self.optimizer.state_dict()

    class FixtureScheduler:
        def state_dict(self) -> dict[str, object]:
            return scheduler_state

    class SchedulerWrapper:
        def __init__(self) -> None:
            self.scheduler = FixtureScheduler()

        def state_dict(self) -> dict[str, object]:
            return self.scheduler.state_dict()

    trainer = Namespace(
        model=object(),
        optimizer=OptimizerWrapper(),
        lr_scheduler=SchedulerWrapper(),
        _clasp_base_sft_pinned_resume={
            "descriptor": descriptor,
            "receipt": payload,
        },
    )
    monkeypatch.setenv(resume_fidelity.EXPECTED_MAX_STEPS_ENV, "31250")
    monkeypatch.setattr(
        resume_fidelity,
        "audit_trainable_state_closure",
        lambda _model: checkpoint_receipt["trainable_closure"],
    )
    monkeypatch.setattr(
        resume_fidelity,
        "EXPECTED_OPTIMIZER_CLASS",
        f"{FixtureOptimizer.__module__}.{FixtureOptimizer.__qualname__}",
    )
    monkeypatch.setattr(
        resume_fidelity,
        "EXPECTED_SCHEDULER_CLASS",
        f"{FixtureScheduler.__module__}.{FixtureScheduler.__qualname__}",
    )
    monkeypatch.setattr(
        resume_fidelity,
        "_rank_synchronized_operation",
        lambda _label, operation, *, require_equal_evidence: [operation()] * 4,
    )
    callback = resume_fidelity.BaseSFTResumeFidelityCallback(trainer)
    try:
        callback.on_train_begin(
            Namespace(),
            Namespace(global_step=500, max_steps=31_250),
            object(),
        )
        assert trainer._clasp_base_sft_loaded_state_verified is True
    finally:
        os.close(descriptor)


def test_resume_seal_fsyncs_receipt_root_and_checkpoint_parent_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _frozen_campaign(tmp_path)
    output = Path(assets["output_root"])
    output.mkdir()
    run = output / "runs" / "cvrp" / "seed42"
    kwargs = {
        "campaign_path": Path(assets["campaign"]),
        "immutable_input_trust_root": Path(assets["trust_root"]),
        "source_root": Path(assets["source"]),
        "base_model_path": Path(assets["base"]),
        "forbidden_source_registry": Path(assets["registry"]),
        "remote_environment_receipt": Path(assets["environment"]),
        "data_root": Path(assets["data_root"]),
        "output_root": output,
        "task": "cvrp",
        "run_dir": run,
    }
    assert validator.resolve_state(**kwargs, initialize_run_binding=True) == "FRESH"
    _hyperparams(run, task="cvrp", base=Path(assets["base"]), data_root=Path(assets["data_root"]))
    events: list[tuple[str, int]] = []
    original_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = "unknown"
        events.append((target, stat.S_IMODE(os.fstat(descriptor).st_mode)))
        original_fsync(descriptor)

    monkeypatch.setattr(resume_fidelity.os, "fsync", tracked_fsync)
    checkpoint = _checkpoint(run, step=500, base=Path(assets["base"]))
    receipt = str(checkpoint / resume_fidelity.RECEIPT_FILENAME)
    root = str(checkpoint)
    parent = str(checkpoint.parent)
    receipt_index = max(index for index, item in enumerate(events) if item == (receipt, 0o440))
    root_index = max(index for index, item in enumerate(events) if item == (root, 0o550))
    parent_index = max(index for index, item in enumerate(events) if item[0] == parent)
    assert receipt_index < root_index < parent_index


def test_pinned_peft_trainer_checkpoint_inventory_matches_publication_allowlist(
    tmp_path: Path,
) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "local_tsp_sft500k_seed42_instruct",
        local_files_only=True,
    )
    tokenizer.chat_template = None
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="all",
            task_type="CAUSAL_LM",
        ),
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "checkpoints"),
        per_device_train_batch_size=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=64,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, processing_class=tokenizer)
    trainer.create_optimizer_and_scheduler(num_training_steps=31_250)
    trainer.state.global_step = 500
    trainer.state.max_steps = 31_250
    trainer._save_checkpoint(model, None)
    checkpoint = tmp_path / "checkpoints" / "checkpoint-500"
    observed = {candidate.name for candidate in checkpoint.iterdir()}
    expected = (
        resume_fidelity.REQUIRED_CHECKPOINT_FILES
        - {"checkpoint_fidelity.json", *(f"rng_state_{rank}.pth" for rank in range(4))}
    ) | {"rng_state.pth"}
    assert observed == expected


def test_real_peft_trainer_resumes_through_retained_fd_after_path_swap(
    tmp_path: Path,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            tokens = torch.tensor([1, 2 + index, 3, 4], dtype=torch.long)
            return {
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "labels": tokens.clone(),
            }

    torch.manual_seed(123)
    model = get_peft_model(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
            )
        ),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="all",
            task_type="CAUSAL_LM",
        ),
    )
    first_output = tmp_path / "first"
    first = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(first_output),
            per_device_train_batch_size=1,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=64,
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=TinyDataset(),
    )
    first.train()
    checkpoint = first_output / "checkpoint-1"
    descriptor = os.open(
        checkpoint,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    retained = first_output / "retained-checkpoint-1"
    checkpoint.rename(retained)
    checkpoint.mkdir()
    (checkpoint / "attacker.txt").write_text("replacement", encoding="utf-8")
    proc_path = f"/proc/self/fd/{descriptor}"
    try:
        second = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(tmp_path / "second"),
                per_device_train_batch_size=1,
                max_steps=2,
                save_strategy="steps",
                save_steps=1,
                save_total_limit=64,
                report_to=[],
                disable_tqdm=True,
            ),
            train_dataset=TinyDataset(),
        )
        result = second.train(resume_from_checkpoint=proc_path)
        assert result.global_step == 2
        optimizer_identity = resume_fidelity._optimizer_state_identity(second.optimizer)
        assert optimizer_identity["wrapper_state_exact"] is True
        assert optimizer_identity["unwrapped_class"] == "torch.optim.adamw.AdamW"
    finally:
        os.close(descriptor)
    assert (checkpoint / "attacker.txt").read_text(encoding="utf-8") == "replacement"
