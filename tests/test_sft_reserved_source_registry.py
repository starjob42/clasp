from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "freeze_sft_reserved_source_registry",
    SCRIPTS / "freeze_sft_reserved_source_registry.py",
)
assert SPEC is not None and SPEC.loader is not None
registry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = registry
SPEC.loader.exec_module(registry)


def _fixture(tmp_path: Path) -> tuple[Path, list[tuple[str, str, Path]], Path]:
    trust = tmp_path / "trust"
    trust.mkdir()
    sources: list[tuple[str, str, Path]] = []
    iid_tasks: dict[str, object] = {}
    for task_index, task in enumerate(registry.TASKS):
        for class_index, source_class in enumerate(registry.SOURCE_CLASSES):
            parent = trust / registry.RESERVED_ROOT_NAME / source_class / task
            parent.mkdir(parents=True)
            path = parent / "rows.json"
            path.write_text(
                json.dumps([{"task": task, "id": 10 * task_index + class_index}]) + "\n",
                encoding="utf-8",
            )
            path.chmod(0o440)
            sources.append((task, source_class, path))
            if source_class == "iid_confirmation":
                ordered = [registry._sha256_json({"task": task, "row": 0})]
                iid_tasks[task] = {
                    "attempts": 1,
                    "distribution_counts": {"fixture": 1},
                    "file_sha256": registry._stable_file_identity(path, rows=False)[
                        "sha256"
                    ],
                    "ordered_instance_hashes_sha256": registry._sha256_json(ordered),
                    "ordered_instance_sha256": ordered,
                    "relative_path": f"{task}/eval/test.json",
                    "rows": 1,
                    "seed": task_index + 1,
                }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "instances_generated",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "contains_references": False,
        "publishable": False,
        "generator_code_sha256": "1" * 64,
        "generator_contract_sha256": "2" * 64,
        "runtime_audit": {
            "source_audit": {
                "root": str(trust.resolve()),
                "verified_locally": True,
                "observed_files_sha256": {"generator.py": "3" * 64},
            },
            "version_mismatches": {},
            "versions": {"python": "3.11", "numpy": "2.3", "networkx": "3.5"},
        },
        "tasks": iid_tasks,
    }
    receipt["receipt_sha256"] = registry._sha256_json(receipt)
    receipt_path = trust / registry.RESERVED_ROOT_NAME / "iid_confirmation" / "generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o440)
    return trust, sources, receipt_path


def test_freeze_and_reopen_exact_twenty_source_registry(tmp_path: Path) -> None:
    trust, sources, generation = _fixture(tmp_path)
    payload = registry.build_registry(
        trust_root=trust,
        sources=sources,
        iid_generation_receipt=generation,
        frozen_at_utc="2026-08-21T00:00:00Z",
    )
    output = trust / registry.REGISTRY_FILENAME
    registry._write_create_once(output, payload)
    identity = registry.validate_registry(output, expected_trust_root=trust)
    assert len(identity["sources"]) == 20
    assert identity["iid_generation_receipt"]["receipt_sha256"] == json.loads(
        generation.read_text(encoding="utf-8")
    )["receipt_sha256"]
    assert (
        identity["iid_generation_receipt"]["provenance_claim"]
        == registry.LEGACY_IID_PROVENANCE_CLAIM
    )
    assert identity["iid_generation_receipt"]["pre_iid_forbidden_registry_identity"] is None
    by_task = registry.sources_by_task_class(identity)
    assert set(by_task) == set(registry.TASKS)
    assert all(set(by_task[task]) == set(registry.SOURCE_CLASSES) for task in registry.TASKS)


def test_registry_rejects_wrong_class_subtree(tmp_path: Path) -> None:
    trust, sources, generation = _fixture(tmp_path)
    task, source_class, path = sources[0]
    bad = path.parent.parent.parent / "rl_eval" / task / "wrong.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("[{}]\n", encoding="utf-8")
    bad.chmod(0o440)
    altered = [(entry_task, entry_class, bad if index == 0 else entry_path) for index, (entry_task, entry_class, entry_path) in enumerate(sources)]
    with pytest.raises(ValueError, match="outside"):
        registry.build_registry(
            trust_root=trust,
            sources=altered,
            iid_generation_receipt=generation,
        )


def test_live_source_tamper_invalidates_registry(tmp_path: Path) -> None:
    trust, sources, generation = _fixture(tmp_path)
    payload = registry.build_registry(
        trust_root=trust,
        sources=sources,
        iid_generation_receipt=generation,
    )
    output = trust / registry.REGISTRY_FILENAME
    registry._write_create_once(output, payload)
    target = sources[-1][2]
    target.chmod(0o640)
    target.write_text('[{"changed":true}]\n', encoding="utf-8")
    target.chmod(0o440)
    with pytest.raises(ValueError, match="differs"):
        registry.validate_registry(output, expected_trust_root=trust)


def test_portable_identity_rejects_noncanonical_embedded_paths(tmp_path: Path) -> None:
    trust, sources, generation = _fixture(tmp_path)
    payload = registry.build_registry(
        trust_root=trust,
        sources=sources,
        iid_generation_receipt=generation,
    )
    output = trust / registry.REGISTRY_FILENAME
    registry._write_create_once(output, payload)
    identity = registry.validate_registry(output, expected_trust_root=trust)

    tampered = json.loads(json.dumps(identity))
    source = tampered["sources"][0]
    source_path = Path(source["canonical_path"])
    source["canonical_path"] = str(source_path.parent / ".." / source_path.parent.name / source_path.name)
    tampered["source_manifest_sha256"] = registry._sha256_json(tampered["sources"])
    unsigned = dict(tampered)
    unsigned.pop("identity_sha256")
    tampered["identity_sha256"] = registry._sha256_json(unsigned)
    with pytest.raises(ValueError, match="canonical absolute"):
        registry.validate_registry_identity(tampered)

    tampered = json.loads(json.dumps(identity))
    tampered["iid_generation_receipt"]["task_bindings"]["cvrp"][
        "relative_path"
    ] = "../cvrp/rows.json"
    unsigned = dict(tampered)
    unsigned.pop("identity_sha256")
    tampered["identity_sha256"] = registry._sha256_json(unsigned)
    with pytest.raises(ValueError, match="cvrp binding"):
        registry.validate_registry_identity(tampered)


def test_legacy_schema1_receipt_requires_ordered_instance_digest(
    tmp_path: Path,
) -> None:
    _trust, _sources, generation = _fixture(tmp_path)
    receipt = json.loads(generation.read_text(encoding="utf-8"))
    del receipt["tasks"]["cvrp"]["ordered_instance_hashes_sha256"]
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = registry._sha256_json(receipt)
    generation.chmod(0o640)
    generation.write_text(json.dumps(receipt), encoding="utf-8")
    generation.chmod(0o440)
    with pytest.raises(ValueError, match="cvrp task record is noncanonical"):
        registry.iid_generation_receipt_identity(generation)
