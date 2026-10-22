from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clasp.data.benchmark import validate_frozen_benchmark_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


freeze = sys.modules.get("freeze_benchmark") or _load_script(
    "freeze_benchmark", ROOT / "scripts" / "freeze_benchmark.py"
)
registry_tool = _load_script(
    "freeze_forbidden_source_registry",
    ROOT / "scripts" / "freeze_forbidden_source_registry.py",
)


def _row(offset: float) -> dict[str, object]:
    return {
        "instruction": "Solve the TSP and return a closed route.",
        "input": "Three nodes.",
        "output": "Route: [0, 1, 2, 0]\nObjective: 3.41421356237",
        "instance_json": [
            [offset, offset],
            [offset + 1.0, offset],
            [offset, offset + 1.0],
        ],
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    labels = ("historical-eval", "rl-train", "sft-train")
    roots: dict[str, Path] = {}
    for label_index, label in enumerate(labels):
        root = tmp_path / "official" / label
        _write_rows(root / "part-000.json", [_row(10.0 + label_index * 10.0)])
        _write_rows(root / "part-001.json", [_row(11.0 + label_index * 10.0)])
        roots[label] = root
    specs = [
        freeze.DatasetSpec("tsp", label, str(roots[label])) for label in labels
    ]
    registry = registry_tool.build_registry(specs, expected_tasks=["tsp"])
    test_path = _write_rows(tmp_path / "iid.json", [_row(100.0)])
    return roots, specs, registry, test_path


def _legacy_iid_generation_receipt(tmp_path: Path) -> Path:
    tasks: dict[str, object] = {}
    for index, task in enumerate(sorted(registry_tool.SUPPORTED_BENCHMARK_TASKS)):
        ordered = [registry_tool.sha256_json({"task": task, "row": 0})]
        tasks[task] = {
            "attempts": 1,
            "distribution_counts": {"fixture": 1},
            "file_sha256": f"{index + 1:x}" * 64,
            "ordered_instance_hashes_sha256": registry_tool.sha256_json(ordered),
            "ordered_instance_sha256": ordered,
            "relative_path": f"{task}/eval/test.json",
            "rows": 1,
            "seed": index + 1,
        }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "instances_generated",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "contains_references": False,
        "publishable": False,
        "generator_code_sha256": "a" * 64,
        "generator_contract_sha256": "b" * 64,
        "runtime_audit": {
            "source_audit": {
                "root": str(tmp_path.resolve()),
                "verified_locally": True,
                "observed_files_sha256": {"generator.py": "c" * 64},
            },
            "version_mismatches": {},
            "versions": {"python": "3.11", "numpy": "2.3", "networkx": "3.5"},
        },
        "tasks": tasks,
    }
    receipt["receipt_sha256"] = registry_tool.sha256_json(receipt)
    path = tmp_path / "reserved_sources_v1" / "iid_confirmation" / "generation_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    path.chmod(0o440)
    return path


def _synthetic_five_task_source_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task_index, task in enumerate(sorted(registry_tool.SUPPORTED_BENCHMARK_TASKS)):
        for label_index, label in enumerate(
            sorted(registry_tool.REQUIRED_FORBIDDEN_SOURCE_LABELS)
        ):
            digest = f"{task_index * 3 + label_index + 1:x}" * 64
            files = [
                {
                    "relative_path": "rows.json",
                    "sha256": digest,
                    "size_bytes": 1,
                    "rows": 1,
                }
            ]
            record: dict[str, object] = {
                "task": task,
                "label": label,
                "registered_source_locator": f"/sealed/{task}/{label}",
                "files": files,
                "file_manifest_sha256": registry_tool.sha256_json(files),
                "rows": 1,
                "unique_instance_hashes": 1,
            }
            record["source_identity_sha256"] = (
                registry_tool.forbidden_source_manifest_identity(record)[
                    "source_identity_sha256"
                ]
            )
            records.append(record)
    return records


def _fake_evaluator(_instruction, _input, _answer, _instance, _completion):
    return {
        "is_feasible": True,
        "ground_truth_objective": 1.0,
        "gap_to_ground_truth": 0.0,
    }


def _freeze_with_registry(
    *,
    test_path: Path,
    specs,
    registry: dict[str, object],
) -> dict[str, object]:
    return freeze.build_manifest(
        test_specs=[freeze.DatasetSpec("tsp", "iid", str(test_path))],
        forbidden_specs=specs,
        forbidden_source_registry=registry,
        publication_mode=False,
        generator_contract={"name": "generator"},
        solver_contract={"name": "solver"},
        minimum_instances=1,
        evaluators={"tsp": _fake_evaluator},
        verifier_identity={"name": "verifier"},
    )


def test_registry_self_hash_and_exact_tree_binding_round_trip(tmp_path: Path) -> None:
    _roots, specs, registry, test_path = _fixture(tmp_path)
    manifest = _freeze_with_registry(
        test_path=test_path,
        specs=specs,
        registry=registry,
    )
    identity = validate_frozen_benchmark_manifest(
        manifest,
        require_generation_receipt=False,
        require_reference_evidence=False,
    )
    assert manifest["benchmark_mode"] == "nonpublication"
    assert identity["forbidden_source_registry_sha256"] == registry[
        "registry_sha256"
    ]
    assert identity["forbidden_source_tree_manifest_sha256"] == registry[
        "source_tree_manifest_sha256"
    ]
    assert identity["forbidden_source_registry_identity"][
        "sources_by_task_label"
    ]["tsp"]["rl-train"]["file_manifest_sha256"]
    with pytest.raises(ValueError, match="nonpublication benchmark"):
        validate_frozen_benchmark_manifest(manifest)


def test_registry_rejects_omitted_and_duplicate_task_label_specs(
    tmp_path: Path,
) -> None:
    _roots, specs, _registry, _test_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly cover every task/label"):
        registry_tool.build_registry(specs[:-1], expected_tasks=["tsp"])
    with pytest.raises(ValueError, match="duplicate forbidden-source registry"):
        registry_tool.build_registry([*specs, specs[0]], expected_tasks=["tsp"])


def test_registry_rejects_symbolic_source_tree(tmp_path: Path) -> None:
    roots, specs, _registry, _test_path = _fixture(tmp_path)
    linked = tmp_path / "linked-rl"
    linked.symlink_to(roots["rl-train"], target_is_directory=True)
    attacked = [spec for spec in specs if spec.label != "rl-train"] + [
        freeze.DatasetSpec("tsp", "rl-train", str(linked))
    ]
    with pytest.raises(ValueError, match="symbolic link"):
        registry_tool.build_registry(attacked, expected_tasks=["tsp"])


def test_freeze_rejects_omitted_registered_tree(tmp_path: Path) -> None:
    _roots, specs, registry, test_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="do not exactly match"):
        _freeze_with_registry(
            test_path=test_path,
            specs=specs[:-1],
            registry=registry,
        )


def test_freeze_rejects_partial_nonempty_decoy_tree(tmp_path: Path) -> None:
    roots, specs, registry, test_path = _fixture(tmp_path)
    partial = freeze.DatasetSpec(
        "tsp",
        "sft-train",
        str(roots["sft-train"] / "part-000.json"),
    )
    attacked = [spec for spec in specs if spec.label != "sft-train"] + [partial]
    with pytest.raises(ValueError, match="do not exactly match"):
        _freeze_with_registry(
            test_path=test_path,
            specs=attacked,
            registry=registry,
        )


def test_freeze_rejects_substituted_registered_file(tmp_path: Path) -> None:
    roots, specs, registry, test_path = _fixture(tmp_path)
    _write_rows(roots["rl-train"] / "part-001.json", [_row(999.0)])
    with pytest.raises(ValueError, match="do not exactly match"):
        _freeze_with_registry(
            test_path=test_path,
            specs=specs,
            registry=registry,
        )


def test_freeze_rejects_duplicate_actual_task_label(tmp_path: Path) -> None:
    _roots, specs, registry, test_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate forbidden source specification"):
        _freeze_with_registry(
            test_path=test_path,
            specs=[*specs, specs[0]],
            registry=registry,
        )


def test_base_sft_registry_has_honest_post_iid_temporal_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _legacy_iid_generation_receipt(tmp_path)
    tasks = sorted(registry_tool.SUPPORTED_BENCHMARK_TASKS)
    records = _synthetic_five_task_source_records()
    monkeypatch.setattr(
        registry_tool,
        "_audit_source_records",
        lambda _specs, *, expected_tasks: (tasks, records),
    )
    payload = registry_tool.build_base_sft_registry(
        [],
        expected_tasks=tasks,
        iid_generation_receipt=receipt,
        created_at_utc="2026-08-21T00:00:00Z",
    )
    identity = registry_tool.validate_base_sft_source_registry(
        payload,
        expected_tasks=tasks,
    )
    assert payload["protocol"] == registry_tool.BASE_SFT_PROTOCOL
    assert payload["status"] == registry_tool.BASE_SFT_STATUS
    assert payload["temporal_claim"] == registry_tool.BASE_SFT_TEMPORAL_CLAIM
    assert payload["pre_iid_forbidden_registry_identity"] is None
    assert (
        identity["iid_generation_receipt"]["provenance_claim"]
        == registry_tool.LEGACY_IID_PROVENANCE_CLAIM
    )
    output = tmp_path / "base_sft_source_registry.json"
    registry_tool._write_create_once_read_only(output, payload)
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        registry_tool._write_create_once_read_only(output, payload)
    with pytest.raises(ValueError, match="protocol is invalid"):
        registry_tool.validate_forbidden_source_registry(
            payload,
            expected_tasks=tasks,
        )

    tampered = json.loads(json.dumps(payload))
    tampered["temporal_claim"] = "frozen_before_confirmation_generation"
    unsigned = dict(tampered)
    unsigned.pop("registry_sha256")
    tampered["registry_sha256"] = registry_tool.sha256_json(unsigned)
    with pytest.raises(ValueError, match="contract is invalid"):
        registry_tool.validate_base_sft_source_registry(
            tampered,
            expected_tasks=tasks,
        )
