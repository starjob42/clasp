from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import sys
import types
from typing import Any
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_clean_sft500k_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_clean_sft500k_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    identity = builder.file_identity(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def _replacement_shard(
    tmp_path: Path, *, task: str, rows: list[dict[str, object]]
) -> tuple[Path, Path]:
    shard = tmp_path / f"{task}-replacement-shard"
    native_root = shard / "native_evidence"
    native_root.mkdir(parents=True)
    replacement = shard / "replacements.json"
    _write(replacement, rows)
    evidence: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        row_dir = native_root / f"attempt-{index + 1:06d}-candidate-{index:09d}"
        row_dir.mkdir()
        replay = row_dir / "strict_replay.json"
        replay.write_text(json.dumps({"strict": True}), encoding="utf-8")
        evidence.append(
            {
                "row_index": index,
                "semantic_instance_sha256": builder.sha256_json(row["semantic"]),
                "demonstration_sha256": builder.sha256_json(row),
                "native_status": "TEST_COMPLETE",
                "strict_replay": _relative_identity(replay, shard),
            }
        )
    evidence_manifest = shard / "evidence_manifest.json"
    evidence_manifest.write_text(json.dumps(evidence), encoding="utf-8")
    source_identity = builder.file_identity(SCRIPT)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": builder.REPLACEMENT_PROTOCOL,
        "status": "strict_solver_backed_replacement_shard_complete",
        "task": task,
        "rows": len(rows),
        "unique_semantic_instances": len(rows),
        "output": _relative_identity(replacement, shard),
        "evidence_manifest": _relative_identity(evidence_manifest, shard),
        "solver_identity": {"name": "test"},
        "solver_parameters": {"deterministic": True},
        "implementation": {
            "producer": source_identity,
            "generator": source_identity,
            "reference_runner": source_identity,
            "strict_verifier": builder.implementation_identity(),
        },
    }
    payload["receipt_sha256"] = builder.sha256_json(payload)
    receipt = shard / "replacement_receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    for path in sorted(shard.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o550 if path.is_dir() else 0o440)
    shard.chmod(0o550)
    return replacement, receipt


def _reserved_sources(
    tmp_path: Path, *, sft_eval_semantic: str = "reserved-sft-eval"
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for source_class in builder.RESERVED_SOURCE_CLASSES:
        path = tmp_path / f"{source_class}.json"
        semantic = (
            sft_eval_semantic
            if source_class == "sft_eval"
            else f"reserved-{source_class}"
        )
        _write(path, [_row(semantic, "reserved-label")])
        result.append((source_class, path))
    return result


def _row(semantic: str, output: str, *, valid: bool = True) -> dict[str, object]:
    return {
        "instruction": "solve",
        "input": semantic,
        "output": output,
        "semantic": semantic,
        "valid": valid,
    }


def _transform(_task: str, row: object) -> dict[str, object]:
    assert isinstance(row, dict)
    return dict(row)


def _semantic(_task: str, row: object) -> str:
    assert isinstance(row, dict)
    return str(row["semantic"])


def _strict(_task: str, row: object) -> dict[str, object]:
    assert isinstance(row, dict)
    if not row["valid"]:
        raise builder.CleanSFTError("invalid label")
    return {"absolute_objective_error": 0.0}


def _merged_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Path]]:
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 1)
    data_root = tmp_path / "data_sft"
    receipt_dir = data_root / "cleanliness_receipts"
    receipt_paths: dict[str, Path] = {}
    for task in builder.TASKS:
        task_root = tmp_path / f"fixture-{task}"
        task_root.mkdir()
        source = task_root / "source.json"
        output = data_root / task / "train" / f"{task}.json"
        _write(source, [_row(f"source-{task}", f"label-{task}")])
        receipt = builder.audit_or_build(
            task=task,
            source_path=source,
            reserved_sources=_reserved_sources(task_root),
            output_path=output,
            expected_rows=1,
            transform=_transform,
            forbidden_transform=_transform,
            semantic_hasher=_semantic,
            strict_verifier=_strict,
        )
        receipt_path = receipt_dir / f"{task}.json"
        builder._write_json_new(receipt_path, receipt)
        receipt_paths[task] = receipt_path
    merged = builder.merge_task_receipts(
        data_root=data_root,
        task_receipt_paths=[receipt_paths[task] for task in builder.TASKS],
        deep=True,
    )
    merged_path = data_root / builder.MERGED_RECEIPT_FILENAME
    builder._write_json_new(merged_path, merged)
    return data_root, merged_path, receipt_paths


def test_streaming_parser_handles_cross_chunk_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    rows = [_row("s1", "x" * 140), _row("s2", "y")]
    _write(path, rows)
    assert list(builder.iter_json_array(path, chunk_size=64)) == list(enumerate(rows))


def test_builder_retains_alternative_labels_and_replaces_union(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    rows = [
        _row("s1", "label-a"),
        _row("s1", "label-b"),  # same instance, distinct valid demonstration
        _row("s2", "label-c"),
        _row("s2", "label-c"),  # exact demonstration duplicate
        _row("reserved", "leaked"),
        _row("bad", "malformed", valid=False),
    ]
    _write(source, rows)
    reserved_sources = _reserved_sources(tmp_path, sft_eval_semantic="reserved")
    replacements, replacement_receipt = _replacement_shard(
        tmp_path,
        task="cvrp",
        rows=[_row("s3", "new-3"), _row("s4", "new-4"), _row("s5", "new-5")],
    )

    receipt = builder.audit_or_build(
        task="cvrp",
        source_path=source,
        reserved_sources=reserved_sources,
        replacement_paths=[replacements],
        replacement_receipt_paths=[replacement_receipt],
        output_path=output,
        expected_rows=6,
        transform=_transform,
        forbidden_transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )

    clean = json.loads(output.read_text(encoding="utf-8"))
    assert [row["output"] for row in clean[:3]] == ["label-a", "label-b", "label-c"]
    assert len(clean) == 6
    assert receipt["accepted_source_rows"] == 3
    assert receipt["replacement_rows"] == 3
    assert receipt["retained_multi_reference_instances"] == 1
    assert receipt["retained_extra_multi_reference_rows"] == 1
    assert receipt["reason_counts"]["exact_demonstration_duplicate"] == 1
    assert receipt["reason_counts"]["reserved_evaluation_collision"] == 1
    assert receipt["reason_counts"]["strict_label_rejection"] == 1
    assert receipt["reason_counts"]["rejected_union"] == 3
    assert receipt["status"] == "clean_exact_500k_corpus_built"
    assert [
        item["source_class"] for item in receipt["reserved_source_manifest"]
    ] == list(builder.RESERVED_SOURCE_CLASSES)
    assert receipt["reserved_source_manifest_sha256"] == builder.sha256_json(
        receipt["reserved_source_manifest"]
    )
    receipt_path = tmp_path / "clean.receipt.json"
    builder._write_json_new(receipt_path, receipt)
    reopened = builder.validate_task_receipt(
        receipt_path, expected_task="cvrp", expected_rows=6, deep=True
    )
    assert reopened["output_sha256"] == builder.sha256_file(output)


def test_replacement_provenance_rejects_self_hashed_unofficial_protocol(
    tmp_path: Path,
) -> None:
    replacement, receipt_path = _replacement_shard(
        tmp_path, task="mis", rows=[_row("new", "label")]
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["protocol"] = "self-hashed-but-unofficial"
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = builder.sha256_json(payload)
    receipt_path.chmod(0o640)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o440)
    with pytest.raises(builder.CleanSFTError, match="official matching-task shard"):
        builder._replacement_provenance("mis", [replacement], [receipt_path])


def test_audit_only_reports_exact_replacement_deficit(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write(source, [_row("a", "x"), _row("a", "x"), _row("b", "y", valid=False)])
    receipt = builder.audit_or_build(
        task="mis",
        source_path=source,
        forbidden_paths=[],
        expected_rows=3,
        transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )
    assert receipt["clean_rows"] == 1
    assert receipt["additional_replacements_required"] == 2
    assert receipt["status"] == "all_source_rows_audited"


def test_audit_only_can_prove_sealed_replacements_fill_the_deficit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _write(source, [_row("a", "x"), _row("a", "x")])
    replacement, replacement_receipt = _replacement_shard(
        tmp_path, task="cvrp", rows=[_row("b", "y")]
    )
    receipt = builder.audit_or_build(
        task="cvrp",
        source_path=source,
        reserved_sources=_reserved_sources(tmp_path),
        replacement_paths=[replacement],
        replacement_receipt_paths=[replacement_receipt],
        expected_rows=2,
        transform=_transform,
        forbidden_transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )
    assert receipt["status"] == "all_source_rows_audited"
    assert receipt["accepted_source_rows"] == 1
    assert receipt["replacement_rows"] == 1
    assert receipt["additional_replacements_required"] == 0


def test_replacement_must_be_semantically_new(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    _write(source, [_row("a", "one"), _row("bad", "invalid", valid=False)])
    replacements, replacement_receipt = _replacement_shard(
        tmp_path, task="op", rows=[_row("a", "two")]
    )
    with pytest.raises(builder.CleanSFTError, match="provide 1 additional replacements"):
        builder.audit_or_build(
                task="op",
                source_path=source,
                reserved_sources=_reserved_sources(tmp_path),
                forbidden_transform=_transform,
            replacement_paths=[replacements],
            replacement_receipt_paths=[replacement_receipt],
            output_path=output,
            expected_rows=2,
            transform=_transform,
            semantic_hasher=_semantic,
            strict_verifier=_strict,
        )
    assert not output.exists()


def test_zero_replacement_build_is_valid_for_already_clean_task(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    _write(source, [_row("a", "one"), _row("b", "two")])
    receipt = builder.audit_or_build(
        task="tsp",
        source_path=source,
        reserved_sources=_reserved_sources(tmp_path),
        forbidden_transform=_transform,
        output_path=output,
        expected_rows=2,
        transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )
    assert receipt["clean_rows"] == 2
    assert receipt["replacement_rows"] == 0
    assert receipt["replacement_provenance"] == []
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        source.read_text(encoding="utf-8")
    )
    assert output.stat().st_mode & 0o222 == 0


def test_invalid_and_reserved_reasons_are_unioned_once(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    forbidden = tmp_path / "reserved.json"
    _write(source, [_row("both", "bad", valid=False), _row("ok", "good")])
    _write(forbidden, [_row("both", "unlabeled-placeholder")])
    receipt = builder.audit_or_build(
        task="op",
        source_path=source,
        forbidden_paths=[forbidden],
        expected_rows=2,
        transform=_transform,
        forbidden_transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )
    assert receipt["reason_counts"]["reserved_evaluation_collision"] == 1
    assert receipt["reason_counts"]["strict_label_rejection"] == 1
    assert receipt["reason_counts"]["rejected_union"] == 1


def test_cli_parser_smoke_accepts_zero_replacement_build(tmp_path: Path) -> None:
    argv = [
        "--task",
        "jssp",
        "--source-json",
        str(tmp_path / "source.json"),
        "--output-json",
        str(tmp_path / "output.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    ]
    for source_class, path in _reserved_sources(tmp_path):
        argv.extend(("--reserved-source", f"{source_class},{path}"))
    args = builder.parse_args(argv)
    assert args.task == "jssp"
    assert args.replacement_json == []
    assert [source_class for source_class, _path in args.reserved_source] == list(
        builder.RESERVED_SOURCE_CLASSES
    )


def test_cli_parser_accepts_replacements_in_audit_only_mode(tmp_path: Path) -> None:
    replacement = tmp_path / "replacements.json"
    replacement_receipt = tmp_path / "replacement_receipt.json"
    args = builder.parse_args(
        [
            "--task",
            "mis",
            "--source-json",
            str(tmp_path / "source.json"),
            "--replacement-json",
            str(replacement),
            "--replacement-receipt",
            str(replacement_receipt),
            "--receipt",
            str(tmp_path / "audit.json"),
        ]
    )
    assert args.output_json is None
    assert args.replacement_json == [replacement]
    assert args.replacement_receipt == [replacement_receipt]


def test_publication_build_fails_closed_without_four_labeled_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    _write(source, [_row("a", "one")])
    with pytest.raises(builder.CleanSFTError, match="four labeled reserved sources"):
        builder.audit_or_build(
            task="tsp",
            source_path=source,
            output_path=output,
            expected_rows=1,
            transform=_transform,
            semantic_hasher=_semantic,
            strict_verifier=_strict,
        )
    assert not output.exists()


def test_publication_reserved_sources_require_unique_files(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    shared = tmp_path / "shared.json"
    _write(source, [_row("a", "one")])
    _write(shared, [_row("reserved", "label")])
    with pytest.raises(builder.CleanSFTError, match="distinct canonical files"):
        builder.audit_or_build(
            task="tsp",
            source_path=source,
            reserved_sources=[
                (source_class, shared)
                for source_class in builder.RESERVED_SOURCE_CLASSES
            ],
            output_path=tmp_path / "clean.json",
            expected_rows=1,
            transform=_transform,
            semantic_hasher=_semantic,
            strict_verifier=_strict,
        )


def test_task_receipt_reopens_and_binds_every_reserved_source(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    receipt_path = tmp_path / "clean.receipt.json"
    reserved_sources = _reserved_sources(tmp_path)
    _write(source, [_row("a", "one"), _row("b", "two")])
    receipt = builder.audit_or_build(
        task="tsp",
        source_path=source,
        reserved_sources=reserved_sources,
        output_path=output,
        expected_rows=2,
        transform=_transform,
        forbidden_transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=_strict,
    )
    builder._write_json_new(receipt_path, receipt)
    identity = builder.validate_task_receipt(
        receipt_path, expected_task="tsp", expected_rows=2, deep=True
    )
    assert identity["reserved_source_manifest"] == receipt["reserved_source_manifest"]
    assert identity["reserved_source_manifest_sha256"] == receipt[
        "reserved_source_manifest_sha256"
    ]

    first_reserved = reserved_sources[0][1]
    first_reserved.chmod(0o640)
    _write(first_reserved, [_row("changed", "changed")])
    with pytest.raises(builder.CleanSFTError, match="live identity mismatch"):
        builder.validate_task_receipt(
            receipt_path, expected_task="tsp", expected_rows=2, deep=False
        )


def test_merge_requires_exact_task_receipt_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _merged_path, receipt_paths = _merged_fixture(tmp_path, monkeypatch)
    cvrp_path = receipt_paths["cvrp"]
    relocated = tmp_path / "relocated-cvrp.json"
    relocated.write_bytes(cvrp_path.read_bytes())
    relocated.chmod(0o440)
    supplied = [
        relocated if task == "cvrp" else receipt_paths[task]
        for task in builder.TASKS
    ]
    with pytest.raises(builder.CleanSFTError, match="must be exactly"):
        builder.merge_task_receipts(
            data_root=data_root, task_receipt_paths=supplied, deep=False
        )


@pytest.mark.parametrize("failure_mode", ["tampered", "missing", "relocated"])
def test_merged_reopen_requires_live_bound_task_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    data_root, merged_path, receipt_paths = _merged_fixture(tmp_path, monkeypatch)
    identity = builder.validate_merged_receipt(
        merged_path, expected_data_root=data_root
    )
    expected_cvrp_path = data_root / "cleanliness_receipts" / "cvrp.json"
    assert identity["task_receipt_canonical_path"]["cvrp"] == str(
        expected_cvrp_path.resolve()
    )

    cvrp_path = receipt_paths["cvrp"]
    if failure_mode == "tampered":
        payload = json.loads(cvrp_path.read_text(encoding="utf-8"))
        payload["created_at_utc"] = "2099-01-01T00:00:00Z"
        payload.pop("receipt_sha256")
        payload["receipt_sha256"] = builder.sha256_json(payload)
        cvrp_path.chmod(0o640)
        cvrp_path.write_text(json.dumps(payload), encoding="utf-8")
        cvrp_path.chmod(0o440)
        expected_error = "file hash mismatch"
    elif failure_mode == "missing":
        cvrp_path.unlink()
        expected_error = "not a regular file"
    else:
        cvrp_path.rename(tmp_path / "moved-cvrp.json")
        expected_error = "not a regular file"
    with pytest.raises(builder.CleanSFTError, match=expected_error):
        builder.validate_merged_receipt(merged_path, expected_data_root=data_root)


def test_float_display_tolerance_tracks_printed_precision() -> None:
    assert builder._float_display_tolerance("10.00") == pytest.approx(0.005000011)
    assert builder._float_display_tolerance("1.2e2") == pytest.approx(5.00000001)


def test_jssp_relabel_changes_only_stale_terminal_makespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def evaluator(
        _instruction: str,
        _input_text: str,
        _completion: str,
        _instance: object,
        _reference: str,
    ) -> dict[str, object]:
        return {
            "is_feasible": True,
            "constraint_fraction": 1.0,
            "objective_value": 3.0,
        }

    monkeypatch.setitem(
        sys.modules,
        "clasp.entrypoints.evaluate",
        types.SimpleNamespace(TASK_EVALUATORS={"jssp": evaluator}),
    )
    transformed = {
        "instruction": "solve",
        "input": "instance",
        "instance_json": [[0, 2, 1, 1], [1, 2, 0, 1]],
        "output": "Schedule: [[0, 1], [1, 0]], Makespan: 4\n",
    }
    original_error = builder.CleanSFTError(
        "declared integer objective 4.0 != recomputed 3.0"
    )
    repaired, evidence = builder.default_jssp_terminal_relabel(
        transformed, original_error=original_error
    )
    assert repaired["output"] == "Schedule: [[0, 1], [1, 0]], Makespan: 3\n"
    assert {key: value for key, value in repaired.items() if key != "output"} == {
        key: value for key, value in transformed.items() if key != "output"
    }
    assert evidence["old_makespan"] == 4
    assert evidence["new_makespan"] == 3
    assert evidence["strict_replay"]["constraint_fraction"] == 1.0


def test_jssp_relabel_rejects_infeasible_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "clasp.entrypoints.evaluate",
        types.SimpleNamespace(
            TASK_EVALUATORS={
                "jssp": lambda *_args: {
                    "is_feasible": False,
                    "constraint_fraction": 0.5,
                    "objective_value": 3.0,
                }
            }
        ),
    )
    transformed = {
        "instruction": "solve",
        "input": "instance",
        "instance_json": [[0, 2]],
        "output": "Schedule: [[0]], Makespan: 4",
    }
    with pytest.raises(builder.CleanSFTError, match="not strictly feasible"):
        builder.default_jssp_terminal_relabel(
            transformed,
            original_error=builder.CleanSFTError(
                "declared integer objective 4.0 != recomputed 3.0"
            ),
        )


def test_audit_pipeline_writes_and_binds_repaired_jssp_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "clean.json"
    receipt_path = tmp_path / "clean.receipt.json"
    _write(source, [_row("jssp-instance", "Makespan: 4")])

    def strict(_task: str, row: object) -> dict[str, object]:
        assert isinstance(row, dict)
        if row["output"] != "Makespan: 3":
            raise builder.CleanSFTError(
                "declared integer objective 4.0 != recomputed 3.0"
            )
        return {
            "declared_objective": 3.0,
            "recomputed_objective": 3.0,
            "absolute_objective_error": 0.0,
            "declared_objective_tolerance": 1e-9,
            "constraint_fraction": 1.0,
        }

    def relabeler(
        transformed: dict[str, object], *, original_error: Exception
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert "declared integer objective" in str(original_error)
        repaired = dict(transformed)
        repaired["output"] = "Makespan: 3"
        return repaired, {
            "old_makespan": 4,
            "new_makespan": 3,
            "terminal_numeric_span": [10, 11],
            "before_output_sha256": builder.hashlib.sha256(
                str(transformed["output"]).encode()
            ).hexdigest(),
            "after_output_sha256": builder.hashlib.sha256(b"Makespan: 3").hexdigest(),
            "before_demonstration_sha256": builder.demonstration_sha256(transformed),
            "after_demonstration_sha256": builder.demonstration_sha256(repaired),
            "strict_replay": strict("jssp", repaired),
        }

    receipt = builder.audit_or_build(
        task="jssp",
        source_path=source,
        reserved_sources=_reserved_sources(tmp_path),
        output_path=output,
        expected_rows=1,
        transform=_transform,
        forbidden_transform=_transform,
        semantic_hasher=_semantic,
        strict_verifier=strict,
        repair_jssp_terminal_makespan=True,
        jssp_relabeler=relabeler,
    )
    assert json.loads(output.read_text(encoding="utf-8"))[0]["output"] == "Makespan: 3"
    repairs = builder.validate_source_repairs(
        receipt["source_repairs"], task="jssp", source_rows=1
    )
    assert repairs["repaired_rows"] == 1
    assert receipt["accepted_source_rows"] == 1
    assert receipt["reason_counts"] == {}

    # Deep receipt reopening must reproduce the claimed transformation from the
    # original source and reject any unrecorded source/output mutation.
    monkeypatch.setattr(builder, "default_transform", _transform)
    monkeypatch.setattr(builder, "default_semantic_hash", _semantic)
    monkeypatch.setattr(builder, "default_strict_verify", strict)
    monkeypatch.setattr(builder, "default_jssp_terminal_relabel", relabeler)
    builder._write_json_new(receipt_path, receipt)
    identity = builder.validate_task_receipt(
        receipt_path, expected_task="jssp", expected_rows=1, deep=True
    )
    assert identity["output_sha256"] == builder.sha256_file(output)

    tampered_output = tmp_path / "tampered-clean.json"
    tampered_row = json.loads(output.read_text(encoding="utf-8"))[0]
    tampered_row["input"] = "changed-instance"
    _write(tampered_output, [tampered_row])
    with pytest.raises(builder.CleanSFTError, match="changed a non-output field"):
        builder.deep_validate_jssp_source_repairs(
            repairs, source_path=source, output_path=tampered_output
        )


def test_unlabeled_reserved_projection_does_not_require_output() -> None:
    row = {
        "instruction": "solve",
        "input": "instance text",
        "instance_json": [[0, 0], [1, 1]],
    }
    assert builder.default_forbidden_transform("tsp", row) == row


def _load_selected_source_functions(
    path: Path,
    names: set[str],
    namespace: dict[str, object],
) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    compiled = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    exec(compiled, namespace)
    return namespace


@pytest.mark.parametrize(
    ("output", "expected_error"),
    [
        ("Route: [0, 1], Objective: 7", None),
        ("Route: [1, 0], Objective: 7", "shared strict verifier rejected"),
        ("Route: [0, 1], Objective: 8", "declared integer objective"),
    ],
)
def test_lightweight_op_transform_and_evaluator_equal_current_shared_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    expected_error: str | None,
) -> None:
    del tmp_path
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from clasp.rewards import co_tasks

    loader_namespace = _load_selected_source_functions(
        ROOT / "src" / "clasp" / "data" / "loaders.py",
        {"augment_input_for_task", "transform_example_for_task"},
        {},
    )
    evaluator_namespace = _load_selected_source_functions(
        ROOT / "src" / "clasp" / "entrypoints" / "evaluate.py",
        {"_objective_gap", "_evaluate_op"},
        {
            "Any": Any,
            "TASK_DIRECTIONS": {"op": "max"},
            "_combined_context": co_tasks._combined_context,
            "_parse_op_solution": co_tasks._parse_op_solution,
            "_op_meta": co_tasks._op_meta,
            "_op_prize": co_tasks._op_prize,
            "_check_op_constraints": co_tasks._check_op_constraints,
        },
    )
    row = {
        "instruction": "Collect prizes without exceeding the route budget.",
        "input": "Two OP nodes.",
        "output": output,
        "instance_json": [[[0.0, 0.0], [1.0, 0.0]], [0, 7], 2.0],
    }
    shared_transform = loader_namespace["transform_example_for_task"]
    reference_transformed = shared_transform(
        "op",
        row["instruction"],
        row["input"],
        row["output"],
        row["instance_json"],
    )
    lightweight_transformed = builder.default_transform("op", row)
    assert lightweight_transformed == reference_transformed

    shared_evaluator = evaluator_namespace["_evaluate_op"]
    reference_metrics = shared_evaluator(
        lightweight_transformed["instruction"],
        lightweight_transformed["input"],
        output,
        lightweight_transformed["instance_json"],
        output,
    )
    assert builder._default_op_evaluator(
        lightweight_transformed["instruction"],
        lightweight_transformed["input"],
        output,
        lightweight_transformed["instance_json"],
        output,
    ) == reference_metrics
    if expected_error is None:
        strict = builder.default_strict_verify("op", lightweight_transformed)
        assert strict["recomputed_objective"] == 7.0
        assert strict["constraint_fraction"] == 1.0
    else:
        with pytest.raises(builder.CleanSFTError, match=expected_error):
            builder.default_strict_verify("op", lightweight_transformed)


def test_op_transform_and_strict_replay_do_not_import_heavy_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    # Preload the intentionally lightweight shared reward implementation before
    # installing the import guard; production has already loaded it through the
    # semantic benchmark module before the first solve.
    import clasp.rewards.co_tasks  # noqa: F401

    blocked_prefixes = (
        "clasp.data.loaders",
        "clasp.entrypoints.evaluate",
        "peft",
        "transformers",
    )
    for module_name in list(sys.modules):
        if module_name.startswith(blocked_prefixes):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith(blocked_prefixes):
            raise AssertionError(f"OP strict replay attempted heavyweight import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    row = {
        "instruction": "Collect prizes.",
        "input": "Two OP nodes.",
        "output": "Route: [0, 1], Objective: 7",
        "instance_json": [[[0.0, 0.0], [1.0, 0.0]], [0, 7], 2.0],
    }
    transformed = builder.default_transform("op", row)
    strict = builder.default_strict_verify("op", transformed)
    assert strict["recomputed_objective"] == 7.0
    assert not any(name.startswith(blocked_prefixes) for name in sys.modules)
