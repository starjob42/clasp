from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clasp.baselines.llmcosolver_f638_public_data import (
    PUBLICATION_TASKS,
    RL_FILES,
    SFT_FILES,
    DriveFile,
    acquire_create_once,
    analyze_identity_projection_v2,
    analyze_json_rows,
    canonical_rendered_input,
    compare_identity_projections_v2,
    iter_json_array,
    seal_directory_tree_read_only,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode()
    path.write_bytes(raw)
    return raw


def test_streaming_parser_hashes_exact_bytes_and_rejects_non_json_trailing_comma(tmp_path):
    path = tmp_path / "rows.json"
    raw = _write_json(path, [{"unicode": "é", "value": 1}, {"value": 2}])
    digest = hashlib.sha256()
    assert list(iter_json_array(path, digest, chunk_size=5)) == [
        {"unicode": "é", "value": 1},
        {"value": 2},
    ]
    assert digest.hexdigest() == hashlib.sha256(raw).hexdigest()

    invalid = tmp_path / "invalid.json"
    invalid.write_text('[{"x": 1}, ]')
    with pytest.raises(ValueError, match="Trailing comma"):
        list(iter_json_array(invalid, hashlib.sha256(), chunk_size=2))


def test_rl_inventory_binds_rows_instances_schema_and_reference_values(tmp_path):
    path = tmp_path / "rl.json"
    rows = [
        {
            "instruction": "solve",
            "input": "first",
            "output": "Route: [0, 1, 0], Objective: 3.50",
            "instance": [[0, 0], [1, 1]],
        },
        {
            "instruction": "solve again",
            "input": "first rendered differently",
            "output": "Schedule: [[0]], Makespan: 9",
            "instance": [[0, 0], [1, 1]],
        },
    ]
    raw = _write_json(path, rows)
    analysis, identities = analyze_json_rows(path, source_kind="rl")
    assert analysis["sha256"] == hashlib.sha256(raw).hexdigest()
    assert analysis["row_count"] == 2
    assert analysis["common_keys"] == ["input", "instance", "instruction", "output"]
    assert analysis["instance_identity"]["unique_count"] == 1
    assert analysis["instance_identity"]["duplicate_count"] == 1
    assert analysis["reference_fields"]["parsed_terminal_objective_count"] == 2
    assert analysis["reference_fields"]["parsed_labels"] == {"makespan": 1, "objective": 1}
    assert analysis["reference_fields"]["parsed_decimal_min"] == "3.50"
    assert analysis["reference_fields"]["parsed_decimal_max"] == "9"
    assert len(identities) == 1


def test_sft_instance_identity_is_input_side_and_not_reference_side(tmp_path):
    path = tmp_path / "sft.json"
    rows = [
        {"instruction": "a", "input": "same", "output": "Objective: 1", "n": "4"},
        {"instruction": "b", "input": "same", "output": "Objective: 2", "n": "4"},
    ]
    _write_json(path, rows)
    analysis, _ = analyze_json_rows(path, source_kind="sft")
    assert analysis["row_identity"]["unique_count"] == 2
    assert analysis["instance_identity"]["unique_count"] == 1
    assert analysis["instance_identity"]["definition"] == (
        "sha256(canonical_json(row excluding instruction and output))"
    )


def test_existing_acquisition_target_is_validated_and_never_replaced(tmp_path, monkeypatch):
    from clasp.baselines import llmcosolver_f638_public_data as public_data

    specification = DriveFile("tsp", "train", "id", 4, "tiny.json", "tsp/tiny.json")
    monkeypatch.setitem(
        public_data.RL_SHA256_BY_PATH,
        specification.relative_path,
        hashlib.sha256(b"good").hexdigest(),
    )
    target = tmp_path / specification.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"good")
    observed, outcome = acquire_create_once(specification, tmp_path)
    assert observed == target
    assert outcome == "validated_existing"
    assert target.read_bytes() == b"good"

    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Existing target"):
        acquire_create_once(specification, tmp_path)


def test_official_item_inventory_is_exactly_the_five_publication_tasks():
    assert len(RL_FILES) == 10
    assert len(SFT_FILES) == 15
    assert {item.task for item in RL_FILES} == set(PUBLICATION_TASKS)
    assert {item.task for item in SFT_FILES} == set(PUBLICATION_TASKS)
    assert len({item.file_id for item in (*RL_FILES, *SFT_FILES)}) == 25
    assert sum(item.byte_length for item in RL_FILES) == 266_602_952


def test_freezer_source_has_create_once_guards_and_no_sft_download_lane():
    source = (ROOT / "scripts/freeze_llmcosolver_f638_public_data.py").read_text()
    module = (ROOT / "src/clasp/baselines/llmcosolver_f638_public_data.py").read_text()
    assert 'output.open("x"' in source
    assert "os.link(temporary, target)" in module
    assert "INSPECTION_ONLY_NO_SFT_DOWNLOADS_PERFORMED" in source


def test_v2_task_identity_ignores_tsp_eval_only_instance_field_but_not_reference(tmp_path):
    train_path = tmp_path / "train.json"
    eval_path = tmp_path / "eval.json"
    different_path = tmp_path / "different.json"
    base = {
        "instruction": "solve",
        "input": "Node 0, coordinates: [0, 0];\n  Node 1, coordinates: [1, 1]",
        "output": "Route: [0, 1, 0], Objective: 2",
        "num_nodes": "2",
    }
    _write_json(train_path, [base])
    _write_json(eval_path, [{**base, "instance": [[0, 0], [1, 1]]}])
    _write_json(
        different_path,
        [{**base, "output": "Route: [0, 1, 0], Objective: 3", "instance": [[0, 0], [1, 1]]}],
    )
    for path in (train_path, eval_path, different_path):
        path.chmod(0o444)

    train = analyze_identity_projection_v2(train_path, task="tsp")
    evaluation = analyze_identity_projection_v2(eval_path, task="tsp")
    different = analyze_identity_projection_v2(different_path, task="tsp")
    same_reference = compare_identity_projections_v2(train, evaluation)
    assert same_reference["shared_unique_instance_count"] == 1
    assert same_reference["shared_instance_with_any_identical_reference_count"] == 1
    assert same_reference["shared_instance_with_disjoint_reference_sets_count"] == 0
    assert same_reference["exact_duplicate_unique_row_count"] == 0
    different_reference = compare_identity_projections_v2(train, different)
    assert different_reference["shared_unique_instance_count"] == 1
    assert different_reference["shared_instance_with_any_identical_reference_count"] == 0
    assert different_reference["shared_instance_with_disjoint_reference_sets_count"] == 1


def test_v2_rendered_input_identity_is_task_prefixed_and_whitespace_canonical():
    decomposed = "Cafe\u0301\nNode 0"
    composed = "Café   Node\t0"
    assert canonical_rendered_input("tsp", decomposed) == canonical_rendered_input("tsp", composed)
    assert canonical_rendered_input("tsp", composed) != canonical_rendered_input("op", composed)


def test_v2_identity_rejects_writable_and_symlink_sources(tmp_path):
    path = tmp_path / "rows.json"
    _write_json(path, [{"input": "x", "output": "Objective: 1"}])
    with pytest.raises(ValueError, match="writable"):
        analyze_identity_projection_v2(path, task="tsp")
    path.chmod(0o444)
    assert analyze_identity_projection_v2(path, task="tsp").summary["row_count"] == 1
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink regular"):
        analyze_identity_projection_v2(link, task="tsp")


def test_read_only_tree_seal_covers_files_and_directories(tmp_path):
    data_root = tmp_path / "data"
    nested = data_root / "task" / "train"
    nested.mkdir(parents=True)
    path = nested / "rows.json"
    path.write_text("[]")
    seal_directory_tree_read_only(data_root)
    try:
        assert path.stat().st_mode & 0o777 == 0o444
        assert nested.stat().st_mode & 0o777 == 0o555
        assert data_root.stat().st_mode & 0o777 == 0o555
    finally:
        for directory in (data_root, data_root / "task", nested):
            directory.chmod(0o755)
        path.chmod(0o644)
