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
SCRIPT = SCRIPTS / "audit_sft_token_lengths.py"
SPEC = importlib.util.spec_from_file_location("audit_sft_token_lengths", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
token_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = token_audit
SPEC.loader.exec_module(token_audit)


class CharacterTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.encode_calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        self.calls.append(dict(kwargs))
        return {"input_ids": [[ord(character) for character in text] for text in texts]}

    def encode(self, text: str, **kwargs: object) -> list[int]:
        self.encode_calls.append((text, dict(kwargs)))
        return [ord(character) for character in text]


def test_registry_loader_rejects_writable_receipt(tmp_path: Path) -> None:
    path = tmp_path / "base_sft_source_registry.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o640)
    with pytest.raises(token_audit.SFTTokenLengthAuditError, match="read-only"):
        token_audit._load_registry_identity(path)


def _source_binding(root: Path, rows: int) -> dict[str, object]:
    return {
        "canonical_path": str(root.resolve()),
        "source_identity_sha256": "1" * 64,
        "file_manifest_sha256": "2" * 64,
        "rows": rows,
        "unique_instance_hashes": rows,
    }


def _passing_task_record(binding: dict[str, object], rows: int) -> dict[str, object]:
    return {
        "source_binding": dict(binding),
        "row_count": rows,
        "prompt_token_max": 2,
        "completion_token_min": 1,
        "completion_token_max": 1,
        "full_token_min": 3,
        "full_token_max": 3,
        "full_token_p50": 3,
        "full_token_p95": 3,
        "full_token_p99": 3,
        "full_token_length_histogram": [[3, rows]],
        "at_max_seq_length_count": 0,
        "over_max_seq_length_count": 0,
        "prompt_completion_alignment_mismatch_count": 0,
        "zero_supervised_completion_count": 0,
        "ordered_row_length_evidence_sha256": "3" * 64,
    }


def test_streaming_parser_handles_rows_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    rows = [
        {"instruction": "solve", "input": "x" * 90, "output": "ok"},
        {"instruction": "solve", "input": "y", "output": "done"},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert list(token_audit._iter_json_array(path, chunk_size=64)) == list(enumerate(rows))


def test_all_row_audit_uses_f638_marker_mask_and_fails_immediately_on_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tsp" / "train"
    source.mkdir(parents=True)
    path = source / "rows.json"
    rows = [{"text": "abc"}, {"text": "abcdefgh"}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    file_record = {
        "relative_path": "rows.json",
        "sha256": token_audit._sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(rows),
    }
    monkeypatch.setattr(
        token_audit,
        "_render_training_row",
        lambda _task, row, *, eos_token, tokenizer: (
            token_audit.RESPONSE_HEADER + ":" + str(row["text"]) + eos_token
        ),
    )
    tokenizer = CharacterTokenizer()
    record = token_audit.audit_task(
        task="tsp",
        source_binding=_source_binding(source, len(rows)),
        files=[("rows.json", path, file_record)],
        tokenizer=tokenizer,
        eos_token="!",
        tokenizer_identity_sha256="4" * 64,
        max_seq_length=26,
        batch_size=1,
    )
    assert record["row_count"] == 2
    assert sum(count for _length, count in record["full_token_length_histogram"]) == 2
    assert record["prompt_token_max"] == len(token_audit.RESPONSE_HEADER)
    assert record["completion_token_min"] == 5
    assert record["over_max_seq_length_count"] == 0
    assert tokenizer.calls
    assert all(call["add_special_tokens"] is True for call in tokenizer.calls)
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert tokenizer.encode_calls == [(token_audit.RESPONSE_HEADER, {"add_special_tokens": False})]

    with pytest.raises(
        token_audit.SFTTokenLengthAuditError,
        match=r"tsp/rows\.json:1: 26 tokens exceed max_seq_length=25",
    ):
        token_audit.audit_task(
            task="tsp",
            source_binding=_source_binding(source, len(rows)),
            files=[("rows.json", path, file_record)],
            tokenizer=CharacterTokenizer(),
            eos_token="!",
            tokenizer_identity_sha256="4" * 64,
            max_seq_length=25,
            batch_size=1,
        )


def test_row_rendering_uses_only_raw_f638_fields_and_preserves_mis_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from clasp.data.prompts import build_jiang_f638_sft_text

    tokenizer = CharacterTokenizer()
    row = {
        "instruction": " Find an independent set. ",
        "input": "Edges: [(0, 1)]\n",
        "output": "Set: [0]",
        # These fields and environment flags affected the generic CLASP path;
        # the Jiang reproduction must ignore both.
        "instance": {"num_nodes": 2, "edges": [[0, 1]]},
    }
    monkeypatch.setenv("CLASP_MIS_JIANG_RESPONSE_OUTPUT", "1")
    monkeypatch.setenv("CLASP_MIS_RELABEL_ORDER", "degree_desc")
    rendered = token_audit._render_training_row(
        "mis",
        row,
        eos_token="<eos>",
        tokenizer=tokenizer,
    )
    expected = build_jiang_f638_sft_text(
        row["instruction"],
        row["input"],
        row["output"],
        eos_token="<eos>",
    )
    assert rendered == expected
    assert rendered.endswith("    ### Response:Set: [0]<eos>")
    assert "### Response:Response:" not in rendered
    assert row["instruction"] in rendered
    assert row["input"] in rendered


@pytest.mark.parametrize(
    ("rendered", "message"),
    [
        ("no response marker", "raw response header is missing"),
        (
            "    ### Response:first     ### Response:second",
            r"raw response header is ambiguous \(2 matches\)",
        ),
        ("    ### Response", "no supervised token follows the Jiang-f638 response header"),
    ],
)
def test_all_row_audit_fails_closed_on_invalid_completion_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rendered: str,
    message: str,
) -> None:
    source = tmp_path / "mis" / "train"
    source.mkdir(parents=True)
    path = source / "rows.json"
    path.write_text(json.dumps([{"unused": True}]), encoding="utf-8")
    file_record = {
        "relative_path": "rows.json",
        "sha256": token_audit._sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": 1,
    }
    monkeypatch.setattr(
        token_audit,
        "_render_training_row",
        lambda _task, _row, *, eos_token, tokenizer: rendered,
    )

    with pytest.raises(token_audit.SFTTokenLengthAuditError, match=message):
        token_audit.audit_task(
            task="mis",
            source_binding=_source_binding(source, 1),
            files=[("rows.json", path, file_record)],
            tokenizer=CharacterTokenizer(),
            eos_token="",
            tokenizer_identity_sha256="4" * 64,
            max_seq_length=20_000,
            batch_size=1,
        )


def test_row_rendering_rejects_missing_released_column() -> None:
    with pytest.raises(
        token_audit.SFTTokenLengthAuditError,
        match=r"missing released dataset fields: \['output'\]",
    ):
        token_audit._render_training_row(
            "tsp",
            {"instruction": "Solve.", "input": "Nodes: []"},
            eos_token="<eos>",
            tokenizer=CharacterTokenizer(),
        )


def test_task_record_requires_exact_prompt_completion_prefix(tmp_path: Path) -> None:
    source = tmp_path / "mis" / "train"
    source.mkdir(parents=True)
    record = {
        "source_binding": _source_binding(source, 1),
        "row_count": 1,
        "prompt_token_max": 2,
        "completion_token_min": 1,
        "completion_token_max": 1,
        "full_token_min": 3,
        "full_token_max": 3,
        "full_token_p50": 3,
        "full_token_p95": 3,
        "full_token_p99": 3,
        "full_token_length_histogram": [[3, 1]],
        "at_max_seq_length_count": 0,
        "over_max_seq_length_count": 0,
        "prompt_completion_alignment_mismatch_count": 1,
        "zero_supervised_completion_count": 0,
        "ordered_row_length_evidence_sha256": "3" * 64,
    }
    with pytest.raises(ValueError, match="fit untruncated"):
        token_audit._validate_task_record(
            record,
            task="mis",
            expected_source=_source_binding(source, 1),
            expected_rows=1,
            max_seq_length=20_000,
        )


def test_five_task_shards_merge_without_retokensizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    tokenizer = tmp_path / "tokenizer"
    shards = tmp_path / "shards"
    data.mkdir()
    tokenizer.mkdir()
    shards.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    artifacts = token_audit.tokenizer_artifact_identity(tokenizer)
    runtime = {
        "tokenizer_class": "test.FastTokenizer",
        "is_fast": True,
        "eos_token": "<eos>",
        "eos_token_id": 1,
        "vocab_size": 10,
    }
    registry: dict[str, object] = {
        "registry_sha256": "5" * 64,
        "source_tree_manifest_sha256": "6" * 64,
        "sources_by_task_label": {},
    }
    bindings: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    sources = registry["sources_by_task_label"]
    assert isinstance(sources, dict)
    for task in token_audit.TASKS:
        source = data / task / "train"
        source.mkdir(parents=True)
        binding = _source_binding(source, token_audit.EXPECTED_ROWS_PER_TASK)
        bindings[task] = binding
        sources[task] = {
            "sft-train": {
                **binding,
                "files": [],
            }
        }
        payload = token_audit.build_task_shard_receipt(
            task=task,
            data_root=data,
            tokenizer_path=tokenizer,
            forbidden_registry_identity=registry,
            source_binding=binding,
            tokenizer_artifacts=artifacts,
            tokenizer_runtime=runtime,
            task_audit=_passing_task_record(binding, token_audit.EXPECTED_ROWS_PER_TASK),
        )
        paths[task] = shards / f"{task}.json"
        token_audit._write_create_once(paths[task], payload)

    monkeypatch.setattr(token_audit, "_load_registry_identity", lambda _path: registry)
    monkeypatch.setattr(
        token_audit,
        "_source_binding",
        lambda *, registry_identity, data_root, task: (bindings[task], []),
    )
    merged = token_audit.merge_task_shards(
        shard_paths=paths,
        data_root=data,
        tokenizer_path=tokenizer,
        forbidden_source_registry=tmp_path / "registry.json",
    )
    assert merged["tasks"] == list(token_audit.TASKS)
    assert merged["total_rows"] == 2_500_000
    assert tuple(merged["task_audits"]) == token_audit.TASKS
