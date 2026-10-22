"""Input-only streaming/projection admission guards before original-corpus scans."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from clasp.baselines.sft_anchor_pilot import graph_semantic_id

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/rloo_sft_anchor_20261013/certify_sft_input_overlap.py"
SPEC = importlib.util.spec_from_file_location("anchor_input_overlap", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(task="op", shift=0):
    instance = [[[0, 0], [3 + shift, 4], [6 + shift, 4]], [0, 7, 5] if task == "op" else [0, 1, 1], 10]
    label = "prize" if task == "op" else "demand"
    text = ";".join(f"Node {i}, coordinates: {point}, {label}: {value}, neighbors: [1: 3.0, 2: 4.0]"
                    for i, (point, value) in enumerate(zip(instance[0], instance[1]))) + "."
    return {"instruction": "input only", "input": text, "instance_json": instance,
            "output": "MUST BE DISCARDED WITHOUT SCORING"}


@pytest.mark.parametrize("task", ("op", "cvrp"))
def test_prompt_projection_matches_exact_frozen_graph_identity(task):
    sample = row(task)
    assert audit.project_graph(task, sample) == graph_semantic_id(task, sample["instance_json"])
    stripped = {key: value for key, value in sample.items() if key != "instance_json"}
    assert audit.project_graph(task, stripped) == audit.project_graph(task, sample)


def test_structured_prompt_mismatch_is_ambiguity():
    sample = row()
    sample["instance_json"][0][1][0] += 1
    with pytest.raises(ValueError, match="actual numeric prompt differ"):
        audit.project_graph("op", sample)


@pytest.mark.parametrize("change", [lambda text: "unparsed " + text,
                                     lambda text: text.replace("Node 1", "Node 2"),
                                     lambda text: text.replace("prize: 7", "prize: NaN"),
                                     lambda text: text.replace("[3, 4]", "[True, 4]")])
def test_ambiguous_numeric_prompt_rejected(change):
    with pytest.raises(ValueError):
        audit.prompt_graph("op", change(row()["input"]))


def test_stream_hashes_all_bytes_across_small_chunks_and_discards_labels(tmp_path):
    path = tmp_path / "train.json"
    raw = (json.dumps([row(), row(shift=1)], ensure_ascii=False) + "\n \n").encode()
    path.write_bytes(raw)
    state = {}
    projected = list(audit.iter_input_rows(path, state, chunk_bytes=64))
    assert len(projected) == 2 and all("output" not in value for _, value in projected)
    assert state["sha256"] == hashlib.sha256(raw).hexdigest() and state["size_bytes"] == len(raw)
    assert state["rows"] == 2 and state["input_snapshot_stable"]


@pytest.mark.parametrize("raw", [b"{}", b"[{]", b"[{},]", b"[{}] extra", b"[{}", b"[1]",
                                 b'[{"input":"a","input":"b"}]'])
def test_malformed_or_ambiguous_json_cannot_be_certified(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        list(audit.iter_input_rows(path, {}, chunk_bytes=64))


def test_zero_overlap_and_match_are_distinguished_with_bound_full_rows(tmp_path):
    path = tmp_path / "train.json"
    raw = json.dumps([row(), row(shift=1)]).encode()
    path.write_bytes(raw)
    binding = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "rows": 2}
    absent = {audit.project_graph("op", row(shift=10)): ["fresh"]}
    assert audit.scan_corpus("op", binding, absent, chunk_bytes=64)["certified_zero_graph_overlap"]
    present = {audit.project_graph("op", row()): ["reused"]}
    result = audit.scan_corpus("op", binding, present, chunk_bytes=64)
    assert not result["certified_zero_graph_overlap"] and result["overlapping_corpus_rows"] == 1
    assert result["matched_target_ids"] == ["reused"]
    with pytest.raises(ValueError, match="full corpus rows/byte SHA"):
        audit.scan_corpus("op", {**binding, "rows": 3}, absent, chunk_bytes=64)


def test_complete_byte_scan_with_projection_ambiguity_is_not_certified(tmp_path):
    path = tmp_path / "train.json"
    bad = row()
    bad["input"] = "unrecognized numerical prompt"
    raw = json.dumps([bad]).encode()
    path.write_bytes(raw)
    binding = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "rows": 1}
    result = audit.scan_corpus("op", binding, {}, chunk_bytes=64)
    assert result["observed"]["rows"] == 1 and result["projection_ambiguities"] == 1
    assert not result["certified_zero_graph_overlap"]
