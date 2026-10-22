from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json"
V2 = ROOT / "experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_v2_manifest_is_self_hashed_read_only_and_explicitly_supersedes_v1():
    v1 = _load(V1)
    v2 = _load(V2)
    expected = v2["manifest_sha256"]
    v2["manifest_sha256"] = None
    assert hashlib.sha256(_canonical(v2)).hexdigest() == expected
    assert expected == "4b4896ae9d627f8306ca505bb245075d2af784dbe1b420c66cf5cf9fab5af06c"
    assert _sha256(V2) == "6bc23d3cd0686385101b5d0fbbba7a8ee783986b57211958f256b644bc278f28"
    assert stat.S_IMODE(V1.stat().st_mode) == 0o444
    assert stat.S_IMODE(V2.stat().st_mode) == 0o444
    assert v2["supersedes"]["manifest_sha256"] == v1["manifest_sha256"]
    assert "TSP eval" in v2["supersedes"]["reason"]


def test_author_training_union_contamination_is_exact_and_claim_guarded():
    manifest = _load(V2)
    expected = {"cvrp": 299, "jssp": 0, "mis": 300, "op": 299, "tsp": 0}
    for task, count in expected.items():
        matrix = manifest["task_identity_matrices"][task]
        union = matrix["author_training_union_to_evaluation"]["sft_train_union_rl_train__rl_eval"]
        assert union["shared_unique_instance_count"] == count
        assert matrix["splits"]["rl_eval"]["row_count"] == 300
    guards = manifest["classification_guards"]
    assert guards["author_eval_is_held_out"] is False
    assert guards["author_eval_is_leakage_free"] is False
    assert guards["fresh_preregistered_disjoint_iid_required_for_claims"] is True
    assert guards["fresh_iid_is_exact_paper_data_reproduction"] is False


def test_matrices_distinguish_exact_rows_from_same_instance_different_reference():
    matrices = _load(V2)["task_identity_matrices"]
    mis = matrices["mis"]["within_stage"]["rl_train__rl_eval"]
    assert mis["shared_unique_instance_count"] == 300
    assert mis["exact_duplicate_unique_row_count"] == 300
    assert mis["shared_instance_with_any_identical_reference_count"] == 300
    assert mis["shared_instance_with_disjoint_reference_sets_count"] == 0

    cvrp = matrices["cvrp"]["cross_stage"]["sft_train__rl_eval"]
    assert cvrp["shared_unique_instance_count"] == 299
    assert cvrp["exact_duplicate_unique_row_count"] == 0
    assert cvrp["shared_instance_with_any_identical_reference_count"] == 14
    assert cvrp["shared_instance_with_disjoint_reference_sets_count"] == 285
    op = matrices["op"]["cross_stage"]["sft_train__rl_eval"]
    assert op["shared_unique_instance_count"] == 299
    assert op["shared_instance_with_any_identical_reference_count"] == 41
    assert op["shared_instance_with_disjoint_reference_sets_count"] == 258


def test_all_within_and_cross_stage_matrices_are_present():
    for matrix in _load(V2)["task_identity_matrices"].values():
        assert set(matrix["within_stage"]) == {
            "rl_train__rl_eval",
            "sft_train__sft_eval",
        }
        assert set(matrix["cross_stage"]) == {
            "sft_train__rl_train",
            "sft_train__rl_eval",
            "sft_eval__rl_train",
            "sft_eval__rl_eval",
        }
    assert (
        _load(V2)["task_identity_matrices"]["tsp"]["raw_file_equality"]["rl_eval_equals_sft_eval"]
        is True
    )


def test_locally_acquired_rl_tree_is_exact_hash_bound_and_read_only():
    manifest = _load(V2)
    root = ROOT / manifest["rl"]["workspace_root"]
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    total = 0
    for record in manifest["rl"]["files"].values():
        path = ROOT / record["workspace_path"]
        assert path.is_file() and not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_size == record["local_bytes"]
        assert _sha256(path) == record["local_sha256"]
        total += record["local_bytes"]
    assert total == 266_602_952


def test_sft_evidence_is_probable_not_misclassified_as_drive_byte_proven():
    manifest = _load(V2)
    assert manifest["sft"]["classification"] == (
        "PROBABLE_RELEASED_BYTES_SIZE_AND_FILENAME_MATCH_ONLY"
    )
    assert manifest["classification_guards"]["sft_drive_byte_identity_proven"] is False
    assert manifest["sft"]["files"]["tsp/instances.pkl"]["local_availability"] == "MISSING"
    assert "NO_SEPARATE_LICENSE_OBSERVED" in manifest["upstream"]["data_license_scope"]
