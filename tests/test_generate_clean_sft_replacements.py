from __future__ import annotations

import contextlib
import errno
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_clean_sft_replacements.py"
SPEC = importlib.util.spec_from_file_location("generate_clean_sft_replacements", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
producer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producer
SPEC.loader.exec_module(producer)


def _test_solver_identity() -> dict[str, object]:
    return {
        "backend_name": "gurobi",
        "native_version": "13.0.1",
        "distribution_name": "gurobipy",
        "module_origin": str(SCRIPT.resolve(strict=True)),
        "installed_distribution_manifest_sha256": "3" * 64,
        "python": producer._file_record(Path(sys.executable).resolve(strict=True)),
        "support_packages": {
            "numpy": {
                "version": "test-numpy",
                "module": producer._file_record(
                    (ROOT / "scripts" / "generate_iid_benchmark.py").resolve(strict=True)
                ),
            },
            "networkx": {
                "version": "test-networkx",
                "module": producer._file_record(
                    (ROOT / "scripts" / "run_reference_evidence.py").resolve(strict=True)
                ),
            },
        },
        "license_class": "gurobi_academic_site_license_nonredistributable",
        "quality_semantics": "best_known_incumbent",
    }


def _test_implementation_identity() -> dict[str, object]:
    return {
        "producer": producer._file_record(SCRIPT.resolve(strict=True)),
        "generator": producer._file_record(
            (ROOT / "scripts" / "generate_iid_benchmark.py").resolve(strict=True)
        ),
        "reference_runner": producer._file_record(
            (ROOT / "scripts" / "run_reference_evidence.py").resolve(strict=True)
        ),
        "strict_verifier": producer.clean.implementation_identity(),
    }


def _minimal_completed_op_tree(
    root: Path,
    *,
    with_phase_evidence: bool,
    with_full_provenance: bool = True,
) -> None:
    root.mkdir()
    native = root / "native_evidence"
    native.mkdir()
    attempt = native / "attempt-000001-candidate-000000000"
    attempt.mkdir()
    producer._write_bytes_new(attempt / "solver.log", b"sealed solver evidence\n")
    producer._write_json_new(
        attempt / "strict_replay.json",
        {"native_result": {"artifacts": {}}},
    )
    replacements: list[object] = [{"instance_json": [], "output": "test"}]
    evidence_manifest: list[object] = [
        {
            "row_index": 0,
            "semantic_instance_sha256": "1" * 64,
            "demonstration_sha256": "2" * 64,
            "native_status": "OPTIMAL",
            "strict_replay": producer._file_record(
                attempt / "strict_replay.json",
                relative_to=root,
            ),
        }
    ]
    producer._write_json_new(root / "replacements.json", replacements)
    producer._write_json_new(root / "evidence_manifest.json", evidence_manifest)
    receipt: dict[str, object] = {
        "schema_version": producer.SCHEMA_VERSION,
        "protocol": producer.PROTOCOL,
        "status": "strict_solver_backed_replacement_shard_complete",
        "task": "op",
        "attempts": 1,
        "rows": 1,
        "unique_semantic_instances": 1,
        "solver_identity": _test_solver_identity(),
        "output": producer._file_record(root / "replacements.json", relative_to=root),
        "evidence_manifest": producer._file_record(
            root / "evidence_manifest.json",
            relative_to=root,
        ),
    }
    if with_full_provenance:
        receipt["implementation"] = _test_implementation_identity()
    if with_phase_evidence:
        receipt["phase_evidence"] = {
            "protocol": "clasp-op-local-stage-then-atomic-remote-publication-v1",
            "remote_publication_started_at_receipt_creation": False,
        }
    receipt["receipt_sha256"] = producer.clean.sha256_json(receipt)
    producer._write_json_new(root / "replacement_receipt.json", receipt)
    producer._seal_tree(root)


def test_candidate_seed_is_stable_and_task_separated() -> None:
    assert producer._candidate_seed(7, "cvrp", 11) == producer._candidate_seed(
        7, "cvrp", 11
    )
    assert producer._candidate_seed(7, "cvrp", 11) != producer._candidate_seed(
        7, "op", 11
    )
    assert producer._candidate_seed(7, "cvrp", 11) != producer._candidate_seed(
        7, "cvrp", 12
    )


def test_lkh_virtual_depots_are_split_into_closed_routes(tmp_path: Path) -> None:
    tour = tmp_path / "solution.tour"
    tour.write_text(
        "NAME : test\nDIMENSION : 6\nTOUR_SECTION\n1\n2\n4\n3\n5\n6\n-1\nEOF\n",
        encoding="utf-8",
    )
    assert producer._lkh_parse_routes(tour, customers=2) == [
        [0, 1, 0],
        [0, 2, 0],
    ]


def test_cvrp_raw_objective_uses_unrounded_euclidean_distance() -> None:
    row = {"instance_json": [[[0, 0], [1, 1]], [0, 1], 2]}
    assert producer._cvrp_raw_objective(row, [[0, 1, 0]]) == pytest.approx(
        2.0 * (2.0**0.5)
    )


def test_parser_requires_lkh_only_for_cvrp(tmp_path: Path) -> None:
    op_local_temp = tmp_path / "op-local"
    op_local_temp.mkdir(mode=0o700)
    op = producer.parse_args(
        [
            "--task",
            "op",
            "--count",
            "1",
            "--output-dir",
            str(tmp_path / "op"),
            "--op-local-temp-root",
            str(op_local_temp),
        ]
    )
    assert op.lkh_executable is None
    assert op.op_local_temp_root == op_local_temp
    with pytest.raises(SystemExit):
        producer.parse_args(
            ["--task", "op", "--count", "1", "--output-dir", str(tmp_path / "op")]
        )
    with pytest.raises(SystemExit):
        producer.parse_args(
            [
                "--task",
                "cvrp",
                "--count",
                "1",
                "--output-dir",
                str(tmp_path / "cvrp"),
            ]
        )


def test_op_local_temp_root_must_be_private_canonical_and_empty(tmp_path: Path) -> None:
    local_root = tmp_path / "private-op-root"
    local_root.mkdir(mode=0o700)
    assert producer._validated_op_local_temp_root(
        local_root,
        require_empty=True,
    ) == local_root

    marker = local_root / "unexpected"
    marker.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(producer.ReplacementGenerationError, match="initially be empty"):
        producer._validated_op_local_temp_root(local_root, require_empty=True)
    marker.unlink()

    local_root.chmod(0o750)
    with pytest.raises(producer.ReplacementGenerationError, match="exactly 0700"):
        producer._validated_op_local_temp_root(local_root, require_empty=True)
    with pytest.raises(producer.ReplacementGenerationError, match="must be absolute"):
        producer._validated_op_local_temp_root(
            Path("relative-op-root"),
            require_empty=True,
        )


def test_op_local_capacity_preflight_checks_bytes_and_reported_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    four_gibibyte_blocks = types.SimpleNamespace(
        f_frsize=4096,
        f_bsize=4096,
        f_bavail=(4 * 1024 * 1024 * 1024) // 4096,
        f_favail=10_000,
    )
    monkeypatch.setattr(producer.os, "statvfs", lambda _root: four_gibibyte_blocks)
    evidence = producer._require_op_local_capacity(
        tmp_path,
        count=90,
        max_attempt_multiplier=10,
    )
    assert evidence["required_available_bytes"] == 1800 * 1024 * 1024
    assert evidence["required_free_inodes"] == 7328
    assert evidence["maximum_attempts_budgeted"] == 900

    monkeypatch.setattr(
        producer.os,
        "statvfs",
        lambda _root: types.SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_bavail=1,
            f_favail=10_000,
        ),
    )
    with pytest.raises(producer.ReplacementGenerationError, match="byte capacity"):
        producer._require_op_local_capacity(
            tmp_path,
            count=90,
            max_attempt_multiplier=10,
        )

    monkeypatch.setattr(
        producer.os,
        "statvfs",
        lambda _root: types.SimpleNamespace(
            f_frsize=4096,
            f_bsize=4096,
            f_bavail=(4 * 1024 * 1024 * 1024) // 4096,
            f_favail=7327,
        ),
    )
    with pytest.raises(producer.ReplacementGenerationError, match="inode capacity"):
        producer._require_op_local_capacity(
            tmp_path,
            count=90,
            max_attempt_multiplier=10,
        )


def test_create_once_writers_always_flush_and_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync_descriptors: list[int] = []
    monkeypatch.setattr(
        producer.os,
        "fsync",
        lambda descriptor: fsync_descriptors.append(descriptor),
    )

    producer._write_bytes_new(tmp_path / "attempt.bin", b"attempt\n")
    producer._write_json_new(tmp_path / "attempt.json", {"kind": "attempt"})
    assert len(fsync_descriptors) == 2


def test_op_two_row_shard_is_completed_locally_before_hidden_remote_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed_writes: list[Path] = []
    copy_calls: list[tuple[Path, Path]] = []
    deep_validation_calls: list[Path] = []
    original_write = producer._write_bytes_new
    original_copy = producer._copy_sealed_tree_once
    original_deep_validate = producer._deep_validate_copied_op_shard

    def recording_write(path: Path, payload: bytes) -> None:
        observed_writes.append(path)
        original_write(path, payload)

    def recording_copy(
        source: Path,
        destination: Path,
        inventory: dict[str, dict[str, object]],
    ) -> None:
        copy_calls.append((source, destination))
        original_copy(source, destination, inventory)

    def recording_deep_validation(
        root: Path,
        inventory: dict[str, dict[str, object]],
    ) -> None:
        assert stat.S_IMODE(root.stat().st_mode) == 0o550
        deep_validation_calls.append(root)
        original_deep_validate(root, inventory)

    def fake_candidate(
        _task: str,
        *,
        base_seed: int,
        candidate_number: int,
    ) -> dict[str, object]:
        del base_seed
        return {
            "candidate_number": candidate_number,
            "replacement_generation": {"derived_seed": candidate_number},
        }

    def fake_op_solve(
        row: dict[str, object],
        *,
        row_dir: Path,
        local_temp_root: Path,
        timeout_seconds: float,
        work_limit: float,
        node_limit: int,
    ) -> tuple[str, dict[str, object]]:
        del local_temp_root, timeout_seconds, work_limit, node_limit
        producer._write_bytes_new(
            row_dir / "gurobi.log",
            b"fake solver evidence\n",
        )
        if row["candidate_number"] == 0:
            raise producer.ReplacementGenerationError("injected rejected attempt")
        return "Route: [0], Objective: 0", {
            "native_status": "OPTIMAL",
            "native_status_code": 2,
            "native_incumbent": 0,
            "native_best_bound": 0.0,
            "native_gap": 0.0,
            "native_wall_time_seconds": 0.01,
            "artifacts": {},
        }

    monkeypatch.setattr(producer, "_write_bytes_new", recording_write)
    monkeypatch.setattr(producer, "_copy_sealed_tree_once", recording_copy)
    monkeypatch.setattr(
        producer,
        "_deep_validate_copied_op_shard",
        recording_deep_validation,
    )
    monkeypatch.setattr(
        producer,
        "_require_distinct_stage_filesystems",
        lambda _local, _remote: None,
    )
    monkeypatch.setattr(producer, "generate_candidate", fake_candidate)
    monkeypatch.setattr(producer, "solve_op_gurobi_incumbent", fake_op_solve)
    monkeypatch.setattr(
        producer,
        "_solver_identity",
        lambda _task, _args: _test_solver_identity(),
    )
    monkeypatch.setattr(producer, "_solver_parameters", lambda _task, _args: {})
    monkeypatch.setattr(
        producer.clean,
        "fixed_representation_environment",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        producer.clean,
        "default_forbidden_transform",
        lambda _task, row: row,
    )
    monkeypatch.setattr(producer.clean, "default_transform", lambda _task, row: row)
    monkeypatch.setattr(
        producer.clean,
        "default_semantic_hash",
        lambda _task, row: f"{int(row['candidate_number']):064x}",
    )
    monkeypatch.setattr(
        producer.clean,
        "default_strict_verify",
        lambda _task, _row: {"valid": True},
    )
    monkeypatch.setattr(
        producer.clean,
        "demonstration_sha256",
        lambda row: f"{int(row['candidate_number']) + 100:064x}",
    )
    local_temp_root = tmp_path / "op-local"
    local_temp_root.mkdir(mode=0o700)
    destination = tmp_path / "replacement-output" / "op" / "shard-000"
    args = types.SimpleNamespace(
        task="op",
        count=2,
        candidate_offset=0,
        base_seed=7,
        output_dir=destination,
        timeout_seconds=1.0,
        work_limit=1.0,
        node_limit=1,
        max_attempt_multiplier=2,
        lkh_executable=None,
        op_local_temp_root=local_temp_root,
    )

    receipt = producer.generate_shard(args)

    assert receipt["rows"] == 2
    phase = receipt["phase_evidence"]
    assert phase["protocol"] == "clasp-op-local-stage-then-atomic-remote-publication-v1"
    assert phase["attempts_completed_before_publication"] == 3
    assert phase["strict_rows_completed_before_publication"] == 2
    assert phase["remote_publication_started_at_receipt_creation"] is False
    assert all(path.is_relative_to(local_temp_root) for path in observed_writes)
    writes_by_name: dict[str, int] = {}
    for path in observed_writes:
        writes_by_name[path.name] = writes_by_name.get(path.name, 0) + 1
    assert writes_by_name["structured_input.json"] == 3
    assert writes_by_name["gurobi.log"] == 3
    assert writes_by_name["rejected.json"] == 1
    assert writes_by_name["strict_replay.json"] == 2
    assert writes_by_name["replacements.json"] == 1
    assert writes_by_name["evidence_manifest.json"] == 1
    assert writes_by_name["replacement_receipt.json"] == 1
    assert len(copy_calls) == 1
    copied_source, hidden_stage = copy_calls[0]
    assert copied_source.is_relative_to(local_temp_root)
    assert hidden_stage.parent == destination.parent
    assert hidden_stage.name.startswith(f".{destination.name}.publish.")
    assert deep_validation_calls == [hidden_stage]
    assert destination.is_dir()
    assert not hidden_stage.exists()
    assert list(local_temp_root.iterdir()) == []

    phase_events = [
        json.loads(line.removeprefix("[ReplacementPhase] "))
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[ReplacementPhase] ")
    ]
    names = [event["event"] for event in phase_events]
    assert names.count("local_attempt_complete") == 0
    expected_order = [
        "all_local_rows_complete",
        "local_shard_sealed",
        "remote_publication_begin",
        "remote_copy_complete",
        "remote_seal_complete",
        "remote_postseal_validation_complete",
        "remote_authoritative_fsync_complete",
        "remote_atomic_rename_begin",
        "remote_publication_complete",
    ]
    assert [names.index(name) for name in expected_order] == sorted(
        names.index(name) for name in expected_order
    )
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert str(local_temp_root) not in serialized_receipt
    assert ".publish." not in serialized_receipt


@pytest.mark.parametrize("with_phase_evidence", [False, True])
def test_remote_deep_validator_accepts_v8_and_v12_compatible_receipts(
    tmp_path: Path,
    with_phase_evidence: bool,
) -> None:
    source = tmp_path / ("v12-local" if with_phase_evidence else "v8-local")
    remote = tmp_path / ("v12-remote" if with_phase_evidence else "v8-remote")
    _minimal_completed_op_tree(source, with_phase_evidence=with_phase_evidence)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    inventory, _json_payloads = producer._sealed_tree_inventory(source)
    assert list(inventory) == sorted(inventory)
    remote.mkdir(mode=0o700)
    remote_identity = producer._stat_identity(remote.stat(follow_symlinks=False))
    try:
        producer._copy_sealed_tree_once(source, remote, inventory)
        producer._seal_tree(remote)
        producer._deep_validate_copied_op_shard(remote, inventory)
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)
        producer._remove_private_tree(remote, expected_identity=remote_identity)


def test_remote_deep_validator_accepts_full_real_implementation_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "local"
    remote = tmp_path / "remote"
    _minimal_completed_op_tree(
        source,
        with_phase_evidence=True,
        with_full_provenance=True,
    )
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    inventory, json_payloads = producer._sealed_tree_inventory(source)
    receipt = json_payloads["replacement_receipt.json"]
    strict_files = receipt["implementation"]["strict_verifier"]["files"]
    assert any(
        set(record) == {"canonical_path", "relative_path", "size_bytes", "sha256"}
        for record in strict_files
    )
    remote.mkdir(mode=0o700)
    remote_identity = producer._stat_identity(remote.stat(follow_symlinks=False))
    try:
        producer._copy_sealed_tree_once(source, remote, inventory)
        producer._seal_tree(remote)
        producer._deep_validate_copied_op_shard(remote, inventory)
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)
        producer._remove_private_tree(remote, expected_identity=remote_identity)


@pytest.mark.parametrize("record_role", ["output", "evidence", "strict_replay"])
def test_remote_deep_validator_rejects_canonical_tag_bypass_on_shard_records(
    tmp_path: Path,
    record_role: str,
) -> None:
    source = tmp_path / "local"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    try:
        for path in (source, *source.rglob("*")):
            if path.is_dir():
                path.chmod(0o700)
            else:
                path.chmod(0o600)
        receipt_path = source / "replacement_receipt.json"
        evidence_path = source / "evidence_manifest.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if record_role == "output":
            receipt["output"]["canonical_path"] = str(SCRIPT.resolve(strict=True))
        elif record_role == "evidence":
            receipt["evidence_manifest"]["canonical_path"] = str(
                SCRIPT.resolve(strict=True)
            )
        else:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence[0]["strict_replay"]["canonical_path"] = str(
                SCRIPT.resolve(strict=True)
            )
            evidence_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt["evidence_manifest"] = producer._file_record(
                evidence_path,
                relative_to=source,
            )
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = producer.clean.sha256_json(receipt)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        producer._seal_tree(source)
        inventory, _json_payloads = producer._sealed_tree_inventory(source)
        with pytest.raises(
            producer.ReplacementGenerationError,
            match="outside an approved provenance context",
        ):
            producer._deep_validate_copied_op_shard(source, inventory)
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_file",
        "unsafe_relative",
        "invalid_size",
        "invalid_sha",
        "manifest_mismatch",
        "inconsistent_root",
    ],
)
def test_receipt_implementation_rejects_malformed_canonical_provenance(
    mutation: str,
) -> None:
    implementation = json.loads(json.dumps(_test_implementation_identity()))
    strict = implementation["strict_verifier"]
    files = strict["files"]
    if mutation == "unknown_file":
        files[0]["relative_path"] = "scripts/unknown.py"
    elif mutation == "unsafe_relative":
        files[0]["relative_path"] = "../build_clean_sft500k_corpus.py"
    elif mutation == "invalid_size":
        files[0]["size_bytes"] = True
    elif mutation == "invalid_sha":
        files[0]["sha256"] = "A" * 64
    elif mutation == "manifest_mismatch":
        strict["implementation_manifest_sha256"] = "0" * 64
    else:
        files[0]["canonical_path"] = "/different/root/" + files[0]["relative_path"]
    with pytest.raises(producer.ReplacementGenerationError):
        producer._validate_receipt_implementation(implementation)


def test_receipt_solver_identity_rejects_malformed_canonical_provenance() -> None:
    solver_identity = _test_solver_identity()
    solver_identity["python"]["canonical_path"] = "relative/python"
    with pytest.raises(
        producer.ReplacementGenerationError,
        match="noncanonical absolute path",
    ):
        producer._validate_receipt_solver_identity(solver_identity)


@pytest.mark.parametrize("mutation", ["short_copy", "same_size_hash_mismatch"])
def test_remote_deep_validator_rejects_short_or_corrupted_copy(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "local"
    remote = tmp_path / "remote"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    inventory, _json_payloads = producer._sealed_tree_inventory(source)
    remote.mkdir(mode=0o700)
    remote_identity = producer._stat_identity(remote.stat(follow_symlinks=False))
    producer._copy_sealed_tree_once(source, remote, inventory)
    corrupted = (
        remote
        / "native_evidence"
        / "attempt-000001-candidate-000000000"
        / "solver.log"
    )
    corrupted.chmod(0o640)
    payload = corrupted.read_bytes()
    if mutation == "short_copy":
        corrupted.write_bytes(payload[:-1])
    else:
        corrupted.write_bytes(b"X" + payload[1:])
    producer._seal_tree(remote)
    try:
        with pytest.raises(producer.ReplacementGenerationError, match="inventory"):
            producer._deep_validate_copied_op_shard(remote, inventory)
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)
        producer._remove_private_tree(remote, expected_identity=remote_identity)


def test_remote_copy_performs_no_fsync_and_authoritative_commit_fsyncs_four_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local"
    remote = tmp_path / "remote"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    inventory, _json_payloads = producer._sealed_tree_inventory(source)
    remote.mkdir(mode=0o700)
    remote_identity = producer._stat_identity(remote.stat(follow_symlinks=False))
    fsync_calls: list[int] = []
    monkeypatch.setattr(producer.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    try:
        producer._copy_sealed_tree_once(source, remote, inventory)
        assert fsync_calls == []
        producer._seal_tree(remote)
        producer._deep_validate_copied_op_shard(remote, inventory)
        producer._fsync_remote_authoritative_artifacts(remote)
        assert len(fsync_calls) == 4
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)
        producer._remove_private_tree(remote, expected_identity=remote_identity)


def test_op_publication_enospc_cleans_a_sealed_hidden_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local"
    destination = tmp_path / "remote-parent" / "shard-000"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    monkeypatch.setattr(
        producer,
        "_require_distinct_stage_filesystems",
        lambda _local, _remote: None,
    )
    monkeypatch.setattr(
        producer,
        "_fsync_remote_authoritative_artifacts",
        lambda _root: (_ for _ in ()).throw(OSError(errno.ENOSPC, "injected")),
    )
    try:
        with pytest.raises(OSError) as raised:
            producer._publish_completed_local_op_shard(
                source,
                destination,
                attempts=1,
                rows=1,
            )
        assert raised.value.errno == errno.ENOSPC
        assert not destination.exists()
        assert list(destination.parent.glob(f".{destination.name}.publish.*")) == []
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)


def test_op_publication_create_once_race_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local"
    destination = tmp_path / "remote-parent" / "shard-000"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    monkeypatch.setattr(
        producer,
        "_require_distinct_stage_filesystems",
        lambda _local, _remote: None,
    )

    def race_before_rename(_source: Path, raced_destination: Path) -> None:
        raced_destination.mkdir()
        (raced_destination / "owner.txt").write_text("competitor\n", encoding="utf-8")
        raise FileExistsError("injected final-name race")

    monkeypatch.setattr(producer, "_rename_directory_noreplace", race_before_rename)
    try:
        with pytest.raises(FileExistsError, match="injected final-name race"):
            producer._publish_completed_local_op_shard(
                source,
                destination,
                attempts=1,
                rows=1,
            )
        assert (destination / "owner.txt").read_text(encoding="utf-8") == "competitor\n"
        assert list(destination.parent.glob(f".{destination.name}.publish.*")) == []
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)


def test_op_publication_failure_after_rename_quarantines_own_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local"
    destination = tmp_path / "remote-parent" / "shard-000"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    monkeypatch.setattr(
        producer,
        "_require_distinct_stage_filesystems",
        lambda _local, _remote: None,
    )

    def rename_then_fail(hidden: Path, final: Path) -> None:
        os.rename(hidden, final)
        raise OSError(errno.EIO, "injected post-rename failure")

    monkeypatch.setattr(producer, "_rename_directory_noreplace", rename_then_fail)
    try:
        with pytest.raises(OSError) as raised:
            producer._publish_completed_local_op_shard(
                source,
                destination,
                attempts=1,
                rows=1,
            )
        assert raised.value.errno == errno.EIO
        assert not destination.exists()
        quarantines = list(destination.parent.glob(f".{destination.name}.quarantine.*"))
        assert len(quarantines) == 1
        quarantine_identity = producer._stat_identity(
            quarantines[0].stat(follow_symlinks=False)
        )
        producer._remove_private_tree(
            quarantines[0],
            expected_identity=quarantine_identity,
        )
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)


def test_inventory_rejects_hardlinks_and_cleanup_refuses_inode_swap(tmp_path: Path) -> None:
    hardlink_root = tmp_path / "hardlink-root"
    hardlink_root.mkdir()
    first = hardlink_root / "first.bin"
    first.write_bytes(b"same inode\n")
    os.link(first, hardlink_root / "second.bin")
    with pytest.raises(producer.ReplacementGenerationError, match="multiply-linked"):
        producer._sealed_tree_inventory(hardlink_root)

    sealed = tmp_path / "sealed"
    _minimal_completed_op_tree(sealed, with_phase_evidence=True)
    sealed_identity = producer._stat_identity(sealed.stat(follow_symlinks=False))
    displaced = tmp_path / "displaced"
    os.rename(sealed, displaced)
    sealed.mkdir()
    with pytest.raises(producer.ReplacementGenerationError, match="replaced private tree"):
        producer._remove_private_tree(sealed, expected_identity=sealed_identity)
    sealed.rmdir()
    producer._remove_private_tree(displaced, expected_identity=sealed_identity)


def test_inventory_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    (symlink_root / "regular").write_bytes(b"regular\n")
    (symlink_root / "link").symlink_to("regular")
    with pytest.raises(producer.ReplacementGenerationError, match="symlink or special"):
        producer._sealed_tree_inventory(symlink_root)

    fifo_root = tmp_path / "fifo-root"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "fifo", 0o600)
    with pytest.raises(producer.ReplacementGenerationError, match="symlink or special"):
        producer._sealed_tree_inventory(fifo_root)


def test_op_publication_requires_distinct_local_and_destination_filesystems(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    destination_parent = tmp_path / "remote"
    local.mkdir()
    destination_parent.mkdir()
    with pytest.raises(producer.ReplacementGenerationError, match="different filesystems"):
        producer._require_distinct_stage_filesystems(local, destination_parent)


def test_hidden_stage_creation_retries_collision_under_pinned_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "shard-000"
    colliding = parent / ".shard-000.publish.collision"
    colliding.mkdir()
    tokens = iter(("collision", "unique"))
    monkeypatch.setattr(producer.secrets, "token_hex", lambda _count: next(tokens))

    hidden, identity = producer._create_hidden_sibling_stage(destination)

    assert hidden.name == ".shard-000.publish.unique"
    assert colliding.is_dir()
    producer._remove_private_tree(hidden, expected_identity=identity)


def test_hidden_stage_normalizes_setgid_parent_inheritance_to_private_mode(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "setgid-parent"
    parent.mkdir()
    parent.chmod(0o2750)
    destination = parent / "shard-000"

    hidden, identity = producer._create_hidden_sibling_stage(destination)

    assert stat.S_IMODE(parent.stat().st_mode) == 0o2750
    assert stat.S_IMODE(hidden.stat().st_mode) == 0o700
    assert hidden.stat().st_uid == os.geteuid()
    producer._remove_private_tree(hidden, expected_identity=identity)


@pytest.mark.parametrize("failure_point", ["open", "fchmod"])
def test_hidden_stage_postmkdir_failure_cleans_its_inode_pinned_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "shard-000"
    monkeypatch.setattr(producer.secrets, "token_hex", lambda _count: "fixed")
    if failure_point == "open":
        original_open = producer.os.open

        def fail_hidden_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
            if path == ".shard-000.publish.fixed":
                raise OSError(errno.EIO, "injected hidden open failure")
            return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(producer.os, "open", fail_hidden_open)
    else:
        monkeypatch.setattr(
            producer.os,
            "fchmod",
            lambda _descriptor, _mode: (_ for _ in ()).throw(
                OSError(errno.EIO, "injected hidden fchmod failure")
            ),
        )

    with pytest.raises(OSError) as raised:
        producer._create_hidden_sibling_stage(destination)
    assert raised.value.errno == errno.EIO
    assert list(parent.glob(".shard-000.publish.*")) == []
    assert not destination.exists()


def test_publication_parent_policy_allows_owner_setgid_but_rejects_unsafe_modes() -> None:
    producer._validate_safe_publication_parent(
        types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o2750, st_uid=os.geteuid())
    )
    with pytest.raises(producer.ReplacementGenerationError, match="wrong owner"):
        producer._validate_safe_publication_parent(
            types.SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o2750,
                st_uid=os.geteuid() + 1,
            )
        )
    with pytest.raises(producer.ReplacementGenerationError, match="group/other writable"):
        producer._validate_safe_publication_parent(
            types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o2770, st_uid=os.geteuid())
        )


def test_postpublication_cleanup_failure_is_deferred_but_prepublish_failure_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage = tmp_path / "local-stage"
    stage.mkdir()
    identity = producer._stat_identity(stage.stat(follow_symlinks=False))

    def fail_cleanup(_root: Path, *, expected_identity: tuple[int, int]) -> None:
        del expected_identity
        raise OSError(errno.EIO, "injected cleanup failure")

    monkeypatch.setattr(producer, "_remove_private_tree", fail_cleanup)
    producer._cleanup_local_op_stage_after_generation(
        stage,
        expected_identity=identity,
        completed=True,
    )
    assert "local_cleanup_deferred_to_launcher" in capsys.readouterr().err
    with pytest.raises(OSError) as raised:
        producer._cleanup_local_op_stage_after_generation(
            stage,
            expected_identity=identity,
            completed=False,
        )
    assert raised.value.errno == errno.EIO


def test_publication_phase_telemetry_failure_cleans_hidden_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local"
    destination = tmp_path / "remote-parent" / "shard-000"
    _minimal_completed_op_tree(source, with_phase_evidence=True)
    source_identity = producer._stat_identity(source.stat(follow_symlinks=False))
    monkeypatch.setattr(
        producer,
        "_require_distinct_stage_filesystems",
        lambda _local, _remote: None,
    )
    monkeypatch.setattr(
        producer,
        "_emit_phase_event",
        lambda _event, **_fields: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected telemetry failure")
        ),
    )
    try:
        with pytest.raises(OSError) as raised:
            producer._publish_completed_local_op_shard(
                source,
                destination,
                attempts=1,
                rows=1,
            )
        assert raised.value.errno == errno.EIO
        assert not destination.exists()
        assert list(destination.parent.glob(f".{destination.name}.publish.*")) == []
    finally:
        producer._remove_private_tree(source, expected_identity=source_identity)


def test_directory_publication_never_replaces_empty_raced_target(tmp_path: Path) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        producer._rename_directory_noreplace(source, destination)
    assert source.is_dir()
    assert destination.is_dir()


def test_directory_publication_uses_lustre_safe_fallback_on_einval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "published"
    source.mkdir()
    (source / "bound.txt").write_text("sealed\n", encoding="utf-8")
    monkeypatch.setattr(
        producer,
        "_renameat2_noreplace_errno",
        lambda **_kwargs: errno.EINVAL,
    )

    producer._rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "bound.txt").read_text(encoding="utf-8") == "sealed\n"


def test_directory_fallback_preserves_raced_preexisting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "published"
    source.mkdir()

    def race_then_report_unsupported(
        *,
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> int:
        del source_name
        os.mkdir(destination_name, dir_fd=parent_descriptor)
        destination_descriptor = os.open(
            destination_name,
            producer._directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            marker_descriptor = os.open(
                "owner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_descriptor,
            )
            try:
                os.write(marker_descriptor, b"pre-existing\n")
            finally:
                os.close(marker_descriptor)
        finally:
            os.close(destination_descriptor)
        return errno.EINVAL

    monkeypatch.setattr(
        producer,
        "_renameat2_noreplace_errno",
        race_then_report_unsupported,
    )

    with pytest.raises(
        FileExistsError,
        match=r"refusing to overwrite replacement shard",
    ):
        producer._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "pre-existing\n"


def test_directory_fallback_cleanup_refuses_swapped_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "published"
    source.mkdir()
    monkeypatch.setattr(
        producer,
        "_renameat2_noreplace_errno",
        lambda **_kwargs: errno.EOPNOTSUPP,
    )

    def swap_reservation_then_fail(
        _source: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        del src_dir_fd
        os.rmdir(destination_name, dir_fd=dst_dir_fd)
        os.mkdir(destination_name, mode=0o700, dir_fd=dst_dir_fd)
        raise OSError(errno.EIO, "injected rename failure")

    monkeypatch.setattr(producer.os, "rename", swap_reservation_then_fail)

    with pytest.raises(
        producer.ReplacementGenerationError,
        match="refusing to clean a replaced shard-publication reservation",
    ):
        producer._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_op_model_constraints_and_node_local_log_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the OP builder and inspect the equations sent to Gurobi.

    This catches accidental reuse of a stale loop variable: every selected
    customer must have exactly one outgoing arc, not repeated copies of the
    final customer's equation.
    """

    class Expression:
        def __init__(
            self,
            terms: dict[str, float] | None = None,
            constant: float = 0.0,
        ) -> None:
            self.terms = dict(terms or {})
            self.constant = float(constant)

        @staticmethod
        def coerce(value: object) -> "Expression":
            if isinstance(value, Expression):
                return value
            if isinstance(value, Variable):
                return Expression({value.token: 1.0})
            return Expression(constant=float(value))

        def _combine(self, other: object, scale: float) -> "Expression":
            result = Expression(self.terms, self.constant)
            converted = self.coerce(other)
            result.constant += scale * converted.constant
            for token, coefficient in converted.terms.items():
                result.terms[token] = result.terms.get(token, 0.0) + scale * coefficient
                if result.terms[token] == 0.0:
                    del result.terms[token]
            return result

        def __add__(self, other: object) -> "Expression":
            return self._combine(other, 1.0)

        def __radd__(self, other: object) -> "Expression":
            return self.coerce(other)._combine(self, 1.0)

        def __sub__(self, other: object) -> "Expression":
            return self._combine(other, -1.0)

        def __rsub__(self, other: object) -> "Expression":
            return self.coerce(other)._combine(self, -1.0)

        def __mul__(self, scalar: object) -> "Expression":
            factor = float(scalar)
            return Expression(
                {token: factor * coefficient for token, coefficient in self.terms.items()},
                factor * self.constant,
            )

        def __rmul__(self, scalar: object) -> "Expression":
            return self * scalar

        def _comparison(self, other: object, sense: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(expression=self - other, sense=sense)

        def __eq__(self, other: object) -> types.SimpleNamespace:  # type: ignore[override]
            return self._comparison(other, "==")

        def __le__(self, other: object) -> types.SimpleNamespace:
            return self._comparison(other, "<=")

        def __ge__(self, other: object) -> types.SimpleNamespace:
            return self._comparison(other, ">=")

    class Variable:
        def __init__(self, token: str) -> None:
            self.token = token
            self.X = 0.0
            self.Start = 0.0

        def _expression(self) -> Expression:
            return Expression({self.token: 1.0})

        def __add__(self, other: object) -> Expression:
            return self._expression() + other

        def __radd__(self, other: object) -> Expression:
            return other + self._expression()  # type: ignore[operator]

        def __sub__(self, other: object) -> Expression:
            return self._expression() - other

        def __rsub__(self, other: object) -> Expression:
            return Expression.coerce(other) - self._expression()

        def __mul__(self, scalar: object) -> Expression:
            return self._expression() * scalar

        def __rmul__(self, scalar: object) -> Expression:
            return self * scalar

        def __eq__(self, other: object) -> types.SimpleNamespace:  # type: ignore[override]
            return self._expression() == other

        def __le__(self, other: object) -> types.SimpleNamespace:
            return self._expression() <= other

        def __ge__(self, other: object) -> types.SimpleNamespace:
            return self._expression() >= other

    models: list[object] = []
    requested_models = [
        {
            "solution_count": 1,
            "objective": 60.0,
            "path": ((0, 1), (1, 2), (2, 3), (3, 4)),
        },
        {
            "solution_count": 0,
            "objective": 60.0,
            "path": ((0, 1), (1, 2), (2, 3), (3, 4)),
        },
        {"solution_count": 1, "objective": 30.0, "path": ((0, 1), (1, 2), (2, 4))},
        {
            "solution_count": 1,
            "objective": 61.0,
            "path": ((0, 1), (1, 2), (2, 3), (3, 4)),
        },
    ]

    class RecordingParams:
        def __init__(self) -> None:
            object.__setattr__(self, "order", [])
            object.__setattr__(self, "values", {})

        def __setattr__(self, name: str, value: object) -> None:
            self.order.append(name)
            self.values[name] = value

        def __getattr__(self, name: str) -> object:
            return self.values[name]

    class FakeModel:
        def __init__(self, _name: str) -> None:
            self.requested = requested_models[len(models)]
            self.Params = RecordingParams()
            self.constraints: list[types.SimpleNamespace] = []
            self.variable_groups: dict[str, dict[object, Variable]] = {}
            self.Status = 2
            self.SolCount = self.requested["solution_count"]
            self.ObjVal = self.requested["objective"]
            self.ObjBound = 60.0
            self.MIPGap = 0.0
            self.Runtime = 0.01
            self.NodeCount = 1.0
            self.Work = 0.01
            models.append(self)

        def addVars(
            self,
            keys: object,
            *,
            name: str,
            **_kwargs: object,
        ) -> dict[object, Variable]:
            def token(key: object) -> str:
                if isinstance(key, tuple):
                    rendered = ",".join(str(part) for part in key)
                else:
                    rendered = str(key)
                return f"{name}[{rendered}]"

            variables = {
                key: Variable(token(key))
                for key in list(keys)  # type: ignore[arg-type]
            }
            self.variable_groups[name] = variables
            return variables

        def addConstr(self, constraint: types.SimpleNamespace) -> None:
            self.constraints.append(constraint)

        def addGenConstrIndicator(self, *_args: object) -> None:
            return None

        def setObjective(self, *_args: object) -> None:
            return None

        def optimize(self) -> None:
            Path(self.Params.LogFile).write_text("fake optimal solve\n", encoding="utf-8")
            used = self.variable_groups["used"]
            edge = self.variable_groups["edge"]
            for node in range(4):
                used[node].X = 1.0
            for arc in self.requested["path"]:
                edge[arc].X = 1.0

        def dispose(self) -> None:
            self.disposed = True
            with Path(self.Params.LogFile).open("a", encoding="utf-8") as handle:
                handle.write("fake dispose flush\n")

    def quicksum(values: object) -> Expression:
        total = Expression()
        for value in values:  # type: ignore[union-attr]
            total += value
        return total

    fake_gurobi = types.SimpleNamespace(
        Model=FakeModel,
        quicksum=quicksum,
        GRB=types.SimpleNamespace(
            BINARY="BINARY",
            CONTINUOUS="CONTINUOUS",
            MAXIMIZE="MAXIMIZE",
            OPTIMAL=2,
            TIME_LIMIT=9,
            WORK_LIMIT=16,
            NODE_LIMIT=8,
            INTERRUPTED=11,
        ),
    )
    instance = types.SimpleNamespace(
        coordinates=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
        prizes=(0, 10, 20, 30),
        route_limit=10.0,
    )
    reference = types.SimpleNamespace(
        _validated_op_instance=lambda _row: instance,
        _op_pair_distance=lambda _instance, left, right: float(abs(left - right)),
        _op_budget_incompatible_pairs=lambda _instance: (),
        _deterministic_op_start=lambda _instance: [0, 1, 2, 3],
    )
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gurobi)
    monkeypatch.setattr(producer, "_REFERENCE_MODULE", reference)
    local_temp_root = tmp_path / "op-local"
    local_temp_root.mkdir(mode=0o700)
    row_dir = tmp_path / "hidden_stage" / "native_evidence" / "candidate-00000000"
    row_dir.mkdir(parents=True)

    output, _evidence = producer.solve_op_gurobi_incumbent(
        {},
        row_dir=row_dir,
        local_temp_root=local_temp_root,
        timeout_seconds=1.0,
        work_limit=1.0,
        node_limit=1,
    )

    assert output == "Route: [0, 1, 2, 3], Objective: 60"
    assert len(models) == 1
    model = models[0]
    configured_log = Path(model.Params.LogFile)  # type: ignore[attr-defined]
    assert configured_log.parent == local_temp_root
    assert row_dir not in configured_log.parents
    assert model.disposed  # type: ignore[attr-defined]
    assert model.Params.order[:3] == [  # type: ignore[attr-defined]
        "OutputFlag",
        "LogToConsole",
        "LogFile",
    ]
    assert (row_dir / "gurobi.log").read_text(encoding="utf-8") == (
        "fake optimal solve\nfake dispose flush\n"
    )
    customers = (1, 2, 3)
    sink = 4
    outgoing_counts: dict[int, int] = {}
    for customer in customers:
        expected_edges = {
            f"edge[{customer},{target}]"
            for target in (*customers, sink)
            if target != customer
        }
        outgoing_counts[customer] = sum(
            constraint.sense == "=="
            and constraint.expression.terms.get(f"used[{customer}]") == -1.0
            and {
                token
                for token, coefficient in constraint.expression.terms.items()
                if token.startswith("edge[") and coefficient == 1.0
            }
            == expected_edges
            for constraint in model.constraints  # type: ignore[attr-defined]
        )
    assert outgoing_counts == {1: 1, 2: 1, 3: 1}

    rejected_row_dir = (
        tmp_path / "hidden_stage" / "native_evidence" / "candidate-00000001"
    )
    rejected_row_dir.mkdir()
    with pytest.raises(producer.ReplacementGenerationError, match="returned no incumbent"):
        producer.solve_op_gurobi_incumbent(
            {},
            row_dir=rejected_row_dir,
            local_temp_root=local_temp_root,
            timeout_seconds=1.0,
            work_limit=1.0,
            node_limit=1,
        )
    assert len(models) == 2
    rejected_model = models[1]
    assert rejected_model.disposed  # type: ignore[attr-defined]
    rejected_log = Path(rejected_model.Params.LogFile)  # type: ignore[attr-defined]
    assert rejected_log.parent == local_temp_root
    assert (rejected_row_dir / "gurobi.log").read_text(encoding="utf-8") == (
        "fake optimal solve\nfake dispose flush\n"
    )

    disconnected_row_dir = (
        tmp_path / "hidden_stage" / "native_evidence" / "candidate-00000002"
    )
    disconnected_row_dir.mkdir()
    with pytest.raises(
        producer.ReplacementGenerationError,
        match="selected nodes do not equal the extracted path",
    ):
        producer.solve_op_gurobi_incumbent(
            {},
            row_dir=disconnected_row_dir,
            local_temp_root=local_temp_root,
            timeout_seconds=1.0,
            work_limit=1.0,
            node_limit=1,
        )
    assert (disconnected_row_dir / "gurobi.log").is_file()

    mismatched_row_dir = (
        tmp_path / "hidden_stage" / "native_evidence" / "candidate-00000003"
    )
    mismatched_row_dir.mkdir()
    with pytest.raises(
        producer.ReplacementGenerationError,
        match="objective does not equal the extracted path prize",
    ):
        producer.solve_op_gurobi_incumbent(
            {},
            row_dir=mismatched_row_dir,
            local_temp_root=local_temp_root,
            timeout_seconds=1.0,
            work_limit=1.0,
            node_limit=1,
        )
    assert (mismatched_row_dir / "gurobi.log").is_file()
