from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_source_snapshot", ROOT / "scripts" / "freeze_source_snapshot.py"
)
assert SPEC is not None and SPEC.loader is not None
freeze_source_snapshot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze_source_snapshot
SPEC.loader.exec_module(freeze_source_snapshot)


def _staging(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot.staging"
    for relative in sorted(freeze_source_snapshot.REQUIRED_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# required test source: {relative}\n")
    legitimate_models = root / "src" / "clasp" / "models" / "training_fidelity.py"
    legitimate_models.parent.mkdir(parents=True, exist_ok=True)
    legitimate_models.write_text("# legitimate source-code model module\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_example.py").write_text("def test_ok(): pass\n")
    return root


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def test_required_snapshot_covers_clean_sft_provenance_and_runtime() -> None:
    assert {
        "scripts/audit_sft_reload_equivalence.py",
        "scripts/audit_sft_token_lengths.py",
        "scripts/base_sft_single_h100_resume_fidelity.py",
        "scripts/build_clean_sft500k_corpus.py",
        "scripts/freeze_benchmark.py",
        "scripts/freeze_forbidden_source_registry.py",
        "scripts/freeze_sft_reserved_source_registry.py",
        "scripts/generate_clean_sft_replacements.py",
        "scripts/generate_clean_sft_replacements_a100.slurm",
        "scripts/generate_iid_benchmark.py",
        "scripts/run_reference_evidence.py",
        "scripts/seal_completed_sft_run.py",
        "scripts/smoke_jiang_f638_sft_environment_h100.slurm",
        "scripts/submit_sft500k_base_specialists_h100.sh",
        "scripts/train_sft500k_base_specialist_h100.slurm",
        "requirements.jiang-f638-runtime-closure.txt",
        "requirements.jiang-f638-sft.txt",
        "src/clasp/models/jiang_f638_base.py",
    } <= freeze_source_snapshot.REQUIRED_FILES
    assert {
        "sft_cleanliness_audit.json",
        "completed_sft_run_seal.json",
        "sft_reload_equivalence.json",
        "sft_reserved_source_registry.json",
        "sft_token_length_audit.json",
    } <= freeze_source_snapshot.FORBIDDEN_RUNTIME_FILE_NAMES


def test_freeze_publishes_manifest_modes_and_self_hashed_receipt(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "snapshot-v3.receipt.json"

    payload = freeze_source_snapshot.freeze(
        staging_root=staging,
        final_root=final,
        receipt_path=receipt,
        snapshot_id="correct_clasp_sota_20260814_v3",
    )

    assert not staging.exists()
    assert final.is_dir()
    assert stat.S_IMODE(final.stat().st_mode) == 0o2550
    assert stat.S_IMODE((final / "src").stat().st_mode) == 0o550
    assert stat.S_IMODE((final / "pyproject.toml").stat().st_mode) == 0o440
    assert stat.S_IMODE((final / "logs").stat().st_mode) == 0o770
    manifest = final / "SOURCE_MANIFEST.sha256"
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o440
    lines = manifest.read_text().splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths)
    assert "logs" not in paths
    assert "SOURCE_MANIFEST.sha256" not in paths
    for line in lines:
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((final / relative).read_bytes()).hexdigest() == expected

    stored = json.loads(receipt.read_text())
    assert stored == payload
    stored_hash = stored.pop("receipt_sha256")
    assert hashlib.sha256(_canonical(stored)).hexdigest() == stored_hash
    assert payload["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert payload["sealed_tree_protocol"] == "clasp-source-sealed-tree-v1"
    assert len(payload["sealed_tree_sha256"]) == 64
    assert payload["source_parent_identity"]["path"] == str(final.parent)
    assert payload["source_root_identity"] == {
        "device": final.stat().st_dev,
        "inode": final.stat().st_ino,
    }


def test_freeze_refuses_existing_final_or_receipt(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    final.mkdir()
    with pytest.raises(ValueError, match="replace existing final snapshot"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def test_rename_noreplace_uses_exclusive_directory_fallback_on_einval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    (source / "bound.txt").write_text("sealed\n", encoding="utf-8")
    monkeypatch.setattr(
        freeze_source_snapshot,
        "_renameat2_noreplace_errno",
        lambda **_kwargs: errno.EINVAL,
    )
    parent_descriptor = os.open(tmp_path, freeze_source_snapshot._directory_open_flags())
    try:
        freeze_source_snapshot._rename_noreplace_at(
            parent_descriptor=parent_descriptor,
            source_name=source.name,
            destination_name=destination.name,
        )
    finally:
        os.close(parent_descriptor)
    assert not source.exists()
    assert (destination / "bound.txt").read_text(encoding="utf-8") == "sealed\n"


def test_directory_fallback_never_replaces_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    marker = destination / "owner.txt"
    marker.write_text("pre-existing\n", encoding="utf-8")
    monkeypatch.setattr(
        freeze_source_snapshot,
        "_renameat2_noreplace_errno",
        lambda **_kwargs: errno.ENOTSUP,
    )
    parent_descriptor = os.open(tmp_path, freeze_source_snapshot._directory_open_flags())
    try:
        with pytest.raises(ValueError, match="destination already exists"):
            freeze_source_snapshot._rename_noreplace_at(
                parent_descriptor=parent_descriptor,
                source_name=source.name,
                destination_name=destination.name,
            )
    finally:
        os.close(parent_descriptor)
    assert source.is_dir()
    assert marker.read_text(encoding="utf-8") == "pre-existing\n"


def test_directory_fallback_cleanup_refuses_swapped_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    monkeypatch.setattr(
        freeze_source_snapshot,
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

    monkeypatch.setattr(freeze_source_snapshot.os, "rename", swap_reservation_then_fail)
    parent_descriptor = os.open(tmp_path, freeze_source_snapshot._directory_open_flags())
    try:
        with pytest.raises(ValueError, match="refusing to clean a replaced"):
            freeze_source_snapshot._rename_noreplace_at(
                parent_descriptor=parent_descriptor,
                source_name=source.name,
                destination_name=destination.name,
            )
    finally:
        os.close(parent_descriptor)
    assert source.is_dir()
    assert destination.is_dir()


def test_freeze_rejects_artifacts_and_links(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    cache = staging / "__pycache__"
    cache.mkdir()
    (cache / "bad.pyc").write_bytes(b"cache")
    with pytest.raises(ValueError, match="forbidden artifact"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )

    (cache / "bad.pyc").unlink()
    cache.rmdir()
    (staging / "linked").symlink_to(staging / "pyproject.toml")
    with pytest.raises(ValueError, match="symbolic link"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


@pytest.mark.parametrize(
    "relative",
    [
        "experiments/sota_20260814/local_development/model/config.json",
        "nested/weights/model-index.json",
        "nested/model/config.json",
        "nested/adapters/adapter_config.json",
        "nested/checkpoints/checkpoint-10/state.json",
        "nested/optimizer/state.json",
        "nested/logs/train.log",
        "nested/raw/generations.json",
        "nested/repair/repaired_details.json",
        "nested/runtime_outputs/result.json",
    ],
)
def test_freeze_rejects_artifact_directory_components_at_any_depth(
    tmp_path: Path, relative: str
) -> None:
    staging = _staging(tmp_path)
    artifact = staging / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("artifact\n")

    with pytest.raises(ValueError, match="forbidden artifact directory component"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


@pytest.mark.parametrize(
    "relative",
    [
        "assets/adapter_model.safetensors",
        "assets/model-00001-of-00004.safetensors",
        "assets/training_args.bin",
        "assets/trainer_state.json",
        "assets/sft_cleanliness_audit.json",
        "assets/sft_reload_equivalence.json",
        "assets/sft_reserved_source_registry.json",
        "assets/sft_token_length_audit.json",
        "assets/raw_generations.json",
    ],
)
def test_freeze_rejects_nested_model_and_runtime_files(tmp_path: Path, relative: str) -> None:
    staging = _staging(tmp_path)
    artifact = staging / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"artifact")

    with pytest.raises(ValueError, match="forbidden artifact"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def test_freeze_accepts_only_static_allowlisted_experiment_files(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    allowed = staging / "experiments" / "sota_20260814" / "campaign_manifest.template.json"
    assert allowed.is_file()
    final = tmp_path / "snapshot-v3"

    freeze_source_snapshot.freeze(
        staging_root=staging,
        final_root=final,
        receipt_path=tmp_path / "receipt.json",
        snapshot_id="v3",
    )

    manifest_paths = {
        line.split("  ", 1)[1]
        for line in (final / "SOURCE_MANIFEST.sha256").read_text().splitlines()
    }
    assert allowed.relative_to(staging).as_posix() in manifest_paths


def test_freeze_rejects_experiment_runtime_evidence_outside_allowlist(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    runtime_evidence = staging / "experiments" / "sota_20260814" / "job_ledger.json"
    runtime_evidence.parent.mkdir(parents=True, exist_ok=True)
    runtime_evidence.write_text("{}\n")

    with pytest.raises(ValueError, match="not in the static allowlist"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def test_static_experiment_allowlist_names_existing_repository_files() -> None:
    assert freeze_source_snapshot.STATIC_EXPERIMENT_FILES
    assert not {
        relative
        for relative in freeze_source_snapshot.STATIC_EXPERIMENT_FILES
        if not (ROOT / relative).is_file()
    }


def test_canonical_repository_source_surface_passes_artifact_filter() -> None:
    for top_level in ("src", "scripts", "tests", "docs"):
        for path in sorted((ROOT / top_level).rglob("*")):
            relative = path.relative_to(ROOT)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise AssertionError(f"canonical source contains a symlink: {relative}")
            freeze_source_snapshot._validate_relative_entry(
                relative,
                is_directory=path.is_dir(),
            )
    for relative in sorted(freeze_source_snapshot.STATIC_EXPERIMENT_FILES):
        freeze_source_snapshot._validate_relative_entry(Path(relative), is_directory=False)


def test_remote_launch_documents_exact_static_experiment_allowlist() -> None:
    runbook = (ROOT / "docs" / "remote_publication_launch.md").read_text()
    documented_block = runbook.split("STATIC_EXPERIMENT_FILES=(", 1)[1].split("\n)", 1)[0]
    documented = {
        line.strip()
        for line in documented_block.splitlines()
        if line.strip().startswith("experiments/")
    }

    assert documented == freeze_source_snapshot.STATIC_EXPERIMENT_FILES
    assert "README.md pyproject.toml src scripts tests docs experiments" not in runbook
    assert "SNAPSHOT_ID=correct_clasp_base_sft_20260815_v3" in runbook


def test_freeze_requires_complete_launch_surface(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    (staging / "scripts" / "prepare_sota_run.py").unlink()
    with pytest.raises(ValueError, match="staging source is incomplete"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def _mutate_snapshot_source(staging: Path, mutation: str) -> None:
    parent = staging / "scripts"
    target = parent / "audit_lora_persistence.py"
    os.chmod(parent, 0o750)
    try:
        if mutation == "add":
            extra = parent / "unexpected_source.py"
            extra.write_text("# injected after enumeration\n")
            os.chmod(extra, 0o440)
        elif mutation == "remove":
            target.unlink()
        elif mutation == "replace":
            replacement = parent / ".same-bytes-replacement"
            replacement.write_bytes(target.read_bytes())
            os.chmod(replacement, 0o440)
            os.replace(replacement, target)
        elif mutation == "symlink":
            target.unlink()
            target.symlink_to("freeze_source_snapshot.py")
        elif mutation == "fifo":
            target.unlink()
            os.mkfifo(target, mode=0o440)
        else:  # pragma: no cover - test helper contract
            raise AssertionError(f"unknown mutation: {mutation}")
    finally:
        os.chmod(parent, 0o550)


@pytest.mark.parametrize("mutation", ["add", "remove", "replace", "symlink", "fifo"])
def test_freeze_rejects_mutation_after_manifest_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    original_write = freeze_source_snapshot._write_atomic_new
    mutated = False

    def write_then_mutate(path: Path, content: bytes, *, mode: int) -> None:
        nonlocal mutated
        original_write(path, content, mode=mode)
        if path.name == freeze_source_snapshot.MANIFEST_NAME and not mutated:
            _mutate_snapshot_source(staging, mutation)
            mutated = True

    monkeypatch.setattr(freeze_source_snapshot, "_write_atomic_new", write_then_mutate)
    with pytest.raises(ValueError):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert mutated
    assert not final.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("mutation", ["add", "remove", "replace", "symlink", "fifo"])
def test_final_fd_verifier_rejects_mutation_after_all_chmods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    original_verify = freeze_source_snapshot._verify_sealed_tree
    mutated = False

    def mutate_then_verify(**kwargs: object) -> str:
        nonlocal mutated
        if not mutated:
            _mutate_snapshot_source(staging, mutation)
            mutated = True
        return original_verify(**kwargs)

    monkeypatch.setattr(freeze_source_snapshot, "_verify_sealed_tree", mutate_then_verify)
    with pytest.raises(ValueError):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert mutated
    assert not final.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("target", ["file", "directory"])
def test_final_fd_verifier_rejects_post_chmod_mode_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    staging = _staging(tmp_path)
    original_verify = freeze_source_snapshot._verify_sealed_tree
    mutated = False

    def mutate_then_verify(**kwargs: object) -> str:
        nonlocal mutated
        if not mutated:
            path = (
                staging / "scripts" / "audit_lora_persistence.py"
                if target == "file"
                else staging / "scripts"
            )
            os.chmod(path, 0o640 if target == "file" else 0o750)
            mutated = True
        return original_verify(**kwargs)

    monkeypatch.setattr(freeze_source_snapshot, "_verify_sealed_tree", mutate_then_verify)
    with pytest.raises(ValueError, match="mode changed"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def test_final_fd_verifier_rejects_external_hardlink(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    source = staging / "scripts" / "audit_lora_persistence.py"
    os.link(source, tmp_path / "outside-hardlink.py")

    with pytest.raises(ValueError, match="hard-link count"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=tmp_path / "snapshot-v3",
            receipt_path=tmp_path / "receipt.json",
            snapshot_id="v3",
        )


def test_second_fd_pass_catches_change_after_first_complete_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    original_verify = freeze_source_snapshot._verify_sealed_tree
    calls = 0

    def verify_then_mutate(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        result = original_verify(**kwargs)
        if calls == 1:
            _mutate_snapshot_source(staging, "replace")
        return result

    monkeypatch.setattr(freeze_source_snapshot, "_verify_sealed_tree", verify_then_mutate)
    with pytest.raises(ValueError, match="was replaced"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert calls == 2
    assert not final.exists()
    assert not receipt.exists()


def test_post_rename_fd_pass_catches_content_mutation_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    original_verify = freeze_source_snapshot._verify_sealed_tree
    calls = 0

    def mutate_third_pass(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            _mutate_snapshot_source(final, "replace")
        return original_verify(**kwargs)

    monkeypatch.setattr(freeze_source_snapshot, "_verify_sealed_tree", mutate_third_pass)
    with pytest.raises(ValueError, match="was replaced"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert calls == 3
    assert final.exists()
    assert not receipt.exists()


def test_published_path_swap_never_receives_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    displaced = tmp_path / "verified-tree-displaced"
    original_rename = freeze_source_snapshot._rename_noreplace_at

    def rename_then_swap(**kwargs: object) -> None:
        original_rename(**kwargs)
        final.rename(displaced)
        final.mkdir()

    monkeypatch.setattr(freeze_source_snapshot, "_rename_noreplace_at", rename_then_swap)
    with pytest.raises(ValueError, match="not the verified staging root"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert displaced.is_dir()
    assert final.is_dir()
    assert not receipt.exists()


def test_swap_during_receipt_publication_removes_transaction_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    displaced = tmp_path / "verified-tree-displaced"
    original_write = freeze_source_snapshot._write_atomic_new_at
    swapped = False

    def write_then_swap(**kwargs: object) -> tuple[int, int]:
        nonlocal swapped
        identity = original_write(**kwargs)
        if kwargs["name"] == receipt.name:
            final.rename(displaced)
            final.mkdir()
            swapped = True
        return identity

    monkeypatch.setattr(freeze_source_snapshot, "_write_atomic_new_at", write_then_swap)
    with pytest.raises(ValueError, match="receipt removed"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert swapped
    assert displaced.is_dir()
    assert final.is_dir()
    assert not receipt.exists()


def test_parent_relocation_during_rename_never_receives_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    displaced_parent = tmp_path.parent / f"{tmp_path.name}.displaced"
    original_rename = freeze_source_snapshot._rename_noreplace_at
    relocated = False

    def relocate_parent_then_rename(**kwargs: object) -> None:
        nonlocal relocated
        tmp_path.rename(displaced_parent)
        tmp_path.mkdir()
        relocated = True
        original_rename(**kwargs)

    monkeypatch.setattr(
        freeze_source_snapshot,
        "_rename_noreplace_at",
        relocate_parent_then_rename,
    )
    try:
        with pytest.raises(ValueError, match="canonical snapshot parent"):
            freeze_source_snapshot.freeze(
                staging_root=staging,
                final_root=final,
                receipt_path=receipt,
                snapshot_id="v3",
            )
        assert relocated
        assert not receipt.exists()
    finally:
        if relocated:
            tmp_path.rmdir()
            displaced_parent.rename(tmp_path)


def test_mutation_after_receipt_parent_check_is_removed_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _staging(tmp_path)
    final = tmp_path / "snapshot-v3"
    receipt = tmp_path / "receipt.json"
    original_check = freeze_source_snapshot._require_canonical_directory_identity
    mutated = False

    def check_then_mutate(
        path: Path,
        expected_identity: tuple[int, int],
        *,
        label: str,
    ) -> None:
        nonlocal mutated
        original_check(path, expected_identity, label=label)
        if label == "receipt parent after publication":
            _mutate_snapshot_source(final, "replace")
            mutated = True

    monkeypatch.setattr(
        freeze_source_snapshot,
        "_require_canonical_directory_identity",
        check_then_mutate,
    )
    with pytest.raises(ValueError, match="receipt removed"):
        freeze_source_snapshot.freeze(
            staging_root=staging,
            final_root=final,
            receipt_path=receipt,
            snapshot_id="v3",
        )

    assert mutated
    assert final.exists()
    assert not receipt.exists()
