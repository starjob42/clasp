from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "seal_completed_sft_run", SCRIPTS / "seal_completed_sft_run.py"
)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)


@pytest.fixture(autouse=True)
def _restore_tmp_tree_modes(tmp_path: Path):
    """Leave pytest able to retire temporary trees after exact-mode assertions."""

    yield
    if not tmp_path.exists():
        return
    for path in (candidate for candidate in tmp_path.rglob("*") if candidate.is_file()):
        if not path.is_symlink():
            path.chmod(0o600)
    for path in sorted(
        (candidate for candidate in tmp_path.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        if not path.is_symlink():
            path.chmod(0o700)
    tmp_path.chmod(0o700)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def _unsealed_tree(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    run = tmp_path / "run"
    nested = run / "checkpoints" / "checkpoint-5000"
    nested.mkdir(parents=True)
    (run / "root.txt").write_text("terminal\n", encoding="utf-8")
    (nested / "state.pt").write_bytes(b"state")
    return run, base


def _accept(run: Path, base: Path) -> dict[str, object]:
    return {"artifact_run": str(run), "artifact_base": str(base)}


def _complete_artifact_tree(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)

    adapter_config = {
        "base_model_name_or_path": str(base),
        "bias": "all",
        "peft_type": "LORA",
    }
    (run / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    (run / "adapter_model.safetensors").write_bytes(b"terminal adapter")
    for name in (
        "adapter_persistence_receipt.json",
        "base_sft_run_contract.json",
        "checkpoint_fidelity.json",
        "sft_reload_equivalence.json",
        "sft_trainer_state.json",
    ):
        (run / name).write_text("{}\n", encoding="utf-8")
    for name in (
        "README.md",
        "added_tokens.json",
        "merges.txt",
        "sft_log_history.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.json",
    ):
        (run / name).write_bytes(f"terminal:{name}".encode())
    (run / "hyperparams.json").write_text(
        json.dumps(
            {
                "protocol": "clasp-jiang-llmcosolver-f638-sft-corrected-v2",
                "stage": "sft",
                "dataset": {"train_size": 500_000, "eval_size": 0},
                "distributed": {"world_size": 1},
            }
        ),
        encoding="utf-8",
    )
    latest = checkpoints / "checkpoint-31250"
    (run / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "completed": True,
                "has_saved_model": True,
                "global_step": 31_250,
                "max_steps": 31_250,
                "latest_checkpoint_dir": str(latest),
            }
        ),
        encoding="utf-8",
    )
    for step in sealer.EXPECTED_TRAINER_CHECKPOINT_STEPS:
        checkpoint = checkpoints / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(adapter_config), encoding="utf-8"
        )
        adapter_bytes = b"terminal adapter" if step == 31_250 else b"periodic adapter"
        (checkpoint / "adapter_model.safetensors").write_bytes(adapter_bytes)
        (checkpoint / "checkpoint_fidelity.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": 31_250}), encoding="utf-8"
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
            (checkpoint / name).write_bytes(f"periodic:{name}".encode())
    return run, base


def _stub_semantic_receipt_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sealer, "publication_fidelity_identity", lambda _path: {})
    monkeypatch.setattr(sealer, "validate_audit_receipt", lambda _path, **_kwargs: {})
    monkeypatch.setattr(sealer, "validate_reload_receipt", lambda _path, **_kwargs: {})


def test_seal_is_create_once_recursive_and_idempotent(tmp_path: Path) -> None:
    run, base = _unsealed_tree(tmp_path)

    evidence = sealer.seal_completed_run(run, base, artifact_validator=_accept)

    assert evidence["status"] == "sealed_and_verified"
    receipt_path = run / sealer.SEAL_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    claimed_hash = unsigned.pop("receipt_sha256")
    assert hashlib.sha256(_canonical(unsigned)).hexdigest() == claimed_hash
    assert receipt["preseal_inventory_sha256"] == hashlib.sha256(
        _canonical(receipt["preseal_inventory"])
    ).hexdigest()
    assert all(
        "sha256" in record
        for record in receipt["preseal_inventory"]
        if record["entry_type"] == "file"
    )
    for path in (candidate for candidate in run.rglob("*") if candidate.is_file()):
        assert stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o440
    for path in [run, *(candidate for candidate in run.rglob("*") if candidate.is_dir())]:
        assert stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o550

    again = sealer.seal_completed_run(run, base, artifact_validator=_accept)
    assert again["status"] == "already_sealed_and_verified"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_seal_chmods_files_then_directories_bottom_up_and_root_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = _unsealed_tree(tmp_path)
    calls: list[tuple[Path, bool]] = []
    original = sealer._fchmod_entry

    def observe(path: Path, *, directory: bool, mode: int) -> None:
        calls.append((path, directory))
        original(path, directory=directory, mode=mode)

    monkeypatch.setattr(sealer, "_fchmod_entry", observe)
    sealer.seal_completed_run(run, base, artifact_validator=_accept)

    first_directory = next(index for index, (_, directory) in enumerate(calls) if directory)
    assert all(not directory for _, directory in calls[:first_directory])
    assert calls[-1] == (run, True)
    directory_depths = [
        len(path.relative_to(run).parts)
        for path, directory in calls[first_directory:-1]
        if directory
    ]
    assert directory_depths == sorted(directory_depths, reverse=True)


def test_exact_terminal_and_seven_checkpoint_artifacts_can_be_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = _complete_artifact_tree(tmp_path)
    _stub_semantic_receipt_validation(monkeypatch)

    evidence = sealer.seal_completed_run(run, base)

    assert evidence["status"] == "sealed_and_verified"
    assert evidence["periodic_checkpoint_steps"] == [
        5_000,
        10_000,
        15_000,
        20_000,
        25_000,
        30_000,
    ]
    assert evidence["forced_terminal_checkpoint_step"] == 31_250
    assert evidence["root_terminal_adapter_sha256"] == hashlib.sha256(
        b"terminal adapter"
    ).hexdigest()
    assert evidence["trainer_checkpoint_steps"] == [
        5_000,
        10_000,
        15_000,
        20_000,
        25_000,
        30_000,
        31_250,
    ]
    sealer.require_exact_sealed_modes(run)


def test_terminal_root_adapter_must_equal_checkpoint_31250_before_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = _complete_artifact_tree(tmp_path)
    _stub_semantic_receipt_validation(monkeypatch)
    (run / "checkpoints" / "checkpoint-31250" / "adapter_model.safetensors").write_bytes(
        b"different terminal adapter"
    )

    with pytest.raises(
        sealer.CompletedSFTRunSealError,
        match="root adapter is not byte-identical to checkpoint-31250 adapter",
    ):
        sealer.seal_completed_run(run, base)

    assert not (run / sealer.SEAL_RECEIPT_FILENAME).exists()


def test_missing_periodic_state_is_rejected_before_any_mode_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = _complete_artifact_tree(tmp_path)
    _stub_semantic_receipt_validation(monkeypatch)
    missing = run / "checkpoints" / "checkpoint-15000" / "rng_state.pth"
    missing.unlink()
    sentinel = run / "adapter_model.safetensors"

    with pytest.raises(sealer.CompletedSFTRunSealError, match="rng_state.pth"):
        sealer.seal_completed_run(run, base)
    assert stat.S_IMODE(run.stat().st_mode) == 0o755
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o644
    assert not (run / sealer.SEAL_RECEIPT_FILENAME).exists()


def test_existing_seal_fails_closed_on_writable_mode_drift(tmp_path: Path) -> None:
    run, base = _unsealed_tree(tmp_path)
    sealer.seal_completed_run(run, base, artifact_validator=_accept)
    drifted = run / "root.txt"
    drifted.chmod(0o640)

    with pytest.raises(sealer.CompletedSFTRunSealError, match="file mode drift"):
        sealer.seal_completed_run(run, base, artifact_validator=_accept)
    assert stat.S_IMODE(drifted.stat().st_mode) == 0o640


def test_existing_seal_detects_same_size_content_drift(tmp_path: Path) -> None:
    run, base = _unsealed_tree(tmp_path)
    sealer.seal_completed_run(run, base, artifact_validator=_accept)
    drifted = run / "root.txt"
    original = drifted.read_bytes()
    drifted.chmod(0o640)
    drifted.write_bytes(b"X" * len(original))
    drifted.chmod(0o440)

    with pytest.raises(sealer.CompletedSFTRunSealError, match="inventory drifted"):
        sealer.seal_completed_run(run, base, artifact_validator=_accept)


def test_mutation_during_receipt_commit_is_detected_before_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = _unsealed_tree(tmp_path)
    sentinel = run / "root.txt"
    original_writer = sealer._write_seal_receipt_create_once

    def mutate_after_commit(path: Path, payload: dict[str, object]) -> None:
        original_writer(path, payload)
        sentinel.write_text("tampered\n", encoding="utf-8")

    monkeypatch.setattr(sealer, "_write_seal_receipt_create_once", mutate_after_commit)
    with pytest.raises(sealer.CompletedSFTRunSealError, match="inventory drifted"):
        sealer.seal_completed_run(run, base, artifact_validator=_accept)
    assert stat.S_IMODE(run.stat().st_mode) == 0o755
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o644


@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo"])
def test_unsafe_entry_is_rejected_before_any_mode_mutation(
    tmp_path: Path, unsafe_kind: str
) -> None:
    run, base = _unsealed_tree(tmp_path)
    existing = run / "root.txt"
    if unsafe_kind == "symlink":
        (run / "unsafe").symlink_to(existing)
        expected = "symbolic link"
    else:
        os.mkfifo(run / "unsafe")
        expected = "special entry"

    with pytest.raises(sealer.CompletedSFTRunSealError, match=expected):
        sealer.seal_completed_run(run, base, artifact_validator=_accept)
    assert stat.S_IMODE(run.stat().st_mode) == 0o755
    assert stat.S_IMODE(existing.stat().st_mode) == 0o644
    assert not (run / sealer.SEAL_RECEIPT_FILENAME).exists()


def test_missing_terminal_artifacts_are_rejected_before_mode_mutation(tmp_path: Path) -> None:
    run, base = _unsealed_tree(tmp_path)
    existing = run / "root.txt"

    with pytest.raises(sealer.CompletedSFTRunSealError, match="missing terminal artifacts"):
        sealer.seal_completed_run(run, base)
    assert stat.S_IMODE(run.stat().st_mode) == 0o755
    assert stat.S_IMODE(existing.stat().st_mode) == 0o644
    assert not (run / sealer.SEAL_RECEIPT_FILENAME).exists()


def test_noncanonical_run_root_is_rejected(tmp_path: Path) -> None:
    run, base = _unsealed_tree(tmp_path)
    alias = tmp_path / "run-alias"
    alias.symlink_to(run, target_is_directory=True)

    with pytest.raises(sealer.CompletedSFTRunSealError, match="not a regular directory"):
        sealer.seal_completed_run(alias, base, artifact_validator=_accept)
    assert not (run / sealer.SEAL_RECEIPT_FILENAME).exists()
