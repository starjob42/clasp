from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from clasp.models import jiang_f638_base as base


def _record(name: str, payload: bytes) -> tuple[str, int, str]:
    return name, len(payload), hashlib.sha256(payload).hexdigest()


def test_exact_base_manifest_binds_every_file_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "config.json": b"config",
        "model.safetensors": b"weights",
        "tokenizer.json": b"tokenizer",
    }
    monkeypatch.setattr(
        base,
        "EXPECTED_FILES",
        tuple(_record(name, payload) for name, payload in sorted(payloads.items())),
    )
    root = tmp_path / "base"
    root.mkdir()
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)

    receipt = base.audit_jiang_f638_base_snapshot(root)

    assert receipt["file_count"] == 3
    assert receipt["files"][2]["path"] == "tokenizer.json"
    assert base.validate_jiang_f638_base_receipt(receipt) == receipt

    (root / "tokenizer.json").write_bytes(b"tokenizeS")
    with pytest.raises(ValueError, match="differs from revision"):
        base.audit_jiang_f638_base_snapshot(root)


def test_exact_base_manifest_rejects_unregistered_loader_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base, "EXPECTED_FILES", (_record("config.json", b"{}"),))
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    (root / "modeling_qwen2.py").write_text("# untrusted loader\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory"):
        base.audit_jiang_f638_base_snapshot(root)
