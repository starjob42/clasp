"""Exact byte identity for the frozen Jiang-f638 Qwen2.5-7B base snapshot."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from typing import Any

REPOSITORY = "unsloth/Qwen2.5-7B"
REVISION = "86e6e1eccea424c7fb77d59f7d2a9125c0673bfd"
TOKENIZER_JSON_SHA256 = "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"

# This is the complete, already-transferred model directory used by the v8
# campaign. It contains the 15 Hub snapshot files plus the two acquisition and
# transfer receipts. Requiring the complete manifest prevents a revision label
# from being attached to different weights or tokenizer bytes.
EXPECTED_FILES: tuple[tuple[str, int, str], ...] = (
    (".gitattributes", 1_570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"),
    ("README.md", 6_751, "1829aed62c78673b39854d255e9e4accde832a14e6e554db709c0eedb4dc5837"),
    ("added_tokens.json", 605, "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b"),
    ("config.json", 776, "b26a5ed46a16c83389be4f0d6ca36340af9d7089931ea4a844937a1fe8102d4b"),
    ("download_contract.json", 249, "be4ce5b3a8632c022573f9c619574355aac8618f5d25c9b6ef5c706ac2afdb20"),
    ("generation_config.json", 172, "9809d6405255be41d508714c24f5ab8071f079f533e6dabc51c924993b5b58ea"),
    ("merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model-00001-of-00004.safetensors", 4_877_660_776, "37753d1c8dc943058221bf364c12695f6c2afafaa29b470af86d40ce0655e7a5"),
    ("model-00002-of-00004.safetensors", 4_932_751_008, "cc3a200b9309fec7d681071e1d38434ce64644fee1a47fb4fcc02f3e282b3bf3"),
    ("model-00003-of-00004.safetensors", 4_330_865_200, "a59fee34dc91e53bff70df11b369bef26b42c978647f879f39e189764edc296b"),
    ("model-00004-of-00004.safetensors", 1_089_994_880, "5be8567a86b9f8fdf251358a5757baf1e45239e4ba2ed2bb04d81d824a7031c5"),
    ("model.safetensors.index.json", 27_752, "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb"),
    ("model_receipt.json", 3_338, "2d0b266b34f34f22b1a07ed3139e151d0680dfd01549615c7b72e8cdcc15543e"),
    ("special_tokens_map.json", 617, "560aa69e8ac8bb38c99dc2d38be9307e5380fe9047d61eb434d2a6b97272d296"),
    ("tokenizer.json", 11_421_896, TOKENIZER_JSON_SHA256),
    ("tokenizer_config.json", 4_716, "a2537cb661e6d09ae28780d1b6b0b17469b124431b00062ad7321bde80e696c4"),
    ("vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_file_records() -> list[dict[str, Any]]:
    return [
        {"path": name, "size_bytes": size, "sha256": digest}
        for name, size, digest in EXPECTED_FILES
    ]


def _aggregate_sha256(records: list[dict[str, Any]]) -> str:
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(str(record["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(record["size_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(record["sha256"]).encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def validate_jiang_f638_base_receipt(receipt: object) -> dict[str, Any]:
    """Validate a portable tree receipt against the known revision manifest."""

    expected = _expected_file_records()
    if not isinstance(receipt, dict) or set(receipt) != {
        "canonical_path",
        "sha256",
        "file_count",
        "size_bytes",
        "files",
    }:
        raise ValueError("Jiang-f638 base receipt has an invalid schema")
    if (
        not Path(str(receipt.get("canonical_path", ""))).is_absolute()
        or receipt.get("files") != expected
        or receipt.get("file_count") != len(expected)
        or receipt.get("size_bytes") != sum(record["size_bytes"] for record in expected)
        or receipt.get("sha256") != _aggregate_sha256(expected)
    ):
        raise ValueError("Jiang-f638 base receipt differs from revision 86e6")
    return dict(receipt)


def audit_jiang_f638_base_snapshot(path: Path) -> dict[str, Any]:
    """Hash and require the complete known revision-86e6 model directory."""

    raw = path.expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"Jiang-f638 base is not a regular directory: {raw}")
    root = raw.resolve(strict=True)
    if root != raw:
        raise ValueError(f"Jiang-f638 base path must be canonical: {raw} != {root}")
    entries = sorted(root.iterdir(), key=lambda candidate: candidate.name)
    observed_names = [entry.name for entry in entries]
    expected_names = [name for name, _size, _digest in EXPECTED_FILES]
    if observed_names != expected_names:
        raise ValueError(
            "Jiang-f638 base file inventory differs from revision 86e6: "
            f"observed={observed_names}, expected={expected_names}"
        )

    files: list[dict[str, Any]] = []
    for entry, (name, expected_size, expected_digest) in zip(
        entries, EXPECTED_FILES, strict=True
    ):
        before = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or entry.is_symlink():
            raise ValueError(f"Jiang-f638 base contains an unsafe entry: {entry}")
        digest = _sha256_file(entry)
        after = entry.stat(follow_symlinks=False)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise ValueError(f"Jiang-f638 base file changed while hashing: {entry}")
        if after.st_size != expected_size or digest != expected_digest:
            raise ValueError(
                f"Jiang-f638 base file differs from revision 86e6: {name}"
            )
        files.append(
            {"path": name, "size_bytes": expected_size, "sha256": expected_digest}
        )
    receipt = {
        "canonical_path": str(root),
        "sha256": _aggregate_sha256(files),
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
    }
    return validate_jiang_f638_base_receipt(receipt)
