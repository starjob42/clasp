from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_op_scalability_gurobi_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_op_scalability_gurobi_artifact", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


def _policy(source: Path) -> object:
    metadata = source.stat()
    return publisher.SourcePolicy(
        path=source.resolve(),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=metadata.st_size,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=f"{stat.S_IMODE(metadata.st_mode):04o}",
        nlink=metadata.st_nlink,
        uid=metadata.st_uid,
        gid=metadata.st_gid,
    )


def test_publisher_atomically_creates_exact_read_only_copy_and_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / publisher.ARTIFACT_NAME
    source.write_bytes(b"test audited gurobipy wheel bytes")
    source.chmod(0o644)
    policy = _policy(source)
    destination = tmp_path / "sealed"
    receipt = publisher.publish_artifact(
        policy=policy,
        destination_dir=destination,
        created_at_utc="2026-08-26T09:24:14Z",
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o550
    artifact = destination / publisher.ARTIFACT_NAME
    receipt_path = destination / "publication_receipt.json"
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o440
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o440
    assert artifact.read_bytes() == source.read_bytes()
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_sha256")
    assert hashlib.sha256(publisher._canonical_json_bytes(unsigned)).hexdigest() == digest
    assert receipt["confirmation_data_access_authorized"] is False
    assert receipt["confirmation_solving_authorized"] is False
    assert receipt["production_authorization"] is False
    expected_publisher_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert receipt["publisher"]["sha256"] == expected_publisher_hash
    verified = publisher.verify_publication(
        destination,
        policy=policy,
        expected_publisher_sha256=expected_publisher_hash,
    )
    assert verified["receipt_sha256"] == receipt["receipt_sha256"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publisher.publish_artifact(policy=policy, destination_dir=destination)


def test_publisher_rejects_source_identity_drift_before_creating_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / publisher.ARTIFACT_NAME
    source.write_bytes(b"original bytes")
    source.chmod(0o644)
    policy = _policy(source)
    source.write_bytes(b"changed bytes")
    destination = tmp_path / "sealed"
    with pytest.raises(ValueError, match="source identity mismatch"):
        publisher.publish_artifact(policy=policy, destination_dir=destination)
    assert not destination.exists()


def test_publication_verifier_rejects_receipt_or_artifact_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / publisher.ARTIFACT_NAME
    source.write_bytes(b"test audited wheel")
    source.chmod(0o644)
    policy = _policy(source)
    destination = tmp_path / "sealed"
    publisher.publish_artifact(policy=policy, destination_dir=destination)
    receipt_path = destination / "publication_receipt.json"
    receipt_path.chmod(0o640)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["confirmation_solving_authorized"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o440)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        publisher.verify_publication(destination, policy=policy)
