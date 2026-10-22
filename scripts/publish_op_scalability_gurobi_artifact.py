#!/usr/bin/env python3
"""Atomically seal the audited Gurobi wheel for the OP development sweep.

The production CLI has no path/hash overrides.  It accepts only the exact
read-only-probed Jean Zay source identity below and atomically creates one new
directory containing a mode-0440 byte-identical wheel plus a self-hashed
publication receipt.  It never reads or authorizes confirmation data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROTOCOL = "clasp-op-scalability-sealed-gurobi-artifact-v1"
STATUS = "sealed_development_only_solver_artifact"
SOURCE_PATH = Path(
    "/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/"
    "gurobi_13.0.1_audit/wheels/"
    "gurobipy-13.0.1-cp310-cp310-manylinux2014_x86_64."
    "manylinux_2_17_x86_64.whl"
)
DESTINATION_DIR = Path(
    "/lustre/fsn1/projects/rech/vas/uag72kz/"
    "clasp_op_scalability_development_iclr_20260826_v1/"
    "inputs/sealed_gurobi_artifact_v1"
)
ARTIFACT_NAME = SOURCE_PATH.name
EXPECTED_SHA256 = "7fc1b5ff46530aee391ab0f616620a1399abea0778b25e374b37f7ed222d5dfb"
EXPECTED_SIZE_BYTES = 15_012_230
EXPECTED_SOURCE_DEVICE = 2_459_147_420
EXPECTED_SOURCE_INODE = 270_218_045_232_884_466
EXPECTED_SOURCE_MODE = "0644"
EXPECTED_SOURCE_NLINK = 1
EXPECTED_SOURCE_UID = 304_433
EXPECTED_SOURCE_GID = 300_401
SEALED_FILE_MODE = 0o440
SEALED_DIRECTORY_MODE = 0o550


@dataclass(frozen=True)
class SourcePolicy:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: str
    nlink: int
    uid: int
    gid: int


PRODUCTION_POLICY = SourcePolicy(
    path=SOURCE_PATH,
    sha256=EXPECTED_SHA256,
    size_bytes=EXPECTED_SIZE_BYTES,
    device=EXPECTED_SOURCE_DEVICE,
    inode=EXPECTED_SOURCE_INODE,
    mode=EXPECTED_SOURCE_MODE,
    nlink=EXPECTED_SOURCE_NLINK,
    uid=EXPECTED_SOURCE_UID,
    gid=EXPECTED_SOURCE_GID,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(value: Mapping[str, object], field: str) -> dict[str, object]:
    sealed = copy.deepcopy(dict(value))
    sealed.pop(field, None)
    sealed[field] = hashlib.sha256(_canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _verify_self_hash(value: Mapping[str, object], field: str, *, label: str) -> None:
    stored = value.get(field)
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError(f"{label} has no valid {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if observed != stored:
        raise ValueError(f"{label} self-hash mismatch")


def _identity(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"symbolic links are forbidden: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError(f"missing artifact: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": int(metadata.st_size),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": int(metadata.st_nlink),
        "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
    }


def validate_source(policy: SourcePolicy = PRODUCTION_POLICY) -> dict[str, object]:
    expected_path = policy.path
    if not expected_path.is_absolute() or expected_path.resolve() != expected_path:
        raise ValueError("source policy path must be absolute and canonical")
    observed = _identity(expected_path)
    expected = {
        "path": str(expected_path),
        "sha256": policy.sha256,
        "size_bytes": policy.size_bytes,
        "device": policy.device,
        "inode": policy.inode,
        "mode": policy.mode,
        "nlink": policy.nlink,
        "uid": policy.uid,
        "gid": policy.gid,
    }
    if observed != expected:
        raise ValueError(
            "audited Gurobi source identity mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def publish_artifact(
    *,
    policy: SourcePolicy = PRODUCTION_POLICY,
    destination_dir: Path = DESTINATION_DIR,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    """Copy and seal one artifact into a new directory without overwrite paths."""

    source_identity = validate_source(policy)
    destination = destination_dir.expanduser()
    if not destination.is_absolute() or destination.resolve() != destination:
        raise ValueError("destination directory must be absolute and canonical")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite sealed artifact directory: {destination}")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise ValueError(f"sealed artifact parent must already be a canonical directory: {parent}")
    timestamp = created_at_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not timestamp.endswith("Z"):
        raise ValueError("created_at_utc must be an explicit UTC timestamp")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        target = temporary / ARTIFACT_NAME
        with policy.path.open("rb") as source_handle, target.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        target.chmod(SEALED_FILE_MODE)
        target_identity = _identity(target)
        for field, expected in (
            ("sha256", policy.sha256),
            ("size_bytes", policy.size_bytes),
            ("mode", "0440"),
            ("nlink", 1),
        ):
            if target_identity[field] != expected:
                raise ValueError(f"sealed artifact {field} mismatch")

        publisher_path = Path(__file__).resolve()
        final_target = destination / ARTIFACT_NAME
        final_identity = dict(target_identity)
        final_identity["path"] = str(final_target)
        receipt = _seal(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": STATUS,
                "created_at_utc": timestamp,
                "development_only": True,
                "confirmation_data_access_authorized": False,
                "confirmation_solving_authorized": False,
                "production_authorization": False,
                "publisher": {
                    "path": str(publisher_path),
                    "sha256": _sha256_file(publisher_path),
                },
                "source": source_identity,
                "sealed_artifact": final_identity,
            },
            "receipt_sha256",
        )
        receipt_path = temporary / "publication_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        receipt_path.chmod(SEALED_FILE_MODE)
        temporary.chmod(SEALED_DIRECTORY_MODE)
        os.replace(temporary, destination)
        return receipt
    except BaseException:
        if temporary.exists():
            for path in temporary.iterdir():
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
            temporary.chmod(0o700)
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_publication(
    destination_dir: Path = DESTINATION_DIR,
    *,
    policy: SourcePolicy = PRODUCTION_POLICY,
    expected_publisher_sha256: str | None = None,
) -> dict[str, object]:
    """Verify the sealed copy/receipt without trusting pathname metadata."""

    destination = destination_dir.expanduser()
    if destination.is_symlink() or not destination.is_dir() or destination.resolve() != destination:
        raise ValueError("sealed artifact directory is missing, noncanonical, or a symlink")
    directory_mode = stat.S_IMODE(destination.stat().st_mode)
    if directory_mode & 0o222:
        raise ValueError("sealed artifact directory remains writable")
    receipt_path = destination / "publication_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("sealed artifact receipt is missing or a symlink")
    if stat.S_IMODE(receipt_path.stat().st_mode) != SEALED_FILE_MODE:
        raise ValueError("sealed artifact receipt mode must equal 0440")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("sealed artifact receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("sealed artifact receipt must be an object")
    _verify_self_hash(receipt, "receipt_sha256", label="sealed artifact receipt")
    if receipt.get("protocol") != PROTOCOL or receipt.get("status") != STATUS:
        raise ValueError("sealed artifact receipt protocol/status mismatch")
    if (
        receipt.get("development_only") is not True
        or receipt.get("confirmation_data_access_authorized") is not False
        or receipt.get("confirmation_solving_authorized") is not False
        or receipt.get("production_authorization") is not False
    ):
        raise ValueError("sealed artifact receipt authorization policy is unsafe")
    publisher = receipt.get("publisher")
    if not isinstance(publisher, dict) or not isinstance(publisher.get("sha256"), str):
        raise ValueError("sealed artifact receipt has no publisher identity")
    if expected_publisher_sha256 is not None and publisher["sha256"] != expected_publisher_sha256:
        raise ValueError("sealed artifact publisher hash mismatch")
    source = receipt.get("source")
    expected_source = {
        "path": str(policy.path),
        "sha256": policy.sha256,
        "size_bytes": policy.size_bytes,
        "device": policy.device,
        "inode": policy.inode,
        "mode": policy.mode,
        "nlink": policy.nlink,
        "uid": policy.uid,
        "gid": policy.gid,
    }
    if source != expected_source:
        raise ValueError("sealed artifact receipt source identity mismatch")
    artifact_path = destination / ARTIFACT_NAME
    artifact = _identity(artifact_path)
    if artifact["sha256"] != policy.sha256 or artifact["size_bytes"] != policy.size_bytes:
        raise ValueError("sealed artifact content identity mismatch")
    if artifact["mode"] != "0440" or artifact["nlink"] != 1:
        raise ValueError("sealed artifact immutability identity mismatch")
    receipt_artifact = receipt.get("sealed_artifact")
    if not isinstance(receipt_artifact, dict):
        raise ValueError("sealed artifact receipt has no target identity")
    for field in ("path", "sha256", "size_bytes", "mode", "nlink", "uid", "gid"):
        if receipt_artifact.get(field) != artifact.get(field):
            raise ValueError(f"sealed artifact receipt target {field} mismatch")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--publish", action="store_true")
    action.add_argument("--verify-publication", action="store_true")
    parser.add_argument("--expected-publisher-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.preflight_only:
            source = validate_source()
            if DESTINATION_DIR.exists() or DESTINATION_DIR.is_symlink():
                raise FileExistsError(
                    f"refusing a non-fresh destination: {DESTINATION_DIR}"
                )
            if not DESTINATION_DIR.parent.is_dir():
                raise ValueError(
                    f"destination parent must be staged first: {DESTINATION_DIR.parent}"
                )
            result: dict[str, object] = {
                "status": "READY_CREATE_ONCE_SEAL",
                "source": source,
                "destination": str(DESTINATION_DIR),
                "confirmation_solving_authorized": False,
            }
        elif args.publish:
            result = publish_artifact()
        else:
            result = verify_publication(
                expected_publisher_sha256=args.expected_publisher_sha256
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"[OPArtifactPublisherError] {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
