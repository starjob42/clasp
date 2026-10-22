"""Versioned identities for the shared GRPO training entry point.

Whole-file hashes used by closed experiments are historical evidence, not a
pointer that should silently follow the mutable workspace.  This registry keeps
the closed TSP Frontier-CDF trainer identity distinct from the later executable
FOARL runtime that shares the same source path.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


TRAIN_GRPO_RELATIVE_PATH = "src/clasp/entrypoints/train_grpo.py"

TSP_FCDF_BETA_PILOT_TRAINER_VERSION = "tsp-fcdf-beta-pilot-v4-historical"
TSP_FCDF_BETA_PILOT_TRAINER_SHA256 = (
    "a35e4a4cb0882bc221469bccb49b60a9e5717b4275169b8b5d13412975923c49"
)

FOARL_F638_CORRECTED_TRAINER_VERSION = "foarl-f638-corrected-shared-trainer-v1"
FOARL_F638_CORRECTED_TRAINER_SHA256 = (
    "bf999a76a33a30f11386e1ea1b0e21f3da99bf6e74b32e850dc057350a8c77c6"
)

FOARL_TEACHER_DEV_TRAINER_VERSION = "foarl-teacher-matched-development-20260909-v1"
FOARL_TEACHER_DEV_TRAINER_SHA256 = (
    "61a9e00b8dbd4cf40ef36c402d696e2f3be814318a22ea702f530dfda5b9b3fa"
)


@dataclass(frozen=True)
class TrainGrpoSourceVersion:
    """One immutable whole-file source identity and its scientific scope."""

    version: str
    sha256: str
    claim_scope: str
    executable_from_current_tree: bool

    def receipt(self) -> dict[str, object]:
        return {
            "version": self.version,
            "path": TRAIN_GRPO_RELATIVE_PATH,
            "sha256": self.sha256,
            "claim_scope": self.claim_scope,
            "executable_from_current_tree": self.executable_from_current_tree,
        }


TSP_FCDF_BETA_PILOT_TRAINER_SOURCE = TrainGrpoSourceVersion(
    version=TSP_FCDF_BETA_PILOT_TRAINER_VERSION,
    sha256=TSP_FCDF_BETA_PILOT_TRAINER_SHA256,
    claim_scope="closed_historical_development_pilot",
    executable_from_current_tree=False,
)
FOARL_F638_CORRECTED_TRAINER_SOURCE = TrainGrpoSourceVersion(
    version=FOARL_F638_CORRECTED_TRAINER_VERSION,
    sha256=FOARL_F638_CORRECTED_TRAINER_SHA256,
    claim_scope="foarl_f638_corrected_common_contract_reimplementation",
    executable_from_current_tree=False,
)
FOARL_TEACHER_DEV_TRAINER_SOURCE = TrainGrpoSourceVersion(
    version=FOARL_TEACHER_DEV_TRAINER_VERSION,
    sha256=FOARL_TEACHER_DEV_TRAINER_SHA256,
    claim_scope="foarl_teacher_fixed_horizon_development",
    executable_from_current_tree=True,
)

TRAIN_GRPO_SOURCE_VERSIONS: Mapping[str, TrainGrpoSourceVersion] = MappingProxyType(
    {
        TSP_FCDF_BETA_PILOT_TRAINER_VERSION: TSP_FCDF_BETA_PILOT_TRAINER_SOURCE,
        FOARL_F638_CORRECTED_TRAINER_VERSION: FOARL_F638_CORRECTED_TRAINER_SOURCE,
        FOARL_TEACHER_DEV_TRAINER_VERSION: FOARL_TEACHER_DEV_TRAINER_SOURCE,
    }
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"versioned trainer source must be a regular unlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    identity = lambda stat: (  # noqa: E731 - compact immutable stat projection
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"trainer source changed while being fingerprinted: {path}")
    return digest.hexdigest()


def identify_train_grpo_source(project_root: Path | None = None) -> TrainGrpoSourceVersion | None:
    """Return the registered identity matching the live trainer, if any."""

    root = (project_root or _project_root()).resolve()
    observed = _sha256_regular_file(root / TRAIN_GRPO_RELATIVE_PATH)
    for source_version in TRAIN_GRPO_SOURCE_VERSIONS.values():
        if observed == source_version.sha256:
            return source_version
    return None


def require_train_grpo_source(
    version: str,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Fail closed unless the live trainer is exactly the requested version."""

    try:
        expected = TRAIN_GRPO_SOURCE_VERSIONS[version]
    except KeyError as exc:
        raise RuntimeError(f"unknown train_grpo source version: {version!r}") from exc
    root = (project_root or _project_root()).resolve()
    path = root / TRAIN_GRPO_RELATIVE_PATH
    observed = _sha256_regular_file(path)
    if observed != expected.sha256:
        identified = identify_train_grpo_source(root)
        identified_version = identified.version if identified is not None else "unregistered"
        raise RuntimeError(
            "train_grpo source identity mismatch: "
            f"requested_version={version}, expected={expected.sha256}, observed={observed}, "
            f"observed_version={identified_version}. Historical hashes must not be reissued "
            "for successor trainer code."
        )
    return {
        **expected.receipt(),
        "observed_sha256": observed,
        "status": "PASS",
    }


__all__ = [
    "FOARL_TEACHER_DEV_TRAINER_SHA256",
    "FOARL_TEACHER_DEV_TRAINER_SOURCE",
    "FOARL_TEACHER_DEV_TRAINER_VERSION",
    "FOARL_F638_CORRECTED_TRAINER_SHA256",
    "FOARL_F638_CORRECTED_TRAINER_SOURCE",
    "FOARL_F638_CORRECTED_TRAINER_VERSION",
    "TRAIN_GRPO_RELATIVE_PATH",
    "TRAIN_GRPO_SOURCE_VERSIONS",
    "TSP_FCDF_BETA_PILOT_TRAINER_SHA256",
    "TSP_FCDF_BETA_PILOT_TRAINER_SOURCE",
    "TSP_FCDF_BETA_PILOT_TRAINER_VERSION",
    "TrainGrpoSourceVersion",
    "identify_train_grpo_source",
    "require_train_grpo_source",
]
