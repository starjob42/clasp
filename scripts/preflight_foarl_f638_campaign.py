#!/usr/bin/env python3
"""Read-only static preflight for one approved corrected-FOARL campaign lane."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    get_foarl_f638_campaign_profile,
)


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"required immutable JSON must not be a symlink: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"required immutable JSON is not a regular file: {resolved}")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o222:
        raise ValueError(f"required immutable JSON is writable (mode {mode:04o}): {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {resolved}")
    return payload


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load frozen helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def static_preflight(
    *,
    method: str,
    task: str,
    train_seed: int,
    final_step: int,
    world_size: int,
    source_root: Path,
    source_manifest: Path,
    forbidden_source_registry: Path,
    model_initialization_registry: Path,
    advancement_policy: Path,
    campaign_id: str,
    prior_stage_receipts: Sequence[Path] = (),
) -> dict[str, Any]:
    """Verify frozen source/registry/policy bindings without writing or scheduling."""

    profile = get_foarl_f638_campaign_profile(method)
    normalized_task = str(task).strip().lower()
    if final_step != profile.final_step(normalized_task):
        raise ValueError("preflight final_step disagrees with immutable FOARL lane")
    if world_size != profile.world_size:
        raise ValueError("preflight world_size disagrees with immutable FOARL lane")
    if profile.required_train_seed is not None and train_seed != profile.required_train_seed:
        raise ValueError("preflight train seed disagrees with immutable native lane")

    raw_root = source_root.expanduser().absolute()
    if raw_root.is_symlink():
        raise ValueError("source root must not be a symlink")
    root = raw_root.resolve(strict=True)
    prepare = _load_script("_foarl_prepare_sota_run", root / "scripts/prepare_sota_run.py")
    advancement = _load_script(
        "_foarl_audit_sota_advancement", root / "scripts/audit_sota_advancement.py"
    )
    raw_manifest = source_manifest.expanduser().absolute()
    if raw_manifest.is_symlink():
        raise ValueError("source manifest must not be a symlink")
    manifest = raw_manifest.resolve(strict=True)
    prepare.validate_source_snapshot(root, manifest)

    raw_forbidden_path = forbidden_source_registry.expanduser().absolute()
    if raw_forbidden_path.is_symlink():
        raise ValueError("forbidden-source registry must not be a symlink")
    forbidden_path = raw_forbidden_path.resolve(strict=True)
    forbidden = prepare.validate_base_sft_source_registry(
        _load_json(forbidden_path),
        expected_tasks=sorted(prepare.TASKS),
    )
    raw_registry_path = model_initialization_registry.expanduser().absolute()
    if raw_registry_path.is_symlink():
        raise ValueError("model-initialization registry must not be a symlink")
    registry_path = raw_registry_path.resolve(strict=True)
    registry = prepare.validate_model_initialization_registry(_load_json(registry_path))
    if (
        registry["forbidden_source_registry_sha256"] != forbidden["registry_sha256"]
        or registry["forbidden_source_tree_manifest_sha256"]
        != forbidden["source_tree_manifest_sha256"]
    ):
        raise ValueError(
            "model-initialization registry belongs to another forbidden-source registry"
        )
    raw_policy_path = advancement_policy.expanduser().absolute()
    if raw_policy_path.is_symlink():
        raise ValueError("advancement policy must not be a symlink")
    policy_path = raw_policy_path.resolve(strict=True)
    policy_mode = stat.S_IMODE(policy_path.stat().st_mode)
    if not policy_path.is_file() or policy_mode & 0o222:
        raise ValueError(
            f"advancement policy must be a read-only regular file: {policy_path}"
        )
    policy = advancement._validate_policy(policy_path)
    prior_authorization = advancement._prior_stage_authorization_audit(
        policy,
        [path.expanduser().resolve(strict=True) for path in prior_stage_receipts],
    )
    if prior_authorization["errors"]:
        raise ValueError(
            "prior-stage authorization failed: "
            + "; ".join(str(error) for error in prior_authorization["errors"][:8])
        )
    expected = {
        "campaign_id": campaign_id,
        "expected_tasks": [normalized_task],
        "expected_methods": [method],
        "train_seeds": [train_seed],
        "target_step": final_step,
        "model_initialization_registry_sha256": registry["registry_sha256"],
    }
    drift = {
        key: {"expected": value, "observed": policy.get(key)}
        for key, value in expected.items()
        if policy.get(key) != value
    }
    requirements = policy.get("contract_requirements", {})
    requirement_drift = {
        "world_size": {"expected": world_size, "observed": requirements.get("world_size")},
        "num_generations": {
            "expected": profile.semantics.num_generations,
            "observed": requirements.get("num_generations"),
        },
        "sft_seed": {"expected": 42, "observed": requirements.get("sft_seed")},
    }
    drift.update(
        {
            f"contract_requirements.{key}": value
            for key, value in requirement_drift.items()
            if value["expected"] != value["observed"]
        }
    )
    if drift:
        raise ValueError(f"advancement policy disagrees with requested FOARL run: {drift}")
    return {
        "status": "PASS",
        "side_effects": {"writes": 0, "scheduler_actions": 0},
        "method": method,
        "task": normalized_task,
        "train_seed": train_seed,
        "final_step": final_step,
        "world_size": world_size,
        "profile_id": profile.semantics.profile_id,
        "classification": profile.attribution,
        "reproduction_claim": "not_an_official_reproduction",
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "forbidden_source_registry_sha256": forbidden["registry_sha256"],
        "model_initialization_registry_sha256": registry["registry_sha256"],
        "advancement_policy_sha256": policy["policy_sha256"],
        "advancement_stage_id": policy["stage_id"],
        "advancement_stage_index": policy["stage_index"],
        "prior_stage_receipt_chain_sha256": hashlib.sha256(
            json.dumps(
                policy["prior_stage_receipt_chain"],
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=FOARL_F638_CAMPAIGN_METHODS, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--forbidden-source-registry", type=Path, required=True)
    parser.add_argument("--model-initialization-registry", type=Path, required=True)
    parser.add_argument("--advancement-policy", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--prior-stage-receipt", action="append", type=Path, default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = static_preflight(
            method=args.method,
            task=args.task,
            train_seed=args.train_seed,
            final_step=args.final_step,
            world_size=args.world_size,
            source_root=args.source_root,
            source_manifest=args.source_manifest,
            forbidden_source_registry=args.forbidden_source_registry,
            model_initialization_registry=args.model_initialization_registry,
            advancement_policy=args.advancement_policy,
            campaign_id=args.campaign_id,
            prior_stage_receipts=args.prior_stage_receipt,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[FOARLPreflightError] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
