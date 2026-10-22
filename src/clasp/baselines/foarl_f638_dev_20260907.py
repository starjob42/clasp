"""FOARL reward/loss port for the 2026-09-07 common-budget development study.

This new profile deliberately leaves both historical campaign profiles intact.
It is neither the paper recipe nor a reproduction of an author checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .llmcosolver_f638_training import (
    FOARL_F638_SEMANTICS,
    FoarlF638CorrectedRuntimeProfile,
)

METHOD_ID = "foarl_f638_dev"
PROFILE_ID = "foarl-f638-matched-b9-dev-20260907-v1"
SOURCE_VERSION = "foarl-f638-dev-20260907-frozen-manifest-successor"
COMMON_PROMPT_SHA256 = "4a3abb36b6b28e1f4d18463e90943055676a3cf97dcaf7af70eacdc5830205ca"
SOURCE_REQUIRED_FILES = (
    "src/clasp/baselines/foarl_f638_dev_20260907.py",
    "src/clasp/baselines/llmcosolver_f638_training.py",
    "src/clasp/baselines/llmcosolver_f638_rewards.py",
    "src/clasp/entrypoints/train_foarl_f638_dev_20260907.py",
    "src/clasp/entrypoints/train_foarl_f638_corrected.py",
    "src/clasp/entrypoints/train_grpo.py",
    "src/clasp/models/training_fidelity.py",
    "src/clasp/trainers/primal_dual_grpo.py",
)

# Values that change the common train budget or its numerical schedule are
# immutable. Dataset/SFT identities remain independently bound by the launcher.
DEV_DEFAULTS = {
    "no_reason": True,
    "num_generations": 9,
    "generation_batch_size": 36,
    "per_device_train_batch_size": 3,
    "per_device_eval_batch_size": 9,
    "gradient_accumulation_steps": 3,
    "learning_rate": 1e-6,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "linear",
    "seed": 43,
    "max_steps": 100,
    "save_steps": 100,
    "logging_steps": 1,
    "max_prompt_length": 29_768,
    "max_completion_length": 3_000,
    "generation_temperature": 0.7,
    "generation_top_p": 0.9,
    "generation_top_k": 50,
    "generation_repetition_penalty": 1.0,
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
}


def diagnostic_updates(environment: Mapping[str, str] | None = None) -> int | None:
    value = (os.environ if environment is None else environment).get(
        "DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES", ""
    )
    if value not in ("", "0", "5"):
        raise ValueError("DEV_DISTRIBUTED_DIAGNOSTIC_UPDATES must be empty, 0, or exactly 5")
    return 5 if value == "5" else None


def verify_dev_source_manifest(
    manifest_path: str | Path,
    expected_sha256: str,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Verify an externally frozen sha256sum manifest against the executing tree.

    The expected digest comes from the pre-launch campaign declaration. This
    does not relabel an old fixed source hash or bless the current tree itself.
    """

    root = (source_root or Path(__file__).resolve().parents[3]).resolve()
    manifest = Path(manifest_path).absolute()
    if manifest.is_symlink() or manifest.parent.resolve() != root or not manifest.is_file():
        raise ValueError("FOARL dev manifest must be a regular file in the executing source root")
    raw = manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest != expected_sha256:
        raise ValueError("FOARL dev frozen source manifest digest mismatch")
    identities: dict[str, dict[str, Any]] = {}
    for line in raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("FOARL dev source manifest requires sha256sum text lines")
        expected, relative = match.groups()
        locator = PurePosixPath(relative)
        if (
            locator.is_absolute()
            or ".." in locator.parts
            or str(locator) != relative
            or relative in identities
        ):
            raise ValueError(f"invalid or duplicate FOARL source locator: {relative}")
        path = root / relative
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError(f"FOARL dev source must be an unlinked regular file: {relative}")
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise ValueError(f"FOARL dev source file digest mismatch: {relative}")
        identities[relative] = {"sha256": observed, "size_bytes": len(payload)}
    required = set(SOURCE_REQUIRED_FILES)
    required.update(path.relative_to(root).as_posix() for path in (root / "src/clasp").rglob("*.py"))
    missing = sorted(required - identities.keys())
    if missing:
        raise ValueError(f"FOARL dev source manifest omits Python/required sources: {missing}")
    return {
        "status": "passed_frozen_manifest_successor_source",
        "version": SOURCE_VERSION,
        "source_root": str(root),
        "manifest": {"path": str(manifest), "sha256": digest, "size_bytes": len(raw)},
        "verified_file_count": len(identities),
        "required_files": {name: identities[name] for name in SOURCE_REQUIRED_FILES},
        "historical_source_hashes_unchanged": True,
    }


@dataclass(frozen=True)
class FoarlF638DevRuntime(FoarlF638CorrectedRuntimeProfile):
    source_binding: Mapping[str, Any] = field(default_factory=dict)

    def validate_args(self, args: Any) -> None:
        # Reuse every historical reward/loss gate. This profile explicitly
        # replaces only its reasoning-enabled prompt restriction; the actual
        # namespace stays unchanged and is independently checked below.
        legacy_validation_args = copy.copy(args)
        legacy_validation_args.no_reason = False
        super().validate_args(legacy_validation_args)
        from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE

        if hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest() != COMMON_PROMPT_SHA256:
            raise ValueError("FOARL dev common no-reasoning prompt bytes changed")
        if os.environ.get("WORLD_SIZE", "1") != "4":
            raise ValueError("FOARL dev requires exactly WORLD_SIZE=4")
        expected = dict(DEV_DEFAULTS)
        diagnostic = diagnostic_updates()
        if diagnostic is not None:
            expected.update(max_steps=diagnostic, save_steps=diagnostic)
        differences = {
            name: {"expected": value, "observed": getattr(args, name, None)}
            for name, value in expected.items()
            if getattr(args, name, None) != value
        }
        if differences:
            raise ValueError(f"FOARL dev common-budget recipe mismatch: {differences}")
        if os.environ.get("GRPO_DATA_SEED", "43") != "43":
            raise ValueError("FOARL dev requires GRPO_DATA_SEED=43")
        if bool(getattr(args, "grpo_verified_teacher", False)):
            raise ValueError("FOARL dev forbids an auxiliary teacher objective")
        if getattr(args, "resolved_generation_batch_size", 36) != 36:
            raise ValueError("FOARL dev resolved generation batch must equal 36")
        # With an explicit generation_batch_size, shared argparse leaves this
        # as None; GRPOConfig derives 36/(4*3)=3 at construction.
        if getattr(args, "resolved_steps_per_generation", None) not in (None, 3):
            raise ValueError("FOARL dev resolved steps per generation must equal 3")

    def metadata(self) -> dict[str, Any]:
        if self.source_binding.get("status") != "passed_frozen_manifest_successor_source":
            raise ValueError("FOARL dev runtime receipt requires a verified frozen source manifest")
        receipt = super().metadata()
        receipt.update(
            reproduction_claim="not_an_official_reproduction",
            study_scope=(
                "distributed_correctness_diagnostic_excluded_from_comparison"
                if diagnostic_updates() is not None
                else "inspected_development_common_budget_100_updates"
            ),
            common_budget={**DEV_DEFAULTS, "world_size": 4},
            source_binding=copy.deepcopy(dict(self.source_binding)),
            changes_from_released_f638=[
                "corrected local SFT checkpoint, response mask, bias persistence, and cleaned data",
                "TRL 0.24 port with an explicit independent frozen SFT reference",
                "9 samples per prompt, 4 ranks, global rollout/update batch 36",
                "100 updates, seed 43, weight decay 0.01, warmup ratio 0.03",
                "prompt cap 29768 and completion cap 3000",
            ],
            comparison_limit="same rollout count is not equal training computation",
            no_reason=True,
            mis_jiang_response_output=False,
        )
        receipt["changes_from_released_f638"].append(
            "common no-reasoning Jiang inference prompt and MIS response output mode 0"
        )
        if diagnostic_updates() is not None:
            receipt["common_budget"].update(max_steps=5, save_steps=5)
        return receipt


def build_foarl_f638_dev_runtime(source_binding: Mapping[str, Any]) -> FoarlF638DevRuntime:
    trainer = source_binding.get("required_files", {}).get("src/clasp/entrypoints/train_grpo.py", {})
    semantics = replace(
        FOARL_F638_SEMANTICS,
        method_id=METHOD_ID,
        profile_id=PROFILE_ID,
        publication_label="FOARL-f638 common-budget development port (2026-09-07)",
        supported_tasks=("cvrp", "mis"),
        num_generations=9,
        prompt_sha256=COMMON_PROMPT_SHA256,
        trainer_source_version=SOURCE_VERSION,
        trainer_source_sha256=str(trainer.get("sha256", "")),
    )
    return FoarlF638DevRuntime(semantics=semantics, source_binding=copy.deepcopy(dict(source_binding)))
