"""All-five-task successor of the frozen 2026-09-07 FOARL development port.

Released reward heads, sample-standard-deviation normalization, stock GRPO
loss, beta=.05 independent frozen SFT reference, and historical sampling are
unchanged. This is an explicit new source/profile contract, not a relabeling
of either the earlier two-task experiments or an official reproduction.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

from clasp.baselines.foarl_f638_dev_20260907 import (
    FoarlF638DevRuntime, build_foarl_f638_dev_runtime, diagnostic_updates,
    verify_dev_source_manifest,
)
from clasp.baselines.llmcosolver_f638_training import FOARL_F638_TASKS

PROFILE_ID = "foarl-f638-matched-b9-all-co-dev-20260910-v1"
SOURCE_VERSION = "foarl-f638-all-co-dev-20260910-frozen-manifest-successor"
SOURCE_REQUIRED_FILES = (
    "src/clasp/baselines/foarl_f638_all_tasks_dev.py",
    "src/clasp/entrypoints/train_foarl_f638_all_tasks_dev.py",
    "experiments/grpo_external_baseline_all_tasks_20260910/train_controls_a100.slurm",
)


def verify_all_tasks_source_manifest(manifest_path, expected_sha256, *, source_root=None):
    binding = verify_dev_source_manifest(manifest_path, expected_sha256, source_root=source_root)
    root = Path(binding["source_root"])
    names = {line.split("  ", 1)[1] for line in Path(manifest_path).read_text().splitlines()}
    if missing := set(SOURCE_REQUIRED_FILES) - names:
        raise ValueError(f"All-task FOARL manifest omits required successor files: {sorted(missing)}")
    for name in SOURCE_REQUIRED_FILES:
        payload = (root / name).read_bytes()
        binding["required_files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
        }
    binding["version"] = SOURCE_VERSION
    return binding


@dataclass(frozen=True)
class FoarlF638AllTasksDevRuntime(FoarlF638DevRuntime):
    def validate_args(self, args):
        super().validate_args(args)
        if len(args.tasks) != 1:
            raise ValueError("All-task FOARL comparison requires one specialist per run")
        if getattr(args, "save_total_limit", None) != 1:
            raise ValueError("All-task FOARL requires exactly one terminal checkpoint")
        if (getattr(args, "resume_from_checkpoint", None)
                or any(os.environ.get(key, "") for key in
                       ("RESUME_FROM_CHECKPOINT", "DEV_RESUME_CHECKPOINT", "GRPO_STOP_AT_GLOBAL_STEP"))
                or any(value for key, value in os.environ.items()
                       if key == "GRPO_REBASE_LOADED_SCHEDULER_AT_STEP" or key.startswith("GRPO_EXTENSION_"))):
            raise ValueError("All-task FOARL controls require fresh uninterrupted runs without resume/extension")

    def metadata(self):
        result = super().metadata()
        result["all_tasks_extension"] = {
            "profile_id": PROFILE_ID, "supported_tasks": list(FOARL_F638_TASKS),
            "historical_reward_and_optimizer_recipe_preserved": True,
            "fresh_uninterrupted_updates": diagnostic_updates() or 100,
            "initialization": "task_specific_common_corrected_sft",
            "resume_allowed": False,
        }
        return result


def build_all_tasks_foarl_runtime(source_binding):
    previous = build_foarl_f638_dev_runtime(source_binding)
    semantics = replace(
        previous.semantics, profile_id=PROFILE_ID, supported_tasks=FOARL_F638_TASKS,
        publication_label="FOARL-f638 all-CO common-budget development port (2026-09-10)",
        trainer_source_version=SOURCE_VERSION,
    )
    return FoarlF638AllTasksDevRuntime(
        semantics=semantics, source_binding=copy.deepcopy(dict(source_binding)),
    )
