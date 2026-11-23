"""Matched FOARL / FOARL+teacher development with a fixed resumable horizon."""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .foarl_f638_dev_20260907 import COMMON_PROMPT_SHA256, DEV_DEFAULTS
from .llmcosolver_f638_training import FOARL_F638_SEMANTICS, FoarlF638CorrectedRuntimeProfile

COMMON_DEFAULTS = {**DEV_DEFAULTS, "max_steps": 400, "save_steps": 100, "save_total_limit": 4}
PROFILE_ID = "foarl-teacher-matched-development-20260909-v1"


@dataclass(frozen=True)
class FoarlTeacherDevRuntime(FoarlF638CorrectedRuntimeProfile):
    source_binding: Mapping[str, Any] = field(default_factory=dict)
    teacher_weight: float = 0.0
    diagnostic: bool = False
    verified_teacher_extension: bool = True

    def validate_args(self, args):
        legacy = copy.copy(args)
        legacy.no_reason = False
        super().validate_args(legacy)
        from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE
        if hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest() != COMMON_PROMPT_SHA256:
            raise ValueError("FOARL teacher common prompt changed")
        if self.teacher_weight not in (0.0, 0.05):
            raise ValueError("Declared teacher weights are 0 and 0.05")
        expected = {**COMMON_DEFAULTS, "foarl_teacher_weight": self.teacher_weight,
                    "foarl_teacher_diagnostic": self.diagnostic}
        if self.diagnostic:
            expected["save_steps"] = 5
        drift = {key: (value, getattr(args, key, None)) for key, value in expected.items()
                 if getattr(args, key, None) != value}
        if drift:
            raise ValueError(f"FOARL teacher common recipe mismatch: {drift}")
        if os.environ.get("WORLD_SIZE", "1") != "4":
            raise ValueError("FOARL teacher requires WORLD_SIZE=4")
        if os.environ.get("GRPO_DATA_SEED", "43") != "43":
            raise ValueError("FOARL teacher requires GRPO_DATA_SEED=43")
        if os.environ.get("GRPO_VERIFIED_IMPROVEMENT", "0") != "0":
            raise ValueError("Legacy teacher environment cannot select the new extension")
        if getattr(args, "resolved_generation_batch_size", 36) != 36:
            raise ValueError("FOARL teacher resolved generation batch must equal 36")
        if getattr(args, "resolved_steps_per_generation", None) not in (None, 3):
            raise ValueError("FOARL teacher requires three microbatches per rollout buffer")
        stop = os.environ.get("GRPO_STOP_AT_GLOBAL_STEP", "")
        boundaries = (5, 10) if self.diagnostic else (100, 200, 300, 400)
        if not stop.isdigit() or int(stop) not in boundaries:
            raise ValueError(f"FOARL teacher stop must be in {boundaries}")
        if os.environ.get("GRPO_STRICT_OPTIMIZER_RESUME") != "1":
            raise ValueError("FOARL teacher requires strict optimizer resume")

    def validate_grpo_config(self, config):
        super().validate_grpo_config(config)
        if bool(getattr(config, "ignore_data_skip", False)):
            raise ValueError("FOARL continuation must resume the original prompt stream")
        if int(config.max_steps) != 400 or int(config.steps_per_generation) != 3:
            raise ValueError("FOARL continuation requires the fixed horizon and full buffers")

    def validate_dataset(self, dataset, *, label):
        receipt = super().validate_dataset(dataset, label=label)
        # TRL's repeat sampler has independent RNG state. Stay in the first
        # epoch until a separate across-epoch continuation audit exists.
        if label == "train" and len(dataset) < 1600:
            raise ValueError("400 updates require at least 1600 retained training prompts")
        return receipt

    def metadata(self):
        if self.source_binding.get("status") != "passed_frozen_manifest_successor_source":
            raise ValueError("FOARL teacher requires a verified frozen source manifest")
        receipt = super().metadata()
        receipt.update(
            reproduction_claim="not_an_official_reproduction",
            study_scope="distributed_diagnostic" if self.diagnostic else "development_only",
            common_budget={**COMMON_DEFAULTS, "world_size": 4,
                           "save_steps": 5 if self.diagnostic else 100},
            source_binding=copy.deepcopy(dict(self.source_binding)),
            no_reason=True, mis_jiang_response_output=False,
            teacher={"weight": self.teacher_weight, "selection": "feasibility_incumbent",
                     "normalization": "optimizer_window", "max_passes": 2,
                     "max_tokens": 3000, "uses_reference_labels": False,
                     "extra_training_compute": self.teacher_weight > 0},
            continuation={"fixed_scheduler_horizon": 400,
                          "stop_at_global_step": os.environ.get("GRPO_STOP_AT_GLOBAL_STEP"),
                          "strict_optimizer_resume": True,
                          "kl_reference": "original_SFT_not_resume_checkpoint"},
        )
        return receipt


def build_runtime(source_binding, teacher_weight, diagnostic=False):
    trainer = source_binding["required_files"]["src/clasp/entrypoints/train_grpo.py"]
    semantics = replace(
        FOARL_F638_SEMANTICS,
        method_id="foarl_teacher" if teacher_weight else "foarl_control",
        profile_id=PROFILE_ID,
        publication_label="FOARL + verified teacher" if teacher_weight else "FOARL matched control",
        supported_tasks=("cvrp", "mis"), num_generations=9,
        prompt_sha256=COMMON_PROMPT_SHA256,
        trainer_source_version=PROFILE_ID, trainer_source_sha256=trainer["sha256"],
    )
    return FoarlTeacherDevRuntime(semantics=semantics,
                                 source_binding=copy.deepcopy(dict(source_binding)),
                                 teacher_weight=teacher_weight, diagnostic=diagnostic)
