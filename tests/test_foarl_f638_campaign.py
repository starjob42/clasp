from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clasp.baselines.foarl_f638_campaign import (
    FOARL_F638_CAMPAIGN_METHODS,
    FOARL_F638_CAMPAIGN_PROFILES,
    FOARL_F638_MATCHED_B9,
    FOARL_F638_NATIVE,
    get_foarl_f638_campaign_profile,
    validate_foarl_f638_campaign_contract,
)
from clasp.entrypoints.train_foarl_f638_campaign import (
    build_foarl_f638_campaign_parser,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_sota_advancement as advancement  # noqa: E402
import preflight_foarl_f638_campaign as preflight  # noqa: E402
import prepare_sota_run as prepare  # noqa: E402


def _policy_payload(method: str, task: str, seed: int) -> dict:
    profile = get_foarl_f638_campaign_profile(method)
    save_steps = 25 if method == FOARL_F638_NATIVE and task == "op" else (
        50 if method == FOARL_F638_NATIVE else 400
    )
    eval_steps = save_steps if method == FOARL_F638_NATIVE else 100
    payload = {
        "schema_version": advancement.POLICY_SCHEMA_VERSION,
        "campaign_id": "foarl-campaign",
        "stage_id": f"{method}-{task}",
        "stage_index": 0,
        "prior_stage_receipt_chain": [],
        "status": "frozen_before_results",
        "frozen_at_utc": "2026-08-26T00:00:00Z",
        "target_step": profile.final_step(task),
        "expected_tasks": [task],
        "expected_methods": [method],
        "train_seeds": [seed],
        "run_path_template": "runs/{task}/{method}/seed{seed}",
        "model_initialization_registry_sha256": "a" * 64,
        "contract_requirements": {
            "world_size": profile.world_size,
            "sft_seed": 42,
            "num_generations": profile.semantics.num_generations,
            "logging_steps": 1,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
        },
        "health_gates": {
            key: True for key in sorted(advancement.HEALTH_GATE_KEYS)
        },
        "resume_gate": {
            "definition": advancement.RESUME_DEFINITION,
            **{key: True for key in sorted(advancement.RESUME_GATE_KEYS)},
        },
        "development_performance_gate": None,
    }
    payload["policy_sha256"] = hashlib.sha256(
        advancement._canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _write_policy(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_two_lane_profiles_are_immutable_and_scientifically_distinct() -> None:
    assert FOARL_F638_CAMPAIGN_METHODS == (
        "foarl_f638_native",
        "foarl_f638_matched_b9",
    )
    native = FOARL_F638_CAMPAIGN_PROFILES[FOARL_F638_NATIVE]
    matched = FOARL_F638_CAMPAIGN_PROFILES[FOARL_F638_MATCHED_B9]
    assert (native.semantics.num_generations, native.world_size) == (8, 1)
    assert (matched.semantics.num_generations, matched.world_size) == (9, 4)
    assert native.generation_contract()["max_prompt_length"] == 20_000
    assert matched.generation_contract()["max_completion_length"] == 1_000
    assert matched.optimizer == "adamw_8bit"
    assert matched.training_contract(task="tsp", train_seed=17)["reproduction_claim"] == (
        "not_an_official_reproduction"
    )
    assert native.final_step("cvrp") == 250
    assert native.final_step("op") == 625
    with pytest.raises(TypeError):
        FOARL_F638_CAMPAIGN_PROFILES["new"] = native  # type: ignore[index]
    with pytest.raises(TypeError):
        native.native_final_step_by_task["tsp"] = 999  # type: ignore[index]


@pytest.mark.parametrize(
    ("method", "task", "seed"),
    [
        (FOARL_F638_NATIVE, "tsp", 42),
        (FOARL_F638_MATCHED_B9, "tsp", 17),
    ],
)
def test_static_lane_contract_accepts_only_exact_profile(method: str, task: str, seed: int) -> None:
    profile = get_foarl_f638_campaign_profile(method)
    config = profile.training_contract(task=task, train_seed=seed)
    generation = profile.generation_contract()
    environment = profile.tracked_environment_contract(task=task, train_seed=seed)
    assert validate_foarl_f638_campaign_contract(
        method=method,
        task=task,
        train_seed=seed,
        final_step=profile.final_step(task),
        world_size=profile.world_size,
        config=config,
        generation_config=generation,
        environment=environment,
    ) is profile
    generation["max_completion_length"] = 3_000
    with pytest.raises(ValueError, match="immutable FOARL lane"):
        validate_foarl_f638_campaign_contract(
            method=method,
            task=task,
            train_seed=seed,
            final_step=profile.final_step(task),
            world_size=profile.world_size,
            config=config,
            generation_config=generation,
            environment=environment,
        )


def test_campaign_parser_resolves_native_and_matched_defaults_without_aliasing() -> None:
    native = build_foarl_f638_campaign_parser(FOARL_F638_NATIVE).parse_args(
        [
            "--foarl_campaign_lane",
            FOARL_F638_NATIVE,
            "--sft_checkpoint_path",
            "/frozen/tsp-sft",
            "--tasks",
            "tsp",
        ]
    )
    matched = build_foarl_f638_campaign_parser(FOARL_F638_MATCHED_B9).parse_args(
        [
            "--foarl_campaign_lane",
            FOARL_F638_MATCHED_B9,
            "--sft_checkpoint_path",
            "/frozen/tsp-sft",
            "--tasks",
            "tsp",
        ]
    )
    assert (native.num_generations, native.generation_batch_size) == (8, 8)
    assert (native.per_device_train_batch_size, native.gradient_accumulation_steps) == (8, 8)
    assert (matched.num_generations, matched.generation_batch_size) == (9, 36)
    assert (matched.per_device_train_batch_size, matched.gradient_accumulation_steps) == (3, 3)
    assert matched.optim == "adamw_8bit"
    assert matched.weight_decay == 0.01
    assert matched.warmup_ratio == 0.03
    assert matched.eval_from_train_split is True
    matched_profile = get_foarl_f638_campaign_profile(FOARL_F638_MATCHED_B9)
    matched_profile.runtime.validate_args(
        SimpleNamespace(
            tasks=["tsp"],
            use_primal_dual=False,
            use_clasp=False,
            full_finetune=False,
            verify_gt_rewards=False,
            num_generations=9,
            prompt_style="jiang_f638",
            no_reason=False,
            grpo_beta=0.05,
            grpo_epsilon=0.2,
            grpo_epsilon_high=0.2,
            grpo_loss_type="grpo",
            foarl_advantage_epsilon=0.0001,
            foarl_scale_rewards="group",
            foarl_num_iterations=1,
            foarl_token_aggregation="completion_token_mean_then_sequence_mean",
            foarl_sync_ref_model=False,
        )
    )
    assert matched_profile.runtime.metadata()["method_id"] == FOARL_F638_MATCHED_B9


@pytest.mark.parametrize(
    ("method", "task", "seed"),
    [
        (FOARL_F638_NATIVE, "op", 42),
        (FOARL_F638_MATCHED_B9, "tsp", 17),
    ],
)
def test_advancement_policy_binds_exactly_one_task_and_one_lane(
    tmp_path: Path, method: str, task: str, seed: int
) -> None:
    policy = _write_policy(tmp_path / "policy.json", _policy_payload(method, task, seed))
    assert advancement._validate_policy(policy)["expected_methods"] == [method]
    assert advancement.method_state_requirements(method) == (False, False)
    assert prepare.method_state_requirements(method) == (False, False)
    assert method in prepare.FOARL_PUBLICATION_METHODS
    assert method in advancement.FOARL_PUBLICATION_METHODS


def test_advancement_rejects_mixed_task_or_mixed_lane_foarl_policy(tmp_path: Path) -> None:
    payload = _policy_payload(FOARL_F638_MATCHED_B9, "tsp", 17)
    payload["expected_tasks"] = ["tsp", "cvrp"]
    payload["policy_sha256"] = hashlib.sha256(
        advancement._canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "policy_sha256"}
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="exactly one task"):
        advancement._validate_policy(_write_policy(tmp_path / "multi-task.json", payload))

    payload = _policy_payload(FOARL_F638_MATCHED_B9, "tsp", 17)
    payload["expected_methods"] = [FOARL_F638_MATCHED_B9, FOARL_F638_NATIVE]
    payload["policy_sha256"] = hashlib.sha256(
        advancement._canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "policy_sha256"}
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="exactly one lane"):
        advancement._validate_policy(_write_policy(tmp_path / "multi-lane.json", payload))


def test_read_only_preflight_binds_source_registry_policy_and_has_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    for name in ("prepare_sota_run.py", "audit_sota_advancement.py"):
        (source / "scripts" / name).write_text("# fixture\n", encoding="utf-8")
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("fixture\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    forbidden_path = tmp_path / "forbidden.json"
    forbidden_path.write_text("{}\n", encoding="utf-8")
    policy_path = _write_policy(
        tmp_path / "policy.json",
        _policy_payload(FOARL_F638_MATCHED_B9, "tsp", 17),
    )
    registry_path.chmod(0o440)
    forbidden_path.chmod(0o440)
    policy_path.chmod(0o440)
    forbidden_identity = {
        "registry_sha256": "b" * 64,
        "source_tree_manifest_sha256": "c" * 64,
    }
    registry_identity = {
        "registry_sha256": "a" * 64,
        "forbidden_source_registry_sha256": forbidden_identity["registry_sha256"],
        "forbidden_source_tree_manifest_sha256": forbidden_identity[
            "source_tree_manifest_sha256"
        ],
    }
    fake_prepare = SimpleNamespace(
        TASKS={"tsp", "cvrp", "op", "mis", "jssp"},
        validate_source_snapshot=lambda _root, _manifest: None,
        validate_base_sft_source_registry=lambda _payload, expected_tasks: forbidden_identity,
        validate_model_initialization_registry=lambda _payload: registry_identity,
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    fake_advancement = SimpleNamespace(
        _validate_policy=lambda _path: policy,
        _prior_stage_authorization_audit=lambda _policy, _paths: {"errors": []},
    )
    monkeypatch.setattr(
        preflight,
        "_load_script",
        lambda name, _path: fake_prepare if "prepare" in name else fake_advancement,
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    receipt = preflight.static_preflight(
        method=FOARL_F638_MATCHED_B9,
        task="tsp",
        train_seed=17,
        final_step=3_200,
        world_size=4,
        source_root=source,
        source_manifest=manifest,
        forbidden_source_registry=forbidden_path,
        model_initialization_registry=registry_path,
        advancement_policy=policy_path,
        campaign_id="foarl-campaign",
    )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert receipt["status"] == "PASS"
    assert receipt["side_effects"] == {"writes": 0, "scheduler_actions": 0}
    assert receipt["reproduction_claim"] == "not_an_official_reproduction"
    assert before == after


def test_launcher_is_dedicated_fail_closed_and_never_self_submits() -> None:
    text = (SCRIPTS / "train_foarl_f638_a100.slurm").read_text(encoding="utf-8")
    assert "foarl_f638_native|foarl_f638_matched_b9" in text
    assert "preflight_foarl_f638_campaign.py" in text
    assert "--nproc_per_node=\"${WORLD_SIZE}\"" in text
    assert "--sft_checkpoint_path \"${SFT_MODEL_PATH}\"" in text
    assert "KL_REFERENCE_RECEIPT" in text and "checkpoint_fidelity.json" in text
    assert "GRPO_STRICT_OPTIMIZER_RESUME=1" in text
    assert "CLASP_TRAINING_RUN_CONTRACT_PATH" in text
    assert "WANDB_RUN_NAME_OVERRIDE=\"${CAMPAIGN_ID}-${TASK}-${METHOD}-seed${TRAIN_SEED}\"" in text
    assert "reproduction_claim=not_an_official_reproduction" in text
    assert "max_prompt_length=20000" in text and "max_completion_length=1000" in text
    assert "--no-full_finetune" not in text
    executable_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("sbatch ") for line in executable_lines)
    assert any(line.startswith("srun ") for line in executable_lines)
    common_block = text.split("COMMON_ARGS=(", 1)[1].split("\n)", 1)[0]
    launcher_options = set(
        re.findall(r"(?<![A-Za-z0-9_])(--[A-Za-z0-9_-]+)", common_block)
    )
    parser_options = {
        option
        for action in build_foarl_f638_campaign_parser(
            FOARL_F638_MATCHED_B9
        )._actions
        for option in action.option_strings
    }
    assert launcher_options <= parser_options


def test_campaign_console_entrypoint_is_installed() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'clasp-foarl-f638-campaign = "clasp.entrypoints.train_foarl_f638_campaign:main"'
        in text
    )
