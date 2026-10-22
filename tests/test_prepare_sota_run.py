from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clasp.data.benchmark import forbidden_source_manifest_identity, sha256_json
from clasp.models import publication_fidelity

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "prepare_sota_run", ROOT / "scripts" / "prepare_sota_run.py"
)
assert SPEC is not None and SPEC.loader is not None
prepare_sota_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_sota_run
SPEC.loader.exec_module(prepare_sota_run)
FREEZER_SPEC = importlib.util.spec_from_file_location(
    "freeze_model_initialization_registry",
    ROOT / "scripts" / "freeze_model_initialization_registry.py",
)
assert FREEZER_SPEC is not None and FREEZER_SPEC.loader is not None
freeze_model_initialization_registry = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = freeze_model_initialization_registry
FREEZER_SPEC.loader.exec_module(freeze_model_initialization_registry)


def _checkpoint_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": publication_fidelity.IDENTITY_SCHEMA_VERSION,
        "protocol": publication_fidelity.IDENTITY_PROTOCOL,
        "status": publication_fidelity.IDENTITY_STATUS,
        "receipt_sha256": "1" * 64,
        "receipt_file_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "adapter_weights_sha256": "4" * 64,
        "trainable_tensor_inventory_sha256": "5" * 64,
        "publication_invariant_sha256": "6" * 64,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return identity


def _reload_identity(path: Path) -> dict[str, object]:
    receipt_path = path / "sft_reload_equivalence.json" if path.is_dir() else path
    checkpoint = _checkpoint_identity()
    identity: dict[str, object] = {
        "schema_version": 2,
        "protocol": "clasp-sft-post-save-reload-equivalence-v2",
        "status": "exact_post_save_reload_verified",
        "receipt_sha256": "7" * 64,
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "checkpoint_fidelity_identity_sha256": checkpoint["identity_sha256"],
        "serialized_trainable_closure_sha256": "8" * 64,
        "runtime_pair_sha256": "9" * 64,
        "functional_equivalence_sha256": "a" * 64,
        "producer_sha256": "b" * 64,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return identity


def _fixed_kl_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": publication_fidelity.FIXED_KL_IDENTITY_SCHEMA_VERSION,
        "protocol": publication_fidelity.FIXED_KL_IDENTITY_PROTOCOL,
        "status": publication_fidelity.FIXED_KL_IDENTITY_STATUS,
        "runtime_schema_version": publication_fidelity.exact_peft_kl.SCHEMA_VERSION,
        "runtime_protocol": publication_fidelity.exact_peft_kl.PUBLICATION_PROTOCOL,
        "runtime_status": publication_fidelity.exact_peft_kl.PUBLICATION_STATUS,
        "runtime_receipt_filename": publication_fidelity.exact_peft_kl.RECEIPT_FILENAME,
        "verifier_protocol": publication_fidelity.exact_peft_kl.VERIFIER_PROTOCOL,
        "post_step_schema_version": publication_fidelity.exact_peft_kl_post_step.SCHEMA_VERSION,
        "post_step_protocol": publication_fidelity.exact_peft_kl_post_step.PUBLICATION_PROTOCOL,
        "post_step_status": publication_fidelity.exact_peft_kl_post_step.PUBLICATION_STATUS,
        "post_step_step1_receipt_filename": (
            publication_fidelity.exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME
        ),
        "post_step_receipt_filename": (
            publication_fidelity.exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME
        ),
        "post_step_verifier_protocol": (
            publication_fidelity.exact_peft_kl_post_step.VERIFIER_PROTOCOL
        ),
        "post_step_verifier_implementation_sha256": (
            publication_fidelity._post_step_verifier_implementation_sha256()
        ),
        "parameterization": "continue_same_bias_all_sft_adapter",
        "saved_adapter_tensor_inventory_sha256": "e" * 64,
        "saved_adapter_dtypes": ["F32"],
        "saved_lora_dtype": "F32",
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
        "source_checkpoint_fidelity": _checkpoint_identity(),
    }
    for index, field in enumerate(
        publication_fidelity.FIXED_KL_IDENTITY_KEYS - set(identity) - {"identity_sha256"},
        start=7,
    ):
        identity[field] = f"{index % 16:x}" * 64
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return identity


@pytest.fixture(autouse=True)
def _stub_large_qwen_persistence_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact sparse-artifact parser has focused tests in its own module."""

    monkeypatch.setattr(prepare_sota_run, "audit_qwen25_base_bias_scope", lambda _path: {})
    monkeypatch.setattr(prepare_sota_run, "audit_adapter_persistence", lambda _path: {})
    monkeypatch.setattr(
        prepare_sota_run,
        "audit_jiang_f638_base_snapshot",
        lambda path: prepare_sota_run._tree_fingerprint(
            Path(path), label="fixture Jiang base"
        ),
    )
    monkeypatch.setattr(
        prepare_sota_run,
        "validate_jiang_f638_base_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        prepare_sota_run,
        "validate_audit_receipt",
        lambda _path, **_kwargs: {},
    )
    identity = _checkpoint_identity()
    monkeypatch.setattr(
        prepare_sota_run,
        "publication_fidelity_identity",
        lambda _path: dict(identity),
    )
    monkeypatch.setattr(
        prepare_sota_run,
        "validate_reload_receipt",
        lambda path, **_kwargs: (
            {}
            if Path(path).is_file()
            else (_ for _ in ()).throw(ValueError(f"missing reload receipt: {path}"))
        ),
    )
    monkeypatch.setattr(
        prepare_sota_run,
        "reload_equivalence_identity",
        lambda path, **_kwargs: _reload_identity(Path(path)),
    )


def _base_model(root: Path) -> Path:
    path = root / "base"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(prepare_sota_run.QWEN25_7B_BASE_SIGNATURE), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"approved-qwen25-7b-base")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return path


def _sft_model(root: Path, *, task: str, base: Path, sft_data_root: Path) -> Path:
    path = root / f"sft-{task}"
    path.mkdir()
    adapter = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "all",
        "target_modules": sorted(prepare_sota_run.JIANG_LORA_MODULES),
        "base_model_name_or_path": str(base.resolve()),
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(f"weights:{task}".encode())
    (path / "adapter_persistence_receipt.json").write_text("{}\n", encoding="utf-8")
    (path / "checkpoint_fidelity.json").write_text("{}\n", encoding="utf-8")
    (path / "sft_reload_equivalence.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "hyperparams.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "tasks": [task],
                "args": {"data_root": str(sft_data_root.resolve())},
                "dataset": {
                    "train_size": 500_000,
                    "eval_size": 0,
                    "train_task_counts": {task: 500_000},
                },
                "stage_hyperparameters": prepare_sota_run.MATCHED_SFT_STAGE_HPARAMETERS,
                "distributed": {"world_size": 1},
            }
        ),
        encoding="utf-8",
    )
    (path / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "sft",
                "completed": True,
                "global_step": 31_250,
                "max_steps": 31_250,
                "has_saved_model": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _source(root: Path) -> Path:
    path = root / "source"
    (path / "src" / "clasp").mkdir(parents=True)
    (path / "scripts").mkdir()
    (path / "src" / "clasp" / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "scripts" / "prepare_sota_run.py").write_text("# helper\n", encoding="utf-8")
    (path / "scripts" / "train_clasp_sota_a100.slurm").write_text("#!/bin/bash\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (path / "logs").mkdir()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.parent != path / "logs"
    )
    manifest_lines = []
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        manifest_lines.append(f"{hashlib.sha256(candidate.read_bytes()).hexdigest()}  {relative}\n")
        candidate.chmod(0o440)
    manifest = path / "SOURCE_MANIFEST.sha256"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")
    manifest.chmod(0o440)
    for directory in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        directory.chmod(0o770 if directory == path / "logs" else 0o550)
    path.chmod(0o550)
    return path


def _registry(root: Path, sources_by_task: dict[str, dict[str, Path]]) -> Path:
    records = []
    for task, sources in sorted(sources_by_task.items()):
        for label, source_root in sorted(sources.items()):
            files = []
            for path in sorted(source_root.rglob("*.json")):
                rows = json.loads(path.read_text(encoding="utf-8"))
                files.append(
                    {
                        "relative_path": path.relative_to(source_root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                        "rows": len(rows),
                    }
                )
            record = {
                "task": task,
                "label": label,
                "registered_source_locator": str(source_root.resolve()),
                "files": files,
                "file_manifest_sha256": sha256_json(files),
                "rows": sum(item["rows"] for item in files),
                "unique_instance_hashes": sum(item["rows"] for item in files),
            }
            record["source_identity_sha256"] = forbidden_source_manifest_identity(record)[
                "source_identity_sha256"
            ]
            records.append(record)
    payload = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": sorted(sources_by_task),
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": records,
        "source_tree_manifest_sha256": sha256_json(
            [forbidden_source_manifest_identity(record) for record in records]
        ),
    }
    payload["registry_sha256"] = sha256_json(payload)
    path = root / "forbidden_registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path: Path, *, method: str = "clasp_dense") -> argparse.Namespace:
    base = _base_model(tmp_path)
    data_root = tmp_path / "data_rl"
    sft_root = tmp_path / "data_sft"
    sources_by_task: dict[str, dict[str, Path]] = {}
    specialists: dict[str, dict[str, object]] = {}
    models: dict[str, Path] = {}
    for index, task in enumerate(sorted(prepare_sota_run.TASKS)):
        data = data_root / task / "train"
        data.mkdir(parents=True)
        (data / "train.json").write_text(
            json.dumps([{"id": index * 10 + 1}]) + "\n", encoding="utf-8"
        )
        historical = data_root / task / "eval"
        historical.mkdir(parents=True)
        (historical / "test.json").write_text(
            json.dumps([{"id": index * 10 + 2}]) + "\n", encoding="utf-8"
        )
        sft_data = sft_root / task / "train"
        sft_data.mkdir(parents=True)
        (sft_data / "train.json").write_text(
            json.dumps([{"id": index * 10 + 3}]) + "\n", encoding="utf-8"
        )
        models[task] = _sft_model(
            tmp_path,
            task=task,
            base=base,
            sft_data_root=sft_root,
        )
        sources_by_task[task] = {
            "historical-eval": historical,
            "rl-train": data,
            "sft-train": sft_data,
        }
        specialists[task] = {
            "model_path": models[task],
            "sft_data_path": sft_data,
            "sft_seed": 42,
        }
    registry = _registry(tmp_path, sources_by_task)
    initialization_payload = prepare_sota_run.build_model_initialization_registry(
        base_model_path=base,
        specialists=specialists,
        forbidden_source_registry=registry,
        frozen_at_utc="2026-08-14T00:00:00Z",
    )
    initialization_registry = tmp_path / "model_initialization_registry.json"
    initialization_registry.write_text(json.dumps(initialization_payload), encoding="utf-8")
    model = models["cvrp"]
    data = data_root / "cvrp" / "train"
    sft_data = sft_root / "cvrp" / "train"
    source = _source(tmp_path)
    return argparse.Namespace(
        run_dir=tmp_path / "run",
        method=method,
        task="cvrp",
        train_seed=17,
        sft_seed=42,
        final_step=3200,
        world_size=4,
        model_path=model,
        base_model_path=base,
        data_path=data,
        sft_data_path=sft_data,
        forbidden_source_registry=registry,
        model_initialization_registry=initialization_registry,
        source_root=source,
        source_manifest=source / "SOURCE_MANIFEST.sha256",
        fixed_kl_reference_model_path=None,
        fixed_kl_reference_receipt_path=None,
        config=[
            "learning_rate=1e-6",
            "no_reason=true",
            "beta=0.0",
            "checkpoint_fidelity=publication_exact_required",
        ],
        generation_config=["num_generations=8", "temperature=0.7"],
        allowed_env=[
            "CLASP_BON_ALIGNED",
            "CLASP_PUBLICATION_FIDELITY",
            "GRPO_DATA_SEED",
        ],
    )


def _trainer_state(step: int, final_step: int = 3200) -> str:
    return json.dumps({"global_step": step, "max_steps": final_step})


def _publication_environ(*, aligned: str = "1") -> dict[str, str]:
    return {
        "CLASP_BON_ALIGNED": aligned,
        "CLASP_PUBLICATION_FIDELITY": "1",
        "GRPO_DATA_SEED": "17",
    }


def _canonical_arm_contract_fields(method: str) -> tuple[dict, dict, dict[str, str]]:
    canonical = method.startswith("clasp_canonical")
    fixed_null = method == "set_null"
    exact_set_credit = canonical or fixed_null
    foarl_comparator = method == "clasp_canonical_foarl_matched_b9"
    recovery = 1.0 if method in {
        "clasp_canonical",
        "clasp_canonical_fixed_lambda",
        "clasp_canonical_foarl_matched_b9",
        "set_null",
    } else 0.0
    component_lr = 0.05 if method in {"dual_scalar_z", "dual_separate_z"} else 0.0
    bon_lr = (
        0.0
        if method
        in {
            "grpo_z",
            "dual_scalar_z",
            "dual_separate_z",
            "clasp_canonical_fixed_lambda",
        }
        else 0.05
    )
    fixed_lambda = 0.7 if method == "clasp_canonical_fixed_lambda" else 0.0
    policy_formula = (
        "group_z_verified_quality_v1"
        if method == "grpo_z"
        else "group_z_scalarized_positive_component_penalty_v1"
        if method == "dual_scalar_z"
        else "group_z_reward_minus_group_z_positive_component_penalty_v1"
        if method == "dual_separate_z"
        else "fixed_null_k8_same_set_utility_v1"
        if fixed_null
        else "canonical_k8_leave_out_set_utility_v1"
    )
    component_controller = (
        "legacy_projected_ema_component_gap"
        if method in {"dual_scalar_z", "dual_separate_z"}
        else "frozen_zero"
    )
    set_controller = (
        "frozen"
        if method == "clasp_canonical_fixed_lambda"
        else "signed_failure_at_k_ema"
        if exact_set_credit
        else "off"
    )
    generation = {
        "num_generations": 9,
        "generation_batch_size": 36,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_prompt_length": 20_000 if foarl_comparator else 29_768,
        "max_completion_length": 1_000 if foarl_comparator else 3_000,
    }
    config = {
        "entrypoint": "clasp.entrypoints.train_grpo",
        "arm_contract_schema": (
            "clasp-canonical-foarl-matched-b9-20k1k-a100-v1"
            if foarl_comparator
            else "clasp-canonical-matched-a100-v1"
        ),
        "policy_surrogate": "clipped_token_level_practical",
        "policy_credit_formula": policy_formula,
        "component_controller": component_controller,
        "set_controller": set_controller,
        "deployment_budget_k": 8,
        "verifier_gate": "strict_recomputed_feasible_only_quality",
        "intentional_method_diff": (
            "policy_credit_formula,component_controller,set_controller,use_clasp,"
            "clasp_canonical_set,clasp_fixed_null_set,pd_group_scalarization,pd_separate_z,"
            "pd_dual_lr,pd_dual_lr_up,"
            "pd_dual_lr_down,clasp_bon_dual_lr,clasp_bon_lambda_init,"
            "clasp_canonical_recovery_scale"
        ),
        "loss_type": "grpo",
        "dtype": "bfloat16",
        "per_device_train_batch_size": 3,
        "per_device_eval_batch_size": 9,
        "gradient_accumulation_steps": 3,
        "use_primal_dual": True,
        "use_clasp": exact_set_credit,
        "clasp_canonical_set": canonical,
        "clasp_fixed_null_set": fixed_null,
        "pd_group_scalarization": method in {"grpo_z", "dual_scalar_z"},
        "pd_separate_z": method == "dual_separate_z",
        "pd_dual_lr": component_lr,
        "pd_dual_lr_up": component_lr,
        "pd_dual_lr_down": component_lr,
        "pd_lambda_init": 0.0,
        "pd_cost_limit": 0.0,
        "pd_dual_update_interval": 1,
        "pd_violation_ema_beta": 0.1,
        "pd_positive_constraint_penalty": True,
        "pd_objective_weight": 1.0,
        "pd_structure_weight": 0.0,
        "pd_soft_objective_for_infeasible": False,
        "pd_component_profile": "co_feasibility_lock_all",
        "clasp_deployment_set_size": 8,
        "clasp_canonical_recovery_scale": recovery,
        "clasp_canonical_recovery_temperature": 0.15,
        "clasp_canonical_max_subsets": 9,
        "clasp_component_dual_recovery_scale": 0.0,
        "clasp_bon_dual_lr": bon_lr,
        "clasp_bon_lambda_init": fixed_lambda,
        "clasp_bon_target_failure_rate": 0.005,
        "clasp_feasibility_warmup_steps": 0,
        "clasp_self_replay": False,
        "clasp_proper_dual": False,
        "clasp_adaptive_length": False,
        "importance_sampling_level": "token",
    }
    if foarl_comparator:
        config.update(
            comparison_protocol="foarl-f638-clasp-shared-compute-data-token-design-v1",
            comparison_baseline="foarl_f638_matched_b9",
            attribution="common_compute_data_token_schedule_only",
            reproduction_claim="not_an_official_jiang_reproduction",
            beta=0.05,
        )
    environment = {
        "CLASP_CANONICAL_SET": "1" if canonical else "0",
        "CLASP_FIXED_NULL_SET": "1" if fixed_null else "0",
        "CLASP_BON_ALIGNED": "1" if exact_set_credit else "0",
        "CLASP_FRONTIER_CDF": "0",
        "CLASP_BON_TARGET_FAILURE_RATE": "0.005",
        "CLASP_BON_TARGET_FAILURE_RATES": "cvrp:0.005",
        "CLASP_FEASIBILITY_GATED": "1" if exact_set_credit else "0",
        "CLASP_CANONICAL_RECOVERY_SCALE": str(recovery),
        "CLASP_BON_DUAL_LR": str(bon_lr),
        "CLASP_BON_LAMBDA_INIT": str(fixed_lambda),
        "CLASP_BON_RECOVERY_SCALE": "0.0",
        "CLASP_COMPONENT_DUAL_RECOVERY_SCALE": "0.0",
        "CLASP_FEASIBILITY_WARMUP_STEPS": "0",
        "CLASP_SELF_REPLAY_ENABLED": "0",
        "CLASP_VALIDATION_GATE_CHECKPOINTS": "0",
        "CLASP_PAIRWISE_LOSS_WEIGHT": "0.0",
        "CLASP_ELITE_IMITATION_LOSS_WEIGHT": "0.0",
        "CLASP_MULTI_ELITE_IMITATION_LOSS_WEIGHT": "0.0",
        "CLASP_REFERENCE_IMITATION_LOSS_WEIGHT": "0.0",
        "CLASP_REFERENCE_CONTRASTIVE_LOSS_WEIGHT": "0.0",
        "CLASP_HARD_GROUP_WEIGHT_SCALE": "0.0",
        "PD_IMPORTANCE_SAMPLING_LEVEL": "token",
        "PD_ALLOW_TOKEN_IMPORTANCE_ABLATION": "1",
        "GRPO_DATA_SEED": "17",
    }
    if fixed_null:
        environment["CLASP_FIXED_NULL_RECEIPT_PATH"] = "/frozen/set-null.json"
    return config, generation, environment


def _pd_state(*, with_bon: bool) -> str:
    payload = {
        "cvrp": {
            "cost_names": ["capacity"],
            "cost_limits": [0.0],
            "cost_weights": [1.0],
            "dual_lambdas": [0.3],
            "controller_state": {},
        }
    }
    if with_bon:
        payload["__clasp_bon__"] = {
            "lambdas": {"cvrp": 1.0},
            "failure_ema": {"cvrp": 0.2},
            "observed_groups": {"cvrp": 12},
        }
    return json.dumps(payload)


@pytest.mark.parametrize("method", sorted(prepare_sota_run.CANONICAL_PUBLICATION_METHODS))
def test_canonical_publication_arm_profiles_are_accepted(method: str) -> None:
    config, generation, environment = _canonical_arm_contract_fields(method)
    prepare_sota_run.validate_canonical_publication_arm(
        method=method,
        task="cvrp",
        train_seed=17,
        config=config,
        generation_config=generation,
        environment=environment,
    )


def test_set_null_profile_is_formula_distinct_and_requires_receipt_environment() -> None:
    config, generation, environment = _canonical_arm_contract_fields("set_null")
    assert config["policy_credit_formula"] == "fixed_null_k8_same_set_utility_v1"
    assert config["clasp_fixed_null_set"] is True
    assert config["clasp_canonical_set"] is False
    prepare_sota_run.validate_canonical_publication_arm(
        method="set_null",
        task="cvrp",
        train_seed=17,
        config=config,
        generation_config=generation,
        environment=environment,
    )
    environment.pop("CLASP_FIXED_NULL_RECEIPT_PATH")
    with pytest.raises(ValueError, match="requires CLASP_FIXED_NULL_RECEIPT_PATH"):
        prepare_sota_run.validate_canonical_publication_arm(
            method="set_null",
            task="cvrp",
            train_seed=17,
            config=config,
            generation_config=generation,
            environment=environment,
        )


def test_fixed_null_receipt_identity_is_part_of_immutable_resume_contract(
    tmp_path: Path,
) -> None:
    first = {
        "method": "set_null",
        "fixed_null_receipt": {
            "receipt_sha256": "a" * 64,
            "null_candidate_sha256": "b" * 64,
            "evaluator_identity_sha256": "c" * 64,
        },
    }
    prepare_sota_run.enforce_contract(tmp_path, first)
    prepare_sota_run.enforce_contract(tmp_path, dict(first))
    drifted = json.loads(json.dumps(first))
    drifted["fixed_null_receipt"]["receipt_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="immutable run contract mismatch"):
        prepare_sota_run.enforce_contract(tmp_path, drifted)


def test_canonical_publication_contract_rejects_common_surrogate_drift() -> None:
    config, generation, environment = _canonical_arm_contract_fields("clasp_canonical")
    environment["PD_IMPORTANCE_SAMPLING_LEVEL"] = "sequence"
    with pytest.raises(ValueError, match="tracked environment differs"):
        prepare_sota_run.validate_canonical_publication_arm(
            method="clasp_canonical",
            task="cvrp",
            train_seed=17,
            config=config,
            generation_config=generation,
            environment=environment,
        )


@pytest.mark.parametrize(
    ("method", "field", "drift"),
    [
        ("grpo_z", "policy_credit_formula", "group_z_unverified_reward"),
        ("dual_scalar_z", "component_controller", "signed_component_residual"),
        ("dual_separate_z", "policy_credit_formula", "group_z_scalarized_positive_component_penalty_v1"),
        ("clasp_canonical", "set_controller", "off"),
        ("clasp_canonical_fixed_lambda", "clasp_bon_dual_lr", 0.05),
        ("clasp_canonical_no_recovery", "clasp_canonical_recovery_scale", 1.0),
        ("grpo_z", "pd_positive_constraint_penalty", False),
    ],
)
def test_canonical_publication_contract_rejects_formula_or_controller_drift(
    method: str,
    field: str,
    drift: object,
) -> None:
    config, generation, environment = _canonical_arm_contract_fields(method)
    config[field] = drift
    with pytest.raises(ValueError, match="differs from canonical matched contract"):
        prepare_sota_run.validate_canonical_publication_arm(
            method=method,
            task="cvrp",
            train_seed=17,
            config=config,
            generation_config=generation,
            environment=environment,
        )


def test_no_recovery_keeps_the_active_set_controller_and_changes_only_rho() -> None:
    main, _, main_env = _canonical_arm_contract_fields("clasp_canonical")
    no_recovery, _, no_recovery_env = _canonical_arm_contract_fields(
        "clasp_canonical_no_recovery"
    )
    assert main["set_controller"] == no_recovery["set_controller"] == "signed_failure_at_k_ema"
    assert main["clasp_bon_dual_lr"] == no_recovery["clasp_bon_dual_lr"] == 0.05
    changed_config = {key for key in main if main[key] != no_recovery[key]}
    changed_environment = {key for key in main_env if main_env[key] != no_recovery_env[key]}
    assert changed_config == {"clasp_canonical_recovery_scale"}
    assert changed_environment == {"CLASP_CANONICAL_RECOVERY_SCALE"}


def test_canonical_fixed_lambda_is_the_only_nonzero_frozen_lambda_arm() -> None:
    config, generation, environment = _canonical_arm_contract_fields("clasp_canonical")
    config["clasp_bon_lambda_init"] = 0.7
    environment["CLASP_BON_LAMBDA_INIT"] = "0.7"
    with pytest.raises(ValueError, match="set lambda differs"):
        prepare_sota_run.validate_canonical_publication_arm(
            method="clasp_canonical",
            task="cvrp",
            train_seed=17,
            config=config,
            generation_config=generation,
            environment=environment,
        )


def _checkpoint(run_dir: Path, step: int, *, method: str = "clasp_dense") -> Path:
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "checkpoint_fidelity.json",
        "optimizer.pt",
        "scheduler.pt",
    ):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(_trainer_state(step), encoding="utf-8")
    for rank in range(4):
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(bytes([rank]))
    if method != "grpo":
        (checkpoint / "multitask_primal_dual_state.json").write_text(
            _pd_state(with_bon=method.startswith("clasp_")), encoding="utf-8"
        )
    return checkpoint


def test_fresh_contract_is_atomic_immutable_and_idempotent(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    contract_path = args.run_dir / "run_contract.json"
    original = contract_path.read_bytes()
    assert contract_path.stat().st_mode & 0o777 == 0o440
    contract = json.loads(original)
    assert contract["data_lineage"]["rl_train"]["label"] == "rl-train"
    assert contract["data_lineage"]["sft_train"]["label"] == "sft-train"
    assert (
        contract["data_lineage"]["forbidden_source_registry"]["registry_sha256"]
        == json.loads(args.forbidden_source_registry.read_text())["registry_sha256"]
    )
    initialization = contract["model_initialization_registry"]
    assert initialization["protocol"] == "clasp-model-initialization-registry-v2"
    assert initialization["status"] == "frozen_before_training"
    assert initialization["model_family"] == "Qwen2.5-7B"
    assert set(initialization["specialist_sft_by_task"]) == prepare_sota_run.TASKS
    assert (
        initialization["base_model_tree_sha256"] == contract["fingerprints"]["base_model"]["sha256"]
    )
    assert (
        initialization["specialist_sft_by_task"]["cvrp"]["initial_model_tree_sha256"]
        == contract["fingerprints"]["initial_model"]["sha256"]
    )
    assert contract["fixed_kl_reference"] is None

    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    assert contract_path.read_bytes() == original


def test_publication_contract_requires_explicit_fidelity_mode(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environment = _publication_environ()
    environment.pop("CLASP_PUBLICATION_FIDELITY")
    with pytest.raises(ValueError, match="requires CLASP_PUBLICATION_FIDELITY=1"):
        prepare_sota_run.prepare(args, environ=environment)


def test_publication_contract_requires_fidelity_training_declaration(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.config = [item for item in args.config if not item.startswith("checkpoint_fidelity=")]
    with pytest.raises(ValueError, match="publication_exact_required"):
        prepare_sota_run.prepare(args, environ=_publication_environ())


def test_positive_beta_requires_and_binds_the_exact_sft_adapter_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.config = ["beta=0.05" if item == "beta=0.0" else item for item in args.config]
    args.fixed_kl_reference_model_path = args.model_path
    args.fixed_kl_reference_receipt_path = args.model_path / "checkpoint_fidelity.json"
    environment = _publication_environ()
    environment["CLASP_KL_REFERENCE_MODEL_PATH"] = str(args.fixed_kl_reference_model_path.resolve())
    environment["CLASP_KL_REFERENCE_RECEIPT"] = str(args.fixed_kl_reference_receipt_path.resolve())
    args.allowed_env.extend(["CLASP_KL_REFERENCE_MODEL_PATH", "CLASP_KL_REFERENCE_RECEIPT"])
    fixed_identity = _fixed_kl_identity()
    monkeypatch.setattr(
        prepare_sota_run,
        "fixed_kl_reference_identity",
        lambda _model, _receipt: fixed_identity,
    )

    assert prepare_sota_run.prepare(args, environ=environment) == "FRESH"
    contract = json.loads((args.run_dir / "run_contract.json").read_text(encoding="utf-8"))
    assert contract["fixed_kl_reference"] == fixed_identity
    assert (
        contract["fixed_kl_reference"]["source_checkpoint_fidelity"]
        == (
            contract["model_initialization_registry"]["specialist_sft_by_task"]["cvrp"][
                "checkpoint_fidelity"
            ]
        )
    )


def test_beta_zero_forbids_fixed_kl_inputs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.fixed_kl_reference_model_path = tmp_path / "unexpected"
    with pytest.raises(ValueError, match="beta=0.*forbid"):
        prepare_sota_run.prepare(args, environ=_publication_environ())


def test_model_initialization_registry_writer_is_create_once_and_read_only(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    payload = json.loads(args.model_initialization_registry.read_text(encoding="utf-8"))
    output = tmp_path / "frozen" / "model_initialization_registry.json"
    freeze_model_initialization_registry._write_create_once(output, payload)
    assert output.stat().st_mode & 0o777 == 0o440
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_model_initialization_registry._write_create_once(output, payload)


def test_contract_change_fails_closed_without_rewrite(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    contract_path = args.run_dir / "run_contract.json"
    original = contract_path.read_bytes()
    args.config = [
        "learning_rate=2e-6",
        "no_reason=true",
        "beta=0.0",
        "checkpoint_fidelity=publication_exact_required",
    ]

    with pytest.raises(ValueError, match="immutable run contract mismatch"):
        prepare_sota_run.prepare(args, environ=environ)
    assert contract_path.read_bytes() == original


def test_actual_rl_tree_must_equal_registered_rl_source(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.data_path / "train.json").write_text('[{"id": 999}]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from registry"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_actual_rl_tree_cannot_omit_registered_shard(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.data_path / "train.json").unlink()
    with pytest.raises(ValueError, match="contains no JSON files"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_initial_sft_recorded_root_must_equal_registered_source(tmp_path: Path) -> None:
    args = _args(tmp_path)
    payload = json.loads((args.model_path / "hyperparams.json").read_text(encoding="utf-8"))
    payload["args"]["data_root"] = str(tmp_path / "unregistered-sft")
    (args.model_path / "hyperparams.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not equal registered sft-train"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_nonmatched_sft_initialization_is_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    payload = json.loads((args.model_path / "hyperparams.json").read_text(encoding="utf-8"))
    payload["dataset"]["train_size"] = 100_000
    (args.model_path / "hyperparams.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not the matched 500k specialist"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_adapter_loader_base_must_equal_contracted_base(tmp_path: Path) -> None:
    args = _args(tmp_path)
    adapter = json.loads((args.model_path / "adapter_config.json").read_text(encoding="utf-8"))
    adapter["base_model_name_or_path"] = str(tmp_path / "different-base")
    (args.model_path / "adapter_config.json").write_text(
        json.dumps(adapter),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not equal contracted base model"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_unapproved_base_model_bytes_are_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.base_model_path / "model.safetensors").write_bytes(b"substituted-larger-model")
    with pytest.raises(ValueError, match="base model bytes differ from the approved"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_unapproved_specialist_sft_bytes_are_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.model_path / "adapter_model.safetensors").write_bytes(b"post-hoc-specialist-substitution")
    with pytest.raises(ValueError, match="specialist SFT bytes differ from the approved"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_instruction_tuned_base_cannot_masquerade_as_qwen_base(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.base_model_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for message in messages %}{{ message }}{% endfor %}"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="instruction/chat template"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_tampered_model_initialization_registry_self_hash_is_rejected(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    registry = json.loads(args.model_initialization_registry.read_text(encoding="utf-8"))
    registry["model_family"] = "arbitrary-model-family"
    args.model_initialization_registry.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_source_content_change_is_detected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    source_file = args.source_root / "src" / "clasp" / "train.py"
    source_file.chmod(0o640)
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    source_file.chmod(0o440)
    with pytest.raises(ValueError, match="source manifest hash mismatch"):
        prepare_sota_run.prepare(args, environ=environ)


def test_source_snapshot_must_be_read_only_and_exact(tmp_path: Path) -> None:
    args = _args(tmp_path)
    source_file = args.source_root / "src" / "clasp" / "train.py"
    source_file.chmod(0o640)
    with pytest.raises(ValueError, match="source snapshot file is writable"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_source_snapshot_rejects_unmanifested_file(tmp_path: Path) -> None:
    args = _args(tmp_path)
    scripts = args.source_root / "scripts"
    scripts.chmod(0o750)
    extra = scripts / "unreviewed.py"
    extra.write_text("VALUE = 3\n", encoding="utf-8")
    extra.chmod(0o440)
    scripts.chmod(0o550)
    with pytest.raises(ValueError, match="does not exactly cover"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_source_snapshot_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    args = _args(tmp_path)
    linked = tmp_path / "linked-source"
    linked.symlink_to(args.source_root, target_is_directory=True)
    args.source_root = linked
    args.source_manifest = linked / "SOURCE_MANIFEST.sha256"
    with pytest.raises(ValueError, match="root must not be a symbolic link"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )


def test_unexpected_research_environment_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    with pytest.raises(ValueError, match="contaminate.*CLASP_UNKNOWN_KNOB"):
        prepare_sota_run.prepare(
            args,
            environ={
                "CLASP_BON_ALIGNED": "1",
                "CLASP_UNKNOWN_KNOB": "surprise",
                "GRPO_DATA_SEED": "17",
            },
        )


def test_resume_requires_full_optimizer_rng_and_clasp_state(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    checkpoint = _checkpoint(args.run_dir, 400)
    assert prepare_sota_run.prepare(args, environ=environ) == f"RESUME:{checkpoint.resolve()}"

    (checkpoint / "rng_state_3.pth").unlink()
    with pytest.raises(
        ValueError,
        match="malformed checkpoint state.*rng_state_3",
    ):
        prepare_sota_run.prepare(args, environ=environ)


def test_chunked_continuations_never_finalize_root_before_fixed_horizon(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    environ = {
        **_publication_environ(),
        "CLASP_TRAINING_RUN_CONTRACT_PATH": str(args.run_dir / "run_contract.json"),
        "GRPO_STOP_AT_GLOBAL_STEP": "400",
    }
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"

    latest: Path | None = None
    for step in range(400, 3200, 400):
        latest = _checkpoint(args.run_dir, step)
        environ["GRPO_STOP_AT_GLOBAL_STEP"] = str(step)
        assert prepare_sota_run.prepare(args, environ=environ) == f"RESUME:{latest.resolve()}"

    assert latest is not None
    (args.run_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonterminal continuation.*terminal root artifacts"):
        prepare_sota_run.prepare(args, environ=environ)
    (args.run_dir / "tokenizer_config.json").unlink()

    terminal = _checkpoint(args.run_dir, 3200)
    (args.run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (args.run_dir / "adapter_model.safetensors").write_bytes(b"final")
    (args.run_dir / "checkpoint_fidelity.json").write_text("{}", encoding="utf-8")
    (args.run_dir / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "completed": True,
                "global_step": 3200,
                "max_steps": 3200,
                "latest_checkpoint_dir": str(terminal.resolve()),
            }
        ),
        encoding="utf-8",
    )
    environ["GRPO_STOP_AT_GLOBAL_STEP"] = "3200"
    assert prepare_sota_run.prepare(args, environ=environ) == f"COMPLETE:{terminal.resolve()}"


def test_nonterminal_continuation_rejects_every_known_final_save_family(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    checkpoint = _checkpoint(args.run_dir, 400)
    assert prepare_sota_run.prepare(args, environ=environ) == f"RESUME:{checkpoint.resolve()}"

    for name in (
        "README.md",
        "training_args.bin",
        "multitask_primal_dual_state.json",
        "clasp_bon_controller_state.json",
        "clasp_component_dual_state.json",
        "model-00001-of-00002.safetensors",
    ):
        artifact = args.run_dir / name
        artifact.write_bytes(b"partial-final-save")
        with pytest.raises(ValueError, match="nonterminal continuation.*terminal root artifacts"):
            prepare_sota_run.prepare(args, environ=environ)
        artifact.unlink()


def test_resume_requires_checkpoint_fidelity_receipt(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    checkpoint = _checkpoint(args.run_dir, 400)
    (checkpoint / "checkpoint_fidelity.json").unlink()
    with pytest.raises(
        ValueError,
        match="malformed checkpoint state.*checkpoint_fidelity.json",
    ):
        prepare_sota_run.prepare(args, environ=environ)


@pytest.mark.parametrize(
    "receipt_name",
    [
        prepare_sota_run.EXACT_PEFT_KL_REFERENCE_FILENAME,
        prepare_sota_run.exact_peft_kl_post_step.STEP1_RECEIPT_FILENAME,
        prepare_sota_run.exact_peft_kl_post_step.POST_STEP_RECEIPT_FILENAME,
    ],
)
def test_partial_exact_kl_receipt_without_checkpoint_is_not_fresh(
    tmp_path: Path,
    receipt_name: str,
) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    (args.run_dir / receipt_name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training artifacts but no checkpoint directory"):
        prepare_sota_run.prepare(args, environ=environ)


def test_complete_requires_fixed_horizon_final_receipt(tmp_path: Path) -> None:
    args = _args(tmp_path)
    environ = _publication_environ()
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    checkpoint = _checkpoint(args.run_dir, 3200)
    (args.run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (args.run_dir / "adapter_model.safetensors").write_bytes(b"final")
    (args.run_dir / "checkpoint_fidelity.json").write_text("{}", encoding="utf-8")
    (args.run_dir / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "completed": True,
                "global_step": 3200,
                "max_steps": 3200,
                "latest_checkpoint_dir": str(checkpoint.resolve()),
            }
        ),
        encoding="utf-8",
    )
    assert prepare_sota_run.prepare(args, environ=environ) == f"COMPLETE:{checkpoint.resolve()}"


def test_plain_grpo_does_not_require_primal_dual_state(tmp_path: Path) -> None:
    args = _args(tmp_path, method="grpo")
    environ = _publication_environ(aligned="0")
    assert prepare_sota_run.prepare(args, environ=environ) == "FRESH"
    checkpoint = _checkpoint(args.run_dir, 50, method="grpo")
    assert prepare_sota_run.prepare(args, environ=environ) == f"RESUME:{checkpoint.resolve()}"


def test_existing_uncontracted_directory_is_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.run_dir.mkdir()
    (args.run_dir / "hyperparams.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="has no run_contract.json"):
        prepare_sota_run.prepare(
            args,
            environ=_publication_environ(),
        )
