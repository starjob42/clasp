from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "audit_sota_advancement", ROOT / "scripts" / "audit_sota_advancement.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

HISTORICAL_V3_POLICY = ROOT / "experiments" / "sota_20260814" / "advancement_policy_40_step_v3.json"


def _fingerprint(root: Path, name: str) -> dict:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    artifact = path / "artifact.json"
    artifact.write_bytes(name.encode())
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    aggregate = hashlib.sha256()
    aggregate.update(b"artifact.json\0")
    aggregate.update(str(artifact.stat().st_size).encode())
    aggregate.update(b"\0")
    aggregate.update(digest.encode())
    aggregate.update(b"\n")
    return {
        "canonical_path": str(path.resolve()),
        "sha256": aggregate.hexdigest(),
        "file_count": 1,
        "size_bytes": artifact.stat().st_size,
        "files": [
            {
                "path": "artifact.json",
                "size_bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ],
    }


def _fidelity_identity(label: str = "test-policy") -> dict:
    identity = {
        "schema_version": 1,
        "protocol": "clasp-checkpoint-fidelity-identity-v1",
        "status": "publication_exact",
        "receipt_sha256": hashlib.sha256(f"{label}:receipt".encode()).hexdigest(),
        "receipt_file_sha256": hashlib.sha256(f"{label}:receipt-file".encode()).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(f"{label}:adapter-config".encode()).hexdigest(),
        "adapter_weights_sha256": hashlib.sha256(f"{label}:adapter-weights".encode()).hexdigest(),
        "trainable_tensor_inventory_sha256": hashlib.sha256(
            f"{label}:inventory".encode()
        ).hexdigest(),
        "publication_invariant_sha256": hashlib.sha256(
            f"{label}:publication-invariant".encode()
        ).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(audit._canonical_json_bytes(identity)).hexdigest()
    return identity


def _reload_identity(label: str, checkpoint_fidelity: dict) -> dict:
    identity = {
        "schema_version": 2,
        "protocol": "clasp-sft-post-save-reload-equivalence-v2",
        "status": "exact_post_save_reload_verified",
        "receipt_sha256": hashlib.sha256(f"{label}:receipt".encode()).hexdigest(),
        "receipt_file_sha256": hashlib.sha256(f"{label}:receipt-file".encode()).hexdigest(),
        "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
        "serialized_trainable_closure_sha256": hashlib.sha256(
            f"{label}:closure".encode()
        ).hexdigest(),
        "runtime_pair_sha256": hashlib.sha256(f"{label}:runtime".encode()).hexdigest(),
        "functional_equivalence_sha256": hashlib.sha256(
            f"{label}:functional".encode()
        ).hexdigest(),
        "producer_sha256": hashlib.sha256(f"{label}:producer".encode()).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(audit._canonical_json_bytes(identity)).hexdigest()
    return identity


@pytest.fixture(autouse=True)
def _stub_live_checkpoint_fidelity(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _fidelity_identity()
    monkeypatch.setattr(
        audit,
        "publication_fidelity_identity",
        lambda _path: dict(identity),
    )


def _compact_source(task: str, source_label: str, fingerprint: dict) -> dict:
    files = [
        {
            "relative_path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "rows": 1,
        }
        for item in fingerprint["files"]
    ]
    source = {
        "task": task,
        "label": source_label,
        "files": files,
        "file_manifest_sha256": hashlib.sha256(audit._canonical_json_bytes(files)).hexdigest(),
        "rows": len(files),
        "unique_instance_hashes": len(files),
    }
    source["source_identity_sha256"] = hashlib.sha256(
        audit._canonical_json_bytes(source)
    ).hexdigest()
    return {key: value for key, value in source.items() if key not in {"task", "label"}}


def _data_lineage(task: str, fingerprints: dict) -> dict:
    source_fingerprints = {
        "historical-eval": fingerprints["source"],
        "rl-train": fingerprints["data"],
        "sft-train": fingerprints["initial_model"],
    }
    registered_by_task = {
        registered_task: {
            label: _compact_source(registered_task, label, fingerprint)
            for label, fingerprint in source_fingerprints.items()
        }
        for registered_task in audit.SUPPORTED_TASKS
    }
    source_projections = [
        {
            "task": registered_task,
            "label": label,
            **registered_by_task[registered_task][label],
        }
        for registered_task in audit.SUPPORTED_TASKS
        for label in audit.REQUIRED_FORBIDDEN_SOURCE_LABELS
    ]
    registry = {
        "schema_version": 1,
        "protocol": audit.FORBIDDEN_SOURCE_REGISTRY_PROTOCOL,
        "status": audit.FORBIDDEN_SOURCE_REGISTRY_STATUS,
        "source_tree_manifest_sha256": hashlib.sha256(
            audit._canonical_json_bytes(source_projections)
        ).hexdigest(),
        "tasks": list(audit.SUPPORTED_TASKS),
        "required_labels": list(audit.REQUIRED_FORBIDDEN_SOURCE_LABELS),
        "source_count": sum(len(sources) for sources in registered_by_task.values()),
        "sources_by_task_label": registered_by_task,
    }
    registry["registry_sha256"] = hashlib.sha256(audit._canonical_json_bytes(registry)).hexdigest()

    def binding(source_label: str) -> dict:
        return {
            "task": task,
            "label": source_label,
            "canonical_path": source_fingerprints[source_label]["canonical_path"],
            "registry_sha256": registry["registry_sha256"],
            "source_tree_manifest_sha256": registry["source_tree_manifest_sha256"],
            **registered_by_task[task][source_label],
        }

    return {
        "forbidden_source_registry": registry,
        "sft_train": binding("sft-train"),
        "rl_train": binding("rl-train"),
    }


def _model_initialization_registry(
    fingerprints: dict,
    data_lineage: dict,
) -> dict:
    forbidden_registry = data_lineage["forbidden_source_registry"]
    registry = {
        "schema_version": 2,
        "protocol": audit.MODEL_INITIALIZATION_REGISTRY_PROTOCOL,
        "status": audit.MODEL_INITIALIZATION_REGISTRY_STATUS,
        "model_family": audit.MODEL_INITIALIZATION_FAMILY,
        "base_model_tree_sha256": fingerprints["base_model"]["sha256"],
        "forbidden_source_registry_sha256": forbidden_registry["registry_sha256"],
        "forbidden_source_tree_manifest_sha256": forbidden_registry["source_tree_manifest_sha256"],
        "specialist_sft_by_task": {
            specialist_task: {
                "initial_model_tree_sha256": fingerprints["initial_model"]["sha256"],
                "sft_train_source_identity_sha256": forbidden_registry["sources_by_task_label"][
                    specialist_task
                ]["sft-train"]["source_identity_sha256"],
                "sft_seed": audit.MODEL_INITIALIZATION_SFT_SEED,
                "matched_sft_contract_sha256": audit.MATCHED_SFT_CONTRACT_SHA256,
                "checkpoint_fidelity": _fidelity_identity(f"initial-{specialist_task}"),
                "reload_equivalence": _reload_identity(
                    f"initial-{specialist_task}",
                    _fidelity_identity(f"initial-{specialist_task}"),
                ),
            }
            for specialist_task in audit.SUPPORTED_TASKS
        },
    }
    registry["registry_sha256"] = hashlib.sha256(audit._canonical_json_bytes(registry)).hexdigest()
    return registry


def _policy(
    root: Path,
    *,
    model_initialization_registry_sha256: str,
    campaign_id: str = "test-campaign",
    stage_id: str = "test-pilot",
    stage_index: int = 0,
    prior_stage_receipt_chain: list[dict] | None = None,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    performance: dict | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "stage_id": stage_id,
        "stage_index": stage_index,
        "prior_stage_receipt_chain": prior_stage_receipt_chain or [],
        "status": "frozen_before_results",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "target_step": 4,
        "expected_tasks": tasks or ["tsp"],
        "expected_methods": methods or list(audit.SUPPORTED_METHODS),
        "train_seeds": [42],
        "run_path_template": "runs/{task}/{method}/seed{seed}",
        "model_initialization_registry_sha256": (model_initialization_registry_sha256),
        "contract_requirements": {
            "world_size": 2,
            "sft_seed": 42,
            "num_generations": 8,
            "logging_steps": 1,
            "save_steps": 4,
            "eval_steps": 4,
        },
        "health_gates": {key: True for key in sorted(audit.HEALTH_GATE_KEYS)},
        "resume_gate": {
            "definition": audit.RESUME_DEFINITION,
            **{key: True for key in sorted(audit.RESUME_GATE_KEYS)},
        },
        "development_performance_gate": performance,
    }
    payload["policy_sha256"] = hashlib.sha256(audit._canonical_json_bytes(payload)).hexdigest()
    path = root / "policy.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_canonical_policy_requires_failure_target_and_fixed_lambda_map(
    tmp_path: Path,
) -> None:
    path = _policy(
        tmp_path,
        model_initialization_registry_sha256="a" * 64,
        methods=list(audit.CANONICAL_PUBLICATION_METHODS),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_requirements"]["num_generations"] = 9
    payload["canonical_target_failure_rate"] = 0.005
    payload["canonical_fixed_lambda_by_task"] = {"tsp": 0.7}
    payload["fixed_null_receipt_sha256_by_task"] = {"tsp": "f" * 64}
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(audit._canonical_json_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert audit._validate_policy(path)["canonical_target_failure_rate"] == 0.005

    payload.pop("canonical_target_failure_rate")
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(audit._canonical_json_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target failure@8"):
        audit._validate_policy(path)


def test_set_null_policy_requires_exact_receipt_identity_by_task(
    tmp_path: Path,
) -> None:
    path = _policy(
        tmp_path,
        model_initialization_registry_sha256="a" * 64,
        methods=["set_null"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_requirements"]["num_generations"] = 9
    payload["canonical_target_failure_rate"] = 0.005
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(
        audit._canonical_json_bytes(payload)
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed-null receipt.*per task"):
        audit._validate_policy(path)

    payload["fixed_null_receipt_sha256_by_task"] = {"tsp": "f" * 64}
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(
        audit._canonical_json_bytes(payload)
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    validated = audit._validate_policy(path)
    assert validated["fixed_null_receipt_sha256_by_task"] == {"tsp": "f" * 64}


def test_set_null_method_profile_and_run_identity_are_fail_closed() -> None:
    profile = audit.METHOD_CONTRACT["set_null"]
    assert profile["training_config"]["policy_credit_formula"] == (
        "fixed_null_k8_same_set_utility_v1"
    )
    assert profile["training_config"]["clasp_fixed_null_set"] is True
    assert profile["training_config"]["clasp_canonical_set"] is False
    identity = {
        "schema_version": 1,
        "protocol": "clasp-fixed-null-construction-v1",
        "status": "frozen_before_training",
        "receipt_sha256": "a" * 64,
        "receipt_file_sha256": "b" * 64,
        "null_candidate_sha256": "c" * 64,
        "evaluator_identity_sha256": "d" * 64,
        "record_count": 10,
        "tasks": ["tsp"],
    }
    assert audit._validate_fixed_null_receipt_identity(
        identity,
        task="tsp",
        expected_receipt_sha256="a" * 64,
    ) == []
    errors = audit._validate_fixed_null_receipt_identity(
        identity,
        task="tsp",
        expected_receipt_sha256="e" * 64,
    )
    assert any("receipt_sha256" in error for error in errors)


def _history(method: str) -> list[dict]:
    rows: list[dict] = []
    for step in range(1, 5):
        row = {"step": step, "loss": 0.0, "grad_norm": 0.1}
        if method == "grpo":
            row["reward_std"] = 0.4
        elif method == "dual_grpo":
            row["constrained_grpo/std_reward_advantage"] = 0.4
            row["constrained_grpo/mean_objective_quality"] = 0.5
        else:
            row["constrained_grpo/clasp/advantage_abs_mean"] = 0.4
            row[
                "constrained_grpo/clasp/bon_dense_quality_nonzero_rate"
                if method == "clasp_dense"
                else "constrained_grpo/clasp/bon_success_credit_nonzero_rate"
            ] = 0.5
            for metric in (
                "constrained_grpo/component_dual_recovery_active",
                "constrained_grpo/component_dual_recovery_api_available",
                "constrained_grpo/clasp/bon_controller_commit",
                "constrained_grpo/clasp/bon_controller_pressure_observed",
            ):
                row[metric] = 1.0
        row["constrained_grpo/objective_quality_fallback"] = 0.0
        rows.append(row)
    rows.append({"step": 4, "eval_loss": 0.1})
    return rows


def _pd_state(task: str, *, clasp: bool) -> dict:
    state = {
        task: {
            "cost_names": ["parse_failure"],
            "cost_limits": [0.0],
            "cost_weights": [1.0],
            "dual_lambdas": [0.1],
            "controller_state": {"proper_dual_gradient_mode": "signed"},
        }
    }
    if clasp:
        state["__clasp_bon__"] = {
            "lambdas": {task: 0.1},
            "failure_ema": {task: 0.2},
            "observed_groups": {task: 4},
        }
    return state


def _write_run(
    root: Path,
    *,
    task: str,
    method: str,
    seed: int,
    fingerprints: dict,
    data_lineage: dict,
    model_initialization_registry: dict,
    authorization: dict,
) -> Path:
    run = root / "runs" / task / method / f"seed{seed}"
    checkpoint = run / "checkpoints" / "checkpoint-4"
    checkpoint.mkdir(parents=True)
    training = {
        "entrypoint": (
            "clasp.entrypoints.train_grpo_standard"
            if method == "grpo"
            else "clasp.entrypoints.train_grpo"
        ),
        "logging_steps": 1,
        "save_steps": 4,
        "eval_steps": 4,
        "final_max_steps": 4,
        "checkpoint_fidelity": "publication_exact_required",
        "beta": 0.0,
        **authorization,
        "clasp_bon_success_credit": ("exact_loo" if method == "clasp_exact" else "dense_rank"),
        "use_primal_dual": method != "grpo",
        "use_clasp": method.startswith("clasp_"),
    }
    environment = {
        "GRPO_STRICT_OPTIMIZER_RESUME": "1",
        "CLASP_AUX_GROUPING_STRICT": "1",
        "CLASP_BON_ALIGNED": "1" if method.startswith("clasp_") else "0",
        "CLASP_FEASIBILITY_GATED": "1" if method.startswith("clasp_") else "0",
        "CLASP_BON_SUCCESS_CREDIT": ("exact_loo" if method == "clasp_exact" else "dense_rank"),
        "CLASP_PUBLICATION_FIDELITY": "1",
    }
    contract = {
        "schema_version": 2,
        "stage": "grpo",
        "method": method,
        "task": task,
        "seeds": {"train": seed, "data": seed, "sft": 42},
        "fixed_horizon": {"max_steps": 4, "world_size": 2},
        "data_lineage": data_lineage,
        "model_initialization_registry": model_initialization_registry,
        "fixed_kl_reference": None,
        "fixed_null_receipt": None,
        "fingerprints": fingerprints,
        "generation": {"num_generations": 8, "temperature": 0.7},
        "training_config": training,
        "tracked_environment": environment,
    }
    contract_path = run / "run_contract.json"
    contract_path.write_text(json.dumps(contract))
    os.chmod(contract_path, 0o440)

    history = _history(method)
    trainer_state = {"global_step": 4, "max_steps": 4, "log_history": history}
    (checkpoint / "trainer_state.json").write_text(json.dumps(trainer_state))
    for name in (
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_model.safetensors",
        "rng_state_0.pth",
        "rng_state_1.pth",
        "checkpoint_fidelity.json",
    ):
        (checkpoint / name).write_bytes(f"{method}:{name}".encode())
    if method != "grpo":
        (checkpoint / "multitask_primal_dual_state.json").write_text(
            json.dumps(_pd_state(task, clasp=method.startswith("clasp_")))
        )
    (run / "adapter_config.json").write_text("{}")
    (run / "adapter_model.safetensors").write_bytes(f"final:{method}".encode())
    (run / "checkpoint_fidelity.json").write_text("{}")
    (run / "grpo_trainer_state.json").write_text(json.dumps(trainer_state))
    (run / "grpo_log_history.json").write_text(json.dumps(history))
    (run / "stage_status.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "completed": True,
                "global_step": 4,
                "max_steps": 4,
                "has_saved_model": True,
                "latest_checkpoint_dir": str(checkpoint.resolve()),
            }
        )
    )
    return run


def _campaign(
    tmp_path: Path,
    *,
    performance: dict | None = None,
    campaign_id: str = "test-campaign",
    stage_id: str = "test-pilot",
    stage_index: int = 0,
    prior_stage_receipt_chain: list[dict] | None = None,
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    fingerprint_root = tmp_path / "fingerprints"
    fingerprints = {
        name: _fingerprint(fingerprint_root, name)
        for name in ("initial_model", "base_model", "data", "source")
    }
    data_lineage = _data_lineage("tsp", fingerprints)
    model_initialization_registry = _model_initialization_registry(
        fingerprints,
        data_lineage,
    )
    policy = _policy(
        tmp_path,
        model_initialization_registry_sha256=model_initialization_registry["registry_sha256"],
        campaign_id=campaign_id,
        stage_id=stage_id,
        stage_index=stage_index,
        prior_stage_receipt_chain=prior_stage_receipt_chain,
        performance=performance,
    )
    policy_payload = json.loads(policy.read_text())
    authorization = {
        "campaign_id": campaign_id,
        "advancement_stage_id": stage_id,
        "advancement_stage_index": stage_index,
        "advancement_policy_sha256": policy_payload["policy_sha256"],
        "advancement_prior_stage_receipt_chain_sha256": hashlib.sha256(
            audit._canonical_json_bytes(policy_payload["prior_stage_receipt_chain"])
        ).hexdigest(),
    }
    for method in audit.SUPPORTED_METHODS:
        _write_run(
            artifact_root,
            task="tsp",
            method=method,
            seed=42,
            fingerprints=fingerprints,
            data_lineage=data_lineage,
            model_initialization_registry=model_initialization_registry,
            authorization=authorization,
        )
    return policy, artifact_root


def _freeze_receipt(
    root: Path,
    name: str,
    receipt: dict,
) -> tuple[Path, dict]:
    path = root / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o440)
    return path, {
        "campaign_id": receipt["campaign_id"],
        "stage_id": receipt["stage_id"],
        "policy_sha256": receipt["policy"]["policy_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_complete_matched_campaign_advances(tmp_path: Path) -> None:
    policy, root = _campaign(tmp_path)
    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    assert receipt["decision"] == "ADVANCE"
    assert receipt["passed_run_count"] == 4
    assert receipt["matched_contract_comparability"]["status"] == "PASS"
    assert receipt["publication_claim_authorized"] is False
    assert receipt["side_effects"]["scheduler_actions"] == []


def test_terminal_root_and_checkpoint_fidelity_must_be_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, root = _campaign(tmp_path)

    def identity(path: Path) -> dict:
        candidate = Path(path)
        return _fidelity_identity(
            "terminal-checkpoint" if candidate.name.startswith("checkpoint-") else "final-root"
        )

    monkeypatch.setattr(audit, "publication_fidelity_identity", identity)
    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    assert receipt["decision"] == "HOLD"
    assert any(
        "final root policy differs from terminal checkpoint fidelity identity" in error
        for error in receipt["errors"]
    )


def test_immutable_historical_v3_policy_is_rejected_by_current_gate() -> None:
    original = HISTORICAL_V3_POLICY.read_bytes()
    with pytest.raises(ValueError, match="stage_index"):
        audit._validate_policy(HISTORICAL_V3_POLICY)
    assert HISTORICAL_V3_POLICY.read_bytes() == original


def test_complete_prior_stage_pass_chain_authorizes_noninitial_stage(
    tmp_path: Path,
) -> None:
    first_policy, first_root = _campaign(
        tmp_path / "first",
        stage_id="initial-smoke",
    )
    first_receipt = audit.audit_campaign(policy_path=first_policy, run_root=first_root)
    first_path, first_identity = _freeze_receipt(
        tmp_path / "receipts",
        "initial-smoke",
        first_receipt,
    )

    second_policy, second_root = _campaign(
        tmp_path / "second",
        stage_id="pilot",
        stage_index=1,
        prior_stage_receipt_chain=[first_identity],
    )
    second_receipt = audit.audit_campaign(
        policy_path=second_policy,
        run_root=second_root,
        prior_stage_receipts=[first_path],
    )
    second_path, second_identity = _freeze_receipt(
        tmp_path / "receipts",
        "pilot",
        second_receipt,
    )

    final_policy, final_root = _campaign(
        tmp_path / "final",
        stage_id="selection",
        stage_index=2,
        prior_stage_receipt_chain=[first_identity, second_identity],
    )
    final_receipt = audit.audit_campaign(
        policy_path=final_policy,
        run_root=final_root,
        prior_stage_receipts=[first_path, second_path],
    )
    assert final_receipt["decision"] == "ADVANCE"
    assert final_receipt["prior_stage_receipt_chain"] == [
        first_identity,
        second_identity,
    ]
    authorization = final_receipt["prior_stage_authorization"]
    assert authorization["status"] == "PASS"
    assert authorization["validated_receipt_count"] == 2
    assert [item["status"] for item in authorization["receipts"]] == ["PASS", "PASS"]


def test_noninitial_stage_holds_when_complete_prior_chain_is_not_supplied(
    tmp_path: Path,
) -> None:
    first_policy, first_root = _campaign(tmp_path / "first", stage_id="initial-smoke")
    first_receipt = audit.audit_campaign(policy_path=first_policy, run_root=first_root)
    _first_path, first_identity = _freeze_receipt(
        tmp_path / "receipts",
        "initial-smoke",
        first_receipt,
    )
    policy, root = _campaign(
        tmp_path / "second",
        stage_id="pilot",
        stage_index=1,
        prior_stage_receipt_chain=[first_identity],
    )

    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    assert receipt["decision"] == "HOLD"
    assert receipt["prior_stage_authorization"]["status"] == "FAIL"
    assert any("receipt count" in error for error in receipt["errors"])


@pytest.mark.parametrize("failure", ["writable", "tampered", "hold", "race"])
def test_noninitial_stage_rejects_invalid_prior_pass_receipt(
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_policy, first_root = _campaign(tmp_path / "first", stage_id="initial-smoke")
    first_receipt = audit.audit_campaign(policy_path=first_policy, run_root=first_root)
    if failure == "hold":
        first_receipt["decision"] = "HOLD"
        first_receipt.pop("receipt_sha256")
        first_receipt["receipt_sha256"] = hashlib.sha256(
            audit._canonical_json_bytes(first_receipt)
        ).hexdigest()
    first_path, first_identity = _freeze_receipt(
        tmp_path / "receipts",
        "initial-smoke",
        first_receipt,
    )
    policy, root = _campaign(
        tmp_path / "second",
        stage_id="pilot",
        stage_index=1,
        prior_stage_receipt_chain=[first_identity],
    )
    if failure == "writable":
        os.chmod(first_path, 0o640)
    elif failure == "tampered":
        os.chmod(first_path, 0o640)
        payload = json.loads(first_path.read_text())
        payload["generated_at_utc"] = "2099-01-01T00:00:00Z"
        first_path.write_text(json.dumps(payload))
        os.chmod(first_path, 0o440)
    elif failure == "race":
        original_identity = audit._stat_identity
        calls = 0

        def unstable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
            nonlocal calls
            calls += 1
            identity = original_identity(metadata)
            if calls == 3:
                return (*identity[:-1], identity[-1] + 1)
            return identity

        monkeypatch.setattr(audit, "_stat_identity", unstable_identity)

    receipt = audit.audit_campaign(
        policy_path=policy,
        run_root=root,
        prior_stage_receipts=[first_path],
    )
    assert receipt["decision"] == "HOLD"
    assert receipt["prior_stage_authorization"]["status"] == "FAIL"
    expected_error = {
        "writable": "writable",
        "tampered": "self-hash mismatch",
        "hold": "decision is not ADVANCE",
        "race": "changed or its pathname was replaced",
    }[failure]
    assert any(expected_error in error for error in receipt["errors"])


def test_policy_must_prefreeze_model_initialization_registry(tmp_path: Path) -> None:
    policy, root = _campaign(tmp_path)
    payload = json.loads(policy.read_text())
    payload.pop("model_initialization_registry_sha256")
    payload.pop("policy_sha256")
    payload["policy_sha256"] = hashlib.sha256(audit._canonical_json_bytes(payload)).hexdigest()
    policy.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="model_initialization_registry_sha256",
    ):
        audit.audit_campaign(policy_path=policy, run_root=root)


@pytest.mark.parametrize("failure", ["nan", "credit", "skip", "rng"])
def test_health_and_resume_defects_hold_campaign(tmp_path: Path, failure: str) -> None:
    policy, root = _campaign(tmp_path)
    run = root / "runs" / "tsp" / "clasp_dense" / "seed42"
    history_path = run / "grpo_log_history.json"
    state_path = run / "grpo_trainer_state.json"
    checkpoint_state_path = run / "checkpoints" / "checkpoint-4" / "trainer_state.json"
    if failure == "rng":
        (run / "checkpoints" / "checkpoint-4" / "rng_state_1.pth").unlink()
    else:
        history = json.loads(history_path.read_text())
        if failure == "nan":
            history[0]["loss"] = float("nan")
        elif failure == "credit":
            for row in history:
                if isinstance(row, dict):
                    row["constrained_grpo/clasp/bon_dense_quality_nonzero_rate"] = 0.0
        else:
            history[0]["constrained_grpo/clasp_group_aux_skipped"] = 1.0
        state = json.loads(state_path.read_text())
        state["log_history"] = history
        checkpoint_state = json.loads(checkpoint_state_path.read_text())
        checkpoint_state["log_history"] = history
        history_path.write_text(json.dumps(history))
        state_path.write_text(json.dumps(state))
        checkpoint_state_path.write_text(json.dumps(checkpoint_state))
    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    assert receipt["decision"] == "HOLD"
    assert receipt["runs"]["tsp/clasp_dense/seed42"]["status"] == "FAIL"


def test_method_contract_drift_fails_comparability_and_run_gate(tmp_path: Path) -> None:
    policy, root = _campaign(tmp_path)
    path = root / "runs" / "tsp" / "dual_grpo" / "seed42" / "run_contract.json"
    os.chmod(path, 0o640)
    contract = json.loads(path.read_text())
    contract["training_config"]["learning_rate"] = 2e-6
    path.write_text(json.dumps(contract))
    os.chmod(path, 0o440)
    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    assert receipt["decision"] == "HOLD"
    assert receipt["matched_contract_comparability"]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("failure", "error_fragment"),
    [
        ("missing", "run_contract.data_lineage is not an object"),
        ("registry", "source_count disagrees"),
        ("sft", "data_lineage.sft_train.label"),
        ("rl", "data_lineage.rl_train.source_identity_sha256 disagrees"),
        ("rl_projection", "fingerprints.data differs from registered RL data"),
    ],
)
def test_missing_or_wrong_training_lineage_holds_campaign(
    tmp_path: Path,
    failure: str,
    error_fragment: str,
) -> None:
    policy, root = _campaign(tmp_path)
    path = root / "runs" / "tsp" / "clasp_dense" / "seed42" / "run_contract.json"
    os.chmod(path, 0o640)
    contract = json.loads(path.read_text())
    if failure == "missing":
        contract.pop("data_lineage")
    elif failure == "registry":
        contract["data_lineage"]["forbidden_source_registry"]["source_count"] += 1
    elif failure == "sft":
        contract["data_lineage"]["sft_train"]["label"] = "rl-train"
    elif failure == "rl":
        contract["data_lineage"]["rl_train"]["source_identity_sha256"] = "0" * 64
    else:
        fingerprint = contract["fingerprints"]["data"]
        fingerprint["files"][0]["path"] = "other.json"
        aggregate = hashlib.sha256()
        for item in fingerprint["files"]:
            aggregate.update(item["path"].encode())
            aggregate.update(b"\0")
            aggregate.update(str(item["size_bytes"]).encode())
            aggregate.update(b"\0")
            aggregate.update(item["sha256"].encode())
            aggregate.update(b"\n")
        fingerprint["sha256"] = aggregate.hexdigest()
    path.write_text(json.dumps(contract))
    os.chmod(path, 0o440)

    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    run = receipt["runs"]["tsp/clasp_dense/seed42"]
    assert receipt["decision"] == "HOLD"
    assert run["contract_validation_status"] == "FAIL"
    assert any(error_fragment in error for error in run["errors"])


@pytest.mark.parametrize(
    ("failure", "error_fragment"),
    [
        ("missing", "model_initialization_registry is not an object"),
        ("policy_hash", "registry_sha256 disagrees with advancement policy"),
        ("base", "base_model_tree_sha256 disagrees with base fingerprint"),
        ("forbidden", "disagrees with the forbidden-source registry hash"),
        ("specialist", "specialist initial model disagrees with initial fingerprint"),
        ("matched", "matched_sft_contract_sha256"),
    ],
)
def test_wrong_model_initialization_registry_holds_campaign(
    tmp_path: Path,
    failure: str,
    error_fragment: str,
) -> None:
    policy, root = _campaign(tmp_path)
    path = root / "runs" / "tsp" / "clasp_dense" / "seed42" / "run_contract.json"
    os.chmod(path, 0o640)
    contract = json.loads(path.read_text())
    registry = contract.get("model_initialization_registry")
    if failure == "missing":
        contract.pop("model_initialization_registry")
    elif failure == "policy_hash":
        registry["registry_sha256"] = "0" * 64
    elif failure == "base":
        registry["base_model_tree_sha256"] = "0" * 64
    elif failure == "forbidden":
        registry["forbidden_source_registry_sha256"] = "0" * 64
    elif failure == "specialist":
        registry["specialist_sft_by_task"]["tsp"]["initial_model_tree_sha256"] = "0" * 64
    else:
        registry["specialist_sft_by_task"]["tsp"].pop("matched_sft_contract_sha256")
    path.write_text(json.dumps(contract))
    os.chmod(path, 0o440)

    receipt = audit.audit_campaign(policy_path=policy, run_root=root)
    run = receipt["runs"]["tsp/clasp_dense/seed42"]
    assert receipt["decision"] == "HOLD"
    assert run["contract_validation_status"] == "FAIL"
    assert any(error_fragment in error for error in run["errors"])


def _aggregate(pass_delta: float) -> dict:
    baseline_pass = 0.60
    challenger_pass = baseline_pass + pass_delta
    summaries = {}
    for method, pass_at_k, gap in (
        ("dual_grpo", baseline_pass, 0.10),
        ("clasp_dense", challenger_pass, 0.09),
    ):
        summaries[method] = {
            "tsp": {
                "per_seed": {
                    "42": {
                        "pass_at_k": pass_at_k,
                        "conditional_gap_mean": gap,
                    }
                }
            }
        }
    payload = {
        "schema_version": 3,
        "evidence_scope": "development_only",
        "expected_candidate_budget": 8,
        "summaries": summaries,
        "comparison": {
            "baseline": "dual_grpo",
            "challenger": "clasp_dense",
            "per_task": {"tsp": {}},
        },
    }
    payload["aggregate_receipt_sha256"] = hashlib.sha256(
        audit._canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def test_optional_pilot_performance_gate_is_prespecified(tmp_path: Path) -> None:
    gate = {
        "baseline_method": "dual_grpo",
        "challenger_method": "clasp_dense",
        "max_pass_at_k_regression": 0.02,
        "min_tasks_improved": 1,
        "required_tasks": ["tsp"],
    }
    policy, root = _campaign(tmp_path, performance=gate)
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps(_aggregate(-0.01)))
    receipt = audit.audit_campaign(
        policy_path=policy,
        run_root=root,
        development_aggregate=aggregate,
    )
    assert receipt["decision"] == "ADVANCE"
    assert receipt["development_performance_gate"]["improved_task_count"] == 1

    aggregate.write_text(json.dumps(_aggregate(-0.03)))
    receipt = audit.audit_campaign(
        policy_path=policy,
        run_root=root,
        development_aggregate=aggregate,
    )
    assert receipt["decision"] == "HOLD"


def test_development_aggregate_tamper_holds_advancement(tmp_path: Path) -> None:
    gate = {
        "baseline_method": "dual_grpo",
        "challenger_method": "clasp_dense",
        "max_pass_at_k_regression": 0.02,
        "min_tasks_improved": 1,
        "required_tasks": ["tsp"],
    }
    policy, root = _campaign(tmp_path, performance=gate)
    aggregate = tmp_path / "aggregate.json"
    payload = _aggregate(0.01)
    aggregate.write_text(json.dumps(payload))
    assert (
        audit.audit_campaign(
            policy_path=policy,
            run_root=root,
            development_aggregate=aggregate,
        )["decision"]
        == "ADVANCE"
    )

    payload["summaries"]["clasp_dense"]["tsp"]["per_seed"]["42"]["pass_at_k"] = 0.0
    aggregate.write_text(json.dumps(payload))
    receipt = audit.audit_campaign(
        policy_path=policy,
        run_root=root,
        development_aggregate=aggregate,
    )
    assert receipt["decision"] == "HOLD"
    assert "development aggregate receipt self-hash mismatch" in receipt["errors"]


def test_cli_writes_atomic_hold_receipt_for_invalid_policy(tmp_path: Path) -> None:
    policy, root = _campaign(tmp_path)
    payload = json.loads(policy.read_text())
    payload["target_step"] = 99
    policy.write_text(json.dumps(payload))
    output = tmp_path / "receipts" / "audit.json"
    assert (
        audit.main(
            [
                "--policy",
                str(policy),
                "--run-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    receipt = json.loads(output.read_text())
    assert receipt["decision"] == "HOLD"
    assert audit._is_sha256(receipt["receipt_sha256"])
    assert output.stat().st_mode & 0o222 == 0
    assert not list(output.parent.glob(".audit.json.*.tmp"))
    original = output.read_bytes()
    assert (
        audit.main(
            [
                "--policy",
                str(policy),
                "--run-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == original
