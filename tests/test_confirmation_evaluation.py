from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from clasp.baselines.foarl_f638_comparison import (
    FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST,
)
from clasp.data.benchmark import forbidden_source_registry_identity
from clasp.entrypoints.evaluate import (
    _APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK,
    _CONFIRMATION_CAMPAIGN_SCHEMA_VERSION,
    _CONFIRMATION_METHOD_SPECIFIC_ALLOWLIST,
    _benchmark_manifest_identity,
    _canonical_json_bytes,
    _checkpoint_selection_receipt_identity,
    _claim_confirmation_attempt,
    _confirmation_attempt_id,
    _dataset_content_identity,
    _load_campaign_manifest_snapshot,
    _sha256_file,
    _strict_verifier_identity,
    _validate_confirmation_campaign_binding,
    _validate_confirmation_checkpoint_selection,
    _validate_confirmation_dataset,
    _validate_confirmation_training_design_contract,
    _validate_training_run_contract,
    build_eval_parser,
    parse_eval_args,
)


def test_evaluator_accepts_only_the_pair_specific_foarl_shared_design_policy() -> None:
    tasks = ["cvrp", "jssp", "mis", "op", "tsp"]
    seeds = list(range(42, 50))
    manifest = {
        "methods": [
            "foarl_f638_matched_b9",
            "clasp_canonical_foarl_matched_b9",
        ],
        "tasks": tasks,
        "training_seeds": seeds,
        "training_design": {
            "schema_version": 1,
            "fixed_horizon": {"max_steps": 3_200, "world_size": 4},
            "method_specific_allowlist": list(
                FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST
            ),
            "shared_design_sha256_by_task_seed": {
                f"{task}/seed{seed}": hashlib.sha256(
                    f"foarl-shared/{task}/{seed}".encode()
                ).hexdigest()
                for task in tasks
                for seed in seeds
            },
            "fixed_null_receipt_sha256_by_task": {},
        },
    }
    _validate_confirmation_training_design_contract(manifest)
    manifest["training_design"]["method_specific_allowlist"] = list(
        _CONFIRMATION_METHOD_SPECIFIC_ALLOWLIST
    )
    with pytest.raises(ValueError, match="matched training-design policy"):
        _validate_confirmation_training_design_contract(manifest)


def _checkpoint_fidelity_identity(label: str = "selected") -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": 1,
        "protocol": "clasp-checkpoint-fidelity-identity-v1",
        "status": "publication_exact",
        "receipt_sha256": hashlib.sha256(f"{label}/receipt".encode()).hexdigest(),
        "receipt_file_sha256": hashlib.sha256(f"{label}/receipt-file".encode()).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(f"{label}/config".encode()).hexdigest(),
        "adapter_weights_sha256": hashlib.sha256(f"{label}/weights".encode()).hexdigest(),
        "trainable_tensor_inventory_sha256": hashlib.sha256(
            f"{label}/inventory".encode()
        ).hexdigest(),
        "publication_invariant_sha256": hashlib.sha256(f"{label}/invariant".encode()).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return identity


def _checkpoint_artifact(
    fidelity: dict[str, object],
    *,
    manifest_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "manifest_sha256": manifest_sha256,
        "files": [
            {
                "path": "adapter_config.json",
                "sha256": fidelity["adapter_config_sha256"],
                "size": 1,
            },
            {
                "path": "adapter_model.safetensors",
                "sha256": fidelity["adapter_weights_sha256"],
                "size": 1,
            },
            {
                "path": "checkpoint_fidelity.json",
                "sha256": fidelity["receipt_file_sha256"],
                "size": 1,
            },
        ],
    }


def _selection_registry(*, instances: int) -> tuple[dict, list[dict], str]:
    evaluation_files = [
        {
            "task": "tsp",
            "split": "eval",
            "path": "tsp/eval/test.json",
            "size": 3,
            "sha256": "e" * 64,
        }
    ]
    sources = []
    for label in ("historical-eval", "rl-train", "sft-train"):
        files = [
            {
                "relative_path": (
                    evaluation_files[0]["path"]
                    if label == "historical-eval"
                    else f"tsp/{label}/data.json"
                ),
                "sha256": (
                    evaluation_files[0]["sha256"]
                    if label == "historical-eval"
                    else hashlib.sha256(label.encode()).hexdigest()
                ),
                "size_bytes": 3,
                "rows": instances,
            }
        ]
        source = {
            "task": "tsp",
            "label": label,
            "files": files,
            "file_manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
            "rows": instances,
            "unique_instance_hashes": instances,
        }
        source["source_identity_sha256"] = hashlib.sha256(_canonical_json_bytes(source)).hexdigest()
        sources.append(source)
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": hashlib.sha256(_canonical_json_bytes(sources)).hexdigest(),
    }
    registry["registry_sha256"] = hashlib.sha256(_canonical_json_bytes(registry)).hexdigest()
    historical = next(source for source in sources if source["label"] == "historical-eval")
    return registry, evaluation_files, str(historical["source_identity_sha256"])


def _write_campaign(
    path: Path,
    *,
    benchmark_manifest_sha256: str,
    selection_identity: dict | None = None,
    selection_payload: dict | None = None,
) -> Path:
    campaign_id = (
        str(selection_identity["campaign_id"])
        if selection_identity is not None
        else "test-confirmation-campaign"
    )
    baseline = "grpo_z"
    challenger = "clasp_canonical"
    methods = [baseline, challenger]
    tasks = ["cvrp", "jssp", "mis", "op", "tsp"]
    seeds = list(range(42, 50))
    receipt_identity = selection_identity or {
        "basename": "selection.json",
        "file_sha256": "1" * 64,
        "size_bytes": 1,
        "receipt_sha256": "2" * 64,
        "campaign_id": campaign_id,
        "stage_id": "selection",
    }
    receipt_payload = selection_payload or {}
    checkpoint_fidelity = _checkpoint_fidelity_identity()
    payload = {
        "schema_version": _CONFIRMATION_CAMPAIGN_SCHEMA_VERSION,
        "status": "frozen_before_confirmation",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "campaign_id": campaign_id,
        "lane": "raw_n8",
        "decode_seed": 42,
        "baseline": baseline,
        "challenger": challenger,
        "methods": methods,
        "training_seeds": seeds,
        "tasks": tasks,
        "lane_contract_registry_sha256": "3" * 64,
        "target_registry_sha256": "4" * 64,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "forbidden_source_registry_sha256": "5" * 64,
        "forbidden_source_tree_manifest_sha256": "6" * 64,
        "approved_fixed_null_receipt_sha256_by_task": dict(
            _APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
        ),
        "analysis": {
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 20_260_814,
            "alpha": 0.05,
            "primary_outcome": {
                "metric": "normalized_feasible_gap_success_curve_auc",
                "gap_range": [0.0, 0.10],
                "infeasible_score": 0.0,
                "estimand": "paired_seed_mean_challenger_minus_baseline",
                "inference_unit": "paired_training_seed",
                "test": "exact_two_sided_paired_seed_sign_flip",
                "multiplicity": "holm_across_five_tasks",
                "direction": "higher_is_better",
                "decision_rule": "holm_adjusted_p <= alpha and effect_mean > 0",
                "negative_direction_label": "difference_detected_against_clasp",
                "family": "normalized_feasible_gap_auc_seed_randomization",
            },
            "pass_at_k_noninferiority_gate": {
                "metric": "strict_shared_verifier_pass_at_8",
                "margin": 0.01,
                "null_hypothesis": "challenger_minus_baseline <= -0.01",
                "test": "exact_one_sided_paired_seed_sign_flip_at_margin_boundary",
                "multiplicity": "holm_across_five_tasks",
                "observed_mean_requirement": "challenger_minus_baseline >= -0.01",
                "overall_rule": "all_five_tasks_pass",
                "family": "pass_at_k_noninferiority_seed_randomization",
            },
            "secondary_metrics": [
                "candidate_feasibility",
                "conditional_verified_gap",
                "lexicographic_win_score",
                "success_at_external_target_gap_when_eligible",
                "generation_and_repair_compute",
            ],
        },
        "training_design": {
            "schema_version": 1,
            "fixed_horizon": {"max_steps": 400, "world_size": 4},
            "method_specific_allowlist": list(_CONFIRMATION_METHOD_SPECIFIC_ALLOWLIST),
            "shared_design_sha256_by_task_seed": {
                f"{task}/seed{seed}": hashlib.sha256(
                    f"shared-design/{task}/{seed}".encode()
                ).hexdigest()
                for task in tasks
                for seed in seeds
            },
            "fixed_null_receipt_sha256_by_task": {},
        },
        "checkpoint_selection_receipt": {
            "identity": receipt_identity,
            "payload": receipt_payload,
        },
        "confirmation_attempt_ledger": {
            "protocol": "posix-create-once-lock-v1",
            "canonical_root": str((path.parent / "shared-attempt-ledger").resolve()),
        },
        "runs": [],
    }
    for method in methods:
        for seed in seeds:
            for task in tasks:
                entry = {
                    "method": method,
                    "training_seed": seed,
                    "task": task,
                    "training_run_contract_sha256": hashlib.sha256(
                        f"contract/{method}/{seed}/{task}".encode()
                    ).hexdigest(),
                    "checkpoint_step": 400,
                    "checkpoint_artifact_sha256": hashlib.sha256(
                        f"checkpoint/{method}/{seed}/{task}".encode()
                    ).hexdigest(),
                    "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
                    "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
                    "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity[
                        "receipt_file_sha256"
                    ],
                    "grpo_beta": 0.0,
                    "fixed_kl_reference": None,
                    "exact_peft_kl_runtime": None,
                    "exact_peft_kl_post_step": None,
                    "checkpoint_selection_receipt_sha256": receipt_identity["receipt_sha256"],
                }
                entry["confirmation_attempt_id"] = _confirmation_attempt_id(payload, entry)
                payload["runs"].append(entry)
    payload["campaign_manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(path: Path, data_file: Path) -> Path:
    file_sha256, size_bytes = _sha256_file(str(data_file))
    verifier = _strict_verifier_identity()
    verifier_sha256 = hashlib.sha256(_canonical_json_bytes(verifier)).hexdigest()
    solver = {
        "schema_version": 2,
        "status": "approved_for_confirmation",
        "reference_evidence_runner": {
            "schema_version": 1,
            "protocol": "clasp-iid-reference-evidence-v1",
            "expected_instances_per_task": 1,
            "strict_verifier_identity_sha256": verifier_sha256,
        },
        "tasks": {
            "tsp": {
                "accepted_statuses": ["BEST_KNOWN_HEURISTIC_VERIFIED"],
                "parameters": {"seed": 1},
                "evidence_runner": {
                    "approval": "approved",
                    "adapter": "test_tsp_v1",
                    "accepted_native_statuses": ["HEURISTIC_COMPLETE"],
                    "objective_semantics": "clasp_raw_euclidean_float64",
                    "reference_quality": "best_known_heuristic",
                    "seed_parameter": "seed",
                    "solver_identity": {
                        "backend_name": "test-tsp",
                        "native_version": "1",
                        "artifact_kind": "test_fixture",
                        "artifact_path": "/frozen/test-tsp.artifact",
                        "artifact_sha256": "a" * 64,
                    },
                },
            }
        },
    }
    solver["contract_sha256"] = hashlib.sha256(_canonical_json_bytes(solver)).hexdigest()
    solver_sha256 = hashlib.sha256(_canonical_json_bytes(solver)).hexdigest()
    generator = {"schema_version": 1, "name": "test-generator"}
    generator_sha256 = hashlib.sha256(_canonical_json_bytes(generator)).hexdigest()
    semantic = ["1" * 64]
    rows = ["2" * 64]
    references = ["3" * 64]
    evidence = ["4" * 64]
    exports = ["5" * 64]
    files = [
        {
            "relative_path": "test.json",
            "sha256": file_sha256,
            "size_bytes": size_bytes,
            "rows": 1,
        }
    ]
    forbidden_sources = []
    registry_sources = []
    for label in ("historical-eval", "rl-train", "sft-train"):
        source = {
            "task": "tsp",
            "label": label,
            "source_locator": f"forbidden/{label}.json",
            "files": files,
            "file_manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        projection = {
            key: source[key]
            for key in (
                "task",
                "label",
                "files",
                "file_manifest_sha256",
                "rows",
                "unique_instance_hashes",
            )
        }
        projection["source_identity_sha256"] = hashlib.sha256(
            _canonical_json_bytes(projection)
        ).hexdigest()
        source["source_identity_sha256"] = projection["source_identity_sha256"]
        forbidden_sources.append(source)
        registry_sources.append(projection)
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": registry_sources,
        "source_tree_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(registry_sources)
        ).hexdigest(),
    }
    registry["registry_sha256"] = hashlib.sha256(_canonical_json_bytes(registry)).hexdigest()
    registry_identity = forbidden_source_registry_identity(registry)
    payload = {
        "schema_version": 3,
        "status": "frozen_before_inference",
        "benchmark_mode": "publication",
        "minimum_instances_per_task": 1,
        "tasks": ["tsp"],
        "contracts": {
            "generator": generator,
            "generator_contract_sha256": generator_sha256,
            "reference_solver": solver,
            "reference_solver_contract_sha256": solver_sha256,
        },
        "strict_verifier": verifier,
        "generation_receipt": {
            "status": "references_attached",
            "receipt_sha256": "7" * 64,
            "generator_code_sha256": "9" * 64,
            "reference_solver_contract_sha256": solver_sha256,
            "strict_verifier": verifier,
            "forbidden_source_registry_identity": registry_identity,
            "reference_evidence_schema_version": 1,
            "reference_export_summary_sha256": "8" * 64,
        },
        "test_sets": {
            "tsp": {
                "source_locator": data_file.resolve().relative_to(path.parent.resolve()).as_posix(),
                "files": files,
                "file_manifest_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
                "rows": 1,
                "ordered_instance_sha256": semantic,
                "ordered_semantic_instance_sha256": semantic,
                "instance_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(semantic)
                ).hexdigest(),
                "ordered_benchmark_row_sha256": rows,
                "row_hashes_sha256": hashlib.sha256(_canonical_json_bytes(rows)).hexdigest(),
                "ordered_reference_output_sha256": references,
                "reference_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(references)
                ).hexdigest(),
                "reference_audit": {
                    "strictly_feasible": 1,
                    "zero_gap": 1,
                    "objective_min": 1.0,
                    "objective_max": 1.0,
                },
                "reference_evidence_schema_version": 1,
                "reference_evidence_summary_sha256": "6" * 64,
                "ordered_reference_evidence_record_sha256": evidence,
                "reference_evidence_records_sha256": hashlib.sha256(
                    _canonical_json_bytes(evidence)
                ).hexdigest(),
                "ordered_reference_export_record_sha256": exports,
                "reference_export_records_sha256": hashlib.sha256(
                    _canonical_json_bytes(exports)
                ).hexdigest(),
            }
        },
        "forbidden_sources": forbidden_sources,
        "forbidden_source_registry": registry,
        "forbidden_source_registry_identity": registry_identity,
        "overlap_audit": {
            "passed": True,
            "tested_forbidden_sources": len(forbidden_sources),
            "collisions": 0,
        },
    }
    payload["generation_receipt"] = {
        "schema_version": 2,
        "status": "references_attached",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "generator_contract_sha256": generator_sha256,
        "generator_code_sha256": "9" * 64,
        "source_generation_receipt_sha256": "0" * 64,
        "forbidden_source_registry_identity": registry_identity,
        "reference_solver_contract_sha256": solver_sha256,
        "strict_verifier": verifier,
        "tasks": {
            "tsp": {
                "rows": 1,
                "relative_path": "tsp/eval/test.json",
                "file_sha256": file_sha256,
                "ordered_instance_sha256": semantic,
                "ordered_instance_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(semantic)
                ).hexdigest(),
                "ordered_semantic_instance_sha256": semantic,
                "semantic_instance_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(semantic)
                ).hexdigest(),
                "ordered_benchmark_row_sha256": rows,
                "benchmark_row_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(rows)
                ).hexdigest(),
                "ordered_reference_output_sha256": references,
                "reference_output_hashes_sha256": hashlib.sha256(
                    _canonical_json_bytes(references)
                ).hexdigest(),
                "strict_reference_objective_min": 1.0,
                "strict_reference_objective_max": 1.0,
                "reference_evidence_summary_sha256": "6" * 64,
                "ordered_reference_evidence_record_sha256": evidence,
                "reference_evidence_records_sha256": hashlib.sha256(
                    _canonical_json_bytes(evidence)
                ).hexdigest(),
                "ordered_reference_export_record_sha256": exports,
                "reference_export_records_sha256": hashlib.sha256(
                    _canonical_json_bytes(exports)
                ).hexdigest(),
            }
        },
        "contains_references": True,
        "publishable": True,
        "reference_evidence_schema_version": 1,
        "reference_export_summary_sha256": "8" * 64,
    }
    payload["generation_receipt"]["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload["generation_receipt"])
    ).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_confirmation_cli_requires_frozen_manifest_and_disables_best_run() -> None:
    parser = build_eval_parser()
    args = parser.parse_args(["--evaluation_purpose", "confirmation"])
    assert args.update_best_run is False
    with patch.object(
        sys,
        "argv",
        ["clasp-eval", "--evaluation_purpose", "confirmation"],
    ):
        with pytest.raises(ValueError, match="requires --benchmark_manifest"):
            parse_eval_args()


def test_confirmation_cli_rejects_selection_and_partial_dataset(tmp_path: Path) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "manifest.json", data_file)
    benchmark_payload = json.loads(manifest.read_text(encoding="utf-8"))
    campaign = _write_campaign(
        tmp_path / "campaign.json",
        benchmark_manifest_sha256=benchmark_payload["manifest_sha256"],
    )
    with patch.object(
        sys,
        "argv",
        [
            "clasp-eval",
            "--evaluation_purpose",
            "confirmation",
            "--benchmark_manifest",
            str(manifest),
        ],
    ):
        with pytest.raises(ValueError, match="requires --campaign_manifest"):
            parse_eval_args()
    common = [
        "clasp-eval",
        "--evaluation_purpose",
        "confirmation",
        "--benchmark_manifest",
        str(manifest),
        "--campaign_manifest",
        str(campaign),
        "--no-prefer_subset_files",
    ]
    with patch.object(sys, "argv", [*common, "--update_best_run"]):
        with pytest.raises(ValueError, match="development"):
            parse_eval_args()
    with patch.object(sys, "argv", [*common, "--max_eval_samples_per_task", "1"]):
        with pytest.raises(ValueError, match="every frozen row"):
            parse_eval_args()
    with patch.object(sys, "argv", [*common, "--no-resume"]):
        with pytest.raises(ValueError, match="requires --resume"):
            parse_eval_args()
    with patch.object(sys, "argv", common):
        with pytest.raises(ValueError, match="checkpoint_selection_receipt"):
            parse_eval_args()


def test_confirmation_checkpoint_receipt_binds_contract_and_model(tmp_path: Path) -> None:
    selection_registry, evaluation_files, historical_identity = _selection_registry(instances=100)
    development_contract = {
        "data_manifest_sha256": hashlib.sha256(_canonical_json_bytes(evaluation_files)).hexdigest(),
        "evaluation_data_files": evaluation_files,
        "forbidden_source_label": "historical-eval",
        "forbidden_source_identity_sha256": historical_identity,
        "protocol_fingerprint": "d" * 64,
        "strict_verifier_manifest_sha256": "f" * 64,
        "split": "eval",
        "eval_start_index": 0,
        "max_eval_samples_per_task": 100,
        "eval_samples_per_task": {},
        "size_bucket": "all",
        "min_instance_size": 0,
        "max_instance_size": 0,
        "prefer_subset_files": False,
        "expected_instances": 100,
    }
    development_contract["development_evaluation_sha256"] = hashlib.sha256(
        _canonical_json_bytes(development_contract)
    ).hexdigest()
    checkpoint_fidelity = _checkpoint_fidelity_identity()
    selected = {
        "task": "tsp",
        "method": "clasp_canonical",
        "train_seed": 42,
        "checkpoint_step": 400,
        "pass_at_k": 0.9,
        "feasible_instances": 90,
        "conditional_gap_mean": 0.1,
        "conditional_gap_n": 90,
        "instances": 100,
        "candidate_budget": 8,
        "grpo_beta": 0.0,
        "fixed_kl_reference": None,
        "exact_peft_kl_runtime": None,
        "exact_peft_kl_post_step": None,
        "checkpoint_artifact_sha256": "a" * 64,
        "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
        "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
        "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity["receipt_file_sha256"],
        "run_contract_sha256": "b" * 64,
        "evaluation_fingerprint": "c" * 64,
        "protocol_fingerprint": "d" * 64,
        "development_evaluation_sha256": development_contract["development_evaluation_sha256"],
        "data_manifest_sha256": development_contract["data_manifest_sha256"],
        "strict_verifier_manifest_sha256": "f" * 64,
    }
    selection_rule = [
        "maximize_pass_at_k",
        "retain_within_0.01_absolute_of_best_pass_at_k",
        "minimize_conditional_verified_gap",
        "tie_break_earlier_checkpoint_step",
    ]
    frozen_policy = {
        "schema_version": 1,
        "campaign_id": "test",
        "stage_id": "selection",
        "status": "frozen_before_development_results",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "evidence_scope": "development_only",
        "expected_tasks": ["tsp"],
        "expected_methods": ["clasp_canonical"],
        "train_seeds": [42],
        "checkpoint_steps": [400],
        "candidate_budget": 8,
        "forbidden_source_registry": selection_registry,
        "development_evaluation_by_task": {"tsp": development_contract},
        "require_complete_checkpoint_grid": True,
        "pass_at_k_absolute_band": 0.01,
        "selection_rule": selection_rule,
    }
    frozen_policy["policy_sha256"] = hashlib.sha256(
        _canonical_json_bytes(frozen_policy)
    ).hexdigest()
    payload = {
        "schema_version": 2,
        "selector": "clasp_development_checkpoint_selection",
        "generated_at_utc": "2026-08-14T01:00:00Z",
        "decision": "SELECTED",
        "evidence_scope": "development_only",
        "confirmation_evidence_used": False,
        "campaign_id": "test",
        "stage_id": "selection",
        "policy": {
            "path": "/frozen/selection-policy.json",
            "file_sha256": "1" * 64,
            "policy_sha256": frozen_policy["policy_sha256"],
            "frozen_at_utc": frozen_policy["frozen_at_utc"],
            "selection_rule": selection_rule,
            "pass_at_k_absolute_band": 0.01,
            "frozen_policy": frozen_policy,
        },
        "evidence": {
            "path": "/frozen/checkpoint-evidence.json",
            "size_bytes": 1,
            "file_sha256": "3" * 64,
            "evidence_sha256": "4" * 64,
            "record_count": 1,
        },
        "expected_group_count": 1,
        "selected_group_count": 1,
        "selections": {
            "tsp/clasp_canonical/seed42": {
                "status": "PASS",
                "selected": selected,
                "selection_audit": {
                    "best_pass_at_k": 0.9,
                    "eligibility_threshold": 0.89,
                    "eligible_checkpoint_steps": [400],
                    "candidates": [
                        {
                            "checkpoint_step": 400,
                            "candidate_budget": 8,
                            "instances": 100,
                            "feasible_instances": 90,
                            "pass_at_k": 0.9,
                            "conditional_gap_n": 90,
                            "conditional_gap_mean": 0.1,
                            "grpo_beta": 0.0,
                            "fixed_kl_reference": None,
                            "exact_peft_kl_runtime": None,
                            "exact_peft_kl_post_step": None,
                            "evaluation_fingerprint": "c" * 64,
                            "protocol_fingerprint": "d" * 64,
                            "development_evaluation_sha256": development_contract[
                                "development_evaluation_sha256"
                            ],
                            "checkpoint_artifact_sha256": "a" * 64,
                            "checkpoint_fidelity_identity_sha256": checkpoint_fidelity[
                                "identity_sha256"
                            ],
                            "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity[
                                "receipt_sha256"
                            ],
                            "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity[
                                "receipt_file_sha256"
                            ],
                            "run_contract_sha256": "b" * 64,
                            "data_manifest_sha256": development_contract["data_manifest_sha256"],
                            "strict_verifier_manifest_sha256": "f" * 64,
                            "disposition": "selected",
                        }
                    ],
                },
                "errors": [],
            }
        },
        "errors": [],
        "side_effects": {
            "checkpoints_modified": False,
            "evaluator_state_modified": False,
            "scheduler_actions": [],
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    identity, loaded = _checkpoint_selection_receipt_identity(str(path))
    assert identity["receipt_sha256"] == payload["receipt_sha256"]
    binding = _validate_confirmation_checkpoint_selection(
        receipt=loaded,
        training_run_contract={
            "task": "tsp",
            "method": "clasp_canonical",
            "seeds": {"train": 42},
            "training_config": {"beta": 0.0},
            "fixed_kl_reference": None,
        },
        training_run_contract_sha256="b" * 64,
        checkpoint_path=str(tmp_path / "run-seed42" / "checkpoint-400"),
        checkpoint_artifact=_checkpoint_artifact(checkpoint_fidelity),
        checkpoint_fidelity=checkpoint_fidelity,
        exact_peft_kl_runtime=None,
        exact_peft_kl_post_step=None,
    )
    assert binding["checkpoint_step"] == 400
    unselected_fidelity = _checkpoint_fidelity_identity("unselected")
    with pytest.raises(RuntimeError, match="fidelity does not match"):
        _validate_confirmation_checkpoint_selection(
            receipt=loaded,
            training_run_contract={
                "task": "tsp",
                "method": "clasp_canonical",
                "seeds": {"train": 42},
                "training_config": {"beta": 0.0},
                "fixed_kl_reference": None,
            },
            training_run_contract_sha256="b" * 64,
            checkpoint_path=str(tmp_path / "run-seed42" / "checkpoint-400"),
            checkpoint_artifact=_checkpoint_artifact(unselected_fidelity),
            checkpoint_fidelity=unselected_fidelity,
            exact_peft_kl_runtime=None,
            exact_peft_kl_post_step=None,
        )

    campaign_path = _write_campaign(
        tmp_path / "campaign-bound.json",
        benchmark_manifest_sha256="9" * 64,
        selection_identity=identity,
        selection_payload=loaded,
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign.pop("campaign_manifest_sha256")
    target = next(
        entry
        for entry in campaign["runs"]
        if entry["method"] == "clasp_canonical"
        and entry["training_seed"] == 42
        and entry["task"] == "tsp"
    )
    target.update(
        {
            "training_run_contract_sha256": "b" * 64,
            "checkpoint_step": 400,
            "checkpoint_artifact_sha256": "a" * 64,
            "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
            "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
            "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity["receipt_file_sha256"],
            "grpo_beta": 0.0,
            "fixed_kl_reference": None,
            "exact_peft_kl_runtime": None,
            "exact_peft_kl_post_step": None,
            "checkpoint_selection_receipt_sha256": payload["receipt_sha256"],
        }
    )
    target["confirmation_attempt_id"] = _confirmation_attempt_id(campaign, target)
    campaign["campaign_manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(campaign)
    ).hexdigest()
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    campaign_payload, campaign_identity = _load_campaign_manifest_snapshot(str(campaign_path))
    campaign_binding = _validate_confirmation_campaign_binding(
        manifest=campaign_payload,
        manifest_identity=campaign_identity,
        benchmark_identity={"manifest_sha256": "9" * 64},
        selection_receipt_identity=identity,
        selection_receipt_payload=loaded,
        training_run_contract={
            "task": "tsp",
            "method": "clasp_canonical",
            "seeds": {"train": 42},
            "training_config": {"beta": 0.0},
            "fixed_kl_reference": None,
        },
        training_run_contract_sha256="b" * 64,
        checkpoint_selection_binding=binding,
        decode_seed=42,
    )
    assert campaign_binding["confirmation_attempt_id"] == target["confirmation_attempt_id"]
    first_results = tmp_path / "inference-a" / "grpo_z" / "checkpoint-400"
    first_config = first_results / "run_config.json"
    first_details = first_results / "run_details.jsonl"
    claim_path, claim_handle = _claim_confirmation_attempt(
        attempt_ledger_root=campaign_payload["confirmation_attempt_ledger"]["canonical_root"],
        campaign_binding=campaign_binding,
        experiment_fingerprint="7" * 64,
        results_dir=str(first_results),
        required_resume_artifacts=(str(first_config), str(first_details)),
    )
    claim_handle.close()
    assert Path(claim_path).is_file()
    claim_payload = json.loads(Path(claim_path).read_text(encoding="utf-8"))
    assert claim_payload["schema_version"] == 2
    assert claim_payload["protocol"] == "clasp-confirmation-attempt-v2"
    first_config.write_text("{}", encoding="utf-8")
    first_details.write_text("", encoding="utf-8")
    _same_path, resume_handle = _claim_confirmation_attempt(
        attempt_ledger_root=campaign_payload["confirmation_attempt_ledger"]["canonical_root"],
        campaign_binding=campaign_binding,
        experiment_fingerprint="7" * 64,
        results_dir=str(first_results),
        required_resume_artifacts=(str(first_config), str(first_details)),
    )
    resume_handle.close()
    first_details.unlink()
    with pytest.raises(RuntimeError, match="removed or never committed"):
        _claim_confirmation_attempt(
            attempt_ledger_root=campaign_payload["confirmation_attempt_ledger"]["canonical_root"],
            campaign_binding=campaign_binding,
            experiment_fingerprint="7" * 64,
            results_dir=str(first_results),
            required_resume_artifacts=(str(first_config), str(first_details)),
        )
    first_details.write_text("", encoding="utf-8")
    copied_campaign = tmp_path / "copied" / "campaign-bound.json"
    copied_campaign.parent.mkdir()
    copied_campaign.write_bytes(campaign_path.read_bytes())
    copied_payload, _copied_identity = _load_campaign_manifest_snapshot(str(copied_campaign))
    assert (
        copied_payload["confirmation_attempt_ledger"]
        == campaign_payload["confirmation_attempt_ledger"]
    )
    with pytest.raises(RuntimeError, match="different output"):
        _claim_confirmation_attempt(
            attempt_ledger_root=copied_payload["confirmation_attempt_ledger"]["canonical_root"],
            campaign_binding=campaign_binding,
            experiment_fingerprint="7" * 64,
            results_dir=str(tmp_path / "inference-b" / "grpo_z" / "checkpoint-400"),
            required_resume_artifacts=(
                str(tmp_path / "inference-b" / "run_config.json"),
                str(tmp_path / "inference-b" / "run_details.jsonl"),
            ),
        )

    wrong = json.loads(json.dumps(payload))
    wrong_policy = wrong["policy"]["frozen_policy"]
    wrong_policy.pop("policy_sha256")
    wrong_policy["checkpoint_steps"] = [200, 400]
    wrong_policy["policy_sha256"] = hashlib.sha256(_canonical_json_bytes(wrong_policy)).hexdigest()
    wrong["policy"]["policy_sha256"] = wrong_policy["policy_sha256"]
    wrong["evidence"]["record_count"] = 2
    group = wrong["selections"]["tsp/clasp_canonical/seed42"]
    group["selection_audit"] = {
        "best_pass_at_k": 0.9,
        "eligibility_threshold": 0.89,
        "eligible_checkpoint_steps": [200, 400],
        "candidates": [
            {
                "checkpoint_step": 200,
                "candidate_budget": 8,
                "instances": 100,
                "feasible_instances": 90,
                "pass_at_k": 0.9,
                "conditional_gap_n": 90,
                "conditional_gap_mean": 0.05,
                "grpo_beta": 0.0,
                "fixed_kl_reference": None,
                "exact_peft_kl_runtime": None,
                "exact_peft_kl_post_step": None,
                "evaluation_fingerprint": "9" * 64,
                "protocol_fingerprint": "d" * 64,
                "development_evaluation_sha256": development_contract[
                    "development_evaluation_sha256"
                ],
                "checkpoint_artifact_sha256": "8" * 64,
                "checkpoint_fidelity_identity_sha256": "7" * 64,
                "checkpoint_fidelity_receipt_sha256": "6" * 64,
                "checkpoint_fidelity_receipt_file_sha256": "5" * 64,
                "run_contract_sha256": "b" * 64,
                "data_manifest_sha256": development_contract["data_manifest_sha256"],
                "strict_verifier_manifest_sha256": "f" * 64,
                "disposition": "selected",
            },
            {
                "checkpoint_step": 400,
                "candidate_budget": 8,
                "instances": 100,
                "feasible_instances": 90,
                "pass_at_k": 0.9,
                "conditional_gap_n": 90,
                "conditional_gap_mean": 0.1,
                "grpo_beta": 0.0,
                "fixed_kl_reference": None,
                "exact_peft_kl_runtime": None,
                "exact_peft_kl_post_step": None,
                "evaluation_fingerprint": "c" * 64,
                "protocol_fingerprint": "d" * 64,
                "development_evaluation_sha256": development_contract[
                    "development_evaluation_sha256"
                ],
                "checkpoint_artifact_sha256": "a" * 64,
                "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
                "checkpoint_fidelity_receipt_sha256": checkpoint_fidelity["receipt_sha256"],
                "checkpoint_fidelity_receipt_file_sha256": checkpoint_fidelity[
                    "receipt_file_sha256"
                ],
                "run_contract_sha256": "b" * 64,
                "data_manifest_sha256": development_contract["data_manifest_sha256"],
                "strict_verifier_manifest_sha256": "f" * 64,
                "disposition": "eligible_but_higher_gap_or_later_exact_tie",
            },
        ],
    }
    wrong.pop("receipt_sha256")
    wrong["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(wrong)).hexdigest()
    wrong_path = tmp_path / "wrong-selection.json"
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="violates frozen rule"):
        _checkpoint_selection_receipt_identity(str(wrong_path))

    duplicate_artifact = json.loads(json.dumps(wrong))
    duplicate_artifact["selections"]["tsp/clasp_canonical/seed42"]["selection_audit"]["candidates"][0][
        "checkpoint_artifact_sha256"
    ] = "a" * 64
    duplicate_artifact.pop("receipt_sha256")
    duplicate_artifact["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(duplicate_artifact)
    ).hexdigest()
    duplicate_path = tmp_path / "duplicate-checkpoint.json"
    duplicate_path.write_text(json.dumps(duplicate_artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates a checkpoint artifact"):
        _checkpoint_selection_receipt_identity(str(duplicate_path))

    mixed_contract = json.loads(json.dumps(wrong))
    mixed_contract["selections"]["tsp/clasp_canonical/seed42"]["selection_audit"]["candidates"][0][
        "run_contract_sha256"
    ] = "7" * 64
    mixed_contract.pop("receipt_sha256")
    mixed_contract["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(mixed_contract)
    ).hexdigest()
    mixed_contract_path = tmp_path / "mixed-contract.json"
    mixed_contract_path.write_text(json.dumps(mixed_contract), encoding="utf-8")
    with pytest.raises(ValueError, match="mixes training run contracts"):
        _checkpoint_selection_receipt_identity(str(mixed_contract_path))

    empty_audit = json.loads(json.dumps(payload))
    empty_audit["selections"]["tsp/clasp_canonical/seed42"]["selection_audit"] = {}
    empty_audit.pop("receipt_sha256")
    empty_audit["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(empty_audit)).hexdigest()
    empty_path = tmp_path / "empty-audit.json"
    empty_path.write_text(json.dumps(empty_audit), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid audit schema"):
        _checkpoint_selection_receipt_identity(str(empty_path))

    wrong_campaign = json.loads(json.dumps(payload))
    wrong_campaign["campaign_id"] = "post-hoc-campaign"
    wrong_campaign.pop("receipt_sha256")
    wrong_campaign["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(wrong_campaign)
    ).hexdigest()
    wrong_campaign_path = tmp_path / "wrong-campaign.json"
    wrong_campaign_path.write_text(json.dumps(wrong_campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign/stage"):
        _checkpoint_selection_receipt_identity(str(wrong_campaign_path))

    with pytest.raises(RuntimeError, match="bytes do not match"):
        _validate_confirmation_checkpoint_selection(
            receipt=loaded,
            training_run_contract={
                "task": "tsp",
                "method": "clasp_canonical",
                "seeds": {"train": 42},
                "training_config": {"beta": 0.0},
                "fixed_kl_reference": None,
            },
            training_run_contract_sha256="b" * 64,
            checkpoint_path=str(tmp_path / "run-seed42" / "checkpoint-400"),
            checkpoint_artifact={"manifest_sha256": "d" * 64},
            checkpoint_fidelity=checkpoint_fidelity,
            exact_peft_kl_runtime=None,
            exact_peft_kl_post_step=None,
        )


def test_positive_beta_contract_live_verifies_exact_fixed_kl_reference(tmp_path: Path) -> None:
    file_record = {"path": "artifact.bin", "size_bytes": 1, "sha256": "a" * 64}
    aggregate = hashlib.sha256()
    aggregate.update(b"artifact.bin\0")
    aggregate.update(b"1\0")
    aggregate.update(("a" * 64).encode("ascii"))
    aggregate.update(b"\n")
    tree = {
        "canonical_path": "/immutable/artifact",
        "sha256": aggregate.hexdigest(),
        "file_count": 1,
        "size_bytes": 1,
        "files": [file_record],
    }
    source_fidelity = _checkpoint_fidelity_identity("source")
    fixed_kl = {
        "identity_sha256": "b" * 64,
        "source_checkpoint_fidelity": source_fidelity,
    }
    model_path = str((tmp_path / "specialist-adapter").resolve())
    receipt_path = str((Path(model_path) / "checkpoint_fidelity.json").resolve())
    contract = {
        "schema_version": 2,
        "stage": "grpo",
        "method": "clasp_canonical",
        "task": "tsp",
        "seeds": {"train": 42, "data": 42, "sft": 42},
        "fixed_horizon": {"max_steps": 400, "world_size": 4},
        "data_lineage": {"registry_sha256": "c" * 64},
        "model_initialization_registry": {
            "specialist_sft_by_task": {
                "tsp": {"checkpoint_fidelity": source_fidelity},
            }
        },
        "fixed_kl_reference": fixed_kl,
        "fixed_null_receipt": None,
        "fingerprints": {name: tree for name in ("initial_model", "base_model", "data", "source")},
        "generation": {"num_generations": 8},
        "training_config": {
            "beta": 0.04,
            "checkpoint_fidelity": "publication_exact_required",
        },
        "tracked_environment": {
            "CLASP_PUBLICATION_FIDELITY": "1",
            "CLASP_KL_REFERENCE_MODEL_PATH": model_path,
            "CLASP_KL_REFERENCE_RECEIPT": receipt_path,
        },
    }
    contract["fingerprints"]["initial_model"]["canonical_path"] = model_path
    with (
        patch(
            "clasp.entrypoints.evaluate.validate_fixed_kl_reference_identity",
            return_value=fixed_kl,
        ),
        patch(
            "clasp.entrypoints.evaluate.fixed_kl_reference_identity",
            return_value=fixed_kl,
        ) as live_verify,
    ):
        _validate_training_run_contract(contract)
    live_verify.assert_called_once_with(model_path, receipt_path)

    changed = {**fixed_kl, "identity_sha256": "d" * 64}
    with (
        patch(
            "clasp.entrypoints.evaluate.validate_fixed_kl_reference_identity",
            return_value=fixed_kl,
        ),
        patch(
            "clasp.entrypoints.evaluate.fixed_kl_reference_identity",
            return_value=changed,
        ),
        pytest.raises(RuntimeError, match="live fixed-KL reference differs"),
    ):
        _validate_training_run_contract(contract)


def test_confirmation_manifest_and_dataset_are_content_bound(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_file = data_root / "tsp" / "eval" / "test.json"
    data_file.parent.mkdir(parents=True)
    data_file.write_text('[{"instance_json":[[0,0],[1,0],[0,1]]}]', encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)

    identity = _benchmark_manifest_identity(str(manifest_path))
    assert identity["status"] == "frozen_before_inference"
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copytree(data_root, relocated / "data")
    relocated_manifest = relocated / manifest_path.name
    relocated_manifest.write_bytes(manifest_path.read_bytes())
    assert _benchmark_manifest_identity(str(relocated_manifest)) == identity
    data_identity = _dataset_content_identity(
        data_root=str(data_root),
        task_json_files={"tsp": [str(data_file)]},
        split="eval",
    )
    _validate_confirmation_dataset(
        manifest_path=str(manifest_path),
        tasks=["tsp"],
        data_identity=data_identity,
    )

    data_file.write_text("[]", encoding="utf-8")
    changed_identity = _dataset_content_identity(
        data_root=str(data_root),
        task_json_files={"tsp": [str(data_file)]},
        split="eval",
    )
    with pytest.raises(ValueError, match="do not match frozen benchmark"):
        _validate_confirmation_dataset(
            manifest_path=str(manifest_path),
            tasks=["tsp"],
            data_identity=changed_identity,
        )


def test_confirmation_manifest_rejects_resealed_evidence_and_contract_tampering(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)
    manifest = json.loads(manifest_path.read_text())

    manifest["test_sets"]["tsp"]["reference_evidence_records_sha256"] = "0" * 64
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="reference_evidence_records_sha256"):
        _benchmark_manifest_identity(str(manifest_path))

    manifest_path = _write_manifest(tmp_path / "manifest2.json", data_file)
    manifest = json.loads(manifest_path.read_text())
    manifest["contracts"]["reference_solver"]["tasks"]["tsp"]["accepted_statuses"] = ["FEASIBLE"]
    solver = manifest["contracts"]["reference_solver"]
    solver.pop("contract_sha256")
    solver["contract_sha256"] = hashlib.sha256(_canonical_json_bytes(solver)).hexdigest()
    manifest["contracts"]["reference_solver_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(solver)
    ).hexdigest()
    manifest["generation_receipt"]["reference_solver_contract_sha256"] = manifest["contracts"][
        "reference_solver_contract_sha256"
    ]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid accepted status"):
        _benchmark_manifest_identity(str(manifest_path))


@pytest.mark.parametrize("case", ["omitted", "partial", "duplicate"])
def test_confirmation_manifest_requires_complete_unique_leakage_sources(
    tmp_path: Path, case: str
) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)
    manifest = json.loads(manifest_path.read_text())
    sources = manifest["forbidden_sources"]
    if case == "omitted":
        manifest["forbidden_sources"] = []
    elif case == "partial":
        manifest["forbidden_sources"] = sources[:2]
    else:
        manifest["forbidden_sources"] = [*sources, json.loads(json.dumps(sources[0]))]
    manifest["overlap_audit"]["tested_forbidden_sources"] = len(manifest["forbidden_sources"])
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    match = "duplicate forbidden source" if case == "duplicate" else "do not exactly match"
    with pytest.raises(ValueError, match=match):
        _benchmark_manifest_identity(str(manifest_path))


def test_confirmation_manifest_self_hash_fails_closed(tmp_path: Path) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "edited_after_freeze"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        _benchmark_manifest_identity(str(manifest_path))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("publishable", "not publishable"),
        ("ordered_rows", "ordered_benchmark_row_sha256 does not match"),
    ],
)
def test_confirmation_manifest_revalidates_full_generation_receipt(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest["generation_receipt"]
    if tamper == "publishable":
        receipt["publishable"] = False
    else:
        receipt["tasks"]["tsp"]["ordered_benchmark_row_sha256"] = ["f" * 64]
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _benchmark_manifest_identity(str(manifest_path))


def test_confirmation_rejects_stale_v2_manifest(tmp_path: Path) -> None:
    data_file = tmp_path / "test.json"
    data_file.write_text("[]", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path / "manifest.json", data_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 3"):
        _benchmark_manifest_identity(str(manifest_path))
