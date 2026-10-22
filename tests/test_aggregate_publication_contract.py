from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from clasp.baselines.foarl_f638_comparison import (
    CLASP_FOARL_MATCHED_B9,
    CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
    FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST,
    SHARED_ENVIRONMENT_CONTRACT,
    SHARED_GENERATION_CONTRACT,
    SHARED_TRAINING_CONTRACT,
)
from clasp.models import publication_fidelity

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_sota_publication", ROOT / "scripts" / "aggregate_sota.py"
)
assert SPEC is not None and SPEC.loader is not None
aggregate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate
SPEC.loader.exec_module(aggregate)


def _digest(value: object) -> str:
    return hashlib.sha256(aggregate._canonical_json_bytes(value)).hexdigest()


def _seal(payload: dict, field: str) -> dict:
    sealed = deepcopy(payload)
    sealed.pop(field, None)
    sealed[field] = _digest(sealed)
    return sealed


def _fidelity_identity(
    label: str,
    *,
    receipt_bytes: bytes,
    adapter_config_bytes: bytes,
    adapter_weights_bytes: bytes,
) -> dict:
    identity = {
        "schema_version": 1,
        "protocol": "clasp-checkpoint-fidelity-identity-v1",
        "status": "publication_exact",
        "receipt_sha256": _digest(f"{label}-semantic-receipt"),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(adapter_config_bytes).hexdigest(),
        "adapter_weights_sha256": hashlib.sha256(adapter_weights_bytes).hexdigest(),
        "trainable_tensor_inventory_sha256": _digest(f"{label}-tensor-inventory"),
        "publication_invariant_sha256": _digest(f"{label}-publication-invariant"),
    }
    identity["identity_sha256"] = _digest(identity)
    return identity


def _reload_identity(
    label: str,
    checkpoint_fidelity: dict,
    *,
    receipt_bytes: bytes | None = None,
) -> dict:
    encoded = receipt_bytes or f"{label}-reload-receipt\n".encode()
    identity = {
        "schema_version": 2,
        "protocol": "clasp-sft-post-save-reload-equivalence-v2",
        "status": "exact_post_save_reload_verified",
        "receipt_sha256": _digest(f"{label}-semantic-reload-receipt"),
        "receipt_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "checkpoint_fidelity_identity_sha256": checkpoint_fidelity["identity_sha256"],
        "serialized_trainable_closure_sha256": _digest(f"{label}-trainable-closure"),
        "runtime_pair_sha256": _digest(f"{label}-runtime-pair"),
        "functional_equivalence_sha256": _digest(f"{label}-functional-equivalence"),
        "producer_sha256": _digest(f"{label}-reload-producer"),
    }
    identity["identity_sha256"] = _digest(identity)
    return identity


def _fixed_kl_identity(source_checkpoint_fidelity: dict) -> dict:
    identity = {
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
        "source_checkpoint_fidelity": deepcopy(source_checkpoint_fidelity),
    }
    for index, field in enumerate(
        publication_fidelity.FIXED_KL_IDENTITY_KEYS - set(identity) - {"identity_sha256"},
        start=1,
    ):
        identity[field] = f"{index % 16:x}" * 64
    identity["identity_sha256"] = _digest(identity)
    return identity


def _attach_forbidden_source_registry(manifest: dict) -> None:
    manifest["forbidden_sources"].sort(key=lambda row: (row["task"], row["label"]))
    sources = []
    for source in manifest["forbidden_sources"]:
        projection = {
            key: deepcopy(source[key])
            for key in (
                "task",
                "label",
                "files",
                "file_manifest_sha256",
                "rows",
                "unique_instance_hashes",
            )
        }
        projection["source_identity_sha256"] = _digest(projection)
        source["source_identity_sha256"] = projection["source_identity_sha256"]
        sources.append(projection)
    sources.sort(key=lambda row: (row["task"], row["label"]))
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": sorted(manifest["tasks"]),
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": _digest(sources),
    }
    registry["registry_sha256"] = _digest(registry)
    manifest["benchmark_mode"] = "publication"
    manifest["forbidden_source_registry"] = registry
    manifest["forbidden_source_registry_identity"] = aggregate.forbidden_source_registry_identity(
        registry
    )
    prior_receipt = manifest["generation_receipt"]
    receipt_tasks = {}
    for task, task_manifest in manifest["test_sets"].items():
        receipt_tasks[task] = {
            "rows": task_manifest["rows"],
            "relative_path": task_manifest["files"][0]["relative_path"],
            "file_sha256": task_manifest["files"][0]["sha256"],
            "ordered_instance_sha256": task_manifest["ordered_instance_sha256"],
            "ordered_instance_hashes_sha256": _digest(task_manifest["ordered_instance_sha256"]),
            "ordered_semantic_instance_sha256": task_manifest["ordered_semantic_instance_sha256"],
            "semantic_instance_hashes_sha256": _digest(
                task_manifest["ordered_semantic_instance_sha256"]
            ),
            "ordered_benchmark_row_sha256": task_manifest["ordered_benchmark_row_sha256"],
            "benchmark_row_hashes_sha256": _digest(task_manifest["ordered_benchmark_row_sha256"]),
            "ordered_reference_output_sha256": task_manifest["ordered_reference_output_sha256"],
            "reference_output_hashes_sha256": _digest(
                task_manifest["ordered_reference_output_sha256"]
            ),
            "strict_reference_objective_min": task_manifest["reference_audit"]["objective_min"],
            "strict_reference_objective_max": task_manifest["reference_audit"]["objective_max"],
            "reference_evidence_summary_sha256": task_manifest["reference_evidence_summary_sha256"],
            "ordered_reference_evidence_record_sha256": task_manifest[
                "ordered_reference_evidence_record_sha256"
            ],
            "reference_evidence_records_sha256": task_manifest["reference_evidence_records_sha256"],
            "ordered_reference_export_record_sha256": task_manifest[
                "ordered_reference_export_record_sha256"
            ],
            "reference_export_records_sha256": task_manifest["reference_export_records_sha256"],
        }
    receipt = {
        "schema_version": 2,
        "status": "references_attached",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "generator_contract_sha256": manifest["contracts"]["generator_contract_sha256"],
        "generator_code_sha256": prior_receipt["generator_code_sha256"],
        "source_generation_receipt_sha256": "6" * 64,
        "forbidden_source_registry_identity": deepcopy(
            manifest["forbidden_source_registry_identity"]
        ),
        "reference_solver_contract_sha256": manifest["contracts"][
            "reference_solver_contract_sha256"
        ],
        "strict_verifier": deepcopy(manifest["strict_verifier"]),
        "tasks": receipt_tasks,
        "contains_references": True,
        "publishable": True,
        "reference_evidence_schema_version": 1,
        "reference_export_summary_sha256": prior_receipt["reference_export_summary_sha256"],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    manifest["generation_receipt"] = receipt


def _valid_tsp_benchmark_manifest(verifier: dict) -> dict:
    verifier_sha = _digest(verifier)
    generator = {"name": "test-generator", "schema_version": 1}
    solver = {
        "schema_version": 2,
        "status": "approved_for_confirmation",
        "reference_evidence_runner": {
            "schema_version": 1,
            "protocol": "clasp-iid-reference-evidence-v1",
            "expected_instances_per_task": 1,
            "strict_verifier_identity_sha256": verifier_sha,
        },
        "tasks": {
            "tsp": {
                "accepted_statuses": ["BEST_KNOWN_HEURISTIC_VERIFIED"],
                "parameters": {"seed": 7},
                "evidence_runner": {
                    "approval": "approved",
                    "adapter": "fixture_tsp_v1",
                    "accepted_native_statuses": ["HEURISTIC_COMPLETE"],
                    "objective_semantics": "clasp_raw_euclidean_float64",
                    "reference_quality": "best_known_heuristic",
                    "seed_parameter": "seed",
                    "solver_identity": {
                        "backend_name": "fixture-tsp",
                        "native_version": "1",
                        "artifact_kind": "test_fixture",
                        "artifact_path": "/frozen/fixture-tsp",
                        "artifact_sha256": "a" * 64,
                    },
                },
            }
        },
    }
    solver["contract_sha256"] = _digest(solver)
    files = [
        {
            "relative_path": "tsp/eval/test.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "rows": 1,
        }
    ]
    semantic = ["c" * 64]
    rows = ["d" * 64]
    references = ["e" * 64]
    evidence = ["f" * 64]
    exports = ["1" * 64]
    forbidden = [
        {
            "task": "tsp",
            "label": label,
            "source_locator": f"forbidden/{label}.json",
            "files": files,
            "file_manifest_sha256": _digest(files),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        for label in ("sft-train", "rl-train", "historical-eval")
    ]
    manifest = {
        "schema_version": 3,
        "status": "frozen_before_inference",
        "minimum_instances_per_task": 1,
        "tasks": ["tsp"],
        "overlap_audit": {
            "passed": True,
            "collisions": 0,
            "tested_forbidden_sources": len(forbidden),
        },
        "forbidden_sources": forbidden,
        "contracts": {
            "generator": generator,
            "generator_contract_sha256": _digest(generator),
            "reference_solver": solver,
            "reference_solver_contract_sha256": _digest(solver),
        },
        "generation_receipt": {
            "status": "references_attached",
            "receipt_sha256": "2" * 64,
            "generator_code_sha256": "3" * 64,
            "reference_solver_contract_sha256": _digest(solver),
            "strict_verifier": verifier,
            "reference_evidence_schema_version": 1,
            "reference_export_summary_sha256": "4" * 64,
        },
        "strict_verifier": verifier,
        "test_sets": {
            "tsp": {
                "rows": 1,
                "source_locator": "source/tsp/eval/test.json",
                "files": files,
                "file_manifest_sha256": _digest(files),
                "ordered_instance_sha256": semantic,
                "ordered_semantic_instance_sha256": semantic,
                "instance_hashes_sha256": _digest(semantic),
                "ordered_benchmark_row_sha256": rows,
                "row_hashes_sha256": _digest(rows),
                "ordered_reference_output_sha256": references,
                "reference_hashes_sha256": _digest(references),
                "reference_audit": {
                    "strictly_feasible": 1,
                    "zero_gap": 1,
                    "objective_min": 10.0,
                    "objective_max": 10.0,
                },
                "reference_evidence_schema_version": 1,
                "reference_evidence_summary_sha256": "5" * 64,
                "ordered_reference_evidence_record_sha256": evidence,
                "reference_evidence_records_sha256": _digest(evidence),
                "ordered_reference_export_record_sha256": exports,
                "reference_export_records_sha256": _digest(exports),
            }
        },
    }
    _attach_forbidden_source_registry(manifest)
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def _native_fixture(tmp_path: Path) -> tuple[aggregate.RunSpec, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry, _registry_hash = aggregate._load_lane_contract_registry(
        str(ROOT / "experiments" / "sota_20260814" / "lane_contracts.json")
    )
    native = registry["native_contract"]
    checkpoint_bytes = b"checkpoint-3200"
    adapter_config_bytes = b"{}"
    checkpoint_receipt_bytes = b"fixture checkpoint fidelity receipt\n"
    checkpoint_fidelity = _fidelity_identity(
        "checkpoint-3200",
        receipt_bytes=checkpoint_receipt_bytes,
        adapter_config_bytes=adapter_config_bytes,
        adapter_weights_bytes=checkpoint_bytes,
    )
    initial_weights_bytes = b"initial-specialist"
    initial_receipt_bytes = b"fixture initial fidelity receipt\n"
    initial_reload_receipt_bytes = b"fixture initial reload equivalence receipt\n"
    initial_fidelity = _fidelity_identity(
        "initial-specialist",
        receipt_bytes=initial_receipt_bytes,
        adapter_config_bytes=adapter_config_bytes,
        adapter_weights_bytes=initial_weights_bytes,
    )

    def training_tree(name: str) -> dict:
        contents = {f"{name}.json": name.encode()}
        if name == "initial_model":
            contents.update(
                {
                    "adapter_config.json": adapter_config_bytes,
                    "adapter_model.safetensors": initial_weights_bytes,
                    "checkpoint_fidelity.json": initial_receipt_bytes,
                    "sft_reload_equivalence.json": initial_reload_receipt_bytes,
                }
            )
        files = [
            {
                "path": relative,
                "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for relative, value in sorted(contents.items())
        ]
        tree_digest = hashlib.sha256()
        for item in files:
            tree_digest.update(item["path"].encode())
            tree_digest.update(b"\0")
            tree_digest.update(str(item["size_bytes"]).encode("ascii"))
            tree_digest.update(b"\0")
            tree_digest.update(item["sha256"].encode("ascii"))
            tree_digest.update(b"\n")
        return {
            "canonical_path": f"/frozen/{name}",
            "sha256": tree_digest.hexdigest(),
            "file_count": len(files),
            "size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        }

    registered_sources = {}
    for source_label, name in (
        ("historical-eval", "historical"),
        ("rl-train", "data"),
        ("sft-train", "sft"),
    ):
        tree = training_tree(name)
        files = [
            {
                "relative_path": row["path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "rows": 1,
            }
            for row in tree["files"]
        ]
        source_base = {
            "task": "tsp",
            "label": source_label,
            "files": files,
            "file_manifest_sha256": _digest(files),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        registered_sources[source_label] = {
            "source_identity_sha256": _digest(source_base),
            "file_manifest_sha256": source_base["file_manifest_sha256"],
            "rows": 1,
            "unique_instance_hashes": 1,
            "files": files,
        }
    registry_identity = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "registry_sha256": "a" * 64,
        "source_tree_manifest_sha256": "b" * 64,
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "source_count": 3,
        "sources_by_task_label": {"tsp": registered_sources},
    }

    def lineage_binding(source_label: str, canonical_path: str) -> dict:
        registered = registered_sources[source_label]
        return {
            "task": "tsp",
            "label": source_label,
            "canonical_path": canonical_path,
            "registry_sha256": registry_identity["registry_sha256"],
            "source_tree_manifest_sha256": registry_identity["source_tree_manifest_sha256"],
            **deepcopy(registered),
        }

    contract = {
        "schema_version": 2,
        "stage": "grpo",
        "method": "clasp_canonical",
        "task": "tsp",
        "seeds": {"train": 42, "data": 42, "sft": 42},
        "fixed_horizon": {"max_steps": 3200, "world_size": 4},
        "data_lineage": {
            "forbidden_source_registry": registry_identity,
            "sft_train": lineage_binding("sft-train", "/frozen/sft"),
            "rl_train": lineage_binding("rl-train", "/frozen/data"),
        },
        "fixed_kl_reference": None,
        "fixed_null_receipt": None,
        "fingerprints": {
            name: training_tree(name) for name in ("initial_model", "base_model", "data", "source")
        },
        "generation": {"num_generations": 8, "temperature": 0.7},
        "training_config": {
            "learning_rate": 1e-6,
            "entrypoint": "clasp.entrypoints.train_grpo",
            "policy_credit_formula": "canonical_k8_leave_out_set_utility_v1",
            "component_controller": "frozen_zero",
            "set_controller": "signed_failure_at_k_ema",
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "clasp_fixed_null_set": False,
            "pd_group_scalarization": False,
            "pd_separate_z": False,
            "pd_dual_lr": 0.0,
            "pd_dual_lr_up": 0.0,
            "pd_dual_lr_down": 0.0,
            "clasp_bon_dual_lr": 0.05,
            "clasp_bon_lambda_init": 0.0,
            "clasp_canonical_recovery_scale": 1.0,
            "checkpoint_fidelity": "publication_exact_required",
            "beta": 0.0,
        },
        "tracked_environment": {
            "CLASP_BON_ALIGNED": "1",
            "CLASP_FEASIBILITY_GATED": "1",
            "CLASP_CANONICAL_SET": "1",
            "CLASP_FIXED_NULL_SET": "0",
            "CLASP_BON_DUAL_LR": "0.05",
            "CLASP_BON_LAMBDA_INIT": "0.0",
            "CLASP_CANONICAL_RECOVERY_SCALE": "1.0",
            "CLASP_PUBLICATION_FIDELITY": "1",
        },
    }
    contract["model_initialization_registry"] = {
        "schema_version": 2,
        "protocol": "clasp-model-initialization-registry-v2",
        "status": "frozen_before_training",
        "registry_sha256": "c" * 64,
        "model_family": "Qwen2.5-7B",
        "base_model_tree_sha256": contract["fingerprints"]["base_model"]["sha256"],
        "forbidden_source_registry_sha256": registry_identity["registry_sha256"],
        "forbidden_source_tree_manifest_sha256": registry_identity["source_tree_manifest_sha256"],
        "specialist_sft_by_task": {
            task: {
                "initial_model_tree_sha256": (
                    contract["fingerprints"]["initial_model"]["sha256"]
                    if task == "tsp"
                    else _digest(f"{task}-approved-sft")
                ),
                "sft_train_source_identity_sha256": (
                    registered_sources["sft-train"]["source_identity_sha256"]
                    if task == "tsp"
                    else _digest(f"{task}-sft-source")
                ),
                "sft_seed": 42,
                "matched_sft_contract_sha256": (aggregate.MATCHED_SFT_CONTRACT_SHA256),
                "checkpoint_fidelity": (
                    initial_fidelity
                    if task == "tsp"
                    else _fidelity_identity(
                        f"{task}-initial",
                        receipt_bytes=f"{task}-receipt".encode(),
                        adapter_config_bytes=adapter_config_bytes,
                        adapter_weights_bytes=f"{task}-weights".encode(),
                    )
                ),
                "reload_equivalence": _reload_identity(
                    f"{task}-initial",
                    initial_fidelity
                    if task == "tsp"
                    else _fidelity_identity(
                        f"{task}-initial",
                        receipt_bytes=f"{task}-receipt".encode(),
                        adapter_config_bytes=adapter_config_bytes,
                        adapter_weights_bytes=f"{task}-weights".encode(),
                    ),
                    receipt_bytes=(
                        initial_reload_receipt_bytes if task == "tsp" else None
                    ),
                ),
            }
            for task in aggregate.PUBLICATION_TASKS
        },
    }
    contract_sha = aggregate._training_contract_file_sha256(contract)
    run_dir = tmp_path / "training_run"
    checkpoint_dir = run_dir / "checkpoint-3200"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(checkpoint_bytes)
    (checkpoint_dir / "adapter_config.json").write_bytes(adapter_config_bytes)
    (checkpoint_dir / "checkpoint_fidelity.json").write_bytes(checkpoint_receipt_bytes)
    (run_dir / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    verifier = aggregate._current_strict_verifier_identity()
    data_files = [
        {
            "task": "tsp",
            "split": "eval",
            "path": "tsp/eval/test.json",
            "size": 3,
            "sha256": "3" * 64,
        }
    ]
    code_files = [{"path": "entrypoints/evaluate.py", "size": 20, "sha256": "2" * 64}]
    model_files = [
        {
            "path": name,
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for name, value in sorted(
            {
                "adapter_config.json": adapter_config_bytes,
                "adapter_model.safetensors": checkpoint_bytes,
                "checkpoint_fidelity.json": checkpoint_receipt_bytes,
            }.items()
        )
    ]
    model_artifact = {
        "kind": "local_directory",
        "basename": "checkpoint-3200",
        "manifest_sha256": _digest(model_files),
        "files": model_files,
    }
    provenance = {
        "schema_version": 1,
        "model": {
            "checkpoint_artifact": model_artifact,
            "checkpoint_fidelity": checkpoint_fidelity,
            "exact_peft_kl_runtime": None,
            "exact_peft_kl_post_step": None,
            "base_model_reference": "Qwen/Qwen2.5-7B",
            "base_model_artifact": {"kind": "external_reference", "reference": "Qwen/Qwen2.5-7B"},
            "loaded_revision": "f" * 40,
            "loaded_config": {"class": "Qwen2Config", "sha256": "5" * 64},
            "tokenizer_artifact": model_artifact,
            "loaded_tokenizer": {"class": "Qwen2Tokenizer"},
        },
        "training_run_contract": contract,
        "training_run_contract_sha256": contract_sha,
        "data": {"manifest_sha256": _digest(data_files), "files": data_files},
        "prompting": deepcopy(native["prompting_exact"]),
        "generation": {
            **deepcopy(native["generation_exact"]),
            "num_return_sequences": 8,
            "generation_batch_size": 8,
        },
        "selection": {
            "evaluation_purpose": "development",
            "benchmark_manifest": None,
            "checkpoint_selection_receipt": None,
            "tasks": ["tsp"],
            "split": "eval",
            "max_eval_samples_per_task": 1,
            "eval_start_index": 0,
            "eval_samples_per_task": {},
            "size_bucket": "all",
            "min_instance_size": 0,
            "max_instance_size": 0,
            "prefer_subset_files": False,
        },
        "verifier": verifier,
        "code": {"manifest_sha256": _digest(code_files), "files": code_files},
        "runtime_versions": {"python": "3.11", "torch": "test", "cuda_available": True},
    }
    experiment_fingerprint = _digest(provenance)
    protocol_material = aggregate._protocol_fingerprint_material(provenance)
    protocol_fingerprint = _digest(protocol_material)
    instruction = "Find a TSP tour. Return Route and Objective."
    input_problem = "Node 0, coordinates: [0, 0];Node 1, coordinates: [3, 4]."
    ground_truth_output = "Route: [0, 1, 0]\nObjective: 10.0"
    instance = [[0.0, 0.0], [3.0, 4.0]]
    source_row = {
        "instruction": instruction,
        "input": input_problem,
        "output": ground_truth_output,
        "instance_json": instance,
    }
    reference_objective = 10.0
    candidates = []
    for index in range(8):
        completion = ground_truth_output if index == 0 else "invalid"
        candidate = aggregate.TASK_EVALUATORS["tsp"](
            instruction,
            input_problem,
            ground_truth_output,
            instance,
            completion,
        )
        candidate.update(
            aggregate._constraint_diagnostics(
                "tsp",
                prompt=instruction,
                input_text=input_problem,
                answer=ground_truth_output,
                instance=instance,
                completion=completion,
            )
        )
        candidate.update(
            {
                "candidate_index": index,
                "completion_tokens": 3,
                "completion": completion,
                "decoding_temperature": 0.7,
                "decoding_top_p": 0.9,
            }
        )
        candidates.append(candidate)
    transformed_identity = {
        "instruction": instruction,
        "input": input_problem,
        "output": ground_truth_output,
        "instance_json": instance,
    }
    record = {
        "task": "tsp",
        "split": "eval",
        "example_index": 0,
        "num_return_sequences": 8,
        "best_of_n_feasible": True,
        "best_feasible_gap": 0.0,
        "best_objective_value": 10.0,
        "ground_truth_objective": reference_objective,
        "best_candidate_index": 0,
        "feasible_candidates": 1,
        "candidate_feasibility_rate": 0.125,
        "feasible_gap_mean_in_batch": 0.0,
        "feasible_gap_std_in_batch": 0.0,
        "candidates": candidates,
        "generated_tokens": 24,
        "generation_time_sec": 1.0,
        "generation_time_sec_per_candidate": 0.125,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 120,
        "instance_json": instance,
        "instance_sha256": _digest(instance),
        "instruction": instruction,
        "input_problem": input_problem,
        "ground_truth_output": ground_truth_output,
        "transformed_example_sha256": _digest(transformed_identity),
        "semantic_instance_sha256": aggregate.semantic_instance_sha256(
            "tsp", source_row, require_structured=True
        ),
        "benchmark_row_sha256": aggregate.benchmark_row_sha256(source_row),
        "reference_output_sha256": aggregate.reference_output_sha256(ground_truth_output),
        "source_row_locator": {
            "data_manifest_sha256": provenance["data"]["manifest_sha256"],
            "task": "tsp",
            "split": "eval",
            "merged_dataset_index": 0,
        },
        "sampling_seed": 123,
        "experiment_fingerprint": experiment_fingerprint,
        "experiment_fingerprint_schema_version": 1,
        "protocol_fingerprint": protocol_fingerprint,
        "strict_verifier": verifier,
        "training_run_contract_sha256": contract_sha,
    }
    record = _seal(record, "record_sha256")
    details_path = tmp_path / "run_details__paper.jsonl"
    details_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    details_sha, details_size = aggregate._sha256_file(str(details_path))
    details_identity = {
        "basename": details_path.name,
        "sha256": details_sha,
        "size_bytes": details_size,
        "record_count": 1,
    }
    aggregate_row = {
        "examples": 1,
        "candidates": 8,
        "feasible_candidates": 1,
        "candidate_feasibility_rate": 0.125,
        "best_of_n_feasible_examples": 1,
        "best_of_n_instance_feasibility_rate": 1.0,
        "mean_best_feasible_gap": 0.0,
        "total_generation_time_sec": 1.0,
        "generated_tokens": 24,
    }
    common = {
        "run_status": "complete",
        "evaluation_purpose": "development",
        "checkpoint_name": "checkpoint-3200",
        "resolved_model_path": str(checkpoint_dir),
        "resolved_run_dir": str(run_dir),
        "experiment_fingerprint": experiment_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "experiment_provenance": provenance,
        "training_run_contract": contract,
        "training_run_contract_sha256": contract_sha,
        "training_run_contract_identity": {
            "basename": "run_contract.json",
            "sha256": contract_sha,
            "size_bytes": len(aggregate._training_contract_file_bytes(contract)),
            "payload_sha256": _digest(contract),
        },
        "code_identity": provenance["code"],
        "verifier_identity": verifier,
        "details_artifact": details_identity,
    }
    config_path = tmp_path / "run_config__paper.json"
    common["files"] = {"run_config": str(config_path)}
    config_path.write_text(json.dumps(_seal(common, "run_config_sha256")), encoding="utf-8")
    summary = {
        **common,
        "overall": aggregate_row,
        "per_task": {"tsp": aggregate_row},
    }
    summary_path = tmp_path / "run_summary__paper.json"
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    return (
        aggregate.RunSpec("clasp_canonical", 42, str(details_path), "tsp"),
        details_path,
        config_path,
        summary_path,
    )


def _rescore_fixture(tmp_path: Path) -> tuple[aggregate.RunSpec, Path, Path]:
    native_spec, native_details, config_path, summary_path = _native_fixture(tmp_path / "native")
    native_record = json.loads(native_details.read_text(encoding="utf-8"))
    config = aggregate._load_json_object(str(config_path))
    native_summary = aggregate._load_json_object(str(summary_path))
    native_sha, native_size = aggregate._sha256_file(str(native_details))
    material = {
        "schema_version": 1,
        "system_name": "external",
        "sources": [
            {
                "basename": native_details.name,
                "size": native_size,
                "sha256": native_sha,
            }
        ],
        "source_config": {
            "basename": config_path.name,
            "size_bytes": config_path.stat().st_size,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "source_experiment_fingerprint": config["experiment_fingerprint"],
        "source_protocol_fingerprint": config["protocol_fingerprint"],
        "source_experiment_provenance": config["experiment_provenance"],
        "training_run_contract": config["training_run_contract"],
        "training_run_contract_sha256": config["training_run_contract_sha256"],
        "source_campaign_manifest": config.get("campaign_manifest"),
        "source_campaign_run_binding": config.get("campaign_run_binding"),
        "source_confirmation_attempt_claim": config.get("confirmation_attempt_claim"),
        "verifier": config["verifier_identity"],
        "rescore_implementation": {
            "basename": "rescore.py",
            "size": 1,
            "sha256": "9" * 64,
        },
    }
    rescore_fingerprint = _digest(material)
    record = deepcopy(native_record)
    record.pop("experiment_fingerprint")
    record.pop("experiment_fingerprint_schema_version")
    record.pop("protocol_fingerprint")
    record.pop("record_sha256")
    record["source_experiment_fingerprint"] = config["experiment_fingerprint"]
    record["source_experiment_fingerprint_schema_version"] = 1
    record["source_protocol_fingerprint"] = config["protocol_fingerprint"]
    record["rescore_fingerprint"] = rescore_fingerprint
    record = _seal(record, "rescore_record_sha256")
    details = tmp_path / "rescored.jsonl"
    details.parent.mkdir(parents=True, exist_ok=True)
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    details_sha, details_size = aggregate._sha256_file(str(details))
    summary = {
        "system_name": "external",
        "system_lane": "strict_rescore_raw_generation",
        "rescore_fingerprint": rescore_fingerprint,
        "fingerprint_material": material,
        "source_experiment_fingerprint": config["experiment_fingerprint"],
        "source_protocol_fingerprint": config["protocol_fingerprint"],
        "source_experiment_provenance": config["experiment_provenance"],
        "training_run_contract": config["training_run_contract"],
        "training_run_contract_sha256": config["training_run_contract_sha256"],
        "source_campaign_manifest": config.get("campaign_manifest"),
        "source_campaign_run_binding": config.get("campaign_run_binding"),
        "source_confirmation_attempt_claim": config.get("confirmation_attempt_claim"),
        "source_config_snapshot": publication_fidelity.build_derived_source_config_snapshot(
            config,
            material["source_config"],
        ),
        "record_count": 1,
        "overall": native_summary["overall"],
        "per_task": native_summary["per_task"],
        "details_artifact": {
            "basename": details.name,
            "sha256": details_sha,
            "size_bytes": details_size,
            "record_count": 1,
        },
    }
    rescore_summary = tmp_path / "rescored__summary.json"
    rescore_summary.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    return (
        aggregate.RunSpec(
            native_spec.method,
            native_spec.seed,
            str(details),
            native_spec.task,
        ),
        details,
        rescore_summary,
    )


def test_publication_loader_verifies_full_native_integrity(tmp_path: Path) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path)
    evidence: list[aggregate.ArtifactEvidence] = []
    records = aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence,
    )
    assert list(records) == [("tsp", "eval", 0)]
    assert evidence[0].artifact_type == "native"


def test_publication_fidelity_identity_and_checkpoint_bytes_are_closed(
    tmp_path: Path,
) -> None:
    _spec, _details, config_path, _summary = _native_fixture(tmp_path)
    provenance = aggregate._load_json_object(str(config_path))["experiment_provenance"]
    identity = aggregate._publication_checkpoint_fidelity(provenance, label="fixture")
    assert identity["status"] == "publication_exact"

    missing = deepcopy(provenance)
    missing["model"].pop("checkpoint_fidelity")
    with pytest.raises(ValueError, match="checkpoint fidelity is not publication-exact"):
        aggregate._publication_checkpoint_fidelity(missing, label="missing")

    mismatched = deepcopy(provenance)
    mismatched["model"]["checkpoint_artifact"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint artifact disagrees"):
        aggregate._publication_checkpoint_fidelity(mismatched, label="mismatched")


def test_training_contract_requires_full_schema_and_internal_tree_hashes(
    tmp_path: Path,
) -> None:
    spec, _details, config_path, _summary = _native_fixture(tmp_path)
    config = aggregate._load_json_object(str(config_path))
    contract = config["training_run_contract"]
    missing = deepcopy(contract)
    missing.pop("generation")
    with pytest.raises(ValueError, match="full official schema"):
        aggregate._validate_training_contract(
            missing,
            aggregate._training_contract_file_sha256(missing),
            method=spec.method,
            seed=spec.seed,
            task=spec.task,
            label="test",
        )

    tampered = deepcopy(contract)
    tampered["fingerprints"]["source"]["files"][0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="aggregate hash/count/size"):
        aggregate._validate_training_contract(
            tampered,
            aggregate._training_contract_file_sha256(tampered),
            method=spec.method,
            seed=spec.seed,
            task=spec.task,
            label="test",
        )


def test_positive_beta_contract_requires_the_exact_live_fixed_kl_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _details, config_path, _summary = _native_fixture(tmp_path)
    contract = aggregate._load_json_object(str(config_path))["training_run_contract"]
    specialist = contract["model_initialization_registry"]["specialist_sft_by_task"]["tsp"]
    identity = _fixed_kl_identity(specialist["checkpoint_fidelity"])
    model_path = os.path.realpath(contract["fingerprints"]["initial_model"]["canonical_path"])
    receipt_path = os.path.join(model_path, "checkpoint_fidelity.json")
    contract["training_config"]["beta"] = 0.05
    contract["fixed_kl_reference"] = identity
    contract["tracked_environment"]["CLASP_KL_REFERENCE_MODEL_PATH"] = model_path
    contract["tracked_environment"]["CLASP_KL_REFERENCE_RECEIPT"] = receipt_path
    aggregate._FIXED_KL_LIVE_CACHE.clear()
    monkeypatch.setattr(
        aggregate,
        "fixed_kl_reference_identity",
        lambda observed_model, observed_receipt: (
            identity
            if (observed_model, observed_receipt) == (model_path, receipt_path)
            else pytest.fail("fixed-KL verifier received unbound paths")
        ),
    )

    validated = aggregate._validate_training_contract(
        contract,
        aggregate._training_contract_file_sha256(contract),
        method=spec.method,
        seed=spec.seed,
        task=spec.task,
        label="positive-beta",
    )
    assert validated["fixed_kl_reference"] == identity

    aggregate._FIXED_KL_LIVE_CACHE.clear()
    monkeypatch.setattr(
        aggregate,
        "fixed_kl_reference_identity",
        lambda _model, _receipt: {**identity, "identity_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="live fixed-KL reference differs"):
        aggregate._validate_training_contract(
            contract,
            aggregate._training_contract_file_sha256(contract),
            method=spec.method,
            seed=spec.seed,
            task=spec.task,
            label="positive-beta",
        )


def test_publication_loader_rejects_record_and_details_tampering(tmp_path: Path) -> None:
    spec, details, _config, _summary = _native_fixture(tmp_path)
    record = json.loads(details.read_text(encoding="utf-8"))
    record["best_feasible_gap"] = 0.2
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record_sha256 mismatch"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_publication_loader_rejects_path_replacement_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, details, _config, _summary = _native_fixture(tmp_path)
    original_validator = aggregate._validate_publication_companion

    def replace_then_validate(path: str, records: list[dict], snapshot: object):
        replacement = details.with_name("replacement.jsonl")
        replacement.write_bytes(details.read_bytes() + b"\n")
        replacement.replace(details)
        return original_validator(path, records, snapshot)

    monkeypatch.setattr(
        aggregate,
        "_validate_publication_companion",
        replace_then_validate,
    )
    with pytest.raises(ValueError, match="pathname changed after snapshot"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)

    spec, details, _config, _summary = _native_fixture(tmp_path / "second")
    with details.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="details_artifact mismatch"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_publication_loader_recomputes_provenance_and_summary(tmp_path: Path) -> None:
    spec, _details, config_path, summary_path = _native_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config["experiment_fingerprint"] = "0" * 64
    summary["experiment_fingerprint"] = "0" * 64
    config_path.write_text(json.dumps(_seal(config, "run_config_sha256")), encoding="utf-8")
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    with pytest.raises(ValueError, match="experiment fingerprint does not match provenance"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_publication_loader_recomputes_headline_from_resealed_candidates(
    tmp_path: Path,
) -> None:
    spec, details, _config, _summary = _native_fixture(tmp_path)
    record = json.loads(details.read_text(encoding="utf-8"))
    record["candidates"][0]["is_feasible"] = False
    record = _seal(record, "record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict verifier replay"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_publication_loader_recomputes_gap_and_requires_full_diagnostics(
    tmp_path: Path,
) -> None:
    spec, details, _config, _summary = _native_fixture(tmp_path / "gap")
    record = json.loads(details.read_text(encoding="utf-8"))
    record["candidates"][0]["gap_to_ground_truth"] = 0.1
    record["best_feasible_gap"] = 0.1
    record["feasible_gap_mean_in_batch"] = 0.1
    record = _seal(record, "record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict verifier replay"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)

    spec, details, _config, _summary = _native_fixture(tmp_path / "diagnostics")
    record = json.loads(details.read_text(encoding="utf-8"))
    record["candidates"][0]["constraint_violations"].pop("missing_customer")
    record = _seal(record, "record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="constraint_violations disagrees with strict verifier replay"
    ):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)

    spec, details, _config, _summary = _native_fixture(tmp_path / "reference")
    record = json.loads(details.read_text(encoding="utf-8"))
    record["candidates"][0]["ground_truth_objective"] = 11.0
    record = _seal(record, "record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="ground_truth_objective disagrees with strict verifier replay"
    ):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_rescore_records_are_bound_to_source_provenance(tmp_path: Path) -> None:
    spec, details, summary_path = _rescore_fixture(tmp_path)
    evidence: list[aggregate.ArtifactEvidence] = []
    aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence,
    )
    assert evidence[0].artifact_type == "rescore"

    record = json.loads(details.read_text(encoding="utf-8"))
    record["source_experiment_fingerprint"] = "0" * 64
    record = _seal(record, "rescore_record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary = aggregate._load_json_object(str(summary_path))
    details_sha, details_size = aggregate._sha256_file(str(details))
    summary["details_artifact"] = {
        "basename": details.name,
        "sha256": details_sha,
        "size_bytes": details_size,
        "record_count": 1,
    }
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    with pytest.raises(ValueError, match="record source experiment mismatch"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_rescore_source_snapshot_is_portable_and_tamper_evident(tmp_path: Path) -> None:
    spec, _details, summary_path = _rescore_fixture(tmp_path)
    summary = aggregate._load_json_object(str(summary_path))
    declared_source = Path(
        summary["source_config_snapshot"]["source_config_payload"]["files"]["run_config"]
    )
    assert declared_source.is_file() and declared_source.parent != summary_path.parent
    declared_source.unlink()
    # Publication validation succeeds from the portable snapshot after the exact
    # source config is no longer available at either its original or a co-located path.
    aggregate.load_run(spec, expected_k=8, publication_mode=True)

    summary["source_config_snapshot"]["source_config_payload"]["run_status"] = "tampered"
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    with pytest.raises(ValueError, match="source-config snapshot self-hash mismatch"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)

    spec, _details, _config, summary_path = _native_fixture(tmp_path / "summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["overall"]["generated_tokens"] = 999
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    with pytest.raises(ValueError, match="generated_tokens disagrees with details"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_rescore_rehashes_original_source_config_when_it_is_still_live(
    tmp_path: Path,
) -> None:
    spec, _details, summary_path = _rescore_fixture(tmp_path)
    summary = aggregate._load_json_object(str(summary_path))
    source_path = Path(
        summary["source_config_snapshot"]["source_config_payload"]["files"]["run_config"]
    )
    source_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live source config differs from its identity"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_rescore_fingerprint_material_cannot_swap_source_provenance(
    tmp_path: Path,
) -> None:
    spec, details, summary_path = _rescore_fixture(tmp_path)
    summary = aggregate._load_json_object(str(summary_path))
    material = summary["fingerprint_material"]
    material["source_experiment_provenance"]["runtime_versions"]["torch"] = "tampered"
    fingerprint = _digest(material)
    summary["rescore_fingerprint"] = fingerprint

    record = json.loads(details.read_text(encoding="utf-8"))
    record["rescore_fingerprint"] = fingerprint
    record = _seal(record, "rescore_record_sha256")
    details.write_text(json.dumps(record) + "\n", encoding="utf-8")
    details_sha, details_size = aggregate._sha256_file(str(details))
    summary["details_artifact"] = {
        "basename": details.name,
        "sha256": details_sha,
        "size_bytes": details_size,
        "record_count": 1,
    }
    summary_path.write_text(json.dumps(_seal(summary, "summary_sha256")), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint omits/changes source provenance"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_repair_top_level_metrics_are_closed_over_nested_strict_views(
    tmp_path: Path,
) -> None:
    _spec, details, _config, _summary = _native_fixture(tmp_path)
    raw_record = json.loads(details.read_text(encoding="utf-8"))
    raw_view = deepcopy(raw_record)
    repaired_view = deepcopy(raw_record)
    raw_view.pop("record_sha256")
    repaired_view.pop("record_sha256")
    wrapped_candidates = [
        {
            **deepcopy(candidate),
            "raw_metrics": deepcopy(candidate),
            "repaired_metrics": deepcopy(candidate),
        }
        for candidate in raw_record["candidates"]
    ]
    repair_record = {
        **deepcopy(repaired_view),
        "raw_record": deepcopy(raw_record),
        "raw_strict_view": raw_view,
        "repaired_strict_view": repaired_view,
        "candidates": wrapped_candidates,
        "repair_fingerprint": "a" * 64,
        "raw_record_sha256": _digest(raw_record),
    }
    repair_record.pop("record_sha256", None)
    repair_record = _seal(repair_record, "repair_record_sha256")
    aggregate._validate_record_self_hash(
        repair_record,
        artifact_type="repair",
        label="repair",
    )
    aggregate._validate_repair_views(repair_record, label="repair")

    tampered = deepcopy(repair_record)
    tampered["candidates"][0]["objective_value"] = 11.0
    tampered["best_objective_value"] = 11.0
    tampered["candidates"][0]["gap_to_ground_truth"] = 0.1
    tampered["best_feasible_gap"] = 0.1
    tampered["feasible_gap_mean_in_batch"] = 0.1
    tampered = _seal(tampered, "repair_record_sha256")
    aggregate._validate_record_self_hash(
        tampered,
        artifact_type="repair",
        label="repair",
    )
    with pytest.raises(ValueError, match="disagrees with repaired_strict_view"):
        aggregate._validate_repair_views(tampered, label="repair")


def test_raw_lane_rejects_rescore_or_repair_ambiguity(tmp_path: Path) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path)
    evidence: list[aggregate.ArtifactEvidence] = []
    records = aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence,
    )
    registry, _ = aggregate._load_lane_contract_registry(
        str(ROOT / "experiments" / "sota_20260814" / "lane_contracts.json")
    )
    aggregate._validate_lane_evidence(
        spec,
        records,
        evidence,
        lane_name="raw_n8",
        lane_registry=registry,
        confirmation=False,
    )
    tampered = deepcopy(records)
    tampered[("tsp", "eval", 0)]["rescore_fingerprint"] = "9" * 64
    with pytest.raises(ValueError, match="exactly one native/repair/rescore"):
        aggregate._validate_lane_evidence(
            spec,
            tampered,
            evidence,
            lane_name="raw_n8",
            lane_registry=registry,
            confirmation=False,
        )


def test_adaptive_lane_is_not_claim_eligible_without_a_producer(tmp_path: Path) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path)
    records = aggregate.load_run(spec, expected_k=8, publication_mode=True)
    fake_repair = deepcopy(next(iter(records.values())))
    fake_repair.pop("experiment_fingerprint")
    fake_repair["repair_fingerprint"] = "a" * 64
    fake_repair["raw_record"] = deepcopy(next(iter(records.values())))
    fixture_config = aggregate._load_json_object(str(tmp_path / "run_config__paper.json"))
    repair_evidence = aggregate.ArtifactEvidence(
        artifact_type="repair",
        details_path=str(tmp_path / "repair.jsonl"),
        companion_path=str(tmp_path / "repair_summary.json"),
        experiment_provenance=deepcopy(fixture_config["experiment_provenance"]),
        training_run_contract=fixture_config["training_run_contract"],
        training_run_contract_sha256=fixture_config["training_run_contract_sha256"],
        checkpoint_selection_binding=None,
        checkpoint_selection_receipt=None,
        checkpoint_fidelity=fixture_config["experiment_provenance"]["model"]["checkpoint_fidelity"],
    )
    repair_evidence.experiment_provenance["generation"]["num_return_sequences"] = 64
    repair_evidence.experiment_provenance["generation"]["generation_batch_size"] = 8
    registry, _ = aggregate._load_lane_contract_registry(
        str(ROOT / "experiments" / "sota_20260814" / "lane_contracts.json")
    )
    with pytest.raises(ValueError, match="not eligible for a publication/SOTA claim"):
        aggregate._validate_lane_evidence(
            spec,
            {("tsp", "eval", 0): fake_repair},
            [repair_evidence],
            lane_name="repair_adaptive_n8_64",
            lane_registry=registry,
            confirmation=False,
        )


def test_confirmation_design_and_embedded_contract_override_cli_labels(tmp_path: Path) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path)
    evidence: list[aggregate.ArtifactEvidence] = []
    aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence,
    )
    wrong = aggregate.RunSpec("dual_scalar_z", 42, spec.path, "tsp")
    with pytest.raises(ValueError, match="CLI identity disagrees with embedded"):
        aggregate._validate_training_contract(
            evidence[0].training_run_contract,
            evidence[0].training_run_contract_sha256,
            method=wrong.method,
            seed=wrong.seed,
            task=str(wrong.task),
            label="test",
        )

    manifest = {
        "schema_version": 1,
        "status": "frozen_before_confirmation",
        "campaign_id": "campaign",
        "lane": "raw_n8",
        "decode_seed": 271828,
        "baseline": "dual_scalar_z",
        "challenger": "clasp_canonical",
        "methods": ["dual_scalar_z", "clasp_canonical"],
        "approved_fixed_null_receipt_sha256_by_task": deepcopy(
            aggregate.APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
        ),
        "training_seeds": [42],
        "tasks": ["tsp"],
        "lane_contract_registry_sha256": "a" * 64,
        "target_registry_sha256": "b" * 64,
        "benchmark_manifest_sha256": "c" * 64,
        "runs": [
            {
                "method": "dual_scalar_z",
                "training_seed": 42,
                "task": "tsp",
            }
        ],
    }
    benchmark = {"manifest_sha256": "c" * 64, "test_sets": {"tsp": {}}}
    with pytest.raises(ValueError, match="exact prespecified training seeds"):
        aggregate._validate_campaign_manifest(
            manifest,
            campaign_manifest_identity={
                "basename": "campaign.json",
                "file_sha256": "d" * 64,
                "size_bytes": 1,
                "campaign_manifest_sha256": "e" * 64,
                "campaign_id": "campaign",
            },
            specs=[wrong],
            by_method_seed={},
            evidence_by_spec={wrong: evidence},
            lane_name="raw_n8",
            lane_registry_sha256="a" * 64,
            target_registry_sha256="b" * 64,
            benchmark_manifest=benchmark,
            baseline="dual_scalar_z",
            challenger="clasp_canonical",
        )


def test_confirmation_comparison_roles_are_frozen_before_outcomes() -> None:
    manifest = {
        "baseline": "dual_scalar_z",
        "challenger": "clasp_canonical",
        "methods": ["dual_scalar_z", "clasp_canonical"],
        "approved_fixed_null_receipt_sha256_by_task": deepcopy(
            aggregate.APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
        ),
        "training_seeds": list(aggregate.CONFIRMATION_TRAINING_SEEDS),
    }
    aggregate._validate_confirmation_comparison_design(
        manifest,
        baseline="dual_scalar_z",
        challenger="clasp_canonical",
    )
    with pytest.raises(ValueError, match="roles disagree"):
        aggregate._validate_confirmation_comparison_design(
            manifest,
            baseline="clasp_canonical",
            challenger="dual_scalar_z",
        )
    extra_method = deepcopy(manifest)
    extra_method["methods"].append("grpo_z")
    with pytest.raises(ValueError, match="methods must equal"):
        aggregate._validate_confirmation_comparison_design(
            extra_method,
            baseline="dual_scalar_z",
            challenger="clasp_canonical",
        )
    legacy = deepcopy(manifest)
    legacy.update(
        {
            "baseline": "dual_grpo",
            "methods": ["dual_grpo", "clasp_canonical"],
        }
    )
    with pytest.raises(ValueError, match="prohibited legacy alias"):
        aggregate._validate_confirmation_comparison_design(
            legacy,
            baseline="dual_grpo",
            challenger="clasp_canonical",
        )
    for foarl_method, message in (
        ("foarl_f638_native", "unmatched descriptive lane"),
        ("foarl_f638_matched_b9", "requires the exact ordered pair"),
    ):
        foarl = deepcopy(manifest)
        foarl.update({"baseline": foarl_method, "methods": [foarl_method, "clasp_canonical"]})
        with pytest.raises(ValueError, match=message):
            aggregate._validate_confirmation_comparison_design(
                foarl,
                baseline=foarl_method,
                challenger="clasp_canonical",
            )
    exact_foarl = deepcopy(manifest)
    exact_foarl.update(
        {
            "baseline": "foarl_f638_matched_b9",
            "challenger": "clasp_canonical_foarl_matched_b9",
            "methods": [
                "foarl_f638_matched_b9",
                "clasp_canonical_foarl_matched_b9",
            ],
        }
    )
    aggregate._validate_confirmation_comparison_design(
        exact_foarl,
        baseline="foarl_f638_matched_b9",
        challenger="clasp_canonical_foarl_matched_b9",
    )
    wrong_fixed_null_registry = deepcopy(manifest)
    wrong_fixed_null_registry["approved_fixed_null_receipt_sha256_by_task"]["tsp"] = "0" * 64
    with pytest.raises(ValueError, match="approved fixed-null receipts"):
        aggregate._validate_confirmation_comparison_design(
            wrong_fixed_null_registry,
            baseline="dual_scalar_z",
            challenger="clasp_canonical",
        )


def test_foarl_confirmation_statistics_require_exact_shared_20k1k_projection() -> None:
    seed = 42
    task = "tsp"
    common = {
        "task": task,
        "seeds": {"train": seed, "data": seed, "sft": 42},
        "fixed_horizon": {"max_steps": 3_200, "world_size": 4},
        "data_lineage": {"registered": "same"},
        "fingerprints": {
            "initial_model": {"sha256": "a" * 64},
            "base_model": {"sha256": "b" * 64},
            "data": {"sha256": "c" * 64},
            "source": {"sha256": "d" * 64},
        },
        "generation": dict(SHARED_GENERATION_CONTRACT),
        "tracked_environment": {
            **SHARED_ENVIRONMENT_CONTRACT,
            "CLASP_MIS_JIANG_RESPONSE_OUTPUT": "0",
            "GRPO_DATA_SEED": str(seed),
        },
    }
    foarl = {
        **deepcopy(common),
        "method": "foarl_f638_matched_b9",
        "training_config": {
            **SHARED_TRAINING_CONTRACT,
            "data_seed": seed,
            "entrypoint": "clasp.entrypoints.train_foarl_f638_campaign",
        },
    }
    clasp = {
        **deepcopy(common),
        "method": CLASP_FOARL_MATCHED_B9,
        "training_config": {
            **SHARED_TRAINING_CONTRACT,
            "data_seed": seed,
            "entrypoint": "clasp.entrypoints.train_grpo",
            "arm_contract_schema": CLASP_FOARL_MATCHED_B9_ARM_SCHEMA,
            "comparison_protocol": (
                "foarl-f638-clasp-shared-compute-data-token-design-v1"
            ),
            "comparison_baseline": "foarl_f638_matched_b9",
            "attribution": "common_compute_data_token_schedule_only",
            "reproduction_claim": "not_an_official_jiang_reproduction",
            "prompt_style": "alpaca",
            "no_reason": True,
            "loss_type": "grpo",
            "beta": 0.05,
            "use_primal_dual": True,
            "use_clasp": True,
            "clasp_canonical_set": True,
            "clasp_fixed_null_set": False,
        },
    }
    shared_hash = aggregate.foarl_matched_b9_shared_design_sha256(foarl)
    policy = {
        "schema_version": 1,
        "fixed_horizon": {"max_steps": 3_200, "world_size": 4},
        "method_specific_allowlist": list(
            FOARL_MATCHED_B9_METHOD_SPECIFIC_ALLOWLIST
        ),
        "shared_design_sha256_by_task_seed": {"tsp/seed42": shared_hash},
        "fixed_null_receipt_sha256_by_task": {},
    }
    aggregate._validate_matched_training_design(
        {"training_design": policy},
        contracts_by_group={(task, seed): {
            "foarl_f638_matched_b9": foarl,
            CLASP_FOARL_MATCHED_B9: clasp,
        }},
        methods=["foarl_f638_matched_b9", CLASP_FOARL_MATCHED_B9],
        tasks=[task],
        seeds=[seed],
    )
    clasp["generation"]["max_completion_length"] = 3_000
    with pytest.raises(ValueError, match="matched-B9 shared design"):
        aggregate._validate_matched_training_design(
            {"training_design": policy},
            contracts_by_group={(task, seed): {
                "foarl_f638_matched_b9": foarl,
                CLASP_FOARL_MATCHED_B9: clasp,
            }},
            methods=["foarl_f638_matched_b9", CLASP_FOARL_MATCHED_B9],
            tasks=[task],
            seeds=[seed],
        )


def test_confirmation_binding_matches_model_artifact_and_receipt(tmp_path: Path) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path)
    evidence_rows: list[aggregate.ArtifactEvidence] = []
    records = aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence_rows,
    )
    source = evidence_rows[0]
    provenance = deepcopy(source.experiment_provenance)
    registry, _ = aggregate._load_lane_contract_registry(
        str(ROOT / "experiments" / "sota_20260814" / "lane_contracts.json")
    )
    receipt_sha = "9" * 64
    receipt_identity = {
        "basename": "selection.json",
        "file_sha256": "8" * 64,
        "size_bytes": 123,
        "receipt_sha256": receipt_sha,
        "campaign_id": "campaign",
        "stage_id": "selection",
    }
    artifact_sha = aggregate._checkpoint_artifact_sha256(provenance)
    benchmark_manifest = _valid_tsp_benchmark_manifest(source.experiment_provenance["verifier"])
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_manifest), encoding="utf-8")
    benchmark_file_sha, benchmark_size = aggregate._sha256_file(str(benchmark_path))
    validated_benchmark = aggregate.validate_frozen_benchmark_manifest(benchmark_manifest)
    benchmark_identity = {
        "basename": benchmark_path.name,
        "file_sha256": benchmark_file_sha,
        "size_bytes": benchmark_size,
        **validated_benchmark,
    }
    binding = {
        "group_id": "tsp/clasp_canonical/seed42",
        "checkpoint_step": 3200,
        "checkpoint_artifact_sha256": artifact_sha,
        "run_contract_sha256": source.training_run_contract_sha256,
        "checkpoint_selection_receipt_sha256": receipt_sha,
        "selection_evaluation_fingerprint": "7" * 64,
        "checkpoint_fidelity_identity_sha256": source.checkpoint_fidelity["identity_sha256"],
        "checkpoint_fidelity_receipt_sha256": source.checkpoint_fidelity["receipt_sha256"],
        "checkpoint_fidelity_receipt_file_sha256": source.checkpoint_fidelity[
            "receipt_file_sha256"
        ],
        "grpo_beta": source.training_run_contract["training_config"]["beta"],
        "fixed_kl_reference": source.training_run_contract["fixed_kl_reference"],
    }
    provenance["selection"] = {
        **deepcopy(registry["native_contract"]["confirmation_selection_exact"]),
        "tasks": ["tsp"],
        "benchmark_manifest": benchmark_identity,
        "checkpoint_selection_binding": binding,
        "checkpoint_selection_receipt": receipt_identity,
    }
    evidence = aggregate.ArtifactEvidence(
        artifact_type="native",
        details_path=source.details_path,
        companion_path=source.companion_path,
        experiment_provenance=provenance,
        training_run_contract=source.training_run_contract,
        training_run_contract_sha256=source.training_run_contract_sha256,
        checkpoint_selection_binding=binding,
        checkpoint_selection_receipt=receipt_identity,
        checkpoint_fidelity=source.checkpoint_fidelity,
    )
    aggregate._validate_lane_evidence(
        spec,
        records,
        [evidence],
        lane_name="raw_n8",
        lane_registry=registry,
        confirmation=True,
    )
    aggregate._validate_benchmark_provenance_bindings(
        {spec: [evidence]},
        benchmark_path=str(benchmark_path),
        benchmark_manifest=benchmark_manifest,
    )

    wrong_benchmark = deepcopy(evidence)
    wrong_benchmark.experiment_provenance["selection"]["benchmark_manifest"]["manifest_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="bound to another benchmark artifact"):
        aggregate._validate_benchmark_provenance_bindings(
            {spec: [wrong_benchmark]},
            benchmark_path=str(benchmark_path),
            benchmark_manifest=benchmark_manifest,
        )

    bad_artifact = deepcopy(binding)
    bad_artifact["checkpoint_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint selection evidence disagrees"):
        aggregate._validate_lane_evidence(
            spec,
            records,
            [
                aggregate.ArtifactEvidence(
                    **{**evidence.__dict__, "checkpoint_selection_binding": bad_artifact}
                )
            ],
            lane_name="raw_n8",
            lane_registry=registry,
            confirmation=True,
        )

    bad_receipt = deepcopy(binding)
    bad_receipt["checkpoint_selection_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint selection evidence disagrees"):
        aggregate._validate_lane_evidence(
            spec,
            records,
            [
                aggregate.ArtifactEvidence(
                    **{**evidence.__dict__, "checkpoint_selection_binding": bad_receipt}
                )
            ],
            lane_name="raw_n8",
            lane_registry=registry,
            confirmation=True,
        )


def test_protocol_fingerprint_is_shared_across_method_selection_receipts(
    tmp_path: Path,
) -> None:
    _spec, _details, config_path, _summary = _native_fixture(tmp_path)
    provenance_a = aggregate._load_json_object(str(config_path))["experiment_provenance"]
    provenance_b = deepcopy(provenance_a)
    provenance_b["model"]["checkpoint_artifact"]["manifest_sha256"] = "0" * 64
    provenance_b["training_run_contract"]["method"] = "dual_scalar_z"
    provenance_b["training_run_contract_sha256"] = "1" * 64
    provenance_a["selection"]["checkpoint_selection_receipt"] = {"receipt_sha256": "2" * 64}
    provenance_a["selection"]["checkpoint_selection_binding"] = {
        "group_id": "tsp/clasp_canonical/seed42"
    }
    provenance_b["selection"]["checkpoint_selection_receipt"] = {"receipt_sha256": "3" * 64}
    provenance_b["selection"]["checkpoint_selection_binding"] = {
        "group_id": "tsp/dual_scalar_z/seed42"
    }
    protocol_a = _digest(aggregate._protocol_fingerprint_material(provenance_a))
    protocol_b = _digest(aggregate._protocol_fingerprint_material(provenance_b))
    assert protocol_a == protocol_b
    provenance_c = deepcopy(provenance_b)
    provenance_c["model"]["tokenizer_artifact"]["manifest_sha256"] = "4" * 64
    protocol_c = _digest(aggregate._protocol_fingerprint_material(provenance_c))
    assert protocol_c != protocol_a


def test_holm_uses_exactly_five_task_level_seed_aware_hypotheses() -> None:
    comparisons = {
        task: {
            "per_training_seed": {
                str(seed): {
                    "challenger_minus_baseline_normalized_feasible_gap_auc": 0.1,
                    "challenger_minus_baseline_pass_at_k": 0.1,
                }
                for seed in aggregate.CONFIRMATION_TRAINING_SEEDS
            }
        }
        for task in aggregate.PUBLICATION_TASKS
    }
    families = aggregate._apply_seed_family_holm(comparisons)
    assert set(families) == set(aggregate.CONFIRMATION_OUTCOME_FAMILIES)
    assert all(row["family_size"] == 5 for row in families.values())
    family = "normalized_feasible_gap_auc_seed_randomization"
    assert set(families[family]["hypotheses"]) == set(
        aggregate.PUBLICATION_TASKS
    )
    assert (
        comparisons["tsp"]["task_level_exact_tests"][family][
            "inference_unit"
        ]
        == "paired_training_seed"
    )
    assert (
        "holm_adjusted_p_across_tasks"
        in comparisons["tsp"]["task_level_exact_tests"][family]
    )
    assert families[family]["hypotheses"]["tsp"]["raw_exact_p"] == pytest.approx(2 / 256)
    assert families[family]["hypotheses"]["tsp"]["holm_adjusted_p"] == pytest.approx(10 / 256)
    assert families[family]["hypotheses"]["tsp"]["primary_superiority_support"] is True
    noninferiority = aggregate._apply_pass_at_k_noninferiority_holm(comparisons)
    assert noninferiority["overall_pass"] is True
    assert noninferiority["tasks"]["tsp"]["exact_one_sided_p"] == pytest.approx(1 / 256)
    assert noninferiority["tasks"]["tsp"]["holm_adjusted_p_across_tasks"] == pytest.approx(
        5 / 256
    )
    assert "pooled_for_descriptive_use_only" not in families


def test_primary_direction_never_labels_significant_negative_auc_as_superiority() -> None:
    family = "normalized_feasible_gap_auc_seed_randomization"
    comparisons = {
        task: {
            "per_training_seed": {
                str(seed): {
                    "challenger_minus_baseline_normalized_feasible_gap_auc": -0.1,
                    "challenger_minus_baseline_pass_at_k": 0.0,
                }
                for seed in aggregate.CONFIRMATION_TRAINING_SEEDS
            }
        }
        for task in aggregate.PUBLICATION_TASKS
    }
    result = aggregate._apply_seed_family_holm(comparisons)
    hypothesis = result[family]["hypotheses"]["tsp"]
    assert hypothesis["difference_detected_at_alpha"] is True
    assert hypothesis["primary_superiority_support"] is False
    assert hypothesis["directional_conclusion"] == "difference_detected_against_clasp"


def test_pass_at_8_noninferiority_requires_inference_and_observed_effect_gate() -> None:
    def comparisons_with(effect: float) -> dict[str, object]:
        return {
            task: {
                "per_training_seed": {
                    str(seed): {
                        "challenger_minus_baseline_pass_at_k": effect,
                    }
                    for seed in aggregate.CONFIRMATION_TRAINING_SEEDS
                }
            }
            for task in aggregate.PUBLICATION_TASKS
        }

    equal = aggregate._apply_pass_at_k_noninferiority_holm(comparisons_with(0.0))
    assert equal["overall_pass"] is True
    assert equal["tasks"]["tsp"]["exact_one_sided_p"] == pytest.approx(1 / 256)

    boundary = aggregate._apply_pass_at_k_noninferiority_holm(comparisons_with(-0.01))
    assert boundary["overall_pass"] is False
    assert boundary["tasks"]["tsp"]["observed_mean_requirement_pass"] is True
    assert boundary["tasks"]["tsp"]["holm_inference_requirement_pass"] is False

    regression = aggregate._apply_pass_at_k_noninferiority_holm(comparisons_with(-0.02))
    assert regression["overall_pass"] is False
    assert regression["tasks"]["tsp"]["observed_mean_requirement_pass"] is False


def test_confirmation_inference_rejects_missing_or_extra_seed() -> None:
    comparisons = {
        task: {
            "per_training_seed": {
                str(seed): {
                    "challenger_minus_baseline_normalized_feasible_gap_auc": 0.1,
                }
                for seed in aggregate.CONFIRMATION_TRAINING_SEEDS
            }
        }
        for task in aggregate.PUBLICATION_TASKS
    }
    comparisons["tsp"]["per_training_seed"].pop("49")
    with pytest.raises(ValueError, match="requires exact seeds"):
        aggregate._apply_seed_family_holm(comparisons)


def test_aggregation_implementation_identity_changes_with_source_bytes(
    tmp_path: Path,
) -> None:
    implementation = tmp_path / "aggregate_sota.py"
    implementation.write_text("print('one')\n", encoding="utf-8")
    first = aggregate._aggregation_implementation_identity(str(implementation))
    implementation.write_text("print('two')\n", encoding="utf-8")
    second = aggregate._aggregation_implementation_identity(str(implementation))
    assert first["implementation"]["sha256"] != second["implementation"]["sha256"]


def test_aggregate_receipt_write_is_atomic_and_immutable(tmp_path: Path) -> None:
    payload = {"schema_version": 3, "lane": "raw_n8"}
    payload["aggregate_receipt_sha256"] = _digest(payload)
    output = tmp_path / "aggregate.json"
    aggregate._atomic_json(str(output), payload)
    observed = json.loads(output.read_text(encoding="utf-8"))
    aggregate._validate_self_hash(
        observed,
        "aggregate_receipt_sha256",
        label=str(output),
    )
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to replace"):
        aggregate._atomic_json(str(output), payload)


def test_campaign_loader_requires_canonical_preconfirmation_freeze_time(
    tmp_path: Path,
) -> None:
    campaign = {
        "schema_version": aggregate.CONFIRMATION_CAMPAIGN_SCHEMA_VERSION,
        "status": "frozen_before_confirmation",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "campaign_id": "campaign",
    }
    campaign = _seal(campaign, "campaign_manifest_sha256")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    aggregate._load_campaign_manifest(str(path))

    campaign["frozen_at_utc"] = "AFTER_RESULTS"
    campaign = _seal(campaign, "campaign_manifest_sha256")
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze time"):
        aggregate._load_campaign_manifest(str(path))


def test_confirmation_attempt_formula_and_durable_claim_are_content_bound(
    tmp_path: Path,
) -> None:
    campaign_hash = "a" * 64
    manifest = {
        "campaign_id": "campaign",
        "benchmark_manifest_sha256": "b" * 64,
    }
    entry = {
        "method": "clasp_canonical",
        "training_seed": 42,
        "task": "tsp",
        "training_run_contract_sha256": "c" * 64,
        "checkpoint_step": 3200,
        "checkpoint_artifact_sha256": "d" * 64,
        "checkpoint_fidelity_identity_sha256": "f" * 64,
        "checkpoint_fidelity_receipt_sha256": "1" * 64,
        "checkpoint_fidelity_receipt_file_sha256": "2" * 64,
        "checkpoint_selection_receipt_sha256": "e" * 64,
        "grpo_beta": 0.0,
        "fixed_kl_reference": None,
        "exact_peft_kl_runtime": None,
        "exact_peft_kl_post_step": None,
    }
    attempt_id = aggregate._confirmation_attempt_id(manifest, entry)
    assert aggregate._is_sha256(attempt_id)
    changed = deepcopy(entry)
    changed["task"] = "mis"
    assert aggregate._confirmation_attempt_id(manifest, changed) != attempt_id
    changed_fidelity = deepcopy(entry)
    changed_fidelity["checkpoint_fidelity_receipt_file_sha256"] = "3" * 64
    assert aggregate._confirmation_attempt_id(manifest, changed_fidelity) != attempt_id

    ledger_root = tmp_path / "shared" / "attempt-ledger"
    ledger_dir = ledger_root / campaign_hash
    results_dir = tmp_path / "results"
    ledger_dir.mkdir(parents=True)
    results_dir.mkdir()
    provenance = {"schema_version": 1, "selection": {}}
    claim = {
        "schema_version": 2,
        "protocol": "clasp-confirmation-attempt-v2",
        "campaign_manifest_sha256": campaign_hash,
        "campaign_id": "campaign",
        "confirmation_attempt_id": attempt_id,
        "experiment_fingerprint": _digest(provenance),
        "results_dir": str(results_dir.resolve()),
    }
    claim = _seal(claim, "claim_sha256")
    claim_path = ledger_dir / f"{attempt_id}.json"
    marker_path = results_dir / ".confirmation_attempt.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    marker_path.write_text(json.dumps(claim), encoding="utf-8")
    claim_path.chmod(0o440)
    marker_path.chmod(0o440)
    details_path = results_dir / "run_details__confirmation.jsonl"
    evidence = aggregate.ArtifactEvidence(
        artifact_type="native",
        details_path=str(details_path),
        companion_path=str(results_dir / "run_config__confirmation.json"),
        experiment_provenance=provenance,
        training_run_contract=None,
        training_run_contract_sha256=None,
        checkpoint_selection_binding=None,
        checkpoint_selection_receipt=None,
        checkpoint_fidelity={},
        confirmation_attempt_claim={
            "basename": claim_path.name,
            "confirmation_attempt_id": attempt_id,
        },
    )
    aggregate._validate_confirmation_attempt_claim(
        evidence=evidence,
        ledger_root=str(ledger_root.resolve()),
        campaign_id="campaign",
        campaign_manifest_sha256=campaign_hash,
        confirmation_attempt_id=attempt_id,
    )
    swapped = aggregate.ArtifactEvidence(
        **{
            **evidence.__dict__,
            "confirmation_attempt_claim": {
                "basename": claim_path.name,
                "confirmation_attempt_id": "f" * 64,
            },
        }
    )
    with pytest.raises(ValueError, match="claim identity"):
        aggregate._validate_confirmation_attempt_claim(
            evidence=swapped,
            ledger_root=str(ledger_root.resolve()),
            campaign_id="campaign",
            campaign_manifest_sha256=campaign_hash,
            confirmation_attempt_id=attempt_id,
        )


def test_target_registry_rejects_resealed_easy_reference_in_main(tmp_path: Path) -> None:
    lane_registry_path = ROOT / "experiments" / "sota_20260814" / "lane_contracts.json"
    target = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "targets.json").read_text(encoding="utf-8")
    )
    target["lanes"]["raw_n8"]["external_context"]["reported_point_estimates"]["tsp"][
        "reported_gap"
    ] = 1.0
    target = _seal(target, "registry_sha256")
    target_path = tmp_path / "easy_targets.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed external-context registry"):
        aggregate.main(
            [
                "--run",
                f"clasp_canonical,42,{tmp_path / 'missing.jsonl'}",
                "--targets",
                str(target_path),
                "--lane-contracts",
                str(lane_registry_path),
                "--lane",
                "raw_n8",
                "--development-only",
                "--output",
                str(tmp_path / "aggregate.json"),
            ]
        )


def test_external_context_cannot_count_as_beating_a_published_point() -> None:
    registry = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "targets.json").read_text(encoding="utf-8")
    )
    context = registry["lanes"]["raw_n8"]["external_context"]
    target = context["reported_point_estimates"]["tsp"]
    assert aggregate._published_gap_target_for_task("tsp", target, external_context=context) is None
    comparison = aggregate._published_point_descriptive_comparison(
        task="tsp",
        observed_feasibility=1.0,
        observed_gap=0.0,
        published_feasibility=1.0,
        published_gap=0.0107,
        external_context=context,
        task_target=target,
    )
    assert comparison == {
        "external_point_comparison_eligible": False,
        "strict_beat_claim_eligible": False,
        "comparison_status": "CONTEXT_ONLY_NOT_COMPARABLE",
        "reason_not_claim_eligible": target["reason_not_claim_eligible"],
    }


def test_external_registry_keeps_iid_and_published_reference_semantics_separate() -> None:
    lane_registry, _ = aggregate._load_lane_contract_registry(
        str(ROOT / "experiments" / "sota_20260814" / "lane_contracts.json")
    )
    registry, registry_sha = aggregate._load_target_registry(
        str(ROOT / "experiments" / "sota_20260814" / "targets.json"),
        lane_registry=lane_registry,
    )
    assert registry_sha == aggregate.APPROVED_TARGET_REGISTRY_SHA256
    iid_op = registry["new_iid_reference_contract"]["tasks"]["op"]
    assert iid_op["intended_reference_kind"] == "certified_optimum_required"
    assert iid_op["permitted_gap_label_if_approved"] == "conditional_optimality_gap"

    raw = registry["lanes"]["raw_n8"]["external_context"]
    assert raw["comparison_scope"] == "historical_first_100_author_protocol_only"
    assert raw["scope"]["row_interval"] == [0, 100]
    assert raw["scope"]["instances_per_task"] == 100
    assert raw["reported_point_estimates"]["op"]["reference_semantics"] == (
        "cross_protocol_invalid"
    )
    assert raw["reported_point_estimates"]["mis"]["reference_semantics"] == (
        "author_reported_gurobi_uncertified"
    )
    assert raw["reported_point_estimates"]["jssp"]["reference_semantics"] == (
        "time_limited_near_optimal_reference"
    )
    assert raw["reported_point_estimates"]["tsp"]["reference_semantics"] == ("heuristic_reference")
    assert raw["reported_point_estimates"]["cvrp"]["reference_semantics"] == ("heuristic_reference")

    for lane in registry["lanes"].values():
        context = lane["external_context"]
        assert context["aggregate_point_comparison_eligible"] is False
        assert context["strict_beat_claim_eligible"] is False
        assert all(
            row["strict_beat_claim_eligible"] is False
            for row in context["reported_point_estimates"].values()
        )


def test_target_and_campaign_templates_have_valid_self_hashes() -> None:
    targets = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "targets.json").read_text(encoding="utf-8")
    )
    assert targets["registry_sha256"] == _digest(
        {key: value for key, value in targets.items() if key != "registry_sha256"}
    )
    campaign = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "campaign_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert campaign["target_registry_sha256"] == targets["registry_sha256"]
    assert campaign["campaign_manifest_sha256"] == _digest(
        {key: value for key, value in campaign.items() if key != "campaign_manifest_sha256"}
    )
    assert campaign["approved_fixed_null_receipt_sha256_by_task"] == (
        aggregate.APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK
    )
    initialization = campaign["model_initialization_registry"]
    assert initialization["schema_version"] == 2
    assert initialization["protocol"] == "clasp-model-initialization-registry-v2"
    for specialist in initialization["specialist_sft_by_task"].values():
        assert specialist["matched_sft_contract_sha256"] == aggregate.MATCHED_SFT_CONTRACT_SHA256
        assert specialist["checkpoint_fidelity"]["protocol"] == (
            "clasp-checkpoint-fidelity-identity-v1"
        )
        assert specialist["checkpoint_fidelity"]["status"] == "publication_exact"
    assert any("clasp-confirmation-attempt-v2" in line for line in campaign["instructions"])
    assert any("fixed_kl_reference" in line for line in campaign["instructions"])


def test_confirmation_requires_exact_sorted_five_task_suite() -> None:
    tasks = list(aggregate.PUBLICATION_TASKS)
    manifest = {"tasks": tasks}
    benchmark = {"tasks": tasks, "test_sets": {task: {} for task in tasks}}
    aggregate._validate_confirmation_task_suite(manifest, benchmark)
    with pytest.raises(ValueError, match="exact five-task"):
        aggregate._validate_confirmation_task_suite(
            {"tasks": ["tsp"]},
            {"tasks": ["tsp"], "test_sets": {"tsp": {}}},
        )


def test_confirmation_analysis_is_frozen_and_cli_exact() -> None:
    template = json.loads(
        (ROOT / "experiments" / "sota_20260814" / "campaign_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {"analysis": deepcopy(template["analysis"])}
    aggregate._validate_confirmation_analysis(
        manifest,
        bootstrap_repetitions=10_000,
        bootstrap_seed=20_260_814,
    )
    with pytest.raises(ValueError, match="CLI bootstrap settings"):
        aggregate._validate_confirmation_analysis(
            manifest,
            bootstrap_repetitions=100,
            bootstrap_seed=1,
        )
    for path, value in (
        (("primary_outcome", "gap_range"), [0.0, 0.2]),
        (("primary_outcome", "test"), "one_sided"),
        (("pass_at_k_noninferiority_gate", "margin"), 0.02),
    ):
        tampered = deepcopy(manifest)
        tampered["analysis"][path[0]][path[1]] = value
        with pytest.raises(ValueError, match="analysis contract is invalid"):
            aggregate._validate_confirmation_analysis(
                tampered,
                bootstrap_repetitions=10_000,
                bootstrap_seed=20_260_814,
            )


def test_publication_loader_requires_exact_completion_for_verifier_replay(
    tmp_path: Path,
) -> None:
    spec, details, _config, _summary = _native_fixture(tmp_path)
    record = json.loads(details.read_text(encoding="utf-8"))
    record["candidates"][0].pop("completion")
    details.write_text(json.dumps(_seal(record, "record_sha256")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no exact completion"):
        aggregate.load_run(spec, expected_k=8, publication_mode=True)


def test_frozen_source_row_closure_supports_file_locator_and_rejects_swap(
    tmp_path: Path,
) -> None:
    spec, _details, _config, _summary = _native_fixture(tmp_path / "run")
    evidence: list[aggregate.ArtifactEvidence] = []
    records = aggregate.load_run(
        spec,
        expected_k=8,
        publication_mode=True,
        evidence_sink=evidence,
    )
    record = next(iter(records.values()))
    source_row = {
        "instruction": record["instruction"],
        "input": record["input_problem"],
        "output": record["ground_truth_output"],
        "instance_json": record["instance_json"],
    }
    source_path = tmp_path / "test.json"
    source_path.write_text(json.dumps([source_row]), encoding="utf-8")
    source_sha, source_size = aggregate._sha256_file(str(source_path))
    benchmark = {
        "test_sets": {
            "tsp": {
                "source_locator": source_path.name,
                "files": [
                    {
                        "relative_path": source_path.name,
                        "sha256": source_sha,
                        "size_bytes": source_size,
                        "rows": 1,
                    }
                ],
                "rows": 1,
            }
        }
    }
    aggregate._validate_frozen_source_row_closure(
        specs=[spec],
        by_method_seed={(spec.method, spec.seed): records},
        evidence_by_spec={spec: evidence},
        benchmark_manifest=benchmark,
        benchmark_path=str(tmp_path / "manifest.json"),
    )
    swapped = deepcopy(records)
    swapped[("tsp", "eval", 0)]["ground_truth_output"] = "Route: [0, 0]"
    with pytest.raises(ValueError, match="transformed field ground_truth_output"):
        aggregate._validate_frozen_source_row_closure(
            specs=[spec],
            by_method_seed={(spec.method, spec.seed): swapped},
            evidence_by_spec={spec: evidence},
            benchmark_manifest=benchmark,
            benchmark_path=str(tmp_path / "manifest.json"),
        )


def test_approved_repair_is_replayed_from_raw_completion(tmp_path: Path) -> None:
    from clasp.entrypoints.repair import _repair_record

    _spec, details, _config, _summary = _native_fixture(tmp_path)
    raw_record = json.loads(details.read_text(encoding="utf-8"))
    repaired, _raw_view, _repaired_view = _repair_record(
        raw_record,
        repair_fingerprint="a" * 64,
        verifier=aggregate._current_strict_verifier_identity(),
        config=deepcopy(aggregate.APPROVED_REPAIR_CONFIG),
    )
    aggregate._validate_deterministic_repair_record(
        repaired,
        config=aggregate.APPROVED_REPAIR_CONFIG,
        label="repair",
    )
    repaired["candidates"][0]["repaired_completion"] = "ground-truth-aware forgery"
    with pytest.raises(ValueError, match="approved deterministic repair"):
        aggregate._validate_deterministic_repair_record(
            repaired,
            config=aggregate.APPROVED_REPAIR_CONFIG,
            label="repair",
        )


def _selection_registry_and_development() -> tuple[dict, dict]:
    evaluation_files = [
        {
            "task": "tsp",
            "split": "eval",
            "path": "tsp/eval/test.json",
            "size": 3,
            "sha256": _digest("historical-eval-bytes"),
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
                    else _digest(f"{label}-bytes")
                ),
                "size_bytes": 3,
                "rows": 100,
            }
        ]
        source = {
            "task": "tsp",
            "label": label,
            "files": files,
            "file_manifest_sha256": _digest(files),
            "rows": 100,
            "unique_instance_hashes": 100,
        }
        source["source_identity_sha256"] = _digest(source)
        sources.append(source)
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": _digest(sources),
    }
    registry["registry_sha256"] = _digest(registry)
    historical = next(source for source in sources if source["label"] == "historical-eval")
    development = {
        "data_manifest_sha256": _digest(evaluation_files),
        "evaluation_data_files": evaluation_files,
        "forbidden_source_label": "historical-eval",
        "forbidden_source_identity_sha256": historical["source_identity_sha256"],
        "protocol_fingerprint": "1" * 64,
        "strict_verifier_manifest_sha256": "5" * 64,
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
    development["development_evaluation_sha256"] = _digest(development)
    return registry, development


def _selection_campaign_fixture() -> tuple[dict, list[tuple[str, int, str]]]:
    methods = ["dual_scalar_z", "clasp_canonical"]
    steps = [100, 200]
    registry, development = _selection_registry_and_development()
    frozen_policy = {
        "campaign_id": "campaign",
        "candidate_budget": 8,
        "checkpoint_steps": steps,
        "evidence_scope": "development_only",
        "expected_methods": methods,
        "expected_tasks": ["tsp"],
        "forbidden_source_registry": registry,
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "pass_at_k_absolute_band": 0.01,
        "require_complete_checkpoint_grid": True,
        "schema_version": 1,
        "selection_rule": list(aggregate.CHECKPOINT_SELECTION_RULE),
        "stage_id": "development-selection",
        "status": "frozen_before_development_results",
        "train_seeds": [42],
        "development_evaluation_by_task": {"tsp": development},
    }
    frozen_policy = _seal(frozen_policy, "policy_sha256")
    selections = {}
    for method_index, method in enumerate(methods):
        candidates = []
        for step, feasible, gap in ((100, 80, 0.1), (200, 81, 0.05)):
            candidates.append(
                {
                    "checkpoint_step": step,
                    "candidate_budget": 8,
                    "instances": 100,
                    "feasible_instances": feasible,
                    "pass_at_k": feasible / 100,
                    "conditional_gap_n": feasible,
                    "conditional_gap_mean": gap,
                    "evaluation_fingerprint": hashlib.sha256(
                        f"eval-{method}-{step}".encode()
                    ).hexdigest(),
                    "protocol_fingerprint": "1" * 64,
                    "development_evaluation_sha256": development["development_evaluation_sha256"],
                    "checkpoint_artifact_sha256": hashlib.sha256(
                        f"checkpoint-{method}-{step}".encode()
                    ).hexdigest(),
                    "checkpoint_fidelity_identity_sha256": hashlib.sha256(
                        f"fidelity-{method}-{step}".encode()
                    ).hexdigest(),
                    "checkpoint_fidelity_receipt_sha256": hashlib.sha256(
                        f"fidelity-receipt-{method}-{step}".encode()
                    ).hexdigest(),
                    "checkpoint_fidelity_receipt_file_sha256": hashlib.sha256(
                        f"fidelity-file-{method}-{step}".encode()
                    ).hexdigest(),
                    "grpo_beta": 0.0,
                    "fixed_kl_reference": None,
                    "exact_peft_kl_runtime": None,
                    "exact_peft_kl_post_step": None,
                    "run_contract_sha256": str(method_index + 2) * 64,
                    "data_manifest_sha256": development["data_manifest_sha256"],
                    "strict_verifier_manifest_sha256": "5" * 64,
                    "disposition": (
                        "selected" if step == 200 else "eligible_but_higher_gap_or_later_exact_tie"
                    ),
                }
            )
        selected_candidate = candidates[1]
        selections[f"tsp/{method}/seed42"] = {
            "status": "PASS",
            "selected": {
                "task": "tsp",
                "method": method,
                "train_seed": 42,
                **{key: value for key, value in selected_candidate.items() if key != "disposition"},
            },
            "selection_audit": {
                "best_pass_at_k": 0.81,
                "eligibility_threshold": 0.8,
                "eligible_checkpoint_steps": steps,
                "candidates": candidates,
            },
            "errors": [],
        }
    payload = {
        "schema_version": 2,
        "selector": "clasp_development_checkpoint_selection",
        "generated_at_utc": "2026-08-14T00:00:00Z",
        "campaign_id": "campaign",
        "stage_id": "development-selection",
        "decision": "SELECTED",
        "evidence_scope": "development_only",
        "confirmation_evidence_used": False,
        "policy": {
            "path": "/frozen/policy.json",
            "file_sha256": "6" * 64,
            "policy_sha256": frozen_policy["policy_sha256"],
            "frozen_at_utc": frozen_policy["frozen_at_utc"],
            "selection_rule": list(aggregate.CHECKPOINT_SELECTION_RULE),
            "pass_at_k_absolute_band": 0.01,
            "frozen_policy": frozen_policy,
        },
        "evidence": {
            "path": "/frozen/evidence.json",
            "size_bytes": 100,
            "file_sha256": "7" * 64,
            "evidence_sha256": "8" * 64,
            "record_count": 4,
        },
        "expected_group_count": 2,
        "selected_group_count": 2,
        "selections": selections,
        "errors": [],
        "side_effects": {
            "checkpoints_modified": False,
            "evaluator_state_modified": False,
            "scheduler_actions": [],
        },
    }
    payload = _seal(payload, "receipt_sha256")
    serialized = aggregate._selector_receipt_file_bytes(payload)
    identity = {
        "basename": "selection.json",
        "file_sha256": hashlib.sha256(serialized).hexdigest(),
        "size_bytes": len(serialized),
        "receipt_sha256": payload["receipt_sha256"],
        "campaign_id": "campaign",
        "stage_id": "development-selection",
    }
    manifest = {
        "methods": methods,
        "checkpoint_selection_receipt": {"identity": identity, "payload": payload},
    }
    grid = [(method, 42, "tsp") for method in methods]
    return manifest, grid


def _reseal_selection_campaign(manifest: dict) -> None:
    payload = _seal(
        manifest["checkpoint_selection_receipt"]["payload"],
        "receipt_sha256",
    )
    serialized = aggregate._selector_receipt_file_bytes(payload)
    manifest["checkpoint_selection_receipt"]["payload"] = payload
    manifest["checkpoint_selection_receipt"]["identity"].update(
        {
            "file_sha256": hashlib.sha256(serialized).hexdigest(),
            "size_bytes": len(serialized),
            "receipt_sha256": payload["receipt_sha256"],
        }
    )


def _reseal_selection_policy(manifest: dict) -> None:
    descriptor = manifest["checkpoint_selection_receipt"]["payload"]["policy"]
    frozen_policy = _seal(descriptor["frozen_policy"], "policy_sha256")
    descriptor["frozen_policy"] = frozen_policy
    descriptor["policy_sha256"] = frozen_policy["policy_sha256"]
    descriptor["frozen_at_utc"] = frozen_policy["frozen_at_utc"]
    _reseal_selection_campaign(manifest)


def test_selection_receipt_closes_policy_grid_audit_and_file_identity() -> None:
    manifest, grid = _selection_campaign_fixture()
    registry = manifest["checkpoint_selection_receipt"]["payload"]["policy"]["frozen_policy"][
        "forbidden_source_registry"
    ]
    registry_identity = aggregate.forbidden_source_registry_identity(registry)
    aggregate._validate_campaign_selection_receipt(
        manifest,
        expected_grid=grid,
        campaign_id="campaign",
        expected_candidate_budget=8,
        expected_forbidden_source_registry_identity=registry_identity,
    )
    omitted = deepcopy(manifest)
    omitted["checkpoint_selection_receipt"]["payload"]["selections"]["tsp/dual_scalar_z/seed42"][
        "selection_audit"
    ]["candidates"].pop(0)
    _reseal_selection_campaign(omitted)
    with pytest.raises(ValueError, match="omits a scheduled checkpoint"):
        aggregate._validate_campaign_selection_receipt(
            omitted,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )

    wrong_file_identity = deepcopy(manifest)
    wrong_file_identity["checkpoint_selection_receipt"]["identity"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity/payload mismatch"):
        aggregate._validate_campaign_selection_receipt(
            wrong_file_identity,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )


def test_aggregate_exact_kl_post_step_projection_is_required_iff_beta_positive() -> None:
    assert aggregate._validate_exact_post_step_projection(
        0.0,
        None,
        None,
        None,
        label="beta-zero fixture",
    ) == (0.0, None, None, None)
    with pytest.raises(ValueError, match="beta=0 must not bind exact PEFT KL post-step"):
        aggregate._validate_exact_post_step_projection(
            0.0,
            None,
            None,
            {"stale": True},
            label="beta-zero fixture",
        )
    with pytest.raises(ValueError, match="fixed-KL identity is invalid"):
        aggregate._validate_exact_post_step_projection(
            0.01,
            None,
            None,
            None,
            label="positive fixture",
        )


def test_selection_receipt_closes_registry_and_development_file_bindings() -> None:
    manifest, grid = _selection_campaign_fixture()
    registry = manifest["checkpoint_selection_receipt"]["payload"]["policy"]["frozen_policy"][
        "forbidden_source_registry"
    ]
    wrong_registry_identity = aggregate.forbidden_source_registry_identity(registry)
    wrong_registry_identity["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="disagrees with the confirmation benchmark"):
        aggregate._validate_campaign_selection_receipt(
            manifest,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
            expected_forbidden_source_registry_identity=wrong_registry_identity,
        )

    missing_files = deepcopy(manifest)
    development = missing_files["checkpoint_selection_receipt"]["payload"]["policy"][
        "frozen_policy"
    ]["development_evaluation_by_task"]["tsp"]
    development.pop("evaluation_data_files")
    _reseal_selection_policy(missing_files)
    with pytest.raises(ValueError, match="development contract is invalid"):
        aggregate._validate_campaign_selection_receipt(
            missing_files,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )

    unregistered_bytes = deepcopy(manifest)
    contracts = unregistered_bytes["checkpoint_selection_receipt"]["payload"]["policy"][
        "frozen_policy"
    ]["development_evaluation_by_task"]
    development = contracts["tsp"]
    development["evaluation_data_files"][0]["sha256"] = "9" * 64
    development["data_manifest_sha256"] = _digest(development["evaluation_data_files"])
    contracts["tsp"] = _seal(development, "development_evaluation_sha256")
    _reseal_selection_policy(unregistered_bytes)
    with pytest.raises(ValueError, match="bytes differ from registered historical-eval"):
        aggregate._validate_campaign_selection_receipt(
            unregistered_bytes,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )

    wrong_development = deepcopy(manifest)
    selected_group = wrong_development["checkpoint_selection_receipt"]["payload"]["selections"][
        "tsp/clasp_canonical/seed42"
    ]
    selected_group["selected"]["development_evaluation_sha256"] = "0" * 64
    for candidate in selected_group["selection_audit"]["candidates"]:
        candidate["development_evaluation_sha256"] = "0" * 64
    _reseal_selection_campaign(wrong_development)
    with pytest.raises(ValueError, match="frozen development corpus/slice"):
        aggregate._validate_campaign_selection_receipt(
            wrong_development,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )

    wrong_protocol = deepcopy(manifest)
    development = wrong_protocol["checkpoint_selection_receipt"]["payload"]["policy"][
        "frozen_policy"
    ]["development_evaluation_by_task"]["tsp"]
    development["protocol_fingerprint"] = "9" * 64
    wrong_protocol["checkpoint_selection_receipt"]["payload"]["policy"]["frozen_policy"][
        "development_evaluation_by_task"
    ]["tsp"] = _seal(development, "development_evaluation_sha256")
    _reseal_selection_policy(wrong_protocol)
    with pytest.raises(ValueError, match="frozen development"):
        aggregate._validate_campaign_selection_receipt(
            wrong_protocol,
            expected_grid=grid,
            campaign_id="campaign",
            expected_candidate_budget=8,
        )


def test_matched_training_design_rejects_unequal_budget_or_shared_config(
    tmp_path: Path,
) -> None:
    _spec, _details, config_path, _summary = _native_fixture(tmp_path)
    challenger = aggregate._load_json_object(str(config_path))["training_run_contract"]
    baseline = deepcopy(challenger)
    baseline["method"] = "dual_scalar_z"
    baseline["training_config"].update(
        {
            "policy_credit_formula": "group_z_scalarized_positive_component_penalty_v1",
            "component_controller": "legacy_projected_ema_component_gap",
            "set_controller": "off",
            "use_clasp": False,
            "clasp_canonical_set": False,
            "pd_group_scalarization": True,
            "pd_dual_lr": 0.05,
            "pd_dual_lr_up": 0.05,
            "pd_dual_lr_down": 0.05,
            "clasp_bon_dual_lr": 0.0,
            "clasp_canonical_recovery_scale": 0.0,
        }
    )
    baseline["tracked_environment"].update(
        {
            "CLASP_BON_ALIGNED": "0",
            "CLASP_FEASIBILITY_GATED": "0",
            "CLASP_CANONICAL_SET": "0",
            "CLASP_BON_DUAL_LR": "0.0",
            "CLASP_CANONICAL_RECOVERY_SCALE": "0.0",
        }
    )
    methods = ["dual_scalar_z", "clasp_canonical"]
    contracts = {
        ("tsp", 42): {"dual_scalar_z": baseline, "clasp_canonical": challenger}
    }
    shared_hash = aggregate._shared_training_design_sha256(challenger)
    manifest = {
        "training_design": {
            "schema_version": 1,
            "fixed_horizon": {"max_steps": 3200, "world_size": 4},
            "method_specific_allowlist": list(aggregate.METHOD_SPECIFIC_TRAINING_ALLOWLIST),
            "shared_design_sha256_by_task_seed": {"tsp/seed42": shared_hash},
            "fixed_null_receipt_sha256_by_task": {},
        }
    }
    aggregate._validate_matched_training_design(
        manifest,
        contracts_by_group=contracts,
        methods=methods,
        tasks=["tsp"],
        seeds=[42],
    )

    unequal = deepcopy(contracts)
    unequal[("tsp", 42)]["dual_scalar_z"]["fixed_horizon"]["max_steps"] = 1600
    with pytest.raises(ValueError, match="unequal training horizon"):
        aggregate._validate_matched_training_design(
            manifest,
            contracts_by_group=unequal,
            methods=methods,
            tasks=["tsp"],
            seeds=[42],
        )

    unmatched = deepcopy(contracts)
    unmatched[("tsp", 42)]["dual_scalar_z"]["training_config"]["learning_rate"] = 2e-6
    with pytest.raises(ValueError, match="shared training design differs"):
        aggregate._validate_matched_training_design(
            manifest,
            contracts_by_group=unmatched,
            methods=methods,
            tasks=["tsp"],
            seeds=[42],
        )


def test_matched_set_null_design_binds_approved_receipt_and_normalizes_only_its_path(
    tmp_path: Path,
) -> None:
    _spec, _details, config_path, _summary = _native_fixture(tmp_path)
    baseline = aggregate._load_json_object(str(config_path))["training_run_contract"]
    set_null = deepcopy(baseline)
    set_null["method"] = "set_null"
    receipt_sha256 = aggregate.APPROVED_FIXED_NULL_RECEIPT_SHA256_BY_TASK["tsp"]
    set_null["fixed_null_receipt"] = {"receipt_sha256": receipt_sha256}
    set_null["tracked_environment"]["CLASP_FIXED_NULL_RECEIPT_PATH"] = (
        "/frozen/fixed-null/tsp.json"
    )
    methods = ["clasp_canonical", "set_null"]
    contracts = {
        ("tsp", 42): {"clasp_canonical": baseline, "set_null": set_null},
    }
    shared_hash = aggregate._shared_training_design_sha256(baseline)
    manifest = {
        "training_design": {
            "schema_version": 1,
            "fixed_horizon": {"max_steps": 3200, "world_size": 4},
            "method_specific_allowlist": list(aggregate.METHOD_SPECIFIC_TRAINING_ALLOWLIST),
            "shared_design_sha256_by_task_seed": {"tsp/seed42": shared_hash},
            "fixed_null_receipt_sha256_by_task": {"tsp": receipt_sha256},
        }
    }
    aggregate._validate_matched_training_design(
        manifest,
        contracts_by_group=contracts,
        methods=methods,
        tasks=["tsp"],
        seeds=[42],
    )

    wrong_contract = deepcopy(contracts)
    wrong_contract[("tsp", 42)]["set_null"]["fixed_null_receipt"]["receipt_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="campaign preregistration"):
        aggregate._validate_matched_training_design(
            manifest,
            contracts_by_group=wrong_contract,
            methods=methods,
            tasks=["tsp"],
            seeds=[42],
        )

    non_method_drift = deepcopy(contracts)
    non_method_drift[("tsp", 42)]["set_null"]["training_config"]["learning_rate"] = 9e-6
    with pytest.raises(ValueError, match="shared training design differs"):
        aggregate._validate_matched_training_design(
            manifest,
            contracts_by_group=non_method_drift,
            methods=methods,
            tasks=["tsp"],
            seeds=[42],
        )

def test_campaign_binds_training_authorization_and_approved_initialization(
    tmp_path: Path,
) -> None:
    _spec, _details, config_path, _summary = _native_fixture(tmp_path)
    contract = deepcopy(aggregate._load_json_object(str(config_path))["training_run_contract"])
    contract["training_config"].update(
        {
            "campaign_id": "campaign",
            "advancement_policy_sha256": "a" * 64,
            "advancement_stage_id": "3200-step-selection",
            "advancement_stage_index": 0,
            "advancement_prior_stage_receipt_chain_sha256": _digest([]),
        }
    )
    contracts = {("tsp", 42): {"clasp_canonical": contract}}
    authorization_manifest = {
        "training_authorization": {
            "campaign_id": "campaign",
            "advancement_policy_sha256": "a" * 64,
            "advancement_stage_id": "3200-step-selection",
            "advancement_stage_index": 0,
            "prior_stage_receipt_chain": [],
            "prior_stage_receipt_chain_sha256": _digest([]),
        }
    }
    aggregate._validate_campaign_training_authorization(
        authorization_manifest,
        campaign_id="campaign",
        contracts_by_group=contracts,
    )
    wrong_campaign = deepcopy(contracts)
    wrong_campaign[("tsp", 42)]["clasp_canonical"]["training_config"]["campaign_id"] = (
        "post-hoc-campaign"
    )
    with pytest.raises(ValueError, match="another campaign/stage/policy"):
        aggregate._validate_campaign_training_authorization(
            authorization_manifest,
            campaign_id="campaign",
            contracts_by_group=wrong_campaign,
        )

    chained = deepcopy(authorization_manifest)
    chained["training_authorization"].update(
        {
            "advancement_stage_index": 1,
            "prior_stage_receipt_chain": [
                {
                    "campaign_id": "campaign",
                    "stage_id": "pilot",
                    "policy_sha256": "b" * 64,
                    "receipt_sha256": "c" * 64,
                    "receipt_file_sha256": "d" * 64,
                }
            ],
        }
    )
    chained["training_authorization"]["prior_stage_receipt_chain_sha256"] = _digest(
        chained["training_authorization"]["prior_stage_receipt_chain"]
    )
    chained_contracts = deepcopy(contracts)
    chained_config = chained_contracts[("tsp", 42)]["clasp_canonical"]["training_config"]
    chained_config["advancement_stage_index"] = 1
    chained_config["advancement_prior_stage_receipt_chain_sha256"] = chained[
        "training_authorization"
    ]["prior_stage_receipt_chain_sha256"]
    aggregate._validate_campaign_training_authorization(
        chained,
        campaign_id="campaign",
        contracts_by_group=chained_contracts,
    )
    chained["training_authorization"]["prior_stage_receipt_chain"][0]["receipt_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="training authorization is invalid"):
        aggregate._validate_campaign_training_authorization(
            chained,
            campaign_id="campaign",
            contracts_by_group=chained_contracts,
        )

    model_registry = deepcopy(contract["model_initialization_registry"])
    model_manifest = {"model_initialization_registry": model_registry}
    aggregate._validate_campaign_model_initialization(
        model_manifest,
        contracts_by_group=contracts,
    )
    wrong_model = deepcopy(contracts)
    wrong_model[("tsp", 42)]["clasp_canonical"]["fingerprints"]["base_model"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="model initialization"):
        aggregate._validate_campaign_model_initialization(
            model_manifest,
            contracts_by_group=wrong_model,
        )
