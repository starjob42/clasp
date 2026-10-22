from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from clasp.config import build_grpo_parser, finalize_grpo_args

ROOT = Path(__file__).resolve().parents[1]
FREEZER_PATH = ROOT / "scripts" / "freeze_tsp_fcdf_beta_pilot.py"
DRY_RUN_PATH = ROOT / "scripts" / "dry_run_tsp_fcdf_beta_pilot.py"
TEMPLATE_PATH = ROOT / "experiments/sota_20260814/tsp_fcdf_beta_pilot.template.json"
REAL_RELOCATION = (
    ROOT / "experiments/sota_20260814/local_development/"
    "tsp_sft_saved_policy_bias_all_frozen_base_v1"
)
REAL_RELOCATION_RECEIPT = REAL_RELOCATION / "adapter_base_relocation_receipt.json"
SEALED_SFT_ROOT = (
    ROOT / "experiments/sota_20260814/local_development/"
    "tsp_dense40_posttrain_k8_seed271828_v1/raw/sft/sft/"
    "local_tsp_sft500k_seed42_instruct"
)
SEALED_SFT_SUMMARY = SEALED_SFT_ROOT / "inference_summary.json"
SEALED_SFT_DETAILS = SEALED_SFT_ROOT / (
    "run_details__sft_only_local_tsp_sft500k_seed42_instruct_eval_k8_"
    "contracted_eval200_k8_seed271828_v1_fp81c9e70b005a.jsonl"
)
V1_SPEC = (
    ROOT / "experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v1.spec.json"
)
V1_CONTRACT = V1_SPEC.with_name("tsp_fcdf_beta_pilot_20260815_v1.contract.json")
V1_OUTPUT = ROOT / "outputs/tsp_fcdf_beta_pilot_20260815_v1"
V2_SPEC = V1_SPEC.with_name("tsp_fcdf_beta_pilot_20260815_v2.spec.json")
V2_CONTRACT = V1_SPEC.with_name("tsp_fcdf_beta_pilot_20260815_v2.contract.json")
V2_OUTPUT = ROOT / "outputs/tsp_fcdf_beta_pilot_20260815_v2"
V3_SPEC = V1_SPEC.with_name("tsp_fcdf_beta_pilot_20260815_v3.spec.json")
V3_CONTRACT = V1_SPEC.with_name("tsp_fcdf_beta_pilot_20260815_v3.contract.json")
V3_OUTPUT = ROOT / "outputs/tsp_fcdf_beta_pilot_20260815_v3"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


freezer = _load_module("freeze_tsp_fcdf_beta_pilot", FREEZER_PATH)
dry_run = _load_module("dry_run_tsp_fcdf_beta_pilot", DRY_RUN_PATH)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _make_read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o550 if path.is_dir() else 0o440)
    root.chmod(0o550)


def _inspection() -> dict[str, object]:
    records = [
        {"key": f"base_model.model.layer.{index}.lora_A.weight", "numel": 1} for index in range(392)
    ]
    records.extend(
        {
            "key": f"base_model.model.layer.{index}.base_layer.bias",
            "numel": 1,
        }
        for index in range(84)
    )
    records[0]["numel"] = 161_609_728 - (len(records) - 1)
    return {
        "adapter_config": {
            "peft_type": "LORA",
            "bias": "all",
            "r": 64,
            "lora_alpha": 64,
            "target_modules": sorted(freezer.EXPECTED_TARGET_MODULES),
        },
        "artifacts": {},
        "tensors": records,
    }


def _finalize_receipt(path: Path, receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = freezer.sha256_json(unsigned)
    _write_json(path, receipt)
    path.chmod(0o440)


def _build_fixture(tmp_path: Path) -> dict[str, object]:
    old_base = tmp_path / "old_base"
    old_base.mkdir()
    _write_json(old_base / "config.json", {"model_type": "qwen2"})
    (old_base / "model.safetensors").write_bytes(b"test-base-model-bytes")

    frozen_base = tmp_path / "frozen_base_with_tokenizer"
    frozen_base.mkdir()
    _write_json(frozen_base / "config.json", {"model_type": "qwen2"})
    (frozen_base / "model.safetensors").write_bytes(b"test-base-model-bytes")
    (frozen_base / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_json(frozen_base / "tokenizer_config.json", {"tokenizer_class": "FakeTokenizer"})

    legacy = tmp_path / "legacy_sft"
    legacy.mkdir()
    converted = tmp_path / "converted_bias_all"
    converted.mkdir()
    adapter_config = dict(_inspection()["adapter_config"])
    adapter_config["base_model_name_or_path"] = str(old_base.resolve())
    _write_json(converted / "adapter_config.json", adapter_config)
    (converted / "adapter_model.safetensors").write_bytes(b"test-adapter-bytes")
    conversion_receipt_path = converted / freezer.CONVERSION_RECEIPT
    conversion_receipt: dict[str, object] = {
        "schema_version": 1,
        "protocol": freezer.CONVERSION_PROTOCOL,
        "status": "converted_saved_policy_verified",
        "development_only": True,
        "recovers_historical_terminal_biases": False,
        "converted_saved_policy_fidelity": True,
        "terminal_sft_policy_fidelity": False,
        "publication_eligible": False,
        "sota_eligible": False,
        "publication_sota_eligible": False,
        "legacy_source": {"path": str(legacy.resolve())},
        "base": {
            "path": str(old_base.resolve()),
            "manifest": freezer.tree_manifest(old_base),
            "config_sha256": freezer.sha256_file(old_base / "config.json"),
        },
        "converted_output": {
            "path": str(converted.resolve()),
            "manifest_excluding_receipt": freezer.tree_manifest(converted),
            "serialization_bias_policy": "all",
            "checkpoint_fidelity_adapter_inspection": _inspection(),
        },
        "verification": {
            "canonical_adapter_state_bitwise_equal": True,
            "qkv_biases_equal_frozen_base": True,
            "deterministic_logits_bitwise_equal": True,
            "deterministic_log_probabilities_bitwise_equal": True,
        },
    }
    _finalize_receipt(conversion_receipt_path, conversion_receipt)
    _make_read_only_tree(converted)

    relocated = tmp_path / "relocated_bias_all"
    relocated.mkdir()
    relocated_config = dict(adapter_config)
    relocated_config["base_model_name_or_path"] = str(frozen_base.resolve())
    _write_json(relocated / "adapter_config.json", relocated_config)
    (relocated / "adapter_model.safetensors").write_bytes(b"test-adapter-bytes")
    relocation_receipt_path = relocated / freezer.RELOCATION_RECEIPT
    _write_json(relocation_receipt_path, {"fixture": True})
    _make_read_only_tree(relocated)
    _make_read_only_tree(frozen_base)

    train_file = tmp_path / "frozen_train/tsp/train/train_rl.json"
    train_file.parent.mkdir(parents=True)
    _write_json(
        train_file,
        [
            {
                "instance": [[index, 0], [index, 1]],
                "input": str(index),
                "output": "Route: [0, 1, 0], Objective: 2.000",
            }
            for index in range(3_200)
        ],
    )
    train_file.chmod(0o440)
    development_file = tmp_path / "frozen_development/tsp/eval/test.json"
    development_file.parent.mkdir(parents=True)
    _write_json(
        development_file,
        [
            {
                "instance": [[10_000 + index, 0], [10_000 + index, 1]],
                "input": str(index),
                "output": "Route: [0, 1, 0], Objective: 2.000",
            }
            for index in range(300)
        ],
    )
    development_file.chmod(0o440)

    output_root = tmp_path / "must_not_exist"
    frozen_contract_path = tmp_path / "frozen_contract.json"
    spec = {
        "schema_version": freezer.SCHEMA_VERSION,
        "protocol": freezer.SPEC_PROTOCOL,
        "status": "READY_TO_FREEZE",
        "campaign_id": "tsp_fcdf_beta_test_03",
        "paths": {
            "project_root": str(ROOT.resolve()),
            "python_executable": str(Path(sys.executable).absolute()),
            "relocated_development_adapter": str(relocated.resolve()),
            "relocation_receipt": str(relocation_receipt_path.resolve()),
            "tokenizer_path": str(frozen_base.resolve()),
            "rl_train_file": str(train_file.resolve()),
            "development_source_file": str(development_file.resolve()),
            "sealed_sft_summary": str(SEALED_SFT_SUMMARY.resolve()),
            "sealed_sft_details": str(SEALED_SFT_DETAILS.resolve()),
            "frozen_contract_path": str(frozen_contract_path.absolute()),
            "output_root": str(output_root.absolute()),
        },
    }
    return {
        "spec": spec,
        "converted": converted,
        "conversion_receipt_path": conversion_receipt_path,
        "relocated": relocated,
        "relocation_receipt_path": relocation_receipt_path,
        "frozen_base": frozen_base,
        "train_file": train_file,
        "development_file": development_file,
        "output_root": output_root,
        "frozen_contract_path": frozen_contract_path,
    }


def _install_relocation_stub(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, object]) -> None:
    def validate(adapter_path, receipt_path, tokenizer_path, *, project_root):
        assert project_root == ROOT.resolve()
        if adapter_path != fixture["relocated"]:
            freezer.fail("wrong relocated adapter")
        if receipt_path != fixture["relocation_receipt_path"]:
            freezer.fail("wrong relocation receipt")
        if tokenizer_path != fixture["frozen_base"] or tokenizer_path == adapter_path:
            freezer.fail("tokenizer_path must be the immutable frozen base/tokenizer")
        conversion = freezer.validate_conversion(
            fixture["converted"], fixture["conversion_receipt_path"]
        )
        relocation = {
            "schema_version": 1,
            "protocol": freezer.RELOCATION_PROTOCOL,
            "status": freezer.RELOCATION_STATUS,
            "adapter_path": str(adapter_path),
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": freezer.FROZEN_RELOCATION_RECEIPT_FILE_SHA256,
            "receipt_sha256": freezer.FROZEN_RELOCATION_RECEIPT_SHA256,
            "frozen_base_path": str(tokenizer_path),
            "tokenizer_path": str(tokenizer_path),
            "relocation_verification_dtype": "float32",
            "relocated_output_tree_sha256": freezer.FROZEN_RELOCATED_OUTPUT_TREE_SHA256,
            "converted_source_tree_sha256": freezer.FROZEN_CONVERTED_SOURCE_TREE_SHA256,
            "converted_payload_tree_sha256": freezer.FROZEN_CONVERTED_PAYLOAD_TREE_SHA256,
            "base_tree_sha256": freezer.FROZEN_BASE_TREE_SHA256,
            "state_inventory_sha256": freezer.FROZEN_RELOCATION_STATE_INVENTORY_SHA256,
            "probe_logits_sha256": freezer.FROZEN_RELOCATION_LOGITS_SHA256,
            "producer": {
                "protocol": freezer.RELOCATION_VERIFIER_PROTOCOL,
                "implementation": "scripts.relocate_development_adapter_base",
                "implementation_sha256": freezer.FROZEN_RELOCATION_IMPLEMENTATION_SHA256,
            },
            "exact_peft_source_binding": {
                "adapter_path": str(adapter_path),
                "base_model_path": str(tokenizer_path),
                "tokenizer_path": str(tokenizer_path),
            },
            "claims": {
                "converted_saved_policy_fidelity": True,
                "terminal_sft_policy_fidelity": False,
                "publication_eligible": False,
                "sota_eligible": False,
            },
        }
        return relocation, conversion

    monkeypatch.setattr(freezer, "validate_relocated_adapter", validate)
    monkeypatch.setattr(
        freezer,
        "probe_runtime_identity",
        lambda python_executable: {
            "versions": dict(freezer.EXPECTED_RUNTIME),
            "python_executable": str(python_executable),
            "python_resolved_executable": str(python_executable.resolve(strict=True)),
            "python_resolved_executable_sha256": freezer.sha256_file(
                python_executable.resolve(strict=True)
            ),
            "probe_method": "closed_environment_imported_package_versions_v1",
        },
    )
    monkeypatch.setattr(
        freezer,
        "probe_local_hardware",
        lambda python_executable: {
            **freezer.EXPECTED_LOCAL_HARDWARE,
            "probe_executable": {
                "entrypoint_path": str(freezer.NVIDIA_SMI_ENTRYPOINT),
                "entrypoint_is_symlink": True,
                "symlink_chain": [{"fixture": True}],
                "resolved_path": str(freezer.FROZEN_NVIDIA_SMI_RESOLVED_PATH),
                "resolved_sha256": freezer.FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
                "resolved_size_bytes": freezer.FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
                "resolved_mode": freezer.FROZEN_NVIDIA_SMI_RESOLVED_MODE,
                "resolved_owner_identity": freezer.TRUSTED_SYSTEM_OWNER_IDENTITY,
                "accepted_owner_ids": [
                    list(value) for value in sorted(freezer.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS)
                ],
                "resolved_executable": True,
                "resolved_writable_by_current_user": False,
            },
            "query": ["fixture"],
        },
    )


def _argument(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _live_attempt_material(command: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    return dry_run._authorized_inner_payload(command), dict(command["env"])


def _memory_gate_attempt_material(command: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    payload, environment = _live_attempt_material(command)
    environment.pop("CLASP_PILOT_CREATE_TMPDIR", None)
    return payload, environment


def _fixture_process_identity(pid: int = 98_765) -> dict[str, object]:
    return {
        "pid": pid,
        "state": "R",
        "parent_pid": max(1, pid - 1),
        "process_group_id": pid,
        "session_id": pid,
        "start_time_ticks": 12_345_678,
    }


def _prepare_memory_gate_runtime(fixture: dict[str, object]) -> None:
    gate_dir = Path(fixture["output_root"]) / dry_run.MEMORY_GATE_DIRNAME
    gate_dir.mkdir(parents=True)
    (gate_dir / "_runtime_tmp").mkdir()


def _safe_process_absence(
    spawned_identity: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "protocol": "clasp-memory-gate-operator-process-absence-v1",
        "procfs_root": "/proc",
        "double_inspection_exact_equal": True,
        "spawned_process_identity": spawned_identity,
        "original_process_status": "absent" if spawned_identity else "no_spawn_marker",
        "nonzombie_process_group_members": [],
        "authorized_payload_process_matches": [],
        "supervisor_wrapper_process_matches": [],
    }


def _safe_gpu_absence() -> dict[str, object]:
    return {
        "protocol": "clasp-memory-gate-operator-gpu-absence-v1",
        "double_inspection_exact_equal": True,
        "normalized_probe_executable": freezer.FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY,
        "query": ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        "stdout_sha256": freezer.sha256_json([]),
        "target_gpu_uuid": freezer.EXPECTED_LOCAL_HARDWARE["uuid"],
        "target_gpu_processes": [],
    }


def _create_memory_gate_execution_outcome(
    fixture: dict[str, object],
    contract: dict[str, object],
    *,
    status: str = dry_run.MEMORY_GATE_SUCCESS_STATUS,
    child: dict[str, object] | None = None,
    before_outcome: Callable[[Path], None] | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    contract_path = Path(fixture["frozen_contract_path"])
    gate_dir = Path(fixture["output_root"]) / dry_run.MEMORY_GATE_DIRNAME
    gate_dir.mkdir(parents=True)
    (gate_dir / "_runtime_tmp").mkdir()
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    claim, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    try:
        dry_run._write_descriptor_all(transcript_descriptor, b"fixture transcript\n")
        os.fsync(transcript_descriptor)
        if before_outcome is not None:
            before_outcome(gate_dir)
        if child is None:
            child = {
                "child_started": True,
                "exit_known": True,
                "returncode": 0,
                "exit_code": 0,
                "signal": None,
                "classification": "exited_zero",
                "error_type": None,
                "error_message": None,
                "process_identity": _fixture_process_identity(),
            }
        outcome = dry_run._persist_memory_gate_execution_outcome(
            contract,
            contract_path=contract_path,
            command=command,
            claim=claim,
            payload_argv=payload_argv,
            environment=environment,
            child=child,
            status=status,
            operator_finalized=False,
            post_execution_absence={
                "process_absence": _safe_process_absence(child.get("process_identity")),
                "gpu_absence": _safe_gpu_absence(),
            },
        )
    finally:
        os.close(transcript_descriptor)
    return Path(outcome["outcome_path"]), outcome, claim


def test_frozen_no_merge_source_and_artifact_identities() -> None:
    assert freezer.sha256_file(ROOT / "src/clasp/models/exact_peft_kl_reference.py") == (
        freezer.FROZEN_EXACT_PEFT_IMPLEMENTATION_SHA256
    )
    assert freezer.sha256_file(ROOT / "src/clasp/models/exact_peft_kl_post_step.py") == (
        freezer.FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256
    )
    assert freezer.sha256_file(ROOT / "src/clasp/models/loading.py") == (
        freezer.FROZEN_MODEL_LOADING_SHA256
    )
    assert freezer.sha256_file(ROOT / "scripts/relocate_development_adapter_base.py") == (
        freezer.FROZEN_RELOCATION_IMPLEMENTATION_SHA256
    )
    assert freezer.sha256_file(ROOT / "src/clasp/models/training_fidelity.py") == (
        freezer.FROZEN_TRAINING_FIDELITY_SHA256
    )
    assert freezer.sha256_file(ROOT / "src/clasp/entrypoints/train_grpo.py") == (
        freezer.FROZEN_TRAIN_GRPO_SHA256
    )
    receipt = freezer.load_json_object(REAL_RELOCATION_RECEIPT, label="real relocation")
    assert freezer.sha256_file(REAL_RELOCATION_RECEIPT) == (
        freezer.FROZEN_RELOCATION_RECEIPT_FILE_SHA256
    )
    assert receipt["receipt_sha256"] == freezer.FROZEN_RELOCATION_RECEIPT_SHA256
    assert receipt["verifier"]["implementation_sha256"] == (
        freezer.FROZEN_RELOCATION_IMPLEMENTATION_SHA256
    )
    assert receipt["frozen_base"]["manifest"]["tree_sha256"] == freezer.FROZEN_BASE_TREE_SHA256
    assert receipt["relocated_output"]["manifest_excluding_receipt"]["tree_sha256"] == (
        freezer.FROZEN_RELOCATED_OUTPUT_TREE_SHA256
    )
    assert receipt["upstream_conversion"]["source_adapter_manifest"]["tree_sha256"] == (
        freezer.FROZEN_CONVERTED_SOURCE_TREE_SHA256
    )
    upstream_receipt = freezer.load_json_object(
        Path(receipt["upstream_conversion"]["authoritative_receipt_path"]),
        label="upstream conversion",
    )
    assert (
        upstream_receipt["converted_output"]["manifest_excluding_receipt"]["tree_sha256"]
        == freezer.FROZEN_CONVERTED_PAYLOAD_TREE_SHA256
    )
    assert receipt["verification"]["requested_dtype"] == "float32"
    assert receipt["verification"]["full_model_state"]["tensor_inventory_sha256"] == (
        freezer.FROZEN_RELOCATION_STATE_INVENTORY_SHA256
    )
    assert receipt["verification"]["functional_equivalence"]["logits"]["value_sha256"] == (
        freezer.FROZEN_RELOCATION_LOGITS_SHA256
    )


def test_historical_v1_v2_failures_are_untouched_and_v3_paths_are_absent() -> None:
    assert freezer.sha256_file(V1_SPEC) == (
        "cc10f2982aefbe9a3eeba05904426d521cd2a1c01c0b0f69a3e5a26c67cc46e8"
    )
    assert freezer.sha256_file(V1_CONTRACT) == (
        "60468e72e21bc30513cf49375e4683e1173e22864918ec22a57c34312127a998"
    )
    assert (
        freezer.load_json_object(V1_CONTRACT, label="historical v1 contract")["contract_sha256"]
        == "284addf001628633eb56059fc6f40bd3c981bfa95e579bfb303a8d751c729827"
    )
    gate = V1_OUTPUT / dry_run.MEMORY_GATE_DIRNAME
    claim = gate / dry_run.CLAIM_ANCHOR_DIRNAME / dry_run.MEMORY_GATE_ATTEMPT_FILENAME
    hyperparameters = gate / "hyperparams.json"
    assert freezer.sha256_file(claim) == (
        "5346809fbafacef5282d8e23f6060630d1bc8ab998eae5e0c07aef303eccb2f9"
    )
    assert freezer.sha256_file(hyperparameters) == (
        "fabc28f90621510081cb9b8311e5dd7ca07ca7a6947e56c4d0a154d8b1c21411"
    )
    assert stat.S_IMODE(claim.stat().st_mode) == 0o440
    assert stat.S_IMODE(hyperparameters.stat().st_mode) == 0o644
    assert {path.name for path in gate.iterdir()} == {
        dry_run.CLAIM_ANCHOR_DIRNAME,
        "_runtime_tmp",
        "checkpoints",
        "hyperparams.json",
    }
    assert not any((gate / "checkpoints").iterdir())
    assert not (V1_OUTPUT / dry_run.MEMORY_GATE_PASS_FILENAME).exists()
    assert not (V1_OUTPUT / dry_run.MATRIX_AUTHORIZATION_FILENAME).exists()
    assert not (V1_OUTPUT / dry_run.RESERVATION_CLOSURE_FILENAME).exists()
    assert freezer.sha256_file(V2_SPEC) == (
        "6f2402c2b578db17dc00b985059a059ba28a9001f89ea897d24f180bf667685a"
    )
    assert freezer.sha256_file(V2_CONTRACT) == (
        "2d11b2b6367ec419536cf5240519b18c04f69078252a1e23e4f62b23ddd3250f"
    )
    assert (
        freezer.load_json_object(V2_CONTRACT, label="historical v2 contract")["contract_sha256"]
        == "614035ede9517ab2f8ebee0abe9ec2738da0978800c795dec22fb6d139d4aec1"
    )
    assert stat.S_IMODE(V2_SPEC.stat().st_mode) == 0o444
    assert stat.S_IMODE(V2_CONTRACT.stat().st_mode) == 0o440
    v2_gate = V2_OUTPUT / dry_run.MEMORY_GATE_DIRNAME
    v2_anchors = v2_gate / dry_run.CLAIM_ANCHOR_DIRNAME
    v2_claim = v2_anchors / dry_run.MEMORY_GATE_ATTEMPT_FILENAME
    v2_transcript = v2_anchors / dry_run.MEMORY_GATE_TRANSCRIPT_FILENAME
    v2_outcome = v2_anchors / dry_run.MEMORY_GATE_OUTCOME_FILENAME
    assert freezer.sha256_file(v2_claim) == (
        "5d9b087f48adf3217f1cf3b5db9cad79afc890c6e9d965eeae822ff0c72cc5ae"
    )
    assert freezer.sha256_file(v2_transcript) == (
        "338c2f60cd7281c314d7e4b7a4cb8493acd8f862eca06267110b846d043ef8d7"
    )
    assert freezer.sha256_file(v2_outcome) == (
        "f0915d11febe0b8d962e648835daadcf060e8005f3702128c7eaf1dbb3dad219"
    )
    outcome = freezer.load_json_object(v2_outcome, label="historical v2 outcome")
    assert outcome["status"] == dry_run.MEMORY_GATE_FAILURE_STATUS
    assert outcome["receipt_sha256"] == (
        "3030e11af13287e743a5e79b0db95844b8af0b365d5f7a41cb52a3f8e669355a"
    )
    assert outcome["child"]["classification"] == "nonzero_exit"
    assert outcome["child"]["exit_code"] == 1
    assert not any((v2_gate / "checkpoints").iterdir())
    assert not (V2_OUTPUT / dry_run.MEMORY_GATE_PASS_FILENAME).exists()
    assert not (V2_OUTPUT / dry_run.MATRIX_AUTHORIZATION_FILENAME).exists()
    assert not (V2_OUTPUT / dry_run.RESERVATION_CLOSURE_FILENAME).exists()
    assert stat.S_IMODE(V2_OUTPUT.stat().st_mode) == 0o550
    assert not V3_SPEC.exists()
    assert not V3_CONTRACT.exists()
    assert not V3_OUTPUT.exists()


def test_nvidia_smi_alternatives_symlink_resolves_to_frozen_regular_executable() -> None:
    assert freezer.normalize_executable_owner(
        0,
        0,
        accepted_owner_ids=freezer.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
        label="direct host",
    ) == freezer.normalize_executable_owner(
        65_534,
        65_534,
        accepted_owner_ids=freezer.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
        label="managed namespace",
    )
    identity = freezer.resolve_executable_identity(
        freezer.NVIDIA_SMI_ENTRYPOINT,
        label="test nvidia-smi",
        expected_resolved_path=freezer.FROZEN_NVIDIA_SMI_RESOLVED_PATH,
        expected_sha256=freezer.FROZEN_NVIDIA_SMI_RESOLVED_SHA256,
        expected_size_bytes=freezer.FROZEN_NVIDIA_SMI_RESOLVED_SIZE_BYTES,
        expected_mode=freezer.FROZEN_NVIDIA_SMI_RESOLVED_MODE,
        accepted_owner_ids=freezer.FROZEN_NVIDIA_SMI_ACCEPTED_OWNER_IDS,
    )
    assert identity["entrypoint_is_symlink"] is True
    assert [record["kind"] for record in identity["symlink_chain"]] == [
        "symlink",
        "symlink",
        "regular_file",
    ]
    assert identity["resolved_path"] == str(freezer.FROZEN_NVIDIA_SMI_RESOLVED_PATH)
    assert identity["resolved_sha256"] == freezer.FROZEN_NVIDIA_SMI_RESOLVED_SHA256
    assert identity["resolved_writable_by_current_user"] is False
    assert freezer.normalized_executable_identity(identity) == (
        freezer.FROZEN_NVIDIA_SMI_NORMALIZED_IDENTITY
    )
    assert identity["resolved_owner_identity"] == freezer.TRUSTED_SYSTEM_OWNER_IDENTITY
    assert all(
        record["owner_identity"] == freezer.TRUSTED_SYSTEM_OWNER_IDENTITY
        and "uid" not in record
        and "gid" not in record
        for record in identity["symlink_chain"]
    )


def test_executable_identity_accepts_safe_two_hop_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real" / "nvidia-smi"
    target.parent.mkdir()
    target.write_bytes(b"safe-test-executable")
    target.chmod(0o555)
    alternative = tmp_path / "alternative"
    alternative.symlink_to(target)
    entrypoint = tmp_path / "nvidia-smi"
    entrypoint.symlink_to(alternative)
    identity = freezer.resolve_executable_identity(
        entrypoint,
        label="fixture executable",
        expected_resolved_path=target,
        expected_sha256=freezer.sha256_file(target),
        expected_size_bytes=target.stat().st_size,
        expected_mode=0o555,
    )
    assert identity["resolved_path"] == str(target)
    assert [record["kind"] for record in identity["symlink_chain"]] == [
        "symlink",
        "symlink",
        "regular_file",
    ]


def test_executable_identity_rejects_broken_nonregular_writable_and_drift(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    with pytest.raises(freezer.ContractError, match="broken"):
        freezer.resolve_executable_identity(broken, label="broken executable")

    directory = tmp_path / "directory"
    directory.mkdir()
    nonregular = tmp_path / "nonregular"
    nonregular.symlink_to(directory)
    with pytest.raises(freezer.ContractError, match="not a regular file"):
        freezer.resolve_executable_identity(nonregular, label="nonregular executable")

    writable_target = tmp_path / "writable-target"
    writable_target.write_bytes(b"writable")
    writable_target.chmod(0o755)
    writable = tmp_path / "writable"
    writable.symlink_to(writable_target)
    with pytest.raises(freezer.ContractError, match="writable by the current user"):
        freezer.resolve_executable_identity(writable, label="writable executable")

    safe_target = tmp_path / "safe-target"
    safe_target.write_bytes(b"safe")
    safe_target.chmod(0o555)
    drifted = tmp_path / "drifted"
    drifted.symlink_to(safe_target)
    with pytest.raises(freezer.ContractError, match="hash drift"):
        freezer.resolve_executable_identity(
            drifted,
            label="drifted executable",
            expected_sha256="0" * 64,
        )
    with pytest.raises(freezer.ContractError, match="untrusted owner mapping"):
        freezer.resolve_executable_identity(
            drifted,
            label="owner-drifted executable",
            accepted_owner_ids=frozenset({(0, 0)}),
        )

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.symlink_to(right)
    right.symlink_to(left)
    with pytest.raises(freezer.ContractError, match="cycle"):
        freezer.resolve_executable_identity(left, label="cyclic executable")


def test_contract_and_dry_run_bind_continued_adapter_matrix_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])

    assert contract["status"] == "frozen_before_development_results"
    assert contract["evidence_boundary"]["development_only"] is True
    assert contract["evidence_boundary"]["publication_eligible"] is False
    assert contract["evidence_boundary"]["sota_eligible"] is False
    assert contract["evidence_boundary"]["initialization_claim"] == (
        "common_relocated_bias_all_adapter_loaded_bfloat16"
    )
    assert contract["evidence_boundary"]["merged_reference_or_policy_used"] is False
    assert contract["policy_initialization"]["model_path"] == str(fixture["relocated"])
    assert contract["policy_initialization"]["tokenizer_path"] == str(fixture["frozen_base"])
    assert contract["policy_initialization"]["attach_fresh_rl_lora"] is False
    assert contract["policy_initialization"]["continue_existing_sft_lora"] is True
    assert contract["design"]["trajectory_count"] == 12
    assert contract["design"]["terminal_step"] == 400
    assert contract["design"]["per_device_train_batch_size"] == 1
    assert contract["design"]["gradient_accumulation_steps"] == 32
    assert contract["fixed_horizon"] == {"max_steps": 400, "world_size": 1}
    assert contract["memory_gate"]["current_readiness"] == (
        "CORE_FROZEN_REQUIRES_UNRUN_ONE_UPDATE_GATE"
    )
    assert (
        contract["sealed_sft_target_audit"]["observed"]["best_of_8_at_or_below_target_count"] == 128
    )

    exact_kl = contract["exact_peft_kl_contract"]
    assert exact_kl["beta_zero_reference"] is None
    assert exact_kl["beta_positive_reference_adapter_path"] == str(fixture["relocated"])
    assert exact_kl["runtime_protocol"] == freezer.EXACT_PEFT_DEVELOPMENT_PROTOCOL
    assert exact_kl["runtime_status"] == freezer.EXACT_PEFT_DEVELOPMENT_STATUS
    assert exact_kl["sync_ref_model"] is False
    assert exact_kl["use_vllm"] is False
    assert exact_kl["policy_adapter_autocast_dtype"] is True
    assert exact_kl["reference_adapter_autocast_dtype"] is True

    plan = dry_run.build_plan(
        contract,
        contract_path=fixture["frozen_contract_path"],
        include_cell_commands=True,
    )
    assert plan["status"] == "READY_FOR_UNRUN_ONE_UPDATE_ENGINEERING_GATE"
    assert plan["global_launch_authorized"] is False
    assert plan["training_or_evaluation_started"] is False
    pre_gate = dry_run.build_plan(
        contract,
        contract_path=fixture["frozen_contract_path"],
    )
    assert not fixture["frozen_contract_path"].exists()
    assert not fixture["output_root"].exists()
    assert pre_gate["training_commands"] == []
    assert pre_gate["fidelity_audit_commands"] == []
    assert pre_gate["evaluation_commands"] == []
    assert plan["counts"] == {
        "relocation_authorization_commands": 1,
        "memory_gate_training_commands": 1,
        "post_step_reference_audit_commands": 1,
        "training_commands": 12,
        "fidelity_audit_commands": 12,
        "evaluation_commands": 12,
    }
    authorization = plan["relocation_authorization_command"]
    assert authorization["argv"][-1] == "--verify-only"
    assert _argument(authorization["argv"], "--output-dir") == str(fixture["relocated"])

    gate = plan["memory_gate_command"]
    assert gate["launch_authorized"] is True
    assert gate["env"]["GRPO_STOP_AT_GLOBAL_STEP"] == "1"
    assert gate["env"]["CLASP_KL_REFERENCE_MODEL_PATH"] == str(fixture["relocated"])
    assert gate["env"]["CLASP_KL_REFERENCE_RECEIPT"] == str(fixture["relocation_receipt_path"])
    assert _argument(gate["argv"], "--max_steps") == "400"
    assert _argument(gate["argv"], "--warmup_ratio") == "0"
    assert _argument(gate["argv"], "--per_device_train_batch_size") == "1"
    assert _argument(gate["argv"], "--gradient_accumulation_steps") == "32"
    assert _argument(gate["argv"], "--grpo_beta") == "0.05"
    assert gate["rendered"].startswith("env -i ")
    assert "supervise_memory_gate_attempt" in " ".join(gate["argv"])
    assert "verify_attempt_execution_authority" in " ".join(gate["argv"])
    assert "apply_claimed_resume_environment" in " ".join(gate["argv"])
    guard_code = gate["argv"][2]
    attempt_guard_code = gate["argv"][6]
    compile(guard_code, "<runtime-guard>", "exec")
    compile(attempt_guard_code, "<attempt-guard>", "exec")
    assert "os.statvfs(tmp)" in guard_code
    assert "tempfile.gettempdir()==str(tmp)" in guard_code
    assert "tmp.relative_to(workspace)" in guard_code
    assert str(freezer.MIN_WORKSPACE_FREE_BYTES) in guard_code
    assert str(freezer.FROZEN_NVIDIA_SMI_RESOLVED_PATH) in guard_code
    assert freezer.FROZEN_NVIDIA_SMI_RESOLVED_SHA256 in guard_code
    assert freezer.TRUSTED_SYSTEM_OWNER_IDENTITY in guard_code
    assert "raw=subprocess.run([str(smi)" in guard_code
    assert "RESUME_FROM_CHECKPOINT" not in attempt_guard_code
    assert "apply_claimed_resume_environment" in attempt_guard_code

    assert len({command["trajectory_id"] for command in plan["training_commands"]}) == 12
    for command in plan["training_commands"]:
        argv = command["argv"]
        environment = command["env"]
        assert command["launch_authorized"] is False
        assert command["requires_successful_phase"] == (
            "one_step_beta_memory_gate_with_post_step_reference_audit"
        )
        assert _argument(argv, "--model_path") == str(fixture["relocated"])
        assert _argument(argv, "--tokenizer_path") == str(fixture["frozen_base"])
        assert _argument(argv, "--max_steps") == "400"
        assert _argument(argv, "--warmup_ratio") == "0.03"
        assert _argument(argv, "--per_device_train_batch_size") == "1"
        assert _argument(argv, "--gradient_accumulation_steps") == "32"
        beta = float(_argument(argv, "--grpo_beta"))
        if command["cell"].startswith("F"):
            assert environment["CLASP_FRONTIER_CDF"] == "1"
            assert environment["CLASP_FRONTIER_CDF_RELATIVE_FAILED_RECOVERY"] == "1"
        else:
            assert environment["CLASP_FRONTIER_CDF"] == "0"
        if beta > 0:
            assert environment["CLASP_KL_REFERENCE_MODEL_PATH"] == str(fixture["relocated"])
            assert environment["CLASP_KL_REFERENCE_RECEIPT"] == str(
                fixture["relocation_receipt_path"]
            )
        else:
            assert set(dry_run.KL_ENV_NAMES).issubset(command["unset_env"])
            assert not set(dry_run.KL_ENV_NAMES) & set(environment)
        assert command["rendered"].startswith("env -i ")
        assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert environment[dry_run.TRAINING_RUN_CONTRACT_ENV].endswith(
            dry_run.MATRIX_AUTHORIZATION_FILENAME
        )
        tmpdir = Path(environment["TMPDIR"])
        assert environment["TMP"] == str(tmpdir)
        assert environment["TEMP"] == str(tmpdir)
        for name in (
            "HOME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_CACHE_PATH",
        ):
            Path(environment[name]).relative_to(tmpdir)
    for command in [*plan["fidelity_audit_commands"], *plan["evaluation_commands"]]:
        environment = command["env"]
        tmpdir = Path(environment["TMPDIR"])
        assert environment["TMP"] == str(tmpdir)
        assert environment["TEMP"] == str(tmpdir)
        for name in (
            "HOME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_CACHE_PATH",
        ):
            Path(environment[name]).relative_to(tmpdir)


def test_gate_pass_atomically_reserves_full_matrix_and_first_attempt_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = fixture["frozen_contract_path"]
    freezer.write_contract_create_once(contract_path, contract)
    output_root = fixture["output_root"]
    gate_dir = output_root / dry_run.MEMORY_GATE_DIRNAME
    outcome_path, outcome, _ = _create_memory_gate_execution_outcome(
        fixture,
        contract,
        before_outcome=lambda gate: (gate / "engineering.txt").write_text(
            "gate\n", encoding="utf-8"
        ),
    )
    gate_evidence = {
        "executor_zero_exit": dry_run._file_identity(outcome_path, outcome),
        "checkpoint_fidelity": {"receipt_sha256": "1" * 64},
        "policy_update": {"changed_trainable_tensor_count": 1},
        "reference_immutability": {"live_equals_pre_step_bitwise": True},
        "cuda": {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2},
    }
    monkeypatch.setattr(
        dry_run,
        "audit_memory_gate_checkpoint",
        lambda contract, *, contract_path, gate_dir: gate_evidence,
    )

    memory_pass, ledger, closure = dry_run.record_memory_gate_pass_and_matrix(
        contract,
        contract_path=contract_path,
    )
    assert memory_pass["status"] == dry_run.MEMORY_GATE_PASS_STATUS
    assert ledger["fixed_horizon"] == {"max_steps": 400, "world_size": 1}
    assert len(ledger["training_commands"]) == 12
    assert len(ledger["evaluation_commands"]) == 12
    assert closure["reservation_count"] == 24
    assert (output_root / dry_run.RESERVATION_CLOSURE_FILENAME).stat().st_mode & 0o222 == 0
    assert all(not (path.stat().st_mode & 0o222) for path in [gate_dir, *gate_dir.rglob("*")])
    for command in ledger["training_commands"]:
        run_dir = Path(command["output_dir"])
        assert {path.name for path in run_dir.iterdir()} == {
            dry_run.CELL_AUTHORIZATION_FILENAME,
            "_runtime_tmp",
            dry_run.CLAIM_ANCHOR_DIRNAME,
        }
        anchors = run_dir / dry_run.CLAIM_ANCHOR_DIRNAME
        assert stat.S_IMODE(anchors.stat().st_mode) == 0o550
        assert len(list(anchors.iterdir())) == 400
        assert all(path.stat().st_size == 0 for path in anchors.iterdir())
    for command in ledger["evaluation_commands"]:
        evaluation_root = Path(_argument(command["argv"], "--inference_root"))
        assert {path.name for path in evaluation_root.iterdir()} == {
            dry_run.EVALUATION_AUTHORIZATION_FILENAME,
            "_runtime_tmp",
            dry_run.CLAIM_ANCHOR_DIRNAME,
        }
        anchors = evaluation_root / dry_run.CLAIM_ANCHOR_DIRNAME
        assert stat.S_IMODE(anchors.stat().st_mode) == 0o550
        assert [path.name for path in anchors.iterdir()] == [dry_run.EVALUATION_ATTEMPT_FILENAME]

    trajectory_id = ledger["training_commands"][0]["trajectory_id"]
    training_command = ledger["training_commands"][0]
    training_argv, training_env = _live_attempt_material(training_command)
    ledger_path = output_root / dry_run.MATRIX_AUTHORIZATION_FILENAME
    monkeypatch.setenv(dry_run.TRAINING_RUN_CONTRACT_ENV, str(ledger_path))
    altered_argv = list(training_argv)
    altered_argv[altered_argv.index("--grpo_beta") + 1] = "0.99"
    with pytest.raises(freezer.ContractError, match="payload argv differs"):
        dry_run.claim_matrix_attempt(
            contract_path,
            trajectory_id=trajectory_id,
            kind="training",
            actual_payload_argv=altered_argv,
            actual_environment=training_env,
        )
    altered_env = dict(training_env)
    altered_env["UNAUTHORIZED_SCIENTIFIC_OVERRIDE"] = "1"
    with pytest.raises(freezer.ContractError, match="environment differs"):
        dry_run.claim_matrix_attempt(
            contract_path,
            trajectory_id=trajectory_id,
            kind="training",
            actual_payload_argv=training_argv,
            actual_environment=altered_env,
        )
    initial_anchor = (
        Path(training_command["output_dir"])
        / dry_run.CLAIM_ANCHOR_DIRNAME
        / dry_run.TRAINING_ATTEMPT_FILENAME
    )
    assert initial_anchor.stat().st_size == 0
    attempt = dry_run.claim_matrix_attempt(
        contract_path,
        trajectory_id=trajectory_id,
        kind="training",
        actual_payload_argv=training_argv,
        actual_environment=training_env,
    )
    assert attempt["status"] == "development_training_attempt_started"
    with pytest.raises(freezer.ContractError, match="surviving nonterminal receipted"):
        dry_run.claim_matrix_attempt(
            contract_path,
            trajectory_id=trajectory_id,
            kind="training",
            actual_payload_argv=training_argv,
            actual_environment=training_env,
        )

    evaluation_id = ledger["evaluation_commands"][0]["trajectory_id"]
    evaluation_command = ledger["evaluation_commands"][0]
    evaluation_argv, evaluation_env = _live_attempt_material(evaluation_command)
    monkeypatch.setattr(
        dry_run,
        "inspect_cell_terminal_audit",
        lambda *args, **kwargs: {"status": dry_run.TERMINAL_AUDIT_STATUS},
    )
    evaluation_attempt = dry_run.claim_matrix_attempt(
        contract_path,
        trajectory_id=evaluation_id,
        kind="evaluation",
        actual_payload_argv=evaluation_argv,
        actual_environment=evaluation_env,
    )
    assert evaluation_attempt["status"] == "development_evaluation_attempt_started"
    with pytest.raises(freezer.ContractError, match="already attempted"):
        dry_run.claim_matrix_attempt(
            contract_path,
            trajectory_id=evaluation_id,
            kind="evaluation",
            actual_payload_argv=evaluation_argv,
            actual_environment=evaluation_env,
        )

    assert stat.S_IMODE(initial_anchor.stat().st_mode) == 0o440
    with pytest.raises(PermissionError):
        initial_anchor.unlink()
    anchor_root = initial_anchor.parent
    encoded_attempt = initial_anchor.read_bytes()
    anchor_root.chmod(0o750)
    initial_anchor.unlink()
    initial_anchor.write_bytes(encoded_attempt)
    initial_anchor.chmod(0o440)
    anchor_root.chmod(0o550)
    with pytest.raises(freezer.ContractError, match="claim-anchor"):
        dry_run.inspect_reservation_closure(
            output_root / dry_run.RESERVATION_CLOSURE_FILENAME,
            ledger_path,
            ledger,
        )


def test_reservation_closure_rejects_deleted_and_recreated_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = fixture["frozen_contract_path"]
    freezer.write_contract_create_once(contract_path, contract)
    output_root = fixture["output_root"]
    outcome_path, outcome, _ = _create_memory_gate_execution_outcome(
        fixture,
        contract,
        before_outcome=lambda gate: (gate / "engineering.txt").write_text(
            "gate\n", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        dry_run,
        "audit_memory_gate_checkpoint",
        lambda contract, *, contract_path, gate_dir: {
            "gate": "valid",
            "executor_zero_exit": dry_run._file_identity(outcome_path, outcome),
        },
    )
    _, ledger, closure = dry_run.record_memory_gate_pass_and_matrix(
        contract,
        contract_path=contract_path,
    )
    ledger_path = output_root / dry_run.MATRIX_AUTHORIZATION_FILENAME
    closure_path = output_root / dry_run.RESERVATION_CLOSURE_FILENAME
    run_dir = Path(ledger["training_commands"][0]["output_dir"])
    displaced = run_dir.with_name(run_dir.name + "-displaced")
    run_dir.rename(displaced)
    run_dir.mkdir()
    marker = displaced / dry_run.CELL_AUTHORIZATION_FILENAME
    (run_dir / dry_run.CELL_AUTHORIZATION_FILENAME).write_bytes(marker.read_bytes())
    (run_dir / dry_run.CELL_AUTHORIZATION_FILENAME).chmod(0o440)
    (run_dir / "_runtime_tmp").mkdir()
    with pytest.raises(freezer.ContractError, match="deleted, recreated, or altered"):
        dry_run.inspect_reservation_closure(closure_path, ledger_path, ledger)


def test_contract_rejects_weakened_upstream_conversion_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    converted = fixture["converted"]
    receipt_path = fixture["conversion_receipt_path"]
    converted.chmod(0o700)
    receipt_path.chmod(0o600)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_sft_policy_fidelity"] = True
    _finalize_receipt(receipt_path, receipt)
    converted.chmod(0o550)
    with pytest.raises(freezer.ContractError, match="claim boundary mismatch"):
        freezer.build_contract(fixture["spec"])


def test_contract_rejects_adapter_as_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    fixture["spec"]["paths"]["tokenizer_path"] = str(fixture["relocated"])
    with pytest.raises(freezer.ContractError, match="immutable frozen base/tokenizer"):
        freezer.build_contract(fixture["spec"])


def test_contract_rejects_semantic_train_development_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    train_file = fixture["train_file"]
    development_file = fixture["development_file"]
    train_rows = json.loads(train_file.read_text(encoding="utf-8"))
    development_rows = json.loads(development_file.read_text(encoding="utf-8"))
    train_file.chmod(0o600)
    train_rows[0]["instance"] = development_rows[0]["instance"]
    _write_json(train_file, train_rows)
    train_file.chmod(0o440)
    with pytest.raises(freezer.ContractError, match="overlaps RL training data"):
        freezer.build_contract(fixture["spec"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("analysis", None, "selection plan differs"),
        ("memory_gate", None, "memory gate differs"),
        ("target_saturation_preregistration", None, "target-saturation plan differs"),
    ],
)
def test_contract_rejects_rehashed_preregistration_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    if field == "analysis":
        contract[field]["advancement_gate"]["cell_eligibility"]["required_mean_gap"] = "<=0.99"
    elif field == "memory_gate":
        contract[field]["required_before_any_400_step_cell"] = value
    else:
        contract[field]["target_gap"] = 0.009
    unsigned = dict(contract)
    unsigned.pop("contract_sha256")
    contract["contract_sha256"] = freezer.sha256_json(unsigned)
    with pytest.raises(freezer.ContractError, match=message):
        freezer.verify_contract(contract)


def test_freezer_is_create_once_and_template_is_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    spec_path = tmp_path / "filled_spec.json"
    _write_json(spec_path, fixture["spec"])
    contract_path = tmp_path / "frozen_contract.json"
    assert freezer.main(["--spec", str(spec_path), "--output", str(contract_path)]) == 0
    assert stat.S_IMODE(contract_path.stat().st_mode) == 0o440
    assert freezer.main(["--spec", str(spec_path), "--output", str(contract_path)]) == 2

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert template["schema_version"] == 4
    assert template["protocol"] == freezer.SPEC_PROTOCOL
    assert template["status"] == "HOLD_TEMPLATE_NOT_FROZEN"
    assert template["campaign_id"] == "tsp_fcdf_beta_20260815_v3"
    assert template["paths"]["frozen_contract_path"] == str(V3_CONTRACT)
    assert template["paths"]["output_root"] == str(V3_OUTPUT)
    assert template["paths"]["relocated_development_adapter"] == str(REAL_RELOCATION)
    with pytest.raises(freezer.ContractError, match="HOLD template"):
        freezer.build_contract(template)


def test_dry_run_has_only_memory_gate_supervision_and_no_general_execute_surface() -> None:
    source = DRY_RUN_PATH.read_text(encoding="utf-8")
    assert "os.system" not in source
    assert "--execute" not in source
    assert source.count("subprocess.Popen(") == 1
    assert "start_new_session=True" in source
    assert "_memory_gate_exec_wrapper_argv" in source
    assert "libc.prctl(1,signal.SIGTERM" in source
    assert "AUTHORIZED_ENGINEERING_PREREQUISITE_BY_FROZEN_CONTRACT" in source
    memory_guard = dry_run.build_attempt_guarded_argv(
        python_executable=Path(sys.executable),
        project_root=ROOT,
        contract_path=Path("/frozen/contract.json"),
        kind="memory_gate",
        trajectory_id="engineering-gate-D5-seed42",
        payload_argv=[str(Path(sys.executable)), "-m", "clasp.entrypoints.train_grpo"],
    )[2]
    matrix_guard = dry_run.build_attempt_guarded_argv(
        python_executable=Path(sys.executable),
        project_root=ROOT,
        contract_path=Path("/frozen/contract.json"),
        kind="training",
        trajectory_id="D0-seed42",
        payload_argv=[str(Path(sys.executable)), "-m", "clasp.entrypoints.train_grpo"],
    )[2]
    assert "supervise_memory_gate_attempt" in memory_guard
    assert "os.execv" in matrix_guard
    action_dests = {action.dest for action in dry_run.build_parser()._actions}
    assert "execute" not in action_dests


def _fake_supervised_child(
    payload_argv: list[str],
    *,
    transcript_descriptor: int,
    returncode: int,
) -> dict[str, object]:
    process_identity = _fixture_process_identity()
    dry_run._write_transcript_event(
        transcript_descriptor,
        {
            "schema_version": 1,
            "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
            "sequence": 2,
            "event": "child_spawn_pending",
            "supervisor_pid": os.getpid(),
            "payload_argv_sha256": freezer.sha256_json(payload_argv),
            "isolation": {
                "start_new_session": True,
                "linux_parent_death_signal": "SIGTERM",
                "parent_death_wrapper_kills_entire_process_group": True,
            },
        },
    )
    dry_run._write_transcript_event(
        transcript_descriptor,
        {
            "schema_version": 1,
            "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
            "sequence": 3,
            "event": "child_spawned",
            "payload_argv_sha256": freezer.sha256_json(payload_argv),
            "process_role": "parent_death_process_group_wrapper",
            "process_identity": process_identity,
        },
    )
    if returncode == 0:
        exit_code, signal_number, classification = 0, None, "exited_zero"
    elif returncode > 0:
        exit_code, signal_number, classification = returncode, None, "nonzero_exit"
    else:
        exit_code, signal_number, classification = None, -returncode, "signal"
    return {
        "child_started": True,
        "exit_known": True,
        "returncode": returncode,
        "exit_code": exit_code,
        "signal": signal_number,
        "classification": classification,
        "error_type": None,
        "error_message": None,
        "process_identity": process_identity,
    }


def test_memory_gate_supervisor_zero_exit_binds_exact_authority_and_is_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    observed: list[tuple[list[str], dict[str, str], Path]] = []

    def fake_child(payload, *, environment, cwd, transcript_descriptor):
        observed.append((list(payload), dict(environment), cwd))
        return _fake_supervised_child(
            list(payload),
            transcript_descriptor=transcript_descriptor,
            returncode=0,
        )

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(dry_run, "_run_memory_gate_child", fake_child)
    monkeypatch.setattr(
        dry_run,
        "_audit_memory_gate_processes_absent",
        lambda *, spawned_identity, **kwargs: _safe_process_absence(spawned_identity),
    )
    monkeypatch.setattr(dry_run, "_audit_memory_gate_gpu_processes_absent", _safe_gpu_absence)
    assert (
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=environment,
        )
        == 0
    )
    assert observed == [(payload_argv, environment, ROOT)]
    outcome_path, outcome = dry_run._inspect_memory_gate_zero_outcome(
        contract,
        contract_path=contract_path,
    )
    assert outcome["status"] == dry_run.MEMORY_GATE_SUCCESS_STATUS
    assert stat.S_IMODE(outcome_path.stat().st_mode) == 0o440
    assert outcome["execution_authority"]["payload_argv"] == payload_argv
    assert outcome["execution_authority"]["environment"] == environment
    claim_path = (
        Path(fixture["output_root"])
        / dry_run.MEMORY_GATE_DIRNAME
        / dry_run.CLAIM_ANCHOR_DIRNAME
        / dry_run.MEMORY_GATE_ATTEMPT_FILENAME
    )
    claim = freezer.load_json_object(claim_path, label="fixture attempt claim")
    for name in ("attempt", "transcript", "outcome"):
        record = claim["execution_anchors"][name]
        primary = Path(record["path"])
        guard = Path(record["generation_guard"]["path"])
        assert (primary.stat().st_dev, primary.stat().st_ino) == (
            guard.stat().st_dev,
            guard.stat().st_ino,
        )
        assert primary.stat().st_nlink == guard.stat().st_nlink == 2
    closure = outcome["execution_anchor_closure"]
    assert set(closure) == {
        "anchor_directory",
        "generation_guard_directory",
        "containing_gate_directory",
    }
    for identity in closure.values():
        assert set(identity) == {
            "path",
            "kind",
            "device",
            "inode",
            "mode",
            "mtime_ns",
            "ctime_ns",
        }
    with pytest.raises(freezer.ContractError, match="contaminated|already exists"):
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=environment,
        )
    assert len(observed) == 1


def test_zero_outcome_rejects_gate_rename_away_and_back_before_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _create_memory_gate_execution_outcome(fixture, contract)
    gate = Path(fixture["output_root"]) / dry_run.MEMORY_GATE_DIRNAME
    displaced = gate.with_name(f"{gate.name}.displaced")
    gate.rename(displaced)
    displaced.rename(gate)
    with pytest.raises(freezer.ContractError, match="execution-anchor closure"):
        dry_run._inspect_memory_gate_zero_outcome(
            contract,
            contract_path=contract_path,
        )


@pytest.mark.parametrize(
    ("returncode", "expected_shell_code", "classification"),
    [(7, 7, "nonzero_exit"), (-signal.SIGTERM, 128 + signal.SIGTERM, "signal")],
)
def test_memory_gate_supervisor_nonzero_and_signal_are_terminal_sealed_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    expected_shell_code: int,
    classification: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    calls = 0

    def fake_child(payload, *, environment, cwd, transcript_descriptor):
        nonlocal calls
        calls += 1
        return _fake_supervised_child(
            list(payload),
            transcript_descriptor=transcript_descriptor,
            returncode=returncode,
        )

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(dry_run, "_run_memory_gate_child", fake_child)
    monkeypatch.setattr(
        dry_run,
        "_audit_memory_gate_processes_absent",
        lambda *, spawned_identity, **kwargs: _safe_process_absence(spawned_identity),
    )
    monkeypatch.setattr(dry_run, "_audit_memory_gate_gpu_processes_absent", _safe_gpu_absence)
    assert (
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=environment,
        )
        == expected_shell_code
    )
    attempt_path = (
        Path(fixture["output_root"])
        / dry_run.MEMORY_GATE_DIRNAME
        / dry_run.CLAIM_ANCHOR_DIRNAME
        / dry_run.MEMORY_GATE_ATTEMPT_FILENAME
    )
    claim = freezer.load_json_object(attempt_path, label="fixture memory-gate claim")
    _, outcome_path, outcome = dry_run._inspect_memory_gate_execution_anchors(claim)
    assert outcome is not None
    outcome = dry_run.inspect_memory_gate_execution_outcome(
        outcome_path,
        contract,
        contract_path=contract_path,
        command=command,
        claim=claim,
    )
    assert outcome["status"] == dry_run.MEMORY_GATE_FAILURE_STATUS
    assert outcome["child"]["classification"] == classification
    output_root = Path(fixture["output_root"])
    assert all(not (path.stat().st_mode & 0o222) for path in [output_root, *output_root.rglob("*")])
    lifecycle = dry_run.inspect_lifecycle(contract, contract_path=contract_path)
    assert lifecycle["status"] == "MEMORY_GATE_FAILED_TERMINAL"
    assert lifecycle["retry_allowed"] is False
    assert lifecycle["matrix_authorized"] is False
    assert lifecycle["cell_commands_reemitted"] is False
    with pytest.raises(freezer.ContractError, match="terminal FAILED"):
        dry_run.record_memory_gate_pass_and_matrix(contract, contract_path=contract_path)
    with pytest.raises(freezer.ContractError):
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=environment,
        )
    assert calls == 1


def test_memory_gate_child_popen_uses_exact_wrapper_env_cwd_and_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [str(Path(sys.executable)), "-m", "clasp.entrypoints.train_grpo", "--x", "1"]
    environment = {"EXACT": "1"}
    process_identity = _fixture_process_identity(pid=45_678)
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 45_678

        def __init__(self):
            self.stdout = io.BytesIO(b"combined stdout/stderr\n")
            self.done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 0

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(dry_run.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        dry_run,
        "_read_linux_process_identity",
        lambda pid: process_identity if pid == 45_678 else None,
    )
    transcript = tmp_path / "transcript"
    descriptor = os.open(transcript, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        dry_run._write_transcript_event(
            descriptor,
            {
                "schema_version": 1,
                "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
                "sequence": 1,
                "event": "supervisor_claimed",
            },
        )
        child = dry_run._run_memory_gate_child(
            payload,
            environment=environment,
            cwd=ROOT,
            transcript_descriptor=descriptor,
        )
    finally:
        os.close(descriptor)
    wrapper = observed["argv"]
    kwargs = observed["kwargs"]
    compile(wrapper[2], "<memory-gate-parent-death-wrapper>", "exec")
    assert wrapper[-len(payload) :] == payload
    assert kwargs["env"] == environment
    assert kwargs["cwd"] == str(ROOT)
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert "preexec_fn" not in kwargs
    assert kwargs["stdout"] is dry_run.subprocess.PIPE
    assert kwargs["stderr"] is dry_run.subprocess.STDOUT
    assert child["classification"] == "exited_zero"
    assert child["process_identity"] == process_identity
    transcript_bytes = transcript.read_bytes()
    assert b'"event":"child_spawn_pending"' in transcript_bytes
    assert b'"event":"child_spawned"' in transcript_bytes
    assert transcript_bytes.endswith(b"combined stdout/stderr\n")


def test_memory_gate_never_spawns_after_preclaim_or_immediate_prespawn_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    calls = 0

    def impossible_child(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("child must not be invoked")

    monkeypatch.setattr(dry_run, "_run_memory_gate_child", impossible_child)
    monkeypatch.setattr(
        dry_run,
        "_audit_memory_gate_processes_absent",
        lambda *, spawned_identity, **kwargs: _safe_process_absence(spawned_identity),
    )
    monkeypatch.setattr(dry_run, "_audit_memory_gate_gpu_processes_absent", _safe_gpu_absence)
    altered = dict(environment)
    altered["UNAUTHORIZED"] = "1"
    with pytest.raises(freezer.ContractError, match="environment differs"):
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=altered,
        )
    monkeypatch.setattr(
        dry_run,
        "verify_attempt_execution_authority",
        lambda *args, **kwargs: freezer.fail("immediate pre-spawn authority drift"),
    )
    assert (
        dry_run.supervise_memory_gate_attempt(
            contract_path,
            trajectory_id="engineering-gate-D5-seed42",
            actual_payload_argv=payload_argv,
            actual_environment=environment,
        )
        == 2
    )
    assert calls == 0
    lifecycle = dry_run.inspect_lifecycle(contract, contract_path=contract_path)
    assert lifecycle["status"] == "MEMORY_GATE_FAILED_TERMINAL"


def test_operator_finalizer_refuses_live_orphan_then_binds_absent_spawn_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    claim, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    with pytest.raises(freezer.ContractError, match="still holds the transcript lock"):
        dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)
    process_identity = _fixture_process_identity(pid=54_321)
    for event in (
        {
            "schema_version": 1,
            "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
            "sequence": 1,
            "event": "supervisor_claimed",
        },
        {
            "schema_version": 1,
            "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
            "sequence": 2,
            "event": "child_spawn_pending",
            "supervisor_pid": 54_320,
        },
        {
            "schema_version": 1,
            "protocol": dry_run.MEMORY_GATE_TRANSCRIPT_PROTOCOL,
            "sequence": 3,
            "event": "child_spawned",
            "process_role": "parent_death_process_group_wrapper",
            "process_identity": process_identity,
        },
    ):
        dry_run._write_transcript_event(transcript_descriptor, event)
    os.close(transcript_descriptor)
    monkeypatch.setattr(
        dry_run,
        "_audit_memory_gate_processes_absent",
        lambda **kwargs: freezer.fail("claimed memory-gate child process is still running"),
    )
    monkeypatch.setattr(dry_run, "_audit_memory_gate_gpu_processes_absent", _safe_gpu_absence)
    with pytest.raises(freezer.ContractError, match="still running"):
        dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)
    outcome_path = Path(claim["execution_anchors"]["outcome"]["path"])
    assert outcome_path.stat().st_size == 0
    assert stat.S_IMODE(outcome_path.stat().st_mode) == 0o660
    observed_identity: list[object] = []

    def safe_processes(*, spawned_identity, payload_argv, wrapper_argv):
        observed_identity.append((spawned_identity, payload_argv, wrapper_argv))
        return _safe_process_absence(spawned_identity)

    monkeypatch.setattr(dry_run, "_audit_memory_gate_processes_absent", safe_processes)
    outcome = dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)
    assert observed_identity[0][0] == process_identity
    assert observed_identity[0][2][-len(payload_argv) :] == payload_argv
    assert outcome["status"] == dry_run.MEMORY_GATE_UNKNOWN_STATUS
    assert outcome["child"]["child_started"] is True
    assert outcome["child"]["process_identity"] == process_identity
    assert (
        outcome["operator_finalization_evidence"]["transcript_prefix"]["last_complete_event"][
            "event"
        ]
        == "child_spawned"
    )
    output_root = Path(fixture["output_root"])
    assert all(not (path.stat().st_mode & 0o222) for path in [output_root, *output_root.rglob("*")])


def test_operator_finalizer_without_spawn_marker_records_child_started_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    _, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    os.close(transcript_descriptor)
    monkeypatch.setattr(
        dry_run,
        "_audit_memory_gate_processes_absent",
        lambda **kwargs: _safe_process_absence(None),
    )
    monkeypatch.setattr(dry_run, "_audit_memory_gate_gpu_processes_absent", _safe_gpu_absence)
    outcome = dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)
    assert outcome["child"]["child_started"] is None
    assert outcome["child"]["process_identity"] is None
    assert outcome["child"]["exit_known"] is False


def test_operator_finalizer_rejects_deleted_and_recreated_transcript_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    claim, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    os.close(transcript_descriptor)
    transcript_path = Path(claim["execution_anchors"]["transcript"]["path"])
    anchor_root = transcript_path.parent
    anchor_root.chmod(0o750)
    transcript_path.unlink()
    transcript_path.write_bytes(b"")
    transcript_path.chmod(0o440)
    anchor_root.chmod(0o550)
    with pytest.raises(
        freezer.ContractError,
        match="deleted/recreated/altered|deleted, recreated, or multiply linked",
    ):
        dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)


def test_operator_finalizer_rejects_extra_execution_anchor_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    claim, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    os.close(transcript_descriptor)
    transcript_path = Path(claim["execution_anchors"]["transcript"]["path"])
    extra_link = (
        Path(fixture["output_root"])
        / dry_run.MEMORY_GATE_DIRNAME
        / "_runtime_tmp"
        / "forbidden_extra_transcript_link"
    )
    os.link(transcript_path, extra_link)
    with pytest.raises(freezer.ContractError, match="multiply linked"):
        dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)


def test_operator_finalizer_rejects_gate_directory_rename_away_and_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    _prepare_memory_gate_runtime(fixture)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    payload_argv, environment = _memory_gate_attempt_material(command)
    _, transcript_descriptor = dry_run.claim_memory_gate_attempt(
        contract_path,
        actual_payload_argv=payload_argv,
        actual_environment=environment,
    )
    os.close(transcript_descriptor)
    gate = Path(fixture["output_root"]) / dry_run.MEMORY_GATE_DIRNAME
    displaced = gate.with_name(f"{gate.name}.displaced")
    gate.rename(displaced)
    displaced.rename(gate)
    with pytest.raises(freezer.ContractError, match="output directory entry"):
        dry_run.finalize_interrupted_memory_gate(contract, contract_path=contract_path)


def test_failed_transcript_tamper_and_exact_manifest_exclusion_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = Path(fixture["frozen_contract_path"])
    freezer.write_contract_create_once(contract_path, contract)
    child = {
        "child_started": True,
        "exit_known": True,
        "returncode": 9,
        "exit_code": 9,
        "signal": None,
        "classification": "nonzero_exit",
        "error_type": None,
        "error_message": None,
        "process_identity": _fixture_process_identity(),
    }
    outcome_path, _, claim = _create_memory_gate_execution_outcome(
        fixture,
        contract,
        status=dry_run.MEMORY_GATE_FAILURE_STATUS,
        child=child,
    )
    output_root = Path(fixture["output_root"])
    nested = output_root / "nested"
    output_root.chmod(0o750)
    nested.mkdir()
    same_basename = nested / dry_run.MEMORY_GATE_OUTCOME_FILENAME
    same_basename.write_text("nested evidence\n", encoding="utf-8")
    manifest = dry_run._tree_manifest_excluding_exact_path(output_root, outcome_path)
    assert any(record["path"] == f"nested/{same_basename.name}" for record in manifest["files"])
    same_basename.unlink()
    nested.rmdir()
    transcript_path = Path(claim["execution_anchors"]["transcript"]["path"])
    anchor_root = transcript_path.parent
    gate_dir = anchor_root.parent
    anchor_root.chmod(0o750)
    transcript_path.chmod(0o640)
    with transcript_path.open("ab") as stream:
        stream.write(b"tamper\n")
    transcript_path.chmod(0o440)
    anchor_root.chmod(0o550)
    gate_dir.chmod(0o550)
    output_root.chmod(0o550)
    command = dry_run.build_plan(contract, contract_path=contract_path)["memory_gate_command"]
    with pytest.raises(
        freezer.ContractError,
        match="execution-anchor directory|transcript differs|tree differs",
    ):
        dry_run.inspect_memory_gate_execution_outcome(
            outcome_path,
            contract,
            contract_path=contract_path,
            command=command,
            claim=claim,
        )


def test_train_argv_resolves_for_continued_adapter_and_nonzero_lr_gate() -> None:
    common = {
        "python_executable": Path("/audit/python"),
        "initialization": Path("/immutable/relocated-adapter"),
        "tokenizer_path": Path("/immutable/frozen-base"),
        "train_data_root": Path("/frozen/train"),
        "output_root": Path("/new/output"),
        "run_dir": Path("/new/output/F5-seed44"),
        "beta": 0.05,
        "seed": 44,
    }
    arm_argv = dry_run.build_train_argv(**common)
    arm = finalize_grpo_args(build_grpo_parser().parse_args(arm_argv[3:]))
    assert arm.model_path == "/immutable/relocated-adapter"
    assert arm.tokenizer_path == "/immutable/frozen-base"
    assert arm.dtype == "bfloat16"
    assert arm.max_steps == 400
    assert arm.per_device_train_batch_size == 1
    assert arm.gradient_accumulation_steps == 32
    assert arm.effective_train_batch_size == 32
    assert arm.warmup_ratio == pytest.approx(0.03)
    assert arm.grpo_beta == pytest.approx(0.05)

    gate_argv = dry_run.build_train_argv(**common, max_steps=400, warmup_ratio=0.0)
    gate = finalize_grpo_args(build_grpo_parser().parse_args(gate_argv[3:]))
    assert gate.max_steps == 400
    assert gate.warmup_ratio == 0.0
    assert gate.effective_train_batch_size == 32


def test_obsolete_merged_builder_is_not_in_pilot_source_registry() -> None:
    assert "src/clasp/models/fixed_kl_reference.py" not in freezer.SOURCE_FILES
    assert "scripts/build_fixed_kl_reference.py" not in freezer.SOURCE_FILES
    assert "src/clasp/models/exact_peft_kl_reference.py" in freezer.SOURCE_FILES
    assert "src/clasp/models/exact_peft_kl_post_step.py" in freezer.SOURCE_FILES
    assert "scripts/relocate_development_adapter_base.py" in freezer.SOURCE_FILES


def test_positive_beta_output_contract_binds_frozen_post_step_proof() -> None:
    evidence = freezer.REQUIRED_OUTPUT_EVIDENCE
    assert evidence["exact_peft_post_step"] == {
        "schema_version": freezer.EXACT_PEFT_POST_SCHEMA_VERSION,
        "protocol": freezer.EXACT_PEFT_POST_DEVELOPMENT_PROTOCOL,
        "status": "development_exact_peft_reference_immutability_verified",
        "verifier_protocol": freezer.EXACT_PEFT_POST_VERIFIER_PROTOCOL,
        "verifier_implementation_sha256": freezer.FROZEN_EXACT_PEFT_POST_IMPLEMENTATION_SHA256,
        "policy_adapter_autocast_dtype": True,
        "reference_adapter_autocast_dtype": True,
        "step1_filename": "exact_peft_kl_reference_step1.json",
        "checkpoint_or_final_filename": "exact_peft_kl_reference_post_step.json",
        "reference_trainable_tensor_count": 0,
        "live_reference_bitwise_equal_to_pre_step": True,
        "policy_reference_storage_independent": True,
        "optimizer_excludes_reference_storage": True,
        "root_and_checkpoint_400_run_contract": "immutable_pre_outcome_matrix_ledger",
    }
    required = set(evidence["beta_positive_trajectory"])
    assert {
        "exact_peft_kl_reference.json",
        "exact_peft_kl_reference_step1.json",
        "exact_peft_kl_reference_post_step.json",
        "checkpoints/checkpoint-400/exact_peft_kl_reference.json",
        "checkpoints/checkpoint-400/exact_peft_kl_reference_step1.json",
        "checkpoints/checkpoint-400/exact_peft_kl_reference_post_step.json",
    } <= required
    assert evidence["beta_zero_trajectory_forbids_exact_peft_runtime_receipt"] is True
    assert (
        evidence["beta_zero_trajectory_forbids_all_pre_step_step1_and_post_step_sidecars"] is True
    )


@pytest.mark.parametrize("bad_value", [None, "nan", True, float("nan"), -1.0])
def test_gate_required_telemetry_rejects_non_numeric_nonfinite_or_negative(
    tmp_path: Path,
    bad_value: object,
) -> None:
    required = {
        "loss": 0.0,
        "learning_rate": 1e-6,
        "grad_norm": 1.0,
        "kl": 0.0,
        "constrained_grpo/candidate_feasibility_rate": 0.5,
        "constrained_grpo/clasp/advantage_l2": 1.0,
        "constrained_grpo/clasp/advantage_abs_mean": 1.0,
    }
    required["grad_norm"] = bad_value
    state_path = tmp_path / "trainer_state.json"
    _write_json(
        state_path,
        {
            "global_step": 1,
            "log_history": [{"step": 1, **required}],
        },
    )
    with pytest.raises(freezer.ContractError, match="required telemetry|nonnegative"):
        dry_run._audit_step1_trainer_state(state_path)


@pytest.mark.parametrize("bad_value", [None, "nan", True, float("inf"), -0.5])
def test_terminal_required_telemetry_rejects_non_numeric_nonfinite_or_negative(
    tmp_path: Path,
    bad_value: object,
) -> None:
    records = [
        {
            "step": step,
            "loss": -0.01,
            "learning_rate": 0.0,
            "grad_norm": 1.0,
        }
        for step in range(1, 401)
    ]
    records[72]["grad_norm"] = bad_value
    state_path = tmp_path / "trainer_state.json"
    _write_json(state_path, {"global_step": 400, "log_history": records})
    with pytest.raises(freezer.ContractError, match="required telemetry|nonnegative"):
        dry_run._audit_terminal_telemetry(
            state_path,
            required=["loss", "learning_rate", "grad_norm"],
        )


def test_gate_root_requires_exact_claim_and_rejects_finalization_contamination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    _install_relocation_stub(monkeypatch, fixture)
    contract = freezer.build_contract(fixture["spec"])
    contract_path = fixture["frozen_contract_path"]
    freezer.write_contract_create_once(contract_path, contract)
    gate_dir = fixture["output_root"] / dry_run.MEMORY_GATE_DIRNAME

    def create_gate_artifacts(gate: Path) -> None:
        (gate / "checkpoints").mkdir()
        (gate / "hyperparams.json").write_text("{}\n", encoding="utf-8")
        (gate / freezer.EXACT_PEFT_RUNTIME_RECEIPT).write_text("{}\n", encoding="utf-8")
        (gate / freezer.EXACT_PEFT_STEP1_RECEIPT).write_text("{}\n", encoding="utf-8")

    outcome_path, outcome, payload = _create_memory_gate_execution_outcome(
        fixture,
        contract,
        before_outcome=create_gate_artifacts,
    )
    anchor_root = gate_dir / dry_run.CLAIM_ANCHOR_DIRNAME
    attempt_path = anchor_root / dry_run.MEMORY_GATE_ATTEMPT_FILENAME
    observed_path, observed, observed_outcome_path, observed_outcome = (
        dry_run._audit_memory_gate_root_claim(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )
    )
    assert observed_path == attempt_path
    assert observed == payload
    assert observed_outcome_path == outcome_path
    assert observed_outcome == outcome

    contamination = gate_dir / "adapter_model.safetensors"
    gate_dir.chmod(0o750)
    contamination.write_bytes(b"forbidden-root-finalization")
    gate_dir.chmod(0o550)
    with pytest.raises(freezer.ContractError, match="execution-anchor closure|exact allowlist"):
        dry_run._audit_memory_gate_root_claim(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )
    gate_dir.chmod(0o750)
    contamination.unlink()
    gate_dir.chmod(0o550)
    anchor_root.chmod(0o750)
    attempt_path.unlink()
    anchor_root.chmod(0o550)
    with pytest.raises(
        freezer.ContractError,
        match="execution-anchor directory|execution-anchor closure|attempt anchor",
    ):
        dry_run._audit_memory_gate_root_claim(
            contract,
            contract_path=contract_path,
            gate_dir=gate_dir,
        )


def test_claimed_strict_resume_sets_only_exact_checkpoint_environment(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint-17"
    checkpoint.mkdir(parents=True)
    environment = {"FROZEN": "1", "RESUME_FROM_CHECKPOINT": "/stale"}
    claimed = {
        "resume_checkpoint": {
            "step": 17,
            "path": str(checkpoint),
        }
    }
    assert dry_run.apply_claimed_resume_environment(claimed, environment) == str(checkpoint)
    assert environment["RESUME_FROM_CHECKPOINT"] == str(checkpoint)
    assert (
        dry_run.apply_claimed_resume_environment({"resume_checkpoint": None}, environment) is None
    )
    assert "RESUME_FROM_CHECKPOINT" not in environment
    with pytest.raises(freezer.ContractError, match="exact live checkpoint"):
        dry_run.apply_claimed_resume_environment(
            {
                "resume_checkpoint": {
                    "step": 18,
                    "path": str(checkpoint),
                }
            },
            environment,
        )
