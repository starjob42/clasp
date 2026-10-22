from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SWEEP_SPEC = importlib.util.spec_from_file_location(
    "remote_launch_sweep", SCRIPTS / "run_op_scalability_sweep.py"
)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
sweep = importlib.util.module_from_spec(SWEEP_SPEC)
sys.modules[SWEEP_SPEC.name] = sweep
SWEEP_SPEC.loader.exec_module(sweep)

PUBLISHER_SPEC = importlib.util.spec_from_file_location(
    "remote_launch_publisher", SCRIPTS / "publish_op_scalability_gurobi_artifact.py"
)
assert PUBLISHER_SPEC is not None and PUBLISHER_SPEC.loader is not None
publisher = importlib.util.module_from_spec(PUBLISHER_SPEC)
sys.modules[PUBLISHER_SPEC.name] = publisher
PUBLISHER_SPEC.loader.exec_module(publisher)

EXECUTION_ROOT = (
    ROOT / "experiments" / "sota_20260814" / "op_scalability_remote_execution_v1"
)
REMOTE_CONTRACT = EXECUTION_ROOT / "sweep_contract.remote_v1.json"
CAMPAIGN_CONTRACT = EXECUTION_ROOT / "remote_execution_contract.json"
RUNTIME_MANIFEST = EXECUTION_ROOT / "runtime_python_bundle_manifest_v1.json"
FAILURE_REPAIR = EXECUTION_ROOT / "remote_preflight_failure_repair_v1.json"
LAUNCHER = SCRIPTS / "launch_op_scalability_cpu_sweep.slurm"
RUNTIME_VERIFIER = SCRIPTS / "verify_op_scalability_runtime_bundle.py"
PENDING_CONTRACT = (
    ROOT
    / "experiments"
    / "sota_20260814"
    / "op_scalability_development_v1"
    / "sweep_contract.pending_remote_artifact.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shell_constant(text: str, name: str) -> str:
    match = re.search(rf"^readonly {re.escape(name)}=([^\n]+)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1).strip("'\"")


def _runtime_manifest() -> dict[str, object]:
    value = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _local_bundle_source(relative_path: str) -> Path:
    path = ROOT / relative_path
    assert path.is_file(), relative_path
    return path


def _stage_runtime_bundle(tmp_path: Path) -> tuple[Path, Path]:
    manifest = _runtime_manifest()
    bundle_root = tmp_path / "runtime_python_bundle_v1"
    entries = manifest["entries"]
    assert isinstance(entries, list)
    for raw_entry in entries:
        assert isinstance(raw_entry, dict)
        relative = str(raw_entry["relative_path"])
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_local_bundle_source(relative), destination)
        destination.chmod(0o440)
    staged_manifest = tmp_path / RUNTIME_MANIFEST.name
    shutil.copyfile(RUNTIME_MANIFEST, staged_manifest)
    staged_manifest.chmod(0o440)
    return bundle_root, staged_manifest


def _dependency_complete_python() -> Path | None:
    candidates = [
        Path(sys.executable),
        ROOT.parent / "my_idea_grpo" / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [
                str(candidate),
                "-c",
                "import datasets, peft, torch, transformers, wandb, tqdm",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    return None


def test_remote_runnable_contract_is_self_hashed_and_preserves_scientific_plan() -> None:
    contract = json.loads(REMOTE_CONTRACT.read_text(encoding="utf-8"))
    sweep._validate_contract_shape(contract)
    sweep._verify_self_hash(contract, "contract_sha256", label="remote contract")
    assert contract["source"]["confirmation_data_used"] is False
    assert contract["source"]["op_dataset_sha256"] == (
        "8459e3fc1fc5f364d4ff19d47023132196316c096753711c8b90337a772603d1"
    )
    assert contract["solver"]["artifact"] == {
        "path": str(publisher.DESTINATION_DIR / publisher.ARTIFACT_NAME),
        "sha256": publisher.EXPECTED_SHA256,
    }
    assert contract["solver"]["base_parameters"]["threads"] == 1
    assert all(row["work_limit"] is None for row in contract["selection"]["strata"])
    assert all(row["node_limit"] is None for row in contract["selection"]["strata"])
    assert contract["production_screen"] == {
        "target_rows": 1000,
        "require_all_sampled_optimal": True,
        "max_projected_native_runtime_seconds": 172800.0,
    }


def test_remote_execution_contract_binds_every_launcher_input_and_forbids_confirmation() -> None:
    campaign = json.loads(CAMPAIGN_CONTRACT.read_text(encoding="utf-8"))
    unsigned = dict(campaign)
    digest = unsigned.pop("contract_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned)).hexdigest() == digest
    assert campaign["status"] == "HOLD_REMOTE_STAGING_AND_ARTIFACT_PUBLICATION_REQUIRED"
    assert campaign["confirmation_policy"] == {
        "access_confirmation_data": False,
        "solve_confirmation_data": False,
        "authorize_confirmation_reference_generation": False,
    }
    assert campaign["operational_packaging_repair"] == {
        "caught_failure": "ModuleNotFoundError: No module named 'clasp.data'",
        "confirmation_data_used": False,
        "diagnosis_record": {
            "diagnosis_sha256": (
                "9ae6bb210edb5541a42fe7c16e5a9fd1155e5ab8945b14a18ccb429ca074250c"
            ),
            "file_sha256": _sha256(FAILURE_REPAIR),
            "remote_path": (
                "/lustre/fsn1/projects/rech/vas/uag72kz/"
                "clasp_op_scalability_development_iclr_20260826_v1/execution/"
                "op_scalability_remote_execution_v1/remote_preflight_failure_repair_v1.json"
            ),
        },
        "remote_mutation_authorized_by_this_contract": False,
        "repair": "sealed_exact_local_python_import_closure_bundle",
        "revision": 2,
        "scientific_sweep_contract_changed": False,
        "strict_evaluator_sha256_preserved": (
            "e9ab65f8486ceacf7f8e020e48efc571a3f2915ce4d3a7db05813aa424198908"
        ),
    }
    required = campaign["required_staged_files"]
    local_map = {
        "code/correct_clasp_op_scalability_v1/scripts/launch_op_scalability_cpu_sweep.slurm": LAUNCHER,
        "code/correct_clasp_op_scalability_v1/scripts/publish_op_scalability_gurobi_artifact.py": SCRIPTS
        / "publish_op_scalability_gurobi_artifact.py",
        "code/correct_clasp_op_scalability_v1/scripts/verify_op_scalability_runtime_bundle.py": RUNTIME_VERIFIER,
        "execution/op_scalability_remote_execution_v1/runtime_python_bundle_manifest_v1.json": RUNTIME_MANIFEST,
        "execution/op_scalability_remote_execution_v1/remote_preflight_failure_repair_v1.json": FAILURE_REPAIR,
        "execution/op_scalability_remote_execution_v1/sweep_contract.remote_v1.json": REMOTE_CONTRACT,
        "plan/op_scalability_development_v1/generation_receipt.json": ROOT
        / "experiments/sota_20260814/op_scalability_development_v1/generation_receipt.json",
        "plan/op_scalability_development_v1/op/eval/test.json": ROOT
        / "experiments/sota_20260814/op_scalability_development_v1/op/eval/test.json",
    }
    manifest = _runtime_manifest()
    for entry in manifest["entries"]:
        relative = str(entry["relative_path"])
        local_map[
            "code/correct_clasp_op_scalability_v1/runtime_python_bundle_v1/"
            + relative
        ] = _local_bundle_source(relative)
    assert set(required) == set(local_map)
    for relative, path in local_map.items():
        assert required[relative] == _sha256(path), relative

    runtime = campaign["runtime_python_bundle"]
    assert runtime["exact_inventory_required"] is True
    assert runtime["file_count"] == 25
    assert runtime["manifest"] == {
        "file_sha256": _sha256(RUNTIME_MANIFEST),
        "manifest_sha256": manifest["manifest_sha256"],
        "remote_path": (
            "/lustre/fsn1/projects/rech/vas/uag72kz/"
            "clasp_op_scalability_development_iclr_20260826_v1/execution/"
            "op_scalability_remote_execution_v1/runtime_python_bundle_manifest_v1.json"
        ),
    }


def test_remote_preflight_failure_repair_record_is_self_hashed_and_non_authorizing() -> None:
    record = json.loads(FAILURE_REPAIR.read_text(encoding="utf-8"))
    unsigned = dict(record)
    digest = unsigned.pop("diagnosis_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned)).hexdigest() == digest
    assert record["failure"] == {
        "cause": "incomplete_staged_local_python_import_closure",
        "exception_message": "No module named 'clasp.data'",
        "exception_type": "ModuleNotFoundError",
        "stage": "exact_remote_launcher_environment_preflight",
    }
    assert record["observed_side_effect_state"]["slurm_job_submitted"] is False
    assert record["observed_side_effect_state"]["scientific_sweep_started"] is False
    assert record["observed_side_effect_state"]["result_output_exists"] is False
    assert record["authorization"] == {
        "confirmation_reference_generation_authorized": False,
        "job_submission_authorized_by_record": False,
        "remote_mutation_authorized_by_record": False,
    }


def test_cpu_launcher_has_valid_shell_and_exact_fail_closed_bindings() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH --no-requeue" in text
    assert "export CUDA_VISIBLE_DEVICES=''" in text
    assert "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1" in text
    assert "--verify-publication" in text
    assert "--expected-manifest-file-sha256" in text
    assert "-B -I \"${BUNDLE_VERIFIER}\"" in text
    assert "--preflight-only" in text
    assert "confirmation" not in text.lower()
    assert _shell_constant(text, "PUBLISHER_SHA256") == _sha256(
        SCRIPTS / "publish_op_scalability_gurobi_artifact.py"
    )
    assert _shell_constant(text, "BUNDLE_VERIFIER_SHA256") == _sha256(
        RUNTIME_VERIFIER
    )
    assert _shell_constant(text, "BUNDLE_MANIFEST_FILE_SHA256") == _sha256(
        RUNTIME_MANIFEST
    )
    assert _shell_constant(text, "SWEEP_RUNNER_SHA256") == _sha256(
        SCRIPTS / "run_op_scalability_sweep.py"
    )
    assert _shell_constant(text, "EXACT_RUNNER_SHA256") == _sha256(
        SCRIPTS / "run_reference_evidence.py"
    )
    assert _shell_constant(text, "CONTRACT_FILE_SHA256") == _sha256(REMOTE_CONTRACT)
    assert _shell_constant(text, "SEALED_WHEEL_SHA256") == publisher.EXPECTED_SHA256


def test_runtime_bundle_manifest_is_self_hashed_complete_and_preserves_evaluator() -> None:
    manifest = _runtime_manifest()
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256")
    assert hashlib.sha256(sweep._canonical_json_bytes(unsigned)).hexdigest() == digest
    assert manifest["confirmation_data_used"] is False
    assert manifest["pythonpath_order"] == ["src", "scripts"]
    entries = manifest["entries"]
    assert len(entries) == 25
    assert [entry["relative_path"] for entry in entries] == sorted(
        entry["relative_path"] for entry in entries
    )
    for entry in entries:
        source = _local_bundle_source(entry["relative_path"])
        assert entry["sha256"] == _sha256(source)
        assert entry["size_bytes"] == source.stat().st_size
        assert entry["mode"] == "0440"
        assert entry["nlink"] == 1
    evaluator = next(
        entry
        for entry in entries
        if entry["relative_path"] == "src/clasp/entrypoints/evaluate.py"
    )
    assert evaluator["sha256"] == (
        "e9ab65f8486ceacf7f8e020e48efc571a3f2915ce4d3a7db05813aa424198908"
    )


def test_runtime_bundle_verifier_rejects_extra_file_mode_and_hardlink(tmp_path: Path) -> None:
    bundle_root, staged_manifest = _stage_runtime_bundle(tmp_path)
    command = [
        sys.executable,
        str(RUNTIME_VERIFIER),
        "--manifest",
        str(staged_manifest),
        "--bundle-root",
        str(bundle_root),
        "--expected-manifest-file-sha256",
        _sha256(RUNTIME_MANIFEST),
    ]
    clean = subprocess.run(command, capture_output=True, text=True, check=False)
    assert clean.returncode == 0, clean.stderr
    assert json.loads(clean.stdout)["status"] == "PASS_EXACT_RUNTIME_BUNDLE"

    extra = bundle_root / "src/clasp/unbound.py"
    extra.write_text("# unbound\n", encoding="utf-8")
    extra.chmod(0o440)
    rejected_extra = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected_extra.returncode == 2
    assert "exact inventory mismatch" in rejected_extra.stderr
    extra.unlink()

    runner = bundle_root / "scripts/run_op_scalability_sweep.py"
    runner.chmod(0o640)
    rejected_mode = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected_mode.returncode == 2
    assert "mode/link mismatch" in rejected_mode.stderr
    runner.chmod(0o440)

    hardlink = tmp_path / "runner.hardlink"
    os.link(runner, hardlink)
    rejected_hardlink = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected_hardlink.returncode == 2
    assert "mode/link mismatch" in rejected_hardlink.stderr


def test_isolated_pythonpath_bundle_runs_full_sweep_preflight(tmp_path: Path) -> None:
    python = _dependency_complete_python()
    if python is None:
        pytest.skip("no local interpreter has the remote preflight dependency surface")
    bundle_root, staged_manifest = _stage_runtime_bundle(tmp_path)
    verified = subprocess.run(
        [
            sys.executable,
            str(RUNTIME_VERIFIER),
            "--manifest",
            str(staged_manifest),
            "--bundle-root",
            str(bundle_root),
            "--expected-manifest-file-sha256",
            _sha256(RUNTIME_MANIFEST),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    environment = {
        "HOME": str(tmp_path),
        "LOGNAME": "clasp-op-bundle-test",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{bundle_root / 'src'}:{bundle_root / 'scripts'}",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "USER": "clasp-op-bundle-test",
    }
    preflight = subprocess.run(
        [
            str(python),
            str(bundle_root / "scripts/run_op_scalability_sweep.py"),
            "--contract",
            str(PENDING_CONTRACT),
            "--preflight-only",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    receipt = json.loads(preflight.stdout)
    assert receipt["status"] == "HOLD_REMOTE_READ_ONLY_ARTIFACT_FINALIZATION_REQUIRED"
    assert receipt["sample_rows"] == 24
    assert receipt["confirmation_data_access_authorized"] is False
    assert receipt["confirmation_solving_authorized"] is False
