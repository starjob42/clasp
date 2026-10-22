from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_checkpoint_evidence", ROOT / "scripts" / "build_checkpoint_evidence.py"
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = builder
BUILDER_SPEC.loader.exec_module(builder)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "aggregate_publication_fixture",
    ROOT / "tests" / "test_aggregate_publication_contract.py",
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = fixture_module
FIXTURE_SPEC.loader.exec_module(fixture_module)


@pytest.fixture(autouse=True)
def _stub_live_checkpoint_fidelity(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity(checkpoint: Path) -> dict:
        root = Path(checkpoint)
        return fixture_module._fidelity_identity(
            root.name,
            receipt_bytes=(root / "checkpoint_fidelity.json").read_bytes(),
            adapter_config_bytes=(root / "adapter_config.json").read_bytes(),
            adapter_weights_bytes=(root / "adapter_model.safetensors").read_bytes(),
        )

    monkeypatch.setattr(builder, "publication_fidelity_identity", identity)


def _policy(tmp_path: Path, *, steps: list[int] | None = None) -> Path:
    config_path = next(tmp_path.rglob("run_config__paper.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_files = [
        {
            "task": "tsp",
            "split": "eval",
            "path": "tsp/eval/test.json",
            "size": 3,
            "sha256": "3" * 64,
        }
    ]
    sources = []
    for label in ("historical-eval", "rl-train", "sft-train"):
        if label == "historical-eval":
            files = [
                {
                    "relative_path": data_files[0]["path"],
                    "sha256": data_files[0]["sha256"],
                    "size_bytes": data_files[0]["size"],
                    "rows": 1,
                }
            ]
        else:
            files = [
                {
                    "relative_path": f"tsp/{label}/data.json",
                    "sha256": hashlib.sha256(label.encode()).hexdigest(),
                    "size_bytes": 3,
                    "rows": 1,
                }
            ]
        source = {
            "task": "tsp",
            "label": label,
            "files": files,
            "file_manifest_sha256": hashlib.sha256(
                builder.selector._canonical_json_bytes(files)
            ).hexdigest(),
            "rows": 1,
            "unique_instance_hashes": 1,
        }
        source["source_identity_sha256"] = hashlib.sha256(
            builder.selector._canonical_json_bytes(source)
        ).hexdigest()
        sources.append(source)
    registry = {
        "schema_version": 1,
        "protocol": "clasp-forbidden-source-registry-v1",
        "status": "frozen_before_confirmation_generation",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "tasks": ["tsp"],
        "required_labels": ["historical-eval", "rl-train", "sft-train"],
        "sources": sources,
        "source_tree_manifest_sha256": hashlib.sha256(
            builder.selector._canonical_json_bytes(sources)
        ).hexdigest(),
    }
    registry["registry_sha256"] = hashlib.sha256(
        builder.selector._canonical_json_bytes(registry)
    ).hexdigest()
    historical = next(source for source in sources if source["label"] == "historical-eval")
    development = {
        "data_manifest_sha256": hashlib.sha256(
            builder.selector._canonical_json_bytes(data_files)
        ).hexdigest(),
        "evaluation_data_files": data_files,
        "forbidden_source_label": "historical-eval",
        "forbidden_source_identity_sha256": historical["source_identity_sha256"],
        "protocol_fingerprint": config["protocol_fingerprint"],
        "strict_verifier_manifest_sha256": config["verifier_identity"][
            "implementation_manifest_sha256"
        ],
        "split": "eval",
        "eval_start_index": 0,
        "max_eval_samples_per_task": 1,
        "eval_samples_per_task": {},
        "size_bucket": "all",
        "min_instance_size": 0,
        "max_instance_size": 0,
        "prefer_subset_files": False,
        "expected_instances": 1,
    }
    development["development_evaluation_sha256"] = hashlib.sha256(
        builder.selector._canonical_json_bytes(development)
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "campaign_id": "campaign",
        "stage_id": "selection",
        "status": "frozen_before_development_results",
        "frozen_at_utc": "2026-08-14T00:00:00Z",
        "evidence_scope": "development_only",
        "expected_tasks": ["tsp"],
        "expected_methods": ["clasp_dense"],
        "train_seeds": [42],
        "checkpoint_steps": steps or [3200],
        "candidate_budget": 8,
        "forbidden_source_registry": registry,
        "development_evaluation_by_task": {"tsp": development},
        "require_complete_checkpoint_grid": True,
        "pass_at_k_absolute_band": 0.01,
        "selection_rule": builder.selector.SELECTION_RULE,
    }
    payload["policy_sha256"] = hashlib.sha256(
        builder.selector._canonical_json_bytes(payload)
    ).hexdigest()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builder_derives_complete_selector_evidence_from_artifacts(tmp_path: Path) -> None:
    spec, details, _config, _summary = fixture_module._native_fixture(tmp_path / "run")
    policy = _policy(tmp_path)
    evidence = builder.build_evidence(policy_path=policy, detail_paths=[details])
    assert evidence["status"] == "complete"
    row = evidence["records"][0]
    assert (row["method"], row["train_seed"], row["task"], row["checkpoint_step"]) == (
        "clasp_dense",
        42,
        "tsp",
        3200,
    )
    assert row["run_contract_sha256"] == fixture_module.aggregate._training_contract_file_sha256(
        fixture_module.aggregate._load_json_object(
            str(Path(spec.path).parent / "run_config__paper.json")
        )["training_run_contract"]
    )
    output = tmp_path / "evidence.json"
    builder._atomic_json(output, evidence)
    validated, receipt = builder.selector._validate_evidence(
        output,
        policy=builder.selector._validate_policy(policy),
    )
    assert len(validated) == 1
    assert receipt["record_count"] == 1
    assert output.stat().st_mode & 0o222 == 0


def test_builder_rejects_incomplete_frozen_grid(tmp_path: Path) -> None:
    _spec, details, _config, _summary = fixture_module._native_fixture(tmp_path / "run")
    policy = _policy(tmp_path, steps=[3100, 3200])
    with pytest.raises(ValueError, match="do not equal the frozen policy grid"):
        builder.build_evidence(policy_path=policy, detail_paths=[details])


@pytest.mark.parametrize("artifact", ["run_contract", "checkpoint"])
def test_builder_rehashes_external_training_artifacts(
    tmp_path: Path,
    artifact: str,
) -> None:
    _spec, details, _config, _summary = fixture_module._native_fixture(tmp_path / "run")
    policy = _policy(tmp_path)
    if artifact == "run_contract":
        contract_path = tmp_path / "run" / "training_run" / "run_contract.json"
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        payload["method"] = "dual_grpo"
        contract_path.chmod(0o600)
        contract_path.write_text(json.dumps(payload), encoding="utf-8")
        expected_error = "exactly one on-disk run_contract"
    else:
        checkpoint_path = (
            tmp_path / "run" / "training_run" / "checkpoint-3200" / "adapter_model.safetensors"
        )
        checkpoint_path.write_bytes(b"tampered")
        expected_error = "checkpoint bytes disagree"
    with pytest.raises(ValueError, match=expected_error):
        builder.build_evidence(policy_path=policy, detail_paths=[details])


def test_builder_refuses_to_replace_evidence(tmp_path: Path) -> None:
    _spec, details, _config, _summary = fixture_module._native_fixture(tmp_path / "run")
    policy = _policy(tmp_path)
    output = tmp_path / "evidence.json"
    evidence = builder.build_evidence(policy_path=policy, detail_paths=[details])
    builder._atomic_json(output, evidence)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        builder._atomic_json(output, evidence)
    assert output.read_bytes() == original
