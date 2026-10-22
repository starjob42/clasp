"""Re-score raw candidate JSONL with CLASP's shared strict CO verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

from clasp.entrypoints.evaluate import (
    TASK_EVALUATORS,
    _aggregate_records,
    _build_instance_record,
    _canonical_json_bytes,
    _constraint_diagnostics,
    _details_artifact_identity,
    _experiment_fingerprint,
    _parse_instance_value,
    _protocol_fingerprint_material,
    _self_hash_payload,
    _sha256_file,
    _strict_verifier_identity,
    _validate_self_hash,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from clasp.models.publication_fidelity import build_derived_source_config_snapshot


def build_rescore_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute feasibility and objective metrics for raw model candidates with "
            "the exact strict verifier used by clasp-eval."
        )
    )
    parser.add_argument(
        "--details",
        nargs="+",
        required=True,
        help="One or more run-details JSONL files following the clasp raw-candidate schema.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSONL for strictly re-scored per-instance records.",
    )
    parser.add_argument(
        "--system_name",
        default="external",
        help="Stable system label recorded in the rescore summary.",
    )
    parser.add_argument(
        "--source_config",
        default=None,
        help=(
            "Optional completed clasp-eval run_config. Publication aggregation requires "
            "this to bind native model, decoder, prompt, and training-contract provenance."
        ),
    )
    return parser


def _load_raw_records(paths: list[str]) -> tuple[list[dict], list[dict[str, Any]]]:
    records: list[dict] = []
    sources: list[dict[str, Any]] = []
    for raw_path in paths:
        path = os.path.abspath(raw_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Raw details file does not exist: {path}")
        path_before = os.stat(path)
        digest = hashlib.sha256()
        observed_size = 0
        with open(path, "rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                observed_size += len(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Invalid UTF-8 at {path}:{line_number}") from exc
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                if "record_sha256" in payload:
                    _validate_self_hash(
                        payload,
                        "record_sha256",
                        label=f"source detail record {path}:{line_number}",
                    )
                records.append(payload)
            descriptor_after = os.fstat(handle.fileno())
        path_after = os.stat(path)
        descriptor_signature = (
            descriptor_before.st_dev,
            descriptor_before.st_ino,
            descriptor_before.st_size,
            descriptor_before.st_mtime_ns,
        )
        if (
            descriptor_signature
            != (
                descriptor_after.st_dev,
                descriptor_after.st_ino,
                descriptor_after.st_size,
                descriptor_after.st_mtime_ns,
            )
            or descriptor_signature
            != (
                path_before.st_dev,
                path_before.st_ino,
                path_before.st_size,
                path_before.st_mtime_ns,
            )
            or descriptor_signature
            != (
                path_after.st_dev,
                path_after.st_ino,
                path_after.st_size,
                path_after.st_mtime_ns,
            )
        ):
            raise RuntimeError(f"Raw details file changed while being read: {path}")
        if observed_size != descriptor_after.st_size:
            raise RuntimeError(f"Raw details file size changed while being read: {path}")
        sources.append(
            {
                "basename": os.path.basename(path),
                "size": observed_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records, sources


def _candidate_completion(candidate: Any, *, record_index: int, candidate_index: int) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict) and isinstance(candidate.get("completion"), str):
        return str(candidate["completion"])
    raise ValueError(
        f"Record {record_index}, candidate {candidate_index} must be a completion string "
        "or an object containing a string 'completion'."
    )


def _rescore_record(
    record: dict,
    *,
    record_index: int,
    rescore_fingerprint: str,
    verifier: dict,
) -> dict:
    task = str(record.get("task", "")).lower()
    if task not in TASK_EVALUATORS:
        raise ValueError(f"Record {record_index} has unsupported task {task!r}")
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input_problem", record.get("input", "")))
    answer = str(record.get("ground_truth_output", record.get("output", "")))
    if not instruction or not input_text or not answer:
        raise ValueError(
            f"Record {record_index} must contain non-empty instruction, input_problem, "
            "and ground_truth_output fields."
        )
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"Record {record_index} must contain a non-empty candidates list")

    raw_instance = record.get("instance_json")
    stored_instance_sha256 = str(record.get("instance_sha256", "") or "")
    direct_instance_sha256 = hashlib.sha256(_canonical_json_bytes(raw_instance)).hexdigest()
    if stored_instance_sha256 and stored_instance_sha256 != direct_instance_sha256:
        raise ValueError(
            f"Record {record_index} instance_json does not match its recorded instance_sha256"
        )
    instance = _parse_instance_value({"instance_json": raw_instance})
    evaluator = TASK_EVALUATORS[task]
    candidates: list[dict] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        completion = _candidate_completion(
            raw_candidate,
            record_index=record_index,
            candidate_index=candidate_index,
        )
        candidate_instruction = instruction
        candidate_input_text = input_text
        candidate_answer = answer
        candidate_instance = instance
        if isinstance(raw_candidate, dict) and isinstance(
            raw_candidate.get("verification_context"), dict
        ):
            verification_context = raw_candidate["verification_context"]
            context_sha256 = hashlib.sha256(_canonical_json_bytes(verification_context)).hexdigest()
            expected_context_sha256 = str(
                raw_candidate.get("verification_context_sha256", "") or ""
            )
            if expected_context_sha256 and expected_context_sha256 != context_sha256:
                raise ValueError(
                    f"Record {record_index}, candidate {candidate_index} has a corrupted "
                    "verification_context"
                )
            candidate_instruction = str(
                verification_context.get("instruction", candidate_instruction)
            )
            candidate_input_text = str(
                verification_context.get("input_problem", candidate_input_text)
            )
            candidate_answer = str(
                verification_context.get("ground_truth_output", candidate_answer)
            )
            candidate_instance = _parse_instance_value(
                {"instance_json": verification_context.get("instance_json")}
            )
        metrics = evaluator(
            candidate_instruction,
            candidate_input_text,
            candidate_answer,
            candidate_instance,
            completion,
        )
        metrics.update(
            _constraint_diagnostics(
                task,
                prompt=candidate_instruction,
                input_text=candidate_input_text,
                answer=candidate_answer,
                instance=candidate_instance,
                completion=completion,
            )
        )
        metrics["candidate_index"] = candidate_index
        metrics["completion"] = completion
        if isinstance(raw_candidate, dict):
            for key in (
                "completion_tokens",
                "decoding_temperature",
                "decoding_top_p",
                "op_size_band_top_p_applied",
                "tsp_label_tta_variant",
            ):
                if key in raw_candidate:
                    metrics[key] = raw_candidate[key]
            if "verification_context" in raw_candidate:
                metrics["verification_context"] = raw_candidate["verification_context"]
                metrics["verification_context_sha256"] = raw_candidate.get(
                    "verification_context_sha256"
                )
        candidates.append(metrics)

    rescored = _build_instance_record(
        task=task,
        split=str(record.get("split", "eval")),
        example_index=int(record.get("example_index", record_index)),
        source_json=record.get("source_json"),
        source_json_files=list(record.get("source_json_files", []) or []),
        instruction=instruction,
        input_text=input_text,
        answer=answer,
        instance_size=record.get("instance_size"),
        prompt_tokens=int(record.get("prompt_tokens", 0) or 0),
        prompt_was_truncated=bool(record.get("prompt_was_truncated", False)),
        generation_time_sec=float(record.get("generation_time_sec", 0.0) or 0.0),
        num_return_sequences=len(candidates),
        candidates=candidates,
    )
    rescored["instance_json"] = record.get("instance_json")
    rescored["instance_sha256"] = direct_instance_sha256
    for identity_field in (
        "semantic_instance_sha256",
        "benchmark_row_sha256",
        "reference_output_sha256",
    ):
        if identity_field in record:
            rescored[identity_field] = record[identity_field]
    rescored["source_row_locator"] = record.get("source_row_locator")
    rescored["sampling_seed"] = record.get("sampling_seed")
    rescored["transformed_example_sha256"] = record.get("transformed_example_sha256")
    rescored["source_experiment_fingerprint"] = record.get("experiment_fingerprint")
    rescored["source_experiment_fingerprint_schema_version"] = record.get(
        "experiment_fingerprint_schema_version"
    )
    rescored["source_protocol_fingerprint"] = record.get("protocol_fingerprint")
    rescored["training_run_contract_sha256"] = record.get("training_run_contract_sha256")
    for field in ("campaign_manifest_sha256", "confirmation_attempt_id"):
        if record.get(field) is not None:
            rescored[field] = record[field]
    rescored["rescore_fingerprint"] = rescore_fingerprint
    rescored["strict_verifier"] = verifier
    return _self_hash_payload(rescored, "rescore_record_sha256")


def _load_source_config(
    path: str | None, sources: list[dict[str, Any]]
) -> tuple[dict | None, dict | None]:
    if not path:
        return None, None
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"Source run config does not exist: {absolute}")
    path_before = os.stat(absolute)
    try:
        with open(absolute, "rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            raw_payload = handle.read()
            descriptor_after = os.fstat(handle.fileno())
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed source run config: {absolute}") from exc
    path_after = os.stat(absolute)
    signature = (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_size,
        descriptor_before.st_mtime_ns,
    )
    if (
        signature
        != (
            descriptor_after.st_dev,
            descriptor_after.st_ino,
            descriptor_after.st_size,
            descriptor_after.st_mtime_ns,
        )
        or signature
        != (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
        )
        or signature
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
    ):
        raise RuntimeError(f"Source run config changed while being read: {absolute}")
    if not isinstance(payload, dict) or payload.get("run_status") != "complete":
        raise ValueError("source_config must be a completed clasp-eval run config")
    _validate_self_hash(payload, "run_config_sha256", label="source run config")
    provenance = payload.get("experiment_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source_config is missing experiment_provenance")
    expected_experiment = _experiment_fingerprint(provenance)
    if payload.get("experiment_fingerprint") != expected_experiment:
        raise ValueError("source_config experiment fingerprint is inconsistent")
    protocol_material = _protocol_fingerprint_material(provenance)
    if payload.get("protocol_fingerprint") != _experiment_fingerprint(protocol_material):
        raise ValueError("source_config protocol fingerprint is inconsistent")
    details_artifact = payload.get("details_artifact")
    if len(sources) != 1 or not isinstance(details_artifact, dict):
        raise ValueError("source_config requires exactly one content-bound details artifact")
    source = sources[0]
    if (
        details_artifact.get("sha256") != source["sha256"]
        or details_artifact.get("size_bytes") != source["size"]
    ):
        raise ValueError("source_config details artifact does not match --details")
    return payload, {
        "basename": os.path.basename(absolute),
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "size_bytes": len(raw_payload),
    }


def _validate_records_against_source_config(records: list[dict], config: dict) -> None:
    provenance = config["experiment_provenance"]
    expected_experiment = config["experiment_fingerprint"]
    expected_protocol = config["protocol_fingerprint"]
    expected_verifier = provenance.get("verifier")
    expected_training = config.get("training_run_contract_sha256")
    expected_tasks = set(config.get("tasks", []))
    expected_split = str(config.get("split", ""))
    selection = provenance.get("selection")
    expected_campaign = (
        selection.get("campaign_run_binding") if isinstance(selection, dict) else None
    )
    if config.get("evaluation_purpose") == "confirmation":
        if (
            not isinstance(expected_campaign, dict)
            or config.get("campaign_manifest") != selection.get("campaign_manifest")
            or config.get("campaign_run_binding") != expected_campaign
            or not isinstance(config.get("confirmation_attempt_claim"), dict)
            or config["confirmation_attempt_claim"].get("confirmation_attempt_id")
            != expected_campaign.get("confirmation_attempt_id")
        ):
            raise ValueError("source_config confirmation campaign binding is incomplete")
    details_artifact = config.get("details_artifact")
    if not isinstance(details_artifact, dict) or details_artifact.get("record_count") != len(
        records
    ):
        raise ValueError("source_config details record count is inconsistent")
    for index, record in enumerate(records):
        if "record_sha256" not in record:
            raise ValueError(f"source record {index} has no record_sha256")
        if record.get("experiment_fingerprint") != expected_experiment:
            raise ValueError(f"source record {index} experiment fingerprint mismatch")
        if record.get("protocol_fingerprint") != expected_protocol:
            raise ValueError(f"source record {index} protocol fingerprint mismatch")
        if record.get("strict_verifier") != expected_verifier:
            raise ValueError(f"source record {index} strict verifier mismatch")
        if record.get("training_run_contract_sha256") != expected_training:
            raise ValueError(f"source record {index} training contract mismatch")
        if expected_campaign is not None and (
            record.get("campaign_manifest_sha256")
            != expected_campaign.get("campaign_manifest_sha256")
            or record.get("confirmation_attempt_id")
            != expected_campaign.get("confirmation_attempt_id")
        ):
            raise ValueError(f"source record {index} confirmation campaign mismatch")
        if (
            str(record.get("task", "")) not in expected_tasks
            or str(record.get("split", "")) != expected_split
        ):
            raise ValueError(f"source record {index} lies outside source_config selection")


def run_rescore(args) -> tuple[str, str]:
    output_path = os.path.abspath(args.output)
    summary_path = f"{os.path.splitext(output_path)[0]}__summary.json"
    protected_inputs = {
        *(os.path.realpath(str(path)) for path in list(args.details)),
    }
    source_config_arg = getattr(args, "source_config", None)
    if source_config_arg:
        protected_inputs.add(os.path.realpath(str(source_config_arg)))
    if (
        os.path.realpath(output_path) in protected_inputs
        or os.path.realpath(summary_path) in protected_inputs
    ):
        raise ValueError("rescore output paths must not overwrite source artifacts")
    input_records, sources = _load_raw_records(list(args.details))
    source_config, source_config_identity = _load_source_config(
        getattr(args, "source_config", None),
        sources,
    )
    if source_config is not None:
        _validate_records_against_source_config(input_records, source_config)
    source_config_snapshot = (
        build_derived_source_config_snapshot(source_config, source_config_identity)
        if source_config is not None and source_config_identity is not None
        else None
    )
    verifier = _strict_verifier_identity()
    implementation_sha256, implementation_size = _sha256_file(os.path.abspath(__file__))
    fingerprint_material = {
        "schema_version": 1,
        "system_name": str(args.system_name),
        "sources": sources,
        "source_config": source_config_identity,
        "source_experiment_fingerprint": (
            source_config.get("experiment_fingerprint") if source_config else None
        ),
        "source_protocol_fingerprint": (
            source_config.get("protocol_fingerprint") if source_config else None
        ),
        "source_experiment_provenance": (
            source_config.get("experiment_provenance") if source_config else None
        ),
        "training_run_contract": (
            source_config.get("training_run_contract") if source_config else None
        ),
        "training_run_contract_sha256": (
            source_config.get("training_run_contract_sha256") if source_config else None
        ),
        "source_campaign_manifest": (
            source_config.get("campaign_manifest") if source_config else None
        ),
        "source_campaign_run_binding": (
            source_config.get("campaign_run_binding") if source_config else None
        ),
        "source_confirmation_attempt_claim": (
            source_config.get("confirmation_attempt_claim") if source_config else None
        ),
        "verifier": verifier,
        "rescore_implementation": {
            "basename": os.path.basename(__file__),
            "size": implementation_size,
            "sha256": implementation_sha256,
        },
    }
    rescore_fingerprint = _experiment_fingerprint(fingerprint_material)
    rescored = [
        _rescore_record(
            record,
            record_index=record_index,
            rescore_fingerprint=rescore_fingerprint,
            verifier=verifier,
        )
        for record_index, record in enumerate(input_records)
    ]

    per_task = {
        task: _aggregate_records([record for record in rescored if record["task"] == task])
        for task in sorted({record["task"] for record in rescored})
    }
    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_name": str(args.system_name),
        "system_lane": "strict_rescore_raw_generation",
        "rescore_fingerprint": rescore_fingerprint,
        "fingerprint_material": fingerprint_material,
        "source_experiment_fingerprint": (
            source_config.get("experiment_fingerprint") if source_config else None
        ),
        "source_protocol_fingerprint": (
            source_config.get("protocol_fingerprint") if source_config else None
        ),
        "source_experiment_provenance": (
            source_config.get("experiment_provenance") if source_config else None
        ),
        "training_run_contract": (
            source_config.get("training_run_contract") if source_config else None
        ),
        "training_run_contract_sha256": (
            source_config.get("training_run_contract_sha256") if source_config else None
        ),
        "source_campaign_manifest": (
            source_config.get("campaign_manifest") if source_config else None
        ),
        "source_campaign_run_binding": (
            source_config.get("campaign_run_binding") if source_config else None
        ),
        "source_confirmation_attempt_claim": (
            source_config.get("confirmation_attempt_claim") if source_config else None
        ),
        "source_config_snapshot": source_config_snapshot,
        "record_count": len(rescored),
        "overall": _aggregate_records(rescored),
        "per_task": per_task,
        "files": {
            "details": output_path,
            "summary": summary_path,
        },
    }
    _write_jsonl_atomic(output_path, rescored)
    summary["details_artifact"] = _details_artifact_identity(
        output_path,
        record_count=len(rescored),
    )
    summary = _self_hash_payload(summary, "summary_sha256")
    _write_json_atomic(summary_path, summary)
    return output_path, summary_path


def main() -> None:
    args = build_rescore_parser().parse_args()
    output_path, summary_path = run_rescore(args)
    print(f"Strictly re-scored details: {output_path}")
    print(f"Strict rescore summary: {summary_path}")


if __name__ == "__main__":
    main()
