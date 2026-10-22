"""New cumulative92+ inference accounting; unchanged legacy92 helper retained.

Pure externally byte-bound certificate validation. No queries, allocation,
reservation substitution or automatic retry occurs here. Synthetic certificates
can exercise smoke mechanics but cannot certify native production accounting.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from clasp.baselines import sft_anchor_recovery_accounting as old

PROTOCOL = "sft-anchor-terminal-evaluation-accounting-v1"
need = old.need


def admit_authority(path, *, expected_sha256, expected_cap, preparation, authority_raw=None):
    raw = Path(path).read_bytes() if authority_raw is None else authority_raw
    need(old.sha(raw) == expected_sha256, "Revised authority SHA differs")
    value = json.loads(raw)
    if preparation["scope"] == "production": old.no_synthetic(value)
    need(value.get("protocol") == "sft-anchor-terminal-evaluation-operator-v1-cumulative-authority" and
         value.get("semantic_sha256") == old.canonical({key: item for key, item in value.items() if key != "semantic_sha256"}),
         "New revised authority protocol/seal differs")
    prior = old.bound_bytes(value["original_authority"], path=old.NAMESPACE + "/control_v2/campaign_authority.json")
    need(old.sha(prior) == old.AUTHORITY_SHA and value.get("original_reservations") == old.reservations(),
         "Do not reset or borrow the original92-hour reservation ledger")
    extras = value.get("rescue_reservations"); need(type(extras) is list and extras, "Explicit additional inference reservations required")
    slots = set()
    for row in extras:
        slot = row.get("slot"); wall = row.get("requested_wall_seconds")
        need(slot in old.ORDER and not slot.endswith(".sft") and slot not in slots and
             row.get("reservation_id") == "evalrescue-v1/" + slot and row.get("gpus") == 1 and
             type(wall) is int and wall == 7200 and row.get("attempt_limit") == 1 and row.get("purpose") == "terminal_evaluation_only",
             "One explicit bounded1-GPU terminal-only first attempt per slot required")
        slots.add(slot)
        timing = row.get("measured_runtime", {})
        if timing is None and slot != preparation["slot"]: continue
        data = old.bound_bytes(timing["progress_binding"], cap=128 * 1024 * 1024)
        groups = [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n")]
        need(groups and timing.get("complete_groups") == len(groups) and type(timing.get("planned_wall_seconds")) is int and 0 < timing["planned_wall_seconds"] <= wall and
             all(type(group.get("wall_seconds")) in (int, float) and math.isfinite(group["wall_seconds"]) and group["wall_seconds"] > 0 for group in groups) and
             math.isclose(sum(group["wall_seconds"] for group in groups), timing.get("measured_group_seconds", -1), rel_tol=1e-12, abs_tol=1e-12),
             "Replay actual complete group timer bytes for every new reservation")
        if slot == preparation["slot"]:
            need(old.sha(data) == preparation["prior_progress"]["sha256"], "Target measured runtime is not its original admitted prefix")
    reserved = 92 + sum(row["requested_wall_seconds"] / 3600 for row in extras)
    need(slots == {item for item in old.ORDER if not item.endswith(".sft")}, "Retain all ten extra inference reservations, including unsubmitted slots")
    need(type(expected_cap) in (int, float) and math.isfinite(expected_cap) and expected_cap > 92 and
         value.get("cumulative_maximum_requested_gpu_hours") == expected_cap == reserved == 112 and
         preparation["slot"] in slots and value.get("no_training") is value.get("no_checkpoint_selection") is True,
         "Caller-pinned cumulative cap/source slot/terminal-only scope differs")
    return value


def validate(certificate_path, *, expected_sha256, preparation, current_job_id,
             revised_authority_path, revised_authority_sha256, expected_cumulative_gpu_hours,
             original_attempt_inventory_sha256, rescue_attempt_inventory_sha256):
    path = Path(certificate_path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024 * 1024,
         "Stable bounded new accounting certificate required")
    raw = path.read_bytes(); need(old.sha(raw) == expected_sha256, "New accounting certificate bytes differ")
    value = json.loads(raw)
    scope = preparation["scope"]
    if scope == "production": old.no_synthetic(value)
    need(value.get("protocol") == PROTOCOL and value.get("schema_version") == 1 and
         value.get("preparation_semantic_sha256") == preparation["semantic_sha256"] and
         value.get("current_job_id") == current_job_id and type(current_job_id) is str and current_job_id.isdecimal(),
         "Current inference job and exact preparation binding required")
    authority = admit_authority(revised_authority_path, expected_sha256=revised_authority_sha256,
        expected_cap=expected_cumulative_gpu_hours, preparation=preparation)
    need(preparation["cumulative_authority"]["binding"]["sha256"] == revised_authority_sha256 and
         preparation["cumulative_authority"]["rescue_reservations"] == authority["rescue_reservations"], "Prepared cumulative authority changed")
    originals = old.reservations(); extras = authority["rescue_reservations"]
    reservations = originals + [{key: row[key] for key in ("reservation_id", "slot", "gpus", "requested_wall_seconds", "attempt_limit", "purpose")} for row in extras]
    need(value.get("reservations") == reservations, "Keep all92 old reservations and every explicit new inference reservation")
    indexed = {row["reservation_id"]: row for row in reservations}
    historical_authorities = set()
    histories = value.get("authority_history_bindings", [])
    need(type(histories) is list, "Preserve exact prior extra-authority snapshots when measured target changes")
    for history in histories:
        slot = history.get("slot")
        history_raw = old.bound_bytes(history["authority_binding"], cap=128 * 1024 * 1024)
        history_value = json.loads(history_raw)
        target = [row for row in history_value.get("rescue_reservations", []) if row.get("slot") == slot]
        need(len(target) == 1 and target[0].get("measured_runtime") is not None,
             "Historical extra-authority requires its actually measured logical target")
        historical_preparation = {**preparation, "slot": slot,
            "prior_progress": {"sha256": target[0]["measured_runtime"]["progress_binding"]["sha256"]}}
        admitted = admit_authority(None, expected_sha256=old.sha(history_raw), expected_cap=expected_cumulative_gpu_hours,
            preparation=historical_preparation, authority_raw=history_raw)
        historical_reservations = admitted["original_reservations"] + [{key: row[key] for key in
            ("reservation_id", "slot", "gpus", "requested_wall_seconds", "attempt_limit", "purpose")} for row in admitted["rescue_reservations"]]
        need(historical_reservations == reservations and old.sha(history_raw) not in historical_authorities,
             "Historical timing updates cannot drop, borrow or enlarge original/additional reservations")
        historical_authorities.add(old.sha(history_raw))
    jobs = value.get("jobs"); need(type(jobs) is list and jobs and all(type(row) is dict for row in jobs), "Actual charged allocation snapshots required")
    closed_attempts = value.get("original_attempts")
    need(type(closed_attempts) is list and all(type(row) is dict for row in closed_attempts), "Closed submitted original attempt inventory required")
    attempts = {}; prior_jobs = set()
    for item in closed_attempts:
        reservation_id = item.get("reservation_id", "")
        slot = reservation_id.removeprefix("phase2/")
        need(reservation_id == "phase2/" + slot and slot in old.ORDER and slot not in attempts and
             item.get("attempt_index") == 1 and item.get("kind") == "original_source2",
             "Unique first original submitted slot inventory required")
        snapshot = old.bound_bytes(item["attempt_binding"], path=old.NAMESPACE + "/control_v2/attempts/" + slot + ".json")
        attempt = json.loads(snapshot)
        script_path = old.NAMESPACE + "/control_v2/job_scripts/" + slot + ".slurm"
        script = old.bound_bytes(item["script_binding"], path=script_path)
        need(attempt.get("state") == "submitted" and attempt.get("slot") == slot and
             attempt.get("schema_version") == 2 and attempt.get("script_sha256") == old.sha(script) and
             attempt.get("argv") == ["sbatch", "--parsable", script_path] and
             attempt.get("automatic_retry") is False and attempt.get("automatic_cancellation") is False and
             attempt.get("returncode") == 0 and attempt.get("stderr") == "" and
             attempt.get("stdout", "").strip().split(";")[0] == attempt.get("job_id") and
             attempt.get("study") == "rloo-sft-anchor-20261013-phase2" and attempt.get("manifest_sha256") == old.LAUNCH_SHA and
             type(attempt.get("job_id")) is str and attempt["job_id"].isdecimal() and attempt["job_id"] not in prior_jobs and
             attempt.get("requested_gpu_hours") == (2 if slot.endswith(".sft") else 8),
             "Original attempt identity/source/cost differs")
        attempts[slot] = attempt; prior_jobs.add(attempt["job_id"])
    inventory_raw = old.bound_bytes(value["original_attempt_inventory_binding"])
    inventory = json.loads(inventory_raw)
    need(inventory.get("protocol") == "closed-original-phase2-attempt-inventory-v1" and
         inventory.get("source_manifest_sha256") == old.SOURCE_SHA and
         inventory.get("entries") == closed_attempts,
         "Caller-pinned complete original attempt inventory differs")
    listing = json.loads(old.bound_bytes(inventory["directory_listing_binding"]))
    need(listing.get("directory") == old.NAMESPACE + "/control_v2/attempts" and
         listing.get("regular_json_files") == sorted(slot + ".json" for slot in attempts) and
         listing.get("unresolved_intents") == [],
         "Reconcile exact native original attempt directory; ambiguous attempts cannot be omitted")
    need(value.get("original_attempt_inventory_sha256") == original_attempt_inventory_sha256 == old.sha(inventory_raw),
         "Caller-pinned complete original attempt inventory SHA differs")
    extra_inventory_raw = old.bound_bytes(value["rescue_attempt_inventory_binding"])
    need(old.sha(extra_inventory_raw) == rescue_attempt_inventory_sha256,
         "Caller-pinned complete extra-attempt inventory SHA differs")
    extra_inventory = json.loads(extra_inventory_raw)
    need(extra_inventory.get("protocol") == "closed-terminal-evaluation-attempt-inventory-v1" and
         type(extra_inventory.get("entries")) is list, "Closed additional inference attempt inventory required")
    extra_index = {}; extra_jobs = set()
    for wrapper in extra_inventory["entries"]:
        reservation_id = wrapper.get("reservation_id"); slot = reservation_id.removeprefix("evalrescue-v1/") if type(reservation_id) is str else ""
        need(reservation_id in indexed and reservation_id.startswith("evalrescue-v1/") and slot not in extra_index and
             wrapper.get("attempt_index") == 1 and wrapper.get("kind") == "one_gpu_evaluation_only",
             "One first extra attempt per reserved logical slot required")
        snapshot = old.bound_bytes(wrapper["attempt_binding"])
        attempt = json.loads(snapshot)
        need(attempt.get("slot") == slot and type(attempt.get("job_id")) is str and attempt["job_id"].isdecimal() and
             attempt["job_id"] not in extra_jobs and attempt.get("state") == "submitted",
             "Exact submitted extra job inventory required; unresolved dispatch cannot disappear")
        extra_index[slot] = old.sha(snapshot); extra_jobs.add(attempt["job_id"])
    extra_listing = json.loads(old.bound_bytes(extra_inventory["directory_listing_binding"]))
    need(extra_listing.get("directory") == old.NAMESPACE + "/evaluation_rescue_operator_v1_control/attempts" and
         extra_listing.get("regular_json_files") == sorted(slot + ".json" for slot in extra_index) and
         extra_listing.get("unresolved_intents") == [], "Extra failed/ambiguous attempts cannot be dropped from native listing")
    seen_jobs = set(); seen_reservations = set(); verified = []; extra_attempts = []
    for row in jobs:
        job = row.get("job_id"); reservation_id = row.get("reservation_id")
        need(type(job) is str and job.isdecimal() and job not in seen_jobs and
             reservation_id in indexed and reservation_id not in seen_reservations,
             "Do not retry, double-charge or borrow another slot's reservation")
        seen_jobs.add(job); seen_reservations.add(reservation_id); reservation = indexed[reservation_id]
        parent, steps, stdout_sha = old.parse_accounting(row["accounting"], job, reservation["gpus"])
        need(all(item["elapsed_seconds"] <= reservation["requested_wall_seconds"] for item in steps.values()),
             "Actual charged elapsed exceeds its fixed reserved allocation ceiling")
        actual_wall = reservation["requested_wall_seconds"]
        if reservation_id == "prior-v1/1030929":
            need(job == "1030929" and parent["state"] == "FAILED" and parent["elapsed_seconds"] == 459 and parent["exit_code"] == "1:0" and
                 steps[job + ".batch"]["state"] == "FAILED" and steps[job + ".extern"]["state"] == "COMPLETED",
                 "Preserve original prior1030929 failure and charged459seconds")
        elif reservation_id.startswith("phase2/"):
            slot = reservation_id.removeprefix("phase2/")
            need(slot in attempts and attempts[slot]["job_id"] == job, "Charged original job not bound to closed attempt inventory")
            if job == preparation["original_parent_job_id"]:
                need(slot == preparation["slot"] and parent["state"] in old.FAILED and
                     all(item["state"] != "RUNNING" for item in steps.values()), "Failed original training parent must remain closed and failed")
        else:
            slot = reservation["slot"]
            snapshot = old.bound_bytes(row["attempt_binding"])
            attempt = json.loads(snapshot)
            need(extra_index.get(slot) == old.sha(snapshot), "Actual extra allocation differs from independently closed attempt inventory")
            need(attempt.get("protocol") == "sft-anchor-terminal-evaluation-attempt-v1" and
                 attempt.get("slot") == slot and attempt.get("job_id") == job and attempt.get("kind") == "one_gpu_evaluation_only" and
                 attempt.get("preparation_semantic_sha256") and
                 attempt.get("authority_sha256") in historical_authorities | {revised_authority_sha256} and
                 attempt.get("gpus") == 1 and type(attempt.get("wall_seconds")) is int and 0 < attempt["wall_seconds"] <= reservation["requested_wall_seconds"] and
                 attempt.get("training_or_resume") is False, "New inference-only first attempt/source/authority identity differs")
            extra_attempts.append({"slot": slot, "job_id": job, "binding_sha256": old.sha(snapshot)})
            actual_wall = attempt["wall_seconds"]
            need(all(item["elapsed_seconds"] <= actual_wall for item in steps.values()),
                 "Actual elapsed exceeds the submitted extra-attempt wall limit")
            if job == current_job_id:
                need(slot == preparation["slot"] and attempt.get("preparation_semantic_sha256") == preparation["semantic_sha256"] and
                     attempt.get("authority_sha256") == revised_authority_sha256 and
                     parent["state"] == "RUNNING", "Current exact one-GPU inference allocation required")
        verified.append({"job_id": job, "reservation_id": reservation_id, **parent,
            "accounting_stdout_sha256": stdout_sha, "steps": steps, "requested_wall_seconds": actual_wall,
            "reserved_wall_ceiling_seconds": reservation["requested_wall_seconds"]})
    need({"1030929", current_job_id, preparation["original_parent_job_id"]} <= seen_jobs and prior_jobs <= seen_jobs and extra_jobs <= seen_jobs,
         "Every original charged attempt, prior failure, parent and current rescue must remain in cumulative accounting")
    need(any(row["job_id"] == current_job_id and row["reservation_id"] == "evalrescue-v1/" + preparation["slot"] for row in verified),
         "Current allocation cannot reuse a statistical training reservation")
    return {"protocol": PROTOCOL, "certificate_sha256": expected_sha256,
        "certificate_verified": True, "native_production_accounting_verified": scope == "production",
        "original_attempt_inventory_sha256": old.sha(inventory_raw), "records": verified, "extra_attempts": extra_attempts,
        "rescue_attempt_inventory_sha256": old.sha(extra_inventory_raw),
        "original_reserved_gpu_hours": 92, "cumulative_maximum_requested_gpu_hours": expected_cumulative_gpu_hours,
        "total_reserved_gpu_hours": sum(row["gpus"] * row["requested_wall_seconds"] for row in reservations) / 3600,
        "total_allocated_gpu_seconds": sum(row["gpus"] * row["elapsed_seconds"] for row in verified),
        "allocated_cost_is_lower_bound_running_snapshot": True, "allocation_accounting_complete": False,
        "no_original_reservation_borrowed_or_dropped": True, "native_queries_or_mutations": False}
