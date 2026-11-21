"""Pure byte-bound recovery accounting; no scheduler or filesystem mutation.

The current v2 authority already reserves 84+8=92 GPU hours. Additional jobs
cannot borrow an unsubmitted statistical arm's reservation. A running snapshot
is a lower bound on allocated cost and never complete accounting. Revising the
authority requires a separately authorized protocol rather than dropping rows.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = "sft-anchor-recovery-accounting-certificate-v1"
NAMESPACE = "/lustre/fsn1/projects/rech/vas/uag72kz/rloo_sft_anchor_20261013"
AUTHORITY_SHA = "87f85e1d98f209d33555c8321f803e67afeb71ec6cf7a5217423ad91b0495034"
LAUNCH_SHA = "ef18aabb588cfcc05f58a8bcb80198540d33e7905e880fea7050c097f94e478a"
SOURCE_SHA = "1bca1dca69dc609c771e3c965a2889682333def68188ccdce35f74c0359c4b73"
HEX = re.compile(r"[0-9a-f]{64}")
ARMS = ("rloo_cardinal_beta005", "rloo_scalar_beta0", "foarl", "sft",
        "rloo_scalar_beta005", "rloo_cardinal_beta0")
ORDER = tuple(f"{task}.{arm}" for task in ("op", "cvrp") for arm in ARMS)
FAILED = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
STATES = FAILED | {"COMPLETED", "RUNNING"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def no_synthetic(value):
    if isinstance(value,dict):
        need(value.get("synthetic_fixture") is not True and value.get("synthetic_offline_fixture") is not True,
             "Synthetic fixture cannot certify production allocation accounting")
        for item in value.values():no_synthetic(item)
    elif isinstance(value,list):
        for item in value:no_synthetic(item)


def bound_bytes(value, *, path=None, cap=2 * 1024 * 1024):
    need(type(value) is dict and type(value.get("sha256")) is str and HEX.fullmatch(value["sha256"]) and
         type(value.get("size_bytes")) is int and 0 < value["size_bytes"] <= cap,
         "Finite exact byte binding required")
    try:
        raw = base64.b64decode(value["content_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed immutable byte snapshot") from exc
    need(len(raw) == value["size_bytes"] and sha(raw) == value["sha256"] and
         (path is None or value.get("native_path") == path), "Immutable byte/path binding differs")
    return raw


def reservations():
    rows = [{"reservation_id": "prior-v1/1030929", "gpus": 4, "requested_wall_seconds": 7200}]
    for slot in ORDER:
        task, arm = slot.split(".")
        rows.append({"reservation_id": "phase2/" + slot, "gpus": 1 if arm == "sft" else 4,
                     "requested_wall_seconds": 7200, "task": task, "arm": arm})
    return rows


def parse_accounting(value, job_id, expected_gpus):
    expected = ["sacct", "--noheader", "--parsable2", "--jobs", job_id,
                "--format=JobIDRaw,State%40,ElapsedRaw,ExitCode,AllocTRES"]
    need(type(value) is dict and value.get("argv") == expected and value.get("returncode") == 0 and value.get("stderr") == "",
         "Exact successful read-only sacct query required")
    raw = bound_bytes(value["stdout_binding"])
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("Non-UTF8 accounting snapshot") from exc
    parsed = {}
    for line in lines:
        parts = line.strip().split("|")
        if parts and parts[-1] == "": parts.pop()
        need(len(parts) == 5, "Malformed raw accounting row")
        identifier, state, elapsed, exit_code, tres = parts
        need((identifier == job_id or identifier.startswith(job_id + ".")) and identifier not in parsed and
             state in STATES and re.fullmatch(r"[0-9]+", elapsed) and re.fullmatch(r"[0-9]+:[0-9]+", exit_code),
             "Accounting contains unrelated/duplicate/unknown/truncated job state")
        resources = {}
        for item in tres.split(","):
            pieces = item.split("=", 1)
            need(len(pieces) == 2 and pieces[0] not in resources, "Malformed duplicate allocation resource")
            resources[pieces[0]] = pieces[1]
        gpu = resources.get("gres/gpu")
        need(type(gpu) is str and gpu.isdecimal() and int(gpu) == expected_gpus and int(gpu) > 0,
             "Positive actual GPU allocation must match the reserved job")
        need(state not in {"RUNNING", "COMPLETED"} or exit_code == "0:0", "Live/completed parent exit code differs")
        parsed[identifier] = {"state": state, "elapsed_seconds": int(elapsed), "exit_code": exit_code,
                              "gpus": int(gpu), "alloc_tres": tres}
    need(job_id in parsed, "Exact parent accounting record missing")
    need({job_id+".batch",job_id+".extern"} <= parsed.keys(), "Exact batch and extern accounting step records required")
    parent = parsed[job_id]
    if parent["state"] == "COMPLETED":
        need(all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in parsed.values()),
             "Completed parent contains failed or unfinished steps")
    if parent["state"] == "RUNNING":
        need(all(row["state"] in {"RUNNING", "COMPLETED"} and row["exit_code"] == "0:0" for row in parsed.values()),
             "Running parent contains failed/unknown steps")
    return parent, parsed, value["stdout_binding"]["sha256"]


def validate_accounting_certificate(accounting_certificate_path, *, expected_sha256,
        training_complete_sha256, pilot_contract_sha256, source_sha256, task, arm,
        current_job_id, authorized_requested_gpu_hours=92):
    """Verify a caller-pinned certificate, preserving every current reservation.

No fresh scheduler query occurs here. The caller must freeze the certificate
from authoritative read-only accounting and pin its exact bytes beforehand.
"""
    need(all(type(x) is str and HEX.fullmatch(x) for x in
             (expected_sha256, training_complete_sha256, pilot_contract_sha256, source_sha256)),
         "Caller must pin certificate/training/contract/source byte identities")
    need(task in ("op", "cvrp") and arm in ARMS and type(current_job_id) is str and current_job_id.isdecimal() and
         type(authorized_requested_gpu_hours) in (int, float) and 0 < authorized_requested_gpu_hours <= 92,
         "Exact current job/task/arm and bounded original authority required")
    path = Path(accounting_certificate_path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024 * 1024,
         "Missing/linked/oversized accounting certificate")
    raw = path.read_bytes()
    need(sha(raw) == expected_sha256, "Caller-pinned accounting certificate SHA differs")
    certificate = json.loads(raw)
    no_synthetic(certificate)
    need(type(certificate) is dict and certificate.get("schema_version") == 1 and certificate.get("protocol") == PROTOCOL and
         certificate.get("current_job_id") == current_job_id and certificate.get("synthetic_fixture") is not True,
         "Unknown/foreign/synthetic recovery accounting certificate")
    authority_raw = bound_bytes(certificate["reservation_authority"],
                               path=NAMESPACE + "/control_v2/campaign_authority.json")
    need(sha(authority_raw) == AUTHORITY_SHA, "Do not silently replace/reset the original reservation authority")
    authority = json.loads(authority_raw)
    need(authority.get("semantic_sha256") == canonical({k:v for k,v in authority.items() if k != "semantic_sha256"}) and
         authority.get("plan", {}).get("prepared_manifest_sha256") == LAUNCH_SHA and
         authority.get("plan", {}).get("source_manifest_sha256") == SOURCE_SHA and
         authority.get("plan", {}).get("order") == list(ORDER) and
         authority.get("combined_maximum_requested_gpu_hours") == 92,
         "Original reservation source/order/cost binding differs")
    declared = certificate.get("reservations")
    need(type(declared) is list and all(type(row) is dict for row in declared) and
         len({row.get("reservation_id") for row in declared}) == len(declared), "Unique closed reservation ledger required")
    required = reservations()
    indexed = {row["reservation_id"]:row for row in declared}
    need(all(indexed.get(row["reservation_id"]) == row for row in required),
         "Retain prior8h and all84h original planned slots, including unsubmitted arms")
    need(all(type(row.get("gpus")) is int and row["gpus"] > 0 and
             type(row.get("requested_wall_seconds")) is int and row["requested_wall_seconds"] > 0 for row in declared),
         "Positive finite requested reservations required")
    reserved_seconds = sum(row["gpus"] * row["requested_wall_seconds"] for row in declared)
    need(reserved_seconds <= authorized_requested_gpu_hours * 3600,
         "Additional recovery reservation exceeds the92h authority; do not drop unsubmitted slots")
    target = certificate.get("training_proof_binding", {})
    need(type(target) is dict and target == {"training_complete_sha256":training_complete_sha256,
                    "pilot_contract_sha256":pilot_contract_sha256,"source_sha256":source_sha256,
                    "task":task,"arm":arm,"training_parent_job_id":target.get("training_parent_job_id")} and
         type(target.get("training_parent_job_id")) is str and target["training_parent_job_id"].isdecimal(),
         "Accounting certificate does not bind the exact original training proof")
    jobs = certificate.get("jobs")
    need(type(jobs) is list and jobs and all(type(row) is dict for row in jobs),
         "Actual prior and current job accounting snapshots required")
    seen_jobs, seen_reservations, verified = set(),set(),[]
    for row in jobs:
        identifier, reservation_id = row.get("job_id"),row.get("reservation_id")
        need(type(identifier) is str and identifier.isdecimal() and identifier not in seen_jobs and
             reservation_id in indexed and reservation_id not in seen_reservations,
             "Duplicate allocation or two jobs consuming one reservation forbidden")
        seen_jobs.add(identifier);seen_reservations.add(reservation_id);reservation=indexed[reservation_id]
        parent, steps, raw_sha = parse_accounting(row["accounting"],identifier,reservation["gpus"])
        if reservation_id == "prior-v1/1030929":
            need(identifier == "1030929" and parent["state"] == "FAILED" and parent["exit_code"] == "1:0" and
                 parent["elapsed_seconds"] == 459, "Original failed prior phase and actual cost must be retained")
            need(steps[identifier+".batch"]["state"]=="FAILED" and steps[identifier+".batch"]["exit_code"]=="1:0" and
                 steps[identifier+".extern"]["state"]=="COMPLETED" and steps[identifier+".extern"]["exit_code"]=="0:0",
                 "Original prior batch failure and completed extern step must be retained")
        else:
            slot = reservation_id.removeprefix("phase2/")
            need(slot in ORDER, "New recovery allocation requires a separately revised authority")
            attempt_raw=bound_bytes(row["attempt_binding"],path=NAMESPACE+"/control_v2/attempts/"+slot+".json")
            attempt=json.loads(attempt_raw)
            need(attempt.get("state")=="submitted" and attempt.get("study")=="rloo-sft-anchor-20261013-phase2" and
                 attempt.get("slot")==slot and attempt.get("job_id")==identifier and attempt.get("manifest_sha256")==LAUNCH_SHA and
                 attempt.get("requested_gpu_hours")==reservation["gpus"]*reservation["requested_wall_seconds"]/3600,
                 "Actual attempt ledger job/slot/recipe/resource binding differs")
        if identifier in (target["training_parent_job_id"],current_job_id):
            need(reservation.get("task")==task and reservation.get("arm")==arm,
                 "Recovery cannot borrow a different task or arm's reservation")
        if identifier==current_job_id:
            need(parent["state"]=="RUNNING", "Current SLURM_JOB_ID requires positive RUNNING allocated GPU proof")
        if identifier==target["training_parent_job_id"] and identifier!=current_job_id:
            need(parent["state"] in FAILED, "Retain the original failed training allocation without relabelling it complete")
        verified.append({"job_id":identifier,"reservation_id":reservation_id,**parent,
                         "requested_wall_seconds":reservation["requested_wall_seconds"],
                         "accounting_stdout_sha256":raw_sha,"accounting_step_records":steps})
    need({"1030929",current_job_id,target["training_parent_job_id"]} <= seen_jobs,
         "Original prior failure, training parent or current recovery allocation is missing")
    return {"protocol":PROTOCOL,"accounting_certificate_sha256":expected_sha256,
            "accounting_certificate_verified":True,"reservation_authority_sha256":AUTHORITY_SHA,
            "reservations":declared,"records":verified,
            "total_requested_gpu_hours":reserved_seconds/3600,
            "submitted_requested_gpu_hours":sum(row["gpus"]*row["requested_wall_seconds"] for row in verified)/3600,
            "total_allocated_gpu_seconds":sum(row["gpus"]*row["elapsed_seconds"] for row in verified),
            "allocation_accounting_complete":False,"allocated_cost_is_lower_bound_running_snapshot":True,
            "prior_scheduler_failure_preserved":True,"authorized_requested_gpu_hours":authorized_requested_gpu_hours,
            "training_complete_sha256":training_complete_sha256,"current_job_id":current_job_id,
            "new_job_authorized":False,"native_queries_or_mutations":False}
