"""Read-only gate on the already accepted repaired TSP SFT continuation.

No training, scheduling, retries or file writes occur here. The caller freezes
the returned initializer once; missing/unfinished predecessors return None.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess

from common import identity, read, require, verify

SOURCE_SHA256 = "3e1c5df5938a5505460d76cf27325b5c5bb9d177463be36063a6ca51dd6bfc68"
JOBS = ("1703800", "1703801", "1703802", "1703803")
JOB_NAMES = dict(zip(JOBS, ("tsp-lr1e4-s20000", "tsp-lr1e4-s25000",
                           "tsp-lr1e4-s30000", "tsp-lr1e4-s31250")))
EXPECTED_CHECKPOINT = Path(
    "/lustre/fsn1/projects/rech/vas/uag72kz/"
    "clasp_tsp_stability_lr1e4_sourcev18_production_gate_iclr_20260831_v1/"
    "runs/tsp/seed42/checkpoints/checkpoint-31250")
WAIT_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
               "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SPECIAL_EXIT"}


def _accounting():
    result = subprocess.run(
        ["sacct", "-n", "-X", "-P", "-j", ",".join(JOBS), "-S", "2026-09-01",
         "--format=JobIDRaw,State,ExitCode,JobName%80,Account,Partition"],
        check=True, capture_output=True, text=True, timeout=45)
    rows = {}
    for line in result.stdout.splitlines():
        cells = line.split("|")
        if cells and cells[-1] == "":
            cells.pop()
        require(len(cells) == 6, "Malformed TSP SFT accounting record")
        job, status, exit_code, name, account, partition = cells
        require(job in JOBS and job not in rows, "Ambiguous TSP SFT accounting")
        require(name == JOB_NAMES[job] and account == "vas@h100" and partition == "gpu_p6",
                "TSP SFT scheduler identity differs")
        rows[job] = {"job_id": job, "state": status.split()[0].rstrip("+"),
                     "exit_code": exit_code, "job_name": name,
                     "account": account, "partition": partition}
    return rows


def _source_inventory(root, manifest):
    names = set()
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        require(match is not None, "Malformed TSP SFT source manifest")
        digest, name = match.groups()
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name
                and name not in names, "Unsafe or duplicate TSP SFT source entry")
        require(identity(root / name)["sha256"] == digest, "TSP SFT source bytes changed")
        names.add(name)
    required = {"src/clasp/entrypoints/train_sft_jiang_f638.py",
                "src/clasp/entrypoints/train_sft_jiang_f638_tsp_stable.py",
                "scripts/train_tsp_sft_stability_h100.slurm",
                "scripts/audit_sft_reload_equivalence.py"}
    required.update(p.relative_to(root).as_posix() for p in (root / "src/clasp").rglob("*.py"))
    require(required <= names, "TSP SFT source manifest is incomplete")


def _verify_certificates(checkpoint, base_model):
    from clasp.models.checkpoint_fidelity import verify_checkpoint_fidelity
    from scripts.audit_sft_reload_equivalence import reload_equivalence_identity

    fidelity = verify_checkpoint_fidelity(checkpoint)
    reload_identity = reload_equivalence_identity(checkpoint, base_model_path=base_model)
    return fidelity["receipt_sha256"], reload_identity


def gate(declaration):
    """Return the certified shared initializer, or None while it is unfinished."""
    pending = declaration["tasks"]["tsp"]["initialization_pending"]
    require(tuple(str(job) for job in pending["jobs"]) == JOBS,
            "TSP SFT continuation job graph differs")
    checkpoint = Path(pending["checkpoint"])
    require(checkpoint == EXPECTED_CHECKPOINT, "TSP initializer is not the repaired terminal checkpoint")
    source = pending["source"]
    require(source["manifest_sha256"] == SOURCE_SHA256, "TSP SFT source declaration differs")
    root = Path(source["root"])
    manifest = root / "SOURCE_MANIFEST.sha256"
    require(identity(manifest)["sha256"] == SOURCE_SHA256, "TSP SFT source manifest changed")
    hyper_binding = pending["hyperparameters"]
    require(Path(hyper_binding["path"]) == checkpoint.parent.parent / "hyperparams.json",
            "TSP SFT immutable hyperparameters belong to another run")
    verify(hyper_binding)
    hyper = read(hyper_binding["path"])
    stage = hyper["stage_hyperparameters"]
    require(hyper["tasks"] == ["tsp"] and hyper["stage"] == "sft"
            and hyper["output_dir"] == str(checkpoint.parent.parent), "TSP SFT run identity differs")
    require(stage["learning_rate"] == 0.0001 and stage["expected_terminal_global_step"] == 31250
            and stage["effective_train_batch_size"] == 16, "TSP SFT stable training recipe differs")
    rows = _accounting()
    waiting = len(rows) != len(JOBS)
    for job in JOBS:
        row = rows.get(job)
        if row is None:
            continue
        if row["state"] == "COMPLETED":
            require(row["exit_code"] == "0:0", "TSP SFT completed with a nonzero exit")
        else:
            require(row["state"] in WAIT_STATES, "TSP SFT predecessor failed: " + row["state"])
            waiting = True
    if waiting:
        return None
    required = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_fidelity.json",
                "sft_reload_equivalence.json", "trainer_state.json")
    if not checkpoint.exists() or any(not (checkpoint / name).exists() for name in required):
        return None
    _source_inventory(root, manifest)
    state = read(checkpoint / "trainer_state.json")
    require(type(state.get("global_step")) is int and state["global_step"] == 31250
            and state.get("max_steps") == 31250 and state.get("epoch") == 1.0,
            "TSP SFT terminal trainer state differs")
    fidelity, reload_identity = _verify_certificates(checkpoint, hyper["model_path"])
    files = {name: identity(checkpoint / name) for name in required}
    return {"checkpoint": str(checkpoint), "files": files,
            "checkpoint_fidelity_receipt_sha256": fidelity,
            "sft_reload_equivalence": reload_identity,
            "completed_sft_jobs": [rows[job] for job in JOBS],
            "sft_source": source, "sft_hyperparameters": hyper_binding}
