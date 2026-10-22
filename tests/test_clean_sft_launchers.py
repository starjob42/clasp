from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_audit_worker_matches_publication_reservations_and_jssp_repair() -> None:
    worker = (ROOT / "scripts" / "audit_clean_sft500k_cpu.slurm").read_text(
        encoding="utf-8"
    )
    assert "CLEAN_SFT_PYTHON:?" in worker
    assert "CLEAN_SFT_CONDA_ENV" not in worker
    for source_class in ("sft_eval", "rl_eval", "rl_train", "iid_confirmation"):
        assert f'--reserved-source "{source_class},' in worker
    assert "--repair-jssp-terminal-makespan" in worker
    assert "CLEAN_SFT_REPLACEMENT_ROOT" in worker
    assert "CLEAN_SFT_REQUIRE_ZERO_DEFICIT" in worker
    assert 'require_zero_text == "1"' in worker
    assert "_iclr" in worker


def test_publication_build_worker_is_create_once_and_audit_gated() -> None:
    worker = (ROOT / "scripts" / "build_clean_sft500k_a100.slurm").read_text(
        encoding="utf-8"
    )
    assert "CLEAN_SFT_FINAL_AUDIT_RECEIPT:?" in worker
    assert 'audit.get("additional_replacements_required") != 0' in worker
    assert '[[ ! -e "${OUTPUT_JSON}" && ! -e "${TASK_RECEIPT}" ]]' in worker
    assert "--repair-jssp-terminal-makespan" in worker
    for source_class in ("sft_eval", "rl_eval", "rl_train", "iid_confirmation"):
        assert f'--reserved-source "{source_class},' in worker
    assert '"${REPLACEMENT_ARGS[@]}"' in worker
    assert "data root must contain _iclr" in worker


def test_replacement_worker_has_no_unsealed_environment_or_solver_defaults() -> None:
    worker = (
        ROOT / "scripts" / "generate_clean_sft_replacements_a100.slurm"
    ).read_text(encoding="utf-8")
    assert "new_unsloth_env" not in worker
    assert "CLEAN_SFT_REPLACEMENT_PYTHON:?" in worker
    assert "CLEAN_SFT_GUROBI_PYTHON:?" in worker
    assert "CLEAN_SFT_LKH_EXECUTABLE:?" in worker
    assert "output root must contain _iclr" in worker
    assert 'PYTHON_BIN="$(readlink -f "${CLEAN_SFT_GUROBI_PYTHON' not in worker
    assert 'PYTHON_BIN="$(readlink -m -- "${GUROBI_PYTHON_RAW}")"' not in worker
    assert 'PYTHON_BIN="${GUROBI_PYTHON_RAW}"' in worker
    assert "*/|*//*|*/./*|*/../*|*/.|*/..)" in worker
    assert '"${PYTHON_BIN}" == "${GUROBI_ENVIRONMENT_PREFIX}/bin/python"' in worker
    assert '-L "${PYTHON_BIN}"' in worker
    assert '"${GUROBI_ENVIRONMENT_PREFIX}/pyvenv.cfg"' in worker
    assert '! -L "${GUROBI_ENVIRONMENT_PREFIX}/pyvenv.cfg"' in worker
    assert "if sys.prefix != expected_prefix:" in worker
    assert 'gurobipy, "__version__", ""' in worker
    assert '!= "13.0.1"' in worker


def test_replacement_worker_pins_and_capacity_tests_full_gurobi_license() -> None:
    worker = (
        ROOT / "scripts" / "generate_clean_sft_replacements_a100.slurm"
    ).read_text(encoding="utf-8")
    assert "CLEAN_SFT_GUROBI_LICENSE_FILE:?" in worker
    assert '[[ "${GUROBI_LICENSE_RAW}" == /* ]]' in worker
    assert 'GUROBI_LICENSE_CANONICAL="$(readlink -f -- "${GUROBI_LICENSE_RAW}")"' in worker
    assert '-f "${GUROBI_LICENSE_CANONICAL}"' in worker
    assert '! -L "${GUROBI_LICENSE_CANONICAL}"' in worker
    assert '-r "${GUROBI_LICENSE_CANONICAL}"' in worker
    assert '! -w "${GUROBI_LICENSE_CANONICAL}"' in worker
    assert 'export GRB_LICENSE_FILE="${GUROBI_LICENSE_CANONICAL}"' in worker
    assert "model.addVars(2001" in worker
    assert "model.optimize()" in worker
    assert "gurobipy.GRB.OPTIMAL" in worker
    module_purge = worker.index("module purge")
    inherited_unset = worker.index("unset GRB_LICENSE_FILE")
    pinned_export = worker.index("export GRB_LICENSE_FILE")
    capacity_probe = worker.index("model.addVars(2001")
    staging = worker.index('mkdir -p "${OUTPUT_ROOT}/${TASK}"')
    assert module_purge < inherited_unset < pinned_export < capacity_probe < staging
    assert 'cat "${GUROBI_LICENSE' not in worker
    assert 'echo "${GUROBI_LICENSE' not in worker


def test_replacement_worker_stages_op_live_logs_in_private_node_local_tmp() -> None:
    worker = (
        ROOT / "scripts" / "generate_clean_sft_replacements_a100.slurm"
    ).read_text(encoding="utf-8")
    assert "OP_LOCAL_TEMP_ROOT=\"\"" in worker
    assert "OP_LOCAL_TEMP_ID=\"\"" in worker
    assert "trap cleanup_op_local_temp EXIT" in worker
    assert 'chmod -R u+w -- "${OP_LOCAL_TEMP_ROOT}"' in worker
    assert "refusing changed OP temp cleanup target after chmod" in worker
    assert "mktemp -d /tmp/clasp-sft-op.XXXXXXXXXX" in worker
    assert 'chmod 0700 "${OP_LOCAL_TEMP_ROOT}"' in worker
    assert "stat -Lc '%d:%i'" in worker
    assert 'export CLEAN_SFT_OP_LOCAL_TEMP_ROOT="${OP_LOCAL_TEMP_ROOT}"' in worker
    assert '--op-local-temp-root "${CLEAN_SFT_OP_LOCAL_TEMP_ROOT}"' in worker
    capacity_probe = worker.index("model.optimize()")
    local_creation = worker.index("mktemp -d /tmp/clasp-sft-op.XXXXXXXXXX")
    staging = worker.index('mkdir -p "${OUTPUT_ROOT}/${TASK}"')
    solver = worker.index('scripts/generate_clean_sft_replacements.py \\\n', staging)
    assert capacity_probe < local_creation < staging < solver


def test_finalize_worker_freezes_both_registries_after_cleanliness_merge() -> None:
    worker = (
        ROOT / "scripts" / "finalize_clean_sft500k_iclr_a100.slurm"
    ).read_text(encoding="utf-8")
    merge = worker.index("--merge-task-receipt")
    reserved = worker.index('"${PYTHON_BIN}" scripts/freeze_sft_reserved_source_registry.py')
    forbidden = worker.index('"${PYTHON_BIN}" scripts/freeze_forbidden_source_registry.py')
    assert merge < reserved < forbidden
    assert "--base-sft-post-iid-generation-receipt" in worker
    assert "output namespace must contain _iclr" in worker


def test_token_worker_runs_task_shards_then_a_create_once_merge() -> None:
    worker = (
        ROOT / "scripts" / "audit_sft_token_lengths_a100.slurm"
    ).read_text(encoding="utf-8")
    assert "CLEAN_SFT_TOKEN_AUDIT_MODE" in worker
    assert "--batch-size 256" in worker
    assert "--max-seq-length 20000" in worker
    assert "--expected-rows-per-task 500000" in worker
    assert "--merge-shard-dir" in worker
    assert 'OUTPUT="${DATA_ROOT}/sft_token_length_audit.json"' in worker
    assert "data and shard roots must contain _iclr" in worker
