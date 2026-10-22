#!/bin/bash -l
# One deterministic, solver-backed replacement shard.  The project CPU
# allocation is closed; this uses the smallest A100 cell but no GPU compute.
# Required: CLEAN_SFT_REPLACEMENT_SOURCE_ROOT, CLEAN_SFT_REPLACEMENT_TASK,
# CLEAN_SFT_REPLACEMENT_ROWS, CLEAN_SFT_REPLACEMENT_OUTPUT_ROOT. CVRP also
# requires CLEAN_SFT_REPLACEMENT_PYTHON and CLEAN_SFT_LKH_EXECUTABLE; MIS/OP
# require CLEAN_SFT_GUROBI_PYTHON, CLEAN_SFT_SUPPORT_SITE_PACKAGES, and a
# canonical read-only CLEAN_SFT_GUROBI_LICENSE_FILE.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=sft_replacements_iclr

set -euo pipefail
umask 027

SOURCE_ROOT="$(readlink -f "${CLEAN_SFT_REPLACEMENT_SOURCE_ROOT:?source root is required}")"
TASK="${CLEAN_SFT_REPLACEMENT_TASK:?task is required}"
ROWS="${CLEAN_SFT_REPLACEMENT_ROWS:?rows per shard is required}"
OUTPUT_ROOT="${CLEAN_SFT_REPLACEMENT_OUTPUT_ROOT:?output root is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${CLEAN_SFT_REPLACEMENT_SHARD_INDEX:-0}}"
BASE_SEED="${CLEAN_SFT_REPLACEMENT_BASE_SEED:-2026082100}"
OFFSET_BASE="${CLEAN_SFT_REPLACEMENT_OFFSET_BASE:-0}"
OFFSET_STRIDE="${CLEAN_SFT_REPLACEMENT_OFFSET_STRIDE:-1000000}"
OP_LOCAL_TEMP_ROOT=""
OP_LOCAL_TEMP_ID=""

cleanup_op_local_temp() {
    local observed_identity=""
    [[ -n "${OP_LOCAL_TEMP_ROOT:-}" ]] || return 0
    case "${OP_LOCAL_TEMP_ROOT}" in
        /tmp/clasp-sft-op.*) ;;
        *)
            echo "[ReplacementWarning] refusing unsafe OP temp cleanup target" >&2
            return 0
            ;;
    esac
    [[ -e "${OP_LOCAL_TEMP_ROOT}" || -L "${OP_LOCAL_TEMP_ROOT}" ]] || return 0
    [[ -d "${OP_LOCAL_TEMP_ROOT}" && ! -L "${OP_LOCAL_TEMP_ROOT}" ]] || {
        echo "[ReplacementWarning] refusing replaced OP temp cleanup target" >&2
        return 0
    }
    observed_identity="$(stat -Lc '%d:%i' -- "${OP_LOCAL_TEMP_ROOT}")" || return 0
    [[ -n "${OP_LOCAL_TEMP_ID}" && "${observed_identity}" == "${OP_LOCAL_TEMP_ID}" ]] || {
        echo "[ReplacementWarning] refusing changed OP temp cleanup target" >&2
        return 0
    }
    # A killed producer can leave its node-local shard fully sealed (0550),
    # which plain rm -rf cannot traverse and empty.  The random root is private,
    # non-symlink, and inode-pinned above; recursive chmod does not follow nested
    # symlinks.  Recheck the root identity after chmod before deleting anything.
    chmod -R u+w -- "${OP_LOCAL_TEMP_ROOT}" || {
        echo "[ReplacementWarning] could not make sealed OP temp tree removable" >&2
        return 0
    }
    observed_identity="$(stat -Lc '%d:%i' -- "${OP_LOCAL_TEMP_ROOT}")" || return 0
    [[ "${observed_identity}" == "${OP_LOCAL_TEMP_ID}" ]] || {
        echo "[ReplacementWarning] refusing changed OP temp cleanup target after chmod" >&2
        return 0
    }
    rm -rf -- "${OP_LOCAL_TEMP_ROOT}" || {
        echo "[ReplacementWarning] could not remove OP node-local temp root" >&2
        return 0
    }
}
trap cleanup_op_local_temp EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

case "${TASK}" in cvrp|mis|op) ;; *) echo "[ReplacementError] invalid task=${TASK}" >&2; exit 2 ;; esac
case "${OUTPUT_ROOT}" in /*) ;; *) echo "[ReplacementError] output root must be absolute" >&2; exit 2 ;; esac
[[ "${OUTPUT_ROOT}" == *"_iclr"* ]] || {
    echo "[ReplacementError] output root must contain _iclr: ${OUTPUT_ROOT}" >&2
    exit 2
}
for value in "${ROWS}" "${SHARD_INDEX}" "${BASE_SEED}" "${OFFSET_BASE}" "${OFFSET_STRIDE}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[ReplacementError] noncanonical integer: ${value}" >&2
        exit 2
    }
done
(( ROWS > 0 && OFFSET_STRIDE > 0 )) || {
    echo "[ReplacementError] rows and offset stride must be positive" >&2
    exit 2
}
CANDIDATE_OFFSET=$((OFFSET_BASE + SHARD_INDEX * OFFSET_STRIDE))
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}/shard-$(printf '%03d' "${SHARD_INDEX}")"

[[ -f "${SOURCE_ROOT}/scripts/generate_clean_sft_replacements.py" \
   && -f "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
   && -f "${SOURCE_ROOT}/scripts/generate_iid_benchmark.py" \
   && -f "${SOURCE_ROOT}/scripts/run_reference_evidence.py" \
   && -d "${SOURCE_ROOT}/src/clasp" \
   && ! -e "${OUTPUT_DIR}" ]] || {
    echo "[ReplacementError] source is incomplete or shard output already exists" >&2
    exit 2
}
module purge
unset GRB_LICENSE_FILE || true
module load arch/a100
module load miniforge/24.9.0
SUPPORT_SITE_PACKAGES=""
GUROBI_ENVIRONMENT_PREFIX=""
GUROBI_LICENSE_CANONICAL=""
if [[ "${TASK}" == "cvrp" ]]; then
    PYTHON_BIN="$(readlink -f "${CLEAN_SFT_REPLACEMENT_PYTHON:?CVRP replacement Python is required}")"
else
    # Keep the lexical virtual-environment entry point.  Resolving this final
    # symlink changes argv[0] from venv/bin/python to the underlying base-Conda
    # interpreter, so CPython no longer discovers the venv's pyvenv.cfg and
    # gurobipy disappears from sys.path.
    GUROBI_PYTHON_RAW="${CLEAN_SFT_GUROBI_PYTHON:?Gurobi Python is required}"
    [[ "${GUROBI_PYTHON_RAW}" == /* ]] || {
        echo "[ReplacementError] Gurobi Python path must be absolute" >&2
        exit 2
    }
    case "${GUROBI_PYTHON_RAW}" in
        */|*//*|*/./*|*/../*|*/.|*/..)
            echo "[ReplacementError] Gurobi Python path must be lexically normalized" >&2
            exit 2
            ;;
    esac
    PYTHON_BIN="${GUROBI_PYTHON_RAW}"
    GUROBI_ENVIRONMENT_PREFIX="$(dirname "$(dirname "${PYTHON_BIN}")")"
    [[ "${PYTHON_BIN}" == "${GUROBI_ENVIRONMENT_PREFIX}/bin/python" \
        && "$(readlink -f "${GUROBI_ENVIRONMENT_PREFIX}")" == "${GUROBI_ENVIRONMENT_PREFIX}" \
        && -L "${PYTHON_BIN}" \
        && -f "${GUROBI_ENVIRONMENT_PREFIX}/pyvenv.cfg" \
        && ! -L "${GUROBI_ENVIRONMENT_PREFIX}/pyvenv.cfg" ]] || {
        echo "[ReplacementError] Gurobi Python is not a virtual-environment entry point" >&2
        exit 2
    }
    SUPPORT_SITE_PACKAGES="$(readlink -f "${CLEAN_SFT_SUPPORT_SITE_PACKAGES:?support site-packages are required}")"
    [[ -d "${SUPPORT_SITE_PACKAGES}" ]] || {
        echo "[ReplacementError] support site-packages directory is missing" >&2
        exit 2
    }
    GUROBI_LICENSE_RAW="${CLEAN_SFT_GUROBI_LICENSE_FILE:?Gurobi license file is required}"
    [[ "${GUROBI_LICENSE_RAW}" == /* ]] || {
        echo "[ReplacementError] Gurobi license path must be absolute" >&2
        exit 2
    }
    case "${GUROBI_LICENSE_RAW}" in
        */|*//*|*/./*|*/../*|*/.|*/..)
            echo "[ReplacementError] Gurobi license path must be lexically normalized" >&2
            exit 2
            ;;
    esac
    GUROBI_LICENSE_CANONICAL="$(readlink -f -- "${GUROBI_LICENSE_RAW}")" || {
        echo "[ReplacementError] Gurobi license path cannot be resolved" >&2
        exit 2
    }
    [[ "${GUROBI_LICENSE_CANONICAL}" == "${GUROBI_LICENSE_RAW}" \
        && -f "${GUROBI_LICENSE_CANONICAL}" \
        && ! -L "${GUROBI_LICENSE_CANONICAL}" \
        && -r "${GUROBI_LICENSE_CANONICAL}" \
        && ! -w "${GUROBI_LICENSE_CANONICAL}" ]] || {
        echo "[ReplacementError] Gurobi license must be a canonical read-only regular file" >&2
        exit 2
    }
    export GRB_LICENSE_FILE="${GUROBI_LICENSE_CANONICAL}"
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "[ReplacementError] Python is not executable" >&2; exit 2; }
export PATH="$(dirname "${PYTHON_BIN}"):/usr/bin:/bin"
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS || true
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts${SUPPORT_SITE_PACKAGES:+:${SUPPORT_SITE_PACKAGES}}"

if [[ "${TASK}" != "cvrp" ]]; then
    "${PYTHON_BIN}" - "${GUROBI_ENVIRONMENT_PREFIX}" <<'PY'
import sys

import gurobipy

expected_prefix = sys.argv[1]
if sys.prefix != expected_prefix:
    raise SystemExit(
        f"[ReplacementError] Gurobi virtual environment was not activated: "
        f"expected={expected_prefix!r}, observed={sys.prefix!r}"
    )
if str(getattr(gurobipy, "__version__", "")) != "13.0.1":
    raise SystemExit(
        f"[ReplacementError] unexpected gurobipy version: {gurobipy.__version__!r}"
    )

# A restricted Gurobi license accepts tiny smoke models and then fails only
# after a real shard has begun.  Optimizing 2,001 binary variables crosses the
# restricted linear-model ceiling while remaining effectively free under the
# required full academic/site license.
environment = None
model = None
try:
    environment = gurobipy.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()
    model = gurobipy.Model("clasp_full_license_capacity_preflight", env=environment)
    variables = model.addVars(2001, vtype=gurobipy.GRB.BINARY, name="capacity_probe")
    model.setObjective(
        gurobipy.quicksum(variables[index] for index in range(2001)),
        gurobipy.GRB.MAXIMIZE,
    )
    model.optimize()
    if (
        int(model.Status) != int(gurobipy.GRB.OPTIMAL)
        or int(model.SolCount) <= 0
        or abs(float(model.ObjVal) - 2001.0) > 1e-9
    ):
        raise SystemExit("[ReplacementError] Gurobi full-license capacity probe did not solve")
except gurobipy.GurobiError as exc:
    error_code = int(getattr(exc, "errno", -1))
    raise SystemExit(
        f"[ReplacementError] Gurobi full-license capacity probe failed; code={error_code}"
    ) from None
finally:
    if model is not None:
        model.dispose()
    if environment is not None:
        environment.dispose()
PY
fi

if [[ "${TASK}" == "op" ]]; then
    OP_LOCAL_TEMP_ROOT="$(mktemp -d /tmp/clasp-sft-op.XXXXXXXXXX)"
    chmod 0700 "${OP_LOCAL_TEMP_ROOT}"
    [[ "$(readlink -f -- "${OP_LOCAL_TEMP_ROOT}")" == "${OP_LOCAL_TEMP_ROOT}" \
        && "$(dirname -- "${OP_LOCAL_TEMP_ROOT}")" == "/tmp" \
        && -d "${OP_LOCAL_TEMP_ROOT}" \
        && ! -L "${OP_LOCAL_TEMP_ROOT}" \
        && "$(stat -Lc '%a' -- "${OP_LOCAL_TEMP_ROOT}")" == "700" ]] || {
        echo "[ReplacementError] mktemp did not create a private canonical /tmp directory" >&2
        exit 2
    }
    OP_LOCAL_TEMP_ID="$(stat -Lc '%d:%i' -- "${OP_LOCAL_TEMP_ROOT}")"
    export CLEAN_SFT_OP_LOCAL_TEMP_ROOT="${OP_LOCAL_TEMP_ROOT}"
fi

mkdir -p "${OUTPUT_ROOT}/${TASK}"
cd "${SOURCE_ROOT}"
COMMON_ARGS=(
    --task "${TASK}"
    --count "${ROWS}"
    --candidate-offset "${CANDIDATE_OFFSET}"
    --base-seed "${BASE_SEED}"
    --output-dir "${OUTPUT_DIR}"
    --timeout-seconds "${CLEAN_SFT_REPLACEMENT_TIMEOUT_SECONDS:-300}"
    --max-attempt-multiplier "${CLEAN_SFT_REPLACEMENT_MAX_ATTEMPT_MULTIPLIER:-10}"
)
if [[ "${TASK}" == "cvrp" ]]; then
    LKH_EXECUTABLE="$(readlink -f "${CLEAN_SFT_LKH_EXECUTABLE:?frozen LKH executable is required}")"
    "${PYTHON_BIN}" scripts/generate_clean_sft_replacements.py \
        "${COMMON_ARGS[@]}" \
        --lkh-executable "${LKH_EXECUTABLE}"
elif [[ "${TASK}" == "op" ]]; then
    "${PYTHON_BIN}" scripts/generate_clean_sft_replacements.py \
        "${COMMON_ARGS[@]}" \
        --op-local-temp-root "${CLEAN_SFT_OP_LOCAL_TEMP_ROOT}" \
        --work-limit "${CLEAN_SFT_OP_WORK_LIMIT:-60}" \
        --node-limit "${CLEAN_SFT_OP_NODE_LIMIT:-10000}"
else
    "${PYTHON_BIN}" scripts/generate_clean_sft_replacements.py "${COMMON_ARGS[@]}"
fi
