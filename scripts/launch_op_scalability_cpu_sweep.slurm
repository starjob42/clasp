#!/bin/bash
# Development-only, one-thread CPU Gurobi sweep.  The vas CPU allocation was
# unavailable when this plan was frozen, so the exact submission uses the
# smallest A100 t3 cell while making the accelerator invisible to the process.
# The scientific solver remains Threads=1 CPU-only.
#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --job-name=op_scalability_dev_v1_iclr
#SBATCH --export=NONE
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_op_scalability_development_iclr_20260826_v1/logs/%x.%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_op_scalability_development_iclr_20260826_v1/logs/%x.%j.err

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly REMOTE_ROOT=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_op_scalability_development_iclr_20260826_v1
readonly PLAN_ROOT="${REMOTE_ROOT}/plan/op_scalability_development_v1"
readonly EXECUTION_ROOT="${REMOTE_ROOT}/execution/op_scalability_remote_execution_v1"
readonly CODE_ROOT="${REMOTE_ROOT}/code/correct_clasp_op_scalability_v1"
readonly RUNTIME_BUNDLE="${CODE_ROOT}/runtime_python_bundle_v1"
readonly SEALED_DIR="${REMOTE_ROOT}/inputs/sealed_gurobi_artifact_v1"
readonly SEALED_WHEEL="${SEALED_DIR}/gurobipy-13.0.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
readonly PUBLICATION_RECEIPT="${SEALED_DIR}/publication_receipt.json"
readonly CONTRACT="${EXECUTION_ROOT}/sweep_contract.remote_v1.json"
readonly OUTPUT_DIR="${REMOTE_ROOT}/results/op_scalability_sweep_v1"
readonly PYTHON_BIN=/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/gurobi_13.0.1_audit/venv/bin/python
readonly SUPPORT_SITE=/lustre/fswork/projects/rech/vas/uag72kz/clasp/publication_sft_env_20260820_v1_parent/publication-sft-environment/lib/python3.10/site-packages
readonly GUROBI_LICENSE=/lustre/fshomisc/sup/prod/gurobi/13.0.1/gurobi.lic
readonly PUBLISHER="${CODE_ROOT}/scripts/publish_op_scalability_gurobi_artifact.py"
readonly BUNDLE_VERIFIER="${CODE_ROOT}/scripts/verify_op_scalability_runtime_bundle.py"
readonly BUNDLE_MANIFEST="${EXECUTION_ROOT}/runtime_python_bundle_manifest_v1.json"
readonly SWEEP_RUNNER="${RUNTIME_BUNDLE}/scripts/run_op_scalability_sweep.py"
readonly EXACT_RUNNER="${RUNTIME_BUNDLE}/scripts/run_reference_evidence.py"
readonly EVALUATOR="${RUNTIME_BUNDLE}/src/clasp/entrypoints/evaluate.py"
readonly CO_TASKS="${RUNTIME_BUNDLE}/src/clasp/rewards/co_tasks.py"
readonly DATASET="${PLAN_ROOT}/op/eval/test.json"
readonly GENERATION_RECEIPT="${PLAN_ROOT}/generation_receipt.json"
readonly PUBLISHER_SHA256=3750e9452f926633dc9c5df2ffacc961e9b34073158c5ed139f52ebfaaa60112
readonly BUNDLE_VERIFIER_SHA256=453912e4ff7af7aab0413396f0424f5487bfa04b6caa2df63aeed16e0c0d96ee
readonly BUNDLE_MANIFEST_FILE_SHA256=aa3a4c4a26bceb16c57fec661d8494bd3f8fc5c58e9e1fb5e4227cb1dff4c45f
readonly SWEEP_RUNNER_SHA256=23be3fe6b37b50e871e6fe6b953d0df940e4b23582d462c146ab4ab6838c108d
readonly EXACT_RUNNER_SHA256=e31a17646b29f02734e742e66617cdb470b21f3c46a716153200118df67c9d22
readonly EVALUATOR_SHA256=e9ab65f8486ceacf7f8e020e48efc571a3f2915ce4d3a7db05813aa424198908
readonly CO_TASKS_SHA256=da72b7826e3cdb70c4092c69fc5af57c46048b38931479f6559988acac454d32
readonly DATASET_SHA256=8459e3fc1fc5f364d4ff19d47023132196316c096753711c8b90337a772603d1
readonly GENERATION_RECEIPT_FILE_SHA256=ac02bd68aee85846e500ea5204f8c93db64a4e828b8e834318767f73dbe4ef4b
readonly CONTRACT_FILE_SHA256=58595aed948b0ac1e11485cb6b0c1ebad4df30bd1ea678c1840b595b8722af42
readonly SEALED_WHEEL_SHA256=7fc1b5ff46530aee391ab0f616620a1399abea0778b25e374b37f7ed222d5dfb

fail() {
    /usr/bin/printf '[OPScalabilityLaunchError] %s\n' "$*" >&2
    exit 2
}

require_file() {
    local path="$1" expected_sha="$2" expected_mode="$3" mode nlink observed_sha
    [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/unsafe file ${path}"
    [[ "$(/usr/bin/readlink -f -- "${path}")" == "${path}" ]] || {
        fail "noncanonical file ${path}"
    }
    IFS=' ' read -r mode nlink < <(/usr/bin/stat -c '%a %h' -- "${path}")
    [[ "${mode}" == "${expected_mode}" && "${nlink}" == 1 ]] || {
        fail "file mode/link mismatch ${path}: ${mode}/${nlink}"
    }
    observed_sha="$(/usr/bin/sha256sum -- "${path}")"
    observed_sha="${observed_sha%% *}"
    [[ "${observed_sha}" == "${expected_sha}" ]] || {
        fail "file SHA-256 mismatch ${path}: ${observed_sha}"
    }
}

[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || fail 'missing/invalid SLURM_JOB_ID'
[[ "${SLURM_JOB_NAME:-}" == op_scalability_dev_v1_iclr ]] || {
    fail "unexpected job name ${SLURM_JOB_NAME:-<unset>}"
}
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 ]] || {
    fail "expected exactly one allocated CPU, got ${SLURM_CPUS_PER_TASK:-<unset>}"
}
[[ -d "${REMOTE_ROOT}" && ! -L "${REMOTE_ROOT}" ]] || fail 'remote root is missing/unsafe'
[[ -d "${RUNTIME_BUNDLE}" && ! -L "${RUNTIME_BUNDLE}" ]] || fail 'runtime bundle is missing/unsafe'
[[ -d "${SUPPORT_SITE}" && ! -L "${SUPPORT_SITE}" ]] || fail 'support site is missing/unsafe'
[[ -x "${PYTHON_BIN}" ]] || fail 'Gurobi Python is missing/nonexecutable'
[[ -f "${GUROBI_LICENSE}" && ! -L "${GUROBI_LICENSE}" ]] || fail 'Gurobi license is missing/unsafe'
[[ -d "${SEALED_DIR}" && ! -L "${SEALED_DIR}" ]] || fail 'sealed artifact directory missing/unsafe'
[[ "$(/usr/bin/stat -c '%a' -- "${SEALED_DIR}")" == 550 ]] || fail 'sealed directory mode is not 0550'
[[ ! -e "${OUTPUT_DIR}" && ! -L "${OUTPUT_DIR}" ]] || fail "refusing existing output ${OUTPUT_DIR}"

require_file "${PUBLISHER}" "${PUBLISHER_SHA256}" 440
require_file "${BUNDLE_VERIFIER}" "${BUNDLE_VERIFIER_SHA256}" 440
require_file "${BUNDLE_MANIFEST}" "${BUNDLE_MANIFEST_FILE_SHA256}" 440
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS
"${PYTHON_BIN}" -B -I "${BUNDLE_VERIFIER}" \
    --manifest "${BUNDLE_MANIFEST}" \
    --bundle-root "${RUNTIME_BUNDLE}" \
    --expected-manifest-file-sha256 "${BUNDLE_MANIFEST_FILE_SHA256}" >/dev/null || {
    fail 'exact runtime Python bundle verification failed'
}
require_file "${SWEEP_RUNNER}" "${SWEEP_RUNNER_SHA256}" 440
require_file "${EXACT_RUNNER}" "${EXACT_RUNNER_SHA256}" 440
require_file "${EVALUATOR}" "${EVALUATOR_SHA256}" 440
require_file "${CO_TASKS}" "${CO_TASKS_SHA256}" 440
require_file "${DATASET}" "${DATASET_SHA256}" 440
require_file "${GENERATION_RECEIPT}" "${GENERATION_RECEIPT_FILE_SHA256}" 440
require_file "${CONTRACT}" "${CONTRACT_FILE_SHA256}" 440
require_file "${SEALED_WHEEL}" "${SEALED_WHEEL_SHA256}" 440
[[ -f "${PUBLICATION_RECEIPT}" && ! -L "${PUBLICATION_RECEIPT}" ]] || {
    fail 'sealed artifact publication receipt missing/unsafe'
}

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${RUNTIME_BUNDLE}/src:${RUNTIME_BUNDLE}/scripts:${SUPPORT_SITE}"
export GRB_LICENSE_FILE="${GUROBI_LICENSE}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"${PYTHON_BIN}" "${PUBLISHER}" --verify-publication \
    --expected-publisher-sha256 "${PUBLISHER_SHA256}" >/dev/null || {
    fail 'sealed artifact publication verification failed'
}
"${PYTHON_BIN}" "${SWEEP_RUNNER}" --contract "${CONTRACT}" --preflight-only || {
    fail 'frozen sweep preflight failed'
}

exec "${PYTHON_BIN}" "${SWEEP_RUNNER}" \
    --contract "${CONTRACT}" \
    --output-dir "${OUTPUT_DIR}"
