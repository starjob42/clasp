#!/bin/bash -l
# Paired held-out CVRP evaluation of completed classic-GRPO and CLASP models.
# Both checkpoints must use the normalized Jiang-f638 representation.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=cvrp-grpo-clasp-val
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${PAIR_EVAL_PROJECT_ROOT:?PAIR_EVAL_PROJECT_ROOT is required}")"
RUNTIME_ENV="$(readlink -f "${PAIR_EVAL_RUNTIME_ENV:?PAIR_EVAL_RUNTIME_ENV is required}")"
DATA_ROOT="$(readlink -f "${PAIR_EVAL_DATA_ROOT:?PAIR_EVAL_DATA_ROOT is required}")"
OUTPUT_ROOT="${PAIR_EVAL_OUTPUT_ROOT:?PAIR_EVAL_OUTPUT_ROOT is required}"
GRPO_CHECKPOINT="$(readlink -f "${PAIR_EVAL_GRPO_CHECKPOINT:?PAIR_EVAL_GRPO_CHECKPOINT is required}")"
CLASP_CHECKPOINT="$(readlink -f "${PAIR_EVAL_CLASP_CHECKPOINT:?PAIR_EVAL_CLASP_CHECKPOINT is required}")"
EVAL_START_INDEX="${PAIR_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${PAIR_EVAL_SAMPLES:-100}"
GRPO_LABEL="${PAIR_EVAL_GRPO_LABEL:-grpo_z_step100}"
CLASP_LABEL="${PAIR_EVAL_CLASP_LABEL:-clasp_sequence_a05_step100}"

[[ -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" \
    && -f "${PROJECT_ROOT}/scripts/evaluate_a100.slurm" \
    && -f "${PROJECT_ROOT}/scripts/compare_development.py" ]] || {
    echo "[PairEvalError] immutable source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[PairEvalError] runtime environment or CVRP validation data is missing" >&2
    exit 2
}
for checkpoint in "${GRPO_CHECKPOINT}" "${CLASP_CHECKPOINT}"; do
    [[ -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/adapter_config.json" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[PairEvalError] checkpoint is incomplete: ${checkpoint}" >&2
        exit 2
    }
done
for value in "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[PairEvalError] invalid evaluation integer=${value}" >&2
        exit 2
    }
done
(( EVAL_SAMPLES > 0 )) || { echo "[PairEvalError] sample count must be positive" >&2; exit 2; }
for label in "${GRPO_LABEL}" "${CLASP_LABEL}"; do
    [[ "${label}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
        echo "[PairEvalError] unsafe checkpoint label=${label}" >&2
        exit 2
    }
done
[[ "${GRPO_LABEL}" != "${CLASP_LABEL}" ]] || {
    echo "[PairEvalError] checkpoint labels must be distinct" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[PairEvalError] create-once output exists: ${OUTPUT_ROOT}" >&2
    exit 2
}
mkdir "${OUTPUT_ROOT}"
mkdir "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/comparisons"

run_eval() {
    local label="$1"
    local checkpoint="$2"
    (
        export CLASP_PROJECT_ROOT="${PROJECT_ROOT}"
        export CONDA_ENV="${RUNTIME_ENV}"
        export DATA_ROOT="${DATA_ROOT}"
        export CHECKPOINT_PATH="${checkpoint}"
        export CKPT_LABEL="${label}"
        export TASK=cvrp STAGE=grpo EVALUATION_PURPOSE=development
        export EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp
        export INFERENCE_PARENT_ROOT="${OUTPUT_ROOT}/eval"
        export EVAL_START_INDEX="${EVAL_START_INDEX}"
        export MAX_EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES}"
        export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8 GENERATION_BATCH_SIZE=8
        export TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50 REPETITION_PENALTY=1.0
        export MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
        bash "${PROJECT_ROOT}/scripts/evaluate_a100.slurm"
    )
}

run_eval "${GRPO_LABEL}" "${GRPO_CHECKPOINT}"
run_eval "${CLASP_LABEL}" "${CLASP_CHECKPOINT}"

find_one_details() {
    local label="$1"
    local -a matches=()
    mapfile -t matches < <(find "${OUTPUT_ROOT}/eval/${label}/cvrp" -type f -name 'run_details__*.jsonl' -print | sort)
    [[ "${#matches[@]}" == 1 ]] || {
        echo "[PairEvalError] expected one details artifact for ${label}, found ${#matches[@]}" >&2
        return 2
    }
    printf '%s\n' "${matches[0]}"
}

GRPO_DETAILS="$(find_one_details "${GRPO_LABEL}")"
CLASP_DETAILS="$(find_one_details "${CLASP_LABEL}")"
COMPARISON="${OUTPUT_ROOT}/comparisons/${GRPO_LABEL}_vs_${CLASP_LABEL}.json"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${RUNTIME_ENV}/bin/python" "${PROJECT_ROOT}/scripts/compare_development.py" \
    --left "${GRPO_DETAILS}" --right "${CLASP_DETAILS}" \
    --left-label "${GRPO_LABEL}" --right-label "${CLASP_LABEL}" \
    --output "${COMPARISON}"

"${RUNTIME_ENV}/bin/python" - "${OUTPUT_ROOT}" "${GRPO_DETAILS}" "${CLASP_DETAILS}" \
    "${COMPARISON}" "${GRPO_LABEL}" "${CLASP_LABEL}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output_root, grpo, clasp, comparison, grpo_label, clasp_label = sys.argv[1:]

def identity(path):
    raw = pathlib.Path(path).read_bytes()
    return {
        "path": os.path.realpath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }

payload = {
    "schema_version": 1,
    "status": "complete_development_only",
    "task": "cvrp",
    "evaluation_slice": {
        "split": "eval",
        "start": int(os.environ.get("PAIR_EVAL_START_INDEX", "100")),
        "count": int(os.environ.get("PAIR_EVAL_SAMPLES", "100")),
    },
    "candidate_budget": 8,
    "generation_seed": 271828,
    "details": {grpo_label: identity(grpo), clasp_label: identity(clasp)},
    "comparison": identity(comparison),
}
unsigned = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
payload["index_sha256"] = hashlib.sha256(unsigned).hexdigest()
destination = pathlib.Path(output_root) / "pair_validation_result_index.json"
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "GRPO_CLASP_PAIR_VALIDATION_COMPLETE index=${OUTPUT_ROOT}/pair_validation_result_index.json"
