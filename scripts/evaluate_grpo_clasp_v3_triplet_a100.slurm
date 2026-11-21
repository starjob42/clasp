#!/bin/bash -l
# Fresh paired CVRP BoN=8 evaluation of classic GRPO, anchored CLASP v2, and
# norm-matched anchored CLASP v3 on one identical held-out slice.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=cvrp-anchor-v3-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${V3_EVAL_PROJECT_ROOT:?V3_EVAL_PROJECT_ROOT is required}")"
RUNTIME_ENV="$(readlink -f "${V3_EVAL_RUNTIME_ENV:?V3_EVAL_RUNTIME_ENV is required}")"
DATA_ROOT="$(readlink -f "${V3_EVAL_DATA_ROOT:?V3_EVAL_DATA_ROOT is required}")"
OUTPUT_ROOT="${V3_EVAL_OUTPUT_ROOT:?V3_EVAL_OUTPUT_ROOT is required}"
GRPO_CHECKPOINT="$(readlink -f "${V3_EVAL_GRPO_CHECKPOINT:?V3_EVAL_GRPO_CHECKPOINT is required}")"
V2_CHECKPOINT="$(readlink -f "${V3_EVAL_V2_CHECKPOINT:?V3_EVAL_V2_CHECKPOINT is required}")"
V3_CHECKPOINT="$(readlink -f "${V3_EVAL_V3_CHECKPOINT:?V3_EVAL_V3_CHECKPOINT is required}")"
EVAL_START_INDEX="${V3_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${V3_EVAL_SAMPLES:-100}"

[[ -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" \
    && -f "${PROJECT_ROOT}/scripts/evaluate_a100.slurm" \
    && -f "${PROJECT_ROOT}/scripts/compare_development.py" ]] || {
    echo "[V3TripletEvalError] immutable source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[V3TripletEvalError] runtime environment or CVRP evaluation data is missing" >&2
    exit 2
}
for checkpoint in "${GRPO_CHECKPOINT}" "${V2_CHECKPOINT}" "${V3_CHECKPOINT}"; do
    [[ -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/adapter_config.json" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[V3TripletEvalError] checkpoint is incomplete: ${checkpoint}" >&2
        exit 2
    }
done
for value in "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[V3TripletEvalError] invalid evaluation integer=${value}" >&2
        exit 2
    }
done
(( EVAL_SAMPLES > 0 )) || {
    echo "[V3TripletEvalError] sample count must be positive" >&2
    exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
    echo "[V3TripletEvalError] create-once output exists: ${OUTPUT_ROOT}" >&2
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
        export SEED=271828 DTYPE=float16 NUM_RETURN_SEQUENCES=8
        export GENERATION_BATCH_SIZE=8 TEMPERATURE=0.7 TOP_P=0.9 TOP_K=50
        export REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=3000 MAX_PROMPT_LENGTH=29768
        bash "${PROJECT_ROOT}/scripts/evaluate_a100.slurm"
    )
}

run_eval grpo_z_step200 "${GRPO_CHECKPOINT}"
run_eval clasp_grpo_anchor_v2_step200 "${V2_CHECKPOINT}"
run_eval clasp_grpo_anchor_v3_step200 "${V3_CHECKPOINT}"

find_one_details() {
    local label="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${OUTPUT_ROOT}/eval/${label}/cvrp" -type f \
            -name 'run_details__*.jsonl' -print | sort
    )
    [[ "${#matches[@]}" == 1 ]] || {
        echo "[V3TripletEvalError] expected one details artifact for ${label}, found ${#matches[@]}" >&2
        return 2
    }
    printf '%s\n' "${matches[0]}"
}

GRPO_DETAILS="$(find_one_details grpo_z_step200)"
V2_DETAILS="$(find_one_details clasp_grpo_anchor_v2_step200)"
V3_DETAILS="$(find_one_details clasp_grpo_anchor_v3_step200)"

compare_pair() {
    local left="$1"
    local right="$2"
    local left_label="$3"
    local right_label="$4"
    local output="$5"
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${RUNTIME_ENV}/bin/python" "${PROJECT_ROOT}/scripts/compare_development.py" \
        --left "${left}" --right "${right}" \
        --left-label "${left_label}" --right-label "${right_label}" \
        --output "${output}"
}

compare_pair "${GRPO_DETAILS}" "${V2_DETAILS}" grpo_z_step200 \
    clasp_grpo_anchor_v2_step200 \
    "${OUTPUT_ROOT}/comparisons/grpo_z_step200_vs_clasp_grpo_anchor_v2_step200.json"
compare_pair "${GRPO_DETAILS}" "${V3_DETAILS}" grpo_z_step200 \
    clasp_grpo_anchor_v3_step200 \
    "${OUTPUT_ROOT}/comparisons/grpo_z_step200_vs_clasp_grpo_anchor_v3_step200.json"
compare_pair "${V2_DETAILS}" "${V3_DETAILS}" clasp_grpo_anchor_v2_step200 \
    clasp_grpo_anchor_v3_step200 \
    "${OUTPUT_ROOT}/comparisons/clasp_grpo_anchor_v2_step200_vs_clasp_grpo_anchor_v3_step200.json"

"${RUNTIME_ENV}/bin/python" - "${OUTPUT_ROOT}" "${GRPO_DETAILS}" \
    "${V2_DETAILS}" "${V3_DETAILS}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output_root, grpo, v2, v3 = sys.argv[1:]

def identity(path):
    raw = pathlib.Path(path).read_bytes()
    return {
        "path": os.path.realpath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }

comparison_paths = sorted((pathlib.Path(output_root) / "comparisons").glob("*.json"))
payload = {
    "schema_version": 1,
    "status": "complete_development_only",
    "task": "cvrp",
    "evaluation_slice": {
        "split": "eval",
        "start": int(os.environ.get("V3_EVAL_START_INDEX", "100")),
        "count": int(os.environ.get("V3_EVAL_SAMPLES", "100")),
    },
    "candidate_budget": 8,
    "generation_seed": 271828,
    "details": {
        "grpo_z_step200": identity(grpo),
        "clasp_grpo_anchor_v2_step200": identity(v2),
        "clasp_grpo_anchor_v3_step200": identity(v3),
    },
    "comparisons": [identity(str(path)) for path in comparison_paths],
}
unsigned = json.dumps(
    payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
).encode()
payload["index_sha256"] = hashlib.sha256(unsigned).hexdigest()
destination = pathlib.Path(output_root) / "triplet_validation_result_index.json"
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "GRPO_CLASP_V3_TRIPLET_COMPLETE index=${OUTPUT_ROOT}/triplet_validation_result_index.json"
