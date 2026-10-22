#!/bin/bash -l
# Evaluate the common SFT initialization and both terminal showdown checkpoints
# on one fresh, paired CVRP best-of-8 development slice.
#
# Required: SHOWDOWN_PROJECT_ROOT, SHOWDOWN_RUNTIME_ENV, SHOWDOWN_DATA_ROOT,
# SHOWDOWN_OUTPUT_ROOT, SHOWDOWN_SFT_CHECKPOINT, SHOWDOWN_GRPO_CHECKPOINT,
# SHOWDOWN_CLASP_CHECKPOINT.

#SBATCH --account=vas@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --job-name=cvrp-showdown-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 027

PROJECT_ROOT="$(readlink -f "${SHOWDOWN_PROJECT_ROOT:?SHOWDOWN_PROJECT_ROOT is required}")"
RUNTIME_ENV="$(readlink -f "${SHOWDOWN_RUNTIME_ENV:?SHOWDOWN_RUNTIME_ENV is required}")"
DATA_ROOT="$(readlink -f "${SHOWDOWN_DATA_ROOT:?SHOWDOWN_DATA_ROOT is required}")"
OUTPUT_ROOT="${SHOWDOWN_OUTPUT_ROOT:?SHOWDOWN_OUTPUT_ROOT is required}"
SFT_CHECKPOINT="$(readlink -f "${SHOWDOWN_SFT_CHECKPOINT:?SHOWDOWN_SFT_CHECKPOINT is required}")"
GRPO_CHECKPOINT="$(readlink -f "${SHOWDOWN_GRPO_CHECKPOINT:?SHOWDOWN_GRPO_CHECKPOINT is required}")"
CLASP_CHECKPOINT="$(readlink -f "${SHOWDOWN_CLASP_CHECKPOINT:?SHOWDOWN_CLASP_CHECKPOINT is required}")"
EVAL_START_INDEX="${SHOWDOWN_EVAL_START_INDEX:-100}"
EVAL_SAMPLES="${SHOWDOWN_EVAL_SAMPLES:-100}"

[[ -f "${PROJECT_ROOT}/SOURCE_MANIFEST.sha256" \
    && -f "${PROJECT_ROOT}/scripts/evaluate_a100.slurm" \
    && -f "${PROJECT_ROOT}/scripts/compare_development.py" ]] || {
    echo "[ShowdownEvalError] frozen source snapshot is incomplete" >&2
    exit 2
}
[[ -x "${RUNTIME_ENV}/bin/python" && -d "${DATA_ROOT}/cvrp/eval" ]] || {
    echo "[ShowdownEvalError] runtime environment or CVRP evaluation data is missing" >&2
    exit 2
}
for checkpoint in "${SFT_CHECKPOINT}" "${GRPO_CHECKPOINT}" "${CLASP_CHECKPOINT}"; do
    [[ -f "${checkpoint}/adapter_model.safetensors" \
        && -f "${checkpoint}/adapter_config.json" \
        && -f "${checkpoint}/checkpoint_fidelity.json" ]] || {
        echo "[ShowdownEvalError] checkpoint is incomplete: ${checkpoint}" >&2
        exit 2
    }
done
for value in "${EVAL_START_INDEX}" "${EVAL_SAMPLES}"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "[ShowdownEvalError] invalid evaluation integer=${value}" >&2
        exit 2
    }
done
(( EVAL_SAMPLES > 0 )) || { echo "[ShowdownEvalError] sample count must be positive" >&2; exit 2; }

EVAL_ROOT="${OUTPUT_ROOT}/eval"
COMPARISON_ROOT="${OUTPUT_ROOT}/comparisons"
[[ ! -e "${EVAL_ROOT}" && ! -e "${COMPARISON_ROOT}" ]] || {
    echo "[ShowdownEvalError] create-once evaluation output already exists" >&2
    exit 2
}
mkdir "${EVAL_ROOT}" "${COMPARISON_ROOT}"

run_eval() {
    local label="$1"
    local stage="$2"
    local checkpoint="$3"
    (
        export CLASP_PROJECT_ROOT="${PROJECT_ROOT}"
        export CONDA_ENV="${RUNTIME_ENV}"
        export DATA_ROOT="${DATA_ROOT}"
        export CHECKPOINT_PATH="${checkpoint}"
        export CKPT_LABEL="${label}"
        export TASK=cvrp
        export STAGE="${stage}"
        export EVALUATION_PURPOSE=development
        export EVALUATION_REPRESENTATION_PROFILE=jiang_f638_clasp
        export INFERENCE_PARENT_ROOT="${EVAL_ROOT}"
        export EVAL_START_INDEX="${EVAL_START_INDEX}"
        export MAX_EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES}"
        export SEED=271828
        export DTYPE=float16
        export NUM_RETURN_SEQUENCES=8
        export GENERATION_BATCH_SIZE=8
        export TEMPERATURE=0.7
        export TOP_P=0.9
        export TOP_K=50
        export REPETITION_PENALTY=1.0
        export MAX_NEW_TOKENS=3000
        export MAX_PROMPT_LENGTH=29768
        bash "${PROJECT_ROOT}/scripts/evaluate_a100.slurm"
    )
}

run_eval sft_final_step31250 sft "${SFT_CHECKPOINT}"
run_eval grpo_z_step100 grpo "${GRPO_CHECKPOINT}"
run_eval clasp_sequence_a05_step100 grpo "${CLASP_CHECKPOINT}"

find_one_details() {
    local label="$1"
    local -a matches=()
    mapfile -t matches < <(find "${EVAL_ROOT}/${label}/cvrp" -type f -name 'run_details__*.jsonl' -print | sort)
    [[ "${#matches[@]}" == 1 ]] || {
        echo "[ShowdownEvalError] expected one details artifact for ${label}, found ${#matches[@]}" >&2
        return 2
    }
    printf '%s\n' "${matches[0]}"
}

SFT_DETAILS="$(find_one_details sft_final_step31250)"
GRPO_DETAILS="$(find_one_details grpo_z_step100)"
CLASP_DETAILS="$(find_one_details clasp_sequence_a05_step100)"

compare_pair() {
    local left="$1"
    local right="$2"
    local left_label="$3"
    local right_label="$4"
    local output="$5"
    "${RUNTIME_ENV}/bin/python" "${PROJECT_ROOT}/scripts/compare_development.py" \
        --left "${left}" --right "${right}" \
        --left-label "${left_label}" --right-label "${right_label}" \
        --output "${output}"
}

compare_pair "${SFT_DETAILS}" "${GRPO_DETAILS}" sft_final_step31250 grpo_z_step100 \
    "${COMPARISON_ROOT}/sft_vs_grpo_z.json"
compare_pair "${SFT_DETAILS}" "${CLASP_DETAILS}" sft_final_step31250 clasp_sequence_a05_step100 \
    "${COMPARISON_ROOT}/sft_vs_clasp_sequence_a05.json"
compare_pair "${GRPO_DETAILS}" "${CLASP_DETAILS}" grpo_z_step100 clasp_sequence_a05_step100 \
    "${COMPARISON_ROOT}/grpo_z_vs_clasp_sequence_a05.json"

"${RUNTIME_ENV}/bin/python" - "${OUTPUT_ROOT}" "${SFT_DETAILS}" "${GRPO_DETAILS}" \
    "${CLASP_DETAILS}" "${COMPARISON_ROOT}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output_root, sft, grpo, clasp, comparisons = sys.argv[1:]

def identity(path):
    raw = pathlib.Path(path).read_bytes()
    return {"path": os.path.realpath(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}

comparison_paths = sorted(pathlib.Path(comparisons).glob("*.json"))
payload = {
    "schema_version": 1,
    "status": "complete_development_only",
    "task": "cvrp",
    "evaluation_slice": {"split": "eval", "start": int(os.environ.get("SHOWDOWN_EVAL_START_INDEX", "100")), "count": int(os.environ.get("SHOWDOWN_EVAL_SAMPLES", "100"))},
    "candidate_budget": 8,
    "generation_seed": 271828,
    "details": {"sft": identity(sft), "grpo_z": identity(grpo), "clasp_sequence_a05": identity(clasp)},
    "comparisons": [identity(str(path)) for path in comparison_paths],
}
unsigned = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
payload["index_sha256"] = hashlib.sha256(unsigned).hexdigest()
destination = pathlib.Path(output_root) / "showdown_result_index.json"
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "GRPO_CLASP_SHOWDOWN_EVAL_COMPLETE index=${OUTPUT_ROOT}/showdown_result_index.json"
