#!/bin/bash -l
# CPU-only semantic overlap audit for the independent development denominator.
#SBATCH --account=vas@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --job-name=clasp-fresh-audit
set -euo pipefail
umask 027
BENCHMARK_ROOT="${FAIR_BENCHMARK_ROOT:?FAIR_BENCHMARK_ROOT is required}"
PYTHON_BIN="${FAIR_AUDIT_PYTHON:?FAIR_AUDIT_PYTHON is required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
"${PYTHON_BIN}" "${BENCHMARK_ROOT}/audit_tool/scripts/build_fair_development_benchmark.py" \
    --audit-only --output-root "${BENCHMARK_ROOT}/data" \
    --forbidden-sources "${BENCHMARK_ROOT}/data/forbidden_sources.jean_zay.json"
