#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACT_DIR="${REPO_ROOT}/examples/baselines/act"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${ACT_DIR}"
"${PYTHON_BIN}" summarize_experiments.py \
  --matrix "${REPO_ROOT}/experiments/act_matrix.json" \
  --runs-root "${ACT_DIR}/runs" \
  --output "${REPO_ROOT}/results/experiment_matrix.csv"
"${PYTHON_BIN}" plot_experiment_matrix.py \
  --input "${REPO_ROOT}/results/experiment_matrix.csv" \
  --output-dir "${REPO_ROOT}/results/curves"
