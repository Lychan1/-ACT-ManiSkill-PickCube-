#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACT_DIR="${REPO_ROOT}/examples/baselines/act"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ACT_DIR}"
"${PYTHON_BIN}" run_experiment_matrix.py \
  --matrix "${REPO_ROOT}/experiments/act_matrix.json" \
  --repo-root "${REPO_ROOT}" \
  "$@"
