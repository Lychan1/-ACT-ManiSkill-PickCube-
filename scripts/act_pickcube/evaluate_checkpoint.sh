#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACT_DIR="${REPO_ROOT}/examples/baselines/act"
DEFAULT_CHECKPOINT="${ACT_DIR}/runs/act-PickCube-v1-state-100demos/checkpoints/best_eval_success_at_end.pt"
CHECKPOINT="${1:-${CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"

CUDA_FLAG="--cuda"
if [[ "${CUDA:-1}" == "0" ]]; then
  CUDA_FLAG="--no-cuda"
fi

cd "${ACT_DIR}"
python evaluate_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --env-id "${ENV_ID:-PickCube-v1}" \
  --control-mode "${CONTROL_MODE:-pd_ee_delta_pos}" \
  --sim-backend "${SIM_BACKEND:-physx_cpu}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-100}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-100}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-1}" \
  --seed "${SEED:-1}" \
  "${CUDA_FLAG}" \
  "$@"
