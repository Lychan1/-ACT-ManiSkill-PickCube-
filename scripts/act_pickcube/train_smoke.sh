#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACT_DIR="${REPO_ROOT}/examples/baselines/act"
ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pos}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
DEMO_PATH="${DEMO_PATH:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning/trajectory.state.${CONTROL_MODE}.${SIM_BACKEND}.h5}"
EXP_NAME="${EXP_NAME:-act-pickcube-state-smoke}"

cd "${ACT_DIR}"
python train.py \
  --env-id "${ENV_ID}" \
  --demo-path "${DEMO_PATH}" \
  --control-mode "${CONTROL_MODE}" \
  --sim-backend "${SIM_BACKEND}" \
  --num-demos "${NUM_DEMOS:-10}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-100}" \
  --total-iters "${TOTAL_ITERS:-1000}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --log-freq "${LOG_FREQ:-100}" \
  --eval-freq "${EVAL_FREQ:-500}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-10}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-1}" \
  --seed "${SEED:-1}" \
  --exp-name "${EXP_NAME}"
