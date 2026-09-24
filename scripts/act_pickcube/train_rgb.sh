#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACT_DIR="${REPO_ROOT}/examples/baselines/act"
ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pos}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
DEMO_PATH="${DEMO_PATH:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning/trajectory.rgb.${CONTROL_MODE}.${SIM_BACKEND}.h5}"
NUM_DEMOS="${NUM_DEMOS:-100}"
SEED="${SEED:-1}"
EXP_NAME="${EXP_NAME:-act-${ENV_ID}-rgb-${NUM_DEMOS}demos-seed${SEED}}"

cd "${ACT_DIR}"
python train_rgbd.py \
  --env-id "${ENV_ID}" \
  --demo-path "${DEMO_PATH}" \
  --control-mode "${CONTROL_MODE}" \
  --sim-backend "${SIM_BACKEND}" \
  --no-include-depth \
  --no-capture-video \
  --num-demos "${NUM_DEMOS}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-100}" \
  --total-iters "${TOTAL_ITERS:-30001}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --log-freq "${LOG_FREQ:-100}" \
  --eval-freq "${EVAL_FREQ:-5000}" \
  --save-freq "${SAVE_FREQ:-5000}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-100}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-4}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}"
