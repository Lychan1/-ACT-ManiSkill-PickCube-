#!/usr/bin/env bash
set -euo pipefail

ENV_ID="${ENV_ID:-PickCube-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pos}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
NUM_ENVS="${NUM_ENVS:-10}"
RAW_DEMO_PATH="${RAW_DEMO_PATH:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning/trajectory.h5}"

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${RAW_DEMO_PATH}" \
  --use-first-env-state \
  -c "${CONTROL_MODE}" \
  -o rgb \
  --save-traj \
  --num-envs "${NUM_ENVS}" \
  -b "${SIM_BACKEND}"
