#!/usr/bin/env bash
set -euo pipefail

ENV_ID="${ENV_ID:-PickCube-v1}"

python -m mani_skill.utils.download_demo "${ENV_ID}"
