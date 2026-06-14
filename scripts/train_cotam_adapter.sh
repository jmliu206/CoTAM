#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m cotam.train_adapter \
  -c configs/cotam_elic_clip_adapter.yaml "$@"
