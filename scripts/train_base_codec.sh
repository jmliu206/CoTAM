#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m cotam.train_base_codec \
  -c configs/base_codec_elic.yaml "$@"
