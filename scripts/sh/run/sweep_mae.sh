#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

processed_dir="data/processed"
if [[ $# -ge 1 && "$1" != -* ]]; then
  processed_dir="$1"
  shift
fi

out_dir="${MAE_SWEEP_OUT_DIR:-models/mae_sweep}"

python "src/models/sweep_mae.py" "$processed_dir" \
  --out-dir "$out_dir" \
  --epochs 10 \
  --save-best-only \
  "$@"

