#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

processed_dir="data/processed"
if [[ $# -ge 1 && "$1" != -* ]]; then
  processed_dir="$1"
  shift
fi

out_path="${MAE_OUT:-models/mae_pretrained.pth}"

python "src/models/train_mae.py" "$processed_dir" \
  --out "$out_path" \
  "$@"
