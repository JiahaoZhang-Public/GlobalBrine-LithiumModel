#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

processed_dir="data/processed"
if [[ $# -ge 1 && "$1" != -* ]]; then
  processed_dir="$1"
  shift
fi

mae_path="${MAE_CKPT:-models/mae_pretrained.pth}"
out_path="${HEAD_OUT:-models/downstream_head.pth}"

python "src/models/finetune_regression.py" "$processed_dir" \
  --mae "$mae_path" \
  --out "$out_path" \
  --epochs 1000 \
  --batch-size 128 \
  --lr 0.0001 \
  --weight-decay 1e-05 \
  --hidden-dim 256 \
  --n-layers 2 \
  --dropout 0.0 \
  --wandb \
  --wandb-name "regression-finetune" \
  --wandb-mode online \
  "$@"

