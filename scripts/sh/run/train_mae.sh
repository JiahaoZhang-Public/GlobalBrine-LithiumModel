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
  --epochs 10000 \
  --batch-size 128 \
  --lr 0.0001 \
  --weight-decay 1e-05 \
  --mask-ratio 0.1 \
  --d-model 256 \
  --n-heads 16 \
  --n-layers 4 \
  --mlp-ratio 2.0 \
  --dropout 0.1 \
  --wandb \
  --wandb-name "mae-pretrain" \
  --wandb-mode online \
  "$@"
