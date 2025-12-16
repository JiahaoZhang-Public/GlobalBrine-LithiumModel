#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

raw_dir="$tmp_dir/data/raw"
processed_dir="$tmp_dir/data/processed"
models_dir="$tmp_dir/models"

mkdir -p "$raw_dir" "$processed_dir" "$models_dir"

cat >"$raw_dir/brines.csv" <<'CSV'
Li_gL,Mg_gL,Na_gL,K_gL,Ca_gL,SO4_gL,Cl_gL,MLR,TDS_gL
1.0,2.0,3.0,4.0,5.0,6.0,,8.0,9.0
2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0
3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0
4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0
5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0
CSV

cat >"$raw_dir/experimental.csv" <<'CSV'
TDS_gL,MLR,Light_kW_m2,Selectivity,Li_Crystallization_mg_m2_h,Evap_kg_m2_h
9.0,8.0,100.0,0.10,1.0,2.0
10.0,9.0,200.0,0.20,1.5,3.0
11.0,10.0,150.0,0.15,1.2,2.5
CSV

python src/data/make_dataset.py "$raw_dir" "$processed_dir" >/dev/null
python src/features/build_features.py "$processed_dir" >/dev/null

python src/models/train_mae.py "$processed_dir" \
  --out "$models_dir/mae_pretrained.pth" \
  --epochs 1 \
  --batch-size 4 \
  --device cpu \
  --d-model 32 \
  --n-heads 4 \
  --n-layers 1 \
  --mlp-ratio 2.0 \
  >/dev/null

python src/models/finetune_regression.py "$processed_dir" \
  --mae "$models_dir/mae_pretrained.pth" \
  --out "$models_dir/downstream_head.pth" \
  --epochs 1 \
  --batch-size 3 \
  --device cpu \
  --hidden-dim 16 \
  --n-layers 1 \
  >/dev/null

python - <<PY
from pathlib import Path
import torch

mae = torch.load(Path("$models_dir") / "mae_pretrained.pth", map_location="cpu")
head = torch.load(Path("$models_dir") / "downstream_head.pth", map_location="cpu")

assert "model_state_dict" in mae and "mae_config" in mae
assert "head_state_dict" in head and "head_config" in head
print("smoke_train: OK")
PY

