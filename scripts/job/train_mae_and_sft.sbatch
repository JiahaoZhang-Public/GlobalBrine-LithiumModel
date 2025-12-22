#!/bin/bash
#SBATCH --job-name=mae_sft
#SBATCH --output=logs/mae_sft_%j.out
#SBATCH --error=logs/mae_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --exclude=gpu-05

set -euo pipefail

ENV_NAME="${ENV_NAME:-global-brine-lithium-model}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs"

source ~/.bashrc
conda activate "${ENV_NAME}" || source "${HOME}/.bashrc"

cd "${PROJECT_ROOT}"

# -----------------------------
# Paths
# -----------------------------
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
RUN_DIR="${RUN_DIR:-models/runs/${SLURM_JOB_ID}}"
MAE_OUT="${MAE_OUT:-${RUN_DIR}/mae_pretrained.pth}"
HEAD_OUT="${HEAD_OUT:-${RUN_DIR}/downstream_head.pth}"

mkdir -p "${RUN_DIR}"

# -----------------------------
# Feature build (required)
# -----------------------------
if [[ ! -f "${PROCESSED_DIR}/X_lake.npy" || ! -f "${PROCESSED_DIR}/X_exp.npy" || ! -f "${PROCESSED_DIR}/y_exp.npy" || ! -f "${PROCESSED_DIR}/feature_scaler.joblib" ]]; then
  echo "Feature artifacts missing; running build_features..."
  python src/features/build_features.py "${PROCESSED_DIR}"
fi

# -----------------------------
# MAE config (best sweep)
# -----------------------------
MAE_EPOCHS="${MAE_EPOCHS:-10}"
MAE_BATCH_SIZE="${MAE_BATCH_SIZE:-128}"
MAE_LR="${MAE_LR:-0.0001}"
MAE_WEIGHT_DECAY="${MAE_WEIGHT_DECAY:-1e-05}"
MAE_MASK_RATIO="${MAE_MASK_RATIO:-0.1}"

MAE_D_MODEL="${MAE_D_MODEL:-256}"
MAE_N_HEADS="${MAE_N_HEADS:-16}"
MAE_N_LAYERS="${MAE_N_LAYERS:-4}"
MAE_MLP_RATIO="${MAE_MLP_RATIO:-2.0}"
MAE_DROPOUT="${MAE_DROPOUT:-0.1}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

echo "Training MAE -> ${MAE_OUT}"
python src/models/train_mae.py "${PROCESSED_DIR}" \
  --out "${MAE_OUT}" \
  --epochs "${MAE_EPOCHS}" \
  --batch-size "${MAE_BATCH_SIZE}" \
  --lr "${MAE_LR}" \
  --weight-decay "${MAE_WEIGHT_DECAY}" \
  --mask-ratio "${MAE_MASK_RATIO}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --d-model "${MAE_D_MODEL}" \
  --n-heads "${MAE_N_HEADS}" \
  --n-layers "${MAE_N_LAYERS}" \
  --mlp-ratio "${MAE_MLP_RATIO}" \
  --dropout "${MAE_DROPOUT}"
  --wandb \
  --wandb-name "mae_pretrain" \
  --wandb-mode online \

# -----------------------------
# SFT (regression head) config
# -----------------------------
SFT_EPOCHS="${SFT_EPOCHS:-200}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_LR="${SFT_LR:-0.001}"
SFT_WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-0.0}"
SFT_HIDDEN_DIM="${SFT_HIDDEN_DIM:-128}"
SFT_N_LAYERS="${SFT_N_LAYERS:-2}"
SFT_DROPOUT="${SFT_DROPOUT:-0.0}"
SFT_FREEZE_ENCODER="${SFT_FREEZE_ENCODER:---freeze-encoder}"

echo "Fine-tuning regression head -> ${HEAD_OUT}"
python src/models/finetune_regression.py "${PROCESSED_DIR}" \
  --mae "${MAE_OUT}" \
  --out "${HEAD_OUT}" \
  --epochs "${SFT_EPOCHS}" \
  --batch-size "${SFT_BATCH_SIZE}" \
  --lr "${SFT_LR}" \
  --weight-decay "${SFT_WEIGHT_DECAY}" \
  --hidden-dim "${SFT_HIDDEN_DIM}" \
  --n-layers "${SFT_N_LAYERS}" \
  --dropout "${SFT_DROPOUT}" \
  ${SFT_FREEZE_ENCODER} \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --wandb \
  --wandb-name "regression_finetune" \
  --wandb-mode online \

echo "Done."
echo "MAE:  ${MAE_OUT}"
echo "HEAD: ${HEAD_OUT}"

