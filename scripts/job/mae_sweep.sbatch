#!/bin/bash
#SBATCH --job-name=mae_sweep
#SBATCH --output=logs/mae_sweep_%j.out
#SBATCH --error=logs/mae_sweep_%j.err
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

PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
OUT_DIR="${OUT_DIR:-models/mae_sweep/${SLURM_JOB_ID}}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

extra_args=()
if [[ -n "${SWEEP_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${SWEEP_ARGS})
fi

python src/models/sweep_mae.py "${PROCESSED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device auto \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  "${extra_args[@]}"

echo "Done. Results in: ${OUT_DIR}"

