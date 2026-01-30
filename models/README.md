# Model artifacts

This directory stores trained checkpoints.

## Main checkpoints

- `models/mae_pretrained.pth`: MAE encoder pretraining output (see `src/models/train_mae.py`)
- `models/downstream_head.pth`: regression head fine-tuned on experimental labels
  (see `src/models/finetune_regression.py`)

## Optional sweep outputs

- `models/mae_sweep/`: MAE sweep outputs (ignored by default in `.gitignore`).

