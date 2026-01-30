# `data/processed`

Generated artifacts used for training and inference.

Typical files after running `python src/data/make_dataset.py ...` and
`python src/features/build_features.py ...`:

- `brines.csv`, `experimental.csv` (validated + consistent schema)
- `X_lake.npy` (MAE pretraining inputs)
- `X_exp.npy`, `y_exp.npy` (fine-tuning inputs/targets)
- `feature_scaler.joblib` (scaler stats for inference)

