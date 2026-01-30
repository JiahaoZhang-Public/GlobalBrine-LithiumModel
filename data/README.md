# Data layout

This repo follows the cookiecutter-data-science `data/` convention.

## What you need to reproduce results

- `data/raw/brines.csv` and `data/raw/experimental.csv`
- (Optional) a Global Solar Atlas GHI GeoTIFF to populate `Light_kW_m2` for brines

This repository may already include example copies of these files. If you replace
them with your own datasets, keep the same filenames and required columns.

## Directories

- `data/raw/`: raw input CSVs (immutable).
- `data/processed/`: validated/normalized CSVs + model-ready arrays (generated).
- `data/predictions/`: inference outputs (generated).

## Required columns

See `src/constants.py` for canonical column lists.

## Generating processed data + features

```bash
python src/data/make_dataset.py data/raw data/processed
python src/features/build_features.py data/processed
```

