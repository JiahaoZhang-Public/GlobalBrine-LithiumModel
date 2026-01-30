# `data/raw`

Raw datasets used by the pipeline.

## Expected files

- `brines.csv` (unlabeled brine chemistry; see `src/constants.py:BRINE_CHEMISTRY_COLUMNS`)
- `experimental.csv` (labeled experiments; see `src/constants.py:EXPERIMENTAL_*_COLUMNS`)

## Notes

- The pipeline treats blank cells as missing values (`NaN`).
- `src/data/make_dataset.py` validates required columns and writes processed CSVs.

