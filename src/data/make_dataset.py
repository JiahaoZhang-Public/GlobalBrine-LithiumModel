"""
This script is used to create the processed datasets from the raw datasets.
"""

# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

try:
    from src.constants import BRINES_DATASET, EXPERIMENTAL_DATASET
    from src.data.datasets import CsvDataset, process_csv_dataset
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import BRINES_DATASET, EXPERIMENTAL_DATASET
    from src.data.datasets import CsvDataset, process_csv_dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Create processed CSVs from raw datasets.

    This project currently assumes raw CSVs are already clean; this step:
    - validates required columns exist
    - enforces a consistent column order
    - writes datasets to the processed directory
    """
    logger = logging.getLogger(__name__)

    raw_dir = Path(input_filepath)
    processed_dir = Path(output_filepath)

    brines_in = raw_dir / BRINES_DATASET.filename
    experimental_in = raw_dir / EXPERIMENTAL_DATASET.filename

    logger.info("processing %s", brines_in)
    process_csv_dataset(
        CsvDataset(brines_in, BRINES_DATASET.required_columns),
        output_path=processed_dir / BRINES_DATASET.filename,
        keep_extra_columns=True,
    )

    logger.info("processing %s", experimental_in)
    process_csv_dataset(
        CsvDataset(experimental_in, EXPERIMENTAL_DATASET.required_columns),
        output_path=processed_dir / EXPERIMENTAL_DATASET.filename,
        keep_extra_columns=True,
    )
    logger.info("wrote processed datasets to %s", processed_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    _project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
