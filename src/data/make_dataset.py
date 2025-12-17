"""
This script is used to create the processed datasets from the raw datasets.
"""

# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    find_dotenv = None

    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False


try:
    from src.constants import BRINES_DATASET, EXPERIMENTAL_DATASET
    from src.data.datasets import (
        CsvDataset,
        normalize_rows_to_columns,
        ordered_columns,
        process_csv_dataset,
        read_csv_rows,
        validate_required_columns,
        write_csv_rows,
    )
    from src.data.light import GeoTiffSampler
    from src.data.location import location_to_lat_lon
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import BRINES_DATASET, EXPERIMENTAL_DATASET
    from src.data.datasets import (
        CsvDataset,
        normalize_rows_to_columns,
        ordered_columns,
        process_csv_dataset,
        read_csv_rows,
        validate_required_columns,
        write_csv_rows,
    )
    from src.data.light import GeoTiffSampler
    from src.data.location import location_to_lat_lon


def process_brines_dataset_with_lat_lon(
    dataset: CsvDataset,
    *,
    output_path: Path,
    keep_extra_columns: bool = True,
    light_geotiff: Path | None = None,
    light_band: int = 1,
    light_scale: float = 1.0,
) -> Path:
    """Process brines CSV and append Latitude/Longitude/Light_kW_m2.

    - Latitude/Longitude are derived from the Location free-text column when missing.
    - Light_kW_m2 is sampled from a GeoTIFF when provided (WGS84 lat/lon inputs).
    """

    def _try_float(value: object) -> float | None:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        text = value.strip()
        if text == "" or text.lower() == "nan":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    rows, header = read_csv_rows(dataset.path)
    validate_required_columns(
        header,
        dataset.required_columns,
        dataset_name=dataset.path.name,
    )
    columns = ordered_columns(
        header,
        dataset.required_columns,
        keep_extra=keep_extra_columns,
    )

    sampler: GeoTiffSampler | None = None
    if light_geotiff is not None:
        sampler = GeoTiffSampler(light_geotiff, band=light_band)

    try:
        for row in rows:
            lat = _try_float(row.get("Latitude", ""))
            lon = _try_float(row.get("Longitude", ""))

            if lat is None or lon is None:
                parsed_lat, parsed_lon = location_to_lat_lon(row.get("Location", ""))
                lat = lat if lat is not None else parsed_lat
                lon = lon if lon is not None else parsed_lon

            if lat is not None:
                row["Latitude"] = f"{lat:.6f}"
            if lon is not None:
                row["Longitude"] = f"{lon:.6f}"

            existing_light = (row.get("Light_kW_m2") or "").strip()
            if (
                sampler is not None
                and existing_light == ""
                and lat is not None
                and lon is not None
            ):
                value = sampler.sample_lat_lon(lat, lon)
                if value is not None:
                    row["Light_kW_m2"] = f"{(value * float(light_scale)):.6f}"
    finally:
        if sampler is not None:
            sampler.close()

    if "Latitude" not in columns:
        columns.append("Latitude")
    if "Longitude" not in columns:
        columns.append("Longitude")
    if "Light_kW_m2" not in columns:
        columns.append("Light_kW_m2")

    normalized = normalize_rows_to_columns(rows, columns)
    write_csv_rows(output_path, normalized, columns)
    return output_path


@click.command()
@click.option(
    "--light-geotiff",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional GeoTIFF to sample into Light_kW_m2 using Latitude/Longitude.",
)
@click.option(
    "--light-band",
    type=int,
    default=1,
    show_default=True,
    help="1-based band index to sample from the GeoTIFF.",
)
@click.option(
    "--light-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Multiply sampled GeoTIFF values by this scale before writing Light_kW_m2.",
)
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(light_geotiff, light_band, light_scale, input_filepath, output_filepath):
    """Create processed CSVs from raw datasets.

    This project currently assumes raw CSVs are already clean; this step:
    - validates required columns exist
    - enforces a consistent column order
    - appends Latitude/Longitude for brines (parsed from Location)
    - optionally samples Light_kW_m2 from a GeoTIFF for brines
    - writes datasets to the processed directory
    """
    logger = logging.getLogger(__name__)

    raw_dir = Path(input_filepath)
    processed_dir = Path(output_filepath)

    brines_in = raw_dir / BRINES_DATASET.filename
    experimental_in = raw_dir / EXPERIMENTAL_DATASET.filename

    logger.info("processing %s", brines_in)
    process_brines_dataset_with_lat_lon(
        CsvDataset(brines_in, BRINES_DATASET.required_columns),
        output_path=processed_dir / BRINES_DATASET.filename,
        keep_extra_columns=True,
        light_geotiff=None if light_geotiff is None else Path(light_geotiff),
        light_band=int(light_band),
        light_scale=float(light_scale),
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
    if find_dotenv is not None:
        load_dotenv(find_dotenv())

    main()
