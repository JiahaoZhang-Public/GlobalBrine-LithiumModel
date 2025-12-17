import csv
import math
import tempfile
import unittest
from pathlib import Path


class TestLocationToLatLon(unittest.TestCase):
    def test_parses_dms_with_backticks_and_slash(self):
        from src.data.location import location_to_lat_lon

        lat, lon = location_to_lat_lon("23°03`S / 67°15`W")
        self.assertTrue(math.isclose(lat or 0.0, -23.05, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(lon or 0.0, -67.25, rel_tol=0.0, abs_tol=1e-6))

    def test_parses_dms_with_seconds_and_quotes(self):
        from src.data.location import location_to_lat_lon

        lat, lon = location_to_lat_lon('31°18’36.89"N, 84°05\'57.66"E')
        self.assertTrue(
            math.isclose(lat or 0.0, 31.3102472222, rel_tol=0.0, abs_tol=1e-6)
        )
        self.assertTrue(math.isclose(lon or 0.0, 84.09935, rel_tol=0.0, abs_tol=1e-6))

    def test_parses_north_latitude_phrase(self):
        from src.data.location import location_to_lat_lon

        text = (
            "Minatomachi, Karatsu, Japan 33°31′29.5″ North Latitude and "
            "129°57′31.6″ East Latitude"
        )
        lat, lon = location_to_lat_lon(text)
        self.assertTrue(
            math.isclose(lat or 0.0, 33.5248611111, rel_tol=0.0, abs_tol=1e-6)
        )
        self.assertTrue(
            math.isclose(lon or 0.0, 129.9587777778, rel_tol=0.0, abs_tol=1e-6)
        )

    def test_falls_back_to_manual_points(self):
        from src.data.location import location_to_lat_lon

        lat, lon = location_to_lat_lon("Israel")
        self.assertEqual((lat, lon), (31.5, 34.8))

    def test_unknown_returns_none(self):
        from src.data.location import location_to_lat_lon

        self.assertEqual(location_to_lat_lon("Unknown Place"), (None, None))
        self.assertEqual(location_to_lat_lon(None), (None, None))
        self.assertEqual(location_to_lat_lon(123), (None, None))


class TestMakeDatasetAddsLatLon(unittest.TestCase):
    def test_process_brines_dataset_with_lat_lon_appends_columns(self):
        from src.data.datasets import CsvDataset, read_csv_rows
        from src.data.make_dataset import process_brines_dataset_with_lat_lon

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "brines.csv"
            out_path = tmp / "processed_brines.csv"

            with in_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Location",
                        "Li_gL",
                        "Mg_gL",
                        "Na_gL",
                        "K_gL",
                        "Ca_gL",
                        "SO4_gL",
                        "Cl_gL",
                        "MLR",
                        "TDS_gL",
                    ]
                )
                writer.writerow(
                    [
                        "23°03`S / 67°15`W",
                        "0.01",
                        "0.02",
                        "0.03",
                        "0.04",
                        "0.05",
                        "0.06",
                        "0.07",
                        "0.08",
                        "0.09",
                    ]
                )

            dataset = CsvDataset(
                in_path,
                required_columns=(
                    "Li_gL",
                    "Mg_gL",
                    "Na_gL",
                    "K_gL",
                    "Ca_gL",
                    "SO4_gL",
                    "Cl_gL",
                    "MLR",
                    "TDS_gL",
                ),
            )
            process_brines_dataset_with_lat_lon(dataset, output_path=out_path)

            rows, header = read_csv_rows(out_path)
            self.assertIn("Latitude", header)
            self.assertIn("Longitude", header)
            self.assertIn("Light_kW_m2", header)
            self.assertEqual(rows[0]["Latitude"], "-23.050000")
            self.assertEqual(rows[0]["Longitude"], "-67.250000")
            self.assertEqual(rows[0]["Light_kW_m2"], "")


if __name__ == "__main__":
    unittest.main()
