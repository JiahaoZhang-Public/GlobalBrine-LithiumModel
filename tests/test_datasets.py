import csv
import tempfile
import unittest
from pathlib import Path

from src.data.datasets import CsvDataset, DatasetError, process_csv_dataset


class TestDatasets(unittest.TestCase):
    def test_process_csv_dataset_reorders_and_preserves_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "raw.csv"
            out_path = tmp / "processed.csv"

            with in_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["b", "a", "extra"])
                writer.writerow(["2", "1", "x"])

            dataset = CsvDataset(in_path, required_columns=("a", "b"))
            process_csv_dataset(dataset, output_path=out_path, keep_extra_columns=True)

            with out_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                row = next(reader)

            self.assertEqual(header, ["a", "b", "extra"])
            self.assertEqual(row, ["1", "2", "x"])

    def test_process_csv_dataset_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "raw.csv"
            out_path = tmp / "processed.csv"

            with in_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["a"])
                writer.writerow(["1"])

            dataset = CsvDataset(in_path, required_columns=("a", "b"))
            with self.assertRaises(DatasetError):
                process_csv_dataset(dataset, output_path=out_path)


class TestSanitizeTds(unittest.TestCase):
    def test_unit_error_divided_by_1000(self):
        from src.data.make_dataset import _sanitize_tds

        rows = [
            {
                "Li_gL": "0.05",
                "Mg_gL": "1.0",
                "Na_gL": "3.0",
                "K_gL": "0.5",
                "Ca_gL": "0.3",
                "SO4_gL": "0.1",
                "Cl_gL": "6.0",
                "TDS_gL": "11000",
            }
        ]
        # ion_sum ~= 10.95, TDS/ion_sum ~= 1004 -> unit error
        fixed = _sanitize_tds(rows)
        self.assertEqual(fixed, 1)
        self.assertAlmostEqual(float(rows[0]["TDS_gL"]), 11.0, places=1)

    def test_normal_tds_unchanged(self):
        from src.data.make_dataset import _sanitize_tds

        rows = [
            {
                "Li_gL": "0.05",
                "Mg_gL": "30.0",
                "Na_gL": "20.0",
                "K_gL": "7.0",
                "Ca_gL": "12.0",
                "SO4_gL": "0.6",
                "Cl_gL": "150.0",
                "TDS_gL": "225.0",
            }
        ]
        fixed = _sanitize_tds(rows)
        self.assertEqual(fixed, 0)
        self.assertAlmostEqual(float(rows[0]["TDS_gL"]), 225.0, places=1)

    def test_hard_ceiling_applied(self):
        from src.data.make_dataset import _sanitize_tds

        # TDS already in g/L but above physical max (no ion data to trigger ratio check)
        rows = [{"TDS_gL": "500.0"}]
        fixed = _sanitize_tds(rows)
        self.assertEqual(fixed, 1)
        self.assertAlmostEqual(float(rows[0]["TDS_gL"]), 450.0, places=1)

    def test_missing_tds_skipped(self):
        from src.data.make_dataset import _sanitize_tds

        rows = [{"Li_gL": "0.05", "TDS_gL": ""}]
        fixed = _sanitize_tds(rows)
        self.assertEqual(fixed, 0)
        self.assertEqual(rows[0]["TDS_gL"], "")


if __name__ == "__main__":
    unittest.main()
