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


if __name__ == "__main__":
    unittest.main()
