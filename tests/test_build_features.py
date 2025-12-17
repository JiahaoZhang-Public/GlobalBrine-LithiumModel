import csv
import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy is required for feature-building tests")
class TestBuildFeatures(unittest.TestCase):
    def _write_csv(self, path: Path, header: list[str], rows: list[list[object]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def _load_scaler(self, path: Path):
        try:
            import joblib  # type: ignore
        except ModuleNotFoundError:
            with path.open("rb") as handle:
                return pickle.load(handle)
        else:
            return joblib.load(path)

    def test_build_and_save_features_outputs_expected_shapes(self):
        from src.constants import (
            BRINE_FEATURE_COLUMNS,
            BRINES_DATASET,
            EXPERIMENTAL_DATASET,
            EXPERIMENTAL_FEATURE_COLUMNS,
            EXPERIMENTAL_TARGET_COLUMNS,
        )
        from src.features.build_features import build_and_save_features

        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"

            self._write_csv(
                processed / BRINES_DATASET.filename,
                header=list(BRINE_FEATURE_COLUMNS),
                rows=[
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 200],
                    [3, 4, 5, 6, 7, 8, 9, 10, 11, 300],
                ],
            )

            exp_header = list(EXPERIMENTAL_FEATURE_COLUMNS) + list(
                EXPERIMENTAL_TARGET_COLUMNS
            )
            self._write_csv(
                processed / EXPERIMENTAL_DATASET.filename,
                header=exp_header,
                rows=[
                    [9, 8, 100, 0.1, 1.0, 2.0],
                    [10, 9, 200, 0.2, 1.5, 3.0],
                ],
            )

            out = build_and_save_features(processed)
            x_lake = np.load(out["X_lake"])
            x_exp = np.load(out["X_exp"])
            y_exp = np.load(out["y_exp"])

            self.assertEqual(x_lake.shape, (3, len(BRINE_FEATURE_COLUMNS)))
            self.assertEqual(x_exp.shape, (2, len(EXPERIMENTAL_FEATURE_COLUMNS)))
            self.assertEqual(y_exp.shape, (2, len(EXPERIMENTAL_TARGET_COLUMNS)))

            scaler = self._load_scaler(out["feature_scaler"])
            self.assertIn("brine_features", scaler)
            self.assertIn("experimental_features", scaler)
            self.assertIn("experimental_targets", scaler)

            # Experimental scaling: TDS/MLR/Light use brines-derived scaling.
            self.assertTrue(
                np.allclose(
                    x_exp[0],
                    [-1.2247449, -1.2247449, -1.2247449],
                    atol=1e-5,
                )
            )

            # Targets are standardized in y_exp.npy.
            self.assertTrue(np.allclose(y_exp[0], [-1.0, -1.0, -1.0], atol=1e-6))

    def test_missing_feature_value_is_preserved_by_default(self):
        from src.constants import BRINE_FEATURE_COLUMNS, BRINES_DATASET
        from src.features.build_features import build_and_save_features

        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            header = list(BRINE_FEATURE_COLUMNS)

            # First row has missing Cl_gL (blank string).
            row1 = [1, 2, 3, 4, 5, 6, "", 8, 9, 100]
            row2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 200]

            self._write_csv(
                processed / BRINES_DATASET.filename,
                header=header,
                rows=[row1, row2],
            )

            # Minimal experimental.csv needed for the function to run.
            self._write_csv(
                processed / "experimental.csv",
                header=[
                    "TDS_gL",
                    "MLR",
                    "Light_kW_m2",
                    "Selectivity",
                    "Li_Crystallization_mg_m2_h",
                    "Evap_kg_m2_h",
                ],
                rows=[[9, 8, 100, 0.1, 1.0, 2.0]],
            )

            out = build_and_save_features(processed)
            x_lake = np.load(out["X_lake"])
            cl_idx = list(BRINE_FEATURE_COLUMNS).index("Cl_gL")
            self.assertTrue(np.isnan(x_lake[0, cl_idx]))
            self.assertTrue(np.isfinite(x_lake[~np.isnan(x_lake)]).all())

    def test_light_is_required_for_brines_features(self):
        from src.constants import BRINE_FEATURE_COLUMNS, BRINES_DATASET
        from src.features.build_features import (
            FeatureBuildError,
            build_and_save_features,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"

            # Light_kW_m2 exists but is entirely missing -> should raise.
            self._write_csv(
                processed / BRINES_DATASET.filename,
                header=list(BRINE_FEATURE_COLUMNS),
                rows=[
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, ""],
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, ""],
                ],
            )

            # Minimal experimental.csv needed for the function to run.
            self._write_csv(
                processed / "experimental.csv",
                header=[
                    "TDS_gL",
                    "MLR",
                    "Light_kW_m2",
                    "Selectivity",
                    "Li_Crystallization_mg_m2_h",
                    "Evap_kg_m2_h",
                ],
                rows=[[9, 8, 100, 0.1, 1.0, 2.0]],
            )

            with self.assertRaises(FeatureBuildError):
                build_and_save_features(processed)


if __name__ == "__main__":
    unittest.main()
