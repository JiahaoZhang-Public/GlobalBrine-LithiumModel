import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(np is None, "numpy is required for attribution tests")
@unittest.skipIf(torch is None, "torch is required for attribution tests")
class TestAttributions(unittest.TestCase):
    def _make_scaler(self):
        from src.constants import (
            BRINE_FEATURE_COLUMNS,
            EXPERIMENTAL_FEATURE_COLUMNS,
            EXPERIMENTAL_TARGET_COLUMNS,
        )

        return {
            "brine_chemistry": {
                "feature_names": list(BRINE_FEATURE_COLUMNS),
                "mean": [0.0] * len(BRINE_FEATURE_COLUMNS),
                "std": [1.0] * len(BRINE_FEATURE_COLUMNS),
            },
            "experimental_features": {
                "feature_names": list(EXPERIMENTAL_FEATURE_COLUMNS),
                "mean": [0.0] * len(EXPERIMENTAL_FEATURE_COLUMNS),
                "std": [1.0] * len(EXPERIMENTAL_FEATURE_COLUMNS),
            },
            "experimental_targets": {
                "feature_names": list(EXPERIMENTAL_TARGET_COLUMNS),
                "mean": [0.0] * len(EXPERIMENTAL_TARGET_COLUMNS),
                "std": [1.0] * len(EXPERIMENTAL_TARGET_COLUMNS),
            },
        }

    def test_integrated_gradients_for_prediction_runs_with_z_only_head(self):
        from src.constants import BRINE_FEATURE_COLUMNS
        from src.interpretability.attributions import (
            integrated_gradients_for_prediction,
        )
        from src.models.inference import InferenceArtifacts
        from src.models.mae import TabularMAE, TabularMAEConfig
        from src.models.regression_head import RegressionHead, RegressionHeadConfig

        mae = TabularMAE(
            num_features=len(BRINE_FEATURE_COLUMNS),
            config=TabularMAEConfig(d_model=8, n_heads=4, n_layers=1),
        )
        head = RegressionHead(
            in_dim=mae.config.d_model,
            config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
        )
        artifacts = InferenceArtifacts(mae=mae, head=head, scaler=self._make_scaler())

        sample = {"TDS_gL": 1.0, "MLR": 2.0, "Light_kW_m2": 3.0}
        res = integrated_gradients_for_prediction(artifacts, sample=sample, steps=4)
        self.assertEqual(set(res.features), {"TDS_gL", "MLR", "Light_kW_m2"})
        self.assertEqual(len(res.targets), 3)
        self.assertTrue(all(np.isfinite(list(res.deltas.values()))))

    def test_integrated_gradients_for_prediction_runs_with_concat_light_head(self):
        from src.constants import BRINE_FEATURE_COLUMNS
        from src.interpretability.attributions import (
            integrated_gradients_for_prediction,
        )
        from src.models.inference import InferenceArtifacts
        from src.models.mae import TabularMAE, TabularMAEConfig
        from src.models.regression_head import RegressionHead, RegressionHeadConfig

        mae = TabularMAE(
            num_features=len(BRINE_FEATURE_COLUMNS),
            config=TabularMAEConfig(d_model=8, n_heads=4, n_layers=1),
        )
        head = RegressionHead(
            in_dim=mae.config.d_model + 1,
            config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
        )
        artifacts = InferenceArtifacts(mae=mae, head=head, scaler=self._make_scaler())

        sample = {"TDS_gL": 1.0, "MLR": 2.0, "Light_kW_m2": 3.0}
        res = integrated_gradients_for_prediction(artifacts, sample=sample, steps=4)
        self.assertEqual(set(res.features), {"TDS_gL", "MLR", "Light_kW_m2"})
        self.assertEqual(len(res.targets), 3)
        self.assertTrue(all(np.isfinite(list(res.deltas.values()))))


if __name__ == "__main__":
    unittest.main()
