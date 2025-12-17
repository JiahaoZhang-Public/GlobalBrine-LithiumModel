import unittest

import numpy as np


class TestInferenceClamp(unittest.TestCase):
    def test_predict_labels_clamps_to_non_negative(self):
        import torch

        from src.models.inference import InferenceArtifacts, predict_labels
        from src.models.mae import TabularMAE, TabularMAEConfig
        from src.models.regression_head import RegressionHead, RegressionHeadConfig

        mae = TabularMAE(
            num_features=9, config=TabularMAEConfig(d_model=8, n_heads=4, n_layers=1)
        )
        head = RegressionHead(
            in_dim=mae.config.d_model + 1,
            config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
        )
        head.eval()
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
            head.net[-1].bias[:] = -1.0

        scaler = {
            "brine_chemistry": {
                "feature_names": [
                    "Li_gL",
                    "Mg_gL",
                    "Na_gL",
                    "K_gL",
                    "Ca_gL",
                    "SO4_gL",
                    "Cl_gL",
                    "MLR",
                    "TDS_gL",
                ],
                "mean": [0.0] * 9,
                "std": [1.0] * 9,
            },
            "experimental_features": {
                "feature_names": ["TDS_gL", "MLR", "Light_kW_m2"],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "experimental_targets": {
                "feature_names": [
                    "Selectivity",
                    "Li_Crystallization_mg_m2_h",
                    "Evap_kg_m2_h",
                ],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
        artifacts = InferenceArtifacts(mae=mae, head=head, scaler=scaler)

        samples = [{"TDS_gL": 1.0, "MLR": 2.0, "Light_kW_m2": 3.0}]
        y = predict_labels(artifacts, samples=samples, impute_missing_chemistry=False)
        self.assertTrue(np.all(y >= 0.0))


if __name__ == "__main__":
    unittest.main()
