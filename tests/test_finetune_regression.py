import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(np is None, "numpy is required for finetune tests")
@unittest.skipIf(torch is None, "torch is required for finetune tests")
class TestFinetuneRegression(unittest.TestCase):
    def test_finetune_runs_with_current_feature_shapes(self):
        from src.constants import BRINE_FEATURE_COLUMNS
        from src.models.finetune_regression import (
            FinetuneConfig,
            finetune_regression_head,
        )
        from src.models.mae import TabularMAE, TabularMAEConfig
        from src.models.regression_head import RegressionHeadConfig

        encoder = TabularMAE(
            num_features=len(BRINE_FEATURE_COLUMNS),
            config=TabularMAEConfig(d_model=16, n_heads=4, n_layers=1),
        )
        x_exp = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
            ],
            dtype=np.float32,
        )
        y_exp = np.zeros((2, 3), dtype=np.float32)

        head = finetune_regression_head(
            x_exp,
            y_exp,
            encoder,
            head_config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
            finetune_config=FinetuneConfig(epochs=1, batch_size=2, device="cpu"),
            freeze_encoder=True,
            mae_feature_names=list(BRINE_FEATURE_COLUMNS),
        )
        self.assertIsNotNone(head)


if __name__ == "__main__":
    unittest.main()
