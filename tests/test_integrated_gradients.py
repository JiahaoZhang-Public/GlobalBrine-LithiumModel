import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is required for IG tests")
class TestIntegratedGradients(unittest.TestCase):
    def test_linear_model_matches_closed_form(self):
        from src.interpretability.integrated_gradients import integrated_gradients

        w = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float32)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            # x: [N, 3] -> y: [N, 1]
            y = (x * w).sum(dim=1, keepdim=True)
            return y

        inputs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        baseline = torch.zeros_like(inputs)

        res = integrated_gradients(
            forward_fn,
            inputs=inputs,
            baseline=baseline,
            target_index=0,
            steps=64,
        )

        expected = (inputs - baseline) * w
        self.assertTrue(torch.allclose(res.attributions, expected, atol=1e-4))
        self.assertTrue(torch.isfinite(res.delta).item())
        self.assertLess(abs(float(res.delta.item())), 1e-4)


if __name__ == "__main__":
    unittest.main()
