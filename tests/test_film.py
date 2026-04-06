"""Tests for FiLM conditioning module."""
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is required for FiLM tests")
class TestFiLMLayer(unittest.TestCase):
    def test_identity_initialization(self):
        from src.models.film import FiLMConfig, FiLMLayer

        layer = FiLMLayer(FiLMConfig(cond_dim=1, latent_dim=16))
        cond = torch.zeros(2, 1)
        gamma, beta = layer(cond)

        # At init with zero input: gamma=1, beta=0 (identity).
        self.assertTrue(torch.allclose(gamma, torch.ones(2, 16), atol=1e-6))
        self.assertTrue(torch.allclose(beta, torch.zeros(2, 16), atol=1e-6))

    def test_output_shapes(self):
        from src.models.film import FiLMConfig, FiLMLayer

        layer = FiLMLayer(FiLMConfig(cond_dim=1, latent_dim=32))
        cond = torch.randn(5, 1)
        gamma, beta = layer(cond)

        self.assertEqual(gamma.shape, (5, 32))
        self.assertEqual(beta.shape, (5, 32))

    def test_parameter_count(self):
        from src.models.film import FiLMConfig, FiLMLayer

        layer = FiLMLayer(FiLMConfig(cond_dim=1, latent_dim=128))
        n_params = sum(p.numel() for p in layer.parameters())
        # Linear(1, 256): 1*256 + 256 = 512
        self.assertEqual(n_params, 512)


@unittest.skipIf(torch is None, "torch is required for FiLM tests")
class TestFiLMRegressionHead(unittest.TestCase):
    def test_forward_pass(self):
        from src.models.film import FiLMConfig, FiLMRegressionHead
        from src.models.regression_head import RegressionHeadConfig

        film_head = FiLMRegressionHead(
            in_dim=16,
            head_config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
            film_config=FiLMConfig(cond_dim=1, latent_dim=16),
        )
        z = torch.randn(4, 16)
        cond = torch.randn(4, 1)
        out = film_head(z, cond)

        self.assertEqual(out.shape, (4, 3))

    def test_different_light_gives_different_output(self):
        from src.models.film import FiLMConfig, FiLMRegressionHead
        from src.models.regression_head import RegressionHeadConfig

        film_head = FiLMRegressionHead(
            in_dim=16,
            head_config=RegressionHeadConfig(hidden_dim=8, n_layers=1, out_dim=3),
            film_config=FiLMConfig(cond_dim=1, latent_dim=16),
        )
        # Train for 1 step so FiLM weights are nonzero.
        z = torch.randn(2, 16)
        cond = torch.tensor([[0.0], [1.0]])
        target = torch.randn(2, 3)

        optim = torch.optim.SGD(film_head.parameters(), lr=0.1)
        loss = torch.nn.MSELoss()(film_head(z, cond), target)
        loss.backward()
        optim.step()

        # Now test: same z, different light should give different outputs.
        film_head.eval()
        z_same = torch.randn(1, 16)
        out_low = film_head(z_same, torch.tensor([[0.0]]))
        out_high = film_head(z_same, torch.tensor([[2.0]]))
        self.assertFalse(torch.allclose(out_low, out_high))


if __name__ == "__main__":
    unittest.main()
