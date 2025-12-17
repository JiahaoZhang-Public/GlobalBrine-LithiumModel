import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is required for MAE tests")
class TestTabularMAE(unittest.TestCase):
    def test_forward_and_encode_handle_nan(self):
        from src.models.mae import TabularMAE, TabularMAEConfig

        model = TabularMAE(
            num_features=4, config=TabularMAEConfig(d_model=32, n_heads=4, n_layers=2)
        )
        x = torch.tensor([[0.0, float("nan"), 1.0, 2.0]], dtype=torch.float32)

        pred = model(x)
        self.assertEqual(pred.shape, x.shape)
        self.assertTrue(torch.isfinite(pred).all().item())

        z = model.encode(x)
        self.assertEqual(z.shape, (1, model.config.d_model))
        self.assertTrue(torch.isfinite(z).all().item())

    def test_pretrain_loss_ignores_missing_targets(self):
        from src.models.mae import TabularMAE, TabularMAEConfig

        model = TabularMAE(
            num_features=3,
            config=TabularMAEConfig(d_model=16, n_heads=4, n_layers=1, mask_ratio=1.0),
        )
        x = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [0.0, float("nan"), 2.0],
            ],
            dtype=torch.float32,
        )
        loss = model.pretrain_loss(x, mask_ratio=1.0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_masking_rule_no_extra_masks_when_missing_exceeds_ratio(self):
        from src.models.mae import build_effective_masks

        x = torch.tensor([[float("nan"), 1.0, float("nan"), 2.0]], dtype=torch.float32)
        input_mask, loss_mask = build_effective_masks(x, mask_ratio=0.25, min_visible=1)
        self.assertTrue(input_mask[0, 0].item())
        self.assertTrue(input_mask[0, 2].item())
        # loss is computed on all observed features only.
        self.assertEqual(int(loss_mask.sum().item()), 2)
        # input_mask includes 2 missing + 1 MAE mask (small ratio still masks one if possible).
        self.assertEqual(int(input_mask.sum().item()), 3)

    def test_masking_ratio_applies_only_to_observed_features(self):
        from src.models.mae import build_effective_masks

        torch.manual_seed(0)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        input_mask, loss_mask = build_effective_masks(x, mask_ratio=0.4, min_visible=2)
        # F_obs=5 => num_mask=floor(5*0.4)=2, max_mask=5-2=3 => 2 masks.
        self.assertEqual(int(loss_mask.sum().item()), 5)
        self.assertEqual(int(input_mask.sum().item()), 2)

        torch.manual_seed(0)
        x2 = torch.tensor(
            [[float("nan"), 1.0, float("nan"), 3.0, 4.0]], dtype=torch.float32
        )
        input_mask2, loss_mask2 = build_effective_masks(
            x2, mask_ratio=0.4, min_visible=2
        )
        # F_obs=3 => num_mask=floor(3*0.4)=1, max_mask=3-2=1 => 1 MAE mask.
        self.assertEqual(int(loss_mask2.sum().item()), 3)
        # input_mask includes 2 missing + 1 MAE mask = 3 total.
        self.assertEqual(int(input_mask2.sum().item()), 3)

    def test_small_mask_ratio_still_masks_one_if_possible(self):
        from src.models.mae import build_effective_masks

        torch.manual_seed(0)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        input_mask, loss_mask = build_effective_masks(x, mask_ratio=0.1, min_visible=1)
        self.assertEqual(int(loss_mask.sum().item()), 4)
        self.assertEqual(int(input_mask.sum().item()), 1)

    def test_zero_loss_is_differentiable_when_no_masked_positions(self):
        from src.models.mae import TabularMAE, TabularMAEConfig

        model = TabularMAE(
            num_features=3,
            config=TabularMAEConfig(d_model=16, n_heads=4, n_layers=1, mask_ratio=0.1),
        )
        x = torch.tensor(
            [[float("nan"), float("nan"), float("nan")]], dtype=torch.float32
        )
        loss = model.pretrain_loss(x, mask_ratio=0.1)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()


if __name__ == "__main__":
    unittest.main()
