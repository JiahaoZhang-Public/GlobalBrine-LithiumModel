import unittest


class TestSweepMae(unittest.TestCase):
    def test_generate_config_grid_filters_invalid(self):
        from src.models.sweep_mae import generate_config_grid

        cfgs = generate_config_grid(
            d_model=[64],
            n_heads=[3, 4],  # 3 is invalid for 64
            n_layers=[1],
            mlp_ratio=[2.0],
            dropout=[0.0],
            mask_ratio=[0.4],
        )
        self.assertEqual(len(cfgs), 1)
        self.assertEqual(cfgs[0].d_model, 64)
        self.assertEqual(cfgs[0].n_heads, 4)


if __name__ == "__main__":
    unittest.main()
