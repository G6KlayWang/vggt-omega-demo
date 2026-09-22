import unittest
import numpy as np
import torch
from core import TinyOmega, RegisterStage, attention_pairs
from geometry import unproject
from toy_data import make_batch, triangulate


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_register_exchange_does_not_directly_change_patches(self):
        stage = RegisterStage().eval()
        x = torch.randn(2, 3, 20, 48)
        local = stage(x, cut_exchange=True)
        mixed = stage(x)
        torch.testing.assert_close(local[:, :, 4:], mixed[:, :, 4:])
        self.assertGreater((local[:, :, :4] - mixed[:, :, :4]).abs().max().item(), .001)

    def test_cross_view_gradient_requires_exchange(self):
        for mode, cut, connected in [("local", False, False), ("register", True, False),
                                      ("register", False, True), ("global", False, True)]:
            model = TinyOmega(mode).eval()
            obs, _ = make_batch(2, torch.Generator().manual_seed(3))
            obs.requires_grad_()
            model(obs, cut_exchange=cut).sum().backward()
            magnitude = obs.grad[:, 1:].abs().sum().item()
            if connected:
                self.assertGreater(magnitude, 1e-6)
            else:
                self.assertEqual(magnitude, 0)

    def test_side_view_permutation_does_not_change_reference(self):
        obs, _ = make_batch(2, torch.Generator().manual_seed(3))
        for mode in ["register", "global", "local"]:
            model = TinyOmega(mode).eval()
            torch.testing.assert_close(model(obs), model(obs[:, [0, 2, 1]]), atol=2e-6, rtol=2e-5)

    def test_reference_has_no_depth_signal(self):
        obs, xyz = make_batch(4, torch.Generator().manual_seed(2), noise=0)
        torch.testing.assert_close(obs[0, 0], obs[1, 0])
        self.assertGreater((xyz[0, :, 2] - xyz[1, :, 2]).abs().sum().item(), 1)

    def test_noiseless_oracle_recovers_depth(self):
        obs, xyz = make_batch(8, torch.Generator().manual_seed(8), noise=0)
        torch.testing.assert_close(triangulate(obs), xyz[..., 2], atol=1e-6, rtol=1e-6)

    def test_attention_cost_counts(self):
        self.assertEqual(attention_pairs(3), {"local": 2800, "register": 3088, "global": 10000})

    def test_pinhole_unprojection_roundtrip(self):
        angle = .4
        r = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        t = np.array([.2, -.3, 1.])
        ext = np.column_stack([r, t])
        k = np.array([[20., 0, 2], [0, 25., 1.5], [0, 0, 1]])
        depth = np.linspace(1, 3, 20).reshape(4, 5)
        world = unproject(depth, k, ext)
        camera = world @ r.T + t
        pixels = camera @ k.T
        pixels = pixels[..., :2] / pixels[..., 2:]
        v, u = np.mgrid[:4, :5]
        np.testing.assert_allclose(pixels, np.stack([u, v], -1), atol=1e-12)
        np.testing.assert_allclose(camera[..., 2], depth, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
