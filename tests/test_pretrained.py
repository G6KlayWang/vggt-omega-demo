import unittest
from types import SimpleNamespace

import numpy as np

from pretrained import camera_frustums, non_sky_mask


class ViewerTests(unittest.TestCase):
    def test_camera_center_and_frustum_reproject(self):
        k = np.array([[80., 0, 40], [0, 80, 30], [0, 0, 1]])
        rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        center = np.array([2., 3, 4])
        e = np.column_stack([rotation, -rotation @ center])
        origin = np.array([1., 1, 1])
        vertices = np.array(camera_frustums([k], [e], (61, 81), origin, 2))[0]
        world = vertices * 2 + origin
        np.testing.assert_allclose(world[0], center)
        camera = world[1:] @ rotation.T + e[:, 3]
        projected = camera @ k.T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:],
                                   [[0, 0], [80, 0], [80, 60], [0, 60]], atol=1e-4)

    def test_sky_mask_alignment_and_direction(self):
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="rgb")]

            def run(self, _, inputs):
                self.tensor = inputs["rgb"]
                score = np.zeros((1, 1, 320, 320), dtype=np.float32)
                score[:, :, :160] = 1  # Sky above, ground below.
                return [score]

        session = Session()
        mask = non_sky_mask(np.ones((2, 40, 80, 3), dtype=np.float32), session)
        self.assertEqual(mask.shape, (2, 40, 80))
        self.assertFalse(mask[:, :19].any())
        self.assertTrue(mask[:, 21:].all())
        self.assertEqual(session.tensor.shape, (1, 3, 320, 320))
        self.assertEqual(session.tensor.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
