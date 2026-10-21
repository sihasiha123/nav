"""Test corridor geometry without starting Isaac Sim."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "source/nav/nav/assets/corridor.py"
spec = importlib.util.spec_from_file_location("corridor_geometry", path)
corridor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corridor)


class CorridorTerrainTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            size=(4.6, 40.0),
            wall_thickness=0.3,
            wall_height=6.0,
            static_obstacle_height=6.0,
            obstacle_width_range=(0.70, 0.90),
            obstacle_depth_range=(0.45, 0.70),
            num_obstacles=10,
            num_dynamic_obstacles=5,
            seed=0,
        )

    def layout(self, seed=0):
        return corridor.corridor_obstacle_layout(
            self.cfg.size,
            self.cfg.wall_thickness,
            self.cfg.num_obstacles,
            self.cfg.num_dynamic_obstacles,
            seed,
            self.cfg.obstacle_width_range,
            self.cfg.obstacle_depth_range,
        )

    def test_layout_and_open_ends(self):
        meshes, origin = corridor.corridor_terrain(0.0, self.cfg)
        self.assertEqual(len(meshes), 13)
        np.testing.assert_allclose(origin, (2.3, 20.0, 0.0))
        self.assertTrue(all(mesh.is_watertight for mesh in meshes))
        self.assertAlmostEqual(meshes[2].bounds[0, 0] - meshes[1].bounds[1, 0], 4.0)
        for pillar in meshes[3:]:
            self.assertGreater(pillar.bounds[0, 1], 4.0)
            self.assertLess(pillar.bounds[1, 1], 36.0)
            self.assertAlmostEqual(pillar.extents[2], self.cfg.static_obstacle_height)

    def test_layout_reproducibility_and_dynamic_bounds(self):
        static, moving = self.layout()
        self.assertEqual((static, moving), self.layout())
        self.assertEqual(len(static), 10)
        self.assertEqual(len(moving), 5)
        for x, y, _ in moving:
            self.assertLess(abs(x) + 1.5 + 0.25, 2.5)
            self.assertLess(abs(y), 16.0)

    def test_invalid_dimensions(self):
        self.cfg.size = (0.5, 40.0)
        with self.assertRaises(ValueError):
            corridor.corridor_terrain(0.0, self.cfg)


if __name__ == "__main__":
    unittest.main()
