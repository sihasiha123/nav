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
            size=(4.6, 40.0), wall_thickness=0.3, wall_height=6.0,
            static_obstacle_height=6.0,
            obstacle_width_range=(0.70, 0.90),
            obstacle_depth_range=(0.45, 0.70),
            num_obstacles=12, num_dynamic_obstacles=0, seed=0,
        )

    def layout(self, seed=0):
        return corridor.corridor_obstacle_layout(
            self.cfg.size, self.cfg.wall_thickness,
            self.cfg.num_obstacles, self.cfg.num_dynamic_obstacles, seed,
            self.cfg.obstacle_width_range, self.cfg.obstacle_depth_range,
        )

    def test_layout_and_open_ends(self):
        meshes, origin = corridor.corridor_terrain(0.0, self.cfg)
        self.assertEqual(len(meshes), 15)
        np.testing.assert_allclose(origin, (2.3, 20.0, 0.0))
        self.assertTrue(all(mesh.is_watertight for mesh in meshes))
        self.assertAlmostEqual(meshes[2].bounds[0, 0] - meshes[1].bounds[1, 0], 4.0)
        for pillar in meshes[3:]:
            self.assertGreater(pillar.bounds[0, 1], 4.0)
            self.assertLess(pillar.bounds[1, 1], 36.0)
            self.assertGreaterEqual(pillar.bounds[0, 0], 0.3 - 1.0e-6)
            self.assertLessEqual(pillar.bounds[1, 0], 4.3 + 1.0e-6)
            self.assertAlmostEqual(pillar.bounds[0, 2], 0.0)
            self.assertAlmostEqual(pillar.extents[2], self.cfg.static_obstacle_height)

    def test_dynamic_sweeps_do_not_touch_walls_or_pillars(self):
        for seed in range(64):
            static, moving = corridor.corridor_obstacle_layout(
                self.cfg.size,
                self.cfg.wall_thickness,
                0,
                num_dynamic=8,
                seed=seed,
            )
            self.assertEqual(len(static), 0)
            self.assertEqual(len(moving), 8)
            for x, y, z in moving:
                self.assertLess(abs(x) + 1.5 + 0.25, 2.5)
                self.assertLess(abs(y), 16.0)

            static, moving = self.layout(seed)
            self.assertEqual(len(static), 12)
            self.assertEqual(len(moving), 0)
            for (x, y), (width, depth) in static:
                self.assertGreaterEqual(x - width / 2.0, -2.0)
                self.assertLessEqual(x + width / 2.0, 2.0)
                self.assertLess(width, 0.91)
                self.assertGreater(width, 0.69)
                self.assertLess(abs(y), 16.0 - depth / 2.0)
    def test_static_layout_matches_mesh_and_is_irregular(self):
        static, _ = self.layout()
        meshes, _ = corridor.corridor_terrain(0.0, self.cfg)
        for index, (((x, y), (width, depth)), mesh) in enumerate(zip(static, meshes[3:])):
            np.testing.assert_allclose(mesh.extents, (width, depth, self.cfg.static_obstacle_height))
            np.testing.assert_allclose(
                mesh.centroid,
                (x + 2.3, y + 20.0, self.cfg.static_obstacle_height / 2.0),
            )
        self.assertGreater(np.std(np.diff([p[0][1] for p in static])), 0.01)
        self.assertGreater(np.std([p[0][0] for p in static]), 0.1)
        for first in range(0, len(static), 3):
            lane_centers = sorted(round(static[index][0][0], 1) for index in range(first, first + 3))
            self.assertEqual(len(lane_centers), 3)
            self.assertLess(lane_centers[0], -1.0)
            self.assertAlmostEqual(lane_centers[1], 0.0, delta=0.2)
            self.assertGreater(lane_centers[2], 1.0)

    def test_zero_dynamic_and_different_seed(self):
        self.cfg.num_dynamic_obstacles = 0
        static, moving = self.layout()
        self.assertEqual(len(static), 12)
        self.assertEqual(moving, [])
        self.assertNotEqual(static, self.layout(seed=1)[0])

    def test_seed_reproducibility_and_no_global_rng_change(self):
        np.random.seed(123)
        state = np.random.get_state()
        first, _ = corridor.corridor_terrain(0.0, self.cfg)
        second, _ = corridor.corridor_terrain(0.0, self.cfg)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.vertices, right.vertices)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(state[2:], np.random.get_state()[2:])

    def test_invalid_dimensions(self):
        self.cfg.size = (0.5, 40.0)
        with self.assertRaises(ValueError):
            corridor.corridor_terrain(0.0, self.cfg)

    def test_overcrowded_layout_is_rejected(self):
        self.cfg.num_obstacles = 100
        with self.assertRaises(ValueError):
            corridor.corridor_terrain(0.0, self.cfg)


if __name__ == "__main__":
    unittest.main()
