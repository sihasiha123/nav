"""Grounded corridor meshes compatible with the navigation collision queries."""

import numpy as np
import trimesh


def corridor_obstacle_layout(
    size,
    wall_thickness,
    num_static,
    num_dynamic,
    seed=0,
    obstacle_width_range=(0.45, 0.70),
    obstacle_depth_range=(0.35, 0.55),
):
    """Sample shared static boxes and moving anchors in centered world coordinates."""
    width, length = size
    clear_width = width - 2.0 * wall_thickness
    if wall_thickness <= 0.0 or clear_width < 4.0 - 1.0e-6 or length < 12.0:
        raise ValueError("Corridor requires positive walls, at least 4 m clear width and 12 m length.")
    if num_static < 0 or num_dynamic < 0:
        raise ValueError("Corridor obstacle counts must be non-negative.")
    if (
        len(obstacle_width_range) != 2
        or len(obstacle_depth_range) != 2
        or obstacle_width_range[0] <= 0.0
        or obstacle_depth_range[0] <= 0.0
        or obstacle_width_range[1] < obstacle_width_range[0]
        or obstacle_depth_range[1] < obstacle_depth_range[0]
    ):
        raise ValueError("Corridor obstacle size ranges must be positive and ordered.")
    count = num_static + num_dynamic
    if count == 0:
        return [], []
    spacing = (length - 8.0) / count
    if 0.8 * spacing <= 0.7:
        raise ValueError("Too many corridor obstacles to preserve longitudinal clearance.")

    rng = np.random.default_rng(seed)
    moving_slots = {int((index + 0.5) * count / num_dynamic) for index in range(num_dynamic)}
    static_boxes, moving_positions = [], []
    y_positions = [
        -length / 2.0 + 4.0 + (index + 0.5) * spacing + rng.uniform(-0.22, 0.22) * spacing
        for index in range(count)
    ]
    lane_centers = (-1.30, 0.0, 1.30)
    lane_orders = [rng.permutation(3).tolist() for _ in range((count + 2) // 3)]
    for index in range(count):
        y = y_positions[index]
        if index in moving_slots:
            moving_positions.append((rng.uniform(-0.15, 0.15), y, rng.uniform(1.25, 1.85)))
        else:
            # Every group of three blocks left, center, and right in a randomized order.
            # This removes a continuous unobstructed lane from start to goal.
            lane = lane_orders[index // 3][index % 3]
            x = lane_centers[lane] + rng.uniform(-0.10, 0.10)
            static_boxes.append(
                (
                    (x, y),
                    (
                        rng.uniform(*obstacle_width_range),
                        rng.uniform(*obstacle_depth_range),
                    ),
                )
            )
    return static_boxes, moving_positions


def corridor_terrain(difficulty, cfg):
    """Return meshes in the generator's positive-XY sub-terrain coordinates."""
    width, length = cfg.size
    wall = cfg.wall_thickness
    if cfg.wall_height <= 0.0:
        raise ValueError("Corridor wall height must be positive.")
    if cfg.static_obstacle_height <= 0.0:
        raise ValueError("Corridor static obstacle height must be positive.")
    static_boxes, _ = corridor_obstacle_layout(
        cfg.size,
        wall,
        cfg.num_obstacles,
        cfg.num_dynamic_obstacles,
        cfg.seed,
        cfg.obstacle_width_range,
        cfg.obstacle_depth_range,
    )

    def box(size, center):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        return mesh

    meshes = [box((width, length, 0.1), (width / 2.0, length / 2.0, -0.05))]
    for x in (wall / 2.0, width - wall / 2.0):
        meshes.append(box((wall, length, cfg.wall_height), (x, length / 2.0, cfg.wall_height / 2.0)))

    # Paired compact gates shift the available passage between rows.
    for (x, y), (pillar_width, pillar_depth) in static_boxes:
        meshes.append(
            box(
                (pillar_width, pillar_depth, cfg.static_obstacle_height),
                (
                    width / 2.0 + x,
                    length / 2.0 + y,
                    cfg.static_obstacle_height / 2.0,
                ),
            )
        )
    return meshes, np.array((width / 2.0, length / 2.0, 0.0))
