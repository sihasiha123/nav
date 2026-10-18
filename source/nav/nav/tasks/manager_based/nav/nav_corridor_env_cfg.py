"""A separate 5 m x 40 m corridor task; the original square task is unchanged."""

from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from nav.assets.corridor import corridor_obstacle_layout, corridor_terrain
from nav.assets.dynamic import (
    GlobalObstacleMotionCfg,
    get_global_obstacle_manager,
    has_scene_entity,
    make_global_obstacle_collection_cfg,
)

from .nav_env_cfg import NavEnvCfg


@configclass
class CorridorTerrainCfg(SubTerrainBaseCfg):
    function = corridor_terrain
    wall_thickness: float = 0.3
    wall_height: float = 6.0
    static_obstacle_height: float = 6.0
    obstacle_width_range: tuple[float, float] = (0.70, 0.90)
    obstacle_depth_range: tuple[float, float] = (0.45, 0.70)
    num_obstacles: int = 12
    num_dynamic_obstacles: int = 0


def initialize_corridor_obstacles(env, env_ids=None):
    if not has_scene_entity(env, "dynamic_obstacles"):
        return
    # Anchors are centered across the passage; swept boxes stay inside the walls.
    motion_cfg = GlobalObstacleMotionCfg(
        motion_half_extent=(1.5, 0.0, 0.3),
        speed_range=(0.25, 0.75),
    )
    get_global_obstacle_manager(env, motion_cfg).initialize()


@configclass
class NavCorridorEnvCfg(NavEnvCfg):
    def __post_init__(self):
        # Include the two 0.3 m walls outside the 4 m clear passage.
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            size=(4.6, 40.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            use_cache=False,
            color_scheme="height",
            sub_terrains={"corridor": CorridorTerrainCfg()},
        )
        generator = self.scene.terrain.terrain_generator
        terrain_cfg = generator.sub_terrains["corridor"]
        _, moving_positions = corridor_obstacle_layout(
            generator.size,
            terrain_cfg.wall_thickness,
            terrain_cfg.num_obstacles,
            terrain_cfg.num_dynamic_obstacles,
            generator.seed,
        )
        self.scene.dynamic_obstacles = make_global_obstacle_collection_cfg(
            count=len(moving_positions),
            terrain_size=(5.0, 40.0),
            margin=1.75,
            obstacle_height=1.5,
        )
        obstacles = (
            list(self.scene.dynamic_obstacles.rigid_objects.values())
            if self.scene.dynamic_obstacles is not None else []
        )
        for obstacle, position in zip(obstacles, moving_positions, strict=True):
            obstacle.init_state.pos = position
        self.scene.depth_camera.mesh_prim_paths = [
            "/World/ground",
            *[
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr=obstacle.prim_path,
                    track_mesh_transforms=True,
                )
                for obstacle in obstacles
            ],
        ]
        self.events.initialize_dynamic_obstacles.func = initialize_corridor_obstacles
        self.events.reset_robot_state.params.update(
            start_x_range=(0.0, 0.0),
            start_y=18.0,
            start_z_range=(1.0, 1.6),
        )
        self.commands.nav_target.target_y = -18.0
        self.terminations.out_of_bounds.params.update(
            x_range=(-2.0, 2.0),
            y_range=(-20.0, 20.0),
        )
        super().__post_init__()
        self.viewer.eye = (12.0, 25.0, 22.0)
        self.viewer.lookat = (0.0, 0.0, 1.5)
