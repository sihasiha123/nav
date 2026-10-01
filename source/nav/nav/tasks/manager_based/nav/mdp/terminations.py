"""导航终止：球体几何碰撞、越界、安全到达与超时。

各项返回 (N,) bool。失败项必须在 success 之前配置，供 success 复用当前步结果。
"""

from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .commands import get_nav_target_command
from .geometry import body_position_w, heightfield_sphere_collision, spheres_touch_boxes

__all__ = ["StaticCollision", "DynamicCollision", "out_of_bounds", "success"]


def _validate_radius(radius: float) -> None:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("robot_radius must be finite and positive.")


class StaticCollision(ManagerTermBase):
    """机身球体与当前静态高度场/水平地面的离散接触检测。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _validate_radius(cfg.params.get("robot_radius", 0.3))
        terrain = env.scene[cfg.params.get("terrain_name", "terrain")]
        if not hasattr(terrain, "collision_mesh"):
            raise ValueError("StaticCollision requires CollisionTerrainImporter with a generated heightfield.")
        self._mesh = terrain.collision_mesh
        self._ray_top = terrain.collision_mesh_top
        self._result = torch.empty(env.num_envs, dtype=torch.int32, device=env.device)
        self._result_wp = wp.from_torch(self._result, dtype=wp.int32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        robot_radius: float = 0.3,
        terrain_name: str = "terrain",
        ground_height: float = 0.0,
    ) -> torch.Tensor:
        # 非有限位置由 out_of_bounds 结束，避免将 NaN/Inf 送入网格查询。
        positions = body_position_w(env, asset_cfg)
        finite = torch.isfinite(positions).all(dim=-1)
        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0).float().contiguous()
        # 与 PyTorch 使用同一 CUDA stream，保证位置读取和结果使用的先后顺序。
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device)) if positions.is_cuda else None
        wp.launch(
            kernel=heightfield_sphere_collision,
            dim=env.num_envs,
            inputs=[
                self._mesh.id, wp.from_torch(positions, dtype=wp.vec3), robot_radius,
                self._ray_top, ground_height, self._result_wp,
            ],
            device=env.device,
            stream=stream,
        )
        return (self._result != 0) & finite


class DynamicCollision(ManagerTermBase):
    """机身球体与全部共享动态方块检测，使用实际位姿，不依赖相机视野。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _validate_radius(cfg.params.get("robot_radius", 0.3))
        self._collection = env.scene[cfg.params.get("obstacle_name", "dynamic_obstacles")]
        if self._collection.num_instances != 1:
            raise ValueError("DynamicCollision expects one shared global obstacle collection.")
        sizes = [self._collection.cfg.rigid_objects[name].spawn.size for name in self._collection.object_names]
        if not sizes or any(
            len(size) != 3 or any(not math.isfinite(v) or v <= 0.0 for v in size) for size in sizes
        ):
            raise ValueError("DynamicCollision requires cuboid obstacles with positive XYZ dimensions.")
        self._half_sizes = 0.5 * torch.tensor(sizes, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        robot_radius: float = 0.3,
        obstacle_name: str = "dynamic_obstacles",
    ) -> torch.Tensor:
        positions = body_position_w(env, asset_cfg)
        poses = self._collection.data.object_link_pose_w[0]
        collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # 分块检查所有障碍物，限制中间 Tensor 大小；不改变碰撞检测范围。
        for first in range(0, poses.shape[0], 32):
            hits = spheres_touch_boxes(
                positions, poses[first:first + 32], self._half_sizes[first:first + 32], robot_radius
            )
            collision |= hits.any(dim=-1)
        return collision


def out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-24.0, 24.0),
    y_range: tuple[float, float] = (-24.0, 24.0),
    z_range: tuple[float, float] = (0.2, 4.0),
) -> torch.Tensor:
    """按 body 原点判断允许飞行区域，坐标相对 env_origin；包含当前起终点。"""
    for bounds in (x_range, y_range, z_range):
        if not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("Flight bounds must be finite and strictly increasing.")
    position = body_position_w(env, asset_cfg) - env.scene.env_origins
    outside = ~torch.isfinite(position).all(dim=-1)
    for axis, (lower, upper) in enumerate((x_range, y_range, z_range)):
        outside |= (position[:, axis] < lower) | (position[:, axis] > upper)
    return outside


def success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal_radius: float = 0.5,
    failure_terms: tuple[str, ...] = ("static_collision", "dynamic_collision", "out_of_bounds"),
) -> torch.Tensor:
    """进入目标半径且当前步没有失败；复用已计算结果，不重复做几何查询。"""
    _validate_radius(goal_radius)
    position = body_position_w(env, asset_cfg)
    distance = torch.linalg.vector_norm(get_nav_target_command(env) - position, dim=-1)
    reached = distance < goal_radius
    manager = env.termination_manager
    for name in failure_terms:
        if name in manager.active_terms:
            reached &= ~manager.get_term(name)
    return reached
