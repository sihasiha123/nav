# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局动态障碍物资产：生成、运动模型和运行时状态。"""

from __future__ import annotations

import math
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

__all__ = [
    "GlobalRigidObjectCollection",
    "GlobalObstacleManager",
    "GlobalObstacleMotionCfg",
    "get_global_obstacle_manager",
    "has_scene_entity",
    "make_global_obstacle_collection_cfg",
    "step_global_obstacles",
]


##
# 运动参数
##


@configclass
class GlobalObstacleMotionCfg:
    """有界随机航点运动配置。"""

    asset_name: str = "dynamic_obstacles"
    """全局 :class:`RigidObjectCollection` 在场景中的注册名称。"""

    motion_half_extent: tuple[float, float, float] = (1.0, 1.0, 0.4)
    """每个障碍物相对初始运动原点允许的 XYZ 最大偏移。"""

    speed_range: tuple[float, float] = (0.25, 0.75)
    """追踪随机航点的最小和最大速度，单位为 m/s。"""

    arrival_threshold: float = 0.05
    """判定到达航点并重新采样的距离阈值，单位为 m。"""


##
# 运动引擎
##


class GlobalObstacleManager:
    """使用有界随机航点驱动一套全局刚体集合。

    场景中的全局集合状态形状为 ``[1, num_objects, ...]``。该管理器将运动状态
    保存在刚体集合所在的 Torch 设备上，并通过一次批量调用更新全部障碍物。
    它不使用机器人的 ``env_ids``，应独立于单个回合，在每个物理步中只调用一次。
    """

    def __init__(self, env: ManagerBasedEnv, cfg: GlobalObstacleMotionCfg | None = None):
        self._physics_dt = env.physics_dt
        self.cfg = (cfg or GlobalObstacleMotionCfg()).copy()
        scene_asset = env.scene[self.cfg.asset_name]
        if not isinstance(scene_asset, RigidObjectCollection):
            raise TypeError(
                f"Scene entity '{self.cfg.asset_name}' must be a RigidObjectCollection, "
                f"received {type(scene_asset).__name__}."
            )
        self.asset = scene_asset

        self._validate_cfg()
        if self.asset.num_instances != 1:
            raise RuntimeError(
                "GlobalObstacleManager expects one global collection instance, "
                f"but '{self.cfg.asset_name}' has {self.asset.num_instances}."
            )

        self._motion_half_extent = torch.tensor(
            self.cfg.motion_half_extent,
            dtype=torch.float32,
            device=self.asset.device,
        ).view(1, 1, 3)

        self._initialized = False
        self._anchor_pos_w: torch.Tensor
        self._position_w: torch.Tensor
        self._target_pos_w: torch.Tensor
        self._linear_velocity_w: torch.Tensor
        self._velocity_w: torch.Tensor
        self._speed: torch.Tensor
        self._pose_w: torch.Tensor

    @property
    def physics_dt(self) -> float:
        """运动更新使用的物理时间步。"""
        return self._physics_dt

    def initialize(self) -> None:
        """初始化运动原点，并为所有障碍物采样第一个航点。"""
        default_state = self.asset.data.default_object_state
        if default_state.ndim != 3 or default_state.shape[0] != 1 or default_state.shape[-1] != 13:
            raise RuntimeError(
                "Global obstacle state must have shape [1, num_objects, 13], "
                f"received {tuple(default_state.shape)}."
            )

        self._anchor_pos_w = default_state[..., :3].clone()
        self._pose_w = self.asset.data.object_link_pose_w.clone()
        self._position_w = self._pose_w[..., :3].clone()
        self._target_pos_w = self._anchor_pos_w.clone()
        self._linear_velocity_w = torch.zeros_like(self._position_w)
        self._velocity_w = torch.zeros(
            (*self._position_w.shape[:-1], 6),
            dtype=self._position_w.dtype,
            device=self._position_w.device,
        )
        self._speed = torch.empty(
            (*self._position_w.shape[:-1], 1),
            dtype=self._position_w.dtype,
            device=self._position_w.device,
        )

        all_objects = torch.ones(
            self._position_w.shape[:-1],
            dtype=torch.bool,
            device=self._position_w.device,
        )
        self._sample_waypoints(all_objects)
        self._initialized = True

    def step(self, dt: float) -> None:
        """推进一个物理步，并批量写入所有障碍物的位姿和速度。"""
        if dt <= 0.0:
            raise ValueError(f"Obstacle motion time-step must be positive, received: {dt}.")
        if not self._initialized:
            self.initialize()

        # 为物理步开始时已经到达目标的障碍物重新采样航点。
        delta = self._target_pos_w - self._position_w
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        arrived = distance[..., 0] <= self.cfg.arrival_threshold
        self._sample_waypoints(arrived)

        # 重新计算目标方向，并执行一次不会越过航点的直线积分。
        delta = self._target_pos_w - self._position_w
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        direction = delta / distance.clamp_min(1.0e-6)
        travel = torch.minimum(self._speed * dt, distance)
        displacement = direction * travel

        self._position_w.add_(displacement)
        self._linear_velocity_w.copy_(displacement / dt)
        self._velocity_w[..., :3] = self._linear_velocity_w
        self._pose_w[..., :3].copy_(self._position_w)

        # 运动学刚体直接按脚本写入位姿和速度（角速度恒为零）；
        # 速度进 PhysX 后，无人机与障碍物接触时碰撞响应使用真实相对速度。
        self.asset.write_object_link_pose_to_sim(self._pose_w)
        self.asset.write_object_link_velocity_to_sim(self._velocity_w)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """障碍物运动原点，形状为 ``[1, num_objects, 3]``。"""
        self._ensure_initialized()
        return self._anchor_pos_w

    @property
    def position_w(self) -> torch.Tensor:
        """脚本控制的障碍物位置，形状为 ``[1, num_objects, 3]``。"""
        self._ensure_initialized()
        return self._position_w

    @property
    def target_pos_w(self) -> torch.Tensor:
        """当前局部航点，形状为 ``[1, num_objects, 3]``。"""
        self._ensure_initialized()
        return self._target_pos_w

    @property
    def linear_velocity_w(self) -> torch.Tensor:
        """实际脚本线速度，形状为 ``[1, num_objects, 3]``。"""
        self._ensure_initialized()
        return self._linear_velocity_w

    def _sample_waypoints(self, object_mask: torch.Tensor) -> None:
        """为 ``object_mask`` 选中的障碍物采样新目标和速度。"""
        random_offset = (2.0 * torch.rand_like(self._target_pos_w) - 1.0) * self._motion_half_extent
        candidate_targets = self._anchor_pos_w + random_offset

        min_speed, max_speed = self.cfg.speed_range
        candidate_speeds = torch.empty_like(self._speed).uniform_(min_speed, max_speed)

        selection = object_mask.unsqueeze(-1)
        self._target_pos_w.copy_(torch.where(selection, candidate_targets, self._target_pos_w))
        self._speed.copy_(torch.where(selection, candidate_speeds, self._speed))

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _validate_cfg(self) -> None:
        if len(self.cfg.motion_half_extent) != 3:
            raise ValueError("motion_half_extent must contain exactly three values.")
        if any(value < 0.0 for value in self.cfg.motion_half_extent):
            raise ValueError("motion_half_extent values must be non-negative.")
        if not any(value > 0.0 for value in self.cfg.motion_half_extent):
            raise ValueError("At least one motion_half_extent value must be positive.")

        min_speed, max_speed = self.cfg.speed_range
        if min_speed <= 0.0 or max_speed < min_speed:
            raise ValueError("speed_range must satisfy 0 < min_speed <= max_speed.")
        if self.cfg.arrival_threshold < 0.0:
            raise ValueError("arrival_threshold must be non-negative.")


##
# 外部接口
##


def get_global_obstacle_manager(
    env: ManagerBasedEnv,
    cfg: GlobalObstacleMotionCfg | None = None,
) -> GlobalObstacleManager:
    """返回动态障碍物资产所拥有的运动管理器。

    管理器引用挂在 ``GlobalRigidObjectCollection`` 资产上，而不是注入环境
    私有字段；这样障碍物的运行时状态和运动逻辑仍由资产领域模块负责。
    """
    scene_asset = env.scene[cfg.asset_name if cfg is not None else "dynamic_obstacles"]
    manager = getattr(scene_asset, "motion_manager", None)
    if manager is None:
        manager = GlobalObstacleManager(env, cfg)
        setattr(scene_asset, "motion_manager", manager)
    return manager


def has_scene_entity(env: ManagerBasedEnv, asset_name: str) -> bool:
    """检查场景中是否注册了指定实体。"""
    try:
        env.scene[asset_name]
        return True
    except KeyError:
        return False


def step_global_obstacles(
    env: ManagerBasedEnv,
    dt: float | None = None,
    cfg: GlobalObstacleMotionCfg | None = None,
) -> None:
    """推进全局障碍物一次，应在物理仿真前调用。"""
    physics_dt = env.physics_dt if dt is None else dt
    get_global_obstacle_manager(env, cfg).step(physics_dt)


##
# 场景配置生成
##


class GlobalRigidObjectCollection(RigidObjectCollection):
    """不会随并行环境重置的全局刚体集合。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 运行时运动状态属于动态障碍物资产本身。
        self.motion_manager: GlobalObstacleManager | None = None

    def reset(self, env_ids=None, object_ids=None) -> None:
        """忽略场景重置。"""
        pass

    def write_data_to_sim(self):
        """在场景写入阶段推进运动学障碍物，再写入其外力数据。"""
        if self.motion_manager is not None:
            self.motion_manager.step(self.motion_manager.physics_dt)
        super().write_data_to_sim()


def make_global_obstacle_collection_cfg(
    count: int = 100,
    terrain_size: tuple[float, float] = (40.0, 40.0),
    margin: float = 2.0,
    obstacle_size: tuple[float, float, float] = (0.5, 0.5, 1.0),
    obstacle_height: float = 1.5,
    obstacle_height_range: tuple[float, float] | None = None,
) -> RigidObjectCollectionCfg | None:
    """按照近似正方形网格创建一套全局障碍物集合。

    配置中的初始位置同时作为运动原点，后续运行时管理器可以从
    ``default_object_state`` 中读取这些位置。``count <= 0`` 时返回
    ``None``，表示禁用动态障碍物。传入 ``obstacle_height_range`` 时，
    每个障碍物的中心高度在范围内独立随机采样。
    """
    if count < 0:
        raise ValueError(f"Obstacle count must be non-negative, received: {count}.")
    if count == 0:
        return None
    if margin < 0.0:
        raise ValueError(f"Obstacle margin must be non-negative, received: {margin}.")
    if obstacle_height_range is not None:
        if obstacle_height_range[0] < 0.0 or obstacle_height_range[1] < obstacle_height_range[0]:
            raise ValueError(
                f"obstacle_height_range must satisfy 0 <= min <= max, received: {obstacle_height_range}."
            )

    terrain_width, terrain_length = terrain_size
    usable_width = terrain_width - 2.0 * margin
    usable_length = terrain_length - 2.0 * margin
    if usable_width <= 0.0 or usable_length <= 0.0:
        raise ValueError("Obstacle margin leaves no usable terrain area.")

    num_cols = math.ceil(math.sqrt(count))
    num_rows = math.ceil(count / num_cols)
    cell_width = usable_width / num_cols
    cell_length = usable_length / num_rows

    obstacle_cfgs: dict[str, RigidObjectCfg] = {}
    for obstacle_index in range(count):
        row, col = divmod(obstacle_index, num_cols)
        x = -0.5 * terrain_width + margin + (col + 0.5) * cell_width
        y = -0.5 * terrain_length + margin + (row + 0.5) * cell_length

        obstacle_name = f"obstacle_{obstacle_index:03d}"
        if obstacle_height_range is not None:
            obstacle_height = random.uniform(*obstacle_height_range)
        obstacle_cfgs[obstacle_name] = RigidObjectCfg(
            prim_path=f"/World/Dynamic/Obstacle_{obstacle_index:03d}",
            spawn=sim_utils.CuboidCfg(
                size=obstacle_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, obstacle_height)),
            collision_group=-1,
        )

    return RigidObjectCollectionCfg(
        class_type=GlobalRigidObjectCollection,
        rigid_objects=obstacle_cfgs,
    )
