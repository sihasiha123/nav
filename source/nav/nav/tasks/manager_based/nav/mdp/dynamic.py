# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局动态障碍物的配置、集合类、运动引擎与动作项接入。"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

__all__ = [
    "GlobalRigidObjectCollection",
    "GlobalObstacleMotionAction",
    "GlobalObstacleMotionActionCfg",
    "GlobalObstacleManager",
    "GlobalObstacleMotionCfg",
    "get_global_obstacle_manager",
    "initialize_global_obstacles",
    "make_global_obstacle_collection_cfg",
    "step_global_obstacles",
]


class GlobalRigidObjectCollection(RigidObjectCollection):
    """不会随并行环境重置的全局刚体集合。"""

    def reset(self, env_ids=None, object_ids=None) -> None:
        """忽略场景重置。"""
        # 全局集合只有一个 instance，而 scene.reset() 传入的 env_ids 是 num_envs 维
        # 机器人索引，与集合的 instance 维度不匹配，因此完全忽略场景重置。
        # 障碍物每次训练都从新进程的 spawn 状态开始，由管理器懒初始化复位，
        # 不依赖场景重置；父类 reset() 仅清 wrench 缓冲，本集合不使用 wrench。
        pass


def make_global_obstacle_collection_cfg(
    count: int = 100,
    terrain_size: tuple[float, float] = (40.0, 40.0),
    margin: float = 2.0,
    obstacle_size: tuple[float, float, float] = (0.5, 0.5, 1.0),
    obstacle_height: float = 1.5,
) -> RigidObjectCollectionCfg:
    """按照近似正方形网格创建一套全局障碍物集合。

    配置中的初始位置同时作为运动原点，后续运行时管理器可以从
    ``default_object_state`` 中读取这些位置。
    """
    if count <= 0:
        raise ValueError(f"Obstacle count must be positive, received: {count}.")
    if margin < 0.0:
        raise ValueError(f"Obstacle margin must be non-negative, received: {margin}.")

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


_MANAGER_ATTRIBUTE = "_nav_global_obstacle_manager"


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


class GlobalObstacleManager:
    """使用有界随机航点驱动一套全局刚体集合。

    场景中的全局集合状态形状为 ``[1, num_objects, ...]``。该管理器将运动状态
    保存在刚体集合所在的 Torch 设备上，并通过一次批量调用更新全部障碍物。
    它不使用机器人的 ``env_ids``，应独立于单个回合，在每个物理步中只调用一次。
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: GlobalObstacleMotionCfg | None = None):
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
        self._angular_velocity_w: torch.Tensor
        self._speed: torch.Tensor
        self._pose_w: torch.Tensor

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
        self._angular_velocity_w = torch.zeros_like(self._position_w)
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
        self._pose_w[..., :3].copy_(self._position_w)

        # 场景配置使用运动学刚体，因此直接按脚本写入位姿和速度。角速度保持为零。
        # 速度写入 PhysX 后，无人机与障碍物接触时碰撞响应能使用真实的相对速度。
        link_velocity = torch.cat([self._linear_velocity_w, self._angular_velocity_w], dim=-1)
        self.asset.write_object_link_pose_to_sim(self._pose_w)
        self.asset.write_object_link_velocity_to_sim(link_velocity)

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


def get_global_obstacle_manager(
    env: ManagerBasedRLEnv,
    cfg: GlobalObstacleMotionCfg | None = None,
) -> GlobalObstacleManager:
    """返回环境的全局障碍物管理器，并在第一次使用时创建它。"""
    manager = getattr(env, _MANAGER_ATTRIBUTE, None)
    if manager is None:
        manager = GlobalObstacleManager(env, cfg)
        setattr(env, _MANAGER_ATTRIBUTE, manager)
    return manager


def initialize_global_obstacles(
    env: ManagerBasedRLEnv,
    cfg: GlobalObstacleMotionCfg | None = None,
) -> None:
    """初始化全局运动状态，但不推进仿真。"""
    get_global_obstacle_manager(env, cfg).initialize()


def step_global_obstacles(
    env: ManagerBasedRLEnv,
    dt: float | None = None,
    cfg: GlobalObstacleMotionCfg | None = None,
) -> None:
    """推进全局障碍物一次，应在物理仿真前调用。"""
    physics_dt = env.physics_dt if dt is None else dt
    get_global_obstacle_manager(env, cfg).step(physics_dt)


class GlobalObstacleMotionAction(ActionTerm):
    """驱动全局动态障碍物运动的自定义动作项。

    该动作项不消耗策略动作（``action_dim`` 为 0），只在每个物理步的
    ``apply_actions()`` 中推进一次全局障碍物。障碍物不随单个机器人环境
    重置，因此 ``reset()`` 为空操作。
    """

    cfg: GlobalObstacleMotionActionCfg
    """动作项配置。"""

    def __init__(self, cfg: GlobalObstacleMotionActionCfg, env: ManagerBasedEnv) -> None:
        # 初始化动作项，并从场景中解析 ``cfg.asset_name`` 对应的实体
        super().__init__(cfg, env)
        # 创建空的原始/处理动作缓冲（维度为 0）
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device, dtype=torch.float32)
        self._processed_actions = self._raw_actions

    @property
    def action_dim(self) -> int:
        """该动作项不消耗策略动作维度。"""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """原始动作缓冲，形状为 ``(num_envs, 0)``。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """处理后的动作缓冲，形状为 ``(num_envs, 0)``。"""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """零维动作无需处理。"""
        pass

    def apply_actions(self) -> None:
        """在每个物理步前推进一次全局障碍物。"""
        step_global_obstacles(self._env)

    def reset(self, env_ids=None) -> None:
        """全局障碍物不随单个机器人环境重置。"""
        pass


@configclass
class GlobalObstacleMotionActionCfg(ActionTermCfg):
    """驱动全局动态障碍物运动的动作项配置。"""

    class_type: type[ActionTerm] = GlobalObstacleMotionAction
    """关联的动作项类。"""

    asset_name: str = "dynamic_obstacles"
    """场景中注册的全局动态障碍物集合名称。"""
