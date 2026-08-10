# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""共享全局动态障碍物集合的运行时运动逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = [
    "GlobalObstacleManager",
    "GlobalObstacleMotionCfg",
    "get_global_obstacle_manager",
    "initialize_global_obstacles",
    "reset_global_obstacles",
    "step_global_obstacles",
]


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
        """推进一个物理步，并批量写入所有障碍物的位姿。"""
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

        # 场景配置使用运动学刚体，因此直接按脚本写入位姿。线速度仅保存在管理器中
        # 供观测使用，不通过只适用于非运动学刚体的 PhysX 速度接口写入。
        self.asset.write_object_link_pose_to_sim(self._pose_w)

    def reset_global(self) -> None:
        """将全部障碍物恢复到配置的运动原点，并重新采样航点。"""
        default_state = self.asset.data.default_object_state.clone()
        self.asset.write_object_link_pose_to_sim(default_state[..., :7])
        self.asset.reset()

        self._initialized = False
        self.initialize()

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


def reset_global_obstacles(env: ManagerBasedRLEnv) -> None:
    """显式重置全局集合，该操作与机器人的环境 ID 无关。"""
    get_global_obstacle_manager(env).reset_global()
