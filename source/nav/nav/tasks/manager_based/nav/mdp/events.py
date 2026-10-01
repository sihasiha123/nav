# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务的 reset 和随机化事件。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from nav.assets.dynamic import get_global_obstacle_manager, has_scene_entity

__all__ = [
    "initialize_dynamic_obstacles",
    "reset_robot_state",
    "yaw_to_quat",
]


##
# 重置事件
##


def initialize_dynamic_obstacles(env: ManagerBasedRLEnv, env_ids=None) -> None:
    """启动时创建共享障碍物运动管理器；不依赖策略观测，也不随回合重置。"""
    if has_scene_entity(env, "dynamic_obstacles"):
        get_global_obstacle_manager(env).initialize()


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """将 yaw 角转为 (w, x, y, z) 四元数。"""
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    half_yaw = yaw * 0.5
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat


def reset_robot_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    start_x_range: tuple[float, float] = (-22.0, 22.0),
    start_y: float = 22.0,
    start_z_range: tuple[float, float] = (0.5, 2.5),
    yaw_angle: float = -1.5707963267948966,
) -> None:
    """从 ``+Y`` 边均匀布置无人机起点。

    X 坐标按环境全局编号均匀铺开，而不是按本次 reset 的子集重新采样。
    这样并行环境异步 reset 时仍保持整条边的均匀分布。位置参数相对各自
    env_origin；只有高度随机。该事件只写入初始物理状态，不生成目标。
    """
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    num_reset_envs = env_ids.numel()
    if num_reset_envs == 0:
        return
    x_min, x_max = start_x_range
    z_min, z_max = start_z_range
    if not all(math.isfinite(value) for value in (x_min, x_max, start_y, z_min, z_max, yaw_angle)):
        raise ValueError("Reset position ranges and yaw must be finite.")
    if x_min > x_max or z_min > z_max:
        raise ValueError("Reset position ranges must be ordered from minimum to maximum.")

    # 使用全局 env_id 计算 X，异步 reset 不会改变无人机在边界上的位置。
    env_id_float = env_ids.to(device=env.device, dtype=torch.float32)
    x_fraction = env_id_float / float(env.num_envs - 1) if env.num_envs > 1 else torch.full_like(env_id_float, 0.5)
    start_pos = torch.empty((num_reset_envs, 3), device=env.device, dtype=torch.float32)
    start_pos[:, 0] = x_min + (x_max - x_min) * x_fraction
    start_pos[:, 1] = start_y
    start_pos[:, 2] = z_min + (z_max - z_min) * torch.rand(
        num_reset_envs, device=env.device
    )

    # 机器人朝向任务的固定行进方向；目标由 NavTargetCommand 生成。
    yaw = torch.full_like(start_pos[:, 0], yaw_angle)

    # 写入无人机初始物理状态
    root_pose = torch.cat([start_pos + env.scene.env_origins[env_ids], yaw_to_quat(yaw)], dim=-1)
    root_vel = torch.zeros((num_reset_envs, 6), device=env.device, dtype=torch.float32)
    robot.write_root_pose_to_sim(root_pose, env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
