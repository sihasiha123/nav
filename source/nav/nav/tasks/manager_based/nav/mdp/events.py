# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务的 reset 和随机化事件。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "reset_robot_state",
    "yaw_to_quat",
]


##
# 重置事件
##


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """将 yaw 角转为 (w, x, y, z) 四元数。"""
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    half_yaw = yaw * 0.5
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat


def reset_robot_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    map_range: tuple[float, float, float] = (20.0, 20.0, 6.0),
    start_z_range: tuple[float, float] = (0.5, 2.5),
    boundary_offset: float = 2.0,
    yaw_angle: float = -1.5707963267948966,
) -> None:
    """从 ``+Y`` 边均匀布置无人机起点。

    X 坐标按环境全局编号均匀铺开，而不是按本次 reset 的子集重新采样。
    这样并行环境异步 reset 时仍保持整条边的均匀分布。起点和目标位于
    ``map_range + boundary_offset`` 的平地上，避免生成在静态障碍物内部。
    """
    robot = env.scene["robot"]
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_reset_envs = env_ids.numel()
    x_range, y_range, z_range = map_range
    x_bound = x_range + boundary_offset
    y_bound = y_range + boundary_offset

    # 使用全局 env_id 计算 X，异步 reset 不会改变无人机在边界上的位置。
    env_id_float = env_ids.to(device=env.device, dtype=torch.float32)
    denominator = max(env.num_envs - 1, 1)
    x_fraction = env_id_float / float(denominator)
    start_pos = torch.empty((num_reset_envs, 3), device=env.device)
    start_pos[:, 0] = -x_bound + 2.0 * x_bound * x_fraction
    start_pos[:, 1] = y_bound
    z_min, z_max = start_z_range
    start_pos[:, 2] = z_min + (min(z_max, z_range) - z_min) * torch.rand(
        num_reset_envs, device=env.device
    )

    # 机器人朝向任务的固定行进方向；目标由 NavTargetCommand 生成。
    yaw = torch.full_like(start_pos[:, 0], yaw_angle)

    # 写入无人机初始物理状态
    root_pose = torch.cat([start_pos + env.scene.env_origins[env_ids], yaw_to_quat(yaw)], dim=-1)
    root_vel = torch.zeros((num_reset_envs, 6), device=env.device)
    robot.write_root_pose_to_sim(root_pose, env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
