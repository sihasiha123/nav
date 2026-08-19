# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务的 reset 事件与任务状态缓冲。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "NavTaskBuffer",
    "get_nav_task_buffer",
    "reset_nav_task",
    "yaw_to_quat",
]


_TASK_BUFFER_ATTRIBUTE = "_nav_task_buffer"


##
# 任务状态缓冲
##


class NavTaskBuffer:
    """每个无人机 episode 的任务状态（目标、方向、历史量）。"""

    REWARD_COMPONENT_NAMES = (
        "progress",
        "goal_velocity",
        "static_avoidance",
        "dynamic_avoidance",
        "height",
        "smoothness",
        "time",
        "goal_first",
        "goal_reached",
        "collision",
        "out_of_bounds",
        "total",
    )

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        num_envs = env.num_envs
        device = env.device
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_dir = torch.zeros(num_envs, 3, device=device)
        self.height_range = torch.zeros(num_envs, 2, device=device)
        self.prev_distance = torch.zeros(num_envs, 1, device=device)
        self.prev_drone_vel_w = torch.zeros(num_envs, 3, device=device)
        self.reached_goal_once = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        self.reward_components = {
            name: torch.zeros(num_envs, 1, device=device)
            for name in self.REWARD_COMPONENT_NAMES
        }


def get_nav_task_buffer(env: ManagerBasedRLEnv) -> NavTaskBuffer:
    """返回环境的任务状态缓冲，并在第一次使用时创建它。"""
    buffer = getattr(env, _TASK_BUFFER_ATTRIBUTE, None)
    if buffer is None:
        buffer = NavTaskBuffer(env)
        setattr(env, _TASK_BUFFER_ATTRIBUTE, buffer)
    return buffer


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


def reset_nav_task(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    map_range: tuple[float, float, float] = (20.0, 20.0, 6.0),
    start_z_range: tuple[float, float] = (0.5, 2.5),
    boundary_offset: float = 2.0,
) -> None:
    """从 ``+Y`` 边均匀布置起点，目标固定在 ``-Y`` 边。

    X 坐标按环境全局编号均匀铺开，而不是按本次 reset 的子集重新采样。
    这样并行环境异步 reset 时仍保持整条边的均匀分布。起点和目标位于
    ``map_range + boundary_offset`` 的平地上，避免生成在静态障碍物内部。
    """
    robot = env.scene["robot"]
    buffer = get_nav_task_buffer(env)
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

    # 目标与起点保持相同 X 和高度，仅沿 Y 轴穿越到对侧。
    target_pos = start_pos.clone()
    target_pos[:, 1] = -y_bound

    target_dir = target_pos - start_pos
    yaw = torch.atan2(target_dir[:, 1], target_dir[:, 0])

    # 写入无人机初始物理状态
    root_pose = torch.cat([start_pos + env.scene.env_origins[env_ids], yaw_to_quat(yaw)], dim=-1)
    root_vel = torch.zeros((num_reset_envs, 6), device=env.device)
    robot.write_root_pose_to_sim(root_pose, env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # 任务状态
    buffer.target_pos[env_ids] = target_pos
    buffer.target_dir[env_ids] = target_dir
    buffer.prev_distance[env_ids] = torch.linalg.norm(target_dir, dim=-1, keepdim=True)
    buffer.prev_drone_vel_w[env_ids] = 0.0
    buffer.reached_goal_once[env_ids] = False
    buffer.height_range[env_ids, 0] = torch.minimum(start_pos[:, 2], target_pos[:, 2])
    buffer.height_range[env_ids, 1] = torch.maximum(start_pos[:, 2], target_pos[:, 2])
