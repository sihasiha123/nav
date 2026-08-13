# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务奖励（移植 uav 权重）。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .dynamic import get_global_obstacle_manager, has_scene_entity
from .events import get_nav_task_buffer
from .observations import _lidar_distance, _obstacle_size

__all__ = ["navigation_reward"]


##
# 导航奖励
##


def navigation_reward(
    env: ManagerBasedRLEnv,
    lidar_range: float = 4.0,
    static_safe_distance: float = 1.2,
    dynamic_safe_distance: float = 1.5,
    goal_radius: float = 0.5,
) -> torch.Tensor:
    """导航奖励：进展、速度、静态/动态避障、高度、平滑、时间、到达与碰撞。"""
    robot = env.scene["robot"]
    buffer = get_nav_task_buffer(env)
    root_state = robot.data.root_state_w
    drone_pos_w = root_state[:, 0:3]
    drone_vel_w = root_state[:, 7:10]
    drone_z = root_state[:, 2:3]

    # 目标进展与朝目标速度
    target_pos_w = buffer.target_pos
    target_dir_w = target_pos_w - drone_pos_w
    distance = torch.linalg.norm(target_dir_w, dim=-1, keepdim=True)
    vel_direction = target_dir_w / distance.clamp_min(1.0e-6)
    reward_progress = buffer.prev_distance - distance
    reward_vel = (drone_vel_w * vel_direction).sum(dim=-1, keepdim=True)

    # 静态障碍（LiDAR 推断）
    lidar = _lidar_distance(env, SceneEntityCfg("lidar"), lidar_range)
    static_clearance = (lidar_range - lidar).amin(dim=(2, 3))
    penalty_static = torch.relu(static_safe_distance - static_clearance).pow(2)
    static_collision = lidar.amax(dim=(2, 3)) > lidar_range - 0.3

    # 动态障碍（最近 5 个）
    dynamic_collision = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
    penalty_dynamic = torch.zeros(env.num_envs, 1, device=env.device)
    if has_scene_entity(env, "dynamic_obstacles"):
        manager = get_global_obstacle_manager(env)
        obstacle_pos_w = manager.position_w[0]
        obstacle_size = _obstacle_size(env)
        num_obstacles = obstacle_pos_w.shape[0]
        num_observed = min(5, num_obstacles)
        if num_observed > 0:
            rel_pos_w = obstacle_pos_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
            distance_2d_all = torch.linalg.norm(rel_pos_w[:, :, :2], dim=-1)
            nearest_ids = torch.topk(distance_2d_all, k=num_observed, largest=False).indices
            range_mask = torch.gather(distance_2d_all, 1, nearest_ids) > lidar_range

            gather_ids = nearest_ids.unsqueeze(-1).expand(-1, -1, 3)
            rel_pos_w = torch.gather(rel_pos_w, 1, gather_ids)
            obstacle_size = obstacle_size.unsqueeze(0).expand(env.num_envs, -1, -1)
            obstacle_size = torch.gather(obstacle_size, 1, gather_ids)
            obstacle_width = obstacle_size[:, :, 0:1]
            obstacle_height = obstacle_size[:, :, 2:3]

            distance_2d = torch.linalg.norm(rel_pos_w[:, :, :2], dim=-1, keepdim=True)
            distance_z = rel_pos_w[:, :, 2:3].abs()
            distance_2d[range_mask] = float("inf")
            distance_z[range_mask] = float("inf")
            collision_2d = distance_2d <= obstacle_width * 0.5 + 0.3
            collision_z = distance_z <= obstacle_height * 0.5 + 0.3
            dynamic_collision = (collision_2d & collision_z).any(dim=1)

            dynamic_clearance = torch.linalg.norm(rel_pos_w, dim=-1) - obstacle_width.squeeze(-1) * 0.5
            dynamic_clearance[range_mask] = lidar_range
            dynamic_clearance = dynamic_clearance.clamp(min=0.0, max=lidar_range)
            penalty_dynamic = torch.relu(dynamic_safe_distance - dynamic_clearance).pow(2).mean(dim=-1, keepdim=True)

    # 高度范围
    height_min = buffer.height_range[:, 0:1]
    height_max = buffer.height_range[:, 1:2]
    penalty_height = torch.zeros(env.num_envs, 1, device=env.device)
    above_height = drone_z > height_max + 0.2
    below_height = drone_z < height_min - 0.2
    penalty_height[above_height] = (drone_z - height_max - 0.2)[above_height].pow(2)
    penalty_height[below_height] = (height_min - 0.2 - drone_z)[below_height].pow(2)

    # 平滑
    penalty_smooth = torch.linalg.norm(drone_vel_w - buffer.prev_drone_vel_w, dim=-1, keepdim=True)

    collision = static_collision | dynamic_collision
    reach_goal = distance < goal_radius
    safe_reach_goal = reach_goal & ~collision
    first_reach_goal = safe_reach_goal & ~buffer.reached_goal_once

    reward = (
        4.0 * reward_progress
        + 0.5 * reward_vel.clamp(min=-2.0, max=2.0)
        - 6.0 * penalty_static
        - 10.0 * penalty_dynamic
        - 2.0 * penalty_height
        - 0.05 * penalty_smooth
        - 0.01
    )
    reward[first_reach_goal] += 50.0
    reward[safe_reach_goal] += 0.5
    reward[collision] -= 120.0

    # 更新历史状态
    buffer.prev_drone_vel_w[:] = drone_vel_w.detach()
    buffer.prev_distance[:] = distance.detach()
    buffer.reached_goal_once |= safe_reach_goal

    return reward.squeeze(-1)
