# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务观测：全部转到 episode 内固定的 goal frame。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .dynamic import get_global_obstacle_manager, has_scene_entity
from .events import get_nav_task_buffer

__all__ = [
    "direction_obs",
    "dynamic_obstacle_obs",
    "lidar_obs",
    "state_obs",
    "vec_to_new_frame",
]


##
# 坐标工具
##


def vec_to_new_frame(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """把向量转到 goal frame（x 轴沿任务方向，z 轴保持世界垂直）。"""
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    z_direction = torch.tensor([0.0, 0.0, 1.0], device=vec.device)
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x, dim=-1)
    goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
    goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    n = vec.size(0)
    if vec.dim() == 3:
        vec_x = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1))
        vec_y = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    return torch.cat((vec_x, vec_y, vec_z), dim=-1)


##
# 观测项
##


def _goal_frame_direction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回归一化的 2D 任务方向（goal frame x 轴），形状 ``(num_envs, 3)``。"""
    target_dir = get_nav_task_buffer(env).target_dir.clone()
    target_dir_2d = target_dir
    target_dir_2d[:, 2] = 0.0
    target_dir_norm = torch.linalg.norm(target_dir_2d, dim=-1, keepdim=True)
    fallback_dir = torch.zeros_like(target_dir_2d)
    fallback_dir[:, 0] = 1.0
    return torch.where(target_dir_norm > 1.0e-6, target_dir_2d, fallback_dir)


def _lidar_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lidar_range: float) -> torch.Tensor:
    """量程内归一化的 LiDAR 读数，形状 ``(num_envs, 1, 36, 4)``，越近越大。"""
    lidar = env.scene[asset_cfg.name]
    ray_hits_w = lidar.data.ray_hits_w
    ray_starts_w = lidar.data.pos_w.unsqueeze(1)
    distance = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
    distance = torch.nan_to_num(distance, nan=lidar_range, posinf=lidar_range, neginf=lidar_range)
    lidar_obs = lidar_range - distance.clamp_max(lidar_range)
    return lidar_obs.reshape(env.num_envs, 1, 36, 4)


def state_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """8 维状态：目标方向(3) + 2D 距离(1) + z 距离(1) + 速度(3)，全部转 goal frame。"""
    robot = env.scene[asset_cfg.name]
    root_state = robot.data.root_state_w
    drone_pos_w = root_state[:, 0:3]
    drone_lin_vel_w = root_state[:, 7:10]

    buffer = get_nav_task_buffer(env)
    target_pos_w = buffer.target_pos
    target_dir_w = target_pos_w - drone_pos_w
    goal_direction = _goal_frame_direction(env)

    distance = torch.linalg.norm(target_dir_w, dim=-1, keepdim=True)
    distance_2d = torch.linalg.norm(target_dir_w[:, :2], dim=-1, keepdim=True)
    distance_z = target_dir_w[:, 2:3]
    target_dir_unit = target_dir_w / distance.clamp_min(1.0e-6)
    target_dir_goal = vec_to_new_frame(target_dir_unit, goal_direction).squeeze(1)
    drone_vel_goal = vec_to_new_frame(drone_lin_vel_w, goal_direction).squeeze(1)

    return torch.cat([target_dir_goal, distance_2d, distance_z, drone_vel_goal], dim=-1)


def lidar_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lidar_range: float = 4.0) -> torch.Tensor:
    """LiDAR 距离图，形状 ``(num_envs, 1, 36, 4)``。"""
    return _lidar_distance(env, asset_cfg, lidar_range)


def direction_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """固定任务方向，形状 ``(num_envs, 1, 3)``。"""
    return _goal_frame_direction(env).unsqueeze(1)


def _obstacle_size(env: ManagerBasedRLEnv) -> torch.Tensor:
    """从场景集合配置读取统一障碍物尺寸，形状 ``(num_obstacles, 3)``。"""
    collection_cfg = env.scene["dynamic_obstacles"].cfg
    first_spawn = next(iter(collection_cfg.rigid_objects.values())).spawn
    size = torch.tensor(first_spawn.size, device=env.device, dtype=torch.float32)
    num_obstacles = len(collection_cfg.rigid_objects)
    return size.unsqueeze(0).repeat(num_obstacles, 1)


def dynamic_obstacle_obs(
    env: ManagerBasedRLEnv,
    num_observed: int = 5,
    lidar_range: float = 4.0,
) -> torch.Tensor:
    """最近 ``num_observed`` 个动态障碍物的 10 维观测，形状 ``(num_envs, 1, 5, 10)``。"""
    drone_pos_w = env.scene["robot"].data.root_state_w[:, 0:3]
    goal_direction = _goal_frame_direction(env)

    dynamic_obstacle = torch.zeros(
        (env.num_envs, 1, num_observed, 10),
        dtype=torch.float32,
        device=env.device,
    )

    if not has_scene_entity(env, "dynamic_obstacles"):
        return dynamic_obstacle

    manager = get_global_obstacle_manager(env)
    obstacle_pos_w = manager.position_w[0]
    obstacle_vel_w = manager.linear_velocity_w[0]
    obstacle_size = _obstacle_size(env)
    num_obstacles = obstacle_pos_w.shape[0]
    num_observed = min(num_observed, num_obstacles)

    if num_observed > 0:
        rel_pos_w = obstacle_pos_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        distance_2d_all = torch.linalg.norm(rel_pos_w[:, :, :2], dim=-1)
        nearest_ids = torch.topk(distance_2d_all, k=num_observed, largest=False).indices
        range_mask = torch.gather(distance_2d_all, 1, nearest_ids) > lidar_range

        gather_ids = nearest_ids.unsqueeze(-1).expand(-1, -1, 3)
        rel_pos_w = torch.gather(rel_pos_w, 1, gather_ids)
        rel_pos_goal = vec_to_new_frame(rel_pos_w, goal_direction)
        rel_pos_goal[range_mask] = 0.0

        obstacle_vel_w = obstacle_vel_w[nearest_ids]
        obstacle_vel_w[range_mask] = 0.0
        obstacle_vel_goal = vec_to_new_frame(obstacle_vel_w, goal_direction)

        obstacle_size = obstacle_size.unsqueeze(0).expand(env.num_envs, -1, -1)
        obstacle_size = torch.gather(obstacle_size, 1, gather_ids)
        obstacle_width = obstacle_size[:, :, 0:1]
        obstacle_height = obstacle_size[:, :, 2:3]

        rel_distance = torch.linalg.norm(rel_pos_w, dim=-1, keepdim=True)
        rel_distance_2d = torch.linalg.norm(rel_pos_goal[:, :, :2], dim=-1, keepdim=True)
        rel_distance_z = rel_pos_goal[:, :, 2:3]
        rel_pos_goal_unit = rel_pos_goal / rel_distance.clamp_min(1.0e-6)

        width_category = obstacle_width / 0.25 - 1.0
        height_category = torch.where(obstacle_height > 1.0, torch.zeros_like(obstacle_height), obstacle_height)
        width_category[range_mask] = 0.0
        height_category[range_mask] = 0.0

        dynamic_obstacle[:, 0, :num_observed, :] = torch.cat(
            [
                rel_pos_goal_unit,
                rel_distance_2d,
                rel_distance_z,
                obstacle_vel_goal,
                width_category,
                height_category,
            ],
            dim=-1,
        )
    return dynamic_obstacle
