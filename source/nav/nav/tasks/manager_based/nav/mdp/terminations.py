# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务终止条件。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from nav.assets.dynamic import get_global_obstacle_manager, has_scene_entity
from .commands import get_nav_target_command

__all__ = [
    "dynamic_collision",
    "out_of_bounds",
    "static_collision",
    "success",
]


##
# 碰撞终止
##


def _lidar_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lidar_range: float) -> torch.Tensor:
    """读取终止判断所需的 LiDAR 距离。"""
    lidar = env.scene[asset_cfg.name]
    ray_starts_w = lidar.data.pos_w.unsqueeze(1)
    distance = torch.linalg.norm(lidar.data.ray_hits_w - ray_starts_w, dim=-1)
    distance = torch.nan_to_num(distance, nan=lidar_range, posinf=lidar_range, neginf=lidar_range)
    return distance.clamp_max(lidar_range)


def _obstacle_size(env: ManagerBasedRLEnv) -> torch.Tensor:
    """读取终止判断所需的动态障碍物尺寸。"""
    collection_cfg = env.scene["dynamic_obstacles"].cfg
    first_spawn = next(iter(collection_cfg.rigid_objects.values())).spawn
    size = torch.tensor(first_spawn.size, device=env.device, dtype=torch.float32)
    return size.unsqueeze(0).repeat(len(collection_cfg.rigid_objects), 1)


def static_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lidar_range: float = 4.0,
) -> torch.Tensor:
    """静态障碍碰撞：LiDAR 任一光束距离小于碰撞阈值。"""
    distance = _lidar_distance(env, asset_cfg, lidar_range)
    return distance.amin(dim=-1) < 0.3


def dynamic_collision(
    env: ManagerBasedRLEnv,
    margin: float = 0.3,
) -> torch.Tensor:
    """动态障碍碰撞：检查全部障碍物的几何距离（2D + 高度分别判断）。"""
    if not has_scene_entity(env, "dynamic_obstacles"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    drone_pos = env.scene["robot"].data.root_state_w[:, 0:3]
    manager = get_global_obstacle_manager(env)
    obstacle_pos_w = manager.position_w[0]
    obstacle_dimensions = _obstacle_size(env)

    rel_pos_w = obstacle_pos_w.unsqueeze(0) - drone_pos.unsqueeze(1)
    distance_2d = torch.linalg.norm(rel_pos_w[:, :, :2], dim=-1)
    distance_z = rel_pos_w[:, :, 2].abs()

    obstacle_width = obstacle_dimensions[:, 0:1]
    obstacle_height = obstacle_dimensions[:, 2:3]
    collision_2d = distance_2d <= obstacle_width.squeeze(-1) * 0.5 + margin
    collision_z = distance_z <= obstacle_height.squeeze(-1) * 0.5 + margin
    return (collision_2d & collision_z).any(dim=-1)


##
# 越界与成功
##


def out_of_bounds(
    env: ManagerBasedRLEnv,
    z_min: float = 0.2,
    z_max: float = 4.0,
) -> torch.Tensor:
    """飞行高度越界。"""
    drone_z = env.scene["robot"].data.root_state_w[:, 2]
    return (drone_z < z_min) | (drone_z > z_max)


def success(
    env: ManagerBasedRLEnv,
    goal_radius: float = 0.5,
) -> torch.Tensor:
    """安全到达目标（到达且未碰撞）。"""
    drone_pos = env.scene["robot"].data.root_state_w[:, 0:3]
    target_pos = get_nav_target_command(env)[:, :3]
    distance = torch.linalg.norm(target_pos - drone_pos, dim=-1)
    reach_goal = distance < goal_radius

    collision = static_collision(env, SceneEntityCfg("lidar")) | dynamic_collision(env)
    return reach_goal & ~collision
