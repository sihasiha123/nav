# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务奖励（移植 uav 权重）。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from nav.assets.dynamic import get_global_obstacle_manager, has_scene_entity
from .commands import get_nav_target_command

__all__ = ["NavigationReward"]


##
# 导航奖励
##


def _lidar_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lidar_range: float) -> torch.Tensor:
    """读取奖励计算所需的 LiDAR 距离。"""
    lidar = env.scene[asset_cfg.name]
    ray_starts_w = lidar.data.pos_w.unsqueeze(1)
    distance = torch.linalg.norm(lidar.data.ray_hits_w - ray_starts_w, dim=-1)
    distance = torch.nan_to_num(distance, nan=lidar_range, posinf=lidar_range, neginf=lidar_range)
    return distance.clamp_max(lidar_range)


def _obstacle_size(env: ManagerBasedRLEnv) -> torch.Tensor:
    """读取奖励计算所需的动态障碍物尺寸。"""
    collection_cfg = env.scene["dynamic_obstacles"].cfg
    first_spawn = next(iter(collection_cfg.rigid_objects.values())).spawn
    size = torch.tensor(first_spawn.size, device=env.device, dtype=torch.float32)
    return size.unsqueeze(0).repeat(len(collection_cfg.rigid_objects), 1)


class NavigationReward(ManagerTermBase):
    """整体导航奖励，并拥有自身需要的跨 step 历史状态。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.prev_distance = torch.zeros((env.num_envs, 1), device=env.device)
        self.prev_drone_vel_w = torch.zeros((env.num_envs, 3), device=env.device)
        self.reached_goal_once = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
        # 每个环境的标志保持为二维形状，与奖励张量的形状一致。
        # 如果使用一维标志，与 (N, 1) 的距离缓存执行 torch.where 时，
        # 会发生广播并错误地产生 (N, N) 的结果。
        self._initialized = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
        self.reward_components = {}
        self._component_names = (
            "progress", "goal_velocity", "static_avoidance", "dynamic_avoidance",
            "height", "smoothness", "time", "goal_first", "goal_reached",
            "collision", "out_of_bounds", "total",
        )
        self.reward_components = {
            name: torch.zeros((env.num_envs, 1), device=env.device) for name in self._component_names
        }

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.prev_distance[env_ids] = 0.0
        self.prev_drone_vel_w[env_ids] = 0.0
        self.reached_goal_once[env_ids] = False
        self._initialized[env_ids] = False
        for value in self.reward_components.values():
            value[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lidar_range: float = 4.0,
        static_safe_distance: float = 1.2,
        dynamic_safe_distance: float = 1.5,
        goal_radius: float = 0.5,
        z_min: float = 0.2,
        z_max: float = 4.0,
        out_of_bounds_penalty: float = 120.0,
    ) -> torch.Tensor:
        """导航奖励：进展、速度、静态/动态避障、高度、平滑、时间、到达与碰撞。"""
        robot = env.scene["robot"]
        root_state = robot.data.root_state_w
        drone_pos_w = root_state[:, 0:3]
        drone_vel_w = root_state[:, 7:10]
        drone_z = root_state[:, 2:3]

        # 目标进展与朝目标速度
        target_command = get_nav_target_command(env)
        target_pos_w = target_command[:, :3]
        target_dir_w = target_pos_w - drone_pos_w
        distance = torch.linalg.norm(target_dir_w, dim=-1, keepdim=True)
        vel_direction = target_dir_w / distance.clamp_min(1.0e-6)
        initial_distance = torch.linalg.norm(target_command[:, 3:], dim=-1, keepdim=True)
        previous_distance = torch.where(self._initialized, self.prev_distance, initial_distance)
        reward_progress = previous_distance - distance
        reward_vel = (drone_vel_w * vel_direction).sum(dim=-1, keepdim=True)

        # 静态障碍（LiDAR 推断）
        lidar_distance_w = _lidar_distance(env, SceneEntityCfg("lidar"), lidar_range)
        static_clearance = lidar_distance_w.amin(dim=-1, keepdim=True)
        penalty_static = torch.relu(static_safe_distance - static_clearance).pow(2)
        static_collision = lidar_distance_w.amin(dim=-1, keepdim=True) < 0.3

        # 动态障碍（最近 5 个）
        dynamic_collision = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
        penalty_dynamic = torch.zeros(env.num_envs, 1, device=env.device)
        if has_scene_entity(env, "dynamic_obstacles"):
            manager = get_global_obstacle_manager(env)
            if manager.enabled:
                obstacle_pos_w = manager.active_position_w[0]
                obstacle_dimensions = _obstacle_size(env).index_select(0, manager.active_indices)
                num_obstacles = obstacle_pos_w.shape[0]
                num_observed = min(5, num_obstacles)
                if num_observed > 0:
                    rel_pos_w = obstacle_pos_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
                    distance_2d_all = torch.linalg.norm(rel_pos_w[:, :, :2], dim=-1)
                    nearest_ids = torch.topk(distance_2d_all, k=num_observed, largest=False).indices
                    range_mask = torch.gather(distance_2d_all, 1, nearest_ids) > lidar_range

                    gather_ids = nearest_ids.unsqueeze(-1).expand(-1, -1, 3)
                    rel_pos_w = torch.gather(rel_pos_w, 1, gather_ids)
                    obstacle_dimensions = obstacle_dimensions.unsqueeze(0).expand(env.num_envs, -1, -1)
                    obstacle_dimensions = torch.gather(obstacle_dimensions, 1, gather_ids)
                    obstacle_width = obstacle_dimensions[:, :, 0:1]
                    obstacle_height = obstacle_dimensions[:, :, 2:3]

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
        height_range = env.command_manager.get_term("nav_target").height_range
        height_min = height_range[:, 0:1]
        height_max = height_range[:, 1:2]
        penalty_height = torch.zeros(env.num_envs, 1, device=env.device)
        above_height = drone_z > height_max + 0.2
        below_height = drone_z < height_min - 0.2
        penalty_height[above_height] = (drone_z - height_max - 0.2)[above_height].pow(2)
        penalty_height[below_height] = (height_min - 0.2 - drone_z)[below_height].pow(2)

        # 平滑
        penalty_smooth = torch.linalg.norm(drone_vel_w - self.prev_drone_vel_w, dim=-1, keepdim=True)

        collision = static_collision | dynamic_collision
        out_of_bounds = (drone_z < z_min) | (drone_z > z_max)
        reach_goal = distance < goal_radius
        safe_reach_goal = reach_goal & ~collision
        first_reach_goal = safe_reach_goal & ~self.reached_goal_once

        progress_term = 4.0 * reward_progress
        goal_velocity_term = 0.5 * reward_vel.clamp(min=-2.0, max=2.0)
        static_avoidance_term = -6.0 * penalty_static
        dynamic_avoidance_term = -10.0 * penalty_dynamic
        height_term = -2.0 * penalty_height
        smoothness_term = -0.05 * penalty_smooth
        time_term = torch.full_like(reward_progress, -0.01)
        goal_first_term = torch.zeros_like(reward_progress)
        goal_reached_term = torch.zeros_like(reward_progress)
        collision_term = torch.zeros_like(reward_progress)
        out_of_bounds_term = torch.zeros_like(reward_progress)
        goal_first_term[first_reach_goal] = 120.0
        goal_reached_term[safe_reach_goal] = 0.5
        collision_term[collision] = -120.0
        out_of_bounds_term[out_of_bounds] = -out_of_bounds_penalty

        reward = (
            progress_term
            + goal_velocity_term
            + static_avoidance_term
            + dynamic_avoidance_term
            + height_term
            + smoothness_term
            + time_term
            + goal_first_term
            + goal_reached_term
            + collision_term
            + out_of_bounds_term
        )

        # Isaac Lab 的 RewardManager 会按 step dt 缩放 reward term。
        # 分项日志也乘同样的尺度，方便直接和 env.step() 返回的 reward 对齐。
        reward_scale = getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation)
        self.reward_components["progress"][:] = progress_term.detach() * reward_scale
        self.reward_components["goal_velocity"][:] = goal_velocity_term.detach() * reward_scale
        self.reward_components["static_avoidance"][:] = static_avoidance_term.detach() * reward_scale
        self.reward_components["dynamic_avoidance"][:] = dynamic_avoidance_term.detach() * reward_scale
        self.reward_components["height"][:] = height_term.detach() * reward_scale
        self.reward_components["smoothness"][:] = smoothness_term.detach() * reward_scale
        self.reward_components["time"][:] = time_term.detach() * reward_scale
        self.reward_components["goal_first"][:] = goal_first_term.detach() * reward_scale
        self.reward_components["goal_reached"][:] = goal_reached_term.detach() * reward_scale
        self.reward_components["collision"][:] = collision_term.detach() * reward_scale
        self.reward_components["out_of_bounds"][:] = out_of_bounds_term.detach() * reward_scale
        self.reward_components["total"][:] = reward.detach() * reward_scale

        # 更新历史状态
        self.prev_drone_vel_w[:] = drone_vel_w.detach()
        self.prev_distance[:] = distance.detach()
        self.reached_goal_once |= safe_reach_goal
        self._initialized[:] = True

        return reward.squeeze(-1)
