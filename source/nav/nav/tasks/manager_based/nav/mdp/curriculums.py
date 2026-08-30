# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务的自适应课程项。"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

from nav.assets.dynamic import get_global_obstacle_manager, has_scene_entity

__all__ = ["SuccessRateDynamicObstacleCurriculum"]


class SuccessRateDynamicObstacleCurriculum(ManagerTermBase):
    """根据最近一段已完成 episode 的成功率推进动态障碍物难度。

    课程状态属于该 Term，自身维护统计窗口；环境只提供官方的终止管理器和
    全局训练步数。初始化 reset 不计入统计，避免把空的终止结果当作失败 episode。
    """

    cfg: "SuccessRateDynamicObstacleCurriculumCfg"

    def __init__(self, cfg: SuccessRateDynamicObstacleCurriculumCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.stage = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.last_success_rate = 0.0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """保留全局课程统计；episode reset 不应清空课程窗口。"""
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        window_size: int = 1024,
        success_thresholds: tuple[float, float] = (0.8, 0.7),
        transition_extent: tuple[float, float, float] = (0.5, 0.5, 0.2),
        full_extent: tuple[float, float, float] = (1.0, 1.0, 0.4),
        transition_speed: tuple[float, float] = (0.125, 0.375),
        full_speed: tuple[float, float] = (0.25, 0.75),
    ) -> dict[str, float]:
        """在窗口满足成功率要求时最多推进一个课程阶段。"""
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if len(success_thresholds) != 2 or any(not 0.0 <= value <= 1.0 for value in success_thresholds):
            raise ValueError("success_thresholds must contain two values in [0, 1].")

        # 初始 env.reset() 也会触发 CurriculumManager，但此时没有完成 episode。
        if isinstance(env_ids, slice):
            selected_ids = slice(None)
            has_selected_envs = env.num_envs > 0
        else:
            selected_ids = env_ids
            has_selected_envs = len(env_ids) > 0
        if int(env.common_step_counter) > 0 and has_selected_envs:
            success = env.termination_manager.get_term("success")[selected_ids].reshape(-1).bool()
            self.completed_episodes += success.numel()
            self.successful_episodes += int(success.sum().item())

        if self.completed_episodes >= window_size:
            self.last_success_rate = self.successful_episodes / float(self.completed_episodes)
            if self.stage < 2 and self.last_success_rate >= success_thresholds[self.stage]:
                self.stage += 1
            self.completed_episodes = 0
            self.successful_episodes = 0

        if has_scene_entity(env, "dynamic_obstacles"):
            manager = get_global_obstacle_manager(env)
            if self.stage == 0:
                manager.set_difficulty(False, (0.0, 0.0, 0.0), full_speed)
            elif self.stage == 1:
                manager.set_difficulty(True, transition_extent, transition_speed)
            else:
                manager.set_difficulty(True, full_extent, full_speed)

        return {
            "stage": float(self.stage),
            "window_success_rate": float(self.last_success_rate),
            "window_completed": float(self.completed_episodes),
        }
