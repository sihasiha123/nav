# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局动态障碍物的动作项：在每个物理步前推进障碍物运动。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .dynamic import step_global_obstacles


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
