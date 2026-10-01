# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务的目标命令。"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


def get_nav_target_command(env: ManagerBasedEnv) -> torch.Tensor:
    """返回世界系目标位置，float32，形状 ``(N, 3)``。"""
    return env.command_manager.get_command("nav_target")


class NavTargetCommand(CommandTerm):
    """为每个环境生成并保存一个世界坐标系目标位置。"""

    cfg: "NavTargetCommandCfg"

    def __init__(self, cfg: NavTargetCommandCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        body_ids, _ = self.robot.find_bodies(cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Navigation target requires exactly one body matching {cfg.body_name!r}.")
        self._body_id = body_ids[0]
        self._command = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._initial_distance = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self._height_range = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        """世界系目标位置 [x, y, z]，形状 ``(N, 3)``，不包含起始方向。"""
        return self._command

    @property
    def target_pos_w(self) -> torch.Tensor:
        return self._command

    @property
    def initial_distance(self) -> torch.Tensor:
        """生成目标时的三维距离，形状 ``(N, 1)``，回合内保持不变。"""
        return self._initial_distance

    @property
    def height_range(self) -> torch.Tensor:
        """起终点高度范围 ``(N, 2)``，供现有高度奖励读取。"""
        return self._height_range

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        # 官方 reset 流程先执行 Event，再重置 Command，因此这里读取新起点。
        # 与 goal_obs 使用同一个 body link 原点。
        start_pos_w = self.robot.data.body_link_pos_w[env_ids, self._body_id]
        target_pos_w = start_pos_w.clone()
        # 目标边界是命令自身的参数，避免依赖 reset 事件的地图采样配置。
        target_pos_w[:, 1] = self._env.scene.env_origins[env_ids, 1] + self.cfg.target_y
        self._command[env_ids] = target_pos_w
        self._initial_distance[env_ids] = torch.linalg.vector_norm(
            target_pos_w - start_pos_w, dim=-1, keepdim=True
        )
        self._height_range[env_ids, 0] = torch.minimum(start_pos_w[:, 2], target_pos_w[:, 2])
        self._height_range[env_ids, 1] = torch.maximum(start_pos_w[:, 2], target_pos_w[:, 2])
        self.metrics["distance"][env_ids] = self._initial_distance[env_ids, 0]

    def _update_metrics(self) -> None:
        current_pos_w = self.robot.data.body_link_pos_w[:, self._body_id]
        self.metrics["distance"][:] = torch.linalg.vector_norm(self.target_pos_w - current_pos_w, dim=-1)

    def _update_command(self) -> None:
        # 世界系终点在回合内固定；机体系相对目标由 goal_obs 每步计算。
        return None


@configclass
class NavTargetCommandCfg(CommandTermCfg):
    """固定边界目标命令配置。

    ``target_y`` 使用每个环境原点为参考，属于目标命令自身的几何约束。
    """

    class_type: type[CommandTerm] = NavTargetCommand
    asset_name: str = "robot"
    body_name: str = "body"
    target_y: float = -22.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """远大于当前回合时长，回合重置时由 CommandManager 重新采样。"""

__all__ = ["NavTargetCommand", "NavTargetCommandCfg", "get_nav_target_command"]
