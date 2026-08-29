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
    """Return the navigation target command ``[position, direction]``."""
    return env.command_manager.get_command("nav_target")


class NavTargetCommand(CommandTerm):
    """为每个环境生成并保存一个世界坐标系目标位置。"""

    cfg: "NavTargetCommandCfg"

    def __init__(self, cfg: NavTargetCommandCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self._command = torch.zeros((self.num_envs, 6), device=self.device)
        self._height_range = torch.zeros((self.num_envs, 2), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """目标位置和目标方向，形状为 ``(num_envs, 6)``。"""
        return self._command

    @property
    def target_pos_w(self) -> torch.Tensor:
        return self._command[:, :3]

    @property
    def target_dir_w(self) -> torch.Tensor:
        return self._command[:, 3:]

    @property
    def height_range(self) -> torch.Tensor:
        return self._height_range

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        start_pos_w = self.robot.data.root_pos_w[env_ids]
        target_pos_w = start_pos_w.clone()
        # 目标边界是命令自身的参数，避免依赖 reset 事件的地图采样配置。
        target_pos_w[:, 1] = self._env.scene.env_origins[env_ids, 1] + self.cfg.target_y
        target_dir_w = target_pos_w - start_pos_w

        self._command[env_ids, :3] = target_pos_w
        self._command[env_ids, 3:] = target_dir_w
        self._height_range[env_ids, 0] = start_pos_w[:, 2]
        self._height_range[env_ids, 1] = target_pos_w[:, 2]

    def _update_metrics(self) -> None:
        self.metrics["distance"] = torch.linalg.vector_norm(self.target_dir_w, dim=-1)

    def _update_command(self) -> None:
        return None


@configclass
class NavTargetCommandCfg(CommandTermCfg):
    """固定边界目标命令配置。

    ``target_y`` 使用每个环境原点为参考，属于目标命令自身的几何约束。
    """

    class_type: type[CommandTerm] = NavTargetCommand
    asset_name: str = "robot"
    target_y: float = -22.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

__all__ = ["NavTargetCommand", "NavTargetCommandCfg", "get_nav_target_command"]
