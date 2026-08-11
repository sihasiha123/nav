# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""无人机世界系速度动作项。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from nav.controllers import VelocityController, VelocityControllerCfg


class UavVelocityAction(ActionTerm):
    """将三维世界系速度指令映射到无人机控制器。

    策略输出 ``[vx, vy, vz]``（m/s），``apply_actions()`` 在每个物理步调用
    :class:`VelocityController` 计算推力/力矩并写入无人机的 permanent wrench。
    """

    cfg: UavVelocityActionCfg
    """动作项配置。"""

    def __init__(self, cfg: UavVelocityActionCfg, env: ManagerBasedEnv) -> None:
        # 初始化动作项，并从场景中解析无人机实体
        super().__init__(cfg, env)
        # 创建速度控制器
        self._controller = VelocityController(
            robot=self._asset,
            cfg=cfg.velocity_controller_cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=env.physics_dt,
        )
        # 原始/处理后的速度指令缓冲
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        """三维速度指令 ``[vx, vy, vz]``。"""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """未经缩放/裁剪的原始速度指令，形状为 ``(num_envs, 3)``。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """缩放/裁剪后的速度指令，形状为 ``(num_envs, 3)``。"""
        return self._processed_actions

    @property
    def controller(self) -> VelocityController:
        """底层速度控制器，供观测和调试访问。"""
        return self._controller

    def process_actions(self, actions: torch.Tensor) -> None:
        """缩放并裁剪速度指令。"""
        actions = actions.to(self.device)
        self._raw_actions[:] = actions
        processed = actions * self.cfg.scale
        if self.cfg.max_velocity is not None:
            processed = processed.clamp(-self.cfg.max_velocity, self.cfg.max_velocity)
        self._processed_actions[:] = processed

    def apply_actions(self) -> None:
        """每个物理步将速度指令送入控制器并写入 wrench。"""
        self._controller.apply_action(self._processed_actions)

    def reset(self, env_ids=None) -> None:
        """重置控制器状态并清空 wrench，避免残留推力。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._controller.reset_idx(env_ids)


@configclass
class UavVelocityActionCfg(ActionTermCfg):
    """无人机世界系速度动作项配置。"""

    class_type: type[ActionTerm] = UavVelocityAction
    """关联的动作项类。"""

    asset_name: str = "robot"
    """场景中注册的无人机名称。"""

    velocity_controller_cfg: VelocityControllerCfg = VelocityControllerCfg()
    """底层速度控制器配置。"""

    scale: float = 1.0
    """速度指令缩放系数。"""

    max_velocity: float | None = None
    """速度指令裁剪上限（m/s）；为 None 时不裁剪。"""
