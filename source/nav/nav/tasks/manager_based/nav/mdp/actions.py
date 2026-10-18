# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""无人机机体系速度动作项：转换到世界系后交给控制器。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from nav.controllers import VelocityController, VelocityControllerCfg


##
# 动作实现
##


class UavVelocityAction(ActionTerm):
    """接收三维机体系速度，每个策略步转换一次，再由控制器执行。

    输入为 ``(N, 3)`` 的实际速度 [前、左、上]（m/s），不负责 Beta 采样或
    归一化动作到速度的映射。转换使用 body 的完整姿态。
    ``apply_actions()`` 在各物理子步保持缓存的世界系速度指令，并调用
    :class:`VelocityController` 计算推力/力矩。
    """

    cfg: UavVelocityActionCfg
    """动作项配置。"""

    def __init__(self, cfg: UavVelocityActionCfg, env: ManagerBasedEnv) -> None:
        # 初始化动作项，并从场景中解析无人机实体
        super().__init__(cfg, env)
        # 与状态、目标观测使用相同的机身坐标系，不假定 articulation root 就是 body。
        body_ids, _ = self._asset.find_bodies(cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"Velocity actions require exactly one body matching {cfg.body_name!r}.")
        self._body_id = body_ids[0]
        # 创建速度控制器
        self._controller = VelocityController(
            robot=self._asset,
            cfg=cfg.velocity_controller_cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=env.physics_dt,
        )
        # 原始机体系速度 / 处理后的世界系速度，均为 (N, 3)、float32。
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        """三维机体系速度指令 ``[vx, vy, vz]``。"""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """原始机体系速度（m/s），形状 ``(N, 3)``。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """缓存的世界系速度（m/s），形状 ``(N, 3)``，直接提供给控制器。"""
        return self._processed_actions

    @property
    def controller(self) -> VelocityController:
        """底层速度控制器，供观测和调试访问。"""
        return self._controller

    def process_actions(self, actions: torch.Tensor) -> None:
        """按当前完整机身姿态，将输入速度从机体系转换为世界系。"""
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Expected body-frame velocity shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}."
            )
        self._raw_actions[:] = actions.to(device=self.device, dtype=torch.float32)
        # 默认 scale=1、max_velocity=None，保持输入速度的大小和方向。
        velocity_b = self._raw_actions * self.cfg.scale
        if self.cfg.max_velocity is not None:
            velocity_b = velocity_b.clamp(-self.cfg.max_velocity, self.cfg.max_velocity)
        quat_w = self._asset.data.body_link_quat_w[:, self._body_id]
        self._processed_actions[:] = quat_apply(quat_w, velocity_b)

    def apply_actions(self) -> None:
        """每个物理步发送缓存的世界系速度，不在子步内重新转换坐标。"""
        self._controller.apply_action(self._processed_actions)

    def reset(self, env_ids=None) -> None:
        """重置控制器状态并清空 wrench，避免残留推力。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._controller.reset_idx(env_ids)


##
# 动作配置
##


@configclass
class UavVelocityActionCfg(ActionTermCfg):
    """无人机机体系速度动作项配置，默认不缩放、不额外限速。"""

    class_type: type[ActionTerm] = UavVelocityAction
    """关联的动作项类。"""

    asset_name: str = "robot"
    """场景中注册的无人机名称。"""

    body_name: str = "body"
    """输入速度所使用的机身坐标系，应与观测中的 body 选择一致。"""

    velocity_controller_cfg: VelocityControllerCfg = VelocityControllerCfg()
    """底层速度控制器配置。"""

    scale: float = 1.0
    """可选机体系速度缩放系数；默认 1.0，输入已经是 m/s。"""

    max_velocity: float | None = None
    """可选机体系逐轴速度上限（m/s）；默认 None，不裁剪。"""
