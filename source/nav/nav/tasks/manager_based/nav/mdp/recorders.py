# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""导航任务 RecorderTerm。

Recorder 只读取已经由各 Manager 计算出的结果并交给官方
``RecorderManager``，不参与奖励、终止或 reset 逻辑。
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

__all__ = ["NavigationRewardRecorder", "NavigationRewardRecorderCfg"]


class NavigationRewardRecorder(RecorderTerm):
    """记录导航奖励项产生的公开分项数据。"""

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """在奖励计算完成后记录本 step 的奖励分项。"""
        try:
            reward_cfg = self._env.reward_manager.get_term_cfg("navigation")
        except (AttributeError, ValueError):
            return None, None

        reward_term = getattr(reward_cfg, "func", None)
        components = getattr(reward_term, "reward_components", None)
        if not components:
            return None, None

        # RecorderManager 接受嵌套字典，并负责按 episode 聚合和导出。
        return "reward_components", {
            name: value.detach().clone() for name, value in components.items()
        }


@configclass
class NavigationRewardRecorderCfg(RecorderTermCfg):
    """导航奖励分项 Recorder 配置。"""

    class_type: type[RecorderTerm] = NavigationRewardRecorder
