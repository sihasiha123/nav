"""在线维护导航策略使用的多帧观测窗口。"""

from __future__ import annotations

import torch


class HistoryBuffer:
    """按环境独立保存最近观测，并按固定步长构造历史窗口。"""

    def __init__(self, num_envs, device, history_length=6, history_stride=6, action_limit=2.0):
        self.num_envs = num_envs
        self.device = device
        self.history_length = history_length
        self.history_stride = history_stride
        self.action_limit = float(action_limit)
        self.capacity = 1 + (history_length - 1) * history_stride
        self._state = None
        self._goal = None
        self._depth = None
        self._action = None
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _allocate(self, obs):
        self._state = torch.zeros(
            self.num_envs, self.capacity, *obs["state"].shape[1:], device=self.device
        )
        self._goal = torch.zeros(
            self.num_envs, self.capacity, *obs["goal"].shape[1:], device=self.device
        )
        self._depth = torch.zeros(
            self.num_envs, self.capacity, *obs["depth"].shape[1:], device=self.device
        )
        self._action = torch.zeros(self.num_envs, self.capacity, 3, device=self.device)

    def reset(self, mask=None):
        if mask is None:
            self._count.zero_()
        else:
            mask = mask.reshape(-1).bool()
            self._count[mask] = 0

    def append(self, obs, previous_action=None):
        if self._state is None:
            self._allocate(obs)
        if previous_action is None:
            previous_action = torch.zeros(self.num_envs, 3, device=self.device)
        previous_action = previous_action.to(self.device).float()
        if previous_action.abs().max() > 1.0 + 1.0e-5:
            previous_action = previous_action / self.action_limit

        self._state = torch.roll(self._state, shifts=-1, dims=1)
        self._goal = torch.roll(self._goal, shifts=-1, dims=1)
        self._depth = torch.roll(self._depth, shifts=-1, dims=1)
        self._action = torch.roll(self._action, shifts=-1, dims=1)
        self._state[:, -1] = obs["state"]
        self._goal[:, -1] = obs["goal"]
        self._depth[:, -1] = obs["depth"]
        self._action[:, -1] = previous_action
        self._count = (self._count + 1).clamp_max(self.capacity)

    def window(self):
        offsets = torch.arange(
            self.history_length - 1, -1, -1, device=self.device
        ) * self.history_stride
        positions = self.capacity - 1 - offsets
        state = self._state[:, positions]
        goal = self._goal[:, positions]
        depth = self._depth[:, positions]
        action = self._action[:, positions]

        age = offsets.to(torch.float32) / 60.0
        age = age.unsqueeze(0).expand(self.num_envs, -1)
        valid = self._count.unsqueeze(1) > offsets.unsqueeze(0)
        return {
            "depth_history": depth,
            "state_history": state,
            "action_history": action,
            "age_seconds": age,
            "valid_mask": valid,
            "current_goal": goal[:, -1],
        }
