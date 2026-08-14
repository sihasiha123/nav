"""PPO 共享工具：观测转换、价值归一化、网络构建与坐标系变换。"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

__all__ = [
    "ValueNorm",
    "make_mlp",
    "obs_to_tensordict",
    "vec_to_world",
]


class ValueNorm(nn.Module):
    """价值回报的运行统计归一化（PopArt 风格）。"""

    def __init__(self, shape, beta=0.995, epsilon=1.0e-5):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = torch.Size(shape)
        self.beta = beta
        self.epsilon = epsilon

        self.register_buffer("running_mean", torch.zeros(self.shape))
        self.register_buffer("running_mean_sq", torch.zeros(self.shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp_min(self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp_min(self.epsilon)
        var = (mean_sq - mean.pow(2)).clamp_min(1.0e-2)
        return mean, var

    @torch.no_grad()
    def update(self, value):
        reduce_dims = tuple(range(value.dim() - len(self.shape)))
        batch_mean = value.mean(dim=reduce_dims)
        batch_mean_sq = value.pow(2).mean(dim=reduce_dims)
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, value):
        mean, var = self.running_mean_var()
        return (value - mean) / torch.sqrt(var)

    def denormalize(self, value):
        mean, var = self.running_mean_var()
        return value * torch.sqrt(var) + mean


def obs_to_tensordict(obs, num_envs, device):
    """把环境返回的观测 dict 包装成 TensorDict（兼容 ``{"policy": {...}}``）。"""
    if isinstance(obs, tuple):
        obs = obs[0]
    if "policy" in obs:
        obs = obs["policy"]
    obs = {
        "state": obs["state"].to(device),
        "lidar": obs["lidar"].to(device),
        "direction": obs["direction"].to(device),
        "dynamic_obstacle": obs["dynamic_obstacle"].to(device),
    }
    return TensorDict(
        {
            "agents": TensorDict(
                {"observation": obs},
                batch_size=[num_envs],
                device=device,
            )
        },
        batch_size=[num_envs],
        device=device,
    )


def make_mlp(hidden_dims, activation=nn.ELU):
    """构建 MLP（LazyLinear + 激活）。"""
    layers = []
    for dim in hidden_dims:
        layers.append(nn.LazyLinear(dim))
        layers.append(activation())
    return nn.Sequential(*layers)


def vec_to_world(vec, goal_direction):
    """把 goal frame 向量转回世界系。"""
    world_dir = torch.tensor([1.0, 0.0, 0.0], device=vec.device).expand_as(goal_direction)
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)
    return vec_to_new_frame(vec, world_frame_new)


def vec_to_new_frame(vec, goal_direction):
    """把向量转到 goal frame（x 轴沿任务方向，z 轴保持世界垂直）。"""
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    z_direction = torch.tensor([0.0, 0.0, 1.0], device=vec.device)
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x, dim=-1)
    goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
    goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    n = vec.size(0)
    if vec.dim() == 3:
        vec_x = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1))
        vec_y = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    return torch.cat((vec_x, vec_y, vec_z), dim=-1)
