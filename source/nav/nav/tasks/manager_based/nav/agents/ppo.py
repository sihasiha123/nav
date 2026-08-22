# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""自定义 PPO：Beta 动作分布、lidar CNN + dynamic MLP 特征提取、ValueNorm。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ValueNorm, vec_to_world

__all__ = [
    "PPO",
    "PPOActor",
    "PPOCritic",
    "PPOFeatureExtractor",
    "compute_gae",
    "make_minibatches",
]


class PPOFeatureExtractor(nn.Module):
    """lidar CNN + dynamic obstacle MLP + state 拼接融合。"""

    def __init__(self):
        super().__init__()

        self.lidar_encoder = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=(5, 3), padding=(2, 1)),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        )

        self.dynamic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

        self.feature_fusion = nn.Sequential(
            nn.LazyLinear(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )

    def forward(self, obs):
        state = obs["state"]
        lidar = obs["lidar"]
        dynamic_obstacle = obs["dynamic_obstacle"]

        lidar_feature = self.lidar_encoder(lidar)
        dynamic_feature = self.dynamic_encoder(dynamic_obstacle)

        feature = torch.cat([lidar_feature, state, dynamic_feature], dim=-1)
        return self.feature_fusion(feature)


class PPOActor(nn.Module):
    """Beta 分布动作头，输出归一化动作 [0,1]。"""

    def __init__(self, action_dim):
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.softplus = nn.Softplus()

    def forward(self, feature):
        alpha = 1.0 + self.softplus(self.alpha_layer(feature)) + 1.0e-6
        beta = 1.0 + self.softplus(self.beta_layer(feature)) + 1.0e-6
        return alpha, beta

    def distribution(self, feature):
        alpha, beta = self.forward(feature)
        dist = torch.distributions.Beta(alpha, beta)
        return torch.distributions.Independent(dist, 1)

    def sample(self, feature):
        dist = self.distribution(feature)
        action_normalized = dist.sample()
        log_prob = dist.log_prob(action_normalized)
        return action_normalized, log_prob

    def evaluate_action(self, feature, action_normalized):
        dist = self.distribution(feature)
        log_prob = dist.log_prob(action_normalized)
        entropy = dist.entropy()
        return log_prob, entropy

    def deterministic(self, feature):
        alpha, beta = self.forward(feature)
        return alpha / (alpha + beta)


class PPOCritic(nn.Module):
    """价值网络。"""

    def __init__(self):
        super().__init__()
        self.value_layer = nn.LazyLinear(1)

    def forward(self, feature):
        return self.value_layer(feature)


def compute_gae(reward, done, value, next_value, gamma, lam):
    """GAE 优势估计，从后往前累积。"""
    advantage = torch.zeros_like(reward)
    not_done = 1.0 - done.float()
    gae = 0.0

    for step in reversed(range(reward.shape[1])):
        delta = reward[:, step] + gamma * next_value[:, step] * not_done[:, step] - value[:, step]
        gae = delta + gamma * lam * not_done[:, step] * gae
        advantage[:, step] = gae

    returns = advantage + value
    return advantage, returns


def make_minibatches(rollout, num_minibatches):
    """把 rollout 打乱切成 minibatch。"""
    rollout = rollout.reshape(-1)

    batch_size = rollout.shape[0]
    usable_size = batch_size // num_minibatches * num_minibatches
    minibatch_size = usable_size // num_minibatches
    indices = torch.randperm(usable_size, device=rollout.device)

    for start in range(0, usable_size, minibatch_size):
        batch_indices = indices[start : start + minibatch_size]
        yield rollout[batch_indices]


class PPO(nn.Module):
    """PPO 算法：act 采样、update 用 GAE + clip 更新。"""

    def __init__(self, cfg, observation_space, action_space, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.action_dim = int(action_space)
        self.action_limit = cfg.actor.action_limit

        self.feature_extractor = PPOFeatureExtractor().to(device)
        self.actor = PPOActor(self.action_dim).to(device)
        self.critic = PPOCritic().to(device)
        self.value_norm = ValueNorm(1).to(device)

        self._initialize_lazy_modules(observation_space)

        self.feature_optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=cfg.feature_extractor.learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic.learning_rate,
        )

    def act(self, tensordict):
        """推理：采样动作 + log_prob + value，写回 tensordict。"""
        with torch.no_grad():
            obs = tensordict["agents", "observation"]
            feature = self.feature_extractor(obs)

            action_normalized, log_prob = self.actor.sample(feature)
            state_value = self.critic(feature)
            action = self._to_world_action(action_normalized, obs["direction"])

            tensordict["agents", "action_normalized"] = action_normalized
            tensordict["agents", "action"] = action
            tensordict["sample_log_prob"] = log_prob
            tensordict["state_value"] = state_value

        return tensordict

    @torch.no_grad()
    def act_deterministic(self, tensordict):
        """评估：使用 Beta 分布均值生成确定性动作。"""
        obs = tensordict["agents", "observation"]
        feature = self.feature_extractor(obs)
        action_normalized = self.actor.deterministic(feature)
        action = self._to_world_action(action_normalized, obs["direction"])

        tensordict["agents", "action_normalized"] = action_normalized
        tensordict["agents", "action"] = action
        return tensordict

    def evaluate_action(self, obs, action_normalized):
        feature = self.feature_extractor(obs)
        log_prob, entropy = self.actor.evaluate_action(feature, action_normalized)
        state_value = self.critic(feature)
        return log_prob, entropy, state_value

    def update(self, rollout):
        """用 rollout 更新网络，返回 loss 指标。"""
        with torch.no_grad():
            next_obs = rollout["next", "agents", "observation"]
            next_obs = self._flatten_observation(next_obs)
            next_feature = self.feature_extractor(next_obs)
            next_value = self.critic(next_feature).view_as(rollout["state_value"])

        reward = rollout["next", "agents", "reward"]
        # 成功、碰撞、越界和超时都会结束回合；用组合 done 防止跨回合自举。
        done = rollout["next", "done"]
        value = rollout["state_value"]
        value = self.value_norm.denormalize(value)
        next_value = self.value_norm.denormalize(next_value)

        advantage, returns = compute_gae(
            reward,
            done,
            value,
            next_value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.0e-6)
        self.value_norm.update(returns)
        returns = self.value_norm.normalize(returns)

        rollout["advantage"] = advantage
        rollout["return"] = returns

        infos = []
        for _ in range(self.cfg.training_epoch_num):
            for minibatch in make_minibatches(rollout, self.cfg.num_minibatches):
                infos.append(self._update_minibatch(minibatch))

        return {
            key: torch.stack([info[key] for info in infos]).mean().item()
            for key in infos[0]
        }

    def _update_minibatch(self, batch):
        obs = batch["agents", "observation"]
        action_normalized = batch["agents", "action_normalized"]
        old_log_prob = batch["sample_log_prob"]
        old_value = batch["state_value"].detach()
        advantage = batch["advantage"]
        returns = batch["return"]

        new_log_prob, entropy, value = self.evaluate_action(obs, action_normalized)
        ratio = torch.exp(new_log_prob - old_log_prob).unsqueeze(-1)

        actor_loss_1 = ratio * advantage
        actor_loss_2 = ratio.clamp(
            1.0 - self.cfg.actor.clip_ratio,
            1.0 + self.cfg.actor.clip_ratio,
        ) * advantage
        actor_loss = -torch.min(actor_loss_1, actor_loss_2).mean()

        value_clipped = old_value + (value - old_value).clamp(
            -self.cfg.critic.clip_ratio,
            self.cfg.critic.clip_ratio,
        )
        critic_loss = torch.max(
            F.smooth_l1_loss(value, returns, reduction="none"),
            F.smooth_l1_loss(value_clipped, returns, reduction="none"),
        ).mean()

        entropy_loss = -self.cfg.entropy_loss_coefficient * entropy.mean()
        loss = actor_loss + self.cfg.critic.value_loss_coefficient * critic_loss + entropy_loss

        self.feature_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.cfg.max_grad_norm,
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.cfg.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), self.cfg.max_grad_norm)

        self.feature_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy.mean().detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
        }

    def save(self, path):
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "value_norm": self.value_norm.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.value_norm.load_state_dict(checkpoint["value_norm"])

    def _initialize_lazy_modules(self, observation_space):
        obs = {
            "state": torch.zeros(1, observation_space["state"], device=self.device),
            "lidar": torch.zeros(1, *observation_space["lidar"], device=self.device),
            "dynamic_obstacle": torch.zeros(
                1,
                *observation_space["dynamic_obstacle"],
                device=self.device,
            ),
        }
        feature = self.feature_extractor(obs)
        self.actor(feature)
        self.critic(feature)

    def _flatten_observation(self, obs):
        return {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in obs.items()
        }

    def _to_world_action(self, action_normalized, direction):
        action = 2.0 * action_normalized * self.action_limit - self.action_limit
        action = vec_to_world(action, direction)
        return action.squeeze(1)
