"""CNN 空间编码与历史 Transformer 融合。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class TemporalFeatureExtractor(nn.Module):
    """每帧 CNN 编码后，用当前决策 query 读取历史 token。"""

    def __init__(self, token_dim=128, num_heads=4, tau=0.5):
        super().__init__()
        self.token_dim = token_dim
        self.tau = tau
        self.depth_encoder = nn.Sequential(
            ConvBlock(1, 16, 2),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, 2),
        )
        self.region_pool = nn.AdaptiveAvgPool2d((2, 3))
        self.visual_projection = nn.Linear(64, token_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(12, token_dim), nn.ELU(), nn.Linear(token_dim, token_dim)
        )
        self.goal_projection = nn.Sequential(
            nn.Linear(4, token_dim), nn.ELU(), nn.Linear(token_dim, token_dim)
        )
        self.query_projection = nn.Sequential(
            nn.Linear(token_dim + 9 + 3 + 4, token_dim), nn.ELU()
        )
        self.region_position = nn.Parameter(torch.zeros(1, 1, 6, token_dim))
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.query = nn.Linear(token_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.output_projection = nn.Linear(token_dim, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.ELU(), nn.Linear(token_dim * 2, token_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(token_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.LayerNorm(256)
        )

    def encode_depth(self, depth):
        batch, length = depth.shape[:2]
        feature = self.depth_encoder(depth.reshape(batch * length, *depth.shape[2:]))
        pooled = self.region_pool(feature).flatten(2).transpose(1, 2)
        return self.visual_projection(pooled).reshape(batch, length, 6, self.token_dim)

    def forward(self, obs):
        depth_history = obs["depth_history"]
        state_history = obs["state_history"]
        action_history = obs["action_history"]
        age = obs["age_seconds"]
        valid = obs["valid_mask"].bool()
        goal = obs["current_goal"]

        visual = self.encode_depth(depth_history)
        visual = visual + self.region_position
        state = self.state_projection(torch.cat((state_history, action_history), dim=-1))
        history_tokens = torch.cat((visual, state.unsqueeze(2)), dim=2)
        history_tokens = history_tokens.reshape(depth_history.shape[0], -1, self.token_dim)
        token_valid = valid.unsqueeze(-1).expand(-1, -1, 7).reshape(depth_history.shape[0], -1)

        current_visual = visual[:, -1].mean(dim=1)
        current_state = state_history[:, -1]
        current_action = action_history[:, -1]
        query = self.query_projection(
            torch.cat((current_visual, current_state, current_action, goal), dim=-1)
        ).unsqueeze(1)
        key_padding_mask = ~token_valid
        attention_bias = (-age / self.tau).repeat_interleave(7, dim=1)
        attention_bias = attention_bias[:, None, None, :].expand(
            -1, self.num_heads, 1, -1
        )

        q = self.query(query).reshape(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(history_tokens).reshape(-1, history_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(history_tokens).reshape(-1, history_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        logits = logits + attention_bias
        logits = logits.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(-1, 1, self.token_dim)
        attended = self.output_projection(attended)
        fused = self.norm1(query + attended)
        fused = self.norm2(fused + self.ffn(fused))
        return self.output(fused.squeeze(1))
