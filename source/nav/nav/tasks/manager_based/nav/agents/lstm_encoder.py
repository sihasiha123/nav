"""Current-frame region encoding with persistent, per-environment LSTM memory."""

from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_encoder import ConvBlock


class LSTMFeatureExtractor(nn.Module):
    """Keep the spatial/query architecture; replace temporal attention with an LSTM."""

    def __init__(self, token_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        if min(token_dim, hidden_dim, num_layers) < 1:
            raise ValueError("LSTM dimensions and layer count must be positive.")
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.depth_encoder = nn.Sequential(
            ConvBlock(1, 16, 2),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, 2),
        )
        self.region_pool = nn.AdaptiveAvgPool2d((2, 3))
        self.visual_projection = nn.Linear(64, token_dim)
        self.region_position = nn.Parameter(torch.zeros(1, 6, token_dim))
        self.state_projection = nn.Sequential(nn.Linear(12, token_dim), nn.ELU(), nn.Linear(token_dim, token_dim))
        self.input_projection = nn.Sequential(nn.Linear(7 * token_dim, hidden_dim), nn.ELU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.memory_projection = nn.Identity() if hidden_dim == token_dim else nn.Linear(hidden_dim, token_dim)
        self.query_projection = nn.Sequential(nn.Linear(token_dim + 9 + 3 + 4, token_dim), nn.ELU())
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(nn.Linear(token_dim, token_dim * 2), nn.ELU(), nn.Linear(token_dim * 2, token_dim))
        self.output = nn.Sequential(nn.Linear(token_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.LayerNorm(256))

    def initial_memory(self, batch_size):
        reference = self.region_position
        shape = (self.num_layers, batch_size, self.hidden_dim)
        return reference.new_zeros(shape), reference.new_zeros(shape)

    def _encode_current(self, obs):
        visual = self.depth_encoder(obs["depth"])
        visual = self.region_pool(visual).flatten(2).transpose(1, 2)
        visual = self.visual_projection(visual) + self.region_position
        state = self.state_projection(torch.cat((obs["state"], obs["previous_action"]), dim=-1))
        tokens = torch.cat((visual, state.unsqueeze(1)), dim=1)
        recurrent_input = self.input_projection(tokens.flatten(1))
        query = self.query_projection(
            torch.cat((visual.mean(dim=1), obs["state"], obs["previous_action"], obs["goal"]), dim=-1)
        )
        return recurrent_input, query

    def _fuse(self, query, memory_output):
        fused = self.norm1(query + self.memory_projection(memory_output))
        return self.output(self.norm2(fused + self.ffn(fused)))

    @staticmethod
    def _mask_memory(memory, episode_start):
        keep = (~episode_start.reshape(-1).bool()).to(memory[0].dtype)[None, :, None]
        return tuple(state * keep for state in memory)

    def forward(self, obs, memory=None, episode_start=None):
        """Process one frame, returning features and the updated (h, c)."""
        recurrent_input, query = self._encode_current(obs)
        if memory is None:
            memory = self.initial_memory(recurrent_input.shape[0])
        if episode_start is not None:
            memory = self._mask_memory(memory, episode_start)
        output, memory = self.lstm(recurrent_input.unsqueeze(1), memory)
        return self._fuse(query, output[:, 0]), memory

    def forward_sequence(self, obs, memory, episode_start):
        """Unroll ordered (batch, time, ...) fragments, resetting before new episodes."""
        batch_size, length = obs["state"].shape[:2]
        flattened = {key: value.reshape(batch_size * length, *value.shape[2:]) for key, value in obs.items()}
        recurrent_input, query = self._encode_current(flattened)
        recurrent_input = recurrent_input.reshape(batch_size, length, -1)
        query = query.reshape(batch_size, length, -1)
        outputs = []
        for step in range(length):
            memory = self._mask_memory(memory, episode_start[:, step])
            output, memory = self.lstm(recurrent_input[:, step : step + 1], memory)
            outputs.append(output[:, 0])
        return self._fuse(query, torch.stack(outputs, dim=1)), memory
