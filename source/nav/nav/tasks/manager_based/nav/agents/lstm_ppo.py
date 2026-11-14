"""Persistent-memory PPO with ordered, masked truncated BPTT updates."""

from __future__ import annotations

import torch

from .lstm_encoder import LSTMFeatureExtractor
from .ppo import PPO


def make_recurrent_minibatches(num_envs, rollout_length, sequence_length, num_minibatches, device):
    """Shuffle whole fragments, covering every transition including short tails."""
    if min(num_envs, rollout_length, sequence_length, num_minibatches) < 1:
        raise ValueError("Recurrent batch sizes and sequence length must be positive.")
    starts = torch.arange(0, rollout_length, sequence_length, device=device)
    env_ids = torch.arange(num_envs, device=device).repeat_interleave(starts.numel())
    starts = starts.repeat(num_envs)
    order = torch.randperm(starts.numel(), device=device)
    for indices in order.tensor_split(min(num_minibatches, order.numel())):
        yield env_ids[indices], starts[indices]


class LSTMPPO(PPO):
    """Share PPO's heads/objective; maintain detached memory only during collection."""

    policy_type = "lstm_ppo"
    is_recurrent = True

    def __init__(self, cfg, observation_space, action_space, device):
        super().__init__(cfg, observation_space, action_space, device)
        self.sequence_length = int(cfg.recurrent.sequence_length)
        if self.sequence_length < 1:
            raise ValueError("recurrent.sequence_length must be positive.")
        self._memory = None
        self._episode_start = None

    def _make_feature_extractor(self, feature_cfg):
        return LSTMFeatureExtractor(
            token_dim=getattr(feature_cfg, "token_dim", 128),
            hidden_dim=getattr(feature_cfg, "hidden_dim", 128),
            num_layers=getattr(feature_cfg, "num_layers", 1),
        )

    def _initial_feature(self, shapes):
        obs = {key: torch.zeros(1, *shape, device=self.device) for key, shape in shapes.items()}
        obs["previous_action"] = torch.zeros(1, 3, device=self.device)
        return self.feature_extractor(obs)[0]

    def _current_observation(self, obs):
        return {
            "state": obs["state"],
            "depth": obs["depth"],
            "goal": obs["goal"],
            "previous_action": obs["action_history"][:, -1],
        }

    def _ensure_memory(self, batch_size):
        if self._memory is None or self._memory[0].shape[1] != batch_size:
            self._memory = self.feature_extractor.initial_memory(batch_size)
            self._episode_start = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def reset_memory(self, done=None):
        """Reset only completed environments; a full reset discards all online state."""
        if done is None:
            self._memory = None
            self._episode_start = None
        elif self._memory is not None:
            done = done.reshape(-1).bool()
            if done.numel() != self._memory[0].shape[1]:
                raise ValueError("Memory reset mask must contain one entry per environment.")
            self._memory = tuple(state.masked_fill(done[None, :, None], 0.0) for state in self._memory)
            self._episode_start[done] = True

    @torch.no_grad()
    def _act(self, tensordict, deterministic):
        obs = self._current_observation(tensordict["agents", "observation"])
        self._ensure_memory(obs["state"].shape[0])
        # Store the PRE-observation state, in TensorDict's environment-first layout.
        tensordict["recurrent_h"] = self._memory[0].transpose(0, 1).clone()
        tensordict["recurrent_c"] = self._memory[1].transpose(0, 1).clone()
        tensordict["episode_start"] = self._episode_start.clone()
        feature, memory = self.feature_extractor(obs, self._memory, self._episode_start)
        self._memory = tuple(state.detach() for state in memory)
        self._episode_start.zero_()
        if deterministic:
            action_normalized = self.actor.deterministic(feature)
            log_prob, _ = self.actor.evaluate_action(feature, action_normalized)
        else:
            action_normalized, log_prob = self.actor.sample(feature)
        tensordict["agents", "action_normalized"] = action_normalized
        tensordict["agents", "action"] = self._to_body_action(action_normalized)
        tensordict["sample_log_prob"] = log_prob
        tensordict["state_value"] = self.critic(feature)
        return tensordict

    def act(self, tensordict):
        return self._act(tensordict, deterministic=False)

    def act_deterministic(self, tensordict):
        return self._act(tensordict, deterministic=True)

    @torch.no_grad()
    def bootstrap_value(self, tensordict):
        """Look ahead without consuming a frame or changing the collector's memory."""
        obs = self._current_observation(tensordict["agents", "observation"])
        self._ensure_memory(obs["state"].shape[0])
        feature, _ = self.feature_extractor(obs, self._memory, self._episode_start)
        return self.critic(feature)

    def evaluate_action(self, obs, action_normalized, memory=None, episode_start=None):
        """Evaluate explicit recurrent inputs without mutating online memory."""
        feature, _ = self.feature_extractor(obs, memory, episode_start)
        log_prob, entropy = self.actor.evaluate_action(feature, action_normalized)
        return log_prob, entropy, self.critic(feature)

    def update(self, rollout):
        transitions = rollout["transitions"]
        num_envs, length = transitions.batch_size
        # Interior next values were already computed during collection. Done masks
        # in GAE suppress all terminal bootstraps, including timeouts, as in RTA-PPO.
        next_value = torch.cat((transitions["state_value"][:, 1:], rollout["bootstrap_value"][:, None]), dim=1)
        self._prepare_returns(transitions, next_value)
        infos = []
        for _ in range(self.cfg.training_epoch_num):
            for env_ids, starts in make_recurrent_minibatches(
                num_envs, length, self.sequence_length, self.cfg.num_minibatches, transitions.device
            ):
                batch, obs, memory, episode_start, valid = self._sequence_batch(rollout, env_ids, starts)
                feature, _ = self.feature_extractor.forward_sequence(obs, memory, episode_start)
                infos.append(self._update_features(batch, feature, valid))
        return {key: torch.stack([info[key] for info in infos]).mean().item() for key in infos[0]}

    def _sequence_batch(self, rollout, env_ids, starts):
        transitions = rollout["transitions"]
        length = transitions.batch_size[1]
        fragment_length = min(self.sequence_length, length)
        times = starts[:, None] + torch.arange(fragment_length, device=self.device)[None]
        valid = times < length
        times = times.clamp_max(length - 1)
        env_index = env_ids[:, None].expand_as(times)
        batch = transitions[env_index, times]
        history = rollout["history"]
        # The LSTM collector keeps exactly one initial frame followed by T next frames.
        if history["sequence_state"].shape[1] != length + 1:
            raise ValueError("LSTM-PPO requires a current-frame buffer (history_length=1).")
        obs = {
            key: history[f"sequence_{source}"][env_index, times]
            for key, source in (
                ("state", "state"),
                ("goal", "goal"),
                ("depth", "depth"),
                ("previous_action", "action"),
            )
        }
        memory = tuple(
            transitions[key][env_ids, starts].transpose(0, 1).contiguous().detach()
            for key in ("recurrent_h", "recurrent_c")
        )
        episode_start = batch["episode_start"].squeeze(-1) | ~valid
        return batch, obs, memory, episode_start, valid
