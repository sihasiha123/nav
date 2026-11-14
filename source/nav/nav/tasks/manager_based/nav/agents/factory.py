"""Policy selection shared by training and evaluation."""

from .lstm_ppo import LSTMPPO
from .ppo import PPO


def make_policy(algo, cfg, observation_space, action_space, device):
    policy_classes = {"ppo": PPO, "lstm_ppo": LSTMPPO}
    if algo not in policy_classes:
        raise ValueError(f"Unsupported algorithm: {algo}")
    configured = getattr(cfg, "policy_type", "ppo")
    if configured != algo:
        raise ValueError(f"Algorithm {algo!r} requires a matching agent config, got {configured!r}.")
    if algo == "lstm_ppo":
        if cfg.feature_extractor.history_length != 1 or cfg.feature_extractor.history_stride != 1:
            raise ValueError("Persistent LSTM-PPO requires history_length=1 and history_stride=1.")
        if min(cfg.training_frame_num, cfg.training_epoch_num, cfg.num_minibatches) < 1:
            raise ValueError("Rollout, epoch and minibatch counts must be positive.")
    return policy_classes[algo](cfg, observation_space, action_space, device)
