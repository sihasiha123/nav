# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""训练无人机导航 PPO。"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train UAV navigation agents.")
parser.add_argument("--task", type=str, default="Template-Nav-v0")
parser.add_argument("--algo", type=str, default="ppo")
parser.add_argument("--agent", type=str, default="ppo_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--log_dir", type=str, default="runs")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import nav.tasks  # noqa: F401, E402
from nav.tasks.manager_based.nav.agents.common import obs_to_tensordict  # noqa: E402
from nav.tasks.manager_based.nav.agents.ppo import PPO  # noqa: E402


WANDB_PROJECT = "nav-drone-rl"


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )
    return value


def load_agent_cfg(task_name, entry_point_key):
    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict) and "algo" in cfg:
        cfg = cfg["algo"]
    return to_namespace(cfg)


def make_run_dir(log_dir, algo):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{algo}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    return float(value)


class EpisodeStatistics:
    """回合统计（第一版：return / length / truncated 率）。"""

    def __init__(self, num_envs, device):
        self._returns = torch.zeros(num_envs, device=device)
        self._lengths = torch.zeros(num_envs, device=device)
        self._return_sum = torch.zeros((), device=device)
        self._length_sum = torch.zeros((), device=device)
        self._count = torch.zeros((), dtype=torch.long, device=device)
        self._truncated_count = torch.zeros((), dtype=torch.long, device=device)

    def reset_rollout(self):
        self._return_sum.zero_()
        self._length_sum.zero_()
        self._count.zero_()
        self._truncated_count.zero_()

    def update(self, reward, terminated, truncated):
        self._returns += reward.squeeze(-1)
        self._lengths += 1

        done = (terminated | truncated).squeeze(-1)
        self._return_sum += self._returns[done].sum()
        self._length_sum += self._lengths[done].sum()
        self._count += done.sum()
        self._truncated_count += truncated.squeeze(-1)[done].sum()

        self._returns[done] = 0.0
        self._lengths[done] = 0.0

    def metrics(self):
        count = self._count.item()
        if count == 0:
            return {}
        return {
            "Episode/return": (self._return_sum / count).item(),
            "Episode/length": (self._length_sum / count).item(),
            "Episode/count": count,
            "Episode/truncated_rate": (self._truncated_count / count).item(),
        }


def collect_log_items(train_info, rollout, episode_statistics):
    log_items = {
        "Rollout/reward_mean": to_float(rollout["next", "agents", "reward"]),
    }
    log_items.update(episode_statistics.metrics())
    log_items.update(
        {
            "Loss/actor": to_float(train_info["actor_loss"]),
            "Loss/critic": to_float(train_info["critic_loss"]),
            "Policy/entropy": to_float(train_info["entropy"]),
            "GradNorm/actor": to_float(train_info["actor_grad_norm"]),
            "GradNorm/critic": to_float(train_info["critic_grad_norm"]),
        }
    )
    return log_items


def format_log(iteration, log_items):
    groups = {}
    for key, value in log_items.items():
        group, name = key.split("/", maxsplit=1)
        groups.setdefault(group, []).append((name, value))

    lines = [f"[TRAIN] iteration={iteration + 1}"]
    for group, metrics in groups.items():
        lines.append(f"  {group}")
        for name, value in metrics:
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"    {name:<16} {formatted_value}")
    return "\n".join(lines)


def init_wandb(run_dir, env_cfg):
    return wandb.init(
        project=WANDB_PROJECT,
        name=run_dir.name,
        mode="offline",
        dir=str(run_dir),
        settings=wandb.Settings(console="off"),
        config={
            "num_envs": env_cfg.scene.num_envs,
            "physics_dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "episode_length_s": env_cfg.episode_length_s,
        },
    )


def make_agent(algo, cfg, env):
    if algo == "ppo":
        # env.observation_space 是带 batch 维（num_envs）的空间，取形状时去掉第一维
        single_obs = env.observation_space["policy"]
        observation_space = {
            "state": single_obs["state"].shape[-1],
            "lidar": single_obs["lidar"].shape[1:],
            "dynamic_obstacle": single_obs["dynamic_obstacle"].shape[1:],
        }
        return PPO(
            cfg=cfg,
            observation_space=observation_space,
            action_space=env.action_space.shape[-1],
            device=env.unwrapped.device,
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


def collect_ppo_rollout(env, agent, obs_td, cfg, episode_statistics):
    frames = []

    for _ in range(cfg.training_frame_num):
        action_td = agent.act(obs_td.clone())

        # nav 环境只接收动作 tensor（不是 TensorDict）
        next_obs, reward, terminated, truncated, _ = env.step(action_td["agents", "action"])
        next_obs_td = obs_to_tensordict(next_obs, env.unwrapped.num_envs, env.unwrapped.device)

        reward = reward.reshape(env.unwrapped.num_envs, 1)
        terminated = terminated.reshape(env.unwrapped.num_envs, 1)
        truncated = truncated.reshape(env.unwrapped.num_envs, 1)
        episode_statistics.update(reward, terminated, truncated)

        next_observation = next_obs_td["agents", "observation"].detach().clone()
        current_observation = action_td["agents", "observation"].detach().clone()
        action_normalized = action_td["agents", "action_normalized"].detach().clone()

        next_data = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": next_observation,
                        "reward": reward.detach().clone(),
                    },
                    batch_size=[env.unwrapped.num_envs],
                    device=env.unwrapped.device,
                ),
                "terminated": terminated.detach().clone(),
                "truncated": truncated.detach().clone(),
                "done": (terminated | truncated).detach().clone(),
            },
            batch_size=[env.unwrapped.num_envs],
            device=env.unwrapped.device,
        )

        transition = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": current_observation,
                        "action_normalized": action_normalized,
                    },
                    batch_size=[env.unwrapped.num_envs],
                    device=env.unwrapped.device,
                ),
                "sample_log_prob": action_td["sample_log_prob"].detach().clone(),
                "state_value": action_td["state_value"].detach().clone(),
                "next": next_data,
            },
            batch_size=[env.unwrapped.num_envs],
            device=env.unwrapped.device,
        )
        frames.append(transition)

        obs_td = next_obs_td

    rollout = torch.stack(frames, dim=1)
    return rollout, obs_td


def train_ppo(env, agent, cfg, max_iterations, run_dir, save_interval, wandb_run):
    obs = env.reset()
    obs_td = obs_to_tensordict(obs, env.unwrapped.num_envs, env.unwrapped.device)
    episode_statistics = EpisodeStatistics(env.unwrapped.num_envs, env.unwrapped.device)

    for iteration in range(max_iterations):
        episode_statistics.reset_rollout()
        rollout, obs_td = collect_ppo_rollout(env, agent, obs_td, cfg, episode_statistics)
        train_info = agent.update(rollout)

        log_items = collect_log_items(train_info, rollout, episode_statistics)
        print(format_log(iteration, log_items))
        wandb_run.log(log_items, step=iteration + 1)

        if save_interval > 0 and (iteration + 1) % save_interval == 0:
            checkpoint_path = run_dir / f"checkpoint_{iteration + 1}.pt"
            agent.save(checkpoint_path)
            wandb_run.summary["checkpoint/latest_path"] = str(checkpoint_path)
            print(f"[INFO] checkpoint saved: {checkpoint_path}")

    final_path = run_dir / "checkpoint_final.pt"
    agent.save(final_path)
    wandb_run.summary["checkpoint/final_path"] = str(final_path)
    print(f"[INFO] final checkpoint saved: {final_path}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    agent_cfg = load_agent_cfg(args_cli.task, args_cli.agent)
    # 日志目录固定在 nav 项目根目录（scripts/..）下，避免写到 IsaacLab 启动目录
    log_root_path = Path(__file__).resolve().parents[1] / args_cli.log_dir
    run_dir = make_run_dir(str(log_root_path), args_cli.algo)

    env = gym.make(args_cli.task, cfg=env_cfg)
    agent = make_agent(args_cli.algo, agent_cfg, env)
    wandb_run = init_wandb(run_dir, env_cfg)

    if args_cli.algo == "ppo":
        train_ppo(
            env,
            agent,
            agent_cfg,
            args_cli.max_iterations,
            run_dir,
            args_cli.save_interval,
            wandb_run,
        )

    wandb_run.finish()
    wandb.teardown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
