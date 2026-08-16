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
from nav.tasks.manager_based.nav.mdp.events import get_nav_task_buffer  # noqa: E402


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


class RolloutEnvStatistics:
    """统计当前 rollout 内所有 done 环境的终止原因。"""

    def __init__(self, env):
        self._env = env
        self._term_names = list(env.termination_manager.active_terms)
        self._done_count = torch.zeros((), dtype=torch.long, device=env.device)
        self._term_counts = {
            term_name: torch.zeros((), dtype=torch.long, device=env.device)
            for term_name in self._term_names
        }
        self._collision_count = torch.zeros((), dtype=torch.long, device=env.device)
        self._completed_returns = []

    def update(self, done, completed_returns=None):
        done = done.reshape(-1).bool()
        self._done_count += done.sum()
        if completed_returns is not None and completed_returns.numel() > 0:
            self._completed_returns.append(completed_returns.detach().float().cpu())
        if not done.any():
            return

        termination_manager = self._env.termination_manager
        collision_done = torch.zeros_like(done)
        for term_name in self._term_names:
            term_done = termination_manager.get_term(term_name).reshape(-1).bool()
            self._term_counts[term_name] += (term_done & done).sum()
            if term_name in ("static_collision", "dynamic_collision"):
                collision_done |= term_done
        self._collision_count += (collision_done & done).sum()

    def metrics(self):
        done_count = int(self._done_count.item())
        if done_count == 0:
            return {}

        metrics = {"Rollout_Done/count": done_count}
        denom = float(done_count)
        for term_name, term_count in self._term_counts.items():
            count = int(term_count.item())
            metrics[f"Rollout_Termination/{term_name}_count"] = count
            metrics[f"Rollout_Termination/{term_name}_rate"] = count / denom
        collision_count = int(self._collision_count.item())
        metrics["Rollout_Result/collision_count"] = collision_count
        metrics["Rollout_Result/collision_rate"] = collision_count / denom
        if self._completed_returns:
            returns = torch.cat(self._completed_returns)
            metrics["Rollout_Reward/done_return_mean"] = returns.mean().item()
            metrics["Rollout_Reward/done_return_min"] = returns.min().item()
            metrics["Rollout_Reward/done_return_max"] = returns.max().item()
        return metrics


class EpisodeReturnTracker:
    """跨 rollout 维护每个并行环境当前 episode 的累计回报。"""

    def __init__(self, num_envs, device):
        self._returns = torch.zeros(num_envs, device=device)

    def update(self, reward, done):
        reward = reward.reshape(-1)
        done = done.reshape(-1).bool()
        self._returns += reward
        if not done.any():
            return torch.empty(0, device=reward.device)

        completed_returns = self._returns[done].clone()
        self._returns[done] = 0.0
        return completed_returns


class RolloutRewardComponentStatistics:
    """统计当前 rollout 的 reward 分项均值。"""

    def __init__(self, env):
        self._env = env
        self._component_sums = {}
        self._component_counts = {}

    def update(self):
        buffer = get_nav_task_buffer(self._env)
        for name, component in buffer.reward_components.items():
            component = component.detach().float()
            if name not in self._component_sums:
                self._component_sums[name] = torch.zeros((), dtype=torch.float32, device=self._env.device)
                self._component_counts[name] = 0
            self._component_sums[name] += component.sum()
            self._component_counts[name] += component.numel()

    def metrics(self):
        metrics = {}
        for name, component_sum in self._component_sums.items():
            component_count = self._component_counts[name]
            if component_count > 0:
                metrics[f"Reward_Component/{name}_mean"] = (component_sum / component_count).item()
        return metrics


def collect_algo_log_items(train_info, rollout):
    return {
        "Rollout_Reward/step_mean": to_float(rollout["next", "agents", "reward"]),
        "Loss/actor": to_float(train_info["actor_loss"]),
        "Loss/critic": to_float(train_info["critic_loss"]),
        "Policy/entropy": to_float(train_info["entropy"]),
        "GradNorm/actor": to_float(train_info["actor_grad_norm"]),
        "GradNorm/critic": to_float(train_info["critic_grad_norm"]),
    }


def collect_terminal_log_items(env_log, algo_log):
    """终端只显示核心训练指标，详细统计保留给 wandb。"""
    env_log = env_log or {}
    log_items = {
        "rollout/done_count": env_log.get("Rollout_Done/count", 0),
    }
    if "Rollout_Reward/done_return_mean" in env_log:
        log_items["rollout/done_return_mean"] = env_log["Rollout_Reward/done_return_mean"]
    log_items["rollout/step_reward_mean"] = algo_log["Rollout_Reward/step_mean"]

    result_keys = {
        "success_rate": "Rollout_Termination/success_rate",
        "collision_rate": "Rollout_Result/collision_rate",
        "out_of_bounds_rate": "Rollout_Termination/out_of_bounds_rate",
        "time_out_rate": "Rollout_Termination/time_out_rate",
    }
    for display_name, source_name in result_keys.items():
        if source_name in env_log:
            log_items[f"result/{display_name}"] = env_log[source_name]

    reward_component_keys = {
        "progress": "Reward_Component/progress_mean",
        "velocity": "Reward_Component/goal_velocity_mean",
        "static": "Reward_Component/static_avoidance_mean",
        "dynamic": "Reward_Component/dynamic_avoidance_mean",
        "height": "Reward_Component/height_mean",
        "smooth": "Reward_Component/smoothness_mean",
        "collision": "Reward_Component/collision_mean",
        "out_of_bounds": "Reward_Component/out_of_bounds_mean",
    }
    for display_name, source_name in reward_component_keys.items():
        if source_name in env_log:
            log_items[f"reward/{display_name}"] = env_log[source_name]
    goal_bonus = 0.0
    has_goal_bonus = False
    for source_name in ("Reward_Component/goal_first_mean", "Reward_Component/goal_reached_mean"):
        if source_name in env_log:
            goal_bonus += env_log[source_name]
            has_goal_bonus = True
    if has_goal_bonus:
        log_items["reward/goal_bonus"] = goal_bonus

    log_items.update(
        {
            "loss/actor": algo_log["Loss/actor"],
            "loss/critic": algo_log["Loss/critic"],
            "loss/entropy": algo_log["Policy/entropy"],
            "grad/actor": algo_log["GradNorm/actor"],
            "grad/critic": algo_log["GradNorm/critic"],
        }
    )
    return log_items


def format_log(iteration, log_items, title="TRAIN"):
    groups = {}
    for key, value in log_items.items():
        if "/" in key:
            group, name = key.split("/", maxsplit=1)
        else:
            group, name = "Metric", key
        groups.setdefault(group, []).append((name, value))

    lines = [f"[{title}] iteration={iteration + 1}"]
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


def collect_ppo_rollout(env, agent, obs_td, cfg, return_tracker):
    frames = []
    env_statistics = RolloutEnvStatistics(env.unwrapped)
    reward_component_statistics = RolloutRewardComponentStatistics(env.unwrapped)

    for _ in range(cfg.training_frame_num):
        action_td = agent.act(obs_td.clone())

        # nav 环境只接收动作 tensor（不是 TensorDict）
        next_obs, reward, terminated, truncated, _ = env.step(action_td["agents", "action"])
        next_obs_td = obs_to_tensordict(next_obs, env.unwrapped.num_envs, env.unwrapped.device)

        reward = reward.reshape(env.unwrapped.num_envs, 1)
        terminated = terminated.reshape(env.unwrapped.num_envs, 1)
        truncated = truncated.reshape(env.unwrapped.num_envs, 1)
        done = terminated | truncated
        completed_returns = return_tracker.update(reward, done)
        env_statistics.update(done, completed_returns)
        reward_component_statistics.update()

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
    env_log = env_statistics.metrics()
    env_log.update(reward_component_statistics.metrics())
    return rollout, obs_td, env_log


def train_ppo(env, agent, cfg, max_iterations, run_dir, save_interval, wandb_run):
    obs = env.reset()
    obs_td = obs_to_tensordict(obs, env.unwrapped.num_envs, env.unwrapped.device)
    return_tracker = EpisodeReturnTracker(env.unwrapped.num_envs, env.unwrapped.device)

    for iteration in range(max_iterations):
        rollout, obs_td, env_log = collect_ppo_rollout(env, agent, obs_td, cfg, return_tracker)
        train_info = agent.update(rollout)

        algo_log = collect_algo_log_items(train_info, rollout)
        terminal_log = collect_terminal_log_items(env_log, algo_log)
        print(format_log(iteration, terminal_log, title="TRAIN"))

        if env_log:
            wandb_run.log(env_log, step=iteration + 1)
        wandb_run.log(algo_log, step=iteration + 1)

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
