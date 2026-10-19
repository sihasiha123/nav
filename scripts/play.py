"""Play one trained navigation episode in Isaac Sim with a persistent bird's-eye view."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained UAV navigation policy in Isaac Sim.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Template-Nav-Corridor-v0")
parser.add_argument("--agent", type=str, default="ppo_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--trail_thickness", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import nav.tasks  # noqa: F401, E402
from nav.tasks.manager_based.nav.agents.common import navigation_observation_shapes, obs_to_tensordict  # noqa: E402
from nav.tasks.manager_based.nav.agents.history_buffer import HistoryBuffer  # noqa: E402
from nav.tasks.manager_based.nav.agents.ppo import PPO  # noqa: E402


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    return value


def load_agent_cfg(task_name, entry_point_key):
    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict) and "algo" in cfg:
        cfg = cfg["algo"]
    return to_namespace(cfg)


def make_agent(cfg, env):
    return PPO(
        cfg=cfg,
        observation_space=navigation_observation_shapes(env),
        action_space=env.action_space.shape[-1],
        device=env.unwrapped.device,
    )


def get_term_mask(env, term_name):
    manager = env.termination_manager
    if term_name == "dynamic_collision" and term_name not in manager.active_terms:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return manager.get_term(term_name).reshape(-1).bool()


def termination_reason(env):
    priority = (
        ("static_collision", get_term_mask(env, "static_collision")),
        ("dynamic_collision", get_term_mask(env, "dynamic_collision")),
        ("out_of_bounds", get_term_mask(env, "out_of_bounds")),
        ("success", get_term_mask(env, "success")),
        ("time_out", get_term_mask(env, "time_out")),
    )
    for name, mask in priority:
        if bool(mask[0].item()):
            return name
    return "unknown"


def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args_cli.num_envs != 1:
        raise ValueError("play.py is intended for exactly one environment; use --num_envs 1.")
    if args_cli.trail_thickness <= 0.0:
        raise ValueError("--trail_thickness must be positive.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.eye = (0.0, 0.0, 32.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    torch.manual_seed(args_cli.seed)

    agent_cfg = load_agent_cfg(args_cli.task, args_cli.agent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    draw = omni_debug_draw.acquire_debug_draw_interface()
    trail_points = []

    try:
        agent = make_agent(agent_cfg, env)
        agent.load(checkpoint)
        agent.eval()

        robot = env.unwrapped.scene["robot"]
        body_ids, _ = robot.find_bodies("body", preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError("Expected exactly one robot body for trajectory drawing.")
        body_id = body_ids[0]
        obs = env.reset()
        obs_td = obs_to_tensordict(obs, 1, env.unwrapped.device)
        history_buffer = HistoryBuffer(
            1,
            env.unwrapped.device,
            history_length=agent.cfg.feature_extractor.history_length,
            history_stride=agent.cfg.feature_extractor.history_stride,
            action_limit=agent.action_limit,
        )
        history_buffer.append(obs_td["agents", "observation"])
        for key, value in history_buffer.window().items():
            obs_td["agents", "observation", key] = value

        episode_return = 0.0
        episode_step = 0
        episode_index = 1
        print(f"[PLAY] task={args_cli.task} checkpoint={checkpoint}")
        print("[PLAY] Only a successful episode stops and keeps its trajectory.")
        print("[PLAY] Collisions, out-of-bounds, and timeouts reset automatically.")

        with torch.inference_mode():
            while simulation_app.is_running():
                action_td = agent.act(obs_td) if args_cli.stochastic else agent.act_deterministic(obs_td)
                next_obs, reward, terminated, truncated, _ = env.step(action_td["agents", "action"])
                reward = reward.reshape(-1).float()
                done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
                episode_return += float(reward[0].item())
                episode_step += 1

                position = robot.data.body_link_pos_w[0, body_id].detach().cpu().tolist()
                trail_points.append(tuple(float(value) for value in position))
                if len(trail_points) >= 2:
                    draw.clear_lines()
                    draw.draw_lines(
                        trail_points[:-1],
                        trail_points[1:],
                        [[0.1, 1.0, 0.2, 1.0]] * (len(trail_points) - 1),
                        [args_cli.trail_thickness] * (len(trail_points) - 1),
                    )

                next_obs_td = obs_to_tensordict(next_obs, 1, env.unwrapped.device)
                history_buffer.reset(done)
                previous_action = 2.0 * action_td["agents", "action_normalized"] - 1.0
                previous_action[done] = 0.0
                history_buffer.append(next_obs_td["agents", "observation"], previous_action=previous_action)
                for key, value in history_buffer.window().items():
                    next_obs_td["agents", "observation", key] = value

                if bool(done[0].item()):
                    reason = termination_reason(env.unwrapped)
                    print(
                        f"[PLAY] episode={episode_index} reason={reason} "
                        f"steps={episode_step} return={episode_return:.3f}"
                    )
                    if reason == "success":
                        if trail_points:
                            trail_points.pop()
                        draw.clear_lines()
                        if len(trail_points) >= 2:
                            draw.draw_lines(
                                trail_points[:-1],
                                trail_points[1:],
                                [[0.1, 1.0, 0.2, 1.0]] * (len(trail_points) - 1),
                                [args_cli.trail_thickness] * (len(trail_points) - 1),
                            )
                        print("[PLAY] Navigation succeeded. Keeping the trajectory on screen.")
                        while simulation_app.is_running():
                            simulation_app.update()
                        break

                    trail_points.clear()
                    draw.clear_lines()
                    episode_index += 1
                    episode_return = 0.0
                    episode_step = 0
                    obs = env.reset()
                    obs_td = obs_to_tensordict(obs, 1, env.unwrapped.device)
                    history_buffer = HistoryBuffer(
                        1,
                        env.unwrapped.device,
                        history_length=agent.cfg.feature_extractor.history_length,
                        history_stride=agent.cfg.feature_extractor.history_stride,
                        action_limit=agent.action_limit,
                    )
                    history_buffer.append(obs_td["agents", "observation"])
                    for key, value in history_buffer.window().items():
                        obs_td["agents", "observation", key] = value

                if not bool(done[0].item()):
                    obs_td = next_obs_td
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
