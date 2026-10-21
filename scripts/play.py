"""Visualize a single-environment navigation policy and save trajectories."""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play and record a navigation PPO policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Template-Nav-Corridor-v0")
parser.add_argument("--agent", type=str, default="ppo_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="Number of successful episodes to record; failed episodes retry automatically.",
)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--trail_thickness", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import nav.tasks  # noqa: F401, E402
from nav.tasks.manager_based.nav.agents.common import navigation_observation_shapes, obs_to_tensordict  # noqa: E402
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
    if term_name not in manager.active_terms:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return manager.get_term(term_name).reshape(-1).bool()


def termination_reason(env):
    for name in ("static_collision", "dynamic_collision", "out_of_bounds", "success", "time_out"):
        if bool(get_term_mask(env, name)[0].item()):
            return name
    return "unknown"


def draw_trail(draw, points, thickness):
    draw.clear_lines()
    if len(points) >= 2:
        draw.draw_lines(
            points[:-1],
            points[1:],
            [[0.1, 1.0, 0.2, 1.0]] * (len(points) - 1),
            [thickness] * (len(points) - 1),
        )


def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args_cli.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args_cli.num_envs != 1:
        raise ValueError("play.py records one trajectory at a time; --num_envs must be 1.")
    if args_cli.trail_thickness <= 0.0:
        raise ValueError("--trail_thickness must be positive")

    output_dir = (
        Path(args_cli.output_dir).expanduser()
        if args_cli.output_dir
        else Path(__file__).resolve().parents[1]
        / "output"
        / f"play_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.csv"
    episodes_path = output_dir / "episodes.csv"

    print(f"[PLAY] starting task={args_cli.task}")
    print(f"[PLAY] checkpoint={checkpoint}")

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

    try:
        print("[PLAY] environment created; building PPO...")
        agent = make_agent(agent_cfg, env)
        try:
            agent.load(checkpoint)
        except (RuntimeError, KeyError) as exc:
            raise RuntimeError(
                "Checkpoint does not match the current single-frame CNN-PPO architecture. "
                f"Original error: {exc}"
            ) from exc
        agent.eval()
        print("[PLAY] checkpoint loaded")

        robot = env.unwrapped.scene["robot"]
        body_ids, _ = robot.find_bodies("body", preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError("Expected exactly one robot body for trajectory recording.")
        body_id = body_ids[0]

        obs = env.reset()
        obs_td = obs_to_tensordict(obs, 1, env.unwrapped.device)
        episode_return = 0.0
        episode_step = 0
        episode_index = 0
        successful_episodes = 0
        position = robot.data.body_link_pos_w[0, body_id].detach().cpu().tolist()
        points = [tuple(float(value) for value in position)]
        print(f"[PLAY] saving trajectories to {output_dir}")

        with trajectory_path.open("w", newline="", encoding="utf-8") as trajectory_file, episodes_path.open(
            "w", newline="", encoding="utf-8"
        ) as episodes_file:
            trajectory_writer = csv.DictWriter(
                trajectory_file,
                fieldnames=("task", "seed", "episode", "step", "x", "y", "z", "episode_return", "outcome"),
            )
            episode_writer = csv.DictWriter(
                episodes_file,
                fieldnames=("task", "seed", "episode", "outcome", "return", "length_steps", "duration_seconds"),
            )
            trajectory_writer.writeheader()
            episode_writer.writeheader()

            if not simulation_app.is_running():
                raise RuntimeError("Isaac Sim is no longer running before the play loop started.")

            with torch.inference_mode():
                while simulation_app.is_running():
                    action_td = agent.act(obs_td) if args_cli.stochastic else agent.act_deterministic(obs_td)
                    next_obs, reward, terminated, truncated, _ = env.step(action_td["agents", "action"])
                    episode_return += float(reward.reshape(-1)[0].item())
                    episode_step += 1
                    done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()

                    if not bool(done[0].item()):
                        position = robot.data.body_link_pos_w[0, body_id].detach().cpu().tolist()
                        points.append(tuple(float(value) for value in position))
                        draw_trail(draw, points, args_cli.trail_thickness)

                    next_obs_td = obs_to_tensordict(next_obs, 1, env.unwrapped.device)

                    if bool(done[0].item()):
                        episode_index += 1
                        reason = termination_reason(env.unwrapped)
                        duration = float(episode_step * env.unwrapped.step_dt)
                        for step, (x, y, z) in enumerate(points):
                            trajectory_writer.writerow({
                                "task": args_cli.task,
                                "seed": args_cli.seed,
                                "episode": episode_index,
                                "step": step,
                                "x": x,
                                "y": y,
                                "z": z,
                                "episode_return": episode_return,
                                "outcome": reason,
                            })
                        episode_writer.writerow({
                            "task": args_cli.task,
                            "seed": args_cli.seed,
                            "episode": episode_index,
                            "outcome": reason,
                            "return": episode_return,
                            "length_steps": episode_step,
                            "duration_seconds": duration,
                        })
                        trajectory_file.flush()
                        episodes_file.flush()
                        print(
                            f"[PLAY] episode={episode_index} outcome={reason} "
                            f"steps={episode_step} return={episode_return:.3f} "
                            f"duration={duration:.2f}s"
                        )

                        if reason == "success":
                            successful_episodes += 1
                        if successful_episodes >= args_cli.episodes:
                            print(f"[PLAY] trajectory saved: {trajectory_path}")
                            print(f"[PLAY] episode summary saved: {episodes_path}")
                            while simulation_app.is_running():
                                simulation_app.update()
                            break

                        points = []
                        draw.clear_lines()
                        episode_return = 0.0
                        episode_step = 0
                        obs = env.reset()
                        obs_td = obs_to_tensordict(obs, 1, env.unwrapped.device)
                        position = robot.data.body_link_pos_w[0, body_id].detach().cpu().tolist()
                        points = [tuple(float(value) for value in position)]
                    else:
                        obs_td = next_obs_td
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
