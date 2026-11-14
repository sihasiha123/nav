"""Play navigation checkpoints and export episode trajectories (one environment)."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    return value


def termination_reason(env):
    manager = env.termination_manager
    for name in ("static_collision", "dynamic_collision", "out_of_bounds", "success", "time_out"):
        if name in manager.active_terms and bool(manager.get_term(name).reshape(-1)[0].item()):
            return name
    return "unknown"


@contextmanager
def defer_episode_reset(env, position):
    """Capture terminal state and defer auto-reset until the next explicit reset."""
    terminal = {}
    # Only this play environment is patched, never the training environment class.
    original = env.__dict__.get("_reset_idx")
    had_override = "_reset_idx" in env.__dict__

    def capture_terminal(env_ids):
        if env_ids.numel():
            terminal.update(position=position(), outcome=termination_reason(env))

    env._reset_idx = capture_terminal
    try:
        yield terminal
    finally:
        if had_override:
            env._reset_idx = original
        else:
            del env._reset_idx


def draw_trajectory(draw, points):
    if draw is not None:
        draw.clear_lines()
        if len(points) > 1:
            draw.draw_lines(
                points[:-1], points[1:], [[0.1, 1.0, 0.2, 1.0]] * (len(points) - 1), [4.0] * (len(points) - 1)
            )


def collect_episode(env, agent, app, draw=None, *, stochastic=False):
    """Record every step including the terminal position; leave the final scene intact."""
    import torch
    from nav.tasks.manager_based.nav.agents.common import obs_to_tensordict
    from nav.tasks.manager_based.nav.agents.history_buffer import HistoryBuffer

    started_at = time.perf_counter()
    obs = obs_to_tensordict(env.reset(), 1, env.unwrapped.device)
    if agent.is_recurrent:
        agent.reset_memory()
    if draw is not None:
        draw.clear_lines()
    feature_cfg = agent.cfg.feature_extractor
    history = HistoryBuffer(
        1, env.unwrapped.device, feature_cfg.history_length, feature_cfg.history_stride, agent.action_limit
    )
    history.append(obs["agents", "observation"])
    for key, value in history.window().items():
        obs["agents", "observation", key] = value
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies("body", preserve_order=True)
    if len(body_ids) != 1:
        raise ValueError("Expected one robot body for trajectory recording.")

    def position():
        return tuple(robot.data.body_link_pos_w[0, body_ids[0]].detach().cpu().tolist())

    points = [position()]
    goal = env.unwrapped.command_manager.get_command("nav_target")[0].detach().cpu().tolist()
    step_dt = float(env.unwrapped.step_dt)
    budget = int(env.unwrapped.max_episode_length) + 1
    total_return, steps, outcome = 0.0, 0, "interrupted"
    with torch.inference_mode(), defer_episode_reset(env.unwrapped, position) as terminal:
        for _ in range(budget):
            if not app.is_running():
                break
            action = agent.act(obs) if stochastic else agent.act_deterministic(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action["agents", "action"])
            total_return += float(reward.reshape(-1)[0].item())
            steps += 1
            done = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
            if agent.is_recurrent:
                agent.reset_memory(done)
            # Capture happens before ManagerBasedRLEnv would reset the scene.
            points.append(terminal["position"] if terminal else position())
            draw_trajectory(draw, points)
            if bool(done[0].item()):
                outcome = terminal.get("outcome", termination_reason(env.unwrapped))
                break
            obs = obs_to_tensordict(next_obs, 1, env.unwrapped.device)
            history.append(
                obs["agents", "observation"], previous_action=2.0 * action["agents", "action_normalized"] - 1.0
            )
            for key, value in history.window().items():
                obs["agents", "observation", key] = value
        else:
            outcome = "step_budget_exceeded"
    summary = {
        "outcome": outcome,
        "return": total_return,
        "length_steps": steps,
        "step_dt": step_dt,
        "duration_seconds": steps * step_dt,
        "wall_time_seconds": time.perf_counter() - started_at,
        "path_length_m": sum(math.dist(a, b) for a, b in zip(points, points[1:])),
        "path_length_xy_m": sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:])),
        "goal": goal,
        "action_mode": "stochastic" if stochastic else "deterministic_mean",
        "terminal_position_recorded": bool(terminal),
        "position_note": "Terminal position captured before reset; path length sums 3D world-coordinate segments.",
    }
    return points, summary


def save_trajectory(output_dir, episode, points, summary, *, stem=None):
    """Save coordinates/metadata and a dependency-free SVG; also PNG if available."""
    stem = stem or ("trajectory" if episode == 1 else f"trajectory_{episode:03d}")
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "time_seconds", "x", "y", "z"))
        for step, point in enumerate(points):
            writer.writerow((step, step * summary["step_dt"], *point))
    (output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    goal = summary["goal"]
    positions = [*points, goal]
    center = [(min(p[d] for p in positions) + max(p[d] for p in positions)) / 2 for d in (0, 1)]
    span = max(*(max(p[d] for p in positions) - min(p[d] for p in positions) for d in (0, 1)), 1.0) * 1.15

    def project(point):
        return 400 + (point[0] - center[0]) * 660 / span, 400 - (point[1] - center[1]) * 660 / span

    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, points))
    sx, sy = project(points[0])
    ex, ey = project(points[-1])
    gx, gy = project(goal)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="840" viewBox="0 0 800 840">'
        '<rect width="800" height="840" fill="white"/>'
        '<text x="400" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">'
        "Navigation trajectory (top view)</text>"
        '<rect x="60" y="60" width="680" height="680" fill="#f8fafc" stroke="#94a3b8"/>'
        f'<polyline points="{line}" fill="none" stroke="#1261a0" stroke-width="3"/>'
        f'<circle cx="{sx}" cy="{sy}" r="7" fill="#15803d"/>'
        f'<circle cx="{ex}" cy="{ey}" r="5" fill="#c2410c"/>'
        f'<path d="M {gx - 7} {gy - 7} L {gx + 7} {gy + 7} M {gx - 7} {gy + 7} L {gx + 7} {gy - 7}" '
        'stroke="#b91c1c" stroke-width="3"/>'
        '<text x="400" y="780" text-anchor="middle" font-family="sans-serif">'
        "Green=start; orange=last sample; red=goal; no obstacle background</text>"
        f'<text x="400" y="810" text-anchor="middle" font-family="sans-serif">'
        f"{660 / span:.2f} pixels per meter; +X right, +Y up</text></svg>"
    )
    (output_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
    try:
        import matplotlib
    except ImportError:
        print("[PLAY] Matplotlib unavailable; saved SVG trajectory and CSV data.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([p[0] for p in points], [p[1] for p in points], color="#1261a0", label="Trajectory")
    axis.scatter(*points[0][:2], color="#15803d", s=50, label="Start", zorder=3)
    axis.scatter(*points[-1][:2], color="#c2410c", s=40, label="Last sample", zorder=3)
    axis.scatter(*goal[:2], color="#b91c1c", marker="*", s=130, label="Goal", zorder=3)
    axis.set(xlabel="World X (m)", ylabel="World Y (m)", title=f"Navigation trajectory: {summary['outcome']}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.png", dpi=200)
    plt.close(figure)


def run_playback(env, agent, app, output_dir, options, metadata, draw=None):
    """Retry failures, persist every attempt, and stop at the first success by default."""
    until_success = options.episodes is None
    fields = (
        "episode",
        "outcome",
        "return",
        "length_steps",
        "path_length_m",
        "path_length_xy_m",
        "duration_seconds",
        "wall_time_seconds",
    )
    started_at = time.perf_counter()
    result = {
        **metadata,
        "mode": "until_success" if until_success else "fixed_episodes",
        "attempted_episodes": 0,
        "completed_episodes": 0,
        "failed_episodes": 0,
        "successful_episodes": 0,
        "success_episode": None,
        "success_path_length_m": None,
        "success_path_length_xy_m": None,
        "success_duration_seconds": None,
        "success_wall_time_seconds": None,
        "total_simulation_seconds": 0.0,
        "total_wall_time_seconds": 0.0,
        "outcome": "interrupted",
    }
    with (output_dir / "attempts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        file.flush()
        while app.is_running():
            episode = result["attempted_episodes"] + 1
            points, summary = collect_episode(env, agent, app, draw, stochastic=options.stochastic)
            summary.update(metadata, episode=episode)
            outcome = summary["outcome"]
            completed = outcome not in ("interrupted", "step_budget_exceeded")
            result["attempted_episodes"] = episode
            result["completed_episodes"] += int(completed)
            result["failed_episodes"] += int(completed and outcome != "success")
            result["successful_episodes"] += int(outcome == "success")
            result["total_simulation_seconds"] += summary["duration_seconds"]
            result["total_wall_time_seconds"] = time.perf_counter() - started_at
            result["outcome"] = outcome
            writer.writerow(summary)
            file.flush()
            if outcome == "success":
                result.update(
                    success_episode=episode,
                    success_path_length_m=summary["path_length_m"],
                    success_path_length_xy_m=summary["path_length_xy_m"],
                    success_duration_seconds=summary["duration_seconds"],
                    success_wall_time_seconds=summary["wall_time_seconds"],
                )
                env.unwrapped.sim.pause()
            summary.update(
                attempted_episodes=episode,
                failed_episodes=result["failed_episodes"],
                total_wall_time_seconds=result["total_wall_time_seconds"],
            )
            limit_reached = bool(options.max_attempts and episode >= options.max_attempts)
            if limit_reached and until_success and outcome != "success":
                result["outcome"] = "max_attempts_reached"
            # Persist metrics before optional plotting so results survive an export error.
            (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if outcome == "success" or not until_success:
                save_trajectory(output_dir, 1 if until_success else episode, points, summary)
            print(
                f"[PLAY] episode={episode} outcome={outcome} return={summary['return']:.3f} "
                f"path={summary['path_length_m']:.3f} m simulation_time={summary['duration_seconds']:.3f} s",
                flush=True,
            )
            if outcome == "success":
                print(
                    f"[PLAY] SUCCESS after {episode} attempt(s); failures={result['failed_episodes']}; "
                    f"episode_wall_time={summary['wall_time_seconds']:.3f} s; "
                    f"total_wall_time={result['total_wall_time_seconds']:.3f} s",
                    flush=True,
                )
            if not completed or limit_reached or (until_success and outcome == "success"):
                break
            if not until_success and episode >= options.episodes:
                break
            if outcome == "success":
                env.unwrapped.sim.play()
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[PLAY] Results saved to {output_dir.resolve()}", flush=True)
    return result


def hold_success(env, app):
    """Render a paused terminal scene so the camera can move without advancing physics."""
    sim = env.unwrapped.sim
    sim.pause()
    print(
        "[PLAY] Success scene paused. Move the camera for screenshots; close the window or Ctrl+C to exit.", flush=True
    )
    while app.is_running():
        if sim.is_playing():
            sim.pause()
        sim.render()


def main():
    import gymnasium as gym
    import nav.tasks  # noqa: F401
    import torch
    from nav.tasks.manager_based.nav.agents.common import navigation_observation_shapes
    from nav.tasks.manager_based.nav.agents.factory import make_policy

    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    algo = data.get("policy_type", "ppo")
    if args_cli.agent is None and "agent_config" in data:
        cfg = to_namespace(data["agent_config"])
    else:
        cfg = load_cfg_from_registry(args_cli.task, args_cli.agent or f"{algo}_cfg_entry_point")
        cfg = to_namespace(cfg.get("algo", cfg) if isinstance(cfg, dict) else cfg)
    del data
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    torch.manual_seed(args_cli.seed)
    output_dir = (
        Path(args_cli.output_dir).expanduser()
        if args_cli.output_dir
        else (Path(__file__).resolve().parents[1] / "output" / f"play_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        agent = make_policy(
            algo, cfg, navigation_observation_shapes(env), env.action_space.shape[-1], env.unwrapped.device
        )
        agent.load(checkpoint)
        agent.eval()
        draw = None
        if not args_cli.headless:
            try:
                from isaacsim.core.utils.extensions import enable_extension

                enable_extension("isaacsim.util.debug_draw")
                import isaacsim.util.debug_draw._debug_draw as debug_draw

                draw = debug_draw.acquire_debug_draw_interface()
            except ImportError:
                print("[PLAY] Debug draw unavailable; trajectory files will still be saved.")
        result = run_playback(
            env,
            agent,
            simulation_app,
            output_dir,
            args_cli,
            dict(checkpoint=str(checkpoint), task=args_cli.task, algorithm=algo, seed=args_cli.seed),
            draw,
        )
        if result["outcome"] == "success" and args_cli.hold and not args_cli.headless:
            hold_success(env, simulation_app)
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="Template-Nav-v0")
    parser.add_argument("--agent", default=None, help="Override saved agent config with a registry key.")
    parser.add_argument("--num_envs", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--until_success", action="store_true", help="Retry failed episodes until success (default).")
    mode.add_argument(
        "--episodes", type=int, default=None, help="Optional fixed episode count instead of retry-until-success."
    )
    parser.add_argument("--max_attempts", type=int, default=0, help="Optional attempt limit; 0 means unlimited.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the successful terminal scene paused for screenshots (default); --no-hold exits.",
    )
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.num_envs != 1 or (args_cli.episodes is not None and args_cli.episodes < 1) or args_cli.max_attempts < 0:
        parser.error("play.py requires --num_envs 1, --episodes >= 1 if provided, and --max_attempts >= 0")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        main()
    except KeyboardInterrupt:
        print("[PLAY] Interrupted; completed attempt records remain in the output directory.")
    finally:
        simulation_app.close()
