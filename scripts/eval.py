# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained UAV navigation policy on complete parallel episodes."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a UAV navigation PPO checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Template-Nav-v0")
parser.add_argument("--agent", type=str, default="ppo_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--episodes_per_env", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions instead of using the deterministic Beta mean.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import nav.tasks  # noqa: F401, E402
from nav.tasks.manager_based.nav.agents.common import obs_to_tensordict  # noqa: E402
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


def get_term_mask(env, term_name):
    try:
        term = env.termination_manager.get_term(term_name)
    except (KeyError, RuntimeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return term.reshape(-1).bool()


def classify_termination(env, done):
    """Return one mutually exclusive reason for each completed environment."""
    static_collision = get_term_mask(env, "static_collision")
    dynamic_collision = get_term_mask(env, "dynamic_collision")
    out_of_bounds = get_term_mask(env, "out_of_bounds")
    success = get_term_mask(env, "success")
    time_out = get_term_mask(env, "time_out")

    reason = ["unknown"] * env.num_envs
    # Collision wins when multiple termination terms are true on one step.
    for index in torch.where(done & dynamic_collision & ~static_collision)[0].tolist():
        reason[index] = "dynamic_collision"
    for index in torch.where(done & static_collision)[0].tolist():
        reason[index] = "static_collision"
    for index in torch.where(done & ~static_collision & ~dynamic_collision & out_of_bounds)[0].tolist():
        reason[index] = "out_of_bounds"
    for index in torch.where(
        done & ~static_collision & ~dynamic_collision & ~out_of_bounds & success
    )[0].tolist():
        reason[index] = "success"
    for index in torch.where(
        done
        & ~static_collision
        & ~dynamic_collision
        & ~out_of_bounds
        & ~success
        & time_out
    )[0].tolist():
        reason[index] = "time_out"
    return reason


def evaluate(env, agent, episodes_per_env, step_dt, stochastic=False):
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    completed = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    records = []

    obs = env.reset()
    obs_td = obs_to_tensordict(obs, num_envs, device)
    max_episode_length = int(getattr(env.unwrapped, "max_episode_length", 0))
    if max_episode_length <= 0:
        episode_length_s = float(getattr(env.unwrapped.cfg, "episode_length_s", 60.0))
        max_episode_length = max(1, round(episode_length_s / step_dt))
    max_steps = max(num_envs * episodes_per_env * max(max_episode_length, 1) * 2, 1000)

    agent.eval()
    with torch.inference_mode():
        for _ in range(max_steps):
            if stochastic:
                action_td = agent.act(obs_td)
            else:
                action_td = agent.act_deterministic(obs_td)

            next_obs, reward, terminated, truncated, _ = env.step(action_td["agents", "action"])
            reward = reward.reshape(-1).float()
            done = (terminated.reshape(-1).bool() | truncated.reshape(-1).bool())
            episode_returns += reward
            episode_lengths += 1

            reasons = classify_termination(env.unwrapped, done)
            for env_id in torch.where(done & (completed < episodes_per_env))[0].tolist():
                reason = reasons[env_id]
                records.append(
                    {
                        "env_id": env_id,
                        "episode_index": int(completed[env_id].item()) + 1,
                        "return": float(episode_returns[env_id].item()),
                        "length_steps": int(episode_lengths[env_id].item()),
                        "duration_seconds": float(episode_lengths[env_id].item() * step_dt),
                        "termination_reason": reason,
                    }
                )
                completed[env_id] += 1
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0

            if torch.all(completed >= episodes_per_env):
                break

            obs_td = obs_to_tensordict(next_obs, num_envs, device)
        else:
            raise RuntimeError(
                "Evaluation exceeded its step budget before every environment "
                "completed the requested number of episodes."
            )

    return records


def summarize(records, checkpoint, task, seed, episodes_per_env, num_envs, step_dt, stochastic):
    reasons = ["success", "static_collision", "dynamic_collision", "out_of_bounds", "time_out"]
    counts = {reason: sum(row["termination_reason"] == reason for row in records) for reason in reasons}
    episode_count = len(records)
    returns = [row["return"] for row in records]
    lengths = [row["length_steps"] for row in records]
    success_durations = [
        row["duration_seconds"] for row in records if row["termination_reason"] == "success"
    ]
    collision_count = counts["static_collision"] + counts["dynamic_collision"]

    def rate(count):
        return count / episode_count if episode_count else 0.0

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "task": task,
        "seed": seed,
        "num_envs": num_envs,
        "episodes_per_env": episodes_per_env,
        "episode_count": episode_count,
        "step_dt": step_dt,
        "action_mode": "stochastic" if stochastic else "deterministic_mean",
        "success_count": counts["success"],
        "success_rate": rate(counts["success"]),
        "static_collision_count": counts["static_collision"],
        "static_collision_rate": rate(counts["static_collision"]),
        "dynamic_collision_count": counts["dynamic_collision"],
        "dynamic_collision_rate": rate(counts["dynamic_collision"]),
        "collision_count": collision_count,
        "collision_rate": rate(collision_count),
        "out_of_bounds_count": counts["out_of_bounds"],
        "out_of_bounds_rate": rate(counts["out_of_bounds"]),
        "time_out_count": counts["time_out"],
        "time_out_rate": rate(counts["time_out"]),
        "return_mean": sum(returns) / episode_count if episode_count else 0.0,
        "return_min": min(returns) if returns else 0.0,
        "return_max": max(returns) if returns else 0.0,
        "episode_length_mean": sum(lengths) / episode_count if episode_count else 0.0,
        "success_duration_mean": (
            sum(success_durations) / len(success_durations) if success_durations else 0.0
        ),
    }
    return summary


def make_output_dir(output_dir):
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / "output" / f"eval_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_html_report(output_dir, records, summary):
    """Write a self-contained browser report from the evaluation records."""
    payload = json.dumps(
        {"summary": summary, "records": records},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    report = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UAV Navigation Evaluation</title>
  <style>
    :root { color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; color: #172033; background: #f4f6f8; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { max-width: 1240px; margin: 0 auto; padding: 28px; }
    header { display: flex; align-items: start; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: 28px; font-weight: 700; }
    h2 { margin: 0 0 16px; font-size: 18px; font-weight: 700; }
    .subtitle, .meta { color: #5b6677; font-size: 14px; line-height: 1.6; }
    .meta { text-align: right; overflow-wrap: anywhere; max-width: 580px; }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .metric, .panel { background: #ffffff; border: 1px solid #d8dee8; border-radius: 6px; }
    .metric { min-height: 105px; padding: 16px; border-top: 4px solid #2e7d66; }
    .metric.warning { border-top-color: #b86a15; }
    .metric.danger { border-top-color: #b33b3b; }
    .metric-label { color: #5b6677; font-size: 13px; }
    .metric-value { margin-top: 10px; color: #172033; font-size: 26px; font-weight: 700; }
    .metric-note { margin-top: 4px; color: #7a8493; font-size: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    .panel { padding: 18px; min-width: 0; }
    .bars { display: grid; gap: 12px; }
    .bar-row { display: grid; grid-template-columns: 112px 1fr 72px; align-items: center; gap: 10px; font-size: 13px; }
    .track { height: 20px; overflow: hidden; background: #e9edf2; }
    .fill { height: 100%; min-width: 2px; }
    .fill.success { background: #2e7d66; }
    .fill.static_collision { background: #b86a15; }
    .fill.dynamic_collision { background: #b33b3b; }
    .fill.out_of_bounds { background: #6d5ba8; }
    .fill.time_out { background: #58708f; }
    .histogram { height: 214px; display: flex; align-items: end; gap: 3px; border-bottom: 1px solid #aeb8c6; padding: 0 4px; }
    .hist-bar { flex: 1 1 0; min-width: 2px; background: #30759f; position: relative; }
    .hist-bar:hover { background: #1d506e; }
    .axis { display: flex; justify-content: space-between; color: #6f7a8a; font-size: 12px; margin-top: 8px; }
    .table-tools { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
    select { height: 32px; padding: 0 8px; border: 1px solid #b9c3d0; border-radius: 4px; background: #fff; color: #172033; }
    .table-wrap { max-height: 480px; overflow: auto; border: 1px solid #d8dee8; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { position: sticky; top: 0; background: #edf1f5; color: #344054; text-align: left; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e3e8ef; white-space: nowrap; }
    td.number { text-align: right; font-variant-numeric: tabular-nums; }
    .tag { display: inline-block; padding: 3px 7px; border-radius: 3px; color: #fff; font-size: 12px; }
    .tag.success { background: #2e7d66; }
    .tag.static_collision { background: #b86a15; }
    .tag.dynamic_collision { background: #b33b3b; }
    .tag.out_of_bounds { background: #6d5ba8; }
    .tag.time_out { background: #58708f; }
    @media (max-width: 900px) { .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid { grid-template-columns: 1fr; } header { display: block; } .meta { max-width: none; margin-top: 10px; text-align: left; } main { padding: 16px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>UAV Navigation Evaluation</h1><div class="subtitle">完成 episode 的评估结果，不包含训练 rollout 数据。</div></div>
      <div class="meta" id="meta"></div>
    </header>
    <section class="metrics" id="metrics"></section>
    <section class="grid">
      <div class="panel"><h2>终止结果</h2><div class="bars" id="termination-bars"></div></div>
      <div class="panel"><h2>评估设置</h2><div class="bars" id="settings-bars"></div></div>
    </section>
    <section class="grid">
      <div class="panel"><h2>Episode Return 分布</h2><div class="histogram" id="return-histogram"></div><div class="axis" id="return-axis"></div></div>
      <div class="panel"><h2>Episode 时长分布</h2><div class="histogram" id="duration-histogram"></div><div class="axis" id="duration-axis"></div></div>
    </section>
    <section class="panel">
      <h2>Episode 明细</h2>
      <div class="table-tools"><span class="subtitle" id="table-count"></span><label>终止原因 <select id="reason-filter"></select></label></div>
      <div class="table-wrap"><table><thead><tr><th>环境</th><th>Episode</th><th>终止原因</th><th>Return</th><th>步数</th><th>时长 (s)</th></tr></thead><tbody id="episode-rows"></tbody></table></div>
    </section>
  </main>
  <script id="evaluation-data" type="application/json">__EVALUATION_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("evaluation-data").textContent);
    const summary = data.summary;
    const records = data.records;
    const reasons = [
      ["success", "成功"], ["static_collision", "静态碰撞"],
      ["dynamic_collision", "动态碰撞"], ["out_of_bounds", "越界"], ["time_out", "超时"],
    ];
    const percent = (value) => `${(value * 100).toFixed(2)}%`;
    const number = (value, digits = 2) => Number(value).toFixed(digits);
    document.getElementById("meta").textContent = `checkpoint: ${summary.checkpoint} | seed: ${summary.seed} | 动作: ${summary.action_mode}`;

    const metricData = [
      ["成功率", percent(summary.success_rate), `${summary.success_count} / ${summary.episode_count}`, ""],
      ["碰撞率", percent(summary.collision_rate), `${summary.collision_count} 个 episode`, "danger"],
      ["越界率", percent(summary.out_of_bounds_rate), `${summary.out_of_bounds_count} 个 episode`, "warning"],
      ["超时率", percent(summary.time_out_rate), `${summary.time_out_count} 个 episode`, "warning"],
      ["平均 Return", number(summary.return_mean), `范围 ${number(summary.return_min)} 至 ${number(summary.return_max)}`, ""],
    ];
    document.getElementById("metrics").innerHTML = metricData.map(([label, value, note, tone]) =>
      `<div class="metric ${tone}"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></div>`
    ).join("");

    function drawBars(containerId, rows) {
      const container = document.getElementById(containerId);
      const maxValue = Math.max(...rows.map((row) => row.value), 1);
      container.innerHTML = rows.map((row) => `<div class="bar-row"><span>${row.label}</span><div class="track"><div class="fill ${row.key || ""}" style="width:${(row.value / maxValue) * 100}%"></div></div><strong>${row.display}</strong></div>`).join("");
    }
    drawBars("termination-bars", reasons.map(([key, label]) => ({ key, label, value: summary[`${key}_count`], display: `${summary[`${key}_count`]} (${percent(summary[`${key}_rate`])})` })));
    drawBars("settings-bars", [
      { label: "评估 episode", value: summary.episode_count, display: String(summary.episode_count) },
      { label: "并行环境", value: summary.num_envs, display: String(summary.num_envs) },
      { label: "每环境 episode", value: summary.episodes_per_env, display: String(summary.episodes_per_env) },
      { label: "平均时长", value: summary.episode_length_mean, display: `${number(summary.episode_length_mean, 1)} step` },
    ]);

    function drawHistogram(histogramId, axisId, values) {
      const container = document.getElementById(histogramId);
      const axis = document.getElementById(axisId);
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const bins = 24;
      const width = Math.max((max - min) / bins, 1.0e-6);
      const counts = Array(bins).fill(0);
      values.forEach((value) => counts[Math.min(Math.floor((value - min) / width), bins - 1)] += 1);
      const maxCount = Math.max(...counts, 1);
      container.innerHTML = counts.map((count, index) => `<div class="hist-bar" title="${(min + index * width).toFixed(2)} 至 ${(min + (index + 1) * width).toFixed(2)}: ${count}" style="height:${(count / maxCount) * 100}%"></div>`).join("");
      axis.innerHTML = `<span>${number(min)}</span><span>${number((min + max) / 2)}</span><span>${number(max)}</span>`;
    }
    drawHistogram("return-histogram", "return-axis", records.map((row) => row.return));
    drawHistogram("duration-histogram", "duration-axis", records.map((row) => row.duration_seconds));

    const filter = document.getElementById("reason-filter");
    filter.innerHTML = `<option value="all">全部</option>${reasons.map(([key, label]) => `<option value="${key}">${label}</option>`).join("")}`;
    function renderTable() {
      const selected = filter.value;
      const visible = selected === "all" ? records : records.filter((row) => row.termination_reason === selected);
      document.getElementById("table-count").textContent = `显示 ${visible.length} / ${records.length} 个 episode`;
      document.getElementById("episode-rows").innerHTML = visible.map((row) => `<tr><td class="number">${row.env_id}</td><td class="number">${row.episode_index}</td><td><span class="tag ${row.termination_reason}">${row.termination_reason}</span></td><td class="number">${number(row.return)}</td><td class="number">${row.length_steps}</td><td class="number">${number(row.duration_seconds, 2)}</td></tr>`).join("");
    }
    filter.addEventListener("change", renderTable);
    renderTable();
  </script>
</body>
</html>
""".replace("__EVALUATION_DATA__", payload)
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def write_results(output_dir, records, summary):
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "env_id",
                "episode_index",
                "return",
                "length_steps",
                "duration_seconds",
                "termination_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    write_html_report(output_dir, records, summary)


def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args_cli.episodes_per_env < 1:
        raise ValueError("--episodes_per_env must be at least 1")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    torch.manual_seed(args_cli.seed)

    agent_cfg = load_agent_cfg(args_cli.task, args_cli.agent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    agent = make_agent(agent_cfg, env)
    agent.load(checkpoint)

    step_dt = float(getattr(env.unwrapped, "step_dt", env_cfg.sim.dt * env_cfg.decimation))
    output_dir = make_output_dir(args_cli.output_dir)
    try:
        records = evaluate(
            env,
            agent,
            args_cli.episodes_per_env,
            step_dt,
            stochastic=args_cli.stochastic,
        )
        summary = summarize(
            records,
            checkpoint,
            args_cli.task,
            args_cli.seed,
            args_cli.episodes_per_env,
            env.unwrapped.num_envs,
            step_dt,
            args_cli.stochastic,
        )
        write_results(output_dir, records, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[EVAL] results saved to {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
