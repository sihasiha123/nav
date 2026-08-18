#!/usr/bin/env python3
"""Plot iteration versus completed-episode return from an offline W&B run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from wandb.proto import wandb_internal_pb2  # noqa: E402
from wandb.sdk.internal.datastore import DataStore  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC = "Rollout_Reward/done_return_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot iteration versus return from a run's offline W&B history."
    )
    parser.add_argument(
        "run_dir",
        help="Run directory, for example runs/ppo_20260818_204907 or ppo_20260818_204907.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Trailing moving-average window. Set to 1 to disable smoothing (default: 50).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"W&B history metric to plot (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <project_root>/figure/<run_name>_iteration_return.png).",
    )
    return parser.parse_args()


def resolve_run_dir(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    if not candidate.is_absolute():
        runs_candidate = PROJECT_ROOT / "runs" / candidate
        if runs_candidate.is_dir():
            return runs_candidate.resolve()

    raise FileNotFoundError(f"Run directory does not exist: {value}")


def find_wandb_file(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("wandb/offline-run-*/run-*.wandb"))
    candidates.extend(run_dir.glob("wandb/run-*/run-*.wandb"))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run-*.wandb file found under: {run_dir / 'wandb'}")
    if len(candidates) > 1:
        print(f"[WARN] Found {len(candidates)} W&B files; using the newest one.")
    return candidates[-1]


def history_item_key(item) -> str:
    if item.nested_key:
        return "/".join(item.nested_key)
    return item.key


def load_metric(wandb_file: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    values_by_step: dict[int, float] = {}

    while True:
        data = datastore.scan_data()
        if data is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        row = {}
        for item in record.history.item:
            key = history_item_key(item)
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                continue

        step = row.get("_step")
        value = row.get(metric)
        if step is None or value is None:
            continue
        if not isinstance(step, (int, float)) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(step)) or not math.isfinite(float(value)):
            continue
        values_by_step[int(step)] = float(value)

    if not values_by_step:
        raise ValueError(f"Metric '{metric}' was not found in {wandb_file}")

    ordered = sorted(values_by_step.items())
    iterations = np.asarray([step for step, _ in ordered], dtype=np.int64)
    values = np.asarray([value for _, value in ordered], dtype=np.float64)
    return iterations, values


def moving_average(
    iterations: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1:
        return iterations, values

    window = min(window, values.size)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    return iterations[window - 1 :], averaged


def plot_return(
    run_dir: Path,
    iterations: np.ndarray,
    values: np.ndarray,
    metric: str,
    window: int,
    output: Path,
) -> None:
    smooth_iterations, smooth_values = moving_average(iterations, values, window)

    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    axis.plot(iterations, values, color="#6b7280", alpha=0.28, linewidth=1.0, label="Raw")
    if window > 1:
        effective_window = min(window, values.size)
        axis.plot(
            smooth_iterations,
            smooth_values,
            color="#1261a0",
            linewidth=2.2,
            label=f"Moving average ({effective_window})",
        )

    axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.45)
    axis.set_title(f"{run_dir.name}: Episode Return")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(metric)
    axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False)
    axis.margins(x=0.01)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"[INFO] W&B points: {values.size} (iteration {iterations[0]} to {iterations[-1]})")
    print(f"[INFO] Last return: {values[-1]:.6f}")
    if window > 1:
        print(f"[INFO] Last moving average: {smooth_values[-1]:.6f}")
    print(f"[INFO] Plot saved: {output}")


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be at least 1")

    run_dir = resolve_run_dir(args.run_dir)
    wandb_file = find_wandb_file(run_dir)
    iterations, values = load_metric(wandb_file, args.metric)
    output = (
        args.output.expanduser()
        if args.output is not None
        else PROJECT_ROOT / "figure" / f"{run_dir.name}_iteration_return.png"
    )
    output = output.resolve()
    plot_return(run_dir, iterations, values, args.metric, args.window, output)


if __name__ == "__main__":
    main()
