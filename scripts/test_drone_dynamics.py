"""Fly the project quadcopter around a rectangular waypoint trajectory.

Usage:
    conda activate env_isaaclab
    source /home/robot/IsaacLab/_isaac_sim/setup_conda_env.sh
    python scripts/test_drone_dynamics.py --headless
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Fly the project quadcopter around a rectangular waypoint trajectory.")
parser.add_argument("--hover_duration_s", type=float, default=3.0, help="Duration of the zero-velocity hover phase.")
parser.add_argument("--forward_distance_m", type=float, default=10.0, help="Rectangle length along world +X in meters.")
parser.add_argument("--left_distance_m", type=float, default=5.0, help="Rectangle width along world +Y in meters.")
parser.add_argument("--cruise_speed_mps", type=float, default=0.8, help="World-frame waypoint tracking speed in m/s.")
parser.add_argument("--waypoint_tolerance_m", type=float, default=0.20, help="Planar distance required to reach a waypoint.")
parser.add_argument("--corner_settle_s", type=float, default=0.25, help="Zero-velocity settling time at each waypoint.")
parser.add_argument("--leg_timeout_scale", type=float, default=2.5, help="Maximum leg duration as a multiple of nominal travel time.")
parser.add_argument("--sim_dt", type=float, default=1.0 / 120.0, help="Physics timestep in seconds.")
parser.add_argument("--strict", action="store_true", help="Exit with a non-zero status when a dynamics criterion fails.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from nav.assets import DRONE_CFG
from nav.controllers import VelocityController, VelocityControllerCfg


@dataclass
class PhaseMetrics:
    name: str
    samples: torch.Tensor

    @property
    def final_position(self) -> torch.Tensor:
        return self.samples[-1, 0:3]

    @property
    def final_velocity(self) -> torch.Tensor:
        return self.samples[-1, 3:6]


@dataclass
class WaypointMetrics:
    name: str
    target_position: torch.Tensor
    samples: torch.Tensor
    reached: bool
    elapsed_s: float

    @property
    def final_position(self) -> torch.Tensor:
        return self.samples[-1, 0:3]

    @property
    def position_error(self) -> float:
        return torch.linalg.vector_norm(self.final_position[:2] - self.target_position[:2]).item()


def _spawn_scene() -> tuple[SimulationContext, Articulation]:
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim_cfg.physx.enable_external_forces_every_iteration = True
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 5.0, 3.5], target=[0.0, 0.0, 1.5])

    ground_cfg = sim_utils.GroundPlaneCfg(size=(20.0, 20.0))
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = DRONE_CFG.replace(prim_path="/World/Drone")
    robot = Articulation(robot_cfg)
    sim.reset()
    print("[INFO] Simulation reset complete.", flush=True)
    return sim, robot


def _reset_robot(robot: Articulation, controller: VelocityController) -> None:
    root_state = robot.data.default_root_state.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    controller.reset_idx(torch.arange(robot.num_instances, device=robot.device))


def _sample_root_state(robot: Articulation) -> torch.Tensor:
    root_state = robot.data.root_state_w[0]
    return torch.cat((root_state[0:3], root_state[7:10], root_state[10:13], root_state[3:7])).detach().cpu()


def _step_simulation(
    sim: SimulationContext,
    robot: Articulation,
    controller: VelocityController,
    command: torch.Tensor,
) -> torch.Tensor:
    controller.apply_action(command)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    return _sample_root_state(robot)


def _hold_velocity(
    sim: SimulationContext,
    robot: Articulation,
    controller: VelocityController,
    name: str,
    duration_s: float,
    velocity_command: tuple[float, float, float],
) -> PhaseMetrics:
    steps = max(1, round(duration_s / sim.get_physics_dt()))
    command = torch.tensor(velocity_command, device=sim.device, dtype=torch.float32).unsqueeze(0)
    samples: list[torch.Tensor] = []

    for _ in range(steps):
        samples.append(_step_simulation(sim, robot, controller, command))

    print(f"[INFO] Completed {name} phase ({steps} physics steps).", flush=True)
    return PhaseMetrics(name=name, samples=torch.stack(samples))


def _fly_to_waypoint(
    sim: SimulationContext,
    robot: Articulation,
    controller: VelocityController,
    name: str,
    target_position: torch.Tensor,
) -> WaypointMetrics:
    position = robot.data.root_state_w[0, 0:3]
    planar_distance = torch.linalg.vector_norm(target_position[:2] - position[:2]).item()
    nominal_duration = planar_distance / args_cli.cruise_speed_mps
    max_steps = max(1, round(max(1.0, nominal_duration * args_cli.leg_timeout_scale) / sim.get_physics_dt()))
    zero_command = torch.zeros((1, 3), device=sim.device)
    samples: list[torch.Tensor] = []
    reached = False

    print(f"[INFO] Flying {name} to {target_position.tolist()}.", flush=True)
    for step in range(max_steps):
        position = robot.data.root_state_w[0, 0:3]
        error = target_position - position
        planar_error = torch.linalg.vector_norm(error[:2])
        if planar_error.item() <= args_cli.waypoint_tolerance_m:
            reached = True
            break

        command = torch.zeros((1, 3), device=sim.device)
        command[0, :2] = args_cli.cruise_speed_mps * error[:2] / planar_error
        command[0, 2] = torch.clamp(2.0 * error[2], min=-0.5, max=0.5)
        samples.append(_step_simulation(sim, robot, controller, command))

    settle_steps = max(1, round(args_cli.corner_settle_s / sim.get_physics_dt()))
    for _ in range(settle_steps):
        samples.append(_step_simulation(sim, robot, controller, zero_command))

    elapsed_s = (len(samples) if reached else max_steps + settle_steps) * sim.get_physics_dt()
    metrics = WaypointMetrics(
        name=name,
        target_position=target_position.detach().cpu(),
        samples=torch.stack(samples),
        reached=reached,
        elapsed_s=elapsed_s,
    )
    print(
        f"[INFO] {name}: reached={metrics.reached}, error={metrics.position_error:.3f} m, "
        f"duration={metrics.elapsed_s:.2f} s.",
        flush=True,
    )
    return metrics


def _validate(hover: PhaseMetrics, legs: list[WaypointMetrics], initial_position: torch.Tensor) -> list[str]:
    failures: list[str] = []
    all_samples = torch.cat([hover.samples, *(leg.samples for leg in legs)])
    if not torch.isfinite(all_samples).all():
        failures.append("non-finite root state encountered")

    hover_altitude_error = torch.max(torch.abs(hover.samples[:, 2] - hover.samples[0, 2])).item()
    hover_vertical_speed = torch.abs(hover.final_velocity[2]).item()
    final_position_error = torch.linalg.vector_norm(legs[-1].final_position[:2] - initial_position[:2]).item()

    print("\n[RESULT] Dynamics summary", flush=True)
    print(f"  hover altitude drift: {hover_altitude_error:.3f} m", flush=True)
    print(f"  hover final vertical speed: {hover_vertical_speed:.3f} m/s", flush=True)
    for leg in legs:
        print(f"  {leg.name}: reached={leg.reached}, planar error={leg.position_error:.3f} m", flush=True)
    print(f"  return-to-origin error: {final_position_error:.3f} m", flush=True)

    if hover_altitude_error > 0.75:
        failures.append(f"hover altitude drift exceeded 0.75 m ({hover_altitude_error:.3f} m)")
    if hover_vertical_speed > 2.0:
        failures.append(f"hover vertical speed exceeded 2.0 m/s ({hover_vertical_speed:.3f} m/s)")
    for leg in legs:
        if not leg.reached:
            failures.append(f"{leg.name} did not reach its waypoint")
        elif leg.position_error > args_cli.waypoint_tolerance_m:
            failures.append(f"{leg.name} settled outside the waypoint tolerance ({leg.position_error:.3f} m)")
    if final_position_error > args_cli.waypoint_tolerance_m:
        failures.append(f"return-to-origin error exceeded tolerance ({final_position_error:.3f} m)")
    return failures


def main() -> int:
    sim, robot = _spawn_scene()
    print("[INFO] Creating velocity controller.", flush=True)
    controller = VelocityController(
        robot=robot,
        cfg=VelocityControllerCfg(),
        num_envs=robot.num_instances,
        device=sim.device,
        dt=sim.get_physics_dt(),
    )
    print("[INFO] Drone articulation loaded", flush=True)
    print(f"  bodies: {robot.body_names}", flush=True)
    print(f"  joints: {robot.joint_names}", flush=True)
    print(f"  total mass: {robot.root_physx_view.get_masses().sum().item():.4f} kg", flush=True)
    print(f"  control body: {controller.body_names[0]} (id={controller.body_id})", flush=True)

    _reset_robot(robot, controller)
    print("[INFO] Robot reset complete; starting hover phase.", flush=True)
    hover = _hold_velocity(sim, robot, controller, "hover", args_cli.hover_duration_s, (0.0, 0.0, 0.0))
    initial_position = robot.data.root_state_w[0, 0:3].detach().clone()
    x_side = torch.tensor([args_cli.forward_distance_m, 0.0, 0.0], device=sim.device)
    y_side = torch.tensor([0.0, args_cli.left_distance_m, 0.0], device=sim.device)
    waypoints = [
        ("forward (+X)", initial_position + x_side),
        ("left (+Y)", initial_position + x_side + y_side),
        ("backward (-X)", initial_position + y_side),
        ("return (-Y)", initial_position),
    ]
    legs: list[WaypointMetrics] = []
    for name, target_position in waypoints:
        leg = _fly_to_waypoint(sim, robot, controller, name, target_position)
        legs.append(leg)
        if not leg.reached:
            break
    failures = _validate(hover, legs, initial_position.detach().cpu())

    if failures:
        print("[FAIL] " + "; ".join(failures), flush=True)
        return 1
    print("[PASS] UAV dynamics smoke test passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    if args_cli.strict and exit_code:
        sys.exit(exit_code)
