
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .controller import LVController

if TYPE_CHECKING:
    from .velocity_controller_cfg import VelocityControllerCfg


class VelocityController:
    """Bridge world-frame velocity actions to LVController."""

    def __init__(
        self,
        robot,
        cfg: "VelocityControllerCfg",
        num_envs: int,
        device: str,
        dt: float,
    ) -> None:
        self.robot = robot
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        self.raw_actions = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.command = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self.last_thrust_torque = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self.yaw_cmd = torch.zeros((num_envs,), device=device, dtype=torch.float32)
        self.yaw_initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

        self.body_ids, self.body_names = self._resolve_control_body()
        self.body_id = self.body_ids[0]
        self.force_body = torch.zeros((num_envs, 1, 3), device=device, dtype=torch.float32)
        self.torque_body = torch.zeros((num_envs, 1, 3), device=device, dtype=torch.float32)

        robot_mass = self._read_total_mass()
        robot_inertia = self._read_or_create_inertia()
        self.controller = LVController(
            cfg=self.cfg.lv_controller_cfg,
            num_envs=num_envs,
            device=device,
            mass=robot_mass,
            inertia=robot_inertia,
            dt=dt,
        )
        self.max_total_thrust, self.max_body_torque = self._compute_wrench_limits()

    def apply_action(self, actions_w: torch.Tensor) -> None:
        """Apply a world-frame velocity action `[vx, vy, vz]`."""

        actions_w = actions_w.to(self.device)
        if actions_w.ndim != 2:
            actions_w = actions_w.reshape(self.num_envs, -1)
        if actions_w.shape[-1] != 3:
            raise RuntimeError(
                f"Expected action dimension 3 for [vx, vy, vz], got {tuple(actions_w.shape)}."
            )

        vel_cmd_w = torch.nan_to_num(actions_w, nan=0.0, posinf=0.0, neginf=0.0)
        self.raw_actions[:] = vel_cmd_w

        yaw_cmd = self._yaw_from_velocity(vel_cmd_w)
        self.command[:, 0] = yaw_cmd
        self.command[:, 1:4] = vel_cmd_w
        self.command[:] = torch.nan_to_num(self.command, nan=0.0, posinf=0.0, neginf=0.0)

        current_state = self._get_controller_state()
        _, thrust_torque = self.controller.compute(current_state, self.command.detach())
        thrust_torque = torch.nan_to_num(thrust_torque, nan=0.0, posinf=0.0, neginf=0.0)
        thrust_torque = self._clamp_wrench(thrust_torque)
        self.last_thrust_torque[:] = thrust_torque.detach()

        self.force_body.zero_()
        self.torque_body.zero_()
        self.force_body[:, 0, 2] = thrust_torque[:, 0]
        self.torque_body[:, 0, :] = thrust_torque[:, 1:4]
        self.force_body[:] = torch.nan_to_num(self.force_body, nan=0.0, posinf=0.0, neginf=0.0)
        self.torque_body[:] = torch.nan_to_num(self.torque_body, nan=0.0, posinf=0.0, neginf=0.0)

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self.force_body,
            torques=self.torque_body,
            body_ids=self.body_ids,
        )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self.controller.reset_idx(env_ids)
        hover_thrust = self.controller.mass[env_ids] * self.controller.g_norm
        self.controller.gross_thrust[env_ids] = hover_thrust
        self.raw_actions[env_ids] = 0.0
        self.command[env_ids] = 0.0
        self.last_thrust_torque[env_ids] = 0.0
        self.yaw_cmd[env_ids] = self._current_yaw_w()[env_ids]
        self.yaw_initialized[env_ids] = True
        self.force_body[env_ids] = 0.0
        self.torque_body[env_ids] = 0.0
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self.force_body[env_ids],
            torques=self.torque_body[env_ids],
            body_ids=self.body_ids,
            env_ids=env_ids,
        )

    def _yaw_from_velocity(self, vel_cmd_w: torch.Tensor) -> torch.Tensor:
        current_yaw = self._current_yaw_w()
        uninitialized = ~self.yaw_initialized
        if uninitialized.any():
            self.yaw_cmd[uninitialized] = current_yaw[uninitialized]
            self.yaw_initialized[uninitialized] = True

        if self.cfg.yaw_mode == "hold":
            self.yaw_cmd[:] = current_yaw
            return self.yaw_cmd
        if self.cfg.yaw_mode != "velocity_vector":
            raise ValueError(f"Unsupported yaw_mode: {self.cfg.yaw_mode}")

        horizontal_speed = torch.linalg.norm(vel_cmd_w[:, :2], dim=-1)
        yaw_from_vel = torch.atan2(vel_cmd_w[:, 1], vel_cmd_w[:, 0])
        target_yaw = torch.where(horizontal_speed > self.cfg.yaw_from_velocity_threshold, yaw_from_vel, self.yaw_cmd)
        yaw_error = torch.atan2(torch.sin(target_yaw - self.yaw_cmd), torch.cos(target_yaw - self.yaw_cmd))
        max_step = torch.full_like(yaw_error, float(self.cfg.yaw_rate_limit) * self.dt)
        yaw_step = torch.clamp(yaw_error, min=-max_step, max=max_step)
        self.yaw_cmd[:] = torch.atan2(torch.sin(self.yaw_cmd + yaw_step), torch.cos(self.yaw_cmd + yaw_step))
        return self.yaw_cmd

    def _current_yaw_w(self) -> torch.Tensor:
        quat = self.robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _get_controller_state(self) -> dict[str, torch.Tensor]:
        root_state_w = self.robot.data.root_state_w
        quat = root_state_w[:, 3:7]
        lin_vel_w = root_state_w[:, 7:10]
        ang_vel_w = root_state_w[:, 10:13]

        return {
            "pos": root_state_w[:, 0:3].detach(),
            "quat": quat.detach(),
            "lin_vel_w": lin_vel_w.detach(),
            "ang_vel_w": ang_vel_w.detach(),
            "lin_vel_b": math_utils.quat_apply_inverse(quat, lin_vel_w).detach(),
            "ang_vel_b": math_utils.quat_apply_inverse(quat, ang_vel_w).detach(),
        }

    def _compute_wrench_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        rotor_thrust_max = max(0.0, float(self.controller.thrust_max))
        max_total_thrust = torch.tensor(
            [4.0 * rotor_thrust_max],
            device=self.device,
            dtype=torch.float32,
        )
        arm_eff = float(self.cfg.lv_controller_cfg.arm_length) * math.sqrt(0.5)
        max_roll_pitch = 2.0 * arm_eff * rotor_thrust_max
        max_yaw = 2.0 * float(self.cfg.lv_controller_cfg.kappa) * rotor_thrust_max
        max_body_torque = torch.tensor(
            [[max_roll_pitch, max_roll_pitch, max_yaw]],
            device=self.device,
            dtype=torch.float32,
        )
        return max_total_thrust, max_body_torque

    def _clamp_wrench(self, thrust_torque: torch.Tensor) -> torch.Tensor:
        thrust_torque = thrust_torque.clone()
        thrust_torque[:, :1] = thrust_torque[:, :1].clamp(0.0, self.max_total_thrust[0])
        thrust_torque[:, 1:4] = torch.max(
            torch.min(thrust_torque[:, 1:4], self.max_body_torque),
            -self.max_body_torque,
        )
        return thrust_torque

    def _resolve_control_body(self):
        for body_name in (self.cfg.body_name, "body", "base"):
            try:
                body_ids, body_names = self.robot.find_bodies(body_name, preserve_order=True)
                if len(body_ids) > 0:
                    return [body_ids[0]], [body_names[0]]
            except Exception:
                continue
        return [0], [self.robot.body_names[0]]

    def _read_total_mass(self) -> torch.Tensor:
        masses = self.robot.root_physx_view.get_masses().to(self.device)
        if masses.ndim == 1:
            mass = masses.sum().reshape(1, 1).repeat(self.num_envs, 1)
        else:
            mass = masses.sum(dim=1, keepdim=True)
        return mass.clamp_min(1.0e-6)

    def _read_or_create_inertia(self) -> torch.Tensor:
        inertia = torch.diag(
            torch.tensor(self.cfg.inertia_diag, device=self.device, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)

        if not self.cfg.use_physx_inertia:
            return inertia

        try:
            inertias = self.robot.root_physx_view.get_inertias().to(self.device)
            body_id = self.body_id
            if inertias.ndim == 2:
                inertia_raw = inertias[body_id].reshape(1, -1).repeat(self.num_envs, 1)
            else:
                inertia_raw = inertias[:, body_id]

            if inertia_raw.shape[-1] == 9:
                return inertia_raw.reshape(self.num_envs, 3, 3).to(dtype=torch.float32)
            if inertia_raw.shape[-1] == 3:
                return torch.diag_embed(inertia_raw.to(dtype=torch.float32))
        except Exception:
            pass

        return inertia
