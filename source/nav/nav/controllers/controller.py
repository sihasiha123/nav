"""Linear-velocity + yaw controller for the quadrotor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from .controller_cfg import LVControllerCfg


class LVController:
    """Linear velocity and yaw loop controller.

    Command format: ``[yaw_cmd, vx_cmd, vy_cmd, vz_cmd]``.
    Output format: ``[total_thrust, torque_x, torque_y, torque_z]``.
    """

    def __init__(
        self,
        cfg: LVControllerCfg,
        num_envs: int,
        device: str,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        dt: float,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = torch.inverse(inertia)

        # gravity and gains
        self.g = torch.tensor([[0.0, 0.0, -float(cfg.g)]], device=self.device)
        self.g_norm = self.g.norm()
        self.speed_gain = torch.tensor(cfg.speed_gain, device=self.device).repeat(self.num_envs, 1)
        self.pose_gain = torch.tensor(cfg.pose_gain, device=self.device).repeat(self.num_envs, 1)
        self.rate_gain = torch.tensor(cfg.rate_gain, device=self.device).repeat(self.num_envs, 1)

        # thrust limits from the static thrust map
        self.thrust_map = cfg.thrustmap
        self.thrust_max = (
            self.thrust_map[0] * cfg.motor_omega[1] ** 2
            + self.thrust_map[1] * cfg.motor_omega[1]
            + self.thrust_map[2]
        )
        self.thrust_min = (
            self.thrust_map[0] * cfg.motor_omega[0] ** 2
            + self.thrust_map[1] * cfg.motor_omega[0]
            + self.thrust_map[2]
        )
        self.gross_thrust_bound = [self.thrust_min * 4, self.thrust_max * 4]
        self.body_rate_bound = cfg.body_rate_bound

        # thrust delay state
        self.gross_thrust = torch.zeros(self.num_envs, 1, device=self.device)
        self.thrust_ctrl_delay = torch.ones(self.num_envs, 1, device=self.device) * cfg.thrust_ctrl_delay

        # state buffers
        self.pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

    def update_state(self, states: dict[str, torch.Tensor]) -> None:
        self.pos = states["pos"]
        self.quat = states["quat"]
        self.lin_vel_w = states["lin_vel_w"]
        self.ang_vel_w = states["ang_vel_w"]
        self.lin_vel_b = states["lin_vel_b"]
        self.ang_vel_b = states["ang_vel_b"]

    def compute(self, now_state: dict[str, torch.Tensor], cmd: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Compute the wrench ``[total_thrust, torque_x, torque_y, torque_z]``."""
        self.update_state(now_state)

        cmd_speed = cmd[:, 1:]
        cmd_yaw = cmd[:, :1]

        # -- linear velocity loop
        err_speed = cmd_speed - self.lin_vel_w
        acc_fb = torch.min(
            torch.norm(self.speed_gain * err_speed, dim=-1, keepdim=True),
            torch.tensor(self.cfg.max_feedback_accel, device=self.device),
        ) * F.normalize(err_speed, p=2, dim=1)
        des_F = self.mass * (acc_fb - self.g)
        gross_thrust_des = math_utils.quat_apply_inverse(self.quat, des_F)[:, 2:]

        # -- attitude loop (compute desired body rates)
        R = math_utils.matrix_from_quat(self.quat)
        R_T = R.transpose(1, 2)
        b1_des = torch.cat(
            [torch.cos(cmd_yaw), torch.sin(cmd_yaw), torch.zeros_like(cmd_yaw)],
            dim=-1,
        )
        b3_des = F.normalize(des_F, p=2, dim=1)
        b2_des = F.normalize(torch.cross(b3_des, b1_des, dim=1), p=2, dim=1)
        R_des = torch.stack([b2_des.cross(b3_des, 1), b2_des, b3_des], dim=-1)
        R_des_T = R_des.transpose(1, 2)
        m = 0.5 * (torch.bmm(R_des_T, R) - torch.bmm(R_T, R_des))
        pose_err = -torch.stack((-m[:, 1, 2], m[:, 0, 2], -m[:, 0, 1]), dim=1)
        bodyrate_des = self.pose_gain * pose_err

        # -- thrust delay
        gross_thrust_des = gross_thrust_des.clamp(
            float(self.gross_thrust_bound[0]),
            float(self.gross_thrust_bound[1]),
        )
        self.gross_thrust = (
            (1 - torch.exp(-self.dt / self.thrust_ctrl_delay)) * gross_thrust_des
            + torch.exp(-self.dt / self.thrust_ctrl_delay) * self.gross_thrust
        )

        # -- angular rate loop
        bodyrate_des = bodyrate_des.clamp(
            float(self.body_rate_bound[0]),
            float(self.body_rate_bound[1]),
        )
        err_rate = bodyrate_des - self.ang_vel_b
        torque_des = (
            self.inertia @ (self.rate_gain * err_rate)[..., None]
        ).squeeze(-1) + torch.linalg.cross(
            self.ang_vel_b,
            (self.inertia @ self.ang_vel_b[..., None]).squeeze(-1),
        )

        thrust_torque_des = torch.cat((self.gross_thrust, torque_des), dim=1)
        return None, thrust_torque_des

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == self.num_envs:
            self.gross_thrust = self.gross_thrust.detach()
            self.gross_thrust.zero_()
            return
        clone = self.gross_thrust.clone()
        clone[env_ids] = clone[env_ids].detach()
        clone[env_ids] = torch.zeros_like(clone[env_ids])
        self.gross_thrust = clone
