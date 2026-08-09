"""Configuration for the linear-velocity + yaw quadrotor controller."""

from __future__ import annotations

from isaaclab.utils import configclass

from .controller import LVController


@configclass
class LVControllerCfg:
    """Linear-velocity + yaw controller configuration.

    Command format: ``[yaw_cmd, vx_cmd, vy_cmd, vz_cmd]``.
    """

    class_type: type = LVController

    # Physical parameters
    arm_length: float = 0.09
    kappa: float = 0.016
    motor_omega: tuple[float, float] = (150.0, 3000.0)

    # Static thrust map: f_i = k2 * omega^2 + k1 * omega + k0
    thrustmap: list[float] = [
        1.3298253500372892e-06,
        0.0038360810526746033,
        -1.7689986848125325,
    ]

    g: float = 9.81

    # Control gains and limits
    max_feedback_accel: float = 20.0
    body_rate_bound: list[float] = [-12.0, 12.0]
    speed_gain: list[float] = [10.0, 10.0, 20.0]
    pose_gain: list[float] = [18.0, 18.0, 20.0]
    rate_gain: list[float] = [180.0, 180.0, 200.0]

    # Thrust delay
    thrust_ctrl_delay: float = 0.03
