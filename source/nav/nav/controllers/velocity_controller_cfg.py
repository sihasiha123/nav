"""Config for the env-facing velocity action adapter."""

from __future__ import annotations

from isaaclab.utils import configclass

from .controller_cfg import LVControllerCfg


@configclass
class VelocityControllerCfg:
    """Convert world-frame velocity actions into LVController commands."""

    lv_controller_cfg: LVControllerCfg = LVControllerCfg()
    body_name: str = "body"
    inertia_diag: tuple[float, float, float] = (0.0015, 0.0020, 0.0040)
    use_physx_inertia: bool = False
    yaw_mode: str = "velocity_vector"  # "hold" or "velocity_vector"
    yaw_from_velocity_threshold: float = 1.0e-3
    yaw_rate_limit: float = 4.0
