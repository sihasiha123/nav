
"""单步导航观测：机体系自身状态、相对目标与前视深度图。"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .commands import get_nav_target_command

__all__ = ["depth_obs", "goal_obs", "state_obs"]


def _body_link_state_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """读取配置指定的唯一机身 link，返回世界系状态 ``(N, 13)``。"""
    robot = env.scene[asset_cfg.name]
    # body_ids 由 Manager 根据 body_names 解析，可以是列表或 slice。
    body_state = robot.data.body_link_state_w[:, asset_cfg.body_ids, :]
    if body_state.shape[1] != 1:
        raise ValueError("Navigation observations require exactly one body selected by asset_cfg.body_names.")
    # 位置、姿态及速度均对应 link 原点，避免混用根节点或质心状态。
    return body_state[:, 0, :].to(dtype=torch.float32)


def state_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """自身状态 ``(N, 9)``，float32，保留在环境设备上。

    顺序为机体系线速度 [0:3]（m/s）、重力单位方向 [3:6]、
    机体系角速度 [6:9]（rad/s）。速度不做缩放，无历史帧。
    机体系跟随所选 body 的完整姿态，包括横滚、俯仰和偏航。
    """
    body_state = _body_link_state_w(env, asset_cfg)
    quat_w = body_state[:, 3:7]
    lin_vel_b = quat_apply_inverse(quat_w, body_state[:, 7:10])
    ang_vel_b = quat_apply_inverse(quat_w, body_state[:, 10:13])

    # 这是仿真重力的单位方向，不是加速度计读数；使用 body 的姿态投影。
    gravity_w = env.scene[asset_cfg.name].data.GRAVITY_VEC_W.to(dtype=torch.float32)
    gravity_b = quat_apply_inverse(quat_w, gravity_w)
    return torch.cat((lin_vel_b, gravity_b, ang_vel_b), dim=-1)


def goal_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """相对目标 ``(N, 4)``：机体系单位方向 [0:3] + 三维距离 [3:4]（m）。

    世界系目标仍由 CommandManager 保存；这里只读取其位置部分。
    目标与 body 原点重合时方向为零。输出 float32，不缩放距离。
    """
    body_state = _body_link_state_w(env, asset_cfg)
    target_pos_w = get_nav_target_command(env)[:, :3].to(dtype=torch.float32)
    relative_pos_w = target_pos_w - body_state[:, :3]
    distance = torch.linalg.vector_norm(relative_pos_w, dim=-1, keepdim=True)
    relative_pos_b = quat_apply_inverse(body_state[:, 3:7], relative_pos_w)
    # 零距离使用分母 1，避免除零；非零距离保留真正的单位方向。
    direction_b = relative_pos_b / torch.where(distance > 0.0, distance, torch.ones_like(distance))
    return torch.cat((direction_b, distance), dim=-1)


def depth_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """单帧深度 ``(N, 1, H, W)``，当前为 ``(N, 1, 54, 96)``。

    读取沿相机光轴的深度，以相机 max_distance（当前 4m）归一化。
    输出 float32、范围 [0, 1]，越远数值越大；1 是通道数而非历史帧数。
    """
    camera = env.scene[asset_cfg.name]
    max_distance = float(camera.cfg.max_distance)
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("Depth normalization requires a finite, positive camera max_distance.")

    depth = camera.data.output["distance_to_image_plane"].to(dtype=torch.float32)
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"Expected camera depth shape (N, H, W, 1), received {tuple(depth.shape)}.")
    # 传感器已负责超量程裁剪；这里作轻量兜底，不修改相机原始缓冲区。
    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    depth = depth.clamp(0.0, max_distance) / max_distance
    return depth.permute(0, 3, 1, 2).contiguous()
