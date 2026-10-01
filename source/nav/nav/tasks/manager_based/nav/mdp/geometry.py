"""环境内部的球体碰撞工具，不向策略提供场景真值。

这是环境步末的离散几何检测，不检测两个采样时刻之间穿越障碍物的情况。
"""

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def body_position_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """读取唯一机身 link 的世界系位置，形状 (N, 3)。"""
    positions = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, :]
    if positions.shape[1] != 1:
        raise ValueError("Collision checks require exactly one selected body.")
    return positions[:, 0, :]


def spheres_touch_boxes(centers_w, poses_w, half_sizes, radius: float) -> torch.Tensor:
    """N 个球与 M 个有向方块检测，返回 (N, M)，方块四元数使用 wxyz。"""
    relative_w = centers_w[:, None, :] - poses_w[None, :, :3]
    rotations = poses_w[None, :, 3:7].expand(centers_w.shape[0], -1, -1)
    relative_b = quat_apply_inverse(rotations, relative_w)
    outside = (relative_b.abs() - half_sizes[None, :, :]).clamp_min(0.0)
    return outside.square().sum(dim=-1) <= radius * radius


@wp.kernel(enable_backward=False)
def heightfield_sphere_collision(
    mesh_id: wp.uint64,
    centers: wp.array(dtype=wp.vec3),
    radius: float,
    ray_top: float,
    ground_height: float,
    result: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = centers[i]
    hit = int(0)

    # 查询附近三角面，处理地形的顶面、侧面、边和角。
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    if wp.mesh_query_point_no_sign(mesh_id, p, radius + 1.0e-6, face, u, v):
        closest = wp.mesh_eval_position(mesh_id, face, u, v)
        if wp.length(closest - p) <= radius:
            hit = 1

    # 高度场不是封闭网格：从最高点上方向下查询表面高度，识别球心已陷入实体。
    # 射线延伸到球心即可；球心以下的表面不会触发内部判定。
    if p[2] < ray_top:
        ray = wp.mesh_query_ray(
            mesh_id, wp.vec3(p[0], p[1], ray_top), wp.vec3(0.0, 0.0, -1.0), ray_top - p[2]
        )
        if ray.result:
            hit = 1

    # 当前独立水平地面覆盖整个允许飞行区域，也检测球体部分接触地面的情况。
    if p[2] - radius <= ground_height:
        hit = 1
    result[i] = hit
