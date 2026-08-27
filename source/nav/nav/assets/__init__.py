# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local UAV assets."""

from .quadcopter import DRONE_CFG, DRONE_NO_COLLIDER_CFG
from .dynamic import (
    GlobalObstacleManager,
    GlobalObstacleMotionCfg,
    GlobalRigidObjectCollection,
    get_global_obstacle_manager,
    has_scene_entity,
    make_global_obstacle_collection_cfg,
)

__all__ = [
    "DRONE_CFG",
    "DRONE_NO_COLLIDER_CFG",
    "GlobalObstacleManager",
    "GlobalObstacleMotionCfg",
    "GlobalRigidObjectCollection",
    "get_global_obstacle_manager",
    "has_scene_entity",
    "make_global_obstacle_collection_cfg",
]
