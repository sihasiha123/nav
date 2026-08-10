
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg

from isaaclab.utils import configclass

from . import mdp

##
# 预定义配置
##

from nav.assets.quadcopter import DRONE_CFG


##
# 场景定义
##


class GlobalRigidObjectCollection(RigidObjectCollection):
    """不会随单个并行环境重置的全局刚体集合。

    使用绝对 ``/World`` 路径生成的集合只有一个实例，而导航任务包含多个机器人
    环境。Manager-based 环境重置时会把机器人的 ``env_ids`` 传给所有场景实体，
    因此不能继续把这些索引传给全局单例集合。
    """

    def reset(self, env_ids=None, object_ids=None) -> None:
        """忽略单环境重置，仅响应显式的全局重置。"""
        if env_ids is None:
            super().reset(env_ids=None, object_ids=object_ids)


def make_global_obstacle_collection_cfg(
    count: int = 100,
    terrain_size: tuple[float, float] = (40.0, 40.0),
    margin: float = 2.0,
    obstacle_size: tuple[float, float, float] = (0.5, 0.5, 1.0),
    obstacle_height: float = 1.5,
) -> RigidObjectCollectionCfg:
    """按照近似正方形网格创建一套全局障碍物集合。

    配置中的初始位置同时作为运动原点，后续运行时管理器可以从
    ``default_object_state`` 中读取这些位置。
    """
    if count <= 0:
        raise ValueError(f"Obstacle count must be positive, received: {count}.")
    if margin < 0.0:
        raise ValueError(f"Obstacle margin must be non-negative, received: {margin}.")

    terrain_width, terrain_length = terrain_size
    usable_width = terrain_width - 2.0 * margin
    usable_length = terrain_length - 2.0 * margin
    if usable_width <= 0.0 or usable_length <= 0.0:
        raise ValueError("Obstacle margin leaves no usable terrain area.")

    num_cols = math.ceil(math.sqrt(count))
    num_rows = math.ceil(count / num_cols)
    cell_width = usable_width / num_cols
    cell_length = usable_length / num_rows

    obstacle_cfgs: dict[str, RigidObjectCfg] = {}
    for obstacle_index in range(count):
        row, col = divmod(obstacle_index, num_cols)
        x = -0.5 * terrain_width + margin + (col + 0.5) * cell_width
        y = -0.5 * terrain_length + margin + (row + 0.5) * cell_length

        obstacle_name = f"obstacle_{obstacle_index:03d}"
        obstacle_cfgs[obstacle_name] = RigidObjectCfg(
            prim_path=f"/World/Dynamic/Obstacle_{obstacle_index:03d}",
            spawn=sim_utils.CuboidCfg(
                size=obstacle_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, obstacle_height)),
            collision_group=-1,
        )

    return RigidObjectCollectionCfg(
        class_type=GlobalRigidObjectCollection,
        rigid_objects=obstacle_cfgs,
    )


@configclass
class NavSceneCfg(InteractiveSceneCfg):
    """共享地图无人机导航场景配置。"""

    # 地面平面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0)),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(40.0, 40.0),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            color_scheme="height",
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=30,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(6.0, 6.0),
                    platform_width=0.0,
                ),
            },
        ),
        visual_material=None,
        collision_group=-1,
        debug_vis=True,
        use_terrain_origins=False,
    )

    # 无人机
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 所有机器人环境共享同一套全局障碍物；接入运行时运动管理器前保持静止。
    dynamic_obstacles: RigidObjectCollectionCfg = make_global_obstacle_collection_cfg()

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP 配置
##


@configclass
class ActionsCfg:
    """MDP 的动作项配置。"""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """MDP 的观测项配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组。"""

        # 观测项（保持此处声明的顺序）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件项配置。"""

    # 重置事件
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """MDP 的奖励项配置。"""

    # （1）恒定存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # （2）失败惩罚
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # （3）主要任务：保持杆直立
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # （4）塑形任务：降低小车速度
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # （5）塑形任务：降低杆的角速度
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """MDP 的终止项配置。"""

    # （1）超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # （2）小车越界
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# 环境配置
##


@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    # 场景配置
    scene: NavSceneCfg = NavSceneCfg(
        num_envs=1024,
        env_spacing=0.0,
        replicate_physics=True,
        filter_collisions=True,
    )
    # 基础配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP 配置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 后初始化
    def __post_init__(self) -> None:
        """完成环境配置的后初始化。"""
        # 通用配置
        self.decimation = 2
        self.episode_length_s = 5
        # 查看器配置
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 仿真配置
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
