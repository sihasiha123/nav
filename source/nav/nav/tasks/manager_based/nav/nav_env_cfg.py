
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg

from isaaclab.utils import configclass

from . import mdp

##
# 预定义配置
##

from nav.assets.quadcopter import DRONE_NO_COLLIDER_CFG
from nav.assets.dynamic import make_global_obstacle_collection_cfg


##
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
                    num_obstacles=200,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(6.0, 6.0),
                    platform_width=0.0,
                ),
            },
        ),
        visual_material=None,
        collision_group=-1,
        debug_vis=False,
        use_terrain_origins=False,
    )

    # 无人机
    robot: ArticulationCfg = DRONE_NO_COLLIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 激光雷达：扫描共享地形，量程 4m，36 水平 x 4 垂直光束
    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=4,
            vertical_fov_range=(-10.0, 20.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=10.0,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 所有机器人环境共享同一套全局障碍物；中心高度在无人机飞行范围内随机；
    # count=0 时返回 None（禁用）
    dynamic_obstacles: RigidObjectCollectionCfg | None = make_global_obstacle_collection_cfg(
        count=100,
        obstacle_height_range=(1.0, 2.5),
    )

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

    # 无人机三维世界系速度指令 [vx, vy, vz]（m/s）
    uav_velocity = mdp.UavVelocityActionCfg(asset_name="robot")



@configclass
class ObservationsCfg:
    """MDP 的观测项配置（导航版，全部转 goal frame）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组。"""

        state = ObsTerm(func=mdp.state_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        lidar = ObsTerm(func=mdp.lidar_obs, params={"asset_cfg": SceneEntityCfg("lidar")})
        direction = ObsTerm(func=mdp.direction_obs)
        dynamic_obstacle: ObsTerm | None = ObsTerm(func=mdp.dynamic_obstacle_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            # 保持 dict 结构（state/lidar/direction/dynamic 分开），
            # 便于后续网络分别编码（lidar 用 CNN，其余用 MLP）。
            self.concatenate_terms = False

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件项配置。"""

    # 重置：从地图边界随机起点；目标由 CommandManager 生成
    reset_robot_state = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
        params={
            "map_range": (20.0, 20.0, 6.0),
            "start_z_range": (0.5, 2.5),
            "boundary_offset": 2.0,
            "yaw_angle": -1.5707963267948966,
        },
    )


@configclass
class CommandsCfg:
    """导航目标命令配置。"""

    nav_target = mdp.NavTargetCommandCfg(
        asset_name="robot",
        target_y=-22.0,
        resampling_time_range=(1.0e9, 1.0e9),
    )


@configclass
class RewardsCfg:
    """MDP 的奖励项配置（uav 权重）。"""

    navigation = RewTerm(func=mdp.NavigationReward, weight=1.0)


@configclass
class TerminationsCfg:
    """MDP 的终止项配置。"""

    static_collision = DoneTerm(func=mdp.static_collision, params={"asset_cfg": SceneEntityCfg("lidar")})
    dynamic_collision: DoneTerm | None = DoneTerm(func=mdp.dynamic_collision)
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds)
    success = DoneTerm(func=mdp.success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """基于 episode 成功率逐步增加动态障碍物数量、速度和运动范围。"""

    dynamic_obstacles = CurrTerm(
        func=mdp.SuccessRateDynamicObstacleCurriculum,
        params={
            "window_size": 1024,
            "success_thresholds": (0.85, 0.65, 0.68, 0.70),
            # 从少量动态障碍物开始，避免从无障碍场景直接切换到动态场景。
            "active_counts": (5, 20, 50, 75, 100),
            "motion_extents": (
                (0.05, 0.05, 0.02),
                (0.15, 0.15, 0.05),
                (0.35, 0.35, 0.15),
                (0.65, 0.65, 0.25),
                (1.0, 1.0, 0.4),
            ),
            "speed_ranges": (
                (0.05, 0.10),
                (0.05, 0.15),
                (0.10, 0.30),
                (0.15, 0.50),
                (0.25, 0.75),
            ),
            "required_passes": 2,
        },
    )


@configclass
class RecorderCfg(RecorderManagerBaseCfg):
    """训练过程与 episode 数据记录项配置。"""

    # 默认关闭，避免普通 PPO 训练无意间创建数据集文件；需要数据集时显式启用。
    reward_components: mdp.NavigationRewardRecorderCfg | None = None


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
    commands: CommandsCfg = CommandsCfg()
    # MDP 配置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()
    recorders: RecorderCfg = RecorderCfg()

    # 后初始化
    def __post_init__(self) -> None:
        """完成环境配置的后初始化。"""
        # 动态障碍物大开关：场景中没有刚体集合时，关闭观测与终止相关项。
        if self.scene.dynamic_obstacles is None:
            self.observations.policy.dynamic_obstacle = None
            self.terminations.dynamic_collision = None
        # 通用配置
        self.decimation = 1
        self.episode_length_s = 60
        # 查看器配置
        self.viewer.eye = (0.0, 0.0, 30.0)   # 相机在原点正上方 30m
        # 仿真配置
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        # 每个物理迭代应用外力，
        # 减小速度反馈噪声，避免悬停抖动。
        self.sim.physx.enable_external_forces_every_iteration = True
