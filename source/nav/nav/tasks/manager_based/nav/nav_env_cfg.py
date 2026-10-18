import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, patterns

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg

from isaaclab.utils import configclass

from . import mdp

##
# 预定义配置
##

from nav.assets.quadcopter import DRONE_NO_COLLIDER_CFG
from nav.assets.dynamic import make_global_obstacle_collection_cfg
from nav.assets.terrain import CollisionTerrainImporter


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
        class_type=CollisionTerrainImporter,
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

    # 所有机器人环境共享同一套全局障碍物；中心高度在无人机飞行范围内随机；
    # count=0 时返回 None（禁用）
    dynamic_obstacles: RigidObjectCollectionCfg | None = make_global_obstacle_collection_cfg(
        count=100,
        obstacle_height_range=(1.0, 2.5),
    )

    # 前视深度相机：96 x 54，水平视场 90°，深度上限 4m。
    # 固定在 body 上并跟随完整姿态，查询共享地形与每个独立运动的障碍物。
    depth_camera: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            # 朝向约定为 +X 向前、+Z 向上；相机仍固定于 body。
            convention="world",
        ),
        ray_alignment="base",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=96,
            height=54,
            focal_length=1.0,
            horizontal_aperture=2.0,
        ),
        data_types=["distance_to_image_plane"],
        max_distance=4.0,
        depth_clipping_behavior="max",
        debug_vis=False,
        mesh_prim_paths=[
            # 静态目标使用字符串路径，默认不跟踪位姿。
            "/World/ground",
            *[
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr=obstacle_cfg.prim_path,
                    track_mesh_transforms=True,
                )
                for obstacle_cfg in (
                    dynamic_obstacles.rigid_objects.values() if dynamic_obstacles is not None else ()
                )
            ],
        ],
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

    # 输入机体系实际速度 [前、左、上]（m/s），动作项内部转为世界系。
    uav_velocity = mdp.UavVelocityActionCfg(
        asset_name="robot",
        body_name="body",
        scale=1.0,
        max_velocity=None,
    )



@configclass
class ObservationsCfg:
    """导航观测：机体系自身状态、相对目标与单帧深度图。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """三个独立 float32 Tensor，不包含历史帧或障碍物真值。"""

        # (N, 9)：线速度 B(3)、重力单位方向 B(3)、角速度 B(3)。
        state = ObsTerm(
            func=mdp.state_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["body"])},
        )
        # (N, 4)：目标单位方向 B(3)、目标距离(1)。
        goal = ObsTerm(
            func=mdp.goal_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["body"])},
        )
        # (N, 1, 54, 96)：深度除以相机最大距离，范围 [0, 1]。
        depth = ObsTerm(func=mdp.depth_obs, params={"asset_cfg": SceneEntityCfg("depth_camera")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            # 环境只提供 state/goal/depth 字典；特征编码由算法负责。
            self.concatenate_terms = False

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件项配置。"""

    # 只在启动时初始化共享运动管理器，不随单架无人机 reset。
    initialize_dynamic_obstacles = EventTerm(func=mdp.initialize_dynamic_obstacles, mode="startup")

    # 重置：+Y 边起飞，X 均匀分布、高度随机；只设置无人机初始状态。
    reset_robot_state = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "start_x_range": (-22.0, 22.0),
            "start_y": 22.0,
            "start_z_range": (0.5, 2.5),
            "yaw_angle": -1.5707963267948966,
        },
    )


@configclass
class CommandsCfg:
    """导航终点配置：保留起点 X/Z，将 Y 设置为地图另一侧。"""

    nav_target = mdp.NavTargetCommandCfg(
        asset_name="robot",
        body_name="body",
        target_y=-22.0,
        resampling_time_range=(1.0e9, 1.0e9),
    )


@configclass
class RewardsCfg:
    """MDP 的奖励项配置（uav 权重）。"""

    navigation = RewTerm(
        func=mdp.NavigationReward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "camera_cfg": SceneEntityCfg("depth_camera"),
        },
    )


@configclass
class TerminationsCfg:
    """每步检查，任一条件成立即结束；保留各原因供日志统计。"""

    # 0.3m 暂沿用旧碰撞半径，需在服务器结合无人机实际尺寸核对。
    static_collision = DoneTerm(
        func=mdp.StaticCollision,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "robot_radius": 0.3,
            "terrain_name": "terrain",
            "ground_height": 0.0,
        },
    )
    dynamic_collision: DoneTerm | None = DoneTerm(
        func=mdp.DynamicCollision,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "robot_radius": 0.3,
            "obstacle_name": "dynamic_obstacles",
        },
    )
    out_of_bounds = DoneTerm(
        func=mdp.out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "x_range": (-24.0, 24.0),
            "y_range": (-24.0, 24.0),
            "z_range": (0.2, 4.0),
        },
    )
    # 必须放在失败项之后，复用它们本步的判断；碰撞/越界时不能同时成功。
    success = DoneTerm(
        func=mdp.success,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["body"]),
            "goal_radius": 0.5,
            "failure_terms": ("static_collision", "dynamic_collision", "out_of_bounds"),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """课程训练项配置。

    当前任务暂不启用主动课程项；保留该配置类是为了让环境结构与
    IsaacLab 的 ManagerBasedRLEnv 完整配置保持一致。后续可在此添加
    障碍物数量、运动速度或采样范围等难度调节项。
    """

    pass


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
        # 动态障碍物大开关：场景中没有刚体集合时，关闭动态碰撞终止项。
        if self.scene.dynamic_obstacles is None:
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
