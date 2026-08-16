# 无人机导航场景与强化学习任务计划

## 当前基线（2026-08-11）

- Isaac Sim 5.1 与 Isaac Lab 2.3.2 已在 `env_isaaclab` 中可运行。
- 无人机 USD 已验证为 7 刚体、4 关节的浮动 Articulation；控制机身为 `body`，飞行器质量 0.351 kg。
- 世界系速度控制器已验证：悬停稳定，0.8 m/s 速度跟踪稳定，10 m x 5 m 矩形航线可闭合回原点，误差约 0.10 m。
- 已通读 direct 参考实现 `/home/lenovo/uav`（`ppo-improvement` 分支）：其观测、奖励、终止、动态障碍物与 PPO 均有可直接移植的实现，本计划后续阶段以它为主要参考。

## 当前代码状态

### 已接入

- **共享地图场景** `NavSceneCfg`：`ground`（300x300 兜底地面）、`terrain`（40x40 + 5m 边框，30 个 6m 静态障碍）、`robot`（1024 架无人机，`env_spacing=0.0`、`replicate_physics=True`、`filter_collisions=True`）、`dynamic_obstacles`（100 个全局动态障碍物）、灯光。
- **全局动态障碍物集合**：`/World/Dynamic/Obstacle_000~099`，Cuboid `(0.5, 0.5, 1.0) m`，`kinematic_enabled=True`、`disable_gravity=True`、`collision_group=-1`；初始位姿即运动锚点（10x10 网格，z=1.5m）。`GlobalRigidObjectCollection` 重写 `reset()` 为无条件忽略，防止框架把 1024 个机器人索引传给单例集合。
- **运动管理器** `GlobalObstacleManager`：锚点 + 有界随机航点 + 直线积分，状态形状 `[1, M, ...]`，每物理步批量写入位姿与线速度（`write_object_link_pose_to_sim` + `write_object_link_velocity_to_sim`）。首次 `step()` 时懒初始化。
- **运动接入**：`GlobalObstacleMotionAction`（0 维 ActionTerm）已注册进 `ActionsCfg`，在每个物理步前调用 `step_global_obstacles()`；不使用 interval EventTerm。
- **无人机动作**：`UavVelocityAction`（3 维世界系速度 `[vx, vy, vz]`）已替换 Cartpole `joint_effort`；`process_actions()` 做缩放/裁剪，`apply_actions()` 调 `VelocityController`，`reset()` 清指令与 wrench。
- **占位 MDP**：观测为 `root_pos_w + root_lin_vel_w`（6 维），事件为空，奖励为 `alive +1`，终止为 `time_out`；环境可实例化，供调试与冒烟。

### 已确定的设计决定

- **动态障碍物不重置**：每次训练都是新进程，场景 spawn 后障碍物位于锚点，第一个物理步懒初始化即“复位”。训练期间只有单无人机部分重置，不碰障碍物。`reset_global()` 及相关代码已删除。
- **碰撞语义**：沿用 uav 路线，无人机使用无碰撞体资产，静态碰撞由 LiDAR 推断，动态碰撞由 GPU 几何距离计算；是否启用真实 PhysX 接触待烟雾测试后再定。

### 已知问题（待处理）

- 场景当前使用 `DRONE_CFG`（带碰撞体、接触传感器），与 uav 的 `DRONE_NO_COLLIDER_CFG` 不一致，阶段 1 切换。
- 动态障碍物当前 `CollisionPropertiesCfg()` 默认开启碰撞，uav 使用 `collision_enabled=False`；若走几何距离路线应关闭，阶段 2 切换。
- 动态障碍物固定 z=1.5m，可能嵌进 6m 静态障碍物；需要采样避让或抬高运动平面。
- 本地 shell 无 GPU（无 `/dev/nvidia*`），`SimulationDocker` 容器不存在；可视化/冒烟需在有 GPU 的 Isaac 环境进行。

## 参考实现（uav direct 版可移植内容）

### 场景与传感器

- 无人机：`DRONE_NO_COLLIDER_CFG`（无碰撞体，`activate_contact_sensors=False`）。
- 动态障碍物：`RigidObjectCollectionCfg`，40 个，`collision_enabled=False`，运动状态由任务维护。
- LiDAR：`RayCasterCfg`，`prim_path="/World/envs/env_.*/Robot/body"`，`ray_alignment="yaw"`，`mesh_prim_paths=["/World/ground"]`；36 水平 x 4 垂直光束，`vertical_fov_range=(-10, 20)`，量程 4m。

### 动态障碍物运动参数（uav 默认）

- 数量 40（nav 现为 100，课程中可渐进）。
- 速度范围 `(0.5, 1.5) m/s`，局部运动范围 `(5.0, 5.0, 4.5) m`。
- 到达阈值 0.5m；每 `dyn_velocity_resample_time=2s` 周期重采样速度；位置 clamp 到地图范围。
- 写入速度使用 `write_object_com_velocity_to_sim`（与 nav 的 link 版本等价）。

### 观测（全部转到 goal frame）

- `state [N, 8]`：`target_dir_goal(3) + distance_2d(1) + distance_z(1) + drone_vel_goal(3)`。
- `lidar [N, 1, 36, 4]`：`range - distance` 归一化，越界填 `lidar_range`。
- `direction [N, 1, 3]`：2D 任务方向（goal frame x 轴）。
- `dynamic_obstacle [N, 1, 5, 10]`：最近 5 个障碍物的 `rel_pos_unit(3) + dist_2d(1) + dist_z(1) + vel_goal(3) + width_category(1) + height_category(1)`，量程外（range_mask）清零。
- 坐标系：reset 时固定 `target_dir = target_pos - start_pos`，episode 内不变；所有相对量转到该坐标系。

### 奖励（uav 起步权重）

```text
reward = 4.0 * progress
       + 0.5 * vel（clamp ±2，沿目标方向速度投影）
       - 6.0 * static_clearance_penalty²
       - 10.0 * dynamic_clearance_penalty²
       - 2.0 * height_penalty²
       - 0.05 * smooth_penalty
       - 0.01（时间惩罚）
首达目标（安全） +50；安全到达持续 +0.5；碰撞 -120
```

### 终止

- 静态碰撞：`lidar.amax > lidar_range - 0.3`。
- 动态碰撞（uav 仅检查最近 5 个，nav 改为检查全部）：`dist_2d <= width*0.5+0.3` 且 `|dist_z| <= height*0.5+0.3`。
- 越界：`z < 0.2` 或 `z > 4.0`。
- 成功：`distance < goal_radius` 且无碰撞（成功即 terminated）。
- 超时：`episode_length_buf >= max_episode_length - 1`。

### Reset（横穿任务）

- 从四条地图边界随机选起点，目标放对侧边界；初始 yaw 朝向目标；根状态/关节状态清零写入。
- 记录 `height_range = [min(start_z, target_z), max(...)]` 供高度奖励/终止使用。
- 清空动作、控制器（`reset_idx`）、`prev_distance`、`reached_goal_once`。

### PPO（可参考）

- 动作分布：Beta 分布（归一化动作）。
- ValueNorm；GAE。
- 特征提取：lidar CNN + dynamic obstacle MLP + state concat 融合。
- Trainer 用 TensorDict 管理 rollout；环境只返回普通 dict。

### 框架分层（Env / Trainer / 算法层）

三个层次各自独立，只通过固定接口接触：

| 层 | 负责 | 知道什么 | 不知道什么 |
|---|---|---|---|
| Env | 仿真：场景、物理、动作、观测、奖励、终止、重置 | 世界如何响应动作 | PPO、rollout、网络 |
| Trainer | 数据与流程：收集 rollout、调度 act/update、日志、checkpoint | env 接口 + 算法接口 | 物理细节、网络内部 |
| Algorithm | 学习：网络结构、GAE、PPO clip、梯度更新 | obs → action/value、如何更新参数 | 仿真、奖励公式 |

三个接口：

```text
env.step(action)      # 世界走一步，返回 obs/reward/done
agent.act(obs)        # 采样 action + log_prob + value
agent.update(rollout) # 用一批数据更新网络
```

完整时间线：

```text
obs_t
  → trainer: agent.act(obs_t)                  # 算法层推理
  → trainer: env.step(action_t)                # 环境层仿真（含自动重置）
  → trainer: 存一条 transition
  → 重复 num_steps_per_env 次，得到 rollout
  → trainer: agent.update(rollout)             # 算法层学习
```

对应本项目：Env = nav（NavEnvCfg + mdp），Trainer = uav 的 scripts/train.py（待移植），Algorithm = uav 的 agents/ppo.py（待移植）。

## 首个可训练任务

构建一个基于 Manager 的三维点到点导航任务：多个无人机在同一张全局地图中并行训练，从地图边界随机起飞，穿过中心障碍区域飞向对侧目标点。场景组织参考 uav direct 版：机器人按 env 并行管理，静态地形、动态障碍物和大平地作为全局共享资源。

固定接口约定：

- 策略动作：3 维归一化速度指令，映射为世界系 `[vx, vy, vz]`。
- 低层控制：复用 `VelocityController`，由 `UavVelocityAction` 在每个物理步写入机身推力和力矩。
- 目标：CommandManager/事件为每个无人机独立采样起点和对侧目标点，目标坐标在全局地图坐标系中。
- 不使用相机；避障使用 RayCaster 与 GPU 几何距离。

## 场景规格

### 共享全局地图版本

- 多个无人机并行训练，共享同一张 `/World/ground` 静态地形，不按 env 克隆地图。
- `scene.env_spacing = 0.0`；同构无人机 `replicate_physics = True`；无人机 prim 使用 `{ENV_REGEX_NS}/Robot`。
- 大平地 `/World/defaultGroundPlane`，尺寸 300x300，作背景和兜底地面。
- 中心障碍地形 `/World/ground`：地图范围 `map_range = [20.0, 20.0, 6.0]`（40x40x6m）。
- 静态障碍迁移 uav 的 `StaticTerrainCfg`：高度场随机矩形障碍，`num_static` 默认 200，尺寸 `(0.4, 1.1)m`，高度 `(1.0, 6.0)m`；先 30 个轻量测试。
- 训练起点从四条边界随机采样，`x/y = +/- map_range`，目标在对侧边界；初始高度 `z in [0.5, 2.5]`。
- LiDAR 安装在 `Robot/body`，扫描 `/World/ground`，观测形状 `[1, 36, 4]`，量程 4m。

### 共享全局动态障碍物

- 一套真正的全局 `RigidObjectCollection`，绝对路径 `/World/Dynamic/Obstacle_*`，数据形状 `[1, M, 13]`。
- 使用 `kinematic_enabled=True`、`disable_gravity=True`、`collision_enabled=False`（几何距离路线）与 `collision_group=-1`。
- 初始布局为规则网格，间距大于“障碍物最大尺寸 + 两倍运动半径”；锚点应避开静态障碍物或查询地形高度。
- 运行时由 `GlobalObstacleManager` 维护 `anchor/position/target/velocity/speed`，`[1, M, ...]` 张量批量推进，一次写入 pose + velocity。
- 运动模型：有界随机航点 + 速度周期重采样 + clamp 到地图；到达后重新采样。
- 全局时钟独立于单架无人机的 episode；单无人机 reset 不修改障碍物；动态障碍物不随场景重置（新进程 + 懒初始化复位）。
- 观测用张量广播计算 `[N, M, 3]` 相对位置，输出最近 `K=5` 个；碰撞终止检查全部障碍物。

### 开发降级版本

- 第一轮场景测试启用大 ground 和空 terrain，或 `num_static=30`，确认 spawn/reset/LiDAR/控制器稳定。
- 动态障碍物先用 2 个验证运动与写入，再扩展到 40、100。
- 目标 marker 仅用于 GUI 调试，不进入策略观测。

## 目录与职责

```text
source/nav/nav/tasks/manager_based/nav/
  terrain.py                   # StaticTerrainCfg（迁移 uav 高度场障碍生成器）
  scene_cfg.py                 # NavSceneCfg、大 ground、共享 terrain、全局动态障碍物、传感器
  nav_env_cfg.py               # 汇总 Manager 配置与仿真参数（当前 MDP 为占位）
  mdp/
    actions.py                 # UavVelocityAction / Cfg（已实现）
    global_obstacle_action.py  # GlobalObstacleMotionAction / Cfg（已实现，0 维动作项）
    dynamic.py                 # GlobalObstacleManager：全局障碍物状态、航点、批量运动（已实现）
    commands.py                # 边界起点、对侧目标、目标 marker 更新
    observations.py            # state、lidar、direction、dynamic_obstacle（goal frame）
    events.py                  # 无人机根状态和目标 reset；不重置全局障碍物
    rewards.py                 # uav 权重起步的导航与避障奖励
    terminations.py            # 到达、碰撞、越界、姿态和超时终止
scripts/
  test_drone_dynamics.py       # 已完成：控制器与飞行动力学回归测试
  test_nav_scene.py            # 共享地图、传感器和多无人机 reset 烟雾测试
  test_uav_action.py           # ActionTerm 与控制器接线测试
```

## 实施顺序

### 阶段 1：场景 MVP

- [x] `NavSceneCfg` 共享地图场景（ground、terrain、robot、动态障碍物集合、灯光）。
- [x] `GlobalRigidObjectCollection` 忽略场景 reset。
- [ ] 无人机切换为 `DRONE_NO_COLLIDER_CFG`。
- [ ] 新建 `terrain.py`，迁移 uav `StaticTerrainCfg`，先用 `num_static=30` 轻量测试。
- [ ] 新建 `scripts/test_nav_scene.py`：1、16、128 个无人机的生成、reset、销毁与状态有限性。

验收条件：多个无人机位于同一张全局地图；`env_origins` 无环境间偏移；reset 后无 NaN/PhysX 报错；地形只生成一份 `/World/ground`。

### 阶段 2：传感器与障碍物

- [ ] 为机身配置 RayCaster（uav 参数）：`mesh_prim_paths=["/World/ground"]`，形状 `[1, 36, 4]`。
- [ ] 动态障碍物 `collision_enabled=False`，与几何距离路线一致。
- [ ] 动态障碍物锚点避让静态障碍（或抬高运动平面）。
- [ ] 运动参数对齐 uav：速度 `(0.5, 1.5)`、局部范围 `(5,5,4.5)`、周期重采样、clamp 到地图。
- [ ] 障碍物数量 30 → 200 静态；动态 2 → 40 → 100。
- [ ] 场景测试：LiDAR 命中共享 terrain；不同无人机读数不同；起点/目标在边界且方向正确。

验收条件：LiDAR 读数有限；静态碰撞可由最近距离稳定推断；初始状态可飞行且目标可达。

### 阶段 3：Manager Action（已完成，待测试）

- [x] `UavVelocityActionCfg`：动作维度 3。
- [x] `UavVelocityAction`：`process_actions()` 缩放/裁剪，`apply_actions()` 调 `VelocityController`，`reset()` 清控制器与 wrench。
- [x] `ActionsCfg` 替换 Cartpole；`global_obstacle_motion` 每物理步推进障碍物。
- [x] 观测/事件/奖励/终止替换为占位，环境可实例化。
- [ ] `test_uav_action.py`：单/多环境悬停、X/Y/Z 速度命令、部分 reset 无残留推力。

验收条件：动作空间 3；零动作悬停；部分 reset 后无残留推力；动态障碍物每步运动。

### 阶段 4：自由空间目标导航（直接移植 uav）

- [ ] 实现边界起点、对侧目标、yaw 朝向目标的 reset 逻辑（uav `_reset_idx` 移植为 Manager 事件）。
- [ ] 观测最小集按 uav：`state=8`、`lidar=[1,36,4]`、`direction=[1,3]`、`dynamic_obstacle=[1,5,10]`，全部转 goal frame。
- [ ] 奖励按 uav 权重起步：progress、vel、static/dynamic 安全距离、height、smooth、时间、到达/碰撞。
- [ ] 终止：成功（即 terminated）、静态/动态碰撞、越界、超时。
- [ ] 将 Gym 注册名改为 `Nav-Drone-PointNav-v0`。

验收条件：zero/random/固定速度动作可完整运行；无障碍版本可完成边界到对侧目标。

### 阶段 5：障碍物导航与课程学习

- [x] 全局动态障碍物创建、运动管理器、动作项接入（未在 Isaac 中验证）。
- [ ] 动态障碍物观测：广播 `[N, M, ...]`，输出最近 5 个；碰撞终止检查全部障碍物。
- [ ] 动态障碍物烟雾测试：2 个验证运动范围与位姿写入；部分无人机 reset 不改变障碍物；扩展到 40/100。
- [ ] 课程从空 terrain、少量静态障碍开始，逐步增加静态/动态数量、速度范围和目标难度。
- [ ] 无人机数量 16 → 128 → 1024，根据吞吐量确定正式规模。

验收条件：场景只有一套 `/World/Dynamic`；状态 `[1, M, 13]`；单无人机 reset 不改变障碍物；独立评测布局统计成功率、碰撞率、平均到达时间、最小障碍距离。

### 阶段 6：鲁棒性与正式训练

- [ ] 域随机化：质量、惯量、推力上限、控制延迟、初始姿态、传感噪声、外力扰动。
- [ ] PPO 配置（可参考 uav：Beta 分布、ValueNorm、lidar CNN + dynamic MLP 特征提取）：实验名、网络规模、并行环境数、训练步数、checkpoint 与评测周期。
- [ ] 固定种子与独立测试集；保存环境、策略、指标配置到日志目录。

验收条件：多随机种子评测稳定；策略在扰动与未见布局中满足成功率与安全指标。

## 实施纪律

- 阶段 1~4 全部通过前，不启动 PPO 长训练。
- 每一阶段只新增一个可验证能力，并保留对应 headless 测试脚本。
- 共享地图是场景假设：不默认改成每个 env 独立克隆地形。
- 全局动态障碍物由独立全局时钟驱动；任何单环境 reset、奖励项、观测项不得修改其状态；动态障碍物不随场景重置。
- 不为训练使用视觉输入，除非 RayCaster 无法满足避障信息需求。
- 除非动力学回归测试失败，不修改已验证的 `VelocityController` 增益和执行流程。
- 观测一律转到 goal frame（episode 内固定的任务坐标系），保持网络输入稳定。
- 碰撞语义以几何距离为主（无碰撞体资产 + LiDAR/距离判断）；启用真实 PhysX 接触前必须单独烟雾测试。
