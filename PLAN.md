# 无人机导航场景与强化学习任务计划

## 当前基线

- Isaac Sim 5.1 与 Isaac Lab 2.3.2 已在 `env_isaaclab` 中可运行。
- 无人机 USD 已验证为 7 刚体、4 关节的浮动 Articulation；控制机身为 `body`，飞行器质量为 0.351 kg。
- 世界系速度控制器已验证：悬停稳定，0.8 m/s 速度跟踪稳定，10 m x 5 m 矩形航线可闭合回原点，误差约 0.10 m。
- 当前 Manager 任务仍是 Cartpole 模板。无人机资产和低层控制器尚未接入任务环境。

## 当前代码改动（2026-08-10）

### 场景中的全局动态障碍物

- 已在 `nav_env_cfg.py` 中实现 `make_global_obstacle_collection_cfg()`，通过 Python 循环生成一个标准 `RigidObjectCollectionCfg`，默认包含 100 个障碍物。
- 障碍物使用绝对 prim 路径 `/World/Dynamic/Obstacle_000` 至 `/World/Dynamic/Obstacle_099`，因此整个场景只有一套全局障碍物，不会跟随 1024 个机器人环境克隆。
- 默认布局为覆盖 `40 m x 40 m` terrain 的近似正方形网格，边缘留出 `2 m`；默认障碍尺寸为 `(0.5, 0.5, 1.0) m`，运动原点高度为 `1.5 m`。
- 障碍物当前采用运动学刚体：`kinematic_enabled=True`、`disable_gravity=True`、`collision_group=-1`。场景中的初始位姿同时作为后续随机运动的运动原点。
- 已将集合以 `dynamic_obstacles` 字段加入 `NavSceneCfg`。预期全局状态形状为 `[1, 100, 13]`，该形状仍需在 Isaac 环境中确认。
- 已实现项目专用的 `GlobalRigidObjectCollection`。它继承官方 `RigidObjectCollection`，但忽略 `scene.reset(env_ids)` 传入的单机器人环境 ID，防止使用 `0..1023` 索引只有一个 instance 的全局集合；只有显式全局重置才调用父类 `reset()`。

### MDP 中的动态障碍物运动逻辑

- 已新增 `mdp/dynamic.py`，将运行时运动逻辑与 `NavSceneCfg` 分离；场景只负责创建资产，MDP 负责状态与逐步运动。
- 已实现 `GlobalObstacleMotionCfg`，当前默认每个障碍物在自身运动原点附近的 `(1.0, 1.0, 0.4) m` 半范围内运动，速度随机范围为 `0.25～0.75 m/s`，到达阈值为 `0.05 m`。
- 已实现 `GlobalObstacleManager`：从 `default_object_state` 读取运动原点，为所有障碍物批量采样局部航点和速度，并使用 Torch 张量一次性推进全部障碍物。
- `step(dt)` 会限制单步位移以避免越过目标，并通过一次 `write_object_link_pose_to_sim()` 写入整个集合。因为障碍物是运动学刚体，脚本速度仅保存在管理器中供后续观测使用，不调用面向非运动学刚体的 PhysX 速度写入接口。
- 已实现 `reset_global()`，用于显式恢复全部障碍物的初始位姿并重新采样航点；单架无人机 reset 不会调用它。
- 已提供 `initialize_global_obstacles()`、`step_global_obstacles()` 和 `reset_global_obstacles()` 辅助函数，并通过 `mdp/__init__.py` 导出。
- 运动管理器目前尚未接入环境逐步调用，因此障碍物生成后仍保持静止。后续应在无人机 `ActionTerm.apply_actions()` 或等价的物理步前置钩子中，每个 physics step 调用一次 `step_global_obstacles()`；不使用普通 `interval` EventTerm 驱动位姿积分。

### 当前验证状态

- 新增代码的注释和文档字符串已统一为中文，许可证、API 标识符和异常文本保留英文。
- 已通过 `git diff --check` 静态格式检查。
- 当前本地机器没有 Isaac Lab/Isaac Sim 运行环境，因此尚未验证资产生成、全局集合状态形状、运动学位姿写入、碰撞过滤和部分环境 reset；这些项目必须在远程 Isaac 环境中进行烟雾测试。
- Cartpole 的动作、观测、奖励、事件和终止项仍保留在 `nav_env_cfg.py` 中，本轮没有替换，以便先完成场景搭建和动态障碍物机制学习。

## 首个可训练任务

构建一个基于 Manager 的三维点到点导航任务：多个无人机在同一张全局地图中并行训练，从地图边界随机起飞，穿过中心障碍区域飞向对侧目标点。场景组织方式参考 `/home/robot/RL/uav` 当前 `ppo-improvement` 分支：机器人按 env 并行管理，但静态地形、动态障碍物和大平地作为全局共享资源存在。

固定的接口约定：

- 策略动作：3 维归一化速度指令，映射为世界系 `[vx, vy, vz]`。
- 低层控制：复用 `VelocityController`，由 ActionTerm 在每个 physics step 写入机身推力和力矩。
- 目标：由 CommandManager 为每个无人机独立采样起点和对侧目标点，但目标坐标位于同一个全局地图坐标系。
- 第一个训练版本不使用相机。避障使用 RayCaster 和必要的碰撞/距离逻辑，保证并行训练吞吐量。

## 场景规格

### 共享全局地图版本

- 多个无人机并行训练，但共享同一张 `/World/ground` 静态地形，不为每个 env 克隆一份障碍地图。
- `scene.env_spacing = 0.0`，同构无人机使用 `replicate_physics = True` 加速克隆；无人机 prim 仍使用 `/World/envs/env_.*/Robot` 或 `{ENV_REGEX_NS}/Robot`，让 Isaac Lab 管理每个无人机的张量状态。
- 大平地使用 `/World/defaultGroundPlane`，尺寸约 `100 m x 100 m` 或更大，用作背景和兜底地面。
- 中心障碍地形使用 `/World/ground`，地图范围参考旧项目：`map_range = [20.0, 20.0, 6.0]`，即主要障碍区域约 `40 m x 40 m x 6 m`。
- 静态障碍物优先迁移当前 `uav` 分支的 `StaticTerrainCfg`：高度场随机矩形障碍，默认 `num_static = 200`，尺寸范围 `(0.4, 1.1)`，高度范围 `(1.0, 6.0)`。
- 训练起点从四条地图边界随机采样，`x/y = +/- map_range`，目标点放在对侧边界；初始高度采样 `z in [0.5, 2.5]`。
- LiDAR 安装在 `/World/envs/env_.*/Robot/body`，只扫描 `mesh_prim_paths=["/World/ground"]`，观测 shape 保持 `[1, 36, 4]`，量程 `4 m`。

### 共享全局动态障碍物方案

- 动态障碍物采用一套真正的全局集合，prim 使用绝对路径 `/World/Dynamic/Obstacle_000` 至 `/World/Dynamic/Obstacle_{M-1}`，不使用 `{ENV_REGEX_NS}`，因此不会为 1024 个无人机分别复制。
- 使用一个 `RigidObjectCollectionCfg` 管理全部障碍物。`rigid_objects` 字典由 Python 循环生成，不手写重复配置；全局集合的数据形状为 `[1, M, 13]`，而不是 `[1024, M, 13]`。
- 首个版本使用简单的 `CuboidCfg` 或 `SphereCfg`，配置 `kinematic_enabled = True`、`disable_gravity = True` 和 `collision_group = -1`。场景配置只声明数量、形状、材质、路径和初始位姿，不包含逐步运动逻辑。
- 初始布局采用规则网格。`M = 100` 时可使用 `10 x 10` 网格均匀覆盖中心 terrain，并保留边界 margin；每个网格点保存为对应障碍物的运动原点 `anchor_pos_w`。障碍物间距应大于“障碍物最大尺寸 + 两倍运动半径”。
- 非平坦 terrain 上不能只写固定地面高度：贴地障碍物需要查询地形高度或使用 terrain 的 flat-patch 候选点；飞行障碍物可以直接为运动原点采样安全高度。
- 运行时由 `GlobalObstacleManager` 维护 `anchor_pos_w`、`position_w`、`target_pos_w`、`velocity_w`、`motion_range` 和 `active_mask`，张量形状以 `[1, M, ...]` 为主。
- 运动模型采用局部随机目标点：每个障碍物只在自身原点附近的 AABB 或球形范围内采样目标，以固定/随机速度平滑移动；到达目标后重新采样。所有障碍物通过 Torch 张量并行更新，禁止在每个 physics step 中逐个执行 Python 写入。
- 每个 physics step 在仿真前调用一次 `GlobalObstacleManager.step(physics_dt)`，随后通过 `write_object_link_pose_to_sim()` 和必要的 velocity 写入接口一次性更新集合。普通 `interval` EventTerm 位于物理步和奖励计算之后，不作为逐步位姿更新入口。
- `dynamic_obstacle` 观测直接用张量广播计算：无人机位置 `[N, 1, 3]` 与全局障碍位置 `[1, M, 3]` 得到 `[N, M, 3]`，再为每架无人机选择最近 `K = 5` 个障碍物。动态障碍物不依赖只适用于静态 mesh 的普通 RayCaster。
- 第一版沿用 `uav` 项目的训练语义：无人机可使用无碰撞体资产，碰撞终止由无人机与全部动态障碍物的 GPU 几何距离计算；是否启用真实 PhysX 接触在独立烟雾测试通过后再决定。
- 全局障碍物使用独立全局时钟，不随某一架无人机的 episode 重新计时。单个无人机 reset 时只重新采样该无人机及目标，并避开障碍物当前所在位置。
- 普通 `ManagerBasedRLEnv` 会把 `env_ids` 传给 `scene.reset(env_ids)`；全局集合只有一个 instance，不能直接接收任意无人机 ID。实现时必须使用忽略局部 reset 的 `GlobalRigidObjectCollection` 包装，或把集合放入不参与逐 env scene reset 的独立全局子系统。

### 开发降级版本

- 第一轮场景测试可以先启用大 ground 和空 terrain，或将 `num_static` 降到 `30`，确认 spawn、reset、LiDAR 和控制器稳定。
- 动态障碍物第一阶段可以关闭；静态地图和多无人机共享逻辑稳定后，先用 2 个全局障碍物验证运动和局部 reset，再扩展到 40 和 100 个。
- 目标 marker 仅用于 GUI 调试，不进入 policy 图像观测。

## 目录与职责

保留 `assets/` 和 `controllers/` 的职责，不把任务逻辑放入低层控制器。目标结构如下：

```text
source/nav/nav/tasks/manager_based/nav/
  terrain.py                   # StaticTerrainCfg，迁移/整理当前 uav 分支的高度场障碍生成器
  scene_cfg.py                 # NavSceneCfg、大 ground、共享 terrain、全局动态障碍物、传感器、目标 marker
  nav_env_cfg.py               # 汇总所有 Manager 配置和仿真参数
  mdp/
    actions.py                 # UavVelocityAction(ActionTerm)
    actions_cfg.py             # UavVelocityActionCfg(ActionTermCfg)
    commands.py                # 边界起点、对侧目标、目标 marker 更新
    observations.py            # state、lidar、direction、dynamic_obstacle
    events.py                  # 无人机根状态和目标 reset；不得按 env_ids 重置全局障碍物
    rewards.py                 # 导航与避障奖励
    terminations.py            # 到达、碰撞、越界、姿态和超时终止
    dynamic.py                 # GlobalObstacleManager：全局障碍物状态、随机目标和批量运动
scripts/
  test_drone_dynamics.py       # 已完成：控制器和飞行动力学回归测试
  test_nav_scene.py            # 新增：共享地图、传感器和多无人机 reset 烟雾测试
  test_uav_action.py           # 新增：ActionTerm 与控制器接线测试
```

## 实施顺序

### 阶段 1：场景 MVP

- [ ] 新建 `scene_cfg.py`，将 Cartpole 替换为 `DRONE_CFG`。
- [ ] 创建大 ground、灯光和空 `/World/ground` terrain；暂不添加动态障碍物。
- [ ] 设置共享地图参数：`env_spacing = 0.0`、`replicate_physics = True`；只克隆同构无人机，全局 terrain 保持绝对路径单例。
- [ ] 新建 `terrain.py`，迁移当前 `uav` 分支的 `StaticTerrainCfg`，先用 `num_static = 30` 做轻量测试。
- [ ] 新建 `scripts/test_nav_scene.py`，在无头模式测试 1、16、128 个无人机的生成、reset、销毁与状态有限性。

验收条件：多个无人机都位于同一张全局地图坐标系；`scene.env_origins` 不引入环境间地图偏移；reset 后无人机状态无 NaN 或 PhysX 报错；地形只生成一份 `/World/ground`。

### 阶段 2：传感器与障碍物

- [ ] 为机身配置 RayCaster，路径为 `/World/envs/env_.*/Robot/body`，扫描 `mesh_prim_paths=["/World/ground"]`。
- [ ] 保持旧任务观测形状：`state=8`、`lidar=[1, 36, 4]`、`direction=[1, 3]`、`dynamic_obstacle=[1, 5, 10]`。
- [ ] 将静态障碍数量从 `30` 提升到当前分支默认 `200`，再根据吞吐量评估是否恢复更密集配置。
- [ ] 增加场景测试：LiDAR 能打到共享 terrain；不同无人机在同一障碍附近获得不同距离读数；起点/目标位于地图边界且方向正确。

验收条件：RayCaster 读数有限；静态碰撞可由 LiDAR 最近距离稳定推断；所有生成的初始状态可飞行且目标可达。

### 阶段 3：接入 Manager Action

- [ ] 实现 `UavVelocityActionCfg(ActionTermCfg)`，动作维度为 3。
- [ ] 实现 `UavVelocityAction(ActionTerm)`：`process_actions()` 完成动作裁剪和速度缩放，`apply_actions()` 调用现有控制器，`reset()` 清空控制器与 wrench 状态。
- [ ] 在 `ActionsCfg` 中用该 ActionTerm 替换 Cartpole `JointEffortActionCfg`。
- [ ] 新建 `test_uav_action.py`，验证单环境与多环境的悬停、X/Y/Z 速度命令和部分环境 reset。

验收条件：Manager 的动作空间为 3；动作不会跨环境泄漏；零动作保持悬停；部分 reset 后无残留推力。

### 阶段 4：自由空间目标导航

- [ ] 实现边界起点和对侧目标 CommandTerm，复刻当前 `uav` 分支 reset 逻辑。
- [ ] 观测最小集按旧任务保持：目标单位方向、xy 距离、z 距离、goal frame 速度、direction。
- [ ] 事件：随机根位置、朝向目标的 yaw、零速度、目标点；保持安全高度和目标最小距离。
- [ ] 奖励：目标距离进展、朝目标速度、静态障碍安全距离、动态障碍安全距离、高度、平滑、碰撞惩罚、到达奖励。
- [ ] 终止策略需要二选一并固定：当前 `ppo-improvement` 分支是成功即 terminated；旧版文档曾采用到达目标后保持到 timeout。Manager 版先以当前分支为准，成功、碰撞、越界、超时均结束回合。
- [ ] 将 Gym 注册名从模板 `Template-Nav-v0` 改为项目命名，例如 `Nav-Drone-PointNav-v0`。

验收条件：`zero_agent`、随机动作和固定速度动作可以完整运行；策略可在共享地图中完成边界到对侧目标的无障碍版本。

### 阶段 5：障碍物导航与课程学习

- [ ] 将 RayCaster 距离和碰撞状态加入观测/终止逻辑。
- [ ] 增加障碍物接近惩罚、碰撞强惩罚和安全到达奖励。
- [x] 编写全局动态障碍物配置生成函数，通过循环构造 `/World/Dynamic/Obstacle_*`，并注册为单个 `RigidObjectCollectionCfg`（代码已完成，待 Isaac 运行验证）。
- [x] 实现全局集合的独立 reset 保护，确保 `scene.reset(env_ids)` 不会使用无人机 ID 索引或重置全局障碍物（代码已完成，待部分环境 reset 测试）。
- [ ] 实现并接入 `GlobalObstacleManager`：运动原点、当前位置、局部目标、速度、运动范围和批量推进逻辑已经完成；尚需接入每个 physics step 的前置调用并运行验证。
- [ ] 实现动态障碍物观测：广播计算 `[num_envs, dyn_num_obstacles, ...]` 相对状态，只向策略输出最近 5 个障碍物；碰撞终止仍检查全部障碍物。
- [ ] 新增动态障碍物烟雾测试：先用 2 个障碍物检查运动范围、边界和位姿写入；再验证部分无人机 reset 不改变障碍物；最后扩展到 40 和 100 个。
- [ ] 课程从空 terrain、少量静态障碍开始，逐步增加静态障碍数量、动态障碍数量、速度范围和目标难度。
- [ ] 将开发无人机数逐步扩展为 16、128、1024，再根据 GPU 显存和步进吞吐量确定正式训练规模。

验收条件：场景中始终只有一套 `/World/Dynamic`；100 个障碍物状态为 `[1, 100, 13]`；单架无人机 reset 不改变障碍物位置和运动相位；独立评测布局中统计成功率、碰撞率、平均到达时间和最小障碍距离；训练集之外的障碍布局不发生明显性能塌缩。

### 阶段 6：鲁棒性与正式训练

- [ ] 对质量、惯量、推力上限、控制延迟、初始姿态、传感噪声和外力扰动做域随机化。
- [ ] 调整 PPO 配置：实验名、网络规模、并行环境数、训练步数、评测和 checkpoint 周期。
- [ ] 固定种子和独立测试集，保存环境、策略和指标配置到日志目录。

验收条件：多个随机种子下的评测结果稳定，策略在扰动和未见场景中仍满足项目设定的成功率与安全指标。

## 实施纪律

- 在阶段 1 至 3 全部通过前，不启动 PPO 长训练。
- 每一阶段只新增一个可验证能力，并保留对应 headless 测试脚本。
- 共享地图是本项目的场景假设：不要默认改成每个 env 独立克隆地形，除非后续实验明确需要。
- 全局动态障碍物由独立全局时钟驱动，任何单环境 reset、奖励项或观测项都不得修改其状态。
- 不为训练使用视觉输入，除非 RayCaster 方案无法满足避障信息需求。
- 除非动力学回归测试失败，不修改已经验证过的 `VelocityController` 增益和执行流程。
