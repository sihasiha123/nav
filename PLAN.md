# 无人机导航场景与强化学习任务计划

## 当前基线

- Isaac Sim 5.1 与 Isaac Lab 2.3.2 已在 `env_isaaclab` 中可运行。
- 无人机 USD 已验证为 7 刚体、4 关节的浮动 Articulation；控制机身为 `body`，飞行器质量为 0.351 kg。
- 世界系速度控制器已验证：悬停稳定，0.8 m/s 速度跟踪稳定，10 m x 5 m 矩形航线可闭合回原点，误差约 0.10 m。
- 当前 Manager 任务仍是 Cartpole 模板。无人机资产和低层控制器尚未接入任务环境。

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
- `scene.env_spacing = 0.0`，`replicate_physics = False`，无人机 prim 仍使用 `/World/envs/env_.*/Robot` 或 `{ENV_REGEX_NS}/Robot`，让 Isaac Lab 管理每个无人机的张量状态。
- 大平地使用 `/World/defaultGroundPlane`，尺寸约 `100 m x 100 m` 或更大，用作背景和兜底地面。
- 中心障碍地形使用 `/World/ground`，地图范围参考旧项目：`map_range = [20.0, 20.0, 6.0]`，即主要障碍区域约 `40 m x 40 m x 6 m`。
- 静态障碍物优先迁移当前 `uav` 分支的 `StaticTerrainCfg`：高度场随机矩形障碍，默认 `num_static = 200`，尺寸范围 `(0.4, 1.1)`，高度范围 `(1.0, 6.0)`。
- 训练起点从四条地图边界随机采样，`x/y = +/- map_range`，目标点放在对侧边界；初始高度采样 `z in [0.5, 2.5]`。
- LiDAR 安装在 `/World/envs/env_.*/Robot/body`，只扫描 `mesh_prim_paths=["/World/ground"]`，观测 shape 保持 `[1, 36, 4]`，量程 `4 m`。

### 开发降级版本

- 第一轮场景测试可以先启用大 ground 和空 terrain，或将 `num_static` 降到 `30`，确认 spawn、reset、LiDAR 和控制器稳定。
- 动态障碍物第一阶段可以关闭；静态地图和多无人机共享逻辑稳定后，再接入动态障碍物集合。
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
    events.py                  # 根状态 reset、目标 reset、共享动态障碍物 reset
    rewards.py                 # 导航与避障奖励
    terminations.py            # 到达、碰撞、越界、姿态和超时终止
    dynamic.py                 # 共享动态障碍物运动逻辑，迁移当前 uav 分支 DynamicObstacles
scripts/
  test_drone_dynamics.py       # 已完成：控制器和飞行动力学回归测试
  test_nav_scene.py            # 新增：共享地图、传感器和多无人机 reset 烟雾测试
  test_uav_action.py           # 新增：ActionTerm 与控制器接线测试
```

## 实施顺序

### 阶段 1：场景 MVP

- [ ] 新建 `scene_cfg.py`，将 Cartpole 替换为 `DRONE_CFG`。
- [ ] 创建大 ground、灯光和空 `/World/ground` terrain；暂不添加动态障碍物。
- [ ] 设置共享地图参数：`env_spacing = 0.0`、`replicate_physics = False`，确认多个无人机共享同一张地图而不是克隆地图。
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
- [ ] 接入共享动态障碍物集合：参考当前 `uav` 分支 `RigidObjectCollectionCfg` 和 `DynamicObstacles`，默认 `dyn_num_obstacles = 40`。
- [ ] 课程从空 terrain、少量静态障碍开始，逐步增加静态障碍数量、动态障碍数量、速度范围和目标难度。
- [ ] 将开发无人机数逐步扩展为 16、128、1024，再根据 GPU 显存和步进吞吐量确定正式训练规模。

验收条件：独立评测布局中统计成功率、碰撞率、平均到达时间和最小障碍距离；训练集之外的障碍布局不发生明显性能塌缩。

### 阶段 6：鲁棒性与正式训练

- [ ] 对质量、惯量、推力上限、控制延迟、初始姿态、传感噪声和外力扰动做域随机化。
- [ ] 调整 PPO 配置：实验名、网络规模、并行环境数、训练步数、评测和 checkpoint 周期。
- [ ] 固定种子和独立测试集，保存环境、策略和指标配置到日志目录。

验收条件：多个随机种子下的评测结果稳定，策略在扰动和未见场景中仍满足项目设定的成功率与安全指标。

## 实施纪律

- 在阶段 1 至 3 全部通过前，不启动 PPO 长训练。
- 每一阶段只新增一个可验证能力，并保留对应 headless 测试脚本。
- 共享地图是本项目的场景假设：不要默认改成每个 env 独立克隆地形，除非后续实验明确需要。
- 不为训练使用视觉输入，除非 RayCaster 方案无法满足避障信息需求。
- 除非动力学回归测试失败，不修改已经验证过的 `VelocityController` 增益和执行流程。
