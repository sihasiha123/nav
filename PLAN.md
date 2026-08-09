# 无人机导航场景与强化学习任务计划

## 当前基线

- Isaac Sim 5.1 与 Isaac Lab 2.3.2 已在 `env_isaaclab` 中可运行。
- 无人机 USD 已验证为 7 刚体、4 关节的浮动 Articulation；控制机身为 `body`，飞行器质量为 0.351 kg。
- 世界系速度控制器已验证：悬停稳定，0.8 m/s 速度跟踪稳定，10 m x 5 m 矩形航线可闭合回原点，误差约 0.10 m。
- 当前 Manager 任务仍是 Cartpole 模板。无人机资产和低层控制器尚未接入任务环境。

## 首个可训练任务

构建一个基于 Manager 的三维点到点导航任务：无人机从随机初始状态出发，在有限高度范围内飞到随机目标点；后续逐步加入静态障碍物和距离观测。

固定的接口约定：

- 策略动作：3 维归一化速度指令，映射为世界系 `[vx, vy, vz]`。
- 低层控制：复用 `VelocityController`，由 ActionTerm 在每个 physics step 写入机身推力和力矩。
- 目标：由 CommandManager 为每个环境独立采样三维目标位置。
- 第一个训练版本不使用相机。避障使用接触传感器和轻量级 RayCaster，保证并行训练吞吐量。

## 场景规格

### 开阔空间版本

- 每个环境是 12 m x 12 m x 5 m 的飞行区域，`env_spacing` 至少 16 m。
- 地面位于 `z = 0`，无人机默认初始位置为 `(0, 0, 2)`。
- 目标采样范围为 `x/y in [-4, 4]`、`z in [1.5, 3.0]`；目标与初始位置保持最小水平距离。
- 用可视化 marker 显示目标，仅用于调试，不进入 policy 图像观测。
- 四周设置可碰撞的边界墙，地面、墙体、越高/越低均可触发失败条件。

### 避障版本

- 在飞行区域内生成 2 至 6 个静态 Cuboid 障碍物，障碍物与起点、目标保持安全间隔。
- 在机身上配置水平 RayCaster，初始使用 16 条等角射线和有限最大量程；接触传感器作为碰撞终止的最终依据。
- 障碍物数量、尺寸、目标距离和初始速度均可由 CurriculumManager 渐进增加。

## 目录与职责

保留 `assets/` 和 `controllers/` 的职责，不把任务逻辑放入低层控制器。目标结构如下：

```text
source/nav/nav/tasks/manager_based/nav/
  scene_cfg.py                 # NavSceneCfg、地面、墙、障碍物、传感器、目标 marker
  nav_env_cfg.py               # 汇总所有 Manager 配置和仿真参数
  mdp/
    actions.py                 # UavVelocityAction(ActionTerm)
    actions_cfg.py             # UavVelocityActionCfg(ActionTermCfg)
    commands.py                # 随机目标点命令和目标 marker 更新
    observations.py            # 无人机状态、相对目标、距离射线、上一动作
    events.py                  # 根状态、目标、障碍物重置与随机化
    rewards.py                 # 导航与避障奖励
    terminations.py            # 到达、碰撞、越界、姿态和超时终止
scripts/
  test_drone_dynamics.py       # 已完成：控制器和飞行动力学回归测试
  test_nav_scene.py            # 新增：场景、传感器和多环境 reset 烟雾测试
  test_uav_action.py           # 新增：ActionTerm 与控制器接线测试
```

## 实施顺序

### 阶段 1：场景 MVP

- [ ] 新建 `scene_cfg.py`，将 Cartpole 替换为 `DRONE_CFG`。
- [ ] 创建地面、四面边界墙、灯光和目标 marker；暂不添加内部障碍物。
- [ ] 设置开发配置为 `num_envs = 1`，确认 GUI 中的尺度、起飞高度、墙体碰撞和目标位置正确。
- [ ] 新建 `scripts/test_nav_scene.py`，在无头模式测试 1 和 16 个环境的生成、reset、销毁与状态有限性。

验收条件：无人机不穿透地面/墙体；每个 clone 内的无人机、墙体和目标互不串扰；16 环境 reset 后无 NaN 或 PhysX 报错。

### 阶段 2：传感器与障碍物

- [ ] 为无人机配置 ContactSensor，用于地面、墙体和障碍物碰撞判定。
- [ ] 为机身配置 RayCaster，确认射线相对机身姿态更新且读数有限。
- [ ] 添加静态 Cuboid 障碍物生成器；先使用固定布局，再实现每环境随机布局。
- [ ] 增加场景测试：目标与障碍物、起点的最小间距以及传感器张量形状。

验收条件：碰撞可稳定检测；射线能区分自由空间与障碍物；所有生成的初始状态可飞行且目标可达。

### 阶段 3：接入 Manager Action

- [ ] 实现 `UavVelocityActionCfg(ActionTermCfg)`，动作维度为 3。
- [ ] 实现 `UavVelocityAction(ActionTerm)`：`process_actions()` 完成动作裁剪和速度缩放，`apply_actions()` 调用现有控制器，`reset()` 清空控制器与 wrench 状态。
- [ ] 在 `ActionsCfg` 中用该 ActionTerm 替换 Cartpole `JointEffortActionCfg`。
- [ ] 新建 `test_uav_action.py`，验证单环境与多环境的悬停、X/Y/Z 速度命令和部分环境 reset。

验收条件：Manager 的动作空间为 3；动作不会跨环境泄漏；零动作保持悬停；部分 reset 后无残留推力。

### 阶段 4：自由空间目标导航

- [ ] 实现目标位置 CommandTerm，并将相对目标位置加入 policy 观测。
- [ ] 观测最小集：相对目标位置、世界/机体线速度、角速度、投影重力、上一动作。
- [ ] 事件：随机根位置、姿态、线速度、目标点；保持安全高度和目标最小距离。
- [ ] 奖励：目标距离进展、到达奖励、动作变化惩罚、过大角速度惩罚、失败惩罚。
- [ ] 终止：到达目标、超时、越界、触地、过大倾角。
- [ ] 将 Gym 注册名从模板 `Template-Nav-v0` 改为项目命名，例如 `Nav-Drone-PointNav-v0`。

验收条件：`zero_agent`、随机动作和固定速度动作可以完整运行；策略可在无障碍场景中达到随机目标。

### 阶段 5：障碍物导航与课程学习

- [ ] 将 RayCaster 距离和碰撞状态加入观测/终止逻辑。
- [ ] 增加障碍物接近惩罚、碰撞强惩罚和安全到达奖励。
- [ ] 课程从无障碍、近距离目标开始，逐步增加目标距离、障碍物数量、障碍物尺寸和初始速度。
- [ ] 将开发环境数逐步扩展为 16、128，再根据 GPU 显存和步进吞吐量确定训练环境数。

验收条件：独立评测布局中统计成功率、碰撞率、平均到达时间和最小障碍距离；训练集之外的障碍布局不发生明显性能塌缩。

### 阶段 6：鲁棒性与正式训练

- [ ] 对质量、惯量、推力上限、控制延迟、初始姿态、传感噪声和外力扰动做域随机化。
- [ ] 调整 PPO 配置：实验名、网络规模、并行环境数、训练步数、评测和 checkpoint 周期。
- [ ] 固定种子和独立测试集，保存环境、策略和指标配置到日志目录。

验收条件：多个随机种子下的评测结果稳定，策略在扰动和未见场景中仍满足项目设定的成功率与安全指标。

## 实施纪律

- 在阶段 1 至 3 全部通过前，不启动 PPO 长训练。
- 每一阶段只新增一个可验证能力，并保留对应 headless 测试脚本。
- 不为训练使用视觉输入，除非 RayCaster 方案无法满足避障信息需求。
- 除非动力学回归测试失败，不修改已经验证过的 `VelocityController` 增益和执行流程。
