# 强化学习项目重建计划

## 设计原则

从项目顶层按“仿真领域、任务 MDP、训练、算法”划分。环境使用官方 `ManagerBasedRLEnv`，只负责装配 Scene 和 Manager，不重复实现官方 `reset/step`。每类数据只有一个所有者；跨 step 状态由产生它的对象或有状态 Term 保存，不建立全局 `NavTaskBuffer`。

## 目录结构

```text
nav/
├── assets/                         # 仿真领域对象
│   ├── quadcopter.py               # 无人机资产、配置和相关状态
│   └── dynamic.py                  # 动态障碍物完整领域模块
├── controllers/                    # 底层控制器
├── tasks/manager_based/nav/
│   ├── nav_env_cfg.py              # Scene 和所有 Manager 配置
│   └── mdp/
│       ├── actions.py              # 策略动作和资产驱动适配器
│       ├── commands.py             # 目标位置/速度 CommandTerm
│       ├── observations.py         # ObservationTerm
│       ├── rewards.py              # RewardTerm
│       ├── terminations.py         # TerminationTerm
│       ├── events.py               # reset/startup/interval EventTerm
│       └── curriculum.py           # CurriculumTerm
├── trainer/                        # rollout、批处理、训练日志
├── algo/                           # PPO 等算法和网络更新
└── scripts/                        # train、eval、测试入口
```

## 职责和数据归属

```text
assets/dynamic.py
    障碍物生成、运动模型、航点、运行时状态、reset、step(dt)

mdp/actions.py
    UavVelocityAction；GlobalObstacleMotionAction 只调用 dynamic.step(dt)

CommandTerm
    当前目标命令和重采样计时器

RewardTerm
    即时奖励；需要历史值时由该 Term 自己保存并在 reset(env_ids) 清理

TerminationTerm
    成功、碰撞、越界、超时；需要历史标志时由该 Term 自己保存

Recorder/RewardManager
    episode 累计、分项奖励和诊断记录
```

动态障碍物的运动算法只存在于 `assets/dynamic.py`。由于官方 Manager 没有“每物理步脚本资产更新”这一独立 Manager，使用无策略维度的 `GlobalObstacleMotionAction` 作为薄适配器，借用 `ActionManager.apply_actions()` 的调用时机；它不重复实现运动逻辑。

## 标准数据流

```text
reset:
    Scene.reset
    EventManager.apply(reset)
    CommandManager.reset
    各 Term.reset(env_ids)
    ObservationManager.compute

step:
    ActionManager.process_actions
    每个物理步：ActionManager.apply_actions -> Scene/Simulation.step -> Scene.update
    TerminationManager.compute
    RewardManager.compute
    RecorderManager.record
    reset 已结束环境
    CommandManager.compute
    ObservationManager.compute
```

## 实施阶段

1. 先固定环境配置、观测字典、动作维度、奖励缩放和终止名称。
2. 建立 `assets` 模块，使无人机和动态障碍物都能独立生成、reset、更新和测试。
3. 实现 `CommandTerm`，让 observation、reward、termination 统一读取同一目标。
4. 保留一个整体 `NavigationReward`，把其历史缓存放进该有状态 Term，不急于拆成多个奖励类。
5. 完成终止、事件、课程和记录器配置，再接入 Trainer 和 Algo。
6. 使用固定 seed、多个并行环境和不同 decimation 做 reset/step smoke test，确认每个物理步只更新一次动态障碍物，且重建前后训练接口一致。
