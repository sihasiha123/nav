# ManagerBasedRLEnv 重构方案

## 目标

按照官方 Manager 的职责重构主任务，删除“大而全”的 `NavTaskBuffer`。跨 step 数据由真正使用它的有状态 Term 保存，避免数据游离和重复拷贝。

## 归属划分

```text
InteractiveScene
    机器人、LiDAR、静态障碍物、动态障碍物及其物理状态

ActionManager
    UavVelocityAction；动作预处理和控制器状态

CommandManager
    NavTargetCommand；target_pos、target_dir、目标重采样

EventManager
    reset/startup/interval 事件；只负责修改场景或初始化状态

ObservationManager
    state、LiDAR、方向、动态障碍物观测；历史观测使用官方 history 配置

RewardManager
    各 RewardTerm 和 episode reward 累计
    ProgressReward/SmoothnessReward 自己保存 previous distance/velocity

TerminationManager
    success、collision、out_of_bounds、timeout
    有历史需求时由对应 Term 保存 episode 标志

CurriculumManager
    障碍物数量、速度、范围等难度参数调整

RecorderManager
    reward 分项、成功率、碰撞率、轨迹等记录
```

## 文件调整

```text
mdp/actions.py         ActionTerm
mdp/commands.py        NavTargetCommand
mdp/events.py          reset/startup/interval 事件
mdp/observations.py    ObservationTerm
mdp/rewards.py         有状态或无状态 RewardTerm
mdp/terminations.py    TerminationTerm
mdp/curriculum.py      CurriculumTerm
mdp/dynamic.py         动态障碍物完整领域模块，保持整体
```

删除 `NavTaskBuffer`。目标由 `CommandTerm` 唯一持有，奖励历史由对应 `RewardTerm` 持有，日志数据由 `RewardManager/RecorderManager` 持有。只有无法归属于任何 Manager 且确实被多个模块共享的数据，才允许作为环境级状态保留。

## 实施顺序

1. 先实现 `NavTargetCommand`，让 observation、reward、termination 统一读取 `env.command_manager` 的目标。
2. 将 `prev_distance`、`prev_drone_vel_w`、`reached_goal_once` 分别迁移到对应的有状态 Term，并实现 `reset(env_ids)`。
3. 将 `reset_nav_task()` 拆为机器人 reset event 和 command reset，不在 event 中保存奖励或任务缓存。
4. 删除 `NavTaskBuffer` 及 `reward_components`，改用官方 Manager 的累计值和 Recorder 记录。
5. 用固定 seed、多个并行环境执行 reset/step smoke test，确认形状、奖励缩放和终止行为不变。
