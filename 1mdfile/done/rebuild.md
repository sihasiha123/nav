# ManagerBasedRLEnv 主任务重构方案

## 核心原则

按照 IsaacLab 官方 ManagerBasedRLEnv 方式，从一个类一个类重新设计：

```text
参数归使用者
状态归所有者
结果通过官方接口共享
NavEnvCfg 只负责装配
```

不创建 `NavTaskBuffer`，不把所有任务参数集中到一个全局配置类，也不通过 `env._nav_*` 字段在模块之间传递数据。

## 顶层配置骨架

`nav_env_cfg.py` 只声明场景和各 Manager 的配置类：

```text
NavSceneCfg
ActionsCfg
ObservationsCfg
EventCfg
CommandsCfg
RewardsCfg
TerminationsCfg
CurriculumCfg
RecorderCfg
NavEnvCfg
```

```python
@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    scene = NavSceneCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    recorders = RecorderCfg()
```

`NavEnvCfg` 只保存通用环境参数，例如 `decimation`、`episode_length_s` 和仿真设置，不实现具体任务逻辑。

## 各配置类和 Term 的职责

### `NavSceneCfg`

描述仿真中有哪些资产和传感器：

```text
robot、terrain、lidar、dynamic_obstacles、light
```

资产参数放在资产或场景配置中，例如 LiDAR 量程、更新周期和障碍物数量。不在 Event 或 Reward 中重复定义。

### `ActionsCfg` / `ActionTerm`

动作配置只包含动作处理所需参数：

```text
动作缩放、裁剪范围、asset_name、控制器配置
```

`ActionTerm` 负责接收策略动作、预处理动作并调用控制器，不生成目标、不计算奖励。

### `ObservationsCfg` / `ObservationTerm`

观测配置只包含观测项及其参数：

```text
state、lidar、command、dynamic_obstacle
```

观测项只读取：

```python
env.scene["robot"].data
env.scene["lidar"].data
env.command_manager.get_command("nav_target")
```

不修改场景，不保存任务状态。策略观测使用官方的 `PolicyCfg(ObsGroup)` 观测组。

### `EventCfg` / `EventTerm`

Event 参数只放在对应的 EventTerm 中，例如：

```text
起点位置范围
起点高度范围
初始速度范围
初始姿态范围
```

`reset_robot_state` 只负责向机器人资产写入 reset 状态，不生成目标，不初始化奖励历史，不记录日志。

### `CommandsCfg` / `CommandTerm`

Command 参数只放在对应的 CommandTerm 中，例如：

```text
目标采样范围
目标边界
目标速度范围
重采样时间
```

`NavTargetCommand` 只保存并更新目标命令。观测、奖励和终止项通过：

```python
env.command_manager.get_command("nav_target")
```

读取目标，不复制目标到其他 buffer。

### `RewardsCfg` / `RewardTerm`

奖励参数放在各自的奖励项中：

```text
goal_radius
safe_distance
progress weight
collision penalty
```

无状态奖励使用函数；需要历史的奖励继承 `ManagerTermBase`，只保存自己的最小历史状态，并实现：

```python
reset(env_ids)
__call__(env, ...)
```

RewardTerm 不保存奖励日志分项，不写入环境私有字段。

### `TerminationsCfg` / `TerminationTerm`

终止参数只放在对应的终止项中：

```text
碰撞阈值
目标半径
高度上下限
timeout 标记
```

终止项只读取 Scene 和 Command，并返回形状为 `(num_envs,)` 的布尔张量，不执行 reset。

### `CurriculumCfg` / `CurriculumTerm`

课程参数只放在课程项中：

```text
难度变化步数
障碍物速度上限
障碍物数量阶段
采样范围阶段
```

课程项只在训练进度满足条件时修改难度配置。没有课程时保留空配置，不在 Event 或 Reward 中实现课程逻辑。

### `RecorderCfg` / `RecorderTerm`

记录参数只放在 Recorder 配置中：

```text
记录字段
记录时机
导出路径
导出格式
```

Recorder 记录 observation、action、reward、command、终止原因和轨迹快照，不参与任务计算。奖励分项和 episode 统计由 RecorderManager 管理，Trainer 不访问 RewardTerm 的私有字段。

## 动态障碍物模块

`assets/dynamic.py` 保持为完整领域模块，负责：

```text
动态障碍物配置
障碍物集合
运动状态
航点采样
运动更新
资产 reset/初始化
```

障碍物运动逻辑不放入 Reward、Observation 或 Event。若需要每个 physics step 更新，应提供明确的环境更新入口，不使用零维 ActionTerm 隐式驱动。

## 官方 reset 流程

环境统一触发 reset，但每个 Manager 只重置自己的 Term：

```text
scene.reset()
-> EventManager 执行 reset_robot_state
-> ObservationManager.reset()
-> ActionManager.reset()
-> RewardManager.reset()
-> CurriculumManager.reset()
-> CommandManager.reset()
-> EventManager.reset()
-> TerminationManager.reset()
-> RecorderManager.reset()
```

因此一次 episode 的状态确实全部被重置，但不是由一个 Event 函数跨模块完成。

## 官方 step 流程

```text
ActionManager.process_action()
-> ActionManager.apply_action()
-> physics stepping
-> scene.update()
-> TerminationManager.compute()
-> RewardManager.compute()
-> reset 已结束环境
-> CommandManager.compute()
-> interval events
-> ObservationManager.compute()
-> 返回 observation、reward、terminated、truncated、extras
```

跨 Manager 的数据流只通过官方对象访问：

```text
Scene asset data
    -> Observation / Reward / Termination

CommandManager command
    -> Observation / Reward / Termination

RewardManager / TerminationManager / RecorderManager outputs
    -> Environment step return
    -> Trainer
```

## 实施顺序

1. 保留并整理 `NavSceneCfg`，先验证机器人、LiDAR 和障碍物能独立创建。
2. 完成 `ActionsCfg` 和 `UavVelocityAction`，验证动作维度和控制器状态。
3. 重写 `EventCfg`，只实现机器人 reset。
4. 重写 `CommandsCfg`，只实现目标生成和命令 reset。
5. 重写 `ObservationsCfg`，全部从 Scene 和 Command 读取数据。
6. 重写 `TerminationsCfg`，只返回终止 mask。
7. 将奖励拆成少数清晰的 RewardTerm，历史状态只留在对应 Term 中。
8. 增加 `CurriculumCfg` 和具体课程项；暂时不用时保持空配置。
9. 增加 `RecorderCfg` 和 RecorderTerm，迁移奖励分项及 episode 统计。
10. 最后接入 Trainer 和 Algo，确保 Trainer 只处理 rollout，不访问 MDP 私有对象。

## 验证要求

每完成一个类都执行最小测试：

```text
构造配置
-> 创建环境
-> reset
-> step 一个动作
-> 检查 tensor shape 和 device
-> 检查 done 后局部环境能再次 reset
```

最终必须验证：

```text
无 NavTaskBuffer
无 env._nav_* 任务状态
无重复的任务参数来源
无 Event 直接调用其他 Manager.reset()
Trainer 不读取 MDP 私有字段
```
