# ManagerBasedRLEnv 任务结构重构方案

## 1. 结论

`dynamic.py` 不需要拆分。它是动态障碍物这一类动态资产的完整领域模块，资产生成、运动逻辑、运行时状态、全局集合 reset 和 ActionTerm 接入放在一起是合理的，类似机器人控制器模块。

当前真正的问题是主任务的 Manager term 混用：`reset_nav_task()` 同时负责机器人 reset、目标采样和任务状态初始化；目标在概念上属于 command，但 `CommandManager` 没有启用。

重构目标：

```text
EventTerm       episode/场景状态事件
CommandTerm     任务目标生成（启用时）
ActionTerm      策略动作执行
ObservationTerm 策略输入生成
RewardTerm      行为评价
TerminationTerm episode 结束判断
CurriculumTerm  难度调整
TaskState       跨 step episode 状态
dynamic.py      动态障碍物完整资产领域
```

## 2. 官方 Manager 边界

| 配置 | Manager | 职责 |
| --- | --- | --- |
| `scene` | `InteractiveScene` | 创建 UAV、地形、LiDAR、动态障碍物 |
| `commands` | `CommandManager` | 生成目标位置、目标速度等指令，可选 |
| `actions` | `ActionManager` | 把策略输出转换为物理动作 |
| `observations` | `ObservationManager` | 生成策略观测 |
| `events` | `EventManager` | startup/reset/interval 状态修改 |
| `rewards` | `RewardManager` | 计算即时奖励 |
| `terminations` | `TerminationManager` | 判断成功、碰撞、越界、超时 |
| `curriculum` | `CurriculumManager` | reset 时调整难度，可选 |
| `recorders` | `RecorderManager` | 记录轨迹和诊断数据，可选 |

`GlobalObstacleManager` 是项目自己的动态资产运行时系统，不是 Isaac Lab 官方 Manager。无需为了名字或形式改写它。

## 3. 当前混用点

`reset_nav_task()` 当前做：

```text
采样 UAV 起点
根据起点生成目标位置和方向
写入 UAV 初始状态
写入 NavTaskBuffer
```

应区分：

```text
起点、速度、姿态、历史状态初始化 = EventTerm
目标位置/目标速度生成 = CommandTerm
```

没有 CommandManager 时，这两部分暂时位于一个 reset event 可以运行，但必须明确这是“无 CommandManager 模式”。

`dynamic.py` 中以下内容可以保持整体：

```text
GlobalObstacleMotionCfg
GlobalObstacleManager
GlobalObstacleMotionAction / Cfg
GlobalRigidObjectCollection
make_global_obstacle_collection_cfg
```

这属于领域资产模块，不属于 Manager 混用问题。

## 4. 目标目录

```text
mdp/
├── __init__.py          # 公开 term 和必要资产工厂
├── actions.py           # UAV ActionTerm
├── observations.py      # ObservationTerm
├── rewards.py           # RewardTerm
├── terminations.py      # TerminationTerm
├── events.py            # EventTerm
├── commands.py          # 启用 CommandManager 时新增
├── curriculum.py        # 启用 CurriculumManager 时新增
├── task_state.py        # NavTaskBuffer
├── reward_state.py      # 后续拆 reward_components
├── geometry.py          # 纯几何函数
├── sensor_queries.py    # 公共 LiDAR/传感器查询
└── dynamic.py           # 动态障碍物完整资产模块，保持整体
```

规则不是每个文件只能有一个类，而是一个官方 term 只能归属一个 Manager；一个动态资产可以在一个领域模块中包含生成、运动、状态和接入适配器；公共查询不能藏在 Observation 文件的私有函数里。

## 5. Event 和 Command 的两种模式

### 模式 A：当前基线，不启用 CommandManager

当前目标拓扑固定为 `+Y -> -Y`，明确配置：

```python
class NavEnvCfg(ManagerBasedRLEnvCfg):
    commands = None
```

拆出一个无副作用的目标函数：

```python
def sample_nav_target(start_pos, map_range, boundary_offset):
    ...
```

保留 reset 流程：

```text
reset_nav_task
    -> 采样起点
    -> 调用 sample_nav_target
    -> 写入 robot 和 NavTaskBuffer
```

文档明确：目标只在 episode reset 时生成，因此有意不使用 CommandManager。这不是遗漏。

### 模式 B：正式启用 CommandManager

目标需要独立重采样、目标速度或多种任务指令时，再实现 `NavTargetCommand(CommandTerm)`：

```text
NavTargetCommand       生成 target_pos/target_dir
reset_nav_task         只初始化 UAV 和历史状态
observations/rewards   读取同一个目标
terminations           读取同一个目标
```

必须遵守 Isaac Lab 当前 reset 顺序：

```text
curriculum_manager.compute(env_ids)
scene.reset(env_ids)
reset events
各 manager reset，包括 command_manager.reset(env_ids)
```

所以 reset event 不能依赖本次刚生成的 command。推荐：event 先设置起点，`NavTargetCommand.reset()` 再根据起点生成目标并同步 task state；当前固定任务的初始 yaw 可直接使用固定 `-Y` 方向。

目标来源必须唯一，不能 event 和 command term 各采样一次。

## 6. 文件职责

`events.py`：只放 startup/reset/interval 事件，不定义 CommandTerm，不计算 reward。

`commands.py`：只放 CommandTerm，负责目标采样、重采样和 command 缓存，不执行 UAV 物理 reset。

`actions.py`：只放策略 ActionTerm。UAV 动作保留；动态障碍物 ActionTerm 可以继续由 `dynamic.py` 提供，作为动态资产接入 ActionManager 的适配层。

`observations.py`：只放 state、LiDAR、方向、动态障碍物观测，不修改环境或生成目标。

`rewards.py`：只计算奖励，并写入 reward 统计状态。

`terminations.py`：只返回终止 mask，不写状态、不触发 reset。

`curriculum.py`：只调整已存在的速度、运动范围或随机化参数。

`task_state.py`：只保存目标、历史距离、历史速度和首次到达标记；`reward_components` 后续迁移到 `reward_state.py`。

`dynamic.py`：整体负责动态障碍物生成、运动、全局集合 reset 行为和 ActionTerm 接入。

## 7. 迁移表

| 当前实现 | 重构处理 |
| --- | --- |
| `reset_nav_task` | 保留在 `events.py`，先拆出 `sample_nav_target` |
| 目标采样 | 模式 A 保持纯函数；模式 B 迁移到 `NavTargetCommand` |
| `NavTaskBuffer` | 保留在 `task_state.py` |
| `reward_components` | 后续迁移到 `reward_state.py` |
| `UavVelocityAction` | 保留在 `actions.py` |
| `GlobalObstacleMotionAction` | 保留在 `dynamic.py` |
| `GlobalObstacleManager` | 保留在 `dynamic.py` |
| `state_obs` 等 | 保留在 `observations.py` |
| `_lidar_distance` | 移到 `sensor_queries.py` |
| `_obstacle_size` | 放到动态资产查询接口或公共查询模块 |
| `navigation_reward` | 保留在 `rewards.py` |
| `success`、碰撞函数 | 保留在 `terminations.py` |

## 8. 标准流程

当前模式 A：

```text
创建环境 -> Scene 建立资产 -> 各 Manager 建立 terms
reset -> scene.reset -> reset_nav_task 生成起点/目标并写 task_state
      -> 各 Manager reset -> 生成 observation
step  -> ActionManager -> dynamic.py ActionTerm -> 物理仿真
      -> TerminationManager -> RewardManager
      -> done env_ids 执行 reset -> ObservationManager
```

未来模式 B：

```text
reset -> event 初始化 robot/历史状态
      -> CommandManager.reset 生成目标并同步 task_state
      -> observation
step  -> observations/rewards/terminations 读取同一目标
```

## 9. 分阶段重构计划

### Phase 0：冻结行为契约

记录 observation keys/shapes、action shape、reward 缩放、termination names、reset 后起点/目标/yaw，以及动态障碍物位置轨迹。

### Phase 1：明确模式 A

- 在 `NavEnvCfg` 显式写出 `commands = None`；
- 将 `reset_nav_task` 拆成 reset 流程和 `sample_nav_target` 纯函数；
- 说明目标是 reset-time task state，不是 CommandManager command；
- 不改变训练行为。

### Phase 2：整理状态和公共查询

- 保留 `task_state.py`；
- 将 `reward_components` 迁移到 `reward_state.py`，或明确为兼容字段；
- 将 `_lidar_distance` 移到 `sensor_queries.py`；
- 禁止 reward/termination 从 `observations.py` 借用私有函数。

### Phase 3：需要时启用模式 B

只有目标需要独立重采样或目标类型增多时，才实现 `commands.py` 和 `NavTargetCommand`，并遵守 command reset 晚于 reset event 的顺序。

### Phase 4：加入课程和记录

Manager 边界稳定后，再新增 `curriculum.py` 和 recorder term。课程只调整动态资产已有参数。

## 10. 验证标准

每个阶段执行 `num_envs=2/4` 的 reset-step-close smoke test，分别覆盖静态和动态场景，并检查 observation/action/reward/termination shape、动态障碍物每物理步只推进一次，以及固定 seed 下重构前后的 episode 结果。

完成标准：

```text
Event 不再被误认为 CommandManager
CommandManager 未启用时有明确理由
启用 CommandManager 时目标只有一个来源并遵守 reset 顺序
dynamic.py 作为动态资产整体保留
各官方 term 只做自己的计算
task_state 与 reward statistics 的关系明确
公共传感器查询不藏在 observations.py 私有函数中
重构前后行为和观测结构一致
```

最终原则：Event 修改状态，Command 生成目标，Action 执行动作，Observation 组织输入，Reward 评价行为，Termination 判断结束，Curriculum 调整难度，TaskState 保存 episode 状态，`dynamic.py` 管理动态障碍物这一完整资产领域。
