# ManagerBasedRLEnv 使用方法

## 1. 基本概念

`ManagerBasedRLEnv` 是 Isaac Lab 的 Manager-based 强化学习环境。环境本身负责调度仿真，任务逻辑由配置类和 MDP 函数组成：

```text
Scene                  场景、机器人、传感器、障碍物
ActionManager          处理并应用策略动作
ObservationManager     生成观测
RewardManager          计算奖励
TerminationManager     计算成功、碰撞、越界、超时
EventManager           执行 startup/reset/interval 事件
CurriculumManager      调整任务难度
```

通常不需要重写环境的 `step()`，而是配置这些 Manager。

## 2. 配置环境

当前导航环境继承 `ManagerBasedRLEnvCfg`：

```python
@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    scene = NavSceneCfg(...)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 60.0
```

每个 term 通过 `func` 指向一个 MDP 函数：

```python
class RewardsCfg:
    navigation = RewTerm(func=mdp.navigation_reward, weight=1.0)

class TerminationsCfg:
    success = DoneTerm(func=mdp.success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

MDP 函数接收 `env`，返回当前所有并行环境的结果，例如成功函数返回形状通常为 `(num_envs,)` 的布尔张量。

## 3. 创建、reset 和 step

任务在 `nav/tasks/manager_based/nav/__init__.py` 注册为 `Template-Nav-v0`。使用方式：

```python
import gymnasium as gym
import torch
import nav.tasks  # 触发任务注册

env_cfg = parse_env_cfg("Template-Nav-v0", device="cuda:0", num_envs=1024)
env = gym.make("Template-Nav-v0", cfg=env_cfg)

obs, info = env.reset()
for _ in range(1000):
    action = torch.zeros(
        (env.unwrapped.num_envs, env.action_space.shape[-1]),
        device=env.unwrapped.device,
    )
    obs, reward, terminated, truncated, extras = env.step(action)
    done = terminated | truncated

env.close()
```

`gym.make()` 可能返回 wrapper。访问 Isaac Lab 属性时使用 `env.unwrapped`，例如 `env.unwrapped.device`、`env.unwrapped.observation_manager`。

## 4. 一次 step 的生命周期

```text
ActionManager.process_action(action)
    -> ActionManager.apply_action()
    -> 推进 decimation 次 Isaac Sim 物理
    -> TerminationManager.compute()
    -> RewardManager.compute()
    -> 找出已结束的 env_ids
    -> CurriculumManager.compute(env_ids)
    -> scene.reset(env_ids)
    -> EventManager.apply(mode="reset", env_ids)
    -> 重置各 Manager buffer
    -> interval events / commands
    -> ObservationManager.compute()
    -> 返回 obs、reward、terminated、truncated、extras
```

并行环境是异步 reset 的，一次 step 可能只有部分环境结束；MDP 函数和 reset event 必须正确处理 `env_ids`。

## 5. 当前导航任务的 Manager

`NavSceneCfg` 创建 200 个静态高度场障碍物、UAV、LiDAR，以及默认 100 个全局动态障碍物。静态地形和动态刚体集合都在环境创建时生成，运行中不能靠课程函数改数量。

当前动作项包括：

```python
uav_velocity = mdp.UavVelocityActionCfg(...)
global_obstacle_motion = mdp.GlobalObstacleMotionActionCfg()
```

无人机动作占 3 个策略维度；动态障碍物动作项的 `action_dim` 是 0，只借用每个物理步的回调推进障碍物。

策略观测当前为：

```text
state / lidar / direction / dynamic_obstacle
```

奖励由 `mdp.navigation_reward` 计算；终止项包括静态碰撞、动态碰撞、越界、成功和超时；`reset_nav_task` 负责设置起点、目标和任务状态。

## 6. CurriculumManager 的使用

课程项必须放到 `cfg.curriculum`：

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class CurriculumCfg:
    dynamic_difficulty = CurrTerm(
        func=mdp.dynamic_difficulty,
        params={"warmup_resets": 200},
    )

@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    curriculum = CurriculumCfg()
```

课程函数至少接收 `env` 和 `env_ids`：

```python
def dynamic_difficulty(env, env_ids, warmup_resets=200):
    state = env.nav_curriculum
    state.reset_count += int(env_ids.numel())
    if state.reset_count >= warmup_resets:
        state.difficulty = min(1.0, state.difficulty + 0.05)
    return {"difficulty": state.difficulty}
```

调用时机是 episode 结束后的 reset：

```text
env_ids 完成 episode
    -> curriculum_manager.compute(env_ids)
    -> scene.reset(env_ids)
    -> reset events
```

返回值只用于 `extras["log"]` 日志。课程函数不负责 PPO 更新、保存 checkpoint、重新创建 USD，或改变已经创建的障碍物数量。课程状态应保存到 `env` 对象上；不能依赖 `scripts/train.py` 中的局部变量。

## 7. 本项目的两阶段用法

```text
Stage 1: 200 静态障碍物，动态障碍物关闭，不启用课程
Stage 2: 重启环境，200 静态 + 100 动态，加载 Stage 1 checkpoint
         CurriculumManager 只调动态速度和运动范围
```

阶段切换使用“重启环境 + 加载网络和 `ValueNorm`”。当前 `PPO.load()` 不加载 optimizer 状态，因此 Stage 2 使用新的 optimizer 是正常的。

注意：当前训练和评估代码固定读取 `dynamic_obstacle` 这个观测 key。如果 Stage 1 把动态观测项设为 `None`，观测结构会改变，Stage 1 checkpoint 不能直接用于 Stage 2。推荐无动态障碍物时仍保留同形状观测，并返回全零张量。

## 8. 调试检查

```python
unwrapped = env.unwrapped
print(unwrapped.observation_manager)
print(unwrapped.action_manager)
print(unwrapped.reward_manager)
print(unwrapped.termination_manager)
print(unwrapped.curriculum_manager)

obs, reward, terminated, truncated, extras = env.step(action)
print(reward.shape, terminated.shape, truncated.shape)
print(extras.get("log", {}))
```

常见问题：观测 key 缺失时检查 `NavEnvCfg.__post_init__()`；动作维度异常时确认零维动态障碍物动作没有被算入策略动作；课程不生效时确认 `curriculum = CurriculumCfg()` 已配置；障碍物数量不变时重新创建环境，而不是修改课程函数。

## 9. 记忆方式

```text
配置类定义模块
    -> Manager 创建并持有模块
    -> ManagerBasedRLEnv.step() 按固定顺序调度
    -> MDP 函数计算局部逻辑
    -> env_ids 处理异步 reset
    -> extras["log"] 输出日志
```

`CurriculumManager` 是难度调节器，不是场景生成器。

## 11. CommandManager 是什么

`CommandManager` 用来生成任务层的目标或指令，例如目标位置、目标速度、目标姿态。它不是策略动作，也不是底层控制器：

```text
CommandManager       生成目标：去哪里、以多快移动
ObservationManager   把目标或目标误差提供给策略
Policy               输出动作
ActionManager        执行动作
```

它由 `CommandTermCfg`、`CommandTerm` 和 `CommandManager` 组成。每个 `CommandTerm` 保存一个形状为 `(num_envs, command_dim)` 的 `command` 张量，并按 `resampling_time_range` 定期重新采样。

生命周期为：

```text
reset -> CommandTerm.reset(env_ids) -> 为这些环境生成初始 command
step  -> CommandManager.compute(dt) -> 更新指标并按时间重新采样
```

最小配置示例：

```python
from isaaclab.managers import CommandTermCfg

@configclass
class CommandsCfg:
    target = CommandTermCfg(
        class_type=TargetCommand,
        resampling_time_range=(5.0, 10.0),
    )

@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):
    commands = CommandsCfg()
```

自定义 `TargetCommand` 需要继承 `CommandTerm`，实现 `command` 属性，以及 `_resample_command()`、`_update_command()` 和 `_update_metrics()`。其他 MDP 函数可以这样读取：

```python
target = env.command_manager.get_command("target")
```

当前导航项目没有配置 `commands`，所以 `CommandManager` 是空的。目标点由 `reset_nav_task` 写入 `NavTaskBuffer.target_pos`，`direction_obs` 和奖励函数直接读取这个 buffer；这种写法可以正常工作，不必为了使用 Manager 强行增加 command term。

注意：`VelocityController` 中的 `self.command` 是底层控制器内部的 `[yaw, vx, vy, vz]` 控制量，不是 Isaac Lab 的 `CommandManager` command。它由 `ActionManager` 的 UAV 速度动作产生。
