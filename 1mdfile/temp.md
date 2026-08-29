# manager

# InteractiveScene

InteractiveScene是 Isaac Lab 对“仿真场景中所有实体”的统一容器，不是 RL Manager

# 数据流转

环境配置 cfg
↓
InteractiveScene 创建场景实体
↓
各 Manager 注册 MDP term
↓
env.reset()
↓
返回 observation
↓
Policy 输出 action
↓
env.step(action)
↓
返回 observation、reward、done、extras


### 第二步：先确定任务定义

先把任务规则写清楚，不写代码：

```
机器人起点：+Y 边界
目标位置：-Y 边界
起点 X：按环境编号分布
起点 Z：指定范围随机
机器人初始速度：0
目标是否重采样：每个 episode 一次
```

如果当前任务就是单侧到单侧，就不要保留四边随机、起点方向、目标方向等暂时不用的参数。

### 第三步：统一配置来源

建立一个简单的任务几何配置：

```
@configclass
class NavTaskCfg:
    map_size = (20.0, 20.0, 6.0)
    boundary_offset = 2.0
    start_z_range = (0.5, 2.5)
```

`Event` 和 `Command` 都读取这个配置，不再各自声明 `map_range` 和 `boundary_offset`。

### 第四步：先重写 reset

`events.py` 只做一件事：

```
reset_robot_state
    设置机器人位置
    设置机器人姿态
    设置机器人速度
    设置关节状态
```

不要在这里：

```
生成目标
初始化奖励历史
保存任务 buffer
记录奖励分项
```

### 第五步：再重写 Command

`commands.py` 只负责：

```
根据当前机器人位置生成目标位置
保存 target_position
可选保存 target_velocity
```

观测、奖励和终止条件统一通过：

```
env.command_manager.get_command("nav_target")
```

获取目标。

### 第六步：逐个实现 MDP 类

建议顺序：

```
SceneCfg
→ NavEnvCfg
→ reset_robot_state
→ NavTargetCommand
→ UavVelocityAction
→ observations
→ terminations
→ rewards
→ recorders
```

每完成一个类，都先测试构造和 reset，不要等全部写完再调试。

### 第七步：删除旧逻辑

确认新流程工作后，再删除：

```
NavTaskBuffer
env._nav_task_buffer
env._nav_reward_term
重复的 map_range 参数
重复的 target_dir 保存
Event 中的目标生成逻辑
```

### 第八步：最后接入动态障碍物和训练

动态障碍物先作为独立资产验证：

```
创建场景
→ 障碍物初始化
→ 推进一步
→ 检查位置变化
```

确认资产运行正常后，再接入环境 step。最后才恢复 PPO trainer。

### 重构的最小目标

先做到这个结构：

```
Event       设置机器人初始状态
Command     生成目标
Action      执行动作
Observation 读取状态和目标
Reward      计算奖励
Termination 判断结束
Scene       保存仿真资产状态
Env         负责装配和生命周期
Trainer     收集 rollout
Algo        更新网络
```

核心原则是：

> 先把单一任务、单一目标、单一 reset 流程跑通，再逐步增加动态障碍物、随机化和复杂奖励。不要一开始就保留所有旧参数和兼容逻辑。
>
