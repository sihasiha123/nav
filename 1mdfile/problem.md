# 问题来源

NavTaskBuffer`不是 IsaacLab 官方类，也不是你一开始就明确设计出来的。它是在把任务迁移到`ManagerBasedRLEnv` 时逐步产生的。

Git 历史显示，它第一次出现在提交：

```
e346149  2026-08-12  “mdp完成”
```

当时新建了 `mdp/events.py`，并把以下内容集中放进了一个对象：

```
class NavTaskBuffer:
    target_pos
    target_dir
    height_range
    prev_distance
    prev_drone_vel_w
    reached_goal_once
```

同时通过：

```
env._nav_task_buffer
```

把它挂到环境上，其他 reward、observation、termination 函数再通过：

```
get_nav_task_buffer(env)
```

访问。

它出现的原因大致是：

```
原来的 reset 函数同时生成机器人和目标
    ↓
reward / observation / termination 都需要目标数据
    ↓
为了让这些函数共享数据
    ↓
把所有任务状态集中放进 NavTaskBuffer
```

后来随着功能增加，又继续把奖励分项统计放进了这个 buffer：

```
目标位置
目标方向
高度范围
上一帧距离
上一帧速度
是否到达
奖励分项
```

于是它从一个临时共享对象逐渐变成了“大一统任务状态容器”。

这里的“全局”容易误解。它并不是跨进程或跨所有环境的全局变量，而是：

```
一个挂在 env 上的对象
内部保存 N 个并行环境的状态
```

例如：

```
_nav_task_buffer.target_pos.shape == (num_envs, 3)
```

所以更准确的名称应该是“环境级任务状态缓存”。

后来的提交 `8a1e601` 才开始删除它，把职责拆开：

```
target_pos / target_dir
    → NavTargetCommand

prev_distance / prev_drone_vel
    → NavigationReward

robot 初始状态
    → reset_robot_state
```

因此你现在看到的混乱，本质上不是某一行代码突然写错，而是一个常见的演化过程：

```
先用共享 buffer 快速让模块通信
    ↓
新功能继续往 buffer 里加字段
    ↓
Event、Reward、Observation、Trainer 都依赖它
    ↓
职责边界逐渐消失
```

所以重新设计时，最重要的是从一开始就规定：

> 不创建任务级共享 buffer；目标由 CommandTerm 持有，奖励历史由 RewardTerm 持有，日志由 RecorderManager 持有，环境只保存通用生命周期状态。
>
