# NavigationReward 旧版备份

本文档备份修改前的 `mdp.NavigationReward` 奖励函数，作为当前奖励修改前的恢复依据。旧版实现文件为：

```text
source/nav/nav/tasks/manager_based/nav/mdp/rewards.py
```

## 旧版计算流程

每个环境、每个 step 计算：

```text
目标距离进展
目标方向速度
静态 LiDAR 风险
动态障碍物风险
高度偏离
速度平滑
时间代价
安全到达奖励
碰撞惩罚
越界惩罚
```

旧版公式为：

```text
progress          = 4.0 × (previous_distance - distance)
goal_velocity     = 0.5 × clamp(dot(drone_velocity, target_direction), -2, 2)
static_avoidance  = -6.0 × relu(1.2 - static_clearance)^2
dynamic_avoidance = -10.0 × relu(1.5 - dynamic_clearance)^2
height            = -2.0 × height_violation^2
smoothness        = -0.05 × ||velocity - previous_velocity||
time              = -0.01
goal_first        = +120.0  （首次安全到达目标）
goal_reached      = +0.5    （安全处于目标半径内）
collision         = -120.0  （静态或动态碰撞）
out_of_bounds     = -120.0  （飞行高度超出范围）
```

总奖励：

```text
reward = progress
       + goal_velocity
       + static_avoidance
       + dynamic_avoidance
       + height
       + smoothness
       + time
       + goal_first
       + goal_reached
       + collision
       + out_of_bounds
```

## 旧版参数

```text
lidar_range              4.0 m
static_safe_distance     1.2 m
dynamic_safe_distance    1.5 m
goal_radius              0.5 m
z_min                    0.2 m
z_max                    4.0 m
out_of_bounds_penalty    120.0
```

动态碰撞使用最近的 5 个障碍物计算风险，但终止判断检查全部动态障碍物。奖励 Term 自身保存以下跨 step 状态：

```text
prev_distance
prev_drone_vel_w
reached_goal_once
_initialized
reward_components
```

## 旧版缩放问题

Isaac Lab `RewardManager` 会执行：

```python
value = reward_term(...) * weight * dt
```

当前 `dt=1/60`，因此旧版终止项实际进入环境 Return 的单步值约为：

```text
goal_first       +120 × 1/60 = +2
collision        -120 × 1/60 = -2
out_of_bounds    -120 × 1/60 = -2
```

这会导致无人机在碰撞前积累较多进展和速度奖励，即使最后碰撞，episode Return 仍可能为正。该问题由当前 `NavigationReward` 的终止奖励调整重点修正。

## 旧版记录的分项

旧版通过 `reward_components` 暴露以下分项，记录值已经乘以 `step_dt`，与 `env.step()` 返回的奖励尺度一致：

```text
progress
goal_velocity
static_avoidance
dynamic_avoidance
height
smoothness
time
goal_first
goal_reached
collision
out_of_bounds
total
```

当前版本保持旧版分项名称，并额外记录 `time_out`，便于区分碰撞、越界和纯超时失败。
