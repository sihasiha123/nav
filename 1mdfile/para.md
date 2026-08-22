# `ppo_20260822_145608` 版本记录

## 训练结果

训练目录：`runs/ppo_20260822_145608`

- 训练迭代：`2000`
- 并行环境：`1024`
- episode 最大时长：`60 s`
- 最后 200 iteration 完成 episode 数：`5314`
- 最后 200 iteration 成功率：`74.52%`
- 最后 200 iteration 总碰撞率：`23.49%`
  - 静态碰撞：`4.31%`
  - 动态碰撞：`19.18%`
- 越界率：`1.69%`
- 超时率：`0.30%`
- Done Return：`+18.02`

## Eval 结果

正式评估目录：`output/eval_20260822_153807`

- checkpoint：`checkpoint_2000.pt`
- 动作模式：deterministic Beta mean
- 评估 episode：`1024`
- 成功率：`76.86%`
- 总碰撞率：`22.07%`
  - 静态碰撞：`0.39%`
  - 动态碰撞：`21.68%`
- 越界率：`0%`
- 超时率：`1.07%`
- 平均 Return：`+17.78`
- 成功 episode 平均到达时间：约 `23.45 s`

结论：这是当前最好的 checkpoint。训练和 deterministic eval 的成功率接近，基础导航已经学会；剩余主要问题是动态障碍物碰撞。

## 场景参数

### 静态场景

- 兜底 ground：`300 m × 300 m`
- 障碍地形：`40 m × 40 m`
- 地形生成 seed：`0`
- 静态障碍物数量：`200`
- 静态障碍物宽度范围：`0.4 ~ 1.1 m`
- 静态障碍物高度：`6.0 m`

### 动态障碍物

- 动态障碍物数量：`100`
- 障碍物尺寸：`0.5 m × 0.5 m × 1.0 m`
- 初始高度范围：`1.0 ~ 2.5 m`
- 固定网格锚点，锚点附近随机航点运动
- 运动范围：相对锚点 `±(1.0, 1.0, 0.4) m`
- 速度范围：`0.25 ~ 0.75 m/s`
- 到达航点阈值：`0.05 m`
- 动态障碍物不随单个无人机 episode reset

### 任务和仿真

- 起点：`+Y` 边
- 目标：`-Y` 边
- 起点 X：按环境编号均匀分布
- 起飞高度：`0.5 ~ 2.5 m`
- 允许飞行高度：`0.2 ~ 4.0 m`
- 仿真 dt：`1/60 s`
- decimation：`1`
- LiDAR：`36` 个水平方向 × `4` 个垂直通道，量程 `4 m`

## 动作和速度限制

策略输出经过 Beta 分布映射为三维世界系速度指令：

```text
vx, vy, vz ∈ [-2.0, 2.0] m/s
```

- PPO actor `action_limit = 2.0`
- 环境动作缩放 `scale = 1.0`
- `max_velocity = None`，没有额外的环境裁剪
- 因此是每个速度分量分别限制在 `[-2, 2] m/s`
- 三维速度合成的理论最大模长约为 `3.46 m/s`
- yaw 模式：`velocity_vector`
- yaw 角速度限制：`4.0 rad/s`

底层速度控制器主要参数：

```text
max_feedback_accel = 20.0 m/s²
speed_gain = [10, 10, 20]
pose_gain = [18, 18, 20]
rate_gain = [180, 180, 200]
body_rate_bound = [-12, 12]
thrust_ctrl_delay = 0.03 s
```

## 奖励参数

当前使用单个 `navigation_reward` Manager term，Manager weight 为 `1.0`。首次安全到达奖励已从 `50` 调整为 `120`：

```text
progress                 4.0 × 距离进展
goal_velocity            0.5 × 目标方向速度（裁剪 ±2）
static_avoidance        -6.0 × 静态风险
dynamic_avoidance      -10.0 × 动态风险
height                  -2.0 × 高度偏离
smoothness              -0.05 × 速度变化
time                    -0.01 / step
goal_first              +120
goal_reached            +0.5
collision               -120
out_of_bounds           -120
```

Isaac Lab RewardManager 会将 Manager term 乘以 `dt=1/60` 后作为环境 reward。

## PPO 超参数

### Rollout 和更新

```text
training_frame_num = 32
training_epoch_num = 4
num_minibatches = 16
gamma = 0.99
gae_lambda = 0.95
max_grad_norm = 5.0
entropy_loss_coefficient = 0.001
```

### Actor/Critic

```text
actor learning_rate = 0.0005
critic learning_rate = 0.0005
feature_extractor learning_rate = 0.0005
actor clip_ratio = 0.1
critic clip_ratio = 0.1
critic value_loss_coefficient = 1.0
```

Actor 使用 Beta 分布输出三个归一化动作，训练时随机采样，eval 默认使用分布均值。

### 网络结构

- LiDAR encoder：`Conv2d 4` → `Conv2d 16` → `Conv2d 16` → `Linear 128` → `LayerNorm`
- 动态障碍物 encoder：`Linear 128` → `Linear 64` → `LayerNorm`
- 特征融合：`Linear 256` → `Linear 256` → `LayerNorm`
- Actor：Beta 分布的 `alpha/beta` 两个线性输出头
- Critic：`Linear 1`
- 使用 `ValueNorm` 归一化 value target

## 下一步

当前不再提高到达奖励，下一步只针对动态障碍物碰撞做诊断和单变量调参，并使用至少 `1024` 个 episode 的 deterministic eval 验证。
