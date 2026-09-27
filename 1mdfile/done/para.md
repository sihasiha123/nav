# 训练参数

本文档记录当前 `Template-Nav-v0` 的代码参数基线。实验结果应单独记录在对应的 run 和 eval 目录中，不在这里混用。

## 启动参数

```text
--task             Template-Nav-v0
--algo             ppo
--agent            ppo_cfg_entry_point
--num_envs         默认使用环境配置值 1024
--seed             默认不覆盖环境 seed
--max_iterations   1000
--save_interval    100
--log_dir          runs
```

常用训练命令：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --save_interval 100 \
  --headless
```

训练使用 W&B offline 模式，日志保存在对应的 `runs/<run_name>/wandb/` 目录。

## 环境与仿真

```text
并行环境数             1024
episode 最大时长       60 s
仿真 dt                1/60 s
decimation             1
环境 step dt           1/60 s
replicate_physics      True
filter_collisions      True
```

### 静态场景

```text
地面尺寸               300 m × 300 m
训练地形尺寸           40 m × 40 m
地形生成 seed          0
静态障碍物数量         200
静态障碍物宽度范围     0.4～1.1 m
静态障碍物高度         6.0 m
```

### 动态障碍物

```text
数量                   100
尺寸                   0.5 m × 0.5 m × 1.0 m
中心高度范围           1.0～2.5 m
运动范围               相对锚点 ±(1.0, 1.0, 0.4) m
速度范围               0.25～0.75 m/s
航点到达阈值           0.05 m
```

动态障碍物是共享的全局 `RigidObjectCollection`，不随单个机器人 episode reset。运动状态由 `GlobalObstacleManager` 保存并在物理步中更新。

## 任务参数

```text
无人机起点             +Y 边
无人机目标             -Y 边
起点 X                 按环境编号均匀分布
起飞高度范围           0.5～2.5 m
允许飞行高度           0.2～4.0 m
目标横向坐标           每个环境原点的 y - 22.0 m
目标重采样时间         1e9～1e9 s（实际固定目标）
目标成功半径           0.5 m
```

LiDAR 参数：

```text
水平通道               36
垂直通道               4
水平角分辨率           10°
垂直视场               -10°～20°
量程                   4.0 m
```

## 动作与控制器

策略输出 3 维归一化 Beta 动作，经 PPO 映射为世界坐标系速度：

```text
vx, vy, vz             [-2.0, 2.0] m/s
动作维度               3
动作 scale              1.0
环境 max_velocity       None
```

`max_velocity=None` 表示动作项本身不再裁剪；`[-2, 2]` 的限制来自 PPO actor 的 `action_limit=2.0`。三维速度分量的理论最大模长约为 `3.46 m/s`。

速度控制器参数：

```text
yaw_mode                       velocity_vector
yaw_from_velocity_threshold    1e-3
yaw_rate_limit                 4.0 rad/s
max_feedback_accel             20.0 m/s²
speed_gain                     [10, 10, 20]
pose_gain                      [18, 18, 20]
rate_gain                      [180, 180, 200]
body_rate_bound                [-12, 12]
thrust_ctrl_delay              0.03 s
inertia_diag                   (0.0015, 0.0020, 0.0040)
use_physx_inertia              False
```

## 观测配置

策略观测保持字典结构，不拼接成单一向量：

```text
state                8 维
lidar                (1, 36, 4)
direction            (1, 3)
dynamic_obstacle     (1, 5, 10)
```

动态障碍物观测最多选择最近的 5 个对象，每个对象包含相对位置、距离、相对速度和尺寸特征。

## 奖励与终止

当前奖励使用一个 `NavigationReward` term，Manager weight 为 `1.0`。修改前的奖励公式已备份在 `1mdfile/temp.md`：

```text
progress              4.0 × 距离进展
goal_velocity         0.5 × 朝目标速度（裁剪到 ±2）
static_avoidance     -6.0 × 静态风险
dynamic_avoidance   -10.0 × 动态风险
height               -2.0 × 高度偏离
smoothness          -0.05 × 速度变化
time                -0.01 / step
goal_first         +1200（经 dt 缩放后约 +20）
goal_reached         +30（经 dt 缩放后约 +0.5）
collision         -1200（经 dt 缩放后约 -20）
out_of_bounds     -1200（经 dt 缩放后约 -20）
```

Isaac Lab `RewardManager` 会将 reward term 乘以环境 `step_dt=1/60`。因此终止项的实际单步贡献需要按 `dt` 换算，不能只看代码中的原始数值。

终止项：

```text
static_collision
dynamic_collision
out_of_bounds
success
time_out             time_out=True
```

## PPO 参数

配置文件：`source/nav/nav/tasks/manager_based/nav/agents/ppo.yaml`

```text
training_frame_num       32
training_epoch_num       4
num_minibatches          16
gamma                     0.99
gae_lambda                0.95
max_grad_norm             5.0
entropy_loss_coefficient  0.001
```

优化器与裁剪：

```text
feature_extractor learning_rate   0.0005
actor learning_rate                0.0005
critic learning_rate               0.0005
actor clip_ratio                   0.1
critic clip_ratio                  0.1
value_loss_coefficient              1.0
```

## 网络结构

```text
LiDAR encoder       LazyConv2d(4) → LazyConv2d(16) → LazyConv2d(16)
                    → LazyLinear(128) → LayerNorm(128)

动态障碍物 encoder  LazyLinear(128) → Linear(64) → LayerNorm(64)

特征融合            LazyLinear(256) → Linear(256) → LayerNorm(256)

Actor               Beta alpha/beta 两个输出头，动作维度 3
Critic              Linear(1)
ValueNorm           beta=0.995，epsilon=1e-5
```

训练阶段使用 Beta 分布随机采样动作；评估默认使用 Beta 分布均值动作，使用 `--stochastic` 才进行随机采样。

## 课程与记录器状态

当前 `NavEnvCfg` 中的 `CurriculumCfg` 为空，默认训练不启用主动课程项；因此正常启动时应看到：

```text
Curriculum Manager contains 0 active terms
```

如果重新启用课程，必须单独记录课程阶段、激活障碍物数量和最终难度评估结果。也不要把历史课程实验结果与当前参数基线混在本文件中。

`RecorderCfg.reward_components` 默认关闭，不会自动创建数据集记录。

## 评估约定

`scripts/eval.py` 会：

```text
关闭 CurriculumManager
固定使用完整动态障碍物场景
默认使用确定性 Beta 均值动作
默认每个并行环境评估 1 个 episode
```

因此比较不同 checkpoint 时，应保持以下参数一致：

```text
--num_envs 1024
--episodes_per_env 1
--seed 0
不使用 --stochastic
```

评估重点同时观察 `success_rate`、`dynamic_collision_rate`、`static_collision_rate`、`time_out_rate` 和 `return_mean`，不能只用 Return 判断导航性能。
