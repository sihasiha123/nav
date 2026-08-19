# 代码修改讨论记录

本文档用于先讨论、再实现、最后审查代码。

- 只记录已经确认的设计决定和会直接影响实现的事实。
- 尚未确认的方案保留在讨论中，不作为代码生成依据。
- 在明确要求生成代码前，不根据本文档提前修改业务代码。

## `eval.py` 设计

### 当前代码事实

- 项目当前没有 `scripts/eval.py`。
- `PPO.load()` 可以加载 feature extractor、actor、critic 和 ValueNorm。
- `PPOActor.deterministic()` 已实现 Beta 分布均值动作，但 `PPO` 还没有对外提供确定性推理接口。
- 环境会并行、异步地结束和自动重置 episode。
- 终止项包括成功、静态碰撞、动态碰撞、高度越界和超时。

### 已确认原则

1. `eval.py` 只执行推理和统计，不更新网络参数。
2. 评估统计以完整 episode 为单位，不能按 rollout 或 iteration 代替。
3. 每个并行环境完成相同数量的 episode 后结束评估，避免“最先完成的 N 个 episode”对短 episode、碰撞和成功样本产生选择偏差。
4. 并行环境各自维护 episode return、episode length 和终止结果，互不影响。

### NavRL `eval.py` 参考结论

NavRL 的评估流程为：

```text
加载 checkpoint
    -> 切换固定评估场景
    -> 使用动作分布均值进行确定性推理
    -> 所有并行环境各完成一个 episode
    -> 提取每个环境第一次 done 时的统计
    -> 汇总平均指标并录制视频
```

#### 值得复用的设计

1. 使用动作分布均值执行确定性评估。NavRL 使用 `ExplorationType.MEAN`，本项目对应 `PPOActor.deterministic()`。
2. 固定评估 seed，使不同 checkpoint 在相同随机条件下进行比较。
3. 不在任意一个环境完成时停止，而是保证每个并行环境都贡献相同数量的完整 episode。
4. NavRL 当前相当于 `episodes_per_env=1`；本项目应支持每个环境完成可配置的 K 个 episode。
5. 训练场景和评估场景可以分开配置。NavRL 的评估场景是起点位于 `+Y`、目标位于 `-Y`，X 坐标均匀分布。
6. episode return 和 episode length 由环境逐步累计，评估流程只在 episode 完成时提取结果。

#### 不能直接照搬的部分

1. NavRL 将 checkpoint 路径写死在代码中，本项目必须通过命令行传入。
2. NavRL 的独立评估入口仍创建 `SyncDataCollector` 并反复调用完整评估，结构冗余。本项目的独立 `eval.py` 只需要运行一次明确的评估循环。
3. NavRL 默认强制渲染并录制视频，会降低批量评估速度。本项目的视频录制应为可选功能。
4. NavRL 只保存汇总平均值，没有保存逐 episode 原始数据，不利于比较不同 checkpoint 或定位特定环境的问题。
5. NavRL 没有分别统计静态碰撞、动态碰撞、越界和超时。
6. NavRL 的 `reach_goal` 不是 episode 终止条件，只保存终止时刻的瞬时值，不能严格表示整个 episode 是否曾到达目标。本项目已经将 `success` 定义为 DoneTerm，应直接使用终止管理器的结果。
7. NavRL 没有明确调用策略网络的 `eval()`。本项目评估时应同时调用 `agent.eval()` 和 `torch.inference_mode()`。

### 当前评估输出方向

每个并行环境完成 K 个 episode，并为每个 episode 保存：

```text
env_id
episode_index
return
length_steps
duration_seconds
termination_reason
```

整体汇总至少包括：

```text
episode_count
success_count / success_rate
static_collision_count / static_collision_rate
dynamic_collision_count / dynamic_collision_rate
collision_count / collision_rate
out_of_bounds_count / out_of_bounds_rate
time_out_count / time_out_rate
return_mean / return_min / return_max
episode_length_mean
success_duration_mean
```

评估结果保存为：

```text
summary.json
episodes.csv
```

视频录制作为可选功能，不影响默认的无渲染批量评估。


# 设计方案
