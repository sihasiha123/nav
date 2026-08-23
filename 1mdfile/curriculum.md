# 导航课程训练方案（两阶段）

## 1. 先给结论

当前任务不需要按障碍物数量做很多级别。采用两个独立训练阶段：

```text
Stage 1: 200 个静态障碍物，动态障碍物关闭
Stage 2: 200 个静态障碍物 + 动态障碍物，加载 Stage 1 checkpoint 继续训练
```

这样做的原因是：静态障碍物是 `TerrainImporterCfg` 在创建环境时生成的高度场，动态障碍物是 `RigidObjectCollectionCfg` 在创建环境时生成的刚体集合。它们的数量不是 episode 运行过程中可以直接修改的变量。因此，**场景级别的两阶段切换要通过重启环境完成，不能指望 `CurriculumManager` 把 0 个对象变成 100 个对象**。

`CurriculumManager` 仍然有用，但它适合在一个已经创建好的场景内调节参数，例如动态障碍物的速度、运动范围、初始状态随机化强度，或一个已经预创建的障碍物集合的激活数量。

## 2. 当前代码对应的场景

当前 `NavSceneCfg` 的默认值是：

```text
静态高度场: 200 个
动态刚体集合: 100 个
```

其中：

- 静态 200 个由 `HfDiscreteObstaclesTerrainCfg(num_obstacles=200)` 生成；
- 动态障碍物由 `make_global_obstacle_collection_cfg(count=100, ...)` 生成；
- 动态障碍物的运动由 `GlobalObstacleManager` 在每个物理步推进；
- 动态障碍物不是按每个并行环境复制，而是一套全局集合。

所以建议保留静态地图为 200 个，训练只分“无动态”和“有动态”两次启动。

## 3. 两个阶段如何运行

### Stage 1：只学静态绕行

配置目标：

```text
num_obstacles = 200
dynamic_obstacles = None
```

训练目标是先学会稳定飞行、根据 LiDAR 绕过静态障碍物并到达目标。不要在这个阶段改变奖励、网络结构和 PPO 超参数。

推荐验收条件（固定 deterministic eval，至少 3 个 seed）：

```text
成功率 >= 85%
静态碰撞率 <= 12%
越界率 + 超时率 <= 8%
```

达到条件后保存 Stage 1 的 actor、critic、feature extractor 和 `ValueNorm`。

### Stage 2：加入动态障碍物

新建环境进程，仍使用 200 个静态障碍物，并把动态集合设为 100 个：

```text
num_obstacles = 200
dynamic_obstacles = make_global_obstacle_collection_cfg(count=100, ...)
```

从 Stage 1 checkpoint 加载网络参数，使用新的 PPO optimizer 继续训练。当前 `PPO.load()` 加载的是网络和 `ValueNorm`，不加载 optimizer 状态，因此阶段切换时应当把它视为新的 run。

Stage 2 先使用较温和的动态参数，再逐渐恢复最终难度：

```text
前 20% 训练: speed_range=(0.25, 0.45), motion_half_extent=(0.5, 0.5, 0.2)
后 80% 训练: speed_range=(0.25, 0.75), motion_half_extent=(1.0, 1.0, 0.4)
```

推荐验收条件：

```text
成功率 >= 75%
总碰撞率 <= 20%
动态碰撞率 <= 15%
越界率 + 超时率 <= 8%
```

如果 Stage 2 一开始完全不稳定，不要恢复十几个数量级别；只增加一个短暂过渡配置（例如 20 个动态障碍物），确认观测和碰撞判定正确后再回到 100 个。

## 4. ManagerBasedRLEnv 的课程到底是什么

ManagerBasedRLEnv 会从 `cfg.curriculum` 创建 `CurriculumManager`。课程项是一个函数配置：

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class CurriculumCfg:
    dynamic_difficulty = CurrTerm(
        func=mdp.dynamic_difficulty,
        params={"warmup_resets": 200},
    )

class NavEnvCfg(ManagerBasedRLEnvCfg):
    curriculum = CurriculumCfg()
```

课程函数的签名至少是：

```python
def dynamic_difficulty(env, env_ids, warmup_resets):
    ...
    return {"speed_max": ..., "difficulty": ...}
```

实际调用位置在 `ManagerBasedRLEnv._reset_idx()`：

```text
某些并行环境完成 episode
    -> curriculum_manager.compute(env_ids)
    -> scene.reset(env_ids)
    -> reset events
    -> 各 manager reset
```

因此课程函数是“某个子环境 reset 时更新一次”，不是每个 PPO iteration 只调用一次，也不是自动替换整个环境配置。`env_ids` 只表示本次完成 episode 的环境；课程状态如果是全局的，应保存到 `env` 上，而不是只保存在局部变量中。

课程函数返回值只用于记录日志，例如：

```python
return {
    "difficulty": float(env.nav_curriculum.difficulty),
    "speed_max": float(env.nav_curriculum.speed_max),
}
```

它会出现在 `extras["log"]` 中，训练脚本可以写入 W&B。课程函数本身不负责 PPO 更新、不负责保存 checkpoint，也不负责重新创建 USD 或刚体对象。

## 5. 本项目中 Manager 课程应该做什么

第一版只实现一个阶段内动态难度函数，不让它切换静态地图：

```text
Stage 1: 不启用课程项，固定静态 200
Stage 2: 启用课程项，只调动态障碍物运动难度
```

Stage 2 的课程状态可以放在环境对象上：

```python
class NavCurriculumState:
    difficulty = 0.0
    speed_max = 0.45
    motion_extent = 0.5
```

课程函数根据环境自己的 episode 统计更新这些值：

```python
def dynamic_difficulty(env, env_ids, warmup_resets=200):
    state = env.nav_curriculum
    state.reset_count += int(env_ids.numel())
    if state.reset_count >= warmup_resets:
        state.difficulty = min(1.0, state.difficulty + 0.05)
        state.speed_max = 0.45 + 0.30 * state.difficulty
        state.motion_extent = 0.5 + 0.5 * state.difficulty
    return {
        "difficulty": state.difficulty,
        "speed_max": state.speed_max,
        "motion_extent": state.motion_extent,
    }
```

上面只是接口示例。正式实现时，难度提升应依据已完成 episode 的成功率、动态碰撞率和超时率，而不是依据单个 episode。统计器应把窗口统计写入 `env`，因为 curriculum term 无法直接访问 `scripts/train.py` 中的局部统计器。

## 6. 一个重要的观测结构问题

当前 `scripts/train.py`、`scripts/eval.py` 和 `obs_to_tensordict()` 默认读取：

```text
state / lidar / direction / dynamic_obstacle
```

但 `NavEnvCfg.__post_init__()` 在 `dynamic_obstacles is None` 时会把 `dynamic_obstacle` observation term 设为 `None`。如果 Stage 1 直接关闭动态障碍物，观测字典就会少一个 key，而现有 PPO 构建代码会报错。

因此 Stage 1 必须先做一个小的兼容修改，二选一：

1. 保留固定形状的 `dynamic_obstacle` 观测项；没有动态集合时返回全零，保证两个阶段使用完全相同的 observation space 和网络结构；
2. 修改训练、评估和 `obs_to_tensordict()`，让它们在缺少该项时自动补同形状的零张量。

推荐方案 1。这样 Stage 2 可以直接加载 Stage 1 checkpoint，不需要改变第一层网络的输入维度。

## 7. 不采用的做法

- 不使用 30、80、150、200 等静态障碍物数量的连续课程；静态高度场需要重建，收益不抵成本。
- 不在课程函数中销毁、创建 USD 或 `RigidObjectCollection`。
- 不在一个 run 中修改 `num_obstacles` 并假设 PhysX 会自动更新碰撞网格。
- 不用训练 rollout 的单个 iteration 成功率决定晋级；至少收集固定数量的完成 episode，再做 deterministic eval。
- 不同时修改场景数量、奖励和 PPO 网络，否则无法判断性能变化来源。

## 8. 实施顺序

```text
1. 固定 dynamic_obstacle 的观测形状，确保无动态阶段也能运行
2. 增加 Stage 1 / Stage 2 两个环境配置入口
3. Stage 1 训练：静态 200，动态关闭
4. 固定 eval 协议并验收 Stage 1
5. Stage 2 重启环境：静态 200 + 动态 100，加载 Stage 1 checkpoint
6. 只在 Stage 2 内用 CurriculumManager 调速度和运动范围
7. 多 seed eval，记录成功率、静态碰撞率、动态碰撞率和超时率
```

建议 run 目录明确写出场景：

```text
runs/nav_stage1_static200_dynamic000/
runs/nav_stage2_static200_dynamic100/
```

最终应把每个 run 的环境配置、checkpoint、seed 和 eval 报告一起保存。这样“阶段切换”由训练命令和 checkpoint 管理，“阶段内渐进”由 `CurriculumManager` 管理，两者职责清晰，不会把 Manager 课程误当成场景重建器。
