# 奖励函数快速设计方法

## 原理：先做回报预算，再调参数

不要直接猜 `progress`、`collision` 的权重。先把每条 episode 拆成：

```text
过程回报 = progress + velocity + avoidance + height + smoothness + time
终止回报 = success / collision / out_of_bounds / time_out
```

从已有评估 CSV 统计不同终止原因的过程回报分布。设失败过程回报的 95% 分位数为
`Q95_fail`，则终止惩罚的实际值可以先设置为：

```text
failure_penalty = -(Q95_fail + safety_margin)
```

这样可以保证绝大多数失败 episode 的总 Return 为负，而不是依靠拍脑袋放大参数。
成功奖励则设置为略高于成功过程回报的上分位数，使“安全到达”稳定优于“途中移动很多但失败”。

## 针对本项目的快速搜索

采用两阶段、固定场景种子的短训练：

1. **离线校准**：使用当前 run 的分项日志和评估 CSV，估计过程回报范围，生成 6~12 组候选权重。候选值用 Sobol 或 Latin hypercube 覆盖，而不是逐个手动尝试。
2. **短 rollout 筛选**：每组只训练少量 iteration，保持相同初始模型、场景种子和并行环境数。记录 `success_rate`、`dynamic_collision_rate`、`time_out_rate`，以及按终止原因分组的 Return。
3. **约束式评分**：不要用总体 Return 排名，使用类似下面的目标：

   ```text
   score = success_rate
         - 0.8 * dynamic_collision_rate
         - 0.5 * static_collision_rate
         - 0.3 * time_out_rate
   ```

   先淘汰碰撞率超过上限的方案，再在剩余方案中选择成功率最高者。
4. **完整训练确认**：只对前 1~2 组候选进行完整 PPO 训练，并用独立场景种子评估，防止短 rollout 过拟合。

## 可以进一步创新的方向

可以实现一个“终止回报校准器”：每隔固定训练阶段读取最近一批 episode 的过程回报分位数，自动计算下一轮实验的终止惩罚候选值，但在一次完整训练开始后冻结参数。这样利用了项目已有的 Recorder 和评估数据，又避免训练中动态改变奖励造成非平稳目标。

如果以后 PPO 增加约束损失，还可以把碰撞率作为单独 cost，使用拉格朗日乘子自动提高碰撞代价；在当前实现下，先采用离线校准 + 短训练筛选更稳妥。
