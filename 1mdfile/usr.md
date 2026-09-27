# 命令行指令

项目目录：

```bash
cd /home/robot/nav
```

## 环境

新终端初始化：

```bash
conda activate env_isaaclab
source /home/robot/IsaacLab/_isaac_sim/setup_conda_env.sh
cd /home/robot/nav
```

重新安装本项目包：

```bash
python -m pip install -e source/nav --no-deps --no-build-isolation
```

## 检查

```bash
python scripts/list_envs.py --keyword Nav
python -m compileall scripts source/nav/nav/tasks/manager_based/nav
```

无人机动力学矩形航线：

```bash
python scripts/test_drone_dynamics.py --headless --strict
```

导航环境零动作和随机动作测试：

```bash
python scripts/zero_agent.py --task Template-Nav-v0 --num_envs 16 --headless
python scripts/random_agent.py --task Template-Nav-v0 --num_envs 16 --headless
```

GUI 模式：去掉命令最后的 `--headless`。

## 训练

训练链路 smoke test：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 16 \
  --max_iterations 5 \
  --save_interval 5 \
  --headless
```

正式训练：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --save_interval 100 \
  --headless
```

训练结果保存在：

```text
runs/ppo_<时间>/
```

查看训练目录和 checkpoint：

```bash
ls -lt runs | head
find runs -name 'checkpoint*.pt' | sort
```

## 曲线

绘制指定训练目录的 iteration 与 completed-episode return 曲线：

```bash
python scripts/plot_return.py \
  runs/ppo_20260819_222322
```

默认使用 50 个 iteration 的移动平均。关闭平滑或修改窗口：

```bash
python scripts/plot_return.py \
  runs/ppo_20260819_222322 \
  --window 1

python scripts/plot_return.py \
  runs/ppo_20260819_222322 \
  --window 100
```

指定 W&B 指标和输出文件：

```bash
python scripts/plot_return.py \
  runs/ppo_20260819_222322 \
  --metric Rollout_Reward/done_return_mean \
  --window 50 \
  --output figure/ppo_20260819_222322_iteration_return.png
```

其中 `run_dir` 必须是包含 `wandb/offline-run-*/run-*.wandb` 的训练目录。也可以只传训练目录名，例如 `ppo_20260819_222322`。

默认图片路径为：

```text
figure/<run_name>_iteration_return.png
```

## 评估

### 快速检查

先用少量环境确认 checkpoint 可以正常加载和运行：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260903_215338/checkpoint_1100.pt \
  --task Template-Nav-v0 \
  --num_envs 16 \
  --episodes_per_env 1 \
  --seed 0 \
  --headless
```

### 标准评估

固定使用 `1024` 个环境、每个环境完成 `1` 个 episode，适合比较不同 checkpoint：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260903_223324/checkpoint_final.pt \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --episodes_per_env 1 \
  --seed 0 \
  --headless
```

默认使用策略分布的确定性均值动作。比较不同模型时，应保持 `task`、`num_envs`、
`episodes_per_env`、`seed` 和动作模式完全一致。

### 多轮评估

每个并行环境运行 `5` 个 episode，并指定输出目录：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260903_215338/checkpoint_1100.pt \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --episodes_per_env 5 \
  --seed 0 \
  --headless \
  --output_dir output/ppo_20260903_215338_checkpoint_1100
```

这会产生 `1024 × 5 = 5120` 条 episode 记录，因此耗时明显高于标准评估。

### 随机动作采样评估

增加 `--stochastic` 后，从策略分布中采样动作：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260903_215338/checkpoint_1100.pt \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --episodes_per_env 1 \
  --seed 0 \
  --stochastic \
  --headless
```

正式报告优先使用默认的确定性评估；随机评估用于观察策略分布的探索行为。

### 参数说明

```text
--checkpoint          必填，待评估的 checkpoint 文件
--task                环境任务名，默认 Template-Nav-v0
--agent               算法配置入口，默认 ppo_cfg_entry_point
--num_envs            并行环境数；快速检查用 16，正式评估用 1024
--episodes_per_env     每个环境需要完成的 episode 数，默认 1
--seed                 场景和随机动作种子，默认 0
--stochastic           使用随机采样动作；不传时使用确定性均值动作
--output_dir           自定义结果目录
--headless             无图形界面运行
--disable_fabric       禁用 Fabric，一般不需要设置
```

### 运行过程

`[INFO]: Completed setting up the environment...` 表示环境已经创建完成，随后才开始
执行评估。脚本需要等待所有并行环境完成指定数量的 episode，才会统一打印汇总并写入
结果文件。单个 episode 最长为 `60` 秒仿真时间，即 `3600` step；如果少数环境直到
超时才结束，终端可能较长时间没有新输出，这不代表仍在创建环境。

不传 `--output_dir` 时，结果默认写入：

```text
output/eval_<时间>/
```

每次评估输出：

```text
episodes.csv    每个完成 episode 的回报、步数、时间、终止原因
summary.json    成功、碰撞、越界、超时和回报汇总
report.html     可直接浏览的评估报告
```

查看最近一次评估结果：

```bash
ls -lt output | head
cat output/eval_<时间>/summary.json
```
