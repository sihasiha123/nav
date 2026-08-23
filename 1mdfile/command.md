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

先进行小规模评估：

```bash

```

正式评估：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260819_222322/checkpoint_2000.pt \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --episodes_per_env 1 \
  --seed 0 \
  --headless
```

多 episode 评估并指定目录：

```bash
python scripts/eval.py \
  --checkpoint runs/ppo_20260819_222322/checkpoint_2000.pt \
  --num_envs 1024 \
  --episodes_per_env 5 \
  --seed 0 \
  --headless \
  --output_dir output/ppo_20260819_222322_checkpoint_2000
```

不传 `--output_dir` 时，结果写入：

```text
output/eval_<时间>/
```

每次评估输出：

```text
episodes.csv    每个完成 episode 的回报、步数、时间、终止原因
summary.json    成功、碰撞、越界、超时和回报汇总
```

评估默认使用确定性均值动作。需要随机采样时增加：

```text
--stochastic
```
