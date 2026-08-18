# 常用命令

所有命令默认在项目根目录执行：

```bash
cd /home/robot/nav
```

## 1. 启动 Isaac Lab 环境

每次新开终端先执行：

```bash
conda activate env_isaaclab
source /home/robot/IsaacLab/_isaac_sim/setup_conda_env.sh
cd /home/robot/nav
```

如果需要重新安装本项目为 editable 包：

```bash
python -m pip install -e /home/robot/nav/source/nav --no-deps --no-build-isolation
```

## 2. 基础检查

查看 git 状态：

```bash
git status --short --branch
```

检查 Python 语法：

```bash
python -m compileall scripts/train.py source/nav/nav/tasks/manager_based/nav
```

检查已注册任务：

```bash
python scripts/list_envs.py --keyword Nav
```

## 3. 无人机动力学回归测试

矩形航线测试：向前 10m、向左 5m、再闭合回原点。

```bash
python scripts/test_drone_dynamics.py --headless --strict
```

GUI 模式：

```bash
python scripts/test_drone_dynamics.py --strict
```

## 4. 环境 smoke test

零动作测试：

```bash
python scripts/zero_agent.py --task Template-Nav-v0 --num_envs 16 --headless
```

随机动作测试：

```bash
python scripts/random_agent.py --task Template-Nav-v0 --num_envs 16 --headless
```

GUI 模式去掉 `--headless`：

```bash
python scripts/zero_agent.py --task Template-Nav-v0 --num_envs 16
```

## 5. PPO 训练 smoke test

小规模快速检查：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 16 \
  --max_iterations 5 \
  --save_interval 5 \
  --headless
```

中等规模测试：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 128 \
  --max_iterations 100 \
  --save_interval 20 \
  --headless
```

正式共享地图并行训练起步：

```bash
python scripts/train.py \
  --task Template-Nav-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --save_interval 100 \
  --headless
```

## 6. 日志和 checkpoint

训练日志默认写入：

```text
runs/ppo_*
```

查看最新运行目录：

```bash
ls -lt runs | head
```

查看 checkpoint：

```bash
find runs -name "checkpoint*.pt" | sort
```

当前训练脚本使用 wandb offline 模式，文件保存在对应运行目录下。

## 7. 绘制 iteration-return 曲线

传入 `runs` 下的训练目录，默认读取 `Rollout_Reward/done_return_mean`，绘制原始曲线和 50 iteration 滑动平均曲线：

```bash
python scripts/plot_return.py runs/ppo_20260818_204907
```

也可以只写 run 目录名称：

```bash
python scripts/plot_return.py ppo_20260818_204907
```

修改滑动平均窗口：

```bash
python scripts/plot_return.py ppo_20260818_204907 --window 100
```

图片默认保存为项目根目录下、与 `runs` 同级的 `figure/<run_name>_iteration_return.png`。指定输出位置：

```bash
python scripts/plot_return.py ppo_20260818_204907 --output runs/return_20260818.png
```
