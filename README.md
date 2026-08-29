# Isaac Lab 无人机导航项目

## 项目简介

本项目是一个基于 Isaac Lab `ManagerBasedRLEnv` 构建的无人机导航强化学习项目。项目代码独立于 Isaac Lab 主仓库，包含无人机资产、速度控制器、导航任务 MDP、PPO 算法以及训练和评估脚本。

当前注册的任务为：

```text
Template-Nav-v0
```

主要目录：

```text
source/nav/nav/assets/                         机器人和动态障碍物资产
source/nav/nav/controllers/                    无人机控制器
source/nav/nav/tasks/manager_based/nav/        导航环境和 MDP 配置
source/nav/nav/tasks/manager_based/nav/agents/ PPO 算法及其配置
scripts/                                       训练、评估和环境测试脚本
```

## 环境安装

1. 按照 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。推荐使用 Conda 或 uv 环境，以便直接从终端运行 Python 脚本。

2. 将本项目放在 Isaac Lab 主仓库之外，例如：

   ```text
   /home/robot/IsaacLab
   /home/robot/nav
   ```

3. 使用安装了 Isaac Lab 的 Python 解释器，以可编辑模式安装本项目：

   ```bash
   python -m pip install -e source/nav
   ```

   如果 Isaac Lab 没有安装到当前 Python 环境，请通过 `isaaclab.sh` 执行：

   ```bash
   /path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/nav
   ```

## 验证安装

列出本项目注册的任务：

```bash
python scripts/list_envs.py
```

如果当前 Python 环境无法直接导入 Isaac Lab，可以使用：

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

## 运行环境

使用零动作检查环境能否正常创建和运行：

```bash
python scripts/zero_agent.py --task Template-Nav-v0
```

使用随机动作检查观测、动作和 reset 流程：

```bash
python scripts/random_agent.py --task Template-Nav-v0
```

测试无人机动力学和控制器：

```bash
python scripts/test_drone_dynamics.py
```

## 训练

启动 PPO 训练：

```bash
python scripts/train.py --task Template-Nav-v0 --algo ppo
```

常用参数：

```text
--num_envs        并行环境数量
--seed            随机种子
--max_iterations  最大训练迭代次数
--save_interval   模型保存间隔
--log_dir         训练日志目录
--headless        无界面运行
```

示例：

```bash
python scripts/train.py \
    --task Template-Nav-v0 \
    --algo ppo \
    --num_envs 1024 \
    --max_iterations 2000 \
    --headless
```

训练结果默认保存在项目根目录的 `runs/` 中。Weights & Biases 当前使用离线模式，训练完成后可执行日志中给出的 `wandb sync` 命令上传数据。

## 评估

使用训练生成的模型进行评估：

```bash
python scripts/eval.py --task Template-Nav-v0 --checkpoint <模型路径>
```

具体参数可通过以下命令查看：

```bash
python scripts/eval.py --help
```

## VS Code 配置（可选）

1. 按 `Ctrl+Shift+P` 打开命令面板。
2. 选择 `Tasks: Run Task`。
3. 运行 `setup_python_env`。
4. 根据提示输入 Isaac Sim 的绝对路径。

任务执行成功后，`.vscode` 目录中会生成 `.python.env`。该文件包含 Isaac Sim、Omniverse 和 Isaac Lab 扩展的 Python 路径，可用于代码补全和模块索引。

## 作为 Omniverse 扩展加载（可选）

项目提供了示例扩展入口：

```text
source/nav/nav/ui_extension_example.py
```

加载步骤：

1. 在 Omniverse 中打开 `Window` -> `Extensions`。
2. 打开扩展管理器设置。
3. 将本项目 `source` 目录的绝对路径加入 `Extension Search Paths`。
4. 确保 Isaac Lab 的 `source` 目录也位于搜索路径中。
5. 刷新扩展列表。
6. 在 `Third Party` 分类中找到并启用本项目扩展。

## 代码格式化

安装 `pre-commit`：

```bash
pip install pre-commit
```

检查并格式化所有文件：

```bash
pre-commit run --all-files
```

## 常见问题

### Pylance 无法索引扩展

如果 Pylance 无法找到 Isaac Lab 或项目模块，请在 `.vscode/settings.json` 的 `python.analysis.extraPaths` 中加入项目和 Isaac Lab 的 Python 路径：

```json
{
    "python.analysis.extraPaths": [
        "/path/to/nav/source/nav",
        "/path/to/IsaacLab/source/isaaclab",
        "/path/to/IsaacLab/source/isaaclab_tasks"
    ]
}
```

### Pylance 占用内存过高或崩溃

Isaac Sim 和 Omniverse 包含大量扩展。若 Pylance 因索引内容过多而占用大量内存，可从 `python.analysis.extraPaths` 中移除当前项目不使用的包，例如动画、Kit UI、Graph UI 和服务类扩展。

### 无法导入 Isaac Lab 模块

如果出现 `ModuleNotFoundError: No module named 'isaaclab'` 或类似错误，请确认：

- 当前 Python 解释器属于 Isaac Lab 安装环境；
- 本项目已通过 `pip install -e source/nav` 安装；
- 或者使用 `/path/to/IsaacLab/isaaclab.sh -p <script>` 启动脚本。
