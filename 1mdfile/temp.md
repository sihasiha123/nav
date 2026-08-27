# manager


# InteractiveScene

InteractiveScene是 Isaac Lab 对“仿真场景中所有实体”的统一容器，不是 RL Manager

# 数据流转

环境配置 cfg
↓
InteractiveScene 创建场景实体
↓
各 Manager 注册 MDP term
↓
env.reset()
↓
返回 observation
↓
Policy 输出 action
↓
env.step(action)
↓
返回 observation、reward、done、extras

## 第一次reset
