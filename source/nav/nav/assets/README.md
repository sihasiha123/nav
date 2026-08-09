策略输出
  (vx, vy, vz) 世界系速度指令
        │
        ▼
VelocityController.apply_action()   ← 适配层
  1. 清洗动作，限制为 (num_envs, 3)
  2. 根据速度方向生成期望 yaw（yaw 速率限制）
  3. 拼成命令 [yaw, vx, vy, vz]
  4. 读取无人机当前状态（位置/姿态/线速度/角速度）
        │
        ▼
LVController.compute()              ← 线速度 + 偏航环
  1. 速度误差 → 期望加速度（限幅 max_feedback_accel）
  2. 期望力 = mass × (acc_fb - g)
  3. 期望推力 = 期望力转到机体 z 轴
  4. 姿态环：由期望 yaw 和期望力方向构造期望姿态 → 期望机体角速度
  5. 角速度环：力矩 = I × K × 角速度误差 + 陀螺耦合项
  6. 输出 [总推力, τx, τy, τz]
        │
        ▼
VelocityController 限幅
  总推力 clamp [0, 4×最大单桨推力]
  力矩 clamp ±最大机体力矩
        │
        ▼
robot.permanent_wrench_composer.set_forces_and_torques()
  把 [0, 0, thrust] 力和 [τx, τy, τz] 力矩施加到机身的 "body"
        │
        ▼
PhysX 物理引擎积分
  更新无人机位置、姿态、速度