# src/read_joint.py
# =========================
# 任务：读取单关节状态并施加控制输入，绘制 qpos/qvel 曲线
# 分类说明：
#   ✅ 必须理解：核心仿真加载、循环、控制输入
#   ⚠️ AI可生成初稿但需你运行验证：数据记录、绘图
#   📄 可直接复制模板：保存截图、输出 notes
# =========================

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 加载模型
# ✅ 必须理解
model = mujoco.MjModel.from_xml_path("models/mjcf_structure_demo.xml")  # 替换为你的模型路径
data = mujoco.MjData(model)

# =========================
# 2️⃣ 检查执行器数量
# ✅ 必须理解
print("Number of actuators (nu):", model.nu)
print("Initial control values:", data.ctrl)

# =========================
# 3️⃣ 仿真参数设置
# ✅ 必须理解
sim_time = 2.0                  # 仿真总时间 (秒)
dt = model.opt.timestep          # 仿真时间步长
steps = int(sim_time / dt)       # 循环步数

# =========================
# 4️⃣ 创建列表保存数据
# ⚠️ AI可生成初稿但需你运行验证
qpos_list = []
qvel_list = []

# =========================
# 5️⃣ 循环推进仿真并施加控制
# ✅ 必须理解
for i in range(steps):
    data.ctrl[0] = 1           # 控制输入大小，可修改
    mujoco.mj_step(model, data)  # 推进仿真一步
    qpos_list.append(data.qpos[7])
    qvel_list.append(data.qvel[6])
    print(f"Step {i}: qpos={data.qpos[7]:.3f}, qvel={data.qvel[6]:.3f}, ctrl={data.ctrl[0]}")

# =========================
# 6️⃣ 绘制曲线
# ⚠️ AI可生成初稿但需你运行验证
times = np.arange(steps) * dt
plt.figure(figsize=(6,4))
plt.plot(times, qpos_list, label='qpos')
plt.plot(times, qvel_list, label='qvel')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Single Joint Motion')
plt.legend()
plt.grid(True)

# =========================
# 7️⃣ 保存截图
# 📄 可直接复制模板
plt.savefig("assets/week02_joint_motion.png")
plt.show()