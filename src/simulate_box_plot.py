import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载模型
model = mujoco.MjModel.from_xml_path("models/simple_box.xml")
data = mujoco.MjData(model)

# 2. 仿真参数
sim_time = 2.0
dt = model.opt.timestep
steps = int(sim_time / dt)

# 3. 保存数据
times = []
z_positions = []
vz_values = []

for i in range(steps):
    mujoco.mj_step(model, data)
    times.append(data.time)
    z_positions.append(data.qpos[2])
    vz_values.append(data.qvel[2])

# 转成 numpy 数组
times = np.array(times)
z_positions = np.array(z_positions)
vz_values = np.array(vz_values)

# 4. 绘制高度曲线
plt.figure(figsize=(6.67, 5))
plt.plot(times, z_positions, label="z position", color='tab:blue')
plt.axhline(0.1, color='tab:orange', linestyle='--', label="floor contact")
plt.axhline(0.5, color='tab:green', linestyle='--', label="initial height")
plt.xlabel("Time [s]")
plt.ylabel("Height z [m]")
plt.title("Free Fall of Box in MuJoCo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()