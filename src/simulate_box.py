import mujoco
import numpy as np

# 1. 加载模型
model = mujoco.MjModel.from_xml_path("models/simple_box.xml")
data = mujoco.MjData(model)

# 2. 仿真循环参数
sim_time = 2.0        # 模拟 2 秒
dt = model.opt.timestep
steps = int(sim_time / dt)

# 3. 保存方块 z 位置
z_positions = []

for i in range(steps):
    mujoco.mj_step(model, data)        # 推进一步仿真
    z_positions.append(data.qpos[2])   # qpos[2] 是 z 坐标
    if i % 100 == 0:                   # 每 100 步打印一次
        print(
    f"t={data.time:.3f}s, "
    f"x={data.qpos[0]:.3f}m, "
    f"y={data.qpos[1]:.3f}m, "
    f"z={data.qpos[2]:.3f}m, "
    f"vz={data.qvel[2]:.3f}m/s"
)

# 4. 最后可以用 numpy 打印完整数据或保存
z_positions = np.array(z_positions)