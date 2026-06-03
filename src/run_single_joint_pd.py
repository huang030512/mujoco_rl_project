import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mujoco.viewer

# =========================
# 1. 加载模型并创建仿真数据
# =========================
model = mujoco.MjModel.from_xml_path("models/hinge_rod.xml")
data = mujoco.MjData(model)

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

# 通过关节名称找到 joint 的编号
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_hinge")

# qpos 与 qvel 中该关节对应的地址
qpos_adr = model.jnt_qposadr[joint_id]
qvel_adr = model.jnt_dofadr[joint_id]

# 当前模型只有一个 actuator，因此控制输入索引为 0
actuator_id = 0

print("joint_id =", joint_id)
print("qpos address =", qpos_adr)
print("qvel address =", qvel_adr)
print("actuator_id =", actuator_id)


# =========================
# 2. 设置 PD 控制参数
# =========================
target_angle_deg = 45.0
target_angle = np.deg2rad(target_angle_deg)

Kp = 20.0
Kd = 6.0

torque_limit = 5.0

sim_time = 5.0
dt = model.opt.timestep
steps = int(sim_time / dt)


# =========================
# 3. 创建数据记录列表
# =========================
times = []
actual_angles = []
target_angles = []
angular_velocities = []
torques = []


# =========================
# 4. PD 闭环控制仿真
# =========================
for _ in range(steps):
    # 读取当前关节状态
    q = data.qpos[qpos_adr]
    qd = data.qvel[qvel_adr]

    # 计算位置误差
    error = target_angle - q

    # PD 控制律：比例项负责拉向目标，微分项负责抑制运动速度
    torque = Kp * error - Kd * qd

    # 与 XML 中 ctrlrange="-5 5" 对应，显式限制控制力矩
    torque = np.clip(torque, -torque_limit, torque_limit)

    # 写入执行器控制输入
    data.ctrl[actuator_id] = torque

    # 推进一步仿真
    mujoco.mj_step(model, data)
    #viewer = mujoco.viewer.launch_passive(model, data)
    # 记录步进后的真实结果
    times.append(data.time)
    actual_angles.append(np.rad2deg(data.qpos[qpos_adr]))
    target_angles.append(target_angle_deg)
    angular_velocities.append(np.rad2deg(data.qvel[qvel_adr]))
    torques.append(torque)


# =========================
# 5. 打印最终结果
# =========================


final_angle = actual_angles[-1]
final_error = target_angle_deg - final_angle
final_velocity = angular_velocities[-1]

print(f"Target angle: {target_angle_deg:.2f} deg")
print(f"Final angle: {final_angle:.2f} deg")
print(f"Final error: {final_error:.2f} deg")
print(f"Final angular velocity: {final_velocity:.2f} deg/s")


# =========================
# 6. 绘制角度跟踪曲线
# =========================
plt.figure(figsize=(8, 5))
plt.plot(times, actual_angles, label="Actual angle")
plt.plot(times, target_angles, linestyle="--", label="Target angle")
plt.xlabel("Time / s")
plt.ylabel("Joint angle / degree")
plt.title(f"Single Joint PD Control: Kp={Kp}, Kd={Kd}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("assets/week02_single_joint_pd_Kp20_Kd6.png", dpi=300)
plt.show()