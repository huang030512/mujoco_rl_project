import robosuite as suite
import numpy as np

# 1. 创建 Panda Lift 环境
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=False,
    reward_shaping=True,
)

# 2. 重置环境，得到初始状态
obs = env.reset()

# 3. 打印 observation 中包含哪些状态，以及每项状态的维度
print("========== Observation Keys and Shapes ==========")
for key, value in obs.items():
    print(f"{key:30s} shape={value.shape}")

# 4. 只打印当前最重要的几个状态
print("\n========== Important Observations ==========")
print("关节角 robot0_joint_pos:", obs["robot0_joint_pos"])
print("末端位置 robot0_eef_pos:", obs["robot0_eef_pos"])
print("夹爪位置 robot0_gripper_qpos:", obs["robot0_gripper_qpos"])
print("方块位置 cube_pos:", obs["cube_pos"])
print("夹爪到方块的相对位置 gripper_to_cube_pos:", obs["gripper_to_cube_pos"])

# 5. 查看动作范围
low, high = env.action_spec
print("\n========== Action Spec ==========")
print("action low:", low)
print("action high:", high)
print("action dimension:", low.shape[0])

# 6. 执行一次随机动作
action = np.random.uniform(low, high)
obs, reward, done, info = env.step(action)

print("\n========== One Random Step ==========")
print("sampled action:", action)
print("reward:", reward)
print("done:", done)
print("info:", info)

# 7. 关闭环境
env.close()