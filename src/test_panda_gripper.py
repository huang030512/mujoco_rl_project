import robosuite as suite
import numpy as np

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=False,
    reward_shaping=True
)

obs = env.reset()
print("初始夹爪位置:", obs["robot0_gripper_qpos"])

# 测试夹爪闭合
action = np.zeros(7)
action[6] = 1.0  # 夹爪闭合
for _ in range(20):
    obs, reward, done, info = env.step(action)
    env.render()
print("夹爪闭合后:", obs["robot0_gripper_qpos"])

# 测试夹爪张开
obs = env.reset()
action = np.zeros(7)
action[6] = -1.0  # 夹爪张开
for _ in range(20):
    obs, reward, done, info = env.step(action)
    env.render()
print("夹爪张开后:", obs["robot0_gripper_qpos"])

env.close()