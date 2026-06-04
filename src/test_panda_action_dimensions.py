import robosuite as suite
import numpy as np
import time

# 创建 Panda Lift 环境
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=False,
    reward_shaping=True,
)

obs = env.reset()

print("========== Initial State ==========")
print("末端初始位置:", obs["robot0_eef_pos"])
print("夹爪初始位置:", obs["robot0_gripper_qpos"])

# 先只测试前三维：末端 x / y / z 方向运动
test_actions = [
    ("action[0]：预期沿 x 方向移动", 0),
    ("action[1]：预期沿 y 方向移动", 1),
    ("action[2]：预期沿 z 方向移动", 2),
]

for description, action_index in test_actions:
    print("\n===================================")
    print(description)

    # 每测试一个维度前都重置环境
    obs = env.reset()
    start_pos = obs["robot0_eef_pos"].copy()

    print("动作前末端位置:", start_pos)

    # 构造只有一维为正值的动作
    action = np.zeros(7)
    action[action_index] = 0.5

    # 连续执行若干步，使运动变化更加明显
    for _ in range(20):
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.02)

    end_pos = obs["robot0_eef_pos"].copy()

    print("动作后末端位置:", end_pos)
    print("末端位置变化量:", end_pos - start_pos)

env.close()