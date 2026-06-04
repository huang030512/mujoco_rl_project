import numpy as np
from stable_baselines3 import PPO

from .hinge_rod_env import HingeRodEnv


env = HingeRodEnv(render_mode="human")
model = PPO.load("models/ppo_hinge_rod/final_model")

num_episodes = 10

for episode in range(num_episodes):
    obs, info = env.reset(seed=episode)

    min_error_deg = float("inf")
    success_steps = 0
    total_reward = 0.0

    for step in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        error_deg = abs(np.rad2deg(info["angle_error"]))
        min_error_deg = min(min_error_deg, error_deg)
        total_reward += reward

        if info["is_success"]:
            success_steps += 1

        if terminated or truncated:
            break

    final_theta_deg = np.rad2deg(info["theta"])
    final_error_deg = np.rad2deg(info["angle_error"])
    final_velocity_deg = np.rad2deg(info["theta_dot"])

    print(
        f"episode={episode + 1:2d}, "
        f"reward={total_reward:8.2f}, "
        f"final_theta={final_theta_deg:7.2f} deg, "
        f"final_error={final_error_deg:7.2f} deg, "
        f"final_velocity={final_velocity_deg:8.2f} deg/s, "
        f"min_error={min_error_deg:6.2f} deg, "
        f"success_steps={success_steps:3d}"
    )

env.close()