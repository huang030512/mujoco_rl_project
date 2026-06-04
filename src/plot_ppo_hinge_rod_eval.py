import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .hinge_rod_env import HingeRodEnv


env = HingeRodEnv(render_mode=None)
model = PPO.load("models/ppo_hinge_rod/final_model")

obs, info = env.reset(seed=0)

times = []
theta_values = []
target_values = []
torque_values = []
error_values = []

control_dt = env.model.opt.timestep * env.frame_skip

for step in range(env.max_episode_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    times.append((step + 1) * control_dt)
    theta_values.append(np.rad2deg(info["theta"]))
    target_values.append(np.rad2deg(env.target_angle))
    torque_values.append(info["torque"])
    error_values.append(np.rad2deg(info["angle_error"]))

    if terminated or truncated:
        break

os.makedirs("assets", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(times, theta_values, label="PPO controlled angle")
plt.plot(times, target_values, linestyle="--", label="Target angle")
plt.xlabel("Time (s)")
plt.ylabel("Joint angle (deg)")
plt.title("PPO Control Result of Single-Joint Hinge Rod")
plt.legend()
plt.tight_layout()
plt.savefig("assets/week02_ppo_hinge_rod_angle_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(times, torque_values)
plt.xlabel("Time (s)")
plt.ylabel("Torque")
plt.title("PPO Control Torque of Single-Joint Hinge Rod")
plt.tight_layout()
plt.savefig("assets/week02_ppo_hinge_rod_torque_curve.png", dpi=300)
plt.close()

print("Evaluation finished.")
print(f"Final angle: {theta_values[-1]:.2f} deg")
print(f"Final error: {error_values[-1]:.2f} deg")
print(f"Saved: assets/week02_ppo_hinge_rod_angle_curve.png")
print(f"Saved: assets/week02_ppo_hinge_rod_torque_curve.png")

env.close()
