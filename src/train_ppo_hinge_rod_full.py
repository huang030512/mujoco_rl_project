# src/train_ppo_hinge_rod_full.py
import os
import numpy as np
import matplotlib.pyplot as plt
from .hinge_rod_env import HingeRodEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# -----------------------------
# 自定义回调：记录 reward 与关节角
# -----------------------------
class RecordCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_thetas = []

    def _on_step(self) -> bool:
        # 每个 step 记录 theta
        theta = self.training_env.get_attr('data')[0].qpos[0]
        self.episode_thetas.append(theta)
        return True

    def _on_rollout_end(self):
        # 每个 rollout 结束记录 episode reward
        ep_reward = sum(self.locals['rewards'])
        self.episode_rewards.append(ep_reward)

# -----------------------------
# 创建环境
# -----------------------------
env = HingeRodEnv(render_mode=None)

# -----------------------------
# 模型保存目录
# -----------------------------
log_dir = "models/ppo_hinge_rod"
os.makedirs(log_dir, exist_ok=True)

# -----------------------------
# 初始化 PPO
# -----------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=64,
    learning_rate=3e-4,
    n_steps=256
)

# -----------------------------
# 回调记录训练数据
# -----------------------------
record_cb = RecordCallback()

# -----------------------------
# 训练
# -----------------------------
total_timesteps = 100000  # 可先小步数测试
model.learn(total_timesteps=total_timesteps, callback=record_cb)

# 保存模型
model_path = os.path.join(log_dir, "final_model")
model.save(model_path)
print(f"Model saved to {model_path}")

# -----------------------------
# 绘制角度曲线和 reward 曲线
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(record_cb.episode_thetas)
plt.axhline(y=env.target_angle, color='r', linestyle='--', label='target')
plt.xlabel("Step")
plt.ylabel("Theta (rad)")
plt.title("Joint Angle Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(record_cb.episode_rewards)
plt.xlabel("Rollout")
plt.ylabel("Episode Reward")
plt.title("Episode Reward Curve")

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "training_curves.png"))
plt.show()

env.close()