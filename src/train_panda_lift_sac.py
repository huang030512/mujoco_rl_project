import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from panda_lift_rl_env import PandaLiftLowDimEnv

class RewardCallback(BaseCallback):
    """记录每个 episode reward 和 success"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.success_flags = []
        self.current_reward = 0.0
        self.current_success = False

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_reward += float(rewards[0])

        if infos is not None:
            info = infos[0]
            if info.get("success", False):
                self.current_success = True

        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.success_flags.append(self.current_success)
            self.current_reward = 0.0
            self.current_success = False
        return True

def plot_training_curve(rewards, success_flags, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rewards = np.array(rewards, dtype=np.float32)
    success_flags = np.array(success_flags, dtype=np.float32)
    window = min(10, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8,5))
    plt.plot(rewards, label="Episode Reward", alpha=0.4)
    plt.plot(np.arange(window-1, window-1+len(moving_avg)), moving_avg, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Panda Lift Minimal SAC Training Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    log_dir = "models/sac_panda_lift_minimal"
    os.makedirs(log_dir, exist_ok=True)

    env = PandaLiftLowDimEnv(render=False, horizon=200, control_freq=20, reward_shaping=True)
    env = Monitor(env)

    callback = RewardCallback()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=2_000,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="auto",
    )

    total_timesteps = 20_000
    print("Start minimal SAC training...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    reward_curve_path = os.path.join(log_dir, "reward_curve.png")
    plot_training_curve(callback.episode_rewards, callback.success_flags, reward_curve_path)
    print(f"Reward curve saved to: {reward_curve_path}")

    env.close()

if __name__ == "__main__":
    main()
