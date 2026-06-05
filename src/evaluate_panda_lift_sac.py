import os
import numpy as np

from stable_baselines3 import SAC

from panda_lift_rl_env import PandaLiftLowDimEnv


def evaluate(model_path, num_episodes=5, render=False):
    env = PandaLiftLowDimEnv(
        render=render,
        horizon=200,
        control_freq=20,
        reward_shaping=True,
    )

    model = SAC.load(model_path)

    episode_rewards = []
    episode_successes = []

    for ep in range(num_episodes):
        obs, info = env.reset()

        total_reward = 0.0
        success = False

        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if info.get("success", False):
                success = True

            if render:
                env.render()

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_successes.append(success)

        print(
            f"Episode {ep} | "
            f"total_reward={total_reward:.3f} | "
            f"success={success}"
        )

    print("\n===== Evaluation Summary =====")
    print(f"Average reward: {np.mean(episode_rewards):.3f}")
    print(f"Success rate: {np.mean(episode_successes):.2f}")

    env.close()


def main():
    model_path = "models/sac_panda_lift_minimal/final_model.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    evaluate(
        model_path=model_path,
        num_episodes=5,
        render=False,
    )


if __name__ == "__main__":
    main()
