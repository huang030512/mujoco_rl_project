import numpy as np
import robosuite as suite


def main():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    low, high = env.action_spec

    num_episodes = 5
    max_steps = 200

    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    print("========== Panda Lift Random Baseline ==========")
    print("num_episodes:", num_episodes)
    print("max_steps per episode:", max_steps)
    print("action low:", low)
    print("action high:", high)
    print("action dimension:", low.shape[0])

    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        success = False
        final_done = False

        for t in range(max_steps):
            action = np.random.uniform(low, high)

            obs, reward, done, info = env.step(action)

            total_reward += reward
            final_done = done

            if info.get("success", False):
                success = True

            if done:
                break

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_lengths.append(t + 1)

        print(
            f"Episode {ep + 1}: "
            f"length={t + 1}, "
            f"total_reward={total_reward:.3f}, "
            f"done={final_done}, "
            f"success={success}"
        )

    env.close()

    print("\n========== Summary ==========")
    print("average reward:", np.mean(episode_rewards))
    print("success rate:", np.mean(episode_successes))
    print("average episode length:", np.mean(episode_lengths))

    print("\nExisting demo video:")
    print("assets/week03_panda_lift_random_demo.mp4")


if __name__ == "__main__":
    main()