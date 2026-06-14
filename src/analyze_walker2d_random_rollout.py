import gymnasium as gym


def main():
    max_steps = 300

    env = gym.make("Walker2d-v5")

    # Observation: reset returns the initial state observed by the agent.
    observation, info = env.reset(seed=0)

    episode_total_reward = 0.0
    episode_length = 0
    final_x_position = info.get("x_position", 0.0)
    final_terminated = False
    final_truncated = False

    print(
        "step | total_reward | x_velocity | reward_forward | "
        "reward_ctrl | reward_survive | terminated | truncated"
    )

    for step_idx in range(max_steps):
        # Action: random control input sampled from the Walker2d action space.
        action = env.action_space.sample()

        # Reward and termination: step returns the scalar reward plus stop flags.
        observation, reward, terminated, truncated, info = env.step(action)

        episode_total_reward += reward
        episode_length = step_idx + 1
        final_x_position = info.get("x_position", final_x_position)
        final_terminated = terminated
        final_truncated = truncated

        x_velocity = info.get("x_velocity", 0.0)
        reward_forward = info.get("reward_forward", 0.0)
        reward_ctrl = info.get("reward_ctrl", 0.0)
        reward_survive = info.get("reward_survive", 0.0)

        print(
            f"{step_idx:4d} | "
            f"{reward:12.6f} | "
            f"{x_velocity:10.6f} | "
            f"{reward_forward:14.6f} | "
            f"{reward_ctrl:11.6f} | "
            f"{reward_survive:14.6f} | "
            f"{terminated!s:10s} | "
            f"{truncated!s:9s}"
        )

        if terminated or truncated:
            break

    env.close()

    print("\nEpisode summary")
    print(f"episode total reward: {episode_total_reward:.6f}")
    print(f"episode length: {episode_length}")
    print(f"final x_position: {final_x_position:.6f}")
    print(f"terminated: {final_terminated}")
    print(f"truncated: {final_truncated}")


if __name__ == "__main__":
    main()
