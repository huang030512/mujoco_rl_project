from panda_lift_rl_env import PandaLiftLowDimEnv


def main():
    env = PandaLiftLowDimEnv(
        render=False,
        horizon=50,
        control_freq=20,
        reward_shaping=True,
    )

    obs, info = env.reset()

    print("===== Reset Test =====")
    print("Observation shape:", obs.shape)
    print("Initial info:", info)

    print("\n===== Space Test =====")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    print("\n===== Step Test =====")
    for step in range(10):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step:03d} | "
            f"obs_shape={obs.shape} | "
            f"reward={reward:.3f} | "
            f"terminated={terminated} | "
            f"truncated={truncated} | "
            f"success={info.get('success', False)}"
        )

        if terminated or truncated:
            print("Episode ended early.")
            break

    env.close()


if __name__ == "__main__":
    main()
