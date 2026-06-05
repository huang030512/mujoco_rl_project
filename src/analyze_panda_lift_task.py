import numpy as np
import robosuite as suite


def print_obs(obs):
    print("\n========== Observation Fields ==========")
    for key, value in obs.items():
        arr = np.array(value)
        print(f"{key:30s} shape={arr.shape}, value={arr}")


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

    print("========== Environment Info ==========")
    print("env class:", env.__class__.__name__)
    print("robots:", env.robots)

    print("\n========== Action Spec ==========")
    low, high = env.action_spec
    print("action low:", low)
    print("action high:", high)
    print("action dimension:", low.shape[0])

    obs = env.reset()
    print_obs(obs)

    print("\n========== Step Test ==========")

    zero_action = np.zeros_like(low)
    obs, reward, done, info = env.step(zero_action)

    print("\nAfter zero action:")
    print("reward:", reward)
    print("done:", done)
    print("info:", info)

    random_action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(random_action)

    print("\nAfter random action:")
    print("random action:", random_action)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)

    print("\n========== Reset Randomization Test ==========")
    for i in range(5):
        obs = env.reset()
        print(f"\nReset {i + 1}:")
        if "cube_pos" in obs:
            print("cube_pos:", obs["cube_pos"])
        if "robot0_eef_pos" in obs:
            print("robot0_eef_pos:", obs["robot0_eef_pos"])
        if "cube_to_robot0_eef_pos" in obs:
            print("cube_to_robot0_eef_pos:", obs["cube_to_robot0_eef_pos"])

    env.close()


if __name__ == "__main__":
    main()