import sys
import numpy as np

sys.path.append("src")

from run_panda_lift_handcrafted_policy import make_env


def eval_random(num_episodes=5):
    env = make_env(record_video=False)

    success_count = 0

    for ep in range(num_episodes):
        obs = env.reset()

        total_reward = 0.0
        success_steps = 0

        init_cube_z = obs["cube_pos"][2]
        max_cube_z = init_cube_z
        min_eef_cube_dist = 999.0

        print(f"\n===== Random Eval Episode {ep} =====")
        print(f"Action dim: {env.action_dim}")

        for step in range(env.horizon):
            low, high = env.action_spec
            action = np.random.uniform(low=low, high=high)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            eef_pos = obs["robot0_eef_pos"]
            cube_pos = obs["cube_pos"]

            eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)
            min_eef_cube_dist = min(min_eef_cube_dist, eef_cube_dist)
            max_cube_z = max(max_cube_z, cube_pos[2])

            success_from_info = bool(info.get("success", False))
            try:
                success_from_env = bool(env._check_success())
            except AttributeError:
                success_from_env = False

            success = success_from_info or success_from_env
            if success:
                success_steps += 1

            if step % 50 == 0:
                print(
                    f"step={step:03d} | reward={reward:.3f} | "
                    f"success={success} | cube_z={cube_pos[2]:.3f} | "
                    f"min_dist={min_eef_cube_dist:.3f} | "
                    f"action={np.round(action, 3)}"
                )

            if done:
                break

        ep_success = success_steps > 0
        if ep_success:
            success_count += 1

        lift_height = max_cube_z - init_cube_z

        print(
            f"Random Eval Episode {ep} finished | "
            f"total_reward={total_reward:.3f} | "
            f"success={ep_success} | "
            f"lift_height={lift_height:.3f} | "
            f"min_eef_cube_dist={min_eef_cube_dist:.3f}"
        )

    env.close()

    print("\n===== Random Evaluation finished =====")
    print(f"Success episodes: {success_count}/{num_episodes}")


if __name__ == "__main__":
    eval_random(num_episodes=5)
