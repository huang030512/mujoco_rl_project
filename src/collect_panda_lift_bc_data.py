import os
import sys
import numpy as np

sys.path.append("src")

from run_panda_lift_handcrafted_policy import (
    make_env,
    build_action,
    update_phase,
)


def obs_to_vec(obs, phase):
    """
    把 robosuite 的 observation 字典转成一维向量。

    MLP 不能直接输入 dict，所以要把关键状态拼接成 numpy array。
    """

    phase_onehot = np.zeros(4, dtype=np.float32)
    phase_onehot[int(phase)] = 1.0

    obs_vec = np.concatenate(
        [
            obs["robot0_eef_pos"],
            obs["robot0_gripper_qpos"],
            obs["cube_pos"],
            obs["gripper_to_cube_pos"],
            phase_onehot,
        ]
    ).astype(np.float32)

    return obs_vec


def collect_data(num_episodes=30, save_path="data/panda_lift_bc_data.npz"):
    os.makedirs("data", exist_ok=True)

    env = make_env(record_video=True)

    obs_list = []
    action_list = []

    success_count = 0

    for ep in range(num_episodes):
        obs = env.reset()
        phase = 0
        total_reward = 0.0
        success_steps = 0

        print(f"\n===== Collect Episode {ep} =====")
        print(f"Action dim: {env.action_dim}")

        for step in range(env.horizon):
            # 1. 使用当前 obs 和 phase 构造神经网络输入
            obs_vec = obs_to_vec(obs, phase)

            # 2. 调用你之前写好的手工策略，得到专家动作
            action, eef_pos, cube_pos, target = build_action(env, obs, phase)

            # 3. 保存 obs-action pair
            obs_list.append(obs_vec)
            action_list.append(action.astype(np.float32))

            # 4. 环境前进一步
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 5. 用执行动作后的新 obs 更新 phase，保证下一轮 obs 和 phase 同步
            eef_pos = obs["robot0_eef_pos"]
            cube_pos = obs["cube_pos"]
            phase = update_phase(phase, step, eef_pos, cube_pos)

            success = bool(info.get("success", False))
            if success:
                success_steps += 1

            if step % 50 == 0:
                print(
                    f"step={step:03d} | phase={phase} | reward={reward:.3f} | "
                    f"success={success} | samples={len(obs_list)}"
                )

            if done:
                break

        ep_success = success_steps > 0
        if ep_success:
            success_count += 1

        print(
            f"Episode {ep} finished | total_reward={total_reward:.3f} | "
            f"success={ep_success}"
        )

    env.close()

    observations = np.array(obs_list, dtype=np.float32)
    actions = np.array(action_list, dtype=np.float32)

    np.savez(
        save_path,
        observations=observations,
        actions=actions,
    )

    print("\n===== BC data collection finished =====")
    print(f"Saved to: {save_path}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Success episodes: {success_count}/{num_episodes}")


if __name__ == "__main__":
    collect_data(num_episodes=30)
