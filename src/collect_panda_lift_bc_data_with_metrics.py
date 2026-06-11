import argparse
import os
import sys

import numpy as np

# 让脚本从项目根目录运行时，也能导入 src/ 目录里的其它文件。
sys.path.append("src")


def make_bc_observation(obs, phase):
    """
    生成 BC 模型使用的一维 observation。

    这里直接复用原采集脚本里的 obs_to_vec，保证新数据和旧数据的
    observation 字段顺序、维度完全一致，后续仍然可以用原 BC 训练脚本读取。
    """
    from collect_panda_lift_bc_data import obs_to_vec

    return obs_to_vec(obs, phase)


def collect_data_with_metrics(
    num_episodes=10,
    horizon=300,
    output="data/panda_lift_bc_data_with_metrics.npz",
):
    """
    使用 handcrafted policy 采集 BC 数据，并额外记录 episode 级诊断指标。

    保存的数据分成两类：
    1. 样本级数据：observations、actions、episode_ids，用于 BC 训练和追踪样本来源。
    2. episode 级数据：reward、success、lift、距离等，用于分析每条轨迹质量。
    """
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # robosuite 导入较慢，而且可能触发环境相关 warning。
    # 放在函数内部，可以让 --help 这类命令不必初始化 robosuite。
    from run_panda_lift_handcrafted_policy import (
        build_action,
        make_env,
        update_phase,
    )

    # record_video=True 会创建 offscreen renderer，和原始 BC 数据采集脚本保持一致。
    env = make_env(record_video=True)
    # 原 make_env 默认 horizon=300；这里允许命令行覆盖，方便短测试或长轨迹采集。
    env.horizon = horizon

    # 每个 step 保存一条 observation-action 样本。
    obs_list = []
    action_list = []
    episode_id_list = []

    # 每个 episode 保存一条诊断记录。
    episode_total_rewards = []
    episode_successes = []
    episode_success_steps = []
    episode_max_lifts = []
    episode_final_lifts = []
    episode_min_eef_cube_dists = []

    for episode_id in range(num_episodes):
        # 每个 episode 都从环境 reset 开始，phase 也回到第 0 阶段。
        obs = env.reset()
        phase = 0

        total_reward = 0.0
        success_steps = 0

        # 用初始方块高度作为基准，后面计算方块被抬高了多少。
        init_cube_z = obs["cube_pos"][2]
        max_cube_z = init_cube_z
        final_cube_z = init_cube_z
        min_eef_cube_dist = float("inf")

        print(f"\n===== Collect Episode {episode_id} =====")
        print(f"Action dim: {env.action_dim}")

        for step in range(horizon):
            # 先用“当前 obs + 当前 phase”构造神经网络输入。
            obs_vec = make_bc_observation(obs, phase)
            # handcrafted policy 是 teacher，给出这个状态下的专家动作。
            action, _, _, _ = build_action(env, obs, phase)

            # 保存当前状态和专家动作，形成一条 BC 训练样本。
            obs_list.append(obs_vec)
            action_list.append(action.astype(np.float32))
            episode_id_list.append(episode_id)

            # 执行动作后，环境返回下一步 observation 和 reward。
            obs, reward, done, info = env.step(action)
            total_reward += reward

            eef_pos = obs["robot0_eef_pos"]
            cube_pos = obs["cube_pos"]

            # 更新诊断指标：最高抬升高度、最终高度、末端和方块最近距离。
            cube_z = cube_pos[2]
            final_cube_z = cube_z
            max_cube_z = max(max_cube_z, cube_z)
            min_eef_cube_dist = min(
                min_eef_cube_dist,
                float(np.linalg.norm(eef_pos - cube_pos)),
            )

            # phase 必须用 step 后的新 obs 更新，保证下一轮输入和阶段同步。
            phase = update_phase(phase, step, eef_pos, cube_pos)

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
                    f"step={step:03d} | phase={phase} | reward={reward:.3f} | "
                    f"success={success} | samples={len(obs_list)}"
                )

            if done:
                break

        # 只要 episode 中任意 step 成功过，就把整个 episode 记为成功。
        ep_success = success_steps > 0
        max_lift = max_cube_z - init_cube_z
        final_lift = final_cube_z - init_cube_z

        # 保存本条 episode 的诊断结果。
        episode_total_rewards.append(total_reward)
        episode_successes.append(ep_success)
        episode_success_steps.append(success_steps)
        episode_max_lifts.append(max_lift)
        episode_final_lifts.append(final_lift)
        episode_min_eef_cube_dists.append(min_eef_cube_dist)

        print(
            f"Episode {episode_id} finished | "
            f"total_reward={total_reward:.3f} | "
            f"success={ep_success} | "
            f"success_steps={success_steps} | "
            f"max_lift={max_lift:.3f} | "
            f"final_lift={final_lift:.3f} | "
            f"min_eef_cube_dist={min_eef_cube_dist:.3f}"
        )

    env.close()

    # 转成 numpy array 后再保存，方便后续 np.load 直接读取。
    observations = np.array(obs_list, dtype=np.float32)
    actions = np.array(action_list, dtype=np.float32)
    episode_ids = np.array(episode_id_list, dtype=np.int32)

    episode_total_rewards = np.array(episode_total_rewards, dtype=np.float32)
    episode_successes = np.array(episode_successes, dtype=np.bool_)
    episode_success_steps = np.array(episode_success_steps, dtype=np.int32)
    episode_max_lifts = np.array(episode_max_lifts, dtype=np.float32)
    episode_final_lifts = np.array(episode_final_lifts, dtype=np.float32)
    episode_min_eef_cube_dists = np.array(
        episode_min_eef_cube_dists,
        dtype=np.float32,
    )

    # observations / actions 保持原 BC 训练格式；其它字段用于数据质量分析。
    np.savez(
        output,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        total_reward=episode_total_rewards,
        success=episode_successes,
        success_steps=episode_success_steps,
        max_lift=episode_max_lifts,
        final_lift=episode_final_lifts,
        min_eef_cube_dist=episode_min_eef_cube_dists,
    )

    success_episodes = int(np.sum(episode_successes))
    average_total_reward = float(np.mean(episode_total_rewards))
    average_max_lift = float(np.mean(episode_max_lifts))

    # 最后打印整体统计，快速判断采集到的数据质量。
    print("\n===== BC data collection with metrics finished =====")
    print(f"Saved to: {output}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Episode ids shape: {episode_ids.shape}")
    print(f"Success episodes: {success_episodes}/{num_episodes}")
    print(f"Average total reward: {average_total_reward:.3f}")
    print(f"Average max_lift: {average_max_lift:.3f}")


def parse_args():
    """解析命令行参数，方便不改代码就调整采集规模和输出路径。"""
    parser = argparse.ArgumentParser(
        description="Collect Panda Lift BC data with episode-level metrics."
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument(
        "--output",
        type=str,
        default="data/panda_lift_bc_data_with_metrics.npz",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_data_with_metrics(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        output=args.output,
    )
