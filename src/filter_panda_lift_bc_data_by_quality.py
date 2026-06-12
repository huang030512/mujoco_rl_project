import argparse
import os

import numpy as np


REQUIRED_KEYS = [
    "observations",
    "actions",
    "episode_ids",
    "total_reward",
    "success",
    "success_steps",
    "max_lift",
    "final_lift",
    "min_eef_cube_dist",
]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Filter Panda Lift BC data by episode-level quality metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/panda_lift_bc_data_with_metrics_10ep.npz",
        help="Input .npz file collected with episode-level metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/panda_lift_bc_data_filtered_success_10ep.npz",
        help="Output .npz file for filtered BC training data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="success",
        choices=["success"],
        help="Filtering mode. success keeps episodes with success=True.",
    )
    return parser.parse_args()


def check_required_keys(data):
    """检查输入文件是否包含筛选所需字段。"""
    missing_keys = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing_keys:
        raise KeyError(f"Input file is missing keys: {missing_keys}")


def get_episode_keep_mask(data, mode):
    """根据 episode 级指标决定保留哪些 episode。"""
    if mode == "success":
        return data["success"].astype(bool)

    raise ValueError(f"Unsupported mode: {mode}")


def filter_bc_data(input_path, output_path, mode):
    """
    读取带 metrics 的 BC 数据，按 episode 质量筛选，再保存新数据文件。

    observations / actions / episode_ids 是样本级数据。
    total_reward / success / success_steps / lift / distance 是 episode 级数据。
    """
    data = np.load(input_path)
    check_required_keys(data)

    observations = data["observations"]
    actions = data["actions"]
    episode_ids = data["episode_ids"]

    total_reward = data["total_reward"]
    success = data["success"]
    success_steps = data["success_steps"]
    max_lift = data["max_lift"]
    final_lift = data["final_lift"]
    min_eef_cube_dist = data["min_eef_cube_dist"]

    original_episode_count = len(success)
    original_sample_count = len(observations)

    episode_keep_mask = get_episode_keep_mask(data, mode)
    kept_episode_ids = np.where(episode_keep_mask)[0].astype(np.int32)

    sample_keep_mask = np.isin(episode_ids, kept_episode_ids)

    filtered_observations = observations[sample_keep_mask]
    filtered_actions = actions[sample_keep_mask]
    filtered_episode_ids = episode_ids[sample_keep_mask]

    filtered_total_reward = total_reward[episode_keep_mask]
    filtered_success = success[episode_keep_mask]
    filtered_success_steps = success_steps[episode_keep_mask]
    filtered_max_lift = max_lift[episode_keep_mask]
    filtered_final_lift = final_lift[episode_keep_mask]
    filtered_min_eef_cube_dist = min_eef_cube_dist[episode_keep_mask]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez(
        output_path,
        observations=filtered_observations,
        actions=filtered_actions,
        episode_ids=filtered_episode_ids,
        total_reward=filtered_total_reward,
        success=filtered_success,
        success_steps=filtered_success_steps,
        max_lift=filtered_max_lift,
        final_lift=filtered_final_lift,
        min_eef_cube_dist=filtered_min_eef_cube_dist,
        kept_episode_ids=kept_episode_ids,
    )

    print("\n===== Panda Lift BC data filtering finished =====")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode: {mode}")
    print(f"Original episodes: {original_episode_count}")
    print(f"Kept episode ids: {kept_episode_ids.tolist()}")
    print(f"Original samples: {original_sample_count}")
    print(f"Filtered samples: {len(filtered_observations)}")

    print("\n===== Kept episode metrics =====")
    for new_index, episode_id in enumerate(kept_episode_ids):
        print(
            f"episode_id={int(episode_id)} | "
            f"success={bool(filtered_success[new_index])} | "
            f"success_steps={int(filtered_success_steps[new_index])} | "
            f"max_lift={float(filtered_max_lift[new_index]):.3f} | "
            f"final_lift={float(filtered_final_lift[new_index]):.3f}"
        )


def main():
    args = parse_args()
    filter_bc_data(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
