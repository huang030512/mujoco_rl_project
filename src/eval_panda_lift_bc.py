import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("src")

from run_panda_lift_handcrafted_policy import (
    make_env,
    update_phase,
)


class MLPPolicy(nn.Module):
    """
    和训练脚本里完全一致的 MLP policy。
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # 网络结构必须和 train_panda_lift_bc.py 中训练时的结构完全一致，
        # 否则后面 load_state_dict 会因为参数形状对不上而失败。
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        # 输入 observation batch，输出模型预测的 action batch。
        return self.net(obs)


def obs_to_vec(obs, phase):
    """
    必须和采集数据时的 obs_to_vec 保持一致。
    """

    # 把离散阶段 phase 转成 one-hot，作为 observation 的一部分输入模型。
    phase_onehot = np.zeros(4, dtype=np.float32)
    phase_onehot[int(phase)] = 1.0

    # 将 robosuite 的 observation 字典拼成一维向量。
    # 这里的字段顺序必须和 collect_panda_lift_bc_data.py 采集数据时保持一致。
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


def eval_bc(
    model_path="models/bc/panda_lift_bc_policy.pt",
    num_episodes=5,
):
    # 读取训练脚本保存的 checkpoint。
    # map_location="cpu" 可以保证即使没有 GPU，也能先把模型文件加载进来。
    checkpoint = torch.load(model_path, map_location="cpu")

    # 从 checkpoint 里取出模型输入维度和输出维度，用来重建同样大小的网络。
    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]

    print(f"Loaded BC policy from: {model_path}")
    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")
    print(f"best_val_loss: {checkpoint.get('best_val_loss', None)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 先创建和训练时相同结构的模型，再加载训练好的参数。
    policy = MLPPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    # 切换到评估模式：这里只做推理，不训练模型。
    policy.eval()

    # 创建 Panda Lift 环境，用训练好的 BC policy 进行测试。
    env = make_env(record_video=True)

    success_count = 0

    for ep in range(num_episodes):
        # 每个 episode 从环境初始状态开始，phase 从 0 开始。
        obs = env.reset()
        phase = 0
        total_reward = 0.0
        success_steps = 0

        print(f"\n===== BC Eval Episode {ep} =====")
        print(f"Action dim: {env.action_dim}")
        print(f"Observation keys: {list(obs.keys())}")

        for step in range(env.horizon):
            # 用当前 obs 和 phase 构造模型输入。
            obs_vec = obs_to_vec(obs, phase)

            # 模型期望输入形状是 (batch_size, obs_dim)。
            # 当前只有一条 observation，所以用 unsqueeze(0) 增加 batch 维度。
            obs_tensor = torch.tensor(
                obs_vec,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # 评估阶段不需要梯度，只做一次前向推理得到 action。
            with torch.no_grad():
                # policy 输出形状是 (1, action_dim)，[0] 取出这一条 action。
                action = policy(obs_tensor).cpu().numpy()[0]

            # 把模型预测的动作送进环境，得到下一步状态和奖励信息。
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 用执行动作后的新 obs 更新 phase，保证下一轮 obs 和 phase 同步。
            eef_pos = obs["robot0_eef_pos"]
            cube_pos = obs["cube_pos"]
            phase = update_phase(phase, step, eef_pos, cube_pos)

            # robosuite 会在 info 里给出当前任务是否成功。
            success = bool(info.get("success", False))
            if success:
                success_steps += 1

            # 每 50 步打印一次运行状态，方便观察模型动作和成功情况。
            if step % 50 == 0:
                print(
                    f"step={step:03d} | phase={phase} | reward={reward:.3f} | "
                    f"success={success} | action={np.round(action, 3)}"
                )

            # 如果环境提前结束，就跳出当前 episode。
            if done:
                break

        # 只要这个 episode 里出现过 success=True，就记为成功。
        ep_success = success_steps > 0
        if ep_success:
            success_count += 1

        print(
            f"BC Eval Episode {ep} finished | total_reward={total_reward:.3f} | "
            f"success={ep_success}"
        )

    env.close()

    print("\n===== BC Evaluation finished =====")
    print(f"Success episodes: {success_count}/{num_episodes}")


if __name__ == "__main__":
    eval_bc(num_episodes=5)
