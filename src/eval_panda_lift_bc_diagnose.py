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
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


def obs_to_vec(obs, phase):
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


def diagnose_bc(
    model_path="models/bc/panda_lift_bc_policy.pt",
    num_episodes=3,
):
    checkpoint = torch.load(model_path, map_location="cpu")

    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]

    print(f"Loaded BC policy from: {model_path}")
    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")
    print(f"best_val_loss: {checkpoint.get('best_val_loss', None)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = MLPPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = make_env(record_video=False)

    success_count = 0

    for ep in range(num_episodes):
        obs = env.reset()
        phase = 0
        total_reward = 0.0
        success_steps = 0

        init_cube_z = obs["cube_pos"][2]
        max_cube_z = init_cube_z
        final_cube_z = init_cube_z
        min_eef_cube_dist = 999.0

        print(f"\n===== BC Diagnose Episode {ep} =====")
        print(f"init_cube_z={init_cube_z:.3f}")

        for step in range(env.horizon):
            obs_vec = obs_to_vec(obs, phase)

            obs_tensor = torch.tensor(
                obs_vec,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, done, info = env.step(action)
            total_reward += reward

            eef_pos = obs["robot0_eef_pos"]
            cube_pos = obs["cube_pos"]

            cube_z = cube_pos[2]
            eef_z = eef_pos[2]
            eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)

            max_cube_z = max(max_cube_z, cube_z)
            final_cube_z = cube_z
            min_eef_cube_dist = min(min_eef_cube_dist, eef_cube_dist)

            phase = update_phase(phase, step, eef_pos, cube_pos)

            success_from_info = bool(info.get("success", False))

            try:
                success_from_env = bool(env._check_success())
            except AttributeError:
                success_from_env = False

            success = success_from_info or success_from_env
            if success:
                success_steps += 1

            if step % 25 == 0:
                print(
                    f"step={step:03d} | phase={phase} | "
                    f"reward={reward:.3f} | success={success} | "
                    f"eef_z={eef_z:.3f} | cube_z={cube_z:.3f} | "
                    f"lift={cube_z - init_cube_z:.3f} | "
                    f"dist={eef_cube_dist:.3f} | "
                    f"az={action[2]:.3f} | grip={action[-1]:.3f}"
                )

            if done:
                break

        ep_success = success_steps > 0
        if ep_success:
            success_count += 1

        print(
            f"BC Diagnose Episode {ep} finished | "
            f"total_reward={total_reward:.3f} | "
            f"success={ep_success} | "
            f"max_lift={max_cube_z - init_cube_z:.3f} | "
            f"final_lift={final_cube_z - init_cube_z:.3f} | "
            f"min_eef_cube_dist={min_eef_cube_dist:.3f}"
        )

    env.close()

    print("\n===== BC Diagnose finished =====")
    print(f"Success episodes: {success_count}/{num_episodes}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose Panda Lift BC policy.")
    parser.add_argument(
        "--model",
        default="models/bc/panda_lift_bc_policy.pt",
        help="Path to the BC policy checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to evaluate.",
    )
    args = parser.parse_args()

    print(f"Evaluation model path: {args.model}")
    print(f"Evaluation episodes: {args.episodes}")

    diagnose_bc(model_path=args.model, num_episodes=args.episodes)
