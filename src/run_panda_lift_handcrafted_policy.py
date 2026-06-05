import os
import numpy as np
import imageio.v2 as imageio
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def get_obs_value(obs, key_candidates):
    """兼容读取不同版本 observation 字典中的末端和方块位置"""
    for key in key_candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"Cannot find any of keys {key_candidates} in obs. Available keys: {list(obs.keys())}")


def make_env(record_video=False):
    """创建 Panda Lift 环境，使用 OSC_POSITION 控制器"""
    controller_config = load_composite_controller_config(controller="BASIC")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=not record_video,
        has_offscreen_renderer=record_video,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=300,
    )
    return env


def build_action(env, obs, phase, gripper_close_sign=1.0):
    """
    根据当前阶段生成手工策略动作
    phase 0: 移动到方块上方
    phase 1: 下降到抓取位置
    phase 2: 闭合夹爪
    phase 3: 抬起方块
    """
    action_dim = env.action_dim
    action = np.zeros(action_dim, dtype=np.float32)

    eef_pos = get_obs_value(obs, ["robot0_eef_pos"])
    cube_pos = get_obs_value(obs, ["cube_pos"])

    # 目标点
    above_cube = cube_pos.copy(); above_cube[2] += 0.12
    grasp_pos = cube_pos.copy(); grasp_pos[2] += 0.015
    lift_pos = cube_pos.copy(); lift_pos[2] += 0.25

    # 不同阶段动作
    if phase == 0:
        target = above_cube
        gripper = -gripper_close_sign
        pos_gain = 6.0

    elif phase == 1:
        target = grasp_pos
        gripper = -gripper_close_sign
        pos_gain = 5.0

    elif phase == 2:
        target = grasp_pos
        gripper = gripper_close_sign
        pos_gain = 2.0

    else:  # phase 3
        target = lift_pos
        gripper = gripper_close_sign
        pos_gain = 5.0

    # 末端增量动作
    delta = target - eef_pos
    xyz_action = np.clip(pos_gain * delta, -1.0, 1.0)

    action[:3] = xyz_action
    action[-1] = gripper

    return action, eef_pos, cube_pos, target


def update_phase(phase, step, eef_pos, cube_pos):
    """根据末端和方块位置切换阶段"""
    above_cube = cube_pos.copy(); above_cube[2] += 0.12
    grasp_pos = cube_pos.copy(); grasp_pos[2] += 0.001

    xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
    z_dist_above = abs(eef_pos[2] - above_cube[2])
    z_dist_grasp = abs(eef_pos[2] - grasp_pos[2])

    if phase == 0:
        if xy_dist < 0.025 and z_dist_above < 0.035:
            return 1
    elif phase == 1:
        if xy_dist < 0.025 and z_dist_grasp < 0.025:
            return 2
    elif phase == 2:
        if step > 160:
            return 3
    return phase


def run_episode(env, episode_id, gripper_close_sign=1.0, save_video=False, video_dir="assets"):
    obs = env.reset()
    phase = 0
    total_reward = 0.0
    success_steps = 0
    frames = []

    print(f"\n===== Episode {episode_id} =====")
    print(f"Action dim: {env.action_dim}")
    print(f"Observation keys: {list(obs.keys())}")

    for step in range(env.horizon):
        action, eef_pos, cube_pos, target = build_action(env, obs, phase, gripper_close_sign)
        phase = update_phase(phase, step, eef_pos, cube_pos)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        success = bool(info.get("success", False))
        if success:
            success_steps += 1

        if save_video:
            frame = env.sim.render(camera_name="frontview", width=640, height=480, depth=False)
            frames.append(frame[::-1])

        if step % 20 == 0:
            print(
                f"step={step:03d} | phase={phase} | reward={reward:.3f} | "
                f"success={success} | eef={eef_pos.round(3)} | cube={cube_pos.round(3)}"
            )

        if done:
            break

    print(f"Episode {episode_id} total_reward = {total_reward:.3f}")
    print(f"Episode {episode_id} success_steps = {success_steps}")

    if save_video and len(frames) > 0:
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"week03_panda_lift_handcrafted_ep{episode_id}.mp4")
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Saved video to {video_path}")

    return {
        "episode": episode_id,
        "total_reward": total_reward,
        "success_steps": success_steps,
    }


def main():
    env = make_env(record_video=True)
    gripper_close_sign = 1.0  # 如果夹爪方向反了，改成 -1.0
    results = []

    for ep in range(3):
        result = run_episode(env, ep, gripper_close_sign, save_video=(ep==0))
        results.append(result)

    env.close()

    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_success = np.mean([r["success_steps"] for r in results])

    print("\n===== Summary =====")
    print(f"Average total reward: {avg_reward:.3f}")
    print(f"Average success steps: {avg_success:.3f}")

    if avg_success > 0:
        print("Conclusion: handcrafted policy can reach task success sometimes. RL training is feasible.")
    else:
        print("Conclusion: handcrafted policy has not reached success yet. Need debug grasping before RL training.")


if __name__ == "__main__":
    main()