import sys

import numpy as np

# 让脚本从项目根目录运行时，也能导入 src/ 里的原始 teacher policy。
sys.path.append("src")

from run_panda_lift_handcrafted_policy import get_obs_value, make_env


# phase 0 的目标高度：先移动到 cube 上方，避免一开始横向撞到 cube。
APPROACH_HEIGHT = 0.12
# phase 1 / phase 2 的抓取高度：末端目标点略高于 cube 中心。
GRASP_HEIGHT = 0.008
# phase 3 的抬升高度：进入 lift 阶段后，从当前末端高度再向上抬这么多。
LIFT_HEIGHT = 0.20

# 进入 phase 2 后，至少等待这些 step，让夹爪有时间闭合并形成接触。
GRASP_WAIT_STEPS = 35
# 如果接触判断一直不满足，也不要永远卡在 phase 2，超过这个步数后进入 lift。
MAX_GRASP_WAIT_STEPS = 80


def proportional_action(eef_pos, target, gain, limit):
    """
    简单比例控制：目标位置 - 当前末端位置，再乘以 gain。

    limit 用来限制动作幅度，避免末端动作太猛。
    """
    # delta 表示“目标点相对于末端当前位置”的方向和距离。
    delta = target - eef_pos
    # gain 决定动作响应强度，limit 决定每个方向最多输出多大动作。
    return np.clip(gain * delta, -limit, limit)


def build_action_v2(
    env,
    obs,
    phase,
    locked_xy=None,
    lift_target_z=None,
    gripper_close_sign=1.0,
):
    """
    根据当前 phase 生成 v2 手工策略动作。

    phase 0: 移动到方块上方，夹爪打开
    phase 1: 缓慢下降到抓取位置，夹爪打开
    phase 2: 在抓取位置闭合夹爪，并等待夹持稳定
    phase 3: 锁定抓取时 xy，保守地向上抬升，并保持夹爪闭合
    """
    action = np.zeros(env.action_dim, dtype=np.float32)

    # robosuite observation 是 dict，这里复用原始脚本里的兼容读取函数。
    eef_pos = get_obs_value(obs, ["robot0_eef_pos"])
    cube_pos = get_obs_value(obs, ["cube_pos"])

    if phase == 0:
        # 先到 cube 正上方，不急着下降。这样更像“对准目标”的阶段。
        target = cube_pos.copy()
        target[2] += APPROACH_HEIGHT

        # approach 阶段可以稍快，但仍然比 v1 更保守，避免过冲。
        xyz_action = proportional_action(
            eef_pos,
            target,
            gain=np.array([4.0, 4.0, 4.0]),
            limit=np.array([0.8, 0.8, 0.8]),
        )
        gripper = -gripper_close_sign

    elif phase == 1:
        # 下降到抓取高度，夹爪仍然保持打开。
        target = cube_pos.copy()
        target[2] += GRASP_HEIGHT

        # 下降阶段限制 z 方向速度，减少把 cube 撞飞或压偏的风险。
        xyz_action = proportional_action(
            eef_pos,
            target,
            gain=np.array([3.0, 3.0, 2.5]),
            limit=np.array([0.5, 0.5, 0.45]),
        )
        gripper = -gripper_close_sign

    elif phase == 2:
        # 闭爪阶段仍然轻微保持在抓取点附近，避免闭爪时末端漂走。
        target = cube_pos.copy()
        target[2] += GRASP_HEIGHT
        if locked_xy is not None:
            # locked_xy 是进入闭爪阶段时记录的末端 xy，用来减少横向追踪扰动。
            target[:2] = locked_xy

        # 闭爪阶段动作幅度更小，重点是等待夹爪稳定闭合。
        xyz_action = proportional_action(
            eef_pos,
            target,
            gain=np.array([1.5, 1.5, 1.0]),
            limit=np.array([0.25, 0.25, 0.2]),
        )
        xyz_action[2] = np.clip(4.0 * (target[2] - eef_pos[2]), -0.15, 0.05)
        gripper = gripper_close_sign

    else:
        # 抬升阶段尽量不追着 cube 的 xy 变化跑，避免横向扰动把方块甩掉。
        hold_xy = locked_xy if locked_xy is not None else eef_pos[:2].copy()
        target_z = lift_target_z if lift_target_z is not None else eef_pos[2] + LIFT_HEIGHT

        # target 主要用于调试理解：xy 锁定，z 使用进入 lift 时确定的固定目标。
        target = eef_pos.copy()
        target[:2] = hold_xy
        target[2] = target_z

        # xy 只允许很小修正，z 只允许向上且最大 0.30，抬升更慢更稳。
        xy_action = np.clip(0.8 * (hold_xy - eef_pos[:2]), -0.10, 0.10)
        z_action = np.clip(1.5 * (target_z - eef_pos[2]), 0.0, 0.30)

        xyz_action = np.array(
            [xy_action[0], xy_action[1], z_action],
            dtype=np.float32,
        )
        gripper = gripper_close_sign

    # Panda Lift 当前 action_dim 是 7：前三维主要控制末端 xyz，最后一维控制夹爪。
    # 中间姿态相关维度保持为 0。
    action[:3] = xyz_action
    action[-1] = gripper

    return action, eef_pos, cube_pos, target


def update_phase_v2(phase, phase_step, eef_pos, cube_pos):
    """
    根据位置误差和当前 phase 内等待步数切换阶段。

    和 v1 不同，phase 2 -> phase 3 不再只看全局 step，而是要求闭爪等待
    一段时间，并且末端已经足够接近方块。
    """
    above_cube = cube_pos.copy()
    above_cube[2] += APPROACH_HEIGHT

    grasp_pos = cube_pos.copy()
    grasp_pos[2] += GRASP_HEIGHT

    xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
    eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)
    z_dist_above = abs(eef_pos[2] - above_cube[2])
    z_dist_grasp = abs(eef_pos[2] - grasp_pos[2])

    if phase == 0:
        # 只有 xy 对齐且高度接近 cube 上方目标，才进入下降阶段。
        if xy_dist < 0.030 and z_dist_above < 0.040:
            return 1

    elif phase == 1:
        # 下降到抓取高度附近后，进入闭爪阶段。
        if xy_dist < 0.025 and z_dist_grasp < 0.025:
            return 2

    elif phase == 2:
        # v1 只看全局 step；v2 改成看“闭爪阶段已经等了多久”。
        waited_enough = phase_step >= GRASP_WAIT_STEPS
        # 距离足够近时，说明末端已经在 cube 附近，可以尝试抬升。
        close_enough = eef_cube_dist < 0.060
        # z 方向也要接近抓取高度，避免 xy 对准但高度还没到位就开始抬升。
        z_ready = z_dist_grasp < 0.012
        # 兜底条件：如果一直判断不够近，也不要永远停在 phase 2。
        waited_too_long = phase_step >= MAX_GRASP_WAIT_STEPS

        if waited_too_long or (waited_enough and close_enough and z_ready):
            return 3

    return phase


def run_episode(env, episode_id, horizon=300, gripper_close_sign=1.0):
    obs = env.reset()
    # phase 表示当前处于 approach / descend / close / lift 哪个阶段。
    phase = 0
    # phase_step 表示进入当前 phase 后已经过了多少 step。
    phase_step = 0

    # locked_xy 会在进入 phase 2 时记录，用于后续闭爪和抬升阶段锁定水平位置。
    locked_xy = None
    # lift_target_z 会在进入 phase 3 时记录，用于抬升阶段使用固定 z 目标。
    lift_target_z = None

    total_reward = 0.0
    success_steps = 0

    # 下面这些是诊断指标，不影响控制，只用于分析 teacher 质量。
    init_cube_z = obs["cube_pos"][2]
    max_cube_z = init_cube_z
    final_cube_z = init_cube_z
    min_eef_cube_dist = float("inf")

    print(f"\n===== V2 Episode {episode_id} =====")
    print(f"Action dim: {env.action_dim}")
    print(f"Observation keys: {list(obs.keys())}")

    for step in range(horizon):
        # 根据当前 phase 和锁定目标生成动作。
        action, _, _, _ = build_action_v2(
            env,
            obs,
            phase,
            locked_xy=locked_xy,
            lift_target_z=lift_target_z,
            gripper_close_sign=gripper_close_sign,
        )

        # 执行动作，得到下一时刻的 observation、reward 和 success 信息。
        obs, reward, done, info = env.step(action)
        total_reward += reward

        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]
        grasp_pos = cube_pos.copy()
        grasp_pos[2] += GRASP_HEIGHT

        xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
        z_dist_grasp = abs(eef_pos[2] - grasp_pos[2])
        eef_cube_dist = np.linalg.norm(eef_pos - cube_pos)

        # 更新 cube 抬升高度和最小末端-cube 距离。
        cube_z = cube_pos[2]
        max_cube_z = max(max_cube_z, cube_z)
        final_cube_z = cube_z
        current_max_lift = max_cube_z - init_cube_z
        min_eef_cube_dist = min(min_eef_cube_dist, eef_cube_dist)

        success_from_info = bool(info.get("success", False))
        try:
            success_from_env = bool(env._check_success())
        except AttributeError:
            success_from_env = False
        success = success_from_info or success_from_env
        if success:
            success_steps += 1

        # 用 step 后的新状态判断是否切换 phase。
        new_phase = update_phase_v2(phase, phase_step, eef_pos, cube_pos)
        if new_phase != phase:
            print(
                f"[phase-change] episode={episode_id} | step={step:03d} | "
                f"old_phase={phase} -> new_phase={new_phase} | "
                f"phase_step={phase_step:03d} | "
                f"xy_dist={xy_dist:.3f} | z_dist_grasp={z_dist_grasp:.3f} | "
                f"eef_cube_dist={eef_cube_dist:.3f} | "
                f"eef_pos={np.round(eef_pos, 3)} | cube_pos={np.round(cube_pos, 3)}"
            )
            phase = new_phase
            phase_step = 0

            if phase == 2:
                # 进入闭爪阶段时记录当前 xy，后续尽量围绕这个位置稳定夹取。
                locked_xy = eef_pos[:2].copy()

            elif phase == 3:
                # 进入抬升阶段时固定一个 z 目标，之后保守上抬。
                if locked_xy is None:
                    locked_xy = eef_pos[:2].copy()
                lift_target_z = eef_pos[2] + LIFT_HEIGHT
        else:
            # 如果 phase 没变，当前 phase 内等待步数 +1。
            phase_step += 1

        if step % 25 == 0:
            print(
                f"step={step:03d} | phase={phase} | phase_step={phase_step:03d} | "
                f"reward={reward:.3f} | success={success} | "
                f"xy_dist={xy_dist:.3f} | z_dist_grasp={z_dist_grasp:.3f} | "
                f"eef_cube_dist={eef_cube_dist:.3f} | "
                f"max_lift={current_max_lift:.3f} | "
                f"action[2]={action[2]:.3f} | gripper_action={action[-1]:.1f}"
            )

        if done:
            break

    # episode 结束后，把最高抬升和最终抬升都相对初始 cube 高度计算。
    max_lift = max_cube_z - init_cube_z
    final_lift = final_cube_z - init_cube_z

    print(
        f"V2 Episode {episode_id} finished | "
        f"total_reward={total_reward:.3f} | "
        f"success_steps={success_steps} | "
        f"max_lift={max_lift:.3f} | "
        f"final_lift={final_lift:.3f} | "
        f"min_eef_cube_dist={min_eef_cube_dist:.3f}"
    )

    return {
        "episode": episode_id,
        "total_reward": total_reward,
        "success_steps": success_steps,
        "max_lift": max_lift,
        "final_lift": final_lift,
        "min_eef_cube_dist": min_eef_cube_dist,
    }


def main():
    num_episodes = 3
    horizon = 300

    # 使用 offscreen renderer，不保存视频文件，也不会写入 assets。
    env = make_env(record_video=True)
    env.horizon = horizon

    # 连续跑多个 episode，观察 v2 teacher 是否比 v1 更稳定。
    results = []
    for episode_id in range(num_episodes):
        result = run_episode(env, episode_id, horizon=horizon)
        results.append(result)

    env.close()

    # 汇总多个 episode 的表现，快速判断 teacher 数据质量。
    success_episodes = sum(result["success_steps"] > 0 for result in results)
    avg_reward = np.mean([result["total_reward"] for result in results])
    avg_max_lift = np.mean([result["max_lift"] for result in results])
    avg_min_dist = np.mean([result["min_eef_cube_dist"] for result in results])

    print("\n===== V2 Summary =====")
    print(f"Success episodes: {success_episodes}/{num_episodes}")
    print(f"Average total reward: {avg_reward:.3f}")
    print(f"Average max_lift: {avg_max_lift:.3f}")
    print(f"Average min_eef_cube_dist: {avg_min_dist:.3f}")


if __name__ == "__main__":
    main()
