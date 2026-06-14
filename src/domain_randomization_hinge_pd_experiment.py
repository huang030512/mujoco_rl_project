import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mujoco_rl_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


# 项目路径与输出路径：脚本可以从仓库任意工作目录下运行。
REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "models" / "hinge_rod.xml"
CSV_PATH = REPO_ROOT / "assets" / "week06_domain_randomization_hinge_pd_results.csv"
PNG_PATH = REPO_ROOT / "assets" / "week06_domain_randomization_hinge_pd.png"

# 固定 PD 控制器参数。实验中只改变物理参数，不重新调参。
TARGET_ANGLE_DEG = 45.0
TARGET_ANGLE = np.pi / 4
KP = 20.0
KD = 2.0
TORQUE_MIN = -5.0
TORQUE_MAX = 5.0
INIT_ANGLE_DEG = 0.0
SIM_TIME = 5.0
SUCCESS_TOL_DEG = 5.0

# CSV 的列名。每一行对应一次完整 episode 的统计结果。
FIELDNAMES = [
    "scenario",
    "mass_scale",
    "damping_scale",
    "frictionloss",
    "action_noise_std",
    "init_angle_deg",
    "final_angle_deg",
    "final_error_deg",
    "last_error_deg",
    "max_overshoot_deg",
    "mean_abs_torque",
    "success",
]

# 固定扰动实验：一次只改变一个主要因素，方便观察控制器鲁棒性。
FIXED_SCENARIOS = [
    {
        "scenario": "nominal",
        "mass_scale": 1.0,
        "damping_scale": 1.0,
        "frictionloss": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "mass_x0.5",
        "mass_scale": 0.5,
        "damping_scale": 1.0,
        "frictionloss": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "mass_x2.0",
        "mass_scale": 2.0,
        "damping_scale": 1.0,
        "frictionloss": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "damping_x0.3",
        "mass_scale": 1.0,
        "damping_scale": 0.3,
        "frictionloss": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "damping_x3.0",
        "mass_scale": 1.0,
        "damping_scale": 3.0,
        "frictionloss": 0.0,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "joint_friction_0.05",
        "mass_scale": 1.0,
        "damping_scale": 1.0,
        "frictionloss": 0.05,
        "action_noise_std": 0.0,
    },
    {
        "scenario": "action_noise_0.5",
        "mass_scale": 1.0,
        "damping_scale": 1.0,
        "frictionloss": 0.0,
        "action_noise_std": 0.5,
    },
]


def apply_domain_parameters(model, data, mass_scale, damping_scale, frictionloss):
    """Apply one domain configuration directly to the compiled MuJoCo model."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hinge_rod")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_hinge")
    dof_adr = model.jnt_dofadr[joint_id]

    # 质量和惯量一起缩放，避免“质量变了但转动惯量没变”的不一致模型。
    model.body_mass[body_id] *= mass_scale
    model.body_inertia[body_id] *= mass_scale

    # hinge 关节的阻尼和摩擦是 sim-to-real 中常见的不确定物理量。
    model.dof_damping[dof_adr] *= damping_scale
    model.dof_frictionloss[dof_adr] = frictionloss

    # 修改模型参数后，让 MuJoCo 重新计算内部常量。
    mujoco.mj_setConst(model, data)


def run_episode(config, rng, record_curve=False):
    """Run one PD-control episode under one physical parameter configuration."""
    # 每个 episode 都重新加载模型，保证不同实验之间不会互相污染参数。
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 在仿真开始前注入 domain randomization / fixed perturbation 参数。
    apply_domain_parameters(
        model,
        data,
        config["mass_scale"],
        config["damping_scale"],
        config["frictionloss"],
    )

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_hinge")
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    actuator_id = 0

    # 初始状态：从 0 度、0 角速度开始，方便比较不同物理参数的影响。
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_adr] = np.deg2rad(INIT_ANGLE_DEG)
    data.qvel[qvel_adr] = 0.0
    mujoco.mj_forward(model, data)

    steps = int(SIM_TIME / model.opt.timestep)
    times = []
    angles_deg = []
    torques = []

    for _ in range(steps):
        # 读取当前 observation 中最关键的两个状态：角度 q 和角速度 qd。
        q = data.qpos[qpos_adr]
        qd = data.qvel[qvel_adr]

        # 固定 PD 控制器：比例项拉向目标角度，微分项抑制速度。
        torque = KP * (TARGET_ANGLE - q) - KD * qd

        # action noise 模拟真实执行器噪声或控制信号误差。
        if config["action_noise_std"] > 0.0:
            torque += rng.normal(0.0, config["action_noise_std"])

        # 与 XML 中 motor ctrlrange 对齐，显式限制最大力矩。
        torque = float(np.clip(torque, TORQUE_MIN, TORQUE_MAX))

        # 写入 actuator control，然后推进一个 MuJoCo 物理步。
        data.ctrl[actuator_id] = torque
        mujoco.mj_step(model, data)

        # 记录 step 后的状态，用于统计指标和画曲线。
        times.append(data.time)
        angles_deg.append(np.rad2deg(data.qpos[qpos_adr]))
        torques.append(torque)

    angles_deg = np.asarray(angles_deg)
    torques = np.asarray(torques)

    final_angle_deg = float(angles_deg[-1])
    final_error_deg = TARGET_ANGLE_DEG - final_angle_deg
    last_error_deg = abs(final_error_deg)
    max_overshoot_deg = max(0.0, float(np.max(angles_deg) - TARGET_ANGLE_DEG))

    # 汇总本 episode 的标量指标，后面会写入 CSV。
    result = {
        "scenario": config["scenario"],
        "mass_scale": config["mass_scale"],
        "damping_scale": config["damping_scale"],
        "frictionloss": config["frictionloss"],
        "action_noise_std": config["action_noise_std"],
        "init_angle_deg": INIT_ANGLE_DEG,
        "final_angle_deg": final_angle_deg,
        "final_error_deg": final_error_deg,
        "last_error_deg": last_error_deg,
        "max_overshoot_deg": max_overshoot_deg,
        "mean_abs_torque": float(np.mean(np.abs(torques))),
        # 这里的 success 是教学实验中的简单判据：最终误差不超过 5 度。
        "success": last_error_deg <= SUCCESS_TOL_DEG,
    }

    # 固定扰动实验需要保留整条轨迹用于画图；随机实验只保留 summary。
    curve = None
    if record_curve:
        curve = {
            "scenario": config["scenario"],
            "time": np.asarray(times),
            "angle_deg": angles_deg,
        }

    return result, curve


def make_random_config(rng, episode_idx):
    """Sample one random domain for the domain randomization experiment."""
    return {
        "scenario": f"domain_randomization_{episode_idx:02d}",
        "mass_scale": float(rng.uniform(0.5, 2.0)),
        "damping_scale": float(rng.uniform(0.3, 3.0)),
        "frictionloss": float(rng.uniform(0.0, 0.08)),
        "action_noise_std": float(rng.uniform(0.0, 0.5)),
    }


def write_results_csv(rows):
    """Save fixed perturbation and random-domain episode summaries."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def plot_fixed_curves(curves):
    """Plot only the fixed perturbation curves so the figure stays readable."""
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for curve in curves:
        ax.plot(curve["time"], curve["angle_deg"], label=curve["scenario"])

    ax.axhline(
        TARGET_ANGLE_DEG,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="45 deg target",
    )
    ax.set_xlabel("time / s")
    ax.set_ylabel("joint angle / degrees")
    ax.set_title("Domain Randomization Hinge PD: Fixed Perturbations")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def print_fixed_summary(rows):
    """Print a compact terminal table for the fixed perturbation experiments."""
    print("Fixed perturbation summary")
    print(
        f"{'scenario':22s} {'final_deg':>10s} {'error_deg':>10s} "
        f"{'overshoot':>10s} {'mean|tau|':>10s} {'success':>8s}"
    )
    for row in rows:
        print(
            f"{row['scenario']:22s} "
            f"{row['final_angle_deg']:10.3f} "
            f"{row['final_error_deg']:10.3f} "
            f"{row['max_overshoot_deg']:10.3f} "
            f"{row['mean_abs_torque']:10.3f} "
            f"{str(row['success']):>8s}"
        )


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Model XML not found: {XML_PATH}")

    # 1) 跑固定扰动实验，并保存曲线用于画 PNG。
    fixed_rows = []
    fixed_curves = []
    for idx, config in enumerate(FIXED_SCENARIOS):
        row, curve = run_episode(config, np.random.default_rng(100 + idx), True)
        fixed_rows.append(row)
        fixed_curves.append(curve)

    # 2) 跑 20 个随机 domain randomization episodes，只记录统计结果。
    rng = np.random.default_rng(0)
    random_rows = []
    for episode_idx in range(20):
        config = make_random_config(rng, episode_idx)
        row, _ = run_episode(config, np.random.default_rng(1000 + episode_idx))
        random_rows.append(row)

    # 3) 输出表格、图片和终端 summary。
    all_rows = fixed_rows + random_rows
    write_results_csv(all_rows)
    plot_fixed_curves(fixed_curves)

    print_fixed_summary(fixed_rows)
    print(f"\nSaved CSV: {CSV_PATH}")
    print(f"Saved plot: {PNG_PATH}")


if __name__ == "__main__":
    main()
