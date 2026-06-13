from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def forward_kinematics(theta1, theta2, l1=1.0, l2=1.0):
    """Return base, elbow, and end-effector points for a 2D two-link arm."""
    base = np.array([0.0, 0.0])

    # theta1 is measured from the x-axis; theta2 is relative to link 1.
    elbow = base + l1 * np.array([np.cos(theta1), np.sin(theta1)])
    end_effector = elbow + l2 * np.array(
        [np.cos(theta1 + theta2), np.sin(theta1 + theta2)]
    )

    return base, elbow, end_effector


def jacobian(theta1, theta2, l1=1.0, l2=1.0):
    """Return the 2x2 Jacobian mapping joint velocity to end-effector velocity."""
    theta12 = theta1 + theta2

    return np.array(
        [
            [
                -l1 * np.sin(theta1) - l2 * np.sin(theta12),
                -l2 * np.sin(theta12),
            ],
            [
                l1 * np.cos(theta1) + l2 * np.cos(theta12),
                l2 * np.cos(theta12),
            ],
        ]
    )


def main():
    l1 = 1.0
    l2 = 1.0
    q = np.deg2rad(np.array([10.0, 10.0]))
    target = np.array([1.2, 0.6])
    Kp = 1.0
    dt = 0.05
    max_steps = 200
    tolerance = 1e-3

    initial_q = q.copy()
    _, _, initial_end_effector = forward_kinematics(q[0], q[1], l1, l2)
    trajectory = [initial_end_effector]

    total_steps = 0
    final_error_norm = np.inf

    for step in range(max_steps):
        _, _, end_effector = forward_kinematics(q[0], q[1], l1, l2)
        error = target - end_effector
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            final_error_norm = error_norm
            total_steps = step
            break

        x_dot_des = Kp * error
        J = jacobian(q[0], q[1], l1, l2)
        q_dot = np.linalg.pinv(J) @ x_dot_des
        q = q + q_dot * dt

        _, _, new_end_effector = forward_kinematics(q[0], q[1], l1, l2)
        trajectory.append(new_end_effector)
        total_steps = step + 1
    else:
        _, _, end_effector = forward_kinematics(q[0], q[1], l1, l2)
        final_error_norm = np.linalg.norm(target - end_effector)

    final_base, final_elbow, final_end_effector = forward_kinematics(q[0], q[1], l1, l2)
    initial_base, initial_elbow, initial_end_effector = forward_kinematics(
        initial_q[0], initial_q[1], l1, l2
    )
    trajectory = np.array(trajectory)

    print(f"final theta1, theta2 (rad): {q[0]:.6f}, {q[1]:.6f}")
    print(f"final theta1, theta2 (deg): {np.rad2deg(q[0]):.6f}, {np.rad2deg(q[1]):.6f}")
    print(f"final end-effector: {final_end_effector}")
    print(f"final error norm: {final_error_norm:.12f}")
    print(f"total steps: {total_steps}")

    initial_points = np.vstack([initial_base, initial_elbow, initial_end_effector])
    final_points = np.vstack([final_base, final_elbow, final_end_effector])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        initial_points[:, 0],
        initial_points[:, 1],
        "-o",
        color="tab:gray",
        alpha=0.35,
        linewidth=2,
        markersize=6,
        label="initial arm",
    )
    ax.plot(
        final_points[:, 0],
        final_points[:, 1],
        "-o",
        color="tab:blue",
        linewidth=3,
        markersize=8,
        label="final arm",
    )
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="tab:orange",
        linewidth=2,
        label="end-effector trajectory",
    )
    ax.scatter(target[0], target[1], s=100, marker="x", color="tab:red", label="target")

    ax.scatter(final_base[0], final_base[1], s=80, color="tab:blue")
    ax.scatter(final_elbow[0], final_elbow[1], s=80, color="tab:blue")
    ax.scatter(final_end_effector[0], final_end_effector[1], s=80, color="tab:blue")

    ax.annotate("base", xy=final_base, xytext=(8, -14), textcoords="offset points")
    ax.annotate("elbow", xy=final_elbow, xytext=(8, 8), textcoords="offset points")
    ax.annotate(
        "end-effector",
        xy=final_end_effector,
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate("target", xy=target, xytext=(8, -16), textcoords="offset points")

    ax.set_title("2D Two-Link Planar Arm Task-Space Control")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.8, 1.4)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "week06_planar_arm_task_space_control.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved task-space control visualization to {output_path}")


if __name__ == "__main__":
    main()
