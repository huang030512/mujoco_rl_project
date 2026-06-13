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


def inverse_kinematics(x, y, l1=1.0, l2=1.0, elbow_up=False):
    """Solve analytical IK for a reachable 2D two-link arm target."""
    r = np.sqrt(x**2 + y**2)
    max_reach = l1 + l2
    min_reach = abs(l1 - l2)

    if r > max_reach:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) is unreachable: r={r:.3f} > {max_reach:.3f}"
        )
    if r < min_reach:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) is unreachable: r={r:.3f} < {min_reach:.3f}"
        )

    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    theta2 = np.arccos(cos_theta2)
    if elbow_up:
        theta2 = -theta2

    theta1 = np.arctan2(y, x) - np.arctan2(
        l2 * np.sin(theta2),
        l1 + l2 * np.cos(theta2),
    )

    return theta1, theta2


def main():
    l1 = 1.0
    l2 = 1.0
    target = np.array([1.2, 0.6])

    theta1, theta2 = inverse_kinematics(target[0], target[1], l1, l2)
    base, elbow, end_effector = forward_kinematics(theta1, theta2, l1, l2)
    points = np.vstack([base, elbow, end_effector])
    error = np.linalg.norm(end_effector - target)

    print(f"target: {target}")
    print(f"theta1, theta2 (rad): {theta1:.6f}, {theta2:.6f}")
    print(f"theta1, theta2 (deg): {np.rad2deg(theta1):.6f}, {np.rad2deg(theta2):.6f}")
    print(f"FK end_effector: {end_effector}")
    print(f"error norm: {error:.12f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], "-o", linewidth=3, markersize=8)

    ax.scatter(base[0], base[1], s=80, label="base")
    ax.scatter(elbow[0], elbow[1], s=80, label="elbow")
    ax.scatter(end_effector[0], end_effector[1], s=80, label="end-effector")
    ax.scatter(target[0], target[1], s=100, marker="x", label="target")

    ax.annotate("base", xy=base, xytext=(8, -14), textcoords="offset points")
    ax.annotate("elbow", xy=elbow, xytext=(8, 8), textcoords="offset points")
    ax.annotate(
        "end-effector",
        xy=end_effector,
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate("target", xy=target, xytext=(8, -16), textcoords="offset points")

    ax.set_title("2D Two-Link Planar Arm Analytical IK")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.8, 1.4)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "assets" / "week06_planar_arm_ik.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved IK visualization to {output_path}")


if __name__ == "__main__":
    main()
