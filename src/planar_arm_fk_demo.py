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


def main():
    l1 = 1.0
    l2 = 1.0
    theta1 = np.deg2rad(30.0)
    theta2 = np.deg2rad(45.0)

    base, elbow, end_effector = forward_kinematics(theta1, theta2, l1, l2)
    points = np.vstack([base, elbow, end_effector])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], "-o", linewidth=3, markersize=8)

    ax.scatter(base[0], base[1], s=80, label="base")
    ax.scatter(elbow[0], elbow[1], s=80, label="elbow")
    ax.scatter(end_effector[0], end_effector[1], s=80, label="end-effector")

    ax.annotate("base", xy=base, xytext=(8, -14), textcoords="offset points")
    ax.annotate("elbow", xy=elbow, xytext=(8, 8), textcoords="offset points")
    ax.annotate(
        "end-effector",
        xy=end_effector,
        xytext=(8, 8),
        textcoords="offset points",
    )

    ax.set_title("2D Two-Link Planar Arm Forward Kinematics")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "assets" / "week06_planar_arm_fk.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved FK visualization to {output_path}")


if __name__ == "__main__":
    main()
