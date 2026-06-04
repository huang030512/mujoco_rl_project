import sys
import os
sys.path.append(os.path.dirname(__file__))  # 当前目录
from hinge_rod_env import HingeRodEnv

env = HingeRodEnv(render_mode=None)
obs, info = env.reset()

print("Initial observation:", obs)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step}, "
        f"action={action[0]:.3f}, "
        f"theta={obs[0]:.3f}, "
        f"theta_dot={obs[1]:.3f}, "
        f"reward={reward:.3f}, "
        f"terminated={terminated}"
    )

    if terminated:
        print(f"Reached target! theta={obs[0]:.3f}")
        break

env.close()