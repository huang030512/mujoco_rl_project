import robosuite as suite

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=False,
    reward_shaping=True
)

for i in range(5):
    obs = env.reset()
    print(f"Reset {i + 1} cube_pos:", obs["cube_pos"])

env.close()
