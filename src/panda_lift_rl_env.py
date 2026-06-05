import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite


class PandaLiftLowDimEnv(gym.Env):
    """
    Panda Lift 低维状态强化学习环境封装。

    作用：
    1. 创建 robosuite Lift 环境；
    2. 将 dict observation 转换为一维 numpy 向量；
    3. 将 robosuite step 接口包装成 Gymnasium 接口；
    4. 记录 success 信息，方便后续评估。
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        horizon=200,
        render=False,
        control_freq=20,
        reward_shaping=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.render_enabled = render

        self.env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=horizon,
        )

        low, high = self.env.action_spec

        self.action_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32,
        )

    def _get_obs_value(self, obs, key, default_dim):
        """
        安全读取 observation 中的某个 key。

        如果 key 存在，就读取它；
        如果 key 不存在，就返回对应维度的零向量，避免程序直接崩溃。
        """
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).ravel()

        return np.zeros(default_dim, dtype=np.float32)

    def _flatten_obs(self, obs):
        """
        将 robosuite 的 dict observation 转成一维状态向量。

        这里暂时只选最关键的低维信息：
        1. 末端执行器位置
        2. 方块位置
        3. 夹爪到方块的相对位置
        4. 夹爪开合状态
        """
        eef_pos = self._get_obs_value(obs, "robot0_eef_pos", default_dim=3)
        cube_pos = self._get_obs_value(obs, "cube_pos", default_dim=3)
        gripper_to_cube = self._get_obs_value(obs, "gripper_to_cube_pos", default_dim=3)
        gripper_qpos = self._get_obs_value(obs, "robot0_gripper_qpos", default_dim=2)

        flat_obs = np.concatenate(
            [
                eef_pos,
                cube_pos,
                gripper_to_cube,
                gripper_qpos,
            ],
            axis=0,
        ).astype(np.float32)

        return flat_obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        info = {
            "success": False,
        }

        return flat_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, done, info = self.env.step(action)

        flat_obs = self._flatten_obs(obs)

        success = False
        if hasattr(self.env, "_check_success"):
            success = bool(self.env._check_success())

        info = dict(info)
        info["success"] = success

        terminated = False
        truncated = bool(done)

        return flat_obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_enabled:
            self.env.render()

    def close(self):
        self.env.close()
