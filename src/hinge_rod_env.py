import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class HingeRodEnv(gym.Env):
    """
    单关节旋转杆 MuJoCo 环境
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="models/hinge_rod.xml", render_mode="human"):
        super().__init__()
        # 载入模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # PPO 输出归一化动作 [-1, 1]，
        # 后续映射到实际电机控制输入 [-5, 5]
        self.max_torque = 5.0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 状态空间：[关节角度, 关节角速度]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # 一个 PPO 动作连续执行 5 个 MuJoCo 物理步
        self.frame_skip = 5

        # 每个 episode 最长 500 个 RL 步
        self.max_episode_steps = 500
        self.current_step = 0

        # 目标角度
        self.target_angle = np.pi / 4  # 45度

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 每次回合从一个随机初始角度开始
        self.data.qpos[0] = self.np_random.uniform(
            low=-np.pi / 6,
            high=np.pi / 6
        )
        self.data.qvel[0] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        obs = np.array(
            [self.data.qpos[0], self.data.qvel[0]],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):
        # PPO 输出范围是 [-1, 1]，映射为实际力矩 [-5, 5]
        normalized_action = float(np.clip(action[0], -1.0, 1.0))
        torque = normalized_action * self.max_torque

        self.data.ctrl[0] = torque

        # 同一个动作连续执行 5 个物理步
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        theta = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[0])

        obs = np.array([theta, theta_dot], dtype=np.float32)

        # 将角度误差限制在 [-pi, pi]
        angle_error = np.arctan2(
            np.sin(theta - self.target_angle),
            np.cos(theta - self.target_angle)
        )

        # 奖励：角度接近目标、速度小、控制输入不要过大
        reward = -(
            5*angle_error ** 2
            + 0.01 * theta_dot ** 2
            + 0.001 * torque ** 2
        )

        # 不再因为瞬间经过目标角度而直接结束
        terminated = False

        # 一个 episode 固定最多运行 500 步
        truncated = self.current_step >= self.max_episode_steps

        is_success = (
            abs(angle_error) < np.deg2rad(3.0)
            and abs(theta_dot) < np.deg2rad(5.0)
        )

        info = {
            "theta": theta,
            "theta_dot": theta_dot,
            "angle_error": angle_error,
            "torque": torque,
            "is_success": is_success
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # 简单渲染
        if self.render_mode == "human":
            mujoco.mj_render(self.model, self.data)
        elif self.render_mode == "rgb_array":
            # TODO: 可返回像素数组
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        pass