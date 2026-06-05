# MuJoCo RL Project

## 项目目标

本项目面向机器人学习与具身智能方向，逐步建立 MuJoCo + Python 仿真实验基础，并为后续强化学习控制任务准备可复现的模型、代码与实验记录。

## 第一周成果：自由落体仿真演示

第一周完成了 MuJoCo 开发环境验证、简单模型加载、模型结构与状态字段检查，以及自由落体仿真结果的记录和可视化分析。

通过本周工作，已经建立从 MJCF 模型定义、MuJoCo 模型加载、仿真状态读取、时间步推进到实验结果绘图分析的基础流程。

### 已完成内容

- 环境验证脚本：`src/check_mujoco.py`
- 自由落体模型：`models/simple_box.xml`
- 基础仿真脚本：`src/run_simple_box.py`
- 状态曲线绘图脚本：`src/simulate_box_plot.py`
- 模型结构与状态字段检查脚本：`src/inspect_model.py`
- `MjModel` 与 `MjData` 学习记录：`notes/week01_day04_mjmodel_mjdata.md`
- 仿真结果分析记录：`notes/week01_day05_simulation_analysis.md`
- 第一周复盘总结：`notes/week01_review.md`
- 位置变化曲线：`assets/week01_box_free_fall_position.png`
- 速度变化曲线：`assets/week01_box_free_fall_velocity.png`

## 第二周成果：单关节控制与强化学习训练闭环

第二周完成了从自由落体模型到可控关节模型的过渡，并基于单关节旋转杆构建了 MuJoCo + Gymnasium + Stable-Baselines3 的强化学习训练闭环。

主要完成内容包括：

- 搭建单关节旋转杆模型 `models/hinge_rod.xml`；
- 理解 MJCF 中 `<joint>` 与 `<actuator>` 的对应关系；
- 实现单关节 PD 控制实验，观察不同 `Kp`、`Kd` 参数对系统响应的影响；
- 将 MuJoCo 单关节模型封装为 Gymnasium 环境；
- 使用 Stable-Baselines3 PPO 训练单关节摆杆到达目标角度；
- 完成模型保存、评估、测试和训练曲线可视化；
- 整理强化学习训练闭环实验记录。

阶段性结果显示，PPO 策略能够将单关节旋转杆稳定控制到目标角度附近，完成从自建 MuJoCo 模型到强化学习控制策略的完整闭环。

相关文件包括：

- `models/hinge_rod.xml`
- `src/hinge_rod_env.py`
- `src/train_hinge_rod_ppo.py`
- `src/evaluate_hinge_rod_ppo.py`
- `src/test_hinge_rod_env.py`
- `models/ppo_hinge_rod/final_model.zip`
- `models/ppo_hinge_rod/training_curves.png`
- `notes/week02_ppo_rl_demo.md`

## 第三周成果：Panda Lift 机械臂操作任务分析

第三周从单关节强化学习环境进入 robosuite 标准机械臂操作任务，完成了 Panda Lift 环境运行、任务接口分析和随机策略 baseline。

主要完成内容包括：

- 跑通 robosuite Lift 任务，使用 Panda 七自由度机械臂和两指夹爪；
- 理解 observation 字段，包括机械臂关节状态、末端执行器位置、方块位置和相对位置信息；
- 分析 7 维连续 action，理解其对末端位置、姿态和夹爪开合的控制作用；
- 记录 reward、done、success 和 reset 随机化对任务的影响；
- 实现随机策略 baseline，运行 5 个 episode；
- 随机策略平均 reward 为 `3.75`，成功率为 `0.0`；
- 归档 Panda Lift 随机动作演示视频；
- 整理 Panda Lift 任务机制和 baseline 实验记录。

阶段性结果显示，Panda Lift 并不是简单的机械臂运动演示，而是一个包含状态观测、连续动作控制、奖励反馈和成功判定的标准机械臂操作强化学习任务。随机策略无法完成抓取与抬升，说明后续需要基于 observation 设计手工策略或训练强化学习策略。

相关文件包括：

- `src/test_robosuite_lift.py`
- `src/test_panda_action_dimensions.py`
- `src/test_panda_gripper.py`
- `src/test_panda_reset.py`
- `src/analyze_panda_lift_task.py`
- `src/run_panda_lift_random_baseline.py`
- `assets/week03_panda_lift_random_demo.mp4`
- `notes/week03_panda_lift_intro.md`
- `notes/week03_panda_lift_baseline_analysis.md`

## 项目结构

```text
mujoco_rl_project/
├── README.md
├── assets/
│   ├── week01_box_free_fall_position.png
│   ├── week01_box_free_fall_velocity.png
│   ├── week02_joint_motion.png
│   ├── week02_single_joint_pd_*.png
│   └── week03_panda_lift_random_demo.mp4
├── models/
│   ├── simple_box.xml
│   ├── hinge_rod.xml
│   └── ppo_hinge_rod/
│       ├── final_model.zip
│       └── training_curves.png
├── notes/
│   ├── week01_day04_mjmodel_mjdata.md
│   ├── week01_day05_simulation_analysis.md
│   ├── week01_review.md
│   ├── week02_pd_demo.md
│   ├── week02_ppo_rl_demo.md
│   ├── week03_panda_lift_intro.md
│   └── week03_panda_lift_baseline_analysis.md
└── src/
    ├── check_mujoco.py
    ├── inspect_model.py
    ├── run_simple_box.py
    ├── simulate_box_plot.py
    ├── read_joint.py
    ├── run_single_joint_pd.py
    ├── hinge_rod_env.py
    ├── train_hinge_rod_ppo.py
    ├── evaluate_hinge_rod_ppo.py
    ├── test_hinge_rod_env.py
    ├── test_robosuite_lift.py
    ├── test_panda_action_dimensions.py
    ├── test_panda_gripper.py
    ├── test_panda_reset.py
    ├── analyze_panda_lift_task.py
    └── run_panda_lift_random_baseline.py