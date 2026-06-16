# MuJoCo RL Project

## 项目简介

本项目是一个面向机器人学习、强化学习控制、模仿学习、运动控制与 Sim-to-Real 方向的 MuJoCo / robosuite / Gymnasium / Stable-Baselines3 / PyTorch 实验项目。

项目从 MuJoCo 基础模型加载、自由落体仿真和单关节 PD 控制开始，逐步扩展到单关节 PPO 强化学习训练闭环、robosuite Panda Lift 机械臂操作任务、SAC 最小训练流程、Behavior Cloning 模仿学习、多轨迹数据筛选、Walker2d / Humanoid locomotion 入门，以及 Domain Randomization / Sim-to-Real 概念实验。

本项目的重点不是声称已经获得高成功率的机械臂抓取策略，而是展示一个完整、可复现、可解释的机器人学习项目搭建过程，包括环境验证、模型理解、状态 / 动作 / 奖励分析、控制实验、策略训练、专家数据采集、模仿学习、rollout 评估、失败原因分析和真实边界表达。

---

## 项目亮点

- 搭建 MuJoCo + Python 仿真环境，完成模型加载、状态读取、自由落体仿真和可视化分析；
- 编写 MJCF 模型，理解 `MjModel`、`MjData`、`qpos`、`qvel`、`ctrl` 等核心数据结构；
- 搭建单关节可控模型，实现 PD 控制，并分析不同参数对响应速度、超调和稳定性的影响；
- 将单关节 MuJoCo 模型封装为 Gymnasium 环境，设计 observation、action、reward 和 termination，并使用 PPO 完成强化学习训练闭环；
- 接入 robosuite Panda Lift 机械臂操作任务，分析 observation、action、reward、success 和 reset 机制；
- 实现 Panda Lift random baseline、handcrafted teacher、最小 SAC 训练流程和 Behavior Cloning 模仿学习实验；
- 基于 handcrafted policy 采集 demonstration 数据，使用 PyTorch MLP policy 拟合专家动作，并完成 rollout 评估；
- 构建多轨迹数据采集与 success-filtered 数据筛选流程，记录 reward、success、max_lift、final_lift、min_dist 等指标，并分析 BC 策略失败原因；
- 完成 Walker2d / Humanoid locomotion 入门实验，理解 torque action、forward reward、control cost、survive reward 与摔倒 termination；
- 完成 MuJoCo Domain Randomization / Sim-to-Real 小实验，观察质量、阻尼、摩擦、初始扰动和 action noise 对控制稳定性的影响；
- 整理实验记录、结果图、演示视频和项目边界，形成可用于简历展示的机器人学习 GitHub 作品集。

---

## 技术栈

- Python 3.10
- MuJoCo
- robosuite
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- Matplotlib
- Linux / Ubuntu
- Git / GitHub

---

## 关键词

MuJoCo, robosuite, Gymnasium, PyTorch, Stable-Baselines3, PPO, SAC, Behavior Cloning, DAgger, Panda Lift, Walker2d, Humanoid, Domain Randomization, Sim-to-Real, Robotics Control

---

## 核心目录

```text
mujoco_rl_project/
├── README.md
├── requirements.txt
├── models/      # MuJoCo / MJCF 模型与训练模型
├── src/         # 仿真、控制、训练、评估和数据采集脚本
├── data/        # Panda Lift BC demonstration 数据
├── assets/      # 实验曲线、结果图和演示视频
├── notes/       # 每周实验记录、失败分析和面试整理材料
└── results/     # 实验输出和结果说明
```

---

## 1. MuJoCo 基础仿真：自由落体实验

第一阶段完成 MuJoCo 环境验证、MJCF 模型加载、仿真状态读取和自由落体结果可视化。

核心内容包括：

- 加载 MuJoCo XML 模型；
- 理解 `MjModel` 和 `MjData`；
- 读取 `qpos`、`qvel` 等状态字段；
- 使用 `mj_step()` 推进仿真；
- 记录位置和速度曲线。

核心文件：

```bash
models/simple_box.xml
src/check_mujoco.py
src/run_simple_box.py
src/simulate_box_plot.py
src/inspect_model.py
```

实验结果中，box 从约 0.5 m 高度下落，与地面接触后稳定在约 0.1 m 高度附近，与 box 半高一致。

---

## 2. 单关节控制：PD 控制实验

第二阶段从自由物体仿真转向可控关节模型，搭建单关节旋转杆 `hinge_rod.xml`，并实现 PD 控制。

核心内容包括：

- 使用 `<joint>` 定义可运动自由度；
- 使用 `<actuator>` 将控制输入作用到关节；
- 理解 `data.ctrl[i]` 作为控制量输入；
- 分析 `Kp` 和 `Kd` 对响应速度、超调和稳定性的影响。

控制形式：

```text
torque = Kp * (target_angle - current_angle) - Kd * current_angular_velocity
```

核心文件：

```bash
models/hinge_rod.xml
src/read_joint.py
src/run_single_joint_pd.py
notes/week02_pd_demo.md
```

---

## 3. 单关节强化学习：PPO 训练闭环

在完成 PD 控制后，本项目将单关节 MuJoCo 模型封装为 Gymnasium 环境，并使用 Stable-Baselines3 PPO 训练策略控制旋转杆到达目标角度。

该阶段完成了从自建 MuJoCo 模型到强化学习训练的完整闭环：

- 自定义 Gymnasium 环境；
- 定义 observation space 和 action space；
- 设计 reward 和 termination；
- 使用 PPO 训练策略；
- 保存模型并进行评估；
- 绘制角度曲线和 torque 曲线。

核心文件：

```bash
src/hinge_rod_env.py
src/test_hinge_rod_env.py
src/train_ppo_hinge_rod_full.py
src/eval_ppo_hinge_rod.py
src/plot_ppo_hinge_rod_eval.py
models/ppo_hinge_rod/final_model.zip
models/ppo_hinge_rod/training_curves.png
notes/week02_ppo_rl_demo.md
```

阶段性结果显示，PPO 策略能够将单关节旋转杆控制到目标角度附近，说明 MuJoCo + Gymnasium + Stable-Baselines3 的训练流程已经跑通。

---

## 4. Panda Lift 机械臂操作任务分析

第三阶段接入 robosuite Panda Lift 任务，从单关节控制扩展到标准机械臂操作任务。

该阶段主要分析：

- Panda 七自由度机械臂和两指夹爪；
- observation 中的关节状态、末端位姿、夹爪状态、方块位置和相对位置；
- 7 维连续 action 对末端运动和夹爪开合的控制作用；
- reward、done、success 和 reset 随机化机制；
- 为什么随机策略难以完成抓取任务。

核心文件：

```bash
src/test_robosuite_lift.py
src/test_panda_action_dimensions.py
src/test_panda_gripper.py
src/test_panda_reset.py
src/analyze_panda_lift_task.py
notes/week03_panda_lift_intro.md
```

---

## 5. Panda Lift Baseline：随机策略与手工策略

为了建立任务基线，本项目首先运行 random baseline，并进一步实现 handcrafted policy，用于验证 Panda Lift 任务接口和控制逻辑。

Random baseline 用于观察随机动作下的 reward、success 和任务失败现象。Handcrafted policy 则通过分阶段逻辑完成接近方块、下降、闭合夹爪和尝试抬升。

手工策略大致分为：

1. 移动到方块上方；
2. 下降到抓取高度；
3. 闭合夹爪；
4. 抬升方块；
5. 根据 success 判断任务是否完成。

核心文件：

```bash
src/run_panda_lift_random_baseline.py
src/run_panda_lift_handcrafted_policy.py
assets/week03_panda_lift_random_demo.mp4
assets/week03_panda_lift_handcrafted_ep0.mp4
notes/week03_panda_lift_baseline_analysis.md
notes/week03_panda_lift_handcrafted_policy.md
```

该部分的意义在于：

- 验证 observation 读取是否正确；
- 验证 action 控制方向是否正确；
- 验证 gripper 开合符号是否正确；
- 验证 reward 和 success 机制是否正常；
- 为后续强化学习训练和模仿学习实验提供任务理解基础。

---

## 6. Panda Lift 最小 SAC 训练流程

在完成任务分析和手工策略验证后，本项目搭建了 Panda Lift 的最小 SAC 强化学习训练流程。

该阶段主要完成：

- 将 robosuite Panda Lift 环境封装为 Stable-Baselines3 可训练接口；
- 对 observation 进行 flatten 处理；
- 适配连续 action space；
- 使用 SAC 进行最小训练实验；
- 保存训练模型；
- 记录 episode reward；
- 绘制 reward 曲线；
- 编写评估脚本检查训练结果。

核心文件：

```bash
src/panda_lift_rl_env.py
src/test_panda_lift_rl_env.py
src/train_panda_lift_sac.py
src/evaluate_panda_lift_sac.py
models/sac_panda_lift_minimal/final_model.zip
models/sac_panda_lift_minimal/reward_curve.png
notes/week03_panda_lift_minimal_rl.md
```

该阶段主要用于验证强化学习训练接口和实验流程，不代表已经获得稳定高成功率的机械臂抓取策略。

---

## 7. Panda Lift Behavior Cloning：模仿学习实验

在完成 Panda Lift 手工策略和最小 SAC 训练流程后，本项目进一步实现 Behavior Cloning 模仿学习实验。

该实验使用 handcrafted policy 作为 scripted expert，采集专家策略在 robosuite Lift 环境中的 observation-action demonstration 数据，并使用 PyTorch 构建 MLP policy 进行监督学习训练，使神经网络策略模仿手工专家策略的动作输出。

实验流程包括：

1. 使用 handcrafted policy 根据当前 observation 和 phase 生成专家动作；
2. 在每个仿真 step 记录 observation vector 和 expert action；
3. 将 demonstration 数据保存为 `.npz` 文件；
4. 使用 PyTorch Dataset / DataLoader 读取 BC 数据；
5. 构建两层隐藏层 MLP policy，输入 observation，输出 7 维连续 action；
6. 使用 MSE loss 训练 MLP policy 拟合专家动作；
7. 将训练后的 policy 加载回 Panda Lift 环境进行闭环评估。

核心文件：

```bash
src/collect_panda_lift_bc_data.py
src/train_panda_lift_bc.py
src/eval_panda_lift_bc.py
data/panda_lift_bc_data.npz
models/bc/panda_lift_bc_policy.pt
```

实验数据：

```text
Number of episodes: 30
Number of samples: 9000
Observation dimension: 15
Action dimension: 7
Best validation loss: 0.000045
```

需要注意的是，当前 handcrafted policy 本身仍属于弱专家策略。因此，BC policy 虽然能够模仿专家动作模式，但尚未获得稳定完成 lift 的能力。该实验的价值在于跑通了从 scripted expert、demonstration data、Behavior Cloning 到 policy evaluation 的完整模仿学习闭环。

---

## 8. 多轨迹数据筛选与 BC 失败分析

在基础 BC 实验之后，本项目进一步加入多轨迹数据采集、轨迹指标记录和 success-filtered 数据筛选，用于分析 demonstration 数据质量对 BC policy 的影响。

记录指标包括：

- `total_reward`
- `success`
- `success_steps`
- `max_lift`
- `final_lift`
- `min_eef_cube_dist`

核心文件：

```bash
src/collect_panda_lift_bc_data_with_metrics.py
data/panda_lift_bc_data_with_metrics_10ep.npz
data/panda_lift_bc_data_filtered_success_10ep.npz
notes/week05_panda_lift_bc_failure_analysis.md
```

该阶段主要分析：

- BC policy 为什么能靠近方块但不一定能稳定抓取；
- phase 切换是否可靠；
- grasp position 是否足够稳定；
- lift 阶段是否真正锁定方块；
- success-filtered 数据是否能提升 imitation learning 数据质量。

该部分帮助项目从“跑通 BC 训练”进一步扩展到“理解模仿学习失败原因和数据质量问题”。

---

## 9. FK / IK 与任务空间控制入门

为了补充机器人运动学基础，本项目实现了平面机械臂 FK、IK 与任务空间控制小实验。

核心内容包括：

- 正运动学 FK：由关节角计算末端位置；
- 逆运动学 IK：由目标末端位置求解关节角；
- 任务空间控制：基于末端误差调整关节运动；
- 理解关节空间和任务空间的区别。

核心文件：

```bash
src/planar_arm_fk_demo.py
src/planar_arm_ik_demo.py
src/planar_arm_task_space_control.py
notes/week06_fk_ik_task_space_control.md
```

该部分用于补充机器人操作、末端控制、任务空间控制和后续 manipulation 学习的基础。

---

## 10. Walker2d / Humanoid Locomotion 入门

为了补充机器人运动控制和人形 / 足式机器人方向基础，本项目学习并分析了 MuJoCo Walker2d / Humanoid locomotion 任务。

分析重点包括：

- observation 中的关节状态、速度和身体姿态；
- action torque 对各关节的控制作用；
- forward reward、control cost、survive reward 的含义；
- terminated / truncated 的区别；
- 为什么机器人会摔倒；
- locomotion 与 Panda Lift manipulation 的区别。

核心文件：

```bash
src/analyze_walker2d_random_rollout.py
notes/week06_walker2d_locomotion_intro.md
```

该阶段目标不是训练高性能 locomotion policy，而是理解人形 / 足式运动控制任务的状态、动作、奖励和终止机制，为后续机器人运动控制和 DRL 岗位面试建立基础。

---

## 11. Domain Randomization / Sim-to-Real 小实验

为了理解 Sim-to-Real gap 和策略鲁棒性，本项目基于 MuJoCo 单关节控制任务完成 Domain Randomization / 参数扰动实验。

扰动因素包括：

- 质量 mass；
- 阻尼 damping；
- 摩擦 friction；
- 初始状态扰动；
- action noise。

观察指标包括：

- final angle；
- error；
- overshoot；
- mean torque；
- success；
- 不同扰动条件下的控制曲线变化。

核心文件：

```bash
src/domain_randomization_hinge_pd_experiment.py
assets/week06_domain_randomization_hinge_pd.png
assets/week06_domain_randomization_hinge_pd_results.csv
notes/week06_domain_randomization_sim2real.md
```

该阶段用于理解：

- 为什么仿真和真实机器人之间存在差距；
- domain randomization 如何提升策略鲁棒性；
- 系统参数变化为什么会影响控制效果；
- 后续真实机器人部署为什么需要 system identification、安全限幅和低层控制接口适配。

---

## 12. 强化学习与模仿学习面试整理

除代码实验外，本项目还整理了面向机器人 DRL / 模仿学习 / 运动控制岗位的原理与面试材料。

覆盖内容包括：

- PPO
- SAC
- Behavior Cloning
- DAgger
- On-policy / Off-policy
- Model-free / Model-based
- rollout
- critic / advantage
- BC 与 RL 的区别
- sim-to-real 与 domain randomization
- locomotion 与 manipulation 的区别

核心文件：

```bash
notes/week06_rl_interview_package.md
notes/week07_job_matching_project_package.md
```

该部分用于将实验项目转化为可讲述、可面试、可投递的作品集表达。

---

## 当前结果与真实边界

### 已完成内容

目前项目已经完成：

- MuJoCo 基础仿真环境搭建；
- MJCF 模型结构与状态字段理解；
- 自由落体仿真和曲线记录；
- 单关节 PD 控制实验；
- 单关节 PPO 强化学习训练闭环；
- robosuite Panda Lift 环境接入；
- Panda Lift observation / action / reward / success 分析；
- Panda Lift random baseline；
- Panda Lift handcrafted policy；
- Panda Lift 最小 SAC 训练流程；
- Panda Lift Behavior Cloning 模仿学习实验；
- Panda Lift 多轨迹 demonstration 数据采集与 success-filtered 数据筛选；
- BC 策略 rollout 诊断与失败原因分析；
- FK / IK 与任务空间控制入门实验；
- Walker2d / Humanoid locomotion 任务分析；
- MuJoCo Domain Randomization / Sim-to-Real 参数扰动实验；
- 面向机器人 DRL / 模仿学习 / 运动控制岗位的项目总结与面试材料整理。

### 真实边界

当前项目仍处于机器人强化学习和模仿学习入门与流程验证阶段，存在以下边界：

- Panda Lift SAC 训练时间较短，尚未获得稳定高成功率策略；
- 当前 SAC 实验主要用于验证训练闭环，不代表已经解决机械臂抓取任务；
- 当前 Behavior Cloning policy 主要模仿 weak handcrafted expert，尚未获得稳定 lift 能力；
- PPO / SAC 超参数尚未进行系统性调参；
- Panda Lift reward shaping、controller 配置和 observation 设计仍有优化空间；
- Domain Randomization 目前为 MuJoCo 概念实验，尚未完成真实机器人部署；
- Walker2d / Humanoid 当前以任务理解和随机 rollout 分析为主，尚未训练高性能 locomotion policy；
- 项目目前全部在仿真环境中完成，尚未迁移到真实机械臂或真实足式机器人。

该项目的真实价值在于展示机器人学习项目从 0 到 1 的搭建能力，包括环境配置、任务理解、控制实验、强化学习训练接口封装、专家数据采集、模仿学习训练、结果记录和问题边界分析。

---

## 如何运行

### 1. 安装依赖

建议使用 conda 环境：

```bash
conda create -n mujoco_rl python=3.10
conda activate mujoco_rl
pip install -r requirements.txt
```

### 2. 检查 MuJoCo 环境

```bash
python src/check_mujoco.py
```

### 3. 运行自由落体仿真

```bash
python src/run_simple_box.py
python src/simulate_box_plot.py
```

### 4. 运行单关节 PD 控制

```bash
python src/read_joint.py
python src/run_single_joint_pd.py
```

### 5. 训练与评估单关节 PPO

```bash
python src/test_hinge_rod_env.py
python src/train_ppo_hinge_rod_full.py
python src/eval_ppo_hinge_rod.py
python src/plot_ppo_hinge_rod_eval.py
```

### 6. 测试 Panda Lift 环境

```bash
python src/test_robosuite_lift.py
python src/test_panda_action_dimensions.py
python src/test_panda_gripper.py
python src/test_panda_reset.py
python src/analyze_panda_lift_task.py
```

### 7. 运行 Panda Lift baseline

```bash
python src/run_panda_lift_random_baseline.py
python src/run_panda_lift_handcrafted_policy.py
```

### 8. 训练与评估 Panda Lift SAC

```bash
python src/test_panda_lift_rl_env.py
python src/train_panda_lift_sac.py
python src/evaluate_panda_lift_sac.py
```

### 9. 训练与评估 Panda Lift Behavior Cloning

```bash
python src/collect_panda_lift_bc_data.py
python src/train_panda_lift_bc.py
python src/eval_panda_lift_bc.py
```

### 10. 运行 Walker2d locomotion 分析

```bash
python src/analyze_walker2d_random_rollout.py
```

### 11. 运行 Domain Randomization 小实验

```bash
python src/domain_randomization_hinge_pd_experiment.py
```

---

## 后续计划

后续计划包括：

- 改进 Panda Lift handcrafted policy，提高 expert demonstration 质量；
- 系统调节 Panda Lift SAC / PPO 超参数；
- 增加 success rate 曲线和多随机种子实验；
- 优化 reward shaping；
- 调整 robosuite controller 配置；
- 尝试使用 Behavior Cloning policy 初始化 SAC / PPO 训练；
- 探索 DAgger 等交互式模仿学习方法；
- 尝试更复杂的 robosuite 操作任务，例如 PickPlace、Stack、NutAssembly；
- 学习 Isaac Gym / Isaac Sim / Isaac Lab，并理解其在人形 / 足式机器人并行训练中的作用；
- 进一步学习阻抗控制、MPC、WBC、QP 等机器人运动控制方法；
- 探索 sim-to-real 和真实机器人操作任务迁移。

---

## 项目定位

本项目是个人面向机器人学习、强化学习控制、模仿学习、运动控制和具身智能方向的阶段性实践项目。

项目重点展示：

- MuJoCo / robosuite 仿真环境搭建能力；
- 机器人状态、动作、奖励和成功判据分析能力；
- Python 强化学习实验代码组织能力；
- Stable-Baselines3 训练流程使用能力；
- PyTorch 模仿学习实验实现能力；
- 从 scripted expert 到 demonstration data，再到 neural policy evaluation 的模仿学习流程理解；
- locomotion 和 manipulation 任务的基础理解；
- domain randomization 和 sim-to-real gap 的基础认知；
- 实验结果记录、可视化和真实边界表达能力。

该项目会作为后续学习机器人操作、模仿学习、强化学习控制和具身智能算法的基础。