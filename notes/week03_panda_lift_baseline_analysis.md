# 第3周任务B：Panda Lift 基线策略与任务分析整理

## 1. 实验目的

本阶段实验基于 robosuite 中的 Panda Lift 任务，重点分析机械臂操作任务中的 Observation、Action、Reward、Done、Reset 等核心接口，并通过随机策略 baseline 观察任务难度和奖励反馈特征。

该任务不是为了立即训练出高成功率策略，而是为了建立对标准机械臂操作强化学习任务的基本理解，包括：

* 策略能够观测到哪些状态信息；
* 动作向量每个维度控制什么；
* reward 如何反映任务进展；
* done 和 success 的含义；
* reset 随机化如何影响任务难度；
* 随机策略在该任务中的表现如何。

通过本阶段整理，可以为后续训练机械臂操作策略、设计手工基线或接入强化学习算法提供任务理解基础。

---

## 2. 环境配置

本实验使用 robosuite 中的 Lift 任务，机械臂型号为 Panda。

主要环境配置如下：

```python
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)
```

其中：

* `env_name="Lift"` 表示任务为抓取并抬起方块；
* `robots="Panda"` 表示使用 Franka Panda 七自由度机械臂；
* `use_camera_obs=False` 表示当前不使用图像观测，而使用低维状态观测；
* `reward_shaping=True` 表示使用更连续的 shaped reward，便于观察任务进展；
* `control_freq=20` 表示控制频率为 20 Hz。

---

## 3. Observation 分析

Panda Lift 环境返回的 observation 是一个字典，包含机械臂状态、末端执行器状态、物体状态以及相对位置信息。

常见 observation 字段包括：

| 字段                       | 含义                 |
| ------------------------ | ------------------ |
| `robot0_joint_pos`       | Panda 机械臂 7 个关节角   |
| `robot0_joint_vel`       | Panda 机械臂 7 个关节速度  |
| `robot0_eef_pos`         | 机械臂末端执行器在世界坐标系下的位置 |
| `robot0_eef_quat`        | 机械臂末端执行器姿态四元数      |
| `cube_pos`               | 方块在世界坐标系下的位置       |
| `cube_quat`              | 方块姿态四元数            |
| `gripper_to_cube_pos`    | 夹爪相对于方块的位置关系       |
| `cube_to_robot0_eef_pos` | 方块相对于机械臂末端的位置关系    |

这些 observation 字段说明，Panda Lift 任务并不是让策略直接“知道应该怎么抓”，而是给出机械臂、夹爪和方块的状态信息，由策略根据这些信息学习或设计动作。

其中，比较关键的是：

* `robot0_eef_pos`：用于判断末端执行器当前在哪里；
* `cube_pos`：用于判断目标物体在哪里；
* `cube_to_robot0_eef_pos` 或类似相对位置字段：用于判断末端执行器和方块之间的距离与方向；
* `robot0_joint_pos` 和 `robot0_joint_vel`：用于描述机械臂自身状态。

对于后续强化学习策略来说，这些低维状态可以作为 policy network 的输入。

---

## 4. Action 分析

本实验中打印得到的 action spec 为：

```text
action low: [-1. -1. -1. -1. -1. -1. -1.]
action high: [1. 1. 1. 1. 1. 1. 1.]
action dimension: 7
```

说明当前 Panda Lift 环境的动作空间是 7 维连续动作，每个维度的取值范围为 `[-1, 1]`。

在默认控制配置下，7 维 action 通常可以理解为：

| 维度        | 含义               |
| --------- | ---------------- |
| action[0] | 末端执行器在 x 方向的运动控制 |
| action[1] | 末端执行器在 y 方向的运动控制 |
| action[2] | 末端执行器在 z 方向的运动控制 |
| action[3] | 末端执行器姿态控制相关维度    |
| action[4] | 末端执行器姿态控制相关维度    |
| action[5] | 末端执行器姿态控制相关维度    |
| action[6] | 夹爪开合控制           |

从任务角度看，策略需要同时完成以下动作阶段：

1. 将机械臂末端移动到方块附近；
2. 调整夹爪与方块的相对位置；
3. 控制夹爪张开或闭合；
4. 抓住方块；
5. 向上抬起方块；
6. 达到任务成功高度。

因此，虽然 action 只有 7 维，但它同时涉及末端位置、姿态和夹爪控制。随机动作很难自然完成这一系列有顺序的操作。

---

## 5. Reward 分析

本实验启用了：

```python
reward_shaping=True
```

这表示环境会返回 shaped reward。相比只在任务成功时给奖励，shaped reward 会在接近物体、抓取物体、抬起物体等中间阶段给予更连续的反馈。

从实验结果看，随机策略虽然没有成功完成任务，但不同 episode 的累计 reward 存在明显差异：

```text
Episode 1: total_reward=1.157
Episode 2: total_reward=0.371
Episode 3: total_reward=13.196
Episode 4: total_reward=3.434
Episode 5: total_reward=0.594
```

其中 Episode 3 的累计 reward 明显高于其他回合，说明随机动作在某些情况下可能偶然让末端靠近方块或产生一定任务进展，但仍然没有完成成功抓取和抬升。

这说明 shaped reward 能反映任务过程中的部分进展，但仅靠随机策略无法稳定完成任务。

---

## 6. Done 与 Success 分析

本次随机 baseline 运行 5 个 episode，每个 episode 最大步数为 200。

实验输出如下：

```text
Episode 1: length=200, total_reward=1.157, done=False, success=False
Episode 2: length=200, total_reward=0.371, done=False, success=False
Episode 3: length=200, total_reward=13.196, done=False, success=False
Episode 4: length=200, total_reward=3.434, done=False, success=False
Episode 5: length=200, total_reward=0.594, done=False, success=False
```

可以看到：

* 所有 episode 的长度都是 200；
* 所有 episode 的 `done=False`；
* 所有 episode 的 `success=False`。

这说明在当前设置下，环境没有因为随机策略完成任务而提前终止。episode 结束主要由外部设置的 `max_steps=200` 控制，而不是任务自然成功终止。

`success=False` 表明随机策略没有成功完成 Lift 任务，也就是没有稳定实现抓取并抬起方块。

---

## 7. Reset 随机化分析

Panda Lift 环境在每次 reset 时会重新初始化机械臂和方块状态。方块初始位置通常会存在一定随机变化。

reset 随机化的意义在于：

* 防止策略只记住一个固定初始位置；
* 要求策略根据 observation 判断当前方块位置；
* 提高策略对不同初始状态的泛化能力；
* 更接近真实机器人操作任务中的不确定性。

对于机械臂操作强化学习来说，reset 随机化非常重要。因为真实环境中，物体位置、姿态、机械臂初始状态往往不会完全固定。如果策略只能在固定位置成功，就不具备实际应用价值。

---

## 8. 随机策略 Baseline 实验

本阶段实现了随机策略 baseline，核心动作生成方式如下：

```python
low, high = env.action_spec
action = np.random.uniform(low, high)
```

该策略在每一步从合法动作范围 `[-1, 1]` 内随机采样 7 维动作。

实验设置如下：

| 项目              | 数值                |
| --------------- | ----------------- |
| episode 数量      | 5                 |
| 每个 episode 最大步数 | 200               |
| action 维度       | 7                 |
| action 范围       | [-1, 1]           |
| reward 类型       | shaped reward     |
| 是否使用图像观测        | 否                 |
| 是否保存视频          | 使用任务 A 已有随机动作演示视频 |

实验结果如下：

| Episode | Length | Total Reward | Done  | Success |
| ------- | -----: | -----------: | ----- | ------- |
| 1       |    200 |        1.157 | False | False   |
| 2       |    200 |        0.371 | False | False   |
| 3       |    200 |       13.196 | False | False   |
| 4       |    200 |        3.434 | False | False   |
| 5       |    200 |        0.594 | False | False   |

汇总结果如下：

```text
average reward: 3.750263479033884
success rate: 0.0
average episode length: 200.0
```

已有演示视频：

```text
assets/week03_panda_lift_random_demo.mp4
```

---

## 9. 实验结论

通过本阶段实验可以得到以下结论：

1. Panda Lift 是一个典型的机械臂操作任务，策略需要根据机械臂状态、末端位置、物体位置和相对位置信息生成连续动作。

2. 当前动作空间为 7 维连续动作，包含末端位置控制、姿态控制和夹爪控制。虽然动作维度不高，但任务本身具有明显的阶段性和时序性。

3. 随机策略在 5 个 episode 中成功率为 0，说明该任务无法依靠无目标随机动作完成。

4. shaped reward 能够反映部分中间任务进展。例如某些随机 episode 的累计 reward 明显更高，可能说明末端偶然接近物体或产生了一定有效动作。

5. reset 随机化使得方块初始位置不固定，要求策略必须利用 observation 进行闭环控制，而不是依赖固定动作序列。

6. 本阶段完成了从“能运行 Panda Lift 环境”到“能分析任务接口并建立 baseline”的过渡，为后续接入手工策略、强化学习训练或项目展示打下基础。

---

## 10. 阶段成果

本阶段完成内容包括：

* 编写 `src/analyze_panda_lift_task.py`，分析 Panda Lift 的 observation、action、reward、done、reset；
* 编写 `src/run_panda_lift_random_baseline.py`，运行随机策略 baseline；
* 记录 5 个 episode 的 reward、done 和 success 情况；
* 沿用任务 A 中录制的随机动作演示视频；
* 整理 Panda Lift 任务机制和 baseline 结果；
* 为后续机械臂操作强化学习实验和作品集展示准备材料。

当前阶段产出文件包括：

```text
src/analyze_panda_lift_task.py
src/run_panda_lift_random_baseline.py
assets/week03_panda_lift_random_demo.mp4
notes/week03_panda_lift_baseline_analysis.md
```
