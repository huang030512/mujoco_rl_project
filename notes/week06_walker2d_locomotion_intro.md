# Week 6 - MuJoCo Locomotion 入门：Walker2d / Humanoid

## 1. 任务概述

本节主要记录 MuJoCo locomotion 任务的入门理解，并以 `Walker2d-v5` 为主要实验对象，同时对比 `Humanoid-v5`。

Walker2d 是一个二维双足行走任务。它的目标是让一个简化的双足机器人在不摔倒的前提下尽量向前移动。和 Panda Lift 这类机械臂操作任务不同，Walker2d 不关注末端执行器位置或物体抓取，而是关注动态平衡、足地接触、多关节协同和关节力矩控制。

一个 MuJoCo locomotion 任务可以从五个方面理解：

* observation：策略能看到什么
* action：策略能控制什么
* reward：环境鼓励什么行为
* termination：什么情况算失败
* control difficulty：为什么这个任务难

---

## 2. Observation：策略看到什么

Walker2d 的 observation 是低维本体状态，主要由 `qpos` 和 `qvel` 组成。

* `qpos`：身体位置、身体姿态、关节角度
* `qvel`：身体速度、角速度、关节速度

本次环境检查中，Walker2d 的 observation space 为：

```text
Box(-inf, inf, (17,), float64)
```

这说明策略每一步接收到的是一个 17 维连续状态向量。

这些状态大致包括：

* 躯干高度
* 躯干角度
* 左右腿关节角
* 身体速度
* 关节速度

对于 locomotion 来说，只知道位置和姿态是不够的，还必须知道速度。因为机器人当前看起来可能还没有倒，但如果速度方向和角速度已经很危险，下一步就可能摔倒。

这和之前做过的单关节 PPO 任务是一致的。单关节任务的 observation 是：

```text
[theta, theta_dot]
```

而 Walker2d 可以理解成多关节版本：

```text
[身体姿态, 多个关节角, 身体速度, 多个关节速度]
```

因此，Walker2d 的 observation 本质上是机器人自己的“本体感觉”。

---

## 3. Action：策略控制什么

Walker2d 的 action space 为：

```text
Box(-1.0, 1.0, (6,), float32)
```

这说明策略每一步输出 6 维连续动作。

这 6 维动作可以理解为两条腿主要关节的控制输入，概念上对应腿部关节的力矩控制。

策略并不是直接输出：

```text
左脚往前迈一步
身体保持直立
向前走
```

而是输出：

```text
6 个关节当前应该施加多大的控制量
```

然后 MuJoCo 根据刚体动力学、关节约束和足地接触推进仿真，最终形成身体运动。

这和 Panda Lift 不一样。Panda Lift 中的 action 更接近末端位置、姿态和夹爪控制；而 Walker2d 更接近底层动力学控制。

需要注意的是，`env.action_space.sample()` 生成的是 6 维归一化动作，范围在 `[-1, 1]`。这些值会被环境送入 MuJoCo actuator 中，不一定等同于真实物理世界里的 `-1 Nm` 到 `1 Nm`，但可以从概念上理解为关节力矩控制输入。

---

## 4. Reward：环境鼓励什么行为

Walker2d 的 reward 可以理解成三部分：

```text
total_reward = reward_survive + reward_forward + reward_ctrl
```

其中：

* `reward_survive`：鼓励机器人保持健康状态，也就是不要摔倒
* `reward_forward`：鼓励机器人沿 x 方向向前移动
* `reward_ctrl`：惩罚过大的控制动作，避免动作太猛或能耗太高

本次 random rollout 中，第 0 步的输出为：

```text
step 0:
total_reward   = 0.890125
reward_forward = -0.108786
reward_ctrl    = -0.001089
reward_survive = 1.000000
```

可以验证：

```text
0.890125 = 1.000000 + (-0.108786) + (-0.001089)
```

这说明 total reward 不是一个黑盒数字，而是可以由 reward 分项解释出来。

Walker2d reward 的本质是：

```text
站稳 + 向前走 - 过度用力
```

如果只鼓励向前速度，机器人可能会猛冲、乱跳甚至很快摔倒。如果只鼓励稳定，机器人可能站着不动。如果只惩罚动作，机器人可能什么都不做。

所以 locomotion reward 的核心是平衡：

```text
稳定性、前进速度、控制代价
```

---

## 5. Termination：什么情况算失败

Walker2d 中，episode 会在机器人进入 unhealthy 状态时结束。最常见的情况是机器人摔倒。

环境通常会根据以下因素判断机器人是否 unhealthy：

* 躯干高度是否超出健康范围
* 躯干角度是否超出健康范围
* 仿真状态是否出现 NaN 或 inf 等异常值

Gymnasium 中需要区分两个信号：

```text
terminated：任务失败或进入终止状态
truncated：达到最大时间步限制
```

在 Walker2d 中：

```text
terminated=True
```

通常表示机器人已经摔倒或失去健康状态。

而：

```text
truncated=True
```

表示 episode 没有失败，只是时间到了。

本次 random rollout 的结果是：

```text
terminated: True
truncated: False
```

说明这次 episode 是因为机器人失去健康状态而结束，不是因为时间到。

---

## 6. Random Rollout Baseline

在训练 PPO 或 SAC 之前，我先做了 Walker2d 的 random rollout baseline。

随机策略使用：

```python
env.action_space.sample()
```

每一步随机生成一个 6 维动作。

本次实验结果为：

```text
episode total reward: -0.216279
episode length: 16
final x_position: -0.120082
terminated: True
truncated: False
```

这说明随机策略：

* 只存活了 16 步
* 最终向后移动了约 0.12 米
* 多数 step 的 `x_velocity` 为负
* `reward_forward` 因为后退而为负
* 最终因为机器人 unhealthy 而 terminated

这也说明随机动作并不能产生稳定步态。Walker2d 不是随机给腿部关节发力就能走起来的任务，它需要策略学习如何协调多关节动作、利用地面接触并保持动态平衡。

random baseline 的意义在于提供后续训练的对照组。后续如果训练 PPO，应该和 random baseline 对比：

* episode length 是否变长
* total reward 是否提高
* x_velocity 是否变为正
* final x_position 是否向前增加
* 是否不再早早 terminated

---

## 7. Locomotion 控制为什么难

Walker2d / Humanoid locomotion 的难点不在于代码复杂，而在于控制问题本身复杂。

### 7.1 浮动基座

固定底座机械臂的 base 是固定的，而 Walker2d / Humanoid 的身体是浮动的。

机器人没有一个电机可以直接控制：

```text
身体不要倒
```

它只能通过腿部关节动作和足地接触产生地面反作用力，间接控制身体姿态和质心运动。

### 7.2 欠驱动

欠驱动指的是系统需要控制的自由度多于可以直接控制的输入。

Walker2d 可以直接控制的是腿部关节动作，但它真正需要实现的是：

* 身体不摔倒
* 躯干姿态稳定
* 质心向前移动
* 脚落在合适位置

这些目标都不是直接控制量，而是通过关节力矩和接触力间接实现的。

### 7.3 接触切换

走路过程中会不断发生足地接触切换：

```text
左脚支撑
右脚摆动
右脚落地
左脚摆动
```

脚接地和离地会改变系统动力学。脚落地时还会产生冲击，因此 locomotion 是一个包含连续运动和离散接触事件的混合动力系统问题。

### 7.4 动态平衡

双足行走不是静态站稳，而是在运动中保持平衡。

机器人需要在身体持续移动的过程中不断调整腿部动作，既不能太保守导致原地不动，也不能太激进导致摔倒。

### 7.5 多关节耦合

一个关节动作不会只影响一个关节。

例如膝关节力矩可能会影响：

* 小腿运动
* 脚和地面的接触
* 躯干高度
* 身体角速度
* 前进速度
* 另一条腿的补偿动作

因此，locomotion 是强耦合的全身动力学控制问题。

---

## 8. Walker2d 和 Humanoid 的区别

Walker2d 是二维简化双足机器人，Humanoid 是三维高维人形机器人。

| 对比项            | Walker2d | Humanoid |
| -------------- | -------: | -------: |
| 空间维度           |       2D |       3D |
| action 维度      |        6 |       17 |
| observation 维度 |       17 |      348 |
| 是否有手臂          |        无 |        有 |
| 平衡难度           |       中等 |        高 |
| 接触复杂度          |       中等 |        高 |

Humanoid 比 Walker2d 更难，主要原因是：

* 需要处理三维空间中的前后、左右和旋转稳定性
* action 维度更高，需要协调更多关节
* observation 维度更高，包含更多动力学和接触信息
* 手臂和躯干也会影响整体角动量和平衡
* 足地接触和身体碰撞更加复杂

不过 Humanoid 仍然是仿真 benchmark，不等于真实人形机器人控制。真实人形机器人还需要考虑：

* 电机延迟
* 关节摩擦
* IMU 噪声
* 足底接触误差
* 地面不平
* 执行器饱和
* 安全约束
* sim-to-real 迁移

---

## 9. 本节总结

通过本节任务，我完成了 Walker2d locomotion 的入门理解和 random baseline 实验。

关键结论如下：

* Walker2d 是二维双足行走任务，目标是不摔倒并尽量向前移动。
* observation 是 17 维本体状态，主要包含姿态、关节角、速度和关节速度。
* action 是 6 维连续控制输入，可以理解为腿部关节的力矩控制。
* reward 由 `reward_survive`、`reward_forward` 和 `reward_ctrl` 组成。
* termination 主要表示机器人摔倒或进入 unhealthy 状态。
* random policy 无法形成稳定步态，本次只存活 16 步并最终 terminated。
* random rollout 是后续 PPO / SAC 训练的重要 baseline。
* locomotion 难在浮动基座、欠驱动、接触切换、动态平衡和多关节耦合。
* Humanoid 是 Walker2d 的 3D 高维版本，更接近人形机器人控制问题，但真实部署还需要处理 sim-to-real 和安全问题。

本节为后续训练 Walker2d PPO policy、分析学习曲线和比较 random baseline 打下了基础。
