# Week 06 Task A：PPO / SAC / BC / DAgger 原理面试包

## 1. 本周任务目标

本任务面向机器人深度强化学习岗位面试，重点整理以下内容：

* BC：Behavior Cloning，行为克隆
* DAgger：Dataset Aggregation，数据集聚合
* PPO：Proximal Policy Optimization
* SAC：Soft Actor-Critic
* On-policy / Off-policy
* Model-free / Model-based
* 这些概念与当前 MuJoCo / robosuite / Panda Lift / BC 项目的关系

本任务暂不写代码，重点是建立算法理解框架，并能把概念和自己的项目结合起来，为后续 DAgger、PPO、SAC 实验做准备。

---

## 2. 机器人强化学习基本闭环

机器人强化学习任务可以理解为一个连续决策闭环：

```text
Observation → Policy → Action → Environment → Reward → Update
```

也就是：

1. 机器人从环境中获得 observation；
2. policy 根据 observation 输出 action；
3. 环境执行 action，状态发生变化；
4. 环境返回 reward 和下一个 observation；
5. 算法根据这些交互数据更新 policy。

在当前 Panda Lift 项目中，各个概念对应如下：

| RL 概念       | Panda Lift 项目中的对应                          |
| ----------- | ------------------------------------------ |
| Environment | MuJoCo / robosuite 中的 Panda Lift 环境        |
| Observation | 机械臂状态、末端位置、夹爪状态、方块位置等                      |
| Action      | Panda 机械臂末端控制和夹爪开合                         |
| Reward      | 接近方块、抓取、抬升、success 等信号                     |
| Policy      | BC / PPO / SAC 训练出的策略网络                    |
| Episode     | 一次完整的抓取抬升尝试                                |
| Trajectory  | 一个 episode 中的 observation-action-reward 序列 |

强化学习的目标不是学习一个固定的单步控制公式，而是在连续决策过程中学习一个策略，使长期累计奖励最大。

---

## 3. Behavior Cloning：BC

### 3.1 BC 是什么

BC 是 Behavior Cloning，中文可以理解为行为克隆。

BC 的本质是 supervised imitation learning，也就是监督式模仿学习。

它不需要 reward，也不需要机器人自己大量试错，而是直接学习 expert demonstration 中的 observation-action mapping。

也就是说，专家在某个 observation 下做了某个 action，BC policy 就学习在类似 observation 下输出类似的 action。

可以表示为：

```text
observation → expert action
```

训练目标是让 policy 输出的 action 尽量接近 expert action。

---

### 3.2 BC 和当前 Panda Lift 项目的关系

在当前 Panda Lift 项目中，BC 的流程是：

```text
handcrafted teacher 生成 demonstration data
↓
保存 observation-action pairs
↓
训练 BC policy
↓
rollout evaluation
↓
统计 success / max_lift / final_lift / min_eef_cube_dist
```

也就是说，当前项目中的 BC policy 学习的是：

```text
Panda Lift observation → teacher action
```

其中 observation 包括机械臂状态、末端位置、夹爪状态、方块位置等信息；action 对应末端运动控制和夹爪开合控制。

---

### 3.3 BC 的优点

BC 的主要优点包括：

* 实现简单；
* 训练稳定；
* 不需要复杂 reward design；
* 不需要一开始就让机器人随机探索；
* 适合机器人策略冷启动；
* 可以利用 scripted teacher、遥操作数据或专家轨迹。

在机器人任务中，完全依靠强化学习随机探索通常很难。比如 Panda Lift 中，如果机械臂一开始完全随机运动，可能很久都碰不到方块，更不用说夹取和抬升。因此 BC 可以先让 policy 学会一个基本行为。

---

### 3.4 BC 的缺点

BC 的主要问题是：

```text
training loss 低，不一定代表 rollout success 高。
```

原因是 BC 只学习 expert demonstration 中出现过的状态。如果 policy 自己执行时进入了 expert 数据中没有覆盖的状态，就可能不知道如何恢复。

BC 的核心缺点包括：

* 对 expert 数据质量敏感；
* 容易出现 distribution shift；
* closed-loop 执行中误差会累积；
* policy 可能在训练集上拟合很好，但在真实 rollout 中表现不好。

---

## 4. Distribution Shift

### 4.1 Distribution Shift 是什么

Distribution shift 是 BC 中最核心的问题之一。

它指的是：

```text
训练时 policy 看到的是 expert 会访问的状态；
测试时 policy 看到的是自己执行动作后造成的状态。
```

这两种状态分布可能不同。

训练时的数据来自 expert，所以大部分状态都比较“正常”。但 policy 自己 rollout 时，只要前几步动作稍微有偏差，就可能进入 expert 数据中很少出现的状态。

---

### 4.2 Panda Lift 中的 Distribution Shift

在 Panda Lift 中，teacher 数据中可能大多是这样的状态：

* 夹爪稳定靠近方块；
* 末端位置比较合理；
* 方块没有被碰歪；
* 夹爪闭合时机比较正确；
* 抬升动作发生在夹住方块之后。

但是 BC policy 自己 rollout 时，可能出现：

* 夹爪偏离方块；
* 下降位置不准确；
* 闭合夹爪太早或太晚；
* 方块被碰歪；
* 没夹住方块却开始上抬；
* 进入 teacher 数据中没有覆盖的异常状态。

这些由 learner 自己造成的状态可以称为 learner-induced states。

如果训练数据中没有这些状态，BC policy 就不知道如何恢复。

---

### 4.3 为什么 BC loss 低但 rollout 可能失败

BC 的训练 loss 只衡量：

```text
在 expert 数据分布上，policy action 和 expert action 是否接近。
```

但 rollout success 衡量的是：

```text
policy 自己连续执行时，是否真的能完成任务。
```

这两个指标不完全等价。

在 closed-loop control 中，每一步 action 都会影响下一步 observation。如果某一步 action 有小误差，这个误差会影响后续状态，后续状态又会影响后续动作，最终可能导致误差逐步累积。

因此：

```text
BC training loss 低 ≠ policy rollout success 高
```

---

### 4.4 面试表达

如果面试官问：

> 为什么 BC 在机器人任务中容易失败？

可以回答：

> BC 的核心问题是 distribution shift。训练时 policy 看到的是 expert demonstration 中的状态分布，但在 closed-loop rollout 中，policy 的小误差会逐步累积，使机器人进入 expert 数据中没有覆盖的状态。此时 policy 没有学过如何恢复，因此即使 training loss 很低，rollout success rate 也可能不高。在我的 Panda Lift 项目中，这可能表现为夹爪偏离方块、闭合时机错误、方块被碰歪，或者没有夹住方块却继续上抬。

---

## 5. DAgger

### 5.1 DAgger 是什么

DAgger 是 Dataset Aggregation，中文可以理解为数据集聚合。

DAgger 是一种交互式模仿学习方法，主要用于缓解 BC 的 distribution shift 问题。

它的核心思想是：

```text
不要只在 expert states 上训练；
还要让 learner 自己 rollout；
收集 learner-induced states；
再让 expert 给这些 states 标注 corrective actions。
```

也就是说，DAgger 不只是学习 expert 的标准轨迹，还学习 policy 自己跑偏之后应该如何纠正。

---

### 5.2 DAgger 的流程

DAgger 的基本流程如下：

1. 使用 expert demonstrations 训练一个初始 BC policy；
2. 让当前 policy 在环境中 rollout；
3. 收集当前 policy 实际访问到的 states；
4. 让 expert 对这些 states 标注正确 action；
5. 将新数据加入原始 dataset；
6. 用聚合后的 dataset 重新训练 policy；
7. 重复多轮。

可以表示为：

```text
Initial expert dataset
↓
Train BC policy
↓
Policy rollout
↓
Collect learner-induced states
↓
Expert relabeling
↓
Aggregate dataset
↓
Retrain policy
```

---

### 5.3 DAgger 和 Panda Lift 项目的关系

在当前 Panda Lift 项目中，后续如果实现 DAgger，可以这样做：

```text
BC policy rollout
↓
收集失败或偏离状态
↓
handcrafted teacher 对这些 states 重新标注 action
↓
加入原始 dataset
↓
训练 DAgger-BC policy
↓
和原始 BC policy 对比 success rate
```

可以重点收集以下状态：

* 夹爪偏离方块；
* 夹爪接近但未对准方块；
* 夹爪闭合失败；
* 方块被碰歪；
* 未夹住方块但 policy 已经开始上抬；
* max_lift 很低但 min_eef_cube_dist 已经较小的失败状态。

---

### 5.4 当前项目为什么适合做 DAgger

当前项目已经具备 DAgger 的前置条件：

| DAgger 需要的条件                 | 当前项目状态 |
| ---------------------------- | ------ |
| Panda Lift 环境                | 已有     |
| handcrafted teacher          | 已有     |
| BC policy                    | 已有     |
| rollout evaluation           | 已有     |
| success / lift / distance 诊断 | 已有     |
| demonstration dataset        | 已有     |

因此，DAgger 是当前 BC 项目自然的下一步。

---

### 5.5 DAgger 的优点和缺点

DAgger 的优点：

* 能缓解 BC 的 distribution shift；
* 能学习偏离后的恢复动作；
* 更适合 closed-loop robot control；
* 可以提升 policy rollout success。

DAgger 的缺点：

* 需要 expert 持续标注；
* 如果是真实机器人，learner rollout 可能有安全风险；
* 如果 expert 本身不够强，DAgger 数据质量也会受影响；
* 如果 teacher 无法处理 learner 造成的异常状态，DAgger 效果有限。

---

### 5.6 面试表达

如果面试官问：

> DAgger 相比 BC 改进在哪里？

可以回答：

> DAgger 主要用于缓解 BC 的 distribution shift。BC 只在 expert states 上训练，因此 policy 自己 rollout 时容易因为误差累积进入未见状态。DAgger 让当前 policy 自己 rollout，收集 learner-induced states，然后让 expert 对这些 states 标注 corrective actions，并聚合进训练集。这样 policy 不仅学习 expert 的标准轨迹，也学习偏离后的恢复动作。在 Panda Lift 中，这可以对应收集夹爪偏离方块、闭合失败、方块被碰歪等状态，再用 handcrafted teacher 重新标注动作。

---

## 6. On-policy vs Off-policy

### 6.1 这个分类在问什么

On-policy / Off-policy 关注的是：

```text
训练数据是否必须来自当前 policy。
```

也就是说，算法能不能复用旧数据。

---

### 6.2 On-policy

On-policy 方法使用当前 policy 刚刚采集的数据来更新当前 policy。

典型算法包括：

* PPO
* TRPO
* A2C
* A3C

On-policy 的训练流程大致是：

```text
当前 policy rollout
↓
收集一批新数据
↓
用这批数据更新当前 policy
↓
旧数据基本不再反复使用
↓
新 policy 再重新采样
```

On-policy 的优点：

* 训练相对稳定；
* 理论直观；
* 适合作为 baseline；
* policy update 和数据分布比较一致。

On-policy 的缺点：

* 样本效率较低；
* 每次更新都需要重新 rollout；
* 真实机器人上采样成本高；
* 不适合频繁丢弃昂贵的真实交互数据。

PPO 是典型的 on-policy 方法。

---

### 6.3 Off-policy

Off-policy 方法可以使用不是当前 policy 采集的数据。

这些数据可以来自：

* 当前 policy；
* 旧 policy；
* 随机策略；
* expert policy；
* replay buffer 中的历史数据。

典型算法包括：

* SAC
* DDPG
* TD3
* DQN

Off-policy 方法通常会使用 replay buffer 存储历史 transition。

transition 的形式通常是：

```text
(s, a, r, s', done)
```

Off-policy 的优点：

* 可以复用历史数据；
* 样本效率更高；
* 更适合采样昂贵的机器人任务；
* 可以利用 replay buffer 进行多次训练。

Off-policy 的缺点：

* 实现更复杂；
* 训练稳定性依赖 Q function、target network、replay buffer 等设计；
* 如果数据分布质量不好，训练也可能不稳定。

SAC 是典型的 off-policy 方法。

---

### 6.4 On-policy 和 Off-policy 的核心区别

| 维度       | On-policy         | Off-policy                       |
| -------- | ----------------- | -------------------------------- |
| 数据来源     | 当前 policy 新采集的数据  | 当前 policy、旧 policy、其他 policy 的数据 |
| 数据复用     | 较少复用              | 可以反复复用                           |
| 样本效率     | 较低                | 较高                               |
| 代表算法     | PPO、TRPO、A2C      | SAC、TD3、DDPG、DQN                 |
| 机器人任务适用性 | 仿真中常用，稳定 baseline | 更适合采样昂贵的连续控制任务                   |

---

### 6.5 面试表达

如果面试官问：

> On-policy 和 Off-policy 的区别是什么？

可以回答：

> On-policy 方法使用当前 policy 采集的数据来更新当前 policy，因此数据分布和当前策略比较一致，训练相对稳定，但样本效率较低。Off-policy 方法可以利用旧 policy、随机策略或 replay buffer 中的数据进行训练，因此可以复用历史经验，样本效率更高。PPO 是典型 on-policy 方法，而 SAC 是典型 off-policy 方法。对于机器人任务，off-policy 方法通常更有吸引力，因为真实机器人采样成本比较高。

---

## 7. PPO

### 7.1 PPO 是什么

PPO 是 Proximal Policy Optimization。

PPO 是一种常用的深度强化学习算法，尤其常作为连续控制任务中的稳定 baseline。

PPO 的标签可以总结为：

```text
On-policy
Model-free
Actor-Critic
Policy Gradient
```

---

### 7.2 PPO 的核心思想

PPO 的核心思想是：

```text
让 policy 变好，但每次不要更新太猛。
```

在强化学习中，如果 policy 更新幅度过大，新的 policy 可能突然变得很差，导致训练崩掉。

比如在 Panda Lift 中，上一轮 policy 还能够接近方块，但如果更新太激进，下一轮 policy 可能突然输出很大的动作，导致机械臂乱动。

PPO 通过 clipping 限制新旧 policy 之间的差距，从而提高训练稳定性。

---

### 7.3 PPO 中的 Actor 和 Critic

PPO 通常使用 Actor-Critic 架构。

| 组件        | 作用                                             |
| --------- | ---------------------------------------------- |
| Actor     | 根据 observation 输出 action 或 action distribution |
| Critic    | 估计当前 state 的 value                             |
| Advantage | 判断某个 action 比预期更好还是更差                          |
| Clipping  | 限制 policy update 幅度                            |

Actor 负责决定：

```text
当前状态下应该做什么动作。
```

Critic 负责估计：

```text
当前状态大概有多好。
```

Advantage 可以理解为：

```text
实际结果相对于 Critic 预期的好坏。
```

如果某个 action 带来的结果比预期好，就提高该 action 的概率；如果比预期差，就降低该 action 的概率。

---

### 7.4 PPO 为什么稳定

PPO 会比较新旧策略对同一个 action 的概率变化。

如果新 policy 相比旧 policy 改动太大，PPO 会通过 clipping 限制这个变化。

直觉上：

```text
policy 可以变好，但不能一步变化太大。
```

这样可以避免一次更新让机器人策略突然崩掉。

---

### 7.5 PPO 的优点和缺点

PPO 的优点：

* 训练稳定；
* 实现相对清晰；
* 适合作为 RL baseline；
* 在很多仿真控制任务中表现可靠；
* 对超参数相对不那么敏感。

PPO 的缺点：

* on-policy，样本效率较低；
* 每轮都需要当前 policy 重新 rollout；
* 真实机器人上采样成本高；
* 对稀疏奖励任务可能效率较低。

---

### 7.6 PPO 和 Panda Lift 项目的关系

如果在 Panda Lift 中使用 PPO，流程大致是：

```text
初始化 policy
↓
让 Panda 在 MuJoCo 中尝试 Lift
↓
收集当前 policy 的 rollout 数据
↓
根据 reward 计算 return 和 advantage
↓
更新 actor 和 critic
↓
继续采样和训练
```

PPO 和 BC 的区别是：

| 方法  | 是否需要 teacher | 是否需要 reward | 数据来源                 |
| --- | ------------ | ----------- | -------------------- |
| BC  | 需要           | 不需要         | expert demonstration |
| PPO | 不需要          | 需要          | 当前 policy rollout    |

PPO 可以作为当前 Panda Lift 项目后续的 on-policy RL baseline。

---

### 7.7 PPO 面试表达

如果面试官问：

> PPO 为什么训练比较稳定？

可以回答：

> PPO 是一种 on-policy model-free actor-critic 方法。它的核心思想是限制每次 policy update 的幅度，避免新 policy 相比旧 policy 变化太大。PPO 通过 clipping 约束新旧策略的概率比，从而防止一次更新让策略崩掉。对于机器人连续控制任务，这种保守更新比较稳定，因此 PPO 常被用作 baseline。不过 PPO 是 on-policy 方法，每次更新都需要当前 policy 新采样数据，所以样本效率相对较低。

---

## 8. SAC

### 8.1 SAC 是什么

SAC 是 Soft Actor-Critic。

SAC 是一种非常常见的机器人连续控制算法。

SAC 的标签可以总结为：

```text
Off-policy
Model-free
Actor-Critic
Maximum Entropy RL
```

---

### 8.2 SAC 的核心思想

SAC 的核心思想是：

```text
不仅要让 reward 高，还要让 policy 保持探索能力。
```

SAC 优化的不只是 reward，还包括 entropy。

Entropy 可以理解为策略的随机性。策略 entropy 越高，说明 policy 越愿意尝试不同动作。

SAC 不希望 policy 太早变得确定，因为在机器人任务中，如果太早收敛到错误动作，后面就很难探索到真正有效的行为。

---

### 8.3 为什么机器人任务需要探索

在 Panda Lift 中，如果 policy 一开始完全不知道怎么抓方块，它可能会学到一些错误行为，例如：

* 一直往上抬；
* 一直接近错误方向；
* 夹爪闭合太早；
* 夹爪不靠近方块就闭合；
* 只学会接近但不会抬升。

如果策略过早变得确定，就可能卡在这些局部行为中。

SAC 通过 entropy regularization 鼓励策略保持一定随机性，从而增加探索机会。

---

### 8.4 SAC 的主要组件

SAC 通常包含以下组件：

| 组件                  | 作用                      |
| ------------------- | ----------------------- |
| Actor               | 输出连续动作                  |
| Q network           | 估计 state-action value   |
| Target Q network    | 稳定 Q-learning 目标        |
| Replay buffer       | 存储历史 transition         |
| Entropy coefficient | 平衡 reward 和 exploration |

Replay buffer 中存储的数据通常是：

```text
(s, a, r, s', done)
```

SAC 会从 replay buffer 中反复采样数据，用于更新 Q network 和 policy。

---

### 8.5 SAC 为什么样本效率高

SAC 是 off-policy 方法，所以它可以复用历史数据。

相比 PPO 每次主要使用当前 policy 新采集的数据，SAC 可以把过去的经验存进 replay buffer，然后反复使用这些数据训练网络。

因此 SAC 的样本效率通常比 PPO 更高。

这对机器人任务很重要，因为机器人采样成本高，尤其是真实机器人不能无限随机试错。

---

### 8.6 SAC 的优点和缺点

SAC 的优点：

* 适合连续动作空间；
* off-policy，样本效率高；
* replay buffer 可以复用历史经验；
* entropy regularization 有利于探索；
* 在 manipulation 和 locomotion 中常作为强 baseline。

SAC 的缺点：

* 实现比 PPO 更复杂；
* Q network 训练可能不稳定；
* 对 reward scale、buffer 数据分布、entropy coefficient 等比较敏感；
* 真实机器人上仍然需要考虑安全探索。

---

### 8.7 SAC 和 Panda Lift 项目的关系

如果在 Panda Lift 中使用 SAC，流程大致是：

```text
policy 和 MuJoCo 环境交互
↓
将 transition 存入 replay buffer
↓
从 buffer 中采样 batch
↓
更新 Q network
↓
更新 actor policy
↓
继续交互和训练
```

SAC 可以作为当前 Panda Lift 项目后续的 off-policy continuous control baseline。

---

### 8.8 SAC 面试表达

如果面试官问：

> 为什么 SAC 适合机器人连续控制？

可以回答：

> SAC 是一种 off-policy model-free actor-critic 方法，适合连续动作空间。它可以利用 replay buffer 复用历史交互数据，因此样本效率比 on-policy 方法更高。同时，SAC 使用 maximum entropy objective，通过 entropy regularization 鼓励探索，避免 policy 太早收敛到局部行为。机器人 manipulation 任务通常动作连续、探索困难、采样昂贵，所以 SAC 是常见的强 baseline。

---

## 9. Model-free vs Model-based

### 9.1 这个分类在问什么

Model-free / Model-based 关注的是：

```text
算法是否显式学习或使用环境动力学模型。
```

这里的动力学模型指的是：

```text
给定当前 state 和 action，预测 next state。
```

也就是：

```text
s, a → s'
```

---

### 9.2 Model-free

Model-free RL 不显式学习或使用 dynamics model。

它不直接预测：

```text
如果我在当前状态执行这个动作，下一步状态会是什么。
```

而是通过环境交互学习：

```text
什么状态下做什么动作，可以获得更高的长期 reward。
```

PPO 和 SAC 都是 model-free 方法。

它们不显式学习环境动力学模型，也不通过模型进行 planning。

---

### 9.3 Model-based

Model-based RL 显式学习或使用 dynamics model。

典型方法包括：

* MPC；
* iLQR；
* learned dynamics model；
* world model；
* MBPO。

Model-based 方法会关心：

```text
如果当前状态下执行某个动作，未来状态会怎么变化？
```

然后基于模型做 planning 或辅助 policy learning。

---

### 9.4 MuJoCo 中训练 PPO / SAC 为什么仍然是 Model-free

这是机器人面试中很容易被问到的问题。

MuJoCo 本身是物理仿真器，内部当然有物理动力学模型。但是，如果 PPO / SAC 只是把 MuJoCo 当作环境来采样 transition，然后根据 reward 更新 policy 或 value function，那么算法本身仍然是 model-free。

原因是：

```text
PPO / SAC 没有显式学习 dynamics model；
PPO / SAC 没有使用 dynamics model 做 planning；
PPO / SAC 只是通过环境交互数据更新策略。
```

因此：

```text
虽然 MuJoCo 是物理仿真器，但在 MuJoCo 中训练 PPO / SAC 仍然属于 model-free RL。
```

---

### 9.5 面试表达

如果面试官问：

> MuJoCo 是物理仿真器，为什么在里面训练 PPO / SAC 仍然叫 model-free？

可以回答：

> Model-free 和 model-based 的区别不取决于环境是不是物理仿真器，而取决于算法是否显式学习或使用 dynamics model 进行 planning。虽然 MuJoCo 内部有物理模型，但 PPO 和 SAC 只是把 MuJoCo 当作环境来采样 transition，并根据 reward 更新 policy 或 value function。它们没有显式学习 s, a 到 s' 的模型，也没有用模型做规划，所以仍然属于 model-free RL。

---

## 10. BC、DAgger、PPO、SAC 对比

| 方法     | 本质      | 数据来源                           | 是否需要 reward | 是否需要 expert | 主要解决什么问题                 |
| ------ | ------- | ------------------------------ | ----------- | ----------- | ------------------------ |
| BC     | 监督式模仿学习 | Expert demonstrations          | 不需要         | 需要          | 快速模仿 expert              |
| DAgger | 交互式模仿学习 | Learner states + expert labels | 不一定         | 需要          | 缓解 BC distribution shift |
| PPO    | 强化学习    | 当前 policy rollout              | 需要          | 不需要         | 稳定的 on-policy 策略优化       |
| SAC    | 强化学习    | Replay buffer                  | 需要          | 不需要         | 高样本效率连续控制                |

---

## 11. PPO 和 SAC 对比

| 维度              | PPO                   | SAC                                    |
| --------------- | --------------------- | -------------------------------------- |
| 类型              | On-policy             | Off-policy                             |
| 是否 Model-free   | 是                     | 是                                      |
| 架构              | Actor-Critic          | Actor-Critic                           |
| 数据使用            | 当前 policy 新采集的数据      | Replay buffer 中的历史数据                   |
| 样本效率            | 较低                    | 较高                                     |
| 探索机制            | 依赖策略随机性               | Maximum entropy                        |
| 稳定性             | 比较稳定                  | 依赖 Q network 和超参数                      |
| 适用场景            | 稳定 baseline、仿真任务      | 连续控制、采样昂贵任务                            |
| Panda Lift 中的定位 | On-policy RL baseline | Off-policy continuous control baseline |

---

## 12. 当前 Panda Lift 项目的算法定位

当前项目已经完成的内容包括：

1. 搭建 MuJoCo / robosuite Panda Lift 环境；
2. 理解 observation 和 action space；
3. 设计 handcrafted teacher；
4. 收集 demonstration dataset；
5. 训练 BC policy；
6. 做 unfiltered BC 和 success-filtered BC 对比；
7. 做 rollout diagnostics；
8. 使用 success、max_lift、final_lift、min_eef_cube_dist 分析策略表现。

这些内容对应的是：

```text
机器人 imitation learning baseline
```

也可以理解为：

```text
Panda Lift BC baseline + rollout diagnostics
```

---

## 13. 后续项目扩展路线

当前项目可以自然扩展为：

```text
BC baseline
↓
DAgger-BC
↓
PPO baseline
↓
SAC baseline
↓
BC / DAgger / PPO / SAC 对比实验
```

每一步的意义如下：

| 阶段           | 作用                                        |
| ------------ | ----------------------------------------- |
| BC baseline  | 用 expert demonstration 快速训练初始策略           |
| DAgger-BC    | 缓解 BC 的 distribution shift                |
| PPO baseline | 建立 on-policy RL baseline                  |
| SAC baseline | 建立 off-policy continuous control baseline |
| 对比实验         | 分析不同学习范式在 Panda Lift 上的效果                 |

可以对比的指标包括：

* success rate；
* total reward；
* max_lift；
* final_lift；
* min_eef_cube_dist；
* rollout 稳定性；
* 样本效率。

---

## 14. 面试高频回答模板

### 14.1 BC 是什么？

BC 是 supervised imitation learning，本质是学习 expert demonstration 中的 observation-action mapping。它不需要 reward，而是直接用 expert action 作为监督信号。在我的 Panda Lift 项目中，我用 handcrafted teacher 生成 demonstration dataset，然后训练 BC policy 模仿 teacher 的连续控制动作。

---

### 14.2 BC 为什么会失败？

BC 的核心问题是 distribution shift。训练时 policy 看到的是 expert 访问的状态，但在 closed-loop rollout 中，policy 的小误差会逐步累积，使机器人进入 expert demonstration 中没有覆盖的状态。因此 training loss 低不一定代表 rollout success 高。在 Panda Lift 中，这可能表现为夹爪偏离方块、闭合时机错误、方块被碰歪，或者没有夹住方块却继续上抬。

---

### 14.3 DAgger 为什么有用？

DAgger 通过让当前 policy 自己 rollout，收集 learner-induced states，并让 expert 对这些 states 标注 corrective actions，再把新数据聚合进训练集。这样 policy 不只学习 expert 的标准轨迹，也学习偏离后的恢复动作，从而缓解 BC 的 distribution shift。

---

### 14.4 PPO 和 SAC 的区别是什么？

PPO 是 on-policy model-free actor-critic 方法，用 clipping 限制新旧策略差距，因此训练稳定但样本效率较低。SAC 是 off-policy model-free actor-critic 方法，可以复用 replay buffer 中的历史经验，并通过 entropy regularization 鼓励探索，因此更适合连续动作和采样昂贵的机器人控制任务。

---

### 14.5 为什么 MuJoCo 中训练 PPO / SAC 仍然是 model-free？

虽然 MuJoCo 是物理仿真器，但 PPO 和 SAC 并没有显式学习或使用 dynamics model 做 planning。它们只是通过环境交互采样 transition，并基于 reward 更新 policy 或 value function。因此从算法分类上看，它们仍然属于 model-free RL。

---

### 14.6 为什么 SAC 适合机器人连续控制？

SAC 适合机器人连续控制，主要因为它是 off-policy 方法，可以利用 replay buffer 复用历史数据，提高样本效率。同时 SAC 适合连续动作空间，并使用 maximum entropy objective 鼓励探索。机器人 manipulation 任务通常探索困难、动作连续、采样代价高，因此 SAC 是常用的强 baseline。

---

### 14.7 PPO 为什么稳定但样本效率低？

PPO 稳定是因为它通过 clipping 限制新旧 policy 之间的差距，避免一次 policy update 过大导致策略崩掉。但 PPO 是 on-policy 方法，每次更新都依赖当前 policy 新采集的数据，旧数据通常不能反复使用，因此样本效率较低。

---

## 15. 本周任务 A 总结

本周任务 A 完成了 PPO、SAC、BC、DAgger、On-policy / Off-policy、Model-free / Model-based 的核心概念整理，并将这些概念映射到当前 MuJoCo / robosuite / Panda Lift / BC 项目。

当前阶段的重点不是写代码，而是建立机器人深度强化学习岗位面试所需的算法理解框架。

本任务完成后，当前项目的学习路线可以继续推进到：

```text
Week 06 Task B：DAgger 设计与实现准备
```

或者：

```text
PPO / SAC baseline 实验规划
```

本周任务 A 的核心收获可以总结为：

```text
BC 解决机器人策略冷启动问题；
DAgger 解决 BC 的 distribution shift 问题；
PPO 提供稳定的 on-policy RL baseline；
SAC 提供高样本效率的 off-policy continuous control baseline；
On-policy / Off-policy 关注数据能不能复用；
Model-free / Model-based 关注是否显式学习或使用动力学模型。
```
