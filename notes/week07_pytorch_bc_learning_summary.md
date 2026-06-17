# Week 7｜PyTorch + Behavior Cloning 学习总结

## 1. 学习目标

本阶段的目标不是盲目继续扩展新项目，而是结合已有的 Panda Lift Behavior Cloning 项目，补强 PyTorch 基础，并能够从机器人模仿学习的角度讲清楚：

- PyTorch 在机器人学习项目中承担什么作用；
- Behavior Cloning 的训练流程如何用 PyTorch 实现；
- Tensor、Dataset、DataLoader、`nn.Module`、Loss、Optimizer 分别对应项目中的哪一部分；
- 为什么 BC 的 validation loss 低，不一定代表 closed-loop rollout 成功；
- 如何把 PyTorch 训练流程转化为面试表达。

本阶段重点不是学习大量 PyTorch API，而是围绕一个完整的机器人模仿学习项目，理解 PyTorch 在其中的作用和基本训练机制。

---

## 2. 对 PyTorch 的整体理解

我将 PyTorch 理解为一个深度学习训练框架。

它主要负责：

- Tensor 数据表示；
- 神经网络构建；
- 自动求导 autograd；
- 参数优化 optimizer；
- 模型保存与加载。

PyTorch 本身不是某一个具体算法，而是实现 BC、PPO、SAC、Transformer Policy、Diffusion Policy 等方法的基础工具。

不同算法虽然目标和细节不同，但底层通常都共享类似的训练流程：

```text
batch data
↓
network forward
↓
compute loss
↓
loss.backward()
↓
optimizer.step()
```

不同算法的主要区别在于：

- 数据来源不同；
- 网络结构不同；
- loss 定义不同；
- 参数更新方式不同。

因此，可以把 PyTorch 理解为 robot learning 中训练神经网络策略的基础框架，而 BC、PPO、SAC 等方法则是在这个框架上定义不同训练目标和训练方式的算法。

---

## 3. PyTorch、BC、PPO、SAC 的关系

PyTorch 负责底层神经网络训练能力，包括：

- Tensor 计算；
- 神经网络模块定义；
- 自动求导；
- 优化器更新；
- 模型保存和加载。

BC、PPO、SAC 等算法则决定：

- 训练什么网络；
- 数据从哪里来；
- loss 怎么定义；
- 参数怎么更新。

可以理解为：

```text
PyTorch
= 深度学习训练框架

BC / PPO / SAC
= 基于 PyTorch 实现的不同训练算法
```

例如：

### Behavior Cloning

BC 训练的是一个 policy network：

```text
observation -> action
```

数据来自专家轨迹，loss 通常是：

```text
MSE(pred_action, expert_action)
```

### PPO

PPO 通常训练：

- Actor / Policy Network；
- Critic / Value Network。

数据来自当前 policy 与环境交互得到的 on-policy rollout，loss 包括：

- policy loss；
- value loss；
- entropy bonus。

### SAC

SAC 通常训练：

- Actor Network；
- Q Critic Network 1；
- Q Critic Network 2；
- Target Critic Networks。

数据来自 replay buffer，loss 包括：

- critic Bellman backup loss；
- actor maximum entropy objective；
- temperature loss。

因此，我对 PyTorch 和强化学习算法的关系可以总结为：

> PyTorch 负责“怎么训练神经网络”，算法负责“训练什么网络、用什么数据、定义什么 loss、如何更新策略”。

---

## 4. PyTorch 和 Panda Lift BC 项目的对应关系

在 Panda Lift Behavior Cloning 项目中，PyTorch 用来训练一个从机器人 observation 到 action 的策略网络。

整体流程是：

```text
专家策略 rollout
↓
保存 observation-action 数据
↓
读取 .npz 数据
↓
转成 PyTorch Tensor
↓
Dataset / DataLoader 组成 batch
↓
MLP policy 预测 action
↓
MSELoss 比较预测动作和专家动作
↓
Adam 优化器更新网络参数
↓
保存 state_dict
↓
加载模型并放回 robosuite rollout
```

这个过程本质上是一个监督学习问题：

```text
输入：机器人当前 observation
输出：专家在该状态下执行的 action
目标：让神经网络学会模仿专家策略
```

---

## 5. Tensor：PyTorch 中的数据表示

Tensor 是 PyTorch 中进行数值计算和神经网络训练的基本数据结构。

可以先将 Tensor 理解为带有深度学习能力的数组。相比 NumPy array，PyTorch Tensor 具有两个重要能力：

- 可以放到 GPU 上加速计算；
- 可以参与 autograd 自动求导。

在 Panda Lift BC 项目中，Tensor 主要表示两类数据：

- observation tensor；
- action tensor。

每条训练数据是一组：

```text
observation -> expert action
```

其中：

```text
observation: 15 维
action:      7 维
```

训练时，DataLoader 会组成 batch：

```text
obs_batch.shape    = [batch_size, 15]
action_batch.shape = [batch_size, 7]
```

模型输出：

```text
pred_action.shape = [batch_size, 7]
```

因此，在 BC 项目中可以将 PyTorch Tensor 理解为：

> 把机器人状态和专家动作变成神经网络可以处理的数据格式。

---

## 6. Observation 和 Action 的含义

在 Panda Lift BC 项目中，observation 是机器人当前状态的低维表示。

主要包括：

- 末端执行器位置 `eef_pos`；
- 夹爪状态 `gripper_qpos`；
- 方块位置 `cube_pos`；
- 末端到方块的相对位置 `gripper_to_cube_pos`；
- 当前阶段 `phase one-hot`。

这些信息拼接后形成 15 维 observation。

action 是 robosuite Panda 控制器接收的 7 维连续控制量，通常包括：

- 末端 x / y / z 方向控制；
- 姿态控制；
- 夹爪开合控制。

因此，BC 训练数据可以写成：

```text
D = {(s_i, a_i)}
```

其中：

```text
s_i = 第 i 个机器人 observation
a_i = 第 i 个专家 action
```

模型要学习的函数是：

```text
πθ(s_i) ≈ a_i
```

也就是：

```text
policy(observation) ≈ expert_action
```

---

## 7. Dataset 和 DataLoader

在 PyTorch 中，`Dataset` 负责定义“一条数据怎么取”，`DataLoader` 负责定义“一批数据怎么取”。

### Dataset

Dataset 通常需要实现两个函数：

```python
def __len__(self):
    return len(self.observations)

def __getitem__(self, idx):
    return self.observations[idx], self.actions[idx]
```

其中：

- `__len__`：返回数据集大小；
- `__getitem__`：返回第 `idx` 条 observation-action pair。

在 BC 项目中，一条数据就是：

```text
一个 observation
一个 expert action
```

也就是：

```text
obs_i, action_i
```

### DataLoader

DataLoader 会根据 Dataset 自动取样，并组成 batch。

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)
```

每次循环会得到：

```text
obs_batch.shape    = [64, 15]
action_batch.shape = [64, 7]
```

训练集通常设置：

```text
shuffle=True
```

原因是机器人专家轨迹数据通常按时间顺序存储，例如：

```text
靠近方块
下降
闭夹爪
抬升
```

如果不打乱，一个 batch 可能只包含某一个阶段的数据。打乱后，一个 batch 更可能混合不同阶段的专家行为，有助于训练稳定。

因此，在 Panda Lift BC 项目中：

```text
Dataset 管理专家轨迹数据
DataLoader 每次取出一批 obs/action 样本用于训练
```

---

## 8. BCDataset 的 Python 机制理解

在学习 Dataset 时，我进一步理解了 Python 类的初始化机制。

例如：

```python
dataset = BCDataset(data_path)
```

这句话的含义是：

```text
创建一个 BCDataset 对象
自动调用 BCDataset.__init__(self, data_path)
```

其中：

- `self`：表示当前创建出来的对象本身；
- `data_path`：是手动传入的参数。

如果类定义为：

```python
class BCDataset(Dataset):
    def __init__(self, data_path):
        ...
```

那么使用时可以写：

```python
dataset = BCDataset(data_path)
```

如果类定义为：

```python
class BCDataset(Dataset):
    def __init__(self, observations, actions):
        ...
```

那么使用时应该写：

```python
dataset = BCDataset(observations, actions)
```

因此，`BCDataset(...)` 中传什么参数，取决于 `__init__(...)` 中定义了什么参数。

这也帮助我理解了：

```python
model = BCPolicy(obs_dim=15, action_dim=7)
```

同样会自动调用：

```python
BCPolicy.__init__(self, obs_dim, action_dim)
```

区别在于：

```text
BCDataset 的 __init__:
初始化数据

BCPolicy 的 __init__:
初始化网络层
```

---

## 9. BCPolicy / MLP 网络

BCPolicy 是一个继承 `nn.Module` 的 MLP policy。

它学习的函数是：

```text
πθ(observation) = action
```

也就是：

```text
15 维机器人状态
↓
MLP policy
↓
7 维连续动作
```

一个典型网络结构可以写成：

```text
Linear(15, 256)
ReLU
Linear(256, 256)
ReLU
Linear(256, 7)
Tanh
```

其中：

- `Linear`：负责特征映射；
- `ReLU`：提供非线性表达能力；
- `Tanh`：将输出限制在 `[-1, 1]`。

### 为什么要继承 nn.Module？

`nn.Module` 是 PyTorch 中定义神经网络的标准基类。继承它以后，PyTorch 才能自动管理：

- 网络层；
- 可训练参数；
- forward 计算逻辑；
- `model.parameters()`；
- `model.train()` / `model.eval()`；
- `state_dict` 保存和加载。

因此，BCPolicy 不是普通 Python 函数，而是一个可以训练、保存、加载、部署的机器人策略网络。

### forward 的作用

`forward` 定义 observation 如何经过网络变成 action。

例如：

```python
def forward(self, obs):
    return self.net(obs)
```

当执行：

```python
pred_action = model(obs_batch)
```

PyTorch 实际会调用模型的 forward 逻辑。

因此，在 BC 项目中：

```text
model(obs_batch)
=
把一批机器人 observation 输入 policy，得到一批 action prediction
```

---

## 10. Loss：动作预测误差

Behavior Cloning 本质是监督学习。

训练目标是让模型预测动作接近专家动作：

```text
pred_action ≈ expert_action
```

因此使用 MSELoss：

```text
loss = MSE(pred_action, expert_action)
```

在训练中：

```python
pred_action = model(obs_batch)
loss = criterion(pred_action, action_batch)
```

其中：

- `pred_action`：模型预测的动作；
- `action_batch`：专家动作标签；
- `loss`：预测动作和专家动作之间的误差。

由于 Panda Lift 的 action 是连续控制量，因此 MSELoss 是合理的选择。

如果动作是离散类别，可能会使用 CrossEntropyLoss；但在连续机器人控制中，MSELoss 常用于 action regression。

---

## 11. Autograd：自动求导机制

PyTorch 的 autograd 会在 forward 过程中动态构建计算图。

在 BC 项目中，计算图可以理解为：

```text
obs_batch
↓
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
↓
Tanh
↓
pred_action
↓
MSELoss
↓
loss
```

当调用：

```python
loss.backward()
```

PyTorch 会自动计算 loss 对所有可训练参数的梯度。

这些梯度会存储在每个参数的：

```python
param.grad
```

中。

需要注意的是：

```text
loss.backward() 只计算梯度
optimizer.step() 才真正更新参数
```

因此：

- `backward`：计算每个参数对 loss 的影响；
- `optimizer.step`：根据梯度修改参数。

在 BC 项目中，autograd 的作用是：

> 根据 action prediction loss 自动计算 MLP policy 中每个 weight 和 bias 的梯度。

---

## 12. Optimizer：参数更新

在项目中使用 Adam 优化器：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

其中：

- `model.parameters()`：返回 MLP policy 中所有需要训练的参数；
- `lr`：learning rate，学习率。

训练循环通常是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义分别是：

- `optimizer.zero_grad()`：清空上一轮梯度；
- `loss.backward()`：计算当前 loss 对模型参数的梯度；
- `optimizer.step()`：根据梯度更新模型参数。

需要注意：

```text
optimizer 更新的是模型参数
不是 observation
不是 expert action
也不是某一次具体的 pred_action
```

训练的本质是：

```text
修改 MLP 的 weight 和 bias
让以后类似 observation 输入时，输出更接近专家动作
```

---

## 13. 完整训练循环理解

Panda Lift BC 的训练循环可以理解为：

```python
for obs_batch, action_batch in train_loader:
    pred_action = model(obs_batch)
    loss = criterion(pred_action, action_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这几行代码对应的含义是：

```text
从 DataLoader 取一批 observation-action 数据
↓
把 observation 输入 MLP policy
↓
得到预测动作 pred_action
↓
用 MSELoss 比较 pred_action 和 expert_action
↓
通过 loss.backward() 计算梯度
↓
通过 optimizer.step() 更新 MLP 参数
```

对于一个 batch：

```text
obs_batch.shape    = [64, 15]
action_batch.shape = [64, 7]
pred_action.shape  = [64, 7]
```

因此 MSELoss 可以直接比较：

```text
pred_action 和 action_batch
```

因为两者 shape 一致。

---

## 14. Train / Validation

训练集中，模型会根据 loss 进行参数更新。

验证集中，模型只做前向预测和 loss 计算，不更新参数。

验证通常使用：

```python
model.eval()

with torch.no_grad():
    ...
```

其中：

- `model.eval()`：切换到评估模式；
- `torch.no_grad()`：不构建计算图，不计算梯度，节省显存和计算量。

在 BC 项目中，validation loss 用于检查模型在未参与训练的专家样本上的 action prediction error。

但是需要注意：

```text
validation loss 低，不一定代表 closed-loop rollout 成功
```

原因是 validation 数据仍然来自专家数据分布，而真正执行时，policy 会进入自己产生的状态分布。

---

## 15. 模型保存与加载

训练完成后保存的是：

```text
state_dict
```

也就是模型每一层的：

```text
weight
bias
```

保存方式通常是：

```python
torch.save(model.state_dict(), "models/bc/panda_lift_bc_policy.pt")
```

加载时需要重新定义相同结构的模型：

```python
model = BCPolicy(obs_dim=15, action_dim=7)
model.load_state_dict(torch.load("models/bc/panda_lift_bc_policy.pt"))
model.eval()
```

原因是：

```text
state_dict 只保存参数
不保存完整 Python 类定义
```

因此，加载模型时必须保证网络结构与训练时一致。

评估时一般使用：

```python
model.eval()

with torch.no_grad():
    action = model(obs_tensor)
```

---

## 16. PyTorch 和 NumPy 的区别

NumPy 主要用于数值计算，适合数组和矩阵操作。

PyTorch Tensor 和 NumPy array 很像，但 PyTorch Tensor 额外支持：

- 自动求导；
- GPU 加速；
- 神经网络训练。

在项目中，数据通常先以 NumPy 格式保存在 `.npz` 文件中，然后训练前转成 Tensor：

```python
obs_tensor = torch.from_numpy(obs).float()
action_tensor = torch.from_numpy(actions).float()
```

这样数据才能被 PyTorch 神经网络处理，并参与训练流程。

可以理解为：

```text
NumPy:
偏科学计算和数据存储

PyTorch:
偏深度学习训练和自动求导
```

---

## 17. PyTorch 和传统机器人控制代码的区别

传统机器人控制中，很多逻辑是人显式写出来的。

例如底盘运动学：

```text
输入 vx, vy, w
↓
运动学公式
↓
每个轮子的速度和角度
```

这是显式建模：

```text
人知道公式
人写规则
```

而 PyTorch 训练神经网络更偏向数据驱动：

```text
输入 observation
↓
神经网络 policy
↓
输出 action
```

模型不是通过人手写完整控制规则，而是通过大量数据学习输入到输出的映射关系。

因此可以理解为：

```text
传统控制:
人写规则

PyTorch 学习:
模型从数据中拟合规则
```

在 Panda Lift BC 中，手工策略提供专家数据，PyTorch 训练 MLP policy 去模仿专家行为。

---

## 18. 对 BC 局限性的理解

一个重要观察是：

```text
validation loss 低，不代表 closed-loop rollout 一定成功
```

原因是 BC 训练优化的是：

```text
专家数据分布上的 one-step action prediction loss
```

训练时模型看到的是专家轨迹中的状态：

```text
s ~ p_expert(s)
```

但真正执行时，模型会根据自己的动作进入新的状态：

```text
s ~ p_policy(s)
```

如果某一步动作有误差，后续 observation 可能偏离专家轨迹，导致：

- distribution shift；
- error accumulation；
- closed-loop failure。

因此，BC 可以作为模仿学习 baseline，但如果要提高真实闭环成功率，还需要：

- 更高质量的专家轨迹；
- 更广的状态覆盖；
- 多轨迹筛选；
- DAgger 等能收集 policy 状态分布的方法。

这也是机器人模仿学习和普通监督学习的重要区别。

---

## 19. 面试表达：PyTorch + BC 项目口述版

可以这样介绍这个项目：

> 我在 Panda Lift 任务中用 PyTorch 实现了一个 Behavior Cloning baseline。数据来自专家/手工策略 rollout，每条样本是 observation-action pair。observation 是 15 维机器人状态，包括末端位置、夹爪状态、方块位置、相对位置和 phase one-hot；action 是 7 维连续控制量。
>
> 我用 Dataset 和 DataLoader 构造 batch，训练一个 MLP policy，将 `[batch_size, 15]` 的 observation 映射到 `[batch_size, 7]` 的 action。训练目标是用 MSELoss 最小化预测动作和专家动作之间的误差，并使用 Adam 通过反向传播更新网络参数。
>
> 训练后保存模型的 state_dict，再加载到 robosuite 环境中进行 closed-loop rollout。这个实验也让我理解到，BC 的 validation loss 低不一定代表机器人任务成功，因为 BC 只在专家状态分布上优化 one-step action prediction，而实际执行时会遇到 policy 自己诱导出的状态分布，可能产生 distribution shift 和误差累积。

---

## 20. 更短的面试表达

如果面试官只要求简单说明，可以压缩为：

> 我用 PyTorch 做了 Panda Lift 的 Behavior Cloning，把专家策略 rollout 得到的 observation-action pair 作为监督学习数据。每条 observation 是 15 维，包括末端位置、夹爪状态、方块位置、相对位置和 phase one-hot；action 是 7 维连续控制量。模型是一个 MLP policy，输入 `[batch_size, 15]`，输出 `[batch_size, 7]`，用 MSELoss 拟合专家动作，用 Adam 通过反向传播更新参数。训练后保存 state_dict，评估时加载同结构网络并在 eval 和 no_grad 模式下 rollout。这个实验也让我理解到 BC 的局限：离线 loss 低不等于闭环成功，因为 policy 执行时会遇到自身诱导出的状态分布，可能产生 distribution shift 和误差累积。

---

## 21. 本阶段收获

通过本阶段学习，我建立了对 PyTorch 的基础理解：

```text
PyTorch 是训练神经网络策略的基础框架
BC / PPO / SAC 等算法可以基于 PyTorch 实现
不同算法共享 forward-loss-backward-step 的训练框架
区别在于数据来源、网络结构、loss 设计和更新方式
```

同时，我能够结合自己的 Panda Lift BC 项目解释：

```text
Tensor 如何表示机器人数据
Dataset / DataLoader 如何组织专家轨迹
nn.Module 如何定义策略网络
MSELoss 如何对应 action imitation
Adam 如何更新 policy 参数
state_dict 如何保存和加载模型
为什么 BC loss 低不等于 rollout 成功
```

本阶段的核心收获是：

> 我不再只是会运行 PyTorch 代码，而是能够理解 PyTorch 在机器人 Behavior Cloning 项目中的作用，并能将其解释为一个完整的 supervised imitation learning pipeline。

---

## 22. 后续学习方向

后续可以在此基础上继续学习：

- PPO / SAC 中 actor-critic 网络如何用 PyTorch 实现；
- BC 与 DAgger 的区别；
- 机器人策略中的 observation / action normalization；
- 更复杂的 policy network，例如 CNN、Transformer、Diffusion Policy；
- PyTorch 在视觉语言动作模型 VLA 中的作用。

现阶段最重要的是先掌握：

- PyTorch 基础训练流程；
- 机器人 BC 项目中的 PyTorch 使用；
- 项目代码与面试表达之间的转换。

这为后续学习强化学习、模仿学习和具身智能算法打下基础。