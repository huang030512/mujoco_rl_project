# Week 08 Task A｜Transformer Behavior Cloning 最小版本

## 1. 任务目标

本任务的目标是将原来的 Panda Lift MLP Behavior Cloning 升级为一个最小版本的 Transformer Behavior Cloning。

原来的 MLP-BC 使用单步 observation 预测当前 action：

```text
obs[t] -> action[t]
```

本次 Transformer-BC 使用过去一段 observation 序列预测当前 action：

```text
obs[t-seq_len+1 : t+1] -> action[t]
```

本任务不是完整 ACT，也不预测未来 action chunk，而是先实现一个最小可运行版本：

```text
过去 seq_len 步 observation -> 当前 action
```

---

## 2. 为什么机器人模仿学习里要用 Transformer

机器人操作任务具有明显的时间顺序。

以 Panda Lift 为例，任务大致可以分为：

```text
靠近方块 -> 下降 -> 闭合夹爪 -> 抬升
```

如果只看当前一帧 observation，模型有时很难判断当前处于哪个阶段。

例如机械臂末端已经接近方块时，当前动作可能是：

```text
继续下降
闭合夹爪
开始抬升
```

这些动作的选择不仅取决于当前状态，也取决于之前发生过什么。

Transformer 的优势是可以输入一段历史 observation，通过 self-attention 建模时间上下文，让模型根据过去若干步信息判断当前应该执行什么动作。

因此，在机器人模仿学习中，Transformer 可以用于：

```text
observation history -> action
```

它比单步 MLP 更适合处理具有阶段变化和时间依赖的操作任务。

---

## 3. MLP-BC 和 Transformer-BC 的区别

### 3.1 MLP Behavior Cloning

原来的 MLP-BC 形式是：

```text
obs[t] -> MLP -> action[t]
```

特点是：

```text
只看当前一帧 observation
不显式建模时间历史
每个训练样本相对独立
结构简单，训练稳定
```

### 3.2 Transformer Behavior Cloning

本次实现的 Transformer-BC 形式是：

```text
obs[t-seq_len+1], ..., obs[t] -> Transformer -> action[t]
```

特点是：

```text
输入一段 observation 序列
通过 self-attention 建模时间关系
可以利用历史信息判断当前任务阶段
更接近现代 robot learning / imitation learning 中的序列建模方法
```

### 3.3 对比总结

| 方法 | 输入 | 输出 | 是否利用历史 | 适合任务 |
|---|---|---|---|---|
| MLP-BC | 当前 obs | 当前 action | 否 | 简单状态到动作映射 |
| Transformer-BC | 过去 seq_len 步 obs | 当前 action | 是 | 有阶段变化和时间依赖的操作任务 |

---

## 4. 数据格式设计

原始 BC 数据格式为：

```text
observations: [N, obs_dim]
actions:      [N, action_dim]
```

本项目中：

```text
observations: [3000, 15]
actions:      [3000, 7]
```

其中：

```text
obs_dim = 15
action_dim = 7
```

Transformer 需要序列输入，因此需要将单步数据转成滑动窗口序列数据。

设：

```text
seq_len = 16
```

则每个训练样本为：

```text
obs_seq = obs[t-15 : t+1]
target_action = action[t]
```

对应形状：

```text
obs_seq:       [seq_len, obs_dim]
target_action: [action_dim]
```

加入 batch 后：

```text
batch_obs_seq:       [batch_size, seq_len, obs_dim]
batch_target_action: [batch_size, action_dim]
```

---

## 5. 避免跨 episode 构造序列

机器人数据通常由多个 episode 拼接而成。

如果直接在整个数组上滑动窗口，可能出现错误样本：

```text
episode 0 的最后几步 + episode 1 的前几步
```

这种序列在物理上是不连续的，不能作为机器人历史输入。

因此，本任务使用 `episode_ids` 按 episode 内部构造序列，只在同一个 episode 内部滑动窗口。

本次数据中：

```text
episode 数量: 10
每个 episode 长度约为 300
seq_len: 16
```

每个 episode 可构造的序列样本数为：

```text
300 - 16 + 1 = 285
```

10 个 episode 总共：

```text
285 * 10 = 2850
```

训练日志中也验证了这一点：

```text
Sequence samples: 2850
```

---

## 6. TransformerBCPolicy 模型结构

本次实现的最小 TransformerBCPolicy 结构为：

```text
obs_seq
  -> obs embedding
  -> positional encoding
  -> TransformerEncoder
  -> 取最后一个时间步 hidden state
  -> action head
  -> pred_action
```

### 6.1 Observation Embedding

原始输入为：

```text
[batch_size, seq_len, obs_dim]
```

其中 `obs_dim = 15`。

Transformer 内部使用 `d_model` 维特征，因此先用线性层将每一帧 observation 映射成 token embedding：

```text
[batch_size, seq_len, obs_dim]
->
[batch_size, seq_len, d_model]
```

可以理解为：

```text
每一帧 observation 是一个 token
```

### 6.2 Positional Encoding

Transformer 本身不知道序列顺序，因此需要加入位置编码。

本任务使用可学习的位置编码：

```text
pos_embed: [1, seq_len, d_model]
```

它的作用是告诉模型：

```text
这是第 1 步 observation
这是第 2 步 observation
...
这是第 seq_len 步 observation
```

### 6.3 TransformerEncoder

TransformerEncoder 对一段 observation token 做 self-attention，让不同时间步之间互相建立关系。

输入输出形状保持一致：

```text
输入: [batch_size, seq_len, d_model]
输出: [batch_size, seq_len, d_model]
```

它可以让最后一个时间步的表示融合前面历史 observation 的信息。

### 6.4 取最后一个时间步

因为本任务预测的是当前 action，也就是序列最后一帧对应的动作：

```text
obs[t-seq_len+1 : t+1] -> action[t]
```

所以取 Transformer 输出的最后一个时间步：

```text
last_hidden = x[:, -1, :]
```

形状为：

```text
[batch_size, d_model]
```

### 6.5 Action Head

最后通过一个 MLP action head 输出动作：

```text
[batch_size, d_model]
->
[batch_size, action_dim]
```

在 Panda Lift 中：

```text
[batch_size, d_model]
->
[batch_size, 7]
```

---

## 7. 训练方法

Transformer-BC 本质上仍然是监督学习。

训练目标是让模型预测动作接近专家动作：

```text
pred_action ≈ expert_action
```

使用 MSELoss：

```text
loss = MSE(pred_action, target_action)
```

优化器使用 Adam。

训练流程：

```text
obs_seq
-> TransformerBCPolicy
-> pred_action
-> 和 target_action 计算 MSELoss
-> 反向传播
-> Adam 更新参数
```

并使用 train / validation split 监控模型泛化能力。

---

## 8. 本次训练结果

运行命令：

```bash
python src/train_panda_lift_transformer_bc.py
```

训练输出关键信息：

```text
Training data path: data/panda_lift_bc_data_with_metrics_10ep.npz
Model save path: models/bc/panda_lift_transformer_bc_policy.pt
Found episode_ids in data.
Unique episode ids: 10
Contiguous episode chunks: 10
Loaded observations: (3000, 15)
Loaded actions: (3000, 7)
seq_len: 16
Sequence samples: 2850
obs_dim: 15
action_dim: 7
Using device: cuda
```

训练过程中，验证集 loss 从：

```text
Epoch 001 | val_loss=0.023337
```

下降到最好：

```text
Best val loss: 0.000169
```

最终模型保存到：

```text
models/bc/panda_lift_transformer_bc_policy.pt
```

---

## 9. 结果分析

本次训练结果说明 Transformer-BC 已经成功学习到：

```text
过去 16 步 observation -> 当前 expert action
```

的离线监督映射。

验证集 loss 明显下降，说明模型能够拟合专家数据中的动作模式。

但是需要注意：

```text
离线 val_loss 很低，不等于真实 rollout 一定成功
```

原因是：

```text
1. 验证集仍然来自专家数据分布
2. 模型 rollout 时会遇到自己产生的状态分布
3. 一旦动作有小误差，后续状态可能逐渐偏离专家轨迹
4. Panda Lift 对抓取位置和夹爪时机比较敏感
```

因此，本任务目前只能说明：

```text
Transformer-BC 离线训练链路已经跑通
```

不能直接说明：

```text
Transformer-BC 已经稳定完成 Panda Lift 抓取
```

后续还需要编写评估脚本，在 robosuite 环境中 rollout 检查实际控制效果。

---

## 10. 当前任务产出

本任务完成了以下文件：

```text
src/train_panda_lift_transformer_bc.py
models/bc/panda_lift_transformer_bc_policy.pt
notes/week08_transformer_bc_intro.md
```

其中：

```text
src/train_panda_lift_transformer_bc.py
```

用于训练最小版本 Transformer Behavior Cloning。

```text
models/bc/panda_lift_transformer_bc_policy.pt
```

保存训练得到的 Transformer BC 策略参数。

```text
notes/week08_transformer_bc_intro.md
```

记录任务原理、数据格式、模型结构、训练结果和局限性分析。

---

## 11. 面试可讲版本

我将原来的 Panda Lift MLP Behavior Cloning 升级成了一个最小版本的 Transformer Behavior Cloning。

原来的 MLP-BC 只使用当前 observation 预测当前 action，而 Transformer-BC 使用过去 16 步 observation 序列预测当前 action。这样可以让模型利用时间上下文，更好地区分靠近、下降、夹取、抬升等不同操作阶段。

在数据处理上，我将原始的 `[N, obs_dim]` 和 `[N, action_dim]` 数据按 episode 内部滑动窗口转换为 `[N_seq, seq_len, obs_dim]` 和 `[N_seq, action_dim]`，并避免跨 episode 拼接不连续序列。

模型结构上，我使用 observation embedding 将每一帧 observation 映射为 token embedding，加入可学习的位置编码，然后通过 TransformerEncoder 建模历史 observation 之间的关系，最后取最后一个时间步的 hidden state，通过 action head 回归当前动作。

本次实验使用 10 条 Panda Lift 轨迹，共 3000 个原始样本，构造出 2850 个序列样本。训练使用 MSELoss 和 Adam，最终验证集最优 loss 达到 0.000169，并保存了 Transformer BC policy。

这个实验说明我已经完成了从单步 MLP 模仿学习到序列 Transformer 模仿学习的最小升级，但目前结果主要是离线监督训练效果，后续还需要进行环境 rollout 评估实际控制表现。