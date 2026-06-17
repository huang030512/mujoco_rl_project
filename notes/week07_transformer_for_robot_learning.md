# 第7周任务D｜面向机器人学习的 Transformer 入门总结

## 1. 学习目标

本节学习的 Transformer 不是面向 NLP 背诵概念，而是为了理解机器人学习、模仿学习、VLA、ACT、Diffusion Policy 等方法中的核心结构。

重点理解：

- Token / Embedding / Sequence
- Attention / Self-Attention / Cross-Attention
- Multi-Head Attention
- Encoder / Decoder / Encoder-Decoder
- Position Embedding
- Mask
- Transformer 如何输出 action 或 action sequence
- 和 Panda Lift Behavior Cloning 项目的关系

---

## 2. 从 MLP BC 到 Transformer Policy

我之前的 Panda Lift BC 是：

```text
当前 observation → MLP → 当前 action
```

也就是：

```text
obs_t → action_t
```

这种方式只看当前时刻的 observation。

Transformer 的思想是：

```text
一串 token → Transformer → action 或 action sequence
```

在机器人任务中，token 不一定是文字，也可以是：

- 一个 observation
- 一个 action
- 一个历史状态
- 一个图像 patch
- 一个语言 token
- 一个 action query token

所以机器人 Transformer policy 可以写成：

```text
历史 obs/action + 图像 + 语言
→ Transformer
→ 当前 action 或未来 action sequence
```

---

## 3. Token、Embedding、Sequence

### Token

Token 是 Transformer 处理的基本单位。

在 NLP 中，token 可能是一个词；在机器人学习中，token 可以是：

```text
obs_t
action_t
image patch
language word
robot state
action query
```

### Embedding

Embedding 是把原始 token 转成统一维度的向量。

例如 Panda Lift 的 observation 是 15 维：

```text
obs_t ∈ R^15
```

可以通过 Linear 层变成：

```text
embedding_t ∈ R^128
```

### Sequence

Sequence 是一串按顺序排列的 token。

比如：

```text
[obs_t-3, action_t-3, obs_t-2, action_t-2, obs_t-1, action_t-1, obs_t]
```

Transformer 擅长处理这种序列数据。

---

## 4. Attention

Attention 的核心作用是：

```text
从一串 token 中判断当前最应该关注哪些信息
```

公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

直觉理解：

```text
Q = 当前 token 想找什么信息
K = 每个 token 用来被匹配的特征
V = 每个 token 真正提供的信息内容
```

计算过程：

```text
token embedding
→ 生成 Q/K/V
→ Q 和 K 计算相关性
→ softmax 得到 attention 权重
→ 用权重加权 V
→ 得到融合上下文的新 token 表示
```

机器人例子：

```text
当前 obs_t 要输出 action
它可能重点关注：
- 当前末端位置
- 方块位置
- 上一步 action
- gripper 是否闭合
- cube_z 是否上升
```

---

## 5. Self-Attention

Self-Attention 是：

```text
同一串 token 内部互相看
```

例如输入：

```text
[obs_t-2, action_t-2, obs_t-1, action_t-1, obs_t]
```

Self-Attention 会让每个 token 都去关注其他 token。

比如 `obs_t` 可以关注：

```text
obs_t-1
action_t-1
obs_t-2
当前 obs_t
```

经过 Self-Attention 后：

```text
obs_t 不再只是当前 observation
而是融合了历史上下文的新表示
```

这对机器人任务很重要，因为机器人动作不是孤立的，而是连续过程。

---

## 6. Multi-Head Attention

Multi-Head Attention 的意思是：

```text
同时计算多套 Attention
```

普通 Attention：

```text
一组 Wq/Wk/Wv → 一套 Q/K/V → 一套 attention
```

Multi-Head Attention：

```text
多组 Wq/Wk/Wv → 多套 Q/K/V → 多套 attention → 融合结果
```

每个 head 可以学习不同关系，例如：

```text
head 1：关注末端和方块的位置关系
head 2：关注 gripper 和历史 action
head 3：关注 cube_z 是否变化
head 4：关注任务阶段
```

这些不是人工指定的，而是通过 loss 自动学习出来的。

---

## 7. Cross-Attention

Self-Attention 是：

```text
同一组 token 内部互相看
```

Cross-Attention 是：

```text
一组 token 去看另一组 token
```

例如在机器人 VLA 中：

```text
action token 去看 image tokens
action token 去看 language tokens
action token 去看 robot state tokens
```

Cross-Attention 中：

```text
Q 来自 action token
K/V 来自 image / language / state tokens
```

直觉：

```text
action token 问：我要输出动作，需要从图像和语言里读取什么信息？
```

Self-Attention 和 Cross-Attention 不是替代关系：

```text
Self-Attention：内部理解
Cross-Attention：跨组读取
```

---

## 8. Encoder、Decoder、Encoder-Decoder

### Encoder-only

Encoder-only 是：

```text
输入 token sequence → Encoder → action head → action
```

用于机器人 BC 时：

```text
历史 obs/action sequence
→ Transformer Encoder
→ 取最后一个 token 的 hidden state
→ Linear action head
→ action_t
```

它适合从 MLP BC 升级到 Transformer BC。

### Decoder-only

Decoder-only 类似 GPT：

```text
过去 tokens → masked self-attention → 预测下一个 token/action
```

机器人中可以写成：

```text
obs_1, action_1, obs_2, action_2, obs_3
→ predict action_3
```

它需要 causal mask，防止模型偷看未来。

### Encoder-Decoder

Encoder-Decoder 是：

```text
输入 tokens → Encoder → encoded features
action query tokens → Decoder → action sequence
```

机器人版本：

```text
图像 / 语言 / 状态 / 历史
→ Encoder
→ encoded features

未来 action query tokens
→ Decoder self-attention
→ Cross-attention 读取 encoded features
→ action head
→ future action sequence
```

例如预测未来 10 步 Panda Lift action：

```text
decoder output: [batch, 10, hidden_dim]
action head: Linear(hidden_dim, 7)
pred_actions: [batch, 10, 7]
```

---

## 9. Action Head

Transformer 本身通常不直接输出 action。

它输出的是：

```text
hidden state / token feature
```

真正输出 action 的是最后的：

```text
Linear 或 MLP action head
```

例如：

```text
hidden_dim = 128
action_dim = 7
```

则：

```text
action = Linear(128, 7)(hidden_state)
```

训练时，action head 和 Transformer 主体一起通过 loss 更新。

---

## 10. Position Embedding

Transformer 本身不知道 token 的顺序。

所以需要 Position Embedding 告诉模型：

```text
这个 token 在第几个位置
```

公式直觉：

```text
input_embedding = token_embedding + position_embedding
```

在 Panda Lift 中，顺序很重要：

```text
靠近 → 下降 → 闭合夹爪 → 抬升
```

如果没有位置编码，Transformer 不知道哪个 token 是过去，哪个 token 是当前。

一句话：

```text
Token embedding 告诉模型“这是什么”
Position embedding 告诉模型“它在第几步”
```

---

## 11. Mask

Mask 的作用是：

```text
告诉 Transformer 哪些 token 可以看，哪些 token 不能看
```

### Causal Mask

用于 Decoder-only 或自回归预测：

```text
当前 token 只能看过去，不能看未来
```

防止模型偷看未来答案。

### Padding Mask

用于处理不同长度的序列。

比如未来动作不够 10 步：

```text
[action_t, action_t+1, action_t+2, PAD, PAD, ...]
```

需要 mask：

```text
[1, 1, 1, 0, 0, ...]
```

只对真实动作算 loss，不对 PAD 算 loss。

---

## 12. 和 ACT 的关系

ACT 可以先理解成 Transformer 在机器人模仿学习中的一个应用。

普通 BC：

```text
obs_t → action_t
```

ACT：

```text
obs_t → [action_t, action_t+1, ..., action_t+k]
```

也就是一次预测未来一小段 action chunk。

训练数据来自专家轨迹滑动窗口：

```text
obs_1 → [action_1, ..., action_10]
obs_2 → [action_2, ..., action_11]
obs_3 → [action_3, ..., action_12]
```

ACT 的核心不是新的 Transformer 原理，而是：

```text
Transformer + action query + action chunk supervision
```

---

## 13. 面试表达版本

我可以这样解释 Transformer 在机器人学习中的作用：

```text
我理解 Transformer 的核心是把输入组织成 token sequence，通过 attention 建模 token 之间的关系。在机器人任务中，token 不一定是文字，也可以是 observation、action、图像 patch、语言 token 或 action query。

相比我之前的 Panda Lift MLP BC 只用当前 observation 预测当前 action，Transformer policy 可以利用历史 observation/action、图像和语言信息。Self-Attention 用来建模同一序列内部关系，Cross-Attention 可以让 action token 读取视觉、语言和机器人状态信息。最后 Transformer 输出 hidden representation，再通过 action head 映射到连续动作空间。

如果使用 Encoder-only，可以把历史 obs/action 编码后输出当前动作；如果使用 Encoder-Decoder，可以用 action query 生成未来一段 action sequence，这也是 ACT 等机器人模仿学习方法的核心思路。
```

---

## 14. 当前阶段需要懂到什么程度

现阶段需要掌握：

- token / embedding / sequence 是什么
- Q/K/V 的直觉含义
- Attention 权重是什么意思
- Self-Attention 和 Cross-Attention 区别
- Multi-Head 为什么是多套 Q/K/V
- Encoder / Decoder 如何输出 action
- Position Embedding 为什么必要
- Mask 是为了防止看未来或忽略 padding
- Transformer 如何从机器人数据预测 action

暂时不需要深入：

- 手推 Attention 反向传播
- 手写 Multi-Head Attention 底层实现
- FlashAttention / CUDA 优化
- 所有 Transformer 变体
- ACT / Diffusion Policy / VLA 的论文细节

---

## 15. 一句话总结

```text
MLP 看一个 observation，Transformer 看一串 token。
Attention 让 token 互相读取信息。
Transformer 输出的是上下文特征。
Action head 把特征变成机器人动作。
```