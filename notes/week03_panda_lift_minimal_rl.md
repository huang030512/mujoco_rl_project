# Week 03 - Panda Lift Minimal RL Training

## 1. 实验目标

本实验面向 Panda Lift 机械臂操作任务，尝试将 robosuite 的 Lift 环境封装为标准 Gymnasium 强化学习环境，并使用 SAC 算法进行最小训练闭环验证。

本实验的目标不是追求高成功率，而是验证以下能力：

1. 将 robosuite dict observation 封装为低维连续向量；
2. 将 robosuite action 接入 Stable-Baselines3 SAC；
3. 能完成 reset、step、reward、success 统计和模型保存；
4. 能根据 reward 曲线和 success rate 做工程取舍判断。

---

## 2. 环境封装方式

- 使用 robosuite Panda Lift 环境，关闭图像 observation，仅使用低维状态信息；
- 低维 observation 由以下部分组成：
  - `robot0_eef_pos`：末端执行器位置（3维）
  - `cube_pos`：方块位置（3维）
  - `gripper_to_cube_pos`：夹爪到方块相对位置（3维）
  - `robot0_gripper_qpos`：夹爪开合状态（2维）
- 封装为一维 numpy 向量，维度共 11；
- Action 维度为 7，连续空间 [-1,1]。

---

## 3. 奖励与成功判断

- reward_shaping=True，提供中间奖励（靠近方块、抓取、抬升等）；
- 成功判断使用 `info["success"]`，由于环境不会因 success 提前结束 episode；
- 取每个 episode 的总 reward 和 success 标记作为评估指标。

---

## 4. SAC 最小训练配置

| 参数 | 设置 |
|---|---|
| 算法 | SAC |
| Policy | MlpPolicy |
| 总训练步数 | 20,000 |
| Horizon | 200 |
| Control frequency | 20 Hz |
| Reward shaping | True |
| Buffer size | 100,000 |
| Batch size | 256 |
| Learning rate | 3e-4 |

训练时间约 5 分钟，成本较低，便于快速验证闭环。

---

## 5. 实验结果

### 5.1 训练过程

- Total timesteps: 20,000  
- Episode reward 平均值：10.6 （比随机 baseline 3.75 高）  
- Actor loss / Critic loss：-38.8 / 0.0108  
- Success rate：0（短期训练未完成完整抓取）

### 5.2 评估

| Episode | Total reward | Success |
|---|---|---|
| 0 | 51.310 | False |
| 1 | 50.940 | False |
| 2 | 51.373 | False |
| 3 | 54.362 | False |
| 4 | 45.888 | False |

- Average reward: 50.775  
- Success rate: 0.00  

### 5.3 Reward 曲线

![Reward Curve](../models/sac_panda_lift_minimal/reward_curve.png)

- 可以看到 reward 明显高于随机 baseline，说明策略学到阶段性行为；
- Success rate 为 0，但已学会靠近、抓取尝试等动作。

---

## 6. 工程取舍与分析

- **Reward 上升说明策略有学习能力**，即使 success 为 0，也体现 RL 封装和训练闭环成功；
- **不必长时间训练**：20k steps 成本低，已经可以作为求职作品集亮点；
- **展示重点**：
  1. Panda Lift 环境 Gym 封装；
  2. Observation/Action/Reward 封装方法；
  3. 随机 baseline vs 手工策略 vs SAC reward 曲线对比；
  4. 工程取舍判断：短训练就能展示 RL 闭环能力，不追求 full success。

---

## 7. 总结

本实验完成了 Panda Lift 最小 RL 训练闭环验证：

- 环境封装成功，Gym 风格接口可用；
- SAC 算法能在低维 observation 下学习阶段性动作；
- Reward curve 显示策略学到有价值行为；
- 工程取舍判断明确，不必长时间训练即可形成作品集展示。

> 成果可以直接作为 Week3 Panda Lift RL 实验亮点，放入项目 README / 求职作品集。