# Week 05 Task B：Panda Lift BC 策略评估与失败分析

## 1. 实验目标

本任务对 Panda Lift Behavior Cloning 策略进行评估与失败分析，重点不是继续训练模型，也不是整理 README，而是通过对比不同策略的 rollout 结果，判断 BC policy 学到了什么、没有学到什么，以及失败主要发生在哪个阶段。

本次对比了三类策略：

* Random policy：随机动作基线，用于判断任务是否能被随机动作偶然完成。
* Handcrafted policy：手工分阶段策略，同时也是 BC 数据采集时的 teacher / expert。
* BC policy：通过 Behavior Cloning 训练得到的模仿策略。

评估重点包括：

* 是否成功完成 Lift；
* 方块是否被实际抬起；
* 末端是否接近方块；
* 策略是否学到“接近、闭爪、抬升”的阶段性动作；
* 失败是否来自 BC 模型本身，还是来自 teacher 策略质量和抓取稳定性。

---

## 2. Random Policy 评估结果

随机策略共评估 5 个 episode，结果为：

```text
Random Evaluation:
Success episodes: 0/5
```

从输出结果看，随机策略在所有 episode 中都没有完成 Panda Lift。部分 episode 中，末端可能会偶然接近方块，例如 `min_eef_cube_dist` 能达到约 `0.085 m`，但方块高度基本没有上升：

```text
lift_height = 0.000
```

这说明随机策略无法稳定形成“靠近—抓取—抬升”的操作链路。即使 reward 在某些时刻有波动，也主要来自 reward shaping 中的接近奖励，并不代表任务成功。

因此，Random policy 可以作为最低基线，证明 Panda Lift 不是随机动作能够轻易完成的任务。

---

## 3. Handcrafted Policy 评估结果

手工策略评估结果为：

```text
Average total reward: 142.102
Average success steps: 0.000
Conclusion: handcrafted policy has not reached success yet.
```

从 rollout 过程看，handcrafted policy 能够完成较明确的阶段行为：

```text
phase 0: 移动到方块上方
phase 1: 下降到抓取位置
phase 2: 闭合夹爪
phase 3: 抬升
```

但是在 phase 3 中，末端执行了抬升动作，方块并没有稳定跟随上升。例如：

```text
step=160 | phase=2 | eef_z≈0.855 | cube_z≈0.825
step=180 | phase=3 | eef_z≈1.005 | cube_z≈0.820
step=200 | phase=3 | eef_z≈1.055 | cube_z≈0.820
```

这说明手工策略能让末端靠近方块并获得较高 reward，但没有稳定建立有效 grasp。也就是说，teacher 本身并不是一个稳定成功的 expert。

这对 BC 结果解释很重要：BC policy 是基于该 handcrafted policy 采集的数据训练的，因此 BC 的表现上限会受到 teacher 策略质量影响。

---

## 4. BC Policy 正式评估结果

BC policy 正式评估 5 个 episode，结果为：

```text
BC Evaluation:
Success episodes: 0/5
```

从动作输出看，BC policy 并不是随机乱动，而是明显学到了手工策略中的阶段行为。

在 phase 0，BC policy 输出开爪并接近方块的动作：

```text
action[-1] ≈ -0.996
```

在 phase 2，BC policy 输出闭合夹爪动作：

```text
action[-1] ≈ 0.998
```

在 phase 3，BC policy 保持闭爪，并尝试进入抬升阶段。

但是在正式 5 次评估中，BC policy 均未成功完成任务。典型失败表现是：策略已经进入 phase 3，但 z 方向抬升动作较小，或者没有形成有效抓取，导致方块没有跟随末端上升。例如：

```text
phase=3 | action[2]≈0.07 或接近 0
success=False
```

因此，正式评估结果说明 BC policy 目前还不是一个稳定成功的 Panda Lift 策略。

---

## 5. BC Policy 诊断评估结果

为了进一步判断 BC 失败阶段，额外运行了诊断脚本，记录 `eef_z`、`cube_z`、`lift`、`min_eef_cube_dist` 和 gripper action。

诊断结果为：

```text
BC Diagnose:
Success episodes: 1/3
```

其中两个 episode 失败：

```text
max_lift=0.000
min_eef_cube_dist≈0.024~0.025
success=False
```

这说明失败 episode 中，BC policy 已经能让末端接近方块，但没有形成有效抓取。末端进入抬升阶段后，方块没有跟随上升。

另一个 episode 成功完成了明显抬升：

```text
step=175 | success=True | cube_z=0.977 | lift=0.147
step=200 | success=True | cube_z=1.286 | lift=0.456
step=225 | success=True | cube_z=1.588 | lift=0.758
step=250 | success=True | cube_z=1.788 | lift=0.957
step=275 | success=True | cube_z=1.905 | lift=1.075
```

该 episode 最终统计为：

```text
success=True
max_lift=1.077
min_eef_cube_dist=0.023
```

这说明 BC policy 并不是完全没有学会 Lift。它在部分初始条件和接触状态下，能够成功抓住方块并将其抬起。

---

## 6. 失败阶段分析

综合 Random、Handcrafted 和 BC 的结果，BC policy 的主要失败点不是“完全不会动”，也不是“没有学到方块位置”，而是发生在：

```text
phase 2 闭合夹爪 → phase 3 抬升
```

具体表现为：

1. BC 能接近方块
   诊断中 `min_eef_cube_dist` 可以达到约 `0.023~0.025 m`，说明末端已经接近方块。

2. BC 能模仿夹爪动作
   phase 2 中 gripper action 接近 `0.998`，说明模型学到了闭爪行为。

3. BC 能进入抬升阶段
   phase 3 中策略会保持闭爪，并尝试输出 z 方向抬升动作。

4. BC 的抓取稳定性不足
   多数失败 episode 中，末端虽然抬起，但方块没有跟随上升，说明夹爪与方块之间没有形成稳定 grasp。

5. BC 在部分 episode 中可以成功 lift
   诊断评估中出现了 `max_lift=1.077` 的成功案例，说明 BC 具备一定的任务能力，但成功率不稳定。

---

## 7. 当前结论

本阶段实验说明，BC policy 相比 random policy 已经有明显进步。它不是随机动作，而是学到了 handcrafted policy 中的阶段性操作流程，包括接近方块、下压、闭爪和抬升。

但是，BC policy 还不是稳定成功的抓取策略。正式评估中 5 个 episode 成功率为 0，而诊断评估中 3 个 episode 有 1 个成功。该结果说明 BC policy 具备一定 lift 能力，但对初始位置、接触状态和抓取阶段非常敏感。

当前最主要的问题不是 BC 没有学到动作流程，而是 teacher 本身质量有限，且 BC 在“闭爪接触到抬升”之间的抓取稳定性不足。因此，后续如果要提升 BC 表现，优先方向应该是提高 expert 数据质量，而不是单纯增加网络训练轮数。

---

## 8. 后续改进方向

后续可以从以下方向继续改进：

1. 改进 handcrafted teacher
   让 teacher 本身能够更稳定地完成抓取和抬升，再重新采集 BC 数据。

2. 增加成功示范数据比例
   当前 BC 可能学习到了大量“靠近但未成功抓取”的轨迹，后续应提高成功 grasp 轨迹占比。

3. 记录并分析失败类型
   将失败分成：未靠近、靠近但未下压、下压但未夹住、夹住但未抬起、抬起后掉落。

4. 引入更多诊断指标
   除 reward 和 success 外，继续记录 `cube_z`、`eef_z`、`lift_height`、`min_eef_cube_dist` 和 gripper action。

5. 后续再考虑 RL fine-tuning
   在 BC policy 已经具备基本接近和抓取动作后，可以考虑用 RL 在 BC 初始化策略基础上继续优化抓取稳定性。

---

## 9. 本阶段总结

本任务完成了 Panda Lift BC policy 的初步评估与失败分析。实验结果表明：

```text
Random policy: 不能完成任务。
Handcrafted policy: 能靠近方块，但 expert 本身不稳定。
BC policy: 学到了阶段性动作，并在部分 episode 中能成功 lift，但整体成功率不稳定。
```

因此，当前 BC 策略的主要价值在于证明模型已经学到了基本操作结构；当前主要问题在于抓取稳定性和 expert 数据质量，而不是模型完全没有学习到任务行为。

## Week 5 Task C Baseline Evaluation

Current model:

- `models/bc/panda_lift_bc_policy.pt`

Evaluation setting:

- Episodes: 3
- Diagnostic script: `src/eval_panda_lift_bc_diagnose.py`

Results:

| Episode | Total Reward | Success | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---|---:|---:|---:|
| 0 | 51.942 | False | 0.000 | -0.010 | 0.024 |
| 1 | 81.184 | True | 0.236 | -0.010 | 0.024 |
| 2 | 47.887 | False | 0.000 | -0.010 | 0.024 |

Summary:

- Success rate: 1/3
- Average reward: approximately 60.34
- Average max lift: approximately 0.079
- The policy can move the end-effector close to the cube, with minimum distance around 0.024 m.
- The main failure is not the reaching phase, but the grasping and lifting phase.
- In failed episodes, the gripper closes near the cube, but the cube is not reliably lifted.
- This suggests that the current single-trajectory BC policy has limited robustness around contact, grasp closure, and lift execution.

Initial diagnosis:

The baseline BC policy has learned part of the reaching behavior, but the grasp-lift behavior is unstable. This supports the next experiment: collecting multiple trajectories to improve state coverage, especially around the grasping and lifting phases.


## Handcrafted Teacher Policy Diagnosis

Before improving the BC dataset, the handcrafted policy used as the teacher was evaluated.

Evaluation command:

- `python src/run_panda_lift_handcrafted_policy.py`

Results:

| Episode | Total Reward | Success Steps | Observation |
|---|---:|---:|---|
| 0 | 194.764 | 0 | Cube was lifted very high, but robosuite success was not triggered. |
| 1 | 79.860 | 0 | Cube was briefly lifted, then dropped during the lift phase. |
| 2 | 193.357 | 0 | Cube was lifted very high, but success was still not triggered. |

Summary:

- Average total reward: 155.994
- Average success steps: 0.000
- The handcrafted teacher can reach the cube and often lift it, but the success condition is not reliably satisfied.
- This means the current BC dataset should not be treated as clean expert demonstrations.
- The BC failure is likely related not only to model generalization, but also to teacher/data quality.

Implication for Task C:

Before adding data augmentation, the data collection pipeline should be improved to save episode-level information such as success, total reward, max lift, final lift, and episode_id. This will make it possible to filter or compare trajectories by quality.

