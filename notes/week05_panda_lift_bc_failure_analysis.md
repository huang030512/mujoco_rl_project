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


## Multi-Trajectory Data Quality Check

A small-scale multi-trajectory dataset was collected using the new metric-aware data collection script.

Command:

- `python src/collect_panda_lift_bc_data_with_metrics.py --num-episodes 5 --horizon 300 --output data/panda_lift_bc_data_with_metrics_5ep.npz`

Saved data:

- Observations shape: `(1500, 15)`
- Actions shape: `(1500, 7)`
- Episode ids shape: `(1500,)`

Results:

| Episode | Total Reward | Success | Success Steps | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---|---:|---:|---:|---:|
| 0 | 68.740 | False | 0 | 0.104 | -0.009 | 0.024 |
| 1 | 56.718 | False | 0 | 0.000 | -0.010 | 0.024 |
| 2 | 77.602 | False | 0 | 0.189 | -0.010 | 0.024 |
| 3 | 61.266 | False | 0 | 0.025 | -0.010 | 0.024 |
| 4 | 61.353 | False | 0 | 0.029 | -0.010 | 0.024 |

Summary:

- Success episodes: 0/5
- Average total reward: 65.136
- Average max lift: 0.069
- The teacher policy consistently reaches the cube, as shown by the minimum end-effector-to-cube distance of about 0.024 m.
- However, none of the 5 trajectories satisfy the success condition.
- Most trajectories fail during the grasping and lifting stage.
- The final lift values are close to zero or negative, meaning the cube is not stably held at the end of the episode.

Diagnosis:

The current multi-trajectory dataset should not be treated as clean expert demonstrations. The main bottleneck is teacher/data quality rather than the lack of behavior cloning model capacity. Before training a new BC model or adding data augmentation, the next step should be to improve the data collection pipeline by either filtering high-quality trajectories or improving the handcrafted teacher around grasp closure and lift stability.


## Handcrafted Policy Failure Cause Analysis

After inspecting `src/run_panda_lift_handcrafted_policy.py`, the main reason for unstable grasping and lifting is that the handcrafted teacher uses a simple phase-based control logic rather than a true grasp-state-aware policy.

Key findings:

- Phase 0/1 mainly perform position-based reaching, which is relatively easy because the policy directly uses `eef_pos`, `cube_pos`, and `gripper_to_cube_pos`.
- The policy can consistently move the end-effector close to the cube, which explains why both the teacher and BC model achieve small EEF-cube distance.
- However, the transition from grasping to lifting is mainly based on step count instead of checking whether the cube has actually been grasped.
- The policy does not verify stable contact, gripper closure state, cube lift state, or whether the cube is following the gripper.
- During the lift phase, the end-effector may move upward even when the cube has not been securely grasped.
- The lift action may also introduce horizontal disturbance because the target is still computed in xyz space rather than using a strictly stable upward motion.
- If the cube is briefly lifted and then slips, the policy does not detect or recover from this failure.
- The lifting speed may be too aggressive for an unstable grasp.

Diagnosis:

The handcrafted policy is useful as an interface validation tool and as a rough demonstration generator, but it is not yet a reliable expert policy. Its main weakness is not reaching, but grasp confirmation and stable lift execution.

Implication:

For Task C, directly training a new multi-trajectory BC model or adding data augmentation on top of this teacher data is not the best next step. The priority should be to improve the teacher policy or add trajectory filtering before treating the collected demonstrations as expert data.


## Handcrafted Teacher V2 Trial and Failure Analysis

A preliminary improved teacher policy was created as:

- `src/run_panda_lift_handcrafted_policy_v2.py`

The goal of V2 was to improve grasp and lift stability by using more conservative phase transitions, adding grasp waiting time, reducing XY disturbance during lifting, and avoiding video saving.

Evaluation results:

| Episode | Total Reward | Success Steps | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---:|---:|---:|---:|
| 0 | 29.012 | 0 | 0.000 | -0.010 | 0.031 |
| 1 | 29.372 | 0 | 0.000 | -0.010 | 0.030 |
| 2 | 29.079 | 0 | 0.000 | -0.010 | 0.031 |

Summary:

- Success episodes: 0/3
- Average total reward: 29.155
- Average max lift: 0.000
- Average min EEF-cube distance: about 0.031 m

Diagnosis:

V2 did not improve the teacher policy. It performed worse than V1 because the policy became too conservative. Although the end-effector still moved near the cube, the cube was never lifted. This suggests that V2 may either fail to enter the grasp/lift phases reliably, or enter them from a poor grasp pose.

Possible causes:

- Phase transition conditions may be too strict, especially from the descending/alignment phase to the grasp phase.
- The grasp height may be too low or unsuitable for the Panda gripper geometry.
- Locking the XY position at the start of grasping may preserve a small alignment error instead of correcting it.
- The lift phase may be too slow or may start before a valid grasp is formed.
- A small EEF-cube distance alone does not guarantee a valid grasp.

Next diagnostic direction:

Before further changing the policy, the next step should be to add explicit phase transition logging, including XY distance, Z distance, EEF-cube distance, current phase, and phase step. This will show whether V2 fails because it cannot enter the correct phase, or because it enters the phase but closes the gripper at a bad pose.


## Teacher V2 Phase-2 Z-Control Improvement

After diagnosing V2, the phase-2 grasp stage was found to have weak downward z control. The policy entered the grasp/lift phases, but `action[2]` was originally too small, so the end-effector did not move sufficiently toward the desired grasp height.

A minimal change was made to strengthen the phase-2 downward z action while keeping the rest of the policy unchanged.

Key diagnostic comparison:

- Before change: `action[2]` was around `-0.022` during phase 2.
- After change: `action[2]` was strengthened to around `-0.080` during phase 2.

Evaluation results after the change:

| Episode | Total Reward | Success Steps | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---:|---:|---:|---:|
| 0 | 78.264 | 0 | 0.080 | -0.011 | 0.025 |
| 1 | 184.724 | 0 | 0.171 | 0.171 | 0.025 |
| 2 | 43.416 | 0 | 0.000 | -0.010 | 0.027 |

Summary:

- Success episodes: 0/3
- Average total reward: 102.134
- Average max lift: 0.084
- Average min EEF-cube distance: 0.026

Diagnosis:

Strengthening the phase-2 downward z control clearly improved the teacher policy. Compared with the initial V2 trial, the average reward increased and the average max lift became positive. In Episode 1, the cube was lifted and remained elevated at the end of the episode.

However, the policy is still not stable enough to be treated as a clean expert. Some episodes still fail to lift the cube, and success_steps remains zero. This suggests that the direction of improvement is correct, but the handcrafted teacher still requires further refinement before collecting high-quality BC demonstrations.


## Success Metric Fix and V2 Re-evaluation

During V2 evaluation, an inconsistency was found: some episodes had high `max_lift` and `final_lift`, but `success_steps` remained zero.

Code inspection showed that `env.step(action)` in the current robosuite setup may return an empty `info` dict, so using only `info.get("success", False)` is unreliable.

The success check was updated to use both `info` and the environment's internal success function:

- `success_from_info = bool(info.get("success", False))`
- `success_from_env = bool(env._check_success())`
- `success = success_from_info or success_from_env`

This correction was applied to:

- `src/run_panda_lift_handcrafted_policy_v2.py`
- `src/collect_panda_lift_bc_data_with_metrics.py`

V2 re-evaluation after fixing success statistics:

| Episode | Total Reward | Success Steps | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---:|---:|---:|---:|
| 0 | 190.421 | 143 | 0.173 | 0.173 | 0.026 |
| 1 | 43.319 | 0 | 0.000 | -0.010 | 0.028 |
| 2 | 44.254 | 0 | 0.000 | -0.010 | 0.026 |

Summary:

- Success episodes: 1/3
- Average total reward: 92.665
- Average max lift: 0.058
- Average min EEF-cube distance: 0.027

Diagnosis:

The previous `success_steps=0` result was partly caused by an unreliable success source. After using `env._check_success()`, V2 correctly reports success when the cube is lifted above the robosuite Lift success threshold.

However, V2 is still not a stable expert policy. It can complete the task in some episodes, but still fails in others. Therefore, it is better to continue treating V2 as an improved but imperfect teacher rather than as a clean expert demonstration source.


## Multi-Trajectory Data Quality Recheck After Success Fix

After fixing the success metric to use `env._check_success()`, the 5-episode metric-aware data collection was repeated.

Command:

- `python src/collect_panda_lift_bc_data_with_metrics.py --num-episodes 5 --horizon 300 --output data/panda_lift_bc_data_with_metrics_5ep_recheck.npz`

Saved data:

- Observations shape: `(1500, 15)`
- Actions shape: `(1500, 7)`
- Episode ids shape: `(1500,)`

Results:

| Episode | Total Reward | Success | Success Steps | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---|---:|---:|---:|---:|
| 0 | 192.133 | True | 134 | 1.079 | -0.015 | 0.023 |
| 1 | 59.187 | False | 0 | 0.001 | -0.010 | 0.024 |
| 2 | 192.791 | True | 136 | 1.066 | 0.219 | 0.022 |
| 3 | 192.280 | True | 135 | 1.056 | -0.017 | 0.023 |
| 4 | 192.619 | True | 135 | 1.068 | -0.012 | 0.023 |

Summary:

- Success episodes: 4/5
- Average total reward: 165.802
- Average max lift: 0.854
- The previous `0/5` success result was caused by unreliable success reading from `info`.
- After using `env._check_success()`, the original teacher can often trigger robosuite's Lift success condition.
- However, several successful episodes still have negative final lift, which means the cube was lifted during the episode but was not held stably until the end.
- Therefore, the teacher data should be considered partially successful but not perfectly stable.

Updated diagnosis:

The main issue is no longer that the teacher cannot trigger success. Instead, the teacher can often lift the cube, but grasp retention is unstable. For BC training, this means trajectory quality should not be judged by success alone. `max_lift`, `final_lift`, and `success_steps` should be considered together.


## 10-Episode Multi-Trajectory Dataset Quality Analysis

A 10-episode metric-aware BC dataset was collected for multi-trajectory analysis.

Command:

- `python src/collect_panda_lift_bc_data_with_metrics.py --num-episodes 10 --horizon 300 --output data/panda_lift_bc_data_with_metrics_10ep.npz`

Saved data:

- Observations shape: `(3000, 15)`
- Actions shape: `(3000, 7)`
- Episode ids shape: `(3000,)`

Episode-level quality table:

| Episode | Total Reward | Success | Success Steps | Max Lift | Final Lift | Min Dist | Quality |
|---|---:|---|---:|---:|---:|---:|---|
| 0 | 191.367 | True | 132 | 1.083 | -0.012 | 0.023 | B_success_but_dropped |
| 1 | 60.342 | False | 0 | 0.000 | -0.010 | 0.024 | D_failed |
| 2 | 190.762 | True | 134 | 1.066 | -0.016 | 0.023 | B_success_but_dropped |
| 3 | 52.050 | False | 0 | 0.000 | -0.010 | 0.024 | D_failed |
| 4 | 191.780 | True | 134 | 1.067 | -0.382 | 0.023 | B_success_but_dropped |
| 5 | 65.832 | True | 8 | 0.079 | -0.012 | 0.024 | C_weak_success |
| 6 | 193.897 | True | 136 | 1.068 | 0.094 | 0.022 | A_good |
| 7 | 51.765 | False | 0 | 0.000 | -0.010 | 0.024 | D_failed |
| 8 | 49.039 | False | 0 | 0.000 | -0.010 | 0.024 | D_failed |
| 9 | 101.927 | True | 44 | 0.483 | -0.010 | 0.024 | C_weak_success |

Summary:

- Success episodes: 6/10
- Average reward: 114.876
- Average max lift: 0.485
- Average final lift: -0.038

Quality distribution:

- `A_good`: Episode 6
- `B_success_but_dropped`: Episodes 0, 2, 4
- `C_weak_success`: Episodes 5, 9
- `D_failed`: Episodes 1, 3, 7, 8

Diagnosis:

The 10-episode dataset contains useful demonstrations but is not uniformly high quality. Only one episode both achieves success and maintains a positive final lift. Several episodes trigger success but drop the cube later, and four episodes fail completely.

Implication for BC training:

The next BC experiment should not blindly train on all collected trajectories. A better next step is to create a filtered dataset that excludes clearly failed episodes and compares it against the unfiltered multi-trajectory dataset.


## Success-Filtered Dataset Generation

A trajectory filtering script was added to create a success-filtered BC dataset from the 10-episode metric-aware dataset.

Script:

- `src/filter_panda_lift_bc_data_by_quality.py`

Input:

- `data/panda_lift_bc_data_with_metrics_10ep.npz`

Output:

- `data/panda_lift_bc_data_filtered_success_10ep.npz`

Filtering rule:

- Keep episodes where `success=True`.

Kept episodes:

| Episode | Success | Success Steps | Max Lift | Final Lift |
|---|---|---:|---:|---:|
| 0 | True | 132 | 1.083 | -0.012 |
| 2 | True | 134 | 1.066 | -0.016 |
| 4 | True | 134 | 1.067 | -0.382 |
| 5 | True | 8 | 0.079 | -0.012 |
| 6 | True | 136 | 1.068 | 0.094 |
| 9 | True | 44 | 0.483 | -0.010 |

Summary:

- Original episodes: 10
- Kept episodes: 6
- Removed failed episodes: 1, 3, 7, 8
- Expected filtered samples: 6 episodes × 300 steps = 1800 samples

Diagnosis:

The success-filtered dataset removes clearly failed trajectories, but it is still not perfectly clean. Several kept trajectories triggered success during the episode but had negative final lift, meaning the cube was lifted and then dropped. Therefore, this dataset is suitable for a first filtered BC experiment, but it should not be interpreted as a fully high-quality expert dataset.

Experiment role:

This dataset will be used as the C group in the BC comparison:

- A group: existing baseline BC
- B group: unfiltered 10-episode multi-trajectory BC
- C group: success-filtered 10-episode multi-trajectory BC


## BC Training and Evaluation: Unfiltered vs Success-Filtered Data

Two additional BC policies were trained using the 10-episode metric-aware dataset.

Training setup:

- B group: unfiltered 10-episode dataset
  - Data: `data/panda_lift_bc_data_with_metrics_10ep.npz`
  - Model: `models/bc/panda_lift_bc_multitraj_10ep.pt`
  - Best validation loss: `0.000263`

- C group: success-filtered 10-episode dataset
  - Data: `data/panda_lift_bc_data_filtered_success_10ep.npz`
  - Model: `models/bc/panda_lift_bc_filtered_success_10ep.pt`
  - Best validation loss: `0.000484`

Evaluation setting:

- Script: `src/eval_panda_lift_bc_diagnose.py`
- Episodes: 3
- Success check: `info["success"]` OR `env._check_success()`

### B Group: Unfiltered 10-Episode BC

| Episode | Total Reward | Success | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---|---:|---:|---:|
| 0 | 63.119 | True | 0.025 | -0.011 | 0.025 |
| 1 | 44.771 | False | 0.000 | -0.010 | 0.028 |
| 2 | 74.759 | True | 0.156 | -0.010 | 0.025 |

Summary:

- Success episodes: 2/3
- Average reward: 60.883
- Average max lift: 0.060
- Average final lift: -0.010

### C Group: Success-Filtered 10-Episode BC

| Episode | Total Reward | Success | Max Lift | Final Lift | Min EEF-Cube Dist |
|---|---:|---|---:|---:|---:|
| 0 | 48.881 | False | 0.000 | -0.010 | 0.025 |
| 1 | 60.030 | False | 0.000 | -0.010 | 0.024 |
| 2 | 3.895 | False | 0.000 | -0.010 | 0.126 |

Summary:

- Success episodes: 0/3
- Average reward: 37.602
- Average max lift: 0.000
- Average final lift: -0.010

### Comparison with Baseline

| Group | Data | Success Episodes | Main Observation |
|---|---|---:|---|
| A | Original baseline BC data | 1/3 | Can reach the cube and occasionally lift, but unstable. |
| B | Unfiltered 10-episode data | 2/3 | Improves success rate, but final lift remains negative. |
| C | Success-filtered 10-episode data | 0/3 | Filtering by success alone hurts performance. |

Diagnosis:

The unfiltered multi-trajectory dataset improves rollout success compared with the baseline, but it still does not produce stable final grasp retention. The success-filtered dataset performs worse, showing that `success=True` alone is not a sufficient criterion for high-quality BC demonstrations.

A likely reason is that many success-filtered trajectories are not truly clean expert trajectories. Several kept trajectories triggered success during the episode but dropped the cube before the end. Filtering by success also reduces data diversity and removes some reaching behavior coverage, which may explain why the filtered policy performs worse.

Conclusion:

For Panda Lift BC, trajectory quality cannot be judged only by environment success. More reliable filtering should combine:

- `success=True`
- sufficiently large `success_steps`
- high `max_lift`
- positive `final_lift`
- stable end-of-episode grasp retention

