# MuJoCo RL Project

## 项目目标

本项目面向机器人学习与具身智能方向，逐步建立 MuJoCo + Python 仿真实验基础，并为后续强化学习控制任务准备可复现的模型、代码与实验记录。

## 第一周成果：自由落体仿真演示

第一周完成了 MuJoCo 开发环境验证、简单模型加载、模型结构与状态字段检查，以及自由落体仿真结果的记录和可视化分析。

通过本周工作，已经建立从 MJCF 模型定义、MuJoCo 模型加载、仿真状态读取、时间步推进到实验结果绘图分析的基础流程。

### 已完成内容

- 环境验证脚本：`src/check_mujoco.py`
- 自由落体模型：`models/simple_box.xml`
- 基础仿真脚本：`src/run_simple_box.py`
- 状态曲线绘图脚本：`src/simulate_box_plot.py`
- 模型结构与状态字段检查脚本：`src/inspect_model.py`
- `MjModel` 与 `MjData` 学习记录：`notes/week01_day04_mjmodel_mjdata.md`
- 仿真结果分析记录：`notes/week01_day05_simulation_analysis.md`
- 第一周复盘总结：`notes/week01_review.md`
- 位置变化曲线：`assets/week01_box_free_fall_position.png`
- 速度变化曲线：`assets/week01_box_free_fall_velocity.png`

## 项目结构

```text
mujoco_rl_project/
├── README.md
├── assets/
│   ├── week01_box_free_fall_position.png
│   └── week01_box_free_fall_velocity.png
├── models/
│   └── simple_box.xml
├── notes/
│   ├── week01_day04_mjmodel_mjdata.md
│   ├── week01_day05_simulation_analysis.md
│   └── week01_review.md
└── src/
    ├── check_mujoco.py
    ├── inspect_model.py
    ├── run_simple_box.py
    └── simulate_box_plot.py
````

## 运行方式

建议按以下顺序执行：检查环境、检查模型结构、运行自由落体仿真、生成结果曲线。

### 1. 检查 MuJoCo 环境

在项目根目录下执行：

```bash
python src/check_mujoco.py
```

该脚本用于确认 Python、MuJoCo 与 NumPy 环境能够正常运行。

### 2. 检查模型结构与状态字段

```bash
python src/inspect_model.py
```

该脚本加载 `models/simple_box.xml`，用于查看模型中的 body、joint、geom、actuator，以及 `qpos`、`qvel`、`ctrl` 等状态字段。

在当前模型中，盒子使用自由关节 `<freejoint/>`，因此模型状态包含：

* `nq = 7`：3 个平移位置分量和 4 个四元数姿态分量；
* `nv = 6`：3 个线速度分量和 3 个角速度分量；
* `nu = 0`：当前自由落体模型未配置 actuator。

### 3. 运行自由落体仿真

```bash
python src/run_simple_box.py
```

该脚本加载 `models/simple_box.xml`，创建仿真状态对象，并通过循环调用 `mj_step(model, data)` 推进盒子自由落体过程。

### 4. 生成自由落体结果曲线

```bash
python src/simulate_box_plot.py
```

运行后生成以下图像：

* `assets/week01_box_free_fall_position.png`：盒子 z 方向位置随时间变化曲线；
* `assets/week01_box_free_fall_velocity.png`：盒子 z 方向速度随时间变化曲线。

其中：

* `qpos[2]` 表示盒子在 z 轴方向的位置，可用于观察高度变化；
* `qvel[2]` 表示盒子在 z 轴方向的速度，负值表示盒子正在向下运动。

## 第一周技术链路

```text
models/simple_box.xml
        ↓
MuJoCo 加载 XML 模型
        ↓
MjModel 保存模型结构和固定参数
MjData 保存运行过程中的动态状态
        ↓
mj_step(model, data) 推进仿真
        ↓
读取 qpos[2] 和 qvel[2]
        ↓
绘制盒子竖直位置与竖直速度曲线
        ↓
分析自由落体、触地、短暂反弹和最终稳定现象
```

`models/simple_box.xml` 定义了仿真中的地面、盒子、质量、尺寸、重力和自由关节等内容。MuJoCo 加载该模型后，使用 `MjModel` 保存模型结构与固定参数，并使用 `MjData` 保存运行过程中的动态状态。程序通过持续调用 `mj_step()` 更新仿真状态，再读取盒子的竖直位置和速度并绘制结果曲线。

## 第一周实验现象

在自由落体实验中，盒子从约 `0.5 m` 的初始高度释放，在重力作用下沿 z 轴负方向下落。

根据实际曲线观察结果：

* 盒子第一次触地时间约为 `0.25 s`；
* 最终稳定高度约为 `0.1 m`；
* 盒子接触地面后出现了短暂反弹或振荡现象；
* 随着仿真继续推进，盒子高度逐渐稳定，竖直速度最终趋近于零。

触地后的短暂反弹是由于盒子接触地面时仍具有向下速度，地面接触作用阻止盒子继续穿透，从而使竖直速度短暂发生方向变化。

## 第一周复盘文档

第一周复盘总结记录了本周实际完成内容、核心概念理解、自由落体实验分析、遇到的问题及解决过程，以及第2周学习准备：

* `notes/week01_review.md`

## 可复现性说明

第一周成果具备基础可复现性，原因如下：

* `models/simple_box.xml` 固定定义了自由落体实验使用的模型结构和仿真参数；
* `src/` 目录中保存了环境检查、模型读取、仿真运行和绘图脚本；
* `assets/` 目录中保存了由实验生成的位置曲线和速度曲线；
* `notes/` 目录中保存了模型字段理解、实验结果分析和周复盘记录；
* 本 README 提供了从环境检查到结果生成的完整运行方式。

在配置相同 MuJoCo 环境后，其他人可以按照本 README 中的步骤重新运行脚本，并观察到同类型的盒子自由落体、触地反弹和最终稳定现象。

## 后续计划

下一阶段将进入 MJCF 建模基础学习，重点理解 `<body>`、`<geom>`、`<joint>` 与 `<actuator>` 等元素之间的关系，并逐步从简单自由落体模型过渡到具有关节和控制输入的自建机器人模型，为后续强化学习控制实验做好准备。

