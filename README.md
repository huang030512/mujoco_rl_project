# MuJoCo RL Project

## 项目目标

本项目面向机器人学习与具身智能方向，逐步建立 MuJoCo + Python 仿真实验基础，并为后续强化学习控制任务准备可复现的模型、代码与实验记录。

## 第一周成果：自由落体仿真演示

第一周完成了 MuJoCo 开发环境验证、简单模型加载、模型状态字段检查，以及自由落体仿真结果的记录与可视化分析。

### 已完成内容

- 环境验证脚本：`src/check_mujoco.py`
- 自由落体模型：`models/simple_box.xml`
- 基础仿真脚本：`src/run_simple_box.py`
- 状态曲线绘图脚本：`src/simulate_box_plot.py`
- 模型结构与状态字段检查脚本：`src/inspect_model.py`
- `MjModel` 与 `MjData` 学习记录：`notes/week01_day04_mjmodel_mjdata.md`
- 仿真结果分析记录：`notes/week01_day05_simulation_analysis.md`
- 位置变化曲线：`assets/week01_box_free_fall_position.png`
- 速度变化曲线：`assets/week01_box_free_fall_velocity.png`
- 展示材料：第一周自由落体仿真运行录屏或截图

## 项目结构

~~~text
mujoco_rl_project/
├── README.md
├── assets/
│   ├── week01_box_free_fall_position.png
│   └── week01_box_free_fall_velocity.png
├── models/
│   └── simple_box.xml
├── notes/
│   ├── week01_day04_mjmodel_mjdata.md
│   └── week01_day05_simulation_analysis.md
└── src/
    ├── check_mujoco.py
    ├── inspect_model.py
    ├── run_simple_box.py
    └── simulate_box_plot.py
~~~

## 运行方式

在项目根目录下执行以下命令。

### 1. 检查 MuJoCo 环境

~~~bash
python src/check_mujoco.py
~~~

### 2. 运行自由落体仿真

~~~bash
python src/run_simple_box.py
~~~

该脚本加载 `models/simple_box.xml`，执行盒子自由落体仿真，并输出关键状态变化。

### 3. 生成结果曲线

~~~bash
python src/simulate_box_plot.py
~~~

运行后生成以下图像：

- `assets/week01_box_free_fall_position.png`：盒子 z 方向位置随时间变化曲线；
- `assets/week01_box_free_fall_velocity.png`：盒子 z 方向速度随时间变化曲线。

### 4. 检查模型结构与状态字段

~~~bash
python src/inspect_model.py
~~~

该脚本用于查看模型中的 body、joint、geom、actuator，以及 `qpos`、`qvel`、`ctrl` 等状态字段。

## 第一周实验现象

在自由落体实验中，盒子由初始高度释放，在重力作用下沿 z 方向下落，并在接触地面后出现短暂回弹，最终稳定在地面上方。位置曲线和速度曲线能够反映下落、碰撞及稳定过程。

## 第一周成果说明

通过第一周工作，已建立基础 MuJoCo 仿真项目结构，能够完成模型加载、状态读取、自由落体仿真运行及结果可视化，并已通过重新运行脚本确认结果可以复现。

## 后续计划

后续将在现有可复现仿真基础上，进一步学习 MJCF 文件结构、执行器配置与基础控制方法，并逐步进入面向机器人控制和强化学习任务的实验开发。
