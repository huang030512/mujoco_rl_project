# Week 2 Day 1 - MJCF 文件结构理解

## qpos 与 qvel 初步验证

运行 `models/mjcf_structure_demo.xml` 后得到：

- `nq = 17`
- `nv = 15`

模型中的关节与状态变量数量对应如下：

| 关节所属 body | 关节类型 | qpos 数量 | qvel 数量 |
| --- | --- | ---: | ---: |
| free_box | freejoint | 7 | 6 |
| hinge_rod | hinge | 1 | 1 |
| slide_block | slide | 1 | 1 |
| parent_body | free | 7 | 6 |
| child_body | hinge | 1 | 1 |

其中，`freejoint` 的 `qpos` 由 3 个位置坐标和 4 个四元数姿态变量组成，因此为 7；其 `qvel` 由 3 个平移速度和 3 个旋转速度组成，因此为 6。

因此：

- `nq = 7 + 1 + 1 + 7 + 1 = 17`
- `nv = 6 + 1 + 1 + 6 + 1 = 15`
## MJCF 文件标签说明

### 1. `<mujoco>` 根标签
- 所有模型的根节点
- 属性：
  - `model`：模型名称

### 2. `<compiler>`
- 用于定义角度单位、精度等编译参数
- 常用属性：
  - `angle="degree"` 或 `"radian"`
  - `coordinate`、`inertiafromgeom` 等

### 3. `<option>`
- 定义仿真参数
- 常用属性：
  - `timestep`：仿真步长（秒）
  - `gravity`：重力 `[x y z]`

### 4. `<worldbody>`
- 世界坐标系下的顶层容器
- 包含所有刚体 `<body>` 或 `<geom>`（静态物体）

### 5. `<body>`
- 刚体或机械臂连杆
- 常用属性：
  - `name`：body 名称
  - `pos`：相对于父 body 的位置 `[x y z]`
- 可以嵌套子 `<body>`，形成父子层次结构

### 6. `<geom>`
- 碰撞与可视几何
- 常用属性：
  - `type`：几何类型 `box` / `sphere` / `capsule` / `plane`
  - `size`：半尺寸 `[x y z]` 或单值
  - `mass`：质量（kg）
  - `rgba`：颜色 `[r g b alpha]`

### 7. `<joint>` 与 `<freejoint>`
- `<joint>`：
  - 类型通过 `type` 属性定义：`hinge` / `slide` / `free` 等
  - 单轴旋转或滑动，轴向通过 `axis` 定义
- `<freejoint>`：
  - 六自由度（位置 + 四元数姿态）
  - 不需要 `type` 属性，MuJoCo 自动识别
- 对应 qpos / qvel：
  - hinge / slide → 1 个 qpos + 1 个 qvel
  - freejoint → 7 个 qpos + 6 个 qvel

### 8. `<actuator>`
- 驱动器标签
- 可用于控制关节
- 属性：
  - `joint`：驱动哪个关节
  - `ctrlrange`：控制范围
  - `gain`：增益等
- 本练习中暂留空，后续模型控制学习时再补充
