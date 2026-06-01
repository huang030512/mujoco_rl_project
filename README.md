# MuJoCo RL Project

## 项目目标

本项目用于建立面向机器人学习与具身智能方向的 MuJoCo + 强化学习实验框架。当前阶段重点完成可复现开发环境、规范化代码仓库与实验记录流程，后续将逐步加入 MuJoCo 仿真模型、控制算法、强化学习训练和评估展示。

## 当前进展

- [x] 创建 Python 独立环境
- [x] 安装并测试 MuJoCo
- [x] 建立项目目录结构
- [x] 初始化 Git 仓库
- [x] 建立依赖与环境信息记录规范
- [ ] 加载第一个 MuJoCo 模型
- [ ] 实现基础控制示例
- [ ] 搭建强化学习训练流程

## 项目结构

```text
mujoco_rl_project/
├── src/          # 核心程序、训练与测试入口
├── envs/         # 自定义强化学习环境
├── models/       # MuJoCo XML / URDF 模型文件
├── configs/      # 实验配置文件
├── results/      # 指标、日志、视频和模型输出
├── assets/       # README 展示图片、GIF 和精选视频
├── docs/         # 环境配置与项目文档
├── notes/        # 每日学习记录与实验复盘
├── .gitignore
├── requirements.txt
└── README.md
```
