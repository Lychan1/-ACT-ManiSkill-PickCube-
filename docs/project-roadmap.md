# 具身智能项目实践路线

本路线面向以项目驱动方式学习具身智能的初学者。核心原则是先分别跑通数据、训练与仿真评估，再逐步增加视觉、真实硬件和语言条件，不在第一阶段强行整合所有框架。

## 状态总览

| 阶段 | 状态 | 验收成果 |
|---|---|---|
| ManiSkill PickCube state ACT | 已完成 | 100 demos、30,000 iterations、100-episode 周期评估 |
| Checkpoint 独立评估 | 已完成 | EMA policy 加载、3 个评估 seed、每个 seed 100 回合 |
| 仓库复现与实验报告 | 已完成 | 环境、脚本、指标、曲线、报告和测试 |
| LeRobot 公开数据 ACT | 计划中 | 数据字段检查、1,000-step smoke test、训练记录 |
| PickCube RGB ACT | 已完成 | RGB 数据、30,000 iterations、3 个评估 seed × 100 回合 |
| 多 seed 与数据量对照 | 进行中 | 评估 seed 对照已完成；独立训练 seed 与 10/50/100 demos 待完成 |
| 视觉域随机化 | 计划中 | 未见颜色、位置、纹理和光照上的泛化评估 |
| 实体机械臂迁移 | 计划中 | 真实示范、至少 30 次测试、安全与延迟记录 |
| 语言条件控制/VLA | 远期计划 | 多任务指令数据与轻量 VLA 对照 |

## 第一阶段：LeRobot 数据与 ACT

目标是理解 `observation.images.*`、`observation.state` 和 `action` 三类字段，以及从数据加载到 checkpoint 的完整训练流程。LeRobot 使用独立的 `lerobot-act` 环境，避免与 ManiSkill 的 PyTorch 和仿真依赖冲突。

验收要求：

- 加载公开数据并打印字段与张量形状。
- 完成 1,000 步冒烟训练，确认 GPU、loss 和 checkpoint 正常。
- 记录版本、batch size、训练时间和输出路径。

## 第二阶段：ManiSkill state ACT

这一阶段已在本仓库完成。数据流为：

```text
PickCube motion-planning demos
              |
       replay to state HDF5
              |
        ACT behavior cloning
              |
  100-episode periodic evaluation
              |
 checkpoint evaluation + videos
```

训练目前只有 `seed=1`。同一个 best checkpoint 已完成评估 seed 0、1、2 各 100 回合的对照；下一步应保持其余参数不变，补齐独立训练 seed 0 和 seed 2。

## 第三阶段：RGB 视觉输入（已完成）

已使用 ManiSkill 轨迹回放生成 RGB 观测，并完成 100 demos、训练 seed=1、30,000 次迭代的 RGB ACT。Best checkpoint 已在环境 seed 0、1、2 上各评估 100 个 episode。当前跨评估 seed 的 `success_once` 为 89.33% ± 1.53 pp，`success_at_end` 为 86.33% ± 2.52 pp。

验收要求：

- 已生成 RGB 轨迹并固定为不含 depth 的训练配置。
- 已完成正式训练、周期闭环评估和 checkpoint 视频评估。
- 已保存训练曲线、三个评估 seed 的 JSON 指标及成功/失败案例。
- 已与 state policy 做结果对照；因模型与训练配置不同，当前不视为严格模态消融。

## 第四阶段：系统对照实验

优先完成三组单变量实验：

1. 示范数量：10、50、100。
2. 输入模态：state、RGB。
3. 随机种子：0、1、2。

每组核心配置至少评估 50 个 episode；主结果使用 100 个 episode。报告成功率均值、标准差、平均回报、训练时间、推理速度和显存占用。

## 第五阶段：视觉泛化

在 RGB baseline 稳定后依次加入方块颜色、初始位置、大小、桌面纹理和光照随机化。每次只加入一种变化，并保留固定的未见条件测试集。

## 第六阶段：实体机械臂

先确认机械臂、夹爪、摄像头、SDK/ROS2、遥操作设备和安全规范。第一版任务限制为“从随机位置抓取一种积木并放入固定盒子”，采集 50 至 100 条成功示范，并至少测试 30 次。

## 第七阶段：语言条件与轻量 VLA

实体 ACT 稳定后，再添加自然语言任务描述和多任务数据。优先尝试适合单卡微调或推理的小模型，不把大型 VLA 的全参数训练作为当前项目目标。

## 每周检查

每周结束时回答：本周跑通了什么、哪个问题耗时最多、什么数据证明结果改善、下周唯一的主要改进是什么。实验执行细则见 [`experiment-protocol.md`](experiment-protocol.md)。
