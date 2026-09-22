# ACT PickCube State Policy 实验报告

## 摘要

本实验使用 ManiSkill `PickCube-v1` 的 100 条 motion-planning 示范训练状态输入 ACT 策略，并在闭环环境中周期评估。训练执行 30,000 次迭代，100-episode 周期评估的最佳 `success_once` 为 75%，最佳 `success_at_end` 为 68%。此外，新增 checkpoint 独立评估入口并保存视频；已有 50 段视频的末帧叠加指标均显示成功，但 fresh CPU 复查得到 70% `success_once` 和 60% `success_at_end`，提示结果存在设备敏感性。

## 实验设置

| 项目 | 配置 |
|---|---|
| 环境 | `PickCube-v1` |
| 观测 | state |
| 控制模式 | `pd_ee_delta_pos` |
| 仿真后端 | `physx_cpu` |
| 示范数量 | 100 |
| 训练迭代 | 30,000 |
| Seed | 1 |
| 最大 episode 长度 | 100 |
| 周期评估规模 | 100 episodes |
| 软件 | Python 3.11.16、ManiSkill 3.0.1、Gymnasium 1.3.0、PyTorch 2.10.0+cu128 |
| 硬件 | 原始日志未记录，不能可靠补写 |

TensorBoard 共记录 300 个 loss 点，最后一步为 29,900。记录的累计 update 时间约 2,073 秒；该数值不包含所有数据准备和人工操作时间。

## 结果

| Iteration | `success_once` | `success_at_end` | 平均回报 |
|---:|---:|---:|---:|
| 0 | 1% | 0% | 0.16 |
| 5,000 | 15% | 12% | 4.20 |
| 10,000 | 75% | 68% | 20.79 |
| 15,000 | 68% | 55% | 16.40 |
| 20,000 | 65% | 54% | 16.81 |
| 25,000 | 68% | 60% | 16.60 |

训练 loss 从 68.11 下降到 29,900 次迭代时的 0.0106。闭环成功率在 10,000 次迭代达到峰值，之后没有随 loss 持续下降而继续提高，说明 imitation loss 不能代替环境评估，并且最佳 checkpoint 选择是必要的。

![Training curves](../results/curves/training_curves.png)

## Checkpoint 视频评估

`best_eval_success_at_end.pt` 通过新增的独立入口加载 EMA policy。已有视频目录包含 50 个 episode，其末帧叠加指标均为 `success_once=1`、`success_at_end=1`；原评估的控制台设备输出没有保留。

仓库整理期间，在 CPU 上重新运行相同 checkpoint 的 10 个 episode，得到 `success_once=70%`、`success_at_end=60%`、平均回报 20.10。该差异说明动作序列闭环执行可能放大 CPU/CUDA 推理的微小数值差异。后续应在固定硬件、固定设备和多个 seed 上重复评估。

## 工程改动

- CPU vector evaluation 使用 Gymnasium `SAME_STEP` autoreset，确保 episode 截断时保留 `final_info`。
- 指标收集兼容 NumPy/Tensor，并忽略 Gymnasium `_metric` 掩码。
- checkpoint 加载器明确使用 EMA 权重，并在 checkpoint 缺少 `ema_agent` 时失败。
- 独立评估入口支持设备、seed、episode 数、并行环境、视频目录和 ACT 结构参数。

## 局限性与后续工作

- 只有一个 seed，尚不能计算跨 seed 均值与标准差。
- 没有记录硬件型号和峰值显存。
- 已有 50 个视频回合没有失败案例，但 fresh CPU 评估出现失败；本次临时验证视频不作为精选媒体提交。
- 原 50 回合视频批次未保存设备与控制台指标，证据只能追溯到视频叠加信息。
- 策略依赖环境 state，尚未覆盖 RGB 泛化和 sim-to-real。
- 下一步先运行 seed 0 和 seed 2，再开展 10/50/100 demos 与 state/RGB 对照。
