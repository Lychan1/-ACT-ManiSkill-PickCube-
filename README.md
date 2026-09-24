# 基于 ACT 的 ManiSkill PickCube 视觉策略训练与评估

> **English summary:** This portfolio fork trains and evaluates ACT policies
> for ManiSkill `PickCube-v1`. The primary result is an RGB policy trained on
> 100 demonstrations. Its best checkpoint reached `89.33% ± 1.53 pp`
> `success_once` and `86.33% ± 2.52 pp` `success_at_end` across three
> 100-episode environment seeds (mean ± sample SD). A state policy is retained
> as an engineering baseline.

这是一个基于 [ManiSkill](https://github.com/haosulab/ManiSkill) 的项目型 fork，完整保留上游源码、许可证和引用信息。项目围绕 ACT 行为克隆打通了以下流程：

```text
官方示范 -> RGB/state 轨迹回放 -> ACT 训练 -> 周期闭环评估
                                           -> checkpoint 独立评估 -> 视频与指标
```

## 核心结果：RGB ACT

RGB 模型使用 `PickCube-v1`、`pd_ee_delta_pos`、`physx_cpu`、100 条示范和训练 `seed=1`。图像通道仅使用 RGB，不包含 depth；训练到 30,000 次迭代，并在每个评估点运行 100 个 episode。

| 训练期间评估 | 迭代 | Episodes | `success_once` | `success_at_end` | 平均回报 |
|---|---:|---:|---:|---:|---:|
| 最佳周期评估 | 30,000 | 100 | 86% | 84% | 25.86 |
| 前一周期评估 | 25,000 | 100 | 85% | 83% | 25.41 |

同一个 `best_eval_success_at_end.pt` 随后使用评估 seed 0、1、2，各运行 100 个 episode：

| Checkpoint 评估 seed | Episodes | `success_once` | `success_at_end` | 平均回报 |
|---:|---:|---:|---:|---:|
| 0 | 100 | 89% | 86% | 27.43 |
| 1 | 100 | 91% | 89% | 28.73 |
| 2 | 100 | 88% | 84% | 25.62 |
| 均值 ± 样本标准差 | 100/seed | 89.33% ± 1.53 pp | 86.33% ± 2.52 pp | 27.26 ± 1.56 |

这里的 seed 控制评估环境与随机数，不代表训练了三个模型。训练周期评估与 checkpoint 独立评估来自不同执行阶段，因此分别记录，不合并统计。

![RGB ACT training curves](results/curves/rgb_training_curves.png)

下图从每个评估 seed 各选一个成功和失败回合，只用于展示行为，不参与成功率计算：

![RGB checkpoint examples](assets/images/rgb-checkpoint-evaluation-seeded-examples.jpg)

RGB 代表视频：[seed 0](assets/videos/rgb_best_eval_success_at_end/seed-0/) · [seed 1](assets/videos/rgb_best_eval_success_at_end/seed-1/) · [seed 2](assets/videos/rgb_best_eval_success_at_end/seed-2/)

## State 基线与结果对照

保留的 state ACT 基线同样使用 100 条示范和训练 `seed=1`。其 best checkpoint 在三个 100-episode 评估 seed 上达到：

| 模型 | `success_once` | `success_at_end` | 平均回报 |
|---|---:|---:|---:|
| State ACT | 75.33% ± 3.06 pp | 68.00% ± 3.61 pp | 20.58 ± 0.51 |
| RGB ACT | **89.33% ± 1.53 pp** | **86.33% ± 2.52 pp** | **27.26 ± 1.56** |

![State and RGB checkpoint comparison](results/curves/state_rgb_checkpoint_comparison.png)

这不是严格的输入模态消融。两个实验还存在 batch size、视觉 backbone、attention heads 和训练实现细节差异，因此只能作为当前两套已完成配置的结果对照，不能把差值完全归因于 RGB/state 输入。

完整标量见 [`results/metrics.csv`](results/metrics.csv)，详细分析见 [`report/experiment_report.md`](report/experiment_report.md)。

## 快速复现

推荐 Ubuntu、NVIDIA GPU、Conda/Miniconda 和至少 50 GB 可用磁盘。当前验证环境为 Python 3.11.16、ManiSkill 3.0.1、Gymnasium 1.3.0 和 PyTorch 2.10.0+cu128。所有命令均从仓库根目录执行。

### 1. 创建环境

```bash
conda env create -f environment/maniskill.yml
conda activate maniskill-act311
```

### 2. 下载并生成 RGB 示范

```bash
scripts/act_pickcube/download_demo.sh
scripts/act_pickcube/replay_rgb.sh
```

默认 RGB 数据路径为：

```text
~/.maniskill/demos/PickCube-v1/motionplanning/
trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
```

### 3. 训练 RGB 模型

```bash
scripts/act_pickcube/train_rgb.sh
```

脚本默认复现当前配置：100 demos、batch size 4、30,001 total iters、每 5,000 次迭代评估 100 episodes。可通过 `DEMO_PATH`、`NUM_DEMOS`、`SEED`、`TOTAL_ITERS`、`BATCH_SIZE`、`NUM_EVAL_EPISODES` 和 `NUM_EVAL_ENVS` 覆盖。

### 4. 独立评估 checkpoint 并生成视频

```bash
for seed in 0 1 2; do
  SEED="$seed" scripts/act_pickcube/evaluate_rgb_checkpoint.sh \
    examples/baselines/act/runs/act-PickCube-v1-rgb-100demos-seed1/checkpoints/best_eval_success_at_end.pt
done
```

默认输出到 `runs/<experiment>/videos/<checkpoint-name>/seed-<seed>/`，每个目录同时写入 `metrics.json`。`num_queries`、Transformer 层数、hidden dim、attention heads、backbone 和 RGB/depth 设置必须与 checkpoint 的训练配置一致。

State 基线仍可通过 `replay_state.sh`、`train_smoke.sh`、`train_state.sh` 和 `evaluate_checkpoint.sh` 复现。

## 仓库结构

```text
environment/                 ManiSkill 与 LeRobot 独立环境
scripts/act_pickcube/        下载、RGB/state 回放、训练和评估入口
examples/baselines/act/      ACT baseline 与 checkpoint 评估代码
results/                     TensorBoard/JSON 派生指标与曲线
assets/                      精选评估视频与末帧总览
report/                      实验报告
docs/project-roadmap.md      项目路线
docs/experiment-protocol.md  对照实验与评估规范
```

## 个人实现内容

- 增加 RGB-only 图像训练与独立 checkpoint 评估入口，评估时从 checkpoint 加载 EMA policy。
- 修复训练后独立推理对全局 CLI 参数的依赖，使 RGB/depth 模式成为 policy 实例配置。
- 修复 Gymnasium vector environment 的 CPU 评估兼容性，使用 `SAME_STEP` autoreset 保留 `final_info`。
- 让指标收集兼容 NumPy/Tensor，并过滤 Gymnasium 自动生成的 `_metric` 掩码字段。
- 将评估 seed 传入首次 reset，并将均值、样本数和 checkpoint 来源持久化为 JSON。
- 提供 RGB/state 数据回放、训练、评估脚本和 15 个回归测试。

## 局限性

- RGB 与 state 都只有一个训练 seed。三组 checkpoint 评估 seed 只描述同一模型在不同环境随机条件下的波动。
- RGB/state 配置并非严格单变量控制，当前比较不能用于断言模态本身的因果提升。
- `metrics.json` 未持久化推理设备与硬件型号，不能据此比较 CPU/CUDA 数值差异或运行效率。
- 精选视频只有六个 RGB 回合和六个 state 回合，不能替代完整的 100-episode 统计。
- 未进行颜色、纹理、光照、相机扰动或 sim-to-real 泛化评估。
- checkpoint、原始 TensorBoard、数据集和完整视频目录不提交 Git；复现 checkpoint 评估前需要先训练。

## 上游项目、引用与许可证

本仓库基于 [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill)，ACT baseline 改编自 [tonyzhaozh/act](https://github.com/tonyzhaozh/act)。使用本项目时请同时参考 [`CITATION.cff`](CITATION.cff)、[`CITATION_MS2.cff`](CITATION_MS2.cff)、[`LICENSE`](LICENSE) 和 [`LICENSE-3RD-PARTY`](LICENSE-3RD-PARTY)。
