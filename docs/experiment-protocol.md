# ACT PickCube 实验规范

## 目的

保证训练结果可以复查、不同实验只改变声明的变量，并避免把小样本视频演示当作正式成功率。

## 固定项

- 环境：`PickCube-v1`
- 控制模式：`pd_ee_delta_pos`
- 观测模式：state baseline 使用 `state`
- 仿真后端：`physx_cpu`
- 最大 episode 长度：100
- ACT 默认结构：2 层 encoder、4 层 decoder、hidden dim 256、30 queries
- 正式周期评估：100 episodes

若修改任一固定项，必须创建新的实验名并在结果表中单独成组。

## 命名规则

```text
act-<env>-<observation>-<num_demos>demos-seed<seed>
```

例如：

```text
act-PickCube-v1-state-100demos-seed1
```

## 最小实验矩阵

| 变量 | 取值 |
|---|---|
| 随机种子 | 0、1、2 |
| 示范数量 | 10、50、100 |
| 输入模态 | state、RGB |

先完成三个随机种子的 state/100 demos 基线，再扩展其他变量。不要在同一次对照中同时改变示范数量和模型结构。

## 每次运行必须记录

- Git commit 和工作区是否有未提交修改。
- Python、ManiSkill、Gymnasium、PyTorch 和 CUDA 版本。
- GPU、显存、CPU、内存和训练耗时。
- 数据路径、示范数量、seed、batch size 和全部非默认参数。
- checkpoint 路径和 TensorBoard event 文件。
- `success_once`、`success_at_end`、平均回报和 episode 数。

## 评估口径

训练周期评估与 checkpoint 视频评估必须分开记录：

- `training_periodic_eval`：用于正式比较，主结果为 100 episodes。
- `checkpoint_video_eval`：用于复查 checkpoint 和生成案例视频，不替代正式比较。

还必须记录推理设备（CPU/CUDA）。闭环动作序列可能放大不同设备上的数值差异，跨设备结果不得合并统计。

多个 seed 的正式结果报告均值和样本标准差，不只报告最好一次。

## 失败分类

失败视频按以下互斥的主要原因标注：

- 未到达方块。
- 到达但没有夹稳。
- 抓起后掉落。
- 抓起但未移动到目标区域。
- 对位置、颜色、视角或光照变化不稳定。

如果一次评估没有失败回合，应写明“本次小样本中未观察到失败”，不能补造案例。

## 产物管理

原始数据、checkpoint、完整 TensorBoard 日志和大视频保留在被 Git 忽略的运行目录。仓库只提交 CSV 摘要、曲线、精选视频和报告，并确保每个摘要都记录来源实验名。
