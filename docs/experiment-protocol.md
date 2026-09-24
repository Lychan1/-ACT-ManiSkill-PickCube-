# ACT PickCube 实验规范

## 目的

保证训练结果可以复查、不同实验只改变声明的变量，并避免把小样本视频演示当作正式成功率。

## 固定项

- 环境：`PickCube-v1`
- 控制模式：`pd_ee_delta_pos`
- 观测模式：state baseline 使用 `state`；RGB baseline 使用 RGB 图像且不含 depth
- 仿真后端：`physx_cpu`
- 最大 episode 长度：100
- ACT 共同结构项：2 层 encoder、4 层 decoder、hidden dim 256、30 queries；heads、backbone 和 batch size 必须逐实验记录
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

当前 state 与 RGB 各完成一个训练 seed，并分别完成三个环境评估 seed。下一步应补齐独立训练 seed。严格模态消融不得同时改变 batch size、heads、backbone 或训练预算；当前已完成的 state/RGB 结果只能标记为配置对照。

## 每次运行必须记录

- Git commit 和工作区是否有未提交修改。
- Python、ManiSkill、Gymnasium、PyTorch 和 CUDA 版本。
- GPU、显存、CPU、内存和训练耗时。
- 数据路径、示范数量、seed、batch size 和全部非默认参数。
- checkpoint 路径和 TensorBoard event 文件。
- `success_once`、`success_at_end`、平均回报和 episode 数。

## 评估口径

训练周期评估与 checkpoint 视频评估必须分开记录：

- `training_periodic_eval`：训练过程中执行，用于观察学习进展和选择 checkpoint。
- `checkpoint_video_eval`：从指定 checkpoint 独立执行；达到每个 seed 100 episodes 且保存 JSON 时可用于正式 checkpoint 对比。

精选视频只用于展示成功和失败模式，不能从精选数量反推成功率。还必须记录推理设备（CPU/CUDA）。闭环动作序列可能放大不同设备上的数值差异，跨设备结果不得合并统计。

多个 seed 的正式结果报告均值和样本标准差，不只报告最好一次。报告必须注明 seed 属于训练 seed 还是评估环境 seed；后者不能替代多个独立训练模型。

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
