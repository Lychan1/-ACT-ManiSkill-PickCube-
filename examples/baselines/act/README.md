# Action Chunking with Transformers (ACT)

Code for running the ACT algorithm based on ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/pdf/2304.13705). It is adapted from the [original code](https://github.com/tonyzhaozh/act).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n act-ms python=3.9
conda activate act-ms
pip install -e .
```

## Setup

Read through the [imitation learning setup documentation](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) which details everything you need to know regarding running imitation learning baselines in ManiSkill. It includes details on how to download demonstration datasets, preprocess them, evaluate policies fairly for comparison, as well as suggestions to improve performance and avoid bugs.

## Training

We provide scripts to train ACT on demonstrations.

Note that some demonstrations are slow (e.g. motion planning or human teleoperated) and can exceed the default max episode steps which can be an issue as imitation learning algorithms learn to solve the task at the same speed the demonstrations solve it. In this case, you can use the `--max-episode-steps` flag to set a higher value so that the policy can solve the task in time. General recommendation is to set `--max-episode-steps` to about 2x the length of the mean demonstrations length you are using for training. We have tuned baselines in the `baselines.sh` script that set a recommended `--max-episode-steps` for each task.

Example state-based training, learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task.

```bash
seed=1
demos=100
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100 \
  --total_iters 30000 --log_freq 100 --eval_freq 5000 \
  --exp-name=act-PickCube-v1-state-${demos}_motionplanning_demos-$seed \
  --track # track training on wandb
```

## Evaluating a checkpoint and recording videos

`evaluate_checkpoint.py` loads the EMA policy stored in a state-based ACT
checkpoint, runs closed-loop evaluation, prints aggregate metrics, and records
MP4 videos:

```bash
python evaluate_checkpoint.py \
  --checkpoint runs/act-PickCube-v1-state-100demos/checkpoints/best_eval_success_at_end.pt \
  --num-eval-episodes 100 \
  --num-eval-envs 1 \
  --seed 0
```

Videos default to
`runs/<experiment>/videos/<checkpoint-name>/seed-<seed>/`; the same directory
also receives `metrics.json`. Repeat formal evaluation with seeds 0, 1, and 2.
The model architecture flags,
including `--num-queries`, `--enc-layers`, `--dec-layers`, and
`--hidden-dim`, must match the values used to create the checkpoint.

For an RGB-only checkpoint trained with `train_rgbd.py`, use the RGB-aware
evaluator and explicitly disable depth to match the training observation mode:

```bash
python evaluate_rgb_checkpoint.py \
  --checkpoint runs/act-PickCube-v1-rgb-100demos-seed1/checkpoints/best_eval_success_at_end.pt \
  --no-include-depth \
  --num-eval-episodes 100 \
  --num-eval-envs 1 \
  --seed 0
```

Run the command with seeds 0, 1, and 2. Each seed writes to its own video and
metrics directory, so the three evaluations do not overwrite one another.

Recent Gymnasium releases return NumPy episode metrics and companion mask keys
from CPU vector environments. This fork uses same-step autoreset, accepts both
NumPy and Tensor metrics, and ignores mask keys prefixed with `_`.

## Reproducible project wrappers

From the repository root, the completed RGB workflow is available through:

```bash
scripts/act_pickcube/replay_rgb.sh
scripts/act_pickcube/train_rgb.sh
SEED=0 scripts/act_pickcube/evaluate_rgb_checkpoint.sh
```

The RGB wrapper defaults match the documented experiment: 100 demos, training
seed 1, batch size 4, 30,001 total iterations, evaluation every 5,000 steps,
and 100 evaluation episodes. State wrappers remain available as
`replay_state.sh`, `train_state.sh`, and `evaluate_checkpoint.sh`.

The repository regression suite currently contains 15 ACT baseline tests,
including RGB policy configuration, checkpoint loading, wrapper defaults,
video paths, Gymnasium metrics, and CLI execution.

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/rss/ZhaoKLF23,
  author       = {Tony Z. Zhao and
                  Vikash Kumar and
                  Sergey Levine and
                  Chelsea Finn},
  editor       = {Kostas E. Bekris and
                  Kris Hauser and
                  Sylvia L. Herbert and
                  Jingjin Yu},
  title        = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle    = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July
                  10-14, 2023},
  year         = {2023},
  url          = {https://doi.org/10.15607/RSS.2023.XIX.016},
  doi          = {10.15607/RSS.2023.XIX.016},
  timestamp    = {Thu, 20 Jul 2023 15:37:49 +0200},
  biburl       = {https://dblp.org/rec/conf/rss/ZhaoKLF23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
