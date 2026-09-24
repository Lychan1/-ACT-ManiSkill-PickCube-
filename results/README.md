# Experiment results

This directory contains version-controlled summaries derived from local ACT
training artifacts. Raw TensorBoard events, checkpoints, datasets, and full
video folders remain under `examples/baselines/act/runs/` and are ignored by
Git.

## Files

- `metrics.csv`: long-form state and RGB training/evaluation metrics.
- `curves/rgb_training_curves.png`: RGB closed-loop evaluation curves.
- `curves/state_rgb_checkpoint_comparison.png`: comparison of the completed
  state and RGB checkpoint evaluations.
- `curves/training_curves.png`: state training curves.
- `curves/checkpoint_seed_comparison.png`: state checkpoint evaluation seeds.

The primary RGB run is `act-PickCube-v1-rgb-100demos-seed1`. TensorBoard
contains 301 loss points through step 30,000 and seven 100-episode periodic
evaluations. Its best checkpoint was evaluated for 100 episodes on environment
seeds 0, 1, and 2.

The state baseline is `act-PickCube-v1-state-100demos-seed1`. Its TensorBoard
writer logged 300 loss points through iteration 29,900 and six periodic
evaluations through iteration 25,000. The same state best checkpoint was also
evaluated on environment seeds 0, 1, and 2.

`training_periodic_eval` and `checkpoint_video_eval` are intentionally kept
separate. Evaluation seeds describe repeated environments for one trained
checkpoint, not independent model-training seeds. The state/RGB comparison is
also not a strict modality ablation because the completed configurations differ
in model and training details.
