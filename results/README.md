# Experiment results

This directory contains small, version-controlled summaries derived from local
training artifacts. Raw TensorBoard events, checkpoints, and full run folders
remain under `examples/baselines/act/runs/` and are intentionally ignored by
Git.

## Files

- `metrics.csv`: long-form scalar data for the 30,000-iteration state-based
  ACT run and three 100-episode evaluations of the best checkpoint.
- `curves/training_curves.png`: loss and closed-loop evaluation curves derived
  from the same TensorBoard event file.
- `curves/checkpoint_seed_comparison.png`: checkpoint results for evaluation
  seeds 0, 1, and 2, plus mean and sample standard deviation across seeds.

The source run directory is `act-PickCube-v1-state-100demos`; its training
configuration used seed 1. The TensorBoard writer logged training through
iteration 29,900 and periodic evaluation at iterations 0, 5,000, 10,000,
15,000, 20,000, and 25,000.

The `checkpoint_video_eval` rows come from the `metrics.json` written for
evaluation seeds 0, 1, and 2. Each seed contains 100 episodes. These three
runs evaluate the same checkpoint trained with seed 1; they measure variation
over evaluation initial conditions, not independent training runs.

`training_periodic_eval` and `checkpoint_video_eval` remain separate because
they were produced by different evaluation executions. The six videos under
`assets/videos/` are curated success/failure examples and are not a source for
the rates in `metrics.csv`.
