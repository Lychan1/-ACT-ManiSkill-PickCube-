# Experiment results

This directory contains small, version-controlled summaries derived from local
training artifacts. Raw TensorBoard events, checkpoints, and full run folders
remain under `examples/baselines/act/runs/` and are intentionally ignored by
Git.

## Files

- `metrics.csv`: long-form scalar data for the 30,000-iteration state-based
  ACT run, the existing 50-video batch, and a fresh 10-episode CPU checkpoint
  verification.
- `curves/training_curves.png`: loss and closed-loop evaluation curves derived
  from the same TensorBoard event file.

The source run is
`act-PickCube-v1-state-100demos-seed1`. Its TensorBoard writer logged training
through iteration 29,900 and periodic evaluation at iterations 0, 5,000,
10,000, 15,000, 20,000, and 25,000.

The `checkpoint_video_eval` rows describe the existing video batch for
`best_eval_success_at_end.pt`; all 50 final-frame overlays report success, but
the original console device output was not retained. The
`checkpoint_cpu_verification` rows come from a fresh CPU run on 2026-09-22.
Neither group is combined with the 100-episode periodic evaluation rows.
