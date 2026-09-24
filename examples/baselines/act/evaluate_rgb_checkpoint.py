import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from act.checkpoint import default_video_dir, load_agent, save_metrics
from act.evaluate import evaluate
from act.make_env import make_eval_envs
from train_rgbd import Agent, FlattenRGBDObservationWrapper


@dataclass
class Args:
    checkpoint: Path
    """Path to an RGB or RGB-D ACT checkpoint."""
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 100
    num_eval_episodes: int = 100
    num_eval_envs: int = 1
    video_dir: Optional[Path] = None
    """Output directory. Defaults to RUN_DIR/videos/CHECKPOINT_NAME/seed-SEED."""
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True
    temporal_agg: bool = True

    # These must match the architecture used to train the checkpoint.
    kl_weight: float = 10
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = True
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False


def build_env_kwargs(args: Args) -> dict:
    return {
        "control_mode": args.control_mode,
        "reward_mode": "sparse",
        "obs_mode": "rgbd" if args.include_depth else "rgb",
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }


def main(args: Args):
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_dir = (
        args.video_dir.expanduser().resolve()
        if args.video_dir is not None
        else default_video_dir(checkpoint_path) / f"seed-{args.seed}"
    )
    video_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        build_env_kwargs(args),
        other_kwargs=None,
        video_dir=str(video_dir),
        wrappers=[
            partial(FlattenRGBDObservationWrapper, depth=args.include_depth)
        ],
    )

    try:
        agent = Agent(envs, args).to(device)
        checkpoint = load_agent(agent, checkpoint_path, device)
        eval_kwargs = {
            "stats": checkpoint.get("norm_stats"),
            "num_queries": args.num_queries,
            "temporal_agg": args.temporal_agg,
            "max_timesteps": args.max_episode_steps,
            "device": device,
            "sim_backend": args.sim_backend,
        }
        metrics = evaluate(
            args.num_eval_episodes,
            agent,
            envs,
            eval_kwargs,
            seed=args.seed,
        )
    finally:
        envs.close()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    for name, values in metrics.items():
        print(f"{name}: {np.mean(values):.4f}")

    metrics_path = video_dir / "metrics.json"
    save_metrics(
        metrics_path,
        metrics,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    print(f"Metrics: {metrics_path}")

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Videos: {video_dir}")
    for video in videos:
        print(f"  {video}")


if __name__ == "__main__":
    main(tyro.cli(Args))
