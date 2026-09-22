from pathlib import Path

import torch


def default_video_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent.parent / "videos" / checkpoint_path.stem


def load_agent(agent, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if "ema_agent" not in checkpoint:
        raise KeyError(f"EMA agent not found in checkpoint: {checkpoint_path}")
    agent.load_state_dict(checkpoint["ema_agent"])
    return checkpoint
