import json
from pathlib import Path

import numpy as np
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


def save_metrics(
    path: Path,
    metrics: dict,
    *,
    checkpoint_path: Path,
    seed: int,
) -> dict:
    if not metrics:
        raise ValueError("Cannot save empty evaluation metrics")

    summary = {
        name: {
            "mean": float(np.mean(values)),
            "samples": int(np.asarray(values).size),
        }
        for name, values in metrics.items()
    }
    payload = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "num_eval_episodes": next(iter(summary.values()))["samples"],
        "metrics": summary,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return payload
