"""Summarize ACT training and checkpoint evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .experiment_matrix import ExperimentSpec, load_matrix
except ImportError:
    from experiment_matrix import ExperimentSpec, load_matrix


SUMMARY_FIELDS = [
    "experiment",
    "observation",
    "num_demos",
    "train_seed",
    "eval_seed",
    "status",
    "success_once",
    "success_at_end",
    "mean_return",
    "episode_count",
    "loss_start",
    "loss_end",
    "metrics_path",
    "checkpoint_path",
]


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: _to_python(value) for key, value in payload.items()}


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _values(metrics: dict[str, Any], *names: str) -> list[float]:
    if isinstance(metrics.get("metrics"), dict):
        metrics = metrics["metrics"]
    for name in names:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, dict) and "mean" in value:
                return [float(value["mean"])]
            if not isinstance(value, list):
                value = [value]
            return [float(item) for item in value]
    return []


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _portable_path(value: str | Path, artifact_root: Path | None) -> str:
    path = Path(value)
    if artifact_root is None or not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError:
        return f"external/{path.name}"


def summarize_loss_values(values: list[float]) -> tuple[float | str, float | str]:
    return (values[0], values[-1]) if values else ("", "")


def _load_loss(run_dir: Path) -> tuple[float | str, float | str]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return "", ""
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return "", ""
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    if "losses/total_loss" not in accumulator.Tags().get("scalars", []):
        return "", ""
    values = [event.value for event in accumulator.Scalars("losses/total_loss")]
    return summarize_loss_values(values)


def _find_metrics(run_dir: Path, eval_seed: int) -> Path | None:
    candidates = [
        run_dir / "videos" / "best_eval_success_at_end" / f"seed-{eval_seed}" / "metrics.json",
        run_dir / "metrics.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def summarize_experiment(
    spec: ExperimentSpec,
    run_dir: Path,
    eval_seed: int | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    metrics_path = _find_metrics(run_dir, eval_seed if eval_seed is not None else spec.train_seed)
    summary: dict[str, Any] = {
        "experiment": spec.name,
        "observation": spec.observation,
        "num_demos": spec.num_demos,
        "train_seed": spec.train_seed,
        "eval_seed": eval_seed if eval_seed is not None else "",
        "status": "missing_metrics",
        "success_once": "",
        "success_at_end": "",
        "mean_return": "",
        "episode_count": "",
        "loss_start": "",
        "loss_end": "",
        "metrics_path": _portable_path(metrics_path, artifact_root) if metrics_path else "",
        "checkpoint_path": "",
    }
    if metrics_path is None:
        return summary

    metrics = load_metrics(metrics_path)
    loss_start, loss_end = _load_loss(run_dir)
    success_once = _values(metrics, "success_once")
    success_at_end = _values(metrics, "success_at_end")
    returns = _values(metrics, "return", "returns", "episode_return", "reward")
    summary.update(
        {
            "status": "complete",
            "eval_seed": eval_seed if eval_seed is not None else metrics.get("seed", ""),
            "success_once": _mean(success_once),
            "success_at_end": _mean(success_at_end),
            "mean_return": _mean(returns),
            "episode_count": metrics.get("num_eval_episodes", len(success_at_end or success_once or returns)),
            "loss_start": loss_start,
            "loss_end": loss_end,
            "checkpoint_path": _portable_path(
                metrics.get("checkpoint", run_dir), artifact_root
            ),
        }
    )
    return summary


def summarize_matrix(matrix_path: Path, runs_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for spec in load_matrix(matrix_path):
        run_dir = runs_root / spec.name
        if not run_dir.exists() and spec.name.endswith("seed1"):
            legacy_name = f"act-PickCube-v1-{spec.observation}-{spec.num_demos}demos"
            run_dir = runs_root / legacy_name
        summaries.append(
            summarize_experiment(spec, run_dir, artifact_root=runs_root)
        )
    return summaries


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(summarize_matrix(args.matrix, args.runs_root), args.output)


if __name__ == "__main__":
    main()
