"""Plot completed and missing experiments from the matrix summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    completed = []
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        success = _float(row["success_at_end"])
        if row["status"] == "complete" and success is not None:
            completed.append(row | {"success_at_end": success})
            grouped[row["observation"]].append((int(row["num_demos"]), success))

    figure, axis = plt.subplots(figsize=(7, 4.5))
    demo_counts = {int(row["num_demos"]) for row in completed}
    if len(demo_counts) >= 2:
        for observation, points in sorted(grouped.items()):
            points.sort()
            axis.plot(
                [point[0] for point in points],
                [point[1] * 100 for point in points],
                marker="o",
                label=observation.upper(),
            )
        axis.set_xlabel("Number of demonstrations")
        axis.set_title("ACT demonstration count comparison")
        axis.legend()
    else:
        labels = [
            f"{row['observation'].upper()}\n{row['num_demos']} demos · seed {row['train_seed']}"
            for row in completed
        ]
        axis.bar(labels, [row["success_at_end"] * 100 for row in completed])
        axis.set_title("Completed ACT checkpoint results")
    axis.set_ylabel("Success at end (%)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_matrix_success.png", dpi=160)
    plt.close(figure)

    status_counts = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(list(status_counts), list(status_counts.values()))
    axis.set_ylabel("Experiments")
    axis.set_title("Experiment matrix completion status")
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_matrix_status.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
