"""Experiment matrix definitions and reproducible training commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


VALID_OBSERVATIONS = {"state", "rgb"}


@dataclass(frozen=True)
class ExperimentSpec:
    observation: str
    num_demos: int
    train_seed: int
    total_iters: int = 30_000
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.observation not in VALID_OBSERVATIONS:
            raise ValueError(f"unsupported observation: {self.observation}")
        if self.num_demos <= 0:
            raise ValueError("num_demos must be positive")
        if self.train_seed < 0:
            raise ValueError("train_seed must be non-negative")
        if self.total_iters <= 0:
            raise ValueError("total_iters must be positive")

    @property
    def name(self) -> str:
        return experiment_name(self.observation, self.num_demos, self.train_seed)


def experiment_name(observation: str, num_demos: int, train_seed: int) -> str:
    if observation not in VALID_OBSERVATIONS:
        raise ValueError(f"unsupported observation: {observation}")
    return f"act-PickCube-v1-{observation}-{num_demos}demos-seed{train_seed}"


def validate_matrix(specs: Iterable[ExperimentSpec]) -> list[ExperimentSpec]:
    normalized = list(specs)
    names = [spec.name for spec in normalized]
    if len(names) != len(set(names)):
        raise ValueError("duplicate experiment name in matrix")
    return normalized


def load_matrix(path: Path) -> list[ExperimentSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = [ExperimentSpec(**item) for item in payload["experiments"]]
    return validate_matrix(specs)


def build_training_command(spec: ExperimentSpec, repo_root: Path) -> list[str]:
    script_name = "train_rgb.sh" if spec.observation == "rgb" else "train_state.sh"
    script = repo_root / "scripts" / "act_pickcube" / script_name
    values = {
        "NUM_DEMOS": str(spec.num_demos),
        "SEED": str(spec.train_seed),
        "TOTAL_ITERS": str(spec.total_iters),
        "EXP_NAME": spec.name,
    }
    if spec.batch_size is not None:
        values["BATCH_SIZE"] = str(spec.batch_size)
    return ["env", *[f"{key}={value}" for key, value in values.items()], str(script)]


def with_batch_size(spec: ExperimentSpec, batch_size: int) -> ExperimentSpec:
    return replace(spec, batch_size=batch_size)
