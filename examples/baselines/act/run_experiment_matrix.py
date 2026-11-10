"""Run the configured ACT experiment matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiment_matrix import build_training_command, load_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for spec in load_matrix(args.matrix):
        command = build_training_command(spec, args.repo_root)
        print("$", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=args.repo_root, check=True)


if __name__ == "__main__":
    main()
