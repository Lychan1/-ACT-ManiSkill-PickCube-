import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from examples.baselines.act.experiment_matrix import (
    ExperimentSpec,
    build_training_command,
    experiment_name,
    load_matrix,
    validate_matrix,
)
from examples.baselines.act.summarize_experiments import (
    load_metrics,
    summarize_matrix,
    summarize_loss_values,
    summarize_experiment,
)


class TestActExperiments(TestCase):
    def test_experiment_name_uses_observation_demos_and_seed(self):
        self.assertEqual(
            experiment_name("state", 50, 2),
            "act-PickCube-v1-state-50demos-seed2",
        )

    def test_validate_matrix_rejects_duplicate_experiment_names(self):
        specs = [ExperimentSpec("state", 10, 1), ExperimentSpec("state", 10, 1)]

        with self.assertRaisesRegex(ValueError, "duplicate experiment name"):
            validate_matrix(specs)

    def test_load_matrix_accepts_json(self):
        with TemporaryDirectory() as directory:
            matrix_path = Path(directory) / "matrix.json"
            matrix_path.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {"observation": "rgb", "num_demos": 50, "train_seed": 0}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            specs = load_matrix(matrix_path)

            self.assertEqual(specs, [ExperimentSpec("rgb", 50, 0)])

    def test_build_training_command_passes_num_demos_and_seed(self):
        spec = ExperimentSpec("state", 10, 2, total_iters=1234)

        command = build_training_command(spec, repo_root=Path("/repo"))

        self.assertIn("scripts/act_pickcube/train_state.sh", command[-1])
        self.assertIn("NUM_DEMOS=10", command)
        self.assertIn("SEED=2", command)
        self.assertIn("TOTAL_ITERS=1234", command)

    def test_load_metrics_normalizes_json_values(self):
        with TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "success_once": {"mean": 0.5, "samples": 2},
                            "success_at_end": {"mean": 0.5, "samples": 2},
                            "return": {"mean": 1.5, "samples": 2},
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = load_metrics(metrics_path)

            self.assertEqual(metrics["metrics"]["success_once"]["mean"], 0.5)
            self.assertEqual(metrics["metrics"]["return"]["mean"], 1.5)

    def test_summarize_experiment_reports_missing_metrics(self):
        with TemporaryDirectory() as directory:
            spec = ExperimentSpec("rgb", 50, 1)

            summary = summarize_experiment(spec, Path(directory) / "missing")

            self.assertEqual(summary["status"], "missing_metrics")
            self.assertEqual(summary["train_seed"], 1)
            self.assertEqual(summary["eval_seed"], "")

    def test_summarize_experiment_separates_train_and_eval_seed(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            metrics_dir = run_dir / "videos" / "best_eval_success_at_end" / "seed-2"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "num_eval_episodes": 2,
                        "metrics": {
                            "success_once": {"mean": 0.5, "samples": 2},
                            "success_at_end": {"mean": 0.5, "samples": 2},
                            "return": {"mean": 3.0, "samples": 2},
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_experiment(
                ExperimentSpec("state", 100, 0), run_dir, eval_seed=2
            )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["train_seed"], 0)
            self.assertEqual(summary["eval_seed"], 2)
            self.assertAlmostEqual(summary["success_once"], 0.5)
            self.assertAlmostEqual(summary["mean_return"], 3.0)

    def test_summarize_loss_values_returns_first_and_last(self):
        self.assertEqual(summarize_loss_values([3.0, 2.0, 1.0]), (3.0, 1.0))
        self.assertEqual(summarize_loss_values([]), ("", ""))

    def test_matrix_summary_uses_portable_artifact_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "observation": "rgb",
                                "num_demos": 100,
                                "train_seed": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runs_root = root / "runs"
            run_dir = runs_root / "act-PickCube-v1-rgb-100demos-seed1"
            metrics_dir = run_dir / "videos/best_eval_success_at_end/seed-1"
            metrics_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoints/best_eval_success_at_end.pt"
            (metrics_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "seed": 1,
                        "num_eval_episodes": 100,
                        "metrics": {
                            "success_once": {"mean": 0.9, "samples": 100},
                            "success_at_end": {"mean": 0.8, "samples": 100},
                            "return": {"mean": 25.0, "samples": 100},
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_matrix(matrix_path, runs_root)[0]

            self.assertEqual(
                summary["metrics_path"],
                "act-PickCube-v1-rgb-100demos-seed1/videos/"
                "best_eval_success_at_end/seed-1/metrics.json",
            )
            self.assertEqual(
                summary["checkpoint_path"],
                "act-PickCube-v1-rgb-100demos-seed1/checkpoints/"
                "best_eval_success_at_end.pt",
            )
