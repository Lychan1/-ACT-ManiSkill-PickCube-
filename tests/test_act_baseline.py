import importlib
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from examples.baselines.act.act.evaluate import evaluate
from examples.baselines.act.act import make_env


class _TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self):
        self.elapsed_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.elapsed_steps += 1
        truncated = self.elapsed_steps == 2
        info = {"episode": {"episode_len": self.elapsed_steps}}
        return np.zeros(1, dtype=np.float32), 0.0, False, truncated, info


class _Agent:
    def eval(self):
        pass

    def train(self):
        pass

    def get_action(self, obs):
        return torch.zeros((obs.shape[0], 1, 1))


class _NumpyMetricEvalEnv:
    single_observation_space = gym.spaces.Box(
        -1.0, 1.0, (2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(-1.0, 1.0, (1, 1), dtype=np.float32)
    num_envs = 1

    def reset(self):
        return np.zeros((1, 2), dtype=np.float32), {}

    def step(self, action):
        info = {
            "final_info": {
                "episode": {"success_at_end": np.array([True])}
            }
        }
        return (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.array([False]),
            np.array([True]),
            info,
        )


class _MaskedNumpyMetricEvalEnv(_NumpyMetricEvalEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["final_info"]["episode"]["_success_at_end"] = np.array(
            [True]
        )
        return obs, reward, terminated, truncated, info


class TestActBaseline(TestCase):
    def test_cpu_eval_env_returns_final_info_on_truncation(self):
        with patch.object(
            make_env.gym, "make", side_effect=lambda *args, **kwargs: _TwoStepEnv()
        ), patch.object(
            make_env, "CPUGymWrapper", side_effect=lambda env, **kwargs: env
        ):
            env = make_env.make_eval_envs(
                "Dummy-v0",
                num_envs=1,
                sim_backend="physx_cpu",
                env_kwargs={},
                other_kwargs=None,
            )
            try:
                env.reset()
                env.step(np.zeros((1, 1), dtype=np.float32))
                _, _, _, truncated, info = env.step(
                    np.zeros((1, 1), dtype=np.float32)
                )
            finally:
                env.close()

        self.assertEqual(truncated.tolist(), [True])
        self.assertIn("final_info", info)
        self.assertEqual(
            info["final_info"]["episode"]["episode_len"].tolist(), [2]
        )

    def test_evaluate_accepts_numpy_episode_metrics(self):
        eval_kwargs = {
            "stats": None,
            "num_queries": 1,
            "temporal_agg": False,
            "max_timesteps": 1,
            "device": torch.device("cpu"),
            "sim_backend": "physx_cpu",
        }

        try:
            metrics = evaluate(1, _Agent(), _NumpyMetricEvalEnv(), eval_kwargs)
        except AttributeError as exc:
            self.fail(f"evaluate rejected NumPy episode metrics: {exc}")

        self.assertEqual(metrics["success_at_end"].tolist(), [[True]])

    def test_evaluate_ignores_gymnasium_metric_masks(self):
        eval_kwargs = {
            "stats": None,
            "num_queries": 1,
            "temporal_agg": False,
            "max_timesteps": 1,
            "device": torch.device("cpu"),
            "sim_backend": "physx_cpu",
        }

        metrics = evaluate(
            1, _Agent(), _MaskedNumpyMetricEvalEnv(), eval_kwargs
        )

        self.assertNotIn("_success_at_end", metrics)

    def test_checkpoint_evaluation_defaults_to_run_video_directory(self):
        try:
            checkpoint = importlib.import_module(
                "examples.baselines.act.act.checkpoint"
            )
        except ModuleNotFoundError as exc:
            self.fail(f"checkpoint evaluation support is missing: {exc}")

        path = Path("runs/example/checkpoints/best_eval_success_at_end.pt")

        self.assertEqual(
            checkpoint.default_video_dir(path),
            Path("runs/example/videos/best_eval_success_at_end"),
        )

    def test_checkpoint_evaluation_loads_ema_agent(self):
        try:
            checkpoint = importlib.import_module(
                "examples.baselines.act.act.checkpoint"
            )
        except ModuleNotFoundError as exc:
            self.fail(f"checkpoint evaluation support is missing: {exc}")

        agent = torch.nn.Linear(2, 1)
        raw_state = {
            key: torch.zeros_like(value) for key, value in agent.state_dict().items()
        }
        ema_state = {
            key: torch.ones_like(value) for key, value in agent.state_dict().items()
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            torch.save({"agent": raw_state, "ema_agent": ema_state}, path)
            checkpoint.load_agent(agent, path, torch.device("cpu"))

        for value in agent.state_dict().values():
            self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_checkpoint_evaluation_cli_is_executable(self):
        script = (
            Path(__file__).parents[1]
            / "examples"
            / "baselines"
            / "act"
            / "evaluate_checkpoint.py"
        )

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parent,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_checkpoint_wrapper_resolves_repo_relative_checkpoint(self):
        repo_root = Path(__file__).parents[1]
        script = repo_root / "scripts" / "act_pickcube" / "evaluate_checkpoint.sh"
        checkpoint = Path("examples/baselines/act/README.md")

        with TemporaryDirectory() as tmp_dir:
            bin_dir = Path(tmp_dir) / "bin"
            bin_dir.mkdir()
            fake_python = bin_dir / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8"
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(script), str(checkpoint)],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        checkpoint_index = arguments.index("--checkpoint")
        self.assertEqual(
            arguments[checkpoint_index + 1], str(repo_root / checkpoint)
        )
