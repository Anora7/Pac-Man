"""
Unit tests for Google Pac-Man training pipeline and curriculum callback.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
from stable_baselines3 import PPO

from google_config import GoogleCurriculumStage
from google_train import (
    GoogleCurriculumCallback,
    make_google_env,
    train_google_curriculum,
)


class TestGoogleTrain:
    def test_make_google_env(self, tmp_path: Path) -> None:
        env = make_google_env(active_ghosts=2, chase_strength=0.5, log_dir=tmp_path)
        obs, info = env.reset(seed=42)
        assert obs.shape == (1321,)
        assert len(env.unwrapped.ghosts) == 2
        assert env.unwrapped.chase_strength == 0.5
        env.close()

    def test_curriculum_callback_tracking(self) -> None:
        stage = GoogleCurriculumStage(
            stage_id=1,
            active_ghosts=2,
            chase_strength=0.5,
            max_steps=1000,
            success_threshold=0.5,
            success_window=2,
        )
        callback = GoogleCurriculumCallback(stage=stage, window=2, verbose=0)
        callback.num_timesteps = 100

        # Simulate 2 successful episode ends
        callback.locals = {
            "dones": [True],
            "infos": [{"is_success": True, "cleared": True, "total_pellets": 127, "pellets_remaining": 0}],
        }
        res1 = callback._on_step()
        assert res1 is True  # Only 1 in window of 2

        callback.locals = {
            "dones": [True],
            "infos": [{"is_success": True, "cleared": True, "total_pellets": 127, "pellets_remaining": 0}],
        }
        res2 = callback._on_step()
        assert res2 is False  # 2 successes in window of 2 -> promoted!
        assert callback.promoted is True

    def test_short_curriculum_run(self, tmp_path: Path) -> None:
        test_stages = (
            GoogleCurriculumStage(stage_id=1, active_ghosts=2, chase_strength=0.5, max_steps=64),
        )
        model = train_google_curriculum(
            stages=test_stages,
            total_steps_override=64,
            seed=42,
            models_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
            logs_dir=tmp_path / "logs",
            verbose=0,
        )
        assert (tmp_path / "models" / "google_pacman_ppo.zip").exists()
        assert (tmp_path / "results" / "curriculum_summary.csv").exists()

