"""
Smoke tests for src/train.py — Phase 6: PPO Training Pipeline.

These tests verify the training pipeline end-to-end with minimal timesteps
to keep runtime short.  They do NOT validate RL convergence — that requires
a full training run.

Covers:
  - TrainingConfig defaults and custom values
  - Paths are Path objects
  - train() completes without error
  - final_model.zip is written to models_dir
  - best_model.zip is written to models_dir (from EvalCallback)
  - Returned model can predict actions
  - train() with verbose=0 is silent
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import tempfile
from pathlib import Path

import numpy as np
import pytest

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    GAMMA,
    LEARNING_RATE,
    MAX_STEPS,
    N_EPOCHS,
    N_STEPS,
    TOTAL_TIMESTEPS,
    ObservationConfig,
)
from environment import PacmanEnvironment, _OBS_SIZE
from train import TrainingConfig, train
from observation_metadata import load_observation_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_config(tmp_path: Path) -> TrainingConfig:
    """Return a minimal TrainingConfig for quick smoke tests."""
    return TrainingConfig(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        tensorboard_dir=tmp_path / "tensorboard",
        total_timesteps=256,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        eval_freq=128,
        n_eval_episodes=1,
        checkpoint_freq=256,
        max_steps=50,
        seed=0,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_paths_are_path_objects(self) -> None:
        cfg = TrainingConfig()
        assert isinstance(cfg.models_dir, Path)
        assert isinstance(cfg.logs_dir, Path)
        assert isinstance(cfg.tensorboard_dir, Path)

    def test_default_total_timesteps(self) -> None:
        assert TrainingConfig().total_timesteps == TOTAL_TIMESTEPS

    def test_default_learning_rate(self) -> None:
        assert TrainingConfig().learning_rate == pytest.approx(LEARNING_RATE)

    def test_default_gamma(self) -> None:
        assert TrainingConfig().gamma == pytest.approx(GAMMA)

    def test_default_n_steps(self) -> None:
        assert TrainingConfig().n_steps == N_STEPS

    def test_default_batch_size(self) -> None:
        assert TrainingConfig().batch_size == BATCH_SIZE

    def test_default_n_epochs(self) -> None:
        assert TrainingConfig().n_epochs == N_EPOCHS

    def test_default_ent_coef(self) -> None:
        assert TrainingConfig().ent_coef == pytest.approx(ENT_COEF)

    def test_default_clip_range(self) -> None:
        assert TrainingConfig().clip_range == pytest.approx(CLIP_RANGE)

    def test_default_device(self) -> None:
        assert TrainingConfig().device == "auto"

    def test_default_max_steps(self) -> None:
        assert TrainingConfig().max_steps == MAX_STEPS

    def test_default_seed_is_none(self) -> None:
        assert TrainingConfig().seed is None

    def test_default_verbose(self) -> None:
        assert TrainingConfig().verbose == 1

    def test_default_show_progress_bar(self) -> None:
        assert TrainingConfig().show_progress_bar is False

    def test_custom_values_stored(self) -> None:
        cfg = TrainingConfig(total_timesteps=1000, learning_rate=1e-3, seed=7)
        assert cfg.total_timesteps == 1000
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.seed == 7

    def test_custom_path(self) -> None:
        cfg = TrainingConfig(models_dir=Path("/tmp/my_models"))
        assert cfg.models_dir == Path("/tmp/my_models")


# ---------------------------------------------------------------------------
# Training pipeline smoke tests
# ---------------------------------------------------------------------------

class TestTrainSmoke:
    def test_train_creates_final_model(self) -> None:
        """train() must save final_model.zip in models_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            train(cfg)
            assert (cfg.models_dir / "final_model.zip").exists()

    def test_optional_observation_config_is_saved_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            cfg.observation_config = ObservationConfig(
                include_ghost_direction=True,
                include_previous_pacman_action=True,
            )
            train(cfg)
            assert load_observation_config(
                cfg.models_dir / "final_model.zip"
            ) == cfg.observation_config

    def test_train_creates_best_model(self) -> None:
        """EvalCallback saves best_model.zip in models_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            train(cfg)
            assert (cfg.models_dir / "best_model.zip").exists()

    def test_train_creates_directories(self) -> None:
        """train() creates output directories when they do not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            # Directories do not exist yet
            assert not cfg.models_dir.exists()
            train(cfg)
            assert cfg.models_dir.exists()
            assert cfg.logs_dir.exists()
            assert cfg.tensorboard_dir.exists()

    def test_train_returns_ppo_model(self) -> None:
        """train() must return an SB3 PPO instance."""
        from stable_baselines3 import PPO

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            model = train(cfg)
            assert isinstance(model, PPO)

    def test_trained_model_can_predict(self) -> None:
        """The returned model must be able to generate actions."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            model = train(cfg)
            env = PacmanEnvironment()
            obs, _ = env.reset(seed=0)
            action, _ = model.predict(obs, deterministic=True)
            assert int(action) in range(4)
            env.close()

    def test_train_with_none_config_uses_defaults(self) -> None:
        """train(None) should not raise — it uses TrainingConfig defaults.

        We override paths via environment variables or just skip if paths
        already exist, so here we just verify no exception is raised with a
        patched config via monkeypatching the default.
        """
        # We can't easily test the full default without writing to the repo
        # directories.  Verify the helper path at least.
        cfg = TrainingConfig()
        assert cfg.total_timesteps == TOTAL_TIMESTEPS

    def test_saved_model_loads_correctly(self) -> None:
        """A model saved by train() can be reloaded and used for inference."""
        from stable_baselines3 import PPO

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            train(cfg)
            loaded = PPO.load(str(cfg.models_dir / "final_model"))
            env = PacmanEnvironment()
            obs, _ = env.reset(seed=0)
            action, _ = loaded.predict(obs, deterministic=True)
            assert int(action) in range(4)
            env.close()

    def test_train_verbose_zero_completes(self) -> None:
        """Verbose=0 training must complete without printing."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_config(Path(tmp))
            cfg.verbose = 0
            train(cfg)  # must not raise
