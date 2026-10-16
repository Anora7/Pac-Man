"""Fast smoke coverage for the isolated SB3 DQN trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import DQN

from config import DQNConfig
from train_dqn import train_dqn


def test_dqn_config_has_reasonable_replay_defaults() -> None:
    config = DQNConfig()
    assert config.buffer_size >= config.learning_starts
    assert config.batch_size < config.buffer_size
    assert config.target_update_interval > 0


def test_dqn_config_rejects_invalid_training_bounds() -> None:
    with pytest.raises(ValueError):
        DQNConfig(total_timesteps=0)


def test_train_dqn_saves_cpu_model_and_monitor_log(tmp_path: Path) -> None:
    config = DQNConfig(
        total_timesteps=8,
        buffer_size=32,
        learning_starts=4,
        batch_size=4,
        train_freq=1,
        target_update_interval=4,
        eval_freq=8,
        n_eval_episodes=1,
        checkpoint_freq=8,
        max_steps=8,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        tensorboard_dir=tmp_path / "tensorboard",
        seed=42,
    )
    model = train_dqn(config)
    assert isinstance(model, DQN)
    assert model.device.type == "cpu"
    assert (config.models_dir / "final_model.zip").exists()
    assert (config.logs_dir / "0.monitor.csv").exists()
