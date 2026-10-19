"""Tests for the isolated ghost-direction PPO experiment."""

from __future__ import annotations

from pathlib import Path

import train_ghost_direction
from evaluate import EpisodeResult, EvaluationResult
from train import TrainingConfig


def test_ghost_direction_experiment_trains_and_evaluates_isolated_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    training_configs: list[TrainingConfig] = []
    evaluation_configs = []
    monkeypatch.setattr(
        train_ghost_direction,
        "MODEL_DIR",
        tmp_path / "models" / "ghost_direction",
    )
    monkeypatch.setattr(
        train_ghost_direction,
        "LOG_DIR",
        tmp_path / "logs" / "ghost_direction",
    )
    monkeypatch.setattr(
        train_ghost_direction,
        "RESULTS_PATH",
        tmp_path / "results" / "comparison.csv",
    )

    def fake_train(config: TrainingConfig):
        training_configs.append(config)
        return object()

    def fake_evaluate(config, model=None):
        evaluation_configs.append(config)
        episodes = [
            EpisodeResult(
                episode=index + 1,
                total_reward=float(index),
                steps=10 + index,
                pellets_collected=2 + index,
                total_pellets=69,
                cleared=False,
                caught=True,
                truncated=False,
                completion_fraction=0.1,
            )
            for index in range(config.n_episodes)
        ]
        return EvaluationResult(
            n_episodes=config.n_episodes,
            episodes=episodes,
            mean_reward=0.5,
            std_reward=0.5,
            mean_steps=10.5,
            mean_pellets_collected=2.5,
            success_rate=0.0,
            avg_completion_fraction=0.1,
        )

    monkeypatch.setattr(train_ghost_direction, "train", fake_train)
    monkeypatch.setattr(train_ghost_direction, "evaluate", fake_evaluate)

    rows = train_ghost_direction.run_experiment(
        total_timesteps=300_000,
        episodes=2,
        seed=44,
    )

    assert training_configs[0].observation_config.include_ghost_direction
    assert training_configs[0].total_timesteps == 300_000
    assert [config.seed for config in evaluation_configs] == [42_000, 42_030]
    assert [row["model"] for row in rows] == ["baseline", "ghost_direction_ppo"]
    assert all(row["episodes"] == 2 for row in rows)
    assert all("pellets_ci_low" in row for row in rows)
    assert train_ghost_direction.RESULTS_PATH.is_file()
