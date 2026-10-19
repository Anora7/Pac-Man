"""Unit coverage for comparison output aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import compare_algorithms
import pytest
from compare_algorithms import (
    AlgorithmComparisonConfig,
    _run_seed,
    _medium_config,
    _write_comparison_csv,
    _write_report,
    priority_config,
)
from config import DQNConfig, MEDIUM_TIMESTEPS


def test_medium_comparison_config_has_two_seeds_and_isolated_outputs() -> None:
    config = _medium_config()
    assert config.seeds == (42, 43)
    assert config.total_timesteps == MEDIUM_TIMESTEPS
    assert config.dqn_config.total_timesteps == MEDIUM_TIMESTEPS
    assert config.evaluation_episodes == 30
    assert "medium" in str(config.models_dir)
    assert "medium" in str(config.results_path)
    assert "medium" in str(config.plots_dir)


def test_priority_comparison_uses_three_seeds_and_300k_steps() -> None:
    config = priority_config()
    assert config.seeds == (42, 43, 44)
    assert config.total_timesteps == 300_000
    assert config.evaluation_episodes == 30
    assert "priority_300k" in str(config.models_dir)


def test_worker_can_train_only_one_selected_algorithm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AlgorithmComparisonConfig(
        seeds=(42,),
        total_timesteps=32,
        evaluation_episodes=30,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    monkeypatch.setattr(compare_algorithms, "train", lambda training_config: object())
    monkeypatch.setattr(
        compare_algorithms,
        "evaluate",
        lambda evaluation_config, model: SimpleNamespace(
            mean_reward=1.0,
            std_reward=0.5,
            success_rate=0.0,
            mean_pellets_collected=2.0,
            mean_steps=3.0,
        ),
    )

    records = _run_seed(42, config, algorithms=("ppo",))

    assert len(records) == 1
    assert records[0]["algorithm"] == "PPO"


def test_eval_only_loads_saved_models_and_never_trains(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AlgorithmComparisonConfig(
        seeds=(42,),
        evaluation_episodes=30,
        max_steps=25,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        results_path=tmp_path / "results.csv",
        report_path=tmp_path / "report.md",
        plots_dir=tmp_path / "plots",
    )
    for algorithm in ("ppo", "dqn"):
        model_path = config.models_dir / algorithm / "seed_42" / "final_model.zip"
        model_path.parent.mkdir(parents=True)
        model_path.touch()

    class FakeModelLoader:
        @staticmethod
        def load(path: str, device: str):
            return path, device

    monkeypatch.setattr(compare_algorithms, "PPO", FakeModelLoader)
    monkeypatch.setattr(compare_algorithms, "DQN", FakeModelLoader)
    monkeypatch.setattr(
        compare_algorithms,
        "train",
        lambda *_: pytest.fail("eval-only must not train PPO"),
    )
    monkeypatch.setattr(
        compare_algorithms,
        "train_dqn",
        lambda *_: pytest.fail("eval-only must not train DQN"),
    )
    evaluations = []

    def fake_evaluate(evaluation_config, model):
        evaluations.append((evaluation_config, model))
        return SimpleNamespace(
            mean_reward=1.0,
            std_reward=0.5,
            success_rate=0.0,
            mean_pellets_collected=2.0,
            mean_steps=3.0,
        )

    monkeypatch.setattr(compare_algorithms, "evaluate", fake_evaluate)
    records = _run_seed(42, config, eval_only=True)

    assert len(records) == 2
    assert len(evaluations) == 2
    assert all(item[0].n_episodes == 30 for item in evaluations)
    assert all(item[0].seed == 10_042 for item in evaluations)


def test_comparison_writes_seed_rows_and_aggregate_intervals(tmp_path: Path) -> None:
    config = AlgorithmComparisonConfig(
        seeds=(42, 43),
        results_path=tmp_path / "comparison.csv",
        report_path=tmp_path / "report.md",
        dqn_config=DQNConfig(),
    )
    records = [
        {
            "algorithm": "PPO",
            "seed": 42,
            "mean_reward": 2.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "mean_pellets": 1.0,
            "mean_steps": 5.0,
        },
        {
            "algorithm": "PPO",
            "seed": 43,
            "mean_reward": 4.0,
            "std_reward": 0.0,
            "success_rate": 1.0,
            "mean_pellets": 3.0,
            "mean_steps": 7.0,
        },
        {
            "algorithm": "DQN",
            "seed": 42,
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "mean_pellets": 1.0,
            "mean_steps": 5.0,
        },
        {
            "algorithm": "DQN",
            "seed": 43,
            "mean_reward": 3.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "mean_pellets": 2.0,
            "mean_steps": 6.0,
        },
    ]
    _write_comparison_csv(config, records)
    _write_report(config, records)
    with config.results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    ppo_summary = next(
        row for row in rows if row["algorithm"] == "PPO" and row["seed"] == "aggregate"
    )
    assert float(ppo_summary["mean_reward"]) == 3.0
    assert float(ppo_summary["reward_ci_low"]) < 3.0
    assert "Decision 001" in config.report_path.read_text(encoding="utf-8")
