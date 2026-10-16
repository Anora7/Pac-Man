"""Unit coverage for comparison output aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

from compare_algorithms import (
    AlgorithmComparisonConfig,
    _write_comparison_csv,
    _write_report,
)
from config import DQNConfig


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
