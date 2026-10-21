"""
Unit tests for Google Pac-Man evaluation and random baseline comparison.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from google_evaluate import (
    EpisodeMetrics,
    compute_summary,
    evaluate_and_compare,
    run_evaluation_episodes,
)


class TestGoogleEvaluate:
    def test_compute_summary(self) -> None:
        metrics = [
            EpisodeMetrics(
                agent_type="random",
                episode=1,
                seed=42,
                score=50.0,
                pellets_collected=20,
                survival_steps=100,
                cleared=False,
                caught=True,
                ghosts_eaten=0,
            ),
            EpisodeMetrics(
                agent_type="random",
                episode=2,
                seed=43,
                score=150.0,
                pellets_collected=40,
                survival_steps=200,
                cleared=True,
                caught=False,
                ghosts_eaten=1,
            ),
        ]
        summary = compute_summary(metrics)
        assert summary["episodes"] == 2.0
        assert summary["mean_pellets"] == 30.0
        assert summary["mean_survival_steps"] == 150.0
        assert summary["clear_rate"] == 0.5
        assert summary["caught_rate"] == 0.5
        assert summary["mean_score"] == 100.0

    def test_run_random_evaluation_episodes(self) -> None:
        metrics = run_evaluation_episodes(
            agent_type="random",
            model=None,
            n_episodes=2,
            base_seed=100,
            render=False,
        )
        assert len(metrics) == 2
        for m in metrics:
            assert m.agent_type == "random"
            assert m.survival_steps > 0
            assert m.pellets_collected >= 0

    def test_evaluate_and_compare_output(self, tmp_path: Path) -> None:
        rand_sum, ppo_sum = evaluate_and_compare(
            model_path=tmp_path / "nonexistent.zip",
            n_episodes=2,
            base_seed=123,
            render=False,
            results_dir=tmp_path / "results",
        )
        assert rand_sum["episodes"] == 2.0
        assert (tmp_path / "results" / "evaluation_episodes.csv").exists()

    def test_evaluate_with_custom_ghosts_and_chase_strength(self, tmp_path: Path) -> None:
        """Requirement 4: Evaluate with custom ghost count and chase strength (e.g. Stage 1)."""
        rand_sum, _ = evaluate_and_compare(
            model_path=None,
            n_episodes=2,
            base_seed=42,
            render=False,
            results_dir=tmp_path / "results",
            active_ghosts=2,
            chase_strength=0.5,
        )
        assert rand_sum["episodes"] == 2.0
        assert "mean_survival_steps" in rand_sum

