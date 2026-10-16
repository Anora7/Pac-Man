"""
Tests for src/evaluate.py — Phase 7: Evaluation.

Uses an untrained PPO model (random policy) to verify pipeline behaviour
without requiring a full training run.  All file-I/O tests write to a
temporary directory.

Covers:
  - EvaluationConfig defaults and custom values
  - EpisodeResult fields and frozen constraint
  - EvaluationResult fields and aggregate correctness
  - evaluate() returns EvaluationResult
  - Episode count matches config
  - All episodes produce valid EpisodeResult objects
  - success_rate, mean_reward, mean_steps in expected ranges
  - Missing model file raises FileNotFoundError
  - evaluate() with pre-loaded model skips file I/O
  - evaluate() with model saved to disk and loaded from path
  - verbose=False produces no extra output
  - render=False works correctly (headless)
  - Episode outcomes are mutually exclusive (cleared XOR caught XOR truncated)
  - completion_fraction in [0.0, 1.0] for every episode
  - pellets_collected <= total_pellets for every episode
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from config import MAX_STEPS
from environment import PacmanEnvironment
from evaluate import (
    EpisodeResult,
    EvaluationConfig,
    EvaluationResult,
    _aggregate,
    evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_model() -> PPO:
    """An untrained PPO model (random policy) shared across the module."""
    env = PacmanEnvironment()
    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    env.close()
    return model


@pytest.fixture(scope="module")
def fast_result(random_model: PPO) -> EvaluationResult:
    """A cached evaluation result to avoid re-running the loop repeatedly."""
    config = EvaluationConfig(n_episodes=5, verbose=False, seed=0, max_steps=50)
    return evaluate(config, model=random_model)


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------

class TestEvaluationConfig:
    def test_default_model_path(self) -> None:
        cfg = EvaluationConfig()
        assert isinstance(cfg.model_path, Path)
        assert cfg.model_path.suffix == ".zip"

    def test_default_n_episodes(self) -> None:
        assert EvaluationConfig().n_episodes == 10

    def test_default_render_false(self) -> None:
        assert EvaluationConfig().render is False

    def test_default_deterministic_true(self) -> None:
        assert EvaluationConfig().deterministic is True

    def test_default_seed_none(self) -> None:
        assert EvaluationConfig().seed is None

    def test_default_max_steps(self) -> None:
        assert EvaluationConfig().max_steps == MAX_STEPS

    def test_default_verbose_true(self) -> None:
        assert EvaluationConfig().verbose is True

    def test_custom_values(self) -> None:
        cfg = EvaluationConfig(
            model_path=Path("models/final_model.zip"),
            n_episodes=20,
            render=True,
            deterministic=False,
            seed=42,
            max_steps=200,
            verbose=False,
        )
        assert cfg.n_episodes == 20
        assert cfg.render is True
        assert cfg.seed == 42
        assert cfg.max_steps == 200

    def test_render_and_ghost_options(self) -> None:
        cfg = EvaluationConfig(
            ghost_type="random",
            render_fps=12,
            save_csv=Path("results/evaluation.csv"),
        )
        assert cfg.ghost_type == "random"
        assert cfg.render_fps == 12
        assert cfg.save_csv == Path("results/evaluation.csv")

    def test_invalid_episode_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_episodes"):
            EvaluationConfig(n_episodes=0)


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

class TestEpisodeResult:
    def _make(self, **kwargs) -> EpisodeResult:
        defaults = dict(
            episode=1,
            total_reward=10.0,
            steps=50,
            pellets_collected=5,
            total_pellets=69,
            cleared=False,
            caught=False,
            truncated=True,
            completion_fraction=0.07,
        )
        defaults.update(kwargs)
        return EpisodeResult(**defaults)

    def test_fields_accessible(self) -> None:
        r = self._make()
        assert r.episode == 1
        assert r.total_reward == pytest.approx(10.0)
        assert r.steps == 50
        assert r.total_pellets == 69

    def test_is_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.total_reward = 999.0  # type: ignore[misc]

    def test_completion_fraction_stored(self) -> None:
        r = self._make(completion_fraction=0.5)
        assert r.completion_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EvaluationResult and _aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _make_episodes(self, n: int, *, cleared_count: int = 0) -> list[EpisodeResult]:
        episodes = []
        for i in range(n):
            cleared = i < cleared_count
            episodes.append(EpisodeResult(
                episode=i + 1,
                total_reward=float(i * 10),
                steps=100 + i,
                pellets_collected=10 + i,
                total_pellets=69,
                cleared=cleared,
                caught=not cleared,
                truncated=False,
                completion_fraction=(10 + i) / 69,
            ))
        return episodes

    def test_n_episodes_stored(self) -> None:
        eps = self._make_episodes(4)
        result = _aggregate(eps, 4)
        assert result.n_episodes == 4

    def test_episodes_list_stored(self) -> None:
        eps = self._make_episodes(3)
        result = _aggregate(eps, 3)
        assert len(result.episodes) == 3

    def test_mean_reward_correct(self) -> None:
        eps = self._make_episodes(3)
        result = _aggregate(eps, 3)
        expected = np.mean([0.0, 10.0, 20.0])
        assert result.mean_reward == pytest.approx(expected)

    def test_std_reward_correct(self) -> None:
        eps = self._make_episodes(3)
        result = _aggregate(eps, 3)
        expected = float(np.std([0.0, 10.0, 20.0]))
        assert result.std_reward == pytest.approx(expected)

    def test_mean_steps_correct(self) -> None:
        eps = self._make_episodes(3)
        result = _aggregate(eps, 3)
        expected = np.mean([100, 101, 102])
        assert result.mean_steps == pytest.approx(expected)

    def test_success_rate_all_cleared(self) -> None:
        eps = self._make_episodes(4, cleared_count=4)
        result = _aggregate(eps, 4)
        assert result.success_rate == pytest.approx(1.0)

    def test_success_rate_none_cleared(self) -> None:
        eps = self._make_episodes(4, cleared_count=0)
        result = _aggregate(eps, 4)
        assert result.success_rate == pytest.approx(0.0)

    def test_success_rate_partial(self) -> None:
        eps = self._make_episodes(4, cleared_count=2)
        result = _aggregate(eps, 4)
        assert result.success_rate == pytest.approx(0.5)

    def test_avg_completion_fraction_bounds(self) -> None:
        eps = self._make_episodes(5)
        result = _aggregate(eps, 5)
        assert 0.0 <= result.avg_completion_fraction <= 1.0

    def test_mean_pellets_collected(self) -> None:
        eps = self._make_episodes(3)
        result = _aggregate(eps, 3)
        expected = np.mean([10, 11, 12])
        assert result.mean_pellets_collected == pytest.approx(expected)

    def test_survival_and_outcome_rates(self) -> None:
        eps = self._make_episodes(4, cleared_count=1)
        eps[1] = EpisodeResult(
            episode=2,
            total_reward=0.0,
            steps=30,
            pellets_collected=5,
            total_pellets=69,
            cleared=False,
            caught=False,
            truncated=True,
            completion_fraction=0.1,
        )
        result = _aggregate(eps, 4)
        assert result.mean_survival_steps == result.mean_steps
        assert result.caught_rate == pytest.approx(0.5)
        assert result.timeout_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# evaluate() — with pre-loaded model
# ---------------------------------------------------------------------------

class TestEvaluateWithModel:
    def test_returns_evaluation_result(self, fast_result: EvaluationResult) -> None:
        assert isinstance(fast_result, EvaluationResult)

    def test_optional_csv_export(self, random_model: PPO, tmp_path: Path) -> None:
        csv_path = tmp_path / "nested" / "episodes.csv"
        result = evaluate(
            EvaluationConfig(
                n_episodes=1,
                max_steps=5,
                verbose=False,
                save_csv=csv_path,
            ),
            model=random_model,
        )
        assert csv_path.exists()
        assert "total_reward" in csv_path.read_text(encoding="utf-8")
        assert result.n_episodes == 1

    def test_episode_count_matches_config(self, random_model: PPO) -> None:
        config = EvaluationConfig(n_episodes=3, verbose=False, max_steps=20)
        result = evaluate(config, model=random_model)
        assert result.n_episodes == 3
        assert len(result.episodes) == 3

    def test_mean_reward_is_float(self, fast_result: EvaluationResult) -> None:
        assert isinstance(fast_result.mean_reward, float)

    def test_std_reward_non_negative(self, fast_result: EvaluationResult) -> None:
        assert fast_result.std_reward >= 0.0

    def test_mean_steps_positive(self, fast_result: EvaluationResult) -> None:
        assert fast_result.mean_steps > 0

    def test_success_rate_in_unit_interval(self, fast_result: EvaluationResult) -> None:
        assert 0.0 <= fast_result.success_rate <= 1.0

    def test_avg_completion_fraction_in_unit_interval(
        self, fast_result: EvaluationResult
    ) -> None:
        assert 0.0 <= fast_result.avg_completion_fraction <= 1.0

    def test_all_episodes_have_correct_index(self, fast_result: EvaluationResult) -> None:
        for i, ep in enumerate(fast_result.episodes):
            assert ep.episode == i + 1

    def test_episode_outcomes_mutually_exclusive(self, fast_result: EvaluationResult) -> None:
        """Each episode ends in exactly one of: cleared, caught, or truncated."""
        for ep in fast_result.episodes:
            outcomes = [ep.cleared, ep.caught, ep.truncated]
            assert sum(outcomes) == 1, (
                f"Episode {ep.episode}: {outcomes} — expected exactly one True"
            )

    def test_completion_fraction_in_bounds(self, fast_result: EvaluationResult) -> None:
        for ep in fast_result.episodes:
            assert 0.0 <= ep.completion_fraction <= 1.0

    def test_pellets_collected_le_total(self, fast_result: EvaluationResult) -> None:
        for ep in fast_result.episodes:
            assert ep.pellets_collected <= ep.total_pellets

    def test_steps_positive(self, fast_result: EvaluationResult) -> None:
        for ep in fast_result.episodes:
            assert ep.steps > 0

    def test_verbose_false_runs_silently(self, random_model: PPO) -> None:
        config = EvaluationConfig(n_episodes=2, verbose=False, max_steps=20)
        result = evaluate(config, model=random_model)
        assert isinstance(result, EvaluationResult)

    def test_none_config_uses_defaults(self, random_model: PPO) -> None:
        """evaluate(None, model=...) must not raise."""
        result = evaluate(None, model=random_model)
        assert result.n_episodes == EvaluationConfig().n_episodes

    def test_render_false_no_crash(self, random_model: PPO) -> None:
        config = EvaluationConfig(n_episodes=1, render=False, verbose=False, max_steps=10)
        result = evaluate(config, model=random_model)
        assert len(result.episodes) == 1

    def test_non_deterministic_policy(self, random_model: PPO) -> None:
        config = EvaluationConfig(
            n_episodes=2, deterministic=False, verbose=False, max_steps=20
        )
        result = evaluate(config, model=random_model)
        assert result.n_episodes == 2


# ---------------------------------------------------------------------------
# evaluate() — file I/O tests
# ---------------------------------------------------------------------------

class TestEvaluateFromFile:
    def test_missing_model_raises_file_not_found(self) -> None:
        cfg = EvaluationConfig(
            model_path=Path("models/nonexistent_model.zip"),
            verbose=False,
        )
        with pytest.raises(FileNotFoundError):
            evaluate(cfg)

    def test_load_model_from_disk(self, random_model: PPO) -> None:
        """Save model to temp dir, load via config, evaluate."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "test_model.zip"
            random_model.save(str(model_path))
            cfg = EvaluationConfig(
                model_path=model_path,
                n_episodes=2,
                verbose=False,
                max_steps=20,
            )
            result = evaluate(cfg)
            assert result.n_episodes == 2

    def test_load_model_without_zip_extension(self, random_model: PPO) -> None:
        """EvaluationConfig path without .zip extension should still load."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "test_model.zip"
            random_model.save(str(model_path))
            # Provide path without .zip extension
            cfg = EvaluationConfig(
                model_path=Path(tmp) / "test_model",
                n_episodes=1,
                verbose=False,
                max_steps=10,
            )
            result = evaluate(cfg)
            assert len(result.episodes) == 1
