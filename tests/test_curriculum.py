"""Tests for configurable ghost-count curriculum behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import CurriculumConfig
from environment import PacmanEnvironment
from train_curriculum import _SeedRun, _plot_curriculum


def test_environment_supports_one_ghost_without_changing_observation_shape() -> None:
    env = PacmanEnvironment(ghost_count=1)
    try:
        observation, _ = env.reset(seed=42)
        assert len(env.ghosts) == 1
        assert observation.shape == (723,)
    finally:
        env.close()


def test_environment_rejects_unavailable_ghost_count() -> None:
    with pytest.raises(ValueError, match="ghost_count"):
        PacmanEnvironment(ghost_count=3)


def test_curriculum_config_validates_promotion_settings() -> None:
    with pytest.raises(ValueError, match="promotion_success_threshold"):
        CurriculumConfig(promotion_success_threshold=1.1)


def test_curriculum_plot_is_saved_even_without_completed_episodes(
    tmp_path: Path,
) -> None:
    config = CurriculumConfig(plot_path=tmp_path / "curriculum.png")
    _plot_curriculum(config, [_SeedRun(seed=42, promotions=[], curves=[])])
    assert config.plot_path.exists()
