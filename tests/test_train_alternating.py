"""Fast unit tests for alternating training configuration and helpers."""

from __future__ import annotations

import random

import pytest

from config import AlternatingConfig
from train_alternating import _quick_config, _sample_opponent


def test_alternating_config_has_three_reproducible_seeds() -> None:
    assert AlternatingConfig().seeds == (42, 43, 44)


def test_alternating_config_validates_opponent_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        AlternatingConfig(opponent_pool_latest_probability=1.5)


def test_quick_config_is_one_seed_and_one_short_round() -> None:
    config = _quick_config()
    assert config.seeds == (42,)
    assert config.n_rounds == 1
    assert config.pacman_steps_per_round < 100


def test_opponent_pool_sampler_can_select_latest_or_older() -> None:
    pool = ["old", "latest"]
    assert _sample_opponent(pool, random.Random(2), 1.0) == "latest"
    assert _sample_opponent(pool, random.Random(2), 0.0) == "old"
