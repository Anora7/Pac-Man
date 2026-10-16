"""Tests for explicit reward variants used by the ablation study."""

from __future__ import annotations

from dataclasses import replace

from config import REWARDS, REWARD_ABLATION_VARIANTS
from environment import PacmanEnvironment


def test_reward_variant_table_covers_required_terms() -> None:
    variants = dict(REWARD_ABLATION_VARIANTS)
    assert variants["baseline"] == REWARDS
    assert variants["caught_penalty_50"].CAUGHT_BY_GHOST == -50.0
    assert variants["caught_penalty_500"].CAUGHT_BY_GHOST == -500.0
    assert variants["step_penalty_0"].STEP == 0.0
    assert variants["step_penalty_2"].STEP == -2.0
    assert variants["no_wall_penalty"].HIT_WALL == 0.0
    assert variants["pellet_reward_5"].PELLET == 5.0
    assert variants["pellet_reward_20"].PELLET == 20.0
    assert variants["clear_bonus_0"].CLEAR_MAZE == 0.0
    assert variants["clear_bonus_1000"].CLEAR_MAZE == 1000.0


def test_environment_uses_custom_reward_config() -> None:
    reward_config = replace(REWARDS, STEP=0.0)
    env = PacmanEnvironment(reward_config=reward_config)
    try:
        env.reset(seed=42)
        _, reward, _, _, info = env.step(3)
        assert reward == 10.0
        assert info["last_step_events"]["pellet_eaten"] is True
    finally:
        env.close()
