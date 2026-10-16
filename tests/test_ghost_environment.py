"""Small, deterministic API checks for the ghost-side Gymnasium environment."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from ghost_environment import GhostEnvironment


def test_shared_ghost_environment_cycles_one_shared_policy_per_ghost() -> None:
    env = GhostEnvironment(max_steps=5)
    observation, _ = env.reset(seed=42)
    assert observation.shape == env.observation_space.shape
    assert env.action_space.n == 4
    assert observation.size == 12 * 12 * 6 + 2
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert next_observation.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "ghost_reward" in info
    assert env._active_ghost_index == 1
    env.close()


def test_single_ghost_mode_has_discrete_action_space() -> None:
    env = GhostEnvironment(shared_policy=False, max_steps=2)
    observation, _ = env.reset(seed=42)
    assert observation.shape == env.observation_space.shape
    assert env.action_space.n == 4
    env.step(0)
    env.close()


def test_ghost_environment_passes_gymnasium_checker() -> None:
    env = GhostEnvironment(max_steps=5)
    check_env(env, skip_render_check=True)
    env.close()
