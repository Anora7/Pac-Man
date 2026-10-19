"""Small, deterministic API checks for the ghost-side Gymnasium environment."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from config import ACTION_DELTAS, Action
from ghost_environment import GhostEnvironment
from ghost_observations import build_ghost_observation


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


def test_training_observation_uses_shared_ghost_observation_builder() -> None:
    env = GhostEnvironment(max_steps=5)
    observation, _ = env.reset(seed=42)
    expected = build_ghost_observation(
        env._env.maze,
        [ghost.position for ghost in env._env.ghosts],
        env._active_ghost_index,
        env._env.pacman_position,
        env._env._power_mode_active,
    )
    assert observation.shape == (866,)
    assert np.array_equal(observation, expected)
    env.close()


def test_single_ghost_mode_has_discrete_action_space() -> None:
    env = GhostEnvironment(shared_policy=False, max_steps=2)
    observation, _ = env.reset(seed=42)
    assert observation.shape == env.observation_space.shape
    assert env.action_space.n == 4
    env.step(0)
    env.close()


def test_training_environment_falls_back_from_wall_action(monkeypatch) -> None:
    env = GhostEnvironment(max_steps=5)
    env.reset(seed=42)
    ghost = env._env.ghosts[0]
    position = ghost.position
    invalid_action = next(
        int(action)
        for action, delta in ACTION_DELTAS.items()
        if not env._env.maze.is_walkable(position + delta)
    )
    original_step = env._env.step
    passed_actions: list[int | None] = []

    def capture_step(pacman_action, *, ghost_actions=None):
        assert ghost_actions is not None
        passed_actions.extend(ghost_actions)
        return original_step(pacman_action, ghost_actions=ghost_actions)

    monkeypatch.setattr(env._env, "step", capture_step)
    env.step(invalid_action)
    selected_action = passed_actions[0]
    assert selected_action is not None
    assert env._env.maze.is_walkable(
        position + ACTION_DELTAS[Action(selected_action)]
    )
    env.close()


def test_ghost_environment_passes_gymnasium_checker() -> None:
    env = GhostEnvironment(max_steps=5)
    check_env(env, skip_render_check=True)
    env.close()
