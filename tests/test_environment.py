"""
Tests for src/environment.py and src/rewards.py — Phase 4.

Covers:
  - calculate_step_reward: all event combinations
  - PacmanEnvironment: observation space, action space
  - reset() return types, shapes, and info keys
  - step() return types and shapes
  - Observation values always in [0, 1]
  - Wall collision: Pac-Man stays in place + HIT_WALL reward
  - Pellet consumption reward
  - Power pellet: activates power mode + ghost vulnerability
  - Ghost caught: terminated=True
  - Ghost caught by moving INTO ghost (pre-ghost-move collision)
  - Ghost caught by ghost moving INTO Pac-Man (post-ghost-move collision)
  - Maze cleared: terminated=True, CLEAR_MAZE reward
  - Episode truncation at max_steps
  - Multiple reset() calls are idempotent
  - Seeded resets are deterministic
  - render_mode=None is a no-op
  - close() before render is safe
  - Gymnasium check_env passes
  - repr string
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from config import (
    MAX_STEPS,
    MAZE_HEIGHT,
    MAZE_WIDTH,
    POWER_MODE_DURATION,
    REWARDS,
    ObservationConfig,
    Action,
)
from environment import PacmanEnvironment, _OBS_SIZE
from maze import Position
from rewards import calculate_step_reward


# ---------------------------------------------------------------------------
# Custom maze layouts for deterministic scenario tests
# ---------------------------------------------------------------------------

# Ghost at (2,1) has exactly one walkable neighbour: (1,1) = Pac-Man's spawn.
# Used to test "ghost moves into Pac-Man" (post-ghost-move collision).
_CAUGHT_LAYOUT = [
    [0, 0, 0, 0, 0],
    [0, 4, 0, 2, 0],   # Pac-Man at (1,1); power pellet at (1,3) for validation
    [0, 5, 0, 0, 0],   # Ghost at (2,1); only neighbour is (1,1)
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# Ghost at (3,3) has no walkable neighbours — stays forever.
# Pac-Man at (1,1) can eat all pellets by moving RIGHT four times.
_CLEARED_LAYOUT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 4, 1, 1, 1, 2, 0],   # spawn; pellets at (1,2-4); power pellet at (1,5)
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0],   # ghost completely surrounded by walls
    [0, 0, 0, 0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# Reward calculation tests
# ---------------------------------------------------------------------------

class TestCalculateStepReward:
    def test_step_penalty_always_applied(self) -> None:
        r = calculate_step_reward()
        assert r == pytest.approx(REWARDS.STEP)

    def test_pellet_reward(self) -> None:
        r = calculate_step_reward(pellet_eaten=True)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.PELLET)

    def test_power_pellet_reward(self) -> None:
        r = calculate_step_reward(power_pellet_eaten=True)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.POWER_PELLET)

    def test_hit_wall_penalty(self) -> None:
        r = calculate_step_reward(hit_wall=True)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.HIT_WALL)

    def test_caught_penalty(self) -> None:
        r = calculate_step_reward(caught=True)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.CAUGHT_BY_GHOST)

    def test_cleared_bonus(self) -> None:
        r = calculate_step_reward(cleared=True)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.CLEAR_MAZE)

    def test_eat_one_ghost(self) -> None:
        r = calculate_step_reward(ghosts_eaten=1)
        assert r == pytest.approx(REWARDS.STEP + REWARDS.EAT_GHOST)

    def test_eat_two_ghosts(self) -> None:
        r = calculate_step_reward(ghosts_eaten=2)
        assert r == pytest.approx(REWARDS.STEP + 2 * REWARDS.EAT_GHOST)

    def test_multiple_events_additive(self) -> None:
        r = calculate_step_reward(
            pellet_eaten=True,
            hit_wall=False,
            ghosts_eaten=1,
        )
        expected = REWARDS.STEP + REWARDS.PELLET + REWARDS.EAT_GHOST
        assert r == pytest.approx(expected)

    def test_cleared_pellet_same_step(self) -> None:
        """Eating the last pellet triggers both PELLET and CLEAR_MAZE rewards."""
        r = calculate_step_reward(pellet_eaten=True, cleared=True)
        expected = REWARDS.STEP + REWARDS.PELLET + REWARDS.CLEAR_MAZE
        assert r == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Environment — spaces and repr
# ---------------------------------------------------------------------------

class TestEnvironmentSpaces:
    def test_observation_space_shape(self) -> None:
        env = PacmanEnvironment()
        assert env.observation_space.shape == (_OBS_SIZE,)
        env.close()

    def test_observation_space_shape_is_723(self) -> None:
        assert _OBS_SIZE == MAZE_HEIGHT * MAZE_WIDTH * 5 + 3

    def test_optional_observation_features_extend_shape(self) -> None:
        env = PacmanEnvironment(
            observation_config=ObservationConfig(
                include_ghost_direction=True,
                include_previous_pacman_action=True,
            )
        )
        obs, _ = env.reset(seed=42)
        assert env.observation_space.shape == (_OBS_SIZE + 2 * 5 + 4,)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs[-14:] == 0.0)
        env.close()

    def test_optional_observation_features_track_actions_and_directions(self) -> None:
        env = PacmanEnvironment(
            observation_config=ObservationConfig(
                include_ghost_direction=True,
                include_previous_pacman_action=True,
            )
        )
        env.reset(seed=42)
        obs, *_ = env.step(Action.LEFT)
        assert obs[-4:].tolist() == [0.0, 0.0, 1.0, 0.0]
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_observation_space_dtype(self) -> None:
        env = PacmanEnvironment()
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_observation_space_bounds(self) -> None:
        env = PacmanEnvironment()
        assert env.observation_space.low.min() == pytest.approx(0.0)
        assert env.observation_space.high.max() == pytest.approx(1.0)
        env.close()

    def test_action_space_size(self) -> None:
        env = PacmanEnvironment()
        assert env.action_space.n == len(Action)  # 4
        env.close()

    def test_action_space_n_is_4(self) -> None:
        env = PacmanEnvironment()
        assert env.action_space.n == 4
        env.close()

    def test_unsupported_render_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="render_mode"):
            PacmanEnvironment(render_mode="rgb_array")

    def test_repr(self) -> None:
        env = PacmanEnvironment()
        assert "PacmanEnvironment" in repr(env)
        env.close()


# ---------------------------------------------------------------------------
# Environment — reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_tuple(self) -> None:
        env = PacmanEnvironment()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        env.close()

    def test_reset_obs_shape(self) -> None:
        env = PacmanEnvironment()
        obs, _ = env.reset()
        assert obs.shape == (_OBS_SIZE,)
        env.close()

    def test_reset_obs_dtype(self) -> None:
        env = PacmanEnvironment()
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_reset_obs_in_bounds(self) -> None:
        env = PacmanEnvironment()
        obs, _ = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        env.close()

    def test_reset_info_has_required_keys(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset()
        required = {
            "step", "episode_reward", "pellets_remaining",
            "total_pellets", "completion_fraction",
            "power_mode_active", "power_mode_steps_remaining",
            "pacman_position", "ghost_positions",
        }
        assert required <= set(info.keys())
        env.close()

    def test_reset_step_is_zero(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset()
        assert info["step"] == 0
        env.close()

    def test_reset_episode_reward_is_zero(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset()
        assert info["episode_reward"] == pytest.approx(0.0)
        env.close()

    def test_reset_pellets_remaining_equals_total(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset()
        assert info["pellets_remaining"] == info["total_pellets"]
        env.close()

    def test_reset_pacman_at_spawn(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset()
        from maze import Maze
        spawn = Maze().pacman_spawn()
        assert info["pacman_position"] == (spawn.row, spawn.col)
        env.close()

    def test_multiple_resets_consistent(self) -> None:
        env = PacmanEnvironment()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    def test_seeded_reset_deterministic(self) -> None:
        env1 = PacmanEnvironment()
        env2 = PacmanEnvironment()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()


# ---------------------------------------------------------------------------
# Environment — step
# ---------------------------------------------------------------------------

class TestStep:
    def setup_method(self) -> None:
        self.env = PacmanEnvironment()
        self.env.reset(seed=0)

    def teardown_method(self) -> None:
        self.env.close()

    def test_step_returns_five_tuple(self) -> None:
        result = self.env.step(Action.UP)
        assert len(result) == 5

    def test_step_obs_shape(self) -> None:
        obs, *_ = self.env.step(Action.UP)
        assert obs.shape == (_OBS_SIZE,)

    def test_step_obs_dtype(self) -> None:
        obs, *_ = self.env.step(Action.UP)
        assert obs.dtype == np.float32

    def test_step_obs_in_bounds(self) -> None:
        obs, *_ = self.env.step(Action.UP)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_step_reward_is_float(self) -> None:
        _, reward, *_ = self.env.step(Action.UP)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self) -> None:
        _, _, terminated, *_ = self.env.step(Action.UP)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self) -> None:
        _, _, _, truncated, _ = self.env.step(Action.UP)
        assert isinstance(truncated, bool)

    def test_step_before_reset_raises(self) -> None:
        env = PacmanEnvironment()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)
        env.close()

    def test_step_increments_step_count(self) -> None:
        _, _, _, _, info = self.env.step(Action.UP)
        assert info["step"] == 1

    def test_obs_in_bounds_over_many_steps(self) -> None:
        """Observation must stay within [0,1] throughout an episode."""
        for _ in range(50):
            obs, _, terminated, truncated, _ = self.env.step(
                self.env.action_space.sample()
            )
            assert obs.min() >= 0.0
            assert obs.max() <= 1.0
            if terminated or truncated:
                self.env.reset()


# ---------------------------------------------------------------------------
# Wall collision
# ---------------------------------------------------------------------------

class TestWallCollision:
    def test_wall_collision_stays_in_place(self) -> None:
        """Action DOWN from Pac-Man's default spawn (10,5) hits row 11 wall."""
        env = PacmanEnvironment()
        _, info = env.reset(seed=0)
        start = info["pacman_position"]
        # (10,5) → DOWN → (11,5) is a wall
        env.step(Action.DOWN)
        _, _, _, _, info2 = env.step(Action.DOWN)
        # After two DOWN attempts, Pac-Man should still be at original row
        assert info2["pacman_position"] == start
        env.close()

    def test_wall_collision_reward_contains_penalty(self) -> None:
        """Stepping into a wall produces STEP + HIT_WALL reward."""
        env = PacmanEnvironment()
        env.reset(seed=0)
        # Move into the bottom wall from spawn
        _, reward, _, _, _ = env.step(Action.DOWN)
        expected = REWARDS.STEP + REWARDS.HIT_WALL
        assert reward == pytest.approx(expected)
        env.close()


# ---------------------------------------------------------------------------
# Pellet reward
# ---------------------------------------------------------------------------

class TestPelletReward:
    def test_move_to_pellet_gives_pellet_reward(self) -> None:
        """Moving UP from (10,5) reaches (9,5) which contains a pellet."""
        env = PacmanEnvironment()
        env.reset(seed=0)
        _, reward, _, _, info = env.step(Action.UP)
        # Reward should include STEP + PELLET (no wall)
        assert reward == pytest.approx(REWARDS.STEP + REWARDS.PELLET)
        assert info["pellets_remaining"] < info["total_pellets"]
        env.close()

    def test_pellet_removed_after_eating(self) -> None:
        env = PacmanEnvironment()
        _, info0 = env.reset(seed=0)
        initial = info0["pellets_remaining"]
        env.step(Action.UP)
        _, _, _, _, info1 = env.step(Action.DOWN)  # move back
        # One pellet was eaten
        assert info1["pellets_remaining"] == initial - 1
        env.close()


# ---------------------------------------------------------------------------
# Power pellet
# ---------------------------------------------------------------------------

class TestPowerPellet:
    def test_power_mode_activates_on_consumption(self) -> None:
        """Eating a power pellet sets power_mode_active=True.

        Uses _CLEARED_LAYOUT where Pac-Man's 4th RIGHT step lands on (1,5),
        the power pellet cell, and the ghost is safely trapped.
        """
        env = PacmanEnvironment(maze_layout=_CLEARED_LAYOUT)
        env.reset(seed=0)
        # Steps 1–3: eat regular pellets at (1,2), (1,3), (1,4)
        for _ in range(3):
            env.step(Action.RIGHT)
        # Step 4: land on power pellet at (1,5)
        _, _, _, _, info = env.step(Action.RIGHT)
        assert info["power_mode_active"] is True
        assert info["power_mode_steps_remaining"] == POWER_MODE_DURATION
        env.close()

    def test_power_mode_decrement(self) -> None:
        """Timer decreases by 1 each step while power mode is active."""
        env = PacmanEnvironment(maze_layout=_CLEARED_LAYOUT)
        env.reset(seed=0)
        for _ in range(3):
            env.step(Action.RIGHT)
        _, _, _, _, info = env.step(Action.RIGHT)  # activate at step 4
        assert info["power_mode_steps_remaining"] == POWER_MODE_DURATION
        # Take one more step (LEFT — stays at (1,5) since it hits wall, or RIGHT hits wall)
        _, _, _, _, info2 = env.step(Action.LEFT)  # move back to (1,4)
        assert info2["power_mode_steps_remaining"] == POWER_MODE_DURATION - 1
        env.close()

    def test_power_mode_inactive_initially(self) -> None:
        env = PacmanEnvironment()
        _, info = env.reset(seed=0)
        assert info["power_mode_active"] is False
        assert info["power_mode_steps_remaining"] == 0
        env.close()


# ---------------------------------------------------------------------------
# Caught by ghost
# ---------------------------------------------------------------------------

class TestCaughtByGhost:
    def test_pacman_moves_into_ghost_is_caught(self) -> None:
        """Pac-Man moving DOWN from (1,1) enters ghost at (2,1) — caught."""
        env = PacmanEnvironment(maze_layout=_CAUGHT_LAYOUT)
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(Action.DOWN)
        assert terminated is True
        assert reward == pytest.approx(
            REWARDS.STEP + REWARDS.CAUGHT_BY_GHOST
        )
        env.close()

    def test_ghost_moves_into_pacman_is_caught(self) -> None:
        """Ghost has only one neighbour: (1,1) where Pac-Man is.
        After Pac-Man stays (UP = wall), ghost must move to (1,1) → caught.
        """
        env = PacmanEnvironment(maze_layout=_CAUGHT_LAYOUT)
        env.reset(seed=0)
        # (1,1): UP→(0,1)=wall — Pac-Man stays; ghost moves from (2,1) to (1,1)
        _, reward, terminated, _, _ = env.step(Action.UP)
        assert terminated is True
        assert reward == pytest.approx(
            REWARDS.STEP + REWARDS.HIT_WALL + REWARDS.CAUGHT_BY_GHOST
        )
        env.close()

    def test_caught_resets_cleanly(self) -> None:
        """After being caught, reset() restores all state."""
        env = PacmanEnvironment(maze_layout=_CAUGHT_LAYOUT)
        env.reset(seed=0)
        env.step(Action.DOWN)  # caught
        obs, info = env.reset(seed=0)
        assert not info["power_mode_active"]
        assert info["step"] == 0
        # obs shape must match the environment's observation_space
        assert obs.shape == env.observation_space.shape
        env.close()


# ---------------------------------------------------------------------------
# Maze cleared
# ---------------------------------------------------------------------------

class TestMazeCleared:
    def test_maze_cleared_terminates_episode(self) -> None:
        """Eat all pellets → cleared=True → terminated."""
        env = PacmanEnvironment(maze_layout=_CLEARED_LAYOUT)
        env.reset(seed=0)
        # Pac-Man at (1,1); pellets at (1,2),(1,3),(1,4); power pellet at (1,5)
        # 4 RIGHT actions eat all pellets
        terminated = False
        for _ in range(4):
            _, _, terminated, _, _ = env.step(Action.RIGHT)
        assert terminated is True
        env.close()

    def test_maze_cleared_reward_includes_bonus(self) -> None:
        """The step that clears the maze includes CLEAR_MAZE in the reward."""
        env = PacmanEnvironment(maze_layout=_CLEARED_LAYOUT)
        env.reset(seed=0)
        for _ in range(3):
            env.step(Action.RIGHT)
        # 4th step clears the maze (eats power pellet at (1,5))
        _, reward, terminated, _, _ = env.step(Action.RIGHT)
        assert terminated is True
        # reward = STEP + POWER_PELLET + CLEAR_MAZE
        expected = REWARDS.STEP + REWARDS.POWER_PELLET + REWARDS.CLEAR_MAZE
        assert reward == pytest.approx(expected)
        env.close()

    def test_maze_cleared_completion_fraction_is_one(self) -> None:
        env = PacmanEnvironment(maze_layout=_CLEARED_LAYOUT)
        env.reset(seed=0)
        for _ in range(4):
            _, _, terminated, _, info = env.step(Action.RIGHT)
        assert info["completion_fraction"] == pytest.approx(1.0)
        env.close()


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_episode_truncates_at_max_steps(self) -> None:
        env = PacmanEnvironment(max_steps=3)
        env.reset(seed=0)
        truncated = False
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(Action.UP)
            if terminated:
                break
        assert truncated is True
        env.close()

    def test_truncated_not_terminated(self) -> None:
        env = PacmanEnvironment(max_steps=5)
        env.reset(seed=0)
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(Action.DOWN)
            if terminated:
                pytest.skip("Episode ended before truncation")
        # After 5 steps from (10,5) all hitting the same wall, should truncate
        assert truncated is True
        assert terminated is False
        env.close()


# ---------------------------------------------------------------------------
# Render and close
# ---------------------------------------------------------------------------

class TestRenderClose:
    def test_close_before_reset_is_safe(self) -> None:
        env = PacmanEnvironment()
        env.close()

    def test_render_none_mode_is_noop(self) -> None:
        env = PacmanEnvironment(render_mode=None)
        env.reset(seed=0)
        env.render()  # must not raise
        env.close()

    def test_render_before_reset_is_safe(self) -> None:
        env = PacmanEnvironment(render_mode=None)
        env.render()  # must not raise
        env.close()


# ---------------------------------------------------------------------------
# Gymnasium API compliance
# ---------------------------------------------------------------------------

class TestGymnasiumCompliance:
    def test_check_env_passes(self) -> None:
        """SB3's check_env validates the full Gymnasium API contract."""
        env = PacmanEnvironment()
        check_env(env, warn=True)
        env.close()
