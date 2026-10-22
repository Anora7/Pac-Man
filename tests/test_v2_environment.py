"""
Unit tests for GoogleEnvironment (Gymnasium interface, step lifecycle, collisions, rewards).
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Action, CellType
from v2_config import GOOGLE_MAX_STEPS
from v2_environment import GoogleEnvironment
from maze import Position


class TestGoogleEnvironment:
    def test_environment_initialization_and_spaces(self) -> None:
        env = GoogleEnvironment()
        assert env.action_space.n == 4
        # 5*16*16 (1280) + 3 + 20 + 5 + 9 + 4 = 1321
        assert env.observation_space.shape == (1321,)
        assert env.max_steps == GOOGLE_MAX_STEPS

    def test_reset(self) -> None:
        env = GoogleEnvironment()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1321,)
        assert np.all((obs >= 0.0) & (obs <= 1.0))
        assert "pellets_remaining" in info
        assert "total_pellets" in info
        assert info["pellets_remaining"] == info["total_pellets"]

    def test_step_pellet_consumption(self) -> None:
        env = GoogleEnvironment()
        obs, info = env.reset(seed=42)
        initial_pellets = info["pellets_remaining"]

        # Pacman spawn is at (12, 7). Col 8 has a pellet.
        # Action.RIGHT moves Pacman to (12, 8)
        obs, reward, terminated, truncated, info = env.step(int(Action.RIGHT))
        assert env.pacman_pos == Position(12, 8)
        assert info["pellets_remaining"] == initial_pellets - 1
        assert info["pellet_eaten"] is True
        assert reward > 0

    def test_step_wall_collision(self) -> None:
        env = GoogleEnvironment()
        env.reset(seed=42)
        # Position (1, 1) has a power pellet; consume it first so only wall penalty applies
        env.maze.consume_pellet(Position(1, 1))
        env.pacman_pos = Position(1, 1)  # (0, 1) is top wall
        obs, reward, terminated, truncated, info = env.step(int(Action.UP))
        assert env.pacman_pos == Position(1, 1)  # Did not move into wall
        assert reward < 0  # Wall penalty applied

    def test_power_pellet_activation_and_ghost_eating(self) -> None:
        env = GoogleEnvironment(power_mode_duration=10)
        env.reset(seed=42)
        # Power pellet at (1, 1). Move Pacman there.
        env.pacman_pos = Position(1, 2)
        obs, reward, terminated, truncated, info = env.step(int(Action.LEFT))
        assert env.pacman_pos == Position(1, 1)
        assert env.power_mode_active is True
        assert env.power_mode_steps_remaining == 10
        assert info["power_pellet_eaten"] is True
        for ghost in env.ghosts:
            if ghost.is_released:
                assert ghost.is_vulnerable is True

        # Now place a vulnerable ghost at (1, 2) and have Pacman move RIGHT into it
        ghost = env.ghosts[0]
        ghost.position = Position(1, 2)
        ghost.is_released = True
        ghost.is_vulnerable = True
        obs, reward, terminated, truncated, info = env.step(int(Action.RIGHT))
        assert info["ghosts_eaten"] == 1
        assert not terminated  # Pacman ate ghost, not caught

    def test_normal_ghost_catches_pacman(self) -> None:
        env = GoogleEnvironment()
        env.reset(seed=42)
        ghost = env.ghosts[0]
        ghost.is_released = True
        ghost.is_vulnerable = False
        # Place ghost right in front of Pac-man
        pac_pos = env.pacman_pos
        assert pac_pos is not None
        ghost.position = Position(pac_pos.row, pac_pos.col + 1)
        # Pacman steps right into ghost
        obs, reward, terminated, truncated, info = env.step(int(Action.RIGHT))
        assert terminated is True
        assert info["caught"] is True
        assert info["is_success"] is False
        assert reward <= -100.0

    def test_truncation_at_max_steps(self) -> None:
        env = GoogleEnvironment(max_steps=5)
        env.reset(seed=42)
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(int(Action.LEFT))
            if terminated:
                break
        if not terminated:
            assert truncated is True
