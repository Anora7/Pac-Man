"""Tests for interchangeable ghost movement policies."""

from __future__ import annotations

import numpy as np

from ghost import Ghost
from ghost_policies import ChaseGhostPolicy, RandomGhostPolicy, RuleBasedGhostPolicy
from maze import Maze, Position


_LAYOUT = [
    [0, 0, 0, 0, 0],
    [0, 4, 1, 2, 0],
    [0, 1, 1, 1, 0],
    [0, 5, 1, 1, 0],
    [0, 0, 0, 0, 0],
]


def test_chase_policy_moves_closer_to_pacman() -> None:
    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(3, 1), policy=ChaseGhostPolicy())
    ghost.move(maze, Position(1, 1), np.random.default_rng(42))
    assert ghost.position == Position(2, 1)


def test_random_policy_stays_in_walkable_cells() -> None:
    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(3, 1), policy=RandomGhostPolicy())
    ghost.move(maze, Position(1, 1), np.random.default_rng(42))
    assert maze.is_walkable(ghost.position)


def test_rule_policy_keeps_original_defaults() -> None:
    policy = RuleBasedGhostPolicy()
    assert policy.chase_probability == 0.30
    assert policy.chase_radius == 5


def test_invalid_policy_action_does_not_move_ghost() -> None:
    class InvalidPolicy:
        def choose_action(self, *args: object, **kwargs: object) -> int:
            return 0

    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(1, 1), policy=InvalidPolicy())
    ghost.move(maze, Position(3, 1), np.random.default_rng(42))
    assert ghost.position == Position(1, 1)
