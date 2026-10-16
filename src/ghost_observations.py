"""Shared ghost-centred observation encoding without an SB3 dependency."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maze import Maze, Position


def build_ghost_observation(
    maze: Maze,
    ghost_positions: Sequence[Position],
    ghost_index: int,
    pacman_position: Position,
    power_mode_active: bool,
) -> np.ndarray:
    """Encode one ghost-centred state for training and policy inference."""
    if not 0 <= ghost_index < len(ghost_positions):
        raise IndexError("ghost_index is outside the ghost positions.")
    own = np.zeros((maze.height, maze.width), dtype=np.float32)
    own_pos = ghost_positions[ghost_index]
    own[own_pos.row, own_pos.col] = 1.0
    pacman = np.zeros((maze.height, maze.width), dtype=np.float32)
    pacman[pacman_position.row, pacman_position.col] = 1.0
    others = np.zeros((maze.height, maze.width), dtype=np.float32)
    for other_index, position in enumerate(ghost_positions):
        if other_index != ghost_index:
            others[position.row, position.col] = 1.0
    total = maze.total_pellets
    return np.concatenate(
        [
            maze.wall_grid().ravel(),
            maze.pellet_grid().ravel(),
            maze.power_pellet_grid().ravel(),
            own.ravel(),
            pacman.ravel(),
            others.ravel(),
            np.asarray(
                [
                    float(power_mode_active),
                    maze.pellets_remaining() / total if total else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
