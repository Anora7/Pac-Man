"""
Unit tests for GoogleMaze (16x16 board, pellet reachability, BFS pathfinding).
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CellType, Action
from google_config import (
    GOOGLE_MAZE_HEIGHT,
    GOOGLE_MAZE_WIDTH,
    GOOGLE_MAZE_LAYOUT,
)
from google_maze import GoogleMaze
from maze import Position


class TestGoogleMaze:
    def test_dimensions_and_initialization(self) -> None:
        maze = GoogleMaze()
        assert maze.height == GOOGLE_MAZE_HEIGHT == 16
        assert maze.width == GOOGLE_MAZE_WIDTH == 16
        assert maze.pellets_remaining() > 0

    def test_power_pellet_count_and_locations(self) -> None:
        maze = GoogleMaze()
        power_cells = maze._find_cells(CellType.POWER_PELLET)
        assert len(power_cells) == 4
        # Verify 4 power pellets near corners
        corners = {(1, 1), (1, 14), (14, 1), (14, 14)}
        found = {(p.row, p.col) for p in power_cells}
        assert found == corners

    def test_ghost_house_and_spawns(self) -> None:
        maze = GoogleMaze()
        spawns = maze.ghost_spawns()
        assert len(spawns) == 4
        doors = maze.ghost_house_doors()
        assert len(doors) >= 1
        for door in doors:
            assert maze.is_walkable(door)

    def test_pacman_spawn(self) -> None:
        maze = GoogleMaze()
        spawn = maze.pacman_spawn()
        assert maze.is_walkable(spawn)
        assert spawn == Position(12, 7)

    def test_every_pellet_is_reachable_with_bfs(self) -> None:
        """Requirement A: Every pellet must be reachable (verified with BFS)."""
        maze = GoogleMaze()
        assert maze.verify_all_pellets_reachable() is True

    def test_no_dead_end_traps(self) -> None:
        """Requirement A: Proper maze with corridors and no dead-end-only traps."""
        maze = GoogleMaze()
        for r in range(maze.height):
            for c in range(maze.width):
                pos = Position(r, c)
                # Ignore ghost spawns inside the house
                if maze._layout[r, c] == CellType.GHOST_SPAWN:
                    continue
                if maze.is_walkable(pos):
                    walkable_neighbors = sum(
                        1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if maze.is_walkable(Position(r + dr, c + dc))
                    )
                    assert walkable_neighbors >= 2, f"Dead end found at {pos}"

    def test_pellet_consumption_and_reset(self) -> None:
        maze = GoogleMaze()
        initial_pellets = maze.pellets_remaining()
        pos = Position(1, 2)
        assert maze.has_pellet(pos)

        consumed = maze.consume_pellet(pos)
        assert consumed == CellType.PELLET
        assert not maze.has_pellet(pos)
        assert maze.pellets_remaining() == initial_pellets - 1

        maze.reset()
        assert maze.has_pellet(pos)
        assert maze.pellets_remaining() == initial_pellets

    def test_closest_walkable_projection(self) -> None:
        maze = GoogleMaze()
        wall_pos = Position(0, 0)
        closest = maze.closest_walkable(wall_pos)
        assert maze.is_walkable(closest)

    def test_bfs_shortest_path_distance(self) -> None:
        maze = GoogleMaze()
        p1 = Position(1, 1)
        p2 = Position(1, 3)
        dist = maze.bfs_shortest_path_distance(p1, p2)
        assert dist == 2

