"""
16x16 Maze representation for the Google Pac-Man RL environment.

Subclasses Maze from maze.py to provide board customization,
BFS reachability verification, and target-tile projection helpers.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from config import CellType, ACTION_DELTAS, Action
from google_config import (
    GOOGLE_MAZE_HEIGHT,
    GOOGLE_MAZE_WIDTH,
    GOOGLE_MAZE_LAYOUT,
)
from maze import Maze, Position


class GoogleMaze(Maze):
    """Subclass of Maze implementing the 16x16 Google Pac-Man board.

    Attributes:
        height: Maze height in rows (default 16).
        width:  Maze width in columns (default 16).
    """

    def __init__(self, layout: list[list[int]] | None = None) -> None:
        """Initialise GoogleMaze.

        Args:
            layout: Optional 2-D list of CellType codes. If None,
                    the 16x16 GOOGLE_MAZE_LAYOUT is used.
        """
        layout_data = layout if layout is not None else GOOGLE_MAZE_LAYOUT
        super().__init__(layout=layout_data)

    @property
    def total_pellets(self) -> int:
        """Return cached total pellets in the layout."""
        return self._total_pellets

    def ghost_house_doors(self) -> list[Position]:
        """Return positions of the ghost house exit doors (CellType.EMPTY)."""
        return self._find_cells(CellType.EMPTY)

    def verify_all_pellets_reachable(self) -> bool:
        """Verify using BFS that every pellet on the board can be reached by Pac-Man.

        Returns:
            True if all regular and power pellets are reachable from Pac-Man's spawn.
        """
        spawn = self.pacman_spawn()
        reachable = self.distances_from(spawn)

        # Iterate over all cells containing pellets
        for row in range(self.height):
            for col in range(self.width):
                pos = Position(row, col)
                if (
                    self._layout[row, col] == CellType.PELLET
                    or self._layout[row, col] == CellType.POWER_PELLET
                ):
                    if pos not in reachable:
                        return False
        return True

    def closest_walkable(self, target: Position) -> Position:
        """Project an arbitrary (possibly wall/out-of-bounds) target to the closest walkable cell.

        Args:
            target: Coordinate (may be out of bounds or inside a wall).

        Returns:
            The walkable Position with minimum Euclidean distance to target.
        """
        # Clamp target coordinate inside bounds first
        clamped_r = max(0, min(self.height - 1, target.row))
        clamped_c = max(0, min(self.width - 1, target.col))
        clamped_target = Position(clamped_r, clamped_c)

        if self.is_walkable(clamped_target):
            return clamped_target

        best_pos = self.pacman_spawn()
        min_dist = float("inf")

        for r in range(self.height):
            for c in range(self.width):
                pos = Position(r, c)
                if self.is_walkable(pos):
                    dist = (pos.row - target.row) ** 2 + (pos.col - target.col) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = pos

        return best_pos

    def bfs_shortest_path_distance(
        self,
        start: Position,
        goal: Position,
    ) -> int | None:
        """Return the BFS shortest path distance from start to goal.

        Args:
            start: Starting walkable position.
            goal: Destination walkable position.

        Returns:
            Step distance as an integer, or None if unreachable.
        """
        if not self.is_walkable(start):
            return None
        if not self.is_walkable(goal):
            goal = self.closest_walkable(goal)

        if start == goal:
            return 0

        distances = self.distances_from(start)
        return distances.get(goal, None)

