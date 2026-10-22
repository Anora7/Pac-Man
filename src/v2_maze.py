"""
16x16 Maze representation for the Google Pac-Man RL environment.

Subclasses Maze from maze.py to provide board customization,
precomputed BFS distance map caching for high speed, fair-start
verification (Pac-Man start >= 8 BFS steps from all ghost starts),
and target-tile projection helpers.
"""

from __future__ import annotations

from collections import deque
import numpy as np

from config import CellType, ACTION_DELTAS, Action
from v2_config import (
    GOOGLE_MAZE_HEIGHT,
    GOOGLE_MAZE_WIDTH,
    GOOGLE_MAZE_LAYOUT,
)
from maze import Maze, Position


class GoogleMaze(Maze):
    """Subclass of Maze implementing the 16x16 Google Pac-Man board.

    Precomputes and caches all-pairs BFS shortest distance maps once at
    construction time for maximal execution speed.

    Attributes:
        height: Maze height in rows (default 16).
        width:  Maze width in columns (default 16).
    """

    def __init__(self, layout: list[list[int]] | None = None) -> None:
        """Initialise GoogleMaze and cache BFS distance maps.

        Args:
            layout: Optional 2-D list of CellType codes. If None,
                    the 16x16 GOOGLE_MAZE_LAYOUT is used.
        """
        layout_data = layout if layout is not None else GOOGLE_MAZE_LAYOUT
        super().__init__(layout=layout_data)

        # Precompute BFS distance maps for all walkable cells
        self._distance_maps: dict[Position, dict[Position, int]] = {}
        self._precompute_distance_maps()

        # Enforce fair start rule
        if not self.verify_fair_start(min_distance=8):
            raise ValueError(
                "Pac-Man start tile must be at least 8 BFS steps away from every ghost start."
            )

    def _precompute_distance_maps(self) -> None:
        """Precompute all-pairs shortest walkable distances using BFS once."""
        self._distance_maps.clear()
        for r in range(self.height):
            for c in range(self.width):
                pos = Position(r, c)
                if self.is_walkable(pos):
                    self._distance_maps[pos] = super().distances_from(pos)

    def distances_from(self, start: Position) -> dict[Position, int]:
        """Return shortest walkable-cell distances from *start* using cached BFS map.

        Args:
            start: Walkable Position.

        Returns:
            Dict mapping reachable Position -> integer step distance.
        """
        if not self.is_walkable(start):
            raise ValueError(f"Start position is not walkable: {start}.")
        return self._distance_maps[start]

    def nearest_pellet_path(
        self,
        start: Position,
    ) -> tuple[int | None, int | None]:
        """Return the first action and BFS distance to the nearest pellet using cached map.

        Args:
            start: Walkable Position.

        Returns:
            Tuple of (first_action_int or None, distance_int or None).
        """
        if not self.is_walkable(start):
            raise ValueError(f"Start position is not walkable: {start}.")

        if self._pellets[start.row, start.col]:
            return None, 0

        rows, cols = np.where(self._pellets)
        if len(rows) == 0:
            return None, None

        dist_map = self._distance_maps[start]
        active_pellet_dists = [
            (dist_map[Position(int(r), int(c))], Position(int(r), int(c)))
            for r, c in zip(rows, cols)
            if Position(int(r), int(c)) in dist_map
        ]
        if not active_pellet_dists:
            return None, None

        min_dist = min(d for d, _ in active_pellet_dists)
        target_pellets = {p for d, p in active_pellet_dists if d == min_dist}

        for action, delta in ACTION_DELTAS.items():
            neighbour = start + delta
            if self.is_walkable(neighbour):
                n_dist_map = self._distance_maps[neighbour]
                if any(n_dist_map.get(tp) == min_dist - 1 for tp in target_pellets):
                    return int(action), min_dist

        return None, min_dist

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

    def verify_fair_start(self, min_distance: int = 8) -> bool:
        """Verify Pac-Man's spawn is at least min_distance BFS steps from all ghost spawns.

        Args:
            min_distance: Minimum BFS steps required (default 8).

        Returns:
            True if all ghost spawns are >= min_distance away from Pac-Man's spawn.
        """
        spawn = self.pacman_spawn()
        dists = self.distances_from(spawn)
        for g_spawn in self.ghost_spawns():
            d = dists.get(g_spawn)
            if d is None or d < min_distance:
                return False
        return True

    def closest_walkable(self, target: Position) -> Position:
        """Project an arbitrary (possibly wall/out-of-bounds) target to the closest walkable cell.

        Args:
            target: Coordinate (may be out of bounds or inside a wall).

        Returns:
            The walkable Position with minimum Euclidean distance to target.
        """
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
        """Return the BFS shortest path distance from start to goal via cached lookup.

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

        return self._distance_maps[start].get(goal, None)
