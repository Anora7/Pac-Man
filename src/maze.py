"""
Maze representation for the Pac-Man RL environment.

This module owns:
  - Position: immutable (row, col) grid coordinate.
  - Maze: static layout + dynamic pellet state.

This module must not contain RL logic, rendering logic, or ghost logic.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from config import (
    ACTION_DELTAS,
    CellType,
    MAZE_HEIGHT,
    MAZE_WIDTH,
    MAZE_LAYOUT,
)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(NamedTuple):
    """An immutable (row, col) grid coordinate.

    Row 0 is the top of the grid; col 0 is the left edge.
    """

    row: int
    col: int

    def __add__(self, delta: tuple[int, int]) -> Position:  # type: ignore[override]
        """Return a new Position offset by (delta_row, delta_col)."""
        return Position(self.row + delta[0], self.col + delta[1])


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

class Maze:
    """Represents the static layout and dynamic pellet state of the maze.

    The static layout defines which cells are walls, walkable paths, spawn
    locations, and initial pellet positions. The layout never changes during
    an episode. The dynamic pellet grid tracks which pellets have been consumed.

    Attributes:
        height: Number of rows in the grid.
        width:  Number of columns in the grid.

    Example::

        maze = Maze()
        spawn = maze.pacman_spawn()       # Position(10, 5)
        maze.consume_pellet(Position(1, 1))
        maze.reset()                      # restore all pellets
    """

    def __init__(self, layout: list[list[int]] | None = None) -> None:
        """Initialise the maze from a layout grid.

        Args:
            layout: Optional 2-D list of CellType integer codes. When None,
                the default MAZE_LAYOUT from config.py is used.

        Raises:
            ValueError: If the layout is invalid (wrong shape, missing spawn
                cells, or unrecognised cell codes).
        """
        layout_data: list[list[int]] = layout if layout is not None else MAZE_LAYOUT
        self._layout: np.ndarray = np.array(layout_data, dtype=np.int8)

        self.height: int = int(self._layout.shape[0])
        self.width: int = int(self._layout.shape[1])

        self._validate_layout()

        # Cache total pellet count — constant across episodes.
        self._total_pellets: int = int(
            np.sum(
                (self._layout == CellType.PELLET)
                | (self._layout == CellType.POWER_PELLET)
            )
        )

        # Dynamic pellet state — reset at the start of each episode.
        self._pellets: np.ndarray = np.zeros((self.height, self.width), dtype=bool)
        self.reset()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_layout(self) -> None:
        """Check that the layout meets minimum structural requirements.

        Raises:
            ValueError: On any layout constraint violation.
        """
        valid_types = set(int(ct) for ct in CellType)
        unique_cells = set(int(v) for v in np.unique(self._layout))
        unknown = unique_cells - valid_types
        if unknown:
            raise ValueError(
                f"Layout contains unrecognised cell codes: {unknown}. "
                f"Valid codes are {valid_types}."
            )

        pacman_count = int(np.sum(self._layout == CellType.PACMAN_SPAWN))
        if pacman_count != 1:
            raise ValueError(
                f"Layout must contain exactly one PACMAN_SPAWN cell "
                f"(CellType={CellType.PACMAN_SPAWN}), found {pacman_count}."
            )

        ghost_count = int(np.sum(self._layout == CellType.GHOST_SPAWN))
        if ghost_count < 1:
            raise ValueError(
                "Layout must contain at least one GHOST_SPAWN cell "
                f"(CellType={CellType.GHOST_SPAWN})."
            )

        power_count = int(np.sum(self._layout == CellType.POWER_PELLET))
        if power_count < 1:
            raise ValueError(
                "Layout must contain at least one POWER_PELLET cell "
                f"(CellType={CellType.POWER_PELLET})."
            )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore all pellets to their initial state for a new episode."""
        self._pellets[:] = (
            (self._layout == CellType.PELLET)
            | (self._layout == CellType.POWER_PELLET)
        )

    # ------------------------------------------------------------------
    # Cell queries
    # ------------------------------------------------------------------

    def is_wall(self, pos: Position) -> bool:
        """Return True if the cell at *pos* is a wall."""
        return bool(self._layout[pos.row, pos.col] == CellType.WALL)

    def is_walkable(self, pos: Position) -> bool:
        """Return True if *pos* is within bounds and is not a wall.

        Args:
            pos: Grid coordinate to check.

        Returns:
            True when the cell can be entered by Pac-Man or a ghost.
        """
        if not (0 <= pos.row < self.height and 0 <= pos.col < self.width):
            return False
        return not self.is_wall(pos)

    # ------------------------------------------------------------------
    # Pellet queries
    # ------------------------------------------------------------------

    def has_pellet(self, pos: Position) -> bool:
        """Return True if an unconsumed regular pellet is present at *pos*."""
        return bool(
            self._pellets[pos.row, pos.col]
            and self._layout[pos.row, pos.col] == CellType.PELLET
        )

    def has_power_pellet(self, pos: Position) -> bool:
        """Return True if an unconsumed power pellet is present at *pos*."""
        return bool(
            self._pellets[pos.row, pos.col]
            and self._layout[pos.row, pos.col] == CellType.POWER_PELLET
        )

    def consume_pellet(self, pos: Position) -> CellType | None:
        """Remove the pellet at *pos* if one is present.

        Args:
            pos: Grid coordinate to consume.

        Returns:
            The ``CellType`` of the consumed pellet (``PELLET`` or
            ``POWER_PELLET``), or ``None`` if the cell has no unconsumed
            pellet.
        """
        if not self._pellets[pos.row, pos.col]:
            return None
        cell_type = CellType(int(self._layout[pos.row, pos.col]))
        self._pellets[pos.row, pos.col] = False
        return cell_type

    def pellets_remaining(self) -> int:
        """Return the total number of unconsumed pellets (regular + power)."""
        return int(self._pellets.sum())

    def is_cleared(self) -> bool:
        """Return True when every pellet has been consumed."""
        return self.pellets_remaining() == 0

    def nearest_pellet_path(
        self,
        start: Position,
    ) -> tuple[int | None, int | None]:
        """Return the first action and BFS distance to the nearest pellet.

        Ties follow the action order in ``ACTION_DELTAS``. If there are no
        reachable pellets, or *start* is itself a pellet, the first action is
        ``None``. The distance is ``None`` only when no pellet is reachable.
        """
        if not self.is_walkable(start):
            raise ValueError(f"Start position is not walkable: {start}.")

        queue: deque[tuple[Position, int | None, int]] = deque(
            [(start, None, 0)]
        )
        visited = {start}
        while queue:
            position, first_action, distance = queue.popleft()
            if self._pellets[position.row, position.col]:
                return first_action, distance
            for action, delta in ACTION_DELTAS.items():
                neighbour = position + delta
                if (
                    neighbour not in visited
                    and self.is_walkable(neighbour)
                ):
                    visited.add(neighbour)
                    queue.append((
                        neighbour,
                        int(action) if first_action is None else first_action,
                        distance + 1,
                    ))
        return None, None

    def distances_from(self, start: Position) -> dict[Position, int]:
        """Return shortest walkable-cell distances from *start* using BFS."""
        if not self.is_walkable(start):
            raise ValueError(f"Start position is not walkable: {start}.")

        distances = {start: 0}
        queue: deque[Position] = deque([start])
        while queue:
            position = queue.popleft()
            for delta in ACTION_DELTAS.values():
                neighbour = position + delta
                if (
                    neighbour not in distances
                    and self.is_walkable(neighbour)
                ):
                    distances[neighbour] = distances[position] + 1
                    queue.append(neighbour)
        return distances

    # ------------------------------------------------------------------
    # Spawn locations
    # ------------------------------------------------------------------

    def pacman_spawn(self) -> Position:
        """Return the unique Pac-Man spawn position.

        Returns:
            A ``Position`` representing Pac-Man's starting cell.
        """
        return self._find_cells(CellType.PACMAN_SPAWN)[0]

    def ghost_spawns(self) -> list[Position]:
        """Return all ghost spawn positions in row-major order.

        Returns:
            A list of ``Position`` objects, one per ghost spawn cell.
        """
        return self._find_cells(CellType.GHOST_SPAWN)

    def _find_cells(self, cell_type: CellType) -> list[Position]:
        """Return all grid positions matching *cell_type* in row-major order."""
        rows, cols = np.where(self._layout == int(cell_type))
        return [Position(int(r), int(c)) for r, c in zip(rows, cols)]

    # ------------------------------------------------------------------
    # Observation grids (used by environment.py)
    # ------------------------------------------------------------------

    def wall_grid(self) -> np.ndarray:
        """Return a float32 grid where 1.0 marks wall cells.

        Returns:
            Array of shape (height, width) with dtype float32.
        """
        return (self._layout == CellType.WALL).astype(np.float32)

    def pellet_grid(self) -> np.ndarray:
        """Return a float32 grid of unconsumed regular pellet positions.

        Returns:
            Array of shape (height, width) with dtype float32.
        """
        return (
            self._pellets & (self._layout == CellType.PELLET)
        ).astype(np.float32)

    def power_pellet_grid(self) -> np.ndarray:
        """Return a float32 grid of unconsumed power pellet positions.

        Returns:
            Array of shape (height, width) with dtype float32.
        """
        return (
            self._pellets & (self._layout == CellType.POWER_PELLET)
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_pellets(self) -> int:
        """Total number of pellets at episode start (constant)."""
        return self._total_pellets

    @property
    def completion_fraction(self) -> float:
        """Fraction of pellets consumed so far in the current episode.

        Returns:
            A float in [0.0, 1.0]; 1.0 means the maze is cleared.
        """
        if self._total_pellets == 0:
            return 1.0
        return 1.0 - self.pellets_remaining() / self._total_pellets

    @property
    def layout(self) -> np.ndarray:
        """Read-only view of the static cell-type layout."""
        view = self._layout.view()
        view.flags.writeable = False
        return view

    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the grid."""
        return (self.height, self.width)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Maze(height={self.height}, width={self.width}, "
            f"pellets_remaining={self.pellets_remaining()}/{self._total_pellets})"
        )
