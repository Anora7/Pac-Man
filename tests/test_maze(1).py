"""
Tests for src/maze.py and src/config.py — Phase 2: Maze Representation.

Covers:
  - Maze initialisation with default and custom layouts
  - Wall and walkability queries
  - Spawn location correctness
  - Pellet presence, consumption, and reset
  - Power pellet detection
  - Maze-cleared detection
  - Observation grids (wall, pellet, power-pellet)
  - Boundary and out-of-bounds behaviour
  - Layout validation (missing spawns, unknown cell codes)
  - Position arithmetic
  - Config constants sanity checks
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    ACTION_DELTAS,
    MAZE_HEIGHT,
    MAZE_LAYOUT,
    MAZE_WIDTH,
    REWARDS,
    Action,
    CellType,
    RewardConfig,
)
from maze import Maze, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_layout(pacman_row: int = 1, pacman_col: int = 1) -> list[list[int]]:
    """Return a minimal 3×3 layout usable for simple structural tests."""
    return [
        [0, 0, 0],
        [0, CellType.PACMAN_SPAWN, CellType.GHOST_SPAWN],
        [0, CellType.POWER_PELLET, 0],
    ]


# ---------------------------------------------------------------------------
# Config sanity checks
# ---------------------------------------------------------------------------

class TestConfig:
    def test_maze_dimensions_match_layout(self) -> None:
        assert len(MAZE_LAYOUT) == MAZE_HEIGHT
        assert all(len(row) == MAZE_WIDTH for row in MAZE_LAYOUT)

    def test_reward_config_is_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            REWARDS.PELLET = 999.0  # type: ignore[misc]

    def test_reward_values_are_expected(self) -> None:
        assert REWARDS.PELLET == 10.0
        assert REWARDS.POWER_PELLET == 50.0
        assert REWARDS.EAT_GHOST == 100.0
        assert REWARDS.CLEAR_MAZE == 500.0
        assert REWARDS.CAUGHT_BY_GHOST == -200.0
        assert REWARDS.HIT_WALL == -5.0
        assert REWARDS.STEP == -1.0

    def test_action_deltas_cover_all_actions(self) -> None:
        assert set(ACTION_DELTAS.keys()) == set(Action)

    def test_action_deltas_are_unit_steps(self) -> None:
        for delta in ACTION_DELTAS.values():
            assert abs(delta[0]) + abs(delta[1]) == 1

    def test_cell_type_values(self) -> None:
        assert CellType.WALL == 0
        assert CellType.PELLET == 1
        assert CellType.POWER_PELLET == 2
        assert CellType.EMPTY == 3
        assert CellType.PACMAN_SPAWN == 4
        assert CellType.GHOST_SPAWN == 5


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_creation(self) -> None:
        pos = Position(3, 7)
        assert pos.row == 3
        assert pos.col == 7

    def test_named_tuple_unpacking(self) -> None:
        row, col = Position(2, 5)
        assert row == 2
        assert col == 5

    def test_equality(self) -> None:
        assert Position(1, 2) == Position(1, 2)
        assert Position(1, 2) != Position(2, 1)

    def test_addition_with_delta(self) -> None:
        pos = Position(3, 3)
        assert pos + (-1, 0) == Position(2, 3)   # UP
        assert pos + (1, 0) == Position(4, 3)    # DOWN
        assert pos + (0, -1) == Position(3, 2)   # LEFT
        assert pos + (0, 1) == Position(3, 4)    # RIGHT

    def test_addition_with_action_delta(self) -> None:
        pos = Position(5, 5)
        assert pos + ACTION_DELTAS[Action.UP] == Position(4, 5)
        assert pos + ACTION_DELTAS[Action.DOWN] == Position(6, 5)
        assert pos + ACTION_DELTAS[Action.LEFT] == Position(5, 4)
        assert pos + ACTION_DELTAS[Action.RIGHT] == Position(5, 6)

    def test_hashable(self) -> None:
        positions = {Position(0, 0), Position(1, 1), Position(0, 0)}
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# Maze initialisation
# ---------------------------------------------------------------------------

class TestMazeInit:
    def test_default_maze_shape(self) -> None:
        maze = Maze()
        assert maze.height == 12
        assert maze.width == 12
        assert maze.shape == (12, 12)

    def test_custom_layout_shape(self) -> None:
        maze = Maze(layout=_minimal_layout())
        assert maze.height == 3
        assert maze.width == 3

    def test_layout_property_is_read_only(self) -> None:
        maze = Maze()
        with pytest.raises(ValueError, match="read-only"):
            maze.layout[0, 0] = 99

    def test_repr_contains_key_info(self) -> None:
        maze = Maze()
        text = repr(maze)
        assert "Maze" in text
        assert "12" in text

    def test_total_pellets_positive(self) -> None:
        maze = Maze()
        assert maze.total_pellets > 0


# ---------------------------------------------------------------------------
# Layout validation
# ---------------------------------------------------------------------------

class TestLayoutValidation:
    def test_missing_pacman_spawn_raises(self) -> None:
        bad = [[0, 0, 0], [0, CellType.GHOST_SPAWN, 0], [0, CellType.POWER_PELLET, 0]]
        with pytest.raises(ValueError, match="PACMAN_SPAWN"):
            Maze(layout=bad)

    def test_duplicate_pacman_spawn_raises(self) -> None:
        bad = [
            [0, 0, 0, 0],
            [0, CellType.PACMAN_SPAWN, CellType.PACMAN_SPAWN, 0],
            [0, CellType.GHOST_SPAWN, 0, 0],
            [0, CellType.POWER_PELLET, 0, 0],
        ]
        with pytest.raises(ValueError, match="PACMAN_SPAWN"):
            Maze(layout=bad)

    def test_missing_ghost_spawn_raises(self) -> None:
        bad = [[0, 0, 0], [0, CellType.PACMAN_SPAWN, 0], [0, CellType.POWER_PELLET, 0]]
        with pytest.raises(ValueError, match="GHOST_SPAWN"):
            Maze(layout=bad)

    def test_missing_power_pellet_raises(self) -> None:
        bad = [[0, 0, 0], [0, CellType.PACMAN_SPAWN, CellType.GHOST_SPAWN], [0, 0, 0]]
        with pytest.raises(ValueError, match="POWER_PELLET"):
            Maze(layout=bad)

    def test_unknown_cell_code_raises(self) -> None:
        bad = [[0, 0, 0], [0, CellType.PACMAN_SPAWN, 99], [0, CellType.POWER_PELLET, CellType.GHOST_SPAWN]]
        with pytest.raises(ValueError, match="unrecognised"):
            Maze(layout=bad)


# ---------------------------------------------------------------------------
# Default maze — structural correctness
# ---------------------------------------------------------------------------

class TestDefaultMazeStructure:
    """Validates that the default MAZE_LAYOUT encodes the expected structure."""

    def setup_method(self) -> None:
        self.maze = Maze()

    def test_border_cells_are_walls(self) -> None:
        maze = self.maze
        for col in range(maze.width):
            assert maze.is_wall(Position(0, col)), f"Top border col {col} should be wall"
            assert maze.is_wall(Position(maze.height - 1, col)), f"Bottom border col {col} should be wall"
        for row in range(maze.height):
            assert maze.is_wall(Position(row, 0)), f"Left border row {row} should be wall"
            assert maze.is_wall(Position(row, maze.width - 1)), f"Right border row {row} should be wall"

    def test_pacman_spawn_position(self) -> None:
        assert self.maze.pacman_spawn() == Position(10, 5)

    def test_ghost_spawns_count(self) -> None:
        spawns = self.maze.ghost_spawns()
        assert len(spawns) == 2

    def test_ghost_spawn_positions(self) -> None:
        spawns = self.maze.ghost_spawns()
        assert Position(5, 5) in spawns
        assert Position(5, 6) in spawns

    def test_power_pellet_positions(self) -> None:
        assert self.maze.has_power_pellet(Position(3, 1))
        assert self.maze.has_power_pellet(Position(3, 10))

    def test_full_horizontal_corridor_row_1(self) -> None:
        for col in range(1, 11):
            assert self.maze.is_walkable(Position(1, col)), f"Row 1 col {col} should be walkable"

    def test_full_horizontal_corridor_row_7(self) -> None:
        for col in range(1, 11):
            assert self.maze.is_walkable(Position(7, col)), f"Row 7 col {col} should be walkable"

    def test_vertical_corridor_col_3(self) -> None:
        for row in range(1, 11):
            assert self.maze.is_walkable(Position(row, 3)), f"Col 3 row {row} should be walkable"

    def test_vertical_corridor_col_8(self) -> None:
        for row in range(1, 11):
            assert self.maze.is_walkable(Position(row, 8)), f"Col 8 row {row} should be walkable"

    def test_ghost_house_door_is_empty_not_pellet(self) -> None:
        # Cells (4,5) and (4,6) are EMPTY (the ghost-house door corridor)
        assert not self.maze.has_pellet(Position(4, 5))
        assert not self.maze.has_pellet(Position(4, 6))
        assert not self.maze.has_power_pellet(Position(4, 5))
        assert not self.maze.has_power_pellet(Position(4, 6))

    def test_ghost_spawn_cells_are_walkable(self) -> None:
        assert self.maze.is_walkable(Position(5, 5))
        assert self.maze.is_walkable(Position(5, 6))

    def test_ghost_spawn_cells_have_no_pellets(self) -> None:
        assert not self.maze.has_pellet(Position(5, 5))
        assert not self.maze.has_power_pellet(Position(5, 5))

    def test_pacman_spawn_cell_is_walkable(self) -> None:
        assert self.maze.is_walkable(Position(10, 5))

    def test_pacman_spawn_cell_has_no_pellet(self) -> None:
        # Spawn cells are not pellet cells
        assert not self.maze.has_pellet(Position(10, 5))
        assert not self.maze.has_power_pellet(Position(10, 5))


# ---------------------------------------------------------------------------
# Walkability
# ---------------------------------------------------------------------------

class TestWalkability:
    def setup_method(self) -> None:
        self.maze = Maze()

    def test_out_of_bounds_negative_row(self) -> None:
        assert not self.maze.is_walkable(Position(-1, 5))

    def test_out_of_bounds_negative_col(self) -> None:
        assert not self.maze.is_walkable(Position(5, -1))

    def test_out_of_bounds_row_too_large(self) -> None:
        assert not self.maze.is_walkable(Position(12, 0))

    def test_out_of_bounds_col_too_large(self) -> None:
        assert not self.maze.is_walkable(Position(0, 12))

    def test_corner_wall_not_walkable(self) -> None:
        assert not self.maze.is_walkable(Position(0, 0))

    def test_inner_wall_not_walkable(self) -> None:
        # (2,2) is a wall in the default layout
        assert not self.maze.is_walkable(Position(2, 2))

    def test_pellet_cell_walkable(self) -> None:
        assert self.maze.is_walkable(Position(1, 1))

    def test_power_pellet_cell_walkable(self) -> None:
        assert self.maze.is_walkable(Position(3, 1))


# ---------------------------------------------------------------------------
# Pellet consumption
# ---------------------------------------------------------------------------

class TestPelletConsumption:
    def setup_method(self) -> None:
        self.maze = Maze()

    def test_regular_pellet_present_initially(self) -> None:
        assert self.maze.has_pellet(Position(1, 1))

    def test_consume_regular_pellet_returns_pellet_type(self) -> None:
        result = self.maze.consume_pellet(Position(1, 1))
        assert result == CellType.PELLET

    def test_consume_removes_pellet(self) -> None:
        pos = Position(1, 1)
        self.maze.consume_pellet(pos)
        assert not self.maze.has_pellet(pos)

    def test_double_consume_returns_none(self) -> None:
        pos = Position(1, 1)
        self.maze.consume_pellet(pos)
        assert self.maze.consume_pellet(pos) is None

    def test_consume_wall_returns_none(self) -> None:
        assert self.maze.consume_pellet(Position(0, 0)) is None

    def test_consume_power_pellet_returns_power_type(self) -> None:
        result = self.maze.consume_pellet(Position(3, 1))
        assert result == CellType.POWER_PELLET

    def test_consume_power_pellet_removes_it(self) -> None:
        pos = Position(3, 1)
        self.maze.consume_pellet(pos)
        assert not self.maze.has_power_pellet(pos)

    def test_pellets_remaining_decrements(self) -> None:
        initial = self.maze.pellets_remaining()
        self.maze.consume_pellet(Position(1, 1))
        assert self.maze.pellets_remaining() == initial - 1

    def test_pellets_remaining_after_multiple_consumes(self) -> None:
        initial = self.maze.pellets_remaining()
        positions = [Position(1, 1), Position(1, 2), Position(1, 3)]
        for pos in positions:
            self.maze.consume_pellet(pos)
        assert self.maze.pellets_remaining() == initial - len(positions)


# ---------------------------------------------------------------------------
# Maze cleared
# ---------------------------------------------------------------------------

class TestMazeCleared:
    def setup_method(self) -> None:
        self.maze = Maze()

    def test_not_cleared_initially(self) -> None:
        assert not self.maze.is_cleared()

    def test_cleared_after_consuming_all(self) -> None:
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                self.maze.consume_pellet(Position(row, col))
        assert self.maze.is_cleared()

    def test_completion_fraction_zero_initially(self) -> None:
        assert self.maze.completion_fraction == pytest.approx(0.0)

    def test_completion_fraction_increases_on_consume(self) -> None:
        self.maze.consume_pellet(Position(1, 1))
        assert self.maze.completion_fraction > 0.0

    def test_completion_fraction_one_when_cleared(self) -> None:
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                self.maze.consume_pellet(Position(row, col))
        assert self.maze.completion_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def setup_method(self) -> None:
        self.maze = Maze()

    def test_reset_restores_pellet_count(self) -> None:
        initial = self.maze.pellets_remaining()
        for col in range(1, 5):
            self.maze.consume_pellet(Position(1, col))
        assert self.maze.pellets_remaining() < initial
        self.maze.reset()
        assert self.maze.pellets_remaining() == initial

    def test_reset_restores_individual_pellet(self) -> None:
        pos = Position(1, 1)
        self.maze.consume_pellet(pos)
        assert not self.maze.has_pellet(pos)
        self.maze.reset()
        assert self.maze.has_pellet(pos)

    def test_reset_restores_power_pellet(self) -> None:
        pos = Position(3, 1)
        self.maze.consume_pellet(pos)
        assert not self.maze.has_power_pellet(pos)
        self.maze.reset()
        assert self.maze.has_power_pellet(pos)

    def test_total_pellets_unchanged_by_reset(self) -> None:
        total_before = self.maze.total_pellets
        for col in range(1, 11):
            self.maze.consume_pellet(Position(1, col))
        self.maze.reset()
        assert self.maze.total_pellets == total_before

    def test_multiple_resets_are_idempotent(self) -> None:
        initial = self.maze.pellets_remaining()
        self.maze.reset()
        self.maze.reset()
        assert self.maze.pellets_remaining() == initial


# ---------------------------------------------------------------------------
# Observation grids
# ---------------------------------------------------------------------------

class TestObservationGrids:
    def setup_method(self) -> None:
        self.maze = Maze()

    def test_wall_grid_shape_and_dtype(self) -> None:
        grid = self.maze.wall_grid()
        assert grid.shape == (12, 12)
        assert grid.dtype == np.float32

    def test_wall_grid_borders_are_one(self) -> None:
        grid = self.maze.wall_grid()
        assert grid[0, 0] == 1.0
        assert grid[0, 11] == 1.0
        assert grid[11, 0] == 1.0
        assert grid[11, 11] == 1.0

    def test_wall_grid_corridor_is_zero(self) -> None:
        grid = self.maze.wall_grid()
        assert grid[1, 1] == 0.0   # pellet cell, not wall

    def test_pellet_grid_shape_and_dtype(self) -> None:
        grid = self.maze.pellet_grid()
        assert grid.shape == (12, 12)
        assert grid.dtype == np.float32

    def test_pellet_grid_has_ones_at_pellet_cells(self) -> None:
        grid = self.maze.pellet_grid()
        assert grid[1, 1] == 1.0

    def test_pellet_grid_has_zero_at_wall(self) -> None:
        grid = self.maze.pellet_grid()
        assert grid[0, 0] == 0.0

    def test_pellet_grid_has_zero_at_power_pellet(self) -> None:
        grid = self.maze.pellet_grid()
        assert grid[3, 1] == 0.0   # power pellet, not regular pellet

    def test_power_pellet_grid_has_one_at_power_cells(self) -> None:
        grid = self.maze.power_pellet_grid()
        assert grid[3, 1] == 1.0
        assert grid[3, 10] == 1.0

    def test_power_pellet_grid_zero_after_consume(self) -> None:
        self.maze.consume_pellet(Position(3, 1))
        grid = self.maze.power_pellet_grid()
        assert grid[3, 1] == 0.0

    def test_pellet_grid_zero_after_consume(self) -> None:
        self.maze.consume_pellet(Position(1, 1))
        grid = self.maze.pellet_grid()
        assert grid[1, 1] == 0.0

    def test_grids_are_independent_copies(self) -> None:
        grid1 = self.maze.pellet_grid()
        grid1[1, 1] = 0.0
        # Modifying the returned grid should not affect the maze state
        assert self.maze.has_pellet(Position(1, 1))


# ---------------------------------------------------------------------------
# Custom layout
# ---------------------------------------------------------------------------

class TestCustomLayout:
    def test_3x3_minimal_layout(self) -> None:
        layout = _minimal_layout()
        maze = Maze(layout=layout)
        assert maze.height == 3
        assert maze.width == 3

    def test_custom_pacman_spawn(self) -> None:
        layout = _minimal_layout()
        maze = Maze(layout=layout)
        assert maze.pacman_spawn() == Position(1, 1)

    def test_custom_ghost_spawn(self) -> None:
        layout = _minimal_layout()
        maze = Maze(layout=layout)
        spawns = maze.ghost_spawns()
        assert Position(1, 2) in spawns

    def test_custom_power_pellet(self) -> None:
        layout = _minimal_layout()
        maze = Maze(layout=layout)
        assert maze.has_power_pellet(Position(2, 1))

    def test_custom_layout_walls_correct(self) -> None:
        layout = _minimal_layout()
        maze = Maze(layout=layout)
        assert maze.is_wall(Position(0, 0))
        assert not maze.is_wall(Position(1, 1))
