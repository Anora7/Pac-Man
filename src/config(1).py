"""
Global configuration and constants for the Pac-Man RL project.

This module is the single source of truth for all project-wide constants.
No other module should define reward values, grid sizes, colors,
hyperparameters, or action mappings.

This module must not import from any other project module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

class CellType(IntEnum):
    """Integer codes used in the maze layout grid.

    Each grid cell is assigned exactly one CellType. Spawn cells are walkable
    but carry no pellet during gameplay.
    """

    WALL = 0
    PELLET = 1
    POWER_PELLET = 2
    EMPTY = 3          # Walkable, no pellet (e.g. ghost-house corridor)
    PACMAN_SPAWN = 4   # Pac-Man starting position
    GHOST_SPAWN = 5    # Ghost starting positions


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Discrete action space for the Pac-Man agent."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Row-delta, col-delta for each action.
ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

MAZE_HEIGHT: int = 12
MAZE_WIDTH: int = 12


# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------
# Cell codes correspond to CellType enum values.
#
# Legend:
#   0 = WALL
#   1 = PELLET
#   2 = POWER_PELLET
#   3 = EMPTY (walkable, no pellet — ghost-house door corridor)
#   4 = PACMAN_SPAWN
#   5 = GHOST_SPAWN
#
# Connectivity notes:
#   - Full horizontal corridors at rows 1, 3, 7, 10.
#   - Full vertical corridors at cols 3 and 8 (rows 1–10).
#   - Ghost house (rows 4–5, cols 5–6) opens upward into row 3.
#   - Top and bottom halves connect only through col 3 and col 8 at row 6.

MAZE_LAYOUT: list[list[int]] = [
    #   0  1  2  3  4  5  6  7  8  9 10 11
    [   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row  0  all wall
    [   0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # row  1  full corridor
    [   0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # row  2  vertical links
    [   0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0],  # row  3  full corridor + power pellets
    [   0, 1, 0, 1, 0, 3, 3, 0, 1, 0, 1, 0],  # row  4  ghost-house door
    [   0, 1, 1, 1, 0, 5, 5, 0, 1, 1, 1, 0],  # row  5  ghost spawns
    [   0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # row  6  bridge cols 3 & 8 only
    [   0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # row  7  full corridor
    [   0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # row  8  vertical links
    [   0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # row  9  vertical links
    [   0, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 0],  # row 10  full corridor + Pac-Man spawn
    [   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 11  all wall
]


# ---------------------------------------------------------------------------
# Reward values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    """Centralised reward values. Modify here to tune reward shaping.

    All values are floats to match Gymnasium's Box space convention.
    """

    PELLET: float = 10.0
    POWER_PELLET: float = 50.0
    EAT_GHOST: float = 100.0
    CLEAR_MAZE: float = 500.0
    CAUGHT_BY_GHOST: float = -200.0
    HIT_WALL: float = -5.0
    STEP: float = -1.0


REWARDS: RewardConfig = RewardConfig()


# ---------------------------------------------------------------------------
# Power-mode timing
# ---------------------------------------------------------------------------

POWER_MODE_DURATION: int = 30  # environment steps while power mode is active


# ---------------------------------------------------------------------------
# Ghost behaviour
# ---------------------------------------------------------------------------

GHOST_CHASE_PROBABILITY: float = 0.30   # probability of chasing over random move
GHOST_CHASE_RADIUS: int = 5             # Manhattan-distance threshold to trigger chase


# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------

MAX_STEPS: int = 1000  # maximum environment steps before truncation


# ---------------------------------------------------------------------------
# PPO hyperparameter defaults
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS: int = 500_000
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
N_STEPS: int = 2048
BATCH_SIZE: int = 64
N_EPOCHS: int = 10
ENT_COEF: float = 0.01
CLIP_RANGE: float = 0.2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CELL_SIZE: int = 40   # pixels per grid cell
FPS: int = 10         # frames per second during visual playback

# RGB colour constants
COLOR_BACKGROUND: tuple[int, int, int] = (0, 0, 0)
COLOR_WALL: tuple[int, int, int] = (33, 33, 222)
COLOR_PELLET: tuple[int, int, int] = (255, 255, 255)
COLOR_POWER_PELLET: tuple[int, int, int] = (255, 184, 82)
COLOR_PACMAN: tuple[int, int, int] = (255, 255, 0)
COLOR_GHOST: tuple[int, int, int] = (255, 0, 0)
COLOR_GHOST_VULNERABLE: tuple[int, int, int] = (0, 0, 255)
COLOR_HUD: tuple[int, int, int] = (255, 255, 255)
