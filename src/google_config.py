"""
Configuration and constants for the Google Pac-Man RL environment.

This module imports shared definitions from config.py and defines
custom settings for the 16x16 Google Pac-Man board, rule-based ghosts,
rewards, power pellet mechanics, and PPO curriculum training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path

from config import (
    ACTION_DELTAS,
    Action,
    CellType,
    ObservationConfig,
    RewardConfig,
    DEFAULT_SEED,
    TRAINING_DEVICE,
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    GAMMA,
    LEARNING_RATE,
    N_EPOCHS,
    N_STEPS,
)

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

OPPOSITE_ACTIONS: dict[Action, Action] = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}

# ---------------------------------------------------------------------------
# Board dimensions & episode settings
# ---------------------------------------------------------------------------

GOOGLE_MAZE_HEIGHT: int = 16
GOOGLE_MAZE_WIDTH: int = 16
GOOGLE_MAX_STEPS: int = 1500

# ---------------------------------------------------------------------------
# Power-mode & Ghost settings
# ---------------------------------------------------------------------------

GOOGLE_POWER_MODE_DURATION: int = 30
GOOGLE_EAT_GHOST_REWARD: float = 100.0

GOOGLE_SCATTER_STEPS: int = 10
GOOGLE_CHASE_STEPS: int = 20
GOOGLE_GHOST_RELEASE_INTERVAL: int = 30
GOOGLE_GHOST_RELEASE_SCHEDULE: tuple[int, ...] = (0, 30, 60, 90)

GOOGLE_CLYDE_CHASE_DISTANCE: int = 6
GOOGLE_PINKY_LOOKAHEAD: int = 4
GOOGLE_INKY_LOOKAHEAD: int = 2

# Ghost identity enumeration
class GhostIdentity(Enum):
    BLINKY = auto()  # Red - Chaser
    PINKY = auto()   # Pink - Ambusher
    INKY = auto()    # Cyan - Vector mirror
    CLYDE = auto()   # Orange - Shy chaser / wanderer

# Scatter corners for each ghost: (row, col)
GHOST_SCATTER_CORNERS: dict[GhostIdentity, tuple[int, int]] = {
    GhostIdentity.BLINKY: (1, 14),   # Top-Right
    GhostIdentity.PINKY: (1, 1),     # Top-Left
    GhostIdentity.INKY: (14, 14),    # Bottom-Right
    GhostIdentity.CLYDE: (14, 1),    # Bottom-Left
}

# ---------------------------------------------------------------------------
# 16x16 Maze layout
# ---------------------------------------------------------------------------
# Symmetrical layout:
# - Corridors with multiple loops and junctions (0 dead-end traps)
# - Central ghost house with door at (6, 7)-(6, 8)
# - 4 Ghost spawns at (7, 6), (7, 7), (7, 8), (7, 9)
# - Pac-Man spawn at (12, 7)
# - 4 Power pellets at corners: (1, 1), (1, 14), (14, 1), (14, 14)

GOOGLE_MAZE_LAYOUT: list[list[int]] = [
    # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # row  0: top wall
    [ 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0], # row  1: top corridor & 2 power pellets
    [ 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], # row  2: wall blocks
    [ 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], # row  3: wall blocks
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # row  4: full cross-corridor
    [ 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0], # row  5: upper ghost house entry
    [ 0, 1, 1, 1, 1, 0, 0, 3, 3, 0, 0, 1, 1, 1, 1, 0], # row  6: ghost house door (cols 7, 8)
    [ 0, 0, 0, 1, 1, 0, 5, 5, 5, 5, 0, 1, 1, 0, 0, 0], # row  7: ghost spawns (cols 6-9)
    [ 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], # row  8: ghost house base wall
    [ 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0], # row  9: mid corridor
    [ 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0], # row 10: lower corridor
    [ 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], # row 11: wall blocks
    [ 0, 1, 1, 0, 1, 1, 1, 4, 1, 1, 1, 1, 0, 1, 1, 0], # row 12: Pac-Man spawn (col 7)
    [ 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], # row 13: lower junctions
    [ 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0], # row 14: bottom corridor & 2 power pellets
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # row 15: bottom wall
]

# ---------------------------------------------------------------------------
# Rewards configuration
# ---------------------------------------------------------------------------

GOOGLE_REWARDS: RewardConfig = RewardConfig(
    PELLET=10.0,
    POWER_PELLET=50.0,
    EAT_GHOST=GOOGLE_EAT_GHOST_REWARD,
    CLEAR_MAZE=500.0,
    CAUGHT_BY_GHOST=-200.0,
    HIT_WALL=-5.0,
    STEP=-1.0,
    PELLET_DISTANCE_DECREASE=0.5,
    PELLET_DISTANCE_INCREASE=-0.5,
    REVISIT_CELL=-0.1,
    GHOST_DISTANCE_DECREASE=0.0,
)

# ---------------------------------------------------------------------------
# Visual & Color Constants
# ---------------------------------------------------------------------------

GOOGLE_CELL_SIZE: int = 36
GOOGLE_FPS: int = 6

GOOGLE_COLOR_BACKGROUND: tuple[int, int, int] = (10, 10, 15)
GOOGLE_COLOR_WALL: tuple[int, int, int] = (35, 45, 185)
GOOGLE_COLOR_PELLET: tuple[int, int, int] = (245, 245, 245)
GOOGLE_COLOR_POWER_PELLET: tuple[int, int, int] = (255, 195, 75)
GOOGLE_COLOR_PACMAN: tuple[int, int, int] = (255, 238, 0)

GOOGLE_COLOR_BLINKY: tuple[int, int, int] = (235, 45, 45)      # Red
GOOGLE_COLOR_PINKY: tuple[int, int, int] = (255, 160, 200)     # Pink
GOOGLE_COLOR_INKY: tuple[int, int, int] = (40, 220, 230)       # Cyan
GOOGLE_COLOR_CLYDE: tuple[int, int, int] = (255, 165, 40)      # Orange

GOOGLE_COLOR_FRIGHTENED: tuple[int, int, int] = (230, 235, 245)  # Light grey / white
GOOGLE_COLOR_FRIGHTENED_FACE: tuple[int, int, int] = (240, 60, 80) # Red / pink face
GOOGLE_COLOR_HUD: tuple[int, int, int] = (255, 255, 255)

# Map identity to regular color
GHOST_COLORS: dict[GhostIdentity, tuple[int, int, int]] = {
    GhostIdentity.BLINKY: GOOGLE_COLOR_BLINKY,
    GhostIdentity.PINKY: GOOGLE_COLOR_PINKY,
    GhostIdentity.INKY: GOOGLE_COLOR_INKY,
    GhostIdentity.CLYDE: GOOGLE_COLOR_CLYDE,
}

# ---------------------------------------------------------------------------
# Curriculum & Training Defaults
# ---------------------------------------------------------------------------

GOOGLE_MODELS_DIR: Path = Path("models/google")
GOOGLE_RESULTS_DIR: Path = Path("results/google")
GOOGLE_LOGS_DIR: Path = Path("logs/google")

@dataclass(frozen=True)
class GoogleCurriculumStage:
    """A stage in the training curriculum."""
    stage_id: int
    active_ghosts: int
    chase_strength: float
    max_steps: int
    success_threshold: float = 0.70
    success_window: int = 20

GOOGLE_CURRICULUM_STAGES: tuple[GoogleCurriculumStage, ...] = (
    GoogleCurriculumStage(stage_id=1, active_ghosts=2, chase_strength=0.50, max_steps=150_000),
    GoogleCurriculumStage(stage_id=2, active_ghosts=3, chase_strength=0.75, max_steps=100_000),
    GoogleCurriculumStage(stage_id=3, active_ghosts=4, chase_strength=1.00, max_steps=50_000),
)

GOOGLE_STAGE_BUDGET_RATIOS: tuple[float, float, float] = (0.50, 0.33, 0.17)


def calculate_stage_budgets(total_steps: int) -> tuple[int, int, int]:
    """Split total training budget into 50% Stage 1, 33% Stage 2, and 17% Stage 3."""
    stage1 = int(total_steps * GOOGLE_STAGE_BUDGET_RATIOS[0])
    stage2 = int(total_steps * GOOGLE_STAGE_BUDGET_RATIOS[1])
    stage3 = total_steps - stage1 - stage2
    return (stage1, stage2, stage3)

