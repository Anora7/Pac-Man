"""
Global configuration and constants for the Pac-Man RL project.

This module is the single source of truth for all project-wide constants.
No other module should define reward values, grid sizes, colors,
hyperparameters, or action mappings.

This module must not import from any other project module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path


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

REWARD_ABLATION_VARIANTS: tuple[tuple[str, RewardConfig], ...] = (
    ("baseline", REWARDS),
    ("caught_penalty_50", replace(REWARDS, CAUGHT_BY_GHOST=-50.0)),
    ("caught_penalty_500", replace(REWARDS, CAUGHT_BY_GHOST=-500.0)),
    ("step_penalty_0", replace(REWARDS, STEP=0.0)),
    ("step_penalty_2", replace(REWARDS, STEP=-2.0)),
    ("no_wall_penalty", replace(REWARDS, HIT_WALL=0.0)),
    ("pellet_reward_5", replace(REWARDS, PELLET=5.0)),
    ("pellet_reward_20", replace(REWARDS, PELLET=20.0)),
    ("clear_bonus_0", replace(REWARDS, CLEAR_MAZE=0.0)),
    ("clear_bonus_1000", replace(REWARDS, CLEAR_MAZE=1000.0)),
)


# ---------------------------------------------------------------------------
# Extension defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationConfig:
    """Optional observation features; defaults preserve the 723-value baseline."""

    include_ghost_direction: bool = False
    include_previous_pacman_action: bool = False


OBSERVATION_CONFIG: ObservationConfig = ObservationConfig()
GHOST_DIRECTION_FEATURES: int = 4
GHOST_OBSERVATION_FEATURES: int = GHOST_DIRECTION_FEATURES + 1
PACMAN_ACTION_FEATURES: int = 4


@dataclass(frozen=True)
class GhostRewardConfig:
    """Reward values for optional ghost-side training."""

    CATCH_PACMAN: float = 100.0
    STEP: float = -0.1
    PACMAN_EATS_PELLET: float = -1.0
    PACMAN_EATS_POWER_PELLET: float = -2.0
    GHOST_EATEN: float = -25.0
    REDUCE_DISTANCE: float = 0.0
    PACMAN_CLEARS_MAZE: float = -50.0


GHOST_REWARDS: GhostRewardConfig = GhostRewardConfig()


@dataclass(frozen=True)
class GhostPersonalityConfig:
    """Movement settings for one optional fixed ghost personality."""

    name: str
    chase_probability: float
    chase_radius: int


GHOST_PERSONALITIES: tuple[GhostPersonalityConfig, ...] = (
    GhostPersonalityConfig("aggressive", 0.9, 7),
    GhostPersonalityConfig("cautious", 0.1, 3),
)


# These defaults are shared by experiment entry points for reproducibility.
DEFAULT_SEED: int = 42
TRAINING_DEVICE: str = "cpu"
GHOST_COUNT: int = 2

# Short CPU-only values used by extension CLI --quick modes.
QUICK_TIMESTEPS: int = 32
QUICK_EPISODES: int = 1
QUICK_SEEDS: int = 1

# Medium CPU-only experiment budgets.
MEDIUM_SEEDS: int = 2
MEDIUM_TIMESTEPS: int = 100_000
MEDIUM_ALTERNATING_ROUNDS: int = 3
MEDIUM_ALTERNATING_STEPS_PER_ROUND: int = 30_000
MEDIUM_GHOST_STEPS_PER_ROUND: int = 50_000
MEDIUM_CPU_STEPS_PER_SECOND: int = 120


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


# PPO hyperparameter defaults used by the experiment dataclasses below.
TOTAL_TIMESTEPS: int = 500_000
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
N_STEPS: int = 2048
BATCH_SIZE: int = 64
N_EPOCHS: int = 10
ENT_COEF: float = 0.01
CLIP_RANGE: float = 0.2


@dataclass(frozen=True)
class AlternatingConfig:
    """Defaults for alternating Pac-Man/ghost PPO experiments."""

    n_rounds: int = 3
    pacman_steps_per_round: int = 50_000
    ghost_steps_per_round: int = 50_000
    seeds: tuple[int, ...] = (42, 43, 44)
    initial_pacman_model_path: Path | None = Path("models/best_model.zip")
    restart_pacman_each_round: bool = False
    models_dir: Path = Path("models/experiments/alternating")
    logs_dir: Path = Path("logs/experiments/alternating")
    results_path: Path = Path("results/alternating_rounds.csv")
    opponent_pool_latest_probability: float = 0.7
    evaluation_episodes: int = 5
    max_steps: int = MAX_STEPS
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 5
    eval_freq: int = 10_000

    def __post_init__(self) -> None:
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be at least 1.")
        if self.pacman_steps_per_round < 1 or self.ghost_steps_per_round < 1:
            raise ValueError("Training timesteps must be positive.")
        if not self.seeds:
            raise ValueError("At least one training seed is required.")
        if not 0.0 <= self.opponent_pool_latest_probability <= 1.0:
            raise ValueError(
                "opponent_pool_latest_probability must be between 0 and 1."
            )
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be at least 1.")


@dataclass(frozen=True)
class RobustnessConfig:
    """Defaults for model-by-ghost-type robustness evaluations."""

    models_dir: Path = Path("models")
    experiment_models_dir: Path | None = None
    additional_experiment_models_dirs: tuple[Path, ...] = ()
    episodes: int = 10
    max_steps: int = MAX_STEPS
    seed: int = DEFAULT_SEED
    results_path: Path = Path("results/robustness_matrix.csv")
    plot_path: Path = Path("plots/experiments/robustness_matrix.png")


@dataclass(frozen=True)
class MultiSeedConfig:
    """Shared reproducibility and aggregation settings for experiments."""

    seeds: tuple[int, ...] = (42, 43, 44)
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one experiment seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Experiment seeds must be unique.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")


@dataclass(frozen=True)
class DQNConfig:
    """CPU DQN settings for a controlled comparison with the PPO baseline."""

    total_timesteps: int = 500_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 50_000  # Replay transitions retained for off-policy updates.
    learning_starts: int = 1_000  # Warm-up steps before optimizing from replay.
    batch_size: int = 32  # Replay samples per gradient update.
    train_freq: int = 4  # Collect this many steps between training calls.
    gradient_steps: int = 1  # Gradient updates per training call.
    target_update_interval: int = 1_000  # Steps between target-network copies.
    exploration_fraction: float = 0.1  # Fraction of training used to decay epsilon.
    exploration_final_eps: float = 0.05  # Minimum epsilon after exploration decay.
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 10_000
    max_steps: int = MAX_STEPS
    seed: int = DEFAULT_SEED
    models_dir: Path = Path("models/experiments/dqn")
    logs_dir: Path = Path("logs/experiments/dqn")
    tensorboard_dir: Path = Path("logs/experiments/dqn/tensorboard")

    def __post_init__(self) -> None:
        if self.total_timesteps < 1 or self.buffer_size < 1:
            raise ValueError("DQN timesteps and replay buffer size must be positive.")
        if self.learning_starts < 0 or self.batch_size < 1:
            raise ValueError("DQN learning_starts must be non-negative and batch size positive.")
        if self.train_freq < 1 or self.gradient_steps < 1:
            raise ValueError("DQN train frequency and gradient steps must be positive.")
        if self.target_update_interval < 1 or self.max_steps < 1:
            raise ValueError("DQN target interval and max_steps must be positive.")


@dataclass(frozen=True)
class AblationConfig:
    """Shared PPO training bounds and output locations for reward ablations."""

    seeds: tuple[int, ...] = (42, 43, 44)
    variant_names: tuple[str, ...] = tuple(
        name for name, _ in REWARD_ABLATION_VARIANTS
    )
    total_timesteps: int = TOTAL_TIMESTEPS
    evaluation_episodes: int = 10
    max_steps: int = MAX_STEPS
    eval_freq: int = 10_000
    n_steps: int = N_STEPS
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    models_dir: Path = Path("models/experiments/ablation")
    logs_dir: Path = Path("logs/experiments/ablation")
    results_path: Path = Path("results/ablation_results.csv")
    report_path: Path = Path("results/ablation_summary.md")
    plots_dir: Path = Path("plots/experiments/ablation")

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Ablation seeds must be non-empty and unique.")
        available_variants = {
            name for name, _ in REWARD_ABLATION_VARIANTS
        }
        if (
            not self.variant_names
            or len(set(self.variant_names)) != len(self.variant_names)
            or "baseline" not in self.variant_names
            or not set(self.variant_names).issubset(available_variants)
        ):
            raise ValueError(
                "Ablation variant_names must be unique, include baseline, "
                "and contain only configured variants."
            )
        if self.total_timesteps < 1 or self.evaluation_episodes < 1:
            raise ValueError("Ablation training and evaluation counts must be positive.")
        if self.max_steps < 1 or self.eval_freq < 1:
            raise ValueError("Ablation step limits and eval frequency must be positive.")


@dataclass(frozen=True)
class CurriculumConfig:
    """PPO training schedule for staged promotion from fewer to more ghosts."""

    seeds: tuple[int, ...] = (42, 43, 44)
    max_steps_per_stage: int = 100_000
    promotion_success_threshold: float = 0.75
    success_window: int = 20
    evaluation_episodes: int = 5
    max_steps: int = MAX_STEPS
    eval_freq: int = 10_000
    n_steps: int = N_STEPS
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    models_dir: Path = Path("models/experiments/curriculum")
    logs_dir: Path = Path("logs/experiments/curriculum")
    results_path: Path = Path("results/curriculum_promotions.csv")
    plot_path: Path = Path("plots/experiments/curriculum/curriculum_progress.png")

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Curriculum seeds must be non-empty and unique.")
        if self.max_steps_per_stage < 1 or self.success_window < 1:
            raise ValueError("Curriculum stage limits and success window must be positive.")
        if not 0.0 <= self.promotion_success_threshold <= 1.0:
            raise ValueError("promotion_success_threshold must be between 0 and 1.")
        if self.evaluation_episodes < 1 or self.max_steps < 1 or self.eval_freq < 1:
            raise ValueError("Curriculum evaluation counts and step bounds must be positive.")


@dataclass(frozen=True)
class AnalysisPlotConfig:
    """Settings for diagnostic and experiment analysis figures."""

    plots_dir: Path = Path("plots/experiments")
    evaluation_episodes: int = 5
    moving_average_window: int = 50
    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: int = 150
    style: str = "seaborn-v0_8-darkgrid"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CELL_SIZE: int = 40   # pixels per grid cell
FPS: int = 8          # frames per second during visual playback

# RGB colour constants
COLOR_BACKGROUND: tuple[int, int, int] = (0, 0, 0)
COLOR_WALL: tuple[int, int, int] = (33, 33, 222)
COLOR_PELLET: tuple[int, int, int] = (255, 255, 255)
COLOR_POWER_PELLET: tuple[int, int, int] = (255, 184, 82)
COLOR_PACMAN: tuple[int, int, int] = (255, 255, 0)
COLOR_GHOST: tuple[int, int, int] = (255, 0, 0)
COLOR_GHOST_VULNERABLE: tuple[int, int, int] = (0, 0, 255)
COLOR_HUD: tuple[int, int, int] = (255, 255, 255)
