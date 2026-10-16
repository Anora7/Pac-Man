# API Reference

All source modules live under `src/`. Import them directly (add `src/` to `sys.path` or run from the project root with `conftest.py` present).

---

## config.py

Central constants. No imports from other project modules.

| Symbol | Type | Description |
|---|---|---|
| `CellType` | `IntEnum` | Grid cell codes: `WALL=0`, `PELLET=1`, `POWER_PELLET=2`, `EMPTY=3`, `PACMAN_SPAWN=4`, `GHOST_SPAWN=5` |
| `Action` | `IntEnum` | `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3` |
| `ACTION_DELTAS` | `dict[Action, tuple[int,int]]` | Row/col delta for each action |
| `MAZE_HEIGHT`, `MAZE_WIDTH` | `int` | Grid dimensions (12×12) |
| `MAZE_LAYOUT` | `list[list[int]]` | Default 12×12 maze |
| `RewardConfig` | frozen dataclass | Reward values (`PELLET`, `POWER_PELLET`, `EAT_GHOST`, `CLEAR_MAZE`, `CAUGHT_BY_GHOST`, `HIT_WALL`, `STEP`) |
| `ObservationConfig` | frozen dataclass | Optional observation features; both default off |
| `GhostRewardConfig` | frozen dataclass | Reward values for ghost-side training |
| `MultiSeedConfig` | frozen dataclass | Reproducible seed list and confidence level |
| `DQNConfig` | frozen dataclass | CPU DQN replay, exploration, and target-network settings |
| `AblationConfig` | frozen dataclass | Reward study seeds, training bounds, and output paths |
| `CurriculumConfig` | frozen dataclass | Success-gated ghost-count promotion and stage bounds |
| `REWARDS` | `RewardConfig` | Singleton reward config instance |
| `MAX_STEPS` | `int` | Default episode step limit (1000) |
| `POWER_MODE_DURATION` | `int` | Power mode duration in steps (30) |
| `GHOST_CHASE_PROBABILITY` | `float` | Probability of chase move (0.30) |
| `GHOST_CHASE_RADIUS` | `int` | Manhattan distance threshold (5) |
| PPO constants | `float/int` | `LEARNING_RATE`, `GAMMA`, `N_STEPS`, `BATCH_SIZE`, `N_EPOCHS`, `ENT_COEF`, `CLIP_RANGE`, `TOTAL_TIMESTEPS` |

---

## maze.py

### `Position(row, col)`

Immutable `NamedTuple`. Supports `pos + delta` arithmetic.

### `Maze(layout=None)`

| Method | Returns | Description |
|---|---|---|
| `reset()` | `None` | Restore all pellets for a new episode |
| `is_wall(pos)` | `bool` | True if cell is a wall |
| `is_walkable(pos)` | `bool` | True if in-bounds and not a wall |
| `has_pellet(pos)` | `bool` | Regular pellet present at pos |
| `has_power_pellet(pos)` | `bool` | Power pellet present at pos |
| `consume_pellet(pos)` | `CellType \| None` | Eat pellet; returns type consumed or None |
| `pellets_remaining()` | `int` | Count of unconsumed pellets |
| `is_cleared()` | `bool` | True when all pellets consumed |
| `pacman_spawn()` | `Position` | Pac-Man's starting cell |
| `ghost_spawns()` | `list[Position]` | All ghost starting cells |
| `wall_grid()` | `np.ndarray (H,W) float32` | 1.0 at wall cells |
| `pellet_grid()` | `np.ndarray (H,W) float32` | 1.0 at remaining regular pellets |
| `power_pellet_grid()` | `np.ndarray (H,W) float32` | 1.0 at remaining power pellets |
| `total_pellets` | `int` property | Total pellets at episode start |
| `completion_fraction` | `float` property | Fraction of pellets consumed |

---

## ghost.py

### `GhostState`

`Enum`: `NORMAL`, `VULNERABLE`

### `Ghost(spawn, chase_probability=0.30, chase_radius=5)`

| Method | Description |
|---|---|
| `reset()` | Return to spawn, restore NORMAL state |
| `respawn()` | Teleport to spawn (called when eaten by Pac-Man) |
| `make_vulnerable()` | Switch to VULNERABLE (power pellet eaten) |
| `restore_normal()` | Return to NORMAL (power mode expired) |
| `move(maze, pacman_pos, rng)` | Move one step per current policy |
| `is_vulnerable` | `bool` property |
| `is_normal` | `bool` property |

## ghost_policies.py

`GhostPolicy` defines an interchangeable action-selection interface. Built-in
policies are `RandomGhostPolicy`, `ChaseGhostPolicy`, `RuleBasedGhostPolicy`,
and `LearnedGhostPolicy`. A learned policy requires an explicit ghost-model
observation; blocked actions are handled safely by `Ghost.move()`.

---

## ghost_policies.py

`GhostPolicy` defines an interchangeable action-selection interface. Built-in
policies are `RandomGhostPolicy`, `ChaseGhostPolicy`, `RuleBasedGhostPolicy`,
and `LearnedGhostPolicy`. A learned policy requires an explicit ghost-model
observation; blocked actions are handled safely by `Ghost.move()`.

## rewards.py

### `calculate_step_reward(**events) -> float`

Keyword-only arguments: `hit_wall`, `pellet_eaten`, `power_pellet_eaten`, `ghosts_eaten`, `caught`, `cleared`. Always includes `REWARDS.STEP`. Returns the total step reward.

## ghost_environment.py

`GhostEnvironment` is an optional Gymnasium environment for ghost-side
training. Shared mode cycles the same PPO policy across per-ghost-centred
observations; single-ghost mode leaves the other ghosts on their configured
policies.

## train_alternating.py

`run_alternating(AlternatingConfig(...))` alternates ghost PPO against a
frozen Pac-Man policy and Pac-Man PPO against a frozen learned ghost. Each
seed and round is saved under `models/experiments/alternating/`; Monitor logs
and round metrics are stored under the matching experiment directories.

## evaluate_matrix.py

`run_evaluation_matrix(RobustnessConfig(...))` compares the baseline and
available extension models against random, chase, rule-based, personality,
and (when available) learned ghosts. It writes `results/robustness_matrix.csv`
and annotated heatmaps under `plots/experiments/`.

## observation_metadata.py

`save_observation_config()` and `load_observation_config()` persist
observation flags next to model archives. Missing metadata means baseline
settings; malformed metadata raises an error.

## experiment_utils.py

`run_across_seeds()` executes one callback per configured seed.
`summarize_metric()` and `aggregate_metrics()` report mean, sample standard
deviation, min/max, and a confidence interval. `plot_monitor_curves()` aligns
seed Monitor series by episode number and plots the mean with a shaded
standard-deviation band.

## analysis_plots.py

Provides Pac-Man/ghost visitation and catch heatmaps, PPO value-function
danger maps, learning curves, alternating co-evolution dashboard,
per-episode outcome bars, and optional PNG frame capture. The smoothed
training chart reads the `is_success` field from Monitor logs; use current
training scripts to generate this field.

---

## environment.py

### `PacmanEnvironment(render_mode=None, max_steps=1000, maze_layout=None, observation_config=...)`

Inherits `gymnasium.Env`. Implements `reset()`, `step()`, `render()`, `close()`.

- **Observation space:** baseline `Box(0.0, 1.0, shape=(H×W×5+3,), float32)` — optional feature lengths are described in `docs/environment.md`.
- **Action space:** `Discrete(4)`.

---

## renderer.py

### `RenderState`

Frozen dataclass: `maze`, `pacman_position`, `ghost_positions`, `ghost_vulnerable`, `score`, `step`, `power_mode_active`, `power_mode_steps_remaining`, `episode`.

### `PacmanRenderer(cell_size=40, fps=8, render_mode="human")`

| Method | Description |
|---|---|
| `render(state)` | Draw one frame (lazy pygame init on first call) |
| `close()` | Shut down pygame (safe before first render) |
| `window_size` | `(width, height)` property |

---

## train.py

### `TrainingConfig`

Dataclass with all hyperparameters and output paths. Defaults read from `config.py`.

### `train(config=None) -> PPO`

Creates environments, builds model, runs `model.learn()`, saves final model, returns model.

## train_dqn.py

`train_dqn(DQNConfig(...))` trains a CPU-only DQN model using the same
Gymnasium environment and Monitor success logging. Replay-buffer capacity,
learning starts, exploration schedule, and target-network update interval are
configurable in `DQNConfig`.

## compare_algorithms.py

`run_comparison(AlgorithmComparisonConfig(...))` trains PPO and DQN using
matched seeds, timesteps, and evaluation episodes. It writes per-seed and
aggregate confidence-interval metrics, shaded learning curves, error-bar
comparisons, and a report tied to Decision 001.

## ablation.py

`run_ablation(AblationConfig(...))` trains the ten centrally configured
reward variants with matched seeds and reward-aware evaluation. It writes
per-seed/aggregate CSV metrics, a scale-independent outcome summary, and a
grouped comparison chart.

## train_curriculum.py

`run_curriculum(CurriculumConfig(...))` trains Pac-Man first against one
ghost, then promotes to the configured full ghost count when rolling
success reaches its threshold or the stage step limit is reached. Promotion
events are written to CSV and marked on a reward-versus-timestep plot.

## run_all_experiments.py

From the repository root, `python run_all_experiments.py --quick` runs every
extension with short CPU bounds. `--full` uses research defaults. Repeat
`--only <name>` for selected workflows (`alternating`, `curriculum`, `dqn`,
`ablation`, `robustness`, `plots`).

---

## evaluate.py

### `EvaluationConfig`

Dataclass: `model_path`, `n_episodes`, `render`, `deterministic`, `seed`, `max_steps`, `verbose`.

### `EpisodeResult`

Frozen dataclass: per-episode metrics.

### `EvaluationResult`

Frozen dataclass: aggregated metrics.

### `evaluate(config=None, model=None) -> EvaluationResult`

Runs N episodes. Accepts pre-loaded model or loads from `config.model_path`.
Supports ghost type selection, observation metadata, Pygame FPS, and optional
per-episode CSV export through `EvaluationConfig`.

---

## plots.py

### `PlotConfig`

Dataclass: `plots_dir`, `window`, `figsize`, `dpi`, `style`.

| Function | Saves | Data source |
|---|---|---|
| `plot_reward_curve(log_path, config)` | `reward_curve.png` | Monitor CSV |
| `plot_moving_average(log_path, config)` | `moving_average.png` | Monitor CSV |
| `plot_episode_length(log_path, config)` | `episode_length.png` | Monitor CSV |
| `plot_success_rate(result, config)` | `success_rate.png` | `EvaluationResult` |
| `generate_all_plots(log_path, result, config)` | all above | both |

All functions return the `Path` of the saved PNG.
