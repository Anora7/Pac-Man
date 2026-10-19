# System Architecture

## Purpose

This document defines the software architecture of the Reinforcement Learning Pac-Man project.

Its purpose is to describe ownership and responsibilities of each module.

It intentionally avoids implementation-specific details.

---

# High-Level Architecture

```text
             User
               │
               ▼
      Training Script
               │
               ▼
 Stable-Baselines3 PPO
               │
               ▼
 Custom Gymnasium Environment
               │
      ┌────────┴────────┐
      ▼                 ▼
 Maze Logic        Ghost Logic
      │                 │
      └────────┬────────┘
               ▼
        Reward Function
               │
               ▼
      Environment State
               │
      ┌────────┴────────┐
      ▼                 ▼
 Renderer         Evaluation
```
---

# Module Responsibilities

## environment.py

### Owns

- Gymnasium interface
- reset()
- step()
- close()
- Episode lifecycle
- State transitions

### Must NOT

- Render graphics
- Train models
- Save models
- Plot graphs

---

## maze.py

### Owns

- Maze representation
- Walls
- Empty paths
- Pellet locations
- Power pellet
- Spawn locations
- Shortest-path queries through walkable cells

### Must NOT

- RL logic
- Rendering
- PPO

---

## ghost.py

### Owns

- Ghost movement
- Ghost decision rules
- Ghost state
- Vulnerable state
- Ghost respawn
- Dispatching movement through a ghost-policy interface

### Must NOT

- Train models or contain PPO logic
- Modify reward function

---

## rewards.py

### Owns

- Reward calculations
- Reward configuration
- Reward shaping

## ghost_policies.py

Owns interchangeable ghost action-selection policies. The baseline
rule-based policy remains the default. Learned-policy inference is isolated
here and receives an explicit observation from its caller; the ghost entity
and game environment do not train models. Learned evaluation cycles PPO
control over one ghost per step and selects actions from the pre-Pac-Man-move
state, matching shared ghost training; illegal model-directed moves use a
random legal-neighbor fallback.

## ghost_observations.py

Owns ghost-centred feature encoding shared by the training wrapper and
learned policy callers. It deliberately has no Stable-Baselines3 dependency.

## ghost_environment.py

Owns the optional ghost-side Gymnasium interface. It reuses
`PacmanEnvironment` for transitions and reward-event detection while keeping
the frozen Pac-Man policy and ghost reward calculation outside the baseline
environment.

## train_alternating.py

Owns PPO model construction, callbacks, opponent pools, and round evaluation.
It uses the dedicated ghost Gymnasium environment and standard
`PacmanEnvironment` factories; neither game environment trains a model.

### Must NOT

- Environment transitions
- Rendering

## train_curriculum.py

Owns opt-in staged PPO training with configurable ghost counts. It reuses
`PacmanEnvironment`, Stable-Baselines3 callbacks, and shared seed/Monitor
utilities; it does not change the baseline environment defaults.

## train_ghost_direction.py

Owns the isolated ghost-direction observation experiment, including its
observation metadata, training output paths, and paired baseline evaluation.
It uses the shared PPO trainer and evaluator; it does not change defaults or
write to the baseline archive.

## train_endgame_variants.py

Owns isolated PPO runs comparing nearest-pellet/action observations and
nearest-pellet/revisit reward shaping, with a nearest-ghost observation and
proximity-penalty variant. It uses separate model, log, and result paths per
seed/step/tag/variant, then evaluates with the shared evaluator. Trajectory
recording is opt-in in `evaluate.py` and is used to measure tight loops.

## run_priority_experiments.py

Owns process-level orchestration for the 300k comparison and multi-seed
alternating refresh. Each worker has seed-specific model, log, and structured
result outputs; the parent limits concurrency and aggregates results.

## evaluate_matrix.py

Owns final-checkpoint discovery, algorithm-aware PPO/DQN model loading,
policy/model evaluation, CSV output, and robustness heatmaps. The priority
profile evaluates selected final models in separate CPU worker processes,
excludes intermediate checkpoints, and qualifies labels with their experiment
source to keep matrix rows unique. The opt-in `--endgame` profile evaluates
only the baseline, the 1M-step endgame demo model, and one 300k
features-shaping model against five ghost policies.

## analysis_plots.py

Owns the endgame seed-comparison chart and diagnostic visitation, caught-cell,
and critic-value maps. The `--endgame` profile reads existing result CSVs and
model observation sidecars; it does not train or change models.

---

## renderer.py

### Owns

- Pygame rendering
- Score display
- HUD
- Animation
- Colors

### Must NOT

- Training
- PPO
- Environment updates

---

## train.py

### Owns

- PPO initialization
- Training loop
- Saving models
- Logging
- TensorBoard

### Must NOT

- Rendering logic
- Maze generation

---

## evaluate.py

### Owns

- Model loading
- Evaluation episodes
- Metrics
- Statistics

### Must NOT

- Training

---

## plots.py

### Owns

- Reward graphs
- Episode length graphs
- Success rate graphs
- Average reward graphs

### Must NOT

- Training
- Environment updates

---

## config.py

### Owns

- All project constants
- Reward values
- Grid dimensions
- Hyperparameter defaults
- Color definitions
- Timing values
- Action mappings

### Must NOT

- Contain game logic
- Depend on any other project module

---

## utils.py

### Owns

- Shared helper functions
- Utility methods

### Must NOT

- Business logic
- Configuration constants (those belong in config.py)

---

# Source Directory Convention

All implementation modules live under `src/`.

| Module | Path |
|---|---|
| config.py | src/config.py |
| maze.py | src/maze.py |
| ghost.py | src/ghost.py |
| rewards.py | src/rewards.py |
| environment.py | src/environment.py |
| renderer.py | src/renderer.py |
| train.py | src/train.py |
| evaluate.py | src/evaluate.py |
| plots.py | src/plots.py |
| utils.py | src/utils.py |

Tests live under `tests/`.

---

# Dependency Rules

Allowed dependency direction:

Renderer
↓
Environment
↓
Maze
↓
Ghost

Training
↓
Environment

Evaluation
↓
Environment

Plots
↓
Logs

Environment must never depend on:

- PPO
- Matplotlib
- TensorBoard

Renderer must never modify game state.

---

# Architectural Principles

- Single Responsibility Principle
- Modular design
- Low coupling
- High cohesion
- Configurable components
- Reusable modules

---

# Future Extensions

Possible future additions include:

- Multiple ghost personalities
- Random maze generation
- Curriculum learning
- Multiple RL algorithms
- Procedural levels
- Teleport tunnels
- Bonus fruits

The architecture should remain extensible without major refactoring.