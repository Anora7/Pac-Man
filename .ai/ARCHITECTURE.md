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
and game environment do not train models.

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