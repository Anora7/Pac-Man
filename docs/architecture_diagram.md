# Architecture Diagram

## High-Level Data Flow

```text
             User
               │
               ▼
         src/train.py
               │
               ▼
   Stable-Baselines3 (PPO)
               │
               ▼
    src/environment.py
     (Gymnasium Interface)
               │
      ┌────────┴────────┐
      ▼                 ▼
  src/maze.py      src/ghost.py
      │                 │
      └────────┬────────┘
               ▼
        src/rewards.py
               │
               ▼
         Game State
               │
      ┌────────┴────────┐
      ▼                 ▼
 src/renderer.py   src/evaluate.py
  (Pygame HUD)     (Metrics)
                        │
                        ▼
                  src/plots.py
                  (Matplotlib)
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/config.py` | All constants: rewards, grid size, colors, hyperparameters |
| `src/maze.py` | Grid layout, walls, pellets, spawn locations |
| `src/ghost.py` | Rule-based ghost movement, vulnerable mode, respawn |
| `src/rewards.py` | Reward calculation using values from config.py |
| `src/environment.py` | Gymnasium API: reset(), step(), close() |
| `src/renderer.py` | Pygame rendering, HUD, score display |
| `src/train.py` | PPO initialization, training loop, model saving |
| `src/evaluate.py` | Model loading, evaluation episodes, metrics |
| `src/plots.py` | Matplotlib graphs from training logs |
| `src/utils.py` | Shared helper utilities |
| `tests/` | Validation tests for each module |

---

## Dependency Rules

- `environment.py` depends on `maze.py`, `ghost.py`, `rewards.py`, `config.py`
- `renderer.py` depends on `environment.py` (read-only, never modifies state)
- `train.py` depends on `environment.py`
- `evaluate.py` depends on `environment.py`, `renderer.py`
- `plots.py` depends on log files only — no environment dependency
- `config.py` has **no** project-internal dependencies

The environment must never depend on PPO, Matplotlib, or TensorBoard.
