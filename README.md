# Pac-Man Reinforcement Learning

A complete Reinforcement Learning implementation of Pac-Man using **Gymnasium** and **Stable-Baselines3 PPO**.

The baseline agent learns to navigate a 12×12 grid maze entirely through
trial-and-error interaction with the environment — no hard-coded Pac-Man
movement rules. Optional research extensions provide learned-ghost
experiments, robustness evaluation, reward ablation, and curriculum learning
without changing baseline defaults.

---

## What This Project Demonstrates

| Concept | Implementation |
|---|---|
| Custom Gymnasium environment | `src/environment.py` |
| Rule-based ghost AI | `src/ghost.py` |
| PPO training with SB3 | `src/train.py` |
| Flat-vector observation space | 723 float32 values |
| Reward shaping | `src/rewards.py` |
| Model evaluation | `src/evaluate.py` |
| Training curve visualisation | `src/plots.py` |
| Pygame rendering | `src/renderer.py` |
| Optional experiment suite | `run_all_experiments.py`, `docs/extensions.md` |

---

## Repository Structure

```
RL/
├── src/
│   ├── config.py        # All constants: rewards, maze, colours, hyperparameters
│   ├── maze.py          # Maze grid, pellets, spawn locations
│   ├── ghost.py         # Rule-based ghost AI (70% random / 30% chase)
│   ├── rewards.py       # Reward calculation function
│   ├── environment.py   # Custom Gymnasium environment (PacmanEnvironment)
│   ├── renderer.py      # Pygame visualisation (RenderState + PacmanRenderer)
│   ├── train.py         # PPO training pipeline (TrainingConfig + train())
│   ├── train_dqn.py     # Optional CPU DQN comparison
│   ├── train_alternating.py # Alternating Pac-Man/ghost PPO experiment
│   ├── train_curriculum.py  # Staged ghost-count curriculum
│   ├── evaluate.py      # Model evaluation (EvaluationConfig + evaluate())
│   └── plots.py         # Matplotlib graph generation (PlotConfig + helpers)
├── run_all_experiments.py # Quick/full extension workflow
├── tests/               # Unit and smoke tests across all modules
├── models/              # Saved models (best_model.zip, final_model.zip, checkpoints)
├── logs/                # SB3 Monitor CSV logs and TensorBoard event files
├── plots/               # Generated PNG figures
├── docs/                # User-facing documentation
└── .ai/                 # Project knowledge base (architecture, decisions, progress)
```

---

## Requirements

Python 3.12+ with the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| stable-baselines3 | 2.9.0 | PPO algorithm |
| gymnasium | 1.3.0 | Environment API |
| pygame | 2.6.1 | Visual rendering |
| matplotlib | 3.11.1 | Plot generation |
| numpy | 2.5.1 | Numerical computation |
| torch | 2.13.0 | Neural network backend |
| tensorboard | 2.21.0 | Training log viewer |

---

## Quick Start

### 1 — Run Tests

```bash
python -m pytest tests/ -q
```

### 2 — Train

```bash
python src/train.py
```

Training saves:
- `models/best_model.zip` — highest-scoring checkpoint (via `EvalCallback`)
- `models/final_model.zip` — model at end of training
- `models/pacman_ppo_*_steps.zip` — periodic checkpoints
- `logs/0.monitor.csv` — per-episode statistics
- `logs/tensorboard/` — TensorBoard event files

To customise hyperparameters:

```python
from train import TrainingConfig, train
from pathlib import Path

config = TrainingConfig(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    device="auto",           # retained for compatibility; training uses CPU
    seed=42,
    models_dir=Path("models"),
    logs_dir=Path("logs"),
)
model = train(config)
```

### 3 — Evaluate

```bash
python src/evaluate.py --model models/best_model.zip --episodes 20
```

Flags: `--render` to open the Pygame window, `--no-deterministic` for stochastic policy.

Or from Python:

```python
from evaluate import EvaluationConfig, evaluate
from pathlib import Path

result = evaluate(EvaluationConfig(
    model_path=Path("models/best_model.zip"),
    n_episodes=20,
    render=True,
))
print(f"Success rate: {result.success_rate:.1%}")
print(f"Mean reward:  {result.mean_reward:+.1f}")
```

### 4 — Generate Plots

```python
from plots import PlotConfig, generate_all_plots
from pathlib import Path

generate_all_plots(
    log_path=Path("logs/0.monitor.csv"),
    result=eval_result,
    config=PlotConfig(plots_dir=Path("plots"), window=50),
)
```

Produces `plots/reward_curve.png`, `plots/moving_average.png`, `plots/episode_length.png`, `plots/success_rate.png`.

### 5 — View TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Maze

```
W W W W W W W W W W W W
W . . . . . . . . . . W
W . W . W . . W . W . W
W O . . . . . . . . O W   ← power pellets at (3,1) and (3,10)
W . W . W · · W . W . W   ← ghost-house door (empty cells)
W . . . W G G W . . . W   ← ghost spawns at (5,5) and (5,6)
W W W . W W W W . W W W   ← bridge at col 3 and col 8
W . . . . . . . . . . W
W . W . W . . W . W . W
W . W . W . . W . W . W
W . . . . P . . . . . W   ← Pac-Man spawn at (10,5)
W W W W W W W W W W W W
```

**67 regular pellets · 2 power pellets · 2 ghost spawns · 1 Pac-Man spawn**

---

## Observation Space

The agent receives a 723-element `float32` vector with all values in [0, 1]:

| Index range | Channel | Description |
|---|---|---|
| 0 – 143 | Wall grid | 1.0 at wall cells |
| 144 – 287 | Pellet grid | 1.0 at uneaten regular pellets |
| 288 – 431 | Power pellet grid | 1.0 at uneaten power pellets |
| 432 – 575 | Pac-Man grid | 1.0 at Pac-Man's cell |
| 576 – 719 | Ghost grid | 1.0 at each ghost's cell |
| 720 | Power mode active | 0.0 or 1.0 |
| 721 | Power timer (normalised) | `steps_remaining / 30` |
| 722 | Pellets remaining (normalised) | `remaining / total` |

---

## Action Space

`Discrete(4)` — `0`=Up, `1`=Down, `2`=Left, `3`=Right.  
Invalid moves (wall collision) leave Pac-Man in place and incur a small penalty.

---

## Reward Function

| Event | Reward |
|---|---|
| Eat pellet | +10 |
| Eat power pellet | +50 |
| Eat ghost (power mode) | +100 |
| Clear maze | +500 |
| Caught by ghost | −200 |
| Hit wall | −5 |
| Each step | −1 |

All values are configurable in `src/config.py` → `RewardConfig`.

---

## Ghost Behaviour

The baseline ghosts are rule-based. Each step:

1. If distance to Pac-Man ≤ `GHOST_CHASE_RADIUS` (5) **and** `random() < 0.30`: move toward Pac-Man (Manhattan distance).
2. Otherwise: move to a random walkable neighbour.
3. During power mode: always random (vulnerable, can be eaten).

---

## Architecture

```
train.py
    └── PacmanEnvironment (environment.py)
            ├── Maze (maze.py)
            ├── Ghost × 2 (ghost.py)
            ├── calculate_step_reward (rewards.py)
            └── PacmanRenderer (renderer.py)  ← only when render_mode="human"

evaluate.py  →  PacmanEnvironment  →  metrics
plots.py     →  Monitor CSV / EvaluationResult  →  PNG figures
```

**Dependency rules:** `renderer.py` never modifies game state.
`environment.py` never depends on training code. Baseline ghosts remain
rule-based; optional ghost PPO is isolated in its training environment.

---

## Optional Extensions

Learned-ghost experiments, improved observations, curriculum training,
PPO-vs-DQN, reward ablations, multi-seed analysis, robustness matrices, and
diagnostic plots are documented in the
[extension guide](docs/extensions.md). Quick verification:

```powershell
python .\run_all_experiments.py --quick
```

---

## Tests

The suite covers core environment behavior, training/evaluation smoke tests,
and the optional extension workflows. The final full-suite run passed all 494
tests:

```powershell
python -m pytest tests\ -q
```
