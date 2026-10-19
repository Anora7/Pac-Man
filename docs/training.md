# Training

## Overview

Training runs PPO via Stable-Baselines3. All settings are controlled by `TrainingConfig`. The `train()` function handles environment creation, callback setup, model saving, and TensorBoard logging.

---

## Quick Start

```bash
python src/train.py
```

This uses all default hyperparameters from `src/config.py` and writes output to `models/` and `logs/`.

---

## TrainingConfig

All hyperparameters and file paths are defined in a single dataclass:

```python
from train import TrainingConfig, train
from pathlib import Path

config = TrainingConfig(
    # Paths
    models_dir      = Path("models"),
    logs_dir        = Path("logs"),
    tensorboard_dir = Path("logs/tensorboard"),

    # PPO hyperparameters (defaults from config.py)
    total_timesteps = 500_000,
    learning_rate   = 3e-4,
    gamma           = 0.99,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    ent_coef        = 0.01,
    clip_range      = 0.2,

    # Device
    device          = "auto",   # retained for compatibility; training uses CPU

    # Reproducibility
    seed            = None,     # omitted seeds resolve to 42 for training

    # Environment
    max_steps       = 1000,     # steps per episode before truncation

    # Callbacks
    eval_freq       = 10_000,   # evaluate every N timesteps
    n_eval_episodes = 5,
    checkpoint_freq = 10_000,

    # Output
    verbose         = 1,
    show_progress_bar = False,  # requires rich + tqdm if True
)

model = train(config)
```

---

## Output Files

| File | Description |
|---|---|
| `models/best_model.zip` | Highest-scoring model across all evaluations |
| `models/final_model.zip` | Model state at the end of training |
| `models/pacman_ppo_N_steps.zip` | Periodic checkpoints every `checkpoint_freq` steps |
| `logs/0.monitor.csv` | Per-episode reward, length, wall-clock time, and success |
| `logs/evaluations.npz` | Evaluation statistics from `EvalCallback` |
| `logs/tensorboard/` | TensorBoard event files |

---

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Key metrics visible in TensorBoard:

- `rollout/ep_rew_mean` — mean episode reward
- `rollout/ep_len_mean` — mean episode length
- `train/entropy_loss`, `train/policy_loss`, `train/value_loss`
- `train/learning_rate`

---

## Loading a Saved Model

```python
from stable_baselines3 import PPO

model = PPO.load("models/best_model")
# model.predict(obs, deterministic=True)
```

---

## Hyperparameter Notes

| Parameter | Guidance |
|---|---|
| `total_timesteps` | Start with 500k; increase to 1–2M for better convergence |
| `learning_rate` | 3e-4 is a safe default; try 1e-4 if training is unstable |
| `n_steps` | Must satisfy `n_steps × n_envs ≥ batch_size` |
| `ent_coef` | Higher values (0.05) promote exploration early in training |
| `gamma` | 0.99 balances short- and long-term rewards well for this maze |

All reward values that influence learning are in `src/config.py` → `RewardConfig`.

The `device` field is retained for source compatibility, but this project
always passes `device="cpu"` to Stable-Baselines3. When `seed` is omitted,
training uses the fixed project seed `42`.

For optional observations, set `observation_config` to an
`ObservationConfig` when constructing `TrainingConfig`. Training passes the
same setting to both environments and writes a `.observation.json` sidecar
next to final and best model archives. Keep these model and output paths under
`models/experiments/` and `logs/experiments/` to avoid replacing baseline
checkpoints.

The dedicated 300,000-step ghost-direction PPO run and its 30-episode
baseline comparison are available from the repository root:

```powershell
python .\src\train_ghost_direction.py --steps 300000 --episodes 30 --seed 42
```

The script enables `ObservationConfig(include_ghost_direction=True)`, stores
the model at `models\experiments\ghost_direction_ppo_300k\final_model.zip`,
and writes per-model metrics with 95% episode-level confidence intervals to
`results\priority_300k\ghost_direction_comparison.csv`. Its evaluation seeds
are disjoint between the baseline and trained model.

### Endgame variants

`src\train_endgame_variants.py` compares nearest-pellet and previous-action
observations with optional BFS-progress and recent-cell revisit shaping. The
`features_shaping_ghost` arm additionally observes nearest hostile-ghost
distance, escape direction, and last move direction, with a -0.5 penalty when
Pac-Man moves closer while within three steps. The four variants each default
to 300,000 PPO steps and seed 42, then evaluate
baseline and trained models for 30 deterministic and 30 stochastic episodes.
Evaluation seed ranges are distinct across variants, models, and policies.

By default, each run is placed in a folder named for its seed and step count
(for example, `seed_42_steps_300000`). Use `--output-tag` to add a run label;
the seed and step count remain part of the folder name, preventing collisions
even when the same label is reused:

```powershell
python .\src\train_endgame_variants.py --output-tag demo
```

Run one variant in its own terminal/process; each has separate model, log, and
result files under the tagged run folder:

```powershell
python .\src\train_endgame_variants.py --variant features_only --steps 300000 --seed 42 --output-tag demo
python .\src\train_endgame_variants.py --variant features_shaping --steps 300000 --seed 42 --output-tag demo
python .\src\train_endgame_variants.py --variant features_shaping_ghost --steps 300000 --seed 42 --output-tag demo
python .\src\train_endgame_variants.py --variant shaping_only --steps 300000 --seed 42 --output-tag demo
```

Models and logs are stored under
`models\experiments\endgame_300k\<run-tag>\<variant>\` and
`logs\experiments\endgame_300k\<run-tag>\<variant>\`. A separate variant
process writes
`results\priority_300k\endgame_variants\<run-tag>\<variant>.csv`; the
all-variants command also writes a combined
`endgame_variants.csv` in that same run-tag folder. The baseline archive is
loaded for evaluation only and is never a training output path.

The `loop_rate` metric is the fraction of episodes containing at least 50
consecutive steps whose trajectory occupies no more than three distinct
cells. `clears` is the number of cleared episodes in that evaluation row.
