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
