# Quick Start Guide

Follow these steps in order. Everything runs from the project root directory.

---

## Step 0 — Install Dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.12+.

---

## Step 1 — Verify the Installation

Run the full test suite to confirm everything is working correctly:

```bash
python -m pytest tests/ -q
```

Expected output: `327 passed`.

---

## Step 2 — Train the Agent

```bash
python src/train.py
```

This will:
- Train PPO for **500,000 steps** (≈ 10–30 minutes depending on hardware)
- Save `models/best_model.zip` — best checkpoint across training
- Save `models/final_model.zip` — model at end of training
- Write `logs/0.monitor.csv` — per-episode statistics
- Write `logs/tensorboard/` — TensorBoard logs

To watch progress in real time (separate terminal):

```bash
tensorboard --logdir logs/tensorboard
```

Then open `http://localhost:6006` in your browser.

---

## Step 3 — Evaluate the Trained Agent

```bash
python src/evaluate.py --model models/best_model.zip --episodes 20
```

To watch the agent play (opens a Pygame window):

```bash
python src/evaluate.py --model models/best_model.zip --episodes 5 --render
```

Sample output:
```
=== Pac-Man RL — Evaluation ===
  Ep   1:  reward=   850.0  steps= 312  pellets=69/69  [CLEARED]
  Ep   2:  reward=  -201.0  steps= 189  pellets=18/69  [CAUGHT]
  ...
──────────────────────────────────────────────────
  Success rate       : 50.0%  (10/20 cleared)
  Mean reward        : +412.50 ± 380.20
──────────────────────────────────────────────────
```

---

## Step 4 — Generate Training Plots

```python
from plots import generate_all_plots, PlotConfig
from pathlib import Path

generate_all_plots(
    log_path=Path("logs/0.monitor.csv"),
    config=PlotConfig(plots_dir=Path("plots"), window=50),
)
```

Or run it as a one-liner from the terminal:

```bash
python -c "
from plots import generate_all_plots, PlotConfig
from pathlib import Path
generate_all_plots(Path('logs/0.monitor.csv'), config=PlotConfig())
"
```

Saves four PNG files to `plots/`:

| File | Content |
|---|---|
| `reward_curve.png` | Raw episode reward over training |
| `moving_average.png` | Smoothed reward trend |
| `episode_length.png` | Steps per episode over training |
| `success_rate.png` | Cleared / Caught / Timeout breakdown |

---

## Step 5 — Generate All Plots Including Evaluation

```python
from evaluate import EvaluationConfig, evaluate
from plots import generate_all_plots, PlotConfig
from pathlib import Path

result = evaluate(EvaluationConfig(
    model_path=Path("models/best_model.zip"),
    n_episodes=20,
    verbose=False,
))

generate_all_plots(
    log_path=Path("logs"),
    result=result,
    config=PlotConfig(plots_dir=Path("plots")),
)
```

---

## Training Configuration (Optional)

To customise training hyperparameters, edit `src/config.py` or pass a `TrainingConfig`:

```python
from train import TrainingConfig, train
from pathlib import Path

model = train(TrainingConfig(
    total_timesteps=1_000_000,   # longer training = better agent
    learning_rate=3e-4,
    seed=42,                     # set for reproducible results
    verbose=1,
))
```

---

## What to Expect

| Phase | Timestep range | Typical agent behaviour |
|---|---|---|
| Early (0–50k) | Random exploration | Moves erratically, often caught |
| Mid (50k–200k) | Improving | Collects some pellets, avoids some ghosts |
| Late (200k–500k+) | Learning strategy | Systematic pellet collection, power pellet use |

A well-trained agent should achieve **>30% maze-clear rate** after 500k steps and improve further with longer training.

---

## File Overview

```
src/
  config.py       ← tweak rewards and hyperparameters here
  train.py        ← start training
  evaluate.py     ← load and run a trained model
  plots.py        ← generate matplotlib figures

models/           ← trained models saved here after training
logs/             ← CSV logs and TensorBoard files saved here
plots/            ← PNG figures saved here
```
