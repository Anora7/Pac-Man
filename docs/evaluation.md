# Evaluation

## Overview

The evaluation pipeline loads a trained PPO model and runs it for a configurable number of episodes, collecting per-episode metrics and printing a summary.

---

## Command Line

```bash
python src/evaluate.py --model models/best_model.zip --episodes 20
python src/evaluate.py --model models/best_model.zip --episodes 10 --render
python src/evaluate.py --model models/final_model.zip --episodes 50 --no-deterministic --seed 0
python src/evaluate.py --model models/best_model.zip --episodes 20 --ghost-type random --seed 42 --save-csv results/evaluation.csv
python src/evaluate.py --model models/experiments/alternating/seed_42/round_01_pacman.zip --ghost-type learned --ghost-model models/experiments/alternating/seed_42/round_01_ghost.zip
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/best_model.zip` | Path to the `.zip` model file |
| `--episodes` | `10` | Number of evaluation episodes |
| `--render` | off | Open Pygame window |
| `--no-deterministic` | off | Use stochastic policy |
| `--seed` | `None` | Seed for `env.reset()` |
| `--ghost-type` | `rule` | Compare against `rule`, `random`, `chase`, or `learned` ghosts |
| `--ghost-model` | — | Ghost PPO archive required for `--ghost-type learned` |
| `--obs-config` | `baseline` | `baseline`, `ghost-direction`, `previous-action`, or `enhanced` |
| `--render-fps` | `8` | Pygame playback rate |
| `--save-csv` | — | Optional per-episode metrics CSV |

---

## Python API

```python
from evaluate import EvaluationConfig, evaluate
from pathlib import Path

config = EvaluationConfig(
    model_path    = Path("models/best_model.zip"),
    n_episodes    = 20,
    render        = False,
    deterministic = True,
    seed          = 42,
    max_steps     = 1000,
    verbose       = True,
)
result = evaluate(config)
```

To evaluate a model that was just trained (no disk I/O):

```python
from train import train, TrainingConfig
from evaluate import evaluate, EvaluationConfig

model = train(TrainingConfig(total_timesteps=100_000, verbose=0))
result = evaluate(EvaluationConfig(n_episodes=10), model=model)
```

---

## Returned Metrics

`evaluate()` returns an `EvaluationResult`:

| Field | Type | Description |
|---|---|---|
| `n_episodes` | `int` | Number of episodes run |
| `mean_reward` | `float` | Mean total reward per episode |
| `std_reward` | `float` | Standard deviation of reward |
| `mean_steps` | `float` | Mean episode length |
| `mean_survival_steps` | `float` | Alias for mean episode length |
| `mean_pellets_collected` | `float` | Mean pellets eaten per episode |
| `success_rate` | `float` | Fraction of episodes where maze was cleared (0.0–1.0) |
| `caught_rate` | `float` | Fraction of episodes ending in a ghost capture |
| `timeout_rate` | `float` | Fraction of episodes reaching the step limit |
| `avg_completion_fraction` | `float` | Mean fraction of pellets collected |
| `episodes` | `list[EpisodeResult]` | Per-episode breakdown |

When a `.observation.json` sidecar exists next to a model, evaluation loads
its observation settings automatically. Missing sidecars mean the baseline
723-value observation; explicitly pass `observation_config` or `--obs-config`
for models trained with optional features.
When a seed is supplied, episode 1 uses that seed and each subsequent episode
uses the next integer seed, so evaluation episodes have distinct reproducible
environment seeds.

Each `EpisodeResult` includes: `total_reward`, `steps`, `pellets_collected`, `total_pellets`, `cleared`, `caught`, `truncated`, `completion_fraction`.

---

## Sample Output

```
=== Pac-Man RL — Evaluation ===
Episodes:      20
Deterministic: True
Max steps/ep:  1000

  Ep   1:  reward=   850.0  steps= 312  pellets=69/69  [CLEARED]
  Ep   2:  reward=  -201.0  steps= 189  pellets=18/69  [CAUGHT]
  ...

──────────────────────────────────────────────────
  EVALUATION SUMMARY
──────────────────────────────────────────────────
  Episodes evaluated : 20
  Mean reward        : +412.50 ± 380.20
  Mean episode steps : 540.2
  Mean pellets/ep    : 48.3
  Avg completion     : 70.0%
  Success rate       : 50.0%  (10/20 cleared)
  Caught             : 6/20
  Timeouts           : 4/20
──────────────────────────────────────────────────
```

---

## Generating Plots from Evaluation Results

```python
from plots import generate_all_plots, PlotConfig
from pathlib import Path

generate_all_plots(
    result=result,
    config=PlotConfig(plots_dir=Path("plots")),
)
# saves: plots/success_rate.png
```
