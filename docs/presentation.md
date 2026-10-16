# Presentation Notes

Academic presentation guide for this Pac-Man Reinforcement Learning project.

---

## Suggested Slide Structure

### Slide 1 — Title

**Pac-Man with Reinforcement Learning**
*Custom Gymnasium Environment + PPO (Stable-Baselines3)*

---

### Slide 2 — Motivation

- Classic AI benchmark with clear objectives
- Sequential decision-making under uncertainty
- Demonstrates the difference between:
  - **Rule-based systems** — explicit IF/THEN logic (ghost AI)
  - **Reinforcement learning** — learn by reward signal (Pac-Man)

---

### Slide 3 — Environment

Show the maze diagram from README.md.

Key points:
- 12×12 grid, 69 total pellets
- 4 discrete actions (Up / Down / Left / Right)
- Episode ends on: maze cleared, caught by ghost, or step limit
- Two rule-based ghosts: 70% random, 30% chase

---

### Slide 4 — Observation Space

Show the observation table:

> "The agent sees 723 numbers, each between 0 and 1."

- Five 12×12 channels (walls, pellets, power pellets, Pac-Man, ghosts)
- Three scalar features (power mode, timer, completion fraction)
- Flat vector → `MlpPolicy` — no image processing needed

---

### Slide 5 — Reward Function

Present the reward table from README.

Key insight: **the step penalty (−1) forces urgency** — an agent that stands still always loses score.

---

### Slide 6 — PPO Algorithm

- Proximal Policy Optimization (Schulman et al., 2017)
- Policy gradient method with clipped surrogate objective
- Stable, sample-efficient, works well on discrete action spaces
- Implemented in Stable-Baselines3 (no modifications to algorithm)

Show: `TrainingConfig` fields — all hyperparameters in one place.

---

### Slide 7 — Training Results

Show `plots/reward_curve.png` and `plots/moving_average.png`.

Points to highlight:
- Episode reward starts negative (random policy)
- Reward increases as agent learns pellet-collecting strategies
- Moving average smooths the noisy per-episode signal

---

### Slide 8 — Evaluation Results

Show `plots/success_rate.png`.

Report:
- Mean episode reward
- Success rate (maze cleared)
- Average completion fraction
- Comparison: random policy vs trained policy

---

### Slide 9 — Live Demo (optional)

```bash
python src/evaluate.py --model models/best_model.zip --render --episodes 5
```

Highlights to point out during the demo:
- Agent moves purposefully toward pellets
- Agent avoids or runs from ghosts
- Power mode: agent may seek ghosts when power is active

---

### Slide 10 — Architecture

Show the architecture diagram from `docs/architecture_diagram.md`.

Key separation of concerns:
- Ghosts are **never trained** (rule-based only)
- Renderer **never modifies** game state
- Environment has **no knowledge** of the training algorithm

---

### Slide 11 — Conclusion

- Demonstrated end-to-end RL pipeline
- Agent learns meaningful behaviour from reward signal alone
- Modular architecture supports future extensions
- Full test suite (327 tests) ensures correctness

---

## Demo Commands

```bash
# Full training run
python src/train.py

# Evaluate best model with rendering
python src/evaluate.py --model models/best_model.zip --render --episodes 10

# View training curves
python -c "
from plots import generate_all_plots, PlotConfig
from pathlib import Path
generate_all_plots(Path('logs'), config=PlotConfig())
"
# then open plots/

# TensorBoard
tensorboard --logdir logs/tensorboard
```

---

## Key Numbers to Cite

| Metric | Value |
|---|---|
| Maze size | 12×12 |
| Observation vector | 723 float32 values |
| Action space | Discrete(4) |
| Regular pellets | 67 |
| Power pellets | 2 |
| Default episode limit | 1000 steps |
| Power mode duration | 30 steps |
| Ghost chase probability | 30% |
| Default training timesteps | 500,000 |
| Test suite | 327 tests |
