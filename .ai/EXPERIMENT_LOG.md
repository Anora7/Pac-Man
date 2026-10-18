# Reinforcement Learning Experiment Log

## Purpose

This document records every meaningful training experiment performed during the project.

The objective is to make reinforcement learning development reproducible and to provide evidence for the final report and presentation.

Every training session that changes the learning configuration should be logged.

---

# Experiment Template

---

## Experiment ID

EXP-001

### Date

YYYY-MM-DD

### Objective

Describe what is being tested.

### Configuration

Algorithm:

PPO

Observation Representation:

TBD

Reward Function:

Pellet: +10

Power Pellet: +50

Ghost: +100

Wall: -5

Step: -1

Goal: +500

Hyperparameters

Learning Rate:

TBD

Gamma:

TBD

Batch Size:

TBD

Total Timesteps:

TBD

Seed:

TBD

---

### Results

Average Reward:

TBD

Average Episode Length:

TBD

Success Rate:

TBD

Training Time:

TBD

---

### Observations

Examples:

- Agent explored excessively.
- Agent became trapped in loops.
- Agent successfully prioritized pellets.
- Ghost avoidance improved.

---

### Problems Encountered

Examples

- Sparse rewards.
- Poor exploration.
- High variance.
- Slow convergence.

---

### Next Planned Change

Describe exactly one modification for the next experiment.

---

# Experiment Summary Table

| ID | Algorithm | Reward Avg | Success Rate | Notes |
|----|-----------|------------|--------------|------|
| EXP-001 | PPO | TBD | TBD | Initial baseline |

---

## Configured Medium CPU Budget

This is a reproducible run configuration, not a completed experiment result.
No outcome metrics have been collected yet.

| Workflow | Seeds | Training budget | Selected variants |
|---|---:|---:|---|
| Alternating PPO | 42, 43 | 3 rounds; Pac-Man 30,000 and ghost 50,000 steps per round | Baseline-initialized Pac-Man and learned ghosts |
| Curriculum PPO | 42, 43 | Up to 100,000 steps per ghost-count stage | One- then two-ghost stages |
| PPO vs DQN | 42, 43 | 100,000 steps per algorithm | PPO and DQN |
| Reward ablation | 42, 43 | 100,000 steps per variant | Baseline, caught penalty 50, no step penalty, pellet reward 20 |

Each workflow prints an approximate CPU training-time estimate before
training and writes to a separate `medium` output tree. Estimates exclude
evaluation time; actual duration depends on the host CPU and environment.
All experiment entry points limit PyTorch to one thread.

The medium alternating profile copies `models/best_model.zip` into each
seed's `round_00_pacman.zip`, then continues Pac-Man from its previous
checkpoint each round while retaining the opponent pool. Medium alternating,
PPO-vs-DQN, and ablation evaluation uses 30 episodes per model with distinct,
reproducible episode seeds. Saved PPO-vs-DQN and ablation models can be
re-evaluated with `--eval-only` without training.

The medium results refresh completed on 2026-10-18. PPO-vs-DQN and reward
ablation were re-evaluated without retraining, each over 30 episodes per model
with a distinct deterministic seed per episode. Medium alternating was
retrained from copied baseline checkpoints for both seeds and all three
rounds. The initial model is a file copy, and later Pac-Man rounds continue
from the previous checkpoint.

| Metric | PPO | DQN |
|---|---:|---:|
| Mean per-seed reward | -481.17 | -175.85 |
| Mean success rate | 0.0% | 0.0% |
| Mean pellets | 15.47 | 10.08 |
| Mean steps | 98.33 | 99.85 |

Each per-seed PPO/DQN row summarizes 30 episodes; aggregate values summarize
the two seeds. Against baseline Pac-Man over 30 episodes, the rule ghost
caught Pac-Man in 100.0% of episodes. The final learned ghost catch rates
were 83.3% for seed 42 and 16.7% for seed 43. Medium robustness evaluation
covered 11 Pac-Man models and five ghost policies (55 cells, 10 episodes per
cell). The analysis dashboard includes rounds 0–3 for each seed and plots
mean Pac-Man pellets alongside ghost catch rate.

# Experiment Guidelines
# Experiment Guidelines

Every experiment should change only one major factor whenever possible.

Examples:

✓ Learning rate

✓ Reward values

✓ Observation encoding

✓ PPO parameters

✓ Ghost behavior

Avoid changing many variables simultaneously.

---

# Metrics to Record

Whenever possible, collect:

- Average reward
- Episode reward
- Episode length
- Success rate
- Pellets collected
- Ghost collisions
- Maze completion percentage
- Training duration
- Number of updates

---

# Visualizations

Each experiment should generate or reference:

- Reward curve
- Moving average reward
- Episode length
- Success rate
- TensorBoard logs (optional)

---

# Final Comparison

By the end of the project, summarize:

- Best-performing configuration
- Worst-performing configuration
- Lessons learned
- Recommended configuration for demonstration

This section will directly support the Results and Discussion portions of the project report.