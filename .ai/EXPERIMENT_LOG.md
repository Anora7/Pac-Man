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