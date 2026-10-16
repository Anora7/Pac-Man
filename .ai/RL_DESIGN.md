# Reinforcement Learning Design

## Purpose

This document defines the reinforcement learning design of the Pac-Man project.

It serves as the authoritative reference for all RL-related decisions.

---

# Learning Agents

The baseline learning agent is Pac-Man, and baseline ghosts remain
rule-based. An optional extension can train ghost PPO policies in the
separate ghost-side environment using alternating SB3 training; it does not
change baseline defaults.

---

# Observation Space

The observation provided to the agent should contain sufficient information for effective decision-making.

Candidate information includes:

- Pac-Man position
- Ghost positions
- Pellet locations
- Power pellet location
- Wall layout
- Remaining pellets
- Power mode status

Preferred representation:

A flat vector observation compatible with `MlpPolicy`.

The observation encodes the grid state as a 1-D NumPy array (float32) derived from a multi-channel grid.

This representation was selected over a raw image grid for the following reasons:

- MlpPolicy trains faster than CnnPolicy for small grids.
- A 10x10 or 12x12 maze does not benefit significantly from convolutional feature extraction.
- CnnPolicy requires observations as uint8 arrays in (H, W, C) channel-last format, introducing unnecessary complexity.
- Flat vector observations keep the environment implementation simple and portable.

If a CNN-based observation is later justified by experimental evidence, this decision should be revisited in DECISIONS.md before changing environment.py.

Alternative representations may be evaluated if justified.

Optional, disabled-by-default observation features append the last movement
direction and vulnerability status of each ghost, and/or the previous
Pac-Man action. The baseline remains 723 values; when both optional features
are enabled on the default two-ghost maze the vector is 737 values. Models
must be evaluated with the same observation configuration used for training.

---

# Action Space

Discrete action space.

| Action | Meaning |
|---------|---------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

Invalid moves should not move Pac-Man.

---

# Reward Function

Initial reward configuration:

| Event | Reward |
|--------|--------|
| Eat pellet | +10 |
| Eat power pellet | +50 |
| Eat ghost | +100 |
| Clear maze | +500 |
| Hit ghost | -200 |
| Hit wall | -5 |
| Movement step | -1 |

Reward values should remain configurable.
The baseline values remain the default. The optional reward ablation runner
passes explicit `RewardConfig` variants through both training and evaluation;
reward totals across variants use different scales and must be interpreted
alongside success rate, pellets collected, and survival steps.

---

# Episode Termination

Episodes terminate when:

- All pellets collected
- Pac-Man is caught
- Maximum step limit reached
- Time limit exceeded

---

# Power Pellet Mechanics

Power pellet activates temporary power mode.

While active:

- Ghosts become vulnerable.
- Pac-Man can consume ghosts.
- Timer decreases every environment step.
- Ghosts return to normal after expiration.

---

# Ghost Behaviour

Default ghost behaviour remains rule-based:

Suggested policy:

- 70% random movement
- 30% chase Pac-Man if within configurable distance

Ghost movement must respect maze walls. Learned ghost policies are opt-in and
are trained only in `ghost_environment.py` and `train_alternating.py`.

---

# Training Algorithm

Preferred algorithm:

PPO

Alternative:

DQN

Training should use Stable-Baselines3 implementations without modifying the underlying algorithms.

---

# Evaluation Metrics

Training should record:

- Episode reward
- Episode length
- Average reward
- Success rate
- Number of pellets collected
- Steps taken
- Completion percentage

---

# Hyperparameters

All hyperparameters should be centralized and configurable.

Examples include:

- Learning rate
- Discount factor (gamma)
- Batch size
- Number of rollout steps
- Entropy coefficient
- Total timesteps
- Evaluation interval
- Checkpoint interval

No hyperparameter should be hardcoded in multiple locations.

---

# Reproducibility

Where practical:

- Set random seeds.
- Log hyperparameters.
- Save experiment metadata.
- Record training configuration.

---

# Future Experiments

Potential investigations include:

- PPO vs DQN
- Reward shaping variations
- Different observation encodings
- Multiple ghost behaviors
- Curriculum learning
- Randomized maze layouts
- Increasing environment complexity

---

# RL Design Principles

- Learning emerges from interaction with the environment.
- Avoid manually encoding optimal behavior.
- Keep rewards interpretable.
- Balance exploration and exploitation.
- Favor reproducibility and modularity.
- Separate learning logic from environment mechanics.

This document should be updated whenever reinforcement learning design decisions change.