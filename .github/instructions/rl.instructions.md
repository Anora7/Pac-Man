# Reinforcement Learning Instructions

## Learning Agent

Pac-Man is the only learning agent.

Ghosts are always rule-based.

---

# Environment

Use Gymnasium conventions.

Implement:

- reset()
- step()
- render()
- close()

Follow the Gymnasium API.

---

# Observation Space

Provide enough information for effective decision-making.

Maintain consistency throughout training and evaluation.

---

# Action Space

Discrete actions only.

Up

Down

Left

Right

Invalid moves should not move Pac-Man.

---

# Reward Design

Reward values must remain configurable.

Avoid embedding reward values throughout the codebase.

Reward shaping should encourage:

- Pellet collection
- Efficient navigation
- Ghost avoidance
- Maze completion

---

# PPO

Use Stable-Baselines3.

Prefer:

- PPO

Alternative:

- DQN (comparison only)

---

# Training

Support:

- CPU
- GPU
- Automatic device selection

Save:

- Latest model
- Best model
- Training logs

---

# Evaluation

Evaluation should:

- Load trained models
- Report metrics
- Optionally render gameplay

---

# Experiments

Whenever training parameters change:

Update:

- EXPERIMENT_LOG.md

Document:

- Hyperparameters
- Results
- Lessons learned

---

# Reproducibility

Whenever practical:

- Use random seeds
- Log configurations
- Save checkpoints