# Project: Reinforcement Learning Pac-Man using Stable-Baselines3

## Project Overview

This project aims to design, implement, train, and evaluate an intelligent Pac-Man agent using Reinforcement Learning (RL). The agent will learn entirely through interaction with a custom Gymnasium environment and will not rely on manually programmed gameplay strategies.

The environment recreates a simplified version of the classic Pac-Man game on a small grid-based maze suitable for reinforcement learning experimentation and academic demonstration.

The final outcome should demonstrate that an RL agent can progressively improve its performance by maximizing cumulative rewards over many training episodes.

---

# Project Objectives

The primary objective is to create a fully functional reinforcement learning pipeline that includes:

- A custom Gymnasium environment.
- A rule-based game world.
- A PPO agent trained using Stable-Baselines3.
- Performance evaluation.
- Visualization of learning progress.
- A presentation-ready demonstration.

The implementation should prioritize modularity, maintainability, reproducibility, and educational value.

---

# Why This Project Exists

This project is intended to demonstrate practical applications of Reinforcement Learning by solving a sequential decision-making problem.

Rather than programming Pac-Man with fixed movement rules, the agent must learn an effective strategy through repeated interaction with the environment.

The project should clearly illustrate the difference between:

- Rule-based systems
- Reinforcement learning
- Reward-driven optimization

---

# Technologies

The project shall use only the following primary technologies.

## Programming Language

- Python

## Reinforcement Learning

- Stable-Baselines3
- PPO (preferred)
- DQN (optional comparison only)

## Environment

- Gymnasium

## Rendering

- Pygame

## Numerical Computation

- NumPy

## Visualization

- Matplotlib

## Logging

- TensorBoard (optional)

---

# Assignment Requirements

The implementation must satisfy the following academic requirements.

## Environment

- Custom Gymnasium environment
- Grid-based maze
- Approximately 10x10 or 12x12

## Game Objects

- Walls
- Empty paths
- Pellets
- One power pellet
- Pac-Man
- One or two ghosts

## Reinforcement Learning

- Pac-Man is the only learning agent.
- Baseline ghosts must remain rule-based; optional experimental ghost
  training is documented separately and never changes the default.

## Evaluation

Training should demonstrate measurable improvement over time.

Graphs should clearly show learning progress.

---

# Learning Objectives

The project should demonstrate understanding of:

- Reinforcement Learning fundamentals
- Markov Decision Processes
- Gymnasium environment design
- Stable-Baselines3 workflows
- PPO training
- Reward shaping
- Environment design
- Policy learning
- Agent evaluation
- Experimental analysis

---

# Expected Deliverables

The final repository should contain:

- Modular source code
- Custom Gymnasium environment
- PPO training pipeline
- Saved trained models
- Evaluation scripts
- Rendering system
- Training graphs
- Documentation
- Presentation materials

---

# Constraints

The project should follow these constraints.

## Functional

- No hard-coded Pac-Man strategies
- Ghosts remain non-learning agents
- Environment follows Gymnasium API
- Code remains modular

## Technical

- Python implementation
- Stable-Baselines3 PPO
- Grid-world environment
- Deterministic evaluation where appropriate

## Development

- Maintain clean architecture.
- Separate rendering from environment logic.
- Separate training from evaluation.
- Keep reward values configurable.

---

# Success Criteria

The project will be considered successful if:

- Environment functions correctly.
- PPO training completes successfully.
- Episode rewards improve over time.
- Agent demonstrates learned behavior.
- Evaluation metrics improve after training.
- Training graphs show positive trends.
- Documentation remains synchronized with implementation.
- Repository is suitable for academic presentation.

---

# Non-Goals

The following are intentionally outside the scope of this project.

- Perfect recreation of the original Pac-Man.
- Human-level gameplay.
- Simultaneous/multi-agent RL as part of the baseline assignment (an optional
  alternating experiment is documented in `docs/extensions.md`).
- Large procedurally generated worlds.
- Multiplayer gameplay.
- Deep neural architecture experimentation.

---

# Documentation Policy

This document intentionally contains no implementation details.

Implementation decisions belong in:

- ARCHITECTURE.md
- RL_DESIGN.md
- IMPLEMENTATION_PLAN.md
- DECISIONS.md

This document should change only when the overall project goals or assignment requirements change.