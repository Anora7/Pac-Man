# Pac-Man RL Task Board

## Purpose

This document acts as the project's internal Kanban board.

It should always reflect the current development status.

Every implementation task should move through the following lifecycle:

TODO → IN PROGRESS → DONE

Only one major feature should normally be in progress at a time.

---

# TODO

## Core Environment

- [x] Create maze representation
- [x] Implement configurable maze loader
- [x] Implement Pac-Man movement (environment.py step())
- [x] Implement ghost movement (wired in step())
- [x] Implement collision detection (dual: pre/post ghost-move)
- [x] Implement power pellet mechanics and timer
- [x] Implement reward system (rewards.py)
- [x] Implement episode termination (cleared, caught, step limit)

---

## Gymnasium

- [x] Create custom environment (environment.py)
- [x] Implement reset()
- [x] Implement step()
- [x] Implement render() (delegates to renderer.py)
- [x] Implement close()
- [x] Validate Gymnasium compliance (SB3 check_env passes)

## Reinforcement Learning

- [x] Configure PPO (TrainingConfig)
- [x] Create training pipeline (train.py)
- [x] Save checkpoints (CheckpointCallback)
- [x] Save best model (EvalCallback)
- [x] Log statistics (Monitor wrapper)
- [x] Configure TensorBoard (tensorboard_log parameter)

---

## Evaluation (Phase 7)

- [x] Load trained model from .zip path
- [x] Support pre-loaded model (no file I/O)
- [x] Run N evaluation episodes
- [x] Report per-episode: reward, steps, pellets, outcome
- [x] Report aggregate: mean/std reward, success rate, completion
- [x] Optional visual rendering
- [x] CLI entry point (argparse)
- [x] 40 tests passing

## Evaluation (TODO)

- [ ] Load trained model
- [ ] Evaluate multiple episodes
- [ ] Record metrics
- [ ] Display gameplay

---

## Visualization

- [x] Pygame renderer (`src/renderer.py`)
- [x] HUD (score, step, episode)
- [x] Score displayed
- [x] Episode number displayed
- [x] Step counter displayed
- [x] Power mode status displayed
- [x] Wall rendering
- [x] Pellet rendering (regular + power)
- [x] Pac-Man rendering
- [x] Ghost rendering (normal vs vulnerable)

## Phase 5 — Renderer

- [x] RenderState frozen dataclass
- [x] PacmanRenderer class
- [x] Lazy pygame initialisation
- [x] close() safe before render()
- [x] Headless tests (SDL_VIDEODRIVER=dummy)
- [x] 37 tests passing

---

## Graphs

- [x] Reward curve (`reward_curve.png`)
- [x] Moving average (`moving_average.png`)
- [x] Episode length (`episode_length.png`)
- [x] Success rate (`success_rate.png`)
- [x] Reads Monitor CSV for training data
- [x] Reads EvaluationResult for eval data
- [x] generate_all_plots() convenience wrapper
- [x] 41 tests passing

---

## Documentation

- [x] API reference (`docs/api_reference.md`)
- [x] Architecture documentation (`docs/architecture_diagram.md`)
- [x] Training guide (`docs/training.md`)
- [x] Evaluation guide (`docs/evaluation.md`)
- [x] Environment reference (`docs/environment.md`)
- [x] Presentation notes (`docs/presentation.md`)

---

---

# COMPLETED — OPTIONAL EXTENSIONS

All requested optional experiment extensions are complete.

## Step 1 — Ground Rules and Configuration

- [x] Preserve baseline observation, rewards, and default ghost behavior
- [x] Centralize optional observation and ghost-training reward defaults
- [x] Ensure training uses CPU and a reproducible default seed
- [x] Step 2 — Opt-in observations; default remains 723 values
- [x] Step 3 — Ghost policy abstraction; rule-based default preserved
- [x] Step 4 — Ghost training environment using shared game transitions
- [x] Step 5 — Alternating PPO training with opponent pools and CSV metrics
- [x] Step 6 — Evaluation options, metadata, and robustness matrix
- [x] Step 7 — Diagnostic, learning, and outcome visualizations
- [x] Step 8 — Multi-seed helper, confidence intervals, and shared curves
- [x] Step 9 — PPO vs DQN comparison
- [x] Step 10 — Reward ablation study
- [x] Step 11 — Curriculum learning with ghost-count promotion
- [x] Step 12 — One-command experiment runner
- [x] Step 13 — Extension documentation and demo guide

Final verification: `python -m pytest tests\ -q` — 494 passed.

---

# DONE

## Phase 1 — Repository Setup

- [x] Repository initialized
- [x] Folder structure established
- [x] AI documentation created
- [x] Copilot instruction system prepared

## Phase 3 — Ghost AI

- [x] Implement Ghost dataclass (`src/ghost.py`)
- [x] GhostState enum (NORMAL, VULNERABLE)
- [x] Random movement respecting maze walls
- [x] Chase behaviour (configurable probability and radius)
- [x] Vulnerable state (always random)
- [x] Ghost respawn (teleport to spawn, restore NORMAL)
- [x] Deterministic execution via caller-supplied RNG
- [x] 48 tests passing

## Phase 2 — Maze Representation

- [x] Create maze representation (`src/maze.py`)
- [x] Implement configurable maze loader (optional layout parameter)
- [x] Create constants module (`src/config.py`)
- [x] Walls defined
- [x] Pellets placed (67 regular, 2 power pellets)
- [x] Spawn points implemented (Pac-Man: row 10 col 5; Ghosts: row 5 cols 5–6)
- [x] Observation grids (wall, pellet, power-pellet) implemented
- [x] Layout validation with meaningful errors
- [x] 79 tests passing

- [x] Repository structure created
- [x] AI knowledge base initialized
- [x] Initial documentation completed

---

# Blocked

None

---

# Notes

Whenever a task is completed:

1. Move it to DONE.
2. Update PROGRESS.md.
3. Record architectural changes if necessary.
4. Update related documentation.