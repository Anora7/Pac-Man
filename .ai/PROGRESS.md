# Project Progress

## Purpose

This document provides a concise summary of the project's current implementation status.

It should allow any developer or AI assistant to understand the repository state without inspecting the source code.

Update this document after every completed implementation phase.

---

# Current Status

## Repository

Status: Complete

Documentation: Complete

Architecture: Finalized

Implementation: Baseline and all 13 opt-in extension steps complete

---

# Current Phase

**Complete** — baseline implementation, optional experiment extensions, and
documentation have been delivered.

---

# Overall Completion

```text
████████████████████ 100% baseline and optional extensions
```

> Progress percentage should be updated after each completed phase.

---

# Completed

## Project Initialization

- Repository created
- Folder structure established
- AI documentation created
- Copilot instruction system prepared

---

## Documentation

Completed: All documentation files populated.

---

## Phase 9 — Documentation & Presentation

Completed: 2026-07-30

Files updated:

- `README.md` — complete project README with setup, usage, maze diagram, API overview
- `docs/environment.md` — observation/action space, episode lifecycle, info dict
- `docs/training.md` — TrainingConfig reference, output files, TensorBoard
- `docs/evaluation.md` — CLI usage, Python API, sample output
- `docs/api_reference.md` — all modules, classes, and key functions
- `docs/presentation.md` — slide structure, demo commands, key numbers
- `.ai/DEMO_CHECKLIST.md` — all items checked

---

## Phase 8 — Graph Generation

Completed: 2026-07-30

Files created:

- `src/plots.py` — `PlotConfig`, `plot_reward_curve`, `plot_moving_average`, `plot_episode_length`, `plot_success_rate`, `generate_all_plots`
- `tests/test_plots.py` — 41 tests, all passing

Outputs: `reward_curve.png`, `moving_average.png`, `episode_length.png`, `success_rate.png`. Reads SB3 Monitor CSV for training curves; reads `EvaluationResult` for outcome chart. All saved to configurable `plots/` directory. Uses Agg backend (headless). Short-series guard: window clamped to series length.

---

## Phase 7 — Evaluation

Completed: 2026-07-30

Files created:

- `src/evaluate.py` — `EvaluationConfig`, `EpisodeResult`, `EvaluationResult`, `evaluate()`
- `tests/test_evaluate.py` — 40 tests, all passing

Accepts pre-loaded model or loads from `.zip` path. Tracks per-episode: total reward, steps, pellets collected, cleared/caught/truncated outcome, completion fraction. Aggregates to mean/std reward, success rate, average completion. CLI entry point via `python src/evaluate.py`.

---

## Phase 6 — PPO Training

Completed: 2026-07-30

Files created:

- `src/train.py` — `TrainingConfig` dataclass + `train()` function
- `tests/test_train.py` — 24 tests (config + smoke), all passing

Uses `make_vec_env`, `EvalCallback` (saves `best_model.zip`), `CheckpointCallback` (periodic saves), TensorBoard logging. All hyperparameters configurable via `TrainingConfig`. Device auto-selected (`device="auto"`).

---

## Phase 4 — Gymnasium Environment

Completed: 2026-07-30

Files created:

- `src/rewards.py` — `calculate_step_reward()` function
- `src/environment.py` — `PacmanEnvironment(gymnasium.Env)` class
- `tests/test_environment.py` — 58 tests, all passing
- `src/config.py` updated: added `MAX_STEPS = 1000`

Observation space: 723-element float32 vector (5 channels × 144 + 3 scalars). Dual collision detection (pre- and post-ghost-move). Power mode timer skips decrement on activation step. SB3 `check_env` passes.

---

## Phase 5 — Renderer

Completed: 2026-07-30

Files created:

- `src/renderer.py` — `RenderState` frozen dataclass + `PacmanRenderer` class
- `tests/test_renderer.py` — 37 tests, all passing

Pygame initialised lazily on first `render()` call. Tests use `SDL_VIDEODRIVER=dummy` for headless execution. Renderer is fully independent of training logic; accepts `RenderState` snapshot each frame.

---

## Phase 3 — Ghost AI

Completed: 2026-07-30

Files created:

- `src/ghost.py` — GhostState enum and Ghost dataclass
- `tests/test_ghost.py` — 48 tests, all passing

Ghost policy: 70% random movement, 30% chase (Manhattan distance, configurable radius). Vulnerable state moves purely randomly. Respawn teleports to spawn position with immediate state restore.

---

## Phase 2 — Maze Representation

Completed: 2026-07-30

Files created:

- `src/config.py` — All project constants (CellType, Action, rewards, hyperparameters, colors, maze layout)
- `src/maze.py` — Position NamedTuple and Maze class
- `src/__init__.py` — Package marker
- `conftest.py` — pytest sys.path configuration
- `tests/test_maze.py` — 79 tests, all passing

Maze layout: 12×12 static grid with 67 regular pellets, 2 power pellets, 2 ghost spawns, 1 Pac-Man spawn.

---

# Current Focus

All 13 steps of the opt-in extension plan are complete. The baseline
723-element observation, original reward values, and rule-based ghost policy
remain the defaults. The high-budget demo refresh is complete: ghost-direction
PPO and PPO/DQN comparison runs finished; four-round alternating runs finished
for seeds 42, 43, and 44; the final-only matrix and priority dashboard were
regenerated. That refresh passed 540 tests; after endgame matrix and
diagnostic-plot coverage was added, the full suite passes 542 tests.

The medium-results refresh is complete. PPO-vs-DQN and reward ablation were
re-evaluated from saved models over 30 episodes per model with distinct episode
seeds. Alternating training ran three rounds for seeds 42 and 43, copying the
baseline for round zero and using 30,000 Pac-Man plus 50,000 ghost steps per
round. Its final learned ghosts caught baseline Pac-Man in 83.3% and 16.7% of
30 episodes respectively; the rule ghost catch rate was 100%. The robustness
matrix contains 11 Pac-Man models across five policies (55 cells, 10 episodes
per cell), and medium analysis plots were regenerated.

---

# Upcoming Milestones

The latest implementation adds opt-in nearest-ghost awareness and a
close-range distance-decrease penalty to the new `features_shaping_ghost`
endgame variant. Saved endgame results have now been compared without
retraining: `features_shaping` averaged 52.93 pellets and 11.67 clears per
30-game seed run; `features_shaping_ghost` averaged 52.60 pellets and 11.33
clears. The demo model was evaluated in a curated 15-cell matrix (three
Pac-Man models × five ghost policies, 30 games per cell), and its visitation,
caught-location, and critic-value maps were generated. Baseline model hash
remains unchanged. The learned-ghost column used an incorrect wall-stalling
and all-ghost PPO protocol. Evaluation now matches shared-training turn order
and pre-step observation timing, with legal-move fallback. Corrected
30-episode catch rates were 0.0%, 23.3%, and 66.7% for learned seeds 42, 43,
and 44 (rule-based: 56.7%). The earlier matrix learned column is stale and
should be regenerated before use.

---

# Risks

Current known risks:

- Reward shaping may require tuning.
- PPO hyperparameters may need experimentation.
- Ghost difficulty must be balanced.
- Rendering performance should remain independent of training.

---

# Recently Completed

| Date | Task | Status |
|------|------|--------|
| 2026-10-18 | Train ghost-direction PPO for 300k steps and evaluate 30 episodes per model | ✅ |
| 2026-10-18 | Train PPO/DQN for 300k steps across seeds 42, 43, and 44 | ✅ |
| 2026-10-19 | Run four-round 100k-per-side alternating experiments for seeds 42–44 | ✅ |
| 2026-10-19 | Evaluate 19 final models against five policies for 30 episodes per cell | ✅ |
| 2026-10-19 | Regenerate priority dashboard and verify baseline hash/full test suite | ✅ |
| 2026-10-19 | Add opt-in endgame navigation features, shaping, runner, and tests | ✅ |
| 2026-10-19 | Add nearest-hostile-ghost awareness and proximity shaping variant | ✅ |
| 2026-10-20 | Evaluate selected endgame models and generate comparison/diagnostic plots | ✅ |
| 2026-10-19 | Slow default Pygame playback to 4 FPS for easier viewing | ✅ |
| 2026-10-19 | Fix learned-ghost evaluation wall stalls and align it to shared training | ✅ |
| 2026-10-18 | Refresh medium evaluations, alternating training, robustness matrix, and analysis plots | ✅ |
| 2026-10-18 | Restrict robustness matrix to final medium checkpoints and add progress output | ✅ |
| 2026-10-16 | Add isolated medium CPU budgets for experiment workflows | ✅ |
| YYYY-MM-DD | Repository initialization | ✅ |
| YYYY-MM-DD | Documentation framework | ✅ |

---

# Next Immediate Task

Use the 1M-step features-shaping model for the demonstration; preserve the
existing baseline and avoid retraining unless a new experiment is requested.

---

# Notes for Future Development

Before implementing any new feature:

1. Read PROJECT.md.
2. Read ARCHITECTURE.md.
3. Read RL_DESIGN.md.
4. Review TASK_BOARD.md.
5. Confirm current phase.
6. Implement only the current task.
7. Update documentation after completion.

Do not skip phases unless explicitly instructed.