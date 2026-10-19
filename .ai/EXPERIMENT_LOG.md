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

## Priority 300k Refresh — Ghost-Direction PPO

Completed 2026-10-18. A PPO model was trained for 300,000 steps with
`ObservationConfig(include_ghost_direction=True)` (733 observation values),
seed 42, and the existing CPU hyperparameters. The final model is isolated
under `models/experiments/ghost_direction_ppo_300k/`. Baseline evaluation
used episode seeds 42,000–42,029; the trained model used 42,030–42,059.
Each model was evaluated against rule-based ghosts for 30 episodes.

| Model | Mean pellets (95% CI) | Survival steps (95% CI) | Success |
|---|---:|---:|---:|
| Baseline | 44.67 [40.65, 48.68] | 82.9 [58.42, 107.38] | 0% |
| Ghost-direction PPO | 32.03 [28.47, 35.59] | 56.0 [44.31, 67.69] | 0% |

No maze clears occurred in either evaluation set. These measurements do not
show improvement over baseline; the new observation variant remains an
experimental model rather than a replacement.

## Priority 300k Refresh — PPO vs DQN

Completed 2026-10-18. PPO and DQN were each trained for 300,000 steps at
seeds 42, 43, and 44. Each algorithm/seed pair ran in its own process, with
at most three workers active. Each trained model was evaluated for 30
episodes with distinct reproducible episode seeds. Confidence intervals below
are 95% Student-t intervals across the three seed-level means.

| Algorithm | Mean reward (95% CI) | Mean pellets (95% CI) | Survival steps (95% CI) | Success |
|---|---:|---:|---:|---:|
| PPO | 360.74 [300.25, 421.24] | 49.51 [46.80, 52.22] | 74.09 [45.05, 103.13] | 0% |
| DQN | 30.36 [-107.36, 168.07] | 19.69 [17.84, 21.54] | 43.03 [19.57, 66.50] | 0% |

No maze clears occurred in the 180 final comparison episodes. PPO had higher
mean reward, pellet count, and survival in this run; the three-seed sample
remains limited.

## Priority 300k Refresh — Alternating, Robustness, and Final Verification

Completed 2026-10-19. Four 100,000-step Pac-Man/ghost rounds ran for seeds 42,
43, and 44. Every Pac-Man round-start archive was verified byte-for-byte
against `models/best_model.zip`; the baseline archive hash remained
`D04661692A40D81D5B5EA186B6257231D8AC283B2072266D6C2A923E65BC644F`.
The final-round Pac-Man evaluations collected 44, 8, and 8 mean pellets for
seeds 42, 43, and 44 respectively, with 1,000 mean steps and 0% clears for
each seed.

The final learned ghosts were evaluated against baseline Pac-Man for 30
episodes each. Rule-based ghosts caught the baseline in 100% of episodes;
learned ghost catch rates were 16.7% (seed 42), 83.3% (seed 43), and 90.0%
(seed 44).

The final-only priority matrix evaluated 19 uniquely labeled Pac-Man models
against five ghost policies (95 cells, 30 episodes per cell). It covers the
baseline, priority experiment finals, ghost-direction PPO, and medium
experiment finals/curriculum models. The latest priority learned ghost was
seed 42 round 4. No matrix cell cleared the maze. Curriculum seed 43 collected
5 pellets in all five cells; its mean rewards remained negative
(-1,149.8 to -5,925), confirming the CSV writer preserves reward signs rather
than producing the earlier suspected positive-reward artifact.

The matrix now loads PPO and DQN archives with their matching Stable-Baselines3
classes and uses experiment-qualified row labels so priority and medium models
cannot overwrite one another in the heatmap. Priority matrix and alternating
dashboard plots were regenerated. Final validation:
`python -m pytest tests\ -q` — 515 passed.

## Endgame Navigation Variants — Configured

Added an opt-in endgame experiment with three isolated 300,000-step PPO arms:
nearest-pellet direction/distance plus previous-action features; those
features with BFS-progress and recent-cell revisit shaping; and shaping with
the baseline observation. The shaping arm uses +0.5 for a decrease in
nearest-pellet BFS distance, -0.5 for an increase, and -0.1 for revisiting a
cell among the last ten recorded movement positions. Reward defaults remain
zero for each new term.

The runner defaults to seed 42, evaluates the unchanged baseline and trained
model for 30 deterministic and 30 stochastic games per variant, and reports
mean pellets, mean steps, clear count, and loop rate. Loop rate is defined as
the share of episodes with a 50-step window occupying at most three distinct
cells. Use `--variant`, `--steps`, `--seed`, and `--output-tag` to run
isolated worker processes. Every output folder includes the seed and step
count; the optional tag is an additional prefix. Training was not run as part
of this implementation task; no
300,000-step variant run, endgame CSV, or endgame model outputs are claimed
yet. The full regression suite passed (540 tests), and a short opt-in-feature
PPO training/evaluation smoke check passed.

## Endgame Nearest-Ghost Variant — Configured

Added `ObservationConfig.include_nearest_ghost` as a disabled-by-default
nine-value feature: BFS escape direction from the nearest reachable
non-vulnerable ghost, normalized path distance to that ghost, and its last
actual move direction. Added the zero-default
`RewardConfig.GHOST_DISTANCE_DECREASE` term. The new `features_shaping_ghost`
arm combines these additions with all `features_shaping` signals and uses a
-0.5 penalty when Pac-Man moves closer while the nearest hostile ghost is
within three steps.

The new arm accepts the existing `--variant`, `--steps`, `--seed`, and
`--output-tag` options and uses separate outputs. Existing variant CSVs and
trained models were not touched. No training was started for this change.
Full regression validation passed (540 tests).

## Endgame Evaluation and Diagnostic Plots

Evaluated the baseline, the 1M-step seed-42 `features_shaping` demo, and the
300k-step seed-44 `features_shaping` final model against random, chase, rule,
personality, and learned ghosts. The matrix used 30 distinct episode seeds
per cell (15 cells total) and the priority alternating seed-42 round-04 ghost.
Each model loaded its matching observation sidecar. The baseline hash remained
`D04661692A40D81D5B5EA186B6257231D8AC283B2072266D6C2A923E65BC644F`.

The 1M demo cleared 13/30 episodes against rule ghosts, 12/30 against
personality ghosts, and 23/30 against random ghosts; it cleared none against
chase ghosts and cleared all 30 against the selected learned ghost. The
300k seed-44 model cleared 13/30 against rule ghosts, 8/30 against
personality ghosts, 20/30 against random ghosts, 1/30 against chase ghosts,
and all 30 against the selected learned ghost. The learned-ghost results in
this earlier matrix used the evaluation protocol diagnosed below and should
not be treated as final learned-ghost metrics.

The deterministic comparison chart uses the existing per-seed CSVs only.
Across seeds, `features_shaping` averaged 11.67 clears (sample SD 1.53) and
52.93 pellets (SD 0.87); `features_shaping_ghost` averaged 11.33 clears (SD
2.08) and 52.60 pellets (SD 3.62). Error bars show sample standard deviation.
The 1M demo diagnostic set contains Pac-Man visitation, ghost occupancy,
caught-cell frequency, and three critic-value scenarios. No training or
baseline-model modification occurred.

## Learned-Ghost Evaluation Diagnosis

Before correction, three episodes per saved final ghost were run against the
1M-step seed-42 `features_shaping` Pac-Man model, using seeds 42–44. All three
ghost models emitted one constant action at every decision: seed 42 always
chose Down (1), seed 43 always chose Right (3), and seed 44 always chose Left
(2). The evaluation environment allowed wall-directed actions to leave a
ghost in place, resulting in 0/267 or 3/267 actual moves per ghost. Across the
two ghosts, wall actions were 100% for seed 42, 99.4% for seed 43, and 99.4%
for seed 44. Visited-cell counts were one cell per ghost except one ghost
each in seeds 43 and 44, which visited two cells.

The inference observation matched the training observation shape (866
float32 values) and shared the same `build_ghost_observation()` construction;
runtime comparison found no mismatches. There was, however, a policy
evaluation mismatch: training cycles PPO control over one ghost per game
step while other ghosts use the rule policy, whereas evaluation had assigned
PPO to both ghosts on every step.

Fixed both issues without retraining: blocked PPO actions now fall back to a
random legal neighboring move in training and evaluation, and evaluation
cycles the learned policy exactly as the training environment does. A final
timing check also moved learned-action selection to the pre-Pac-Man-move
state, matching the state timing used by `GhostEnvironment`. The feature-demo
Pac-Man evaluation used 30 distinct seeds (42–71) for each opponent. Rule
ghosts caught Pac-Man in 17/30 games (56.7%). The final seed-42, seed-43, and
seed-44 learned ghosts caught Pac-Man in 0/30 (0.0%), 7/30 (23.3%), and
20/30 (66.7%), respectively. In final three-episode motion checks, all six
ghosts moved on 100% of their movement decisions and visited 5–30 distinct
cells. Their raw PPO outputs remained direction-collapsed; 44.4–57.5% of
learned decisions pointed at walls and were diverted to legal fallback
actions. Counts by ghost (index 0 / index 1): seed 42 had 135/132 learned
decisions, 60/135 and 63/132 wall actions, and 5/6 distinct cells; seed 43
had 113/111 decisions, 65/113 and 54/111 wall actions, and 30/20 cells; seed
44 had 107/104 decisions, 60/107 and 58/104 wall actions, and 19/18 cells.
Observation checks found no shape or construction mismatch. No training was
run and the baseline hash was unchanged.

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