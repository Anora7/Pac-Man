# Optional Experiment Extensions

The original Pac-Man setup remains the default: two rule-based ghosts, the
723-value observation, the baseline reward table, PPO, and CPU execution.
Extension models and logs are saved separately so they do not replace the
baseline checkpoints.

Run commands from the repository root in PowerShell, with project
dependencies installed. `--quick` is a smoke test, not a learning result.
The full workflows can take hours on CPU; estimates are printed by the
scripts.

## One-command runner

The orchestration entry point is useful for demonstrations and end-to-end
verification.

```powershell
python .\run_all_experiments.py --quick
python .\run_all_experiments.py --medium
python .\run_all_experiments.py --full
```

Medium mode runs alternating, curriculum, PPO-vs-DQN, and reward ablation
with two seeds. Its estimates target roughly one to two CPU hours per
experiment (evaluation adds overhead). The medium outputs are isolated under
`models\experiments\medium\`, `logs\experiments\medium\`,
`results\medium\`, and `plots\experiments\medium\`; they do not replace full
or quick-run artifacts or the baseline checkpoints.

Evaluate the baseline against Pac-Man checkpoints from only the medium
experiment tree: final per-round alternating checkpoints (including round 00)
and final curriculum checkpoints, excluding callback and timestep archives.
The learned-ghost column uses the newest medium alternating ghost checkpoint.
The default is 10 episodes per model/ghost cell; override it with
`--episodes`:

```powershell
python .\src\evaluate_matrix.py --medium --episodes 10
```

The matrix is written to `results\medium\robustness_matrix.csv` and
`plots\experiments\medium\robustness_matrix.png`.

Generate diagnostic plots comparing the baseline against the medium PPO
checkpoint, with learning curves from both medium PPO Monitor logs:

```powershell
python .\src\analysis_plots.py --medium --episodes 20
```

The plots are saved under `plots\experiments\medium\`, with model-specific
diagnostics and per-seed training-curve subdirectories.

Run selected workflows by repeating `--only`:

```powershell
python .\run_all_experiments.py --quick --only dqn --only plots
python .\run_all_experiments.py --medium --only alternating
python .\run_all_experiments.py --medium --only ablation
```

It prints the workflows that ran or were skipped and lists output paths. The
quick command executes the experiments in dependency order and generates
their tables and plots. Medium mode defaults to the four training workflows;
selecting one with `--only` makes it convenient to run each in a separate
terminal without sharing its model, log, result, or plot files.

## Learned ghosts and alternating PPO

**Idea.** Standard SB3 algorithms are single-agent. Rather than train both
sides simultaneously against moving opponents, alternate: freeze Pac-Man
while training a shared ghost PPO, then freeze the ghost policy while
continuing Pac-Man PPO. This makes each training phase a more stable
single-agent problem, though each agent still faces a changing opponent
between rounds.

**Code.** `src/ghost_policies.py` defines random, chase, rule-based, and
learned policies; `src/ghost_environment.py` provides the ghost-side
Gymnasium interface; `src/train_alternating.py` manages training and an
opponent pool.

```powershell
python .\src\train_alternating.py --quick
python .\src\train_alternating.py --medium
python .\src\train_alternating.py --full
```

Medium mode uses two seeds and three rounds. Round 00 copies
`models\best_model.zip` into each seed's medium checkpoint directory; each
Pac-Man round continues from its prior model for 30,000 steps, and each ghost
round trains for 50,000 steps. The opponent pool remains enabled. Reported
training/evaluation metrics use 30 episodes with distinct episode seeds.
After training, the final learned ghost for each seed is compared against
baseline Pac-Man over 30 episodes and reported alongside the rule-ghost catch
rate.

**Outputs.** Round checkpoints are in
`models\experiments\alternating\seed_<seed>\`; Monitor logs are under
`logs\experiments\alternating\`; round evaluations are in
`results\alternating_rounds.csv`.

**Limitations.** A ghost can overfit to a single Pac-Man snapshot; the
configurable opponent pool mixes the newest and earlier checkpoints to reduce
that risk. A newly learned ghost may make the task harder, so Pac-Man success
can drop initially. The co-evolution plot is descriptive, not a guarantee of
monotonic improvement.

## Improved observation

**Idea.** Add ghost movement direction and vulnerability flags, and/or the
previous Pac-Man action, to the baseline vector. Both options are off by
default. The size formula is
`H × W × 5 + 3 + (ghost count × 5 if ghost direction is enabled) + (4 if the
previous action is enabled)`. With the default 12×12 maze and two ghosts,
both options produce 737 values; the baseline remains 723.

**Code.** `ObservationConfig` is in `src/config.py`;
`src/environment.py` encodes the extra features; `src/train.py` accepts the
same observation setting through `TrainingConfig` and saves it beside model
archives so evaluation uses the matching shape.

Quick (32 steps, isolated model/log paths):

```powershell
python -c "from pathlib import Path; from config import ObservationConfig; from train import TrainingConfig, train; train(TrainingConfig(models_dir=Path('models/experiments/improved_observation/quick'), logs_dir=Path('logs/experiments/improved_observation/quick'), tensorboard_dir=Path('logs/experiments/improved_observation/quick/tensorboard'), total_timesteps=32, n_steps=32, batch_size=16, n_epochs=1, eval_freq=16, n_eval_episodes=1, checkpoint_freq=32, max_steps=32, seed=42, verbose=0, observation_config=ObservationConfig(include_ghost_direction=True)))"
```

Full (uses the normal 500,000 PPO steps):

```powershell
python -c "from pathlib import Path; from config import ObservationConfig; from train import TrainingConfig, train; train(TrainingConfig(models_dir=Path('models/experiments/improved_observation/full'), logs_dir=Path('logs/experiments/improved_observation/full'), tensorboard_dir=Path('logs/experiments/improved_observation/full/tensorboard'), seed=42, observation_config=ObservationConfig(include_ghost_direction=True)))"
```

To evaluate a saved enhanced-observation model, its sidecar is read
automatically:

```powershell
python .\src\evaluate.py --model .\models\experiments\improved_observation\quick\final_model.zip --episodes 1 --seed 42
```

**Outputs.** Model archives and `.observation.json` files are saved under
`models\experiments\improved_observation\`; Monitor and TensorBoard output is
under the matching `logs\experiments\improved_observation\` directory.

## Curriculum learning

**Idea.** Train against one ghost first, then promote to two when the rolling
success rate reaches a threshold or the stage step limit is reached.

**Code.** `src/train_curriculum.py` reuses the configurable
`PacmanEnvironment`, a continuing PPO model, Monitor success events, and the
shared seed runner.

```powershell
python .\src\train_curriculum.py --quick
python .\src\train_curriculum.py --medium
python .\src\train_curriculum.py --full
```

Medium mode uses two seeds and up to 100,000 steps for each ghost-count
stage. Its outputs are isolated under the `medium` experiment directories.

**Outputs.** Per-stage checkpoints and logs are under
`models\experiments\curriculum\` and `logs\experiments\curriculum\`.
Promotions are recorded in `results\curriculum_promotions.csv`; reward versus
timestep, with vertical promotion markers, is
`plots\experiments\curriculum\curriculum_progress.png`.

## PPO versus DQN

**Idea.** Compare PPO with SB3 DQN on the same environment, observation,
reward, CPU device, training steps, seeds, and test episodes. DQN replay,
exploration, and target-network parameters are in `DQNConfig`.

**Code.** `src/train_dqn.py` owns DQN training;
`src/compare_algorithms.py` runs the matched multi-seed comparison and
generates the Decision 001 report.

```powershell
python .\src\compare_algorithms.py --quick
python .\src\compare_algorithms.py --medium
python .\src\compare_algorithms.py --full
```

Medium mode compares PPO and DQN at 100,000 steps per algorithm for each of
two matched seeds and evaluates each saved model over 30 episodes with
distinct episode seeds. To update its CSV and plots without retraining:

```powershell
python .\src\compare_algorithms.py --eval-only
```

Its models, logs, reports, CSV, and plots use separate `medium` output
directories.

**Outputs.** Models and Monitor logs are under
`models\experiments\algorithm_comparison\` and
`logs\experiments\algorithm_comparison\`. The per-seed and aggregate table
is `results\algorithm_comparison.csv`; the short discussion is
`results\algorithm_comparison_report.md`; mean curves and error-bar charts
are under `plots\experiments\algorithm_comparison\`.

## Reward ablation

**Idea.** Change one reward term at a time while keeping the other baseline
values, seeds, and training settings fixed. The ten variants cover baseline,
caught penalty, step penalty, wall penalty, pellet reward, and maze-clear
bonus.

**Code.** The table is `REWARD_ABLATION_VARIANTS` in `src/config.py`.
`src/ablation.py` passes the same `RewardConfig` variant to training and
evaluation.

```powershell
python .\src\ablation.py --quick
python .\src\ablation.py --medium
python .\src\ablation.py --full
```

Medium mode trains the baseline plus three focused reward variants
(`caught_penalty_50`, `step_penalty_0`, and `pellet_reward_20`) for 100,000
steps each across two seeds and evaluates each saved model over 30 episodes
with distinct episode seeds. Re-evaluate the existing models without
retraining:

```powershell
python .\src\ablation.py --eval-only
```

Its artifacts are stored separately under `medium` output directories; full
mode retains the complete variant set.

**Outputs.** Models and Monitor logs are under
`models\experiments\ablation\` and `logs\experiments\ablation\`.
`results\ablation_results.csv` contains per-seed and aggregate metrics;
`results\ablation_summary.md` highlights the largest measured
scale-independent change; `plots\experiments\ablation\reward_ablation.png`
shows grouped metrics. Reward totals are not directly comparable across
different reward scales, so also inspect success rate, pellets, and survival
steps.

## Multi-seed protocol

**Idea.** Repeated seeds reveal variability that a single run hides. The
shared helper reports mean, sample standard deviation, min/max, and a
confidence interval; learning curves align by episode and show the mean with
a shaded standard-deviation band. Full defaults use three seeds.

**Code.** `src/experiment_utils.py` provides `run_across_seeds`,
`aggregate_metrics`, and Monitor curve plotting. It is used by alternating,
curriculum, PPO-vs-DQN, and ablation workflows. Each seed writes its own
Monitor CSV, including `is_success`.

Quick and full examples:

```powershell
python .\src\compare_algorithms.py --quick
python .\src\compare_algorithms.py --full
python .\src\ablation.py --quick
python .\src\ablation.py --full
```

**Outputs.** Seed-specific Monitor files are under each workflow's
`logs\experiments\<workflow>\...` directory. Aggregate confidence intervals
are included in comparison and ablation CSVs. Learning curves are saved
under `plots\experiments\algorithm_comparison\`.

## Ghost-type robustness matrix

**Idea.** Measure each available Pac-Man model against random, chase-only,
rule-based, personality, and latest learned ghost policies. Models with
different observation settings are evaluated using their observation
sidecars.

**Code.** `src/evaluate_matrix.py` discovers available models, skips missing
models with an explicit message, writes long-form measurements, and renders
annotated heatmaps.

```powershell
python .\src\evaluate_matrix.py --quick
python .\src\evaluate_matrix.py --full
```

**Outputs.** `results\robustness_matrix.csv` and
`plots\experiments\robustness_matrix.png`.

## Analysis visualizations

**Idea.** Diagnostic plots make behavior and training dynamics easier to
inspect, not just aggregate reward.

**Code.** `src/analysis_plots.py` contains the plotting functions; the
runner's plots workflow supplies quick or full episode bounds.

```powershell
python .\run_all_experiments.py --quick --only plots
python .\run_all_experiments.py --full --only plots
```

**Outputs.** Files are saved under `plots\experiments\`:

- `pacman_visitation_<ghost-type>.png`, `ghost_positions_<ghost-type>.png`,
  and `caught_locations_<ghost-type>.png`.
- `value_danger_ghost_nearby.png`, `value_danger_ghost_far.png`, and
  `value_danger_power_mode_on.png`.
- Existing reward, moving-average, episode-length, and success charts, plus
  `learning_curves_smoothed.png`.
- `alternating_dashboard.png` when alternating results exist and
  `episode_outcomes.png`.

PNG gameplay frame capture is an optional helper and is not required by the
runner; no extra GIF dependency is introduced.

### Results interpretation template

For a report or lab notebook, record:

| Item | What to report |
|---|---|
| Experiment/configuration | Workflow, reward/observation settings, ghost policy, and training steps |
| Seeds | Seed IDs, number of seeds, and whether the run was quick or full |
| Learning curves | Mean trend and shaded ±1 standard deviation; do not infer convergence from quick runs |
| Final performance | Mean reward with variability/CI, success rate, pellets, and survival steps |
| Robustness | Which model/ghost combinations preserve success and pellet collection |
| Reward ablation | Changed term and scale-independent outcome differences from baseline |
| Limitation | Small sample count, short training, or opponent mismatch |

**Value-function danger maps.** Colors show the PPO critic's predicted state
value for a constructed state, not a probability of death. Compare relative
values between the nearby-ghost, far-ghost, and power-mode panels: lower value
means the critic expects lower future return in that state. The map is only as
reliable as the policy's training distribution and critic calibration.

**Ghost/Pac-Man co-evolution.** The dashboard uses separate y-axes for
Pac-Man maze-clear success and ghost catch rate across alternating rounds.
Opposing trends can indicate competitive adaptation; both can fluctuate, and
one quick seed cannot establish a general learning trend.

**Visitation and outcomes.** Visitation maps count occupied cells and caught
locations; they are conditional on the tested policy, ghost type, and episode
seeds. Stacked outcome bars distinguish caught, cleared, and timeout episodes.

**Algorithm and robustness tables.** Compare per-seed rows before aggregate
rows. Confidence intervals summarize across-seed uncertainty; reward means
from different reward scales should not be compared as if they shared one
objective.

## Five-minute lab demo cheat sheet

1. **0:00–0:40 — Baseline.** Show the maze, 723-value observation, and the
   unchanged baseline reward and ghost defaults.
2. **0:40–1:30 — Quick pipeline.** Run
   `python .\run_all_experiments.py --quick`; point out that the run is a
   smoke test and show the output summary.
3. **1:30–2:30 — Alternating learning.** Open
   `plots\experiments\alternating_dashboard.png` and explain that each
   learner faces a frozen opponent during its phase.
4. **2:30–3:20 — Critic map.** Compare nearby-ghost and far-ghost value maps;
   emphasize relative expected return, not death probability.
5. **3:20–4:10 — Controlled comparison.** Show PPO/DQN or reward-ablation
   results with per-seed variation and scale-independent metrics.
6. **4:10–5:00 — Robustness and questions.** Show the ghost-type matrix and
   summarize what further full-length, multi-seed runs would establish.

### Likely questions

- **Why alternating training?** SB3 PPO is single-agent; freezing the
  opponent avoids simultaneous policy changes during each update phase.
- **What is non-stationarity?** The learning agent's effective environment
  changes as its opponent's policy changes.
- **What does the value map show?** The critic's expected return for
  constructed states; lower values indicate less favorable predicted futures.
- **Why use multiple seeds?** Outcomes depend on initialization and sampled
  trajectories; several seeds expose run-to-run variability.
- **What did the reward ablation reveal?** Use the report's measured,
  seed-conditioned success/pellet/survival differences; quick mode alone
  cannot establish a reliable conclusion.
