# Architectural & Design Decisions

## Purpose

This document records important architectural, reinforcement learning, and software engineering decisions made throughout the project.

The objective is to provide historical context for future development and prevent previously resolved decisions from being repeatedly reconsidered.

Whenever a significant decision is made, create a new entry.

Each entry should include:

- Decision
- Date
- Context
- Alternatives Considered
- Rationale
- Consequences
- Status

Possible statuses:

- Proposed
- Accepted
- Deprecated
- Replaced

---

# Decision Log

---

## Decision 001

### Title

Use PPO as the Primary Reinforcement Learning Algorithm

### Date

YYYY-MM-DD

### Status

Accepted

### Context

The assignment requires training Pac-Man using Stable-Baselines3.

Both PPO and DQN are acceptable.

### Alternatives Considered

- PPO
- DQN

### Decision

Use PPO as the primary algorithm.

### Rationale

PPO generally provides more stable policy optimization and works well with discrete navigation problems while requiring relatively little algorithm customization.

DQN may later be implemented as an optional comparison.

### Consequences

Training scripts should be designed so that additional algorithms can be added later without modifying the environment.

---

## Decision 002

### Title

Ghosts Remain Rule-Based

### Date

YYYY-MM-DD

### Status

Accepted

### Context

Only Pac-Man should learn.

### Alternatives Considered

- Reinforcement learning ghosts
- Rule-based ghosts

### Decision

Ghosts remain deterministic or probabilistic rule-based agents.

### Rationale

This satisfies the assignment requirements while keeping the environment computationally manageable.

---

## Decision 003

### Title

Separate Rendering from Environment Logic

### Date

YYYY-MM-DD

### Status

Accepted

### Context

Rendering should never interfere with training speed.

### Decision

Rendering is handled by renderer.py.

Environment logic remains independent.

### Consequences

Training can run without opening a graphical window.

Evaluation can optionally enable rendering.

---

## Decision 004

### Title

Reward Values Are Configurable

### Status

Accepted

### Context

Reward shaping often changes during experimentation.

### Decision

Reward values will not be hardcoded throughout the codebase.

Instead, they should be centralized.

### Consequences

Reward tuning becomes significantly easier.

---

## Decision 005

### Title

Use Flat Vector Observation with MlpPolicy

### Date

2026-07-30

### Status

Accepted

### Context

RL_DESIGN.md initially described a multi-channel grid representation compatible with CNNs. This required CnnPolicy and imposed constraints on the observation space format (uint8, channel-last). For a 10x10–12x12 maze, convolutional feature extraction offers no meaningful advantage over a flat vector.

### Alternatives Considered

- Multi-channel grid (H, W, C) uint8 observation with CnnPolicy
- Flat float32 vector observation with MlpPolicy

### Decision

Use a flat float32 NumPy vector as the observation space with MlpPolicy.

### Rationale

- MlpPolicy is simpler to implement and faster to train at small grid sizes.
- Avoids CnnPolicy's strict requirement for uint8 (H, W, C) observations.
- Keeps environment.py portable and easy to validate.
- If CNN results prove significantly better, this decision can be revisited.

### Consequences

Observation space will be `gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)` where N is the flattened multi-channel grid size.

---

## Decision 006

### Title

Use Alternating PPO for Optional Learned-Ghost Experiments

### Date

2026-07-30

### Status

Accepted

### Context

The extension study compares a learning ghost with a learning Pac-Man while
using Stable-Baselines3, whose standard PPO interface is single-agent.

### Alternatives Considered

- Simultaneous two-policy learning in one environment
- Alternating training with one side frozen
- Rule-based ghosts only

### Decision

Keep the baseline rule-based. For explicitly selected experiments, alternate
ghost PPO training against frozen Pac-Man with Pac-Man PPO training against a
frozen ghost policy; sample prior opponents from a pool.

### Rationale

Each training phase remains a conventional single-agent SB3 problem and
avoids changing both policies simultaneously. The opponent pool reduces
overfitting to one checkpoint.

### Consequences

Use `src/ghost_environment.py` and `src/train_alternating.py`. Expect
non-monotonic metrics and possible initial drops in Pac-Man success.

---

## Decision 007

### Title

Keep Observation Extensions Opt-In and Sidecar-Tracked

### Date

2026-07-30

### Status

Accepted

### Context

Direction, vulnerability, and previous-action features may improve
decision-making but change the policy input size and make old model archives
incompatible.

### Alternatives Considered

- Change the default observation for every model
- Append features only under explicit configuration

### Decision

Keep the 723-value baseline unchanged. `ObservationConfig` enables extra
features and is saved alongside newly trained models.

### Rationale

Existing models remain loadable and model/environment observation shape
mismatches are avoided.

### Consequences

Evaluate optional-observation models using their metadata sidecar; outputs
belong under experiment directories.

---

## Decision 008

### Title

Compare DQN Without Replacing PPO as the Primary Algorithm

### Date

2026-07-30

### Status

Accepted

### Context

The project needs an empirical PPO/DQN comparison while retaining the
assignment's PPO baseline.

### Alternatives Considered

- Replace PPO with DQN
- Compare both algorithms under matched CPU, seed, step, and evaluation settings

### Decision

Add DQN as an optional comparison using SB3 and retain PPO as the primary
workflow.

### Rationale

Matched evaluation tests Decision 001 without changing the default training
algorithm.

### Consequences

Report across-seed variability and scale-independent success, pellet, and
survival metrics in addition to reward.

---

## Decision 009

### Title

Change One Reward Term at a Time in Ablation Studies

### Date

2026-07-30

### Status

Accepted

### Context

Reward shaping influences behavior, and multiple simultaneous changes would
make results difficult to interpret.

### Alternatives Considered

- Tune many reward terms together
- Compare one explicit reward term variation at a time

### Decision

Keep the baseline reward values and define ablation variants centrally in
`src/config.py`; pass each selected `RewardConfig` consistently through
training and evaluation.

### Rationale

Controlled variants support interpretable comparisons and preserve the
baseline.

### Consequences

Reward totals across variants are not directly comparable; also report
success, pellet collection, and survival steps.

---

## Decision 010

### Title

Use Matched Multi-Seed Reporting for Experiments

### Date

2026-07-30

### Status

Accepted

### Context

One random seed can make a small RL experiment appear better or worse by
chance.

### Alternatives Considered

- Report one representative seed
- Run matched experiments over a configurable seed set

### Decision

Use a shared default of three seeds and report mean, sample standard
deviation, min/max, and a confidence interval. Log Monitor data separately
for each seed.

### Rationale

This exposes run-to-run variability and allows learning curves to be
aggregated consistently across experiments.

### Consequences

Full experiment budgets are substantially longer; quick mode is for workflow
verification only.

---

## Decision 011

### Title

Provide an Isolated Medium CPU Experiment Profile

### Date

2026-10-16

### Status

Accepted

### Context

The full experiment workflow takes too long for routine runs, while quick
mode is only a smoke test. Experiments may also be launched simultaneously
from separate terminals.

### Alternatives Considered

- Reduce the full-mode budgets globally
- Add a bounded medium profile with separate outputs

### Decision

Keep full and quick settings unchanged. Add a two-seed medium profile for
alternating training, curriculum, PPO-vs-DQN, and a four-variant ablation.
Use separate medium model, log, result, and plot paths, print a CPU-time
estimate before training, and limit PyTorch to one thread in each entry point.
The runner's medium default selects only these four training workflows.

### Rationale

This provides a repeatable intermediate budget without changing prior
experiments, and it allows different experiment types to run concurrently
without sharing output files.

### Consequences

Medium runs are separate from full/quick artifacts and do not train into or
replace `models/best_model.zip`. Estimated times are approximate and exclude
evaluation overhead.

---

## Decision 012

### Title

Keep Endgame Navigation Features and Shaping Opt-In

### Date

2026-10-19

### Status

Accepted

### Context

Pac-Man can oscillate after nearby pellets are exhausted. Nearest-pellet
direction/distance features and small progress/revisit rewards are being
evaluated as isolated experiment variants.

### Alternatives Considered

- Change the default observation and reward for every model
- Add opt-in features and shaping while preserving existing model inputs

### Decision

Keep all added observation flags and shaping values disabled by default.
Train and evaluate the feature-only, feature-plus-shaping, and shaping-only
variants in isolated output directories; keep baseline model files untouched.

### Rationale

Separate experiment arms make it possible to distinguish observation effects
from reward effects without breaking existing model compatibility.

### Consequences

Variant models must carry their observation metadata and be evaluated with
matching settings. The endgame loop-rate result will determine whether the
approach addresses the reported oscillation.

---

## Decision 013

### Title

Keep Nearest-Ghost Awareness as a Separate Endgame Variant

### Date

2026-10-19

### Status

Accepted

### Context

Pellet-directed endgame signals do not explicitly tell the policy the
topological escape direction or movement trend of a nearby active ghost.

### Alternatives Considered

- Add ghost awareness to all feature variants
- Isolate ghost awareness and its proximity penalty in a dedicated arm

### Decision

Add a disabled-by-default `include_nearest_ghost` feature and a
zero-default ghost-distance reward term. Enable both only in
`features_shaping_ghost`.

### Rationale

This permits a controlled evaluation of hostile-ghost awareness without
changing the established feature/shaping variants or baseline model inputs.

### Consequences

The feature arm appends nine values: a BFS escape direction, normalized
distance to the nearest reachable non-vulnerable ghost, and that ghost's last
move direction. Its reward term applies -0.5 only when Pac-Man's move reduces
distance while the nearest hostile ghost is within three steps.

---

## Decision 014

### Title

Match Learned-Ghost Evaluation to Shared Training

### Date

2026-10-19

### Status

Accepted

### Context

Final learned ghosts could repeatedly select a direction blocked by a wall.
Evaluation also applied the shared PPO model to every ghost each step, while
shared training controls one ghost per step and leaves the others on the
default rule policy.

### Decision

Use a random legal-neighbor fallback for wall-directed learned actions in
both ghost training and inference. During evaluation, rotate learned control
across one ghost per environment step and select its action from the
pre-Pac-Man-move observation, matching the shared training schedule and timing.

### Rationale

This prevents a blocked model action from leaving a ghost stationary and
makes final catch-rate evaluations follow the trained opponent protocol.

### Consequences

Learned-ghost evaluation is now intentionally different from assigning the
learned policy to every ghost simultaneously. Existing saved ghost models
remain unchanged and should be judged using this shared-policy evaluation.

---

## Future Decisions

Record future decisions regarding:

- Observation encoding
- Hyperparameter selection
- PPO configuration
- Ghost behavior changes
- Reward shaping experiments
- Performance optimizations
- Refactoring
- New project features

---

# Decision Guidelines

Only record decisions that have long-term impact.

Do not record temporary implementation details.

Examples of good decisions:

✓ PPO vs DQN

✓ Observation representation

✓ Reward philosophy

✓ Module ownership

✓ Rendering architecture

Examples of poor decisions:

✗ Variable names

✗ Temporary debugging changes

✗ Minor code formatting