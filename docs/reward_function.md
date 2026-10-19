# Reward Function

## Overview

The reward function shapes Pac-Man's learning signal. All reward values are configurable via `src/config.py` and must never be hardcoded elsewhere.

---

## Reward Table

| Event | Reward |
|---|---|
| Eat pellet | +10 |
| Eat power pellet | +50 |
| Eat ghost (during power mode) | +100 |
| Clear maze (all pellets collected) | +500 |
| Caught by ghost | -200 |
| Hit wall (invalid move) | -5 |
| Each time step | -1 |

---

## Design Rationale

- The step penalty (-1) creates urgency and discourages idle behavior.
- The wall penalty (-5) discourages repeatedly attempting invalid moves.
- The maze completion bonus (+500) provides a clear terminal reward for the primary objective.
- Ghost avoidance is implicitly encouraged by the large caught penalty (-200).
- Power pellet and ghost-eating rewards create an incentive to use power mode strategically.

---

## Configurable Location

All values are defined in `src/config.py` under the `RewardConfig`
dataclass. `PacmanEnvironment` accepts an optional `reward_config`, while its
default and the normal training/evaluation paths continue to use `REWARDS`.
The ablation runner passes the same selected variant into training and
evaluation.

The `rewards.py` module computes the appropriate reward for each environment
event from the selected `RewardConfig`.

Three optional endgame terms are also defined in `RewardConfig`, all with
zero defaults: `PELLET_DISTANCE_DECREASE`, `PELLET_DISTANCE_INCREASE`, and
`REVISIT_CELL`. When enabled, the environment compares shortest BFS distance
to a remaining pellet before and after movement, then applies the configured
signed value for a decrease or increase. A revisit is a move into one of the
last ten recorded movement positions; wall bumps do not count as revisits.
The endgame experiment uses +0.5, -0.5, and -0.1 respectively. These terms
do not affect baseline training or `models/best_model.zip`.

`GHOST_DISTANCE_DECREASE` is another zero-default opt-in term. When enabled,
the environment applies it if Pac-Man's move reduces the shortest-path
distance to a non-vulnerable ghost and leaves the nearest such ghost within
three maze steps. The `features_shaping_ghost` experiment sets this penalty
to -0.5; it does not affect the existing variants or baseline.

---

## Notes

- Reward shaping adjustments should be logged in `.ai/EXPERIMENT_LOG.md`.
- Any change to reward values constitutes a new experiment and should be recorded.
- Reward totals are not directly comparable between differently scaled
  variants; compare pellets, survival steps, and success rate as well.
