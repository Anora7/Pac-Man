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

---

## Notes

- Reward shaping adjustments should be logged in `.ai/EXPERIMENT_LOG.md`.
- Any change to reward values constitutes a new experiment and should be recorded.
- Reward totals are not directly comparable between differently scaled
  variants; compare pellets, survival steps, and success rate as well.
