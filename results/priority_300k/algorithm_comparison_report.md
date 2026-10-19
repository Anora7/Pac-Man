# PPO vs DQN comparison

Matched seeds: 42, 43, 44.
Training steps per seed and algorithm: 300,000.

## Results

| Algorithm | Mean reward | Success rate | Mean pellets | Mean steps |
|---|---:|---:|---:|---:|
| PPO | 360.74 | 0.0% | 49.51 | 74.1 |
| DQN | 30.36 | 0.0% | 19.69 | 43.0 |

95% confidence intervals across seed-level means:

| Algorithm | Reward | Pellets | Survival steps |
|---|---:|---:|---:|
| PPO | 360.74 [300.25, 421.24] | 49.51 [46.80, 52.22] | 74.09 [45.05, 103.13] |
| DQN | 30.36 [-107.36, 168.07] | 19.69 [17.84, 21.54] | 43.03 [19.57, 66.50] |

## Relation to Decision 001

Decision 001 selects PPO as the primary algorithm because its clipped policy updates are expected to be stable and practical for this small discrete navigation task. This comparison tests that rationale rather than assuming PPO wins: interpret the matched-seed reward and scale-independent success, pellet, and survival metrics together.
