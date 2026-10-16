# PPO vs DQN comparison

Matched seeds: 42.
Training steps per seed and algorithm: 32.

## Results

| Algorithm | Mean reward | Success rate | Mean pellets | Mean steps |
|---|---:|---:|---:|---:|
| PPO | -117.00 | 0.0% | 5.00 | 32.0 |
| DQN | -329.00 | 0.0% | 3.00 | 29.0 |

## Relation to Decision 001

Decision 001 selects PPO as the primary algorithm because its clipped policy updates are expected to be stable and practical for this small discrete navigation task. This comparison tests that rationale rather than assuming PPO wins: interpret the matched-seed reward and scale-independent success, pellet, and survival metrics together.
