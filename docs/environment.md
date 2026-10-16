# Environment

## Overview

`PacmanEnvironment` is a custom [Gymnasium](https://gymnasium.farama.org/) environment that recreates a simplified Pac-Man game on a 12×12 grid. It is the interface between the PPO agent and the game world.

```python
import gymnasium as gym
from environment import PacmanEnvironment

env = PacmanEnvironment()
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)  # move Up
env.close()
```

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `render_mode` | `str \| None` | `None` | `"human"` opens a Pygame window; `None` for headless training |
| `max_steps` | `int` | `1000` | Maximum steps per episode before truncation |
| `maze_layout` | `list[list[int]] \| None` | `None` | Custom grid layout; `None` uses the default 12×12 maze |
| `render_fps` | `int` | `8` | Target Pygame playback speed |
| `reward_config` | `RewardConfig` | `REWARDS` | Reward variant; the baseline remains the default |
| `ghost_count` | `int \| None` | `None` (two in the default maze) | Select ghost spawns; `None` uses all layout spawns |

---

## Observation Space

`Box(low=0.0, high=0.1, shape=(723,), dtype=float32)`

The 723-element flat vector is built by concatenating five 12×12 grid channels (144 values each) and three scalar features:

| Index range | Content |
|---|---|
| 0 – 143 | Wall grid — 1.0 at wall cells |
| 144 – 287 | Remaining regular pellets |
| 288 – 431 | Remaining power pellets |
| 432 – 575 | Pac-Man position — 1.0 at current cell |
| 576 – 719 | Ghost positions — 1.0 at each occupied cell |
| 720 | Power mode active (0.0 or 1.0) |
| 721 | Power timer normalised (`remaining / POWER_MODE_DURATION`) |
| 722 | Pellets remaining normalised (`remaining / total`) |

All values are guaranteed to be in [0.0, 1.0].

### Optional observation features

`PacmanEnvironment(observation_config=ObservationConfig(...))` can append
ghost movement directions and/or the previous Pac-Man action. Both flags are
disabled by default. With `include_ghost_direction=True`, five values are
added per ghost (four direction one-hot values and a vulnerability flag).
`include_previous_pacman_action=True` adds four action one-hot values (zeros
before the first action). The resulting size is:

`H × W × 5 + 3 + (ghost count × 5, when enabled) + (4, when enabled)`.

For the default 12×12, two-ghost maze, enabling both flags produces 737
values. The original model-compatible shape remains 723 when both are off.

---

## Action Space

`Discrete(4)`

| Value | Direction |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

Invalid actions (move into a wall) leave Pac-Man in place and incur the `HIT_WALL` penalty.

---

## Episode Lifecycle

### reset(seed, options)

Restores all pellets, returns all agents to spawn positions, and resets the power-mode timer and step counter.

Returns `(obs, info)`.

### step(action)

Executes one game tick:

1. Move Pac-Man (or stay on wall).
2. Consume pellet at new cell.
3. Check collision with ghosts at Pac-Man's new position.
4. Decrement power-mode timer (skipped on the step power mode is activated).
5. Move all ghosts.
6. Check collision again (ghost walked into Pac-Man).
7. Check maze-cleared condition.
8. Compute reward, increment step counter.

Returns `(obs, reward, terminated, truncated, info)`.

### Termination (`terminated=True`)

- All pellets collected (maze cleared).
- Pac-Man caught by a normal ghost.

### Truncation (`truncated=True`)

- Step count reaches `max_steps` (default 1000).

---

## Info Dictionary

Both `reset` and `step` return an `info` dict:

```python
{
    "step":                      int,    # steps taken this episode
    "episode_reward":            float,  # cumulative reward
    "pellets_remaining":         int,
    "total_pellets":             int,
    "completion_fraction":       float,  # 0.0 – 1.0
    "power_mode_active":         bool,
    "power_mode_steps_remaining": int,
    "pacman_position":           tuple[int, int],
    "ghost_positions":           list[tuple[int, int]],
}
```

---

## Rendering

```python
env = PacmanEnvironment(render_mode="human")
obs, _ = env.reset()
for _ in range(100):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        break
env.close()
```

Rendering is handled by `renderer.PacmanRenderer`. The renderer is created lazily on the first `render()` call and does not affect training performance when `render_mode=None`.

---

## Gymnasium Compliance

`PacmanEnvironment` passes SB3's `check_env` validation:

```python
from stable_baselines3.common.env_checker import check_env
check_env(PacmanEnvironment())
```
