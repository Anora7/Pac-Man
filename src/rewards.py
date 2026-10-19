"""
Reward calculations for the Pac-Man RL environment.

This module owns all reward computation logic and reads values centrally
from :data:`config.REWARDS`.

This module must not modify environment state or perform rendering.
"""

from __future__ import annotations

from config import REWARDS, RewardConfig


def calculate_step_reward(
    *,
    hit_wall: bool = False,
    pellet_eaten: bool = False,
    power_pellet_eaten: bool = False,
    ghosts_eaten: int = 0,
    caught: bool = False,
    cleared: bool = False,
    pellet_distance_decreased: bool = False,
    pellet_distance_increased: bool = False,
    revisited_cell: bool = False,
    ghost_distance_decreased: bool = False,
    reward_config: RewardConfig = REWARDS,
) -> float:
    """Calculate the total scalar reward for one environment step.

    All parameters are keyword-only to prevent argument-order bugs.
    The per-step penalty (:attr:`config.RewardConfig.STEP`) is always
    included in the return value regardless of other events.

    Multiple events can occur in a single step (e.g. eating a pellet and
    hitting the step limit in the same tick); their rewards are added.

    Args:
        hit_wall:           Pac-Man attempted to move into a wall.
        pellet_eaten:       A regular pellet was consumed this step.
        power_pellet_eaten: A power pellet was consumed this step.
        ghosts_eaten:       Number of vulnerable ghosts eaten this step.
        caught:             Pac-Man was caught by a normal ghost.
        cleared:            All pellets were consumed (maze cleared).

    Returns:
        Total float reward for the step.

    Example::

        r = calculate_step_reward(pellet_eaten=True)
        # r == REWARDS.STEP + REWARDS.PELLET == -1.0 + 10.0 == 9.0
    """
    reward: float = reward_config.STEP  # per-step penalty — always applied

    if hit_wall:
        reward += reward_config.HIT_WALL

    if pellet_eaten:
        reward += reward_config.PELLET

    if power_pellet_eaten:
        reward += reward_config.POWER_PELLET

    reward += ghosts_eaten * reward_config.EAT_GHOST

    if caught:
        reward += reward_config.CAUGHT_BY_GHOST

    if cleared:
        reward += reward_config.CLEAR_MAZE

    if pellet_distance_decreased:
        reward += reward_config.PELLET_DISTANCE_DECREASE

    if pellet_distance_increased:
        reward += reward_config.PELLET_DISTANCE_INCREASE

    if revisited_cell:
        reward += reward_config.REVISIT_CELL

    if ghost_distance_decreased:
        reward += reward_config.GHOST_DISTANCE_DECREASE

    return reward
