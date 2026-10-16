"""Gymnasium environment for training Pac-Man's ghosts with SB3."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from config import (
    ACTION_DELTAS,
    DEFAULT_SEED,
    GHOST_REWARDS,
    MAX_STEPS,
    PACMAN_ACTION_FEATURES,
    GhostRewardConfig,
)
from environment import PacmanEnvironment
from ghost_observations import build_ghost_observation
from maze import Position


class GhostEnvironment(gym.Env):
    """Train ghost actions while Pac-Man is frozen under an existing policy.

    Shared mode cycles a common single-action policy across each ghost's
    ghost-centred observation. Single-ghost mode controls only the selected
    ghost while leaving the others on their configured policies.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        pacman_model_path: str | Path | None = None,
        *,
        shared_policy: bool = True,
        controlled_ghost_index: int = 0,
        max_steps: int = MAX_STEPS,
        reward_config: GhostRewardConfig = GHOST_REWARDS,
    ) -> None:
        super().__init__()
        self._pacman_model = (
            PPO.load(str(pacman_model_path), device="cpu")
            if pacman_model_path is not None
            else None
        )
        self._shared_policy = shared_policy
        self._controlled_ghost_index = controlled_ghost_index
        self._env = PacmanEnvironment(max_steps=max_steps)
        self._reward_config = reward_config
        self._last_distances: list[int] = []

        ghost_count = len(self._env.ghosts)
        if not 0 <= controlled_ghost_index < ghost_count:
            self._env.close()
            raise ValueError("controlled_ghost_index is outside the ghost list.")
        self._observation_per_ghost = self._env.maze.height * self._env.maze.width * 6 + 2
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(self._observation_per_ghost,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(PACMAN_ACTION_FEATURES)
        self._active_ghost_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        reset_seed = DEFAULT_SEED if seed is None else seed
        _, info = self._env.reset(seed=reset_seed, options=options)
        self._last_distances = self._distances_to_pacman()
        self._active_ghost_index = 0
        return self._build_observation(), info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self._last_distances:
            raise RuntimeError("reset() must be called before step().")

        ghost_count = len(self._env.ghosts)
        active_index = (
            self._active_ghost_index
            if self._shared_policy
            else self._controlled_ghost_index
        )
        ghost_actions: list[int | None] = [None] * ghost_count
        ghost_actions[active_index] = int(action)

        pacman_action = self._pacman_action()
        _, _, terminated, truncated, info = self._env.step(
            pacman_action,
            ghost_actions=ghost_actions,
        )
        events = info["last_step_events"]
        current_distances = self._distances_to_pacman()
        reward = self._calculate_reward(events, current_distances)
        self._last_distances = current_distances
        if self._shared_policy:
            self._active_ghost_index = (active_index + 1) % ghost_count
        info["ghost_reward"] = reward
        return self._build_observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _build_observation(self) -> np.ndarray:
        selected_index = (
            self._active_ghost_index
            if self._shared_policy
            else self._controlled_ghost_index
        )
        return self._ghost_observation(selected_index)

    def _ghost_observation(self, index: int) -> np.ndarray:
        return build_ghost_observation(
            self._env.maze,
            [ghost.position for ghost in self._env.ghosts],
            index,
            self._env.pacman_position,
            self._env.power_mode_active,
        )

    def _pacman_action(self) -> int:
        observation = self._env._build_observation()
        if self._pacman_model is not None:
            action, _ = self._pacman_model.predict(
                observation,
                deterministic=True,
            )
            return int(np.asarray(action).item())
        return self._greedy_pellet_action()

    def _greedy_pellet_action(self) -> int:
        maze = self._env.maze
        start = self._env.pacman_position
        targets = {
            Position(row, col)
            for row in range(maze.height)
            for col in range(maze.width)
            if maze.has_pellet(Position(row, col))
            or maze.has_power_pellet(Position(row, col))
        }
        if not targets:
            return 0
        queue: deque[tuple[Position, int | None]] = deque([(start, None)])
        visited = {start}
        while queue:
            position, first_action = queue.popleft()
            if position in targets:
                return 0 if first_action is None else first_action
            for move, delta in ACTION_DELTAS.items():
                neighbour = position + delta
                if maze.is_walkable(neighbour) and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((
                        neighbour,
                        int(move) if first_action is None else first_action,
                    ))
        return 0

    def _distances_to_pacman(self) -> list[int]:
        pacman = self._env.pacman_position
        return [
            abs(ghost.position.row - pacman.row)
            + abs(ghost.position.col - pacman.col)
            for ghost in self._env.ghosts
        ]

    def _calculate_reward(
        self,
        events: dict[str, bool | int],
        distances: list[int],
    ) -> float:
        reward = self._reward_config.STEP
        if events["caught"]:
            reward += self._reward_config.CATCH_PACMAN
        if events["pellet_eaten"]:
            reward += self._reward_config.PACMAN_EATS_PELLET
        if events["power_pellet_eaten"]:
            reward += self._reward_config.PACMAN_EATS_POWER_PELLET
        reward += int(events["ghosts_eaten"]) * self._reward_config.GHOST_EATEN
        if events["cleared"]:
            reward += self._reward_config.PACMAN_CLEARS_MAZE
        if self._reward_config.REDUCE_DISTANCE:
            reward += self._reward_config.REDUCE_DISTANCE * sum(
                previous - current
                for previous, current in zip(self._last_distances, distances)
            )
        return float(reward)
