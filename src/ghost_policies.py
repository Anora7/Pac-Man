"""Interchangeable movement policies for rule-based and learned ghosts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from config import ACTION_DELTAS, Action, GHOST_CHASE_PROBABILITY, GHOST_CHASE_RADIUS
from maze import Maze, Position

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class GhostPolicy(Protocol):
    """Interface implemented by policies that select a ghost movement."""

    def choose_action(
        self,
        maze: Maze,
        ghost_position: Position,
        pacman_position: Position,
        rng: np.random.Generator,
        *,
        vulnerable: bool = False,
        observation: np.ndarray | None = None,
    ) -> int | None:
        """Choose an action index, or None if no adjacent cell is walkable."""


class RandomGhostPolicy:
    """Choose uniformly among wall-free adjacent cells."""

    def choose_action(
        self,
        maze: Maze,
        ghost_position: Position,
        pacman_position: Position,
        rng: np.random.Generator,
        *,
        vulnerable: bool = False,
        observation: np.ndarray | None = None,
    ) -> int | None:
        del pacman_position, vulnerable, observation
        actions = _valid_actions(maze, ghost_position)
        if not actions:
            return None
        return actions[int(rng.integers(len(actions)))]


class ChaseGhostPolicy:
    """Always move to the walkable neighbour closest to Pac-Man."""

    def choose_action(
        self,
        maze: Maze,
        ghost_position: Position,
        pacman_position: Position,
        rng: np.random.Generator,
        *,
        vulnerable: bool = False,
        observation: np.ndarray | None = None,
    ) -> int | None:
        del observation
        actions = _valid_actions(maze, ghost_position)
        if not actions:
            return None
        if vulnerable:
            return RandomGhostPolicy().choose_action(
                maze,
                ghost_position,
                pacman_position,
                rng,
            )
        return min(
            actions,
            key=lambda action: _manhattan_distance(
                ghost_position + ACTION_DELTAS[Action(action)],
                pacman_position,
            ),
        )


class RuleBasedGhostPolicy:
    """Preserve the baseline probabilistic chase/random policy."""

    def __init__(
        self,
        chase_probability: float = GHOST_CHASE_PROBABILITY,
        chase_radius: int = GHOST_CHASE_RADIUS,
    ) -> None:
        if not 0.0 <= chase_probability <= 1.0:
            raise ValueError("chase_probability must be between 0 and 1.")
        if chase_radius < 0:
            raise ValueError("chase_radius must be non-negative.")
        self.chase_probability = chase_probability
        self.chase_radius = chase_radius

    def choose_action(
        self,
        maze: Maze,
        ghost_position: Position,
        pacman_position: Position,
        rng: np.random.Generator,
        *,
        vulnerable: bool = False,
        observation: np.ndarray | None = None,
    ) -> int | None:
        del observation
        actions = _valid_actions(maze, ghost_position)
        if not actions:
            return None
        distance = _manhattan_distance(ghost_position, pacman_position)
        if (
            not vulnerable
            and distance <= self.chase_radius
            and rng.random() < self.chase_probability
        ):
            return min(
                actions,
                key=lambda action: _manhattan_distance(
                    ghost_position + ACTION_DELTAS[Action(action)],
                    pacman_position,
                ),
            )
        return actions[int(rng.integers(len(actions)))]


class LearnedGhostPolicy:
    """Use an SB3 PPO model to select a ghost action."""

    def __init__(self, model: PPO, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(
        cls,
        model_path: str,
        deterministic: bool = True,
    ) -> LearnedGhostPolicy:
        """Load a PPO ghost model from a Stable-Baselines3 archive."""
        from stable_baselines3 import PPO

        return cls(PPO.load(model_path, device="cpu"), deterministic)

    def choose_action(
        self,
        maze: Maze,
        ghost_position: Position,
        pacman_position: Position,
        rng: np.random.Generator,
        *,
        vulnerable: bool = False,
        observation: np.ndarray | None = None,
    ) -> int | None:
        del pacman_position, rng, vulnerable
        actions = _valid_actions(maze, ghost_position)
        if not actions:
            return None
        if observation is None:
            raise ValueError(
                "LearnedGhostPolicy requires a ghost-model observation."
            )
        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
        )
        action_values = np.asarray(action).reshape(-1)
        if action_values.size != 1:
            raise ValueError(
                "LearnedGhostPolicy expects a PPO model with one discrete "
                "ghost action."
            )
        return int(action_values[0])


def _valid_actions(maze: Maze, position: Position) -> list[int]:
    return [
        int(action)
        for action, delta in ACTION_DELTAS.items()
        if maze.is_walkable(position + delta)
    ]


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first.row - second.row) + abs(first.col - second.col)
