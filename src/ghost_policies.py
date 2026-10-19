"""Interchangeable movement policies for rule-based and learned ghosts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

from config import ACTION_DELTAS, Action, GHOST_CHASE_PROBABILITY, GHOST_CHASE_RADIUS
from ghost_observations import build_ghost_observation
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
        del pacman_position, vulnerable
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
        return legal_action_or_fallback(
            maze,
            ghost_position,
            int(action_values[0]),
            rng,
        )


class SharedLearnedGhostController:
    """Cycle one learned ghost action per environment step, as in training."""

    def __init__(
        self,
        model: PPO,
        ghost_count: int,
        deterministic: bool = True,
    ) -> None:
        if ghost_count < 1:
            raise ValueError("ghost_count must be at least one.")
        self.learned_policy = LearnedGhostPolicy(model, deterministic)
        self.ghost_count = ghost_count
        self.active_ghost_index = 0

    def reset(self) -> None:
        """Start each episode with the first controlled ghost."""
        self.active_ghost_index = 0

    def advance(self) -> None:
        """Select the next ghost for the next environment step."""
        self.active_ghost_index = (
            self.active_ghost_index + 1
        ) % self.ghost_count

    def choose_action(
        self,
        maze: Maze,
        ghost_positions: Sequence[Position],
        pacman_position: Position,
        power_mode_active: bool,
        vulnerable: bool,
        rng: np.random.Generator,
    ) -> int | None:
        """Choose from the same pre-step observation used during training."""
        index = self.active_ghost_index
        observation = build_ghost_observation(
            maze,
            ghost_positions,
            index,
            pacman_position,
            power_mode_active,
        )
        return self.learned_policy.choose_action(
            maze,
            ghost_positions[index],
            pacman_position,
            rng,
            vulnerable=vulnerable,
            observation=observation,
        )

    def policy_for(self, ghost_index: int) -> GhostPolicy:
        """Return the learned-or-rule policy used for one ghost slot."""
        if not 0 <= ghost_index < self.ghost_count:
            raise IndexError("ghost_index is outside the controller's ghost set.")
        return _SharedLearnedGhostPolicy(self, ghost_index)


class _SharedLearnedGhostPolicy:
    def __init__(
        self,
        controller: SharedLearnedGhostController,
        ghost_index: int,
    ) -> None:
        self._controller = controller
        self._ghost_index = ghost_index
        self._rule_policy = RuleBasedGhostPolicy()

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
        if self._ghost_index == self._controller.active_ghost_index:
            return self._controller.learned_policy.choose_action(
                maze,
                ghost_position,
                pacman_position,
                rng,
                vulnerable=vulnerable,
                observation=observation,
            )
        return self._rule_policy.choose_action(
            maze,
            ghost_position,
            pacman_position,
            rng,
            vulnerable=vulnerable,
            observation=observation,
        )


def legal_action_or_fallback(
    maze: Maze,
    position: Position,
    action: int,
    rng: np.random.Generator,
) -> int | None:
    """Replace a wall-directed action with a random legal move."""
    valid_actions = _valid_actions(maze, position)
    if not valid_actions:
        return None
    try:
        selected_action = int(Action(action))
    except (TypeError, ValueError):
        selected_action = -1
    if selected_action in valid_actions:
        return selected_action
    return valid_actions[int(rng.integers(len(valid_actions)))]


def _valid_actions(maze: Maze, position: Position) -> list[int]:
    return [
        int(action)
        for action, delta in ACTION_DELTAS.items()
        if maze.is_walkable(position + delta)
    ]


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first.row - second.row) + abs(first.col - second.col)
