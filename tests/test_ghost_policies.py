"""Tests for interchangeable ghost movement policies."""

from __future__ import annotations

import numpy as np

from config import ACTION_DELTAS, Action
from ghost import Ghost
from ghost_observations import build_ghost_observation
from ghost_policies import (
    ChaseGhostPolicy,
    LearnedGhostPolicy,
    RandomGhostPolicy,
    RuleBasedGhostPolicy,
    SharedLearnedGhostController,
    legal_action_or_fallback,
)
from maze import Maze, Position


_LAYOUT = [
    [0, 0, 0, 0, 0],
    [0, 4, 1, 2, 0],
    [0, 1, 1, 1, 0],
    [0, 5, 1, 1, 0],
    [0, 0, 0, 0, 0],
]


def test_chase_policy_moves_closer_to_pacman() -> None:
    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(3, 1), policy=ChaseGhostPolicy())
    ghost.move(maze, Position(1, 1), np.random.default_rng(42))
    assert ghost.position == Position(2, 1)


def test_random_policy_stays_in_walkable_cells() -> None:
    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(3, 1), policy=RandomGhostPolicy())
    ghost.move(maze, Position(1, 1), np.random.default_rng(42))
    assert maze.is_walkable(ghost.position)


def test_rule_policy_keeps_original_defaults() -> None:
    policy = RuleBasedGhostPolicy()
    assert policy.chase_probability == 0.30
    assert policy.chase_radius == 5


def test_invalid_policy_action_does_not_move_ghost() -> None:
    class InvalidPolicy:
        def choose_action(self, *args: object, **kwargs: object) -> int:
            return 0

    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(1, 1), policy=InvalidPolicy())
    ghost.move(maze, Position(3, 1), np.random.default_rng(42))
    assert ghost.position == Position(1, 1)


def test_wall_directed_model_override_falls_back_to_a_legal_move() -> None:
    maze = Maze(_LAYOUT)
    ghost = Ghost(Position(1, 1))
    ghost.move(
        maze,
        Position(3, 1),
        np.random.default_rng(42),
        action_override=int(Action.UP),
    )
    assert ghost.position != Position(1, 1)
    assert maze.is_walkable(ghost.position)


def test_learned_policy_falls_back_when_model_selects_wall() -> None:
    class FixedActionModel:
        def predict(
            self,
            observation: np.ndarray,
            deterministic: bool,
        ) -> tuple[int, None]:
            del observation, deterministic
            return 0, None

    maze = Maze(_LAYOUT)
    policy = LearnedGhostPolicy(FixedActionModel())
    action = policy.choose_action(
        maze,
        Position(1, 1),
        Position(3, 1),
        np.random.default_rng(42),
        observation=np.zeros(maze.height * maze.width * 6 + 2, dtype=np.float32),
    )
    assert action is not None
    assert maze.is_walkable(
        Position(1, 1) + ACTION_DELTAS[Action(action)]
    )


def test_legal_action_fallback_returns_none_without_walkable_neighbours() -> None:
    maze = Maze(_LAYOUT)
    assert legal_action_or_fallback(
        maze,
        Position(0, 0),
        0,
        np.random.default_rng(42),
    ) is None


def test_shared_learned_controller_cycles_one_learned_ghost_per_step() -> None:
    class CountingModel:
        calls = 0

        def predict(
            self,
            observation: np.ndarray,
            deterministic: bool,
        ) -> tuple[int, None]:
            self.calls += 1
            return 0, None

    model = CountingModel()
    controller = SharedLearnedGhostController(model, ghost_count=2)
    first = controller.policy_for(0)
    second = controller.policy_for(1)
    maze = Maze(_LAYOUT)
    observation = np.zeros(maze.height * maze.width * 6 + 2, dtype=np.float32)
    rng = np.random.default_rng(42)

    first.choose_action(
        maze,
        Position(3, 1),
        Position(1, 1),
        rng,
        observation=observation,
    )
    second.choose_action(
        maze,
        Position(3, 2),
        Position(1, 1),
        rng,
        observation=observation,
    )
    assert model.calls == 1

    controller.advance()
    first.choose_action(
        maze,
        Position(3, 1),
        Position(1, 1),
        rng,
        observation=observation,
    )
    second.choose_action(
        maze,
        Position(3, 2),
        Position(1, 1),
        rng,
        observation=observation,
    )
    assert model.calls == 2


def test_shared_controller_uses_training_observation_construction() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.observation: np.ndarray | None = None

        def predict(
            self,
            observation: np.ndarray,
            deterministic: bool,
        ) -> tuple[int, None]:
            del deterministic
            self.observation = observation.copy()
            return 0, None

    maze = Maze(_LAYOUT)
    positions = [Position(3, 1), Position(3, 2)]
    pacman = Position(1, 1)
    model = RecordingModel()
    controller = SharedLearnedGhostController(model, ghost_count=2)
    controller.choose_action(
        maze,
        positions,
        pacman,
        power_mode_active=False,
        vulnerable=False,
        rng=np.random.default_rng(42),
    )
    expected = build_ghost_observation(
        maze,
        positions,
        0,
        pacman,
        power_mode_active=False,
    )
    assert model.observation is not None
    assert np.array_equal(model.observation, expected)
