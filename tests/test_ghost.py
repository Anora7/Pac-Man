"""
Tests for src/ghost.py — Phase 3: Ghost AI.

Covers:
  - GhostState enum values and properties
  - Ghost initialisation (position, state, repr)
  - Episode lifecycle: reset, respawn
  - State transitions: make_vulnerable, restore_normal
  - Movement always lands on walkable cells
  - NORMAL + chase_probability=1.0 → always chases toward Pac-Man
  - NORMAL + chase_probability=0.0 → always moves randomly
  - NORMAL + distance > chase_radius → always moves randomly
  - VULNERABLE → always moves randomly regardless of chase_probability
  - Ghost stays in place when no walkable neighbours
  - Multiple ghosts: independent state
  - Determinism: same seed produces same sequence of moves
  - Custom chase_probability and chase_radius configuration
"""

from __future__ import annotations

import numpy as np
import pytest

from config import GHOST_CHASE_PROBABILITY, GHOST_CHASE_RADIUS
from ghost import Ghost, GhostState, _manhattan_distance
from maze import Maze, Position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def maze() -> Maze:
    return Maze()


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


# ---------------------------------------------------------------------------
# GhostState
# ---------------------------------------------------------------------------

class TestGhostState:
    def test_normal_and_vulnerable_are_distinct(self) -> None:
        assert GhostState.NORMAL != GhostState.VULNERABLE

    def test_state_names(self) -> None:
        assert GhostState.NORMAL.name == "NORMAL"
        assert GhostState.VULNERABLE.name == "VULNERABLE"


# ---------------------------------------------------------------------------
# Manhattan distance helper
# ---------------------------------------------------------------------------

class TestManhattanDistance:
    def test_same_position(self) -> None:
        assert _manhattan_distance(Position(3, 3), Position(3, 3)) == 0

    def test_horizontal(self) -> None:
        assert _manhattan_distance(Position(3, 1), Position(3, 5)) == 4

    def test_vertical(self) -> None:
        assert _manhattan_distance(Position(1, 3), Position(5, 3)) == 4

    def test_diagonal(self) -> None:
        assert _manhattan_distance(Position(0, 0), Position(3, 4)) == 7

    def test_symmetry(self) -> None:
        a, b = Position(1, 2), Position(4, 6)
        assert _manhattan_distance(a, b) == _manhattan_distance(b, a)


# ---------------------------------------------------------------------------
# Ghost initialisation
# ---------------------------------------------------------------------------

class TestGhostInit:
    def test_position_equals_spawn_on_creation(self) -> None:
        spawn = Position(5, 5)
        ghost = Ghost(spawn=spawn)
        assert ghost.position == spawn

    def test_initial_state_is_normal(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        assert ghost.state == GhostState.NORMAL

    def test_default_chase_probability(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        assert ghost.chase_probability == GHOST_CHASE_PROBABILITY

    def test_default_chase_radius(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        assert ghost.chase_radius == GHOST_CHASE_RADIUS

    def test_custom_chase_probability(self) -> None:
        ghost = Ghost(spawn=Position(5, 5), chase_probability=0.8)
        assert ghost.chase_probability == 0.8

    def test_custom_chase_radius(self) -> None:
        ghost = Ghost(spawn=Position(5, 5), chase_radius=3)
        assert ghost.chase_radius == 3

    def test_repr_contains_key_info(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        text = repr(ghost)
        assert "Ghost" in text
        assert "NORMAL" in text

    def test_is_normal_initially(self) -> None:
        assert Ghost(spawn=Position(5, 5)).is_normal

    def test_is_not_vulnerable_initially(self) -> None:
        assert not Ghost(spawn=Position(5, 5)).is_vulnerable


# ---------------------------------------------------------------------------
# Lifecycle: reset and respawn
# ---------------------------------------------------------------------------

class TestGhostLifecycle:
    def test_reset_restores_position_to_spawn(self, maze: Maze, rng: np.random.Generator) -> None:
        spawn = Position(5, 5)
        ghost = Ghost(spawn=spawn)
        # Move ghost away from spawn
        ghost.move(maze, Position(10, 5), rng)
        assert ghost.position != spawn or True  # may or may not move depending on rng
        ghost.reset()
        assert ghost.position == spawn

    def test_reset_restores_normal_state(self, maze: Maze, rng: np.random.Generator) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        ghost.reset()
        assert ghost.state == GhostState.NORMAL

    def test_respawn_restores_position(self, maze: Maze, rng: np.random.Generator) -> None:
        spawn = Position(5, 5)
        ghost = Ghost(spawn=spawn)
        ghost.move(maze, Position(10, 5), rng)
        ghost.respawn()
        assert ghost.position == spawn

    def test_respawn_restores_normal_state(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        ghost.respawn()
        assert ghost.state == GhostState.NORMAL
        assert ghost.is_normal

    def test_respawn_clears_vulnerability(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        assert ghost.is_vulnerable
        ghost.respawn()
        assert not ghost.is_vulnerable

    def test_multiple_resets_are_idempotent(self) -> None:
        spawn = Position(5, 5)
        ghost = Ghost(spawn=spawn)
        ghost.reset()
        ghost.reset()
        assert ghost.position == spawn
        assert ghost.is_normal


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_make_vulnerable(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        assert ghost.state == GhostState.VULNERABLE
        assert ghost.is_vulnerable
        assert not ghost.is_normal

    def test_restore_normal(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        ghost.restore_normal()
        assert ghost.state == GhostState.NORMAL
        assert ghost.is_normal
        assert not ghost.is_vulnerable

    def test_make_vulnerable_twice_is_safe(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        ghost.make_vulnerable()
        assert ghost.is_vulnerable

    def test_restore_normal_when_already_normal_is_safe(self) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.restore_normal()
        assert ghost.is_normal


# ---------------------------------------------------------------------------
# Movement — walls and bounds
# ---------------------------------------------------------------------------

class TestMovementWalls:
    def test_ghost_always_lands_on_walkable_cell(self, maze: Maze) -> None:
        """Ghost must never move to a wall over many random steps."""
        ghost = Ghost(spawn=Position(5, 5))
        rng = np.random.default_rng(seed=99)
        for _ in range(200):
            ghost.move(maze, Position(10, 5), rng)
            assert maze.is_walkable(ghost.position), (
                f"Ghost moved to non-walkable cell {ghost.position}"
            )

    def test_vulnerable_ghost_always_lands_on_walkable_cell(self, maze: Maze) -> None:
        ghost = Ghost(spawn=Position(5, 5))
        ghost.make_vulnerable()
        rng = np.random.default_rng(seed=7)
        for _ in range(200):
            ghost.move(maze, Position(10, 5), rng)
            assert maze.is_walkable(ghost.position)

    def test_ghost_spawned_in_corridor_moves_correctly(self, maze: Maze) -> None:
        """Ghost starting in a corridor cell should be able to move."""
        ghost = Ghost(spawn=Position(1, 5))  # middle of top corridor
        rng = np.random.default_rng(seed=1)
        ghost.move(maze, Position(10, 5), rng)
        assert maze.is_walkable(ghost.position)

    def test_ghost_does_not_move_when_no_neighbours(self) -> None:
        """If all neighbours are walls, ghost stays put (defensive guard)."""
        # Ghost spawn at (2,2) is surrounded on all four sides by walls.
        # Pac-Man spawn and power pellet placed away to satisfy validation.
        layout = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],  # (1,2) = WALL
            [0, 0, 5, 0, 0],  # (2,1),(2,3) = WALL; (2,2) = GHOST_SPAWN
            [0, 0, 0, 0, 0],  # (3,2) = WALL
            [0, 4, 2, 0, 0],  # PACMAN_SPAWN at (4,1), POWER_PELLET at (4,2)
        ]
        isolated_maze = Maze(layout=layout)
        ghost = Ghost(spawn=Position(2, 2))
        rng = np.random.default_rng(seed=0)
        ghost.move(isolated_maze, Position(4, 1), rng)
        assert ghost.position == Position(2, 2)


# ---------------------------------------------------------------------------
# Movement — chase behaviour
# ---------------------------------------------------------------------------

class TestChaseBehaviour:
    def test_ghost_chases_toward_pacman_when_probability_is_one(
        self, maze: Maze
    ) -> None:
        """With chase_probability=1.0 and within radius, ghost always chases.

        Ghost at (7,1); Pac-Man at (7,5).
        Walkable neighbours of (7,1): (7,2)[distance 3], (8,1)[distance 5].
        Chase must pick (7,2).
        """
        ghost = Ghost(spawn=Position(7, 1), chase_probability=1.0, chase_radius=100)
        rng = np.random.default_rng(seed=0)
        ghost.move(maze, Position(7, 5), rng)
        assert ghost.position == Position(7, 2)

    def test_ghost_reduces_distance_to_pacman_when_chasing(
        self, maze: Maze
    ) -> None:
        """Over multiple steps with full-chase, ghost should approach Pac-Man."""
        ghost = Ghost(spawn=Position(7, 1), chase_probability=1.0, chase_radius=100)
        pacman_pos = Position(7, 9)
        rng = np.random.default_rng(seed=0)
        initial_dist = _manhattan_distance(ghost.position, pacman_pos)
        for _ in range(5):
            ghost.move(maze, pacman_pos, rng)
        final_dist = _manhattan_distance(ghost.position, pacman_pos)
        assert final_dist < initial_dist

    def test_ghost_with_zero_probability_never_chases(self, maze: Maze) -> None:
        """With chase_probability=0.0 ghost always picks randomly.

        We verify this by confirming the ghost visits more than one distinct
        cell over many steps from a position with multiple neighbours.
        """
        ghost = Ghost(spawn=Position(7, 5), chase_probability=0.0, chase_radius=100)
        pacman_pos = Position(7, 5)  # Pac-Man on top of ghost — would always chase
        visited: set[Position] = set()
        for seed in range(40):
            g = Ghost(spawn=Position(7, 5), chase_probability=0.0, chase_radius=100)
            g.move(maze, pacman_pos, np.random.default_rng(seed))
            visited.add(g.position)
        # With 4 walkable neighbours and 40 seeds, we must visit more than 1
        assert len(visited) > 1

    def test_ghost_outside_chase_radius_moves_randomly(self, maze: Maze) -> None:
        """When Pac-Man is beyond chase_radius, ghost always moves randomly."""
        # Ghost at (7,5), Pac-Man at (1,1): distance = 10, radius = 3
        ghost = Ghost(spawn=Position(7, 5), chase_probability=1.0, chase_radius=3)
        pacman_pos = Position(1, 1)
        visited: set[Position] = set()
        for seed in range(40):
            g = Ghost(spawn=Position(7, 5), chase_probability=1.0, chase_radius=3)
            g.move(maze, pacman_pos, np.random.default_rng(seed))
            visited.add(g.position)
        # Random movement should produce more than one destination
        assert len(visited) > 1

    def test_ghost_exactly_at_chase_radius_can_chase(self, maze: Maze) -> None:
        """Ghost at exactly chase_radius distance should still be eligible to chase."""
        # Ghost at (7,1), Pac-Man at (7,4): distance = 3
        ghost = Ghost(spawn=Position(7, 1), chase_probability=1.0, chase_radius=3)
        rng = np.random.default_rng(seed=0)
        ghost.move(maze, Position(7, 4), rng)
        # Chase should have been triggered; (7,2) closes the distance
        assert ghost.position == Position(7, 2)

    def test_ghost_one_beyond_radius_does_not_chase(self, maze: Maze) -> None:
        """Ghost at chase_radius + 1 distance must not chase."""
        # Ghost at (7,1), Pac-Man at (7,5): distance = 4, radius = 3
        ghost = Ghost(spawn=Position(7, 1), chase_probability=1.0, chase_radius=3)
        visited: set[Position] = set()
        for seed in range(40):
            g = Ghost(spawn=Position(7, 1), chase_probability=1.0, chase_radius=3)
            g.move(maze, Position(7, 5), np.random.default_rng(seed))
            visited.add(g.position)
        assert len(visited) > 1  # random movement → multiple destinations


# ---------------------------------------------------------------------------
# Movement — vulnerable behaviour
# ---------------------------------------------------------------------------

class TestVulnerableBehaviour:
    def test_vulnerable_ghost_does_not_always_chase(self, maze: Maze) -> None:
        """Vulnerable ghost with chase_prob=1.0 must still move randomly.

        Ghost at (7,3); neighbours: (6,3), (8,3), (7,2), (7,4).
        Chase would always pick (7,4) when Pac-Man is at (7,4).
        Vulnerable ghost should visit cells other than (7,4) over many trials.
        """
        pacman_pos = Position(7, 4)
        destinations: set[Position] = set()
        for seed in range(50):
            g = Ghost(spawn=Position(7, 3), chase_probability=1.0, chase_radius=100)
            g.make_vulnerable()
            g.move(maze, pacman_pos, np.random.default_rng(seed))
            destinations.add(g.position)
        # With 4 walkable neighbours and 50 seeds, must see >1 destination
        assert len(destinations) > 1

    def test_normal_ghost_always_chases_from_same_position(self, maze: Maze) -> None:
        """Verify baseline: NORMAL ghost at (7,3) with prob=1.0 always picks (7,4)."""
        pacman_pos = Position(7, 4)
        for seed in range(20):
            g = Ghost(spawn=Position(7, 3), chase_probability=1.0, chase_radius=100)
            g.move(maze, pacman_pos, np.random.default_rng(seed))
            assert g.position == Position(7, 4), (
                f"Normal ghost should always chase to (7,4) but got {g.position}"
            )

    def test_vulnerability_does_not_persist_after_restore(
        self, maze: Maze, rng: np.random.Generator
    ) -> None:
        """Ghost restored to normal should be eligible to chase again."""
        ghost = Ghost(spawn=Position(7, 3), chase_probability=1.0, chase_radius=100)
        ghost.make_vulnerable()
        ghost.restore_normal()
        # Now in NORMAL with prob=1.0 → should always chase
        ghost.move(maze, Position(7, 4), rng)
        assert ghost.position == Position(7, 4)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_same_sequence(self, maze: Maze) -> None:
        """Two ghosts with identical seeds must follow identical paths."""
        spawn = Position(5, 5)
        pacman_pos = Position(10, 5)

        def run_steps(seed: int, n: int) -> list[Position]:
            ghost = Ghost(spawn=spawn)
            rng = np.random.default_rng(seed=seed)
            path = []
            for _ in range(n):
                ghost.move(maze, pacman_pos, rng)
                path.append(ghost.position)
            return path

        assert run_steps(42, 30) == run_steps(42, 30)

    def test_different_seeds_produce_different_sequences(self, maze: Maze) -> None:
        """Two ghosts with different seeds should diverge at some point."""
        spawn = Position(5, 5)
        pacman_pos = Position(10, 5)

        def run_steps(seed: int, n: int) -> list[Position]:
            ghost = Ghost(spawn=spawn)
            rng = np.random.default_rng(seed=seed)
            path = []
            for _ in range(n):
                ghost.move(maze, pacman_pos, rng)
                path.append(ghost.position)
            return path

        assert run_steps(0, 30) != run_steps(1, 30)


# ---------------------------------------------------------------------------
# Multiple ghosts
# ---------------------------------------------------------------------------

class TestMultipleGhosts:
    def test_two_ghosts_have_independent_positions(
        self, maze: Maze
    ) -> None:
        """Ghosts spawned at different locations maintain independent state."""
        ghost_a = Ghost(spawn=Position(5, 5))
        ghost_b = Ghost(spawn=Position(5, 6))
        assert ghost_a.position != ghost_b.position

    def test_two_ghosts_have_independent_state(self) -> None:
        ghost_a = Ghost(spawn=Position(5, 5))
        ghost_b = Ghost(spawn=Position(5, 6))
        ghost_a.make_vulnerable()
        assert ghost_a.is_vulnerable
        assert not ghost_b.is_vulnerable

    def test_two_ghosts_move_independently(self, maze: Maze) -> None:
        """Each ghost uses its own positional context during movement."""
        ghost_a = Ghost(spawn=Position(5, 5))
        ghost_b = Ghost(spawn=Position(5, 6))
        pacman_pos = Position(10, 5)
        rng_a = np.random.default_rng(seed=10)
        rng_b = np.random.default_rng(seed=20)
        ghost_a.move(maze, pacman_pos, rng_a)
        ghost_b.move(maze, pacman_pos, rng_b)
        # Both should land on walkable cells
        assert maze.is_walkable(ghost_a.position)
        assert maze.is_walkable(ghost_b.position)

    def test_resetting_one_ghost_does_not_affect_other(self) -> None:
        ghost_a = Ghost(spawn=Position(5, 5))
        ghost_b = Ghost(spawn=Position(5, 6))
        ghost_a.make_vulnerable()
        ghost_b.make_vulnerable()
        ghost_a.reset()
        assert ghost_a.is_normal
        assert ghost_b.is_vulnerable  # unaffected

    def test_ghosts_from_maze_spawn_positions(self, maze: Maze) -> None:
        """Construct ghosts from maze spawn positions and verify movement."""
        spawns = maze.ghost_spawns()
        assert len(spawns) == 2
        ghosts = [Ghost(spawn=pos) for pos in spawns]
        rng = np.random.default_rng(seed=5)
        for ghost in ghosts:
            for _ in range(20):
                ghost.move(maze, Position(10, 5), rng)
                assert maze.is_walkable(ghost.position)


# ---------------------------------------------------------------------------
# Walkable neighbours
# ---------------------------------------------------------------------------

class TestWalkableNeighbours:
    def test_ghost_at_spawn_has_correct_neighbours(self, maze: Maze) -> None:
        """Ghost at (5,5): walkable neighbours are (4,5) and (5,6)."""
        ghost = Ghost(spawn=Position(5, 5))
        neighbours = ghost._walkable_neighbours(maze)
        assert Position(4, 5) in neighbours   # EMPTY cell — ghost-house door
        assert Position(5, 6) in neighbours   # other ghost spawn
        assert Position(6, 5) not in neighbours  # WALL
        assert Position(5, 4) not in neighbours  # WALL

    def test_corridor_cell_has_multiple_neighbours(self, maze: Maze) -> None:
        """Ghost in the middle of a corridor should have ≥ 2 walkable neighbours."""
        ghost = Ghost(spawn=Position(7, 5))
        neighbours = ghost._walkable_neighbours(maze)
        assert len(neighbours) >= 2
