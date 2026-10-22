"""
Unit tests for GoogleGhost (targeting rules, scatter/chase, frightened, release schedule).
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Action
from v2_config import (
    GhostIdentity,
    GHOST_SCATTER_CORNERS,
    GOOGLE_CHASE_STEPS,
    GOOGLE_SCATTER_STEPS,
)
from v2_ghost import (
    GoogleGhost,
    GhostMode,
    create_google_ghosts,
)
from v2_maze import GoogleMaze
from maze import Position


class TestGoogleGhost:
    def test_factory_and_release_schedule(self) -> None:
        maze = GoogleMaze()
        ghosts = create_google_ghosts(maze, chase_strength=1.0, active_ghost_count=4)
        assert len(ghosts) == 4

        # Blinky released immediately
        assert ghosts[0].identity == GhostIdentity.BLINKY
        assert ghosts[0].is_released is True
        assert ghosts[0].release_step == 0

        # Others released at 30, 60, 90
        assert ghosts[1].identity == GhostIdentity.PINKY
        assert ghosts[1].is_released is False
        assert ghosts[1].release_step == 30

        assert ghosts[2].identity == GhostIdentity.INKY
        assert ghosts[2].is_released is False
        assert ghosts[2].release_step == 60

        assert ghosts[3].identity == GhostIdentity.CLYDE
        assert ghosts[3].is_released is False
        assert ghosts[3].release_step == 90

        # Unreleased ghost does not move
        rng = np.random.default_rng(42)
        move = ghosts[1].move(maze, Position(12, 7), Action.UP, rng)
        assert move is None

        # Updating step count releases Pinky at step 30
        ghosts[1].update_mode(step_count=30, power_mode_active=False)
        assert ghosts[1].is_released is True

        # Test custom release schedule
        custom_ghosts = create_google_ghosts(maze, release_schedule=(0, 15, 45, 75))
        assert custom_ghosts[1].release_step == 15
        assert custom_ghosts[2].release_step == 45
        assert custom_ghosts[3].release_step == 75

    def test_scatter_chase_cycle(self) -> None:
        maze = GoogleMaze()
        ghost = GoogleGhost(
            identity=GhostIdentity.BLINKY,
            spawn=Position(7, 7),
            release_step=0,
        )

        # Step 0: Scatter
        ghost.update_mode(step_count=0, power_mode_active=False)
        assert ghost.current_mode == GhostMode.SCATTER

        # Step 9: Still Scatter (10 steps: 0..9)
        ghost.update_mode(step_count=9, power_mode_active=False)
        assert ghost.current_mode == GhostMode.SCATTER

        # Step 10: Chase (20 steps: 10..29)
        ghost.update_mode(step_count=10, power_mode_active=False)
        assert ghost.current_mode == GhostMode.CHASE

        # Step 30: Second Scatter cycle
        ghost.update_mode(step_count=30, power_mode_active=False)
        assert ghost.current_mode == GhostMode.SCATTER

    def test_targeting_rules(self) -> None:
        maze = GoogleMaze()
        pacman_pos = Position(10, 5)
        pacman_action = Action.UP  # Delta is (-1, 0)
        blinky_pos = Position(12, 5)

        blinky = GoogleGhost(GhostIdentity.BLINKY, Position(7, 6))
        pinky = GoogleGhost(GhostIdentity.PINKY, Position(7, 7))
        inky = GoogleGhost(GhostIdentity.INKY, Position(7, 8))
        clyde = GoogleGhost(GhostIdentity.CLYDE, Position(7, 9))

        for g in [blinky, pinky, inky, clyde]:
            g.is_released = True
            g.current_mode = GhostMode.CHASE

        # Blinky targets pacman_pos
        assert blinky.get_target_tile(maze, pacman_pos, pacman_action) == pacman_pos

        # Pinky targets 4 tiles ahead in pacman direction (row 10 - 4 = 6, col 5)
        expected_pinky = Position(6, 5)
        assert pinky.get_target_tile(maze, pacman_pos, pacman_action) == expected_pinky

        # Inky targets 4 tiles behind Pac-Man (opposite of his direction)
        # Pac-Man (10, 5) facing UP, delta (-1, 0): (10 + 4, 5) = (14, 5)
                # Inky far from Pac-Man: targets the mirrored point across the board
        inky.position = Position(1, 1)
        assert inky.get_target_tile(maze, pacman_pos, pacman_action, blinky_pos) == Position(5, 10)

        # Inky close to Pac-Man: chases directly
        inky.position = Position(10, 6)
        assert inky.get_target_tile(maze, pacman_pos, pacman_action, blinky_pos) == pacman_pos

        # Clyde: Chases if distance > 6, retreats to scatter corner if <= 6
        # Put Clyde close (e.g. at (10, 6), distance = 1)
        clyde.position = Position(10, 6)
        assert clyde.get_target_tile(maze, pacman_pos, pacman_action) == clyde.scatter_corner

        # Put Clyde far away (e.g. at (1, 1), distance >> 6)
        clyde.position = Position(1, 1)
        assert clyde.get_target_tile(maze, pacman_pos, pacman_action) == pacman_pos

    def test_no_reverse_unless_forced(self) -> None:
        maze = GoogleMaze()
        ghost = GoogleGhost(GhostIdentity.BLINKY, Position(1, 2))
        ghost.is_released = True
        ghost.current_mode = GhostMode.CHASE

        # Corridor along row 1: cols 1, 2, 3 are walkable
        # If last action was RIGHT (moved from (1, 1) to (1, 2))
        ghost.last_action = Action.RIGHT
        ghost.position = Position(1, 2)
        legal = ghost.legal_actions(maze)

        # Opposite of RIGHT is LEFT. Since other moves exist, LEFT must not be in legal
        assert Action.LEFT not in legal

        # If mode just changed, reverse is allowed
        ghost.mode_just_changed = True
        legal_after_mode_change = ghost.legal_actions(maze)
        assert Action.LEFT in legal_after_mode_change

    def test_frightened_mode_retreat_and_respawn(self) -> None:
        maze = GoogleMaze()
        ghost = GoogleGhost(GhostIdentity.BLINKY, Position(7, 6))
        ghost.is_released = True
        ghost.make_vulnerable()
        assert ghost.is_vulnerable is True
        assert ghost.current_mode == GhostMode.FRIGHTENED

        # Move should maximize distance away from Pac-Man
        rng = np.random.default_rng(123)
        pacman_pos = Position(6, 4)
        ghost.position = Position(6, 7)
        initial_dist = maze.bfs_shortest_path_distance(ghost.position, pacman_pos)
        ghost.move(maze, pacman_pos, Action.UP, rng)
        # Should step away from Pac-Man
        new_dist = maze.bfs_shortest_path_distance(ghost.position, pacman_pos)
        assert new_dist is not None and initial_dist is not None and new_dist >= initial_dist

        # Respawn returns to spawn and clears vulnerable
        ghost.respawn()
        assert ghost.position == Position(7, 6)
        assert ghost.is_vulnerable is False
        assert ghost.current_mode == GhostMode.CHASE
