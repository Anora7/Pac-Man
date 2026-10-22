"""
Unit tests for GoogleRenderer (Pygame rendering, animated mouth, ghost shapes, HUD).
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from config import Action
from v2_config import GOOGLE_CELL_SIZE, GOOGLE_FPS
from v2_ghost import create_google_ghosts
from v2_maze import GoogleMaze
from v2_render import GoogleRenderer, GoogleRenderState
from maze import Position


class TestGoogleRenderer:
    def test_renderer_initialization(self) -> None:
        renderer = GoogleRenderer(cell_size=30, fps=10, render_mode="human")
        assert renderer.cell_size == 30
        assert renderer.fps == 10
        assert renderer.window_w == 30 * 16
        assert renderer.window_h == 30 * 16 + 90
        assert renderer.window_size == (480, 570)
        assert renderer.window_closed is False
        renderer.close()

    def test_invalid_render_mode(self) -> None:
        with pytest.raises(ValueError):
            GoogleRenderer(render_mode="invalid")

    def test_render_state_and_frame_execution(self) -> None:
        maze = GoogleMaze()
        ghosts = create_google_ghosts(maze, active_ghost_count=4)
        renderer = GoogleRenderer(cell_size=20, fps=15, render_mode="human")

        state = GoogleRenderState(
            maze=maze,
            pacman_position=maze.pacman_spawn(),
            pacman_action=Action.RIGHT,
            ghosts=ghosts,
            score=150.0,
            step=12,
            power_mode_active=False,
            power_mode_steps_remaining=0,
            episode=1,
        )

        # Test normal render
        renderer.render(state)
        assert renderer._initialized is True
        assert renderer._surface is not None

        # Test power mode render with flashing
        power_state = GoogleRenderState(
            maze=maze,
            pacman_position=maze.pacman_spawn(),
            pacman_action=Action.UP,
            ghosts=ghosts,
            score=250.0,
            step=5,
            power_mode_active=True,
            power_mode_steps_remaining=4,
            episode=1,
        )
        for g in ghosts:
            g.is_vulnerable = True

        renderer.render(power_state)
        renderer.close()
        assert renderer._initialized is False

