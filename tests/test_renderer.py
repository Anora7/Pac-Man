"""
Tests for src/renderer.py — Phase 5: Renderer.

SDL_VIDEODRIVER and SDL_AUDIODRIVER are set to "dummy" at module level so
that pygame can initialise without a physical display server.  These must
be set before pygame (or the renderer, which imports it) is first loaded.

Covers:
  - RenderState construction and validation
  - PacmanRenderer default and custom configuration
  - Lazy pygame initialisation (first render() call opens the window)
  - render() completes without error for various state combinations
  - Window dimensions match expected values
  - close() is safe before and after render()
  - Multiple render() calls do not accumulate errors
  - Vulnerable ghost rendering (different colour)
  - Empty ghost list
  - Repr strings
"""

from __future__ import annotations

import os

# Must be set before pygame is imported for the first time.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest
import pygame

from config import CELL_SIZE, FPS, MAZE_HEIGHT, MAZE_WIDTH
from maze import Maze, Position
from renderer import PacmanRenderer, RenderState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_state(
    maze: Maze,
    *,
    pacman: Position | None = None,
    ghost_positions: list[Position] | None = None,
    ghost_vulnerable: list[bool] | None = None,
    score: float = 0.0,
    step: int = 0,
    power_mode_active: bool = False,
    power_mode_steps_remaining: int = 0,
    episode: int = 1,
) -> RenderState:
    """Build a RenderState with sensible defaults for testing."""
    spawns = maze.ghost_spawns()
    return RenderState(
        maze=maze,
        pacman_position=pacman or maze.pacman_spawn(),
        ghost_positions=ghost_positions if ghost_positions is not None else spawns,
        ghost_vulnerable=ghost_vulnerable if ghost_vulnerable is not None else [False] * len(ghost_positions or spawns),
        score=score,
        step=step,
        power_mode_active=power_mode_active,
        power_mode_steps_remaining=power_mode_steps_remaining,
        episode=episode,
    )


# ---------------------------------------------------------------------------
# RenderState
# ---------------------------------------------------------------------------

class TestRenderState:
    def test_creates_with_valid_data(self) -> None:
        maze = Maze()
        state = _default_state(maze)
        assert state.pacman_position == maze.pacman_spawn()

    def test_default_episode_is_one(self) -> None:
        maze = Maze()
        state = _default_state(maze)
        assert state.episode == 1

    def test_is_frozen(self) -> None:
        maze = Maze()
        state = _default_state(maze)
        with pytest.raises((AttributeError, TypeError)):
            state.score = 999.0  # type: ignore[misc]

    def test_mismatched_ghost_lists_raise(self) -> None:
        maze = Maze()
        with pytest.raises(ValueError, match="ghost_positions length"):
            RenderState(
                maze=maze,
                pacman_position=maze.pacman_spawn(),
                ghost_positions=[Position(5, 5), Position(5, 6)],
                ghost_vulnerable=[True],  # length mismatch
                score=0.0,
                step=0,
                power_mode_active=False,
                power_mode_steps_remaining=0,
            )

    def test_empty_ghost_list_is_valid(self) -> None:
        maze = Maze()
        state = RenderState(
            maze=maze,
            pacman_position=maze.pacman_spawn(),
            ghost_positions=[],
            ghost_vulnerable=[],
            score=0.0,
            step=0,
            power_mode_active=False,
            power_mode_steps_remaining=0,
        )
        assert state.ghost_positions == []

    def test_power_mode_fields(self) -> None:
        maze = Maze()
        state = _default_state(
            maze,
            power_mode_active=True,
            power_mode_steps_remaining=25,
        )
        assert state.power_mode_active is True
        assert state.power_mode_steps_remaining == 25

    def test_score_and_step_stored(self) -> None:
        maze = Maze()
        state = _default_state(maze, score=340.0, step=42)
        assert state.score == pytest.approx(340.0)
        assert state.step == 42

    def test_single_ghost(self) -> None:
        maze = Maze()
        state = _default_state(
            maze,
            ghost_positions=[Position(5, 5)],
            ghost_vulnerable=[True],
        )
        assert len(state.ghost_positions) == 1
        assert state.ghost_vulnerable[0] is True


# ---------------------------------------------------------------------------
# PacmanRenderer — configuration
# ---------------------------------------------------------------------------

class TestRendererConfig:
    def test_default_cell_size(self) -> None:
        r = PacmanRenderer()
        assert r.cell_size == CELL_SIZE

    def test_default_fps(self) -> None:
        r = PacmanRenderer()
        assert r.fps == FPS

    def test_default_render_mode(self) -> None:
        r = PacmanRenderer()
        assert r.render_mode == "human"

    def test_custom_cell_size(self) -> None:
        r = PacmanRenderer(cell_size=20)
        assert r.cell_size == 20

    def test_custom_fps(self) -> None:
        r = PacmanRenderer(fps=30)
        assert r.fps == 30

    def test_unsupported_render_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="render_mode"):
            PacmanRenderer(render_mode="rgb_array")

    def test_window_size_default(self) -> None:
        r = PacmanRenderer()
        expected_w = CELL_SIZE * MAZE_WIDTH
        expected_h = CELL_SIZE * MAZE_HEIGHT + 80
        assert r.window_size == (expected_w, expected_h)

    def test_window_size_custom_cell(self) -> None:
        r = PacmanRenderer(cell_size=20)
        expected_w = 20 * MAZE_WIDTH
        expected_h = 20 * MAZE_HEIGHT + 80
        assert r.window_size == (expected_w, expected_h)

    def test_not_initialized_before_first_render(self) -> None:
        r = PacmanRenderer()
        assert not r._initialized

    def test_repr_before_render(self) -> None:
        r = PacmanRenderer()
        text = repr(r)
        assert "PacmanRenderer" in text
        assert "not initialised" in text


# ---------------------------------------------------------------------------
# PacmanRenderer — lifecycle
# ---------------------------------------------------------------------------

class TestRendererLifecycle:
    def test_close_before_render_is_safe(self) -> None:
        r = PacmanRenderer()
        r.close()  # must not raise
        assert not r._initialized

    def test_close_is_idempotent(self) -> None:
        r = PacmanRenderer()
        r.close()
        r.close()  # second close must not raise

    def test_render_initialises_pygame(self) -> None:
        r = PacmanRenderer()
        maze = Maze()
        state = _default_state(maze)
        assert not r._initialized
        r.render(state)
        assert r._initialized
        r.close()

    def test_surface_available_after_render(self) -> None:
        r = PacmanRenderer()
        maze = Maze()
        r.render(_default_state(maze))
        assert r._surface is not None
        r.close()

    def test_surface_none_after_close(self) -> None:
        r = PacmanRenderer()
        maze = Maze()
        r.render(_default_state(maze))
        r.close()
        assert r._surface is None

    def test_repr_after_render(self) -> None:
        r = PacmanRenderer()
        maze = Maze()
        r.render(_default_state(maze))
        text = repr(r)
        assert "active" in text
        r.close()

    def test_repr_after_close(self) -> None:
        r = PacmanRenderer()
        maze = Maze()
        r.render(_default_state(maze))
        r.close()
        text = repr(r)
        assert "not initialised" in text


# ---------------------------------------------------------------------------
# PacmanRenderer — render correctness (no crash tests)
# ---------------------------------------------------------------------------

class TestRenderNoCrash:
    """Verify that render() completes without error for various game states."""

    def setup_method(self) -> None:
        self.maze = Maze()
        self.renderer = PacmanRenderer()

    def teardown_method(self) -> None:
        self.renderer.close()

    def test_window_close_event_is_exposed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        maze = Maze()
        renderer = PacmanRenderer()
        renderer.render(_default_state(maze))
        monkeypatch.setattr(
            pygame.event,
            "get",
            lambda: [pygame.event.Event(pygame.QUIT)],
        )
        renderer.render(_default_state(maze))
        assert renderer.window_closed
        renderer.close()

    def test_render_initial_state(self) -> None:
        self.renderer.render(_default_state(self.maze))

    def test_render_with_score_and_steps(self) -> None:
        state = _default_state(self.maze, score=250.0, step=30)
        self.renderer.render(state)

    def test_render_power_mode_active(self) -> None:
        state = _default_state(
            self.maze,
            power_mode_active=True,
            power_mode_steps_remaining=20,
        )
        self.renderer.render(state)

    def test_render_power_mode_inactive(self) -> None:
        state = _default_state(self.maze, power_mode_active=False)
        self.renderer.render(state)

    def test_render_vulnerable_ghosts(self) -> None:
        spawns = self.maze.ghost_spawns()
        state = _default_state(
            self.maze,
            ghost_positions=spawns,
            ghost_vulnerable=[True] * len(spawns),
        )
        self.renderer.render(state)

    def test_render_mixed_ghost_states(self) -> None:
        spawns = self.maze.ghost_spawns()
        state = _default_state(
            self.maze,
            ghost_positions=spawns,
            ghost_vulnerable=[True, False],
        )
        self.renderer.render(state)

    def test_render_no_ghosts(self) -> None:
        state = RenderState(
            maze=self.maze,
            pacman_position=self.maze.pacman_spawn(),
            ghost_positions=[],
            ghost_vulnerable=[],
            score=0.0,
            step=0,
            power_mode_active=False,
            power_mode_steps_remaining=0,
        )
        self.renderer.render(state)

    def test_render_multiple_times(self) -> None:
        for step in range(5):
            state = _default_state(self.maze, step=step)
            self.renderer.render(state)

    def test_render_pacman_at_various_positions(self) -> None:
        walkable_cells = [
            Position(1, 1),
            Position(7, 5),
            Position(10, 5),
        ]
        for pos in walkable_cells:
            state = _default_state(self.maze, pacman=pos)
            self.renderer.render(state)

    def test_render_after_pellets_consumed(self) -> None:
        """Renderer should not crash when pellets have been eaten."""
        self.maze.consume_pellet(Position(1, 1))
        self.maze.consume_pellet(Position(1, 2))
        self.renderer.render(_default_state(self.maze))

    def test_render_high_episode_number(self) -> None:
        state = _default_state(self.maze, episode=9999)
        self.renderer.render(state)

    def test_render_negative_score(self) -> None:
        """Negative scores (penalty-heavy episode) must display without error."""
        state = _default_state(self.maze, score=-350.0)
        self.renderer.render(state)
