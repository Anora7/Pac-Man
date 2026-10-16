"""
Pygame renderer for the Pac-Man RL environment.

This module owns:
  - RenderState: frozen dataclass carrying all display information for one frame.
  - PacmanRenderer: Pygame-based visual renderer.

This module must not modify game state, access PPO, or perform training.
It imports from maze.py and config.py only.

The environment (environment.py) constructs a RenderState each step and
passes it to PacmanRenderer.render().  The renderer never holds a reference
to the live environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config import (
    CELL_SIZE,
    COLOR_BACKGROUND,
    COLOR_GHOST,
    COLOR_GHOST_VULNERABLE,
    COLOR_HUD,
    COLOR_PACMAN,
    COLOR_PELLET,
    COLOR_POWER_PELLET,
    COLOR_WALL,
    FPS,
    MAZE_HEIGHT,
    MAZE_WIDTH,
)
from maze import Maze, Position


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_HUD_HEIGHT: int = 80      # pixels reserved for the HUD below the maze grid
_HUD_PADDING: int = 10     # margin inside the HUD area
_HUD_LINE_SPACING: int = 8 # gap between the two HUD text lines
_HUD_FONT_SIZE: int = 22   # point size for HUD text


# ---------------------------------------------------------------------------
# RenderState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderState:
    """Immutable snapshot of game state required to draw one frame.

    The renderer reads every field here and never mutates the environment.

    Attributes:
        maze:                       Maze instance (walls, pellet grids).
        pacman_position:            Current grid cell of Pac-Man.
        ghost_positions:            Current position of each ghost, in order.
        ghost_vulnerable:           Parallel list; True when ghost[i] is
                                    vulnerable (power mode active).
        score:                      Cumulative episode reward so far.
        step:                       Current step count within the episode.
        power_mode_active:          True while the power pellet effect is live.
        power_mode_steps_remaining: Steps remaining in power mode (0 if off).
        episode:                    Episode counter for display (1-indexed).

    Raises:
        ValueError: If ``ghost_positions`` and ``ghost_vulnerable`` differ in
                    length.
    """

    maze: Maze
    pacman_position: Position
    ghost_positions: list[Position]
    ghost_vulnerable: list[bool]
    score: float
    step: int
    power_mode_active: bool
    power_mode_steps_remaining: int
    episode: int = 1

    def __post_init__(self) -> None:
        if len(self.ghost_positions) != len(self.ghost_vulnerable):
            raise ValueError(
                f"ghost_positions length ({len(self.ghost_positions)}) must "
                f"equal ghost_vulnerable length ({len(self.ghost_vulnerable)})."
            )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PacmanRenderer:
    """Pygame-based renderer for the Pac-Man RL environment.

    Pygame is initialised lazily on the first :meth:`render` call, so
    importing this module never requires a live display.

    Attributes:
        cell_size:   Pixels per grid cell.
        fps:         Target frames per second for the display clock.
        render_mode: Rendering mode; only ``"human"`` is currently supported.

    Example::

        renderer = PacmanRenderer()
        renderer.render(state)   # opens pygame window on the first call
        # ... call render() each step ...
        renderer.close()         # shuts down pygame
    """

    def __init__(
        self,
        cell_size: int = CELL_SIZE,
        fps: int = FPS,
        render_mode: str = "human",
    ) -> None:
        """Configure the renderer without starting pygame.

        Args:
            cell_size:   Pixels per grid cell.
            fps:         Target frames per second.
            render_mode: Only ``"human"`` is supported (pygame window).

        Raises:
            ValueError: If *render_mode* is not ``"human"``.
        """
        if render_mode != "human":
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                "Only 'human' is currently supported."
            )

        self.cell_size: int = cell_size
        self.fps: int = fps
        self.render_mode: str = render_mode

        # Derived layout dimensions
        self._grid_w: int = cell_size * MAZE_WIDTH
        self._grid_h: int = cell_size * MAZE_HEIGHT
        self._window_w: int = self._grid_w
        self._window_h: int = self._grid_h + _HUD_HEIGHT

        # Derived entity sizes (scale with cell_size)
        self._pellet_radius: int = max(2, cell_size // 10)
        self._power_radius: int = max(4, cell_size // 5)
        self._entity_radius: int = max(6, cell_size // 3)
        self._eye_radius: int = max(1, cell_size // 12)

        # Pygame objects — None until _init_pygame() is called
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._initialized: bool = False
        self._window_closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the display window in pixels."""
        return (self._window_w, self._window_h)

    @property
    def window_closed(self) -> bool:
        """Whether a window-close event has been received."""
        return self._window_closed

    def render(self, state: RenderState) -> None:
        """Draw one frame from *state* and advance the display clock.

        Pygame is initialised on the first call; subsequent calls reuse the
        existing window.

        Args:
            state: Snapshot of the current game state.
        """
        if not self._initialized:
            if self._window_closed:
                return
            self._init_pygame()

        assert self._surface is not None
        assert self._clock is not None
        assert self._font is not None

        # Process OS events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._window_closed = True
                self.close()
                return

        self._surface.fill(COLOR_BACKGROUND)
        self._draw_maze(state.maze)
        self._draw_pellets(state.maze)
        self._draw_entities(state)
        self._draw_hud(state)

        pygame.display.flip()
        self._clock.tick(self.fps)

    def close(self) -> None:
        """Shut down pygame and release the display window.

        Safe to call even if :meth:`render` was never invoked.
        """
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self._surface = None
            self._clock = None
            self._font = None

    # ------------------------------------------------------------------
    # Pygame initialisation
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        """Initialise pygame, create the window, clock, and font."""
        pygame.init()
        pygame.display.set_caption("Pac-Man RL")
        self._surface = pygame.display.set_mode((self._window_w, self._window_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, _HUD_FONT_SIZE + 4)
        self._initialized = True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Return the :class:`pygame.Rect` covering grid cell (*row*, *col*)."""
        return pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _cell_center(self, row: int, col: int) -> tuple[int, int]:
        """Return the pixel centre of grid cell (*row*, *col*)."""
        return (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2,
        )

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_maze(self, maze: Maze) -> None:
        """Fill all wall cells with COLOR_WALL."""
        assert self._surface is not None
        wall_grid = maze.wall_grid()
        for row in range(maze.height):
            for col in range(maze.width):
                if wall_grid[row, col]:
                    pygame.draw.rect(
                        self._surface,
                        COLOR_WALL,
                        self._cell_rect(row, col),
                    )

    def _draw_pellets(self, maze: Maze) -> None:
        """Draw unconsumed regular pellets and power pellets."""
        assert self._surface is not None
        pellet_grid = maze.pellet_grid()
        power_grid = maze.power_pellet_grid()

        for row in range(maze.height):
            for col in range(maze.width):
                center = self._cell_center(row, col)
                if pellet_grid[row, col]:
                    pygame.draw.circle(
                        self._surface,
                        COLOR_PELLET,
                        center,
                        self._pellet_radius,
                    )
                elif power_grid[row, col]:
                    pygame.draw.circle(
                        self._surface,
                        COLOR_POWER_PELLET,
                        center,
                        self._power_radius,
                    )

    def _draw_entities(self, state: RenderState) -> None:
        """Draw ghosts then Pac-Man (Pac-Man is drawn on top)."""
        assert self._surface is not None

        # --- Ghosts ---
        for pos, vulnerable in zip(state.ghost_positions, state.ghost_vulnerable):
            color = COLOR_GHOST_VULNERABLE if vulnerable else COLOR_GHOST
            center = self._cell_center(pos.row, pos.col)
            pygame.draw.circle(self._surface, color, center, self._entity_radius)

            # Two small white eyes for normal (non-vulnerable) ghosts
            if not vulnerable:
                offset_x = self._entity_radius // 3
                offset_y = self._entity_radius // 4
                left_eye = (center[0] - offset_x, center[1] - offset_y)
                right_eye = (center[0] + offset_x, center[1] - offset_y)
                pygame.draw.circle(self._surface, (255, 255, 255), left_eye, self._eye_radius)
                pygame.draw.circle(self._surface, (255, 255, 255), right_eye, self._eye_radius)

        # --- Pac-Man (drawn last so it appears above ghosts) ---
        pac_center = self._cell_center(
            state.pacman_position.row,
            state.pacman_position.col,
        )
        pygame.draw.circle(
            self._surface,
            COLOR_PACMAN,
            pac_center,
            self._entity_radius,
        )

    def _draw_hud(self, state: RenderState) -> None:
        """Draw the dark HUD bar below the maze with score and status text."""
        assert self._surface is not None
        assert self._font is not None

        hud_top = self._grid_h

        # Dark background for HUD area
        pygame.draw.rect(
            self._surface,
            (20, 20, 20),
            pygame.Rect(0, hud_top, self._window_w, _HUD_HEIGHT),
        )

        # Line 1: Score | Step | Episode
        line1 = (
            f"Score: {state.score:.0f}   "
            f"Step: {state.step}   "
            f"Episode: {state.episode}"
        )
        surf1 = self._font.render(line1, True, COLOR_HUD)
        self._surface.blit(surf1, (_HUD_PADDING, hud_top + _HUD_PADDING))

        # Line 2: Power mode status
        if state.power_mode_active:
            line2 = f"POWER MODE  ({state.power_mode_steps_remaining} steps remaining)"
            color2 = COLOR_POWER_PELLET
        else:
            line2 = "Power Mode: OFF"
            color2 = (100, 100, 100)

        surf2 = self._font.render(line2, True, color2)
        line2_y = hud_top + _HUD_PADDING + _HUD_FONT_SIZE + _HUD_LINE_SPACING
        self._surface.blit(surf2, (_HUD_PADDING, line2_y))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._initialized else "not initialised"
        return (
            f"PacmanRenderer(cell_size={self.cell_size}, fps={self.fps}, "
            f"window={self._window_w}x{self._window_h}, {status})"
        )
