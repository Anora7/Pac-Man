"""
Pygame renderer for the PAC-MAN RL environment.

Features:
  - Yellow Pac-Man with animated directional mouth.
  - Distinct ghosts (Blinky, Pinky, Inky, Clyde) with rounded tops,
    wavy bottom skirts, and directional eyes.
  - Light-blue frightened ghosts with navy eyes that flash white
    during the final countdown of power mode.
  - High-contrast walls and dark background.
  - Multi-line HUD displaying score, episode, step, pellets, and ghost modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import pygame

from config import ACTION_DELTAS, Action
from v2_config import (
    GOOGLE_CELL_SIZE,
    GOOGLE_COLOR_BACKGROUND,
    GOOGLE_COLOR_HUD,
    GOOGLE_COLOR_PACMAN,
    GOOGLE_COLOR_PELLET,
    GOOGLE_COLOR_POWER_PELLET,
    GOOGLE_COLOR_WALL,
    GOOGLE_FPS,
    GHOST_COLORS,
    GhostIdentity,
)
from v2_ghost import GhostMode, GoogleGhost
from v2_maze import GoogleMaze
from maze import Position

_HUD_HEIGHT: int = 90
_HUD_PADDING: int = 12
_HUD_FONT_SIZE: int = 20

_FRIGHTENED_BODY = (140, 200, 255)   # light blue
_FRIGHTENED_FLASH = (255, 255, 255)  # white, power mode ending
_FRIGHTENED_EYE = (20, 30, 90)       # dark navy eyes


@dataclass(frozen=True)
class GoogleRenderState:
    """Immutable snapshot of PAC-MAN state for rendering one frame."""
    maze: GoogleMaze
    pacman_position: Position
    pacman_action: Action | None
    ghosts: list[GoogleGhost]
    score: float
    step: int
    power_mode_active: bool
    power_mode_steps_remaining: int
    episode: int = 1


class GoogleRenderer:
    """Pygame-based visual renderer for the PAC-MAN environment."""

    def __init__(
        self,
        cell_size: int = GOOGLE_CELL_SIZE,
        fps: int = GOOGLE_FPS,
        render_mode: str = "human",
    ) -> None:
        if render_mode != "human":
            raise ValueError(f"Unsupported render_mode '{render_mode}'. Must be 'human'.")

        self.cell_size: int = cell_size
        self.fps: int = fps
        self.render_mode: str = render_mode

        # Derived layout
        self.grid_w: int = cell_size * 16
        self.grid_h: int = cell_size * 16
        self.window_w: int = self.grid_w
        self.window_h: int = self.grid_h + _HUD_HEIGHT

        # Entity sizing
        self._pellet_radius: int = max(2, cell_size // 10)
        self._power_radius: int = max(5, cell_size // 4)
        self._entity_radius: int = max(7, cell_size // 2 - 2)
        self._eye_radius: int = max(2, cell_size // 8)
        self._pupil_radius: int = max(1, cell_size // 16)

        # Pygame resources
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._initialized: bool = False
        self._window_closed: bool = False
        self._mouth_phase: int = 0
        self._paused: bool = False

    @property
    def window_size(self) -> tuple[int, int]:
        return (self.window_w, self.window_h)

    @property
    def window_closed(self) -> bool:
        return self._window_closed

    def _init_pygame(self) -> None:
        """Initialize pygame display, clock, and fonts."""
        pygame.init()
        pygame.display.set_caption("Pac-Man RL")
        self._surface = pygame.display.set_mode((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, _HUD_FONT_SIZE + 4)
        self._small_font = pygame.font.Font(None, _HUD_FONT_SIZE)
        self._initialized = True
    
    def _handle_events(self) -> bool:
        """Process window and key events. Returns False if the window closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._window_closed = True
                self.close()
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    self.fps = min(60, self.fps + 1)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    self.fps = max(1, self.fps - 1)
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
        return True

    def render(self, state: GoogleRenderState) -> None:
        """Render a single frame snapshot."""
        if not self._initialized:
            if self._window_closed:
                return
            self._init_pygame()

        assert self._surface is not None
        assert self._clock is not None

        if not self._handle_events():
            return

        self._mouth_phase = (self._mouth_phase + 1) % 8

        self._surface.fill(GOOGLE_COLOR_BACKGROUND)
        self._draw_maze(state.maze)
        self._draw_pellets(state.maze)
        self._draw_ghosts(state)
        self._draw_pacman(state)
        self._draw_hud(state)

        pygame.display.flip()
        while self._paused:
            if not self._handle_events():
                return
            self._draw_hud(state)
            pygame.display.flip()
            self._clock.tick(30)
        self._clock.tick(self.fps)

    def close(self) -> None:
        """Close display window."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self._surface = None
            self._clock = None
            self._font = None
            self._small_font = None

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _cell_center(self, row: int, col: int) -> tuple[int, int]:
        return (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2,
        )

    def _draw_maze(self, maze: GoogleMaze) -> None:
        """Draw walls and ghost house door."""
        assert self._surface is not None
        wall_grid = maze.wall_grid()

        for r in range(maze.height):
            for c in range(maze.width):
                if wall_grid[r, c]:
                    rect = self._cell_rect(r, c)
                    pygame.draw.rect(self._surface, GOOGLE_COLOR_WALL, rect)
                    pygame.draw.rect(self._surface, (55, 70, 225), rect, 1)

        doors = maze.ghost_house_doors()
        for d in doors:
            rect = self._cell_rect(d.row, d.col)
            door_bar = pygame.Rect(rect.left, rect.centery - 2, rect.width, 4)
            pygame.draw.rect(self._surface, (255, 180, 200), door_bar)

    def _draw_pellets(self, maze: GoogleMaze) -> None:
        """Draw regular pellets and power pellets."""
        assert self._surface is not None
        pellet_grid = maze.pellet_grid()
        power_grid = maze.power_pellet_grid()

        for r in range(maze.height):
            for c in range(maze.width):
                center = self._cell_center(r, c)
                if pellet_grid[r, c]:
                    pygame.draw.circle(
                        self._surface,
                        GOOGLE_COLOR_PELLET,
                        center,
                        self._pellet_radius,
                    )
                elif power_grid[r, c]:
                    pygame.draw.circle(
                        self._surface,
                        GOOGLE_COLOR_POWER_PELLET,
                        center,
                        self._power_radius,
                    )
                    pygame.draw.circle(
                        self._surface,
                        (255, 255, 220),
                        center,
                        max(1, self._power_radius // 2),
                    )

    def _draw_pacman(self, state: GoogleRenderState) -> None:
        """Draw animated yellow Pac-Man with directional open/close mouth."""
        assert self._surface is not None
        center = self._cell_center(state.pacman_position.row, state.pacman_position.col)
        r = self._entity_radius

        angle_map = {
            Action.RIGHT: 0.0,
            Action.DOWN: math.pi / 2,
            Action.LEFT: math.pi,
            Action.UP: 3 * math.pi / 2,
        }
        facing_angle = angle_map.get(state.pacman_action, 0.0)

        mouth_opening = abs(self._mouth_phase - 4) / 4.0 * 0.45

        if mouth_opening < 0.05:
            pygame.draw.circle(self._surface, GOOGLE_COLOR_PACMAN, center, r)
        else:
            start_angle = facing_angle + mouth_opening
            end_angle = facing_angle + 2 * math.pi - mouth_opening
            points = [center]
            num_segments = 24
            span = end_angle - start_angle
            for i in range(num_segments + 1):
                theta = start_angle + (i / num_segments) * span
                px = center[0] + r * math.cos(theta)
                py = center[1] + r * math.sin(theta)
                points.append((px, py))
            pygame.draw.polygon(self._surface, GOOGLE_COLOR_PACMAN, points)

    def _draw_ghosts(self, state: GoogleRenderState) -> None:
        """Draw ghosts with rounded heads, wavy skirts, and directional eyes."""
        assert self._surface is not None

        for ghost in state.ghosts:
            center = self._cell_center(ghost.position.row, ghost.position.col)
            r = self._entity_radius

            normal_color = GHOST_COLORS.get(ghost.identity, (200, 50, 50))
            if ghost.is_vulnerable:
                # Light blue, flashing white in the last 8 steps of power mode
                if state.power_mode_steps_remaining <= 8 and (state.step % 2 == 0):
                    body_color = _FRIGHTENED_FLASH
                else:
                    body_color = _FRIGHTENED_BODY
            else:
                body_color = normal_color

            cx, cy = center
            bot_y = cy + r
            left_x = cx - r

            # 1. Rounded head
            pygame.draw.circle(self._surface, body_color, (cx, cy - r // 4), r)

            # 2. Main body torso
            torso_rect = pygame.Rect(left_x, cy - r // 4, 2 * r, r + r // 4)
            pygame.draw.rect(self._surface, body_color, torso_rect)

            # 3. Wavy bottom skirt
            skirt_width = (2 * r) / 3.0
            for i in range(3):
                x0 = left_x + i * skirt_width
                x1 = x0 + skirt_width / 2.0
                x2 = x0 + skirt_width
                y_peak = bot_y
                y_notch = bot_y - (r // 3)
                notch_poly = [(x0, y_peak), (x1, y_notch), (x2, y_peak)]
                pygame.draw.polygon(self._surface, GOOGLE_COLOR_BACKGROUND, notch_poly)

            # 4. Eyes and pupils
            eye_offset_x = r // 2
            eye_offset_y = r // 3
            left_eye_center = (cx - eye_offset_x, cy - eye_offset_y)
            right_eye_center = (cx + eye_offset_x, cy - eye_offset_y)

            if ghost.is_vulnerable:
                # Frightened face: dark navy eyes (not red)
                pygame.draw.circle(self._surface, _FRIGHTENED_EYE, left_eye_center, self._eye_radius)
                pygame.draw.circle(self._surface, _FRIGHTENED_EYE, right_eye_center, self._eye_radius)
            else:
                pygame.draw.circle(self._surface, (255, 255, 255), left_eye_center, self._eye_radius)
                pygame.draw.circle(self._surface, (255, 255, 255), right_eye_center, self._eye_radius)

                p_dx, p_dy = 0, 0
                if ghost.last_action == Action.UP:
                    p_dy = -self._pupil_radius
                elif ghost.last_action == Action.DOWN:
                    p_dy = self._pupil_radius
                elif ghost.last_action == Action.LEFT:
                    p_dx = -self._pupil_radius
                elif ghost.last_action == Action.RIGHT:
                    p_dx = self._pupil_radius

                left_pupil = (left_eye_center[0] + p_dx, left_eye_center[1] + p_dy)
                right_pupil = (right_eye_center[0] + p_dx, right_eye_center[1] + p_dy)
                pygame.draw.circle(self._surface, (20, 20, 100), left_pupil, self._pupil_radius)
                pygame.draw.circle(self._surface, (20, 20, 100), right_pupil, self._pupil_radius)

    def _draw_hud(self, state: GoogleRenderState) -> None:
        """Draw status HUD bar below the maze."""
        assert self._surface is not None
        assert self._font is not None
        assert self._small_font is not None

        hud_top = self.grid_h
        pygame.draw.rect(
            self._surface,
            (18, 18, 25),
            pygame.Rect(0, hud_top, self.window_w, _HUD_HEIGHT),
        )
        pygame.draw.line(self._surface, GOOGLE_COLOR_WALL, (0, hud_top), (self.window_w, hud_top), 2)

        line1 = (
            f"Score: {state.score:4.0f}  "
            f"Step: {state.step:4d}  "
            f"Pellets: {state.maze.pellets_remaining():3d}  "
            f"Ep: {state.episode}"
        )
        s1 = self._font.render(line1, True, GOOGLE_COLOR_HUD)
        self._surface.blit(s1, (_HUD_PADDING, hud_top + 10))

        if state.power_mode_active:
            status_text = f"POWER MODE: {state.power_mode_steps_remaining} steps left!"
            status_color = GOOGLE_COLOR_POWER_PELLET
        else:
            released_count = sum(1 for g in state.ghosts if g.is_released)
            mode_name = state.ghosts[0].current_mode.name if state.ghosts else "NORMAL"
            status_text = f"Ghosts Active: {released_count}/4   Mode: {mode_name}"
            status_color = (180, 180, 210)

        s2 = self._small_font.render(status_text, True, status_color)
        self._surface.blit(s2, (_HUD_PADDING, hud_top + 45))

        pause_note = "  [PAUSED]" if self._paused else ""
        s3 = self._small_font.render(
            f"Speed: {self.fps} fps (UP/DOWN to change, SPACE to pause){pause_note}",
            True,
            (150, 150, 180),
        )
        self._surface.blit(s3, (_HUD_PADDING, hud_top + 67))