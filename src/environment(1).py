"""
Custom Gymnasium environment for the Pac-Man RL project.

This module owns:
  - PacmanEnvironment: the Gymnasium Env wrapping Maze + Ghost AI.

Observation space (723 float32 values in [0, 1]):
  Channels 0–4 are 12×12 grids flattened in row-major order (144 values each):
    0: wall grid
    1: remaining regular-pellet grid
    2: remaining power-pellet grid
    3: Pac-Man position grid
    4: ghost position grid (all ghosts combined)
  Scalars (3 values):
    720: power_mode_active  (0.0 or 1.0)
    721: power_timer normalised  (steps_remaining / POWER_MODE_DURATION)
    722: pellets_remaining normalised  (remaining / total)

Action space: Discrete(4) — 0=Up, 1=Down, 2=Left, 3=Right.

This module must not perform training, save models, or plot graphs.
Rendering is delegated to renderer.py via a RenderState snapshot.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    Action,
    ACTION_DELTAS,
    CellType,
    MAZE_HEIGHT,
    MAZE_WIDTH,
    MAX_STEPS,
    POWER_MODE_DURATION,
)
from ghost import Ghost
from maze import Maze, Position
from renderer import PacmanRenderer, RenderState
from rewards import calculate_step_reward


# Observation vector length: 5 channels × (H × W) + 3 scalars
_OBS_SIZE: int = MAZE_HEIGHT * MAZE_WIDTH * 5 + 3  # 723 for the 12×12 maze


class PacmanEnvironment(gym.Env):
    """Gymnasium environment for Pac-Man RL training and evaluation.

    The environment manages the full episode lifecycle:

    * ``reset()`` restores the maze, repositions all entities, and returns
      the initial observation.
    * ``step(action)`` moves Pac-Man, checks collisions, moves ghosts,
      computes rewards, and checks termination conditions.
    * ``render()`` delegates to :class:`renderer.PacmanRenderer` when
      ``render_mode="human"``.
    * ``close()`` shuts down the renderer if one was created.

    Termination (``terminated=True``):
      - All pellets consumed (maze cleared).
      - Pac-Man caught by a normal ghost.

    Truncation (``truncated=True``):
      - Step count reaches ``max_steps``.

    Attributes:
        render_mode: ``"human"`` or ``None``.

    Example::

        env = PacmanEnvironment()
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        env.close()
    """

    metadata: dict = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = MAX_STEPS,
        maze_layout: list[list[int]] | None = None,
    ) -> None:
        """Initialise the environment.

        Args:
            render_mode:  ``"human"`` to open a pygame window, or ``None``
                          (default) for headless training.
            max_steps:    Maximum steps per episode before truncation.
            maze_layout:  Optional custom maze grid (list[list[int]]).
                          ``None`` uses the default layout from config.py.

        Raises:
            ValueError: If ``render_mode`` is not in ``(None, "human")``.
        """
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                "Supported: None, 'human'."
            )

        self.render_mode = render_mode
        self._max_steps = max_steps

        # Game objects — persistent across episodes; reset() restores state
        self._maze = Maze(layout=maze_layout)
        self._ghosts: list[Ghost] = [
            Ghost(spawn=pos) for pos in self._maze.ghost_spawns()
        ]

        # Observation size depends on the actual maze dimensions
        obs_size: int = self._maze.height * self._maze.width * 5 + 3

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(Action))

        # Runtime state — initialised by reset()
        self._pacman_pos: Position | None = None
        self._power_mode_active: bool = False
        self._power_mode_steps_remaining: int = 0
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._episode: int = 0

        # Renderer — created lazily on the first render() call
        self._renderer: PacmanRenderer | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to the initial state for a new episode.

        Args:
            seed:    Optional integer seed for ``self.np_random``.
            options: Unused; accepted for API compliance.

        Returns:
            A ``(observation, info)`` tuple.
        """
        super().reset(seed=seed)  # initialises self.np_random

        self._maze.reset()
        self._pacman_pos = self._maze.pacman_spawn()
        for ghost in self._ghosts:
            ghost.reset()

        self._power_mode_active = False
        self._power_mode_steps_remaining = 0
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode += 1

        return self._build_observation(), self._build_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one step.

        Step sequence:
          1. Compute candidate Pac-Man position from *action*.
          2. Wall check — Pac-Man stays if move is invalid (``hit_wall``).
          3. Move Pac-Man.
          4. Consume pellet at new position (regular or power pellet).
          5. Collision check at Pac-Man's new position (pre-ghost-move).
          6. Decrement power-mode timer; restore ghosts when timer hits 0.
          7. Move all ghosts.
          8. Collision check again (post-ghost-move).
          9. Maze-cleared check.
          10. Compute reward and increment step counter.

        Args:
            action: Integer action in ``{0, 1, 2, 3}``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``

        Raises:
            RuntimeError: If called before :meth:`reset`.
        """
        if self._pacman_pos is None:
            raise RuntimeError("reset() must be called before step().")

        terminated = False
        truncated = False

        hit_wall = False
        pellet_eaten = False
        power_pellet_eaten = False
        ghosts_eaten = 0
        caught = False
        cleared = False

        # 1–2. Move Pac-Man (or stay on wall)
        delta = ACTION_DELTAS[Action(action)]
        candidate = self._pacman_pos + delta
        if not self._maze.is_walkable(candidate):
            hit_wall = True
            candidate = self._pacman_pos

        # 3. Apply move
        self._pacman_pos = candidate

        # 4. Pellet consumption
        power_activated_this_step = False
        consumed = self._maze.consume_pellet(self._pacman_pos)
        if consumed == CellType.PELLET:
            pellet_eaten = True
        elif consumed == CellType.POWER_PELLET:
            power_pellet_eaten = True
            power_activated_this_step = True
            self._power_mode_active = True
            self._power_mode_steps_remaining = POWER_MODE_DURATION
            for ghost in self._ghosts:
                ghost.make_vulnerable()

        # 5. Collision: Pac-Man moved into a ghost
        _caught, _eaten = self._check_collisions()
        if _caught:
            caught = True
            terminated = True
        ghosts_eaten += _eaten

        # 6. Power-mode timer (not decremented on the step it was activated,
        #    so the full POWER_MODE_DURATION steps are available)
        if not terminated and self._power_mode_active and not power_activated_this_step:
            self._power_mode_steps_remaining -= 1
            if self._power_mode_steps_remaining <= 0:
                self._power_mode_active = False
                self._power_mode_steps_remaining = 0
                for ghost in self._ghosts:
                    ghost.restore_normal()

        # 7. Ghost movement
        if not terminated:
            for ghost in self._ghosts:
                ghost.move(self._maze, self._pacman_pos, self.np_random)

        # 8. Collision: a ghost moved into Pac-Man
        if not terminated:
            _caught2, _eaten2 = self._check_collisions()
            if _caught2:
                caught = True
                terminated = True
            ghosts_eaten += _eaten2

        # 9. Maze-cleared
        if not terminated and self._maze.is_cleared():
            cleared = True
            terminated = True

        # 10. Reward + step counter
        reward = calculate_step_reward(
            hit_wall=hit_wall,
            pellet_eaten=pellet_eaten,
            power_pellet_eaten=power_pellet_eaten,
            ghosts_eaten=ghosts_eaten,
            caught=caught,
            cleared=cleared,
        )
        self._step_count += 1
        if not terminated and self._step_count >= self._max_steps:
            truncated = True
        self._episode_reward += reward

        return (
            self._build_observation(),
            float(reward),
            terminated,
            truncated,
            self._build_info(),
        )

    def render(self) -> None:
        """Draw the current state via :class:`renderer.PacmanRenderer`.

        No-op when ``render_mode`` is ``None`` or before ``reset()``.
        Pygame is initialised lazily on the first call.
        """
        if self.render_mode != "human" or self._pacman_pos is None:
            return

        if self._renderer is None:
            self._renderer = PacmanRenderer(render_mode="human")

        self._renderer.render(
            RenderState(
                maze=self._maze,
                pacman_position=self._pacman_pos,
                ghost_positions=[g.position for g in self._ghosts],
                ghost_vulnerable=[g.is_vulnerable for g in self._ghosts],
                score=self._episode_reward,
                step=self._step_count,
                power_mode_active=self._power_mode_active,
                power_mode_steps_remaining=self._power_mode_steps_remaining,
                episode=self._episode,
            )
        )

    def close(self) -> None:
        """Shut down the renderer and release resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_collisions(self) -> tuple[bool, int]:
        """Check whether any ghost occupies Pac-Man's current cell.

        Vulnerable ghosts at Pac-Man's position are immediately respawned
        (eaten). Normal ghosts at Pac-Man's position cause termination.

        Returns:
            ``(caught, ghosts_eaten)`` where *caught* is ``True`` if a
            normal ghost was at Pac-Man's cell, and *ghosts_eaten* is the
            count of vulnerable ghosts that were eaten this check.
        """
        assert self._pacman_pos is not None
        caught = False
        ghosts_eaten = 0

        for ghost in self._ghosts:
            if ghost.position == self._pacman_pos:
                if ghost.is_vulnerable:
                    ghost.respawn()
                    ghosts_eaten += 1
                else:
                    caught = True

        return caught, ghosts_eaten

    def _build_observation(self) -> np.ndarray:
        """Build the 723-element flat float32 observation vector.

        Returns:
            Array of shape ``(_OBS_SIZE,)`` with values in ``[0.0, 1.0]``.
        """
        assert self._pacman_pos is not None

        H, W = self._maze.height, self._maze.width

        pacman_ch = np.zeros((H, W), dtype=np.float32)
        pacman_ch[self._pacman_pos.row, self._pacman_pos.col] = 1.0

        ghost_ch = np.zeros((H, W), dtype=np.float32)
        for ghost in self._ghosts:
            ghost_ch[ghost.position.row, ghost.position.col] = 1.0

        power_active = np.float32(1.0 if self._power_mode_active else 0.0)
        power_timer = np.float32(
            self._power_mode_steps_remaining / POWER_MODE_DURATION
            if self._power_mode_active else 0.0
        )
        total = self._maze.total_pellets
        pellets_norm = np.float32(
            self._maze.pellets_remaining() / total if total > 0 else 0.0
        )

        return np.concatenate([
            self._maze.wall_grid().ravel(),
            self._maze.pellet_grid().ravel(),
            self._maze.power_pellet_grid().ravel(),
            pacman_ch.ravel(),
            ghost_ch.ravel(),
            [power_active, power_timer, pellets_norm],
        ]).astype(np.float32)

    def _build_info(self) -> dict:
        """Build the info dictionary returned by ``reset`` and ``step``."""
        pos = self._pacman_pos
        return {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "pellets_remaining": self._maze.pellets_remaining(),
            "total_pellets": self._maze.total_pellets,
            "completion_fraction": self._maze.completion_fraction,
            "power_mode_active": self._power_mode_active,
            "power_mode_steps_remaining": self._power_mode_steps_remaining,
            "pacman_position": (pos.row, pos.col) if pos is not None else None,
            "ghost_positions": [
                (g.position.row, g.position.col) for g in self._ghosts
            ],
        }

    def __repr__(self) -> str:
        return (
            f"PacmanEnvironment("
            f"obs={_OBS_SIZE}, actions={len(Action)}, "
            f"max_steps={self._max_steps})"
        )
