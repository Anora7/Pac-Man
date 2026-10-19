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

from collections import deque
from collections.abc import Sequence

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
    FPS,
    OBSERVATION_CONFIG,
    REWARDS,
    RewardConfig,
    ObservationConfig,
    GHOST_DIRECTION_FEATURES,
    GHOST_OBSERVATION_FEATURES,
    PACMAN_ACTION_FEATURES,
    NEAREST_PELLET_DIRECTION_FEATURES,
    NEAREST_PELLET_FEATURES,
    NEAREST_GHOST_FEATURES,
    GHOST_PROXIMITY_DISTANCE,
    POWER_MODE_DURATION,
)
from ghost import Ghost
from ghost_observations import build_ghost_observation
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

    metadata: dict = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = MAX_STEPS,
        maze_layout: list[list[int]] | None = None,
        observation_config: ObservationConfig = OBSERVATION_CONFIG,
        render_fps: int = FPS,
        reward_config: RewardConfig = REWARDS,
        ghost_count: int | None = None,
    ) -> None:
        """Initialise the environment.

        Args:
            render_mode:  ``"human"`` to open a pygame window, or ``None``
                          (default) for headless training.
            max_steps:    Maximum steps per episode before truncation.
            maze_layout:  Optional custom maze grid (list[list[int]]).
                          ``None`` uses the default layout from config.py.
            observation_config: Optional appended features. Its defaults
                          preserve the original 723-element observation.
            render_fps:  Target playback rate when rendering.
            reward_config: Reward values; defaults preserve baseline rewards.
            ghost_count:  Number of ghosts; ``None`` uses all maze spawns.

        Raises:
            ValueError: If ``render_mode`` is not in ``(None, "human")``.
        """
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                "Supported: None, 'human'."
            )
        if render_fps < 1:
            raise ValueError("render_fps must be a positive integer.")

        self.render_mode = render_mode
        self._max_steps = max_steps
        self._render_fps = render_fps
        self._observation_config = observation_config
        self._reward_config = reward_config

        # Game objects — persistent across episodes; reset() restores state
        self._maze = Maze(layout=maze_layout)
        ghost_spawns = self._maze.ghost_spawns()
        selected_ghost_count = (
            len(ghost_spawns) if ghost_count is None else ghost_count
        )
        if not 1 <= selected_ghost_count <= len(ghost_spawns):
            raise ValueError(
                f"ghost_count must be between 1 and {len(ghost_spawns)}."
            )
        self._ghosts: list[Ghost] = [
            Ghost(spawn=pos) for pos in ghost_spawns[:selected_ghost_count]
        ]
        self._ghost_last_directions: list[int | None] = [
            None for _ in self._ghosts
        ]

        # Observation size depends on the actual maze dimensions
        obs_size: int = self._maze.height * self._maze.width * 5 + 3
        if observation_config.include_ghost_direction:
            obs_size += len(self._ghosts) * GHOST_OBSERVATION_FEATURES
        if observation_config.include_nearest_pellet:
            obs_size += NEAREST_PELLET_FEATURES
        if observation_config.include_nearest_ghost:
            obs_size += NEAREST_GHOST_FEATURES
        if (
            observation_config.include_previous_pacman_action
            or observation_config.include_previous_action
        ):
            obs_size += PACMAN_ACTION_FEATURES
        self._observation_size = obs_size

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
        self._previous_pacman_action: int | None = None
        self._recent_pacman_positions: deque[Position] = deque(maxlen=10)

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
        self._ghost_last_directions = [None for _ in self._ghosts]

        self._power_mode_active = False
        self._power_mode_steps_remaining = 0
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode += 1
        self._previous_pacman_action = None
        self._recent_pacman_positions.clear()
        self._recent_pacman_positions.append(self._pacman_pos)
        self._last_step_events = {
            "pellet_eaten": False,
            "power_pellet_eaten": False,
            "ghosts_eaten": 0,
            "caught": False,
            "cleared": False,
        }

        return self._build_observation(), self._build_info()

    def step(
        self,
        action: int,
        *,
        ghost_actions: Sequence[int | None] | None = None,
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
            ghost_actions: Optional per-ghost actions used by the ghost-side
                           training environment. ``None`` entries retain the
                           configured policy for that ghost.

        Returns:
            ``(observation, reward, terminated, truncated, info)``

        Raises:
            RuntimeError: If called before :meth:`reset`.
        """
        if self._pacman_pos is None:
            raise RuntimeError("reset() must be called before step().")
        if ghost_actions is not None and len(ghost_actions) != len(self._ghosts):
            raise ValueError(
                f"Expected {len(self._ghosts)} ghost actions, got "
                f"{len(ghost_actions)}."
            )

        terminated = False
        truncated = False

        hit_wall = False
        pellet_distance_decreased = False
        pellet_distance_increased = False
        ghost_distance_decreased = False
        pellet_eaten = False
        power_pellet_eaten = False
        ghosts_eaten = 0
        caught = False
        cleared = False

        # 1–2. Move Pac-Man (or stay on wall)
        previous_position = self._pacman_pos
        ghost_distance_before = (
            self._nearest_non_vulnerable_ghost_distance(previous_position)
            if self._reward_config.GHOST_DISTANCE_DECREASE != 0.0
            else None
        )
        action = Action(action)
        delta = ACTION_DELTAS[action]
        candidate = previous_position + delta
        if not self._maze.is_walkable(candidate):
            hit_wall = True
            candidate = previous_position

        if (
            self._reward_config.PELLET_DISTANCE_DECREASE != 0.0
            or self._reward_config.PELLET_DISTANCE_INCREASE != 0.0
        ):
            _, distance_before = self._maze.nearest_pellet_path(
                previous_position
            )
            _, distance_after = self._maze.nearest_pellet_path(candidate)
            if distance_before is not None and distance_after is not None:
                pellet_distance_decreased = distance_after < distance_before
                pellet_distance_increased = distance_after > distance_before

        # 3. Apply move
        revisited_cell = (
            candidate != previous_position
            and candidate in self._recent_pacman_positions
        )
        self._pacman_pos = candidate
        if candidate != previous_position:
            self._recent_pacman_positions.append(candidate)
        self._previous_pacman_action = int(action)

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

        if (
            ghost_distance_before is not None
            and self._reward_config.GHOST_DISTANCE_DECREASE != 0.0
        ):
            ghost_distance_after = self._nearest_non_vulnerable_ghost_distance(
                self._pacman_pos
            )
            ghost_distance_decreased = (
                ghost_distance_after is not None
                and ghost_distance_after <= GHOST_PROXIMITY_DISTANCE
                and ghost_distance_after < ghost_distance_before
            )

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
            for index, ghost in enumerate(self._ghosts):
                previous_position = ghost.position
                override = (
                    None if ghost_actions is None else ghost_actions[index]
                )
                ghost.move(
                    self._maze,
                    self._pacman_pos,
                    self.np_random,
                    observation=build_ghost_observation(
                        self._maze,
                        [item.position for item in self._ghosts],
                        index,
                        self._pacman_pos,
                        self._power_mode_active,
                    ),
                    action_override=override,
                )
                ghost_direction = _direction_index(
                    previous_position,
                    ghost.position,
                )
                if ghost_direction is not None:
                    self._ghost_last_directions[index] = ghost_direction

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
            pellet_distance_decreased=pellet_distance_decreased,
            pellet_distance_increased=pellet_distance_increased,
            revisited_cell=revisited_cell,
            ghost_distance_decreased=ghost_distance_decreased,
            reward_config=self._reward_config,
        )
        self._step_count += 1
        if not terminated and self._step_count >= self._max_steps:
            truncated = True
        self._episode_reward += reward
        self._last_step_events = {
            "pellet_eaten": pellet_eaten,
            "power_pellet_eaten": power_pellet_eaten,
            "ghosts_eaten": ghosts_eaten,
            "caught": caught,
            "cleared": cleared,
        }

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
            self._renderer = PacmanRenderer(
                render_mode="human",
                fps=self._render_fps,
            )

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

    def _nearest_non_vulnerable_ghost_distance(
        self,
        position: Position,
    ) -> int | None:
        """Return the closest reachable normal ghost's BFS distance."""
        pacman_distances = self._maze.distances_from(position)
        distances = [
            pacman_distances[ghost.position]
            for ghost in self._ghosts
            if not ghost.is_vulnerable and ghost.position in pacman_distances
        ]
        return min(distances) if distances else None

    def _nearest_ghost_features(self) -> np.ndarray:
        """Encode escape direction, distance, and last move of nearest ghost."""
        assert self._pacman_pos is not None
        features = np.zeros(NEAREST_GHOST_FEATURES, dtype=np.float32)
        pacman_distances = self._maze.distances_from(self._pacman_pos)
        candidates = [
            (pacman_distances[ghost.position], index, ghost)
            for index, ghost in enumerate(self._ghosts)
            if (
                not ghost.is_vulnerable
                and ghost.position in pacman_distances
            )
        ]
        if not candidates:
            return features

        distance, ghost_index, ghost = min(candidates, key=lambda item: item[0])
        ghost_distances = self._maze.distances_from(ghost.position)
        farther_steps = [
            (ghost_distances[neighbour], int(action))
            for action, delta in ACTION_DELTAS.items()
            if self._maze.is_walkable(
                neighbour := self._pacman_pos + delta
            )
            and neighbour in ghost_distances
            and ghost_distances[neighbour] > distance
        ]
        if farther_steps:
            _, escape_action = max(
                farther_steps,
                key=lambda item: item[0],
            )
            features[escape_action] = 1.0

        max_distance = max(1, self._maze.height * self._maze.width - 1)
        features[4] = np.float32(distance / max_distance)
        last_direction = self._ghost_last_directions[ghost_index]
        if last_direction is not None:
            features[5 + last_direction] = 1.0
        return features

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

        observation_parts = [
            self._maze.wall_grid().ravel(),
            self._maze.pellet_grid().ravel(),
            self._maze.power_pellet_grid().ravel(),
            pacman_ch.ravel(),
            ghost_ch.ravel(),
            [power_active, power_timer, pellets_norm],
        ]

        if self._observation_config.include_ghost_direction:
            for ghost, direction in zip(
                self._ghosts,
                self._ghost_last_directions,
            ):
                direction_features = np.zeros(
                    GHOST_DIRECTION_FEATURES,
                    dtype=np.float32,
                )
                if direction is not None:
                    direction_features[direction] = 1.0
                observation_parts.append(direction_features)
                observation_parts.append(
                    np.asarray([float(ghost.is_vulnerable)], dtype=np.float32)
                )

        if self._observation_config.include_nearest_pellet:
            direction, distance = self._maze.nearest_pellet_path(
                self._pacman_pos
            )
            direction_features = np.zeros(
                NEAREST_PELLET_DIRECTION_FEATURES,
                dtype=np.float32,
            )
            if direction is not None:
                direction_features[direction] = 1.0
            normalized_distance = (
                distance / max(1, H * W - 1)
                if distance is not None else 0.0
            )
            observation_parts.append(
                np.concatenate((
                    direction_features,
                    np.asarray([normalized_distance], dtype=np.float32),
                ))
            )

        if self._observation_config.include_nearest_ghost:
            observation_parts.append(self._nearest_ghost_features())

        if (
            self._observation_config.include_previous_pacman_action
            or self._observation_config.include_previous_action
        ):
            action_features = np.zeros(PACMAN_ACTION_FEATURES, dtype=np.float32)
            if self._previous_pacman_action is not None:
                action_features[self._previous_pacman_action] = 1.0
            observation_parts.append(action_features)

        return np.concatenate(observation_parts).astype(np.float32)

    def _build_info(self) -> dict:
        """Build the info dictionary returned by ``reset`` and ``step``."""
        pos = self._pacman_pos
        return {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "pellets_remaining": self._maze.pellets_remaining(),
            "total_pellets": self._maze.total_pellets,
            "completion_fraction": self._maze.completion_fraction,
            "is_success": self._maze.is_cleared(),
            "power_mode_active": self._power_mode_active,
            "power_mode_steps_remaining": self._power_mode_steps_remaining,
            "pacman_position": (pos.row, pos.col) if pos is not None else None,
            "ghost_positions": [
                (g.position.row, g.position.col) for g in self._ghosts
            ],
            "last_step_events": getattr(
                self,
                "_last_step_events",
                {
                    "pellet_eaten": False,
                    "power_pellet_eaten": False,
                    "ghosts_eaten": 0,
                    "caught": False,
                    "cleared": False,
                },
            ).copy(),
        }

    @property
    def maze(self) -> Maze:
        """Return the live maze model for cooperating environment wrappers."""
        return self._maze

    @property
    def pacman_position(self) -> Position:
        """Return Pac-Man's current position after reset."""
        if self._pacman_pos is None:
            raise RuntimeError("reset() must be called before reading state.")
        return self._pacman_pos

    @property
    def ghosts(self) -> tuple[Ghost, ...]:
        """Return a read-only sequence view of the episode's ghosts."""
        return tuple(self._ghosts)

    @property
    def power_mode_active(self) -> bool:
        """Whether the current episode is in power mode."""
        return self._power_mode_active

    @property
    def power_mode_steps_remaining(self) -> int:
        """Number of remaining power-mode steps."""
        return self._power_mode_steps_remaining

    @property
    def window_closed(self) -> bool:
        """Whether the user closed the rendered Pygame window."""
        return self._renderer is not None and self._renderer.window_closed

    def __repr__(self) -> str:
        return (
            f"PacmanEnvironment("
            f"obs={self._observation_size}, actions={len(Action)}, "
            f"max_steps={self._max_steps})"
        )


def _direction_index(previous: Position, current: Position) -> int | None:
    """Return the Action index for a one-cell move, or None if unchanged."""
    delta = (current.row - previous.row, current.col - previous.col)
    return next(
        (int(action) for action, action_delta in ACTION_DELTAS.items()
         if action_delta == delta),
        None,
    )
