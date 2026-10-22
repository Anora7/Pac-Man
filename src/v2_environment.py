"""
Custom Gymnasium environment for the Google Pac-Man RL project.

Features:
  - 16x16 board (configurable) with 4 rule-based ghosts.
  - Scaled observation space with 5 grid channels + scalar statuses +
    best-performing features (nearest pellet, previous action, nearest ghost,
    ghost directions/vulnerabilities).
  - Configurable power mode duration, rewards, ghost release schedule,
    and curriculum chase strength.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    ACTION_DELTAS,
    Action,
    CellType,
    RewardConfig,
)
from v2_config import (
    GOOGLE_COLOR_BACKGROUND,
    GOOGLE_FPS,
    GOOGLE_MAX_STEPS,
    GOOGLE_MAZE_HEIGHT,
    GOOGLE_MAZE_WIDTH,
    GOOGLE_POWER_MODE_DURATION,
    GOOGLE_REWARDS,
    GhostIdentity,
)
from v2_ghost import GoogleGhost, create_google_ghosts
from v2_maze import GoogleMaze
from maze import Position
from rewards import calculate_step_reward

if TYPE_CHECKING:
    from v2_render import GoogleRenderer


class GoogleEnvironment(gym.Env):
    """Gymnasium environment for Google Pac-Man RL training and evaluation."""

    metadata = {"render_modes": ["human"], "render_fps": GOOGLE_FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = GOOGLE_MAX_STEPS,
        maze_layout: list[list[int]] | None = None,
        chase_strength: float = 1.0,
        active_ghost_count: int = 4,
        reward_config: RewardConfig = GOOGLE_REWARDS,
        power_mode_duration: int = GOOGLE_POWER_MODE_DURATION,
        render_fps: int = GOOGLE_FPS,
        ghost_release_schedule: Sequence[int] | None = None,
        ghost_release_interval: int | None = None,
    ) -> None:
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode '{render_mode}'. Must be None or 'human'.")
        if render_fps < 1:
            raise ValueError("render_fps must be positive.")
        if not 0.0 <= chase_strength <= 1.0:
            raise ValueError("chase_strength must be between 0.0 and 1.0.")
        if not 1 <= active_ghost_count <= 4:
            raise ValueError("active_ghost_count must be between 1 and 4.")

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.chase_strength = chase_strength
        self.active_ghost_count = active_ghost_count
        self.reward_config = reward_config
        self.power_mode_duration = power_mode_duration
        self.render_fps = render_fps
        self.ghost_release_schedule = ghost_release_schedule
        self.ghost_release_interval = ghost_release_interval

        # Maze and ghosts
        self.maze = GoogleMaze(layout=maze_layout)
        self.ghosts = create_google_ghosts(
            self.maze,
            chase_strength=self.chase_strength,
            active_ghost_count=self.active_ghost_count,
            release_schedule=self.ghost_release_schedule,
            release_interval=self.ghost_release_interval,
        )

        self.action_space = spaces.Discrete(4)

        # Observation size calculation:
        # 5 channels * H * W
        # + 3 scalars (power_active, power_timer_norm, pellets_remaining_norm)
        # + 4 ghosts * 5 (4-way dir one-hot + vulnerable) = 20
        # + nearest pellet (4-way dir one-hot + normalized distance) = 5
        # + nearest ghost (4-way escape dir + normalized distance + 4-way ghost action) = 9
        # + previous pacman action (4-way one-hot) = 4
        self._obs_size = (
            5 * self.maze.height * self.maze.width
            + 3
            + (4 * 5)
            + 5
            + 9
            + 4
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_size,),
            dtype=np.float32,
        )

        # Internal state
        self.pacman_pos: Position | None = None
        self.pacman_last_action: Action | None = None
        self.step_count: int = 0
        self.cumulative_score: float = 0.0
        self.episode_count: int = 0
        self.power_mode_active: bool = False
        self.power_mode_steps_remaining: int = 0
        self.recent_pacman_positions: deque[Position] = deque(maxlen=10)
        self.rng: np.random.Generator = np.random.default_rng()

        self._renderer: GoogleRenderer | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.maze.reset()
        self.pacman_pos = self.maze.pacman_spawn()
        self.pacman_last_action = None
        self.step_count = 0
        self.cumulative_score = 0.0
        self.episode_count += 1
        self.power_mode_active = False
        self.power_mode_steps_remaining = 0
        self.recent_pacman_positions.clear()
        self.recent_pacman_positions.append(self.pacman_pos)

        for ghost in self.ghosts:
            ghost.reset()

        obs = self._build_observation()
        info: dict[str, Any] = {
            "pellets_remaining": self.maze.pellets_remaining(),
            "total_pellets": self.maze.total_pellets,
            "step_count": self.step_count,
        }
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step."""
        if self.pacman_pos is None:
            raise RuntimeError("reset() must be called before step().")

        self.step_count += 1
        pacman_action = Action(action)
        previous_pos = self.pacman_pos

        hit_wall = False
        pellet_eaten = False
        power_pellet_eaten = False
        ghosts_eaten = 0
        caught = False
        cleared = False
        pellet_distance_decreased = False
        pellet_distance_increased = False

        # 1. Candidate move & wall check
        delta = ACTION_DELTAS[pacman_action]
        candidate_pos = previous_pos + delta

        # Distance shaping before move
        if (
            self.reward_config.PELLET_DISTANCE_DECREASE != 0.0
            or self.reward_config.PELLET_DISTANCE_INCREASE != 0.0
        ):
            _, dist_before = self.maze.nearest_pellet_path(previous_pos)
            if self.maze.is_walkable(candidate_pos):
                _, dist_after = self.maze.nearest_pellet_path(candidate_pos)
            else:
                dist_after = dist_before

            if dist_before is not None and dist_after is not None:
                pellet_distance_decreased = dist_after < dist_before
                pellet_distance_increased = dist_after > dist_before

        if not self.maze.is_walkable(candidate_pos):
            hit_wall = True
            candidate_pos = previous_pos

        revisited_cell = (
            candidate_pos != previous_pos
            and candidate_pos in self.recent_pacman_positions
        )

        self.pacman_pos = candidate_pos
        self.pacman_last_action = pacman_action
        if candidate_pos != previous_pos:
            self.recent_pacman_positions.append(candidate_pos)

        # 2. Pellet consumption
        consumed = self.maze.consume_pellet(self.pacman_pos)
        if consumed == CellType.PELLET:
            pellet_eaten = True
        elif consumed == CellType.POWER_PELLET:
            power_pellet_eaten = True
            self.power_mode_active = True
            self.power_mode_steps_remaining = self.power_mode_duration
            for ghost in self.ghosts:
                ghost.make_vulnerable()

        # 3. Collision check 1: Pac-Man moved into ghost
        c_caught, c_eaten = self._check_collisions()
        ghosts_eaten += c_eaten
        if c_caught:
            caught = True

        # 4. Power timer update
        if self.power_mode_active and not power_pellet_eaten:
            self.power_mode_steps_remaining -= 1
            if self.power_mode_steps_remaining <= 0:
                self.power_mode_active = False
                for ghost in self.ghosts:
                    ghost.restore_normal()

        # 5. Ghost updates & movement
        if not caught:
            blinky_pos = None
            for g in self.ghosts:
                if g.identity == GhostIdentity.BLINKY:
                    blinky_pos = g.position
                    break

            for ghost in self.ghosts:
                ghost.update_mode(self.step_count, self.power_mode_active)
                ghost.move(
                    maze=self.maze,
                    pacman_pos=self.pacman_pos,
                    pacman_action=self.pacman_last_action,
                    rng=self.rng,
                    blinky_pos=blinky_pos,
                )

            # 6. Collision check 2: Ghost moved into Pac-Man
            c_caught, c_eaten = self._check_collisions()
            ghosts_eaten += c_eaten
            if c_caught:
                caught = True

        # 7. Maze cleared check
        if self.maze.is_cleared():
            cleared = True

        # 8. Termination and truncation
        terminated = bool(caught or cleared)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        # 9. Reward calculation
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
            reward_config=self.reward_config,
        )
        self.cumulative_score += reward

        obs = self._build_observation()
        info = {
            "score": self.cumulative_score,
            "step_count": self.step_count,
            "pellets_remaining": self.maze.pellets_remaining(),
            "total_pellets": self.maze.total_pellets,
            "is_success": cleared,
            "caught": caught,
            "cleared": cleared,
            "ghosts_eaten": ghosts_eaten,
            "pellet_eaten": pellet_eaten,
            "power_pellet_eaten": power_pellet_eaten,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _check_collisions(self) -> tuple[bool, int]:
        """Check for collisions between Pac-Man and ghosts."""
        assert self.pacman_pos is not None
        caught = False
        ghosts_eaten = 0

        for ghost in self.ghosts:
            if not ghost.is_released:
                continue
            if ghost.position == self.pacman_pos:
                if ghost.is_vulnerable:
                    ghost.respawn()
                    ghosts_eaten += 1
                else:
                    caught = True
        return caught, ghosts_eaten

    def _nearest_ghost_features(self) -> np.ndarray:
        """Escape direction, normalized distance, and action of nearest non-vulnerable ghost."""
        assert self.pacman_pos is not None
        features = np.zeros(9, dtype=np.float32)
        pacman_distances = self.maze.distances_from(self.pacman_pos)

        candidates = [
            (pacman_distances[g.position], g)
            for g in self.ghosts
            if g.is_released and not g.is_vulnerable and g.position in pacman_distances
        ]
        if not candidates:
            return features

        min_dist, nearest_ghost = min(candidates, key=lambda x: x[0])
        ghost_distances = self.maze.distances_from(nearest_ghost.position)

        # Escape move that maximizes distance to nearest ghost
        escape_candidates = [
            (ghost_distances.get(self.pacman_pos + ACTION_DELTAS[a], -1), int(a))
            for a in [Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT]
            if self.maze.is_walkable(self.pacman_pos + ACTION_DELTAS[a])
        ]
        if escape_candidates:
            _, escape_action = max(escape_candidates, key=lambda x: x[0])
            features[escape_action] = 1.0

        max_dist = max(1, self.maze.height * self.maze.width - 1)
        features[4] = np.float32(min_dist / max_dist)

        if nearest_ghost.last_action is not None:
            features[5 + int(nearest_ghost.last_action)] = 1.0

        return features

    def _build_observation(self) -> np.ndarray:
        """Construct full float32 observation vector."""
        assert self.pacman_pos is not None
        H, W = self.maze.height, self.maze.width

        # 1. 5 Grid channels
        wall_ch = self.maze.wall_grid().ravel()
        pellet_ch = self.maze.pellet_grid().ravel()
        power_ch = self.maze.power_pellet_grid().ravel()

        pacman_ch = np.zeros((H, W), dtype=np.float32)
        pacman_ch[self.pacman_pos.row, self.pacman_pos.col] = 1.0

        ghost_ch = np.zeros((H, W), dtype=np.float32)
        for ghost in self.ghosts:
            if ghost.is_released:
                ghost_ch[ghost.position.row, ghost.position.col] = 1.0

        # 2. Scalars
        power_active = np.float32(1.0 if self.power_mode_active else 0.0)
        power_timer = np.float32(
            self.power_mode_steps_remaining / self.power_mode_duration
            if self.power_mode_active else 0.0
        )
        total_p = self.maze.total_pellets
        pellets_norm = np.float32(
            self.maze.pellets_remaining() / total_p if total_p > 0 else 0.0
        )
        scalars = np.array([power_active, power_timer, pellets_norm], dtype=np.float32)

        # 3. Ghosts directions and vulnerability (4 ghosts * 5 features = 20)
        ghost_features = np.zeros(20, dtype=np.float32)
        for i, ghost in enumerate(self.ghosts[:4]):
            offset = i * 5
            if ghost.is_released and ghost.last_action is not None:
                ghost_features[offset + int(ghost.last_action)] = 1.0
            if ghost.is_vulnerable:
                ghost_features[offset + 4] = 1.0

        # 4. Nearest pellet features (5)
        nearest_pellet_features = np.zeros(5, dtype=np.float32)
        p_dir, p_dist = self.maze.nearest_pellet_path(self.pacman_pos)
        if p_dir is not None:
            nearest_pellet_features[p_dir] = 1.0
        if p_dist is not None:
            nearest_pellet_features[4] = np.float32(p_dist / max(1, H * W - 1))

        # 5. Nearest ghost features (9)
        nearest_ghost_features = self._nearest_ghost_features()

        # 6. Previous pacman action (4)
        prev_action_features = np.zeros(4, dtype=np.float32)
        if self.pacman_last_action is not None:
            prev_action_features[int(self.pacman_last_action)] = 1.0

        obs = np.concatenate([
            wall_ch,
            pellet_ch,
            power_ch,
            pacman_ch.ravel(),
            ghost_ch.ravel(),
            scalars,
            ghost_features,
            nearest_pellet_features,
            nearest_ghost_features,
            prev_action_features,
        ]).astype(np.float32)

        return obs

    def render(self) -> None:
        """Render frame using GoogleRenderer."""
        if self._renderer is None:
            from v2_render import GoogleRenderer
            self._renderer = GoogleRenderer(
                cell_size=36,
                fps=self.render_fps,
                render_mode="human",
            )
        from v2_render import GoogleRenderState
        state = GoogleRenderState(
            maze=self.maze,
            pacman_position=self.pacman_pos or self.maze.pacman_spawn(),
            pacman_action=self.pacman_last_action,
            ghosts=self.ghosts,
            score=self.cumulative_score,
            step=self.step_count,
            power_mode_active=self.power_mode_active,
            power_mode_steps_remaining=self.power_mode_steps_remaining,
            episode=self.episode_count,
        )
        self._renderer.render(state)

    def close(self) -> None:
        """Close renderer window if opened."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

