"""
Rule-based ghosts for the Google Pac-Man RL environment.

Implements the 4 classic personalities (Blinky, Pinky, Inky, Clyde)
with BFS shortest-path targeting, scatter/chase timing, frightened avoidance,
release scheduling, and curriculum chase strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from config import ACTION_DELTAS, Action
from google_config import (
    OPPOSITE_ACTIONS,
    GHOST_SCATTER_CORNERS,
    GhostIdentity,
    GOOGLE_CHASE_STEPS,
    GOOGLE_CLYDE_CHASE_DISTANCE,
    GOOGLE_GHOST_RELEASE_INTERVAL,
    GOOGLE_INKY_LOOKAHEAD,
    GOOGLE_PINKY_LOOKAHEAD,
    GOOGLE_SCATTER_STEPS,
)
from google_maze import GoogleMaze
from maze import Position


class GhostMode(Enum):
    """Behavioral modes for a Google Pac-Man ghost."""
    IN_HOUSE = auto()
    SCATTER = auto()
    CHASE = auto()
    FRIGHTENED = auto()


@dataclass
class GoogleGhost:
    """A rule-based ghost with unique personality and movement rules.

    Attributes:
        identity:       Enum identifying Blinky, Pinky, Inky, or Clyde.
        spawn:          Initial spawn Position in the ghost house.
        release_step:   Step at which the ghost exits the house.
        scatter_corner: Coordinate targeted during scatter mode.
        chase_strength: Probability of following rule vs random legal move (0.5-1.0).
        scatter_steps:  Number of steps per scatter phase.
        chase_steps:    Number of steps per chase phase.
    """

    identity: GhostIdentity
    spawn: Position
    release_step: int = 0
    scatter_corner: Position = field(default_factory=lambda: Position(1, 1))
    chase_strength: float = 1.0
    scatter_steps: int = GOOGLE_SCATTER_STEPS
    chase_steps: int = GOOGLE_CHASE_STEPS
    phase_offset: int = 0

    # Runtime state
    position: Position = field(init=False)
    last_action: Action | None = field(init=False, default=None)
    is_released: bool = field(init=False, default=False)
    is_vulnerable: bool = field(init=False, default=False)
    mode_just_changed: bool = field(init=False, default=False)
    current_mode: GhostMode = field(init=False, default=GhostMode.IN_HOUSE)

    def __post_init__(self) -> None:
        corner = GHOST_SCATTER_CORNERS.get(self.identity, (1, 1))
        self.scatter_corner = Position(corner[0], corner[1])
        self.reset()

    def reset(self) -> None:
        """Reset ghost to initial episode state."""
        self.position = self.spawn
        self.last_action = None
        self.is_vulnerable = False
        self.mode_just_changed = False
        # Blinky (release_step == 0) starts immediately released
        self.is_released = (self.release_step == 0)
        self.current_mode = GhostMode.SCATTER if self.is_released else GhostMode.IN_HOUSE

    def respawn(self) -> None:
        """Return to ghost house spawn after being eaten by Pac-Man."""
        self.position = self.spawn
        self.is_vulnerable = False
        self.mode_just_changed = True
        self.is_released = True
        self.current_mode = GhostMode.CHASE

    def make_vulnerable(self) -> None:
        """Activate frightened mode."""
        if self.is_released:
            self.is_vulnerable = True
            self.current_mode = GhostMode.FRIGHTENED
            self.mode_just_changed = True

    def restore_normal(self) -> None:
        """Deactivate frightened mode when power timer expires."""
        if self.is_vulnerable:
            self.is_vulnerable = False
            self.mode_just_changed = True

    def update_mode(self, step_count: int, power_mode_active: bool) -> None:
        """Update release status and scatter/chase mode based on step count."""
        # Release check
        if not self.is_released and step_count >= self.release_step:
            self.is_released = True
            self.mode_just_changed = True

        if not self.is_released:
            self.current_mode = GhostMode.IN_HOUSE
            return

        if power_mode_active:
            if not self.is_vulnerable:
                self.make_vulnerable()
            return

        # Power mode not active
        if self.is_vulnerable:
            self.restore_normal()

        # Scatter / Chase cycle
        cycle = self.scatter_steps + self.chase_steps
        phase = (step_count + self.phase_offset) % cycle
        new_mode = GhostMode.SCATTER if phase < self.scatter_steps else GhostMode.CHASE

        if new_mode != self.current_mode and self.current_mode != GhostMode.IN_HOUSE:
            self.mode_just_changed = True
        self.current_mode = new_mode

    def get_target_tile(
        self,
        maze: GoogleMaze,
        pacman_pos: Position,
        pacman_action: Action | None,
        blinky_pos: Position | None = None,
    ) -> Position:
        """Compute the target tile according to ghost identity and current mode."""
        if self.current_mode == GhostMode.SCATTER:
            return self.scatter_corner

        # Default pacman direction delta
        delta = (
            ACTION_DELTAS[pacman_action]
            if pacman_action is not None and pacman_action in ACTION_DELTAS
            else (-1, 0)  # Default facing UP
        )

        if self.identity == GhostIdentity.BLINKY:
            # Red: Directly targets Pac-Man's tile
            return pacman_pos

        elif self.identity == GhostIdentity.PINKY:
            # Pink: 4 tiles ahead in Pac-Man's direction
            return Position(
                pacman_pos.row + GOOGLE_PINKY_LOOKAHEAD * delta[0],
                pacman_pos.col + GOOGLE_PINKY_LOOKAHEAD * delta[1],
            )

        elif self.identity == GhostIdentity.INKY:
            # Cyan: flanker. Far away, cut off the opposite side of the board.
            # Close up, chase directly.
            dist = maze.bfs_shortest_path_distance(self.position, pacman_pos)
            if dist is None:
                dist = abs(self.position.row - pacman_pos.row) + abs(self.position.col - pacman_pos.col)
            if dist > 8:
                return Position(
                    maze.height - 1 - pacman_pos.row,
                    maze.width - 1 - pacman_pos.col,
                )
            return pacman_pos
            

        elif self.identity == GhostIdentity.CLYDE:
            # Orange: Chases if distance > 6 tiles; otherwise retreats to scatter corner
            dist = maze.bfs_shortest_path_distance(self.position, pacman_pos)
            if dist is None:
                dist = abs(self.position.row - pacman_pos.row) + abs(self.position.col - pacman_pos.col)
            if dist > GOOGLE_CLYDE_CHASE_DISTANCE:
                return pacman_pos
            else:
                return self.scatter_corner

        return pacman_pos

    def legal_actions(self, maze: GoogleMaze) -> list[Action]:
        """Return walkable adjacent actions, enforcing no-reverse rule unless forced."""
        # Prefer standard Pac-Man priority: UP, LEFT, DOWN, RIGHT
        priority_actions = [Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT]
        walkable = [
            a for a in priority_actions
            if maze.is_walkable(self.position + ACTION_DELTAS[a])
        ]
        if not walkable:
            return []

        # If reverse not allowed:
        if self.last_action is not None and not self.mode_just_changed:
            opposite = OPPOSITE_ACTIONS[self.last_action]
            non_reverse = [a for a in walkable if a != opposite]
            if non_reverse:
                return non_reverse

        return walkable

    def move(
        self,
        maze: GoogleMaze,
        pacman_pos: Position,
        pacman_action: Action | None,
        rng: np.random.Generator,
        blinky_pos: Position | None = None,
    ) -> Action | None:
        """Choose and execute next move for this ghost."""
        if not self.is_released:
            return None

        actions = self.legal_actions(maze)
        if not actions:
            return None

        # Reset mode_just_changed flag after reading legal actions
        was_mode_changed = self.mode_just_changed
        self.mode_just_changed = False

        # Curriculum chase_strength check: follow rule vs uniform random
        if rng.random() > self.chase_strength:
            chosen = actions[int(rng.integers(len(actions)))]
            self.last_action = chosen
            self.position = self.position + ACTION_DELTAS[chosen]
            return chosen

        # Rule-based decision
        if self.current_mode == GhostMode.FRIGHTENED:
            # Move to maximize BFS distance away from Pac-Man
            best_action = actions[0]
            max_dist = -1
            walkable_pacman = pacman_pos if maze.is_walkable(pacman_pos) else maze.closest_walkable(pacman_pos)
            pacman_distances = maze.distances_from(walkable_pacman)

            for a in actions:
                neighbour = self.position + ACTION_DELTAS[a]
                d = pacman_distances.get(neighbour, -1)
                if d > max_dist:
                    max_dist = d
                    best_action = a
            chosen = best_action
        else:
            # Normal targeting (Chase or Scatter)
            raw_target = self.get_target_tile(maze, pacman_pos, pacman_action, blinky_pos)
            target = maze.closest_walkable(raw_target)
            target_distances = maze.distances_from(target)

            best_actions: list[Action] = []
            min_dist = float("inf")

            for a in actions:
                neighbour = self.position + ACTION_DELTAS[a]
                d = target_distances.get(neighbour, float("inf"))
                if d < min_dist:
                    min_dist = d
                    best_actions = [a]
                elif d == min_dist:
                    best_actions.append(a)
            chosen = best_actions[int(rng.integers(len(best_actions)))]

        self.last_action = chosen
        self.position = self.position + ACTION_DELTAS[chosen]
        return chosen


def create_google_ghosts(
    maze: GoogleMaze,
    chase_strength: float = 1.0,
    active_ghost_count: int = 4,
    release_schedule: Sequence[int] | None = None,
    release_interval: int | None = None,
) -> list[GoogleGhost]:
    """Factory helper to instantiate the 4 Google ghosts with configurable release timing."""
    spawns = maze.ghost_spawns()
    if len(spawns) < 4:
        # Fallback if custom layout has fewer spawns
        spawns = spawns + [spawns[0]] * (4 - len(spawns))

    if release_schedule is not None:
        schedule = list(release_schedule)
        while len(schedule) < 4:
            schedule.append(schedule[-1] + 30)
    elif release_interval is not None:
        schedule = [i * release_interval for i in range(4)]
    else:
        from google_config import GOOGLE_GHOST_RELEASE_SCHEDULE
        schedule = list(GOOGLE_GHOST_RELEASE_SCHEDULE)

    ghost_configs = [
        (GhostIdentity.BLINKY, spawns[0], schedule[0]),
        (GhostIdentity.PINKY, spawns[1], schedule[1]),
        (GhostIdentity.INKY, spawns[2], schedule[2]),
        (GhostIdentity.CLYDE, spawns[3], schedule[3]),
    ]

    personality_strength = {
        GhostIdentity.BLINKY: 1.0,
        GhostIdentity.PINKY: 0.95,
        GhostIdentity.INKY: 0.90,
        GhostIdentity.CLYDE: 0.85,
    }
    ghosts = [
        GoogleGhost(
            identity=ident,
            spawn=sp,
            release_step=rel,
            chase_strength=chase_strength * personality_strength[ident],
            phase_offset=i * 8,
        )
        for i, (ident, sp, rel) in enumerate(ghost_configs[:active_ghost_count])
    ]
    return ghosts
