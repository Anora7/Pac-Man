"""
Rule-based ghost AI for the Pac-Man RL environment.

This module owns:
  - GhostState: behaviour mode enum.
  - Ghost: single rule-based ghost agent.

This module must not contain RL logic, reward calculations, or rendering.
Ghosts are never trained; they follow deterministic or probabilistic rules
defined at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from config import ACTION_DELTAS, Action, GHOST_CHASE_PROBABILITY, GHOST_CHASE_RADIUS
from ghost_policies import GhostPolicy, RuleBasedGhostPolicy
from maze import Maze, Position


# ---------------------------------------------------------------------------
# Ghost state
# ---------------------------------------------------------------------------

class GhostState(Enum):
    """Behaviour modes for a ghost.

    NORMAL:     Standard movement — probabilistic chase or random walk.
    VULNERABLE: Power mode is active — ghost moves randomly and can be eaten
                by Pac-Man.
    """

    NORMAL = auto()
    VULNERABLE = auto()


# ---------------------------------------------------------------------------
# Ghost
# ---------------------------------------------------------------------------

@dataclass
class Ghost:
    """A single rule-based ghost agent.

    Movement policy per step:

    * ``NORMAL`` state:

      1. Compute Manhattan distance from ghost to Pac-Man.
      2. If ``distance <= chase_radius`` **and** ``rng.random() < chase_probability``:
         move to the walkable neighbour that minimises distance to Pac-Man.
      3. Otherwise: move to a uniformly random walkable neighbour.

    * ``VULNERABLE`` state:

      Always move to a uniformly random walkable neighbour regardless of
      ``chase_probability``.

    The caller (``environment.py``) owns the ``np.random.Generator`` and
    passes it to :meth:`move`. This keeps ``Ghost`` stateless with respect
    to randomness and enables fully reproducible episodes.

    Attributes:
        spawn:             Home cell; ghost returns here on :meth:`respawn`.
        chase_probability: Probability of chasing when within *chase_radius*.
        chase_radius:      Manhattan-distance threshold to consider chasing.
        position:          Current grid position (updated by :meth:`move`).
        state:             Current behaviour mode.

    Example::

        rng = np.random.default_rng(seed=42)
        ghost = Ghost(spawn=Position(5, 5))
        ghost.move(maze, pacman_pos=Position(7, 5), rng=rng)
        print(ghost.position)
    """

    spawn: Position
    chase_probability: float = GHOST_CHASE_PROBABILITY
    chase_radius: int = GHOST_CHASE_RADIUS
    policy: GhostPolicy | None = None
    last_action: int | None = field(init=False, default=None)

    # Mutable runtime state — not part of __init__, set in __post_init__
    position: Position = field(init=False)
    state: GhostState = field(init=False, default=GhostState.NORMAL)

    def __post_init__(self) -> None:
        self.position = self.spawn
        if self.policy is None:
            self.policy = RuleBasedGhostPolicy(
                self.chase_probability,
                self.chase_radius,
            )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return the ghost to its spawn position and restore NORMAL state.

        Call this at the start of every episode.
        """
        self.position = self.spawn
        self.state = GhostState.NORMAL
        self.last_action = None

    def respawn(self) -> None:
        """Teleport the ghost back to its spawn after being eaten.

        The ghost immediately returns to NORMAL state.  Called by
        ``environment.py`` when Pac-Man eats a vulnerable ghost.
        """
        self.position = self.spawn
        self.state = GhostState.NORMAL
        self.last_action = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def make_vulnerable(self) -> None:
        """Switch to VULNERABLE mode.

        Called by the environment when Pac-Man eats a power pellet.
        """
        self.state = GhostState.VULNERABLE

    def restore_normal(self) -> None:
        """Return to NORMAL mode.

        Called by the environment when the power-mode timer expires.
        """
        self.state = GhostState.NORMAL

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(
        self,
        maze: Maze,
        pacman_pos: Position,
        rng: np.random.Generator,
        *,
        observation: np.ndarray | None = None,
        action_override: int | None = None,
    ) -> None:
        """Move the ghost one grid step according to its current policy.

        Args:
            maze:       Used for wall-collision checks.
            pacman_pos: Current position of Pac-Man.
            rng:        NumPy random generator owned by the caller.

        Notes:
            The ghost stays in place when it has no walkable neighbours.
            This is a defensive guard; it should not occur in a well-
            connected maze.
        """
        if action_override is not None:
            action = action_override
        elif self.policy is not None:
            action = self.policy.choose_action(
                maze,
                self.position,
                pacman_pos,
                rng,
                vulnerable=self.is_vulnerable,
                observation=observation,
            )
        else:
            return
        if action is None:
            self.last_action = None
            return
        try:
            direction = Action(action)
        except ValueError:
            self.last_action = None
            return
        candidate = self.position + ACTION_DELTAS[direction]
        if maze.is_walkable(candidate):
            self.position = candidate
            self.last_action = int(direction)
        else:
            self.last_action = None

    def _walkable_neighbours(self, maze: Maze) -> list[Position]:
        """Return all walkable positions adjacent to the current position."""
        return [
            self.position + delta
            for delta in ACTION_DELTAS.values()
            if maze.is_walkable(self.position + delta)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_vulnerable(self) -> bool:
        """True when this ghost can be eaten by Pac-Man."""
        return self.state == GhostState.VULNERABLE

    @property
    def is_normal(self) -> bool:
        """True when this ghost is in standard movement mode."""
        return self.state == GhostState.NORMAL

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Ghost(position={self.position}, state={self.state.name}, "
            f"spawn={self.spawn})"
        )


# ---------------------------------------------------------------------------
# Private movement helpers
# ---------------------------------------------------------------------------

def _random_move(
    neighbours: list[Position],
    rng: np.random.Generator,
) -> Position:
    """Return a uniformly random neighbour from *neighbours*."""
    return neighbours[int(rng.integers(len(neighbours)))]


def _chase_move(
    neighbours: list[Position],
    pacman_pos: Position,
) -> Position:
    """Return the neighbour that minimises Manhattan distance to *pacman_pos*.

    When multiple neighbours share the minimum distance, the first one in
    the iteration order of :data:`config.ACTION_DELTAS` is chosen.  This
    is deterministic and avoids any additional random call.
    """
    return min(neighbours, key=lambda p: _manhattan_distance(p, pacman_pos))


def _manhattan_distance(a: Position, b: Position) -> int:
    """Return the Manhattan distance between two grid positions."""
    return abs(a.row - b.row) + abs(a.col - b.col)
