"""
Evaluation pipeline for trained Pac-Man RL models.

This module owns:
  - EvaluationConfig: evaluation settings and model path.
  - EpisodeResult:    per-episode metrics (frozen dataclass).
  - EvaluationResult: aggregated metrics across all episodes.
  - evaluate():       main evaluation entry point.

This module must not train models, modify environment logic, or render
graphics directly — rendering is delegated to the environment's render()
method when render_mode="human".

Usage::

    from evaluate import EvaluationConfig, evaluate

    config = EvaluationConfig(
        model_path=Path("models/best_model.zip"),
        n_episodes=20,
        render=False,
    )
    result = evaluate(config)
    print(f"Success rate: {result.success_rate:.1%}")

Or pass a pre-loaded model to skip disk I/O::

    model = PPO.load("models/best_model")
    result = evaluate(config, model=model)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from config import (
    FPS,
    GHOST_PERSONALITIES,
    GHOST_COUNT,
    MAX_STEPS,
    ObservationConfig,
    REWARDS,
    RewardConfig,
)
from environment import PacmanEnvironment
from ghost_policies import (
    ChaseGhostPolicy,
    RandomGhostPolicy,
    RuleBasedGhostPolicy,
    SharedLearnedGhostController,
)
from observation_metadata import load_observation_config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Settings for a single evaluation run.

    Attributes:
        model_path:    Path to the ``.zip`` model file produced by training.
                       Ignored when a pre-loaded model is passed to
                       :func:`evaluate` directly.
        n_episodes:    Number of episodes to evaluate.
        render:        When ``True``, open a pygame window and render each
                       step.  Requires a display.
        deterministic: Use the deterministic policy (argmax over action
                       probabilities).  Recommended for evaluation.
        seed:          Seed passed to ``env.reset()``.  ``None`` means the
                       environment chooses a random seed each episode.
        max_steps:     Maximum steps per episode (forwarded to the env).
        verbose:       Print per-episode and summary results to stdout.
        ghost_type:    Ghost movement policy for evaluation.
        ghost_model_path: Optional learned-ghost PPO archive.
        observation_config: Observation settings; absent settings load the
                            model sidecar or use baseline settings.
        render_fps:    Target Pygame playback rate.
        save_csv:      Optional per-episode metrics export path.
        record_trajectory: Include each episode's Pac-Man positions in results.
    """

    model_path: Path = field(default_factory=lambda: Path("models/best_model.zip"))
    n_episodes: int = 10
    render: bool = False
    deterministic: bool = True
    seed: int | None = None
    max_steps: int = MAX_STEPS
    verbose: bool = True
    ghost_type: str = "rule"
    ghost_model_path: Path | None = None
    observation_config: ObservationConfig | None = None
    render_fps: int = FPS
    save_csv: Path | None = None
    reward_config: RewardConfig = REWARDS
    ghost_count: int = GHOST_COUNT
    record_trajectory: bool = False

    def __post_init__(self) -> None:
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be at least 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeResult:
    """Metrics captured from a single evaluation episode.

    Attributes:
        episode:             1-indexed episode number.
        total_reward:        Sum of all step rewards in the episode.
        steps:               Number of steps taken.
        pellets_collected:   Regular + power pellets eaten.
        total_pellets:       Total pellets at episode start.
        cleared:             True if the maze was fully cleared.
        caught:              True if Pac-Man was caught by a ghost.
        truncated:           True if the episode hit ``max_steps``.
        completion_fraction: Fraction of pellets collected (0.0–1.0).
    """

    episode: int
    total_reward: float
    steps: int
    pellets_collected: int
    total_pellets: int
    cleared: bool
    caught: bool
    truncated: bool
    completion_fraction: float
    positions: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated metrics across all evaluation episodes.

    Attributes:
        n_episodes:               Total episodes evaluated.
        episodes:                 Individual :class:`EpisodeResult` objects.
        mean_reward:              Mean total reward per episode.
        std_reward:               Standard deviation of total reward.
        mean_steps:               Mean episode length in steps.
        mean_pellets_collected:   Mean pellets collected per episode.
        success_rate:             Fraction of episodes where maze was cleared.
        avg_completion_fraction:  Mean completion fraction across episodes.
    """

    n_episodes: int
    episodes: list[EpisodeResult]
    mean_reward: float
    std_reward: float
    mean_steps: float
    mean_pellets_collected: float
    success_rate: float
    avg_completion_fraction: float

    @property
    def mean_survival_steps(self) -> float:
        """Alias for mean episode steps, including caught and cleared runs."""
        return self.mean_steps

    @property
    def caught_rate(self) -> float:
        """Fraction of evaluated episodes ending with Pac-Man caught."""
        return sum(episode.caught for episode in self.episodes) / self.n_episodes

    @property
    def timeout_rate(self) -> float:
        """Fraction of evaluated episodes reaching the step limit."""
        return sum(episode.truncated for episode in self.episodes) / self.n_episodes


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    config: EvaluationConfig | None = None,
    model: BaseAlgorithm | None = None,
) -> EvaluationResult:
    """Evaluate a trained PPO model over multiple episodes.

    Args:
        config: :class:`EvaluationConfig` instance.  Uses defaults if
                ``None``.
        model:  Pre-loaded Stable-Baselines3 model. When provided,
                ``config.model_path`` is ignored. Useful for
                testing and for chaining directly with :func:`train.train`.

    Returns:
        An :class:`EvaluationResult` containing per-episode and aggregate
        metrics.

    Raises:
        FileNotFoundError: If ``config.model_path`` does not exist and no
                           pre-loaded model was supplied.

    Example::

        result = evaluate(EvaluationConfig(n_episodes=5, verbose=False))
        assert 0.0 <= result.success_rate <= 1.0
    """
    if config is None:
        config = EvaluationConfig()

    if model is None:
        model_zip = config.model_path
        if not model_zip.exists():
            # SB3 may append .zip automatically; try without extension too
            model_zip_no_ext = config.model_path.with_suffix("")
            if Path(str(model_zip_no_ext) + ".zip").exists():
                model_zip = Path(str(model_zip_no_ext) + ".zip")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {config.model_path}\n"
                    "Train a model first with src/train.py."
                )
        model = PPO.load(str(model_zip), device="cpu")

    if config.ghost_type not in {"rule", "random", "chase", "learned", "personality"}:
        raise ValueError(f"Unsupported ghost_type: {config.ghost_type}")
    if config.render_fps < 1:
        raise ValueError("render_fps must be a positive integer.")
    if config.ghost_type == "learned" and config.ghost_model_path is None:
        raise ValueError("ghost_model_path is required for ghost_type='learned'.")

    observation_config = config.observation_config
    if observation_config is None:
        observation_config = load_observation_config(config.model_path)

    render_mode = "human" if config.render else None
    env = PacmanEnvironment(
        render_mode=render_mode,
        max_steps=config.max_steps,
        observation_config=observation_config,
        render_fps=config.render_fps,
        reward_config=config.reward_config,
        ghost_count=config.ghost_count,
    )
    learned_ghost_controller = _configure_ghosts(
        env,
        config.ghost_type,
        config.ghost_model_path,
    )
    if model.observation_space.shape != env.observation_space.shape:
        env.close()
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not "
            f"match requested environment shape {env.observation_space.shape}. "
            "Use the matching --obs-config/model metadata."
        )

    if config.verbose:
        print("\n=== Pac-Man RL - Evaluation ===")
        print(f"Episodes:      {config.n_episodes}")
        print(f"Deterministic: {config.deterministic}")
        print(f"Max steps/ep:  {config.max_steps}")
        print()

    episodes: list[EpisodeResult] = []

    try:
        for ep in range(1, config.n_episodes + 1):
            episode_seed = (
                None if config.seed is None else config.seed + ep - 1
            )
            obs, info = env.reset(seed=episode_seed)
            if learned_ghost_controller is not None:
                learned_ghost_controller.reset()
            positions: list[tuple[int, int]] = []
            if config.record_trajectory:
                positions.append(tuple(info["pacman_position"]))
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated or env.window_closed):
                action, _ = model.predict(obs, deterministic=config.deterministic)
                ghost_actions = None
                if learned_ghost_controller is not None:
                    active_index = learned_ghost_controller.active_ghost_index
                    ghost_actions = [None] * len(env.ghosts)
                    ghost_actions[active_index] = (
                        learned_ghost_controller.choose_action(
                            env.maze,
                            [ghost.position for ghost in env.ghosts],
                            env.pacman_position,
                            env._power_mode_active,
                            env.ghosts[active_index].is_vulnerable,
                            env.np_random,
                        )
                    )
                obs, reward, terminated, truncated, info = env.step(
                    int(action),
                    ghost_actions=ghost_actions,
                )
                total_reward += reward
                if learned_ghost_controller is not None:
                    learned_ghost_controller.advance()
                if config.record_trajectory:
                    positions.append(tuple(info["pacman_position"]))
                if config.render:
                    env.render()
            if env.window_closed and not terminated:
                truncated = True

            steps = info["step"]
            total_pellets = info["total_pellets"]
            pellets_remaining = info["pellets_remaining"]
            pellets_collected = total_pellets - pellets_remaining
            completion = info["completion_fraction"]
            cleared = terminated and completion >= 1.0
            caught = terminated and not cleared

            result = EpisodeResult(
                episode=ep,
                total_reward=total_reward,
                steps=steps,
                pellets_collected=pellets_collected,
                total_pellets=total_pellets,
                cleared=cleared,
                caught=caught,
                truncated=truncated,
                completion_fraction=completion,
                positions=tuple(positions),
            )
            episodes.append(result)

            if config.verbose:
                status = "CLEARED" if cleared else ("CAUGHT" if caught else "TIMEOUT")
                print(
                    f"  Ep {ep:3d}:  "
                    f"reward={total_reward:9.1f}  "
                    f"steps={steps:4d}  "
                    f"pellets={pellets_collected:2d}/{total_pellets}  "
                    f"[{status}]"
                )
            if env.window_closed:
                break

    finally:
        env.close()

    evaluation = _aggregate(episodes, len(episodes))
    if config.save_csv is not None:
        _save_episode_csv(config.save_csv, evaluation.episodes)

    if config.verbose:
        _print_summary(evaluation)

    return evaluation


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate(episodes: list[EpisodeResult], n_episodes: int) -> EvaluationResult:
    """Compute aggregate statistics from per-episode results."""
    rewards = [ep.total_reward for ep in episodes]
    return EvaluationResult(
        n_episodes=n_episodes,
        episodes=episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=float(np.mean([ep.steps for ep in episodes])),
        mean_pellets_collected=float(np.mean([ep.pellets_collected for ep in episodes])),
        success_rate=float(sum(ep.cleared for ep in episodes) / n_episodes),
        avg_completion_fraction=float(
            np.mean([ep.completion_fraction for ep in episodes])
        ),
    )


def _configure_ghosts(
    env: PacmanEnvironment,
    ghost_type: str,
    ghost_model_path: Path | None,
) -> SharedLearnedGhostController | None:
    """Install the requested ghost policy without changing default rules."""
    if ghost_type == "rule":
        return None
    if ghost_type == "random":
        policies = [RandomGhostPolicy() for _ in env.ghosts]
    elif ghost_type == "chase":
        policies = [ChaseGhostPolicy() for _ in env.ghosts]
    elif ghost_type == "learned":
        assert ghost_model_path is not None
        from stable_baselines3 import PPO

        ghost_model = PPO.load(str(ghost_model_path), device="cpu")
        controller = SharedLearnedGhostController(
            ghost_model,
            len(env.ghosts),
        )
        for index, ghost in enumerate(env.ghosts):
            ghost.policy = controller.policy_for(index)
        return controller
    elif ghost_type == "personality":
        personalities = GHOST_PERSONALITIES
        policies = [
            RuleBasedGhostPolicy(
                personality.chase_probability,
                personality.chase_radius,
            )
            for personality in (
                personalities[index % len(personalities)]
                for index in range(len(env.ghosts))
            )
        ]
    else:
        raise ValueError(f"Unsupported ghost_type: {ghost_type}")
    for ghost, policy in zip(env.ghosts, policies):
        ghost.policy = policy
    return None


def _save_episode_csv(path: Path, episodes: list[EpisodeResult]) -> None:
    """Write detailed evaluation metrics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "episode",
        "total_reward",
        "steps",
        "pellets_collected",
        "total_pellets",
        "cleared",
        "caught",
        "truncated",
        "completion_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for result in episodes:
            writer.writerow({
                field_name: getattr(result, field_name)
                for field_name in columns
            })


def _print_summary(result: EvaluationResult) -> None:
    """Print a formatted summary of evaluation results to stdout."""
    cleared = sum(ep.cleared for ep in result.episodes)
    caught = sum(ep.caught for ep in result.episodes)
    timeouts = sum(ep.truncated for ep in result.episodes)

    print()
    print("-" * 50)
    print("  EVALUATION SUMMARY")
    print("-" * 50)
    print(f"  Episodes evaluated : {result.n_episodes}")
    print(f"  Mean reward        : {result.mean_reward:+.2f} +/- {result.std_reward:.2f}")
    print(f"  Mean episode steps : {result.mean_steps:.1f}")
    print(f"  Mean survival     : {result.mean_survival_steps:.1f}")
    print(f"  Mean pellets/ep    : {result.mean_pellets_collected:.1f}")
    print(f"  Avg completion     : {result.avg_completion_fraction:.1%}")
    print(f"  Success rate       : {result.success_rate:.1%}  ({cleared}/{result.n_episodes} cleared)")
    print(f"  Caught             : {caught}/{result.n_episodes}")
    print(f"  Timeouts           : {timeouts}/{result.n_episodes}")
    print(f"  Caught rate        : {result.caught_rate:.1%}")
    print(f"  Timeout rate       : {result.timeout_rate:.1%}")
    print("-" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained Pac-Man RL model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best_model.zip"),
        help="Path to the .zip model file.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ghost-type",
        choices=("rule", "random", "chase", "learned"),
        default="rule",
    )
    parser.add_argument("--ghost-model", type=Path)
    parser.add_argument(
        "--obs-config",
        choices=(
            "baseline",
            "ghost-direction",
            "previous-action",
            "enhanced",
            "metadata",
        ),
        default="baseline",
    )
    parser.add_argument("--render-fps", type=int, default=FPS)
    parser.add_argument("--save-csv", type=Path)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    observation_config = None
    if args.obs_config != "metadata":
        observation_config = {
            "baseline": ObservationConfig(),
            "ghost-direction": ObservationConfig(include_ghost_direction=True),
            "previous-action": ObservationConfig(
                include_previous_pacman_action=True
            ),
            "enhanced": ObservationConfig(
                include_ghost_direction=True,
                include_previous_pacman_action=True,
            ),
        }[args.obs_config]

    _config = EvaluationConfig(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=args.deterministic,
        seed=args.seed,
        ghost_type=args.ghost_type,
        ghost_model_path=args.ghost_model,
        observation_config=observation_config,
        render_fps=args.render_fps,
        save_csv=args.save_csv,
    )
    evaluate(_config)
