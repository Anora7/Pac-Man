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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config import MAX_STEPS
from environment import PacmanEnvironment


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
    """

    model_path: Path = field(default_factory=lambda: Path("models/best_model.zip"))
    n_episodes: int = 10
    render: bool = False
    deterministic: bool = True
    seed: int | None = None
    max_steps: int = MAX_STEPS
    verbose: bool = True


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


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    config: EvaluationConfig | None = None,
    model: PPO | None = None,
) -> EvaluationResult:
    """Evaluate a trained PPO model over multiple episodes.

    Args:
        config: :class:`EvaluationConfig` instance.  Uses defaults if
                ``None``.
        model:  Pre-loaded :class:`~stable_baselines3.PPO` model.  When
                provided, ``config.model_path`` is ignored.  Useful for
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
        model = PPO.load(str(model_zip))

    render_mode = "human" if config.render else None
    env = PacmanEnvironment(render_mode=render_mode, max_steps=config.max_steps)

    if config.verbose:
        print("\n=== Pac-Man RL — Evaluation ===")
        print(f"Episodes:      {config.n_episodes}")
        print(f"Deterministic: {config.deterministic}")
        print(f"Max steps/ep:  {config.max_steps}")
        print()

    episodes: list[EpisodeResult] = []

    try:
        for ep in range(1, config.n_episodes + 1):
            obs, info = env.reset(seed=config.seed)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=config.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                if config.render:
                    env.render()

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

    finally:
        env.close()

    evaluation = _aggregate(episodes, config.n_episodes)

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


def _print_summary(result: EvaluationResult) -> None:
    """Print a formatted summary of evaluation results to stdout."""
    cleared = sum(ep.cleared for ep in result.episodes)
    caught = sum(ep.caught for ep in result.episodes)
    timeouts = sum(ep.truncated for ep in result.episodes)

    print()
    print("─" * 50)
    print("  EVALUATION SUMMARY")
    print("─" * 50)
    print(f"  Episodes evaluated : {result.n_episodes}")
    print(f"  Mean reward        : {result.mean_reward:+.2f} ± {result.std_reward:.2f}")
    print(f"  Mean episode steps : {result.mean_steps:.1f}")
    print(f"  Mean pellets/ep    : {result.mean_pellets_collected:.1f}")
    print(f"  Avg completion     : {result.avg_completion_fraction:.1%}")
    print(f"  Success rate       : {result.success_rate:.1%}  ({cleared}/{result.n_episodes} cleared)")
    print(f"  Caught             : {caught}/{result.n_episodes}")
    print(f"  Timeouts           : {timeouts}/{result.n_episodes}")
    print("─" * 50)


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
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    _config = EvaluationConfig(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    evaluate(_config)
