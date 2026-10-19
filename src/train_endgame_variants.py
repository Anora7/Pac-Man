"""Train and compare opt-in Pac-Man endgame variants."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import replace
from pathlib import Path

from config import MAX_STEPS, REWARDS, ObservationConfig, RewardConfig
from evaluate import EvaluationConfig, EvaluationResult, evaluate
from train import TrainingConfig, train


VARIANTS = (
    "features_only",
    "features_shaping",
    "shaping_only",
    "features_shaping_ghost",
)
EPISODES = 30
TRAINING_STEPS = 300_000
DEFAULT_SEED = 42
BASELINE_MODEL = Path("models/best_model.zip")
MODEL_ROOT = Path("models/experiments/endgame_300k")
LOG_ROOT = Path("logs/experiments/endgame_300k")
RESULTS_ROOT = Path("results/priority_300k")
VARIANT_RESULTS_DIR = RESULTS_ROOT / "endgame_variants"
DISTANCE_DECREASE_BONUS = 0.5
DISTANCE_INCREASE_PENALTY = -0.5
REVISIT_CELL_PENALTY = -0.1
GHOST_DISTANCE_DECREASE_PENALTY = -0.5
LOOP_WINDOW_STEPS = 50
LOOP_MAX_UNIQUE_CELLS = 3

_MODEL_INDEX = {"baseline": 0, "trained": 1}
_POLICY_INDEX = {"deterministic": 0, "stochastic": 1}


def _variant_settings(
    variant: str,
) -> tuple[ObservationConfig, RewardConfig]:
    """Return the observation and reward settings for one experiment arm."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}; choose from {VARIANTS}.")

    use_ghost_features = variant == "features_shaping_ghost"
    use_features = variant in {
        "features_only",
        "features_shaping",
        "features_shaping_ghost",
    }
    use_shaping = variant in {
        "features_shaping",
        "shaping_only",
        "features_shaping_ghost",
    }
    observation_config = ObservationConfig(
        include_nearest_pellet=use_features,
        include_previous_action=use_features,
        include_nearest_ghost=use_ghost_features,
    )
    reward_config = (
        replace(
            REWARDS,
            PELLET_DISTANCE_DECREASE=DISTANCE_DECREASE_BONUS,
            PELLET_DISTANCE_INCREASE=DISTANCE_INCREASE_PENALTY,
            REVISIT_CELL=REVISIT_CELL_PENALTY,
            GHOST_DISTANCE_DECREASE=(
                GHOST_DISTANCE_DECREASE_PENALTY
                if use_ghost_features
                else 0.0
            ),
        )
        if use_shaping
        else REWARDS
    )
    return observation_config, reward_config


def _evaluation_seed(
    seed: int,
    variant: str,
    model_name: str,
    policy_name: str,
) -> int:
    """Create a distinct reproducible seed range for each evaluation group."""
    return (
        seed * 1_000_000
        + VARIANTS.index(variant) * 100_000
        + _POLICY_INDEX[policy_name] * 10_000
        + _MODEL_INDEX[model_name] * 1_000
    )


def _run_output_tag(
    seed: int,
    total_timesteps: int,
    output_tag: str | None,
) -> str:
    """Build a safe output folder name unique to tag, seed, and step budget."""
    if output_tag is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        output_tag,
    ):
        raise ValueError(
            "output_tag must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens."
        )
    suffix = f"seed_{seed}_steps_{total_timesteps}"
    return f"{output_tag}_{suffix}" if output_tag else suffix


def _has_tight_loop(
    positions: tuple[tuple[int, int], ...],
) -> bool:
    """Detect any 50-step window occupying at most three distinct cells."""
    window_size = LOOP_WINDOW_STEPS + 1
    return any(
        len(set(positions[start:start + window_size])) <= LOOP_MAX_UNIQUE_CELLS
        for start in range(len(positions) - window_size + 1)
    )


def _evaluation_row(
    variant: str,
    model_name: str,
    policy_name: str,
    result: EvaluationResult,
) -> dict[str, str | int | float]:
    """Summarize one evaluation protocol and its tight-loop frequency."""
    return {
        "variant": variant,
        "model": model_name,
        "policy": policy_name,
        "episodes": result.n_episodes,
        "mean_pellets": result.mean_pellets_collected,
        "mean_steps": result.mean_steps,
        "clears": sum(episode.cleared for episode in result.episodes),
        "loop_rate": (
            sum(_has_tight_loop(episode.positions) for episode in result.episodes)
            / result.n_episodes
        ),
    }


def _write_results(
    path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    """Write evaluation summaries to a dedicated CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "variant",
        "model",
        "policy",
        "episodes",
        "mean_pellets",
        "mean_steps",
        "clears",
        "loop_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_variant(
    variant: str,
    *,
    total_timesteps: int = TRAINING_STEPS,
    seed: int = DEFAULT_SEED,
    episodes: int = EPISODES,
    output_tag: str | None = None,
) -> list[dict[str, str | int | float]]:
    """Train one variant and evaluate it alongside the unchanged baseline."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if episodes < 1:
        raise ValueError("episodes must be positive.")

    observation_config, reward_config = _variant_settings(variant)
    run_tag = _run_output_tag(seed, total_timesteps, output_tag)
    model_dir = MODEL_ROOT / run_tag / variant
    log_dir = LOG_ROOT / run_tag / variant
    model = train(
        TrainingConfig(
            models_dir=model_dir,
            logs_dir=log_dir,
            tensorboard_dir=log_dir / "tensorboard",
            total_timesteps=total_timesteps,
            seed=seed,
            max_steps=MAX_STEPS,
            reward_config=reward_config,
            observation_config=observation_config,
            eval_freq=max(1, total_timesteps // 3),
            checkpoint_freq=total_timesteps,
            n_eval_episodes=5,
            verbose=0,
        )
    )

    rows: list[dict[str, str | int | float]] = []
    for policy_name, deterministic in (
        ("deterministic", True),
        ("stochastic", False),
    ):
        for model_name in ("baseline", "trained"):
            evaluation_config = EvaluationConfig(
                model_path=(
                    BASELINE_MODEL
                    if model_name == "baseline"
                    else model_dir / "final_model.zip"
                ),
                n_episodes=episodes,
                max_steps=MAX_STEPS,
                deterministic=deterministic,
                seed=_evaluation_seed(seed, variant, model_name, policy_name),
                verbose=False,
                observation_config=(
                    None if model_name == "baseline" else observation_config
                ),
                reward_config=(
                    REWARDS if model_name == "baseline" else reward_config
                ),
                record_trajectory=True,
            )
            result = evaluate(
                evaluation_config,
                model=model if model_name == "trained" else None,
            )
            rows.append(
                _evaluation_row(
                    variant,
                    model_name,
                    policy_name,
                    result,
                )
            )

    output_path = VARIANT_RESULTS_DIR / run_tag / f"{variant}.csv"
    _write_results(output_path, rows)
    print(f"Trained model: {model_dir / 'final_model.zip'}")
    print(f"Variant results: {output_path}")
    return rows


def run_experiment(
    *,
    variant: str | None = None,
    total_timesteps: int = TRAINING_STEPS,
    seed: int = DEFAULT_SEED,
    output_tag: str | None = None,
) -> list[dict[str, str | int | float]]:
    """Run one selected variant or all variants and write their summaries."""
    if variant is not None:
        return run_variant(
            variant,
            total_timesteps=total_timesteps,
            seed=seed,
            output_tag=output_tag,
        )

    run_tag = _run_output_tag(seed, total_timesteps, output_tag)
    rows: list[dict[str, str | int | float]] = []
    for variant_name in VARIANTS:
        rows.extend(
            run_variant(
                variant_name,
                total_timesteps=total_timesteps,
                seed=seed,
                output_tag=output_tag,
            )
        )
    results_path = VARIANT_RESULTS_DIR / run_tag / "endgame_variants.csv"
    _write_results(results_path, rows)
    print(f"Combined endgame results: {results_path}")
    return rows


def main() -> None:
    """Parse CLI settings and start the requested endgame experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Run one experiment arm in an isolated model/log/result folder.",
    )
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-tag",
        help=(
            "Optional safe label prefixed to a folder name that also includes "
            "the seed and step count."
        ),
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    try:
        _run_output_tag(args.seed, args.steps, args.output_tag)
    except ValueError as error:
        parser.error(str(error))
    run_experiment(
        variant=args.variant,
        total_timesteps=args.steps,
        seed=args.seed,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
