"""Train and evaluate Pac-Man with ghost-direction observations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import MAX_STEPS, ObservationConfig
from evaluate import EvaluationConfig, EvaluationResult, evaluate
from experiment_utils import summarize_metric
from train import TrainingConfig, train


MODEL_DIR = Path("models/experiments/ghost_direction_ppo_300k")
LOG_DIR = Path("logs/experiments/ghost_direction_ppo_300k")
RESULTS_PATH = Path("results/priority_300k/ghost_direction_comparison.csv")
EPISODES = 30
BASELINE_EVALUATION_SEED = 42_000
TRAINED_EVALUATION_SEED = 42_030


def _evaluation_row(name: str, result: EvaluationResult) -> dict[str, str | int | float]:
    """Summarize one evaluation, including episode-level 95% confidence intervals."""
    metrics = {
        "reward": [episode.total_reward for episode in result.episodes],
        "pellets": [episode.pellets_collected for episode in result.episodes],
        "survival_steps": [episode.steps for episode in result.episodes],
    }
    row: dict[str, str | int | float] = {
        "model": name,
        "episodes": result.n_episodes,
        "success_rate": result.success_rate,
    }
    for metric_name, values in metrics.items():
        summary = summarize_metric(values)
        row[f"mean_{metric_name}"] = summary.mean
        row[f"{metric_name}_ci_low"] = summary.confidence_interval[0]
        row[f"{metric_name}_ci_high"] = summary.confidence_interval[1]
    return row


def run_experiment(
    *,
    total_timesteps: int = 300_000,
    episodes: int = EPISODES,
    seed: int = 42,
) -> list[dict[str, str | int | float]]:
    """Train the observation variant, then compare it with the baseline."""
    if total_timesteps < 1 or episodes < 1:
        raise ValueError("Training steps and evaluation episodes must be positive.")

    model = train(
        TrainingConfig(
            models_dir=MODEL_DIR,
            logs_dir=LOG_DIR,
            tensorboard_dir=LOG_DIR / "tensorboard",
            total_timesteps=total_timesteps,
            seed=seed,
            max_steps=MAX_STEPS,
            observation_config=ObservationConfig(include_ghost_direction=True),
            eval_freq=max(1, total_timesteps // 3),
            checkpoint_freq=total_timesteps,
            n_eval_episodes=5,
            verbose=0,
        )
    )
    baseline_result = evaluate(
        EvaluationConfig(
            model_path=Path("models/best_model.zip"),
            n_episodes=episodes,
            max_steps=MAX_STEPS,
            seed=BASELINE_EVALUATION_SEED,
            verbose=False,
        )
    )
    trained_result = evaluate(
        EvaluationConfig(
            model_path=MODEL_DIR / "final_model.zip",
            n_episodes=episodes,
            max_steps=MAX_STEPS,
            seed=TRAINED_EVALUATION_SEED,
            verbose=False,
        ),
        model=model,
    )
    rows = [
        _evaluation_row("baseline", baseline_result),
        _evaluation_row("ghost_direction_ppo", trained_result),
    ]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ghost-direction model: {MODEL_DIR / 'final_model.zip'}")
    print(f"Evaluation results: {RESULTS_PATH}")
    for row in rows:
        print(
            f"{row['model']}: pellets={row['mean_pellets']:.2f}, "
            f"survival_steps={row['mean_survival_steps']:.1f}, "
            f"success={row['success_rate']:.1%} "
            f"({row['episodes']} episodes)"
        )
    return rows


def main() -> None:
    """Run the 300k-step ghost-direction observation experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(total_timesteps=args.steps, episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
