"""Compare PPO and DQN with matching seeds and evaluation episodes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from config import (
    DEFAULT_SEED,
    DQNConfig,
    MAX_STEPS,
    MultiSeedConfig,
    QUICK_EPISODES,
    QUICK_SEEDS,
    QUICK_TIMESTEPS,
    TOTAL_TIMESTEPS,
)
from evaluate import EvaluationConfig, evaluate
from experiment_utils import (
    aggregate_metrics,
    plot_monitor_curves,
    run_across_seeds,
)
from train import TrainingConfig, train
from train_dqn import train_dqn


@dataclass(frozen=True)
class AlgorithmComparisonConfig:
    """Settings shared by matched PPO and DQN experiments."""

    seeds: tuple[int, ...] = (42, 43, 44)
    total_timesteps: int = TOTAL_TIMESTEPS
    evaluation_episodes: int = 10
    max_steps: int = MAX_STEPS
    models_dir: Path = Path("models/experiments/algorithm_comparison")
    logs_dir: Path = Path("logs/experiments/algorithm_comparison")
    results_path: Path = Path("results/algorithm_comparison.csv")
    report_path: Path = Path("results/algorithm_comparison_report.md")
    plots_dir: Path = Path("plots/experiments/algorithm_comparison")
    dqn_config: DQNConfig = DQNConfig()


CSV_COLUMNS = (
    "algorithm",
    "seed",
    "mean_reward",
    "std_reward",
    "success_rate",
    "mean_pellets",
    "mean_steps",
    "reward_ci_low",
    "reward_ci_high",
    "success_ci_low",
    "success_ci_high",
)


def run_comparison(
    config: AlgorithmComparisonConfig | None = None,
) -> list[dict[str, str | int | float]]:
    """Train and evaluate both algorithms, then write tables and plots."""
    cfg = config or AlgorithmComparisonConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    seed_records = run_across_seeds(cfg.seeds, lambda seed: _run_seed(seed, cfg))
    records = [record for seed_result in seed_records for record in seed_result]
    _write_comparison_csv(cfg, records)
    _write_comparison_plots(cfg, records)
    _write_report(cfg, records)
    print(f"Algorithm comparison CSV: {cfg.results_path}")
    print(f"Algorithm comparison report: {cfg.report_path}")
    print(f"Algorithm comparison plots: {cfg.plots_dir}")
    return records


def _run_seed(
    seed: int,
    config: AlgorithmComparisonConfig,
) -> list[dict[str, str | int | float]]:
    """Run the two algorithms for one seed and match their test episodes."""
    result_rows: list[dict[str, str | int | float]] = []
    for algorithm in ("ppo", "dqn"):
        model_dir = config.models_dir / algorithm / f"seed_{seed}"
        log_dir = config.logs_dir / algorithm / f"seed_{seed}"
        if algorithm == "ppo":
            ppo_config = TrainingConfig(
                models_dir=model_dir,
                logs_dir=log_dir,
                tensorboard_dir=log_dir / "tensorboard",
                total_timesteps=config.total_timesteps,
                n_steps=max(2, min(2048, config.total_timesteps)),
                batch_size=min(64, max(2, min(2048, config.total_timesteps))),
                n_epochs=10,
                eval_freq=max(1, config.total_timesteps // 2),
                n_eval_episodes=1,
                checkpoint_freq=max(1, config.total_timesteps),
                max_steps=config.max_steps,
                seed=seed,
                verbose=0,
            )
            model = train(ppo_config)
        else:
            dqn_config = replace(
                config.dqn_config,
                total_timesteps=config.total_timesteps,
                models_dir=model_dir,
                logs_dir=log_dir,
                tensorboard_dir=log_dir / "tensorboard",
                eval_freq=max(1, config.total_timesteps // 2),
                n_eval_episodes=1,
                checkpoint_freq=max(1, config.total_timesteps),
                max_steps=config.max_steps,
                seed=seed,
            )
            if config.total_timesteps <= 128:
                dqn_config = replace(
                    dqn_config,
                    learning_starts=min(8, config.total_timesteps // 2),
                    buffer_size=max(32, config.total_timesteps * 4),
                    batch_size=min(8, max(2, config.total_timesteps // 4)),
                    target_update_interval=max(1, config.total_timesteps // 2),
                )
            model = train_dqn(dqn_config)
        evaluation = evaluate(
            EvaluationConfig(
                model_path=model_dir / "final_model.zip",
                n_episodes=config.evaluation_episodes,
                max_steps=config.max_steps,
                seed=seed + 10_000,
                verbose=False,
            ),
            model=model,
        )
        result_rows.append({
            "algorithm": algorithm.upper(),
            "seed": seed,
            "mean_reward": evaluation.mean_reward,
            "std_reward": evaluation.std_reward,
            "success_rate": evaluation.success_rate,
            "mean_pellets": evaluation.mean_pellets_collected,
            "mean_steps": evaluation.mean_steps,
        })
    return result_rows


def _write_comparison_csv(
    config: AlgorithmComparisonConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Write per-seed evaluations followed by aggregate confidence intervals."""
    confidence_config = MultiSeedConfig(seeds=config.seeds)
    with config.results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for algorithm in ("PPO", "DQN"):
            selected = [row for row in records if row["algorithm"] == algorithm]
            writer.writerows(
                {
                    **row,
                    "reward_ci_low": "",
                    "reward_ci_high": "",
                    "success_ci_low": "",
                    "success_ci_high": "",
                }
                for row in selected
            )
            summary = aggregate_metrics(
                selected,
                ("mean_reward", "success_rate", "mean_pellets", "mean_steps"),
                confidence_config,
            )
            writer.writerow({
                "algorithm": algorithm,
                "seed": "aggregate",
                "mean_reward": summary["mean_reward"].mean,
                "std_reward": summary["mean_reward"].standard_deviation,
                "success_rate": summary["success_rate"].mean,
                "mean_pellets": summary["mean_pellets"].mean,
                "mean_steps": summary["mean_steps"].mean,
                "reward_ci_low": summary["mean_reward"].confidence_interval[0],
                "reward_ci_high": summary["mean_reward"].confidence_interval[1],
                "success_ci_low": summary["success_rate"].confidence_interval[0],
                "success_ci_high": summary["success_rate"].confidence_interval[1],
            })


def _write_comparison_plots(
    config: AlgorithmComparisonConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Create shaded per-episode curves and seed-level outcome bars."""
    for algorithm in ("ppo", "dqn"):
        monitor_paths = [
            config.logs_dir / algorithm / f"seed_{seed}" / "0.monitor.csv"
            for seed in config.seeds
        ]
        plot_monitor_curves(
            monitor_paths,
            "r",
            config.plots_dir / f"{algorithm}_reward_curve.png",
            title=f"{algorithm.upper()} reward across seeds",
            y_label="Episode reward",
        )
        plot_monitor_curves(
            monitor_paths,
            "is_success",
            config.plots_dir / f"{algorithm}_success_curve.png",
            title=f"{algorithm.upper()} success rate across seeds",
            y_label="Success (0/1)",
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, title, label in (
        (axes[0], "mean_reward", "Final mean reward", "Reward"),
        (axes[1], "success_rate", "Final success rate", "Success rate"),
    ):
        values = [
            [float(row[metric]) for row in records if row["algorithm"] == algorithm]
            for algorithm in ("PPO", "DQN")
        ]
        means = [mean(group) for group in values]
        deviations = [
            stdev(group) if len(group) > 1 else 0.0
            for group in values
        ]
        axis.bar(("PPO", "DQN"), means, yerr=deviations, capsize=5)
        axis.set_title(title)
        axis.set_ylabel(label)
        if metric == "success_rate":
            axis.set_ylim(0.0, 1.0)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        config.plots_dir / "algorithm_comparison_bars.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_report(
    config: AlgorithmComparisonConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Write a short result and design-decision note beside the CSV."""
    lines = [
        "# PPO vs DQN comparison",
        "",
        f"Matched seeds: {', '.join(map(str, config.seeds))}.",
        f"Training steps per seed and algorithm: {config.total_timesteps:,}.",
        "",
        "## Results",
        "",
        "| Algorithm | Mean reward | Success rate | Mean pellets | Mean steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in ("PPO", "DQN"):
        selected = [row for row in records if row["algorithm"] == algorithm]
        lines.append(
            f"| {algorithm} | {mean(float(r['mean_reward']) for r in selected):.2f} "
            f"| {mean(float(r['success_rate']) for r in selected):.1%} "
            f"| {mean(float(r['mean_pellets']) for r in selected):.2f} "
            f"| {mean(float(r['mean_steps']) for r in selected):.1f} |"
        )
    lines.extend([
        "",
        "## Relation to Decision 001",
        "",
        "Decision 001 selects PPO as the primary algorithm because its clipped "
        "policy updates are expected to be stable and practical for this small "
        "discrete navigation task. This comparison tests that rationale rather "
        "than assuming PPO wins: interpret the matched-seed reward and "
        "scale-independent success, pellet, and survival metrics together.",
        "",
    ])
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text("\n".join(lines), encoding="utf-8")


def _quick_config() -> AlgorithmComparisonConfig:
    """Return a CPU-bounded configuration for end-to-end smoke testing."""
    quick_dqn = replace(
        DQNConfig(),
        total_timesteps=QUICK_TIMESTEPS,
        learning_starts=8,
        buffer_size=128,
        batch_size=8,
        target_update_interval=16,
    )
    return AlgorithmComparisonConfig(
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + QUICK_SEEDS)),
        total_timesteps=QUICK_TIMESTEPS,
        evaluation_episodes=QUICK_EPISODES,
        max_steps=32,
        models_dir=Path("models/experiments/algorithm_comparison"),
        logs_dir=Path("logs/experiments/algorithm_comparison"),
        results_path=Path("results/algorithm_comparison.csv"),
        report_path=Path("results/algorithm_comparison_report.md"),
        plots_dir=Path("plots/experiments/algorithm_comparison"),
        dqn_config=quick_dqn,
    )


def main() -> None:
    """Run the algorithm comparison in quick or full mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = _quick_config() if args.quick else AlgorithmComparisonConfig()
    estimated_minutes = (
        2 * config.total_timesteps * len(config.seeds) / 90 / 60
    )
    print(
        f"Comparing PPO and DQN: {len(config.seeds)} seed(s), "
        f"{config.total_timesteps:,} steps each, CPU only. "
        f"Estimated training time: at least {estimated_minutes:.1f} min."
    )
    run_comparison(config)


if __name__ == "__main__":
    main()
