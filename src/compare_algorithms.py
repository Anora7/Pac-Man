"""Compare PPO and DQN with matching seeds and evaluation episodes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from config import (
    DEFAULT_SEED,
    DQNConfig,
    MAX_STEPS,
    MEDIUM_CPU_STEPS_PER_SECOND,
    MEDIUM_SEEDS,
    MEDIUM_TIMESTEPS,
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
    summarize_metric,
)
from train import TrainingConfig, train
from train_dqn import train_dqn


torch.set_num_threads(1)

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
    "pellets_ci_low",
    "pellets_ci_high",
    "steps_ci_low",
    "steps_ci_high",
)


def run_comparison(
    config: AlgorithmComparisonConfig | None = None,
    *,
    eval_only: bool = False,
) -> list[dict[str, str | int | float]]:
    """Train and evaluate both algorithms, then write tables and plots."""
    cfg = config or AlgorithmComparisonConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_only:
        print(
            f"Re-evaluating saved PPO and DQN models for {len(cfg.seeds)} "
            f"seed(s), {cfg.evaluation_episodes} episodes per model; no training."
        )
    else:
        estimated_hours = (
            2 * cfg.total_timesteps * len(cfg.seeds)
            / MEDIUM_CPU_STEPS_PER_SECOND
            / 3600
        )
        print(
            f"Comparing PPO and DQN: {len(cfg.seeds)} seed(s), "
            f"{cfg.total_timesteps:,} steps per algorithm, CPU only; estimated "
            f"training time about {estimated_hours:.1f} hours."
        )
    seed_records = run_across_seeds(
        cfg.seeds,
        lambda seed: _run_seed(seed, cfg, eval_only=eval_only),
    )
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
    *,
    eval_only: bool = False,
    algorithms: tuple[str, ...] = ("ppo", "dqn"),
) -> list[dict[str, str | int | float]]:
    """Run selected algorithms for one seed with a matched episode seed set."""
    result_rows: list[dict[str, str | int | float]] = []
    if not algorithms or not set(algorithms).issubset({"ppo", "dqn"}):
        raise ValueError("algorithms must select PPO and/or DQN.")
    for algorithm in algorithms:
        model_dir = config.models_dir / algorithm / f"seed_{seed}"
        log_dir = config.logs_dir / algorithm / f"seed_{seed}"
        model_path = model_dir / "final_model.zip"
        if eval_only:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Missing saved {algorithm.upper()} model for seed {seed}: "
                    f"{model_path}"
                )
            algorithm_class = PPO if algorithm == "ppo" else DQN
            model = algorithm_class.load(str(model_path), device="cpu")
        elif algorithm == "ppo":
            ppo_config = TrainingConfig(
                models_dir=model_dir,
                logs_dir=log_dir,
                tensorboard_dir=log_dir / "tensorboard",
                total_timesteps=config.total_timesteps,
                n_steps=max(2, min(2048, config.total_timesteps)),
                batch_size=min(64, max(2, min(2048, config.total_timesteps))),
                n_epochs=10,
                eval_freq=max(1, config.total_timesteps // 2),
                n_eval_episodes=config.evaluation_episodes,
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
                n_eval_episodes=config.evaluation_episodes,
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
                model_path=model_path,
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
                    "pellets_ci_low": "",
                    "pellets_ci_high": "",
                    "steps_ci_low": "",
                    "steps_ci_high": "",
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
                "pellets_ci_low": summary["mean_pellets"].confidence_interval[0],
                "pellets_ci_high": summary["mean_pellets"].confidence_interval[1],
                "steps_ci_low": summary["mean_steps"].confidence_interval[0],
                "steps_ci_high": summary["mean_steps"].confidence_interval[1],
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
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title, label in (
        (axes[0], "mean_reward", "Mean reward", "Reward"),
        (axes[1], "mean_pellets", "Mean pellets", "Pellets"),
        (axes[2], "mean_steps", "Survival steps", "Steps"),
    ):
        values = [
            [float(row[metric]) for row in records if row["algorithm"] == algorithm]
            for algorithm in ("PPO", "DQN")
        ]
        summaries = [summarize_metric(group) for group in values]
        means = [summary.mean for summary in summaries]
        errors = np.asarray([
            [summary.mean - summary.confidence_interval[0] for summary in summaries],
            [summary.confidence_interval[1] - summary.mean for summary in summaries],
        ])
        axis.bar(("PPO", "DQN"), means, yerr=errors, capsize=5)
        axis.set_title(title)
        axis.set_ylabel(label)
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
        lines.append(f"| {algorithm} | {mean(float(r['mean_reward']) for r in selected):.2f} "
                     f"| {mean(float(r['success_rate']) for r in selected):.1%} "
                     f"| {mean(float(r['mean_pellets']) for r in selected):.2f} "
                     f"| {mean(float(r['mean_steps']) for r in selected):.1f} |")
    lines.extend(["", "95% confidence intervals across seed-level means:", ""])
    lines.append("| Algorithm | Reward | Pellets | Survival steps |")
    lines.append("|---|---:|---:|---:|")
    for algorithm in ("PPO", "DQN"):
        selected = [row for row in records if row["algorithm"] == algorithm]
        summaries = {
            metric: summarize_metric([float(row[metric]) for row in selected])
            for metric in ("mean_reward", "mean_pellets", "mean_steps")
        }
        format_ci = lambda summary: (
            f"{summary.mean:.2f} [{summary.confidence_interval[0]:.2f}, "
            f"{summary.confidence_interval[1]:.2f}]"
        )
        lines.append(
            f"| {algorithm} | {format_ci(summaries['mean_reward'])} "
            f"| {format_ci(summaries['mean_pellets'])} "
            f"| {format_ci(summaries['mean_steps'])} |"
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


def _medium_config() -> AlgorithmComparisonConfig:
    """Return a two-seed, 100k-step matched PPO/DQN comparison."""
    medium_dqn = replace(
        DQNConfig(),
        total_timesteps=MEDIUM_TIMESTEPS,
        eval_freq=MEDIUM_TIMESTEPS // 2,
        checkpoint_freq=MEDIUM_TIMESTEPS,
    )
    return AlgorithmComparisonConfig(
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + MEDIUM_SEEDS)),
        total_timesteps=MEDIUM_TIMESTEPS,
        evaluation_episodes=30,
        max_steps=MAX_STEPS,
        models_dir=Path("models/experiments/medium/algorithm_comparison"),
        logs_dir=Path("logs/experiments/medium/algorithm_comparison"),
        results_path=Path("results/medium/algorithm_comparison.csv"),
        report_path=Path("results/medium/algorithm_comparison_report.md"),
        plots_dir=Path("plots/experiments/medium/algorithm_comparison"),
        dqn_config=medium_dqn,
    )


def priority_config() -> AlgorithmComparisonConfig:
    """Return the three-seed, 300k-step experiment configuration."""
    steps = 300_000
    dqn = replace(
        DQNConfig(),
        total_timesteps=steps,
        eval_freq=steps // 2,
        checkpoint_freq=steps,
    )
    return AlgorithmComparisonConfig(
        seeds=(42, 43, 44),
        total_timesteps=steps,
        evaluation_episodes=30,
        max_steps=MAX_STEPS,
        models_dir=Path("models/experiments/priority_300k/algorithm_comparison"),
        logs_dir=Path("logs/experiments/priority_300k/algorithm_comparison"),
        results_path=Path("results/priority_300k/algorithm_comparison.csv"),
        report_path=Path("results/priority_300k/algorithm_comparison_report.md"),
        plots_dir=Path("plots/experiments/priority_300k/algorithm_comparison"),
        dqn_config=dqn,
    )


def main() -> None:
    """Run the algorithm comparison in quick or full mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--medium", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-evaluate saved medium models without training.",
    )
    args = parser.parse_args()
    if args.eval_only and (args.quick or args.full):
        parser.error("--eval-only is only supported for the medium preset.")
    if args.eval_only:
        config = _medium_config()
    elif args.quick:
        config = _quick_config()
    elif args.medium:
        config = _medium_config()
    else:
        config = AlgorithmComparisonConfig()
    run_comparison(config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
