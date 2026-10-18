"""Run controlled PPO reward ablations with matched seeds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

from config import (
    AblationConfig,
    DEFAULT_SEED,
    MEDIUM_CPU_STEPS_PER_SECOND,
    MEDIUM_SEEDS,
    MEDIUM_TIMESTEPS,
    MultiSeedConfig,
    QUICK_EPISODES,
    QUICK_SEEDS,
    QUICK_TIMESTEPS,
    REWARD_ABLATION_VARIANTS,
    RewardConfig,
)
from evaluate import EvaluationConfig, evaluate
from experiment_utils import aggregate_metrics, run_across_seeds
from train import TrainingConfig, train


torch.set_num_threads(1)

MEDIUM_VARIANTS = (
    "baseline",
    "caught_penalty_50",
    "step_penalty_0",
    "pellet_reward_20",
)

RESULT_COLUMNS = (
    "variant",
    "seed",
    "mean_reward",
    "success_rate",
    "mean_pellets",
    "mean_steps",
    "reward_ci_low",
    "reward_ci_high",
    "success_ci_low",
    "success_ci_high",
)


def run_ablation(
    config: AblationConfig | None = None,
    *,
    eval_only: bool = False,
) -> list[dict[str, str | int | float]]:
    """Train/evaluate every reward variant and save the study outputs."""
    cfg = config or AblationConfig()
    variants = _variants_for_config(cfg)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_only:
        print(
            f"Re-evaluating saved models for {len(variants)} reward variants "
            f"and {len(cfg.seeds)} seed(s), "
            f"{cfg.evaluation_episodes} episodes per model; no training."
        )
    else:
        estimate_hours = (
            len(variants) * len(cfg.seeds) * cfg.total_timesteps
            / MEDIUM_CPU_STEPS_PER_SECOND / 3600
        )
        print(
            f"Running {len(variants)} reward variants with {len(cfg.seeds)} seed(s), "
            f"{cfg.total_timesteps:,} steps each on CPU; estimated training time "
            f"about {estimate_hours:.1f} hours."
        )
    records: list[dict[str, str | int | float]] = []
    for variant_name, reward_config in variants:
        print(
            f"Reward ablation {variant_name}: {len(cfg.seeds)} seed(s), "
            f"{cfg.total_timesteps:,} CPU steps per seed."
        )
        variant_records = run_across_seeds(
            cfg.seeds,
            lambda seed: _run_variant_seed(
                variant_name,
                reward_config,
                seed,
                cfg,
                eval_only=eval_only,
            ),
        )
        records.extend(variant_records)
    _write_results(cfg, records)
    _write_plot(cfg, records)
    _write_summary(cfg, records)
    print(f"Reward-ablation CSV: {cfg.results_path}")
    print(f"Reward-ablation summary: {cfg.report_path}")
    print(f"Reward-ablation plot: {cfg.plots_dir}")
    return records


def _run_variant_seed(
    variant_name: str,
    reward_config: RewardConfig,
    seed: int,
    config: AblationConfig,
    *,
    eval_only: bool = False,
) -> dict[str, str | int | float]:
    """Run one PPO training and evaluation pair for a variant and seed."""
    model_dir = config.models_dir / variant_name / f"seed_{seed}"
    log_dir = config.logs_dir / variant_name / f"seed_{seed}"
    model_path = model_dir / "final_model.zip"
    if eval_only:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing saved model for variant {variant_name}, seed {seed}: "
                f"{model_path}"
            )
        model = PPO.load(str(model_path), device="cpu")
    else:
        model = train(
            TrainingConfig(
                models_dir=model_dir,
                logs_dir=log_dir,
                tensorboard_dir=log_dir / "tensorboard",
                total_timesteps=config.total_timesteps,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                eval_freq=config.eval_freq,
                n_eval_episodes=config.evaluation_episodes,
                checkpoint_freq=max(1, config.total_timesteps),
                max_steps=config.max_steps,
                seed=seed,
                reward_config=reward_config,
                verbose=0,
            )
        )
    result = evaluate(
        EvaluationConfig(
            model_path=model_path,
            n_episodes=config.evaluation_episodes,
            seed=seed + 10_000,
            max_steps=config.max_steps,
            verbose=False,
            reward_config=reward_config,
        ),
        model=model,
    )
    return {
        "variant": variant_name,
        "seed": seed,
        "mean_reward": result.mean_reward,
        "success_rate": result.success_rate,
        "mean_pellets": result.mean_pellets_collected,
        "mean_steps": result.mean_steps,
    }


def _summaries(
    config: AblationConfig,
    records: list[dict[str, str | int | float]],
) -> dict[str, dict[str, object]]:
    """Compute metric statistics per reward variant."""
    output: dict[str, dict[str, object]] = {}
    seed_config = MultiSeedConfig(seeds=config.seeds)
    for variant_name, _ in _variants_for_config(config):
        selected = [r for r in records if r["variant"] == variant_name]
        output[variant_name] = aggregate_metrics(
            selected,
            ("mean_reward", "success_rate", "mean_pellets", "mean_steps"),
            seed_config,
        )
    return output


def _write_results(
    config: AblationConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Write per-seed data and aggregated confidence intervals."""
    summaries = _summaries(config, records)
    with config.results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow({
                **row,
                "reward_ci_low": "",
                "reward_ci_high": "",
                "success_ci_low": "",
                "success_ci_high": "",
            })
        for variant_name, _ in _variants_for_config(config):
            summary = summaries[variant_name]
            reward = summary["mean_reward"]
            success = summary["success_rate"]
            writer.writerow({
                "variant": variant_name,
                "seed": "aggregate",
                "mean_reward": reward.mean,
                "success_rate": success.mean,
                "mean_pellets": summary["mean_pellets"].mean,
                "mean_steps": summary["mean_steps"].mean,
                "reward_ci_low": reward.confidence_interval[0],
                "reward_ci_high": reward.confidence_interval[1],
                "success_ci_low": success.confidence_interval[0],
                "success_ci_high": success.confidence_interval[1],
            })


def _write_plot(
    config: AblationConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Plot reward and scale-independent outcomes by variant."""
    variant_names = [name for name, _ in _variants_for_config(config)]
    metrics = (
        ("mean_reward", "Mean reward"),
        ("success_rate", "Success rate"),
        ("mean_pellets", "Mean pellets"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(14, 12))
    for axis, (metric, title) in zip(axes, metrics):
        values_by_variant = [
            [
                float(row[metric])
                for row in records
                if row["variant"] == variant_name
            ]
            for variant_name in variant_names
        ]
        centers = [mean(values) for values in values_by_variant]
        errors = [
            stdev(values) if len(values) > 1 else 0.0
            for values in values_by_variant
        ]
        axis.bar(variant_names, centers, yerr=errors, capsize=3)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
    axes[1].set_ylim(0.0, 1.0)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        config.plots_dir / "reward_ablation.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_summary(
    config: AblationConfig,
    records: list[dict[str, str | int | float]],
) -> None:
    """Describe the largest scale-independent observed ablation effects."""
    baseline = [row for row in records if row["variant"] == "baseline"]
    base_metrics = {
        metric: mean(float(row[metric]) for row in baseline)
        for metric in ("success_rate", "mean_pellets", "mean_steps")
    }
    effects: list[tuple[float, str, str, float]] = []
    for variant_name, _ in _variants_for_config(config):
        if variant_name == "baseline":
            continue
        rows = [row for row in records if row["variant"] == variant_name]
        for metric, baseline_value in base_metrics.items():
            current = mean(float(row[metric]) for row in rows)
            effects.append(
                (abs(current - baseline_value), variant_name, metric, current - baseline_value)
            )
    largest = max(effects, default=(0.0, "none", "none", 0.0))
    lines = [
        "# Reward ablation summary",
        "",
        f"Seeds: {', '.join(map(str, config.seeds))}; "
        f"training steps per variant/seed: {config.total_timesteps:,}.",
        "",
        "Reward totals are not directly comparable across reward scales. "
        "Interpret them alongside scale-independent success rate, pellets "
        "collected, and survival steps.",
        "",
        "Largest observed scale-independent change versus baseline: "
        f"`{largest[1]}` changed `{largest[2]}` by {largest[3]:+.3f}. "
        "This is a descriptive result for the selected seeds, not a causal "
        "claim beyond the controlled reward-term change.",
        "",
        "Quick-mode runs are smoke tests, not evidence of convergence.",
        "",
    ]
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text("\n".join(lines), encoding="utf-8")


def _quick_config() -> AblationConfig:
    """Return small bounds for checking all variant paths end to end."""
    return AblationConfig(
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + QUICK_SEEDS)),
        total_timesteps=QUICK_TIMESTEPS,
        evaluation_episodes=QUICK_EPISODES,
        max_steps=32,
        eval_freq=16,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
    )


def _medium_config() -> AblationConfig:
    """Return a two-seed, four-variant medium training budget."""
    return replace(
        AblationConfig(),
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + MEDIUM_SEEDS)),
        variant_names=MEDIUM_VARIANTS,
        total_timesteps=MEDIUM_TIMESTEPS,
        evaluation_episodes=30,
        eval_freq=10_000,
        models_dir=Path("models/experiments/medium/ablation"),
        logs_dir=Path("logs/experiments/medium/ablation"),
        results_path=Path("results/medium/ablation_results.csv"),
        report_path=Path("results/medium/ablation_summary.md"),
        plots_dir=Path("plots/experiments/medium/ablation"),
    )


def _variants_for_config(
    config: AblationConfig,
) -> tuple[tuple[str, RewardConfig], ...]:
    """Select configured variants while preserving their canonical order."""
    selected = set(config.variant_names)
    return tuple(
        (name, reward_config)
        for name, reward_config in REWARD_ABLATION_VARIANTS
        if name in selected
    )


def main() -> None:
    """Run reward ablation study in quick or full mode."""
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
        config = AblationConfig()
    run_ablation(config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
