"""Run Pac-Man research extensions from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ablation import _medium_config as medium_ablation
from ablation import _quick_config as quick_ablation
from ablation import run_ablation
from analysis_plots import (
    plot_learning_curves,
    plot_outcome_breakdown,
    plot_value_function_danger_maps,
    plot_visitation_heatmaps,
    plot_alternating_dashboard,
)
from compare_algorithms import _medium_config as medium_comparison
from compare_algorithms import _quick_config as quick_comparison
from compare_algorithms import AlgorithmComparisonConfig
from compare_algorithms import run_comparison
from config import (
    AblationConfig,
    AnalysisPlotConfig,
    AlternatingConfig,
    CurriculumConfig,
    DEFAULT_SEED,
    GHOST_COUNT,
    QUICK_EPISODES,
    QUICK_TIMESTEPS,
    MEDIUM_CPU_STEPS_PER_SECOND,
    MEDIUM_TIMESTEPS,
    REWARD_ABLATION_VARIANTS,
)
from evaluate import EvaluationConfig, evaluate
from evaluate_matrix import _quick_config as quick_robustness
from evaluate_matrix import run_evaluation_matrix
from train import TrainingConfig, train
from train_alternating import _medium_config as medium_alternating
from train_alternating import _quick_config as quick_alternating
from train_alternating import run_alternating
from train_curriculum import _medium_config as medium_curriculum
from train_curriculum import _quick_config as quick_curriculum
from train_curriculum import run_curriculum


torch.set_num_threads(1)

EXPERIMENTS = (
    "alternating",
    "curriculum",
    "dqn",
    "ablation",
    "robustness",
    "plots",
)
MEDIUM_EXPERIMENTS = ("alternating", "curriculum", "dqn", "ablation")


def _ensure_baseline_model(quick: bool, medium: bool = False) -> Path:
    """Train a baseline only when absent; never overwrite existing checkpoints."""
    best_model = ROOT / "models" / "best_model.zip"
    final_model = ROOT / "models" / "final_model.zip"
    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model
    print("Baseline checkpoint missing; creating one without replacing any model.")
    steps = (
        QUICK_TIMESTEPS
        if quick
        else MEDIUM_TIMESTEPS
        if medium
        else 500_000
    )
    baseline_dir = (
        ROOT / "models" / "experiments" / "medium" / "baseline"
        if medium
        else ROOT / "models" / "experiments" / "baseline"
    )
    baseline_logs = (
        ROOT / "logs" / "experiments" / "medium" / "baseline"
        if medium
        else ROOT / "logs" / "experiments" / "baseline"
    )
    train(
        TrainingConfig(
            models_dir=baseline_dir,
            logs_dir=baseline_logs,
            tensorboard_dir=baseline_logs / "tensorboard",
            total_timesteps=steps,
            n_steps=min(2048, steps),
            batch_size=min(64, max(2, steps)),
            n_epochs=1 if quick else 10,
            eval_freq=max(1, steps // 2),
            n_eval_episodes=QUICK_EPISODES if quick else 5,
            checkpoint_freq=max(1, steps),
            max_steps=32 if quick else 1000,
            seed=DEFAULT_SEED,
            verbose=0,
        )
    )
    fallback_best = baseline_dir / "best_model.zip"
    fallback_final = baseline_dir / "final_model.zip"
    if fallback_best.exists():
        return fallback_best
    if fallback_final.exists():
        return fallback_final
    raise FileNotFoundError("Baseline training completed without saving a model.")


def _run_plots(quick: bool, medium: bool = False) -> list[str]:
    """Generate analysis plots from the latest compatible local checkpoints."""
    experiment_root = ROOT / "models" / "experiments"
    if medium:
        experiment_root /= "medium"
    model_path = (
        experiment_root
        / "algorithm_comparison"
        / "ppo"
        / f"seed_{DEFAULT_SEED}"
        / "final_model.zip"
    )
    if not model_path.exists():
        model_path = _ensure_baseline_model(quick, medium)
    log_root = ROOT / "logs" / "experiments"
    if medium:
        log_root /= "medium"
    monitor_path = (
        log_root
        / "algorithm_comparison"
        / "ppo"
        / f"seed_{DEFAULT_SEED}"
        / "0.monitor.csv"
    )
    if not monitor_path.exists():
        baseline_logs = (
            ROOT / "logs" / "experiments" / "medium" / "baseline"
            if medium
            else ROOT / "logs" / "experiments" / "baseline"
        )
        monitor_path = baseline_logs / "0.monitor.csv"
    plot_config = AnalysisPlotConfig(
        plots_dir=(
            ROOT / "plots" / "experiments" / "medium"
            if medium
            else ROOT / "plots" / "experiments"
        ),
        evaluation_episodes=QUICK_EPISODES if quick else 5,
        moving_average_window=5 if quick else 50,
    )
    result = evaluate(
        EvaluationConfig(
            model_path=model_path,
            n_episodes=QUICK_EPISODES if quick else 5,
            max_steps=32 if quick else 1000,
            seed=DEFAULT_SEED,
            verbose=False,
        )
    )
    outputs: list[str] = [
        str(path)
        for path in plot_visitation_heatmaps(
            model_path,
            episodes=QUICK_EPISODES if quick else 5,
            seed=DEFAULT_SEED,
            max_steps=32 if quick else 1000,
            config=plot_config,
        ).values()
    ]
    outputs.extend(
        str(path)
        for path in plot_value_function_danger_maps(
            model_path,
            config=plot_config,
        ).values()
    )
    if monitor_path.exists():
        outputs.extend(
            str(path)
            for path in plot_learning_curves(
                monitor_path,
                result=result,
                config=plot_config,
            )
        )
    else:
        print(f"Skipping learning curves: missing Monitor file {monitor_path}.")
    alternating_csv = (
        ROOT / "results" / "medium" / "alternating_rounds.csv"
        if medium
        else ROOT / "results" / "alternating_rounds.csv"
    )
    if alternating_csv.exists():
        outputs.append(
            str(plot_alternating_dashboard(alternating_csv, plot_config))
        )
    else:
        print(f"Skipping alternating dashboard: missing {alternating_csv}.")
    outputs.append(
        str(plot_outcome_breakdown({"PPO": result}, config=plot_config))
    )
    return outputs


def _run_one(name: str, mode: str) -> list[str]:
    """Dispatch one experiment and return its primary artifacts."""
    quick = mode == "quick"
    medium = mode == "medium"
    if name == "alternating":
        config = (
            quick_alternating()
            if quick
            else medium_alternating()
            if medium
            else AlternatingConfig()
        )
        run_alternating(config)
        return [str(config.results_path)]
    if name == "curriculum":
        config = (
            quick_curriculum()
            if quick
            else medium_curriculum()
            if medium
            else CurriculumConfig()
        )
        run_curriculum(config)
        return [
            str(config.results_path),
            str(config.plot_path),
        ]
    if name == "dqn":
        config = (
            quick_comparison()
            if quick
            else medium_comparison()
            if medium
            else None
        )
        config = config or AlgorithmComparisonConfig()
        run_comparison(config)
        return [
            str(config.results_path),
            str(config.report_path),
            str(config.plots_dir),
        ]
    if name == "ablation":
        config = (
            quick_ablation()
            if quick
            else medium_ablation()
            if medium
            else AblationConfig()
        )
        run_ablation(config)
        return [
            str(config.results_path),
            str(config.report_path),
            str(config.plots_dir / "reward_ablation.png"),
        ]
    if name == "robustness":
        run_evaluation_matrix(quick_robustness() if quick else None)
        return [
            str(ROOT / "results" / "robustness_matrix.csv"),
            str(ROOT / "plots" / "experiments" / "robustness_matrix.png"),
        ]
    if name == "plots":
        return _run_plots(quick, medium)
    raise ValueError(f"Unsupported experiment: {name}")


def _estimated_training_steps(name: str, mode: str) -> int:
    """Estimate training steps for one selected experiment and mode."""
    if name == "alternating":
        config = (
            quick_alternating()
            if mode == "quick"
            else medium_alternating()
            if mode == "medium"
            else AlternatingConfig()
        )
        initial_round_steps = (
            config.pacman_steps_per_round
            if config.initial_pacman_model_path is None
            or not config.initial_pacman_model_path.is_file()
            else 0
        )
        return len(config.seeds) * (
            config.pacman_steps_per_round * config.n_rounds
            + initial_round_steps
            + config.ghost_steps_per_round * config.n_rounds
        )
    if name == "curriculum":
        config = (
            quick_curriculum()
            if mode == "quick"
            else medium_curriculum()
            if mode == "medium"
            else CurriculumConfig()
        )
        return len(config.seeds) * GHOST_COUNT * config.max_steps_per_stage
    if name == "dqn":
        config = (
            quick_comparison()
            if mode == "quick"
            else medium_comparison()
            if mode == "medium"
            else AlgorithmComparisonConfig()
        )
        return 2 * len(config.seeds) * config.total_timesteps
    if name == "ablation":
        config = (
            quick_ablation()
            if mode == "quick"
            else medium_ablation()
            if mode == "medium"
            else AblationConfig()
        )
        return (
            len(config.seeds)
            * len(config.variant_names)
            * config.total_timesteps
        )
    if name == "plots":
        model_root = ROOT / "models" / "experiments"
        if mode == "medium":
            model_root /= "medium"
        model_path = (
            model_root
            / "algorithm_comparison"
            / "ppo"
            / f"seed_{DEFAULT_SEED}"
            / "final_model.zip"
        )
        available_models = (
            ROOT / "models" / "best_model.zip",
            ROOT / "models" / "final_model.zip",
            model_path,
        )
        if not any(path.exists() for path in available_models):
            return (
                QUICK_TIMESTEPS
                if mode == "quick"
                else MEDIUM_TIMESTEPS
                if mode == "medium"
                else 500_000
            )
    return 0


def main() -> None:
    """Run selected experiment extensions and report their outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--medium", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=EXPERIMENTS,
        help="Run only the named experiment; repeat to select more than one.",
    )
    args = parser.parse_args()
    mode = "quick" if args.quick else "medium" if args.medium else "full"
    selected = args.only or (
        list(MEDIUM_EXPERIMENTS) if args.medium else list(EXPERIMENTS)
    )
    selected = list(dict.fromkeys(selected))
    completed: dict[str, list[str]] = {}
    skipped = [name for name in EXPERIMENTS if name not in selected]
    print(
        f"Pac-Man experiment runner: {mode} mode; "
        f"selected: {', '.join(selected)}."
    )
    estimated_steps = sum(
        _estimated_training_steps(name, mode) for name in selected
    )
    if estimated_steps:
        print(
            f"Estimated CPU training time: about "
            f"{estimated_steps / MEDIUM_CPU_STEPS_PER_SECOND / 3600:.1f} "
            "hours, excluding evaluation."
        )
    else:
        print("Estimated runtime: evaluation/plotting only (no training).")
    for name in selected:
        print(f"\n=== Running {name} ===")
        completed[name] = _run_one(name, mode)
    print("\n=== Experiment summary ===")
    for name, paths in completed.items():
        print(f"Ran {name}:")
        for path in paths:
            print(f"  {path}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print("Runner completed.")


if __name__ == "__main__":
    main()
