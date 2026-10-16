"""Run Pac-Man research extensions from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ablation import _quick_config as quick_ablation
from ablation import run_ablation
from analysis_plots import (
    plot_learning_curves,
    plot_outcome_breakdown,
    plot_value_function_danger_maps,
    plot_visitation_heatmaps,
    plot_alternating_dashboard,
)
from compare_algorithms import _quick_config as quick_comparison
from compare_algorithms import run_comparison
from config import (
    AnalysisPlotConfig,
    DEFAULT_SEED,
    QUICK_EPISODES,
    QUICK_TIMESTEPS,
)
from evaluate import EvaluationConfig, evaluate
from evaluate_matrix import _quick_config as quick_robustness
from evaluate_matrix import run_evaluation_matrix
from train import TrainingConfig, train
from train_alternating import _quick_config as quick_alternating
from train_alternating import run_alternating
from train_curriculum import _quick_config as quick_curriculum
from train_curriculum import run_curriculum


EXPERIMENTS = (
    "alternating",
    "curriculum",
    "dqn",
    "ablation",
    "robustness",
    "plots",
)


def _ensure_baseline_model(quick: bool) -> Path:
    """Train a baseline only when absent; never overwrite existing checkpoints."""
    best_model = ROOT / "models" / "best_model.zip"
    final_model = ROOT / "models" / "final_model.zip"
    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model
    print("Baseline checkpoint missing; creating one without replacing any model.")
    steps = QUICK_TIMESTEPS if quick else 500_000
    train(
        TrainingConfig(
            models_dir=ROOT / "models",
            logs_dir=ROOT / "logs" / "experiments" / "baseline",
            tensorboard_dir=ROOT / "logs" / "experiments" / "baseline" / "tensorboard",
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
    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model
    raise FileNotFoundError("Baseline training completed without saving a model.")


def _run_plots(quick: bool) -> list[str]:
    """Generate analysis plots from the latest compatible local checkpoints."""
    model_path = (
        ROOT
        / "models"
        / "experiments"
        / "algorithm_comparison"
        / "ppo"
        / f"seed_{DEFAULT_SEED}"
        / "final_model.zip"
    )
    if not model_path.exists():
        model_path = _ensure_baseline_model(quick)
    monitor_path = (
        ROOT
        / "logs"
        / "experiments"
        / "algorithm_comparison"
        / "ppo"
        / f"seed_{DEFAULT_SEED}"
        / "0.monitor.csv"
    )
    if not monitor_path.exists():
        monitor_path = (
            ROOT / "logs" / "experiments" / "baseline" / "0.monitor.csv"
        )
    plot_config = AnalysisPlotConfig(
        plots_dir=ROOT / "plots" / "experiments",
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
    alternating_csv = ROOT / "results" / "alternating_rounds.csv"
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


def _run_one(name: str, quick: bool) -> list[str]:
    """Dispatch one experiment and return its primary artifacts."""
    if name == "alternating":
        run_alternating(quick_alternating() if quick else None)
        return [str(ROOT / "results" / "alternating_rounds.csv")]
    if name == "curriculum":
        run_curriculum(quick_curriculum() if quick else None)
        return [
            str(ROOT / "results" / "curriculum_promotions.csv"),
            str(ROOT / "plots" / "experiments" / "curriculum" / "curriculum_progress.png"),
        ]
    if name == "dqn":
        run_comparison(quick_comparison() if quick else None)
        return [
            str(ROOT / "results" / "algorithm_comparison.csv"),
            str(ROOT / "results" / "algorithm_comparison_report.md"),
            str(ROOT / "plots" / "experiments" / "algorithm_comparison"),
        ]
    if name == "ablation":
        run_ablation(quick_ablation() if quick else None)
        return [
            str(ROOT / "results" / "ablation_results.csv"),
            str(ROOT / "results" / "ablation_summary.md"),
            str(ROOT / "plots" / "experiments" / "ablation" / "reward_ablation.png"),
        ]
    if name == "robustness":
        run_evaluation_matrix(quick_robustness() if quick else None)
        return [
            str(ROOT / "results" / "robustness_matrix.csv"),
            str(ROOT / "plots" / "experiments" / "robustness_matrix.png"),
        ]
    if name == "plots":
        return _run_plots(quick)
    raise ValueError(f"Unsupported experiment: {name}")


def main() -> None:
    """Run selected experiment extensions and report their outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=EXPERIMENTS,
        help="Run only the named experiment; repeat to select more than one.",
    )
    args = parser.parse_args()
    selected = args.only or list(EXPERIMENTS)
    quick = args.quick
    completed: dict[str, list[str]] = {}
    skipped = [name for name in EXPERIMENTS if name not in selected]
    print(
        f"Pac-Man experiment runner: {'quick' if quick else 'full'} mode; "
        f"selected: {', '.join(selected)}."
    )
    if not quick:
        estimated_steps = sum({
            "alternating": 3 * 3 * (50_000 + 50_000),
            "curriculum": 3 * 2 * 100_000,
            "dqn": 3 * 2 * 500_000,
            "ablation": 10 * 3 * 500_000,
            "robustness": 0,
            "plots": 0,
        }[name] for name in selected)
        print(
            "Estimated CPU training time is at least "
            f"{estimated_steps / 90 / 3600:.1f} hours, excluding evaluation."
        )
    for name in selected:
        print(f"\n=== Running {name} ===")
        completed[name] = _run_one(name, quick)
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
