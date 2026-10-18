"""Evaluate Pac-Man checkpoints against several ghost policies."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_SEED, RobustnessConfig
from evaluate import EvaluationConfig, evaluate
from observation_metadata import load_observation_config


@dataclass(frozen=True)
class ModelEntry:
    """A model archive and its display label."""

    name: str
    path: Path


def discover_models(
    models_dir: Path,
    experiment_models_dir: Path | None = None,
) -> list[ModelEntry]:
    """Find the baseline and final Pac-Man checkpoints from medium runs."""
    entries: list[ModelEntry] = []
    baseline = models_dir / "best_model.zip"
    if baseline.is_file():
        entries.append(ModelEntry("baseline", baseline))
    else:
        print(f"Skipping missing baseline model: {baseline}")

    medium_dir = experiment_models_dir or models_dir / "experiments" / "medium"
    alternating_dir = medium_dir / "alternating"
    if alternating_dir.is_dir():
        for model_path in sorted(alternating_dir.rglob("round_*_pacman.zip")):
            if (
                model_path.is_file()
                and not any(part.endswith("_callbacks") for part in model_path.parts)
                and re.fullmatch(r"round_\d{2}_pacman\.zip", model_path.name)
            ):
                relative = model_path.relative_to(alternating_dir)
                label = "-".join((*relative.parts[:-1], model_path.stem))
                entries.append(ModelEntry(label, model_path))

    curriculum_dir = medium_dir / "curriculum"
    if curriculum_dir.is_dir():
        for model_path in sorted(curriculum_dir.rglob("final_model.zip")):
            if (
                model_path.is_file()
                and not any(part.endswith("_callbacks") for part in model_path.parts)
            ):
                relative = model_path.relative_to(curriculum_dir)
                label = "-".join(("curriculum", *relative.parts[:-1], model_path.stem))
                entries.append(ModelEntry(label, model_path))

    return entries


def latest_ghost_model(
    models_dir: Path,
    experiment_models_dir: Path | None = None,
) -> Path | None:
    """Return the most recently modified alternating ghost checkpoint."""
    medium_dir = experiment_models_dir or models_dir / "experiments" / "medium"
    alternating_dir = medium_dir / "alternating"
    candidates = list(alternating_dir.rglob("*_ghost.zip"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_evaluation_matrix(
    config: RobustnessConfig | None = None,
) -> list[dict[str, str | float]]:
    """Evaluate available models and save CSV plus annotated heatmaps."""
    cfg = config or RobustnessConfig()
    models = discover_models(cfg.models_dir, cfg.experiment_models_dir)
    if not models:
        print(f"No Pac-Man models available under {cfg.models_dir}.")
    print(f"Found {len(models)} Pac-Man models to evaluate.")
    learned_path = latest_ghost_model(
        cfg.models_dir,
        cfg.experiment_models_dir,
    )
    ghost_types = ["random", "chase", "rule", "personality"]
    if learned_path is None:
        print("Skipping learned ghost column: no alternating ghost model found.")
    else:
        print(f"Using learned ghost model: {learned_path}")
        ghost_types.append("learned")

    rows: list[dict[str, str | float]] = []
    total_evaluations = len(models) * len(ghost_types)
    running_count = 0
    for model_entry in models:
        observation_config = load_observation_config(model_entry.path)
        for ghost_type in ghost_types:
            running_count += 1
            print(
                f"Progress {running_count}/{total_evaluations}: "
                f"model={model_entry.name}, ghost_type={ghost_type}"
            )
            result = evaluate(
                EvaluationConfig(
                    model_path=model_entry.path,
                    n_episodes=cfg.episodes,
                    max_steps=cfg.max_steps,
                    seed=cfg.seed,
                    verbose=False,
                    ghost_type=ghost_type,
                    ghost_model_path=learned_path if ghost_type == "learned" else None,
                    observation_config=observation_config,
                )
            )
            rows.append({
                "model": model_entry.name,
                "ghost_type": ghost_type,
                "mean_reward": result.mean_reward,
                "success_rate": result.success_rate,
                "mean_pellets": result.mean_pellets_collected,
            })

    print(
        f"Evaluated {len(rows)} model/ghost combinations across "
        f"{len(models)} Pac-Man models and {len(ghost_types)} ghost policies."
    )
    _write_matrix_csv(cfg.results_path, rows)
    _plot_matrix(cfg.plot_path, rows)
    print(f"Robustness matrix CSV: {cfg.results_path}")
    print(f"Robustness matrix plot: {cfg.plot_path}")
    return rows


def _write_matrix_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    """Write matrix measurements in long-form CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "model",
        "ghost_type",
        "mean_reward",
        "success_rate",
        "mean_pellets",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_matrix(path: Path, rows: list[dict[str, str | float]]) -> Path:
    """Render reward, success, and pellet matrices with values annotated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(dict.fromkeys(str(row["model"]) for row in rows))
    ghost_names = list(dict.fromkeys(str(row["ghost_type"]) for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, max(4, len(model_names) * 0.7)))
    metrics = (
        ("mean_reward", "Mean reward"),
        ("success_rate", "Success rate"),
        ("mean_pellets", "Mean pellets"),
    )
    if not model_names or not ghost_names:
        for axis in axes:
            axis.set_axis_off()
        figure.text(
            0.5,
            0.5,
            "No model/ghost combinations were available.",
            ha="center",
            va="center",
        )
    else:
        for axis, (metric, title) in zip(axes, metrics):
            values = np.full((len(model_names), len(ghost_names)), np.nan)
            for row in rows:
                model_index = model_names.index(str(row["model"]))
                ghost_index = ghost_names.index(str(row["ghost_type"]))
                values[model_index, ghost_index] = float(row[metric])
            image = axis.imshow(values, aspect="auto", cmap="viridis")
            axis.set_title(title)
            axis.set_xticks(range(len(ghost_names)), ghost_names, rotation=35, ha="right")
            axis.set_yticks(range(len(model_names)), model_names)
            for row_index in range(len(model_names)):
                for column_index in range(len(ghost_names)):
                    value = values[row_index, column_index]
                    text = "—" if np.isnan(value) else (
                        f"{value:.1%}" if metric == "success_rate" else f"{value:.1f}"
                    )
                    axis.text(column_index, row_index, text, ha="center", va="center")
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.subplots_adjust(
        left=0.22,
        right=0.97,
        bottom=0.24,
        top=0.88,
        wspace=0.7,
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _quick_config() -> RobustnessConfig:
    """Return fast bounds suitable for a verification run."""
    return replace(
        RobustnessConfig(),
        episodes=1,
        max_steps=32,
        seed=DEFAULT_SEED,
    )


def _medium_config() -> RobustnessConfig:
    """Use medium experiment checkpoints and isolated robustness outputs."""
    return replace(
        RobustnessConfig(),
        experiment_models_dir=Path("models/experiments/medium"),
        episodes=10,
        results_path=Path("results/medium/robustness_matrix.csv"),
        plot_path=Path("plots/experiments/medium/robustness_matrix.png"),
    )


def main() -> None:
    """Run the command-line robustness evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--medium", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Override the number of evaluation episodes per model/ghost cell.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    config = (
        _quick_config()
        if args.quick
        else _medium_config()
        if args.medium
        else RobustnessConfig()
    )
    config = replace(config, episodes=args.episodes)
    run_evaluation_matrix(config)


if __name__ == "__main__":
    main()
