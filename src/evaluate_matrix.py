"""Evaluate Pac-Man checkpoints against several ghost policies."""

from __future__ import annotations

import argparse
import csv
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


def discover_models(models_dir: Path) -> list[ModelEntry]:
    """Find the baseline and available extension Pac-Man checkpoints."""
    entries: list[ModelEntry] = []
    baseline = models_dir / "best_model.zip"
    if baseline.exists():
        entries.append(ModelEntry("baseline", baseline))
    else:
        print(f"Skipping missing baseline model: {baseline}")

    for model_path in sorted(models_dir.rglob("*.zip")):
        relative = model_path.relative_to(models_dir)
        parts = relative.parts
        stem = model_path.stem.lower()
        if model_path == baseline:
            continue
        if "improved_observation" in stem or "improved-observation" in stem:
            label = f"improved-observation-{model_path.stem}"
        elif "alternating" in parts and stem.startswith("round_") and "pacman" in stem:
            label = "-".join((*parts[-3:-1], model_path.stem))
        elif "curriculum" in parts:
            label = f"curriculum-{model_path.stem}"
        else:
            continue
        entries.append(ModelEntry(label, model_path))
    return entries


def latest_ghost_model(models_dir: Path) -> Path | None:
    """Return the most recently modified alternating ghost checkpoint."""
    candidates = list(
        (models_dir / "experiments" / "alternating").rglob("*_ghost.zip")
    )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_evaluation_matrix(
    config: RobustnessConfig | None = None,
) -> list[dict[str, str | float]]:
    """Evaluate available models and save CSV plus annotated heatmaps."""
    cfg = config or RobustnessConfig()
    models = discover_models(cfg.models_dir)
    if not models:
        print(f"No Pac-Man models available under {cfg.models_dir}.")
    learned_path = latest_ghost_model(cfg.models_dir)
    ghost_types = ["random", "chase", "rule", "personality"]
    if learned_path is None:
        print("Skipping learned ghost column: no alternating ghost model found.")
    else:
        ghost_types.append("learned")

    rows: list[dict[str, str | float]] = []
    for model_entry in models:
        observation_config = load_observation_config(model_entry.path)
        for ghost_type in ghost_types:
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


def main() -> None:
    """Run the command-line robustness evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_evaluation_matrix(_quick_config() if args.quick else RobustnessConfig())


if __name__ == "__main__":
    main()
