"""Matplotlib-only diagnostic plots for RL and alternating experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from matplotlib.colors import ListedColormap
from stable_baselines3 import PPO

from config import (
    AnalysisPlotConfig,
    DEFAULT_SEED,
    MAX_STEPS,
    ObservationConfig,
)
from environment import PacmanEnvironment
from evaluate import EvaluationConfig, EvaluationResult, _configure_ghosts, evaluate
from observation_metadata import load_observation_config
from plots import PlotConfig, generate_all_plots


def plot_visitation_heatmaps(
    model_path: Path,
    *,
    ghost_type: str = "rule",
    ghost_model_path: Path | None = None,
    episodes: int = 5,
    seed: int = 42,
    max_steps: int = MAX_STEPS,
    config: AnalysisPlotConfig | None = None,
) -> dict[str, Path]:
    """Plot Pac-Man visits, ghost occupancy, and caught-cell frequency."""
    cfg = config or AnalysisPlotConfig()
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path), device="cpu")
    observation_config = load_observation_config(model_path)
    env = PacmanEnvironment(
        max_steps=max_steps,
        observation_config=observation_config,
    )
    _configure_ghosts(env, ghost_type, ghost_model_path)
    height, width = env.maze.height, env.maze.width
    pacman_visits = np.zeros((height, width), dtype=np.float64)
    ghost_visits = np.zeros((height, width), dtype=np.float64)
    caught_cells = np.zeros((height, width), dtype=np.float64)

    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                row, col = env.pacman_position
                pacman_visits[row, col] += 1
                for ghost in env.ghosts:
                    ghost_visits[ghost.position.row, ghost.position.col] += 1
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                if info["last_step_events"]["caught"]:
                    row, col = env.pacman_position
                    caught_cells[row, col] += 1
    finally:
        env.close()

    walls = env.maze.wall_grid()
    return {
        "pacman_visitation": _save_heatmap(
            pacman_visits,
            walls,
            cfg.plots_dir / f"pacman_visitation_{ghost_type}.png",
            "Pac-Man visitation",
            cfg,
        ),
        "ghost_positions": _save_heatmap(
            ghost_visits,
            walls,
            cfg.plots_dir / f"ghost_positions_{ghost_type}.png",
            "Ghost position frequency",
            cfg,
        ),
        "caught_locations": _save_heatmap(
            caught_cells,
            walls,
            cfg.plots_dir / f"caught_locations_{ghost_type}.png",
            "Pac-Man caught locations",
            cfg,
        ),
    }


def plot_value_function_danger_maps(
    model_path: Path,
    *,
    observation_config: ObservationConfig | None = None,
    config: AnalysisPlotConfig | None = None,
) -> dict[str, Path]:
    """Query a PPO critic across walkable cells in three representative states."""
    cfg = config or AnalysisPlotConfig()
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path), device="cpu")
    obs_config = observation_config or load_observation_config(model_path)
    env = PacmanEnvironment(observation_config=obs_config)
    if model.observation_space.shape != env.observation_space.shape:
        env.close()
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not "
            f"match environment shape {env.observation_space.shape}."
        )
    env.reset(seed=42)
    maze = env.maze
    walkable = [
        (row, col)
        for row in range(maze.height)
        for col in range(maze.width)
        if maze.is_walkable(type(maze.pacman_spawn())(row, col))
    ]
    scenarios = {
        "ghost_nearby": False,
        "ghost_far": False,
        "power_mode_on": True,
    }
    paths: dict[str, Path] = {}
    try:
        for name, power_active in scenarios.items():
            values = np.full((maze.height, maze.width), np.nan, dtype=np.float64)
            for row, col in walkable:
                pacman_position = type(maze.pacman_spawn())(row, col)
                env._pacman_pos = pacman_position
                if name == "ghost_nearby":
                    neighbours = [
                        pacman_position + delta
                        for delta in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        if maze.is_walkable(pacman_position + delta)
                    ]
                    ghost_positions = neighbours or maze.ghost_spawns()
                elif name == "power_mode_on":
                    ghost_positions = [
                        neighbour
                        for neighbour in (
                            pacman_position + delta
                            for delta in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        )
                        if maze.is_walkable(neighbour)
                    ] or maze.ghost_spawns()
                else:
                    ghost_positions = maze.ghost_spawns()
                for index, ghost in enumerate(env.ghosts):
                    position = ghost_positions[index % len(ghost_positions)]
                    ghost.position = position
                    if power_active:
                        ghost.make_vulnerable()
                    else:
                        ghost.restore_normal()
                env._power_mode_active = power_active
                env._power_mode_steps_remaining = (
                    1 if power_active else 0
                )
                observation = env._build_observation()
                tensor, _ = model.policy.obs_to_tensor(observation)
                with torch.no_grad():
                    value = model.policy.predict_values(tensor)
                values[row, col] = float(value.detach().cpu().item())
            paths[name] = _save_heatmap(
                values,
                maze.wall_grid(),
                cfg.plots_dir / f"value_danger_{name}.png",
                f"Critic value — {name.replace('_', ' ')}",
                cfg,
                colorbar_label="Predicted state value",
            )
    finally:
        env.close()
    return paths


def plot_learning_curves(
    monitor_csv: Path,
    result: EvaluationResult | None = None,
    config: AnalysisPlotConfig | None = None,
) -> list[Path]:
    """Generate existing reward, moving-average, episode-length, success plots."""
    cfg = config or AnalysisPlotConfig()
    plot_config = PlotConfig(
        plots_dir=cfg.plots_dir,
        window=cfg.moving_average_window,
        figsize=(int(cfg.figsize[0]), int(cfg.figsize[1])),
        dpi=cfg.dpi,
        style=cfg.style,
    )
    outputs = generate_all_plots(monitor_csv, result=result, config=plot_config)
    outputs.append(plot_smoothed_training_curves(monitor_csv, cfg))
    return outputs


def plot_smoothed_training_curves(
    monitor_csv: Path,
    config: AnalysisPlotConfig | None = None,
) -> Path:
    """Plot moving-average reward, episode length, and success from Monitor."""
    cfg = config or AnalysisPlotConfig()
    with monitor_csv.open(newline="", encoding="utf-8") as source:
        first_line = source.readline()
        if not first_line.startswith("#"):
            source.seek(0)
        reader = csv.DictReader(source)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No completed episodes found in {monitor_csv}.")
    if "is_success" not in rows[0]:
        raise ValueError(
            f"{monitor_csv} lacks the is_success Monitor field. Train with "
            "the current src/train.py to collect success-rate curves."
        )
    rewards = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    lengths = np.asarray([float(row["l"]) for row in rows], dtype=np.float64)
    successes = np.asarray(
        [
            float(row["is_success"].lower() == "true")
            if row["is_success"].lower() in {"true", "false"}
            else float(row["is_success"])
            for row in rows
        ],
        dtype=np.float64,
    )
    window = min(cfg.moving_average_window, len(rows))
    kernel = np.ones(window, dtype=np.float64) / window
    x_values = np.arange(window, len(rows) + 1)
    smoothed = [
        np.convolve(values, kernel, mode="valid")
        for values in (rewards, lengths, successes)
    ]
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(cfg.figsize[0], cfg.figsize[1] * 1.5))
    for axis, values, title, ylabel in zip(
        axes,
        smoothed,
        ("Moving-average episode reward", "Moving-average episode length", "Rolling success rate"),
        ("Reward", "Steps", "Success rate"),
    ):
        axis.plot(x_values, values)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
    axes[2].set_ylim(0.0, 1.0)
    output = cfg.plots_dir / "learning_curves_smoothed.png"
    figure.tight_layout()
    figure.savefig(output, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_alternating_dashboard(
    results_csv: Path,
    config: AnalysisPlotConfig | None = None,
) -> Path:
    """Plot Pac-Man pellet collection and ghost catches by seed and round."""
    cfg = config or AnalysisPlotConfig()
    rows = _read_csv_rows(results_csv)
    if not rows:
        raise ValueError(f"No alternating result rows found in {results_csv}.")
    seed_values = sorted({int(row["seed"]) for row in rows})
    markers = ("o", "s", "^", "D", "v", "P", "X")
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    figure, left = plt.subplots(figsize=cfg.figsize)
    right = left.twinx()
    for index, seed in enumerate(seed_values):
        seed_rows = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in seed_rows]
        marker = markers[index % len(markers)]
        left.plot(
            rounds,
            [float(row["pacman_mean_pellets"]) for row in seed_rows],
            marker=marker,
            label=f"Seed {seed}: Pac-Man mean pellets",
        )
        right.plot(
            rounds,
            [float(row["ghost_catch_rate"]) for row in seed_rows],
            marker=marker,
            linestyle="--",
            label=f"Seed {seed}: Ghost catch rate",
        )
    left.set_xlabel("Alternating training round")
    left.set_ylabel("Mean Pac-Man pellets eaten")
    right.set_ylabel("Ghost catch rate")
    right.set_ylim(0.0, 1.0)
    left.set_title("Pac-Man and ghost co-evolution")
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right)
    output = cfg.plots_dir / "alternating_dashboard.png"
    figure.savefig(output, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_outcome_breakdown(
    results: Mapping[str, EvaluationResult],
    config: AnalysisPlotConfig | None = None,
) -> Path:
    """Plot caught, cleared, and timeout counts for each evaluated model."""
    if not results:
        raise ValueError("At least one evaluation result is required.")
    cfg = config or AnalysisPlotConfig()
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    cleared = [sum(ep.cleared for ep in results[name].episodes) for name in names]
    caught = [sum(ep.caught for ep in results[name].episodes) for name in names]
    timeout = [sum(ep.truncated for ep in results[name].episodes) for name in names]
    figure, axis = plt.subplots(figsize=cfg.figsize)
    axis.bar(names, caught, label="Caught", color="#d73027")
    axis.bar(names, cleared, bottom=caught, label="Cleared", color="#1a9850")
    axis.bar(
        names,
        timeout,
        bottom=np.asarray(caught) + np.asarray(cleared),
        label="Timeout",
        color="#fee08b",
    )
    axis.set_ylabel("Episodes")
    axis.set_title("Evaluation outcomes by model")
    axis.legend()
    axis.tick_params(axis="x", rotation=30)
    output = cfg.plots_dir / "episode_outcomes.png"
    figure.savefig(output, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def capture_episode_frames(
    model_path: Path,
    output_dir: Path,
    *,
    max_steps: int = MAX_STEPS,
) -> list[Path]:
    """Save rendered Pygame frames as PNGs; avoids an optional GIF dependency."""
    model = PPO.load(str(model_path), device="cpu")
    env = PacmanEnvironment(render_mode="human", max_steps=max_steps)
    frames: list[Path] = []
    try:
        observation, _ = env.reset(seed=42)
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(
                int(np.asarray(action).item())
            )
            env.render()
            if env._renderer is None or env._renderer._surface is None:
                print("Skipping frame capture: renderer has no Pygame surface.")
                return []
            frame_path = output_dir / f"frame_{frame_index:05d}.png"
            pygame.image.save(env._renderer._surface, str(frame_path))
            frames.append(frame_path)
            if terminated or truncated or env.window_closed:
                break
    finally:
        env.close()
    return frames


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _save_heatmap(
    values: np.ndarray,
    walls: np.ndarray,
    output_path: Path,
    title: str,
    config: AnalysisPlotConfig,
    colorbar_label: str = "Visit count",
) -> Path:
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=config.figsize)
    image = axis.imshow(masked, cmap="magma", interpolation="nearest")
    wall_mask = np.ma.masked_where(walls == 0.0, walls)
    axis.imshow(
        wall_mask,
        cmap=ListedColormap(["#222222"]),
        interpolation="nearest",
        alpha=0.9,
    )
    axis.set_title(title)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_medium_analysis(episodes: int = 20) -> list[Path]:
    """Compare the baseline and medium PPO models and plot medium training logs."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    model_paths = {
        "baseline": Path("models/best_model.zip"),
        "medium_ppo_seed_42": Path(
            "models/experiments/medium/algorithm_comparison/ppo/"
            "seed_42/final_model.zip"
        ),
    }
    monitor_paths = {
        seed_name: Path(
            "logs/experiments/medium/algorithm_comparison/ppo"
            f"/{seed_name}/0.monitor.csv"
        )
        for seed_name in ("seed_42", "seed_43")
    }
    alternating_results = Path("results/medium/alternating_rounds.csv")
    required_paths = (*model_paths.values(), *monitor_paths.values(), alternating_results)
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing medium analysis inputs: {missing_paths}")

    base_config = AnalysisPlotConfig(
        plots_dir=Path("plots/experiments/medium"),
        evaluation_episodes=episodes,
    )
    results: dict[str, EvaluationResult] = {}
    outputs: list[Path] = []
    for name, model_path in model_paths.items():
        model_config = replace(base_config, plots_dir=base_config.plots_dir / name)
        results[name] = evaluate(
            EvaluationConfig(
                model_path=model_path,
                n_episodes=episodes,
                max_steps=MAX_STEPS,
                seed=DEFAULT_SEED,
                verbose=False,
            )
        )
        outputs.extend(
            plot_visitation_heatmaps(
                model_path,
                episodes=episodes,
                seed=DEFAULT_SEED,
                max_steps=MAX_STEPS,
                config=model_config,
            ).values()
        )
        outputs.extend(
            plot_value_function_danger_maps(
                model_path,
                config=model_config,
            ).values()
        )

    for seed_name, monitor_path in monitor_paths.items():
        log_config = replace(
            base_config,
            plots_dir=base_config.plots_dir / "training_curves" / seed_name,
        )
        outputs.extend(plot_learning_curves(monitor_path, config=log_config))
    outputs.append(plot_alternating_dashboard(alternating_results, base_config))
    outputs.append(plot_outcome_breakdown(results, base_config))
    for output in outputs:
        print(f"Analysis plot: {output}")
    return outputs


def main() -> None:
    """Run analysis plots for the baseline and medium experiment outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Use medium PPO checkpoints, Monitor logs, and output directories.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation and visitation episodes per model (default: 20).",
    )
    args = parser.parse_args()
    if not args.medium:
        parser.error("select --medium to run analysis on medium experiment outputs.")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    run_medium_analysis(args.episodes)


if __name__ == "__main__":
    main()
