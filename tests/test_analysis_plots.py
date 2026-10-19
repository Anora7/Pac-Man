"""Fast rendering and aggregation tests for analysis plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import analysis_plots
from analysis_plots import (
    _save_heatmap,
    plot_alternating_dashboard,
    plot_endgame_variant_comparison,
    plot_outcome_breakdown,
    plot_smoothed_training_curves,
)
from config import AnalysisPlotConfig
from evaluate import EpisodeResult, EvaluationResult


def test_heatmap_saves_file_and_masks_wall_cells(tmp_path: Path) -> None:
    config = AnalysisPlotConfig(plots_dir=tmp_path)
    values = np.ones((3, 3), dtype=np.float64)
    walls = np.zeros((3, 3), dtype=np.float32)
    walls[0, 0] = 1.0
    output = _save_heatmap(values, walls, tmp_path / "map.png", "Visits", config)
    assert output.exists()


def test_alternating_dashboard_reads_results_csv_and_omits_round_zero_catch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "rounds.csv"
    csv_path.write_text(
        "\ufeffseed,round,pacman_success_rate,pacman_mean_pellets,ghost_catch_rate\n"
        "42,0,0.0,3.0,0.0\n"
        "42,1,0.0,5.0,0.4\n"
        "42,2,0.0,6.0,0.5\n"
        "42,3,0.0,7.0,0.6\n"
        "43,0,0.0,4.0,0.0\n"
        "43,1,0.0,6.0,0.5\n"
        "43,2,0.0,7.0,0.6\n"
        "43,3,0.0,8.0,0.7\n",
        encoding="utf-8",
    )
    figures = []
    close_figure = plt.close
    monkeypatch.setattr(analysis_plots.plt, "close", lambda figure: figures.append(figure))
    output = plot_alternating_dashboard(
        csv_path,
        AnalysisPlotConfig(plots_dir=tmp_path / "plots"),
    )
    assert output.exists()
    figure = figures.pop()
    ghost_lines = figure.axes[1].lines
    assert len(ghost_lines) == 2
    assert all(list(line.get_xdata()) == [1, 2, 3] for line in ghost_lines)
    close_figure(figure)


def test_outcome_breakdown_writes_stacked_chart(tmp_path: Path) -> None:
    episodes = [
        EpisodeResult(1, -2.0, 2, 0, 69, False, True, False, 0.0),
        EpisodeResult(2, 1.0, 2, 1, 69, False, False, True, 0.1),
    ]
    result = EvaluationResult(2, episodes, -0.5, 1.5, 2.0, 0.5, 0.0, 0.05)
    output = plot_outcome_breakdown(
        {"test": result},
        AnalysisPlotConfig(plots_dir=tmp_path),
    )
    assert output.exists()


def test_smoothed_training_curves_use_monitor_success_field(tmp_path: Path) -> None:
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        '#{"t_start":0}\n'
        "r,l,t,is_success\n"
        "1.0,2,0.1,0\n"
        "3.0,4,0.2,1\n",
        encoding="utf-8",
    )
    output = plot_smoothed_training_curves(
        monitor_path,
        AnalysisPlotConfig(
            plots_dir=tmp_path / "plots",
            moving_average_window=2,
        ),
    )
    assert output.exists()


def test_endgame_comparison_uses_deterministic_rows_and_seed_error_bars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def write_csv(path: Path, variant: str, trained_pellets: float, clears: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "variant,model,policy,episodes,mean_pellets,mean_steps,clears,loop_rate\n"
            f"{variant},baseline,deterministic,30,44.0,80.0,0,0.0\n"
            f"{variant},trained,deterministic,30,{trained_pellets},70.0,{clears},0.0\n"
            f"{variant},trained,stochastic,30,99.0,70.0,30,0.0\n",
            encoding="utf-8",
        )
        return path

    figures = []
    monkeypatch.setattr(
        analysis_plots.plt,
        "close",
        lambda figure: figures.append(figure),
    )
    feature_csvs = tuple(
        write_csv(tmp_path / f"feature_{seed}.csv", "features_shaping", 50 + seed, seed)
        for seed in range(3)
    )
    ghost_csvs = tuple(
        write_csv(
            tmp_path / f"ghost_{seed}.csv",
            "features_shaping_ghost",
            40 + seed,
            seed + 1,
        )
        for seed in range(3)
    )
    output = plot_endgame_variant_comparison(
        feature_csvs,
        ghost_csvs,
        tmp_path / "plots" / "comparison.png",
    )
    assert output.exists()
    assert output.stat().st_size > 0
    figure = figures[0]
    bar_values = [
        [
            container.datavalues.tolist()
            for container in axis.containers
            if hasattr(container, "datavalues")
        ][0]
        for axis in figure.axes
    ]
    assert bar_values == [[0.0, 1.0, 2.0], [44.0, 51.0, 41.0]]
    plt.close(figure)
