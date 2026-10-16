"""Tests for robustness-matrix discovery and plot generation."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluate_matrix import (
    ModelEntry,
    _plot_matrix,
    _write_matrix_csv,
    discover_models,
    latest_ghost_model,
)


def test_discovery_skips_unrelated_files_and_includes_available_models(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "best_model.zip"
    baseline.touch()
    alternating = (
        tmp_path / "experiments" / "alternating" / "seed_42"
        / "round_01_pacman.zip"
    )
    alternating.parent.mkdir(parents=True)
    alternating.touch()
    (tmp_path / "unrelated.zip").touch()

    found = discover_models(tmp_path)
    assert [entry.path for entry in found] == [baseline, alternating]


def test_latest_ghost_model_returns_none_when_missing(tmp_path: Path) -> None:
    assert latest_ghost_model(tmp_path) is None


def test_matrix_writes_csv_and_annotated_plot(tmp_path: Path) -> None:
    rows = [
        {
            "model": "baseline",
            "ghost_type": "rule",
            "mean_reward": 5.0,
            "success_rate": 0.25,
            "mean_pellets": 12.0,
        }
    ]
    csv_path = tmp_path / "results" / "matrix.csv"
    plot_path = tmp_path / "plots" / "matrix.png"

    _write_matrix_csv(csv_path, rows)
    _plot_matrix(plot_path, rows)

    with csv_path.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["model"] == "baseline"
    assert plot_path.exists()
