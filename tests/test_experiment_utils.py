"""Tests for shared multi-seed experiment helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AnalysisPlotConfig, MultiSeedConfig
from experiment_utils import (
    aggregate_metrics,
    plot_mean_curve,
    read_monitor_metric,
    run_across_seeds,
    summarize_metric,
)


def test_run_across_seeds_preserves_seed_order() -> None:
    assert run_across_seeds((7, 3, 11), lambda seed: seed * 2) == [14, 6, 22]


@pytest.mark.parametrize("seeds", [(), (42, 42)])
def test_run_across_seeds_rejects_empty_or_duplicate_seeds(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        run_across_seeds(seeds, lambda seed: seed)


def test_summarize_metric_reports_student_t_confidence_interval() -> None:
    summary = summarize_metric((1.0, 2.0, 3.0))
    assert summary.count == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.standard_deviation == pytest.approx(1.0)
    assert summary.confidence_interval == pytest.approx(
        (2.0 - 4.303 / 3**0.5, 2.0 + 4.303 / 3**0.5)
    )


def test_aggregate_metrics_uses_configured_confidence() -> None:
    summaries = aggregate_metrics(
        [{"reward": 1.0}, {"reward": 3.0}],
        config=MultiSeedConfig(seeds=(1, 2), confidence_level=0.90),
    )
    assert summaries["reward"].mean == pytest.approx(2.0)
    assert summaries["reward"].minimum == 1.0
    assert summaries["reward"].maximum == 3.0


def test_read_monitor_metric_skips_sb3_metadata_header(tmp_path: Path) -> None:
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        '#{"t_start":0}\nr,l,t\n1.0,4,0.1\n2.5,7,0.2\n',
        encoding="utf-8",
    )
    assert read_monitor_metric(monitor_path, "r").tolist() == [1.0, 2.5]


def test_read_monitor_metric_converts_boolean_success_values(tmp_path: Path) -> None:
    monitor_path = tmp_path / "success.csv"
    monitor_path.write_text(
        '#{"t_start":0}\nr,l,t,is_success\n1.0,4,0.1,False\n'
        "2.5,7,0.2,True\n",
        encoding="utf-8",
    )
    assert read_monitor_metric(monitor_path, "is_success").tolist() == [0.0, 1.0]


def test_plot_mean_curve_saves_mean_and_band_plot(tmp_path: Path) -> None:
    output = plot_mean_curve(
        ((1.0, 2.0, 3.0), (2.0, 3.0)),
        tmp_path / "mean.png",
        title="Reward",
        y_label="Return",
        config=AnalysisPlotConfig(plots_dir=tmp_path),
    )
    assert output.exists()
