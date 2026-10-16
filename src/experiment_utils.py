"""Shared seed execution, metric aggregation, and Monitor curve plotting."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Callable, Mapping, Sequence, TypeVar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import AnalysisPlotConfig, MultiSeedConfig


ResultT = TypeVar("ResultT")

_T_95_CRITICAL = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


@dataclass(frozen=True)
class MetricSummary:
    """Descriptive statistics and a confidence interval for one metric."""

    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    confidence_level: float
    confidence_interval: tuple[float, float]


def run_across_seeds(
    seeds: Sequence[int],
    run_one: Callable[[int], ResultT],
) -> list[ResultT]:
    """Execute one experiment callback per unique seed, in supplied order."""
    if not seeds:
        raise ValueError("At least one experiment seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique.")
    return [run_one(seed) for seed in seeds]


def summarize_metric(
    values: Sequence[float],
    confidence_level: float = MultiSeedConfig().confidence_level,
) -> MetricSummary:
    """Compute sample statistics and a two-sided Student-t confidence interval."""
    if not values:
        raise ValueError("At least one metric value is required.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    numeric = [float(value) for value in values]
    center = mean(numeric)
    deviation = stdev(numeric) if len(numeric) > 1 else 0.0
    if len(numeric) == 1:
        margin = 0.0
    elif confidence_level == 0.95 and len(numeric) <= 31:
        margin = _T_95_CRITICAL[len(numeric) - 1] * deviation / math.sqrt(len(numeric))
    else:
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        margin = z_value * deviation / math.sqrt(len(numeric))
    return MetricSummary(
        count=len(numeric),
        mean=center,
        standard_deviation=deviation,
        minimum=min(numeric),
        maximum=max(numeric),
        confidence_level=confidence_level,
        confidence_interval=(center - margin, center + margin),
    )


def aggregate_metrics(
    records: Sequence[Mapping[str, float]],
    metrics: Sequence[str] | None = None,
    config: MultiSeedConfig | None = None,
) -> dict[str, MetricSummary]:
    """Aggregate requested numeric metric fields across seed result records."""
    if not records:
        raise ValueError("At least one seed result record is required.")
    selected = list(metrics) if metrics is not None else list(records[0])
    settings = config or MultiSeedConfig()
    summaries: dict[str, MetricSummary] = {}
    for metric in selected:
        try:
            values = [float(record[metric]) for record in records]
        except KeyError as error:
            raise ValueError(f"Metric {metric!r} is missing from a seed result.") from error
        summaries[metric] = summarize_metric(values, settings.confidence_level)
    return summaries


def read_monitor_metric(monitor_csv: Path, metric: str) -> np.ndarray:
    """Read one numeric column from an SB3 Monitor CSV."""
    with monitor_csv.open(newline="", encoding="utf-8") as source:
        header = source.readline()
        if not header.startswith("#"):
            source.seek(0)
        reader = csv.DictReader(source)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"Monitor CSV {monitor_csv} has no {metric!r} column.")
        values: list[float] = []
        for row in reader:
            value = row[metric]
            if value.lower() in {"true", "false"}:
                values.append(float(value.lower() == "true"))
            else:
                values.append(float(value))
        return np.asarray(values, dtype=np.float64)


def plot_mean_curve(
    curves: Sequence[Sequence[float]],
    output_path: Path,
    *,
    title: str,
    y_label: str,
    config: AnalysisPlotConfig | None = None,
) -> Path:
    """Plot per-episode mean and standard-deviation bands across seed curves."""
    if not curves or any(len(curve) == 0 for curve in curves):
        raise ValueError("At least one non-empty seed curve is required.")
    shared_length = min(len(curve) for curve in curves)
    matrix = np.asarray(
        [np.asarray(curve[:shared_length], dtype=np.float64) for curve in curves]
    )
    x_values = np.arange(1, shared_length + 1)
    curve_mean = matrix.mean(axis=0)
    curve_std = matrix.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(shared_length)
    cfg = config or AnalysisPlotConfig(plots_dir=output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=cfg.figsize)
    axis.plot(x_values, curve_mean, label="Mean across seeds")
    axis.fill_between(
        x_values,
        curve_mean - curve_std,
        curve_mean + curve_std,
        alpha=0.25,
        label="±1 standard deviation",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_monitor_curves(
    monitor_csvs: Sequence[Path],
    metric: str,
    output_path: Path,
    *,
    title: str,
    y_label: str,
    config: AnalysisPlotConfig | None = None,
) -> Path:
    """Read one Monitor metric per seed and plot its aggregated learning curve."""
    curves = [read_monitor_metric(path, metric) for path in monitor_csvs]
    return plot_mean_curve(
        curves,
        output_path,
        title=title,
        y_label=y_label,
        config=config,
    )
