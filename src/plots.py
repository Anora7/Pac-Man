"""
Matplotlib plotting utilities for the Pac-Man RL project.

This module owns all figure generation. It reads from two data sources:

  * SB3 Monitor CSV files (``*.monitor.csv``) produced during training —
    used for training curves (reward, episode length).
  * :class:`evaluate.EvaluationResult` objects — used for evaluation charts
    (success rate, per-episode breakdown).

All figures are saved as PNG to a configurable ``plots/`` directory.
No interactive display is required — the module always uses the ``Agg``
backend so it runs safely in headless environments.

Usage::

    from plots import PlotConfig, generate_all_plots

    config = PlotConfig(plots_dir=Path("plots"), window=50)
    paths = generate_all_plots(
        log_path=Path("logs/0.monitor.csv"),
        result=eval_result,          # optional
        config=config,
    )
    for p in paths:
        print(f"Saved: {p}")

Or generate individual charts::

    from plots import plot_reward_curve, plot_success_rate

    plot_reward_curve(Path("logs/0.monitor.csv"))
    plot_success_rate(eval_result)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from evaluate import EvaluationResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Settings for figure generation.

    Attributes:
        plots_dir: Directory where PNG files are saved.
        window:    Moving average window size (in episodes).
        figsize:   ``(width, height)`` in inches.
        dpi:       Figure resolution.
        style:     Matplotlib style sheet name.  Falls back to the
                   default style when the name is not recognised.
    """

    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    window: int = 50
    figsize: tuple[int, int] = (10, 6)
    dpi: int = 150
    style: str = "seaborn-v0_8-darkgrid"


# ---------------------------------------------------------------------------
# Training charts  (data source: Monitor CSV)
# ---------------------------------------------------------------------------

def plot_reward_curve(
    log_path: Path,
    config: PlotConfig | None = None,
) -> Path:
    """Plot raw episode reward over every training episode.

    Args:
        log_path: Path to the SB3 Monitor ``.monitor.csv`` file.
        config:   Plot configuration; uses defaults when ``None``.

    Returns:
        Path of the saved ``reward_curve.png``.

    Raises:
        FileNotFoundError: If *log_path* does not exist.
        ValueError:        If the file contains no episode data.
    """
    cfg = config or PlotConfig()
    data = _load_monitor_csv(log_path)
    rewards = data["r"]
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = _new_figure(cfg)
    ax.plot(episodes, rewards, color="#4e9ece", linewidth=0.8,
            alpha=0.7, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Training — Episode Reward")
    ax.legend()

    out = _save_figure(fig, cfg.plots_dir / "reward_curve.png", cfg)
    return out


def plot_moving_average(
    log_path: Path,
    config: PlotConfig | None = None,
) -> Path:
    """Plot a moving-average smoothing of the training reward.

    The raw rewards are shown as a faint background; the smoothed curve
    is drawn prominently on top.

    Args:
        log_path: Path to the SB3 Monitor ``.monitor.csv`` file.
        config:   Plot configuration; uses defaults when ``None``.

    Returns:
        Path of the saved ``moving_average.png``.
    """
    cfg = config or PlotConfig()
    data = _load_monitor_csv(log_path)
    rewards = data["r"]
    episodes = np.arange(1, len(rewards) + 1)

    ma = _moving_average(rewards, cfg.window)
    effective_window = min(cfg.window, len(rewards))
    ma_episodes = np.arange(effective_window, len(rewards) + 1)

    fig, ax = _new_figure(cfg)
    ax.plot(episodes, rewards, color="#4e9ece", linewidth=0.6,
            alpha=0.35, label="Episode reward")
    ax.plot(ma_episodes, ma, color="#e07b39", linewidth=2.0,
            label=f"Moving avg (window={effective_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Training \u2014 Moving Average Reward (window={effective_window})")
    ax.legend()

    out = _save_figure(fig, cfg.plots_dir / "moving_average.png", cfg)
    return out


def plot_episode_length(
    log_path: Path,
    config: PlotConfig | None = None,
) -> Path:
    """Plot episode length (number of steps) over training.

    Also draws a moving-average trend line.

    Args:
        log_path: Path to the SB3 Monitor ``.monitor.csv`` file.
        config:   Plot configuration; uses defaults when ``None``.

    Returns:
        Path of the saved ``episode_length.png``.
    """
    cfg = config or PlotConfig()
    data = _load_monitor_csv(log_path)
    lengths = data["l"]
    episodes = np.arange(1, len(lengths) + 1)

    ma = _moving_average(lengths, cfg.window)
    effective_window = min(cfg.window, len(lengths))
    ma_episodes = np.arange(effective_window, len(lengths) + 1)

    fig, ax = _new_figure(cfg)
    ax.plot(episodes, lengths, color="#6aaa64", linewidth=0.6,
            alpha=0.35, label="Episode length")
    ax.plot(ma_episodes, ma, color="#2d7d46", linewidth=2.0,
            label=f"Moving avg (window={effective_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Training — Episode Length")
    ax.legend()

    out = _save_figure(fig, cfg.plots_dir / "episode_length.png", cfg)
    return out


# ---------------------------------------------------------------------------
# Evaluation chart  (data source: EvaluationResult)
# ---------------------------------------------------------------------------

def plot_success_rate(
    result: EvaluationResult,
    config: PlotConfig | None = None,
) -> Path:
    """Horizontal bar chart showing episode outcome proportions.

    Displays three outcome categories — Cleared, Caught, and Timeout —
    as percentages, with absolute counts annotated on each bar.

    Args:
        result: Aggregated evaluation metrics from :func:`evaluate.evaluate`.
        config: Plot configuration; uses defaults when ``None``.

    Returns:
        Path of the saved ``success_rate.png``.
    """
    cfg = config or PlotConfig()
    n = result.n_episodes

    cleared = sum(ep.cleared for ep in result.episodes)
    caught = sum(ep.caught for ep in result.episodes)
    timeouts = sum(ep.truncated for ep in result.episodes)

    labels = ["Cleared", "Caught", "Timeout"]
    counts = [cleared, caught, timeouts]
    percents = [c / n * 100 for c in counts]
    colours = ["#4caf50", "#f44336", "#ff9800"]

    fig, ax = _new_figure(cfg)
    bars = ax.barh(labels, percents, color=colours, edgecolor="white",
                   linewidth=0.8, height=0.5)

    # Annotate each bar with the absolute count
    for bar, count, pct in zip(bars, counts, percents):
        label_x = max(pct + 0.5, 1.5)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{n}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=11,
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Percentage of episodes (%)")
    ax.set_title(
        f"Evaluation Outcomes  ({n} episodes)\n"
        f"Mean reward: {result.mean_reward:+.1f} ± {result.std_reward:.1f}   "
        f"Avg completion: {result.avg_completion_fraction:.1%}"
    )
    ax.invert_yaxis()  # Cleared on top

    out = _save_figure(fig, cfg.plots_dir / "success_rate.png", cfg)
    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_all_plots(
    log_path: Path | None = None,
    result: EvaluationResult | None = None,
    config: PlotConfig | None = None,
) -> list[Path]:
    """Generate all available plots based on the provided data.

    Training charts are generated when *log_path* is given.
    The success rate chart is generated when *result* is given.
    At least one of the two must be provided.

    When *log_path* is a directory, the first ``*.monitor.csv`` file
    found in that directory is used.

    Args:
        log_path: Path to a Monitor CSV file or the directory that
                  contains it.
        result:   Evaluation result from :func:`evaluate.evaluate`.
        config:   Plot configuration; uses defaults when ``None``.

    Returns:
        List of paths to the saved PNG files.

    Raises:
        ValueError: If neither *log_path* nor *result* is provided.
    """
    if log_path is None and result is None:
        raise ValueError(
            "At least one of 'log_path' or 'result' must be provided."
        )

    cfg = config or PlotConfig()
    saved: list[Path] = []

    if log_path is not None:
        resolved = _resolve_log_path(log_path)
        saved.append(plot_reward_curve(resolved, cfg))
        saved.append(plot_moving_average(resolved, cfg))
        saved.append(plot_episode_length(resolved, cfg))

    if result is not None:
        saved.append(plot_success_rate(result, cfg))

    return saved


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_log_path(path: Path) -> Path:
    """Return a concrete ``.monitor.csv`` path.

    If *path* is a directory, return the first ``*.monitor.csv`` file
    found inside it.  Otherwise return *path* unchanged.

    Raises:
        FileNotFoundError: If no monitor CSV can be found.
    """
    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*.monitor.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No *.monitor.csv file found in directory: {p}"
            )
        return candidates[0]
    return p


def _load_monitor_csv(log_path: Path) -> dict[str, np.ndarray]:
    """Load a SB3 Monitor CSV into numpy arrays.

    The Monitor format has one JSON comment line (starting with ``#``),
    followed by a ``r,l,t`` header, followed by one row per episode.

    Args:
        log_path: Path to the ``.monitor.csv`` file.

    Returns:
        Dictionary with keys ``"r"`` (rewards), ``"l"`` (lengths),
        ``"t"`` (elapsed wall-clock time per episode).

    Raises:
        FileNotFoundError: If *log_path* does not exist.
        ValueError:        If the file contains no episode rows.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Monitor log not found: {path}")

    # Strip comment lines (starting with '#') then parse as CSV
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]

    if not lines:
        raise ValueError(f"No data in monitor log: {path}")

    reader = csv.DictReader(lines)
    rows = list(reader)

    if not rows:
        raise ValueError(f"No episode rows found in monitor log: {path}")

    try:
        rewards = np.array([float(r["r"]) for r in rows])
        lengths = np.array([float(r["l"]) for r in rows])
        times = np.array([float(r["t"]) for r in rows])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Unexpected Monitor CSV format in {path}: {exc}"
        ) from exc

    return {"r": rewards, "l": lengths, "t": times}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing moving average using convolution.

    Args:
        values: 1-D array of values.
        window: Averaging window size.  Clamped to ``len(values)`` when
                the series is shorter than the requested window.

    Returns:
        1-D array of length ``max(0, len(values) - window + 1)``.
    """
    if len(values) == 0:
        return np.array([])
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _new_figure(cfg: PlotConfig) -> tuple[plt.Figure, plt.Axes]:
    """Create a new Figure/Axes pair with the configured style."""
    try:
        plt.style.use(cfg.style)
    except OSError:
        pass  # unknown style — fall back to matplotlib default
    fig, ax = plt.subplots(figsize=cfg.figsize)
    return fig, ax


def _save_figure(fig: plt.Figure, out_path: Path, cfg: PlotConfig) -> Path:
    """Save *fig* to *out_path* and close it to release memory.

    Creates the parent directory when it does not already exist.

    Args:
        fig:      The figure to save.
        out_path: Full path for the output PNG.
        cfg:      Plot configuration (used for DPI).

    Returns:
        *out_path* after saving.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
