"""Make presentation charts from the training and evaluation CSV files.

Run from the project root:
    .\.venvRL\Scripts\python.exe src\make_ppt_plots.py

Charts are saved in plots\ppt\ . The script prints the CSV columns it finds,
so if a chart is skipped you can see why.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/google")
OUT = Path("plots/ppt")
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, YELLOW, BLUE, GREY = "#0b0b10", "#f2f2f2", "#ffd400", "#4a6cff", "#8a8a99"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(fig, ax) -> None:
    fig.patch.set_facecolor(BG)
    for a in np.atleast_1d(ax):
        a.set_facecolor(BG)
        a.tick_params(colors=FG)
        a.xaxis.label.set_color(FG)
        a.yaxis.label.set_color(FG)
        a.title.set_color(FG)
        for spine in a.spines.values():
            spine.set_color(GREY)
        a.grid(alpha=0.15, color=GREY)


def find_col(row: dict, *keywords: str) -> str | None:
    for name in row:
        if all(k in name.lower() for k in keywords):
            return name
    return None


def learning_curve() -> None:
    path = RESULTS / "curriculum_episodes_final.csv"
    if not path.exists():
        path = RESULTS / "curriculum_episodes.csv"
    rows = read_csv(path)
    print(f"[learning curve] {path} columns: {list(rows[0].keys())}")
    pellet_col = find_col(rows[0], "pellet")
    stage_col = find_col(rows[0], "stage")
    if pellet_col is None:
        print("  skipped: no column with 'pellet' in its name")
        return

    pellets = np.array([float(r[pellet_col]) for r in rows])
    window = min(100, max(10, len(pellets) // 20))
    smooth = np.convolve(pellets, np.ones(window) / window, mode="valid")
    x = np.arange(len(smooth)) + window

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(pellets)), pellets, color=BLUE, alpha=0.15, lw=0.6)
    ax.plot(x, smooth, color=YELLOW, lw=2, label=f"Pellets per episode ({window}-episode average)")

    if stage_col is not None:
        stages = [r[stage_col] for r in rows]
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax.axvline(i, color=FG, ls="--", lw=1)
                ax.text(i, ax.get_ylim()[1] * 0.97, f" Stage {stages[i]}", color=FG, va="top")
    ax.axhline(127, color=GREY, ls=":", lw=1)
    ax.text(0, 129, "127 pellets (whole maze)", color=GREY, va="bottom")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Pellets collected")
    ax.set_title("Learning curve across curriculum stages")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GREY, loc="lower right")
    style(fig, ax)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=200)
    plt.close(fig)
    print("  saved plots/ppt/learning_curve.png")


def stage_summary() -> None:
    path = RESULTS / "curriculum_summary_run4.csv"
    if not path.exists():
        path = RESULTS / "curriculum_summary.csv"
    rows = read_csv(path)
    print(f"[stage summary] {path} columns: {list(rows[0].keys())}")
    labels = [f"Stage {r['stage']}\n{r['active_ghosts']} ghosts, {r['chase_strength']}" for r in rows]
    pellets = [float(r["mean_pellets"]) for r in rows]
    steps = [float(r["mean_survival_steps"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, values, title, color in (
        (axes[0], pellets, "Mean pellets collected (of 127)", YELLOW),
        (axes[1], steps, "Mean survival steps", BLUE),
    ):
        bars = ax.bar(labels, values, color=color)
        ax.set_title(title)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", color=FG)
    style(fig, axes)
    fig.tight_layout()
    fig.savefig(OUT / "stage_summary.png", dpi=200)
    plt.close(fig)
    print("  saved plots/ppt/stage_summary.png")


def random_vs_trained() -> None:
    path = RESULTS / "evaluation_episodes.csv"
    if not path.exists():
        print("[random vs trained] skipped: evaluation_episodes.csv not found")
        return
    rows = read_csv(path)
    print(f"[random vs trained] {path} columns: {list(rows[0].keys())}")
    agent_col = find_col(rows[0], "agent")
    pellet_col = find_col(rows[0], "pellet")
    step_col = find_col(rows[0], "surviv") or find_col(rows[0], "step")
    if not (agent_col and pellet_col and step_col):
        print("  skipped: could not find agent, pellet and step columns")
        return

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[agent_col], []).append(r)
    names = list(groups)
    print(f"  groups found: {names} (episodes: {[len(groups[n]) for n in names]})")
    colors = [GREY, YELLOW, BLUE, FG]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, col, title in ((axes[0], pellet_col, "Mean pellets collected"), (axes[1], step_col, "Mean survival steps")):
        means = [np.mean([float(r[col]) for r in groups[n]]) for n in names]
        stds = [np.std([float(r[col]) for r in groups[n]]) for n in names]
        bars = ax.bar(names, means, yerr=stds, color=colors[: len(names)], ecolor=FG, capsize=4)
        ax.set_title(title)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", color=FG)
    style(fig, axes)
    fig.tight_layout()
    fig.savefig(OUT / "random_vs_trained.png", dpi=200)
    plt.close(fig)
    print("  saved plots/ppt/random_vs_trained.png")


if __name__ == "__main__":
    for fn in (learning_curve, stage_summary, random_vs_trained):
        try:
            fn()
        except Exception as exc:  # keep going so one bad chart does not block the rest
            print(f"  {fn.__name__} failed: {exc}")