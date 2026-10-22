"""
Evaluation and demo script for the PAC-MAN RL project.

Features:
  - 30 episodes evaluated across seeds for both the trained PPO agent
    and a random-action Pac-Man baseline.
  - Reports side-by-side comparison of mean pellets, survival steps,
    clear rate, caught rate, and cumulative score.
  - Supports live visual playback via --render flag.
  - Exports metrics to CSV under results/google/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3 import PPO

from config import DEFAULT_SEED
from v2_config import (
    GOOGLE_FPS,
    GOOGLE_MODELS_DIR,
    GOOGLE_RESULTS_DIR,
)
from v2_environment import GoogleEnvironment


@dataclass
class EpisodeMetrics:
    agent_type: str
    episode: int
    seed: int
    score: float
    pellets_collected: int
    survival_steps: int
    cleared: bool
    caught: bool
    ghosts_eaten: int


def run_evaluation_episodes(
    agent_type: str,
    model: PPO | None,
    n_episodes: int = 30,
    base_seed: int = DEFAULT_SEED,
    render: bool = False,
    render_fps: int = GOOGLE_FPS,
    active_ghosts: int = 4,
    chase_strength: float = 1.0,
) -> list[EpisodeMetrics]:
    """Run n_episodes with distinct seeds and record performance."""
    render_mode = "human" if render else None
    env = GoogleEnvironment(
        render_mode=render_mode,
        render_fps=render_fps,
        active_ghost_count=active_ghosts,
        chase_strength=chase_strength,
    )

    metrics_list: list[EpisodeMetrics] = []

    for ep in range(1, n_episodes + 1):
        ep_seed = base_seed + ep
        obs, info = env.reset(seed=ep_seed)
        terminated = False
        truncated = False
        ep_score = 0.0
        steps = 0
        ghosts_eaten = 0
        cleared = False
        caught = False

        while not (terminated or truncated):
            if agent_type == "ppo" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
                act = int(action)
            else:
                # Random action baseline
                act = int(env.action_space.sample())

            obs, reward, terminated, truncated, info = env.step(act)
            ep_score += reward
            steps += 1
            ghosts_eaten += info.get("ghosts_eaten", 0)

            if render:
                # Pace rendering if needed
                time.sleep(1.0 / render_fps)

        cleared = bool(info.get("cleared", False))
        caught = bool(info.get("caught", False))
        total_p = info.get("total_pellets", 127)
        rem_p = info.get("pellets_remaining", 0)
        pellets = total_p - rem_p

        metrics = EpisodeMetrics(
            agent_type=agent_type,
            episode=ep,
            seed=ep_seed,
            score=ep_score,
            pellets_collected=pellets,
            survival_steps=steps,
            cleared=cleared,
            caught=caught,
            ghosts_eaten=ghosts_eaten,
        )
        metrics_list.append(metrics)

    env.close()
    return metrics_list


def compute_summary(metrics: list[EpisodeMetrics]) -> dict[str, float]:
    """Compute aggregated statistics from episode metrics."""
    if not metrics:
        return {}
    n = len(metrics)
    pellets = [m.pellets_collected for m in metrics]
    survivals = [m.survival_steps for m in metrics]
    clears = [1.0 if m.cleared else 0.0 for m in metrics]
    catches = [1.0 if m.caught else 0.0 for m in metrics]
    scores = [m.score for m in metrics]

    return {
        "episodes": float(n),
        "mean_pellets": float(np.mean(pellets)),
        "std_pellets": float(np.std(pellets)),
        "mean_survival_steps": float(np.mean(survivals)),
        "std_survival_steps": float(np.std(survivals)),
        "clear_rate": float(np.mean(clears)),
        "caught_rate": float(np.mean(catches)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def evaluate_and_compare(
    model_path: Path | None = None,
    n_episodes: int = 30,
    base_seed: int = DEFAULT_SEED,
    render: bool = False,
    results_dir: Path = GOOGLE_RESULTS_DIR,
    active_ghosts: int = 4,
    chase_strength: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate both PPO agent and Random Baseline and print comparison."""
    results_dir.mkdir(parents=True, exist_ok=True)

    model = None
    has_model = False
    if model_path is not None and model_path.exists():
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(str(model_path), device="cpu")
        has_model = True
    else:
        print(f"No model found at {model_path}. Evaluating Random baseline only.")

    all_records: list[EpisodeMetrics] = []

    # 1. Random Baseline Evaluation
    print(
        f"\nEvaluating Random-Action Pac-Man baseline across {n_episodes} seeds "
        f"(active_ghosts={active_ghosts}, chase_strength={chase_strength})..."
    )
    random_metrics = run_evaluation_episodes(
        agent_type="random",
        model=None,
        n_episodes=n_episodes,
        base_seed=base_seed,
        render=render and not has_model,
        active_ghosts=active_ghosts,
        chase_strength=chase_strength,
    )
    all_records.extend(random_metrics)
    random_summary = compute_summary(random_metrics)

    # 2. Trained Agent Evaluation
    ppo_summary: dict[str, float] = {}
    if has_model and model is not None:
        print(
            f"Evaluating Trained PPO Pac-Man agent across {n_episodes} seeds "
            f"(active_ghosts={active_ghosts}, chase_strength={chase_strength})..."
        )
        ppo_metrics = run_evaluation_episodes(
            agent_type="ppo",
            model=model,
            n_episodes=n_episodes,
            base_seed=base_seed,
            render=render,
            active_ghosts=active_ghosts,
            chase_strength=chase_strength,
        )
        all_records.extend(ppo_metrics)
        ppo_summary = compute_summary(ppo_metrics)

    # 3. Print Side-by-Side Comparison
    config_banner = f"ghosts={active_ghosts}, chase_strength={chase_strength}"
    print("\n" + "=" * 76)
    print(f"{'PAC-MAN EVALUATION RESULTS (' + str(n_episodes) + ' EPISODES)':^76}")
    print(f"{'[' + config_banner + ']':^76}")
    print("=" * 76)
    print(f"{'Metric':<28} | {'Random Baseline':<20} | {'Trained PPO Agent':<20}")
    print("-" * 76)

    def _fmt(val: float, is_pct: bool = False) -> str:
        return f"{val:.1%}" if is_pct else f"{val:.1f}"

    def _fmt_pm(mean: float, std: float) -> str:
        return f"{mean:.1f} +/- {std:.1f}"

    if ppo_summary:
        print(f"{'Mean Pellets Collected':<28} | {_fmt_pm(random_summary['mean_pellets'], random_summary['std_pellets']):<20} | {_fmt_pm(ppo_summary['mean_pellets'], ppo_summary['std_pellets']):<20}")
        print(f"{'Mean Survival Steps':<28} | {_fmt_pm(random_summary['mean_survival_steps'], random_summary['std_survival_steps']):<20} | {_fmt_pm(ppo_summary['mean_survival_steps'], ppo_summary['std_survival_steps']):<20}")
        print(f"{'Maze Clear Rate':<28} | {_fmt(random_summary['clear_rate'], True):<20} | {_fmt(ppo_summary['clear_rate'], True):<20}")
        print(f"{'Caught by Ghost Rate':<28} | {_fmt(random_summary['caught_rate'], True):<20} | {_fmt(ppo_summary['caught_rate'], True):<20}")
        print(f"{'Mean Cumulative Score':<28} | {_fmt_pm(random_summary['mean_score'], random_summary['std_score']):<20} | {_fmt_pm(ppo_summary['mean_score'], ppo_summary['std_score']):<20}")
    else:
        print(f"{'Mean Pellets Collected':<28} | {_fmt_pm(random_summary['mean_pellets'], random_summary['std_pellets']):<20} | {'N/A (no model)':<20}")
        print(f"{'Mean Survival Steps':<28} | {_fmt_pm(random_summary['mean_survival_steps'], random_summary['std_survival_steps']):<20} | {'N/A (no model)':<20}")
        print(f"{'Maze Clear Rate':<28} | {_fmt(random_summary['clear_rate'], True):<20} | {'N/A (no model)':<20}")
        print(f"{'Caught by Ghost Rate':<28} | {_fmt(random_summary['caught_rate'], True):<20} | {'N/A (no model)':<20}")
        print(f"{'Mean Cumulative Score':<28} | {_fmt_pm(random_summary['mean_score'], random_summary['std_score']):<20} | {'N/A (no model)':<20}")

    print("=" * 76 + "\n")

    # 4. Save episode-level metrics to CSV
    csv_file = results_dir / "evaluation_episodes.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent_type",
                "episode",
                "seed",
                "score",
                "pellets_collected",
                "survival_steps",
                "cleared",
                "caught",
                "ghosts_eaten",
            ],
        )
        writer.writeheader()
        for r in all_records:
            writer.writerow(r.__dict__)
    print(f"Episode evaluation results exported to {csv_file}")

    return random_summary, ppo_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAC-MAN Agent")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(GOOGLE_MODELS_DIR / "google_pacman_ppo.zip"),
        help="Path to trained PPO model archive",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes per agent (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Watch live rendering via Pygame window",
    )
    parser.add_argument(
        "--ghosts",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of active ghosts (default: 4)",
    )
    parser.add_argument(
        "--chase-strength",
        type=float,
        default=1.0,
        help="Ghost chase strength probability 0.0-1.0 (default: 1.0)",
    )
    args = parser.parse_args()

    evaluate_and_compare(
        model_path=Path(args.model_path),
        n_episodes=args.episodes,
        base_seed=args.seed,
        render=args.render,
        active_ghosts=args.ghosts,
        chase_strength=args.chase_strength,
    )


if __name__ == "__main__":
    main()


