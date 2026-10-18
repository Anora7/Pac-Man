"""Train Pac-Man through a staged increase in the number of ghosts."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from config import (
    DEFAULT_SEED,
    GHOST_COUNT,
    MEDIUM_CPU_STEPS_PER_SECOND,
    MEDIUM_SEEDS,
    MEDIUM_TIMESTEPS,
    QUICK_EPISODES,
    QUICK_SEEDS,
    QUICK_TIMESTEPS,
    TRAINING_DEVICE,
    CurriculumConfig,
    ObservationConfig,
)
from experiment_utils import read_monitor_metric, run_across_seeds
from environment import PacmanEnvironment
from observation_metadata import save_observation_config


torch.set_num_threads(1)

PROMOTION_COLUMNS = (
    "seed",
    "from_ghosts",
    "to_ghosts",
    "timestep",
    "reason",
    "rolling_success_rate",
)


class _PromotionCallback(BaseCallback):
    """Stop a stage once its rolling completed-episode success passes target."""

    def __init__(
        self,
        *,
        window: int,
        threshold: float,
    ) -> None:
        super().__init__(verbose=0)
        self.successes: deque[float] = deque(maxlen=window)
        self.threshold = threshold
        self.promotion_success_rate: float | None = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", ())
        infos = self.locals.get("infos", ())
        for done, info in zip(dones, infos):
            if done and "is_success" in info:
                self.successes.append(float(info["is_success"]))
        if len(self.successes) == self.successes.maxlen:
            rate = float(np.mean(self.successes))
            if rate >= self.threshold:
                self.promotion_success_rate = rate
                return False
        return True


@dataclass
class _SeedRun:
    """Outputs needed to aggregate one curriculum seed."""

    seed: int
    promotions: list[dict[str, int | float | str]]
    curves: list[tuple[int, np.ndarray, np.ndarray]]


def run_curriculum(
    config: CurriculumConfig | None = None,
) -> list[dict[str, int | float | str]]:
    """Train across ghost-count stages and report every promotion."""
    cfg = config or CurriculumConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_minutes = (
        cfg.max_steps_per_stage
        * GHOST_COUNT
        * len(cfg.seeds)
        / MEDIUM_CPU_STEPS_PER_SECOND
        / 60
    )
    print(
        f"Curriculum: ghosts 1..{GHOST_COUNT}, {len(cfg.seeds)} seed(s), "
        f"estimated maximum {estimated_minutes:.1f} min on CPU."
    )
    seed_runs = run_across_seeds(
        cfg.seeds,
        lambda seed: _run_seed(seed, cfg),
    )
    promotion_rows = [
        row for seed_run in seed_runs for row in seed_run.promotions
    ]
    _write_promotions(cfg, promotion_rows)
    _plot_curriculum(cfg, seed_runs)
    print(f"Curriculum promotions: {cfg.results_path}")
    print(f"Curriculum learning curve: {cfg.plot_path}")
    return promotion_rows


def _run_seed(seed: int, config: CurriculumConfig) -> _SeedRun:
    """Run each ghost-count stage, continuing one PPO model across stages."""
    seed_models = config.models_dir / f"seed_{seed}"
    seed_logs = config.logs_dir / f"seed_{seed}"
    seed_models.mkdir(parents=True, exist_ok=True)
    seed_logs.mkdir(parents=True, exist_ok=True)
    model: PPO | None = None
    promotions: list[dict[str, int | float | str]] = []
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    total_steps = 0

    for ghost_count in range(1, GHOST_COUNT + 1):
        stage_models = seed_models / f"ghosts_{ghost_count}"
        stage_logs = seed_logs / f"ghosts_{ghost_count}"
        stage_models.mkdir(parents=True, exist_ok=True)
        train_env = make_vec_env(
            lambda count=ghost_count: PacmanEnvironment(
                max_steps=config.max_steps,
                ghost_count=count,
            ),
            n_envs=1,
            seed=seed + ghost_count,
            monitor_dir=str(stage_logs),
            monitor_kwargs={"info_keywords": ("is_success",)},
        )
        eval_env = make_vec_env(
            lambda count=ghost_count: PacmanEnvironment(
                max_steps=config.max_steps,
                ghost_count=count,
            ),
            n_envs=1,
            seed=seed + ghost_count + 100,
            monitor_kwargs={"info_keywords": ("is_success",)},
        )
        if model is None:
            model = PPO(
                "MlpPolicy",
                train_env,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                device=TRAINING_DEVICE,
                seed=seed,
                verbose=0,
            )
        else:
            model.set_env(train_env)

        stage_start = model.num_timesteps
        promotion_callback = _PromotionCallback(
            window=config.success_window,
            threshold=config.promotion_success_threshold,
        )
        callback = [
            promotion_callback,
            EvalCallback(
                eval_env,
                best_model_save_path=str(stage_models / "best"),
                log_path=str(stage_logs),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.evaluation_episodes,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=max(1, config.eval_freq),
                save_path=str(stage_models / "checkpoints"),
                name_prefix="pacman_curriculum",
                verbose=0,
            ),
        ]
        try:
            model.learn(
                total_timesteps=config.max_steps_per_stage,
                callback=callback,
                reset_num_timesteps=ghost_count == 1,
            )
            stage_steps = model.num_timesteps - stage_start
            total_steps += stage_steps
            model_path = stage_models / "pacman_model.zip"
            model.save(str(model_path.with_suffix("")))
            save_observation_config(model_path, ObservationConfig())
            reward_path = stage_logs / "0.monitor.csv"
            if reward_path.exists():
                rewards = read_monitor_metric(reward_path, "r")
                lengths = read_monitor_metric(reward_path, "l")
                if len(rewards):
                    curves.append((total_steps - stage_steps, np.cumsum(lengths), rewards))
            if ghost_count < GHOST_COUNT:
                if promotion_callback.promotion_success_rate is not None:
                    reason = "success_threshold"
                    success_rate = promotion_callback.promotion_success_rate
                else:
                    reason = "max_steps"
                    success_rate = (
                        float(np.mean(promotion_callback.successes))
                        if promotion_callback.successes
                        else 0.0
                    )
                promotion = {
                    "seed": seed,
                    "from_ghosts": ghost_count,
                    "to_ghosts": ghost_count + 1,
                    "timestep": total_steps,
                    "reason": reason,
                    "rolling_success_rate": success_rate,
                }
                promotions.append(promotion)
                print(
                    f"Seed {seed}: promoting {ghost_count} to {ghost_count + 1} "
                    f"ghosts at timestep {total_steps} ({reason})."
                )
        finally:
            train_env.close()
            eval_env.close()
    if model is not None:
        final_path = seed_models / "final_model.zip"
        model.save(str(final_path.with_suffix("")))
        save_observation_config(final_path, ObservationConfig())
    return _SeedRun(seed=seed, promotions=promotions, curves=curves)


def _write_promotions(
    config: CurriculumConfig,
    records: list[dict[str, int | float | str]],
) -> None:
    """Write the curriculum stage promotion table."""
    with config.results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=PROMOTION_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _plot_curriculum(config: CurriculumConfig, seed_runs: list[_SeedRun]) -> None:
    """Plot episode reward over environment steps with promotion markers."""
    config.plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for seed_run in seed_runs:
        for stage_start, episode_steps, rewards in seed_run.curves:
            x_values = stage_start + np.cumsum(episode_steps)
            axis.plot(x_values, rewards, marker=".", label=f"Seed {seed_run.seed}")
        for promotion in seed_run.promotions:
            axis.axvline(
                int(promotion["timestep"]),
                linestyle="--",
                alpha=0.6,
                label=f"Seed {seed_run.seed}: {promotion['to_ghosts']} ghosts",
            )
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Episode reward")
    axis.set_title("Curriculum reward and ghost-count promotions")
    if any(seed_run.curves or seed_run.promotions for seed_run in seed_runs):
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No completed episodes recorded", ha="center")
    figure.tight_layout()
    figure.savefig(config.plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _quick_config() -> CurriculumConfig:
    """Return a minimal curriculum run for pipeline verification."""
    return replace(
        CurriculumConfig(),
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + QUICK_SEEDS)),
        max_steps_per_stage=QUICK_TIMESTEPS,
        promotion_success_threshold=1.0,
        success_window=max(1, QUICK_EPISODES),
        evaluation_episodes=QUICK_EPISODES,
        max_steps=QUICK_TIMESTEPS,
        eval_freq=max(1, QUICK_TIMESTEPS // 2),
        n_steps=QUICK_TIMESTEPS,
        batch_size=max(2, QUICK_TIMESTEPS // 2),
        n_epochs=1,
    )


def _medium_config() -> CurriculumConfig:
    """Return a two-seed curriculum with 100k steps per ghost-count stage."""
    return replace(
        CurriculumConfig(),
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + MEDIUM_SEEDS)),
        max_steps_per_stage=MEDIUM_TIMESTEPS,
        evaluation_episodes=2,
        eval_freq=10_000,
        models_dir=Path("models/experiments/medium/curriculum"),
        logs_dir=Path("logs/experiments/medium/curriculum"),
        results_path=Path("results/medium/curriculum_promotions.csv"),
        plot_path=Path(
            "plots/experiments/medium/curriculum/curriculum_progress.png"
        ),
    )


def main() -> None:
    """Run curriculum training in quick or full mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--medium", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.quick:
        config = _quick_config()
    elif args.medium:
        config = _medium_config()
    else:
        config = CurriculumConfig()
    run_curriculum(config)


if __name__ == "__main__":
    main()
