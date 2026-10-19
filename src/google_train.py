"""
PPO curriculum training script for the Google Pac-Man RL project.

Features:
  - Stable-Baselines3 PPO with MlpPolicy.
  - Curriculum progression across chase_strength (0.5 -> 0.75 -> 1.0)
    and active ghost count (2 -> 3 -> 4).
  - Promotes on rolling clear/success rate threshold or stage max steps.
  - Logs per-episode and per-stage metrics to CSV under results/google/.
  - Saves trained models under models/google/.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    GAMMA,
    LEARNING_RATE,
    N_EPOCHS,
    N_STEPS,
    DEFAULT_SEED,
)
from google_config import (
    GOOGLE_CURRICULUM_STAGES,
    GOOGLE_LOGS_DIR,
    GOOGLE_MODELS_DIR,
    GOOGLE_RESULTS_DIR,
    GoogleCurriculumStage,
)
from google_environment import GoogleEnvironment


class GoogleCurriculumCallback(BaseCallback):
    """Callback monitoring episode completion and promoting stages."""

    def __init__(
        self,
        stage: GoogleCurriculumStage,
        window: int = 20,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.stage = stage
        self.window = window
        self.successes: deque[float] = deque(maxlen=window)
        self.episode_records: list[dict[str, Any]] = []
        self.promoted: bool = False
        self.promotion_reason: str = "max_steps"
        self._current_ep_steps: int = 0

    def _on_step(self) -> bool:
        self._current_ep_steps += 1
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        for done, info in zip(dones, infos):
            if done:
                is_success = float(info.get("is_success", False))
                self.successes.append(is_success)

                total_p = info.get("total_pellets", 127)
                rem_p = info.get("pellets_remaining", 0)
                pellets_eaten = total_p - rem_p

                record = {
                    "stage": self.stage.stage_id,
                    "active_ghosts": self.stage.active_ghosts,
                    "chase_strength": self.stage.chase_strength,
                    "episode": len(self.episode_records) + 1,
                    "total_timesteps": self.num_timesteps,
                    "survival_steps": self._current_ep_steps,
                    "pellets_collected": pellets_eaten,
                    "cleared": int(info.get("cleared", False)),
                    "caught": int(info.get("caught", False)),
                    "score": float(info.get("score", 0.0)),
                }
                self.episode_records.append(record)
                self._current_ep_steps = 0

                # Check promotion threshold
                if len(self.successes) >= self.window:
                    rolling_rate = float(np.mean(self.successes))
                    if rolling_rate >= self.stage.success_threshold:
                        self.promoted = True
                        self.promotion_reason = (
                            f"rolling_success_rate_{rolling_rate:.2f}>={self.stage.success_threshold:.2f}"
                        )
                        if self.verbose > 0:
                            print(
                                f"\n>>> Stage {self.stage.stage_id} PROMOTED early at step "
                                f"{self.num_timesteps}! Rolling success: {rolling_rate:.1%}"
                            )
                        return False
        return True


def make_google_env(
    active_ghosts: int,
    chase_strength: float,
    log_dir: Path | None = None,
) -> gym.Env:
    """Create a monitored GoogleEnvironment."""
    env = GoogleEnvironment(
        active_ghost_count=active_ghosts,
        chase_strength=chase_strength,
    )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir / "monitor.csv"))
    return env


def train_google_curriculum(
    stages: tuple[GoogleCurriculumStage, ...] = GOOGLE_CURRICULUM_STAGES,
    total_steps_override: int | None = None,
    seed: int = DEFAULT_SEED,
    models_dir: Path = GOOGLE_MODELS_DIR,
    results_dir: Path = GOOGLE_RESULTS_DIR,
    logs_dir: Path = GOOGLE_LOGS_DIR,
    verbose: int = 1,
) -> PPO:
    """Run full curriculum PPO training across stages."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_episode_records: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []

    model: PPO | None = None

    print("=" * 70)
    print("STARTING GOOGLE PAC-MAN PPO CURRICULUM TRAINING")
    print(f"Stages: {len(stages)} | Base Seed: {seed}")
    print(f"Saving models to: {models_dir}")
    print(f"Saving results to: {results_dir}")
    print("=" * 70)

    for i, stage in enumerate(stages):
        stage_steps = (
            total_steps_override // len(stages)
            if total_steps_override is not None
            else stage.max_steps
        )
        stage_steps = max(stage_steps, 256)

        stage_log_dir = logs_dir / f"stage_{stage.stage_id}"
        env = make_google_env(
            active_ghosts=stage.active_ghosts,
            chase_strength=stage.chase_strength,
            log_dir=stage_log_dir,
        )

        print(f"\n--- [Stage {stage.stage_id}/{len(stages)}] ---")
        print(f"Active Ghosts: {stage.active_ghosts} | Chase Strength: {stage.chase_strength}")
        print(f"Budget: {stage_steps} steps | Promotion Target: {stage.success_threshold:.0%}")

        if model is None:
            # Initialize PPO with MlpPolicy for 16x16 board
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=LEARNING_RATE,
                n_steps=min(N_STEPS, stage_steps),
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                ent_coef=ENT_COEF,
                clip_range=CLIP_RANGE,
                device="cpu",
                seed=seed,
                verbose=verbose,
            )
        else:
            model.set_env(env)

        callback = GoogleCurriculumCallback(
            stage=stage,
            window=stage.success_window,
            verbose=verbose,
        )

        t_start = time.time()
        model.learn(
            total_timesteps=stage_steps,
            callback=callback,
            reset_num_timesteps=False,
        )
        elapsed = time.time() - t_start

        all_episode_records.extend(callback.episode_records)

        # Stage metrics
        ep_count = len(callback.episode_records)
        mean_pellets = (
            float(np.mean([r["pellets_collected"] for r in callback.episode_records]))
            if ep_count > 0 else 0.0
        )
        clear_rate = (
            float(np.mean([r["cleared"] for r in callback.episode_records]))
            if ep_count > 0 else 0.0
        )
        mean_survival = (
            float(np.mean([r["survival_steps"] for r in callback.episode_records]))
            if ep_count > 0 else 0.0
        )
        catch_rate = (
            float(np.mean([r["caught"] for r in callback.episode_records]))
            if ep_count > 0 else 0.0
        )

        stage_summary = {
            "stage": stage.stage_id,
            "active_ghosts": stage.active_ghosts,
            "chase_strength": stage.chase_strength,
            "episodes": ep_count,
            "mean_pellets": round(mean_pellets, 2),
            "clear_rate": round(clear_rate, 4),
            "mean_survival_steps": round(mean_survival, 2),
            "catch_rate": round(catch_rate, 4),
            "promoted_reason": callback.promotion_reason,
            "elapsed_seconds": round(elapsed, 2),
        }
        stage_summaries.append(stage_summary)

        # Save stage checkpoint
        stage_model_path = models_dir / f"stage_{stage.stage_id}_model.zip"
        model.save(stage_model_path)
        print(f"Stage {stage.stage_id} checkpoint saved to {stage_model_path}")
        print(f"Stage Results: Pellets={mean_pellets:.1f}, Clears={clear_rate:.1%}, Survival={mean_survival:.1f}")

    # Save final best model
    final_model_path = models_dir / "google_pacman_ppo.zip"
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Write episode metrics to CSV
    episodes_csv_path = results_dir / "curriculum_episodes.csv"
    if all_episode_records:
        keys = list(all_episode_records[0].keys())
        with open(episodes_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_episode_records)
        print(f"Detailed episode metrics logged to {episodes_csv_path}")

    # Write stage summary to CSV
    summary_csv_path = results_dir / "curriculum_summary.csv"
    if stage_summaries:
        keys = list(stage_summaries[0].keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stage_summaries)
        print(f"Stage curriculum summary logged to {summary_csv_path}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Google Pac-Man PPO Agent")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test (~20k steps total)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total timesteps across stages",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    args = parser.parse_args()

    total_steps = 20_000 if args.smoke_test else args.steps

    print("\n[COMMAND REFERENCE]")
    print("Smoke test:  python src/google_train.py --smoke-test")
    print("Full run:    python src/google_train.py --steps 1000000\n")

    train_google_curriculum(total_steps_override=total_steps, seed=args.seed)


if __name__ == "__main__":
    main()

