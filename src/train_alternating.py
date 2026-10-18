"""Alternating PPO training for Pac-Man and its rule-based/learned ghosts."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from config import (
    DEFAULT_SEED,
    MEDIUM_ALTERNATING_ROUNDS,
    MEDIUM_ALTERNATING_STEPS_PER_ROUND,
    MEDIUM_GHOST_STEPS_PER_ROUND,
    MEDIUM_CPU_STEPS_PER_SECOND,
    MEDIUM_SEEDS,
    TRAINING_DEVICE,
    AlternatingConfig,
    ObservationConfig,
)
from environment import PacmanEnvironment
from evaluate import EvaluationConfig, evaluate
from experiment_utils import run_across_seeds
from ghost_environment import GhostEnvironment
from ghost_policies import LearnedGhostPolicy
from observation_metadata import load_observation_config, save_observation_config


torch.set_num_threads(1)

RESULT_COLUMNS = (
    "seed",
    "round",
    "pacman_mean_reward",
    "pacman_success_rate",
    "pacman_mean_pellets",
    "pacman_survival_steps",
    "ghost_catch_rate",
)


def run_alternating(
    config: AlternatingConfig | None = None,
) -> list[dict[str, float | int]]:
    """Run alternating ghost PPO and Pac-Man PPO rounds and save metrics."""
    cfg = config or AlternatingConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_steps = len(cfg.seeds) * (
        cfg.pacman_steps_per_round * cfg.n_rounds
        + cfg.ghost_steps_per_round * cfg.n_rounds
    )
    estimated_hours = estimated_steps / MEDIUM_CPU_STEPS_PER_SECOND / 3600
    print(
        f"Alternating training: {cfg.n_rounds} rounds, {len(cfg.seeds)} seed(s), "
        f"{cfg.pacman_steps_per_round:,} Pac-Man and "
        f"{cfg.ghost_steps_per_round:,} ghost steps per round; estimated CPU "
        f"training time about {estimated_hours:.1f} hours."
    )
    records: list[dict[str, float | int]] = []
    with cfg.results_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        seed_records = run_across_seeds(
            cfg.seeds,
            lambda seed: _run_seed(seed, cfg, writer, result_file),
        )
        for seed_result in seed_records:
            records.extend(seed_result)
    _compare_final_ghosts_to_baseline(cfg)
    print(f"Alternating experiment results: {cfg.results_path}")
    return records


def _run_seed(
    seed: int,
    config: AlternatingConfig,
    writer: csv.DictWriter,
    result_file: TextIO,
) -> list[dict[str, float | int]]:
    """Run all alternating rounds for one reproducible seed."""
    rng = random.Random(seed)
    seed_models = config.models_dir / f"seed_{seed}"
    seed_logs = config.logs_dir / f"seed_{seed}"
    seed_models.mkdir(parents=True, exist_ok=True)
    pacman_path = _prepare_round_zero(config, seed, seed_models, seed_logs)
    pacman_pool: list[Path] = [pacman_path]
    ghost_pool: list[Path] = []
    records: list[dict[str, float | int]] = []
    initial_record = _evaluate_pacman(
        pacman_path,
        ghost_model_path=None,
        seed=seed,
        round_index=0,
        config=config,
    )
    writer.writerow(initial_record)
    result_file.flush()
    records.append(initial_record)

    for round_index in range(1, config.n_rounds + 1):
        started = time.monotonic()
        pacman_opponent = _sample_opponent(
            pacman_pool,
            rng,
            config.opponent_pool_latest_probability,
        )
        ghost_path = seed_models / f"round_{round_index:02d}_ghost.zip"
        _train_ghost(
            pacman_opponent,
            ghost_path,
            seed_logs / f"round_{round_index:02d}_ghost",
            seed + round_index,
            config,
        )
        ghost_pool.append(ghost_path)
        ghost_opponent = _sample_opponent(
            ghost_pool,
            rng,
            config.opponent_pool_latest_probability,
        )
        next_pacman_path = seed_models / f"round_{round_index:02d}_pacman.zip"
        _train_pacman(
            pacman_path,
            ghost_opponent,
            next_pacman_path,
            seed_logs / f"round_{round_index:02d}_pacman",
            seed + round_index,
            config,
        )
        pacman_path = next_pacman_path
        pacman_pool.append(pacman_path)
        record = _evaluate_pacman(
            pacman_path,
            ghost_model_path=ghost_path,
            seed=seed + round_index,
            round_index=round_index,
            config=config,
        )
        record["seed"] = seed
        record["ghost_catch_rate"] = _evaluate_ghost(
            ghost_path,
            pacman_path,
            seed + round_index,
            config,
        )
        writer.writerow(record)
        result_file.flush()
        records.append(record)
        elapsed = time.monotonic() - started
        remaining = (config.n_rounds - round_index) * elapsed
        print(
            f"Seed {seed}, round {round_index}/{config.n_rounds} complete. "
            f"Estimated remaining: {remaining / 60:.1f} min."
        )
    return records


def _prepare_round_zero(
    config: AlternatingConfig,
    seed: int,
    models_dir: Path,
    logs_dir: Path,
) -> Path:
    """Copy a compatible baseline Pac-Man model or train round zero."""
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "round_00_pacman.zip"
    initial = config.initial_pacman_model_path
    if initial is not None and initial.exists():
        initial_observation_config = load_observation_config(initial)
        if initial_observation_config == ObservationConfig():
            shutil.copy2(initial, output_path)
            save_observation_config(output_path, initial_observation_config)
            return output_path
        print(
            "Initial Pac-Man checkpoint uses optional observations; "
            "training round zero from scratch with baseline observations."
        )
    _train_pacman(
        None,
        None,
        output_path,
        logs_dir / "round_00_pacman",
        seed,
        replace(
            config,
            n_rounds=1,
            pacman_steps_per_round=config.pacman_steps_per_round,
        ),
    )
    return output_path


def _train_ghost(
    pacman_model_path: Path,
    output_path: Path,
    logs_dir: Path,
    seed: int,
    config: AlternatingConfig,
) -> None:
    """Train one shared PPO ghost policy against a frozen Pac-Man model."""
    train_env = make_vec_env(
        lambda: GhostEnvironment(
            pacman_model_path=pacman_model_path,
            max_steps=config.max_steps,
            shared_policy=True,
        ),
        n_envs=1,
        seed=seed,
        monitor_dir=str(logs_dir / "monitor"),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    eval_env = make_vec_env(
        lambda: GhostEnvironment(
            pacman_model_path=pacman_model_path,
            max_steps=config.max_steps,
            shared_policy=True,
        ),
        n_envs=1,
        seed=seed + 1,
    )
    callbacks = _callbacks(output_path, logs_dir, eval_env, config)
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
    try:
        model.learn(
            total_timesteps=config.ghost_steps_per_round,
            callback=callbacks,
            reset_num_timesteps=True,
        )
        model.save(str(output_path.with_suffix("")))
    finally:
        train_env.close()
        eval_env.close()


def _train_pacman(
    initial_model_path: Path | None,
    ghost_model_path: Path | None,
    output_path: Path,
    logs_dir: Path,
    seed: int,
    config: AlternatingConfig,
) -> None:
    """Train or continue Pac-Man PPO against frozen ghost policies."""
    train_env = make_vec_env(
        _pacman_env_factory(ghost_model_path, config.max_steps),
        n_envs=1,
        seed=seed,
        monitor_dir=str(logs_dir / "monitor"),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    eval_env = make_vec_env(
        _pacman_env_factory(ghost_model_path, config.max_steps),
        n_envs=1,
        seed=seed + 1,
    )
    callbacks = _callbacks(output_path, logs_dir, eval_env, config)
    if initial_model_path is not None and initial_model_path.exists():
        model = PPO.load(
            str(initial_model_path),
            env=train_env,
            device=TRAINING_DEVICE,
        )
        resume_timesteps = False
    else:
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
        resume_timesteps = True
    try:
        model.learn(
            total_timesteps=config.pacman_steps_per_round,
            callback=callbacks,
            reset_num_timesteps=resume_timesteps,
        )
        model.save(str(output_path.with_suffix("")))
        save_observation_config(output_path, ObservationConfig())
    finally:
        train_env.close()
        eval_env.close()


def _pacman_env_factory(ghost_model_path: Path | None, max_steps: int):
    """Return a factory that gives each Pac-Man env the shared frozen ghost."""
    def create() -> PacmanEnvironment:
        env = PacmanEnvironment(max_steps=max_steps)
        if ghost_model_path is not None:
            ghost_model = PPO.load(str(ghost_model_path), device=TRAINING_DEVICE)
            for ghost in env.ghosts:
                ghost.policy = LearnedGhostPolicy(ghost_model)
        return env
    return create


def _callbacks(
    output_path: Path,
    logs_dir: Path,
    eval_env: Any,
    config: AlternatingConfig,
) -> list[Any]:
    """Create isolated evaluation and checkpoint callbacks."""
    callback_dir = output_path.parent / f"{output_path.stem}_callbacks"
    callback_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return [
        EvalCallback(
            eval_env,
            best_model_save_path=str(callback_dir),
            log_path=str(logs_dir / "evaluation"),
            eval_freq=max(1, config.eval_freq),
            n_eval_episodes=config.evaluation_episodes,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(1, config.eval_freq),
            save_path=str(callback_dir / "checkpoints"),
            name_prefix=output_path.stem,
            verbose=0,
        ),
    ]


def _sample_opponent(
    pool: list[Path | None],
    rng: random.Random,
    latest_probability: float,
) -> Path | None:
    """Sample the latest checkpoint with configured bias, else an older one."""
    if not pool:
        raise ValueError("Opponent pool must contain at least one checkpoint.")
    if len(pool) == 1 or rng.random() < latest_probability:
        return pool[-1]
    return rng.choice(pool[:-1])


def _evaluate_pacman(
    pacman_model_path: Path,
    ghost_model_path: Path | None,
    seed: int,
    round_index: int,
    config: AlternatingConfig,
) -> dict[str, float | int]:
    """Evaluate Pac-Man against a selected ghost type."""
    model = PPO.load(str(pacman_model_path), device=TRAINING_DEVICE)
    env_factory = _pacman_env_factory(ghost_model_path, config.max_steps)
    rewards: list[float] = []
    pellets: list[int] = []
    steps: list[int] = []
    successes = 0
    for episode in range(config.evaluation_episodes):
        env = env_factory()
        try:
            observation, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                total_reward += reward
            rewards.append(total_reward)
            pellets.append(info["total_pellets"] - info["pellets_remaining"])
            steps.append(info["step"])
            successes += int(info["completion_fraction"] >= 1.0)
        finally:
            env.close()
    return {
        "seed": seed,
        "round": round_index,
        "pacman_mean_reward": float(np.mean(rewards)),
        "pacman_success_rate": successes / config.evaluation_episodes,
        "pacman_mean_pellets": float(np.mean(pellets)),
        "pacman_survival_steps": float(np.mean(steps)),
        "ghost_catch_rate": 0.0,
    }


def _evaluate_ghost(
    ghost_model_path: Path,
    pacman_model_path: Path,
    seed: int,
    config: AlternatingConfig,
) -> float:
    """Measure the learned ghost policy's Pac-Man catch rate."""
    model = PPO.load(str(ghost_model_path), device=TRAINING_DEVICE)
    env = GhostEnvironment(
        pacman_model_path=pacman_model_path,
        max_steps=config.max_steps,
        shared_policy=True,
    )
    catches = 0
    try:
        for episode in range(config.evaluation_episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
            catches += int(info["last_step_events"]["caught"])
    finally:
        env.close()
    return catches / config.evaluation_episodes


def _compare_final_ghosts_to_baseline(config: AlternatingConfig) -> None:
    """Compare each final learned ghost against baseline Pac-Man and rule ghosts."""
    baseline_path = Path("models/best_model.zip")
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Cannot compare final ghosts without baseline model {baseline_path}."
        )

    comparison_seed = DEFAULT_SEED + 20_000
    rule_evaluation = evaluate(
        EvaluationConfig(
            model_path=baseline_path,
            n_episodes=config.evaluation_episodes,
            max_steps=config.max_steps,
            seed=comparison_seed,
            verbose=False,
            ghost_type="rule",
        )
    )
    rule_catch_rate = (
        sum(episode.caught for episode in rule_evaluation.episodes)
        / config.evaluation_episodes
    )
    print(
        "Baseline Pac-Man vs rule ghosts "
        f"({config.evaluation_episodes} episodes): "
        f"catch rate {rule_catch_rate:.1%}"
    )

    comparison_path = config.results_path.with_name(
        "baseline_ghost_catch_rates.csv"
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("seed", "ghost_type", "episodes", "catch_rate"),
        )
        writer.writeheader()
        writer.writerow({
            "seed": "baseline",
            "ghost_type": "rule",
            "episodes": config.evaluation_episodes,
            "catch_rate": rule_catch_rate,
        })
        for seed in config.seeds:
            ghost_path = (
                config.models_dir
                / f"seed_{seed}"
                / f"round_{config.n_rounds:02d}_ghost.zip"
            )
            if not ghost_path.is_file():
                raise FileNotFoundError(
                    f"Missing final learned ghost checkpoint: {ghost_path}"
                )
            learned_catch_rate = _evaluate_ghost(
                ghost_path,
                baseline_path,
                comparison_seed,
                config,
            )
            writer.writerow({
                "seed": seed,
                "ghost_type": "learned",
                "episodes": config.evaluation_episodes,
                "catch_rate": learned_catch_rate,
            })
            print(
                f"Baseline Pac-Man vs learned ghost seed {seed} "
                f"({config.evaluation_episodes} episodes): "
                f"catch rate {learned_catch_rate:.1%}"
            )
    print(f"Baseline ghost comparison: {comparison_path}")


def _quick_config() -> AlternatingConfig:
    """Return a short, one-seed configuration for smoke testing."""
    return replace(
        AlternatingConfig(),
        n_rounds=1,
        pacman_steps_per_round=32,
        ghost_steps_per_round=32,
        seeds=(DEFAULT_SEED,),
        initial_pacman_model_path=None,
        evaluation_episodes=1,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        eval_freq=32,
    )


def _medium_config() -> AlternatingConfig:
    """Return the two-seed, three-round medium training budget."""
    return replace(
        AlternatingConfig(),
        n_rounds=MEDIUM_ALTERNATING_ROUNDS,
        pacman_steps_per_round=MEDIUM_ALTERNATING_STEPS_PER_ROUND,
        ghost_steps_per_round=MEDIUM_GHOST_STEPS_PER_ROUND,
        seeds=tuple(range(DEFAULT_SEED, DEFAULT_SEED + MEDIUM_SEEDS)),
        initial_pacman_model_path=Path("models/best_model.zip"),
        evaluation_episodes=30,
        models_dir=Path("models/experiments/medium/alternating"),
        logs_dir=Path("logs/experiments/medium/alternating"),
        results_path=Path("results/medium/alternating_rounds.csv"),
    )


def main() -> None:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--medium", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.quick:
        config = _quick_config()
    elif args.medium:
        config = _medium_config()
    else:
        config = AlternatingConfig()
    run_alternating(config)


if __name__ == "__main__":
    main()
