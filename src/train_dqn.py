"""Train a CPU-only Stable-Baselines3 DQN Pac-Man baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from config import DEFAULT_SEED, DQNConfig, QUICK_TIMESTEPS
from environment import PacmanEnvironment


def train_dqn(config: DQNConfig | None = None) -> DQN:
    """Train DQN with the baseline environment and save an isolated checkpoint."""
    cfg = config or DQNConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(
        lambda: PacmanEnvironment(max_steps=cfg.max_steps),
        n_envs=1,
        seed=cfg.seed,
        monitor_dir=str(cfg.logs_dir),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    eval_env = make_vec_env(
        lambda: PacmanEnvironment(max_steps=cfg.max_steps),
        n_envs=1,
        seed=cfg.seed,
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(cfg.models_dir),
            log_path=str(cfg.logs_dir),
            eval_freq=cfg.eval_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=cfg.checkpoint_freq,
            save_path=str(cfg.models_dir),
            name_prefix="pacman_dqn",
            verbose=0,
        ),
    ]
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        tensorboard_log=str(cfg.tensorboard_dir),
        device="cpu",
        seed=cfg.seed,
        verbose=0,
    )
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)
        model.save(str(cfg.models_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()
    return model


def main() -> None:
    """Run the DQN trainer in a short smoke-test or full-training mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    args = parser.parse_args()
    config = DQNConfig()
    if args.quick:
        config = replace(
            config,
            total_timesteps=QUICK_TIMESTEPS,
            learning_starts=8,
            buffer_size=128,
            batch_size=8,
            target_update_interval=16,
            eval_freq=16,
            n_eval_episodes=1,
            checkpoint_freq=32,
            max_steps=32,
            seed=DEFAULT_SEED,
        )
    print(
        f"Training DQN for {config.total_timesteps:,} CPU steps; "
        f"estimated minimum {config.total_timesteps / 90 / 60:.1f} min."
    )
    train_dqn(config)
    print(f"DQN model saved under {config.models_dir}.")


if __name__ == "__main__":
    main()
