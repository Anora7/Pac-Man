"""
PPO training pipeline for the Pac-Man RL project.

This module owns:
  - TrainingConfig: all hyperparameters and file paths in one place.
  - train(): main training entry point; returns the trained PPO model.

This module must not contain game logic, rendering, or environment state.

Usage::

    from train import TrainingConfig, train

    config = TrainingConfig(total_timesteps=500_000, verbose=1)
    model = train(config)

Or from the command line::

    python src/train.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    GAMMA,
    LEARNING_RATE,
    MAX_STEPS,
    N_EPOCHS,
    N_STEPS,
    TOTAL_TIMESTEPS,
    DEFAULT_SEED,
    GHOST_COUNT,
    OBSERVATION_CONFIG,
    REWARDS,
    ObservationConfig,
    RewardConfig,
    TRAINING_DEVICE,
)
from environment import PacmanEnvironment
from observation_metadata import save_observation_config


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All training hyperparameters and output paths in one dataclass.

    Defaults are read from :mod:`config` so that a single change there
    propagates to both the environment and the training pipeline.

    Attributes:
        models_dir:       Directory for saved models (best, checkpoints, final).
        logs_dir:         Directory for Monitor CSV logs and eval results.
        tensorboard_dir:  Directory for TensorBoard event files.
        total_timesteps:  Total environment steps for training.
        learning_rate:    PPO learning rate.
        gamma:            Discount factor.
        n_steps:          Steps collected per environment per update.
        batch_size:       Mini-batch size for each PPO epoch.
        n_epochs:         Number of optimisation epochs per update.
        ent_coef:         Entropy coefficient for exploration.
        clip_range:       PPO clipping range.
        device:           PyTorch device. This project uses CPU for reproducible
                          experiments and compatibility with its target setup.
        seed:             Integer seed for reproducibility.
        max_steps:        Maximum steps per episode (passed to the env).
        eval_freq:        Evaluation frequency in environment steps.
        n_eval_episodes:  Number of episodes per evaluation run.
        checkpoint_freq:  Checkpoint save frequency in environment steps.
        verbose:          SB3 verbosity level (0 = silent, 1 = info).
        show_progress_bar: Enable ``rich``/``tqdm`` progress bar.
                           Requires ``rich`` and ``tqdm`` to be installed.
    """

    # ---- Output paths ----
    models_dir: Path = field(default_factory=lambda: Path("models"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    tensorboard_dir: Path = field(default_factory=lambda: Path("logs/tensorboard"))

    # ---- PPO hyperparameters ----
    total_timesteps: int = TOTAL_TIMESTEPS
    learning_rate: float = LEARNING_RATE
    gamma: float = GAMMA
    n_steps: int = N_STEPS
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    ent_coef: float = ENT_COEF
    clip_range: float = CLIP_RANGE

    # ---- Device ----
    device: str = "auto"

    # ---- Environment ----
    seed: int | None = None
    max_steps: int = MAX_STEPS
    reward_config: RewardConfig = REWARDS
    ghost_count: int = GHOST_COUNT
    observation_config: ObservationConfig = OBSERVATION_CONFIG

    # ---- Callbacks ----
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 10_000

    # ---- Logging ----
    verbose: int = 1
    show_progress_bar: bool = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_env(
    max_steps: int = MAX_STEPS,
    reward_config: RewardConfig = REWARDS,
    ghost_count: int = GHOST_COUNT,
    observation_config: ObservationConfig = OBSERVATION_CONFIG,
) -> PacmanEnvironment:
    """Create a single unwrapped :class:`PacmanEnvironment` instance.

    The :class:`~stable_baselines3.common.monitor.Monitor` wrapper is applied
    by :func:`make_vec_env` when ``monitor_dir`` is provided.
    """
    return PacmanEnvironment(
        max_steps=max_steps,
        reward_config=reward_config,
        ghost_count=ghost_count,
        observation_config=observation_config,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(config: TrainingConfig | None = None) -> PPO:
    """Run the full PPO training loop and return the trained model.

    Creates output directories, builds vectorised training and evaluation
    environments, configures callbacks, trains the PPO agent, saves the
    final model, and closes all environments.

    Args:
        config: :class:`TrainingConfig` instance.  ``None`` uses defaults.

    Returns:
        The trained :class:`stable_baselines3.PPO` model.

    Example::

        model = train(TrainingConfig(total_timesteps=100_000, verbose=0))
        obs = env.reset()[0]
        action, _ = model.predict(obs, deterministic=True)
    """
    if config is None:
        config = TrainingConfig()

    # ---- Directories ----
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # ---- Environments ----
    # Keep the public config defaults backward-compatible while ensuring all
    # project experiments use a fixed seed and the required CPU backend.
    training_seed = config.seed if config.seed is not None else DEFAULT_SEED

    vec_env = make_vec_env(
        _make_env,
        n_envs=1,
        seed=training_seed,
        monitor_dir=str(config.logs_dir),
        monitor_kwargs={"info_keywords": ("is_success",)},
        env_kwargs={
            "max_steps": config.max_steps,
            "reward_config": config.reward_config,
            "ghost_count": config.ghost_count,
            "observation_config": config.observation_config,
        },
    )

    eval_env = make_vec_env(
        _make_env,
        n_envs=1,
        seed=training_seed,
        monitor_kwargs={"info_keywords": ("is_success",)},
        env_kwargs={
            "max_steps": config.max_steps,
            "reward_config": config.reward_config,
            "ghost_count": config.ghost_count,
            "observation_config": config.observation_config,
        },
    )

    # ---- Model ----
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        device=TRAINING_DEVICE,
        tensorboard_log=str(config.tensorboard_dir),
        verbose=config.verbose,
        seed=training_seed,
    )

    # ---- Callbacks ----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.models_dir),
        log_path=str(config.logs_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=config.verbose,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(config.models_dir),
        name_prefix="pacman_ppo",
        verbose=0,  # checkpoint saves are noisy at verbose=1
    )

    # ---- Train ----
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=config.show_progress_bar,
    )

    # ---- Save final model ----
    final_path = config.models_dir / "final_model"
    model.save(str(final_path))
    save_observation_config(
        final_path.with_suffix(".zip"),
        config.observation_config,
    )
    best_model_path = config.models_dir / "best_model.zip"
    if best_model_path.exists():
        save_observation_config(best_model_path, config.observation_config)
    if config.verbose:
        print(f"\nFinal model saved: {final_path}.zip")

    vec_env.close()
    eval_env.close()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config = TrainingConfig()
    _model = train(_config)
