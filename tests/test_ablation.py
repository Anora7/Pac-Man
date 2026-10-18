"""Tests for explicit reward variants used by the ablation study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import ablation
import pytest
from ablation import MEDIUM_VARIANTS, _medium_config, _variants_for_config
from config import MEDIUM_TIMESTEPS, REWARDS, REWARD_ABLATION_VARIANTS
from environment import PacmanEnvironment


def test_medium_ablation_uses_baseline_and_three_variants() -> None:
    config = _medium_config()
    variants = _variants_for_config(config)
    assert config.seeds == (42, 43)
    assert config.total_timesteps == MEDIUM_TIMESTEPS
    assert tuple(name for name, _ in variants) == MEDIUM_VARIANTS
    assert len(variants) == 4
    assert config.evaluation_episodes == 30
    assert "medium" in str(config.models_dir)
    assert "medium" in str(config.results_path)


def test_eval_only_loads_saved_variant_and_does_not_train(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _medium_config()
    config = replace(
        config,
        evaluation_episodes=30,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    model_path = config.models_dir / "baseline" / "seed_42" / "final_model.zip"
    model_path.parent.mkdir(parents=True)
    model_path.touch()

    class FakePPO:
        @staticmethod
        def load(path: str, device: str):
            return path, device

    monkeypatch.setattr(ablation, "PPO", FakePPO)
    monkeypatch.setattr(
        ablation,
        "train",
        lambda *_: pytest.fail("eval-only must not train PPO"),
    )
    evaluations = []

    def fake_evaluate(evaluation_config, model):
        evaluations.append((evaluation_config, model))
        return SimpleNamespace(
            mean_reward=1.0,
            success_rate=0.0,
            mean_pellets_collected=2.0,
            mean_steps=3.0,
        )

    monkeypatch.setattr(ablation, "evaluate", fake_evaluate)
    row = ablation._run_variant_seed(
        "baseline",
        REWARDS,
        42,
        config,
        eval_only=True,
    )

    assert row["variant"] == "baseline"
    assert len(evaluations) == 1
    assert evaluations[0][0].n_episodes == 30
    assert evaluations[0][0].seed == 10_042


def test_reward_variant_table_covers_required_terms() -> None:
    variants = dict(REWARD_ABLATION_VARIANTS)
    assert variants["baseline"] == REWARDS
    assert variants["caught_penalty_50"].CAUGHT_BY_GHOST == -50.0
    assert variants["caught_penalty_500"].CAUGHT_BY_GHOST == -500.0
    assert variants["step_penalty_0"].STEP == 0.0
    assert variants["step_penalty_2"].STEP == -2.0
    assert variants["no_wall_penalty"].HIT_WALL == 0.0
    assert variants["pellet_reward_5"].PELLET == 5.0
    assert variants["pellet_reward_20"].PELLET == 20.0
    assert variants["clear_bonus_0"].CLEAR_MAZE == 0.0
    assert variants["clear_bonus_1000"].CLEAR_MAZE == 1000.0


def test_environment_uses_custom_reward_config() -> None:
    reward_config = replace(REWARDS, STEP=0.0)
    env = PacmanEnvironment(reward_config=reward_config)
    try:
        env.reset(seed=42)
        _, reward, _, _, info = env.step(3)
        assert reward == 10.0
        assert info["last_step_events"]["pellet_eaten"] is True
    finally:
        env.close()
