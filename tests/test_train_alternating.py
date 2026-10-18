"""Fast unit tests for alternating training configuration and helpers."""

from __future__ import annotations

import hashlib
import csv
import random
from io import StringIO
from pathlib import Path

import pytest

import train_alternating
from config import AlternatingConfig
from train_alternating import (
    RESULT_COLUMNS,
    _medium_config,
    _prepare_round_zero,
    _quick_config,
    _run_seed,
    _sample_opponent,
)


def test_alternating_config_has_three_reproducible_seeds() -> None:
    assert AlternatingConfig().seeds == (42, 43, 44)


def test_alternating_config_validates_opponent_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        AlternatingConfig(opponent_pool_latest_probability=1.5)


def test_quick_config_is_one_seed_and_one_short_round() -> None:
    config = _quick_config()
    assert config.seeds == (42,)
    assert config.n_rounds == 1
    assert config.pacman_steps_per_round < 100


def test_medium_config_has_two_seeds_and_isolated_outputs() -> None:
    config = _medium_config()
    assert config.seeds == (42, 43)
    assert config.n_rounds == 3
    assert config.pacman_steps_per_round == 30_000
    assert config.ghost_steps_per_round == 50_000
    assert config.evaluation_episodes == 30
    assert config.initial_pacman_model_path == Path("models/best_model.zip")
    assert "medium" in str(config.models_dir)
    assert "medium" in str(config.results_path)


def test_round_zero_copies_baseline_without_changing_source(tmp_path: Path) -> None:
    baseline = tmp_path / "models" / "best_model.zip"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"baseline-checkpoint")
    original_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    output_dir = tmp_path / "medium" / "seed_42"

    output_path = _prepare_round_zero(
        AlternatingConfig(initial_pacman_model_path=baseline),
        seed=42,
        models_dir=output_dir,
        logs_dir=tmp_path / "logs",
    )

    assert output_path.read_bytes() == baseline.read_bytes()
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == original_hash
    assert output_path.with_name(
        f"{output_path.stem}.observation.json"
    ).is_file()


def test_alternating_rows_keep_the_training_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AlternatingConfig(
        n_rounds=1,
        seeds=(42,),
        initial_pacman_model_path=None,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        results_path=tmp_path / "results.csv",
        evaluation_episodes=1,
    )
    monkeypatch.setattr(
        train_alternating,
        "_prepare_round_zero",
        lambda _config, _seed, _models, _logs: tmp_path / "round_00_pacman.zip",
    )
    monkeypatch.setattr(train_alternating, "_train_ghost", lambda *args: None)
    monkeypatch.setattr(train_alternating, "_train_pacman", lambda *args: None)

    def evaluate_pacman(*args, **kwargs):
        return {
            "seed": kwargs.get("seed", args[2] if len(args) > 2 else 42),
            "round": kwargs["round_index"],
            "pacman_mean_reward": 0.0,
            "pacman_success_rate": 0.0,
            "pacman_mean_pellets": 1.0,
            "pacman_survival_steps": 1.0,
            "ghost_catch_rate": 0.0,
        }

    monkeypatch.setattr(train_alternating, "_evaluate_pacman", evaluate_pacman)
    monkeypatch.setattr(train_alternating, "_evaluate_ghost", lambda *args: 0.5)
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)

    records = _run_seed(42, config, writer, output)

    assert [record["seed"] for record in records] == [42, 42]


def test_opponent_pool_sampler_can_select_latest_or_older() -> None:
    pool = ["old", "latest"]
    assert _sample_opponent(pool, random.Random(2), 1.0) == "latest"
    assert _sample_opponent(pool, random.Random(2), 0.0) == "old"
