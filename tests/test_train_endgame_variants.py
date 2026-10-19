"""Tests for the isolated endgame training experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import REWARDS, ObservationConfig
from evaluate import EpisodeResult, EvaluationResult
from maze import Maze, Position
import train_endgame_variants
from train_endgame_variants import (
    EPISODES,
    VARIANTS,
    _evaluation_row,
    _evaluation_seed,
    _has_tight_loop,
    _run_output_tag,
    _variant_settings,
    _write_results,
    run_variant,
)


_PELLET_PATH_LAYOUT = [
    [0, 0, 0, 0, 0, 0],
    [0, 4, 1, 1, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


def _episode(
    positions: tuple[tuple[int, int], ...],
    *,
    cleared: bool = False,
) -> EpisodeResult:
    return EpisodeResult(
        episode=1,
        total_reward=1.0,
        steps=len(positions) - 1,
        pellets_collected=3,
        total_pellets=5,
        cleared=cleared,
        caught=False,
        truncated=False,
        completion_fraction=0.6,
        positions=positions,
    )


def test_maze_bfs_returns_first_action_and_shortest_distance() -> None:
    maze = Maze(_PELLET_PATH_LAYOUT)

    first_action, distance = maze.nearest_pellet_path(Position(1, 1))

    assert first_action == 3
    assert distance == 1


def test_maze_bfs_returns_none_when_no_pellets_remain() -> None:
    maze = Maze(_PELLET_PATH_LAYOUT)
    for position in (
        Position(1, 2),
        Position(1, 3),
        Position(1, 4),
    ):
        maze.consume_pellet(position)

    assert maze.nearest_pellet_path(Position(1, 1)) == (None, None)


def test_variant_settings_enable_only_the_requested_feature_and_shaping() -> None:
    features, features_rewards = _variant_settings("features_only")
    assert features == ObservationConfig(
        include_nearest_pellet=True,
        include_previous_action=True,
    )
    assert features_rewards == REWARDS

    combined, combined_rewards = _variant_settings("features_shaping")
    assert combined.include_nearest_pellet
    assert combined.include_previous_action
    assert combined_rewards.PELLET_DISTANCE_DECREASE == 0.5
    assert combined_rewards.PELLET_DISTANCE_INCREASE == -0.5
    assert combined_rewards.REVISIT_CELL == -0.1

    ghost_variant, ghost_rewards = _variant_settings("features_shaping_ghost")
    assert ghost_variant.include_nearest_pellet
    assert ghost_variant.include_previous_action
    assert ghost_variant.include_nearest_ghost
    assert ghost_rewards.PELLET_DISTANCE_DECREASE == 0.5
    assert ghost_rewards.PELLET_DISTANCE_INCREASE == -0.5
    assert ghost_rewards.REVISIT_CELL == -0.1
    assert ghost_rewards.GHOST_DISTANCE_DECREASE == -0.5

    shaping, shaping_rewards = _variant_settings("shaping_only")
    assert shaping == ObservationConfig()
    assert shaping_rewards == combined_rewards


def test_variant_settings_reject_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported variant"):
        _variant_settings("unknown")


def test_evaluation_seed_ranges_are_distinct_for_every_run_group() -> None:
    ranges = [
        set(range(
            _evaluation_seed(42, variant, model, policy),
            _evaluation_seed(42, variant, model, policy) + EPISODES,
        ))
        for variant in VARIANTS
        for model in ("baseline", "trained")
        for policy in ("deterministic", "stochastic")
    ]
    flattened = [seed for seed_range in ranges for seed in seed_range]

    assert len(flattened) == len(set(flattened))


def test_output_tag_separates_seed_step_count_and_custom_label() -> None:
    assert _run_output_tag(42, 300_000, None) == "seed_42_steps_300000"
    assert _run_output_tag(43, 300_000, None) == "seed_43_steps_300000"
    assert _run_output_tag(42, 200_000, None) == "seed_42_steps_200000"
    assert (
        _run_output_tag(42, 300_000, "demo")
        == "demo_seed_42_steps_300000"
    )


@pytest.mark.parametrize("output_tag", ["", "../outside", "has space", "a/b"])
def test_output_tag_rejects_invalid_directory_components(
    output_tag: str,
) -> None:
    with pytest.raises(ValueError, match="output_tag"):
        _run_output_tag(42, 300_000, output_tag)


def test_loop_rate_flags_three_cell_oscillation_only() -> None:
    loop = tuple((1, 1 + index % 3) for index in range(51))
    broader_walk = tuple((1, 1 + index % 4) for index in range(51))

    assert _has_tight_loop(loop)
    assert not _has_tight_loop(broader_walk)
    assert not _has_tight_loop(((1, 1),) * 50)


def test_evaluation_row_reports_clears_and_loop_rate() -> None:
    result = EvaluationResult(
        n_episodes=2,
        episodes=[
            _episode(tuple((1, 1 + index % 3) for index in range(51))),
            _episode(((1, 1), (1, 2)), cleared=True),
        ],
        mean_reward=2.0,
        std_reward=1.0,
        mean_steps=26.0,
        mean_pellets_collected=3.0,
        success_rate=0.5,
        avg_completion_fraction=0.8,
    )

    row = _evaluation_row(
        "features_only",
        "trained",
        "deterministic",
        result,
    )

    assert row["mean_pellets"] == 3.0
    assert row["mean_steps"] == 26.0
    assert row["clears"] == 1
    assert row["loop_rate"] == 0.5


def test_result_writer_uses_stable_columns(tmp_path: Path) -> None:
    path = tmp_path / "endgame.csv"
    _write_results(
        path,
        [{
            "variant": "features_only",
            "model": "baseline",
            "policy": "deterministic",
            "episodes": 30,
            "mean_pellets": 42.0,
            "mean_steps": 100.0,
            "clears": 1,
            "loop_rate": 0.2,
        }],
    )

    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["model"] == "baseline"
    assert rows[0]["clears"] == "1"


def test_variant_run_uses_isolated_paths_and_writes_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models"
    log_root = tmp_path / "logs"
    results_root = tmp_path / "results"
    monkeypatch.setattr(train_endgame_variants, "MODEL_ROOT", model_root)
    monkeypatch.setattr(train_endgame_variants, "LOG_ROOT", log_root)
    monkeypatch.setattr(train_endgame_variants, "VARIANT_RESULTS_DIR", results_root)

    captured_training = []
    captured_evaluations = []

    def fake_train(config):
        captured_training.append(config)
        return SimpleNamespace()

    def fake_evaluate(config, model=None):
        captured_evaluations.append((config, model))
        episodes = [
            _episode(tuple((1, 1 + index % 3) for index in range(51))),
            _episode(((1, 1), (1, 2)), cleared=True),
        ]
        return EvaluationResult(
            n_episodes=2,
            episodes=episodes,
            mean_reward=0.0,
            std_reward=0.0,
            mean_steps=26.0,
            mean_pellets_collected=3.0,
            success_rate=0.5,
            avg_completion_fraction=0.8,
        )

    monkeypatch.setattr(train_endgame_variants, "train", fake_train)
    monkeypatch.setattr(train_endgame_variants, "evaluate", fake_evaluate)

    rows = run_variant(
        "features_shaping",
        total_timesteps=32,
        seed=42,
        episodes=2,
        output_tag="demo",
    )

    assert len(captured_training) == 1
    run_tag = "demo_seed_42_steps_32"
    assert (
        captured_training[0].models_dir
        == model_root / run_tag / "features_shaping"
    )
    assert captured_training[0].logs_dir == log_root / run_tag / "features_shaping"
    assert captured_training[0].total_timesteps == 32
    assert captured_training[0].observation_config.include_nearest_pellet
    assert len(captured_evaluations) == 4
    assert all(config.record_trajectory for config, _ in captured_evaluations)
    assert len({config.seed for config, _ in captured_evaluations}) == 4
    assert captured_evaluations[0][0].model_path == train_endgame_variants.BASELINE_MODEL
    assert captured_evaluations[1][1] is not None
    assert len(rows) == 4
    assert (results_root / run_tag / "features_shaping.csv").is_file()


def test_ghost_variant_can_run_with_full_cli_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_endgame_variants, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(train_endgame_variants, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(
        train_endgame_variants,
        "VARIANT_RESULTS_DIR",
        tmp_path / "results",
    )
    training_configs = []
    evaluation_configs = []

    monkeypatch.setattr(
        train_endgame_variants,
        "train",
        lambda config: training_configs.append(config) or SimpleNamespace(),
    )

    def fake_evaluate(config, model=None):
        evaluation_configs.append(config)
        episodes = [_episode(((1, 1), (1, 2)))]
        return EvaluationResult(
            n_episodes=1,
            episodes=episodes,
            mean_reward=0.0,
            std_reward=0.0,
            mean_steps=1.0,
            mean_pellets_collected=1.0,
            success_rate=0.0,
            avg_completion_fraction=0.2,
        )

    monkeypatch.setattr(train_endgame_variants, "evaluate", fake_evaluate)

    rows = train_endgame_variants.run_experiment(
        variant="features_shaping_ghost",
        total_timesteps=64,
        seed=7,
        output_tag="ghosttest",
    )

    assert len(rows) == 4
    assert len(training_configs) == 1
    training_config = training_configs[0]
    assert training_config.seed == 7
    assert training_config.total_timesteps == 64
    assert training_config.observation_config.include_nearest_ghost
    assert training_config.reward_config.GHOST_DISTANCE_DECREASE == -0.5
    assert len(evaluation_configs) == 4
    assert all(
        config.observation_config is None
        or config.observation_config.include_nearest_ghost
        for config in evaluation_configs
    )
    assert (
        tmp_path
        / "results"
        / "ghosttest_seed_7_steps_64"
        / "features_shaping_ghost.csv"
    ).is_file()
