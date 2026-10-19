"""Tests for robustness-matrix discovery and plot generation."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluate_matrix import (
    ModelEntry,
    _plot_matrix,
    _write_matrix_csv,
    discover_models,
    latest_ghost_model,
    _medium_config,
)


def test_discovery_skips_unrelated_files_and_includes_available_models(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "best_model.zip"
    baseline.touch()
    alternating = (
        tmp_path / "experiments" / "medium" / "alternating" / "seed_42"
        / "round_01_pacman.zip"
    )
    alternating.parent.mkdir(parents=True)
    alternating.touch()
    (tmp_path / "unrelated.zip").touch()

    found = discover_models(tmp_path)
    assert [entry.path for entry in found] == [baseline, alternating]


def test_latest_ghost_model_returns_none_when_missing(tmp_path: Path) -> None:
    assert latest_ghost_model(tmp_path) is None


def test_medium_discovery_limits_checkpoints_and_uses_medium_ghost(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    medium_dir = models_dir / "experiments" / "medium"
    baseline = models_dir / "best_model.zip"
    baseline.parent.mkdir(parents=True)
    baseline.touch()

    medium_pacman = (
        medium_dir / "alternating" / "seed_42" / "round_03_pacman.zip"
    )
    medium_pacman.parent.mkdir(parents=True)
    medium_pacman.touch()
    medium_ghost = medium_pacman.with_name("round_03_ghost.zip")
    medium_ghost.touch()

    round_zero = medium_pacman.with_name("round_00_pacman.zip")
    round_zero.touch()
    round_one = medium_pacman.with_name("round_01_pacman.zip")
    round_one.touch()
    callbacks_model = medium_pacman.parent / "round_04_pacman_callbacks" / "round_04_pacman.zip"
    callbacks_model.parent.mkdir()
    callbacks_model.touch()
    steps_model = medium_pacman.with_name("round_03_pacman_30000_steps.zip")
    steps_model.touch()
    wrong_round_name = medium_pacman.with_name("round_3_pacman.zip")
    wrong_round_name.touch()

    curriculum_final = (
        medium_dir / "curriculum" / "seed_42" / "final_model.zip"
    )
    curriculum_final.parent.mkdir(parents=True)
    curriculum_final.touch()
    curriculum_callback = (
        medium_dir / "curriculum" / "seed_42_callbacks" / "final_model.zip"
    )
    curriculum_callback.parent.mkdir(parents=True)
    curriculum_callback.touch()

    legacy_ghost = (
        models_dir / "experiments" / "alternating" / "seed_99" / "round_99_ghost.zip"
    )
    legacy_ghost.parent.mkdir(parents=True)
    legacy_ghost.touch()
    out_of_scope_pacman = (
        models_dir / "experiments" / "alternating" / "seed_99"
        / "round_99_pacman.zip"
    )
    out_of_scope_pacman.touch()

    entries = discover_models(models_dir, medium_dir)
    assert {entry.path for entry in entries} == {
        baseline,
        medium_pacman,
        curriculum_final,
    }
    assert round_zero not in {entry.path for entry in entries}
    assert round_one not in {entry.path for entry in entries}
    assert callbacks_model not in {entry.path for entry in entries}
    assert steps_model not in {entry.path for entry in entries}
    assert wrong_round_name not in {entry.path for entry in entries}
    assert curriculum_callback not in {entry.path for entry in entries}
    assert out_of_scope_pacman not in {entry.path for entry in entries}
    assert latest_ghost_model(models_dir, medium_dir) == medium_ghost


def test_medium_config_uses_isolated_paths_and_ten_episodes() -> None:
    config = _medium_config()
    assert config.experiment_models_dir == Path("models/experiments/medium")
    assert config.episodes == 10
    assert config.results_path == Path("results/medium/robustness_matrix.csv")
    assert config.plot_path == Path("plots/experiments/medium/robustness_matrix.png")


def test_matrix_writes_csv_and_annotated_plot(tmp_path: Path) -> None:
    rows = [
        {
            "model": "baseline",
            "ghost_type": "rule",
            "mean_reward": 5.0,
            "success_rate": 0.25,
            "mean_pellets": 12.0,
        }
    ]
    csv_path = tmp_path / "results" / "matrix.csv"
    plot_path = tmp_path / "plots" / "matrix.png"

    _write_matrix_csv(csv_path, rows)
    _plot_matrix(plot_path, rows)

    with csv_path.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["model"] == "baseline"
    assert plot_path.exists()


def test_matrix_csv_preserves_negative_reward_sign(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    _write_matrix_csv(
        path,
        [{
            "model": "curriculum-seed_43",
            "ghost_type": "rule",
            "mean_reward": -5925.0,
            "success_rate": 0.0,
            "mean_pellets": 5.0,
        }],
    )
    with path.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert float(row["mean_reward"]) == -5925.0


def test_discovery_keeps_only_latest_round_per_seed(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    experiment_dir = models_dir / "experiments" / "priority_300k"
    baseline = models_dir / "best_model.zip"
    baseline.parent.mkdir(parents=True)
    baseline.touch()
    paths = []
    for round_number in (0, 1, 4):
        path = (
            experiment_dir
            / "alternating"
            / "seed_42"
            / f"round_{round_number:02d}_pacman.zip"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    final_algorithm_model = (
        experiment_dir
        / "algorithm_comparison"
        / "ppo"
        / "seed_42"
        / "final_model.zip"
    )
    final_algorithm_model.parent.mkdir(parents=True)
    final_algorithm_model.touch()
    final_dqn_model = (
        experiment_dir
        / "algorithm_comparison"
        / "dqn"
        / "seed_42"
        / "final_model.zip"
    )
    final_dqn_model.parent.mkdir(parents=True)
    final_dqn_model.touch()
    ghost_direction_dir = (
        models_dir / "experiments" / "ghost_direction_ppo_300k"
    )
    ghost_direction_dir.mkdir(parents=True)
    ghost_direction = ghost_direction_dir / "final_model.zip"
    ghost_direction.touch()

    entries = discover_models(
        models_dir,
        experiment_dir,
        (ghost_direction_dir,),
    )
    assert {entry.path for entry in entries} == {
        baseline,
        paths[-1],
        final_algorithm_model,
        final_dqn_model,
        ghost_direction,
    }
    assert next(entry for entry in entries if entry.path == final_dqn_model).algorithm == "dqn"
    assert next(
        entry for entry in entries if entry.path == final_algorithm_model
    ).algorithm == "ppo"
    assert len({entry.name for entry in entries}) == len(entries)
