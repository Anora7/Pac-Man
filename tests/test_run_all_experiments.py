"""Tests for the medium experiment runner selection and estimates."""

from __future__ import annotations

import sys

import run_all_experiments


def test_medium_runner_selects_the_four_training_experiments(
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run_all_experiments,
        "_run_one",
        lambda name, mode: calls.append((name, mode)) or [],
    )
    monkeypatch.setattr(sys, "argv", ["run_all_experiments.py", "--medium"])

    run_all_experiments.main()

    assert calls == [
        ("alternating", "medium"),
        ("curriculum", "medium"),
        ("dqn", "medium"),
        ("ablation", "medium"),
    ]
    output = capsys.readouterr().out
    assert "Estimated CPU training time" in output
    assert output.index("Estimated CPU training time") < output.index(
        "=== Running alternating ==="
    )
    estimates = {
        name: run_all_experiments._estimated_training_steps(name, "medium")
        for name in ("alternating", "curriculum", "dqn", "ablation")
    }
    assert estimates == {
        "alternating": 480_000,
        "curriculum": 400_000,
        "dqn": 400_000,
        "ablation": 800_000,
    }


def test_medium_runner_only_dispatches_selected_experiment(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run_all_experiments,
        "_run_one",
        lambda name, mode: calls.append((name, mode)) or [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_experiments.py", "--medium", "--only", "dqn"],
    )

    run_all_experiments.main()

    assert calls == [("dqn", "medium")]


def test_baseline_fallback_saves_outside_root_models_directory(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(run_all_experiments, "ROOT", tmp_path)

    def save_fallback(config) -> None:
        config.models_dir.mkdir(parents=True, exist_ok=True)
        (config.models_dir / "best_model.zip").touch()

    monkeypatch.setattr(run_all_experiments, "train", save_fallback)

    result = run_all_experiments._ensure_baseline_model(
        quick=False,
        medium=True,
    )

    assert result == (
        tmp_path
        / "models"
        / "experiments"
        / "medium"
        / "baseline"
        / "best_model.zip"
    )
    assert not (tmp_path / "models" / "best_model.zip").exists()
