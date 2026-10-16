"""Tests for observation configuration sidecars."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ObservationConfig
from observation_metadata import (
    load_observation_config,
    observation_metadata_path,
    save_observation_config,
)


def test_save_and_load_optional_observation_configuration(tmp_path: Path) -> None:
    model_path = tmp_path / "pacman.zip"
    expected = ObservationConfig(
        include_ghost_direction=True,
        include_previous_pacman_action=True,
    )

    metadata_path = save_observation_config(model_path, expected)

    assert metadata_path == observation_metadata_path(model_path)
    assert load_observation_config(model_path) == expected


def test_missing_metadata_defaults_to_baseline(tmp_path: Path) -> None:
    assert load_observation_config(tmp_path / "unknown.zip") == ObservationConfig()


def test_invalid_metadata_raises_clear_error(tmp_path: Path) -> None:
    model_path = tmp_path / "pacman.zip"
    observation_metadata_path(model_path).write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid observation metadata"):
        load_observation_config(model_path)
